"""Offline cross-encoder contracts, with synthetic tokenizers and sessions."""
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors


@pytest.fixture
def adapter():
    return import_module('app.adapters.text_reranking').LocalTextReranker


def write_model(path, *, revision='a' * 40, config=None):
    path.mkdir(exist_ok=True)
    (path/'onnx').mkdir(exist_ok=True)
    (path/'onnx/model_quint8_avx2.onnx').write_bytes(b'synthetic classifier')
    (path/'tokenizer.json').write_text('synthetic tokenizer',encoding='utf-8')
    config = config or {'id2label': {'0':'LABEL_0'}, 'max_position_embeddings':514, 'pad_token_id':0}
    (path/'config.json').write_text(json.dumps(config),encoding='utf-8')
    files={name:{'sha256':hashlib.sha256((path/name).read_bytes()).hexdigest()} for name in
           ('config.json','tokenizer.json','onnx/model_quint8_avx2.onnx')}
    (path/'manifest.json').write_text(json.dumps({'repo_id':'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1',
        'revision':revision,'model_file':'onnx/model_quint8_avx2.onnx','files':files}),encoding='utf-8')


def tokenizer():
    tokens=['[UNK]','[PAD]','[CLS]','[SEP]','q','front','late','other','minus','zero','plus']
    tk=Tokenizer(models.WordLevel({word:i for i,word in enumerate(tokens)},unk_token='[UNK]'))
    tk.pre_tokenizer=pre_tokenizers.Whitespace()
    tk.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',
        pair='[CLS] $A [SEP] $B:1 [SEP]:1',special_tokens=[('[CLS]',2),('[SEP]',3)])
    return tk


class Session:
    def __init__(self, output=None, dtype='tensor(int64)', names=('input_ids','attention_mask','token_type_ids')):
        self.calls=[]; self.output=output; self.dtype=dtype; self.names=names
        self.active=0; self.maximum_active=0
    def get_inputs(self):
        return [SimpleNamespace(name=name,type=self.dtype) for name in self.names]
    def run(self,names,feed):
        self.active+=1;self.maximum_active=max(self.maximum_active,self.active)
        try:
            time.sleep(.002);self.calls.append(feed)
            if self.output is not None:return [self.output(feed)]
            return [np.array([[6. if 6 in ids else 0.] for ids in feed['input_ids']])]
        finally:self.active-=1


@pytest.fixture
def parts(tmp_path,adapter):
    model=tmp_path/'model';write_model(model);tk=tokenizer();session=Session();created=[]
    def factory(path,**kwargs):created.append((path,kwargs));return session
    def make(**kwargs):
        return adapter(model,tokenizer_factory=lambda path:tk,session_factory=factory,**kwargs)
    return model,tk,session,created,make


def test_standard_pairs_cpu_and_bounded_scores(parts):
    model,_,session,created,make=parts
    score=make()('q',['front'])[0]
    assert score==pytest.approx(.5)
    assert created==[(str(model/'onnx/model_quint8_avx2.onnx'),{'providers':['CPUExecutionProvider']})]
    feed=session.calls[0]
    assert feed['input_ids'][0].tolist()==[2,4,3,5,3]
    assert feed['token_type_ids'][0].tolist()==[0,0,0,1,1]
    assert all(v.dtype==np.int64 for v in feed.values())


def test_document_tail_is_scored_and_max_window_returns_one_score_per_passage(parts):
    _,_,session,_,make=parts
    scorer=make(batch_size=2,max_length=12,stride=2,query_max_tokens=2)
    scores=scorer('q',['front '*20+'late','other','front '*20+'late'])
    assert scores[0]==scores[2]==pytest.approx(1/(1+math.exp(-6)))
    assert scores[1]==pytest.approx(.5)
    assert scorer.last_run['unique_passages']==2
    assert scorer.last_run['passage_window_counts'][0]>1
    assert scorer.last_run['computed_windows']>2
    assert all(call['input_ids'].shape[1]<=12 and len(call['input_ids'])<=2 for call in session.calls)
    assert scorer.last_run['winning_window_indices'][0]>0


def test_query_limit_is_token_based_and_query_overflow_is_never_scored_as_document(parts):
    _,_,session,_,make=parts
    scorer=make(query_max_tokens=2,max_length=12,stride=2)
    score=scorer('q q late late',['front'])[0]
    assert score==pytest.approx(.5)
    assert scorer.last_run['query_truncated'] is True
    assert scorer.last_run['query_tokens_before']==4 and scorer.last_run['query_tokens_used']<=2
    assert all(6 not in call['input_ids'] for call in session.calls)


def test_short_query_status_and_padding_masks(parts):
    _,_,session,_,make=parts
    scorer=make(batch_size=2);scorer('q',['front','other other other'])
    assert scorer.last_run['query_truncated'] is False
    assert scorer.last_run['query_tokens_before']==scorer.last_run['query_tokens_used']==1
    feed=session.calls[0]
    assert feed['input_ids'][0,-1]==0 and feed['attention_mask'][0,-1]==0


def test_single_logit_shape_and_stable_sigmoid_extremes(parts):
    _,_,session,_,make=parts
    session.output=lambda feed:np.array([-1000.,0.,1000.])
    assert make(batch_size=3)('q',['minus','zero','plus'])==[0.,.5,1.]


def test_two_input_int32_model_is_supported(parts):
    _,_,session,_,make=parts
    session.names=('input_ids','attention_mask');session.dtype='tensor(int32)'
    assert make()('q',['front'])==[.5]
    assert set(session.calls[0])=={'input_ids','attention_mask'}
    assert all(value.dtype==np.int32 for value in session.calls[0].values())


@pytest.mark.parametrize('bad_output',[
    lambda f:np.ones((len(f['input_ids']),2)),lambda f:np.ones((len(f['input_ids']),1,1)),
    lambda f:np.ones((len(f['input_ids'])+1,1)),lambda f:np.array([[float('nan')]]),
    lambda f:np.array([[float('inf')]]),lambda f:np.array([[True]]),lambda f:np.array([['text']]),
])
def test_invalid_output_is_not_a_successful_score(parts,bad_output):
    _,_,session,_,make=parts;session.output=bad_output;scorer=make()
    with pytest.raises(RuntimeError,match='重排'):
        scorer('q',['front'])
    assert scorer.last_run['status']=='error'


def test_private_backend_exception_is_sanitized(parts):
    _,_,session,_,make=parts
    def fail(feed):raise RuntimeError('PRIVATE_QUERY F:/private-materials/secret.txt')
    session.output=fail;scorer=make()
    with pytest.raises(RuntimeError) as caught:scorer('PRIVATE_QUERY',['PRIVATE_PASSAGE'])
    assert 'PRIVATE_' not in str(caught.value) and 'F:' not in str(caught.value)
    assert scorer.last_run['status']=='error'
    assert 'PRIVATE_' not in json.dumps(scorer.last_run)


@pytest.mark.parametrize('query,passages',[(None,['front']),(' ',['front']),('q',None),('q','front'),
                                        ('q',['']),('q',['  ']),('q',[False])])
def test_invalid_inputs_raise_clear_errors(parts,query,passages):
    *_,make=parts
    with pytest.raises(ValueError):make()(query,passages)


def test_empty_passages_do_not_tokenize_or_infer(parts):
    _,_,session,_,make=parts;scorer=make()
    assert scorer('q',[])==[] and session.calls==[]
    assert scorer.last_run['documents']==0 and scorer.last_run['computed_windows']==0


def test_sha_failure_precedes_tokenizer_and_session_creation(parts):
    model,_,_,created,make=parts;(model/'onnx/model_quint8_avx2.onnx').write_bytes(b'altered')
    with pytest.raises(ValueError,match='SHA'):make()
    assert created==[]


@pytest.mark.parametrize('filename',['../outside.onnx','/outside.onnx'])
def test_manifest_model_path_cannot_escape_model_directory(parts,filename):
    model,_,_,created,make=parts;p=model/'manifest.json';manifest=json.loads(p.read_text());manifest['model_file']=filename;p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):make()
    assert created==[]


def test_config_single_label_mapping_without_num_labels_is_valid_and_multi_label_is_rejected(parts):
    model,_,_,_,make=parts;assert make()('q',['front'])==[.5]
    write_model(model,config={'num_labels':2,'max_position_embeddings':514,'pad_token_id':0})
    with pytest.raises(ValueError):make()


@pytest.mark.parametrize('kwargs',[{'batch_size':True},{'batch_size':0},{'max_length':0},
    {'stride':-1},{'query_max_tokens':0},{'max_length':16,'stride':8,'query_max_tokens':8}])
def test_invalid_window_settings_are_rejected(parts,kwargs):
    *_,make=parts
    with pytest.raises(ValueError):make(**kwargs)


def test_model_key_binds_files_revision_and_window_strategy(parts):
    model,_,_,_,make=parts;first=make();write_model(model,revision='b'*40);second=make()
    assert first.model_key!=second.model_key
    assert second.model_key!=make(max_length=256,stride=64).model_key
    assert 'max_window' in str(first.window_strategy)


def test_threaded_calls_share_no_tokenizer_or_session_overlap(parts):
    _,_,session,_,make=parts;scorer=make()
    with ThreadPoolExecutor(2) as pool:results=list(pool.map(lambda _:scorer('q',['front']),range(4)))
    assert results==[[.5]]*4 and session.maximum_active==1


@pytest.mark.parametrize('name',['manifest.json','config.json'])
def test_malformed_metadata_has_a_safe_validation_error(parts,name):
    model,_,_,_,make=parts
    path=model/name;path.write_text('[]',encoding='utf-8')
    if name=='config.json':
        manifest_path=model/'manifest.json';manifest=json.loads(manifest_path.read_text())
        manifest['files'][name]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):make()


def test_failed_first_batch_does_not_report_uncomputed_windows_as_completed(parts):
    _,_,session,_,make=parts
    session.output=lambda feed:np.ones((len(feed['input_ids']),2))
    scorer=make()
    with pytest.raises(RuntimeError):scorer('q',['front'])
    assert scorer.last_run['planned_windows']==1
    assert scorer.last_run['computed_windows']==0


def test_default_scores_each_window_in_its_own_inference_batch(parts):
    _,_,session,_,make=parts
    scorer=make()
    assert scorer('q',['front','other other','late'])==pytest.approx([.5,.5,1/(1+math.exp(-6))])
    assert scorer.batch_size==1
    assert len(session.calls)==3
    assert all(len(call['input_ids'])==1 for call in session.calls)
    assert scorer.last_run['computed_batches']==3


def test_model_identity_and_observability_bind_actual_batch_and_padding(parts):
    *_,make=parts
    single=make(batch_size=1);batched=make(batch_size=2)
    assert single.model_key!=batched.model_key
    single('q',['front'])
    assert single.inference_strategy=={'batch_size':1,'padding':'longest_in_batch','pad_token_id':0,'version':1}
    assert single.last_run['inference_strategy']==single.inference_strategy
    assert batched.inference_strategy['batch_size']==2


def test_runtime_batch_size_cannot_silently_change_a_frozen_identity(parts):
    *_,make=parts
    scorer=make()
    with pytest.raises(AttributeError):scorer.batch_size=2
