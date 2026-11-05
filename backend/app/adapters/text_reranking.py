"""Offline single-logit ONNX pair reranking; scores express relevance, not facts."""

import hashlib
import json
from pathlib import Path
import re
from threading import Lock

__all__ = ['LocalTextReranker']


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class LocalTextReranker:
    """Encode standard query/document pairs and keep each passage's best window.

    Loads only verified local files. Factories enable offline tests. Query prefixes
    are limited by tokenizer offsets, then re-encoded as ordinary standard pairs;
    query overflow encodings never enter the document-window scoring path.
    """

    def __init__(self, model_dir, *, batch_size=1, max_length=512, stride=128,
                 query_max_tokens=128, tokenizer_factory=None, session_factory=None):
        if (type(batch_size) is not int or not 1 <= batch_size <= 128
                or type(max_length) is not int or max_length < 4
                or type(stride) is not int or stride < 0
                or type(query_max_tokens) is not int or query_max_tokens < 1):
            raise ValueError('相关性重排批次或窗口参数无效')
        self.model_dir = Path(model_dir).resolve()
        try:
            manifest = json.loads((self.model_dir / 'manifest.json').read_text(encoding='utf-8'))
            model_file = manifest['model_file']
            if (not isinstance(manifest.get('repo_id'), str) or not manifest['repo_id'].strip()
                    or not re.fullmatch(r'[a-f0-9]{40}', manifest['revision'])
                    or not isinstance(model_file, str) or Path(model_file).is_absolute()
                    or '..' in Path(model_file).parts or Path(model_file).suffix != '.onnx'):
                raise ValueError('manifest')
            runtime_files = {}
            for name in ('config.json', 'tokenizer.json', model_file):
                path = (self.model_dir / name).resolve()
                if not path.is_relative_to(self.model_dir):
                    raise ValueError('runtime path')
                expected = manifest['files'][name]['sha256']
                if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
                    raise ValueError('digest')
                runtime_files[name] = _sha256(path)
                if runtime_files[name] != expected:
                    raise ValueError('SHA')
            config = json.loads((self.model_dir / 'config.json').read_text(encoding='utf-8'))
            if not isinstance(config, dict):
                raise ValueError('config')
        except (OSError, KeyError, TypeError, ValueError):
            raise ValueError('本机重排模型清单或SHA核验失败') from None
        labels = config.get('num_labels')
        if labels is None and isinstance(config.get('id2label'), dict):
            labels = len(config['id2label'])
        positions, pad = config.get('max_position_embeddings'), config.get('pad_token_id', 0)
        if (type(labels) is not int or labels != 1 or type(positions) is not int
                or max_length > positions or type(pad) is not int or pad < 0):
            raise ValueError('本机重排模型须为单标签且窗口配置有效')
        self._batch_size, self.max_length = batch_size, max_length
        self.stride, self.query_max_tokens, self.pad_id = stride, query_max_tokens, pad
        try:
            import numpy as np
            if tokenizer_factory is None:
                from tokenizers import Tokenizer
                tokenizer_factory = Tokenizer.from_file
            if session_factory is None:
                import onnxruntime
                onnxruntime.disable_telemetry_events()
                options = onnxruntime.SessionOptions()
                options.intra_op_num_threads = 4
                options.inter_op_num_threads = 1
                def session_factory(path, **kwargs):
                    return onnxruntime.InferenceSession(path, sess_options=options, **kwargs)
        except ImportError:
            raise RuntimeError('本机相关性重排依赖尚未安装') from None
        self._np = np
        try:
            self._tokenizer = tokenizer_factory(str(self.model_dir / 'tokenizer.json'))
            self._tokenizer.no_padding()
            self._special_tokens = self._tokenizer.num_special_tokens_to_add(is_pair=True)
        except Exception:
            raise RuntimeError('本机相关性重排分词器初始化失败') from None
        if (type(self._special_tokens) is not int or self._special_tokens < 2
                or query_max_tokens + self._special_tokens + stride >= max_length):
            raise ValueError('重排查询词元上限与文档窗口或重叠长度不兼容')
        try:
            self._session = session_factory(str(self.model_dir / model_file), providers=['CPUExecutionProvider'])
            self._inputs = self._session.get_inputs()
            names = {item.name for item in self._inputs}
            if (not {'input_ids', 'attention_mask'} <= names or len(names) != len(self._inputs)
                    or names - {'input_ids', 'attention_mask', 'token_type_ids'}
                    or any(item.type not in ('tensor(int64)', 'tensor(int32)') for item in self._inputs)):
                raise ValueError('inputs')
        except Exception:
            raise RuntimeError('本机相关性重排模型初始化或输入格式无效') from None
        self.window_strategy = {'max_length': max_length, 'stride': stride,
                                'query_max_tokens': query_max_tokens, 'truncation': 'only_second',
                                'aggregation': 'max_window_sigmoid', 'version': 1}
        self._inference_strategy = {'batch_size': batch_size, 'padding': 'longest_in_batch',
                                    'pad_token_id': pad, 'version': 1}
        identity = {'repo_id': manifest['repo_id'], 'revision': manifest['revision'],
                    'files': runtime_files, 'window_strategy': self.window_strategy,
                    'inference_strategy': self.inference_strategy}
        self.model_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self._lock = Lock()
        self.last_run = {'status': 'idle', 'model_key': self.model_key}

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def inference_strategy(self):
        return dict(self._inference_strategy)

    def _query_prefix(self, query):
        self._tokenizer.no_truncation()
        encoding = self._tokenizer.encode(query, add_special_tokens=False)
        before = len(encoding.ids)
        if not before:
            raise ValueError('empty tokens')
        if before <= self.query_max_tokens:
            return query, before, before, False
        end = encoding.offsets[self.query_max_tokens - 1][1]
        if not 0 < end <= len(query):
            raise ValueError('offset')
        prefix = query[:end]
        while prefix:
            bounded = self._tokenizer.encode(prefix, add_special_tokens=False)
            if 0 < len(bounded.ids) <= self.query_max_tokens:
                return prefix, before, len(bounded.ids), True
            end = min(bounded.offsets[self.query_max_tokens][0], len(prefix) - 1)
            prefix = prefix[:end]
        raise ValueError('query token budget')

    def _infer(self, windows):
        np = self._np
        length = max(len(window.ids) for window in windows)
        if not 1 <= length <= self.max_length:
            raise ValueError('window length')
        values = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
        for window in windows:
            count = len(window.ids)
            if len(window.attention_mask) != count or len(window.type_ids) != count:
                raise ValueError('token shape')
            padding = length - count
            values['input_ids'].append([*window.ids, *([self.pad_id] * padding)])
            values['attention_mask'].append([*window.attention_mask, *([0] * padding)])
            values['token_type_ids'].append([*window.type_ids, *([0] * padding)])
        feed = {item.name: np.asarray(values[item.name], dtype=np.int64 if item.type == 'tensor(int64)' else np.int32)
                for item in self._inputs}
        output = self._session.run(None, feed)
        if not output:
            raise ValueError('no output')
        scores = np.asarray(output[0])
        if scores.shape == (len(windows), 1):
            scores = scores[:, 0]
        if scores.shape != (len(windows),) or scores.dtype.kind not in 'fiu':
            raise ValueError('logit shape')
        scores = scores.astype(np.float64)
        if not np.isfinite(scores).all():
            raise ValueError('nonfinite logits')
        result = np.empty_like(scores)
        positive = scores >= 0
        result[positive] = 1 / (1 + np.exp(-scores[positive]))
        exp_values = np.exp(scores[~positive])
        result[~positive] = exp_values / (1 + exp_values)
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError('invalid score')
        return result

    def __call__(self, query: str, passages: list[str]) -> list[float]:
        with self._lock:
            stats = {'status': 'started', 'model_key': self.model_key, 'window_strategy': dict(self.window_strategy),
                     'inference_strategy': self.inference_strategy,
                     'documents': len(passages) if isinstance(passages, list) else 0,
                     'unique_passages': 0, 'query_tokens_before': 0, 'query_tokens_used': 0,
                     'query_truncated': False, 'planned_windows': 0,
                     'computed_windows': 0, 'computed_batches': 0,
                     'passage_window_counts': [], 'winning_window_indices': []}
            self.last_run = stats
            if (not isinstance(query, str) or not query.strip() or not isinstance(passages, list)
                    or any(not isinstance(text, str) or not text.strip() for text in passages)):
                stats['status'] = 'invalid_input'
                raise ValueError('重排查询须为非空文本，段落须为非空文本组成的列表')
            if not passages:
                stats['status'] = 'ok'
                return []
            stage = 'tokenization'
            try:
                bounded, before, used, truncated = self._query_prefix(query)
                stats.update(query_tokens_before=before, query_tokens_used=used, query_truncated=truncated)
                unique = list(dict.fromkeys(passages))
                indices = {text: index for index, text in enumerate(unique)}
                stats['unique_passages'] = len(unique)
                self._tokenizer.enable_truncation(max_length=self.max_length, stride=self.stride,
                                                 strategy='only_second')
                encodings = self._tokenizer.encode_batch([(bounded, text) for text in unique])
                if len(encodings) != len(unique):
                    raise ValueError('pair count')
                windows, counts = [], []
                for index, encoding in enumerate(encodings):
                    parts = [encoding, *encoding.overflowing]
                    counts.append(len(parts))
                    for ordinal, window in enumerate(parts):
                        if not window.ids or 1 not in window.sequence_ids:
                            raise ValueError('empty passage tokens')
                        windows.append((index, ordinal, window))
                stats['passage_window_counts'] = [counts[indices[text]] for text in passages]
                stats['planned_windows'] = len(windows)
                best = [-1.] * len(unique)
                winners = [-1] * len(unique)
                stage = 'inference'
                for start in range(0, len(windows), self.batch_size):
                    batch = windows[start:start + self.batch_size]
                    scores = self._infer([window for _, _, window in batch])
                    stats['computed_windows'] += len(batch)
                    stats['computed_batches'] += 1
                    for (index, ordinal, _), score in zip(batch, scores):
                        if score > best[index]:
                            best[index], winners[index] = float(score), ordinal
                stats.update(status='ok', winning_window_indices=[winners[indices[text]] for text in passages])
                return [best[indices[text]] for text in passages]
            except Exception:
                stats.update(status='error', error_code=stage + '_failed')
                label = '分词' if stage == 'tokenization' else '推理或输出校验'
                raise RuntimeError('本机相关性重排' + label + '失败') from None
