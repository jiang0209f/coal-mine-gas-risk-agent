"""Frozen evidence labels and independent evaluation arithmetic."""
from importlib import import_module
import json
from pathlib import Path
import pytest
FIXTURES = Path(__file__).parents[1] / 'fixtures'
DATA = FIXTURES / 'knowledge_retrieval_benchmark.json'

@pytest.fixture
def benchmark():
    return import_module('app.scripts.evaluate_knowledge_retrieval')

def test_frozen_fixture_sha_and_balanced_topic_splits(benchmark):
    main = benchmark.load_benchmark(DATA, split='all')
    manifest = json.loads((FIXTURES / 'knowledge_retrieval_benchmark_manifest.json').read_text(encoding='utf-8'))
    assert manifest['files']['dev_and_candidates']['sha256'] == 'fa5b25806696984903f1077f56cce8809a5d45c25c0c17c67a16bd29ef3e32b8'
    assert manifest['files']['heldout']['sha256'] == 'f35ef5a3b994ea72911eccd84a03fe689058d82df1d6bfd86120dbf5d3ae7ebf'
    assert len(main['candidates']) == 36 and len(main['queries']) == 78
    supported = [row for row in main['queries'] if row['label'] == 'supported']
    assert len(supported) == 48
    assert len({row['topic_id'] for row in supported}) == 12
    for topic in {row['topic_id'] for row in supported}:
        rows = [row for row in supported if row['topic_id'] == topic]
        assert len(rows) == 4 and sum((row['split'] == 'dev' for row in rows)) == 2
        assert all((len(row['supporting_reference_ids']) == 1 and len(row['hard_negative_reference_ids']) == 2 for row in rows))

def test_development_loading_does_not_open_heldout_questions(benchmark, monkeypatch):
    original = Path.read_text
    original_bytes = Path.read_bytes

    def guarded(path, *args, **kwargs):
        if path.name == 'knowledge_retrieval_heldout.json':
            raise AssertionError('development run opened heldout questions')
        return original(path, *args, **kwargs)

    def guarded_bytes(path, *args, **kwargs):
        if path.name == 'knowledge_retrieval_heldout.json':
            raise AssertionError('development run opened heldout questions')
        return original_bytes(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    monkeypatch.setattr(Path, 'read_bytes', guarded_bytes)
    loaded = benchmark.load_benchmark(DATA)
    assert len(loaded['queries']) == 39 and all((row['split'] == 'dev' for row in loaded['queries']))
    with pytest.raises(AssertionError, match='opened heldout questions'):
        benchmark.load_benchmark(DATA, split='heldout')

def test_changed_fixture_is_rejected_before_evaluation(benchmark, tmp_path):
    for name in ('knowledge_retrieval_benchmark.json', 'knowledge_retrieval_benchmark_manifest.json'):
        (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
    (tmp_path / DATA.name).write_bytes((tmp_path / DATA.name).read_bytes() + b' ')
    with pytest.raises(ValueError, match='SHA'):
        benchmark.load_benchmark(tmp_path / DATA.name)

def _mini():
    candidates = [{'reference_id': name, 'title': name, 'text': '合成正文', 'locator': {'paragraph': 1}, 'version': 'v1'} for name in 'ABC']

    def query(query_id, label, gold, **extra):
        return {'query_id': query_id, 'topic_id': 'synthetic', 'split': 'dev', 'label': label, 'query': query_id, 'supporting_reference_ids': gold, 'retrieval_expectation': 'no_supporting_passage', **extra}
    questions = [query('q1', 'supported', ['A']), query('q2', 'supported', ['B']), query('q3', 'supported', ['B', 'C']), query('q4', 'irrelevant', []), query('q5', 'unsupported', []), query('q6', 'unsupported', [], retrieval_expectation='background_allowed', background_relevant_reference_ids=['C'])]
    return (candidates, questions)

def test_metrics_count_support_recall_ranks_and_separate_background(benchmark):
    candidates, queries = _mini()
    ranks = {'q1': ['A', 'C'], 'q2': ['A', 'B'], 'q3': ['C'], 'q4': [], 'q5': ['A'], 'q6': ['C']}
    sentinel = object()
    calls = []

    def retrieve(query, candidates, embedding=None, limit=5):
        calls.append((query, embedding, limit))
        return [{'reference_id': name, 'score': 1.0} for name in ranks[query]]
    report = benchmark.evaluate_benchmark(candidates, queries, retrieve, embedding=sentinel)
    metrics = report['metrics']
    assert metrics['positive_queries'] == 3
    assert metrics['recall_at_5'] == pytest.approx(5 / 6)
    assert metrics['hit_at_5'] == 1
    assert metrics['mrr_at_5'] == pytest.approx(5 / 6)
    assert metrics['top1'] == pytest.approx(2 / 3)
    assert metrics['negative_queries'] == 2 and metrics['negative_returned_queries'] == 1
    assert metrics['negative_false_return_rate'] == 0.5
    assert metrics['background_probes'] == 1 and metrics['background_match_rate'] == 1
    assert len(report['per_query']) == 6
    assert all((embedding is sentinel and limit == 5 for _, embedding, limit in calls))

def test_empty_return_has_zero_positive_recall_and_no_false_return(benchmark):
    candidates, queries = _mini()
    report = benchmark.evaluate_benchmark(candidates, queries, lambda *args, **kwargs: [])
    metrics = report['metrics']
    assert metrics['recall_at_5'] == 0 and metrics['mrr_at_5'] == 0 and (metrics['top1'] == 0)
    assert metrics['negative_false_return_rate'] == 0

@pytest.mark.parametrize('bad_rows', [[{'reference_id': 'UNKNOWN'}], [{'reference_id': 'A'}, {'reference_id': 'A'}], None])
def test_bad_retriever_output_is_not_scored_as_a_valid_rank(benchmark, bad_rows):
    candidates, queries = _mini()
    with pytest.raises(ValueError, match='检索结果'):
        benchmark.evaluate_benchmark(candidates, queries[:1], lambda *args, **kwargs: bad_rows)

def test_unknown_gold_is_rejected(benchmark):
    candidates, queries = _mini()
    queries[0]['supporting_reference_ids'] = ['UNKNOWN']
    with pytest.raises(ValueError, match='金标'):
        benchmark.evaluate_benchmark(candidates, queries, lambda *args, **kwargs: [])

def test_only_first_k_results_contribute_to_metrics(benchmark):
    candidates, queries = _mini()
    result = benchmark.evaluate_benchmark(candidates, queries[:1], lambda *args, **kwargs: [{'reference_id': 'B'}, {'reference_id': 'A'}], k=1)
    assert result['metrics']['recall_at_1'] == 0 and result['metrics']['top1'] == 0

def test_background_only_probe_is_not_counted_as_a_generated_fact_error(benchmark):
    candidates, queries = _mini()
    report = benchmark.evaluate_benchmark(candidates, queries[-1:], lambda *args, **kwargs: [{'reference_id': 'C'}])
    assert report['metrics']['negative_queries'] == 0
    assert report['metrics']['negative_false_return_rate'] is None
    assert report['metrics']['background_match_rate'] == 1
    assert report['per_query'][0]['retrieval_expectation'] == 'background_allowed'
