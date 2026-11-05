"""Reranking evaluation records provenance and invalidates safe fallbacks."""

import importlib
import json
import os
from pathlib import Path
import sys
import warnings

import pytest

from app.scripts import evaluate_knowledge_retrieval as harness
from app.services.knowledge.retrieval import retrieve


def candidates():
    return [{"reference_id": name, "title": "合成瓦斯记录", "text": "瓦斯检查记录。",
             "locator": {"paragraph": index + 1}, "version": "synthetic"}
            for index, name in enumerate(("support", "near"))]


def questions():
    return [{"query_id": "supported", "topic_id": "synthetic", "split": "dev", "label": "supported",
             "query": "瓦斯", "supporting_reference_ids": ["support"], "hard_negative_reference_ids": ["near"]},
            {"query_id": "negative", "topic_id": "synthetic", "split": "dev", "label": "irrelevant",
             "query": "巴黎时差", "supporting_reference_ids": []}]


def orthogonal_embeddings(texts):
    return [[1.0, 0.0], *([[0.0, 1.0]] * (len(texts) - 1))]


def test_no_reranker_keeps_legacy_retriever_signature():
    def legacy(query, rows, embedding=None, limit=5):
        return []

    report = harness.evaluate_benchmark(candidates(), questions(), legacy)
    assert report["validity"]["valid"] is True
    assert report["diagnostics"]["reranker_status_counts"] == {"not_configured": 2}


def test_reranker_exception_invalidates_nonempty_and_empty_fallback():
    def broken(query, passages):
        raise RuntimeError("private synthetic error body")

    report = harness.evaluate_benchmark(candidates(), questions(), retrieve,
                                        embedding=orthogonal_embeddings, reranker=broken)
    assert report["validity"]["valid"] is False
    assert report["validity"]["invalid_query_ids"] == ["supported", "negative"]
    assert report["diagnostics"]["reranker_fallback_queries"] == 2
    assert report["diagnostics"]["reranker_status_counts"] == {"error:RuntimeError": 2}
    assert all(row["reranking_mode"] == "hybrid_fallback" for row in report["per_query"])
    assert report["per_query"][1]["retrieved_reference_ids"] == []
    assert report["metrics"]["positive_queries"] == 0
    assert report["metrics_all_queries"]["positive_queries"] == 1
    assert "private synthetic error body" not in str(report)


@pytest.mark.parametrize("bad", [[], None, [float("nan"), 0.5], [2.0, 0.5]])
def test_invalid_scores_invalidate_even_when_fallback_returns_nothing(bad):
    report = harness.evaluate_benchmark(candidates(), questions(), retrieve,
        embedding=orthogonal_embeddings, reranker=lambda query, passages: bad)
    assert report["validity"]["valid"] is False
    assert report["diagnostics"]["reranker_status_counts"] == {"invalid_scores": 2}
    assert report["diagnostics"]["reranker_fallback_queries"] == 2
    assert report["per_query"][1]["retrieved_reference_ids"] == []


def test_empty_warning_only_result_is_invalid():
    def fallback(query, rows, embedding=None, limit=5, reranker=None):
        warnings.warn("检索重排不可用，已退回原混合检索", RuntimeWarning)
        return []

    report = harness.evaluate_benchmark(candidates(), questions()[:1], fallback,
                                        reranker=lambda query, passages: [0.5] * len(passages))
    assert report["validity"]["valid"] is False
    assert report["per_query"][0]["reranker_status"] == "unavailable"
    assert report["diagnostics"]["reranker_fallback_queries"] == 1


def test_successful_reranker_can_validly_return_no_admitted_candidates():
    report = harness.evaluate_benchmark(candidates(), questions()[1:], retrieve,
        embedding=orthogonal_embeddings, reranker=lambda query, passages: [0.0] * len(passages))
    assert report["validity"]["valid"] is True
    assert report["per_query"][0]["retrieved_reference_ids"] == []
    assert report["per_query"][0]["reranker_status"] == "available"
    assert report["diagnostics"]["reranker_fallback_queries"] == 0


def test_mixed_reranker_status_keeps_failure_code():
    def mixed(query, rows, embedding=None, limit=5, reranker=None):
        return [{"reference_id": "support", "reranker_status": "available"},
                {"reference_id": "near", "reranker_status": "invalid_scores"}]

    report = harness.evaluate_benchmark(candidates(), questions()[:1], mixed,
                                        reranker=lambda query, passages: [0.5] * len(passages))
    assert report["validity"]["valid"] is False
    assert report["per_query"][0]["reranker_status"] == "invalid_scores"
    assert report["diagnostics"]["reranker_fallback_queries"] == 1


def test_report_preserves_rerank_path_two_ranks_and_fusion():
    def ranked(query, rows, embedding=None, limit=5, reranker=None):
        assert callable(reranker)
        assert reranker(query, ["synthetic passage"]) == [0.75]
        return [{"reference_id": "support", "score": 0.8, "rerank_score": 0.75,
                 "reranker_status": "available", "reranking_mode": "cross_encoder",
                 "semantic_rank": 3, "rerank_rank": 1, "fusion_method": "reciprocal_rank",
                 "retrieval_rank": 2, "retrieval_rank_source": "hybrid",
                 "recall_path": "original_hybrid", "retrieval_mode": "hybrid", "embedding_status": "available"}]

    report = harness.evaluate_benchmark(candidates(), questions()[:1], ranked,
                                        reranker=lambda query, passages: [0.75])
    row = report["per_query"][0]
    assert row["reranker_status"] == "available"
    assert row["recall_paths"] == ["original_hybrid"]
    ranking = row["ranking"][0]
    assert {key: ranking[key] for key in ("semantic_rank", "rerank_rank", "fusion_method", "recall_path")} == {
        "semantic_rank": 3, "rerank_rank": 1, "fusion_method": "reciprocal_rank", "recall_path": "original_hybrid"}
    assert ranking["rerank_score"] == 0.75
    assert ranking["retrieval_rank"] == 2 and ranking["retrieval_rank_source"] == "hybrid"


def test_hard_negative_metrics_use_only_annotated_questions():
    queries = [{**questions()[0], "query_id": str(index), "query": str(index)} for index in range(4)]
    queries[-1].pop("hard_negative_reference_ids")
    ranks = {"0": ["near", "support"], "1": ["support", "near"], "2": [], "3": ["near"]}
    report = harness.evaluate_benchmark(candidates(), queries,
        lambda query, rows, embedding=None, limit=5: [{"reference_id": ref} for ref in ranks[query]])
    assert report["metrics"]["hard_negative_queries"] == 3
    assert report["metrics"]["hard_negative_top1_rate"] == pytest.approx(1 / 3)
    assert report["metrics"]["hard_negative_at_5_rate"] == pytest.approx(2 / 3)
    assert report["per_query"][0]["top1_is_hard_negative"] is True


@pytest.mark.parametrize("fails", [False, True])
def test_cli_loads_optional_local_reranker_and_reports_provenance(monkeypatch, tmp_path, fails):
    embedding_module = importlib.import_module("app.adapters.text_embeddings")
    reranker_module = importlib.import_module("app.adapters.text_reranking")
    class Encoder:
        model_key = "synthetic-embedding-key"
        last_run = {}
        def __init__(self, *args):
            pass
        def __call__(self, texts):
            return orthogonal_embeddings(texts)

    initialized = []
    class Reranker:
        model_key = "synthetic-reranker-key"
        window_strategy = {"max_length": 512, "stride": 128, "aggregation": "max_window_sigmoid"}
        inference_strategy = {"batch_size": 1, "padding": "longest_in_batch", "pad_token_id": 1, "version": 1}
        last_run = {"status": "ok"}
        def __init__(self, path):
            initialized.append(Path(path))
        def __call__(self, query, passages):
            if fails:
                raise RuntimeError("private synthetic failure")
            return [0.7] * len(passages)

    monkeypatch.setattr(embedding_module, "LocalTextEmbeddings", Encoder)
    monkeypatch.setattr(reranker_module, "LocalTextReranker", Reranker)
    loaded = {"candidates": candidates(), "queries": questions()[:1], "metadata": {"split": "dev"}}
    monkeypatch.setattr(harness, "load_benchmark", lambda *args, **kwargs: loaded)
    embedding_dir, reranker_dir = tmp_path / "embedding", tmp_path / "reranker"
    embedding_dir.mkdir()
    reranker_dir.mkdir()
    (embedding_dir / "manifest.json").write_text(json.dumps({"repo_id": "synthetic-embedding", "revision": "a" * 40,
        "files": {"model.onnx": {"sha256": "b" * 64}}}), encoding="utf-8")
    reranker_manifest = {"repo_id": "synthetic-reranker", "revision": "c" * 40,
                         "model_file": "onnx/model.onnx", "files": {"onnx/model.onnx": {"sha256": "d" * 64}}}
    (reranker_dir / "manifest.json").write_text(json.dumps(reranker_manifest), encoding="utf-8")
    for name in ("HF_HOME", "TMP", "TEMP", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["evaluate", "--dataset", str(tmp_path / "unused.json"),
        "--model-dir", str(embedding_dir), "--index-dir", str(tmp_path / "index"),
        "--reranker-model-dir", str(reranker_dir), "--output", str(output), "--system-revision", "synthetic"])
    assert harness.main() == (1 if fails else 0)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert initialized == [reranker_dir]
    provenance = report["reranker_model"]
    assert provenance["repo_id"] == "synthetic-reranker"
    assert provenance["revision"] == "c" * 40
    assert provenance["onnx_sha256"] == "d" * 64
    assert provenance["model_key"] == "synthetic-reranker-key"
    assert provenance["window_strategy"] == Reranker.window_strategy
    assert provenance["inference_strategy"] == Reranker.inference_strategy
    assert len(provenance["manifest_sha256"]) == 64
    assert report["validity"]["valid"] is (not fails)
    assert report["network_attempts"] == 0 and report["cloud_calls"] == 0
