"""Benchmark validity must expose vector fallback and honor CLI top-k."""
import importlib
import json
import os
from pathlib import Path
import sys

import pytest

from app.scripts.evaluate_knowledge_retrieval import evaluate_benchmark
from app.services.knowledge.retrieval import retrieve


def candidates(count=1):
    return [{"reference_id": f"ref-{index}", "title": "合成瓦斯记录", "text": "瓦斯检查记录。",
             "locator": {"paragraph": index + 1}, "version": "synthetic"} for index in range(count)]


def questions():
    return [{"query_id": "supported", "topic_id": "synthetic", "split": "dev", "label": "supported",
             "query": "瓦斯", "supporting_reference_ids": ["ref-0"]},
            {"query_id": "negative", "topic_id": "irrelevant", "split": "dev", "label": "irrelevant",
             "query": "巴黎时差", "supporting_reference_ids": []}]


def test_vector_failure_marks_even_empty_result_invalid():
    def broken(texts):
        raise RuntimeError("synthetic private exception body")

    report = evaluate_benchmark(candidates(), questions(), retrieve, embedding=broken)
    assert report["validity"]["valid"] is False
    assert report["validity"]["invalid_query_ids"] == ["supported", "negative"]
    assert report["diagnostics"]["fallback_queries"] == 2
    assert report["diagnostics"]["embedding_status_counts"] == {"error:RuntimeError": 2}
    assert all(row["retrieval_mode"] == "lexical_fallback" and not row["valid"] for row in report["per_query"])
    assert report["metrics"]["positive_queries"] == 0
    assert report["metrics_all_queries"]["positive_queries"] == 1
    assert "private exception body" not in str(report)


def test_invalid_vectors_are_counted_as_invalid_not_successful_semantics():
    report = evaluate_benchmark(candidates(), questions(), retrieve, embedding=lambda texts: [])
    assert report["validity"]["valid"] is False
    assert report["diagnostics"]["embedding_status_counts"] == {"invalid_vectors": 2}
    assert report["diagnostics"]["fallback_queries"] == 2


@pytest.mark.parametrize("change", ["unknown_expectation", "unknown_hard_negative"])
def test_typo_or_unknown_distractor_is_rejected(change):
    query = questions()[0]
    if change == "unknown_expectation":
        query["retrieval_expectation"] = "background_allowd"
    else:
        query["hard_negative_reference_ids"] = ["UNKNOWN-SYNTHETIC"]
    with pytest.raises(ValueError, match="金标"):
        evaluate_benchmark(candidates(), [query], lambda *args, **kwargs: [])


def setup_cli(monkeypatch, tmp_path, encoder, retriever=None, count=1):
    harness = importlib.import_module("app.scripts.evaluate_knowledge_retrieval")
    adapter_module = importlib.import_module("app.adapters.text_embeddings")
    retrieval_module = importlib.import_module("app.services.knowledge.retrieval")
    monkeypatch.setattr(adapter_module, "LocalTextEmbeddings", encoder)
    if retriever is not None:
        monkeypatch.setattr(retrieval_module, "retrieve", retriever)
    loaded = {"candidates": candidates(count), "queries": questions()[:1], "metadata": {"split": "dev"}}
    monkeypatch.setattr(harness, "load_benchmark", lambda *args, **kwargs: loaded)
    model = tmp_path / "model"
    model.mkdir()
    (model / "manifest.json").write_text(json.dumps({
        "repo_id": "synthetic", "revision": "a" * 40, "files": {"model.onnx": {"sha256": "b" * 64}},
    }), encoding="utf-8")
    for name in ("HF_HOME", "TMP", "TEMP", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    args = ["evaluate-benchmark", "--dataset", str(tmp_path / "unused.json"), "--model-dir", str(model),
            "--index-dir", str(tmp_path / "index"), "--output", str(tmp_path / "report.json"),
            "--system-revision", "synthetic"]
    return harness, args, tmp_path / "report.json"


@pytest.mark.parametrize("k", [5, 10, 50])
def test_cli_top_k_is_forwarded_and_caps_report(monkeypatch, tmp_path, k):
    class Encoder:
        model_key = "synthetic"
        last_run = {}
        def __init__(self, *args):
            pass
        def __call__(self, texts):
            return [[1.0, 0.0] for _ in texts]

    calls = []
    def ranked(query, rows, embedding=None, limit=5):
        calls.append(limit)
        return [{"reference_id": row["reference_id"], "score": 1} for row in rows]

    harness, args, path = setup_cli(monkeypatch, tmp_path, Encoder, ranked, count=75)
    monkeypatch.setattr(sys, "argv", [*args, "--k", str(k)])
    assert harness.main() == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert calls == [k] and report["k"] == k
    assert len(report["per_query"][0]["retrieved_reference_ids"]) == k


def test_cli_writes_invalid_fallback_report_and_returns_nonzero(monkeypatch, tmp_path):
    class Encoder:
        model_key = "synthetic"
        last_run = {}
        def __init__(self, *args):
            self.calls = 0
        def __call__(self, texts):
            self.calls += 1
            if self.calls == 1:
                return [[1.0, 0.0] for _ in texts]
            raise RuntimeError("synthetic unavailable")

    harness, args, path = setup_cli(monkeypatch, tmp_path, Encoder)
    monkeypatch.setattr(sys, "argv", args)
    assert harness.main() != 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["validity"]["valid"] is False
    assert report["diagnostics"]["fallback_queries"] == 1
