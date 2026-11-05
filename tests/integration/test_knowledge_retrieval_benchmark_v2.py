"""Version-2 fixture loading keeps version-1 defaults and sealed data separate."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

from app.scripts import evaluate_knowledge_retrieval as harness

FIXTURES = Path(__file__).parents[1] / "fixtures" / "knowledge_retrieval_v2"


def test_v2_dev_loads_with_an_explicit_manifest_and_balanced_styles():
    loaded = harness.load_benchmark(FIXTURES / "dev.json", manifest_path=FIXTURES / "manifest.json")
    assert len(loaded["candidates"]) == 60 and len(loaded["queries"]) == 60
    positives = [row for row in loaded["queries"] if row["label"] == "supported"]
    assert len(positives) == 36
    assert Counter(row["style"] for row in positives) == {
        "direct": 6, "colloquial": 6, "ellipsis": 6, "negation": 6, "conditional": 6, "long": 6,
    }
    assert Counter(row["topic_id"] for row in positives) == {topic: 3 for topic in {row["topic_id"] for row in positives}}
    assert all(row["split"] == "dev" for row in loaded["queries"])
    assert loaded["metadata"]["split"] == "dev"


def test_v2_dev_never_opens_heldout_bytes_or_text(monkeypatch):
    original_bytes, original_text = Path.read_bytes, Path.read_text

    def guarded_bytes(path, *args, **kwargs):
        assert path.name != "heldout.json", "sealed questions were opened"
        return original_bytes(path, *args, **kwargs)

    def guarded_text(path, *args, **kwargs):
        assert path.name != "heldout.json", "sealed questions were opened"
        return original_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_text)
    loaded = harness.load_benchmark(FIXTURES / "dev.json", manifest_path=FIXTURES / "manifest.json")
    assert len(loaded["queries"]) == 60


def test_custom_manifest_name_can_load_a_synthetic_reserved_file(tmp_path):
    main = json.loads((FIXTURES / "dev.json").read_text(encoding="utf-8"))
    query = {**main["queries"][0], "query_id": "loader-only-synthetic", "query": "synthetic loader question",
             "split": "heldout"}
    sealed = {"queries": [query]}
    main_bytes = json.dumps(main, ensure_ascii=False).encode()
    sealed_bytes = json.dumps(sealed, ensure_ascii=False).encode()
    (tmp_path / "open.json").write_bytes(main_bytes)
    (tmp_path / "reserved-loader-only.json").write_bytes(sealed_bytes)
    manifest = {"revision": "loader-test", "frozen_at": "2026-11-04T00:00:00+00:00",
                "files": {"dev_and_candidates": {"path": "open.json", "sha256": hashlib.sha256(main_bytes).hexdigest()},
                          "heldout": {"path": "reserved-loader-only.json", "sha256": hashlib.sha256(sealed_bytes).hexdigest()}}}
    manifest_path = tmp_path / "custom-freeze.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = harness.load_benchmark(tmp_path / "open.json", split="heldout", manifest_path=manifest_path)
    assert [row["query_id"] for row in loaded["queries"]] == ["loader-only-synthetic"]


def test_cli_forwards_optional_manifest_without_loading_a_model(monkeypatch, tmp_path):
    calls = []

    def stop_after_load(path, split="dev", manifest_path=None):
        calls.append((path, split, manifest_path))
        raise RuntimeError("stop before model construction")

    monkeypatch.setattr(harness, "load_benchmark", stop_after_load)
    for name in ("HF_HOME", "TMP", "TEMP", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "HF_HUB_DISABLE_TELEMETRY", "TOKENIZERS_PARALLELISM"):
        monkeypatch.setenv(name, os.environ.get(name, ""))
    manifest = str(FIXTURES / "manifest.json")
    monkeypatch.setattr(sys, "argv", ["evaluate", "--dataset", str(FIXTURES / "dev.json"), "--manifest", manifest,
        "--model-dir", str(tmp_path / "unused-model"), "--index-dir", str(tmp_path / "unused-index"),
        "--output", str(tmp_path / "report.json"), "--system-revision", "synthetic"])
    with pytest.raises(RuntimeError, match="stop before model"):
        harness.main()
    assert calls == [(str(FIXTURES / "dev.json"), "dev", manifest)]
