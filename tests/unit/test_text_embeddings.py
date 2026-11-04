"""Offline embedding and incremental cache contracts, with synthetic backends."""
from importlib import import_module
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def adapter():
    return import_module("app.adapters.text_embeddings").LocalTextEmbeddings


def write_model(path, revision="a" * 40, tokenizer_text="synthetic tokenizer"):
    path.mkdir(exist_ok=True)
    config = {"hidden_size": 3, "max_position_embeddings": 512, "pad_token_id": 0, "model_type": "bert"}
    (path / "model.onnx").write_bytes(b"synthetic graph")
    (path / "tokenizer.json").write_text(tokenizer_text, encoding="utf-8")
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    files = {name: {"sha256": hashlib.sha256((path / name).read_bytes()).hexdigest()}
             for name in ("model.onnx", "tokenizer.json", "config.json")}
    (path / "manifest.json").write_text(json.dumps({"repo_id": "BAAI/bge-small-zh-v1.5", "revision": revision, "files": files}), encoding="utf-8")


class FakeTokenizer:
    def __init__(self):
        self.encoded = []
        self.max_length = 512
        self.stride = 128

    def no_padding(self):
        pass

    def enable_truncation(self, max_length, stride):
        self.max_length, self.stride = max_length, stride

    def encode_batch(self, texts):
        self.encoded.extend(texts)
        results = []
        for text in texts:
            values = [7 if character == "末" else 1 for character in text]
            step = self.max_length - 2 - self.stride
            windows = []
            for start in range(0, len(values), step):
                ids = [101, *values[start:start + self.max_length - 2], 102]
                windows.append(SimpleNamespace(ids=ids, attention_mask=[1] * len(ids), type_ids=[0] * len(ids), overflowing=[]))
                if start + self.max_length - 2 >= len(values):
                    break
            windows[0].overflowing = windows[1:]
            results.append(windows[0])
        return results


class FakeSession:
    def __init__(self, output=None):
        self.calls = []
        self.output = output

    def get_inputs(self):
        return [SimpleNamespace(name=name, type="tensor(int64)")
                for name in ("input_ids", "attention_mask", "token_type_ids")]

    def run(self, names, feed):
        self.calls.append(feed)
        if self.output is not None:
            return [self.output(feed)]
        batch, length = feed["input_ids"].shape
        output = np.ones((batch, length, 3)) * 100
        for index, ids in enumerate(feed["input_ids"]):
            output[index, 0] = [0, 1, 0] if 7 in ids else [3, 4, 0]
        return [output]


@pytest.fixture
def parts(tmp_path, adapter):
    model = tmp_path / "model"
    write_model(model)
    index = tmp_path / "index"
    tokenizer, session = FakeTokenizer(), FakeSession()
    factories = []
    def session_factory(path, **kwargs):
        factories.append((path, kwargs))
        return session
    def make(**kwargs):
        return adapter(model, index, tokenizer_factory=lambda path: tokenizer,
                       session_factory=session_factory, **kwargs)
    return model, index, tokenizer, session, factories, make


def test_offline_paths_cpu_provider_cls_pooling_and_normalization(parts):
    model, _, tokenizer, session, factories, make = parts
    result = make()(["合成条文"])
    assert np.allclose(result, [[0.6, 0.8, 0]])
    assert factories[0][0] == str(model / "model.onnx")
    assert factories[0][1]["providers"] == ["CPUExecutionProvider"]
    assert len(session.calls) == 1 and tokenizer.max_length == 512
    feed = session.calls[0]
    assert set(feed) == {"input_ids", "attention_mask", "token_type_ids"}
    assert all(value.dtype == np.int64 for value in feed.values())


def test_cache_survives_new_adapter_and_only_new_text_is_computed(parts):
    _, _, tokenizer, session, _, make = parts
    first = make()(["甲", "乙", "甲"])
    assert len(tokenizer.encoded) == 2 and first[0] == first[2]
    session.calls.clear()
    second = make()
    assert second(["乙", "甲"]) == [first[1], first[0]]
    assert session.calls == [] and second.last_run["cache_hits"] == 2
    second(["甲", "新条文"])
    assert tokenizer.encoded[-1:] == ["新条文"]
    assert second.last_run["computed_texts"] == 1


@pytest.mark.parametrize("change", ["revision", "tokenizer", "window"])
def test_cache_identity_binds_revision_tokenizer_and_window_strategy(parts, change):
    model, _, tokenizer, _, _, make = parts
    first = make()
    first(["合成条文"])
    if change == "revision":
        write_model(model, revision="b" * 40)
    elif change == "tokenizer":
        write_model(model, tokenizer_text="changed tokenizer")
    second = make(max_length=256) if change == "window" else make()
    second(["合成条文"])
    assert second.model_key != first.model_key
    assert tokenizer.encoded.count("合成条文") == 2


def test_long_passage_uses_late_windows_instead_of_silently_truncating(parts):
    _, _, _, session, _, make = parts
    embeddings = make(batch_size=2, max_length=8, stride=2)
    result = embeddings(["前" * 18 + "末"])
    assert len(session.calls) >= 2
    assert embeddings.last_run["computed_windows"] > 1
    assert result[0][1] > 0.8 and result[0][0] < 0.6
    assert all(call["input_ids"].shape[1] <= 8 for call in session.calls)
    assert np.linalg.norm(result[0]) == pytest.approx(1.0)


def test_all_batches_use_masks_and_only_requested_batch_size(parts):
    _, _, _, session, _, make = parts
    result = make(batch_size=2)(["甲", "甲乙丙丁", "乙", "丙丁", "戊"])
    assert len(result) == 5
    assert [item["input_ids"].shape[0] for item in session.calls] == [2, 2, 1]
    assert session.calls[0]["attention_mask"][0, -1] == 0
    assert session.calls[0]["input_ids"][0, -1] == 0


@pytest.mark.parametrize("value", [None, "瓦斯", [""], ["   "], ["甲", 1]])
def test_input_is_a_list_of_nonempty_texts(parts, value):
    *_, make = parts
    with pytest.raises(ValueError, match="文本"):
        make()(value)


def test_empty_batch_does_not_encode_or_infer(parts):
    _, _, tokenizer, session, _, make = parts
    assert make()([]) == []
    assert tokenizer.encoded == [] and session.calls == []


@pytest.mark.parametrize("output", [
    lambda feed: np.ones((1, 2)),
    lambda feed: np.zeros((1, 3)),
    lambda feed: np.array([[float("nan"), 1, 2]]),
    lambda feed: np.ones((2, 3)),
    lambda feed: np.ones((1,)),
])
def test_invalid_output_is_rejected_and_not_cached(parts, output):
    _, _, _, session, _, make = parts
    session.output = output
    embeddings = make()
    with pytest.raises(RuntimeError, match="向量"):
        embeddings(["合成条文"])
    session.output = None
    session.calls.clear()
    assert embeddings(["合成条文"])
    assert len(session.calls) == 1


def test_sentence_vector_output_is_also_supported(parts):
    _, _, _, session, _, make = parts
    session.output = lambda feed: np.array([[3, 4, 0]] * len(feed["input_ids"]))
    assert np.allclose(make()(["甲"]), [[0.6, 0.8, 0]])


def test_artifact_integrity_is_checked_before_creating_a_session(parts):
    model, _, _, _, factories, make = parts
    (model / "model.onnx").write_bytes(b"changed graph")
    with pytest.raises(ValueError, match="SHA"):
        make()
    assert factories == []


def test_corrupt_cached_vectors_are_recomputed(parts):
    _, index, _, session, _, make = parts
    embeddings = make()
    embeddings(["合成条文"])
    import sqlite3
    with sqlite3.connect(index / "text-embeddings.sqlite3") as connection:
        connection.execute("UPDATE embeddings SET vector_json = '[0,0,0]'")
    session.calls.clear()
    assert np.allclose(embeddings(["合成条文"]), [[0.6, 0.8, 0]])
    assert len(session.calls) == 1


def test_nested_cached_vector_is_recomputed_as_a_flat_vector(parts):
    _, index, _, session, _, make = parts
    embeddings = make()
    embeddings(["合成条文"])
    import sqlite3
    with sqlite3.connect(index / "text-embeddings.sqlite3") as connection:
        connection.execute("UPDATE embeddings SET vector_json = '[[0.6,0.8,0]]'")
    session.calls.clear()
    result = embeddings(["合成条文"])
    assert np.asarray(result).shape == (1, 3)
    assert len(session.calls) == 1


def test_indexing_2645_passages_and_one_incremental_update(parts):
    _, _, tokenizer, session, _, make = parts
    embeddings = make(batch_size=32)
    corpus = [f"合成条文{index:04d}" for index in range(2645)]
    assert len(embeddings(corpus)) == 2645
    assert embeddings.last_run["computed_texts"] == 2645
    assert len(session.calls) == 83
    assert len(tokenizer.encoded) == 2645
    session.calls.clear()
    assert len(embeddings([*corpus, "新增合成条文"])) == 2646
    assert embeddings.last_run["cache_hits"] == 2645
    assert embeddings.last_run["computed_texts"] == 1
    assert len(session.calls) == 1
