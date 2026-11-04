"""Offline ONNX text embeddings and a local, version-bound vector cache."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from threading import Lock

__all__ = ["LocalTextEmbeddings"]


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LocalTextEmbeddings:
    """Load verified local assets; encode all windows and cache document vectors.

    Each token window uses CLS pooling and L2 normalization. Window vectors are
    averaged, then normalized again. The cache stores hashes and vectors only.
    Optional factories allow tests to replace tokenizer/session without a model.
    """

    def __init__(self, model_dir, index_dir, *, batch_size=16, max_length=512, stride=128,
                 tokenizer_factory=None, session_factory=None):
        self.model_dir = Path(model_dir).resolve()
        self.index_dir = Path(index_dir).resolve()
        if (isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128
                or isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 4
                or isinstance(stride, bool) or not isinstance(stride, int) or not 0 <= stride < max_length - 2):
            raise ValueError("本机向量批次或窗口参数无效")
        try:
            manifest = json.loads((self.model_dir / "manifest.json").read_text(encoding="utf-8"))
            if not isinstance(manifest.get("repo_id"), str) or not re.fullmatch(r"[a-f0-9]{40}", manifest["revision"]):
                raise ValueError("model revision")
            runtime_files = {}
            for name in ("model.onnx", "tokenizer.json", "config.json"):
                actual = _sha256(self.model_dir / name)
                if actual != manifest["files"][name]["sha256"]:
                    raise ValueError("SHA mismatch")
                runtime_files[name] = actual
            config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, KeyError, TypeError, ValueError):
            raise ValueError("本机向量模型清单或SHA核验失败") from None
        dimensions = config.get("hidden_size")
        positions = config.get("max_position_embeddings")
        if (isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0
                or isinstance(positions, bool) or not isinstance(positions, int) or max_length > positions):
            raise ValueError("本机向量模型维度或窗口配置无效")
        self.dimensions, self.batch_size = dimensions, batch_size
        self.max_length, self.stride = max_length, stride
        self.pad_id = config.get("pad_token_id", 0)
        self.window_strategy = {"max_length": max_length, "stride": stride,
                                "pooling": "l2_cls_window_mean_l2", "version": 1}
        identity = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                    "files": runtime_files, "window_strategy": self.window_strategy}
        self.model_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
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
            raise RuntimeError("本机文本向量依赖尚未安装") from None
        self._np = np
        self._tokenizer = tokenizer_factory(str(self.model_dir / "tokenizer.json"))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=max_length, stride=stride)
        self._session = session_factory(str(self.model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self._inputs = self._session.get_inputs()
        if not self._inputs or any(item.name not in ("input_ids", "attention_mask", "token_type_ids")
                                   or item.type not in ("tensor(int64)", "tensor(int32)") for item in self._inputs):
            raise ValueError("本机向量模型输入格式不支持")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.index_dir / "text-embeddings.sqlite3"
        with closing(sqlite3.connect(self.cache_path, timeout=30)) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS embeddings (model_key TEXT NOT NULL, "
                               "text_sha256 TEXT NOT NULL, vector_json TEXT NOT NULL, "
                               "PRIMARY KEY (model_key, text_sha256))")
            connection.commit()
        self._lock = Lock()
        self.last_run = {}

    def _normalize(self, values, expected_rows):
        np = self._np
        try:
            array = np.asarray(values)
            if array.ndim == 3 and array.shape[1] > 0:
                array = array[:, 0, :]
            if array.shape != (expected_rows, self.dimensions) or array.dtype.kind not in "fiu":
                raise ValueError("shape")
            array = array.astype(np.float64)
            if not np.isfinite(array).all():
                raise ValueError("nonfinite")
            scale = np.max(np.abs(array), axis=1, keepdims=True)
            if np.any(scale == 0):
                raise ValueError("zero vector")
            array = array / scale
            return array / np.linalg.norm(array, axis=1, keepdims=True)
        except (TypeError, ValueError, OverflowError):
            raise RuntimeError("本机向量模型输出的形状或数值无效") from None

    def _cached(self, connection, text_sha):
        row = connection.execute("SELECT vector_json FROM embeddings WHERE model_key=? AND text_sha256=?",
                                 (self.model_key, text_sha)).fetchone()
        if row is None:
            return None
        try:
            values = json.loads(row[0])
            if (not isinstance(values, list) or len(values) != self.dimensions
                    or any(type(value) not in (int, float) for value in values)):
                return None
            vector = self._normalize([values], 1)[0]
            if not self._np.allclose(values, vector, atol=1e-6):
                return None
            return values
        except (RuntimeError, TypeError, ValueError):
            return None

    def _infer(self, windows):
        np = self._np
        length = max(len(window.ids) for window in windows)
        if length > self.max_length or length < 1:
            raise RuntimeError("本机向量分词窗口无效")
        ids, masks, types = [], [], []
        for window in windows:
            count = len(window.ids)
            if len(window.attention_mask) != count or len(window.type_ids) != count:
                raise RuntimeError("本机向量分词输出长度不一致")
            padding = length - count
            ids.append([*window.ids, *([self.pad_id] * padding)])
            masks.append([*window.attention_mask, *([0] * padding)])
            types.append([*window.type_ids, *([0] * padding)])
        values = {"input_ids": ids, "attention_mask": masks, "token_type_ids": types}
        feed = {item.name: np.asarray(values[item.name], dtype=np.int64 if item.type == "tensor(int64)" else np.int32)
                for item in self._inputs}
        outputs = self._session.run(None, feed)
        if not outputs:
            raise RuntimeError("本机向量模型未返回向量")
        return self._normalize(outputs[0], len(windows))

    def __call__(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("向量输入必须为非空文本组成的列表")
        if not texts:
            self.last_run = {"texts": 0, "unique_texts": 0, "cache_hits": 0, "computed_texts": 0, "computed_windows": 0}
            return []
        with self._lock, closing(sqlite3.connect(self.cache_path, timeout=30)) as connection:
            keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
            unique = dict(zip(keys, texts))
            vectors = {key: self._cached(connection, key) for key in unique}
            missing = [(key, text) for key, text in unique.items() if vectors[key] is None]
            stats = {"texts": len(texts), "unique_texts": len(unique), "cache_hits": len(unique) - len(missing),
                     "computed_texts": 0, "computed_windows": 0, "model_key": self.model_key,
                     "window_strategy": dict(self.window_strategy)}
            self.last_run = stats
            for start in range(0, len(missing), self.batch_size):
                documents = missing[start:start + self.batch_size]
                encoded = self._tokenizer.encode_batch([text for _, text in documents])
                if len(encoded) != len(documents):
                    raise RuntimeError("本机向量分词批次长度不一致")
                windows = [(index, window) for index, encoding in enumerate(encoded)
                           for window in [encoding, *encoding.overflowing]]
                sums = self._np.zeros((len(documents), self.dimensions))
                for offset in range(0, len(windows), self.batch_size):
                    batch = windows[offset:offset + self.batch_size]
                    embeddings = self._infer([window for _, window in batch])
                    for (index, _), vector in zip(batch, embeddings):
                        sums[index] += vector
                combined = self._normalize(sums, len(documents))
                for (key, _), vector in zip(documents, combined):
                    vectors[key] = vector.tolist()
                    connection.execute("INSERT OR REPLACE INTO embeddings(model_key,text_sha256,vector_json) VALUES(?,?,?)",
                                       (self.model_key, key, json.dumps(vectors[key], allow_nan=False)))
                connection.commit()
                stats["computed_texts"] += len(documents)
                stats["computed_windows"] += len(windows)
            return [vectors[key] for key in keys]
