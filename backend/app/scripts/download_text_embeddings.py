"""Acquire pinned original BGE safetensors and optionally export ONNX offline."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import sys

import httpx

MODEL_REPO = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
MODEL_FILES = (
    "model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.txt", "README.md", "1_Pooling/config.json",
    "sentence_bert_config.json", "config_sentence_transformers.json", "modules.json",
)
_HF = "https://huggingface.co"
_OFFLINE = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blob(path):
    digest = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned(revision):
    if not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        raise ValueError("revision必须为固定的40位提交SHA")


def _inside(root, filename):
    target = (root / filename).resolve()
    if not target.is_relative_to(root) or filename not in (*MODEL_FILES, "model.onnx"):
        raise ValueError("模型清单文件路径无效")
    return target


def _verify_remote(path, remote):
    lfs = remote.get("lfs")
    expected_size = lfs["size"] if lfs else remote.get("size")
    expected = lfs["sha256"] if lfs else remote.get("blobId")
    if not expected or expected_size is None:
        raise ValueError("原作者文件缺少校验信息")
    if path.stat().st_size != expected_size:
        return False
    return (_sha(path) if lfs else _blob(path)) == expected


def _save_manifest(root, manifest):
    partial = root / "manifest.json.part"
    partial.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(root / "manifest.json")


def download_model(model_dir, *, revision=MODEL_REVISION, client=None):
    """Download an allowlisted original checkpoint, checking Hub commit metadata."""
    _pinned(revision)
    root = Path(model_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if client is None:
        with httpx.Client(follow_redirects=True, timeout=120, trust_env=False) as owned_client:
            return download_model(root, revision=revision, client=owned_client)
    response = client.get(f"{_HF}/api/models/{MODEL_REPO}/revision/{revision}", params={"blobs": "true"})
    response.raise_for_status()
    metadata = response.json()
    if metadata.get("sha") != revision:
        raise ValueError("原作者返回的revision与固定提交不一致")
    siblings = {entry["rfilename"]: entry for entry in metadata.get("siblings", [])}
    if set(MODEL_FILES) - set(siblings):
        raise ValueError("原作者仓库缺少所需模型文件")
    manifest = {"schema_version": "local-text-embeddings-v1", "repo_id": MODEL_REPO,
                "revision": revision, "license": "MIT", "source_url": f"{_HF}/{MODEL_REPO}/tree/{revision}",
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    previous_path = root / "manifest.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else None
    if previous and (previous.get("repo_id") != MODEL_REPO or previous.get("revision") != revision):
        raise ValueError("已有模型清单的revision不一致，请使用独立目录")
    for filename in MODEL_FILES:
        target = _inside(root, filename)
        remote = siblings[filename]
        url = f"{_HF}/{MODEL_REPO}/resolve/{revision}/{filename}"
        if target.exists():
            if not _verify_remote(target, remote):
                raise ValueError("已有模型文件校验不一致：" + filename)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            try:
                expected_size = (remote.get("lfs") or remote).get("size")
                downloaded = 0
                with client.stream("GET", url) as result:
                    result.raise_for_status()
                    with partial.open("wb") as output:
                        for chunk in result.iter_bytes(1024 * 1024):
                            downloaded += len(chunk)
                            if expected_size is not None and downloaded > expected_size:
                                raise ValueError("下载文件大小校验失败：" + filename)
                            output.write(chunk)
                if not _verify_remote(partial, remote):
                    raise ValueError("下载文件摘要校验失败：" + filename)
                partial.replace(target)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        manifest["files"][filename] = {
            "sha256": _sha(target), "bytes": target.stat().st_size, "source_url": url,
            "verification": "lfs_sha256" if remote.get("lfs") else "git_blob_sha1",
            "source_digest": (remote.get("lfs") or {}).get("sha256") or remote["blobId"],
        }
    if previous and "model.onnx" in previous.get("files", {}):
        exported = previous["files"]["model.onnx"]
        target = _inside(root, "model.onnx")
        if not target.is_file() or _sha(target) != exported["sha256"]:
            raise ValueError("已有ONNX派生文件校验不一致")
        manifest["files"]["model.onnx"] = exported
        if "export" in previous:
            manifest["export"] = previous["export"]
    _save_manifest(root, manifest)
    return manifest


def _verified_manifest(model_dir):
    root = Path(model_dir).resolve()
    file = root / "manifest.json"
    if not file.is_file():
        raise ValueError("模型清单不存在，请先下载固定revision原件")
    manifest = json.loads(file.read_text(encoding="utf-8"))
    _pinned(manifest.get("revision"))
    if manifest.get("repo_id") != MODEL_REPO or set(MODEL_FILES) - set(manifest.get("files", {})):
        raise ValueError("模型清单缺少原作者文件")
    for filename, entry in manifest["files"].items():
        target = _inside(root, filename)
        if not target.is_file() or target.stat().st_size != entry["bytes"] or _sha(target) != entry["sha256"]:
            raise ValueError("模型文件校验失败：" + filename)
    return root, manifest


def _versions(names):
    found = {"python": sys.version.split()[0]}
    for name in names:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def export_model(model_dir, *, parity_path=None):
    """Export the verified original safetensors with offline-only Transformers."""
    root, manifest = _verified_manifest(model_dir)
    for name in _OFFLINE:
        os.environ[name] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(root, local_files_only=True, trust_remote_code=False,
                                     use_safetensors=True, attn_implementation="eager").cpu().eval()

    class BertOutput(torch.nn.Module):
        def __init__(self, original):
            super().__init__()
            self.original = original

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.original(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids, return_dict=False)[0]

    samples = ["瓦斯超限后应核查现场记录。", "甲烷传感器应按要求调校。", "财务票据按月归档。"]
    encoded = tokenizer(samples, padding=True, truncation=True, max_length=512, return_tensors="pt")
    arguments = tuple(encoded[name] for name in ("input_ids", "attention_mask", "token_type_ids"))
    target = root / "model.onnx"
    partial = root / "model.onnx.part"
    wrapper = BertOutput(model).eval()
    try:
        with torch.no_grad():
            torch.onnx.export(wrapper, arguments, str(partial), input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["last_hidden_state"], dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"}, "token_type_ids": {0: "batch", 1: "sequence"},
                    "last_hidden_state": {0: "batch", 1: "sequence"}},
                opset_version=17, dynamo=False, external_data=False)
            expected = torch.nn.functional.normalize(wrapper(*arguments)[:, 0], p=2, dim=1).tolist()
        import onnx
        graph = onnx.load(str(partial), load_external_data=False)
        onnx.checker.check_model(graph)
        if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
            raise ValueError("ONNX导出包含外部权重文件")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    manifest["files"]["model.onnx"] = {"sha256": _sha(target), "bytes": target.stat().st_size,
        "derived_from": {"file": "model.safetensors", "sha256": manifest["files"]["model.safetensors"]["sha256"]}}
    manifest["export"] = {"created_at": datetime.now(timezone.utc).isoformat(), "opset": 17,
        "format": "onnx", "output": "last_hidden_state", "pooling": "CLS then L2 normalization",
        "providers": ["CPUExecutionProvider"], "external_data": False, "trust_remote_code": False,
        "local_files_only": True, "dependencies": _versions(("torch", "transformers", "onnx", "safetensors", "tokenizers")),
        "script_sha256": _sha(Path(__file__))}
    _save_manifest(root, manifest)
    if parity_path is not None:
        output = Path(parity_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        parity = {"repo_id": MODEL_REPO, "revision": manifest["revision"], "model_sha256": manifest["files"]["model.onnx"]["sha256"],
                  "texts": samples, "pooling": "CLS then L2 normalization", "vectors": expected,
                  "dependencies": manifest["export"]["dependencies"]}
        output.write_text(json.dumps(parity, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="下载固定原作者BGE权重或离线导出ONNX")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--parity-output", type=Path)
    args = parser.parse_args()
    try:
        manifest = (export_model(args.model_dir, parity_path=args.parity_output) if args.export_only
                    else download_model(args.model_dir, revision=args.revision))
    except (OSError, ValueError, httpx.HTTPError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "message": str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": "ready", "repo_id": manifest["repo_id"], "revision": manifest["revision"],
                      "files": len(manifest["files"]), "manifest": str(args.model_dir / "manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
