"""Explicitly acquire one pinned official ONNX reranker; no runtime download."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import httpx

REPO = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
MODEL_FILE = "onnx/model_quint8_avx2.onnx"
MODEL_SHA = "6c2513767fb63d008a4377bef7a7a3555433d9436342bb53e35a3a72ffc52d4b"
MODEL_BYTES = 118620016
FILES = (MODEL_FILE, "tokenizer.json", "config.json", "tokenizer_config.json",
         "special_tokens_map.json", "README.md")
_HF = "https://huggingface.co"
_LICENSE_FILE = "LICENSE.apache-2.0.txt"
_LICENSE_URL = "https://www.apache.org/licenses/LICENSE-2.0.txt"


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _blob(path):
    digest = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _expected(entry):
    lfs = entry.get("lfs")
    size = lfs.get("size") if lfs else entry.get("size")
    digest = lfs.get("sha256") if lfs else entry.get("blobId")
    if (type(size) is not int or size <= 0 or not isinstance(digest, str)
            or not re.fullmatch(r"[a-f0-9]{64}" if lfs else r"[a-f0-9]{40}", digest)):
        raise ValueError("原作者文件缺少有效大小或摘要")
    return size, digest


def _valid(path, entry):
    size, digest = _expected(entry)
    return (path.is_file() and path.stat().st_size == size
            and (_sha(path) if entry.get("lfs") else _blob(path)) == digest)


def _save(path, data):
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)


def _inside(root, filename):
    target = (root / filename).resolve()
    if filename not in (*FILES, _LICENSE_FILE) or not target.is_relative_to(root):
        raise ValueError("模型清单文件路径无效")
    return target


def verify_manifest_identity(manifest):
    """Check pinned identity and allowlist; acquisition separately verifies bytes."""
    if (not isinstance(manifest, dict) or manifest.get("repo_id") != REPO
            or manifest.get("revision") != REVISION or manifest.get("model_file") != MODEL_FILE
            or manifest.get("license") != "apache-2.0"):
        raise ValueError("重排模型清单与固定官方版本不一致")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {*FILES, _LICENSE_FILE}:
        raise ValueError("重排模型清单文件缺失或包含非许可文件")
    for entry in files.values():
        if (not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
                or type(entry.get("bytes")) is not int or entry["bytes"] <= 0):
            raise ValueError("重排模型清单文件摘要或大小无效")
    if files[MODEL_FILE]["sha256"] != MODEL_SHA or files[MODEL_FILE]["bytes"] != MODEL_BYTES:
        raise ValueError("重排模型清单与官方AVX2模型摘要或大小不一致")


def download_model(model_dir, client=None):
    """Download only fixed-revision public assets, preserving verified originals."""
    root = Path(model_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if client is None:
        with httpx.Client(trust_env=False, follow_redirects=True, timeout=120) as owned:
            return download_model(root, client=owned)
    response = client.get(f"{_HF}/api/models/{REPO}/revision/{REVISION}", params={"blobs": "true"})
    response.raise_for_status()
    metadata = response.json()
    if (not isinstance(metadata, dict) or metadata.get("sha") != REVISION
            or not isinstance(metadata.get("cardData"), dict)
            or metadata["cardData"].get("license") != "apache-2.0"):
        raise ValueError("原作者返回的固定版本或许可证不一致")
    try:
        siblings = {entry["rfilename"]: entry for entry in metadata["siblings"]}
        for name in FILES:
            _expected(siblings[name])
        model_size, model_digest = _expected(siblings[MODEL_FILE])
        if not siblings[MODEL_FILE].get("lfs") or model_digest != MODEL_SHA or model_size != MODEL_BYTES:
            raise ValueError("model")
    except (KeyError, TypeError, AttributeError, ValueError):
        raise ValueError("原作者所需文件清单或固定模型摘要不一致") from None
    previous = root / "manifest.json"
    if previous.is_file():
        verify_manifest_identity(json.loads(previous.read_text(encoding="utf-8")))
    manifest = {"schema_version": "local-text-reranker-v1", "repo_id": REPO,
                "revision": REVISION, "model_file": MODEL_FILE, "license": "apache-2.0",
                "quantization": "official_quint8_avx2", "files": {},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_url": f"{_HF}/{REPO}/tree/{REVISION}", "trust_remote_code": False}
    for name in FILES:
        target = _inside(root, name)
        entry = siblings[name]
        url = f"{_HF}/{REPO}/resolve/{REVISION}/{name}"
        if target.exists() and not _valid(target, entry):
            raise ValueError("已有模型文件校验失败：" + name)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".part")
            expected_size, _ = _expected(entry)
            try:
                written = 0
                with client.stream("GET", url) as result:
                    result.raise_for_status()
                    with partial.open("wb") as output:
                        for block in result.iter_bytes(1024 * 1024):
                            written += len(block)
                            if written > expected_size:
                                raise ValueError("下载文件大小校验失败：" + name)
                            output.write(block)
                if not _valid(partial, entry):
                    raise ValueError("下载文件摘要校验失败：" + name)
                partial.replace(target)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        manifest["files"][name] = {"sha256": _sha(target), "bytes": target.stat().st_size,
            "source_url": url, "verification": "lfs_sha256" if entry.get("lfs") else "git_blob_sha1",
            "source_digest": _expected(entry)[1]}
    license_response = client.get(_LICENSE_URL)
    license_response.raise_for_status()
    license_data = license_response.content
    if b"Apache License" not in license_data or b"Version 2.0, January 2004" not in license_data:
        raise ValueError("Apache官方许可证正文无效")
    license_path = _inside(root, _LICENSE_FILE)
    license_path.write_bytes(license_data)
    manifest["files"][_LICENSE_FILE] = {"sha256": _sha(license_path), "bytes": len(license_data),
        "source_url": _LICENSE_URL, "verification": "canonical_https_sha256"}
    manifest["license_declaration"] = {"location": "fixed revision README.md YAML license: apache-2.0",
        "repo_license_file_present": "LICENSE" in siblings, "canonical_text": _LICENSE_FILE}
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if (not isinstance(config, dict) or config.get("architectures") != ["XLMRobertaForSequenceClassification"]
            or config.get("auto_map")):
        raise ValueError("模型结构与官方本机重排模型不一致")
    manifest["architecture"] = config["architectures"][0]
    manifest["download_bytes"] = sum(entry["bytes"] for entry in manifest["files"].values())
    manifest["script_sha256"] = _sha(Path(__file__))
    verify_manifest_identity(manifest)
    _save(root / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="显式下载固定官方MiniLM量化重排模型及校验清单")
    parser.add_argument("--model-dir", required=True, help="模型资产保存目录（必须显式指定）")
    parser.add_argument("--report-dir", type=Path, help="可选校验报告目录")
    args = parser.parse_args(argv)
    if not args.model_dir.strip():
        parser.error("--model-dir不能为空")
    manifest = download_model(Path(args.model_dir))
    if args.report_dir is not None:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        _save(args.report_dir / "download-verification.json", manifest)
    print(json.dumps({"status": "verified", "repo_id": REPO, "revision": REVISION,
                      "file_count": len(manifest["files"]), "download_bytes": manifest["download_bytes"]}))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
