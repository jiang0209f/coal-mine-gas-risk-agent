"""Read local model manifests and verify artifacts before deserialization."""

import hashlib
import json
from pathlib import Path, PureWindowsPath
import re

__all__ = ["ModelBundle", "ModelNotReady", "read_json"]


class ModelNotReady(Exception):
    def __init__(self, missing_fields: list[str]):
        self.missing_fields = list(dict.fromkeys(missing_fields))
        super().__init__("模型运行条件未齐备")


def read_json(path: Path):
    def invalid_constant(value):
        raise ValueError("JSON中不允许非有限数值")
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("模型JSON格式无效") from error


class ModelBundle:
    def __init__(self, path: Path | str, expected_kind: str):
        self.root = Path(path).resolve()
        manifest = self.root / "manifest.json"
        if not manifest.is_file():
            raise ModelNotReady(["manifest.json"])
        self.manifest = read_json(manifest)
        if not isinstance(self.manifest, dict):
            raise ValueError("模型清单须为对象")
        for field in ("kind", "model_id", "version"):
            value = self.manifest.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("模型清单身份字段无效")
        if self.manifest["kind"] != expected_kind:
            raise ValueError("模型种类与运行入口不一致")

    def require_artifacts(self, names) -> dict[str, Path]:
        descriptors = self.manifest.get("artifacts")
        if not isinstance(descriptors, dict):
            raise ValueError("模型产物声明须为对象")
        result, missing = {}, []
        for name in names:
            if name not in descriptors:
                missing.append(name)
                continue
            descriptor = descriptors[name]
            if not isinstance(descriptor, dict):
                raise ValueError("模型产物声明无效")
            relative, digest = descriptor.get("path"), descriptor.get("sha256")
            if not isinstance(relative, str) or not relative.strip():
                raise ValueError("模型产物路径无效")
            windows = PureWindowsPath(relative)
            if Path(relative).is_absolute() or windows.is_absolute() or windows.drive or ".." in windows.parts:
                raise ValueError("模型产物须位于当前目录内")
            path = (self.root / relative).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("模型产物路径超出当前目录")
            if path.suffix.lower() not in (".json", ".skops", ".pt", ".pth", ".txt", ".npz"):
                raise ValueError("模型产物格式不受支持")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                raise ValueError("模型产物SHA-256无效")
            if not path.is_file():
                missing.append(name)
                continue
            sha = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    sha.update(chunk)
            if sha.hexdigest() != digest.lower():
                raise ValueError("模型产物SHA-256不一致")
            result[name] = path
        if missing:
            raise ModelNotReady(missing)
        return result
