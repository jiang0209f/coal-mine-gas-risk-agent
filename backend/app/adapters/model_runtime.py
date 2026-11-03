"""Run trusted local model packages in a separate Python process."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

from app.adapters.model_bundle import ModelBundle, read_json

__all__ = ["ModelRuntime"]

_KINDS = ("gas_forecast", "gas_attribution")
_ROOT = Path(__file__).resolve().parents[3]


class ModelRuntime:
    def __init__(self, catalog_path: Path | None = None):
        self.python = Path(sys.executable)
        self.timeout = 120
        self.entries = {}
        if catalog_path is None:
            return
        path = Path(catalog_path).resolve()
        catalog = read_json(path)
        if not isinstance(catalog, dict) or not isinstance(catalog.get("models"), list):
            raise ValueError("本地模型目录格式无效")
        if "python_executable" in catalog:
            self.python = Path(catalog["python_executable"]).resolve()
        if "timeout_seconds" in catalog:
            timeout = catalog["timeout_seconds"]
            if type(timeout) is not int or not 1 <= timeout <= 300:
                raise ValueError("模型超时须为1至300秒")
            self.timeout = timeout
        for entry in catalog["models"]:
            if not isinstance(entry, dict):
                raise ValueError("本地模型目录条目无效")
            for field in ("bundle_dir", "source_id", "face_id"):
                if not isinstance(entry.get(field), str) or not entry[field].strip():
                    raise ValueError("本地模型目录缺少范围字段")
            directory = (path.parent / entry["bundle_dir"]).resolve()
            manifest = read_json(directory / "manifest.json")
            if not isinstance(manifest, dict) or manifest.get("kind") not in _KINDS:
                raise ValueError("本地模型种类无效")
            bundle = ModelBundle(directory, manifest["kind"])
            key = (bundle.manifest["model_id"], bundle.manifest["version"])
            if key in self.entries:
                raise ValueError("本地模型编号和版本重复")
            self.entries[key] = dict(bundle_dir=directory, manifest=bundle.manifest,
                                     source_id=entry["source_id"], face_id=entry["face_id"])

    def list_packages(self):
        result = []
        for entry in self.entries.values():
            manifest = entry["manifest"]
            public = {"model_id": manifest["model_id"], "model_version": manifest["version"],
                      "kind": manifest["kind"], "face_id": entry["face_id"], "source_id": entry["source_id"]}
            state = self._invoke(entry, inspect=True)
            public.update(status=state["status"], missing_fields=state.get("missing_fields", []))
            if state.get("error_code"):
                public["error_code"] = state["error_code"]
            result.append(public)
        return deepcopy(result)

    def infer(self, model_id, version, request):
        entry = self.entries.get((model_id, version))
        if entry is None:
            raise ValueError("本地模型版本不存在")
        # Paths and Python commands come exclusively from the local catalog.
        forbidden = {"bundle_dir", "legacy_source_root", "python_executable", "artifacts"}
        if not isinstance(request, dict) or forbidden.intersection(request):
            raise ValueError("推理输入包含非特征字段")
        return self._invoke(entry, request)

    def _invoke(self, entry, request=None, inspect=False):
        model_id, version = entry["manifest"]["model_id"], entry["manifest"]["version"]
        if not self.python.is_file():
            return dict(status="not_ready", model_id=model_id, model_version=version, missing_fields=["model_python"])
        environment = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP") if key in os.environ}
        environment.update(PYTHONPATH=str(_ROOT / "backend"), PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           MPLCONFIGDIR=str(self.python.parent.parent / "cache" / "matplotlib"))
        try:
            command = [str(self.python), "-m", "app.scripts.model_worker", "--kind", entry["manifest"]["kind"],
                       "--bundle", str(entry["bundle_dir"])]
            if inspect:
                command.append("--inspect")
            completed = subprocess.run(
                command,
                input=json.dumps(request, ensure_ascii=False, allow_nan=False), text=True, encoding="utf-8",
                capture_output=True, timeout=self.timeout, cwd=_ROOT, env=environment, check=False,
            )
            result = json.loads(completed.stdout)
            states = ("ready", "not_ready", "failed") if inspect else ("completed", "not_ready", "failed")
            if completed.returncode != 0 or not isinstance(result, dict) or result.get("status") not in states:
                raise ValueError("模型进程输出无效")
            if result.get("model_id", model_id) != model_id or result.get("model_version", version) != version:
                raise ValueError("模型进程身份不一致")
            if result.get("version", version) != version:
                raise ValueError("模型进程版本不一致")
            if result["status"] in ("ready", "completed") and (result.get("model_id") != model_id or result.get("model_version") != version):
                raise ValueError("模型进程缺少身份信息")
            return result
        except subprocess.TimeoutExpired:
            return dict(status="failed", model_id=model_id, model_version=version, error_code="model_timeout")
        except (OSError, ValueError):
            return dict(status="failed", model_id=model_id, model_version=version, error_code="model_worker_failed")
