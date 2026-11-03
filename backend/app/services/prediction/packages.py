"""Authenticated execution of prepared features through local model packages."""

from copy import deepcopy
from datetime import datetime
import hashlib
import json
from uuid import uuid4

from app.schemas.common import DomainError, now_utc, utcstr

__all__ = ["ModelPackagesService"]


class ModelPackagesService:
    def __init__(self, store, runtime):
        self.store, self.runtime = store, runtime

    def _source(self, context):
        context.actor.require_role("ventilation", "gas_governance", "safety_management")
        source = self.store.get("source", context.source_id)
        if source is None:
            raise DomainError("数据源不存在", "not_found", 404)
        context.actor.require_face(source["face_id"])
        return source

    def list_packages(self, context):
        source = self._source(context)
        return [p for p in self.runtime.list_packages()
                if p["source_id"] == context.source_id and p["face_id"] == source["face_id"]]

    def run(self, context, model_id, features, version=None):
        self._source(context)
        catalog = self.runtime.describe_packages() if hasattr(self.runtime, "describe_packages") else self.runtime.list_packages()
        candidates = [p for p in catalog if p["model_id"] == model_id
                      and (version is None or p["model_version"] == version)]
        if not candidates:
            raise DomainError("模型封装版本不存在", "not_found", 404)
        source = self._source(context)
        allowed = [p for p in candidates if p["source_id"] == context.source_id and p["face_id"] == source["face_id"]]
        if not allowed:
            raise DomainError("模型与会话来源不一致", "forbidden", 403)
        if len(allowed) != 1:
            raise DomainError("需要明确模型版本")
        package = allowed[0]
        if not isinstance(features, dict):
            raise DomainError("模型特征输入须为对象")
        request = deepcopy(features)
        if "as_of" in request and request["as_of"] != utcstr(context.as_of):
            try:
                matches = datetime.fromisoformat(request["as_of"]) == context.as_of
            except (TypeError, ValueError):
                matches = False
            if not matches:
                raise DomainError("模型输入时钟与会话不一致")
        request["as_of"] = utcstr(context.as_of)
        try:
            for value in request.get("times", []):
                timestamp = datetime.fromisoformat(value)
                if timestamp.utcoffset() is None or timestamp > context.as_of:
                    raise ValueError("输入时刻超过会话时钟")
            encoded = json.dumps(request, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")
            if len(encoded) > 8 * 1024 * 1024:
                raise ValueError("输入超过限制")
        except (TypeError, ValueError) as error:
            raise DomainError("模型输入时刻或数值格式无效") from error
        try:
            result = self.runtime.infer(model_id, package["model_version"], request)
        except ValueError as error:
            raise DomainError("模型输入字段无效") from error
        run = dict(result, run_id=str(uuid4()), session_id=context.session_id, source_id=context.source_id,
                   face_id=package["face_id"], kind=package["kind"], model_id=model_id,
                   model_version=package["model_version"], as_of=utcstr(context.as_of),
                   input_origin="submitted_features", input_sha256=hashlib.sha256(encoded).hexdigest(),
                   input_feature_names=request.get("feature_names", []), created_at=utcstr(now_utc()))
        self.store.put("model_package_run", run["run_id"], run)
        return deepcopy(run)

    def get_run(self, context, run_id):
        self._source(context)
        run = self.store.get("model_package_run", run_id)
        if run is None:
            raise DomainError("模型运行不存在", "not_found", 404)
        if run["source_id"] != context.source_id or run["session_id"] != context.session_id or datetime.fromisoformat(run["as_of"]) > context.as_of:
            raise DomainError("模型运行与会话不一致", "forbidden", 403)
        context.actor.require_face(run["face_id"])
        return deepcopy(run)
