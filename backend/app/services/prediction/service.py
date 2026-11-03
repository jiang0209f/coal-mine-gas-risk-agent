"""模型登记、闭合输入窗口和可追溯预测；内置持久性离线基线。"""

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from math import floor, isfinite
from uuid import uuid4

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc, utcstr

__all__ = ["PredictionService"]

_ROLES = ("ventilation", "gas_governance", "safety_management")


def _datetime(value: str | datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        utcstr(parsed)
        return parsed.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as error:
        raise DomainError("时间格式无效") from error


class PredictionService:
    def __init__(self, store: Store, data_service) -> None:
        self.store = store
        self.data = data_service
        self._adapters: dict[str, Callable] = {"persistence": self._persistence}

    def register_adapter(self, adapter_kind: str, predict: Callable) -> None:
        if not isinstance(adapter_kind, str) or not adapter_kind.strip() or not callable(predict):
            raise DomainError("模型适配器需要名称与预测函数")
        if adapter_kind in self._adapters:
            raise DomainError("模型适配器已登记", code="conflict", status_code=409)
        self._adapters[adapter_kind] = predict

    def _source(self, context: ExecutionContext) -> dict:
        context.actor.require_role(*_ROLES)
        source = self.store.get("source", context.source_id)
        if source is None:
            raise DomainError("数据源不存在", code="not_found", status_code=404)
        context.actor.require_face(source["face_id"])
        return source

    def _point(self, context: ExecutionContext, point_id: str) -> dict:
        source = self._source(context)
        point = self.store.get("point", point_id)
        if point is None:
            raise DomainError("点位不存在", code="not_found", status_code=404)
        if point["source_id"] != context.source_id:
            raise DomainError("点位与会话数据源不一致", code="forbidden", status_code=403)
        context.actor.require_face(point.get("face_id", source["face_id"]))
        return point

    def register_model(self, actor: Actor, manifest: dict) -> dict:
        actor.require_role("safety_management")
        if not isinstance(manifest, dict):
            raise DomainError("模型清单必须为对象")
        value = deepcopy(manifest)
        required = (
            "model_id", "version", "adapter_kind", "face_id", "source_id", "point_ids",
            "target_quantity", "unit", "aggregation_seconds", "input_window", "horizon_seconds",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise DomainError("模型清单缺少字段", missing_fields=missing)
        for key in ("model_id", "version", "adapter_kind", "face_id", "source_id", "target_quantity", "unit"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise DomainError(f"模型字段 {key} 必须为非空字符串")
        for key in ("aggregation_seconds", "input_window", "horizon_seconds"):
            if type(value[key]) is not int or value[key] <= 0:
                raise DomainError(f"模型字段 {key} 必须为正整数")
        if value["input_window"] > 10000:
            raise DomainError("模型输入窗口超过10000个桶")
        if value["horizon_seconds"] % value["aggregation_seconds"]:
            raise DomainError("预测时长必须为聚合间隔的整数倍")
        points = value["point_ids"]
        if not isinstance(points, list) or not points or any(not isinstance(item, str) or not item for item in points):
            raise DomainError("模型必须指定适用点位")
        lineage = value.setdefault("allowed_lineage", ["raw"])
        if not isinstance(lineage, list) or not lineage or any(item not in ("raw", "unknown", "processed") for item in lineage):
            raise DomainError("模型输入来源类型无效")
        source = self.store.get("source", value["source_id"])
        if source is None:
            raise DomainError("数据源不存在", code="not_found", status_code=404)
        actor.require_face(source["face_id"])
        if source["face_id"] != value["face_id"]:
            raise DomainError("模型工作面与数据源不一致")
        for point_id in points:
            point = self.store.get("point", point_id)
            if point is None or point["source_id"] != value["source_id"]:
                raise DomainError("模型点位与数据源不一致")
            if point.get("quantity") != value["target_quantity"] or point.get("unit") != value["unit"]:
                raise DomainError("模型目标与点位量纲不一致")
        if value["adapter_kind"] == "persistence":
            if value.get("model_type", "baseline") != "baseline":
                raise DomainError("持久性模型必须标记为离线基线")
            value["model_type"] = "baseline"
            value["purpose"] = "offline_pipeline_verification"
        else:
            if value.get("model_type", "professional") != "professional":
                raise DomainError("专业适配器模型类型无效")
            value["model_type"] = "professional"
        value["created_at"] = utcstr(now_utc())
        key = f"{value['model_id']}:{value['version']}"
        with self.store.transaction() as connection:
            if self.store.get("model", key, connection) is not None:
                raise DomainError("模型版本已登记", code="conflict", status_code=409)
            self.store.put("model", key, value, connection)
        return deepcopy(value)

    def list_models(self, context: ExecutionContext, point_id: str | None = None) -> list[dict]:
        source = self._source(context)
        if point_id is not None:
            self._point(context, point_id)
        return [model for model in self.store.list("model") if
                model["source_id"] == context.source_id and model["face_id"] == source["face_id"]
                and (point_id is None or point_id in model["point_ids"])]

    def get_model(self, context: ExecutionContext, model_id: str, version: str | None = None) -> dict:
        self._source(context)
        candidates = [item for item in self.store.list("model") if item["model_id"] == model_id
                      and (version is None or item["version"] == version)]
        if not candidates:
            raise DomainError("模型不存在", code="not_found", status_code=404)
        allowed = [item for item in candidates if item["source_id"] == context.source_id
                   and item["face_id"] in context.actor.face_ids]
        if not allowed:
            raise DomainError("模型与会话范围不一致", code="forbidden", status_code=403)
        return max(allowed, key=lambda item: item["created_at"])

    def _save(self, run: dict) -> dict:
        self.store.put("forecast_run", run["run_id"], run)
        return deepcopy(run)

    def run_prediction(
        self, context: ExecutionContext, point_id: str,
        model_id: str | None = None, horizon_seconds: int | None = None,
    ) -> dict:
        point = self._point(context, point_id)
        source = self._source(context)
        run = {
            "run_id": str(uuid4()), "session_id": context.session_id, "source_id": context.source_id,
            "face_id": source["face_id"], "point_id": point_id, "as_of": utcstr(context.as_of),
            "cutoff": utcstr(context.as_of), "config_version": context.config_version,
            "model_id": model_id, "model_version": None, "target_quantity": point.get("quantity"),
            "status": "not_ready", "missing_fields": [], "input_refs": [], "input_window": None,
            "outputs": [], "created_at": utcstr(now_utc()),
        }
        if model_id is not None:
            try:
                model = self.get_model(context, model_id)
            except DomainError as error:
                if error.status_code != 404:
                    raise
                run["missing_fields"] = ["model_id"]
                return self._save(run)
            if point_id not in model["point_ids"]:
                raise DomainError("模型不适用于该点位", code="forbidden", status_code=403)
        else:
            candidates = self.list_models(context, point_id)
            if not candidates:
                run["missing_fields"] = ["model_id"]
                return self._save(run)
            model = max(candidates, key=lambda item: item["created_at"])
        step = model["aggregation_seconds"]
        horizon = model["horizon_seconds"] if horizon_seconds is None else horizon_seconds
        if type(horizon) is not int or horizon <= 0 or horizon % step or horizon > model["horizon_seconds"]:
            raise DomainError("预测时长须在模型范围内且为聚合间隔的正整数倍")
        cutoff = datetime.fromtimestamp(floor(context.as_of.timestamp() / step) * step, timezone.utc)
        start = cutoff - timedelta(seconds=model["input_window"] * step)
        run.update({
            "model_id": model["model_id"], "model_version": model["version"],
            "model_type": model["model_type"], "aggregation": {"seconds": step},
            "cutoff": utcstr(cutoff), "horizon_seconds": horizon,
            "input_window": {"start_at": utcstr(start), "end_at": utcstr(cutoff),
                             "aggregation_seconds": step, "count": model["input_window"]},
        })
        adapter = self._adapters.get(model["adapter_kind"])
        if adapter is None:
            run["missing_fields"] = ["model_adapter"]
            return self._save(run)
        if point.get("mapping_status") != "confirmed":
            run["missing_fields"] = ["confirmed_mapping"]
            return self._save(run)
        if point.get("lineage", "unknown") not in model["allowed_lineage"]:
            run["missing_fields"] = ["allowed_lineage"]
            return self._save(run)
        result = self.data.query_series(
            context=context, point_id=point_id, start_at=start, end_at=cutoff,
            aggregation_seconds=step, fill_max_seconds=0, limit=model["input_window"] + 1, offset=0,
        )
        rows = result["rows"]
        if not self._valid_window(rows, model, start, context.as_of):
            run["missing_fields"] = ["valid_closed_input_window"]
            return self._save(run)
        run["input_refs"] = deepcopy(rows)
        try:
            outputs = adapter(deepcopy(model), deepcopy(rows), horizon)
            self._validate_outputs(outputs, model, cutoff, horizon)
        except Exception:
            run["status"] = "failed"
            run["error_code"] = "model_adapter_failed"
            return self._save(run)
        run["outputs"] = deepcopy(outputs)
        run["status"] = "completed"
        return self._save(run)

    @staticmethod
    def _valid_window(rows: list, model: dict, start: datetime, as_of: datetime) -> bool:
        if len(rows) != model["input_window"]:
            return False
        step = timedelta(seconds=model["aggregation_seconds"])
        for index, row in enumerate(rows):
            try:
                expected = start + index * step
                if _datetime(row["event_time"]) != expected or _datetime(row["bucket_end"]) != expected + step:
                    return False
                if _datetime(row["bucket_end"]) > as_of:
                    return False
                if row.get("available_at") and _datetime(row["available_at"]) > as_of:
                    return False
                value = row.get("value")
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not isfinite(value) or row.get("count", 0) <= 0:
                    return False
                flags = row.get("quality_flags", [])
                if any(flag in flags for flag in ("invalid_value", "duplicate_timestamp", "forward_filled")):
                    return False
                for lineage in ("unknown", "processed"):
                    if f"lineage_{lineage}" in flags and lineage not in model["allowed_lineage"]:
                        return False
            except (DomainError, KeyError, TypeError, ValueError):
                return False
        return True

    @staticmethod
    def _persistence(model: dict, rows: list[dict], horizon_seconds: int) -> list[dict]:
        cutoff = _datetime(rows[-1]["bucket_end"])
        step = model["aggregation_seconds"]
        return [{
            "prediction_stage": "baseline_persistence", "quantity_semantics": "physical_forecast",
            "unit": model["unit"], "interval_values": [
                {"start_at": utcstr(cutoff + timedelta(seconds=index * step)),
                 "end_at": utcstr(cutoff + timedelta(seconds=(index + 1) * step)),
                 "value": rows[-1]["value"]}
                for index in range(horizon_seconds // step)
            ],
        }]

    @staticmethod
    def _validate_outputs(outputs: list[dict], model: dict, cutoff: datetime, horizon: int) -> None:
        if not isinstance(outputs, list) or not outputs:
            raise DomainError("模型输出为空")
        step = model["aggregation_seconds"]
        physical = False
        for output in outputs:
            semantics = output["quantity_semantics"]
            if semantics not in ("physical_forecast", "decision_score"):
                raise DomainError("预测量语义无效")
            if semantics == "physical_forecast":
                physical = True
                if output["unit"] != model["unit"]:
                    raise DomainError("物理预测单位与模型不一致")
            elif output["unit"] not in ("1", "dimensionless"):
                raise DomainError("决策分值必须使用无量纲单位")
            if not isinstance(output.get("prediction_stage"), str) or not output["prediction_stage"]:
                raise DomainError("预测阶段缺失")
            values = output["interval_values"]
            if not isinstance(values, list) or len(values) != horizon // step:
                raise DomainError("预测区间数量与时长不一致")
            for index, item in enumerate(values):
                expected = cutoff + timedelta(seconds=index * step)
                if _datetime(item["start_at"]) != expected or _datetime(item["end_at"]) != expected + timedelta(seconds=step):
                    raise DomainError("模型输出区间不连续")
                number = item["value"]
                if isinstance(number, bool) or not isinstance(number, (int, float)) or not isfinite(number):
                    raise DomainError("模型输出含无效数值")
        if not physical:
            raise DomainError("模型输出缺少物理预测")

    def get_forecast(self, context: ExecutionContext, run_id: str) -> dict:
        self._source(context)
        run = self.store.get("forecast_run", run_id)
        if run is None:
            raise DomainError("预测运行不存在", code="not_found", status_code=404)
        context.actor.require_face(run["face_id"])
        if run["session_id"] != context.session_id or run["source_id"] != context.source_id or _datetime(run["as_of"]) > context.as_of:
            raise DomainError("预测运行与会话时钟或来源不一致", code="forbidden", status_code=403)
        return run
