"""冻结导出记录、一对一事件匹配和显式分母的离线评估。"""

from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import isfinite
from uuid import uuid4

from app.db import Store
from app.schemas.common import Actor, DomainError, now_utc, utcstr

__all__ = ["EvaluationService"]


def _canonical(value: dict) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise DomainError("冻结记录必须为有限数值的JSON数据") from error


def _digest(value: dict) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _time(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        utcstr(result)
        return result
    except (AttributeError, TypeError, ValueError) as error:
        raise DomainError("评估事件时间必须包含时区") from error


def _number(value, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise DomainError(f"{name}必须为有限数值")
    if minimum is not None and value < minimum:
        raise DomainError(f"{name}不能小于{minimum}")
    return float(value)


class EvaluationService:
    def __init__(self, store: Store) -> None:
        self.store = store

    @staticmethod
    def _authorize(actor: Actor, face_id: str) -> None:
        actor.require_role("safety_management")
        actor.require_face(face_id)

    @staticmethod
    def _validate_records(records: dict, face_id: str) -> None:
        if not isinstance(records, dict):
            raise DomainError("冻结记录必须为对象")
        required = ("actual_events", "alert_events", "samples", "tasks", "valid_monitoring_hours")
        missing = [key for key in required if key not in records]
        if missing:
            raise DomainError("冻结记录缺少字段", missing_fields=missing)
        _number(records["valid_monitoring_hours"], "有效监测小时数", 0)
        if records.get("valid_monitoring_hours_excluding_suspected") is not None:
            filtered_hours = _number(records["valid_monitoring_hours_excluding_suspected"], "剔除疑似标校后的有效监测小时数", 0)
            if filtered_hours > records["valid_monitoring_hours"]:
                raise DomainError("剔除疑似标校后的时长不能超过全部有效监测时长")
        for group, id_field in (("actual_events", "event_id"), ("alert_events", "event_id"),
                                ("samples", "sample_id"), ("tasks", "task_id")):
            rows = records[group]
            if not isinstance(rows, list):
                raise DomainError(f"{group}必须为记录数组")
            identifiers = set()
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get(id_field), str) or not row[id_field]:
                    raise DomainError(f"{group}记录缺少唯一编号")
                if row[id_field] in identifiers:
                    raise DomainError(f"{group}包含重复编号")
                identifiers.add(row[id_field])
                if row.get("face_id", face_id) != face_id:
                    raise DomainError("冻结记录工作面与清单不一致")
                flags = row.get("quality_flags", [])
                if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
                    raise DomainError("记录质量标记必须为字符串数组")
                if group in ("actual_events", "alert_events"):
                    if not isinstance(row.get("point_id"), str) or not row["point_id"]:
                        raise DomainError("评估事件缺少点位编号")
                    if "start_at" not in row or "end_at" not in row:
                        raise DomainError("评估事件缺少起止时间")
                    if _time(row["end_at"]) < _time(row["start_at"]):
                        raise DomainError("评估事件结束时间早于开始时间")
                elif group == "samples":
                    if type(row.get("valid", True)) is not bool:
                        raise DomainError("样本有效性必须为布尔值")
                    if type(row.get("actual_positive")) is not bool or type(row.get("predicted_positive")) is not bool:
                        raise DomainError("样本分类需同时提供真实与预测布尔值")
                    for key in ("actual_value", "predicted_value"):
                        if row.get(key) is not None:
                            _number(row[key], key)
                    if (row.get("actual_value") is None) != (row.get("predicted_value") is None):
                        raise DomainError("回归样本需同时提供真实值与预测值")
                else:
                    if row.get("state") not in ("awaiting_input", "running", "completed", "partial", "failed", "cancelled"):
                        raise DomainError("评估任务状态无效")
                    for key in ("normal_task", "evidence_complete"):
                        if type(row.get(key, key == "normal_task")) is not bool:
                            raise DomainError(f"任务{key}必须为布尔值")
                    for key in ("advice_items", "referenced_advice_items"):
                        if type(row.get(key, 0)) is not int or row.get(key, 0) < 0:
                            raise DomainError("建议条数必须为非负整数")
                    if row.get("referenced_advice_items", 0) > row.get("advice_items", 0):
                        raise DomainError("有出处的建议条数超过总条数")

    def freeze_manifest(
        self, actor: Actor, evaluation_spec: dict, artifact_refs: dict | None = None,
    ) -> dict:
        if not isinstance(evaluation_spec, dict):
            raise DomainError("评估清单必须为对象")
        required = (
            "face_id", "dataset_refs", "dataset_split", "model_version", "rule_version",
            "preprocess_version", "matching_config", "quality_policy", "task_set_ref",
        )
        missing = [key for key in required if key not in evaluation_spec]
        if missing:
            raise DomainError("评估清单缺少字段", missing_fields=missing)
        self._authorize(actor, evaluation_spec["face_id"])
        value = deepcopy(evaluation_spec)
        for key in ("face_id", "dataset_split", "model_version", "rule_version", "preprocess_version", "task_set_ref"):
            if not isinstance(value[key], str) or not value[key]:
                raise DomainError(f"评估字段{key}必须为非空字符串")
        if not isinstance(value["dataset_refs"], list) or not value["dataset_refs"] or any(not isinstance(ref, str) or not ref for ref in value["dataset_refs"]):
            raise DomainError("评估必须指定数据集引用")
        if not isinstance(value["matching_config"], dict) or not isinstance(value["quality_policy"], dict):
            raise DomainError("匹配配置与质量策略必须为对象")
        tolerance = value["matching_config"].get("tolerance_seconds", 0)
        _number(tolerance, "事件匹配容差", 0)
        value["matching_config"]["tolerance_seconds"] = tolerance
        policy = value["quality_policy"]
        if policy.get("mape_zero_policy") != "exclude_zero_actuals":
            raise DomainError("MAPE零值策略需明确为exclude_zero_actuals")
        if type(policy.get("exclude_invalid_samples", True)) is not bool:
            raise DomainError("无效样本排除配置必须为布尔值")
        artifacts = artifact_refs or {}
        if not isinstance(artifacts, dict):
            raise DomainError("评估导出引用必须为对象")
        frozen = artifacts.get("frozen_records", value.pop("frozen_records", None))
        self._validate_records(frozen, value["face_id"])
        value.update({
            "manifest_id": str(uuid4()), "records_sha256": _digest(frozen),
            "frozen_records": deepcopy(frozen), "frozen_by": actor.user_id,
            "created_at": utcstr(now_utc()),
            "frozen_record_refs": deepcopy(artifacts.get("frozen_record_refs", [])),
        })
        if not value["frozen_record_refs"]:
            value["frozen_record_refs"] = [f"sha256:{value['records_sha256']}"]
        self.store.put("evaluation_manifest", value["manifest_id"], value)
        return deepcopy(value)

    @staticmethod
    def _match_events(actual: list[dict], alerts: list[dict], tolerance_seconds: float) -> list[dict]:
        actual = sorted(actual, key=lambda item: (item["point_id"], _time(item["start_at"]), item["event_id"]))
        alerts = sorted(alerts, key=lambda item: (item["point_id"], _time(item["start_at"]), item["event_id"]))
        tolerance = timedelta(seconds=tolerance_seconds)
        adjacency = []
        for alert in alerts:
            compatible = []
            for index, observed in enumerate(actual):
                if (observed["point_id"] == alert["point_id"]
                        and _time(alert["start_at"]) <= _time(observed["end_at"]) + tolerance
                        and _time(alert["end_at"]) >= _time(observed["start_at"]) - tolerance):
                    compatible.append(index)
            adjacency.append(compatible)
        matched_actual: dict[int, int] = {}
        for root in range(len(alerts)):
            queue = deque([root])
            parent_alert: dict[int, int | None] = {root: None}
            parent_actual: dict[int, int] = {}
            unmatched = None
            while queue and unmatched is None:
                alert_index = queue.popleft()
                for actual_index in adjacency[alert_index]:
                    if actual_index in parent_actual:
                        continue
                    parent_actual[actual_index] = alert_index
                    if actual_index not in matched_actual:
                        unmatched = actual_index
                        break
                    next_alert = matched_actual[actual_index]
                    if next_alert not in parent_alert:
                        parent_alert[next_alert] = actual_index
                        queue.append(next_alert)
            while unmatched is not None:
                alert_index = parent_actual[unmatched]
                previous_actual = parent_alert[alert_index]
                matched_actual[unmatched] = alert_index
                unmatched = previous_actual
        return [
            {"actual_event_id": actual[actual_index]["event_id"],
             "alert_event_id": alerts[alert_index]["event_id"]}
            for actual_index, alert_index in sorted(matched_actual.items())
        ]

    def _summarize(self, manifest: dict, records: dict) -> dict:
        matches = self._match_events(records["actual_events"], records["alert_events"],
                                     manifest["matching_config"]["tolerance_seconds"])
        true_alerts = len(matches)
        false_alerts = len(records["alert_events"]) - true_alerts
        missed = len(records["actual_events"]) - true_alerts
        metrics, numerators, denominators = {}, {}, {}

        def ratio(name: str, numerator: float, denominator: float):
            numerators[name] = numerator
            denominators[name] = denominator
            metrics[name] = numerator / denominator if denominator else None

        ratio("precision", true_alerts, len(records["alert_events"]))
        ratio("recall", true_alerts, len(records["actual_events"]))
        metrics["missed_events"] = missed
        numerators["missed_events"] = missed
        denominators["missed_events"] = 1
        ratio("false_alerts_per_valid_monitoring_hour", false_alerts, records["valid_monitoring_hours"])
        samples = [sample for sample in records["samples"] if sample.get("valid", True)
                   or not manifest["quality_policy"].get("exclude_invalid_samples", True)]
        tp = sum(sample["actual_positive"] and sample["predicted_positive"] for sample in samples)
        fp = sum(not sample["actual_positive"] and sample["predicted_positive"] for sample in samples)
        tn = sum(not sample["actual_positive"] and not sample["predicted_positive"] for sample in samples)
        fn = sum(sample["actual_positive"] and not sample["predicted_positive"] for sample in samples)
        ratio("sample_precision", tp, tp + fp)
        ratio("sample_recall", tp, tp + fn)
        ratio("sample_fpr", fp, fp + tn)
        regression = [sample for sample in samples if sample.get("actual_value") is not None]
        absolute_error = sum(abs(sample["predicted_value"] - sample["actual_value"]) for sample in regression)
        ratio("mae", absolute_error, len(regression))
        nonzero = [sample for sample in regression if sample["actual_value"] != 0]
        percentage_error = sum(abs((sample["predicted_value"] - sample["actual_value"]) / sample["actual_value"]) * 100
                               for sample in nonzero)
        ratio("mape_with_zero_policy", percentage_error, len(nonzero))
        tasks = [task for task in records["tasks"] if task.get("normal_task", True)]
        ratio("tool_task_completion_rate", sum(task["state"] == "completed" for task in tasks), len(tasks))
        ratio("evidence_completeness", sum(task.get("evidence_complete", False) for task in tasks), len(tasks))
        ratio("advice_reference_coverage", sum(task.get("referenced_advice_items", 0) for task in tasks),
              sum(task.get("advice_items", 0) for task in tasks))
        return {
            "metric_values": metrics, "numerators": numerators, "denominators": denominators,
            "sample_refs": {key: [row[id_key] for row in records[key]] for key, id_key in (
                ("actual_events", "event_id"), ("alert_events", "event_id"),
                ("samples", "sample_id"), ("tasks", "task_id"),
            )},
            "event_matches": matches,
        }

    def build_report(self, manifest_id: str, frozen_records: dict | None = None) -> dict:
        manifest = self.store.get("evaluation_manifest", manifest_id)
        if manifest is None:
            raise DomainError("评估清单不存在", code="not_found", status_code=404)
        records = manifest["frozen_records"]
        if _digest(records) != manifest["records_sha256"]:
            raise DomainError("冻结记录校验失败", code="conflict", status_code=409)
        if frozen_records is not None and _digest(frozen_records) != manifest["records_sha256"]:
            raise DomainError("不能替换已冻结的评估记录", code="conflict", status_code=409)
        primary = self._summarize(manifest, records)
        filtered = deepcopy(records)
        excluded = {}
        for key, identifier in (("actual_events", "event_id"), ("alert_events", "event_id"), ("samples", "sample_id")):
            excluded[key] = [row[identifier] for row in records[key] if "suspected_calibration" in row.get("quality_flags", [])]
            filtered[key] = [row for row in records[key] if "suspected_calibration" not in row.get("quality_flags", [])]
        excluded_any = any(excluded.values())
        filtered_hours = records.get("valid_monitoring_hours_excluding_suspected")
        if filtered_hours is None and not excluded_any:
            filtered_hours = records["valid_monitoring_hours"]
        filtered["valid_monitoring_hours"] = filtered_hours
        sensitivity = self._summarize(manifest, filtered)
        sensitivity["excluded_refs"] = excluded
        sensitivity["missing_fields"] = ["valid_monitoring_hours_excluding_suspected"] if filtered_hours is None else []
        primary["excluded_refs"] = {key: [] for key in excluded}
        primary["missing_fields"] = []
        report = {
            **deepcopy(primary), "evaluation_id": str(uuid4()), "manifest_id": manifest_id,
            "face_id": manifest["face_id"], "records_sha256": manifest["records_sha256"],
            "mape_zero_policy": manifest["quality_policy"]["mape_zero_policy"],
            "primary_calibration_view": "all_events",
            "calibration_views": {"all_events": primary, "excluding_suspected": sensitivity},
            "generated_at": utcstr(now_utc()),
        }
        self.store.put("evaluation_report", report["evaluation_id"], report)
        return deepcopy(report)

    def run(self, actor: Actor, spec: dict, records: dict) -> dict:
        manifest = self.freeze_manifest(actor, spec, {"frozen_records": records})
        return self.build_report(manifest["manifest_id"])

    def get_report(self, actor: Actor, evaluation_id: str) -> dict:
        report = self.store.get("evaluation_report", evaluation_id)
        if report is None:
            raise DomainError("评估报告不存在", code="not_found", status_code=404)
        self._authorize(actor, report["face_id"])
        return report
