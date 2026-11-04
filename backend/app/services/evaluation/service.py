"""冻结导出记录、一对一事件匹配和显式分母的离线评估。"""

from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import isfinite
import re
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


def _public(value):
    """Project a local export while preserving the original frozen hashes separately."""
    private = {"messages", "context", "question", "answer", "final_text", "prompt", "system_prompt",
               "password", "password_hash", "api_key", "deepseek_api_key", "authorization",
               "token", "access_token", "refresh_token", "file_ref", "path", "text", "content"}
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items()
                if key.lower() not in private and not key.lower().endswith("_path")}
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)(?:\b[A-Z]:[\\/]|\\\\)[^\s\"<>]*", "[本地路径]", value)
        value = re.sub(r"(?<![:\w])/(?:home|mnt|tmp|var|Users|srv|opt)/[^\s\"<>]*", "[本地路径]", value)
        return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[密钥]", value)
    return deepcopy(value)


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
        missing_fields = records.get("missing_fields", [])
        if not isinstance(missing_fields, list) or any(not isinstance(field, str) for field in missing_fields):
            raise DomainError("评估缺项必须为字符串数组")
        for group in ("observations", "monitoring_intervals"):
            if group in records and (not isinstance(records[group], list)
                    or any(not isinstance(row, dict) for row in records[group])):
                raise DomainError(f"{group}必须为记录数组")
        advice = records.get("advice", [])
        if not isinstance(advice, list) or any(not isinstance(row, dict) for row in advice):
            raise DomainError("建议必须为记录数组")
        seen_advice = set()
        for row in advice:
            if not isinstance(row.get("advice_id"), str) or not row["advice_id"]:
                raise DomainError("建议记录缺少编号")
            key = (row["advice_id"], str(row.get("version", "")))
            if key in seen_advice:
                raise DomainError("建议包含重复版本")
            seen_advice.add(key)
            for field in ("advice_items", "referenced_advice_items"):
                if type(row.get(field, 0)) is not int or row.get(field, 0) < 0:
                    raise DomainError("建议条数必须为非负整数")
            if row.get("referenced_advice_items", 0) > row.get("advice_items", 0):
                raise DomainError("有出处的建议条数超过总条数")
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
        value["manifest_sha256"] = _digest(value)
        self.store.put("evaluation_manifest", value["manifest_id"], value)
        return deepcopy(value)

    @staticmethod
    def _match_events(actual: list[dict], alerts: list[dict], tolerance_seconds: float,
                      time_semantics: str | None = None) -> list[dict]:
        actual = sorted(actual, key=lambda item: (item["point_id"], _time(item["start_at"]), item["event_id"]))
        alerts = sorted(alerts, key=lambda item: (item["point_id"], _time(item["start_at"]), item["event_id"]))
        tolerance = timedelta(seconds=tolerance_seconds)
        adjacency = []
        for alert in alerts:
            compatible = []
            for index, observed in enumerate(actual):
                right = _time(observed["start_at"]) - tolerance
                ends_after_start = (_time(alert["end_at"]) > right if time_semantics == "prediction_target_windows"
                                    else _time(alert["end_at"]) >= right)
                if (observed["point_id"] == alert["point_id"]
                        and _time(alert["start_at"]) <= _time(observed["end_at"]) + tolerance
                        and ends_after_start):
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
                                     manifest["matching_config"]["tolerance_seconds"],
                                     manifest["matching_config"].get("time_semantics"))
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
        advice = records.get("advice", tasks)
        ratio("advice_reference_coverage", sum(row.get("referenced_advice_items", 0) for row in advice),
              sum(row.get("advice_items", 0) for row in advice))
        return {
            "metric_values": metrics, "numerators": numerators, "denominators": denominators,
            "sample_refs": {key: [row[id_key] for row in records[key]] for key, id_key in (
                ("actual_events", "event_id"), ("alert_events", "event_id"),
                ("samples", "sample_id"), ("tasks", "task_id"),
            )},
            "event_matches": matches,
        }

    @staticmethod
    def _view_records(records: dict, variant: str) -> tuple[dict, dict, list[str]]:
        if variant not in ("primary", "all_events", "excluding_suspected"):
            raise DomainError("评估视图不存在")
        missing = list(records.get("missing_fields", []))
        excluded = {key: [] for key in ("actual_events", "alert_events", "samples")}
        if variant in ("primary", "all_events"):
            return records, excluded, missing
        filtered = deepcopy(records)
        for key, identifier in (("actual_events", "event_id"), ("alert_events", "event_id"), ("samples", "sample_id"),
                                ("observations", "id"), ("monitoring_intervals", "interval_id")):
            rows = records.get(key, [])
            flagged = [row for row in rows if "suspected_calibration" in row.get("quality_flags", [])]
            if key in excluded:
                excluded[key] = [row[identifier] for row in flagged]
            filtered[key] = [row for row in rows if "suspected_calibration" not in row.get("quality_flags", [])]
        filtered_hours = records.get("valid_monitoring_hours_excluding_suspected")
        has_suspected = any("suspected_calibration" in row.get("quality_flags", [])
                            for key in ("actual_events", "alert_events", "samples", "observations", "monitoring_intervals")
                            for row in records.get(key, []))
        if filtered_hours is None and not has_suspected:
            filtered_hours = records["valid_monitoring_hours"]
        filtered["valid_monitoring_hours"] = filtered_hours
        if filtered_hours is None:
            missing.append("valid_monitoring_hours_excluding_suspected")
        return filtered, excluded, list(dict.fromkeys(missing))

    @staticmethod
    def _metric_statuses(summary, records, missing):
        reasons = {
            "precision": "no_alert_events", "recall": "no_actual_events",
            "false_alerts_per_valid_monitoring_hour": "missing_valid_monitoring_hours",
            "sample_precision": "no_predicted_positive_samples", "sample_recall": "no_positive_samples",
            "sample_fpr": "no_negative_samples", "mae": "no_regression_samples",
            "mape_with_zero_policy": "zero_actuals_excluded", "tool_task_completion_rate": "no_tasks",
            "evidence_completeness": "no_tasks", "advice_reference_coverage": "no_advice_items",
        }
        event_metrics = {"precision", "recall", "missed_events", "false_alerts_per_valid_monitoring_hour"}
        sample_metrics = {"sample_precision", "sample_recall", "sample_fpr", "mae", "mape_with_zero_policy"}
        unavailable = None
        if "forecast_runs" in missing:
            unavailable = "no_forecast_runs"
        elif "mature_forecast_samples" in missing:
            unavailable = "no_mature_forecast_samples"
        event_unavailable = "forecast_event_policy_not_supported" if "forecast_event_policy" in missing else unavailable
        if "observed_event_evaluation" in missing:
            event_unavailable = "no_observed_assessments"
        statuses = {}
        for name, value in summary["metric_values"].items():
            reason = event_unavailable if name in event_metrics else unavailable if name in sample_metrics else None
            if reason:
                summary["metric_values"][name] = None
                statuses[name] = {"status": "unavailable", "reason": reason}
            elif value is None:
                reason = reasons.get(name, "zero_denominator")
                if name == "false_alerts_per_valid_monitoring_hour" and summary["denominators"][name] == 0:
                    reason = "no_valid_monitoring_hours"
                if name == "mape_with_zero_policy" and summary["denominators"]["mae"] == 0:
                    reason = "no_regression_samples"
                statuses[name] = {"status": "unavailable", "reason": reason}
            else:
                statuses[name] = {"status": "calculated", "reason": None}
        return statuses

    def _summary_view(self, manifest, records, variant):
        values, excluded, missing = self._view_records(records, variant)
        summary = self._summarize(manifest, values)
        summary["excluded_refs"] = excluded
        summary["missing_fields"] = missing
        summary["metric_statuses"] = self._metric_statuses(summary, values, missing)
        return summary

    @staticmethod
    def _verify_manifest(manifest):
        if not isinstance(manifest, dict) or "frozen_records" not in manifest or "records_sha256" not in manifest:
            raise DomainError("冻结清单缺失或格式错误", code="conflict", status_code=409)
        if _digest(manifest["frozen_records"]) != manifest["records_sha256"]:
            raise DomainError("冻结记录校验失败", code="conflict", status_code=409)
        expected = manifest.get("manifest_sha256")
        if expected and _digest({key:value for key,value in manifest.items() if key != "manifest_sha256"}) != expected:
            raise DomainError("冻结清单校验失败", code="conflict", status_code=409)

    @staticmethod
    def _verify_report(report, manifest):
        EvaluationService._verify_manifest(manifest)
        if report["face_id"] != manifest["face_id"] or report["records_sha256"] != manifest["records_sha256"] or (
            report.get("manifest_sha256") and report["manifest_sha256"] != manifest.get("manifest_sha256")
        ):
            raise DomainError("报告与冻结清单不一致", code="conflict", status_code=409)
        expected = report.get("report_sha256")
        if expected and _digest({key:value for key,value in report.items() if key != "report_sha256"}) != expected:
            raise DomainError("冻结报告校验失败", code="conflict", status_code=409)

    def build_report(self, manifest_id: str, frozen_records: dict | None = None) -> dict:
        with self.store.transaction() as connection:
            return self._build_report(manifest_id, frozen_records, connection)

    def _build_report(self, manifest_id, frozen_records, connection):
        manifest = self.store.get("evaluation_manifest", manifest_id, connection)
        if manifest is None:
            raise DomainError("评估清单不存在", code="not_found", status_code=404)
        records = manifest["frozen_records"]
        self._verify_manifest(manifest)
        if frozen_records is not None and _digest(frozen_records) != manifest["records_sha256"]:
            raise DomainError("不能替换已冻结的评估记录", code="conflict", status_code=409)
        existing = [report for report in self.store.list("evaluation_report", connection) if report["manifest_id"] == manifest_id]
        if existing:
            report = min(existing, key=lambda item: (item["generated_at"], item["evaluation_id"]))
            self._verify_report(report, manifest)
            return deepcopy(report)
        primary = self._summary_view(manifest, records, "primary")
        sensitivity = self._summary_view(manifest, records, "excluding_suspected")
        scope_keys = ("session_id", "source_id", "point_ids", "start_at", "end_at", "as_of", "mode",
                      "prediction_stage", "model_id", "model_version", "dataset_split", "dataset_split_semantics",
                      "rule_version", "preprocess_version", "matching_config", "quality_policy")
        report = {
            **deepcopy(primary), "evaluation_id": str(uuid4()), "manifest_id": manifest_id,
            "face_id": manifest["face_id"], "records_sha256": manifest["records_sha256"],
            "manifest_sha256": manifest.get("manifest_sha256"),
            "scope_metadata": {key: deepcopy(manifest[key]) for key in scope_keys if key in manifest},
            "collection_summary": deepcopy(manifest.get("collection_summary", records.get("collection_summary", {}))),
            "mape_zero_policy": manifest["quality_policy"]["mape_zero_policy"],
            "primary_calibration_view": "all_events",
            "calibration_views": {"all_events": primary, "excluding_suspected": sensitivity},
            "generated_at": utcstr(now_utc()),
        }
        report = _public(report)
        report["report_sha256"] = _digest(report)
        self.store.put("evaluation_report", report["evaluation_id"], report, connection)
        return deepcopy(report)

    def run(self, actor: Actor, spec: dict, records: dict) -> dict:
        manifest = self.freeze_manifest(actor, spec, {"frozen_records": records})
        return self.build_report(manifest["manifest_id"])

    def get_report(self, actor: Actor, evaluation_id: str) -> dict:
        if not isinstance(evaluation_id, str) or not evaluation_id.strip():
            raise DomainError("评估编号无效")
        report = self.store.get("evaluation_report", evaluation_id)
        if report is None:
            raise DomainError("评估报告不存在", code="not_found", status_code=404)
        self._authorize(actor, report["face_id"])
        manifest = self.store.get("evaluation_manifest", report["manifest_id"])
        if manifest is None:
            raise DomainError("冻结清单不存在", code="conflict", status_code=409)
        self._verify_report(report, manifest)
        return deepcopy(report)

    def get_manifest(self, actor: Actor, evaluation_id: str) -> dict:
        report = self.get_report(actor, evaluation_id)
        original = self.store.get("evaluation_manifest", report["manifest_id"])
        self._verify_report(report, original)
        exported = _public(original)
        exported["export_redacted"] = exported != original
        exported["export_records_sha256"] = _digest(exported["frozen_records"])
        exported["export_manifest_sha256"] = _digest(exported)
        return exported

    def export_report(self, actor: Actor, evaluation_id: str) -> dict:
        return {"report": self.get_report(actor, evaluation_id), "manifest": self.get_manifest(actor, evaluation_id)}

    def get_frozen_observation(self, actor: Actor, evaluation_id: str, observation_id: str) -> dict:
        report = self.get_report(actor, evaluation_id)
        manifest = self.store.get("evaluation_manifest", report["manifest_id"])
        self._verify_report(report, manifest)
        row = next((item for item in manifest["frozen_records"].get("observations", [])
                    if item.get("id", item.get("observation_id")) == observation_id), None)
        if row is None:
            raise DomainError("冻结报告中不存在该观测", code="not_found", status_code=404)
        return {"evaluation_id": evaluation_id, "manifest_id": report["manifest_id"],
                "records_sha256": report["records_sha256"], "manifest_sha256": report["manifest_sha256"],
                "report_sha256": report["report_sha256"], "kind": "observation", "record": _public(row)}

    @staticmethod
    def _metric_rows(manifest, records, metric):
        matches = EvaluationService._match_events(records["actual_events"], records["alert_events"],
                                                   manifest["matching_config"]["tolerance_seconds"],
                                                   manifest["matching_config"].get("time_semantics"))
        actual_by_id = {row["event_id"]: row for row in records["actual_events"]}
        alert_by_id = {row["event_id"]: row for row in records["alert_events"]}
        by_actual = {pair["actual_event_id"]: pair["alert_event_id"] for pair in matches}
        by_alert = {pair["alert_event_id"]: pair["actual_event_id"] for pair in matches}
        if metric in ("precision", "recall", "missed_events", "false_alerts_per_valid_monitoring_hour"):
            actual_view = metric in ("recall", "missed_events")
            rows = records["actual_events"] if actual_view else records["alert_events"]
            paired = by_actual if actual_view else by_alert
            for record in rows:
                matched = record["event_id"] in paired
                if metric in ("missed_events", "false_alerts_per_valid_monitoring_hour") and matched:
                    continue
                snapshots = {"actual_event" if actual_view else "alert_event": deepcopy(record)}
                if matched:
                    snapshots["alert_event" if actual_view else "actual_event"] = deepcopy(
                        alert_by_id[paired[record["event_id"]]] if actual_view else actual_by_id[paired[record["event_id"]]])
                yield {"kind": "event_match" if matched else "actual_event" if actual_view else "alert_event",
                       "record": deepcopy(record), "snapshots": snapshots, "matched": matched, "included": True,
                       "numerator_contribution": int(matched) if metric in ("precision", "recall") else 1,
                       "denominator_contribution": 1 if metric in ("precision", "recall") else 0}
            if metric == "false_alerts_per_valid_monitoring_hour":
                intervals = records.get("monitoring_intervals", [])
                if not intervals and records["valid_monitoring_hours"] is not None:
                    intervals = [{"hours": records["valid_monitoring_hours"], "semantics": "provided_aggregate"}]
                for record in intervals:
                    hours = record.get("hours", record.get("duration_seconds", 0) / 3600)
                    yield {"kind": "monitoring_interval", "record": deepcopy(record), "hours": hours,
                           "snapshots": {"monitoring_interval": deepcopy(record)}, "included": True,
                           "numerator_contribution": 0, "denominator_contribution": hours}
        elif metric in ("sample_precision", "sample_recall", "sample_fpr", "mae", "mape_with_zero_policy"):
            for record in records["samples"]:
                actual, predicted = record["actual_positive"], record["predicted_positive"]
                classification = "TP" if actual and predicted else "FN" if actual else "FP" if predicted else "TN"
                valid = record.get("valid", True) or not manifest["quality_policy"].get("exclude_invalid_samples", True)
                numeric = record.get("actual_value") is not None and record.get("predicted_value") is not None
                error = abs(record["predicted_value"] - record["actual_value"]) if numeric else None
                percentage = error / abs(record["actual_value"]) * 100 if numeric and record["actual_value"] != 0 else None
                if metric == "sample_precision":
                    included, numerator, denominator = valid, int(classification == "TP"), int(predicted)
                elif metric == "sample_recall":
                    included, numerator, denominator = valid, int(classification == "TP"), int(actual)
                elif metric == "sample_fpr":
                    included, numerator, denominator = valid, int(classification == "FP"), int(not actual)
                elif metric == "mae":
                    included, numerator, denominator = valid and numeric, error or 0, 1
                else:
                    included, numerator, denominator = valid and percentage is not None, percentage or 0, 1
                yield {"kind": "sample", "record": deepcopy(record), "snapshots": {"sample": deepcopy(record)},
                       "classification": classification, "included": bool(included), "absolute_error": error,
                       "percentage_error": percentage, "numerator_contribution": numerator if included else 0,
                       "denominator_contribution": denominator if included else 0}
        else:
            independent_advice = metric == "advice_reference_coverage" and "advice" in records
            for record in records["advice"] if independent_advice else records["tasks"]:
                included = True if independent_advice else record.get("normal_task", True)
                if metric == "tool_task_completion_rate":
                    numerator, denominator = int(record["state"] == "completed"), 1
                elif metric == "evidence_completeness":
                    numerator, denominator = int(record.get("evidence_complete", False)), 1
                else:
                    numerator, denominator = record.get("referenced_advice_items", 0), record.get("advice_items", 0)
                kind = "advice" if independent_advice else "task"
                yield {"kind": kind, "record": deepcopy(record), "snapshots": {kind: deepcopy(record)},
                       "included": included, "numerator_contribution": numerator if included else 0,
                       "denominator_contribution": denominator if included else 0}

    def get_metric_details(self, actor: Actor, evaluation_id: str, metric: str, variant="primary", offset=0, limit=50) -> dict:
        report = self.get_report(actor, evaluation_id)
        if not isinstance(metric, str) or metric not in report["metric_values"]:
            raise DomainError("评估指标不存在")
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise DomainError("明细分页范围无效")
        manifest = self.store.get("evaluation_manifest", report["manifest_id"])
        self._verify_report(report, manifest)
        records, _, _ = self._view_records(manifest["frozen_records"], variant)
        summary = report if variant in ("primary", "all_events") else report["calibration_views"]["excluding_suspected"]
        status = summary.get("metric_statuses", {}).get(metric, {
            "status": "calculated" if summary["metric_values"][metric] is not None else "unavailable",
            "reason": None if summary["metric_values"][metric] is not None else "zero_denominator"})
        page, total = [], 0
        for total, row in enumerate(self._metric_rows(manifest, records, metric), 1):
            if offset < total <= offset + limit:
                page.append(_public(row))
        unit = "valid_monitoring_hours" if metric == "false_alerts_per_valid_monitoring_hour" else (
            "count" if metric == "missed_events" else "tasks" if metric in ("tool_task_completion_rate", "evidence_completeness")
            else "advice_items" if metric == "advice_reference_coverage" else "events" if metric in ("precision", "recall") else "samples")
        return {"evaluation_id": evaluation_id, "manifest_id": report["manifest_id"],
                "records_sha256": report["records_sha256"], "manifest_sha256": report.get("manifest_sha256"),
                "report_sha256": report.get("report_sha256"), "manifest_hash": report.get("manifest_sha256"),
                "metric": metric, "variant": variant, "value": summary["metric_values"][metric],
                "numerator": summary["numerators"][metric], "denominator": summary["denominators"][metric],
                **status, "undefined_reason": status["reason"], "denominator_unit": unit,
                "total": total, "offset": offset, "limit": limit, "rows": page, "items": deepcopy(page)}
