"""Read-only collection of source-bound replay records and matured outcomes."""
from copy import deepcopy
from datetime import datetime, timedelta
from hashlib import sha256
import json
from math import isfinite

from sqlalchemy import select

from app.db import observations, records as record_table
from app.schemas.common import DomainError, utcstr

__all__ = ["SessionEvaluationCollector"]

_VERSION = "session-evaluation-v1"
_INVALID = {"invalid", "invalid_value", "missing", "duplicate_timestamp", "forward_filled"}
_PHYSICAL_UNITS = {"%", "%CH4", "%CH4（推定）"}
_EVENT_FIELDS = ("event_id","session_id","source_id","point_id","event_type","event_revision","revision_as_of",
    "event_time","start_at","end_at","alert_level","event_state","rule_version","rule_status","forecast_run_id",
    "quality_flags","evidence_refs","evidence_status","late_observation_refs","evidence_reassessment")


def _time(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        utcstr(parsed)
        return parsed
    except (AttributeError, TypeError, ValueError):
        raise DomainError("采集时间必须为带时区的时间") from None


def _pick(value, keys):
    return {key: deepcopy(value[key]) for key in keys if key in value}


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


def _valid(row):
    return _number(row.get("value")) and not (_INVALID & set(row.get("quality_flags", [])))


def _id(*parts):
    return sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _reference(row):
    return dict(kind="observation", observation_id=row["id"], point_id=row["point_id"],
        source_file_id=row["source_file_id"], row_number=row["row_number"],
        event_time=row["event_time"], available_at=row["available_at"])


class SessionEvaluationCollector:
    def __init__(self, store, services):
        self.store, self.services = store, services

    def collect(self, context, start_at, end_at, point_ids=None, model_id=None, model_version=None,
                dataset_split="unclassified", tolerance_seconds=0, prediction_stage=None):
        context.actor.require_role("safety_management")
        start, end = _time(start_at), _time(end_at)
        if start >= end or end > context.as_of:
            raise DomainError("评估区间须有正时长且结束不晚于会话数据时间")
        if not isinstance(dataset_split, str) or not dataset_split.strip():
            raise DomainError("数据分区标签不能为空")
        if not _number(tolerance_seconds) or tolerance_seconds < 0:
            raise DomainError("事件匹配容差须为非负有限秒数")
        for value in (model_id, model_version, prediction_stage):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DomainError("模型编号、版本和预测阶段须为非空字符串")
        if model_version is not None and model_id is None:
            raise DomainError("指定模型版本时需要模型编号")
        with self.store.transaction() as connection:
            session = self.store.get("session", context.session_id, connection)
            source = self.store.get("source", context.source_id, connection)
            if session is None or source is None:
                raise DomainError("评估会话或数据源不存在", "not_found", 404)
            context.actor.require_face(source["face_id"])
            if (session["source_id"] != context.source_id or session["mode"] != context.mode
                    or session.get("face_id", source["face_id"]) != source["face_id"]):
                raise DomainError("评估会话与来源不一致", "forbidden", 403)
            if context.as_of > _time(session["as_of"]) or start < _time(session["start_as_of"]):
                raise DomainError("评估范围超出会话数据时钟")
            point_catalog = {point["point_id"]:point for point in self.store.list("point", connection)}
            if point_ids is not None:
                if not isinstance(point_ids,list) or not point_ids or any(not isinstance(point,str) for point in point_ids):
                    raise DomainError("评估测点须为非空编号列表")
                for point_id in point_ids:
                    point = point_catalog.get(point_id)
                    if point is None:
                        raise DomainError("评估测点不存在", "not_found", 404)
                    if point["source_id"] != context.source_id:
                        raise DomainError("评估测点不属于当前数据源", "forbidden", 403)
            points = {point["point_id"]: point for point in point_catalog.values()
                      if point["source_id"] == context.source_id and point.get("quantity") == "methane"}
            if point_ids is not None and (not isinstance(point_ids, list) or not point_ids
                    or any(not isinstance(point, str) or point not in points for point in point_ids)):
                raise DomainError("评估测点须为当前来源的甲烷测点")
            requested_points = set(point_ids) if point_ids is not None else set(points)
            if not requested_points:
                raise DomainError("当前来源没有可评估的甲烷测点")
            models = [model for model in self.store.list("model", connection)
                      if model.get("source_id") == context.source_id and model.get("face_id") == source["face_id"]]
            saved_runs = [run for run in self.store.list("forecast_run", connection)
                          if self._in_scope(run, context) and run.get("point_id") in requested_points
                          and self._visible(run.get("as_of"), end)
                          and _time(run["as_of"]) >= _time(session["start_as_of"])]
            runs, selected_model = self._models(saved_runs, models, model_id, model_version)
            if selected_model:
                supported_points = set(selected_model.get("point_ids", [])) & set(points)
                if point_ids is None:
                    requested_points &= supported_points
                elif not requested_points <= supported_points:
                    raise DomainError("所选模型不适用于评估测点")
            if not requested_points:
                raise DomainError("所选模型没有适用的甲烷测点")
            runs = [run for run in runs if run["point_id"] in requested_points]
            rules = {point: self.services["alerts"].get_rules(context.actor, source["face_id"], point)
                     for point in sorted(requested_points)}
            max_gap = max(rule["max_observation_gap_seconds"] for rule in rules.values())
            rows = [dict(row) for row in connection.execute(select(observations).where(
                observations.c.source_id == context.source_id,
                observations.c.point_id.in_(sorted(requested_points)),
                observations.c.event_time >= utcstr(start - timedelta(seconds=max_gap)),
                observations.c.event_time <= utcstr(end), observations.c.available_at <= utcstr(end),
            ).order_by(observations.c.point_id, observations.c.event_time, observations.c.id)).mappings()]
            raw = {kind: deepcopy(self.store.list(kind, connection)) for kind in
                   ("event_revision", "task", "advice", "case_action", "session_action")}
            payload = record_table.c.payload
            tick_records = [dict(row) for row in connection.execute(select(
                record_table.c.id.label("tick_id"),
                payload["session"]["session_id"].as_string().label("session_id"),
                payload["session"]["source_id"].as_string().label("source_id"),
                payload["session"]["as_of"].as_string().label("as_of"),
                payload["assessments"].label("assessments"),
            ).where(record_table.c.kind=="tick_result",
                    payload["session"]["session_id"].as_string()==context.session_id,
                    payload["session"]["source_id"].as_string()==context.source_id,
                    payload["session"]["as_of"].as_string()>=utcstr(start),
                    payload["session"]["as_of"].as_string()<=utcstr(end))).mappings()]
            files = {row["source_file_id"] for row in rows}
            file_refs = [_pick(self.store.get("data_file", file_id, connection) or {"file_id":file_id},
                              ("file_id", "sha256", "size")) for file_id in sorted(files)]
        # All database reads above share one snapshot. The remaining work changes only copies.
        stage = self._stage(runs, prediction_stage)
        missing = []
        if not runs:
            missing.append("forecast_runs")
        if not stage:
            missing.append("prediction_stage")
        monitoring = self._monitoring(rows, rules, start, end)
        by_point = {point:[row for row in rows if row["point_id"] == point] for point in requested_points}
        support = {point:[item for item in monitoring if item["point_id"] == point] for point in requested_points}
        samples, excluded_intervals = [], {}
        eligible_intervals = {}
        for run in runs:
            selected = next((output for output in run.get("outputs", []) if self._physical(output)
                             and output.get("prediction_stage") == stage), None)
            if selected is None:
                continue
            statistic = self._statistic(run, selected, selected_model)
            if statistic is None and "target_statistic" not in missing:
                missing.append("target_statistic")
            for index, interval in enumerate(selected.get("interval_values", [])):
                a, b = _time(interval["start_at"]), _time(interval["end_at"])
                reason = None
                if not _number(interval.get("value")) or a >= b:
                    reason = "invalid_forecast_interval"
                elif b > end:
                    reason = "unmature_or_outside_window"
                elif a < start or a < _time(run["as_of"]):
                    reason = "outside_window_or_started_before_issue"
                elif not self._covered(a, b, support[run["point_id"]]):
                    reason = "incomplete_observation_support"
                if reason:
                    excluded_intervals[reason] = excluded_intervals.get(reason, 0) + 1
                    continue
                eligible_intervals.setdefault(run["run_id"], []).append((a, b, interval["value"]))
                if statistic is None:
                    continue
                actual_rows = [row for row in by_point[run["point_id"]] if a <= _time(row["event_time"]) < b and _valid(row)]
                if not actual_rows:
                    continue
                actual = max(row["value"] for row in actual_rows) if statistic == "max" else sum(row["value"] for row in actual_rows) / len(actual_rows)
                threshold = rules[run["point_id"]]["observed_red_threshold"]
                flags = sorted({flag for row in actual_rows for flag in row.get("quality_flags", [])})
                samples.append(dict(sample_id=_id(run["run_id"], stage, index), session_id=context.session_id,
                    source_id=context.source_id, face_id=source["face_id"], point_id=run["point_id"], run_id=run["run_id"],
                    model_id=run["model_id"], model_version=run["model_version"], prediction_stage=stage,
                    interval_index=index, start_at=utcstr(a), end_at=utcstr(b), target_start_at=utcstr(a), target_end_at=utcstr(b),
                    issued_at=run["as_of"], target_statistic=statistic, quantity_semantics="physical_forecast", unit=selected["unit"],
                    threshold=threshold, actual_value=actual, predicted_value=interval["value"], valid=True,
                    actual_positive=actual >= threshold, predicted_positive=interval["value"] >= threshold,
                    quality_flags=flags, evidence_refs=[_reference(row) for row in actual_rows]))
        if not samples:
            missing.append("mature_forecast_samples")
        if stage != "residual_corrected":
            missing.append("forecast_event_policy")
        actual_events, alert_events, event_revisions = self._events(raw["event_revision"], context, requested_points,
            runs, eligible_intervals, rules, stage, start, end)
        evaluated_points = {event["point_id"] for event in actual_events}
        for tick in tick_records:
            tick["assessments"] = [_pick(item,("point_id","as_of","status","observed_status","missing_fields"))
                for item in tick.get("assessments") or [] if item.get("point_id") in requested_points
                and item.get("observed_status") is not None]
            evaluated_points.update(item["point_id"] for item in tick["assessments"] if item.get("observed_status")=="evaluated")
        missing_points = requested_points-evaluated_points
        if missing_points:
            missing.append("observed_event_evaluation")
        event_ids = {event["event_id"] for event in event_revisions}
        tasks, advice, actions, session_actions = self._audit(raw, context, event_ids, requested_points == set(points), start, end)
        hours = sum(item["duration_seconds"] for item in monitoring) / 3600
        filtered_hours = sum(item["duration_seconds"] for item in monitoring if "suspected_calibration" not in item["quality_flags"]) / 3600
        frozen_rows = [{**row, "boundary_support": not start <= _time(row["event_time"]) < end} for row in rows]
        model_info = _pick(selected_model or {}, ("model_id", "version", "adapter_kind", "model_type", "purpose", "point_ids",
            "target_quantity", "unit", "aggregation_seconds", "target_statistic", "input_statistic", "training_range", "deployment_stage"))
        per_point = {point:dict(valid_monitoring_hours=sum(row["duration_seconds"] for row in support[point])/3600,
            valid_monitoring_hours_excluding_suspected=sum(row["duration_seconds"] for row in support[point] if "suspected_calibration" not in row["quality_flags"])/3600)
            for point in sorted(requested_points)}
        summary = dict(collector_version=_VERSION, truth_semantics="observed_threshold_derived", independent_truth_labels=False,
            actual_event_collection="saved_session_observed_threshold_events",
            observed_event_evaluated_point_ids=sorted(evaluated_points), observed_event_missing_point_ids=sorted(missing_points),
            forecast_event_semantics="saved_forecast_notifications_mapped_to_mature_target_windows",
            event_window_coverage="mature_prediction_targets_with_valid_observation_support",
            sample_semantics="one_saved_forecast_run_stage_interval", target_aggregation="declared_model_statistic",
            monitoring_time_semantics="sensor_hours_sum_of_adjacent_valid_observation_support", per_point_monitoring=per_point,
            point_ids=sorted(requested_points), model=model_info, prediction_stage=stage,
            eligible_forecast_runs=len(runs), mature_forecast_samples=len(samples), observed_event_count=len(actual_events),
            forecast_target_event_count=len(alert_events), observation_count=len(rows), task_count=len(tasks), advice_count=len(advice),
            case_action_count=len(actions), session_action_count=len(session_actions), excluded_intervals=excluded_intervals,
            task_scope="session_tasks_and_point_scoped_event_tasks" if requested_points == set(points) else "selected_event_tasks_only",
            missing_fields=sorted(set(missing)))
        records = dict(actual_events=actual_events, alert_events=alert_events, samples=samples, tasks=tasks,
            valid_monitoring_hours=hours, valid_monitoring_hours_excluding_suspected=filtered_hours,
            observations=frozen_rows, observation_refs=[_reference(row) for row in rows], monitoring_intervals=monitoring,
            event_revisions=event_revisions, forecast_runs=[self._run_snapshot(run, stage) for run in runs],
            tick_records=tick_records,
            advice=advice, case_actions=actions, session_actions=session_actions, rules=[rules[point] for point in sorted(rules)],
            model=model_info, source_snapshot=_pick(source,("source_id","name","face_id","timezone")),
            session_snapshot=_pick(session,("session_id","source_id","face_id","mode","start_as_of","as_of","state","speed","config_version","parent_session_id","prediction_models")),
            data_file_refs=file_refs, missing_fields=sorted(set(missing)), collection_summary=summary)
        spec = dict(face_id=source["face_id"], dataset_refs=[context.source_id] +
            ["sha256:"+item["sha256"] if item.get("sha256") else "source_file:"+item["file_id"] for item in file_refs],
            dataset_split=dataset_split, dataset_split_semantics="user_declared", model_version=(model_info.get("model_id", model_id or "none")+":"+model_info.get("version", model_version or "not_available")),
            rule_version=",".join(sorted({rule["rule_version"] for rule in rules.values()})), preprocess_version=_VERSION,
            matching_config=dict(tolerance_seconds=tolerance_seconds, time_semantics="prediction_target_windows"),
            quality_policy=dict(exclude_invalid_samples=True, mape_zero_policy="exclude_zero_actuals", monitoring_gap_seconds=max_gap,
                                calibration_view="all_events_primary", target_aggregation="declared_model_statistic"),
            task_set_ref="session:"+context.session_id, source_id=context.source_id, session_id=context.session_id,
            start_at=utcstr(start), end_at=utcstr(end), as_of=utcstr(end), session_as_of=utcstr(context.as_of),
            point_ids=sorted(requested_points), prediction_stage=stage, collection_summary=summary)
        return dict(spec=spec, records=records, collection_summary=deepcopy(summary),
            artifact_refs=dict(frozen_record_refs=["session:"+context.session_id]+["observation:"+row["id"] for row in rows]
                               +["forecast_run:"+run["run_id"] for run in runs]))

    @staticmethod
    def _in_scope(row, context):
        return row.get("session_id") == context.session_id and row.get("source_id") == context.source_id

    @staticmethod
    def _visible(value, cutoff):
        return value is not None and _time(value) <= cutoff

    @staticmethod
    def _physical(output):
        return output.get("quantity_semantics") == "physical_forecast" and output.get("unit") in _PHYSICAL_UNITS

    def _models(self, saved, models, model_id, version):
        valid = [run for run in saved if run.get("status") == "completed" and any(self._physical(output) for output in run.get("outputs", []))
                 and self._visible(run.get("cutoff", run["as_of"]), _time(run["as_of"]))
                 and all(self._visible(ref.get("event_time", run["as_of"]), _time(run.get("cutoff", run["as_of"])))
                         and self._visible(ref.get("available_at", run["as_of"]), _time(run["as_of"])) for ref in run.get("input_refs", []))]
        valid = [run for run in valid if (model_id is None or run.get("model_id") == model_id)
                 and (version is None or run.get("model_version") == version)]
        identities = {(run.get("model_id"), run.get("model_version")) for run in valid}
        if len(identities) > 1:
            raise DomainError("评估需要指定一个模型及版本", missing_fields=["model_id", "model_version"])
        if identities:
            model_id, version = next(iter(identities))
        candidates = [model for model in models if model.get("model_id") == model_id and (version is None or model.get("version") == version)]
        if len(candidates) > 1:
            raise DomainError("评估需要指定模型版本", missing_fields=["model_version"])
        model = candidates[0] if candidates else None
        if model is None and valid:
            model = dict(model_id=model_id, version=version, point_ids=sorted({run["point_id"] for run in valid}))
        return valid, model

    def _stage(self, runs, requested):
        stages = {output.get("prediction_stage") for run in runs for output in run.get("outputs", []) if self._physical(output)} - {None}
        if requested is not None:
            return requested if requested in stages else None
        if "residual_corrected" in stages:
            return "residual_corrected"
        if len(stages) > 1:
            raise DomainError("评估需要指定单个物理预测阶段", missing_fields=["prediction_stage"])
        return next(iter(stages)) if stages else None

    @staticmethod
    def _statistic(run, output, model):
        declared = output.get("target_statistic", run.get("target_statistic", (model or {}).get("target_statistic")))
        if declared is None and (model or {}).get("adapter_kind") == "persistence":
            declared = (model or {}).get("input_statistic", "mean")
        return {"max":"max", "interval_maximum":"max", "interval_max":"max", "mean":"mean", "interval_mean":"mean"}.get(declared)

    @staticmethod
    def _monitoring(rows, rules, start, end):
        result = []
        for point, rule in rules.items():
            point_rows = [row for row in rows if row["point_id"] == point]
            for left, right in zip(point_rows, point_rows[1:]):
                a, b = _time(left["event_time"]), _time(right["event_time"])
                if not (_valid(left) and _valid(right)) or not 0 < (b-a).total_seconds() <= rule["max_observation_gap_seconds"]:
                    continue
                a, b = max(a,start), min(b,end)
                if a < b:
                    result.append(dict(interval_id=_id(left["id"],right["id"],utcstr(a),utcstr(b)), point_id=point,
                        start_at=utcstr(a), end_at=utcstr(b), duration_seconds=(b-a).total_seconds(),
                        quality_flags=sorted(set(left.get("quality_flags", [])) | set(right.get("quality_flags", []))),
                        evidence_refs=[_reference(left),_reference(right)]))
        return result

    @staticmethod
    def _covered(start, end, support):
        cursor = start
        for item in support:
            a, b = _time(item["start_at"]), _time(item["end_at"])
            if b <= cursor:
                continue
            if a > cursor:
                return False
            cursor = max(cursor,b)
            if cursor >= end:
                return True
        return False

    def _events(self, revisions, context, points, runs, eligible, rules, stage, start, end):
        run_map = {run["run_id"]:run for run in runs}
        visible = [event for event in revisions if self._in_scope(event,context) and event.get("point_id") in points
                   and self._visible(event.get("revision_as_of"),end) and _time(event["start_at"]) < end]
        latest = {}
        for event in visible:
            if event["event_id"] not in latest or event["event_revision"] > latest[event["event_id"]]["event_revision"]:
                latest[event["event_id"]] = event
        actual, alerts, snapshots = [], [], []
        for event in sorted(latest.values(),key=lambda row:(row["start_at"],row["event_id"])):
            lower = max(_time(event["start_at"]),start)
            upper = min(_time(event.get("end_at") or event["event_time"]),end)
            snapshot = _pick(event,_EVENT_FIELDS)
            if event["event_type"] == "observed_exceedance":
                if event.get("evidence_status") == "insufficient_after_late_arrival":
                    if any(row["event_id"] == event["event_id"] and
                           _time(row.get("end_at") or row["event_time"]) >= start for row in visible):
                        snapshots.append(snapshot)
                    continue
                if upper < start or lower > upper:
                    continue
                if event.get("end_at") is not None and _time(event["end_at"]) == start and _time(event["start_at"]) < start:
                    continue
                snapshots.append(snapshot)
                actual.append(dict(snapshot, start_at=utcstr(lower),end_at=utcstr(upper), original_start_at=event["start_at"],
                    semantics="observed_threshold_events", truth_semantics="observed_threshold_derived"))
            elif event["event_type"] == "forecast_exceedance" and stage == "residual_corrected":
                ids = {ref.get("run_id") for ref in event.get("evidence_refs",[]) if ref.get("kind")=="forecast_run"}
                ids.add(event.get("forecast_run_id"))
                ids &= set(run_map)
                if not ids:
                    continue
                snapshots.append(snapshot)
                targets = []
                threshold = rules[event["point_id"]]["forecast_yellow_threshold"]
                for run_id in ids:
                    for a,b,value in eligible.get(run_id,[]):
                        if value >= threshold:
                            targets.append((a,b,run_id))
                merged = []
                for a,b,run_id in sorted(targets):
                    if merged and a <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1],b)
                        merged[-1][2].add(run_id)
                    else:
                        merged.append([a,b,{run_id}])
                for a,b,target_runs in merged:
                    alerts.append(dict(event_id="target:"+_id(event["event_id"],utcstr(a),utcstr(b)), source_event_id=event["event_id"],
                        event_revision=event["event_revision"], session_id=context.session_id,source_id=context.source_id,point_id=event["point_id"],
                        start_at=utcstr(a),end_at=utcstr(b), issued_at=min(run_map[run_id]["as_of"] for run_id in target_runs),
                        notification_start_at=event["start_at"],notification_end_at=event.get("end_at"),
                        run_ids=sorted(target_runs), rule_version=event["rule_version"], threshold=threshold,
                        prediction_stage=stage,semantics="forecast_target_windows", quality_flags=event.get("quality_flags",[]),
                        evidence_refs=[dict(kind="forecast_run",run_id=run_id) for run_id in sorted(target_runs)]))
        selected_ids = {event["event_id"] for event in snapshots}
        history = {(event["event_id"], event["event_revision"]): _pick(event,_EVENT_FIELDS)
                   for event in visible if event["event_id"] in selected_ids}
        return actual,alerts,sorted(history.values(),key=lambda event:(event["start_at"],event["event_id"],event["event_revision"]))

    def _audit(self, raw, context, event_ids, all_points, start, end):
        def in_window(row, field="data_as_of", allow_missing_source=False):
            return (row.get("session_id") == context.session_id and
                (row.get("source_id",context.source_id if allow_missing_source else None) == context.source_id)
                and row.get(field) is not None and start <= _time(row[field]) <= end)
        tasks = []
        for task in sorted(raw["task"], key=lambda row:row["task_id"]):
            if not in_window(task,"as_of") or (task.get("event_id") not in event_ids and not (all_points and not task.get("event_id"))):
                continue
            saved = _pick(task,("task_id","task_type","session_id","source_id","event_id","event_revision","as_of","state","method",
                "model","created_at","updated_at","missing_fields","evidence_refs","citations","usage"))
            saved.update(normal_task=True,evidence_complete=bool(task.get("evidence_refs")) and not task.get("missing_fields"),
                         advice_items=task.get("advice_items",0),referenced_advice_items=task.get("referenced_advice_items",0))
            if saved["state"] == "queued":
                saved.update(state="running",original_state="queued")
            tasks.append(saved)
        advice = []
        for row in raw["advice"]:
            if not in_window(row,allow_missing_source=True) or row.get("event_id") not in event_ids:
                continue
            saved = _pick(row,("advice_id","version","event_id","event_revision","session_id","face_id","data_as_of","knowledge_as_of","reference_ids","operation_context"))
            saved.update(source_id=context.source_id,advice_items=len(row.get("items",[])),
                referenced_advice_items=sum(bool(item.get("reference_id")) for item in row.get("items",[])),
                items=[_pick(item,("reference_id","entry_id","version","locator","kind","file_id")) for item in row.get("items",[])])
            advice.append(saved)
        actions = [_pick(row,("action_id","case_id","event_id","event_revision","session_id","source_id","face_id","action_type",
            "actor_id","role","data_as_of","occurred_at","advice_id","advice_version","review_decision","operation_context"))
            for row in raw["case_action"] if in_window(row) and row.get("event_id") in event_ids]
        session_actions = [deepcopy(row) for row in raw["session_action"] if in_window(row)]
        return tasks, sorted(advice,key=lambda row:(row["advice_id"],row["version"])), sorted(actions,key=lambda row:(row["data_as_of"],row["action_id"])), sorted(session_actions,key=lambda row:row.get("sequence",0))

    @staticmethod
    def _run_snapshot(run, stage):
        saved = _pick(run,("run_id","session_id","source_id","face_id","point_id","model_id","model_version","model_type","as_of","cutoff",
            "status","aggregation","horizon_seconds","input_window","input_refs","input_origin","feature_pipeline_version","quality_flags"))
        saved["outputs"] = [deepcopy(output) for output in run.get("outputs",[]) if output.get("prediction_stage") == stage
                            and output.get("quantity_semantics") == "physical_forecast"]
        return saved
