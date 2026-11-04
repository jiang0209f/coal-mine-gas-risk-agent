"""团队草案规则评估与按逻辑时间可见的事件修订。"""

from datetime import datetime, timedelta
from math import isfinite
from uuid import uuid4

from sqlalchemy import or_, select

from app.db import Store, observations as observation_table
from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc, utcstr

__all__ = ["AlertsService"]

_ROLES = ("ventilation", "gas_governance", "safety_management")
_VERSION = "team-draft-v1"
_RANK = {"blue": 0, "yellow": 1, "orange": 2, "red": 3}


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value) and value >= 0


class AlertsService:
    def __init__(self, store: Store, data_service=None) -> None:
        self.store = store
        self.data_service = data_service

    def _scope(self, context: ExecutionContext, point_id: str | None = None) -> tuple[dict, dict | None]:
        context.actor.require_role(*_ROLES)
        source = self.store.get("source", context.source_id)
        if source is None:
            raise DomainError("数据源不存在", code="not_found", status_code=404)
        context.actor.require_face(source["face_id"])
        point = None
        if point_id is not None:
            point = self.store.get("point", point_id)
            if point is None:
                raise DomainError("测点不存在", code="not_found", status_code=404)
            if point["source_id"] != context.source_id:
                raise DomainError("测点不属于当前数据源", code="forbidden", status_code=403)
        return source, point

    def get_rules(self, actor: Actor, face_id=None, point_id=None, version=None) -> dict:
        actor.require_role(*_ROLES)
        if version not in (None, _VERSION):
            raise DomainError("规则版本不存在", code="not_found", status_code=404)
        if point_id is not None:
            point = self.store.get("point", point_id)
            if point is None:
                raise DomainError("测点不存在", code="not_found", status_code=404)
            source = self.store.get("source", point["source_id"])
            if source is None:
                raise DomainError("数据源不存在", code="not_found", status_code=404)
            actor.require_face(source["face_id"])
            if face_id is not None and source["face_id"] != face_id:
                raise DomainError("测点与工作面不匹配")
            face_id = source["face_id"]
        if face_id is not None:
            actor.require_face(face_id)
        return {
            "rule_version": _VERSION,
            "status": "team-draft",
            "source": "团队拟定，待矿方确认",
            "face_id": face_id,
            "face_ids": [face_id] if face_id is not None else list(actor.face_ids),
            "point_id": point_id,
            "unit": "%CH4（推定）",
            "observed_red_threshold": 0.8,
            "observed_orange_threshold": 0.65,
            "orange_duration_seconds": 300,
            "max_observation_gap_seconds": 60,
            "forecast_yellow_threshold": 0.55 if face_id == "3411" else 0.65,
            "forecast_orange_threshold": 0.8,
            "forecast_horizon_seconds": 3600,
            "forecast_interval_seconds": 300,
            "prediction_stage": "residual_corrected",
            "quantity_semantics": "physical_forecast",
        }

    @staticmethod
    def _reference(row: dict) -> dict:
        return {
            "kind": "observation",
            "observation_id": row.get("id", row.get("observation_id")),
            "source_file_id": row.get("source_file_id"),
            "row_number": row.get("row_number"),
            "event_time": row["event_time"],
            "available_at": row.get("available_at", row["event_time"]),
        }

    def _write_event(self, context, point_id, event_type, level, event_time, start_at,
                     evidence_refs, quality_flags, previous=None, ended=False, forecast_run_id=None, connection=None):
        if previous is None:
            event = {
                "event_id": str(uuid4()), "session_id": context.session_id,
                "source_id": context.source_id, "point_id": point_id,
                "event_type": event_type, "alert_level": level, "event_state": "new",
                "rule_version": _VERSION, "rule_status": "team-draft",
                "start_at": start_at, "end_at": None, "event_revision": 0,
                "created_at": utcstr(now_utc()), "evidence_refs": [], "quality_flags": [],
            }
        else:
            event = {**previous, "evidence_refs": list(previous["evidence_refs"])}
            if level is not None and _RANK[level] > _RANK[event["alert_level"]]:
                event["alert_level"] = level
                event["event_state"] = "upgraded"
        event["event_time"] = event_time
        event["forecast_run_id"] = forecast_run_id
        event["quality_flags"] = sorted(set(event["quality_flags"]) | set(quality_flags))
        for reference in evidence_refs:
            if reference not in event["evidence_refs"]:
                event["evidence_refs"].append(reference)
        if ended:
            event["event_state"] = "ended"
            event["end_at"] = event_time
        if previous is not None and event == previous:
            return previous
        event["event_revision"] += 1
        event["decision_as_of"] = utcstr(context.as_of)
        event["revision_as_of"] = utcstr(context.as_of)
        event["updated_at"] = utcstr(now_utc())
        self.store.put("event", event["event_id"], event, connection=connection)
        self.store.put("event_revision", f"{event['event_id']}@{event['event_revision']:08d}", event, connection=connection)
        return event

    def _rows(self, context, point_id, state, connection):
        start_at = context.as_of - timedelta(minutes=5)
        if state.get("last_event_time"):
            start_at = min(start_at, _time(state["last_event_time"]))
        if state.get("last_context_as_of"):
            start_at = min(start_at, _time(state["last_context_as_of"]))
        else:
            session = self.store.get("session", context.session_id, connection=connection)
            if session is not None:
                if session["source_id"] != context.source_id:
                    raise DomainError("会话与数据源不匹配", code="forbidden", status_code=403)
                start_at = min(start_at, _time(session["start_as_of"]))
        return self._range_rows(context, point_id, start_at, context.as_of, connection)

    def _range_rows(self, context, point_id, start_at, end_at, connection):
        if self.data_service is not None:
            rows, offset = [], 0
            while True:
                result = self.data_service.query_series(context, point_id, start_at,
                    end_at + timedelta(microseconds=1), limit=100000, offset=offset)
                rows.extend(result["rows"])
                offset += len(result["rows"])
                if offset >= result["total"]:
                    return rows
                if not result["rows"]:
                    raise DomainError("监测序列分页未完整返回", code="data_incomplete", status_code=503)
        return [dict(row) for row in connection.execute(select(observation_table).where(
            observation_table.c.source_id == context.source_id,
            observation_table.c.point_id == point_id,
            observation_table.c.event_time >= utcstr(start_at),
            observation_table.c.event_time <= utcstr(end_at),
            observation_table.c.available_at <= utcstr(context.as_of),
        )).mappings()]

    @staticmethod
    def _late_segments(rows, rules):
        """Rebuild an affected interval without changing previously saved events."""
        segments, active, previous_time = [], None, None
        tail = {"coverage_since": None, "high_since": None, "high_refs": []}
        reset_coverage = False
        for row in sorted(rows, key=lambda item: (item["event_time"], item.get("id", item.get("observation_id", "")))):
            moment = utcstr(_time(row["event_time"]))
            valid = _numeric(row.get("value")) and not ({"invalid", "invalid_value"} & set(row.get("quality_flags", [])))
            gap = (_time(moment) - _time(previous_time)).total_seconds() if previous_time else None
            if not valid or gap is None or gap > rules["max_observation_gap_seconds"]:
                tail.update(coverage_since=moment if valid else None, high_since=None, high_refs=[])
                reset_coverage = reset_coverage or previous_time is not None
            previous_time = moment
            if not valid:
                continue
            tail["coverage_since"] = tail["coverage_since"] or moment
            reference = AlertsService._reference(row)
            if row["value"] >= rules["observed_orange_threshold"]:
                tail["high_since"] = tail["high_since"] or moment
                tail["high_refs"].append(reference)
            else:
                tail.update(high_since=None, high_refs=[])
            duration = (_time(moment) - _time(tail["high_since"])).total_seconds() if tail["high_since"] else 0
            level = ("red" if row["value"] >= rules["observed_red_threshold"] else "orange"
                     if duration >= rules["orange_duration_seconds"] else None)
            if level is None and active is None:
                continue
            if active is None:
                active = {"start_at": tail["high_since"] if level == "orange" else moment,
                          "event_time": moment, "end_at": None, "alert_level": level,
                          "evidence_refs": [], "quality_flags": []}
            if level and _RANK[level] > _RANK[active["alert_level"]]:
                active["alert_level"] = level
            active["event_time"] = moment
            for ref in tail["high_refs"] if level else [reference]:
                if ref not in active["evidence_refs"]:
                    active["evidence_refs"].append(ref)
            active["quality_flags"] = sorted(set(active["quality_flags"]) | set(row.get("quality_flags", [])))
            if level is None:
                active["end_at"] = moment
                segments.append(active)
                active = None
        if active:
            segments.append(active)
        merged = []
        for segment in sorted(segments, key=lambda entry: (entry["start_at"], entry["event_time"])):
            # An immediate red excursion and a later five-minute orange trigger
            # can share the same continuous high-value evidence interval.
            if merged and segment["start_at"] < (merged[-1]["end_at"] or merged[-1]["event_time"]):
                previous = merged[-1]
                previous["alert_level"] = max((previous["alert_level"], segment["alert_level"]), key=_RANK.get)
                previous["event_time"] = max(previous["event_time"], segment["event_time"])
                previous["end_at"] = max(previous["end_at"], segment["end_at"]) if previous["end_at"] and segment["end_at"] else None
                previous["evidence_refs"] += [ref for ref in segment["evidence_refs"] if ref not in previous["evidence_refs"]]
                previous["evidence_refs"].sort(key=lambda ref: (ref["event_time"], ref.get("observation_id", "")))
                previous["quality_flags"] = sorted(set(previous["quality_flags"]) | set(segment["quality_flags"]))
            else:
                merged.append(segment)
        return merged, tail, reset_coverage

    def _save_late_segment(self, context, point_id, segment, previous, connection, late_refs=None):
        if previous is None:
            return self._write_event(context, point_id, "observed_exceedance", segment["alert_level"],
                segment["event_time"], segment["start_at"], segment["evidence_refs"],
                segment["quality_flags"], ended=segment["end_at"] is not None, connection=connection)
        event = {**previous, **segment}
        event["quality_flags"] = sorted(set(previous["quality_flags"]) | set(segment["quality_flags"]))
        event["event_state"] = ("ended" if segment["end_at"] is not None else "upgraded"
            if _RANK[segment["alert_level"]] > _RANK[previous["alert_level"]] else "new"
            if previous["event_state"] == "ended" else previous["event_state"])
        if late_refs is not None:
            event["evidence_status"] = "insufficient_after_late_arrival"
            event["late_observation_refs"] = list(previous.get("late_observation_refs", []))
            for ref in late_refs:
                if ref not in event["late_observation_refs"]:
                    event["late_observation_refs"].append(ref)
            if event != previous:
                event["evidence_reassessment"] = {"previous_event_revision": previous["event_revision"],
                    "as_of": utcstr(context.as_of), "reason": "continuous_high_values_interrupted"}
        elif previous.get("evidence_status") == "insufficient_after_late_arrival":
            event["evidence_status"] = "supported_after_late_arrival"
        if event == previous:
            return previous
        event.update(event_revision=previous["event_revision"] + 1, decision_as_of=utcstr(context.as_of),
                     revision_as_of=utcstr(context.as_of), updated_at=utcstr(now_utc()))
        self.store.put("event", event["event_id"], event, connection)
        self.store.put("event_revision", f"{event['event_id']}@{event['event_revision']:08d}", event, connection)
        return event

    def _reassess_late(self, context, point_id, state, rules, connection):
        if not state.get("last_context_as_of") or not state.get("last_event_time"):
            return []
        cursor = _time(state["last_event_time"])
        # Inclusive boundary also handles a new record committed at an identical cutoff.
        late = [dict(row) for row in connection.execute(select(observation_table).where(
            observation_table.c.point_id == point_id, observation_table.c.source_id == context.source_id,
            observation_table.c.available_at >= state["last_context_as_of"],
            observation_table.c.available_at <= utcstr(context.as_of),
            observation_table.c.event_time < utcstr(cursor))).mappings()]
        if not late:
            return []
        existing = [event for event in self.store.list("event", connection)
                    if event["session_id"] == context.session_id and event["source_id"] == context.source_id
                    and event["point_id"] == point_id and event["event_type"] == "observed_exceedance"]
        pad = timedelta(seconds=rules["orange_duration_seconds"])
        windows = []
        for row in sorted(late, key=lambda item: item["event_time"]):
            start, end = _time(row["event_time"]) - pad, min(cursor, _time(row["event_time"]) + pad)
            if windows and start <= windows[-1][1]:
                windows[-1] = (windows[-1][0], max(end, windows[-1][1]))
            else:
                windows.append((start, end))
        intervals = []
        for start, end in windows:
            for event in existing:
                if _time(event["start_at"]) <= end and _time(event.get("end_at") or event["event_time"]) >= start:
                    start = min(start, _time(event["start_at"]))
                    end = min(cursor, max(end, _time(event.get("end_at") or event["event_time"])))
            # Read through the next ending sample, not just the trigger window.
            ending = connection.execute(select(observation_table.c.event_time).where(
                observation_table.c.point_id == point_id, observation_table.c.source_id == context.source_id,
                observation_table.c.event_time > utcstr(end), observation_table.c.event_time <= utcstr(cursor),
                observation_table.c.available_at <= utcstr(context.as_of),
                or_(observation_table.c.value < rules["observed_orange_threshold"], observation_table.c.value.is_(None)))
                .order_by(observation_table.c.event_time).limit(1)).scalar_one_or_none()
            end = _time(ending) if ending else cursor
            if intervals and start <= intervals[-1][1]:
                intervals[-1] = (min(start, intervals[-1][0]), max(end, intervals[-1][1]))
            else:
                intervals.append((start, end))
        refs, used = [], set()
        for start, end in intervals:
            rows = self._range_rows(context, point_id, start, end, connection)
            segments, tail, reset = self._late_segments(rows, rules)
            current = None
            for segment in segments:
                ids = {ref.get("observation_id") for ref in segment["evidence_refs"]}
                candidates = [event for event in existing if event["event_id"] not in used and
                    (ids & {ref.get("observation_id") for ref in event["evidence_refs"]} or
                     event["start_at"] <= (segment["end_at"] or segment["event_time"]) and
                     (event.get("end_at") or event["event_time"]) >= segment["start_at"])]
                previous = min(candidates, key=lambda event: (event["start_at"], event["event_id"])) if candidates else None
                saved = self._save_late_segment(context, point_id, segment, previous, connection)
                used.add(saved["event_id"])
                if previous is None or previous["event_revision"] != saved["event_revision"]:
                    refs.append({"event_id": saved["event_id"], "event_revision": saved["event_revision"]})
                if saved["end_at"] is None:
                    current = saved["event_id"]
            for event in existing:
                if event["event_id"] in used or event["alert_level"] != "orange":
                    continue
                lows = [row for row in late if event["start_at"] <= row["event_time"] <=
                        (event.get("end_at") or event["event_time"]) and _numeric(row.get("value"))
                        and not ({"invalid", "invalid_value"} & set(row.get("quality_flags", [])))
                        and row["value"] < rules["observed_orange_threshold"]]
                if not lows:
                    continue
                closed = {name: event[name] for name in ("start_at", "event_time", "end_at", "alert_level", "evidence_refs", "quality_flags")}
                closed["event_time"] = closed["end_at"] = min(row["event_time"] for row in lows)
                saved = self._save_late_segment(context, point_id, closed, event, connection,
                    late_refs=[self._reference(row) for row in lows])
                used.add(event["event_id"])
                if saved["event_revision"] != event["event_revision"]:
                    refs.append({"event_id": saved["event_id"], "event_revision": saved["event_revision"]})
            if end == cursor:
                if not reset and state.get("coverage_since") and tail["coverage_since"]:
                    tail["coverage_since"] = min(state["coverage_since"], tail["coverage_since"])
                state.update(tail, active_event_id=current)
        return refs

    def evaluate_observed(self, context, point_id, observations=None, rule_version=_VERSION):
        source, _ = self._scope(context, point_id)
        rules = self.get_rules(context.actor, source["face_id"], point_id, rule_version)
        key = f"{context.session_id}:{point_id}:{rule_version}"
        with self.store.transaction() as connection:
            state = self.store.get("alert_observed_state", key, connection=connection) or {}
            if state.get("last_context_as_of") and _time(state["last_context_as_of"]) > context.as_of:
                raise DomainError("已评估较晚时刻，回退需创建新会话", code="conflict", status_code=409)
            rows = self._rows(context, point_id, state, connection) if observations is None else observations
            rows = [row for row in rows if row.get("source_id", context.source_id) == context.source_id
                    and row.get("point_id", point_id) == point_id
                    and _time(row["event_time"]) <= context.as_of
                    and _time(row.get("available_at", row["event_time"])) <= context.as_of]
            event_refs = self._reassess_late(context, point_id, state, rules, connection) if observations is None else []
            by_observation = {ref["observation_id"]: event
                for event in self.store.list("event", connection=connection)
                if event["session_id"] == context.session_id and event["source_id"] == context.source_id
                and event["point_id"] == point_id
                for ref in event["evidence_refs"] if ref.get("observation_id")}
            for row in sorted(rows, key=lambda item: item["event_time"]):
                event_time = utcstr(_time(row["event_time"]))
                last_time = state.get("last_event_time")
                previous = self.store.get("event", state["active_event_id"], connection=connection) if state.get("active_event_id") else None
                if last_time is not None and event_time <= last_time:
                    sample_id = row.get("id", row.get("observation_id"))
                    previous = by_observation.get(sample_id)
                    if previous and row.get("quality_flags"):
                        updated = self._write_event(context, point_id, "observed_exceedance", previous["alert_level"],
                            previous["event_time"], previous["start_at"], [], row["quality_flags"], previous,
                            connection=connection)
                        by_observation[sample_id] = updated
                        if updated["event_revision"] != previous["event_revision"]:
                            event_refs.append({"event_id": updated["event_id"], "event_revision": updated["event_revision"]})
                    elif previous is None and _numeric(row.get("value")) and row["value"] >= rules["observed_red_threshold"] and not ({"invalid", "invalid_value"} & set(row.get("quality_flags", []))):
                        updated = self._write_event(context, point_id, "observed_exceedance", "red", event_time,
                            event_time, [self._reference(row)], row.get("quality_flags", []), connection=connection)
                        if event_time < last_time and state.get("last_value") is not None and state["last_value"] < rules["observed_red_threshold"]:
                            updated = self._write_event(context, point_id, "observed_exceedance", None, last_time,
                                event_time, [], [], updated, ended=True, connection=connection)
                        by_observation[sample_id] = updated
                        event_refs.append({"event_id": updated["event_id"], "event_revision": updated["event_revision"]})
                    continue
                gap = (_time(event_time) - _time(last_time)).total_seconds() if last_time else None
                valid = _numeric(row.get("value")) and not ({"invalid", "invalid_value"} & set(row.get("quality_flags", [])))
                if not valid or gap is None or gap > rules["max_observation_gap_seconds"]:
                    state["coverage_since"] = event_time if valid else None
                    state["high_since"] = None
                    state["high_refs"] = []
                state["last_event_time"] = event_time
                state["last_value"] = row.get("value") if valid else None
                if not valid:
                    continue
                state["coverage_since"] = state.get("coverage_since") or event_time
                reference = self._reference(row)
                if row["value"] >= rules["observed_orange_threshold"]:
                    state["high_since"] = state.get("high_since") or event_time
                    state.setdefault("high_refs", []).append(reference)
                else:
                    state["high_since"] = None
                    state["high_refs"] = []
                duration = (_time(event_time) - _time(state["high_since"])).total_seconds() if state.get("high_since") else 0
                level = "red" if row["value"] >= rules["observed_red_threshold"] else "orange" if duration >= 300 else None
                if level is not None or previous is not None:
                    event = self._write_event(context, point_id, "observed_exceedance", level, event_time,
                        state["high_since"] if level == "orange" else event_time,
                        state["high_refs"] if level else [reference], row.get("quality_flags", []),
                        previous, ended=level is None, connection=connection)
                    event_refs.append({"event_id": event["event_id"], "event_revision": event["event_revision"]})
                    for ref in event["evidence_refs"]:
                        if ref.get("observation_id"):
                            by_observation[ref["observation_id"]] = event
                    state["active_event_id"] = event["event_id"] if level else None
            state["last_context_as_of"] = utcstr(context.as_of)
            self.store.put("alert_observed_state", key, state, connection=connection)
        fresh = state.get("last_event_time") and 0 <= (context.as_of - _time(state["last_event_time"])).total_seconds() <= 60
        active = self.store.get("event", state["active_event_id"]) if state.get("active_event_id") else None
        complete = fresh and state.get("coverage_since") and (_time(state["last_event_time"]) - _time(state["coverage_since"])).total_seconds() >= 300
        level = active["alert_level"] if active and fresh and state.get("last_value") is not None else "blue" if complete else None
        return {"point_id": point_id, "as_of": utcstr(context.as_of), "status": "evaluated" if level else "not_ready",
                "observed_status": "evaluated" if level else "insufficient_data", "forecast_status": None,
                "alert_level": level, "event_refs": event_refs, "missing_fields": [] if level else ["continuous_valid_observations"]}

    def evaluate_forecast(self, context, forecast_run, rule_version=_VERSION):
        point_id = forecast_run["point_id"]
        source, _ = self._scope(context, point_id)
        if forecast_run.get("session_id") != context.session_id or forecast_run.get("source_id") != context.source_id:
            raise DomainError("预测运行不属于当前会话或数据源", code="forbidden", status_code=403)
        rules = self.get_rules(context.actor, source["face_id"], point_id, rule_version)
        values = []
        interval_starts = set()
        step = rules["forecast_interval_seconds"]
        cutoff = context.as_of
        try:
            cutoff = _time(forecast_run.get("cutoff", forecast_run["as_of"]))
            issued_at = _time(forecast_run["as_of"])
            declared_step = forecast_run.get("aggregation", {}).get("seconds", step)
            current_cycle = (type(declared_step) is int and declared_step == step
                and cutoff <= issued_at <= context.as_of
                and 0 <= (context.as_of - cutoff).total_seconds() < step)
        except (AttributeError, TypeError, ValueError):
            current_cycle = False
        if forecast_run.get("status") == "completed" and current_cycle:
            for output in forecast_run.get("outputs", []):
                if output.get("prediction_stage") != "residual_corrected" or output.get("quantity_semantics") != "physical_forecast" or output.get("unit") not in ("%", "%CH4", "%CH4（推定）"):
                    continue
                for interval in output.get("interval_values", []):
                    start, end = _time(interval["start_at"]), _time(interval["end_at"])
                    if start >= cutoff and end <= cutoff + timedelta(seconds=rules["forecast_horizon_seconds"]) and (end - start).total_seconds() == step and _numeric(interval.get("value")):
                        values.append(interval["value"])
                        interval_starts.add(start)
        complete_horizon = interval_starts == {cutoff + timedelta(seconds=step * index)
            for index in range(rules["forecast_horizon_seconds"] // step)}
        if not values or (max(values) < rules["forecast_yellow_threshold"] and not complete_horizon):
            return {"point_id": point_id, "as_of": utcstr(context.as_of), "status": "not_ready", "observed_status": None,
                    "forecast_status": "model_not_ready", "alert_level": None, "event_refs": [], "missing_fields": ["physical_residual_corrected_forecast"]}
        level = "orange" if max(values) >= 0.8 else "yellow" if max(values) >= rules["forecast_yellow_threshold"] else "blue"
        event_refs = []
        with self.store.transaction() as connection:
            previous = next((event for event in self.store.list("event", connection=connection)
                if event["session_id"] == context.session_id and event["source_id"] == context.source_id
                and event["point_id"] == point_id and event["event_type"] == "forecast_exceedance"
                and event["event_state"] != "ended" and _time(event["revision_as_of"]) <= context.as_of), None)
            if level != "blue" or previous:
                event = self._write_event(context, point_id, "forecast_exceedance", None if level == "blue" else level,
                    utcstr(context.as_of), utcstr(context.as_of), [{"kind": "forecast_run", "run_id": forecast_run["run_id"]}],
                    forecast_run.get("quality_flags", []), previous, ended=level == "blue", forecast_run_id=forecast_run["run_id"],
                    connection=connection)
                event_refs.append({"event_id": event["event_id"], "event_revision": event["event_revision"]})
        return {"point_id": point_id, "as_of": utcstr(context.as_of), "status": "evaluated", "observed_status": None,
                "forecast_status": "evaluated", "alert_level": level, "event_refs": event_refs, "missing_fields": []}

    def query_events(self, context, event_id=None, point_id=None, start_at=None, end_at=None, event_type=None, include_revisions=False):
        self._scope(context, point_id)
        events = [event for event in self.store.list("event_revision")
                  if event["session_id"] == context.session_id and event["source_id"] == context.source_id
                  and _time(event["revision_as_of"]) <= context.as_of
                  and (event_id is None or event["event_id"] == event_id)
                  and (point_id is None or event["point_id"] == point_id)
                  and (event_type is None or event["event_type"] == event_type)
                  and (start_at is None or _time(event.get("end_at") or event["event_time"]) >= start_at)
                  and (end_at is None or _time(event["start_at"]) <= end_at)]
        if not include_revisions:
            latest = {}
            for event in events:
                if event["event_id"] not in latest or event["event_revision"] > latest[event["event_id"]]["event_revision"]:
                    latest[event["event_id"]] = event
            events = list(latest.values())
        return sorted(events, key=lambda event: (event["start_at"], event["event_id"], event["event_revision"]))

    def get_event_snapshot(self, context, event_id, event_revision=None):
        events = self.query_events(context, event_id=event_id, include_revisions=True)
        if event_revision is not None:
            events = [event for event in events if event["event_revision"] == event_revision]
        if not events:
            raise DomainError("事件修订不存在或在当前时刻不可见", code="not_found", status_code=404)
        return events[-1]
