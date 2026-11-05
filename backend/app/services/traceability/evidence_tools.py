"""Read-only event investigations across explicitly bound historical sources."""

from bisect import bisect_right
from datetime import datetime, timedelta
from math import fsum, isfinite
import re

from sqlalchemy import select

from app.db import observations
from app.schemas.common import DomainError, utcstr

__all__ = ["EvidenceToolsService"]

_CONDITIONS = ("speed", "position", "current_left", "current_right", "wind_speed")


def _time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value


def _group(source):
    return (source.get("metadata") or {}).get("dataset_group") or source.get("dataset_group")


class EvidenceToolsService:
    def __init__(self, store, services):
        self.store, self.services = store, services

    @staticmethod
    def _channel(point):
        channel = (point.get("metadata") or {}).get("channel")
        if channel:
            return channel
        tag = str(point.get("source_tag", point.get("point_id", "")))
        match = re.match(r"(?i)(T[0125])(?:\b|_|-)", tag)
        if match:
            return match[1].upper()
        return {"shearer_speed": "speed", "shearer_position": "position", "wind_speed": "wind_speed"}.get(point.get("quantity"))

    def _scope(self, context, event_id):
        context.actor.require_role("ventilation", "gas_governance", "safety_management")
        source = self.store.get("source", context.source_id)
        if source is None:
            raise DomainError("数据源不存在", "not_found", 404)
        context.actor.require_face(source["face_id"])
        event = self.services["alerts"].get_event_snapshot(context, event_id)
        if (event.get("event_id") != event_id or event.get("source_id") != context.source_id
                or event.get("session_id") != context.session_id or _time(event["revision_as_of"]) > context.as_of):
            raise DomainError("事件与当前执行范围不符", "forbidden", 403)
        center = _time(event.get("start_at") or event["event_time"])
        lower, upper = center-timedelta(minutes=10), min(center+timedelta(minutes=10), context.as_of)
        if not _group(source):
            return event, {}, lower, upper, ["dataset_group"]
        source_ids = {item["source_id"] for item in self.store.list("source")
                      if item.get("face_id") == source["face_id"] and _group(item) == _group(source)}
        points = {}
        candidates = sorted(self.store.list("point"), key=lambda item: (item.get("source_id") != context.source_id, item["point_id"]))
        for point in candidates:
            channel = self._channel(point)
            if point.get("source_id") in source_ids and point.get("face_id") == source["face_id"] and channel:
                points.setdefault(channel, point)
        return event, points, lower, upper, []

    def _rows(self, context, point, lower, upper):
        statement = select(observations).where(
            observations.c.source_id == point["source_id"], observations.c.point_id == point["point_id"],
            observations.c.event_time >= utcstr(lower), observations.c.event_time <= utcstr(upper),
            observations.c.event_time <= utcstr(context.as_of), observations.c.available_at <= utcstr(context.as_of),
        ).order_by(observations.c.event_time, observations.c.id)
        with self.store.transaction() as connection:
            rows = [dict(row) for row in connection.execute(statement).mappings()]
        return [row for row in rows if isinstance(row.get("value"), (int, float)) and isfinite(row["value"])]

    @staticmethod
    def _summary(rows):
        values = [row["value"] for row in rows]
        return {"sample_count": len(values), "minimum": min(values) if values else None,
                "maximum": max(values) if values else None, "mean": fsum(values)/len(values) if values else None,
                "first": values[0] if values else None, "last": values[-1] if values else None}

    @staticmethod
    def _refs(rows):
        if not rows:
            return []
        selected = [rows[0], max(rows, key=lambda row: row["value"]), rows[-1], *rows[:9]]
        output, seen = [], set()
        for row in selected:
            if row["id"] not in seen:
                seen.add(row["id"])
                output.append({"kind": "observation", "observation_id": row["id"], "point_id": row["point_id"],
                    "source_file_id": row["source_file_id"], "row_number": row["row_number"],
                    "event_time": row["event_time"], "available_at": row["available_at"]})
        return output

    @staticmethod
    def _result(metrics, references, missing):
        missing = list(dict.fromkeys(missing))
        return {"status": "not_ready" if not metrics else "partial" if missing else "completed",
                "metrics": metrics, "evidence_refs": references, "missing_fields": missing}

    def compare_points(self, context, event_id):
        _, points, lower, upper, missing = self._scope(context, event_id)
        if missing:
            return self._result({}, [], missing)
        metrics, references = [], []
        for channel in ("T0", "T1", "T2", "T5"):
            point = points.get(channel)
            rows = self._rows(context, point, lower, upper) if point else []
            if not rows:
                missing.append(channel)
                continue
            peak = max(rows, key=lambda row: row["value"])
            amplitude = max(row["value"] for row in rows)-min(row["value"] for row in rows)
            rise = next((row for row in rows if amplitude > 0 and row["value"] > rows[0]["value"]+0.1*amplitude), None)
            metrics.append({"channel": channel, "point_id": point["point_id"], **self._summary(rows),
                "peak": peak["value"], "peak_time": peak["event_time"], "rise_time": rise["event_time"] if rise else None,
                "unit": point.get("unit"), "unit_status": point.get("unit_status"), "lineage": point.get("lineage")})
            references.extend(self._refs(rows))
        baseline = next((item for item in metrics if item["channel"] == "T0"), None)
        for item in metrics:
            for name in ("peak", "rise"):
                item[name+"_delay_seconds"] = ((_time(item[name+"_time"])-_time(baseline[name+"_time"])).total_seconds()
                    if baseline and item[name+"_time"] and baseline[name+"_time"] else None)
        return self._result({"window_start": utcstr(lower), "window_end": utcstr(upper), "points": metrics,
            "rise_criterion": "first_above_initial_plus_10pct_range", "delay_reference": "T0"}, references, missing)

    @staticmethod
    def _at(rows, stamp, timestamps):
        index = bisect_right(timestamps, stamp)-1
        return rows[index]["value"] if index >= 0 and stamp-timestamps[index] <= timedelta(seconds=60) else None

    @staticmethod
    def _state(values):
        speed = values.get("speed")
        currents = [values[key] for key in ("current_left", "current_right") if values.get(key) is not None]
        if speed is None or not currents:
            return "unknown"
        moving, powered = abs(speed) > 1e-6, any(value > 1e-6 for value in currents)
        return ("moving" if moving else "stationary")+("_with_current" if powered else "_without_current")

    def _conditions(self, context, points, lower, upper):
        data, missing, references = {}, [], []
        for channel in _CONDITIONS:
            rows = self._rows(context, points[channel], lower, upper) if channel in points else []
            data[channel] = rows
            if not rows:
                missing.append(channel)
            references.extend(self._refs(rows))
        return data, references, missing

    def get_condition_timeline(self, context, event_id):
        _, points, lower, upper, missing = self._scope(context, event_id)
        if missing:
            return self._result({}, [], missing)
        data, references, missing = self._conditions(context, points, lower, upper)
        clocks = {channel: [_time(row["event_time"]) for row in rows] for channel, rows in data.items()}
        timestamps = sorted({row["event_time"] for rows in data.values() for row in rows})
        timeline = []
        for stamp in timestamps[:160]:
            values = {channel: self._at(rows, _time(stamp), clocks[channel]) for channel, rows in data.items()}
            timeline.append({"event_time": stamp, **values, "operating_state": self._state(values)})
        return self._result({"window_start": utcstr(lower), "window_end": utcstr(upper), "timeline": timeline,
            "channels": {channel: {**self._summary(rows), "unit": points.get(channel, {}).get("unit"),
                "unit_status": points.get(channel, {}).get("unit_status")} for channel, rows in data.items()},
            "state_basis": "speed_and_motor_current_proxy", "alignment_max_age_seconds": 60}, references, [*missing, "cutting_state"])

    def get_face_statistics(self, context, event_id):
        event, points, lower, upper, missing = self._scope(context, event_id)
        if missing:
            return self._result({}, [], missing)
        lower, upper = context.as_of-timedelta(hours=24), context.as_of
        point = points.get("T0")
        rows = self._rows(context, point, lower, upper) if point else []
        data, references, condition_missing = self._conditions(context, points, lower, upper)
        clocks = {channel: [_time(row["event_time"]) for row in values] for channel, values in data.items()}
        rules_method = getattr(self.services.get("alerts"), "get_rules", None)
        rules = rules_method(context.actor, self.store.get("source", context.source_id)["face_id"], event["point_id"], event["rule_version"]) if callable(rules_method) else {}
        threshold = rules.get("observed_orange_threshold")
        if not isinstance(threshold, (int, float)) or not isfinite(threshold):
            threshold = None
            missing.append("high_threshold")
        groups = {}
        for row in rows:
            state = self._state({channel: self._at(values, _time(row["event_time"]), clocks[channel]) for channel, values in data.items()})
            entry = groups.setdefault(state, {"operating_state": state, "sample_count": 0, "high_count": 0})
            entry["sample_count"] += 1
            entry["high_count"] += int(threshold is not None and row["value"] >= threshold)
        for entry in groups.values():
            entry["high_ratio"] = entry["high_count"]/entry["sample_count"] if threshold is not None else None
            if threshold is None:
                entry["high_count"] = None
        if not rows:
            missing.append("T0")
        return self._result({"window_start": utcstr(lower), "window_end": utcstr(upper), "window_basis": "past_24_hours", "groups": list(groups.values()),
            "threshold": threshold, "threshold_basis": "event_rule_orange", "rule_status": rules.get("status"),
            "state_basis": "speed_and_motor_current_proxy"}, [*references, *self._refs(rows)], [*missing, *condition_missing, "cutting_state"])

    def find_similar_events(self, context, event_id):
        event, _, _, _, missing = self._scope(context, event_id)
        if missing:
            return self._result({}, [], missing)
        source = self.store.get("source", context.source_id)
        sources = {row["source_id"] for row in self.store.list("source")
                   if row.get("face_id") == source["face_id"] and _group(row) == _group(source)}
        latest = {}
        for row in self.store.list("event_revision"):
            if (row.get("source_id") not in sources or row.get("event_id") == event_id
                    or row.get("event_type") != event["event_type"] or _time(row["revision_as_of"]) > context.as_of
                    or _time(row["start_at"]) >= _time(event["start_at"])):
                continue
            if row.get("event_revision", 0) >= latest.get(row["event_id"], {}).get("event_revision", -1):
                latest[row["event_id"]] = row
        reviews = {row["event_id"]: row for row in self.store.list("reviewed_case_snapshot")
            if row.get("source_id") in sources and row.get("face_id") == source["face_id"]
            and row.get("operation_context") == "field_record" and row.get("reviewed_at") and row.get("data_as_of")
            and _time(row["reviewed_at"]) <= context.as_of and _time(row["data_as_of"]) <= context.as_of}
        def visible(value):
            try:
                return value is not None and _time(value) <= context.as_of
            except (ValueError, TypeError):
                return False

        def operation_allowed(row):
            return (row.get("operation_context") in ("field_record", "rehearsal")
                    and not (context.mode == "realtime" and row.get("operation_context") == "rehearsal"))

        approved_reports = {}
        for report in self.store.list("investigation_report"):
            review = report.get("review") or {}
            past_event = report.get("event_overview") or latest.get(report.get("event_id"), {})
            if (report.get("state") != "approved" or report.get("source_id") not in sources
                    or report.get("face_id") != source["face_id"] or not operation_allowed(report)
                    or not visible(report.get("as_of")) or review.get("decision") != "approved"
                    or not visible(review.get("data_as_of")) or report.get("event_id") == event_id
                    or past_event.get("event_type") != event["event_type"] or not past_event.get("start_at")
                    or _time(past_event["start_at"]) >= _time(event["start_at"])):
                continue
            previous = approved_reports.get(report["event_id"])
            if previous is None or _time(report["as_of"]) > _time(previous["as_of"]):
                approved_reports[report["event_id"]] = report
                latest.setdefault(report["event_id"], past_event)
        found, references = [], []
        for row in sorted(latest.values(), key=lambda item: (item.get("alert_level") != event.get("alert_level"), -_time(item["start_at"]).timestamp()))[:12]:
            review = reviews.get(row["event_id"])
            report = approved_reports.get(row["event_id"])
            conclusion = next((action["content"] for action in reversed(review.get("actions", []))
                               if action.get("action_type") == "review_case" and isinstance(action.get("content"), str)), None) if review else None
            conclusions = [{"cause": item.get("cause"), "assessment": item.get("support_status"), "rationale": item.get("rationale")}
                           for item in (report.get("hypotheses", []) if report else []) if isinstance(item, dict)
                           and isinstance(item.get("cause"), str) and item.get("support_status") in ("supported", "opposed", "undetermined")]
            if conclusions:
                conclusion = "；".join(f"{item['cause']}（{item['assessment']}）" for item in conclusions)
            found.append({"event_id": row["event_id"], "event_type": row["event_type"], "alert_level": row.get("alert_level"),
                "start_at": row["start_at"], "reviewed_conclusion": conclusion, "conclusions": conclusions,
                "reviewed_source": "investigation_report" if report else "reviewed_case" if review else None,
                "operation_context": (report or review or {}).get("operation_context")})
            references.append({"kind": "event", "event_id": row["event_id"], "event_revision": row.get("event_revision"), "revision_as_of": row["revision_as_of"]})
            if review:
                references.append({"kind": "reviewed_case", "case_id": review.get("case_id"), "review_action_ref": review.get("review_action_ref")})
            if report:
                references.append({"kind": "investigation_report", "report_id": report["report_id"], "event_id": row["event_id"],
                    "as_of": report["as_of"], "review_data_as_of": report["review"]["data_as_of"]})
        experiences = []
        knowledge = self.services.get("knowledge")
        for experience in self.store.list("control_experience"):
            audits = [item for item in experience.get("history", []) if item.get("action") == "review_experience"]
            review_as_of = audits[-1].get("data_as_of") if audits else experience.get("reviewed_at")
            plan = self.store.get("control_plan", experience.get("plan_id", "")) or {}
            if (experience.get("state") != "approved" or experience.get("source_id") not in sources
                    or experience.get("face_id") != source["face_id"] or not operation_allowed(experience)
                    or not visible((experience.get("effect") or {}).get("as_of")) or not visible(review_as_of)
                    or plan.get("event_id") not in latest or not visible(plan.get("as_of", review_as_of))):
                continue
            valid_refs = []
            for reference_id in experience.get("reference_ids", []):
                try:
                    if knowledge is not None:
                        knowledge._approved_fragment(reference_id, source["face_id"], context.mode == "realtime")
                        valid_refs.append(reference_id)
                except DomainError:
                    continue
            if not valid_refs:
                missing.append("approved_experience_references")
                continue
            experiences.append({"experience_id": experience["experience_id"], "event_id": plan["event_id"], "state": "approved",
                "operation_context": experience["operation_context"], "effect_status": experience["effect"].get("status"),
                "cause_ids": [item for item in experience.get("cause_ids", []) if item in {
                    "ventilation_abnormal", "gas_extraction_inadequate", "goaf_gas_emission", "adjacent_seam_gas", "coal_cutting"}],
                "reference_count": len(valid_refs)})
            references.append({"kind": "control_experience", "experience_id": experience["experience_id"], "plan_id": experience["plan_id"]})
            references.extend({"kind": "knowledge", "reference_id": reference_id} for reference_id in valid_refs)
        if not found:
            missing.append("historical_similar_events")
        if not any(item["reviewed_conclusion"] for item in found):
            missing.append("reviewed_conclusions")
        return self._result({"events": found, "experiences": experiences[:12], "experience_count": len(experiences),
            "similarity_basis": "same_event_type_then_same_level_then_recent"}, references, missing)
