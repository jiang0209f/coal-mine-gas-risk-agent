"""读取事件修订的原始依据与工况提示。"""

from datetime import datetime

from sqlalchemy import select

from app.db import Store, observations
from app.schemas.common import DomainError

__all__ = ["TraceabilityService"]


class TraceabilityService:
    def __init__(self, store: Store, alerts_service, data_service=None, prediction_service=None, trained_models=None) -> None:
        self.store = store
        self.alerts_service = alerts_service
        self.data_service = data_service
        self.prediction_service = prediction_service
        self.trained_models = trained_models

    def get_event_evidence(self, context, event_id, event_revision=None):
        snapshot = self.alerts_service.get_event_snapshot(context, event_id, event_revision)
        cutoff = snapshot["revision_as_of"]
        frozen_context = context.model_copy(update={"as_of": datetime.fromisoformat(cutoff)})
        ids = [reference["observation_id"] for reference in snapshot["evidence_refs"]
               if reference.get("kind") == "observation" and reference.get("observation_id")]
        late_refs = snapshot.get("late_observation_refs", [])
        late_ids = [reference["observation_id"] for reference in late_refs
                    if reference.get("kind") == "observation" and reference.get("observation_id")]
        available = []
        if ids or late_ids:
            with self.store.engine.connect() as connection:
                available = [dict(row) for row in connection.execute(select(observations).where(
                    observations.c.id.in_(set(ids + late_ids)), observations.c.source_id == context.source_id,
                    observations.c.point_id == snapshot["point_id"],
                    observations.c.event_time <= cutoff, observations.c.available_at <= cutoff,
                ).order_by(observations.c.event_time, observations.c.id)).mappings()]
        fragments = [row for row in available if row["id"] in ids]
        late_fragments = [row for row in available if row["id"] in late_ids]
        valid_late = {row["id"] for row in late_fragments}
        evidence_refs = list(snapshot["evidence_refs"])
        for reference in late_refs:
            if reference.get("observation_id") in valid_late and reference not in evidence_refs:
                evidence_refs.append(reference)
        forecast = None
        run_id = snapshot.get("forecast_run_id")
        if run_id:
            if self.prediction_service is not None:
                forecast = self.prediction_service.get_forecast(frozen_context, run_id)
            else:
                forecast = self.store.get("forecast_run", run_id)
                if forecast and (forecast.get("session_id") != context.session_id
                    or forecast.get("source_id") != context.source_id or forecast["as_of"] > cutoff):
                    raise DomainError("预测运行不属于当前事件快照", code="forbidden", status_code=403)
        hints = []
        for flag in ("suspected_calibration", "planned_calibration", "confirmed_calibration"):
            if flag in snapshot["quality_flags"]:
                hints.append({"kind": flag, "text": {
                    "suspected_calibration": "存在疑似标校标记，关联核对依据为标校台账与原始监测片段。",
                    "planned_calibration": "事件处于登记的计划标校窗口。",
                    "confirmed_calibration": "事件带有已确认标校记录。",
                }[flag], "evidence_refs": snapshot["evidence_refs"]})
        source = self.store.get("source", context.source_id)
        rules = self.alerts_service.get_rules(context.actor, source["face_id"], snapshot["point_id"], snapshot["rule_version"])
        missing = []
        if len(fragments) < len(set(ids)):
            missing.append("observation_fragments")
        if len(late_fragments) < len(set(late_ids)):
            missing.append("late_observation_fragments")
        if run_id and forecast is None:
            missing.append("forecast_run")
        supported = snapshot.get("evidence_status") != "insufficient_after_late_arrival"
        if not supported:
            missing.append("evidence_supported")
        result = {"event_snapshot": snapshot, "snapshot_as_of": cutoff,
                "observation_fragments": fragments, "late_observation_fragments": late_fragments, "forecast_run": forecast,
                "rules": rules, "condition_hints": hints, "causal_status": "evidence_only",
                "evidence_refs": evidence_refs, "missing_fields": missing}
        if not supported:
            result["candidate_attribution"] = {"status": "not_ready", "missing_fields": ["evidence_supported"],
                                               "causal_status": "evidence_only"}
        elif self.trained_models is not None:
            attribution = self.trained_models.attribute_event(frozen_context, snapshot, fragments)
            result["candidate_attribution"] = attribution
            if attribution["status"] == "completed":
                result["causal_status"] = "candidate_attribution"
        return result
