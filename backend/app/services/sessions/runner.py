"""会话时钟推进和各模块的单次组合调用。"""

from datetime import datetime, timedelta
import hashlib
import json
from math import floor

from sqlalchemy import func, select

from app.db import observations

from app.schemas.common import DomainError, now_utc, utcstr


class SessionRunner:
    def __init__(self, services):
        self.services = services

    def _submit_brief(self, context, event, schedule):
        store = self.services["store"]
        identity = [context.session_id, context.source_id, event["event_id"], event["alert_level"]]
        key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        reference = {"event_id": event["event_id"], "event_revision": event["event_revision"],
                     "task_id": None, "status": "queued"}
        with store.transaction() as connection:
            existing = store.get("event_brief_trigger", key, connection)
            if existing is not None:
                return existing["reference"]
            claim = {"session_id": context.session_id, "source_id": context.source_id,
                     "event_id": event["event_id"], "alert_level": event["alert_level"],
                     "reference": reference, "created_at": utcstr(now_utc())}
            store.put("event_brief_trigger", key, claim, connection)
        # The claim is committed before scheduling; model work never holds its transaction.
        try:
            investigation = self.services.get("investigation")
            handler = investigation if investigation is not None and investigation.enabled(context) else self.services["event_briefs"]
            brief = handler.submit(context, event["event_id"],
                event_revision=event["event_revision"], schedule=schedule)
            reference = {name: brief.get(name) for name in ("event_id", "event_revision", "task_id", "status")}
        except Exception:
            reference.update(status="failed", error_code="brief_submission_failed")
        claim["reference"] = reference
        store.put("event_brief_trigger", key, claim)
        return reference

    def _new_briefs(self, context, before, events, schedule):
        if schedule is None or self.services.get("event_briefs") is None:
            return []
        briefs = []
        levels = {"blue": 0, "yellow": 1, "orange": 2, "red": 3}
        for event in events:
            prior = before.get(event["event_id"])
            if prior is not None and levels.get(event["alert_level"], -1) <= levels.get(prior["alert_level"], -1):
                continue
            briefs.append(self._submit_brief(context, event, schedule))
        return briefs

    def run_forecast(self, context, point_id, model_id=None, horizon_seconds=None, *, version=None, schedule=None):
        with self.services["sessions"].execution_lock(context.session_id):
            current = self.services["sessions"].get_context(context.actor, context.session_id)
            return self._run_forecast(current, point_id, model_id, horizon_seconds, version=version, schedule=schedule)

    def _run_forecast(self, context, point_id, model_id=None, horizon_seconds=None, *, version=None, schedule=None):
        """Run a requested model, then publish eligible forecast events and briefs."""
        alerts = self.services["alerts"]
        before = {event["event_id"]: event for event in alerts.query_events(context, point_id=point_id)}
        forecast = self.services["prediction"].run_prediction(context, point_id, model_id, horizon_seconds, version=version)
        assessment = alerts.evaluate_forecast(context, forecast) if forecast["status"] == "completed" else None
        events = alerts.query_events(context, point_id=point_id)
        return {**forecast, "assessment": assessment, "event_refs": assessment["event_refs"] if assessment else [],
                "briefs": self._new_briefs(context, before, events, schedule)}

    def run_once(self, actor, session_id, target_as_of=None, schedule=None):
        with self.services["sessions"].execution_lock(session_id):
            return self._run_once(actor, session_id, target_as_of, schedule)

    def _selected_prediction(self, context, point_id, binding):
        prediction = self.services["prediction"]
        if not binding:
            return prediction.run_prediction(context, point_id)
        try:
            model = prediction.get_model(context, binding["model_id"], binding["version"])
        except DomainError as error:
            if error.status_code != 404:
                raise
            model = None
        if model is not None and model["adapter_kind"] == "fedformer_lgbm":
            step = model["aggregation_seconds"]
            cutoff = floor(context.as_of.timestamp() / step) * step
            previous = prediction.list_forecasts(context, point_id, model["model_id"], model["version"], limit=1)
            if (previous and previous[0]["status"] == "completed"
                    and datetime.fromisoformat(previous[0]["cutoff"]).timestamp() == cutoff):
                return {**previous[0], "reused": True}
        return prediction.run_prediction(context, point_id, binding["model_id"], version=binding["version"])

    def _run_once(self, actor, session_id, target_as_of=None, schedule=None):
        sessions = self.services["sessions"]
        session = sessions.get_session(actor, session_id)
        if session["state"] == "ended":
            raise DomainError("会话已结束", "conflict", 409)
        current = datetime.fromisoformat(session["as_of"])
        if target_as_of is None:
            target_as_of = now_utc() if session["mode"] == "realtime" else current + timedelta(seconds=60 * session["speed"])
        sessions.apply_action(actor, session_id, "advance", target_as_of=target_as_of)
        context = sessions.get_context(actor, session_id)
        store = self.services["store"]
        with store.transaction() as connection:
            visible_count = connection.execute(select(func.count()).select_from(observations).where(
                observations.c.source_id == context.source_id,
                observations.c.event_time <= utcstr(context.as_of),
                observations.c.available_at <= utcstr(context.as_of))).scalar_one()
        signature = {"session": session_id, "as_of": context.as_of.isoformat(),
                     "config": context.config_version, "visible_count": visible_count,
                     "prediction_models": session.get("prediction_models", {}),
                     "models": self.services["prediction"].list_models(context)}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        previous = store.get("tick_result", key)
        if previous:
            return {**previous, "session": sessions.get_session(actor, session_id),
                    "briefs": previous.get("briefs", []), "reused": True}
        before = {event["event_id"]: event for event in self.services["alerts"].query_events(context)}
        assessments, forecasts = [], []
        for point in self.services["data"].list_points(actor, context.source_id):
            if point["quantity"] != "methane":
                continue
            assessment = self.services["alerts"].evaluate_observed(context, point["point_id"])
            binding = session.get("prediction_models", {}).get(point["point_id"], {})
            forecast = self._selected_prediction(context, point["point_id"], binding)
            assessments.append(assessment)
            forecasts.append(forecast)
            if forecast["status"] == "completed":
                assessments.append(self.services["alerts"].evaluate_forecast(context, forecast))
        events = self.services["alerts"].query_events(context)
        briefs = self._new_briefs(context, before, events, schedule)
        result = {"session": sessions.get_session(actor, session_id), "assessments": assessments,
                  "forecasts": forecasts, "events": events, "briefs": briefs}
        store.put("tick_result", key, result)
        return result
