"""会话时钟推进和各模块的单次组合调用。"""

from datetime import datetime, timedelta
import hashlib
import json

from sqlalchemy import func, select

from app.db import observations

from app.schemas.common import DomainError, now_utc, utcstr


class SessionRunner:
    def __init__(self, services):
        self.services = services

    def run_once(self, actor, session_id, target_as_of=None):
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
                     "models": self.services["prediction"].list_models(context)}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        previous = store.get("tick_result", key)
        if previous:
            return {**previous, "session": sessions.get_session(actor, session_id), "reused": True}
        assessments, forecasts = [], []
        for point in self.services["data"].list_points(actor, context.source_id):
            if point["quantity"] != "methane":
                continue
            assessment = self.services["alerts"].evaluate_observed(context, point["point_id"])
            forecast = self.services["prediction"].run_prediction(context, point["point_id"])
            assessments.append(assessment)
            forecasts.append(forecast)
            if forecast["status"] == "completed":
                assessments.append(self.services["alerts"].evaluate_forecast(context, forecast))
        result = {"session": sessions.get_session(actor, session_id), "assessments": assessments,
                  "forecasts": forecasts, "events": self.services["alerts"].query_events(context)}
        store.put("tick_result", key, result)
        return result
