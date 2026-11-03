"""会话状态与服务端执行时钟。"""

from datetime import datetime
from math import isfinite
from uuid import uuid4

from sqlalchemy.engine import Connection

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc, utcstr

__all__ = ["SessionsService"]

_ROLES = ("ventilation", "gas_governance", "safety_management")
_MODES = ("historical_replay", "simulated_stream", "realtime")
_ACTIONS = ("pause", "resume", "set_speed", "stop", "advance")


class SessionsService:
    def __init__(self, store: Store) -> None:
        self.store = store

    def _source(self, actor: Actor, source_id: str, connection: Connection) -> dict:
        source = self.store.get("source", source_id, connection=connection)
        if source is None:
            raise DomainError("数据源不存在", code="not_found", status_code=404)
        actor.require_face(source["face_id"])
        return source

    def _session(self, actor: Actor, session_id: str, connection: Connection) -> dict:
        session = self.store.get("session", session_id, connection=connection)
        if session is None:
            raise DomainError("会话不存在", code="not_found", status_code=404)
        self._source(actor, session["source_id"], connection)
        return session

    @staticmethod
    def _speed(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
            raise DomainError("播放速度必须为大于零的有限数值")
        return float(value)

    def create_session(
        self,
        actor: Actor,
        mode: str,
        source_id: str,
        start_as_of: datetime,
        config_version: str = "demo-v1",
        speed: float = 1,
        parent_session_id: str | None = None,
    ) -> dict:
        actor.require_role(*_ROLES)
        if mode not in _MODES:
            raise DomainError("运行模式无效")
        initial_time = utcstr(start_as_of)
        if mode == "realtime" and start_as_of > now_utc():
            raise DomainError("实时会话时间不能晚于服务器当前时间")
        session_speed = self._speed(speed)
        with self.store.transaction() as connection:
            source = self._source(actor, source_id, connection)
            if parent_session_id is not None:
                self._session(actor, parent_session_id, connection)
            created_at = utcstr(now_utc())
            session = {
                "session_id": str(uuid4()),
                "mode": mode,
                "source_id": source_id,
                "face_id": source["face_id"],
                "state": "paused",
                "start_as_of": initial_time,
                "as_of": initial_time,
                "config_version": config_version,
                "speed": session_speed,
                "parent_session_id": parent_session_id,
                "created_at": created_at,
                "updated_at": created_at,
            }
            return self.store.put("session", session["session_id"], session, connection=connection)

    def get_session(self, actor: Actor, id: str) -> dict:
        actor.require_role(*_ROLES)
        with self.store.transaction() as connection:
            return self._session(actor, id, connection)

    def get_context(self, actor: Actor, id: str) -> ExecutionContext:
        session = self.get_session(actor, id)
        return ExecutionContext(
            session_id=session["session_id"],
            mode=session["mode"],
            source_id=session["source_id"],
            as_of=datetime.fromisoformat(session["as_of"]),
            config_version=session["config_version"],
            actor=actor,
        )

    def apply_action(
        self,
        actor: Actor,
        id: str,
        action: str,
        speed: float | None = None,
        target_as_of: datetime | None = None,
    ) -> dict:
        actor.require_role(*_ROLES)
        with self.store.transaction() as connection:
            session = self._session(actor, id, connection)
            if session["state"] == "ended":
                raise DomainError("会话已结束", code="conflict", status_code=409)
            if action not in _ACTIONS:
                raise DomainError("会话操作无效")
            if action == "set_speed":
                session["speed"] = self._speed(speed)
            elif action == "advance":
                if target_as_of is None:
                    raise DomainError("推进会话需要目标时间", missing_fields=["target_as_of"])
                target_time = utcstr(target_as_of)
                if session["mode"] == "realtime" and target_as_of > now_utc():
                    raise DomainError("实时会话时间不能晚于服务器当前时间")
                if target_time < session["as_of"]:
                    raise DomainError("回退时间需创建新会话", code="conflict", status_code=409)
                session["as_of"] = target_time
            else:
                session["state"] = {"pause": "paused", "resume": "running", "stop": "ended"}[action]
            session["updated_at"] = utcstr(now_utc())
            return self.store.put("session", id, session, connection=connection)

    def list_sessions(self, actor: Actor) -> list[dict]:
        actor.require_role(*_ROLES)
        with self.store.transaction() as connection:
            sessions = self.store.list("session", connection=connection)
            visible = []
            for session in sessions:
                source = self.store.get("source", session["source_id"], connection=connection)
                if source is not None and source["face_id"] in actor.face_ids:
                    visible.append(session)
            return visible
