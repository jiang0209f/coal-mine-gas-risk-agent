"""共享错误、身份与执行上下文。"""

from datetime import datetime, timezone
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

__all__ = ["Actor", "DomainError", "ExecutionContext", "now_utc", "utcstr"]


class DomainError(Exception):
    def __init__(
        self,
        message: str,
        code: str = "invalid_request",
        status_code: int = 400,
        missing_fields: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.missing_fields = missing_fields


class Actor(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    roles: tuple[str, ...]
    face_ids: tuple[str, ...]
    is_system: bool = False

    def require_face(self, face_id: str) -> None:
        if face_id not in self.face_ids:
            raise DomainError("无权访问该工作面", code="forbidden", status_code=403)

    def require_role(self, *roles: str) -> None:
        if not set(self.roles).intersection(roles):
            raise DomainError("当前角色无权执行该操作", code="forbidden", status_code=403)


class ExecutionContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    mode: Literal["historical_replay", "simulated_stream", "realtime"]
    source_id: str
    as_of: AwareDatetime
    config_version: str = "demo-v1"
    actor: Actor


def utcstr(value: datetime) -> str:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise DomainError("时间必须包含时区")
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
