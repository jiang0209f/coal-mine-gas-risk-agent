"""本地账户、散列令牌与角色授权。"""

import hashlib
import hmac
import secrets
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from uuid import uuid4

from app.db import Store
from app.schemas.common import Actor, DomainError, now_utc, utcstr

__all__ = ["AuthService"]

_ITERATIONS = 600000
_ROLES = frozenset({"ventilation", "gas_governance", "safety_management"})
_GENERAL_ACTIONS = frozenset({
    "read", "create_session", "chat", "draft_advice", "confirm_event",
    "record_execution", "submit_feedback", "review_case",
})
_SAFETY_ACTIONS = frozenset({
    "manage_data", "review_reference", "review_advice", "review_experience", "evaluate",
})
_DUMMY_HASH = f"pbkdf2_sha256${_ITERATIONS}${'00' * 16}${'00' * 32}"


class AuthService:
    def __init__(
        self,
        store: Store,
        token_ttl_seconds: int = 28800,
        clock: Callable[[], datetime] = now_utc,
    ) -> None:
        if type(token_ttl_seconds) is not int or token_ttl_seconds <= 0:
            raise ValueError("令牌有效期必须为正整数")
        self.store = store
        self.token_ttl_seconds = token_ttl_seconds
        self.clock = clock

    def create_user(
        self,
        username: str,
        password: str,
        roles: Sequence[str],
        face_ids: Sequence[str],
    ) -> dict:
        if not isinstance(username, str) or not username.strip():
            raise DomainError("用户名不能为空")
        username = username.strip()
        if not isinstance(password, str) or len(password) < 12:
            raise DomainError("密码至少需要12个字符")
        role_values = self._string_list(roles, "角色")
        if not role_values or not set(role_values).issubset(_ROLES):
            raise DomainError("角色必须来自通风、瓦斯治理或安全管理角色")
        face_values = self._string_list(face_ids, "工作面编号")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
        user = {
            "user_id": str(uuid4()),
            "username": username,
            "roles": role_values,
            "face_ids": face_values,
            "password_hash": f"pbkdf2_sha256${_ITERATIONS}${salt.hex()}${digest.hex()}",
            "created_at": utcstr(self.clock()),
        }
        with self.store.transaction() as connection:
            if self.store.get("auth_usernames", username, connection=connection) is not None:
                raise DomainError("用户名已存在", code="conflict", status_code=409)
            self.store.put("users", user["user_id"], user, connection=connection)
            self.store.put(
                "auth_usernames", username, {"user_id": user["user_id"]},
                connection=connection,
            )
        return self._public_user(user)

    def login(self, username: str, password: str) -> dict:
        user = None
        if isinstance(username, str):
            with self.store.transaction() as connection:
                index = self.store.get("auth_usernames", username.strip(), connection=connection)
                if index is not None:
                    user = self.store.get("users", index["user_id"], connection=connection)
        encoded = user["password_hash"] if user is not None else _DUMMY_HASH
        valid = self._password_matches(password, encoded)
        if user is None or not valid:
            raise DomainError("用户名或密码错误", code="unauthorized", status_code=401)
        token = secrets.token_urlsafe(32)
        token_hash = self._token_hash(token)
        self.store.put("auth_tokens", token_hash, {
            "token_hash": token_hash,
            "user_id": user["user_id"],
            "expires_at": utcstr(self.clock() + timedelta(seconds=self.token_ttl_seconds)),
        })
        return {"access_token": token, "token_type": "bearer", "user": self._public_user(user)}

    def get_actor(self, token: str) -> Actor:
        with self.store.transaction() as connection:
            saved = self._active_token(token, connection)
            user = self.store.get("users", saved["user_id"], connection=connection)
            if user is None:
                raise self._unauthorized()
        return Actor(
            user_id=user["user_id"], roles=tuple(user["roles"]), face_ids=tuple(user["face_ids"]),
        )

    def logout(self, token: str) -> None:
        with self.store.transaction() as connection:
            self._active_token(token, connection)
            self.store.delete("auth_tokens", self._token_hash(token), connection=connection)

    def authorize(self, actor: Actor, action: str, face_id: str | None = None) -> None:
        if face_id is not None:
            actor.require_face(face_id)
        if actor.is_system and action != "read":
            raise DomainError("系统身份不能执行人工操作", code="forbidden", status_code=403)
        if action in _SAFETY_ACTIONS:
            actor.require_role("safety_management")
        elif action in _GENERAL_ACTIONS:
            actor.require_role(*_ROLES)
        else:
            raise DomainError("无权执行该操作", code="forbidden", status_code=403)

    def _active_token(self, token: str, connection) -> dict:
        token_hash = self._token_hash(token)
        saved = self.store.get("auth_tokens", token_hash, connection=connection)
        if saved is None or not hmac.compare_digest(saved["token_hash"], token_hash):
            raise self._unauthorized()
        expires_at = datetime.fromisoformat(saved["expires_at"])
        if expires_at <= self.clock():
            raise self._unauthorized()
        return saved

    @staticmethod
    def _token_hash(token: str) -> str:
        if not isinstance(token, str) or not token:
            raise AuthService._unauthorized()
        return hashlib.sha256(token.encode()).hexdigest()

    @staticmethod
    def _password_matches(password: str, encoded: str) -> bool:
        try:
            algorithm, iterations, salt_text, digest_text = encoded.split("$")
            if algorithm != "pbkdf2_sha256" or int(iterations) != _ITERATIONS:
                return False
            salt = bytes.fromhex(salt_text)
            expected = bytes.fromhex(digest_text)
            if len(salt) != 16 or len(expected) != 32:
                return False
        except (AttributeError, TypeError, ValueError):
            return False
        supplied = password if isinstance(password, str) else ""
        actual = hashlib.pbkdf2_hmac("sha256", supplied.encode(), salt, _ITERATIONS)
        return hmac.compare_digest(actual, expected) and isinstance(password, str)

    @staticmethod
    def _public_user(user: dict) -> dict:
        return {
            "user_id": user["user_id"], "username": user["username"],
            "roles": list(user["roles"]), "face_ids": list(user["face_ids"]),
        }

    @staticmethod
    def _string_list(values: Sequence[str], label: str) -> list[str]:
        if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise DomainError(f"{label}必须为非空字符串列表")
        return list(dict.fromkeys(values))

    @staticmethod
    def _unauthorized() -> DomainError:
        return DomainError("登录令牌无效或已过期", code="unauthorized", status_code=401)
