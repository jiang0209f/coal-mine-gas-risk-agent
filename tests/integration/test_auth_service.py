import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError
from app.services.auth.service import AuthService


PASSWORD = "test-password-31002"
GENERAL_ACTIONS = (
    "read", "create_session", "chat", "draft_advice", "confirm_event",
    "record_execution", "submit_feedback", "review_case",
)
SAFETY_ACTIONS = (
    "manage_data", "review_reference", "review_advice", "review_experience", "evaluate",
)
ROLES = ("ventilation", "gas_governance", "safety_management")


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 11, 3, 10, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value


@pytest.fixture
def store() -> Store:
    value = Store("sqlite+pysqlite:///:memory:")
    yield value
    value.close()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def service(store: Store, clock: Clock) -> AuthService:
    return AuthService(store, token_ttl_seconds=60, clock=clock)


def create_user(service: AuthService, username: str = "operator") -> dict:
    return service.create_user(username, PASSWORD, ["ventilation"], ["31002"])


def test_create_user_returns_public_fields_and_hashes_each_password_with_fresh_salt(
    service: AuthService, store: Store,
) -> None:
    first = create_user(service)
    second = create_user(service, "other")

    assert set(first) == {"user_id", "username", "roles", "face_ids"}
    assert first["username"] == "operator"
    assert first["roles"] == ["ventilation"]
    assert first["face_ids"] == ["31002"]
    assert first["user_id"] != second["user_id"]
    first_saved = store.get("users", first["user_id"])
    second_saved = store.get("users", second["user_id"])
    assert PASSWORD not in json.dumps(store.list("users"))
    algorithm, rounds, salt, digest = first_saved["password_hash"].split("$")
    assert algorithm == "pbkdf2_sha256"
    assert int(rounds) == 600000
    assert len(bytes.fromhex(salt)) == 16
    assert bytes.fromhex(digest) == hashlib.pbkdf2_hmac(
        "sha256", PASSWORD.encode(), bytes.fromhex(salt), 600000
    )
    assert first_saved["password_hash"] != second_saved["password_hash"]


def test_username_is_unique_without_replacing_original_account(
    service: AuthService, store: Store,
) -> None:
    first = create_user(service)

    with pytest.raises(DomainError) as caught:
        service.create_user("operator", "another-password", ["safety_management"], ["3411"])

    assert caught.value.status_code == 409
    assert len(store.list("users")) == 1
    assert service.login("operator", PASSWORD)["user"] == first


@pytest.mark.parametrize(
    "overrides",
    [
        {"username": ""}, {"username": "   "}, {"password": "12345678901"},
        {"roles": ["admin"]}, {"roles": []}, {"roles": "ventilation"},
        {"face_ids": [""]}, {"face_ids": [31002]}, {"face_ids": "31002"},
    ],
)
def test_create_user_rejects_invalid_input_without_persisting(
    service: AuthService, store: Store, overrides: dict,
) -> None:
    arguments = {
        "username": "operator", "password": PASSWORD,
        "roles": ["ventilation"], "face_ids": ["31002"],
    }
    arguments.update(overrides)

    with pytest.raises(DomainError) as caught:
        service.create_user(**arguments)

    assert caught.value.status_code == 400
    assert store.list("users") == []
    assert PASSWORD not in str(caught.value)


def test_login_resolves_actor_and_stores_only_token_hash(
    service: AuthService, store: Store,
) -> None:
    public = create_user(service)

    result = service.login("operator", PASSWORD)

    assert result["token_type"] == "bearer"
    assert result["user"] == public
    token = result["access_token"]
    assert len(token) >= 43
    actor = service.get_actor(token)
    assert actor == Actor(
        user_id=public["user_id"], roles=("ventilation",), face_ids=("31002",)
    )
    saved = store.get("auth_tokens", hashlib.sha256(token.encode()).hexdigest())
    assert saved["user_id"] == public["user_id"]
    assert token not in json.dumps(store.list("auth_tokens"))
    assert PASSWORD not in json.dumps(result)


@pytest.mark.parametrize(
    "username,password",
    [("operator", "wrong-password"), ("missing", PASSWORD), ("operator", None)],
)
def test_bad_login_uses_same_safe_error_and_creates_no_tokens(
    service: AuthService, store: Store, username: str, password: str | None,
) -> None:
    create_user(service)

    with pytest.raises(DomainError) as caught:
        service.login(username, password)

    assert caught.value.status_code == 401
    assert caught.value.message == "用户名或密码错误"
    assert store.list("auth_tokens") == []
    assert PASSWORD not in str(caught.value)
    assert "wrong-password" not in str(caught.value)


def test_login_issues_independent_tokens_and_logout_revokes_one(
    service: AuthService,
) -> None:
    create_user(service)
    first = service.login("operator", PASSWORD)["access_token"]
    second = service.login("operator", PASSWORD)["access_token"]

    assert first != second
    assert service.logout(first) is None
    with pytest.raises(DomainError) as caught:
        service.get_actor(first)
    assert caught.value.status_code == 401
    assert first not in str(caught.value)
    assert service.get_actor(second).face_ids == ("31002",)


def test_token_expires_at_exact_ttl_boundary(
    service: AuthService, clock: Clock,
) -> None:
    create_user(service)
    token = service.login("operator", PASSWORD)["access_token"]
    clock.value += timedelta(seconds=59, microseconds=999999)
    assert service.get_actor(token).roles == ("ventilation",)

    clock.value += timedelta(microseconds=1)
    with pytest.raises(DomainError) as caught:
        service.get_actor(token)
    assert caught.value.status_code == 401
    assert token not in str(caught.value)


@pytest.mark.parametrize("token", ["unknown-token", "", None])
@pytest.mark.parametrize("operation", ["get_actor", "logout"])
def test_bad_tokens_are_rejected_with_safe_unauthorized_error(
    service: AuthService, token: str | None, operation: str,
) -> None:
    with pytest.raises(DomainError) as caught:
        getattr(service, operation)(token)

    assert caught.value.status_code == 401
    assert "unknown-token" not in str(caught.value)


def test_existing_token_uses_current_user_roles_and_rejects_removed_user(
    service: AuthService, store: Store,
) -> None:
    user = create_user(service)
    token = service.login("operator", PASSWORD)["access_token"]
    saved = store.get("users", user["user_id"])
    saved["roles"] = ["gas_governance"]
    store.put("users", user["user_id"], saved)
    assert service.get_actor(token).roles == ("gas_governance",)

    store.delete("users", user["user_id"])
    with pytest.raises(DomainError) as caught:
        service.get_actor(token)
    assert caught.value.status_code == 401


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("action", GENERAL_ACTIONS)
def test_all_business_roles_can_perform_general_actions(
    service: AuthService, role: str, action: str,
) -> None:
    actor = Actor(user_id="operator", roles=(role,), face_ids=("31002",))

    assert service.authorize(actor, action, "31002") is None


@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("action", SAFETY_ACTIONS)
def test_only_safety_role_can_perform_management_and_review_actions(
    service: AuthService, role: str, action: str,
) -> None:
    actor = Actor(user_id="operator", roles=(role,), face_ids=("31002",))

    if role == "safety_management":
        assert service.authorize(actor, action, "31002") is None
    else:
        with pytest.raises(DomainError) as caught:
            service.authorize(actor, action, "31002")
        assert caught.value.status_code == 403


@pytest.mark.parametrize("action", GENERAL_ACTIONS + SAFETY_ACTIONS)
def test_even_safety_role_cannot_access_an_unassigned_face(
    service: AuthService, action: str,
) -> None:
    actor = Actor(user_id="operator", roles=("safety_management",), face_ids=("31002",))

    with pytest.raises(DomainError) as caught:
        service.authorize(actor, action, "3411")

    assert caught.value.status_code == 403


@pytest.mark.parametrize("action", GENERAL_ACTIONS[1:] + SAFETY_ACTIONS)
def test_system_actor_cannot_perform_human_actions(
    service: AuthService, action: str,
) -> None:
    actor = Actor(
        user_id="system", roles=("safety_management",), face_ids=("31002",), is_system=True,
    )

    with pytest.raises(DomainError) as caught:
        service.authorize(actor, action, "31002")

    assert caught.value.status_code == 403


@pytest.mark.parametrize("roles,action", [((), "read"), (("admin",), "read"), (ROLES, "delete_all")])
def test_unknown_roles_or_actions_are_denied(
    service: AuthService, roles: tuple[str, ...], action: str,
) -> None:
    actor = Actor(user_id="operator", roles=roles, face_ids=("31002",))

    with pytest.raises(DomainError) as caught:
        service.authorize(actor, action, "31002")

    assert caught.value.status_code == 403
