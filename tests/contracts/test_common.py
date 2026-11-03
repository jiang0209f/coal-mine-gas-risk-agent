from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc, utcstr


@pytest.fixture
def actor() -> Actor:
    return Actor(user_id="operator-1", roles=("operator",), face_ids=("face-1",))


def test_domain_error_preserves_message_and_defaults() -> None:
    error = DomainError("缺少工作面")

    assert str(error) == "缺少工作面"
    assert error.message == "缺少工作面"
    assert error.code == "invalid_request"
    assert error.status_code == 400
    assert error.missing_fields is None


def test_domain_error_preserves_status_code_and_missing_fields() -> None:
    error = DomainError(
        "缺少参数", code="missing_context", status_code=422, missing_fields=["face_id"]
    )

    assert error.code == "missing_context"
    assert error.status_code == 422
    assert error.missing_fields == ["face_id"]


def test_actor_is_immutable(actor: Actor) -> None:
    with pytest.raises(ValidationError):
        actor.user_id = "another-user"


def test_actor_allows_assigned_face_and_any_matching_role(actor: Actor) -> None:
    actor.require_face("face-1")
    actor.require_role("admin", "operator")


@pytest.mark.parametrize("operation", ["face", "role"])
def test_actor_rejects_unassigned_face_or_role(actor: Actor, operation: str) -> None:
    with pytest.raises(DomainError) as caught:
        if operation == "face":
            actor.require_face("face-2")
        else:
            actor.require_role("admin")

    assert caught.value.status_code == 403
    assert caught.value.code == "forbidden"


def test_actor_without_requested_roles_is_rejected(actor: Actor) -> None:
    with pytest.raises(DomainError) as caught:
        actor.require_role()

    assert caught.value.status_code == 403


@pytest.mark.parametrize("operation", ["face", "role"])
def test_system_actor_requires_explicit_face_and_role_permissions(operation: str) -> None:
    actor = Actor(user_id="system", roles=(), face_ids=(), is_system=True)

    with pytest.raises(DomainError) as caught:
        if operation == "face":
            actor.require_face("face-2")
        else:
            actor.require_role("admin")

    assert caught.value.status_code == 403


@pytest.mark.parametrize("mode", ["historical_replay", "simulated_stream", "realtime"])
def test_execution_context_accepts_modes_and_freezes_fields(actor: Actor, mode: str) -> None:
    context = ExecutionContext(
        session_id="session-1",
        mode=mode,
        source_id="source-1",
        as_of=datetime(2026, 11, 3, 8, tzinfo=timezone.utc),
        actor=actor,
    )

    assert context.config_version == "demo-v1"
    assert context.actor is actor
    with pytest.raises(ValidationError):
        context.source_id = "other-source"


@pytest.mark.parametrize(
    "overrides", [{"mode": "unknown"}, {"as_of": datetime(2026, 11, 3, 8)}]
)
def test_execution_context_rejects_invalid_mode_and_naive_time(
    actor: Actor, overrides: dict
) -> None:
    values = {
        "session_id": "session-1",
        "mode": "realtime",
        "source_id": "source-1",
        "as_of": datetime(2026, 11, 3, 8, tzinfo=timezone.utc),
        "actor": actor,
    }
    values.update(overrides)

    with pytest.raises(ValidationError):
        ExecutionContext(**values)


def test_utcstr_normalizes_offset_with_six_fractional_digits() -> None:
    value = datetime(2026, 11, 3, 10, 5, 2, 1234, tzinfo=timezone(timedelta(hours=8)))

    assert utcstr(value) == "2026-11-03T02:05:02.001234+00:00"


def test_utcstr_rejects_naive_datetimes() -> None:
    with pytest.raises(DomainError) as caught:
        utcstr(datetime(2026, 11, 3))

    assert caught.value.status_code == 400


def test_now_utc_returns_current_aware_utc_time() -> None:
    before = datetime.now(timezone.utc)
    value = now_utc()
    after = datetime.now(timezone.utc)

    assert before <= value <= after
    assert value.tzinfo is timezone.utc
