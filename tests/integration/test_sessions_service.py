from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError
from app.services.sessions.service import SessionsService

START = datetime(2026, 10, 1, 8, tzinfo=timezone(timedelta(hours=8)))


@pytest.fixture
def store() -> Store:
    value = Store("sqlite+pysqlite:///:memory:")
    value.put(
        "source", "source-1",
        {"source_id": "source-1", "name": "Source 1", "face_id": "face-1", "timezone": "Asia/Shanghai"},
    )
    value.put(
        "source", "source-2",
        {"source_id": "source-2", "name": "Source 2", "face_id": "face-2", "timezone": "Asia/Shanghai"},
    )
    yield value
    value.close()


@pytest.fixture
def service(store: Store) -> SessionsService:
    return SessionsService(store)


@pytest.fixture
def actor() -> Actor:
    return Actor(user_id="operator-1", roles=("ventilation",), face_ids=("face-1",))


def create_session(service: SessionsService, actor: Actor, **overrides: object) -> dict:
    values = {
        "actor": actor,
        "mode": "historical_replay",
        "source_id": "source-1",
        "start_as_of": START,
    }
    values.update(overrides)
    return service.create_session(**values)


def test_new_session_is_paused_with_uuid_and_normalized_times(
    service: SessionsService, actor: Actor
) -> None:
    before = datetime.now(timezone.utc)
    session = create_session(service, actor)
    after = datetime.now(timezone.utc)

    assert str(UUID(session["session_id"])) == session["session_id"]
    assert session["state"] == "paused"
    assert session["source_id"] == "source-1"
    assert session["face_id"] == "face-1"
    assert session["mode"] == "historical_replay"
    assert session["start_as_of"] == "2026-10-01T00:00:00.000000+00:00"
    assert session["as_of"] == session["start_as_of"]
    assert session["config_version"] == "demo-v1"
    assert session["speed"] == 1
    assert session["parent_session_id"] is None
    assert before <= datetime.fromisoformat(session["updated_at"]) <= after
    assert service.get_session(actor, session["session_id"]) == session


@pytest.mark.parametrize("mode", ["historical_replay", "simulated_stream", "realtime"])
def test_supported_modes_can_create_sessions(
    service: SessionsService, actor: Actor, mode: str
) -> None:
    assert create_session(service, actor, mode=mode)["mode"] == mode


@pytest.mark.parametrize("role", ["ventilation", "gas_governance", "safety_management"])
def test_each_business_role_can_create_sessions(service: SessionsService, role: str) -> None:
    actor = Actor(user_id="user-1", roles=(role,), face_ids=("face-1",))

    assert create_session(service, actor)["state"] == "paused"


@pytest.mark.parametrize("is_system", [False, True])
def test_source_face_authorization_also_applies_to_system_actor(
    service: SessionsService, is_system: bool
) -> None:
    actor = Actor(
        user_id="user-1", roles=("ventilation",), face_ids=("face-2",), is_system=is_system
    )

    with pytest.raises(DomainError) as caught:
        create_session(service, actor)

    assert caught.value.status_code == 403


@pytest.mark.parametrize("is_system", [False, True])
def test_sessions_require_a_business_role(service: SessionsService, is_system: bool) -> None:
    actor = Actor(user_id="user-1", roles=("viewer",), face_ids=("face-1",), is_system=is_system)

    with pytest.raises(DomainError) as caught:
        create_session(service, actor)

    assert caught.value.status_code == 403


@pytest.mark.parametrize(
    ("overrides", "status", "code"),
    [
        ({"source_id": "unknown"}, 404, "not_found"),
        ({"mode": "unknown"}, 400, "invalid_request"),
        ({"start_as_of": datetime(2026, 10, 1)}, 400, "invalid_request"),
    ],
)
def test_invalid_session_parameters_are_rejected(
    service: SessionsService, actor: Actor, overrides: dict, status: int, code: str
) -> None:
    with pytest.raises(DomainError) as caught:
        create_session(service, actor, **overrides)

    assert caught.value.status_code == status
    assert caught.value.code == code


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf"), True])
def test_session_speed_must_be_positive_and_finite(
    service: SessionsService, actor: Actor, speed: float
) -> None:
    with pytest.raises(DomainError) as caught:
        create_session(service, actor, speed=speed)

    assert caught.value.status_code == 400


def test_context_uses_stored_clock_config_and_source(
    service: SessionsService, actor: Actor
) -> None:
    session = create_session(service, actor, config_version="config-test")
    service.apply_action(actor, session["session_id"], "advance", target_as_of=START + timedelta(minutes=1))

    context = service.get_context(actor, session["session_id"])

    assert context.session_id == session["session_id"]
    assert context.source_id == "source-1"
    assert context.config_version == "config-test"
    assert context.mode == "historical_replay"
    assert context.as_of == START + timedelta(minutes=1)
    assert context.actor is actor


@pytest.mark.parametrize("operation", ["get", "context", "action"])
def test_unknown_session_returns_not_found(
    service: SessionsService, actor: Actor, operation: str
) -> None:
    with pytest.raises(DomainError) as caught:
        if operation == "get":
            service.get_session(actor, "missing-session")
        elif operation == "context":
            service.get_context(actor, "missing-session")
        else:
            service.apply_action(actor, "missing-session", "pause")

    assert caught.value.status_code == 404
    assert caught.value.code == "not_found"


@pytest.mark.parametrize("operation", ["get", "context", "action"])
def test_existing_session_is_protected_by_source_face(
    service: SessionsService, actor: Actor, operation: str
) -> None:
    session = create_session(service, actor)
    outsider = Actor(user_id="other", roles=("safety_management",), face_ids=("face-2",))

    with pytest.raises(DomainError) as caught:
        if operation == "get":
            service.get_session(outsider, session["session_id"])
        elif operation == "context":
            service.get_context(outsider, session["session_id"])
        else:
            service.apply_action(outsider, session["session_id"], "resume")

    assert caught.value.status_code == 403


@pytest.mark.parametrize("operation", ["get", "context", "action", "list"])
def test_existing_sessions_require_business_role_for_every_entrypoint(
    service: SessionsService, actor: Actor, operation: str
) -> None:
    session = create_session(service, actor)
    outsider = Actor(user_id="viewer", roles=("viewer",), face_ids=("face-1",))

    with pytest.raises(DomainError) as caught:
        if operation == "get":
            service.get_session(outsider, session["session_id"])
        elif operation == "context":
            service.get_context(outsider, session["session_id"])
        elif operation == "list":
            service.list_sessions(outsider)
        else:
            service.apply_action(outsider, session["session_id"], "resume")

    assert caught.value.status_code == 403


def test_authorization_uses_source_face_instead_of_copied_session_face(
    service: SessionsService, store: Store, actor: Actor
) -> None:
    session = create_session(service, actor)
    session["face_id"] = "face-2"
    store.put("session", session["session_id"], session)
    outsider = Actor(user_id="other", roles=("safety_management",), face_ids=("face-2",))

    with pytest.raises(DomainError) as caught:
        service.get_session(outsider, session["session_id"])

    assert caught.value.status_code == 403


def test_actions_update_state_speed_and_clock_without_changing_configuration(
    service: SessionsService, actor: Actor
) -> None:
    session = create_session(service, actor, config_version="fixed-config")
    session_id = session["session_id"]

    assert service.apply_action(actor, session_id, "resume")["state"] == "running"
    assert service.apply_action(actor, session_id, "set_speed", speed=2.5)["speed"] == 2.5
    advanced = service.apply_action(actor, session_id, "advance", target_as_of=START + timedelta(hours=1))
    assert advanced["as_of"] == "2026-10-01T01:00:00.000000+00:00"
    assert advanced["config_version"] == "fixed-config"
    assert advanced["start_as_of"] == session["start_as_of"]
    assert datetime.fromisoformat(advanced["updated_at"]) >= datetime.fromisoformat(session["updated_at"])
    assert service.apply_action(actor, session_id, "pause")["state"] == "paused"
    assert service.apply_action(actor, session_id, "stop")["state"] == "ended"


@pytest.mark.parametrize("action", ["pause", "resume", "set_speed", "stop", "advance"])
def test_ended_session_rejects_every_action(
    service: SessionsService, actor: Actor, action: str
) -> None:
    session = create_session(service, actor)
    ended = service.apply_action(actor, session["session_id"], "stop")

    with pytest.raises(DomainError) as caught:
        service.apply_action(actor, session["session_id"], action, speed=2, target_as_of=START + timedelta(hours=1))

    assert caught.value.status_code == 409
    assert caught.value.code == "conflict"
    assert service.get_session(actor, session["session_id"]) == ended


def test_rewinding_clock_is_rejected_and_keeps_session_unchanged(
    service: SessionsService, actor: Actor
) -> None:
    session = create_session(service, actor)

    with pytest.raises(DomainError) as caught:
        service.apply_action(actor, session["session_id"], "advance", target_as_of=START - timedelta(seconds=1))

    assert caught.value.status_code == 409
    assert caught.value.code == "conflict"
    assert service.get_session(actor, session["session_id"]) == session


def test_equal_advance_is_allowed(service: SessionsService, actor: Actor) -> None:
    session = create_session(service, actor)

    assert service.apply_action(actor, session["session_id"], "advance", target_as_of=START)["as_of"] == session["as_of"]


@pytest.mark.parametrize(
    ("action", "arguments"),
    [
        ("unknown", {}),
        ("set_speed", {}),
        ("set_speed", {"speed": 0}),
        ("advance", {}),
        ("advance", {"target_as_of": datetime(2026, 10, 1)}),
    ],
)
def test_invalid_action_arguments_leave_session_unchanged(
    service: SessionsService, actor: Actor, action: str, arguments: dict
) -> None:
    session = create_session(service, actor)

    with pytest.raises(DomainError) as caught:
        service.apply_action(actor, session["session_id"], action, **arguments)

    assert caught.value.status_code == 400
    assert caught.value.code == "invalid_request"
    assert service.get_session(actor, session["session_id"]) == session


def test_new_child_session_has_independent_clock_and_no_inherited_records(
    service: SessionsService, store: Store, actor: Actor
) -> None:
    original = create_session(service, actor, config_version="old-config", speed=5)
    service.apply_action(actor, original["session_id"], "advance", target_as_of=START + timedelta(hours=2))
    store.put("prediction", "old-result", {"session_id": original["session_id"], "value": 2.0})

    child = create_session(service, actor, parent_session_id=original["session_id"])

    assert child["session_id"] != original["session_id"]
    assert child["parent_session_id"] == original["session_id"]
    assert child["as_of"] == original["start_as_of"]
    assert child["state"] == "paused"
    assert child["config_version"] == "demo-v1"
    assert child["speed"] == 1
    assert store.list("prediction") == [{"session_id": original["session_id"], "value": 2.0}]


def test_missing_parent_session_is_rejected(service: SessionsService, actor: Actor) -> None:
    with pytest.raises(DomainError) as caught:
        create_session(service, actor, parent_session_id="missing-session")

    assert caught.value.status_code == 404


def test_session_list_only_contains_authorized_faces(
    service: SessionsService, actor: Actor
) -> None:
    visible = create_session(service, actor)
    other = Actor(user_id="other", roles=("ventilation",), face_ids=("face-2",))
    create_session(service, other, source_id="source-2")

    assert service.list_sessions(actor) == [visible]
