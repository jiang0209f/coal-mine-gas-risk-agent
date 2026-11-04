"""Replay controls keep immutable, scoped records without rewinding old sessions."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError
from app.services.sessions.service import SessionsService


@pytest.fixture
def replay():
    store = Store("sqlite+pysqlite:///:memory:")
    for source in ("source", "other"):
        store.put("source", source, {"source_id": source, "face_id": "31002"})
    actor = Actor(user_id="qa", roles=("safety_management",), face_ids=("31002",))
    yield store, SessionsService(store), actor, datetime(2022, 2, 5, tzinfo=timezone.utc)
    store.close()


def test_replay_controls_preserve_before_after_clock_and_order(replay):
    store, sessions, actor, start = replay
    session = sessions.create_session(actor, "historical_replay", "source", start)
    sid = session["session_id"]
    sessions.apply_action(actor, sid, "resume")
    sessions.apply_action(actor, sid, "set_speed", speed=2)
    sessions.apply_action(actor, sid, "advance", target_as_of=start + timedelta(minutes=2))
    sessions.apply_action(actor, sid, "pause")
    sessions.apply_action(actor, sid, "stop")

    records = sorted(store.list("session_action"), key=lambda row: row["sequence"])
    assert [row["action_type"] for row in records] == ["create", "resume", "set_speed", "advance", "pause", "stop"]
    assert [row["sequence"] for row in records] == list(range(1, 7))
    assert all(row["session_id"] == sid and row["source_id"] == "source" and row["face_id"] == "31002" for row in records)
    assert all(row["actor_id"] == actor.user_id and row["occurred_at"] and row["data_as_of"] for row in records)
    assert records[2]["before"]["speed"] == 1 and records[2]["after"]["speed"] == 2
    assert datetime.fromisoformat(records[3]["before"]["as_of"]) == start
    assert datetime.fromisoformat(records[3]["after"]["as_of"]) == start + timedelta(minutes=2)
    assert records[-1]["after"]["state"] == "ended"
    previous = deepcopy(records)
    records[0]["after"]["speed"] = 999
    assert sorted(store.list("session_action"), key=lambda row: row["sequence"]) == previous


def test_failed_rewind_creates_no_audit_and_parent_replay_is_separate(replay):
    store, sessions, actor, start = replay
    original = sessions.create_session(actor, "historical_replay", "source", start)
    sid = original["session_id"]
    sessions.apply_action(actor, sid, "advance", target_as_of=start + timedelta(minutes=10))
    previous = deepcopy(store.list("session_action"))
    with pytest.raises(DomainError):
        sessions.apply_action(actor, sid, "advance", target_as_of=start)
    assert store.list("session_action") == previous
    child = sessions.create_session(actor, "historical_replay", "source", start, parent_session_id=sid)
    assert child["session_id"] != sid and child["parent_session_id"] == sid
    assert sessions.get_session(actor, sid)["as_of"] != child["as_of"]
    assert any(row["session_id"] == child["session_id"] and row["action_type"] == "create" for row in store.list("session_action"))


def test_replay_parent_cannot_reference_another_source(replay):
    _, sessions, actor, start = replay
    original = sessions.create_session(actor, "historical_replay", "other", start)
    with pytest.raises(DomainError) as caught:
        sessions.create_session(actor, "historical_replay", "source", start, parent_session_id=original["session_id"])
    assert caught.value.status_code == 403
