"""Arrival-time incremental reassessment preserves historical event identities."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.db import Store
from app.schemas.common import Actor, utcstr
from app.services.alerts.service import AlertsService
from app.services.cases.service import CasesService
from app.services.data.service import DataService
from app.services.sessions.service import SessionsService

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def parts(tmp_path):
    store = Store("sqlite+pysqlite:///:memory:")
    data = DataService(store, tmp_path / "imports")
    data.register_source("s", "Synthetic source", "31002")
    data.register_point("p", "s", "T0", "T0", "methane", "%", unit_status="confirmed")
    actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
    sessions = SessionsService(store)
    session = sessions.create_session(actor, "historical_replay", "s", START)
    service = AlertsService(store, data)

    def ingest(minute, value, available=None, identity=None):
        data.ingest_readings(actor, "s", [{"reading_id": identity or str(minute), "point_id": "p",
            "event_time": utcstr(START + timedelta(minutes=minute)), "raw_value": value}],
            received_at=START + timedelta(minutes=minute if available is None else available))

    def context(minute):
        sessions.apply_action(actor, session["session_id"], "advance", target_as_of=START + timedelta(minutes=minute))
        return sessions.get_context(actor, session["session_id"])

    def assess(minute):
        return service.evaluate_observed(context(minute), "p")

    def events(minute):
        return service.query_events(context(minute))

    yield SimpleNamespace(store=store, data=data, actor=actor, service=service, sessions=sessions,
        session=session, ingest=ingest, context=context, assess=assess, events=events)
    store.close()


@pytest.mark.parametrize(("late", "level"), [([(10, .9)], "red"), ([(t, .70) for t in range(10, 16)], "orange")])
def test_old_newly_available_red_and_orange_are_detected(parts, late, level):
    parts.ingest(20, .1)
    parts.assess(20)
    for minute, value in late:
        parts.ingest(minute, value, available=21)
    parts.assess(21)
    events = parts.events(21)
    assert len(events) == 1 and events[0]["alert_level"] == level
    assert events[0]["start_at"] == utcstr(START + timedelta(minutes=10))
    assert events[0]["end_at"] == utcstr(START + timedelta(minutes=20))
    assert events[0]["revision_as_of"] == utcstr(START + timedelta(minutes=21))
    assert parts.service.query_events(parts.context(21).model_copy(update={"as_of": START + timedelta(minutes=20)})) == []


def test_multiple_late_batches_can_complete_the_same_high_segment(parts):
    parts.ingest(20, .1)
    parts.assess(20)
    for minute in range(10, 13):
        parts.ingest(minute, .70, available=21)
    parts.assess(21)
    assert parts.events(21) == []
    for minute in range(13, 16):
        parts.ingest(minute, .70, available=22)
    parts.assess(22)
    assert len(parts.events(22)) == 1
    assert parts.events(22)[0]["alert_level"] == "orange"
    assert len(parts.events(22)[0]["evidence_refs"]) >= 6


def test_late_sample_fills_a_gap_and_reconstructs_continuous_orange(parts):
    for minute in (10, 11, 13, 14, 15):
        parts.ingest(minute, .70)
    parts.ingest(20, .1)
    parts.assess(20)
    assert parts.events(20) == []
    parts.ingest(12, .70, available=21)
    parts.assess(21)
    event = parts.events(21)[0]
    assert event["alert_level"] == "orange"
    assert event["start_at"] == utcstr(START + timedelta(minutes=10))


def test_late_red_reuses_existing_orange_id_and_retains_manual_snapshot(parts):
    for minute in range(10, 16):
        parts.ingest(minute, .70)
    parts.ingest(20, .1)
    parts.assess(20)
    original = deepcopy(parts.events(20)[0])
    cases = CasesService(parts.store, parts.service)
    cases.record_action(parts.actor, parts.context(20), original["event_id"],
        {"action_type": "confirm_event", "content": "已确认原预警", "operation_context": "rehearsal"})
    saved_actions = deepcopy(parts.store.list("case_action"))
    parts.ingest(13.5, .9, available=21)
    parts.assess(21)
    event = parts.events(21)[0]
    assert len(parts.events(21)) == 1
    assert event["event_id"] == original["event_id"] and event["alert_level"] == "red"
    assert event["event_revision"] > original["event_revision"]
    assert parts.service.get_event_snapshot(parts.context(21), original["event_id"], original["event_revision"]) == original
    assert parts.store.list("case_action") == saved_actions
    assert cases.get_case(parts.context(21), original["event_id"])["case"]["phase"] == "confirmed"


def test_late_normal_closes_original_segment_and_keeps_later_red_separate(parts):
    parts.ingest(10, .9)
    parts.ingest(20, .9)
    parts.assess(20)
    original = deepcopy(parts.events(20)[0])
    parts.ingest(15, .1, available=21)
    parts.assess(21)
    events = parts.events(21)
    assert len(events) == 2
    first = next(event for event in events if event["event_id"] == original["event_id"])
    assert first["event_state"] == "ended" and first["end_at"] == utcstr(START + timedelta(minutes=15))
    second = next(event for event in events if event["event_id"] != original["event_id"])
    assert second["start_at"] == utcstr(START + timedelta(minutes=20))
    assert second["event_state"] != "ended" and second["alert_level"] == "red"
    assert parts.service.get_event_snapshot(parts.context(21), first["event_id"], original["event_revision"]) == original
    state = parts.store.list("alert_observed_state")[0]
    assert state["active_event_id"] == second["event_id"]


def test_adjacent_existing_events_keep_their_ids_and_unaffected_revision(parts):
    for minute, value in ((10, .9), (11, .1), (20, .9), (21, .1), (30, .1)):
        parts.ingest(minute, value)
    parts.assess(30)
    original = deepcopy(parts.events(30))
    parts.ingest(10.5, .9, available=31)
    parts.assess(31)
    events = parts.events(31)
    assert {event["event_id"] for event in events} == {event["event_id"] for event in original}
    unaffected = next(event for event in original if event["start_at"] == utcstr(START + timedelta(minutes=20)))
    assert next(event for event in events if event["event_id"] == unaffected["event_id"]) == unaffected
    saved = deepcopy(parts.store.list("event_revision"))
    parts.assess(31)
    assert parts.store.list("event_revision") == saved


def test_late_normal_invalidates_old_orange_support_without_erasing_history(parts):
    for minute in range(10, 16):
        parts.ingest(minute, .70)
    parts.assess(15)
    original = deepcopy(parts.events(15)[0])
    parts.ingest(12.5, .1, available=16)
    result = parts.assess(16)
    event = parts.events(16)[0]
    assert len(parts.events(16)) == 1 and event["event_id"] == original["event_id"]
    assert event["event_state"] == "ended" and event["end_at"] == utcstr(START + timedelta(minutes=12.5))
    assert event["evidence_status"] == "insufficient_after_late_arrival"
    assert event["late_observation_refs"][0]["event_time"] == utcstr(START + timedelta(minutes=12.5))
    assert event["evidence_reassessment"]["previous_event_revision"] == original["event_revision"]
    assert result["alert_level"] != "orange"
    assert parts.service.get_event_snapshot(parts.context(16), event["event_id"], original["event_revision"]) == original
    saved = deepcopy(parts.store.list("event_revision"))
    parts.assess(16)
    assert parts.store.list("event_revision") == saved


def test_future_available_late_rows_wait_then_same_cutoff_retries_are_idempotent(parts):
    parts.ingest(20, .1)
    parts.assess(20)
    parts.ingest(10, .9, available=22)
    parts.assess(21)
    assert parts.events(21) == []
    parts.assess(22)
    saved = deepcopy(parts.store.list("event_revision"))
    assert len(parts.events(22)) == 1
    parts.assess(22)
    parts.assess(23)
    assert parts.store.list("event_revision") == saved


def test_late_row_available_exactly_at_prior_cutoff_is_not_lost(parts):
    parts.ingest(20, .1)
    parts.assess(20)
    parts.ingest(10, .9, available=20)
    parts.assess(20)
    assert len(parts.events(20)) == 1


def test_ordinary_ticks_do_not_reload_old_year_of_observations(parts, monkeypatch):
    parts.ingest(-365 * 24 * 60, .9)
    for minute in range(10, 21):
        parts.ingest(minute, .1)
    parts.assess(20)
    query_starts = []
    original_query = parts.data.query_series

    def query(context, point_id, start_at, end_at, **kwargs):
        query_starts.append(start_at)
        return original_query(context, point_id, start_at, end_at, **kwargs)

    monkeypatch.setattr(parts.data, "query_series", query)
    parts.assess(21)
    assert query_starts and min(query_starts) >= START + timedelta(minutes=16)
