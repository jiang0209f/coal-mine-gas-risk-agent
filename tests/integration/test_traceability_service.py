from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert

from app.db import Store, observations
from app.schemas.common import Actor, DomainError, ExecutionContext, utcstr
from app.services.alerts.service import AlertsService
from app.services.traceability.service import TraceabilityService

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def setup():
    store = Store("sqlite://")
    store.put("source", "s", {"source_id": "s", "face_id": "31002"})
    store.put("point", "p", {"point_id": "p", "source_id": "s", "face_id": "31002", "unit": "%"})
    actor = Actor(user_id="u", roles=("ventilation",), face_ids=("31002",))
    context = ExecutionContext(session_id="session", mode="historical_replay", source_id="s", as_of=START, actor=actor)
    row = dict(id="o", point_id="p", source_id="s", event_time=utcstr(START), available_at=utcstr(START), value=0.9, raw_value="0.900", quality_flags=["suspected_calibration"], source_file_id="file", row_number=2, import_id="import")
    with store.transaction() as connection:
        connection.execute(insert(observations).values(**row))
    alerts = AlertsService(store)
    alerts.evaluate_observed(context, "p", [row])
    event = alerts.query_events(context)[0]
    yield store, alerts, TraceabilityService(store, alerts), context, event
    store.close()


def test_evidence_contains_exact_snapshot_original_row_and_draft_rule(setup):
    _, _, service, context, event = setup
    bundle = service.get_event_evidence(context, event["event_id"])
    assert bundle["event_snapshot"] == event
    assert bundle["observation_fragments"][0]["raw_value"] == "0.900"
    assert bundle["observation_fragments"][0]["source_file_id"] == "file"
    assert bundle["rules"]["status"] == "team-draft"
    assert bundle["causal_status"] == "evidence_only"
    assert bundle["condition_hints"][0]["kind"] == "suspected_calibration"
    assert bundle["condition_hints"][0]["evidence_refs"]


def test_reading_evidence_does_not_create_records_or_revisions(setup):
    store, alerts, service, context, event = setup
    before_events = alerts.query_events(context, include_revisions=True)
    before_records = store.list("event_revision")
    service.get_event_evidence(context, event["event_id"])
    service.get_event_evidence(context, event["event_id"])
    assert alerts.query_events(context, include_revisions=True) == before_events
    assert store.list("event_revision") == before_records


def test_later_data_does_not_change_old_revision_evidence(setup):
    store, alerts, service, context, event = setup
    before = service.get_event_evidence(context, event["event_id"])
    future_time = START + timedelta(minutes=1)
    future = dict(id="future", point_id="p", source_id="s", event_time=utcstr(future_time), available_at=utcstr(future_time), value=0.99, raw_value="0.990", quality_flags=[], source_file_id="later-file", row_number=3, import_id="import")
    with store.transaction() as connection:
        connection.execute(insert(observations).values(**future))
    later_context = context.model_copy(update={"as_of": future_time})
    alerts.evaluate_observed(later_context, "p", [future])
    after = service.get_event_evidence(later_context, event["event_id"], event_revision=1)
    assert after == before
    assert "later-file" not in str(after)


def test_evidence_rejects_other_source_face_and_unknown_event(setup):
    _, _, service, context, event = setup
    outsider = Actor(user_id="other", roles=("ventilation",), face_ids=("3411",))
    with pytest.raises(DomainError) as caught:
        service.get_event_evidence(context.model_copy(update={"actor": outsider}), event["event_id"])
    assert caught.value.status_code == 403
    with pytest.raises(DomainError) as caught:
        service.get_event_evidence(context, "missing")
    assert caught.value.status_code == 404
