"""Late-evidence revisions propagate to evidence and newly frozen evaluations."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.db import Store
from app.schemas.common import Actor, utcstr
from app.services.alerts.service import AlertsService
from app.services.data.service import DataService
from app.services.evaluation.collector import SessionEvaluationCollector
from app.services.evaluation.service import EvaluationService
from app.services.sessions.service import SessionsService
from app.services.traceability.service import TraceabilityService

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


def at(minute):
    return START + timedelta(minutes=minute)


@pytest.fixture
def parts(tmp_path):
    store = Store("sqlite+pysqlite:///:memory:")
    data = DataService(store, tmp_path / "imports")
    data.register_source("s", "Synthetic source", "31002")
    data.register_point("p", "s", "T0", "T0", "methane", "%", unit_status="confirmed")
    actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
    sessions = SessionsService(store)
    session = sessions.create_session(actor, "historical_replay", "s", START)
    alerts = AlertsService(store, data)
    evaluations = EvaluationService(store)
    collector = SessionEvaluationCollector(store, {"alerts": alerts})
    calls = []

    def attribute_event(*args):
        calls.append(args)
        return {"status": "completed", "run_id": "synthetic-run", "candidate_causes": ["synthetic-candidate"]}

    trace = TraceabilityService(store, alerts, data, trained_models=SimpleNamespace(attribute_event=attribute_event))

    def ingest(minute, value, available=None, point_id="p", source_id="s", identity=None):
        return data.ingest_readings(actor, source_id, [{"reading_id": identity or str(minute), "point_id": point_id,
            "event_time": utcstr(at(minute)), "raw_value": value}], received_at=at(minute if available is None else available))

    def context(minute):
        sessions.apply_action(actor, session["session_id"], "advance", target_as_of=at(minute))
        return sessions.get_context(actor, session["session_id"])

    for minute in range(10, 16):
        ingest(minute, .70)
    alerts.evaluate_observed(context(15), "p")
    original = deepcopy(alerts.query_events(context(15))[0])

    def invalidate():
        ingest(12.5, .1, available=16)
        alerts.evaluate_observed(context(16), "p")
        return alerts.query_events(context(16))[0]

    yield SimpleNamespace(store=store, data=data, actor=actor, sessions=sessions, alerts=alerts, original=original,
        trace=trace, calls=calls, ingest=ingest, context=context, invalidate=invalidate, collector=collector, evaluations=evaluations)
    store.close()


def test_late_fragment_is_openable_and_unsupported_event_skips_candidate_model(parts):
    event = parts.invalidate()
    evidence = parts.trace.get_event_evidence(parts.context(16), event["event_id"])
    assert evidence["event_snapshot"]["evidence_status"] == "insufficient_after_late_arrival"
    late = evidence["late_observation_fragments"]
    assert len(late) == 1 and late[0]["raw_value"] == "0.1"
    assert late[0]["event_time"] == utcstr(at(12.5)) and late[0]["available_at"] == utcstr(at(16))
    assert any(ref.get("observation_id") == late[0]["id"] for ref in evidence["evidence_refs"])
    assert evidence["candidate_attribution"]["status"] == "not_ready"
    assert evidence["candidate_attribution"]["missing_fields"] == ["evidence_supported"]
    assert evidence["causal_status"] == "evidence_only" and parts.calls == []


def test_late_fragment_lookup_filters_source_point_and_both_cutoff_times(parts):
    event = parts.invalidate()
    parts.data.register_point("other-point", "s", "OTHER", "Other", "methane", "%")
    parts.data.register_source("foreign", "Foreign", "31002")
    parts.data.register_point("foreign-point", "foreign", "FOREIGN", "Foreign", "methane", "%")
    parts.ingest(12, .2, available=16, point_id="other-point", identity="other")
    parts.ingest(12, .3, available=16, source_id="foreign", point_id="foreign-point", identity="foreign")
    parts.ingest(13, .4, available=17, identity="future-available")
    parts.ingest(17, .5, available=16, identity="future-event")
    from sqlalchemy import select
    from app.db import observations
    with parts.store.transaction() as connection:
        rows = [dict(row) for row in connection.execute(select(observations)).mappings()]
    bad_refs = [parts.alerts._reference(row) for row in rows if row["point_id"] != "p" or row["available_at"] > utcstr(at(16)) or row["event_time"] > utcstr(at(16))]
    event["late_observation_refs"] += bad_refs
    parts.store.put("event_revision", f"{event['event_id']}@{event['event_revision']:08d}", event)
    evidence = parts.trace.get_event_evidence(parts.context(17), event["event_id"], event["event_revision"])
    assert len(evidence["late_observation_fragments"]) == 1
    valid_id = evidence["late_observation_fragments"][0]["id"]
    appended = {ref.get("observation_id") for ref in evidence["evidence_refs"]} - {ref.get("observation_id") for ref in event["evidence_refs"]}
    assert appended == {valid_id}
    assert "late_observation_fragments" in evidence["missing_fields"]


def test_old_revision_keeps_original_evidence_and_candidate_behavior(parts):
    before = parts.trace.get_event_evidence(parts.context(15), parts.original["event_id"], parts.original["event_revision"])
    parts.invalidate()
    after = parts.trace.get_event_evidence(parts.context(16), parts.original["event_id"], parts.original["event_revision"])
    assert after == before
    assert after["candidate_attribution"]["status"] == "completed"
    assert after["late_observation_fragments"] == []


def test_new_freeze_excludes_invalidated_truth_but_preserves_all_visible_event_revisions(parts):
    event = parts.invalidate()
    future = {**event, "event_revision": event["event_revision"] + 1, "revision_as_of": utcstr(at(17)), "evidence_status": "FUTURE"}
    parts.store.put("event_revision", "future", future)
    parts.store.put("event_revision", "foreign", {**event, "event_id": "foreign", "source_id": "foreign"})
    parts.store.put("event_revision", "other-point", {**event, "event_id": "other-point", "point_id": "not-selected"})
    collected = parts.collector.collect(parts.context(16), at(0), at(16))
    assert collected["records"]["actual_events"] == []
    snapshots = collected["records"]["event_revisions"]
    assert {(row["event_id"], row["event_revision"]) for row in snapshots} == {
        (parts.original["event_id"], parts.original["event_revision"]), (event["event_id"], event["event_revision"])}
    current = next(row for row in snapshots if row["event_revision"] == event["event_revision"])
    assert current["evidence_status"] == event["evidence_status"]
    assert current["late_observation_refs"] == event["late_observation_refs"]
    assert current["evidence_reassessment"] == event["evidence_reassessment"]
    narrow = parts.collector.collect(parts.context(16), at(14), at(16))
    assert narrow["records"]["actual_events"] == []
    assert len(narrow["records"]["event_revisions"]) == 2


def test_preexisting_frozen_report_and_export_do_not_change_after_late_revision(parts):
    original = parts.collector.collect(parts.context(15), at(0), at(15))
    report = parts.evaluations.run(parts.actor, original["spec"], original["records"])
    before_manifest = parts.evaluations.get_manifest(parts.actor, report["evaluation_id"])
    before_export = parts.evaluations.export_report(parts.actor, report["evaluation_id"])
    parts.invalidate()
    new = parts.collector.collect(parts.context(16), at(0), at(16))
    assert new["records"]["actual_events"] == []
    assert parts.evaluations.get_report(parts.actor, report["evaluation_id"]) == report
    assert parts.evaluations.get_manifest(parts.actor, report["evaluation_id"]) == before_manifest
    assert parts.evaluations.export_report(parts.actor, report["evaluation_id"]) == before_export
    assert len(before_manifest["frozen_records"]["actual_events"]) == 1


def test_forecast_notification_history_is_retained_without_duplicate_target_truth(parts):
    notice = {**parts.original, "event_id": "notice", "event_type": "forecast_exceedance", "forecast_run_id": "run",
        "start_at": utcstr(at(5)), "event_time": utcstr(at(6)), "end_at": utcstr(at(6)),
        "revision_as_of": utcstr(at(6)), "evidence_refs": [{"kind": "forecast_run", "run_id": "run"}]}
    later = {**notice, "event_revision": notice["event_revision"] + 1, "revision_as_of": utcstr(at(7)), "event_time": utcstr(at(7))}
    actual, targets, snapshots = parts.collector._events([notice, later], parts.context(15), {"p"},
        [{"run_id": "run", "as_of": utcstr(at(5))}], {"run": [(at(6), at(7), .9)]},
        {"p": {"forecast_yellow_threshold": .65}}, "residual_corrected", at(0), at(15))
    assert actual == [] and len(targets) == 1
    assert [row["event_revision"] for row in snapshots] == [notice["event_revision"], later["event_revision"]]
