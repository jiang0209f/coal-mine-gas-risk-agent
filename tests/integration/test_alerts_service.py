from datetime import datetime, timedelta, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, utcstr
from app.services.alerts.service import AlertsService
from app.services.data.service import DataService

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def setup():
    store = Store("sqlite://")
    store.put("source", "s", {"source_id": "s", "face_id": "31002"})
    store.put("point", "p", {"point_id": "p", "source_id": "s", "face_id": "31002", "unit": "%"})
    actor = Actor(user_id="u", roles=("ventilation",), face_ids=("31002",))
    yield store, AlertsService(store), actor
    store.close()


def ctx(actor, seconds=0, **overrides):
    values = dict(session_id="session", mode="historical_replay", source_id="s", as_of=START + timedelta(seconds=seconds), actor=actor)
    values.update(overrides)
    return ExecutionContext(**values)


def row(seconds, value, **overrides):
    values = dict(id=f"o-{seconds}", point_id="p", source_id="s", event_time=utcstr(START + timedelta(seconds=seconds)), available_at=utcstr(START + timedelta(seconds=seconds)), value=value, quality_flags=[], source_file_id="file", row_number=seconds + 2)
    values.update(overrides)
    return values


def test_rules_are_explicit_team_draft_with_continuity_parameters(setup):
    _, service, actor = setup
    rules = service.get_rules(actor, face_id="31002", point_id="p")
    assert rules["rule_version"] == "team-draft-v1"
    assert rules["status"] == "team-draft"
    assert rules["observed_red_threshold"] == 0.8
    assert rules["observed_orange_threshold"] == 0.65
    assert rules["orange_duration_seconds"] == 300
    assert rules["max_observation_gap_seconds"] == 60
    assert rules["prediction_stage"] == "residual_corrected"


def test_red_is_immediate_and_calibration_flag_does_not_suppress_it(setup):
    _, service, actor = setup
    result = service.evaluate_observed(ctx(actor), "p", [row(0, 0.8, quality_flags=["suspected_calibration"])])
    assert result["status"] == "evaluated"
    assert result["alert_level"] == "red"
    event = service.query_events(ctx(actor))[0]
    assert event["alert_level"] == "red"
    assert event["event_type"] == "observed_exceedance"
    assert event["quality_flags"] == ["suspected_calibration"]
    assert event["event_time"] == utcstr(START)
    assert event["revision_as_of"] == utcstr(START)


def test_data_service_observation_exactly_at_cutoff_triggers_red(tmp_path):
    store = Store("sqlite://")
    try:
        data = DataService(store, tmp_path / "uploads")
        data.register_source("s", "Source", face_id="31002")
        data.register_point("p", "s", "tag", "Methane", "methane", "%")
        actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
        file = tmp_path / "at-cutoff.csv"
        file.write_text("STime,SValue\n2026-10-01T00:00:00+00:00,0.8\n2026-10-01T00:01:00+00:00,0.99\n", encoding="utf-8")
        registered = data.register_file(file, actor)
        data.import_history(actor, "s", registered["file_id"], {"time_column": "STime", "value_columns": {"SValue": "p"}, "lineage": "raw"})
        service = AlertsService(store, data)
        result = service.evaluate_observed(ctx(actor), "p")
        assert result["alert_level"] == "red"
        event = service.query_events(ctx(actor))[0]
        assert event["evidence_refs"][0]["event_time"] == utcstr(START)
        assert len(event["evidence_refs"]) == 1
    finally:
        store.close()


def test_observed_evaluation_reads_all_raw_pages(setup):
    _, service, actor = setup
    offsets = []
    low = row(0, 0.1)

    class PagedData:
        def query_series(self, context, point_id, start_at, end_at, limit, offset):
            offsets.append(offset)
            return {"rows": [low] * 100000 if offset == 0 else [row(60, 0.9)], "total": 100001}

    service.data_service = PagedData()
    result = service.evaluate_observed(ctx(actor, 60), "p")
    assert result["alert_level"] == "red"
    assert offsets == [0, 100000]


def test_orange_requires_five_minutes_of_continuous_high_values(setup):
    _, service, actor = setup
    result = service.evaluate_observed(ctx(actor, 300), "p", [row(t, 0.65) for t in range(0, 301, 60)])
    assert result["alert_level"] == "orange"
    event = service.query_events(ctx(actor, 300))[0]
    assert event["start_at"] == utcstr(START)
    assert event["event_time"] == utcstr(START + timedelta(minutes=5))


def test_continuous_orange_accumulates_across_calls(setup):
    _, service, actor = setup
    for seconds in range(0, 301, 60):
        result = service.evaluate_observed(ctx(actor, seconds), "p", [row(seconds, 0.7)])
    assert result["alert_level"] == "orange"
    assert len(service.query_events(ctx(actor, 300))) == 1


@pytest.mark.parametrize("rows", [[], [row(0, 0.2)], [row(0, 0.7), row(300, 0.7)], [row(0, None)], [row(0, 9, quality_flags=["invalid_value"])]])
def test_insufficient_or_invalid_data_is_not_ready_instead_of_blue(setup, rows):
    _, service, actor = setup
    result = service.evaluate_observed(ctx(actor, 300 if len(rows) > 1 else 0), "p", rows)
    assert result["status"] == "not_ready"
    assert result["alert_level"] is None
    assert service.query_events(ctx(actor, 300)) == []


def test_complete_valid_low_values_can_be_evaluated_blue(setup):
    _, service, actor = setup
    result = service.evaluate_observed(ctx(actor, 300), "p", [row(t, 0.4) for t in range(0, 301, 60)])
    assert result["status"] == "evaluated"
    assert result["alert_level"] == "blue"


def test_future_event_or_availability_is_excluded(setup):
    _, service, actor = setup
    result = service.evaluate_observed(ctx(actor), "p", [row(60, 0.9), row(0, 0.9, available_at=utcstr(START + timedelta(seconds=60)))])
    assert result["status"] == "not_ready"
    assert service.query_events(ctx(actor)) == []


def test_event_revisions_upgrade_end_and_remain_visible_at_logical_cutoff(setup):
    _, service, actor = setup
    service.evaluate_observed(ctx(actor, 300), "p", [row(t, 0.7) for t in range(0, 301, 60)])
    original = service.query_events(ctx(actor, 300))[0]
    service.evaluate_observed(ctx(actor, 360), "p", [row(360, 0.9)])
    upgraded = service.query_events(ctx(actor, 360))[0]
    service.evaluate_observed(ctx(actor, 420), "p", [row(420, 0.2)])
    ended = service.query_events(ctx(actor, 420))[0]
    assert original["event_id"] == upgraded["event_id"] == ended["event_id"]
    assert upgraded["event_state"] == "upgraded"
    assert upgraded["event_revision"] == 2
    assert ended["event_state"] == "ended"
    assert ended["end_at"] == utcstr(START + timedelta(seconds=420))
    assert service.get_event_snapshot(ctx(actor, 300), original["event_id"]) == original
    assert len(service.query_events(ctx(actor, 420), include_revisions=True)) == 3
    with pytest.raises(DomainError) as caught:
        service.get_event_snapshot(ctx(actor, 300), original["event_id"], event_revision=3)
    assert caught.value.status_code == 404


def test_duplicate_evaluation_is_idempotent_and_quality_append_adds_revision(setup):
    _, service, actor = setup
    sample = row(0, 0.8)
    service.evaluate_observed(ctx(actor), "p", [sample])
    service.evaluate_observed(ctx(actor), "p", [sample])
    assert len(service.query_events(ctx(actor), include_revisions=True)) == 1
    sample["quality_flags"] = ["suspected_calibration"]
    service.evaluate_observed(ctx(actor, 60), "p", [sample])
    snapshots = service.query_events(ctx(actor, 60), include_revisions=True)
    assert len(snapshots) == 2
    assert snapshots[-1]["quality_flags"] == ["suspected_calibration"]
    assert snapshots[-1]["alert_level"] == "red"


def test_event_and_revision_are_rolled_back_together_on_write_failure(setup, monkeypatch):
    store, service, actor = setup
    original_put = store.put

    def fail_revision(kind, id, payload, connection=None):
        if kind == "event_revision":
            raise RuntimeError("revision write failed")
        return original_put(kind, id, payload, connection=connection)

    monkeypatch.setattr(store, "put", fail_revision)
    with pytest.raises(RuntimeError, match="revision write failed"):
        service.evaluate_observed(ctx(actor), "p", [row(0, 0.8)])
    assert store.list("event") == []
    assert store.list("event_revision") == []
    assert store.list("alert_observed_state") == []


def test_calibration_flag_can_be_appended_after_event_ended(setup):
    _, service, actor = setup
    service.evaluate_observed(ctx(actor), "p", [row(0, 0.8)])
    service.evaluate_observed(ctx(actor, 60), "p", [row(60, 0.2)])
    service.evaluate_observed(ctx(actor, 120), "p", [row(0, 0.8, quality_flags=["suspected_calibration"])])
    event = service.query_events(ctx(actor, 120))[0]
    assert event["event_state"] == "ended"
    assert event["alert_level"] == "red"
    assert event["quality_flags"] == ["suspected_calibration"]
    assert event["event_revision"] == 3


def test_valid_coverage_restarts_after_invalid_sample(setup):
    _, service, actor = setup
    samples = [row(0, None)] + [row(t, 0.3) for t in range(60, 361, 60)]
    assert service.evaluate_observed(ctx(actor, 360), "p", samples)["alert_level"] == "blue"


def test_delayed_red_sample_is_recorded_when_it_becomes_available(setup):
    _, service, actor = setup
    service.evaluate_observed(ctx(actor, 60), "p", [row(60, 0.2)])
    delayed = row(0, 0.9, available_at=utcstr(START + timedelta(seconds=120)))
    service.evaluate_observed(ctx(actor, 120), "p", [delayed])
    events = service.query_events(ctx(actor, 120))
    assert len(events) == 1
    assert events[0]["alert_level"] == "red"
    assert events[0]["start_at"] == utcstr(START)
    assert events[0]["revision_as_of"] == utcstr(START + timedelta(seconds=120))
    assert service.query_events(ctx(actor, 60)) == []
    service.evaluate_observed(ctx(actor, 120), "p", [delayed])
    assert len(service.query_events(ctx(actor, 120))) == 1


def test_rewound_evaluation_cannot_mutate_future_session_state(setup):
    _, service, actor = setup
    service.evaluate_observed(ctx(actor, 60), "p", [row(60, 0.9)])
    with pytest.raises(DomainError) as caught:
        service.evaluate_observed(ctx(actor), "p", [row(0, 0.9)])
    assert caught.value.status_code == 409
    assert service.query_events(ctx(actor)) == []


def forecast(value=0.7, stage="residual_corrected", semantics="physical_forecast", **overrides):
    values = dict(run_id="run", session_id="session", source_id="s", point_id="p", face_id="31002", as_of=utcstr(START), status="completed", input_refs=[], outputs=[dict(prediction_stage=stage, quantity_semantics=semantics, unit="%", interval_values=[dict(start_at=utcstr(START), end_at=utcstr(START + timedelta(minutes=5)), value=value)])])
    values.update(overrides)
    return values


@pytest.mark.parametrize(("value", "level"), [(0.65, "yellow"), (0.8, "orange")])
def test_forecast_compares_physical_residual_corrected_values(setup, value, level):
    _, service, actor = setup
    result = service.evaluate_forecast(ctx(actor), forecast(value))
    assert result["alert_level"] == level
    events = service.query_events(ctx(actor), event_type="forecast_exceedance")
    assert len(events) == (0 if level == "blue" else 1)


def test_partial_low_forecast_is_not_ready_and_full_horizon_can_be_blue(setup):
    _, service, actor = setup
    run = forecast(0.4)
    assert service.evaluate_forecast(ctx(actor), run)["status"] == "not_ready"
    run["outputs"][0]["interval_values"] = [dict(
        start_at=utcstr(START + timedelta(minutes=5 * index)),
        end_at=utcstr(START + timedelta(minutes=5 * (index + 1))), value=0.4,
    ) for index in range(12)]
    result = service.evaluate_forecast(ctx(actor), run)
    assert result["status"] == "evaluated"
    assert result["alert_level"] == "blue"


def full_forecast(value=0.4, cutoff_seconds=0, issued_seconds=0):
    cutoff = START + timedelta(seconds=cutoff_seconds)
    run = forecast(value, as_of=utcstr(START + timedelta(seconds=issued_seconds)),
                   cutoff=utcstr(cutoff), aggregation={"seconds": 300})
    run["outputs"][0]["interval_values"] = [dict(
        start_at=utcstr(cutoff + timedelta(minutes=5 * index)),
        end_at=utcstr(cutoff + timedelta(minutes=5 * (index + 1))), value=value,
    ) for index in range(12)]
    return run


def test_full_low_forecast_remains_evaluated_between_bucket_boundaries(setup):
    _, service, actor = setup
    result = service.evaluate_forecast(ctx(actor, 60), full_forecast(0.4, issued_seconds=60))
    assert result["status"] == "evaluated"
    assert result["alert_level"] == "blue"


def test_first_current_forecast_bucket_can_trigger_between_boundaries(setup):
    _, service, actor = setup
    run = full_forecast(0.4, issued_seconds=60)
    run["outputs"][0]["interval_values"][0]["value"] = 0.7
    result = service.evaluate_forecast(ctx(actor, 60), run)
    assert result["status"] == "evaluated"
    assert result["alert_level"] == "yellow"
    event = service.query_events(ctx(actor, 60), event_type="forecast_exceedance")[0]
    assert event["revision_as_of"] == utcstr(START + timedelta(seconds=60))


def test_complete_low_forecast_ends_existing_event_between_boundaries(setup):
    _, service, actor = setup
    service.evaluate_forecast(ctx(actor), full_forecast(0.7))
    original = service.query_events(ctx(actor), event_type="forecast_exceedance")[0]
    result = service.evaluate_forecast(ctx(actor, 60), full_forecast(0.4, issued_seconds=60))
    assert result["status"] == "evaluated"
    assert result["alert_level"] == "blue"
    event = service.query_events(ctx(actor, 60), event_type="forecast_exceedance")[0]
    assert event["event_id"] == original["event_id"]
    assert event["event_state"] == "ended"
    assert event["revision_as_of"] == utcstr(START + timedelta(seconds=60))
    assert event["end_at"] == utcstr(START + timedelta(seconds=60))


@pytest.mark.parametrize(("context_seconds", "cutoff_seconds"), [(300, 0), (60, 300), (60, -300)])
def test_expired_or_future_forecast_cutoff_does_not_trigger_rules(setup, context_seconds, cutoff_seconds):
    _, service, actor = setup
    result = service.evaluate_forecast(ctx(actor, context_seconds), full_forecast(
        0.9, cutoff_seconds=cutoff_seconds, issued_seconds=context_seconds))
    assert result["status"] == "not_ready"
    assert result["alert_level"] is None
    assert service.query_events(ctx(actor, context_seconds)) == []


@pytest.mark.parametrize(("stage", "semantics"), [("threshold_shifted", "decision_score"), ("fedformer_raw", "physical_forecast"), ("baseline_persistence", "physical_forecast"), ("residual_corrected", "decision_score")])
def test_other_prediction_stages_and_decision_scores_do_not_trigger_rules(setup, stage, semantics):
    _, service, actor = setup
    result = service.evaluate_forecast(ctx(actor), forecast(5, stage, semantics))
    assert result["status"] == "not_ready"
    assert result["alert_level"] is None
    assert service.query_events(ctx(actor)) == []


def test_incomplete_forecast_is_not_ready(setup):
    _, service, actor = setup
    assert service.evaluate_forecast(ctx(actor), forecast(status="not_ready", outputs=[]))["status"] == "not_ready"


@pytest.mark.parametrize("operation", ["observed", "forecast", "events", "rules"])
def test_source_face_permission_is_enforced_for_all_entrypoints(setup, operation):
    _, service, _ = setup
    actor = Actor(user_id="other", roles=("ventilation",), face_ids=("3411",))
    with pytest.raises(DomainError) as caught:
        if operation == "observed":
            service.evaluate_observed(ctx(actor), "p", [row(0, 0.9)])
        elif operation == "forecast":
            service.evaluate_forecast(ctx(actor), forecast())
        elif operation == "rules":
            service.get_rules(actor, face_id="31002")
        else:
            service.query_events(ctx(actor))
    assert caught.value.status_code == 403
