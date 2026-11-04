"""Session collection uses saved runs and frozen, visible observation support."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import insert, update

from app.db import Store, observations
from app.schemas.common import Actor, DomainError, ExecutionContext, utcstr


def at(minute=0):
    return datetime(2022, 2, 5, tzinfo=timezone.utc) + timedelta(minutes=minute)


def interval(start, end, value):
    return dict(start_at=utcstr(at(start)), end_at=utcstr(at(end)), value=value)


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    context = ExecutionContext(session_id="session", source_id="source", mode="historical_replay", as_of=at(12), actor=actor)
    store.put("source", "source", dict(source_id="source", name="Synthetic only", face_id="31002"))
    store.put("session", "session", dict(session_id="session", source_id="source", face_id="31002", mode="historical_replay",
        start_as_of=utcstr(at()), as_of=utcstr(at(12)), state="paused", speed=1, config_version="qa"))
    store.put("point", "p", dict(point_id="p", source_id="source", face_id="31002", quantity="methane", unit="%"))
    store.put("point", "wind", dict(point_id="wind", source_id="source", face_id="31002", quantity="wind_speed", unit="m/s"))
    store.put("point", "foreign", dict(point_id="foreign", source_id="foreign-source", face_id="31002", quantity="methane", unit="%"))
    model = dict(model_id="model", version="v1", source_id="source", face_id="31002", point_ids=["p"],
        adapter_kind="fedformer_lgbm", target_quantity="methane", target_statistic="interval_mean", aggregation_seconds=120,
        unit="%", legacy_source_root="F:/PRIVATE_MODEL_PATH")
    store.put("model", "model:v1", model)
    rows = [dict(id=f"obs-{i}", source_id="source", point_id="p", event_time=utcstr(at(i)), available_at=utcstr(at(i)),
        raw_value=str(.9 if i == 2 else .5), value=.9 if i == 2 else .5,
        quality_flags=["suspected_calibration"] if i == 2 else [], source_file_id="file", row_number=i+1, import_id="import") for i in range(13)]
    with store.transaction() as connection:
        connection.execute(insert(observations), rows)
    run = dict(run_id="run", session_id="session", source_id="source", face_id="31002", point_id="p", model_id="model",
        model_version="v1", as_of=utcstr(at(1)), cutoff=utcstr(at(1)), status="completed", aggregation={"seconds": 120},
        input_refs=[dict(kind="observation", observation_id="obs-0", event_time=utcstr(at()), available_at=utcstr(at()))],
        outputs=[dict(prediction_stage="residual_corrected", quantity_semantics="physical_forecast", unit="%",
                      interval_values=[interval(1,3,.9), interval(3,5,.4), interval(9,11,.9)]),
                 dict(prediction_stage="decision", quantity_semantics="decision_score", unit="1", interval_values=[interval(1,3,.99)])])
    store.put("forecast_run", "run", run)
    actual = dict(event_id="actual", session_id="session", source_id="source", point_id="p", event_type="observed_exceedance",
        event_revision=1, revision_as_of=utcstr(at(3)), event_time=utcstr(at(3)), start_at=utcstr(at(2)), end_at=utcstr(at(3)),
        rule_version="qa-rule", alert_level="red", quality_flags=["suspected_calibration"],
        evidence_refs=[dict(kind="observation", observation_id="obs-2", source_file_id="file", row_number=3,
                            event_time=utcstr(at(2)), available_at=utcstr(at(2)))])
    forecast = dict(event_id="forecast", session_id="session", source_id="source", point_id="p", event_type="forecast_exceedance",
        event_revision=1, revision_as_of=utcstr(at(5)), event_time=utcstr(at(5)), start_at=utcstr(at(1)), end_at=utcstr(at(5)),
        rule_version="qa-rule", alert_level="yellow", forecast_run_id="run", quality_flags=[], evidence_refs=[dict(kind="forecast_run", run_id="run")])
    for row in (actual, forecast):
        store.put("event_revision", row["event_id"] + "@1", row)
    store.put("task", "task", dict(task_id="task", session_id="session", source_id="source", face_id="31002", as_of=utcstr(at(4)),
        state="completed", final_text="PRIVATE_FINAL_TEXT", messages=[dict(role="user", content="PRIVATE_QUESTION")],
        evidence_refs=actual["evidence_refs"], result_cards=[dict(data="PRIVATE_RESULT_CARD")], missing_fields=[]))
    store.put("task", "queued", dict(task_id="queued", session_id="session", source_id="source", as_of=utcstr(at(5)), state="queued"))
    store.put("advice", "advice:1", dict(advice_id="advice", version=1, event_id="actual", event_revision=1, session_id="session",
        face_id="31002", data_as_of=utcstr(at(4)), reference_ids=["ref"], items=[dict(text="PRIVATE_ADVICE", reference_id="ref")]))
    store.put("case_action", "action", dict(action_id="action", event_id="actual", session_id="session", source_id="source",
        face_id="31002", event_revision=1, action_type="confirm_event", data_as_of=utcstr(at(5)),
        occurred_at="2026-11-04T12:00:00+00:00", content="PRIVATE_CASE_TEXT", actor_id="manager"))
    store.put("session_action", "session-created", dict(action_id="session-created", action_type="create", sequence=1,
        session_id="session", source_id="source", face_id="31002", data_as_of=utcstr(at()),
        before=None, after=dict(as_of=utcstr(at()), state="paused", speed=1)))
    rules = dict(rule_version="qa-rule", observed_red_threshold=.8, observed_orange_threshold=.65,
        forecast_yellow_threshold=.65, forecast_orange_threshold=.8, max_observation_gap_seconds=60, status="team-draft")
    yield SimpleNamespace(store=store, actor=actor, context=context, run=run, actual=actual, forecast=forecast,
        services={"alerts": SimpleNamespace(get_rules=lambda *args, **kwargs: deepcopy(rules))})
    store.close()


def collect(parts, **kwargs):
    from app.services.evaluation.collector import SessionEvaluationCollector
    return SessionEvaluationCollector(parts.store, parts.services).collect(parts.context, at(), at(10), **kwargs)


def test_collects_only_mature_physical_samples_with_original_observation_refs(parts):
    result = collect(parts)
    samples = result["records"]["samples"]
    assert len(samples) == 2
    first = samples[0]
    assert first["run_id"] == "run" and first["prediction_stage"] == "residual_corrected"
    assert first["actual_value"] == pytest.approx(.7) and first["predicted_value"] == .9
    assert first["actual_positive"] is False and first["predicted_positive"] is True
    assert {ref["observation_id"] for ref in first["evidence_refs"]} == {"obs-1", "obs-2"}
    assert result["spec"]["dataset_split"] == "unclassified"
    assert result["collection_summary"]["truth_semantics"] == "observed_threshold_derived"
    assert result["spec"]["prediction_stage"] == "residual_corrected"
    assert result["records"]["valid_monitoring_hours"] == pytest.approx(10/60)
    assert result["records"]["valid_monitoring_hours_excluding_suspected"] == pytest.approx(8/60)


def test_forecast_events_use_target_time_and_merge_repeated_overlapping_runs(parts):
    second = deepcopy(parts.run)
    second.update(run_id="run-2", as_of=utcstr(at(2)), cutoff=utcstr(at(2)))
    second["outputs"] = [dict(prediction_stage="residual_corrected", quantity_semantics="physical_forecast", unit="%",
                              interval_values=[interval(2,4,.85)])]
    parts.store.put("forecast_run", "run-2", second)
    updated = {**parts.forecast, "event_revision": 2, "revision_as_of": utcstr(at(6)), "forecast_run_id": "run-2",
               "evidence_refs": [dict(kind="forecast_run", run_id="run"), dict(kind="forecast_run", run_id="run-2")]}
    parts.store.put("event_revision", "forecast@2", updated)
    alerts = collect(parts)["records"]["alert_events"]
    assert len(alerts) == 1
    assert alerts[0]["start_at"] == utcstr(at(1)) and alerts[0]["end_at"] == utcstr(at(4))
    assert alerts[0]["source_event_id"] == "forecast" and set(alerts[0]["run_ids"]) == {"run", "run-2"}
    assert alerts[0]["issued_at"] == utcstr(at(1))


def test_early_notification_outside_analysis_window_keeps_its_inside_target(parts):
    from app.services.evaluation.collector import SessionEvaluationCollector
    run = deepcopy(parts.run)
    run["outputs"][0]["interval_values"][1]["value"] = .9
    parts.store.put("forecast_run", "run", run)
    notice = {**parts.forecast, "event_time": utcstr(at(2)), "end_at": utcstr(at(2)), "revision_as_of": utcstr(at(2))}
    parts.store.put("event_revision", "forecast@1", notice)

    result = SessionEvaluationCollector(parts.store,parts.services).collect(parts.context,at(3),at(10))

    assert len(result["records"]["alert_events"]) == 1
    assert result["records"]["alert_events"][0]["start_at"] == utcstr(at(3))
    assert result["records"]["alert_events"][0]["end_at"] == utcstr(at(5))
    assert result["records"]["alert_events"][0]["issued_at"] == utcstr(at(1))


def test_freeze_excludes_future_availability_revisions_and_foreign_scope(parts):
    future = {**parts.actual, "event_revision": 2, "revision_as_of": utcstr(at(11)), "quality_flags": ["FUTURE_REVISION"]}
    parts.store.put("event_revision", "actual@2", future)
    foreign = {**parts.run, "run_id": "foreign-run", "session_id": "foreign-session"}
    parts.store.put("forecast_run", foreign["run_id"], foreign)
    with parts.store.transaction() as connection:
        connection.execute(update(observations).where(observations.c.id=="obs-2").values(available_at=utcstr(at(11))))
    result = collect(parts)
    assert all(row["available_at"] <= utcstr(at(10)) for row in result["records"]["observations"])
    assert all(row["event_time"] <= utcstr(at(10)) for row in result["records"]["observations"])
    assert result["records"]["actual_events"][0]["event_revision"] == 1
    assert "FUTURE_REVISION" not in json.dumps(result)
    assert "foreign-run" not in json.dumps(result)
    assert len(result["records"]["samples"]) == 1
    assert result["spec"]["as_of"] == utcstr(at(10))


def test_missing_and_invalid_observations_do_not_fill_valid_monitoring_time(parts):
    with parts.store.transaction() as connection:
        connection.execute(update(observations).where(observations.c.id=="obs-5").values(value=None, quality_flags=["invalid_value"]))
    result = collect(parts)
    assert result["records"]["valid_monitoring_hours"] == pytest.approx(8/60)
    assert len(result["records"]["samples"]) == 1
    assert result["collection_summary"]["excluded_intervals"]["incomplete_observation_support"] == 1


def test_tasks_advice_and_actions_are_local_structured_audit_without_private_text(parts):
    result = collect(parts)
    exported = json.dumps(result)
    assert "PRIVATE_" not in exported and "F:/" not in exported
    tasks = {task["task_id"]:task for task in result["records"]["tasks"]}
    assert tasks["queued"]["state"] == "running" and tasks["queued"]["original_state"] == "queued"
    assert tasks["task"]["evidence_complete"] is True
    assert len(result["records"]["advice"]) == 1 and result["records"]["advice"][0]["advice_items"] == 1
    assert len(result["records"]["case_actions"]) == 1
    assert len(result["records"]["session_actions"]) == 1


def test_collection_is_read_only_and_reproducible(parts):
    kinds = ("session", "forecast_run", "event_revision", "task", "advice", "case_action", "session_action")
    before = {kind:deepcopy(parts.store.list(kind)) for kind in kinds}
    first, second = collect(parts), collect(parts)
    assert first["records"] == second["records"]
    assert before == {kind:parts.store.list(kind) for kind in kinds}
    assert first["artifact_refs"]["frozen_record_refs"]


def test_source_file_identifiers_are_not_presented_as_unverified_sha256(parts):
    result = collect(parts)
    assert "source_file:file" in result["spec"]["dataset_refs"]
    assert "sha256:file" not in result["spec"]["dataset_refs"]
    parts.store.put("data_file", "file", dict(file_id="file", sha256="a"*64, size=100, stored_path="PRIVATE_LOCAL_PATH"))
    verified = collect(parts)
    assert "sha256:" + "a"*64 in verified["spec"]["dataset_refs"]
    assert "PRIVATE_LOCAL_PATH" not in json.dumps(verified)


def test_multiple_model_versions_require_explicit_choice(parts):
    other = deepcopy(parts.run)
    other.update(run_id="v2-run", model_version="v2")
    parts.store.put("forecast_run", other["run_id"], other)
    with pytest.raises(DomainError):
        collect(parts, model_id="model")
    chosen = collect(parts, model_id="model", model_version="v1")
    assert {sample["run_id"] for sample in chosen["records"]["samples"]} == {"run"}


def test_maximum_target_uses_declared_statistic_and_not_mean(parts):
    model = parts.store.get("model", "model:v1")
    model["target_statistic"] = "interval_maximum"
    parts.store.put("model", "model:v1", model)
    result = collect(parts)
    assert result["records"]["samples"][0]["actual_value"] == .9
    assert result["records"]["samples"][0]["actual_positive"] is True


def test_unknown_professional_target_statistic_is_not_guessed(parts):
    model = parts.store.get("model", "model:v1")
    model.pop("target_statistic")
    parts.store.put("model", "model:v1", model)
    result = collect(parts)
    assert result["records"]["samples"] == []
    assert "target_statistic" in result["records"]["missing_fields"]


@pytest.mark.parametrize("change", [dict(roles=("ventilation",)), dict(face_ids=("other",))])
def test_collection_requires_evaluation_role_and_face_scope(parts, change):
    parts.context = parts.context.model_copy(update={"actor": parts.actor.model_copy(update=change)})
    with pytest.raises(DomainError) as error:
        collect(parts)
    assert error.value.status_code == 403


@pytest.mark.parametrize("selection", [["foreign"], ["wind"], []])
def test_foreign_nonmethane_or_empty_point_selection_is_rejected(parts, selection):
    with pytest.raises(DomainError):
        collect(parts, point_ids=selection)


def test_invalid_or_future_window_is_rejected_without_writes(parts):
    from app.services.evaluation.collector import SessionEvaluationCollector
    collector = SessionEvaluationCollector(parts.store, parts.services)
    for start,end in ((at(3),at(2)), (at(),at(13)), (at(-1),at(5))):
        with pytest.raises(DomainError):
            collector.collect(parts.context, start, end)


def test_no_completed_forecasts_keeps_observed_records_and_explicit_missing_fields(parts):
    parts.store.delete("forecast_run", "run")
    result = collect(parts)
    assert result["records"]["samples"] == [] and result["records"]["alert_events"] == []
    assert result["records"]["actual_events"]
    assert "forecast_runs" in result["records"]["missing_fields"]
    assert "mature_forecast_samples" in result["records"]["missing_fields"]


def test_no_observed_evaluation_is_missing_truth_instead_of_negative_labels(parts):
    parts.store.delete("event_revision", "actual@1")
    result = collect(parts)
    assert "observed_event_evaluation" in result["records"]["missing_fields"]
    assert result["collection_summary"]["observed_event_missing_point_ids"] == ["p"]
    parts.store.put("tick_result", "tick", dict(session=dict(session_id="session", source_id="source", as_of=utcstr(at(10))),
        assessments=[dict(point_id="p", observed_status="evaluated", status="evaluated")],
        forecasts=[dict(private="PRIVATE_TICK_PAYLOAD")]))
    result = collect(parts)
    assert "observed_event_evaluation" not in result["records"]["missing_fields"]
    assert result["collection_summary"]["observed_event_evaluated_point_ids"] == ["p"]
    assert len(result["records"]["tick_records"]) == 1
    assert "PRIVATE_TICK_PAYLOAD" not in json.dumps(result)


def test_half_open_left_edge_excludes_recovered_events_but_keeps_open_or_instant_points(parts):
    from app.services.evaluation.collector import SessionEvaluationCollector
    collector = SessionEvaluationCollector(parts.store,parts.services)
    recovered = collector.collect(parts.context,at(3),at(10))
    assert recovered["records"]["actual_events"] == []
    instant = {**parts.actual, "event_id":"instant", "start_at":utcstr(at(3)), "end_at":utcstr(at(3))}
    parts.store.put("event_revision", "instant@1", instant)
    assert [row["event_id"] for row in collector.collect(parts.context,at(3),at(10))["records"]["actual_events"]] == ["instant"]
    open_event = {**parts.actual, "end_at":None, "event_state":"new"}
    parts.store.put("event_revision", "actual@1", open_event)
    assert {row["event_id"] for row in collector.collect(parts.context,at(3),at(10))["records"]["actual_events"]} == {"actual","instant"}
