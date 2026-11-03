"""Frozen synthetic exports and explicit evaluation denominators."""

from copy import deepcopy

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError
from app.services.evaluation.service import EvaluationService


def event(event_id, start="08:00:00", end="08:01:00", point_id="31002:T0"):
    return {
        "event_id": event_id, "point_id": point_id,
        "start_at": f"2026-10-01T{start}+08:00",
        "end_at": f"2026-10-01T{end}+08:00",
    }


def spec(**changes):
    value = {
        "face_id": "31002", "dataset_refs": ["synthetic-export"], "dataset_split": "test",
        "model_version": "baseline-v1", "rule_version": "test-rule-v1",
        "preprocess_version": "test-preprocess-v1", "matching_config": {"tolerance_seconds": 0},
        "quality_policy": {"exclude_invalid_samples": True, "mape_zero_policy": "exclude_zero_actuals"},
        "task_set_ref": "synthetic-tasks",
    }
    value.update(changes)
    return value


def records(**changes):
    value = {
        "actual_events": [], "alert_events": [], "samples": [], "tasks": [],
        "valid_monitoring_hours": 0,
    }
    value.update(changes)
    return value


@pytest.fixture
def evaluation():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
    yield EvaluationService(store), actor
    store.close()


def test_freezing_records_copies_inputs_and_records_content_hash(evaluation):
    service, actor = evaluation
    inputs = records(actual_events=[event("a1")])
    original_spec = spec()

    manifest = service.freeze_manifest(actor, original_spec, {"frozen_records": inputs})
    inputs["actual_events"].clear()
    original_spec["model_version"] = "mutated"
    report = service.build_report(manifest["manifest_id"])

    assert len(manifest["records_sha256"]) == 64
    assert manifest["model_version"] == "baseline-v1"
    assert report["denominators"]["recall"] == 1
    assert report["metric_values"]["missed_events"] == 1
    assert report["metric_values"]["recall"] == 0
    assert service.get_report(actor, report["evaluation_id"]) == report


def test_build_report_rejects_replacement_of_frozen_records(evaluation):
    service, actor = evaluation
    inputs = records(actual_events=[event("a1")])
    manifest = service.freeze_manifest(actor, spec(), {"frozen_records": inputs})
    changed = deepcopy(inputs)
    changed["actual_events"].append(event("a2"))

    with pytest.raises(DomainError) as caught:
        service.build_report(manifest["manifest_id"], changed)

    assert caught.value.status_code == 409


def test_event_matching_maximizes_one_to_one_pairs_and_keeps_point_scope(evaluation):
    service, actor = evaluation
    inputs = records(
        actual_events=[
            event("a1", "08:00:00", "08:02:00"),
            event("a2", "08:03:00", "08:04:00"),
            event("a3", "08:20:00", "08:21:00"),
        ],
        alert_events=[
            event("p1", "08:01:00", "08:03:00"),
            event("p2", "08:01:00", "08:01:30"),
            event("p3", "08:00:00", "08:04:00", point_id="31002:T1"),
        ],
        valid_monitoring_hours=2,
    )

    report = service.run(actor, spec(), inputs)

    assert len(report["event_matches"]) == 2
    assert len({item["actual_event_id"] for item in report["event_matches"]}) == 2
    assert len({item["alert_event_id"] for item in report["event_matches"]}) == 2
    assert report["metric_values"]["precision"] == pytest.approx(2 / 3)
    assert report["metric_values"]["recall"] == pytest.approx(2 / 3)
    assert report["metric_values"]["missed_events"] == 1
    assert report["metric_values"]["false_alerts_per_valid_monitoring_hour"] == 0.5
    assert report["numerators"]["precision"] == 2
    assert report["denominators"]["precision"] == 3


def test_duplicate_alerts_cannot_reuse_one_actual_event(evaluation):
    service, actor = evaluation

    report = service.run(actor, spec(), records(
        actual_events=[event("a1")], alert_events=[event("p1"), event("p2")],
        valid_monitoring_hours=1,
    ))

    assert report["metric_values"]["recall"] == 1
    assert report["metric_values"]["precision"] == 0.5
    assert report["metric_values"]["false_alerts_per_valid_monitoring_hour"] == 1


def test_frozen_match_tolerance_is_applied(evaluation):
    service, actor = evaluation
    inputs = records(
        actual_events=[event("a1", "08:02:00", "08:03:00")],
        alert_events=[event("p1", "08:00:00", "08:01:30")],
    )

    strict = service.run(actor, spec(), inputs)
    tolerant = service.run(actor, spec(matching_config={"tolerance_seconds": 30}), inputs)

    assert strict["metric_values"]["recall"] == 0
    assert tolerant["metric_values"]["recall"] == 1


def test_sample_false_positive_rate_uses_valid_negative_sample_denominator(evaluation):
    service, actor = evaluation
    samples = [
        {"sample_id": str(index), "actual_positive": actual, "predicted_positive": predicted}
        for index, (actual, predicted) in enumerate([
            (True, True), (False, True), (False, False), (False, False), (True, False),
        ])
    ]
    samples.append({
        "sample_id": "invalid", "actual_positive": False, "predicted_positive": True, "valid": False,
    })

    report = service.run(actor, spec(), records(samples=samples))

    assert report["metric_values"]["sample_fpr"] == pytest.approx(1 / 3)
    assert report["denominators"]["sample_fpr"] == 3
    assert report["numerators"]["sample_fpr"] == 1
    assert report["metric_values"]["sample_precision"] == 0.5
    assert report["metric_values"]["sample_recall"] == 0.5


def test_mae_and_mape_expose_zero_actual_policy_and_denominators(evaluation):
    service, actor = evaluation
    samples = [
        {"sample_id": str(index), "actual_positive": False, "predicted_positive": False,
         "actual_value": actual, "predicted_value": predicted}
        for index, (actual, predicted) in enumerate([(0, 1), (10, 12), (20, 18)])
    ]

    report = service.run(actor, spec(), records(samples=samples))

    assert report["metric_values"]["mae"] == pytest.approx(5 / 3)
    assert report["metric_values"]["mape_with_zero_policy"] == pytest.approx(15)
    assert report["denominators"]["mae"] == 3
    assert report["denominators"]["mape_with_zero_policy"] == 2
    assert report["mape_zero_policy"] == "exclude_zero_actuals"


def test_normal_failed_and_partial_tasks_remain_in_completion_denominator(evaluation):
    service, actor = evaluation

    report = service.run(actor, spec(), records(tasks=[
        {"task_id": "ok", "state": "completed", "evidence_complete": True,
         "advice_items": 2, "referenced_advice_items": 1},
        {"task_id": "failed", "state": "failed"},
        {"task_id": "partial", "state": "partial"},
        {"task_id": "excluded", "state": "failed", "normal_task": False},
    ]))

    assert report["denominators"]["tool_task_completion_rate"] == 3
    assert report["metric_values"]["tool_task_completion_rate"] == pytest.approx(1 / 3)
    assert report["metric_values"]["evidence_completeness"] == pytest.approx(1 / 3)
    assert report["metric_values"]["advice_reference_coverage"] == 0.5


def test_empty_denominators_return_none(evaluation):
    service, actor = evaluation

    report = service.run(actor, spec(), records())

    for key, value in report["metric_values"].items():
        if key != "missed_events":
            assert value is None, key
    assert report["metric_values"]["missed_events"] == 0


@pytest.mark.parametrize("changes", [
    {"valid_monitoring_hours": -1},
    {"actual_events": [event("same"), event("same")]},
    {"samples": [{"sample_id": "s1", "actual_positive": False}]},
    {"tasks": [{"task_id": "t1", "state": "completed", "advice_items": 1, "referenced_advice_items": 2}]},
])
def test_malformed_exports_are_rejected_before_freezing(evaluation, changes):
    service, actor = evaluation

    with pytest.raises(DomainError):
        service.freeze_manifest(actor, spec(), {"frozen_records": records(**changes)})


def test_report_access_requires_safety_role_and_authorized_work_face(evaluation):
    service, actor = evaluation
    report = service.run(actor, spec(), records())
    for changes in ({"face_ids": ("32001",)}, {"roles": ("ventilation",)}):
        outsider = actor.model_copy(update=changes)
        with pytest.raises(DomainError) as caught:
            service.get_report(outsider, report["evaluation_id"])
        assert caught.value.status_code == 403


def test_calibration_views_keep_all_records_primary_and_match_filtered_records_again(evaluation):
    service, actor = evaluation
    suspected_actual = {**event("a-suspected"), "quality_flags": ["suspected_calibration"]}
    suspected_alert = {**event("p-suspected"), "quality_flags": ["suspected_calibration"]}
    normal_actual = event("a-normal", "08:10:00", "08:11:00")
    normal_alert = event("p-normal", "08:10:00", "08:11:00")
    inputs = records(
        actual_events=[suspected_actual, normal_actual],
        alert_events=[suspected_alert, normal_alert, event("p-false", "08:20:00", "08:21:00")],
        samples=[
            {"sample_id": "suspected", "actual_positive": False, "predicted_positive": True,
             "actual_value": 1, "predicted_value": 10, "quality_flags": ["suspected_calibration"]},
            {"sample_id": "normal-negative", "actual_positive": False, "predicted_positive": False,
             "actual_value": 2, "predicted_value": 3},
            {"sample_id": "normal-false", "actual_positive": False, "predicted_positive": True},
        ],
        valid_monitoring_hours=2,
        valid_monitoring_hours_excluding_suspected=1,
    )

    report = service.run(actor, spec(), inputs)

    primary = report["calibration_views"]["all_events"]
    sensitivity = report["calibration_views"]["excluding_suspected"]
    assert report["metric_values"] == primary["metric_values"]
    assert report["denominators"]["recall"] == 2
    assert primary["metric_values"]["precision"] == pytest.approx(2 / 3)
    assert primary["metric_values"]["sample_fpr"] == pytest.approx(2 / 3)
    assert sensitivity["denominators"]["recall"] == 1
    assert sensitivity["denominators"]["precision"] == 2
    assert sensitivity["metric_values"]["precision"] == 0.5
    assert sensitivity["metric_values"]["sample_fpr"] == 0.5
    assert sensitivity["metric_values"]["mae"] == 1
    assert sensitivity["metric_values"]["false_alerts_per_valid_monitoring_hour"] == 1
    assert sensitivity["excluded_refs"]["actual_events"] == ["a-suspected"]
    assert sensitivity["excluded_refs"]["samples"] == ["suspected"]
    assert len(sensitivity["event_matches"]) == 1


def test_calibration_sensitivity_does_not_invent_filtered_monitoring_hours(evaluation):
    service, actor = evaluation
    report = service.run(actor, spec(), records(
        actual_events=[{**event("a1"), "quality_flags": ["suspected_calibration"]}],
        valid_monitoring_hours=2,
    ))

    secondary = report["calibration_views"]["excluding_suspected"]

    assert report["metric_values"]["false_alerts_per_valid_monitoring_hour"] == 0
    assert secondary["denominators"]["false_alerts_per_valid_monitoring_hour"] is None
    assert secondary["metric_values"]["false_alerts_per_valid_monitoring_hour"] is None
    assert "valid_monitoring_hours_excluding_suspected" in secondary["missing_fields"]
