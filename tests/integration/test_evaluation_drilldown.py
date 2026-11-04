"""Frozen reports expose auditable metric rows and detect content drift."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from threading import Barrier
from time import sleep

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError
from app.services.evaluation.service import EvaluationService


def event(identifier, minute=0, **extra):
    return {"event_id": identifier, "point_id": "t0", "start_at": f"2022-02-05T13:{minute:02}:00+00:00",
            "end_at": f"2022-02-05T13:{minute+1:02}:00+00:00", **extra}


def spec(**extra):
    return {"face_id": "31002", "dataset_refs": ["synthetic"], "dataset_split": "unclassified",
            "model_version": "qa-v1", "rule_version": "qa-rule", "preprocess_version": "qa-preprocess",
            "matching_config": {"tolerance_seconds": 0},
            "quality_policy": {"exclude_invalid_samples": True, "mape_zero_policy": "exclude_zero_actuals"},
            "task_set_ref": "qa-tasks", **extra}


def records(**extra):
    return {"actual_events": [], "alert_events": [], "samples": [], "tasks": [],
            "valid_monitoring_hours": 0, **extra}


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="qa-manager", roles=("safety_management",), face_ids=("31002",))
    yield EvaluationService(store), actor
    store.close()


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_report_manifest_and_repeat_build_have_stable_hashes(parts):
    service, actor = parts
    manifest = service.freeze_manifest(actor, spec(session_id="session", source_id="synthetic", point_ids=["t0"],
        collection_summary={"truth_semantics": "observed_threshold_derived"}), {"frozen_records": records()})
    report = service.build_report(manifest["manifest_id"])
    again = service.build_report(manifest["manifest_id"])

    assert again == report and len(service.store.list("evaluation_report")) == 1
    assert len(report["manifest_sha256"]) == len(report["report_sha256"]) == 64
    assert report["scope_metadata"]["session_id"] == "session"
    assert report["collection_summary"]["truth_semantics"] == "observed_threshold_derived"
    frozen = service.get_manifest(actor, report["evaluation_id"])
    assert frozen["records_sha256"] == report["records_sha256"]
    assert frozen["manifest_sha256"] == report["manifest_sha256"]
    assert frozen["frozen_records"] == records()


@pytest.mark.parametrize("target", ["records", "manifest", "report"])
def test_queries_and_exports_detect_tampering_after_report_creation(parts, target):
    service, actor = parts
    report = service.run(actor, spec(), records(actual_events=[event("a1")]))
    if target == "report":
        changed = deepcopy(report)
        changed["metric_values"]["missed_events"] = 999
        service.store.put("evaluation_report", report["evaluation_id"], changed)
    else:
        changed = service.store.get("evaluation_manifest", report["manifest_id"])
        if target == "records":
            changed["frozen_records"]["actual_events"].clear()
        else:
            changed["matching_config"]["tolerance_seconds"] = 999
        service.store.put("evaluation_manifest", changed["manifest_id"], changed)
    for operation in (lambda: service.get_report(actor, report["evaluation_id"]),
                      lambda: service.get_manifest(actor, report["evaluation_id"]),
                      lambda: service.get_metric_details(actor, report["evaluation_id"], "recall"),
                      lambda: service.export_report(actor, report["evaluation_id"])):
        with pytest.raises(DomainError) as caught:
            operation()
        assert caught.value.status_code == 409


def test_event_rows_explain_matches_misses_false_alerts_and_frozen_evidence(parts):
    service, actor = parts
    frozen = records(actual_events=[event("a1"), event("a2", 10)],
        alert_events=[event("p1", evidence_refs=[{"kind": "forecast_run", "run_id": "run1"}]),
                      event("p2", 20), event("p3", 30)], valid_monitoring_hours=2,
        monitoring_intervals=[{"point_id": "t0", "start_at": "2022-02-05T12:00:00+00:00",
                               "end_at": "2022-02-05T14:00:00+00:00", "hours": 2}])
    report = service.run(actor, spec(), frozen)
    frozen["alert_events"][0]["evidence_refs"].clear()
    precision = service.get_metric_details(actor, report["evaluation_id"], "precision")
    recall = service.get_metric_details(actor, report["evaluation_id"], "recall")
    misses = service.get_metric_details(actor, report["evaluation_id"], "missed_events")
    rate = service.get_metric_details(actor, report["evaluation_id"], "false_alerts_per_valid_monitoring_hour")

    assert (precision["numerator"], precision["denominator"], precision["total"]) == (1, 3, 3)
    assert sum(row["numerator_contribution"] for row in precision["rows"]) == 1
    assert any(row["snapshots"].get("alert_event", {}).get("evidence_refs") for row in precision["rows"])
    assert (recall["numerator"], recall["denominator"], recall["total"]) == (1, 2, 2)
    assert misses["rows"][0]["record"]["event_id"] == "a2"
    assert rate["value"] == 1 and rate["denominator_unit"] == "valid_monitoring_hours"
    assert any(row["kind"] == "monitoring_interval" and row["hours"] == 2 for row in rate["rows"])
    assert precision["records_sha256"] == report["records_sha256"]


def sample(identifier, actual, predicted, actual_value, predicted_value, **extra):
    return {"sample_id": identifier, "actual_positive": actual, "predicted_positive": predicted,
            "actual_value": actual_value, "predicted_value": predicted_value, **extra}


def test_sample_rows_reconcile_confusion_counts_errors_and_exclusions(parts):
    service, actor = parts
    report = service.run(actor, spec(), records(samples=[
        sample("tp", True, True, 2, 3), sample("fp", False, True, 0, 2),
        sample("tn", False, False, 0, 0), sample("fn", True, False, 2, 1),
        sample("invalid", False, True, 0, 99, valid=False),
    ]))
    precision = service.get_metric_details(actor, report["evaluation_id"], "sample_precision")
    mae = service.get_metric_details(actor, report["evaluation_id"], "mae")
    mape = service.get_metric_details(actor, report["evaluation_id"], "mape_with_zero_policy")

    assert {row["classification"] for row in precision["rows"]} == {"TP", "FP", "TN", "FN"}
    assert sum(row["numerator_contribution"] for row in precision["rows"]) == precision["numerator"] == 1
    assert sum(row["denominator_contribution"] for row in precision["rows"]) == precision["denominator"] == 2
    assert next(row for row in precision["rows"] if row["record"]["sample_id"] == "invalid")["included"] is False
    assert sum(row["numerator_contribution"] for row in mae["rows"]) == mae["numerator"] == 4
    assert mae["denominator"] == 4 and mae["value"] == 1
    assert mape["value"] == 50 and mape["denominator"] == 2
    assert sum(row["denominator_contribution"] for row in mape["rows"]) == 2


def test_task_rows_preserve_failed_normal_tasks_and_reference_coverage(parts):
    service, actor = parts
    report = service.run(actor, spec(), records(tasks=[
        {"task_id": "ok", "state": "completed", "evidence_complete": True,
         "advice_items": 2, "referenced_advice_items": 1, "evidence_refs": [{"kind": "event", "event_id": "a1"}]},
        {"task_id": "failed", "state": "failed"},
        {"task_id": "control", "state": "failed", "normal_task": False},
    ]))
    details = service.get_metric_details(actor, report["evaluation_id"], "tool_task_completion_rate")
    coverage = service.get_metric_details(actor, report["evaluation_id"], "advice_reference_coverage")
    assert details["value"] == .5 and details["denominator"] == 2
    assert next(row for row in details["rows"] if row["record"]["task_id"] == "control")["included"] is False
    assert (coverage["numerator"], coverage["denominator"]) == (1, 2)
    assert sum(row["numerator_contribution"] for row in coverage["rows"]) == 1


def test_unavailable_metric_reasons_and_calibration_denominator_are_explicit(parts):
    service, actor = parts
    report = service.run(actor, spec(), records(actual_events=[event("suspected", quality_flags=["suspected_calibration"])],
                                               valid_monitoring_hours=1))
    precision = service.get_metric_details(actor, report["evaluation_id"], "precision")
    secondary = service.get_metric_details(actor, report["evaluation_id"], "false_alerts_per_valid_monitoring_hour", "excluding_suspected")
    assert precision["status"] == "unavailable" and precision["reason"] == "no_alert_events"
    assert secondary["denominator"] is None and secondary["reason"] == "missing_valid_monitoring_hours"
    assert report["metric_statuses"]["precision"]["reason"] == "no_alert_events"
    assert service.get_metric_details(actor, report["evaluation_id"], "recall", "excluding_suspected")["total"] == 0


def test_metric_details_page_is_stable_and_validates_inputs(parts):
    service, actor = parts
    report = service.run(actor, spec(), records(actual_events=[event(f"a{i}", i*3) for i in range(5)]))
    first = service.get_metric_details(actor, report["evaluation_id"], "recall", offset=0, limit=2)
    second = service.get_metric_details(actor, report["evaluation_id"], "recall", offset=2, limit=2)
    assert first["total"] == second["total"] == 5
    assert {row["record"]["event_id"] for row in first["rows"]}.isdisjoint(row["record"]["event_id"] for row in second["rows"])
    for values in ({"metric": "invented"}, {"variant": "invented"}, {"offset": -1}, {"limit": 0}, {"limit": 201}):
        with pytest.raises(DomainError):
            service.get_metric_details(actor, report["evaluation_id"], **{"metric": "recall", **values})


@pytest.mark.parametrize("changes", [{"face_ids": ("other",)}, {"roles": ("ventilation",)}])
def test_frozen_manifest_details_and_export_enforce_report_scope(parts, changes):
    service, actor = parts
    report = service.run(actor, spec(), records())
    outsider = actor.model_copy(update=changes)
    for operation in (lambda: service.get_manifest(outsider, report["evaluation_id"]),
                      lambda: service.get_metric_details(outsider, report["evaluation_id"], "recall"),
                      lambda: service.export_report(outsider, report["evaluation_id"])):
        with pytest.raises(DomainError) as caught:
            operation()
        assert caught.value.status_code == 403


def test_export_redacts_private_fields_without_relabeling_original_hash(parts):
    service, actor = parts
    secret = "sk-private-placeholder-for-offline-tests"
    raw = records(tasks=[{"task_id": "t", "state": "failed", "api_key": secret,
                          "messages": [{"role": "user", "content": "private chat"}],
                          "evidence_refs": [{"kind": "observation", "source_file_id": "opaque-id"}],
                          "file_path": "F:/private-mine/full-source.csv"}])
    report = service.run(actor, spec(), raw)
    exported = service.export_report(actor, report["evaluation_id"])
    text = json.dumps(exported)
    assert secret not in text and "private chat" not in text and "F:/private-mine" not in text
    assert exported["manifest"]["records_sha256"] == digest(raw)
    assert exported["manifest"]["export_redacted"] is True
    assert exported["manifest"]["export_records_sha256"] == digest(exported["manifest"]["frozen_records"])
    assert exported["manifest"]["export_records_sha256"] != report["records_sha256"]
    assert exported["manifest"]["frozen_records"]["tasks"][0]["evidence_refs"][0]["source_file_id"] == "opaque-id"


def test_concurrent_builds_publish_one_frozen_report(parts, monkeypatch):
    service, actor = parts
    manifest = service.freeze_manifest(actor, spec(), {"frozen_records": records()})
    summarize = service._summary_view
    def slow_summary(*args):
        sleep(.02)
        return summarize(*args)
    monkeypatch.setattr(service, "_summary_view", slow_summary)
    barrier = Barrier(4)
    def build(_):
        barrier.wait()
        return service.build_report(manifest["manifest_id"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(build, range(4)))
    assert len({report["evaluation_id"] for report in reports}) == 1
    assert len(service.store.list("evaluation_report")) == 1


@pytest.mark.parametrize("method", ["get_manifest", "get_metric_details"])
def test_projection_rechecks_the_manifest_loaded_after_report_validation(parts, monkeypatch, method):
    service, actor = parts
    report = service.run(actor, spec(), records(actual_events=[event("a")]))
    original = service.get_report
    def validate_then_tamper(*args):
        result = original(*args)
        manifest = service.store.get("evaluation_manifest", report["manifest_id"])
        manifest["frozen_records"]["actual_events"].clear()
        service.store.put("evaluation_manifest", report["manifest_id"], manifest)
        return result
    monkeypatch.setattr(service, "get_report", validate_then_tamper)
    with pytest.raises(DomainError) as caught:
        getattr(service, method)(actor, report["evaluation_id"], *(('recall',) if method == 'get_metric_details' else ()))
    assert caught.value.status_code == 409


@pytest.mark.parametrize("missing,reason", [
    (["forecast_runs"], "no_forecast_runs"),
    (["mature_forecast_samples"], "no_mature_forecast_samples"),
    (["forecast_event_policy"], "forecast_event_policy_not_supported"),
    (["observed_event_evaluation"], "no_observed_assessments"),
])
def test_unavailable_forecast_evaluation_is_not_reported_as_model_misses(parts, missing, reason):
    service, actor = parts
    report = service.run(actor, spec(), records(actual_events=[event("a")], missing_fields=missing,
        samples=[sample("mature", False, False, .5, .6)] if missing in (["forecast_event_policy"], ["observed_event_evaluation"]) else []))
    assert report["metric_values"]["missed_events"] is None
    details = service.get_metric_details(actor, report["evaluation_id"], "missed_events")
    assert details["status"] == "unavailable" and details["reason"] == reason
    if missing in (["forecast_event_policy"], ["observed_event_evaluation"]):
        assert report["metric_values"]["mae"] == pytest.approx(.1)


def test_mape_distinguishes_missing_regression_samples_from_zero_actuals(parts):
    service, actor = parts
    empty = service.run(actor, spec(), records())
    zeros = service.run(actor, spec(), records(samples=[sample("zero", False, False, 0, .1)]))
    assert empty["metric_statuses"]["mape_with_zero_policy"]["reason"] == "no_regression_samples"
    assert zeros["metric_statuses"]["mape_with_zero_policy"]["reason"] == "zero_actuals_excluded"


def test_independent_advice_coverage_does_not_count_the_same_task_payload_again(parts):
    service, actor = parts
    report = service.run(actor, spec(), records(tasks=[{
        "task_id": "draft-tool", "state": "completed", "advice_items": 99, "referenced_advice_items": 99,
    }], advice=[{"advice_id": "a", "version": 1, "advice_items": 3, "referenced_advice_items": 2,
                 "reference_ids": ["reference-1", "reference-2"]}]))
    details = service.get_metric_details(actor, report["evaluation_id"], "advice_reference_coverage")
    assert details["value"] == pytest.approx(2 / 3)
    assert (details["numerator"], details["denominator"], details["total"]) == (2, 3, 1)
    assert details["rows"][0]["kind"] == "advice"
    assert details["rows"][0]["snapshots"]["advice"]["reference_ids"] == ["reference-1", "reference-2"]
