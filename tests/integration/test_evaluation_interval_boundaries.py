"""Target-window events use the same right-open interval as collected samples."""
import pytest

from app.db import Store
from app.schemas.common import Actor
from app.services.evaluation.service import EvaluationService


@pytest.mark.parametrize("semantics,actual_time,expected_matches", [
    ("prediction_target_windows", "08:01:00", 0),
    ("prediction_target_windows", "08:00:00", 1),
    ("legacy", "08:01:00", 1),
])
def test_event_report_and_details_agree_on_prediction_target_boundaries(semantics, actual_time, expected_matches):
    store = Store("sqlite+pysqlite:///:memory:")
    service = EvaluationService(store)
    actor = Actor(user_id="qa", roles=("safety_management",), face_ids=("31002",))
    moment = f"2022-02-05T{actual_time}+08:00"
    spec = dict(face_id="31002", dataset_refs=["synthetic"], dataset_split="unclassified",
        model_version="qa-v1", rule_version="qa-rule", preprocess_version="qa",
        matching_config={"tolerance_seconds": 0, "time_semantics": semantics},
        quality_policy={"exclude_invalid_samples": True, "mape_zero_policy": "exclude_zero_actuals"}, task_set_ref="synthetic")
    records = dict(actual_events=[dict(event_id="actual", point_id="t0", start_at=moment, end_at=moment)],
        alert_events=[dict(event_id="target", point_id="t0", start_at="2022-02-05T08:00:00+08:00", end_at="2022-02-05T08:01:00+08:00")],
        samples=[], tasks=[], valid_monitoring_hours=1)
    try:
        report = service.run(actor, spec, records)
        assert len(report["event_matches"]) == expected_matches
        assert report["numerators"]["precision"] == expected_matches
        details = service.get_metric_details(actor, report["evaluation_id"], "precision")
        assert details["rows"][0]["matched"] == bool(expected_matches)
        assert details["numerator"] == expected_matches
    finally:
        store.close()
