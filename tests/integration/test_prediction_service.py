"""Synthetic prediction registry and closed-window acceptance checks."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.data.service import DataService
from app.services.prediction.service import PredictionService


def at(clock):
    return datetime.fromisoformat(f"2026-10-01T{clock}+08:00")


def manifest(**changes):
    result = {
        "model_id": "offline-persistence",
        "version": "v1",
        "adapter_kind": "persistence",
        "face_id": "31002",
        "source_id": "source",
        "point_ids": ["31002:T0"],
        "target_quantity": "methane",
        "unit": "%",
        "aggregation_seconds": 60,
        "input_window": 3,
        "horizon_seconds": 120,
        "allowed_lineage": ["raw"],
    }
    result.update(changes)
    return result


class ClosedData:
    def __init__(self):
        self.calls = []
        self.rows = []
        for index, value in enumerate((0.2, 0.4, 0.6)):
            start = at("08:00:00") + timedelta(minutes=index)
            self.rows.append({
                "event_time": start.astimezone(timezone.utc).isoformat(),
                "bucket_end": (start + timedelta(minutes=1)).astimezone(timezone.utc).isoformat(),
                "available_at": (start + timedelta(seconds=30)).astimezone(timezone.utc).isoformat(),
                "value": value,
                "count": 1,
                "quality_flags": [],
                "source_refs": [f"synthetic-row-{index}"],
            })

    def query_series(self, **kwargs):
        self.calls.append(kwargs)
        return {"rows": deepcopy(self.rows), "total": len(self.rows), "quality": {"flags": []}}


@pytest.fixture
def prediction():
    store = Store("sqlite+pysqlite:///:memory:")
    store.put("source", "source", {"source_id": "source", "face_id": "31002"})
    store.put("point", "31002:T0", {
        "point_id": "31002:T0", "source_id": "source", "face_id": "31002",
        "quantity": "methane", "unit": "%", "lineage": "raw", "mapping_status": "confirmed",
    })
    actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
    execution = ExecutionContext(
        session_id="session", mode="historical_replay", source_id="source",
        as_of=at("08:03:30"), config_version="test-v1", actor=actor,
    )
    data = ClosedData()
    service = PredictionService(store, data)
    yield service, data, actor, execution
    store.close()


def test_no_published_model_returns_persisted_not_ready_without_predictions(prediction):
    service, data, _, execution = prediction

    run = service.run_prediction(execution, "31002:T0")

    assert run["status"] == "not_ready"
    assert run["outputs"] == []
    assert "model_id" in run["missing_fields"]
    assert data.calls == []
    assert service.get_forecast(execution, run["run_id"]) == run


@pytest.mark.parametrize("changes", [
    {"input_window": 0}, {"input_window": True}, {"aggregation_seconds": -1},
    {"horizon_seconds": 65}, {"allowed_lineage": ["invented"]},
    {"point_ids": []}, {"model_type": "professional"},
])
def test_invalid_manifests_are_rejected_before_publication(prediction, changes):
    service, _, actor, execution = prediction

    with pytest.raises(DomainError):
        service.register_model(actor, manifest(**changes))

    assert service.list_models(execution) == []


def test_manifest_versions_are_immutable_and_baseline_is_explicit(prediction):
    service, _, actor, execution = prediction
    registered = service.register_model(actor, manifest())

    assert registered["model_type"] == "baseline"
    assert registered["purpose"] == "offline_pipeline_verification"
    assert service.get_model(execution, "offline-persistence")["version"] == "v1"
    with pytest.raises(DomainError) as caught:
        service.register_model(actor, manifest(input_window=4))
    assert caught.value.status_code == 409


def test_persistence_uses_exact_closed_window_and_preserves_forecast_provenance(prediction):
    service, data, actor, execution = prediction
    service.register_model(actor, manifest())

    run = service.run_prediction(execution, "31002:T0")

    assert run["status"] == "completed"
    assert run["model_id"] == "offline-persistence"
    assert run["model_version"] == "v1"
    assert datetime.fromisoformat(run["cutoff"]) == at("08:03:00")
    assert run["input_window"]["count"] == 3
    assert len(run["input_refs"]) == 3
    call = data.calls[0]
    assert call["start_at"] == at("08:00:00")
    assert call["end_at"] == at("08:03:00")
    assert call["aggregation_seconds"] == 60
    assert call["fill_max_seconds"] == 0
    output = run["outputs"][0]
    assert output["prediction_stage"] == "baseline_persistence"
    assert output["quantity_semantics"] == "physical_forecast"
    assert output["unit"] == "%"
    assert [item["value"] for item in output["interval_values"]] == [0.6, 0.6]
    assert datetime.fromisoformat(output["interval_values"][0]["start_at"]) == at("08:03:00")
    assert datetime.fromisoformat(output["interval_values"][-1]["end_at"]) == at("08:05:00")
    assert service.get_forecast(execution, run["run_id"]) == run


def test_persistence_can_use_interval_maximum_for_maximum_target(prediction):
    service, data, actor, execution = prediction
    data.rows[-1]["max"] = .9
    service.register_model(actor, manifest(input_statistic="max"))
    result = service.run_prediction(execution, "31002:T0")
    assert result["outputs"][0]["interval_values"][0]["value"] == .9


def test_experimental_model_is_explicitly_callable_but_not_default(prediction):
    service, _, actor, execution = prediction
    service.register_model(actor, manifest())
    service.register_model(actor, manifest(model_id="candidate", default_eligible=False))
    assert service.run_prediction(execution, "31002:T0")["model_id"] == "offline-persistence"
    assert service.run_prediction(execution, "31002:T0", "candidate")["model_id"] == "candidate"


@pytest.mark.parametrize("problem", ["short", "gap", "open", "late", "invalid", "processed", "unknown"])
def test_incomplete_or_invisible_or_disallowed_windows_return_not_ready(prediction, problem):
    service, data, actor, execution = prediction
    service.register_model(actor, manifest())
    if problem == "short":
        data.rows.pop()
    elif problem == "gap":
        data.rows[1]["event_time"] = at("08:00:00").isoformat()
    elif problem == "open":
        data.rows[-1]["bucket_end"] = at("08:04:00").isoformat()
    elif problem == "late":
        data.rows[-1]["available_at"] = at("08:04:00").isoformat()
    elif problem == "invalid":
        data.rows[1]["value"] = None
    else:
        data.rows[1]["quality_flags"] = [f"lineage_{problem}"]

    run = service.run_prediction(execution, "31002:T0")

    assert run["status"] == "not_ready"
    assert run["outputs"] == []
    assert run["missing_fields"]


def test_unavailable_professional_adapter_returns_not_ready(prediction):
    service, _, actor, execution = prediction
    service.register_model(actor, manifest(adapter_kind="fedformer", model_type="professional"))

    run = service.run_prediction(execution, "31002:T0")

    assert run["status"] == "not_ready"
    assert "model_adapter" in run["missing_fields"]
    assert run["outputs"] == []


def test_registered_adapter_keeps_concentration_and_decision_score_separate(prediction):
    service, _, actor, execution = prediction

    def professional(model, rows, horizon_seconds):
        start = at("08:03:00").astimezone(timezone.utc).isoformat()
        end = at("08:04:00").astimezone(timezone.utc).isoformat()
        return [
            {"prediction_stage": "residual_corrected", "quantity_semantics": "physical_forecast",
             "unit": "%", "interval_values": [{"start_at": start, "end_at": end, "value": 0.7}]},
            {"prediction_stage": "threshold_shifted", "quantity_semantics": "decision_score",
             "unit": "1", "interval_values": [{"start_at": start, "end_at": end, "value": 0.9}]},
        ]

    service.register_adapter("test-professional", professional)
    service.register_model(actor, manifest(
        adapter_kind="test-professional", model_type="professional", horizon_seconds=60,
    ))

    run = service.run_prediction(execution, "31002:T0")

    assert run["status"] == "completed"
    physical, score = run["outputs"]
    assert physical["unit"] == "%"
    assert score["unit"] == "1"
    assert physical["interval_values"][0]["value"] == 0.7
    assert score["interval_values"][0]["value"] == 0.9


@pytest.mark.parametrize("wrong_field", ["source_id", "actor"])
def test_prediction_enforces_source_and_work_face_scope(prediction, wrong_field):
    service, _, actor, execution = prediction
    service.register_model(actor, manifest())
    service.store.put("source", "another-source", {"source_id": "another-source", "face_id": "31002"})
    value = "another-source" if wrong_field == "source_id" else actor.model_copy(update={"face_ids": ("32001",)})
    other_context = execution.model_copy(update={wrong_field: value})

    with pytest.raises(DomainError) as caught:
        service.run_prediction(other_context, "31002:T0")

    assert caught.value.status_code == 403


def test_forecasts_cannot_be_read_from_another_session_or_earlier_cutoff(prediction):
    service, _, actor, execution = prediction
    service.register_model(actor, manifest())
    run = service.run_prediction(execution, "31002:T0")

    for changes in ({"session_id": "another-session"}, {"as_of": at("08:02:00")}):
        with pytest.raises(DomainError) as caught:
            service.get_forecast(execution.model_copy(update=changes), run["run_id"])
        assert caught.value.status_code == 403


@pytest.mark.parametrize("extra_processed", [False, True])
def test_real_data_import_excludes_partial_late_and_disallowed_lineage_before_prediction(tmp_path, extra_processed):
    store = Store("sqlite+pysqlite:///:memory:")
    try:
        actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
        execution = ExecutionContext(
            session_id="pipeline", mode="historical_replay", source_id="source",
            as_of=at("08:03:30"), actor=actor,
        )
        data = DataService(store, tmp_path / "uploads")
        data.register_source("source", "Synthetic pipeline", face_id="31002")
        data.register_point("31002:T0", "source", "SYNTHETIC.TAG", "Synthetic", "methane", "%")
        path = tmp_path / "pipeline.csv"
        path.write_text(
            "STime,SValue,AvailableAt\n"
            "2026-10-01T08:00:10+08:00,0.2,2026-10-01T08:00:10+08:00\n"
            "2026-10-01T08:01:10+08:00,0.4,2026-10-01T08:01:10+08:00\n"
            "2026-10-01T08:02:10+08:00,0.6,2026-10-01T08:02:10+08:00\n"
            "2026-10-01T08:02:30+08:00,77,2026-10-01T08:04:00+08:00\n"
            "2026-10-01T08:03:10+08:00,99,2026-10-01T08:03:10+08:00\n",
            encoding="utf-8",
        )
        registered = data.register_file(path, actor)
        data.import_history(actor, "source", registered["file_id"], {
            "time_column": "STime", "timezone": "+08:00", "value_columns": {"SValue": "31002:T0"},
            "availability_column": "AvailableAt", "header_row": 1, "sheet": None,
            "invalid_values": [-999999], "lineage": "raw",
        })
        if extra_processed:
            processed_path = tmp_path / "processed.csv"
            processed_path.write_text("STime,SValue\n2026-10-01T08:01:20+08:00,0.5\n", encoding="utf-8")
            processed_file = data.register_file(processed_path, actor)
            data.import_history(actor, "source", processed_file["file_id"], {
                "time_column": "STime", "timezone": "+08:00", "value_columns": {"SValue": "31002:T0"},
                "availability_column": None, "header_row": 1, "sheet": None,
                "invalid_values": [-999999], "lineage": "processed",
            })
        service = PredictionService(store, data)
        service.register_model(actor, manifest())

        run = service.run_prediction(execution, "31002:T0")

        if extra_processed:
            assert data.list_points(actor)[0]["lineage"] == "raw"
            assert run["status"] == "not_ready"
            assert run["outputs"] == []
            assert "valid_closed_input_window" in run["missing_fields"]
            return
        assert run["status"] == "completed"
        assert [row["value"] for row in run["input_refs"]] == [0.2, 0.4, 0.6]
        assert [item["value"] for item in run["outputs"][0]["interval_values"]] == [0.6, 0.6]
        assert all(row["source_refs"][0]["source_file_id"] == registered["file_id"] for row in run["input_refs"])
    finally:
        store.close()
