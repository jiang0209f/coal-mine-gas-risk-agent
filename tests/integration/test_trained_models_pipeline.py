"""Trained models consume server-built, source-bound, past-only features."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from app.adapters.model_features import CHANNEL_NAMES, FEATURE_PIPELINE_VERSION, FORECAST_FEATURE_NAMES, RESIDUAL_FEATURE_NAMES
from app.db import Store
from app.schemas.common import Actor, ExecutionContext, DomainError
from app.services.prediction.service import PredictionService
from app.services.prediction.trained import TrainedModelsPipeline


def at(hour, minute=0):
    return datetime(2022, 2, 5, hour, minute, tzinfo=timezone.utc)


class MinuteData:
    def __init__(self):
        self.missing = False
        self.calls = []

    def query_series(self, context, point_id, start_at, end_at, **kwargs):
        self.calls.append((point_id, start_at, end_at, context.as_of))
        channel = int(point_id[-1])
        time = start_at
        rows = []
        while time < end_at and time <= context.as_of:
            if not (self.missing and channel == 1 and time == start_at):
                rows.append(dict(id=f"{point_id}@{time.isoformat()}", point_id=point_id,
                    source_file_id="source-file", row_number=int(time.timestamp()/60),
                    event_time=time.isoformat(), available_at=time.isoformat(), value=.1+channel+.001*(time.minute%5),
                    quality_flags=["lineage_processed"]))
            time += timedelta(minutes=1)
        return dict(rows=rows, total=len(rows))


class Runtime:
    def __init__(self, points):
        self.calls = []
        self.manifests = {}
        for kind, mid in [("gas_forecast", "forecast"), ("gas_attribution", "attribution")]:
            self.manifests[mid] = dict(kind=kind, model_id=mid, version="v1", feature_points=points,
                feature_pipeline_version=FEATURE_PIPELINE_VERSION, feature_names=FORECAST_FEATURE_NAMES if mid=="forecast" else CHANNEL_NAMES,
                residual_feature_names=RESIDUAL_FEATURE_NAMES, horizon_seconds=3600 if mid=="forecast" else 300,
                automatic_attribution=True,
                training_range=dict(start_at="2022-01-01T00:00:00+00:00", end_at="2022-02-01T00:00:00+00:00"))

    def describe_packages(self):
        return [dict(model_id=k, model_version="v1", kind=v["kind"], source_id="source", face_id="31002") for k,v in self.manifests.items()]

    def get_spec(self, model_id, version):
        return dict(manifest=deepcopy(self.manifests[model_id]), source_id="source", face_id="31002")

    def infer(self, model_id, version, request):
        self.calls.append((model_id, version, deepcopy(request)))
        if model_id == "attribution":
            return dict(status="completed", model_id=model_id, model_version=version,
                        causal_status="candidate_attribution", candidate_causes=["候选原因"], target_time=(datetime.fromisoformat(request["as_of"])+timedelta(minutes=5)).isoformat())
        cutoff = datetime.fromisoformat(request["as_of"])
        return dict(status="completed", model_id=model_id, model_version=version, outputs=[
            dict(prediction_stage="residual_corrected", quantity_semantics="physical_forecast", unit="%",
                 interval_values=[dict(start_at=(cutoff+timedelta(minutes=5*i)).isoformat(),
                                      end_at=(cutoff+timedelta(minutes=5*(i+1))).isoformat(), value=.2) for i in range(12)])])


@pytest.fixture
def trained():
    store = Store("sqlite+pysqlite:///:memory:")
    store.put("source", "source", dict(source_id="source", face_id="31002"))
    points = dict(zip(CHANNEL_NAMES, [f"p{i}" for i in range(6)]))
    for i, point in enumerate(points.values()):
        store.put("point", point, dict(point_id=point, source_id="source", face_id="31002",
                  mapping_status="confirmed", lineage="processed", quantity="methane" if i==0 else "signal", unit="%" if i==0 else "source_native"))
    data = MinuteData()
    runtime = Runtime(points)
    pipeline = TrainedModelsPipeline(store, data, runtime)
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    context = ExecutionContext(session_id="session", mode="historical_replay", source_id="source", as_of=at(8,2), actor=actor)
    yield store, data, runtime, pipeline, context, points
    store.close()


def model(points):
    return dict(model_id="forecast", version="v1", adapter_kind="fedformer_lgbm", face_id="31002",
                source_id="source", point_ids=["p0"], target_quantity="methane", unit="%",
                aggregation_seconds=300, input_window=90, horizon_seconds=3600, allowed_lineage=["processed"],
                feature_pipeline_version=FEATURE_PIPELINE_VERSION, feature_points=points)


def test_standard_prediction_service_uses_six_channel_server_features(trained):
    store, data, runtime, pipeline, context, points = trained
    service = PredictionService(store, data, professional_pipeline=pipeline)
    service.register_model(context.actor, model(points))
    result = service.run_prediction(context, "p0")
    assert result["status"] == "completed"
    request = runtime.calls[-1][2]
    assert datetime.fromisoformat(request["as_of"]) == at(8)
    assert request["values"][-1][-1] == pytest.approx(.104)
    assert len(result["input_refs"]) == 2700
    assert all(c[3] == at(8) for c in data.calls)
    assert result["input_origin"] == "server_observations"


def test_missing_secondary_channel_is_not_ready_without_model_call(trained):
    store, data, runtime, pipeline, context, points = trained
    data.missing = True
    service = PredictionService(store, data, professional_pipeline=pipeline)
    service.register_model(context.actor, model(points))
    result = service.run_prediction(context, "p0")
    assert result["status"] == "not_ready"
    assert result["missing_fields"]
    assert runtime.calls == []


def test_catalog_bound_points_cannot_be_replaced_by_model_registration(trained):
    store, data, runtime, pipeline, context, points = trained
    bad = model(dict(points, **{CHANNEL_NAMES[1]: "foreign"}))
    with pytest.raises(DomainError):
        pipeline.forecast(context, "p0", bad, 3600)
    assert runtime.calls == []
    assert data.calls == []


def test_training_future_cutoff_is_not_ready(trained):
    store, data, runtime, pipeline, context, points = trained
    earlier = context.model_copy(update={"as_of": datetime(2022,1,20,tzinfo=timezone.utc)})
    service = PredictionService(store, data, professional_pipeline=pipeline)
    service.register_model(context.actor, model(points))
    assert service.run_prediction(earlier, "p0")["status"] == "not_ready"
    assert runtime.calls == []


def test_event_candidate_uses_observed_target_and_preceding_horizon(trained):
    _, data, runtime, pipeline, context, _ = trained
    event = dict(event_id="event", event_revision=2, event_type="observed_exceedance", point_id="p0",
                 start_at=at(7,55).isoformat(), revision_as_of=context.as_of.isoformat())
    fragments = [dict(id="target", point_id="p0", event_time=at(8).isoformat(), available_at=at(8).isoformat(), value=.7)]
    result = pipeline.attribute_event(context, event, fragments)
    assert result["status"] == "completed"
    assert datetime.fromisoformat(runtime.calls[-1][2]["as_of"]) == at(7,55)
    assert result["target_observation_id"] == "target"
    assert result["event_revision"] == 2
    assert result["causal_status"] == "candidate_attribution"


def test_forecast_event_is_not_mislabeled_as_observed_cause(trained):
    _, _, runtime, pipeline, context, _ = trained
    result = pipeline.attribute_event(context, dict(event_type="forecast_exceedance"), [])
    assert result["status"] == "not_ready"
    assert runtime.calls == []


def test_disabled_attribution_is_not_automatically_used(trained):
    _, _, runtime, pipeline, context, _ = trained
    runtime.manifests["attribution"]["automatic_attribution"] = False
    event = dict(event_id="event", event_revision=1, event_type="observed_exceedance", point_id="p0")
    fragments = [dict(id="target", point_id="p0", event_time=at(8).isoformat(), value=.7)]
    result = pipeline.attribute_event(context, event, fragments)
    assert result["missing_fields"] == ["automatic_attribution_enabled"]
    assert runtime.calls == []


def test_same_event_revision_reuses_persisted_attribution_and_respects_switch(trained):
    _, _, runtime, pipeline, context, _ = trained
    event = dict(event_id="event", event_revision=1, event_type="observed_exceedance", point_id="p0")
    fragments = [dict(id="target", point_id="p0", event_time=at(8).isoformat(), value=.7)]
    first = pipeline.attribute_event(context, event, fragments)
    repeated = pipeline.attribute_event(context, event, fragments)
    assert first["run_id"] == repeated["run_id"]
    assert repeated["reused"] is True
    assert len(runtime.calls) == 1
    revised = pipeline.attribute_event(context, dict(event, event_revision=2), fragments)
    assert revised["run_id"] != first["run_id"]
    assert len(runtime.calls) == 2
    runtime.manifests["attribution"]["automatic_attribution"] = False
    disabled = pipeline.attribute_event(context, event, fragments)
    assert disabled["status"] == "not_ready"
    assert len(runtime.calls) == 2
