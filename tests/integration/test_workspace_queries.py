"""Workspace catalog and persisted forecasts stay scoped to their source and clock."""

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import insert

from app.config import Settings
from app.db import Store, observations
from app.main import create_app
from app.schemas.common import Actor, DomainError, ExecutionContext, utcstr
from app.services.data.service import DataService
from app.services.prediction.service import PredictionService


def at(minute=0):
    return datetime(2022, 2, 5, 8, minute, tzinfo=timezone.utc)


@pytest.fixture
def workspace(tmp_path):
    store = Store("sqlite+pysqlite:///:memory:")
    data = DataService(store, tmp_path / "imports")
    for source_id, face in (("source", "31002"), ("other", "32001")):
        data.register_source(source_id, source_id, face)
    for point_id, source_id, quantity, unit in (
        ("t0", "source", "methane", "%"), ("wind", "source", "wind_speed", "source_native"),
        ("empty", "source", "current", "A"), ("foreign", "other", "methane", "%"),
    ):
        data.register_point(point_id, source_id, point_id, point_id, quantity, unit)
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    context = ExecutionContext(session_id="session", mode="historical_replay", source_id="source", as_of=at(20), actor=actor)
    prediction = PredictionService(store, data)
    yield store, data, prediction, actor, context
    store.close()


def save_run(store, run_id, **changes):
    run = dict(run_id=run_id, session_id="session", source_id="source", face_id="31002", point_id="t0",
               as_of=utcstr(at(10)), created_at=utcstr(at(30)), model_id="model", model_version="v1",
               status="completed", outputs=[], input_refs=[{"id": "local-input"}])
    run.update(changes)
    store.put("forecast_run", run_id, run)
    return run


def test_forecast_list_filters_context_model_point_and_future_then_orders(workspace):
    store, _, prediction, _, context = workspace
    older = save_run(store, "older", as_of=utcstr(at(9)))
    latest = save_run(store, "latest", as_of=utcstr(at(15)))
    save_run(store, "future", as_of=utcstr(at(21)))
    save_run(store, "other-session", session_id="elsewhere")
    save_run(store, "other-source", source_id="other", face_id="32001")
    save_run(store, "other-face", face_id="32001")
    save_run(store, "other-point", point_id="wind")
    save_run(store, "other-model", model_id="different")
    save_run(store, "other-version", model_version="v2")

    result = prediction.list_forecasts(context, "t0", "model", "v1")

    assert result == [latest, older]
    assert prediction.list_forecasts(context, "t0", "model", "v1", limit=1) == [latest]
    assert prediction.list_forecasts(context, "t0", "unknown") == []


def test_forecast_list_authorizes_source_and_point_before_reading(workspace):
    _, _, prediction, actor, context = workspace
    with pytest.raises(DomainError) as denied:
        prediction.list_forecasts(context, "foreign")
    assert denied.value.status_code == 403
    outsider = actor.model_copy(update={"face_ids": ("32001",)})
    with pytest.raises(DomainError) as denied:
        prediction.list_forecasts(context.model_copy(update={"actor": outsider}))
    assert denied.value.status_code == 403
    unknown = actor.model_copy(update={"roles": ("unknown",)})
    with pytest.raises(DomainError) as denied:
        prediction.list_forecasts(context.model_copy(update={"actor": unknown}))
    assert denied.value.status_code == 403


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5])
def test_forecast_list_rejects_invalid_limit(workspace, limit):
    _, _, prediction, _, context = workspace
    with pytest.raises(DomainError):
        prediction.list_forecasts(context, limit=limit)


def add_observation(store, observation_id, point_id, time, value, source_id="source"):
    with store.transaction() as connection:
        connection.execute(insert(observations).values(id=observation_id, source_id=source_id,
            point_id=point_id, event_time=utcstr(time), available_at=utcstr(time+timedelta(minutes=10)),
            raw_value=str(value), value=value, quality_flags=[] if value is not None else ["invalid_value"],
            source_file_id="local-file", row_number=1, import_id="local-import"))


def test_coverage_summarizes_valid_history_and_empty_points_without_paths(workspace):
    store, data, _, actor, _ = workspace
    add_observation(store, "invalid-before", "t0", at(0), None)
    add_observation(store, "valid-1", "t0", at(1), .2)
    add_observation(store, "valid-2", "t0", at(2), .3)
    add_observation(store, "wind-1", "wind", at(3), 3.5)
    add_observation(store, "invalid-after", "t0", at(4), None)
    add_observation(store, "foreign", "foreign", at(5), .9, "other")

    result = data.get_coverage(actor, "source")

    assert result["semantics"] == "stored_history_catalog"
    assert result["validity_policy"] == "non_null_value"
    assert result["source_id"] == "source" and result["face_id"] == "31002"
    assert result["observation_count"] == 5
    assert result["valid_observation_count"] == 3
    assert result["start_at"] == utcstr(at(1)) and result["end_at"] == utcstr(at(3))
    points = {point["point_id"]: point for point in result["points"]}
    assert set(points) == {"t0", "wind", "empty"}
    assert points["t0"]["observation_count"] == 4
    assert points["t0"]["valid_observation_count"] == 2
    assert points["t0"]["end_at"] == utcstr(at(2))
    assert points["wind"]["quantity"] == "wind_speed" and points["wind"]["unit"] == "source_native"
    assert points["wind"]["unit_status"] == "inferred"
    assert points["empty"]["observation_count"] == 0
    assert points["empty"]["start_at"] is None and points["empty"]["end_at"] is None
    assert "stored_path" not in str(result) and "local-file" not in str(result)


def test_history_catalog_does_not_change_late_observation_visibility(workspace):
    store, data, _, actor, context = workspace
    add_observation(store, "late", "t0", at(0), .2)
    early = context.model_copy(update={"as_of": at(2)})

    assert data.get_coverage(actor, "source")["valid_observation_count"] == 1
    assert data.query_series(early, "t0", at(0), at(4))["rows"] == []
    arrived = early.model_copy(update={"as_of": at(10)})
    assert data.query_series(arrived, "t0", at(0), at(4))["rows"][0]["id"] == "late"


def test_coverage_requires_authorized_face_and_business_role(workspace):
    _, data, _, actor, _ = workspace
    with pytest.raises(DomainError) as denied:
        data.get_coverage(actor, "other")
    assert denied.value.status_code == 403
    with pytest.raises(DomainError) as denied:
        data.get_coverage(actor.model_copy(update={"roles": ("unknown",)}), "source")
    assert denied.value.status_code == 403


@pytest.fixture
def api(tmp_path):
    application = create_app(Settings(_env_file=None, runtime_dir=tmp_path, database_url="sqlite+pysqlite:///:memory:", deepseek_api_key=None, model_catalog_path=None))
    with TestClient(application) as client:
        services = application.state.services
        services["auth"].create_user("manager", "test-password-12345", ["safety_management"], ["31002"])
        token = client.post("/api/auth/login", json={"username":"manager", "password":"test-password-12345"}).json()["access_token"]
        client.headers["Authorization"] = "Bearer " + token
        services["data"].register_source("source", "Test source", "31002")
        services["data"].register_point("t0", "source", "T0", "T0", "methane", "%")
        session = client.post("/api/sessions", json=dict(mode="historical_replay", source_id="source", start_as_of=at(20).isoformat())).json()
        yield client, services, session


def test_query_routes_use_authenticated_context_and_validate_limit(api):
    client, services, session = api
    run = save_run(services["store"], "visible", session_id=session["session_id"])
    params = dict(session_id=session["session_id"], point_id="t0", model_id="model", version="v1", limit=1)
    response = client.get("/api/forecasts", params=params)
    assert response.status_code == 200, response.text
    assert response.json() == [run]
    assert client.get("/api/forecasts", params=dict(params, limit=101)).status_code == 422
    coverage = client.get("/api/sources/source/coverage")
    assert coverage.status_code == 200 and coverage.json()["observation_count"] == 0
    assert client.get("/api/sources/source/coverage", headers={"Authorization":"Bearer invalid"}).status_code == 401


def test_forecast_post_selects_exact_version_and_rejects_unbound_version(api):
    client, services, session = api
    actor = services["auth"].get_actor(client.headers["Authorization"][7:])
    for version in ("v1", "v2"):
        services["prediction"].register_model(actor, dict(model_id="baseline", version=version,
            adapter_kind="persistence", face_id="31002", source_id="source", point_ids=["t0"],
            target_quantity="methane", unit="%", aggregation_seconds=60, input_window=2, horizon_seconds=120))
    response = client.post("/api/forecasts", json=dict(session_id=session["session_id"], point_id="t0", model_id="baseline", version="v1"))
    assert response.status_code == 200, response.text
    assert response.json()["model_version"] == "v1"
    response = client.post("/api/forecasts", json=dict(session_id=session["session_id"], point_id="t0", version="v1"))
    assert response.status_code == 400
