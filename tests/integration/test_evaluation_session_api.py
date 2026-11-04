"""Authenticated replay collection, frozen metric drilldown and local export."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import update

from app.config import Settings
from app.db import observations
from app.main import create_app

START = datetime(2022, 2, 5, 13, tzinfo=timezone.utc)
PASSWORD = "synthetic-evaluation-password"


@pytest.fixture
def api(tmp_path):
    application = create_app(Settings(_env_file=None, runtime_dir=tmp_path,
        database_url="sqlite+pysqlite:///:memory:", deepseek_api_key=None, event_brief_allowed_source_ids=[]))
    with TestClient(application) as client:
        services = application.state.services
        for username, role, face in (("manager", "safety_management", "31002"),
                                      ("reader", "ventilation", "31002"),
                                      ("other", "safety_management", "32001")):
            services["auth"].create_user(username, PASSWORD, [role], [face])
        token = client.post("/api/auth/login", json={"username": "manager", "password": PASSWORD}).json()["access_token"]
        client.headers["Authorization"] = "Bearer " + token
        assert client.post("/api/sources", json={"source_id": "synthetic-evaluation", "name": "Synthetic evaluation", "face_id": "31002"}).status_code == 200
        services["data"].register_source("other-source", "other", "32001")
        for point, source, quantity, unit in (("t0", "synthetic-evaluation", "methane", "%"),
                                              ("wind", "synthetic-evaluation", "wind_speed", "m/s"),
                                              ("foreign", "other-source", "methane", "%")):
            services["data"].register_point(point, source, point, point, quantity, unit)
        values = [.2, .3, .9, .9, .4, .1, .2, .7, .7, .7, .7, .7, .3, .2, .1]
        content = "STime,SValue\n" + "\n".join(f"{(START+timedelta(minutes=i)).isoformat()},{value}" for i, value in enumerate(values))
        upload = client.post("/api/files", files={"file": ("synthetic.csv", content.encode(), "text/csv")})
        imported = client.post("/api/history-imports", json={"source_id": "synthetic-evaluation", "file_id": upload.json()["file_id"],
            "mapping": {"time_column": "STime", "value_columns": {"SValue": "t0"}, "lineage": "raw"}})
        assert imported.status_code == 200, imported.text
        model = dict(model_id="qa-persistence", version="v1", adapter_kind="persistence", face_id="31002",
            source_id="synthetic-evaluation", point_ids=["t0"], target_quantity="methane", unit="%",
            aggregation_seconds=60, input_window=2, horizon_seconds=120, allowed_lineage=["raw"])
        assert client.post("/api/models", json=model).status_code == 200
        session = client.post("/api/sessions", json={"source_id": "synthetic-evaluation", "mode": "historical_replay", "start_as_of": START.isoformat()}).json()
        for minute in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
            response = client.post(f"/api/sessions/{session['session_id']}/tick", json={"target_as_of": (START+timedelta(minutes=minute)).isoformat()})
            assert response.status_code == 200, response.text
        spec = dict(session_id=session["session_id"], start_at=START.isoformat(),
                    end_at=(START+timedelta(minutes=12)).isoformat(), point_ids=["t0"],
                    model_id="qa-persistence", model_version="v1", dataset_split="unclassified", tolerance_seconds=0)
        yield client, services, spec


def freeze(api):
    client, _, spec = api
    response = client.post("/api/evaluations/from-session", json=spec)
    assert response.status_code == 200, response.text
    return response.json()


def test_session_collection_frozen_report_details_export_and_mutation_isolation(api):
    client, services, _ = api
    report = freeze(api)
    evaluation_id = report["evaluation_id"]
    manifest = client.get(f"/api/evaluations/{evaluation_id}/manifest")
    assert manifest.status_code == 200, manifest.text
    assert manifest.json()["records_sha256"] == report["records_sha256"]
    observed = manifest.json()["frozen_records"]["observations"][0]
    observation_path = f"/api/evaluations/{evaluation_id}/records/observations/{observed['id']}"
    original = client.get(observation_path)
    assert original.status_code == 200, original.text
    assert original.json()["record"]["value"] == observed["value"]
    details = client.get(f"/api/evaluations/{evaluation_id}/metrics/mae", params={"limit": 1})
    assert details.status_code == 200, details.text
    body = details.json()
    assert body["total"] > 0 and len(body["rows"]) == 1
    assert body["numerator"] == report["numerators"]["mae"] and body["denominator"] == report["denominators"]["mae"]
    assert body["records_sha256"] == report["records_sha256"]
    exported = client.get(f"/api/evaluations/{evaluation_id}/export")
    assert exported.status_code == 200 and "attachment" in exported.headers["content-disposition"]
    assert exported.json()["report"]["evaluation_id"] == evaluation_id
    assert exported.json()["manifest"]["records_sha256"] == report["records_sha256"]
    before = deepcopy(body)
    for run in services["store"].list("forecast_run"):
        run["outputs"] = []
        services["store"].put("forecast_run", run["run_id"], run)
    with services["store"].transaction() as connection:
        connection.execute(update(observations).where(observations.c.id == observed["id"]).values(value=999))
    assert client.get(f"/api/evaluations/{evaluation_id}").json() == report
    assert client.get(f"/api/evaluations/{evaluation_id}/metrics/mae", params={"limit": 1}).json() == before
    assert client.get(observation_path).json() == original.json()


@pytest.mark.parametrize("username", ["reader", "other"])
def test_session_reports_require_safety_role_and_assigned_face(api, username):
    client, _, spec = api
    report = freeze(api)
    token = client.post("/api/auth/login", json={"username": username, "password": PASSWORD}).json()["access_token"]
    client.headers["Authorization"] = "Bearer " + token
    assert client.post("/api/evaluations/from-session", json=spec).status_code == 403
    for suffix in ("", "/manifest", "/metrics/mae", "/export", "/records/observations/unknown"):
        assert client.get(f"/api/evaluations/{report['evaluation_id']}{suffix}").status_code == 403


def test_future_clock_foreign_point_and_caller_records_are_rejected(api):
    client, _, spec = api
    for changes, statuses in (({"end_at": (START+timedelta(hours=1)).isoformat()}, (400, 409)),
                              ({"point_ids": ["foreign"]}, (403,)),
                              ({"point_ids": ["wind"]}, (400, 409))):
        response = client.post("/api/evaluations/from-session", json={**spec, **changes})
        assert response.status_code in statuses, response.text
    assert client.post("/api/evaluations/from-session", json={**spec, "records": {"samples": []}}).status_code == 422


def test_metric_query_limits_and_unknown_metric_are_request_errors(api):
    client, _, _ = api
    report = freeze(api)
    path = f"/api/evaluations/{report['evaluation_id']}/metrics/mae"
    for query in ({"offset": -1}, {"limit": 0}, {"limit": 201}):
        assert client.get(path, params=query).status_code == 422
    assert client.get(path, params={"variant": "invalid"}).status_code in (400, 422)
    assert client.get(path.replace("/mae", "/unknown")).status_code in (400, 404)


def test_linked_replay_reproduces_metrics_and_keeps_original_session_clock(api):
    client, _, spec = api
    first = freeze(api)
    parent = client.get(f"/api/sessions/{spec['session_id']}").json()
    replayed = client.post("/api/sessions", json={"source_id": "synthetic-evaluation", "mode": "historical_replay",
        "start_as_of": START.isoformat(), "parent_session_id": spec["session_id"]})
    assert replayed.status_code == 200, replayed.text
    child = replayed.json()
    for minute in range(3, 13):
        assert client.post(f"/api/sessions/{child['session_id']}/tick",
            json={"target_as_of": (START+timedelta(minutes=minute)).isoformat()}).status_code == 200
    repeated = client.post("/api/evaluations/from-session", json={**spec, "session_id": child["session_id"]})
    assert repeated.status_code == 200, repeated.text
    second = repeated.json()
    for key in ("mae", "mape_with_zero_policy", "sample_precision", "sample_recall", "sample_fpr", "precision", "recall"):
        assert second["metric_values"][key] == first["metric_values"][key]
        assert second["numerators"][key] == first["numerators"][key]
        assert second["denominators"][key] == first["denominators"][key]
    assert client.get(f"/api/sessions/{spec['session_id']}").json() == parent
    assert client.get(f"/api/evaluations/{first['evaluation_id']}").json() == first
