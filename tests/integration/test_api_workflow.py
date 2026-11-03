from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app


@pytest.fixture
def api(tmp_path):
    application = create_app(Settings(_env_file=None, runtime_dir=tmp_path, database_url="sqlite+pysqlite:///:memory:"))
    with TestClient(application) as client:
        auth = application.state.services["auth"]
        auth.create_user("manager", "test-password-12345", ["safety_management"], ["31002"])
        auth.create_user("outsider", "test-password-12345", ["safety_management"], ["32001"])
        token = client.post("/api/auth/login", json={"username": "manager", "password": "test-password-12345"}).json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        yield client, application


def test_authenticated_import_replay_red_evidence_and_scoping(api):
    client, _ = api
    assert client.get("/api/sources", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/api/sources", json={"source_id": "s", "name": "Synthetic", "face_id": "31002"}).status_code == 200
    assert client.post("/api/points", json={"point_id": "p", "source_id": "s", "source_tag": "TAG", "name": "T0 fixture", "quantity": "methane", "unit": "%"}).status_code == 200
    upload = client.post("/api/files", files={"file": ("sample.csv", b"STime,SValue\n2022-01-01 00:00:00,0.1\n2022-01-01 00:01:00,0.9\n", "text/csv")})
    assert upload.status_code == 200
    imported = client.post("/api/history-imports", json={"source_id": "s", "file_id": upload.json()["file_id"], "mapping": {"time_column": "STime", "value_columns": {"SValue": "p"}, "timezone": "+08:00", "lineage": "raw"}})
    assert imported.status_code == 200, imported.text
    assert imported.json()["imported_rows"] == 2
    session = client.post("/api/sessions", json={"mode": "historical_replay", "source_id": "s", "start_as_of": "2022-01-01T00:00:00+08:00"}).json()
    sid = session["session_id"]
    query = dict(session_id=sid, point_id="p", start_at="2022-01-01T00:00:00+08:00", end_at="2022-01-01T01:00:00+08:00")
    assert client.get("/api/observations", params=query).json()["total"] == 1
    tick = client.post(f"/api/sessions/{sid}/tick", json={"target_as_of": "2022-01-01T00:01:00+08:00"})
    assert tick.status_code == 200, tick.text
    events = client.get("/api/events", params={"session_id": sid}).json()
    assert len(events) == 1 and events[0]["alert_level"] == "red"
    evidence = client.get("/api/evidence", params={"session_id": sid, "event_id": events[0]["event_id"]})
    assert evidence.status_code == 200
    assert evidence.json()["causal_status"] == "evidence_only"
    task = client.post("/api/chat", json={"session_id": sid, "message": "当前预警情况"})
    assert task.status_code == 200, task.text
    assert task.json()["method"] == "structured_template"
    token = client.post("/api/auth/login", json={"username": "outsider", "password": "test-password-12345"}).json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    assert client.get("/api/sessions/" + sid).status_code == 403
    assert client.get("/api/evidence", params={"session_id": sid, "event_id": events[0]["event_id"]}).status_code == 403


def test_validation_never_echoes_password_or_token(api):
    client, _ = api
    secret = "test-password-should-be-hidden"
    response = client.post("/api/auth/login", json={"username": [], "password": secret})
    assert response.status_code == 422
    assert secret not in response.text
    assert "input" not in response.text


def test_file_api_does_not_accept_arbitrary_server_path(api):
    client, _ = api
    response = client.post("/api/files", json={"path": "C:/Windows/win.ini"})
    assert response.status_code == 422
