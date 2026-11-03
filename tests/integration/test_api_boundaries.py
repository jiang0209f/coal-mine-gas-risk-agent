from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app


@pytest.fixture
def api(tmp_path):
    application = create_app(Settings(_env_file=None, runtime_dir=tmp_path, database_url="sqlite://"))
    with TestClient(application, raise_server_exceptions=False) as client:
        auth = application.state.services["auth"]
        auth.create_user("manager", "fixture-password-12345", ["safety_management"], ["31002"])
        auth.create_user("ventilation", "fixture-password-12345", ["ventilation"], ["31002"])
        token = client.post("/api/auth/login", json={"username": "manager", "password": "fixture-password-12345"}).json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        for source in ("source-1", "source-2"):
            assert client.post("/api/sources", json={"source_id": source, "name": source, "face_id": "31002"}).status_code == 200
        assert client.post("/api/points", json={"point_id": "point", "source_id": "source-1", "source_tag": "TAG", "name": "T0", "quantity": "methane", "unit": "%"}).status_code == 200
        yield client, application


def new_session(client, *, source_id="source-1", mode="historical_replay", start=None):
    start = start or datetime(2022, 1, 1, tzinfo=timezone.utc)
    response = client.post("/api/sessions", json={"source_id": source_id, "mode": mode, "start_as_of": start.isoformat()})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


@pytest.mark.parametrize("entry", ["create", "action", "tick"])
def test_realtime_context_cannot_be_advanced_to_client_future(api, entry):
    client, _ = api
    now = datetime.now(timezone.utc)
    future = (now + timedelta(days=365)).isoformat()
    if entry == "create":
        response = client.post("/api/sessions", json={"source_id": "source-1", "mode": "realtime", "start_as_of": future})
    else:
        session_id = new_session(client, mode="realtime", start=now - timedelta(seconds=1))
        if entry == "action":
            response = client.post(f"/api/sessions/{session_id}/actions", json={"action": "advance", "target_as_of": future})
        else:
            response = client.post(f"/api/sessions/{session_id}/tick", json={"target_as_of": future})
    assert 400 <= response.status_code < 500


def test_long_replay_tick_does_not_skip_earlier_red_pulse(api):
    client, _ = api
    uploaded = client.post("/api/files", files={"file": ("pulse.csv", b"STime,SValue\n2022-01-01T00:00:00+00:00,0.1\n2022-01-01T00:01:00+00:00,0.9\n2022-01-01T00:02:00+00:00,0.1\n2022-01-01T00:30:00+00:00,0.1\n", "text/csv")}).json()
    imported = client.post("/api/history-imports", json={"source_id": "source-1", "file_id": uploaded["file_id"], "mapping": {"time_column": "STime", "value_columns": {"SValue": "point"}, "lineage": "raw"}})
    assert imported.status_code == 200, imported.text
    session_id = new_session(client)
    tick = client.post(f"/api/sessions/{session_id}/tick", json={"target_as_of": "2022-01-01T00:30:00+00:00"})
    assert tick.status_code == 200, tick.text
    events = client.get("/api/events", params={"session_id": session_id}).json()
    assert any(event["alert_level"] == "red" and event["start_at"] == "2022-01-01T00:01:00.000000+00:00" for event in events)


def test_same_face_other_source_point_is_not_queryable(api):
    client, _ = api
    session_id = new_session(client, source_id="source-2")
    response = client.get("/api/observations", params={"session_id": session_id, "point_id": "point", "start_at": "2022-01-01T00:00:00Z", "end_at": "2022-01-02T00:00:00Z"})
    assert response.status_code == 403


def test_malformed_nested_reference_input_is_client_error(api):
    client, _ = api
    response = client.post("/api/references/imports", json={"spec": []})
    assert 400 <= response.status_code < 500
    assert response.headers["content-type"].startswith("application/json")


def test_ventilation_role_cannot_review_case_advice(api):
    client, application = api
    session_id = new_session(client)
    event = {"event_id": "event", "session_id": session_id, "source_id": "source-1", "point_id": "point", "event_type": "observed_exceedance", "start_at": "2022-01-01T00:00:00.000000+00:00", "event_time": "2022-01-01T00:00:00.000000+00:00", "revision_as_of": "2022-01-01T00:00:00.000000+00:00", "event_revision": 1}
    application.state.services["store"].put("event_revision", "event@00000001", event)
    token = client.post("/api/auth/login", json={"username": "ventilation", "password": "fixture-password-12345"}).json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    response = client.post("/api/cases/event/actions", json={"session_id": session_id, "action_type": "review_advice", "operation_context": "rehearsal", "content": "人工审核。"})
    assert response.status_code == 403


def test_duplicate_tick_reuses_forecast_and_async_chat_exposes_task(api):
    client, application = api
    sid = new_session(client)
    body = {"target_as_of": "2022-01-01T00:01:00Z"}
    first = client.post(f"/api/sessions/{sid}/tick", json=body)
    second = client.post(f"/api/sessions/{sid}/tick", json=body)
    assert first.status_code == second.status_code == 200
    assert len(application.state.services["store"].list("forecast_run")) == 1
    submitted = client.post("/api/chat", json={"session_id": sid, "message": "查询当前事件", "async_execution": True})
    assert submitted.status_code == 200
    assert submitted.json()["state"] == "running"
    fetched = client.get("/api/tasks/" + submitted.json()["task_id"])
    assert fetched.status_code == 200
    assert fetched.json()["state"] == "completed"
    assert "messages" not in fetched.json() and "context" not in fetched.json()
