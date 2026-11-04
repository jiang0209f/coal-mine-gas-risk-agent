"""Chat binds the selected point on the server before invoking any tool."""
from datetime import timedelta

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app
from app.schemas.common import now_utc


@pytest.fixture
def client(tmp_path):
    app = create_app(Settings(_env_file=None, runtime_dir=tmp_path, database_url="sqlite+pysqlite:///:memory:",
        realtime_monitor_enabled=False, ingestion_worker_enabled=False, deepseek_api_key=None,
        event_brief_allowed_source_ids=[]))
    with TestClient(app) as client:
        services = app.state.services
        services["auth"].create_user("qa", "synthetic-chat-password", ["safety_management"], ["synthetic"])
        for sid in ("one", "other"):
            services["data"].register_source(sid, sid, "synthetic")
        for pid, sid in (("t0", "one"), ("t1", "one"), ("foreign", "other")):
            services["data"].register_point(pid, sid, pid, pid, "methane", "%")
        login = client.post("/api/auth/login", json={"username": "qa", "password": "synthetic-chat-password"})
        client.headers["Authorization"] = "Bearer " + login.json()["access_token"]
        session = client.post("/api/sessions", json={"source_id": "one", "mode": "historical_replay",
            "start_as_of": (now_utc()-timedelta(hours=1)).isoformat()}).json()
        yield client, services, session["session_id"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_selected_point_is_bound_for_sync_and_async_questions(client, asynchronous):
    client, services, session_id = client
    response = client.post("/api/chat", json={"session_id": session_id, "point_id": "t1",
        "message": "查看最近一小时监测数据", "async_execution": asynchronous})
    assert response.status_code == 200, response.text
    task = services["store"].get("task", response.json()["task_id"])
    assert task["selected_point_id"] == "t1"
    assert task["tool_calls"][0]["name"] == "query_series"


def test_foreign_point_and_conflicting_embedded_point_are_rejected_before_task_creation(client):
    client, services, session_id = client
    response = client.post("/api/chat", json={"session_id": session_id, "point_id": "foreign", "message": "查询监测"})
    assert response.status_code == 403
    response = client.post("/api/chat", json={"session_id": session_id, "point_id": "t0",
        "message": {"message": "查询监测", "selected_point_id": "t1"}})
    assert response.status_code == 400
    assert services["store"].list("task") == []
