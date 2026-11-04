"""Authenticated durable ingestion receipts, local restart recovery and source scope."""
from datetime import datetime, timedelta, timezone
import time

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import func, select

from app.config import Settings
from app.db import observations
from app.main import create_app

PASSWORD = "synthetic-ingestion-password"


def login(client, username="manager"):
    response = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200
    client.headers["Authorization"] = "Bearer " + response.json()["access_token"]


@pytest.fixture
def api(tmp_path):
    app = create_app(Settings(_env_file=None, runtime_dir=tmp_path,
        database_url="sqlite+pysqlite:///:memory:", deepseek_api_key=None,
        event_brief_allowed_source_ids=[], ingestion_worker_enabled=False))
    with TestClient(app) as client:
        services = app.state.services
        for name, role, face in (("manager", "safety_management", "31002"),
                                  ("reader", "ventilation", "31002"), ("other", "safety_management", "32001")):
            services["auth"].create_user(name, PASSWORD, [role], [face])
        for source, face in (("live-qa", "31002"), ("other-source", "32001")):
            services["data"].register_source(source, source, face)
        services["data"].register_point("t0", "live-qa", "T0", "T0", "methane", "%", unit_status="confirmed")
        services["data"].register_point("foreign", "other-source", "OTHER", "OTHER", "methane", "%")
        login(client)
        yield client, services


def connector(client):
    response = client.post("/api/ingestion/connectors", json={"name": "本地统一接入", "source_id": "live-qa", "adapter_kind": "standard_json"})
    assert response.status_code == 200, response.text
    return response.json()


def payload(value="0.70", reading_id="row-1"):
    return {"readings": [{"reading_id": reading_id, "point_id": "t0",
        "event_time": (datetime.now(timezone.utc)-timedelta(seconds=10)).isoformat(), "raw_value": value}]}


def row_count(services):
    with services["store"].transaction() as connection:
        return connection.execute(select(func.count()).select_from(observations)).scalar_one()


def test_batch_receipt_is_durable_paused_then_applied_once_without_payload_leak(api):
    client, services = api
    saved = connector(client)
    cid = saved["connector_id"]
    assert saved["enabled"] is False
    body = {"batch_id": "batch-1", "payload": payload()}
    response = client.post(f"/api/ingestion/connectors/{cid}/batches", json=body)
    assert response.status_code in (200, 202), response.text
    receipt = response.json()
    assert receipt["durable"] and receipt["batch_id"] == "batch-1" and receipt["connector_id"] == cid
    assert receipt["status"] == "pending" and row_count(services) == 0
    assert client.post(f"/api/ingestion/connectors/{cid}/batches", json=body).json()["durable"]
    assert client.post(f"/api/ingestion/connectors/{cid}/actions", json={"action": "enable"}).status_code == 200
    services["ingestion"].process_pending()
    applied = client.get(f"/api/ingestion/connectors/{cid}/batches/batch-1")
    assert applied.status_code == 200 and applied.json()["status"] == "completed"
    assert row_count(services) == 1
    assert client.post(f"/api/ingestion/connectors/{cid}/batches", json=body).json()["durable"]
    services["ingestion"].process_pending()
    assert row_count(services) == 1
    batches = client.get(f"/api/ingestion/connectors/{cid}/batches").json()
    assert batches["total"] == 1 and len(batches["items"]) == 1
    assert not any(key in applied.json() for key in ("payload", "readings", "normalized_readings", "messages"))
    status = client.get("/api/ingestion/status", params={"source_id": "live-qa"})
    assert status.status_code == 200 and status.json()["connectors"][0]["counts"]["ingested_rows"] == 1


def test_batch_identifier_conflict_does_not_overwrite_saved_reading(api):
    client, services = api
    cid = connector(client)["connector_id"]
    original = payload()
    body = {"batch_id": "batch-1", "payload": original}
    assert client.post(f"/api/ingestion/connectors/{cid}/batches", json=body).status_code in (200, 202)
    different = {"readings": [{**original["readings"][0], "raw_value": "0.90"}]}
    assert client.post(f"/api/ingestion/connectors/{cid}/batches", json={**body, "payload": different}).status_code == 409
    client.post(f"/api/ingestion/connectors/{cid}/actions", json={"action": "enable"})
    services["ingestion"].process_pending()
    with services["store"].transaction() as connection:
        value = connection.execute(select(observations.c.raw_value)).scalar_one()
    assert value == "0.70"


def test_roles_source_scope_and_server_owned_fields_are_enforced(api):
    client, _ = api
    cid = connector(client)["connector_id"]
    login(client, "reader")
    assert client.get("/api/ingestion/connectors").status_code == 200
    assert client.get(f"/api/ingestion/connectors/{cid}").status_code == 200
    assert client.post("/api/ingestion/connectors", json={"name": "denied", "source_id": "live-qa"}).status_code == 403
    assert client.post(f"/api/ingestion/connectors/{cid}/actions", json={"action": "enable"}).status_code == 403
    assert client.post(f"/api/ingestion/connectors/{cid}/batches", json={"batch_id": "denied", "payload": payload()}).status_code == 403
    login(client, "other")
    assert client.get(f"/api/ingestion/connectors/{cid}").status_code == 403
    assert client.get("/api/ingestion/status", params={"source_id": "live-qa"}).status_code == 403
    login(client)
    assert client.post(f"/api/ingestion/connectors/{cid}/batches", json={"batch_id": "bad", "payload": payload(), "received_at": "2022-01-01T00:00:00Z"}).status_code == 422
    assert client.get(f"/api/ingestion/connectors/{cid}/batches", params={"offset": -1}).status_code == 422
    assert client.get(f"/api/ingestion/connectors/{cid}/batches", params={"limit": 201}).status_code == 422


def test_restart_automatically_applies_an_enabled_durable_queue(tmp_path):
    database = "sqlite+pysqlite:///" + (tmp_path / "restart.db").as_posix()
    def settings(enabled):
        return Settings(_env_file=None, runtime_dir=tmp_path, database_url=database,
            deepseek_api_key=None, event_brief_allowed_source_ids=[], ingestion_worker_enabled=enabled)
    first = create_app(settings(False))
    with TestClient(first) as client:
        services = first.state.services
        services["auth"].create_user("manager", PASSWORD, ["safety_management"], ["31002"])
        services["data"].register_source("live-qa", "live", "31002")
        services["data"].register_point("t0", "live-qa", "T0", "T0", "methane", "%")
        login(client)
        cid = connector(client)["connector_id"]
        client.post(f"/api/ingestion/connectors/{cid}/actions", json={"action": "enable"})
        response = client.post(f"/api/ingestion/connectors/{cid}/batches", json={"batch_id": "persisted", "payload": payload()})
        assert response.json()["durable"] and row_count(services) == 0
    second = create_app(settings(True))
    with TestClient(second) as client:
        login(client)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = client.get(f"/api/ingestion/connectors/{cid}/batches/persisted")
            if response.status_code == 200 and response.json()["status"] == "completed":
                break
            time.sleep(.05)
        assert response.json()["status"] == "completed"
        assert row_count(second.state.services) == 1


def test_ingestion_endpoints_require_existing_login(tmp_path):
    app = create_app(Settings(_env_file=None, runtime_dir=tmp_path, ingestion_worker_enabled=False))
    with TestClient(app) as client:
        assert client.get("/api/ingestion/status").status_code == 401
        assert client.get("/api/ingestion/connectors").status_code == 401
        assert client.post("/api/ingestion/connectors", json={"name": "a", "source_id": "b"}).status_code == 401
