"""Real loopback HTTP: receiver outages, lost ACK and persisted receiver queue."""
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socket
import threading
import time

import httpx
from sqlalchemy import func, select
import uvicorn

from app.config import Settings
from app.db import observations
from app.main import build_services, create_app
from app.services.data.outbox import DurableOutbox


class Receiver:
    def __init__(self, settings):
        self.settings = settings
        with socket.socket() as reserve:
            reserve.bind(("127.0.0.1", 0))
            self.port = reserve.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = None

    def start(self):
        self.app = create_app(self.settings)
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port,
            log_level="critical", access_log=False))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 5
        while not self.server.started and time.monotonic() < deadline:
            time.sleep(.02)
        assert self.server.started

    def stop(self):
        if self.server is not None:
            self.server.should_exit = True
            self.thread.join(5)
            assert not self.thread.is_alive()
            self.server = None


def wait_completed(client, connector_id, batch_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/ingestion/connectors/{connector_id}/batches/{batch_id}")
        assert response.status_code == 200
        if response.json()["status"] == "completed":
            return response.json()
        time.sleep(.04)
    raise AssertionError("receiver did not apply the durable batch")


def test_outbox_and_actual_api_survive_outage_lost_ack_and_receiver_restart(tmp_path):
    settings = Settings(_env_file=None, runtime_dir=tmp_path,
        database_url="sqlite+pysqlite:///" + (tmp_path / "receiver.db").as_posix(),
        deepseek_api_key=None, model_catalog_path=None, event_brief_allowed_source_ids=[])
    services = build_services(settings)
    services["auth"].create_user("bridge-qa", "bridge-local-test-password", ["safety_management"], ["31002"])
    services["data"].register_source("bridge-qa", "bridge", "31002")
    services["data"].register_point("qa-t0", "bridge-qa", "T0", "T0", "methane", "%", unit_status="confirmed")
    services["store"].close()
    receiver = Receiver(settings)
    receiver.start()
    box = DurableOutbox(tmp_path / "sender.db")
    payload = {"readings": [{"reading_id": "measurement-1", "point_id": "qa-t0",
        "event_time": (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(), "raw_value": "0.70"}]}
    try:
        with httpx.Client(base_url=receiver.url, trust_env=False, timeout=3) as client:
            login = client.post("/api/auth/login", json={"username": "bridge-qa", "password": "bridge-local-test-password"})
            assert login.status_code == 200
            token = login.json()["access_token"]
            client.headers["Authorization"] = "Bearer " + token
            response = client.post("/api/ingestion/connectors", json={"name": "实际HTTP联调", "source_id": "bridge-qa"})
            assert response.status_code == 200
            cid = response.json()["connector_id"]
            receiver.stop()
            box.enqueue(cid, "offline", payload)
            assert box.flush_once(receiver.url, token)["pending"] == 1
            assert DurableOutbox(tmp_path / "sender.db").get(cid, "offline")["last_error_code"] == "network_error"
            receiver.start()
            box.retry(cid, "offline")
            assert box.flush_once(receiver.url, token)["delivered"] == 1
            assert client.get(f"/api/ingestion/connectors/{cid}/batches/offline").json()["status"] == "pending"
            # The server owns this accepted batch even if both programs restart.
            receiver.stop()
            receiver.start()
            assert client.post(f"/api/ingestion/connectors/{cid}/actions", json={"action": "enable"}).status_code == 200
            assert wait_completed(client, cid, "offline")["imported_rows"] == 1

            # Forward to the real API, then lose the HTTP response after commit.
            class LostAck(BaseHTTPRequestHandler):
                def do_POST(self):
                    body = self.rfile.read(int(self.headers["Content-Length"]))
                    with httpx.Client(trust_env=False) as forward:
                        response = forward.post(receiver.url + self.path, content=body,
                            headers={"Authorization": self.headers["Authorization"], "Content-Type": "application/json"})
                    self.server.accepted = response.status_code == 200 and response.json().get("durable") is True
                    self.close_connection = True

                def log_message(self, *args):
                    pass

            proxy = ThreadingHTTPServer(("127.0.0.1", 0), LostAck)
            worker = threading.Thread(target=proxy.serve_forever, daemon=True)
            worker.start()
            try:
                box.enqueue(cid, "lost-ack", payload)
                assert box.flush_once(f"http://127.0.0.1:{proxy.server_port}", token)["pending"] == 1
                assert proxy.accepted
            finally:
                proxy.shutdown()
                proxy.server_close()
                worker.join(2)
            box = DurableOutbox(tmp_path / "sender.db")
            box.retry(cid, "lost-ack")
            assert box.flush_once(receiver.url, token)["delivered"] == 1
            assert wait_completed(client, cid, "lost-ack")["duplicate_rows"] == 1
            state = client.get(f"/api/ingestion/connectors/{cid}").json()
            assert state["counts"]["ingested_rows"] == 1
            assert state["counts"]["duplicate_rows"] == 1
            assert state["counts"]["duplicate_batches"] == 1
            with receiver.app.state.services["store"].transaction() as connection:
                assert connection.execute(select(func.count()).select_from(observations)).scalar_one() == 1
                row = connection.execute(select(observations)).mappings().one()
            assert row["raw_value"] == "0.70"
            receipt = client.get(f"/api/ingestion/connectors/{cid}/batches/offline").json()
            assert row["available_at"] == receipt["applied_at"]
            assert receipt["accepted_at"] < receipt["applied_at"]
    finally:
        receiver.stop()
