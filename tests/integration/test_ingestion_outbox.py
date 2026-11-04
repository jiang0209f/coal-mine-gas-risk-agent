"""Local durable sender survives disconnection, lost receipts and process restart."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading

import httpx
import pytest

from app.schemas.common import DomainError
from app.services.data.outbox import DurableOutbox

NOW = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
TOKEN = "synthetic-transport-token"
PAYLOAD = [{"reading_id": "one", "point_id": "p", "event_time": "2026-11-04T11:59:00Z", "raw_value": .7}]


def receipt(request, **changes):
    body = json.loads(request.content)
    result = dict(connector_id="c1", batch_id=body["batch_id"], durable=True, status="pending", imported_rows=0, duplicate_rows=0)
    return httpx.Response(202, json={**result, **changes})


@pytest.fixture
def outbox(tmp_path):
    clock = [NOW]
    box = DurableOutbox(tmp_path / "outbox.db", clock=lambda: clock[0])
    return box, clock, tmp_path / "outbox.db"


def test_enqueue_is_durable_private_and_same_batch_content_is_idempotent(outbox):
    box, clock, path = outbox
    first = box.enqueue("c1", "b1", PAYLOAD)
    assert first["status"] == "pending" and first["attempts"] == 0
    assert "payload" not in first
    assert box.enqueue("c1", "b1", PAYLOAD) == first
    restarted = DurableOutbox(path, clock=lambda: clock[0])
    assert restarted.get("c1", "b1") == first
    assert restarted.status()["counts"]["pending"] == 1
    with pytest.raises(DomainError) as error:
        restarted.enqueue("c1", "b1", [{**PAYLOAD[0], "raw_value": .8}])
    assert error.value.status_code == 409
    assert len(box.list_batches()) == 1


def test_network_failure_preserves_payload_then_acknowledged_pending_is_delivered(outbox):
    box, clock, path = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    attempts = []

    def disconnected(request):
        attempts.append(request)
        raise httpx.ConnectError("sensitive transport detail must stay out of records")

    box.flush_once("http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(disconnected))
    waiting = box.get("c1", "b1")
    assert waiting["status"] == "pending" and waiting["last_error_code"] == "network_error"
    assert datetime.fromisoformat(waiting["next_attempt_at"]) > NOW
    assert box.flush_once("http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(receipt))["attempted"] == 0
    clock[0] += timedelta(seconds=2)
    result = box.flush_once("http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(receipt))
    assert result["delivered"] == 1
    saved = box.get("c1", "b1")
    assert saved["status"] == "delivered" and saved["receipt"]["status"] == "pending"
    assert TOKEN.encode() not in path.read_bytes() and b"sensitive transport detail" not in path.read_bytes()


def test_lost_receipt_retries_same_identity_without_duplicate_receiver_effect(outbox):
    box, clock, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    received, requests = set(), []

    def receiver(request):
        body = json.loads(request.content)
        requests.append(body)
        duplicate = body["batch_id"] in received
        received.add(body["batch_id"])
        if len(requests) == 1:
            raise httpx.ReadError("receipt lost")
        return receipt(request, status="completed", imported_rows=0, duplicate_rows=int(duplicate))

    box.flush_once("http://localhost:8037", TOKEN, transport=httpx.MockTransport(receiver))
    clock[0] += timedelta(seconds=2)
    box.flush_once("http://localhost:8037", TOKEN, transport=httpx.MockTransport(receiver))
    assert len(received) == 1 and requests[0] == requests[1]
    assert box.get("c1", "b1")["status"] == "delivered"


@pytest.mark.parametrize("status,body,error_code", [
    (401, {}, "authentication_required"), (409, {}, "server_conflict"),
    (202, dict(connector_id="c1", batch_id="b1", durable=True, status="rejected"), "server_rejected"),
])
def test_auth_conflict_and_server_rejection_remain_blocked_until_explicit_retry(outbox, status, body, error_code):
    box, _, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    box.flush_once("http://127.0.0.1:8037", TOKEN,
                   transport=httpx.MockTransport(lambda request: httpx.Response(status, json=body)))
    saved = box.get("c1", "b1")
    assert saved["status"] == "blocked" and saved["last_error_code"] == error_code
    assert box.status()["counts"]["blocked"] == 1
    box.retry("c1", "b1")
    assert box.get("c1", "b1")["status"] == "pending"


@pytest.mark.parametrize("changes", [{"batch_id": "wrong"}, {"connector_id": "wrong"}, {"durable": False}, {"status": "unknown"}, {"status": []}])
def test_unmatched_or_unconfirmed_receipt_is_never_marked_delivered(outbox, changes):
    box, _, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    box.flush_once("http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(lambda request: receipt(request, **changes)))
    saved = box.get("c1", "b1")
    assert saved["status"] == "pending" and saved["last_error_code"] == "invalid_receipt"


def test_malformed_receipt_body_is_retained_for_retry(outbox):
    box, _, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    box.flush_once("http://127.0.0.1:8037", TOKEN,
                   transport=httpx.MockTransport(lambda request: httpx.Response(202, content=b"missing acknowledgement")))
    assert box.get("c1", "b1")["last_error_code"] == "invalid_receipt"


@pytest.mark.parametrize("target", ["https://example.com", "http://192.168.1.1", "http://user:password@localhost", "http://localhost?token=hidden"])
def test_nonlocal_or_credential_bearing_targets_are_rejected_before_transport(outbox, target):
    box, _, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    calls = []
    with pytest.raises(DomainError):
        box.flush_once(target, TOKEN, transport=httpx.MockTransport(lambda request: calls.append(request)))
    assert calls == [] and box.get("c1", "b1")["attempts"] == 0


def test_redirect_does_not_forward_credentials_or_data(outbox):
    box, _, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    calls = []

    def redirect(request):
        calls.append(str(request.url))
        return httpx.Response(307, headers={"Location": "https://example.com/leak"})

    box.flush_once("http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(redirect))
    assert len(calls) == 1 and calls[0].startswith("http://127.0.0.1:")
    assert box.get("c1", "b1")["last_error_code"] == "unexpected_redirect"


def test_invalid_header_credential_is_rejected_without_sending_or_exposing_value(outbox):
    box, _, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    credential = "synthetic-非法凭据"
    calls = []
    with pytest.raises(DomainError) as error:
        box.flush_once("http://127.0.0.1:8037", credential,
                       transport=httpx.MockTransport(lambda request: calls.append(request)))
    assert credential not in str(error.value)
    assert calls == [] and box.get("c1", "b1")["attempts"] == 0


def test_blocked_batch_does_not_prevent_other_batches_from_being_delivered(outbox):
    box, _, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    box.enqueue("c1", "b2", PAYLOAD)

    def receive(request):
        if json.loads(request.content)["batch_id"] == "b1":
            return httpx.Response(409)
        return receipt(request)

    result = box.flush_once("http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(receive))
    assert result["blocked"] == result["delivered"] == 1 and result["attempted"] == 2
    assert box.get("c1", "b1")["status"] == "blocked"
    assert box.get("c1", "b2")["status"] == "delivered"


def test_two_workers_do_not_send_a_batch_while_an_active_lease_is_held(outbox):
    box, clock, path = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    second = DurableOutbox(path, clock=lambda: clock[0])
    entered, release = threading.Event(), threading.Event()
    requests = []

    def receive(request):
        requests.append(request)
        entered.set()
        assert release.wait(5)
        return receipt(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(box.flush_once, "http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(receive))
        assert entered.wait(5)
        assert second.flush_once("http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(receipt))["attempted"] == 0
        release.set()
        assert first.result()["delivered"] == 1
    assert len(requests) == 1


def test_crashed_sender_is_recovered_from_its_durable_expired_lease(outbox):
    box, clock, path = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    program = """
import json,os,sys
from datetime import datetime
import httpx
sys.path.insert(0,sys.argv[1])
from app.services.data.outbox import DurableOutbox
args=json.load(sys.stdin)
box=DurableOutbox(args['database'],clock=lambda:datetime.fromisoformat(args['now']))
box.flush_once('http://127.0.0.1:8037',args['token'],transport=httpx.MockTransport(lambda request:os._exit(23)))
"""
    result = subprocess.run([sys.executable, "-c", program, str(Path(__file__).resolve().parents[2] / "backend")],
        input=json.dumps(dict(database=str(path), now=NOW.isoformat(), token=TOKEN)), text=True, capture_output=True)
    assert result.returncode == 23, result.stderr
    assert box.get("c1", "b1")["status"] == "sending"
    clock[0] += timedelta(seconds=31)
    restarted = DurableOutbox(path, clock=lambda: clock[0])
    assert restarted.flush_once("http://127.0.0.1:8037", TOKEN, transport=httpx.MockTransport(receipt))["delivered"] == 1


def test_actual_loopback_disconnection_and_reconnection_delivers_buffered_records(outbox):
    box, clock, _ = outbox
    box.enqueue("c1", "b1", PAYLOAD)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    target = f"http://127.0.0.1:{port}"
    assert box.flush_once(target, TOKEN)["delivered"] == 0
    requests = []

    class Receiver(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/api/ingestion/connectors/c1/batches"
            assert self.headers["Authorization"] == "Bearer " + TOKEN
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            response = json.dumps(dict(connector_id="c1", batch_id=body["batch_id"], durable=True, status="pending")).encode()
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    receiver = HTTPServer(("127.0.0.1", port), Receiver)
    worker = threading.Thread(target=receiver.serve_forever, daemon=True)
    worker.start()
    try:
        clock[0] += timedelta(seconds=2)
        assert box.flush_once(target, TOKEN)["delivered"] == 1
        assert requests == [{"batch_id": "b1", "payload": PAYLOAD}]
    finally:
        receiver.shutdown()
        receiver.server_close()
        worker.join(5)


def test_bridge_cli_enqueue_status_and_missing_environment_token_are_local(tmp_path, monkeypatch, capsys):
    from app.scripts.ingestion_bridge import main

    source = tmp_path / "payload.json"
    source.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    database = tmp_path / "bridge.db"
    common = ["--database", str(database)]
    assert main([*common, "enqueue", "--connector-id", "c1", "--batch-id", "b1", "--payload-file", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pending"
    assert main([*common, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["pending"] == 1
    monkeypatch.delenv("COAL_INGEST_TOKEN", raising=False)
    assert main([*common, "flush-once"]) == 2
    assert json.loads(capsys.readouterr().out)["error"] == "token_required"
    assert DurableOutbox(database).get("c1", "b1")["attempts"] == 0


def test_bridge_cli_flush_reads_token_only_from_environment(tmp_path, monkeypatch, capsys):
    from app.scripts.ingestion_bridge import main

    database = tmp_path / "bridge.db"
    DurableOutbox(database).enqueue("c1", "b1", PAYLOAD)
    monkeypatch.setenv("COAL_INGEST_TOKEN", TOKEN)
    calls = []

    def flush(self, target, token, max_batches):
        calls.append((target, token, max_batches))
        return dict(attempted=1, delivered=1, blocked=0, pending=0)

    monkeypatch.setattr(DurableOutbox, "flush_once", flush)
    assert main(["--database", str(database), "flush-once", "--base-url", "http://localhost:8037"]) == 0
    assert calls == [("http://localhost:8037", TOKEN, 10)]
    output = capsys.readouterr().out
    assert json.loads(output)["delivered"] == 1 and TOKEN not in output
