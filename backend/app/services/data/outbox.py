"""Persist batches locally until the loopback receiver confirms durable custody."""
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

from app.schemas.common import DomainError, now_utc, utcstr

__all__ = ["DurableOutbox"]


def _json(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError):
        raise DomainError("待发批次必须为有效 JSON", code="invalid_payload") from None


def _identity(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or any(ord(c) < 32 or c in "/\\" for c in value):
        raise DomainError("接入器或批次编号无效")
    return value


def _target(base_url):
    try:
        url = urlsplit(base_url)
        port = url.port
        if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                or url.username is not None or url.password is not None or url.query or url.fragment
                or url.path not in ("", "/") or port == 0):
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise DomainError("采集缓存仅允许发送至本机 HTTP 服务根地址", code="nonlocal_target") from None
    return base_url.rstrip("/")


class DurableOutbox:
    """A 30-second durable lease prevents concurrent senders from claiming a batch."""

    def __init__(self, database, clock=None):
        if str(database).startswith(("\\\\", "//")) or str(database) == ":memory:":
            raise DomainError("采集缓存需要本机持久文件")
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock or now_utc
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS outbox (
                connector_id TEXT NOT NULL, batch_id TEXT NOT NULL,
                payload TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                last_error_code TEXT, delivered_at TEXT, lease_until TEXT, claim_token TEXT, receipt TEXT,
                PRIMARY KEY (connector_id,batch_id))""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _metadata(row):
        result = dict(row)
        for private in ("payload", "claim_token"):
            result.pop(private, None)
        result["receipt"] = json.loads(result["receipt"]) if result["receipt"] else None
        return result

    def enqueue(self, connector_id, batch_id, payload):
        _identity(connector_id)
        _identity(batch_id)
        serialized = _json(payload)
        if len(serialized.encode("utf-8")) > 8 * 1024 * 1024:
            raise DomainError("待发批次超过 8 MiB", code="payload_too_large", status_code=413)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        stamp = utcstr(self.clock())
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM outbox WHERE connector_id=? AND batch_id=?", (connector_id, batch_id)).fetchone()
            if existing:
                if existing["payload_sha256"] != digest:
                    raise DomainError("同一批次编号的内容发生变化", code="batch_conflict", status_code=409)
                return self._metadata(existing)
            db.execute("""INSERT INTO outbox
                (connector_id,batch_id,payload,payload_sha256,status,next_attempt_at,created_at,updated_at)
                VALUES (?,?,?,?,'pending',?,?,?)""", (connector_id, batch_id, serialized, digest, stamp, stamp, stamp))
            row = db.execute("SELECT * FROM outbox WHERE connector_id=? AND batch_id=?", (connector_id, batch_id)).fetchone()
            return self._metadata(row)

    def get(self, connector_id, batch_id):
        with self._connect() as db:
            row = db.execute("SELECT * FROM outbox WHERE connector_id=? AND batch_id=?", (connector_id, batch_id)).fetchone()
        if row is None:
            raise DomainError("待发批次不存在", code="not_found", status_code=404)
        return self._metadata(row)

    def list_batches(self):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM outbox ORDER BY rowid").fetchall()
        return [self._metadata(row) for row in rows]

    def status(self):
        counts = dict.fromkeys(("pending", "sending", "delivered", "blocked"), 0)
        with self._connect() as db:
            for row in db.execute("SELECT status,COUNT(*) AS count FROM outbox GROUP BY status"):
                counts[row["status"]] = row["count"]
        return {"counts": counts}

    def retry(self, connector_id, batch_id):
        stamp = utcstr(self.clock())
        with self._connect() as db:
            changed = db.execute("""UPDATE outbox SET status='pending',next_attempt_at=?,updated_at=?,
                last_error_code=NULL,lease_until=NULL,claim_token=NULL WHERE connector_id=? AND batch_id=?
                AND status IN ('pending','blocked')""", (stamp, stamp, connector_id, batch_id)).rowcount
            if not changed:
                raise DomainError("仅待发或阻塞批次可重试", code="batch_not_retryable", status_code=409)
        return self.get(connector_id, batch_id)

    def _claim(self):
        now = self.clock()
        stamp = utcstr(now)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT * FROM outbox WHERE
                (status='pending' AND next_attempt_at<=?) OR (status='sending' AND lease_until<=?)
                ORDER BY rowid LIMIT 1""", (stamp, stamp)).fetchone()
            if row is None:
                return None
            claim = dict(row)
            claim["claim_token"] = uuid4().hex
            claim["attempts"] += 1
            db.execute("""UPDATE outbox SET status='sending',attempts=?,updated_at=?,lease_until=?,claim_token=?
                WHERE connector_id=? AND batch_id=?""", (claim["attempts"], stamp, utcstr(now + timedelta(seconds=30)),
                claim["claim_token"], claim["connector_id"], claim["batch_id"]))
        return claim

    def _finish(self, claim, status, error_code=None, receipt=None):
        now = self.clock()
        stamp = utcstr(now)
        next_attempt = utcstr(now + timedelta(seconds=min(60, 2 ** min(claim["attempts"] - 1, 6))))
        with self._connect() as db:
            changed = db.execute("""UPDATE outbox SET status=?,last_error_code=?,receipt=?,next_attempt_at=?,
                updated_at=?,delivered_at=?,lease_until=NULL,claim_token=NULL
                WHERE connector_id=? AND batch_id=? AND status='sending' AND claim_token=?""",
                (status, error_code, _json(receipt) if receipt else None, next_attempt, stamp,
                 stamp if status == "delivered" else None, claim["connector_id"], claim["batch_id"], claim["claim_token"])).rowcount
        return bool(changed)

    @staticmethod
    def _receipt(response, claim):
        try:
            result = response.json()
        except (ValueError, UnicodeError):
            return "pending", "invalid_receipt", None
        if (not isinstance(result, dict) or result.get("connector_id") != claim["connector_id"]
                or result.get("batch_id") != claim["batch_id"] or result.get("durable") is not True
                or result.get("status") not in ("pending", "completed", "retry_wait", "rejected")):
            return "pending", "invalid_receipt", None
        # Persist only the protocol receipt; a response may carry unrelated text or credentials.
        saved = {key: result[key] for key in ("connector_id", "batch_id", "durable", "status")}
        for key in ("imported_rows", "duplicate_rows"):
            if type(result.get(key)) is int and result[key] >= 0:
                saved[key] = result[key]
        if saved["status"] == "rejected":
            return "blocked", "server_rejected", saved
        return "delivered", None, saved

    def flush_once(self, base_url, token, max_batches=10, transport=None):
        base_url = _target(base_url)
        if (not isinstance(token, str) or not token.strip() or not token.isascii()
                or any(ord(c) < 32 or ord(c) == 127 for c in token)):
            raise DomainError("请通过环境变量配置接入令牌", code="token_required")
        if type(max_batches) is not int or not 1 <= max_batches <= 1000:
            raise DomainError("单次发送批次数量应为 1 至 1000")
        result = dict(attempted=0, delivered=0, pending=0, blocked=0)
        with httpx.Client(timeout=10, trust_env=False, follow_redirects=False, transport=transport) as client:
            for _ in range(max_batches):
                claim = self._claim()
                if claim is None:
                    break
                result["attempted"] += 1
                try:
                    response = client.post(
                        f"{base_url}/api/ingestion/connectors/{quote(claim['connector_id'], safe='')}/batches",
                        headers={"Authorization": "Bearer " + token},
                        json={"batch_id": claim["batch_id"], "payload": json.loads(claim["payload"])})
                    if 200 <= response.status_code < 300:
                        status, error, receipt = self._receipt(response, claim)
                    else:
                        status, receipt = "blocked", None
                        error = {401: "authentication_required", 409: "server_conflict"}.get(response.status_code, "request_rejected")
                        if 300 <= response.status_code < 400:
                            error = "unexpected_redirect"
                        elif response.status_code >= 500 or response.status_code in (408, 429):
                            status, error = "pending", "http_error"
                except httpx.TransportError:
                    status, error, receipt = "pending", "network_error", None
                if self._finish(claim, status, error, receipt):
                    result[status] += 1
        return result
