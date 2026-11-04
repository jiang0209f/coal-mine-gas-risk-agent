"""Local durable ingestion receipts and source-bound retry processing."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
from uuid import uuid4

from app.schemas.common import Actor, DomainError, now_utc, utcstr
from app.services.auth.service import AuthService
from app.services.data.ingestion_adapters import normalize, validate_config

__all__ = ["IngestionService"]

_CONNECTORS = "ingestion_connector"
_BATCHES = "ingestion_batch"
_WAITING = {"pending", "retry_wait"}


class IngestionService:
    def __init__(self, store, data, clock=now_utc):
        self.store, self.data, self.clock = store, data, clock
        self.auth = AuthService(store)

    @contextmanager
    def _transaction(self):
        with self.store.transaction() as connection:
            # Claim the SQLite write lock before reading a receipt; separate app
            # instances cannot both publish the same pending batch.
            if connection.dialect.name == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            yield connection

    @staticmethod
    def _identifier(value, label):
        if (not isinstance(value, str) or not value.strip() or len(value) > 200
                or any(character in value for character in ("/", "\\", "\x00", "\r", "\n"))):
            raise DomainError(f"{label}无效")
        return value

    @staticmethod
    def _digest(value):
        try:
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, RecursionError):
            raise DomainError("接入批次须为有限数值JSON") from None
        if len(encoded) > 8 * 1024 * 1024:
            raise DomainError("单次接入批次上限8MiB", code="batch_too_large", status_code=413)
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _batch_key(connector_id, batch_id):
        return hashlib.sha256(json.dumps([connector_id, batch_id], separators=(",", ":")).encode()).hexdigest()

    def _get(self, actor, connector_id, manage=False, connection=None):
        connector_id = self._identifier(connector_id, "接入编号")
        entry = self.store.get(_CONNECTORS, connector_id, connection)
        if entry is None:
            raise DomainError("接入配置不存在", code="not_found", status_code=404)
        self.auth.authorize(actor, "manage_data" if manage else "read", entry["face_id"])
        source = self.data._source(entry["source_id"], connection)
        if source["face_id"] != entry["face_id"]:
            raise DomainError("接入来源范围已变化", code="forbidden", status_code=403)
        return entry

    @staticmethod
    def _receipt(batch):
        return {key: deepcopy(value) for key, value in batch.items() if key not in ("readings", "storage_key")}

    def _state(self, entry, connection=None):
        batches = [row for row in self.store.list(_BATCHES, connection) if row["connector_id"] == entry["connector_id"]]
        counts = {"accepted_batches": len(batches), "duplicate_batches": entry.get("duplicate_batches", 0),
                  "completed_batches": sum(row["status"] == "completed" for row in batches),
                  "pending_batches": sum(row["status"] == "pending" for row in batches),
                  "retry_batches": sum(row["status"] == "retry_wait" for row in batches),
                  "rejected_batches": sum(row["status"] == "rejected" for row in batches),
                  "queued_rows": sum(row["row_count"] for row in batches if row["status"] in _WAITING),
                  "ingested_rows": sum(row["imported_rows"] for row in batches),
                  "duplicate_rows": sum(row["duplicate_rows"] for row in batches),
                  "conflict_batches": sum(row.get("error_code") == "reading_conflict" for row in batches)}
        status = "disabled"
        if entry["enabled"]:
            if counts["retry_batches"]:
                status = "retry_wait"
            elif entry.get("last_error_code"):
                status = "error"
            elif entry.get("last_received_at") is None:
                status = "awaiting_data"
            elif (self.clock() - datetime.fromisoformat(entry["last_received_at"])).total_seconds() > entry["stale_after_seconds"]:
                status = "stale"
            else:
                status = "receiving"
        public = {key: deepcopy(value) for key, value in entry.items() if key not in ("batch_sequence", "duplicate_batches")}
        return {**public, "status": status, "counts": counts}

    def create_connector(self, actor, spec):
        self.auth.authorize(actor, "manage_data")
        fields = {"connector_id", "name", "source_id", "adapter_kind", "mapping", "tag_map", "enabled", "stale_after_seconds"}
        if not isinstance(spec, dict) or set(spec) - fields:
            raise DomainError("接入配置字段无效")
        if spec.get("enabled", False) is not False:
            raise DomainError("新接入须先登记，再显式启用")
        name = spec.get("name")
        if not isinstance(name, str) or not name.strip() or len(name) > 200:
            raise DomainError("接入名称需要1至200个字符")
        source_id = self._identifier(spec.get("source_id"), "来源编号")
        connector_id = self._identifier(spec.get("connector_id") or str(uuid4()), "接入编号")
        stale_after = spec.get("stale_after_seconds", 120)
        if type(stale_after) is not int or not 1 <= stale_after <= 86400:
            raise DomainError("数据中断判定须为1至86400秒")
        with self._transaction() as connection:
            source = self.data._source(source_id, connection)
            actor.require_face(source["face_id"])
            if self.store.get(_CONNECTORS, connector_id, connection) is not None:
                raise DomainError("接入编号已登记，配置不可覆盖", code="conflict", status_code=409)
            config = dict(source_id=source_id, adapter_kind=spec.get("adapter_kind", "standard_json"),
                          mapping=deepcopy(spec.get("mapping", {})), tag_map=deepcopy(spec.get("tag_map", {})),
                          timezone=source.get("timezone", "+08:00"))
            validate_config(config, self.store.list("point", connection))
            moment = utcstr(self.clock())
            entry = {**config, "connector_id": connector_id, "name": name.strip(), "face_id": source["face_id"],
                "enabled": False, "stale_after_seconds": stale_after, "config_version": "ingestion-v1:" + self._digest(config),
                "created_by": actor.user_id, "created_at": moment, "updated_at": moment,
                "batch_sequence": 0, "duplicate_batches": 0, "last_received_at": None,
                "last_success_at": None, "last_heartbeat_at": None, "latest_event_time": None, "last_error_code": None}
            self.store.put(_CONNECTORS, connector_id, entry, connection)
            return self._state(entry, connection)

    def list_connectors(self, actor, source_id=None):
        self.auth.authorize(actor, "read")
        if source_id is not None:
            actor.require_face(self.data._source(source_id)["face_id"])
        return [self._state(entry) for entry in self.store.list(_CONNECTORS)
                if entry["face_id"] in actor.face_ids and (source_id is None or entry["source_id"] == source_id)]

    def get_connector(self, actor, connector_id):
        return self._state(self._get(actor, connector_id))

    def action(self, actor, connector_id, action):
        if action not in ("enable", "pause", "retry"):
            raise DomainError("接入动作须为enable、pause或retry")
        with self._transaction() as connection:
            entry = self._get(actor, connector_id, manage=True, connection=connection)
            if action == "retry":
                for batch in self.store.list(_BATCHES, connection):
                    if batch["connector_id"] == connector_id and batch["status"] in ("retry_wait", "rejected"):
                        batch.update(status="pending", next_retry_at=None, error_code=None)
                        self.store.put(_BATCHES, batch["storage_key"], batch, connection)
                entry["last_error_code"] = None
            else:
                entry["enabled"] = action == "enable"
            entry["updated_at"] = utcstr(self.clock())
            self.store.put(_CONNECTORS, connector_id, entry, connection)
            return self._state(entry, connection)

    def heartbeat(self, actor, connector_id):
        with self._transaction() as connection:
            entry = self._get(actor, connector_id, manage=True, connection=connection)
            entry["last_heartbeat_at"] = entry["updated_at"] = utcstr(self.clock())
            self.store.put(_CONNECTORS, connector_id, entry, connection)
            return self._state(entry, connection)

    def submit_batch(self, actor, connector_id, batch_id, payload):
        entry = self._get(actor, connector_id, manage=True)
        batch_id = self._identifier(batch_id, "批次编号")
        payload_sha = self._digest(payload)
        now = self.clock()
        readings = normalize(entry, payload, self.store.list("point"), received_at=now)
        key = self._batch_key(connector_id, batch_id)
        with self._transaction() as connection:
            entry = self._get(actor, connector_id, manage=True, connection=connection)
            existing = self.store.get(_BATCHES, key, connection)
            moment = utcstr(now)
            entry["last_received_at"] = entry["updated_at"] = moment
            if existing is not None:
                if existing["payload_sha256"] != payload_sha:
                    raise DomainError("同一批次编号的内容冲突", code="batch_conflict", status_code=409)
                entry["duplicate_batches"] += 1
                self.store.put(_CONNECTORS, connector_id, entry, connection)
                return self._receipt(existing)
            entry["batch_sequence"] += 1
            latest = max(row["event_time"] for row in readings)
            entry["latest_event_time"] = max(entry.get("latest_event_time") or latest, latest)
            batch = dict(connector_id=connector_id, source_id=entry["source_id"], batch_id=batch_id, storage_key=key,
                config_version=entry["config_version"], sequence=entry["batch_sequence"], payload_sha256=payload_sha,
                readings=deepcopy(readings), row_count=len(readings), accepted_at=moment, applied_at=None,
                attempt_count=0, next_retry_at=None, error_code=None, status="pending", durable=True,
                imported_rows=0, duplicate_rows=0)
            self.store.put(_BATCHES, key, batch, connection)
            self.store.put(_CONNECTORS, connector_id, entry, connection)
            return self._receipt(batch)

    def get_batch(self, actor, connector_id, batch_id):
        self._get(actor, connector_id)
        key = self._batch_key(connector_id, self._identifier(batch_id, "批次编号"))
        batch = self.store.get(_BATCHES, key)
        if batch is None:
            raise DomainError("接入批次不存在", code="not_found", status_code=404)
        return self._receipt(batch)

    def list_batches(self, actor, connector_id, offset=0, limit=50):
        self._get(actor, connector_id)
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise DomainError("批次分页参数无效")
        rows = sorted((row for row in self.store.list(_BATCHES) if row["connector_id"] == connector_id),
                      key=lambda row: row["sequence"])
        return dict(connector_id=connector_id, total=len(rows), offset=offset, limit=limit,
                    items=[self._receipt(row) for row in rows[offset:offset+limit]])

    def _next(self):
        with self.store.transaction() as connection:
            enabled = {entry["connector_id"] for entry in self.store.list(_CONNECTORS, connection) if entry["enabled"]}
            heads = {}
            for batch in self.store.list(_BATCHES, connection):
                if batch["connector_id"] not in enabled or batch["status"] not in _WAITING:
                    continue
                previous = heads.get(batch["connector_id"])
                if previous is None or batch["sequence"] < previous["sequence"]:
                    heads[batch["connector_id"]] = batch
            eligible = [batch for batch in heads.values() if batch["next_retry_at"] is None
                        or datetime.fromisoformat(batch["next_retry_at"]) <= self.clock()]
            return min(eligible, key=lambda row: (row["accepted_at"], row["connector_id"], row["sequence"]))["storage_key"] if eligible else None

    def _process(self, key):
        with self._transaction() as connection:
            batch = self.store.get(_BATCHES, key, connection)
            if batch is None or batch["status"] not in _WAITING:
                return None
            entry = self.store.get(_CONNECTORS, batch["connector_id"], connection)
            if not entry["enabled"] or (batch["next_retry_at"] is not None
                    and datetime.fromisoformat(batch["next_retry_at"]) > self.clock()):
                return None
            moment = self.clock()
            attempt = batch["attempt_count"] + 1
            actor = Actor(user_id="ingestion:"+entry["connector_id"], roles=("safety_management",),
                          face_ids=(entry["face_id"],), is_system=True)
            try:
                with connection.begin_nested():
                    result = self.data.ingest_readings(actor, entry["source_id"], batch["readings"],
                                                       received_at=moment, connection=connection)
                    batch.update(status="completed", applied_at=utcstr(moment), attempt_count=attempt,
                        next_retry_at=None, error_code=None, imported_rows=result["imported_rows"], duplicate_rows=result["duplicate_rows"])
                    entry.update(last_success_at=utcstr(moment), last_error_code=None, updated_at=utcstr(moment))
                    self.store.put(_BATCHES, key, batch, connection)
                    self.store.put(_CONNECTORS, entry["connector_id"], entry, connection)
            except Exception as error:
                batch = self.store.get(_BATCHES, key, connection)
                entry = self.store.get(_CONNECTORS, batch["connector_id"], connection)
                rejected = isinstance(error, DomainError) and error.status_code in (400, 403, 404, 409, 413, 422)
                code = "reading_conflict" if isinstance(error, DomainError) and error.status_code == 409 else "invalid_readings" if rejected else "ingest_unavailable"
                delay = min(300, 2 ** min(attempt-1, 9))
                batch.update(status="rejected" if rejected else "retry_wait", attempt_count=attempt, error_code=code,
                             next_retry_at=None if rejected else utcstr(moment+timedelta(seconds=delay)))
                entry.update(last_error_code=code, updated_at=utcstr(moment))
                self.store.put(_BATCHES, key, batch, connection)
                self.store.put(_CONNECTORS, entry["connector_id"], entry, connection)
            return self._receipt(batch)

    def process_pending(self, max_batches=10):
        if type(max_batches) is not int or not 1 <= max_batches <= 100:
            raise DomainError("单轮处理批次须为1至100")
        results = []
        for _ in range(max_batches):
            key = self._next()
            if key is None:
                break
            result = self._process(key)
            if result is not None:
                results.append(result)
        return {"processed_batches": len(results), "completed_batches": sum(row["status"] == "completed" for row in results),
                "retry_batches": sum(row["status"] == "retry_wait" for row in results),
                "rejected_batches": sum(row["status"] == "rejected" for row in results)}
