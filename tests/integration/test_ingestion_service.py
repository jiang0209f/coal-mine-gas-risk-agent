"""Durable source-bound ingestion receipts, retries and observation deduplication."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.db import Store, observations
from app.schemas.common import Actor, DomainError
from app.services.data.service import DataService


@pytest.fixture
def parts(tmp_path):
    from app.services.data.ingestion import IngestionService

    store = Store("sqlite+pysqlite:///:memory:")
    data = DataService(store, tmp_path)
    data.register_source("source", "Synthetic local source", "31002")
    data.register_source("other", "Foreign source", "32001")
    data.register_point("t0", "source", "TAG-T0", "T0", "methane", "%", unit_status="confirmed")
    data.register_point("foreign", "other", "TAG-X", "Foreign", "methane", "%")
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    clock = [datetime(2026, 11, 4, 12, tzinfo=timezone.utc)]
    service = IngestionService(store, data, clock=lambda: clock[0])
    yield SimpleNamespace(store=store, data=data, actor=actor, clock=clock, service=service)
    store.close()


def connector(parts, **changes):
    return parts.service.create_connector(parts.actor, {
        "connector_id": "connector", "name": "Synthetic gateway", "source_id": "source",
        "adapter_kind": "standard_json", "enabled": False, **changes})


def payload(parts, reading_id="r1", **changes):
    return [{"reading_id": reading_id, "point_id": "t0", "event_time": (parts.clock[0]-timedelta(minutes=1)).isoformat(),
             "raw_value": "0.70", **changes}]


def count(parts):
    with parts.store.transaction() as connection:
        return connection.execute(select(func.count()).select_from(observations)).scalar_one()


def test_connector_registration_defaults_disabled_and_not_connected(parts):
    saved = connector(parts)
    assert saved["enabled"] is False and saved["status"] == "disabled"
    assert saved["last_received_at"] is None and saved["last_success_at"] is None
    assert saved["counts"]["accepted_batches"] == 0
    assert parts.service.list_connectors(parts.actor, "source") == [saved]
    generated = connector(parts, connector_id=None, name="Another")
    assert generated["connector_id"] and generated["connector_id"] != "connector"


@pytest.mark.parametrize("actor", [
    Actor(user_id="reader", roles=("ventilation",), face_ids=("31002",)),
    Actor(user_id="other", roles=("safety_management",), face_ids=("32001",)),
    Actor(user_id="system", roles=("safety_management",), face_ids=("31002",), is_system=True),
])
def test_registration_requires_human_management_in_source_scope(parts, actor):
    with pytest.raises(DomainError) as caught:
        parts.service.create_connector(actor, {"name": "Source", "source_id": "source"})
    assert caught.value.status_code == 403


def test_config_identity_and_source_mapping_cannot_be_replaced(parts):
    saved = connector(parts)
    with pytest.raises(DomainError) as caught:
        connector(parts, name="Changed")
    assert caught.value.status_code == 409
    assert parts.service.get_connector(parts.actor, "connector")["name"] == saved["name"]
    with pytest.raises(DomainError):
        connector(parts, connector_id="foreign-map", adapter_kind="mapped_json", tag_map={"X": "foreign"})
    with pytest.raises(DomainError):
        connector(parts, connector_id="enabled", enabled=True)


def test_paused_acceptance_is_durable_and_recreation_replays_at_actual_apply_time(parts):
    from app.services.data.ingestion import IngestionService

    connector(parts)
    accepted = parts.service.submit_batch(parts.actor, "connector", "batch", payload(parts))
    assert accepted["durable"] is True and accepted["status"] == "pending"
    assert count(parts) == 0
    parts.service = IngestionService(parts.store, parts.data, clock=lambda: parts.clock[0])
    parts.service.process_pending()
    assert count(parts) == 0
    parts.clock[0] += timedelta(minutes=10)
    parts.service.action(parts.actor, "connector", "enable")
    parts.service.process_pending()
    receipt = parts.service.get_batch(parts.actor, "connector", "batch")
    assert receipt["status"] == "completed" and receipt["imported_rows"] == 1
    with parts.store.transaction() as connection:
        row = connection.execute(select(observations)).mappings().one()
    assert datetime.fromisoformat(row["available_at"]) == parts.clock[0]
    assert datetime.fromisoformat(row["event_time"]) < parts.clock[0] - timedelta(minutes=9)


def test_batch_replay_is_idempotent_and_different_content_conflicts(parts):
    connector(parts)
    original = payload(parts)
    accepted = parts.service.submit_batch(parts.actor, "connector", "batch", original)
    repeated = parts.service.submit_batch(parts.actor, "connector", "batch", deepcopy(original))
    assert repeated["payload_sha256"] == accepted["payload_sha256"]
    assert repeated["accepted_at"] == accepted["accepted_at"]
    assert parts.service.get_connector(parts.actor, "connector")["counts"]["duplicate_batches"] == 1
    with pytest.raises(DomainError) as caught:
        parts.service.submit_batch(parts.actor, "connector", "batch", payload(parts, raw_value="0.80"))
    assert caught.value.status_code == 409
    assert parts.service.list_batches(parts.actor, "connector")["total"] == 1


def test_repeated_reading_in_another_batch_keeps_first_availability(parts):
    connector(parts)
    parts.service.action(parts.actor, "connector", "enable")
    readings = payload(parts)
    parts.service.submit_batch(parts.actor, "connector", "first", readings)
    parts.service.process_pending()
    first_time = parts.clock[0]
    parts.clock[0] += timedelta(minutes=5)
    parts.service.submit_batch(parts.actor, "connector", "repeat", readings)
    parts.service.process_pending()
    assert count(parts) == 1
    receipt = parts.service.get_batch(parts.actor, "connector", "repeat")
    assert receipt["imported_rows"] == 0 and receipt["duplicate_rows"] == 1
    with parts.store.transaction() as connection:
        row = connection.execute(select(observations)).mappings().one()
    assert datetime.fromisoformat(row["available_at"]) == first_time


def test_reading_conflict_rejects_entire_batch_without_partial_insert(parts):
    connector(parts)
    parts.service.action(parts.actor, "connector", "enable")
    parts.service.submit_batch(parts.actor, "connector", "first", payload(parts))
    parts.service.process_pending()
    conflicting = payload(parts, "r2") + payload(parts, raw_value="0.90")
    parts.service.submit_batch(parts.actor, "connector", "conflict", conflicting)
    parts.service.process_pending()
    receipt = parts.service.get_batch(parts.actor, "connector", "conflict")
    assert receipt["status"] == "rejected" and receipt["error_code"] == "reading_conflict"
    assert count(parts) == 1
    assert parts.service.get_connector(parts.actor, "connector")["counts"]["conflict_batches"] == 1


def test_transient_failure_waits_then_retries_with_safe_error(parts, monkeypatch):
    connector(parts)
    parts.service.action(parts.actor, "connector", "enable")
    parts.service.submit_batch(parts.actor, "connector", "batch", payload(parts))
    original = parts.data.ingest_readings
    calls = []

    def flaky(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise OSError("sk-private-secret-0123456789 F:/private/password")
        return original(*args, **kwargs)

    monkeypatch.setattr(parts.data, "ingest_readings", flaky)
    parts.service.process_pending()
    first = parts.service.get_batch(parts.actor, "connector", "batch")
    assert first["status"] == "retry_wait" and first["attempt_count"] == 1
    assert datetime.fromisoformat(first["next_retry_at"]) > parts.clock[0]
    assert "sk-private" not in json.dumps(first) and "F:/private" not in json.dumps(first)
    parts.service.process_pending()
    assert len(calls) == 1
    parts.clock[0] = datetime.fromisoformat(first["next_retry_at"])
    parts.service.process_pending()
    assert parts.service.get_batch(parts.actor, "connector", "batch")["status"] == "completed"
    assert count(parts) == 1 and len(calls) == 2


def test_failed_ack_rolls_back_observations_before_retry(parts, monkeypatch):
    connector(parts)
    parts.service.action(parts.actor, "connector", "enable")
    parts.service.submit_batch(parts.actor, "connector", "batch", payload(parts))
    original = parts.data.ingest_readings

    def fail_after_insert(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("crash before receipt")

    monkeypatch.setattr(parts.data, "ingest_readings", fail_after_insert)
    parts.service.process_pending()
    assert count(parts) == 0
    assert parts.service.get_batch(parts.actor, "connector", "batch")["status"] == "retry_wait"
    monkeypatch.setattr(parts.data, "ingest_readings", original)
    parts.service.action(parts.actor, "connector", "retry")
    parts.service.process_pending()
    receipt = parts.service.get_batch(parts.actor, "connector", "batch")
    assert receipt["status"] == "completed" and receipt["imported_rows"] == 1 and receipt["duplicate_rows"] == 0


def test_parallel_replay_and_processing_apply_once(parts):
    connector(parts)
    readings = payload(parts)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(lambda _: parts.service.submit_batch(parts.actor, "connector", "batch", readings), range(4)))
    assert len({row["payload_sha256"] for row in receipts}) == 1
    parts.service.action(parts.actor, "connector", "enable")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: parts.service.process_pending(), range(4)))
    assert count(parts) == 1
    state = parts.service.get_connector(parts.actor, "connector")
    assert state["counts"]["completed_batches"] == 1 and state["counts"]["ingested_rows"] == 1


def test_status_uses_real_receipt_and_stale_clock_not_registration(parts):
    connector(parts, stale_after_seconds=30)
    parts.service.action(parts.actor, "connector", "enable")
    assert parts.service.get_connector(parts.actor, "connector")["status"] == "awaiting_data"
    parts.service.heartbeat(parts.actor, "connector")
    assert parts.service.get_connector(parts.actor, "connector")["status"] == "awaiting_data"
    parts.service.submit_batch(parts.actor, "connector", "batch", payload(parts))
    parts.service.process_pending()
    assert parts.service.get_connector(parts.actor, "connector")["status"] == "receiving"
    parts.clock[0] += timedelta(seconds=31)
    assert parts.service.get_connector(parts.actor, "connector")["status"] == "stale"
    parts.service.action(parts.actor, "connector", "pause")
    assert parts.service.get_connector(parts.actor, "connector")["status"] == "disabled"


def test_queue_payload_stays_private_and_pagination_is_stable(parts):
    connector(parts)
    parts.service.submit_batch(parts.actor, "connector", "a", payload(parts, raw_value="PRIVATE_PAYLOAD"))
    parts.clock[0] += timedelta(seconds=1)
    parts.service.submit_batch(parts.actor, "connector", "b", payload(parts, "r2"))
    public = parts.service.list_batches(parts.actor, "connector", offset=1, limit=1)
    assert public["total"] == 2 and public["items"][0]["batch_id"] == "b"
    assert "PRIVATE_PAYLOAD" not in json.dumps(public)
    assert "readings" not in parts.service.get_batch(parts.actor, "connector", "a")


def test_mutations_forbid_readers_and_foreign_reads(parts):
    connector(parts)
    reader = Actor(user_id="reader", roles=("ventilation",), face_ids=("31002",))
    assert parts.service.get_connector(reader, "connector")["connector_id"] == "connector"
    for invoke in (lambda: parts.service.action(reader, "connector", "enable"),
                   lambda: parts.service.submit_batch(reader, "connector", "batch", payload(parts)),
                   lambda: parts.service.heartbeat(reader, "connector")):
        with pytest.raises(DomainError) as caught:
            invoke()
        assert caught.value.status_code == 403
    foreign = Actor(user_id="foreign", roles=("safety_management",), face_ids=("32001",))
    with pytest.raises(DomainError) as caught:
        parts.service.get_connector(foreign, "connector")
    assert caught.value.status_code == 403


def test_disk_restart_preserves_pending_and_retry_receipts(tmp_path, monkeypatch):
    from app.services.data.ingestion import IngestionService

    database_url = "sqlite+pysqlite:///" + (tmp_path / "restart.sqlite3").as_posix()
    moment = [datetime(2026, 11, 4, 12, tzinfo=timezone.utc)]
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    first = Store(database_url)
    try:
        data = DataService(first, tmp_path / "uploads")
        data.register_source("source", "Synthetic", "31002")
        data.register_point("t0", "source", "TAG-T0", "T0", "methane", "%", unit_status="confirmed")
        service = IngestionService(first, data, clock=lambda: moment[0])
        service.create_connector(actor, {"connector_id": "connector", "name": "Synthetic", "source_id": "source"})
        reading = [{"reading_id": "r1", "point_id": "t0", "event_time": (moment[0]-timedelta(minutes=1)).isoformat(), "raw_value": "0.7"}]
        service.submit_batch(actor, "connector", "batch", reading)
    finally:
        first.close()

    second = Store(database_url)
    try:
        data = DataService(second, tmp_path / "uploads")
        service = IngestionService(second, data, clock=lambda: moment[0])
        assert service.get_batch(actor, "connector", "batch")["status"] == "pending"
        service.action(actor, "connector", "enable")

        def offline(*args, **kwargs):
            raise OSError("temporary local storage unavailable")

        monkeypatch.setattr(data, "ingest_readings", offline)
        service.process_pending()
        retry = service.get_batch(actor, "connector", "batch")
        assert retry["status"] == "retry_wait" and retry["attempt_count"] == 1
    finally:
        second.close()

    third = Store(database_url)
    try:
        service = IngestionService(third, DataService(third, tmp_path / "uploads"), clock=lambda: moment[0])
        assert service.get_batch(actor, "connector", "batch") == retry
        assert service.process_pending()["processed_batches"] == 0
        moment[0] = datetime.fromisoformat(retry["next_retry_at"])
        assert service.process_pending()["completed_batches"] == 1
        with third.transaction() as connection:
            rows = list(connection.execute(select(observations)).mappings())
        assert len(rows) == 1 and datetime.fromisoformat(rows[0]["available_at"]) == moment[0]
        assert service.get_batch(actor, "connector", "batch")["attempt_count"] == 2
    finally:
        third.close()


def test_two_sqlite_stores_do_not_ack_or_apply_one_batch_twice(tmp_path):
    from app.services.data.ingestion import IngestionService

    database_url = "sqlite+pysqlite:///" + (tmp_path / "shared.sqlite3").as_posix()
    stores = [Store(database_url), Store(database_url)]
    moment = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    try:
        data = [DataService(store, tmp_path / "uploads") for store in stores]
        data[0].register_source("source", "Synthetic", "31002")
        data[0].register_point("t0", "source", "TAG-T0", "T0", "methane", "%", unit_status="confirmed")
        services = [IngestionService(store, item, clock=lambda: moment) for store, item in zip(stores, data)]
        services[0].create_connector(actor, {"connector_id": "connector", "name": "Synthetic", "source_id": "source"})
        reading = [{"reading_id": "r1", "point_id": "t0", "event_time": (moment-timedelta(minutes=1)).isoformat(), "raw_value": "0.7"}]
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(pool.map(lambda index: services[index % 2].submit_batch(actor, "connector", "batch", reading), range(4)))
        assert len({row["accepted_at"] for row in receipts}) == 1
        services[0].action(actor, "connector", "enable")
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda index: services[index % 2].process_pending(), range(4)))
        assert sum(result["completed_batches"] for result in results) == 1
        state = services[0].get_connector(actor, "connector")
        assert state["counts"]["accepted_batches"] == 1 and state["counts"]["duplicate_batches"] == 3
        assert state["counts"]["ingested_rows"] == 1 and state["counts"]["duplicate_rows"] == 0
        with stores[0].transaction() as connection:
            assert connection.execute(select(func.count()).select_from(observations)).scalar_one() == 1
    finally:
        for store in stores:
            store.close()


def test_retrying_head_blocks_later_batch_until_success(parts, monkeypatch):
    connector(parts)
    parts.service.action(parts.actor, "connector", "enable")
    parts.service.submit_batch(parts.actor, "connector", "head", payload(parts))
    parts.service.submit_batch(parts.actor, "connector", "tail", payload(parts, "r2"))
    original = parts.data.ingest_readings

    def offline(*args, **kwargs):
        raise OSError("temporary local storage unavailable")

    monkeypatch.setattr(parts.data, "ingest_readings", offline)
    assert parts.service.process_pending()["processed_batches"] == 1
    retry = parts.service.get_batch(parts.actor, "connector", "head")
    assert retry["status"] == "retry_wait"
    assert parts.service.get_batch(parts.actor, "connector", "tail")["attempt_count"] == 0
    monkeypatch.setattr(parts.data, "ingest_readings", original)
    parts.clock[0] = datetime.fromisoformat(retry["next_retry_at"])
    assert parts.service.process_pending(max_batches=1)["completed_batches"] == 1
    assert parts.service.get_batch(parts.actor, "connector", "tail")["status"] == "pending"
    assert parts.service.process_pending(max_batches=1)["completed_batches"] == 1
    assert count(parts) == 2
