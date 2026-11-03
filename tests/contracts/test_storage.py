from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import JSON, inspect, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from app.db import Store, observations, records


@pytest.fixture
def store() -> Store:
    value = Store("sqlite+pysqlite:///:memory:")
    yield value
    value.close()


def test_records_keep_kinds_separate_and_support_updates(store: Store) -> None:
    payload = {"id": "same-id", "name": "工作面 A", "thresholds": {"gas": 1.0}}

    assert store.put("faces", "same-id", payload) == payload
    store.put("sessions", "same-id", {"status": "running"})
    store.put("faces", "same-id", {"name": "工作面 B"})

    assert store.get("faces", "same-id") == {"name": "工作面 B"}
    assert store.get("sessions", "same-id") == {"status": "running"}
    assert store.get("faces", "unknown") is None
    assert store.list("unknown") == []


def test_list_only_returns_requested_kind_in_id_order(store: Store) -> None:
    store.put("faces", "b", {"name": "B"})
    store.put("sessions", "a", {"name": "excluded"})
    store.put("faces", "a", {"name": "A"})

    assert store.list("faces") == [{"name": "A"}, {"name": "B"}]


def test_delete_removes_only_requested_record_and_allows_missing_id(store: Store) -> None:
    store.put("faces", "face-1", {"name": "A"})
    store.put("sessions", "face-1", {"status": "running"})

    store.delete("faces", "face-1")
    store.delete("faces", "unknown")

    assert store.get("faces", "face-1") is None
    assert store.get("sessions", "face-1") == {"status": "running"}


def test_json_values_are_independent_from_caller_mutation(store: Store) -> None:
    payload = {"flags": ["quality_ok"], "details": {"threshold": 1.0}}
    store.put("faces", "face-1", payload)
    payload["flags"].append("caller-change")

    loaded = store.get("faces", "face-1")
    loaded["details"]["threshold"] = 5.0

    assert store.get("faces", "face-1") == {
        "flags": ["quality_ok"],
        "details": {"threshold": 1.0},
    }


def test_transaction_commits_explicit_connection_operations(store: Store) -> None:
    with store.transaction() as connection:
        store.put("faces", "face-1", {"name": "A"}, connection=connection)
        store.put("faces", "face-2", {"name": "B"}, connection=connection)
        assert store.get("faces", "face-1", connection=connection) == {"name": "A"}
        assert len(store.list("faces", connection=connection)) == 2
        store.delete("faces", "face-2", connection=connection)

    assert store.list("faces") == [{"name": "A"}]


def test_transaction_exception_rolls_back_insert_update_and_delete(store: Store) -> None:
    store.put("faces", "old", {"name": "original"})
    store.put("faces", "deleted", {"name": "keep"})

    with pytest.raises(RuntimeError, match="abort transaction"):
        with store.transaction() as connection:
            store.put("faces", "new", {"name": "new"}, connection=connection)
            store.put("faces", "old", {"name": "changed"}, connection=connection)
            store.delete("faces", "deleted", connection=connection)
            raise RuntimeError("abort transaction")

    assert store.get("faces", "new") is None
    assert store.get("faces", "old") == {"name": "original"}
    assert store.get("faces", "deleted") == {"name": "keep"}


def test_transactions_serialize_concurrent_read_modify_write(store: Store) -> None:
    store.put("counters", "total", {"value": 0})

    def increment(_: int) -> None:
        for _ in range(12):
            with store.transaction() as connection:
                payload = store.get("counters", "total", connection=connection)
                store.put(
                    "counters", "total", {"value": payload["value"] + 1}, connection=connection
                )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(increment, range(4)))

    assert store.get("counters", "total") == {"value": 48}
    assert isinstance(store.engine.pool, StaticPool)


def test_sqlite_file_creates_parent_directories_and_persists(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "data" / "state.db"
    url = "sqlite+pysqlite:///" + database.as_posix()
    first = Store(url)
    try:
        first.put("faces", "face-1", {"name": "A"})
    finally:
        first.close()

    assert database.is_file()
    second = Store(url)
    try:
        assert second.get("faces", "face-1") == {"name": "A"}
    finally:
        second.close()


def test_records_and_observations_have_required_columns_and_indexes(store: Store) -> None:
    inspector = inspect(store.engine)

    assert set(inspector.get_table_names()) == {"records", "observations"}
    assert inspector.get_pk_constraint("records")["constrained_columns"] == ["kind", "id"]
    assert inspector.get_pk_constraint("observations")["constrained_columns"] == ["id"]
    assert set(observations.c.keys()) == {
        "id", "point_id", "source_id", "event_time", "available_at", "raw_value",
        "value", "quality_flags", "source_file_id", "row_number", "import_id",
    }
    indexes = inspector.get_indexes("observations")
    assert any(index["column_names"] == ["point_id"] for index in indexes)
    assert any(
        index["column_names"] == ["point_id", "event_time", "available_at"]
        for index in indexes
    )
    assert isinstance(records.c.payload.type, JSON)
    assert store.engine.hide_parameters is True


def test_database_errors_hide_bound_values(store: Store) -> None:
    store.put("faces", "face-1", {"name": "A"})

    with pytest.raises(IntegrityError) as caught:
        with store.transaction() as connection:
            connection.execute(
                insert(records).values(
                    kind="faces", id="face-1", payload={"value": "private-test-value"}
                )
            )

    assert "private-test-value" not in str(caught.value)
    assert store.get("faces", "face-1") == {"name": "A"}


def test_observations_store_nullable_values_and_metadata_in_transaction(store: Store) -> None:
    row = {
        "id": "observation-1",
        "point_id": "point-1",
        "source_id": "source-1",
        "event_time": "2026-11-03T02:00:00.000000+00:00",
        "available_at": "2026-11-03T02:00:01.000000+00:00",
        "raw_value": None,
        "value": None,
        "quality_flags": ["missing"],
        "source_file_id": "file-1",
        "row_number": 2,
        "import_id": "import-1",
    }

    with store.transaction() as connection:
        connection.execute(insert(observations).values(**row))

    with store.engine.connect() as connection:
        actual = connection.execute(select(observations)).mappings().one()
    assert dict(actual) == row
