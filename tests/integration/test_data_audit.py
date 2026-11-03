from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import insert

from app.db import Store, observations
from app.schemas.common import Actor, DomainError, ExecutionContext, utcstr
from app.services.data.service import DataService


@pytest.fixture
def setup(tmp_path):
    store = Store("sqlite://")
    service = DataService(store, tmp_path / "uploads")
    service.register_source("s", "Synthetic", face_id="31002")
    service.register_point("p", "s", "tag", "Methane", "methane", "%")
    actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
    yield store, service, actor
    store.close()


def test_corrupt_xlsx_is_reported_as_domain_error(setup, tmp_path):
    _, service, actor = setup
    file = tmp_path / "broken.xlsx"
    file.write_bytes(b"not-an-xlsx")
    registered = service.register_file(file, actor)
    with pytest.raises(DomainError):
        service.import_history(actor, "s", registered["file_id"], {"time_column": "STime", "value_columns": {"SValue": "p"}, "lineage": "raw"})


def test_stream_failure_after_flushed_batch_rolls_back_import(setup, tmp_path, monkeypatch):
    store, service, actor = setup
    file = tmp_path / "source.csv"
    file.write_text("STime,SValue\n", encoding="utf-8")
    registered = service.register_file(file, actor)

    def broken_rows(path, mapping):
        for index in range(501):
            yield index + 2, {"STime": "2026-10-01T08:00:00+08:00", "SValue": "0.4"}
        raise OSError("stream read failed")

    monkeypatch.setattr(service, "_rows", broken_rows)
    with pytest.raises(DomainError):
        service.import_history(actor, "s", registered["file_id"], {"time_column": "STime", "value_columns": {"SValue": "p"}, "lineage": "raw"})
    with store.engine.connect() as connection:
        assert list(connection.execute(observations.select())) == []
    assert store.list("data_import") == []


def test_multiple_value_columns_cannot_silently_overwrite_same_point(setup, tmp_path):
    _, service, actor = setup
    file = tmp_path / "ambiguous.csv"
    file.write_text("STime,SValue,Other\n2026-10-01T08:00:00+08:00,0.4,0.9\n", encoding="utf-8")
    registered = service.register_file(file, actor)
    with pytest.raises(DomainError):
        service.import_history(actor, "s", registered["file_id"], {"time_column": "STime", "value_columns": {"SValue": "p", "Other": "p"}, "lineage": "raw"})


def test_query_sql_also_enforces_observation_source_partition(setup):
    store, service, actor = setup
    time = datetime(2026, 10, 1, tzinfo=timezone.utc)
    with store.transaction() as connection:
        connection.execute(insert(observations).values(id="foreign", point_id="p", source_id="other-source", event_time=utcstr(time), available_at=utcstr(time), value=0.9, raw_value="0.9", quality_flags=[], source_file_id="file", row_number=2, import_id="import"))
    context = ExecutionContext(session_id="s1", mode="historical_replay", source_id="s", as_of=time, actor=actor)
    packet = service.query_series(context, "p", time - timedelta(seconds=1), time + timedelta(seconds=1))
    assert packet["total"] == 0
