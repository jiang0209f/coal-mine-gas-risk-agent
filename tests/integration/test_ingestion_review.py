"""Field ingestion identity remains unambiguous and reads legacy observations."""
from datetime import datetime, timezone
import hashlib

import pytest
from sqlalchemy import select, update

from app.db import Store, observations
from app.schemas.common import Actor, DomainError
from app.services.data.service import DataService


@pytest.fixture
def live(tmp_path):
    store = Store("sqlite+pysqlite:///:memory:")
    data = DataService(store, tmp_path)
    data.register_source("s", "Synthetic", "31002")
    for point in ("p", "p:r"):
        data.register_point(point, "s", point, point, "methane", "%", unit_status="confirmed")
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    moment = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    yield store, data, actor, moment
    store.close()


def reading(moment, point="p", identity="r:q", value="0.70"):
    return {"reading_id": identity, "point_id": point, "event_time": moment, "raw_value": value}


def legacy_row(live):
    store, data, actor, moment = live
    row = reading(moment)
    data.ingest_readings(actor, "s", [row], received_at=moment)
    legacy_id = hashlib.sha256(b"live:s:p:r:q").hexdigest()
    with store.transaction() as connection:
        current_id = connection.execute(select(observations.c.id)).scalar_one()
        connection.execute(update(observations).where(observations.c.id == current_id).values(id=legacy_id))
    return row, legacy_id


def test_old_live_record_keeps_idempotency_and_first_availability(live):
    store, data, actor, moment = live
    row, legacy_id = legacy_row(live)
    later = moment.replace(minute=10)
    result = data.ingest_readings(actor, "s", [row], received_at=later)
    assert result["imported_rows"] == 0 and result["duplicate_rows"] == 1
    with store.transaction() as connection:
        saved = connection.execute(select(observations)).mappings().one()
    assert saved["id"] == legacy_id and datetime.fromisoformat(saved["available_at"]) == moment


def test_old_live_record_still_rejects_same_identity_changed_content(live):
    store, data, actor, moment = live
    row, legacy_id = legacy_row(live)
    with pytest.raises(DomainError) as caught:
        data.ingest_readings(actor, "s", [{**row, "raw_value": "0.90"}], received_at=moment)
    assert caught.value.status_code == 409
    with store.transaction() as connection:
        saved = connection.execute(select(observations)).mappings().one()
    assert saved["id"] == legacy_id and saved["raw_value"] == "0.70"


def test_distinct_source_point_reading_tuples_with_colons_both_import(live):
    store, data, actor, moment = live
    _, legacy_id = legacy_row(live)
    different = reading(moment, point="p:r", identity="q")
    result = data.ingest_readings(actor, "s", [different], received_at=moment)
    assert result["imported_rows"] == 1 and result["duplicate_rows"] == 0
    repeated = data.ingest_readings(actor, "s", [different], received_at=moment)
    assert repeated["imported_rows"] == 0 and repeated["duplicate_rows"] == 1
    with store.transaction() as connection:
        rows = list(connection.execute(select(observations)).mappings())
    assert len(rows) == 2 and legacy_id in {row["id"] for row in rows}
    assert {row["point_id"] for row in rows} == {"p", "p:r"}
