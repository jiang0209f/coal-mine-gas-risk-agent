from datetime import datetime, timedelta, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.data.service import DataService


def test_live_arrival_visibility_dedup_and_atomic_validation(tmp_path):
    store = Store("sqlite:///:memory:")
    service = DataService(store, tmp_path)
    service.register_source("s", "live", "31002")
    service.register_point("p", "s", "tag", "T0", "methane", "%")
    actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
    arrival = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row = {"reading_id": "a", "point_id": "p", "event_time": arrival - timedelta(minutes=1), "raw_value": "0.90"}
    result = service.ingest_readings(actor, "s", [row], received_at=arrival)
    assert result["imported_rows"] == 1
    assert service.ingest_readings(actor, "s", [row], received_at=arrival)["imported_rows"] == 0
    ctx = ExecutionContext(actor=actor, session_id="live", mode="realtime", source_id="s", as_of=arrival-timedelta(seconds=1))
    assert service.query_series(ctx, "p", arrival-timedelta(hours=1), arrival)["total"] == 0
    visible = service.query_series(ctx.model_copy(update={"as_of": arrival}), "p", arrival-timedelta(hours=1), arrival)
    assert visible["rows"][0]["available_at"] == arrival.isoformat(timespec="microseconds")
    assert visible["rows"][0]["raw_value"] == "0.90"
    assert "availability_inferred" not in visible["rows"][0]["quality_flags"]
    with pytest.raises(DomainError):
        service.ingest_readings(actor, "s", [{**row, "reading_id": "b"}, {**row, "point_id": "missing"}], received_at=arrival)
    assert service.query_series(ctx.model_copy(update={"as_of": arrival}), "p", arrival-timedelta(hours=1), arrival)["total"] == 1
    with pytest.raises(DomainError) as error:
        service.ingest_readings(actor, "s", [{**row, "raw_value": "5"}], received_at=arrival)
    assert error.value.status_code == 409
