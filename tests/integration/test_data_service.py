"""Offline acceptance checks using explicit, synthetic 31002 observations."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from openpyxl import Workbook
import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.data.service import DataService


SOURCE_ID = "history-31002"
POINT_ID = "31002:T0"
SOURCE_TAG = "METHANE.SENSOR.7"


def local_time(clock: str) -> datetime:
    return datetime.fromisoformat(f"2026-10-01T{clock}+08:00")


def parsed_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def context(actor: Actor, as_of: str = "09:00:00", **overrides) -> ExecutionContext:
    values = {
        "session_id": "data-acceptance",
        "mode": "historical_replay",
        "source_id": SOURCE_ID,
        "as_of": local_time(as_of),
        "config_version": "test-v1",
        "actor": actor,
    }
    values.update(overrides)
    return ExecutionContext(**values)


def mapping(**overrides) -> dict:
    values = {
        "time_column": "STime",
        "timezone": "+08:00",
        "value_columns": {"SValue": POINT_ID},
        "availability_column": None,
        "sheet": None,
        "header_row": 1,
        "invalid_values": [-999999],
        "lineage": "raw",
    }
    values.update(overrides)
    return values


@pytest.fixture
def actor() -> Actor:
    return Actor(
        user_id="u",
        roles=("safety_management",),
        face_ids=("31002",),
    )


@pytest.fixture
def data_service(tmp_path: Path):
    store = Store("sqlite+pysqlite:///:memory:")
    service = DataService(store=store, upload_dir=tmp_path / "uploads")
    service.register_source(SOURCE_ID, "31002 history", face_id="31002")
    service.register_point(
        POINT_ID,
        SOURCE_ID,
        SOURCE_TAG,
        "Synthetic methane sensor",
        "methane",
        "%",
    )
    yield service
    store.engine.dispose()


@pytest.fixture
def import_csv(data_service: DataService, actor: Actor, tmp_path: Path):
    file_number = 0

    def import_rows(rows, *, columns=("STime", "SValue"), column_mapping=None, **kwargs):
        nonlocal file_number
        file_number += 1
        path = tmp_path / f"synthetic-{file_number}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)
        registered = data_service.register_file(path, actor)
        report = data_service.import_history(
            actor=actor,
            source_id=SOURCE_ID,
            file_id=registered["file_id"],
            mapping=column_mapping or mapping(),
            **kwargs,
        )
        return path, registered, report

    return import_rows


def query(data_service: DataService, actor: Actor, **overrides) -> dict:
    values = {
        "context": context(actor),
        "point_id": POINT_ID,
        "start_at": local_time("08:00:00"),
        "end_at": local_time("09:00:00"),
    }
    values.update(overrides)
    return data_service.query_series(**values)


def test_register_file_copies_bytes_and_preserves_original_hash(
    data_service: DataService, actor: Actor, tmp_path: Path
):
    original = tmp_path / "original.csv"
    original.write_bytes(b"STime,SValue\r\n2026-10-01T08:00:00+08:00,0.4\r\n")
    original_bytes = original.read_bytes()
    original_mtime = original.stat().st_mtime_ns
    digest = hashlib.sha256(original_bytes).hexdigest()

    registered = data_service.register_file(original, actor)

    assert registered["file_id"]
    assert registered["sha256"] == digest
    assert original.read_bytes() == original_bytes
    assert original.stat().st_mtime_ns == original_mtime
    copies = [item for item in (tmp_path / "uploads").rglob("*") if item.is_file()]
    assert len(copies) == 1
    assert copies[0].read_bytes() == original_bytes


def test_catalog_preserves_explicit_point_metadata_and_scopes_work_face(
    data_service: DataService, actor: Actor
):
    data_service.register_source("history-32001", "Other face", face_id="32001")
    data_service.register_point(
        "32001:T0", "history-32001", "OTHER.TAG", "Other sensor", "methane", "%"
    )

    sources = data_service.list_sources(actor)
    points = data_service.list_points(actor)

    assert [item["source_id"] for item in sources] == [SOURCE_ID]
    assert [item["point_id"] for item in points] == [POINT_ID]
    assert points[0]["source_tag"] == SOURCE_TAG
    assert points[0]["unit"] == "%"
    assert points[0]["unit_status"] == "inferred"
    assert points[0]["mapping_status"] == "confirmed"
    assert points[0]["lineage"] == "raw"


def test_csv_keeps_utc_times_raw_values_and_source_row_provenance(
    data_service: DataService, actor: Actor, import_csv
):
    _, registered, report = import_csv(
        [("2026-10-01T08:00:00+08:00", "0.400")]
    )

    result = query(data_service, actor)

    assert report["imported_rows"] == 1
    assert result["total"] == 1
    row = result["rows"][0]
    assert parsed_time(row["event_time"]) == local_time("08:00:00")
    assert parsed_time(row["event_time"]).utcoffset() == timezone.utc.utcoffset(None)
    assert parsed_time(row["available_at"]) == local_time("08:00:00")
    assert row["raw_value"] == "0.400"
    assert row["value"] == pytest.approx(0.4)
    assert row["source_file_id"] == registered["file_id"]
    assert row["row_number"] == 2
    assert "availability_inferred" in row["quality_flags"]


@pytest.mark.parametrize("raw_value", ["-0.1", "-999999", "nan", "inf", "-inf", "bad"])
def test_invalid_methane_values_retain_raw_input_and_null_numeric_value(
    data_service: DataService, actor: Actor, import_csv, raw_value: str
):
    _, _, report = import_csv([("2026-10-01T08:00:00+08:00", raw_value)])

    row = query(data_service, actor)["rows"][0]

    assert report["imported_rows"] == 1
    assert row["raw_value"] == raw_value
    assert row["value"] is None
    assert "invalid_value" in row["quality_flags"]


@pytest.mark.parametrize("raw_value", ["0", "45"])
def test_zero_and_high_methane_values_remain_numeric(
    data_service: DataService, actor: Actor, import_csv, raw_value: str
):
    import_csv([("2026-10-01T08:00:00+08:00", raw_value)])

    row = query(data_service, actor)["rows"][0]

    assert row["value"] == float(raw_value)
    assert "invalid_value" not in row["quality_flags"]


def test_invalid_timestamps_are_counted_without_inventing_observation_times(
    data_service: DataService, actor: Actor, import_csv
):
    _, _, report = import_csv(
        [("not-a-timestamp", "0.9"), ("2026-10-01T08:01:00+08:00", "0.4")]
    )

    result = query(data_service, actor)

    assert report["invalid_time_rows"] == 1
    assert report["imported_rows"] == 1
    assert result["total"] == 1
    assert result["rows"][0]["row_number"] == 3


def test_conflicting_duplicate_timestamps_keep_both_observations(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv(
        [
            ("2026-10-01T08:00:00+08:00", "0.4"),
            ("2026-10-01T08:00:00+08:00", "0.6"),
        ]
    )

    result = query(data_service, actor)

    assert result["total"] == 2
    assert sorted(row["value"] for row in result["rows"]) == [0.4, 0.6]
    assert {row["row_number"] for row in result["rows"]} == {2, 3}
    assert all("duplicate_timestamp" in row["quality_flags"] for row in result["rows"])


def test_repeated_file_and_identical_copy_skip_observation_import(
    data_service: DataService, actor: Actor, import_csv, tmp_path: Path
):
    path, registered, first = import_csv([("2026-10-01T08:00:00+08:00", "0.4")])
    repeated = data_service.import_history(
        actor, SOURCE_ID, registered["file_id"], mapping()
    )
    copy_path = tmp_path / "identical-copy.csv"
    copy_path.write_bytes(path.read_bytes())
    copied_file = data_service.register_file(copy_path, actor)
    copied = data_service.import_history(
        actor, SOURCE_ID, copied_file["file_id"], mapping()
    )

    assert first["imported_rows"] == 1
    assert repeated["duplicate_file"] is True
    assert copied["duplicate_file"] is True
    assert repeated["imported_rows"] == copied["imported_rows"] == 0
    assert query(data_service, actor)["total"] == 1


def test_different_date_ranges_can_import_other_segments_of_one_file(
    data_service: DataService, actor: Actor, import_csv
):
    _, registered, first = import_csv(
        [
            ("2026-10-01T08:00:00+08:00", "0.4"),
            ("2026-10-01T08:02:00+08:00", "0.6"),
        ],
        start_at=local_time("08:00:00"),
        end_at=local_time("08:01:00"),
    )
    second = data_service.import_history(
        actor,
        SOURCE_ID,
        registered["file_id"],
        mapping(),
        start_at=local_time("08:02:00"),
        end_at=local_time("08:03:00"),
    )

    assert first["imported_rows"] == second["imported_rows"] == 1
    assert second["duplicate_file"] is False
    assert query(data_service, actor)["total"] == 2


def test_long_csv_uses_exact_registered_tag_instead_of_face_prefix(
    data_service: DataService, actor: Actor, import_csv
):
    _, _, report = import_csv(
        [
            (SOURCE_TAG, "2026-10-01T08:00:00+08:00", "0.4"),
            ("31002-UNREGISTERED", "2026-10-01T08:01:00+08:00", "99"),
        ],
        columns=("tagname", "timestamp", "value"),
        column_mapping=mapping(time_column="timestamp", value_columns={"value": POINT_ID}),
    )

    result = query(data_service, actor)

    assert report["imported_rows"] == 1
    assert result["total"] == 1
    assert result["rows"][0]["value"] == 0.4


def test_xlsx_respects_sheet_header_and_column_mapping(
    data_service: DataService, actor: Actor, tmp_path: Path
):
    path = tmp_path / "synthetic.xlsx"
    workbook = Workbook()
    workbook.active.title = "Ignore"
    workbook.active.append(["Unused", "Rows"])
    sheet = workbook.create_sheet("Readings")
    sheet.append(["Synthetic fixture", "No measured observations"])
    sheet.append(["STime", "SValue"])
    sheet.append(["2026-10-01T08:00:00+08:00", 0.4])
    workbook.save(path)
    original = path.read_bytes()
    registered = data_service.register_file(path, actor)

    report = data_service.import_history(
        actor,
        SOURCE_ID,
        registered["file_id"],
        mapping(sheet="Readings", header_row=2),
    )
    result = query(data_service, actor)

    assert report["imported_rows"] == 1
    assert result["total"] == 1
    assert result["rows"][0]["value"] == 0.4
    assert result["rows"][0]["row_number"] == 3
    assert path.read_bytes() == original


def test_query_respects_both_as_of_and_available_at(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv(
        [
            ("2026-10-01T08:00:00+08:00", "0.4", "2026-10-01T08:00:00+08:00"),
            ("2026-10-01T08:01:00+08:00", "0.6", "2026-10-01T08:10:00+08:00"),
            ("2026-10-01T08:06:00+08:00", "0.8", "2026-10-01T08:06:00+08:00"),
        ],
        columns=("STime", "SValue", "AvailableAt"),
        column_mapping=mapping(availability_column="AvailableAt"),
    )

    result = query(data_service, actor, context=context(actor, "08:05:00"))

    assert result["total"] == 1
    assert result["rows"][0]["value"] == 0.4


def test_query_end_at_limits_a_later_execution_context(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv(
        [
            ("2026-10-01T08:01:00+08:00", "0.4"),
            ("2026-10-01T08:03:00+08:00", "0.6"),
        ]
    )

    result = query(
        data_service,
        actor,
        context=context(actor, "08:05:00"),
        end_at=local_time("08:02:00"),
    )

    assert result["total"] == 1
    assert result["rows"][0]["value"] == 0.4


@pytest.mark.parametrize("failure", ["source", "face"])
def test_query_rejects_context_source_or_actor_face_mismatch(
    data_service: DataService, actor: Actor, import_csv, failure: str
):
    import_csv([("2026-10-01T08:00:00+08:00", "0.4")])
    if failure == "source":
        execution = context(actor, source_id="different-source")
    else:
        outsider = actor.model_copy(update={"face_ids": ("32001",)})
        execution = context(outsider)

    with pytest.raises(DomainError) as caught:
        query(data_service, actor, context=execution)

    assert caught.value.status_code == 403


def test_import_requires_actor_access_to_source_face(
    data_service: DataService, actor: Actor, tmp_path: Path
):
    path = tmp_path / "restricted.csv"
    path.write_text("STime,SValue\n2026-10-01T08:00:00+08:00,0.4\n", encoding="utf-8")
    registered = data_service.register_file(path, actor)
    outsider = actor.model_copy(update={"face_ids": ("32001",)})

    with pytest.raises(DomainError) as caught:
        data_service.import_history(outsider, SOURCE_ID, registered["file_id"], mapping())

    assert caught.value.status_code == 403
    assert query(data_service, actor)["total"] == 0


def test_query_returns_stable_chronological_pagination(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv(
        [
            ("2026-10-01T08:03:00+08:00", "0.4"),
            ("2026-10-01T08:00:00+08:00", "0.1"),
            ("2026-10-01T08:02:00+08:00", "0.3"),
            ("2026-10-01T08:01:00+08:00", "0.2"),
        ]
    )

    result = query(data_service, actor, limit=2, offset=1)

    assert result["total"] == 4
    assert [row["value"] for row in result["rows"]] == [0.2, 0.3]


def test_aggregation_uses_utc_half_open_buckets_and_only_closed_intervals(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv(
        [
            ("2026-10-01T08:00:10+08:00", "0.2"),
            ("2026-10-01T08:00:55+08:00", "0.6"),
            ("2026-10-01T08:01:00+08:00", "1.0"),
            ("2026-10-01T08:01:50+08:00", "1.4"),
            ("2026-10-01T08:02:10+08:00", "99"),
        ]
    )

    result = query(
        data_service,
        actor,
        context=context(actor, "08:02:30"),
        aggregation_seconds=60,
    )

    assert result["total"] == 2
    first, second = result["rows"]
    assert parsed_time(first["event_time"]) == local_time("08:00:00")
    assert parsed_time(first["bucket_end"]) == local_time("08:01:00")
    assert parsed_time(second["event_time"]) == local_time("08:01:00")
    assert parsed_time(second["bucket_end"]) == local_time("08:02:00")
    assert (first["min"], first["max"], first["count"]) == (0.2, 0.6, 2)
    assert first["mean"] == pytest.approx(0.4)
    assert (second["min"], second["max"], second["count"]) == (1.0, 1.4, 2)
    assert second["mean"] == pytest.approx(1.2)


def test_forward_fill_uses_past_valid_values_and_stops_at_gap_limit(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv(
        [
            ("2026-10-01T08:00:10+08:00", "0.4"),
            ("2026-10-01T08:01:15+08:00", "-1"),
            ("2026-10-01T08:04:10+08:00", "0.9"),
        ]
    )

    result = query(
        data_service,
        actor,
        context=context(actor, "08:06:00"),
        aggregation_seconds=60,
        fill_max_seconds=120,
    )

    assert [row["value"] for row in result["rows"]] == [0.4, 0.4, 0.4, None, 0.9, 0.9]
    assert "forward_filled" in result["rows"][1]["quality_flags"]
    assert result["rows"][1]["count"] == 0
    assert "forward_filled" not in result["rows"][3]["quality_flags"]


def test_non_hour_divisor_aggregation_stays_on_utc_epoch_boundaries(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv(
        [
            ("2026-10-01T08:00:30+08:00", "0.2"),
            ("2026-10-01T08:06:59+08:00", "0.6"),
            ("2026-10-01T08:07:00+08:00", "1.0"),
        ]
    )

    result = query(
        data_service,
        actor,
        context=context(actor, "08:14:00"),
        aggregation_seconds=420,
    )

    assert [parsed_time(row["event_time"]) for row in result["rows"]] == [
        local_time("08:00:00"),
        local_time("08:07:00"),
    ]
    assert [row["count"] for row in result["rows"]] == [2, 1]
    assert result["rows"][0]["mean"] == pytest.approx(0.4)


def test_future_value_does_not_fill_earlier_empty_buckets(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv([("2026-10-01T08:02:10+08:00", "0.6")])

    result = query(
        data_service,
        actor,
        context=context(actor, "08:03:00"),
        aggregation_seconds=60,
        fill_max_seconds=120,
    )

    assert [row["value"] for row in result["rows"]] == [None, None, 0.6]


def test_future_or_late_arriving_records_do_not_change_past_aggregation_and_fill(
    data_service: DataService, actor: Actor, import_csv
):
    import_csv(
        [("2026-10-01T08:00:10+08:00", "0.4", "2026-10-01T08:00:10+08:00")],
        columns=("STime", "SValue", "AvailableAt"),
        column_mapping=mapping(availability_column="AvailableAt"),
    )
    options = {
        "context": context(actor, "08:03:00"),
        "aggregation_seconds": 60,
        "fill_max_seconds": 120,
    }
    before = query(data_service, actor, **options)
    import_csv(
        [
            ("2026-10-01T08:00:10+08:00", "99", "2026-10-01T08:04:00+08:00"),
            ("2026-10-01T08:02:10+08:00", "77", "2026-10-01T08:04:00+08:00"),
            ("2026-10-01T08:05:00+08:00", "55", "2026-10-01T08:05:00+08:00"),
        ],
        columns=("STime", "SValue", "AvailableAt"),
        column_mapping=mapping(availability_column="AvailableAt"),
    )

    after = query(data_service, actor, **options)

    assert after["rows"] == before["rows"]
    assert after["quality"] == before["quality"]
    assert after["total"] == before["total"]


@pytest.mark.parametrize("lineage", ["unknown", "processed"])
def test_lineage_quality_is_exposed_on_observations_and_query_summary(
    data_service: DataService, actor: Actor, import_csv, lineage: str
):
    import_csv(
        [("2026-10-01T08:00:00+08:00", "0.4")],
        column_mapping=mapping(lineage=lineage),
    )

    result = query(data_service, actor)

    flag = f"lineage_{lineage}"
    assert flag in result["rows"][0]["quality_flags"]
    assert flag in result["quality"]["flags"]
