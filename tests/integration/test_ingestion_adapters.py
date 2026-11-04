"""Source-bound JSON adapters keep stable identities and explicit source time."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.common import DomainError
from app.services.data.ingestion_adapters import normalize, validate_config

NOW = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
POINTS = [dict(point_id="p", source_id="live", source_tag="T0"),
          dict(point_id="foreign", source_id="other", source_tag="T1")]


def spec(kind="standard_json", **changes):
    result = dict(source_id="live", adapter_kind=kind, mapping={}, tag_map={})
    if kind == "mapped_json":
        result.update(mapping=dict(records_path="data.rows", id_field="key", tag_field="tag", time_field="time",
                                   value_field="value", sequence_field="sequence", timezone="+08:00"), tag_map={"T0": "p"})
    return {**result, **changes}


def test_standard_json_preserves_original_id_value_and_converts_source_time():
    payload = {"readings": [dict(reading_id="source-7", point_id="p", event_time="2026-11-04T19:59:00+08:00", raw_value="0.70")]}
    original = deepcopy(payload)
    rows = normalize(spec(), payload, POINTS, received_at=NOW)
    assert rows == [dict(reading_id="source-7", point_id="p", event_time="2026-11-04T11:59:00.000000+00:00", raw_value="0.70")]
    assert payload == original


def test_mapped_json_uses_nested_fields_registered_tags_and_explicit_timezone():
    payload = {"data": {"rows": [dict(key=12, tag="T0", time="2026-11-04 19:59:00", value=.7, sequence=3)]}}
    rows = normalize(spec("mapped_json"), payload, POINTS, received_at=NOW)
    assert rows[0]["reading_id"] == "12" and rows[0]["point_id"] == "p"
    assert rows[0]["event_time"] == "2026-11-04T11:59:00.000000+00:00" and rows[0]["raw_value"] == .7


def test_root_array_and_stable_fallback_identity_do_not_depend_on_value_or_arrival():
    config = spec("mapped_json")
    config["mapping"]["records_path"] = ""
    a = dict(tag="T0", time="2026-11-04T11:59:00Z", value=.7, sequence=3)
    first = normalize(config, [a], POINTS, received_at=NOW)[0]
    duplicate = normalize(config, [{**a, "time": "2026-11-04T19:59:00+08:00", "value": .8}], POINTS,
                          received_at=NOW+timedelta(minutes=1))[0]
    changed_sequence = normalize(config, [{**a, "sequence": 4}], POINTS, received_at=NOW)[0]
    assert first["reading_id"] == duplicate["reading_id"]
    assert first["reading_id"] != changed_sequence["reading_id"]


@pytest.mark.parametrize("changes", [
    {"adapter_kind": "unknown"},
    {"mapping": {"records_path": "data..rows", "tag_field": "tag", "time_field": "time", "value_field": "value"}},
    {"mapping": {"tag_field": "tag", "time_field": "time"}},
    {"tag_map": {}},
    {"tag_map": {"T0": "missing"}},
    {"mapping": {"tag_field": "tag", "time_field": "time", "value_field": "value", "timezone": "+99:00"}},
])
def test_invalid_adapter_configuration_is_rejected_before_payloads(changes):
    with pytest.raises(DomainError):
        validate_config(spec("mapped_json", **changes), POINTS)


def test_cross_source_mapping_and_native_point_are_forbidden():
    with pytest.raises(DomainError) as mapped:
        validate_config(spec("mapped_json", tag_map={"T0": "foreign"}), POINTS)
    assert mapped.value.status_code == 403
    with pytest.raises(DomainError) as native:
        normalize(spec(), [dict(point_id="foreign", event_time=NOW.isoformat(), raw_value=.1)], POINTS, received_at=NOW)
    assert native.value.status_code == 403


@pytest.mark.parametrize("reading", [
    dict(point_id="p", event_time="2026-11-04 11:59:00", raw_value=.1),
    dict(point_id="p", event_time="2026-11-04T12:00:01Z", raw_value=.1),
    dict(point_id="p", event_time=123, raw_value=.1),
    dict(point_id="p", event_time="2026-11-04T11:59:00Z", raw_value=True),
    dict(point_id="p", event_time="2026-11-04T11:59:00Z", raw_value={"value": .1}),
    dict(point_id="p", event_time="2026-11-04T11:59:00Z"),
])
def test_malformed_reading_time_or_scalar_value_is_rejected(reading):
    with pytest.raises(DomainError):
        normalize(spec(), [reading], POINTS, received_at=NOW)


def test_configured_source_timezone_can_resolve_native_naive_time_and_null_is_preserved():
    rows = normalize(spec(timezone="+08:00"), [dict(point_id="p", event_time="2026-11-04 19:59:00", raw_value=None)],
                     POINTS, received_at=NOW)
    assert rows[0]["raw_value"] is None and rows[0]["reading_id"]


def test_source_time_outside_utc_datetime_range_is_a_validation_error():
    with pytest.raises(DomainError):
        normalize(spec(timezone="+08:00"),
                  [dict(point_id="p", event_time="0001-01-01T00:00:00+08:00", raw_value=.1)], POINTS, received_at=NOW)


@pytest.mark.parametrize("payload", [[], {}, {"readings": "rows"}, [None]])
def test_payload_requires_a_bounded_array_of_record_objects(payload):
    with pytest.raises(DomainError):
        normalize(spec(), payload, POINTS, received_at=NOW)
