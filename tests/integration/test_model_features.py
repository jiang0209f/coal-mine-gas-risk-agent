"""The same past-only features are used in training and service inference."""

from datetime import datetime, timedelta, timezone
from copy import deepcopy

import pytest

from app.adapters.model_bundle import ModelNotReady
from app.adapters.model_features import (
    FORECAST_FEATURE_NAMES, RESIDUAL_FEATURE_NAMES,
    aggregate_five_minute, forecast_request, attribution_request, residual_vector,
)


def minutes(count=450):
    start = datetime(2022, 2, 5, tzinfo=timezone.utc)
    return [dict(event_time=(start+timedelta(minutes=i)).isoformat(),
                 values=[i / 1000, i, 2, 3, 4, 5]) for i in range(count)]


def test_complete_five_minute_bucket_uses_target_maximum_and_other_means():
    rows = minutes(5)
    result = aggregate_five_minute(rows)
    assert len(result) == 1
    assert result[0]["values"] == [2.0, 2.0, 3.0, 4.0, 5.0, .004]
    assert datetime.fromisoformat(result[0]["bucket_end"]) - datetime.fromisoformat(result[0]["event_time"]) == timedelta(minutes=5)


def test_incomplete_bucket_is_omitted_and_duplicate_minute_rejected():
    assert aggregate_five_minute(minutes(4)) == []
    with pytest.raises(ValueError):
        aggregate_five_minute(minutes(5)+minutes(1))


def test_forecast_features_ignore_every_observation_after_cutoff():
    cutoff = datetime(2022, 2, 5, 7, 30, tzinfo=timezone.utc)
    original = forecast_request(minutes(), cutoff)
    rows = minutes(460)
    for row in rows[450:]:
        row["values"] = [9999] * 6
    assert forecast_request(rows, cutoff) == original
    assert original["feature_names"] == FORECAST_FEATURE_NAMES
    assert len(original["values"]) == 90
    assert original["values"][-1][-1] == .449
    assert len(original["residual_values"]) == len(RESIDUAL_FEATURE_NAMES) == 72


def test_missing_minute_prevents_forecast_instead_of_interpolation():
    rows = minutes()
    rows.pop(100)
    with pytest.raises(ModelNotReady):
        forecast_request(rows, datetime(2022, 2, 5, 7, 30, tzinfo=timezone.utc))


def test_residual_features_are_old_to_new_and_do_not_modify_inputs():
    values = [[float(i)] * 6 for i in range(90)]
    before = deepcopy(values)
    result = residual_vector(values)
    assert result[:6] == [78.] * 6
    assert result[-6:] == [89.] * 6
    assert values == before


def test_attribution_uses_61_inclusive_minutes_and_never_later_values():
    rows = minutes(70)
    cutoff = datetime(2022, 2, 5, 1, tzinfo=timezone.utc)
    result = attribution_request(rows, cutoff)
    assert len(result["values"]) == 61
    assert result["values"][-1] == rows[60]["values"]
    assert datetime.fromisoformat(result["times"][-1]) == cutoff


@pytest.mark.parametrize("value", [True, None, float("nan"), float("inf")])
def test_invalid_inputs_are_rejected(value):
    rows = minutes(5)
    rows[1]["values"][2] = value
    with pytest.raises(ValueError):
        aggregate_five_minute(rows)
