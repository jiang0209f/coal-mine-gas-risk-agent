"""Training splits respect complete future labels and training-only scaling."""

from datetime import datetime, timedelta, timezone
import math

import pytest

from app.scripts.train_forecast import build_windows, select_split, fit_scaler, forecast_metrics, time_marks, read_minute_csv, validation_assessment


def buckets(count=110):
    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    return [dict(event_time=(start + timedelta(minutes=5*i)).isoformat(),
                 bucket_end=(start + timedelta(minutes=5*(i+1))).isoformat(),
                 values=[float(i + j) for j in range(6)]) for i in range(count)]


def test_windows_use_only_past_features_and_next_twelve_targets():
    windows = build_windows(buckets())
    assert len(windows) == 9
    first = windows[0]
    assert len(first["values"]) == 90
    assert first["values"][-1][-1] == 94.0
    assert first["targets"] == [float(i+5) for i in range(90, 102)]
    assert first["origin"] == datetime(2022, 1, 1, 7, 30, tzinfo=timezone.utc)
    assert first["target_end"] == datetime(2022, 1, 1, 8, 30, tzinfo=timezone.utc)


def test_gap_windows_are_excluded_without_interpolation():
    rows = buckets(220)
    rows.pop(100)
    windows = build_windows(rows)
    assert windows
    for sample in windows:
        stamp = [datetime.fromisoformat(t) for t in sample["times"]]
        assert all(b-a == timedelta(seconds=300) for a,b in zip(stamp, stamp[1:]))
        assert not (stamp[0] < datetime(2022, 1, 1, 8, 20, tzinfo=timezone.utc) < sample["target_end"])


def test_partition_purges_targets_crossing_boundary():
    rows = build_windows(buckets())
    start = rows[0]["origin"]
    end = rows[0]["target_end"]
    selected = select_split(rows, start, end)
    assert len(selected) == 1
    assert selected[0]["target_end"] <= end
    assert select_split(rows, start+timedelta(seconds=1), end) == []


def test_scaler_ignores_validation_values_after_cutoff():
    rows = buckets(4)
    cutoff = datetime.fromisoformat(rows[2]["event_time"])
    mean, scale = fit_scaler(rows, cutoff)
    rows[2]["values"] = [1e9] * 6
    rows[3]["values"] = [-1e9] * 6
    assert fit_scaler(rows, cutoff) == (mean, scale)
    assert mean[0] == .5
    assert scale[0] == .5


def test_scaler_rejects_no_training_data_and_nonfinite_training_values():
    with pytest.raises(ValueError):
        fit_scaler([], datetime.now(timezone.utc))
    rows = buckets(2)
    rows[0]["values"][0] = math.nan
    with pytest.raises(ValueError):
        fit_scaler(rows, datetime.fromisoformat(rows[-1]["bucket_end"]))


def test_metrics_keep_interval_and_window_counts_separate():
    true = [[.1,.8],[.7,.2]]
    pred = [[.2,.6],[.8,.1]]
    result = forecast_metrics(true, pred, threshold=.65)
    assert result["interval_pairs"] == 4
    assert result["window_pairs"] == 2
    assert result["mae"] == pytest.approx(.125)
    assert result["interval_detection"] == dict(tp=1, fp=0, tn=2, fn=1)
    assert result["window_detection"] == dict(tp=1, fp=0, tn=0, fn=1)
    assert result["per_lead"][0]["mae"] == pytest.approx(.1)


def test_time_marks_use_training_timezone_and_last_minute_offset():
    result = time_marks([datetime(2021,12,31,15,56,tzinfo=timezone.utc)])
    assert result[0][0] == -.5
    assert result[0][2] == -.5
    assert result[0][3] == -.5


def test_training_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        forecast_metrics([[.2,.3]], [[.2]])


def test_csv_clock_and_signed_features_are_preserved(tmp_path):
    from app.adapters.model_features import CHANNEL_NAMES
    file = tmp_path / "minute.csv"
    file.write_text(','.join(["date",*CHANNEL_NAMES])+"\n2022-01-04 11:20:00,0.2,1.5,-2,0.1,10,12\n",encoding="utf-8")
    result = read_minute_csv(file)
    assert datetime.fromisoformat(result[0]["event_time"]).utcoffset() == timedelta(hours=8)
    assert result[0]["values"][2] == -2


def test_csv_gap_is_rejected_without_filling(tmp_path):
    from app.adapters.model_features import CHANNEL_NAMES
    file = tmp_path / "minute.csv"
    file.write_text(','.join(["date",*CHANNEL_NAMES])+"\n2022-01-04 11:20:00,0.2,1.5,-2,0.1,10,12\n2022-01-04 11:22:00,0.2,1.5,-2,0.1,10,12\n",encoding="utf-8")
    with pytest.raises(ValueError):
        read_minute_csv(file)


def test_default_candidate_uses_validation_mae_and_ignores_holdout_results():
    metrics = {"residual_validation": {"persistence": {"mae": .1,"window_detection": {}},
               "residual_corrected": {"mae": .2,"window_detection": {}}},
               "test": {"persistence": {"mae": .9},"residual_corrected": {"mae": .01}}}
    result = validation_assessment(metrics)
    assert result["default_candidate"] == "persistence"
    metrics["test"]["residual_corrected"]["mae"] = 100
    assert validation_assessment(metrics) == result
