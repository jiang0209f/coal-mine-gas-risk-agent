"""Chronological training and frozen cluster references for attribution."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import csv
import json

import pytest

from app.adapters.gas_attribution import GasAttributionAdapter, FEATURE_NAMES
from app.scripts.train_attribution import (
    build_samples, cause_map_from_centers, extract_features, fit_reference,
    label_samples, read_minute_rows, split_name, train_bundle,
)


def minute_rows(start="2022-01-04T00:00:00+08:00", count=130):
    origin = datetime.fromisoformat(start)
    return [{"event_time": (origin + timedelta(minutes=i)).isoformat(),
             "values": [.8, 1.0, 2.0, .4, 30.0, 40.0], "row_number": i + 2}
            for i in range(count)]


def test_sample_window_ends_five_minutes_before_target_and_contains_61_minutes():
    rows = minute_rows()
    samples = build_samples(rows)
    first = samples[0]
    assert first["target_time"] == rows[65]["event_time"]
    assert first["as_of"] == rows[60]["event_time"]
    assert first["times"] == [r["event_time"] for r in rows[:61]]
    assert first["history"] == [r["values"] for r in rows[:61]]
    assert len(first["history"]) == 61


def test_future_target_changes_labels_but_never_input_features():
    rows = minute_rows()
    samples = build_samples(rows)
    changed = deepcopy(rows)
    changed[65]["values"] = [.9, 100, 200, 300, 400, 500]
    next_samples = build_samples(changed)
    assert samples[0]["history"] == next_samples[0]["history"]
    assert extract_features(samples[0]["history"]) == extract_features(next_samples[0]["history"])
    assert samples[0]["target_values"] != next_samples[0]["target_values"]


def test_missing_minute_invalidates_history_instead_of_interpolating():
    rows = minute_rows()
    del rows[20]
    samples = build_samples(rows)
    assert samples[0]["times"][0] == rows[20]["event_time"]
    assert all(len(s["history"]) == 61 for s in samples)


@pytest.mark.parametrize("timestamp,expected", [
    ("2022-01-23T23:59:00+08:00", "train"),
    ("2022-01-24T00:00:00+08:00", "validation"),
    ("2022-02-01T00:00:00+08:00", "test"),
    ("2022-02-28T23:59:00+08:00", "test"),
])
def test_target_time_determines_split(timestamp, expected):
    assert split_name(timestamp) == expected


def test_2023_other_working_face_is_rejected():
    with pytest.raises(ValueError):
        split_name("2023-09-01T00:00:00+08:00")


def test_feature_extraction_calls_the_runtime_legacy_implementation(monkeypatch):
    history = [row["values"] for row in minute_rows(count=61)]
    recorded = []
    def runtime_features(self, values):
        recorded.append((self.manifest["preprocessing"], values))
        return [[7.0] * 24]
    monkeypatch.setattr(GasAttributionAdapter, "_features", runtime_features)
    assert extract_features(history) == [7.0] * 24
    assert recorded == [("absolute", history)]


def test_original_center_rank_rules_and_tie_order_are_preserved():
    centers = [[float(i)] * 6 for i in range(5)]
    causes = cause_map_from_centers(centers)
    assert set(causes["4"]) == {"割煤速度快", "邻近层瓦斯涌出", "采空区瓦斯涌出"}
    assert causes["0"] == ["通风系统异常"]
    assert causes["2"] == ["其他原因"]
    tied = cause_map_from_centers([[0.] * 6 for _ in range(5)])
    assert set(tied["4"]) == set(causes["4"])


def test_csv_localizes_original_naive_dates_and_preserves_column_order(tmp_path):
    path = tmp_path / "input.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["date", *FEATURE_NAMES])
        writer.writerow(["2022-01-04 12:00:00", .8, 1, 2, .3, 4, 5])
    rows = read_minute_rows(path)
    assert datetime.fromisoformat(rows[0]["event_time"]).utcoffset() == timedelta(hours=8)
    assert rows[0]["values"] == [.8, 1., 2., .3, 4., 5.]
    assert rows[0]["row_number"] == 2


def test_frozen_reference_does_not_refit_on_validation_or_test_extremes():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    rows = minute_rows(count=180)
    for i, row in enumerate(rows):
        row["values"] = [.8 + (i % 5) * .01, 1+i % 5, 2+i % 7, .4+i % 3, 30+i % 11, 40+i % 13]
    reference = fit_reference(rows)
    before = (reference["scaler"].data_min_.copy(), reference["scaler"].data_max_.copy(),
              reference["kmeans"].cluster_centers_.copy())
    sample = build_samples(minute_rows())[0]
    sample["target_values"] = [10000.] * 6
    label_samples([sample], reference)
    assert np.array_equal(before[0], reference["scaler"].data_min_)
    assert np.array_equal(before[1], reference["scaler"].data_max_)
    assert np.array_equal(before[2], reference["kmeans"].cluster_centers_)


def test_training_and_safe_export_roundtrip_use_real_dependencies(tmp_path):
    pytest.importorskip("skops.io")
    pytest.importorskip("sklearn")
    def fixture_file(name, start):
        path = tmp_path / name
        rows = minute_rows(start, count=700)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["date", *FEATURE_NAMES])
            for i, row in enumerate(rows):
                group = (i // 20) % 5
                values = [.7 + group * .03, 1 + group, 2 + group, .1 + group, 10 + group * 5, 20 + group * 3]
                writer.writerow([datetime.fromisoformat(row["event_time"]).strftime("%Y-%m-%d %H:%M:%S"), *values])
        return path
    january = fixture_file("january.csv", "2022-01-04T00:00:00+08:00")
    february = fixture_file("february.csv", "2022-02-05T00:00:00+08:00")
    # Empty validation is allowed for a synthetic serialization check.
    output, reports = tmp_path / "trained", tmp_path / "reports"
    report = train_bundle(january, february, output, reports)
    loaded = GasAttributionAdapter(output)
    assert loaded.inspect()["status"] == "ready"
    assert report["reference_label_semantics"] == "frozen_training_kmeans_membership"
    assert report["partitions"]["test"]["sample_count"] > 0
    assert json.loads((reports / "report.json").read_text(encoding="utf-8"))["model_version"] == "trained-20261103-v1"
    sample = build_samples(read_minute_rows(february))[0]
    request = {"as_of": sample["as_of"], "times": sample["times"], "values": sample["history"], "feature_names": FEATURE_NAMES}
    result = loaded.predict(request)
    assert result["status"] == "completed"
    assert report["serialization_max_probability_difference"] < 1e-12
