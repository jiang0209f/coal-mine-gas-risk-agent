"""Forecast bundle contract: training transforms, closed inputs and output semantics."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

import pytest

from app.adapters.gas_forecast import GasForecastAdapter
from app.adapters.model_bundle import ModelNotReady


def architecture():
    return {
        "version": "Fourier", "mode_select": "random", "modes": 1,
        "seq_len": 90, "label_len": 45, "pred_len": 12,
        "enc_in": 2, "dec_in": 2, "c_out": 1, "d_model": 8,
        "n_heads": 8, "e_layers": 1, "d_layers": 1, "d_ff": 16,
        "moving_avg": [3], "dropout": 0.0, "embed": "timeF", "freq": "h",
        "activation": "gelu", "output_attention": False, "seed": 2021,
        "time_feature_offset_seconds": 240, "time_feature_timezone": "+08:00",
    }


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "legacy"
    (source / "models").mkdir(parents=True)
    (source / "models" / "FEDformer.py").write_text("# synthetic source marker\n")
    manifest = {
        "kind": "gas_forecast", "model_id": "synthetic-fedformer", "version": "test-v1",
        "aggregation_seconds": 300, "input_window": 90, "horizon_seconds": 3600,
        "unit": "%", "feature_names": ["gas_lag", "OT"],
        "residual_feature_names": ["t2_mean", "machine_speed"],
        "residual_target": "physical_error", "residual_selection_mode": "v1",
        "legacy_source_root": str(source), "architecture": architecture(),
        "training_range": {"start_at": "2021-01-01T00:00:00+08:00", "end_at": "2022-01-01T00:00:00+08:00"},
        "training_data": [{"sha256": "1" * 64, "source_id": "synthetic-training"}],
        "decision_offset": {"true_threshold": 0.8, "best_model_threshold": 0.7},
        "artifacts": {},
    }
    write_artifact(tmp_path, manifest, "checkpoint", b"synthetic checkpoint")
    write_artifact(tmp_path, manifest, "scaler", json.dumps({
        "feature_names": ["gas_lag", "OT"], "mean": [1.0, 0.2], "scale": [2.0, 0.1],
    }).encode())
    for index in range(1, 13):
        write_artifact(tmp_path, manifest, f"residual_{index:02}", b"synthetic booster")
    save_manifest(tmp_path, manifest)
    return tmp_path, manifest


def save_manifest(root, manifest):
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_artifact(root, manifest, name, content):
    suffix = ".pth" if name == "checkpoint" else ".json" if name == "scaler" else ".txt"
    path = root / f"{name}{suffix}"
    path.write_bytes(content)
    manifest["artifacts"][name] = {"path": path.name, "sha256": hashlib.sha256(content).hexdigest()}


def request():
    cutoff = datetime(2022, 2, 5, 11, 0, tzinfo=timezone.utc)
    return {
        "as_of": cutoff.isoformat(), "unit": "%",
        "feature_names": ["gas_lag", "OT"],
        "times": [(cutoff - timedelta(minutes=(90 - index) * 5)).isoformat() for index in range(90)],
        "values": [[3.0, 0.4] for _ in range(90)],
        "residual_feature_names": ["t2_mean", "machine_speed"],
        "residual_values": [0.3, 2.0],
    }


class Booster:
    def __init__(self, value):
        self.value = value

    def predict(self, values):
        assert values == [[0.3, 2.0]]
        return [self.value]


def fake_predictors(adapter, monkeypatch, base=None):
    observed = {}

    def predict_base(values, times, cutoff):
        observed.update(values=values, times=times, cutoff=cutoff)
        return base if base is not None else [float(i) for i in range(1, 13)]

    monkeypatch.setattr(adapter, "_load_predictors", lambda artifacts: (predict_base, [Booster(0.05) for _ in range(12)]))
    return observed


def test_forecast_reuses_training_scaler_and_keeps_decision_score_separate(bundle, monkeypatch):
    root, _ = bundle
    adapter = GasForecastAdapter(root)
    seen = fake_predictors(adapter, monkeypatch)
    inputs = request()
    before = deepcopy(inputs)

    result = adapter.predict(inputs)

    assert inputs == before
    assert result["status"] == "completed"
    assert result["model_id"] == "synthetic-fedformer"
    assert result["model_version"] == "test-v1"
    assert result["residual_target"] == "physical_error"
    assert result["residual_selection_mode"] == "v1"
    assert seen["values"] == [[1.0, 2.0] for _ in range(90)]
    stages = {item["prediction_stage"]: item for item in result["outputs"]}
    assert set(stages) == {"base_prediction", "residual", "residual_corrected", "decision_score"}
    assert stages["base_prediction"]["interval_values"][0]["value"] == pytest.approx(0.3)
    assert stages["residual_corrected"]["interval_values"][0]["value"] == pytest.approx(0.35)
    assert stages["decision_score"]["interval_values"][0]["value"] == pytest.approx(0.45)
    assert stages["decision_score"]["quantity_semantics"] == "decision_score"
    assert stages["decision_score"]["unit"] == "dimensionless"
    for name in ("base_prediction", "residual", "residual_corrected"):
        assert stages[name]["quantity_semantics"] == "physical_forecast"
        assert stages[name]["unit"] == "%"
    intervals = stages["residual_corrected"]["interval_values"]
    assert len(intervals) == 12
    assert datetime.fromisoformat(intervals[0]["start_at"]) == datetime.fromisoformat(inputs["as_of"])
    assert datetime.fromisoformat(intervals[-1]["end_at"]) == datetime.fromisoformat(inputs["as_of"]) + timedelta(hours=1)
    assert result["target_statistic"] == "interval_maximum"


@pytest.mark.parametrize("problem", ["short", "future", "gap", "naive", "nan", "bool", "order", "residual_order", "residual_shape", "unit", "future_truth"])
def test_invalid_or_future_inputs_are_rejected_before_loading_models(bundle, monkeypatch, problem):
    root, _ = bundle
    adapter = GasForecastAdapter(root)
    monkeypatch.setattr(adapter, "_load_predictors", lambda _: pytest.fail("invalid input loaded model"))
    inputs = request()
    if problem == "short": inputs["values"].pop()
    elif problem == "future": inputs["times"][-1] = inputs["as_of"]
    elif problem == "gap": inputs["times"][1] = inputs["times"][0]
    elif problem == "naive": inputs["as_of"] = "2022-02-05T11:00:00"
    elif problem == "nan": inputs["values"][0][0] = float("nan")
    elif problem == "bool": inputs["values"][0][0] = True
    elif problem == "order": inputs["feature_names"].reverse()
    elif problem == "residual_order": inputs["residual_feature_names"].reverse()
    elif problem == "residual_shape": inputs["residual_values"].append(1.0)
    elif problem == "unit": inputs["unit"] = "ppm"
    elif problem == "future_truth": inputs["future_values"] = [0.9] * 12
    with pytest.raises(ValueError):
        adapter.predict(inputs)


@pytest.mark.parametrize("changes", [{"scale": [2.0, 0.0]}, {"scale": [2.0, -0.1]}, {"mean": [0.0]}, {"feature_names": ["OT", "gas_lag"]}])
def test_scaler_must_match_training_feature_order_and_positive_scales(bundle, monkeypatch, changes):
    root, manifest = bundle
    scaler = {"feature_names": ["gas_lag", "OT"], "mean": [1.0, 0.2], "scale": [2.0, 0.1]}
    scaler.update(changes)
    write_artifact(root, manifest, "scaler", json.dumps(scaler).encode())
    save_manifest(root, manifest)
    adapter = GasForecastAdapter(root)
    monkeypatch.setattr(adapter, "_load_predictors", lambda _: pytest.fail("invalid scaler loaded model"))
    with pytest.raises(ValueError):
        adapter.predict(request())


def test_missing_artifacts_report_not_ready_and_do_not_generate_forecasts(bundle):
    root, manifest = bundle
    manifest["artifacts"].pop("checkpoint")
    save_manifest(root, manifest)
    adapter = GasForecastAdapter(root)
    report = adapter.inspect()
    assert report["status"] == "not_ready"
    assert any("checkpoint" in item for item in report["missing_fields"])
    with pytest.raises(ModelNotReady):
        adapter.predict(request())


def test_model_trained_after_replay_clock_is_rejected(bundle, monkeypatch):
    root, manifest = bundle
    manifest["training_range"]["end_at"] = "2022-02-06T00:00:00+08:00"
    save_manifest(root, manifest)
    adapter = GasForecastAdapter(root)
    monkeypatch.setattr(adapter, "_load_predictors", lambda _: pytest.fail("future-trained model loaded"))
    with pytest.raises(ValueError, match="训练"):
        adapter.predict(request())


def test_training_source_hashes_are_validated_and_retained(bundle, monkeypatch):
    root, manifest = bundle
    adapter = GasForecastAdapter(root)
    fake_predictors(adapter, monkeypatch)
    assert adapter.predict(request())["training_data"] == manifest["training_data"]
    manifest["training_data"][0]["sha256"] = "not-a-sha256"
    save_manifest(root, manifest)
    with pytest.raises(ValueError):
        GasForecastAdapter(root).predict(request())


@pytest.mark.parametrize("base", [[float("nan")] * 12, [True] * 12, [0.1] * 11])
def test_malformed_model_outputs_are_rejected(bundle, monkeypatch, base):
    root, _ = bundle
    adapter = GasForecastAdapter(root)
    fake_predictors(adapter, monkeypatch, base)
    with pytest.raises(ValueError):
        adapter.predict(request())


def test_shorter_horizon_returns_only_requested_closed_intervals(bundle, monkeypatch):
    root, _ = bundle
    adapter = GasForecastAdapter(root)
    fake_predictors(adapter, monkeypatch)
    inputs = request()
    inputs["horizon_seconds"] = 600
    result = adapter.predict(inputs)
    assert all(len(stage["interval_values"]) == 2 for stage in result["outputs"])


@pytest.mark.parametrize("bad", [float("nan"), True, 1.0, [1.0, 2.0]])
def test_invalid_residual_output_is_rejected(bundle, monkeypatch, bad):
    root, _ = bundle
    adapter = GasForecastAdapter(root)

    class InvalidBooster:
        def predict(self, values):
            return bad if isinstance(bad, list) or bad == 1.0 else [bad]

    monkeypatch.setattr(adapter, "_load_predictors", lambda _: (lambda *_: [0.0] * 12, [InvalidBooster()] * 12))
    with pytest.raises(ValueError, match="残差"):
        adapter.predict(request())


def test_missing_training_time_encoding_is_reported_before_runtime_loading(bundle):
    root, manifest = bundle
    manifest["architecture"].pop("time_feature_offset_seconds")
    save_manifest(root, manifest)
    report = GasForecastAdapter(root).inspect()
    assert report["status"] == "not_ready"
    assert "时间特征" in report["missing_fields"][0]


@pytest.mark.parametrize("failure", [ValueError("invalid Booster feature count"), RuntimeError("checkpoint dimensions mismatch")])
def test_inspect_rejects_hash_valid_but_unloadable_model_artifacts(bundle, monkeypatch, failure):
    root, _ = bundle
    adapter = GasForecastAdapter(root)
    monkeypatch.setattr("app.adapters.gas_forecast.importlib.util.find_spec", lambda _: object())

    def invalid_predictors(artifacts):
        raise failure

    monkeypatch.setattr(adapter, "_load_predictors", invalid_predictors)
    report = adapter.inspect()

    assert report["status"] == "not_ready"
    assert report["error_code"] == "invalid_model_artifacts"
    assert "model_artifacts" in report["missing_fields"]


def test_inspect_reports_ready_only_after_loading_model_and_all_boosters(bundle, monkeypatch):
    root, _ = bundle
    adapter = GasForecastAdapter(root)
    monkeypatch.setattr("app.adapters.gas_forecast.importlib.util.find_spec", lambda _: object())
    loaded = []

    def valid_predictors(artifacts):
        loaded.extend(artifacts)
        return object(), [object()] * 12

    monkeypatch.setattr(adapter, "_load_predictors", valid_predictors)
    assert adapter.inspect()["status"] == "ready"
    assert len(loaded) == 14


def test_decoder_history_must_match_legacy_fourier_half_window(bundle, monkeypatch):
    root, manifest = bundle
    manifest["architecture"]["label_len"] = 1
    save_manifest(root, manifest)
    adapter = GasForecastAdapter(root)
    fake_predictors(adapter, monkeypatch)
    with pytest.raises(ValueError, match="架构窗口"):
        adapter.predict(request())


@pytest.mark.parametrize("changes", [
    {"residual_target": "threshold_shifted_error"},
    {"residual_selection_mode": "v3"},
    {"residual_target": None},
    {"residual_selection_mode": None},
])
def test_residual_artifacts_must_use_original_unshifted_physical_target(bundle, monkeypatch, changes):
    root, manifest = bundle
    manifest.update(changes)
    save_manifest(root, manifest)
    adapter = GasForecastAdapter(root)
    monkeypatch.setattr(adapter, "_load_predictors", lambda _: pytest.fail("biased residual artifacts loaded"))
    with pytest.raises(ValueError, match="残差目标"):
        adapter.predict(request())


def test_optional_native_fedformer_and_lightgbm_checkpoint_roundtrip(bundle, monkeypatch):
    """An explicitly supplied local source enables real CPU checkpoint/Booster inference."""
    source = os.environ.get("COAL_FORECAST_LEGACY_ROOT")
    if not source:
        pytest.skip("set COAL_FORECAST_LEGACY_ROOT for native legacy-model smoke")
    torch = pytest.importorskip("torch")
    numpy = pytest.importorskip("numpy")
    lgb = pytest.importorskip("lightgbm")
    pytest.importorskip("scipy")
    pytest.importorskip("einops")
    pytest.importorskip("sympy")
    root, manifest = bundle
    manifest["legacy_source_root"] = source
    save_manifest(root, manifest)
    adapter = GasForecastAdapter(root)
    model = adapter._build_model(torch, numpy)
    checkpoint = root / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint)
    write_artifact(root, manifest, "checkpoint", checkpoint.read_bytes())
    samples = numpy.array([[0.1, 1], [0.3, 2], [0.5, 3], [0.7, 4]], dtype=float)
    trained = lgb.train({"objective": "regression", "verbose": -1, "min_data_in_leaf": 1, "num_threads": 1}, lgb.Dataset(samples, label=[0.05] * 4), num_boost_round=1)
    for index in range(1, 13):
        write_artifact(root, manifest, f"residual_{index:02}", trained.model_to_string().encode())
    save_manifest(root, manifest)

    result = GasForecastAdapter(root).predict(request())

    assert result["status"] == "completed"
    stages = {item["prediction_stage"]: item for item in result["outputs"]}
    base = stages["base_prediction"]["interval_values"]
    corrected = stages["residual_corrected"]["interval_values"]
    assert len(base) == 12
    for raw, final in zip(base, corrected):
        assert final["value"] == pytest.approx(raw["value"] + 0.05)
    from app.adapters.model_runtime import ModelRuntime
    import sys
    catalog = root / "catalog.json"
    catalog.write_text(json.dumps({"python_executable": sys.executable, "models": [
        {"bundle_dir": str(root), "source_id": "synthetic", "face_id": "31002"}
    ]}), encoding="utf-8")
    runtime = ModelRuntime(catalog)
    assert runtime.list_packages()[0]["status"] == "ready"
    isolated = runtime.infer("synthetic-fedformer", "test-v1", request())
    assert isolated["status"] == "completed"
    isolated_values = next(s for s in isolated["outputs"] if s["prediction_stage"] == "residual_corrected")["interval_values"]
    assert [v["value"] for v in isolated_values] == pytest.approx([v["value"] for v in corrected], abs=1e-6)
