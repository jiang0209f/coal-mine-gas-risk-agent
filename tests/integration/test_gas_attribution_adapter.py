"""Six-channel attribution bundle and pure-inference acceptance checks."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from app.adapters.gas_attribution import GasAttributionAdapter, export_bundle
from app.adapters.model_bundle import ModelNotReady


FEATURES = ["T0瓦斯浓度", "风速", "采煤机速度", "采煤机瓦斯浓度", "左截割电机电流", "右截割电机电流"]
CAUSES = {str(i): [f"候选原因{i}"] for i in range(5)}


def training_metadata():
    return {
        "cluster_centers": [[i / 4] * 6 for i in range(5)],
        "scaler": {"data_min": [0] * 6, "data_max": [1] * 6,
                   "scale": [1] * 6, "min": [0] * 6},
        "source_hashes": {"cluster_model.py": "a" * 64},
        "training_range": {"start_at": "2022-01-01T00:00:00+08:00",
                           "end_at": "2022-02-01T00:00:00+08:00"},
    }


def bundle(tmp_path, **changes):
    root = tmp_path / "attribution"
    root.mkdir()
    classifier = root / "classifier.skops"
    classifier.write_bytes(b"mock-only classifier")
    metadata = training_metadata()
    metadata["cause_map"] = deepcopy(CAUSES)
    training = root / "training.json"
    training.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "kind": "gas_attribution", "model_id": "gas-candidates", "version": "v1",
        "feature_names": FEATURES, "feature_version": "legacy_row_stats_v1",
        "preprocessing": "absolute", "aggregation_seconds": 60, "input_window": 61,
        "horizon_seconds": 300,
        "artifacts": {
            name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in {"classifier": classifier, "training_metadata": training}.items()
        },
    }
    manifest.update(changes)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return root


def request():
    cutoff = datetime(2022, 2, 5, 12, tzinfo=timezone(timedelta(hours=8)))
    return {
        "as_of": cutoff.isoformat(), "feature_names": deepcopy(FEATURES),
        "times": [(cutoff - timedelta(minutes=60 - i)).isoformat() for i in range(61)],
        "values": [[(-1 if i % 2 else 1) * (i + 1) * (j + 1) for j in range(6)] for i in range(61)],
    }


class FakeClassifier:
    classes_ = [4, 0, 2, 1, 3]
    n_features_in_ = 24

    def __init__(self, probabilities=None):
        self.features = None
        self.probabilities = probabilities or [.55, .20, .13, .08, .04]

    def predict_proba(self, features):
        self.features = deepcopy(features)
        return [self.probabilities]


def adapter(tmp_path, monkeypatch, classifier=None, **changes):
    instance = GasAttributionAdapter(bundle(tmp_path, **changes))
    model = classifier or FakeClassifier()
    monkeypatch.setattr(instance, "_load_classifier", lambda path: model)
    return instance, model


def test_legacy_features_use_latest_six_rows_in_reverse_time_order(tmp_path, monkeypatch):
    instance, model = adapter(tmp_path, monkeypatch)
    payload = request()
    original = deepcopy(payload)

    result = instance.predict(payload)

    first = [61 * i for i in range(1, 7)]
    average = sum(first) / 6
    expected = [average, sum((v - average) ** 2 for v in first) / 6, max(first), min(first)]
    assert model.features[0][:4] == pytest.approx(expected)
    assert len(model.features[0]) == 24
    assert payload == original
    assert result["status"] == "completed"
    assert result["feature_version"] == "legacy_row_stats_v1"
    assert result["causal_status"] == "candidate_attribution"
    assert result["probability_semantics"] == "cluster_membership"
    assert result["model_id"] == "gas-candidates"
    assert result["version"] == "v1"


def test_identity_preprocessing_preserves_negative_values(tmp_path, monkeypatch):
    instance, model = adapter(tmp_path, monkeypatch, preprocessing="identity")
    payload = request()
    payload["values"][-1] = [-1, -2, -3, -4, -5, -6]
    instance.predict(payload)
    assert model.features[0][:4] == pytest.approx([-3.5, 35 / 12, -1, -6])


def test_probabilities_map_through_classifier_classes_instead_of_column_index(tmp_path, monkeypatch):
    instance, _ = adapter(tmp_path, monkeypatch)
    result = instance.predict(request())
    assert result["cluster_probabilities"][0] == {"cluster_id": 4, "probability": .55}
    assert result["selected_cluster_ids"] == [4]
    assert result["candidate_causes"] == ["候选原因4"]


def test_maximum_probability_gap_selects_two_candidate_clusters(tmp_path, monkeypatch):
    model = FakeClassifier([.4, .35, .1, .08, .07])
    instance, _ = adapter(tmp_path, monkeypatch, model)
    result = instance.predict(request())
    assert result["selected_cluster_ids"] == [4, 0]
    assert result["candidate_causes"] == ["候选原因4", "候选原因0"]


def test_inspection_exposes_exact_input_and_probability_contract(tmp_path, monkeypatch):
    instance, _ = adapter(tmp_path, monkeypatch)
    result = instance.inspect()
    assert result["status"] == "ready"
    assert result["input_shape"] == [61, 6]
    assert result["feature_names"] == FEATURES
    assert result["preprocessing"] == "absolute"
    assert result["probability_semantics"] == "cluster_membership"


@pytest.mark.parametrize("field", ["feature_names", "times", "values", "as_of"])
def test_missing_required_request_fields_are_not_ready(tmp_path, monkeypatch, field):
    instance, _ = adapter(tmp_path, monkeypatch)
    payload = request()
    del payload[field]
    with pytest.raises(ModelNotReady):
        instance.predict(payload)


@pytest.mark.parametrize("change", ["methane-only", "reordered", "short", "gap", "future"])
def test_incomplete_misaligned_or_future_history_cannot_produce_causes(tmp_path, monkeypatch, change):
    instance, _ = adapter(tmp_path, monkeypatch)
    payload = request()
    if change == "methane-only":
        payload["feature_names"] = FEATURES[:1]
        payload["values"] = [[row[0]] for row in payload["values"]]
    elif change == "reordered":
        payload["feature_names"] = list(reversed(FEATURES))
    elif change == "short":
        payload["times"].pop(0)
        payload["values"].pop(0)
    elif change == "gap":
        payload["times"][10] = payload["times"][9]
    else:
        payload["as_of"] = payload["times"][-2]
    with pytest.raises((ModelNotReady, ValueError)):
        instance.predict(payload)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5", True, None])
def test_non_numeric_or_non_finite_feature_values_are_rejected(tmp_path, monkeypatch, bad):
    instance, _ = adapter(tmp_path, monkeypatch)
    payload = request()
    payload["values"][10][2] = bad
    with pytest.raises(ValueError):
        instance.predict(payload)


def test_finite_inputs_that_overflow_statistics_are_rejected(tmp_path, monkeypatch):
    instance, model = adapter(tmp_path, monkeypatch)
    payload = request()
    payload["values"][-1] = [1e308] * 6
    with pytest.raises(ValueError):
        instance.predict(payload)
    assert model.features is None


@pytest.mark.parametrize("changes", [
    {"feature_version": "channel_stats_v2"}, {"preprocessing": "implicit"},
    {"aggregation_seconds": 30}, {"input_window": 60}, {"horizon_seconds": 301},
    {"feature_names": FEATURES[:1]},
])
def test_unsupported_manifest_contract_is_rejected(tmp_path, monkeypatch, changes):
    with pytest.raises(ValueError):
        instance, _ = adapter(tmp_path, monkeypatch, **changes)
        instance.inspect()


def test_missing_classifier_artifact_is_not_ready(tmp_path):
    root = bundle(tmp_path)
    (root / "classifier.skops").unlink()
    with pytest.raises(ModelNotReady):
        GasAttributionAdapter(root).inspect()


def test_training_end_cannot_follow_inference_clock(tmp_path, monkeypatch):
    instance, _ = adapter(tmp_path, monkeypatch)
    payload = request()
    old_as_of = datetime.fromisoformat(payload["as_of"])
    new_as_of = datetime.fromisoformat("2022-01-15T12:00:00+08:00")
    payload["as_of"] = new_as_of.isoformat()
    payload["times"] = [(datetime.fromisoformat(t) - (old_as_of - new_as_of)).isoformat() for t in payload["times"]]
    with pytest.raises(ModelNotReady):
        instance.predict(payload)


def test_invalid_classifier_probabilities_are_rejected(tmp_path, monkeypatch):
    instance, _ = adapter(tmp_path, monkeypatch, FakeClassifier([.5, .5, .5, .5, .5]))
    with pytest.raises(ValueError):
        instance.predict(request())


@pytest.mark.parametrize("field,replacement", [
    ("cause_map", {"0": ["候选原因"]}),
    ("cluster_centers", [[0] * 5 for _ in range(5)]),
    ("scaler", {"data_min": [0] * 6, "data_max": [1] * 6, "scale": [0] * 6, "min": [0] * 6}),
    ("source_hashes", {"source.py": "not-a-sha256"}),
    ("training_range", {"start_at": "2022-02-01T00:00:00+08:00", "end_at": "2022-01-01T00:00:00+08:00"}),
])
def test_inconsistent_training_metadata_is_rejected_before_inference(tmp_path, monkeypatch, field, replacement):
    root = bundle(tmp_path)
    training_path = root / "training.json"
    metadata = json.loads(training_path.read_text(encoding="utf-8"))
    metadata[field] = replacement
    training_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"]["training_metadata"]["sha256"] = hashlib.sha256(training_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    instance = GasAttributionAdapter(root)
    loaded = []
    monkeypatch.setattr(instance, "_load_classifier", lambda path: loaded.append(True))
    with pytest.raises(ValueError):
        instance.predict(request())
    assert loaded == []


@pytest.mark.parametrize("classes", [[0, 1, 2, 3], [0, 1, 2, 3, 3], ["0", "1", "2", "3", "4"]])
def test_classifier_labels_must_match_all_five_cluster_ids(tmp_path, monkeypatch, classes):
    model = FakeClassifier()
    model.classes_ = classes
    instance, _ = adapter(tmp_path, monkeypatch, model)
    with pytest.raises(ValueError):
        instance.predict(request())


@pytest.mark.parametrize("probabilities", [[1, 0, 0, 0], [float("nan"), 0, 0, 0, 1], [-.1, .2, .3, .3, .3]])
def test_probability_shape_and_bounds_are_validated(tmp_path, monkeypatch, probabilities):
    instance, _ = adapter(tmp_path, monkeypatch, FakeClassifier(probabilities))
    with pytest.raises(ValueError):
        instance.predict(request())


def test_loader_rejects_unknown_types_before_constructing_model(tmp_path, monkeypatch):
    from app.adapters import gas_attribution

    loaded = []
    io = SimpleNamespace(get_untrusted_types=lambda **kwargs: ["unreviewed.CustomModel"],
                         load=lambda *args, **kwargs: loaded.append(True))
    monkeypatch.setattr(gas_attribution, "_dependencies", lambda: (io, object))
    instance = GasAttributionAdapter(bundle(tmp_path))
    with pytest.raises(ValueError, match="未批准"):
        instance.inspect()
    assert loaded == []


def test_loader_never_automatically_trusts_bundle_types(tmp_path, monkeypatch):
    from app.adapters import gas_attribution

    calls = []

    class ReviewedClassifier(FakeClassifier):
        def get_params(self):
            return {"kernel": "rbf", "C": 100, "gamma": .01, "probability": True}

    def load(path, *, trusted):
        calls.append(trusted)
        return ReviewedClassifier()

    io = SimpleNamespace(get_untrusted_types=lambda **kwargs: [], load=load)
    monkeypatch.setattr(gas_attribution, "_dependencies", lambda: (io, ReviewedClassifier))
    instance = GasAttributionAdapter(bundle(tmp_path))
    assert instance.inspect()["status"] == "ready"
    assert calls == [[]]


def test_missing_optional_dependencies_is_not_ready(tmp_path, monkeypatch):
    from app.adapters import gas_attribution

    def absent(name):
        raise ImportError("not installed")

    monkeypatch.setattr(gas_attribution.importlib, "import_module", absent)
    with pytest.raises(ModelNotReady):
        GasAttributionAdapter(bundle(tmp_path)).inspect()


def test_pickle_suffix_is_rejected_before_loading(tmp_path, monkeypatch):
    root = bundle(tmp_path)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    original = root / "classifier.skops"
    original.rename(root / "classifier.pkl")
    manifest["artifacts"]["classifier"]["path"] = "classifier.pkl"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    instance = GasAttributionAdapter(root)
    loaded = []
    monkeypatch.setattr(instance, "_load_classifier", lambda path: loaded.append(True))
    with pytest.raises(ValueError):
        instance.inspect()
    assert loaded == []


def test_export_and_safe_svc_roundtrip_preserve_probabilities(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("skops.io")
    svm = pytest.importorskip("sklearn.svm")
    rng = np.random.default_rng(7)
    x = rng.normal(size=(50, 24))
    y = np.repeat(np.arange(5), 10)
    classifier = svm.SVC(C=100, gamma=.01, kernel="rbf", probability=True, random_state=0).fit(x, y)
    root = tmp_path / "exported"

    exported = export_bundle(classifier, CAUSES, training_metadata(), root,
                             model_id="roundtrip", version="test-v1", horizon_seconds=300,
                             preprocessing="absolute")
    instance = GasAttributionAdapter(root)
    result = instance.predict(request())

    assert exported["kind"] == "gas_attribution"
    assert result["status"] == "completed"
    expected = classifier.predict_proba(instance._features(request()["values"]))[0]
    by_class = {int(c): float(p) for c, p in zip(classifier.classes_, expected)}
    assert {p["cluster_id"]: p["probability"] for p in result["cluster_probabilities"]} == pytest.approx(by_class)
    assert instance.inspect()["status"] == "ready"
    from app.adapters.model_runtime import ModelRuntime
    import sys
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"python_executable": sys.executable, "models": [
        {"bundle_dir": str(root), "source_id": "synthetic", "face_id": "31002"}
    ]}), encoding="utf-8")
    runtime = ModelRuntime(catalog)
    assert runtime.list_packages()[0]["status"] == "ready"
    isolated = runtime.infer("roundtrip", "test-v1", request())
    assert isolated["status"] == "completed"
    assert isolated["cluster_probabilities"] == result["cluster_probabilities"]
