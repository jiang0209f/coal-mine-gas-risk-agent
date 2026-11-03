"""Six-channel KMeans/SVC candidate-attribution bundle and pure inference."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from math import fsum, isclose, isfinite
from numbers import Integral, Real
from pathlib import Path
import re

from app.adapters.model_bundle import ModelBundle, ModelNotReady, read_json

__all__ = ["GasAttributionAdapter", "export_bundle"]

FEATURE_NAMES = ["T0瓦斯浓度", "风速", "采煤机速度", "采煤机瓦斯浓度", "左截割电机电流", "右截割电机电流"]
_FEATURE_VERSION = "legacy_row_stats_v1"


def _time(value):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            raise ValueError
        return parsed.astimezone(timezone.utc)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("溯源时间须为带时区的ISO时间") from error


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("溯源输入及模型数值须为有限数值")
    return float(value)


def _vector(values, length):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError("溯源模型元数据维度无效")
    return [_number(value) for value in values]


def _validate_manifest(manifest):
    if manifest.get("feature_names") != FEATURE_NAMES:
        raise ValueError("溯源模型须按规定顺序声明六路特征")
    if manifest.get("feature_version") != _FEATURE_VERSION:
        raise ValueError("仅支持legacy_row_stats_v1特征版本")
    if manifest.get("preprocessing") not in ("identity", "absolute"):
        raise ValueError("模型须明确声明identity或absolute预处理")
    if type(manifest.get("aggregation_seconds")) is not int or manifest["aggregation_seconds"] != 60:
        raise ValueError("溯源模型采样间隔须为60秒")
    if type(manifest.get("input_window")) is not int or manifest["input_window"] != 61:
        raise ValueError("旧版溯源模型包含两端的历史窗口须为61行")
    horizon = manifest.get("horizon_seconds")
    if type(horizon) is not int or not 300 <= horizon <= 3600 or horizon % 300:
        raise ValueError("溯源提前量须为300至3600秒且为300秒的整数倍")


def _validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("溯源训练元数据须为对象")
    causes = metadata.get("cause_map")
    if not isinstance(causes, dict) or set(causes) != {str(i) for i in range(5)}:
        raise ValueError("训练元数据须提供五个聚类的候选原因")
    for values in causes.values():
        if not isinstance(values, list) or not values or any(not isinstance(s, str) or not s.strip() for s in values):
            raise ValueError("候选原因须为非空文字列表")
    centers = metadata.get("cluster_centers")
    if not isinstance(centers, list) or len(centers) != 5:
        raise ValueError("训练元数据须提供五个六维聚类中心")
    for center in centers:
        _vector(center, 6)
    scaler = metadata.get("scaler")
    if not isinstance(scaler, dict):
        raise ValueError("训练元数据须提供固定缩放器参数")
    for field in ("data_min", "data_max", "scale", "min"):
        _vector(scaler.get(field), 6)
    if any(high < low for low, high in zip(scaler["data_min"], scaler["data_max"])) or any(v <= 0 for v in scaler["scale"]):
        raise ValueError("固定缩放器参数无效")
    hashes = metadata.get("source_hashes")
    if not isinstance(hashes, dict) or not hashes or any(
        not isinstance(name, str) or not name or not isinstance(digest, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None for name, digest in hashes.items()
    ):
        raise ValueError("训练元数据须提供源文件SHA256")
    training_range = metadata.get("training_range")
    if not isinstance(training_range, dict):
        raise ValueError("训练元数据须提供训练时间范围")
    if _time(training_range.get("start_at")) >= _time(training_range.get("end_at")):
        raise ValueError("训练时间范围无效")


def _dependencies():
    try:
        sio = importlib.import_module("skops.io")
        svm = importlib.import_module("sklearn.svm")
        return sio, svm.SVC
    except ImportError as error:
        raise ModelNotReady(["dependency:skops", "dependency:scikit-learn"]) from error


def _classifier_contract(classifier):
    classes = list(getattr(classifier, "classes_", []))
    if any(isinstance(c, bool) or not isinstance(c, Integral) for c in classes) or sorted(classes) != list(range(5)):
        raise ValueError("SVC须保留0至4的五个聚类标签")
    if getattr(classifier, "n_features_in_", None) != 24:
        raise ValueError("SVC输入须为24维旧版统计特征")
    return [int(c) for c in classes]


def _svc_contract(classifier, svc_type):
    if type(classifier) is not svc_type:
        raise ValueError("溯源产物仅支持sklearn.svm.SVC")
    params = classifier.get_params()
    if params.get("kernel") != "rbf" or params.get("C") != 100 or params.get("gamma") != .01 or params.get("probability") is not True:
        raise ValueError("SVC参数与原溯源模型不一致")
    _classifier_contract(classifier)


class GasAttributionAdapter:
    def __init__(self, bundle_dir):
        self.bundle = ModelBundle(bundle_dir, "gas_attribution")
        self.manifest = self.bundle.manifest
        _validate_manifest(self.manifest)

    def _artifacts(self):
        paths = self.bundle.require_artifacts(["classifier", "training_metadata"])
        if paths["classifier"].suffix.lower() != ".skops":
            raise ValueError("SVC产物须使用skops格式")
        metadata = read_json(paths["training_metadata"])
        _validate_metadata(metadata)
        return paths, metadata

    def _load_classifier(self, path):
        sio, svc_type = _dependencies()
        if sio.get_untrusted_types(file=path):
            raise ValueError("SVC产物包含未批准的类型")
        classifier = sio.load(path, trusted=[])
        _svc_contract(classifier, svc_type)
        return classifier

    def inspect(self):
        paths, metadata = self._artifacts()
        _classifier_contract(self._load_classifier(paths["classifier"]))
        return {
            "status": "ready", "kind": "gas_attribution", "model_id": self.manifest["model_id"],
            "version": self.manifest["version"], "model_version": self.manifest["version"], "feature_version": _FEATURE_VERSION,
            "feature_names": deepcopy(FEATURE_NAMES), "input_shape": [61, 6],
            "preprocessing": self.manifest["preprocessing"], "aggregation_seconds": 60,
            "horizon_seconds": self.manifest["horizon_seconds"],
            "window_policy": "inclusive_endpoints", "probability_semantics": "cluster_membership",
            "causal_status": "candidate_attribution", "training_range": deepcopy(metadata["training_range"]),
        }

    def _features(self, values):
        features = []
        try:
            for row in list(reversed(values))[:6]:
                values = [abs(value) for value in row] if self.manifest["preprocessing"] == "absolute" else row
                average = fsum(values) / 6
                features.extend([average, fsum((value - average) ** 2 for value in values) / 6,
                                 max(values), min(values)])
        except OverflowError as error:
            raise ValueError("溯源统计特征数值溢出") from error
        if not all(isfinite(value) for value in features):
            raise ValueError("溯源统计特征须为有限数值")
        return [features]

    def predict(self, request):
        if not isinstance(request, dict):
            raise ValueError("溯源请求须为对象")
        missing = [name for name in ("as_of", "feature_names", "times", "values") if name not in request]
        if missing:
            raise ModelNotReady(missing)
        if request["feature_names"] != FEATURE_NAMES:
            raise ModelNotReady(["six_channel_feature_order"])
        values, times = request["values"], request["times"]
        if not isinstance(values, list) or len(values) != 61 or any(not isinstance(row, list) or len(row) != 6 for row in values):
            raise ModelNotReady(["complete_six_channel_history"])
        for row in values:
            for value in row:
                _number(value)
        if not isinstance(times, list) or len(times) != 61:
            raise ModelNotReady(["history_times"])
        as_of = _time(request["as_of"])
        timestamps = [_time(value) for value in times]
        if timestamps[-1] != as_of or any(b - a != timedelta(seconds=60) for a, b in zip(timestamps, timestamps[1:])):
            raise ModelNotReady(["continuous_60_second_history_ending_at_as_of"])
        paths, metadata = self._artifacts()
        if _time(metadata["training_range"]["end_at"]) > as_of:
            raise ModelNotReady(["training_cutoff_not_after_as_of"])
        classifier = self._load_classifier(paths["classifier"])
        classes = _classifier_contract(classifier)
        probabilities = classifier.predict_proba(self._features(values))
        if len(probabilities) != 1 or len(probabilities[0]) != 5:
            raise ValueError("SVC概率输出维度无效")
        probabilities = [_number(value) for value in probabilities[0]]
        if any(value < 0 or value > 1 for value in probabilities) or not isclose(fsum(probabilities), 1, abs_tol=1e-6):
            raise ValueError("SVC概率输出无效")
        ranked = sorted(zip(classes, probabilities), key=lambda item: (-item[1], item[0]))
        cut = max(range(1, 5), key=lambda i: ranked[i - 1][1] - ranked[i][1])
        selected = [cluster for cluster, _ in ranked[:cut]]
        causes = list(dict.fromkeys(cause for cluster in selected for cause in metadata["cause_map"][str(cluster)]))
        return {
            "status": "completed", "model_id": self.manifest["model_id"],
            "version": self.manifest["version"], "model_version": self.manifest["version"],
            "feature_version": _FEATURE_VERSION, "preprocessing": self.manifest["preprocessing"],
            "as_of": as_of.isoformat(), "horizon_seconds": self.manifest["horizon_seconds"],
            "target_time": (as_of + timedelta(seconds=self.manifest["horizon_seconds"])).isoformat(),
            "causal_status": "candidate_attribution", "probability_semantics": "cluster_membership",
            "cluster_probabilities": [{"cluster_id": c, "probability": p} for c, p in ranked],
            "selected_cluster_ids": selected, "candidate_causes": causes,
            "training_range": deepcopy(metadata["training_range"]),
        }


def export_bundle(classifier, cause_map, training_metadata, output_dir, *, model_id, version,
                  horizon_seconds, preprocessing):
    """Export an already fitted, reviewed SVC; this function performs no training."""
    manifest = {
        "kind": "gas_attribution", "model_id": model_id, "version": version,
        "feature_names": deepcopy(FEATURE_NAMES), "feature_version": _FEATURE_VERSION,
        "preprocessing": preprocessing, "aggregation_seconds": 60, "input_window": 61,
        "horizon_seconds": horizon_seconds,
    }
    _validate_manifest(manifest)
    if any(not isinstance(value, str) or not value.strip() for value in (model_id, version)):
        raise ValueError("模型编号与版本须为非空文字")
    metadata = deepcopy(training_metadata)
    if not isinstance(metadata, dict):
        raise ValueError("训练元数据须为对象")
    metadata["cause_map"] = deepcopy(cause_map)
    _validate_metadata(metadata)
    sio, svc_type = _dependencies()
    _svc_contract(classifier, svc_type)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValueError("模型导出目录须为空，已有版本保持不变")
    root.mkdir(parents=True, exist_ok=True)
    classifier_path, training_path = root / "classifier.skops", root / "training.json"
    sio.dump(classifier, classifier_path)
    if sio.get_untrusted_types(file=classifier_path):
        raise ValueError("导出的SVC包含未批准的类型")
    training_path.write_text(json.dumps(metadata, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    manifest["artifacts"] = {
        name: {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in {"classifier": classifier_path, "training_metadata": training_path}.items()
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    return manifest
