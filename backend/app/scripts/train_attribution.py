"""Train the versioned six-channel KMeans/SVC model on chronological partitions."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from app.adapters.gas_attribution import GasAttributionAdapter, FEATURE_NAMES, export_bundle

LOCAL_ZONE = timezone(timedelta(hours=8))
FIT_END = datetime(2022, 1, 24, tzinfo=LOCAL_ZONE)
VALIDATION_END = datetime(2022, 2, 1, tzinfo=LOCAL_ZONE)
TEST_END = datetime(2022, 3, 1, tzinfo=LOCAL_ZONE)
MODEL_ID = "kmeans-svc-31002"
MODEL_VERSION = "trained-20261103-v1"


def _time(value):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("训练样本时间须带时区")
    return parsed.astimezone(LOCAL_ZONE)


def split_name(timestamp):
    timestamp = _time(timestamp)
    if not datetime(2022, 1, 4, tzinfo=LOCAL_ZONE) <= timestamp < TEST_END:
        raise ValueError("训练只接受31002的2022年1至2月数据")
    if timestamp < FIT_END:
        return "train"
    return "validation" if timestamp < VALIDATION_END else "test"


def read_minute_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["date", *FEATURE_NAMES]:
            raise ValueError("训练CSV的六通道列名及顺序不一致")
        previous = None
        for number, row in enumerate(reader, 2):
            timestamp = datetime.fromisoformat(row["date"])
            if timestamp.utcoffset() is None:
                timestamp = timestamp.replace(tzinfo=LOCAL_ZONE)
            timestamp = timestamp.astimezone(LOCAL_ZONE)
            split_name(timestamp)
            if timestamp.second or timestamp.microsecond or (previous is not None and timestamp <= previous):
                raise ValueError("训练数据须为升序、无重复的分钟时刻")
            values = [float(row[name]) for name in FEATURE_NAMES]
            if any(not isfinite(value) for value in values):
                raise ValueError("训练数据含非有限值")
            rows.append(dict(event_time=timestamp.isoformat(), values=values, row_number=number))
            previous = timestamp
    if not rows:
        raise ValueError("训练CSV为空")
    return rows


def build_samples(rows, horizon_seconds=300, threshold=.65):
    if type(horizon_seconds) is not int or not 300 <= horizon_seconds <= 3600 or horizon_seconds % 300:
        raise ValueError("提前量须为300至3600秒的300秒整数倍")
    lookup = {}
    for row in rows:
        timestamp = _time(row["event_time"])
        if timestamp in lookup:
            raise ValueError("训练数据存在重复时刻")
        lookup[timestamp] = row
    samples = []
    for target in sorted(lookup):
        row = lookup[target]
        if row["values"][0] < threshold:
            continue
        origin = target - timedelta(seconds=horizon_seconds)
        times = [origin - timedelta(minutes=60 - index) for index in range(61)]
        if any(timestamp not in lookup for timestamp in times):
            continue
        samples.append(dict(
            target_time=target.isoformat(), as_of=origin.isoformat(),
            split=split_name(target), times=[timestamp.isoformat() for timestamp in times],
            history=[lookup[timestamp]["values"] for timestamp in times],
            target_values=list(row["values"]), target_row_number=row.get("row_number"),
        ))
    return samples


def extract_features(history):
    """Use the exact frozen runtime feature function for training and evaluation."""
    return GasAttributionAdapter._features(SimpleNamespace(manifest={"preprocessing": "absolute"}), history)[0]


def cause_map_from_centers(centers):
    if len(centers) != 5 or any(len(row) != 6 or any(not isfinite(value) for value in row) for row in centers):
        raise ValueError("原因映射须使用五个六维有限聚类中心")
    order = [[0] * 6 for _ in range(5)]
    for column in range(6):
        # Original tuple sort breaks equal-center ties by descending cluster id.
        ranked = sorted(((float(centers[cluster][column]), cluster) for cluster in range(5)), reverse=True)
        for rank, (_, cluster) in enumerate(ranked, 1):
            order[cluster][column] = rank
    mapping = {}
    for cluster, ranks in enumerate(order):
        causes = set()
        if ranks[3] <= 2 and ranks[2] > 2 and ranks[4] > 2 and ranks[5] > 2:
            causes.update(("落煤瓦斯涌出", "片帮"))
        if ranks[2] <= 2 and (ranks[4] <= 2 or ranks[5] <= 2):
            causes.add("割煤速度快")
        if ranks[2] > 2 and (ranks[4] <= 2 or ranks[5] <= 2):
            causes.add("超距截割")
        if ranks[0] == 1:
            causes.update(("邻近层瓦斯涌出", "采空区瓦斯涌出"))
        if ranks[1] == 5:
            causes.add("通风系统异常")
        mapping[str(cluster)] = sorted(causes or {"其他原因"})
    return mapping


def fit_reference(rows, threshold=.65):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    selected = [row for row in rows if split_name(row["event_time"]) == "train" and row["values"][0] >= threshold]
    if len(selected) < 5:
        raise ValueError("KMeans训练超限样本不足")
    values = [[abs(value) for value in row["values"]] for row in selected]
    scaler = MinMaxScaler().fit(values)
    transformed = scaler.transform(values)
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(transformed)
    if len(set(int(value) for value in kmeans.labels_)) != 5:
        raise ValueError("训练数据未形成五个有效聚类")
    return dict(scaler=scaler, kmeans=kmeans, cause_map=cause_map_from_centers(kmeans.cluster_centers_),
                sample_count=len(selected), fit_start=selected[0]["event_time"], fit_end=FIT_END.isoformat())


def label_samples(samples, reference):
    if not samples:
        return []
    absolute = [[abs(value) for value in sample["target_values"]] for sample in samples]
    return [int(label) for label in reference["kmeans"].predict(reference["scaler"].transform(absolute))]


def _selected_classes(classes, probabilities):
    ranked = sorted(zip(classes, probabilities), key=lambda item: (-float(item[1]), int(item[0])))
    cut = max(range(1, 5), key=lambda i: ranked[i - 1][1] - ranked[i][1])
    return [int(cluster) for cluster, _ in ranked[:cut]]


def evaluate(classifier, reference, samples, majority_class):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    labels = label_samples(samples, reference)
    if not samples:
        return {"sample_count": 0}, []
    features = [extract_features(sample["history"]) for sample in samples]
    predicted = [int(value) for value in classifier.predict(features)]
    probabilities = classifier.predict_proba(features)
    candidates = [_selected_classes(classifier.classes_, row) for row in probabilities]
    counts = {str(i): labels.count(i) for i in range(5)}
    baseline = [majority_class] * len(labels)
    metrics = {
        "sample_count": len(samples), "first_target_time": samples[0]["target_time"],
        "last_target_time": samples[-1]["target_time"], "class_counts": counts,
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=list(range(5)), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(5))).tolist(),
        "majority_baseline_class": majority_class,
        "majority_baseline_accuracy": float(accuracy_score(labels, baseline)),
        "majority_baseline_macro_f1": float(f1_score(labels, baseline, labels=list(range(5)), average="macro", zero_division=0)),
        "candidate_set_coverage": sum(label in values for label, values in zip(labels, candidates)) / len(labels),
        "mean_candidate_cluster_count": sum(map(len, candidates)) / len(candidates),
        "probability_top1_accuracy": sum(int(classifier.classes_[int(row.argmax())]) == label for row, label in zip(probabilities, labels)) / len(labels),
    }
    records = [dict(target_time=sample["target_time"], as_of=sample["as_of"],
                    reference_cluster=label, predicted_cluster=prediction, selected_cluster_ids=selected,
                    probabilities={str(int(c)): float(p) for c, p in zip(classifier.classes_, probability)},
                    target_row_number=sample["target_row_number"])
               for sample, label, prediction, selected, probability in zip(samples, labels, predicted, candidates, probabilities)]
    return metrics, records


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_bundle(january, february, output, report_dir):
    import numpy as np
    from sklearn.svm import SVC
    from threadpoolctl import threadpool_limits
    output, report_dir = Path(output), Path(report_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("已有模型产物目录保持不变")
    if (report_dir / "report.json").exists():
        raise ValueError("已有训练报告保持不变")
    start = perf_counter()
    paths = [Path(january), Path(february)]
    hashes = {path.name: _sha(path) for path in paths}
    january_rows, february_rows = [read_minute_rows(path) for path in paths]
    if any(split_name(row["event_time"]) == "test" for row in january_rows) or any(split_name(row["event_time"]) != "test" for row in february_rows):
        raise ValueError("一月训练文件和二月测试文件范围不一致")
    samples = build_samples(january_rows) + build_samples(february_rows)
    partitions = {name: [sample for sample in samples if sample["split"] == name]
                  for name in ("train", "validation", "test")}
    if not partitions["train"] or not partitions["test"]:
        raise ValueError("训练及独立测试样本均须存在")
    print(json.dumps({"stage": "samples", "counts": {name: len(values) for name, values in partitions.items()}}, ensure_ascii=False), flush=True)
    with threadpool_limits(limits=2):
        reference = fit_reference(january_rows)
        labels = label_samples(partitions["train"], reference)
        if sorted(set(labels)) != list(range(5)):
            raise ValueError("SVC训练样本缺少部分聚类类别")
        features = [extract_features(sample["history"]) for sample in partitions["train"]]
        classifier = SVC(kernel="rbf", C=100, gamma=.01, probability=True, random_state=0).fit(features, labels)
        fit_seconds = perf_counter() - start
        print(json.dumps({"stage": "fitted", "seconds": round(fit_seconds, 3)}, ensure_ascii=False), flush=True)
        majority = max(range(5), key=lambda label: (labels.count(label), -label))
        evaluated = {name: evaluate(classifier, reference, values, majority) for name, values in partitions.items()}
    scaler = reference["scaler"]
    metadata = {
        "cluster_centers": reference["kmeans"].cluster_centers_.tolist(),
        "scaler": {name: getattr(scaler, attribute).tolist() for name, attribute in (
            ("data_min", "data_min_"), ("data_max", "data_max_"), ("scale", "scale_"), ("min", "min_"))},
        "source_hashes": hashes,
        "training_range": {"start_at": january_rows[0]["event_time"], "end_at": VALIDATION_END.isoformat()},
        "fit_range": {"start_at": january_rows[0]["event_time"], "end_at": FIT_END.isoformat()},
        "validation_range": {"start_at": FIT_END.isoformat(), "end_at": VALIDATION_END.isoformat()},
        "validation_usage": "report_only_no_hyperparameter_tuning",
        "training_sources": [paths[0].name], "test_sources": [paths[1].name],
        "reference_label_semantics": "frozen_training_kmeans_membership",
        "threshold": .65, "horizon_seconds": 300,
        "kmeans": {"n_clusters": 5, "random_state": 0, "n_init": 10},
        "source_lineage": "existing_prepared_multisensor_csv",
    }
    manifest = export_bundle(classifier, reference["cause_map"], metadata, output,
                             model_id=MODEL_ID, version=MODEL_VERSION, horizon_seconds=300,
                             preprocessing="absolute")
    manifest.update(source_id="multisensor-31002-2022", face_id="31002",
                    training_profile="chronological_january_fit_february_test_v1")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    loaded = GasAttributionAdapter(output)
    loaded_classifier = loaded._load_classifier(output / "classifier.skops")
    check_features = [extract_features(sample["history"]) for sample in partitions["test"]]
    difference = float(np.max(np.abs(classifier.predict_proba(check_features) - loaded_classifier.predict_proba(check_features))))
    if difference > 1e-12:
        raise ValueError("SVC保存加载概率不一致")
    sample = partitions["test"][0]
    actual = loaded.predict(dict(as_of=sample["as_of"], feature_names=FEATURE_NAMES,
                                 times=sample["times"], values=sample["history"]))
    if actual["status"] != "completed" or loaded.inspect()["status"] != "ready":
        raise ValueError("已训练溯源模型加载验证失败")
    if any(_sha(path) != hashes[path.name] for path in paths):
        raise ValueError("训练过程中原始输入文件发生变化")
    report = {
        "model_id": MODEL_ID, "model_version": MODEL_VERSION, "status": "completed",
        "reference_label_semantics": "frozen_training_kmeans_membership",
        "evaluation_scope": "cluster_category_classification_not_independent_physical_cause_validation",
        "input_feature_version": "legacy_row_stats_v1", "preprocessing": "absolute",
        "sample_gate": "future_target_T0_at_least_0.65", "horizon_seconds": 300,
        "fit_range": metadata["fit_range"], "information_cutoff": VALIDATION_END.isoformat(),
        "validation_usage": metadata["validation_usage"],
        "partitions": {name: metrics for name, (metrics, _) in evaluated.items()},
        "frozen_reference_fit_sample_count": reference["sample_count"],
        "cause_map": reference["cause_map"], "source_hashes": hashes,
        "source_files_modified": False, "fit_seconds": fit_seconds,
        "total_seconds": perf_counter() - start,
        "serialization_max_probability_difference": difference,
        "bundle_dir": str(output.resolve()), "first_test_inference": actual,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, (_, records) in evaluated.items():
        (report_dir / f"{name}-predictions.json").write_text(json.dumps(records, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "completed", "test": report["partitions"]["test"], "total_seconds": report["total_seconds"]}, ensure_ascii=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="按时间分区训练31002候选溯源模型")
    parser.add_argument("--january", type=Path, required=True)
    parser.add_argument("--february", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    train_bundle(args.january, args.february, args.output, args.report_dir)


if __name__ == "__main__":
    main()
