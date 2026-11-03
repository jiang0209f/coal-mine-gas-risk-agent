"""Shared, past-only minute-to-model features for the trained 31002 versions."""

from datetime import datetime, timedelta, timezone
from math import floor, fsum, isfinite
from numbers import Real

from app.adapters.model_bundle import ModelNotReady

CHANNEL_NAMES = ["T0瓦斯浓度", "风速", "采煤机速度", "采煤机瓦斯浓度", "左截割电机电流", "右截割电机电流"]
FORECAST_FEATURE_NAMES = ["风速均值", "采煤机速度均值", "采煤机瓦斯均值", "左截割电流均值", "右截割电流均值", "OT"]
RESIDUAL_FEATURE_NAMES = [f"lag_{lag}:{name}" for lag in range(11, -1, -1) for name in FORECAST_FEATURE_NAMES]
FEATURE_PIPELINE_VERSION = "six_channel_5min_v1"


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.utcoffset() is None:
        raise ValueError("模型输入时刻须包含时区")
    return result.astimezone(timezone.utc)


def _index(rows):
    indexed = {}
    for row in rows:
        moment = timestamp(row["event_time"])
        values = row.get("values")
        if moment.timestamp() % 60 or moment in indexed:
            raise ValueError("模型分钟序列存在重复或未对齐时刻")
        if not isinstance(values, list) or len(values) != 6 or any(
            isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) for v in values
        ):
            raise ValueError("模型分钟输入须为六个有限测值")
        indexed[moment] = list(values)
    return indexed


def aggregate_five_minute(rows):
    """Only complete, aligned five-minute groups; no interpolation or filling."""
    indexed = _index(rows)
    starts = sorted({datetime.fromtimestamp(floor(t.timestamp() / 300) * 300, timezone.utc) for t in indexed})
    result = []
    for start in starts:
        times = [start + timedelta(minutes=i) for i in range(5)]
        if not all(t in indexed for t in times):
            continue
        members = [indexed[t] for t in times]
        values = [fsum(row[column] for row in members) / 5 for column in range(1, 6)]
        values.append(max(row[0] for row in members))
        result.append(dict(event_time=start.isoformat(), bucket_end=(start+timedelta(minutes=5)).isoformat(), values=values))
    return result


def residual_vector(history_values):
    if len(history_values) < 12:
        raise ModelNotReady(["residual_12_bucket_history"])
    selected = history_values[-12:]
    if any(len(row) != 6 for row in selected):
        raise ValueError("残差历史特征维度无效")
    return [float(value) for row in selected for value in row]


def forecast_request(rows, as_of):
    cutoff = timestamp(as_of)
    if cutoff.timestamp() % 300:
        raise ValueError("预测时钟须对齐五分钟闭合边界")
    start = cutoff - timedelta(minutes=450)
    visible = [row for row in rows if start <= timestamp(row["event_time"]) < cutoff]
    buckets = aggregate_five_minute(visible)
    if len(buckets) != 90 or any(timestamp(row["event_time"]) != start + timedelta(minutes=5*i) for i, row in enumerate(buckets)):
        raise ModelNotReady(["complete_450_minute_six_channel_history"])
    values = [row["values"] for row in buckets]
    return dict(as_of=cutoff.isoformat(), unit="%", feature_names=list(FORECAST_FEATURE_NAMES),
                times=[row["event_time"] for row in buckets], values=values,
                residual_feature_names=list(RESIDUAL_FEATURE_NAMES), residual_values=residual_vector(values))


def attribution_request(rows, as_of):
    cutoff = timestamp(as_of)
    if cutoff.timestamp() % 60:
        raise ValueError("溯源输入时刻须对齐分钟")
    start = cutoff - timedelta(minutes=60)
    visible = [row for row in rows if start <= timestamp(row["event_time"]) <= cutoff]
    indexed = _index(visible)
    times = [start + timedelta(minutes=i) for i in range(61)]
    if len(indexed) != 61 or any(t not in indexed for t in times):
        raise ModelNotReady(["complete_61_minute_six_channel_history"])
    return dict(as_of=cutoff.isoformat(), feature_names=list(CHANNEL_NAMES),
                times=[t.isoformat() for t in times], values=[indexed[t] for t in times])
