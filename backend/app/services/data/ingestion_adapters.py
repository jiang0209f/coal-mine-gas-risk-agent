"""Source-bound JSON normalization with stable identities and explicit source time."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math

from app.schemas.common import DomainError, now_utc, utcstr
from app.services.data.service import _zone

__all__ = ["normalize", "validate_config"]

_FIELDS = {"records_path", "id_field", "tag_field", "time_field", "value_field", "sequence_field", "timezone"}
_ABSENT = object()


def _path(value, path, default=_ABSENT):
    result = value
    for part in path.split(".") if path else ():
        if not isinstance(result, dict) or part not in result:
            if default is not _ABSENT:
                return default
            raise DomainError("接入记录缺少已配置字段", code="mapping_field_missing")
        result = result[part]
    return result


def _point(points, point_id, source_id):
    point = next((point for point in points if point.get("point_id") == point_id), None)
    if point is None:
        raise DomainError("接入记录测点尚未登记", code="point_not_registered")
    if point.get("source_id") != source_id:
        raise DomainError("接入测点与绑定来源不一致", code="forbidden", status_code=403)
    return point


def validate_config(config, points):
    if not isinstance(config, dict) or not isinstance(config.get("source_id"), str) or not config["source_id"].strip():
        raise DomainError("接入配置需要来源编号")
    if config.get("adapter_kind") not in ("standard_json", "mapped_json"):
        raise DomainError("接入适配器不存在", code="unsupported_adapter")
    mapping, tags = config.get("mapping", {}), config.get("tag_map", {})
    if not isinstance(mapping, dict) or not isinstance(tags, dict) or set(mapping) - _FIELDS:
        raise DomainError("接入字段映射无效")
    for key, value in mapping.items():
        if key == "timezone":
            if value is not None:
                _zone(value)
        elif value is not None and (not isinstance(value, str) or len(value) > 256 or
                                   (value and any(not part for part in value.split(".")))):
            raise DomainError("接入字段路径无效")
    if config.get("timezone") is not None:
        _zone(config["timezone"])
    if config["adapter_kind"] == "mapped_json":
        if not tags or any(not mapping.get(field) for field in ("tag_field", "time_field", "value_field")):
            raise DomainError("字段适配需要测点、时间、数值映射和标签绑定")
    elif set(mapping) - {"timezone"}:
        raise DomainError("统一测值格式仅需配置源时区")
    for tag, point_id in tags.items():
        if not isinstance(tag, str) or not tag.strip() or not isinstance(point_id, str) or not point_id.strip():
            raise DomainError("测点标签绑定无效")
        _point(points, point_id, config["source_id"])
    return deepcopy(config)


def _event_time(value, timezone, received_at):
    try:
        event = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if not isinstance(event, datetime):
            raise ValueError
        if event.utcoffset() is None:
            if timezone is None:
                raise DomainError("无时区的源时间需要配置源时区")
            event = event.replace(tzinfo=_zone(timezone))
        if event > received_at:
            raise DomainError("接入测值时间晚于服务端接收时间", code="future_event_time")
        return utcstr(event)
    except (ValueError, TypeError, AttributeError, OverflowError):
        raise DomainError("接入测值时间格式无效") from None


def _identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise DomainError("源记录编号或顺序号需要字符串或整数")
    result = str(value)
    if not result.strip() or len(result) > 256 or any(ord(character) < 32 for character in result):
        raise DomainError("源记录编号或顺序号无效")
    return result


def normalize(config, payload, points, received_at=None):
    validate_config(config, points)
    received_at = received_at or now_utc()
    utcstr(received_at)
    mapping = config.get("mapping", {})
    timezone = mapping.get("timezone") or config.get("timezone")
    standard = config["adapter_kind"] == "standard_json"
    rows = payload.get("readings") if standard and isinstance(payload, dict) else payload
    if not standard:
        rows = _path(payload, mapping.get("records_path") or "")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 10000 or any(not isinstance(row, dict) for row in rows):
        raise DomainError("接入批次需要1至10000条记录")
    result = []
    for row in rows:
        if standard:
            point_id = row.get("point_id")
            value = _path(row, "raw_value")
            event = _path(row, "event_time")
            identity, sequence = row.get("reading_id"), row.get("sequence")
        else:
            tag = _identifier(_path(row, mapping["tag_field"]))
            point_id = config["tag_map"].get(tag)
            if point_id is None:
                raise DomainError("源测点标签尚未绑定", code="unknown_source_tag")
            value = _path(row, mapping["value_field"])
            event = _path(row, mapping["time_field"])
            identity = _path(row, mapping["id_field"], None) if mapping.get("id_field") else None
            sequence = _path(row, mapping["sequence_field"], None) if mapping.get("sequence_field") else None
        if not isinstance(point_id, str) or not point_id:
            raise DomainError("接入记录缺少有效测点编号")
        _point(points, point_id, config["source_id"])
        if (isinstance(value, bool) or value is not None and not isinstance(value, (str, int, float))
                or isinstance(value, float) and not math.isfinite(value)):
            raise DomainError("接入测值需要数值、原始文本或空值")
        event = _event_time(event, timezone, received_at)
        if identity is None or identity == "":
            sequence = _identifier(sequence) if sequence is not None else None
            stable = json.dumps([config["source_id"], point_id, event, sequence], separators=(",", ":"), ensure_ascii=False)
            identity = "derived:" + hashlib.sha256(stable.encode("utf-8")).hexdigest()
        else:
            identity = _identifier(identity)
        result.append(dict(reading_id=identity, point_id=point_id, event_time=event, raw_value=value))
    return result
