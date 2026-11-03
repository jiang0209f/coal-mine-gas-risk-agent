"""监测数据目录、流式导入与按执行时钟查询。"""

from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from zipfile import BadZipFile

from openpyxl import load_workbook
from sqlalchemy import func, insert, select

from app.db import Store, observations
from app.schemas.common import Actor, DomainError, ExecutionContext, now_utc, utcstr

__all__ = ["DataService"]


def _zone(value):
    if not isinstance(value, str):
        raise DomainError("需要明确源数据时区")
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", value)
    if match:
        hours, minutes = int(match[2]), int(match[3])
        if hours > 23 or minutes > 59:
            raise DomainError("源数据时区无效")
        return timezone(timedelta(minutes=(hours * 60 + minutes) * (1 if match[1] == "+" else -1)))
    try:
        return ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise DomainError("源数据时区无效") from exc


def _time(value, zone):
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    inferred = parsed.utcoffset() is None
    if inferred:
        parsed = parsed.replace(tzinfo=zone)
    return utcstr(parsed), inferred


class DataService:
    def __init__(self, store: Store, upload_dir: Path):
        self.store = store
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def register_source(self, source_id, name, face_id="31002", timezone="+08:00"):
        if not all(isinstance(v, str) and v.strip() for v in (source_id, name, face_id)):
            raise DomainError("数据源编号、名称和工作面不能为空")
        _zone(timezone)
        source = dict(source_id=source_id, name=name, face_id=face_id, timezone=timezone)
        existing = self.store.get("source", source_id)
        if existing is not None and existing != source:
            raise DomainError("已注册数据源的元数据不可覆盖", "conflict", 409)
        return self.store.put("source", source_id, source)

    def register_point(self, point_id, source_id, source_tag, name, quantity, unit,
                       unit_status="inferred", mapping_status="confirmed", lineage="raw"):
        source = self._source(source_id)
        if not all(isinstance(v, str) and v.strip() for v in (point_id, source_tag, name, quantity, unit)):
            raise DomainError("点位元数据不能为空")
        if unit_status not in ("inferred", "confirmed") or mapping_status not in ("confirmed", "pending"):
            raise DomainError("点位单位或映射状态无效")
        if lineage not in ("raw", "processed", "unknown"):
            raise DomainError("数据加工来源无效")
        point = dict(point_id=point_id, source_id=source_id, face_id=source["face_id"],
                     source_tag=source_tag, name=name, quantity=quantity, unit=unit,
                     unit_status=unit_status, mapping_status=mapping_status, lineage=lineage)
        existing = self.store.get("point", point_id)
        if existing is not None and existing != point:
            raise DomainError("已注册点位的元数据不可覆盖", "conflict", 409)
        if any(p["source_id"] == source_id and p["source_tag"] == source_tag and p["point_id"] != point_id
               for p in self.store.list("point")):
            raise DomainError("同一数据源的源标签已映射", "conflict", 409)
        return self.store.put("point", point_id, point)

    def _source(self, source_id):
        source = self.store.get("source", source_id)
        if source is None:
            raise DomainError("数据源不存在", "not_found", 404)
        return source

    def _point(self, point_id):
        point = self.store.get("point", point_id)
        if point is None:
            raise DomainError("点位不存在", "not_found", 404)
        return point

    def list_sources(self, actor):
        return [s for s in self.store.list("source") if s["face_id"] in actor.face_ids]

    def list_points(self, actor, source_id=None):
        if source_id is not None:
            actor.require_face(self._source(source_id)["face_id"])
        return [p for p in self.store.list("point") if p["face_id"] in actor.face_ids
                and (source_id is None or p["source_id"] == source_id)]

    def register_file(self, path: Path, actor: Actor):
        actor.require_role("safety_management")
        if actor.is_system:
            raise DomainError("文件导入需要人工身份", "forbidden", 403)
        path = Path(path)
        if path.suffix.lower() not in (".csv", ".xlsx") or not path.is_file():
            raise DomainError("监测文件须为CSV或XLSX")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        file_id = digest.hexdigest()
        stored_path = self.upload_dir / (file_id + path.suffix.lower())
        if not stored_path.exists():
            shutil.copyfile(path, stored_path)
        value = dict(file_id=file_id, sha256=file_id, name=path.name, size=path.stat().st_size,
                     stored_path=str(stored_path.resolve()), owner_id=actor.user_id)
        existing = self.store.get("data_file", file_id)
        if existing is not None:
            value = existing
        self.store.put("data_file", file_id, value)
        # 同字节文件可由多个有权限的上传者独立使用。
        self.store.put("data_file_access", f"{file_id}:{actor.user_id}", {"file_id": file_id, "user_id": actor.user_id})
        return {k: v for k, v in value.items() if k not in ("stored_path", "owner_id")}

    @staticmethod
    def _rows(path, mapping):
        header_row = mapping.get("header_row", 1)
        if not isinstance(header_row, int) or isinstance(header_row, bool) or header_row < 1:
            raise DomainError("表头行须为正整数")
        workbook = None
        stream = None
        try:
            if path.suffix.lower() == ".xlsx":
                workbook = load_workbook(path, read_only=True, data_only=True)
                sheet_name = mapping.get("sheet")
                if sheet_name is not None and sheet_name not in workbook.sheetnames:
                    raise DomainError("工作表不存在")
                iterator = (workbook[sheet_name] if sheet_name else workbook.active).iter_rows(values_only=True)
            else:
                encoding = "utf-8-sig"
                try:
                    with path.open("r", encoding=encoding) as probe:
                        for _ in probe:
                            pass
                except UnicodeDecodeError:
                    encoding = "gb18030"
                stream = path.open("r", encoding=encoding, newline="")
                iterator = csv.reader(stream)
            header = None
            for row_number, values in enumerate(iterator, 1):
                if row_number < header_row:
                    continue
                if row_number == header_row:
                    header = [str(v).strip() if v is not None else "" for v in values]
                    required = [mapping["time_column"], *mapping["value_columns"]]
                    if mapping.get("availability_column"):
                        required.append(mapping["availability_column"])
                    if any(header.count(column) != 1 for column in required):
                        raise DomainError("文件表头缺少映射列或存在同名列")
                    continue
                yield row_number, dict(zip(header, values))
            if header is None:
                raise DomainError("文件中没有指定的表头行")
        finally:
            if workbook:
                workbook.close()
            if stream:
                stream.close()

    def import_history(self, actor, source_id, file_id, mapping, start_at=None, end_at=None):
        actor.require_role("safety_management")
        if actor.is_system:
            raise DomainError("历史导入需要人工身份", "forbidden", 403)
        source = self._source(source_id)
        actor.require_face(source["face_id"])
        if not isinstance(mapping, dict) or not isinstance(mapping.get("value_columns"), dict) or not mapping["value_columns"] or not mapping.get("time_column"):
            raise DomainError("需要时间列和测值列映射")
        if len(set(mapping["value_columns"].values())) != len(mapping["value_columns"]):
            raise DomainError("同一文件的多个测值列不能映射到同一测点")
        zone = _zone(mapping.get("timezone", source["timezone"]))
        lineage = mapping.get("lineage", "unknown")
        if lineage not in ("raw", "processed", "unknown"):
            raise DomainError("数据加工来源无效")
        points = {col: self._point(pid) for col, pid in mapping["value_columns"].items()}
        if any(p["source_id"] != source_id for p in points.values()):
            raise DomainError("点位与导入数据源不符", "forbidden", 403)
        file = self.store.get("data_file", file_id)
        if file is None:
            raise DomainError("上传文件不存在", "not_found", 404)
        if self.store.get("data_file_access", f"{file_id}:{actor.user_id}") is None:
            raise DomainError("无权使用此上传文件", "forbidden", 403)
        lower, upper = utcstr(start_at) if start_at else None, utcstr(end_at) if end_at else None
        if lower and upper and lower >= upper:
            raise DomainError("导入结束时间须晚于开始时间")
        mapping_digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
        spec = dict(file_id=file_id, source_id=source_id, mapping=mapping, start_at=lower, end_at=upper)
        import_id = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        with self.store.transaction() as connection:
            existing = self.store.get("data_import", import_id, connection)
            if existing:
                return {**existing, "duplicate_file": True, "imported_rows": 0}
            report = dict(import_id=import_id, source_id=source_id, file_id=file_id,
                          imported_rows=0, invalid_time_rows=0, invalid_value_rows=0,
                          skipped_rows=0, duplicate_rows=0, duplicate_file=False,
                          mapping=mapping, start_at=lower, end_at=upper, created_at=utcstr(now_utc()))
            batch = []

            def flush():
                if not batch:
                    return
                ids = [r["id"] for r in batch]
                previous = set(connection.execute(select(observations.c.id).where(observations.c.id.in_(ids))).scalars())
                unique = {r["id"]: r for r in batch if r["id"] not in previous}
                report["duplicate_rows"] += len(batch) - len(unique)
                if unique:
                    connection.execute(insert(observations), list(unique.values()))
                    report["imported_rows"] += len(unique)
                batch.clear()

            try:
                for row_number, row in self._rows(Path(file["stored_path"]), mapping):
                    try:
                        event_time, inferred = _time(row.get(mapping["time_column"]), zone)
                        availability_column = mapping.get("availability_column")
                        available_at, _ = _time(row.get(availability_column), zone) if availability_column else (event_time, False)
                    except (ValueError, TypeError, OverflowError):
                        report["invalid_time_rows"] += 1
                        continue
                    if (lower and event_time < lower) or (upper and event_time >= upper):
                        report["skipped_rows"] += 1
                        continue
                    for column, point in points.items():
                        if "tagname" in row and str(row["tagname"]) != point["source_tag"]:
                            report["skipped_rows"] += 1
                            continue
                        raw = row.get(column)
                        flags = []
                        if not availability_column:
                            flags.append("availability_inferred")
                        if inferred:
                            flags.append("timezone_inferred")
                        if point["unit_status"] != "confirmed":
                            flags.append("unit_inferred")
                        if point["mapping_status"] != "confirmed":
                            flags.append("mapping_pending")
                        effective_lineage = lineage if point["lineage"] == "raw" else point["lineage"]
                        if effective_lineage != "raw":
                            flags.append(f"lineage_{effective_lineage}")
                        try:
                            value = float(raw)
                            sentinels = mapping.get("invalid_values", [-999999])
                            if not math.isfinite(value) or value in sentinels or (point["quantity"] == "methane" and value < 0):
                                raise ValueError
                        except (ValueError, TypeError, OverflowError):
                            value = None
                            flags.append("invalid_value")
                            report["invalid_value_rows"] += 1
                        key = f"{file_id}:{source_id}:{mapping_digest}:{row_number}:{point['point_id']}"
                        batch.append(dict(id=hashlib.sha256(key.encode()).hexdigest(), point_id=point["point_id"], source_id=source_id,
                                          event_time=event_time, available_at=available_at, raw_value=None if raw is None else str(raw),
                                          value=value, quality_flags=flags, source_file_id=file_id, row_number=row_number, import_id=import_id))
                        if len(batch) >= 500:
                            flush()
                flush()
            except (OSError, ValueError, KeyError, csv.Error, BadZipFile, UnicodeError) as exc:
                raise DomainError("文件读取失败，请检查格式和映射") from exc
            self.store.put("data_import", import_id, report, connection)
            return report

    def ingest_readings(self, actor, source_id, readings, received_at=None):
        """适配现有监测系统的统一入口，接收时间由服务端确定。"""
        actor.require_role("safety_management")
        actor.require_face(self._source(source_id)["face_id"])
        if not isinstance(readings, list) or not 1 <= len(readings) <= 10000:
            raise DomainError("单次接收须为1至10000条测值")
        arrived = utcstr(received_at or now_utc())
        batch = []
        for reading in readings:
            if not isinstance(reading, dict) or not isinstance(reading.get("reading_id"), str) or not reading["reading_id"]:
                raise DomainError("实时测值需要稳定的reading_id")
            point = self._point(reading.get("point_id"))
            if point["source_id"] != source_id:
                raise DomainError("实时点位与数据源不符", "forbidden", 403)
            event = reading.get("event_time")
            if isinstance(event, str):
                try:
                    event = datetime.fromisoformat(event.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise DomainError("实时测值时间格式无效") from exc
            event = utcstr(event)
            raw = reading.get("raw_value")
            flags = []
            try:
                value = float(raw)
                if not math.isfinite(value) or value == -999999 or (point["quantity"] == "methane" and value < 0):
                    raise ValueError
            except (TypeError, ValueError, OverflowError):
                value = None
                flags.append("invalid_value")
            if point["unit_status"] != "confirmed":
                flags.append("unit_inferred")
            if point["lineage"] != "raw":
                flags.append("lineage_" + point["lineage"])
            key = hashlib.sha256(f"live:{source_id}:{point['point_id']}:{reading['reading_id']}".encode()).hexdigest()
            batch.append(dict(id=key, point_id=point["point_id"], source_id=source_id, event_time=event,
                              available_at=arrived, value=value, raw_value=None if raw is None else str(raw),
                              quality_flags=flags, source_file_id="live:" + source_id, row_number=0,
                              import_id="live:" + reading["reading_id"]))
        imported = 0
        with self.store.transaction() as connection:
            for row in batch:
                existing = connection.execute(select(observations).where(observations.c.id == row["id"])).mappings().first()
                if existing:
                    if any(existing[k] != row[k] for k in ("event_time", "raw_value", "point_id")):
                        raise DomainError("同一reading_id的测值内容冲突", "conflict", 409)
                    continue
                connection.execute(insert(observations).values(**row))
                imported += 1
        return dict(source_id=source_id, imported_rows=imported, duplicate_rows=len(batch) - imported, received_at=arrived)

    def query_series(self, context: ExecutionContext, point_id, start_at, end_at,
                     aggregation_seconds=None, fill_max_seconds=0, limit=1000, offset=0):
        point = self._point(point_id)
        context.actor.require_face(point["face_id"])
        if context.source_id != point["source_id"]:
            raise DomainError("查询点位与会话数据源不符", "forbidden", 403)
        lower, upper, cutoff = utcstr(start_at), utcstr(end_at), utcstr(context.as_of)
        if lower >= upper:
            raise DomainError("查询结束时间须晚于开始时间")
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (limit, offset, fill_max_seconds)) or not 1 <= limit <= 100000 or offset < 0 or fill_max_seconds < 0:
            raise DomainError("查询分页或填补参数无效")
        if aggregation_seconds is not None and (isinstance(aggregation_seconds, bool) or not isinstance(aggregation_seconds, int) or aggregation_seconds < 1):
            raise DomainError("聚合间隔须为正整数秒")
        if aggregation_seconds and max(0, (min(end_at, context.as_of).timestamp() - start_at.timestamp()) // aggregation_seconds) > 100000:
            raise DomainError("单次聚合最多100000个时间桶，请缩小时间范围")
        with self.store.transaction() as connection:
            result = connection.execute(select(observations, func.count().over(partition_by=observations.c.event_time).label("_duplicate_count")).where(
                observations.c.point_id == point_id, observations.c.source_id == context.source_id,
                observations.c.event_time >= lower,
                observations.c.event_time < upper, observations.c.event_time <= cutoff,
                observations.c.available_at <= cutoff).order_by(
                    observations.c.event_time, observations.c.row_number, observations.c.id)).mappings()
            rows, flags, valid, total = [], set(), 0, 0
            for record in result:
                row = dict(record)
                duplicates = row.pop("_duplicate_count")
                row["quality_flags"] = sorted(set(row["quality_flags"]) | ({"duplicate_timestamp"} if duplicates > 1 else set()))
                flags.update(row["quality_flags"])
                valid += row["value"] is not None
                if aggregation_seconds is not None or offset <= total < offset + limit:
                    rows.append(row)
                total += 1
        if aggregation_seconds is not None:
            rows = self._aggregate(rows, start_at, min(end_at, context.as_of), aggregation_seconds, fill_max_seconds)
            total = len(rows)
            flags = {flag for row in rows for flag in row["quality_flags"]}
            valid = sum(row["value"] is not None for row in rows)
            rows = rows[offset:offset + limit]
        quality = {"flags": sorted(flags), "valid_rows": valid, "invalid_rows": total - valid}
        return dict(point_id=point_id, as_of=cutoff, rows=rows, total=total,
                    limit=limit, offset=offset, quality=quality, unit=point["unit"], lineage=point["lineage"])

    @staticmethod
    def _aggregate(rows, start_at, end_at, seconds, fill_max_seconds):
        start = math.ceil(start_at.timestamp() / seconds) * seconds
        end = end_at.timestamp()
        if max(0, (end - start) // seconds) > 100000:
            raise DomainError("单次聚合最多100000个时间桶，请缩小时间范围")
        groups = {}
        for row in rows:
            key = math.floor(datetime.fromisoformat(row["event_time"]).timestamp() / seconds) * seconds
            groups.setdefault(key, []).append(row)
        output, previous, previous_time = [], None, None
        cursor = start
        while cursor + seconds <= end:
            members = groups.get(cursor, [])
            values = [r["value"] for r in members if r["value"] is not None]
            flags = {flag for r in members for flag in r["quality_flags"]}
            mean = sum(values) / len(values) if values else None
            value = mean
            if values:
                previous, previous_time = mean, cursor
            else:
                flags.add("missing_bucket")
                if previous_time is not None and cursor - previous_time <= fill_max_seconds and fill_max_seconds > 0:
                    value = previous
                    flags.add("forward_filled")
            output.append(dict(event_time=utcstr(datetime.fromtimestamp(cursor, timezone.utc)),
                               bucket_end=utcstr(datetime.fromtimestamp(cursor + seconds, timezone.utc)),
                               value=value, min=min(values) if values else None, max=max(values) if values else None,
                               mean=mean, count=len(values), quality_flags=sorted(flags),
                               available_at=max((r["available_at"] for r in members), default=None),
                               source_refs=[dict(observation_id=r["id"], source_file_id=r["source_file_id"],
                                                 row_number=r["row_number"], event_time=r["event_time"],
                                                 available_at=r["available_at"]) for r in members]))
            cursor += seconds
        return output
