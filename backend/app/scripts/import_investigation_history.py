"""Import explicitly selected historical channels without writing to source files."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path
import re

from sqlalchemy import insert, select
from sqlalchemy.engine import make_url

from app.db import Store, observations
from app.schemas.common import DomainError, now_utc, utcstr

__all__ = ["run_import", "main"]

_ZONE = timezone(timedelta(hours=8))
_CMJ = {
    "采煤机速度": ("speed", "shearer_speed", "m/min"),
    "采煤机位置": ("position", "shearer_position", "source_unit_unconfirmed"),
    "左截割电机电流": ("current_left", "current", "A"),
    "右截割电机电流": ("current_right", "current", "A"),
    "左变频输出电流": ("inverter_left", "current", "A"),
    "右变频输出电流": ("inverter_right", "current", "A"),
    "瓦斯浓度": ("cmj_gas", "methane", "%"),
    "支架号": ("support", "support_number", "1"),
}


def _timestamp(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(value, datetime):
        raise DomainError("时间格式无效")
    return value.replace(tzinfo=_ZONE) if value.utcoffset() is None else value


def _specifications(root, start, end, channels):
    files = list(root.rglob("*.csv"))
    specs = []
    for channel in channels:
        if channel in ("T0", "T1", "T2", "T5"):
            paths = [path for path in files if path.name == f"{channel}_resample_15s.csv" and path.parent.name == f"{channel}_2022" and "处理后数据" in path.parts]
            columns = {f"{channel.lower()}_max": (channel, "methane", "%")}
            step = 15
        elif channel == "FS":
            paths = [path for path in files if path.name == "FS_resample_60s.csv" and path.parent.name == "FS_202201_202212"]
            columns, step = {"FSmean_045A11_1": ("wind_speed", "wind_speed", "m/s")}, 60
        elif channel == "CMJ":
            paths = []
            for path in files:
                month = re.fullmatch(r"CMJ_(\d{6})", path.parent.parent.name)
                if path.name != "CMJ_resample_1min.csv" or path.parent.name != "preprocess" or not month:
                    continue
                first = datetime.strptime(month[1], "%Y%m").replace(tzinfo=_ZONE)
                last = (first.replace(day=28)+timedelta(days=4)).replace(day=1)
                if first < end and last > start:
                    paths.append(path)
            columns, step = _CMJ, 60
        else:
            raise DomainError("不支持的调查数据通道")
        for path in sorted(paths):
            specs.append({"channel": channel, "path": path, "columns": columns, "aggregation_seconds": step})
    return specs


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_import(source_root, database_url, output_dir, start, end, *, channels=("T0", "T1", "T2", "T5", "CMJ", "FS"), face_id="31002", dataset_group="31002-2022"):
    source_root, output_dir = Path(source_root).resolve(), Path(output_dir).resolve()
    start, end = _timestamp(start), _timestamp(end)
    if start >= end:
        raise DomainError("结束时间须晚于开始时间")
    database_path = make_url(database_url).database
    if output_dir.is_relative_to(source_root) or (database_path and database_path != ":memory:" and Path(database_path).resolve().is_relative_to(source_root)):
        raise DomainError("数据库和产物目录须位于源资料目录之外")
    specs = _specifications(source_root, start, end, channels)
    output_dir.mkdir(parents=True, exist_ok=True)
    store = Store(database_url)
    report = {"status": "completed", "start_at": utcstr(start), "end_at": utcstr(end), "dataset_group": dataset_group,
              "imported_rows": 0, "duplicate_rows": 0, "invalid_value_rows": 0, "files": [],
              "missing_channels": [channel for channel in channels if not any(spec["channel"] == channel for spec in specs)]}
    try:
        for spec in sorted(specs, key=lambda item: item["path"].stat().st_size):
            path, channel = spec["path"], spec["channel"]
            source_id = f"investigation-{dataset_group}-{channel.lower()}"
            source = store.get("source", source_id)
            if source and (source.get("face_id") != face_id or (source.get("metadata") or {}).get("dataset_group") != dataset_group):
                raise DomainError("已有调查数据源的工作面或分组不一致", "conflict", 409)
            source = source or {"source_id": source_id, "name": f"调查历史{channel}", "face_id": face_id, "timezone": "+08:00",
                "metadata": {"dataset_group": dataset_group, "lineage": "processed", "timezone_basis": "project_inferred", "purpose": "investigation_evidence"}}
            store.put("source", source_id, source)
            file_id = _digest(path)
            if store.get("data_file", file_id) is None:
                store.put("data_file", file_id, {"file_id": file_id, "sha256": file_id, "name": path.name,
                    "stored_path": str(path), "size": path.stat().st_size, "owner_id": "investigation-history-import", "source_read_only": True})
            import_id = hashlib.sha256(f"{file_id}:{source_id}:investigation-v1".encode()).hexdigest()
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or "STime" not in reader.fieldnames:
                    raise DomainError("调查文件缺少STime列")
                mappings = {column: values for column, values in spec["columns"].items() if column in reader.fieldnames}
                if not mappings:
                    raise DomainError("调查文件缺少已确认的通道列")
                point_ids = {}
                for column, (role, quantity, unit) in mappings.items():
                    suffix = "max" if channel in ("T0", "T1", "T2", "T5") else role
                    point_id = f"{source_id}-{suffix}"
                    point = {"point_id": point_id, "source_id": source_id, "face_id": face_id, "source_tag": column,
                        "name": role, "quantity": quantity, "unit": unit, "unit_status": "inferred", "mapping_status": "confirmed", "lineage": "processed",
                        "metadata": {"channel": role, "dataset_group": dataset_group, "source_column": column,
                            "aggregation_seconds": spec["aggregation_seconds"], "value_semantics": "interval_maximum" if suffix == "max" else "interval_mean",
                            "unit_basis": "header_has_no_unit", "availability_basis": "bucket_end_inferred"}}
                    existing = store.get("point", point_id)
                    if existing is not None and existing != point:
                        raise DomainError("已有调查测点元数据不一致", "conflict", 409)
                    store.put("point", point_id, point)
                    point_ids[column] = point_id
                packet = {"source_id": source_id, "file_id": file_id, "path": str(path), "point_ids": list(point_ids.values()),
                          "header": reader.fieldnames, "imported_rows": 0, "duplicate_rows": 0, "invalid_value_rows": 0, "first_at": None, "last_at": None,
                          "aggregation_seconds": spec["aggregation_seconds"], "timezone": "+08:00", "units_confirmed": False}
                batch = []
                lower_text = start.astimezone(_ZONE).strftime("%Y-%m-%d %H:%M:%S")
                upper_text = end.astimezone(_ZONE).strftime("%Y-%m-%d %H:%M:%S")

                def flush():
                    if not batch:
                        return
                    with store.transaction() as connection:
                        known = set(connection.execute(select(observations.c.id).where(observations.c.id.in_([row["id"] for row in batch]))).scalars())
                        unique = [row for row in batch if row["id"] not in known]
                        if unique:
                            connection.execute(insert(observations), unique)
                        packet["imported_rows"] += len(unique)
                        packet["duplicate_rows"] += len(batch)-len(unique)
                    batch.clear()

                for row_number, row in enumerate(reader, 2):
                    raw_time = row.get("STime", "")
                    # These inspected CSVs use sortable local ISO timestamps.
                    if not lower_text <= raw_time < upper_text:
                        continue
                    try:
                        stamp = _timestamp(raw_time)
                    except (ValueError, TypeError):
                        continue
                    event_time = utcstr(stamp)
                    packet["first_at"] = packet["first_at"] or event_time
                    packet["last_at"] = event_time
                    for column, point_id in point_ids.items():
                        raw_value, value = row.get(column), None
                        flags = ["lineage_processed", "timezone_inferred", "unit_inferred", "availability_inferred"]
                        try:
                            value = float(raw_value)
                            if not isfinite(value) or value == -999999 or (mappings[column][1] == "methane" and value < 0):
                                raise ValueError
                        except (ValueError, TypeError, OverflowError):
                            value = None
                            flags.append("invalid_value")
                            packet["invalid_value_rows"] += 1
                        identity = f"{file_id}:{source_id}:{point_id}:{row_number}"
                        batch.append({"id": hashlib.sha256(identity.encode()).hexdigest(), "point_id": point_id, "source_id": source_id,
                            "event_time": event_time, "available_at": utcstr(stamp+timedelta(seconds=spec["aggregation_seconds"])),
                            "raw_value": raw_value, "value": value, "quality_flags": flags, "source_file_id": file_id,
                            "row_number": row_number, "import_id": import_id})
                        if len(batch) >= 500:
                            flush()
                flush()
                report["files"].append(packet)
                for field in ("imported_rows", "duplicate_rows", "invalid_value_rows"):
                    report[field] += packet[field]
                store.put("data_import", import_id, {**packet, "import_id": import_id, "start_at": utcstr(start), "end_at": utcstr(end),
                    "created_at": utcstr(now_utc()), "mapping": {"time_column": "STime", "value_columns": point_ids, "lineage": "processed", "timezone": "+08:00"}})
        if report["missing_channels"]:
            report["status"] = "partial"
        filename = f"import-{start.strftime('%Y%m%dT%H%M')}-{end.strftime('%Y%m%dT%H%M')}.json"
        report["report_path"] = str(output_dir/filename)
        (output_dir/filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读F盘调查资料，按明确时间窗增量导入独立来源")
    parser.add_argument("--root", type=Path, default=Path("F:/瓦斯"))
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", default="2022-04-01T00:00:00+08:00")
    parser.add_argument("--end", default="2022-05-01T00:00:00+08:00")
    parser.add_argument("--face-id", default="31002")
    parser.add_argument("--dataset-group", default="31002-2022")
    parser.add_argument("--channels", nargs="+", choices=("T0", "T1", "T2", "T5", "CMJ", "FS"), default=("T0", "T1", "T2", "T5", "CMJ", "FS"))
    args = parser.parse_args(argv)
    result = run_import(args.root, args.database_url, args.output_dir, args.start, args.end,
                        channels=args.channels, face_id=args.face_id, dataset_group=args.dataset_group)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
