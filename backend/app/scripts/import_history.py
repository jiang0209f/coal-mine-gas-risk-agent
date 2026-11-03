"""在明确的时间范围内导入本地原始监测文件。"""

import argparse
from collections.abc import Sequence
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import Settings
from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, utcstr
from app.services.data.service import DataService

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按[start,end)导入CSV/XLSX原始监测数据")
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--face-id", required=True)
    parser.add_argument("--point-id", required=True)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--point-name")
    parser.add_argument("--sheet")
    parser.add_argument("--timezone", help="无时区时间的源时区，默认+08:00")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--mapping-file", type=Path, help="高级列映射JSON；多点元数据可放入points列表")
    return parser


def _timestamp(value: str, zone_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.utcoffset() is None:
            if zone_name.startswith(("+", "-")):
                zone = datetime.fromisoformat("2000-01-01T00:00:00" + zone_name).tzinfo
            else:
                zone = ZoneInfo(zone_name)
            parsed = parsed.replace(tzinfo=zone)
        utcstr(parsed)
        return parsed
    except (ValueError, TypeError, ZoneInfoNotFoundError) as exc:
        raise DomainError("导入时间或源时区无效") from exc


def _run(arguments: argparse.Namespace, settings) -> dict:
    mapping = {"time_column": "STime", "value_columns": {"SValue": arguments.point_id},
               "lineage": "raw", "header_row": 1, "invalid_values": [-999999]}
    if arguments.mapping_file:
        try:
            advanced = json.loads(arguments.mapping_file.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise DomainError("映射文件须为有效JSON对象") from exc
        if not isinstance(advanced, dict):
            raise DomainError("映射文件须为JSON对象")
        mapping.update(advanced)
    point_specs = mapping.pop("points", [])
    required = {"point_id", "source_tag", "name", "quantity", "unit"}
    optional = {"unit_status", "mapping_status", "lineage"}
    if not isinstance(point_specs, list) or any(
        not isinstance(spec, dict) or not required.issubset(spec)
        or set(spec) - required - optional for spec in point_specs
    ):
        raise DomainError("points须为明确包含point_id/source_tag/name/quantity/unit的元数据列表")
    zone_name = arguments.timezone or mapping.get("timezone", "+08:00")
    if not isinstance(zone_name, str) or not zone_name:
        raise DomainError("源时区须为非空字符串")
    mapping["timezone"] = zone_name
    if arguments.sheet is not None:
        mapping["sheet"] = arguments.sheet
    start_at = _timestamp(arguments.start, zone_name)
    end_at = _timestamp(arguments.end, zone_name)
    if start_at >= end_at:
        raise DomainError("导入结束时间须晚于开始时间")
    actor = Actor(user_id="local-cli-import", roles=("safety_management",), face_ids=(arguments.face_id,))
    runtime = Path(settings.runtime_dir).resolve()
    database_url = settings.database_url or f"sqlite+pysqlite:///{(runtime / 'app.db').as_posix()}"
    store = Store(database_url)
    try:
        service = DataService(store, runtime / "uploads")
        source = service.register_source(arguments.source_id, arguments.source_name or arguments.source_id,
                                         face_id=arguments.face_id, timezone=zone_name)
        if not any(spec["point_id"] == arguments.point_id for spec in point_specs):
            service.register_point(arguments.point_id, arguments.source_id, arguments.source_tag,
                                   arguments.point_name or arguments.point_id, "methane", "%", lineage="raw")
        for spec in point_specs:
            service.register_point(source_id=arguments.source_id, **spec)
        columns = mapping.get("value_columns")
        if not isinstance(columns, dict) or not columns or any(not isinstance(value, str) for value in columns.values()):
            raise DomainError("需要非空的测值列到测点编号映射")
        missing = [f"points[{point_id}]" for point_id in dict.fromkeys(columns.values())
                   if store.get("point", point_id) is None]
        if missing:
            raise DomainError("多点映射缺少明确的测点元数据", missing_fields=missing)
        file = service.register_file(arguments.path, actor)
        report = service.import_history(actor, arguments.source_id, file["file_id"], mapping,
                                        start_at=start_at, end_at=end_at)
        context = ExecutionContext(session_id=f"import:{report['import_id']}", mode="historical_replay",
                                   source_id=arguments.source_id, as_of=end_at, actor=actor)
        summaries = []
        for point_id in dict.fromkeys(mapping["value_columns"].values()):
            packet = service.query_series(context, point_id, start_at, end_at, limit=1)
            summaries.append({key: packet[key] for key in ("point_id", "as_of", "total", "quality", "unit", "lineage")})
        return {"status": "completed", "source": source, "file": file,
                "report": report, "query_quality": summaries}
    finally:
        store.close()


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = _run(arguments, settings if settings is not None else Settings(_env_file=None))
    except DomainError as exc:
        print(json.dumps({"status": "failed", "error": exc.code, "message": exc.message,
                          "missing_fields": exc.missing_fields}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
