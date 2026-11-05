"""Run one persisted historical event investigation and stop at human review."""

import argparse
import ast
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import shutil
import sqlite3
import textwrap
from urllib.parse import urlsplit

from app.config import PROJECT_ROOT, Settings
from app.schemas.common import Actor, DomainError, now_utc, utcstr


_RUNTIME = Path("F:/煤矿瓦斯智能体/data/runtime")
_OUTPUT = Path("F:/煤矿瓦斯智能体/outputs/source-control-20261105/demo")
_SOURCE = "investigation-31002-2022-t0"
_POINT = _SOURCE + "-max"
_SOURCES = [f"investigation-31002-2022-{channel}" for channel in ("t0", "t1", "t2", "t5", "cmj", "fs")]
_START = datetime.fromisoformat("2022-04-27T03:00:00+08:00")
_END = datetime.fromisoformat("2022-04-27T03:12:00+08:00")


def _parser():
    parser = argparse.ArgumentParser(description="真实历史事件经DeepSeek只读排查，报告停在人工审核")
    parser.add_argument("--env-file", type=Path, default=Path(os.environ.get("COAL_AGENT_ENV_FILE", PROJECT_ROOT / ".env")))
    parser.add_argument("--runtime-dir", type=Path, default=_RUNTIME)
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT)
    parser.add_argument("--source-id", choices=[_SOURCE], default=_SOURCE)
    parser.add_argument("--prepare-only", action="store_true", help="只读核对来源并写拟发送字段清单，不创建会话或调用API")
    parser.add_argument("--actor-id", help="已存在且有该工作面权限的安全管理账户，默认使用manager")
    parser.add_argument("--new-session", action="store_true", help="显式创建新演示会话；默认复用已有manifest，不重复API")
    return parser


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _actor(database, actor_id):
    # Select identity fields only; password hashes and tokens are not read.
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        query = ("SELECT json_extract(payload,'$.user_id'), json_extract(payload,'$.roles'), "
                 "json_extract(payload,'$.face_ids') FROM records WHERE kind='users' AND ")
        row = connection.execute(query + ("id=?" if actor_id else "json_extract(payload,'$.username')=?"),
                                 (actor_id or "manager",)).fetchone()
    if row is None:
        raise DomainError("演示需要现有本机安全管理账户", "actor_not_found")
    value = Actor(user_id=row[0], roles=tuple(json.loads(row[1])), face_ids=tuple(json.loads(row[2])))
    value.require_role("safety_management")
    return value


def _prepare(output, database):
    from app.services.investigation.service import InvestigationService

    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        found = {row[0] for row in connection.execute(
            "SELECT id FROM records WHERE kind='source' AND id IN (" + ",".join("?" for _ in _SOURCES) + ")", _SOURCES)}
        point = connection.execute("SELECT json_extract(payload,'$.source_id') FROM records WHERE kind='point' AND id=?", (_POINT,)).fetchone()
        count = connection.execute("SELECT count(*) FROM observations WHERE source_id=? AND point_id=? AND event_time>=? AND event_time<=? AND available_at<=?",
                                  (_SOURCE, _POINT, utcstr(_START), utcstr(_END), utcstr(_END))).fetchone()[0]
    if found != set(_SOURCES) or point is None or point[0] != _SOURCE or not count:
        raise DomainError("指定调查来源或关键时间窗未准备完整", "demo_input_not_ready")
    tree = ast.parse(textwrap.dedent(inspect.getsource(InvestigationService._safe_metrics)))
    metrics_keys, enumerations = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Set):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if "keys" in names:
                metrics_keys = sorted(ast.literal_eval(node.value))
            if "safe_strings" in names:
                enumerations = sorted(ast.literal_eval(node.value))
    plan = {
        "authorization": "真实历史数据经 DeepSeek API 处理：只发送必要字段，测点用别名，每次外发记录字段清单。",
        "authorization_scope": "本轮用户在‘数据与大模型’任务的新授权，替代11/4仅合成数据和公开法规的外发限制。",
        "endpoint": "https://api.deepseek.com/chat/completions",
        "local_database": str(database), "allowed_source_ids": _SOURCES,
        "target_source_id": _SOURCE, "target_point_id": _POINT,
        "visible_target_rows": count, "maximum_investigation_tasks": 1,
        "top_level_payload_fields": ["model", "messages", "tools", "tool_choice", "thinking", "stream", "max_tokens"],
        "message_content": {
            "initial": ["固定排查任务说明", "event_id=E1", "事件证据的白名单数字摘要", "EV引用别名"],
            "tool_result": ["status", "metrics", "reference_ids", "missing_field_count"],
            "event_evidence": ["候选归因的固定原因类别与聚类归属概率", "团队草案阈值及状态", "观测数量/极值/均值", "缺项状态"],
            "metrics_allowed_keys": metrics_keys, "metrics_allowed_enumerations": enumerations,
            "alias_policy": "事件E1、测点Pn、EV引用；真实标识及绝对时刻留在本机，仅发相对事件秒数",
        },
        "tools": ["get_event_evidence", "compare_points", "get_condition_timeline", "get_face_statistics", "find_similar_events"],
        "excluded": ["矿井私有规程正文", "完整原始时序", "源文件路径", "真实测点及人员标识", "绝对日期时刻", "API密钥和鉴权头"],
        "audit": "每轮外发均经InvestigationService._wire/_disclose，cloud_disclosure记录字段路径清单和SHA256；产物disclosures.json保存在本机F盘",
        "production_env_modified": False, "approval_actions": 0, "final_state_required": "draft",
        "network_requests": 0, "session_created": False, "prepared_at": utcstr(now_utc()),
    }
    output.mkdir(parents=True, exist_ok=True)
    target = output / "proposed-disclosure-fields.json"
    _json(target, plan)
    return {"state": "prepared", "plan": str(target), "network_requests": 0,
            "source_count": len(found), "visible_target_rows": count, "approval_actions": 0}


def _reuse(output, database):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if Path(manifest.get("database", "")).resolve() != database:
        raise DomainError("已有manifest对应其他运行库", "demo_scope_mismatch")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT json_extract(payload,'$.state') FROM records WHERE kind='investigation_report' AND id=?",
                                 (manifest.get("report_id"),)).fetchone()
    state = row[0] if row else manifest.get("state", "not_found")
    return {"reused": True, "session_id": manifest.get("session_id"), "event_id": manifest.get("event_id"),
            "report_id": manifest.get("report_id"), "state": state, "additional_api_requests": 0,
            "manifest": str(output / "manifest.json")}


def _choose_event(store, events):
    # Event references point to original local observations; use them only here
    # to select the red event nearest the inspected high-value interval.
    from sqlalchemy import func, select
    from app.db import observations

    scored = []
    for event in events:
        if event.get("point_id") != _POINT or event.get("alert_level") != "red":
            continue
        identifiers = [row["observation_id"] for row in event.get("evidence_refs", []) if row.get("observation_id")]
        with store.transaction() as connection:
            peak = connection.execute(select(func.max(observations.c.value)).where(
                observations.c.source_id == _SOURCE, observations.c.point_id == _POINT,
                observations.c.id.in_(identifiers))).scalar_one_or_none() if identifiers else None
        scored.append((peak if isinstance(peak, (float, int)) else float("-inf"), event["start_at"], event))
    if not scored:
        raise DomainError("该历史窗口尚未发布红色事件，保留会话等待核对", "red_event_not_published")
    return max(scored, key=lambda item: (item[0], item[1]))[2]


def _export(output, services, context, manifest, report):
    disclosure_ids = report.get("disclosure_ids", [])
    disclosures = [services["store"].get("cloud_disclosure", identifier) for identifier in disclosure_ids]
    _json(output / "report.json", report)
    _json(output / "disclosures.json", [row for row in disclosures if row])
    (output / "report.docx").write_bytes(services["investigation"].export_docx(context, report["report_id"]))
    manifest.update(state=report["state"], provider_model=report.get("model"),
        event_revision=report["event_revision"], disclosure_count=len(disclosure_ids),
        tool_steps=[{"tool": row.get("tool"), "status": row.get("status"), "step_id": row.get("step_id")}
                    for row in report.get("investigation_steps", [])],
        hypothesis_count=len(report.get("hypotheses", [])),
        supported_count=len(report.get("supported_causes", [])),
        excluded_count=len(report.get("excluded_items", [])),
        pending_count=len(report.get("pending_items", [])),
        error_code=report.get("error_code"), updated_at=utcstr(now_utc()))
    _json(output / "manifest.json", manifest)


def run(arguments):
    runtime, output = arguments.runtime_dir.resolve(), arguments.output_dir.resolve()
    database = (runtime / "app.db").resolve()
    if not database.is_file():
        raise DomainError("指定F盘运行库不存在", "database_not_found")
    if arguments.prepare_only:
        return _prepare(output, database)
    if (output / "manifest.json").is_file() and not arguments.new_session:
        return _reuse(output, database)
    settings = Settings(_env_file=arguments.env_file.resolve())
    endpoint = urlsplit(str(settings.deepseek_base_url))
    if endpoint.scheme != "https" or endpoint.hostname != "api.deepseek.com":
        raise DomainError("演示仅使用DeepSeek官方HTTPS接口", "endpoint_not_authorized")
    if not settings.deepseek_configured:
        raise DomainError("DeepSeek API尚未配置", "deepseek_not_configured")
    actor = _actor(database, arguments.actor_id)
    from app.main import build_services
    configuration = settings.model_copy(update={
        "runtime_dir": runtime, "database_url": "sqlite+pysqlite:///" + database.as_posix(),
        "agent_real_data_mode": "summary", "agent_real_source_ids": _SOURCES,
        "deepseek_max_tokens": 2500, "deepseek_timeout_seconds": 45,
        "deepseek_max_retries": 0,
    })
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").is_file():
        previous = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        archive = output / "history" / previous.get("session_id", now_utc().strftime("%Y%m%dT%H%M%S"))
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("manifest.json", "report.json", "disclosures.json", "report.docx"):
            if (output / name).is_file():
                shutil.copy2(output / name, archive / name)
    services = build_services(configuration)
    manifest = {"database": str(database), "runtime_dir": str(runtime), "source_id": _SOURCE,
        "point_id": _POINT, "actor_id": actor.user_id, "start_as_of": utcstr(_START), "as_of": utcstr(_END),
        "created_at": utcstr(now_utc()), "state": "preparing", "approval_actions": 0,
        "manual_review_required": True, "production_env_modified": False,
        "files": {name: str(output / name) for name in ("report.json", "disclosures.json", "report.docx")}}
    try:
        missing = [source_id for source_id in _SOURCES if services["store"].get("source", source_id) is None]
        if missing:
            raise DomainError("调查通道尚未全部导入", "investigation_sources_missing", missing_fields=missing)
        session = services["sessions"].create_session(actor, "historical_replay", _SOURCE, _START)
        manifest["session_id"] = session["session_id"]
        _json(output / "manifest.json", manifest)
        result = services["runner"].run_once(actor, session["session_id"], target_as_of=_END, schedule=None)
        event = _choose_event(services["store"], result["events"])
        manifest.update(event_id=event["event_id"], event_revision=event["event_revision"],
                        event_level=event["alert_level"], published_event_count=len(result["events"]))
        context = services["sessions"].get_context(actor, session["session_id"])
        scheduled = []
        report = services["investigation"].submit(context, event["event_id"],
            event_revision=event["event_revision"], schedule=scheduled.append)
        manifest.update(report_id=report["report_id"], state=report["state"])
        _json(output / "manifest.json", manifest)
        if len(scheduled) != 1:
            raise DomainError("本次演示应且仅应提交一个排查任务", "unexpected_investigation_count")
        print(json.dumps({"stage": "investigation_started", "session_id": session["session_id"],
                          "event_id": event["event_id"], "report_id": report["report_id"]}), flush=True)
        # Production service performs every tool execution, whitelist projection,
        # disclosure record and API request. No substitute data or adapter.
        scheduled[0]()
        report = services["investigation"].get(context, report["report_id"])
        _export(output, services, context, manifest, report)
        if report["state"] != "draft":
            raise DomainError("排查未形成待人工审核草稿，真实结果已保存", report.get("error_code") or "investigation_not_draft")
        return {"reused": False, **{key: manifest.get(key) for key in (
            "session_id", "event_id", "report_id", "state", "provider_model", "disclosure_count",
            "tool_steps", "hypothesis_count", "supported_count", "excluded_count", "pending_count")},
            "approval_actions": 0, "manifest": str(output / "manifest.json")}
    except Exception as error:
        manifest.update(error_code=getattr(error, "code", "demo_execution_failed"), updated_at=utcstr(now_utc()))
        if manifest.get("state") not in ("failed", "draft"):
            manifest["state"] = "failed"
        _json(output / "manifest.json", manifest)
        raise
    finally:
        services["store"].close()


def main(argv=None):
    try:
        result = run(_parser().parse_args(argv))
    except Exception as error:
        print(json.dumps({"state": "failed", "error_code": getattr(error, "code", "demo_execution_failed"),
                          "error_type": type(error).__name__}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get("state") in ("prepared", "draft", "approved", "rejected") else 1


if __name__ == "__main__":
    raise SystemExit(main())
