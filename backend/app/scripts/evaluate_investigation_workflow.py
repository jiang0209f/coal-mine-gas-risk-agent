"""Resumable real-API acceptance of 21 independent generated scenarios."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from sqlalchemy import insert

from app.adapters.deepseek import DeepSeekAdapter
from app.config import PROJECT_ROOT, Settings
from app.db import observations
from app.main import build_services
from app.schemas.common import Actor, ExecutionContext, utcstr
from app.services.evaluation.workflow_acceptance import build_acceptance

__all__ = ["main"]

_SCENARIOS = [
    ("01-wind-drop", "风速骤降后甲烷上升", "condition_analysis", "get_condition_timeline", "wind_drop"),
    ("02-cutting-load", "割煤速度和双侧电流同步上升", "condition_analysis", "get_condition_timeline", "cutting_load"),
    ("03-neighbor-late", "邻点晚两分钟起升", "neighbor_comparison", "compare_points", "neighbor_late"),
    ("04-neighbor-early", "邻点提前两分钟起升", "neighbor_comparison", "compare_points", "neighbor_early"),
    ("05-single-point", "仅有目标甲烷测点，核查邻点缺项", "neighbor_comparison", "compare_points", "single_point"),
    ("06-missing-wind", "缺风速通道，通风原因保持待核", "condition_analysis", "get_condition_timeline", "missing_wind"),
    ("07-missing-motion", "缺位置与速度，检查工况判定边界", "condition_analysis", "get_condition_timeline", "missing_motion"),
    ("08-short-spike", "一分钟尖峰后恢复，检查高位占比", "face_statistics", "get_face_statistics", "short_spike"),
    ("09-persistent-high", "长时间持续高位，核对持续性", "face_statistics", "get_face_statistics", "persistent_high"),
    ("10-simultaneous", "四个甲烷点同时起升", "neighbor_comparison", "compare_points", "simultaneous"),
    ("11-sequential", "四点依次起升，检查逐级时差", "neighbor_comparison", "compare_points", "sequential"),
    ("12-upstream-lead", "上游点提前三分钟升高", "neighbor_comparison", "compare_points", "upstream_lead"),
    ("13-idle-high", "停机无电流时甲烷升高", "face_statistics", "get_face_statistics", "idle_high"),
    ("14-motion-no-current", "有移动速度但无电流", "face_statistics", "get_face_statistics", "motion_no_current"),
    ("15-current-asymmetry", "左右电流不对称", "condition_analysis", "get_condition_timeline", "current_asymmetry"),
    ("16-wind-gap", "风速在事件附近断测", "condition_analysis", "get_condition_timeline", "wind_gap"),
    ("17-late-neighbor", "邻点记录可用时间晚于截止时间", "neighbor_comparison", "compare_points", "late_neighbor"),
    ("18-no-group", "缺数据组绑定，检查证据组织缺项", "neighbor_comparison", "compare_points", "no_group"),
    ("19-prior-event", "有历史事件但无审核结论", "similar_history", "find_similar_events", "prior_event"),
    ("20-no-history", "没有可用历史事件", "similar_history", "find_similar_events", "no_history"),
    ("21-repeated-peaks", "多次短峰与稳定工况并存", "face_statistics", "get_face_statistics", "repeated_peaks"),
]
_CHANNELS = {"T0": ("methane", "%"), "T1": ("methane", "%"), "T2": ("methane", "%"), "T5": ("methane", "%"),
             "speed": ("shearer_speed", "m/min"), "position": ("shearer_position", "m"),
             "current_left": ("motor_current", "A"), "current_right": ("motor_current", "A"), "wind_speed": ("wind_speed", "m/s")}


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _paths(value, prefix=""):
    if isinstance(value, dict):
        return [path for key, child in value.items() for path in _paths(child, f"{prefix}.{key}".lstrip("."))]
    if isinstance(value, list):
        return [path for index, child in enumerate(value) for path in _paths(child, f"{prefix}[{index}]")]
    return [prefix]


class LoggedDeepSeekAdapter(DeepSeekAdapter):
    """Record actual requests and model choices without modifying either."""
    def __init__(self, settings, log):
        super().__init__(settings)
        self.log = log

    def chat(self, messages, **kwargs):
        envelope = deepcopy({"messages": messages, **kwargs})
        for message in envelope["messages"]:
            try:
                message["content"] = json.loads(message.get("content", ""))
            except (ValueError, TypeError):
                pass
        encoded = json.dumps(envelope, ensure_ascii=False, allow_nan=False)
        record = {"request_index": len(self.log) + 1, "outbound_field_paths": sorted(set(_paths(envelope))),
                  "request_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "synthetic_only": True}
        self.log.append(record)
        started = perf_counter()
        try:
            reply = super().chat(messages, **kwargs)
            record.update(model=reply.model, usage=reply.usage, selected_tools=[call["function"]["name"] for call in reply.message.get("tool_calls") or []])
            return reply
        except Exception as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            record["elapsed_seconds"] = round(perf_counter() - started, 3)


def _configuration(base, runtime, source_id):
    return base.model_copy(update={"runtime_dir": runtime, "database_url": "sqlite+pysqlite:///" + (runtime / "app.db").as_posix(),
        "model_catalog_path": None, "knowledge_embedding_model_dir": None, "knowledge_reranker_model_dir": None,
        "ingestion_worker_enabled": False, "realtime_monitor_enabled": False, "control_worker_enabled": False,
        "event_brief_allowed_source_ids": [], "agent_real_source_ids": [source_id], "agent_real_data_mode": "summary",
        "deepseek_timeout_seconds": 45, "deepseek_max_tokens": 1800, "deepseek_max_retries": 0})


def _value(channel, minute, variant):
    if channel == "T0":
        if minute < -20: return .92 if minute in (-65, -64) else .25
        if variant == "short_spike": return .97 if minute == 0 else .3
        if variant == "repeated_peaks": return .95 if minute in (0, 3, 6) else .3
        if variant == "persistent_high": return .72 if -8 <= minute < 0 else .93 if minute >= 0 else .3
        return .3 if minute < 0 else .91 + .002 * minute
    if channel.startswith("T"):
        delay = {"T1": 2, "T2": 3, "T5": 4}[channel]
        if variant == "neighbor_early" and channel == "T1": delay = -2
        if variant == "upstream_lead" and channel == "T2": delay = -3
        if variant == "simultaneous": delay = 0
        if variant == "sequential": delay = {"T1": 1, "T2": 3, "T5": 5}[channel]
        return .25 if minute < delay else .75 + .001 * minute
    if channel == "wind_speed": return 1.1 if variant == "wind_drop" and minute >= -1 else 3.2
    if channel == "speed": return 0 if variant == "idle_high" else 4 if variant == "cutting_load" and minute >= 0 else 1.2
    if channel == "position": return 40 + minute * 1.2
    if variant in ("idle_high", "motion_no_current"): return 0
    if variant == "current_asymmetry": return 150 if channel == "current_left" else 20
    return 150 if variant == "cutting_load" and minute >= 0 else 65


def _prepare(scenario, directory, base, force=False):
    identifier, description, goal, expected, variant = scenario
    checkpoint = directory / "prepared.json"
    if checkpoint.is_file() and not force:
        prepared = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prepared.get("scenario_seed_version") == 2:
            return prepared
    runtime = directory / "attempts" / str(uuid4()) / "runtime"
    source_id, face = "synthetic-acceptance-" + identifier, "BENCHMARK-" + identifier
    actor = Actor(user_id="synthetic-benchmark-operator", roles=("safety_management",), face_ids=(face,))
    services = build_services(_configuration(base, runtime, source_id))
    try:
        center = datetime(2026, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=int(identifier[:2]))
        source = services["data"].register_source(source_id, description, face)
        source["metadata"] = {"synthetic": True, "scenario": identifier, "provenance": "deterministic_generated_observations"}
        if variant != "no_group": source["metadata"]["dataset_group"] = source_id + "-group"
        services["store"].put("source", source_id, source)
        channels = [channel for channel in _CHANNELS if not (
            (variant == "single_point" and channel != "T0") or (variant == "missing_wind" and channel == "wind_speed")
            or (variant == "missing_motion" and channel in ("position", "speed")))]
        rows = []
        for channel in channels:
            quantity, unit = _CHANNELS[channel]
            point = services["data"].register_point(source_id + ":" + channel, source_id, channel, channel, quantity, unit, unit_status="confirmed")
            point["metadata"] = {"channel": channel, "synthetic": True}
            services["store"].put("point", point["point_id"], point)
            minutes = [*range(-10, 11), *([-65, -64, -63] if variant == "prior_event" and channel == "T0" else [])]
            for minute in sorted(minutes):
                if variant == "wind_gap" and channel == "wind_speed" and -2 <= minute <= 2: continue
                stamp = center + timedelta(minutes=minute)
                available = center + timedelta(minutes=30) if variant == "late_neighbor" and channel == "T1" else stamp
                rows.append({"id": str(uuid4()), "point_id": point["point_id"], "source_id": source_id,
                    "event_time": utcstr(stamp), "available_at": utcstr(available), "value": _value(channel, minute, variant),
                    "raw_value": str(_value(channel, minute, variant)), "quality_flags": [], "import_id": source_id})
        csv_path = runtime / "synthetic-observations.csv"
        csv_path.write_text("point_id,event_time,available_at,value\n" + "\n".join(
            f"{row['point_id']},{row['event_time']},{row['available_at']},{row['value']}" for row in rows), encoding="utf-8")
        file = services["data"].register_file(csv_path, actor)
        for index, row in enumerate(rows, 2): row.update(source_file_id=file["file_id"], row_number=index)
        with services["store"].transaction() as connection:
            connection.execute(insert(observations), rows)
        initial_time = min(datetime.fromisoformat(row["event_time"]) for row in rows)
        session = services["sessions"].create_session(actor, "historical_replay", source_id, initial_time)
        services["sessions"].apply_action(actor, session["session_id"], "advance", target_as_of=center + timedelta(minutes=10))
        context = services["sessions"].get_context(actor, session["session_id"])
        services["alerts"].evaluate_observed(context, source_id + ":T0")
        events = services["alerts"].query_events(context, point_id=source_id + ":T0")
        if not events: raise RuntimeError("synthetic_event_not_created")
        event = min(events, key=lambda row: abs((datetime.fromisoformat(row["start_at"]) - center).total_seconds()))
        prepared = {"scenario_seed_version": 2, "scenario_id": identifier, "scenario": description, "variant": variant, "task_goal": goal,
                    "expected_tools": ["get_event_evidence", expected], "expected_model_tool": expected,
                    "runtime_dir": str(runtime), "source_id": source_id, "event": event,
                    "context": context.model_dump(mode="json"), "synthetic_observation_count": len(rows),
                    "synthetic_csv_sha256": file["sha256"]}
        _write(checkpoint, prepared)
        return prepared
    finally:
        services["store"].close()


def _scenario_check(prepared, report):
    variant, expected = prepared["variant"], prepared["expected_model_tool"]
    step = next((row for row in report.get("investigation_steps", []) if row["tool"] == expected), None)
    if not step: return False
    missing, metrics = set(step.get("missing_fields", [])), step.get("metrics", {})
    if variant == "single_point": return {"T1", "T2", "T5"} <= missing
    if variant == "missing_wind": return "wind_speed" in missing
    if variant == "missing_motion": return {"speed", "position"} <= missing
    if variant == "late_neighbor": return "T1" in missing
    if variant == "no_group": return "dataset_group" in missing
    if variant == "no_history": return "historical_similar_events" in missing
    if variant == "prior_event": return bool(metrics.get("events")) and "reviewed_conclusions" in missing
    if variant == "wind_gap": return any(row.get("wind_speed") is None for row in metrics.get("timeline", []))
    if variant == "short_spike": return bool(prepared["event"].get("end_at"))
    if expected == "compare_points": return len(metrics.get("points", [])) >= 2
    if expected == "get_condition_timeline": return bool(metrics.get("timeline"))
    return bool(metrics.get("groups"))


def _run_case(scenario, output_dir, base, rerun_failed=False):
    directory, destination = output_dir / scenario[0], output_dir / scenario[0] / "result.json"
    if destination.is_file():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        if not rerun_failed or previous.get("task_passed"):
            return previous
    prepared = _prepare(scenario, directory, base, force=destination.is_file() and rerun_failed)
    context = ExecutionContext(**prepared["context"])
    services = build_services(_configuration(base, Path(prepared["runtime_dir"]), prepared["source_id"]))
    api_log, started = [], perf_counter()
    try:
        services["investigation"].adapter_factory = lambda settings: LoggedDeepSeekAdapter(settings, api_log)
        existing = services["investigation"].for_event(context, prepared["event"]["event_id"])["report"]
        if existing and existing["state"] not in ("queued", "running"):
            report = existing
        elif existing and existing["state"] == "running":
            report = {**existing, "state": "failed", "error_code": "interrupted_previous_attempt"}
        else:
            queued = existing or services["investigation"].submit(context, prepared["event"]["event_id"], task_goal=prepared["task_goal"])
            report = services["investigation"].run(queued["report_id"])
        _write(directory / "report.json", report)
        steps, hypotheses = report.get("investigation_steps", []), report.get("hypotheses", [])
        selected = {name for call in api_log for name in call.get("selected_tools", [])}
        known = {row["reference_id"] for row in report.get("evidence_refs", [])}
        valid_refs = bool(hypotheses) and all(row.get("evidence_refs") and all(ref.get("reference_id") in known for ref in row["evidence_refs"]) for row in hypotheses)
        actual = {"report": {"state": report["state"], "event_id": report["event_id"]},
                  "report_schema_valid": all(key in report for key in ("report_id", "event_id", "event_revision", "hypotheses", "pending_items", "evidence_refs")),
                  "hypotheses_have_valid_references": valid_refs, "model_selected_required_tool": prepared["expected_model_tool"] in selected,
                  "scenario_requirement_verified": _scenario_check(prepared, report), "no_approval": report["state"] == "draft"}
        expected = {"report.state": "draft", "report.event_id": prepared["event"]["event_id"], "report_schema_valid": True,
                    "hypotheses_have_valid_references": True, "model_selected_required_tool": True,
                    "scenario_requirement_verified": True, "no_approval": True}
        task = {"task_id": "acceptance-" + prepared["scenario_id"], "scenario": prepared["scenario"],
                "expected_tools": prepared["expected_tools"], "expected_outputs": expected, "actual_output": actual,
                "tool_calls": [{"name": row["tool"], "status": "success" if row.get("status") in ("success", "completed", "partial", "not_ready") else "error",
                                "result_status": row.get("status"), "selected_by": "application" if index == 0 else "deepseek"}
                               for index, row in enumerate(steps)]}
        evidence = services["traceability"].get_event_evidence(context, prepared["event"]["event_id"], prepared["event"]["event_revision"])
        event = {**prepared["event"], "observation_refs": evidence.get("evidence_refs", []),
                 "model_or_rule_refs": [{"kind": "rule", "rule_version": evidence["rules"]["rule_version"]}]}
        check = build_acceptance([task], [event], [report], [])
        failed_checks = [row["path"] for row in check["tool_collaboration"]["per_task"][0]["output_checks"] if not row["passed"]]
        result = {"scenario_id": prepared["scenario_id"], "scenario": prepared["scenario"], "task_goal": prepared["task_goal"],
                  "synthetic_only": True, "runtime_dir": prepared["runtime_dir"], "task": task, "event": event,
                  "report_id": report["report_id"], "state": report["state"], "model": report.get("model"),
                  "task_passed": bool(check["tool_collaboration"]["numerator"]), "actual_tools": [row["tool"] for row in steps],
                  "model_selected_tools": sorted(selected), "failure_reason": report.get("error_code") or failed_checks,
                  "api_calls": len(api_log), "elapsed_seconds": round(perf_counter() - started, 3), "approval_actions": 0}
        _write(directory / "outbound-fields.json", {"api_calls": api_log, "disclosures": services["store"].list("cloud_disclosure")})
        _write(destination, result)
        return result
    except Exception as error:
        result = {"scenario_id": scenario[0], "scenario": scenario[1], "synthetic_only": True,
                  "state": "failed", "task_passed": False, "failure_reason": getattr(error, "code", type(error).__name__),
                  "api_calls": len(api_log), "elapsed_seconds": round(perf_counter() - started, 3), "approval_actions": 0,
                  "task": {"task_id": "acceptance-" + scenario[0], "expected_tools": prepared["expected_tools"],
                           "expected_outputs": {"report.state": "draft"}, "actual_output": {}, "tool_calls": []}}
        _write(directory / "outbound-fields.json", {"api_calls": api_log})
        _write(destination, result)
        return result
    finally:
        services["store"].close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="21项独立合成场景真实DeepSeek工具协同验收")
    parser.add_argument("--output-dir", type=Path, default=Path("F:/煤矿瓦斯智能体/outputs/source-control-20261105/acceptance"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--env-file",type=Path,default=Path(os.environ.get("COAL_AGENT_ENV_FILE",PROJECT_ROOT/".env")))
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.drive.upper() != "F:":
        raise ValueError("验收环境与产物必须放F盘")
    base = Settings(_env_file=args.env_file)
    if not args.prepare_only and not base.deepseek_configured:
        raise ValueError("deepseek_not_configured")
    manifest = {"synthetic_only": True, "independent_scenario_count": len(_SCENARIOS), "max_workers": 3,
                "timeout_per_api_call_seconds": 45, "max_tokens": 1800, "approval_actions": 0,
                "scenarios": [{"scenario_id": row[0], "scenario": row[1], "task_goal": row[2],
                               "expected_tools": ["get_event_evidence", row[3]], "variant": row[4]} for row in _SCENARIOS]}
    _write(output / "manifest.json", manifest)
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {}
        for scenario in _SCENARIOS:
            future = (executor.submit(_prepare, scenario, output / scenario[0], base) if args.prepare_only
                      else executor.submit(_run_case, scenario, output, base, args.rerun_failed))
            futures[future] = scenario[0]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({"scenario_id": futures[future], "state": "prepared" if args.prepare_only else result["state"],
                              "task_passed": result.get("task_passed"), "actual_tools": result.get("actual_tools", [])}, ensure_ascii=False), flush=True)
    if not args.prepare_only:
        results.sort(key=lambda row: row["scenario_id"])
        summary = build_acceptance([row["task"] for row in results], [row["event"] for row in results if row.get("event")], [], [])
        summary.update(synthetic_only=True, approval_actions=0, scenario_count=len(results),
                       actual_api_call_count=sum(row.get("api_calls", 0) for row in results),
                       measure_coverage_status="not_evaluable_no_approved_plans",
                       failures=[{"scenario_id": row["scenario_id"], "reason": row.get("failure_reason")} for row in results if not row["task_passed"]])
        _write(output / "acceptance.json", summary)
        print(json.dumps({"completed": True, "task_numerator": summary["tool_collaboration"]["numerator"],
                          "task_denominator": summary["tool_collaboration"]["denominator"],
                          "evidence_numerator": summary["evidence_completeness"]["numerator"],
                          "evidence_denominator": summary["evidence_completeness"]["denominator"],
                          "api_calls": summary["actual_api_call_count"], "approval_actions": 0}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
