"""Small offline checks for scoped investigation evidence and idempotent imports."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import insert, select, func

from app.db import Store, observations
from app.schemas.common import Actor, ExecutionContext, utcstr
from app.services.traceability.evidence_tools import EvidenceToolsService
from app.scripts.import_investigation_history import run_import


BASE = datetime(2022, 4, 27, 3, 0, tzinfo=timezone(timedelta(hours=8)))


@pytest.fixture
def parts():
    store = Store("sqlite://")
    for source, group in (("anchor", "group"), ("neighbor", "group"), ("foreign", "other")):
        store.put("source", source, {"source_id": source, "face_id": "face", "metadata": {"dataset_group": group}})
    for point, source, channel in (("t0", "anchor", "T0"), ("t1", "neighbor", "T1"),
                                    ("foreign", "foreign", "T2"), ("speed", "neighbor", "speed"),
                                    ("current", "neighbor", "current_left"), ("position", "neighbor", "position"),
                                    ("wind", "neighbor", "wind_speed")):
        store.put("point", point, {"point_id": point, "source_id": source, "face_id": "face", "unit": "%",
                                   "metadata": {"channel": channel}, "lineage": "processed", "unit_status": "inferred"})
    actor = Actor(user_id="u", roles=("safety_management",), face_ids=("face",))
    context = ExecutionContext(session_id="session", source_id="anchor", mode="historical_replay", actor=actor, as_of=BASE + timedelta(minutes=20))
    event = {"event_id": "event", "point_id": "t0", "source_id": "anchor", "session_id": "session",
             "event_type": "observed_exceedance", "alert_level": "orange", "start_at": utcstr(BASE + timedelta(minutes=10)),
             "event_time": utcstr(BASE + timedelta(minutes=10)), "revision_as_of": utcstr(BASE + timedelta(minutes=15)), "rule_version": "rule"}
    rows = []

    def reading(point, minute, value, available=None):
        source = store.get("point", point)["source_id"]
        rows.append({"id": f"{point}-{minute}-{len(rows)}", "point_id": point, "source_id": source,
            "event_time": utcstr(BASE + timedelta(minutes=minute)), "available_at": utcstr(BASE + timedelta(minutes=minute if available is None else available)),
            "value": value, "raw_value": str(value), "quality_flags": [], "source_file_id": "file", "row_number": len(rows)+2, "import_id": "import"})
    for point, peak_minute in (("t0", 10), ("t1", 12)):
        reading(point, 0, 0.1); reading(point, peak_minute, 0.9); reading(point, 15, 0.2)
    reading("t1", 11, 8, available=30)  # known only after the execution clock
    reading("t1", 25, 9)  # future measurement
    reading("foreign", 10, 20)
    for minute in (0, 10, 12, 15):
        reading("speed", minute, 2 if minute else 0)
        reading("current", minute, 100 if minute else 0)
        reading("position", minute, 100+minute)
        reading("wind", minute, 1.5)
    with store.transaction() as connection:
        connection.execute(insert(observations), rows)
    alerts = SimpleNamespace(get_event_snapshot=lambda context, event_id: dict(event),
        get_rules=lambda *args: {"status": "team-draft", "observed_orange_threshold": 0.65})
    service = EvidenceToolsService(store, {"alerts": alerts})
    yield store, context, event, service
    store.close()


def test_neighbor_comparison_enforces_dataset_group_and_both_clocks(parts):
    _, context, _, service = parts
    result = service.compare_points(context, "event")
    by_channel = {item["channel"]: item for item in result["metrics"]["points"]}
    assert set(by_channel) == {"T0", "T1"}
    assert by_channel["T1"]["peak"] == 0.9
    assert by_channel["T1"]["peak_delay_seconds"] == 120
    assert result["missing_fields"] == ["T2", "T5"]
    assert result["status"] == "partial" and result["evidence_refs"]


def test_condition_timeline_reports_proxy_and_missing_actual_cutting_state(parts):
    _, context, _, service = parts
    result = service.get_condition_timeline(context, "event")
    assert "cutting_state" in result["missing_fields"]
    assert result["metrics"]["state_basis"] == "speed_and_motor_current_proxy"
    assert any(row["operating_state"] == "moving_with_current" for row in result["metrics"]["timeline"])
    assert result["metrics"]["channels"]["position"]["maximum"] == 115


def test_face_statistics_uses_existing_rule_and_identifiable_state(parts):
    _, context, _, service = parts
    result = service.get_face_statistics(context, "event")
    assert result["metrics"]["threshold"] == 0.65 and result["metrics"]["rule_status"] == "team-draft"
    assert sum(item["sample_count"] for item in result["metrics"]["groups"]) == 3
    assert sum(item["high_count"] for item in result["metrics"]["groups"]) == 1


def test_similar_history_only_uses_visible_same_group_reviewed_conclusions(parts):
    store, context, event, service = parts
    prior = {**event, "event_id": "previous", "start_at": utcstr(BASE-timedelta(days=1)), "event_time": utcstr(BASE-timedelta(days=1)),
             "revision_as_of": utcstr(BASE-timedelta(days=1)), "event_revision": 1}
    store.put("event_revision", "previous@1", prior)
    store.put("reviewed_case_snapshot", "case", {"event_id": "previous", "source_id": "anchor", "face_id": "face",
        "data_as_of": utcstr(BASE-timedelta(days=1)), "reviewed_at": utcstr(BASE-timedelta(hours=1)),
        "operation_context": "field_record", "review_action_ref": "review", "actions": [
            {"action_type": "review_case", "content": "已复核的历史结论"}]})
    result = service.find_similar_events(context, "event")
    assert result["metrics"]["events"][0]["reviewed_conclusion"] == "已复核的历史结论"
    snapshot = store.get("reviewed_case_snapshot", "case")
    snapshot["reviewed_at"] = utcstr(context.as_of+timedelta(days=1))
    store.put("reviewed_case_snapshot", "case", snapshot)
    assert "reviewed_conclusions" in service.find_similar_events(context, "event")["missing_fields"]


def test_missing_dataset_binding_returns_missing_instead_of_cross_source_data(parts):
    store, context, _, service = parts
    source = store.get("source", "anchor"); source.pop("metadata")
    store.put("source", "anchor", source)
    result = service.compare_points(context, "event")
    assert result["status"] == "not_ready" and "dataset_group" in result["missing_fields"]


def test_daily_statistics_and_approved_history_respect_data_clock_and_rehearsal_scope(parts):
    store, context, event, service = parts
    stats = service.get_face_statistics(context, "event")
    assert stats["metrics"]["window_basis"] == "past_24_hours"
    assert datetime.fromisoformat(stats["metrics"]["window_start"]) == context.as_of-timedelta(hours=24)
    prior = {**event, "event_id": "old-investigation", "source_id": "neighbor", "start_at": utcstr(BASE-timedelta(days=1)),
             "event_time": utcstr(BASE-timedelta(days=1)), "revision_as_of": utcstr(BASE-timedelta(days=1)), "event_revision": 1}
    store.put("event_revision", "old-investigation@1", prior)
    report = {"report_id": "approved-report", "source_id": "neighbor", "face_id": "face", "event_id": prior["event_id"],
        "event_overview": prior, "as_of": utcstr(BASE-timedelta(hours=12)), "state": "approved", "operation_context": "rehearsal",
        "review": {"decision": "approved", "data_as_of": utcstr(BASE-timedelta(hours=11))},
        "hypotheses": [{"cause": "通风系统异常", "support_status": "supported", "rationale": "已审历史结论"}]}
    store.put("investigation_report", "approved-report", report)
    store.put("control_plan", "plan", {"plan_id": "plan", "event_id": prior["event_id"], "source_id": "neighbor", "face_id": "face"})
    experience = {"experience_id": "experience", "plan_id": "plan", "source_id": "neighbor", "face_id": "face",
        "state": "approved", "operation_context": "rehearsal", "effect": {"as_of": utcstr(BASE-timedelta(hours=9)), "status": "effective"},
        "cause_ids": ["ventilation_abnormal"], "reference_ids": ["approved-ref"],
        "history": [{"action": "review_experience", "data_as_of": utcstr(BASE-timedelta(hours=8))}]}
    store.put("control_experience", "experience", experience)
    service.services["knowledge"] = SimpleNamespace(_approved_fragment=lambda *args: {"reference_id": "approved-ref"})
    result = service.find_similar_events(context, "event")
    assert result["metrics"]["events"][0]["reviewed_source"] == "investigation_report"
    assert result["metrics"]["events"][0]["conclusions"][0]["cause"] == "通风系统异常"
    assert result["metrics"]["experience_count"] == 1
    assert any(ref.get("reference_id") == "approved-ref" for ref in result["evidence_refs"])
    assert service.find_similar_events(context.model_copy(update={"mode": "realtime"}), "event")["metrics"]["experience_count"] == 0
    report["review"]["data_as_of"] = utcstr(context.as_of+timedelta(days=1))
    experience["history"][0]["data_as_of"] = utcstr(context.as_of+timedelta(days=1))
    store.put("investigation_report", "approved-report", report)
    store.put("control_experience", "experience", experience)
    invisible = service.find_similar_events(context, "event")
    assert invisible["metrics"]["experience_count"] == 0
    assert "reviewed_conclusions" in invisible["missing_fields"]


def test_import_keeps_source_read_only_and_deduplicates_overlapping_windows(tmp_path):
    source_root = tmp_path / "source"; folder = source_root / "dataset" / "处理后数据" / "T0_2022"
    folder.mkdir(parents=True)
    path = folder / "T0_resample_15s.csv"
    raw = "STime,t0_max,t0_min,t0_mean\n2022-04-27 03:00:00,0.8,0.7,0.75\n2022-04-27 03:00:15,0.9,0.8,0.85\n"
    path.write_text(raw, encoding="utf-8")
    database = f"sqlite+pysqlite:///{(tmp_path/'test.db').as_posix()}"
    start, end = BASE, BASE+timedelta(minutes=1)
    first = run_import(source_root, database, tmp_path/"output", start, end, channels=("T0",), face_id="face", dataset_group="group")
    initial_store = Store(database)
    protected_file = initial_store.list("data_file")[0]
    protected_file["owner_id"] = "previous-uploader"
    protected_file["existing_record"] = True
    initial_store.put("data_file", protected_file["file_id"], protected_file)
    initial_store.close()
    second = run_import(source_root, database, tmp_path/"output", start, end, channels=("T0",), face_id="face", dataset_group="group")
    assert first["imported_rows"] == 2 and second["imported_rows"] == 0
    assert path.read_text(encoding="utf-8") == raw
    store = Store(database)
    assert store.get("data_file", protected_file["file_id"]) == protected_file
    with store.transaction() as connection:
        assert connection.execute(select(func.count()).select_from(observations)).scalar_one() == 2
        row = connection.execute(select(observations).order_by(observations.c.event_time)).mappings().first()
    assert datetime.fromisoformat(row["available_at"]) - datetime.fromisoformat(row["event_time"]) == timedelta(seconds=15)
    assert store.list("source")[0]["metadata"]["dataset_group"] == "group"
    store.close()
