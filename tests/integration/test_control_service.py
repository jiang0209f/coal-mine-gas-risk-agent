"""The minimum reviewed-measures to execution/effect/experience chain."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.control.service import ControlService


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    now = datetime(2026, 11, 5, 10, tzinfo=timezone.utc)
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("face",))
    context = ExecutionContext(source_id="source", session_id="session", actor=actor,
                               mode="historical_replay", as_of=now)
    store.put("source", "source", {"source_id": "source", "face_id": "face"})
    event = {"event_id": "event", "source_id": "source", "session_id": "session", "face_id": "face",
             "event_revision": 1, "revision_as_of": now.isoformat(), "point_id": "point", "rule_version": "draft"}
    report = {"report_id": "report", "event_id": "event", "source_id": "source", "session_id": "session",
              "face_id": "face", "event_revision": 1,
              "state": "approved", "supported_causes": [{"cause": "通风系统异常", "support_status": "supported"}], "current_conditions": {}}
    store.put("investigation_report", "report", report)
    fragment = {"reference_id": "ref", "entry_id": "doc", "version": 1, "text": "通风系统异常时，由通风部门负责，核查通风设施并按规程恢复正常通风。",
                "locator": {"page": 4}, "source": "已审核规程", "applicability": {"face_ids": ["face"]}}
    samples = [{"id": str(index), "value": .9 if index < 5 else .4,
                "event_time": (now + timedelta(minutes=index - 5)).isoformat(), "quality_flags": [],
                "source_file_id": "file", "row_number": index + 1} for index in range(10)]

    def series(context, point_id, start_at, end_at, **kwargs):
        return {"rows": [row for row in samples if start_at <= datetime.fromisoformat(row["event_time"]) < end_at], "total": 5}

    services = {
        "knowledge": SimpleNamespace(_approved_fragment=lambda *args, **kwargs: fragment),
        "alerts": SimpleNamespace(get_event_snapshot=lambda *args, **kwargs: dict(event),
            get_rules=lambda *args, **kwargs: {"observed_red_threshold": .8, "status": "team-draft", "unit": "%"}),
        "data": SimpleNamespace(query_series=series),
    }
    service = ControlService(store, services)
    yield SimpleNamespace(store=store, context=context, service=service, now=now, samples=samples, fragment=fragment)
    store.close()


def reviewed_plan(parts):
    measure = parts.service.import_measures(parts.context, ["ref"])[0]
    assert measure["state"] == "pending"
    parts.service.review_measure(parts.context, measure["measure_id"], "approved", "核对原文及适用条件")
    plan = parts.service.draft_control_plan(parts.context, "event", "report")
    assert plan["state"] == "draft"
    return parts.service.review_plan(parts.context, plan["plan_id"], "approved", "确认本事件适用")


def test_reviewed_chain_generates_pending_experience_then_reuses_only_after_review(parts):
    workflow = reviewed_plan(parts)
    task = workflow["tasks"][0]
    assert task["assigned_role"] == "ventilation" and task["due_at"] is None
    assert task["deadline"] == "待矿方确定"
    executed = parts.service.record_execution(parts.context, task["task_id"], "完成核查并记录结果", parts.now)
    assert executed["tasks"][0]["state"] == "executed"
    later = parts.context.model_copy(update={"as_of": parts.now + timedelta(minutes=5)})
    effect = parts.service.verify_control_effect(later, workflow["plan"]["plan_id"])
    assert effect["status"] == "effective"
    assert effect["after"]["maximum"] == .4 and effect["before"]["mean"] == .9
    assert effect["observation_refs"] and effect["semantics"] == "observational_comparison"
    reviewed = parts.service.review_completion(later, workflow["plan"]["plan_id"], "复核观测及执行记录")
    experience = reviewed["experience"]
    assert experience["state"] == "pending"
    parts.service.review_experience(later, experience["experience_id"], "approved", "核对归档")
    next_plan = parts.service.draft_control_plan(later, "event", "report")
    assert experience["experience_id"] in next_plan["experience_ids"]


def test_unreviewed_measure_and_plan_never_create_execution_tasks(parts):
    parts.service.import_measures(parts.context, ["ref"])
    plan = parts.service.draft_control_plan(parts.context, "event", "report")
    assert plan["items"] == [] and plan["missing_fields"]
    with pytest.raises(DomainError):
        parts.service.review_plan(parts.context, plan["plan_id"], "approved", "")
    assert parts.store.list("control_task") == []


def test_replay_task_deadlines_and_overdue_use_the_data_clock(parts):
    fragment=parts.fragment
    fragment["text"] += "在3分钟内完成核查。"
    workflow=reviewed_plan(parts)
    task=workflow["tasks"][0]
    assert datetime.fromisoformat(task["due_at"]) == parts.context.as_of+timedelta(minutes=3)
    assert task["overdue"] is False
    later=parts.context.model_copy(update={"as_of":parts.context.as_of+timedelta(minutes=4)})
    assert parts.service.get_workflow(later,workflow["plan"]["plan_id"])["tasks"][0]["overdue"] is True


def test_effect_requires_post_execution_observations_and_manual_review_role(parts):
    workflow = reviewed_plan(parts)
    task = workflow["tasks"][0]
    parts.service.record_execution(parts.context, task["task_id"], "已执行", parts.now)
    effect = parts.service.verify_control_effect(parts.context, workflow["plan"]["plan_id"])
    assert effect["status"] == "insufficient_data"
    with pytest.raises(DomainError):
        parts.service.review_completion(parts.context, workflow["plan"]["plan_id"], "尝试复核")
    worker = parts.context.actor.model_copy(update={"roles": ("ventilation",)})
    with pytest.raises(DomainError):
        parts.service.review_plan(parts.context.model_copy(update={"actor": worker}), workflow["plan"]["plan_id"], "approved", "")


def test_report_requires_human_approval_and_measure_revocation_blocks_execution(parts):
    report = parts.store.get("investigation_report", "report")
    report["state"] = "draft"
    parts.store.put("investigation_report", "report", report)
    with pytest.raises(DomainError):
        parts.service.draft_control_plan(parts.context, "event", "report")
    report["state"] = "approved"
    parts.store.put("investigation_report", "report", report)
    workflow = reviewed_plan(parts)
    parts.service.review_measure(parts.context, workflow["plan"]["items"][0]["measure_id"], "rejected", "资料需重新核对")
    with pytest.raises(DomainError):
        parts.service.record_execution(parts.context, workflow["tasks"][0]["task_id"], "尝试执行")


def test_higher_tier_is_a_reviewed_escalation_candidate_after_ineffective_result(parts):
    low = parts.service.import_measures(parts.context, ["ref"])[0]
    parts.service.review_measure(parts.context, low["measure_id"], "approved", "常规措施", tier=1)
    other = dict(parts.fragment, reference_id="ref-higher", text="通风系统异常时，由通风部门负责，按专项措施强化通风核查。")
    parts.service.services["knowledge"]._approved_fragment = lambda identifier, *args, **kwargs: other if identifier == "ref-higher" else parts.fragment
    high = parts.service.import_measures(parts.context, ["ref-higher"])[0]
    parts.service.review_measure(parts.context, high["measure_id"], "approved", "专项升级措施", tier=2)
    plan = parts.service.draft_control_plan(parts.context, "event", "report")
    assert [item["tier"] for item in plan["items"]] == [1]
    workflow = parts.service.review_plan(parts.context, plan["plan_id"], "approved", "先执行常规措施")
    parts.service.record_execution(parts.context, workflow["tasks"][0]["task_id"], "已执行", parts.now)
    for row in parts.samples[5:]:
        row["value"] = .95
    later = parts.context.model_copy(update={"as_of": parts.now + timedelta(minutes=5)})
    effect = parts.service.verify_control_effect(later, plan["plan_id"])
    assert effect["status"] == "ineffective"
    assert effect["escalation_state"] == "awaiting_human_decision"
    assert [row["measure_id"] for row in effect["escalation_candidates"]] == [high["measure_id"]]


def test_background_effect_worker_checks_once_per_data_clock_and_never_reviews(parts):
    workflow = reviewed_plan(parts)
    parts.service.record_execution(parts.context, workflow["tasks"][0]["task_id"], "已执行", parts.now)
    later = parts.context.model_copy(update={"as_of": parts.now + timedelta(minutes=5)})
    parts.service.services["sessions"] = SimpleNamespace(get_context=lambda actor, identifier: later.model_copy(update={"actor": actor}))
    assert parts.service.process_pending_effects()["processed"] == 1
    assert parts.service.process_pending_effects()["waiting"] == 1
    assert parts.service.get_workflow(later, workflow["plan"]["plan_id"])["plan"]["state"] == "approved"
    system = later.model_copy(update={"actor": later.actor.model_copy(update={"is_system": True})})
    for method, arguments in (("review_measure", ("missing", "approved")), ("review_plan", ("missing", "approved")),
                              ("review_completion", ("missing", "复核")), ("review_experience", ("missing", "approved"))):
        with pytest.raises(DomainError):
            getattr(parts.service, method)(system, *arguments)


def test_pending_reference_can_be_collected_but_cannot_approve_measure(parts):
    fragment = dict(parts.fragment, reference_id="pending-ref", entry_id="pending-doc", text="通风异常时应核查风量。")
    parts.store.put("reference_fragment", "pending-ref", fragment)
    parts.store.put("reference", "pending-doc:1", {"entry_id": "pending-doc", "version": 1,
        "state": "pending_review", "applicability": {"face_ids": ["face"]}})

    def unapproved(*args, **kwargs):
        raise DomainError("资料尚未审核")

    parts.service.services["knowledge"]._approved_fragment = unapproved
    measure = parts.service.import_measures(parts.context, ["pending-ref"])[0]
    assert measure["state"] == "pending" and measure["reference_state"] == "pending_review"
    assert measure["assigned_role"] is None and measure["role"] == "待矿方确定"
    with pytest.raises(DomainError):
        parts.service.review_measure(parts.context, measure["measure_id"], "approved", "尝试审核")


def test_same_level_event_end_keeps_frozen_approved_plan_reviewable(parts):
    workflow = reviewed_plan(parts)
    parts.service.record_execution(parts.context, workflow["tasks"][0]["task_id"], "已执行", parts.now)
    frozen = parts.service.services["alerts"].get_event_snapshot(parts.context, "event")
    ended = dict(frozen, event_revision=2, event_state="ended", alert_level="orange")
    frozen["alert_level"] = "orange"
    parts.service.services["alerts"].get_event_snapshot = lambda context, event_id, event_revision=None: dict(frozen if event_revision == 1 else ended)
    later = parts.context.model_copy(update={"as_of": parts.now + timedelta(minutes=5)})
    parts.service.verify_control_effect(later, workflow["plan"]["plan_id"])
    result = parts.service.review_completion(later, workflow["plan"]["plan_id"], "事件同级结束，复核执行后观测")
    assert result["plan"]["state"] == "completed"
    assert result["plan"]["event_revision"] == 1


def test_event_grade_upgrade_blocks_execution_of_previously_approved_plan(parts):
    workflow = reviewed_plan(parts)
    frozen = parts.service.services["alerts"].get_event_snapshot(parts.context, "event")
    frozen["alert_level"] = "orange"
    upgraded = dict(frozen, event_revision=2, alert_level="red")
    parts.service.services["alerts"].get_event_snapshot = lambda context, event_id, event_revision=None: dict(frozen if event_revision == 1 else upgraded)
    with pytest.raises(DomainError) as caught:
        parts.service.record_execution(parts.context, workflow["tasks"][0]["task_id"], "尝试沿用旧计划")
    assert caught.value.code == "event_risk_upgraded"


def test_local_accumulation_measure_matches_supported_report_wording(parts):
    parts.fragment["text"] = "当发生瓦斯积聚时，必须及时处理。"
    measure = parts.service.import_measures(parts.context, ["ref"])[0]
    assert "local_gas_accumulation" in measure["cause_ids"]
    parts.service.review_measure(parts.context, measure["measure_id"], "approved", "隔离夹具人工核对", assigned_role="gas_governance")
    report = parts.store.get("investigation_report", "report")
    report["supported_causes"] = [{"cause": "局部（本测点附近）瓦斯异常积聚或涌出，非全断面扩散", "support_status": "supported"}]
    parts.store.put("investigation_report", "report", report)
    plan = parts.service.draft_control_plan(parts.context, "event", "report")
    assert [item["measure_id"] for item in plan["items"]] == [measure["measure_id"]]
