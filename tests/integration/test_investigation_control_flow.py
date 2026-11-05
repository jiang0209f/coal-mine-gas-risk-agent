"""One complete workflow; explicit reviewer actions occur only in this isolated fixture."""
from datetime import datetime, timedelta, timezone
import json

import pytest

from app.adapters.deepseek import ChatReply
from app.config import Settings
from app.main import build_services
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.investigation.service import InvestigationService


def test_condition_group_labels_survive_the_outgoing_metric_whitelist():
    metrics={"groups":[{"operating_state":"moving_with_current","high_ratio":.5},
                       {"operating_state":"stationary_without_current","high_ratio":.1}]}
    assert InvestigationService._safe_metrics(metrics)==metrics


def test_missing_real_cutting_state_cannot_exclude_cutting_from_proxy_data():
    report={"investigation_steps":[{"tool":"get_condition_timeline","missing_fields":["cutting_state"]}],
            "evidence_refs":[{"reference_id":"EV1","kind":"tool_result"}]}
    InvestigationService._finish(report,{"hypotheses":[{"cause":"割煤活动导致瓦斯升高", "support_status":"opposed",
        "rationale":"速度和电流均为零。", "reference_ids":["EV1"]}],"pending_items":[]})
    assert report["hypotheses"][0]["support_status"] == "undetermined"
    assert report["hypotheses"][0]["model_support_status"] == "opposed"
    assert not report["excluded_items"]
    assert report["model_hypotheses"][0]["support_status"] == "opposed"


def test_event_to_reviewed_experience_and_next_event_plan(tmp_path):
    s = build_services(Settings(_env_file=None, runtime_dir=tmp_path,
        database_url="sqlite+pysqlite:///:memory:", deepseek_api_key="offline-only",
        agent_real_data_mode="summary", agent_real_source_ids=["workflow-source"],
        ingestion_worker_enabled=False, realtime_monitor_enabled=False))
    try:
        actor = Actor(user_id="fixture-reviewer", roles=("safety_management", "ventilation", "gas_governance"), face_ids=("FLOW",))
        base = datetime(2022, 4, 27, 0, 0, tzinfo=timezone.utc)
        s["data"].register_source("workflow-source", "流程夹具", "FLOW")
        s["data"].register_point("flow-t0", "workflow-source", "T0", "T0", "methane", "%", unit_status="confirmed")
        for minute in range(16):
            s["data"].ingest_readings(actor, "workflow-source", [dict(reading_id=str(minute),
                point_id="flow-t0", event_time=base+timedelta(minutes=minute),
                raw_value=.9 if minute < 6 else .2)], received_at=base+timedelta(minutes=minute))
        context = ExecutionContext(actor=actor, session_id="flow-session", source_id="workflow-source",
            mode="historical_replay", as_of=base+timedelta(minutes=5))
        s["alerts"].evaluate_observed(context, "flow-t0")
        event = s["alerts"].query_events(context)[0]
        document = s["knowledge"].import_reference(actor, dict(title="隔离流程依据", source="软件夹具",
            version="1", applicability={"face_ids":["FLOW"]}, fragments=[dict(
                text="通风异常时核查通风设施并连续监测甲烷，由通风部门负责。", locator={"page":1})]))
        s["knowledge"].review_reference(actor, document["document_id"], "1", "approved", "夹具明确审核动作")
        refs = s["knowledge"].search_documents(context, "通风异常")
        measures = s["control"].import_measures(context, [refs[0]["reference_id"]])
        for measure in measures:
            assert measure["state"] != "approved"
            s["control"].review_measure(context, measure["measure_id"], "approved", "夹具审核")

        class Adapter:
            def chat(self, messages, **kwargs):
                return ChatReply(dict(role="assistant", content=json.dumps({
                    "hypotheses":[{"cause":"通风异常", "support_status":"supported", "rationale":"夹具明确提供的排查结论",
                                   "reference_ids":["EV1"]}], "pending_items":[]})), "stop", "fixture-api", {})
            def close(self): pass
        s["investigation"].adapter_factory = lambda settings: Adapter()
        pending = []
        submitted = s["investigation"].submit(context, event["event_id"], schedule=pending.append)
        pending[0]()
        report = s["investigation"].get(context, submitted["report_id"])
        assert report["state"] == "draft" and report["supported_causes"]
        with pytest.raises(DomainError):
            s["control"].draft_control_plan(context, event["event_id"], report["report_id"])
        s["investigation"].review(context, report["report_id"], "approved", "夹具人工签发动作")
        plan = s["control"].draft_control_plan(context, event["event_id"], report["report_id"])
        assert plan["state"] == "draft"
        s["control"].review_plan(context, plan["plan_id"], "approved", "夹具人工批准")
        workflow = s["control"].get_workflow(context, plan["plan_id"])
        assert workflow["tasks"]
        execution_context=context.model_copy(update={"as_of":base+timedelta(minutes=6)})
        for task in workflow["tasks"]:
            s["control"].record_execution(execution_context, task["task_id"], "夹具登记执行")
        after = context.model_copy(update={"as_of":base+timedelta(minutes=15)})
        effect = s["control"].verify_control_effect(after, plan["plan_id"])
        assert effect["status"] in ("effective", "见效")
        completed = s["control"].review_completion(after, plan["plan_id"], "夹具复核")
        experience = completed["experience"]
        assert experience["state"] != "approved"
        s["control"].review_experience(after, experience["experience_id"], "approved", "夹具经验审核")
        next_event = {**event, "event_id":"next-event", "revision_as_of":after.as_of.isoformat()}
        s["store"].put("event", "next-event", next_event)
        s["store"].put("event_revision", "next-event:1", next_event)
        next_report = {**s["store"].get("investigation_report", report["report_id"]),
                       "report_id":"next-report", "event_id":"next-event", "as_of":after.as_of.isoformat()}
        s["store"].put("investigation_report", "next-report", next_report)
        next_plan = s["control"].draft_control_plan(after, "next-event", "next-report")
        assert experience["experience_id"] in next_plan["experience_ids"]
    finally:
        s["store"].close()
