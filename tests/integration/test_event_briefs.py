"""Proactive briefs freeze evidence, share event scope and keep model traffic bounded."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from app.adapters.deepseek import ChatReply, DeepSeekRequestError
from app.config import Settings
from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.agent.service import AgentService
from app.services.agent.briefs import EventBriefsService


def at(minute=0):
    return datetime(2022, 2, 5, 13, minute, tzinfo=timezone.utc)


class Adapter:
    def __init__(self):
        self.requests = []
        self.responses = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.requests.append(deepcopy(dict(messages=messages, **kwargs)))
        value = self.responses.pop(0) if self.responses else ChatReply(
            dict(role="assistant", content="事件达到当前规则阈值，核对监测及处置记录。[E1]"), "stop", "offline-test", {"total_tokens": 12})
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed = True


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    store.put("source", "source", dict(source_id="source", face_id="31002"))
    actor = Actor(user_id="u1", roles=("safety_management",), face_ids=("31002",))
    context = ExecutionContext(session_id="session", source_id="source", mode="historical_replay", as_of=at(15), actor=actor)
    events = {}
    for revision, minute, level in ((1, 10, "orange"), (2, 12, "orange"), (3, 14, "red")):
        events[revision] = dict(event_id="event", session_id="session", source_id="source", point_id="t0",
            event_revision=revision, revision_as_of=at(minute).isoformat(), start_at=at(5).isoformat(),
            event_time=at(minute).isoformat(), event_type="observed_exceedance", alert_level=level,
            rule_version="team-draft-v1", quality_flags=[], evidence_refs=[])
    calls, adapter = [], Adapter()

    def snapshot(context, event_id, event_revision=None):
        visible = [value for key,value in events.items() if (event_revision is None or key==event_revision)
                   and datetime.fromisoformat(value["revision_as_of"]) <= context.as_of]
        if event_id != "event" or not visible:
            raise DomainError("事件修订不存在", "not_found", 404)
        return deepcopy(visible[-1])

    def evidence(context, event_id, event_revision=None):
        current = snapshot(context,event_id,event_revision)
        calls.append(("evidence", event_revision, context.as_of))
        refs = [dict(id=f"input-{i}", point_id="t0", source_file_id="file", row_number=i+1,
                     event_time=at(0).isoformat(), available_at=at(0).isoformat(), raw_value="PRIVATE_RAW") for i in range(366)]
        return dict(event_snapshot=current, observation_fragments=[dict(refs[0],value=.7)],
            rules=dict(rule_version="team-draft-v1", status="team-draft", observed_orange_threshold=.65),
            candidate_attribution=dict(status="completed", run_id="attribution-1", model_version="svc-v1",
                candidate_causes=["候选工况"], input_refs=refs),
            private_path="F:/private/materials.py", evidence_refs=[dict(kind="observation",observation_id="input-0",source_file_id="file",row_number=1,event_time=at(0).isoformat())],
            forecast_run=None, missing_fields=[])

    services = {
        "alerts": SimpleNamespace(get_event_snapshot=snapshot),
        "traceability": SimpleNamespace(get_event_evidence=evidence),
        "knowledge": SimpleNamespace(search_documents=lambda **kwargs: [dict(reference_id="ref-1", entry_id="doc-1",document_id="doc-1",version=2,
            locator=dict(page=7), title="审核规程",source="矿井规程",text="已审核规程条文。", kind="document")]),
        "cases": SimpleNamespace(get_case=lambda **kwargs: dict(case=dict(phase="confirmed"),actions=[])),
        "prediction": SimpleNamespace(list_models=lambda context: []),
        "data": SimpleNamespace(list_points=lambda actor,source_id: []),
    }
    settings = Settings(_env_file=None,deepseek_api_key="test-only-key",event_brief_allowed_source_ids=["source"])
    services["agent"] = AgentService(store,services,settings,adapter_factory=lambda configuration: adapter)
    briefs = EventBriefsService(store,services,settings)
    yield SimpleNamespace(store=store,actor=actor,context=context,events=events,calls=calls,adapter=adapter,services=services,settings=settings,briefs=briefs)
    store.close()


def test_submit_is_nonblocking_frozen_and_repeated_version_schedules_once(parts):
    pending = []
    first = parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    second = parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    assert first["status"] == "queued"
    assert first["task_id"] == second["task_id"] and len(pending)==1
    assert parts.calls == [] and parts.adapter.requests == []
    pending[0]()
    result = parts.briefs.get(parts.context,"event",1)
    assert result["status"]=="completed"
    assert result["as_of"] == parts.events[1]["revision_as_of"]
    assert parts.calls == [("evidence",1,at(10))]
    assert result["task"]["method"]=="deepseek"
    assert result["task"]["tool_steps"][0]["name"]=="get_event_evidence"
    assert result["task"]["tool_steps"][0]["status"]=="success"
    assert len(result["task"]["evidence_refs"])>=366
    assert result["task"]["citations"][0]["citation_id"]=="E1"
    assert parts.adapter.closed


def test_concurrent_submit_claims_one_task(parts):
    pending=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _: parts.briefs.submit(parts.context,"event",1,schedule=pending.append),range(8)))
    assert len({r["task_id"] for r in results})==1
    assert len(pending)==1


def test_transport_contains_bounded_facts_instead_of_raw_inputs_or_paths(parts):
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    text=json.dumps(parts.adapter.requests,ensure_ascii=False)
    assert "PRIVATE_RAW" not in text and "private/materials" not in text
    assert "input_refs" not in text and "test-only-key" not in text
    assert len(text)<22000
    assert "E1" in text and "候选工况" in text


def test_get_scope_is_shared_but_chat_task_owner_remains_private(parts):
    pending=[]
    first=parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    viewer=parts.actor.model_copy(update=dict(user_id="u2",roles=("ventilation",)))
    result=parts.briefs.get(parts.context.model_copy(update=dict(actor=viewer)),"event",1)
    assert result["status"]=="completed" and not result["can_cancel"]
    with pytest.raises(DomainError):
        parts.services["agent"].get_task(viewer,first["task_id"])
    for replacement in (dict(session_id="foreign"),dict(source_id="foreign"),dict(actor=viewer.model_copy(update=dict(face_ids=("other",))))):
        with pytest.raises(DomainError):
            parts.briefs.get(parts.context.model_copy(update=replacement),"event",1)


def test_read_only_fallback_keeps_actual_revision_and_never_crosses_grade(parts):
    pending=[]
    first=parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    fallback=parts.briefs.get(parts.context,"event",2)
    assert fallback["event_revision"]==1 and fallback["requested_event_revision"]==2 and fallback["stale"]
    assert fallback["task_id"]==first["task_id"] and len(pending)==1
    assert parts.briefs.get(parts.context,"event",3)["status"]=="not_started"
    current=parts.briefs.submit(parts.context,"event",2,schedule=pending.append)
    assert current["event_revision"]==2 and current["task_id"]!=first["task_id"]
    assert not current["stale"]


def test_future_revision_is_unavailable_and_get_does_not_create(parts):
    assert parts.briefs.get(parts.context,"event",1)["status"]=="not_started"
    assert parts.store.list("task")==[]
    with pytest.raises(DomainError):
        parts.briefs.get(parts.context.model_copy(update=dict(as_of=at(11))),"event",2)


def test_failure_keeps_evidence_and_explicit_retry_creates_one_new_attempt(parts):
    parts.adapter.responses=[DeepSeekRequestError("upstream unavailable",503)]
    pending=[]
    first=parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    failed=parts.briefs.get(parts.context,"event",1)
    assert failed["status"]=="failed" and failed["task"]["upstream_status_code"]==503
    assert failed["task"]["result_cards"] and failed["task"]["evidence_refs"] and failed["can_retry"]
    assert parts.briefs.submit(parts.context,"event",1,schedule=pending.append)["task_id"]==first["task_id"]
    retried=parts.briefs.submit(parts.context,"event",1,retry=True,schedule=pending.append)
    assert retried["task_id"]!=first["task_id"] and retried["attempt"]==2
    pending[-1]()
    assert parts.briefs.get(parts.context,"event",1)["status"]=="completed"


def test_queued_cancel_prevents_model_call(parts):
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    assert parts.briefs.cancel(parts.context,"event",1)["status"]=="cancelled"
    pending[0]()
    assert parts.calls==[] and parts.adapter.requests==[]


def test_missing_key_retains_evidence_without_reporting_model_success(parts):
    parts.briefs.settings=Settings(_env_file=None,deepseek_api_key=None,event_brief_allowed_source_ids=["source"])
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    result=parts.briefs.get(parts.context,"event",1)
    assert result["status"]=="failed"
    assert result["task"]["error_code"]=="deepseek_not_configured"
    assert result["task"]["result_cards"] and not parts.adapter.requests


def test_unverified_model_citation_is_rejected(parts):
    parts.adapter.responses=[ChatReply(dict(role="assistant",content="无法核实的结论[E999]"),"stop","offline",{})]
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    result=parts.briefs.get(parts.context,"event",1)
    assert result["status"]=="failed" and result["task"]["error_code"]=="invalid_brief_citations"
    assert "E999" not in result["task"]["final_text"]


def model_tool(name, arguments):
    return ChatReply(dict(role="assistant",content=None,tool_calls=[dict(id="tool-extra",type="function",
        function=dict(name=name,arguments=json.dumps(arguments)))]),"tool_calls","offline",{})


def test_progress_is_saved_before_evidence_tool_completes(parts):
    original=parts.services["traceability"].get_event_evidence
    seen=[]
    def inspect(*args,**kwargs):
        task=parts.store.list("task")[0]
        seen.append((task["state"],task["tool_steps"][-1]["status"]))
        return original(*args,**kwargs)
    parts.services["traceability"].get_event_evidence=inspect
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]();pending[0]()
    assert seen==[("running","running")]
    assert len(parts.adapter.requests)==1


def test_approved_reference_citation_has_exact_document_version_and_locator(parts):
    parts.adapter.responses=[model_tool("search_documents",dict(query="规程",event_id="event")),
        ChatReply(dict(role="assistant",content="事件已核对[E1]，资料条文见[E5]。"),"stop","offline",{})]
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    result=parts.briefs.get(parts.context,"event",1)
    assert result["status"]=="completed"
    citation=next(item for item in result["task"]["citations"] if item["kind"]=="reference")
    assert citation["entry_id"]=="doc-1" and citation["version"]==2 and citation["locator"]==dict(page=7)
    assert citation["source"]=="矿井规程" and citation["title"]=="审核规程"
    assert "text" not in result["task"]["result_cards"][-1]["data"][0]
    assert "已审核规程条文" in json.dumps(parts.adapter.requests,ensure_ascii=False)


@pytest.mark.parametrize("name,args",[("run_prediction",dict(point_id="t0")),("get_event_evidence",dict(event_id="foreign"))])
def test_model_cannot_start_prediction_or_read_another_event(parts,name,args):
    parts.adapter.responses=[model_tool(name,args)]
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    result=parts.briefs.get(parts.context,"event",1)
    assert result["status"]=="partial"
    assert result["task"]["tool_steps"][-1]["status"]=="error"
    assert len(parts.calls)==1
    assert all(tool["function"]["name"] not in ("run_prediction","draft_advice") for tool in parts.adapter.requests[0]["tools"])


def test_cancellation_during_model_request_keeps_finished_tool_results(parts):
    original=parts.adapter.chat
    def cancel_after_request(*args,**kwargs):
        parts.briefs.cancel(parts.context,"event",1)
        return original(*args,**kwargs)
    parts.adapter.chat=cancel_after_request
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    result=parts.briefs.get(parts.context,"event",1)
    assert result["status"]=="cancelled"
    assert len(result["task"]["tool_steps"])==1 and result["task"]["result_cards"]
    assert "[E1]" not in result["task"]["final_text"]


def test_parallel_retries_share_the_queued_attempt(parts):
    pending=[]
    parts.adapter.responses=[DeepSeekRequestError("unavailable",503)]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _: parts.briefs.submit(parts.context,"event",1,retry=True,schedule=pending.append),range(4)))
    assert len({item["task_id"] for item in results})==1 and len(pending)==2
    assert results[0]["attempt"]==2


def test_proactive_brief_is_excluded_from_regular_chat_history(parts):
    pending=[]
    first=parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    question=parts.store.get("task",first["task_id"])["question"]
    parts.services["agent"].chat(parts.context,"普通问题")
    assert question not in json.dumps(parts.adapter.requests[-1]["messages"],ensure_ascii=False)


def test_owner_cannot_continue_shared_brief_through_private_chat(parts):
    pending=[]
    brief=parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    before=deepcopy(parts.store.get("task",brief["task_id"]))
    requests_before=len(parts.adapter.requests)
    with pytest.raises(DomainError) as denied:
        parts.services["agent"].chat(parts.context,dict(message="私人续聊",task_id=brief["task_id"]))
    assert denied.value.status_code in (403,409)
    assert parts.store.get("task",brief["task_id"])==before
    assert len(parts.adapter.requests)==requests_before


def test_owner_cannot_bypass_brief_cancellation_through_generic_task_endpoint(parts):
    brief=parts.briefs.submit(parts.context,"event",1)
    before=deepcopy(parts.store.get("task",brief["task_id"]))
    with pytest.raises(DomainError) as denied:
        parts.services["agent"].cancel_task(parts.actor,brief["task_id"])
    assert denied.value.status_code==403
    assert parts.store.get("task",brief["task_id"])==before


def test_mixed_unknown_citation_prefix_is_rejected(parts):
    parts.adapter.responses=[ChatReply(dict(role="assistant",content="已核对[E1]，另一个依据[R999]"),"stop","offline",{})]
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    assert parts.briefs.get(parts.context,"event",1)["status"]=="failed"


def test_missing_evidence_fields_mark_a_successful_brief_partial(parts):
    original=parts.services["traceability"].get_event_evidence
    parts.services["traceability"].get_event_evidence=lambda *args,**kwargs: dict(original(*args,**kwargs),missing_fields=["observation_fragments"])
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    assert parts.briefs.get(parts.context,"event",1)["status"]=="partial"


def test_fallback_never_reuses_a_different_model_configuration(parts):
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    changed=EventBriefsService(parts.store,parts.services,parts.settings.model_copy(update=dict(deepseek_model="different-model")))
    assert changed.get(parts.context,"event",1)["status"]=="not_started"
    assert changed.get(parts.context,"event",2)["status"]=="not_started"


def test_restart_marks_only_unfinished_briefs_failed_and_keeps_evidence(parts):
    first=parts.briefs.submit(parts.context,"event",1)
    second=parts.briefs.submit(parts.context,"event",2)
    third=parts.briefs.submit(parts.context,"event",3)
    parts.briefs.cancel(parts.context,"event",3)
    task=parts.store.get("task",first["task_id"])
    task.update(state="running",tool_calls=[dict(name="get_event_evidence",status="success")],citations=[dict(citation_id="E1",kind="event")])
    parts.services["agent"]._save(task)
    chat,_=parts.services["agent"]._begin(parts.context,"普通待运行任务")
    assert parts.briefs.recover_interrupted()==2
    for brief in (first,second):
        saved=parts.store.get("task",brief["task_id"])
        assert saved["state"]=="failed" and saved["error_code"]=="task_interrupted"
    assert parts.store.get("task",first["task_id"])["citations"]
    assert parts.store.get("task",third["task_id"])["state"]=="cancelled"
    assert parts.store.get("task",chat["task_id"])["state"]=="running"
    assert parts.briefs.recover_interrupted()==0


def test_long_event_summary_preserves_peak_and_latest_measurements(parts):
    rows=[dict(id=str(i),event_time=at(0).isoformat(),value=.5) for i in range(100)]
    rows[70]=dict(id="peak",event_time=at(7).isoformat(),value=.9)
    rows[-1]=dict(id="last",event_time=at(10).isoformat(),value=.7)
    summary=parts.briefs._summary("get_event_evidence",dict(event_snapshot=parts.events[1],observation_fragments=rows))
    assert summary["observation_summary"]["maximum"]==.9
    assert datetime.fromisoformat(summary["observation_summary"]["peak_time"])==at(7)
    assert summary["observation_summary"]["count"]==100
    assert summary["observations"][-1]["value"]==.7


def unfinished_tool_task(parts):
    brief=parts.briefs.submit(parts.context,"event",1)
    task=parts.store.get("task",brief["task_id"])
    task.update(state="running",tool_calls=[dict(name="get_event_evidence",status="success",finished_at="existing")]
                +[dict(name="get_case",status=state) for state in ("queued","started","running")],
                result_cards=[dict(tool="get_event_evidence",data=dict(event_id="event"))],
                evidence_refs=[dict(kind="event",event_id="event")])
    parts.services["agent"]._save(task)
    return brief,task


def test_cancel_finishes_pending_tools_and_preserves_completed_evidence(parts):
    brief,before=unfinished_tool_task(parts)
    task=parts.briefs.cancel(parts.context,"event",1)["task"]
    assert task["state"]=="cancelled"
    assert task["tool_calls"][0]==before["tool_calls"][0]
    assert all(step["status"]=="cancelled" and step.get("finished_at") for step in task["tool_calls"][1:])
    assert task["tool_steps"]==task["tool_calls"]
    assert task["result_cards"]==before["result_cards"] and task["evidence_refs"]==before["evidence_refs"]


def test_restart_finishes_pending_tools_and_preserves_completed_evidence(parts):
    brief,before=unfinished_tool_task(parts)
    assert parts.briefs.recover_interrupted()==1
    task=parts.store.get("task",brief["task_id"])
    assert task["state"]=="failed"
    assert task["tool_calls"][0]==before["tool_calls"][0]
    assert all(step["status"]=="error" and step.get("finished_at") and step.get("error_code")=="task_interrupted" for step in task["tool_calls"][1:])
    assert task["tool_steps"]==task["tool_calls"]
    assert task["result_cards"]==before["result_cards"] and task["evidence_refs"]==before["evidence_refs"]


def test_cloud_source_allowlist_defaults_to_empty():
    assert Settings(_env_file=None).event_brief_allowed_source_ids==[]


def test_cloud_source_allowlist_can_be_loaded_from_json_environment(monkeypatch):
    monkeypatch.setenv("EVENT_BRIEF_ALLOWED_SOURCE_IDS",'["synthetic-brief-qa"]')
    assert Settings(_env_file=None).event_brief_allowed_source_ids==["synthetic-brief-qa"]


def test_unauthorized_source_keeps_local_evidence_without_model_requests(parts):
    parts.briefs.settings=parts.settings.model_copy(update=dict(event_brief_allowed_source_ids=[]))
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    result=parts.briefs.get(parts.context,"event",1)
    assert result["status"]=="failed"
    assert result["task"]["error_code"]=="cloud_source_not_authorized"
    assert result["task"]["result_cards"] and result["task"]["evidence_refs"]
    assert parts.adapter.requests==[]


def test_cloud_projection_uses_mine_time_and_chinese_business_terms(parts):
    parts.events[1]["quality_flags"]=["suspected_calibration","timezone_inferred"]
    before=deepcopy(parts.events[1])
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    payload=json.loads(parts.adapter.requests[0]["messages"][1]["content"])
    event=payload["event_facts"]["event"]
    assert event["revision_as_of"].endswith("+08:00")
    assert datetime.fromisoformat(event["revision_as_of"])==datetime.fromisoformat(before["revision_as_of"])
    assert event["event_type"]=="实测超限"
    assert event["quality_flags"]==["疑似标校","时区按配置"]
    assert parts.events[1]==before
    assert parts.briefs.get(parts.context,"event",1)["as_of"]==before["revision_as_of"]


def test_initial_payload_distinguishes_documents_not_queried(parts):
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    payload=json.loads(parts.adapter.requests[0]["messages"][1]["content"])
    assert payload["document_search_state"]=="not_queried"
    system=parts.adapter.requests[0]["messages"][0]["content"]
    assert "纯文本" in system and "UTC+8" in system and "尚未查询" in system


@pytest.mark.parametrize("has_documents,expected",[(True,"matched"),(False,"no_matches")])
def test_document_search_state_tracks_actual_tool_result(parts,has_documents,expected):
    if not has_documents:
        parts.services["knowledge"].search_documents=lambda **kwargs: []
    parts.adapter.responses=[model_tool("search_documents",dict(query="规程",event_id="event"))]
    pending=[]
    parts.briefs.submit(parts.context,"event",1,schedule=pending.append)
    pending[0]()
    payload=json.loads(parts.adapter.requests[1]["messages"][-1]["content"])
    assert payload["document_search_state"]==expected
