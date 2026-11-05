"""Approved real-source briefs use the shared bounded API projection."""
import json

from app.adapters.deepseek import DeepSeekRequestError
from test_event_briefs import parts, model_tool


def approve_summary(parts):
    settings = parts.settings.model_copy(update={
        "event_brief_allowed_source_ids": [],
        "agent_real_source_ids": ["source"],
        "agent_real_data_mode": "summary",
    })
    parts.settings = settings
    parts.briefs.settings = settings
    parts.services["agent"].settings = settings
    for event in parts.events.values():
        event["point_id"] = "REAL_POINT_CANARY"
    parts.services["data"].list_points = lambda actor, source_id: [
        dict(point_id="REAL_POINT_CANARY", source_id="source", face_id="31002", quantity="methane", unit="%")]


def run_brief(parts):
    pending = []
    parts.briefs.submit(parts.context, "event", 1, schedule=pending.append)
    pending[0]()
    return parts.briefs.get(parts.context, "event", 1)


def test_real_summary_brief_uses_api_and_preserves_local_evidence(parts):
    approve_summary(parts)
    result = run_brief(parts)
    assert result["status"] == "completed"
    assert result["task"]["model"] == "offline-test"
    assert result["task"]["method"] == "deepseek"
    assert result["task"]["result_cards"][0]["data"]["event_snapshot"]["point_id"] == "REAL_POINT_CANARY"
    wire = json.dumps(parts.adapter.requests, ensure_ascii=False)
    assert "REAL_POINT_CANARY" not in wire
    assert "31002" not in wire and "PRIVATE_RAW" not in wire
    assert "private/materials" not in wire and "input_refs" not in wire
    assert "2022-02-05" not in wire
    tools = parts.adapter.requests[0]["tools"]
    assert {item["function"]["name"] for item in tools} == {
        "get_event_evidence", "get_forecast", "search_documents", "get_case"}
    assert result["task"]["final_text"].endswith("[E1]")


def test_real_summary_brief_decodes_model_selected_event_alias(parts):
    approve_summary(parts)
    parts.adapter.responses = [model_tool("get_event_evidence", {"event_id": "E1"})]
    result = run_brief(parts)
    assert result["status"] == "completed"
    assert len(parts.calls) == 2
    assert result["task"]["tool_steps"][1]["arguments"] == {"event_id": "event"}
    assert result["task"]["tool_steps"][1]["status"] == "success"
    returned = [entry for entry in parts.adapter.requests[-1]["messages"] if entry["role"] == "tool"]
    assert returned[0]["tool_call_id"] == "tool-extra"
    assert json.loads(returned[0]["content"])["status"] == "success"


def test_real_summary_brief_api_error_keeps_local_evidence(parts):
    approve_summary(parts)
    parts.adapter.responses = [DeepSeekRequestError("network", 503)]
    result = run_brief(parts)
    assert result["status"] == "failed"
    assert result["task"]["error_code"] == "model_service_unavailable"
    assert result["task"]["result_cards"] and result["task"]["evidence_refs"]
    assert result["task"]["model"] is None
    assert len(parts.adapter.requests) == 1


def test_real_summary_brief_rejects_unknown_event_alias(parts):
    approve_summary(parts)
    parts.adapter.responses = [model_tool("get_event_evidence", {"event_id": "E999"})]
    result = run_brief(parts)
    assert len(parts.calls) == 1
    assert result["status"] == "partial"
    assert result["task"]["tool_steps"][1]["status"] == "error"


def test_brief_policy_change_does_not_reuse_old_blocked_task(parts):
    parts.briefs.settings = parts.settings.model_copy(update={"event_brief_allowed_source_ids": []})
    parts.services["agent"].settings = parts.briefs.settings
    blocked = run_brief(parts)
    assert blocked["task"]["error_code"] == "cloud_source_not_authorized"
    approve_summary(parts)
    completed = run_brief(parts)
    assert completed["task_id"] != blocked["task_id"]
    assert completed["status"] == "completed"


def test_unapproved_real_brief_stays_local(parts):
    parts.briefs.settings = parts.settings.model_copy(update={
        "event_brief_allowed_source_ids": [], "agent_real_source_ids": ["source"],
        "agent_real_data_mode": "disabled"})
    result = run_brief(parts)
    assert result["status"] == "failed"
    assert result["task"]["error_code"] == "cloud_source_not_authorized"
    assert parts.adapter.requests == []


def test_summary_authorization_revoked_between_rounds_stops_further_requests(parts):
    approve_summary(parts)
    parts.adapter.responses = [model_tool("get_event_evidence", {"event_id": "E1"})]
    original = parts.adapter.chat

    def revoke(*args, **kwargs):
        response = original(*args, **kwargs)
        parts.briefs.settings = parts.settings.model_copy(update={"agent_real_source_ids": []})
        return response

    parts.adapter.chat = revoke
    pending = []
    entry = parts.briefs.submit(parts.context, "event", 1, schedule=pending.append)
    pending[0]()
    result = parts.store.get("task", entry["task_id"])
    assert len(parts.adapter.requests) == 1
    assert result["state"] == "partial"
    assert result["error_code"] == "cloud_source_not_authorized"
    assert result["result_cards"]
