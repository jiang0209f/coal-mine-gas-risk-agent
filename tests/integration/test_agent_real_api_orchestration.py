"""Offline API orchestration with synthetic canaries marked as a real source."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pytest

from app.adapters.deepseek import ChatReply, DeepSeekRequestError
from app.config import Settings
from app.db import Store
from app.schemas.common import Actor, ExecutionContext
from app.services.agent.service import AgentService


IDS = {
    "source": "REAL_SOURCE_CANARY", "face": "REAL_FACE_CANARY",
    "session": "REAL_SESSION_CANARY", "actor": "REAL_ACTOR_CANARY",
    "point": "REAL_POINT_CANARY", "event": "REAL_EVENT_CANARY",
    "model": "REAL_MODEL_CANARY", "run": "REAL_RUN_CANARY",
    "reference": "PUBLIC_REFERENCE_LOCAL_ID_CANARY",
}
RAW_QUESTION = "请梳理当前情况并说明依据。现场备注：RAW_QUESTION_CANARY"
PRIVATE_PATH = "F:/private/PRIVATE_PATH_CANARY.csv"
VIRTUAL_NOW = datetime(2000, 1, 2, tzinfo=timezone.utc)


def reply(content="已完成工具查询。", calls=None):
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return ChatReply(message, "tool_calls" if calls else "stop", "offline-fake", {"total_tokens": 3})


def call(name, arguments, call_id):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
    }}


class FakeAdapter:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.requests.append({"messages": deepcopy(messages), **deepcopy(kwargs)})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


@pytest.fixture
def real_parts(monkeypatch):
    import os

    for name in tuple(os.environ):
        if name.upper().startswith(("DEEPSEEK_", "AGENT_REAL_", "EVENT_BRIEF_ALLOWED_SOURCE_IDS")):
            monkeypatch.delenv(name, raising=False)
    store = Store("sqlite+pysqlite:///:memory:")
    store.put("source", IDS["source"], {"source_id": IDS["source"], "face_id": IDS["face"]})
    actor = Actor(user_id=IDS["actor"], roles=("safety_management",), face_ids=(IDS["face"],))
    context = ExecutionContext(
        session_id=IDS["session"], source_id=IDS["source"], mode="historical_replay",
        as_of=datetime.fromisoformat("2026-10-01T08:05:00+08:00"), actor=actor,
    )
    calls = []

    def capture(name, result):
        def invoke(*args, **kwargs):
            calls.append((name, deepcopy(args), deepcopy(kwargs)))
            return deepcopy(result)
        return invoke

    point = {"point_id": IDS["point"], "source_id": IDS["source"], "face_id": IDS["face"],
             "quantity": "methane", "unit": "%", "file_path": PRIVATE_PATH}
    model = {"model_id": IDS["model"], "source_id": IDS["source"], "point_ids": [IDS["point"]],
             "version": "v1", "model_type": "professional", "default_eligible": True,
             "deployment_stage": "experimental", "legacy_source_root": PRIVATE_PATH}
    event = {
        "event_id": IDS["event"], "source_id": IDS["source"], "session_id": IDS["session"],
        "face_id": IDS["face"], "point_id": IDS["point"], "event_revision": 1,
        "revision_as_of": "2026-10-01T00:04:00+00:00", "level": "orange",
        "event_type": "observed_exceedance", "private_path": PRIVATE_PATH,
    }
    observations = [{
        "id": f"RAW_OBSERVATION_CANARY_{index}", "point_id": IDS["point"],
        "source_id": IDS["source"], "source_file_id": "RAW_FILE_CANARY", "row_number": index,
        "event_time": (context.as_of - timedelta(seconds=30 * index)).isoformat(),
        "available_at": context.as_of.isoformat(), "value": 0.3 + index / 100,
        "raw_value": "RAW_TIME_SERIES_CANARY", "quality_flags": [],
    } for index in range(1, 25)]
    evidence = {
        "event_snapshot": event, "observation_fragments": observations,
        "evidence_refs": [{"kind": "observation", "observation_id": row["id"],
                           "source_file_id": row["source_file_id"], "row_number": row["row_number"]}
                          for row in observations],
        "rules": [{"threshold": 0.65}],
        "candidate_attribution": {"candidate_causes": ["候选工况A"], "input_refs": observations},
        "private_path": PRIVATE_PATH, "operator_notes": "PRIVATE_KNOWLEDGE_CANARY",
    }
    fragment = {
        "reference_id": IDS["reference"], "text": "公开标准中的监测要求。", "title": "公开监测标准",
        "locator": {"page": 1, "clause": "5.1"}, "version": "2026",
        "official_url": "https://www.mem.gov.cn/example.pdf", "review_state": "approved",
        "effective_state": "effective", "disclosure_class": "public_normative",
        "local_path": PRIVATE_PATH,
    }
    services = {
        "data": SimpleNamespace(
            list_points=lambda actor, source_id: deepcopy([point]),
            query_series=capture("query_series", {"rows": observations, "total": len(observations), "quality": {}}),
        ),
        "prediction": SimpleNamespace(
            list_models=lambda context: deepcopy([model]),
            run_prediction=capture("run_prediction", {"run_id": IDS["run"], "point_id": IDS["point"],
                                                       "model_id": IDS["model"], "status": "completed"}),
            get_forecast=capture("get_forecast", {"run_id": IDS["run"], "status": "completed",
                                                   "predictions": [{"value": 0.6, "event_time": context.as_of.isoformat()}]}),
        ),
        "alerts": SimpleNamespace(
            query_events=capture("query_events", [event]),
            get_event_snapshot=capture("get_event_snapshot", event),
        ),
        "traceability": SimpleNamespace(get_event_evidence=capture("get_event_evidence", evidence)),
        "knowledge": SimpleNamespace(
            search_documents=capture("search_documents", [fragment]),
            search_public_documents=capture("search_public_documents", [fragment]),
            approved_public_fragment=capture("approved_public_fragment", fragment),
            draft_advice=capture("draft_advice", {"state": "draft", "reference_ids": [IDS["reference"]]}),
        ),
        "cases": SimpleNamespace(get_case=capture("get_case", {"state": "draft", "event_id": IDS["event"]})),
    }
    yield SimpleNamespace(store=store, actor=actor, context=context, services=services, calls=calls)
    store.close()


def configured_service(parts, responses, *, mode="summary", sources=None):
    adapter = FakeAdapter(responses)
    constructions = []

    def factory(settings):
        constructions.append(settings)
        return adapter

    settings = Settings(
        _env_file=None, deepseek_api_key="offline-test-key", event_brief_allowed_source_ids=[],
        agent_real_source_ids=[IDS["source"]] if sources is None else sources,
        agent_real_data_mode=mode,
    )
    service = AgentService(parts.store, parts.services, settings, adapter_factory=factory)
    return service, adapter, constructions


def forbid_template(monkeypatch, service):
    def fail(*args, **kwargs):
        pytest.fail("An authorized real-summary task must use API tool orchestration")
    monkeypatch.setattr(service, "_template", fail)


def assert_outbound_is_projected(adapter):
    outbound = json.dumps(adapter.requests, ensure_ascii=False)
    for canary in [*IDS.values(), RAW_QUESTION, "RAW_QUESTION_CANARY", PRIVATE_PATH,
                   "PRIVATE_KNOWLEDGE_CANARY", "RAW_OBSERVATION_CANARY", "RAW_FILE_CANARY",
                   "RAW_TIME_SERIES_CANARY", "offline-test-key", "2026-10-01"]:
        assert canary not in outbound, f"Local-only content escaped: {canary}"


@pytest.mark.parametrize("mode,sources", [("disabled", [IDS["source"]]), ("summary", []), ("disabled", [])])
def test_unapproved_real_source_keeps_local_results_without_constructing_adapter(real_parts, mode, sources):
    service, adapter, constructions = configured_service(real_parts, [], mode=mode, sources=sources)

    task = service.chat(real_parts.context, "查看当前事件")

    assert constructions == [] and adapter.requests == []
    assert task["method"] == "structured_template" and task["state"] == "completed"
    assert task["model"] is None and task["usage"] == {}
    assert IDS["event"] in json.dumps(task["result_cards"])


def test_authorized_summary_uses_model_selected_multiround_tools_and_preserves_local_evidence(real_parts, monkeypatch):
    service, adapter, constructions = configured_service(real_parts, [
        reply(None, [call("query_events", {}, "events-call")]),
        reply(None, [call("get_event_evidence", {"event_id": "E1"}, "evidence-call")]),
        reply("已依据工具结果整理当前情况。"),
    ])
    forbid_template(monkeypatch, service)

    task = service.chat(real_parts.context, {"question": RAW_QUESTION, "selected_point_id": IDS["point"]})

    assert len(constructions) == 1 and len(adapter.requests) == 3
    assert task["method"] == "deepseek" and task["model"] == "offline-fake"
    assert task["state"] == "completed" and adapter.closed
    assert [name for name, _, _ in real_parts.calls] == ["query_events", "get_event_evidence"]
    assert real_parts.calls[1][2]["event_id"] == IDS["event"]
    assert real_parts.calls[1][2]["context"] == real_parts.context
    assert "PRIVATE_KNOWLEDGE_CANARY" in json.dumps(task["result_cards"])
    assert task["evidence_refs"]
    tool_messages = [message for message in adapter.requests[-1]["messages"] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["events-call", "evidence-call"]
    assert all(json.loads(message["content"])["status"] == "success" for message in tool_messages)
    assert "E1" in tool_messages[0]["content"]
    assert_outbound_is_projected(adapter)


def test_unknown_bracket_reference_is_rejected_after_successful_tool(real_parts, monkeypatch):
    service, adapter, _ = configured_service(real_parts, [
        reply(None, [call("query_events", {}, "events")]),
        reply("已检查当前预警，依据[REF999]。"),
    ])
    forbid_template(monkeypatch, service)
    task = service.chat(real_parts.context, "查询当前预警")
    assert task["state"] == "partial"
    assert task["error_code"] == "unverified_reference_id"
    assert task["result_cards"] and len(adapter.requests) == 2


def test_summary_authorization_revoked_between_rounds_stops_api(real_parts, monkeypatch):
    service, adapter, _ = configured_service(real_parts, [reply(None, [call("query_events", {}, "events")])])
    forbid_template(monkeypatch, service)
    original = real_parts.services["alerts"].query_events

    def revoke(**kwargs):
        result = original(**kwargs)
        service.settings.agent_real_data_mode = "disabled"
        return result

    real_parts.services["alerts"].query_events = revoke
    task = service.chat(real_parts.context, "查询当前预警")
    assert task["state"] == "partial" and task["error_code"] == "cloud_source_not_authorized"
    assert len(adapter.requests) == 1 and task["result_cards"]


def test_summary_internal_messages_and_alias_maps_are_not_public(real_parts):
    service, adapter, _ = configured_service(real_parts, [
        reply(None, [call("query_events", {}, "events")]), reply("已查询当前预警。"),
    ])
    task = service.chat(real_parts.context, "查询当前预警")
    assert task["state"] == "completed"
    assert "_cloud_messages" not in task and "_cloud_bindings" not in task
    assert "messages" not in task and "context" not in task
    assert_outbound_is_projected(adapter)


def test_virtual_point_and_clock_map_to_original_query_window_only_on_local_execution(real_parts, monkeypatch):
    service, adapter, _ = configured_service(real_parts, [
        reply(None, [call("query_series", {"point_id": "P1", "start_at": "2000-01-01T23:55:00+00:00",
                                           "end_at": "2000-01-02T00:00:00+00:00"}, "series-call")]),
        reply("已完成当前时间窗口查询。"),
    ])
    forbid_template(monkeypatch, service)

    task = service.chat(real_parts.context, {"question": "查询最近五分钟监测趋势，RAW_QUESTION_CANARY", "selected_point_id": IDS["point"]})

    assert task["state"] == "completed" and len(adapter.requests) == 2
    name, _, arguments = real_parts.calls[0]
    assert name == "query_series" and arguments["point_id"] == IDS["point"]
    assert arguments["context"] == real_parts.context
    assert arguments["end_at"] == real_parts.context.as_of
    assert arguments["start_at"] == real_parts.context.as_of - timedelta(minutes=5)
    assert arguments["end_at"] - arguments["start_at"] == timedelta(minutes=5)
    initial_context = json.loads(adapter.requests[0]["messages"][0]["content"].split("\n执行上下文：", 1)[1])
    assert datetime.fromisoformat(initial_context["as_of"].replace("Z", "+00:00")) == VIRTUAL_NOW
    assert initial_context["selected_point_id"] == "P1"
    assert initial_context["available_points"][0]["point_id"] == "P1"
    assert initial_context["available_models"][0]["model_id"] == "M1"
    assert adapter.requests[0]["tools"] == service.tools
    assert len(service.tools) == 8
    assert "RAW_TIME_SERIES_CANARY" in json.dumps(task["result_cards"])
    assert any(ref.get("observation_id", "").startswith("RAW_OBSERVATION_CANARY") for ref in task["evidence_refs"])
    payload = json.loads(next(message["content"] for message in adapter.requests[1]["messages"] if message["role"] == "tool"))
    assert "rows" not in payload.get("data", {})
    assert_outbound_is_projected(adapter)


def test_model_run_and_public_reference_aliases_roundtrip_to_local_identifiers(real_parts, monkeypatch):
    service, adapter, _ = configured_service(real_parts, [
        reply(None, [call("query_events", {}, "events"),
                     call("run_prediction", {"point_id": "P1", "model_id": "M1", "horizon_seconds": 300}, "predict")]),
        reply(None, [call("get_forecast", {"run_id": "R1"}, "forecast"),
                     call("search_documents", {"query": "瓦斯监测规定", "event_id": "E1"}, "documents")]),
        reply(None, [call("draft_advice", {"event_id": "E1", "reference_ids": ["REF1"]}, "draft")]),
        reply("已生成建议草稿，reference_id=REF1。"),
    ])
    forbid_template(monkeypatch, service)

    task = service.chat(real_parts.context, "预测并检索公开标准，起草建议。RAW_QUESTION_CANARY")

    assert task["state"] == "completed" and len(adapter.requests) == 4
    prediction = next(kwargs for name, _, kwargs in real_parts.calls if name == "run_prediction")
    forecast = next(kwargs for name, _, kwargs in real_parts.calls if name == "get_forecast")
    advice = next(kwargs for name, _, kwargs in real_parts.calls if name == "draft_advice")
    assert prediction["point_id"] == IDS["point"] and prediction["model_id"] == IDS["model"]
    assert forecast["run_id"] == IDS["run"]
    assert advice["reference_ids"] == [IDS["reference"]]
    assert task["result_cards"][-1]["data"]["state"] == "draft"
    assert any(ref.get("reference_id") == IDS["reference"] for ref in task["evidence_refs"])
    assert IDS["reference"] in task["final_text"]
    assert any(name == "approved_public_fragment" for name, _, _ in real_parts.calls)
    assert_outbound_is_projected(adapter)


@pytest.mark.parametrize("status_code", [429, 503])
def test_real_api_failure_preserves_local_results_and_reports_failure(real_parts, monkeypatch, status_code):
    service, adapter, _ = configured_service(real_parts, [
        reply(None, [call("query_events", {}, "events")]),
        DeepSeekRequestError("PRIVATE_PROVIDER_BODY_CANARY offline-test-key", status_code=status_code),
    ])
    forbid_template(monkeypatch, service)

    task = service.chat(real_parts.context, RAW_QUESTION)

    assert len(adapter.requests) == 2 and adapter.closed
    assert task["method"] == "deepseek" and task["state"] == "failed"
    assert task["error_code"] == "model_service_unavailable" and task["upstream_status_code"] == status_code
    assert task["tool_calls"][0]["status"] == "success" and task["result_cards"]
    assert IDS["event"] in json.dumps(task["result_cards"])
    assert "PRIVATE_PROVIDER_BODY_CANARY" not in json.dumps(task)
    assert "offline-test-key" not in json.dumps(task)
    assert_outbound_is_projected(adapter)
