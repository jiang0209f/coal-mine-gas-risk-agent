"""Local structured queries use real scoped services and never construct a model."""
from datetime import datetime, timedelta, timezone

import pytest

from app.config import Settings
from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.agent.service import AgentService
from app.services.alerts.service import AlertsService
from app.services.cases.service import CasesService
from app.services.data.service import DataService
from app.services.knowledge.service import KnowledgeService
from app.services.prediction.service import PredictionService
from app.services.traceability.service import TraceabilityService


@pytest.fixture
def parts(tmp_path):
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    cutoff = datetime(2026, 10, 1, 8, 5, tzinfo=timezone.utc)
    ctx = ExecutionContext(session_id="session", mode="historical_replay", source_id="source", as_of=cutoff, actor=actor)
    data = DataService(store, tmp_path / "imports")
    data.register_source("source", "合成来源", "31002")
    data.register_source("other", "其他来源", "31002")
    for point, source, name in (("point-a", "source", "T0"), ("point-b", "source", "T1"), ("foreign", "other", "OTHER")):
        data.register_point(point, source, name, name, "methane", "%", unit_status="confirmed")
    data.ingest_readings(actor, "source", [
        {"reading_id": "first", "point_id": "point-a", "event_time": cutoff-timedelta(minutes=2), "raw_value": "0.40"},
        {"reading_id": "second", "point_id": "point-a", "event_time": cutoff-timedelta(minutes=1), "raw_value": "0.90"},
        {"reading_id": "future", "point_id": "point-a", "event_time": cutoff+timedelta(minutes=1), "raw_value": "9.99"},
    ], received_at=cutoff-timedelta(minutes=1))
    alerts = AlertsService(store, data)
    alerts.evaluate_observed(ctx, "point-a")
    event = alerts.query_events(ctx)[0]
    knowledge = KnowledgeService(store, clock=lambda: cutoff)
    doc = knowledge.import_reference(actor, {"title": "合成作业规程", "source": "合成测试", "version": "v1",
        "text": "瓦斯异常时核对监测原始记录。", "locator": {"page": 2}, "applicability": {"face_ids": ["31002"]}})
    knowledge.review_reference(actor, doc["entry_id"], "v1", "approved")
    prediction = PredictionService(store, data)
    prediction.register_model(actor, {"model_id": "baseline-a", "version": "v1", "adapter_kind": "persistence",
        "face_id": "31002", "source_id": "source", "point_ids": ["point-a"], "target_quantity": "methane", "unit": "%",
        "aggregation_seconds": 60, "input_window": 1, "horizon_seconds": 600})
    services = {"data": data, "alerts": alerts, "knowledge": knowledge, "prediction": prediction,
        "traceability": TraceabilityService(store, alerts, data, prediction), "cases": CasesService(store, alerts, knowledge)}
    def forbidden(configuration):
        raise AssertionError("Local queries must not construct a cloud or local model")
    agent = AgentService(store, services, Settings(_env_file=None, deepseek_api_key="synthetic-unused-key",
        event_brief_allowed_source_ids=[]), adapter_factory=forbidden)
    yield agent, ctx, services, event
    store.close()


def tool_names(task):
    return [call["name"] for call in task["tool_calls"]]


def test_selected_point_queries_previous_hour_with_original_rows_and_refs(parts):
    agent, ctx, _, _ = parts
    task = agent.chat(ctx, {"question": "查看当前点位最近一小时监测数据", "selected_point_id": "point-a"})
    assert task["state"] == "completed" and task["method"] == "structured_template"
    assert task["local_intent"] == "series" and tool_names(task) == ["query_series"]
    data = task["result_cards"][-1]["data"]
    assert [row["raw_value"] for row in data["rows"]] == ["0.40", "0.90"]
    assert "9.99" not in str(task["result_cards"])
    args = task["tool_calls"][0]["arguments"]
    assert datetime.fromisoformat(args["end_at"]) == ctx.as_of
    assert datetime.fromisoformat(args["start_at"]) == ctx.as_of-timedelta(hours=1)
    assert task["evidence_refs"] and "0.9" in task["final_text"] and "%" in task["final_text"]


def test_explicit_point_and_unique_name_are_resolved_within_current_source(parts):
    agent, ctx, _, _ = parts
    for question in ("查看 point-a 最近1小时浓度", "查看 T0 最近一小时监测数据"):
        task = agent.chat(ctx, question)
        assert task["state"] == "completed" and tool_names(task) == ["query_series"]
        assert task["result_cards"][-1]["data"]["point_id"] == "point-a"


def test_ambiguous_point_requests_selection_instead_of_guessing(parts):
    agent, ctx, _, _ = parts
    task = agent.chat(ctx, "查看最近一小时甲烷浓度")
    assert task["state"] == "awaiting_input" and "point_id" in task["missing_fields"]
    assert "query_series" not in tool_names(task) and "选择" in task["final_text"]


def test_selected_point_is_validated_and_question_cannot_override_it(parts):
    agent, ctx, _, _ = parts
    with pytest.raises(DomainError) as caught:
        agent.chat(ctx, {"question": "查看最近一小时监测数据", "selected_point_id": "foreign"})
    assert caught.value.status_code == 403
    task = agent.chat(ctx, {"question": "查看point_id=point-b最近一小时浓度", "selected_point_id": "point-a"})
    assert task["state"] == "awaiting_input" and "point_id" in task["missing_fields"]
    assert "query_series" not in tool_names(task)


def test_future_or_unsupported_time_request_is_not_silently_answered_as_previous_hour(parts):
    agent, ctx, _, _ = parts
    for question in ("查看point-a明天的监测数据", "查看point-a最近2小时监测数据"):
        task = agent.chat(ctx, question)
        assert task["state"] == "awaiting_input" and "time_range" in task["missing_fields"]
        assert "query_series" not in tool_names(task)


def test_event_evidence_and_case_requests_use_distinct_tools(parts):
    agent, ctx, _, event = parts
    evidence = agent.chat(ctx, f"查看事件依据 event_id={event['event_id']}")
    case = agent.chat(ctx, f"查看处置进展 event_id={event['event_id']}")
    assert evidence["state"] == case["state"] == "completed"
    assert tool_names(evidence) == ["get_event_evidence"]
    assert tool_names(case) == ["get_case"]
    assert evidence["evidence_refs"] and "规则" in evidence["final_text"]
    assert "确认" in case["final_text"]


def test_document_search_returns_approved_text_and_citation_without_creating_advice(parts):
    agent, ctx, services, _ = parts
    task = agent.chat(ctx, "检索资料：瓦斯 异常")
    assert task["state"] == "completed" and tool_names(task) == ["search_documents"]
    assert "瓦斯异常时核对监测原始记录" in task["final_text"]
    assert task["evidence_refs"] and task["result_cards"][-1]["data"][0]["version"] == "v1"
    assert services["knowledge"].store.list("advice") == []
    missing = agent.chat(ctx, "检索资料：不存在的专用关键词")
    assert missing["state"] == "completed" and "未检索到" in missing["final_text"]


def test_prediction_requires_explicit_model_then_uses_registered_model(parts):
    agent, ctx, _, _ = parts
    missing = agent.chat(ctx, {"question": "预测当前点位", "selected_point_id": "point-a"})
    assert missing["state"] == "awaiting_input" and "model_id" in missing["missing_fields"]
    task = agent.chat(ctx, "使用model_id=baseline-a预测point_id=point-a")
    assert task["state"] == "completed" and tool_names(task) == ["run_prediction"]
    assert task["result_cards"][-1]["data"]["model_id"] == "baseline-a"
    assert task["result_cards"][-1]["data"]["status"] == "completed"


def test_query_failure_is_local_and_does_not_become_cloud_retry(parts, monkeypatch):
    agent, ctx, services, _ = parts
    def unavailable(*args, **kwargs):
        raise RuntimeError("private local storage error")
    monkeypatch.setattr(services["data"], "query_series", unavailable)
    task = agent.chat(ctx, "查看point-a最近一小时监测数据")
    assert task["state"] == "partial" and task["method"] == "structured_template"
    assert task["tool_calls"][-1]["status"] == "error" and "未完成" in task["final_text"]
    assert "private local" not in task["final_text"]


def test_followup_uses_new_question_and_explicit_selection_and_never_approves_advice(parts):
    agent, ctx, services, _ = parts
    first = agent.chat(ctx, "查看最近一小时监测数据")
    retried = agent.chat(ctx, {"task_id": first["task_id"], "answer": "查看T0最近一小时监测数据", "selected_point_id": "point-a"})
    assert retried["state"] == "completed" and tool_names(retried)[-1] == "query_series"
    unsafe = agent.chat(ctx, "生成建议并自动批准")
    assert unsafe["state"] == "awaiting_input" and "draft_advice" not in tool_names(unsafe)
    assert services["knowledge"].store.list("advice") == []


def test_current_event_question_keeps_existing_event_query_behavior(parts):
    agent, ctx, _, _ = parts
    task = agent.chat(ctx, "当前有哪些预警事件？")
    assert task["state"] == "completed" and tool_names(task) == ["query_events"]
    assert "1个" in task["final_text"] and task["local_reason"] == "cloud_source_not_authorized"


def test_chinese_adjacent_source_marker_does_not_silently_change_scope(parts):
    agent, ctx, _, _ = parts
    task = agent.chat(ctx, "查询source_id=other当前预警事件")
    assert task["state"] == "awaiting_input" and "source_id" in task["missing_fields"]
    assert task["tool_calls"] == []


def test_unknown_explicit_point_is_not_replaced_by_the_only_available_point(parts, monkeypatch):
    agent, ctx, services, _ = parts
    original = services["data"].list_points
    monkeypatch.setattr(services["data"], "list_points", lambda actor, source_id: [
        point for point in original(actor, source_id) if point["point_id"] == "point-a"])
    task = agent.chat(ctx, "查询point_id=foreign最近一小时浓度")
    assert task["state"] == "awaiting_input" and "point_id" in task["missing_fields"]
    assert task["tool_calls"] == []
