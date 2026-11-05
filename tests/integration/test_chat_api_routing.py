"""The visible chat must distinguish public conversation and source-bound data."""
import json

import pytest

from test_agent_real_api_orchestration import (
    IDS, FakeAdapter, configured_service, real_parts, reply, assert_outbound_is_projected,
)


def available_session(parts, service):
    parts.store.put("source","allowed-source",dict(source_id="allowed-source",face_id=IDS["face"],name="已授权回放"))
    parts.store.put("session","allowed-session",dict(session_id="allowed-session",source_id="allowed-source",
        face_id=IDS["face"],mode="historical_replay",as_of=parts.context.as_of.isoformat(),created_at=parts.context.as_of.isoformat()))
    service.settings.agent_real_source_ids=["allowed-source"]


def test_blocked_source_explains_route_and_offers_only_authorized_sessions(real_parts):
    service,adapter,constructions=configured_service(real_parts,[],sources=[])
    available_session(real_parts,service)
    task=service.chat(real_parts.context,"帮我梳理当前情况")
    assert not constructions and not adapter.requests
    assert task["local_reason"] == "cloud_source_not_authorized"
    assert task["error_code"] == "cloud_source_not_authorized"
    assert "请说明查询对象" not in task["final_text"]
    assert task["chat_status"]["mode"] == "local_only"
    assert task["chat_status"]["available_sessions"][0]["session_id"] == "allowed-session"
    assert task["chat_status"]["general_chat"] is True


def test_local_business_query_is_labeled_without_losing_local_result(real_parts):
    service,adapter,constructions=configured_service(real_parts,[],sources=[])
    task=service.chat(real_parts.context,"查看当前事件")
    assert not constructions and not adapter.requests
    assert task["state"] == "completed" and task["result_cards"]
    assert task["chat_status"]["mode"] == "local_only"


@pytest.mark.parametrize("question",["你是谁？", "你能正常说话吗？"])
def test_public_identity_chat_uses_deepseek_without_sending_private_question(real_parts,question):
    service,adapter,constructions=configured_service(real_parts,[reply("我是风险管控助手，通过DeepSeek API提供帮助。")],sources=[])
    task=service.chat(real_parts.context,question+" RAW_QUESTION_CANARY "+IDS["source"])
    assert len(constructions)==1 and len(adapter.requests)==1
    assert task["method"] == "deepseek" and task["model"] == "offline-fake"
    assert task["chat_scope"] == "general" and task["state"] == "completed"
    assert task["tool_calls"] == [] and real_parts.calls == []
    assert adapter.requests[0].get("tools") is None
    assert_outbound_is_projected(adapter)
