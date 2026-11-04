"""Legacy cloud tools must honor the same public disclosure boundary."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from app.adapters.deepseek import ChatReply
from app.config import Settings
from app.db import Store
from app.schemas.common import Actor, ExecutionContext
from app.services.agent.briefs import EventBriefsService
from app.services.agent.service import AgentService
from app.services.knowledge.service import KnowledgeService


PRIVATE = "PRIVATE_REAL_MINE_REGULATION_CANARY"


class Capture:
    def __init__(self):
        self.requests = []
        self.responses = [ChatReply({"role": "assistant", "content": None, "tool_calls": [{
            "id": "public-search", "type": "function", "function": {"name": "search_documents",
            "arguments": json.dumps({"query": "甲烷调校"})}}]}, "tool_calls", "offline-fake", {}),
            ChatReply({"role": "assistant", "content": "公开条文已查询。[E1]"}, "stop", "offline-fake", {})]

    def chat(self, messages, **kwargs):
        self.requests.append(deepcopy({"messages": messages, **kwargs}))
        return self.responses.pop(0)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def clear_deepseek(monkeypatch):
    import os
    for name in tuple(os.environ):
        if name.upper().startswith("DEEPSEEK_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="operator", roles=("safety_management",), face_ids=("synthetic-and-local-face",))
    context = ExecutionContext(session_id="synthetic-session", source_id="synthetic-source", mode="simulated_stream",
        as_of=datetime(2026, 11, 4, tzinfo=timezone.utc), actor=actor)
    store.put("source", context.source_id, {"source_id": context.source_id, "face_id": actor.face_ids[0]})
    knowledge = KnowledgeService(store)
    for public in (False, True):
        doc = knowledge.import_reference(actor, {"title": "公开标准" if public else "矿井真实作业规程",
            "source": "https://www.mem.gov.cn/test.pdf" if public else "矿井本机规程",
            "version": "v1", "applicability": {"face_ids": list(actor.face_ids)},
            "fragments": [{"text": "甲烷调校应记录结果。" if public else "甲烷调校 " + PRIVATE,
                           "locator": {"page": 1}}]})
        knowledge.review_reference(actor, doc["entry_id"], "v1", "approved")
        if public:
            knowledge.set_disclosure(actor, doc["entry_id"], "v1", "public_normative",
                official_url="https://www.mem.gov.cn/test.pdf", category="standard", comment="合成公开资料边界测试")
    event = {"event_id": "synthetic-event", "session_id": context.session_id, "source_id": context.source_id,
        "point_id": "synthetic-point", "event_revision": 1, "revision_as_of": context.as_of.isoformat(),
        "start_at": context.as_of.isoformat(), "event_type": "observed_exceedance", "alert_level": "orange"}
    services = {"knowledge": knowledge,
        "alerts": SimpleNamespace(get_event_snapshot=lambda *args, **kwargs: deepcopy(event)),
        "traceability": SimpleNamespace(get_event_evidence=lambda *args, **kwargs: {
            "event_snapshot": deepcopy(event), "observation_fragments": [], "rules": {}, "missing_fields": []}),
    }
    settings = Settings(_env_file=None, deepseek_api_key="offline-fake-key", event_brief_allowed_source_ids=[context.source_id])
    yield SimpleNamespace(store=store, actor=actor, context=context, services=services, knowledge=knowledge, settings=settings)
    store.close()


def test_legacy_chat_cloud_search_never_returns_local_only_knowledge(parts):
    capture = Capture()
    agent = AgentService(parts.store, parts.services, parts.settings, adapter_factory=lambda _: capture)
    result = agent.chat(parts.context, "检索甲烷调校条文")
    assert result["state"] == "completed"
    assert len(capture.requests) == 2
    outbound = json.dumps(capture.requests, ensure_ascii=False)
    assert PRIVATE not in outbound
    assert "甲烷调校应记录结果" in outbound


def test_legacy_brief_cloud_search_never_returns_local_only_knowledge(parts):
    capture = Capture()
    parts.services["agent"] = AgentService(parts.store, parts.services, parts.settings, adapter_factory=lambda _: capture)
    briefs = EventBriefsService(parts.store, parts.services, parts.settings)
    submitted = briefs.submit(parts.context, "synthetic-event", 1)
    result = briefs.run(submitted["task_id"])
    assert result["state"] == "completed"
    assert len(capture.requests) == 2
    outbound = json.dumps(capture.requests, ensure_ascii=False)
    assert PRIVATE not in outbound
    assert "甲烷调校应记录结果" in outbound


def test_legacy_cloud_chat_does_not_reuse_local_only_knowledge_from_saved_history(parts):
    local_settings = parts.settings.model_copy(update={"deepseek_api_key": None})
    local_agent = AgentService(parts.store, parts.services, local_settings)
    local = local_agent.chat(parts.context, "检索甲烷调校条文")
    assert PRIVATE in local["final_text"]
    capture = Capture()
    cloud_agent = AgentService(parts.store, parts.services, parts.settings, adapter_factory=lambda _: capture)
    cloud_agent.chat(parts.context, "继续检索公开甲烷调校条文")
    assert PRIVATE not in json.dumps(capture.requests, ensure_ascii=False)


def test_queued_cloud_task_rechecks_inherited_references_before_its_first_tool(parts):
    capture = Capture()
    capture.responses[1] = ChatReply({"role": "assistant", "content": "甲烷调校应记录结果。"}, "stop", "offline-fake", {})
    agent = AgentService(parts.store, parts.services, parts.settings, adapter_factory=lambda _: capture)
    agent.chat(parts.context, "检索甲烷调校条文")
    jobs = []
    queued = agent.submit(parts.context, "继续讨论", schedule=jobs.append)
    assert queued["tool_calls"] == [] and queued["cloud_reference_ids"]
    reference = parts.knowledge.search_public_documents(parts.context, "甲烷调校")[0]
    parts.knowledge.set_disclosure(parts.actor, reference["entry_id"], reference["version"], "local_only", comment="排队期间撤回公开用途")
    capture.responses.append(ChatReply({"role": "assistant", "content": "请重新检索公开依据。"}, "stop", "offline-fake", {}))
    jobs[0]()
    assert "甲烷调校应记录结果。" not in json.dumps(capture.requests[-1], ensure_ascii=False)
