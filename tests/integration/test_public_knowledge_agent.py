"""Public-only cloud requests are captured offline; no real API is called."""

from copy import deepcopy
from datetime import datetime, timezone
import json

import httpx
import pytest

from app.adapters.deepseek import ChatReply, DeepSeekAdapter, DeepSeekRequestError
from app.config import Settings
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.agent.public_knowledge import PublicKnowledgeService


CANARY = "PRIVATE_MINE_CANARY_7319"


class CapturingAdapter:
    def __init__(self, payload=None, error=None):
        self.payload = payload or {"answer": "系统应保留故障记录。[PUB-001]", "reference_ids": ["PUB-001"], "conflicts": []}
        self.error = error
        self.requests = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.requests.append({"messages": deepcopy(messages), **deepcopy(kwargs)})
        if self.error:
            raise self.error
        return ChatReply({"role": "assistant", "content": json.dumps(self.payload, ensure_ascii=False)},
                         "stop", "offline-capturing-adapter", {"total_tokens": 12})

    def close(self):
        self.closed = True


class LocalKnowledge:
    def __init__(self, fragments):
        self.fragments = {item["reference_id"]: deepcopy(item) for item in fragments}
        self.queries = []
        self.rechecked = []

    def search_public_documents(self, context, query):
        self.queries.append(query)
        return deepcopy(list(self.fragments.values()))

    def approved_public_fragment(self, context, reference_id):
        self.rechecked.append(reference_id)
        return deepcopy(self.fragments[reference_id])


def fragment(reference_id="local-public-ref", **changes):
    return {
        "reference_id": reference_id, "title": "煤矿安全监控系统通用技术要求", "version": "2019",
        "text": "系统应具有自诊断功能。", "locator": {"page": 14, "clause": "5.5.8", "local_path": CANARY},
        "official_url": "https://www.mem.gov.cn/fw/flfgbz/bz/bzwb/201909/P020220607369920865461.pdf",
        "state": "approved", "effective": True, "disclosure": "public_normative",
        "conflicts": [], "conflict_status": "none",
        "source_id": CANARY, "face_id": CANARY, "file_ref": "F:/" + CANARY,
        "entry_id": CANARY, "reviewer_id": CANARY, "applicability": {"face_ids": [CANARY]},
        **changes,
    }


@pytest.fixture(autouse=True)
def isolate_deepseek_environment(monkeypatch):
    import os

    for name in tuple(os.environ):
        if name.upper().startswith("DEEPSEEK_"):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def context():
    return ExecutionContext(
        session_id="session-" + CANARY, source_id="source-" + CANARY,
        mode="realtime", as_of=datetime(2026, 11, 4, tzinfo=timezone.utc),
        actor=Actor(user_id="user-" + CANARY, roles=("safety_management",), face_ids=("31002-" + CANARY,)),
    )


def service(fragments=None, *, payload=None, error=None, configured=True):
    knowledge = LocalKnowledge(fragments if fragments is not None else [fragment()])
    adapter = CapturingAdapter(payload, error)
    settings = Settings(_env_file=None, deepseek_api_key="offline-test-key" if configured else None)
    agent = PublicKnowledgeService(knowledge, settings, adapter_factory=lambda _: adapter)
    return agent, knowledge, adapter


@pytest.mark.parametrize("confirmed", [False, None, "true", 1])
def test_public_mode_requires_explicit_boolean_authorization(context, confirmed):
    agent, knowledge, adapter = service()
    with pytest.raises(DomainError) as error:
        agent.run(context, "解释公开条文", public_query_confirmed=confirmed)
    assert error.value.code == "public_mode_required"
    assert not knowledge.queries and not adapter.requests


def test_default_mode_and_system_actor_do_not_trigger_cloud(context):
    agent, _, adapter = service()
    with pytest.raises(DomainError):
        agent.run(context, "解释公开条文")
    system = context.model_copy(update={"actor": context.actor.model_copy(update={"is_system": True})})
    with pytest.raises(DomainError):
        agent.run(system, "解释公开条文", public_query_confirmed=True)
    assert not adapter.requests


def test_mixed_question_context_private_ids_and_extra_metadata_never_leave_machine(context):
    agent, knowledge, adapter = service([fragment("ref-" + CANARY)])
    question = f"31002工作面测点T1现场浓度0.73%，{CANARY}，张三已处置，请解释故障记录标准。"
    result = agent.run(context, question, public_query_confirmed=True)
    assert knowledge.queries == [question]
    assert knowledge.rechecked == ["ref-" + CANARY]
    outbound = json.dumps(adapter.requests, ensure_ascii=False)
    for private_value in (CANARY, "31002", "T1", "0.73", "张三", question, context.source_id, context.session_id):
        assert private_value not in outbound
    assert len(adapter.requests) == 1
    assert set(adapter.requests[0]) == {"messages", "tools", "tool_choice"}
    assert adapter.requests[0]["tools"] == []
    assert adapter.requests[0]["tool_choice"] == "none"
    assert [message["role"] for message in adapter.requests[0]["messages"]] == ["system", "user"]
    assert result["state"] == "completed" and result["public_context_only"] is True
    assert result["citations"][0]["reference_id"] == "ref-" + CANARY
    assert result["citations"][0]["public_reference_id"] == "PUB-001"
    assert adapter.closed


@pytest.mark.parametrize("change", [
    {"state": "draft"}, {"state": "superseded"}, {"effective": False},
    {"disclosure": "local_only"}, {"disclosure": None}, {"disclosure": "public"},
    {"official_url": "http://www.mem.gov.cn/a.pdf"},
    {"official_url": "https://www.mem.gov.cn.evil.example/a.pdf"},
    {"official_url": "https://example.org/a.pdf"},
    {"official_url": "https://user:secret@www.mem.gov.cn/a.pdf"},
    {"official_url": "https://www.mem.gov.cn:8443/a.pdf"},
    {"official_url": "file:///F:/private.pdf"},
])
def test_only_explicit_current_approved_public_normative_official_fragments_are_sent(context, change):
    agent, _, adapter = service([fragment("denied", text=CANARY, **change), fragment("allowed")])
    result = agent.run(context, "说明标准要求", public_query_confirmed=True)
    assert result["state"] == "completed"
    assert CANARY not in json.dumps(adapter.requests, ensure_ascii=False)
    assert [item["reference_id"] for item in result["references"]] == ["allowed"]


def test_rechecks_locally_and_cannot_trust_search_results_alone(context):
    agent, knowledge, adapter = service()
    knowledge.approved_public_fragment = lambda *_: (_ for _ in ()).throw(DomainError("已失效"))
    result = agent.run(context, "解释公开要求", public_query_confirmed=True)
    assert result["state"] == "not_ready"
    assert not adapter.requests


def test_missing_public_evidence_or_key_does_not_call_cloud(context):
    agent, _, adapter = service([])
    result = agent.run(context, "解释标准", public_query_confirmed=True)
    assert result["state"] == "not_ready" and result["error_code"] == "public_evidence_missing"
    assert not adapter.requests
    agent, _, adapter = service(configured=False)
    assert agent.run(context, "解释标准", public_query_confirmed=True)["error_code"] == "model_not_configured"
    assert not adapter.requests


def test_bounded_public_fragments_and_clean_locator_projection(context):
    agent, _, adapter = service([fragment(f"r{i}", text="公开内容。" * 2000) for i in range(35)])
    result = agent.run(context, "公开条文", public_query_confirmed=True)
    payload = json.loads(adapter.requests[0]["messages"][1]["content"])
    references = payload["references"]
    assert len(references) <= 8
    assert sum(len(item["text"]) for item in references) <= 20000
    assert all(len(item["text"]) <= 4000 for item in references)
    assert all(set(item["locator"]) == {"page", "clause"} for item in references)
    assert CANARY not in json.dumps(adapter.requests, ensure_ascii=False)
    assert result["references"]


@pytest.mark.parametrize("payload", [
    {"answer": "引用假条文[PUB-999]", "reference_ids": ["PUB-999"], "conflicts": []},
    {"answer": "正文夹带假引用[PUB-999]", "reference_ids": ["PUB-001"], "conflicts": []},
    {"answer": "无引用的结论", "reference_ids": [], "conflicts": []},
    {"answer": "结论[PUB-001]", "reference_ids": ["PUB-001"], "conflicts": [{"reference_ids": ["PUB-001", "PUB-999"], "description": "冲突"}]},
])
def test_unknown_or_missing_citations_are_not_accepted(context, payload):
    agent, _, adapter = service(payload=payload)
    result = agent.run(context, "解释公开条文", public_query_confirmed=True)
    assert result["state"] == "partial" and result["error_code"] == "unverified_public_reference"
    assert "假条文" not in result["final_text"] and "结论" not in result["final_text"]
    assert adapter.closed


def test_conflicts_are_parallel_public_citations_and_never_automated_actions(context):
    payload = {"answer": "条文要求存在差异，应结合适用条件复核。[PUB-001][PUB-002]",
               "reference_ids": ["PUB-001", "PUB-002"],
               "conflicts": [{"reference_ids": ["PUB-001", "PUB-002"], "description": "适用条件不同，应并列核对。"}]}
    agent, _, adapter = service([fragment("r1"), fragment("r2", version="2020")], payload=payload)
    result = agent.run(context, "比较公开条文", public_query_confirmed=True)
    assert result["state"] == "completed"
    assert len(result["conflicts"][0]["reference_ids"]) == 2
    assert not result.get("actions") and result["automatic_actions"] is False
    assert "冲突" in adapter.requests[0]["messages"][0]["content"]


def test_provider_error_is_bounded_and_adapter_closed(context):
    agent, _, adapter = service(error=DeepSeekRequestError("private provider detail " + CANARY, 503))
    result = agent.run(context, "公开标准", public_query_confirmed=True)
    assert result["state"] == "failed" and result["error_code"] == "model_service_unavailable"
    assert CANARY not in result["final_text"]
    assert adapter.closed


def test_registered_conflict_fetches_both_public_parties_and_strips_local_notes(context):
    conflict = {"reference_ids": ["r1", "r2"], "reason": CANARY, "conflict_id": CANARY, "status": "open"}
    payload = {"answer": "两处条文应按适用条件并列复核。[PUB-001][PUB-002]",
               "reference_ids": ["PUB-001", "PUB-002"],
               "conflicts": [{"reference_ids": ["PUB-001", "PUB-002"], "description": "条文存在待复核差异。"}]}
    agent, knowledge, adapter = service([
        fragment("r1", conflicts=[conflict], conflict_status="open"),
        fragment("r2", conflicts=[conflict], conflict_status="open"),
    ], payload=payload)
    knowledge.search_public_documents = lambda *_: [{"reference_id": "r1"}]
    result = agent.run(context, "比较公开条文", public_query_confirmed=True)
    assert result["state"] == "completed"
    assert knowledge.rechecked == ["r1", "r2"]
    outbound = json.loads(adapter.requests[0]["messages"][1]["content"])
    assert outbound["references"][0]["conflicts"][0]["reference_ids"] == ["PUB-001", "PUB-002"]
    assert CANARY not in json.dumps(adapter.requests, ensure_ascii=False)


def test_private_or_missing_conflict_party_prevents_cloud_generation(context):
    conflict = {"references": [{"reference_id": "r1"}, {"reference_id": "r2"}], "status": "possible_conflict"}
    agent, _, adapter = service([
        fragment("r1", conflicts=[conflict], conflict_status="possible_conflict"),
        fragment("r2", disclosure="local_only", text=CANARY),
    ])
    result = agent.run(context, "比较条文", public_query_confirmed=True)
    assert result["state"] == "not_ready" and result["error_code"] == "public_conflict_evidence_incomplete"
    assert not adapter.requests


def test_known_conflict_cannot_be_answered_with_only_one_party(context):
    conflict = {"reference_ids": ["r1", "r2"], "status": "open"}
    agent, _, _ = service([
        fragment("r1", conflicts=[conflict], conflict_status="open"),
        fragment("r2", conflicts=[conflict], conflict_status="open"),
    ])
    result = agent.run(context, "比较条文", public_query_confirmed=True)
    assert result["state"] == "partial" and result["error_code"] == "unverified_public_reference"


@pytest.mark.parametrize("unknown", ["PUB-999", "ＰＵＢ－９９９", "pub-999", "PUB\u2011999"])
def test_chinese_adjacent_unknown_citation_is_rejected(context, unknown):
    payload = {"answer": f"根据{unknown}规定进行解释。[PUB-001]", "reference_ids": ["PUB-001"], "conflicts": []}
    agent, _, _ = service(payload=payload)
    result = agent.run(context, "公开条文", public_query_confirmed=True)
    assert result["state"] == "partial" and result["error_code"] == "unverified_public_reference"


def test_unknown_citation_in_conflict_description_is_rejected(context):
    payload = {"answer": "并列两项条文。[PUB-001][PUB-002]", "reference_ids": ["PUB-001", "PUB-002"],
               "conflicts": [{"reference_ids": ["PUB-001", "PUB-002"], "description": "根据PUB-999可以确定处理。"}]}
    agent, _, _ = service([fragment("r1"), fragment("r2")], payload=payload)
    result = agent.run(context, "比较条文", public_query_confirmed=True)
    assert result["state"] == "partial" and result["error_code"] == "unverified_public_reference"


def test_actual_adapter_wire_json_contains_only_fixed_public_request(context):
    captured = []

    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "deepseek-chat", "usage": {"total_tokens": 12},
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({
                "answer": "应记录故障。[PUB-001]", "reference_ids": ["PUB-001"], "conflicts": []})}}]})

    knowledge = LocalKnowledge([fragment("ref-" + CANARY)])
    settings = Settings(_env_file=None, deepseek_api_key="offline-test-key", deepseek_max_retries=0)
    agent = PublicKnowledgeService(knowledge, settings,
        adapter_factory=lambda value: DeepSeekAdapter(value, transport=httpx.MockTransport(handle)))
    result = agent.run(context, "张三已记录现场读数" + CANARY, public_query_confirmed=True)
    assert result["state"] == "completed"
    assert len(captured) == 1
    assert CANARY not in json.dumps(captured, ensure_ascii=False)
    assert "张三" not in json.dumps(captured, ensure_ascii=False)
    assert captured[0]["tools"] == [] and captured[0]["tool_choice"] == "none"


def test_unrelated_url_parameters_and_fragments_stay_local(context):
    url = "https://std.samr.gov.cn/hb/search/stdHBDetailed?id=91C6420EB3AC22E7E05397BE0A0A8398&local=" + CANARY + "#" + CANARY
    agent, _, adapter = service([fragment(official_url=url)])
    result = agent.run(context, "公开标准", public_query_confirmed=True)
    assert result["state"] == "completed"
    assert CANARY not in json.dumps(adapter.requests, ensure_ascii=False)
    assert "id=91C6420EB3AC22E7E05397BE0A0A8398" in result["citations"][0]["official_url"]
