"""Local reranking receives only candidates already authorized for that entry point."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from app.config import Settings
from app.db import Store
from app.main import build_services
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.knowledge.service import KnowledgeService


NOW = datetime(2026, 11, 5, 12, tzinfo=timezone.utc)
DENIED = "DENIED_CANDIDATE_CANARY"
PRIVATE = "LOCAL_ONLY_CANDIDATE_CANARY"


class CapturingEmbedding:
    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class CapturingReranker:
    def __init__(self):
        self.calls = []

    def __call__(self, query, passages):
        self.calls.append({"query": query, "passages": list(passages)})
        return [.9 for _ in passages]


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    import os
    for name in tuple(os.environ):
        if name.upper().startswith(("DEEPSEEK_", "KNOWLEDGE_")):
            monkeypatch.delenv(name, raising=False)


def add_document(knowledge, actor, text, *, public=False, approved=True, faces=None,
                 valid_from=None, valid_to=None, entry_id=None, version="v1"):
    applicability = {"face_ids": faces or ["FACE-A"]}
    if valid_from:
        applicability["valid_from"] = valid_from
    if valid_to:
        applicability["valid_to"] = valid_to
    spec = {"title": "报警处理记录条文", "version": version, "source": "https://www.mem.gov.cn/public.pdf",
            "applicability": applicability, "fragments": [{"text": text,
                "locator": {"page": 7, "clause": "9.2.2", "file_path": "PRIVATE-LOCAL-PATH"}}]}
    if entry_id:
        spec["entry_id"] = entry_id
    document = knowledge.import_reference(actor, spec)
    if approved:
        original_clock = knowledge.clock
        effective_review_at = original_clock()
        if valid_from and effective_review_at < datetime.fromisoformat(valid_from):
            effective_review_at = datetime.fromisoformat(valid_from)
        if valid_to and effective_review_at >= datetime.fromisoformat(valid_to):
            effective_review_at = datetime.fromisoformat(valid_to) - timedelta(seconds=1)
        knowledge.clock = lambda: effective_review_at
        try:
            knowledge.review_reference(actor, document["entry_id"], version, "approved")
        finally:
            knowledge.clock = original_clock
    if public:
        knowledge.set_disclosure(actor, document["entry_id"], version, "public_normative",
            official_url="https://www.mem.gov.cn/public.pdf", category="standard", comment="合成公开用途集成测试")
    return knowledge.get_reference(actor, document["entry_id"], version)


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    embedding, reranker = CapturingEmbedding(), CapturingReranker()
    knowledge = KnowledgeService(store, clock=lambda: NOW, embedding=embedding, reranker=reranker)
    manager = Actor(user_id="manager", roles=("safety_management",), face_ids=("FACE-A", "FACE-B"))
    reader = Actor(user_id="reader", roles=("ventilation",), face_ids=("FACE-A",))
    store.put("source", "source-a", {"source_id": "source-a", "face_id": "FACE-A"})
    store.put("source", "source-b", {"source_id": "source-b", "face_id": "FACE-B"})
    context = ExecutionContext(session_id="session", source_id="source-a", mode="historical_replay",
        as_of=datetime(2022, 1, 1, tzinfo=timezone.utc), actor=reader)
    old = add_document(knowledge, manager, DENIED + " 旧版报警记录。", public=True)
    public = add_document(knowledge, manager, "PUBLIC_CURRENT 报警处理结果应记录。", public=True,
        entry_id=old["entry_id"], version="v2")
    second = add_document(knowledge, manager, "PUBLIC_SECOND 报警记录包括处理时间。", public=True)
    private = add_document(knowledge, manager, PRIVATE + " 报警处理记录。")
    add_document(knowledge, manager, DENIED + " 待核报警记录。", approved=False)
    add_document(knowledge, manager, DENIED + " 已过期报警记录。", valid_to="2026-11-04T00:00:00+00:00")
    add_document(knowledge, manager, DENIED + " 尚未生效报警记录。", valid_from="2026-11-06T00:00:00+00:00")
    add_document(knowledge, manager, DENIED + " 另一工作面报警记录。", faces=["FACE-B"])
    yield {"store": store, "knowledge": knowledge, "embedding": embedding, "reranker": reranker,
           "manager": manager, "context": context, "public": public, "second": second, "private": private}
    store.close()


def test_reranker_setting_is_optional_and_environment_path_is_supported(monkeypatch, tmp_path):
    assert Settings(_env_file=None).knowledge_reranker_model_dir is None
    path = tmp_path / "local-reranker"
    monkeypatch.setenv("KNOWLEDGE_RERANKER_MODEL_DIR", str(path))
    assert Settings(_env_file=None).knowledge_reranker_model_dir == path


@pytest.mark.parametrize("value", ["", "   "])
def test_empty_reranker_configuration_stays_disabled(monkeypatch, value):
    monkeypatch.setenv("KNOWLEDGE_RERANKER_MODEL_DIR", value)
    assert Settings(_env_file=None).knowledge_reranker_model_dir is None


def test_public_policy_filters_before_embedding_and_reranking(parts):
    rows = parts["knowledge"].search_public_documents(parts["context"], "报警处理记录")
    assert {row["entry_id"] for row in rows} == {parts["public"]["entry_id"], parts["second"]["entry_id"]}
    assert parts["embedding"].calls and parts["reranker"].calls
    model_inputs = json.dumps({"embedding": parts["embedding"].calls, "reranker": parts["reranker"].calls})
    assert PRIVATE not in model_inputs and DENIED not in model_inputs
    assert "PUBLIC_CURRENT" in model_inputs and "PUBLIC_SECOND" in model_inputs


def test_local_entry_still_searches_private_documents_but_filters_invalid_scope_and_dates(parts):
    rows = parts["knowledge"].search_documents(parts["context"], "报警处理记录")
    assert parts["private"]["entry_id"] in {row["entry_id"] for row in rows}
    model_inputs = json.dumps({"embedding": parts["embedding"].calls, "reranker": parts["reranker"].calls})
    assert PRIVATE in model_inputs and DENIED not in model_inputs


def test_unauthorized_source_is_rejected_before_either_local_model(parts):
    foreign = parts["context"].model_copy(update={"source_id": "source-b"})
    with pytest.raises(DomainError) as error:
        parts["knowledge"].search_public_documents(foreign, "报警处理记录")
    assert error.value.status_code == 403
    assert not parts["embedding"].calls and not parts["reranker"].calls


def test_revocation_removes_public_body_before_both_models_on_next_search(parts):
    document = parts["public"]
    parts["knowledge"].set_disclosure(parts["manager"], document["entry_id"], document["version"],
        "local_only", comment="撤销公开用途")
    rows = parts["knowledge"].search_public_documents(parts["context"], "报警处理记录")
    assert {row["entry_id"] for row in rows} == {parts["second"]["entry_id"]}
    model_inputs = json.dumps({"embedding": parts["embedding"].calls, "reranker": parts["reranker"].calls})
    assert "PUBLIC_CURRENT" not in model_inputs and PRIVATE not in model_inputs and DENIED not in model_inputs


def test_public_fragment_projection_is_unchanged_by_reranking(parts):
    hit = parts["knowledge"].search_public_documents(parts["context"], "报警处理记录")[0]
    public = parts["knowledge"].approved_public_fragment(parts["context"], hit["reference_id"])
    assert set(public) == {"reference_id", "entry_id", "document_id", "title", "version", "text", "locator",
        "source", "official_url", "state", "effective", "disclosure", "conflict_status", "conflicts"}
    assert public["locator"] == {"page": 7, "clause": "9.2.2"}
    assert public["state"] == "approved" and public["effective"] is True and public["disclosure"] == "public_normative"
    assert "PRIVATE-LOCAL-PATH" not in json.dumps(public)


def test_unconfigured_service_keeps_existing_lexical_search():
    store = Store("sqlite+pysqlite:///:memory:")
    try:
        knowledge = KnowledgeService(store, clock=lambda: NOW)
        actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("FACE-A",))
        store.put("source", "source-a", {"source_id": "source-a", "face_id": "FACE-A"})
        document = add_document(knowledge, actor, "报警处理结果应记录。")
        context = ExecutionContext(actor=actor, session_id="session", source_id="source-a", mode="realtime", as_of=NOW)
        rows = knowledge.search_documents(context, "报警处理记录")
        assert knowledge.reranker is None
        assert [row["entry_id"] for row in rows] == [document["entry_id"]]
    finally:
        store.close()


@pytest.mark.parametrize("configured", [False, True])
def test_main_builds_one_optional_local_reranker_shared_by_both_entry_points(monkeypatch, tmp_path, configured):
    constructed = []
    reranker = CapturingReranker()
    module = ModuleType("app.adapters.text_reranking")

    def factory(model_dir):
        constructed.append(Path(model_dir))
        return reranker

    module.LocalTextReranker = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    model_dir = tmp_path / "local-minilm" if configured else None
    settings = Settings(_env_file=None, runtime_dir=tmp_path / "runtime", database_url="sqlite+pysqlite:///:memory:",
        knowledge_embedding_model_dir=None, knowledge_reranker_model_dir=model_dir,
        ingestion_worker_enabled=False, realtime_monitor_enabled=False, deepseek_api_key=None)
    services = build_services(settings)
    try:
        assert services["public_knowledge"].knowledge is services["knowledge"]
        assert services["knowledge"].reranker is (reranker if configured else None)
        assert constructed == ([model_dir] if configured else [])
        actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("FACE-A",))
        services["store"].put("source", "source-a", {"source_id": "source-a", "face_id": "FACE-A"})
        add_document(services["knowledge"], actor, "报警处理结果应记录。", public=True)
        add_document(services["knowledge"], actor, "报警处理记录保留处置时间。", public=True)
        context = ExecutionContext(actor=actor, session_id="session", source_id="source-a", mode="realtime", as_of=NOW)
        assert services["knowledge"].search_documents(context, "报警处理记录")
        assert services["knowledge"].search_public_documents(context, "报警处理记录")
        assert len(reranker.calls) == (2 if configured else 0)
        assert constructed == ([model_dir] if configured else [])
    finally:
        services["store"].close()
