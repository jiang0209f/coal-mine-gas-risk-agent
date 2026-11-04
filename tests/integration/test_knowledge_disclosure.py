"""Public normative exports stay version-bound; private field content stays local."""
from datetime import datetime, timezone
import json

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.knowledge.service import KnowledgeService


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="manager-private-canary", roles=("safety_management",), face_ids=("FACE-PRIVATE",))
    store.put("source", "source-private-canary", {"source_id": "source-private-canary", "face_id": "FACE-PRIVATE"})
    context = ExecutionContext(actor=actor, session_id="session-private-canary", source_id="source-private-canary",
        mode="historical_replay", as_of=datetime(2022, 1, 1, tzinfo=timezone.utc))
    knowledge = KnowledgeService(store)
    yield store, actor, context, knowledge
    store.close()


def material(parts, text="甲烷传感器调校应记录结果。", **extra):
    _, actor, _, knowledge = parts
    spec = {"title": "公开标准测试条文", "version": "v1", "source": "https://www.mem.gov.cn/test.pdf",
        "applicability": {"face_ids": ["FACE-PRIVATE"]}, "fragments": [{"text": text, "locator": {"page": 7, "original_file": "F:/PRIVATE-PATH/file.pdf", "clause": "8.4.6"}}], **extra}
    doc = knowledge.import_reference(actor, spec)
    if extra.get("approve", True):
        knowledge.review_reference(actor, doc["entry_id"], doc["version"], "approved")
    return doc


def publish(parts, doc):
    _, actor, _, knowledge = parts
    return knowledge.set_disclosure(actor, doc["entry_id"], doc["version"], "public_normative",
        official_url="https://www.mem.gov.cn/test.pdf", category="standard", comment="用户授权公开法规标准；核对官方原文")


def test_old_and_explicit_import_metadata_do_not_silently_enable_exports(parts):
    _, _, context, knowledge = parts
    material(parts, disclosure="public_normative")
    assert knowledge.search_documents(context, "甲烷")
    assert knowledge.search_public_documents(context, "甲烷") == []


def test_public_only_search_and_export_projection_hide_all_private_metadata(parts):
    _, _, context, knowledge = parts
    public = material(parts)
    material(parts, text="PRIVATE-TEXT-DO-NOT-SEND 甲烷调校。", title="矿井真实规程")
    policy = publish(parts, public)
    assert policy["disclosure"] == "public_normative"
    hits = knowledge.search_public_documents(context, "探头校准")
    assert hits and all(row["entry_id"] == public["entry_id"] for row in hits)
    exported = knowledge.approved_public_fragment(context, hits[0]["reference_id"])
    raw = json.dumps(exported, ensure_ascii=False)
    assert exported["official_url"] == "https://www.mem.gov.cn/test.pdf"
    assert exported["state"] == "approved" and exported["effective"] is True
    assert exported["locator"] == {"page": 7, "clause": "8.4.6"}
    assert not any(value in raw for value in ("PRIVATE-TEXT", "PRIVATE-PATH", "FACE-PRIVATE", "manager-private", "session-private", "source-private"))


@pytest.mark.parametrize("url", ["http://www.mem.gov.cn/test.pdf", "https://www.mem.gov.cn.attacker.test/x", "https://mem.gov.cn@attacker.test/x", "file:///F:/private.pdf", "https://attacker.test/?source=mem.gov.cn"])
def test_public_policy_requires_https_official_origin(parts, url):
    _, actor, _, knowledge = parts
    doc = material(parts)
    with pytest.raises(DomainError):
        knowledge.set_disclosure(actor, doc["entry_id"], "v1", "public_normative", official_url=url,
            category="standard", comment="测试错误来源")


def test_reader_cannot_change_disclosure_and_pending_is_not_exportable(parts):
    _, actor, context, knowledge = parts
    doc = material(parts, approve=False)
    with pytest.raises(DomainError):
        publish(parts, doc)
    knowledge.review_reference(actor, doc["entry_id"], "v1", "approved")
    reader = Actor(user_id="reader", roles=("ventilation",), face_ids=("FACE-PRIVATE",))
    with pytest.raises(DomainError):
        knowledge.set_disclosure(reader, doc["entry_id"], "v1", "public_normative", official_url="https://www.mem.gov.cn/test.pdf", category="standard", comment="无权")
    assert knowledge.search_public_documents(context, "甲烷") == []


def test_new_revision_requires_its_own_disclosure_and_revocation_is_immediate(parts):
    _, actor, context, knowledge = parts
    doc = material(parts)
    publish(parts, doc)
    old_ref = knowledge.search_public_documents(context, "甲烷")[0]["reference_id"]
    material(parts, text="新版甲烷传感器调校条文。", version="v2", entry_id=doc["entry_id"])
    assert knowledge.search_public_documents(context, "甲烷") == []
    with pytest.raises(DomainError):
        knowledge.approved_public_fragment(context, old_ref)
    publish(parts, {**doc, "version": "v2"})
    new_ref = knowledge.search_public_documents(context, "甲烷")[0]["reference_id"]
    knowledge.set_disclosure(actor, doc["entry_id"], "v2", "local_only", comment="撤回公开用途")
    assert knowledge.search_public_documents(context, "甲烷") == []
    with pytest.raises(DomainError):
        knowledge.approved_public_fragment(context, new_ref)


def test_modified_public_body_fails_closed_even_if_stored_policy_remains(parts):
    store, _, context, knowledge = parts
    doc = material(parts)
    publish(parts, doc)
    hit = knowledge.search_public_documents(context, "甲烷")[0]
    raw = store.get("reference_fragment", hit["reference_id"])
    store.put("reference_fragment", hit["reference_id"], {**raw, "text": "PRIVATE-TEXT-DO-NOT-SEND 甲烷。"})
    assert knowledge.search_public_documents(context, "甲烷") == []
    with pytest.raises(DomainError):
        knowledge.approved_public_fragment(context, hit["reference_id"])


def test_registered_open_conflict_is_shown_but_blocks_advice_until_explicitly_resolved(parts):
    _, actor, context, knowledge = parts
    one = material(parts, "同一设备调校周期为三天。")
    two = material(parts, "同一设备调校周期为十五天。", title="另一现行资料")
    hits = knowledge.search_documents(context, "调校")
    ids = [row["reference_id"] for row in hits]
    record = {"conflict_id": "test-conflict", "reference_ids": ids, "topic": "同一设备调校周期", "applicability_condition": "合成同一设备同一时间范围", "reason": "两份原文不一致，需签批依据", "status": "open"}
    knowledge.set_conflict(actor, record)
    assert all(row["conflict_status"] == "open" for row in knowledge.search_documents(context, "调校"))
    event = {"event_id": "event", "event_revision": 1, "session_id": context.session_id, "source_id": context.source_id, "face_id": "FACE-PRIVATE", "revision_as_of": context.as_of.isoformat()}
    with pytest.raises(DomainError) as caught:
        knowledge.draft_advice(context, {"event": event}, ids)
    assert caught.value.code == "knowledge_conflict_unresolved"
    record.update(status="resolved", resolution_reference={"reference_id": ids[1]})
    knowledge.set_conflict(actor, record)
    assert all(row["conflict_status"] == "resolved" for row in knowledge.search_documents(context, "调校"))


def test_private_conflict_notes_are_not_in_public_export(parts):
    _, actor, context, knowledge = parts
    public = material(parts)
    private = material(parts, text="PRIVATE-TEXT-DO-NOT-SEND 调校。", title="私有记录")
    publish(parts, public)
    rows = knowledge.search_documents(context, "调校")
    ids = [row["reference_id"] for row in rows]
    knowledge.set_conflict(actor, {"conflict_id": "private-conflict", "reference_ids": ids, "topic": "PRIVATE-TOPIC", "applicability_condition": "PRIVATE-CONDITION", "reason": "PRIVATE-REASON", "status": "open"})
    hit = knowledge.search_public_documents(context, "调校")[0]
    exported = knowledge.approved_public_fragment(context, hit["reference_id"])
    assert "PRIVATE-" not in json.dumps(exported, ensure_ascii=False)


def test_later_conflict_invalidates_an_earlier_advice_before_review(parts):
    _, actor, context, knowledge = parts
    material(parts, "同一设备调校周期为三天。")
    rows = knowledge.search_documents(context, "调校")
    event = {"event_id": "event", "event_revision": 1, "session_id": context.session_id, "source_id": context.source_id, "face_id": "FACE-PRIVATE", "revision_as_of": context.as_of.isoformat()}
    advice = knowledge.draft_advice(context, {"event": event}, [rows[0]["reference_id"]])
    material(parts, "同一设备调校周期为十五天。", title="另一现行资料")
    ids = [row["reference_id"] for row in knowledge.search_documents(context, "调校")]
    knowledge.set_conflict(actor, {"conflict_id": "later", "reference_ids": ids, "topic": "调校周期", "applicability_condition": "合成同设备", "reason": "新增分歧", "status": "open"})
    with pytest.raises(DomainError) as caught:
        knowledge.validate_advice_references(context, advice)
    assert caught.value.code == "knowledge_conflict_unresolved"


def test_narrow_manager_cannot_publish_material_covering_another_face(parts):
    _, actor, _, knowledge = parts
    wide = Actor(user_id="wide", roles=("safety_management",), face_ids=("FACE-PRIVATE", "OTHER"))
    doc = knowledge.import_reference(wide, {"title": "跨区合成公开标准", "source": "https://www.mem.gov.cn/test.pdf", "version": "v1", "applicability": {"face_ids": ["FACE-PRIVATE", "OTHER"]}, "text": "合成调校。", "locator": {"page": 1}})
    knowledge.review_reference(wide, doc["entry_id"], "v1", "approved")
    with pytest.raises(DomainError):
        publish(parts, doc)


def test_configured_offline_embedding_is_shared_by_local_and_public_search(tmp_path, monkeypatch):
    from app.config import Settings
    from app.main import build_services
    from app.adapters import text_embeddings
    captured = []
    class FakeLocalEmbeddings:
        def __init__(self, model_dir, index_dir):
            captured.append((model_dir, index_dir))
        def __call__(self, texts):
            return [[1.0, 0.0] for _ in texts]
    monkeypatch.setattr(text_embeddings, "LocalTextEmbeddings", FakeLocalEmbeddings)
    model_dir, index_dir = tmp_path / "offline-model", tmp_path / "local-index"
    settings = Settings(_env_file=None, runtime_dir=tmp_path, database_url="sqlite+pysqlite:///:memory:",
        knowledge_embedding_model_dir=model_dir, knowledge_index_dir=index_dir,
        ingestion_worker_enabled=False, realtime_monitor_enabled=False)
    services = build_services(settings)
    try:
        assert captured == [(model_dir, index_dir)]
        assert services["public_knowledge"].knowledge is services["knowledge"]
        assert isinstance(services["knowledge"].embedding, FakeLocalEmbeddings)
    finally:
        services["store"].close()


def test_local_knowledge_answer_reports_registered_conflict_without_choosing_a_period(parts):
    from app.config import Settings
    from app.services.agent.service import AgentService
    store, actor, context, knowledge = parts
    material(parts, "同一设备调校周期为三天。")
    material(parts, "同一设备调校周期为十五天。", title="另一现行资料")
    ids = [row["reference_id"] for row in knowledge.search_documents(context, "调校")]
    knowledge.set_conflict(actor, {"conflict_id": "local-answer", "reference_ids": ids, "topic": "周期", "applicability_condition": "同一合成设备", "reason": "两份原文不同", "status": "open"})
    agent = AgentService(store, {"knowledge": knowledge}, Settings(_env_file=None, deepseek_api_key=None))
    result = agent.chat(context, "检索调校规程")
    assert result["state"] == "partial" and "分歧" in result["final_text"]
    assert "knowledge_conflict_resolution" in result["missing_fields"]
