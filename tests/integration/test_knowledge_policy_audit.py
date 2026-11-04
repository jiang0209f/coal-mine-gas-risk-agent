"""Cross-face mutation and resolution disclosure remain explicit authority checks."""
from datetime import datetime, timezone
import json

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.knowledge.service import KnowledgeService


def document(knowledge, actor, *, public=False):
    row = knowledge.import_reference(actor, {"title": "公开标准条文" if public else "本机资料",
        "source": "https://www.mem.gov.cn/test.pdf" if public else "矿井本机记录", "version": "v1",
        "applicability": {"face_ids": list(actor.face_ids)},
        "fragments": [{"text": "甲烷传感器调校应记录结果。", "locator": {"page": 1}}]})
    knowledge.review_reference(actor, row["entry_id"], "v1", "approved")
    if public:
        knowledge.set_disclosure(actor, row["entry_id"], "v1", "public_normative",
            official_url="https://www.mem.gov.cn/test.pdf", category="standard", comment="核对公开原文")
    row = knowledge.get_reference(actor, row["entry_id"], "v1")
    return row["fragments"][0]["reference_id"]


def conflict(reference_ids, **extra):
    return {"conflict_id": "shared-client-conflict-id", "reference_ids": reference_ids,
        "topic": "甲烷传感器调校", "applicability_condition": "合成同一适用条件",
        "reason": "条文要求待核对", "status": "open", **extra}


def test_another_face_cannot_overwrite_existing_conflict_by_reusing_client_id():
    store = Store("sqlite+pysqlite:///:memory:")
    try:
        knowledge = KnowledgeService(store)
        one = Actor(user_id="one", roles=("safety_management",), face_ids=("FACE-A",))
        two = Actor(user_id="two", roles=("safety_management",), face_ids=("FACE-B",))
        one_refs = [document(knowledge, one), document(knowledge, one)]
        two_refs = [document(knowledge, two), document(knowledge, two)]
        knowledge.set_conflict(one, conflict(one_refs))
        with pytest.raises(DomainError):
            knowledge.set_conflict(two, conflict(two_refs))
        stored = store.get("knowledge_conflict", "shared-client-conflict-id")
        assert stored["face_ids"] == ["FACE-A"] and stored["reference_ids"] == one_refs
    finally:
        store.close()


def test_public_resolved_conflict_requires_public_resolution_evidence():
    store = Store("sqlite+pysqlite:///:memory:")
    try:
        knowledge = KnowledgeService(store)
        actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("FACE-A",))
        refs = [document(knowledge, actor, public=True), document(knowledge, actor, public=True)]
        private_resolution = document(knowledge, actor)
        with pytest.raises(DomainError):
            knowledge.set_conflict(actor, conflict(refs, status="resolved", disclosure="public_normative",
                resolution_reference={"reference_id": private_resolution}))
        store.put("source", "source", {"source_id": "source", "face_id": "FACE-A"})
        context = ExecutionContext(session_id="session", source_id="source", mode="realtime",
            as_of=datetime(2026, 11, 4, tzinfo=timezone.utc), actor=actor)
        assert all(row["conflict_status"] == "none" for row in knowledge.search_public_documents(context, "调校"))
    finally:
        store.close()


def test_resolution_must_cover_every_face_shared_by_the_conflicting_clauses():
    store = Store("sqlite+pysqlite:///:memory:")
    try:
        knowledge = KnowledgeService(store)
        wide = Actor(user_id="wide", roles=("safety_management",), face_ids=("FACE-A", "FACE-B"))
        narrow = Actor(user_id="narrow", roles=("safety_management",), face_ids=("FACE-A",))
        refs = [document(knowledge, wide), document(knowledge, wide)]
        resolution = document(knowledge, narrow)
        with pytest.raises(DomainError):
            knowledge.set_conflict(wide, conflict(refs, status="resolved", resolution_reference={"reference_id": resolution}))
        assert store.get("knowledge_conflict", "shared-client-conflict-id") is None
    finally:
        store.close()


def test_expired_resolution_reopens_the_conflict_for_current_advice():
    store = Store("sqlite+pysqlite:///:memory:")
    try:
        knowledge = KnowledgeService(store)
        actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("FACE-A",))
        refs = [document(knowledge, actor), document(knowledge, actor)]
        resolution = document(knowledge, actor)
        knowledge.set_conflict(actor, conflict(refs, status="resolved", resolution_reference={"reference_id": resolution}))
        fragment = store.get("reference_fragment", resolution)
        key = knowledge._key(fragment["entry_id"], fragment["version"])
        row = store.get("reference", key)
        row["applicability"]["valid_to"] = "2020-01-01T00:00:00+00:00"
        store.put("reference", key, row)
        store.put("source", "source", {"source_id": "source", "face_id": "FACE-A"})
        context = ExecutionContext(session_id="session", source_id="source", mode="realtime", as_of=datetime.now(timezone.utc), actor=actor)
        assert all(row["conflict_status"] == "open" for row in knowledge.search_documents(context, "调校"))
        with pytest.raises(DomainError):
            knowledge.validate_advice_references(context, {"reference_ids": refs, "face_id": "FACE-A"})
    finally:
        store.close()


@pytest.mark.parametrize("revoked", ["party", "resolution"])
def test_public_conflict_membership_and_resolution_are_rechecked_after_revocation(revoked):
    store = Store("sqlite+pysqlite:///:memory:")
    try:
        knowledge = KnowledgeService(store)
        actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("FACE-A",))
        refs = [document(knowledge, actor, public=True), document(knowledge, actor, public=True)]
        resolution = document(knowledge, actor, public=True)
        status = "resolved" if revoked == "resolution" else "open"
        extra = {"resolution_reference": {"reference_id": resolution}} if status == "resolved" else {}
        knowledge.set_conflict(actor, conflict(refs, status=status, disclosure="public_normative", **extra))
        revoked_id = refs[1] if revoked == "party" else resolution
        fragment = store.get("reference_fragment", revoked_id)
        knowledge.set_disclosure(actor, fragment["entry_id"], fragment["version"], "local_only", comment="撤回公开使用")
        store.put("source", "source", {"source_id": "source", "face_id": "FACE-A"})
        context = ExecutionContext(session_id="session", source_id="source", mode="realtime",
            as_of=datetime(2026, 11, 4, tzinfo=timezone.utc), actor=actor)
        exported = knowledge.approved_public_fragment(context, refs[0])
        assert exported["conflict_status"] == "none" and exported["conflicts"] == []
        assert revoked_id not in json.dumps(exported)
        assert any(row["conflict_status"] == status for row in knowledge.search_documents(context, "调校"))
    finally:
        store.close()
