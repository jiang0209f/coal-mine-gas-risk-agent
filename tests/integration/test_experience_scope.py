"""Reviewed experience retains its case's authorization scope."""
from datetime import datetime, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.cases.service import CasesService
from app.services.knowledge.service import KnowledgeService


@pytest.fixture
def reviewed():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002", "3411"))
    context = ExecutionContext(session_id="session", mode="historical_replay", source_id="synthetic",
        as_of=datetime(2022, 2, 5, tzinfo=timezone.utc), actor=actor)
    store.put("source", "synthetic", {"source_id": "synthetic", "face_id": "31002"})
    event = {"event_id": "event", "session_id": "session", "source_id": "synthetic", "face_id": "31002",
             "event_revision": 1, "revision_as_of": context.as_of.isoformat()}
    store.put("event", "event", event)
    knowledge = KnowledgeService(store)
    cases = CasesService(store, knowledge=knowledge)
    document = knowledge.import_reference(actor, {"title": "合成31002资料", "source": "合成测试", "version": "v1",
        "text": "合成原文。", "locator": {"line": 1}, "applicability": {"face_ids": ["31002"]}})
    knowledge.review_reference(actor, document["entry_id"], "v1", "approved")
    reference = knowledge.search_documents(context, "原文")[0]
    advice = knowledge.draft_advice(context, {"event": event}, [reference["reference_id"]])
    for name in ("confirm_event", "review_advice", "record_execution", "submit_feedback", "review_case"):
        payload = {"action_type": name, "operation_context": "rehearsal", "content": "合成操作记录"}
        if name in ("review_advice", "record_execution"):
            payload.update(advice_id=advice["advice_id"], advice_version=1)
        if name == "review_advice":
            payload["review_decision"] = "approved"
        result = cases.record_action(actor, context, "event", payload)
    snapshot = cases.export_reviewed_case(actor, result["case"]["case_id"])
    yield store, actor, knowledge, cases, snapshot
    store.close()


@pytest.mark.parametrize("face_ids", [["3411"], ["31002", "3411"]])
def test_experience_cannot_publish_original_case_snapshot_to_another_face(reviewed, face_ids):
    store, actor, knowledge, _, snapshot = reviewed
    with pytest.raises(DomainError) as caught:
        knowledge.submit_experience(actor, snapshot, {"content": "合成经验", "applicability": {"face_ids": face_ids}})
    assert caught.value.status_code == 403
    assert store.list("experience") == []


@pytest.mark.parametrize("case_id", [None, "", ["case"], {"case_id": "case"}])
def test_export_reviewed_case_rejects_nonstring_or_empty_ids_as_request_errors(reviewed, case_id):
    _, actor, _, cases, _ = reviewed
    with pytest.raises(DomainError) as caught:
        cases.export_reviewed_case(actor, case_id)
    assert caught.value.status_code == 400


def test_same_face_reviewed_experience_remains_available_after_approval(reviewed):
    _, actor, knowledge, _, snapshot = reviewed
    experience = knowledge.submit_experience(actor, snapshot,
        {"content": "合成同工作面经验", "applicability": {"face_ids": ["31002"]}})
    assert experience["state"] == "pending_review"
    knowledge.review_experience(actor, experience["entry_id"], 1, "approved")
    reader = Actor(user_id="reader", roles=("ventilation",), face_ids=("31002",))
    result = knowledge.get_reference(reader, experience["entry_id"], 1)
    assert result["source_case_snapshot"]["snapshot_ref"] == snapshot["snapshot_ref"]
