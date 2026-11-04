"""Current knowledge validity and document revisions are separate from replay time."""
from datetime import datetime, timedelta, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.knowledge.service import KnowledgeService

NOW = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    store.put("source", "synthetic", {"source_id": "synthetic", "face_id": "31002"})
    context = ExecutionContext(session_id="session", source_id="synthetic", mode="historical_replay",
        as_of=datetime(2022, 2, 5, tzinfo=timezone.utc), actor=actor)
    clock = [NOW]
    knowledge = KnowledgeService(store, clock=lambda: clock[0])
    yield store, actor, context, knowledge, clock
    store.close()


def material(version="v1", **values):
    return {"title": "合成现行资料", "source": "合成测试", "version": version,
            "text": "合成瓦斯记录核对条文。", "locator": {"page": 1},
            "applicability": {"face_ids": ["31002"], "valid_from": "2026-02-01T00:00:00+08:00"}, **values}


def event(context):
    return {"event_id": "event", "event_revision": 1, "session_id": context.session_id,
            "source_id": context.source_id, "face_id": "31002", "revision_as_of": context.as_of.isoformat()}


def approve(parts, spec):
    _, actor, _, knowledge, _ = parts
    entry = knowledge.import_reference(actor, spec)
    return knowledge.review_reference(actor, entry["entry_id"], entry["version"], "approved", "核对当前效力")


def test_historical_replay_uses_current_knowledge_and_records_both_time_bases(parts):
    _, actor, context, knowledge, _ = parts
    entry = approve(parts, material())
    found = knowledge.search_documents(context, "瓦斯")
    assert len(found) == 1 and found[0]["entry_id"] == entry["entry_id"]
    assert datetime.fromisoformat(found[0]["knowledge_as_of"]) == NOW
    advice = knowledge.draft_advice(context, {"event": event(context)}, [found[0]["reference_id"]])
    assert datetime.fromisoformat(advice["knowledge_as_of"]) == NOW
    assert datetime.fromisoformat(advice["data_as_of"]) == context.as_of


def test_narrow_reviewer_cannot_supersede_wider_approved_document(parts):
    _, narrow_actor, _, knowledge, _ = parts
    wide_actor = Actor(user_id="wide-manager", roles=("safety_management",), face_ids=("31002", "3411"))
    first = knowledge.import_reference(wide_actor, material(applicability={"face_ids": ["31002", "3411"]}))
    knowledge.review_reference(wide_actor, first["entry_id"], "v1", "approved")
    knowledge.import_reference(wide_actor, material("v2", entry_id=first["entry_id"]))

    with pytest.raises(DomainError) as caught:
        knowledge.review_reference(narrow_actor, first["entry_id"], "v2", "approved")

    assert caught.value.status_code == 403
    assert knowledge.get_reference(wide_actor, first["entry_id"], "v1")["state"] == "approved"
    assert knowledge.get_reference(wide_actor, first["entry_id"], "v2")["state"] == "pending_review"


@pytest.mark.parametrize("period", [
    {"valid_from": "2026-11-05T00:00:00+00:00"},
    {"valid_to": "2026-11-04T12:00:00+00:00"},
])
def test_only_current_effective_material_can_be_approved(parts, period):
    _, actor, context, knowledge, _ = parts
    entry = knowledge.import_reference(actor, material(applicability={"face_ids": ["31002"], **period}))
    with pytest.raises(DomainError) as caught:
        knowledge.review_reference(actor, entry["entry_id"], "v1", "approved")
    assert caught.value.code == "reference_not_effective"
    assert knowledge.get_reference(actor, entry["entry_id"], "v1")["state"] == "pending_review"
    assert knowledge.search_documents(context, "瓦斯") == []


def test_expired_approved_material_is_blocked_in_search_and_direct_advice(parts):
    _, actor, context, knowledge, clock = parts
    expiry = NOW + timedelta(days=1)
    entry = approve(parts, material(applicability={"face_ids": ["31002"], "valid_to": expiry.isoformat()}))
    reference = knowledge.search_documents(context, "瓦斯")[0]
    clock[0] = expiry
    assert knowledge.search_documents(context, "瓦斯") == []
    with pytest.raises(DomainError) as caught:
        knowledge.draft_advice(context, {"event": event(context)}, [reference["reference_id"]])
    assert caught.value.code == "reference_not_effective"
    assert knowledge.get_reference(actor, entry["entry_id"], "v1")["fragments"]


def test_new_approved_revision_supersedes_old_but_old_citations_stay_readable(parts):
    _, actor, context, knowledge, _ = parts
    first = approve(parts, material("v1"))
    old_reference = knowledge.search_documents(context, "瓦斯")[0]
    pending = knowledge.import_reference(actor, material("v2", entry_id=first["entry_id"], text="新版合成瓦斯核对条文。"))
    assert pending["entry_id"] == first["entry_id"] and pending["state"] == "pending_review"
    assert knowledge.search_documents(context, "瓦斯")[0]["version"] == "v1"
    knowledge.review_reference(actor, pending["entry_id"], "v2", "approved")
    results = knowledge.search_documents(context, "瓦斯")
    assert len(results) == 1 and results[0]["version"] == "v2"
    old = knowledge.get_reference(actor, first["entry_id"], "v1")
    assert old["state"] == "superseded" and old["fragments"][0]["reference_id"] == old_reference["reference_id"]
    assert old["reviewer_id"] == actor.user_id
    assert knowledge.get_reference(actor, first["entry_id"])["version"] == "v2"
    with pytest.raises(DomainError):
        knowledge.draft_advice(context, {"event": event(context)}, [old_reference["reference_id"]])


def test_revision_version_is_unique_and_unknown_or_conflicting_ids_are_rejected(parts):
    _, actor, _, knowledge, _ = parts
    first = approve(parts, material())
    for values in ({"entry_id": first["entry_id"]}, {"entry_id": "unknown"},
                   {"entry_id": first["entry_id"], "document_id": "different"}):
        with pytest.raises(DomainError):
            knowledge.import_reference(actor, material(**values))


def test_older_pending_revision_cannot_override_a_newer_approved_one(parts):
    _, actor, context, knowledge, _ = parts
    first = approve(parts, material("v1"))
    knowledge.import_reference(actor, material("v2", entry_id=first["entry_id"]))
    knowledge.import_reference(actor, material("v3", document_id=first["entry_id"]))
    knowledge.review_reference(actor, first["entry_id"], "v3", "approved")
    with pytest.raises(DomainError) as caught:
        knowledge.review_reference(actor, first["entry_id"], "v2", "approved")
    assert caught.value.status_code == 409
    assert knowledge.search_documents(context, "瓦斯")[0]["version"] == "v3"


def test_future_revision_stays_pending_until_effective_without_hiding_current_version(parts):
    _, actor, context, knowledge, clock = parts
    first = approve(parts, material("v1"))
    effective = NOW + timedelta(days=1)
    knowledge.import_reference(actor, material("v2", entry_id=first["entry_id"],
        applicability={"face_ids": ["31002"], "valid_from": effective.isoformat()}))
    with pytest.raises(DomainError):
        knowledge.review_reference(actor, first["entry_id"], "v2", "approved")
    assert knowledge.search_documents(context, "瓦斯")[0]["version"] == "v1"
    clock[0] = effective
    knowledge.review_reference(actor, first["entry_id"], "v2", "approved")
    assert knowledge.search_documents(context, "瓦斯")[0]["version"] == "v2"


def test_advice_reference_validation_detects_superseded_material(parts):
    _, actor, context, knowledge, _ = parts
    first = approve(parts, material("v1"))
    reference = knowledge.search_documents(context, "瓦斯")[0]
    advice = knowledge.draft_advice(context, {"event": event(context)}, [reference["reference_id"]])
    assert knowledge.validate_advice_references(context, advice) is None
    knowledge.import_reference(actor, material("v2", entry_id=first["entry_id"]))
    knowledge.review_reference(actor, first["entry_id"], "v2", "approved")
    with pytest.raises(DomainError):
        knowledge.validate_advice_references(context, advice)


@pytest.mark.parametrize("reference_ids", [[], ["invented-reference"]])
def test_advice_reference_validation_rejects_missing_or_unknown_refs(parts, reference_ids):
    _, _, context, knowledge, _ = parts
    with pytest.raises(DomainError):
        knowledge.validate_advice_references(context, {"reference_ids": reference_ids})
