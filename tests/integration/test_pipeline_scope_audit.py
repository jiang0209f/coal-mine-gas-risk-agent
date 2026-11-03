from datetime import datetime, timedelta, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext, utcstr
from app.services.cases.service import CasesService
from app.services.knowledge.service import KnowledgeService

START = datetime(2026, 10, 1, tzinfo=timezone.utc)


@pytest.fixture
def setup():
    store = Store("sqlite://")
    actor = Actor(user_id="u", roles=("safety_management",), face_ids=("31002",))
    context = ExecutionContext(session_id="session", mode="historical_replay", source_id="source-1", as_of=START, actor=actor)
    for id in ("source-1", "source-2"):
        store.put("source", id, {"source_id": id, "face_id": "31002"})
    store.put("point", "point", {"point_id": "point", "source_id": "source-1", "face_id": "31002"})
    event = dict(event_id="event", session_id="session", source_id="source-1", face_id="31002", point_id="point", event_revision=1, revision_as_of=utcstr(START))
    store.put("event", "event", event)
    knowledge = KnowledgeService(store)
    document = knowledge.import_reference(actor, {"title": "Synthetic", "source": "Test", "version": "1", "applicability": {"face_ids": ["31002"]}, "fragments": [{"text": "核查监测记录。", "locator": {"page": 1}}]})
    knowledge.review_reference(actor, document["document_id"], "1", "approved")
    reference = knowledge.search_documents(context, "核查")[0]["reference_id"]
    yield store, actor, context, event, knowledge, CasesService(store, knowledge=knowledge), reference
    store.close()


@pytest.mark.parametrize("change", [{"source_id": "source-2"}, {"session_id": "other-session"}, {"revision_as_of": utcstr(START + timedelta(seconds=1))}])
def test_advice_rejects_other_source_session_or_future_revision(setup, change):
    _, _, context, event, knowledge, _, reference = setup
    with pytest.raises(DomainError) as caught:
        knowledge.draft_advice(context, {"event_snapshot": {**event, **change}}, [reference])
    assert caught.value.status_code == (409 if "revision_as_of" in change else 403)


@pytest.mark.parametrize("change", [{"source_id": "source-2"}, {"source_id": None}, {"session_id": "other-session"}, {"revision_as_of": utcstr(START + timedelta(seconds=1))}])
def test_cases_fallback_rejects_other_source_session_or_future_revision(setup, change):
    store, _, context, event, _, cases, _ = setup
    store.put("event", "event", {**event, **change})
    with pytest.raises(DomainError) as caught:
        cases.get_case(context, "event")
    assert caught.value.status_code == (409 if "revision_as_of" in change else 403)


def test_case_write_requires_same_actor_as_execution_context(setup):
    _, _, context, _, _, cases, _ = setup
    another_actor = Actor(user_id="another", roles=("safety_management",), face_ids=("31002",))
    with pytest.raises(DomainError) as caught:
        cases.record_action(another_actor, context, "event", {"action_type": "confirm_event", "operation_context": "rehearsal", "content": "核查。"})
    assert caught.value.status_code == 403


def test_system_actor_cannot_draft_human_advice(setup):
    _, actor, context, event, knowledge, _, reference = setup
    system_context = context.model_copy(update={"actor": actor.model_copy(update={"is_system": True})})
    with pytest.raises(DomainError) as caught:
        knowledge.draft_advice(system_context, {"event_snapshot": event}, [reference])
    assert caught.value.status_code == 403
