"""Offline closure checks for event revisions and frozen, traceable case records."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.cases.service import CasesService
from app.services.knowledge.service import KnowledgeService


@pytest.fixture
def closure():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="qa-reviewer", roles=("safety_management",), face_ids=("QA-CLOSURE",))
    moment = datetime(2022, 2, 5, 13, 12, tzinfo=timezone.utc)
    context = ExecutionContext(session_id="qa-session", source_id="qa-source", mode="historical_replay",
                               as_of=moment, actor=actor)
    store.put("source", "qa-source", dict(source_id="qa-source", face_id="QA-CLOSURE"))
    event = dict(event_id="qa-event", session_id=context.session_id, source_id=context.source_id,
                 face_id="QA-CLOSURE", event_revision=1, revision_as_of=moment.isoformat(),
                 event_type="observed_exceedance", alert_level="orange")
    store.put("event", "qa-event", event)
    knowledge = KnowledgeService(store)
    document = knowledge.import_reference(actor, dict(
        title="合成闭环资料", source="离线测试夹具", version="qa-v1",
        applicability={"face_ids": ["QA-CLOSURE"]},
        fragments=[{"text": "合成测试：核对事件记录。", "locator": {"paragraph": 1}}],
    ))
    knowledge.review_reference(actor, document["document_id"], "qa-v1", "approved")
    reference = knowledge.search_documents(context, "合成测试")[0]
    cases = CasesService(store, knowledge=knowledge)
    yield SimpleNamespace(store=store, actor=actor, context=context, event=event, knowledge=knowledge,
                          reference=reference, cases=cases)
    store.close()


def draft(parts, context=None):
    return parts.knowledge.draft_advice(context or parts.context, {"event": parts.event},
                                        [parts.reference["reference_id"]])


def act(parts, action_type, advice=None, **overrides):
    payload = dict(action_type=action_type, operation_context="rehearsal", content="合成人工记录")
    if advice is not None:
        payload.update(advice_id=advice["advice_id"], advice_version=advice["version"])
    payload.update(overrides)
    return parts.cases.record_action(parts.actor, parts.context, "qa-event", payload)


def advance_event(parts):
    parts.context = parts.context.model_copy(update={"as_of": parts.context.as_of + timedelta(minutes=1)})
    parts.event = {**parts.event, "event_revision": 2, "alert_level": "red",
                   "revision_as_of": parts.context.as_of.isoformat()}
    parts.store.put("event", "qa-event", parts.event)


@pytest.mark.parametrize("action_type", ["review_advice", "record_execution"])
def test_old_event_revision_advice_cannot_be_reviewed_or_executed(closure, action_type):
    advice = draft(closure)
    act(closure, "confirm_event")
    act(closure, "review_advice", advice, review_decision="approved")
    before = closure.cases.get_case(closure.context, "qa-event")["actions"]
    advance_event(closure)

    with pytest.raises(DomainError) as caught:
        act(closure, action_type, advice, review_decision="approved")

    assert caught.value.status_code == 409
    assert closure.cases.get_case(closure.context, "qa-event")["actions"] == before


def test_case_pending_items_require_a_new_draft_for_the_current_revision(closure):
    advice = draft(closure)
    act(closure, "confirm_event")
    act(closure, "review_advice", advice, review_decision="approved")
    advance_event(closure)

    result = closure.cases.get_case(closure.context, "qa-event")

    assert result["current_event_revision"] == 2
    assert result["pending_items"] == ["draft_advice"]
    assert result["advice_validity"][f"{advice['advice_id']}:1"] == {
        "event_revision": 1, "current_event_revision": 2, "stale": True,
    }
    current = draft(closure)
    result = closure.cases.get_case(closure.context, "qa-event")
    assert result["pending_items"] == ["review_advice"]
    assert result["advice_validity"][f"{current['advice_id']}:2"]["stale"] is False


def test_future_advice_cannot_be_reviewed_and_is_not_a_pending_option(closure):
    future = closure.context.model_copy(update={"as_of": closure.context.as_of + timedelta(minutes=1)})
    advice = draft(closure, future)
    act(closure, "confirm_event")

    with pytest.raises(DomainError) as caught:
        act(closure, "review_advice", advice, review_decision="approved")

    assert caught.value.status_code == 409
    result = closure.cases.get_case(closure.context, "qa-event")
    assert result["pending_items"] == ["draft_advice"]
    assert result["advice_validity"][f"{advice['advice_id']}:1"]["stale"] is True


def test_rejected_draft_then_current_version_completes_traceable_frozen_closure(closure):
    first = draft(closure)
    assert closure.cases.get_case(closure.context, "qa-event")["reviewed_snapshot"] is None
    act(closure, "confirm_event")
    act(closure, "review_advice", first, review_decision="rejected")
    second = draft(closure)
    act(closure, "review_advice", second, review_decision="approved")
    act(closure, "record_execution", second)
    act(closure, "submit_feedback")
    result = act(closure, "review_case")
    snapshot = closure.cases.export_reviewed_case(closure.actor, result["case"]["case_id"])

    assert snapshot["session_id"] == closure.context.session_id
    assert snapshot["source_id"] == closure.context.source_id
    assert snapshot["event_revision"] == 1
    assert snapshot["event_snapshot"] == closure.event
    assert snapshot["operation_context"] == "rehearsal"
    assert snapshot["source_refs"] == [closure.reference["reference_id"]]
    assert len(snapshot["advice_refs"]) == 1 and snapshot["advice_refs"][0]["version"] == 2
    assert all(action["event_revision"] == 1 and action["event_snapshot"] == closure.event
               for action in snapshot["actions"])
    assert closure.cases.get_case(closure.context, "qa-event")["reviewed_snapshot"] == snapshot
    original = deepcopy(snapshot)
    snapshot["event_snapshot"]["alert_level"] = "caller-mutation"
    snapshot["advice_refs"][0]["items"].clear()
    advance_event(closure)
    assert closure.cases.export_reviewed_case(closure.actor, result["case"]["case_id"]) == original
    assert closure.cases.get_case(closure.context, "qa-event")["reviewed_snapshot"] == original


def test_action_records_retain_each_event_revision_and_data_cutoff(closure):
    confirmation = act(closure, "confirm_event")["case_action"]
    advance_event(closure)
    advice = draft(closure)
    review = act(closure, "review_advice", advice, review_decision="approved")["case_action"]

    assert confirmation["event_revision"] == 1
    assert confirmation["event_snapshot"]["alert_level"] == "orange"
    assert review["event_revision"] == 2
    assert review["event_snapshot"]["alert_level"] == "red"
    assert datetime.fromisoformat(review["data_as_of"]) == closure.context.as_of
