"""Approved advice must still cite current knowledge at review and execution time."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.cases.service import CasesService
from app.services.knowledge.service import KnowledgeService


@pytest.fixture
def parts():
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="qa-reviewer", roles=("safety_management",), face_ids=("QA-CURRENCY",))
    now = [datetime(2026, 11, 4, 12, tzinfo=timezone.utc)]
    context = ExecutionContext(session_id="qa-session", source_id="qa-source", mode="historical_replay",
        as_of=datetime(2022, 2, 5, 13, 12, tzinfo=timezone.utc), actor=actor)
    store.put("source", "qa-source", {"source_id": "qa-source", "face_id": "QA-CURRENCY"})
    event = dict(event_id="qa-event", session_id=context.session_id, source_id=context.source_id,
        event_revision=1, revision_as_of=context.as_of.isoformat(), face_id="QA-CURRENCY")
    store.put("event", "qa-event", event)
    knowledge = KnowledgeService(store, clock=lambda: now[0])
    cases = CasesService(store, knowledge=knowledge, clock=lambda: now[0])
    yield SimpleNamespace(store=store, actor=actor, now=now, context=context, event=event,
                          knowledge=knowledge, cases=cases)
    store.close()


def document(parts, version="v1", entry_id=None, valid_to=None):
    spec = dict(title="合成现行资料", source="隔离版本测试", version=version,
                applicability={"face_ids": ["QA-CURRENCY"]}, text=f"合成版本{version}：核对记录。", locator={"page": 1})
    if entry_id is not None:
        spec["entry_id"] = entry_id
    if valid_to is not None:
        spec["applicability"]["valid_to"] = valid_to.isoformat()
    result = parts.knowledge.import_reference(parts.actor, spec)
    parts.knowledge.review_reference(parts.actor, result["entry_id"], version, "approved")
    return result


def advice(parts, entry):
    reference = parts.knowledge.get_reference(parts.actor, entry["entry_id"], entry["version"])["fragments"][0]
    return parts.knowledge.draft_advice(parts.context, {"event": parts.event}, [reference["reference_id"]])


def act(parts, action_type, draft=None):
    payload = dict(action_type=action_type, operation_context="rehearsal", content="合成版本核验记录")
    if draft:
        payload.update(advice_id=draft["advice_id"], advice_version=draft["version"], review_decision="approved")
    return parts.cases.record_action(parts.actor, parts.context, "qa-event", payload)


def approved_advice(parts, *, valid_to=None):
    entry = document(parts, valid_to=valid_to)
    draft = advice(parts, entry)
    act(parts, "confirm_event")
    act(parts, "review_advice", draft)
    return entry, draft


@pytest.mark.parametrize("action_type", ["review_advice", "record_execution"])
def test_superseded_reference_invalidates_previously_approved_advice(parts, action_type):
    old, draft = approved_advice(parts)
    before = deepcopy(parts.cases.get_case(parts.context, "qa-event")["actions"])
    document(parts, version="v2", entry_id=old["entry_id"])

    with pytest.raises(DomainError) as caught:
        act(parts, action_type, draft)

    assert caught.value.status_code == 409
    result = parts.cases.get_case(parts.context, "qa-event")
    assert result["actions"] == before
    assert result["pending_items"] == ["draft_advice"]
    assert result["advice_validity"][f"{draft['advice_id']}:1"]["stale"] is True


@pytest.mark.parametrize("action_type", ["review_advice", "record_execution"])
def test_expired_reference_invalidates_advice_using_current_knowledge_clock(parts, action_type):
    expiry = parts.now[0] + timedelta(hours=1)
    _, draft = approved_advice(parts, valid_to=expiry)
    parts.now[0] = expiry

    with pytest.raises(DomainError) as caught:
        act(parts, action_type, draft)

    assert caught.value.status_code == 409
    result = parts.cases.get_case(parts.context, "qa-event")
    assert result["advice_validity"][f"{draft['advice_id']}:1"]["stale"] is True
    assert "record_execution" not in result["pending_items"]


def test_current_reference_new_advice_can_be_reviewed_and_executed(parts):
    old, previous = approved_advice(parts)
    newest = document(parts, version="v2", entry_id=old["entry_id"])
    current = advice(parts, newest)
    assert current["version"] == 2 and current["data_as_of"] == previous["data_as_of"]
    assert datetime.fromisoformat(current["knowledge_as_of"]) == parts.now[0]
    act(parts, "review_advice", current)
    result = act(parts, "record_execution", current)

    assert result["case"]["phase"] == "executed"
    assert result["case_action"]["advice_version"] == 2
    assert current["items"][0]["version"] == "v2"
    validity = parts.cases.get_case(parts.context, "qa-event")["advice_validity"]
    assert validity[f"{current['advice_id']}:2"]["stale"] is False
    assert validity[f"{previous['advice_id']}:1"]["stale"] is True


def test_superseding_knowledge_preserves_closed_case_actions_and_snapshot(parts):
    old, draft = approved_advice(parts)
    act(parts, "record_execution", draft)
    act(parts, "submit_feedback")
    closed = act(parts, "review_case")
    original = parts.cases.export_reviewed_case(parts.actor, closed["case"]["case_id"])
    document(parts, version="v2", entry_id=old["entry_id"])

    archived = parts.cases.get_case(parts.context, "qa-event")

    assert archived["case"]["phase"] == "reviewed" and archived["pending_items"] == []
    assert archived["reviewed_snapshot"] == original
    assert archived["actions"] == original["actions"]
    assert original["advice_refs"][0]["items"][0]["version"] == "v1"
    assert archived["advice_validity"][f"{draft['advice_id']}:1"]["stale"] is True
