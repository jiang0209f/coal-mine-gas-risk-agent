from copy import deepcopy
from datetime import datetime, timezone

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.cases.service import CasesService
from app.services.knowledge.service import KnowledgeService


@pytest.fixture
def setup(tmp_path):
    store = Store("sqlite+pysqlite:///:memory:")
    actor = Actor(user_id="safety", roles=("safety_management",), face_ids=("31002",))
    store.put("source", "source-1", {"source_id": "source-1", "face_id": "31002"})
    store.put("point", "point-1", {"point_id": "point-1", "source_id": "source-1", "face_id": "31002"})
    context = ExecutionContext(
        session_id="session-1", mode="historical_replay", source_id="source-1",
        as_of=datetime(2022, 4, 1, tzinfo=timezone.utc), actor=actor,
    )
    event = {
        "event_id": "event-1", "session_id": "session-1", "point_id": "point-1",
        "source_id": "source-1",
        "event_revision": 1, "revision_as_of": "2022-04-01T00:00:00+00:00",
        "face_id": "31002",
    }
    store.put("event", "event-1", event)
    knowledge = KnowledgeService(store, allowed_roots=[tmp_path])
    cases = CasesService(store, knowledge=knowledge)
    yield store, actor, context, event, knowledge, cases
    store.close()


def import_document(knowledge, actor, **overrides):
    spec = {
        "title": "通风规程", "source": "矿方作业规程", "version": "1",
        "applicability": {"face_ids": ["31002"]},
        "fragments": [{"text": "核查通风与甲烷监测记录。", "locator": {"page": 76}}],
    }
    spec.update(overrides)
    return knowledge.import_reference(actor, spec)


def approved_fragment(setup):
    _, actor, context, _, knowledge, _ = setup
    doc = import_document(knowledge, actor)
    knowledge.review_reference(actor, doc["document_id"], "1", "approved")
    return knowledge.search_documents(context, "通风")[0]


def draft(setup):
    _, _, context, event, knowledge, _ = setup
    ref = approved_fragment(setup)
    return knowledge.draft_advice(context, {"event": event}, [ref["reference_id"]])


def action(setup, action_type, **kwargs):
    _, actor, context, _, _, cases = setup
    return cases.record_action(actor, context, "event-1", {
        "action_type": action_type, "operation_context": "rehearsal",
        "content": "人工核实记录", **kwargs,
    })


def completed_case(setup):
    advice = draft(setup)
    action(setup, "confirm_event")
    action(setup, "review_advice", advice_id=advice["advice_id"], advice_version=1,
           review_decision="approved")
    action(setup, "record_execution", advice_id=advice["advice_id"], advice_version=1)
    action(setup, "submit_feedback")
    return action(setup, "review_case")


def test_search_only_approved_scope_and_preserves_page_source(setup):
    _, actor, context, _, knowledge, _ = setup
    first = import_document(knowledge, actor)
    assert knowledge.search_documents(context, "通风") == []
    knowledge.review_reference(actor, first["document_id"], "1", "approved")
    ref = knowledge.search_documents(context, "通风")[0]
    assert ref["source"] == "矿方作业规程"
    assert ref["locator"] == {"page": 76}
    assert ref["text"] == "核查通风与甲烷监测记录。"
    second = import_document(knowledge, actor, title="退回规程")
    knowledge.review_reference(actor, second["document_id"], "1", "rejected")
    assert len(knowledge.search_documents(context, "通风")) == 1


def test_document_access_rejects_unassigned_workface(setup):
    _, actor, _, _, knowledge, _ = setup
    with pytest.raises(DomainError) as caught:
        import_document(knowledge, actor, applicability={"face_ids": ["3411"]})
    assert caught.value.status_code == 403


def test_reference_import_from_allowlisted_text_and_blocks_other_roots(setup, tmp_path):
    _, actor, _, _, knowledge, _ = setup
    path = tmp_path / "regulation.txt"
    path.write_text("第一条：核查通风记录。", encoding="utf-8")
    doc = import_document(knowledge, actor, fragments=None, file_ref=str(path))
    assert doc["fragment_count"] == 1
    with pytest.raises(DomainError):
        import_document(knowledge, actor, fragments=None, file_ref="C:/outside.txt")


@pytest.mark.parametrize("method", ["review_reference", "review_experience"])
def test_reference_and_experience_review_require_safety_role(setup, method):
    _, _, _, _, knowledge, _ = setup
    actor = Actor(user_id="ventilation", roles=("ventilation",), face_ids=("31002",))
    with pytest.raises(DomainError) as caught:
        getattr(knowledge, method)(actor, "unknown", "1", "approved")
    assert caught.value.status_code == 403


@pytest.mark.parametrize("reference_ids", [[], ["invented-reference"]])
def test_advice_cannot_invent_or_omit_references(setup, reference_ids):
    _, _, context, event, knowledge, _ = setup
    with pytest.raises(DomainError):
        knowledge.draft_advice(context, {"event": event}, reference_ids)


def test_pending_reference_cannot_be_used_for_advice(setup):
    store, actor, context, event, knowledge, _ = setup
    import_document(knowledge, actor)
    ref = store.list("reference_fragment")[0]
    with pytest.raises(DomainError):
        knowledge.draft_advice(context, {"event": event}, [ref["reference_id"]])


def test_advice_versions_are_immutable_and_do_not_store_review_facts(setup):
    store, actor, context, event, knowledge, _ = setup
    first = draft(setup)
    original = deepcopy(first)
    first["items"][0]["text"] = "调用方改写"
    second = knowledge.draft_advice(context, {"event": event}, original["reference_ids"])
    assert second["advice_id"] == original["advice_id"]
    assert second["version"] == 2
    assert knowledge.get_advice(actor, original["advice_id"], 1) == original
    assert "review_status" not in original
    assert "state" not in store.get("advice", f"{original['advice_id']}:1")


def test_case_query_is_read_only_and_lists_pending_confirmation(setup):
    store, _, context, _, _, cases = setup
    result = cases.get_case(context, "event-1")
    assert result["actions"] == []
    assert "confirm_event" in result["pending_items"]
    assert store.list("case") == []


def test_case_actions_record_real_operation_clock_and_logical_data_clock(setup):
    store, _, context, _, knowledge, _ = setup
    actual = datetime(2026, 11, 3, 15, tzinfo=timezone.utc)
    cases = CasesService(store, knowledge=knowledge, clock=lambda: actual)
    result = cases.record_action(context.actor, context, "event-1", {
        "action_type": "confirm_event", "operation_context": "rehearsal", "content": "已确认",
        "actor_id": "forged", "occurred_at": "forged",
    })
    record = result["case_action"]
    assert record["actor_id"] == "safety"
    assert datetime.fromisoformat(record["occurred_at"]) == actual
    assert datetime.fromisoformat(record["data_as_of"]) == context.as_of


@pytest.mark.parametrize("action_type", ["confirm_event", "review_advice", "record_execution", "submit_feedback", "review_case"])
def test_system_actor_cannot_submit_case_actions(setup, action_type):
    _, _, context, _, _, cases = setup
    actor = Actor(user_id="system", roles=("safety_management",), face_ids=("31002",), is_system=True)
    system_context = context.model_copy(update={"actor": actor})
    with pytest.raises(DomainError) as caught:
        cases.record_action(actor, system_context, "event-1", {
            "action_type": action_type, "operation_context": "rehearsal", "content": "记录",
        })
    assert caught.value.status_code == 403


@pytest.mark.parametrize("action_type", ["review_advice", "record_execution", "submit_feedback", "review_case"])
def test_case_flow_rejects_actions_before_event_confirmation(setup, action_type):
    with pytest.raises(DomainError) as caught:
        action(setup, action_type)
    assert caught.value.status_code == 409


def test_execution_requires_exact_approved_advice_version(setup):
    _, _, context, event, knowledge, _ = setup
    advice = draft(setup)
    action(setup, "confirm_event")
    with pytest.raises(DomainError):
        action(setup, "record_execution", advice_id=advice["advice_id"], advice_version=1)
    action(setup, "review_advice", advice_id=advice["advice_id"], advice_version=1, review_decision="approved")
    second = knowledge.draft_advice(context, {"event": event}, advice["reference_ids"])
    with pytest.raises(DomainError):
        action(setup, "record_execution", advice_id=second["advice_id"], advice_version=2)
    assert action(setup, "record_execution", advice_id=advice["advice_id"], advice_version=1)["case"]["phase"] == "executed"


def test_rejected_advice_cannot_be_executed(setup):
    advice = draft(setup)
    action(setup, "confirm_event")
    action(setup, "review_advice", advice_id=advice["advice_id"], advice_version=1, review_decision="rejected")
    with pytest.raises(DomainError):
        action(setup, "record_execution", advice_id=advice["advice_id"], advice_version=1)


def test_case_pending_items_reflect_rejection_and_new_advice_version(setup):
    _, _, context, event, knowledge, cases = setup
    advice = draft(setup)
    action(setup, "confirm_event")
    action(setup, "review_advice", advice_id=advice["advice_id"], advice_version=1, review_decision="rejected")
    assert "record_execution" not in cases.get_case(context, "event-1")["pending_items"]
    knowledge.draft_advice(context, {"event": event}, advice["reference_ids"])
    assert "review_advice" in cases.get_case(context, "event-1")["pending_items"]


def test_reviewed_case_exports_immutable_snapshot_and_experience_inherits_context(setup):
    _, actor, context, _, knowledge, cases = setup
    result = completed_case(setup)
    case_id = result["case"]["case_id"]
    snapshot = cases.export_reviewed_case(actor, case_id)
    experience = knowledge.submit_experience(actor, snapshot, {
        "content": "复核后的演练经验", "applicability": {"face_ids": ["31002"]},
        "operation_context": "field_record",
    })
    assert experience["operation_context"] == "rehearsal"
    assert experience["source_case_snapshot_ref"] == snapshot["snapshot_ref"]
    knowledge.review_experience(actor, experience["experience_id"], 1, "approved")
    rehearsal_refs = knowledge.search_documents(context, "演练经验")
    assert len(rehearsal_refs) == 1
    realtime = context.model_copy(update={"mode": "realtime"})
    assert knowledge.search_documents(realtime, "演练经验") == []
    with pytest.raises(DomainError):
        knowledge.draft_advice(realtime, {"event": setup[3]}, [rehearsal_refs[0]["reference_id"]])
    snapshot["advice_refs"].clear()
    assert cases.export_reviewed_case(actor, case_id)["advice_refs"]
    with pytest.raises(DomainError):
        action(setup, "submit_feedback")


def test_unreviewed_case_cannot_be_exported_or_used_as_experience(setup):
    _, actor, _, _, knowledge, cases = setup
    created = action(setup, "confirm_event")
    with pytest.raises(DomainError):
        cases.export_reviewed_case(actor, created["case"]["case_id"])
    with pytest.raises(DomainError):
        knowledge.submit_experience(actor, {"case_id": "forged"}, {"content": "文本"})


def test_case_scope_and_session_are_checked(setup):
    _, _, context, _, _, cases = setup
    other = context.model_copy(update={"session_id": "other-session"})
    with pytest.raises(DomainError):
        cases.get_case(other, "event-1")
    actor = Actor(user_id="other", roles=("safety_management",), face_ids=("3411",))
    denied = context.model_copy(update={"actor": actor})
    with pytest.raises(DomainError) as caught:
        cases.get_case(denied, "event-1")
    assert caught.value.status_code == 403
