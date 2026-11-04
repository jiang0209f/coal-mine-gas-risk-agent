"""Local material initialization preserves approval scope and is repeatable."""
from datetime import datetime, timezone
import hashlib
import json

import pytest

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.knowledge.service import KnowledgeService


@pytest.fixture
def setup(tmp_path):
    from app.scripts.import_knowledge_materials import import_materials
    store = Store("sqlite+pysqlite:///:memory:")
    knowledge = KnowledgeService(store, upload_dir=tmp_path / "uploads")
    actor = Actor(user_id="authorized-test-import", roles=("safety_management",), face_ids=("31002",))
    store.put("source", "source", {"source_id": "source", "face_id": "31002"})
    context = ExecutionContext(actor=actor, session_id="s", source_id="source", mode="historical_replay",
        as_of=datetime.now(timezone.utc))
    content = "甲烷监测资料：核对通风情况，保存监测记录。".encode("utf-8")
    (tmp_path / "confirmed.txt").write_bytes(content)
    material = {"material_id": "confirmed", "relative_path": "confirmed.txt", "sha256": hashlib.sha256(content).hexdigest(),
        "currency_status": "current_user_confirmed", "import_spec": {"title": "已确认资料", "source": "合成资料",
        "version": "1", "applicability": {"face_ids": ["31002"]}}}
    path = tmp_path / "materials.json"
    def save(rows):
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    save([material])
    yield import_materials, knowledge, actor, context, path, material, save
    store.close()


def test_import_defaults_pending_then_explicit_confirmed_approval_is_idempotent(setup):
    run, knowledge, actor, context, path, material, _ = setup
    first = run(knowledge, actor, path)
    assert first["imported"] == 1 and first["approved"] == 0
    assert knowledge.search_documents(context, "甲烷") == []
    second = run(knowledge, actor, path, approve_ids=["confirmed"], review_comment="用户确认现行版本")
    assert second["imported"] == 0 and second["reused"] == 1 and second["approved"] == 1
    assert knowledge.search_documents(context, "甲烷")[0]["text"]
    third = run(knowledge, actor, path, approve_ids=["confirmed"], review_comment="用户确认现行版本")
    assert third["approved"] == 0 and third["reused"] == 1
    assert len(knowledge.store.list("knowledge_file")) == len(knowledge.store.list("reference")) == 1


@pytest.mark.parametrize("status", ["unverified", "research_draft"])
def test_draft_or_unverified_material_is_never_approved(setup, status):
    run, knowledge, actor, _, path, material, save = setup
    material["currency_status"] = status
    save([material])
    with pytest.raises(DomainError):
        run(knowledge, actor, path, approve_ids=["confirmed"], review_comment="test")
    assert knowledge.store.list("reference") == knowledge.store.list("knowledge_file") == []


@pytest.mark.parametrize("problem", ["hash", "traversal", "unknown_approval", "future"])
def test_all_materials_are_validated_before_any_write(setup, problem):
    run, knowledge, actor, _, path, material, save = setup
    invalid = {**material, "material_id": "bad", "import_spec": {**material["import_spec"]}}
    approvals = []
    if problem == "hash":
        invalid["sha256"] = "0" * 64
    elif problem == "traversal":
        invalid["relative_path"] = "../outside.txt"
    elif problem == "unknown_approval":
        approvals = ["unknown"]
    else:
        invalid["import_spec"]["applicability"] = {"face_ids": ["31002"], "valid_from": "2999-01-01T00:00:00Z"}
        approvals = ["bad"]
    save([material, invalid])
    with pytest.raises(DomainError):
        run(knowledge, actor, path, approve_ids=approvals, review_comment="test")
    assert knowledge.store.list("reference") == knowledge.store.list("knowledge_file") == []


def test_reader_cannot_initialize_or_approve_materials(setup):
    run, knowledge, _, _, path, _, _ = setup
    reader = Actor(user_id="reader", roles=("ventilation",), face_ids=("31002",))
    with pytest.raises(DomainError) as failure:
        run(knowledge, reader, path)
    assert failure.value.status_code == 403
    assert knowledge.store.list("reference") == []
