"""Local uploads, original locators and approved multi-term retrieval."""
from datetime import datetime, timezone
import hashlib
from io import BytesIO

import pytest
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.db import Store
from app.schemas.common import Actor, DomainError, ExecutionContext
from app.services.knowledge.service import KnowledgeService


@pytest.fixture
def parts(tmp_path):
    store = Store("sqlite+pysqlite:///:memory:")
    manager = Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",))
    reader = Actor(user_id="reader", roles=("ventilation",), face_ids=("31002",))
    store.put("source", "synthetic-local", {"source_id": "synthetic-local", "face_id": "31002"})
    context = ExecutionContext(session_id="session", source_id="synthetic-local", mode="historical_replay",
                               as_of=datetime(2022, 2, 5, tzinfo=timezone.utc), actor=reader)
    yield store, manager, reader, context, tmp_path
    store.close()


def spec(**values):
    return {"title": "合成资料", "source": "合成离线测试", "version": "v1",
            "applicability": {"face_ids": ["31002"]}, **values}


def test_upload_import_retains_hash_original_line_and_requires_review(parts):
    store, manager, reader, context, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    content = "第一条：核对瓦斯记录。\n\n第三行：核对通风记录。".encode("utf-8")
    uploaded = knowledge.upload_file(manager, "合成资料.txt", content, ["31002"])
    assert uploaded["sha256"] == hashlib.sha256(content).hexdigest()
    assert uploaded["size_bytes"] == len(content)
    assert "path" not in uploaded and "stored_path" not in uploaded
    document = knowledge.import_reference(manager, spec(file_id=uploaded["file_id"]))
    assert document["state"] == "pending_review"
    assert document["file_id"] == uploaded["file_id"]
    assert document["file_sha256"] == uploaded["sha256"]
    assert knowledge.search_documents(context, "瓦斯") == []
    detailed = knowledge.get_reference(reader, document["entry_id"], "v1")
    assert [fragment["locator"] for fragment in detailed["fragments"]] == [{"line": 1}, {"line": 3}]
    knowledge.review_reference(manager, document["entry_id"], "v1", "approved", "已核对合成内容")
    found = knowledge.search_documents(context, "瓦斯")
    assert found[0]["file_id"] == uploaded["file_id"]
    assert found[0]["title"] == "合成资料"
    metadata, original = knowledge.get_file_content(reader, uploaded["file_id"])
    assert metadata["filename"] == "合成资料.txt" and original == content


@pytest.mark.parametrize("filename,content", [
    ("../secret.txt", b"text"), (r"..\secret.txt", b"text"), ("C:secret.txt", b"text"),
    ("script.exe", b"text"), ("fake.pdf", b"not a pdf"), ("fake.docx", b"not a document"),
    ("empty.txt", b""), ("binary.txt", b"\xff\xfe\x80"), ("bad\r\n.txt", b"synthetic"),
])
def test_upload_rejects_paths_unsupported_corrupt_or_empty_content(parts, filename, content):
    store, manager, _, _, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    with pytest.raises(DomainError):
        knowledge.upload_file(manager, filename, content)
    assert store.list("knowledge_file") == []


def test_upload_limit_and_manager_permission_are_enforced(parts):
    store, manager, reader, _, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    with pytest.raises(DomainError) as caught:
        knowledge.upload_file(reader, "test.txt", b"synthetic")
    assert caught.value.status_code == 403
    with pytest.raises(DomainError) as caught:
        knowledge.upload_file(manager, "large.txt", b"a" * (20 * 1024 * 1024 + 1))
    assert caught.value.status_code == 413


def test_file_id_is_immutable_and_bound_to_uploader_and_faces(parts):
    store, manager, _, _, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    first = knowledge.upload_file(manager, "same.txt", b"first")
    second = knowledge.upload_file(manager, "same.txt", b"second")
    assert first["file_id"] != second["file_id"]
    assert knowledge.get_file_content(manager, first["file_id"])[1] == b"first"
    other = Actor(user_id="other", roles=("safety_management",), face_ids=("31002",))
    with pytest.raises(DomainError) as caught:
        knowledge.import_reference(other, spec(file_id=first["file_id"]))
    assert caught.value.status_code == 403
    foreign = Actor(user_id="foreign", roles=("safety_management",), face_ids=("3411",))
    with pytest.raises(DomainError) as caught:
        knowledge.get_file_content(foreign, first["file_id"])
    assert caught.value.status_code == 403
    with pytest.raises(DomainError):
        knowledge.import_reference(manager, spec(file_id="unknown"))


def test_file_id_cannot_be_combined_with_inline_text_or_path(parts):
    store, manager, _, _, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    uploaded = knowledge.upload_file(manager, "source.txt", b"original")
    for values in ({"text": "different"}, {"fragments": [{"text": "different", "locator": {"line": 1}}]},
                   {"file_ref": str(root / "outside.txt")}):
        with pytest.raises(DomainError):
            knowledge.import_reference(manager, spec(file_id=uploaded["file_id"], **values))
    assert store.list("reference") == []


def test_mutated_upload_does_not_become_a_trusted_reference(parts):
    store, manager, _, _, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    uploaded = knowledge.upload_file(manager, "source.txt", b"original")
    next((root / "uploads").iterdir()).write_bytes(b"changed")
    with pytest.raises(DomainError) as caught:
        knowledge.import_reference(manager, spec(file_id=uploaded["file_id"]))
    assert caught.value.code == "file_integrity_failed"
    assert store.list("reference") == []


def test_docx_paragraph_and_long_fragment_offsets_preserve_original(parts):
    store, manager, reader, _, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    docx = Document()
    docx.add_paragraph("第一段合成条文。")
    docx.add_paragraph("")
    long_text = "第三段瓦斯通风核对记录。" * 180
    docx.add_paragraph(long_text)
    output = BytesIO()
    docx.save(output)
    uploaded = knowledge.upload_file(manager, "source.docx", output.getvalue())
    document = knowledge.import_reference(manager, spec(file_id=uploaded["file_id"]))
    fragments = knowledge.get_reference(reader, document["entry_id"], "v1")["fragments"]
    assert fragments[0]["locator"] == {"paragraph": 1}
    chunks = [item for item in fragments if item["locator"].get("paragraph") == 3]
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 1200
        locator = chunk["locator"]
        assert long_text[locator["start_offset"]:locator["end_offset"]] == chunk["text"]


def pdf_bytes():
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (Synthetic methane ventilation record.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_empty_pages_do_not_renumber_original_page_locator(parts):
    store, manager, reader, _, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    uploaded = knowledge.upload_file(manager, "source.pdf", pdf_bytes())
    document = knowledge.import_reference(manager, spec(file_id=uploaded["file_id"]))
    fragments = knowledge.get_reference(reader, document["entry_id"], "v1")["fragments"]
    assert len(fragments) == 1 and fragments[0]["locator"] == {"page": 2}
    assert "Synthetic methane" in fragments[0]["text"]


def test_docx_table_rows_are_searchable_and_keep_table_row_locator(parts):
    store, manager, reader, context, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    document = Document()
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "合成参数"
    table.cell(0, 1).text = "数值"
    table.cell(1, 0).text = "瓦斯参数"
    table.cell(1, 1).text = "合成值"
    output = BytesIO()
    document.save(output)
    uploaded = knowledge.upload_file(manager, "table.docx", output.getvalue())
    entry = knowledge.import_reference(manager, spec(file_id=uploaded["file_id"]))
    knowledge.review_reference(manager, entry["entry_id"], "v1", "approved")
    found = knowledge.search_documents(context, "瓦斯参数")
    assert len(found) == 1
    assert found[0]["text"] == "瓦斯参数\t合成值"
    assert found[0]["locator"] == {"table": 1, "row": 2}


def test_multi_keyword_search_ranks_full_matches_and_preserves_approved_scope(parts):
    store, manager, reader, context, _ = parts
    knowledge = KnowledgeService(store)
    for title, text, state in [("部分命中", "核对瓦斯记录。", "approved"),
                               ("完整命中", "核对瓦斯记录，并对照通风记录。", "approved"),
                               ("待审完整", "瓦斯与通风原文。", None)]:
        document = knowledge.import_reference(manager, spec(title=title, text=text, locator={"page": 6}))
        if state:
            knowledge.review_reference(manager, document["entry_id"], "v1", state)
    results = knowledge.search_documents(context, "瓦斯 通风")
    assert [item["title"] for item in results] == ["完整命中", "部分命中"]
    assert results[0]["matched_terms"] == ["瓦斯", "通风"]
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["version"] == "v1" and results[0]["locator"] == {"page": 6}
    assert knowledge.search_documents(context, "未命中词") == []


def test_search_matches_title_and_source_without_changing_stored_fragment(parts):
    store, manager, _, context, _ = parts
    knowledge = KnowledgeService(store)
    document = knowledge.import_reference(manager, spec(title="瓦斯监测规程", text="核对记录。", locator={"paragraph": 1}))
    knowledge.review_reference(manager, document["entry_id"], "v1", "approved")
    found = knowledge.search_documents(context, "瓦斯 合成")
    assert len(found) == 1
    assert found[0]["matched_terms"] == ["瓦斯", "合成"]
    assert "score" not in store.get("reference_fragment", found[0]["reference_id"])


@pytest.mark.parametrize("period", [
    {"valid_from": "2027-02-01T00:00:00+08:00"},
    {"valid_to": "2026-11-04T00:00:00+00:00"},
])
def test_future_or_expired_material_is_excluded_from_search_and_direct_advice(parts, period):
    store, manager, reader, context, _ = parts
    knowledge = KnowledgeService(store, clock=lambda: datetime(2026, 11, 4, 12, tzinfo=timezone.utc))
    document = knowledge.import_reference(manager, spec(text="核对合成瓦斯记录。", locator={"page": 1},
        applicability={"face_ids": ["31002"], **period}))
    with pytest.raises(DomainError) as review_error:
        knowledge.review_reference(manager, document["entry_id"], "v1", "approved")
    assert review_error.value.code == "reference_not_effective"
    fragment = knowledge.get_reference(reader, document["entry_id"], "v1")["fragments"][0]

    assert knowledge.search_documents(context, "瓦斯") == []
    event = {"event_id": "event", "event_revision": 1, "session_id": context.session_id,
             "source_id": context.source_id, "face_id": "31002", "revision_as_of": context.as_of.isoformat()}
    with pytest.raises(DomainError) as caught:
        knowledge.draft_advice(context, {"event": event}, [fragment["reference_id"]])
    assert caught.value.code == "invalid_request"


@pytest.mark.parametrize("period", [
    {"valid_from": "not-a-date"}, {"valid_from": "2022-01-01T00:00:00"},
    {"valid_from": "2026-01-01T00:00:00Z", "valid_to": "2022-01-01T00:00:00Z"},
])
def test_effective_period_requires_timezone_and_increasing_bounds(parts, period):
    store, manager, _, _, _ = parts
    knowledge = KnowledgeService(store)
    with pytest.raises(DomainError):
        knowledge.import_reference(manager, spec(text="合成原文", locator={"paragraph": 1},
            applicability={"face_ids": ["31002"], **period}))
    assert store.list("reference") == []


def test_advice_items_locate_the_exact_uploaded_document_version(parts):
    store, manager, _, context, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    uploaded = knowledge.upload_file(manager, "source.txt", "核对合成记录。".encode())
    entry = knowledge.import_reference(manager, spec(file_id=uploaded["file_id"]))
    knowledge.review_reference(manager, entry["entry_id"], "v1", "approved")
    reference = knowledge.search_documents(context, "记录")[0]
    event = {"event_id": "event", "event_revision": 1, "session_id": context.session_id,
             "source_id": context.source_id, "face_id": "31002", "revision_as_of": context.as_of.isoformat()}
    advice = knowledge.draft_advice(context, {"event": event}, [reference["reference_id"]])
    item = advice["items"][0]
    assert item["entry_id"] == entry["entry_id"]
    assert item["version"] == "v1" and item["kind"] == "document"
    assert item["file_id"] == uploaded["file_id"]
    assert item["locator"] == {"line": 1}


def test_file_import_scope_cannot_expand_beyond_uploaded_faces(parts):
    store, _, _, _, root = parts
    manager = Actor(user_id="broad-manager", roles=("safety_management",), face_ids=("31002", "3411"))
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    uploaded = knowledge.upload_file(manager, "scope.txt", b"synthetic", ["31002"])
    with pytest.raises(DomainError) as caught:
        knowledge.import_reference(manager, spec(file_id=uploaded["file_id"], applicability={"face_ids": ["3411"]}))
    assert caught.value.status_code == 403


def test_effective_period_includes_start_and_unbounded_legacy_remains_searchable(parts):
    store, manager, _, context, _ = parts
    boundary = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    knowledge = KnowledgeService(store, clock=lambda: boundary)
    for applicability in ({"face_ids": ["31002"], "valid_from": boundary.isoformat()}, {"face_ids": ["31002"]}):
        entry = knowledge.import_reference(manager, spec(text="有效合成原文", applicability=applicability, locator={"line": 1}))
        knowledge.review_reference(manager, entry["entry_id"], "v1", "approved")
    assert len(knowledge.search_documents(context, "有效")) == 2


@pytest.mark.parametrize("file_id", [["invented"], {"file_id": "invented"}])
def test_malformed_file_ids_are_request_errors(parts, file_id):
    store, manager, _, _, root = parts
    knowledge = KnowledgeService(store, upload_dir=root / "uploads")
    with pytest.raises(DomainError):
        knowledge.import_reference(manager, spec(file_id=file_id))
