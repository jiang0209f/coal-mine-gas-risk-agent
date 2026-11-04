"""Authenticated local file upload, import, review, search and original access."""
import hashlib
import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest

from app.api.routes import actor as actor_dependency
from app.config import Settings
from app.main import create_app
from app.schemas.common import Actor


@pytest.fixture
def api(tmp_path):
    application = create_app(Settings(_env_file=None, runtime_dir=tmp_path,
                                      database_url="sqlite+pysqlite:///:memory:", deepseek_api_key=None,
                                      event_brief_allowed_source_ids=[]))
    current = {"actor": Actor(user_id="qa-manager", roles=("safety_management",), face_ids=("31002",))}
    application.dependency_overrides[actor_dependency] = lambda: current["actor"]
    with TestClient(application) as client:
        services = application.state.services
        services["data"].register_source("synthetic-knowledge-api", "合成知识验证", "31002")
        session = services["sessions"].create_session(current["actor"], "historical_replay",
            "synthetic-knowledge-api", datetime(2022, 2, 5, tzinfo=timezone.utc))
        yield client, services, current, session, tmp_path


def test_uploaded_file_import_review_search_and_original_roundtrip(api):
    client, _, _, session, root = api
    content = "纯合成条文：核对监测记录。\n纯合成条文：登记复核记录。".encode()
    response = client.post("/api/references/files", files={"file": ("合成资料.txt", content, "text/plain")},
                           data={"face_ids": json.dumps(["31002"])})
    assert response.status_code == 200, response.text
    uploaded = response.json()
    assert uploaded["filename"] == "合成资料.txt"
    assert uploaded["sha256"] == hashlib.sha256(content).hexdigest()
    assert uploaded["size_bytes"] == len(content)
    assert not any(key in uploaded for key in ("path", "file_ref", "storage_path"))
    assert any(path.is_file() for path in (root / "knowledge-files").rglob("*"))
    spec = dict(file_id=uploaded["file_id"], title="合成审核资料", source="离线API夹具",
                version="qa-v1", applicability={"face_ids": ["31002"]})
    imported = client.post("/api/references/imports", json={"spec": spec})
    assert imported.status_code == 200, imported.text
    document = imported.json()
    assert document["state"] == "pending_review"
    body = {"session_id": session["session_id"], "query": "监测 记录"}
    assert client.post("/api/references/search", json=body).json() == []
    reviewed = client.post(f"/api/references/{document['entry_id']}/review",
                           json={"version": "qa-v1", "decision": "approved", "comment": "合成测试审核"})
    assert reviewed.status_code == 200, reviewed.text
    results = client.post("/api/references/search", json=body)
    assert results.status_code == 200 and results.json()
    hit = results.json()[0]
    assert hit["entry_id"] == document["entry_id"] and hit["version"] == "qa-v1"
    assert hit["locator"] and hit["source"] == "离线API夹具"
    original = client.get(f"/api/references/files/{uploaded['file_id']}/content")
    assert original.status_code == 200 and original.content == content
    assert original.headers["x-content-type-options"] == "nosniff"
    assert "filename*=" in original.headers["content-disposition"]
    assert client.get(f"/api/references/files/{uploaded['file_id']}").json()["sha256"] == uploaded["sha256"]


def test_upload_requires_management_and_valid_authorized_faces(api):
    client, _, current, _, _ = api
    files = {"file": ("synthetic.txt", b"synthetic only", "text/plain")}
    current["actor"] = Actor(user_id="qa-reader", roles=("ventilation",), face_ids=("31002",))
    assert client.post("/api/references/files", files=files).status_code == 403
    current["actor"] = Actor(user_id="qa-manager", roles=("safety_management",), face_ids=("31002",))
    for faces in ('not-json', '{"face":"31002"}', '[]'):
        response = client.post("/api/references/files", files=files, data={"face_ids": faces})
        assert response.status_code == 400, response.text
    assert client.post("/api/references/files", files=files, data={"face_ids": '["other-face"]'}).status_code == 403


def test_file_access_and_import_cannot_cross_working_faces(api):
    client, _, current, _, _ = api
    response = client.post("/api/references/files", files={"file": ("synthetic.txt", b"synthetic", "text/plain")},
                           data={"face_ids": '["31002"]'})
    assert response.status_code == 200, response.text
    file_id = response.json()["file_id"]
    current["actor"] = Actor(user_id="qa-other", roles=("safety_management",), face_ids=("other-face",))
    assert client.get(f"/api/references/files/{file_id}").status_code in (403, 404)
    assert client.get(f"/api/references/files/{file_id}/content").status_code in (403, 404)
    result = client.post("/api/references/imports", json={"file_id": file_id, "title": "other",
        "source": "synthetic", "version": "1", "applicability": {"face_ids": ["other-face"]}})
    assert result.status_code in (403, 404)


def test_file_endpoints_require_existing_login(tmp_path):
    app = create_app(Settings(_env_file=None, runtime_dir=tmp_path, database_url="sqlite+pysqlite:///:memory:"))
    with TestClient(app) as client:
        assert client.post("/api/references/files", files={"file": ("a.txt", b"synthetic", "text/plain")}).status_code == 401
        assert client.get("/api/references/files/unknown").status_code == 401
        assert client.get("/api/references/files/unknown/content").status_code == 401
