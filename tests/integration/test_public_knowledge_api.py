"""Authenticated public explanations never inherit a mining session payload."""
from copy import deepcopy
from datetime import datetime, timezone
import json

from fastapi.testclient import TestClient
import pytest

from app.adapters.deepseek import ChatReply
from app.api.routes import actor as actor_dependency
from app.config import Settings
from app.main import create_app
from app.schemas.common import Actor


@pytest.fixture
def api(tmp_path):
    application = create_app(Settings(_env_file=None, runtime_dir=tmp_path,
        database_url="sqlite+pysqlite:///:memory:", deepseek_api_key="offline-synthetic-key",
        event_brief_allowed_source_ids=[], ingestion_worker_enabled=False, realtime_monitor_enabled=False))
    current = {"actor": Actor(user_id="PRIVATE-USER", roles=("safety_management",), face_ids=("PRIVATE-FACE",))}
    application.dependency_overrides[actor_dependency] = lambda: current["actor"]
    with TestClient(application) as client:
        services = application.state.services
        services["data"].register_source("PRIVATE-SOURCE", "PRIVATE-MINE", "PRIVATE-FACE")
        session = services["sessions"].create_session(current["actor"], "historical_replay", "PRIVATE-SOURCE", datetime(2022, 1, 1, tzinfo=timezone.utc))
        doc = services["knowledge"].import_reference(current["actor"], {
            "title": "合成公开标准", "source": "https://www.mem.gov.cn/test.pdf", "version": "v1",
            "applicability": {"face_ids": ["PRIVATE-FACE"]}, "text": "甲烷传感器调校应记录。", "locator": {"page": 1}})
        services["knowledge"].review_reference(current["actor"], doc["entry_id"], "v1", "approved")
        yield client, services, current, session, doc


def test_public_disclosure_and_answer_roundtrip_with_captured_http_inputs(api):
    client, services, _, session, doc = api
    public = client.post(f"/api/references/{doc['entry_id']}/disclosure", json={"version": "v1", "disclosure": "public_normative", "official_url": "https://www.mem.gov.cn/test.pdf", "category": "standard", "comment": "合成公开正文核验"})
    assert public.status_code == 200, public.text
    calls = []
    class Adapter:
        def chat(self, messages, **kwargs):
            calls.append({"messages": deepcopy(messages), **kwargs})
            return ChatReply(message={"role": "assistant", "content": json.dumps({"answer": "调校应有记录。[PUB-001]", "reference_ids": ["PUB-001"], "conflicts": []})}, finish_reason="stop", model="synthetic", usage={})
        def close(self):
            pass
    services["public_knowledge"].adapter_factory = lambda settings: Adapter()
    query = "PRIVATE-MEASUREMENT=1.37% 探头校准"
    response = client.post("/api/references/public-answer", json={"session_id": session["session_id"], "query": query})
    assert response.status_code == 200, response.text
    assert response.json()["state"] == "completed"
    assert "PRIVATE-" not in json.dumps(calls, ensure_ascii=False)
    assert response.json()["references"][0]["locator"]["page"] == 1
    assert calls[0]["tools"] == []


def test_default_local_document_is_not_sent_even_if_api_key_exists(api):
    client, services, _, session, _ = api
    assert "public_knowledge" in services
    services["public_knowledge"].adapter_factory = lambda settings: pytest.fail("private content must not reach the provider")
    response = client.post("/api/references/public-answer", json={"session_id": session["session_id"], "query": "甲烷传感器调校"})
    assert response.status_code == 200 and response.json()["state"] == "not_ready"


def test_public_endpoint_cannot_override_source_or_read_another_session(api):
    client, _, current, session, _ = api
    response = client.post("/api/references/public-answer", json={"session_id": session["session_id"], "query": "甲烷", "event_id": "private-event"})
    assert response.status_code == 422
    current["actor"] = Actor(user_id="other", roles=("ventilation",), face_ids=("OTHER",))
    response = client.post("/api/references/public-answer", json={"session_id": session["session_id"], "query": "甲烷"})
    assert response.status_code == 403


def test_disclosure_management_is_not_available_to_readers(api):
    client, _, current, _, doc = api
    current["actor"] = Actor(user_id="reader", roles=("ventilation",), face_ids=("PRIVATE-FACE",))
    response = client.post(f"/api/references/{doc['entry_id']}/disclosure", json={"version": "v1", "disclosure": "public_normative", "official_url": "https://www.mem.gov.cn/test.pdf", "category": "standard", "comment": "无权变更"})
    assert response.status_code == 403
