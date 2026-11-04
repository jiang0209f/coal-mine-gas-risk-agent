"""Authenticated HTTP workflow with an isolated, entirely synthetic corpus and event."""

import hashlib
import json

from fastapi.testclient import TestClient
import pytest

from app.config import Settings
from app.main import create_app


FACE = "QA-KNOWLEDGE-CLOSURE"
START = "2022-02-05T13:11:00+00:00"
END = "2022-02-05T13:12:00+00:00"
PASSWORD = "isolated-workflow-password-12345"
SYNTHETIC_TEXT = "合成闭环依据：核对当前事件记录，仅用于软件验收。\n合成检索词：瓦斯监测记录。\n".encode()


@pytest.fixture
def workflow_api(tmp_path, monkeypatch):
    import os
    for name in tuple(os.environ):
        if name.upper().startswith("DEEPSEEK_") or name.upper() == "EVENT_BRIEF_ALLOWED_SOURCE_IDS":
            monkeypatch.delenv(name, raising=False)
    application = create_app(Settings(
        _env_file=None, runtime_dir=tmp_path, database_url="sqlite+pysqlite:///:memory:",
        deepseek_api_key="isolated-test-key-never-sent", event_brief_allowed_source_ids=[],
    ))
    with TestClient(application) as client:
        assert application.dependency_overrides == {}
        services = application.state.services
        cloud_attempts = []

        def forbidden_adapter(settings):
            cloud_attempts.append("adapter_construction")
            raise AssertionError("The local workflow must not construct a cloud adapter")

        services["agent"].adapter_factory = forbidden_adapter
        headers = {}
        for username, role in (("manager", "safety_management"), ("ventilation", "ventilation"),
                               ("gas", "gas_governance")):
            services["auth"].create_user(username, PASSWORD, [role], [FACE])
            login = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
            assert login.status_code == 200, login.text
            headers[username] = {"Authorization": "Bearer " + login.json()["access_token"]}
            assert client.get("/api/auth/me", headers=headers[username]).json()["roles"] == [role]

        audit = []

        def request(method, path, role="manager", expected=200, **kwargs):
            response = client.request(method, path, headers=headers[role], **kwargs)
            audit.append({"method": method, "path": path, "role": role, "status": response.status_code})
            assert response.status_code == expected, response.text
            return response

        assert client.get("/api/references").status_code == 401
        request("POST", "/api/sources", json={"source_id": "synthetic-workflow", "name": "隔离合成闭环",
                                                "face_id": FACE, "timezone": "+00:00"})
        request("POST", "/api/points", json={"point_id": "qa-workflow-t0", "source_id": "synthetic-workflow",
            "source_tag": "SYNTHETIC-T0", "name": "合成T0", "quantity": "methane", "unit": "%"})
        observation_file = request("POST", "/api/files", files={"file": (
            "synthetic-observations.csv", b"STime,SValue\n2022-02-05T13:11:00+00:00,0.1\n2022-02-05T13:12:00+00:00,0.9\n", "text/csv",
        )}).json()
        imported = request("POST", "/api/history-imports", json={"source_id": "synthetic-workflow",
            "file_id": observation_file["file_id"], "mapping": {"time_column": "STime",
                "value_columns": {"SValue": "qa-workflow-t0"}, "timezone": "+00:00", "lineage": "raw"}}).json()
        assert imported["imported_rows"] == 2
        session = request("POST", "/api/sessions", json={"source_id": "synthetic-workflow",
            "mode": "historical_replay", "start_as_of": START}).json()
        yield client, application, request, session, cloud_attempts, audit
        assert cloud_attempts == []


def upload_document(request):
    uploaded = request("POST", "/api/references/files", files={"file": (
        "synthetic-workflow.txt", SYNTHETIC_TEXT, "text/plain",
    )}, data={"face_ids": json.dumps([FACE])}).json()
    assert uploaded["sha256"] == hashlib.sha256(SYNTHETIC_TEXT).hexdigest()
    original = request("GET", f"/api/references/files/{uploaded['file_id']}/content")
    assert original.content == SYNTHETIC_TEXT
    document = request("POST", "/api/references/imports", json={"file_id": uploaded["file_id"],
        "title": "合成闭环验收资料", "source": "隔离软件测试文本", "version": "qa-v1",
        "applicability": {"face_ids": [FACE]}}).json()
    assert document["state"] == "pending_review" and document["fragment_count"] == 2
    assert document["file_sha256"] == uploaded["sha256"]
    return document


def generate_event(request, session):
    tick = request("POST", f"/api/sessions/{session['session_id']}/tick", json={"target_as_of": END}).json()
    events = request("GET", "/api/events", params={"session_id": session["session_id"]}).json()
    observed = [event for event in events if event["event_type"] == "observed_exceedance"]
    assert len(observed) == 1 and observed[0]["alert_level"] == "red"
    assert tick["events"] == events and tick["briefs"]
    return observed[0]


def test_authenticated_knowledge_case_and_experience_workflow(workflow_api, tmp_path):
    client, application, request, session, cloud_attempts, audit = workflow_api
    sid = session["session_id"]
    document = upload_document(request)
    search = lambda query: request("POST", "/api/references/search", role="ventilation",
                                    json={"session_id": sid, "query": query}).json()
    assert search("合成闭环依据") == []
    detail = request("GET", f"/api/references/{document['entry_id']}", params={"version": "qa-v1"}).json()
    assert detail["fragments"][0]["locator"]["line"] == 1
    request("POST", f"/api/references/{document['entry_id']}/review",
            json={"version": "qa-v1", "decision": "approved", "comment": "核对合成夹具原文与定位"})
    references = search("合成闭环依据")
    assert len(references) == 1 and references[0]["version"] == "qa-v1"
    reference_id = references[0]["reference_id"]
    assert references[0]["locator"]["line"] == 1
    event = generate_event(request, session)
    event_id = event["event_id"]
    assert event["source_id"] == "synthetic-workflow"

    def action(action_type, advice=None, role="ventilation", expected=200, **extra):
        body = {"session_id": sid, "action_type": action_type, "operation_context": "rehearsal",
                "content": "隔离合成处置记录"}
        if advice is not None:
            body.update(advice_id=advice["advice_id"], advice_version=advice["version"])
        body.update(extra)
        response = request("POST", f"/api/cases/{event_id}/actions", role=role, expected=expected, json=body)
        return response.json()

    action("confirm_event")
    first = request("POST", "/api/advice", role="ventilation", json={"session_id": sid,
        "event_id": event_id, "reference_ids": [reference_id]}).json()
    assert first["version"] == 1 and first["event_revision"] == event["event_revision"]
    action("review_advice", first, role="manager", review_decision="rejected", content="合成第一版退回测试")
    action("record_execution", first, expected=409)
    second = request("POST", "/api/advice", role="ventilation", json={"session_id": sid,
        "event_id": event_id, "reference_ids": [reference_id]}).json()
    assert second["advice_id"] == first["advice_id"] and second["version"] == 2
    action("review_advice", second, role="manager", review_decision="approved", content="合成第二版批准测试")
    action("record_execution", second)
    action("submit_feedback")
    before_review = request("GET", f"/api/cases/{event_id}", params={"session_id": sid}).json()
    assert before_review["case"]["phase"] == "feedback_submitted"
    assert before_review["reviewed_snapshot"] is None
    action("review_case")
    closed = request("GET", f"/api/cases/{event_id}", params={"session_id": sid}).json()
    snapshot = closed["reviewed_snapshot"]
    assert closed["case"]["phase"] == "reviewed" and closed["pending_items"] == []
    assert len(closed["actions"]) == 6
    assert snapshot["session_id"] == sid and snapshot["source_id"] == event["source_id"]
    assert snapshot["event_revision"] == event["event_revision"]
    assert snapshot["source_refs"] == [reference_id]
    assert len(snapshot["advice_refs"]) == 1 and snapshot["advice_refs"][0]["version"] == 2
    action("submit_feedback", expected=409)

    experience = request("POST", "/api/experiences", role="ventilation", json={"case_id": closed["case"]["case_id"],
        "spec": {"content": "合成闭环经验：保留复核快照和引用记录。", "applicability": {"face_ids": [FACE]}}}).json()
    assert experience["state"] == "pending_review" and experience["operation_context"] == "rehearsal"
    assert experience["source_case_snapshot_ref"] == snapshot["snapshot_ref"]
    assert search("合成闭环经验") == []
    request("POST", f"/api/experiences/{experience['experience_id']}/review", json={"version": 1,
        "decision": "approved", "comment": "核对合成复核快照"})
    found = search("合成闭环经验")
    assert len(found) == 1 and found[0]["kind"] == "experience" and found[0]["operation_context"] == "rehearsal"
    realtime = request("POST", "/api/sessions", json={"source_id": "synthetic-workflow",
        "mode": "realtime", "start_as_of": START}).json()
    field_search = request("POST", "/api/references/search", json={"session_id": realtime["session_id"],
        "query": "合成闭环经验"}).json()
    assert field_search == []
    task = request("POST", "/api/chat", role="ventilation", json={"session_id": sid, "message": "当前事件"}).json()
    assert task["method"] == "structured_template" and task["local_reason"] == "cloud_source_not_authorized"
    assert cloud_attempts == []
    assert application.dependency_overrides == {}
    verification = {"status": "passed", "data_scope": "fully_synthetic_isolated_database", "actual_authentication": True,
        "auth_dependency_overrides": False, "event_generated_by_runner": True, "case_phase": "reviewed",
        "action_count": 6, "advice_versions": 2, "executed_advice_version": 2,
        "pending_experience_hidden": True, "approved_experience_searchable": True,
        "rehearsal_experience_excluded_from_realtime": True, "cloud_adapter_constructions": 0,
        "request_count": len(audit), "requests": audit}
    (tmp_path / "workflow-verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.parametrize("role", ["ventilation", "gas"])
def test_non_safety_roles_cannot_review_reference_or_advice(workflow_api, role):
    _, _, request, session, cloud_attempts, _ = workflow_api
    document = upload_document(request)
    request("POST", f"/api/references/{document['entry_id']}/review", role=role, expected=403,
            json={"version": "qa-v1", "decision": "approved"})
    document_state = request("GET", f"/api/references/{document['entry_id']}", params={"version": "qa-v1"}).json()
    assert document_state["state"] == "pending_review"
    request("POST", f"/api/references/{document['entry_id']}/review",
            json={"version": "qa-v1", "decision": "approved"})
    event = generate_event(request, session)
    request("POST", f"/api/cases/{event['event_id']}/actions", role=role, json={"session_id": session["session_id"],
        "action_type": "confirm_event", "operation_context": "rehearsal", "content": "合成事件确认"})
    references = request("POST", "/api/references/search", role=role,
        json={"session_id": session["session_id"], "query": "合成闭环依据"}).json()
    advice = request("POST", "/api/advice", role=role, json={"session_id": session["session_id"],
        "event_id": event["event_id"], "reference_ids": [references[0]["reference_id"]]}).json()
    request("POST", f"/api/cases/{event['event_id']}/actions", role=role, expected=403,
        json={"session_id": session["session_id"], "action_type": "review_advice", "operation_context": "rehearsal",
              "advice_id": advice["advice_id"], "advice_version": advice["version"], "review_decision": "approved"})
    current = request("GET", f"/api/cases/{event['event_id']}", params={"session_id": session["session_id"]}).json()
    assert current["advice_review_status"] == {} and len(current["actions"]) == 1
    assert cloud_attempts == []
