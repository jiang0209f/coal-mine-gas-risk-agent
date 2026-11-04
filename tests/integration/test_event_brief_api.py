"""New alert briefs are scheduled after persistence and remain event-scoped."""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr
import pytest

from app.adapters.deepseek import ChatReply
from app.api.routes import actor as actor_dependency
from app.config import Settings
from app.db import Store
from app.main import build_services, create_app
from app.schemas.common import Actor, utcstr


START = datetime(2022, 2, 5, 8, tzinfo=timezone.utc)


class CapturedBriefs:
    """Scheduling spy using the real event snapshot authorization."""

    def __init__(self, services):
        self.services = services
        self.actual = services.get("event_briefs")
        self.submissions = []
        self.executed = []
        self.reads = []
        self.cancellations = []

    def _snapshot(self, context, event_id, event_revision):
        context.actor.require_role("ventilation", "gas_governance", "safety_management")
        return self.services["alerts"].get_event_snapshot(context, event_id, event_revision)

    def submit(self, context, event_id, event_revision=None, retry=False, schedule=None):
        snapshot = self._snapshot(context, event_id, event_revision)
        stored = self.services["store"].get("event_revision", f"{event_id}@{snapshot['event_revision']:08d}")
        assert stored == snapshot, "Event revision must be committed before scheduling"
        self.submissions.append(dict(context=context, event=deepcopy(snapshot), retry=retry))
        task_id = f"brief-{len(self.submissions)}"
        if schedule is not None:
            schedule(lambda: self.executed.append(task_id))
        return dict(event_id=event_id, event_revision=snapshot["event_revision"], session_id=context.session_id,
                    source_id=context.source_id, as_of=snapshot["revision_as_of"], status="queued", task_id=task_id,
                    task={"task_id": task_id, "state": "queued"}, can_retry=False, can_cancel=True, attempt=1)

    def get(self, context, event_id, event_revision=None):
        snapshot = self._snapshot(context, event_id, event_revision)
        self.reads.append((context, event_id, event_revision))
        return dict(event_id=event_id, event_revision=snapshot["event_revision"], session_id=context.session_id,
                    source_id=context.source_id, as_of=snapshot["revision_as_of"], status="not_started", task_id=None,
                    task=None, can_retry=False, can_cancel=False, attempt=0)

    def cancel(self, context, event_id, event_revision=None):
        snapshot = self._snapshot(context, event_id, event_revision)
        self.cancellations.append((event_id, snapshot["event_revision"]))
        return dict(event_id=event_id, event_revision=snapshot["event_revision"], session_id=context.session_id,
                    source_id=context.source_id, as_of=snapshot["revision_as_of"], status="cancelled", task_id="brief-1",
                    task={"task_id": "brief-1", "state": "cancelled"}, can_retry=True, can_cancel=False, attempt=1)


@pytest.fixture
def api(tmp_path, monkeypatch):
    import os
    for name in tuple(os.environ):
        if name.upper().startswith("DEEPSEEK_"):
            monkeypatch.delenv(name, raising=False)
    app = create_app(Settings(_env_file=None, runtime_dir=tmp_path, database_url="sqlite+pysqlite:///:memory:",
                              event_brief_allowed_source_ids=["source"]))
    actor = Actor(user_id="operator", roles=("ventilation",), face_ids=("31002",))
    app.dependency_overrides[actor_dependency] = lambda: actor
    with TestClient(app) as client:
        services = app.state.services
        data = services["data"]
        for source in ("source", "other-source"):
            data.register_source(source, source, "31002")
        data.register_point("t0", "source", "T0", "T0", "methane", "%")
        values = [.70, .70, .70, .70, .70, .70, .71, .90, .91, .10, .90, .10]
        path = tmp_path / "brief-observations.csv"
        path.write_text("STime,SValue\n" + "\n".join(
            f"{(START+timedelta(minutes=i)).isoformat()},{value}" for i, value in enumerate(values)), encoding="utf-8")
        file = data.register_file(path, Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",)))
        data.import_history(Actor(user_id="manager", roles=("safety_management",), face_ids=("31002",)),
                            "source", file["file_id"], {"time_column": "STime", "value_columns": {"SValue": "t0"}, "lineage": "raw"})
        session = services["sessions"].create_session(actor, "historical_replay", "source", START)
        capture = CapturedBriefs(services)
        services["event_briefs"] = capture
        yield client, services, actor, session, capture


def tick(client, session_id, minute):
    response = client.post(f"/api/sessions/{session_id}/tick", json={"target_as_of": (START+timedelta(minutes=minute)).isoformat()})
    assert response.status_code == 200, response.text
    return response.json()


def create_event(api, minute=5):
    client, services, actor, session, capture = api
    tick(client, session["session_id"], minute)
    context = services["sessions"].get_context(actor, session["session_id"])
    event = services["alerts"].query_events(context)[0]
    return event


def test_http_tick_schedules_only_new_events_and_grade_upgrades_after_commit(api):
    client, _, _, session, captured = api
    result = tick(client, session["session_id"], 5)
    assert len(result["briefs"]) == len(captured.submissions) == len(captured.executed) == 1
    first = captured.submissions[0]["event"]
    assert first["alert_level"] == "orange"
    repeated = tick(client, session["session_id"], 5)
    assert repeated["reused"] is True
    assert len(captured.submissions) == 1
    tick(client, session["session_id"], 6)
    assert len(captured.submissions) == 1, "Normal revisions do not create another paid brief"
    tick(client, session["session_id"], 7)
    assert len(captured.submissions) == 2
    upgraded = captured.submissions[-1]["event"]
    assert upgraded["event_id"] == first["event_id"] and upgraded["alert_level"] == "red"
    assert upgraded["event_revision"] > first["event_revision"]
    tick(client, session["session_id"], 8)
    tick(client, session["session_id"], 9)
    assert len(captured.submissions) == 2
    tick(client, session["session_id"], 10)
    assert len(captured.submissions) == 3
    assert captured.submissions[-1]["event"]["event_id"] != first["event_id"]


def test_direct_offline_runner_never_schedules_model_work(api):
    _, services, actor, session, captured = api
    result = services["runner"].run_once(actor, session["session_id"], START+timedelta(minutes=5))
    assert result["briefs"] == []
    assert result["events"] and captured.submissions == [] and captured.executed == []


def test_long_tick_retains_briefs_for_short_events_that_already_ended(api):
    client, _, _, session, captured = api
    result = tick(client, session["session_id"], 11)
    assert len(result["events"]) == len(captured.submissions) == 2
    assert all(item["event"]["event_state"] == "ended" for item in captured.submissions)
    assert all(item["task_id"] for item in result["briefs"])


def test_brief_get_is_read_only_and_post_retry_and_cancel_are_event_scoped(api):
    client, _, _, session, captured = api
    event = create_event(api)
    params = dict(session_id=session["session_id"], event_revision=event["event_revision"])
    count = len(captured.submissions)
    path = f"/api/events/{event['event_id']}/brief"
    response = client.get(path, params=params)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "not_started" and response.json()["task"] is None
    assert len(captured.submissions) == count
    response = client.post(path, json={**params, "retry": True})
    assert response.status_code == 200, response.text
    assert captured.submissions[-1]["retry"] is True and len(captured.executed) == count+1
    response = client.post(path+"/cancel", json=params)
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "cancelled"
    assert captured.cancellations == [(event["event_id"], event["event_revision"])]


@pytest.mark.parametrize("method", ["get", "post", "cancel"])
def test_brief_routes_reject_cross_source_session_and_invisible_revision(api, method):
    client, services, actor, session, _ = api
    event = create_event(api)
    other = services["sessions"].create_session(actor, "historical_replay", "other-source", START+timedelta(minutes=6))
    path = f"/api/events/{event['event_id']}/brief" + ("/cancel" if method == "cancel" else "")
    request = client.get if method == "get" else client.post
    key = "params" if method == "get" else "json"
    response = request(path, **{key: dict(session_id=other["session_id"], event_revision=event["event_revision"])})
    assert response.status_code in (403, 404)
    response = request(path, **{key: dict(session_id=session["session_id"], event_revision=999)})
    assert response.status_code == 404


@pytest.mark.parametrize("values", [{"event_revision": 0}, {"event_revision": -1}, {"extra": "untrusted"}])
def test_brief_mutation_validates_body_before_execution(api, values):
    client, _, _, session, captured = api
    event = create_event(api)
    count = len(captured.submissions)
    response = client.post(f"/api/events/{event['event_id']}/brief", json={"session_id": session["session_id"], **values})
    assert response.status_code == 422
    assert len(captured.submissions) == count


def test_brief_scope_keeps_role_and_face_authorization(api):
    client, services, _, session, _ = api
    event = create_event(api)
    path = f"/api/events/{event['event_id']}/brief"
    for actor in (Actor(user_id="outside", roles=("ventilation",), face_ids=("32001",)),
                  Actor(user_id="unknown", roles=("unknown",), face_ids=("31002",))):
        client.app.dependency_overrides[actor_dependency] = lambda actor=actor: actor
        assert client.get(path, params={"session_id": session["session_id"]}).status_code == 403


def test_submission_failure_preserves_alert_and_tick_result(api, monkeypatch):
    client, _, _, session, captured = api
    def unavailable(*args, **kwargs):
        raise RuntimeError("injected brief submission failure")
    monkeypatch.setattr(captured, "submit", unavailable)
    result = tick(client, session["session_id"], 5)
    assert len(result["events"]) == 1
    assert result["briefs"][0]["status"] == "failed"
    assert result["briefs"][0]["error_code"] == "brief_submission_failed"
    repeated = tick(client, session["session_id"], 5)
    assert repeated["reused"] is True
    assert repeated["events"] == result["events"]


@pytest.mark.parametrize("revision", [0, -1])
def test_brief_read_validates_requested_revision(api, revision):
    client, _, _, session, captured = api
    event = create_event(api)
    count = len(captured.reads)
    response = client.get(f"/api/events/{event['event_id']}/brief",
                          params={"session_id": session["session_id"], "event_revision": revision})
    assert response.status_code == 422
    assert len(captured.reads) == count


def test_real_core_missing_key_preserves_tools_and_shared_event_read_not_chat_tasks(api):
    client, services, actor, session, captured = api
    event = create_event(api)
    assert captured.actual is not None, "Application registers the real brief service"
    services["event_briefs"] = captured.actual
    path = f"/api/events/{event['event_id']}/brief"
    body = dict(session_id=session["session_id"], event_revision=event["event_revision"])
    submitted = client.post(path, json=body)
    assert submitted.status_code == 200, submitted.text
    fetched = client.get(path, params=body)
    assert fetched.status_code == 200, fetched.text
    brief = fetched.json()
    assert brief["status"] == "failed" and brief["task"]["error_code"] == "deepseek_not_configured"
    assert brief["task"]["tool_steps"][0]["name"] == "get_event_evidence"
    assert brief["task"]["tool_steps"][0]["status"] == "success"
    assert brief["task"]["citations"] and brief["can_retry"]
    assert "messages" not in brief["task"] and "context" not in brief["task"]
    reader = actor.model_copy(update={"user_id": "same-face-reader", "roles": ("gas_governance",)})
    client.app.dependency_overrides[actor_dependency] = lambda: reader
    shared = client.get(path, params=body)
    assert shared.status_code == 200 and shared.json()["task_id"] == brief["task_id"]
    assert client.get(f"/api/tasks/{brief['task_id']}").status_code == 403


def test_real_core_http_background_finishes_without_duplicate_model_requests(api):
    client, services, _, session, captured = api
    services["event_briefs"] = captured.actual
    services["agent"].settings.deepseek_api_key = SecretStr("offline-injected")
    calls = []
    class Adapter:
        def chat(self, messages, **kwargs):
            calls.append((deepcopy(messages), kwargs))
            return ChatReply({"role": "assistant", "content": "T0预警事件已记录，按规则核对监测依据。[E1]"},
                             "stop", "offline-model", {"total_tokens": 10})
        def close(self):
            pass
    services["agent"].adapter_factory = lambda settings: Adapter()
    result = tick(client, session["session_id"], 5)
    event = result["events"][0]
    assert result["briefs"][0]["status"] == "queued"
    path = f"/api/events/{event['event_id']}/brief"
    body = dict(session_id=session["session_id"], event_revision=event["event_revision"])
    finished = client.get(path, params=body).json()
    assert finished["status"] == "completed" and finished["task"]["model"] == "offline-model"
    assert len(calls) == 1 and finished["task"]["citations"]
    repeated = client.post(path, json=body)
    assert repeated.json()["task_id"] == finished["task_id"] and len(calls) == 1
    ordinary_revision = tick(client, session["session_id"], 6)["events"][0]["event_revision"]
    previous = client.get(path, params={**body, "event_revision": ordinary_revision}).json()
    assert previous["stale"] is True and previous["task_id"] == finished["task_id"] and len(calls) == 1
    assert previous["event_revision"] == event["event_revision"]
    assert previous["requested_event_revision"] == ordinary_revision
    upgraded = tick(client, session["session_id"], 7)
    assert len(upgraded["briefs"]) == 1 and len(calls) == 2
    assert upgraded["briefs"][0]["task_id"] != finished["task_id"]


def test_real_core_cancelled_queue_never_starts_adapter(api):
    client, services, actor, session, captured = api
    event = create_event(api)
    services["event_briefs"] = captured.actual
    queued = []
    context = services["sessions"].get_context(actor, session["session_id"])
    brief = captured.actual.submit(context, event["event_id"], event["event_revision"], schedule=queued.append)
    assert brief["status"] == "queued"
    path = f"/api/events/{event['event_id']}/brief"
    body = dict(session_id=session["session_id"], event_revision=event["event_revision"])
    cancelled = client.post(path+"/cancel", json=body)
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    assert len(queued) == 1
    queued[0]()
    assert client.get(path, params=body).json()["status"] == "cancelled"


def test_unapproved_source_preserves_local_evidence_and_never_constructs_adapter(api):
    client, services, _, session, captured = api
    event = create_event(api)
    services["event_briefs"] = captured.actual
    settings = services["agent"].settings
    settings.deepseek_api_key = SecretStr("offline-injected")
    settings.event_brief_allowed_source_ids = []
    constructions = []
    services["agent"].adapter_factory = lambda configuration: constructions.append(configuration)
    path = f"/api/events/{event['event_id']}/brief"
    body = dict(session_id=session["session_id"], event_revision=event["event_revision"])
    submitted = client.post(path, json=body)
    assert submitted.status_code == 200
    brief = client.get(path, params=body).json()
    assert brief["status"] == "failed" and brief["task"]["error_code"] == "cloud_source_not_authorized"
    assert constructions == []
    assert brief["task"]["result_cards"] and brief["task"]["citations"]


def test_service_startup_marks_interrupted_chats_and_briefs_and_preserves_evidence(tmp_path):
    settings = Settings(_env_file=None, runtime_dir=tmp_path,
                        database_url=f"sqlite+pysqlite:///{(tmp_path/'restart.db').as_posix()}", deepseek_api_key=None)
    store = Store(settings.database_url)
    calls = [{"call_id": "evidence", "name": "get_event_evidence", "status": "success"}]
    citations = [{"citation_id": "E1", "kind": "event", "event_id": "event", "event_revision": 1}]
    for task_id, state in (("queued-brief", "queued"), ("running-brief", "running"), ("completed-brief", "completed")):
        store.put("task", task_id, dict(task_id=task_id, task_type="event_brief", state=state,
                  tool_calls=deepcopy(calls), citations=deepcopy(citations), evidence_refs=[], result_cards=[], final_text=""))
    ordinary = dict(task_id="chat-task", state="queued", tool_calls=[], final_text="", messages=[])
    store.put("task", "chat-task", ordinary)
    store.close()
    services = build_services(settings)
    try:
        for task_id in ("queued-brief", "running-brief"):
            task = services["store"].get("task", task_id)
            assert task["state"] == "failed" and task["error_code"] == "task_interrupted"
            assert task["tool_calls"] == calls and task["citations"] == citations
        assert services["store"].get("task", "completed-brief")["state"] == "completed"
        chat = services["store"].get("task", "chat-task")
        assert chat["state"] == "failed" and chat["error_code"] == "task_interrupted"
        assert chat["tool_calls"] == ordinary["tool_calls"] and chat["messages"] == ordinary["messages"]
    finally:
        services["store"].close()


def test_concurrent_ticks_claim_one_auto_brief_per_event_grade(api, monkeypatch):
    client, services, actor, session, captured = api
    barrier = Barrier(2)
    original_query = services["alerts"].query_events
    def forecast(context, point_id):
        cutoff = START
        return dict(run_id=str(uuid4()), session_id=context.session_id, source_id=context.source_id,
            point_id=point_id, as_of=utcstr(context.as_of), cutoff=utcstr(cutoff), status="completed",
            aggregation={"seconds": 300}, outputs=[dict(prediction_stage="residual_corrected",
            quantity_semantics="physical_forecast", unit="%", interval_values=[dict(
                start_at=utcstr(cutoff+timedelta(minutes=5*i)), end_at=utcstr(cutoff+timedelta(minutes=5*(i+1))),
                value=.9) for i in range(12)])])
    monkeypatch.setattr(services["prediction"], "run_prediction", forecast)
    jobs = []
    def run_tick():
        barrier.wait(timeout=10)
        return services["runner"].run_once(actor, session["session_id"], START+timedelta(minutes=1), schedule=jobs.append)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a, future_b = pool.submit(run_tick), pool.submit(run_tick)
        results = [future_a.result(timeout=15), future_b.result(timeout=15)]
    assert len(captured.submissions) == len(jobs) == 1, "Same event and grade must claim one automatic model task"
    assert results[0]["events"][0]["event_id"] == results[1]["events"][0]["event_id"]
    assert len({result["events"][0]["event_revision"] for result in results}) == 1
    assert sum(result.get("reused",False) for result in results) == 1
    assert results[0]["briefs"][0]["task_id"] == results[1]["briefs"][0]["task_id"]
    current = original_query(services["sessions"].get_context(actor, session["session_id"]))[0]
    response = client.post(f"/api/events/{current['event_id']}/brief", json={
        "session_id": session["session_id"], "event_revision": current["event_revision"]})
    assert response.status_code == 200 and len(captured.submissions) == 2, "Explicit revision generation remains available"
