"""Disk restart recovery ends ordinary chat work without replaying side effects."""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.db import Store
from app.schemas.common import Actor, ExecutionContext
from app.services.agent.service import AgentService


def service(store):
    def forbidden(*args, **kwargs):
        raise AssertionError("Recovery must not invoke a model or side-effect tool")
    return AgentService(store, {
        "alerts": SimpleNamespace(query_events=lambda **kwargs: []),
        "knowledge": SimpleNamespace(draft_advice=forbidden),
    }, Settings(_env_file=None, deepseek_api_key=None, event_brief_allowed_source_ids=[]),
        adapter_factory=forbidden)


def context(store):
    store.put("source", "source", {"source_id": "source", "face_id": "31002"})
    return ExecutionContext(session_id="session", mode="historical_replay", source_id="source",
        as_of=datetime.now(timezone.utc),
        actor=Actor(user_id="owner", roles=("safety_management",), face_ids=("31002",)))


@pytest.mark.parametrize("state", ["queued", "running"])
def test_disk_restart_marks_ordinary_task_interrupted_and_preserves_finished_results(tmp_path, state):
    url = "sqlite+pysqlite:///" + (tmp_path / "chat.db").as_posix()
    store = Store(url)
    ctx = context(store)
    current = service(store)
    task = current.submit(ctx, "查看监测数据", schedule=lambda work: None)
    raw = store.get("task", task["task_id"])
    finished = {"call_id": "done", "name": "query_events", "status": "success", "elapsed_ms": 12}
    card = {"tool": "query_events", "status": "success", "data": []}
    raw.update(state=state, tool_calls=[finished, {"call_id": "unfinished", "name": "draft_advice", "status": "started"}],
               result_cards=[card], evidence_refs=[{"kind": "event", "event_id": "event"}])
    current._save(raw)
    store.close()

    restarted_store = Store(url)
    restarted = service(restarted_store)
    try:
        assert restarted.recover_interrupted() == 1
        recovered = restarted.get_task(ctx.actor, task["task_id"])
        assert recovered["state"] == "failed" and recovered["error_code"] == "task_interrupted"
        assert recovered["final_text"] and recovered["answer"] == recovered["final_text"]
        assert recovered["tool_calls"][0] == finished
        assert recovered["tool_calls"][1]["status"] == "error"
        assert recovered["tool_calls"][1]["error_code"] == "task_interrupted"
        assert recovered["tool_calls"][1]["finished_at"]
        assert recovered["tool_steps"] == recovered["tool_calls"]
        assert recovered["metrics"] == {"tool_call_count": 2, "tool_call_success_rate": .5}
        assert recovered["result_cards"] == [card]
        assert recovered["evidence_refs"] == raw["evidence_refs"]
        assert restarted.recover_interrupted() == 0
    finally:
        restarted_store.close()


def test_recovery_leaves_terminal_awaiting_input_and_brief_tasks_unchanged(tmp_path):
    store = Store("sqlite+pysqlite:///" + (tmp_path / "chat.db").as_posix())
    current = service(store)
    ctx = context(store)
    snapshots = {}
    try:
        for state in ("completed", "failed", "cancelled", "partial", "awaiting_input"):
            task = current.submit(ctx, state, schedule=lambda work: None)
            raw = store.get("task", task["task_id"])
            raw["state"] = state
            store.put("task", task["task_id"], raw)
            snapshots[task["task_id"]] = deepcopy(raw)
        for state in ("queued", "running"):
            task = current.submit(ctx, state, schedule=lambda work: None)
            raw = store.get("task", task["task_id"])
            raw.update(task_type="event_brief", state=state)
            store.put("task", task["task_id"], raw)
            snapshots[task["task_id"]] = deepcopy(raw)
        assert current.recover_interrupted() == 0
        assert all(store.get("task", task_id) == original for task_id, original in snapshots.items())
    finally:
        store.close()


def test_recovery_closes_unanswered_tool_messages_without_reexecuting_them(tmp_path):
    store = Store("sqlite+pysqlite:///" + (tmp_path / "chat.db").as_posix())
    current = service(store)
    ctx = context(store)
    task = current.submit(ctx, "查询后形成建议", schedule=lambda work: None)
    raw = store.get("task", task["task_id"])
    raw["messages"].extend([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "done", "type": "function", "function": {"name": "query_events", "arguments": "{}"}},
            {"id": "unfinished", "type": "function", "function": {"name": "draft_advice", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "done", "content": '{"status":"success","data":[]}'},
    ])
    raw["tool_calls"] = [{"call_id": "done", "name": "query_events", "status": "success"}]
    current._save(raw)
    try:
        assert current.recover_interrupted() == 1
        recovered = store.get("task", task["task_id"])
        assert recovered["messages"][:-1] == raw["messages"]
        last = recovered["messages"][-1]
        assert last["role"] == "tool" and last["tool_call_id"] == "unfinished"
        assert '"task_interrupted"' in last["content"]
        assert recovered["tool_calls"][-1]["call_id"] == "unfinished"
        assert recovered["tool_calls"][-1]["status"] == "error"
        assert recovered["tool_calls"][-1]["error_code"] == "task_interrupted"
        retried = current.chat(ctx, {"task_id": task["task_id"], "answer": "重新查询事件"})
        assert retried["state"] == "completed"
        assert retried["tool_calls"][-1]["name"] == "query_events"
        assert retried["tool_calls"][-1]["status"] == "success"
    finally:
        store.close()
