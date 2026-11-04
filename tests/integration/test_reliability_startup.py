"""Application restart recovery and the arrival-time index on existing stores."""
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from sqlalchemy import inspect

from app.config import Settings
from app.db import Store
from app.main import create_app
from app.schemas.common import Actor


def test_existing_database_gets_arrival_cursor_index(tmp_path):
    url = "sqlite+pysqlite:///" + (tmp_path / "old.db").as_posix()
    store = Store(url)
    with store.transaction() as connection:
        connection.exec_driver_sql("DROP INDEX IF EXISTS ix_observations_point_available_event")
    store.close()
    restarted = Store(url)
    try:
        indices = inspect(restarted.engine).get_indexes("observations")
        assert any(index["name"] == "ix_observations_point_available_event"
                   and index["column_names"] == ["point_id", "available_at", "event_time"] for index in indices)
    finally:
        restarted.close()


def test_application_startup_recovers_interrupted_ordinary_chat(tmp_path):
    settings = Settings(_env_file=None, runtime_dir=tmp_path,
        database_url="sqlite+pysqlite:///" + (tmp_path / "restart.db").as_posix(),
        ingestion_worker_enabled=False, deepseek_api_key=None, event_brief_allowed_source_ids=[])
    app = create_app(settings)
    actor = Actor(user_id="local-test", roles=("safety_management",), face_ids=("31002",))
    with TestClient(app):
        services = app.state.services
        services["data"].register_source("synthetic-startup", "synthetic", "31002")
        session = services["sessions"].create_session(actor, "historical_replay", "synthetic-startup",
            datetime.now(timezone.utc)-timedelta(hours=1))
        context = services["sessions"].get_context(actor, session["session_id"])
        task = services["agent"].submit(context, "当前有哪些预警事件", schedule=lambda callback: None)
        assert task["state"] == "running"
    second = create_app(settings)
    with TestClient(second):
        recovered = second.state.services["agent"].get_task(actor, task["task_id"])
        assert recovered["state"] == "failed" and recovered["error_code"] == "task_interrupted"
