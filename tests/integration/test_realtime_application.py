"""Application-owned realtime monitoring progresses without a browser tick."""
from datetime import timedelta
import time

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from app.config import Settings
from app.main import create_app
from app.schemas.common import Actor, now_utc


def test_realtime_worker_configuration_is_bounded():
    settings = Settings(_env_file=None)
    assert settings.realtime_monitor_enabled and settings.realtime_monitor_interval_seconds == 5
    for value in (0, -1, float("nan"), 61):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, realtime_monitor_interval_seconds=value)


def test_application_background_monitors_real_time_without_http_tick(tmp_path):
    settings = Settings(_env_file=None, runtime_dir=tmp_path,
        database_url="sqlite+pysqlite:///:memory:", ingestion_worker_enabled=False,
        realtime_monitor_enabled=True, realtime_monitor_interval_seconds=.02,
        deepseek_api_key=None, event_brief_allowed_source_ids=[])
    app = create_app(settings)
    actor = Actor(user_id="qa", roles=("safety_management",), face_ids=("synthetic",))
    with TestClient(app):
        services = app.state.services
        services["data"].register_source("synthetic-realtime", "synthetic", "synthetic")
        services["data"].register_point("qa-t0", "synthetic-realtime", "T0", "T0", "methane", "%", unit_status="confirmed")
        session = services["sessions"].create_session(actor, "realtime", "synthetic-realtime", now_utc()-timedelta(seconds=2))
        services["sessions"].apply_action(actor, session["session_id"], "resume")
        services["data"].ingest_readings(actor, "synthetic-realtime", [{"reading_id": "red", "point_id": "qa-t0",
            "event_time": (now_utc()-timedelta(seconds=1)).isoformat(), "raw_value": "1.2"}])
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            saved = services["sessions"].get_session(actor, session["session_id"])
            if saved.get("monitor", {}).get("last_success_at") and services["store"].list("event"):
                break
            time.sleep(.02)
        assert saved["monitor"]["state"] == "running"
        assert services["store"].list("event")[0]["alert_level"] == "red"
        assert app.state.realtime_monitor_status["enabled"]
    assert app.state.realtime_monitor_status["state"] == "stopped"


def test_application_can_explicitly_disable_background_monitoring(tmp_path):
    app = create_app(Settings(_env_file=None, runtime_dir=tmp_path, database_url="sqlite+pysqlite:///:memory:",
        ingestion_worker_enabled=False, realtime_monitor_enabled=False, deepseek_api_key=None,
        event_brief_allowed_source_ids=[]))
    with TestClient(app):
        assert app.state.realtime_monitor_status == {"enabled": False, "state": "disabled"}
