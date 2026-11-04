"""Server-side realtime monitoring without a browser, scoped and restartable."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Event

import pytest

from app.config import Settings
from app.main import build_services
from app.schemas.common import Actor


START = datetime(2022,2,5,12,5,tzinfo=timezone.utc)


@pytest.fixture
def live(tmp_path,monkeypatch):
    clock = [START]
    monkeypatch.setattr("app.services.sessions.service.now_utc",lambda:clock[0])
    monkeypatch.setattr("app.services.sessions.runner.now_utc",lambda:clock[0])
    settings = Settings(_env_file=None,runtime_dir=tmp_path,database_url="sqlite+pysqlite:///:memory:",
        model_catalog_path=None,deepseek_api_key=None,event_brief_allowed_source_ids=[],ingestion_worker_enabled=False)
    services = build_services(settings)
    for source,face in (("source","31002"),("other","32001")):
        services["data"].register_source(source,source,face)
        services["data"].register_point(source+"-t0",source,"T0","T0","methane","%",unit_status="confirmed")
        actor = Actor(user_id="synthetic-operator",roles=("safety_management",),face_ids=(face,))
        services["data"].ingest_readings(actor,source,[{"reading_id":"signal","point_id":source+"-t0",
            "event_time":(START+timedelta(seconds=30)).isoformat(),"raw_value":".90"}],received_at=START+timedelta(seconds=31))
    actor = Actor(user_id="synthetic-operator",roles=("safety_management",),face_ids=("31002","32001"))
    try:
        yield services,actor,clock
    finally:
        services["store"].close()


def session(services,actor,source="source",mode="realtime",running=True):
    entry = services["sessions"].create_session(actor,mode,source,START)
    if running:
        services["sessions"].apply_action(actor,entry["session_id"],"resume")
    return entry["session_id"]


def monitor(services,clock):
    from app.services.sessions.realtime_worker import RealtimeMonitor
    return RealtimeMonitor(services,clock=lambda:clock[0])


def test_only_running_realtime_sessions_advance_without_any_browser(live):
    services,actor,clock = live
    running = session(services,actor)
    paused = session(services,actor,running=False)
    historic = session(services,actor,mode="historical_replay")
    ended = session(services,actor)
    services["sessions"].apply_action(actor,ended,"stop")
    clock[0] += timedelta(minutes=1)
    result = monitor(services,clock).process_running()
    assert result["processed_sessions"] == 1 and result["failed_sessions"] == 0
    assert services["sessions"].get_context(actor,running).as_of == clock[0]
    for sid in (paused,historic,ended):
        assert services["sessions"].get_context(actor,sid).as_of == START
    live_session = services["sessions"].get_session(actor,running)
    assert live_session["monitor"]["state"] == "running" and live_session["monitor"]["last_success_at"]
    assert "monitor" not in services["store"].get("session",running)


def test_new_realtime_alert_schedules_a_brief_and_pause_stop_prevent_further_ticks(live):
    services,actor,clock = live
    sid = session(services,actor)
    jobs,submitted = [],[]
    class Briefs:
        def submit(self,context,event_id,event_revision=None,schedule=None):
            assert context.actor.is_system and context.actor.face_ids == ("31002",)
            submitted.append(event_id)
            schedule(lambda:None)
            return dict(event_id=event_id,event_revision=event_revision,task_id="local-brief",status="queued")
    services["event_briefs"] = Briefs()
    service = monitor(services,clock)
    clock[0] += timedelta(minutes=1)
    assert service.process_running(schedule=jobs.append)["processed_sessions"] == 1
    assert len(submitted) == len(jobs) == 1
    services["sessions"].apply_action(actor,sid,"pause")
    clock[0] += timedelta(minutes=1)
    assert service.process_running(schedule=jobs.append)["processed_sessions"] == 0
    assert services["sessions"].get_context(actor,sid).as_of == START+timedelta(minutes=1)
    assert services["sessions"].get_session(actor,sid)["monitor"]["state"] == "paused"
    services["sessions"].apply_action(actor,sid,"resume")
    assert service.process_running(schedule=jobs.append)["processed_sessions"] == 1
    services["sessions"].apply_action(actor,sid,"stop")
    clock[0] += timedelta(minutes=1)
    assert service.process_running()["processed_sessions"] == 0
    assert services["sessions"].get_context(actor,sid).as_of == START+timedelta(minutes=2)
    assert services["sessions"].get_session(actor,sid)["monitor"]["state"] == "stopped"


def test_one_failed_session_does_not_block_another_and_error_text_stays_local(live,monkeypatch):
    services,actor,clock = live
    broken = session(services,actor)
    healthy = session(services,actor,source="other")
    original = services["runner"].run_once
    def run_once(actor,sid,**kwargs):
        if sid == broken:
            raise RuntimeError("PRIVATE sk-synthetic-private-error F:/private/place")
        return original(actor,sid,**kwargs)
    monkeypatch.setattr(services["runner"],"run_once",run_once)
    clock[0] += timedelta(minutes=1)
    result = monitor(services,clock).process_running()
    assert result["processed_sessions"] == 1 and result["failed_sessions"] == 1
    assert services["sessions"].get_context(actor,healthy).as_of == clock[0]
    state = services["sessions"].get_session(actor,broken)["monitor"]
    assert state["state"] == "error" and state["last_error_code"] == "realtime_tick_failed"
    assert not any(private in str(state) for private in ("PRIVATE","sk-","F:/"))


def test_running_session_resumes_after_store_close_and_process_restart(tmp_path,monkeypatch):
    clock = [START]
    monkeypatch.setattr("app.services.sessions.service.now_utc",lambda:clock[0])
    monkeypatch.setattr("app.services.sessions.runner.now_utc",lambda:clock[0])
    settings = Settings(_env_file=None,runtime_dir=tmp_path,database_url="sqlite+pysqlite:///"+(tmp_path/"restart.db").as_posix(),
        model_catalog_path=None,deepseek_api_key=None,event_brief_allowed_source_ids=[])
    first = build_services(settings)
    actor = Actor(user_id="synthetic",roles=("safety_management",),face_ids=("31002",))
    first["data"].register_source("source","source","31002")
    first["data"].register_point("source-t0","source","T0","T0","methane","%",unit_status="confirmed")
    first["prediction"].register_model(actor,dict(model_id="selected-professional",version="v1",adapter_kind="not-installed",
        default_eligible=False,face_id="31002",source_id="source",point_ids=["source-t0"],target_quantity="methane",
        unit="%",aggregation_seconds=300,input_window=1,horizon_seconds=3600))
    sid = session(first,actor)
    first["sessions"].set_prediction_model(actor,sid,"source-t0","selected-professional","v1")
    clock[0] += timedelta(minutes=1)
    monitor(first,clock).process_running()
    first["store"].close()
    second = build_services(settings)
    try:
        clock[0] += timedelta(minutes=1)
        assert monitor(second,clock).process_running()["processed_sessions"] == 1
        assert second["sessions"].get_context(actor,sid).as_of == clock[0]
        assert second["sessions"].get_session(actor,sid)["monitor"]["consecutive_failures"] == 0
        assert second["sessions"].get_session(actor,sid)["prediction_models"]["source-t0"]["model_id"] == "selected-professional"
        tick = second["runner"].run_once(actor,sid,target_as_of=clock[0])
        assert tick["forecasts"][0]["model_id"] == "selected-professional" and tick["forecasts"][0]["status"] == "not_ready"
    finally:
        second["store"].close()


def test_worker_and_manual_tick_share_one_session_lock_and_reuse_the_result(live,monkeypatch):
    services,actor,clock = live
    sid = session(services,actor)
    entered,release,manual_started = Event(),Event(),Event()
    original = services["prediction"].run_prediction
    calls = []
    def slow_prediction(context,point_id):
        calls.append(point_id)
        entered.set()
        assert release.wait(3)
        return original(context,point_id)
    monkeypatch.setattr(services["prediction"],"run_prediction",slow_prediction)
    clock[0] += timedelta(minutes=1)
    def manual():
        manual_started.set()
        return services["runner"].run_once(actor,sid,target_as_of=clock[0])
    with ThreadPoolExecutor(max_workers=2) as pool:
        automatic = pool.submit(monitor(services,clock).process_running)
        assert entered.wait(2)
        second = pool.submit(manual)
        assert manual_started.wait(2) and not second.done()
        release.set()
        assert automatic.result(timeout=3)["processed_sessions"] == 1
        assert second.result(timeout=3)["reused"] is True
    assert calls == ["source-t0"]


def test_background_worker_waits_for_inflight_tick_and_scheduled_brief_before_shutdown():
    from app.services.sessions.realtime_worker import run_realtime_monitor
    async def scenario():
        tick_entered,release_tick,brief_entered,release_brief = Event(),Event(),Event(),Event()
        stop,status = asyncio.Event(),{}
        class Service:
            def process_running(self,schedule):
                tick_entered.set()
                assert release_tick.wait(3)
                def brief():
                    brief_entered.set()
                    assert release_brief.wait(3)
                schedule(brief)
                return {"processed_sessions":1,"failed_sessions":0}
        task = asyncio.create_task(run_realtime_monitor(Service(),stop,status,interval_seconds=.01))
        assert await asyncio.to_thread(tick_entered.wait,2)
        stop.set()
        await asyncio.sleep(.01)
        assert not task.done()
        release_tick.set()
        assert await asyncio.to_thread(brief_entered.wait,2)
        assert not task.done()
        release_brief.set()
        await asyncio.wait_for(task,3)
        assert status["state"] == "stopped" and status["pending_briefs"] == 0
    asyncio.run(scenario())


def test_background_worker_publishes_safe_scanner_error():
    from app.services.sessions.realtime_worker import run_realtime_monitor
    async def scenario():
        stop,status = asyncio.Event(),{}
        class Service:
            def process_running(self,schedule):
                raise RuntimeError("PRIVATE sk-private-error F:/private")
        task = asyncio.create_task(run_realtime_monitor(Service(),stop,status,interval_seconds=.01))
        for _ in range(100):
            if status.get("state") == "degraded":
                break
            await asyncio.sleep(.01)
        assert status["last_error_code"] == "realtime_scan_failed"
        assert "PRIVATE" not in str(status) and "sk-" not in str(status)
        stop.set()
        await asyncio.wait_for(task,2)
    asyncio.run(scenario())


@pytest.mark.parametrize("enabled,lifecycle,expected",[(False,"running","disabled"),(True,"running","running"),
    (False,"paused","paused"),(True,"paused","paused"),(False,"ended","stopped"),(True,"ended","stopped")])
def test_monitor_projection_obeys_runtime_switch_and_lifecycle_without_rewriting_records(live,enabled,lifecycle,expected):
    from app.services.sessions.service import SessionsService
    services,actor,_ = live
    sid = session(services,actor)
    if lifecycle != "running":
        services["sessions"].apply_action(actor,sid,"stop" if lifecycle == "ended" else "pause")
    saved = dict(session_id=sid,state="running",last_run_at=START.isoformat(),last_success_at=START.isoformat())
    services["store"].put("session_monitor",sid,saved)
    sessions = SessionsService(services["store"],realtime_monitor_enabled=enabled)
    projected = sessions.get_session(actor,sid)["monitor"]
    assert projected["state"] == expected and projected["enabled"] is enabled
    assert projected["last_success_at"] == saved["last_success_at"]
    assert "monitor" not in services["store"].get("session",sid)
    assert services["store"].get("session_monitor",sid) == saved
