"""Local realtime session monitoring and managed event-brief work."""
import asyncio

from app.schemas.common import Actor, now_utc, utcstr

__all__ = ["RealtimeMonitor", "run_realtime_monitor"]


class RealtimeMonitor:
    def __init__(self, services, clock=now_utc):
        self.services, self.store, self.clock = services, services["store"], clock

    def process_running(self, schedule=None):
        processed, failed = 0, 0
        sessions = self.services["sessions"]
        for saved in self.store.list("session"):
            if saved["mode"] != "realtime" or saved["state"] != "running":
                continue
            sid = saved["session_id"]
            with sessions.execution_lock(sid):
                current = self.store.get("session", sid)
                if current is None or current["mode"] != "realtime" or current["state"] != "running":
                    continue
                state = self.store.get("session_monitor", sid) or dict(
                    session_id=sid, source_id=current["source_id"], face_id=current["face_id"],
                    last_success_at=None, consecutive_failures=0)
                state["last_run_at"] = utcstr(self.clock())
                try:
                    source = self.services["data"]._source(current["source_id"])
                    actor = Actor(user_id="realtime-monitor:" + sid, roles=("ventilation",),
                                  face_ids=(source["face_id"],), is_system=True)
                    self.services["runner"].run_once(actor, sid, target_as_of=self.clock(), schedule=schedule)
                    state.update(state="running", last_success_at=utcstr(self.clock()),
                                 last_error_code=None, consecutive_failures=0)
                    processed += 1
                except Exception:
                    state.update(state="error", last_error_code="realtime_tick_failed",
                                 consecutive_failures=state["consecutive_failures"] + 1)
                    failed += 1
                self.store.put("session_monitor", sid, state)
        return {"processed_sessions": processed, "failed_sessions": failed}


async def run_realtime_monitor(service, stop, status, interval_seconds=5):
    """Stop scanning, finish the in-flight tick, then drain owned brief tasks."""
    status.update(enabled=True, state="starting", last_run_at=None, last_success_at=None,
                  last_error_code=None, pending_briefs=0)
    loop, jobs, slots = asyncio.get_running_loop(), set(), asyncio.Semaphore(2)

    async def run_brief(work):
        async with slots:
            try:
                await asyncio.to_thread(work)
            except Exception:
                status["last_brief_error_code"] = "realtime_brief_failed"

    def finished(task):
        jobs.discard(task)
        status["pending_briefs"] = len(jobs)

    def enqueue(work):
        task = asyncio.create_task(run_brief(work))
        jobs.add(task)
        status["pending_briefs"] = len(jobs)
        task.add_done_callback(finished)

    def schedule(work):
        loop.call_soon_threadsafe(enqueue, work)

    try:
        while not stop.is_set():
            status["last_run_at"] = utcstr(now_utc())
            try:
                result = await asyncio.to_thread(service.process_running, schedule=schedule)
                status.update(state="degraded" if result["failed_sessions"] else "running",
                              last_success_at=utcstr(now_utc()),
                              last_error_code="realtime_session_failed" if result["failed_sessions"] else None)
            except Exception:
                status.update(state="degraded", last_error_code="realtime_scan_failed")
            if not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
                except TimeoutError:
                    pass
    finally:
        await asyncio.sleep(0)
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)
        status.update(state="stopped", pending_briefs=0)
