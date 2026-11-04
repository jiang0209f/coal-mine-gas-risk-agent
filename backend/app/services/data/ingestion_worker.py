"""Bounded local ingestion retries with graceful shutdown and safe health state."""
import asyncio

from app.schemas.common import now_utc, utcstr

__all__ = ["run_ingestion_worker"]


async def run_ingestion_worker(service, stop, status, interval_seconds=1):
    status.update(enabled=True, state="starting", last_run_at=None,
                  last_success_at=None, last_error_code=None)
    try:
        while not stop.is_set():
            status["last_run_at"] = utcstr(now_utc())
            try:
                await asyncio.to_thread(service.process_pending, max_batches=10)
                status.update(state="running", last_success_at=utcstr(now_utc()), last_error_code=None)
            except Exception:
                status.update(state="degraded", last_error_code="ingestion_worker_failed")
            if not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
                except TimeoutError:
                    pass
    finally:
        status["state"] = "stopped"
