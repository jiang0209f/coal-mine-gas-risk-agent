"""Worker shutdown waits for in-flight durable processing and publishes safe health."""
import asyncio
import threading

from app.services.data.ingestion_worker import run_ingestion_worker


def test_worker_waits_for_current_transaction_before_shutdown():
    async def scenario():
        entered, release = threading.Event(), threading.Event()
        stop, status = asyncio.Event(), {}

        class Service:
            calls = 0

            def process_pending(self, max_batches=10):
                self.calls += 1
                entered.set()
                assert release.wait(2)
                return {"processed": 1}

        service = Service()
        task = asyncio.create_task(run_ingestion_worker(service, stop, status, interval_seconds=.01))
        assert await asyncio.to_thread(entered.wait, 2)
        stop.set()
        await asyncio.sleep(.01)
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, 2)
        assert service.calls == 1 and status["state"] == "stopped"
        assert status["last_success_at"] and status["last_error_code"] is None

    asyncio.run(scenario())


def test_worker_failure_is_visible_without_private_exception_text():
    async def scenario():
        stop, status = asyncio.Event(), {}

        class Service:
            def process_pending(self, max_batches=10):
                raise RuntimeError("PRIVATE_PAYLOAD sk-private-example-secret F:/private/config")

        task = asyncio.create_task(run_ingestion_worker(Service(), stop, status, interval_seconds=1))
        for _ in range(100):
            if status.get("state") == "degraded":
                break
            await asyncio.sleep(.01)
        assert status["state"] == "degraded"
        assert status["last_error_code"] == "ingestion_worker_failed"
        assert "PRIVATE_PAYLOAD" not in str(status) and "sk-private" not in str(status)
        stop.set()
        await asyncio.wait_for(task, 2)

    asyncio.run(scenario())
