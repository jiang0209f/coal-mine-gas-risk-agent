import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings


@pytest.fixture(autouse=True)
def isolate_deepseek_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
        "DEEPSEEK_TIMEOUT_SECONDS",
        "DEEPSEEK_MAX_RETRIES",
        "DEEPSEEK_MAX_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_health_succeeds_without_deepseek_key() -> None:
    from app.main import create_app

    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "coal-mine-gas-risk-agent"
    assert payload["deepseek"]["configured"] is False
    assert payload["deepseek"]["model"] == "deepseek-flash"


def test_health_reports_key_presence_without_returning_the_key() -> None:
    from app.main import create_app

    secret = "test-key-for-health-only"
    settings = Settings(
        _env_file=None,
        deepseek_api_key=secret,
        deepseek_model="configured-test-model",
    )
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["deepseek"]["configured"] is True
    assert response.json()["deepseek"]["model"] == "configured-test-model"
    assert secret not in response.text
    assert "api_key" not in response.text.lower()
    assert "authorization" not in response.text.lower()


def test_health_does_not_call_an_external_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.main import create_app

    def reject_external_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("Health checks must not send external HTTP requests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_external_request)
    settings = Settings(_env_file=None, deepseek_api_key="test-key-for-no-network")

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200


def test_default_app_is_available_for_the_server_entrypoint() -> None:
    from app.main import app

    assert isinstance(app, FastAPI)
