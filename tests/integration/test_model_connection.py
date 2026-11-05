"""Model connection settings stay local; connectivity runs only on explicit test."""
from copy import deepcopy
import json

import pytest
from pydantic import SecretStr

from app.adapters.deepseek import ChatReply, DeepSeekRequestError
from app.config import Settings
from app.db import Store
from app.schemas.common import Actor, DomainError
from app.services.model_connection.service import ModelConnectionService


KEY_ONE = "sk-offline-connection-test-secret-728463"
KEY_TWO = "sk-offline-replacement-test-secret-915372"


class Adapter:
    def __init__(self, settings, responses):
        self.settings, self.responses = settings, responses
        self.requests = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.requests.append({"messages": deepcopy(messages), **deepcopy(kwargs)})
        response = self.responses.pop(0) if self.responses else ChatReply(
            {"role": "assistant", "content": "OK"}, "stop", "deepseek-flash", {"total_tokens": 1})
        if isinstance(response, Exception):
            raise response
        return response

    def close(self):
        self.closed = True


@pytest.fixture
def parts(tmp_path):
    store = Store("sqlite+pysqlite:///:memory:")
    settings = Settings(_env_file=None, runtime_dir=tmp_path, deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com", deepseek_model="previous-model",
        deepseek_max_tokens=512, deepseek_max_retries=1, deepseek_timeout_seconds=30)
    actor = Actor(user_id="manager", roles=("safety_management",), face_ids=("FACE",))
    instances, responses = [], []

    def factory(configuration):
        adapter = Adapter(configuration, responses)
        instances.append(adapter)
        return adapter

    yield store, settings, actor, instances, responses, factory
    store.close()


def assert_public(value):
    encoded = json.dumps(value, ensure_ascii=False)
    for secret in (KEY_ONE, KEY_TWO, KEY_ONE[-6:], KEY_TWO[-6:]):
        assert secret not in encoded
    assert "api_key" not in value
    assert value["local_available"] is False
    assert {"configured", "provider", "base_url", "model", "source", "local_available"} <= value.keys()


def test_save_updates_shared_settings_keeps_blank_key_and_restores_local_file(parts):
    store, settings, actor, adapters, _, factory = parts
    service = ModelConnectionService(store, settings, adapter_factory=factory)
    assert service.get(actor)["configured"] is False
    result = service.save(actor, provider="deepseek", base_url="https://api.deepseek.com/v1",
                          model="deepseek-flash", api_key=SecretStr(KEY_ONE))
    assert_public(result)
    assert result["configured"] is True and settings.deepseek_configured
    assert settings.deepseek_api_key.get_secret_value() == KEY_ONE
    assert str(settings.deepseek_base_url).rstrip("/") == "https://api.deepseek.com/v1"
    assert settings.deepseek_model == "deepseek-flash"
    service.save(actor, provider="deepseek", base_url="https://api.deepseek.com", model="deepseek-flash", api_key=KEY_TWO)
    service.save(actor, provider="deepseek", base_url="https://api.deepseek.com", model="deepseek-flash", api_key="")
    assert settings.deepseek_api_key.get_secret_value() == KEY_TWO
    assert adapters == []
    assert (settings.runtime_dir / "model-connection.json").is_file()
    restored = Settings(_env_file=None, runtime_dir=settings.runtime_dir, deepseek_api_key=None)
    reloaded = ModelConnectionService(store, restored, adapter_factory=factory)
    assert_public(reloaded.get(actor))
    assert reloaded.get(actor)["configured"] is True
    assert restored.deepseek_api_key.get_secret_value() == KEY_TWO
    assert adapters == []


def test_only_safety_management_can_get_save_or_test_connection(parts):
    store, settings, actor, adapters, _, factory = parts
    service = ModelConnectionService(store, settings, adapter_factory=factory)
    for role in ("ventilation", "gas_governance"):
        worker = actor.model_copy(update={"roles": (role,)})
        for method, kwargs in ((service.get, {}), (service.save, {"provider": "deepseek", "base_url": "https://api.deepseek.com", "model": "deepseek-flash", "api_key": KEY_ONE}), (service.test, {})):
            with pytest.raises(DomainError) as caught:
                method(worker, **kwargs)
            assert caught.value.status_code == 403
            assert KEY_ONE not in str(caught.value)
    assert adapters == []
    assert not (settings.runtime_dir / "model-connection.json").exists()


def test_only_official_deepseek_endpoint_is_allowed_and_local_is_reserved(parts):
    store, settings, actor, adapters, _, factory = parts
    service = ModelConnectionService(store, settings, adapter_factory=factory)
    for endpoint in (
        "http://api.deepseek.com", "https://other.example.com", "https://api.deepseek.com.evil.example",
        "https://user:password@api.deepseek.com", "https://api.deepseek.com:443",
        "https://api.deepseek.com:8443", "https://api.deepseek.com/v2",
        "https://api.deepseek.com/v1/chat/completions", "https://api.deepseek.com?token=PRIVATE_QUERY",
        "https://api.deepseek.com#fragment",
    ):
        with pytest.raises(DomainError) as caught:
            service.save(actor, provider="deepseek", base_url=endpoint, model="deepseek-flash", api_key=KEY_ONE)
        assert KEY_ONE not in str(caught.value)
    with pytest.raises(DomainError) as caught:
        service.save(actor, provider="local", base_url="https://api.deepseek.com", model="reserved", api_key=None)
    assert caught.value.code == "not_implemented"
    assert adapters == [] and not settings.deepseek_configured
    assert not (settings.runtime_dir / "model-connection.json").exists()


def test_connectivity_test_is_explicit_small_and_sends_only_fixed_synthetic_prompt(parts):
    store, settings, actor, adapters, _, factory = parts
    settings.event_brief_allowed_source_ids = ["REAL_SOURCE_CANARY"]
    settings.agent_real_source_ids = ["REAL_SOURCE_CANARY"]
    service = ModelConnectionService(store, settings, adapter_factory=factory)
    service.save(actor, provider="deepseek", base_url="https://api.deepseek.com", model="deepseek-flash", api_key=KEY_ONE)
    service.get(actor)
    assert adapters == []
    result = service.test(actor)
    assert result["ok"] is True and isinstance(result.get("message"), str)
    assert len(adapters) == 1 and len(adapters[0].requests) == 1 and adapters[0].closed
    adapter = adapters[0]
    assert adapter.settings.deepseek_max_tokens == 16
    assert adapter.settings.deepseek_max_retries == 0
    assert 0 < adapter.settings.deepseek_timeout_seconds <= 10
    assert settings.deepseek_max_tokens == 512 and settings.deepseek_max_retries == 1
    wire = json.dumps(adapter.requests, ensure_ascii=False)
    assert "Reply OK" in wire
    assert "REAL_SOURCE_CANARY" not in wire and KEY_ONE not in wire
    assert adapter.requests[0].get("tools") in (None, [])
    assert KEY_ONE not in json.dumps(result, ensure_ascii=False)


def test_connectivity_errors_are_generic_and_never_echo_upstream_body_or_key(parts):
    store, settings, actor, adapters, responses, factory = parts
    service = ModelConnectionService(store, settings, adapter_factory=factory)
    service.save(actor, provider="deepseek", base_url="https://api.deepseek.com", model="deepseek-flash", api_key=KEY_ONE)
    for error in (DeepSeekRequestError("RAW_UPSTREAM_BODY " + KEY_ONE, 500), RuntimeError("RAW_UPSTREAM_BODY " + KEY_ONE)):
        responses.append(error)
        result = service.test(actor)
        assert result["ok"] is False and result.get("error_code")
        encoded = json.dumps(result, ensure_ascii=False)
        assert KEY_ONE not in encoded and "RAW_UPSTREAM_BODY" not in encoded
        assert isinstance(result.get("message"), str) and adapters[-1].closed
