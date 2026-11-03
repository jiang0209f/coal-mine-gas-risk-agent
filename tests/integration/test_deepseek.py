"""Offline acceptance checks for the DeepSeek transport boundary."""

from __future__ import annotations

import json
import os

import httpx
import pytest

from app.adapters.deepseek import (
    DeepSeekAdapter,
    DeepSeekConfigurationError,
    DeepSeekProtocolError,
    DeepSeekRequestError,
)
from app.config import Settings


TEST_KEY = "test-key-not-a-real-credential"
MESSAGES = [{"role": "user", "content": "Reply with ready."}]


@pytest.fixture(autouse=True)
def isolate_deepseek_environment(monkeypatch):
    for name in tuple(os.environ):
        if name.upper().startswith("DEEPSEEK_"):
            monkeypatch.delenv(name, raising=False)


def settings(**overrides):
    values = {
        "deepseek_api_key": TEST_KEY,
        "deepseek_model": "deepseek-flash",
        "deepseek_max_retries": 1,
        "deepseek_timeout_seconds": 30,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def completion(message=None, finish_reason="stop"):
    return {
        "id": "chatcmpl-offline-test",
        "object": "chat.completion",
        "model": "deepseek-flash",
        "choices": [
            {
                "index": 0,
                "message": message or {"role": "assistant", "content": "ready"},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9},
    }


def test_chat_sends_configured_request_and_returns_provider_reply():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    with DeepSeekAdapter(settings(), transport=httpx.MockTransport(handle)) as adapter:
        reply = adapter.chat(MESSAGES)

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.deepseek.com/chat/completions"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "deepseek-flash"
    assert body["messages"] == MESSAGES
    assert body["thinking"] == {"type": "disabled"}
    assert body["stream"] is False
    assert reply.message == {"role": "assistant", "content": "ready"}
    assert reply.finish_reason == "stop"
    assert reply.model == "deepseek-flash"
    assert reply.usage["total_tokens"] == 9


def test_chat_preserves_tool_ids_and_arguments_and_forwards_tool_configuration():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "integration_echo",
                "description": "Return the supplied nonce.",
                "parameters": {
                    "type": "object",
                    "properties": {"nonce": {"type": "string"}},
                    "required": ["nonce"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    choice = {"type": "function", "function": {"name": "integration_echo"}}
    tool_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-offline-1",
                "type": "function",
                "function": {
                    "name": "integration_echo",
                    "arguments": '{"nonce":"roundtrip-123"}',
                },
            }
        ],
    }
    bodies = []

    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion(tool_message, "tool_calls"))

    with DeepSeekAdapter(settings(), transport=httpx.MockTransport(handle)) as adapter:
        reply = adapter.chat(MESSAGES, tools=tools, tool_choice=choice)

    assert bodies[0]["tools"] == tools
    assert bodies[0]["tool_choice"] == choice
    assert reply.message == tool_message
    assert reply.finish_reason == "tool_calls"


def test_chat_uses_the_configured_base_url_without_losing_its_path():
    urls = []

    def handle(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=completion())

    config = settings(deepseek_base_url="https://api.deepseek.com/v1/")
    with DeepSeekAdapter(config, transport=httpx.MockTransport(handle)) as adapter:
        adapter.chat(MESSAGES)

    assert urls == ["https://api.deepseek.com/v1/chat/completions"]


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_transient_http_error_retries_once_then_returns_success(status_code):
    requests = []
    sleeps = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(status_code, json={"error": "temporary"})
        return httpx.Response(200, json=completion())

    with DeepSeekAdapter(
        settings(), transport=httpx.MockTransport(handle), sleep=sleeps.append
    ) as adapter:
        reply = adapter.chat(MESSAGES)

    assert len(requests) == 2
    assert len(sleeps) == 1
    assert reply.message["content"] == "ready"


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_transient_network_error_retries_once_then_returns_success(error_type):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            raise error_type("temporary transport failure", request=request)
        return httpx.Response(200, json=completion())

    with DeepSeekAdapter(
        settings(), transport=httpx.MockTransport(handle), sleep=lambda _: None
    ) as adapter:
        reply = adapter.chat(MESSAGES)

    assert len(requests) == 2
    assert reply.message["content"] == "ready"


@pytest.mark.parametrize("status_code", [429, 503])
def test_exhausted_http_retries_raise_sanitized_error(status_code):
    requests = []
    upstream_message = f"upstream-private-message {TEST_KEY}"

    def handle(request):
        requests.append(request)
        return httpx.Response(status_code, json={"error": upstream_message})

    with DeepSeekAdapter(
        settings(), transport=httpx.MockTransport(handle), sleep=lambda _: None
    ) as adapter:
        with pytest.raises(DeepSeekRequestError) as caught:
            adapter.chat(MESSAGES)

    assert len(requests) == 2
    assert caught.value.status_code == status_code
    assert TEST_KEY not in str(caught.value)
    assert upstream_message not in str(caught.value)


def test_exhausted_timeouts_stop_after_two_attempts_and_hide_transport_details():
    requests = []

    def handle(request):
        requests.append(request)
        raise httpx.ReadTimeout(f"private transport details {TEST_KEY}", request=request)

    with DeepSeekAdapter(
        settings(), transport=httpx.MockTransport(handle), sleep=lambda _: None
    ) as adapter:
        with pytest.raises(DeepSeekRequestError) as caught:
            adapter.chat(MESSAGES)

    assert len(requests) == 2
    assert caught.value.status_code is None
    assert TEST_KEY not in str(caught.value)
    assert "private transport details" not in str(caught.value)


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_permanent_http_error_does_not_retry(status_code):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(status_code, json={"error": f"private {TEST_KEY}"})

    with DeepSeekAdapter(
        settings(), transport=httpx.MockTransport(handle), sleep=lambda _: None
    ) as adapter:
        with pytest.raises(DeepSeekRequestError) as caught:
            adapter.chat(MESSAGES)

    assert len(requests) == 1
    assert caught.value.status_code == status_code
    assert TEST_KEY not in str(caught.value)


def test_retry_can_be_disabled():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(429, json={"error": "temporary"})

    with DeepSeekAdapter(
        settings(deepseek_max_retries=0),
        transport=httpx.MockTransport(handle),
        sleep=lambda _: None,
    ) as adapter:
        with pytest.raises(DeepSeekRequestError):
            adapter.chat(MESSAGES)

    assert len(requests) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"finish_reason": "stop"}]},
        completion({"role": "assistant", "content": None}),
        completion({"role": "assistant", "content": ""}),
    ],
    ids=["missing-choices", "empty-choices", "missing-message", "null-message", "empty-message"],
)
def test_incomplete_response_is_protocol_error_without_retry(payload):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    with DeepSeekAdapter(
        settings(), transport=httpx.MockTransport(handle), sleep=lambda _: None
    ) as adapter:
        with pytest.raises(DeepSeekProtocolError):
            adapter.chat(MESSAGES)

    assert len(requests) == 1


def test_invalid_response_json_is_protocol_error_without_leaking_its_body():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, text=f"not-json private {TEST_KEY}")

    with DeepSeekAdapter(
        settings(), transport=httpx.MockTransport(handle), sleep=lambda _: None
    ) as adapter:
        with pytest.raises(DeepSeekProtocolError) as caught:
            adapter.chat(MESSAGES)

    assert len(requests) == 1
    assert TEST_KEY not in str(caught.value)
    assert "not-json private" not in str(caught.value)


def test_truncated_response_is_not_accepted_as_a_completed_reply():
    def handle(request):
        return httpx.Response(200, json=completion(finish_reason="length"))

    with DeepSeekAdapter(settings(), transport=httpx.MockTransport(handle)) as adapter:
        with pytest.raises(DeepSeekProtocolError):
            adapter.chat(MESSAGES)


@pytest.mark.parametrize("tool_calls", [{}, "", 0, False])
def test_non_list_tool_calls_is_rejected_even_when_content_is_valid(tool_calls):
    def handle(request):
        return httpx.Response(
            200,
            json=completion(
                {"role": "assistant", "content": "ready", "tool_calls": tool_calls}
            ),
        )

    with DeepSeekAdapter(settings(), transport=httpx.MockTransport(handle)) as adapter:
        with pytest.raises(DeepSeekProtocolError):
            adapter.chat(MESSAGES)


@pytest.mark.parametrize("content", [{"text": "ready"}, ["ready"], 1, True])
def test_non_text_content_is_rejected_even_when_tool_calls_are_valid(content):
    message = {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call-structure-test",
                "type": "function",
                "function": {
                    "name": "integration_echo",
                    "arguments": '{"nonce":"roundtrip-123"}',
                },
            }
        ],
    }

    def handle(request):
        return httpx.Response(200, json=completion(message, "tool_calls"))

    with DeepSeekAdapter(settings(), transport=httpx.MockTransport(handle)) as adapter:
        with pytest.raises(DeepSeekProtocolError):
            adapter.chat(MESSAGES)


@pytest.mark.parametrize("finish_reason", [[], {}])
def test_invalid_finish_reason_type_is_reported_as_protocol_error(finish_reason):
    def handle(request):
        return httpx.Response(200, json=completion(finish_reason=finish_reason))

    with DeepSeekAdapter(settings(), transport=httpx.MockTransport(handle)) as adapter:
        with pytest.raises(DeepSeekProtocolError):
            adapter.chat(MESSAGES)


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_missing_key_fails_before_any_network_request(api_key):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    with pytest.raises(DeepSeekConfigurationError):
        DeepSeekAdapter(
            settings(deepseek_api_key=api_key), transport=httpx.MockTransport(handle)
        )

    assert requests == []
