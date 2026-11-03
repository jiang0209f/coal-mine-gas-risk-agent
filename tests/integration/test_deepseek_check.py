import json
import os
import re

import httpx
import pytest

from app.adapters.deepseek import DeepSeekAdapter, DeepSeekProtocolError
from app.config import Settings
from app.scripts.check_deepseek import run_checks


@pytest.fixture(autouse=True)
def isolate_deepseek_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.upper().startswith("DEEPSEEK_"):
            monkeypatch.delenv(name, raising=False)


def completion(message: dict, finish_reason: str = "stop") -> dict:
    return {
        "model": "deepseek-flash",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def nonce_from_request(body: dict) -> str:
    content = "\n".join(
        message["content"]
        for message in body["messages"]
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
    )
    match = re.search(r"nonce=([A-Za-z0-9-]+)", content)
    assert match is not None, "The diagnostic prompt must include its nonce"
    return match.group(1)


def test_checks_complete_chat_and_a_three_request_tool_roundtrip() -> None:
    requests = []
    expected_nonce = None
    expected_tool_calls = None

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal expected_nonce, expected_tool_calls
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            assert not body.get("tools")
            return httpx.Response(
                200, json=completion({"role": "assistant", "content": "ready"})
            )
        if len(requests) == 2:
            expected_nonce = nonce_from_request(body)
            assert body["tool_choice"] == {
                "type": "function",
                "function": {"name": "integration_echo"},
            }
            assert [tool["function"]["name"] for tool in body["tools"]] == [
                "integration_echo"
            ]
            expected_tool_calls = [
                {
                    "id": "call-integration-echo",
                    "type": "function",
                    "function": {
                        "name": "integration_echo",
                        "arguments": json.dumps({"nonce": expected_nonce}),
                    },
                }
            ]
            return httpx.Response(
                200,
                json=completion(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": expected_tool_calls,
                    },
                    finish_reason="tool_calls",
                ),
            )
        assert len(requests) == 3, "The diagnostic must finish in three requests"
        assistant_calls = [
            message["tool_calls"]
            for message in body["messages"]
            if message.get("role") == "assistant" and message.get("tool_calls")
        ]
        assert assistant_calls == [expected_tool_calls]
        tool_results = [
            message for message in body["messages"] if message.get("role") == "tool"
        ]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_call_id"] == "call-integration-echo"
        assert json.loads(tool_results[0]["content"])["nonce"] == expected_nonce
        assert body["tool_choice"] == "none"
        return httpx.Response(
            200,
            json=completion(
                {"role": "assistant", "content": f"roundtrip passed nonce={expected_nonce}"}
            ),
        )

    settings = Settings(
        _env_file=None,
        deepseek_api_key="test-key-for-diagnostic-only",
        deepseek_max_retries=0,
    )
    with DeepSeekAdapter(settings, transport=httpx.MockTransport(handle)) as adapter:
        result = run_checks(adapter)

    assert len(requests) == 3
    assert result["chat"]["status"] == "passed"
    assert result["tool_roundtrip"]["status"] == "passed"
    assert result["tool_roundtrip"]["tool_name"] == "integration_echo"


@pytest.mark.parametrize("invalid_reply", ["unknown_tool", "wrong_nonce", "bad_json"])
def test_checks_reject_invalid_tool_calls_before_the_final_request(
    invalid_reply: str,
) -> None:
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                200, json=completion({"role": "assistant", "content": "ready"})
            )
        assert len(requests) == 2, "An invalid tool call must not be executed"
        nonce = nonce_from_request(body)
        name = "unexpected_function" if invalid_reply == "unknown_tool" else "integration_echo"
        arguments = json.dumps(
            {"nonce": "incorrect-nonce" if invalid_reply == "wrong_nonce" else nonce}
        )
        if invalid_reply == "bad_json":
            arguments = "{invalid-json"
        return httpx.Response(
            200,
            json=completion(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-invalid-diagnostic",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                finish_reason="tool_calls",
            ),
        )

    settings = Settings(
        _env_file=None,
        deepseek_api_key="test-key-for-diagnostic-only",
        deepseek_max_retries=0,
    )
    with DeepSeekAdapter(settings, transport=httpx.MockTransport(handle)) as adapter:
        with pytest.raises(DeepSeekProtocolError):
            run_checks(adapter)

    assert len(requests) == 2
