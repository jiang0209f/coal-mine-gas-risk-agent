"""验证普通聊天及一次完整的工具调用往返。"""

import json
import sys
from uuid import uuid4

from pydantic import ValidationError

from app.adapters.deepseek import (
    DeepSeekAdapter,
    DeepSeekConfigurationError,
    DeepSeekError,
    DeepSeekProtocolError,
)
from app.config import Settings


def run_checks(adapter: DeepSeekAdapter) -> dict:
    basic = adapter.chat([{"role": "user", "content": "Reply exactly: ready"}])
    if basic.message.get("tool_calls"):
        raise DeepSeekProtocolError("普通对话诊断收到了意外工具调用。")

    nonce = uuid4().hex
    tools = [{
        "type": "function",
        "function": {
            "name": "integration_echo",
            "description": "Return the supplied diagnostic nonce unchanged.",
            "parameters": {
                "type": "object",
                "properties": {"nonce": {"type": "string"}},
                "required": ["nonce"],
                "additionalProperties": False,
            },
        },
    }]
    messages = [{
        "role": "user",
        "content": (
            f"Call integration_echo exactly once with nonce={nonce}. "
            "After receiving its result, reply with the same nonce."
        ),
    }]
    requested = adapter.chat(
        messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "integration_echo"}},
    )
    calls = requested.message.get("tool_calls", [])
    if len(calls) != 1 or calls[0]["function"]["name"] != "integration_echo":
        raise DeepSeekProtocolError("诊断只允许一次integration_echo工具调用。")
    try:
        arguments = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError):
        raise DeepSeekProtocolError("诊断工具参数不是有效JSON。") from None
    if not isinstance(arguments, dict) or arguments != {"nonce": nonce}:
        raise DeepSeekProtocolError("诊断工具参数与预期不一致。")

    messages.extend([
        requested.message,
        {"role": "tool", "tool_call_id": calls[0]["id"], "content": json.dumps(arguments)},
    ])
    final = adapter.chat(messages, tools=tools, tool_choice="none")
    if final.message.get("tool_calls") or nonce not in (final.message.get("content") or ""):
        raise DeepSeekProtocolError("工具结果回传后的答复未通过校验。")
    return {
        "chat": {"status": "passed", "model": basic.model, "reply": basic.message["content"]},
        "tool_roundtrip": {
            "status": "passed",
            "tool_name": "integration_echo",
            "model": final.model,
            "reply": final.message["content"],
        },
    }


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        settings = Settings()
        with DeepSeekAdapter(settings) as adapter:
            result = run_checks(adapter)
    except ValidationError:
        print("DeepSeek配置无效，请检查.env中的地址、超时和参数。", file=sys.stderr)
        return 2
    except DeepSeekConfigurationError as error:
        print(str(error), file=sys.stderr)
        return 2
    except DeepSeekError as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
