"""DeepSeek聊天接口适配，使用普通工具调用模式。"""

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

import httpx

from app.config import Settings


__all__ = [
    "ChatReply",
    "DeepSeekAdapter",
    "DeepSeekError",
    "DeepSeekConfigurationError",
    "DeepSeekRequestError",
    "DeepSeekProtocolError",
]


class DeepSeekError(Exception):
    """可直接用于诊断输出的错误。"""


class DeepSeekConfigurationError(DeepSeekError):
    pass


class DeepSeekRequestError(DeepSeekError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class DeepSeekProtocolError(DeepSeekError):
    pass


@dataclass(frozen=True)
class ChatReply:
    message: dict[str, Any]
    finish_reason: str
    model: str
    usage: dict[str, Any]


class DeepSeekAdapter:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not settings.deepseek_configured:
            raise DeepSeekConfigurationError(
                "未配置DEEPSEEK_API_KEY，请在项目根目录.env或环境变量中填写。"
            )
        self._settings = settings.model_copy(deep=True)
        self._sleep = sleep
        self._url = str(self._settings.deepseek_base_url).rstrip("/") + "/chat/completions"
        self._client = httpx.Client(
            headers={
                "Authorization": "Bearer " + self._settings.deepseek_api_key.get_secret_value(),
                "Content-Type": "application/json",
            },
            timeout=self._settings.deepseek_timeout_seconds,
            transport=transport,
            follow_redirects=False,
        )

    def __enter__(self) -> "DeepSeekAdapter":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatReply:
        payload: dict[str, Any] = {
            "model": self._settings.deepseek_model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "stream": False,
            "max_tokens": self._settings.deepseek_max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        for attempt in range(self._settings.deepseek_max_retries + 1):
            can_retry = attempt < self._settings.deepseek_max_retries
            try:
                response = self._client.post(self._url, json=payload)
            except httpx.TransportError:
                if can_retry:
                    self._sleep(0.5)
                    continue
                raise DeepSeekRequestError("DeepSeek连接失败或请求超时。") from None

            if not response.is_success:
                code = response.status_code
                if can_retry and (code == 429 or code >= 500):
                    self._sleep(0.5)
                    continue
                raise DeepSeekRequestError(
                    f"DeepSeek请求失败（HTTP {code}）。", status_code=code
                ) from None

            try:
                data = response.json()
            except ValueError:
                raise DeepSeekProtocolError("DeepSeek返回了无效JSON。") from None
            return self._parse_reply(data)

        raise DeepSeekRequestError("DeepSeek请求未完成。")

    def _parse_reply(self, data: Any) -> ChatReply:
        if not isinstance(data, dict):
            raise DeepSeekProtocolError("DeepSeek返回结构不完整。")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise DeepSeekProtocolError("DeepSeek未返回有效候选答复。")
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or finish_reason not in {"stop", "tool_calls"}:
            raise DeepSeekProtocolError("DeepSeek答复未完整生成，请检查输出长度或状态。")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise DeepSeekProtocolError("DeepSeek未返回有效助手消息。")
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if content is not None and not isinstance(content, str):
            raise DeepSeekProtocolError("DeepSeek消息正文不是文本。")
        if tool_calls is not None and not isinstance(tool_calls, list):
            raise DeepSeekProtocolError("DeepSeek工具调用结构无效。")
        if tool_calls:
            ids = set()
            for call in tool_calls:
                function = call.get("function") if isinstance(call, dict) else None
                if (
                    not isinstance(call, dict)
                    or call.get("type") != "function"
                    or not isinstance(call.get("id"), str)
                    or not call["id"]
                    or call["id"] in ids
                    or not isinstance(function, dict)
                    or not isinstance(function.get("name"), str)
                    or not function["name"]
                    or not isinstance(function.get("arguments"), str)
                ):
                    raise DeepSeekProtocolError("DeepSeek工具调用字段不完整。")
                ids.add(call["id"])
        elif not isinstance(content, str) or not content.strip():
            raise DeepSeekProtocolError("DeepSeek返回了空答复。")
        if finish_reason == "tool_calls" and not tool_calls:
            raise DeepSeekProtocolError("DeepSeek缺少工具调用内容。")

        model = data.get("model")
        usage = data.get("usage", {})
        if not isinstance(model, str) or not model or not isinstance(usage, dict):
            raise DeepSeekProtocolError("DeepSeek返回的模型元数据无效。")
        return ChatReply(
            message={k: v for k, v in message.items() if k in {"role", "content", "tool_calls", "reasoning_content"}},
            finish_reason=finish_reason,
            model=model,
            usage=usage,
        )
