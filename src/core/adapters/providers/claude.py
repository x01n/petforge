from __future__ import annotations

import asyncio
import inspect
import json
import secrets
from collections.abc import AsyncIterator, Mapping
from typing import Any

from core.adapters.direct.base import ProviderAdapter, ProviderEvent, ensure_context
from core.adapters.direct.errors import (
    AdapterCancelled,
    AdapterConfigurationError,
    ProviderAdapterError,
)
from core.adapters.direct.events import ToolCallDelta
from core.contracts.chat import ChatRequest
from core.events.types import ConversationContext, ReasoningDelta, TextDelta, TurnFinished

from .converters import build_tool_name_map, claude_messages, provider_tool_definitions, tool_choice
from .http import (
    classify_stream_exception,
    configured_headers,
    iter_sse_json,
    join_base_path,
    normalize_stream_error,
    open_stream,
    provider_error_details,
    response_status,
)


def _usage(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    aliases = {
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
        "cache_read_input_tokens": "cached_tokens",
        "cache_creation_input_tokens": "cache_creation_tokens",
    }
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        try:
            number = int(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number >= 0:
            result[aliases.get(str(raw_key), str(raw_key))] = number
    if "prompt_tokens" in result and "completion_tokens" in result:
        result.setdefault("total_tokens", result["prompt_tokens"] + result["completion_tokens"])
    return result


class ClaudeProviderAdapter(ProviderAdapter):
    provider = "claude"
    protocol = "anthropic_messages"
    capabilities = frozenset({"streaming", "tools", "vision", "reasoning"})

    def __init__(
        self, channel: Any, *, client: Any | None = None, anthropic_version: str = "2023-06-01"
    ) -> None:
        self.channel = channel
        self.anthropic_version = str(anthropic_version or "2023-06-01").strip()
        if not self.anthropic_version or any(char in self.anthropic_version for char in "\r\n\x00"):
            raise AdapterConfigurationError("anthropic_version is invalid")
        self._owns_client = client is None
        if client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise AdapterConfigurationError(
                    "httpx is required for ClaudeProviderAdapter"
                ) from exc
            client = httpx.AsyncClient(trust_env=False, timeout=None)
        self._client = client
        self._closed = False

    @property
    def channel_id(self) -> str:
        return str(getattr(self.channel, "id", "") or "")

    def supports(self, capability: str) -> bool:
        configured = getattr(self.channel, "capabilities", None)
        if configured:
            return str(capability or "").strip().lower() in {
                str(item).lower() for item in configured
            }
        return super().supports(capability)

    def _url(self) -> str:
        # Claude 的默认 endpoint 是 /v1/messages；ChannelConfig 的默认聊天路径
        # 仍为 OpenAI 兼容路径，因此仅在未显式配置时选择 Claude 路径。
        endpoint = str(getattr(self.channel, "endpoints", {}).get("messages", "") or "").strip()
        if endpoint:
            base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
            return (
                endpoint
                if endpoint.lower().startswith(("http://", "https://"))
                else join_base_path(base, endpoint)
            )
        configured = str(getattr(self.channel, "endpoint", "") or "").strip()
        if configured and configured != "/chat/completions":
            base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
            return join_base_path(base, configured)
        base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
        if not base:
            raise AdapterConfigurationError("channel base_url is required")
        return join_base_path(base, "/v1/messages")

    def _headers(self) -> dict[str, str]:
        result = configured_headers(self.channel, authorization=None)
        api_key = str(getattr(self.channel, "api_key", "") or "").strip()
        if api_key:
            # Anthropic 原生接口使用 ``x-api-key``，而 Claude Code/本地
            # 网关通过 ``ANTHROPIC_AUTH_TOKEN`` 约定使用 Bearer。两者
            # 可携带同一凭据，避免把环境变量名称映射成错误的认证方案；
            # 代理通常优先读取 Authorization，原生服务仍读取 x-api-key。
            result["x-api-key"] = api_key
            result["Authorization"] = f"Bearer {api_key}"
        result["anthropic-version"] = self.anthropic_version
        return result

    def build_payload(self, request: ChatRequest) -> tuple[dict[str, Any], object]:
        names = build_tool_name_map(request.tools)
        system, messages = claude_messages(request, names)
        payload: dict[str, Any] = dict(getattr(self.channel, "request_body_template", {}) or {})
        payload.update(
            {
                "model": request.model,
                "max_tokens": request.max_tokens,
                "messages": messages,
                "stream": True,
            }
        )
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = provider_tool_definitions(request, names, "claude")
            choice = tool_choice(request.tool_choice, names, "claude")
            if choice is not None:
                payload["tool_choice"] = choice
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        extra_body = (
            request.metadata.get("extra_body") if isinstance(request.metadata, Mapping) else None
        )
        if isinstance(extra_body, Mapping):
            payload.update(
                {
                    str(key): value
                    for key, value in extra_body.items()
                    if str(key) not in {"messages", "stream", "model"}
                }
            )
        return payload, names

    @staticmethod
    def _events_from(
        value: Mapping[str, Any],
        event_name: str,
        context: ConversationContext,
        names: object,
        state: dict[str, Any],
    ) -> tuple[ProviderEvent, ...]:
        events: list[ProviderEvent] = []
        event_type = str(value.get("type", event_name) or event_name)
        if event_type == "message_start":
            message = value.get("message")
            if isinstance(message, Mapping):
                response_model = message.get("model")
                if isinstance(response_model, str) and response_model.strip():
                    state["response_model"] = response_model.strip()
            usage = _usage(message.get("usage") if isinstance(message, Mapping) else {})
            if usage:
                state.setdefault("usage", {}).update(usage)
        elif event_type == "content_block_start":
            block = value.get("content_block")
            index = int(value.get("index", 0) or 0)
            if isinstance(block, Mapping) and block.get("type") == "tool_use":
                name = str(block.get("name", "") or "").strip()
                if name:
                    call_id = (
                        str(block.get("id", "") or "").strip() or f"claude-{secrets.token_hex(8)}"
                    )
                    state.setdefault("tool_ids", {})[index] = call_id
                    state.setdefault("tool_names", {})[index] = name
                    initial = block.get("input")
                    args = ""
                    if isinstance(initial, Mapping) and initial:
                        args = json.dumps(
                            dict(initial),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    identity = names.core(name) if hasattr(names, "core") else name
                    events.append(
                        ToolCallDelta(
                            context=context,
                            index=index,
                            call_id=call_id,
                            identity=identity,
                            arguments_delta=args,
                        )
                    )
        elif event_type == "content_block_delta":
            index = int(value.get("index", 0) or 0)
            delta = value.get("delta")
            if not isinstance(delta, Mapping):
                delta = {}
            delta_type = str(delta.get("type", "") or "")
            if delta_type == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    events.append(TextDelta(context=context, delta=text))
            elif delta_type in {"thinking_delta", "signature_delta"}:
                thinking = delta.get("thinking") or delta.get("text")
                if isinstance(thinking, str) and thinking:
                    events.append(ReasoningDelta(context=context, delta=thinking))
            elif delta_type == "input_json_delta":
                partial = delta.get("partial_json")
                if isinstance(partial, str) and partial:
                    call_id = str(state.setdefault("tool_ids", {}).get(index, ""))
                    raw_name = str(state.setdefault("tool_names", {}).get(index, ""))
                    identity = names.core(raw_name) if hasattr(names, "core") else raw_name
                    events.append(
                        ToolCallDelta(
                            context=context,
                            index=index,
                            call_id=call_id,
                            identity=identity,
                            arguments_delta=partial,
                        )
                    )
        elif event_type == "message_delta":
            delta = value.get("delta")
            if isinstance(delta, Mapping) and delta.get("stop_reason"):
                state["finish_reason"] = str(delta.get("stop_reason"))
            usage = _usage(value.get("usage"))
            if usage:
                state.setdefault("usage", {}).update(usage)
        elif event_type == "error" or value.get("error"):
            category, message, retryable = provider_error_details(value)
            raise ProviderAdapterError(message, category=category, retryable=retryable)
        if events:
            state["event_emitted"] = True
        return tuple(events)

    async def stream(
        self,
        request: ChatRequest,
        *,
        context: ConversationContext | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        if self._closed:
            raise ProviderAdapterError("adapter is closed", category="configuration")
        current_context = ensure_context(context)
        cancellation = cancel_event or asyncio.Event()
        if cancellation.is_set():
            raise AdapterCancelled()
        payload, names = self.build_payload(request)
        timeout = float(getattr(self.channel, "timeout_seconds", 0.0) or 0.0)
        kwargs: dict[str, Any] = {"headers": self._headers(), "json": payload}
        if timeout > 0:
            kwargs["timeout"] = timeout
        state: dict[str, Any] = {
            "usage": {},
            "event_emitted": False,
            "tool_ids": {},
            "tool_names": {},
        }
        try:
            stream_cm = await open_stream(self._client, "POST", self._url(), **kwargs)
            async with stream_cm as response:
                if cancellation.is_set():
                    raise AdapterCancelled()
                response_status(response)
                async for event_name, value in iter_sse_json(response):
                    if cancellation.is_set():
                        raise AdapterCancelled()
                    for event in self._events_from(
                        value, event_name, current_context, names, state
                    ):
                        yield event
            yield TurnFinished(
                context=current_context,
                finish_reason=str(state.get("finish_reason") or "stop"),
                usage=dict(state.get("usage") or {}),
                response_model=str(state.get("response_model") or ""),
            )
        except (asyncio.CancelledError, AdapterCancelled):
            raise
        except ProviderAdapterError as exc:
            normalized = normalize_stream_error(
                exc, before_first_event=not state.get("event_emitted", False)
            )
            if normalized is exc:
                raise
            raise normalized from exc
        except Exception as exc:
            classified = classify_stream_exception(
                exc, before_first_event=not state.get("event_emitted", False)
            )
            if classified.category in {"timeout", "network"}:
                raise classified from exc
            raise ProviderAdapterError(
                "provider request failed",
                category="unknown",
                retryable=False,
                before_first_event=not state.get("event_emitted", False),
                cause=exc,
            ) from exc

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            close = getattr(self._client, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result


ClaudeMessagesAdapter = ClaudeProviderAdapter

__all__ = ["ClaudeMessagesAdapter", "ClaudeProviderAdapter"]
