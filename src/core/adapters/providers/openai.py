"""OpenAI Chat Completions/Responses 适配器。

Chat Completions 与既有 ``direct.openai_chat_sse`` 使用相同的 SSE 事件契约，
本层增加工具名称安全映射；旧类仍保留原始兼容行为供已有调用方使用。
"""

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
    AdapterProtocolError,
    ProviderAdapterError,
)
from core.adapters.direct.events import ToolCallDelta
from core.contracts.chat import ChatRequest
from core.events.types import ConversationContext, ReasoningDelta, TextDelta, TurnFinished

from .converters import (
    build_tool_name_map,
    openai_messages,
    openai_responses_input,
    provider_tool_definitions,
)
from .http import (
    classify_stream_exception,
    configured_headers,
    endpoint_url,
    iter_sse_json,
    join_base_path,
    normalize_stream_error,
    open_stream,
    response_status,
)


def _usage(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, Mapping):
            nested = _usage(raw)
            for nested_key, nested_value in nested.items():
                result[f"{key}.{nested_key}"] = nested_value
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if number >= 0:
            result[str(key)] = number
    aliases = {
        "prompt_tokens": "prompt_tokens",
        "completion_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
    }
    for source, target in aliases.items():
        if source in result:
            result[target] = result[source]
    if "prompt_tokens" in result and "completion_tokens" in result:
        result.setdefault("total_tokens", result["prompt_tokens"] + result["completion_tokens"])
    return result


def _provider_error_details(value: object) -> tuple[str, str, bool]:
    """把供应商错误码映射为不泄露正文的错误类别和提示。"""

    if isinstance(value, Mapping):
        code = str(value.get("code", value.get("type", "server_error")) or "server_error")
    else:
        code = "server_error"
    lowered = code.lower()
    if any(token in lowered for token in ("auth", "api_key", "apikey", "credential", "key")):
        return "authentication", "provider authentication failed; check the API key", False
    if any(token in lowered for token in ("permission", "forbidden", "access_denied")):
        return "authorization", "provider authorization failed; check API permissions", False
    if "rate" in lowered or "limit" in lowered:
        return "rate_limit", "provider rate limit reached", True
    return "server", "provider returned an error payload", True


def _omit_temperature(request: ChatRequest, channel: object) -> bool:
    """返回渠道或单次请求是否显式禁止发送采样 temperature。"""

    request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
    channel_metadata = getattr(channel, "metadata", {}) or {}
    return request_metadata.get("omit_temperature") is True or (
        isinstance(channel_metadata, Mapping) and channel_metadata.get("omit_temperature") is True
    )


class OpenAIProviderAdapter(ProviderAdapter):
    """OpenAI Chat Completions 兼容接口。"""

    provider = "openai"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming", "tools", "vision", "reasoning"})

    def __init__(self, channel: Any, *, client: Any | None = None) -> None:
        self.channel = channel
        self._owns_client = client is None
        if client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise AdapterConfigurationError(
                    "httpx is required for OpenAIProviderAdapter"
                ) from exc
            # SSE 连接的读取超时由渠道 timeout_seconds 控制；httpx 默认
            # 5 秒 read timeout 会截断模型思考较久但仍有效的流。
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
        return endpoint_url(self.channel, default_path="/chat/completions", endpoint_name="chat")

    def _headers(self) -> dict[str, str]:
        key = str(getattr(self.channel, "api_key", "") or "").strip()
        return configured_headers(self.channel, authorization=f"Bearer {key}" if key else None)

    def build_payload(self, request: ChatRequest) -> tuple[dict[str, Any], object]:
        names = build_tool_name_map(request.tools)
        payload: dict[str, Any] = dict(getattr(self.channel, "request_body_template", {}) or {})
        payload.update(
            {
                "model": request.model,
                "messages": openai_messages(request, names),
                "max_tokens": request.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        omit_temperature = _omit_temperature(request, self.channel)
        if not omit_temperature:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = provider_tool_definitions(request, names, "openai")
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        extra_body = (
            request.metadata.get("extra_body") if isinstance(request.metadata, Mapping) else None
        )
        if isinstance(extra_body, Mapping):
            payload.update(
                {
                    str(key): value
                    for key, value in extra_body.items()
                    if str(key) not in {"model", "messages", "stream"}
                }
            )
        if omit_temperature:
            # GPT-5/Codex 风格的 Chat 端点会在收到 sampling 参数时返回
            # 上游解析错误；``omit_temperature`` 是显式配置，不根据模型名猜测。
            payload.pop("temperature", None)
        return payload, names

    @staticmethod
    def _events_from(
        value: Mapping[str, Any], context: ConversationContext, names: object, state: dict[str, Any]
    ) -> tuple[ProviderEvent, ...]:
        if isinstance(value.get("error"), Mapping):
            category, message, retryable = _provider_error_details(value["error"])
            raise ProviderAdapterError(
                message,
                category=category,
                retryable=retryable,
            )
        events: list[ProviderEvent] = []
        response_model = value.get("model")
        if not isinstance(response_model, str):
            nested_response = value.get("response")
            if isinstance(nested_response, Mapping):
                response_model = nested_response.get("model")
        if isinstance(response_model, str) and response_model.strip():
            state["response_model"] = response_model.strip()
        usage = _usage(value.get("usage"))
        if usage:
            state.setdefault("usage", {}).update(usage)
        choices = value.get("choices", ())
        if not isinstance(choices, list):
            choices = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            if choice.get("finish_reason"):
                state["finish_reason"] = str(choice["finish_reason"])
            delta = choice.get("delta", choice.get("message", {}))
            if not isinstance(delta, Mapping):
                continue
            text = delta.get("content")
            if isinstance(text, str) and text:
                events.append(TextDelta(context=context, delta=text))
            elif isinstance(text, list):
                for part in text:
                    if (
                        isinstance(part, Mapping)
                        and isinstance(part.get("text"), str)
                        and part["text"]
                    ):
                        events.append(TextDelta(context=context, delta=str(part["text"])))
            for key in ("reasoning_content", "reasoning", "thinking", "analysis"):
                reasoning = delta.get(key)
                if isinstance(reasoning, str) and reasoning:
                    events.append(ReasoningDelta(context=context, delta=reasoning))
            murmur = delta.get("murmur")
            if isinstance(murmur, str) and murmur:
                from core.events.types import MurmurDelta

                events.append(MurmurDelta(context=context, delta=murmur))
            chunks = delta.get("tool_calls", ())
            if isinstance(chunks, Mapping):
                chunks = [chunks]
            if isinstance(chunks, list):
                for fallback, raw in enumerate(chunks):
                    if not isinstance(raw, Mapping):
                        continue
                    try:
                        index = int(raw.get("index", fallback))
                    except (TypeError, ValueError):
                        index = fallback
                    function = raw.get("function", {})
                    if not isinstance(function, Mapping):
                        function = {}
                    provider_name = str(function.get("name", raw.get("name", "")) or "").strip()
                    identity = (
                        names.core(provider_name)
                        if provider_name and hasattr(names, "core")
                        else provider_name
                    )
                    call_id = str(raw.get("id", raw.get("call_id", "")) or "").strip()
                    if call_id:
                        state.setdefault("call_ids", {})[index] = call_id
                    else:
                        call_id = (
                            str(state.setdefault("call_ids", {}).get(index, ""))
                            or f"openai-{secrets.token_hex(8)}"
                        )
                        state["call_ids"][index] = call_id
                    arguments = function.get("arguments", raw.get("arguments", ""))
                    if isinstance(arguments, Mapping):
                        arguments = json.dumps(
                            dict(arguments),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    events.append(
                        ToolCallDelta(
                            context=context,
                            index=index,
                            call_id=call_id,
                            identity=identity,
                            arguments_delta=str(arguments or ""),
                        )
                    )
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
        kwargs: dict[str, Any] = {"headers": self._headers(), "json": payload}
        timeout = float(getattr(self.channel, "timeout_seconds", 0.0) or 0.0)
        if timeout > 0:
            kwargs["timeout"] = timeout
        state: dict[str, Any] = {"usage": {}, "event_emitted": False, "call_ids": {}}
        try:
            stream_cm = await open_stream(self._client, "POST", self._url(), **kwargs)
            async with stream_cm as response:
                if cancellation.is_set():
                    raise AdapterCancelled()
                response_status(response)
                async for _event_name, value in iter_sse_json(response):
                    if cancellation.is_set():
                        raise AdapterCancelled()
                    for event in self._events_from(value, current_context, names, state):
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


class OpenAIResponsesAdapter(OpenAIProviderAdapter):
    """OpenAI Responses API 的最小流式适配器。"""

    protocol = "openai_responses"

    def _url(self) -> str:
        configured = str(getattr(self.channel, "endpoints", {}).get("responses", "") or "").strip()
        if configured:
            base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
            return (
                configured
                if configured.lower().startswith(("http://", "https://"))
                else join_base_path(base, configured)
            )
        endpoint = str(getattr(self.channel, "endpoint", "") or "").strip()
        if endpoint and endpoint != "/chat/completions":
            base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
            return join_base_path(base, endpoint)
        base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
        if not base:
            raise AdapterConfigurationError("channel base_url is required")
        return join_base_path(base, "/responses")

    def build_payload(self, request: ChatRequest) -> tuple[dict[str, Any], object]:
        names = build_tool_name_map(request.tools)
        tools: list[dict[str, Any]] = []
        for item in provider_tool_definitions(request, names, "openai"):
            function = item.get("function", {})
            if isinstance(function, Mapping):
                tools.append(
                    {
                        "type": "function",
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "parameters": function.get("parameters", {}),
                    }
                )
        payload: dict[str, Any] = dict(getattr(self.channel, "request_body_template", {}) or {})
        payload.update(
            {
                "model": request.model,
                "input": openai_responses_input(request, names),
                "stream": True,
                "max_output_tokens": request.max_tokens,
            }
        )
        omit_temperature = _omit_temperature(request, self.channel)
        if request.temperature is not None and not omit_temperature:
            payload["temperature"] = request.temperature
        if tools:
            payload["tools"] = tools
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        extra_body = (
            request.metadata.get("extra_body") if isinstance(request.metadata, Mapping) else None
        )
        if isinstance(extra_body, Mapping):
            payload.update(
                {
                    str(key): value
                    for key, value in extra_body.items()
                    if str(key) not in {"model", "input", "stream"}
                }
            )
        if omit_temperature:
            # ``request_body_template`` 和 ``extra_body`` 都可能预填 temperature；
            # 显式禁止采样时 Responses 与 Chat 必须一致地移除该字段。
            payload.pop("temperature", None)
        return payload, names

    @staticmethod
    def _events_from(
        value: Mapping[str, Any], context: ConversationContext, names: object, state: dict[str, Any]
    ) -> tuple[ProviderEvent, ...]:
        event_type = str(value.get("type", "") or "")
        events: list[ProviderEvent] = []
        response_model = value.get("model")
        if not isinstance(response_model, str):
            nested_response = value.get("response")
            if isinstance(nested_response, Mapping):
                response_model = nested_response.get("model")
        if isinstance(response_model, str) and response_model.strip():
            state["response_model"] = response_model.strip()

        def event_index(raw: object) -> int:
            try:
                index = int(raw or 0)
            except (TypeError, ValueError) as exc:
                raise AdapterProtocolError(
                    "provider response output index is invalid",
                    before_first_event=not state.get("event_emitted", False),
                ) from exc
            if index < 0:
                raise AdapterProtocolError(
                    "provider response output index is invalid",
                    before_first_event=not state.get("event_emitted", False),
                )
            return index

        items_by_id: dict[str, dict[str, Any]] = state.setdefault("items_by_id", {})
        items_by_index: dict[int, dict[str, Any]] = state.setdefault("items_by_index", {})
        arguments_by_key: dict[str, str] = state.setdefault("arguments", {})

        def remember(
            index: int,
            *,
            item_id: str = "",
            raw_name: str = "",
            call_id: str = "",
        ) -> tuple[str, str, str]:
            prior = items_by_id.get(item_id, {}) if item_id else items_by_index.get(index, {})
            name = raw_name or str(prior.get("name", "") or "")
            resolved_call_id = call_id or str(prior.get("call_id", "") or "")
            resolved_item_id = item_id or str(prior.get("item_id", "") or "")
            item = {"item_id": resolved_item_id, "name": name, "call_id": resolved_call_id}
            items_by_index[index] = item
            if resolved_item_id:
                items_by_id[resolved_item_id] = item
            return resolved_item_id, name, resolved_call_id

        def emit_call(
            index: int,
            *,
            item_id: str = "",
            raw_name: str = "",
            call_id: str = "",
            argument_delta: str = "",
            complete_arguments: str | None = None,
        ) -> None:
            resolved_item_id, resolved_name, resolved_call_id = remember(
                index, item_id=item_id, raw_name=raw_name, call_id=call_id
            )
            if not resolved_call_id:
                resolved_call_id = resolved_item_id or f"responses-{secrets.token_hex(8)}"
                items_by_index[index]["call_id"] = resolved_call_id
                if resolved_item_id:
                    items_by_id[resolved_item_id]["call_id"] = resolved_call_id
            key = resolved_item_id or f"index:{index}"
            if complete_arguments is not None:
                previous = arguments_by_key.get(key, "")
                if complete_arguments == previous:
                    argument_delta = ""
                elif previous and complete_arguments.startswith(previous):
                    argument_delta = complete_arguments[len(previous) :]
                else:
                    argument_delta = complete_arguments
                arguments_by_key[key] = (
                    complete_arguments
                    if not previous or complete_arguments.startswith(previous)
                    else previous + complete_arguments
                )
            elif argument_delta:
                arguments_by_key[key] = arguments_by_key.get(key, "") + argument_delta
            identity = (
                names.core(resolved_name)
                if resolved_name and hasattr(names, "core")
                else resolved_name
            )
            events.append(
                ToolCallDelta(
                    context=context,
                    index=index,
                    call_id=resolved_call_id,
                    identity=identity,
                    arguments_delta=argument_delta,
                )
            )

        def raise_response_error(error_value: object) -> None:
            category, message, retryable = _provider_error_details(error_value)
            raise ProviderAdapterError(
                message,
                category=category,
                retryable=retryable,
            )

        if event_type in {"response.output_text.delta", "response.text.delta"}:
            text = value.get("delta", value.get("text", ""))
            if isinstance(text, str) and text:
                events.append(TextDelta(context=context, delta=text))
        elif event_type in {"response.reasoning_summary_text.delta", "response.reasoning.delta"}:
            text = value.get("delta", value.get("text", ""))
            if isinstance(text, str) and text:
                events.append(ReasoningDelta(context=context, delta=text))
        elif event_type in {
            "response.function_call_arguments.delta",
            "response.custom_tool_call_input.delta",
        }:
            item_id = str(value.get("item_id", "") or "").strip()
            index = event_index(value.get("output_index", value.get("index", 0)))
            call_id = str(value.get("call_id", "") or "").strip()
            raw_name = str(value.get("name", "") or "").strip()
            delta = value.get("delta", "")
            emit_call(
                index,
                item_id=item_id,
                raw_name=raw_name,
                call_id=call_id,
                argument_delta=str(delta or ""),
            )
        elif event_type in {
            "response.function_call_arguments.done",
            "response.custom_tool_call_input.done",
        }:
            item_id = str(value.get("item_id", "") or "").strip()
            index = event_index(value.get("output_index", value.get("index", 0)))
            call_id = str(value.get("call_id", "") or "").strip()
            raw_name = str(value.get("name", "") or "").strip()
            complete = value.get("arguments", value.get("input", ""))
            emit_call(
                index,
                item_id=item_id,
                raw_name=raw_name,
                call_id=call_id,
                complete_arguments=str(complete or ""),
            )
        elif event_type == "response.output_item.added":
            item = value.get("item")
            if isinstance(item, Mapping) and str(item.get("type", "")) in {
                "function_call",
                "custom_tool_call",
            }:
                index = event_index(value.get("output_index", value.get("index", 0)))
                item_id = str(item.get("id", "") or "").strip()
                raw_name = str(item.get("name", "") or "").strip()
                call_id = str(item.get("call_id", "") or "").strip()
                initial = item.get("arguments", item.get("input", ""))
                emit_call(
                    index,
                    item_id=item_id,
                    raw_name=raw_name,
                    call_id=call_id,
                    complete_arguments=str(initial or ""),
                )
        elif event_type == "response.output_item.done":
            item = value.get("item")
            if isinstance(item, Mapping) and str(item.get("type", "")) in {
                "function_call",
                "custom_tool_call",
            }:
                index = event_index(value.get("output_index", value.get("index", 0)))
                item_id = str(item.get("id", "") or "").strip()
                raw_name = str(item.get("name", "") or "").strip()
                call_id = str(item.get("call_id", "") or "").strip()
                complete = item.get("arguments", item.get("input", ""))
                emit_call(
                    index,
                    item_id=item_id,
                    raw_name=raw_name,
                    call_id=call_id,
                    complete_arguments=str(complete or ""),
                )
        elif event_type in {"response.completed", "response.done"}:
            response = value.get("response", value)
            if isinstance(response, Mapping):
                response_model = response.get("model")
                if isinstance(response_model, str) and response_model.strip():
                    state["response_model"] = response_model.strip()
                if response.get("error") or str(response.get("status", "")).lower() == "failed":
                    raise_response_error(response.get("error"))
                state["finish_reason"] = str(response.get("status", "completed"))
                usage = _usage(response.get("usage"))
                if usage:
                    state.setdefault("usage", {}).update(usage)
        elif event_type == "error" or event_type == "response.failed":
            response = value.get("response")
            error_value = value.get("error")
            if isinstance(response, Mapping) and response.get("error"):
                error_value = response.get("error")
            raise_response_error(error_value or value)
        elif value.get("error"):
            raise_response_error(value.get("error"))
        if events:
            state["event_emitted"] = True
        return tuple(events)


OpenAIResponsesSSEAdapter = OpenAIResponsesAdapter

__all__ = ["OpenAIProviderAdapter", "OpenAIResponsesAdapter", "OpenAIResponsesSSEAdapter"]
