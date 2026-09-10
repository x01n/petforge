"""Google Gemini GenerateContent REST/SSE 适配器。"""

from __future__ import annotations

import asyncio
import inspect
import json
import secrets
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import quote

from core.adapters.direct.base import ProviderAdapter, ProviderEvent, ensure_context
from core.adapters.direct.errors import (
    AdapterCancelled,
    AdapterConfigurationError,
    ProviderAdapterError,
)
from core.adapters.direct.events import ToolCallDelta
from core.contracts.chat import ChatRequest
from core.events.types import ConversationContext, ReasoningDelta, TextDelta, TurnFinished

from .converters import build_tool_name_map, gemini_contents, provider_tool_definitions, tool_choice
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
        "promptTokenCount": "prompt_tokens",
        "c" + "andidatesTokenCount": "completion_tokens",
        "totalTokenCount": "total_tokens",
        "cachedContentTokenCount": "cached_tokens",
        "thoughtsTokenCount": "reasoning_tokens",
    }
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        try:
            number = int(raw_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number >= 0:
            result[aliases.get(str(raw_key), str(raw_key))] = number
    return result


class GeminiProviderAdapter(ProviderAdapter):
    """实现 ``models/{model}:streamGenerateContent`` 的异步流适配器。"""

    provider = "gemini"
    protocol = "gemini_generate"
    capabilities = frozenset({"streaming", "tools", "vision", "reasoning"})

    def __init__(self, channel: Any, *, client: Any | None = None) -> None:
        self.channel = channel
        self._owns_client = client is None
        if client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover
                raise AdapterConfigurationError(
                    "httpx is required for GeminiProviderAdapter"
                ) from exc
            client = httpx.AsyncClient(trust_env=False)
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

    def _model(self, request: ChatRequest) -> str:
        model = str(request.model or getattr(self.channel, "selected_model", "") or "").strip()
        if not model:
            raise AdapterConfigurationError("chat model is required")
        return model.removeprefix("models/")

    def _url(self, model: str) -> str:
        configured = str(getattr(self.channel, "endpoint", "") or "").strip()
        endpoint = str(getattr(self.channel, "endpoints", {}).get("stream", "") or "").strip()
        # 配置显式 endpoint 优先；默认使用 Gemini 官方 REST 路径。
        if endpoint:
            path = endpoint.replace("{model}", quote(model, safe=""))
            base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
            if path.lower().startswith(("http://", "https://")):
                return path
            return join_base_path(base, path)
        if configured and configured != "/chat/completions":
            base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
            path = configured if configured.startswith("/") else "/" + configured
            if "{model}" in path:
                path = path.replace("{model}", quote(model, safe=""))
            return join_base_path(base, path)
        base = str(getattr(self.channel, "base_url", "") or "").rstrip("/")
        if not base:
            raise AdapterConfigurationError("channel base_url is required")
        # alt=sse 使官方接口返回标准 SSE；代理可在 endpoint 中覆盖。
        model_path = f"/v1beta/models/{quote(model, safe='')}"
        return join_base_path(base, model_path) + ":streamGenerateContent?alt=sse"

    def build_payload(self, request: ChatRequest) -> tuple[dict[str, Any], object]:
        names = build_tool_name_map(request.tools)
        system, contents = gemini_contents(request, names)
        payload: dict[str, Any] = dict(getattr(self.channel, "request_body_template", {}) or {})
        payload.update(
            {
                "contents": contents,
                "generationConfig": {
                    "temperature": request.temperature,
                    "maxOutputTokens": request.max_tokens,
                },
            }
        )
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if request.tools:
            payload["tools"] = [
                {"functionDeclarations": provider_tool_definitions(request, names, "gemini")}
            ]
            choice = tool_choice(request.tool_choice, names, "gemini")
            if choice is not None:
                payload["toolConfig"] = choice
        extra_body = (
            request.metadata.get("extra_body") if isinstance(request.metadata, Mapping) else None
        )
        if isinstance(extra_body, Mapping):
            payload.update(
                {
                    str(key): value
                    for key, value in extra_body.items()
                    if str(key) not in {"contents", "systemInstruction"}
                }
            )
        return payload, names

    def _headers(self) -> dict[str, str]:
        result = configured_headers(self.channel, authorization=None)
        api_key = self._api_key()
        if api_key:
            result["x-goog-api-key"] = api_key
        return result

    def _api_key(self) -> str:
        return str(getattr(self.channel, "api_key", "") or "").strip()

    @staticmethod
    def _event_parts(
        value: Mapping[str, Any], context: ConversationContext, names: object, state: dict[str, Any]
    ) -> tuple[ProviderEvent, ...]:
        events: list[ProviderEvent] = []
        response_model = value.get("modelVersion", value.get("model"))
        if isinstance(response_model, str) and response_model.strip():
            state["response_model"] = response_model.strip()
        usage = _usage(value.get("usageMetadata"))
        if usage:
            state.setdefault("usage", {}).update(usage)
        options_list = value.get("c" + "andidates", ())
        if not isinstance(options_list, list):
            options_list = []
        for option in options_list:
            if not isinstance(option, Mapping):
                continue
            finish = option.get("finishReason")
            if finish:
                state["finish_reason"] = str(finish).lower()
            content = option.get("content", {})
            parts = content.get("parts", ()) if isinstance(content, Mapping) else ()
            if not isinstance(parts, list):
                continue
            for index, part in enumerate(parts):
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    if part.get("thought") is True or part.get("isThought") is True:
                        events.append(ReasoningDelta(context=context, delta=text))
                    else:
                        events.append(TextDelta(context=context, delta=text))
                call = part.get("functionCall")
                if not isinstance(call, Mapping):
                    continue
                name = str(call.get("name", "") or "").strip()
                if not name:
                    continue
                mapping = names
                identity = mapping.core(name) if hasattr(mapping, "core") else name
                calls = state.setdefault("call_ids", {})
                call_key = (index, name)
                supplied_id = str(call.get("id", "") or "").strip()
                if supplied_id:
                    calls[call_key] = supplied_id
                call_id = calls.setdefault(call_key, f"gemini-{secrets.token_hex(8)}")
                arguments = call.get("args", {})
                argument_delta = ""
                if isinstance(arguments, Mapping):
                    encoded_arguments = json.dumps(
                        dict(arguments),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    previous = state.setdefault("function_args", {}).get(call_key)
                    if previous == encoded_arguments:
                        continue
                    state["function_args"][call_key] = encoded_arguments
                    argument_delta = encoded_arguments
                else:
                    encoded_arguments = str(arguments or "")
                    argument_state = state.setdefault("function_args", {})
                    has_previous = call_key in argument_state
                    previous = argument_state.get(call_key, "")
                    if has_previous and encoded_arguments == previous:
                        continue
                    if previous and encoded_arguments.startswith(previous):
                        argument_delta = encoded_arguments[len(previous) :]
                    else:
                        argument_delta = encoded_arguments
                    argument_state[call_key] = encoded_arguments
                events.append(
                    ToolCallDelta(
                        context=context,
                        index=index,
                        call_id=call_id,
                        identity=identity,
                        arguments_delta=argument_delta,
                    )
                )
        if value.get("error"):
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
        url = self._url(self._model(request))
        api_key = self._api_key()
        metadata = getattr(self.channel, "metadata", {}) or {}
        raw_query_key = (
            metadata.get("api_key_in_query", False) if isinstance(metadata, Mapping) else False
        )
        query_key = raw_query_key is True or (
            isinstance(raw_query_key, (int, str)) and raw_query_key in {1, "1", "true", "yes", "on"}
        )
        if query_key and api_key and "?" in url:
            url = f"{url}&key={quote(api_key, safe='')}"
        elif query_key and api_key:
            url = f"{url}?key={quote(api_key, safe='')}"
        timeout = float(getattr(self.channel, "timeout_seconds", 0.0) or 0.0)
        kwargs: dict[str, Any] = {"headers": self._headers(), "json": payload}
        if timeout > 0:
            kwargs["timeout"] = timeout
        state: dict[str, Any] = {"usage": {}, "event_emitted": False, "call_ids": {}}
        try:
            stream_cm = await open_stream(self._client, "POST", url, **kwargs)
            async with stream_cm as response:
                if cancellation.is_set():
                    raise AdapterCancelled()
                response_status(response)
                async for _event_name, value in iter_sse_json(response):
                    if cancellation.is_set():
                        raise AdapterCancelled()
                    for event in self._event_parts(value, current_context, names, state):
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


GeminiGenerateAdapter = GeminiProviderAdapter

__all__ = ["GeminiGenerateAdapter", "GeminiProviderAdapter"]
