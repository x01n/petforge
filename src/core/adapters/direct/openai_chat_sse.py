"""OpenAI Chat Completions 兼容接口的 SSE 适配器。"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from core.contracts.chat import ChatMessage, ChatRequest
from core.events.types import (
    ConversationContext,
    MurmurDelta,
    ReasoningDelta,
    TextDelta,
    TurnFinished,
)

from .base import ProviderAdapter, ProviderEvent, ensure_context
from .errors import (
    AdapterCancelled,
    AdapterConfigurationError,
    AdapterProtocolError,
    ProviderAdapterError,
)
from .events import ToolCallDelta

try:  # 允许在未安装可选依赖的静态检查环境中导入模块。
    import httpx
except ImportError:  # pragma: no cover - 运行时由构造函数给出明确错误
    httpx = None  # type: ignore[assignment]


_PROTECTED_HEADERS = frozenset({"authorization", "content-type", "accept"})

# 渠道未显式配置超时（timeout_seconds 为 0）时叠加的默认总时限。
# 0 在 ChannelConfig 中的语义是“由客户端决定”，而非零超时；若不叠加，
# httpx 无超时且 SSE 可能无限挂起，重试回退永不触发。
_DEFAULT_TOTAL_TIMEOUT_SECONDS = 60.0
# drain 队列单次读取的等待上限：正常情况下 token 会连续到达，只有在流
# 静默时才会命中，用于及时观察到取消事件而不引入新的总时限。
_DRAIN_TIMEOUT_SECONDS = 10.0
# 取消监控的轮询粒度；桥接的外部取消事件以该步长进入 drain 循环。
_CANCEL_POLL_GRACE_SECONDS = 0.05


def _as_jsonable_content(content: object) -> object:
    if isinstance(content, Mapping):
        content_type = str(content.get("type", "")).strip().lower()
        if content_type == "image" and content.get("media_type") and content.get("data"):
            media_type = str(content["media_type"])
            data = str(content["data"])
            return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
        return {str(key): _as_jsonable_content(value) for key, value in content.items()}
    if isinstance(content, (list, tuple)):
        return [_as_jsonable_content(item) for item in content]
    return content


def _message_payload(message: ChatMessage) -> dict[str, Any]:
    raw = message.as_mapping()
    payload: dict[str, Any] = {
        "role": raw["role"],
        "content": _as_jsonable_content(raw.get("content")),
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.identity,
                    "arguments": json.dumps(
                        dict(call.arguments), ensure_ascii=False, separators=(",", ":")
                    ),
                },
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id:
        payload["tool_call_id"] = message.tool_call_id
    if message.tool_name:
        payload["name"] = message.tool_name
    return payload


def _safe_int(value: object) -> int | None:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _usage_payload(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, raw in value.items():
        if isinstance(raw, Mapping):
            nested = _usage_payload(raw)
            for nested_key, nested_value in nested.items():
                result[f"{key}.{nested_key}"] = nested_value
            continue
        number = _safe_int(raw)
        if number is not None:
            result[str(key)] = number
    aliases = {
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
    }
    for source, target in aliases.items():
        if source in result:
            result[target] = result[source]
    if "prompt_tokens" in result and "completion_tokens" in result:
        result.setdefault("total_tokens", result["prompt_tokens"] + result["completion_tokens"])
    return result


class OpenAIChatSSEAdapter(ProviderAdapter):
    """实现 ``POST /chat/completions`` 的流式适配器。"""

    provider = "openai"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming", "tools", "vision", "reasoning"})

    def __init__(
        self,
        channel: Any,
        *,
        client: Any | None = None,
        endpoint: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.channel = channel
        self._owns_client = client is None
        if client is None:
            if httpx is None:
                raise AdapterConfigurationError("httpx is required for OpenAIChatSSEAdapter")
            # 模型渠道地址由配置明确给出；不继承 HTTP_PROXY/HTTPS_PROXY，
            # 防止本地回环端点或用户指定代理被透明改写。
            # 流式思考可能长时间没有新字节，httpx 层禁用默认 5 秒读取超时；
            # 总时限由 ProviderAdapterRuntime 的默认总时限常量兜底，渠道
            # 显式配置 timeout_seconds 时仍优先使用渠道值。
            client = httpx.AsyncClient(trust_env=False, timeout=None)
        self._client = client
        self._endpoint_override = str(endpoint or "").strip()
        self._headers_override = {
            str(key): str(value) for key, value in dict(headers or {}).items()
        }
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
        base_url = str(getattr(self.channel, "base_url", "") or "").strip().rstrip("/")
        endpoint = self._endpoint_override
        if not endpoint:
            endpoint_getter = getattr(self.channel, "endpoint_path", None)
            endpoint = (
                endpoint_getter("chat")
                if callable(endpoint_getter)
                else str(getattr(self.channel, "endpoint", "/chat/completions"))
            )
        endpoint = str(endpoint or "/chat/completions").strip()
        if not base_url:
            raise AdapterConfigurationError("channel base_url is required")
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            return endpoint
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        if base_url.endswith(endpoint.rstrip("/")):
            return base_url
        return base_url + endpoint

    def _headers(self) -> dict[str, str]:
        configured = getattr(self.channel, "headers", {})
        result = {str(key): str(value) for key, value in dict(configured or {}).items()}
        result.update(self._headers_override)
        # 认证和协议头由适配器掌控，渠道附加头不能覆盖它们。
        for key in tuple(result):
            if key.lower() in _PROTECTED_HEADERS:
                result.pop(key, None)
        result["Accept"] = "text/event-stream"
        result["Content-Type"] = "application/json"
        api_key = str(getattr(self.channel, "api_key", "") or "").strip()
        if api_key:
            result["Authorization"] = f"Bearer {api_key}"
        return result

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        if not isinstance(request, ChatRequest):
            raise TypeError("request must be a ChatRequest")
        model = str(request.model or getattr(self.channel, "selected_model", "") or "").strip()
        if not model:
            raise AdapterConfigurationError("chat model is required")
        payload: dict[str, Any] = dict(getattr(self.channel, "request_body_template", {}) or {})
        payload.update(
            {
                "model": model,
                "messages": [_message_payload(message) for message in request.messages],
                "max_tokens": request.max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        request_metadata = request.metadata if isinstance(request.metadata, Mapping) else {}
        channel_metadata = getattr(self.channel, "metadata", {}) or {}
        omit_temperature = request_metadata.get("omit_temperature") is True or (
            isinstance(channel_metadata, Mapping)
            and channel_metadata.get("omit_temperature") is True
        )
        if not omit_temperature:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [tool.as_openai_tool() for tool in request.tools]
            if request.tool_choice is not None:
                payload["tool_choice"] = request.tool_choice
        elif request.tool_choice == "none":
            payload["tool_choice"] = "none"
        extra_body = (
            request.metadata.get("extra_body") if isinstance(request.metadata, Mapping) else None
        )
        if isinstance(extra_body, Mapping):
            for key, value in extra_body.items():
                if str(key) not in {"model", "messages", "stream", "stream_options"}:
                    payload[str(key)] = value
        if omit_temperature:
            payload.pop("temperature", None)
        return payload

    async def _open_stream(self, request: ChatRequest) -> Any:
        timeout = getattr(self.channel, "timeout_seconds", None)
        kwargs: dict[str, Any] = {"headers": self._headers(), "json": self.build_payload(request)}
        if timeout is not None and float(timeout) > 0:
            kwargs["timeout"] = float(timeout)
        stream_cm = self._client.stream("POST", self._url(), **kwargs)
        if inspect.isawaitable(stream_cm):
            stream_cm = await stream_cm
        return stream_cm

    @staticmethod
    def _status_error(status_code: int, *, retry_after: str | None = None) -> ProviderAdapterError:
        if status_code in {401, 403}:
            category = "authentication" if status_code == 401 else "authorization"
            retryable = False
            message = (
                "provider authentication failed"
                if status_code == 401
                else "provider authorization failed"
            )
        elif status_code == 429:
            category = "rate_limit"
            retryable = True
            message = "provider rate limit exceeded"
        elif status_code in {408, 425}:
            category = "timeout"
            retryable = True
            message = "provider request timed out"
        elif status_code >= 500:
            category = "server"
            retryable = True
            message = "provider service is unavailable"
        else:
            category = "protocol"
            retryable = False
            message = f"provider returned HTTP {status_code}"
        error = ProviderAdapterError(
            message,
            category=category,  # type: ignore[arg-type]
            status_code=status_code,
            retryable=retryable,
        )
        if retry_after:
            try:
                error.retry_after_seconds = max(0.0, float(retry_after))  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                pass
        return error

    @staticmethod
    async def _iter_payloads(response: Any) -> AsyncIterator[str]:
        buffer: list[str] = []

        async def flush() -> AsyncIterator[str]:
            if not buffer:
                return
            payload = "\n".join(buffer).strip()
            buffer.clear()
            if payload:
                yield payload

        async for raw_line in response.aiter_lines():
            line = str(raw_line)
            if not line:
                async for payload in flush():
                    yield payload
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                buffer.append(line[5:].lstrip())
        async for payload in flush():
            yield payload

    @staticmethod
    def _events_from_payload(
        payload: str,
        context: ConversationContext,
        state: dict[str, Any],
    ) -> tuple[ProviderEvent, ...]:
        if payload == "[DONE]":
            if state.get("done"):
                return ()
            state["done"] = True
            return (
                TurnFinished(
                    context=context,
                    finish_reason=str(state.get("finish_reason") or "stop"),
                    usage=dict(state.get("usage") or {}),
                    response_model=str(state.get("response_model") or ""),
                ),
            )
        try:
            value = json.loads(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AdapterProtocolError(
                "provider SSE payload is not valid JSON",
                before_first_event=not state.get("event_emitted", False),
            ) from exc
        if not isinstance(value, Mapping):
            raise AdapterProtocolError(
                "provider SSE payload must be an object",
                before_first_event=not state.get("event_emitted", False),
            )
        response_model = value.get("model")
        if isinstance(response_model, str) and response_model.strip():
            state["response_model"] = response_model.strip()
        if isinstance(value.get("error"), Mapping):
            raise ProviderAdapterError(
                "provider returned an error payload", category="server", retryable=False
            )
        events: list[ProviderEvent] = []
        usage = _usage_payload(value.get("usage"))
        if usage:
            state.setdefault("usage", {}).update(usage)
        choices = value.get("choices", ())
        if not isinstance(choices, list):
            choices = []
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            reason = choice.get("finish_reason")
            if reason:
                state["finish_reason"] = str(reason)
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
                events.append(MurmurDelta(context=context, delta=murmur))
            tool_chunks = delta.get("tool_calls", ())
            if isinstance(tool_chunks, Mapping):
                tool_chunks = [tool_chunks]
            if isinstance(tool_chunks, list):
                for fallback_index, raw_tool in enumerate(tool_chunks):
                    if not isinstance(raw_tool, Mapping):
                        continue
                    index = raw_tool.get("index", fallback_index)
                    try:
                        index = int(index)
                    except (TypeError, ValueError):
                        index = fallback_index
                    function = raw_tool.get("function", raw_tool.get("tool", {}))
                    if not isinstance(function, Mapping):
                        function = {}
                    call_id = str(raw_tool.get("id", raw_tool.get("call_id", "")) or "")
                    identity = str(function.get("name", raw_tool.get("name", "")) or "")
                    arguments = function.get("arguments", raw_tool.get("arguments", ""))
                    if isinstance(arguments, Mapping):
                        arguments = json.dumps(
                            dict(arguments), ensure_ascii=False, separators=(",", ":")
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

    async def _drain_under_timeout(
        self,
        request: ChatRequest,
        current_context: ConversationContext,
        cancellation: asyncio.Event,
        state: dict[str, Any],
        *,
        timeout_seconds: float,
    ) -> AsyncIterator[ProviderEvent]:
        """在总时限与取消监控下消费单个流响应，超时映射为可重试错误。"""

        async def inner() -> AsyncIterator[ProviderEvent]:
            stream_cm = await self._open_stream(request)
            async with stream_cm as response:
                if cancellation.is_set():
                    raise AdapterCancelled()
                status_code = int(getattr(response, "status_code", 200))
                if status_code < 200 or status_code >= 300:
                    headers = getattr(response, "headers", {}) or {}
                    raise self._status_error(status_code, retry_after=headers.get("retry-after"))
                async for payload in self._iter_payloads(response):
                    if cancellation.is_set():
                        raise AdapterCancelled()
                    for event in self._events_from_payload(payload, current_context, state):
                        yield event
            if not state.get("done"):
                state["done"] = True
                yield TurnFinished(
                    context=current_context,
                    finish_reason=str(state.get("finish_reason") or "stop"),
                    usage=dict(state.get("usage") or {}),
                    response_model=str(state.get("response_model") or ""),
                )

        # drain 任务与消费者解耦：总时限到期时 finally 取消 pump，
        # 取消能同步送达底层读取循环，把挂起的 httpx 读真正拆开。
        queue: asyncio.Queue[ProviderEvent | BaseException] = asyncio.Queue()
        drain_done = asyncio.Event()
        _SENTINEL: object = object()

        async def pump() -> None:
            try:
                async for event in inner():
                    await queue.put(event)
                await queue.put(_SENTINEL)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                await queue.put(exc)
            finally:
                drain_done.set()

        async def waiter() -> None:
            while not drain_done.is_set():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(drain_done.wait()),
                        timeout=_CANCEL_POLL_GRACE_SECONDS,
                    )
                    return
                except TimeoutError:
                    if cancellation.is_set():
                        pump_task.cancel()
                        return

        async def consume() -> AsyncIterator[ProviderEvent]:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=_DRAIN_TIMEOUT_SECONDS)
                except TimeoutError:
                    if drain_done.is_set():
                        break
                    continue
                if item is _SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item

        pump_task = asyncio.create_task(pump())
        waiter_task = asyncio.create_task(waiter())
        try:
            async with asyncio.timeout(timeout_seconds):
                async for event in consume():
                    yield event
        except TimeoutError as exc:
            raise ProviderAdapterError(
                "provider request timed out",
                category="timeout",
                retryable=True,
                before_first_event=not state.get("event_emitted", False),
                cause=exc,
            ) from exc
        finally:
            pump_task.cancel()
            waiter_task.cancel()

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
        state: dict[str, Any] = {"usage": {}, "done": False, "event_emitted": False}
        # 渠道 0/未配置时，总时限由 ProviderAdapterRuntime 的默认常量兜底；
        # 直接使用本适配器的宿主仍叠加本底线，避免流无限挂起。
        timeout = getattr(self.channel, "timeout_seconds", None)
        if timeout is not None and float(timeout) > 0:
            adjusted = self._drain_under_timeout(
                request,
                current_context,
                cancellation,
                state,
                timeout_seconds=float(timeout),
            )
        else:
            adjusted = self._drain_under_timeout(
                request,
                current_context,
                cancellation,
                state,
                timeout_seconds=_DEFAULT_TOTAL_TIMEOUT_SECONDS,
            )
        try:
            async for event in adjusted:
                yield event
        except asyncio.CancelledError:
            raise
        except (ProviderAdapterError, AdapterCancelled):
            raise
        except Exception as exc:
            if httpx is not None and isinstance(
                exc, (httpx.TimeoutException, httpx.TransportError)
            ):
                category = "timeout" if isinstance(exc, httpx.TimeoutException) else "network"
                raise ProviderAdapterError(
                    "provider request timed out"
                    if category == "timeout"
                    else "provider network request failed",
                    category=category,  # type: ignore[arg-type]
                    retryable=True,
                    before_first_event=not state.get("event_emitted", False),
                    cause=exc,
                ) from exc
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


OpenAIChatAdapter = OpenAIChatSSEAdapter
OpenAIChatSseAdapter = OpenAIChatSSEAdapter
OpenAIChatSSE = OpenAIChatSSEAdapter


__all__ = [
    "OpenAIChatAdapter",
    "OpenAIChatSSE",
    "OpenAIChatSSEAdapter",
    "OpenAIChatSseAdapter",
]
