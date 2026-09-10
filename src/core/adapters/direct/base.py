from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.contracts.chat import ChatRequest, ToolCall
from core.events.types import (
    ConversationContext,
    MurmurDelta,
    ReasoningDelta,
    TextDelta,
    TurnFinished,
)

from .errors import AdapterCancelled, ProviderAdapterError, classify_exception
from .events import AdapterMetadata, ToolCallDelta
from .retry import RetryPolicy

ProviderEvent = (
    TextDelta | ReasoningDelta | MurmurDelta | ToolCallDelta | TurnFinished | AdapterMetadata
)
EventSink = Callable[[ProviderEvent], Awaitable[None] | None]


def ensure_context(context: ConversationContext | Mapping[str, Any] | None) -> ConversationContext:
    """确保适配器事件始终带有合法的会话上下文。"""

    if isinstance(context, ConversationContext):
        return context
    if isinstance(context, Mapping):
        return ConversationContext(
            profile_id=str(context.get("profile_id", "default")),
            session_id=str(context.get("session_id", "local")),
            turn_id=str(context.get("turn_id", "adapter-turn")),
            generation_id=int(context.get("generation_id", 0)),
            mode=str(context.get("mode", "direct")),
        )
    return ConversationContext(
        profile_id="default",
        session_id="local",
        turn_id=f"adapter-{uuid.uuid4().hex}",
        generation_id=0,
        mode="direct",
    )


class ProviderAdapter(ABC):
    """供应商协议适配器的最小接口。"""

    provider: str = "unknown"
    protocol: str = "unknown"
    capabilities: frozenset[str] = frozenset({"streaming"})

    @abstractmethod
    def stream(
        self,
        request: ChatRequest,
        *,
        context: ConversationContext | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """把一次请求转换成统一事件流。"""

    async def complete(
        self,
        request: ChatRequest,
        *,
        context: ConversationContext | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ProviderResponse:
        """消费完整流并返回聚合结果。"""

        response = ProviderResponse()
        async for event in self.stream(request, context=context, cancel_event=cancel_event):
            response = response.add(event)
        return response

    def supports(self, capability: str) -> bool:
        return str(capability or "").strip().lower() in self.capabilities

    def diagnostics(self) -> Mapping[str, object]:
        """返回可公开的协议与能力信息，不包含渠道配置。"""

        effective = tuple(sorted(item for item in self.capabilities if self.supports(item)))
        return {
            "provider": str(self.provider or "unknown")[:64],
            "protocol": str(self.protocol or "unknown")[:64],
            "capabilities": effective,
        }

    async def aclose(self) -> None:
        """释放适配器持有的网络资源。"""

    async def __aenter__(self) -> ProviderAdapter:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


class ProviderAdapterLike(Protocol):
    provider: str
    protocol: str
    capabilities: frozenset[str]

    def stream(
        self,
        request: ChatRequest,
        *,
        context: ConversationContext | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[ProviderEvent]: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True)
class ProviderResponse:
    """跨供应商的完整响应。"""

    text: str = ""
    reasoning: str = ""
    murmur: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    raw_tool_calls: tuple[Mapping[str, Any], ...] = ()
    usage: Mapping[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def add(self, event: ProviderEvent) -> ProviderResponse:
        text = self.text
        reasoning = self.reasoning
        murmur = self.murmur
        raw_calls = list(self.raw_tool_calls)
        calls = list(self.tool_calls)
        usage = dict(self.usage)
        finish_reason = self.finish_reason
        metadata = dict(self.metadata)
        if isinstance(event, TextDelta):
            text += event.delta
        elif isinstance(event, ReasoningDelta):
            reasoning += event.delta
        elif isinstance(event, MurmurDelta):
            murmur += event.delta
        elif isinstance(event, ToolCallDelta):
            # 以 index 作为稳定槽位；参数仍保持原始字符串，直到结束时才解析。
            while len(raw_calls) <= event.index:
                raw_calls.append(
                    {"index": len(raw_calls), "call_id": "", "identity": "", "arguments": ""}
                )
            item = dict(raw_calls[event.index])
            if event.call_id:
                item["call_id"] = event.call_id
            if event.identity:
                item["identity"] = event.identity
            item["arguments"] = str(item.get("arguments", "")) + event.arguments_delta
            raw_calls[event.index] = item
        elif isinstance(event, TurnFinished):
            usage.update(event.usage)
            finish_reason = event.finish_reason
            if event.response_model:
                metadata["response_model"] = event.response_model
            metadata.update(dict(event.metadata))
        elif isinstance(event, AdapterMetadata):
            metadata.update(event.values)
        return ProviderResponse(
            text=text,
            reasoning=reasoning,
            murmur=murmur,
            tool_calls=tuple(calls),
            raw_tool_calls=tuple(raw_calls),
            usage=usage,
            finish_reason=finish_reason,
            metadata=metadata,
        )

    def finalized(self) -> ProviderResponse:
        """把原始工具调用参数解析为核心 ``ToolCall``。"""

        calls: list[ToolCall] = []
        invalid_calls: list[dict[str, str]] = []
        for item in self.raw_tool_calls:
            call_id = str(item.get("call_id") or "").strip()
            identity = str(item.get("identity") or "").strip()
            if not call_id or not identity or ":" not in identity:
                continue
            raw_arguments = item.get("arguments", "")
            if isinstance(raw_arguments, Mapping):
                arguments = dict(raw_arguments)
            else:
                try:
                    parsed = json.loads(str(raw_arguments or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    invalid_calls.append(
                        {
                            "call_id": call_id,
                            "identity": identity,
                        }
                    )
                    continue
                if not isinstance(parsed, Mapping):
                    invalid_calls.append(
                        {
                            "call_id": call_id,
                            "identity": identity,
                        }
                    )
                    continue
                arguments = dict(parsed)
            try:
                calls.append(ToolCall(call_id=call_id, identity=identity, arguments=arguments))
            except ValueError:
                continue
        metadata = dict(self.metadata)
        if invalid_calls:
            # 不把损坏的 JSON 静默降级为空对象；上层会将其作为协议错误终止
            # 本回合，避免空参数意外触发一个无 required 字段的副作用工具。
            metadata["invalid_tool_calls"] = tuple(invalid_calls)
        return ProviderResponse(
            text=self.text,
            reasoning=self.reasoning,
            murmur=self.murmur,
            tool_calls=tuple(calls),
            raw_tool_calls=self.raw_tool_calls,
            usage=dict(self.usage),
            finish_reason=self.finish_reason,
            metadata=metadata,
        )


class ProviderAdapterRuntime:
    """统一封装适配器的取消、超时、重试和结果聚合。"""

    def __init__(
        self,
        adapter: ProviderAdapterLike,
        *,
        retry_policy: RetryPolicy | None = None,
        timeout_seconds: float | None = None,
        event_sink: EventSink | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not hasattr(adapter, "stream") or not callable(adapter.stream):
            raise TypeError("adapter must expose stream")
        self.adapter = adapter
        self.retry_policy = retry_policy or RetryPolicy()
        self.timeout_seconds = (
            None if timeout_seconds is None else max(0.001, float(timeout_seconds))
        )
        self.event_sink = event_sink
        self._sleep = sleep
        self._cancel_events: dict[tuple[str, str, str, str, int], asyncio.Event] = {}
        self._closed = False

    @property
    def provider(self) -> str:
        return str(getattr(self.adapter, "provider", "unknown"))

    @property
    def protocol(self) -> str:
        return str(getattr(self.adapter, "protocol", "unknown"))

    @property
    def capabilities(self) -> frozenset[str]:
        declared = frozenset(getattr(self.adapter, "capabilities", frozenset()))
        checker = getattr(self.adapter, "supports", None)
        if not callable(checker):
            return declared
        effective: set[str] = set()
        for capability in declared:
            try:
                if checker(capability):
                    effective.add(capability)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                continue
        return frozenset(effective)

    def supports(self, capability: str) -> bool:
        normalized = str(capability or "").strip().lower()
        checker = getattr(self.adapter, "supports", None)
        if callable(checker):
            try:
                return bool(checker(normalized))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
        return normalized in self.capabilities

    def diagnostics(self) -> Mapping[str, object]:
        """返回当前适配器生命周期的脱敏状态。"""

        return {
            "provider": self.provider[:64],
            "protocol": self.protocol[:64],
            "capabilities": tuple(sorted(self.capabilities)),
            "closed": self._closed,
            "active_requests": len(self._cancel_events),
        }

    def _key(self, context: ConversationContext) -> tuple[str, str, str, str, int]:
        return (
            context.mode,
            context.profile_id,
            context.session_id,
            context.turn_id,
            context.generation_id,
        )

    def cancel(self, context: ConversationContext) -> bool:
        event = self._cancel_events.get(self._key(context))
        if event is None:
            event = asyncio.Event()
            self._cancel_events[self._key(context)] = event
        event.set()
        return True

    async def _emit(self, event: ProviderEvent) -> None:
        if self.event_sink is None:
            return
        result = self.event_sink(event)
        if inspect.isawaitable(result):
            await result

    async def _attempt(
        self,
        request: ChatRequest,
        context: ConversationContext,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[ProviderEvent]:
        async def iterate() -> AsyncIterator[ProviderEvent]:
            async for event in self.adapter.stream(
                request, context=context, cancel_event=cancel_event
            ):
                if cancel_event.is_set():
                    raise AdapterCancelled()
                if not isinstance(
                    event,
                    (
                        TextDelta,
                        ReasoningDelta,
                        MurmurDelta,
                        ToolCallDelta,
                        TurnFinished,
                        AdapterMetadata,
                    ),
                ):
                    raise ProviderAdapterError(
                        "adapter emitted an unsupported event", category="protocol"
                    )
                yield event

        if self.timeout_seconds is None:
            async for event in iterate():
                yield event
            return
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async for event in iterate():
                    yield event
        except TimeoutError as exc:
            raise ProviderAdapterError(
                "provider request timed out",
                category="timeout",
                retryable=True,
                cause=exc,
            ) from exc

    async def stream(
        self,
        request: ChatRequest,
        *,
        context: ConversationContext | Mapping[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        if self._closed:
            raise ProviderAdapterError("adapter runtime is closed", category="configuration")
        if not isinstance(request, ChatRequest):
            raise TypeError("request must be a ChatRequest")
        current_context = ensure_context(context)
        external_cancel = cancel_event
        internal_cancel = self._cancel_events.setdefault(
            self._key(current_context), asyncio.Event()
        )
        attempt = 1
        # 回退边界遵循统一契约：任何已发布事件都会锁定当前尝试，避免
        # reasoning、工具分片或元数据已到达后重新请求造成重复副作用。
        event_emitted = False
        bridge_task: asyncio.Task[None] | None = None
        if external_cancel is not None and external_cancel is not internal_cancel:
            if external_cancel.is_set():
                internal_cancel.set()
            else:

                async def bridge_cancel() -> None:
                    await external_cancel.wait()
                    internal_cancel.set()

                bridge_task = asyncio.create_task(bridge_cancel())
        try:
            while True:
                if (
                    external_cancel is not None and external_cancel.is_set()
                ) or internal_cancel.is_set():
                    raise AdapterCancelled()
                try:
                    async for event in self._attempt(request, current_context, internal_cancel):
                        event_emitted = True
                        await self._emit(event)
                        yield event
                    return
                except asyncio.CancelledError:
                    raise
                except AdapterCancelled:
                    raise
                except Exception as raw_error:
                    error = classify_exception(raw_error, before_first_event=not event_emitted)
                    if self.retry_policy.should_retry(
                        error, attempt=attempt, before_first_event=not event_emitted
                    ):
                        if internal_cancel.is_set():
                            raise AdapterCancelled()
                        await self._sleep(
                            self.retry_policy.delay_for(
                                attempt,
                                retry_after_seconds=getattr(error, "retry_after_seconds", None),
                            )
                        )
                        attempt += 1
                        continue
                    raise error from raw_error
        finally:
            if bridge_task is not None:
                bridge_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await bridge_task
            self._cancel_events.pop(self._key(current_context), None)

    async def complete(
        self,
        request: ChatRequest,
        *,
        context: ConversationContext | Mapping[str, Any] | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ProviderResponse:
        response = ProviderResponse()
        async for event in self.stream(request, context=context, cancel_event=cancel_event):
            response = response.add(event)
        return response.finalized()

    async def aclose(self) -> None:
        self._closed = True
        close = getattr(self.adapter, "aclose", None)
        if close is not None:
            result = close()
            if inspect.isawaitable(result):
                await result

    async def __aenter__(self) -> ProviderAdapterRuntime:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


ProviderRuntime = ProviderAdapterRuntime
AdapterRuntime = ProviderAdapterRuntime


__all__ = [
    "EventSink",
    "AdapterRuntime",
    "ProviderAdapter",
    "ProviderAdapterLike",
    "ProviderAdapterRuntime",
    "ProviderEvent",
    "ProviderResponse",
    "ProviderRuntime",
    "ensure_context",
]
