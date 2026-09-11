"""按任务和能力选择模型渠道，并在首个事件前执行显式回退。"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

from core.adapters.direct.base import (
    ProviderAdapterLike,
    ProviderAdapterRuntime,
    ProviderEvent,
    ProviderResponse,
)
from core.adapters.direct.errors import ProviderAdapterError, classify_exception
from core.adapters.direct.events import AdapterMetadata, ToolCallDelta
from core.adapters.providers.registry import create_provider_adapter
from core.contracts.chat import ChatMessage, ChatRequest
from core.events.types import (
    ConversationContext,
    MurmurDelta,
    ReasoningDelta,
    TextDelta,
    TurnFinished,
)
from db.api_call_audit_repository import (
    ApiCallAuditHandle,
    ApiCallAuditRecord,
    ApiCallAuditRepository,
    sanitize_audit_value,
)
from logger.events import log_event

from .channels import (
    SUPPORTED_PROTOCOLS,
    ChannelConfig,
    channel_from_mapping,
    load_channels,
    select_channels,
    sort_channels,
)
from .vision import (
    VisionSummaryPolicy,
    build_vision_request,
    request_has_inline_images,
    strip_images_and_attach_summary,
    validate_inline_images,
)

logger = logging.getLogger(__name__)


def _audit_request_payload(request: ChatRequest) -> dict[str, object]:
    """生成有界请求快照；图片只保存格式和编码长度，不保存原始图像。"""

    messages: list[dict[str, object]] = []
    for message in request.messages:
        mapped = message.as_mapping()
        content = mapped.get("content")
        if isinstance(content, (list, tuple)):
            parts: list[object] = []
            for part in content:
                if isinstance(part, Mapping) and str(part.get("type", "")) == "image":
                    image = dict(part)
                    data = image.get("data")
                    image["data"] = {
                        "encoding": "base64",
                        "length": len(str(data or "")),
                    }
                    parts.append(image)
                else:
                    parts.append(part)
            mapped["content"] = parts
        messages.append(mapped)
    payload = {
        "model": request.model,
        "messages": messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "tools": [tool.as_openai_tool() for tool in request.tools],
        "tool_choice": request.tool_choice,
        "metadata": dict(request.metadata),
    }
    sanitized = sanitize_audit_value(payload)
    return dict(sanitized) if isinstance(sanitized, Mapping) else {}


def _audit_cache_fields(
    usage: Mapping[str, Any], metadata: Mapping[str, Any]
) -> tuple[int | None, int | None, float | None, float | None, dict[str, object]]:
    """从各适配器已规范化的 usage/metadata 提取缓存读写指标。"""

    read_keys = ("cached_tokens", "cache_read_tokens", "prompt_tokens_details.cached_tokens")
    write_keys = ("cache_creation_tokens", "cache_write_tokens")

    def integer(keys: tuple[str, ...]) -> tuple[int | None, str | None]:
        for key in keys:
            value = usage.get(key)
            if isinstance(value, bool):
                continue
            try:
                number = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if number >= 0:
                return number, key
        return None, None

    def duration(key: str) -> float | None:
        value = metadata.get(key)
        if value is None:
            value = usage.get(key)
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if number >= 0 else None

    read_tokens, read_source = integer(read_keys)
    write_tokens, write_source = integer(write_keys)
    cache_info: dict[str, object] = {
        "read": read_tokens is not None,
        "write": write_tokens is not None,
    }
    if read_source is not None:
        cache_info["read_source"] = read_source
    if write_source is not None:
        cache_info["write_source"] = write_source
    return (
        read_tokens,
        write_tokens,
        duration("cache_read_duration_ms"),
        duration("cache_write_duration_ms"),
        cache_info,
    )


class ModelRoutingError(RuntimeError):
    """没有满足请求约束的可用渠道。"""

    def __init__(self, message: str, *, safe_message: str | None = None) -> None:
        """保留内部诊断，同时提供不会泄露配置键名的公开回执。"""

        detail = str(message or "model route is unavailable").strip()
        super().__init__(detail)
        self.safe_message = str(
            safe_message or "模型渠道未就绪，请点击“配置模型”检查连接信息"
        ).strip()


def _connection_failure_message(error: BaseException) -> str:
    """把探测异常压缩为不会回显密钥、端点或响应正文的用户提示。"""

    category = str(getattr(error, "category", "") or "").strip().lower()
    return {
        "authentication": "模型连接失败，请检查密钥设置",
        "authorization": "模型连接失败，请检查访问权限",
        "network": "模型连接失败，请检查服务地址和网络",
        "timeout": "模型连接超时，请稍后重试",
        "rate_limit": "模型连接受限，请稍后重试",
        "server": "模型服务暂时不可用，请稍后重试",
        "protocol": "模型响应格式不受支持",
        "configuration": "模型连接配置不完整",
        "cancelled": "模型连接测试已停止",
    }.get(category, "模型连接失败，请检查服务设置")


_CONNECTION_FAILURE_REASONS = frozenset(
    {
        "authentication",
        "authorization",
        "network",
        "timeout",
        "rate_limit",
        "server",
        "protocol",
        "configuration",
        "cancelled",
        "unknown",
    }
)


@dataclass(frozen=True)
class RouteSpec:
    """一个任务角色的路由约束。"""

    task: str
    channel_id: str = ""
    model: str = ""
    protocol: str = ""
    required_capabilities: frozenset[str] = frozenset()
    fallback_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        task = str(self.task or "").strip()
        if not task:
            raise ValueError("route task is required")
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "channel_id", str(self.channel_id or "").strip())
        object.__setattr__(self, "model", str(self.model or "").strip())
        object.__setattr__(self, "protocol", str(self.protocol or "").strip().lower())
        raw_required = self.required_capabilities
        if isinstance(raw_required, str):
            raw_required = (item.strip() for item in raw_required.split(","))
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(str(item).strip().lower() for item in raw_required if str(item).strip()),
        )
        raw_fallbacks = self.fallback_channels
        if isinstance(raw_fallbacks, str):
            raw_fallbacks = (item.strip() for item in raw_fallbacks.split(","))
        object.__setattr__(
            self,
            "fallback_channels",
            tuple(str(item).strip() for item in raw_fallbacks if str(item).strip()),
        )

    @classmethod
    def from_mapping(cls, task: str, value: object) -> RouteSpec:
        if isinstance(value, str):
            return cls(task=task, channel_id=value)
        if value is None:
            return cls(task=task)
        if not isinstance(value, Mapping):
            raise ModelRoutingError(f"route {task} must be a string or mapping")
        required = value.get("required_capabilities", value.get("capabilities", ()))
        fallback = value.get("fallback_channels", value.get("fallback", ()))
        if isinstance(required, str):
            required = [item.strip() for item in required.split(",") if item.strip()]
        if isinstance(fallback, str):
            fallback = [item.strip() for item in fallback.split(",") if item.strip()]
        return cls(
            task=task,
            channel_id=value.get("channel_id", value.get("channel", "")),
            model=value.get("model", ""),
            protocol=value.get("protocol", ""),
            required_capabilities=frozenset(required or ()),
            fallback_channels=tuple(fallback or ()),
        )


@dataclass(frozen=True)
class RouteSelection:
    task: str
    channels: tuple[ChannelConfig, ...]
    requested_model: str = ""
    required_capabilities: frozenset[str] = frozenset()

    @property
    def primary(self) -> ChannelConfig:
        if not self.channels:
            raise ModelRoutingError(f"no channel selected for task: {self.task}")
        return self.channels[0]


@dataclass
class _ChannelHealth:
    failures: int = 0
    last_category: str = ""
    last_error: str = ""
    cooldown_until: float = 0.0


_HEALTH_COOLDOWN_BASE_SECONDS = 1.0
_HEALTH_COOLDOWN_MAX_SECONDS = 60.0


class ModelRouter:
    """渠道注册、路由和同协议首事件前回退。"""

    def __init__(
        self,
        channels: Iterable[ChannelConfig | Mapping[str, Any]] = (),
        *,
        routes: Mapping[str, object] | None = None,
        adapter_factory: Callable[[ChannelConfig], ProviderAdapterLike | ProviderAdapterRuntime]
        | None = None,
        runtime_factory: Callable[[ProviderAdapterLike, ChannelConfig], ProviderAdapterRuntime]
        | None = None,
        clock: Callable[[], float] | None = None,
        vision_policy: VisionSummaryPolicy | Mapping[str, Any] | None = None,
        audit_repository: ApiCallAuditRepository | None = None,
    ) -> None:
        self._channels: dict[str, ChannelConfig] = {}
        self._routes: dict[str, RouteSpec] = {}
        self._runtimes: dict[str, ProviderAdapterRuntime] = {}
        self._health: dict[str, _ChannelHealth] = {}
        self._adapter_factory = adapter_factory
        self._runtime_factory = runtime_factory
        self._clock = clock or __import__("time").time
        self._vision_policy = (
            vision_policy
            if isinstance(vision_policy, VisionSummaryPolicy)
            else VisionSummaryPolicy.from_mapping(vision_policy)
        )
        self._audit_repository = audit_repository
        self._audit_handles: dict[tuple[str, str, str, str, int], ApiCallAuditHandle] = {}
        self._lock = asyncio.Lock()
        # Qt/Web 控制面可能在主线程点选渠道，而对话路由运行在
        # RuntimeLoop 线程；路由快照和热切换必须在同一把锁下完成。
        self._route_lock = threading.RLock()
        for raw_channel in channels:
            self.register(raw_channel)
        for task, value in (routes or {}).items():
            self.set_route(str(task), value)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        adapter_factory: Callable[[ChannelConfig], ProviderAdapterLike | ProviderAdapterRuntime]
        | None = None,
        runtime_factory: Callable[[ProviderAdapterLike, ChannelConfig], ProviderAdapterRuntime]
        | None = None,
        audit_repository: ApiCallAuditRepository | None = None,
    ) -> ModelRouter:
        if not isinstance(value, Mapping):
            raise ModelRoutingError("routing configuration must be a mapping")
        llm = value.get("llm", value)
        if not isinstance(llm, Mapping):
            raise ModelRoutingError("llm configuration must be a mapping")
        routing = llm.get("routing", {})
        if not isinstance(routing, Mapping):
            raise ModelRoutingError("llm.routing must be a mapping")
        routing_values = dict(routing)
        # 兼容把视觉路由直接放在 ``llm.vision`` 的配置写法；只有出现
        # 明确路由字段时才提升为 route，单独的摘要策略仍保持策略配置。
        direct_vision = llm.get("vision")
        route_keys = {
            "channel",
            "channel_id",
            "model",
            "protocol",
            "required_capabilities",
            "capabilities",
            "fallback_channels",
            "fallback",
        }
        if "vision" not in routing_values:
            if isinstance(direct_vision, Mapping) and route_keys.intersection(direct_vision):
                routing_values["vision"] = direct_vision
            elif isinstance(direct_vision, str) and direct_vision.strip():
                routing_values["vision"] = direct_vision
        # ``llm.routing.vision`` 是视觉渠道的首选/回退路由；若调用方希望把
        # 摘要上限和提示词独立配置，也可使用 ``llm.vision``。路由项优先，
        # 但两者都只在本地构造策略，不会把配置原文发送到公开状态。
        vision_route = routing_values.get("vision")
        vision_policy = direct_vision
        if isinstance(vision_route, bool):
            vision_policy = {"enabled": vision_route}
            routing_values.pop("vision", None)
            vision_route = None
        if isinstance(vision_route, Mapping):
            policy_keys = (
                "enabled",
                "max_summary_chars",
                "summary_max_chars",
                "max_tokens",
                "prompt",
                "summary_prompt",
            )
            route_policy = {key: vision_route[key] for key in policy_keys if key in vision_route}
            if isinstance(vision_policy, Mapping):
                merged_policy = dict(vision_policy)
                merged_policy.update(route_policy)
                vision_policy = merged_policy
            elif route_policy:
                vision_policy = route_policy
        return cls(
            load_channels(llm.get("channels", ())),
            routes=routing_values,
            adapter_factory=adapter_factory,
            runtime_factory=runtime_factory,
            vision_policy=vision_policy,
            audit_repository=audit_repository,
        )

    def set_audit_repository(self, repository: ApiCallAuditRepository | None) -> None:
        """绑定或替换持久化 API 调用审计仓储。"""

        if repository is not None and not isinstance(repository, ApiCallAuditRepository):
            raise TypeError("audit repository must be an ApiCallAuditRepository or None")
        self._audit_repository = repository

    @staticmethod
    def _audit_context_key(context: object) -> tuple[str, str, str, str, int] | None:
        if isinstance(context, ConversationContext):
            return (
                context.mode,
                context.profile_id,
                context.session_id,
                context.turn_id,
                context.generation_id,
            )
        if isinstance(context, Mapping):
            try:
                generation_id = int(context.get("generation_id", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                generation_id = 0
            return (
                str(context.get("mode", "direct") or "direct"),
                str(context.get("profile_id", "") or ""),
                str(context.get("session_id", "") or ""),
                str(context.get("turn_id", "") or ""),
                max(0, generation_id),
            )
        return None

    async def record_tool_execution(self, record: object) -> None:
        """持久化工具执行明细，并追加到当前回合最近一次模型调用。"""

        if self._audit_repository is None:
            return
        context = getattr(record, "context", None)
        context_metadata = getattr(context, "metadata", {})
        if not isinstance(context_metadata, Mapping):
            context_metadata = {}
        mode = str(
            getattr(record, "mode", "") or context_metadata.get("mode", "direct") or "direct"
        )
        try:
            generation_id = max(
                0,
                int(
                    getattr(record, "generation_id", None)
                    or context_metadata.get("generation_id", 0)
                    or 0
                ),
            )
        except (TypeError, ValueError, OverflowError):
            generation_id = 0
        key = self._audit_context_key(context)
        if key is None and context is not None:
            key = self._audit_context_key(
                {
                    "mode": mode,
                    "profile_id": getattr(context, "profile_id", ""),
                    "session_id": getattr(context, "session_id", ""),
                    "turn_id": getattr(context, "turn_id", ""),
                    "generation_id": generation_id,
                }
            )
        if key is None:
            key = self._audit_context_key(
                {
                    "mode": "agent" if str(getattr(record, "source", "")) == "agent" else "direct",
                    "profile_id": getattr(record, "profile_id", ""),
                    "session_id": getattr(record, "session_id", ""),
                    "turn_id": getattr(record, "turn_id", ""),
                    "generation_id": 0,
                }
            )
        with self._route_lock:
            handle = self._audit_handles.get(key) if key is not None else None
        detail = record.as_dict() if callable(getattr(record, "as_dict", None)) else record
        if isinstance(detail, Mapping):
            safe_detail = sanitize_audit_value(detail)
            try:
                started_at = float(detail.get("started_at") or self._audit_repository.now())
            except (TypeError, ValueError, OverflowError):
                started_at = self._audit_repository.now()
            finished_at = detail.get("finished_at")
            try:
                finished_value = float(finished_at) if finished_at is not None else started_at
            except (TypeError, ValueError, OverflowError):
                finished_value = started_at
            try:
                duration_ms = max(0.0, float(detail.get("duration_ms") or 0.0))
            except (TypeError, ValueError, OverflowError):
                duration_ms = 0.0
            identity = str(detail.get("identity", "") or "")
            call_id = str(detail.get("call_id", "") or "")
            phase = str(detail.get("phase", "execute") or "execute")
            try:
                tool_record = ApiCallAuditRecord(
                    request_id=f"tool-{uuid.uuid4().hex}",
                    mode=mode,
                    profile_id=str(detail.get("profile_id", "") or ""),
                    session_id=str(detail.get("session_id", "") or ""),
                    turn_id=str(detail.get("turn_id", "") or ""),
                    generation_id=generation_id,
                    attempt=max(1, int(detail.get("attempt", 1) or 1)),
                    kind="tool",
                    operation=f"tool.{phase}",
                    status=str(detail.get("status", "failed") or "failed"),
                    started_at=started_at,
                    completed_at=finished_value,
                    total_duration_ms=duration_ms,
                    provider="local",
                    protocol="tool",
                    channel_id=str(detail.get("source", "") or "local"),
                    channel_name=str(detail.get("source", "") or "local"),
                    requested_model="",
                    input_payload={
                        "identity": identity,
                        "arguments": detail.get("arguments", {}),
                    },
                    output_payload=detail.get("result", {}),
                    tool_calls=(
                        {
                            "call_id": call_id,
                            "identity": identity,
                            "arguments": detail.get("arguments", {}),
                        },
                    ),
                    tool_executions=(safe_detail,),
                    error_type=str(detail.get("error_type", "") or ""),
                    error_message=str(detail.get("error", "") or ""),
                    metadata={
                        "plan_id": detail.get("plan_id", ""),
                        "step_id": detail.get("step_id", ""),
                        "parallel_group": detail.get("parallel_group", 0),
                        "batch_id": detail.get("batch_id", ""),
                        "batch_index": detail.get("batch_index", -1),
                    },
                )
                await self._audit_repository.asave(tool_record)
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.debug("standalone tool audit write failed", exc_info=True)
            if handle is not None:
                try:
                    await handle.aadd_tool_execution(safe_detail)
                except (KeyError, TypeError, ValueError, RuntimeError):
                    logger.debug("api audit tool execution append failed", exc_info=True)

    def clear_audit_context(self, context: object) -> None:
        """释放已结束回合的审计句柄，避免上下文长期占用内存。"""

        key = self._audit_context_key(context)
        if key is not None:
            with self._route_lock:
                self._audit_handles.pop(key, None)

    def register(self, channel: ChannelConfig | Mapping[str, Any]) -> ChannelConfig:
        config = channel if isinstance(channel, ChannelConfig) else channel_from_mapping(channel)
        with self._route_lock:
            if config.id in self._channels:
                raise ModelRoutingError(f"duplicate channel id: {config.id}")
            self._channels[config.id] = config
            self._health.setdefault(config.id, _ChannelHealth())
        return config

    def replace(self, channel: ChannelConfig | Mapping[str, Any]) -> ChannelConfig:
        config = channel if isinstance(channel, ChannelConfig) else channel_from_mapping(channel)
        with self._route_lock:
            self._channels[config.id] = config
            # 新配置不继承旧密钥/端点的失败冷却，避免健康状态污染替换后的渠道。
            self._health[config.id] = _ChannelHealth()
            runtime = self._runtimes.pop(config.id, None)
        if runtime is not None:
            close = getattr(runtime, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # 同步配置热替换发生在事件循环外时，仍然完整释放网络资源。
                        asyncio.run(result)
                    else:
                        loop.create_task(result)
        return config

    def set_route(self, task: str, value: object) -> RouteSpec:
        task_name = str(task or "").strip() or "dialogue"
        route_keys = {
            "channel",
            "channel_id",
            "model",
            "protocol",
            "required_capabilities",
            "capabilities",
            "fallback_channels",
            "fallback",
        }
        # vision 映射也可只承载摘要策略（enabled/max_tokens/prompt），
        # 这不是渠道路由；保留空 RouteSpec 供 diagnostics 使用。
        policy_only_vision = task_name.casefold() == "vision" and (
            value is None or (isinstance(value, Mapping) and not route_keys.intersection(value))
        )
        spec = RouteSpec.from_mapping(task_name, value)
        with self._route_lock:
            if spec.channel_id and spec.channel_id not in self._channels:
                raise ModelRoutingError(
                    f"route {spec.task} references unknown channel: {spec.channel_id}"
                )
            if len(set(spec.fallback_channels)) != len(spec.fallback_channels):
                raise ModelRoutingError(f"route {spec.task} contains duplicate fallback channels")
            if spec.channel_id and spec.channel_id in spec.fallback_channels:
                raise ModelRoutingError(
                    f"route {spec.task} cannot use its primary channel as a fallback"
                )
            unknown_fallbacks = tuple(
                channel_id
                for channel_id in spec.fallback_channels
                if channel_id not in self._channels
            )
            if unknown_fallbacks:
                raise ModelRoutingError(
                    f"route {spec.task} references unknown fallback channel: {unknown_fallbacks[0]}"
                )
            if spec.fallback_channels and not spec.channel_id:
                raise ModelRoutingError(
                    f"route {spec.task} requires a primary channel when fallbacks are set"
                )
            if spec.protocol and spec.protocol not in SUPPORTED_PROTOCOLS:
                raise ModelRoutingError(
                    f"route {spec.task} has unsupported protocol: {spec.protocol}"
                )

            required = set(spec.required_capabilities)
            if spec.task.casefold() == "vision" and not policy_only_vision:
                required.add("vision")
            option_ids = (
                (spec.channel_id, *spec.fallback_channels)
                if spec.channel_id
                else tuple(self._channels)
            )
            eligible = tuple(
                channel
                for channel_id in option_ids
                if (channel := self._channels[channel_id]).is_ready
                and (not spec.model or spec.model in channel.models)
                and (not spec.protocol or channel.protocol == spec.protocol)
                and required.issubset(channel.capabilities)
            )
            if (spec.channel_id or spec.model or spec.protocol or required) and not eligible:
                raise ModelRoutingError(
                    f"route {spec.task} has no ready channel satisfying its constraints"
                )
            self._routes[spec.task] = spec
        return spec

    def channels(self) -> tuple[ChannelConfig, ...]:
        with self._route_lock:
            return sort_channels(tuple(self._channels.values()))

    @property
    def vision_policy(self) -> VisionSummaryPolicy:
        """返回当前视觉摘要策略的不可变快照。"""

        return self._vision_policy

    def vision_status(self) -> Mapping[str, object]:
        """返回视觉回退状态，不包含地址、密钥或供应商原始错误。"""

        with self._route_lock:
            route = self._routes.get("vision")
        if not self._vision_policy.enabled:
            return {
                "enabled": False,
                "configured": route is not None,
                "ready": False,
                "channel_id": route.channel_id if route is not None else "",
                "model": route.model if route is not None else "",
                "fallback_channels": route.fallback_channels if route is not None else (),
                "reason": "disabled",
            }
        try:
            selection = self.resolve(
                "vision",
                required_capabilities=("streaming", "vision"),
            )
        except ModelRoutingError:
            return {
                "enabled": self._vision_policy.enabled,
                "configured": route is not None,
                "ready": False,
                "channel_id": route.channel_id if route is not None else "",
                "model": route.model if route is not None else "",
                "fallback_channels": route.fallback_channels if route is not None else (),
                "reason": "unavailable",
            }
        primary = selection.primary
        health = self.health(primary.id) or {}
        now = float(self._clock())
        cooldown_until = float(health.get("cooldown_until", 0.0))
        effective_ready = primary.is_ready and self._effective_supports(primary, "vision")
        return {
            "enabled": self._vision_policy.enabled,
            "configured": route is not None,
            "ready": bool(effective_ready),
            "channel_id": primary.id,
            "model": selection.requested_model or primary.selected_model,
            "fallback_channels": route.fallback_channels if route is not None else (),
            "failures": int(health.get("failures", 0) or 0),
            "cooling": cooldown_until > now,
            "reason": "ready" if effective_ready else "configuration",
        }

    def image_capability_status(self, task: str = "dialogue") -> Mapping[str, object]:
        """返回当前对话图片处理能力，区分直传与视觉摘要回退。

        控制面只需要知道图片是否可处理以及采用哪条路径，不应自行读取
        渠道能力或复制路由配置；返回值不包含地址、密钥和供应商错误。
        """

        task_name = str(task or "dialogue").strip() or "dialogue"
        try:
            selection = self.resolve(task_name)
        except ModelRoutingError:
            return {
                "ready": False,
                "mode": "unavailable",
                "channel_id": "",
                "model": "",
                "reason": "dialogue_route_unavailable",
            }
        primary = selection.primary
        health = self.health(primary.id) or {}
        now = float(self._clock())
        try:
            cooldown_until = float(health.get("cooldown_until", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            cooldown_until = 0.0
        if self._effective_supports(primary, "vision") and cooldown_until <= now:
            return {
                "ready": True,
                "mode": "direct",
                "channel_id": primary.id,
                "model": selection.requested_model or primary.selected_model,
                "reason": "primary_supports_vision",
            }
        fallback = self.vision_status()
        if bool(fallback.get("ready")):
            return {
                "ready": True,
                "mode": "summary",
                "channel_id": str(fallback.get("channel_id", "") or ""),
                "model": str(fallback.get("model", "") or ""),
                "reason": "vision_summary_fallback",
            }
        return {
            "ready": False,
            "mode": "unavailable",
            "channel_id": "",
            "model": "",
            "reason": "vision_fallback_unavailable",
        }

    def diagnostics(self, task: str = "dialogue") -> Mapping[str, Any]:
        """返回不含密钥的渠道就绪诊断，供启动界面和配置向导使用。"""

        task_name = str(task or "dialogue").strip() or "dialogue"
        with self._route_lock:
            spec = self._routes.get(task_name, RouteSpec(task_name))
        rows: list[dict[str, Any]] = []
        for channel in self.channels():
            health = self.health(channel.id) or {}
            now = float(self._clock())
            cooldown_until = float(health.get("cooldown_until", 0.0))
            rows.append(
                {
                    "id": channel.id,
                    "protocol": channel.protocol,
                    "enabled": channel.enabled,
                    "base_url_configured": bool(channel.base_url),
                    "model_configured": bool(channel.selected_model),
                    "api_key_configured": bool(channel.api_key),
                    "ready": channel.is_ready,
                    "reason": channel.readiness_reason,
                    "failures": int(health.get("failures", 0)),
                    "last_category": str(health.get("last_category", "") or ""),
                    "cooling": cooldown_until > now,
                }
            )
        try:
            selection = self.resolve(task_name)
        except ModelRoutingError as exc:
            result = {
                "task": task_name,
                "route_channel": spec.channel_id,
                "route_model": spec.model,
                "ready": False,
                "reason": str(exc),
                "channels": tuple(rows),
            }
            if task_name == "dialogue":
                result["vision"] = self.vision_status()
                result["image"] = self.image_capability_status(task_name)
            return result
        result = {
            "task": task_name,
            "route_channel": spec.channel_id,
            "route_model": spec.model,
            "selected_channel": selection.primary.id,
            "selected_model": selection.requested_model or selection.primary.selected_model,
            "ready": bool(selection.primary.is_ready),
            "reason": "ready" if selection.primary.is_ready else selection.primary.readiness_reason,
            "channels": tuple(rows),
        }
        if task_name == "dialogue":
            result["vision"] = self.vision_status()
            result["image"] = self.image_capability_status(task_name)
        return result

    def route(self, task: str) -> RouteSpec | None:
        with self._route_lock:
            return self._routes.get(str(task or "").strip())

    def select_channel(
        self,
        channel_id: str,
        *,
        task: str = "dialogue",
    ) -> Mapping[str, object]:
        """在当前进程内切换任务的首选渠道。

        这是配置向导和本地控制面共用的热切换入口。它只修改路由内存状态，
        不写入 YAML、不读取或返回密钥；下一轮对话会使用新渠道，当前正在
        执行的流式回合仍由其已有请求完成。调用方负责在切换前检查活动回合。
        """

        task_name = str(task or "dialogue").strip() or "dialogue"
        selected_id = str(channel_id or "").strip()
        with self._route_lock:
            if not selected_id:
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": "configuration",
                    "message": "请选择模型渠道",
                }
            channel = self._channels.get(selected_id)
            if channel is None:
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": "not_found",
                    "message": "模型渠道不存在",
                }
            if not channel.enabled:
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": "disabled",
                    "message": "模型渠道已停用",
                }
            if not channel.base_url or not channel.selected_model:
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": "configuration",
                    "message": "模型渠道尚未配置完成",
                }
            health = self._health.get(selected_id, _ChannelHealth())
            if float(health.cooldown_until) > float(self._clock()):
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": "cooldown",
                    "message": "模型渠道暂时冷却，请稍后重试",
                }
            current = self._routes.get(task_name) or RouteSpec(task_name)
            required = set(current.required_capabilities)
            if task_name.casefold() == "vision":
                required.add("vision")
            if not required.issubset(channel.capabilities):
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": "capability",
                    "message": "模型渠道不支持当前任务能力",
                }
            if current.channel_id == selected_id and not current.model and not current.protocol:
                return {
                    "status": "updated",
                    "ready": True,
                    "channel_id": selected_id,
                    "model": channel.selected_model,
                    "message": "当前已使用此模型渠道",
                }
            fallback_ids = tuple(
                dict.fromkeys(
                    item
                    for item in (current.channel_id, *current.fallback_channels)
                    if item and item != selected_id and item in self._channels
                )
            )
            # 显式选择渠道时清除旧的模型/协议筛选，否则旧路由约束可能把用户
            # 刚点选的渠道再次筛掉；任务能力约束仍然保留。
            self._routes[task_name] = RouteSpec(
                task=task_name,
                channel_id=selected_id,
                required_capabilities=frozenset(required),
                fallback_channels=fallback_ids,
            )
            return {
                "status": "updated",
                "ready": True,
                "channel_id": selected_id,
                "model": channel.selected_model,
                "message": "模型渠道已切换（仅当前运行）",
            }

    def select_model(
        self,
        model: str,
        *,
        task: str = "dialogue",
        channel_id: str | None = None,
    ) -> Mapping[str, object]:
        """在当前进程内切换任务使用的模型。

        模型必须已经由渠道配置显式声明；不会根据名称猜测供应商，也不会把
        密钥、地址或完整渠道配置写入回执。指定 ``channel_id`` 时只在该渠道
        的 ``model/models`` 集合中查找，否则按优先级和健康状态选择渠道，并
        保留其余同模型渠道作为有序回退。
        """

        task_name = str(task or "dialogue").strip() or "dialogue"
        selected_model = str(model or "").strip()
        if not selected_model:
            return {
                "status": "unavailable",
                "ready": False,
                "reason": "configuration",
                "message": "请选择模型",
            }
        requested_channel = str(channel_id or "").strip()
        with self._route_lock:
            current = self._routes.get(task_name) or RouteSpec(task_name)
            required = set(current.required_capabilities)
            if task_name.casefold() == "vision":
                required.add("vision")
            channels = tuple(
                channel
                for channel in self.channels()
                if channel.is_ready
                and selected_model in channel.models
                and (not requested_channel or channel.id == requested_channel)
                and required.issubset(channel.capabilities)
            )
            if not channels:
                if requested_channel and requested_channel not in self._channels:
                    reason = "not_found"
                    message = "模型渠道不存在"
                elif requested_channel:
                    requested = self._channels[requested_channel]
                    if not requested.enabled:
                        reason = "disabled"
                        message = "模型渠道已停用"
                    elif not requested.base_url or not requested.selected_model:
                        reason = "configuration"
                        message = "模型渠道尚未配置完成"
                    elif selected_model not in requested.models:
                        reason = "model"
                        message = "该渠道未声明此模型"
                    else:
                        reason = "capability"
                        message = "模型渠道不支持当前任务能力"
                else:
                    reason = "model"
                    message = "没有可用渠道声明此模型"
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": reason,
                    "message": message,
                }
            healthy = self._healthy_options(channels)
            if not healthy:
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": "cooldown",
                    "message": "模型渠道暂时冷却，请稍后重试",
                }
            preferred = next(
                (item for item in healthy if item.id == current.channel_id),
                None,
            )
            channel = preferred or healthy[0]
            health = self._health.get(channel.id, _ChannelHealth())
            if health.cooldown_until > float(self._clock()):
                return {
                    "status": "unavailable",
                    "ready": False,
                    "reason": "cooldown",
                    "message": "模型渠道暂时冷却，请稍后重试",
                }
            fallback_ids = tuple(item.id for item in channels if item.id != channel.id)
            self._routes[task_name] = RouteSpec(
                task=task_name,
                channel_id=channel.id,
                model=selected_model,
                required_capabilities=frozenset(required),
                fallback_channels=fallback_ids,
            )
            return {
                "status": "updated",
                "ready": True,
                "channel_id": channel.id,
                "model": selected_model,
                "message": "模型已切换（仅当前运行）",
            }

    def model_options(self, task: str = "dialogue") -> tuple[Mapping[str, object], ...]:
        """返回任务可选模型的脱敏快照，供配置控制面动态切换。"""

        task_name = str(task or "dialogue").strip() or "dialogue"
        with self._route_lock:
            spec = self._routes.get(task_name, RouteSpec(task_name))
            required = set(spec.required_capabilities)
            if task_name.casefold() == "vision":
                required.add("vision")
            rows: list[Mapping[str, object]] = []
            seen: set[tuple[str, str]] = set()
            now = float(self._clock())
            for channel in self.channels():
                if not channel.is_ready or not required.issubset(channel.capabilities):
                    continue
                health = self._health.get(channel.id, _ChannelHealth())
                for model in channel.models:
                    key = (channel.id, model)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "channel_id": channel.id,
                            "model": model,
                            "priority": channel.priority,
                            "ready": health.cooldown_until <= now,
                            "cooling": health.cooldown_until > now,
                            "failures": health.failures,
                        }
                    )
            return tuple(rows)

    def _route_options(self, spec: RouteSpec) -> tuple[ChannelConfig, ...]:
        all_channels = self.channels()
        if spec.channel_id:
            ordered_ids = (spec.channel_id, *spec.fallback_channels)
            by_id = {channel.id: channel for channel in all_channels}
            options = tuple(
                by_id[item] for item in ordered_ids if item in by_id and by_id[item].is_ready
            )
        else:
            options = select_channels(
                all_channels,
                model=spec.model or None,
                protocol=spec.protocol or None,
                required_capabilities=spec.required_capabilities,
            )
        return options

    def _healthy_options(self, options: tuple[ChannelConfig, ...]) -> tuple[ChannelConfig, ...]:
        """过滤熔断冷却中的渠道；全部冷却时保留最早探测渠道。"""

        if not options:
            return ()
        now = float(self._clock())
        with self._route_lock:
            cooldowns = {
                channel.id: self._health.get(channel.id, _ChannelHealth()).cooldown_until
                for channel in options
            }
        available = tuple(channel for channel in options if cooldowns.get(channel.id, 0.0) <= now)
        if available:
            return available
        # 避免所有渠道同时失败后永久不可用，但在最早冷却窗口结束前不
        # 提前重试；窗口结束后才允许一次半开探测。
        earliest = min(
            options,
            key=lambda channel: cooldowns.get(channel.id, 0.0),
        )
        return (earliest,) if cooldowns.get(earliest.id, 0.0) <= now else ()

    def resolve(
        self,
        task: str = "dialogue",
        *,
        model: str | None = None,
        channel_id: str | None = None,
        protocol: str | None = None,
        required_capabilities: Iterable[str] = (),
    ) -> RouteSelection:
        task_name = str(task or "dialogue").strip() or "dialogue"
        with self._route_lock:
            spec = self._routes.get(task_name, RouteSpec(task_name))
        explicit_channel = str(channel_id or "").strip()
        # 调用方传入 channel_id 时，它是对任务路由的明确覆盖；路由配置中
        # 原有的 model/protocol 约束不能把用户指定的频道再次筛掉。显式
        # model/protocol 参数仍保留为更高优先级的筛选条件。
        requested_model = str(
            model if model is not None else ("" if explicit_channel else spec.model)
        ).strip()
        requested_protocol = (
            str(protocol if protocol is not None else ("" if explicit_channel else spec.protocol))
            .strip()
            .lower()
        )
        required = frozenset(
            {
                *spec.required_capabilities,
                *(str(item).strip().lower() for item in required_capabilities if str(item).strip()),
            }
        )
        if task_name.casefold() == "vision":
            required = frozenset((*required, "vision"))
        if explicit_channel:
            spec = RouteSpec(
                task=task_name,
                channel_id=explicit_channel,
                model=requested_model,
                protocol=requested_protocol,
                required_capabilities=required,
                fallback_channels=spec.fallback_channels,
            )
        options = self._route_options(spec)
        if requested_model or requested_protocol or required:
            options = tuple(
                channel
                for channel in options
                if (
                    not requested_model
                    or requested_model in channel.models
                    or requested_model == channel.model
                )
                and (not requested_protocol or channel.protocol == requested_protocol)
                and required.issubset(channel.capabilities)
            )
        options = self._healthy_options(options)
        if not options:
            if not self.channels():
                raise ModelRoutingError(
                    f"no enabled channel satisfies task: {task_name}; "
                    "configure llm.channels with id, protocol, base_url, and model "
                    "or set MEAPET_API_BASE and MEAPET_MODEL"
                )
            raise ModelRoutingError(
                f"no enabled channel satisfies task: {task_name}; "
                "check llm.channels enabled, base_url, model, and routing.channel"
            )
        return RouteSelection(task_name, options, requested_model, required)

    def _make_runtime(self, channel: ChannelConfig) -> ProviderAdapterRuntime:
        with self._route_lock:
            cached = self._runtimes.get(channel.id)
            if cached is not None:
                return cached
            runtime = self._create_runtime(channel)
            self._runtimes[channel.id] = runtime
            return runtime

    def _effective_supports(self, channel: ChannelConfig, capability: str) -> bool:
        """使用已创建运行时的能力回读，并在构造失败时安全降级。"""

        try:
            runtime = self._make_runtime(channel)
            checker = getattr(runtime, "supports", None)
            if callable(checker):
                return bool(checker(capability))
        except Exception:
            return False
        return channel.supports(capability)

    def _create_runtime(self, channel: ChannelConfig) -> ProviderAdapterRuntime:
        """按渠道配置创建一个未缓存的适配器运行时。"""

        if self._adapter_factory is not None:
            produced = self._adapter_factory(channel)
        else:
            produced = create_provider_adapter(channel)
        if isinstance(produced, ProviderAdapterRuntime):
            runtime = produced
        else:
            runtime = (
                self._runtime_factory(produced, channel)
                if self._runtime_factory is not None
                else ProviderAdapterRuntime(
                    produced,
                    retry_policy=channel.retry,
                    timeout_seconds=channel.timeout_seconds or None,
                )
            )
        return runtime

    def runtime(self, selection: RouteSelection | ChannelConfig) -> ProviderAdapterRuntime:
        channel = selection.primary if isinstance(selection, RouteSelection) else selection
        return self._make_runtime(channel)

    def runtime_for(
        self,
        task: str = "dialogue",
        *,
        model: str | None = None,
        channel_id: str | None = None,
        protocol: str | None = None,
        required_capabilities: Iterable[str] = (),
    ) -> ProviderAdapterRuntime:
        return self.runtime(
            self.resolve(
                task,
                model=model,
                channel_id=channel_id,
                protocol=protocol,
                required_capabilities=required_capabilities,
            )
        )

    async def _generate_vision_summary(
        self,
        request: ChatRequest,
        *,
        context: ConversationContext | Mapping[str, Any] | None,
        cancel_event: asyncio.Event | None,
    ) -> str:
        """以流式事件收集有界摘要，不把视觉请求暴露给主模型渠道。

        视觉摘要是非视觉主模型请求的前置依赖，因此主模型仍须等到
        摘要完成后才能安全发送；这里直接消费 stream，避免额外创建
        ProviderResponse 完整响应对象，并保留适配器的首事件/重试语义。
        """

        policy = self._vision_policy
        if not policy.enabled:
            raise ModelRoutingError(
                "vision summary fallback is disabled",
                safe_message="当前主模型无法识别图片，视觉回退已关闭",
            )
        try:
            vision_selection = self.resolve(
                "vision",
                required_capabilities=("streaming", "vision"),
            )
        except ModelRoutingError as exc:
            raise ModelRoutingError(
                str(exc),
                safe_message="当前主模型无法识别图片，请先配置支持图片的视觉模型",
            ) from exc
        vision_model = vision_selection.requested_model or vision_selection.primary.selected_model
        try:
            vision_request = build_vision_request(
                request,
                model=vision_model,
                prompt=policy.prompt,
                max_tokens=policy.max_tokens,
            )
        except (TypeError, ValueError) as exc:
            raise ModelRoutingError(
                "vision request contains an unsupported image part",
                safe_message="图片格式不受支持，无法交给视觉模型处理",
            ) from exc
        summary_parts: list[str] = []
        try:
            async for event in self.stream(
                vision_request,
                task="vision",
                context=context,
                required_capabilities=("streaming", "vision"),
                cancel_event=cancel_event,
            ):
                if isinstance(event, TextDelta):
                    summary_parts.append(event.delta)
        except asyncio.CancelledError:
            raise
        except ModelRoutingError:
            raise
        except ProviderAdapterError as exc:
            if exc.category == "cancelled":
                raise
            raise ModelRoutingError(
                f"vision summary failed: {exc.category}",
                safe_message="视觉模型暂时不可用，请检查视觉渠道后重试",
            ) from exc
        except Exception as exc:
            # 自定义 runtime 也可能把底层异常直接抛出；视觉回退边界不能
            # 把其文本（可能含 URL、请求头或密钥）传给对话层。
            raise ModelRoutingError(
                "vision summary failed: unknown",
                safe_message="视觉模型暂时不可用，请检查视觉渠道后重试",
            ) from exc
        summary = "".join(summary_parts).strip()
        if not summary:
            raise ModelRoutingError(
                "vision model returned an empty summary",
                safe_message="视觉模型未返回有效图片摘要，请稍后重试",
            )
        return summary[: policy.max_summary_chars]

    async def stream(
        self,
        request: ChatRequest,
        *,
        task: str = "dialogue",
        context: ConversationContext | Mapping[str, Any] | None = None,
        model: str | None = None,
        channel_id: str | None = None,
        protocol: str | None = None,
        required_capabilities: Iterable[str] = (),
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        task_name = str(task or "dialogue").strip() or "dialogue"
        route_capabilities = tuple(required_capabilities)
        # 直接调用 vision 任务时必须显式选择具备视觉能力的渠道；视觉回退
        # 只在普通对话任务中发生，避免递归生成摘要。
        if task_name.casefold() == "vision":
            route_capabilities = (*route_capabilities, "vision")
        selection = self.resolve(
            task_name,
            model=model,
            channel_id=channel_id,
            protocol=protocol,
            required_capabilities=route_capabilities,
        )
        image_request = request_has_inline_images(request)
        if image_request:
            try:
                validate_inline_images(request)
            except (TypeError, ValueError) as exc:
                raise ModelRoutingError(
                    "vision request contains an unsupported image part",
                    safe_message="图片格式不受支持，无法交给模型处理",
                ) from exc
        vision_summary: str | None = None
        vision_summary_task: asyncio.Task[str] | None = None

        def start_vision_summary_task() -> asyncio.Task[str]:
            task = asyncio.create_task(
                self._generate_vision_summary(
                    request,
                    context=context,
                    cancel_event=cancel_event,
                )
            )

            def consume_summary_task(completed: asyncio.Task[str]) -> None:
                if completed.cancelled():
                    return
                try:
                    completed.exception()
                except BaseException:
                    # 读取异常本身只用于避免事件循环告警；真正等待
                    # 任务的路径仍会按原边界向调用方传播。
                    return

            task.add_done_callback(consume_summary_task)
            owner_task = asyncio.current_task()
            if owner_task is not None:

                def cancel_summary_on_owner_done(
                    _completed: asyncio.Task[object],
                    summary_task: asyncio.Task[str] = task,
                ) -> None:
                    if not summary_task.done():
                        summary_task.cancel()

                owner_task.add_done_callback(cancel_summary_on_owner_done)
            return task

        # 视觉摘要是非视觉主模型的必要前置输入；先准备主渠道运行时，
        # 再启动视觉流，让摘要网络等待与后续主请求建立连接重叠。
        # 主模型仍只会收到已完成且有界的摘要，不会收到未完成的占位内容。
        if image_request and task_name.casefold() != "vision":
            primary_runtime = self._make_runtime(selection.primary)
            primary_supports_vision = selection.primary.supports("vision")
            primary_checker = getattr(primary_runtime, "supports", None)
            if callable(primary_checker):
                try:
                    primary_supports_vision = bool(primary_checker("vision"))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    primary_supports_vision = False
            if not primary_supports_vision:
                vision_summary_task = start_vision_summary_task()
                # 让视觉流先进入其首个 await；随后主模型请求可以与视觉
                # 摘要的网络等待重叠一个事件循环切片。
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError:
                    vision_summary_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await vision_summary_task
                    vision_summary_task = None
                    raise
        # 只有当前渠道的有效内容事件才会锁定回退；适配器为正常收尾
        # 合成的 TurnFinished 不应把空流误判为已开始响应。
        # 没有路由模型约束时，请求体通常使用首选频道的默认模型。回退到
        # 不同模型的频道前，复制请求并切换为该频道的默认模型；显式 model
        # 或 routing.model 则保持固定，避免覆盖调用方的模型选择。
        use_channel_models = (
            not selection.requested_model and request.model == selection.primary.selected_model
        )
        correlation = (
            getattr(context, "turn_id", "")
            if isinstance(context, ConversationContext)
            else context.get("turn_id", "")
            if isinstance(context, Mapping)
            else ""
        ) or f"model-call-{uuid.uuid4().hex}"
        for attempt, channel in enumerate(selection.channels, start=1):
            runtime = self._make_runtime(channel)
            attempt_request = request
            runtime_vision_support = getattr(runtime, "supports", None)
            channel_supports_vision = channel.supports("vision")
            if callable(runtime_vision_support):
                try:
                    channel_supports_vision = bool(runtime_vision_support("vision"))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    channel_supports_vision = False
            if image_request and task_name.casefold() != "vision" and not channel_supports_vision:
                # 只有真正切换到不支持图片的渠道时才生成摘要。首选渠道若
                # 已声明 vision 能力，应直接保留原图，避免无意义的额外请求
                # 和视觉渠道故障阻断本来可用的主模型。
                if vision_summary is None:
                    if vision_summary_task is not None:
                        try:
                            vision_summary = await vision_summary_task
                        finally:
                            vision_summary_task = None
                    else:
                        vision_summary = await self._generate_vision_summary(
                            request,
                            context=context,
                            cancel_event=cancel_event,
                        )
                try:
                    attempt_request = strip_images_and_attach_summary(
                        request,
                        vision_summary,
                        max_chars=self._vision_policy.max_summary_chars,
                    )
                except (TypeError, ValueError) as exc:
                    raise ModelRoutingError(
                        "failed to merge vision summary into main request",
                        safe_message="视觉摘要无法合并到主模型请求",
                    ) from exc
            if selection.requested_model and attempt_request.model != selection.requested_model:
                # 路由规范指定的模型优先于调用方请求体中的旧模型；否则
                # 直接调用 ModelRouter 时会选对渠道却把旧模型名发送出去。
                attempt_request = replace(attempt_request, model=selection.requested_model)
            elif use_channel_models:
                channel_model = channel.selected_model
                if channel_model and channel_model != attempt_request.model:
                    attempt_request = replace(attempt_request, model=channel_model)
            channel_event_emitted = False
            pending_metadata: list[AdapterMetadata] = []
            pending_finished: TurnFinished | None = None
            event_count = 0
            started_at = perf_counter()
            audit_handle: ApiCallAuditHandle | None = None
            audit_first_event = False
            audit_text_parts: list[str] = []
            audit_reasoning_parts: list[str] = []
            audit_murmur_parts: list[str] = []
            audit_tool_calls: dict[int, dict[str, object]] = {}
            audit_usage: dict[str, int] = {}
            audit_finish_reason = "stop"
            audit_metadata: dict[str, object] = {
                "task": task_name,
                "correlation_id": str(correlation),
                "attempt": attempt,
                "requested_channel_id": str(channel_id or ""),
                "requested_protocol": str(protocol or ""),
            }
            repository = self._audit_repository
            if repository is not None:
                try:
                    provider_name = str(getattr(runtime, "provider", "unknown") or "unknown")
                    channel_info = channel.as_mapping(redact_api_key=True)
                    audit_handle = await repository.astart(
                        request_id=f"model-{uuid.uuid4().hex}",
                        mode=(
                            context.mode
                            if isinstance(context, ConversationContext)
                            else str(context.get("mode", "direct"))
                            if isinstance(context, Mapping)
                            else "direct"
                        ),
                        profile_id=(
                            context.profile_id
                            if isinstance(context, ConversationContext)
                            else str(context.get("profile_id", ""))
                            if isinstance(context, Mapping)
                            else ""
                        ),
                        session_id=(
                            context.session_id
                            if isinstance(context, ConversationContext)
                            else str(context.get("session_id", ""))
                            if isinstance(context, Mapping)
                            else ""
                        ),
                        turn_id=str(correlation),
                        generation_id=(
                            context.generation_id
                            if isinstance(context, ConversationContext)
                            else int(context.get("generation_id", 0) or 0)
                            if isinstance(context, Mapping)
                            else 0
                        ),
                        attempt=attempt,
                        provider=provider_name,
                        protocol=channel.protocol,
                        channel_id=channel.id,
                        channel_name=channel.id,
                        channel_info=channel_info,
                        requested_model=attempt_request.model,
                        input_payload=_audit_request_payload(attempt_request),
                        metadata=audit_metadata,
                    )
                    key = self._audit_context_key(context)
                    if key is not None:
                        with self._route_lock:
                            self._audit_handles[key] = audit_handle
                except (OSError, RuntimeError, TypeError, ValueError):
                    # 审计是旁路能力；数据库故障不能阻断模型请求。
                    audit_handle = None
                    logger.debug("api audit start failed", exc_info=True)
            log_event(
                logger,
                "model.call.started",
                component="model_router",
                status="started",
                correlation_id=correlation,
                operation_id=correlation,
                fields={
                    "task": task_name,
                    "attempt": attempt,
                    "provider": str(getattr(runtime, "provider", "unknown") or "unknown"),
                    "channel_id": channel.id,
                    "protocol": channel.protocol,
                    "model": attempt_request.model,
                    "tool_count": len(attempt_request.tools),
                    "has_images": image_request,
                },
            )

            def log_failure(error: ProviderAdapterError) -> None:
                log_event(
                    logger,
                    "model.call.failed",
                    component="model_router",
                    status="failed",
                    level=logging.WARNING,
                    correlation_id=correlation,
                    operation_id=correlation,
                    duration_ms=(perf_counter() - started_at) * 1000.0,
                    reason_code=error.category,
                    fields={
                        "task": task_name,
                        "attempt": attempt,
                        "provider": str(getattr(runtime, "provider", "unknown") or "unknown"),
                        "channel_id": channel.id,
                        "protocol": channel.protocol,
                        "category": error.category,
                        "retryable": error.retryable,
                        "before_first_event": error.before_first_event,
                        "event_count": event_count,
                    },
                )

            async def fail_audit(
                error_type: str,
                error_message: str,
                *,
                status: str = "failed",
            ) -> None:
                if audit_handle is None:
                    return
                try:
                    await audit_handle.aupdate(
                        output_text="".join(audit_text_parts),
                        output_payload={
                            "reasoning": "".join(audit_reasoning_parts),
                            "murmur": "".join(audit_murmur_parts),
                            "event_count": event_count,
                        },
                        usage=dict(audit_usage),
                        tool_calls=tuple(audit_tool_calls.values()),
                        metadata=audit_metadata,
                    )
                    await audit_handle.afail(
                        error_type=error_type,
                        error_message=error_message,
                        status=status,
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("api audit failure update failed", exc_info=True)

            try:
                async for event in runtime.stream(
                    attempt_request, context=context, cancel_event=cancel_event
                ):
                    event_count += 1
                    if isinstance(event, TurnFinished):
                        audit_usage.update(dict(event.usage))
                        audit_finish_reason = event.finish_reason
                        if event.response_model:
                            audit_metadata["response_model"] = event.response_model
                        audit_metadata.update(dict(event.metadata))
                        # 已确认内容的渠道保持原有实时终止语义；空渠道的
                        # 合成终止事件则暂存，避免抢先结束后续备用渠道。
                        if channel_event_emitted:
                            yield event
                        else:
                            pending_finished = event
                        continue
                    if isinstance(event, AdapterMetadata):
                        # 元数据可能在首段正文之后到达（例如代理直到收尾
                        # 才补充真实模型 ID 或缓存计时）。审计必须先合并，
                        # 但只有首段正文之前的元数据需要延迟发布以保留事件顺序。
                        audit_metadata.update(dict(event.values))
                        if not channel_event_emitted:
                            pending_metadata.append(event)
                            continue
                        yield event
                        continue
                    if (
                        isinstance(event, TextDelta)
                        and not event.delta.strip()
                        and not channel_event_emitted
                    ):
                        # 某些 SSE 实现会先发换行/空格占位；它既不是首字，
                        # 也不能进入备用渠道的聚合正文，避免回退结果带入空白前缀。
                        continue
                    content_event = (
                        (
                            isinstance(event, (ReasoningDelta, MurmurDelta))
                            and bool(event.delta.strip())
                        )
                        or (
                            isinstance(event, ToolCallDelta)
                            and bool(event.call_id or event.identity or event.arguments_delta)
                        )
                        or (isinstance(event, TextDelta) and bool(event.delta.strip()))
                    )
                    if not channel_event_emitted and content_event:
                        channel_event_emitted = True
                        for metadata_event in pending_metadata:
                            yield metadata_event
                        pending_metadata.clear()
                    if (
                        audit_handle is not None
                        and not audit_first_event
                        and isinstance(event, TextDelta)
                        and bool(event.delta.strip())
                    ):
                        try:
                            first_char = next(
                                (character for character in event.delta if not character.isspace()),
                                "",
                            )
                            await audit_handle.amark_first_token(
                                occurred_at=event.occurred_at,
                                first_char=first_char,
                            )
                        except (OSError, RuntimeError, TypeError, ValueError):
                            logger.debug("api audit first event update failed", exc_info=True)
                        audit_first_event = True
                    if isinstance(event, TextDelta):
                        audit_text_parts.append(event.delta)
                    elif isinstance(event, ReasoningDelta):
                        audit_reasoning_parts.append(event.delta)
                    elif isinstance(event, MurmurDelta):
                        audit_murmur_parts.append(event.delta)
                    elif isinstance(event, ToolCallDelta):
                        item = audit_tool_calls.setdefault(
                            int(event.index),
                            {
                                "index": int(event.index),
                                "call_id": "",
                                "identity": "",
                                "arguments": "",
                            },
                        )
                        if event.call_id:
                            item["call_id"] = event.call_id
                        if event.identity:
                            item["identity"] = event.identity
                        item["arguments"] = str(item.get("arguments", "")) + event.arguments_delta
                    yield event
                if not channel_event_emitted:
                    # 空流通常意味着代理返回了空 SSE 或协议端提前断开；
                    # 不应把它标成成功，否则既不会尝试备用渠道，也会让
                    # 用户看到无响应的“完成”状态。
                    empty_error = ProviderAdapterError(
                        "provider returned no events",
                        category="protocol",
                        retryable=True,
                        before_first_event=True,
                    )
                    await fail_audit("protocol", "provider returned no events")
                    self._mark_failure(channel.id, empty_error)
                    if channel is selection.channels[-1]:
                        raise empty_error
                    log_failure(empty_error)
                    continue
                if pending_finished is not None:
                    yield pending_finished
                self._mark_success(channel.id)
                finalized_usage = dict(audit_usage)
                response_model = (
                    pending_finished.response_model if pending_finished is not None else ""
                ) or str(
                    audit_metadata.get("response_model", audit_metadata.get("model", "")) or ""
                )
                cache_read, cache_write, cache_read_ms, cache_write_ms, cache_info = (
                    _audit_cache_fields(finalized_usage, audit_metadata)
                )
                audit_record = None
                if audit_handle is not None:
                    try:
                        audit_record = await audit_handle.afinish(
                            output_text="".join(audit_text_parts),
                            output_payload={
                                "reasoning": "".join(audit_reasoning_parts),
                                "murmur": "".join(audit_murmur_parts),
                                "event_count": event_count,
                            },
                            response_model=response_model,
                            usage=finalized_usage,
                            finish_reason=audit_finish_reason,
                            cache_read_tokens=cache_read,
                            cache_write_tokens=cache_write,
                            cache_read_duration_ms=cache_read_ms,
                            cache_write_duration_ms=cache_write_ms,
                            cache_info=cache_info,
                            tool_calls=tuple(
                                sanitize_audit_value(item) for item in audit_tool_calls.values()
                            ),
                        )
                    except (OSError, RuntimeError, TypeError, ValueError):
                        logger.debug("api audit completion update failed", exc_info=True)
                log_event(
                    logger,
                    "model.call.completed",
                    component="model_router",
                    status="completed",
                    correlation_id=correlation,
                    operation_id=correlation,
                    duration_ms=(perf_counter() - started_at) * 1000.0,
                    reason_code="ok",
                    fields={
                        "task": task_name,
                        "attempt": attempt,
                        "provider": str(getattr(runtime, "provider", "unknown") or "unknown"),
                        "channel_id": channel.id,
                        "protocol": channel.protocol,
                        "model": attempt_request.model,
                        "event_count": event_count,
                        "response_model": response_model,
                        "time_to_first_token_ms": (
                            audit_record.time_to_first_token_ms
                            if audit_record is not None
                            else None
                        ),
                        "cache_read_tokens": cache_read,
                        "cache_write_tokens": cache_write,
                        "tool_call_count": len(audit_tool_calls),
                    },
                )
                return
            except asyncio.CancelledError:
                await fail_audit("cancelled", "model request cancelled", status="cancelled")
                log_event(
                    logger,
                    "model.call.cancelled",
                    component="model_router",
                    status="cancelled",
                    level=logging.INFO,
                    correlation_id=correlation,
                    operation_id=correlation,
                    duration_ms=(perf_counter() - started_at) * 1000.0,
                    reason_code="cancelled",
                    fields={
                        "task": task_name,
                        "attempt": attempt,
                        "provider": str(getattr(runtime, "provider", "unknown") or "unknown"),
                        "channel_id": channel.id,
                    },
                )
                raise
            except ProviderAdapterError as error:
                error = classify_exception(
                    error,
                    before_first_event=not channel_event_emitted,
                )
                self._mark_failure(channel.id, error)
                await fail_audit(error.category, error.safe_message)
                log_failure(error)
                if (
                    error.category == "cancelled"
                    or channel_event_emitted
                    or not error.before_first_event
                ):
                    raise
                if channel is selection.channels[-1]:
                    raise
                if not error.retryable and error.category not in {
                    "network",
                    "timeout",
                    "rate_limit",
                    "server",
                }:
                    raise
                continue
            except Exception as raw_error:
                # 自定义 runtime 可能没有复用 ProviderAdapterRuntime；
                # 仍将网络/超时异常统一分类，确保首事件前可以走渠道回退，
                # 同时把未知异常转换为不泄露供应商正文的错误。
                error = classify_exception(
                    raw_error,
                    before_first_event=not channel_event_emitted,
                )
                self._mark_failure(channel.id, error)
                await fail_audit(error.category, error.safe_message)
                log_failure(error)
                if (
                    error.category == "cancelled"
                    or channel_event_emitted
                    or not error.before_first_event
                ):
                    raise error from raw_error
                if channel is selection.channels[-1]:
                    raise error from raw_error
                if not error.retryable and error.category not in {
                    "network",
                    "timeout",
                    "rate_limit",
                    "server",
                }:
                    raise error from raw_error
                continue
        if vision_summary_task is not None:
            vision_summary_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await vision_summary_task
        raise ModelRoutingError(f"all channels failed for task: {selection.task}")

    async def test_connection(
        self,
        channel: ChannelConfig | Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> Mapping[str, object]:
        """以最小请求测试一个渠道，并返回脱敏的用户状态。

        连接测试不复用对话路由，也不执行工具调用，避免用户只是点击“测试”
        时触发桌面副作用。调用方可以传入尚未注册的 ``ChannelConfig`` 映射；
        映射中的 API Key 只在适配器内部使用，返回值不会包含它、请求地址或
        提供方原始响应。
        """

        try:
            config = (
                channel if isinstance(channel, ChannelConfig) else channel_from_mapping(channel)
            )
        except Exception:
            return {
                "status": "unavailable",
                "ready": False,
                "reason": "configuration",
                "message": "模型连接配置不完整",
            }
        selected_model = str(model or config.selected_model or "").strip()
        base_result: dict[str, object] = {
            "channel_id": config.id,
            "protocol": config.protocol,
            "model": selected_model,
        }
        if not config.enabled or not config.base_url or not selected_model:
            return {
                **base_result,
                "status": "unavailable",
                "ready": False,
                "reason": "configuration",
                "message": "模型连接配置不完整",
            }
        request = ChatRequest(
            model=selected_model,
            messages=(ChatMessage("user", "连接测试"),),
            temperature=0.0,
            max_tokens=1,
            tools=(),
            tool_choice="none",
        )
        context = ConversationContext(
            "system",
            "model-test",
            f"model-test-{uuid.uuid4().hex}",
            0,
        )
        probe_started = perf_counter()
        log_event(
            logger,
            "model.probe.started",
            component="model_router",
            status="started",
            correlation_id=context.turn_id,
            operation_id=context.turn_id,
            fields={
                "channel_id": config.id,
                "protocol": config.protocol,
                "model": selected_model,
                "tool_count": 0,
            },
        )
        events = 0
        # 配置向导可能正在测试同一个渠道 ID 的新地址/模型；探测必须使用
        # 新配置，不能误取对话运行时缓存中的旧适配器。探测运行时始终独立
        # 创建并在 finally 中关闭，不污染下一轮对话的连接池。
        probe_runtime: ProviderAdapterRuntime | None = None
        probe_audit: ApiCallAuditHandle | None = None
        probe_text: list[str] = []
        probe_usage: dict[str, int] = {}
        probe_response_model = ""
        try:
            probe_runtime = self._create_runtime(config)
            if self._audit_repository is not None:
                try:
                    probe_audit = await self._audit_repository.astart(
                        request_id=f"probe-{uuid.uuid4().hex}",
                        mode="direct",
                        profile_id=context.profile_id,
                        session_id=context.session_id,
                        turn_id=context.turn_id,
                        generation_id=context.generation_id,
                        kind="probe",
                        operation="model.probe",
                        provider=str(getattr(probe_runtime, "provider", "unknown") or "unknown"),
                        protocol=config.protocol,
                        channel_id=config.id,
                        channel_name=config.id,
                        channel_info=config.as_mapping(redact_api_key=True),
                        requested_model=selected_model,
                        input_payload=_audit_request_payload(request),
                        metadata={"probe": True},
                    )
                except (OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("model probe audit start failed", exc_info=True)
            async for _event in probe_runtime.stream(request, context=context):
                # 适配器会在所有正常结束路径发出 ``TurnFinished``，包括
                # 供应商返回空 SSE 的情况。连接探测必须要求至少一个真实
                # 提供方事件，否则空响应会被误报为“连接通过”。
                if not isinstance(_event, TurnFinished):
                    events += 1
                    if isinstance(_event, TextDelta):
                        probe_text.append(_event.delta)
                        if probe_audit is not None and probe_audit.record.first_token_at is None:
                            await probe_audit.amark_first_token(
                                occurred_at=_event.occurred_at,
                                first_char=_event.delta,
                            )
                    elif isinstance(_event, (ReasoningDelta, MurmurDelta)):
                        probe_text.append(_event.delta)
                else:
                    probe_usage.update(dict(_event.usage))
                    if _event.response_model:
                        probe_response_model = _event.response_model
            if probe_audit is not None:
                await probe_audit.afinish(
                    output_text="".join(probe_text),
                    response_model=probe_response_model,
                    usage=probe_usage,
                    finish_reason="probe",
                )
        except asyncio.CancelledError:
            if probe_audit is not None:
                await probe_audit.afail(
                    error_type="cancelled",
                    error_message="model probe cancelled",
                    status="cancelled",
                )
            log_event(
                logger,
                "model.probe.cancelled",
                component="model_router",
                status="cancelled",
                correlation_id=context.turn_id,
                operation_id=context.turn_id,
                duration_ms=(perf_counter() - probe_started) * 1000.0,
                reason_code="cancelled",
                fields={"channel_id": config.id, "protocol": config.protocol},
            )
            raise
        except ProviderAdapterError as error:
            if probe_audit is not None:
                await probe_audit.afail(
                    error_type=str(getattr(error, "category", "unknown") or "unknown"),
                    error_message=getattr(error, "safe_message", "model probe failed"),
                )
            raw_reason = str(getattr(error, "category", "unknown") or "unknown").strip().lower()
            reason = raw_reason if raw_reason in _CONNECTION_FAILURE_REASONS else "unknown"
            log_event(
                logger,
                "model.probe.failed",
                component="model_router",
                status="failed",
                level=logging.WARNING,
                correlation_id=context.turn_id,
                operation_id=context.turn_id,
                duration_ms=(perf_counter() - probe_started) * 1000.0,
                reason_code=reason,
                fields={
                    "channel_id": config.id,
                    "protocol": config.protocol,
                    "category": reason,
                },
            )
            return {
                **base_result,
                "status": "unavailable",
                "ready": False,
                "reason": reason,
                "message": _connection_failure_message(error),
            }
        except Exception as error:
            if probe_audit is not None:
                await probe_audit.afail(
                    error_type=type(error).__name__,
                    error_message="model probe failed",
                )
            # 适配器注册/传输层的未知异常只能映射到固定提示，不能把异常文本
            # 或其可能包含的 URL、请求头和密钥带到模型向导。
            log_event(
                logger,
                "model.probe.failed",
                component="model_router",
                status="failed",
                level=logging.WARNING,
                correlation_id=context.turn_id,
                operation_id=context.turn_id,
                duration_ms=(perf_counter() - probe_started) * 1000.0,
                reason_code="unknown",
                fields={
                    "channel_id": config.id,
                    "protocol": config.protocol,
                    "category": "unknown",
                    "error_type": type(error).__name__,
                },
            )
            return {
                **base_result,
                "status": "unavailable",
                "ready": False,
                "reason": "unknown",
                "message": _connection_failure_message(error),
            }
        finally:
            if probe_runtime is not None:
                try:
                    await probe_runtime.aclose()
                except asyncio.CancelledError:
                    raise
                except Exception as cleanup_error:
                    # 清理失败不覆盖已确定的探测结果，但必须留下脱敏终态，
                    # 否则持久连接泄漏只能表现为后续请求随机失败。
                    log_event(
                        logger,
                        "model.probe.cleanup_failed",
                        component="model_router",
                        status="degraded",
                        level=logging.WARNING,
                        correlation_id=context.turn_id,
                        operation_id=context.turn_id,
                        duration_ms=(perf_counter() - probe_started) * 1000.0,
                        reason_code="cleanup_failed",
                        fields={
                            "channel_id": config.id,
                            "protocol": config.protocol,
                            "error_type": type(cleanup_error).__name__,
                        },
                    )
        if events <= 0:
            if probe_audit is not None:
                await probe_audit.afail(
                    error_type="protocol",
                    error_message="provider returned no events",
                )
            log_event(
                logger,
                "model.probe.failed",
                component="model_router",
                status="failed",
                level=logging.WARNING,
                correlation_id=context.turn_id,
                operation_id=context.turn_id,
                duration_ms=(perf_counter() - probe_started) * 1000.0,
                reason_code="protocol",
                fields={
                    "channel_id": config.id,
                    "protocol": config.protocol,
                    "category": "protocol",
                    "event_count": events,
                },
            )
            return {
                **base_result,
                "status": "unavailable",
                "ready": False,
                "reason": "protocol",
                "message": "模型未返回有效响应",
            }
        log_event(
            logger,
            "model.probe.completed",
            component="model_router",
            status="completed",
            correlation_id=context.turn_id,
            operation_id=context.turn_id,
            duration_ms=(perf_counter() - probe_started) * 1000.0,
            reason_code="ok",
            fields={
                "channel_id": config.id,
                "protocol": config.protocol,
                "model": selected_model,
                "event_count": events,
            },
        )
        return {
            **base_result,
            "status": "available",
            "ready": True,
            "reason": "ready",
            "message": "模型连接测试通过",
            "events": events,
        }

    # ``probe_connection`` 是给宿主接入层的语义别名；两者共享同一请求和
    # 脱敏回执契约，避免不同 GUI 各自实现连接探测。
    probe_connection = test_connection

    async def complete(
        self,
        request: ChatRequest,
        *,
        task: str = "dialogue",
        context: ConversationContext | Mapping[str, Any] | None = None,
        model: str | None = None,
        channel_id: str | None = None,
        protocol: str | None = None,
        required_capabilities: Iterable[str] = (),
        cancel_event: asyncio.Event | None = None,
    ) -> Any:
        response = ProviderResponse()
        async for event in self.stream(
            request,
            task=task,
            context=context,
            model=model,
            channel_id=channel_id,
            protocol=protocol,
            required_capabilities=required_capabilities,
            cancel_event=cancel_event,
        ):
            response = response.add(event)
        return response.finalized()

    def _mark_success(self, channel_id: str) -> None:
        with self._route_lock:
            health = self._health.setdefault(channel_id, _ChannelHealth())
            health.failures = 0
            health.last_category = ""
            health.last_error = ""
            health.cooldown_until = 0.0

    def _mark_failure(self, channel_id: str, error: BaseException) -> None:
        with self._route_lock:
            health = self._health.setdefault(channel_id, _ChannelHealth())
            category = str(getattr(error, "category", "unknown") or "unknown").strip().lower()
            # 适配器异常通常已经带有 safe_message，但自定义适配器可能错误地
            # 把 URL/令牌放入文本；健康快照只保留固定分类提示。
            health.last_category = category
            health.last_error = _connection_failure_message(error)
            if category == "cancelled":
                # 用户主动取消不代表渠道故障；保留最近一次取消状态，但不
                # 增加失败次数，也不改变已有的冷却窗口，避免取消请求污染
                # 后续视觉回退和主模型重试的健康判断。
                return
            health.failures += 1
            retry_after = getattr(error, "retry_after_seconds", None)
            try:
                requested = max(0.0, float(retry_after)) if retry_after is not None else 0.0
            except (TypeError, ValueError):
                requested = 0.0
            exponential = _HEALTH_COOLDOWN_BASE_SECONDS * 2 ** max(0, health.failures - 1)
            health.cooldown_until = self._clock() + min(
                _HEALTH_COOLDOWN_MAX_SECONDS,
                max(exponential, requested),
            )

    def health(
        self, channel_id: str | None = None
    ) -> Mapping[str, Mapping[str, Any]] | Mapping[str, Any] | None:
        with self._route_lock:
            if channel_id is not None:
                health = self._health.get(str(channel_id).strip())
                if health is None:
                    return None
                return {
                    "failures": health.failures,
                    "last_category": health.last_category,
                    "last_error": health.last_error,
                    "cooldown_until": health.cooldown_until,
                }
            return {
                key: {
                    "failures": value.failures,
                    "last_category": value.last_category,
                    "last_error": value.last_error,
                    "cooldown_until": value.cooldown_until,
                }
                for key, value in self._health.items()
            }

    async def aclose(self) -> None:
        with self._route_lock:
            runtimes = tuple(self._runtimes.values())
            self._runtimes.clear()
            self._audit_handles.clear()
        first_error: BaseException | None = None
        for runtime in runtimes:
            try:
                await runtime.aclose()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


ModelRouterService = ModelRouter


__all__ = [
    "ModelRouter",
    "ModelRouterService",
    "ModelRoutingError",
    "RouteSelection",
    "RouteSpec",
    "VisionSummaryPolicy",
]
