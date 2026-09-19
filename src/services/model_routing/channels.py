from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from core.adapters.direct.errors import AdapterConfigurationError
from core.adapters.direct.retry import RetryPolicy
from core.adapters.protocols import SUPPORTED_PROTOCOLS


def _text(value: object, field_name: str, *, required: bool = False, maximum: int = 512) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise AdapterConfigurationError(f"channel {field_name} is required")
    if len(result) > maximum or any(char in result for char in "\r\n\x00"):
        raise AdapterConfigurationError(f"channel {field_name} is invalid")
    return result


def _normalise_url(value: object, *, required: bool) -> str:
    raw = _text(value, "base_url", required=required, maximum=2048)
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise AdapterConfigurationError("channel base_url must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise AdapterConfigurationError("channel base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise AdapterConfigurationError("channel base_url must not contain query or fragment")
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _mapping(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AdapterConfigurationError(f"channel {field_name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _endpoint(value: object, field_name: str) -> str:
    """校验相对路径或显式 HTTP(S) 端点。"""

    endpoint = _text(value, field_name, required=True, maximum=512)
    if endpoint.startswith("/"):
        return endpoint
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme.lower() in {"http", "https"}
        and bool(parsed.netloc)
        and parsed.username is None
        and parsed.password is None
        and not parsed.fragment
    ):
        return endpoint
    raise AdapterConfigurationError(f"channel {field_name} must be an absolute path or URL")


def _string_set(value: object, field_name: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Iterable):
        values = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise AdapterConfigurationError(f"channel {field_name} must be a list or string")
    if any(any(char in item for char in "\r\n\x00") for item in values):
        raise AdapterConfigurationError(f"channel {field_name} contains an unsafe value")
    return frozenset(values)


def _bool(value: object, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0", ""}:
        return False
    raise AdapterConfigurationError(f"channel {field_name} must be a boolean")


@dataclass(frozen=True)
class ChannelConfig:
    """一个模型渠道的显式配置。"""

    id: str
    # 空值表示未显式指定；构造函数会按 protocol > adapter > openai_chat 归一化。
    protocol: str = ""
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    model: str = ""
    models: tuple[str, ...] = ()
    enabled: bool = True
    # ``enable`` 保留为 YAML 兼容字段；读取后以 enabled 为规范值。
    enable: bool | None = None
    priority: int = 100
    # 0 表示由 HTTP 客户端使用其默认超时，不把配置层的“未指定”变成零超时。
    timeout_seconds: float = 0.0
    headers: Mapping[str, str] = field(default_factory=dict)
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({"streaming"}))
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    endpoint: str = "/chat/completions"
    endpoints: Mapping[str, str] = field(default_factory=dict)
    request_body_template: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    adapter: str = ""
    # 渠道可能没有图片理解能力，或“能收图但理解不可靠”；置 true 表示
    # 路由层判定该渠道接收图片后，图片会先由视觉渠道转成文本摘要。
    image_summary_fallback: bool = False
    # 允许该渠道路由到的任务集合；空集表示不参与任何可路由任务。
    tasks: frozenset[str] = field(default_factory=frozenset)
    # 该渠道 API 审计快照中单条消息正文本的截断阈值（字符数）。默认 4000；
    # 超出窗口的值不生效，低于全局默认视为配置错误（不能主动缩小审计口径）。
    audit_text_limit: int = 4000

    def __post_init__(self) -> None:
        channel_id = _text(self.id, "id", required=True)
        raw_protocol = str(self.protocol or "").strip()
        raw_adapter = str(self.adapter or "").strip()
        protocol = _text(
            raw_protocol or raw_adapter or "openai_chat",
            "protocol",
            required=True,
        ).lower()
        if protocol not in SUPPORTED_PROTOCOLS:
            raise AdapterConfigurationError(f"unsupported channel protocol: {protocol}")
        # 空地址保留为“未就绪渠道”，由 ``is_enabled``/路由过滤；这样配置向导
        # 可以先保存渠道标识和协议，再补齐端点而不会构造出隐式地址。
        base_url = _normalise_url(self.base_url, required=False)
        model = _text(self.model, "model", maximum=512)
        models = tuple(_text(item, "models", required=True, maximum=512) for item in self.models)
        if model and model not in models:
            models = (model, *models)
        enabled = _bool(self.enabled, "enabled")
        if self.enable is not None:
            enabled = _bool(self.enable, "enable")
        priority = int(self.priority)
        if priority < -1_000_000 or priority > 1_000_000:
            raise AdapterConfigurationError("channel priority is out of range")
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout < 0 or timeout > 86_400:
            raise AdapterConfigurationError("channel timeout_seconds is out of range")
        headers = _mapping(self.headers, "headers")
        safe_headers: dict[str, str] = {}
        for key, value in headers.items():
            header = _text(key, "header name", required=True, maximum=256)
            header_value = _text(value, "header value", maximum=4096)
            safe_headers[header] = header_value
        endpoint = _endpoint(self.endpoint, "endpoint")
        endpoints = _mapping(self.endpoints, "endpoints")
        normalised_endpoints: dict[str, str] = {}
        for key, value in endpoints.items():
            path = _endpoint(value, "endpoint path")
            normalised_endpoints[str(key).strip()] = path
        retry = (
            self.retry
            if isinstance(self.retry, RetryPolicy)
            else RetryPolicy.from_mapping(self.retry)
        )
        capabilities = {
            item.strip().lower()
            for item in _string_set(self.capabilities, "capabilities")
            if item.strip()
        }
        capabilities.add("streaming")
        object.__setattr__(self, "id", channel_id)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "adapter", protocol)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "api_key", _text(self.api_key, "api_key", maximum=4096))
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "enable", enabled)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "headers", safe_headers)
        object.__setattr__(self, "capabilities", frozenset(capabilities))
        object.__setattr__(self, "retry", retry)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "endpoints", normalised_endpoints)
        object.__setattr__(
            self,
            "request_body_template",
            _mapping(self.request_body_template, "request_body_template"),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        # 图片缺口标志和任务白名单走同一配置文件校验，保证实例化时已安全。非
        # 布尔值按失败关闭处理。
        image_summary_fallback = _bool(self.image_summary_fallback, "image_summary_fallback")
        tasks = {
            str(item).strip().lower()
            for item in _string_set(self.tasks, "tasks")
            if str(item).strip()
        }
        object.__setattr__(self, "image_summary_fallback", image_summary_fallback)
        object.__setattr__(self, "tasks", frozenset(tasks))
        if isinstance(self.audit_text_limit, bool) or not isinstance(self.audit_text_limit, int):
            raise AdapterConfigurationError("channel audit_text_limit must be an integer")
        audit_text_limit = int(self.audit_text_limit)
        if audit_text_limit < 4000 or audit_text_limit > 32768:
            raise AdapterConfigurationError("channel audit_text_limit is out of range")
        object.__setattr__(self, "audit_text_limit", audit_text_limit)

    @property
    def selected_model(self) -> str:
        return self.model or (self.models[0] if self.models else "")

    @property
    def is_enabled(self) -> bool:
        return self.enabled and bool(self.base_url)

    @property
    def is_ready(self) -> bool:
        """是否具备发起对话请求所需的地址和模型标识。"""

        return self.is_enabled and bool(self.selected_model)

    @property
    def readiness_reason(self) -> str:
        """返回不适合发起请求时的脱敏原因。"""

        if not self.enabled:
            return "disabled"
        if not self.base_url:
            return "base_url is missing"
        if not self.selected_model:
            return "model is missing"
        return "ready"

    def supports(self, capability: str) -> bool:
        return str(capability or "").strip().lower() in self.capabilities

    def endpoint_path(self, name: str = "chat") -> str:
        key = str(name or "chat").strip()
        return self.endpoints.get(key, self.endpoint)

    def as_mapping(self, *, redact_api_key: bool = False) -> dict[str, Any]:
        """导出完整渠道配置，供配置中心和诊断快照使用。

        早期版本只导出了基础连接字段，导致使用该方法进行配置编辑/导出时
        丢失自定义 endpoint、请求体模板和元数据。这里保留所有可配置字段，
        并在请求体模板和元数据中递归脱敏，避免 ``redact_api_key=True`` 只
        保护顶层 ``api_key`` 而泄漏嵌套 token。
        """

        def redact(value: object) -> object:
            if isinstance(value, Mapping):
                result: dict[str, Any] = {}
                for raw_key, raw_value in value.items():
                    key = str(raw_key)
                    normalized = key.strip().lower().replace("_", "-")
                    sensitive = (
                        normalized
                        in {
                            "authorization",
                            "proxy-authorization",
                            "cookie",
                            "set-cookie",
                        }
                        or "api-key" in normalized
                        or "token" in normalized
                        or "secret" in normalized
                        or "password" in normalized
                        or "credential" in normalized
                        or "authorization" in normalized
                    )
                    result[key] = "***" if sensitive else redact(raw_value)
                return result
            if isinstance(value, list):
                return [redact(item) for item in value]
            if isinstance(value, tuple):
                return [redact(item) for item in value]
            return value

        headers = dict(self.headers)
        if redact_api_key:
            headers = redact(headers)  # type: ignore[assignment]
        return {
            "id": self.id,
            "protocol": self.protocol,
            "adapter": self.adapter,
            "base_url": self.base_url,
            "api_key": "***" if redact_api_key and self.api_key else self.api_key,
            "model": self.model,
            "models": list(self.models),
            "enabled": self.enabled,
            "enable": self.enabled,
            "priority": self.priority,
            "timeout_seconds": self.timeout_seconds,
            "headers": headers,
            "capabilities": sorted(self.capabilities),
            "retry": {
                "max_attempts": self.retry.max_attempts,
                "initial_delay_seconds": self.retry.initial_delay_seconds,
                "max_delay_seconds": self.retry.max_delay_seconds,
                "backoff_multiplier": self.retry.backoff_multiplier,
                "jitter_ratio": self.retry.jitter_ratio,
            },
            "endpoint": self.endpoint,
            "endpoints": dict(self.endpoints),
            "request_body_template": redact(self.request_body_template)
            if redact_api_key
            else dict(self.request_body_template),
            "metadata": redact(self.metadata) if redact_api_key else dict(self.metadata),
            "image_summary_fallback": self.image_summary_fallback,
            "audit_text_limit": self.audit_text_limit,
            "tasks": sorted(self.tasks),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ChannelConfig:
        if not isinstance(value, Mapping):
            raise AdapterConfigurationError("channel must be a mapping")
        raw = {str(key): item for key, item in value.items()}
        # 与加载器保持同一优先级：显式 protocol > adapter > type > 默认协议。
        protocol = raw.get("protocol")
        if not str(protocol or "").strip():
            protocol = raw.get("adapter")
        if not str(protocol or "").strip():
            protocol = raw.get("type", "openai_chat")
        enabled = raw.get("enabled", raw.get("enable", True))
        models_value = raw.get("models", ())
        if isinstance(models_value, str):
            models_value = (models_value,)
        elif models_value is None:
            models_value = ()
        elif not isinstance(models_value, Iterable) or isinstance(models_value, Mapping):
            raise AdapterConfigurationError("channel models must be a list or string")
        headers = raw.get("headers", raw.get("extra_headers", {}))
        capabilities = raw.get("capabilities", raw.get("features", ()))
        retry = raw.get("retry", None)
        endpoint = raw.get("endpoint", "/chat/completions")
        endpoints = raw.get("endpoints", {})
        return cls(
            id=raw.get("id", ""),
            protocol=protocol,
            base_url=raw.get("base_url", raw.get("url", "")),
            api_key=raw.get("api_key", raw.get("token", "")),
            model=raw.get("model", raw.get("default_model", "")),
            models=tuple(models_value),
            enabled=_bool(enabled, "enabled"),
            enable=_bool(enabled, "enabled"),
            priority=raw.get("priority", 100),
            timeout_seconds=raw.get("timeout_seconds", raw.get("timeout", 0.0)),
            headers=headers,
            capabilities=capabilities,
            retry=RetryPolicy.from_mapping(retry),
            endpoint=endpoint,
            endpoints=endpoints,
            request_body_template=raw.get("request_body_template", {}),
            metadata=raw.get("metadata", raw.get("extra", {})),
            adapter=str(raw.get("adapter", protocol) or ""),
            image_summary_fallback=_bool(
                raw.get("image_summary_fallback", False), "image_summary_fallback"
            ),
            audit_text_limit=int(raw.get("audit_text_limit", 4000)),
            tasks=_parse_tasks(raw.get("tasks")),
        )


def _parse_tasks(value: object) -> frozenset[str]:
    """把渠道级任务白名单收敛为小写字符串集合。"""

    if value is None:
        return frozenset()
    if isinstance(value, str):
        return frozenset({value.strip().lower()}) if value.strip() else frozenset()
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return frozenset(str(item).strip().lower() for item in value if str(item or "").strip())
    raise AdapterConfigurationError("channel tasks must be a list of strings")


def channel_from_mapping(value: Mapping[str, Any]) -> ChannelConfig:
    return ChannelConfig.from_mapping(value)


def load_channels(values: object) -> tuple[ChannelConfig, ...]:
    if values is None:
        return ()
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        raise AdapterConfigurationError("llm.channels must be a list")
    channels = tuple(channel_from_mapping(item) for item in values)
    ids = [channel.id for channel in channels]
    if len(set(ids)) != len(ids):
        raise AdapterConfigurationError("duplicate channel id")
    return channels


def sort_channels(channels: Iterable[ChannelConfig]) -> tuple[ChannelConfig, ...]:
    """按优先级排序；同优先级保留配置顺序。"""

    return tuple(sorted(channels, key=lambda channel: channel.priority))


def select_channels(
    channels: Iterable[ChannelConfig],
    *,
    model: str | None = None,
    required_capabilities: Iterable[str] = (),
    protocol: str | None = None,
) -> tuple[ChannelConfig, ...]:
    requested_model = str(model or "").strip()
    requested_protocol = str(protocol or "").strip().lower()
    required = {
        str(value or "").strip().lower()
        for value in required_capabilities
        if str(value or "").strip()
    }
    selected: list[ChannelConfig] = []
    for channel in channels:
        if not channel.is_ready:
            continue
        if requested_protocol and channel.protocol != requested_protocol:
            continue
        if (
            requested_model
            and requested_model not in channel.models
            and requested_model != channel.model
        ):
            continue
        if required and not required.issubset(channel.capabilities):
            continue
        selected.append(channel)
    return sort_channels(selected)


__all__ = [
    "SUPPORTED_PROTOCOLS",
    "ChannelConfig",
    "channel_from_mapping",
    "load_channels",
    "select_channels",
    "sort_channels",
]
