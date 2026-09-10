"""供应商适配器注册表。

路由只依赖 ``protocol``，不会根据 URL 猜测供应商。别名显式列出并经过同一
注册表创建，便于配置校验、测试替换和未来插件扩展。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from threading import RLock
from typing import Any

from core.adapters.direct.base import ProviderAdapterLike
from core.adapters.direct.errors import AdapterConfigurationError

from .claude import ClaudeProviderAdapter
from .gemini import GeminiProviderAdapter
from .openai import OpenAIProviderAdapter, OpenAIResponsesAdapter

AdapterFactory = Callable[[Any], ProviderAdapterLike]


class ProviderAdapterRegistry:
    """线程安全的协议到构造器映射。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._factories: dict[str, AdapterFactory] = {}

    def register(self, protocol: str, factory: AdapterFactory, *, replace: bool = False) -> None:
        name = str(protocol or "").strip().lower()
        if not name or any(char in name for char in "\r\n\x00"):
            raise ValueError("adapter protocol is invalid")
        if not callable(factory):
            raise TypeError("adapter factory must be callable")
        with self._lock:
            if name in self._factories and not replace:
                raise ValueError(f"duplicate adapter protocol: {name}")
            self._factories[name] = factory

    def unregister(self, protocol: str) -> bool:
        with self._lock:
            return self._factories.pop(str(protocol or "").strip().lower(), None) is not None

    def protocols(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._factories))

    def diagnostics(self) -> tuple[Mapping[str, object], ...]:
        """返回不实例化渠道、不包含端点或密钥的协议能力清单。"""

        with self._lock:
            entries = tuple(sorted(self._factories.items()))
        result: list[Mapping[str, object]] = []
        for protocol, factory in entries:
            raw_capabilities = getattr(factory, "capabilities", ())
            if isinstance(raw_capabilities, str):
                raw_capabilities = (raw_capabilities,)
            try:
                capabilities = tuple(
                    sorted(
                        {
                            str(item or "").strip().lower()
                            for item in raw_capabilities
                            if str(item or "").strip()
                        }
                    )
                )
            except TypeError:
                capabilities = ()
            result.append(
                {
                    "protocol": protocol,
                    "provider": str(getattr(factory, "provider", "unknown") or "unknown")[:64],
                    "capabilities": capabilities,
                }
            )
        return tuple(result)

    def factory(self, protocol: str) -> AdapterFactory | None:
        with self._lock:
            return self._factories.get(str(protocol or "").strip().lower())

    def create(self, channel: Any) -> ProviderAdapterLike:
        protocol = str(getattr(channel, "protocol", "") or "").strip().lower()
        factory = self.factory(protocol)
        if factory is None:
            raise AdapterConfigurationError(f"protocol {protocol} has no registered adapter")
        try:
            adapter = factory(channel)
        except AdapterConfigurationError:
            raise
        except (TypeError, ValueError, RuntimeError) as exc:
            raise AdapterConfigurationError(
                f"protocol {protocol} adapter could not be created"
            ) from exc
        if not hasattr(adapter, "stream") or not callable(adapter.stream):
            raise AdapterConfigurationError(f"protocol {protocol} adapter is invalid")
        return adapter


provider_registry = ProviderAdapterRegistry()
for _protocol, _factory in {
    "openai": OpenAIProviderAdapter,
    "openai_chat": OpenAIProviderAdapter,
    "openai_responses": OpenAIResponsesAdapter,
    "ollama_chat": OpenAIProviderAdapter,
    "gemini": GeminiProviderAdapter,
    "gemini_generate": GeminiProviderAdapter,
    "google_gemini": GeminiProviderAdapter,
    "claude": ClaudeProviderAdapter,
    "anthropic": ClaudeProviderAdapter,
    "anthropic_messages": ClaudeProviderAdapter,
}.items():
    provider_registry.register(_protocol, _factory)


def create_provider_adapter(channel: Any) -> ProviderAdapterLike:
    """按渠道协议创建适配器。"""

    return provider_registry.create(channel)


__all__ = [
    "AdapterFactory",
    "ProviderAdapterRegistry",
    "create_provider_adapter",
    "provider_registry",
]
