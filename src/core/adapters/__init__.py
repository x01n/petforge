"""模型适配器边界；具体协议位于 ``core.adapters.direct`` 和 ``providers``。"""

from .protocols import SUPPORTED_PROTOCOLS
from .providers import (
    ClaudeMessagesAdapter,
    ClaudeProviderAdapter,
    GeminiGenerateAdapter,
    GeminiProviderAdapter,
    OpenAIProviderAdapter,
    OpenAIResponsesAdapter,
    ProviderAdapterRegistry,
    create_provider_adapter,
    provider_registry,
)
from .types import (
    AdapterRuntime,
    ProviderAdapter,
    ProviderAdapterLike,
    ProviderAdapterRuntime,
    ProviderEvent,
    ProviderResponse,
    ProviderRuntime,
)

__all__ = [
    "AdapterRuntime",
    "ProviderAdapter",
    "ProviderAdapterLike",
    "ProviderAdapterRuntime",
    "ProviderEvent",
    "ProviderResponse",
    "ProviderRuntime",
    "ClaudeMessagesAdapter",
    "ClaudeProviderAdapter",
    "GeminiGenerateAdapter",
    "GeminiProviderAdapter",
    "OpenAIProviderAdapter",
    "OpenAIResponsesAdapter",
    "ProviderAdapterRegistry",
    "create_provider_adapter",
    "provider_registry",
    "SUPPORTED_PROTOCOLS",
]
