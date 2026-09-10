"""HTTP 直连模型适配器。"""

from .base import (
    AdapterRuntime,
    ProviderAdapter,
    ProviderAdapterLike,
    ProviderAdapterRuntime,
    ProviderEvent,
    ProviderResponse,
    ProviderRuntime,
    ensure_context,
)
from .errors import (
    AdapterCancelled,
    AdapterConfigurationError,
    AdapterProtocolError,
    ErrorCategory,
    ErrorContext,
    ProviderAdapterError,
    classify_exception,
)
from .events import AdapterMetadata, ToolCallDelta
from .openai_chat_sse import (
    OpenAIChatAdapter,
    OpenAIChatSSE,
    OpenAIChatSSEAdapter,
    OpenAIChatSseAdapter,
)
from .retry import RetryPolicy

__all__ = [
    "AdapterRuntime",
    "AdapterCancelled",
    "AdapterConfigurationError",
    "AdapterMetadata",
    "AdapterProtocolError",
    "ErrorCategory",
    "ErrorContext",
    "OpenAIChatAdapter",
    "OpenAIChatSSE",
    "OpenAIChatSSEAdapter",
    "OpenAIChatSseAdapter",
    "ProviderAdapter",
    "ProviderAdapterError",
    "ProviderAdapterLike",
    "ProviderAdapterRuntime",
    "ProviderEvent",
    "ProviderResponse",
    "ProviderRuntime",
    "RetryPolicy",
    "ToolCallDelta",
    "classify_exception",
    "ensure_context",
]
