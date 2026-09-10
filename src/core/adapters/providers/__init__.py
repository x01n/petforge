"""模型供应商适配器。

``core.adapters.direct`` 保存低层事件和生命周期契约；本包只负责把具体
供应商的 HTTP/SSE 线协议转换成这些契约。适配器不记录原始请求或响应正文。
"""

from .claude import ClaudeMessagesAdapter, ClaudeProviderAdapter
from .gemini import GeminiGenerateAdapter, GeminiProviderAdapter
from .openai import (
    OpenAIProviderAdapter,
    OpenAIResponsesAdapter,
    OpenAIResponsesSSEAdapter,
)
from .registry import ProviderAdapterRegistry, create_provider_adapter, provider_registry

__all__ = [
    "ClaudeMessagesAdapter",
    "ClaudeProviderAdapter",
    "GeminiGenerateAdapter",
    "GeminiProviderAdapter",
    "OpenAIProviderAdapter",
    "OpenAIResponsesAdapter",
    "OpenAIResponsesSSEAdapter",
    "ProviderAdapterRegistry",
    "create_provider_adapter",
    "provider_registry",
]
