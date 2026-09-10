"""模型适配器协议白名单。

该模块只保存稳定的协议身份，不依赖配置、服务或具体供应商实现；配置加载、
路由和适配器注册表都从这里读取同一份集合，避免协议列表漂移。
"""

from __future__ import annotations

SUPPORTED_PROTOCOLS = frozenset(
    {
        "openai",
        "openai_chat",
        "openai_responses",
        "anthropic",
        "anthropic_messages",
        "claude",
        "gemini",
        "gemini_generate",
        "google_gemini",
        "ollama_chat",
    }
)

__all__ = ["SUPPORTED_PROTOCOLS"]
