from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from time import time
from typing import Any

from core.events.types import ConversationContext


@dataclass(frozen=True)
class ToolCallDelta:
    """工具调用的一段增量。"""

    context: ConversationContext
    index: int = 0
    call_id: str = ""
    identity: str = ""
    arguments_delta: str = ""
    occurred_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        if not isinstance(self.context, ConversationContext):
            raise TypeError("context must be a ConversationContext")
        index = int(self.index)
        if index < 0:
            raise ValueError("tool call index cannot be negative")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "call_id", str(self.call_id or "").strip())
        object.__setattr__(self, "identity", str(self.identity or "").strip())
        object.__setattr__(self, "arguments_delta", str(self.arguments_delta or ""))
        object.__setattr__(self, "occurred_at", float(self.occurred_at))


@dataclass(frozen=True)
class AdapterMetadata:
    """可选的响应元数据，不包含原始响应正文。"""

    context: ConversationContext
    values: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        if not isinstance(self.context, ConversationContext):
            raise TypeError("context must be a ConversationContext")
        object.__setattr__(self, "values", dict(self.values or {}))
        object.__setattr__(self, "occurred_at", float(self.occurred_at))


__all__ = ["AdapterMetadata", "ToolCallDelta"]
