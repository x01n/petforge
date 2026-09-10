"""按会话隔离的 generation 管理，阻止旧异步结果覆盖新回合。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from core.events.types import ConversationContext


@dataclass(frozen=True)
class ConversationKey:
    mode: str
    profile_id: str
    session_id: str

    def __post_init__(self) -> None:
        mode = str(self.mode or "").strip().lower()
        if mode not in {"direct", "agent"}:
            raise ValueError("conversation mode is unsupported")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self, "profile_id", str(self.profile_id or "default").strip() or "default"
        )
        object.__setattr__(self, "session_id", str(self.session_id or "local").strip() or "local")


class GenerationGate:
    """线程安全的会话代际表。"""

    def __init__(self) -> None:
        self._lock = RLock()
        self._generation_by_key: dict[ConversationKey, int] = {}

    def begin(self, key: ConversationKey, turn_id: str) -> ConversationContext:
        if not isinstance(key, ConversationKey):
            raise TypeError("key must be a ConversationKey")
        with self._lock:
            generation = self._generation_by_key.get(key, 0) + 1
            self._generation_by_key[key] = generation
        return ConversationContext(
            mode=key.mode,
            profile_id=key.profile_id,
            session_id=key.session_id,
            turn_id=turn_id,
            generation_id=generation,
        )

    def cancel(self, key: ConversationKey) -> int:
        """推进代际，使已在路上的事件立即失效。"""
        with self._lock:
            generation = self._generation_by_key.get(key, 0) + 1
            self._generation_by_key[key] = generation
            return generation

    def accepts(self, context: ConversationContext) -> bool:
        key = ConversationKey(context.mode, context.profile_id, context.session_id)
        with self._lock:
            return self._generation_by_key.get(key, 0) == context.generation_id
