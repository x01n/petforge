"""跨模型、工具、语音和界面的统一流事件。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import time
from typing import Literal


def _required(value: object, field_name: str, maximum: int = 256) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(character in text for character in "\r\n\x00"):
        raise ValueError(f"{field_name} is invalid")
    return text


@dataclass(frozen=True)
class ConversationContext:
    """用于丢弃迟到事件的会话代际标识。"""

    profile_id: str
    session_id: str
    turn_id: str
    generation_id: int
    mode: Literal["direct", "agent"] = "direct"

    def __post_init__(self) -> None:
        mode = str(self.mode or "").strip().lower()
        if mode not in {"direct", "agent"}:
            raise ValueError("mode is unsupported")
        generation_id = int(self.generation_id)
        if generation_id < 0:
            raise ValueError("generation_id cannot be negative")
        object.__setattr__(self, "profile_id", _required(self.profile_id, "profile_id"))
        object.__setattr__(self, "session_id", _required(self.session_id, "session_id"))
        object.__setattr__(self, "turn_id", _required(self.turn_id, "turn_id"))
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "mode", mode)


@dataclass(frozen=True)
class _Event:
    context: ConversationContext
    occurred_at: float = field(default_factory=time, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.context, ConversationContext):
            raise TypeError("context must be a ConversationContext")
        object.__setattr__(self, "occurred_at", float(self.occurred_at))


@dataclass(frozen=True)
class TextDelta(_Event):
    delta: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "delta", str(self.delta or ""))


@dataclass(frozen=True)
class SentenceReady(_Event):
    """一个完整可展示、可提交 TTS 的句子事件。

    该事件不继承 ``TextDelta``，避免与供应商原始 token 增量重复累加。
    ``ConversationService`` 将它只提交给展示/TTS 边界；原始 ``TextDelta``
    仍可独立进入输出框和内部答案聚合。
    """

    text: str = ""
    sequence: int = 0
    final: bool = False
    forced: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        text = str(self.text or "")
        if not text.strip():
            raise ValueError("sentence text is required")
        sequence = int(self.sequence)
        if sequence <= 0:
            raise ValueError("sentence sequence must be positive")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "final", bool(self.final))
        object.__setattr__(self, "forced", bool(self.forced))


@dataclass(frozen=True)
class ReasoningDelta(_Event):
    """供应商的原始 reasoning，仅限诊断，不能自动渲染给用户。"""

    delta: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "delta", str(self.delta or ""))


@dataclass(frozen=True)
class MurmurDelta(_Event):
    """模型显式产出的可见碎碎念，不等同于原始 reasoning。"""

    delta: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "delta", str(self.delta or ""))


@dataclass(frozen=True)
class ToolCallStarted(_Event):
    call_id: str = ""
    identity: str = ""
    safe_summary: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "call_id", _required(self.call_id, "call_id"))
        object.__setattr__(self, "identity", _required(self.identity, "identity"))
        object.__setattr__(self, "safe_summary", str(self.safe_summary or "").strip())


@dataclass(frozen=True)
class ToolStatusChanged(_Event):
    state: Literal["running", "completed", "denied", "failed"] = "running"
    safe_summary: str = ""
    call_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        state = str(self.state or "").strip().lower()
        if state not in {"running", "completed", "denied", "failed"}:
            raise ValueError("tool status is unsupported")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "safe_summary", str(self.safe_summary or "").strip())
        object.__setattr__(self, "call_id", _required(self.call_id, "call_id"))


@dataclass(frozen=True)
class ApprovalRequested(_Event):
    approval_id: str = ""
    call_id: str = ""
    safe_summary: str = ""
    expires_at: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "approval_id", _required(self.approval_id, "approval_id"))
        object.__setattr__(self, "call_id", _required(self.call_id, "call_id"))
        object.__setattr__(self, "safe_summary", str(self.safe_summary or "").strip())
        expires_at = float(self.expires_at)
        if expires_at <= self.occurred_at:
            raise ValueError("approval expiry must be in the future")
        object.__setattr__(self, "expires_at", expires_at)


@dataclass(frozen=True)
class AudioChunk(_Event):
    data: bytes = b""
    sample_rate: int = 24000
    channels: int = 1
    sample_format: Literal["s16le"] = "s16le"
    is_final: bool = False
    request_id: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        data = bytes(self.data or b"")
        sample_rate = int(self.sample_rate)
        channels = int(self.channels)
        if sample_rate <= 0 or channels not in {1, 2}:
            raise ValueError("audio format is invalid")
        if self.sample_format != "s16le":
            raise ValueError("audio format is unsupported")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sample_rate", sample_rate)
        object.__setattr__(self, "channels", channels)
        object.__setattr__(self, "is_final", bool(self.is_final))
        object.__setattr__(self, "request_id", str(self.request_id or "").strip())


_AUDIO_FEATURE_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")


@dataclass(frozen=True)
class AudioFeature(_Event):
    """由外部音频分析器确认的时间化口型特征。

    这个事件不从文本、语言或情绪字段推导任何音素。只有明确标记为
    ``confirmed=True`` 且带有分析器来源的结果才能进入渲染边界；没有分析器
    时不会生成该事件，调用方应继续使用文本/overlay 回退。
    """

    request_id: str = ""
    viseme: str = ""
    intensity: float = 0.0
    start_ms: int = 0
    duration_ms: int = 0
    source: str = ""
    confirmed: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        request_id = _required(self.request_id, "request_id")
        viseme = _required(self.viseme, "viseme", maximum=32)
        if _AUDIO_FEATURE_LABEL.fullmatch(viseme) is None:
            raise ValueError("viseme contains unsupported characters")
        try:
            intensity = float(self.intensity)
        except (TypeError, ValueError):
            raise ValueError("audio feature intensity is invalid") from None
        if not math.isfinite(intensity) or not 0.0 <= intensity <= 1.0:
            raise ValueError("audio feature intensity must be between 0 and 1")
        try:
            start_ms = int(self.start_ms)
            duration_ms = int(self.duration_ms)
        except (TypeError, ValueError):
            raise ValueError("audio feature timing is invalid") from None
        if start_ms < 0 or duration_ms < 0:
            raise ValueError("audio feature timing cannot be negative")
        source = _required(self.source, "source", maximum=128)
        if self.confirmed is not True:
            raise ValueError("audio feature must be explicitly confirmed")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "viseme", viseme)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "start_ms", start_ms)
        object.__setattr__(self, "duration_ms", duration_ms)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "confirmed", True)


@dataclass(frozen=True)
class TurnFinished(_Event):
    finish_reason: str = "stop"
    usage: Mapping[str, int] = field(default_factory=dict)
    response_model: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        usage = {str(key): int(value) for key, value in dict(self.usage or {}).items()}
        if any(value < 0 for value in usage.values()):
            raise ValueError("usage cannot contain negative values")
        object.__setattr__(
            self, "finish_reason", str(self.finish_reason or "stop").strip() or "stop"
        )
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "response_model", str(self.response_model or "").strip())
        if not isinstance(self.metadata, Mapping):
            raise ValueError("turn metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class TurnFailed(_Event):
    category: str = "unknown"
    safe_message: str = ""
    retryable: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "category", _required(self.category, "category"))
        object.__setattr__(self, "safe_message", str(self.safe_message or "").strip())
        object.__setattr__(self, "retryable", bool(self.retryable))
