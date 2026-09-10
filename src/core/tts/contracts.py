"""流式 TTS 引擎的稳定协议。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Protocol

from core.events.types import AudioChunk, AudioFeature

from .language import canonical_tts_language, is_safe_tts_token


@dataclass(frozen=True)
class SpeechRequest:
    request_id: str
    text: str
    language: str = "zh"
    mood: str = "neutral"
    reference_audio: str = ""
    reference_text: str = ""
    options: Mapping[str, object] = field(default_factory=dict)
    profile_id: str = ""
    voice: str = ""
    role: str = ""

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        text = str(self.text or "").strip()
        if not request_id or not text:
            raise ValueError("speech request_id and text are required")
        if len(text) > 10000:
            raise ValueError("speech text is too long")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "text", text)
        language = str(self.language or "").strip()
        if not language:
            raise ValueError("speech language is required")
        if not canonical_tts_language(language):
            raise ValueError("speech language is invalid")
        object.__setattr__(self, "language", language)
        mood = str(self.mood or "neutral").strip().lower()
        if not is_safe_tts_token(mood):
            raise ValueError("speech mood is invalid")
        object.__setattr__(self, "mood", mood)
        object.__setattr__(self, "reference_audio", str(self.reference_audio or "").strip())
        object.__setattr__(self, "reference_text", str(self.reference_text or "").strip())
        object.__setattr__(self, "options", dict(self.options or {}))
        profile_id = str(self.profile_id or "").strip()
        if len(profile_id) > 128 or any(char in profile_id for char in "\r\n\x00"):
            raise ValueError("speech profile_id is invalid")
        object.__setattr__(self, "profile_id", profile_id)
        voice = str(self.voice or "").strip()
        if len(voice) > 128 or any(char in voice for char in "\r\n\x00"):
            raise ValueError("speech voice is invalid")
        object.__setattr__(self, "voice", voice)
        role = str(self.role or "").strip()
        if len(role) > 128 or any(char in role for char in "\r\n\x00"):
            raise ValueError("speech role is invalid")
        object.__setattr__(self, "role", role)


@dataclass(frozen=True)
class SpeechChunk:
    request_id: str
    data: bytes
    sample_rate: int = 24000
    channels: int = 1
    sample_format: str = "s16le"
    is_final: bool = False

    def __post_init__(self) -> None:
        if not str(self.request_id or "").strip():
            raise ValueError("speech chunk request_id is required")
        if self.sample_format != "s16le":
            raise ValueError("only s16le speech chunks are supported")
        if int(self.sample_rate) <= 0 or int(self.channels) not in {1, 2}:
            raise ValueError("speech audio format is invalid")
        object.__setattr__(self, "request_id", str(self.request_id).strip())
        object.__setattr__(self, "data", bytes(self.data or b""))
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "channels", int(self.channels))
        object.__setattr__(self, "is_final", bool(self.is_final))


@dataclass(frozen=True)
class EngineHealth:
    engine: str
    available: bool
    message: str = ""
    latency_ms: float | None = None


@dataclass(frozen=True)
class SpeechMethodCall:
    """由 TTS 适配器生成的 HTTP 方法调用描述。

    调用者只提交 :class:`SpeechRequest`；端点、协议字段和流式策略均由
    后端适配器固化在该只读对象中。
    """

    request_id: str
    method: str
    endpoint: str
    payload: Mapping[str, object] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    streaming: bool = True

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        method = str(self.method or "").strip().upper()
        endpoint = str(self.endpoint or "").strip()
        if not request_id:
            raise ValueError("speech method call request_id is required")
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("speech method call method is invalid")
        if not endpoint:
            raise ValueError("speech method call endpoint is required")
        if not isinstance(self.streaming, bool):
            raise ValueError("speech method call streaming must be boolean")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "payload", dict(self.payload or {}))
        object.__setattr__(
            self,
            "headers",
            {str(key): str(value) for key, value in (self.headers or {}).items()},
        )


class SpeechBackend(Protocol):
    async def start(self) -> EngineHealth: ...

    async def health(self) -> EngineHealth: ...

    def stream(self, request: SpeechRequest) -> AsyncIterator[SpeechChunk]: ...

    async def aclose(self) -> None: ...


class AudioFeatureAnalyzer(Protocol):
    """把一段 PCM 交给外部分析器，返回已确认的时间化特征。

    实现必须自己拥有可验证的声学/模型证据，并直接构造
    :class:`core.events.types.AudioFeature`。协调器不会从文本、语言、情绪
    或音频振幅猜测 viseme，也不会替分析器补全缺失字段。
    """

    def analyze(
        self, chunk: AudioChunk
    ) -> Iterable[AudioFeature] | Awaitable[Iterable[AudioFeature] | None] | None: ...
