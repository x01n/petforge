"""内置语音识别服务的纯数据契约。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.ipc import IPCProtocolSpec

ASR_IPC_PROTOCOL = "meapet.asr.jsonl"
ASR_IPC_VERSION = 1
ASR_IPC_CAPABILITIES = (
    "health",
    "model_load",
    "transcription",
    "cancel",
    "shutdown",
    "wav",
    "pcm_s16le",
)
ASR_IPC_COMMANDS = frozenset(
    {"startup", "load_model", "health", "transcribe", "cancel", "shutdown"}
)
ASR_IPC_EVENTS = frozenset(
    {
        "ready",
        "started",
        "model_loaded",
        "health",
        "transcript",
        "cancelled",
        "shutdown",
        "status",
        "error",
    }
)
ASR_IPC_SPEC = IPCProtocolSpec(
    ASR_IPC_PROTOCOL,
    ASR_IPC_VERSION,
    ASR_IPC_COMMANDS,
    ASR_IPC_EVENTS,
    ASR_IPC_CAPABILITIES,
)

SUPPORTED_ASR_BACKENDS = frozenset({"sensevoice"})
SUPPORTED_ASR_FORMATS = frozenset({"wav", "pcm_s16le"})
SUPPORTED_ASR_LANGUAGES = frozenset({"auto", "zh", "en", "yue", "ja", "ko", "nospeech"})


@dataclass(frozen=True, slots=True)
class ASRHealth:
    """可安全暴露给控制台的语音识别健康快照。"""

    status: str
    available: bool
    backend: str
    model: str
    model_loaded: bool
    device: str
    language: str
    message: str = ""

    @property
    def ready(self) -> bool:
        return self.available and self.model_loaded and self.status == "ready"


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    """一次只携带内存音频字节的转写请求。"""

    audio: bytes
    audio_format: str = "wav"
    sample_rate: int = 16000
    channels: int = 1
    language: str = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.audio, bytes) or not self.audio:
            raise ValueError("ASR audio must be non-empty bytes")
        audio_format = str(self.audio_format or "").strip().lower()
        if audio_format not in SUPPORTED_ASR_FORMATS:
            raise ValueError("ASR audio format is invalid")
        if isinstance(self.sample_rate, bool) or not 8000 <= int(self.sample_rate) <= 192000:
            raise ValueError("ASR sample rate is invalid")
        if isinstance(self.channels, bool) or not 1 <= int(self.channels) <= 8:
            raise ValueError("ASR channel count is invalid")
        language = str(self.language or "").strip().lower()
        if language not in SUPPORTED_ASR_LANGUAGES:
            raise ValueError("ASR language is invalid")
        if audio_format == "wav" and (
            len(self.audio) < 12 or self.audio[:4] != b"RIFF" or self.audio[8:12] != b"WAVE"
        ):
            raise ValueError("ASR WAV payload is invalid")
        if audio_format == "pcm_s16le" and len(self.audio) % (2 * int(self.channels)):
            raise ValueError("ASR PCM payload is not frame aligned")
        object.__setattr__(self, "audio_format", audio_format)
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "channels", int(self.channels))
        object.__setattr__(self, "language", language)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """转写正文及模型明确提供的语言、置信信息。"""

    request_id: str
    text: str
    language: str
    confidence: float | None
    confidence_available: bool
    duration_ms: int

    def __post_init__(self) -> None:
        request_id = str(self.request_id or "").strip()
        text = str(self.text or "").strip()
        language = str(self.language or "unknown").strip().lower()
        if not request_id:
            raise ValueError("ASR request id is required")
        confidence = self.confidence
        if confidence is not None:
            confidence = float(confidence)
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError("ASR confidence is invalid")
        if bool(self.confidence_available) != (confidence is not None):
            raise ValueError("ASR confidence availability does not match")
        if isinstance(self.duration_ms, bool) or int(self.duration_ms) < 0:
            raise ValueError("ASR duration is invalid")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "duration_ms", int(self.duration_ms))

    def public(self) -> dict[str, object]:
        """返回调用方可消费、不会隐式写日志的结果。"""

        return {
            "request_id": self.request_id,
            "text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "confidence_available": self.confidence_available,
            "duration_ms": self.duration_ms,
        }


__all__ = [
    "ASRHealth",
    "ASR_IPC_CAPABILITIES",
    "ASR_IPC_COMMANDS",
    "ASR_IPC_EVENTS",
    "ASR_IPC_PROTOCOL",
    "ASR_IPC_SPEC",
    "ASR_IPC_VERSION",
    "SUPPORTED_ASR_BACKENDS",
    "SUPPORTED_ASR_FORMATS",
    "SUPPORTED_ASR_LANGUAGES",
    "TranscriptionRequest",
    "TranscriptionResult",
]
