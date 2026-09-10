"""Provider-independent speech contracts and audio protocol helpers."""

from .audio import AudioProtocolError, PcmFormat, StreamingPcmValidator, StreamingWavDecoder
from .contracts import (
    AudioFeatureAnalyzer,
    EngineHealth,
    SpeechChunk,
    SpeechMethodCall,
    SpeechRequest,
)
from .ipc import (
    TTS_IPC_CAPABILITIES,
    TTS_IPC_COMMANDS,
    TTS_IPC_EVENTS,
    TTS_IPC_PROTOCOL,
    TTS_IPC_SPEC,
    TTS_IPC_VERSION,
    TTSIPCMessage,
    build_ipc_message,
    parse_ipc_message,
)
from .language import (
    canonical_tts_language,
    is_safe_tts_token,
    normalize_reference_audios,
    protocol_tts_language,
    reference_language_prefix,
    resolve_mood_reference,
)

__all__ = [
    "AudioProtocolError",
    "AudioFeatureAnalyzer",
    "EngineHealth",
    "PcmFormat",
    "SpeechChunk",
    "SpeechMethodCall",
    "SpeechRequest",
    "StreamingPcmValidator",
    "StreamingWavDecoder",
    "TTSIPCMessage",
    "TTS_IPC_CAPABILITIES",
    "TTS_IPC_COMMANDS",
    "TTS_IPC_EVENTS",
    "TTS_IPC_PROTOCOL",
    "TTS_IPC_SPEC",
    "TTS_IPC_VERSION",
    "build_ipc_message",
    "canonical_tts_language",
    "is_safe_tts_token",
    "normalize_reference_audios",
    "protocol_tts_language",
    "parse_ipc_message",
    "reference_language_prefix",
    "resolve_mood_reference",
]
