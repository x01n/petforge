"""语音识别的协议和纯数据契约。"""

from .contracts import (
    ASR_IPC_CAPABILITIES,
    ASR_IPC_COMMANDS,
    ASR_IPC_EVENTS,
    ASR_IPC_PROTOCOL,
    ASR_IPC_SPEC,
    ASR_IPC_VERSION,
    SUPPORTED_ASR_BACKENDS,
    SUPPORTED_ASR_FORMATS,
    SUPPORTED_ASR_LANGUAGES,
    ASRHealth,
    TranscriptionRequest,
    TranscriptionResult,
)

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
