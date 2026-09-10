"""MeaPet 内置 TTS worker 的稳定 JSONL IPC 契约。"""

from __future__ import annotations

from collections.abc import Mapping

from core.ipc import (
    IPCProtocolSpec,
    JSONLMessage,
    build_jsonl_message,
    parse_jsonl_message,
    safe_ipc_id,
)

TTS_IPC_PROTOCOL = "meapet.tts.jsonl"
TTS_IPC_VERSION = 1
TTS_IPC_CAPABILITIES = (
    "health",
    "model_load",
    "streaming_synthesis",
    "cancel",
    "shutdown",
)
TTS_IPC_COMMANDS = frozenset({"health", "load_model", "synthesize", "cancel", "shutdown"})
TTS_IPC_EVENTS = frozenset(
    {
        "ready",
        "health",
        "model_loaded",
        "audio",
        "done",
        "cancelled",
        "shutdown",
        "status",
        "log",
        "error",
    }
)
TTS_IPC_SPEC = IPCProtocolSpec(
    TTS_IPC_PROTOCOL,
    TTS_IPC_VERSION,
    TTS_IPC_COMMANDS,
    TTS_IPC_EVENTS,
    TTS_IPC_CAPABILITIES,
)
TTSIPCMessage = JSONLMessage


def build_ipc_message(
    message_type: str,
    *,
    request_id: object = "",
    fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """构造带协议头的命令或事件，调用方只能添加显式字段。"""

    return build_jsonl_message(
        TTS_IPC_SPEC,
        message_type,
        request_id=request_id,
        fields=fields,
    )


def parse_ipc_message(
    payload: bytes | str,
    *,
    max_bytes: int,
    allowed_types: frozenset[str] | None = None,
) -> TTSIPCMessage:
    """解析一行 JSONL，并严格核对协议、版本和消息类型。"""

    return parse_jsonl_message(
        TTS_IPC_SPEC,
        payload,
        max_bytes=max_bytes,
        allowed_types=allowed_types,
    )


__all__ = [
    "TTSIPCMessage",
    "TTS_IPC_CAPABILITIES",
    "TTS_IPC_COMMANDS",
    "TTS_IPC_EVENTS",
    "TTS_IPC_PROTOCOL",
    "TTS_IPC_SPEC",
    "TTS_IPC_VERSION",
    "build_ipc_message",
    "parse_ipc_message",
    "safe_ipc_id",
]
