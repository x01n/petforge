"""供内置模型 worker 复用的 JSONL IPC 契约。"""

from .jsonl import (
    IPCProtocolSpec,
    JSONLMessage,
    build_jsonl_message,
    parse_jsonl_message,
    safe_ipc_id,
)

__all__ = [
    "IPCProtocolSpec",
    "JSONLMessage",
    "build_jsonl_message",
    "parse_jsonl_message",
    "safe_ipc_id",
]
