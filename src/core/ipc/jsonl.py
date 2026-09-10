"""与模型类型无关的有界 JSONL 消息构造和解析。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class IPCProtocolSpec:
    protocol: str
    version: int
    commands: frozenset[str]
    events: frozenset[str]
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        protocol = str(self.protocol or "").strip()
        if not protocol or len(protocol) > 128 or any(char in protocol for char in "\x00\r\n"):
            raise ValueError("IPC protocol is invalid")
        if isinstance(self.version, bool) or int(self.version) <= 0:
            raise ValueError("IPC protocol version is invalid")
        commands = _message_types(self.commands, "commands")
        events = _message_types(self.events, "events")
        # 同名请求/回执（例如 health、shutdown）由传输方向区分，协议可显式
        # 允许；调用端仍会分别用 commands/events 集合校验收发方向。
        capabilities = tuple(str(item or "").strip() for item in self.capabilities)
        if any(not item or len(item) > 64 or "\n" in item for item in capabilities):
            raise ValueError("IPC capabilities are invalid")
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "version", int(self.version))
        object.__setattr__(self, "commands", commands)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "capabilities", capabilities)

    @property
    def message_types(self) -> frozenset[str]:
        return self.commands | self.events


@dataclass(frozen=True)
class JSONLMessage:
    type: str
    request_id: str
    payload: Mapping[str, object]


def _message_types(values: object, label: str) -> frozenset[str]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise ValueError(f"IPC {label} are invalid")
    try:
        rendered = frozenset(str(item or "").strip().lower() for item in values)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError(f"IPC {label} are invalid") from exc
    if any(
        not item or len(item) > 64 or any(char in item for char in "\x00\r\n") for item in rendered
    ):
        raise ValueError(f"IPC {label} are invalid")
    return rendered


def safe_ipc_id(value: object, *, required: bool = False) -> str:
    """规范化 IPC 关联 ID，并拒绝控制字符和无界输入。"""

    rendered = str(value or "").strip()
    if required and not rendered:
        raise ValueError("IPC request_id is required")
    if len(rendered) > 256 or any(char in rendered for char in "\x00\r\n"):
        raise ValueError("IPC request_id is invalid")
    return rendered


def build_jsonl_message(
    spec: IPCProtocolSpec,
    message_type: str,
    *,
    request_id: object = "",
    fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """构造带协议头的命令或事件，调用方只能添加显式字段。"""

    normalized_type = str(message_type or "").strip().lower()
    if normalized_type not in spec.message_types:
        raise ValueError("IPC message type is invalid")
    value: dict[str, object] = {
        "type": normalized_type,
        "protocol": spec.protocol,
        "version": spec.version,
    }
    normalized_id = safe_ipc_id(request_id)
    if normalized_id:
        value["request_id"] = normalized_id
    for raw_key, item in (fields or {}).items():
        key = str(raw_key or "").strip()
        if not key or len(key) > 128 or key in value or any(char in key for char in "\x00\r\n"):
            raise ValueError("IPC field name is invalid")
        value[key] = item
    return value


def parse_jsonl_message(
    spec: IPCProtocolSpec,
    payload: bytes | str,
    *,
    max_bytes: int,
    allowed_types: frozenset[str] | None = None,
) -> JSONLMessage:
    """解析一行 JSONL，并严格核对大小、协议、版本和消息类型。"""

    if isinstance(payload, bytes):
        if len(payload) > max_bytes:
            raise ValueError("IPC message exceeds limit")
        try:
            rendered = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("IPC message is not UTF-8") from exc
    else:
        rendered = str(payload)
        if len(rendered.encode("utf-8")) > max_bytes:
            raise ValueError("IPC message exceeds limit")
    try:
        value = json.loads(rendered)
    except json.JSONDecodeError as exc:
        raise ValueError("IPC message is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("IPC message must be an object")
    if value.get("protocol") != spec.protocol:
        raise ValueError("IPC protocol does not match")
    version = value.get("version")
    if isinstance(version, bool) or version != spec.version:
        raise ValueError("IPC protocol version does not match")
    message_type = str(value.get("type") or "").strip().lower()
    accepted = allowed_types if allowed_types is not None else spec.message_types
    if message_type not in accepted:
        raise ValueError("IPC message type is invalid")
    request_id = safe_ipc_id(value.get("request_id"))
    return JSONLMessage(message_type, request_id, dict(value))


__all__ = [
    "IPCProtocolSpec",
    "JSONLMessage",
    "build_jsonl_message",
    "parse_jsonl_message",
    "safe_ipc_id",
]
