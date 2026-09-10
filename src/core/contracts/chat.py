"""统一的对话请求、工具调用和内联图片契约。"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

ChatRole = Literal["system", "developer", "user", "assistant", "tool"]
_VALID_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
_IMAGE_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_MAX_IMAGE_BYTES = 5 * 1024 * 1024


def _safe_identifier(value: object, *, field_name: str, maximum: int = 256) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    if len(text) > maximum or any(character in text for character in "\r\n\x00"):
        raise ValueError(f"{field_name} is unsafe")
    return text


@dataclass(frozen=True)
class ImageAttachment:
    """只允许有界内联图片，避免模型适配器间产生远端读取行为。"""

    media_type: str
    data: str
    file_name: str = "screenshot.png"

    def __post_init__(self) -> None:
        media_type = str(self.media_type or "").strip().lower()
        if media_type not in _IMAGE_MEDIA_TYPES:
            raise ValueError("image media_type is unsupported")
        data = str(self.data or "").strip()
        try:
            decoded = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("image data must be valid base64") from exc
        if not decoded or len(decoded) > _MAX_IMAGE_BYTES:
            raise ValueError("image data exceeds the allowed size")
        file_name = _safe_identifier(self.file_name, field_name="image file_name", maximum=128)
        if "/" in file_name or "\\" in file_name:
            raise ValueError("image file_name is unsafe")
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "file_name", file_name)

    def as_content_part(self) -> dict[str, str]:
        return {"type": "image", "media_type": self.media_type, "data": self.data}


@dataclass(frozen=True)
class ToolDefinition:
    """模型可见的工具定义；真实执行器始终由工具运行时持有。"""

    identity: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        identity = _safe_identifier(self.identity, field_name="tool identity")
        if ":" not in identity:
            raise ValueError("tool identity must use namespace:name")
        parameters = dict(self.parameters or {})
        if parameters.get("type") not in {None, "object"}:
            raise ValueError("tool parameters must be an object schema")
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "parameters", parameters)

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.identity,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class ToolCall:
    """已解析完成的工具调用。"""

    call_id: str
    identity: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "call_id", _safe_identifier(self.call_id, field_name="tool call_id")
        )
        identity = _safe_identifier(self.identity, field_name="tool identity")
        if ":" not in identity:
            raise ValueError("tool identity must use namespace:name")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("tool arguments must be a mapping")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class ToolResult:
    """工具运行时返回给模型和展示层的已清洗结果。"""

    call_id: str
    identity: str
    content: Mapping[str, Any]
    is_error: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "call_id", _safe_identifier(self.call_id, field_name="tool call_id")
        )
        object.__setattr__(
            self, "identity", _safe_identifier(self.identity, field_name="tool identity")
        )
        if not isinstance(self.content, Mapping):
            raise ValueError("tool result content must be a mapping")
        object.__setattr__(self, "content", dict(self.content))
        object.__setattr__(self, "is_error", bool(self.is_error))

    def as_message(self) -> dict[str, object]:
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "tool_name": self.identity,
            "content": dict(self.content),
        }


@dataclass(frozen=True)
class ChatMessage:
    """供应商无关的单条消息。"""

    role: ChatRole
    content: str | Sequence[Mapping[str, object]] | Mapping[str, object] | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""
    tool_name: str = ""

    def __post_init__(self) -> None:
        role = str(self.role or "").strip().lower()
        if role not in _VALID_ROLES:
            raise ValueError("message role is unsupported")
        tool_calls = tuple(self.tool_calls or ())
        if any(not isinstance(call, ToolCall) for call in tool_calls):
            raise ValueError("tool_calls must contain ToolCall values")
        content = self.content
        if isinstance(content, Mapping):
            content = dict(content)
        elif isinstance(content, Sequence) and not isinstance(content, str):
            parts: list[dict[str, object]] = []
            for part in content:
                if not isinstance(part, Mapping):
                    raise ValueError("message content parts must be mappings")
                parts.append(dict(part))
            content = tuple(parts)
        elif content is not None:
            content = str(content)
        if role == "assistant" and content is None and not tool_calls:
            raise ValueError("assistant content is required without tool calls")
        if role != "assistant" and content is None:
            raise ValueError("message content is required")
        tool_call_id = str(self.tool_call_id or "").strip()
        tool_name = str(self.tool_name or "").strip()
        if role == "tool" and not (tool_call_id or tool_name):
            raise ValueError("tool messages require tool_call_id or tool_name")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "tool_calls", tool_calls)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "tool_name", tool_name)

    def as_mapping(self) -> dict[str, object]:
        message: dict[str, object] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.identity, "arguments": dict(call.arguments)},
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            message["tool_name"] = self.tool_name
        return message


@dataclass(frozen=True)
class ChatRequest:
    """进入提供方适配器运行时的完整标准请求。"""

    model: str
    messages: tuple[ChatMessage, ...]
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: tuple[ToolDefinition, ...] = ()
    tool_choice: str | None = "auto"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", _safe_identifier(self.model, field_name="model"))
        messages = tuple(self.messages or ())
        if not messages or any(not isinstance(message, ChatMessage) for message in messages):
            raise ValueError("messages must contain ChatMessage values")
        temperature = float(self.temperature)
        max_tokens = int(self.max_tokens)
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        tools = tuple(self.tools or ())
        if any(not isinstance(tool, ToolDefinition) for tool in tools):
            raise ValueError("tools must contain ToolDefinition values")
        if len({tool.identity for tool in tools}) != len(tools):
            raise ValueError("tool identities must be unique")
        choice = None if self.tool_choice is None else str(self.tool_choice).strip().lower()
        if choice not in {None, "auto", "none", "required"}:
            raise ValueError("tool_choice is unsupported")
        if choice == "required" and not tools:
            raise ValueError("tool_choice=required needs visible tools")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "messages", messages)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "max_tokens", max_tokens)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "tool_choice", choice)
        object.__setattr__(self, "metadata", dict(self.metadata))
