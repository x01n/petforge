"""MCP 2025-06-18 resources/prompts 的有界只读数据契约。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from core.json_values import validate_json_value as validate_shared_json_value

from .protocol import MCPProtocolError, MCPServerError, MCPTimeoutError

MCPRequest = Callable[[str, Mapping[str, Any] | None], Awaitable[Mapping[str, Any]]]

_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")
_MIME_TOKEN = re.compile(r"^[A-Za-z0-9!#$&^_.+-]+$")


@dataclass(frozen=True, slots=True)
class MCPContentLimits:
    """一个内容操作的条目、正文与总时限上限。"""

    max_pages: int = 64
    max_list_entries: int = 1024
    max_cursor_chars: int = 8192
    max_uri_chars: int = 8192
    max_messages: int = 128
    max_content_items: int = 128
    max_text_chars: int = 1_000_000
    max_block_bytes: int = 1024 * 1024
    max_total_bytes: int = 2 * 1024 * 1024
    max_prompt_arguments: int = 64
    max_argument_chars: int = 65_536
    max_argument_bytes: int = 256 * 1024
    operation_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        integer_fields = (
            "max_pages",
            "max_list_entries",
            "max_cursor_chars",
            "max_uri_chars",
            "max_messages",
            "max_content_items",
            "max_text_chars",
            "max_block_bytes",
            "max_total_bytes",
            "max_prompt_arguments",
            "max_argument_chars",
            "max_argument_bytes",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"MCP content limit {name} is invalid")
        if self.max_block_bytes > self.max_total_bytes:
            raise ValueError("MCP content block limit exceeds total limit")
        timeout = self.operation_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("MCP content timeout is invalid")
        timeout = float(timeout)
        if not math.isfinite(timeout) or not 0.1 <= timeout <= 60.0:
            raise ValueError("MCP content timeout is outside the allowed range")
        object.__setattr__(self, "operation_timeout_seconds", timeout)

    @classmethod
    def for_transport(
        cls,
        *,
        max_message_bytes: int,
        request_timeout_seconds: float,
    ) -> MCPContentLimits:
        """把传输消息上限同时收紧为内容上限，不扩大既有配置。"""

        wire_limit = int(max_message_bytes)
        total = min(2 * 1024 * 1024, wire_limit)
        block = min(1024 * 1024, total)
        timeout = min(30.0, float(request_timeout_seconds))
        return cls(
            max_block_bytes=block,
            max_total_bytes=total,
            operation_timeout_seconds=timeout,
        )


DEFAULT_MCP_CONTENT_LIMITS = MCPContentLimits()


def _required_text(
    value: object,
    *,
    label: str,
    maximum: int,
    allow_newlines: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")
    forbidden = "\x00" if allow_newlines else "\x00\r\n"
    if any(char in value for char in forbidden):
        raise ValueError(f"{label} is invalid")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} is invalid") from exc
    return value


def _optional_text(
    value: object,
    *,
    label: str,
    maximum: int,
    allow_newlines: bool = True,
) -> str | None:
    if not isinstance(value, str) or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{label} is invalid")
    if not allow_newlines and any(char in value for char in "\r\n"):
        raise ValueError(f"{label} is invalid")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} is invalid") from exc
    return value


def validate_resource_uri(value: object, *, maximum: int = 8192) -> str:
    """按 RFC 3986 形状校验资源 URI；保留原始大小写和查询串。"""

    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("MCP resource URI is invalid")
    if len(value) > maximum or "\\" in value or "{" in value or "}" in value:
        raise ValueError("MCP resource URI is invalid")
    if any(ord(char) < 0x21 or ord(char) > 0x7E for char in value):
        raise ValueError("MCP resource URI is invalid")
    index = 0
    while True:
        index = value.find("%", index)
        if index < 0:
            break
        if index + 2 >= len(value) or any(
            char not in "0123456789abcdefABCDEF" for char in value[index + 1 : index + 3]
        ):
            raise ValueError("MCP resource URI is invalid")
        index += 3
    scheme, separator, remainder = value.partition(":")
    if not separator or not remainder or _URI_SCHEME.fullmatch(scheme) is None:
        raise ValueError("MCP resource URI is invalid")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        has_credentials = parsed.username is not None or parsed.password is not None
        # 访问 ``port`` 可触发 urllib 对非法/越界端口的严格校验。
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("MCP resource URI is invalid") from exc
    if has_credentials:
        raise ValueError("MCP resource URI credentials are not allowed")
    if scheme.lower() in {"http", "https"} and not hostname:
        raise ValueError("MCP resource URI is invalid")
    return value


def validate_prompt_name(value: object) -> str:
    """校验 prompt/参数逻辑名；不折叠大小写或改变格式。"""

    return _required_text(value, label="MCP prompt name", maximum=256)


def validate_mime_type(value: object, *, expected_prefix: str | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > 255:
        raise ValueError("MCP MIME type is invalid")
    if any(ord(char) < 0x20 or ord(char) > 0x7E for char in value):
        raise ValueError("MCP MIME type is invalid")
    media_type = value.split(";", 1)[0].strip()
    parts = media_type.split("/")
    if len(parts) != 2 or any(_MIME_TOKEN.fullmatch(part) is None for part in parts):
        raise ValueError("MCP MIME type is invalid")
    if expected_prefix is not None and parts[0].lower() != expected_prefix:
        raise ValueError("MCP content MIME type does not match its type")
    return value


def mcp_source_identity(server_name: object) -> str:
    """返回不泄露服务器名称、无端点信息且可供输出归因的来源身份。"""

    name = _required_text(server_name, label="MCP server name", maximum=128)
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    return f"mcp:source-{digest}"


def _validate_meta(value: Mapping[str, Any], *, label: str) -> None:
    if "_meta" not in value:
        return
    meta = value["_meta"]
    if not isinstance(meta, Mapping):
        raise ValueError(f"{label} _meta must be an object")
    validate_shared_json_value(meta, label=f"{label} _meta")


@dataclass(frozen=True, slots=True)
class MCPAnnotations:
    audience: tuple[str, ...] = ()
    priority: float | None = None
    last_modified: str | None = None

    @classmethod
    def from_value(cls, value: object, *, label: str) -> MCPAnnotations:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError(f"{label} annotations must be an object")
        raw_audience = value.get("audience", [])
        if not isinstance(raw_audience, list) or any(
            role not in {"user", "assistant"} for role in raw_audience
        ):
            raise ValueError(f"{label} annotations audience is invalid")
        if len(raw_audience) > 2 or len(set(raw_audience)) != len(raw_audience):
            raise ValueError(f"{label} annotations audience is invalid")
        priority: float | None = None
        if "priority" in value:
            raw_priority = value["priority"]
            if isinstance(raw_priority, bool) or not isinstance(raw_priority, (int, float)):
                raise ValueError(f"{label} annotations priority is invalid")
            priority = float(raw_priority)
            if not math.isfinite(priority) or not 0.0 <= priority <= 1.0:
                raise ValueError(f"{label} annotations priority is invalid")
        last_modified = (
            _optional_text(
                value["lastModified"],
                label=f"{label} annotations lastModified",
                maximum=128,
                allow_newlines=False,
            )
            if "lastModified" in value
            else None
        )
        _validate_meta(value, label=f"{label} annotations")
        return cls(tuple(raw_audience), priority, last_modified)

    def public(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.audience:
            result["audience"] = list(self.audience)
        if self.priority is not None:
            result["priority"] = self.priority
        if self.last_modified is not None:
            result["lastModified"] = self.last_modified
        return result


@dataclass(frozen=True, slots=True)
class MCPResource:
    uri: str
    name: str
    title: str | None = None
    description: str | None = None
    mime_type: str | None = None
    annotations: MCPAnnotations = field(default_factory=MCPAnnotations)
    size: int | None = None
    server: str = "default"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        server: str,
        limits: MCPContentLimits,
    ) -> MCPResource:
        if not isinstance(value, Mapping):
            raise ValueError("MCP resource entry must be an object")
        uri = validate_resource_uri(value.get("uri"), maximum=limits.max_uri_chars)
        name = _required_text(value.get("name"), label="MCP resource name", maximum=512)
        title = (
            _optional_text(value["title"], label="MCP resource title", maximum=1024)
            if "title" in value
            else None
        )
        description = (
            _optional_text(value["description"], label="MCP resource description", maximum=8192)
            if "description" in value
            else None
        )
        mime_type = None
        if "mimeType" in value:
            mime_type = validate_mime_type(value["mimeType"])
        size: int | None = None
        if "size" in value:
            raw_size = value["size"]
            if isinstance(raw_size, bool) or not isinstance(raw_size, (int, float)):
                raise ValueError("MCP resource size is invalid")
            numeric = float(raw_size)
            if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
                raise ValueError("MCP resource size is invalid")
            size = int(numeric)
            if size > 2**63 - 1:
                raise ValueError("MCP resource size is invalid")
        annotations = (
            MCPAnnotations.from_value(value["annotations"], label="MCP resource")
            if "annotations" in value
            else MCPAnnotations()
        )
        _validate_meta(value, label="MCP resource")
        server_name = _required_text(server, label="MCP server name", maximum=128)
        return cls(uri, name, title, description, mime_type, annotations, size, server_name)

    def public(self) -> dict[str, Any]:
        result: dict[str, Any] = {"uri": self.uri, "name": self.name}
        if self.title is not None:
            result["title"] = self.title
        if self.description is not None:
            result["description"] = self.description
        if self.mime_type is not None:
            result["mimeType"] = self.mime_type
        annotations = self.annotations.public()
        if annotations:
            result["annotations"] = annotations
        if self.size is not None:
            result["size"] = self.size
        return result


def _decoded_base64(
    value: object,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[str, bytes]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a base64 string")
    maximum_encoded = ((maximum_bytes + 2) // 3) * 4
    if len(value) > maximum_encoded:
        raise ValueError(f"{label} exceeds the encoded size limit")
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} is not valid base64") from exc
    try:
        decoded = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{label} is not valid base64") from exc
    if len(decoded) > maximum_bytes:
        raise ValueError(f"{label} exceeds the decoded size limit")
    if base64.b64encode(decoded).decode("ascii") != value:
        raise ValueError(f"{label} is not canonical base64")
    return value, decoded


def _text_bytes(value: object, *, label: str, limits: MCPContentLimits) -> tuple[str, int]:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if len(value) > limits.max_text_chars:
        raise ValueError(f"{label} exceeds the character limit")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} is invalid UTF-8 text") from exc
    if len(encoded) > limits.max_block_bytes:
        raise ValueError(f"{label} exceeds the byte limit")
    return value, len(encoded)


@dataclass(frozen=True, slots=True)
class MCPResourceContent:
    uri: str
    mime_type: str | None
    kind: str
    value: str
    byte_size: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        limits: MCPContentLimits,
    ) -> MCPResourceContent:
        if not isinstance(value, Mapping):
            raise ValueError("MCP resource content must be an object")
        uri = validate_resource_uri(value.get("uri"), maximum=limits.max_uri_chars)
        mime_type = None
        if "mimeType" in value:
            mime_type = validate_mime_type(value["mimeType"])
        has_text = "text" in value
        has_blob = "blob" in value
        if has_text == has_blob:
            raise ValueError("MCP resource content must contain exactly one of text or blob")
        if has_text:
            text, byte_size = _text_bytes(value["text"], label="MCP resource text", limits=limits)
            kind = "text"
            content_value = text
        else:
            blob, decoded = _decoded_base64(
                value["blob"],
                label="MCP resource blob",
                maximum_bytes=limits.max_block_bytes,
            )
            kind = "blob"
            content_value = blob
            byte_size = len(decoded)
        _validate_meta(value, label="MCP resource content")
        return cls(uri, mime_type, kind, content_value, byte_size)

    def public(self) -> dict[str, Any]:
        result: dict[str, Any] = {"uri": self.uri, self.kind: self.value}
        if self.mime_type is not None:
            result["mimeType"] = self.mime_type
        return result


@dataclass(frozen=True, slots=True)
class MCPReadResourceResult:
    contents: tuple[MCPResourceContent, ...]
    server: str
    total_bytes: int

    def public(self) -> dict[str, Any]:
        return {
            "source": mcp_source_identity(self.server),
            "untrusted": True,
            "trust": "untrusted_external_content",
            "contents": [item.public() for item in self.contents],
        }

    def safe_public(self) -> dict[str, Any]:
        """把外部资源包在仅数据用途的信封中，避免伪装成指令。"""

        result = self.public()
        contents = result.pop("contents", [])
        result["content_policy"] = "quoted_external_data"
        result["external_resource"] = {"contents": contents}
        return result


@dataclass(frozen=True, slots=True)
class MCPPromptArgument:
    name: str
    title: str | None = None
    description: str | None = None
    required: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> MCPPromptArgument:
        if not isinstance(value, Mapping):
            raise ValueError("MCP prompt argument must be an object")
        name = validate_prompt_name(value.get("name"))
        title = (
            _optional_text(value["title"], label="MCP prompt argument title", maximum=1024)
            if "title" in value
            else None
        )
        description = (
            _optional_text(
                value["description"], label="MCP prompt argument description", maximum=4096
            )
            if "description" in value
            else None
        )
        raw_required = value.get("required", False)
        if not isinstance(raw_required, bool):
            raise ValueError("MCP prompt argument required must be a boolean")
        _validate_meta(value, label="MCP prompt argument")
        return cls(name, title, description, raw_required)

    def public(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        if self.title is not None:
            result["title"] = self.title
        if self.description is not None:
            result["description"] = self.description
        if self.required:
            result["required"] = True
        return result


@dataclass(frozen=True, slots=True)
class MCPPrompt:
    name: str
    title: str | None = None
    description: str | None = None
    arguments: tuple[MCPPromptArgument, ...] = ()
    server: str = "default"

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        server: str,
        limits: MCPContentLimits,
    ) -> MCPPrompt:
        if not isinstance(value, Mapping):
            raise ValueError("MCP prompt entry must be an object")
        name = validate_prompt_name(value.get("name"))
        title = (
            _optional_text(value["title"], label="MCP prompt title", maximum=1024)
            if "title" in value
            else None
        )
        description = (
            _optional_text(value["description"], label="MCP prompt description", maximum=8192)
            if "description" in value
            else None
        )
        raw_arguments = value.get("arguments", [])
        if not isinstance(raw_arguments, list):
            raise ValueError("MCP prompt arguments must be a list")
        if len(raw_arguments) > limits.max_prompt_arguments:
            raise ValueError("MCP prompt has too many arguments")
        arguments = tuple(MCPPromptArgument.from_mapping(item) for item in raw_arguments)
        names = [item.name for item in arguments]
        if len(names) != len(set(names)):
            raise ValueError("MCP prompt contains duplicate argument names")
        _validate_meta(value, label="MCP prompt")
        server_name = _required_text(server, label="MCP server name", maximum=128)
        return cls(name, title, description, arguments, server_name)

    def validate_arguments(
        self,
        values: Mapping[str, Any] | None,
        *,
        limits: MCPContentLimits,
    ) -> dict[str, str]:
        if values is None:
            arguments: dict[str, Any] = {}
        elif not isinstance(values, Mapping):
            raise ValueError("MCP prompt arguments must be an object")
        else:
            arguments = dict(values)
        if len(arguments) > limits.max_prompt_arguments:
            raise ValueError("MCP prompt arguments exceed the entry limit")
        declared = {item.name: item for item in self.arguments}
        if any(not isinstance(name, str) or name not in declared for name in arguments):
            raise ValueError("MCP prompt arguments contain an undeclared name")
        missing = [
            item.name for item in self.arguments if item.required and item.name not in arguments
        ]
        if missing:
            raise ValueError("MCP prompt arguments are missing a required value")
        total_bytes = 0
        result: dict[str, str] = {}
        for name, raw_value in arguments.items():
            if not isinstance(raw_value, str) or len(raw_value) > limits.max_argument_chars:
                raise ValueError("MCP prompt argument value is invalid")
            if "\x00" in raw_value:
                raise ValueError("MCP prompt argument value is invalid")
            try:
                encoded = raw_value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("MCP prompt argument value is invalid") from exc
            total_bytes += len(encoded)
            if total_bytes > limits.max_argument_bytes:
                raise ValueError("MCP prompt arguments exceed the byte limit")
            result[name] = raw_value
        return result

    def public(self) -> dict[str, Any]:
        result: dict[str, Any] = {"name": self.name}
        if self.title is not None:
            result["title"] = self.title
        if self.description is not None:
            result["description"] = self.description
        if self.arguments:
            result["arguments"] = [item.public() for item in self.arguments]
        return result


@dataclass(frozen=True, slots=True)
class MCPPromptContent:
    value: Mapping[str, Any]
    byte_size: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        limits: MCPContentLimits,
    ) -> MCPPromptContent:
        if not isinstance(value, Mapping):
            raise ValueError("MCP prompt content must be an object")
        kind = value.get("type")
        if not isinstance(kind, str):
            raise ValueError("MCP prompt content type is invalid")
        annotations = (
            MCPAnnotations.from_value(value["annotations"], label="MCP prompt content")
            if "annotations" in value
            else MCPAnnotations()
        )
        result: dict[str, Any] = {"type": kind}
        byte_size = 0
        if kind == "text":
            text, byte_size = _text_bytes(value.get("text"), label="MCP prompt text", limits=limits)
            result["text"] = text
        elif kind in {"image", "audio"}:
            mime_type = validate_mime_type(value.get("mimeType"), expected_prefix=kind)
            data, decoded = _decoded_base64(
                value.get("data"),
                label=f"MCP prompt {kind} data",
                maximum_bytes=limits.max_block_bytes,
            )
            result.update({"data": data, "mimeType": mime_type})
            byte_size = len(decoded)
        elif kind == "resource":
            raw_resource = value.get("resource")
            if not isinstance(raw_resource, Mapping):
                raise ValueError("MCP embedded resource must be an object")
            resource = MCPResourceContent.from_mapping(raw_resource, limits=limits)
            result["resource"] = resource.public()
            byte_size = resource.byte_size
        elif kind == "resource_link":
            resource = MCPResource.from_mapping(value, server="prompt", limits=limits)
            result.update(resource.public())
            result["type"] = "resource_link"
        else:
            raise ValueError("MCP prompt content type is unsupported")
        public_annotations = annotations.public()
        if public_annotations:
            result["annotations"] = public_annotations
        _validate_meta(value, label="MCP prompt content")
        return cls(result, byte_size)

    def public(self) -> dict[str, Any]:
        result = dict(self.value)
        if isinstance(result.get("annotations"), Mapping):
            result["annotations"] = dict(result["annotations"])
        if isinstance(result.get("resource"), Mapping):
            result["resource"] = dict(result["resource"])
        return result


@dataclass(frozen=True, slots=True)
class MCPPromptMessage:
    role: str
    content: MCPPromptContent

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        limits: MCPContentLimits,
    ) -> MCPPromptMessage:
        if not isinstance(value, Mapping):
            raise ValueError("MCP prompt message must be an object")
        role = value.get("role")
        if role not in {"user", "assistant"}:
            raise ValueError("MCP prompt message role must be user or assistant")
        content = MCPPromptContent.from_mapping(value.get("content"), limits=limits)
        _validate_meta(value, label="MCP prompt message")
        return cls(role, content)

    def public(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content.public()}


@dataclass(frozen=True, slots=True)
class MCPGetPromptResult:
    messages: tuple[MCPPromptMessage, ...]
    server: str
    description: str | None = None
    total_bytes: int = 0

    def public(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "source": mcp_source_identity(self.server),
            "untrusted": True,
            "trust": "untrusted_external_content",
            "messages": [message.public() for message in self.messages],
        }
        if self.description is not None:
            result["description"] = self.description
        return result

    def safe_public(self) -> dict[str, Any]:
        """把外部 prompt 角色降级为数据字段，不生成可执行消息角色。"""

        result = self.public()
        messages = result.pop("messages", [])
        quoted_messages = [
            {
                "speaker": message.get("role"),
                "content": message.get("content"),
            }
            for message in messages
            if isinstance(message, Mapping)
        ]
        prompt_payload: dict[str, Any] = {"messages": quoted_messages}
        if "description" in result:
            prompt_payload["description"] = result.pop("description")
        result["content_policy"] = "quoted_external_data"
        result["external_prompt"] = prompt_payload
        return result


def _next_cursor(
    result: Mapping[str, Any],
    *,
    method: str,
    limits: MCPContentLimits,
) -> tuple[bool, str]:
    if "next_cursor" in result:
        raise MCPProtocolError(f"MCP {method} uses unsupported next_cursor field")
    if "nextCursor" not in result:
        return False, ""
    value = result["nextCursor"]
    if not isinstance(value, str) or len(value) > limits.max_cursor_chars:
        raise MCPProtocolError(f"MCP {method} nextCursor must be a bounded string")
    return True, value


def _safe_server_error(exc: MCPServerError, *, method: str) -> MCPServerError:
    return MCPServerError(exc.code, f"MCP {method} request failed")


async def list_mcp_resources(
    request: MCPRequest,
    *,
    server: str,
    limits: MCPContentLimits = DEFAULT_MCP_CONTENT_LIMITS,
) -> tuple[MCPResource, ...]:
    resources: list[MCPResource] = []
    seen_uris: set[str] = set()
    seen_cursors: set[str] = set()
    cursor: str | None = None
    try:
        async with asyncio.timeout(limits.operation_timeout_seconds):
            for _page in range(limits.max_pages):
                params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
                result = await request("resources/list", params)
                raw_resources = result.get("resources")
                if not isinstance(raw_resources, list):
                    raise MCPProtocolError("MCP resources/list result.resources must be a list")
                if len(resources) + len(raw_resources) > limits.max_list_entries:
                    raise MCPProtocolError("MCP resources/list exceeds the entry limit")
                for raw_resource in raw_resources:
                    try:
                        resource = MCPResource.from_mapping(
                            raw_resource,
                            server=server,
                            limits=limits,
                        )
                    except (TypeError, ValueError) as exc:
                        raise MCPProtocolError(
                            "MCP resources/list contains an invalid resource"
                        ) from exc
                    if resource.uri in seen_uris:
                        raise MCPProtocolError("MCP resources/list contains a duplicate URI")
                    seen_uris.add(resource.uri)
                    resources.append(resource)
                has_cursor, next_cursor = _next_cursor(
                    result, method="resources/list", limits=limits
                )
                if not has_cursor:
                    return tuple(resources)
                if next_cursor in seen_cursors:
                    raise MCPProtocolError("MCP resources/list pagination repeated a cursor")
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            raise MCPProtocolError("MCP resources/list exceeds the page limit")
    except TimeoutError as exc:
        raise MCPTimeoutError("MCP resources/list operation timed out") from exc
    except MCPServerError as exc:
        raise _safe_server_error(exc, method="resources/list") from None


async def read_mcp_resource(
    request: MCPRequest,
    uri: object,
    *,
    server: str,
    limits: MCPContentLimits = DEFAULT_MCP_CONTENT_LIMITS,
) -> MCPReadResourceResult:
    try:
        resource_uri = validate_resource_uri(uri, maximum=limits.max_uri_chars)
    except ValueError as exc:
        raise MCPProtocolError("MCP resources/read URI is invalid") from exc
    try:
        async with asyncio.timeout(limits.operation_timeout_seconds):
            result = await request("resources/read", {"uri": resource_uri})
    except TimeoutError as exc:
        raise MCPTimeoutError("MCP resources/read operation timed out") from exc
    except MCPServerError as exc:
        raise _safe_server_error(exc, method="resources/read") from None
    raw_contents = result.get("contents")
    if not isinstance(raw_contents, list):
        raise MCPProtocolError("MCP resources/read result.contents must be a list")
    if len(raw_contents) > limits.max_content_items:
        raise MCPProtocolError("MCP resources/read exceeds the content item limit")
    contents: list[MCPResourceContent] = []
    total_bytes = 0
    for raw_content in raw_contents:
        try:
            content = MCPResourceContent.from_mapping(raw_content, limits=limits)
        except (TypeError, ValueError) as exc:
            raise MCPProtocolError("MCP resources/read contains invalid content") from exc
        if content.uri != resource_uri:
            raise MCPProtocolError("MCP resources/read content URI does not match request")
        total_bytes += content.byte_size
        if total_bytes > limits.max_total_bytes:
            raise MCPProtocolError("MCP resources/read exceeds the total byte limit")
        contents.append(content)
    try:
        server_name = _required_text(server, label="MCP server name", maximum=128)
    except ValueError as exc:
        raise MCPProtocolError("MCP resources/read source is invalid") from exc
    return MCPReadResourceResult(tuple(contents), server_name, total_bytes)


async def list_mcp_prompts(
    request: MCPRequest,
    *,
    server: str,
    limits: MCPContentLimits = DEFAULT_MCP_CONTENT_LIMITS,
) -> tuple[MCPPrompt, ...]:
    prompts: list[MCPPrompt] = []
    seen_names: set[str] = set()
    seen_cursors: set[str] = set()
    cursor: str | None = None
    try:
        async with asyncio.timeout(limits.operation_timeout_seconds):
            for _page in range(limits.max_pages):
                params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
                result = await request("prompts/list", params)
                raw_prompts = result.get("prompts")
                if not isinstance(raw_prompts, list):
                    raise MCPProtocolError("MCP prompts/list result.prompts must be a list")
                if len(prompts) + len(raw_prompts) > limits.max_list_entries:
                    raise MCPProtocolError("MCP prompts/list exceeds the entry limit")
                for raw_prompt in raw_prompts:
                    try:
                        prompt = MCPPrompt.from_mapping(
                            raw_prompt,
                            server=server,
                            limits=limits,
                        )
                    except (TypeError, ValueError) as exc:
                        raise MCPProtocolError(
                            "MCP prompts/list contains an invalid prompt"
                        ) from exc
                    if prompt.name in seen_names:
                        raise MCPProtocolError("MCP prompts/list contains a duplicate name")
                    seen_names.add(prompt.name)
                    prompts.append(prompt)
                has_cursor, next_cursor = _next_cursor(result, method="prompts/list", limits=limits)
                if not has_cursor:
                    return tuple(prompts)
                if next_cursor in seen_cursors:
                    raise MCPProtocolError("MCP prompts/list pagination repeated a cursor")
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            raise MCPProtocolError("MCP prompts/list exceeds the page limit")
    except TimeoutError as exc:
        raise MCPTimeoutError("MCP prompts/list operation timed out") from exc
    except MCPServerError as exc:
        raise _safe_server_error(exc, method="prompts/list") from None


def validate_prompt_arguments(
    values: Mapping[str, Any] | None,
    *,
    limits: MCPContentLimits,
) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping) or len(values) > limits.max_prompt_arguments:
        raise ValueError("MCP prompt arguments must be a bounded object")
    total_bytes = 0
    result: dict[str, str] = {}
    for raw_name, raw_value in values.items():
        name = validate_prompt_name(raw_name)
        if not isinstance(raw_value, str) or len(raw_value) > limits.max_argument_chars:
            raise ValueError("MCP prompt argument value is invalid")
        if "\x00" in raw_value:
            raise ValueError("MCP prompt argument value is invalid")
        try:
            encoded = raw_value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("MCP prompt argument value is invalid") from exc
        total_bytes += len(encoded)
        if total_bytes > limits.max_argument_bytes:
            raise ValueError("MCP prompt arguments exceed the byte limit")
        result[name] = raw_value
    return result


async def get_mcp_prompt(
    request: MCPRequest,
    name: object,
    arguments: Mapping[str, Any] | None = None,
    *,
    server: str,
    limits: MCPContentLimits = DEFAULT_MCP_CONTENT_LIMITS,
) -> MCPGetPromptResult:
    try:
        prompt_name = validate_prompt_name(name)
        values = validate_prompt_arguments(arguments, limits=limits)
    except ValueError as exc:
        raise MCPProtocolError("MCP prompts/get parameters are invalid") from exc
    params: dict[str, Any] = {"name": prompt_name}
    if arguments is not None:
        params["arguments"] = values
    try:
        async with asyncio.timeout(limits.operation_timeout_seconds):
            result = await request("prompts/get", params)
    except TimeoutError as exc:
        raise MCPTimeoutError("MCP prompts/get operation timed out") from exc
    except MCPServerError as exc:
        raise _safe_server_error(exc, method="prompts/get") from None
    description = (
        _optional_text(result["description"], label="MCP prompt result description", maximum=8192)
        if "description" in result
        else None
    )
    raw_messages = result.get("messages")
    if not isinstance(raw_messages, list):
        raise MCPProtocolError("MCP prompts/get result.messages must be a list")
    if len(raw_messages) > limits.max_messages:
        raise MCPProtocolError("MCP prompts/get exceeds the message limit")
    messages: list[MCPPromptMessage] = []
    total_bytes = 0
    for raw_message in raw_messages:
        try:
            message = MCPPromptMessage.from_mapping(raw_message, limits=limits)
        except (TypeError, ValueError) as exc:
            raise MCPProtocolError("MCP prompts/get contains an invalid message") from exc
        total_bytes += message.content.byte_size
        if total_bytes > limits.max_total_bytes:
            raise MCPProtocolError("MCP prompts/get exceeds the total byte limit")
        messages.append(message)
    try:
        server_name = _required_text(server, label="MCP server name", maximum=128)
    except ValueError as exc:
        raise MCPProtocolError("MCP prompts/get source is invalid") from exc
    _validate_meta(result, label="MCP prompt result")
    return MCPGetPromptResult(tuple(messages), server_name, description, total_bytes)


__all__ = [
    "DEFAULT_MCP_CONTENT_LIMITS",
    "MCPAnnotations",
    "MCPContentLimits",
    "MCPGetPromptResult",
    "MCPPrompt",
    "MCPPromptArgument",
    "MCPPromptContent",
    "MCPPromptMessage",
    "MCPReadResourceResult",
    "MCPResource",
    "MCPResourceContent",
    "get_mcp_prompt",
    "list_mcp_prompts",
    "list_mcp_resources",
    "mcp_source_identity",
    "read_mcp_resource",
    "validate_mime_type",
    "validate_prompt_arguments",
    "validate_prompt_name",
    "validate_resource_uri",
]
