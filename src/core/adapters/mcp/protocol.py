"""MCP/JSON-RPC 2.0 的无第三方依赖数据契约。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from core.json_values import validate_json_value as validate_shared_json_value

if TYPE_CHECKING:
    from .content import MCPGetPromptResult, MCPPrompt, MCPReadResourceResult, MCPResource


# tools/list 分页必须有统一上限，避免恶意或失控服务端造成无限请求与内存增长。
MCP_TOOL_LIST_MAX_PAGES = 64
MCP_TOOL_LIST_MAX_ENTRIES = 1024


class MCPError(RuntimeError):
    """MCP 客户端错误基类。"""


class MCPTransportError(MCPError):
    """stdio 进程或管道不可用。"""


class MCPProtocolError(MCPError):
    """JSON-RPC 或 MCP 响应不符合协议。"""


class MCPTimeoutError(MCPTransportError):
    """请求在边界时间内没有得到响应。"""


class MCPServerError(MCPError):
    """远端 JSON-RPC error 对象。"""

    def __init__(self, code: int, message: str, data: object = None) -> None:
        self.code = int(code)
        self.data = data
        super().__init__(str(message or "MCP server error").strip() or "MCP server error")


MCPNotificationHandler = Callable[[str], Awaitable[None] | None]
MCPCloseHandler = Callable[[], Awaitable[None] | None]
MCPConnectionHandler = Callable[[str], None]

MCP_PROTOCOL_VERSION = "2025-06-18"


def negotiate_protocol_version(value: object) -> str:
    """严格接受当前客户端支持的 MCP 协议版本。"""

    if not isinstance(value, str) or value != MCP_PROTOCOL_VERSION:
        raise MCPProtocolError("MCP protocol version is unsupported")
    return MCP_PROTOCOL_VERSION


MCP_LIST_CHANGED_NOTIFICATIONS = frozenset(
    {
        "notifications/tools/list_changed",
        "notifications/resources/list_changed",
        "notifications/prompts/list_changed",
    }
)


@dataclass(frozen=True, slots=True)
class MCPServerCapabilities:
    """初始化结果中的服务器能力；仅协商服务器实际声明的能力。"""

    tools: bool = False
    tools_list_changed: bool = False
    resources: bool = False
    resources_list_changed: bool = False
    prompts: bool = False
    prompts_list_changed: bool = False

    @staticmethod
    def _feature(capabilities: Mapping[str, Any], name: str) -> tuple[bool, bool]:
        value = capabilities.get(name)
        if value is None:
            return False, False
        if not isinstance(value, Mapping):
            raise MCPProtocolError(f"MCP server capability {name} must be an object")
        if (
            name == "resources"
            and "subscribe" in value
            and not isinstance(value["subscribe"], bool)
        ):
            raise MCPProtocolError("MCP server capability resources.subscribe must be a boolean")
        list_changed = value.get("listChanged", False)
        if not isinstance(list_changed, bool):
            raise MCPProtocolError(f"MCP server capability {name}.listChanged must be a boolean")
        return True, list_changed

    @classmethod
    def from_initialize_result(cls, result: Mapping[str, Any]) -> MCPServerCapabilities:
        if not isinstance(result, Mapping):
            raise MCPProtocolError("MCP initialize result must be an object")
        raw = result.get("capabilities", {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise MCPProtocolError("MCP server capabilities must be an object")
        tools, tools_changed = cls._feature(raw, "tools")
        resources, resources_changed = cls._feature(raw, "resources")
        prompts, prompts_changed = cls._feature(raw, "prompts")
        return cls(
            tools=tools,
            tools_list_changed=tools_changed,
            resources=resources,
            resources_list_changed=resources_changed,
            prompts=prompts,
            prompts_list_changed=prompts_changed,
        )

    def allows_notification(self, method: object) -> bool:
        normalized = str(method or "").strip()
        return (
            (normalized == "notifications/tools/list_changed" and self.tools_list_changed)
            or (
                normalized == "notifications/resources/list_changed" and self.resources_list_changed
            )
            or (normalized == "notifications/prompts/list_changed" and self.prompts_list_changed)
        )

    def public_status(self) -> dict[str, object]:
        """只返回能力状态，不包含服务器名、端点或通知载荷。"""

        return {
            "tools": "ready" if self.tools else "unsupported",
            "tools_list_changed": self.tools_list_changed,
            # 该字段描述模型默认公开面；只读内容服务通过显式调用提供，
            # 因而仍不把 resources/prompts 当作模型自动能力暴露。
            "resources": "declared_unsupported" if self.resources else "unsupported",
            "resources_list_changed": self.resources_list_changed,
            "prompts": "declared_unsupported" if self.prompts else "unsupported",
            "prompts_list_changed": self.prompts_list_changed,
            "sampling": "unsupported",
        }


class MCPNotificationDispatcher:
    """隔离服务器通知：仅分发已声明的固定方法名，不传递不可信参数。"""

    def __init__(self) -> None:
        self._capabilities = MCPServerCapabilities()
        self._handlers: set[MCPNotificationHandler] = set()
        self._close_handlers: set[MCPCloseHandler] = set()
        self._connection_handlers: set[MCPConnectionHandler] = set()
        self._dispatch_keys: set[tuple[MCPNotificationHandler, str]] = set()
        self._connection_state = "disconnected"
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    @property
    def capabilities(self) -> MCPServerCapabilities:
        return self._capabilities

    @property
    def has_handlers(self) -> bool:
        return bool(self._handlers)

    def set_capabilities(self, value: MCPServerCapabilities) -> None:
        if not isinstance(value, MCPServerCapabilities):
            raise TypeError("MCP capabilities must use MCPServerCapabilities")
        self._capabilities = value

    def add_handler(self, handler: MCPNotificationHandler) -> Callable[[], None]:
        if not callable(handler):
            raise TypeError("MCP notification handler must be callable")
        if self._closed:
            raise MCPTransportError("MCP notification dispatcher is closed")
        self._handlers.add(handler)

        def unsubscribe() -> None:
            self._handlers.discard(handler)

        return unsubscribe

    def add_close_handler(self, handler: MCPCloseHandler) -> Callable[[], None]:
        if not callable(handler):
            raise TypeError("MCP close handler must be callable")
        if self._closed:
            raise MCPTransportError("MCP notification dispatcher is closed")
        self._close_handlers.add(handler)

        def unsubscribe() -> None:
            self._close_handlers.discard(handler)

        return unsubscribe

    @property
    def connection_state(self) -> str:
        return self._connection_state

    def add_connection_handler(self, handler: MCPConnectionHandler) -> Callable[[], None]:
        if not callable(handler):
            raise TypeError("MCP connection handler must be callable")
        if self._closed:
            raise MCPTransportError("MCP notification dispatcher is closed")
        self._connection_handlers.add(handler)

        def unsubscribe() -> None:
            self._connection_handlers.discard(handler)

        return unsubscribe

    def notify_connection(self, state: str) -> None:
        normalized = str(state or "").strip().casefold()
        if normalized not in {"connected", "disconnected", "closed"}:
            raise ValueError("MCP connection state is invalid")
        if self._closed and normalized != "closed":
            return
        if normalized == self._connection_state:
            return
        self._connection_state = normalized
        for handler in tuple(self._connection_handlers):
            try:
                handler(normalized)
            except Exception:
                pass

    def dispatch(self, method: object, params: object = None) -> bool:
        """返回通知是否被接受；载荷只校验形状，永不交给业务回调。"""

        if self._closed or not isinstance(method, str):
            return False
        normalized = method.strip()
        if normalized not in MCP_LIST_CHANGED_NOTIFICATIONS:
            return False
        if params is not None and not isinstance(params, Mapping):
            return False
        if not self._capabilities.allows_notification(normalized):
            return False
        for handler in tuple(self._handlers):
            dispatch_key = (handler, normalized)
            if dispatch_key in self._dispatch_keys:
                continue
            self._dispatch_keys.add(dispatch_key)

            async def invoke(
                selected: MCPNotificationHandler = handler,
                selected_method: str = normalized,
            ) -> None:
                try:
                    result = selected(selected_method)
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # 通知回调是旁路；失败不能杀死 JSON-RPC 读取循环。
                    return

            task = asyncio.create_task(invoke(), name="mcp-notification-dispatch")
            self._tasks.add(task)

            def finished(
                done: asyncio.Task[None],
                selected_key: tuple[MCPNotificationHandler, str] = dispatch_key,
            ) -> None:
                self._tasks.discard(done)
                self._dispatch_keys.discard(selected_key)

            task.add_done_callback(finished)
        return True

    async def close(self) -> None:
        self._closed = True
        close_handlers = tuple(self._close_handlers)
        self._close_handlers.clear()
        for handler in close_handlers:
            try:
                result = handler()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass
        self.notify_connection("closed")
        self._handlers.clear()
        self._dispatch_keys.clear()
        self._connection_handlers.clear()
        tasks = tuple(self._tasks)
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class MCPClient(Protocol):
    """MCP 传输客户端的最小异步契约。

    stdio、HTTP 和 SSE 客户端均实现该契约，工具桥接层不依赖具体传输。
    """

    @property
    def server_name(self) -> str: ...

    @property
    def capabilities(self) -> MCPServerCapabilities: ...

    @property
    def connection_state(self) -> str: ...

    def add_notification_handler(self, handler: MCPNotificationHandler) -> Callable[[], None]: ...

    def add_close_handler(self, handler: MCPCloseHandler) -> Callable[[], None]: ...

    def add_connection_handler(self, handler: MCPConnectionHandler) -> Callable[[], None]: ...

    async def list_tools(self) -> tuple[MCPTool, ...]: ...

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]: ...

    async def list_resources(self) -> tuple[MCPResource, ...]: ...

    async def read_resource(self, uri: str) -> MCPReadResourceResult: ...

    async def list_prompts(self) -> tuple[MCPPrompt, ...]: ...

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> MCPGetPromptResult: ...

    async def close(self) -> None: ...


def _safe_text(value: object, field_name: str, maximum: int = 256) -> str:
    text = str(value or "").strip()
    if not text or len(text) > maximum or any(char in text for char in "\r\n\x00"):
        raise ValueError(f"{field_name} is invalid")
    return text


def validate_tool_name(value: object) -> str:
    """校验 MCP 工具名并返回规范化文本。"""

    name = _safe_text(value, "MCP tool name", 128)
    if any(
        char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
        for char in name
    ):
        raise ValueError("MCP tool name contains unsupported characters")
    return name


@dataclass(frozen=True)
class MCPTool:
    """``tools/list`` 返回的远端工具定义。"""

    name: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    server: str = "default"
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = validate_tool_name(self.name)
        server = _safe_text(self.server or "default", "MCP server", 128)
        if self.input_schema is not None and not isinstance(self.input_schema, Mapping):
            raise ValueError("MCP tool inputSchema must be an object")
        schema = dict(self.input_schema or {})
        if schema.get("type", "object") != "object":
            raise ValueError("MCP tool inputSchema must be an object schema")
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})
        if not isinstance(schema.get("properties"), Mapping):
            raise ValueError("MCP tool inputSchema properties must be an object")
        validate_shared_json_value(schema, label="MCP tool inputSchema")
        if self.annotations is not None and not isinstance(self.annotations, Mapping):
            raise ValueError("MCP tool annotations must be an object")
        annotations = dict(self.annotations or {})
        validate_shared_json_value(annotations, label="MCP tool annotations")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "server", server)
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "input_schema", schema)
        object.__setattr__(self, "annotations", annotations)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, server: str = "default") -> MCPTool:
        if not isinstance(value, Mapping):
            raise MCPProtocolError("MCP tool entry must be an object")
        return cls(
            name=value.get("name", ""),
            description=value.get("description", ""),
            input_schema=value.get("inputSchema", value.get("input_schema", {})),
            server=server,
            annotations=value.get("annotations", {}),
        )


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def validate_json_value(value: object) -> None:
    """拒绝 MCP 工具参数中的 NaN/Infinity，避免 JSON 编码差异。"""

    validate_shared_json_value(value, label="MCP JSON")


__all__ = [
    "MCPError",
    "MCPClient",
    "MCPCloseHandler",
    "MCPConnectionHandler",
    "MCP_LIST_CHANGED_NOTIFICATIONS",
    "MCP_PROTOCOL_VERSION",
    "MCP_TOOL_LIST_MAX_ENTRIES",
    "MCP_TOOL_LIST_MAX_PAGES",
    "MCPNotificationHandler",
    "MCPNotificationDispatcher",
    "MCPProtocolError",
    "MCPServerCapabilities",
    "MCPServerError",
    "MCPTimeoutError",
    "MCPTool",
    "MCPTransportError",
    "negotiate_protocol_version",
    "reject_json_constant",
    "validate_tool_name",
    "validate_json_value",
]
