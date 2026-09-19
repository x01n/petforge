"""MCP stdio JSON-RPC 客户端。"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from services.processes import process_group_spawn_kwargs, terminate_process_tree

from .content import (
    MCPContentLimits,
    MCPGetPromptResult,
    MCPPrompt,
    MCPReadResourceResult,
    MCPResource,
    get_mcp_prompt,
    list_mcp_prompts,
    list_mcp_resources,
    read_mcp_resource,
)
from .protocol import (
    MCP_PROTOCOL_VERSION,
    MCP_TOOL_LIST_MAX_ENTRIES,
    MCP_TOOL_LIST_MAX_PAGES,
    MCPCloseHandler,
    MCPConnectionHandler,
    MCPNotificationDispatcher,
    MCPNotificationHandler,
    MCPProtocolError,
    MCPServerCapabilities,
    MCPServerError,
    MCPTimeoutError,
    MCPTool,
    MCPTransportError,
    negotiate_protocol_version,
    reject_json_constant,
    validate_json_value,
)


def _safe_server_error_message(value: object, secrets: Sequence[str] = ()) -> str:
    """截断并脱敏 stdio 服务器返回的错误正文。"""

    text = str(value or "MCP server error").strip() or "MCP server error"
    text = " ".join(text.replace("\x00", " ").replace("\r", " ").replace("\n", " ").split())
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "<redacted>")
    return text[:256]


def _validated_command(command: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(item) for item in command)
    if not values or not values[0].strip():
        raise ValueError("MCP command is required")
    if len(values) > 128 or any(not item or len(item) > 4096 or "\x00" in item for item in values):
        raise ValueError("MCP command contains an invalid argument")
    return values


def _strict_bool(value: object, field_name: str, default: bool | None = None) -> bool:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0", ""}:
        return False
    raise ValueError(f"{field_name} must be a boolean")


@dataclass(frozen=True)
class StdioMCPServerConfig:
    """一个 MCP stdio 服务器的安全启动配置。"""

    name: str
    command: tuple[str, ...]
    cwd: str | None = None
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    request_timeout_seconds: float = 30.0
    startup_timeout_seconds: float = 10.0
    max_message_bytes: int = 4 * 1024 * 1024
    initialize: bool = True

    def __repr__(self) -> str:
        """返回不含环境变量值的配置摘要。"""

        return (
            "StdioMCPServerConfig("
            f"name={self.name!r}, command={self.command!r}, cwd={self.cwd!r}, "
            f"env_names={tuple(sorted(self.env))!r}, "
            f"request_timeout_seconds={self.request_timeout_seconds!r}, "
            f"startup_timeout_seconds={self.startup_timeout_seconds!r}, "
            f"max_message_bytes={self.max_message_bytes!r}, initialize={self.initialize!r})"
        )

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name or len(name) > 128 or any(char in name for char in "\r\n\x00"):
            raise ValueError("MCP server name is invalid")
        command = _validated_command(self.command)
        timeout = float(self.request_timeout_seconds)
        startup = float(self.startup_timeout_seconds)
        if not 0.1 <= timeout <= 600 or not 0.1 <= startup <= 600:
            raise ValueError("MCP timeout is outside the allowed range")
        max_bytes = int(self.max_message_bytes)
        if not 1024 <= max_bytes <= 64 * 1024 * 1024:
            raise ValueError("MCP max_message_bytes is outside the allowed range")
        environment = {str(key): str(value) for key, value in dict(self.env or {}).items()}
        for key, value in environment.items():
            if not key or any(char in key + value for char in "\r\n\x00"):
                raise ValueError("MCP environment contains an invalid value")
        cwd = str(self.cwd).strip() if self.cwd else None
        if cwd is not None and (
            not cwd or len(cwd) > 4096 or any(char in cwd for char in "\r\n\x00")
        ):
            raise ValueError("MCP cwd is invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "cwd", cwd)
        object.__setattr__(self, "env", environment)
        object.__setattr__(self, "request_timeout_seconds", timeout)
        object.__setattr__(self, "startup_timeout_seconds", startup)
        object.__setattr__(self, "max_message_bytes", max_bytes)
        object.__setattr__(self, "initialize", _strict_bool(self.initialize, "MCP initialize"))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> StdioMCPServerConfig:
        if not isinstance(value, Mapping):
            raise ValueError("MCP server must be a mapping")
        command = value.get("command", ())
        if isinstance(command, str):
            # 不调用 shell；字符串只接受单个可执行文件，参数需使用列表。
            command = (command,)
        if not isinstance(command, Sequence) or isinstance(command, (bytes, bytearray)):
            raise ValueError("MCP command must be a list")
        return cls(
            name=str(value.get("name", "default")),
            command=tuple(command),
            cwd=value.get("cwd"),
            env=value.get("env", {}),
            request_timeout_seconds=value.get(
                "request_timeout_seconds", value.get("timeout_seconds", 30.0)
            ),
            startup_timeout_seconds=value.get("startup_timeout_seconds", 10.0),
            max_message_bytes=value.get("max_message_bytes", 4 * 1024 * 1024),
            initialize=value.get("initialize", True),
        )


class StdioMCPClient:
    """换行分隔 JSON-RPC 2.0 的异步 stdio 客户端。

    客户端严格要求 ``initialize`` 返回当前支持的 ``2025-06-18`` 版本；只有
    配置明确跳过握手时才不发送 InitializeRequest。每个请求仍校验 id、错误
    对象和消息大小，关闭时回收整个子进程组。
    """

    def __init__(self, config: StdioMCPServerConfig | Mapping[str, Any]) -> None:
        self.config = (
            config
            if isinstance(config, StdioMCPServerConfig)
            else StdioMCPServerConfig.from_mapping(config)
        )
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[Mapping[str, Any]]] = {}
        # 请求超时/取消后，服务端可能仍会把旧回包写入 stdout；记录有限的
        # 放弃 ID，避免迟到回包误杀仍可复用的 MCP 会话。
        self._abandoned_ids: deque[int] = deque(maxlen=256)
        self._next_id = 0
        self._closed = False
        self._started = False
        self._initialized = False
        self._failure: BaseException | None = None
        # 首次握手失败后，下一代进程使用一个有界恢复宽限，避免系统负载
        # 让无延迟的重试握手在极短启动超时下再次误判；成功后清零。
        self._startup_failures = 0
        self._stderr_tail: deque[str] = deque(maxlen=64)
        self._notifications = MCPNotificationDispatcher()

    @property
    def capabilities(self) -> MCPServerCapabilities:
        return self._notifications.capabilities

    @property
    def connection_state(self) -> str:
        return self._notifications.connection_state

    def add_notification_handler(self, handler: MCPNotificationHandler) -> Callable[[], None]:
        return self._notifications.add_handler(handler)

    def add_close_handler(self, handler: MCPCloseHandler) -> Callable[[], None]:
        return self._notifications.add_close_handler(handler)

    def add_connection_handler(self, handler: MCPConnectionHandler) -> Callable[[], None]:
        return self._notifications.add_connection_handler(handler)

    @property
    def server_name(self) -> str:
        return self.config.name

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    async def _cancel_io_tasks_unlocked(self) -> tuple[asyncio.Task[None], ...]:
        """在生命周期锁内取消并回收 stdout/stderr 读取任务。"""

        current = asyncio.current_task()
        tasks = tuple(
            task
            for task in (self._reader_task, self._stderr_task)
            if task is not None and task is not current
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._reader_task = None
        self._stderr_task = None
        return tasks

    def _abandon_request(self, request_id: int) -> None:
        """记录已经取消或超时的请求，允许其迟到回包被安全丢弃。"""

        if request_id not in self._abandoned_ids:
            self._abandoned_ids.append(request_id)

    async def _terminate_process_unlocked(
        self, process: asyncio.subprocess.Process | None, *, timeout: float = 0.5
    ) -> None:
        """终止并回收独立 worker 进程树，保持跨平台的有界清理契约。"""

        await terminate_process_tree(process, timeout=timeout)

    async def _close_unlocked(self, *, mark_closed: bool) -> None:
        """执行关闭/重启清理；调用方必须持有 ``_start_lock``。"""

        if mark_closed:
            self._closed = True
            await self._notifications.close()
        else:
            self._notifications.set_capabilities(MCPServerCapabilities())
        process = self._process
        await self._cancel_io_tasks_unlocked()
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(MCPTransportError("MCP client is closing"))
        self._pending.clear()
        self._abandoned_ids.clear()
        self._process = None
        if process is not None and process.stdin is not None:
            process.stdin.close()
        await self._terminate_process_unlocked(process)
        self._started = False
        self._initialized = False
        self._failure = None

    async def start(self) -> None:
        if (
            self._started
            and self._process is not None
            and self._process.returncode is None
            and self._failure is None
        ):
            return
        async with self._start_lock:
            if (
                self._started
                and self._process is not None
                and self._process.returncode is None
                and self._failure is None
            ):
                return
            if self._closed:
                raise MCPTransportError("MCP client is closed")
            # 进程意外退出后允许一次新的请求重新拉起服务；旧 reader/stderr
            # 任务必须先回收，且不能把上一代故障带入新进程。
            await self._close_unlocked(mark_closed=False)
            environment = dict(os.environ)
            environment.update(self.config.env)
            kwargs: dict[str, Any] = {
                "stdin": asyncio.subprocess.PIPE,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "cwd": self.config.cwd,
                "env": environment,
                "limit": self.config.max_message_bytes + 1,
            }
            kwargs.update(process_group_spawn_kwargs())
            try:
                self._process = await asyncio.create_subprocess_exec(*self.config.command, **kwargs)
            except (OSError, ValueError) as exc:
                self._started = False
                self._process = None
                raise MCPTransportError("MCP server process could not be started") from exc
            self._reader_task = asyncio.create_task(
                self._read_loop(), name=f"mcp-reader-{self.server_name}"
            )
            if self._process.stderr is not None:
                self._stderr_task = asyncio.create_task(
                    self._stderr_loop(), name=f"mcp-stderr-{self.server_name}"
                )
            if self.config.initialize:
                # 首轮失败仍使用用户配置的严格启动超时；仅重试代际获得
                # 一个上限为 1 秒的恢复宽限，避免调度抖动吞掉即时握手。
                startup_timeout = self.config.startup_timeout_seconds
                if self._startup_failures:
                    startup_timeout = max(
                        startup_timeout,
                        min(self.config.request_timeout_seconds, 1.0),
                    )
                try:
                    # ``_initialize`` 已对 InitializeRequest 使用独立的启动
                    # 超时；这里不再套同值的外层 wait_for，避免两个超时
                    # 同时取消 Future 时把 MCPTimeoutError 降级成裸 TimeoutError。
                    await self._initialize(timeout_seconds=startup_timeout)
                except BaseException:
                    # 启动/握手失败只回收这一代进程；显式 ``close()`` 才是
                    # 不可重用的终态。保留可重试语义，允许瞬时启动超时后
                    # 下一次请求重新拉起 MCP 服务。
                    self._startup_failures = min(self._startup_failures + 1, 8)
                    await self._close_unlocked(mark_closed=False)
                    raise
            # 握手完成前保持未启动状态，让并发调用等待生命周期锁。
            self._started = True
            self._startup_failures = 0
            self._notifications.notify_connection("connected")

    async def _initialize(self, *, timeout_seconds: float | None = None) -> None:
        result = await self._request_raw(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "meapet", "version": "0.1.0"},
            },
            ensure_started=False,
            timeout_seconds=(
                self.config.startup_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
        )
        negotiate_protocol_version(result.get("protocolVersion"))
        self._notifications.set_capabilities(MCPServerCapabilities.from_initialize_result(result))
        # 只有握手成功后才发送 initialized 通知；握手错误时发送通知会
        # 覆盖原始错误，也可能让服务误以为客户端已经完成协商。
        await self._notify("notifications/initialized", {})
        self._initialized = True

    async def _stderr_loop(self) -> None:
        stream = self._process.stderr if self._process is not None else None
        if stream is None:
            return
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if text:
                self._stderr_tail.append(text[:1024])

    async def _read_loop(self) -> None:
        process = self._process
        stream = process.stdout if process is not None else None
        if stream is None:
            return
        failure: BaseException | None = None
        try:
            while True:
                raw = await stream.readline()
                if not raw:
                    failure = MCPTransportError("MCP server closed stdout")
                    break
                if len(raw) > self.config.max_message_bytes:
                    failure = MCPProtocolError("MCP response exceeds the message limit")
                    break
                try:
                    value = json.loads(raw.decode("utf-8"), parse_constant=reject_json_constant)
                except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
                    failure = MCPProtocolError("MCP response is not valid JSON")
                    break
                if not isinstance(value, Mapping):
                    failure = MCPProtocolError("MCP response must be a JSON object")
                    break
                if value.get("jsonrpc") != "2.0":
                    failure = MCPProtocolError("MCP response jsonrpc must be 2.0")
                    break
                raw_id = value.get("id")
                if raw_id is None or isinstance(raw_id, bool):
                    # 仅忽略没有响应 id 的通知；带 result/error 的消息缺少 id 是协议错误。
                    if "result" in value or "error" in value:
                        failure = MCPProtocolError("MCP response id is missing")
                        break
                    self._notifications.dispatch(value.get("method"), value.get("params"))
                    continue
                if not isinstance(raw_id, int):
                    failure = MCPProtocolError("MCP response id is invalid")
                    break
                request_id = raw_id
                if request_id not in self._pending:
                    if request_id in self._abandoned_ids:
                        self._abandoned_ids.remove(request_id)
                        continue
                    failure = MCPProtocolError("MCP response id does not match a pending request")
                    break
                if ("result" in value) == ("error" in value):
                    failure = MCPProtocolError(
                        "MCP response must contain exactly one result or error"
                    )
                    break
                future = self._pending.pop(request_id, None)
                if future is not None and not future.done():
                    future.set_result(dict(value))
        except asyncio.CancelledError:
            return
        except Exception:  # pragma: no cover - defensive transport boundary
            failure = MCPTransportError("MCP stdout reader failed")
        self._notifications.set_capabilities(MCPServerCapabilities())
        self._notifications.notify_connection("disconnected")
        self._failure = failure or MCPTransportError("MCP server is unavailable")
        for future in tuple(self._pending.values()):
            if not future.done():
                future.set_exception(self._failure)
        self._pending.clear()

    async def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise MCPTransportError("MCP server stdin is unavailable")
        payload = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        await self._write(payload)

    async def _write(self, payload: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise MCPTransportError("MCP server stdin is unavailable")
        try:
            raw = (
                json.dumps(
                    dict(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                + b"\n"
            )
        except (TypeError, ValueError) as exc:
            raise MCPProtocolError("MCP request cannot be encoded as JSON") from exc
        if len(raw) > self.config.max_message_bytes:
            raise MCPProtocolError("MCP request exceeds the message limit")
        async with self._write_lock:
            try:
                process.stdin.write(raw)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionError, OSError) as exc:
                raise MCPTransportError("MCP server stdin is unavailable") from exc

    async def _request_raw(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        ensure_started: bool = True,
        timeout_seconds: float | None = None,
    ) -> Mapping[str, Any]:
        if ensure_started:
            await self.start()
        if self._failure is not None:
            raise MCPTransportError("MCP server is unavailable") from self._failure
        self._next_id += 1
        request_id = self._next_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Mapping[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        request_timeout = (
            self.config.request_timeout_seconds
            if timeout_seconds is None
            else float(timeout_seconds)
        )
        try:
            validate_json_value(params or {})
            await self._write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": str(method),
                    "params": dict(params or {}),
                }
            )
            try:
                response = await asyncio.wait_for(future, timeout=request_timeout)
            except TimeoutError as exc:
                self._abandon_request(request_id)
                self._pending.pop(request_id, None)
                raise MCPTimeoutError(f"MCP request timed out: {method}") from exc
        except BaseException:
            self._abandon_request(request_id)
            self._pending.pop(request_id, None)
            raise
        if "error" in response:
            error = response.get("error")
            if not isinstance(error, Mapping):
                raise MCPProtocolError("MCP JSON-RPC error must be an object")
            code = error.get("code", -32603)
            if isinstance(code, bool) or not isinstance(code, int):
                raise MCPProtocolError("MCP JSON-RPC error code is invalid")
            raise MCPServerError(
                code,
                _safe_server_error_message(error.get("message"), tuple(self.config.env.values())),
            )
        result = response.get("result")
        if not isinstance(result, Mapping):
            raise MCPProtocolError("MCP JSON-RPC result must be an object")
        return result

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        name = str(method or "").strip()
        if not name or any(char in name for char in "\r\n\x00"):
            raise ValueError("MCP method is invalid")
        return await self._request_raw(name, params)

    async def list_tools(self) -> tuple[MCPTool, ...]:
        tools: list[MCPTool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _page in range(MCP_TOOL_LIST_MAX_PAGES):
            params: dict[str, Any] = {} if cursor is None else {"cursor": cursor}
            result = await self.request("tools/list", params)
            raw_tools = result.get("tools", ())
            if not isinstance(raw_tools, list):
                raise MCPProtocolError("MCP tools/list result.tools must be a list")
            if len(tools) + len(raw_tools) > MCP_TOOL_LIST_MAX_ENTRIES:
                raise MCPProtocolError("MCP tools/list exceeds the entry limit")
            for raw_tool in raw_tools:
                if not isinstance(raw_tool, Mapping):
                    raise MCPProtocolError("MCP tools/list tool entry must be an object")
                try:
                    tools.append(MCPTool.from_mapping(raw_tool, server=self.server_name))
                except (TypeError, ValueError) as exc:
                    raise MCPProtocolError("MCP tools/list contains an invalid tool") from exc
            next_cursor = result.get("nextCursor", result.get("next_cursor"))
            if next_cursor is None:
                return tuple(tools)
            if not isinstance(next_cursor, str) or not next_cursor.strip():
                raise MCPProtocolError("MCP tools/list nextCursor must be a non-empty string")
            cursor = next_cursor
            if cursor in seen_cursors:
                raise MCPProtocolError("MCP tools/list pagination repeated a cursor")
            seen_cursors.add(cursor)
        raise MCPProtocolError("MCP tools/list exceeds the page limit")

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        remote_name = str(name or "").strip()
        if not remote_name:
            raise ValueError("MCP tool name is required")
        try:
            remote_name = MCPTool(name=remote_name, server=self.server_name).name
        except (TypeError, ValueError) as exc:
            raise MCPProtocolError("MCP tool name is invalid") from exc
        values = dict(arguments or {})
        validate_json_value(values)
        result = await self.request("tools/call", {"name": remote_name, "arguments": values})
        if "content" in result:
            content = result["content"]
            if not isinstance(content, list) or any(
                not isinstance(item, Mapping) for item in content
            ):
                raise MCPProtocolError("MCP tools/call content must be a list of objects")
        if "structuredContent" in result and not isinstance(result["structuredContent"], Mapping):
            raise MCPProtocolError("MCP tools/call structuredContent must be an object")
        for key in ("isError", "is_error"):
            if key in result and not isinstance(result[key], bool):
                raise MCPProtocolError(f"MCP tools/call {key} must be a boolean")
        # 保留 MCP 的 content/structuredContent/isError 结构，供上层审计和展示。
        return dict(result)

    def _content_limits(self) -> MCPContentLimits:
        return MCPContentLimits.for_transport(
            max_message_bytes=self.config.max_message_bytes,
            request_timeout_seconds=self.config.request_timeout_seconds,
        )

    async def list_resources(self) -> tuple[MCPResource, ...]:
        await self.start()
        if not self.capabilities.resources:
            raise MCPProtocolError("MCP server did not declare the resources capability")
        return await list_mcp_resources(
            self.request,
            server=self.server_name,
            limits=self._content_limits(),
        )

    async def read_resource(self, uri: str) -> MCPReadResourceResult:
        await self.start()
        if not self.capabilities.resources:
            raise MCPProtocolError("MCP server did not declare the resources capability")
        return await read_mcp_resource(
            self.request,
            uri,
            server=self.server_name,
            limits=self._content_limits(),
        )

    async def list_prompts(self) -> tuple[MCPPrompt, ...]:
        await self.start()
        if not self.capabilities.prompts:
            raise MCPProtocolError("MCP server did not declare the prompts capability")
        return await list_mcp_prompts(
            self.request,
            server=self.server_name,
            limits=self._content_limits(),
        )

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> MCPGetPromptResult:
        await self.start()
        if not self.capabilities.prompts:
            raise MCPProtocolError("MCP server did not declare the prompts capability")
        return await get_mcp_prompt(
            self.request,
            name,
            arguments,
            server=self.server_name,
            limits=self._content_limits(),
        )

    async def close(self) -> None:
        async with self._start_lock:
            if self._closed and self._process is None and not self._pending:
                await self._notifications.close()
                await self._cancel_io_tasks_unlocked()
                return
            await self._close_unlocked(mark_closed=True)

    async def __aenter__(self) -> StdioMCPClient:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


__all__ = ["StdioMCPClient", "StdioMCPServerConfig"]
