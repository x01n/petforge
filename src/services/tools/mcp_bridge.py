"""把 MCP 远端工具接入本地权限和执行注册表。"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
from collections.abc import Mapping, Sequence
from time import monotonic
from typing import Any

from core.adapters.mcp.protocol import MCPClient, MCPProtocolError, MCPServerError, MCPTool
from logger.events import event_fingerprint, log_event

from .mcp_content import MCPContentService
from .registry import ToolRegistry
from .types import RiskLevel, ToolCallContext, ToolKind, ToolSpec

logger = logging.getLogger(__name__)


def _slug(value: str, *, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    allowed = "abcdefghijklmnopqrstuvwxyz0123456789_.-"
    result = "".join(char if char in allowed else "-" for char in raw).strip(".-_")
    return result or fallback


class MCPToolBridge:
    """把远端工具投影为 ``mcp:server.tool`` 本地身份。"""

    def __init__(
        self,
        client: MCPClient,
        *,
        risk: RiskLevel = RiskLevel.HIGH,
        group: str = "mcp",
        notification_debounce_seconds: float = 0.3,
        notification_retry_delays: Sequence[float] = (0.1, 0.3, 1.0),
    ) -> None:
        self.client = client
        self.risk = RiskLevel(risk)
        self.group = str(group or "mcp").strip() or "mcp"
        if isinstance(notification_debounce_seconds, bool):
            raise ValueError("MCP notification debounce is invalid")
        debounce = float(notification_debounce_seconds)
        if not math.isfinite(debounce) or not 0.01 <= debounce <= 5.0:
            raise ValueError("MCP notification debounce is outside the allowed range")
        if isinstance(notification_retry_delays, (str, bytes, bytearray)):
            raise ValueError("MCP notification retry delays must be a sequence")
        delays = tuple(float(value) for value in notification_retry_delays)
        if len(delays) > 5 or any(
            not math.isfinite(value) or not 0.01 <= value <= 10.0 for value in delays
        ):
            raise ValueError("MCP notification retry delays are invalid")
        self.notification_debounce_seconds = debounce
        self.notification_retry_delays = delays
        self.content = MCPContentService(
            client,
            risk=self.risk,
            group=self.group,
            notification_debounce_seconds=debounce,
            notification_retry_delays=delays,
        )
        self._bindings: dict[str, MCPTool] = {}
        # 一个桥可以在测试/多运行时注册到多个 registry；分别记录对象归属，
        # 这样刷新时只移除自己此前登记的工具，不会误删用户或其他桥接器的工具。
        self._registered_specs: dict[ToolRegistry, dict[str, ToolSpec]] = {}
        self._refresh_lock = asyncio.Lock()
        self._notification_task: asyncio.Task[None] | None = None
        self._notification_count = 0
        self._refresh_count = 0
        self._failure_count = 0
        self._last_status = "idle"
        self._closed = False
        self._unsubscribe_notification: Any | None = None
        self._unsubscribe_close: Any | None = None
        self._unsubscribe_connection: Any | None = None
        self._recovering = False

    @property
    def bindings(self) -> Mapping[str, MCPTool]:
        return dict(self._bindings)

    def status(self) -> Mapping[str, object]:
        """返回计数、刷新状态和只读能力；不包含端点、参数或工具正文。"""

        capabilities = getattr(self.client, "capabilities", None)
        public_capabilities = getattr(capabilities, "public_status", None)
        capability_status = (
            public_capabilities() if callable(public_capabilities) else {"tools": "unsupported"}
        )
        connection = str(getattr(self.client, "connection_state", "unknown") or "unknown")
        if connection not in {"connected", "disconnected", "closed"}:
            connection = "unknown"
        return {
            "status": "closed" if self._closed else self._last_status,
            "connection": connection,
            "server_fingerprint": event_fingerprint(getattr(self.client, "server_name", "mcp")),
            "registered_count": len(self._bindings),
            "registry_count": len(self._registered_specs),
            "notification_count": self._notification_count,
            "refresh_count": self._refresh_count,
            "failure_count": self._failure_count,
            "refresh_pending": bool(
                self._notification_task is not None and not self._notification_task.done()
            ),
            "capabilities": capability_status,
            "content": self.content.status(),
        }

    def _identity_for(self, tool: MCPTool, bindings: Mapping[str, MCPTool]) -> str:
        server = _slug(tool.server, fallback="server")
        name = _slug(tool.name, fallback="tool")
        identity = f"mcp:{server}.{name}"
        existing = bindings.get(identity)
        if existing is not None and existing.server == tool.server and existing.name == tool.name:
            raise ValueError("duplicate MCP tool name")
        if existing is not None and (existing.server != tool.server or existing.name != tool.name):
            digest = hashlib.sha256(f"{tool.server}:{tool.name}".encode()).hexdigest()[:12]
            identity = f"mcp:{server}.{name}-{digest}"
        if len(identity) > 256:
            digest = hashlib.sha256(f"{tool.server}:{tool.name}".encode()).hexdigest()[:12]
            identity = f"mcp:{server[:80]}.{name[:80]}-{digest}"
        if identity in bindings and (
            bindings[identity].server != tool.server or bindings[identity].name != tool.name
        ):
            raise ValueError("MCP tool identity collision")
        return identity

    def identity_for(self, tool: MCPTool) -> str:
        """返回当前桥接器作用域内稳定且无碰撞的本地身份。"""

        return self._identity_for(tool, self._bindings)

    async def discover(self) -> tuple[MCPTool, ...]:
        self._ensure_notification_subscription()
        async with self._refresh_lock:
            tools, bindings = await self._fetch_bindings()
            if self._registered_specs:
                self._apply_bindings(bindings)
            else:
                self._bindings = bindings
            return tools

    async def _fetch_bindings(self) -> tuple[tuple[MCPTool, ...], dict[str, MCPTool]]:
        starter = getattr(self.client, "start", None)
        if callable(starter):
            started = starter()
            if inspect.isawaitable(started):
                await started
        capabilities = getattr(self.client, "capabilities", None)
        content_only = getattr(capabilities, "tools", False) is not True and (
            getattr(capabilities, "resources", False) is True
            or getattr(capabilities, "prompts", False) is True
        )
        tools = () if content_only else tuple(await self.client.list_tools())
        bindings: dict[str, MCPTool] = {}
        for tool in tools:
            if not isinstance(tool, MCPTool):
                raise MCPProtocolError("MCP tools/list returned an invalid tool object")
            identity = self._identity_for(tool, bindings)
            bindings[identity] = tool
        return tools, bindings

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._spec_map(self._bindings, {}, {}).values())

    def _spec_map(
        self,
        bindings: Mapping[str, MCPTool],
        previous_specs: Mapping[str, ToolSpec],
        previous_bindings: Mapping[str, MCPTool],
    ) -> dict[str, ToolSpec]:
        specs: dict[str, ToolSpec] = {}
        for identity, tool in sorted(bindings.items()):
            previous = previous_specs.get(identity)
            if previous is not None and previous_bindings.get(identity) == tool:
                specs[identity] = previous
                continue

            async def handler(
                arguments: Mapping[str, Any],
                context: ToolCallContext,
                *,
                _tool: MCPTool = tool,
            ) -> Mapping[str, Any]:
                del context
                result = await self.client.call_tool(_tool.name, arguments)
                for key in ("isError", "is_error"):
                    if key in result and not isinstance(result[key], bool):
                        raise MCPProtocolError(f"MCP tools/call {key} must be a boolean")
                if result.get("isError") is True or result.get("is_error") is True:
                    raise MCPServerError(-32000, "MCP tool reported an error")
                return result

            specs[identity] = ToolSpec(
                identity=identity,
                description=tool.description or f"MCP 工具 {tool.server}.{tool.name}",
                parameters=tool.input_schema,
                handler=handler,
                risk=self.risk,
                group=self.group,
                public=True,
                read_only=False,
                kind=ToolKind.MCP,
            )
        for content_spec in self.content.specs():
            if content_spec.identity in specs:
                raise ValueError("MCP content tool identity collision")
            previous = previous_specs.get(content_spec.identity)
            if previous is not None and previous is content_spec:
                specs[content_spec.identity] = previous
            else:
                specs[content_spec.identity] = content_spec
        return specs

    def _apply_bindings(self, bindings: Mapping[str, MCPTool]) -> tuple[ToolSpec, ...]:
        """对每个 registry 原子 diff；失败时恢复已同步的 registry。"""

        next_bindings = dict(bindings)
        previous_bindings = dict(self._bindings)
        plans: list[tuple[ToolRegistry, dict[str, ToolSpec], dict[str, ToolSpec]]] = []
        for registry, previous_specs in tuple(self._registered_specs.items()):
            current_specs = self._spec_map(
                next_bindings,
                previous_specs,
                previous_bindings,
            )
            plans.append((registry, dict(previous_specs), current_specs))

        applied: list[tuple[ToolRegistry, dict[str, ToolSpec], dict[str, ToolSpec]]] = []
        try:
            for registry, previous_specs, current_specs in plans:
                registry.sync_owned(previous_specs, current_specs.values())
                applied.append((registry, previous_specs, current_specs))
        except BaseException:
            for registry, previous_specs, current_specs in reversed(applied):
                try:
                    registry.sync_owned(current_specs, previous_specs.values())
                except Exception:
                    # registry 自身仍以对象所有权拒绝不安全覆盖；外层刷新失败。
                    pass
            raise

        self._bindings = next_bindings
        for registry, _previous_specs, current_specs in plans:
            self._registered_specs[registry] = current_specs
        if plans:
            return tuple(plans[0][2].values())
        return tuple(self._spec_map(next_bindings, {}, {}).values())

    def register_into(self, registry: ToolRegistry) -> tuple[ToolSpec, ...]:
        if self._closed:
            raise RuntimeError("MCP tool bridge is closed")
        self._ensure_notification_subscription()
        previous = self._registered_specs.get(registry, {})
        current_specs = self._spec_map(self._bindings, previous, self._bindings)
        registry.sync_owned(previous, current_specs.values())
        self._registered_specs[registry] = current_specs
        return tuple(current_specs.values())

    async def refresh_into(self, registry: ToolRegistry) -> tuple[ToolSpec, ...]:
        if self._closed:
            raise RuntimeError("MCP tool bridge is closed")
        self._registered_specs.setdefault(registry, {})
        self._ensure_notification_subscription()
        await self._refresh_with_retry(raise_on_failure=True)
        return tuple(self._registered_specs.get(registry, {}).values())

    def _ensure_notification_subscription(self) -> None:
        if self._closed or self._unsubscribe_notification is not None:
            return
        subscribe = getattr(self.client, "add_notification_handler", None)
        if callable(subscribe):
            self._unsubscribe_notification = subscribe(self._on_notification)
        close_subscribe = getattr(self.client, "add_close_handler", None)
        if callable(close_subscribe):
            self._unsubscribe_close = close_subscribe(self._client_closing)
        connection_subscribe = getattr(self.client, "add_connection_handler", None)
        if callable(connection_subscribe):
            self._unsubscribe_connection = connection_subscribe(self._client_connection_changed)

    async def _on_notification(self, method: str) -> None:
        if self._closed:
            return
        if method in {
            "notifications/resources/list_changed",
            "notifications/prompts/list_changed",
        }:
            await self.content.on_notification(method)
            return
        if method != "notifications/tools/list_changed":
            return
        capabilities = getattr(self.client, "capabilities", None)
        if getattr(capabilities, "tools_list_changed", False) is not True:
            return
        self._notification_count += 1
        if self._notification_task is None or self._notification_task.done():
            self._notification_task = asyncio.create_task(
                self._notification_refresh_loop(),
                name="mcp-tools-list-changed",
            )

    async def _notification_refresh_loop(self) -> None:
        observed = -1
        try:
            while not self._closed and observed != self._notification_count:
                await asyncio.sleep(self.notification_debounce_seconds)
                observed = self._notification_count
                await self._refresh_with_retry(raise_on_failure=False)
        except asyncio.CancelledError:
            raise
        finally:
            if asyncio.current_task() is self._notification_task:
                self._notification_task = None

    async def _refresh_with_retry(self, *, raise_on_failure: bool) -> tuple[ToolSpec, ...]:
        async with self._refresh_lock:
            if self._closed:
                if raise_on_failure:
                    raise RuntimeError("MCP tool bridge is closed")
                return ()
            started = monotonic()
            log_event(
                logger,
                "mcp.tools.refresh.start",
                component="tools.mcp_bridge",
                status="started",
                correlation_id=event_fingerprint(getattr(self.client, "server_name", "mcp")),
                fields={
                    "notification_count": self._notification_count,
                    "registry_count": len(self._registered_specs),
                },
            )
            last_error: BaseException | None = None
            attempts = len(self.notification_retry_delays) + 1
            for attempt in range(attempts):
                if self._closed:
                    raise asyncio.CancelledError
                try:
                    _tools, bindings = await self._fetch_bindings()
                    previous = dict(self._bindings)
                    specs = self._apply_bindings(bindings)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_error = exc
                    retryable = not isinstance(
                        exc,
                        (MCPProtocolError, MCPServerError, TypeError, ValueError),
                    )
                    if not retryable or attempt >= len(self.notification_retry_delays):
                        break
                    await asyncio.sleep(self.notification_retry_delays[attempt])
                    continue
                added = len(set(bindings) - set(previous))
                removed = len(set(previous) - set(bindings))
                updated = sum(
                    1
                    for identity in set(bindings) & set(previous)
                    if bindings[identity] != previous[identity]
                )
                self._refresh_count += 1
                self._last_status = "ready"
                fingerprint = self._toolset_fingerprint(bindings)
                log_event(
                    logger,
                    "mcp.tools.refresh.complete",
                    component="tools.mcp_bridge",
                    status="completed",
                    correlation_id=event_fingerprint(getattr(self.client, "server_name", "mcp")),
                    duration_ms=(monotonic() - started) * 1000,
                    fields={
                        "added_count": added,
                        "updated_count": updated,
                        "removed_count": removed,
                        "registered_count": len(bindings),
                        "toolset_fingerprint": fingerprint,
                    },
                )
                return specs

            self._failure_count += 1
            self._last_status = "degraded"
            log_event(
                logger,
                "mcp.tools.refresh.failed",
                component="tools.mcp_bridge",
                status="failed",
                level=logging.WARNING,
                correlation_id=event_fingerprint(getattr(self.client, "server_name", "mcp")),
                duration_ms=(monotonic() - started) * 1000,
                reason_code="refresh_failed",
                fields={
                    "attempt_count": attempts,
                    "registered_count": len(self._bindings),
                    "error_type": (
                        type(last_error).__name__ if last_error is not None else "unknown"
                    ),
                },
            )
            if raise_on_failure and last_error is not None:
                raise last_error
            return tuple(
                spec
                for specs_by_registry in self._registered_specs.values()
                for spec in specs_by_registry.values()
            )

    @staticmethod
    def _toolset_fingerprint(bindings: Mapping[str, MCPTool]) -> str:
        payload = [
            {
                "identity": identity,
                "name": tool.name,
                "description": tool.description,
                "schema": tool.input_schema,
                "annotations": tool.annotations,
            }
            for identity, tool in sorted(bindings.items())
        ]
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return event_fingerprint(rendered)

    def _client_closed(self) -> None:
        if self._closed:
            return
        removed = len(self._bindings)
        self._closed = True
        self.content.connection_changed("closed")
        task = self._notification_task
        self._notification_task = None
        if task is not None and not task.done():
            task.cancel()
        self._withdraw_bindings(keep_registries=False)
        self._last_status = "closed"
        log_event(
            logger,
            "mcp.tools.connection",
            component="tools.mcp_bridge",
            status="closed",
            correlation_id=event_fingerprint(getattr(self.client, "server_name", "mcp")),
            fields={"removed_count": removed},
        )

    async def _client_closing(self) -> None:
        task = self._notification_task
        self._client_closed()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await self.content.close()

    def _withdraw_bindings(self, *, keep_registries: bool) -> None:
        for registry, previous in tuple(self._registered_specs.items()):
            try:
                registry.sync_owned(previous, ())
            except Exception:
                for identity, spec in previous.items():
                    registry.unregister(identity, expected=spec)
            if keep_registries:
                self._registered_specs[registry] = {}
        if not keep_registries:
            self._registered_specs.clear()
        self._bindings.clear()

    def _client_connection_changed(self, state: str) -> None:
        if self._closed:
            return
        normalized = str(state or "").strip().casefold()
        self.content.connection_changed(normalized)
        if normalized == "closed":
            self._client_closed()
            return
        if normalized != "disconnected":
            return
        removed = len(self._bindings)
        self._last_status = "degraded"
        self._withdraw_bindings(keep_registries=True)
        log_event(
            logger,
            "mcp.tools.connection",
            component="tools.mcp_bridge",
            status="disconnected",
            level=logging.WARNING,
            correlation_id=event_fingerprint(getattr(self.client, "server_name", "mcp")),
            fields={"removed_count": removed},
        )
        if self._recovering:
            return
        task = self._notification_task
        if task is not None and not task.done():
            task.cancel()
        self._recovering = True
        self._notification_task = asyncio.create_task(
            self._recover_after_disconnect(),
            name="mcp-tools-reconnect",
        )

    async def _recover_after_disconnect(self) -> None:
        try:
            await self._refresh_with_retry(raise_on_failure=False)
        except asyncio.CancelledError:
            raise
        finally:
            self._recovering = False
            if asyncio.current_task() is self._notification_task:
                self._notification_task = None

    async def close(self) -> None:
        task = self._notification_task
        self._client_closed()
        unsubscribe = self._unsubscribe_notification
        self._unsubscribe_notification = None
        if callable(unsubscribe):
            unsubscribe()
        unsubscribe_close = self._unsubscribe_close
        self._unsubscribe_close = None
        if callable(unsubscribe_close):
            unsubscribe_close()
        unsubscribe_connection = self._unsubscribe_connection
        self._unsubscribe_connection = None
        if callable(unsubscribe_connection):
            unsubscribe_connection()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await self.content.close()


__all__ = ["MCPToolBridge"]
