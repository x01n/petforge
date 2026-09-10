from __future__ import annotations

import asyncio
import json
import logging
import sys
import textwrap
from collections.abc import Callable, Mapping

import httpx
import pytest

from core.adapters.mcp import (
    HTTPMCPClient,
    HTTPMCPServerConfig,
    MCPNotificationDispatcher,
    MCPProtocolError,
    MCPServerCapabilities,
    MCPServerError,
    MCPTool,
    MCPTransportError,
    StdioMCPClient,
    StdioMCPServerConfig,
)
from core.adapters.mcp.npm import npm_stdio_config
from services.tools.executor import ToolExecutionService
from services.tools.mcp_bridge import MCPToolBridge
from services.tools.permissions import PermissionService
from services.tools.registry import ToolRegistry
from services.tools.types import ToolCallContext, ToolKind, ToolSpec


class _DynamicClient:
    server_name = "dynamic-secret-server"

    def __init__(self, tools: tuple[MCPTool, ...]) -> None:
        self.tools = tools
        self.capabilities = MCPServerCapabilities(tools=True, tools_list_changed=True)
        self.handlers: set[Callable[[str], object]] = set()
        self.close_handlers: set[Callable[[], object]] = set()
        self.connection_handlers: set[Callable[[str], None]] = set()
        self.list_calls = 0
        self.call_names: list[str] = []
        self.failures = 0
        self.list_entered: asyncio.Event | None = None
        self.list_release: asyncio.Event | None = None
        self.call_entered: asyncio.Event | None = None
        self.call_release: asyncio.Event | None = None
        self.active_lists = 0
        self.max_active_lists = 0

    def add_notification_handler(self, handler: Callable[[str], object]):
        self.handlers.add(handler)

        def unsubscribe() -> None:
            self.handlers.discard(handler)

        return unsubscribe

    def add_close_handler(self, handler: Callable[[], object]):
        self.close_handlers.add(handler)

        def unsubscribe() -> None:
            self.close_handlers.discard(handler)

        return unsubscribe

    def add_connection_handler(self, handler: Callable[[str], None]):
        self.connection_handlers.add(handler)

        def unsubscribe() -> None:
            self.connection_handlers.discard(handler)

        return unsubscribe

    def disconnect(self) -> None:
        for handler in tuple(self.connection_handlers):
            handler("disconnected")

    async def emit(self, method: str = "notifications/tools/list_changed") -> None:
        values = []
        for handler in tuple(self.handlers):
            value = handler(method)
            if asyncio.iscoroutine(value):
                values.append(value)
        if values:
            await asyncio.gather(*values)

    async def list_tools(self) -> tuple[MCPTool, ...]:
        self.list_calls += 1
        self.active_lists += 1
        self.max_active_lists = max(self.max_active_lists, self.active_lists)
        try:
            if self.list_entered is not None:
                self.list_entered.set()
            if self.list_release is not None:
                await self.list_release.wait()
            if self.failures > 0:
                self.failures -= 1
                raise MCPTransportError("transient list failure")
            return self.tools
        finally:
            self.active_lists -= 1

    async def call_tool(
        self, name: str, arguments: Mapping[str, object] | None = None
    ) -> Mapping[str, object]:
        del arguments
        self.call_names.append(name)
        if self.call_entered is not None:
            self.call_entered.set()
        if self.call_release is not None:
            await self.call_release.wait()
        return {"content": [], "structuredContent": {"tool": name}}

    async def close(self) -> None:
        pending = []
        for handler in tuple(self.close_handlers):
            value = handler()
            if asyncio.iscoroutine(value):
                pending.append(value)
        if pending:
            await asyncio.gather(*pending)
        for handler in tuple(self.connection_handlers):
            handler("closed")
        self.handlers.clear()
        self.close_handlers.clear()
        self.connection_handlers.clear()


def _tool(
    name: str,
    *,
    description: str = "",
    properties: Mapping[str, object] | None = None,
    annotations: Mapping[str, object] | None = None,
) -> MCPTool:
    return MCPTool(
        name,
        description,
        {
            "type": "object",
            "properties": dict(properties or {}),
            "additionalProperties": False,
        },
        "dynamic-secret-server",
        dict(annotations or {}),
    )


def test_server_capabilities_are_strict_and_unsupported_features_are_explicit() -> None:
    capabilities = MCPServerCapabilities.from_initialize_result(
        {
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"listChanged": True},
                "prompts": {"listChanged": False},
            }
        }
    )
    assert capabilities.tools_list_changed is True
    assert capabilities.allows_notification("notifications/tools/list_changed") is True
    assert capabilities.public_status() == {
        "tools": "ready",
        "tools_list_changed": True,
        "resources": "declared_unsupported",
        "resources_list_changed": True,
        "prompts": "declared_unsupported",
        "prompts_list_changed": False,
        "sampling": "unsupported",
    }
    with pytest.raises(MCPProtocolError, match="listChanged"):
        MCPServerCapabilities.from_initialize_result(
            {"capabilities": {"tools": {"listChanged": "true"}}}
        )
    with pytest.raises(ValueError, match="properties"):
        MCPTool("bad_schema", input_schema={"type": "object", "properties": []})


def test_notification_dispatcher_ignores_payloads_and_undeclared_capabilities() -> None:
    async def scenario() -> None:
        dispatcher = MCPNotificationDispatcher()
        seen: list[str] = []
        dispatcher.add_handler(lambda method: seen.append(method))
        assert (
            dispatcher.dispatch("notifications/tools/list_changed", {"tool": "untrusted"}) is False
        )
        dispatcher.set_capabilities(MCPServerCapabilities(tools=True, tools_list_changed=True))
        assert dispatcher.dispatch("notifications/tools/list_changed", ["invalid"]) is False
        assert (
            dispatcher.dispatch("notifications/tools/list_changed", {"tool": "untrusted"}) is True
        )
        await asyncio.sleep(0)
        assert seen == ["notifications/tools/list_changed"]
        await dispatcher.close()

        storm = MCPNotificationDispatcher()
        storm.set_capabilities(MCPServerCapabilities(tools=True, tools_list_changed=True))
        entered = asyncio.Event()
        release = asyncio.Event()
        storm_calls = 0

        async def slow_handler(_method: str) -> None:
            nonlocal storm_calls
            storm_calls += 1
            entered.set()
            await release.wait()

        storm.add_handler(slow_handler)
        for _index in range(1_000):
            assert storm.dispatch("notifications/tools/list_changed", {}) is True
        await entered.wait()
        assert len(storm._tasks) == 1
        release.set()
        await asyncio.sleep(0)
        assert storm_calls == 1
        await storm.close()

    asyncio.run(scenario())


def test_bridge_notification_storm_atomically_adds_updates_and_removes_tools(
    caplog,
) -> None:
    async def scenario() -> None:
        client = _DynamicClient(
            (
                _tool("same", description="same", annotations={"readOnlyHint": True}),
                _tool("changed", description="old"),
                _tool("removed", description="removed"),
            )
        )
        bridge = MCPToolBridge(
            client,
            notification_debounce_seconds=0.01,
            notification_retry_delays=(0.01,),
        )
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        same_before = registry.require("mcp:dynamic-secret-server.same")
        changed_before = registry.require("mcp:dynamic-secret-server.changed")

        client.tools = (
            _tool("same", description="same", annotations={"readOnlyHint": True}),
            _tool(
                "changed",
                description="new",
                properties={"value": {"type": "string"}},
            ),
            _tool("added", description="added"),
        )
        await asyncio.gather(*(client.emit() for _ in range(25)))
        await asyncio.sleep(0.08)

        assert client.list_calls == 2
        assert registry.identities() == (
            "mcp:dynamic-secret-server.added",
            "mcp:dynamic-secret-server.changed",
            "mcp:dynamic-secret-server.same",
        )
        assert registry.require("mcp:dynamic-secret-server.same") is same_before
        assert same_before.read_only is False
        assert same_before.risk.name == "HIGH"
        assert registry.require("mcp:dynamic-secret-server.changed") is not changed_before
        assert all(spec.kind is ToolKind.MCP for spec in registry.visible())
        status = bridge.status()
        assert status["registered_count"] == 3
        assert status["notification_count"] == 25
        assert status["refresh_count"] == 2
        assert "dynamic-secret-server" not in repr(status)
        await bridge.close()
        await client.close()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(scenario())
    assert "dynamic-secret-server" not in caplog.text
    assert "description" not in caplog.text
    assert "properties" not in caplog.text


def test_bridge_refresh_is_single_flight_and_retries_transient_failure() -> None:
    async def scenario() -> None:
        client = _DynamicClient((_tool("one"),))
        bridge = MCPToolBridge(
            client,
            notification_debounce_seconds=0.01,
            notification_retry_delays=(0.01, 0.01),
        )
        registry = ToolRegistry()
        await bridge.refresh_into(registry)

        client.tools = (_tool("two"),)
        client.list_entered = asyncio.Event()
        client.list_release = asyncio.Event()
        manual = asyncio.create_task(bridge.refresh_into(registry))
        await client.list_entered.wait()
        await asyncio.gather(*(client.emit() for _ in range(5)))
        client.list_release.set()
        await manual
        await asyncio.sleep(0.05)
        assert client.max_active_lists == 1
        assert registry.identities() == ("mcp:dynamic-secret-server.two",)

        client.tools = (_tool("three"),)
        client.failures = 2
        client.list_entered = None
        client.list_release = None
        await client.emit()
        await asyncio.sleep(0.1)
        assert registry.identities() == ("mcp:dynamic-secret-server.three",)
        assert bridge.status()["failure_count"] == 0
        await bridge.close()
        await client.close()

    asyncio.run(scenario())


def test_bridge_does_not_retry_server_or_protocol_failures() -> None:
    class FailedClient(_DynamicClient):
        def __init__(self, error: Exception) -> None:
            super().__init__((_tool("one"),))
            self.error = error

        async def list_tools(self) -> tuple[MCPTool, ...]:
            self.list_calls += 1
            raise self.error

    async def scenario() -> None:
        for error in (
            MCPServerError(-32602, "invalid request"),
            MCPProtocolError("invalid response"),
        ):
            client = FailedClient(error)
            bridge = MCPToolBridge(
                client,
                notification_retry_delays=(0.01, 0.01, 0.01),
            )
            with pytest.raises(type(error)):
                await bridge.refresh_into(ToolRegistry())
            assert client.list_calls == 1
            await bridge.close()
            await client.close()

    asyncio.run(scenario())


def test_bridge_multi_registry_failure_rolls_back_previously_applied_diff() -> None:
    async def scenario() -> None:
        client = _DynamicClient((_tool("old"),))
        bridge = MCPToolBridge(client, notification_retry_delays=())
        first = ToolRegistry()
        second = ToolRegistry()
        await bridge.refresh_into(first)
        bridge.register_into(second)
        external = ToolSpec(
            "mcp:dynamic-secret-server.blocked",
            "外部占用",
            {"type": "object"},
            lambda _arguments, _context: {"ok": True},
        )
        second.register(external)
        client.tools = (_tool("new"), _tool("blocked"))
        with pytest.raises(ValueError, match="duplicate tool identity"):
            await bridge.refresh_into(first)
        assert first.identities() == ("mcp:dynamic-secret-server.old",)
        assert second.identities() == (
            "mcp:dynamic-secret-server.blocked",
            "mcp:dynamic-secret-server.old",
        )
        assert bridge.bindings.keys() == {"mcp:dynamic-secret-server.old"}
        await bridge.close()
        await client.close()

    asyncio.run(scenario())


def test_running_and_pending_approval_keep_old_mcp_spec_snapshot() -> None:
    async def scenario() -> None:
        client = _DynamicClient((_tool("old_tool"),))
        bridge = MCPToolBridge(client, notification_debounce_seconds=0.01)
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        context = ToolCallContext("profile", "session", "turn")

        pending_service = ToolExecutionService(
            registry,
            PermissionService(auto_allow_low_risk=False),
        )
        pending = await pending_service.execute(
            call_id="approval-call",
            identity="mcp:dynamic-secret-server.old_tool",
            arguments={},
            context=context,
        )
        assert pending.status == "approval_required"
        assert pending.approval is not None

        client.tools = (_tool("new_tool"),)
        await bridge.refresh_into(registry)
        assert registry.get("mcp:dynamic-secret-server.old_tool") is None
        approved = await pending_service.approve_and_execute(
            pending.approval.approval_id,
            context=context,
        )
        assert approved.status == "completed"
        assert client.call_names == ["old_tool"]

        # 正在执行的调用同样持有 execute() 开始时读取的旧冻结 spec。
        client.tools = (_tool("running_tool"),)
        await bridge.refresh_into(registry)
        running_service = ToolExecutionService(
            registry,
            PermissionService(bypass_approval=True),
        )
        client.call_entered = asyncio.Event()
        client.call_release = asyncio.Event()
        running = asyncio.create_task(
            running_service.execute(
                call_id="running-call",
                identity="mcp:dynamic-secret-server.running_tool",
                arguments={},
                context=ToolCallContext("profile", "session", "running-turn"),
            )
        )
        await client.call_entered.wait()
        client.tools = (_tool("replacement"),)
        await bridge.refresh_into(registry)
        client.call_release.set()
        outcome = await running
        assert outcome.status == "completed"
        assert client.call_names[-1] == "running_tool"
        await bridge.close()
        await client.close()

    asyncio.run(scenario())


def test_client_close_cancels_pending_notification_refresh_and_unregisters_tools() -> None:
    async def scenario() -> None:
        client = _DynamicClient((_tool("old"),))
        bridge = MCPToolBridge(client, notification_debounce_seconds=0.01)
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        client.list_entered = asyncio.Event()
        client.list_release = asyncio.Event()
        await client.emit()
        await asyncio.wait_for(client.list_entered.wait(), timeout=1.0)
        await client.close()
        await asyncio.sleep(0)
        assert registry.identities() == ()
        assert bridge.status()["status"] == "closed"
        assert bridge.status()["refresh_pending"] is False
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("mcp-tools-")
        ]

    asyncio.run(scenario())


def test_disconnect_withdraws_stale_tools_and_bounded_reconnect_restores_diff() -> None:
    async def scenario() -> None:
        client = _DynamicClient((_tool("old"),))
        bridge = MCPToolBridge(
            client,
            notification_debounce_seconds=0.01,
            notification_retry_delays=(0.01, 0.01),
        )
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        client.tools = (_tool("after_reconnect"),)
        client.failures = 1
        client.disconnect()
        assert registry.identities() == ()
        await asyncio.sleep(0.08)
        assert registry.identities() == ("mcp:dynamic-secret-server.after_reconnect",)
        assert bridge.status()["status"] == "ready"
        await bridge.close()
        await client.close()

    asyncio.run(scenario())


def test_repeated_disconnect_during_recovery_stays_within_retry_budget() -> None:
    class DisconnectingClient(_DynamicClient):
        async def list_tools(self) -> tuple[MCPTool, ...]:
            self.list_calls += 1
            if self.list_calls > 1:
                self.disconnect()
                raise MCPTransportError("repeated disconnect")
            return self.tools

    async def scenario() -> None:
        client = DisconnectingClient((_tool("old"),))
        bridge = MCPToolBridge(
            client,
            notification_debounce_seconds=0.01,
            notification_retry_delays=(0.01, 0.01),
        )
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        client.disconnect()
        await asyncio.sleep(0.08)
        assert client.list_calls == 4  # initial + 3 bounded recovery attempts
        assert registry.identities() == ()
        assert bridge.status()["status"] == "degraded"
        assert bridge.status()["failure_count"] == 1
        assert bridge.status()["refresh_pending"] is False
        await bridge.close()
        await client.close()

    asyncio.run(scenario())


def test_stdio_list_changed_notification_refreshes_bridge() -> None:
    script = textwrap.dedent(
        """
        import json, sys
        lists = 0
        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            if method == "notifications/initialized":
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": True}},
                }
            elif method == "tools/list":
                lists += 1
                name = "old" if lists == 1 else "new"
                result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}]}
            else:
                result = {}
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
            print(json.dumps(response), flush=True)
            if method == "tools/list" and lists == 1:
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/tools/list_changed",
                    "params": {"ignored": "payload"},
                }), flush=True)
        """
    )

    async def scenario() -> None:
        client = StdioMCPClient(
            StdioMCPServerConfig(
                "stdio-dynamic",
                (sys.executable, "-u", "-c", script),
                request_timeout_seconds=2.0,
            )
        )
        bridge = MCPToolBridge(client, notification_debounce_seconds=0.01)
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        assert registry.identities() == ("mcp:stdio-dynamic.old",)
        await asyncio.sleep(0.08)
        assert registry.identities() == ("mcp:stdio-dynamic.new",)
        assert client.capabilities.tools_list_changed is True
        await client.close()
        assert registry.identities() == ()
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("mcp-tools-")
        ]

    asyncio.run(scenario())


def test_declared_stdio_invalid_notification_is_ignored_without_disconnect() -> None:
    script = textwrap.dedent(
        """
        import json, sys
        lists = 0
        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            if method == "notifications/initialized":
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": True}},
                }
            elif method == "tools/list":
                lists += 1
                result = {"tools": [{"name": "old" if lists == 1 else "new"}]}
            else:
                result = {}
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
            print(json.dumps(response), flush=True)
            if method == "tools/list":
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/tools/list_changed",
                    "params": ["invalid"],
                }), flush=True)
        """
    )

    async def scenario() -> None:
        client = StdioMCPClient(
            StdioMCPServerConfig(
                "stdio-static",
                (sys.executable, "-u", "-c", script),
                request_timeout_seconds=2.0,
            )
        )
        bridge = MCPToolBridge(client, notification_debounce_seconds=0.01)
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        await asyncio.sleep(0.05)
        assert registry.identities() == ("mcp:stdio-static.old",)
        assert bridge.status()["notification_count"] == 0
        assert [tool.name for tool in await client.list_tools()] == ["new"]
        await client.close()

    asyncio.run(scenario())


def test_stdio_disconnect_restarts_process_and_reconciles_tools(tmp_path) -> None:
    marker = tmp_path / "mcp-generation"
    script = textwrap.dedent(
        """
        import json, os, pathlib, sys
        marker = pathlib.Path(os.environ["MEAPET_MCP_TEST_MARKER"])
        first = not marker.exists()
        marker.write_text("started", encoding="utf-8")
        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            if method == "notifications/initialized":
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": True}},
                }
            elif method == "tools/list":
                result = {"tools": [{"name": "old" if first else "reconnected"}]}
            else:
                result = {}
            response = {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
            print(json.dumps(response), flush=True)
            if first and method == "tools/list":
                raise SystemExit(0)
        """
    )

    async def scenario() -> None:
        client = StdioMCPClient(
            StdioMCPServerConfig(
                "stdio-reconnect",
                (sys.executable, "-u", "-c", script),
                env={"MEAPET_MCP_TEST_MARKER": str(marker)},
                request_timeout_seconds=2.0,
            )
        )
        bridge = MCPToolBridge(
            client,
            notification_debounce_seconds=0.01,
            notification_retry_delays=(0.01, 0.03),
        )
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        for _attempt in range(100):
            if registry.identities() == ("mcp:stdio-reconnect.reconnected",):
                break
            await asyncio.sleep(0.01)
        assert registry.identities() == ("mcp:stdio-reconnect.reconnected",)
        assert bridge.status()["status"] == "ready"
        await client.close()

    asyncio.run(scenario())


def test_streamable_http_and_legacy_sse_deliver_dynamic_notification() -> None:
    async def streamable_scenario() -> None:
        list_calls = 0
        get_calls = 0
        initialize_capabilities: object = None

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal initialize_capabilities, list_calls, get_calls
            if request.method == "GET":
                get_calls += 1
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=(
                        b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n\n'
                        if get_calls == 1
                        else b": keepalive\n\n"
                    ),
                )
            body = json.loads(request.content)
            method = body["method"]
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "initialize":
                initialize_capabilities = body.get("params", {}).get("capabilities")
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": True}},
                }
            elif method == "tools/list":
                list_calls += 1
                result = {"tools": [{"name": "old" if list_calls == 1 else "new"}]}
            else:
                result = {}
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
            )

        client = HTTPMCPClient(
            HTTPMCPServerConfig("http-dynamic", "https://mcp.invalid/rpc"),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        bridge = MCPToolBridge(client, notification_debounce_seconds=0.01)
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        await asyncio.sleep(0.15)
        assert 2 <= get_calls <= 3
        assert list_calls == 2
        assert initialize_capabilities == {}
        assert registry.identities() == ("mcp:http-dynamic.new",)
        await client.close()
        assert not [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith(("mcp-http-events-", "mcp-tools-"))
        ]

    async def legacy_sse_scenario() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=b"event: endpoint\ndata: /messages\n\n",
                )
            body = json.loads(request.content)
            if body["method"] == "notifications/initialized":
                return httpx.Response(202)
            result = (
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": True}},
                }
                if body["method"] == "initialize"
                else {"tools": [{"name": "legacy"}]}
            )
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
            )

        client = HTTPMCPClient(
            HTTPMCPServerConfig(
                "legacy-sse",
                "https://mcp.invalid/sse",
                transport="sse",
            ),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        seen: list[str] = []
        client.add_notification_handler(lambda method: seen.append(method))
        await client.list_tools()
        await client._deliver_response(
            {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}
        )
        await asyncio.sleep(0)
        assert seen == ["notifications/tools/list_changed"]
        await client.close()

    asyncio.run(streamable_scenario())
    asyncio.run(legacy_sse_scenario())


def test_streamable_http_405_notification_stream_keeps_post_tools_available() -> None:
    async def scenario() -> None:
        methods: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            methods.append(request.method)
            if request.method == "GET":
                return httpx.Response(405)
            body = json.loads(request.content)
            if body["method"] == "notifications/initialized":
                return httpx.Response(202)
            result = (
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": True}},
                }
                if body["method"] == "initialize"
                else {"tools": [{"name": "static_after_405"}]}
            )
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
            )

        client = HTTPMCPClient(
            HTTPMCPServerConfig("http-405", "https://mcp.invalid/rpc"),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        bridge = MCPToolBridge(client, notification_debounce_seconds=0.01)
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        await asyncio.sleep(0.02)
        assert "GET" in methods
        assert registry.identities() == ("mcp:http-405.static_after_405",)
        assert bridge.status()["status"] == "ready"
        await client.list_tools()
        assert methods.count("GET") == 1
        await client.close()

    asyncio.run(scenario())


def test_streamable_http_session_404_reinitializes_without_stale_session() -> None:
    async def scenario() -> None:
        initialize_headers: list[str | None] = []
        initialize_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal initialize_count
            if request.method == "GET":
                if request.headers.get("mcp-session-id") == "expired-session":
                    return httpx.Response(404)
                return httpx.Response(405)
            body = json.loads(request.content)
            if body["method"] == "notifications/initialized":
                return httpx.Response(202)
            if body["method"] == "initialize":
                initialize_headers.append(request.headers.get("mcp-session-id"))
                initialize_count += 1
                return httpx.Response(
                    200,
                    headers={
                        "mcp-session-id": (
                            "expired-session" if initialize_count == 1 else "new-session"
                        )
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {"tools": {"listChanged": True}},
                        },
                    },
                )
            result = {
                "tools": [{"name": "old" if initialize_count == 1 else "after_session_reconnect"}]
            }
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": body.get("id"), "result": result},
            )

        client = HTTPMCPClient(
            HTTPMCPServerConfig("http-session", "https://mcp.invalid/rpc"),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        bridge = MCPToolBridge(
            client,
            notification_debounce_seconds=0.01,
            notification_retry_delays=(0.01, 0.03),
        )
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        for _attempt in range(100):
            if registry.identities() == ("mcp:http-session.after_session_reconnect",):
                break
            await asyncio.sleep(0.01)
        assert initialize_count == 2
        assert initialize_headers == [None, None]
        assert registry.identities() == ("mcp:http-session.after_session_reconnect",)
        await client.close()

    asyncio.run(scenario())


def test_streamable_http_post_404_reinitializes_before_retrying_tools_list() -> None:
    async def scenario() -> None:
        initialize_headers: list[str | None] = []
        initialize_count = 0
        expired_list_seen = False

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal expired_list_seen, initialize_count
            if request.method == "GET":
                return httpx.Response(405)
            body = json.loads(request.content)
            method = body["method"]
            if method == "notifications/initialized":
                return httpx.Response(202)
            if method == "initialize":
                initialize_headers.append(request.headers.get("mcp-session-id"))
                initialize_count += 1
                return httpx.Response(
                    200,
                    headers={
                        "mcp-session-id": (
                            "expired-session" if initialize_count == 1 else "new-session"
                        )
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": body.get("id"),
                        "result": {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {"tools": {"listChanged": True}},
                        },
                    },
                )
            if request.headers.get("mcp-session-id") == "expired-session":
                expired_list_seen = True
                return httpx.Response(404)
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": body.get("id"),
                    "result": {"tools": [{"name": "post_reconnected"}]},
                },
            )

        client = HTTPMCPClient(
            HTTPMCPServerConfig("http-post-session", "https://mcp.invalid/rpc"),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        bridge = MCPToolBridge(
            client,
            notification_debounce_seconds=0.01,
            notification_retry_delays=(0.01, 0.03),
        )
        registry = ToolRegistry()
        await bridge.refresh_into(registry)
        assert expired_list_seen is True
        assert initialize_count == 2
        assert initialize_headers == [None, None]
        assert registry.identities() == ("mcp:http-post-session.post_reconnected",)
        await client.close()

    asyncio.run(scenario())


def test_npm_transport_reuses_dynamic_stdio_client_contract() -> None:
    config = npm_stdio_config({"name": "npm", "package": "mcp-tools"})
    client = StdioMCPClient(config)
    unsubscribe = client.add_notification_handler(lambda _method: None)
    assert callable(unsubscribe)
    assert client.capabilities.public_status()["tools_list_changed"] is False
    asyncio.run(client.close())
