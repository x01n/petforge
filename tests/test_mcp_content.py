from __future__ import annotations

import asyncio
import base64
import sys
from collections.abc import Mapping

import pytest

from app.runtime import build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from core.adapters.mcp import (
    MCPContentLimits,
    MCPGetPromptResult,
    MCPPrompt,
    MCPPromptContent,
    MCPPromptMessage,
    MCPProtocolError,
    MCPReadResourceResult,
    MCPResource,
    MCPResourceContent,
    MCPServerCapabilities,
    NpmMCPServerConfig,
    StdioMCPClient,
    StdioMCPServerConfig,
    get_mcp_prompt,
    list_mcp_prompts,
    list_mcp_resources,
    read_mcp_resource,
)
from services.tools import (
    MCPToolBridge,
    PermissionService,
    ToolCallContext,
    ToolExecutionService,
    ToolRegistry,
)

LIMITS = MCPContentLimits(
    max_pages=4,
    max_list_entries=8,
    max_cursor_chars=16,
    max_uri_chars=128,
    max_messages=4,
    max_content_items=4,
    max_text_chars=8,
    max_block_bytes=16,
    max_total_bytes=24,
    max_prompt_arguments=2,
    max_argument_chars=8,
    max_argument_bytes=16,
    operation_timeout_seconds=0.2,
)


async def _request_from(
    values: list[Mapping[str, object]], seen: list[tuple[str, Mapping[str, object]]]
):
    async def request(method: str, params: Mapping[str, object] | None = None):
        seen.append((method, params or {}))
        if not values:
            raise AssertionError("unexpected MCP request")
        return values.pop(0)

    return request


def _resource(uri: str = "file:///one", name: str = "one") -> dict[str, object]:
    return {"uri": uri, "name": name, "mimeType": "text/plain"}


def _prompt(name: str = "review") -> dict[str, object]:
    return {
        "name": name,
        "description": "external description",
        "arguments": [{"name": "code", "required": True}],
    }


def test_resources_and_prompts_follow_2025_pagination_and_strict_shapes() -> None:
    async def scenario() -> None:
        seen: list[tuple[str, Mapping[str, object]]] = []
        resources_request = await _request_from(
            [{"resources": [_resource()], "nextCursor": ""}, {"resources": []}], seen
        )
        resources = await list_mcp_resources(resources_request, server="secret", limits=LIMITS)
        assert [value.uri for value in resources] == ["file:///one"]
        assert seen == [("resources/list", {}), ("resources/list", {"cursor": ""})]

        seen.clear()
        prompts_request = await _request_from(
            [{"prompts": [_prompt()], "nextCursor": "next"}, {"prompts": []}], seen
        )
        prompts = await list_mcp_prompts(prompts_request, server="secret", limits=LIMITS)
        assert [value.name for value in prompts] == ["review"]
        assert seen[-1] == ("prompts/list", {"cursor": "next"})

        repeated = await _request_from(
            [{"resources": [], "nextCursor": "same"}, {"resources": [], "nextCursor": "same"}],
            [],
        )
        with pytest.raises(MCPProtocolError, match="repeated"):
            await list_mcp_resources(repeated, server="secret", limits=LIMITS)

        malformed = await _request_from([{"resources": [{"uri": "not a uri", "name": "x"}]}], [])
        with pytest.raises(MCPProtocolError, match="invalid resource"):
            await list_mcp_resources(malformed, server="secret", limits=LIMITS)

    asyncio.run(scenario())


def test_runtime_exposes_explicit_console_content_api_and_closes_bridges(tmp_path) -> None:
    """内容浏览必须走运行时权限管线，且关闭时不遗留桥接任务。"""

    async def scenario() -> None:
        runtime = build_runtime(
            LoadedConfiguration(tmp_path / "config.yaml", {}),
            inspect_resources(tmp_path / "resources"),
        )
        client = _ContentClient()
        bridge = MCPToolBridge(client)
        runtime.mcp_clients = (client,)
        runtime.mcp_bridges = (bridge,)
        try:
            summaries = runtime.mcp_content_servers()
            assert len(summaries) == 1
            assert summaries[0]["name"] == client.server_name
            assert summaries[0]["source"].startswith("mcp:source-")
            assert client.server_name not in repr(summaries[0]["source"])

            listed = await runtime.mcp_list_resources(client.server_name)
            assert listed["status"] == "approval_required"
            approval_id = str(listed["approval_id"])
            assert any(
                item.approval_id == approval_id for item in runtime.pending_approvals_for_ui()
            )
            completed = await runtime.continue_approval(approval_id)
            assert completed["status"] == "completed"
            assert completed["resources"][0]["uri"] == "file:///one"

            read_pending = await runtime.mcp_read_resource(client.server_name, "file:///one")
            assert read_pending["status"] == "approval_required"
            read_completed = await runtime.approve_mcp_content(str(read_pending["approval_id"]))
            assert read_completed["content_policy"] == "quoted_external_data"
            assert read_completed["external_resource"]["contents"][0]["uri"] == "file:///one"

            content_spec = next(
                spec for spec in bridge.content.specs() if spec.identity.endswith("resources-list")
            )
            with pytest.raises(PermissionError, match="application-controlled"):
                await content_spec.handler(
                    {},
                    ToolCallContext("profile", "session", "turn", source="model"),
                )
        finally:
            await runtime.close()
        assert bridge.content.status()["status"] == "closed"

    asyncio.run(scenario())


def test_resource_content_bounds_base64_and_uri_validation() -> None:
    async def scenario() -> None:
        request = await _request_from(
            [{"contents": [{"uri": "file:///one", "text": "你好你好你好你好"}]}], []
        )
        with pytest.raises(MCPProtocolError, match="invalid content"):
            await read_mcp_resource(request, "file:///one", server="secret", limits=LIMITS)

        oversized = base64.b64encode(b"x" * (LIMITS.max_block_bytes + 1)).decode()
        request = await _request_from(
            [{"contents": [{"uri": "file:///one", "blob": oversized}]}], []
        )
        with pytest.raises(MCPProtocolError, match="invalid content"):
            await read_mcp_resource(request, "file:///one", server="secret", limits=LIMITS)

        request = await _request_from(
            [{"contents": [{"uri": "file:///one", "text": "ok", "blob": "b2s="}]}], []
        )
        with pytest.raises(MCPProtocolError, match="invalid content"):
            await read_mcp_resource(request, "file:///one", server="secret", limits=LIMITS)

    asyncio.run(scenario())


def test_resource_read_content_uri_must_match_requested_uri() -> None:
    async def scenario() -> None:
        request = await _request_from([{"contents": [{"uri": "file:///other", "text": "ok"}]}], [])
        with pytest.raises(MCPProtocolError, match="URI does not match"):
            await read_mcp_resource(request, "file:///one", server="secret", limits=LIMITS)

    asyncio.run(scenario())


def test_prompt_messages_are_bounded_and_never_system_role() -> None:
    async def scenario() -> None:
        request = await _request_from(
            [{"messages": [{"role": "system", "content": {"type": "text", "text": "x"}}]}],
            [],
        )
        with pytest.raises(MCPProtocolError, match="invalid message"):
            await get_mcp_prompt(request, "review", {"code": "ok"}, server="secret", limits=LIMITS)

        request = await _request_from(
            [{"messages": [{"role": "user", "content": {"type": "text", "text": "x"}}]}],
            [],
        )
        result = await get_mcp_prompt(
            request, "review", {"code": "ok"}, server="secret", limits=LIMITS
        )
        assert result.public()["trust"] == "untrusted_external_content"
        assert result.public()["messages"][0]["role"] == "user"

        request = await _request_from(
            [
                {
                    "messages": [
                        {"role": "assistant", "content": {"type": "text", "text": "123456789"}}
                    ]
                }
            ],
            [],
        )
        with pytest.raises(MCPProtocolError, match="invalid message"):
            await get_mcp_prompt(request, "review", {"code": "ok"}, server="secret", limits=LIMITS)

    asyncio.run(scenario())


def test_clients_share_content_contract_and_npm_uses_stdio() -> None:
    class StubStdio(StdioMCPClient):
        def __init__(self, values: list[Mapping[str, object]]) -> None:
            super().__init__(
                StdioMCPServerConfig(name="stub", command=(sys.executable, "-c", "pass"))
            )
            self.values = values
            self._notifications.set_capabilities(
                MCPServerCapabilities(resources=True, prompts=True)
            )

        async def start(self) -> None:
            return None

        async def request(self, method: str, params: Mapping[str, object] | None = None):
            del method, params
            return self.values.pop(0)

    async def scenario() -> None:
        client = StubStdio(
            [
                {"resources": [_resource()]},
                {"contents": [{"uri": "file:///one", "text": "ok"}]},
                {"prompts": [_prompt()]},
                {"messages": [{"role": "user", "content": {"type": "text", "text": "ok"}}]},
            ]
        )
        assert (await client.list_resources())[0].uri == "file:///one"
        assert (await client.read_resource("file:///one")).contents[0].kind == "text"
        assert (await client.list_prompts())[0].name == "review"
        assert (await client.get_prompt("review", {"code": "ok"})).messages[0].role == "user"

    asyncio.run(scenario())
    npm = NpmMCPServerConfig.from_mapping({"name": "npm", "package": "mcp-test"})
    assert npm.stdio.command[:3] == ("npx", "--yes", "mcp-test")


class _ContentClient:
    server_name = "private-content-server"
    connection_state = "connected"

    def __init__(self) -> None:
        self.capabilities = MCPServerCapabilities(
            resources=True,
            resources_list_changed=True,
            prompts=True,
            prompts_list_changed=True,
        )
        self.resources = (
            MCPResource.from_mapping(_resource(), server=self.server_name, limits=LIMITS),
        )
        self.prompts = (MCPPrompt.from_mapping(_prompt(), server=self.server_name, limits=LIMITS),)
        self.handlers = set()
        self.close_handlers = set()
        self.connection_handlers = set()
        self.resource_calls = 0
        self.prompt_calls = 0

    async def start(self) -> None:
        return None

    async def list_tools(self):
        return ()

    async def list_resources(self):
        self.resource_calls += 1
        return self.resources

    async def read_resource(self, uri: str):
        return MCPReadResourceResult(
            (MCPResourceContent.from_mapping({"uri": uri, "text": "ok"}, limits=LIMITS),),
            self.server_name,
            2,
        )

    async def list_prompts(self):
        self.prompt_calls += 1
        return self.prompts

    async def get_prompt(self, name: str, arguments=None):
        del name, arguments
        return MCPGetPromptResult(
            (MCPPromptMessage("user", MCPPromptContent({"type": "text", "text": "ok"}, 2)),),
            self.server_name,
        )

    def add_notification_handler(self, handler):
        self.handlers.add(handler)
        return lambda: self.handlers.discard(handler)

    def add_close_handler(self, handler):
        self.close_handlers.add(handler)
        return lambda: self.close_handlers.discard(handler)

    def add_connection_handler(self, handler):
        self.connection_handlers.add(handler)
        return lambda: self.connection_handlers.discard(handler)

    async def close(self) -> None:
        return None


def test_content_tools_are_private_read_only_and_use_approval_and_debounce() -> None:
    async def scenario() -> None:
        client = _ContentClient()
        bridge = MCPToolBridge(
            client,
            notification_debounce_seconds=0.01,
            notification_retry_delays=(0.01,),
        )
        registry = ToolRegistry()
        specs = await bridge.refresh_into(registry)
        assert len(specs) == 4
        assert all(not spec.public and spec.read_only for spec in specs)
        assert registry.visible() == ()

        read_spec = next(spec for spec in specs if spec.identity.endswith("resources-read"))
        executor = ToolExecutionService(registry, PermissionService())
        context = ToolCallContext("profile", "session", "turn", source="console")
        pending = await executor.execute(
            call_id="read-1",
            identity=read_spec.identity,
            arguments={"uri": "file:///one"},
            context=context,
        )
        assert pending.status == "approval_required"
        assert pending.approval is not None
        completed = await executor.approve_and_execute(
            pending.approval.approval_id, context=context
        )
        assert completed.status == "completed"
        assert completed.content["untrusted"] is True
        assert completed.content["content_policy"] == "quoted_external_data"
        assert completed.content["external_resource"]["contents"][0]["uri"] == "file:///one"
        assert "private-content-server" not in repr(bridge.status())

        prompt_spec = next(spec for spec in specs if spec.identity.endswith("prompts-get"))
        prompt_pending = await executor.execute(
            call_id="prompt-1",
            identity=prompt_spec.identity,
            arguments={"name": "review", "arguments": {"code": "ok"}},
            context=context,
        )
        assert prompt_pending.status == "approval_required"
        prompt_completed = await executor.approve_and_execute(
            prompt_pending.approval.approval_id, context=context
        )
        assert prompt_completed.status == "completed"
        prompt_payload = prompt_completed.content["external_prompt"]
        assert prompt_completed.content["content_policy"] == "quoted_external_data"
        assert prompt_payload["messages"][0]["speaker"] == "user"
        assert "role" not in prompt_payload["messages"][0]

        client.resources = (
            MCPResource.from_mapping(
                _resource("file:///two", "two"), server=client.server_name, limits=LIMITS
            ),
        )
        await asyncio.gather(
            *(bridge._on_notification("notifications/resources/list_changed") for _ in range(8))
        )
        await asyncio.sleep(0.04)
        assert client.resource_calls >= 2
        await bridge.close()
        assert bridge.status()["status"] == "closed"

    asyncio.run(scenario())
