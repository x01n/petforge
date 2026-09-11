from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from textwrap import dedent

import httpx
import pytest

from app.runtime import _build_mcp, validate_runtime_configuration
from config.loader import ConfigurationError, LoadedConfiguration
from core.adapters.mcp import (
    HTTPMCPClient,
    HTTPMCPServerConfig,
    MCPProtocolError,
    MCPTimeoutError,
    NpmMCPServerConfig,
    StdioMCPClient,
    StdioMCPServerConfig,
)
from core.adapters.mcp.http import _BoundedHTTPResponse, iter_sse_events


def _rpc_response(request: dict[str, object], result: object) -> httpx.Response:
    return httpx.Response(
        200,
        json={"jsonrpc": "2.0", "id": request.get("id"), "result": result},
    )


def test_http_mcp_discovers_and_calls_with_redacted_config() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        method = body["method"]
        if method == "initialize":
            return _rpc_response(body, {"protocolVersion": "2025-06-18"})
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "tools/list":
            return _rpc_response(
                body,
                {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]},
            )
        if method == "tools/call":
            return _rpc_response(body, {"content": [{"type": "text", "text": "ok"}]})
        return _rpc_response(body, {})

    async def run() -> None:
        config = HTTPMCPServerConfig(
            name="remote",
            url="https://mcp.example.invalid/v1",
            headers={"Authorization": "Bearer secret-value"},
        )
        assert "secret-value" not in repr(config)
        client = HTTPMCPClient(
            config,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == ["echo"]
        result = await client.call_tool("echo", {})
        assert result["content"]
        assert seen[0].headers["authorization"] == "Bearer secret-value"
        await client.close()

    asyncio.run(run())


def test_sse_mcp_uses_endpoint_event_and_session_header() -> None:
    seen: list[tuple[str, str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, str(request.url), request.headers.get("mcp-session-id")))
        if request.method == "GET":
            return httpx.Response(
                200,
                headers={
                    "content-type": "text/event-stream",
                    "mcp-session-id": "session-1",
                },
                content=b": keepalive\n\nevent: endpoint\ndata: /messages\n\n",
            )
        if request.method == "DELETE":
            return httpx.Response(204)
        body = json.loads(request.content)
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        result = (
            {"protocolVersion": "2025-06-18"} if body["method"] == "initialize" else {"tools": []}
        )
        return _rpc_response(body, result)

    async def run() -> None:
        client = HTTPMCPClient(
            HTTPMCPServerConfig(
                name="sse",
                url="https://mcp.example.invalid/sse",
                transport="sse",
                headers={"X-Trace": "enabled"},
            ),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        assert await client.list_tools() == ()
        assert seen[0][0] == "GET"
        assert seen[1][0] == "POST"
        assert seen[1][1] == "https://mcp.example.invalid/messages"
        assert seen[1][2] == "session-1"
        await client.close()
        assert seen[-1][0] == "DELETE"

    asyncio.run(run())


@pytest.mark.parametrize("close_session", [True, False])
@pytest.mark.parametrize("session_id", ["http-session", None])
@pytest.mark.parametrize("delete_status", [204, 405])
def test_http_close_releases_negotiated_session_once(
    close_session: bool, session_id: str | None, delete_status: int
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(405)
        if request.method == "DELETE":
            return httpx.Response(delete_status)
        body = json.loads(request.content)
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        response = _rpc_response(body, {"protocolVersion": "2025-06-18"})
        if session_id is not None:
            response.headers["Mcp-Session-Id"] = session_id
        return response

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = HTTPMCPClient(
                HTTPMCPServerConfig(
                    name="http-close",
                    url="https://mcp.example.invalid/mcp",
                    close_session=close_session,
                ),
                client=transport,
            )
            await client.start()
            await client.close()
            await client.close()
            deletes = [request for request in seen if request.method == "DELETE"]
            assert len(deletes) == int(close_session and session_id is not None)
            if deletes:
                assert str(deletes[0].url) == "https://mcp.example.invalid/mcp"
                assert deletes[0].headers["mcp-session-id"] == session_id
                assert deletes[0].headers["mcp-protocol-version"] == "2025-06-18"
            assert client.session_id is None
            assert client._event_task is None
            assert not transport.is_closed

    asyncio.run(run())


def test_http_concurrent_request_waits_for_initialize_notification() -> None:
    async def run() -> None:
        initializing = asyncio.Event()
        release_initialize = asyncio.Event()
        methods: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(405)
            body = json.loads(request.content)
            method = body["method"]
            methods.append(method)
            if method == "initialize":
                initializing.set()
                await release_initialize.wait()
                return _rpc_response(body, {"protocolVersion": "2025-06-18"})
            if method == "notifications/initialized":
                return httpx.Response(202)
            return _rpc_response(body, {"tools": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = HTTPMCPClient(
                HTTPMCPServerConfig(name="http-start-race", url="https://mcp.example.invalid/mcp"),
                client=transport,
            )
            start_task = asyncio.create_task(client.start())
            list_task = None
            try:
                await asyncio.wait_for(initializing.wait(), 1)
                list_task = asyncio.create_task(client.list_tools())
                await asyncio.sleep(0)
                release_initialize.set()
                await asyncio.wait_for(start_task, 1)
                assert await asyncio.wait_for(list_task, 1) == ()
                assert methods == ["initialize", "notifications/initialized", "tools/list"]
            finally:
                release_initialize.set()
                tasks = [task for task in (start_task, list_task) if task is not None]
                await asyncio.gather(*tasks, return_exceptions=True)
                await client.close()

    asyncio.run(run())


def test_legacy_sse_reconnect_releases_failed_stream_and_session() -> None:
    async def run() -> None:
        disconnect = asyncio.Event()
        session_headers: list[str | None] = []

        class _SSEStream(httpx.AsyncByteStream):
            def __init__(self, generation: int) -> None:
                self.generation = generation
                self.close_count = 0

            async def __aiter__(self):
                yield f"event: endpoint\ndata: /messages-{self.generation}\n\n".encode()
                if self.generation == 1:
                    await disconnect.wait()
                    raise httpx.ReadError("connection interrupted")
                await asyncio.Event().wait()

            async def aclose(self) -> None:
                self.close_count += 1

        streams: list[_SSEStream] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                session_headers.append(request.headers.get("mcp-session-id"))
                stream = _SSEStream(len(streams) + 1)
                streams.append(stream)
                return httpx.Response(
                    200,
                    headers={
                        "content-type": "text/event-stream",
                        "mcp-session-id": f"sse-session-{stream.generation}",
                    },
                    stream=stream,
                )
            if request.method == "DELETE":
                return httpx.Response(204)
            body = json.loads(request.content)
            if body["method"] == "notifications/initialized":
                return httpx.Response(202)
            result = (
                {"protocolVersion": "2025-06-18"}
                if body["method"] == "initialize"
                else {"tools": []}
            )
            return _rpc_response(body, result)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as transport:
            client = HTTPMCPClient(
                HTTPMCPServerConfig(
                    name="sse-failed-reconnect",
                    url="https://mcp.example.invalid/events",
                    transport="sse",
                    request_timeout_seconds=0.5,
                    startup_timeout_seconds=0.5,
                ),
                client=transport,
            )
            try:
                assert await client.list_tools() == ()
                previous_reader = client._sse_task
                assert previous_reader is not None
                disconnect.set()
                await asyncio.wait_for(previous_reader, 1)
                assert client.connection_state == "disconnected"
                assert await client.list_tools() == ()
                assert session_headers == [None, None]
                assert streams[0].close_count == 1
                assert client._message_url == "https://mcp.example.invalid/messages-2"
                assert client.session_id == "sse-session-2"
            finally:
                await client.close()
            assert [stream.close_count for stream in streams] == [1, 1]

    asyncio.run(run())


def test_legacy_sse_eof_invalidates_session_and_reopens_on_next_request() -> None:
    seen_gets = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_gets
        if request.method == "GET":
            seen_gets += 1
            endpoint = "/messages-one" if seen_gets == 1 else "/messages-two"
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=f"event: endpoint\ndata: {endpoint}\n\n".encode(),
            )
        body = json.loads(request.content)
        if body["method"] == "notifications/initialized":
            return httpx.Response(202)
        return _rpc_response(
            body,
            {"protocolVersion": "2025-06-18"} if body["method"] == "initialize" else {"tools": []},
        )

    async def run() -> None:
        client = HTTPMCPClient(
            HTTPMCPServerConfig(
                name="sse-eof-reconnect",
                url="https://mcp.example.invalid/events",
                transport="sse",
                request_timeout_seconds=0.5,
                startup_timeout_seconds=0.5,
            ),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            assert await client.list_tools() == ()
            assert await client.list_tools() == ()
            assert seen_gets == 2
            assert client._message_url == "https://mcp.example.invalid/messages-two"
        finally:
            await client.close()

    asyncio.run(run())


def test_mcp_session_header_is_bounded_and_invalid_values_do_not_replace_state() -> None:
    client = HTTPMCPClient(
        HTTPMCPServerConfig(name="session-bound", url="https://mcp.example.invalid")
    )

    class _Response:
        def __init__(self, value: str) -> None:
            self.headers = {"mcp-session-id": value}

    client._capture_session(_Response("session-1"))
    assert client.session_id == "session-1"
    client._capture_session(_Response("x" * 257))
    assert client.session_id == "session-1"
    client._capture_session(_Response("bad\r\nvalue"))
    assert client.session_id == "session-1"
    asyncio.run(client.close())


def test_http_mcp_rejects_malformed_json_rpc_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "1.0", "id": 1, "result": {}})

    async def run() -> None:
        client = HTTPMCPClient(
            {"name": "bad", "url": "https://mcp.example.invalid"},
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(MCPProtocolError, match="jsonrpc"):
            await client.request("tools/list", {})
        await client.close()

    asyncio.run(run())


def test_http_mcp_start_cancellation_resets_generation_and_allows_retry() -> None:
    class _Response:
        status_code = 200
        headers: dict[str, str] = {"content-type": "application/json"}

        def __init__(self, response_id: int, release: asyncio.Event | None) -> None:
            self.response_id = response_id
            self.release = release

        async def aiter_bytes(self):
            if self.release is not None:
                await self.release.wait()
            yield json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": self.response_id,
                    "result": {"protocolVersion": "2025-06-18"},
                }
            ).encode()

        async def aclose(self) -> None:
            return None

    class _Context:
        def __init__(
            self,
            entered: asyncio.Event,
            release: asyncio.Event,
            response_id: int,
            block: bool,
        ) -> None:
            self.entered = entered
            self.release = release
            self.exited = 0
            self.response_id = response_id
            self.block = block

        async def __aenter__(self) -> _Response:
            self.entered.set()
            return _Response(self.response_id, self.release if self.block else None)

        async def __aexit__(self, *_: object) -> None:
            self.exited += 1

    class _Client:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.contexts: list[_Context] = []
            self.response_id = 0

        def stream(self, method: str, _url: str, **_kwargs: object) -> _Context:
            assert method == "POST"
            payload = _kwargs.get("json", {})
            response_id = int(payload.get("id", 0)) if isinstance(payload, dict) else 0
            context = _Context(
                self.entered,
                self.release,
                response_id,
                not self.contexts,
            )
            self.contexts.append(context)
            return context

    async def run() -> None:
        transport = _Client()
        client = HTTPMCPClient(
            HTTPMCPServerConfig(
                name="cancel-retry",
                url="https://mcp.example.invalid",
                startup_timeout_seconds=0.5,
            ),
            client=transport,
        )
        first = asyncio.create_task(client.start())
        await transport.entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert client._started is False
        assert client._initialized is False
        assert client.session_id is None
        assert transport.contexts[0].exited == 1

        # 取消后允许新一代初始化；释放第二次请求并验证状态可用。
        transport.release.set()
        await client.start()
        assert client._started is True
        assert client._initialized is True
        await client.close()

    asyncio.run(run())


def test_sse_open_cancellation_closes_unclaimed_context() -> None:
    class _Context:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.exited = 0

        async def __aenter__(self):
            self.entered.set()
            await self.release.wait()
            return object()

        async def __aexit__(self, *_: object) -> None:
            self.exited += 1

    class _Client:
        def __init__(self) -> None:
            self.context = _Context()

        def stream(self, method: str, _url: str, **_kwargs: object) -> _Context:
            assert method == "GET"
            return self.context

    async def run() -> None:
        transport = _Client()
        client = HTTPMCPClient(
            HTTPMCPServerConfig(
                name="sse-cancel-open",
                url="https://mcp.example.invalid/events",
                transport="sse",
            ),
            client=transport,
        )
        opening = asyncio.create_task(client._open_sse())
        await transport.context.entered.wait()
        opening.cancel()
        with pytest.raises(asyncio.CancelledError):
            await opening
        assert transport.context.exited == 1
        assert client._sse_cm is None
        assert client._sse_response is None

    asyncio.run(run())


def test_http_mcp_discards_abandoned_late_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(202)

    async def run() -> None:
        client = HTTPMCPClient(
            {"name": "late", "url": "https://mcp.example.invalid"},
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        try:
            loop = asyncio.get_running_loop()
            client._pending[7] = loop.create_future()
            client._abandoned_ids.append(7)
            await client._deliver_response({"jsonrpc": "2.0", "id": 7, "result": {"late": True}})
            assert 7 not in client._pending
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("version", [None, "2024-11-05"])
def test_mcp_initialize_requires_supported_protocol_version(version: str | None) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if body["method"] == "initialize":
            result: dict[str, object] = {"capabilities": {}}
            if version is not None:
                result["protocolVersion"] = version
            return _rpc_response(body, result)
        return httpx.Response(202)

    async def run() -> None:
        client = HTTPMCPClient(
            HTTPMCPServerConfig(
                name="version-check",
                url="https://mcp.example.invalid",
                headers={"mcp-protocol-version": "attacker-value"},
            ),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(MCPProtocolError, match="protocol version"):
            await client.start()
        assert seen
        assert seen[0].headers.get("mcp-protocol-version") is None
        assert len(seen) == 1
        await client.close()

    asyncio.run(run())


def test_http_response_content_length_is_rejected_before_reading() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["method"] == "initialize":
            return _rpc_response(body, {"protocolVersion": "2025-06-18", "capabilities": {}})
        return httpx.Response(
            200,
            headers={"content-length": "2048"},
            content=b"x" * 2048,
        )

    async def run() -> None:
        client = HTTPMCPClient(
            HTTPMCPServerConfig(
                name="bounded",
                url="https://mcp.example.invalid",
                max_message_bytes=1024,
            ),
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(MCPProtocolError, match="message limit"):
            await client.request("tools/list", {})
        await client.close()

    asyncio.run(run())


def test_bounded_http_sse_allows_multiple_bounded_events() -> None:
    class _Response:
        status_code = 200
        headers: dict[str, str] = {}

        async def aiter_lines(self):
            for line in (
                'data: {"jsonrpc":"2.0","id":1,"result":{}}',
                "",
                'data: {"jsonrpc":"2.0","id":2,"result":{}}',
                "",
            ):
                yield line

        async def aclose(self) -> None:
            return None

    async def run() -> None:
        response = _BoundedHTTPResponse(_Response(), max_bytes=48)
        events = [event async for event in iter_sse_events(response, max_message_bytes=48)]
        assert len(events) == 2
        await response.aclose()

    asyncio.run(run())


def test_stdio_initialize_requires_supported_protocol_version() -> None:
    script = dedent(
        """
        import json, sys
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("method") == "initialize":
                print(json.dumps({
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {"protocolVersion": "1999", "capabilities": {}},
                }), flush=True)
        """
    )

    async def run() -> None:
        client = StdioMCPClient(
            StdioMCPServerConfig(
                name="stdio-version-check",
                command=(sys.executable, "-u", "-c", script),
            )
        )
        with pytest.raises(MCPProtocolError, match="protocol version"):
            await client.start()
        assert client.process is None
        await client.close()

    asyncio.run(run())


def test_stdio_initialize_timeout_releases_generation_and_allows_retry(
    tmp_path: Path, monkeypatch
) -> None:
    marker = tmp_path / "initialize-attempts"
    script = dedent(
        """
        import json, pathlib, sys, time
        marker = pathlib.Path(sys.argv[1])
        attempts = int(marker.read_text()) if marker.exists() else 0
        marker.write_text(str(attempts + 1))
        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            if method == "initialize" and attempts == 0:
                time.sleep(0.25)
            if method == "notifications/initialized":
                continue
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
            print(
                json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result}),
                flush=True,
            )
        """
    )

    async def run() -> None:
        client = StdioMCPClient(
            StdioMCPServerConfig(
                name="stdio-retry-after-timeout",
                command=(sys.executable, "-u", "-c", script, str(marker)),
                startup_timeout_seconds=0.1,
                request_timeout_seconds=1.0,
            )
        )
        initialize = client._initialize
        generation = 0

        async def initialize_when_worker_ready(*, timeout_seconds: float | None = None) -> None:
            nonlocal generation
            generation += 1

            async def wait_for_worker() -> None:
                while not marker.exists() or marker.read_text() != str(generation):
                    await asyncio.sleep(0.005)

            # Keep the 100 ms deadline focused on the deliberately delayed handshake.
            await asyncio.wait_for(wait_for_worker(), 5)
            await initialize(timeout_seconds=timeout_seconds)

        monkeypatch.setattr(client, "_initialize", initialize_when_worker_ready)
        try:
            with pytest.raises(MCPTimeoutError, match="timed out"):
                await client.start()
            assert client.process is None
            assert client._closed is False
            await client.start()
            assert client._initialized is True
            assert marker.read_text() == "2"
        finally:
            await client.close()

    asyncio.run(run())


def test_stdio_and_npm_config_repr_omit_environment_values() -> None:
    stdio = StdioMCPServerConfig(
        name="stdio-secret",
        command=("mcp",),
        env={"TOKEN": "secret-value"},
    )
    npm = NpmMCPServerConfig.from_mapping(
        {
            "name": "npm-secret",
            "package": "mcp-tools",
            "env": {"TOKEN": "secret-value"},
        }
    )
    assert "secret-value" not in repr(stdio)
    assert "secret-value" not in repr(npm)
    assert "TOKEN" in repr(stdio)


@pytest.mark.parametrize(
    "mapping",
    [
        {"name": "x", "url": "ftp://example.invalid"},
        {"name": "x", "url": "https://user:pass@example.invalid"},
        {"name": "x", "url": "https://example.invalid", "headers": {"X": "a\nsecret"}},
        {"name": "x", "url": "https://example.invalid", "request_timeout_seconds": 0.01},
        {"name": "x", "url": "https://example.invalid", "transport": "websocket"},
    ],
)
def test_http_mcp_config_is_strict(mapping: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HTTPMCPServerConfig.from_mapping(mapping)


def test_npm_transport_is_a_non_shell_stdio_alias() -> None:
    config = NpmMCPServerConfig.from_mapping(
        {
            "name": "npm-tools",
            "package": "@scope/mcp-tools@1.2.3",
            "args": ["/tmp/work"],
        }
    )
    assert config.command == ("npx", "--yes", "@scope/mcp-tools@1.2.3", "/tmp/work")
    with pytest.raises(ValueError):
        NpmMCPServerConfig.from_mapping(
            {"name": "bad", "package": "@scope/mcp-tools", "args": "--unsafe"}
        )


def test_runtime_builds_http_sse_and_npm_clients(tmp_path: Path) -> None:
    values = {
        "mcp": {
            "enabled": True,
            "servers": [
                {"name": "http", "transport": "http", "url": "https://mcp.example.invalid"},
                {"name": "sse", "transport": "sse", "url": "https://mcp.example.invalid/sse"},
                {"name": "npm", "transport": "npm", "package": "mcp-tools"},
            ],
        }
    }
    clients, bridges = _build_mcp(values, configuration_directory=tmp_path)
    try:
        assert [type(client).__name__ for client in clients] == [
            "HttpMCPClient",
            "HttpMCPClient",
            "StdioMCPClient",
        ]
        assert clients[0].config.transport == "http"
        assert clients[1].config.transport == "sse"
        assert clients[2].config.command[:3] == ("npx", "--yes", "mcp-tools")
        assert len(bridges) == 3
    finally:
        asyncio.run(_close_all(clients))


async def _close_all(clients: tuple[object, ...]) -> None:
    for client in clients:
        close = getattr(client, "close")
        await close()


def test_runtime_validation_accepts_network_mcp(tmp_path: Path) -> None:
    config = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "mcp": {
                "enabled": True,
                "servers": [
                    {
                        "name": "remote",
                        "transport": "http",
                        "url": "https://mcp.example.invalid/rpc",
                        "headers": {"Authorization": "Bearer secret"},
                    }
                ],
            }
        },
    )
    # validate_runtime_configuration also checks unrelated defaults, so this
    # assertion only needs to prove the MCP transport is not rejected.
    try:
        validate_runtime_configuration(config)
    except ConfigurationError as exc:
        assert "mcp.servers[0]" not in str(exc)
