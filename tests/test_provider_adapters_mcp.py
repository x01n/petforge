from __future__ import annotations

import asyncio
import sys
import textwrap
from collections.abc import AsyncIterator

import httpx
import pytest

from core.adapters.direct.base import ProviderResponse
from core.adapters.direct.errors import ProviderAdapterError
from core.adapters.direct.openai_chat_sse import OpenAIChatSSEAdapter
from core.adapters.mcp import MCPProtocolError, MCPTool, StdioMCPClient, StdioMCPServerConfig
from core.adapters.providers.claude import ClaudeProviderAdapter
from core.adapters.providers.converters import build_tool_name_map
from core.adapters.providers.gemini import GeminiProviderAdapter
from core.adapters.providers.http import iter_sse_json
from core.adapters.providers.openai import OpenAIProviderAdapter, OpenAIResponsesAdapter
from core.adapters.providers.registry import create_provider_adapter, provider_registry
from core.contracts.chat import ChatMessage, ChatRequest, ToolCall, ToolDefinition
from core.events.types import ConversationContext
from services.model_routing import SUPPORTED_PROTOCOLS, ChannelConfig
from services.tools import MCPToolBridge, ToolKind, ToolRegistry, ToolSpec


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def stream(self, *args: object, **kwargs: object) -> _Response:
        self.calls.append((args, kwargs))
        return self.response


def _request() -> ChatRequest:
    return ChatRequest(
        model="demo",
        messages=(ChatMessage("user", "hello"),),
        tools=(
            ToolDefinition(
                "pet:move", "move", {"type": "object", "properties": {"x": {"type": "number"}}}
            ),
        ),
    )


async def _collect(iterator: AsyncIterator):
    return [event async for event in iterator]


def test_provider_tool_name_mapping_is_safe_and_reversible() -> None:
    names = build_tool_name_map(_request().tools)
    provider_name = names.provider("pet:move")
    assert ":" not in provider_name
    assert len(provider_name) <= 64
    assert names.core(provider_name) == "pet:move"


def test_direct_openai_client_does_not_inherit_process_proxy() -> None:
    """直连兼容适配器与提供方适配器保持相同的回环代理隔离。"""

    adapter = OpenAIChatSSEAdapter(
        ChannelConfig(
            "direct-proxy-isolated",
            protocol="openai_chat",
            base_url="http://127.0.0.1:13000",
            model="demo",
        )
    )
    try:
        assert getattr(adapter._client, "_trust_env", True) is False
    finally:
        asyncio.run(adapter.aclose())


def test_direct_openai_stream_client_disables_default_read_timeout() -> None:
    import httpx

    adapter = OpenAIChatSSEAdapter(
        ChannelConfig(
            "direct-stream-timeout",
            protocol="openai_chat",
            base_url="https://example.invalid/v1",
            model="demo",
        )
    )
    try:
        assert isinstance(adapter._client, httpx.AsyncClient)
        assert adapter._client.timeout.read is None
    finally:
        asyncio.run(adapter.aclose())


def test_direct_openai_chat_can_omit_sampling_temperature() -> None:
    adapter = OpenAIChatSSEAdapter(
        ChannelConfig(
            "direct-no-temperature",
            protocol="openai_chat",
            base_url="https://example.invalid/v1",
            model="demo",
            metadata={"omit_temperature": True},
        ),
        client=object(),
    )
    payload = adapter.build_payload(ChatRequest("demo", (ChatMessage("user", "hello"),)))
    assert "temperature" not in payload


def test_openai_provider_builds_safe_tools_and_restores_tool_identity() -> None:
    response = _Response(
        [
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1",'
                '"function":{"name":"pet_x3a_move__PLACEHOLDER","arguments":"{\\"x\\":1}"}}]}}]} '
            ).rstrip(),
            "",
            'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
            "",
            "data: [DONE]",
            "",
        ]
    )
    client = _Client(response)
    channel = ChannelConfig(
        "openai", protocol="openai_chat", base_url="https://example.invalid/v1", model="demo"
    )
    adapter = OpenAIProviderAdapter(channel, client=client)
    # 使用转换器的精确名称，避免测试依赖供应商名称的手工猜测。
    safe_name = build_tool_name_map(_request().tools).provider("pet:move")
    response.lines[0] = response.lines[0].replace("pet_x3a_move__PLACEHOLDER", safe_name)
    events = asyncio.run(_collect(adapter.stream(_request())))
    tool_event = next(event for event in events if hasattr(event, "identity"))
    assert tool_event.identity == "pet:move"
    body = client.calls[0][1]["json"]
    assert body["tools"][0]["function"]["name"] != "pet:move"


def test_openai_chat_can_omit_sampling_temperature_explicitly() -> None:
    """推理型 Chat 网关拒绝 temperature 时，显式控制项不得进入请求体。"""

    request = ChatRequest(
        "demo",
        (ChatMessage("user", "hello"),),
        metadata={"omit_temperature": True},
    )
    adapter = OpenAIProviderAdapter(
        ChannelConfig(
            "openai-no-temperature",
            protocol="openai_chat",
            base_url="https://example.invalid/v1",
            model="demo",
        ),
        client=object(),
    )
    payload, _names = adapter.build_payload(request)
    assert "temperature" not in payload


def test_openai_chat_channel_metadata_can_omit_sampling_temperature() -> None:
    channel = ChannelConfig(
        "openai-no-temperature-channel",
        protocol="openai_chat",
        base_url="https://example.invalid/v1",
        model="demo",
        metadata={"omit_temperature": True},
    )
    adapter = OpenAIProviderAdapter(channel, client=object())
    payload, _names = adapter.build_payload(ChatRequest("demo", (ChatMessage("user", "hello"),)))
    assert "temperature" not in payload


def test_openai_responses_can_omit_sampling_temperature_explicitly() -> None:
    """Responses 端点也必须遵守显式的推理模型采样禁用配置。"""

    request = ChatRequest(
        "demo",
        (ChatMessage("user", "hello"),),
        metadata={"omit_temperature": True},
    )
    adapter = OpenAIResponsesAdapter(
        ChannelConfig(
            "responses-no-temperature",
            protocol="openai_responses",
            base_url="https://example.invalid/v1",
            model="demo",
            request_body_template={"temperature": 0.7},
        ),
        client=object(),
    )

    payload, _names = adapter.build_payload(request)

    assert "temperature" not in payload


def test_openai_responses_channel_metadata_can_omit_sampling_temperature() -> None:
    adapter = OpenAIResponsesAdapter(
        ChannelConfig(
            "responses-no-temperature-channel",
            protocol="openai_responses",
            base_url="https://example.invalid/v1",
            model="demo",
            metadata={"omit_temperature": True},
            request_body_template={"temperature": 0.7},
        ),
        client=object(),
    )

    payload, _names = adapter.build_payload(ChatRequest("demo", (ChatMessage("user", "hello"),)))

    assert "temperature" not in payload


def test_openai_responses_preserves_extra_body_without_overriding_protocol_fields() -> None:
    """Responses 兼容网关的扩展字段不能篡改标准流式请求边界。"""

    request = ChatRequest(
        "demo",
        (ChatMessage("user", "hello"),),
        metadata={
            "extra_body": {
                "reasoning": {"effort": "low"},
                "model": "overridden-model",
                "input": "overridden-input",
                "stream": False,
            }
        },
    )
    adapter = OpenAIResponsesAdapter(
        ChannelConfig(
            "responses-extra-body",
            protocol="openai_responses",
            base_url="https://example.invalid/v1",
            model="demo",
        ),
        client=object(),
    )

    payload, _names = adapter.build_payload(request)

    assert payload["reasoning"] == {"effort": "low"}
    assert payload["model"] == "demo"
    assert payload["stream"] is True
    assert isinstance(payload["input"], list)


def test_openai_responses_omit_temperature_overrides_extra_body() -> None:
    request = ChatRequest(
        "demo",
        (ChatMessage("user", "hello"),),
        metadata={"omit_temperature": True, "extra_body": {"temperature": 0.2}},
    )
    adapter = OpenAIResponsesAdapter(
        ChannelConfig(
            "responses-extra-temperature",
            protocol="openai_responses",
            base_url="https://example.invalid/v1",
            model="demo",
            request_body_template={"temperature": 0.7},
        ),
        client=object(),
    )

    payload, _names = adapter.build_payload(request)

    assert "temperature" not in payload


@pytest.mark.parametrize(
    ("error_code", "category", "retryable"),
    (
        ("invalid_api_key", "authentication", False),
        ("authentication_error", "authentication", False),
        ("permission_denied", "authorization", False),
        ("rate_limit_error", "rate_limit", True),
    ),
)
def test_openai_chat_sse_error_codes_are_classified_for_routing(
    error_code: str, category: str, retryable: bool
) -> None:
    client = _Client(_Response([f'data: {{"error": {{"code": "{error_code}"}}}}', ""]))
    adapter = OpenAIProviderAdapter(
        ChannelConfig(
            "openai-errors",
            protocol="openai_chat",
            base_url="https://example.invalid/v1",
            model="demo",
        ),
        client=client,
    )

    async def run() -> None:
        with pytest.raises(ProviderAdapterError) as raised:
            _ = [event async for event in adapter.stream(_request())]
        assert raised.value.category == category
        assert raised.value.retryable is retryable
        if category == "authentication":
            assert "API key" in raised.value.safe_message

    asyncio.run(run())


@pytest.mark.parametrize(
    ("provider", "error_type", "category", "retryable"),
    (
        ("claude", "authentication_error", "authentication", False),
        ("claude", "permission_error", "authorization", False),
        ("claude", "rate_limit_error", "rate_limit", True),
        ("gemini", "UNAUTHENTICATED", "authentication", False),
        ("gemini", "MISSING_KEY", "authentication", False),
        ("gemini", "PERMISSION_DENIED", "authorization", False),
        ("gemini", "RESOURCE_EXHAUSTED", "rate_limit", True),
    ),
)
def test_non_openai_sse_error_codes_are_classified_for_routing(
    provider: str, error_type: str, category: str, retryable: bool
) -> None:
    if provider == "claude":
        response = _Response(
            [
                "event: error",
                f'data: {{"type":"error","error":{{"type":"{error_type}"}}}}',
                "",
            ]
        )
        adapter = ClaudeProviderAdapter(
            ChannelConfig(
                "claude-errors",
                protocol="anthropic_messages",
                base_url="https://example.invalid/v1",
                model="demo",
            ),
            client=_Client(response),
        )
    else:
        response = _Response([f'data: {{"error":{{"status":"{error_type}"}}}}', ""])
        adapter = GeminiProviderAdapter(
            ChannelConfig(
                "gemini-errors",
                protocol="gemini_generate",
                base_url="https://example.invalid/v1beta",
                model="demo",
            ),
            client=_Client(response),
        )

    async def run() -> None:
        with pytest.raises(ProviderAdapterError) as raised:
            _ = [event async for event in adapter.stream(_request())]
        assert raised.value.category == category
        assert raised.value.retryable is retryable
        assert "error_type" not in raised.value.safe_message

    asyncio.run(run())


def test_gemini_and_claude_paths_are_base_path_aware() -> None:
    gemini = GeminiProviderAdapter(
        ChannelConfig(
            "g",
            protocol="gemini_generate",
            base_url="https://example.invalid/v1beta",
            model="gemini-demo",
        ),
        client=object(),
    )
    claude = ClaudeProviderAdapter(
        ChannelConfig(
            "c",
            protocol="anthropic_messages",
            base_url="https://example.invalid/v1",
            model="claude-demo",
        ),
        client=object(),
    )
    assert gemini._url("gemini-demo").startswith("https://example.invalid/v1beta/models/")
    assert "/v1beta/v1beta/" not in gemini._url("gemini-demo")
    assert claude._url() == "https://example.invalid/v1/messages"


def test_gemini_key_uses_header_by_default() -> None:
    channel = ChannelConfig(
        "g-key",
        protocol="gemini_generate",
        base_url="https://example.invalid/v1beta",
        model="gemini-demo",
        api_key="secret-value",
    )
    adapter = GeminiProviderAdapter(channel, client=object())
    assert "secret-value" not in adapter._url("gemini-demo")
    assert adapter._headers()["x-goog-api-key"] == "secret-value"


def test_claude_key_supports_native_and_auth_token_headers() -> None:
    channel = ChannelConfig(
        "claude-key",
        protocol="anthropic_messages",
        base_url="https://example.invalid/v1",
        model="claude-demo",
        api_key="secret-value",
    )
    adapter = ClaudeProviderAdapter(channel, client=object())
    headers = adapter._headers()
    assert headers["x-api-key"] == "secret-value"
    assert headers["Authorization"] == "Bearer secret-value"
    assert headers["anthropic-version"] == "2023-06-01"


def test_provider_streaming_clients_disable_httpx_default_read_timeout() -> None:
    """默认流式客户端不应在五秒无增量时截断模型输出。"""

    import httpx

    claude = ClaudeProviderAdapter(
        ChannelConfig(
            "claude-timeout",
            protocol="anthropic_messages",
            base_url="https://example.invalid",
            model="demo",
        )
    )
    openai = OpenAIProviderAdapter(
        ChannelConfig(
            "openai-timeout",
            protocol="openai_chat",
            base_url="https://example.invalid/v1",
            model="demo",
        )
    )
    try:
        assert isinstance(claude._client, httpx.AsyncClient)
        assert isinstance(openai._client, httpx.AsyncClient)
        assert claude._client.timeout.read is None
        assert openai._client.timeout.read is None
    finally:
        asyncio.run(claude.aclose())
        asyncio.run(openai.aclose())


def test_claude_messages_sse_emits_text_reasoning_and_usage() -> None:
    response = _Response(
        [
            "event: message_start",
            'data: {"type":"message_start","message":{"usage":{"input_tokens":3}}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":'
            '{"type":"thinking_delta","thinking":"思考"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":'
            '{"type":"text_delta","text":"你好。"}}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
            '"usage":{"output_tokens":2}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]
    )
    adapter = ClaudeProviderAdapter(
        ChannelConfig(
            "claude-stream",
            protocol="anthropic_messages",
            base_url="https://example.invalid",
            model="claude-demo",
        ),
        client=_Client(response),
    )
    events = asyncio.run(
        _collect(adapter.stream(ChatRequest("claude-demo", (ChatMessage("user", "x"),))))
    )
    assert [type(event).__name__ for event in events] == [
        "ReasoningDelta",
        "TextDelta",
        "TurnFinished",
    ]
    assert events[1].delta == "你好。"
    assert events[-1].finish_reason == "end_turn"
    assert events[-1].usage["prompt_tokens"] == 3
    assert events[-1].usage["completion_tokens"] == 2


def test_claude_messages_tool_use_stream_restores_identity_and_arguments() -> None:
    request = _request()
    safe_name = build_tool_name_map(request.tools).provider("pet:move")
    response = _Response(
        [
            "event: content_block_start",
            (
                'data: {"type":"content_block_start","index":0,"content_block":'
                '{"type":"tool_use","id":"call-1","name":"' + safe_name + '","input":{}}}'
            ),
            "",
            "event: content_block_delta",
            (
                'data: {"type":"content_block_delta","index":0,"delta":'
                '{"type":"input_json_delta","partial_json":"{\\"x\\":1}"}}'
            ),
            "",
        ]
    )
    adapter = ClaudeProviderAdapter(
        ChannelConfig(
            "claude-tool-stream",
            protocol="anthropic_messages",
            base_url="https://example.invalid",
            model="claude-demo",
        ),
        client=_Client(response),
    )
    events = asyncio.run(_collect(adapter.stream(request)))
    tool_events = [event for event in events if type(event).__name__ == "ToolCallDelta"]
    assert [event.identity for event in tool_events] == ["pet:move", "pet:move"]
    assert tool_events[0].call_id == "call-1"
    assert tool_events[1].arguments_delta == '{"x":1}'


def test_full_url_endpoints_are_accepted_and_idempotent() -> None:
    channel = ChannelConfig(
        "full",
        protocol="openai_chat",
        base_url="https://example.invalid/v1/chat/completions",
        endpoint="https://proxy.invalid/complete",
        model="demo",
    )
    adapter = OpenAIProviderAdapter(channel, client=object())
    assert adapter._url() == "https://proxy.invalid/complete"
    same = OpenAIProviderAdapter(
        ChannelConfig(
            "same",
            protocol="openai_chat",
            base_url="https://example.invalid/v1/chat/completions",
            model="demo",
        ),
        client=object(),
    )
    assert same._url() == "https://example.invalid/v1/chat/completions"


def test_openai_endpoint_can_add_version_path_for_proxy_base() -> None:
    """代理根地址不含 ``/v1`` 时，可显式配置 OpenAI 版本路径。"""

    adapter = OpenAIProviderAdapter(
        ChannelConfig(
            "proxy-openai",
            protocol="openai_chat",
            base_url="http://127.0.0.1:13000",
            endpoint="/v1/chat/completions",
            model="demo",
        ),
        client=object(),
    )
    assert adapter._url() == "http://127.0.0.1:13000/v1/chat/completions"


def test_gemini_stream_endpoint_expands_model_and_preserves_call_id() -> None:
    channel = ChannelConfig(
        "g-template",
        protocol="gemini_generate",
        base_url="https://example.invalid/v1beta",
        endpoint="/chat/completions",
        endpoints={"stream": "/v1beta/models/{model}:streamGenerateContent?alt=sse"},
        model="gemini-demo",
    )
    adapter = GeminiProviderAdapter(channel, client=object())
    assert "{model}" not in adapter._url("gemini-demo")
    names = build_tool_name_map(_request().tools)
    context = ConversationContext("p", "s", "t", 0)
    value = {
        "c" + "andidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "id": "gem-call",
                                "name": names.provider("pet:move"),
                                "args": {"x": 1},
                            }
                        }
                    ]
                }
            }
        ]
    }
    event = adapter._event_parts(value, context, names, {})[0]
    assert event.call_id == "gem-call"
    assert event.identity == "pet:move"


def test_responses_payload_and_function_call_events_round_trip() -> None:
    tool = ToolDefinition("pet:move", "move", {"type": "object", "properties": {}})
    request = ChatRequest(
        "demo",
        (
            ChatMessage("user", "go"),
            ChatMessage("assistant", None, tool_calls=(ToolCall("call-1", "pet:move", {"x": 1}),)),
            ChatMessage("tool", '{"ok":1}', tool_call_id="call-1", tool_name="pet:move"),
        ),
        tools=(tool,),
    )
    adapter = OpenAIResponsesAdapter(
        ChannelConfig(
            "responses",
            protocol="openai_responses",
            base_url="https://example.invalid/v1",
            model="demo",
        ),
        client=object(),
    )
    payload, names = adapter.build_payload(request)
    assert payload["input"][1]["type"] == "function_call"
    assert payload["input"][2]["type"] == "function_call_output"
    assert payload["tool_choice"] == "auto"
    context = ConversationContext("p", "s", "t", 0)
    state: dict[str, object] = {}
    events = []
    OpenAIResponsesAdapter._events_from(
        {
            "type": "response.created",
            "response": {"model": "served-responses"},
        },
        context,
        names,
        state,
    )
    assert state["response_model"] == "served-responses"
    events.extend(
        OpenAIResponsesAdapter._events_from(
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "item-1",
                    "call_id": "call-1",
                    "name": names.provider("pet:move"),
                },
            },
            context,
            names,
            state,
        )
    )
    events.extend(
        OpenAIResponsesAdapter._events_from(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "item-1",
                "output_index": 0,
                "delta": '{"x":1}',
            },
            context,
            names,
            state,
        )
    )
    events.extend(
        OpenAIResponsesAdapter._events_from(
            {
                "type": "response.function_call_arguments.done",
                "item_id": "item-1",
                "output_index": 0,
                "call_id": "call-1",
                "name": names.provider("pet:move"),
                "arguments": '{"x":1}',
            },
            context,
            names,
            state,
        )
    )
    response = ProviderResponse()
    for event in events:
        response = response.add(event)
    call = response.finalized().tool_calls[0]
    assert call.call_id == "call-1"
    assert call.identity == "pet:move"
    assert call.arguments == {"x": 1}


def test_stream_parser_ignores_non_data_sse_fields() -> None:
    class Response:
        async def aiter_lines(self):
            for line in ("id: 1", "retry: 1000", 'data: {"ok":true}', ""):
                yield line

    async def collect():
        return [item async for item in iter_sse_json(Response())]

    assert asyncio.run(collect()) == [("message", {"ok": True})]


def test_stream_transport_error_is_retryable() -> None:
    class FailingResponse:
        async def __aenter__(self):
            raise httpx.ReadTimeout("timeout")

        async def __aexit__(self, *_: object) -> None:
            return None

    class FailingClient:
        def stream(self, *_args: object, **_kwargs: object):
            return FailingResponse()

    adapter = OpenAIProviderAdapter(
        ChannelConfig("timeout", base_url="https://example.invalid", model="demo"),
        client=FailingClient(),
    )

    async def run():
        with pytest.raises(Exception) as caught:
            async for _ in adapter.stream(ChatRequest("demo", (ChatMessage("user", "x"),))):
                pass
        return caught.value

    error = asyncio.run(run())
    assert getattr(error, "category", "") == "timeout"
    assert getattr(error, "retryable", False) is True


def test_stream_read_error_is_classified_as_network() -> None:
    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def aiter_lines(self):
            raise httpx.ReadError("connection lost")
            yield ""

    class Client:
        def stream(self, *_args: object, **_kwargs: object):
            return Response()

    adapter = OpenAIProviderAdapter(
        ChannelConfig("network", base_url="https://example.invalid", model="demo"),
        client=Client(),
    )

    async def run():
        with pytest.raises(Exception) as caught:
            async for _ in adapter.stream(ChatRequest("demo", (ChatMessage("user", "x"),))):
                pass
        return caught.value

    error = asyncio.run(run())
    assert getattr(error, "category", "") == "network"
    assert getattr(error, "retryable", False) is True


def test_provider_registry_creates_all_requested_protocols() -> None:
    for protocol, expected in (
        ("openai_chat", OpenAIProviderAdapter),
        ("gemini_generate", GeminiProviderAdapter),
        ("anthropic_messages", ClaudeProviderAdapter),
    ):
        channel = ChannelConfig(
            "channel-" + protocol,
            protocol=protocol,
            base_url="https://example.invalid",
            model="demo",
        )
        adapter = create_provider_adapter(channel)
        assert isinstance(adapter, expected)


def test_model_channel_protocol_allowlist_matches_provider_registry() -> None:
    """配置校验允许的协议必须与实际可构造的适配器集合保持同步。"""

    assert SUPPORTED_PROTOCOLS == frozenset(provider_registry.protocols())


def test_stdio_mcp_discovers_calls_and_bridges_tools() -> None:
    script = textwrap.dedent(
        """
        import json, sys
        for line in sys.stdin:
            req = json.loads(line)
            method = req.get('method')
            if method == 'notifications/initialized':
                continue
            if method == 'initialize':
                result = {
                    'protocolVersion': '2025-06-18',
                    'capabilities': {},
                    'serverInfo': {'name': 'fake', 'version': '1'},
                }
            elif method == 'tools/list':
                result = {
                    'tools': [{
                        'name': 'echo_tool',
                        'description': 'echo',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {'value': {'type': 'string'}},
                        },
                    }],
                }
            elif method == 'tools/call':
                value = req['params']['arguments']['value']
                result = {
                    'content': [{'type': 'text', 'text': value}],
                    'structuredContent': {'echo': value},
                }
            else:
                result = {}
            print(json.dumps({'jsonrpc': '2.0', 'id': req.get('id'), 'result': result}), flush=True)
        """
    )

    async def run() -> None:
        config = StdioMCPServerConfig(
            name="fake",
            command=(sys.executable, "-u", "-c", script),
            request_timeout_seconds=2,
        )
        client = StdioMCPClient(config)
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == ["echo_tool"]
        result = await client.call_tool("echo_tool", {"value": "ok"})
        assert result["structuredContent"] == {"echo": "ok"}
        bridge = MCPToolBridge(client)
        await bridge.discover()
        registry = ToolRegistry()
        specs = bridge.register_into(registry)
        assert len(specs) == 1
        assert specs[0].identity.startswith("mcp:fake.")
        assert specs[0].kind is ToolKind.MCP
        await client.close()

    asyncio.run(run())


def test_stdio_concurrent_requests_wait_for_initialization(monkeypatch) -> None:
    script = textwrap.dedent(
        """
        import json, sys
        initialized = False
        for line in sys.stdin:
            request = json.loads(line)
            method = request['method']
            if method == 'notifications/initialized':
                initialized = True
                continue
            if method == 'initialize':
                response = {'result': {'protocolVersion': '2025-06-18'}}
            elif not initialized:
                response = {'error': {'code': -32002, 'message': 'request before initialized'}}
            else:
                response = {'result': {'tools': [{'name': 'ready_tool'}]}}
            print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], **response}), flush=True)
        """
    )

    async def run() -> None:
        client = StdioMCPClient(
            StdioMCPServerConfig(
                name="concurrent-initialize",
                command=(sys.executable, "-u", "-c", script),
                request_timeout_seconds=2,
            )
        )
        initializing = asyncio.Event()
        release = asyncio.Event()
        initialize = client._initialize

        async def gated_initialize(*, timeout_seconds: float | None = None) -> None:
            initializing.set()
            await release.wait()
            await initialize(timeout_seconds=timeout_seconds)

        monkeypatch.setattr(client, "_initialize", gated_initialize)
        requests = [asyncio.create_task(client.list_tools())]
        try:
            await asyncio.wait_for(initializing.wait(), timeout=2)
            requests.append(asyncio.create_task(client.list_tools()))
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*requests), timeout=3)
            assert [[tool.name for tool in result] for result in results] == [
                ["ready_tool"],
                ["ready_tool"],
            ]
        finally:
            release.set()
            for request in requests:
                request.cancel()
            await asyncio.gather(*requests, return_exceptions=True)
            await client.close()

    asyncio.run(run())


def test_mcp_bridge_refresh_replaces_and_removes_stale_registry_tools() -> None:
    """远端 tools/list 变化后，registry 只保留当前桥接集合且刷新幂等。"""

    class Client:
        server_name = "mutable"

        def __init__(self) -> None:
            self.tools = (MCPTool(name="old_tool", server="mutable"),)

        async def list_tools(self):
            return self.tools

        async def call_tool(self, _name, _arguments):
            return {"content": []}

        async def close(self):
            return None

    async def run() -> None:
        client = Client()
        bridge = MCPToolBridge(client)
        registry = ToolRegistry()

        await bridge.refresh_into(registry)
        old_identity = "mcp:mutable.old_tool"
        assert registry.identities() == (old_identity,)

        client.tools = (MCPTool(name="new_tool", server="mutable"),)
        await bridge.refresh_into(registry)
        assert registry.identities() == ("mcp:mutable.new_tool",)

        # 同一发现结果可重复同步，不应被 registry 的 duplicate 检查拒绝。
        await bridge.refresh_into(registry)
        assert registry.identities() == ("mcp:mutable.new_tool",)

        # 新集合含有外部占用的身份时，原集合不能只同步一半。
        external = ToolSpec(
            "mcp:mutable.blocked",
            "外部工具",
            {"type": "object"},
            lambda _arguments, _context: {"ok": True},
        )
        registry.register(external)
        client.tools = (
            MCPTool(name="next_tool", server="mutable"),
            MCPTool(name="blocked", server="mutable"),
        )
        with pytest.raises(ValueError, match="duplicate tool identity"):
            await bridge.refresh_into(registry)
        assert registry.identities() == ("mcp:mutable.blocked", "mcp:mutable.new_tool")

    asyncio.run(run())


def test_stdio_mcp_rejects_non_finite_arguments() -> None:
    config = StdioMCPServerConfig(name="fake", command=(sys.executable, "-c", "pass"))
    client = StdioMCPClient(config)

    async def run() -> None:
        with pytest.raises(ValueError, match="finite"):
            await client.call_tool("echo_tool", {"value": float("nan")})
        await client.close()

    asyncio.run(run())


def test_stdio_mcp_ignores_late_response_after_timeout() -> None:
    script = textwrap.dedent(
        """
        import json, sys, time
        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            request_id = request.get("id")
            if method == "initialize":
                time.sleep(0.2)
                result = {"protocolVersion": "2025-06-18"}
            elif method == "notifications/initialized":
                continue
            elif method == "slow":
                time.sleep(0.2)
                result = {"ok": "slow"}
            elif method == "fast":
                result = {"ok": "fast"}
            else:
                result = {}
            print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)
        """
    )

    async def run() -> None:
        client = StdioMCPClient(
            StdioMCPServerConfig(
                name="late-response",
                command=(sys.executable, "-u", "-c", script),
                request_timeout_seconds=0.1,
                startup_timeout_seconds=1.0,
            )
        )
        try:
            with pytest.raises(Exception, match="timed out"):
                await client.request("slow", {})
            await asyncio.sleep(0.3)
            assert await client.request("fast", {}) == {"ok": "fast"}
        finally:
            await client.close()

    asyncio.run(run())


def test_stdio_mcp_rejects_malformed_tool_entries_and_cursors() -> None:
    class StubClient(StdioMCPClient):
        def __init__(self, responses: list[dict[str, object]]) -> None:
            super().__init__(
                StdioMCPServerConfig(name="stub", command=(sys.executable, "-c", "pass"))
            )
            self.responses = responses

        async def request(self, method: str, params: dict[str, object] | None = None):
            del method, params
            return self.responses.pop(0)

    async def run() -> None:
        with pytest.raises(MCPProtocolError, match="tool entry"):
            await StubClient([{"tools": [None]}]).list_tools()
        with pytest.raises(MCPProtocolError, match="nextCursor"):
            await StubClient([{"tools": [], "nextCursor": 1}]).list_tools()

    asyncio.run(run())


def test_stdio_mcp_rejects_malformed_tool_result() -> None:
    class StubClient(StdioMCPClient):
        async def request(self, method: str, params: dict[str, object] | None = None):
            del method, params
            return {"content": "not-a-list", "isError": "true"}

    async def run() -> None:
        client = StubClient(
            StdioMCPServerConfig(name="stub", command=(sys.executable, "-c", "pass"))
        )
        with pytest.raises(MCPProtocolError, match="content"):
            await client.call_tool("echo_tool", {})

    asyncio.run(run())
