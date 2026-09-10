from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest

from config.loader import load_configuration
from core.adapters.direct.errors import AdapterCancelled
from core.adapters.direct.openai_chat_sse import OpenAIChatSSEAdapter
from core.adapters.providers.claude import ClaudeProviderAdapter
from core.adapters.providers.gemini import GeminiProviderAdapter
from core.adapters.providers.openai import OpenAIProviderAdapter
from core.contracts.chat import ChatMessage, ChatRequest
from core.events.types import (
    ApprovalRequested,
    MurmurDelta,
    ReasoningDelta,
    TextDelta,
    TurnFinished,
)
from services.conversation import ConversationService
from services.model_routing import ChannelConfig, ModelRouter, RetryPolicy
from services.tools import PermissionService, ToolExecutionService, ToolRegistry
from services.tools.types import RiskLevel, ToolSpec
from wizard.configuration import ConfigurationWizard


class _MockOpenAIService:
    """进程内 OpenAI Chat Completions SSE 服务，记录真实请求体和调用顺序。"""

    def __init__(self, *, tool_first: bool = False, empty: bool = False) -> None:
        self.tool_first = tool_first
        self.empty = empty
        self.calls = 0
        self.requests: list[dict[str, object]] = []
        self.authorizations: list[str] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        payload = json.loads(request.content)
        self.requests.append(payload)
        self.authorizations.append(str(request.headers.get("authorization", "")))
        if self.empty:
            body = "data: [DONE]\n\n"
        elif self.tool_first and self.calls == 1:
            tools = payload.get("tools")
            assert isinstance(tools, list) and tools
            tool = tools[0]
            assert isinstance(tool, dict)
            function = tool.get("function")
            assert isinstance(function, dict)
            name = str(function["name"])
            chunks = (
                {"choices": [{"delta": {"reasoning_content": "正在检查"}}]},
                {"choices": [{"delta": {"murmur": "嗯……"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-danger-1",
                                        "function": {
                                            "name": name,
                                            "arguments": '{"value":7}',
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                },
                {"choices": [{"finish_reason": "tool_calls", "delta": {}}]},
            )
            body = (
                "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
        else:
            chunks = (
                {"choices": [{"delta": {"reasoning_content": "正在回答"}}]},
                {"choices": [{"delta": {"murmur": "让我想想。"}}]},
                {"choices": [{"delta": {"content": "模型已完成。"}}]},
                {"choices": [{"finish_reason": "stop", "delta": {}}]},
            )
            body = (
                "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks)
                + "data: [DONE]\n\n"
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
        )


def _adapter_factory(service: _MockOpenAIService):
    transport = httpx.MockTransport(service)

    def factory(channel: ChannelConfig) -> OpenAIProviderAdapter:
        return OpenAIProviderAdapter(
            channel,
            client=httpx.AsyncClient(transport=transport),
        )

    return factory


def _router(service: _MockOpenAIService, *, retry: RetryPolicy | None = None) -> ModelRouter:
    return ModelRouter(
        [
            ChannelConfig(
                "mock",
                protocol="openai_chat",
                base_url="https://mock.invalid/v1",
                model="mock-model",
                api_key="mock-secret",
                capabilities=("streaming", "tools", "reasoning"),
                retry=retry or RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            )
        ],
        adapter_factory=_adapter_factory(service),
    )


async def _collect(iterator: AsyncIterator[object]) -> list[object]:
    return [event async for event in iterator]


def test_openai_sse_conversation_streams_reasoning_murmur_and_text() -> None:
    service = _MockOpenAIService()
    router = _router(service)
    conversation = ConversationService(router)

    async def run() -> None:
        try:
            events = await _collect(conversation.stream("你好"))
            assert any(isinstance(event, ReasoningDelta) for event in events)
            assert any(isinstance(event, MurmurDelta) for event in events)
            assert any(
                isinstance(event, TextDelta) and event.delta == "模型已完成。" for event in events
            )
            assert any(isinstance(event, TurnFinished) for event in events)
            assert service.calls == 1
            assert service.requests[0]["stream"] is True
            assert "mock-secret" not in repr(conversation.presentation.snapshot)
        finally:
            await conversation.aclose()
            await router.aclose()

    asyncio.run(run())


def test_openai_sse_tool_permission_pauses_and_resumes_same_turn() -> None:
    service = _MockOpenAIService(tool_first=True)
    router = _router(service)
    registry = ToolRegistry()
    invocations: list[int] = []

    async def dangerous(arguments, _context):
        invocations.append(int(arguments["value"]))
        return {"ok": int(arguments["value"])}

    registry.register(
        ToolSpec(
            "system:danger",
            "危险操作",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            dangerous,
            RiskLevel.HIGH,
        )
    )
    conversation = ConversationService(
        router,
        tools=ToolExecutionService(registry, PermissionService()),
    )

    async def run() -> None:
        try:
            first = await _collect(conversation.stream("执行操作"))
            approval = next(event for event in first if isinstance(event, ApprovalRequested))
            assert invocations == []
            assert service.calls == 1
            resumed = await _collect(conversation.resume_approval_stream(approval.approval_id))
            assert invocations == [7]
            assert service.calls == 2
            assert any(
                isinstance(event, TextDelta) and event.delta == "模型已完成。" for event in resumed
            )
            follow_up = service.requests[1]
            messages = follow_up["messages"]
            assert isinstance(messages, list)
            assert any(
                isinstance(message, dict)
                and message.get("role") == "tool"
                and message.get("tool_call_id") == "call-danger-1"
                for message in messages
            )
        finally:
            await conversation.aclose()
            await router.aclose()

    asyncio.run(run())


def test_router_retries_transient_http_failure_before_first_sse_event() -> None:
    class RetryService(_MockOpenAIService):
        async def __call__(self, request: httpx.Request) -> httpx.Response:
            self.calls += 1
            self.requests.append(json.loads(request.content))
            if self.calls == 1:
                return httpx.Response(503, headers={"retry-after": "0"}, content=b"unavailable")
            body = 'data: {"choices":[{"delta":{"content":"重试成功"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body.encode(),
            )

    service = RetryService()
    router = _router(service, retry=RetryPolicy(max_attempts=2, initial_delay_seconds=0))
    request = ChatRequest("mock-model", (ChatMessage("user", "重试"),))

    async def run() -> None:
        try:
            response = await router.complete(request)
            assert response.text == "重试成功"
            assert service.calls == 2
        finally:
            await router.aclose()

    asyncio.run(run())


def test_connection_probe_rejects_empty_sse_and_pre_cancel_avoids_http() -> None:
    empty_service = _MockOpenAIService(empty=True)
    router = _router(empty_service)

    async def run() -> None:
        try:
            result = await router.test_connection(router.channels()[0])
            assert result["status"] == "unavailable"
            assert result["reason"] == "protocol"
            assert empty_service.calls == 1

            cancelled_service = _MockOpenAIService()
            channel = router.channels()[0]
            adapter = OpenAIProviderAdapter(
                channel,
                client=httpx.AsyncClient(transport=httpx.MockTransport(cancelled_service)),
            )
            cancel = asyncio.Event()
            cancel.set()
            with pytest.raises(AdapterCancelled):
                _ = [
                    event
                    async for event in adapter.stream(
                        ChatRequest("mock-model", (ChatMessage("user", "取消"),)),
                        cancel_event=cancel,
                    )
                ]
            assert cancelled_service.calls == 0
            await adapter.aclose()
        finally:
            await router.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("adapter_type", "channel"),
    (
        (
            OpenAIChatSSEAdapter,
            ChannelConfig(
                "direct-openai",
                protocol="openai_chat",
                base_url="https://mock.invalid/v1",
                model="mock-model",
            ),
        ),
        (
            OpenAIProviderAdapter,
            ChannelConfig(
                "provider-openai",
                protocol="openai_chat",
                base_url="https://mock.invalid/v1",
                model="mock-model",
            ),
        ),
        (
            ClaudeProviderAdapter,
            ChannelConfig(
                "provider-claude",
                protocol="anthropic_messages",
                base_url="https://mock.invalid/v1",
                model="mock-model",
            ),
        ),
        (
            GeminiProviderAdapter,
            ChannelConfig(
                "provider-gemini",
                protocol="gemini_generate",
                base_url="https://mock.invalid/v1",
                model="mock-model",
            ),
        ),
    ),
)
def test_pre_cancelled_adapter_does_not_open_network_stream(
    adapter_type, channel: ChannelConfig
) -> None:
    class NoRequestClient:
        def stream(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("预取消请求不得打开网络流")

    async def run() -> None:
        adapter = adapter_type(channel, client=NoRequestClient())
        cancel = asyncio.Event()
        cancel.set()
        with pytest.raises(AdapterCancelled):
            _ = [
                event
                async for event in adapter.stream(
                    ChatRequest("mock-model", (ChatMessage("user", "取消"),)),
                    cancel_event=cancel,
                )
            ]
        await adapter.aclose()

    asyncio.run(run())


def test_configuration_wizard_save_loads_route_without_persisting_secret(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    wizard = ConfigurationWizard(config_path)
    wizard.configure_channel(
        channel_id="mock",
        protocol="openai_chat",
        base_url="https://mock.invalid/v1",
        model="mock-model",
        api_key_env="MEAPET_ROUND90_KEY",
    )
    yaml_text = config_path.read_text(encoding="utf-8")
    assert "${MEAPET_ROUND90_KEY}" in yaml_text
    assert "mock-secret" not in yaml_text

    loaded = load_configuration(
        config_path,
        defaults={},
        environment={"MEAPET_ROUND90_KEY": "mock-secret"},
    )
    service = _MockOpenAIService()
    router = ModelRouter.from_mapping(loaded.values, adapter_factory=_adapter_factory(service))

    async def run() -> None:
        try:
            response = await router.complete(
                ChatRequest("mock-model", (ChatMessage("user", "配置后对话"),))
            )
            assert response.text == "模型已完成。"
            assert router.resolve("dialogue").primary.id == "mock"
            assert service.requests[0]["model"] == "mock-model"
            assert service.authorizations == ["Bearer mock-secret"]
            assert "mock-secret" not in repr(router.diagnostics())
        finally:
            await router.aclose()

    asyncio.run(run())
