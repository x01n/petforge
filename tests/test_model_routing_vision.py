from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator

import pytest

from core.adapters.direct import ProviderAdapter
from core.adapters.direct.errors import ProviderAdapterError
from core.contracts.chat import ChatMessage, ChatRequest, ToolCall
from core.events.types import ConversationContext, TextDelta, TurnFinished
from services.model_routing import (
    ChannelConfig,
    ModelRouter,
    ModelRoutingError,
    RetryPolicy,
    VisionSummaryPolicy,
    strip_images_and_attach_summary,
)

_IMAGE_DATA = base64.b64encode(b"minimal-png-fixture").decode("ascii")


class _RecordingAdapter(ProviderAdapter):
    provider = "recording"
    protocol = "openai_chat"

    def __init__(
        self,
        text: str,
        *,
        capabilities: frozenset[str] = frozenset({"streaming"}),
        error: ProviderAdapterError | None = None,
    ) -> None:
        self.text = text
        self.capabilities = capabilities
        self.error = error
        self.requests: list[ChatRequest] = []
        self.calls = 0

    async def stream(
        self,
        request: ChatRequest,
        *,
        context: ConversationContext | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AsyncIterator[object]:
        del cancel_event
        self.calls += 1
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        current = context or ConversationContext("profile", "session", "turn", 0)
        yield TextDelta(current, self.text)
        yield TurnFinished(current)


def _image_request(model: str = "main-model") -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=(
            ChatMessage(
                "user",
                (
                    {"type": "text", "text": "请看看这张图"},
                    {"type": "image", "media_type": "image/png", "data": _IMAGE_DATA},
                ),
            ),
        ),
    )


def _channel(
    channel_id: str,
    model: str,
    *,
    capabilities: frozenset[str] = frozenset({"streaming"}),
    priority: int = 10,
) -> ChannelConfig:
    return ChannelConfig(
        channel_id,
        base_url=f"https://{channel_id}.invalid/v1",
        model=model,
        capabilities=capabilities,
        priority=priority,
    )


def test_router_generates_bounded_vision_summary_before_nonvision_main() -> None:
    main = _RecordingAdapter("主模型回答")
    vision = _RecordingAdapter("一只橙色猫", capabilities=frozenset({"streaming", "vision"}))
    router = ModelRouter(
        [
            _channel("main", "main-model"),
            _channel("vision", "vision-model", capabilities=frozenset({"streaming", "vision"})),
        ],
        routes={
            "dialogue": {"channel": "main"},
            "vision": {"channel": "vision", "required_capabilities": ["streaming", "vision"]},
        },
        adapter_factory=lambda channel: {"main": main, "vision": vision}[channel.id],
        vision_policy={"max_summary_chars": 128, "max_tokens": 64},
    )

    response = asyncio.run(router.complete(_image_request()))

    assert response.text == "主模型回答"
    assert vision.calls == 1
    assert main.calls == 1
    vision_request = vision.requests[0]
    assert vision_request.tools == ()
    assert vision_request.tool_choice == "none"
    assert any(
        part.get("type") == "image"
        for message in vision_request.messages
        if isinstance(message.content, tuple)
        for part in message.content
    )
    main_request = main.requests[0]
    assert "一只橙色猫" in repr(main_request.messages[-1].content)
    assert '"type": "image"' not in repr(main_request.messages[-1].content)


def test_vision_summary_consumes_streamed_text_without_recursive_complete() -> None:
    trace: list[str] = []

    class _OrderedAdapter(_RecordingAdapter):
        async def stream(
            self,
            request: ChatRequest,
            *,
            context: ConversationContext | None = None,
            cancel_event: asyncio.Event | None = None,
        ) -> AsyncIterator[object]:
            del cancel_event
            self.calls += 1
            self.requests.append(request)
            current = context or ConversationContext("profile", "session", "turn", 0)
            has_image = any(
                part.get("type") == "image"
                for message in request.messages
                if isinstance(message.content, tuple)
                for part in message.content
            )
            trace.append("vision" if has_image else "main")
            if has_image:
                yield TextDelta(current, "一只")
                await asyncio.sleep(0)
                yield TextDelta(current, "橙色猫")
            else:
                yield TextDelta(current, "主模型回答")
            yield TurnFinished(current)

    class _NoVisionCompleteRouter(ModelRouter):
        async def complete(self, request: ChatRequest, *, task: str = "dialogue", **kwargs: object):
            if task == "vision":
                raise AssertionError("视觉摘要必须走流式路由")
            return await super().complete(request, task=task, **kwargs)

    main = _OrderedAdapter("主模型回答")
    vision = _OrderedAdapter("视觉摘要", capabilities=frozenset({"streaming", "vision"}))
    router = _NoVisionCompleteRouter(
        [
            _channel("main", "main-model"),
            _channel("vision", "vision-model", capabilities=frozenset({"streaming", "vision"})),
        ],
        routes={
            "dialogue": {"channel": "main"},
            "vision": {"channel": "vision"},
        },
        adapter_factory=lambda channel: {"main": main, "vision": vision}[channel.id],
    )

    response = asyncio.run(router.complete(_image_request()))

    assert response.text == "主模型回答"
    assert trace == ["vision", "main"]
    assert "一只橙色猫" in repr(main.requests[0].messages[-1].content)


def test_vision_summary_task_is_cancelled_with_main_stream() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()

        class _BlockingVisionAdapter(_RecordingAdapter):
            async def stream(
                self,
                request: ChatRequest,
                *,
                context: ConversationContext | None = None,
                cancel_event: asyncio.Event | None = None,
            ) -> AsyncIterator[object]:
                del cancel_event
                self.calls += 1
                self.requests.append(request)
                current = context or ConversationContext("profile", "session", "turn", 0)
                has_image = any(
                    part.get("type") == "image"
                    for message in request.messages
                    if isinstance(message.content, tuple)
                    for part in message.content
                )
                if has_image:
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        cancelled.set()
                    return
                yield TextDelta(current, "不会到达")

        main = _BlockingVisionAdapter("主模型", capabilities=frozenset({"streaming"}))
        vision = _BlockingVisionAdapter(
            "视觉摘要",
            capabilities=frozenset({"streaming", "vision"}),
        )
        router = ModelRouter(
            [
                _channel("main", "main-model"),
                _channel(
                    "vision",
                    "vision-model",
                    capabilities=frozenset({"streaming", "vision"}),
                ),
            ],
            routes={
                "dialogue": {"channel": "main"},
                "vision": {"channel": "vision"},
            },
            adapter_factory=lambda channel: {"main": main, "vision": vision}[channel.id],
        )

        async def consume() -> None:
            async for _event in router.stream(_image_request()):
                pass

        running = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), timeout=1)
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_vision_summary_request_removes_image_fields_from_main_extra_body() -> None:
    main = _RecordingAdapter("主模型回答")
    vision = _RecordingAdapter("图像摘要", capabilities=frozenset({"streaming", "vision"}))
    request = ChatRequest(
        model="main-model",
        messages=_image_request().messages,
        metadata={
            "extra_body": {
                "image": {"data": _IMAGE_DATA},
                "keep": "value",
                "nested": {"images": [{"data": _IMAGE_DATA}], "ok": True},
                "attachments": [{"type": "image", "data": _IMAGE_DATA}],
            }
        },
    )
    router = ModelRouter(
        [
            _channel("main", "main-model"),
            _channel("vision", "vision-model", capabilities=frozenset({"streaming", "vision"})),
        ],
        routes={"dialogue": {"channel": "main"}, "vision": {"channel": "vision"}},
        adapter_factory=lambda channel: {"main": main, "vision": vision}[channel.id],
    )

    asyncio.run(router.complete(request))

    extra_body = main.requests[0].metadata["extra_body"]
    assert extra_body == {"keep": "value", "nested": {"ok": True}}
    assert _IMAGE_DATA not in repr(main.requests[0])


def test_router_retries_vision_route_on_next_channel_before_main_request() -> None:
    main = _RecordingAdapter("主模型回答")
    broken = _RecordingAdapter(
        "不会返回",
        capabilities=frozenset({"streaming", "vision"}),
        error=ProviderAdapterError("unavailable", category="network", retryable=True),
    )
    backup = _RecordingAdapter(
        "备用视觉摘要",
        capabilities=frozenset({"streaming", "vision"}),
    )
    router = ModelRouter(
        [
            _channel("main", "main-model", priority=1),
            ChannelConfig(
                "vision-1",
                base_url="https://vision-1.invalid/v1",
                model="vision-a",
                capabilities=("streaming", "vision"),
                priority=2,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
            ChannelConfig(
                "vision-2",
                base_url="https://vision-2.invalid/v1",
                model="vision-b",
                capabilities=("streaming", "vision"),
                priority=3,
            ),
        ],
        routes={
            "dialogue": {"channel": "main"},
            "vision": {
                "channel": "vision-1",
                "fallback_channels": ["vision-2"],
                "required_capabilities": ["streaming", "vision"],
            },
        },
        adapter_factory=lambda channel: {
            "main": main,
            "vision-1": broken,
            "vision-2": backup,
        }[channel.id],
    )

    asyncio.run(router.complete(_image_request()))

    assert broken.calls == 1
    assert backup.calls == 1
    assert main.calls == 1
    assert "备用视觉摘要" in repr(main.requests[0].messages[-1].content)
    assert router.health("vision-1")["failures"] == 1


def test_router_does_not_call_vision_when_primary_declares_vision() -> None:
    main = _RecordingAdapter(
        "直接回答",
        capabilities=frozenset({"streaming", "vision"}),
    )
    vision = _RecordingAdapter("不应调用", capabilities=frozenset({"streaming", "vision"}))
    router = ModelRouter(
        [
            _channel("main", "main-model", capabilities=frozenset({"streaming", "vision"})),
            _channel("vision", "vision-model", capabilities=frozenset({"streaming", "vision"})),
        ],
        routes={"dialogue": {"channel": "main"}, "vision": {"channel": "vision"}},
        adapter_factory=lambda channel: {"main": main, "vision": vision}[channel.id],
    )

    asyncio.run(router.complete(_image_request()))

    assert main.calls == 1
    assert vision.calls == 0
    assert any(
        part.get("type") == "image"
        for part in main.requests[0].messages[-1].content
        if isinstance(main.requests[0].messages[-1].content, tuple)
    )


def test_router_uses_vision_summary_when_adapter_rejects_declared_vision() -> None:
    class RestrictedVisionAdapter(_RecordingAdapter):
        def supports(self, capability: str) -> bool:
            return capability != "vision"

    main = RestrictedVisionAdapter(
        "主模型回答",
        capabilities=frozenset({"streaming", "vision"}),
    )
    vision = _RecordingAdapter("视觉摘要", capabilities=frozenset({"streaming", "vision"}))
    router = ModelRouter(
        [
            _channel("main", "main-model", capabilities=frozenset({"streaming", "vision"})),
            _channel("vision", "vision-model", capabilities=frozenset({"streaming", "vision"})),
        ],
        routes={"dialogue": {"channel": "main"}, "vision": {"channel": "vision"}},
        adapter_factory=lambda channel: {"main": main, "vision": vision}[channel.id],
    )

    response = asyncio.run(router.complete(_image_request()))

    assert response.text == "主模型回答"
    assert main.calls == 1
    assert vision.calls == 1
    assert "视觉摘要" in repr(main.requests[0].messages[-1].content)


def test_image_capability_status_distinguishes_direct_and_summary_paths() -> None:
    direct = ModelRouter(
        [_channel("main", "main-model", capabilities=frozenset({"streaming", "vision"}))],
        adapter_factory=lambda _channel: _RecordingAdapter(
            "unused", capabilities=frozenset({"streaming", "vision"})
        ),
    )
    direct_status = direct.image_capability_status()
    assert direct_status["ready"] is True
    assert direct_status["mode"] == "direct"
    assert direct_status["channel_id"] == "main"

    summary = ModelRouter(
        [
            _channel("main", "main-model"),
            _channel("vision", "vision-model", capabilities=frozenset({"streaming", "vision"})),
        ],
        routes={"dialogue": {"channel": "main"}, "vision": {"channel": "vision"}},
        adapter_factory=lambda channel: _RecordingAdapter(
            "unused",
            capabilities=(
                frozenset({"streaming", "vision"})
                if channel.id == "vision"
                else frozenset({"streaming"})
            ),
        ),
    )
    summary_status = summary.image_capability_status()
    assert summary_status["ready"] is True
    assert summary_status["mode"] == "summary"
    assert summary_status["channel_id"] == "vision"

    unavailable = ModelRouter(
        [_channel("main", "main-model")],
        adapter_factory=lambda _channel: _RecordingAdapter("unused"),
    )
    unavailable_status = unavailable.image_capability_status()
    assert unavailable_status["ready"] is False
    assert unavailable_status["mode"] == "unavailable"


def test_router_fails_closed_when_vision_route_is_unavailable() -> None:
    main = _RecordingAdapter("不应调用")
    router = ModelRouter(
        [_channel("main", "main-model")],
        routes={"dialogue": {"channel": "main"}},
        adapter_factory=lambda _channel: main,
    )

    with pytest.raises(ModelRoutingError) as raised:
        asyncio.run(router.complete(_image_request()))

    assert raised.value.safe_message == "当前主模型无法识别图片，请先配置支持图片的视觉模型"
    assert main.calls == 0
    assert "https://" not in raised.value.safe_message
    assert _IMAGE_DATA not in str(raised.value)


def test_router_rejects_remote_image_part_without_sending_it_to_any_channel() -> None:
    main = _RecordingAdapter("不应调用")
    vision = _RecordingAdapter("不应调用", capabilities=frozenset({"streaming", "vision"}))
    request = ChatRequest(
        model="main-model",
        messages=(
            ChatMessage(
                "user",
                (
                    {"type": "text", "text": "看图"},
                    {"type": "image_url", "image_url": {"url": "https://private.invalid/a.png"}},
                ),
            ),
        ),
    )
    router = ModelRouter(
        [
            _channel("main", "main-model"),
            _channel("vision", "vision-model", capabilities=frozenset({"streaming", "vision"})),
        ],
        routes={"dialogue": {"channel": "main"}, "vision": {"channel": "vision"}},
        adapter_factory=lambda channel: {"main": main, "vision": vision}[channel.id],
    )

    with pytest.raises(ModelRoutingError) as raised:
        asyncio.run(router.complete(request))

    assert raised.value.safe_message == "图片格式不受支持，无法交给模型处理"
    assert main.calls == 0
    assert vision.calls == 0
    assert "private.invalid" not in raised.value.safe_message


def test_router_from_mapping_reads_vision_policy_from_route_without_exposing_prompt() -> None:
    router = ModelRouter.from_mapping(
        {
            "llm": {
                "channels": [
                    {
                        "id": "main",
                        "base_url": "https://main.invalid/v1",
                        "model": "main-model",
                    },
                    {
                        "id": "vision",
                        "base_url": "https://vision.invalid/v1",
                        "model": "vision-model",
                        "capabilities": ["streaming", "vision"],
                    },
                ],
                "routing": {
                    "dialogue": {"channel": "main"},
                    "vision": {
                        "channel": "vision",
                        "max_summary_chars": 300,
                        "max_tokens": 96,
                        "prompt": "只输出观察结果",
                    },
                },
            }
        },
        adapter_factory=lambda channel: _RecordingAdapter(
            "unused",
            capabilities=(
                frozenset({"streaming", "vision"})
                if channel.id == "vision"
                else frozenset({"streaming"})
            ),
        ),
    )

    assert router.vision_policy.max_summary_chars == 300
    assert router.vision_policy.max_tokens == 96
    assert router.vision_policy.prompt == "只输出观察结果"
    status = router.vision_status()
    assert status["ready"] is True
    assert status["channel_id"] == "vision"
    assert "https://" not in repr(status)


def test_router_from_mapping_accepts_direct_llm_vision_route_shape() -> None:
    router = ModelRouter.from_mapping(
        {
            "llm": {
                "channels": [
                    {
                        "id": "main",
                        "base_url": "https://main.invalid/v1",
                        "model": "main-model",
                    },
                    {
                        "id": "vision",
                        "base_url": "https://vision.invalid/v1",
                        "model": "vision-model",
                        "capabilities": ["vision"],
                    },
                ],
                "routing": {"dialogue": {"channel": "main"}},
                "vision": {"channel": "vision", "max_summary_chars": 256},
            }
        }
    )

    assert router.route("vision").channel_id == "vision"
    assert router.vision_policy.max_summary_chars == 256


def test_router_from_mapping_accepts_direct_llm_vision_channel_id() -> None:
    router = ModelRouter.from_mapping(
        {
            "llm": {
                "channels": [
                    {
                        "id": "vision",
                        "base_url": "https://vision.invalid/v1",
                        "model": "vision-model",
                        "capabilities": ["vision"],
                    }
                ],
                "vision": "vision",
            }
        }
    )

    assert router.route("vision").channel_id == "vision"


def test_router_from_mapping_allows_boolean_vision_disable() -> None:
    router = ModelRouter.from_mapping({"llm": {"channels": [], "routing": {"vision": False}}})

    assert router.vision_policy.enabled is False
    assert router.vision_status()["reason"] == "disabled"


def test_router_select_model_switches_channel_and_keeps_secret_out_of_receipt() -> None:
    router = ModelRouter(
        [
            ChannelConfig(
                "primary",
                base_url="https://primary.invalid/v1",
                model="model-a",
                models=("shared-model",),
                api_key="top-secret",
            ),
            ChannelConfig(
                "backup",
                base_url="https://backup.invalid/v1",
                model="model-b",
                models=("shared-model",),
                priority=20,
            ),
        ],
        routes={"dialogue": {"channel": "primary"}},
    )

    result = router.select_model("shared-model")

    assert result["status"] == "updated"
    assert result["channel_id"] == "primary"
    assert result["model"] == "shared-model"
    assert router.route("dialogue").model == "shared-model"
    assert "top-secret" not in repr(result)
    assert "https://" not in repr(result)


def test_router_health_snapshot_redacts_custom_provider_error() -> None:
    class LeakyAdapter(_RecordingAdapter):
        async def stream(self, request, *, context=None, cancel_event=None):
            del request, context, cancel_event
            raise ProviderAdapterError(
                "https://bad.invalid?api_key=secret-value",
                category="network",
                retryable=True,
            )
            yield  # 保持异步生成器契约

    adapter = LeakyAdapter("unused")
    router = ModelRouter(
        [
            ChannelConfig(
                "leaky",
                base_url="https://leaky.invalid/v1",
                model="model",
                api_key="secret-value",
            )
        ],
        adapter_factory=lambda _channel: adapter,
    )

    with pytest.raises(ProviderAdapterError):
        asyncio.run(router.complete(ChatRequest("model", (ChatMessage("user", "hello"),))))

    health = router.health("leaky")
    assert health["last_category"] == "network"
    assert health["last_error"] == "模型连接失败，请检查服务地址和网络"
    assert "secret-value" not in repr(health)
    assert "https://" not in repr(health)


def test_select_channel_for_vision_requires_declared_vision_capability() -> None:
    router = ModelRouter(
        [
            _channel("text", "text-model"),
            _channel(
                "vision",
                "vision-model",
                capabilities=frozenset({"streaming", "vision"}),
            ),
        ]
    )

    assert router.select_channel("text", task="vision")["reason"] == "capability"
    assert router.select_channel("vision", task="vision")["status"] == "updated"


def test_vision_task_auto_selects_only_vision_capable_channels() -> None:
    router = ModelRouter(
        [
            _channel("text", "text-model"),
            _channel(
                "vision",
                "vision-model",
                capabilities=frozenset({"streaming", "vision"}),
            ),
        ]
    )

    assert [row["channel_id"] for row in router.model_options("vision")] == ["vision"]
    assert router.resolve("vision").primary.id == "vision"


def test_model_options_are_deduplicated_and_report_cooldown_without_secrets() -> None:
    now = [100.0]
    router = ModelRouter(
        [
            ChannelConfig(
                "primary",
                base_url="https://primary.invalid/v1",
                model="model-a",
                models=("shared", "shared"),
                api_key="secret-value",
            ),
            ChannelConfig(
                "backup",
                base_url="https://backup.invalid/v1",
                model="model-b",
                models=("shared",),
                priority=20,
            ),
        ],
        clock=lambda: now[0],
    )
    router._mark_failure(
        "primary",
        ProviderAdapterError("leak", category="network", retryable=True),
    )

    options = router.model_options()

    assert [(row["channel_id"], row["model"]) for row in options] == [
        ("backup", "model-b"),
        ("backup", "shared"),
        ("primary", "model-a"),
        ("primary", "shared"),
    ]
    primary_shared = next(
        row for row in options if row["channel_id"] == "primary" and row["model"] == "shared"
    )
    assert primary_shared["cooling"] is True
    assert "secret-value" not in repr(options)


def test_resolve_does_not_bypass_cooldown_when_all_channels_are_cooling() -> None:
    now = [100.0]
    router = ModelRouter(
        [
            _channel("primary", "model-a", priority=1),
            _channel("backup", "model-b", priority=2),
        ],
        clock=lambda: now[0],
    )
    router._mark_failure(
        "primary",
        ProviderAdapterError("failure", category="network", retryable=True),
    )
    router._mark_failure(
        "backup",
        ProviderAdapterError("failure", category="network", retryable=True),
    )

    with pytest.raises(ModelRoutingError):
        router.resolve()
    now[0] += 1.0
    assert router.resolve().primary.id == "primary"


def test_channel_capabilities_are_normalized_for_case_insensitive_routing() -> None:
    channel = ChannelConfig(
        "vision",
        base_url="https://vision.invalid/v1",
        model="vision-model",
        capabilities=("Vision", "TOOLS"),
    )

    assert channel.supports("vision") is True
    assert channel.supports("tools") is True


def test_channel_constructor_honors_adapter_when_protocol_is_omitted() -> None:
    channel = ChannelConfig(
        "claude",
        adapter="anthropic_messages",
        base_url="https://claude.invalid",
        model="claude-model",
    )

    assert channel.protocol == "anthropic_messages"
    assert channel.adapter == "anthropic_messages"


def test_explicit_protocol_has_priority_over_adapter_alias() -> None:
    channel = ChannelConfig(
        "explicit",
        protocol="openai_chat",
        adapter="anthropic_messages",
        base_url="https://explicit.invalid/v1",
        model="model",
    )

    assert channel.protocol == "openai_chat"


@pytest.mark.parametrize(
    "value",
    (0, -1, 1.5, float("nan"), float("inf")),
)
def test_vision_summary_policy_rejects_invalid_limits(value: float) -> None:
    with pytest.raises(ValueError):
        VisionSummaryPolicy(max_summary_chars=value)


def test_strip_images_removes_camel_case_image_metadata() -> None:
    request = ChatRequest(
        model="main-model",
        messages=_image_request().messages,
        metadata={
            "nested": {
                "imageData": _IMAGE_DATA,
                "image_data": _IMAGE_DATA,
                "keep": "value",
            }
        },
    )

    cleaned = strip_images_and_attach_summary(request, "一只橙色猫", max_chars=128)

    assert cleaned.metadata == {"nested": {"keep": "value"}}
    assert _IMAGE_DATA not in repr(cleaned.metadata)


def test_strip_images_removes_image_fields_from_tool_call_arguments() -> None:
    tool_call = ToolCall(
        "call-1",
        "desktop:observe_foreground",
        {
            "image": _IMAGE_DATA,
            "imageData": _IMAGE_DATA,
            "nested": {"image_data": _IMAGE_DATA},
            "keep": "value",
        },
    )
    request = ChatRequest(
        model="main-model",
        messages=(
            ChatMessage("assistant", None, tool_calls=(tool_call,)),
            *_image_request().messages,
        ),
    )

    cleaned = strip_images_and_attach_summary(request, "一只橙色猫", max_chars=128)

    arguments = cleaned.messages[0].tool_calls[0].arguments
    assert arguments == {"keep": "value"}
    assert _IMAGE_DATA not in repr(arguments)
