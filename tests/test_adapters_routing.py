from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import pytest

from core.adapters.direct import (
    AdapterConfigurationError,
    AdapterMetadata,
    OpenAIChatSSEAdapter,
    ProviderAdapter,
    ProviderAdapterError,
    ProviderAdapterRuntime,
    ProviderResponse,
    RetryPolicy,
    ToolCallDelta,
)
from core.adapters.providers import provider_registry
from core.contracts.chat import ChatMessage, ChatRequest, ToolDefinition
from core.events.types import ConversationContext, ReasoningDelta, TextDelta, TurnFinished
from logger.events import event_fingerprint
from services.model_routing import ChannelConfig, ModelRouter, ModelRoutingError


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


class _RetryAdapter(ProviderAdapter):
    provider = "test"
    protocol = "test"

    def __init__(self, failures: int, *, fail_after_text: bool = False) -> None:
        self.failures = failures
        self.fail_after_text = fail_after_text
        self.calls = 0
        self.requested_models: list[str] = []
        self.requests: list[ChatRequest] = []

    async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
        self.requests.append(request)
        self.requested_models.append(request.model)
        self.calls += 1
        if self.fail_after_text:
            yield TextDelta(context or ConversationContext("p", "s", "t", 0), "partial")
            raise ProviderAdapterError("network", category="network", retryable=True)
        if self.calls <= self.failures:
            raise ProviderAdapterError("network", category="network", retryable=True)
        current = context or ConversationContext("p", "s", "t", 0)
        yield TextDelta(current, "ok")
        yield TurnFinished(current)


def _request(model: str = "demo") -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=(ChatMessage("user", "hello"),),
        tools=(ToolDefinition("pet:move", "move", {"type": "object"}),),
    )


def test_channel_priority_and_explicit_route() -> None:
    router = ModelRouter(
        [
            ChannelConfig("slow", base_url="https://slow.invalid/v1", model="demo", priority=20),
            ChannelConfig("fast", base_url="https://fast.invalid/v1", model="demo", priority=10),
        ],
        routes={"dialogue": {"model": "demo"}},
    )
    selection = router.resolve("dialogue")
    assert [channel.id for channel in selection.channels] == ["fast", "slow"]


def test_router_stream_applies_route_model_to_direct_request() -> None:
    adapter = _RetryAdapter(0)
    router = ModelRouter(
        [ChannelConfig("primary", base_url="https://primary.invalid/v1", model="route-model")],
        routes={"dialogue": {"channel": "primary", "model": "route-model"}},
        adapter_factory=lambda _channel: adapter,
    )

    response = asyncio.run(router.complete(_request("stale-model")))
    assert response.text == "ok"
    assert adapter.requested_models == ["route-model"]


def test_model_invocation_logs_are_structured_and_do_not_include_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = _RetryAdapter(0)
    router = ModelRouter(
        [ChannelConfig("primary", base_url="https://primary.invalid/v1", model="route-model")],
        adapter_factory=lambda _channel: adapter,
    )
    request = ChatRequest(
        model="route-model",
        messages=(ChatMessage("user", "private-user-prompt"),),
    )
    context = ConversationContext("profile", "session", "turn-sensitive", 1)

    with caplog.at_level(logging.INFO, logger="services.model_routing.router"):
        response = asyncio.run(router.complete(request, context=context))

    assert response.text == "ok"
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert '"event": "model.call.started"' in joined
    assert '"event": "model.call.completed"' in joined
    assert '"channel_id": "primary"' in joined
    payloads = [json.loads(record.getMessage()) for record in caplog.records]
    started = next(payload for payload in payloads if payload["event"] == "model.call.started")
    completed = next(payload for payload in payloads if payload["event"] == "model.call.completed")
    expected_fingerprint = event_fingerprint("turn-sensitive")
    assert started["correlation"] == expected_fingerprint
    assert started["operation"] == expected_fingerprint
    assert completed["operation"] == expected_fingerprint
    assert completed["reason_code"] == "ok"
    assert completed["duration_ms"] >= 0
    assert "private-user-prompt" not in joined
    assert "turn-sensitive" not in joined
    assert "https://primary.invalid" not in joined


def test_model_invocation_without_context_still_pairs_operation_fingerprint(
    caplog: pytest.LogCaptureFixture,
) -> None:
    router = ModelRouter(
        [ChannelConfig("primary", base_url="https://primary.invalid/v1", model="route-model")],
        adapter_factory=lambda _channel: _RetryAdapter(0),
    )
    request = ChatRequest(
        model="route-model",
        messages=(ChatMessage("user", "private-contextless-prompt"),),
    )

    with caplog.at_level(logging.INFO, logger="services.model_routing.router"):
        response = asyncio.run(router.complete(request))

    assert response.text == "ok"
    payloads = [json.loads(record.getMessage()) for record in caplog.records]
    started = next(payload for payload in payloads if payload["event"] == "model.call.started")
    completed = next(payload for payload in payloads if payload["event"] == "model.call.completed")
    assert len(started["operation"]) == 16
    assert started["operation"] == completed["operation"]
    assert started["correlation"] == completed["correlation"]
    assert "private-contextless-prompt" not in caplog.text


def test_explicit_channel_override_ignores_route_model_constraint() -> None:
    router = ModelRouter(
        [
            ChannelConfig("primary", base_url="https://primary.invalid/v1", model="model-a"),
            ChannelConfig("secondary", base_url="https://secondary.invalid/v1", model="model-b"),
        ],
        routes={"dialogue": {"channel": "primary", "model": "model-a"}},
    )

    selection = router.resolve("dialogue", channel_id="secondary")

    assert selection.primary.id == "secondary"
    assert selection.requested_model == ""


def test_router_select_channel_hot_switches_dialogue_route_without_persisting_secrets() -> None:
    router = ModelRouter(
        [
            ChannelConfig("primary", base_url="https://primary.invalid/v1", model="model-a"),
            ChannelConfig("backup", base_url="https://backup.invalid/v1", model="model-b"),
        ],
        routes={"dialogue": {"channel": "primary"}},
    )

    result = router.select_channel("backup")

    assert result["status"] == "updated"
    assert result["channel_id"] == "backup"
    assert router.resolve("dialogue").primary.id == "backup"
    assert router.resolve("dialogue").primary.selected_model == "model-b"
    # 热切换只影响内存中的当前路由，不会把旧的 API 密钥或端点塞进回执。
    assert "api_key" not in repr(result)
    assert "https://" not in repr(result)


def test_router_select_channel_rejects_disabled_or_incomplete_channel() -> None:
    router = ModelRouter(
        [
            ChannelConfig(
                "disabled", enabled=False, base_url="https://disabled.invalid/v1", model="m"
            ),
            ChannelConfig("incomplete", base_url="https://incomplete.invalid/v1"),
        ]
    )

    assert router.select_channel("disabled")["reason"] == "disabled"
    assert router.select_channel("incomplete")["reason"] == "configuration"
    assert router.select_channel("missing")["reason"] == "not_found"


def test_router_resolution_excludes_channels_without_model() -> None:
    router = ModelRouter([ChannelConfig("incomplete", base_url="https://incomplete.invalid/v1")])

    with pytest.raises(ModelRoutingError):
        router.resolve()


def test_explicit_route_fallback_skips_incomplete_channel() -> None:
    router = ModelRouter(
        [
            ChannelConfig("incomplete", base_url="https://incomplete.invalid/v1"),
            ChannelConfig("ready", base_url="https://ready.invalid/v1", model="demo"),
        ],
        routes={"dialogue": {"channel": "incomplete", "fallback_channels": ["ready"]}},
    )

    assert router.resolve().primary.id == "ready"


def test_direct_router_route_rejects_unknown_or_duplicate_fallbacks() -> None:
    router = ModelRouter(
        [ChannelConfig("primary", base_url="https://primary.invalid/v1", model="demo")],
    )

    with pytest.raises(ModelRoutingError, match="unknown fallback"):
        router.set_route("dialogue", {"channel": "primary", "fallback_channels": ["missing"]})
    with pytest.raises(ModelRoutingError, match="duplicate"):
        router.set_route(
            "dialogue",
            {"channel": "primary", "fallback_channels": ["primary", "primary"]},
        )
    with pytest.raises(ModelRoutingError, match="primary channel"):
        router.set_route("dialogue", {"channel": "primary", "fallback_channels": ["primary"]})


def test_route_spec_normalizes_direct_string_constraints() -> None:
    from services.model_routing.router import RouteSpec

    spec = RouteSpec(
        "vision",
        required_capabilities="streaming, vision",
        fallback_channels="backup, secondary",
    )

    assert spec.required_capabilities == frozenset({"streaming", "vision"})
    assert spec.fallback_channels == ("backup", "secondary")


def test_direct_router_route_rejects_unmatched_model_protocol_and_capability() -> None:
    router = ModelRouter(
        [
            ChannelConfig(
                "primary",
                protocol="openai_chat",
                base_url="https://primary.invalid/v1",
                model="model-a",
            )
        ],
    )

    with pytest.raises(ModelRoutingError, match="no ready channel"):
        router.set_route("dialogue", {"channel": "primary", "model": "missing"})
    with pytest.raises(ModelRoutingError, match="no ready channel"):
        router.set_route("dialogue", {"channel": "primary", "protocol": "anthropic_messages"})
    with pytest.raises(ModelRoutingError, match="no ready channel"):
        router.set_route("vision", {"channel": "primary"})


def test_router_resolution_keeps_detailed_diagnostic_out_of_public_error() -> None:
    error = None
    try:
        ModelRouter().resolve("dialogue")
    except Exception as exc:  # noqa: BLE001 - assert the concrete public contract below.
        error = exc
    assert error is not None
    assert isinstance(error, ModelRoutingError)
    assert "llm.channels" in str(error)
    assert error.safe_message == "模型渠道未就绪，请点击“配置模型”检查连接信息"
    assert "llm.channels" not in error.safe_message
    assert "MEAPET_API_BASE" not in error.safe_message


def test_router_select_channel_rejects_cooling_channel_until_probe_window_ends() -> None:
    now = [100.0]
    router = ModelRouter(
        [ChannelConfig("primary", base_url="https://primary.invalid/v1", model="m")],
        clock=lambda: now[0],
    )
    router._mark_failure("primary", ProviderAdapterError("network", category="network"))

    result = router.select_channel("primary")

    assert result["reason"] == "cooldown"


def test_router_diagnostics_redact_secret_and_explain_missing_channel() -> None:
    empty = ModelRouter()
    status = empty.diagnostics()
    assert status["ready"] is False
    assert "llm.channels" in status["reason"]
    configured = ModelRouter(
        [
            ChannelConfig(
                "primary",
                base_url="https://example.invalid/v1",
                model="demo",
                api_key="secret-value",
            )
        ]
    )
    channel_status = configured.diagnostics()["channels"][0]
    assert channel_status["ready"] is True
    assert channel_status["api_key_configured"] is True
    assert "secret-value" not in repr(channel_status)


def test_router_connection_probe_uses_minimal_tool_free_request(
    caplog: pytest.LogCaptureFixture,
) -> None:
    adapter = _RetryAdapter(0)
    router = ModelRouter(
        (),
        adapter_factory=lambda _channel: adapter,
    )
    channel = ChannelConfig(
        "probe",
        base_url="https://probe.invalid/v1",
        model="probe-model",
        api_key="probe-secret",
    )

    with caplog.at_level(logging.INFO, logger="services.model_routing.router"):
        result = asyncio.run(router.test_connection(channel))

    assert result["status"] == "available"
    assert result["ready"] is True
    assert result["channel_id"] == "probe"
    assert result["model"] == "probe-model"
    assert "probe-secret" not in repr(result)
    # 探测直接复用统一适配器运行时，但不会把桌面工具暴露给供应商。
    assert adapter.calls == 1
    assert adapter.requested_models == ["probe-model"]
    assert adapter.requests[0].messages == (ChatMessage("user", "连接测试"),)
    assert adapter.requests[0].tools == ()
    assert adapter.requests[0].tool_choice == "none"
    assert adapter.requests[0].max_tokens == 1
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert '"event": "model.probe.started"' in joined
    assert '"event": "model.probe.completed"' in joined
    assert '"tool_count": 0' in joined
    payloads = [json.loads(record.getMessage()) for record in caplog.records]
    started = next(payload for payload in payloads if payload["event"] == "model.probe.started")
    completed = next(payload for payload in payloads if payload["event"] == "model.probe.completed")
    assert started["operation"] == started["correlation"]
    assert completed["operation"] == started["operation"]
    assert completed["reason_code"] == "ok"
    assert completed["duration_ms"] >= 0
    assert "probe-secret" not in joined
    assert "https://probe.invalid" not in joined
    assert "连接测试" not in joined


def test_router_connection_probe_logs_cleanup_failure_without_exception_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "sk-private-probe-cleanup"

    class CleanupFailAdapter(_RetryAdapter):
        async def aclose(self) -> None:
            raise RuntimeError(f"token={secret} /home/private/provider.sock")

    router = ModelRouter((), adapter_factory=lambda _channel: CleanupFailAdapter(0))
    channel = ChannelConfig(
        "probe-cleanup",
        base_url="https://probe.invalid/v1",
        model="probe-model",
    )

    with caplog.at_level(logging.INFO, logger="services.model_routing.router"):
        result = asyncio.run(router.test_connection(channel))

    assert result["status"] == "available"
    payloads = [json.loads(record.getMessage()) for record in caplog.records]
    cleanup = next(
        payload for payload in payloads if payload["event"] == "model.probe.cleanup_failed"
    )
    completed = next(payload for payload in payloads if payload["event"] == "model.probe.completed")
    assert cleanup["status"] == "degraded"
    assert cleanup["reason_code"] == "cleanup_failed"
    assert cleanup["error_type"] == "RuntimeError"
    assert cleanup["operation"] == completed["operation"]
    assert secret not in caplog.text
    assert "/home/private" not in caplog.text


def test_router_connection_probe_does_not_reuse_cached_runtime_for_edited_channel() -> None:
    adapters: list[_RetryAdapter] = []

    def factory(_channel: ChannelConfig) -> _RetryAdapter:
        adapter = _RetryAdapter(0)
        adapters.append(adapter)
        return adapter

    router = ModelRouter(
        [
            ChannelConfig(
                "primary",
                base_url="https://old.invalid/v1",
                model="old-model",
            )
        ],
        adapter_factory=factory,
    )
    # 建立并缓存真实对话运行时，随后用相同 ID 测试一份新配置。
    router.runtime_for()
    result = asyncio.run(
        router.test_connection(
            {
                "id": "primary",
                "protocol": "openai_chat",
                "base_url": "https://new.invalid/v1",
                "model": "new-model",
            }
        )
    )

    assert result["status"] == "available"
    assert len(adapters) == 2
    assert adapters[0].requested_models == []
    assert adapters[1].requested_models == ["new-model"]


def test_router_connection_probe_does_not_change_active_channel_health() -> None:
    class FailingAdapter(ProviderAdapter):
        capabilities = frozenset({"streaming"})

        async def stream(self, _request, *, context=None, cancel_event=None):
            del context, cancel_event
            raise ProviderAdapterError("probe failure", category="network", retryable=True)
            yield  # 保持异步生成器契约

    router = ModelRouter(
        [ChannelConfig("primary", base_url="https://active.invalid/v1", model="active-model")],
        adapter_factory=lambda _channel: FailingAdapter(),
    )
    before = router.health("primary")

    result = asyncio.run(
        router.test_connection(
            ChannelConfig("primary", base_url="https://edited.invalid/v1", model="edited-model")
        )
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == "network"
    assert router.health("primary") == before


def test_router_connection_probe_returns_redacted_failure() -> None:
    class FailingAdapter(ProviderAdapter):
        capabilities = frozenset({"streaming"})

        async def stream(self, _request, *, context=None, cancel_event=None):
            del context, cancel_event
            raise ProviderAdapterError(
                "authorization=probe-secret",
                category="authentication",
                retryable=False,
            )
            yield  # 保持异步生成器契约

    router = ModelRouter((), adapter_factory=lambda _channel: FailingAdapter())
    channel = ChannelConfig(
        "probe-failure",
        base_url="https://probe.invalid/v1",
        model="probe-model",
        api_key="probe-secret",
    )

    result = asyncio.run(router.test_connection(channel))

    assert result["status"] == "unavailable"
    assert result["ready"] is False
    assert result["reason"] == "authentication"
    assert result["message"] == "模型连接失败，请检查密钥设置"
    assert "probe-secret" not in repr(result)


def test_router_connection_probe_rejects_incomplete_channel_without_adapter_call() -> None:
    called = False

    def factory(_channel):
        nonlocal called
        called = True
        return _RetryAdapter(0)

    router = ModelRouter((), adapter_factory=factory)
    result = asyncio.run(router.test_connection({"id": "incomplete", "protocol": "openai_chat"}))

    assert result["status"] == "unavailable"
    assert result["reason"] == "configuration"
    assert result["message"] == "模型连接配置不完整"
    assert called is False


@pytest.mark.parametrize(
    "field",
    ("initial_delay_seconds", "max_delay_seconds", "backoff_multiplier", "jitter_ratio"),
)
def test_retry_policy_rejects_non_finite_numeric_values(field: str) -> None:
    values = {field: float("nan")}
    with pytest.raises(ValueError, match="finite"):
        RetryPolicy(**values)


@pytest.mark.parametrize(
    "values",
    (
        {"max_attempts": True},
        {"max_attempts": 1.5},
        {"retryable_status_codes": frozenset({True})},
        {"retryable_status_codes": frozenset({503, 503.5})},
        {"respect_retry_after": 1},
    ),
)
def test_retry_policy_rejects_ambiguous_control_types(values: dict[str, object]) -> None:
    """重试边界不得静默把布尔值或小数截断为控制参数。"""

    with pytest.raises(ValueError):
        RetryPolicy(**values)


def test_retry_policy_applies_categories_and_status_codes_to_all_errors() -> None:
    """类别禁用和状态码白名单必须对原始异常/已分类异常保持一致。"""

    no_categories = RetryPolicy(
        max_attempts=2,
        retryable_categories=frozenset(),
    )
    assert not no_categories.should_retry(TimeoutError(), attempt=1)

    policy = RetryPolicy(
        max_attempts=2,
        retryable_categories=frozenset({"network"}),
        retryable_status_codes=frozenset({503}),
    )
    rejected = ProviderAdapterError(
        "network failed",
        category="network",
        status_code=400,
        retryable=True,
    )
    accepted = ProviderAdapterError(
        "network failed",
        category="network",
        status_code=503,
        retryable=True,
    )
    assert not policy.should_retry(rejected, attempt=1)
    assert policy.should_retry(accepted, attempt=1)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_channel_rejects_non_finite_timeout(value: float) -> None:
    with pytest.raises(AdapterConfigurationError, match="timeout_seconds"):
        ChannelConfig("invalid-timeout", base_url="https://example.invalid", timeout_seconds=value)


def test_channel_mapping_redacts_sensitive_headers_with_api_key() -> None:
    channel = ChannelConfig(
        "secret-channel",
        base_url="https://example.invalid/v1",
        headers={
            "Authorization": "Bearer hidden",
            "X-Api-Key": "hidden-key",
            "X-Trace-Id": "trace-visible",
        },
    )
    mapped = channel.as_mapping(redact_api_key=True)
    assert mapped["api_key"] == ""
    assert mapped["headers"] == {
        "Authorization": "***",
        "X-Api-Key": "***",
        "X-Trace-Id": "trace-visible",
    }


def test_channel_mapping_preserves_custom_request_fields_and_redacts_nested_secrets() -> None:
    channel = ChannelConfig(
        "custom-channel",
        base_url="https://example.invalid/v1",
        model="demo",
        endpoint="/chat/custom",
        endpoints={"health": "/health"},
        request_body_template={"nested": {"token": "hidden", "keep": "value"}},
        metadata={"credential": "hidden", "tag": "visible"},
    )

    mapped = channel.as_mapping(redact_api_key=True)

    assert mapped["adapter"] == "openai_chat"
    assert mapped["endpoint"] == "/chat/custom"
    assert mapped["endpoints"] == {"health": "/health"}
    assert mapped["request_body_template"] == {"nested": {"token": "***", "keep": "value"}}
    assert mapped["metadata"] == {"credential": "***", "tag": "visible"}


def test_router_explicit_protocol_filters_channels_and_runtime_for() -> None:
    openai = _RetryAdapter(0)
    claude = _RetryAdapter(0)
    router = ModelRouter(
        [
            ChannelConfig(
                "openai",
                protocol="openai_chat",
                base_url="https://openai.invalid/v1",
                model="demo",
                priority=1,
            ),
            ChannelConfig(
                "claude",
                protocol="anthropic_messages",
                base_url="https://claude.invalid/v1",
                model="demo",
                priority=2,
            ),
        ],
        adapter_factory=lambda channel: {"openai": openai, "claude": claude}[channel.id],
    )

    selection = router.resolve(protocol="anthropic_messages")
    assert [channel.id for channel in selection.channels] == ["claude"]
    assert router.runtime_for(protocol="anthropic_messages").adapter is claude
    response = asyncio.run(router.complete(_request(), protocol="anthropic_messages"))
    assert response.text == "ok"
    assert claude.calls == 1
    assert openai.calls == 0


def test_router_complete_uses_configured_fallback() -> None:
    first = _RetryAdapter(1)
    second = _RetryAdapter(0)
    channels = [
        ChannelConfig(
            "first",
            base_url="https://first.invalid/v1",
            model="demo",
            priority=1,
            retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
        ),
        ChannelConfig(
            "second",
            base_url="https://second.invalid/v1",
            model="demo",
            priority=2,
            retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
        ),
    ]
    adapters = {"first": first, "second": second}
    router = ModelRouter(channels, adapter_factory=lambda channel: adapters[channel.id])
    response = asyncio.run(router.complete(_request()))
    assert response.text == "ok"
    assert first.calls == 1
    assert second.calls == 1


def test_router_falls_back_when_primary_returns_an_empty_stream() -> None:
    class EmptyAdapter(ProviderAdapter):
        provider = "empty"
        protocol = "openai_chat"

        async def stream(self, _request, *, context=None, cancel_event=None):
            del context, cancel_event
            if False:
                yield TextDelta(ConversationContext("p", "s", "t", 0), "")

    backup = _RetryAdapter(0)
    router = ModelRouter(
        [
            ChannelConfig(
                "empty",
                base_url="https://empty.invalid/v1",
                model="demo",
                priority=1,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
            ChannelConfig(
                "backup",
                base_url="https://backup.invalid/v1",
                model="demo",
                priority=2,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
        ],
        adapter_factory=lambda channel: EmptyAdapter() if channel.id == "empty" else backup,
    )

    response = asyncio.run(router.complete(_request()))
    assert response.text == "ok"
    assert backup.calls == 1
    assert router.health("empty")["last_category"] == "protocol"


def test_router_ignores_synthetic_finish_from_empty_openai_sse() -> None:
    first_channel = ChannelConfig(
        "openai-empty",
        base_url="https://openai-empty.invalid/v1",
        model="demo",
        priority=1,
        retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
    )
    backup_channel = ChannelConfig(
        "backup",
        base_url="https://backup.invalid/v1",
        model="demo",
        priority=2,
        retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
    )
    empty_response = _Response([])
    backup = _RetryAdapter(0)

    def factory(channel):
        if channel.id == "openai-empty":
            return OpenAIChatSSEAdapter(channel, client=_Client(empty_response))
        return backup

    router = ModelRouter(
        [first_channel, backup_channel],
        adapter_factory=factory,
    )

    response = asyncio.run(router.complete(_request()))

    assert response.text == "ok"
    assert backup.calls == 1
    assert router.health("openai-empty")["last_category"] == "protocol"


def test_router_withholds_empty_channel_metadata_and_terminal_before_failover() -> None:
    class EmptyMetadataAdapter(ProviderAdapter):
        provider = "empty-metadata"
        protocol = "openai_chat"

        async def stream(self, _request, *, context=None, cancel_event=None):
            del cancel_event
            current = context or ConversationContext("p", "s", "empty-metadata", 0)
            yield AdapterMetadata(current, {"source": "empty"})
            yield TurnFinished(current)

    backup = _RetryAdapter(0)
    router = ModelRouter(
        [
            ChannelConfig(
                "empty-metadata",
                base_url="https://empty.invalid/v1",
                model="demo",
                priority=1,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
            ChannelConfig(
                "backup-after-metadata",
                base_url="https://backup.invalid/v1",
                model="demo",
                priority=2,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
        ],
        adapter_factory=lambda channel: (
            EmptyMetadataAdapter() if channel.id == "empty-metadata" else backup
        ),
    )

    async def collect() -> list[object]:
        return [event async for event in router.stream(_request())]

    events = asyncio.run(collect())

    assert [type(event) for event in events] == [TextDelta, TurnFinished]
    assert router.health("empty-metadata")["last_category"] == "protocol"


def test_router_releases_buffered_metadata_after_first_content_event() -> None:
    class MetadataThenTextAdapter(ProviderAdapter):
        provider = "metadata-text"
        protocol = "openai_chat"

        async def stream(self, _request, *, context=None, cancel_event=None):
            del cancel_event
            current = context or ConversationContext("p", "s", "metadata-text", 0)
            yield AdapterMetadata(current, {"source": "primary"})
            yield TextDelta(current, "第一句。")
            yield TurnFinished(current)

    router = ModelRouter(
        [
            ChannelConfig(
                "metadata-text",
                base_url="https://primary.invalid/v1",
                model="demo",
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            )
        ],
        adapter_factory=lambda _channel: MetadataThenTextAdapter(),
    )

    async def collect() -> list[object]:
        return [event async for event in router.stream(_request())]

    events = asyncio.run(collect())

    assert [type(event) for event in events] == [AdapterMetadata, TextDelta, TurnFinished]
    assert events[0].values == {"source": "primary"}


def test_router_classifies_raw_runtime_network_error_for_failover() -> None:
    class RawRuntime:
        def __init__(self, *, failing: bool) -> None:
            self.failing = failing

        async def stream(self, request, *, context=None, cancel_event=None):
            del request, cancel_event
            if self.failing:
                raise OSError("transport secret")
            current = context or ConversationContext("p", "s", "t", 0)
            yield TextDelta(current, "raw-ok")
            yield TurnFinished(current)

        async def aclose(self) -> None:
            return None

    runtimes = {
        "first": RawRuntime(failing=True),
        "second": RawRuntime(failing=False),
    }
    router = ModelRouter(
        [
            ChannelConfig(
                "first",
                base_url="https://first.invalid/v1",
                model="demo",
                priority=1,
            ),
            ChannelConfig(
                "second",
                base_url="https://second.invalid/v1",
                model="demo",
                priority=2,
            ),
        ],
        adapter_factory=lambda channel: object(),
        runtime_factory=lambda _adapter, channel: runtimes[channel.id],
    )

    response = asyncio.run(router.complete(_request()))
    assert response.text == "raw-ok"
    assert router.health("first")["last_category"] == "network"


def test_router_fallback_switches_to_next_channel_model() -> None:
    first = _RetryAdapter(1)
    second = _RetryAdapter(0)
    router = ModelRouter(
        [
            ChannelConfig(
                "first-model",
                base_url="https://first.invalid/v1",
                model="model-a",
                priority=1,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
            ChannelConfig(
                "second-model",
                base_url="https://second.invalid/v1",
                model="model-b",
                priority=2,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
        ],
        adapter_factory=lambda channel: {
            "first-model": first,
            "second-model": second,
        }[channel.id],
    )

    response = asyncio.run(router.complete(_request("model-a")))

    assert response.text == "ok"
    assert first.requested_models == ["model-a"]
    assert second.requested_models == ["model-b"]


def test_router_explicit_model_stays_fixed_across_fallback() -> None:
    first = _RetryAdapter(1)
    second = _RetryAdapter(0)
    router = ModelRouter(
        [
            ChannelConfig(
                "first-model",
                base_url="https://first.invalid/v1",
                model="model-a",
                models=("shared",),
                priority=1,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
            ChannelConfig(
                "second-model",
                base_url="https://second.invalid/v1",
                model="model-b",
                models=("shared",),
                priority=2,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
        ],
        adapter_factory=lambda channel: {
            "first-model": first,
            "second-model": second,
        }[channel.id],
    )

    response = asyncio.run(router.complete(_request("shared"), model="shared"))

    assert response.text == "ok"
    assert first.requested_models == ["shared"]
    assert second.requested_models == ["shared"]


def test_runtime_retries_only_before_first_event() -> None:
    async def run() -> None:
        waits: list[float] = []
        adapter = _RetryAdapter(1)
        runtime = ProviderAdapterRuntime(
            adapter,
            retry_policy=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
            sleep=lambda delay: _record_wait(waits, delay),
        )
        events = [event async for event in runtime.stream(_request())]
        assert adapter.calls == 2
        assert waits == [0]
        assert [event.delta for event in events if isinstance(event, TextDelta)] == ["ok"]

        partial = ProviderAdapterRuntime(
            _RetryAdapter(0, fail_after_text=True),
            retry_policy=RetryPolicy(max_attempts=3, initial_delay_seconds=0),
            sleep=lambda delay: _record_wait(waits, delay),
        )
        with pytest.raises(ProviderAdapterError):
            _ = [event async for event in partial.stream(_request())]

    asyncio.run(run())


def test_runtime_does_not_retry_after_non_text_event() -> None:
    class ReasoningThenFailure(ProviderAdapter):
        provider = "reasoning-failure"
        protocol = "test"
        capabilities = frozenset({"streaming", "reasoning"})

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request, *, context=None, cancel_event=None):
            del request, cancel_event
            self.calls += 1
            yield ReasoningDelta(context or ConversationContext("p", "s", "t", 0), "思考")
            raise ProviderAdapterError("network", category="network", retryable=True)

    async def run() -> None:
        adapter = ReasoningThenFailure()
        runtime = ProviderAdapterRuntime(
            adapter,
            retry_policy=RetryPolicy(max_attempts=3, initial_delay_seconds=0),
        )
        with pytest.raises(ProviderAdapterError):
            _ = [event async for event in runtime.stream(_request())]
        assert adapter.calls == 1

    asyncio.run(run())


def test_router_health_cooldown_skips_failed_channel_until_probe() -> None:
    now = [100.0]
    first = _RetryAdapter(1)
    second = _RetryAdapter(0)
    adapters = {"first": first, "second": second}
    router = ModelRouter(
        [
            ChannelConfig(
                "first",
                base_url="https://first.invalid/v1",
                model="demo",
                priority=1,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
            ChannelConfig(
                "second",
                base_url="https://second.invalid/v1",
                model="demo",
                priority=2,
                retry=RetryPolicy(max_attempts=1, initial_delay_seconds=0),
            ),
        ],
        adapter_factory=lambda channel: adapters[channel.id],
        clock=lambda: now[0],
    )
    response = asyncio.run(router.complete(_request()))
    assert response.text == "ok"
    assert router.health("first")["cooldown_until"] > now[0]
    assert [channel.id for channel in router.resolve().channels] == ["second"]
    now[0] += 2.0
    assert [channel.id for channel in router.resolve().channels] == ["first", "second"]


def test_runtime_honors_external_cancel_event() -> None:
    class BlockingAdapter(ProviderAdapter):
        async def stream(self, _request, *, context=None, cancel_event=None):
            while cancel_event is not None and not cancel_event.is_set():
                await asyncio.sleep(0.01)
            raise asyncio.CancelledError()
            yield  # 保持异步生成器契约

    async def run() -> None:
        runtime = ProviderAdapterRuntime(BlockingAdapter())
        cancel = asyncio.Event()
        task = asyncio.create_task(_collect(runtime.stream(_request(), cancel_event=cancel)))
        await asyncio.sleep(0.02)
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())


async def _record_wait(values: list[float], delay: float) -> None:
    values.append(delay)


def test_openai_chat_sse_events_and_tool_delta() -> None:
    response = _Response(
        [
            'data: {"model":"served-legacy","choices":[{"delta":{"content":"hi"}}]}',
            "",
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-1",'
                '"function":{"name":"pet:move","arguments":"{\\"x\\":1}"}}]}}]} '
            ).rstrip(),
            "",
            'data: {"choices":[{"finish_reason":"stop","delta":{}}]}',
            "",
            "data: [DONE]",
            "",
        ]
    )
    client = _Client(response)
    channel = ChannelConfig("main", base_url="https://example.invalid/v1", model="demo")
    adapter = OpenAIChatSSEAdapter(channel, client=client)
    context = ConversationContext("p", "s", "t", 1)
    events = asyncio.run(_collect(adapter.stream(_request(), context=context)))
    assert [event.delta for event in events if isinstance(event, TextDelta)] == ["hi"]
    assert any(
        isinstance(event, ToolCallDelta) and event.identity == "pet:move" for event in events
    )
    assert isinstance(events[-1], TurnFinished)
    assert events[-1].response_model == "served-legacy"
    body = client.calls[0][1]["json"]
    assert body["stream"] is True
    assert body["tools"][0]["function"]["name"] == "pet:move"


def test_provider_response_rejects_malformed_tool_arguments() -> None:
    context = ConversationContext("p", "s", "t", 1)
    response = ProviderResponse().add(
        ToolCallDelta(
            context=context,
            index=0,
            call_id="call-invalid",
            identity="pet:move",
            arguments_delta="not-json",
        )
    )

    finalized = response.finalized()

    assert finalized.tool_calls == ()
    invalid = finalized.metadata["invalid_tool_calls"]
    assert invalid == ({"call_id": "call-invalid", "identity": "pet:move"},)


def test_adapter_diagnostics_are_sanitized_and_capability_focused() -> None:
    adapter = _RetryAdapter(0)
    runtime = ProviderAdapterRuntime(adapter)

    assert runtime.diagnostics() == {
        "provider": "test",
        "protocol": "test",
        "capabilities": ("streaming",),
        "closed": False,
        "active_requests": 0,
    }
    rows = provider_registry.diagnostics()
    assert any(row["protocol"] == "openai_responses" for row in rows)
    assert all(set(row) == {"protocol", "provider", "capabilities"} for row in rows)


def test_adapter_runtime_diagnostics_use_effective_channel_capabilities() -> None:
    class RestrictedAdapter(_RetryAdapter):
        capabilities = frozenset({"streaming", "tools", "vision"})

        def supports(self, capability: str) -> bool:
            return capability in {"streaming", "tools"}

    runtime = ProviderAdapterRuntime(RestrictedAdapter(0))

    assert runtime.supports("vision") is False
    assert runtime.capabilities == frozenset({"streaming", "tools"})
    assert runtime.diagnostics()["capabilities"] == ("streaming", "tools")


async def _collect(iterator: AsyncIterator):
    return [event async for event in iterator]
