"""第 9 轮方向 2：模型韧性与凭证卫生的定向回归测试。

覆盖：
  A. 未配置 Timedelta 渠道组装有界 Runtime（超时盲区）
     A2. 适配器独立使用渠道 0 超时时叠加默认总时限
  B. 审计入库正文长度受限与短消息原样
  C. logger 密钥正则覆盖裸露 sk- 与 x-goog-api-key/sessionKey
  D. scheduler 最低频率下限拒绝高频表达式
  F. scheduler 状态接口不回传异常正文
仅本地假客户端，不发起任何真实网络调用。异常时间轴全部走
无网络、短超时的局部事件循环，属于快层。
"""

from __future__ import annotations

import asyncio

import pytest

from core.adapters.direct.base import ProviderAdapterRuntime
from core.adapters.direct.errors import ProviderAdapterError
from core.adapters.direct.openai_chat_sse import (
    _DEFAULT_TOTAL_TIMEOUT_SECONDS,
    OpenAIChatSSEAdapter,
)
from core.contracts.chat import ChatMessage, ChatRequest
from logger.events import sanitize_log_text
from services.model_routing.channels import ChannelConfig
from services.model_routing.router import (
    _AUDIT_MESSAGE_TEXT_LIMIT,
    _AUDIT_TRUNCATED_MARK,
    ModelRouter,
    _audit_request_payload,
)
from services.scheduler.scheduler import (
    _MIN_SCHEDULE_INTERVAL_SECONDS,
    ScheduleExpressionError,
    SchedulerService,
)
from services.tools.builtins import register_builtin_tools
from services.tools.executor import ToolExecutionService
from services.tools.permissions import PermissionService
from services.tools.registry import ToolRegistry
from services.tools.types import ToolCallContext


def _chat_request(
    messages: tuple[ChatMessage, ...] | None = None,
    model: str = "demo",
) -> ChatRequest:
    return ChatRequest(
        model=model,
        messages=messages or (ChatMessage(role="user", content="你好"),),
    )


def _channel(timeout: float = 0.0) -> ChannelConfig:
    return ChannelConfig(
        "round9",
        base_url="https://round9.invalid/v1",
        model="demo",
        timeout_seconds=timeout,
    )


class _RecordingAdapter:
    """记录构造参数的轻量适配器钩子，用于验证 Runtime 的时限。"""

    def __init__(self) -> None:
        self.constructed_with: tuple[object, ...] = ()

    async def stream(self, request, *, context=None, cancel_event=None):
        del request, context, cancel_event
        if False:  # pragma: no cover - 保持 async 生成器形态
            yield None

    def supports(self, capability: str) -> bool:
        del capability
        return True


def test_channel_zero_timeout_assembles_bounded_runtime() -> None:
    """A：未配置超时（0.0）的渠道经默认工厂组装出默认总时限 Runtime。"""

    adapter = _RecordingAdapter()
    router = ModelRouter(
        (_channel(0.0),),
        adapter_factory=lambda _channel: adapter,
    )
    runtime = router.runtime_for(task="dialogue")
    assert isinstance(runtime, ProviderAdapterRuntime)
    assert runtime.timeout_seconds == _DEFAULT_TOTAL_TIMEOUT_SECONDS


def test_router_explicit_timeout_is_passed_through() -> None:
    """A：渠道显式超时保持原值透传，不受默认值影响。"""

    adapter = _RecordingAdapter()
    router = ModelRouter(
        (_channel(25.0),),
        adapter_factory=lambda _channel: adapter,
    )
    runtime = router.runtime_for(task="dialogue")
    assert isinstance(runtime, ProviderAdapterRuntime)
    assert runtime.timeout_seconds == 25.0


class _StalledClient:
    """返回一个在异步上下文内永不回字节的响应，让时限自行到期。"""

    async def __aenter__(self) -> _StalledClient:
        return self

    def stream(self, *_args, **_kwargs):
        return _StalledResponse()


class _StalledResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc_info):
        await asyncio.Event().wait()

    def aiter_lines(self):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()
        raise StopAsyncIteration


@pytest.mark.parametrize("budget_seconds", (0.2,))
def test_adapter_standalone_zero_timeout_uses_default_budget(
    monkeypatch: pytest.MonkeyPatch,
    budget_seconds: float,
) -> None:
    """A2：直接驱动的适配器在渠道 0 超时时以模块默认常量为总时限。"""

    monkeypatch.setattr(
        "core.adapters.direct.openai_chat_sse._DEFAULT_TOTAL_TIMEOUT_SECONDS",
        budget_seconds,
    )

    channel = _channel(0.0)
    adapter = OpenAIChatSSEAdapter(channel, client=_StalledClient())

    async def scenario() -> None:
        with pytest.raises(ProviderAdapterError, match="timed out"):
            async for _event in adapter.stream(
                _chat_request(),
                context=None,
                cancel_event=None,
            ):
                raise AssertionError("must not emit any event")

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_channel_positive_timeout_still_wins() -> None:
    """A：渠道显式超时保持优先，行为不变。"""

    channel = _channel(0.3)
    adapter = OpenAIChatSSEAdapter(channel, client=_StalledClient())

    async def scenario() -> None:
        with pytest.raises(ProviderAdapterError, match="timed out"):
            async for _event in adapter.stream(_chat_request()):
                raise AssertionError("must not emit any event")

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))


def test_audit_payload_truncates_long_messages_and_keeps_short_ones() -> None:
    """B：长消息入库以固定预算截断并标记，短消息保持原样。"""

    long_text = "密" * (_AUDIT_MESSAGE_TEXT_LIMIT + 500)
    short_text = "今天天气不错"
    request = _chat_request(
        (
            ChatMessage(role="user", content=long_text),
            ChatMessage(role="assistant", content=short_text),
        )
    )
    payload = _audit_request_payload(request)
    messages = payload["messages"]
    assert isinstance(messages, list)
    stored_long = str(messages[0]["content"])
    stored_short = str(messages[1]["content"])
    assert len(stored_long) <= _AUDIT_MESSAGE_TEXT_LIMIT + len(_AUDIT_TRUNCATED_MARK)
    assert stored_long.endswith(_AUDIT_TRUNCATED_MARK)
    assert _AUDIT_TRUNCATED_MARK not in stored_long[:-_AUDIT_MESSAGE_TEXT_LIMIT]
    assert stored_short == short_text


def test_audit_payload_bounds_multipart_messages() -> None:
    """B：多模态构件里字符串同样截断，短文本保持原样且结构不变。"""

    long_text = "多" * (_AUDIT_MESSAGE_TEXT_LIMIT + 200)
    short_text = "图说短"
    request = _chat_request(
        (
            ChatMessage(
                role="user",
                content=(
                    {"type": "text", "text": long_text},
                    {"type": "text", "text": short_text},
                ),
            ),
        )
    )
    payload = _audit_request_payload(request)
    parts = payload["messages"][0]["content"]
    assert isinstance(parts, list)
    assert isinstance(parts[0], dict) and parts[0]["type"] == "text"
    stored_long = str(parts[0]["text"])
    assert stored_long.endswith(_AUDIT_TRUNCATED_MARK)
    assert parts[1]["text"] == short_text


def test_sanitize_covers_plain_sk_and_google_and_session_keys() -> None:
    """C：裸露 sk-/sk-ant-、x-goog-api-key、sessionKey 值形均不含明文。"""

    samples = (
        "sk-proj-1234567890abcdef",
        "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ-abcdefgh",
        "x-goog-api-key=AIzaSyAbCdEfGhIjKlMnOp",
        'headers={"x-goog-api-key": "AIzaSyAbCdEfGhIjKlMnOp"}',
        "sessionKey=deadbeef0123456789",
    )
    for sample in samples:
        rendered = sanitize_log_text(sample)
        assert "AIzaSyAbCdEfGhIjKlMnOp" not in rendered
        assert "sk-proj-1234567890abcdef" not in rendered
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ-abcdefgh" not in rendered
        assert "deadbeef0123456789" not in rendered
        assert "[redacted]" in rendered


def test_scheduler_rejects_high_frequency_every_expression() -> None:
    """D：低于下限的 every 表达式被拒，daily 与合规间隔不受影响。"""

    scheduler = SchedulerService(action_runner=None)
    with pytest.raises(ScheduleExpressionError):
        scheduler.upsert(
            task_id="fast",
            name="fast",
            expression="every:0.1s",
            action={"identity": "pet:play_motion"},
            owner="p:s",
        )
    assert "fast" not in scheduler._tasks
    for raw in ("every:0.9s", "every:900ms"):
        with pytest.raises(ScheduleExpressionError):
            scheduler.upsert(
                task_id="below-floor",
                name="below",
                expression=raw,
                action={"identity": "pet:play_motion"},
                owner="p:s",
            )
        assert "below-floor" not in scheduler._tasks
    scheduler.upsert(
        task_id="daily",
        name="daily",
        expression="daily:08:00",
        action={"identity": "pet:play_motion"},
        owner="p:s",
    )
    scheduler.upsert(
        task_id="slow",
        name="slow",
        expression="every:5m",
        action={"identity": "pet:play_motion"},
        owner="p:s",
    )
    assert "daily" in scheduler._tasks and "slow" in scheduler._tasks
    assert _MIN_SCHEDULE_INTERVAL_SECONDS >= 1.0


def test_builtin_scheduler_upsert_returns_explicit_failure_for_fast_expression() -> None:
    """D：模型入口对高频表达式返回明确失败回执，合规任务照常创建。"""

    class Pet:
        def play_motion(self, name):
            del name
            return True

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    scheduler = SchedulerService(action_runner=None)
    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=Pet(),
        scheduler=scheduler,
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(bypass_approval=True),
    )

    async def scenario() -> None:
        fast = await executor.execute(
            call_id="round9-fast",
            identity="scheduler:upsert",
            arguments={
                "name": "fast",
                "expression": "every:0.1s",
                "action": {"identity": "pet:play_motion", "arguments": {"name": "wave"}},
            },
            context=ToolCallContext("p-round9", "s-round9", "t-round9"),
        )
        assert fast.status == "failed"
        assert "fast" not in scheduler._tasks

        slow = await executor.execute(
            call_id="round9-slow",
            identity="scheduler:upsert",
            arguments={
                "name": "slow",
                "expression": "every:5m",
                "action": {"identity": "pet:play_motion", "arguments": {"name": "wave"}},
            },
            context=ToolCallContext("p-round9", "s-round9", "t-round9"),
        )
        assert slow.status == "completed"

    asyncio.run(scenario())


def test_scheduler_last_error_hides_exception_text() -> None:
    """F：动作异常正文不进入状态接口，仅类名回传。"""

    secret = "sk-round9-abcdefghijklmnop"

    async def run(_action, _task):
        raise RuntimeError(f"boom {secret}")

    now = [100.0]
    scheduler = SchedulerService(action_runner=run, clock=lambda: now[0])
    scheduler.upsert(
        task_id="task-1",
        name="tick",
        expression="every:5m",
        action={"identity": "pet:play_motion"},
        owner="p:s",
    )
    now[0] = now[0] + scheduler._tasks["task-1"].next_run_at
    asyncio.run(scheduler.tick())
    last_error = str(scheduler.status()["last_error"])
    assert secret not in last_error
    assert last_error == "RuntimeError"
