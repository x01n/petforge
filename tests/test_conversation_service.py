from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator

import pytest

from core.adapters.direct import AdapterCancelled, ProviderAdapter, ToolCallDelta
from core.events.types import (
    ApprovalRequested,
    ConversationContext,
    MurmurDelta,
    ReasoningDelta,
    SentenceReady,
    TextDelta,
    ToolCallStarted,
    ToolStatusChanged,
    TurnFailed,
    TurnFinished,
)
from core.tts.contracts import EngineHealth, SpeechChunk, SpeechRequest
from db import ConversationRepository, Database
from services.affection import AffectionService
from services.conversation import ConversationService, PresentationService
from services.conversation.interaction_contract import InteractionPhase, InteractionState
from services.memory import MemoryService
from services.memory.summarizer import MemorySummaryCoordinator
from services.model_routing import ChannelConfig, ModelRouter
from services.tools import PermissionService, ToolExecutionService, ToolOutcome, ToolRegistry
from services.tools.types import RiskLevel, ToolCallContext, ToolSpec
from services.tts import TTSCoordinator
from services.tts.router import TTSProfile, TTSProfileRouter


class _DialogueAdapter(ProviderAdapter):
    provider = "fake"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming", "tools", "reasoning"})

    def __init__(self, *, with_tool: bool = False, tool_identity: str = "pet:ping") -> None:
        self.with_tool = with_tool
        self.tool_identity = tool_identity
        self.calls = 0
        self.requested_models: list[str] = []

    async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
        self.calls += 1
        self.requested_models.append(request.model)
        if self.with_tool and self.calls == 1:
            yield ToolCallDelta(context, 0, "call-1", self.tool_identity, '{"value":1}')
            yield TurnFinished(context, "tool_calls")
            return
        yield ReasoningDelta(context, "先检查一下")
        yield MurmurDelta(context, "嗯……")
        yield TextDelta(context, "你好。")
        yield TurnFinished(context)


class _FakeSpeechBackend:
    async def health(self) -> EngineHealth:
        return EngineHealth("fake", True)

    async def stream(self, request):
        yield SpeechChunk(request.request_id, b"pcm", 24000, 1)
        yield SpeechChunk(request.request_id, b"", 24000, 1, is_final=True)


class _BlockingAdapter(ProviderAdapter):
    provider = "blocking"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming"})

    async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
        del request, context, cancel_event
        await asyncio.Event().wait()
        if False:
            yield None


class _CaptureAdapter(_DialogueAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.messages = ()

    async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
        self.messages = request.messages
        async for event in super().stream(
            request,
            context=context,
            cancel_event=cancel_event,
        ):
            yield event


class _BurstAdapter(ProviderAdapter):
    provider = "burst"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming"})

    async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
        del request, cancel_event
        yield TextDelta(context, "首段。")
        for _ in range(5000):
            yield TextDelta(context, "x")
        yield TurnFinished(context)


class _ProbeTTS:
    def __init__(self, *, block: bool = False) -> None:
        self.block = block
        self.calls: list[tuple[ConversationContext, str, bool]] = []
        self.cancelled: list[ConversationContext] = []
        self.cleared: list[ConversationContext] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def enqueue_text(
        self,
        context: ConversationContext,
        text: str,
        *,
        language: str,
        mood: str,
        flush: bool,
    ) -> tuple[str, ...]:
        del language, mood
        self.calls.append((context, text, flush))
        self.started.set()
        if self.block:
            await self.release.wait()
        return (text,) if text else ()

    def cancel(self, context: ConversationContext) -> None:
        self.cancelled.append(context)

    def clear_cancelled(self, context: ConversationContext) -> None:
        self.cleared.append(context)


def _router(adapter: _DialogueAdapter) -> ModelRouter:
    channel = ChannelConfig("fake", base_url="https://fake.invalid/v1", model="demo")
    return ModelRouter([channel], adapter_factory=lambda _channel: adapter)


def test_conversation_stream_persists_memory_affection_and_starts_tts() -> None:
    database = Database(":memory:")
    memory = MemoryService(database)
    affection = AffectionService(database)
    repository = ConversationRepository(database)
    audio: list[object] = []
    tts = TTSCoordinator(_FakeSpeechBackend(), audio_sink=audio.append)
    service = ConversationService(
        _router(_DialogueAdapter()),
        memory=memory,
        affection=affection,
        repository=repository,
        tts=tts,
    )

    result = asyncio.run(service.complete("你好，今天过得怎么样？"))

    assert result.text == "你好。"
    assert result.reasoning == "先检查一下"
    assert result.murmur == "嗯……"
    assert affection.get() == 6
    assert affection.get_total_chats() == 1
    assert affection.get_today_chat_count() == 1
    assert memory.get_recent_chats(2)[-1]["role"] == "assistant"
    assert repository.list_recent("direct", "default", "local")[-1].status == "completed"
    assert audio and audio[0].data == b"pcm"


def test_conversation_promotes_explicit_user_fact_after_completed_turn() -> None:
    database = Database(":memory:")
    memory = MemoryService(database)
    service = ConversationService(_router(_DialogueAdapter()), memory=memory)

    result = asyncio.run(service.complete("我叫小林"))

    assert result.status == "completed"
    extracted = memory.list_memories(page_size=20)[0]
    assert len(extracted) == 1
    assert "小林" in memory.get_important_memories(1)[0]


def test_completed_turn_runs_local_extraction_before_single_model_queue_item() -> None:
    database = Database(":memory:")
    memory = MemoryService(database)
    coordinator = MemorySummaryCoordinator(memory, lambda: object())
    observed_local_ids: list[tuple[int, ...]] = []
    original_enqueue = coordinator.enqueue_extraction

    def observe_enqueue(text: str, source_id: int) -> bool:
        observed_local_ids.append(tuple(item.id for item in memory.list(memory_type="fact")))
        return original_enqueue(text, source_id)

    coordinator.enqueue_extraction = observe_enqueue  # type: ignore[method-assign]
    service = ConversationService(_router(_DialogueAdapter()), memory=memory)

    result = asyncio.run(service.complete("我叫小林"))

    facts = memory.list(memory_type="fact")
    assert result.status == "completed"
    assert observed_local_ids == [(facts[0].id,)]
    assert facts[0].source_ids == (1,)
    assert coordinator.status().extraction_pending == 1
    asyncio.run(coordinator.stop())
    database.close()


def test_conversation_memory_extraction_failure_does_not_fail_turn() -> None:
    class BrokenMemory:
        enabled = True

        def build_context_prompt(self, _query):
            return ""

        def add_chat(self, _role, _content):
            return 1

        def store_chat_exchange(self, _user, _answer):
            return None

        def increment_message_counter(self):
            return 1

        def extract_and_promote(self, _text, *, source_ids=()):
            del source_ids
            raise RuntimeError("simulated extraction failure")

    service = ConversationService(_router(_DialogueAdapter()), memory=BrokenMemory())
    result = asyncio.run(service.complete("我喜欢咖啡"))
    assert result.status == "completed"


def test_conversation_uses_async_memory_context_without_sync_builder() -> None:
    database = Database(":memory:")
    memory = MemoryService(database)
    adapter = _CaptureAdapter()

    async def async_builder(_query: str) -> str:
        await asyncio.sleep(0)
        return "异步语义上下文"

    memory.abuild_context_prompt = async_builder  # type: ignore[method-assign]
    memory.build_context_prompt = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("sync builder used"))
    )
    service = ConversationService(_router(adapter), memory=memory)
    try:
        result = asyncio.run(service.complete("测试异步记忆上下文"))
        assert result.status == "completed"
        assert any(
            message.role == "system" and "异步语义上下文" in str(message.content)
            for message in adapter.messages
        )
    finally:
        database.close()


def test_stale_async_memory_context_is_dropped_before_provider_call() -> None:
    database = Database(":memory:")
    memory = MemoryService(database)
    adapter = _CaptureAdapter()
    started = asyncio.Event()
    release = asyncio.Event()

    async def async_builder(_query: str) -> str:
        started.set()
        await release.wait()
        return "不应送入旧回合"

    memory.abuild_context_prompt = async_builder  # type: ignore[method-assign]
    service = ConversationService(_router(adapter), memory=memory)

    async def scenario() -> None:
        task = asyncio.create_task(_collect(service.stream("旧回合")))
        await started.wait()
        service.begin_context(turn_id="新回合")
        release.set()
        with pytest.raises(AdapterCancelled):
            await task

    try:
        asyncio.run(scenario())
        assert adapter.calls == 0
    finally:
        database.close()


def test_local_extraction_failure_still_queues_model_extraction(monkeypatch) -> None:
    database = Database(":memory:")
    memory = MemoryService(database)
    coordinator = MemorySummaryCoordinator(memory, lambda: object())

    def fail_local(*_args, **_kwargs):
        raise RuntimeError("local extraction internals")

    monkeypatch.setattr(memory, "extract_and_promote", fail_local)
    service = ConversationService(_router(_DialogueAdapter()), memory=memory)

    result = asyncio.run(service.complete("我从事软件开发"))

    assert result.status == "completed"
    assert coordinator.status().extraction_pending == 1
    asyncio.run(coordinator.stop())
    database.close()


def test_cancel_without_created_tts_worker_keeps_cancellation_tombstone() -> None:
    tts = _ProbeTTS()
    service = ConversationService(_router(_DialogueAdapter()), tts=tts)
    context = service.begin_context()

    service._cancel_tts_turn(context)

    assert tts.cancelled == [context]
    assert tts.cleared == []


def test_presentation_direct_speech_survives_tool_status_until_text_arrives() -> None:
    presentation = PresentationService()
    context = ConversationContext("p", "s", "presentation", 1)
    presentation.set_direct_speech("主动提示", mood="happy")
    presentation.consume(ToolStatusChanged(context, "completed", "工具已完成", "call-presentation"))
    assert presentation.snapshot.rendered_text == "主动提示"
    assert presentation.snapshot.rendered_mood == "happy"
    presentation.consume(TextDelta(context, "模型回答"))
    assert presentation.snapshot.rendered_text.startswith("模型回答")
    assert presentation.snapshot.rendered_mood == "neutral"


def test_presentation_direct_speech_clear_restores_underlying_snapshot() -> None:
    presentation = PresentationService()
    context = ConversationContext("p", "s", "presentation-clear", 1)
    presentation.consume(TextDelta(context, "当前回合"))
    presentation.set_direct_speech("点击反馈", mood="happy")
    presentation.consume(ToolStatusChanged(context, "running", "工具处理中", "call-clear"))

    assert presentation.snapshot.rendered_text == "点击反馈"
    restored = presentation.clear_direct_speech()
    assert restored.rendered_text.startswith("当前回合")
    assert "工具处理中" in restored.rendered_text


def test_missing_model_channel_emits_actionable_configuration_failure() -> None:
    service = ConversationService(ModelRouter())

    events = asyncio.run(_collect(service.stream("你好")))

    failures = [event for event in events if isinstance(event, TurnFailed)]
    assert failures
    assert failures[0].category == "configuration"
    assert failures[0].safe_message == "模型渠道未就绪，请点击“配置模型”检查连接信息"
    assert "llm.channels" not in failures[0].safe_message
    assert "MEAPET_API_BASE" not in failures[0].safe_message
    assert "[configuration]" in service.presentation.snapshot.tool_status


def test_configuration_failure_remains_visible_when_tool_status_is_hidden() -> None:
    presentation = PresentationService(show_tool_status=False)
    context = ConversationContext("default", "local", "turn-visible-error", 1)
    presentation.consume(TurnFailed(context, "configuration", "请先配置模型渠道", False))
    assert presentation.snapshot.rendered_text == "[configuration] 请先配置模型渠道"


def test_conversation_channel_override_uses_selected_channel_model() -> None:
    adapter = _DialogueAdapter()
    router = ModelRouter(
        [
            ChannelConfig("primary", base_url="https://primary.invalid/v1", model="model-a"),
            ChannelConfig("secondary", base_url="https://secondary.invalid/v1", model="model-b"),
        ],
        routes={"dialogue": {"channel": "primary"}},
        adapter_factory=lambda _channel: adapter,
    )
    service = ConversationService(router)

    result = asyncio.run(service.complete("切换渠道", channel_id="secondary"))

    assert result.status == "completed"
    assert adapter.requested_models == ["model-b"]


def test_conversation_channel_override_ignores_global_default_model() -> None:
    adapter = _DialogueAdapter()
    router = ModelRouter(
        [
            ChannelConfig("primary", base_url="https://primary.invalid/v1", model="model-a"),
            ChannelConfig("secondary", base_url="https://secondary.invalid/v1", model="model-b"),
        ],
        adapter_factory=lambda _channel: adapter,
    )
    service = ConversationService(router, model="model-a")

    result = asyncio.run(service.complete("使用备用频道", channel_id="secondary"))

    assert result.status == "completed"
    assert adapter.requested_models == ["model-b"]


def test_conversation_uses_runtime_selected_route_model_over_default() -> None:
    adapter = _DialogueAdapter()
    router = _router(adapter)
    router.select_model("demo")
    service = ConversationService(router, model="stale-default")

    result = asyncio.run(service.complete("使用已选模型"))

    assert result.status == "completed"
    assert adapter.requested_models == ["demo"]


def test_conversation_tool_loop_and_approval_preserve_call_id() -> None:
    async def ping(arguments, _context):
        return {"pong": arguments["value"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "pet:ping",
            "测试工具",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            ping,
            RiskLevel.LOW,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    service = ConversationService(_router(_DialogueAdapter(with_tool=True)), tools=executor)
    result = asyncio.run(service.complete("调用一下"))
    assert result.text == "你好。"
    assert result.tool_results and result.tool_results[0].content["pong"] == 1

    high_registry = ToolRegistry()
    high_registry.register(
        ToolSpec(
            "system:danger",
            "高风险工具",
            {"type": "object"},
            ping,
            RiskLevel.HIGH,
        )
    )
    high_executor = ToolExecutionService(high_registry, PermissionService())
    interaction = InteractionState(diagnostics={"ready": True, "channels": ({"id": "fake"},)})

    async def interaction_sink(event: object) -> None:
        interaction.consume(event)

    approval_service = ConversationService(
        _router(_DialogueAdapter(with_tool=True, tool_identity="system:danger")),
        tools=high_executor,
        event_sink=interaction_sink,
    )
    events = asyncio.run(_collect(approval_service.stream("需要确认")))
    approvals = [event for event in events if isinstance(event, ApprovalRequested)]
    assert approvals and approvals[0].call_id == "call-1"
    assert interaction.snapshot.phase is InteractionPhase.APPROVAL_REQUIRED
    assert interaction.snapshot.busy is True
    assert interaction.snapshot.approval is not None
    assert len(interaction.snapshot.actions) == 3
    assert approval_service.has_active_conversation() is True

    resumed = asyncio.run(approval_service.resume_approval(approvals[0].approval_id))
    assert resumed.status == "completed"
    assert interaction.snapshot.phase is InteractionPhase.COMPLETED
    assert interaction.snapshot.busy is False
    assert interaction.snapshot.approval is None
    assert approval_service.has_active_conversation() is False


def test_agent_plan_ocr_result_reference_click_and_approval_continuation() -> None:
    clicks: list[tuple[int, int]] = []

    async def ocr(_arguments, _context):
        return {
            "coordinate_space": "screen",
            "boxes": ({"text": "确定", "x": 320, "y": 180},),
        }

    async def click(arguments, _context):
        clicks.append((arguments["x"], arguments["y"]))
        return {"status": "completed"}

    class PlanAdapter(ProviderAdapter):
        provider = "fake"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming", "tools"})

        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        async def stream(
            self,
            request,
            *,
            context=None,
            cancel_event=None,
        ) -> AsyncIterator:
            del cancel_event
            self.calls += 1
            self.requests.append(request)
            if self.calls == 1:
                arguments = {
                    "version": 1,
                    "steps": [
                        {
                            "step_id": "ocr",
                            "identity": "desktop:ocr",
                            "arguments": {},
                        },
                        {
                            "step_id": "click",
                            "identity": "desktop:click_at",
                            "arguments": {
                                "x": {
                                    "$step_result": {
                                        "step_id": "ocr",
                                        "path": ["boxes", 0, "x"],
                                    }
                                },
                                "y": {
                                    "$step_result": {
                                        "step_id": "ocr",
                                        "path": ["boxes", 0, "y"],
                                    }
                                },
                            },
                            "depends_on": ["ocr"],
                        },
                    ],
                }
                yield ToolCallDelta(
                    context,
                    0,
                    "outer-plan-call",
                    "agent:execute_plan",
                    json.dumps(arguments),
                )
                yield TurnFinished(context, "tool_calls")
                return
            yield TextDelta(context, "已经点击。")
            yield TurnFinished(context)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "desktop:ocr",
            "识别屏幕文字",
            {"type": "object", "additionalProperties": False},
            ocr,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    registry.register(
        ToolSpec(
            "desktop:click_at",
            "点击屏幕坐标",
            {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
                "additionalProperties": False,
            },
            click,
            RiskLevel.HIGH,
        )
    )
    adapter = PlanAdapter()
    service = ConversationService(
        _router(adapter),
        tools=ToolExecutionService(
            registry,
            PermissionService(auto_allow_low_risk=True),
        ),
    )

    async def scenario() -> None:
        first = [event async for event in service.stream("识别确定按钮并点击")]
        approval = next(event for event in first if isinstance(event, ApprovalRequested))
        assert approval.call_id.startswith("agent-plan-")
        assert clicks == []
        resumed = [event async for event in service.resume_approval_stream(approval.approval_id)]
        assert any(
            isinstance(event, TextDelta) and event.delta == "已经点击。" for event in resumed
        )
        assert clicks == [(320, 180)]
        assert adapter.calls == 2
        first_request = adapter.requests[0]
        plan_definition = next(
            tool for tool in first_request.tools if tool.identity == "agent:execute_plan"
        )
        assert plan_definition.parameters["required"] == ["version", "steps"]
        second_messages = adapter.requests[1].messages
        tool_message = next(
            message
            for message in second_messages
            if message.role == "tool" and message.tool_call_id == "outer-plan-call"
        )
        assert isinstance(tool_message.content, str)
        assert '"step_id": "click"' in tool_message.content
        assert '"status": "completed"' in tool_message.content

    asyncio.run(scenario())


def test_agent_plan_multiple_approvals_resume_without_extra_model_turn() -> None:
    actions: list[str] = []

    async def action(arguments, _context):
        actions.append(arguments["name"])
        return {"ok": arguments["name"]}

    class MultiApprovalAdapter(ProviderAdapter):
        provider = "fake"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming", "tools"})

        def __init__(self) -> None:
            self.calls = 0

        async def stream(
            self,
            request,
            *,
            context=None,
            cancel_event=None,
        ) -> AsyncIterator:
            del request, cancel_event
            self.calls += 1
            if self.calls == 1:
                arguments = {
                    "version": 1,
                    "steps": [
                        {
                            "step_id": "first",
                            "identity": "system:action",
                            "arguments": {"name": "first"},
                        },
                        {
                            "step_id": "second",
                            "identity": "system:action",
                            "arguments": {"name": "second"},
                        },
                    ],
                }
                yield ToolCallDelta(
                    context,
                    0,
                    "multi-plan",
                    "agent:execute_plan",
                    json.dumps(arguments),
                )
                yield TurnFinished(context, "tool_calls")
                return
            yield TextDelta(context, "两步都完成。")
            yield TurnFinished(context)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:action",
            "动作",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
            action,
            RiskLevel.HIGH,
        )
    )
    adapter = MultiApprovalAdapter()
    interaction = InteractionState(diagnostics={"ready": True, "channels": ({"id": "fake"},)})

    async def interaction_sink(event: object) -> None:
        interaction.consume(event)

    service = ConversationService(
        _router(adapter),
        tools=ToolExecutionService(registry, PermissionService()),
        event_sink=interaction_sink,
    )

    async def scenario() -> None:
        first = [event async for event in service.stream("执行两步")]
        approval_one = next(event for event in first if isinstance(event, ApprovalRequested))
        resumed_one = [
            event async for event in service.resume_approval_stream(approval_one.approval_id)
        ]
        approval_two = next(event for event in resumed_one if isinstance(event, ApprovalRequested))
        assert not any(isinstance(event, TextDelta) for event in resumed_one)
        assert actions == ["first"]
        assert interaction.snapshot.phase is InteractionPhase.APPROVAL_REQUIRED
        assert interaction.snapshot.busy is True
        assert interaction.snapshot.approval is not None
        assert interaction.snapshot.approval.approval_id == approval_two.approval_id
        resumed_two = [
            event async for event in service.resume_approval_stream(approval_two.approval_id)
        ]
        assert any(
            isinstance(event, TextDelta) and event.delta == "两步都完成。" for event in resumed_two
        )
        assert actions == ["first", "second"]
        assert adapter.calls == 2

    asyncio.run(scenario())


def test_agent_plan_resume_can_be_cancelled_while_approved_step_is_running() -> None:
    """取消按钮在审批恢复的工具阶段也必须中断正在运行的步骤。"""

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocking_action(_arguments, _context):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    class BlockingPlanAdapter(ProviderAdapter):
        provider = "fake"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming", "tools"})

        def __init__(self) -> None:
            self.calls = 0

        async def stream(
            self,
            request,
            *,
            context=None,
            cancel_event=None,
        ) -> AsyncIterator:
            del request, cancel_event
            self.calls += 1
            if self.calls == 1:
                yield ToolCallDelta(
                    context,
                    0,
                    "blocking-plan",
                    "agent:execute_plan",
                    json.dumps(
                        {
                            "version": 1,
                            "steps": [
                                {
                                    "step_id": "action",
                                    "identity": "system:blocking_action",
                                }
                            ],
                        }
                    ),
                )
                yield TurnFinished(context, "tool_calls")
                return
            yield TextDelta(context, "完成。")
            yield TurnFinished(context)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:blocking_action",
            "阻塞动作",
            {"type": "object", "additionalProperties": False},
            blocking_action,
            RiskLevel.HIGH,
        )
    )
    service = ConversationService(
        _router(BlockingPlanAdapter()),
        tools=ToolExecutionService(registry, PermissionService()),
    )

    async def scenario() -> None:
        first = [event async for event in service.stream("执行阻塞动作")]
        approval = next(event for event in first if isinstance(event, ApprovalRequested))

        async def consume_resume() -> list[object]:
            return [event async for event in service.resume_approval_stream(approval.approval_id)]

        resume_task = asyncio.create_task(consume_resume())
        await started.wait()
        service.cancel(approval.context)
        with pytest.raises(asyncio.CancelledError):
            await resume_task
        assert cancelled.is_set()
        assert service.pending_approvals_for_ui() == ()

    asyncio.run(scenario())


def test_conversation_resume_approval_rejects_stale_generation() -> None:
    calls: list[dict] = []

    async def danger(arguments, _context):
        calls.append(dict(arguments))
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "高风险工具",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            danger,
            RiskLevel.HIGH,
        )
    )
    events: list[object] = []

    async def sink(event):
        events.append(event)

    executor = ToolExecutionService(registry, PermissionService())
    service = ConversationService(
        _router(_DialogueAdapter(with_tool=True, tool_identity="system:danger")),
        tools=executor,
        event_sink=sink,
    )

    async def scenario():
        stream_events = [event async for event in service.stream("需要确认")]
        approval = next(event for event in stream_events if isinstance(event, ApprovalRequested))
        service.begin_context(
            profile_id=approval.context.profile_id,
            session_id=approval.context.session_id,
            mode=approval.context.mode,
        )
        resumed = await service.resume_approval(
            approval.approval_id,
            context=approval.context,
        )
        assert resumed.status == "denied"
        assert resumed.content["error"] == "approval context is stale"
        assert not calls
        assert not any(
            isinstance(event, ToolStatusChanged) and event.state == "completed" for event in events
        )

    asyncio.run(scenario())


def test_conversation_resume_approval_executes_original_call() -> None:
    calls: list[tuple[dict, ToolCallContext]] = []

    async def danger(arguments, context):
        calls.append((dict(arguments), context))
        return {"ok": arguments["value"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "高风险工具",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            danger,
            RiskLevel.HIGH,
        )
    )
    events: list[object] = []

    async def sink(event):
        events.append(event)

    adapter = _DialogueAdapter(with_tool=True, tool_identity="system:danger")
    executor = ToolExecutionService(registry, PermissionService())
    service = ConversationService(
        _router(adapter),
        tools=executor,
        event_sink=sink,
    )

    async def scenario():
        stream_events = [event async for event in service.stream("需要确认")]
        approval = next(event for event in stream_events if isinstance(event, ApprovalRequested))
        resumed = await service.resume_approval(approval.approval_id)
        assert resumed.status == "completed"
        assert resumed.content == {"ok": 1}
        assert len(calls) == 1
        assert calls[0][0] == {"value": 1}
        assert calls[0][1].turn_id == approval.context.turn_id
        # 审批恢复只消费并执行原始快照；下一轮模型请求必须由上层显式编排。
        assert adapter.calls == 1
        assert any(
            isinstance(event, ToolStatusChanged)
            and event.call_id == approval.call_id
            and event.state == "completed"
            for event in events
        )

    asyncio.run(scenario())


def test_conversation_approval_continuation_returns_to_same_model_turn() -> None:
    async def danger(arguments, _context):
        return {"ok": arguments["value"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "高风险工具",
            {"type": "object", "properties": {"value": {"type": "integer"}}},
            danger,
            RiskLevel.HIGH,
        )
    )
    adapter = _DialogueAdapter(with_tool=True, tool_identity="system:danger")

    class _Affection:
        chats = 0
        changes = 0

        def mark_today_chatted(self, *, increment_total: bool = True):
            del increment_total
            self.chats += 1

        def adjust(self, _delta: int):
            self.changes += 1

    affection = _Affection()
    service = ConversationService(
        _router(adapter),
        tools=ToolExecutionService(registry, PermissionService()),
        affection=affection,
    )

    async def scenario() -> None:
        first = [event async for event in service.stream("需要确认")]
        approval = next(event for event in first if isinstance(event, ApprovalRequested))
        assert affection.chats == 0
        assert affection.changes == 0
        resumed = [event async for event in service.resume_approval_stream(approval.approval_id)]
        assert adapter.calls == 2
        assert any(isinstance(event, TextDelta) and event.delta == "你好。" for event in resumed)
        assert any(isinstance(event, TurnFinished) for event in resumed)
        assert service.pending_approvals_for_ui() == ()
        assert service.presentation.snapshot.text == "你好。"
        assert affection.chats == 1
        assert affection.changes == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("initial_role", ("", "assistant"))
def test_approval_continuation_keeps_tts_profile_snapshot(initial_role: str) -> None:
    class Adapter(ProviderAdapter):
        provider = "approval-tts"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming", "tools"})

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request, *, context=None, cancel_event=None):
            del request, cancel_event
            self.calls += 1
            if self.calls == 1:
                yield ToolCallDelta(context, 0, "approval-tts-call", "system:danger", "{}")
                yield TurnFinished(context, "tool_calls")
                return
            yield TextDelta(context, "审批后继续回答。")
            yield TurnFinished(context)

    async def danger(_arguments, _context):
        return {"ok": True}

    class Backend:
        def __init__(self, name: str) -> None:
            self.name = name
            self.requests: list[SpeechRequest] = []

        async def health(self) -> EngineHealth:
            return EngineHealth(self.name, True)

        async def stream(self, request: SpeechRequest):
            self.requests.append(request)
            yield SpeechChunk(request.request_id, self.name.encode(), 24000, 1, is_final=True)

    registry = ToolRegistry()
    registry.register(
        ToolSpec("system:danger", "危险操作", {"type": "object"}, danger, RiskLevel.HIGH)
    )
    first = Backend("first")
    second = Backend("second")
    profile_router = TTSProfileRouter(
        {
            "first": TTSProfile("first", first, languages=frozenset({"zh"})),
            "second": TTSProfile("second", second, languages=frozenset({"zh", "jp"})),
        },
        default_profile="first",
        language_profiles={"ja": "second"},
        role_profiles={"assistant": "first", "narrator": "second"},
    )
    service = ConversationService(
        _router(Adapter()),
        tools=ToolExecutionService(registry, PermissionService()),
        tts=TTSCoordinator(profile_router),
        tts_language="zh",
        tts_role=initial_role,
    )

    async def scenario() -> None:
        initial = [event async for event in service.stream("需要确认")]
        approval = next(event for event in initial if isinstance(event, ApprovalRequested))
        profile_router.select("second", language="ja")
        service.tts_language = "ja"
        service.tts_role = "narrator"
        resumed = [event async for event in service.resume_approval_stream(approval.approval_id)]
        assert any(isinstance(event, TextDelta) for event in resumed)
        assert not service._tts_role_snapshots

    asyncio.run(scenario())
    assert [(request.text, request.language, request.profile_id) for request in first.requests] == [
        ("审批后继续回答。", "zh", "first")
    ]
    assert [request.role for request in first.requests] == [initial_role]
    assert second.requests == []


def test_denied_approval_releases_tts_context_snapshot() -> None:
    async def danger(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec("system:danger", "危险操作", {"type": "object"}, danger, RiskLevel.HIGH)
    )
    coordinator = TTSCoordinator(_FakeSpeechBackend())
    service = ConversationService(
        _router(_DialogueAdapter(with_tool=True, tool_identity="system:danger")),
        tools=ToolExecutionService(registry, PermissionService()),
        tts=coordinator,
    )

    # 先构造一个真正需要审批的快照，再调用 deny_approval。
    async def denial_scenario() -> None:
        initial = [event async for event in service.stream("拒绝操作")]
        approval = next(event for event in initial if isinstance(event, ApprovalRequested))
        assert service.deny_approval(approval.approval_id) is True
        await asyncio.sleep(0)
        assert not service._tts_turns
        assert not service._tts_tasks

    asyncio.run(denial_scenario())
    assert not service._tts_language_snapshots
    assert not service._tts_role_snapshots
    assert not coordinator._pinned_contexts
    assert not coordinator._context_routes


def test_concurrent_approval_resume_starts_only_one_model_continuation() -> None:
    class _SlowContinuationAdapter(ProviderAdapter):
        provider = "slow-continuation"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming", "tools"})

        def __init__(self) -> None:
            self.calls = 0
            self.continuation_started = asyncio.Event()
            self.release = asyncio.Event()

        async def stream(self, request, *, context=None, cancel_event=None):
            del request, cancel_event
            self.calls += 1
            if self.calls == 1:
                yield ToolCallDelta(context, 0, "approval-call", "system:danger", "{}")
                yield TurnFinished(context, "tool_calls")
                return
            self.continuation_started.set()
            await self.release.wait()
            yield TextDelta(context, "续接完成。")
            yield TurnFinished(context)

    async def danger(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec("system:danger", "危险操作", {"type": "object"}, danger, RiskLevel.HIGH)
    )
    adapter = _SlowContinuationAdapter()
    service = ConversationService(
        _router(adapter),
        tools=ToolExecutionService(registry, PermissionService()),
    )

    async def collect(approval_id: str) -> list[object]:
        return [event async for event in service.resume_approval_stream(approval_id)]

    async def scenario() -> None:
        initial = [event async for event in service.stream("需要确认")]
        approval = next(event for event in initial if isinstance(event, ApprovalRequested))

        first = asyncio.create_task(collect(approval.approval_id))
        await asyncio.wait_for(adapter.continuation_started.wait(), timeout=1.0)
        second = asyncio.create_task(collect(approval.approval_id))
        await asyncio.sleep(0)
        adapter.release.set()
        first_events, second_events = await asyncio.gather(first, second)

        assert adapter.calls == 2
        assert any(isinstance(event, TextDelta) for event in first_events)
        assert any(isinstance(event, ToolStatusChanged) for event in second_events)

    asyncio.run(scenario())


def test_approval_continuation_keeps_all_tool_call_messages_consistent() -> None:
    calls: list[str] = []

    async def first_tool(_arguments, _context):
        calls.append("first")
        return {"ok": "first"}

    async def second_tool(_arguments, _context):
        calls.append("second")
        return {"ok": "second"}

    class _MultiToolAdapter(ProviderAdapter):
        provider = "multi"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming", "tools"})

        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        async def stream(self, request, *, context=None, cancel_event=None):
            del cancel_event
            self.calls += 1
            self.requests.append(request)
            if self.calls == 1:
                yield ToolCallDelta(context, 0, "first-call", "system:first", "{}")
                yield ToolCallDelta(context, 1, "second-call", "system:second", "{}")
                yield TurnFinished(context, "tool_calls")
                return
            yield TextDelta(context, "两个调用的结果已整理。")
            yield TurnFinished(context)

    registry = ToolRegistry()
    registry.register(
        ToolSpec("system:first", "第一个操作", {"type": "object"}, first_tool, RiskLevel.HIGH)
    )
    registry.register(
        ToolSpec("system:second", "第二个操作", {"type": "object"}, second_tool, RiskLevel.HIGH)
    )
    adapter = _MultiToolAdapter()
    service = ConversationService(
        _router(adapter),
        tools=ToolExecutionService(registry, PermissionService()),
    )

    async def scenario() -> None:
        initial = [event async for event in service.stream("需要两个操作")]
        approval = next(event for event in initial if isinstance(event, ApprovalRequested))
        resumed = [event async for event in service.resume_approval_stream(approval.approval_id)]
        assert adapter.calls == 2
        assert calls == ["first"]
        messages = adapter.requests[1].messages
        tool_messages = [message for message in messages if message.role == "tool"]
        assert [message.tool_call_id for message in tool_messages] == ["first-call", "second-call"]
        assert any(isinstance(event, TextDelta) and "结果" in event.delta for event in resumed)

    asyncio.run(scenario())


def test_independent_low_risk_read_tools_run_in_parallel_and_replay_in_order() -> None:
    active = 0
    maximum_active = 0
    started = 0
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def observe(_arguments, _context):
        nonlocal active, maximum_active, started
        active += 1
        maximum_active = max(maximum_active, active)
        started += 1
        if started == 2:
            both_started.set()
        await release.wait()
        active -= 1
        return {"status": "available", "value": started}

    class _ParallelAdapter(ProviderAdapter):
        provider = "parallel"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming", "tools"})

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request, *, context=None, cancel_event=None):
            del request, cancel_event
            self.calls += 1
            if self.calls == 1:
                yield ToolCallDelta(context, 0, "observe-a", "desktop:a", "{}")
                yield ToolCallDelta(context, 1, "observe-b", "desktop:b", "{}")
                yield TurnFinished(context, "tool_calls")
                return
            yield TextDelta(context, "已完成两项观察。")
            yield TurnFinished(context)

    registry = ToolRegistry()
    for identity in ("desktop:a", "desktop:b"):
        registry.register(
            ToolSpec(
                identity,
                identity,
                {"type": "object"},
                observe,
                RiskLevel.LOW,
                read_only=True,
            )
        )
    adapter = _ParallelAdapter()
    service = ConversationService(
        _router(adapter),
        tools=ToolExecutionService(registry, PermissionService(auto_allow_low_risk=True)),
    )

    async def scenario() -> list[object]:
        task = asyncio.create_task(_collect(service.stream("同时观察")))
        await asyncio.wait_for(both_started.wait(), timeout=1.0)
        release.set()
        return await task

    events = asyncio.run(scenario())

    assert maximum_active == 2
    started_events = [event.call_id for event in events if isinstance(event, ToolCallStarted)]
    assert started_events[:2] == ["observe-a", "observe-b"]
    assert any(isinstance(event, TextDelta) and "两项" in event.delta for event in events)


def test_tool_status_uses_user_label_instead_of_internal_identity() -> None:
    async def danger(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "高风险工具",
            {"type": "object"},
            danger,
            RiskLevel.HIGH,
            display_name="运行安全操作",
        )
    )
    service = ConversationService(
        _router(_DialogueAdapter(with_tool=True, tool_identity="system:danger")),
        tools=ToolExecutionService(registry, PermissionService()),
    )

    events = asyncio.run(_collect(service.stream("需要确认")))
    approval = next(event for event in events if isinstance(event, ApprovalRequested))
    assert approval.safe_summary == "需要确认：运行安全操作"
    assert "system:danger" not in service.presentation.snapshot.rendered_text


def test_conversation_stream_does_not_expose_provider_tool_argument_deltas() -> None:
    async def danger(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "高风险工具",
            {"type": "object", "properties": {"value": {"type": "integer"}}},
            danger,
            RiskLevel.HIGH,
        )
    )
    service = ConversationService(
        _router(_DialogueAdapter(with_tool=True, tool_identity="system:danger")),
        tools=ToolExecutionService(registry, PermissionService()),
    )

    events = asyncio.run(_collect(service.stream("需要确认")))
    assert not any(isinstance(event, ToolCallDelta) for event in events)
    serialized = " ".join(str(event) for event in events)
    assert '{"value":1}' not in serialized


def test_tool_failure_summary_hides_sensitive_error_text() -> None:
    service = ConversationService(_router(_DialogueAdapter()))
    outcome = service._tool_outcome_summary(
        ToolOutcome(
            "failed",
            "system:danger",
            "call-secret",
            {"error": "authorization=secret-value"},
        )
    )
    assert outcome == "操作失败（敏感详情已隐藏）"


def test_deny_approval_updates_presentation_without_running_event_loop() -> None:
    async def danger(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "高风险工具",
            {"type": "object"},
            danger,
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    service = ConversationService(
        _router(_DialogueAdapter(with_tool=True, tool_identity="system:danger")),
        tools=executor,
    )

    events = asyncio.run(_collect(service.stream("拒绝这个调用")))
    approval = next(event for event in events if isinstance(event, ApprovalRequested))

    assert service.deny_approval(approval.approval_id) is True
    assert "denied" in service.presentation.snapshot.tool_status


def test_capture_tool_result_becomes_inline_vision_content() -> None:
    converted = ConversationService._tool_message_content(
        {"status": "available", "format": "png", "data": b"small"}
    )
    assert isinstance(converted, tuple)
    assert converted[1]["type"] == "image"
    oversized = ConversationService._tool_message_content(
        {"status": "available", "format": "png", "data": b"x" * (5 * 1024 * 1024 + 1)}
    )
    assert isinstance(oversized, str)


def test_cancel_stops_blocked_provider_task_immediately() -> None:
    service = ConversationService(_router(_BlockingAdapter()))

    async def scenario():
        context = service.begin_context()
        task = asyncio.create_task(service.complete("等待取消", context=context))
        await asyncio.sleep(0)
        service.cancel(context)
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled conversation completed unexpectedly")
        return context

    context = asyncio.run(scenario())
    assert not service._active_task_by_key
    assert not service.generation_gate.accepts(context)


def test_tts_deltas_share_one_bounded_worker_and_keep_first_segment() -> None:
    async def scenario() -> None:
        tts = _ProbeTTS(block=True)
        service = ConversationService(_router(_BurstAdapter()), tts=tts)
        context = service.begin_context(turn_id="burst")
        task = asyncio.create_task(service.complete("开始输出", context=context))
        await tts.started.wait()

        state = service._tts_turns[context]
        assert len(service._tts_tasks) == 1
        assert len(state.pending_text) <= service._TTS_PENDING_CHAR_LIMIT
        assert tts.calls[0][0] == context
        assert tts.calls[0][1].startswith("首段。")

        service.cancel(context)
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled conversation completed unexpectedly")
        await asyncio.sleep(0)
        assert not service._tts_turns
        assert not service._tts_tasks
        assert context in tts.cancelled

    asyncio.run(scenario())


def test_wait_flushed_tts_propagates_outer_cancel_without_cancelling_worker() -> None:
    async def scenario() -> None:
        tts = _ProbeTTS(block=True)
        service = ConversationService(_router(_DialogueAdapter()), tts=tts)
        context = service.begin_context(turn_id="flush-outer-cancel")
        service._schedule_tts(context, "待冲刷。", flush=True)
        await tts.started.wait()

        state = service._tts_turns[context]
        worker = state.task
        assert worker is not None
        waiter = asyncio.create_task(service._await_flushed_tts_turn(context))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        # 外层取消由调用方继续处理，shield 保证 flush worker 不被连带取消。
        assert not worker.cancelled()
        assert service._tts_turns.get(context) is state

        tts.release.set()
        await asyncio.wait_for(worker, timeout=1.0)
        await asyncio.sleep(0)
        assert context not in service._tts_turns

    asyncio.run(scenario())


def test_wait_flushed_tts_swallows_only_worker_cancel() -> None:
    async def scenario() -> None:
        tts = _ProbeTTS(block=True)
        service = ConversationService(_router(_DialogueAdapter()), tts=tts)
        context = service.begin_context(turn_id="flush-worker-cancel")
        service._schedule_tts(context, "取消冲刷。", flush=True)
        await tts.started.wait()

        state = service._tts_turns[context]
        worker = state.task
        assert worker is not None
        worker.cancel()

        # worker 自身取消属于预期的旧回合收尾，不应伪装成外层回合取消。
        await service._await_flushed_tts_turn(context)
        assert worker.cancelled()
        assert context not in service._tts_turns

    asyncio.run(scenario())


def test_tts_backlog_spills_in_order_without_dropping_model_text() -> None:
    async def scenario() -> None:
        source = "首段。" + ("x" * 5000) + "尾段。"

        class StreamingAdapter(ProviderAdapter):
            provider = "streaming"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            def __init__(self) -> None:
                self.finished = asyncio.Event()

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "首段。")
                for _ in range(5000):
                    yield TextDelta(context, "x")
                yield TextDelta(context, "尾段。")
                self.finished.set()
                yield TurnFinished(context)

        adapter = StreamingAdapter()
        tts = _ProbeTTS(block=True)
        service = ConversationService(_router(adapter), tts=tts)
        context = service.begin_context(turn_id="tts-backlog-preserved")
        task = asyncio.create_task(service.complete("保留完整语音", context=context))

        await asyncio.wait_for(tts.started.wait(), timeout=1.0)
        await asyncio.wait_for(adapter.finished.wait(), timeout=1.0)
        state = service._tts_turns[context]
        assert state.pending_chunks

        tts.release.set()
        result = await asyncio.wait_for(task, timeout=2.0)
        spoken = "".join(text for _context, text, flush in tts.calls if not flush)
        assert spoken == source
        assert result.text == source

    asyncio.run(scenario())


def test_tts_backpressure_pauses_stream_and_cancel_wakes_waiter() -> None:
    async def scenario() -> None:
        class StreamingAdapter(ProviderAdapter):
            provider = "backpressure"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            def __init__(self) -> None:
                self.finished = asyncio.Event()

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "x" * 4096)
                self.finished.set()
                yield TurnFinished(context)

        adapter = StreamingAdapter()
        tts = _ProbeTTS(block=True)
        service = ConversationService(_router(adapter), tts=tts)
        # 使用较小的回合上限验证模型流会在 worker 堵塞时暂停，而不是
        # 继续无限追加；生产环境仍使用默认的有界上限。
        service._TTS_BACKLOG_CHAR_LIMIT = 128
        context = service.begin_context(turn_id="tts-backpressure")
        task = asyncio.create_task(service.complete("限流", context=context))

        await asyncio.wait_for(tts.started.wait(), timeout=1.0)
        await asyncio.sleep(0)
        assert not adapter.finished.is_set()

        service.cancel(context)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        await asyncio.sleep(0)
        assert not service._tts_turns

    asyncio.run(scenario())


@pytest.mark.parametrize("delta_size", [73, 4096])
def test_tts_backpressure_preserves_unicode_text_and_capacity_on_resume(delta_size: int) -> None:
    async def scenario() -> None:
        source = ("第一句。第二句🙂。\n" * 400) + "末尾完整保留。"
        backpressured = asyncio.Event()

        class ObservedService(ConversationService):
            async def _await_tts_backpressure(self, context):
                backpressured.set()
                await super()._await_tts_backpressure(context)

        class Adapter(_DialogueAdapter):
            async def stream(self, request, *, context=None, cancel_event=None):
                for start in range(0, len(source), delta_size):
                    yield TextDelta(context, source[start : start + delta_size])
                yield TurnFinished(context)

        tts = _ProbeTTS(block=True)
        service = ObservedService(_router(Adapter()), tts=tts)
        service._TTS_BACKLOG_CHAR_LIMIT = 512
        context = service.begin_context(turn_id="bounded-unicode")
        task = asyncio.create_task(service.complete("开始", context=context))
        try:
            await asyncio.wait_for(tts.started.wait(), timeout=1.0)
            await asyncio.wait_for(backpressured.wait(), timeout=1.0)
            state = service._tts_turns[context]
            queued = len(state.pending_text) + sum(map(len, state.pending_chunks))
            assert 0 < queued == state.backlog_chars <= service._TTS_BACKLOG_CHAR_LIMIT
            assert not task.done()
            tts.release.set()
            result = await asyncio.wait_for(task, timeout=2.0)
            assert result.text == source
            assert "".join(text for _, text, flush in tts.calls if not flush) == source
            assert state.backlog_chars == 0
            assert not service._tts_tasks
        finally:
            await service.aclose()

    asyncio.run(scenario())


def test_tts_worker_cancellation_terminates_backpressured_conversation() -> None:
    """语音消费者退出后，模型不能继续等待永远不会释放的队列容量。"""

    async def scenario() -> None:
        class Adapter(_DialogueAdapter):
            async def stream(self, request, *, context=None, cancel_event=None):
                yield TextDelta(context, "长段。" * 2048)
                yield TurnFinished(context)

        tts = _ProbeTTS(block=True)
        service = ConversationService(_router(Adapter()), tts=tts)
        service._TTS_BACKLOG_CHAR_LIMIT = 512
        context = service.begin_context(turn_id="worker-exit-backpressure")
        task = asyncio.create_task(service.complete("开始", context=context))
        try:
            await asyncio.wait_for(tts.started.wait(), timeout=1.0)
            await asyncio.sleep(0)
            state = service._tts_turns[context]
            assert state.backlog_chars == service._TTS_BACKLOG_CHAR_LIMIT
            assert state.task is not None
            state.task.cancel()
            done, _ = await asyncio.wait({task}, timeout=1.0)
            assert task in done, "conversation still waits on a stopped TTS worker"
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not service._tts_turns
            assert not service._tts_tasks
            assert context in tts.cancelled
        finally:
            await service.aclose()

    asyncio.run(scenario())


def test_model_stream_continues_after_first_tts_segment_is_blocked() -> None:
    async def scenario() -> None:
        class StreamingAdapter(ProviderAdapter):
            provider = "streaming"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            def __init__(self) -> None:
                self.finished = asyncio.Event()

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "首句。")
                yield TextDelta(context, "后句。")
                self.finished.set()
                yield TurnFinished(context)

        adapter = StreamingAdapter()
        tts = _ProbeTTS(block=True)
        service = ConversationService(_router(adapter), tts=tts)
        context = service.begin_context(turn_id="tts-does-not-block-model")
        task = asyncio.create_task(service.complete("继续", context=context))

        await asyncio.wait_for(tts.started.wait(), timeout=1.0)
        await asyncio.wait_for(adapter.finished.wait(), timeout=1.0)
        assert tts.calls[0][1].startswith("首句。")
        assert task.done() is False

        service.cancel(context)
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled conversation completed unexpectedly")

    asyncio.run(scenario())


def test_stream_to_real_tts_coordinator_starts_first_segment_without_waiting() -> None:
    async def scenario() -> None:
        class StreamingAdapter(ProviderAdapter):
            provider = "streaming"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            def __init__(self) -> None:
                self.finished = asyncio.Event()

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "首句。")
                yield TextDelta(context, "后句。")
                self.finished.set()
                yield TurnFinished(context)

        class BlockingSpeechBackend:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def health(self) -> EngineHealth:
                return EngineHealth("blocking", True)

            async def stream(self, request):
                self.started.set()
                await asyncio.Event().wait()
                yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

        adapter = StreamingAdapter()
        speech_backend = BlockingSpeechBackend()
        tts = TTSCoordinator(speech_backend)
        service = ConversationService(_router(adapter), tts=tts)
        context = service.begin_context(turn_id="coordinator-does-not-block-model")
        task = asyncio.create_task(service.complete("继续", context=context))

        await asyncio.wait_for(speech_backend.started.wait(), timeout=1.0)
        await asyncio.wait_for(adapter.finished.wait(), timeout=1.0)
        assert task.done() is False

        service.cancel(context)
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled conversation completed unexpectedly")

    asyncio.run(scenario())


def test_first_text_delta_yields_only_after_tts_backend_starts() -> None:
    async def scenario() -> None:
        class Adapter(ProviderAdapter):
            provider = "streaming"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "首句。")
                yield TurnFinished(context)

        class Backend:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def health(self) -> EngineHealth:
                return EngineHealth("first-segment", True)

            async def stream(self, request):
                self.started.set()
                await asyncio.Event().wait()
                yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

        backend = Backend()
        service = ConversationService(
            ModelRouter(
                [ChannelConfig("streaming", base_url="https://streaming.invalid/v1", model="demo")],
                adapter_factory=lambda _channel: Adapter(),
            ),
            tts=TTSCoordinator(backend),
        )
        stream = service.stream("开始")

        event = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert isinstance(event, TextDelta)
        assert backend.started.is_set()

        await stream.aclose()
        await service.aclose()

    asyncio.run(scenario())


def test_first_unpunctuated_text_delta_is_submitted_to_tts_before_next_model_event() -> None:
    """半句也要立即进入 TTS worker，后续句末再由分段器决定合成边界。"""

    async def scenario() -> None:
        class StreamingAdapter(ProviderAdapter):
            provider = "first-delta-submit"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            def __init__(self) -> None:
                self.release = asyncio.Event()

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "首个半句")
                await self.release.wait()
                yield TextDelta(context, "完成。")
                yield TurnFinished(context)

        adapter = StreamingAdapter()
        tts = _ProbeTTS()
        service = ConversationService(_router(adapter), tts=tts)
        stream = service.stream("开始")
        try:
            first = await asyncio.wait_for(anext(stream), timeout=1.0)
            assert isinstance(first, TextDelta)
            await asyncio.wait_for(tts.started.wait(), timeout=1.0)
            assert tts.calls[0][1] == "首个半句"
            adapter.release.set()
            remaining = [event async for event in stream]
            assert any(isinstance(event, TurnFinished) for event in remaining)
        finally:
            await service.aclose()

    asyncio.run(scenario())


def test_sentence_tts_dispatch_does_not_block_following_model_delta() -> None:
    """首句进入合成后，后续模型文本仍能继续流式展示。"""

    async def scenario() -> None:
        class Adapter(ProviderAdapter):
            provider = "sentence-streaming"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "第一句。")
                yield TextDelta(context, "第二句。")
                yield TurnFinished(context)

        class Backend:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()
                self.requests: list[str] = []

            async def health(self) -> EngineHealth:
                return EngineHealth("sentence-streaming", True)

            async def stream(self, request):
                self.requests.append(request.text)
                if request.text == "第一句。":
                    self.started.set()
                    await self.release.wait()
                yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

        backend = Backend()
        service = ConversationService(
            ModelRouter(
                [
                    ChannelConfig(
                        "sentence-streaming",
                        base_url="https://sentence-streaming.invalid/v1",
                        model="demo",
                    )
                ],
                adapter_factory=lambda _channel: Adapter(),
            ),
            tts=TTSCoordinator(backend),
        )
        stream = service.stream("继续说")

        first = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert isinstance(first, TextDelta)
        await asyncio.wait_for(backend.started.wait(), timeout=1.0)

        # 首个 TTS 请求故意阻塞；第二个模型增量仍必须在短时间内返回，
        # 不能因为音频合成或播放端背压而停住模型 SSE。
        second = await asyncio.wait_for(anext(stream), timeout=1.0)
        assert isinstance(second, TextDelta)
        assert second.delta == "第二句。"
        assert backend.requests == ["第一句。"]

        backend.release.set()
        remaining = [event async for event in stream]
        assert any(isinstance(event, TurnFinished) for event in remaining)
        assert backend.requests == ["第一句。", "第二句。"]
        await service.aclose()

    asyncio.run(scenario())


def test_tts_request_starts_before_a_slow_event_sink() -> None:
    """展示层背压不能延迟模型首句的 TTS 请求。"""

    async def scenario() -> None:
        sink_entered = asyncio.Event()
        sink_release = asyncio.Event()

        class Adapter(ProviderAdapter):
            provider = "streaming"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "首句。")
                yield TurnFinished(context)

        class Backend:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def health(self) -> EngineHealth:
                return EngineHealth("slow-sink", True)

            async def stream(self, request):
                self.started.set()
                yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

        async def slow_sink(event: object) -> None:
            if isinstance(event, TextDelta):
                sink_entered.set()
                await sink_release.wait()

        backend = Backend()
        router = ModelRouter(
            [ChannelConfig("streaming", base_url="https://streaming.invalid/v1", model="demo")],
            adapter_factory=lambda _channel: Adapter(),
        )
        service = ConversationService(
            router,
            tts=TTSCoordinator(backend),
            event_sink=slow_sink,
        )
        stream = service.stream("开始")
        first_event = asyncio.create_task(anext(stream))
        await asyncio.wait_for(sink_entered.wait(), timeout=1.0)
        # 如果 TTS 调度在 event_sink 之后，下面会一直等到 sink_release；
        # 这正是此前首句语音被 UI 背压拖延的回归。
        await asyncio.wait_for(backend.started.wait(), timeout=1.0)
        sink_release.set()
        assert isinstance(await asyncio.wait_for(first_event, timeout=1.0), TextDelta)
        await stream.aclose()
        await service.aclose()

    asyncio.run(scenario())


def test_unpunctuated_stream_yields_to_tts_at_segment_limit() -> None:
    async def scenario() -> None:
        class StreamingAdapter(ProviderAdapter):
            provider = "unpunctuated"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            def __init__(self) -> None:
                self.finished = asyncio.Event()

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                for _ in range(80):
                    yield TextDelta(context, "a")
                self.finished.set()
                yield TurnFinished(context)

        class BlockingSpeechBackend:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def health(self) -> EngineHealth:
                return EngineHealth("unpunctuated", True)

            async def stream(self, request):
                del request
                self.started.set()
                await asyncio.Event().wait()
                if False:
                    yield SpeechChunk("", b"")

        adapter = StreamingAdapter()
        backend = BlockingSpeechBackend()
        router = ModelRouter(
            [ChannelConfig("unpunctuated", base_url="https://streaming.invalid/v1", model="demo")],
            adapter_factory=lambda _channel: adapter,
        )
        service = ConversationService(
            router,
            tts=TTSCoordinator(backend, segment_max_chars=20),
        )
        context = service.begin_context(turn_id="unpunctuated-tts")
        task = asyncio.create_task(service.complete("开始", context=context))

        await asyncio.wait_for(backend.started.wait(), timeout=1.0)
        assert adapter.finished.is_set() is False

        service.cancel(context)
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancelled conversation completed unexpectedly")

    asyncio.run(scenario())


@pytest.mark.parametrize("initial_role", ("", "first-role"))
def test_tts_language_and_role_are_snapshotted_when_the_turn_starts(initial_role: str) -> None:
    async def scenario() -> None:
        class DelayedAdapter(ProviderAdapter):
            provider = "delayed"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            def __init__(self) -> None:
                self.ready = asyncio.Event()
                self.release = asyncio.Event()

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                self.ready.set()
                await self.release.wait()
                yield TextDelta(context, "当前回合。")
                yield TurnFinished(context)

        class RecordingSpeechBackend:
            def __init__(self) -> None:
                self.languages: list[str] = []
                self.roles: list[str] = []

            async def health(self) -> EngineHealth:
                return EngineHealth("recording", True)

            async def stream(self, request):
                self.languages.append(request.language)
                self.roles.append(request.role)
                yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

        adapter = DelayedAdapter()
        backend = RecordingSpeechBackend()
        router = ModelRouter(
            [ChannelConfig("delayed", base_url="https://delayed.invalid/v1", model="demo")],
            adapter_factory=lambda _channel: adapter,
        )
        service = ConversationService(
            router,
            tts=TTSCoordinator(backend),
            tts_language="zh",
            tts_role=initial_role,
        )
        context = service.begin_context(turn_id="language-snapshot")
        task = asyncio.create_task(service.complete("开始", context=context))

        await asyncio.wait_for(adapter.ready.wait(), timeout=1.0)
        service.tts_language = "ja"
        service.tts_role = "second-role"
        adapter.release.set()
        result = await asyncio.wait_for(task, timeout=1.0)

        assert result.text == "当前回合。"
        assert backend.languages == ["zh"]
        assert backend.roles == [initial_role]
        assert not service._tts_language_snapshots
        assert not service._tts_role_snapshots

        await asyncio.wait_for(service.complete("下一回合"), timeout=1.0)
        assert backend.languages == ["zh", "ja"]
        assert backend.roles == [initial_role, "second-role"]
        await service.aclose()

    asyncio.run(scenario())


def test_blocking_audio_sink_does_not_pause_following_model_deltas() -> None:
    async def scenario() -> None:
        class StreamingAdapter(ProviderAdapter):
            provider = "streaming"
            protocol = "openai_chat"
            capabilities = frozenset({"streaming"})

            def __init__(self) -> None:
                self.finished = asyncio.Event()

            async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
                del request, cancel_event
                yield TextDelta(context, "首句。")
                yield TextDelta(context, "后句。")
                self.finished.set()
                yield TurnFinished(context)

        class SpeechBackend:
            def __init__(self) -> None:
                self.started = asyncio.Event()

            async def health(self) -> EngineHealth:
                return EngineHealth("blocking-sink", True)

            async def stream(self, request):
                self.started.set()
                yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

        sink_entered = threading.Event()
        sink_release = threading.Event()

        def blocking_sink(_chunk: object) -> None:
            sink_entered.set()
            sink_release.wait(timeout=2.0)

        adapter = StreamingAdapter()
        backend = SpeechBackend()
        service = ConversationService(
            _router(adapter),
            tts=TTSCoordinator(backend, audio_sink=blocking_sink),
        )
        context = service.begin_context(turn_id="blocking-audio-sink")
        task = asyncio.create_task(service.complete("继续", context=context))

        await asyncio.wait_for(backend.started.wait(), timeout=1.0)
        assert await asyncio.to_thread(sink_entered.wait, 1.0)
        await asyncio.wait_for(adapter.finished.wait(), timeout=1.0)
        assert task.done() is False

        sink_release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
        assert result.text == "首句。后句。"

    asyncio.run(scenario())


def test_tts_dispatch_isolated_by_turn_context() -> None:
    async def scenario() -> None:
        tts = _ProbeTTS()
        service = ConversationService(_router(_DialogueAdapter()), tts=tts)
        first = service.begin_context(turn_id="first")
        second = service.begin_context(turn_id="second")

        service._schedule_tts(first, "甲。")
        service._schedule_tts(second, "乙。")
        await service._flush_tts(first)
        await service._flush_tts(second)

        first_calls = [call for call in tts.calls if call[0] == first]
        second_calls = [call for call in tts.calls if call[0] == second]
        assert [(text, flush) for _, text, flush in first_calls] == [("甲。", False), ("", True)]
        assert [(text, flush) for _, text, flush in second_calls] == [("乙。", False), ("", True)]
        assert not service._tts_turns

    asyncio.run(scenario())


def test_conversation_aclose_cancels_tts_worker() -> None:
    async def scenario() -> None:
        tts = _ProbeTTS(block=True)
        service = ConversationService(_router(_DialogueAdapter()), tts=tts)
        context = service.begin_context(turn_id="close")
        service._schedule_tts(context, "需要取消。")
        await tts.started.wait()

        await service.aclose()

        assert not service._tts_turns
        assert not service._tts_tasks
        assert context in tts.cancelled

    asyncio.run(scenario())


async def _collect(iterator):
    return [event async for event in iterator]


def test_sentence_pipeline_keeps_raw_delta_and_splits_one_provider_packet() -> None:
    class Adapter(ProviderAdapter):
        provider = "sentence-packet"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming"})

        async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
            del request, cancel_event
            yield TextDelta(context, "第一句。第二句。")
            yield TurnFinished(context)

    class Backend:
        def __init__(self) -> None:
            self.requests: list[str] = []

        async def health(self) -> EngineHealth:
            return EngineHealth("sentence-packet", True)

        async def stream(self, request: SpeechRequest):
            self.requests.append(request.text)
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    backend = Backend()
    router = ModelRouter(
        [ChannelConfig("sentence-packet", base_url="https://sentence.invalid/v1", model="demo")],
        adapter_factory=lambda _channel: Adapter(),
    )
    service = ConversationService(router, tts=TTSCoordinator(backend))

    async def scenario() -> None:
        try:
            events = [event async for event in service.stream("开始")]
            raw = [event.delta for event in events if isinstance(event, TextDelta)]
            assert raw == ["第一句。第二句。"]
            assert backend.requests == ["第一句。", "第二句。"]
            snapshot = service.presentation.snapshot
            assert snapshot.text == "第一句。第二句。"
            assert snapshot.speech_text == "第二句。"
            assert snapshot.speech_sequence == 2
            queued = []
            while (sentence := service.presentation.pop_next_sentence()) is not None:
                queued.append(sentence.text)
            assert queued == ["第一句。", "第二句。"]
        finally:
            await service.aclose()
            await router.aclose()

    asyncio.run(scenario())


def test_sentence_pipeline_joins_cross_delta_text_and_flushes_tail_once() -> None:
    class Adapter(ProviderAdapter):
        provider = "sentence-cross-delta"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming"})

        async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
            del request, cancel_event
            yield TextDelta(context, "跨")
            yield TextDelta(context, "包句。尾部")
            yield TurnFinished(context)

    class Backend:
        def __init__(self) -> None:
            self.requests: list[str] = []

        async def health(self) -> EngineHealth:
            return EngineHealth("sentence-cross-delta", True)

        async def stream(self, request: SpeechRequest):
            self.requests.append(request.text)
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    backend = Backend()
    router = ModelRouter(
        [
            ChannelConfig(
                "sentence-cross-delta",
                base_url="https://sentence-cross.invalid/v1",
                model="demo",
            )
        ],
        adapter_factory=lambda _channel: Adapter(),
    )
    service = ConversationService(router, tts=TTSCoordinator(backend))

    async def scenario() -> None:
        try:
            events = [event async for event in service.stream("开始")]
            assert [event.delta for event in events if isinstance(event, TextDelta)] == [
                "跨",
                "包句。尾部",
            ]
            assert backend.requests == ["跨包句。", "尾部"]
            snapshot = service.presentation.snapshot
            assert snapshot.text == "跨包句。尾部"
            assert snapshot.speech_text == "尾部"
            assert snapshot.speech_sequence == 2
        finally:
            await service.aclose()
            await router.aclose()

    asyncio.run(scenario())


def test_sentence_ready_duplicate_sequence_is_ignored() -> None:
    presentation = PresentationService()
    context = ConversationContext("p", "s", "sentence-sequence", 1)

    first = presentation.commit_sentence(SentenceReady(context, "嗯。", 1))
    duplicate = presentation.commit_sentence(SentenceReady(context, "嗯。", 1))
    second = presentation.commit_sentence(SentenceReady(context, "嗯。", 2))

    assert first.speech_text == "嗯。"
    assert duplicate == first
    assert second.speech_text == "嗯。"
    assert second.speech_sequence == 2


def test_sentence_pipeline_without_tts_still_commits_final_tail() -> None:
    class Adapter(ProviderAdapter):
        provider = "sentence-no-tts"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming"})

        async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
            del request, cancel_event
            yield TextDelta(context, "没有句号的尾部")
            yield TurnFinished(context)

    router = ModelRouter(
        [
            ChannelConfig(
                "sentence-no-tts",
                base_url="https://sentence-no-tts.invalid/v1",
                model="demo",
            )
        ],
        adapter_factory=lambda _channel: Adapter(),
    )
    service = ConversationService(router)

    async def scenario() -> None:
        try:
            events = [event async for event in service.stream("开始")]
            assert any(isinstance(event, TurnFinished) for event in events)
            snapshot = service.presentation.snapshot
            assert snapshot.text == "没有句号的尾部"
            assert snapshot.speech_text == "没有句号的尾部"
            assert snapshot.speech_sequence == 1
        finally:
            await service.aclose()
            await router.aclose()

    asyncio.run(scenario())


def test_sentence_pipeline_flushes_before_tool_and_reuses_context_tts_state() -> None:
    class Adapter(ProviderAdapter):
        provider = "sentence-tool"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming", "tools"})

        def __init__(self) -> None:
            self.round = 0

        async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
            del request, cancel_event
            self.round += 1
            if self.round == 1:
                yield TextDelta(context, "前置内容。")
                yield ToolCallDelta(context, 0, "call-sentence", "pet:ping", "{}")
                yield TurnFinished(context, "tool_calls")
                return
            yield TextDelta(context, "续接完成。")
            yield TurnFinished(context)

    class Backend:
        def __init__(self) -> None:
            self.requests: list[str] = []

        async def health(self) -> EngineHealth:
            return EngineHealth("sentence-tool", True)

        async def stream(self, request: SpeechRequest):
            self.requests.append(request.text)
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    async def ping(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "pet:ping",
            "测试工具",
            {"type": "object"},
            ping,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    backend = Backend()
    adapter = Adapter()
    router = ModelRouter(
        [
            ChannelConfig(
                "sentence-tool",
                base_url="https://sentence-tool.invalid/v1",
                model="demo",
                capabilities=("streaming", "tools"),
            )
        ],
        adapter_factory=lambda _channel: adapter,
    )
    service = ConversationService(
        router,
        tools=ToolExecutionService(registry, PermissionService()),
        tts=TTSCoordinator(backend),
    )

    async def scenario() -> None:
        try:
            events = [event async for event in service.stream("开始")]
            assert any(isinstance(event, ToolStatusChanged) for event in events)
            assert any(isinstance(event, TextDelta) for event in events)
        finally:
            await service.aclose()
            await router.aclose()

    asyncio.run(scenario())
    assert backend.requests == ["前置内容。", "续接完成。"]
