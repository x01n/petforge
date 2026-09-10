"""组合根交互状态的代际隔离回归。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.runtime import build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from core.events.types import ConversationContext, TextDelta, ToolStatusChanged, TurnFinished
from services.conversation import PresentationService
from services.conversation.interaction_contract import InteractionPhase
from services.tools.types import ToolCallContext


def test_runtime_cancel_keeps_late_sse_events_from_overwriting_the_next_turn(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {}),
        inspect_resources(tmp_path / "resources"),
    )
    old = runtime.conversation.begin_context(turn_id="old-turn")
    state = runtime.interaction_state
    assert state is not None

    try:
        state.consume(TextDelta(old, "旧回合首段"))
        asyncio.run(runtime.cancel_conversation(old))
        assert state.snapshot.finish_reason == "cancelled"
        new = runtime.conversation.begin_context(turn_id="new-turn")

        # 模拟 SSE/工具队列在取消之后才抵达宿主；它们只能返回当前快照。
        stale = state.consume(TextDelta(old, "迟到文本"))
        stale = state.consume(TurnFinished(old, "stop"))
        stale = state.consume(ToolStatusChanged(old, "completed", "旧工具完成", "old-call"))
        assert stale.finish_reason == "cancelled"
        assert stale.text == ""
        assert stale.phase is InteractionPhase.CONFIGURATION_REQUIRED

        current = state.consume(TextDelta(new, "新回合首段"))
        assert current.turn_id == "new-turn"
        assert current.text == "新回合首段"
        assert current.phase is InteractionPhase.STREAMING
    finally:
        asyncio.run(runtime.close())


def test_presentation_drops_late_cancelled_sse_before_new_bubble() -> None:
    presentation = PresentationService()
    old = ConversationContext("default", "local", "old-bubble", 1)
    new = ConversationContext("default", "local", "new-bubble", 2)

    presentation.consume(TextDelta(old, "旧气泡"))
    presentation.reset(context=old, cancelled=True)
    assert presentation.consume(TextDelta(old, "迟到气泡")).rendered_text == ""

    current = presentation.consume(TextDelta(new, "新气泡"))
    assert current.rendered_text == "新气泡"


def test_stale_cancel_does_not_invalidate_new_generation(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {}),
        inspect_resources(tmp_path / "resources"),
    )
    try:
        old = runtime.conversation.begin_context(turn_id="old-generation")
        new = runtime.conversation.begin_context(turn_id="new-generation")
        assert not runtime.conversation.generation_gate.accepts(old)
        assert runtime.conversation.generation_gate.accepts(new)
        asyncio.run(runtime.cancel_conversation(old))
        assert runtime.conversation.generation_gate.accepts(new)
    finally:
        asyncio.run(runtime.close())


def test_runtime_cancel_consumes_pending_approval_for_the_cancelled_turn(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {}),
        inspect_resources(tmp_path / "resources"),
    )
    context = ConversationContext("default", "local", "approval-cancel", 1)

    async def scenario() -> None:
        tool_context = ToolCallContext(
            context.profile_id,
            context.session_id,
            context.turn_id,
            source="model",
            metadata={"mode": context.mode, "generation_id": context.generation_id},
        )
        outcome = await runtime.tools.execute(
            call_id="approval-cancel-call",
            identity="system:run_command",
            arguments={"command": ["true"]},
            context=tool_context,
        )
        assert outcome.status == "approval_required"
        assert runtime.tools.pending_approvals()
        await runtime.cancel_conversation(context)
        assert runtime.tools.pending_approvals() == ()

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(runtime.close())
