"""交互契约的纯 Python 回归测试。"""

from __future__ import annotations

import json
import time

from core.events.types import (
    ApprovalRequested,
    ConversationContext,
    MurmurDelta,
    ReasoningDelta,
    TextDelta,
    ToolCallStarted,
    ToolStatusChanged,
    TurnFailed,
    TurnFinished,
)
from gui.qt6.pet_interaction import classify_pet_part
from services.conversation.interaction_contract import (
    InteractionActionKind,
    InteractionPhase,
    InteractionState,
    model_readiness,
    pet_feedback,
)


def _context(turn_id: str = "turn-1") -> ConversationContext:
    return ConversationContext("profile", "session", turn_id, 1)


def test_model_readiness_has_clickable_setup_step_without_secrets() -> None:
    result = model_readiness(
        {
            "ready": False,
            "reason": "base_url is missing",
            "channels": (),
            "api_key": "secret-value",
        }
    )

    assert result.ready is False
    assert result.channel_count == 0
    assert "点击“配置模型”" in result.message
    assert result.action is not None
    assert result.action.kind is InteractionActionKind.CONFIGURE_MODEL
    assert result.action.payload == {"target": "model_setup", "mode": "click"}
    assert "secret-value" not in json.dumps(result.to_mapping(), ensure_ascii=False)
    assert "base_url" not in result.reason
    assert "点击“配置模型”" in result.reason


def test_model_readiness_reports_ready_channel_count() -> None:
    result = model_readiness(
        {
            "ready": True,
            "channels": (
                {"id": "primary", "api_key_configured": True},
                {"id": "backup", "api_key_configured": False},
            ),
        }
    )

    assert result.ready is True
    assert result.channel_count == 2
    assert result.message == "模型渠道已就绪（2 个）"
    assert result.action is None


def test_streaming_state_exposes_text_murmur_and_stop_action() -> None:
    state = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    context = _context()

    state.consume(ReasoningDelta(context, "内部推理不应显示"))
    state.consume(TextDelta(context, "你好"))
    snapshot = state.consume(MurmurDelta(context, "（轻声）"))

    assert snapshot.phase is InteractionPhase.STREAMING
    assert snapshot.streaming is True
    assert snapshot.busy is True
    assert snapshot.text == "你好"
    assert snapshot.murmur == "（轻声）"
    assert "内部推理" not in snapshot.rendered_text
    assert [item.kind for item in snapshot.actions] == [InteractionActionKind.STOP]


def test_reasoning_can_be_rendered_as_murmur_when_explicitly_enabled() -> None:
    """显式开启推理展示时，统一快照应把它归入碎碎念而非丢弃。"""

    state = InteractionState(
        diagnostics={"ready": True, "channels": ({"id": "primary"},)},
        show_reasoning=True,
    )
    snapshot = state.consume(ReasoningDelta(_context("reasoning"), "正在想办法…"))

    assert snapshot.phase is InteractionPhase.STREAMING
    assert snapshot.streaming is True
    assert snapshot.busy is True
    assert snapshot.murmur == "正在想办法…"
    assert snapshot.text == ""


def test_approval_state_exposes_only_safe_summary_and_three_click_actions() -> None:
    state = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    context = _context()
    request = {
        "approval_id": "approval-1",
        "call_id": "call-1",
        "identity": "system:open",
        "safe_summary": "需要打开一个文件",
        "expires_at": 2_000_000_000,
        "arguments": {"path": "/private/secret"},
    }
    state.set_pending_approvals((request,))
    snapshot = state.consume(
        ApprovalRequested(
            context,
            "approval-1",
            "call-1",
            "需要打开一个文件",
            2_000_000_000,
        )
    )

    assert snapshot.phase is InteractionPhase.APPROVAL_REQUIRED
    assert snapshot.approval is not None
    assert snapshot.approval.identity == "system:open"
    assert "secret" not in json.dumps(snapshot.to_mapping(), ensure_ascii=False)
    assert [item.approval_id for item in snapshot.pending_approvals] == ["approval-1"]
    assert [item.kind for item in snapshot.actions] == [
        InteractionActionKind.APPROVE,
        InteractionActionKind.GRANT_SESSION,
        InteractionActionKind.DENY,
    ]
    assert snapshot.actions[0].payload == {"approval_id": "approval-1", "grant_session": False}
    assert snapshot.actions[1].payload == {"approval_id": "approval-1", "grant_session": True}


def test_public_interaction_mapping_hides_internal_approval_identifiers() -> None:
    state = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    context = _context("public")
    state.set_pending_approvals(
        (
            {
                "approval_id": "approval-public",
                "call_id": "call-private",
                "identity": "system:run_command",
                "safe_summary": "需要确认系统操作",
                "expires_at": 2_000_000_000,
            },
        )
    )
    snapshot = state.consume(
        ApprovalRequested(
            context,
            "approval-public",
            "call-private",
            "需要确认系统操作",
            2_000_000_000,
        )
    )
    public = snapshot.to_public_mapping()
    encoded = json.dumps(public, ensure_ascii=False)
    assert "call-private" not in encoded
    assert "approval-public" not in encoded
    assert "system:run_command" not in encoded
    assert "需要确认系统操作" in encoded
    assert public["actions"]
    assert public["model"]["action"] is None
    assert state.action_for_public_kind("approve") is not None
    assert state.action_for_public_kind("unknown") is None


def test_approval_turn_finished_keeps_confirmation_until_tool_terminal_status() -> None:
    state = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    context = _context("approval-turn")
    state.set_pending_approvals(
        (
            {
                "approval_id": "approval-turn",
                "call_id": "call-turn",
                "identity": "system:open",
                "safe_summary": "需要确认",
                "expires_at": time.time() + 60,
            },
        )
    )
    state.consume(
        ApprovalRequested(context, "approval-turn", "call-turn", "需要确认", time.time() + 60)
    )

    paused = state.consume(TurnFinished(context, "tool_calls"))
    assert paused.phase is InteractionPhase.APPROVAL_REQUIRED
    assert paused.busy is True
    assert paused.approval is not None
    assert len(paused.actions) == 3

    completed = state.consume(ToolStatusChanged(context, "completed", "工具已完成", "call-turn"))
    assert completed.phase is InteractionPhase.COMPLETED
    assert completed.busy is False
    assert completed.approval is None
    assert completed.pending_approvals == ()
    assert completed.actions == ()


def test_reset_can_clear_pending_approvals_for_cancelled_turn() -> None:
    state = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    state.set_pending_approvals(
        (
            {
                "approval_id": "approval-reset",
                "call_id": "call-reset",
                "identity": "system:open",
                "safe_summary": "需要确认",
                "expires_at": time.time() + 60,
            },
        )
    )

    snapshot = state.reset(preserve_pending=False)

    assert snapshot.pending_approvals == ()
    assert snapshot.approval is None


def test_late_events_from_cancelled_turn_cannot_overwrite_new_turn() -> None:
    state = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    old = _context("old-turn")
    new = ConversationContext("profile", "session", "new-turn", 2)

    state.consume(TextDelta(old, "旧回合首段"))
    cancelled = state.reset(context=old, preserve_pending=False, finish_reason="cancelled")
    assert cancelled.finish_reason == "cancelled"

    # 取消后仍到达的 SSE 文本和终态事件都必须被丢弃。
    assert state.consume(TextDelta(old, "迟到文本")).text == ""
    late_finished = state.consume(TurnFinished(old, "stop"))
    assert late_finished.finish_reason == "cancelled"
    assert late_finished.phase is InteractionPhase.IDLE

    current = state.consume(TextDelta(new, "新回合"))
    assert current.turn_id == "new-turn"
    assert current.text == "新回合"
    assert current.phase is InteractionPhase.STREAMING

    # 旧回合工具终态不能把新回合标记为已完成或清空正文。
    unchanged = state.consume(ToolStatusChanged(old, "completed", "旧工具完成", "old-call"))
    assert unchanged.turn_id == "new-turn"
    assert unchanged.text == "新回合"
    assert unchanged.phase is InteractionPhase.STREAMING


def test_late_approval_terminal_event_cannot_clear_current_approval() -> None:
    state = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    old = _context("old-approval")
    new = ConversationContext("profile", "session", "new-approval", 2)
    expiry = time.time() + 60

    state.set_pending_approvals(
        (
            {
                "approval_id": "new-approval-id",
                "call_id": "new-call",
                "identity": "system:open",
                "safe_summary": "新回合需要确认",
                "expires_at": expiry,
            },
        )
    )
    state.consume(ApprovalRequested(new, "new-approval-id", "new-call", "新回合需要确认", expiry))
    assert state.snapshot.phase is InteractionPhase.APPROVAL_REQUIRED

    unchanged = state.consume(ToolStatusChanged(old, "completed", "旧审批完成", "new-call"))
    assert unchanged.phase is InteractionPhase.APPROVAL_REQUIRED
    assert unchanged.approval is not None
    assert unchanged.approval.approval_id == "new-approval-id"


def test_expired_approval_is_removed_from_the_visible_state() -> None:
    now = [100.0]
    state = InteractionState(
        diagnostics={"ready": True, "channels": ({"id": "primary"},)},
        clock=lambda: now[0],
    )
    state.set_pending_approvals(
        (
            {
                "approval_id": "approval-expired",
                "call_id": "call-expired",
                "safe_summary": "过期请求",
                "expires_at": 99.0,
            },
        )
    )

    assert state.snapshot.approval is None
    assert state.snapshot.actions == ()


def test_expired_approval_event_does_not_leave_a_stale_confirmation_button() -> None:
    event_clock = time.time()
    state = InteractionState(
        diagnostics={"ready": True, "channels": ({"id": "primary"},)},
        clock=lambda: event_clock + 10,
    )

    snapshot = state.consume(
        ApprovalRequested(_context(), "approval-old", "call-old", "已过期", event_clock + 1)
    )

    assert snapshot.phase is InteractionPhase.TOOL_RUNNING
    assert snapshot.approval is None
    assert all(
        item.kind
        not in {
            InteractionActionKind.APPROVE,
            InteractionActionKind.GRANT_SESSION,
            InteractionActionKind.DENY,
        }
        for item in snapshot.actions
    )


def test_configuration_and_retryable_failures_have_distinct_next_steps() -> None:
    context = _context()

    configured = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    config_snapshot = configured.consume(
        TurnFailed(context, "configuration", "请先配置模型", False)
    )
    assert config_snapshot.phase is InteractionPhase.CONFIGURATION_REQUIRED
    assert [item.kind for item in config_snapshot.actions] == [
        InteractionActionKind.CONFIGURE_MODEL
    ]

    retryable = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    retry_snapshot = retryable.consume(TurnFailed(context, "network", "网络暂时不可用", True))
    assert retry_snapshot.phase is InteractionPhase.FAILED
    assert [item.kind for item in retry_snapshot.actions] == [InteractionActionKind.RETRY]


def test_tool_call_and_turn_finished_update_busy_and_usage() -> None:
    state = InteractionState(diagnostics={"ready": True, "channels": ({"id": "primary"},)})
    context = _context("turn-2")

    running = state.consume(ToolCallStarted(context, "call-2", "pet:move", "正在移动桌宠"))
    assert running.phase is InteractionPhase.TOOL_RUNNING
    assert running.busy is True
    assert running.tool_state == "running"

    finished = state.consume(TurnFinished(context, "stop", {"input_tokens": 3, "output_tokens": 4}))
    assert finished.phase is InteractionPhase.COMPLETED
    assert finished.busy is False
    assert finished.usage == {"input_tokens": 3, "output_tokens": 4}
    assert finished.actions == ()


def test_pet_feedback_matches_existing_click_zones_and_rejects_unknown_zone() -> None:
    assert pet_feedback("upper").to_mapping() == {
        "zone": "upper",
        "expression": "happy",
        "motion": "wave",
        "phrase": "喵？摸摸头～",
        "mood": "happy",
    }
    assert pet_feedback("head").phrase == "喵？摸摸头～"
    assert pet_feedback("body").to_mapping() == {
        "zone": "body",
        "expression": "curious",
        "motion": "blink",
        "phrase": "呼噜……这里也可以摸。",
        "mood": "curious",
    }
    assert pet_feedback(" lower_right ").mood == "shy"
    assert pet_feedback("unknown") is None


def test_native_pet_part_fallback_distinguishes_head_and_body() -> None:
    assert classify_pet_part(210, 120, 420, 480) == "head"
    assert classify_pet_part(210, 220, 420, 480) == "body"
    assert classify_pet_part(80, 390, 420, 480) == "lower_left"
    assert classify_pet_part(340, 390, 420, 480) == "lower_right"
    assert classify_pet_part(5, 5, 420, 480) == "head"
    assert classify_pet_part(-1, 5, 420, 480) is None
