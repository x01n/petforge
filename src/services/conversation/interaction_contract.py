from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from math import isfinite

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
from services.tools.types import tool_user_label


class InteractionPhase(StrEnum):
    """面向用户的回合阶段。"""

    IDLE = "idle"
    STREAMING = "streaming"
    TOOL_RUNNING = "tool_running"
    APPROVAL_REQUIRED = "approval_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFIGURATION_REQUIRED = "configuration_required"


class InteractionActionKind(StrEnum):
    """宿主可绑定到按钮或快捷操作的动作类型。"""

    CONFIGURE_MODEL = "configure_model"
    STOP = "stop"
    APPROVE = "approve"
    DENY = "deny"
    GRANT_SESSION = "grant_session"
    RETRY = "retry"


@dataclass(frozen=True)
class InteractionAction:
    """一个不携带敏感参数的用户动作。"""

    kind: InteractionActionKind
    label: str
    payload: Mapping[str, object] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        label = str(self.label or "").strip()
        if not label:
            raise ValueError("interaction action label is required")
        object.__setattr__(self, "kind", InteractionActionKind(self.kind))
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "payload", dict(self.payload or {}))
        object.__setattr__(self, "enabled", bool(self.enabled))

    def to_mapping(self) -> dict[str, object]:
        """返回可直接发送给前端的 JSON 结构。"""

        return {
            "kind": self.kind.value,
            "label": self.label,
            "payload": dict(self.payload),
            "enabled": self.enabled,
        }

    def to_public_mapping(self) -> dict[str, object]:
        """返回普通界面可消费的动作，不携带审批或回合内部标识。"""

        payload: dict[str, object] = {}
        if self.kind is InteractionActionKind.CONFIGURE_MODEL:
            # 配置向导的目标是公开 UI 路由；其余动作的 payload 可能包含
            # approval_id 等内部标识，由宿主按当前上下文解析。
            for key in ("target", "mode"):
                value = self.payload.get(key)
                if isinstance(value, (str, int, float, bool)):
                    payload[key] = value
        return {
            "kind": self.kind.value,
            "label": self.label,
            "payload": payload,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class ModelReadiness:
    """模型渠道就绪状态及可执行的下一步。"""

    ready: bool | None = None
    message: str = "模型渠道状态未读取"
    reason: str = ""
    channel_count: int = 0
    action: InteractionAction | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "ready": self.ready,
            "message": self.message,
            "reason": self.reason,
            "channel_count": self.channel_count,
            "action": self.action.to_mapping() if self.action is not None else None,
        }

    def to_public_mapping(self) -> dict[str, object]:
        """返回普通界面所需的模型状态，不携带内部动作载荷。"""

        return {
            "ready": self.ready,
            "message": self.message,
            "reason": self.reason,
            "channel_count": self.channel_count,
            "action": self.action.to_public_mapping() if self.action is not None else None,
        }


@dataclass(frozen=True)
class ApprovalView:
    """审批卡片所需的最小字段；不会复制工具参数。"""

    approval_id: str
    call_id: str
    identity: str
    safe_summary: str
    expires_at: float
    display_name: str = ""

    def __post_init__(self) -> None:
        for name in ("approval_id", "call_id"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"approval {name} is required")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "identity", str(self.identity or "").strip())
        object.__setattr__(self, "safe_summary", str(self.safe_summary or "").strip())
        object.__setattr__(
            self,
            "display_name",
            str(self.display_name or "").strip() or "工具操作",
        )
        expires_at = float(self.expires_at)
        if not isfinite(expires_at) or expires_at <= 0:
            raise ValueError("approval expires_at must be a finite positive number")
        object.__setattr__(self, "expires_at", expires_at)

    def remaining_seconds(self, *, now: float | None = None) -> float:
        """计算展示倒计时；实际批准仍由权限服务再次校验。"""

        current = time.time() if now is None else float(now)
        return max(0.0, self.expires_at - current)

    def to_mapping(self) -> dict[str, object]:
        return {
            "approval_id": self.approval_id,
            "call_id": self.call_id,
            "identity": self.identity,
            "display_name": self.display_name,
            "safe_summary": self.safe_summary,
            "expires_at": self.expires_at,
        }

    def to_public_mapping(self) -> dict[str, object]:
        """返回网页/普通界面可见字段，不携带内部调用标识。"""

        return {
            "display_name": self.display_name,
            "safe_summary": self.safe_summary,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class PetFeedback:
    """桌宠点击分区对应的视觉/文案反馈。"""

    zone: str
    expression: str
    motion: str
    phrase: str
    mood: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "zone": self.zone,
            "expression": self.expression,
            "motion": self.motion,
            "phrase": self.phrase,
            "mood": self.mood,
        }


# 与 ``src/gui/qt6/app.py`` 当前点击分区逐项对应；这里单独提供纯数据版本，
# 供 Web/Qt 渲染层复用，不直接调用渲染器。
_PET_FEEDBACK: dict[str, PetFeedback] = {
    "upper": PetFeedback("upper", "happy", "wave", "喵？摸摸头～", "happy"),
    "head": PetFeedback("head", "happy", "wave", "喵？摸摸头～", "happy"),
    "body": PetFeedback("body", "curious", "blink", "呼噜……这里也可以摸。", "curious"),
    "lower_left": PetFeedback("lower_left", "curious", "blink", "左边被发现啦。", "curious"),
    "lower_right": PetFeedback("lower_right", "surprised", "blink", "右边也要轻轻碰哦。", "shy"),
}


def pet_feedback(zone: str) -> PetFeedback | None:
    """按现有点击分区返回反馈；未知分区不触发任何动作。"""

    return _PET_FEEDBACK.get(str(zone or "").strip().lower())


def _model_setup_action() -> InteractionAction:
    return InteractionAction(
        InteractionActionKind.CONFIGURE_MODEL,
        "配置模型",
        {
            "target": "model_setup",
            "mode": "click",
        },
    )


def _channel_rows(value: object) -> tuple[object, ...]:
    """仅统计 router diagnostics 的 channels，不读取任何密钥字段。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)) or value is None:
        return ()
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return ()


def model_readiness(diagnostics: Mapping[str, object] | None) -> ModelReadiness:
    """把 ``ModelRouter.diagnostics("dialogue")`` 转成用户可读状态。

    只读取当前路由器已经公开的 ``ready``、``channels`` 和 ``reason`` 字段，
    因此不会把 ``api_key`` 或自定义请求头带入前端状态。
    """

    if not isinstance(diagnostics, Mapping):
        return ModelReadiness()
    rows = _channel_rows(diagnostics.get("channels"))
    ready = bool(diagnostics.get("ready"))
    if ready:
        return ModelReadiness(
            ready=True,
            message=f"模型渠道已就绪（{len(rows)} 个）",
            reason="ready",
            channel_count=len(rows),
        )
    reason = _public_model_reason(
        diagnostics.get("reason", "渠道未就绪"),
        has_channels=bool(rows),
    )
    if not rows:
        return ModelReadiness(
            ready=False,
            message="模型渠道未配置：点击“配置模型”填写渠道",
            reason=reason,
            action=_model_setup_action(),
        )
    return ModelReadiness(
        ready=False,
        message=f"模型渠道未就绪：{reason}",
        reason=reason,
        channel_count=len(rows),
        action=_model_setup_action(),
    )


def _public_model_reason(value: object, *, has_channels: bool) -> str:
    """将路由诊断压缩为用户下一步，不暴露字段名或环境变量。"""

    normalized = " ".join(str(value or "").split()).casefold()
    if not has_channels:
        return "请点击“配置模型”添加服务连接"
    if "disabled" in normalized or "停用" in normalized:
        return "模型渠道已停用，请点击“配置模型”启用服务"
    if "cooldown" in normalized or "冷却" in normalized:
        return "模型渠道暂时冷却，请稍后重试或切换渠道"
    return "请点击“配置模型”检查服务连接"


def _field(value: object, name: str, default: object = "") -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _approval_view(value: object) -> ApprovalView | None:
    approval_id = str(_field(value, "approval_id", "") or "").strip()
    call_id = str(_field(value, "call_id", "") or "").strip()
    if not approval_id or not call_id:
        return None
    try:
        expires_at = float(_field(value, "expires_at", 0.0))
    except (TypeError, ValueError):
        return None
    try:
        return ApprovalView(
            approval_id=approval_id,
            call_id=call_id,
            identity=str(_field(value, "identity", "") or "").strip(),
            safe_summary=str(_field(value, "safe_summary", "") or "").strip(),
            expires_at=expires_at,
            display_name=str(
                _field(value, "display_name", _field(value, "label", "工具操作")) or "工具操作"
            ).strip(),
        )
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class InteractionSnapshot:
    """一次可直接序列化的前端交互快照。"""

    phase: InteractionPhase = InteractionPhase.IDLE
    streaming: bool = False
    busy: bool = False
    text: str = ""
    murmur: str = ""
    tool_state: str = ""
    tool_status: str = ""
    mood: str = "neutral"
    approval: ApprovalView | None = None
    pending_approvals: tuple[ApprovalView, ...] = ()
    model: ModelReadiness = field(default_factory=ModelReadiness)
    turn_id: str = ""
    finish_reason: str = ""
    usage: Mapping[str, int] = field(default_factory=dict)
    error_category: str = ""
    safe_message: str = ""
    retryable: bool = False
    actions: tuple[InteractionAction, ...] = ()

    @property
    def rendered_text(self) -> str:
        """按用户优先级合并正文、碎碎念和工具状态。"""

        return "\n".join(item for item in (self.text, self.murmur, self.tool_status) if item)

    def to_mapping(self) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "streaming": self.streaming,
            "busy": self.busy,
            "text": self.text,
            "murmur": self.murmur,
            "tool_state": self.tool_state,
            "tool_status": self.tool_status,
            "mood": self.mood,
            "approval": self.approval.to_mapping() if self.approval is not None else None,
            "pending_approvals": [item.to_mapping() for item in self.pending_approvals],
            "model": self.model.to_mapping(),
            "turn_id": self.turn_id,
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "error_category": self.error_category,
            "safe_message": self.safe_message,
            "retryable": self.retryable,
            "actions": [action.to_mapping() for action in self.actions],
            "rendered_text": self.rendered_text,
        }

    def to_public_mapping(self) -> dict[str, object]:
        """返回点击式前端所需的最小状态，隐藏回合/调用内部字段。"""

        return {
            "phase": self.phase.value,
            "streaming": self.streaming,
            "busy": self.busy,
            "text": self.text,
            "murmur": self.murmur,
            "tool_status": self.tool_status,
            "mood": self.mood,
            "approval": self.approval.to_public_mapping() if self.approval is not None else None,
            "pending_approvals": [item.to_public_mapping() for item in self.pending_approvals],
            "model": self.model.to_public_mapping(),
            "safe_message": self.safe_message,
            "retryable": self.retryable,
            "actions": [action.to_public_mapping() for action in self.actions],
            "rendered_text": self.rendered_text,
        }


class InteractionState:
    """消费核心事件并维护一份无 GUI 依赖的用户交互状态。"""

    def __init__(
        self,
        *,
        diagnostics: Mapping[str, object] | None = None,
        max_text_length: int = 4000,
        show_reasoning: bool = False,
        clock=time.time,
    ) -> None:
        self.max_text_length = max(256, min(int(max_text_length), 20000))
        # 推理内容默认不进入公开快照；用户显式开启后才按碎碎念展示。
        # 这样 Qt/Web 两条展示路径与 PresentationService 保持同一策略，
        # 同时避免把供应商的内部思考默认暴露给桌面控制面。
        self.show_reasoning = bool(show_reasoning)
        self._clock = clock
        self._model = model_readiness(diagnostics)
        self._pending: dict[str, ApprovalView] = {}
        # 交互状态必须和 SSE/事件流的会话代际绑定。旧回合被取消后，提供方
        # 仍可能把已经排队的 delta、工具终态或审批事件送到宿主；如果只看
        # ``turn_id`` 的最后写入值，这些迟到事件会覆盖新回合的气泡和按钮。
        self._active_context: ConversationContext | None = None
        self._cancelled_context: ConversationContext | None = None
        self._snapshot = InteractionSnapshot(
            phase=(
                InteractionPhase.CONFIGURATION_REQUIRED
                if self._model.ready is False
                else InteractionPhase.IDLE
            ),
            model=self._model,
            actions=((self._model.action,) if self._model.action is not None else ()),
        )

    @property
    def snapshot(self) -> InteractionSnapshot:
        return self._snapshot

    def action_for_public_kind(self, kind: object) -> InteractionAction | None:
        """按公开动作类型解析当前内部动作，避免前端携带审批标识。"""

        normalized = str(getattr(kind, "value", kind) or "").strip().lower()
        for action in self._snapshot.actions:
            if action.kind.value == normalized and action.enabled:
                return action
        return None

    def set_model_diagnostics(
        self, diagnostics: Mapping[str, object] | None
    ) -> InteractionSnapshot:
        """刷新渠道诊断；不会改变当前流式正文。"""

        self._model = model_readiness(diagnostics)
        phase = self._snapshot.phase
        if self._model.ready is False and not self._snapshot.busy:
            phase = InteractionPhase.CONFIGURATION_REQUIRED
        elif self._model.ready is True and phase is InteractionPhase.CONFIGURATION_REQUIRED:
            phase = InteractionPhase.IDLE
        self._snapshot = replace(
            self._snapshot,
            phase=phase,
            model=self._model,
            actions=self._actions_for_snapshot(
                replace(self._snapshot, phase=phase, model=self._model)
            ),
        )
        return self._snapshot

    def set_pending_approvals(self, requests: Sequence[object]) -> InteractionSnapshot:
        """更新审批摘要；仅保留 UI 所需字段，不复制工具参数。"""

        pending: dict[str, ApprovalView] = {}
        now = float(self._clock())
        for request in requests or ():
            view = _approval_view(request)
            if view is not None and view.expires_at > now:
                pending[view.approval_id] = view
        self._pending = pending
        approval = self._snapshot.approval
        if approval is not None:
            approval = pending.get(approval.approval_id)
        phase = self._snapshot.phase
        if phase is InteractionPhase.APPROVAL_REQUIRED and approval is None:
            phase = InteractionPhase.TOOL_RUNNING if self._snapshot.busy else InteractionPhase.IDLE
        updated = replace(
            self._snapshot,
            phase=phase,
            approval=approval,
            pending_approvals=tuple(
                sorted(pending.values(), key=lambda item: (item.expires_at, item.approval_id))
            ),
        )
        self._snapshot = replace(updated, actions=self._actions_for_snapshot(updated))
        return self._snapshot

    def reset(
        self,
        *,
        turn_id: str = "",
        context: ConversationContext | None = None,
        preserve_pending: bool = True,
        finish_reason: str = "",
    ) -> InteractionSnapshot:
        """开始新回合；可选择保留审批列表并记录上一次停止原因。"""

        if context is not None and not isinstance(context, ConversationContext):
            raise TypeError("context must be a ConversationContext")
        if context is not None:
            self._active_context = context
            self._cancelled_context = context if finish_reason == "cancelled" else None
            turn_id = context.turn_id
        elif finish_reason != "cancelled":
            # 没有上下文的普通 reset 不能伪造一个新的代际；保留当前游标，
            # 让后续真正带上下文的首个事件负责切换到新回合。
            self._cancelled_context = None
        if not preserve_pending:
            self._pending.clear()
        self._snapshot = InteractionSnapshot(
            phase=(
                InteractionPhase.CONFIGURATION_REQUIRED
                if self._model.ready is False
                else InteractionPhase.IDLE
            ),
            model=self._model,
            turn_id=str(turn_id or "").strip(),
            finish_reason=str(finish_reason or "").strip(),
            pending_approvals=tuple(
                sorted(
                    self._pending.values(),
                    key=lambda item: (item.expires_at, item.approval_id),
                )
            ),
            actions=((self._model.action,) if self._model.action is not None else ()),
        )
        return self._snapshot

    def _accept_event_context(self, context: object) -> bool:
        """拒绝迟到事件，并在新代际首事件到来时清理旧展示状态。"""

        if not isinstance(context, ConversationContext):
            # 非对话事件不会携带代际；保留现有兼容行为。
            return True
        active = self._active_context
        if active is None:
            self._active_context = context
            self._cancelled_context = None
            return True
        if self._cancelled_context == context:
            return False
        if context == active:
            return True

        same_session = (
            context.mode == active.mode
            and context.profile_id == active.profile_id
            and context.session_id == active.session_id
        )
        if same_session:
            # GenerationGate 单调递增；更小代际或同代不同 turn 都是迟到/畸形
            # 事件。更大代际表示用户已经开始下一轮，清除旧正文和审批按钮。
            if context.generation_id <= active.generation_id:
                return False
        elif self._snapshot.busy or self._snapshot.phase is InteractionPhase.APPROVAL_REQUIRED:
            # 不同会话不能在当前回合进行中抢占桌宠控制台。
            return False

        self.reset(context=context, preserve_pending=False)
        return True

    def consume(self, event: object) -> InteractionSnapshot:
        """消费一个核心事件，返回新的不可变快照。"""

        if not self._accept_event_context(getattr(event, "context", None)):
            return self._snapshot
        snapshot = self._snapshot
        context = getattr(event, "context", None)
        event_turn_id = str(getattr(context, "turn_id", "") or "").strip()
        if event_turn_id:
            snapshot = replace(snapshot, turn_id=event_turn_id)

        if isinstance(event, TextDelta):
            snapshot = replace(
                snapshot,
                phase=InteractionPhase.STREAMING,
                streaming=True,
                busy=True,
                text=self._append(snapshot.text, event.delta),
                tool_state="",
                tool_status="",
                approval=None,
                error_category="",
                safe_message="",
                retryable=False,
            )
        elif isinstance(event, MurmurDelta):
            snapshot = replace(
                snapshot,
                phase=InteractionPhase.STREAMING,
                streaming=True,
                busy=True,
                murmur=self._append(snapshot.murmur, event.delta),
            )
        elif isinstance(event, ReasoningDelta):
            if not self.show_reasoning:
                # 原始 reasoning 默认不进入用户状态，遵循当前
                # PresentationService 策略。
                return self._snapshot
            snapshot = replace(
                snapshot,
                phase=InteractionPhase.STREAMING,
                streaming=True,
                busy=True,
                murmur=self._append(snapshot.murmur, event.delta),
            )
        elif isinstance(event, ToolCallStarted):
            snapshot = replace(
                snapshot,
                phase=InteractionPhase.TOOL_RUNNING,
                streaming=False,
                busy=True,
                tool_state="running",
                tool_status=event.safe_summary or f"正在准备 {tool_user_label(event.identity)}",
                approval=None,
            )
        elif isinstance(event, ToolStatusChanged):
            terminal = event.state in {"completed", "denied", "failed"}
            if terminal:
                pending = dict(self._pending)
                for approval_id, pending_view in tuple(pending.items()):
                    if pending_view.call_id == event.call_id:
                        pending.pop(approval_id, None)
                self._pending = pending
                approval = snapshot.approval
                if approval is not None and approval.call_id == event.call_id:
                    approval = None
                snapshot = replace(
                    snapshot,
                    phase=(
                        InteractionPhase.FAILED
                        if event.state == "failed"
                        else InteractionPhase.COMPLETED
                    ),
                    streaming=False,
                    busy=False,
                    tool_state=event.state,
                    tool_status=event.safe_summary,
                    approval=approval,
                    pending_approvals=tuple(
                        sorted(
                            pending.values(),
                            key=lambda item: (item.expires_at, item.approval_id),
                        )
                    ),
                )
            else:
                snapshot = replace(
                    snapshot,
                    phase=InteractionPhase.TOOL_RUNNING,
                    streaming=False,
                    busy=True,
                    tool_state=event.state,
                    tool_status=event.safe_summary,
                    approval=None,
                )
        elif isinstance(event, ApprovalRequested):
            approval = self._pending.get(event.approval_id)
            if approval is None:
                approval = _approval_view(
                    {
                        "approval_id": event.approval_id,
                        "call_id": event.call_id,
                        "safe_summary": event.safe_summary,
                        "expires_at": event.expires_at,
                    }
                )
            if approval is not None and approval.expires_at <= float(self._clock()):
                approval = None
            pending = self._pending
            if approval is not None:
                pending = dict(pending)
                pending[approval.approval_id] = approval
                self._pending = pending
            snapshot = replace(
                snapshot,
                phase=(
                    InteractionPhase.APPROVAL_REQUIRED
                    if approval is not None
                    else InteractionPhase.TOOL_RUNNING
                ),
                streaming=False,
                busy=True,
                tool_state="approval_required",
                tool_status=event.safe_summary,
                approval=approval,
                pending_approvals=tuple(
                    sorted(
                        pending.values(),
                        key=lambda item: (item.expires_at, item.approval_id),
                    )
                ),
            )
        elif isinstance(event, TurnFinished):
            # 对话在等待工具审批时也会发出一条回合结束事件；此时回合是
            # “暂停等待确认”，不能把批准按钮清成已完成状态。
            if (
                snapshot.phase is InteractionPhase.APPROVAL_REQUIRED
                and snapshot.approval is not None
            ):
                snapshot = replace(
                    snapshot,
                    phase=InteractionPhase.APPROVAL_REQUIRED,
                    streaming=False,
                    busy=True,
                    finish_reason=event.finish_reason,
                    usage=dict(event.usage),
                )
            else:
                snapshot = replace(
                    snapshot,
                    phase=InteractionPhase.COMPLETED,
                    streaming=False,
                    busy=False,
                    finish_reason=event.finish_reason,
                    usage=dict(event.usage),
                    approval=None,
                    actions=(),
                )
        elif isinstance(event, TurnFailed):
            configuration_error = event.category == "configuration"
            model = snapshot.model
            if configuration_error and model.action is None:
                model = replace(
                    model,
                    ready=False,
                    message=event.safe_message or "模型渠道未就绪，请先配置模型",
                    reason=event.safe_message or model.reason,
                    action=_model_setup_action(),
                )
            if configuration_error:
                self._model = model
            snapshot = replace(
                snapshot,
                phase=(
                    InteractionPhase.CONFIGURATION_REQUIRED
                    if configuration_error
                    else InteractionPhase.FAILED
                ),
                streaming=False,
                busy=False,
                approval=None,
                error_category=event.category,
                safe_message=event.safe_message,
                retryable=event.retryable,
                model=model,
            )

        self._snapshot = replace(snapshot, actions=self._actions_for_snapshot(snapshot))
        return self._snapshot

    def _append(self, current: str, delta: str) -> str:
        return (current + str(delta or ""))[-self.max_text_length :]

    def _actions_for_snapshot(self, snapshot: InteractionSnapshot) -> tuple[InteractionAction, ...]:
        actions: list[InteractionAction] = []
        if snapshot.phase is InteractionPhase.CONFIGURATION_REQUIRED:
            if snapshot.model.action is not None:
                actions.append(snapshot.model.action)
        elif snapshot.phase is InteractionPhase.APPROVAL_REQUIRED and snapshot.approval is not None:
            approval_id = snapshot.approval.approval_id
            actions.extend(
                (
                    InteractionAction(
                        InteractionActionKind.APPROVE,
                        "批准",
                        {"approval_id": approval_id, "grant_session": False},
                    ),
                    InteractionAction(
                        InteractionActionKind.GRANT_SESSION,
                        "批准并允许本会话",
                        {"approval_id": approval_id, "grant_session": True},
                    ),
                    InteractionAction(
                        InteractionActionKind.DENY,
                        "拒绝",
                        {"approval_id": approval_id},
                    ),
                )
            )
        elif snapshot.phase is InteractionPhase.FAILED and snapshot.retryable:
            actions.append(InteractionAction(InteractionActionKind.RETRY, "重试"))
        if snapshot.busy and snapshot.phase not in {
            InteractionPhase.APPROVAL_REQUIRED,
            InteractionPhase.CONFIGURATION_REQUIRED,
        }:
            actions.append(InteractionAction(InteractionActionKind.STOP, "停止"))
        return tuple(actions)


__all__ = [
    "ApprovalView",
    "InteractionAction",
    "InteractionActionKind",
    "InteractionPhase",
    "InteractionSnapshot",
    "InteractionState",
    "ModelReadiness",
    "PetFeedback",
    "model_readiness",
    "pet_feedback",
]
