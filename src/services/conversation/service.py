"""对话回合编排：流式文本、工具循环、记忆、好感度与 TTS。"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import secrets
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal, cast

from core.adapters.direct.base import ProviderResponse
from core.adapters.direct.errors import AdapterCancelled, ProviderAdapterError
from core.adapters.direct.events import AdapterMetadata, ToolCallDelta
from core.contracts.chat import ChatMessage, ChatRequest, ImageAttachment, ToolCall, ToolDefinition
from core.conversation.runtime import ConversationKey, GenerationGate
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
from db.conversation_repository import ConversationRepository, ConversationTurn
from services.affection.service import AffectionService
from services.memory.service import MemoryService
from services.memory.summarizer import enqueue_model_memory_extraction
from services.model_routing.router import ModelRouter, ModelRoutingError
from services.tools.executor import ToolExecutionService, ToolOutcome, ToolPlanResult
from services.tools.plan import (
    AGENT_PLAN_TOOL_IDENTITY,
    agent_plan_outcome,
    agent_plan_tool_definition,
    parse_agent_plan_call,
)
from services.tools.types import RiskLevel, ToolCallContext, tool_user_label
from services.tts.coordinator import SpeechSegmenter, TTSCoordinator

from .presentation import PresentationService

EventSink = Callable[[object], Awaitable[None] | None]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationResult:
    """一个回合结束后的安全聚合结果。"""

    context: ConversationContext
    text: str
    reasoning: str = ""
    murmur: str = ""
    tool_results: tuple[ToolOutcome, ...] = ()
    finish_reason: str = "stop"
    usage: Mapping[str, int] = field(default_factory=dict)
    status: str = "completed"


@dataclass
class _TTSTurnState:
    """单个会话回合的有界语音待发缓冲。"""

    context: ConversationContext
    language: str = "zh"
    role: str = ""
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    capacity_wake: asyncio.Event = field(default_factory=asyncio.Event)
    pending_text: str = ""
    pending_chunks: deque[str] = field(default_factory=deque)
    backlog_chars: int = 0
    flush_requested: bool = False
    cancelled: bool = False
    overflow_logged: bool = False
    task: asyncio.Task[object] | None = None


@dataclass
class _SentenceTurnState:
    """一个对话代际共享的句子分段器和序号。"""

    segmenter: SpeechSegmenter
    sequence: int = 0
    cancelled: bool = False


@dataclass
class _PausedTurn:
    """等待用户审批的回合快照。

    ``ConversationService.stream`` 在审批点会结束当前异步生成器，Qt 线程
    不能把一个已经结束的生成器重新 ``send`` 回去。因此把下一次模型请求
    所需的消息、分段和上下文保存在运行时边界，批准后由显式 continuation
    API 恢复；工具参数仍只保存在 ``ToolExecutionService`` 内部。
    """

    context: ConversationContext
    user_text: str
    messages: list[ChatMessage]
    definitions: tuple[ToolDefinition, ...]
    next_round_index: int
    pending_call_id: str
    pending_approval_id: str
    segments: list[dict[str, str]]
    reasoning: str
    murmur: str
    answer: str
    outcomes: list[ToolOutcome]
    usage: dict[str, int]
    finish_reason: str
    model: str | None = None
    channel_id: str | None = None
    protocol: str | None = None
    approved_outcome: ToolOutcome | None = None
    pending_plan: ToolPlanResult | None = None
    approved_step_outcome: ToolOutcome | None = None
    memory_user_chat_id: int | None = None


class ConversationService:
    """面向 UI 的单一对话入口。"""

    # 单回合仅保留有限的待合成文本。语音引擎落后于模型输出时，文本展示继续完整
    # 进行，而语音保留最早尚未处理的片段，避免为每个 delta 堆积协程和任务对象。
    _TTS_PENDING_CHAR_LIMIT = 2_048
    # 模型生产速度可能暂时高于语音后端；超过这个回合级上限后，流式
    # 消费者会等待 worker 消费，而不是继续无限增长内存。已经接收的文本
    # 始终保留在 FIFO 中，取消/关闭时由对应回合统一清理。
    _TTS_BACKLOG_CHAR_LIMIT = 32 * 1024
    _TTS_DISPATCH_CHAR_LIMIT = 256
    _TTS_DEFAULT_SEGMENT_CHARS = 120
    _TTS_BOUNDARY_CHARS = frozenset("。！？!?…｡．.\n\r")

    def _tool_label(self, identity: object) -> str:
        """返回工具的用户标签，不把内部命名空间写入气泡。"""

        if self.tools is not None:
            spec = self.tools.registry.get(str(identity or ""))
            if spec is not None:
                return spec.user_label
        return tool_user_label(identity)

    def _tool_outcome_summary(self, outcome: ToolOutcome) -> str:
        error = outcome.content.get("error") if isinstance(outcome.content, Mapping) else None
        if error:
            error_text = str(error).strip()
            lowered = error_text.casefold()
            if any(
                marker in lowered
                for marker in (
                    "api_key",
                    "api-key",
                    "token",
                    "secret",
                    "password",
                    "authorization",
                    "cookie",
                )
            ):
                return "操作失败（敏感详情已隐藏）"
            if error_text == "unknown tool":
                return "工具暂不可用"
            if error_text == "invalid arguments":
                return "操作参数不符合安全要求"
            if error_text == "tool call limit exceeded":
                return "本回合操作次数已达上限"
            return error_text[:240]
        label = self._tool_label(outcome.identity)
        return {
            "completed": f"已完成：{label}",
            "duplicate": f"已拦截重复调用：{label}",
            "denied": f"已拒绝：{label}",
            "failed": f"执行失败：{label}",
        }.get(outcome.status, label)

    def __init__(
        self,
        router: ModelRouter,
        *,
        memory: MemoryService | None = None,
        affection: AffectionService | None = None,
        tools: ToolExecutionService | None = None,
        tts: TTSCoordinator | None = None,
        repository: ConversationRepository | None = None,
        event_sink: EventSink | None = None,
        presentation: PresentationService | None = None,
        generation_gate: GenerationGate | None = None,
        system_prompt: str = "你是一个有边界感、会主动行动的桌宠。",
        task: str = "dialogue",
        model: str | None = None,
        tool_groups: Iterable[str] | None = None,
        max_tool_rounds: int = 4,
        history_limit: int = 12,
        show_reasoning: bool = False,
        tts_language: str = "zh",
        tts_role: str = "",
    ) -> None:
        if not isinstance(router, ModelRouter):
            raise TypeError("router must be a ModelRouter")
        self.router = router
        self.memory = memory
        self.affection = affection
        self.tools = tools
        self.tts = tts
        self.repository = repository
        self.event_sink = event_sink
        self.presentation = presentation or PresentationService(show_reasoning=show_reasoning)
        self.generation_gate = generation_gate or GenerationGate()
        self.system_prompt = str(system_prompt or "").strip()
        self.task = str(task or "dialogue").strip() or "dialogue"
        self.model = str(model or "").strip() or None
        self.tool_groups = None if tool_groups is None else tuple(str(item) for item in tool_groups)
        self.max_tool_rounds = max(1, min(int(max_tool_rounds), 8))
        self.history_limit = max(0, min(int(history_limit), 50))
        self.tts_language = str(tts_language or "zh").strip() or "zh"
        self.tts_role = str(tts_role or "").strip()
        self._tts_language_snapshots: dict[ConversationContext, str] = {}
        self._tts_role_snapshots: dict[ConversationContext, str] = {}
        self._tts_turns: dict[ConversationContext, _TTSTurnState] = {}
        self._sentence_turns: dict[ConversationContext, _SentenceTurnState] = {}
        self._tts_tasks: set[asyncio.Task[object]] = set()
        self._results: dict[tuple[str, str, str, str, int], ConversationResult] = {}
        self._paused_turns: dict[str, _PausedTurn] = {}
        # 同一审批可能同时收到多个批准回调；只串行消费审批快照，
        # 防止第二个回调重复启动模型续接请求。
        self._approval_resume_lock = asyncio.Lock()
        self._active_cancels: dict[ConversationKey, asyncio.Event] = {}
        self._active_task_by_key: dict[ConversationKey, asyncio.Task[object]] = {}
        self._active_tasks: set[asyncio.Task[object]] = set()
        self._closed = False

    def begin_context(
        self,
        *,
        profile_id: str = "default",
        session_id: str = "local",
        mode: str = "direct",
        turn_id: str | None = None,
    ) -> ConversationContext:
        """为新回合分配单调代际标识。"""

        key = ConversationKey(mode, profile_id, session_id)
        normalized_turn = str(turn_id or "").strip() or f"turn-{secrets.token_urlsafe(8)}"
        return self.generation_gate.begin(key, normalized_turn)

    def cancel(self, context: ConversationContext) -> None:
        """取消回合并推进该会话代际。"""

        key = ConversationKey(context.mode, context.profile_id, context.session_id)
        # 旧回合的停止请求可能在用户已经提交新回合后才到达；不能让
        # ``generation_gate.cancel`` 再推进一次并误伤当前活动任务。待审批
        # 快照即使没有登记在 GenerationGate 中也必须清理，避免旧按钮残留。
        accepted = self.generation_gate.accepts(context)
        active_cancel = self._active_cancels.get(key)
        if accepted and active_cancel is not None:
            active_cancel.set()
        active_task = self._active_task_by_key.get(key)
        try:
            current_task = asyncio.current_task()
        except RuntimeError:
            current_task = None
        if accepted and active_task is not None and active_task is not current_task:
            active_task.cancel()
        if accepted:
            self.generation_gate.cancel(key)
            self.presentation.reset(context=context, cancelled=True)
        sentence_state = self._sentence_turns.get(context)
        if sentence_state is not None:
            # 保留对象到 stream 的 finally，避免取消与句子提交并发时旧尾部
            # 再次进入展示/TTS；finally 再按回合状态回收整份状态。
            sentence_state.cancelled = True
        self._cancel_tts_turn(context)
        self._release_tts_context(context)
        for approval_id, paused in tuple(self._paused_turns.items()):
            if paused.context == context:
                self._paused_turns.pop(approval_id, None)
        if self.tools is not None:
            # 审批暂停时模型任务已经结束，但工具快照仍然存在；停止/替换
            # 回合必须同时消费该会话的待审批项，否则旧批准卡会阻塞热加载，
            # 并可能在新回合中留下不可执行的旧按钮。
            self.tools.clear_turn(
                ToolCallContext(
                    context.profile_id,
                    context.session_id,
                    context.turn_id,
                    source="model",
                    metadata={"mode": context.mode, "generation_id": context.generation_id},
                ),
                clear_pending=True,
            )

    def has_active_conversation(self) -> bool:
        """返回仍在进行或等待用户确认的对话回合。"""

        if self._active_tasks or self._active_cancels:
            return True
        if self.tools is None:
            return False
        try:
            # 审批暂停后模型任务已结束，但用户交互仍未完成；热替换应继续
            # 保守拒绝，避免配置切换与旧回合的确认状态交错。
            return bool(self.tools.pending_approvals())
        except Exception:
            # 无法可靠读取审批状态时按活动处理，保持 fail-safe。
            return True

    def pending_approvals_for_ui(self) -> tuple[object, ...]:
        """返回当前桌宠对话回合的审批，不混入调度器或其他会话。"""

        if self.tools is None:
            return ()
        paused = tuple(self._paused_turns.values())
        if not paused:
            return ()
        requests: list[object] = []
        for item in paused:
            context = ToolCallContext(
                item.context.profile_id,
                item.context.session_id,
                item.context.turn_id,
                source="model",
                metadata={"mode": item.context.mode, "generation_id": item.context.generation_id},
            )
            requests.extend(self.tools.pending_approvals_for_context(context))
        return tuple(requests)

    async def resume_approval(
        self,
        approval_id: str,
        *,
        grant_session: bool = False,
        context: ConversationContext | None = None,
    ) -> ToolOutcome:
        """消费审批并执行原始调用，但保持旧的“只执行工具”语义。

        Qt/Web 新入口应使用 :meth:`resume_approval_stream` 或
        :meth:`continue_approval`，这样工具结果才会回填到同一回合的模型。
        保留本方法的非续接语义，兼容脚本和无 UI 调用方。
        """

        async with self._approval_resume_lock:
            current, outcome, _event = await self._resume_approval_core(
                approval_id,
                grant_session=grant_session,
                context=context,
                preserve_paused=False,
            )
        # 该兼容入口只执行工具，不会继续模型回合；无论工具是否成功，
        # 原审批快照都已终止，必须释放语言/profile pin，避免一个被拒绝
        # 的审批永久占住路由。
        if current is not None:
            self._sentence_turns.pop(current, None)
            self._release_tts_context(current)
        return outcome

    async def _resume_approval_core(
        self,
        approval_id: str,
        *,
        grant_session: bool,
        context: ConversationContext | None,
        preserve_paused: bool,
    ) -> tuple[ConversationContext | None, ToolOutcome, ToolStatusChanged | None]:
        """在一个位置完成审批代际校验、原子消费和状态事件。"""

        if self._closed:
            raise RuntimeError("conversation service is closed")
        key = str(approval_id or "").strip()
        paused = self._paused_turns.get(key)
        if paused is not None and paused.approved_step_outcome is not None:
            current = paused.context
            outcome = paused.approved_step_outcome
            state = (
                outcome.status if outcome.status in {"completed", "denied", "failed"} else "failed"
            )
            event = ToolStatusChanged(
                current,
                cast(Literal["completed", "denied", "failed"], state),
                self._tool_outcome_summary(outcome),
                outcome.call_id or key,
            )
            return current, outcome, event
        if self.tools is None:
            return None, ToolOutcome("denied", "", "", {"error": "tools unavailable"}), None
        stored_context = self.tools.approval_context(key)
        current = context
        if current is None and stored_context is not None:
            metadata = stored_context.metadata
            try:
                generation_id = int(metadata.get("generation_id", 0))
            except (TypeError, ValueError):
                generation_id = 0
            current = ConversationContext(
                stored_context.profile_id,
                stored_context.session_id,
                stored_context.turn_id,
                generation_id,
                str(metadata.get("mode", "direct")),
            )
        if current is None:
            return (
                None,
                ToolOutcome("denied", "", "", {"error": "approval request is missing or expired"}),
                None,
            )
        if not self.generation_gate.accepts(current):
            pending = self.tools.pending_approval(key)
            stale_paused = self._paused_turns.pop(key, None)
            if stale_paused is not None:
                self.tools.clear_turn(
                    ToolCallContext(
                        stale_paused.context.profile_id,
                        stale_paused.context.session_id,
                        stale_paused.context.turn_id,
                        source="model",
                        metadata={
                            "mode": stale_paused.context.mode,
                            "generation_id": stale_paused.context.generation_id,
                        },
                    ),
                    clear_pending=True,
                )
            return (
                current,
                ToolOutcome(
                    "denied",
                    pending.identity if pending is not None else "",
                    pending.call_id if pending is not None else "",
                    {"error": "approval context is stale"},
                    pending,
                ),
                None,
            )
        approval_tool_context = ToolCallContext(
            current.profile_id,
            current.session_id,
            current.turn_id,
            source="model",
            metadata={"mode": current.mode, "generation_id": current.generation_id},
        )
        outcome = await self.tools.approve_and_execute(
            key,
            grant_session=grant_session,
            context=approval_tool_context,
        )
        state = outcome.status if outcome.status in {"completed", "denied", "failed"} else "failed"
        event = ToolStatusChanged(
            current,
            cast(Literal["completed", "denied", "failed"], state),
            self._tool_outcome_summary(outcome),
            outcome.call_id or key,
        )
        await self._emit(event)
        paused = self._paused_turns.get(key)
        if paused is not None and preserve_paused:
            paused.approved_step_outcome = outcome
            if paused.pending_plan is None:
                paused.approved_outcome = outcome
            if outcome.status != "completed":
                self._paused_turns.pop(key, None)
        elif paused is not None:
            self._paused_turns.pop(key, None)
        return current, outcome, event

    async def resume_approval_stream(
        self,
        approval_id: str,
        *,
        grant_session: bool = False,
        context: ConversationContext | None = None,
    ) -> AsyncIterator[object]:
        """批准工具或计划步骤后，从同一代际继续原模型回合。"""

        key = str(approval_id or "").strip()
        next_approval_event: ApprovalRequested | None = None
        # 审批恢复在进入 ``stream`` 前也可能执行一个耗时的工具（尤其是
        # 多步计划的已批准步骤）。将恢复协程登记到同一取消表，使桌面端
        # 的 ``cancel(context)`` 能在这段窗口内立即中断，而不是只能等
        # 工具完成后才生效。
        resume_task = asyncio.current_task()
        resume_cancel: asyncio.Event | None = None
        resume_cancel_key: ConversationKey | None = None
        resume_registered = False
        resume_cancel_owner = False
        resume_task_owner = False
        paused_before = self._paused_turns.get(key)
        resume_context = context or (paused_before.context if paused_before is not None else None)
        if resume_context is not None:
            resume_cancel_key = ConversationKey(
                resume_context.mode,
                resume_context.profile_id,
                resume_context.session_id,
            )
            existing_cancel = self._active_cancels.get(resume_cancel_key)
            resume_cancel = existing_cancel or asyncio.Event()
            if existing_cancel is None:
                self._active_cancels[resume_cancel_key] = resume_cancel
                resume_cancel_owner = True
            existing_task = self._active_task_by_key.get(resume_cancel_key)
            if resume_task is not None and (existing_task is None or existing_task.done()):
                self._active_task_by_key[resume_cancel_key] = resume_task
                self._active_tasks.add(resume_task)
                resume_task_owner = True
            resume_registered = True
        already_approved = paused_before is not None and paused_before.approved_outcome is not None
        try:
            async with self._approval_resume_lock:
                # 必须在串行审批锁内重新读取快照，否则两个并发批准回调
                # 都可能在获取锁前看到未批准状态，第二个回调会重复续接
                # 同一模型回合。
                paused_before = self._paused_turns.get(key)
                already_approved = (
                    paused_before is not None and paused_before.approved_outcome is not None
                )
                current, approved_step_outcome, status_event = await self._resume_approval_core(
                    key,
                    grant_session=grant_session,
                    context=context,
                    preserve_paused=True,
                )
                paused = self._paused_turns.get(key)
                outcome = approved_step_outcome
                if (
                    not already_approved
                    and outcome.status == "completed"
                    and paused is not None
                    and paused.pending_plan is not None
                ):
                    outcome, next_approval_event = await self._advance_paused_plan(
                        key,
                        paused,
                        approved_step_outcome,
                        cancel_event=resume_cancel,
                    )
                    if next_approval_event is not None:
                        # 计划仍在等待下一项审批。不能再发“外层计划已完成”终态，
                        # 否则 InteractionState 会把刚登记的下一审批清成完成态。
                        await self._emit(next_approval_event)
                    else:
                        status_event = ToolStatusChanged(
                            paused.context,
                            (
                                "failed"
                                if outcome.status == "failed"
                                else "denied"
                                if outcome.status == "denied"
                                else "completed"
                            ),
                            self._tool_outcome_summary(outcome),
                            paused.pending_call_id,
                        )
                        await self._emit(status_event)
            if status_event is not None:
                yield status_event
            elif outcome.status != "completed":
                yield outcome
            if next_approval_event is not None:
                yield next_approval_event
                return
            if not already_approved and outcome.status != "completed" and current is not None:
                self._paused_turns.pop(key, None)
                self._sentence_turns.pop(current, None)
                self._release_tts_context(current)
            if already_approved:
                return
            if outcome.status != "completed" or current is None:
                return
            paused = self._paused_turns.get(key)
            if paused is None:
                self._sentence_turns.pop(current, None)
                self._release_tts_context(current)
                return
            if paused.context != current or not self.generation_gate.accepts(current):
                self._paused_turns.pop(key, None)
                self._sentence_turns.pop(current, None)
                self._release_tts_context(current)
                return
            if not self._patch_paused_tool_outcome(paused, outcome):
                self._paused_turns.pop(key, None)
                self._sentence_turns.pop(current, None)
                self._release_tts_context(current)
                return
            async for event in self.stream(
                paused.user_text,
                context=paused.context,
                model=paused.model,
                channel_id=paused.channel_id,
                protocol=paused.protocol,
                _initial_messages=tuple(paused.messages),
                _continuation_state=paused,
                _preserve_presentation=True,
            ):
                yield event
        finally:
            self._paused_turns.pop(key, None)
            if resume_registered and resume_cancel_key is not None:
                if (
                    resume_task_owner
                    and self._active_task_by_key.get(resume_cancel_key) is resume_task
                ):
                    self._active_task_by_key.pop(resume_cancel_key, None)
                if (
                    resume_cancel_owner
                    and self._active_cancels.get(resume_cancel_key) is resume_cancel
                ):
                    self._active_cancels.pop(resume_cancel_key, None)
                if resume_task_owner and resume_task is not None:
                    self._active_tasks.discard(resume_task)

    async def continue_approval(
        self,
        approval_id: str,
        *,
        grant_session: bool = False,
        context: ConversationContext | None = None,
    ) -> ToolOutcome:
        """面向 Qt/Web 回调的续接便捷入口，消费完整续接事件流。"""

        paused = self._paused_turns.get(str(approval_id or "").strip())
        first: ToolOutcome | None = None
        continuation_failure: TurnFailed | None = None
        async for event in self.resume_approval_stream(
            approval_id,
            grant_session=grant_session,
            context=context,
        ):
            if isinstance(event, ToolOutcome) and first is None:
                first = event
            elif isinstance(event, ToolStatusChanged) and first is None:
                first = ToolOutcome(
                    event.state,
                    "",
                    event.call_id,
                    {"status": event.safe_summary},
                )
            elif isinstance(event, TurnFailed):
                continuation_failure = event
        if paused is not None and paused.approved_outcome is not None:
            if continuation_failure is not None:
                approved = paused.approved_outcome
                return ToolOutcome(
                    "failed",
                    approved.identity,
                    approved.call_id,
                    {
                        "error": continuation_failure.safe_message or "模型续接失败",
                        "continuation_status": "failed",
                    },
                    approved.approval,
                )
            return paused.approved_outcome
        return first or ToolOutcome(
            "denied", "", str(approval_id or ""), {"error": "approval request is unavailable"}
        )

    def deny_approval(
        self, approval_id: str, *, context: ConversationContext | None = None
    ) -> bool:
        """拒绝并消费审批请求，避免同一调用被重复提交。"""

        if self.tools is None:
            return False
        stored_context = self.tools.approval_context(approval_id)
        current = context
        if current is None and stored_context is not None:
            metadata = stored_context.metadata
            try:
                generation_id = int(metadata.get("generation_id", 0))
            except (TypeError, ValueError):
                generation_id = 0
            current = ConversationContext(
                stored_context.profile_id,
                stored_context.session_id,
                stored_context.turn_id,
                generation_id,
                str(metadata.get("mode", "direct")),
            )
        if current is not None and not self.generation_gate.accepts(current):
            return False
        request = self.tools.deny_approval(approval_id)
        if request is None:
            return False
        self._paused_turns.pop(str(approval_id or "").strip(), None)
        if current is not None:
            self._sentence_turns.pop(current, None)
            self._release_tts_context(current)
            # 拒绝事件沿与审批请求相同的上下文回到展示层。
            event = ToolStatusChanged(current, "denied", "用户拒绝了工具调用", request.call_id)
            self.presentation.consume(event)
            if self.event_sink is not None:
                try:
                    emission = self.event_sink(event)
                except Exception:
                    logger.debug("approval denial event sink failed", exc_info=True)
                else:
                    if inspect.isawaitable(emission):
                        try:
                            asyncio.create_task(emission)
                        except RuntimeError:
                            emission.close()
        return True

    async def _emit(self, event: object, *, consume: bool = True) -> None:
        if consume:
            self.presentation.consume(event)
        if self.event_sink is not None:
            result = self.event_sink(event)
            if inspect.isawaitable(result):
                await result

    def _tool_definitions(self) -> tuple[ToolDefinition, ...]:
        if self.tools is None:
            return ()
        specs = tuple(
            spec
            for spec in self.tools.registry.visible(self.tool_groups)
            if spec.identity != AGENT_PLAN_TOOL_IDENTITY
        )
        definitions = tuple(
            ToolDefinition(spec.identity, spec.description, spec.parameters) for spec in specs
        )
        if not definitions:
            return ()
        return (*definitions, agent_plan_tool_definition())

    def _history_messages(self, context: ConversationContext) -> list[ChatMessage]:
        if self.repository is None or self.history_limit <= 0:
            return []
        turns = self.repository.list_recent(
            context.mode,
            context.profile_id,
            context.session_id,
            self.history_limit,
        )
        messages: list[ChatMessage] = []
        for turn in turns:
            if turn.turn_id == context.turn_id:
                continue
            if turn.user_text:
                messages.append(ChatMessage("user", turn.user_text))
            answer = "".join(
                str(entry.get("delta", ""))
                for entry in turn.segments
                if entry.get("kind") == "text"
            )
            if answer:
                messages.append(ChatMessage("assistant", answer))
        return messages[-self.history_limit * 2 :]

    def _memory_enabled(self) -> bool:
        """读取记忆策略开关；旧版替身未提供属性时保持启用。"""

        return self.memory is not None and bool(getattr(self.memory, "enabled", True))

    def _promote_user_memories(
        self,
        user_text: str,
        answer: str,
        status: str,
        source_id: int,
    ) -> int:
        """在成功回合末尾执行有界规则式记忆提升。

        提取器只接收用户原文，避免把模型输出当作事实；异常被隔离在记忆
        边界，不得让已完成的对话变成失败。候选数量和扫描长度由记忆服务
        自身策略限制，因此这里保持同步短路径，不阻塞模型增量流。
        """

        if status not in {"completed", "tool_limit"} or not str(answer or "").strip():
            return 0
        if not self._memory_enabled() or self.memory is None:
            return 0
        extractor = getattr(self.memory, "extract_and_promote", None)
        if not callable(extractor):
            return 0
        promoted_count = 0
        try:
            promoted = extractor(str(user_text or ""), source_ids=(int(source_id),))
        except Exception as exc:
            logger.warning("本地记忆提取失败：%s", type(exc).__name__)
        else:
            if isinstance(promoted, Sequence) and not isinstance(promoted, (str, bytes, bytearray)):
                promoted_count = len(promoted)
        try:
            enqueue_model_memory_extraction(self.memory, str(user_text or ""), int(source_id))
        except Exception as exc:
            logger.warning("模型记忆入队失败：%s", type(exc).__name__)
        return promoted_count

    async def _build_messages(
        self,
        context: ConversationContext,
        user_text: str,
        images: Sequence[ImageAttachment],
    ) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        system_parts = [self.system_prompt] if self.system_prompt else []
        if self._memory_enabled():
            try:
                async_builder = getattr(self.memory, "abuild_context_prompt", None)
                if callable(async_builder):
                    memory_prompt = await async_builder(user_text)
                else:
                    memory_prompt = self.memory.build_context_prompt(user_text)
            except (RuntimeError, TypeError, ValueError):
                memory_prompt = ""
            if memory_prompt:
                system_parts.append(memory_prompt)
        if system_parts:
            messages.append(ChatMessage("system", "\n\n".join(system_parts)))
        messages.extend(self._history_messages(context))
        content: str | tuple[Mapping[str, object], ...] = user_text
        if images:
            parts: list[Mapping[str, object]] = [{"type": "text", "text": user_text}]
            parts.extend(image.as_content_part() for image in images)
            content = tuple(parts)
        messages.append(ChatMessage("user", content))
        return messages

    @staticmethod
    def _segment(kind: str, value: str) -> dict[str, str]:
        return {"kind": kind, "delta": str(value or "")}

    @staticmethod
    def _json_safe(value: object) -> object:
        """把工具结果限制为可传给模型和 SQLite 的 JSON 值。"""

        if isinstance(value, bytes):
            if len(value) > 256 * 1024:
                return {"encoding": "base64", "bytes": len(value), "truncated": True}
            return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
        if isinstance(value, Mapping):
            return {str(key): ConversationService._json_safe(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [ConversationService._json_safe(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    @classmethod
    def _tool_message_content(
        cls, content: Mapping[str, Any]
    ) -> str | tuple[Mapping[str, object], ...]:
        """把截图工具结果转换为文本加内联图片，供视觉渠道直接消费。"""

        raw_data = content.get("data")
        raw_format = str(content.get("format", "")).strip().lower()
        if isinstance(raw_data, bytes) and raw_format in {"png", "jpeg", "jpg", "webp"}:
            media_type = "image/jpeg" if raw_format in {"jpeg", "jpg"} else f"image/{raw_format}"
            try:
                attachment = ImageAttachment(
                    media_type=media_type,
                    data=base64.b64encode(raw_data).decode("ascii"),
                    file_name=f"desktop-capture.{raw_format}",
                )
            except ValueError:
                attachment = None
            if attachment is not None:
                text = dict(content)
                text.pop("data", None)
                return (
                    {"type": "text", "text": json.dumps(cls._json_safe(text), ensure_ascii=False)},
                    attachment.as_content_part(),
                )
        return json.dumps(cls._json_safe(content), ensure_ascii=False)

    async def _enqueue_tts(
        self,
        context: ConversationContext,
        text: str,
        *,
        flush: bool = False,
        language: str | None = None,
        role: str | None = None,
    ) -> None:
        if self.tts is None or (not text and not flush):
            return
        selected_language = str(language or self.tts_language or "zh").strip() or "zh"
        selected_role = str(role if role is not None else self.tts_role or "").strip()
        arguments: dict[str, object] = {
            "language": selected_language,
            "mood": self.presentation.snapshot.rendered_mood,
            "flush": flush,
        }
        # 空角色不传新关键字，兼容旧的 TTS 替身/扩展；有角色时才
        # 启用角色路由能力并由新协调器严格校验。
        if selected_role:
            arguments["role"] = selected_role
        await self.tts.enqueue_text(context, text, **arguments)

    def _create_tts_turn(self, context: ConversationContext) -> _TTSTurnState:
        """为回合创建唯一语音工作协程，杜绝按文本增量创建任务。"""

        language = self._tts_language_snapshots.get(context)
        if not language:
            language = str(self.tts_language or "zh").strip() or "zh"
        role = self._tts_role_snapshots.get(context, self.tts_role)
        state = _TTSTurnState(context, language=language, role=role)
        state.capacity_wake.set()
        task = asyncio.create_task(self._run_tts_turn(state))
        state.task = task
        self._tts_turns[context] = state
        self._tts_tasks.add(task)
        task.add_done_callback(lambda finished: self._finish_tts_turn(state, finished))
        return state

    async def _await_flushed_tts_turn(self, context: ConversationContext) -> None:
        """等待兼容入口留下的已请求 flush worker 完整退出。"""

        state = self._tts_turns.get(context)
        if state is None or not state.flush_requested or state.task is None:
            return
        worker = state.task
        try:
            # 外层 stream 可能在等待上一回合的 flush 时被取消。直接 await
            # 会把取消传播给 worker，而下面的 CancelledError 处理又会把
            # 外层取消吞掉，导致调用方误以为回合正常结束。shield 保留
            # worker 的独立生命周期；只有确认是 worker 自身取消时才吞掉
            # 该异常，外层任务的取消必须继续向上传递。
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            worker_cancelled = worker.cancelled()
            outer_cancelled = bool(current_task is not None and current_task.cancelling())
            if worker_cancelled and not outer_cancelled:
                # 上一轮语音被取消时，继续让模型续接；取消墓碑仍由
                # TTSCoordinator 保留并过滤迟到音频。
                return
            raise
        finally:
            # 外层取消时 shield 会让 worker 继续运行，不能提前移除状态；
            # _finish_tts_turn 会在 worker 结束时完成回收。
            if worker.done() and self._tts_turns.get(context) is state:
                self._tts_turns.pop(context, None)

    def _finish_tts_turn(self, state: _TTSTurnState, task: asyncio.Task[object]) -> None:
        """回收已结束工作协程及其回合级资源。"""

        self._tts_tasks.discard(task)
        # worker 自身取消或异常结束时，也要唤醒可能正在等待缓冲容量的模型流。
        state.capacity_wake.set()
        # ``_run_tts_turn`` 已处理预期的后端异常；读取 exception 仍可防止
        # 未预期异常在事件循环关闭时打印 ``Task exception was never retrieved``。
        if not task.cancelled():
            try:
                task.exception()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug(
                    "speech worker finished with an unexpected error: %s", type(exc).__name__
                )
        if self._tts_turns.get(state.context) is state:
            self._tts_turns.pop(state.context, None)

    def _append_tts_text(self, state: _TTSTurnState, text: str) -> None:
        """把 delta 合并进前置缓冲和有序溢出队列，保持音频不丢失。"""

        value = str(text or "")
        if not value or state.cancelled or state.flush_requested:
            return
        # 溢出队列中已经存在更早到达的文本时，后续 delta 必须继续
        # 追加到队列尾部，不能回填前置缓冲，否则新文本会越过旧文本。
        if state.pending_chunks:
            state.pending_chunks.append(value)
            state.backlog_chars += len(value)
            if not state.overflow_logged:
                logger.info("speech backlog spilled to the ordered per-turn queue")
                state.overflow_logged = True
            return
        available = self._TTS_PENDING_CHAR_LIMIT - len(state.pending_text)
        if available > 0:
            prefix = value[:available]
            state.pending_text += prefix
            state.backlog_chars += len(prefix)
            value = value[available:]
        if value:
            # 前置缓冲只用于限制单次调度的内存占用；溢出部分进入同一
            # 回合的 FIFO 队列，绝不丢弃模型已经返回的文本。后端慢时
            # 队列会增长，但音频内容仍保持完整且顺序不变。
            state.pending_chunks.append(value)
            state.backlog_chars += len(value)
            if not state.overflow_logged:
                logger.info("speech backlog spilled to the ordered per-turn queue")
                state.overflow_logged = True

    def _new_sentence_segmenter(self) -> SpeechSegmenter:
        """创建当前回合的句子边界快照。

        展示和 TTS 必须共享同一套硬/软边界；配置在回合开始后改变时，
        只影响下一回合，避免同一答案中途换用另一套切分规则。
        """

        max_chars = self._TTS_DEFAULT_SEGMENT_CHARS
        hard_boundaries: str | None = None
        soft_boundaries: str | None = None
        if self.tts is not None:
            try:
                max_chars = int(getattr(self.tts, "segment_max_chars", max_chars))
            except (TypeError, ValueError, OverflowError):
                max_chars = self._TTS_DEFAULT_SEGMENT_CHARS
            hard_value = getattr(self.tts, "segment_hard_boundaries", None)
            soft_value = getattr(self.tts, "segment_soft_boundaries", None)
            hard_boundaries = None if hard_value is None else str(hard_value)
            soft_boundaries = None if soft_value is None else str(soft_value)
        try:
            # 宠物气泡与 TTS 必须使用同一段文本；展示上限较小时收紧
            # 分段上限，避免 UI 截断而语音仍朗读更长内容。
            display_limit = int(getattr(self.presentation, "max_text_length", max_chars))
            max_chars = min(max_chars, max(20, display_limit))
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            return SpeechSegmenter(
                max_chars=max_chars,
                hard_boundaries=hard_boundaries,
                soft_boundaries=soft_boundaries,
            )
        except (TypeError, ValueError, OverflowError):
            # 自定义 TTS 替身或旧配置暴露非法边界时，展示仍使用安全默认，
            # 不能因为语音增强配置损坏而丢失模型正文。
            return SpeechSegmenter(max_chars=self._TTS_DEFAULT_SEGMENT_CHARS)

    async def _commit_sentence_events(
        self,
        context: ConversationContext,
        sentences: Iterable[str],
        *,
        state: _SentenceTurnState,
        final: bool = False,
        forced: bool = False,
    ) -> None:
        """按完成句子顺序更新展示，不重复进入原始事件流。

        TTS 在每个原始 ``TextDelta`` 到达时已经排入独立 worker；这里仅
        提交共享分段器产出的展示事件。若在这里再次提交完整句，会把同一
        片段送入 TTS 两次，尤其是一个供应商包包含多句时更容易重复播放。
        """

        values = tuple(str(sentence or "") for sentence in sentences if str(sentence or "").strip())
        for index, value in enumerate(values):
            if state.cancelled or not self.generation_gate.accepts(context):
                raise AdapterCancelled("conversation context was superseded")
            state.sequence += 1
            event = SentenceReady(
                context,
                value,
                state.sequence,
                final=bool(final and index == len(values) - 1),
                forced=bool(forced),
            )
            # 先更新当前气泡，再让独立 worker 获得调度；外部 sink 即使较慢，
            # 首个句子的 TTS 也已经提交。
            self.presentation.commit_sentence(event)
            # 句子已经完成，即使自定义硬边界不在固定 boundary 集合中，也
            # 必须让展示事件在下一个原始模型事件前进入宿主队列。
            await asyncio.sleep(0)
            if state.cancelled or not self.generation_gate.accepts(context):
                raise AdapterCancelled("conversation context was superseded")

    async def _run_tts_turn(self, state: _TTSTurnState) -> None:
        """串行消费一个回合的合并文本，并在结束时冲刷分段器。"""

        while not state.cancelled:
            await state.wake.wait()
            state.wake.clear()
            while not state.cancelled:
                if not state.pending_text and state.pending_chunks:
                    state.pending_text = state.pending_chunks.popleft()
                if not state.pending_text:
                    break
                text = state.pending_text[: self._TTS_DISPATCH_CHAR_LIMIT]
                state.pending_text = state.pending_text[self._TTS_DISPATCH_CHAR_LIMIT :]
                state.backlog_chars = max(0, state.backlog_chars - len(text))
                if state.backlog_chars <= self._TTS_BACKLOG_CHAR_LIMIT:
                    state.capacity_wake.set()
                try:
                    await self._enqueue_tts(
                        state.context,
                        text,
                        language=state.language,
                        role=state.role,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("speech dispatch failed: %s", type(exc).__name__)
            if state.cancelled:
                return
            if state.flush_requested:
                try:
                    await self._enqueue_tts(
                        state.context,
                        "",
                        flush=True,
                        language=state.language,
                        role=state.role,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("speech flush failed: %s", type(exc).__name__)
                return
        # 被取消时也必须唤醒正在等待容量的模型流，避免取消路径遗留挂起任务。
        state.capacity_wake.set()

    async def _await_tts_backpressure(self, context: ConversationContext) -> None:
        """在语音待发缓冲达到上限时等待 worker 消费。

        该等待只约束模型流的生产速率；TTS worker 仍独立运行。回合取消或
        关闭会设置同一个事件，因此不会把取消变成永久等待。
        """

        state = self._tts_turns.get(context)
        if state is None:
            return
        while (
            state.backlog_chars >= self._TTS_BACKLOG_CHAR_LIMIT
            and not state.cancelled
            and not self._closed
        ):
            # 完成回调只唤醒一次；消费者已退出时必须传递其终态，
            # 不能清掉通知后继续等待永远不会释放的容量。
            if state.task is not None and state.task.done():
                await state.task
                return
            state.capacity_wake.clear()
            if (
                state.backlog_chars < self._TTS_BACKLOG_CHAR_LIMIT
                or state.cancelled
                or self._closed
            ):
                break
            await state.capacity_wake.wait()

    async def _schedule_tts_with_backpressure(
        self, context: ConversationContext, text: str, *, flush: bool = False
    ) -> bool:
        """以有界片段提交增量，并在容量耗尽时异步等待 worker。

        ``_schedule_tts`` 保留给旧的同步扩展入口；模型流使用这个异步入口，
        因而即使供应商一次返回超长 delta，也不会把整段文本一次性压入内存。
        """

        if self.tts is None or self._closed or not text:
            return False
        value = str(text)
        should_yield = False
        while value:
            state = self._tts_turns.get(context)
            if state is None:
                state = self._create_tts_turn(context)
                should_yield = True
            if state.backlog_chars >= self._TTS_BACKLOG_CHAR_LIMIT:
                await self._await_tts_backpressure(context)
                if state.cancelled or self._closed:
                    return should_yield
                continue
            available = max(1, self._TTS_BACKLOG_CHAR_LIMIT - state.backlog_chars)
            piece, value = value[:available], value[available:]
            should_yield = self._schedule_tts(context, piece) or should_yield
        state = self._tts_turns.get(context)
        if flush and state is not None:
            state.flush_requested = True
            state.wake.set()
        return should_yield

    def _schedule_tts(
        self, context: ConversationContext, text: str, *, flush: bool = False
    ) -> bool:
        """提交文本增量，并返回是否应让出一次事件循环。

        返回值只表示“worker 至少应获得一次调度机会”，不表示等待 TTS
        合成完成。首个回合或达到分段阈值的增量会触发让出，使已完成
        的首段在当前 TextDelta 返回给调用方前进入后端；后续合成仍由
        独立 worker 负责，不会把模型流绑定到音频耗时。
        """

        if self.tts is None or self._closed or not text:
            return False
        state = self._tts_turns.get(context)
        created = state is None
        state = state or self._create_tts_turn(context)
        self._append_tts_text(state, text)
        if flush:
            state.flush_requested = True
        state.wake.set()
        value = str(text or "")
        try:
            segment_limit = int(
                getattr(self.tts, "segment_max_chars", self._TTS_DEFAULT_SEGMENT_CHARS)
            )
        except (TypeError, ValueError, OverflowError):
            segment_limit = self._TTS_DISPATCH_CHAR_LIMIT
        yield_threshold = max(1, min(segment_limit, self._TTS_DISPATCH_CHAR_LIMIT))
        return (
            created
            or len(value) >= self._TTS_DISPATCH_CHAR_LIMIT
            or len(state.pending_text) >= yield_threshold
            or any(character in self._TTS_BOUNDARY_CHARS for character in value)
        )

    async def _flush_tts(self, context: ConversationContext) -> None:
        if self.tts is None:
            return
        state = self._tts_turns.get(context)
        if state is None:
            await self._enqueue_tts(context, "", flush=True)
            return
        if state.cancelled:
            return
        state.flush_requested = True
        state.wake.set()
        task = state.task
        if task is not None:
            await task

    def _cancel_tts_turn(self, context: ConversationContext) -> None:
        """同步丢弃回合缓存并中断其唯一语音工作协程。"""

        state = self._tts_turns.get(context)
        if state is not None:
            state.cancelled = True
            state.pending_text = ""
            state.pending_chunks.clear()
            state.backlog_chars = 0
            state.flush_requested = False
            state.capacity_wake.set()
            state.wake.set()
            if state.task is not None:
                state.task.cancel()
        if self.tts is not None:
            self.tts.cancel(context)

    def _release_tts_context(self, context: ConversationContext) -> None:
        """释放回合的语言、角色和 profile 快照并通知 TTS 协调器回收路由。"""

        self._tts_language_snapshots.pop(context, None)
        self._tts_role_snapshots.pop(context, None)
        if self.tts is None:
            return
        release_context = getattr(self.tts, "release_context", None)
        if not callable(release_context):
            return
        try:
            release_context(context)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug("speech context release failed", exc_info=True)

    def _visible_plan_identities(self) -> frozenset[str]:
        if self.tools is None:
            return frozenset()
        return frozenset(
            spec.identity
            for spec in self.tools.registry.visible(self.tool_groups)
            if spec.identity != AGENT_PLAN_TOOL_IDENTITY
        )

    async def _execute_agent_plan(
        self,
        call: ToolCall,
        tool_context: ToolCallContext,
        *,
        cancel_event: asyncio.Event | None,
    ) -> tuple[ToolOutcome, ToolPlanResult | None]:
        """解析模型计划并交给统一工具执行器。"""

        if self.tools is None:
            return (
                ToolOutcome(
                    "denied",
                    AGENT_PLAN_TOOL_IDENTITY,
                    call.call_id,
                    {"error": "tools unavailable"},
                ),
                None,
            )
        try:
            plan = parse_agent_plan_call(
                call,
                allowed_identities=self._visible_plan_identities(),
            )
            result = await self.tools.execute_plan(
                plan.steps,
                context=tool_context,
                timeout_seconds=plan.timeout_seconds,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            raise
        except (OverflowError, TypeError, ValueError):
            return (
                ToolOutcome(
                    "denied",
                    AGENT_PLAN_TOOL_IDENTITY,
                    call.call_id,
                    {"error": "invalid agent plan"},
                ),
                None,
            )
        return agent_plan_outcome(call.call_id, result), result

    @classmethod
    def _patch_paused_tool_outcome(
        cls,
        paused: _PausedTurn,
        outcome: ToolOutcome,
    ) -> bool:
        """把审批后的最终结果替入原 assistant 工具调用对应消息。"""

        patched = False
        for index, message in enumerate(paused.messages):
            if message.role != "tool" or message.tool_call_id != paused.pending_call_id:
                continue
            paused.messages[index] = ChatMessage(
                "tool",
                cls._tool_message_content(outcome.content),
                tool_call_id=message.tool_call_id,
                tool_name=message.tool_name,
            )
            patched = True
            break
        for index, previous in enumerate(paused.outcomes):
            if previous.call_id == paused.pending_call_id:
                paused.outcomes[index] = outcome
                break
        return patched

    async def _advance_paused_plan(
        self,
        approval_id: str,
        paused: _PausedTurn,
        approved_outcome: ToolOutcome,
        *,
        cancel_event: asyncio.Event | None = None,
    ) -> tuple[ToolOutcome, ApprovalRequested | None]:
        """批准内部步骤后继续计划；下一审批会原子迁移暂停键。"""

        pending_plan = paused.pending_plan
        checkpoint = pending_plan.checkpoint if pending_plan is not None else None
        if self.tools is None or checkpoint is None:
            failed = ToolOutcome(
                "failed",
                AGENT_PLAN_TOOL_IDENTITY,
                paused.pending_call_id,
                {"error": "agent plan checkpoint is unavailable"},
            )
            paused.pending_plan = None
            paused.approved_outcome = failed
            self._patch_paused_tool_outcome(paused, failed)
            return failed, None
        tool_context = ToolCallContext(
            paused.context.profile_id,
            paused.context.session_id,
            paused.context.turn_id,
            source="model",
            metadata={
                "mode": paused.context.mode,
                "generation_id": paused.context.generation_id,
            },
        )
        try:
            result = await self.tools.resume_plan(
                checkpoint,
                approved_outcome,
                context=tool_context,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            raise
        except (RuntimeError, TypeError, ValueError):
            failed = ToolOutcome(
                "failed",
                AGENT_PLAN_TOOL_IDENTITY,
                paused.pending_call_id,
                {"error": "agent plan continuation failed"},
            )
            paused.pending_plan = None
            paused.approved_outcome = failed
            self._patch_paused_tool_outcome(paused, failed)
            return failed, None

        outcome = agent_plan_outcome(paused.pending_call_id, result)
        if not self._patch_paused_tool_outcome(paused, outcome):
            failed = ToolOutcome(
                "failed",
                AGENT_PLAN_TOOL_IDENTITY,
                paused.pending_call_id,
                {"error": "agent plan message checkpoint is unavailable"},
            )
            paused.pending_plan = None
            paused.approved_outcome = failed
            return failed, None
        next_approval = result.pending_approval
        if next_approval is None:
            paused.pending_plan = None
            paused.approved_outcome = outcome
            return outcome, None

        paused.pending_plan = result
        paused.pending_approval_id = next_approval.approval_id
        paused.approved_outcome = None
        paused.approved_step_outcome = None
        old_key = str(approval_id or "").strip()
        if self._paused_turns.get(old_key) is paused:
            self._paused_turns.pop(old_key, None)
        self._paused_turns[next_approval.approval_id] = paused
        event = ApprovalRequested(
            paused.context,
            next_approval.approval_id,
            next_approval.call_id,
            next_approval.safe_summary,
            next_approval.expires_at,
        )
        return outcome, event

    async def _tool_events(
        self,
        context: ConversationContext,
        call: ToolCall,
        tool_context: ToolCallContext,
        *,
        emit_events: bool = True,
    ) -> tuple[ToolOutcome, tuple[object, ...], ToolPlanResult | None]:
        emitted: list[object] = []

        async def emit(event: object) -> None:
            emitted.append(event)
            if emit_events:
                await self._emit(event)

        if self.tools is None:
            outcome = ToolOutcome(
                "denied", call.identity, call.call_id, {"error": "tools unavailable"}
            )
            await emit(ToolStatusChanged(context, "denied", "工具运行时不可用", call.call_id))
            return outcome, tuple(emitted), None
        spec = self.tools.registry.get(call.identity)
        label = getattr(spec, "user_label", None) or tool_user_label(call.identity)
        await emit(ToolCallStarted(context, call.call_id, call.identity, f"准备执行：{label}"))
        await emit(ToolStatusChanged(context, "running", f"正在执行：{label}", call.call_id))
        if call.identity == AGENT_PLAN_TOOL_IDENTITY:
            outcome, plan_result = await self._execute_agent_plan(
                call,
                tool_context,
                cancel_event=self._active_cancels.get(
                    ConversationKey(context.mode, context.profile_id, context.session_id)
                ),
            )
        else:
            plan_result = None
            outcome = await self.tools.execute(
                call_id=call.call_id,
                identity=call.identity,
                arguments=call.arguments,
                context=tool_context,
            )
        if outcome.status == "approval_required" and outcome.approval is not None:
            await emit(
                ApprovalRequested(
                    context,
                    outcome.approval.approval_id,
                    outcome.approval.call_id,
                    outcome.approval.safe_summary,
                    outcome.approval.expires_at,
                )
            )
        else:
            state = cast(
                Literal["completed", "denied", "failed"],
                "completed"
                if outcome.status == "duplicate"
                else outcome.status
                if outcome.status in {"completed", "denied", "failed"}
                else "failed",
            )
            await emit(
                ToolStatusChanged(
                    context,
                    state,
                    self._tool_outcome_summary(outcome),
                    call.call_id,
                )
            )
        return outcome, tuple(emitted), plan_result

    def _can_parallel_tool_calls(
        self,
        calls: Sequence[ToolCall],
        context: ToolCallContext,
    ) -> bool:
        """只为已获无审批资格的低风险只读调用开启并行。"""

        if self.tools is None or len(calls) < 2:
            return False
        permissions = getattr(self.tools, "permissions", None)
        probe = getattr(permissions, "allows_without_approval", None)
        if not callable(probe):
            return False
        for call in calls:
            spec = self.tools.registry.get(call.identity)
            if spec is None or not bool(getattr(spec, "read_only", False)):
                return False
            if getattr(spec, "risk", None) is not RiskLevel.LOW:
                return False
            try:
                if not bool(probe(spec, context)):
                    return False
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
        return True

    async def stream(
        self,
        user_text: str,
        *,
        context: ConversationContext | None = None,
        profile_id: str = "default",
        session_id: str = "local",
        mode: str = "direct",
        model: str | None = None,
        channel_id: str | None = None,
        protocol: str | None = None,
        images: Sequence[ImageAttachment] = (),
        cancel_event: asyncio.Event | None = None,
        _initial_messages: Sequence[ChatMessage] | None = None,
        _continuation_state: _PausedTurn | None = None,
        _preserve_presentation: bool = False,
    ) -> AsyncIterator[object]:
        """流式执行一个回合；每个核心事件都会先送入展示层再交给调用方。"""

        if self._closed:
            raise RuntimeError("conversation service is closed")
        text_input = str(user_text or "").strip()
        if _continuation_state is not None:
            text_input = _continuation_state.user_text
        if not text_input or len(text_input) > 20000:
            raise ValueError("user_text must contain 1 to 20000 characters")
        current = context or self.begin_context(
            profile_id=profile_id,
            session_id=session_id,
            mode=mode,
        )
        if not self.generation_gate.accepts(current):
            raise RuntimeError("conversation context is stale")
        # 兼容入口可能已经请求上一批语音 flush；续接同一 context 前
        # 等待旧 worker 退出，避免新文本被 flush_requested 状态吞掉。
        await self._await_flushed_tts_turn(current)
        if self.tts is not None:
            # 在回合真正开始时固定语言和角色；配置面随后切换只影响新回合，
            # 不会在首个 TTS 请求尚未发出或审批续接时改变当前回合音色。
            snapshot_language = self._tts_language_snapshots.setdefault(
                current,
                str(self.tts_language or "zh").strip() or "zh",
            )
            snapshot_role = self._tts_role_snapshots.setdefault(
                current,
                str(self.tts_role or "").strip(),
            )
            pin_context = getattr(self.tts, "pin_context", None)
            if callable(pin_context):
                try:
                    pin_context(current, language=snapshot_language, role=snapshot_role)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    # 路由快照是增强能力；TTS 本身仍须遵循文本降级，
                    # 不能因为 profile 读取失败而中断模型输出。
                    logger.debug("speech context pin failed", exc_info=True)
        cancel_key = ConversationKey(current.mode, current.profile_id, current.session_id)
        active_cancel = cancel_event or self._active_cancels.get(cancel_key) or asyncio.Event()
        self._active_cancels[cancel_key] = active_cancel
        current_task = asyncio.current_task()
        if current_task is not None:
            self._active_tasks.add(current_task)
            self._active_task_by_key[cancel_key] = current_task
        # 在首个 provider 事件到达前先绑定当前代际；旧回合已经排队的 SSE
        # 事件不能在新回合启动窗口内覆盖气泡。
        if not _preserve_presentation:
            self.presentation.reset(context=current)
        segments: list[dict[str, str]] = list(
            _continuation_state.segments if _continuation_state is not None else ()
        )
        reasoning = _continuation_state.reasoning if _continuation_state is not None else ""
        murmur = _continuation_state.murmur if _continuation_state is not None else ""
        answer = _continuation_state.answer if _continuation_state is not None else ""
        # 增量文本先进入列表，避免长流中每个 token 都复制完整回答。
        answer_parts: list[str] = [answer] if answer else []
        outcomes: list[ToolOutcome] = list(
            _continuation_state.outcomes if _continuation_state is not None else ()
        )
        usage: dict[str, int] = dict(
            _continuation_state.usage if _continuation_state is not None else {}
        )
        finish_reason = (
            _continuation_state.finish_reason if _continuation_state is not None else "stop"
        )
        status = "completed"
        tts_flushed = False
        sentence_state: _SentenceTurnState | None = None
        messages: list[ChatMessage] = list(_initial_messages or ())
        definitions: tuple[ToolDefinition, ...] = (
            _continuation_state.definitions if _continuation_state is not None else ()
        )
        tool_context = ToolCallContext(
            current.profile_id,
            current.session_id,
            current.turn_id,
            metadata={"mode": current.mode, "generation_id": current.generation_id},
        )
        try:
            if not messages:
                messages = await self._build_messages(current, text_input, images)
                # 语义上下文可能在线程中等待模型；回到事件循环后必须再次
                # 检查回合代际/取消状态，迟到结果不得继续创建旧轮次或覆盖新回合。
                if active_cancel.is_set() or not self.generation_gate.accepts(current):
                    raise AdapterCancelled("conversation context was superseded")
            if not definitions:
                definitions = self._tool_definitions()
            if self.repository is not None and _continuation_state is None:
                self.repository.create(
                    mode=current.mode,
                    profile_id=current.profile_id,
                    session_id=current.session_id,
                    turn_id=current.turn_id,
                    source="user",
                    user_text=text_input,
                )
            first_round = (
                _continuation_state.next_round_index if _continuation_state is not None else 0
            )
            sentence_state = self._sentence_turns.get(current)
            if sentence_state is None:
                sentence_state = _SentenceTurnState(self._new_sentence_segmenter())
                self._sentence_turns[current] = sentence_state
            for round_index in range(first_round, self.max_tool_rounds + 1):
                round_text_parts: list[str] = []
                explicit_model = str(model or "").strip() or None
                # 显式指定频道时，频道本身优先于全局 default_model；否则全局
                # 模型仍作为默认约束。运行时通过 router.select_model() 选择
                # 的路由模型优先于旧的全局 default_model，确保控制台切换后
                # 下一轮真的使用新模型。
                route = self.router.route(self.task)
                routed_model = str(getattr(route, "model", "") or "").strip() or None
                route_model = explicit_model or (
                    routed_model or self.model if channel_id is None else None
                )
                selection = self.router.resolve(
                    self.task,
                    model=route_model,
                    channel_id=channel_id,
                    protocol=protocol,
                    required_capabilities=("streaming",),
                )
                request_model = str(
                    explicit_model
                    or (routed_model or self.model if channel_id is None else None)
                    or selection.primary.selected_model
                ).strip()
                if not request_model:
                    raise ModelRoutingError(
                        f"selected channel has no model for task: {self.task}; "
                        "configure llm.channels[].model or llm.default_model"
                    )
                request = ChatRequest(
                    model=request_model,
                    messages=tuple(messages),
                    tools=definitions,
                    tool_choice="auto" if definitions else "none",
                )
                response = ProviderResponse()
                calls_seen: set[str] = set()
                async for provider_event in self.router.stream(
                    request,
                    task=self.task,
                    context=current,
                    model=route_model,
                    channel_id=channel_id,
                    protocol=protocol,
                    required_capabilities=("streaming",),
                    cancel_event=active_cancel,
                ):
                    if not self.generation_gate.accepts(current):
                        raise AdapterCancelled("conversation context was superseded")
                    # 文本/推理/碎碎念已经分别由 answer/reasoning/murmur
                    # 累积；不再让不可变 ProviderResponse 每个增量复制
                    # 全部字符串，避免长流形成 O(n²) 内存复制。
                    if not isinstance(provider_event, (TextDelta, ReasoningDelta, MurmurDelta)):
                        response = response.add(provider_event)
                    if isinstance(provider_event, TextDelta):
                        answer_parts.append(provider_event.delta)
                        round_text_parts.append(provider_event.delta)
                        segments.append(self._segment("text", provider_event.delta))
                        # 先同步更新展示快照（TTS 需要读取当前 mood），再
                        # 让独立 worker 获得调度；外部 event_sink 留到之后，
                        # 避免跨线程/慢速 UI 回调挡住首句转换。
                        self.presentation.consume(provider_event)
                        # 每个有效文本事件都先进入 TTS worker。worker 内部仍
                        # 使用同一套有界分段器，因此半句会继续缓冲，遇到句末、
                        # 段落或长度上限即可开始合成；不必等到本回合终态才把
                        # 首个无标点事件提交给语音链路。
                        if await self._schedule_tts_with_backpressure(
                            current, provider_event.delta
                        ):
                            await asyncio.sleep(0)
                        await self._commit_sentence_events(
                            current,
                            sentence_state.segmenter.push(provider_event.delta),
                            state=sentence_state,
                        )
                        # 先把文本交给独立的 TTS worker，再通知可能较慢的
                        # 展示层。Qt/Web 的 event_sink 可能等待跨线程渲染或
                        # 队列背压；如果顺序反过来，首句语音会被 UI 回调
                        # 阻塞，无法满足“模型返回阶段即开始转换”。
                        await self._emit(provider_event, consume=False)
                        yield provider_event
                    elif isinstance(provider_event, ReasoningDelta):
                        reasoning += provider_event.delta
                        segments.append(self._segment("reasoning", provider_event.delta))
                        self.presentation.consume(provider_event)
                        # 只有显式开启思考展示时才把思考内容作为碎碎念送入
                        # TTS，默认不泄漏内部推理文本。
                        if self.presentation.show_reasoning:
                            if await self._schedule_tts_with_backpressure(
                                current, provider_event.delta
                            ):
                                await asyncio.sleep(0)
                        await self._emit(provider_event, consume=False)
                        yield provider_event
                    elif isinstance(provider_event, MurmurDelta):
                        murmur += provider_event.delta
                        segments.append(self._segment("murmur", provider_event.delta))
                        self.presentation.consume(provider_event)
                        if await self._schedule_tts_with_backpressure(
                            current, provider_event.delta
                        ):
                            await asyncio.sleep(0)
                        await self._emit(provider_event, consume=False)
                        yield provider_event
                    elif isinstance(provider_event, ToolCallDelta):
                        if provider_event.call_id or provider_event.identity:
                            call_key = provider_event.call_id or f"index-{provider_event.index}"
                            if call_key not in calls_seen:
                                calls_seen.add(call_key)
                                await self._emit(
                                    ToolCallStarted(
                                        current,
                                        provider_event.call_id or call_key,
                                        provider_event.identity or "tool:unknown",
                                        "模型正在准备工具调用",
                                    )
                                )
                        # 工具参数是供应商协议的内部增量，不能进入 Qt/Web
                        # 展示流；完整调用会在本地校验、审批后才以安全状态事件
                        # 暴露给宿主。
                        continue
                    elif isinstance(provider_event, AdapterMetadata):
                        segments.append(
                            {
                                "kind": "metadata",
                                "value": json.dumps(
                                    dict(provider_event.values), ensure_ascii=False
                                ),
                            }
                        )
                        yield provider_event
                    elif isinstance(provider_event, TurnFinished):
                        usage.update(provider_event.usage)
                        finish_reason = provider_event.finish_reason
                        # 提供方每一轮都会结束；对外只在工具循环完全结束后发一次。
                        continue
                # 仅在每轮结束时拼接一次，保持长流增量处理为线性复杂度。
                round_text = "".join(round_text_parts)
                answer = "".join(answer_parts)
                finalized = response.finalized()
                invalid_tool_calls = finalized.metadata.get("invalid_tool_calls", ())
                if invalid_tool_calls:
                    raise ProviderAdapterError(
                        "provider returned invalid tool arguments",
                        category="protocol",
                        retryable=False,
                    )
                calls = finalized.tool_calls
                if not calls:
                    # 正常回合在最终终态前冲刷没有句末标点的尾部；该尾部
                    # 仍只交给句子展示/TTS 一次，随后才发送统一 TurnFinished。
                    await self._commit_sentence_events(
                        current,
                        sentence_state.segmenter.flush(),
                        state=sentence_state,
                        final=True,
                    )
                    break
                if round_index >= self.max_tool_rounds:
                    # 工具轮达到上限时也不能静默丢掉已经返回的正文；按强制
                    # 分段提交尾部，但不把它标记成最终正常句。
                    await self._commit_sentence_events(
                        current,
                        sentence_state.segmenter.flush(),
                        state=sentence_state,
                        forced=True,
                    )
                    status = "tool_limit"
                    break
                assistant_content = round_text or None
                messages.append(ChatMessage("assistant", assistant_content, tool_calls=calls))
                round_outcomes: list[ToolOutcome] = []
                round_plan_results: dict[str, ToolPlanResult] = {}
                if self._can_parallel_tool_calls(calls, tool_context):
                    # 观察工具没有副作用且已由权限服务明确允许；并行执行
                    # 可减少“读取前台+进程列表”这类组合调用的等待。事件
                    # 在 gather 完成后按模型原始顺序回放，展示层不会乱序。
                    # 先同步展示“准备/运行中”，避免慢速桌面读取期间控制台
                    # 看起来像没有响应；完成/失败事件仍按原始调用顺序回放。
                    for call in calls:
                        spec = self.tools.registry.get(call.identity) if self.tools else None
                        label = getattr(spec, "user_label", None) or tool_user_label(call.identity)
                        progress_events = (
                            ToolCallStarted(
                                current,
                                call.call_id,
                                call.identity,
                                f"准备执行：{label}",
                            ),
                            ToolStatusChanged(
                                current,
                                "running",
                                f"正在执行：{label}",
                                call.call_id,
                            ),
                        )
                        for progress_event in progress_events:
                            await self._emit(progress_event)
                            yield progress_event
                    parallel_results = await asyncio.gather(
                        *(
                            self._tool_events(
                                current,
                                call,
                                tool_context,
                                emit_events=False,
                            )
                            for call in calls
                        )
                    )
                    for outcome, tool_events, plan_result in parallel_results:
                        round_outcomes.append(outcome)
                        if plan_result is not None:
                            round_plan_results[outcome.call_id] = plan_result
                        for tool_event in tool_events:
                            if isinstance(tool_event, ToolCallStarted) or (
                                isinstance(tool_event, ToolStatusChanged)
                                and tool_event.state == "running"
                            ):
                                continue
                            await self._emit(tool_event)
                            yield tool_event
                else:
                    for call in calls:
                        outcome, tool_events, plan_result = await self._tool_events(
                            current,
                            call,
                            tool_context,
                        )
                        round_outcomes.append(outcome)
                        if plan_result is not None:
                            round_plan_results[outcome.call_id] = plan_result
                        for tool_event in tool_events:
                            yield tool_event
                        # 一个回合只允许挂起一个审批。继续执行同一响应里的后续
                        # 副作用调用会让用户无法知道批准按钮对应哪一项，也会在
                        # 旧审批尚未处理时产生不可逆动作。
                        if outcome.status == "approval_required":
                            break
                if any(outcome.status == "approval_required" for outcome in round_outcomes):
                    # OpenAI 兼容协议要求 assistant tool_calls 的每个 call_id 都有
                    # 对应 tool message。首个审批挂起时，后续副作用调用不执行，
                    # 但仍写入安全的“未执行”结果，续接时模型不会收到残缺消息。
                    for skipped_call in calls[len(round_outcomes) :]:
                        skipped = ToolOutcome(
                            "denied",
                            skipped_call.identity,
                            skipped_call.call_id,
                            {"error": "前一项操作等待确认，后续操作未执行"},
                        )
                        round_outcomes.append(skipped)
                outcomes.extend(round_outcomes)
                for call, outcome in zip(calls, round_outcomes, strict=False):
                    messages.append(
                        ChatMessage(
                            "tool",
                            self._tool_message_content(outcome.content),
                            tool_call_id=call.call_id,
                            tool_name=call.identity,
                        )
                    )
                    if outcome.status == "approval_required":
                        status = "approval_required"
                        # 即使首个调用等待审批，也要把同一 assistant
                        # 响应中其余调用的安全“未执行”结果写入消息序列，
                        # 保持每个 tool_call_id 都有对应 tool message。
                        # 这里不能提前 break，否则续接请求会丢失后续
                        # call_id，部分 OpenAI 兼容服务会拒绝该消息序列。
                if status == "approval_required":
                    pending_call = next(
                        (
                            call
                            for call, outcome in zip(calls, round_outcomes, strict=False)
                            if outcome.status == "approval_required"
                            and outcome.approval is not None
                        ),
                        None,
                    )
                    pending_outcome = next(
                        (
                            outcome
                            for outcome in round_outcomes
                            if outcome.status == "approval_required"
                        ),
                        None,
                    )
                    if pending_call is not None and pending_outcome is not None:
                        approval = pending_outcome.approval
                        if approval is not None:
                            self._paused_turns[approval.approval_id] = _PausedTurn(
                                context=current,
                                user_text=text_input,
                                messages=list(messages),
                                definitions=definitions,
                                next_round_index=round_index + 1,
                                pending_call_id=pending_call.call_id,
                                pending_approval_id=approval.approval_id,
                                segments=list(segments),
                                reasoning=reasoning,
                                murmur=murmur,
                                answer=answer,
                                outcomes=list(outcomes),
                                usage=dict(usage),
                                finish_reason=finish_reason,
                                # 续接必须锁定本次实际选中的渠道/模型，不能
                                # 因健康排序变化而把批准结果送到另一渠道。
                                model=request_model,
                                channel_id=channel_id or str(selection.primary.id),
                                protocol=protocol,
                                pending_plan=round_plan_results.get(pending_call.call_id),
                                memory_user_chat_id=(
                                    _continuation_state.memory_user_chat_id
                                    if _continuation_state is not None
                                    else None
                                ),
                            )
                    break
            await self._flush_tts(current)
            tts_flushed = True
            # 外部停止请求与 TTS worker 的取消可能同时抵达；某些异步适配器
            # 会先消费掉 task.cancel() 再返回。终态写入前再次检查代际和取消
            # 事件，确保这种竞态不会把已取消回合误报为 completed。
            if active_cancel.is_set():
                raise asyncio.CancelledError
            if not self.generation_gate.accepts(current):
                raise AdapterCancelled("conversation context was superseded")
            # 先更新好感度的每日会话计数；MemoryService 写入 assistant 历史时
            # 会同步 total_chats，保持两个服务共享状态时不会重复累计。
            # 审批暂停的首段不能提前计入会话；续接最终完成时再计入一次。
            # 普通回合仍沿用同一分支，因此不会因为 continuation 重复累计。
            if self.affection is not None and status != "approval_required":
                self.affection.mark_today_chatted(increment_total=False)
                self.affection.adjust(1)
            if self._memory_enabled():
                if _continuation_state is None:
                    user_chat_id = self.memory.add_chat("user", text_input)
                    if status == "approval_required":
                        for paused_state in self._paused_turns.values():
                            if paused_state.context == current:
                                paused_state.memory_user_chat_id = int(user_chat_id)
                    if answer:
                        self.memory.add_chat("assistant", answer)
                        self.memory.store_chat_exchange(text_input, answer)
                        self._promote_user_memories(
                            text_input,
                            answer,
                            status,
                            int(user_chat_id),
                        )
                    self.memory.increment_message_counter()
                else:
                    previous_answer = _continuation_state.answer
                    appended_answer = (
                        answer[len(previous_answer) :]
                        if answer.startswith(previous_answer)
                        else answer
                    )
                    if appended_answer:
                        self.memory.add_chat("assistant", appended_answer)
                    if answer and _continuation_state.memory_user_chat_id is not None:
                        self._promote_user_memories(
                            _continuation_state.user_text,
                            answer,
                            status,
                            _continuation_state.memory_user_chat_id,
                        )
            finished = TurnFinished(current, finish_reason, usage)
            result = ConversationResult(
                current,
                answer,
                reasoning,
                murmur,
                tuple(outcomes),
                finish_reason,
                dict(usage),
                status,
            )
            if len(self._results) >= 256:
                self._results.pop(next(iter(self._results)))
            self._results[
                (
                    current.mode,
                    current.profile_id,
                    current.session_id,
                    current.turn_id,
                    current.generation_id,
                )
            ] = result
            await self._emit(finished)
            yield finished
        except AdapterCancelled:
            status = "cancelled"
            raise
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except ModelRoutingError as exc:
            status = "failed"
            # 路由异常的原文可能包含 ``llm.channels``、环境变量名或内部
            # 路由约束；这些信息只留在路由诊断/日志边界，不能进入气泡、
            # 控制台或 Web 状态。
            failed = TurnFailed(
                current,
                "configuration",
                str(getattr(exc, "safe_message", "模型渠道未就绪，请点击“配置模型”检查连接信息")),
                False,
            )
            await self._emit(failed)
            yield failed
        except ProviderAdapterError as exc:
            status = "failed"
            failed = TurnFailed(current, exc.category, exc.safe_message, exc.retryable)
            await self._emit(failed)
            yield failed
        except Exception as exc:
            status = "failed"
            failed = TurnFailed(
                current,
                "runtime",
                f"conversation runtime failed: {type(exc).__name__}",
                False,
            )
            await self._emit(failed)
            yield failed
        finally:
            # 异常/取消也要保留已经到达的增量正文，但只在终态拼接一次。
            if answer_parts:
                answer = "".join(answer_parts)
            if self.repository is not None:
                now = time()
                try:
                    self.repository.save(
                        ConversationTurn(
                            current.mode,
                            current.profile_id,
                            current.session_id,
                            current.turn_id,
                            "user",
                            text_input,
                            tuple(segments),
                            tuple(
                                {"kind": "system", "text": item}
                                for item in messages[0].content.split("\n\n")
                            )
                            if messages
                            and messages[0].role == "system"
                            and isinstance(messages[0].content, str)
                            else (),
                            now,
                            now,
                            status,
                            "",
                        )
                    )
                except Exception as exc:
                    logger.warning("conversation persistence failed: %s", type(exc).__name__)
            if self.tts is not None:
                if not tts_flushed:
                    self._cancel_tts_turn(current)
                if status != "approval_required":
                    self._release_tts_context(current)
            if status != "approval_required":
                self._sentence_turns.pop(current, None)
            elif sentence_state is not None and not sentence_state.cancelled:
                # 审批续接仍属于同一 ConversationContext；保留分段器/序号，
                # 防止恢复后的句子重新从 1 开始或丢失尾部。
                self._sentence_turns[current] = sentence_state
            if self.tools is not None:
                self.tools.clear_turn(tool_context)
            if status != "approval_required":
                clear_audit = getattr(self.router, "clear_audit_context", None)
                if callable(clear_audit):
                    clear_audit(current)
            if status != "approval_required":
                for approval_id, paused in tuple(self._paused_turns.items()):
                    if paused.context == current:
                        self._paused_turns.pop(approval_id, None)
            if self._active_cancels.get(cancel_key) is active_cancel:
                self._active_cancels.pop(cancel_key, None)
            if self._active_task_by_key.get(cancel_key) is current_task:
                self._active_task_by_key.pop(cancel_key, None)
            if current_task is not None:
                self._active_tasks.discard(current_task)

    async def aclose(self) -> None:
        """取消并等待所有对话与增量 TTS 任务。"""

        self._closed = True
        for state in tuple(self._tts_turns.values()):
            self._cancel_tts_turn(state.context)
        for context in tuple(self._tts_language_snapshots):
            self._release_tts_context(context)
        current_task = asyncio.current_task()
        conversation_tasks = tuple(task for task in self._active_tasks if task is not current_task)
        for task in conversation_tasks:
            task.cancel()
        if conversation_tasks:
            await asyncio.gather(*conversation_tasks, return_exceptions=True)
        tts_tasks = tuple(task for task in self._tts_tasks if task is not current_task)
        for task in tts_tasks:
            task.cancel()
        if tts_tasks:
            await asyncio.gather(*tts_tasks, return_exceptions=True)
        self._active_cancels.clear()
        self._tts_language_snapshots.clear()
        self._tts_role_snapshots.clear()
        self._active_task_by_key.clear()
        self._active_tasks.clear()
        self._sentence_turns.clear()
        self._tts_tasks.clear()
        if self.tools is not None:
            self.tools.clear_pending()

    async def complete(self, user_text: str, **kwargs: Any) -> ConversationResult:
        """消费完整事件流，保留与 ``stream`` 相同的权限和持久化路径。"""

        context = kwargs.get("context")
        current = context if isinstance(context, ConversationContext) else None
        text = ""
        reasoning = ""
        murmur = ""
        outcomes: list[ToolOutcome] = []
        usage: dict[str, int] = {}
        finish_reason = "stop"
        status = "completed"
        async for event in self.stream(user_text, **kwargs):
            if isinstance(event, TextDelta):
                text += event.delta
            elif isinstance(event, ReasoningDelta):
                reasoning += event.delta
            elif isinstance(event, MurmurDelta):
                murmur += event.delta
            elif isinstance(event, TurnFinished):
                usage.update(event.usage)
                finish_reason = event.finish_reason
            elif isinstance(event, TurnFailed):
                status = "failed"
            event_context = getattr(event, "context", None)
            if current is None and isinstance(event_context, ConversationContext):
                current = event_context
        if current is None:
            current = self.begin_context()
        stored = self._results.pop(
            (
                current.mode,
                current.profile_id,
                current.session_id,
                current.turn_id,
                current.generation_id,
            ),
            None,
        )
        if stored is not None:
            return stored
        return ConversationResult(
            current,
            text,
            reasoning,
            murmur,
            tuple(outcomes),
            finish_reason,
            usage,
            status,
        )


__all__ = ["ConversationResult", "ConversationService", "EventSink"]
