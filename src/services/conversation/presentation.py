from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, replace
from threading import RLock

from core.events.types import (
    ApprovalRequested,
    ConversationContext,
    MurmurDelta,
    ReasoningDelta,
    SentenceReady,
    TextDelta,
    ToolStatusChanged,
    TurnFailed,
)

logger = logging.getLogger(__name__)


def _without_xml_marker(raw: str) -> str:
    """剥离增量中出现的 XML 指令标记；未闭合标签只去掉已输出部分。"""

    value = str(raw or "")
    if "<meapet" not in value:
        return value
    return value.split("<meapet", 1)[0]


@dataclass(frozen=True)
class BubbleSnapshot:
    """一次可直接交给渲染层的气泡快照。"""

    text: str = ""
    murmur: str = ""
    tool_status: str = ""
    mood: str = "neutral"
    visible: bool = True
    direct_text: str = ""
    direct_mood: str = "neutral"
    # ``text`` 保留完整正文，供控制台/历史追加；``speech_text`` 只保存最近
    # 一个已完成句，供桌宠气泡与 TTS 同步。序号保证相同句子连续出现时仍会
    # 触发一次新的 UI 更新。
    speech_text: str = ""
    speech_sequence: int = 0

    @property
    def rendered_text(self) -> str:
        """按当前展示策略拼出气泡文本。"""

        if self.direct_text:
            return self.direct_text
        parts = [item for item in (self.text, self.murmur, self.tool_status) if item]
        return "\n".join(parts)

    @property
    def rendered_mood(self) -> str:
        """返回当前优先展示内容对应的情绪。"""

        return self.direct_mood if self.direct_text else self.mood

    @property
    def current_sentence(self) -> str:
        """返回当前应由桌宠气泡展示的已完成句。"""

        return self.speech_text


class PresentationService:
    """集中处理碎碎念、工具状态和文本增量，避免 GUI 依赖模型协议。"""

    # 覆盖一次 20k 字符上限下的最小句长，避免 Qt 轮询短暂阻塞时
    # 静默丢失已完成句；仍保留硬上限，防止宿主永久不消费造成无限增长。
    _SPEECH_QUEUE_LIMIT = 1024

    def __init__(
        self,
        *,
        show_reasoning: bool = False,
        show_murmur: bool = True,
        show_tool_status: bool = True,
        max_text_length: int = 4000,
    ) -> None:
        self.show_reasoning = bool(show_reasoning)
        self.show_murmur = bool(show_murmur)
        self.show_tool_status = bool(show_tool_status)
        self.max_text_length = max(256, min(int(max_text_length), 20000))
        self._lock = RLock()
        self._active_context: ConversationContext | None = None
        self._cancelled_context: ConversationContext | None = None
        self._snapshot = BubbleSnapshot()
        self._pending_sentences: deque[SentenceReady] = deque(maxlen=self._SPEECH_QUEUE_LIMIT)
        self._delivered_speech_sequence = 0
        self._queue_overflow_logged = False

    @property
    def snapshot(self) -> BubbleSnapshot:
        with self._lock:
            return self._snapshot

    def reset(
        self,
        *,
        mood: str = "neutral",
        context: ConversationContext | None = None,
        cancelled: bool = False,
    ) -> BubbleSnapshot:
        """清空展示层，并可绑定到一个新的对话代际。"""

        if context is not None and not isinstance(context, ConversationContext):
            raise TypeError("context must be a ConversationContext")
        with self._lock:
            if context is not None:
                self._active_context = context
                self._cancelled_context = context if cancelled else None
            elif not cancelled:
                # 保留当前游标；ConversationService 在首个新事件到达前仍需
                # 拒绝旧回合的迟到 SSE。
                pass
            elif self._active_context is not None:
                self._cancelled_context = self._active_context
            self._pending_sentences.clear()
            self._delivered_speech_sequence = 0
            self._queue_overflow_logged = False
            self._snapshot = BubbleSnapshot(mood=str(mood or "neutral").strip() or "neutral")
            return self._snapshot

    def _accept_event_context(self, context: object) -> bool:
        """仅接受当前或更高代际事件，避免迟到流覆盖气泡。"""

        if not isinstance(context, ConversationContext):
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
            if context.generation_id <= active.generation_id:
                return False
        elif self._snapshot.rendered_text:
            return False
        self.reset(context=context)
        return True

    def set_direct_speech(self, text: str, *, mood: str = "neutral") -> BubbleSnapshot:
        """设置模型主动说话层；工具状态不会覆盖该层。"""

        value = str(text or "").strip()
        if not value:
            return self.clear_direct_speech()
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                direct_text=value[: self.max_text_length],
                direct_mood=str(mood or "neutral").strip() or "neutral",
            )
            return self._snapshot

    def clear_direct_speech(self) -> BubbleSnapshot:
        """清除模型主动说话层，恢复当前回合的流式内容。"""

        with self._lock:
            self._snapshot = replace(self._snapshot, direct_text="", direct_mood="neutral")
            return self._snapshot

    def commit_sentence(self, event: SentenceReady) -> BubbleSnapshot:
        """提交一个已完成句给桌宠气泡，不改动累计正文。

        原始 ``TextDelta`` 仍由 :meth:`consume` 维护完整输出框；该方法只更新
        最近一句和单调序号，因此同一批次不会把累计全文再次追加到控制台。
        迟到或重复序号会被丢弃，和对话代际过滤保持一致。
        """

        if not isinstance(event, SentenceReady):
            raise TypeError("event must be a SentenceReady")
        with self._lock:
            if not self._accept_event_context(event.context):
                return self._snapshot
            if event.sequence <= self._snapshot.speech_sequence:
                return self._snapshot
            if len(self._pending_sentences) >= self._SPEECH_QUEUE_LIMIT:
                # 宿主长期不可用时按最旧优先丢弃，避免内存无界增长；
                # 正常 20k 字符回合不会触发此保护。
                self._pending_sentences.popleft()
                if not self._queue_overflow_logged:
                    logger.warning(
                        "speech presentation queue reached its limit; dropping oldest sentence"
                    )
                    self._queue_overflow_logged = True
            self._pending_sentences.append(event)
            self._snapshot = replace(
                self._snapshot,
                direct_text="",
                direct_mood="neutral",
                tool_status="",
                speech_text=str(event.text)[: self.max_text_length],
                speech_sequence=event.sequence,
            )
            return self._snapshot

    def pop_next_sentence(self) -> SentenceReady | None:
        """按提交顺序取出一条待显示句，供宿主逐帧投递到桌宠。"""

        with self._lock:
            while self._pending_sentences:
                event = self._pending_sentences[0]
                if not self._accept_event_context(event.context):
                    self._pending_sentences.popleft()
                    continue
                if event.sequence <= self._delivered_speech_sequence:
                    self._pending_sentences.popleft()
                    continue
                if self._ack_sentence_locked(event):
                    return event
                # 只有队列被并发重置/消费时才会到这里；继续检查新的队首，
                # 不把已经确认失败的对象交给宿主造成乱序。
            return None

    def peek_next_sentence(self) -> SentenceReady | None:
        """查看下一条待显示句但不出队，供宿主成功投递后确认。"""

        with self._lock:
            for event in tuple(self._pending_sentences):
                if not self._accept_event_context(event.context):
                    continue
                if event not in self._pending_sentences:
                    continue
                if event.sequence > self._delivered_speech_sequence:
                    return event
            return None

    def ack_sentence(self, event: SentenceReady) -> bool:
        """确认宿主已成功显示一条句子；失败时可保留队列重试。"""

        if not isinstance(event, SentenceReady):
            return False
        with self._lock:
            return self._ack_sentence_locked(event)

    def _ack_sentence_locked(self, event: SentenceReady) -> bool:
        if event.sequence <= self._delivered_speech_sequence:
            return False
        for index, queued in enumerate(self._pending_sentences):
            if queued.context != event.context or queued.sequence != event.sequence:
                continue
            del self._pending_sentences[index]
            self._delivered_speech_sequence = event.sequence
            return True
        return False

    @property
    def pending_sentence_count(self) -> int:
        """返回尚未交给宿主显示的句子数量。"""

        with self._lock:
            return len(self._pending_sentences)

    def _append(self, current: str, delta: str) -> str:
        value = (current + str(delta or ""))[-self.max_text_length :]
        return value

    def consume(self, event: object) -> BubbleSnapshot:
        """消费一个核心事件并返回新的不可变快照。"""

        if isinstance(event, SentenceReady):
            return self.commit_sentence(event)

        with self._lock:
            if not self._accept_event_context(getattr(event, "context", None)):
                return self._snapshot
            snapshot = self._snapshot
            if isinstance(event, TextDelta):
                if event.delta:
                    # XML 指令行不进入气泡；标签可能在多个 delta 之间
                    # 拆分，未闭合片段先剥掉再追加，收尾由会话层剔除残留。
                    visible = _without_xml_marker(str(event.delta))
                    snapshot = replace(
                        snapshot,
                        direct_text="",
                        direct_mood="neutral",
                        tool_status="",
                        text=self._append(snapshot.text, visible),
                    )
            elif isinstance(event, MurmurDelta) and self.show_murmur:
                snapshot = replace(snapshot, murmur=self._append(snapshot.murmur, event.delta))
            elif isinstance(event, ReasoningDelta) and self.show_reasoning:
                snapshot = replace(snapshot, murmur=self._append(snapshot.murmur, event.delta))
            elif isinstance(event, ToolStatusChanged) and self.show_tool_status:
                snapshot = replace(snapshot, tool_status=f"[{event.state}] {event.safe_summary}")
            elif isinstance(event, ApprovalRequested) and self.show_tool_status:
                snapshot = replace(
                    snapshot,
                    tool_status=f"[approval_required] {event.safe_summary}",
                )
            elif isinstance(event, TurnFailed):
                # 失败和配置错误不能被普通工具状态开关吞掉；否则无模型渠道时
                # 气泡与控制台都会保持空白，用户无法知道下一步配置方式。
                summary = event.safe_message or "对话请求失败"
                snapshot = replace(snapshot, tool_status=f"[{event.category}] {summary}")
            self._snapshot = snapshot
            return snapshot


__all__ = ["BubbleSnapshot", "PresentationService"]
