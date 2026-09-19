"""文本增量到 TTS 分段和有界音频事件的非阻塞协调器。"""

from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event as ThreadEvent
from threading import Lock as ThreadLock
from threading import RLock, Thread
from time import monotonic

from core.events.types import AudioChunk, AudioFeature, ConversationContext
from core.tts.contracts import AudioFeatureAnalyzer, EngineHealth, SpeechBackend, SpeechRequest
from logger.events import log_event

logger = logging.getLogger(__name__)


def _tts_role_key(value: object) -> str:
    """返回回合角色快照比较使用的稳定键。"""

    role = str(value or "").strip()
    return role.casefold() if role else ""


class _CrossLoopLock:
    """可在多个 asyncio 事件循环之间安全等待的轻量串行锁。"""

    def __init__(self) -> None:
        self._lock = ThreadLock()

    async def __aenter__(self) -> _CrossLoopLock:
        while not self._lock.acquire(blocking=False):
            # 非阻塞轮询让出当前事件循环；协程取消时不会遗留
            # 一个在线程池中永远等待的锁获取任务。
            await asyncio.sleep(0.005)
        return self

    async def __aexit__(self, *_args: object) -> None:
        self._lock.release()

    def locked(self) -> bool:
        return self._lock.locked()


class SpeechSegmenter:
    """把增量文本切成适合低延迟合成的、有严格上限的片段。

    句末标点或换行会立即结束一个片段；当模型长时间不输出句末标点时，
    软边界（逗号、分号、空白等）只在达到上限后用于选择较自然的断点。
    这样首个完整句子可以马上进入 TTS，同时不会因为异常长的单句把一次
    请求膨胀到后端或音频队列的上限之外。
    """

    _DEFAULT_HARD_BOUNDARIES = "。！？!?…｡．.\n\r"
    _DEFAULT_SOFT_BOUNDARIES = "，,、；;：: \t"
    _CLOSING_MARKS = frozenset("\"'”’」』）》】〕〉》）)]}>")
    MAX_CHARS = 2_000

    def __init__(
        self,
        *,
        max_chars: int = 120,
        hard_boundaries: str | None = None,
        soft_boundaries: str | None = None,
        soft_cut_trigger: int | None = None,
    ) -> None:
        try:
            limit = int(max_chars)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("speech segment max_chars must be an integer") from exc
        self.max_chars = max(20, min(limit, self.MAX_CHARS))
        self.hard_boundaries = frozenset(
            str(hard_boundaries) if hard_boundaries is not None else self._DEFAULT_HARD_BOUNDARIES
        )
        self.soft_boundaries = frozenset(
            str(soft_boundaries) if soft_boundaries is not None else self._DEFAULT_SOFT_BOUNDARIES
        )
        if not self.hard_boundaries:
            raise ValueError("speech segment hard_boundaries cannot be empty")
        if soft_cut_trigger is None:
            soft_cut_trigger = int(self.max_chars * 0.4)
        try:
            trigger = int(soft_cut_trigger)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("speech segment soft_cut_trigger must be an integer") from exc
        self.soft_cut_trigger = max(12, min(trigger, self.max_chars))
        self._buffer = ""

    @property
    def buffered_chars(self) -> int:
        """返回尚未形成完整片段的字符数。"""

        return len(self._buffer)

    def _hard_boundary(self) -> int:
        for index, char in enumerate(self._buffer):
            if char not in self.hard_boundaries:
                continue
            # ASCII/全角句点出现在两个字母或数字之间时通常是小数、版本号、
            # 邮箱或域名的一部分，不应把一个 TTS 片段截成 ``3.``/``14``。
            # 句点后接空格、换行或句末时仍按正常句号处理。
            if char in {".", "．"}:
                previous = self._buffer[index - 1] if index > 0 else ""
                following = self._buffer[index + 1] if index + 1 < len(self._buffer) else ""
                if (
                    previous.isascii()
                    and previous.isalnum()
                    and following.isascii()
                    and following.isalnum()
                ):
                    continue
            return index
        return -1

    def _soft_cut(self, limit: int) -> int:
        """在 ``limit`` 内选最后一个软边界，且不返回纯空白片段。"""

        for index in range(min(limit, len(self._buffer)) - 1, -1, -1):
            if self._buffer[index] not in self.soft_boundaries:
                continue
            if self._buffer[: index + 1].strip():
                return index + 1
        return 0

    def _hard_cut(self, boundary: int) -> int:
        """返回不超过 ``max_chars`` 的句末切点，并吸收相邻闭合标记。"""

        cut = min(boundary + 1, self.max_chars)
        # ``？！``、句末引号等属于同一个可听片段；同时严格保留硬上限，
        # 超过上限的闭合字符留到下一片，避免任何路径返回超长文本。
        while cut < min(len(self._buffer), self.max_chars):
            if (
                self._buffer[cut] in self.hard_boundaries
                or self._buffer[cut] in self._CLOSING_MARKS
            ):
                cut += 1
                continue
            break
        return cut

    def _drain(self, *, force: bool = False) -> tuple[str, ...]:
        segments: list[str] = []
        while self._buffer:
            boundary = self._hard_boundary()
            if boundary >= 0 and boundary + 1 <= self.max_chars:
                cut = self._hard_cut(boundary)
                # 流式增量可能恰好截在 ``3.`` 或省略号中间；尾部句点
                # 没有右侧字符时无法判断它是句末还是版本号/小数点。
                # 延迟到下一增量或最终 flush，再决定是否提交给 TTS。
                if not force and self._buffer[boundary] in {".", "．"} and cut == len(self._buffer):
                    # 段落换行的尾部句点不参与小数延迟；换行本身已经是
                    # 明确的切分信号，句子在跨行时立即提交。
                    if "\r" not in self._buffer[: cut + 1] and "\n" not in self._buffer[: cut + 1]:
                        break
            elif len(self._buffer) < self.max_chars and not force:
                # 无标点长句但已越过软切点：即使未达到硬上限，也在最近
                # 的软边界提前断句，避免整段长句一次性输出。没有软边界
                # 可用时按呼吸点硬切一个触发长度，保证输出仍是流式的。
                if self._buffer.strip() and len(self._buffer) >= self.soft_cut_trigger:
                    soft = self._soft_cut(len(self._buffer))
                    if soft > 0:
                        cut = self._consume_segment(soft, segments)
                        if cut > 0:
                            continue
                    cut = self._consume_segment(self.soft_cut_trigger, segments)
                    if cut > 0:
                        continue
                break
            else:
                limit = min(len(self._buffer), self.max_chars)
                cut = self._soft_cut(limit) or limit
            # 防御性保障：即使调用方提供了异常边界字符集合，也必须前进。
            cut = max(1, min(cut, self.max_chars, len(self._buffer)))
            raw_segment = self._buffer[:cut]
            self._buffer = self._buffer[cut:]
            # 保留软边界中的普通空格，避免英文片段拼接时黏词；段落换行
            # 只作为切分信号，不送入 TTS。纯空白片段直接丢弃，防止前导
            # 空格堆积后把下一片推过 max_chars 上限。
            segment = raw_segment.replace("\r", "").replace("\n", "")
            if segment.strip():
                segments.append(segment)
        return tuple(segments)

    def _consume_segment(self, cut: int, segments: list[str]) -> int:
        """把缓冲前 ``cut`` 字符制成一个非空片段。

        Args:
            cut: 待消费的字符数。
            segments: 输出列表，非空时追加。

        Returns:
            实际消费的字符数；片段为纯空白时返回 0 并丢弃缓冲。
        """

        bounded = max(1, min(int(cut), self.max_chars, len(self._buffer)))
        if bounded <= 0:
            return 0
        raw_segment = self._buffer[:bounded]
        self._buffer = self._buffer[bounded:]
        segment = raw_segment.replace("\r", "").replace("\n", "")
        if segment.strip():
            segments.append(segment)
        return bounded

    def push(self, text: str) -> tuple[str, ...]:
        self._buffer += str(text or "")
        return self._drain()

    def flush(self) -> tuple[str, ...]:
        """冲刷尾部文本；即使调用方没有句末标点也遵守长度上限。"""

        return self._drain(force=True)


@dataclass(frozen=True)
class SpeechStatus:
    context: ConversationContext
    state: str
    message: str


class TTSCoordinator:
    # 状态回调属于 UI 诊断旁路；即使宿主线程停止响应，也不能让音频
    # 合成或取消路径无限等待。快速回调仍保持完整的状态顺序。
    _STATUS_SINK_TIMEOUT_SECONDS = 0.25

    def __init__(
        self,
        backend: object,
        *,
        queue_size: int = 16,
        segment_max_chars: int = 120,
        segment_hard_boundaries: str | None = None,
        segment_soft_boundaries: str | None = None,
        segment_soft_cut_trigger: int | None = None,
        status_sink: Callable[[SpeechStatus], Awaitable[None] | None] | None = None,
        audio_sink: Callable[[AudioChunk], Awaitable[None] | None] | None = None,
        feature_analyzer: AudioFeatureAnalyzer | None = None,
        feature_sink: Callable[[AudioFeature], Awaitable[None] | None] | None = None,
    ) -> None:
        self.backend = backend
        # 观察队列可被不同事件循环的宿主消费；Qt 实际播放仍走 audio_sink 信号边界。
        self._queue: Queue[AudioChunk] = Queue(maxsize=max(1, min(int(queue_size), 128)))
        # ``next_audio`` 可能长期等待。使用线程事件跨事件循环唤醒消费者，
        # 避免空队列时固定间隔轮询造成无效 CPU 唤醒。
        self._audio_available: ThreadEvent = ThreadEvent()
        self.segment_max_chars = max(20, min(int(segment_max_chars), SpeechSegmenter.MAX_CHARS))
        self.segment_hard_boundaries = segment_hard_boundaries
        self.segment_soft_boundaries = segment_soft_boundaries
        if segment_soft_cut_trigger is None:
            segment_soft_cut_trigger = int(self.segment_max_chars * 0.4)
        self.segment_soft_cut_trigger = max(
            12, min(int(segment_soft_cut_trigger), self.segment_max_chars)
        )
        self._status_sink = status_sink
        self._audio_sink = audio_sink
        self._feature_analyzer = feature_analyzer
        self._feature_sink = feature_sink
        self._segments: dict[tuple[str, str, str, str, int], SpeechSegmenter] = {}
        self._serializers: dict[tuple[str, str, str, str, int], _CrossLoopLock] = {}
        # 每个会话代际固定语言与首选 profile；控制台热切换只影响新回合，
        # 避免同一回合的后续句段突然换音色或协议语言。
        self._context_routes: dict[tuple[str, str, str, str, int], tuple[str, str]] = {}
        self._context_route_roles: dict[tuple[str, str, str, str, int], str] = {}
        # ConversationService 在回合开始时显式保留路由。审批暂停会先冲刷
        # 已有句段，但续接仍属于同一代际，不能因为中途切换 profile 就改用
        # 新音色；终态由 release_context 释放，旧的直接 coordinator 调用
        # 不需要显式 pin，仍沿用 flush 后清理的兼容语义。
        self._pinned_contexts: set[tuple[str, str, str, str, int]] = set()
        self._release_pending: set[tuple[str, str, str, str, int]] = set()
        self._cancelled: set[tuple[str, str, str, str, int]] = set()
        self._cancelled_order: deque[tuple[str, str, str, str, int]] = deque()
        self._active_tasks: dict[tuple[str, str, str, str, int], set[asyncio.Task[object]]] = {}
        self._task_loops: dict[asyncio.Task[object], asyncio.AbstractEventLoop] = {}
        self._audio_lock = RLock()
        self._start_lock = _CrossLoopLock()
        self._started = False
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def queue_size(self) -> int:
        """返回观察音频队列的当前上限。"""

        return int(self._queue.maxsize)

    def set_queue_size(self, queue_size: int) -> int:
        """在不丢弃已生成音频的前提下更新队列上限。"""

        if isinstance(queue_size, bool):
            raise ValueError("speech queue_size must be an integer")
        try:
            normalized = int(queue_size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("speech queue_size must be an integer") from exc
        if normalized < 1 or normalized > 128:
            raise ValueError("speech queue_size is outside the allowed range")
        with self._queue.mutex:
            self._queue.maxsize = normalized
            self._queue.not_full.notify_all()
        return normalized

    def set_audio_sink(self, sink: Callable[[AudioChunk], Awaitable[None] | None] | None) -> None:
        """在 Qt/其他宿主初始化后绑定音频消费端。"""

        self._audio_sink = sink

    def set_feature_analyzer(self, analyzer: AudioFeatureAnalyzer | None) -> None:
        """绑定可选的真实音频特征分析器。

        未绑定分析器时不会生成任何 viseme；这是有意保留文本/overlay 回退的
        默认行为。分析器只接收已经通过 PCM 协议校验的音频事件。
        """

        self._feature_analyzer = analyzer

    def set_feature_sink(
        self, sink: Callable[[AudioFeature], Awaitable[None] | None] | None
    ) -> None:
        """绑定已确认特征的渲染消费端。"""

        self._feature_sink = sink

    def has_active_tasks(self) -> bool:
        """返回是否仍有语音分片在生成或播放桥接。"""

        with self._audio_lock:
            return any(not task.done() for tasks in self._active_tasks.values() for task in tasks)

    @property
    def started(self) -> bool:
        with self._audio_lock:
            return bool(self._started and not self._closing and not self._closed)

    @property
    def closed(self) -> bool:
        with self._audio_lock:
            return bool(self._closed)

    def clear_context_routes(self) -> None:
        """清除已完成回合的语言/profile 快照；活动或固定回合不应调用。"""

        with self._audio_lock:
            if any(not task.done() for tasks in self._active_tasks.values() for task in tasks):
                return
            # ``enqueue_text`` 在尚未遇到句末标点时会保留分段器尾部，
            # 而其短暂空闲窗口不会出现在 ``_active_tasks``。这类回合
            # 仍需沿用第一次入队时的语言/profile；否则热切换或宿主
            # 直接调用本方法会让尾部文本改用新音色。已由对话服务
            # ``pin_context`` 固定的回合即使没有尾部文本，也必须保留
            # 快照，直到显式 ``release_context``，否则审批暂停后的
            # 续接会错误地切换到新 profile。
            for key in tuple(self._context_routes):
                if key in self._pinned_contexts:
                    continue
                segmenter = self._segments.get(key)
                if segmenter is not None and segmenter.buffered_chars:
                    continue
                self._context_routes.pop(key, None)
                self._context_route_roles.pop(key, None)

    def pin_context(
        self,
        context: ConversationContext,
        *,
        language: str = "zh",
        role: str = "",
    ) -> tuple[str, str]:
        """固定一个回合的语言/profile 路由并返回快照。

        对话回合在审批点可能暂时冲刷 TTS 分段；只要没有调用
        :meth:`release_context`，同一 ``ConversationContext`` 的续接就会
        继续使用首次选择的语言和 profile。该方法是幂等的，也可跨事件
        循环调用；已取消的上下文不会重新建立路由。
        """

        key = self._context_key(context)
        with self._audio_lock:
            if self._closing or self._closed or key in self._cancelled:
                return ("", "")
            self._pinned_contexts.add(key)
            self._release_pending.discard(key)
        return self._route_for_context(key, language, role=role)

    def release_context(self, context: ConversationContext) -> None:
        """释放回合路由快照；若仍有合成任务则延迟到任务收尾。"""

        key = self._context_key(context)
        with self._audio_lock:
            self._pinned_contexts.discard(key)
            if self._active_tasks.get(key):
                self._release_pending.add(key)
                return
            self._release_pending.discard(key)
            self._segments.pop(key, None)
            self._context_routes.pop(key, None)
            self._context_route_roles.pop(key, None)
            serializer = self._serializers.get(key)
            if serializer is not None and not serializer.locked():
                self._serializers.pop(key, None)

    def set_segmentation(
        self,
        *,
        max_chars: int,
        hard_boundaries: str | None = None,
        soft_boundaries: str | None = None,
        soft_cut_trigger: int | None = None,
    ) -> None:
        """更新后续语音分段策略，并安全回收已完成的旧分段器。"""

        self.segment_max_chars = max(20, min(int(max_chars), SpeechSegmenter.MAX_CHARS))
        self.segment_hard_boundaries = hard_boundaries
        self.segment_soft_boundaries = soft_boundaries
        if soft_cut_trigger is None:
            soft_cut_trigger = int(self.segment_max_chars * 0.4)
        self.segment_soft_cut_trigger = max(12, min(int(soft_cut_trigger), self.segment_max_chars))
        # 运行时通常只在空闲时替换配置；若旧回合仍在收尾，保留其活动
        # 分段器及未形成句子的尾部，避免热重载截断已经到达的文本。新建
        # 回合会使用上面的新配置。整个检查和回收必须持有同一把锁，
        # 否则跨事件循环的 enqueue_text 可能在遍历期间修改字典。
        with self._audio_lock:
            active_keys = set(self._active_tasks)
            for key, segmenter in tuple(self._segments.items()):
                if key in active_keys or segmenter.buffered_chars:
                    continue
                self._segments.pop(key, None)

    def set_status_sink(
        self, sink: Callable[[SpeechStatus], Awaitable[None] | None] | None
    ) -> None:
        """绑定语音状态消费端。"""

        self._status_sink = sink

    async def _status(self, status: SpeechStatus) -> None:
        if self._status_sink is None:
            return
        try:
            await self._invoke_sink(self._status_sink, status)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("speech status sink failed: %s", type(exc).__name__)

    async def _wait_status_task(self, task: asyncio.Task[object]) -> None:
        """有界等待 started 状态任务，并在 UI 背压时安全回收。"""

        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._STATUS_SINK_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            logger.debug("speech status sink timed out")
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _emit_status_bounded(
        self,
        status: SpeechStatus,
        *,
        swallow_cancel: bool = False,
    ) -> None:
        """有界投递终态；UI 背压不能拖住音频/会话收尾。"""

        task = asyncio.create_task(self._status(status))
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._STATUS_SINK_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            logger.debug("speech status sink timed out")
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if not swallow_cancel:
                raise
            # 调用方的 CancelledError 会在外层重新抛出；这里不让状态旁路
            # 覆盖原始取消信号。
            return

    @staticmethod
    async def _invoke_sync(call: Callable[[object], object], value: object) -> object:
        """在线程池之外以 daemon 线程调用同步回调，避免关闭时无限等待。"""

        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[object] = loop.create_future()
        cancel_requested = ThreadEvent()

        def set_result(result: object) -> None:
            if result_future.done():
                # 回调可能在协程取消后才返回；不要让它创建的 coroutine
                # 对象在事件循环关闭时产生 ``was never awaited`` 警告。
                if inspect.iscoroutine(result):
                    result.close()
                return
            result_future.set_result(result)

        def set_error(error: BaseException) -> None:
            if not result_future.done():
                result_future.set_exception(error)

        def run() -> None:
            if cancel_requested.is_set():
                return
            try:
                result = call(value)
            except BaseException as exc:
                try:
                    loop.call_soon_threadsafe(set_error, exc)
                except RuntimeError:
                    # 事件循环已关闭；daemon 线程不能再向其投递回调。
                    return
            else:
                try:
                    loop.call_soon_threadsafe(set_result, result)
                except RuntimeError:
                    if inspect.iscoroutine(result):
                        result.close()

        Thread(target=run, name="meapet-tts-sink", daemon=True).start()
        try:
            return await asyncio.shield(result_future)
        except asyncio.CancelledError:
            cancel_requested.set()
            # 已进入同步 sink 的线程无法被强制终止。给它一个很短的收尾
            # 窗口可稳定释放音频对象；永久阻塞的第三方 sink 仍由 daemon
            # 线程隔离，不会拖住应用退出或用户取消。
            try:
                late_result = await asyncio.wait_for(
                    asyncio.shield(result_future),
                    timeout=0.1,
                )
            except BaseException:
                # 取消已经成为调用方的终态；同步 sink 在收尾窗口内产生的
                # 异常不能反向覆盖该终态，后台线程也不能再持有 Future。
                result_future.cancel()
            else:
                if inspect.iscoroutine(late_result):
                    late_result.close()
            raise

    @staticmethod
    async def _invoke_sink(
        sink: Callable[[object], Awaitable[object] | object], value: object
    ) -> None:
        """在不占用模型事件循环的情况下调用同步或异步消费端。

        Qt 音频桥通常是同步的信号发射；直接在 TTS worker 中调用时，某个
        慢消费端会反向卡住同一事件循环里的模型流。同步调用统一放到
        daemon 线程并等待其完成，因此同一回合的分片顺序仍保持，而模型流
        可以继续处理后续 delta。异步消费端仍在原事件循环中等待自己的让出点。
        """

        call = getattr(sink, "__call__", sink)
        if inspect.iscoroutinefunction(sink) or inspect.iscoroutinefunction(call):
            result = sink(value)
        else:
            result = await TTSCoordinator._invoke_sync(sink, value)
        if inspect.isawaitable(result):
            await result

    async def start(self) -> EngineHealth:
        """随应用启动初始化 TTS 后端，并返回真实健康状态。"""

        async with self._start_lock:
            return await self._start_once()

    async def _start_once(self) -> EngineHealth:
        """在启动 single-flight 内再次检查状态并初始化后端。"""

        with self._audio_lock:
            if self._closing or self._closed:
                return EngineHealth("tts", False, "TTS coordinator is closed")
            already_started = self._started
        if already_started:
            return await self.health()
        log_event(
            logger,
            "tts.coordinator.initialize",
            component="tts.coordinator",
            status="started",
        )
        initializer = getattr(self.backend, "start", None)
        try:
            if callable(initializer):
                result = initializer()
                health = await result if inspect.isawaitable(result) else result
                if not isinstance(health, EngineHealth):
                    health = await self.health()
            else:
                health = await self.health()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log_event(
                logger,
                "tts.coordinator.initialize",
                component="tts.coordinator",
                status="failed",
                level=logging.WARNING,
                fields={"error": type(exc).__name__},
            )
            return EngineHealth(
                "tts",
                False,
                f"TTS initialization failed: {type(exc).__name__}",
            )
        with self._audio_lock:
            if not self._closing and not self._closed:
                self._started = bool(health.available)
        log_event(
            logger,
            "tts.coordinator.initialize",
            component="tts.coordinator",
            status="ready" if health.available else "unavailable",
            level=logging.INFO if health.available else logging.WARNING,
            duration_ms=health.latency_ms,
            fields={"engine": health.engine, "available": health.available},
        )
        return health

    async def replace_backend(
        self, backend: SpeechBackend, *, initialized: bool = False
    ) -> SpeechBackend:
        """在无活动语音时原子接管一个已构造后端并返回旧后端。"""

        if backend is None:
            raise TypeError("backend is required")
        async with self._start_lock:
            with self._audio_lock:
                if self._closing or self._closed:
                    raise RuntimeError("TTS coordinator is closed")
                if self._active_tasks:
                    raise RuntimeError("cannot replace TTS backend while speech is active")
                previous = self.backend
                self.backend = backend
                self._started = bool(initialized)
            return previous

    async def health(self) -> EngineHealth:
        result = self.backend.health()
        return await result if inspect.isawaitable(result) else result

    def _profile_for(
        self,
        language: str,
        preferred_profile: str = "",
        *,
        role: str = "",
    ) -> str:
        """读取后端的语言和角色路由首选项；旧后端返回空标识。"""

        preferred = str(preferred_profile or "").strip()
        if preferred:
            enabled_probe = getattr(self.backend, "profile_enabled", None)
            if callable(enabled_probe):
                try:
                    if not bool(enabled_probe(preferred)):
                        raise ValueError("requested TTS voice profile is unavailable")
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    raise ValueError("requested TTS voice profile is unavailable") from None
                return preferred
            profile_ids = getattr(self.backend, "profile_ids", None)
            if not callable(profile_ids) or preferred not in tuple(profile_ids()):
                raise ValueError("requested TTS voice profile is unavailable")
            return preferred

        resolver = getattr(self.backend, "profile_for", None)
        if not callable(resolver):
            return ""
        try:
            value = resolver(language, role=role) if role else resolver(language)
        except TypeError:
            # 兼容尚未支持角色参数的旧 profile 路由器；只有在其签名
            # 不接受角色时才回落到语言路由，不改变旧后端行为。
            if not role:
                return ""
            try:
                value = resolver(language)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return ""
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return ""
        return str(value or "").strip()[:128]

    def _route_for_context(
        self,
        key: tuple[str, str, str, str, int],
        language: str,
        preferred_profile: str = "",
        role: str = "",
    ) -> tuple[str, str]:
        with self._audio_lock:
            existing = self._context_routes.get(key)
            existing_role = self._context_route_roles.get(key, "")
        normalized_language = str(language or "zh").strip() or "zh"
        # 已固定的对话回合必须始终使用首次路由，即使配置 watcher
        # 或全局角色映射在合成期间发生变化；未固定的独立工具调用
        # 仍可以用 voice/role 覆盖默认路由。
        if existing is not None and (
            not preferred_profile
            and (not role or _tts_role_key(role) == _tts_role_key(existing_role))
        ):
            # 热重载可能移除当前回合最初使用的 profile。继续返回失效
            # ``profile_id`` 会让路由器拒绝后续句段，导致文本正常完成而
            # TTS 静默丢失。仅在后端明确提供 profile 探针且 profile 已
            # 不可用时迁移快照；仍存在的 profile 继续保持回合内音色固定。
            profile_id = existing[1]
            enabled_probe = getattr(self.backend, "profile_enabled", None)
            if not profile_id or not callable(enabled_probe):
                return existing
            try:
                profile_available = bool(enabled_probe(profile_id))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                profile_available = False
            if profile_available:
                return existing
            with self._audio_lock:
                if self._context_routes.get(key) == existing:
                    self._context_routes.pop(key, None)
                    self._context_route_roles.pop(key, None)
        profile = self._profile_for(normalized_language, preferred_profile, role=role)
        if existing is not None and (preferred_profile or role):
            return normalized_language, profile
        with self._audio_lock:
            # 两个并发入口可能同时首次到达；只保留先提交的快照。
            existing = self._context_routes.setdefault(key, (normalized_language, profile))
            self._context_route_roles.setdefault(key, str(role or "").strip())
            return existing

    async def enqueue_text(
        self,
        context: ConversationContext,
        text: str,
        *,
        language: str = "zh",
        mood: str = "neutral",
        profile_id: str = "",
        voice: str = "",
        role: str = "",
        flush: bool = False,
    ) -> tuple[str, ...]:
        if self._closing or self._closed:
            return ()
        key = self._context_key(context)
        with self._audio_lock:
            if key in self._cancelled:
                return ()
        effective_language, effective_profile = self._route_for_context(
            key,
            language,
            profile_id,
            role=role,
        )
        task_key = key
        task = asyncio.current_task()
        if task is not None:
            with self._audio_lock:
                self._active_tasks.setdefault(task_key, set()).add(task)
                try:
                    self._task_loops[task] = asyncio.get_running_loop()
                except RuntimeError:
                    pass
        with self._audio_lock:
            serializer = self._serializers.setdefault(key, _CrossLoopLock())
        try:
            async with serializer:
                # 取消可能发生在等待串行锁期间；重新检查墓碑，阻止
                # 已排队的旧请求在锁释放后重新启动后端。
                with self._audio_lock:
                    if self._closing or self._closed or key in self._cancelled:
                        return ()
                segmenter = self._segments.setdefault(
                    key,
                    SpeechSegmenter(
                        max_chars=self.segment_max_chars,
                        hard_boundaries=self.segment_hard_boundaries,
                        soft_boundaries=self.segment_soft_boundaries,
                        soft_cut_trigger=getattr(self, "segment_soft_cut_trigger", None),
                    ),
                )
                segments = list(segmenter.push(text))
                if flush:
                    segments.extend(segmenter.flush())
                for segment in segments:
                    with self._audio_lock:
                        if self._closing or self._closed or key in self._cancelled:
                            return ()
                    await self._synthesize(
                        context,
                        segment,
                        language=effective_language,
                        mood=mood,
                        profile_id=effective_profile,
                        voice=voice,
                        role=role,
                    )
                if flush:
                    with self._audio_lock:
                        self._segments.pop(key, None)
                        # 被 ConversationService pin 住的回合可能在审批后
                        # 继续生成模型文本；仅清理未显式保留的兼容调用。
                        if key not in self._pinned_contexts:
                            self._context_routes.pop(key, None)
                            self._context_route_roles.pop(key, None)
                return tuple(segments)
        finally:
            with self._audio_lock:
                if task is not None:
                    active = self._active_tasks.get(task_key)
                    if active is not None:
                        active.discard(task)
                        if not active:
                            self._active_tasks.pop(task_key, None)
                    self._task_loops.pop(task, None)
                if (
                    task_key in self._release_pending
                    and not self._active_tasks.get(task_key)
                    and not serializer.locked()
                ):
                    self._release_pending.discard(task_key)
                    self._segments.pop(task_key, None)
                    self._context_routes.pop(task_key, None)
                    self._context_route_roles.pop(task_key, None)
                    self._pinned_contexts.discard(task_key)
                if not serializer.locked() and not self._active_tasks.get(task_key):
                    self._serializers.pop(task_key, None)

    async def _synthesize(
        self,
        context: ConversationContext,
        text: str,
        *,
        language: str,
        mood: str,
        profile_id: str = "",
        voice: str = "",
        role: str = "",
    ) -> None:
        with self._audio_lock:
            if self._closing or self._closed or self._context_key(context) in self._cancelled:
                return
        request = SpeechRequest(
            request_id=f"speech-{secrets.token_urlsafe(8)}",
            text=text,
            language=language,
            mood=mood,
            profile_id=profile_id,
            voice=voice,
            role=role,
        )
        log_event(
            logger,
            "tts.synthesis.start",
            component="tts.coordinator",
            status="started",
            correlation_id=request.request_id,
            fields={
                "language": request.language,
                "mood": request.mood,
                "voice": request.voice or request.profile_id or "default",
            },
        )
        # 状态 sink 通常跨线程投递到 Qt；它可能等待主线程当前事件循环。
        # 不能在这里直接 await，否则“started”状态的 UI 回调会挡住真正的
        # backend.stream，首个音频请求就不再是模型流期间即时启动。状态任务
        # 与后端迭代并行，正常收尾时再等待它以保持 started → completed 顺序。
        started_status_task = asyncio.create_task(
            self._status(SpeechStatus(context, "started", "语音分段已排队"))
        )
        synthesis_started = monotonic()
        first_audio_at: float | None = None
        audio_chunk_count = 0
        audio_bytes = 0
        cancelled = False
        stream: object | None = None
        try:
            stream = self.backend.stream(request)
            try:
                async for chunk in stream:
                    with self._audio_lock:
                        if (
                            self._closing
                            or self._closed
                            or self._context_key(context) in self._cancelled
                        ):
                            cancelled = True
                            break
                    if chunk.request_id != request.request_id:
                        raise RuntimeError("TTS backend returned mismatched request_id")
                    event = AudioChunk(
                        context=context,
                        data=chunk.data,
                        sample_rate=chunk.sample_rate,
                        channels=chunk.channels,
                        is_final=chunk.is_final,
                        request_id=request.request_id,
                    )
                    audio_chunk_count += 1
                    audio_bytes += len(event.data)
                    if event.data and first_audio_at is None:
                        first_audio_at = monotonic()
                    with self._audio_lock:
                        if (
                            self._closing
                            or self._closed
                            or self._context_key(context) in self._cancelled
                        ):
                            cancelled = True
                            break
                        self._put_audio(event)
                        sink = self._audio_sink
                        # cancel 可能在写入观察队列后、调用播放桥之前到达。
                        # 再次检查墓碑可避免任意同步 sink 把已取消片段送入播放；
                        # QtAudioPlayer 也会在自己的线程边界重复校验上下文。
                        if (
                            self._closing
                            or self._closed
                            or self._context_key(context) in self._cancelled
                        ):
                            cancelled = True
                            sink = None
                    # 外部 sink（尤其是 Qt 信号桥或测试 fake）不得在状态锁内
                    # 执行；同步 sink 阻塞时，cancel()/next_audio() 仍须可运行。
                    if sink is not None:
                        try:
                            await self._invoke_sink(sink, event)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            # 播放端故障不能中断后续文本分片和 TTS 生成。
                            logger.debug("speech audio sink await failed", exc_info=True)
                    await self._dispatch_features(event)
            finally:
                try:
                    close_stream = getattr(stream, "aclose", None)
                    if callable(close_stream):
                        try:
                            close_result = close_stream()
                            if inspect.isawaitable(close_result):
                                await close_result
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            logger.debug("speech stream close failed: %s", type(exc).__name__)
                finally:
                    # 自定义后端可能在异常/取消时没有发送 is_final；
                    # 让有状态分析器释放本请求的帧偏移，避免请求 ID 重用
                    # 后沿用旧时间轴。
                    await self._reset_feature_request(context, request.request_id)
            # 状态 sink 的异常已在 ``_status`` 内隔离；等待任务只为确保
            # 快速 sink 的 started 事件先于终态，慢 sink 不会影响首个音频。
            await self._wait_status_task(started_status_task)
            await self._emit_status_bounded(
                SpeechStatus(
                    context,
                    "cancelled" if cancelled else "completed",
                    "语音分段已取消" if cancelled else "语音分段完成",
                )
            )
            log_event(
                logger,
                "tts.synthesis.cancel" if cancelled else "tts.synthesis.complete",
                component="tts.coordinator",
                status="cancelled" if cancelled else "completed",
                correlation_id=request.request_id,
                duration_ms=(monotonic() - synthesis_started) * 1000.0,
                fields={
                    "audio_chunk_count": audio_chunk_count,
                    "audio_bytes": audio_bytes,
                    "time_to_first_audio_ms": (
                        (first_audio_at - synthesis_started) * 1000.0
                        if first_audio_at is not None
                        else None
                    ),
                },
            )
        except asyncio.CancelledError:
            if not started_status_task.done():
                started_status_task.cancel()
            await asyncio.gather(started_status_task, return_exceptions=True)
            await self._emit_status_bounded(
                SpeechStatus(context, "cancelled", "语音分段已取消"),
                swallow_cancel=True,
            )
            log_event(
                logger,
                "tts.synthesis.cancel",
                component="tts.coordinator",
                status="cancelled",
                correlation_id=request.request_id,
                duration_ms=(monotonic() - synthesis_started) * 1000.0,
                fields={
                    "audio_chunk_count": audio_chunk_count,
                    "audio_bytes": audio_bytes,
                    "time_to_first_audio_ms": (
                        (first_audio_at - synthesis_started) * 1000.0
                        if first_audio_at is not None
                        else None
                    ),
                },
            )
            raise
        except Exception as exc:
            # 后端在首个分片前失败时仍要消费 started 任务，避免未处理异常
            # 留在事件循环；其 sink 自身即使失败也只影响状态展示。
            try:
                await self._wait_status_task(started_status_task)
            except asyncio.CancelledError:
                raise
            log_event(
                logger,
                "tts.synthesis.failed",
                component="tts.coordinator",
                status="failed",
                level=logging.WARNING,
                correlation_id=request.request_id,
                duration_ms=(monotonic() - synthesis_started) * 1000.0,
                fields={
                    "error": type(exc).__name__,
                    "audio_chunk_count": audio_chunk_count,
                    "audio_bytes": audio_bytes,
                    "time_to_first_audio_ms": (
                        (first_audio_at - synthesis_started) * 1000.0
                        if first_audio_at is not None
                        else None
                    ),
                },
            )
            await self._emit_status_bounded(
                SpeechStatus(context, "degraded", "语音不可用，保留文本输出")
            )

    async def _reset_feature_request(
        self,
        context: ConversationContext,
        request_id: str,
    ) -> None:
        """在音频流异常终止时释放分析器的请求状态。"""

        analyzer = self._feature_analyzer
        reset = getattr(analyzer, "reset", None) if analyzer is not None else None
        if not callable(reset):
            return
        try:
            result = reset(context, request_id)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("speech feature analyzer reset failed: %s", type(exc).__name__)

    async def _dispatch_features(self, chunk: AudioChunk) -> None:
        """将分析器明确确认的特征送往渲染边界。

        该步骤是可选的、隔离故障的旁路：分析器缺失、返回非法事件或消费端
        失败都不能改变 PCM 播放和文本输出。上下文与 request_id 不匹配的
        结果会被丢弃，避免迟到音频驱动当前回合的模型。
        """

        analyzer = self._feature_analyzer
        sink = self._feature_sink
        if analyzer is None or sink is None:
            return
        try:
            analyze = getattr(analyzer, "analyze", None)
            if not callable(analyze):
                return
            call = getattr(analyze, "__call__", analyze)
            if inspect.iscoroutinefunction(analyze) or inspect.iscoroutinefunction(call):
                result = analyze(chunk)
            else:
                result = await self._invoke_sync(analyze, chunk)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                return
            features = tuple(result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "speech feature analyzer failed; keeping overlay fallback: %s",
                type(exc).__name__,
            )
            return
        for feature in features:
            if not isinstance(feature, AudioFeature):
                logger.warning("speech feature analyzer returned an unsupported event")
                continue
            if (
                feature.context != chunk.context
                or feature.request_id != chunk.request_id
                or feature.confirmed is not True
            ):
                logger.warning("speech feature analyzer returned a stale or unconfirmed event")
                continue
            with self._audio_lock:
                if (
                    self._closing
                    or self._closed
                    or self._context_key(chunk.context) in self._cancelled
                ):
                    return
            try:
                await self._invoke_sink(sink, feature)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "speech feature sink failed; keeping audio playback: %s", type(exc).__name__
                )

    def _put_audio(self, event: AudioChunk) -> None:
        """向有界观察队列写入音频；队列满时丢弃最旧观察项而不阻塞对话。"""

        with self._audio_lock:
            try:
                self._queue.put_nowait(event)
                self._audio_available.set()
                return
            except Full:
                pass
            try:
                self._queue.get_nowait()
            except Empty:
                pass
            try:
                self._queue.put_nowait(event)
                self._audio_available.set()
            except Full:
                # 音频 sink 已在上方收到事件，观察队列仍满时保持文本/语音主流程可用。
                return

    def _drop_context_audio(self, context: ConversationContext | None = None) -> None:
        """从观察队列移除已取消回合，避免旧音频继续被消费。"""

        with self._audio_lock:
            with self._queue.mutex:
                if context is None:
                    self._queue.queue.clear()
                else:
                    retained = [item for item in self._queue.queue if item.context != context]
                    self._queue.queue.clear()
                    self._queue.queue.extend(retained)
                self._queue.not_full.notify_all()
                # 取消回合可能移除了事件刚唤醒的最后一个分片；清除已无
                # 对应队列项的通知，避免 next_audio 在空队列上额外自旋一次。
                if not self._queue.queue:
                    self._audio_available.clear()

    async def next_audio(self) -> AudioChunk:
        """读取任意事件循环可消费的观察分片。"""

        while True:
            try:
                chunk = self._queue.get_nowait()
            except Empty:
                if self._closing or self._closed:
                    raise RuntimeError("TTS coordinator is closed")
                # ``threading.Event.wait`` 放入线程池，等待期间不占用事件循环；
                # timeout 仅用于在关闭状态变更后重新检查终止条件。
                await asyncio.to_thread(self._audio_available.wait, 0.25)
                # 只能在队列仍为空时清除通知。若生产者恰好在 wait 返回
                # 后投递新分片，直接 clear 会吞掉这次唤醒并增加无意义的
                # 250ms 延迟，多个观察消费者同时存在时尤其明显。
                with self._queue.mutex:
                    if not self._queue.queue:
                        self._audio_available.clear()
                continue
            with self._audio_lock:
                if self._context_key(chunk.context) in self._cancelled:
                    continue
            return chunk

    def iter_audio(self, context: ConversationContext | None = None) -> Iterator[AudioChunk]:

        with self._queue.mutex:
            snapshot = tuple(self._queue.queue)
        remaining: list[AudioChunk] = []
        for chunk in snapshot:
            if context is None or self._context_key(chunk.context) == self._context_key(context):
                remaining.append(chunk)
        for item in remaining:
            try:
                self._queue.get_nowait()
            except Empty:  # pragma: no cover - 并发消费竞态下的保守兜底
                continue
            with self._audio_lock:
                if self._context_key(item.context) in self._cancelled:
                    continue
            yield item

    async def stream_audio(
        self,
        context: ConversationContext,
        *,
        timeout_seconds: float = 0.0,
    ) -> AsyncIterator[AudioChunk]:
        """异步拉取音频分片流。

        Args:
            context: 只消费该会话代际的音频；其余分片保留给其它消费者。
            timeout_seconds: 队列连续空闲超过该秒数时结束迭代；0 表示一直
                等待到协调器关闭，适合流式播放的长期消费者。

        Yields:
            按到达顺序的 :class:`.SpeechChunk` 音频分片。
        """

        idle = 0.0
        while not (self._closing or self._closed):
            try:
                chunk = self._queue.get_nowait()
            except Empty:
                if timeout_seconds > 0 and idle >= timeout_seconds:
                    return
                await asyncio.to_thread(self._audio_available.wait, 0.25)
                idle += 0.25
                with self._queue.mutex:
                    if not self._queue.queue:
                        self._audio_available.clear()
                continue
            idle = 0.0
            key = self._context_key(chunk.context)
            if key != self._context_key(context):
                # 不属于本回合的分片留在队列中供其它消费者读取。先取锁
                # 再写队列以保持 ``_audio_lock → _queue.mutex`` 顺序，
                # 与 ``_drop_context_audio``/``next_audio`` 的取锁顺序一致。
                with self._audio_lock:
                    with self._queue.mutex:
                        self._queue.queue.appendleft(chunk)
                        self._queue.not_full.notify_all()
                continue
            with self._audio_lock:
                if key in self._cancelled:
                    continue
            yield chunk
        if self._closing or self._closed:
            raise RuntimeError("TTS coordinator is closed")

    def cancel(self, context: ConversationContext) -> None:
        if self._closing or self._closed:
            return
        task_key = self._context_key(context)
        with self._audio_lock:
            if task_key not in self._cancelled:
                self._cancelled.add(task_key)
                self._cancelled_order.append(task_key)
            # 取消墓碑必须优先保留仍有活动任务的上下文，避免任意 set.pop()
            # 恰好移除当前回合并让迟到音频穿透。无活动墓碑超过上限时按 FIFO 淘汰。
            while len(self._cancelled) > 1024 and self._cancelled_order:
                oldest = self._cancelled_order.popleft()
                if oldest not in self._cancelled:
                    continue
                if oldest in self._active_tasks:
                    self._cancelled_order.append(oldest)
                    if all(item in self._active_tasks for item in self._cancelled_order):
                        break
                    continue
                self._cancelled.discard(oldest)
            self._segments.pop(task_key, None)
            self._context_routes.pop(task_key, None)
            self._context_route_roles.pop(task_key, None)
            self._pinned_contexts.discard(task_key)
            self._release_pending.discard(task_key)
            self._drop_context_audio(context)
        current_loop: asyncio.AbstractEventLoop | None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        with self._audio_lock:
            active_tasks = tuple(self._active_tasks.get(task_key, ()))
            task_loops = {task: self._task_loops.get(task) for task in active_tasks}
        for task in active_tasks:
            task_loop = task_loops.get(task)
            if task_loop is not None and task_loop is not current_loop and task_loop.is_running():
                try:
                    task_loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    task.cancel()
            else:
                task.cancel()
        sink_cancel = getattr(self._audio_sink, "cancel", None)
        if callable(sink_cancel):
            try:
                sink_cancel(context)
            except Exception as exc:
                logger.warning("speech audio sink cancellation failed: %s", type(exc).__name__)

    def clear_cancelled(self, context: ConversationContext) -> None:
        with self._audio_lock:
            key = self._context_key(context)
            self._cancelled.discard(key)
            if self._cancelled_order:
                self._cancelled_order = deque(item for item in self._cancelled_order if item != key)
        sink_clear = getattr(self._audio_sink, "clear_cancelled", None)
        if callable(sink_clear):
            try:
                sink_clear(context)
            except Exception as exc:
                logger.warning("speech audio sink tombstone cleanup failed: %s", type(exc).__name__)

    async def aclose(self) -> None:
        """共享实际清理任务；取消当前等待不会中断后端卸载。"""

        with self._audio_lock:
            if self._closed:
                return
            close_task = self._close_task
            if close_task is not None and close_task.done() and not close_task.cancelled():
                close_task.exception()
            if close_task is None or close_task.done():
                self._closing = True
                self._started = False
                close_task = asyncio.create_task(
                    self._close_after_start(),
                    name="meapet-tts-coordinator-close",
                )
                self._close_task = close_task
        await asyncio.shield(close_task)

    async def _close_after_start(self) -> None:
        """等待正在执行的启动 single-flight，再运行一次真实清理。"""

        async with self._start_lock:
            await self._close_once()

    async def _cancel_active_tasks(self, tasks: tuple[asyncio.Task[object], ...]) -> None:
        """在线程安全的所属事件循环中取消活动语音任务并等待其收尾。

        协调器的队列和路由支持跨事件循环调用；关闭也可能由配置观察器线程
        发起。asyncio.gather 不能直接接收另一个事件循环拥有的任务，因此
        跨循环任务通过所属循环上的取消回调，再以线程安全 Future 回传完成。
        """

        if not tasks:
            return
        current_loop = asyncio.get_running_loop()
        local_tasks: list[asyncio.Task[object]] = []
        cross_loop_waiters: list[asyncio.Future[None]] = []

        for task in tasks:
            if task.done():
                continue
            task_loop = self._task_loops.get(task)
            if task_loop is None:
                try:
                    task_loop = task.get_loop()
                except (AttributeError, RuntimeError):
                    task_loop = None
            if task_loop is current_loop or task_loop is None:
                task.cancel()
                if task_loop is current_loop:
                    local_tasks.append(task)
                continue
            if not task_loop.is_running():
                # 所属循环已经停止时无法再驱动 finally；关闭流程会清理
                # 活动任务索引，不把这个任务交给当前循环 await。
                continue

            waiter = current_loop.create_future()

            def resolve_waiter(waiter: asyncio.Future[None] = waiter) -> None:
                if not waiter.done():
                    waiter.set_result(None)

            def notify_done(
                done_task: asyncio.Task[object],
                waiter: asyncio.Future[None] = waiter,
            ) -> None:
                # 消费非取消异常，避免关闭路径留下未取出的任务异常。
                try:
                    done_task.exception()
                except BaseException:
                    pass
                try:
                    current_loop.call_soon_threadsafe(resolve_waiter)
                except RuntimeError:
                    # 当前关闭循环已无法接收回调；等待者也即将终止。
                    return

            def cancel_on_owner_loop(
                owned_task: asyncio.Task[object] = task,
                on_done: Callable[[asyncio.Task[object]], None] = notify_done,
            ) -> None:
                if not owned_task.done():
                    owned_task.cancel()
                if owned_task.done():
                    on_done(owned_task)
                else:
                    owned_task.add_done_callback(on_done)

            try:
                task_loop.call_soon_threadsafe(cancel_on_owner_loop)
            except RuntimeError:
                # 竞态下所属循环刚停止；不能把跨循环 Task 交给当前循环。
                resolve_waiter()
            cross_loop_waiters.append(waiter)

        if local_tasks:
            await asyncio.gather(*local_tasks, return_exceptions=True)
        if cross_loop_waiters:
            await asyncio.gather(*cross_loop_waiters, return_exceptions=True)

    async def _close_once(self) -> None:
        """在启动 single-flight 之后按固定顺序卸载协调器和后端。"""

        with self._audio_lock:
            if self._closed:
                return
            self._started = False
            tasks = tuple(task for active in self._active_tasks.values() for task in active)
        # 唤醒可能正在等待音频的跨循环消费者，使其尽快观察 closed 状态。
        self._audio_available.set()
        await self._cancel_active_tasks(tasks)
        with self._audio_lock:
            self._active_tasks.clear()
            self._task_loops.clear()
            self._segments.clear()
            self._context_routes.clear()
            self._context_route_roles.clear()
            self._pinned_contexts.clear()
            self._release_pending.clear()
            self._serializers.clear()
            self._cancelled.clear()
            self._cancelled_order.clear()
        self._drop_context_audio()
        close_backend = getattr(self.backend, "aclose", None)
        if not callable(close_backend):
            close_backend = getattr(self.backend, "close", None)
        if callable(close_backend):
            try:
                result = close_backend()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log_event(
                    logger,
                    "tts.backend.unload",
                    component="tts.coordinator",
                    status="failed",
                    level=logging.WARNING,
                    fields={"error": type(exc).__name__},
                )
        log_event(
            logger,
            "tts.coordinator.unload",
            component="tts.coordinator",
            status="completed",
        )
        with self._audio_lock:
            # closed 只描述全部活动任务与后端卸载均已结束的真实终态。
            self._closed = True
            self._closing = False

    @staticmethod
    def _context_key(context: ConversationContext) -> tuple[str, str, str, str, int]:
        """返回包含模式与代际的完整语音隔离键。"""

        return (
            context.mode,
            context.profile_id,
            context.session_id,
            context.turn_id,
            context.generation_id,
        )
