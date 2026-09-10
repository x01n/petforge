from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import time
from typing import Any

from core.contracts.chat import ChatMessage, ChatRequest
from core.conversation.persona import PromptProfile
from core.events.types import ConversationContext

from .service import MemoryService, SummarizationBatch

logger = logging.getLogger(__name__)

_COORDINATOR_ATTRIBUTE = "_meapet_memory_summary_coordinator"
_EXTRACTION_MAX_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class _ExtractionItem:
    text: str
    source_id: int
    attempts: int = 0


def enqueue_model_memory_extraction(memory: object, text: str, source_id: int) -> bool:
    """把一次完整用户回合交给已注册协调器；未注册时安全跳过。"""

    coordinator = getattr(memory, _COORDINATOR_ATTRIBUTE, None)
    enqueue = getattr(coordinator, "enqueue_extraction", None)
    if not callable(enqueue):
        return False
    return bool(enqueue(text, source_id))


@dataclass(frozen=True, slots=True)
class MemorySummaryStatus:
    """可安全展示的自动总结状态。"""

    running: bool
    busy: bool
    completed: int
    failures: int
    last_reason: str = ""
    last_status: str = "idle"
    last_run_at: float = 0.0
    extraction_pending: int = 0
    extraction_completed: int = 0
    extraction_failures: int = 0
    extraction_dropped: int = 0
    extraction_last_status: str = "idle"

    def public(self) -> dict[str, object]:
        """返回不含对话正文、模型地址或凭据的状态映射。"""

        return {
            "running": self.running,
            "busy": self.busy,
            "completed": self.completed,
            "failures": self.failures,
            "last_reason": self.last_reason,
            "last_status": self.last_status,
            "last_run_at": self.last_run_at,
            "extraction_pending": self.extraction_pending,
            "extraction_completed": self.extraction_completed,
            "extraction_failures": self.extraction_failures,
            "extraction_dropped": self.extraction_dropped,
            "extraction_last_status": self.extraction_last_status,
        }


class MemorySummaryCoordinator:
    """在后台为未总结对话生成长期记忆摘要。"""

    def __init__(
        self,
        memory: MemoryService,
        router_provider: Callable[[], object],
        *,
        poll_seconds: float = 30.0,
        task: str = "memory",
        max_tokens: int = 768,
        clock: Callable[[], float] = time,
    ) -> None:
        if not isinstance(memory, MemoryService):
            raise TypeError("memory must be a MemoryService")
        if not callable(router_provider):
            raise TypeError("router_provider must be callable")
        self._memory = memory
        self._router_provider = router_provider
        self._poll_seconds = max(1.0, min(float(poll_seconds), 3600.0))
        self._task_name = str(task or "memory").strip() or "memory"
        self._max_tokens = max(64, min(int(max_tokens), 4096))
        default_prompts = PromptProfile()
        self._summary_prompt = default_prompts.memory_summary
        self._extraction_prompt_text = default_prompts.memory_extract
        self._clock = clock
        self._wake = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_lock = asyncio.Lock()
        self._task: asyncio.Task[object] | None = None
        self._closed = False
        self._busy = False
        self._completed = 0
        self._failures = 0
        self._consecutive_failures = 0
        self._last_reason = ""
        self._last_status = "idle"
        self._last_run_at = 0.0
        self._retry_after = 0.0
        self._extraction_queue: deque[_ExtractionItem] = deque()
        self._extraction_busy = False
        self._extraction_completed = 0
        self._extraction_failures = 0
        self._extraction_consecutive_failures = 0
        self._extraction_dropped = 0
        self._extraction_last_status = "idle"
        self._extraction_retry_after = 0.0
        setattr(memory, _COORDINATOR_ATTRIBUTE, self)

    def configure_prompts(self, *, summary: str, extraction: str) -> None:
        """原子替换后续记忆任务提示词，不影响正在进行的请求。"""

        summary_text = str(summary or "").replace("\x00", "").strip()
        extraction_text = str(extraction or "").replace("\x00", "").strip()
        if not summary_text or len(summary_text) > 8_000:
            raise ValueError("memory summary prompt is invalid")
        if not extraction_text or len(extraction_text) > 8_000:
            raise ValueError("memory extraction prompt is invalid")
        self._summary_prompt = summary_text
        self._extraction_prompt_text = extraction_text

    @property
    def running(self) -> bool:
        task = self._task
        return bool(task is not None and not task.done() and not self._closed)

    def status(self) -> MemorySummaryStatus:
        """读取协调器状态；不会触发模型请求。"""

        return MemorySummaryStatus(
            running=self.running,
            busy=self._busy or self._extraction_busy,
            completed=self._completed,
            failures=self._failures,
            last_reason=self._last_reason,
            last_status=self._last_status,
            last_run_at=self._last_run_at,
            extraction_pending=len(self._extraction_queue),
            extraction_completed=self._extraction_completed,
            extraction_failures=self._extraction_failures,
            extraction_dropped=self._extraction_dropped,
            extraction_last_status=self._extraction_last_status,
        )

    async def start(self) -> None:
        """启动单一轮询任务；重复调用保持幂等。"""

        if self._closed:
            raise RuntimeError("memory summary coordinator is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop and not self._loop.is_closed():
            raise RuntimeError("memory summary coordinator is bound to another event loop")
        self._loop = loop
        if self.running:
            return
        self._task = asyncio.create_task(self._run(), name="meapet-memory-summary")
        self._wake.set()

    def wake(self) -> bool:
        """请求尽快检查总结条件；关闭后返回 ``False``。

        可从配置观察线程或其它非 asyncio 线程调用；事件设置会投递到
        创建协调器的事件循环，避免直接跨线程操作 asyncio.Event。
        """

        if self._closed:
            return False
        loop = self._loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if loop is not None and loop is not current_loop:
            if loop.is_closed():
                return False
            try:
                loop.call_soon_threadsafe(self._wake.set)
            except RuntimeError:
                return False
            return True
        self._wake.set()
        return True

    async def pause(self) -> None:
        """暂停可重试的后台轮询；保留待提取队列和协调器所有权。

        该路径只用于运行时启动事务回滚；与 stop 不同，它不会封闭协调器，
        因此后续 start 可以重新建立轮询任务。正在进行的模型请求被取消时，
        原有提取批次仍由 run_extraction_once 的取消分支原样放回队列。
        """

        if self._closed:
            return
        task = self._task
        self._task = None
        self._wake.clear()
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)
        self._loop = None
        self._busy = False
        self._extraction_busy = False

    async def stop(self) -> None:
        """停止后台任务；不会取消已经写入的摘要事务。"""

        self._closed = True
        self._wake.set()
        task = self._task
        self._task = None
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)
        self._loop = None
        self._busy = False
        self._extraction_busy = False
        self._extraction_queue.clear()
        if getattr(self._memory, _COORDINATOR_ATTRIBUTE, None) is self:
            delattr(self._memory, _COORDINATOR_ATTRIBUTE)

    def _extraction_limits(self) -> tuple[int, int, int]:
        settings = self._memory.settings
        item_limit = max(0, int(settings.extract_max_items))
        queue_limit = max(16, min(128, max(1, item_limit) * 16))
        batch_limit = max(1, min(4, item_limit))
        input_chars = max(256, min(4_000, int(settings.extract_max_chars) * 8))
        return queue_limit, batch_limit, input_chars

    def enqueue_extraction(self, text: str, source_id: int) -> bool:
        """有界保存一次完整用户消息；不在调用线程中发起模型请求。"""

        settings = self._memory.settings
        if (
            self._closed
            or not self._memory.enabled
            or not settings.auto_extract_enabled
            or settings.extract_max_items <= 0
        ):
            return False
        content = str(text or "").replace("\x00", "").strip()
        try:
            identity = int(source_id)
        except (TypeError, ValueError, OverflowError):
            return False
        if not content or identity <= 0:
            return False
        queue_limit, _, input_chars = self._extraction_limits()
        content = content[:input_chars]
        while len(self._extraction_queue) >= queue_limit:
            self._extraction_queue.popleft()
            self._extraction_dropped += 1
        self._extraction_queue.append(_ExtractionItem(content, identity))
        self._wake.set()
        return True

    def _take_extraction_batch(self) -> tuple[_ExtractionItem, ...]:
        settings = self._memory.settings
        if not self._memory.enabled or not settings.auto_extract_enabled:
            self._extraction_dropped += len(self._extraction_queue)
            self._extraction_queue.clear()
            return ()
        _, batch_limit, _ = self._extraction_limits()
        return tuple(
            self._extraction_queue.popleft()
            for _ in range(min(batch_limit, len(self._extraction_queue)))
        )

    def _restore_extraction_batch(
        self,
        batch: Sequence[_ExtractionItem],
        *,
        increment_attempt: bool = True,
    ) -> None:
        restored: list[_ExtractionItem] = []
        for item in batch:
            attempts = item.attempts + int(increment_attempt)
            if attempts < _EXTRACTION_MAX_ATTEMPTS:
                restored.append(_ExtractionItem(item.text, item.source_id, attempts))
            else:
                self._extraction_dropped += 1
        queue_limit, _, _ = self._extraction_limits()
        for item in reversed(restored):
            if len(self._extraction_queue) >= queue_limit:
                self._extraction_queue.pop()
                self._extraction_dropped += 1
            self._extraction_queue.appendleft(item)

    async def _run(self) -> None:
        try:
            while not self._closed:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=self._poll_seconds)
                except TimeoutError:
                    pass
                self._wake.clear()
                if self._closed:
                    return
                try:
                    await self.run_extraction_once()
                    await self.run_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # 两条 ``run_*_once`` 路径已处理预期的模型和存储异常；
                    # 这里兜住其他运行期错误，避免轮询任务永久退出。日志
                    # 只能保留异常类型，不能带出聊天正文或连接信息。
                    self._record_failure(exc, reason="coordinator")
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _batch_prompt(batch: SummarizationBatch) -> str:
        """把角色化消息转为有界纯文本，不插入工具或系统指令。"""

        labels = {"user": "用户", "assistant": "桌宠", "system": "系统"}
        lines: list[str] = []
        remaining = 12_000
        for item in batch.messages:
            role = labels.get(str(item.get("role", "")).strip().lower(), "对话")
            content = str(item.get("content", "") or "").replace("\x00", "").strip()
            if not content or remaining <= 0:
                continue
            content = content[:remaining]
            lines.append(f"{role}：{content}")
            remaining -= len(content)
        return "\n".join(lines)

    async def _generate(self, batch: SummarizationBatch) -> str:
        router = self._router_provider()
        resolve = getattr(router, "resolve", None)
        complete = getattr(router, "complete", None)
        if not callable(resolve) or not callable(complete):
            raise RuntimeError("memory summary model router is unavailable")
        selection = resolve(self._task_name)
        primary = getattr(selection, "primary", None)
        model = str(getattr(primary, "selected_model", "") or "").strip()
        if not model:
            raise RuntimeError("memory summary model is unavailable")
        transcript = self._batch_prompt(batch)
        if not transcript:
            raise RuntimeError("memory summary batch is empty")
        request = ChatRequest(
            model=model,
            messages=(
                ChatMessage(
                    "system",
                    self._summary_prompt,
                ),
                ChatMessage("user", transcript),
            ),
            temperature=0.2,
            max_tokens=self._max_tokens,
            tools=(),
            tool_choice="none",
            metadata={"purpose": "memory_summary", "trigger": batch.reason},
        )
        context = ConversationContext(
            "system",
            "memory-summary",
            f"summary-{uuid.uuid4().hex}",
            0,
        )
        response = complete(request, task=self._task_name, context=context)
        if inspect.isawaitable(response):
            response = await response
        text = str(getattr(response, "text", "") or "").replace("\x00", "").strip()
        if not text:
            raise RuntimeError("memory summary model returned no text")
        return text[:4_000]

    @staticmethod
    def _extraction_prompt(batch: Sequence[_ExtractionItem]) -> str:
        payload = [{"item": index, "text": item.text} for index, item in enumerate(batch, start=1)]
        return json.dumps(payload, ensure_ascii=False, allow_nan=False)

    def _parse_extraction_response(self, value: object) -> tuple[dict[str, Any], ...]:
        settings = self._memory.settings
        text = str(value or "").replace("\x00", "").strip()
        response_limit = max(
            4_096,
            min(32_000, settings.extract_max_items * (settings.extract_max_chars + 512)),
        )
        if not text or len(text) > response_limit:
            raise ValueError("memory extraction response is empty or oversized")
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("memory extraction response must be a JSON array")
        if not parsed:
            return ()
        allowed_fields = {"content", "priority", "confidence", "tags", "rule"}
        result: list[dict[str, Any]] = []
        for item in parsed[: settings.extract_max_items]:
            if not isinstance(item, Mapping) or set(item) - allowed_fields:
                continue
            content = item.get("content")
            if not isinstance(content, str):
                continue
            content = content.strip()[: settings.extract_max_chars]
            if not content:
                continue
            normalized: dict[str, Any] = {"content": content}
            priority = item.get("priority")
            if priority is not None:
                if isinstance(priority, bool) or not isinstance(priority, int):
                    continue
                normalized["priority"] = max(0, min(10, priority))
            confidence = item.get("confidence")
            if confidence is not None:
                if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                    continue
                confidence_value = float(confidence)
                if not math.isfinite(confidence_value):
                    continue
                normalized["confidence"] = max(0.0, min(1.0, confidence_value))
            tags = item.get("tags")
            if tags is not None:
                if (
                    isinstance(tags, (str, bytes, bytearray))
                    or not isinstance(tags, Sequence)
                    or not all(isinstance(tag, str) for tag in tags)
                ):
                    continue
                normalized["tags"] = tuple(tags[:16])
            rule = item.get("rule")
            if rule is not None:
                if not isinstance(rule, str):
                    continue
                normalized["rule"] = rule[:128]
            result.append(normalized)
        if not result:
            raise ValueError("memory extraction response contains no valid items")
        return tuple(result)

    async def _generate_extractions(
        self,
        batch: Sequence[_ExtractionItem],
    ) -> tuple[dict[str, Any], ...]:
        router = self._router_provider()
        resolve = getattr(router, "resolve", None)
        complete = getattr(router, "complete", None)
        if not callable(resolve) or not callable(complete):
            raise RuntimeError("memory extraction model router is unavailable")
        selection = resolve(self._task_name)
        primary = getattr(selection, "primary", None)
        model = str(getattr(primary, "selected_model", "") or "").strip()
        if not model:
            raise RuntimeError("memory extraction model is unavailable")
        request = ChatRequest(
            model=model,
            messages=(
                ChatMessage(
                    "system",
                    self._extraction_prompt_text
                    + f"\n最多返回 {self._memory.settings.extract_max_items} 项。",
                ),
                ChatMessage("user", self._extraction_prompt(batch)),
            ),
            temperature=0.0,
            max_tokens=min(
                self._max_tokens,
                max(128, self._memory.settings.extract_max_items * 160),
            ),
            tools=(),
            tool_choice="none",
            metadata={"purpose": "memory_extract", "batch_size": len(batch)},
        )
        context = ConversationContext(
            "system",
            "memory-extract",
            f"extract-{uuid.uuid4().hex}",
            0,
        )
        response = complete(request, task=self._task_name, context=context)
        if inspect.isawaitable(response):
            response = await response
        return self._parse_extraction_response(getattr(response, "text", ""))

    def _record_extraction_failure(
        self,
        exc: Exception,
        *,
        occurred_at: float,
    ) -> dict[str, object]:
        self._extraction_failures += 1
        self._extraction_consecutive_failures += 1
        self._extraction_last_status = "failed"
        delay = min(
            900.0,
            self._poll_seconds * (2 ** min(self._extraction_consecutive_failures, 5)),
        )
        self._extraction_retry_after = occurred_at + delay
        error_name = type(exc).__name__
        logger.warning("动态记忆提取失败：%s", error_name)
        return {
            "status": "failed",
            "error": error_name,
            "pending": len(self._extraction_queue),
        }

    async def run_extraction_once(self, *, force: bool = False) -> Mapping[str, Any]:
        """最多消费一个模型提取微批次，不阻塞对话完成路径。"""

        if self._closed:
            return {"status": "closed"}
        async with self._run_lock:
            try:
                now = float(self._clock())
            except Exception as exc:
                return self._record_extraction_failure(exc, occurred_at=float(time()))
            if not force and now < self._extraction_retry_after:
                return {"status": "deferred", "retry_after": self._extraction_retry_after}
            batch = self._take_extraction_batch()
            if not batch:
                self._extraction_last_status = "idle"
                return {"status": "idle"}
            self._extraction_busy = True
            self._extraction_last_status = "running"
            try:
                extracted = await self._generate_extractions(batch)
                promoted = self._memory.promote_extracted_batch(
                    extracted,
                    source="conversation:model_extract",
                    source_ids=tuple(item.source_id for item in batch),
                )
            except asyncio.CancelledError:
                self._restore_extraction_batch(batch, increment_attempt=False)
                self._extraction_last_status = "cancelled"
                raise
            except Exception as exc:
                self._restore_extraction_batch(batch)
                return self._record_extraction_failure(exc, occurred_at=now)
            finally:
                self._extraction_busy = False
            self._extraction_completed += 1
            self._extraction_consecutive_failures = 0
            self._extraction_retry_after = 0.0
            self._extraction_last_status = "completed"
            if self._extraction_queue:
                self._wake.set()
            return {
                "status": "completed",
                "source_count": len(batch),
                "promoted": len(promoted),
                "pending": len(self._extraction_queue),
            }

    def _record_failure(
        self,
        exc: Exception,
        *,
        reason: str,
        occurred_at: float | None = None,
    ) -> dict[str, str]:
        """记录不包含异常正文的失败状态，并计算有界退避时间。"""

        if occurred_at is None:
            try:
                failure_time = float(self._clock())
            except Exception:
                failure_time = float(time())
        else:
            failure_time = float(occurred_at)
        self._failures += 1
        self._consecutive_failures += 1
        self._last_reason = str(reason)
        self._last_status = "failed"
        self._last_run_at = failure_time
        delay = min(900.0, self._poll_seconds * (2 ** min(self._consecutive_failures, 5)))
        self._retry_after = failure_time + delay
        error_name = type(exc).__name__
        logger.warning("自动记忆总结失败：%s", error_name)
        return {
            "status": "failed",
            "reason": str(reason),
            "error": error_name,
        }

    async def run_once(self, *, force: bool = False) -> Mapping[str, Any]:
        """检查并最多提交一个批次；适合测试、调度器和手动诊断。"""

        if self._closed:
            return {"status": "closed"}
        async with self._run_lock:
            try:
                now = float(self._clock())
            except Exception as exc:
                return self._record_failure(exc, reason="coordinator")
            if not force and now < self._retry_after:
                return {"status": "deferred", "retry_after": self._retry_after}
            try:
                batch = self._memory.prepare_summarization_batch(now=now, force=force)
            except Exception as exc:
                return self._record_failure(exc, reason="prepare", occurred_at=now)
            if batch is None:
                self._last_status = "idle"
                return {"status": "idle"}
            self._busy = True
            self._last_reason = batch.reason
            self._last_status = "running"
            self._last_run_at = now
            try:
                summary = await self._generate(batch)
                memory_id = self._memory.store_summary(
                    summary,
                    batch.source_ids,
                    trigger_reason=batch.reason,
                    summarized_at=float(self._clock()),
                )
                if memory_id <= 0:
                    self._last_status = "stale"
                    return {"status": "stale", "reason": batch.reason}
            except asyncio.CancelledError:
                self._last_status = "cancelled"
                raise
            except Exception as exc:
                return self._record_failure(
                    exc,
                    reason=batch.reason,
                    occurred_at=now,
                )
            finally:
                self._busy = False
            self._completed += 1
            self._consecutive_failures = 0
            self._retry_after = 0.0
            self._last_status = "completed"
            return {
                "status": "completed",
                "reason": batch.reason,
                "memory_id": memory_id,
                "source_count": len(batch.source_ids),
            }


__all__ = [
    "MemorySummaryCoordinator",
    "MemorySummaryStatus",
    "enqueue_model_memory_extraction",
]
