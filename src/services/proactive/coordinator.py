from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from logger.events import log_event
from services.scheduler import normalize_trigger_conditions, trigger_conditions_match

logger = logging.getLogger(__name__)

_MAX_RULES = 64
_MAX_EVENT_NAME = 64
_MAX_INSTRUCTION_CHARS = 600
_MAX_PAYLOAD_TEXT = 256
_CONVERSATION_WAIT_TIMEOUT_SECONDS = 0.5
_PAYLOAD_FIELDS = frozenset(
    {
        "status",
        "active",
        "state",
        "source",
        "observed_at",
        "idle_for_seconds",
        "active_for_seconds",
        "user_active",
        "window_id",
        "title",
        "app_id",
        "pid",
        "process_name",
        "geometry",
        "task_id",
        "name",
        "expression",
        "scheduled_at",
        "label",
        "value",
    }
)


def _number(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} is outside the allowed range")
    return result


def _integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    number = _number(value, name=name, minimum=minimum, maximum=maximum)
    integer = int(number)
    if integer != number:
        raise ValueError(f"{name} must be an integer")
    return integer


def _event_name(value: object) -> str:
    text = str(value or "").strip().lower()
    if (
        not text
        or len(text) > _MAX_EVENT_NAME
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in text)
    ):
        raise ValueError("proactive event name is invalid")
    return text


@dataclass(frozen=True, slots=True)
class ProactiveRule:
    rule_id: str
    event_name: str
    instruction: str
    conditions: Mapping[str, Any]
    debounce_seconds: float = 0.0
    cooldown_seconds: float = 0.0
    enabled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, index: int) -> ProactiveRule:
        if not isinstance(value, Mapping):
            raise ValueError("proactive rule must be a mapping")
        allowed = {
            "id",
            "event",
            "instruction",
            "conditions",
            "debounce_seconds",
            "cooldown_seconds",
            "enabled",
        }
        if set(value) - allowed:
            raise ValueError("proactive rule contains unsupported fields")
        rule_id = str(value.get("id", f"rule-{index + 1}") or "").strip()
        if not rule_id or len(rule_id) > 128 or any(c in rule_id for c in "\r\n\x00"):
            raise ValueError("proactive rule id is invalid")
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("proactive rule enabled must be a boolean")
        instruction = str(value.get("instruction", "") or "").strip()
        if not instruction or len(instruction) > _MAX_INSTRUCTION_CHARS or "\x00" in instruction:
            raise ValueError("proactive rule instruction is invalid")
        return cls(
            rule_id=rule_id,
            event_name=_event_name(value.get("event")),
            instruction=instruction,
            conditions=normalize_trigger_conditions(value.get("conditions", {})),
            debounce_seconds=_number(
                value.get("debounce_seconds", 0.0),
                name="proactive rule debounce_seconds",
                minimum=0.0,
                maximum=86400.0,
            ),
            cooldown_seconds=_number(
                value.get("cooldown_seconds", 0.0),
                name="proactive rule cooldown_seconds",
                minimum=0.0,
                maximum=86400.0,
            ),
            enabled=enabled,
        )


@dataclass(frozen=True, slots=True)
class ProactiveSettings:
    enabled: bool = False
    hourly_budget: int = 6
    daily_budget: int = 24
    global_cooldown_seconds: float = 15.0
    dedupe_seconds: float = 60.0
    max_pending_events: int = 32
    memory_context_chars: int = 2400
    max_tool_rounds: int = 2
    rules: tuple[ProactiveRule, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> ProactiveSettings:
        values = value if isinstance(value, Mapping) else {}
        allowed = {
            "enabled",
            "hourly_budget",
            "daily_budget",
            "global_cooldown_seconds",
            "dedupe_seconds",
            "max_pending_events",
            "memory_context_chars",
            "max_tool_rounds",
            "rules",
        }
        if set(values) - allowed:
            raise ValueError("proactive configuration contains unsupported fields")
        enabled = values.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("proactive.enabled must be a boolean")
        raw_rules = values.get("rules", ())
        if not isinstance(raw_rules, Sequence) or isinstance(raw_rules, (str, bytes, bytearray)):
            raise ValueError("proactive.rules must be a list")
        if len(raw_rules) > _MAX_RULES:
            raise ValueError("proactive rule limit exceeded")
        rules = tuple(
            ProactiveRule.from_mapping(item, index=index)
            for index, item in enumerate(raw_rules)
            if isinstance(item, Mapping)
        )
        if len(rules) != len(raw_rules) or len({rule.rule_id for rule in rules}) != len(rules):
            raise ValueError("proactive rules contain invalid or duplicate ids")
        return cls(
            enabled=enabled,
            hourly_budget=_integer(
                values.get("hourly_budget", 6),
                name="proactive.hourly_budget",
                minimum=1,
                maximum=120,
            ),
            daily_budget=_integer(
                values.get("daily_budget", 24),
                name="proactive.daily_budget",
                minimum=1,
                maximum=1000,
            ),
            global_cooldown_seconds=_number(
                values.get("global_cooldown_seconds", 15.0),
                name="proactive.global_cooldown_seconds",
                minimum=0.0,
                maximum=86400.0,
            ),
            dedupe_seconds=_number(
                values.get("dedupe_seconds", 60.0),
                name="proactive.dedupe_seconds",
                minimum=0.0,
                maximum=86400.0,
            ),
            max_pending_events=_integer(
                values.get("max_pending_events", 32),
                name="proactive.max_pending_events",
                minimum=1,
                maximum=256,
            ),
            memory_context_chars=_integer(
                values.get("memory_context_chars", 2400),
                name="proactive.memory_context_chars",
                minimum=512,
                maximum=12000,
            ),
            max_tool_rounds=_integer(
                values.get("max_tool_rounds", 2),
                name="proactive.max_tool_rounds",
                minimum=1,
                maximum=4,
            ),
            rules=rules,
        )


@dataclass(frozen=True, slots=True)
class ProactiveEvent:
    event_name: str
    rule: ProactiveRule
    payload: Mapping[str, Any]
    fingerprint: str
    observed_at: float


class ProactiveCoordinator:
    """把脱敏事件串行提交给独立 proactive 对话任务。"""

    def __init__(
        self,
        conversation: object,
        *,
        memory: object,
        affection: object,
        settings: ProactiveSettings | None = None,
        activity_provider: Callable[[], object] | None = None,
        gate_provider: Callable[[], Mapping[str, object]] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.conversation = conversation
        self._memory = memory
        self._affection = affection
        self._settings = settings or ProactiveSettings()
        self._activity_provider = activity_provider
        self._gate_provider = gate_provider
        self._clock = clock or time.time
        self._queue: deque[ProactiveEvent] = deque()
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # 主动事件在用户对话期间仍保留在有界队列；对话终态通过
        # ``wake`` 唤醒等待任务，避免轮询或事件永久滞留。
        self._conversation_idle_event = asyncio.Event()
        self._conversation_wake_generation = 0
        self._closed = False
        self._dedupe: dict[str, float] = {}
        self._rule_last: dict[str, float] = {}
        self._hourly_runs: deque[float] = deque()
        self._daily_key = ""
        self._daily_runs = 0
        self._last_run_at = 0.0
        self._last_status = "disabled" if not self._settings.enabled else "idle"
        self._accepted = 0
        self._completed = 0
        self._dropped = 0
        self._budget_rejections = 0

    @property
    def settings(self) -> ProactiveSettings:
        return self._settings

    @property
    def running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    def reconfigure(self, settings: ProactiveSettings) -> None:
        if not isinstance(settings, ProactiveSettings):
            raise TypeError("settings must be ProactiveSettings")
        # 热重载会使旧主动回合和旧审批失效；否则暂停中的模型上下文会
        # 在新路由/规则下继续执行，且旧审批卡会永久占住活动门禁。
        pending = tuple(self.pending_approvals_for_ui())
        task = self._worker
        if task is not None and not task.done():
            task.cancel()
        self._queue.clear()
        for request in pending:
            approval_id = str(getattr(request, "approval_id", "") or "").strip()
            if not approval_id:
                continue
            try:
                getattr(self.conversation, "deny_approval")(approval_id)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("proactive approval cleanup failed", exc_info=True)
        self._conversation_wake_generation += 1
        self._conversation_idle_event.set()
        self._settings = settings
        # 取消是异步的；不能用 ``running`` 判断新配置是否已完成切换，
        # 否则旧 task 尚未进入 finally 时会把状态永久留在
        # ``waiting_for_conversation``/``running``。wait_idle() 会继续等待
        # 旧 task 收尾，之后空队列保持 idle。
        self._last_status = "disabled" if not settings.enabled else "idle"

    @staticmethod
    def _safe_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        values = payload if isinstance(payload, Mapping) else {}
        result: dict[str, Any] = {}
        for key in _PAYLOAD_FIELDS:
            if key not in values:
                continue
            value = values[key]
            if value is None or isinstance(value, (bool, int, float)):
                result[key] = value
            elif isinstance(value, str):
                result[key] = value.replace("\x00", "")[:_MAX_PAYLOAD_TEXT]
            elif key == "geometry" and isinstance(value, Sequence):
                result[key] = [int(item) for item in tuple(value)[:4] if isinstance(item, int)]
        return result

    @staticmethod
    def _fingerprint(event_name: str, rule_id: str, payload: Mapping[str, Any]) -> str:
        stable = {
            key: payload.get(key)
            for key in ("process_name", "app_id", "title", "task_id", "label", "value")
            if key in payload
        }
        encoded = json.dumps(
            {"event": event_name, "rule": rule_id, "payload": stable},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _gate_open(
        self,
        rule: ProactiveRule,
        payload: Mapping[str, Any],
        *,
        allow_dialogue_active: bool = False,
    ) -> bool:
        try:
            gate = self._gate_provider() if callable(self._gate_provider) else {}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            # 门禁状态读取失败时必须拒绝主动行为；不能让 worker 因宿主
            # 状态异常反复崩溃并自旋重建。
            return False
        if isinstance(gate, Mapping):
            blocked_keys = ("closed", "closing", "dragging", "locked")
            if any(gate.get(key) is True for key in blocked_keys):
                return False
            # ``conversation_finished``/``conversation_failed`` 终态事件由
            # ConversationService 在 finally 清理活动任务前发出；允许它们
            # 先入队，worker 会在下一拍确认对话已空闲，避免终态触发器被
            # 当前事件回调中的 ``dialogue_active`` 门禁吞掉。
            if not allow_dialogue_active and gate.get("dialogue_active") is True:
                return False
        if rule.conditions.get("require_user_active") is True:
            provider = self._activity_provider
            try:
                active = (
                    bool(provider()) if callable(provider) else payload.get("user_active") is True
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                active = False
            if not active:
                return False
        return True

    def _budget_available(self, now: float) -> bool:
        while self._hourly_runs and now - self._hourly_runs[0] >= 3600:
            self._hourly_runs.popleft()
        day_key = datetime.fromtimestamp(now).date().isoformat()
        if day_key != self._daily_key:
            self._daily_key = day_key
            self._daily_runs = 0
        return (
            len(self._hourly_runs) < self._settings.hourly_budget
            and self._daily_runs < self._settings.daily_budget
        )

    async def notify(
        self,
        event_name: object,
        payload: Mapping[str, Any] | None = None,
        *,
        instruction: str | None = None,
    ) -> Mapping[str, object]:
        if self._closed:
            return {"status": "closed"}
        settings = self._settings
        if not settings.enabled:
            return {"status": "disabled"}
        name = _event_name(event_name)
        safe_payload = self._safe_payload(payload)
        matching = tuple(
            rule
            for rule in settings.rules
            if rule.enabled
            and rule.event_name == name
            and trigger_conditions_match(rule.conditions, safe_payload)
        )
        if instruction is not None:
            direct_instruction = str(instruction or "").strip()
            if not direct_instruction or len(direct_instruction) > _MAX_INSTRUCTION_CHARS:
                return {"status": "rejected", "reason": "instruction"}
            matching = (
                ProactiveRule(
                    "direct",
                    name,
                    direct_instruction,
                    {},
                    enabled=True,
                ),
            )
        if not matching:
            return {"status": "ignored", "reason": "no_matching_rule"}
        try:
            now = float(self._clock())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
            return {"status": "unavailable", "reason": "clock"}
        if not math.isfinite(now):
            return {"status": "unavailable", "reason": "clock"}
        allow_dialogue_active = name in {"conversation_finished", "conversation_failed"}
        accepted = 0
        rejection_status = ""
        for rule in matching:
            if not self._gate_open(
                rule,
                safe_payload,
                allow_dialogue_active=allow_dialogue_active,
            ):
                self._dropped += 1
                rejection_status = rejection_status or "blocked"
                continue
            fingerprint = self._fingerprint(name, rule.rule_id, safe_payload)
            dedupe_window = max(settings.dedupe_seconds, rule.debounce_seconds)
            last_seen = self._dedupe.get(fingerprint, 0.0)
            if last_seen and now - last_seen < dedupe_window:
                self._dropped += 1
                rejection_status = rejection_status or "duplicate"
                continue
            rule_last = self._rule_last.get(rule.rule_id, 0.0)
            cooldown = max(settings.global_cooldown_seconds, rule.cooldown_seconds)
            if rule_last and now - rule_last < cooldown:
                self._dropped += 1
                rejection_status = rejection_status or "cooldown"
                continue
            if len(self._queue) >= settings.max_pending_events:
                self._dropped += 1
                rejection_status = rejection_status or "queue_full"
                continue
            self._dedupe[fingerprint] = now
            if len(self._dedupe) > 512:
                oldest = sorted(self._dedupe.items(), key=lambda item: item[1])[:128]
                for key, _value in oldest:
                    self._dedupe.pop(key, None)
            self._queue.append(ProactiveEvent(name, rule, safe_payload, fingerprint, now))
            self._accepted += 1
            accepted += 1
        if accepted <= 0:
            if rejection_status == "blocked":
                return {"status": "blocked", "reason": "gate"}
            return {"status": rejection_status or "ignored"}
        self._ensure_worker()
        result: dict[str, object] = {"status": "queued", "pending": len(self._queue)}
        if accepted > 1:
            result["accepted"] = accepted
        if rejection_status:
            result["rejected"] = rejection_status
        return result

    def _conversation_is_active(self) -> bool:
        """读取对话活动状态；读取失败时按活动处理并让主动队列等待。"""

        try:
            checker = getattr(self.conversation, "has_active_conversation")
            return bool(checker())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            # 对话状态读取异常时保持等待，避免主动 worker 因 finally
            # 自动重建而自旋消耗 CPU。
            return True

    def _ensure_worker(self) -> None:
        if self._closed or not self._settings.enabled or self.running:
            return
        if not self._queue:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._worker = asyncio.create_task(self._run(), name="meapet-proactive-coordinator")

    def wake(self) -> None:
        """通知主动队列重新检查用户对话是否已经空闲。"""

        if self._closed:
            return
        loop = self._loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if loop is not None and loop is not current_loop and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._wake_in_loop)
            except RuntimeError:
                return
            return
        self._wake_in_loop()

    def _wake_in_loop(self) -> None:
        if self._closed:
            return
        self._conversation_wake_generation += 1
        self._conversation_idle_event.set()
        self._ensure_worker()

    def _event_prompt(self, event: ProactiveEvent) -> str:
        payload_text = json.dumps(event.payload, ensure_ascii=False, sort_keys=True)
        query = " ".join(
            str(event.payload.get(key, "") or "")
            for key in ("process_name", "app_id", "title", "label", "value")
        ).strip()
        memory_prompt = ""
        builder = getattr(self._memory, "build_context_prompt", None)
        if callable(builder):
            try:
                memory_prompt = str(
                    builder(query, max_chars=self._settings.memory_context_chars) or ""
                )[: self._settings.memory_context_chars]
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                memory_prompt = ""
        try:
            affection = int(getattr(self._affection, "get_affection")())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            affection = 0
        sections = [
            "这是桌宠后台主动事件，不是用户输入。只在确有帮助时简短回应；需要行动时调用现有工具。",
            "不得声称执行未获得成功回执的操作，中高风险工具必须等待用户审批。",
            f"事件类型：{event.event_name}",
            f"事件摘要：{payload_text}",
            f"本次目标：{event.rule.instruction}",
            f"当前好感度：{max(0, min(100, affection))}/100",
        ]
        if memory_prompt:
            sections.append(memory_prompt)
        return "\n\n".join(sections)[:8000]

    async def _run(self) -> None:
        try:
            while self._queue and not self._closed and self._settings.enabled:
                # 对话进行中只暂停主动回合，不丢弃已验收事件。对话终态会
                # 调用 ``wake``；代际校验避免 wake 与 clear 的时序竞态。
                while self._conversation_is_active():
                    self._last_status = "waiting_for_conversation"
                    observed_generation = self._conversation_wake_generation
                    self._conversation_idle_event.clear()
                    if observed_generation != self._conversation_wake_generation:
                        continue
                    if not self._conversation_is_active():
                        break
                    try:
                        await asyncio.wait_for(
                            self._conversation_idle_event.wait(),
                            timeout=_CONVERSATION_WAIT_TIMEOUT_SECONDS,
                        )
                    except TimeoutError:
                        # TurnFinished 回调可能早于 ConversationService.finally；
                        # 低频复查确保错位唤醒不会让队列永久滞留。
                        continue
                    if self._closed or not self._settings.enabled:
                        return
                if not self._queue:
                    break
                event = self._queue.popleft()
                now = float(self._clock())
                if not self._gate_open(
                    event.rule,
                    event.payload,
                    allow_dialogue_active=event.event_name
                    in {"conversation_finished", "conversation_failed"},
                ):
                    self._dropped += 1
                    continue
                if not self._budget_available(now):
                    self._budget_rejections += 1
                    self._last_status = "budget_exhausted"
                    continue
                if self._last_run_at and now - self._last_run_at < max(
                    self._settings.global_cooldown_seconds,
                    event.rule.cooldown_seconds,
                ):
                    self._dropped += 1
                    continue
                self._hourly_runs.append(now)
                self._daily_runs += 1
                self._last_run_at = now
                self._rule_last[event.rule.rule_id] = now
                self._last_status = "running"
                log_event(
                    logger,
                    "proactive.event.started",
                    component="proactive.coordinator",
                    status="started",
                    fields={"event": event.event_name, "rule_id": event.rule.rule_id},
                )
                try:
                    result = await getattr(self.conversation, "complete")(
                        self._event_prompt(event),
                        profile_id="system",
                        session_id="proactive",
                        mode="agent",
                    )
                except asyncio.CancelledError:
                    self._last_status = "cancelled"
                    raise
                except Exception as exc:
                    self._last_status = "failed"
                    log_event(
                        logger,
                        "proactive.event.failed",
                        component="proactive.coordinator",
                        status="failed",
                        level=logging.WARNING,
                        reason_code=type(exc).__name__,
                        fields={"event": event.event_name, "rule_id": event.rule.rule_id},
                    )
                    continue
                status = str(getattr(result, "status", "completed") or "completed")
                self._last_status = status
                if status == "approval_required":
                    return
                if status in {"completed", "tool_limit"}:
                    self._completed += 1
        finally:
            self._worker = None
            if self._queue and not self._closed and self._settings.enabled:
                self._ensure_worker()

    def pending_approvals_for_ui(self) -> tuple[object, ...]:
        getter = getattr(self.conversation, "pending_approvals_for_ui", None)
        return tuple(getter()) if callable(getter) else ()

    def owns_approval(self, approval_id: str) -> bool:
        key = str(approval_id or "").strip()
        return any(
            str(getattr(item, "approval_id", "")) == key for item in self.pending_approvals_for_ui()
        )

    async def continue_approval(self, approval_id: str, *, grant_session: bool = False) -> object:
        result = await getattr(self.conversation, "continue_approval")(
            approval_id,
            grant_session=grant_session,
        )
        self._ensure_worker()
        return result

    def deny_approval(self, approval_id: str) -> bool:
        denied = bool(getattr(self.conversation, "deny_approval")(approval_id))
        self._ensure_worker()
        return denied

    def diagnostics(self) -> dict[str, object]:
        return {
            "status": self._last_status,
            "enabled": self._settings.enabled,
            "running": self.running,
            "pending_events": len(self._queue),
            "waiting_for_conversation": self._last_status == "waiting_for_conversation",
            "rule_count": len(self._settings.rules),
            "hourly_used": len(self._hourly_runs),
            "hourly_budget": self._settings.hourly_budget,
            "daily_used": self._daily_runs,
            "daily_budget": self._settings.daily_budget,
            "accepted": self._accepted,
            "completed": self._completed,
            "dropped": self._dropped,
            "budget_rejections": self._budget_rejections,
        }

    async def wait_idle(self) -> None:
        """等待当前及其取消后重建的 worker 全部结束。"""

        # ``_run`` 的 finally 可能在旧 task 结束瞬间根据仍保留的队列
        # 创建新 task；只等待一次会把调用方错误地告知队列已经空闲。
        while True:
            task = self._worker
            if task is None:
                return
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)
            if self._worker is task:
                return

    def interrupt(self, *, clear_queue: bool = True) -> None:
        """让真实用户活动抢占后台主动模型回合。

        Qt 事件回调可能运行在 asyncio 线程之外；对已绑定的运行时循环必须
        通过 call_soon_threadsafe 取消任务，不能直接跨线程操作 Task/Event。
        """

        loop = self._loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if loop is not None and loop is not current_loop and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._interrupt_in_loop, clear_queue)
            except RuntimeError:
                # 事件循环刚好关闭时，不能再操作其中的任务；关闭路径会
                # 负责最终回收资源。
                return
            return
        self._interrupt_in_loop(clear_queue)

    def _interrupt_in_loop(self, clear_queue: bool) -> None:
        if clear_queue:
            self._queue.clear()
        task = self._worker
        if task is not None and not task.done():
            task.cancel()
        if not self._closed:
            self._last_status = "interrupted"

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.clear()
        task = self._worker
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for request in self.pending_approvals_for_ui():
            approval_id = str(getattr(request, "approval_id", "") or "").strip()
            if approval_id:
                self.deny_approval(approval_id)
        self._last_status = "closed"
        self._loop = None
