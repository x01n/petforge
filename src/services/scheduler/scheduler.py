"""无外部 cron 依赖的可取消定时任务服务。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import secrets
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from datetime import time as datetime_time
from typing import Any

from logger.events import log_event

from .persistence import SchedulerStateStore

logger = logging.getLogger(__name__)


class ScheduleExpressionError(ValueError):
    pass


def parse_interval(expression: str) -> float:
    """解析 every:<number><ms|s|m|h> 固定间隔表达式。"""

    raw_value = str(expression or "")
    if any(ord(char) in (0, 10, 13) for char in raw_value):
        raise ScheduleExpressionError("schedule expression is invalid")
    raw = raw_value.strip().lower()
    if not raw or len(raw) > 128:
        raise ScheduleExpressionError("schedule expression is invalid")
    match = re.fullmatch(r"every:(\d+(?:\.\d+)?)(ms|s|m|h)", raw)
    if match is None:
        raise ScheduleExpressionError("only every:<number><ms|s|m|h> is supported")
    try:
        amount = float(match.group(1))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScheduleExpressionError("schedule interval is invalid") from exc
    unit = match.group(2)
    multiplier = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    seconds = amount * multiplier
    if not math.isfinite(seconds) or seconds < 0.1 or seconds > 31_536_000:
        raise ScheduleExpressionError("schedule interval is outside the allowed range")
    return seconds


_DAILY_PATTERN = re.compile(r"daily:(\d{2}):(\d{2})")


def _normalize_schedule_expression(expression: object) -> str:
    raw_value = str(expression or "")
    if any(ord(char) in (0, 10, 13) for char in raw_value):
        raise ScheduleExpressionError("schedule expression is invalid")
    raw = raw_value.strip().lower()
    if not raw or len(raw) > 128:
        raise ScheduleExpressionError("schedule expression is invalid")
    return raw


def _daily_parts(expression: str) -> tuple[int, int]:
    match = _DAILY_PATTERN.fullmatch(expression)
    if match is None:
        raise ScheduleExpressionError("daily schedule must use daily:HH:MM")
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        raise ScheduleExpressionError("daily schedule time is outside the allowed range")
    return hour, minute


def validate_schedule_expression(expression: object) -> str:
    """校验固定间隔或本地时区每日时刻表达式并返回规范化文本。"""

    raw = _normalize_schedule_expression(expression)
    if raw.startswith("every:"):
        parse_interval(raw)
        return raw
    if raw.startswith("daily:"):
        _daily_parts(raw)
        return raw
    raise ScheduleExpressionError("only every:<number><ms|s|m|h> or daily:HH:MM is supported")


def next_schedule_at(expression: object, now: object) -> float:
    """返回表达式在本地时钟下严格晚于 ``now`` 的下一次时间戳。"""

    raw = validate_schedule_expression(expression)
    if isinstance(now, bool):
        raise ScheduleExpressionError("schedule clock must be a finite timestamp")
    try:
        current = float(now)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ScheduleExpressionError("schedule clock must be a finite timestamp") from exc
    if not math.isfinite(current):
        raise ScheduleExpressionError("schedule clock must be a finite timestamp")
    if raw.startswith("every:"):
        result = current + parse_interval(raw)
        if not math.isfinite(result):
            raise ScheduleExpressionError("schedule next time is invalid")
        return result

    hour, minute = _daily_parts(raw)
    try:
        current_local = datetime.fromtimestamp(current)
        target_local = datetime.combine(
            current_local.date(), datetime_time(hour=hour, minute=minute)
        )
        target = target_local.timestamp()
        if target <= current:
            target = (target_local + timedelta(days=1)).timestamp()
    except (OverflowError, OSError, ValueError) as exc:
        raise ScheduleExpressionError("daily schedule time is unavailable") from exc
    if not math.isfinite(target) or target <= current:
        raise ScheduleExpressionError("daily schedule next time is invalid")
    return target


@dataclass
class ScheduledTask:
    task_id: str
    name: str
    expression: str
    action: Mapping[str, Any]
    owner: str
    next_run_at: float
    enabled: bool = True
    last_run_at: float = 0.0
    run_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def interval_seconds(self) -> float:
        # 保留旧调用方读取该属性的兼容语义；每日时刻不是固定秒数
        # （夏令时可能改变当天长度），实际调度始终使用 ``next_run_after``。
        if str(self.expression or "").strip().lower().startswith("daily:"):
            return 86400.0
        return parse_interval(self.expression)

    def next_run_after(self, now: object) -> float:
        """计算本任务在指定时刻之后的下一次运行时间。"""

        return next_schedule_at(self.expression, now)


class SchedulerService:
    def __init__(
        self,
        *,
        action_runner: Callable[[Mapping[str, Any], ScheduledTask], Awaitable[Any] | Any]
        | None = None,
        clock=time.time,
        max_tasks: int = 128,
        state_store: SchedulerStateStore | None = None,
        activity_provider: Callable[[], object] | None = None,
    ) -> None:
        self._tasks: dict[str, ScheduledTask] = {}
        self._action_runner = action_runner
        self._clock = clock
        self._max_tasks = max(1, min(int(max_tasks), 1024))
        self._loop_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._last_error = ""
        self._state_store: SchedulerStateStore | None = None
        self._persistence_error = ""
        self._activity_provider = activity_provider
        self._last_skip: dict[str, object] | None = None
        if state_store is not None:
            self.attach_state_store(state_store)

    @staticmethod
    def _validate_metadata(value: object) -> dict[str, Any]:
        """校验任务元数据；活跃条件必须是明确的布尔值。"""

        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("task metadata must be a mapping")
        metadata = dict(value)
        if "require_user_active" in metadata and not isinstance(
            metadata["require_user_active"], bool
        ):
            raise ValueError("task metadata.require_user_active must be a boolean")
        try:
            json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("task metadata must be JSON serializable") from exc
        return metadata

    def set_activity_provider(self, provider: Callable[[], object] | None) -> None:
        """设置同步用户活跃探针；探针异常按非活跃处理。"""

        self._activity_provider = provider

    def _user_is_active(self) -> bool:
        provider = self._activity_provider
        if not callable(provider):
            return False
        try:
            value = provider()
        except Exception:
            return False
        if inspect.isawaitable(value):
            closer = getattr(value, "close", None)
            if callable(closer):
                closer()
            return False
        if isinstance(value, Mapping):
            status = str(value.get("status", "") or "").strip().lower()
            if status in {"unavailable", "unknown", "disabled"}:
                return False
            value = value.get("active", value.get("user_active", False))
        return value is True

    def _now(self, value: object | None = None) -> float:
        """读取有限时间戳，拒绝坏时钟污染下一次运行时间。"""

        raw = self._clock() if value is None else value
        if isinstance(raw, bool):
            raise ValueError("scheduler clock must return a finite timestamp")
        try:
            current = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scheduler clock must return a finite timestamp") from exc
        if not math.isfinite(current):
            raise ValueError("scheduler clock must return a finite timestamp")
        return current

    def attach_state_store(
        self, state_store: SchedulerStateStore | None, *, restore: bool = True
    ) -> None:
        """绑定可选状态存储；恢复失败只记录诊断，不阻塞内存调度。"""

        self._state_store = state_store
        self._persistence_error = ""
        if state_store is None or not restore:
            return
        loader = getattr(state_store, "load_tasks", None)
        if not callable(loader):
            self._persistence_error = "scheduler state store does not implement load_tasks"
            return
        try:
            snapshots = loader()
        except Exception as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"
            return
        for snapshot in snapshots or ():
            self._restore_snapshot(snapshot)

    def _restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """校验并恢复单个任务快照；坏记录不影响其他任务。"""

        try:
            if not isinstance(snapshot, Mapping):
                raise ValueError("task snapshot must be a mapping")
            task_id = str(snapshot.get("task_id", "")).strip()
            name = str(snapshot.get("name", "")).strip()
            expression = str(snapshot.get("expression", "")).strip().lower()
            owner = str(snapshot.get("owner", "")).strip()
            action = snapshot.get("action")
            enabled = snapshot.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError("task snapshot enabled must be a boolean")
            if (
                not task_id
                or len(task_id) > 128
                or any(char in task_id for char in "\r\n\x00")
                or not name
                or len(name) > 128
                or not owner
                or len(owner) > 256
                or any(char in owner for char in "\r\n\x00")
                or not isinstance(action, Mapping)
                or not action
            ):
                raise ValueError("task snapshot identifiers are invalid")
            validate_schedule_expression(expression)
            next_run_at = float(snapshot.get("next_run_at"))
            last_run_at = float(snapshot.get("last_run_at", 0.0))
            run_count = int(snapshot.get("run_count", 0))
            if not math.isfinite(next_run_at) or not math.isfinite(last_run_at):
                raise ValueError("task snapshot timestamps are invalid")
            if last_run_at < 0 or run_count < 0:
                raise ValueError("task snapshot counters are invalid")
            metadata = snapshot.get("metadata", {})
            metadata = self._validate_metadata(metadata)
            json.dumps(dict(action), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"
            return
        if task_id in self._tasks or len(self._tasks) >= self._max_tasks:
            return
        self._tasks[task_id] = ScheduledTask(
            task_id=task_id,
            name=name,
            expression=expression,
            action=dict(action),
            owner=owner,
            next_run_at=next_run_at,
            enabled=enabled,
            last_run_at=last_run_at,
            run_count=run_count,
            metadata=metadata,
        )

    @staticmethod
    def _snapshot(task: ScheduledTask) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "name": task.name,
            "expression": task.expression,
            "action": dict(task.action),
            "owner": task.owner,
            "next_run_at": task.next_run_at,
            "enabled": task.enabled,
            "last_run_at": task.last_run_at,
            "run_count": task.run_count,
            "metadata": dict(task.metadata),
        }

    def _persist_task(self, task: ScheduledTask) -> None:
        store = self._state_store
        saver = getattr(store, "save_task", None) if store is not None else None
        if not callable(saver):
            return
        try:
            saver(self._snapshot(task))
        except Exception as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"

    def _delete_persisted_task(self, task_id: str) -> None:
        store = self._state_store
        remover = getattr(store, "delete_task", None) if store is not None else None
        if not callable(remover):
            return
        try:
            remover(task_id)
        except Exception as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"

    def _record_run(
        self,
        task: ScheduledTask,
        *,
        started_at: float,
        finished_at: float,
        status: str,
        error_text: str = "",
    ) -> None:
        log_event(
            logger,
            "scheduler.task.completed",
            component="scheduler",
            status=status,
            duration_ms=max(0.0, (float(finished_at) - float(started_at)) * 1000.0),
            fields={"task_id": task.task_id, "run_count": task.run_count},
        )
        store = self._state_store
        recorder = getattr(store, "record_run", None) if store is not None else None
        if not callable(recorder):
            return
        try:
            recorder(
                kind="task",
                item_id=task.task_id,
                owner=task.owner,
                started_at=started_at,
                finished_at=finished_at,
                status=status,
                error_text=error_text,
                payload={"run_count": task.run_count, "expression": task.expression},
            )
        except Exception as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"

    def upsert(
        self,
        *,
        task_id: str | None,
        name: str,
        expression: str,
        action: Mapping[str, Any],
        owner: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw_name = str(name or "")
        raw_owner = str(owner or "")
        if any(char in raw_name + raw_owner for char in "\r\n\x00"):
            raise ValueError("task name and owner are invalid")
        safe_name = raw_name.strip()
        safe_owner = raw_owner.strip()
        if not safe_name or len(safe_name) > 128 or not safe_owner or len(safe_owner) > 256:
            raise ValueError("task name and owner are invalid")
        raw_expression = str(expression or "")
        if any(char in raw_expression for char in "\r\n\x00"):
            raise ValueError("task expression is invalid")
        normalized_expression = raw_expression.strip().lower()
        if not normalized_expression or len(normalized_expression) > 128:
            raise ValueError("task expression is invalid")
        validate_schedule_expression(normalized_expression)
        if not isinstance(action, Mapping) or not action:
            raise ValueError("task action must be a non-empty mapping")
        try:
            json.dumps(dict(action), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("task action must be JSON serializable") from exc
        supplied_metadata = metadata is not None
        normalized_metadata = self._validate_metadata(metadata)
        if task_id is None:
            task_id = f"task-{secrets.token_urlsafe(8)}"
        raw_task_id = str(task_id)
        if any(char in raw_task_id for char in "\r\n\x00"):
            raise ValueError("task_id is invalid")
        task_id = raw_task_id.strip()
        if not task_id or len(task_id) > 128:
            raise ValueError("task_id is invalid")
        existing = self._tasks.get(task_id)
        if existing is not None and existing.owner != safe_owner:
            raise PermissionError("task owner mismatch")
        if existing is None and len(self._tasks) >= self._max_tasks:
            raise ValueError("scheduler task limit exceeded")
        next_run_at = next_schedule_at(normalized_expression, self._now())
        if existing is not None and existing.expression == normalized_expression:
            next_run_at = existing.next_run_at
        task = ScheduledTask(
            task_id=task_id,
            name=safe_name[:128],
            expression=normalized_expression,
            action=dict(action),
            owner=safe_owner[:256],
            next_run_at=next_run_at,
            enabled=existing.enabled if existing is not None else True,
            last_run_at=existing.last_run_at if existing is not None else 0.0,
            run_count=existing.run_count if existing is not None else 0,
            metadata=(
                normalized_metadata
                if supplied_metadata
                else dict(existing.metadata)
                if existing is not None
                else {}
            ),
        )
        self._tasks[task_id] = task
        self._persist_task(task)
        return self.describe(task)

    def remove(self, task_id: str, *, owner: str | None = None) -> bool:
        task = self._tasks.get(str(task_id or "").strip())
        if task is None or (owner is not None and task.owner != owner):
            return False
        del self._tasks[task.task_id]
        self._delete_persisted_task(task.task_id)
        return True

    def set_enabled(self, task_id: str, enabled: bool, *, owner: str | None = None) -> bool:
        task = self._tasks.get(str(task_id or "").strip())
        if task is None or (owner is not None and task.owner != owner):
            return False
        task.enabled = bool(enabled)
        self._persist_task(task)
        return True

    def list_tasks(self, *, owner: str | None = None) -> tuple[dict[str, Any], ...]:
        return tuple(
            self.describe(task)
            for task in sorted(
                self._tasks.values(), key=lambda item: (item.next_run_at, item.task_id)
            )
            if owner is None or task.owner == owner
        )

    @staticmethod
    def describe(task: ScheduledTask) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "name": task.name,
            "expression": task.expression,
            "owner": task.owner,
            "enabled": task.enabled,
            "next_run_at": task.next_run_at,
            "last_run_at": task.last_run_at,
            "run_count": task.run_count,
            "metadata": dict(task.metadata),
        }

    async def tick(self, *, now: float | None = None) -> tuple[str, ...]:
        current = self._now() if now is None else self._now(now)
        due = [
            task for task in self._tasks.values() if task.enabled and task.next_run_at <= current
        ]
        executed: list[str] = []
        for task in sorted(due, key=lambda item: (item.next_run_at, item.task_id)):
            started_at = current
            task.last_run_at = current
            task.run_count += 1
            task.next_run_at = task.next_run_after(current)
            self._persist_task(task)
            if bool(task.metadata.get("require_user_active", False)) and not self._user_is_active():
                # 到期点只消费一次，避免用户长期离开时每个 poll 都重复排队。
                # 这是正常的策略跳过，不进入 ``last_error``，但写入同一有界
                # 执行审计，便于控制台解释“为什么没有提醒”。
                self._last_skip = {
                    "task_id": task.task_id,
                    "reason": "user_inactive",
                    "at": current,
                }
                self._record_run(
                    task,
                    started_at=started_at,
                    finished_at=self._now(),
                    status="skipped_inactive",
                    error_text="user is not active",
                )
                executed.append(task.task_id)
                continue
            run_status = "completed"
            run_error = ""
            if self._action_runner is not None:
                try:
                    result = self._action_runner(dict(task.action), task)
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    run_status = "cancelled"
                    self._record_run(
                        task,
                        started_at=started_at,
                        finished_at=self._now(),
                        status=run_status,
                    )
                    raise
                except Exception as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    run_status = "failed"
                    run_error = self._last_error
            self._record_run(
                task,
                started_at=started_at,
                finished_at=self._now(),
                status=run_status,
                error_text=run_error,
            )
            executed.append(task.task_id)
        return tuple(executed)

    def status(self) -> dict[str, object]:
        """返回调度器生命周期和最近一次动作错误。"""

        return {
            "running": self._loop_task is not None and not self._loop_task.done(),
            "task_count": len(self._tasks),
            "last_error": self._last_error,
            "persistence": "attached" if self._state_store is not None else "detached",
            "persistence_error": self._persistence_error,
            "last_skip": dict(self._last_skip) if self._last_skip is not None else None,
            "activity": self._activity_status(),
        }

    def _activity_status(self) -> object:
        provider = self._activity_provider
        owner = getattr(provider, "__self__", None) if callable(provider) else None
        status = getattr(owner, "status", None)
        if callable(status):
            try:
                value = status()
            except Exception:
                return {"status": "unavailable"}
            return dict(value) if isinstance(value, Mapping) else {"status": "unavailable"}
        return {"status": "unavailable" if provider is None else "unknown"}

    async def start(self, *, poll_seconds: float = 0.5) -> None:
        if self._loop_task is not None and not self._loop_task.done():
            return
        try:
            interval = float(poll_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("scheduler poll_seconds is invalid") from exc
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("scheduler poll_seconds is invalid")
        self._stop_event = asyncio.Event()
        self._loop_task = asyncio.create_task(self._run_loop(max(0.05, interval)))

    async def _run_loop(self, poll_seconds: float) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=poll_seconds)
            except TimeoutError:
                pass

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._loop_task
        if task is not None:
            if task is not asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._loop_task = None
        self._stop_event = None
