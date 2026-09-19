"""启动、交互、窗口变化和空闲等事件触发器。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from logger.events import log_event

from .persistence import SchedulerStateStore

logger = logging.getLogger(__name__)

_CONDITION_TEXT_FIELDS = ("process_name", "app_id", "title")
_MATCH_MODES = frozenset({"exact", "prefix", "contains", "regex"})
_MAX_PATTERN_CHARS = 128
_MAX_MATCH_TEXT_CHARS = 256
_UNSAFE_REGEX = re.compile(r"\(\?|\\[1-9]|\{|\([^)]*[+*][^)]*\)[+*?]")


def normalize_trigger_conditions(value: object) -> dict[str, Any]:
    """严格规范化事件条件；正则只允许有界、无回溯扩展的子集。"""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("trigger conditions must be a mapping")
    allowed = {
        *_CONDITION_TEXT_FIELDS,
        "min_duration_seconds",
        "require_user_active",
        "cooldown_seconds",
    }
    unknown = set(value) - allowed
    if unknown:
        raise ValueError("trigger conditions contain unsupported fields")
    result: dict[str, Any] = {}
    for field_name in _CONDITION_TEXT_FIELDS:
        raw = value.get(field_name)
        if raw in (None, ""):
            continue
        if isinstance(raw, str):
            mode = "exact"
            pattern = raw
        elif isinstance(raw, Mapping):
            mode = str(raw.get("mode", "exact") or "exact").strip().lower()
            pattern = raw.get("value", "")
        else:
            raise ValueError(f"trigger condition {field_name} is invalid")
        if mode not in _MATCH_MODES or not isinstance(pattern, str):
            raise ValueError(f"trigger condition {field_name} is invalid")
        pattern = pattern.strip()
        if not pattern or len(pattern) > _MAX_PATTERN_CHARS or "\x00" in pattern:
            raise ValueError(f"trigger condition {field_name} is invalid")
        if mode == "regex":
            if (
                _UNSAFE_REGEX.search(pattern)
                or any(character in pattern for character in "(){}|")
                or sum(pattern.count(character) for character in "*+") > 4
            ):
                raise ValueError("trigger regex uses an unsupported construct")
            try:
                re.compile(pattern, flags=re.IGNORECASE)
            except re.error as exc:
                raise ValueError("trigger regex is invalid") from exc
        result[field_name] = {"mode": mode, "value": pattern}
    for field_name, maximum in (
        ("min_duration_seconds", 7 * 24 * 60 * 60),
        ("cooldown_seconds", 24 * 60 * 60),
    ):
        raw = value.get(field_name, 0.0)
        if isinstance(raw, bool):
            raise ValueError(f"trigger condition {field_name} is invalid")
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"trigger condition {field_name} is invalid") from exc
        if not math.isfinite(number) or not 0 <= number <= maximum:
            raise ValueError(f"trigger condition {field_name} is out of range")
        if number:
            result[field_name] = number
    require_active = value.get("require_user_active", False)
    if not isinstance(require_active, bool):
        raise ValueError("trigger condition require_user_active must be a boolean")
    if require_active:
        result["require_user_active"] = True
    return result


def trigger_conditions_match(
    conditions: Mapping[str, Any] | None,
    payload: Mapping[str, Any] | None,
) -> bool:
    """在固定字段和有界文本上匹配一个事件 payload。"""

    normalized = normalize_trigger_conditions(conditions)
    values = payload if isinstance(payload, Mapping) else {}
    if normalized.get("require_user_active") is True and values.get("user_active") is not True:
        return False
    duration = values.get("active_for_seconds", values.get("duration_seconds", 0.0))
    try:
        duration_value = float(duration)
    except (TypeError, ValueError, OverflowError):
        duration_value = 0.0
    if duration_value < float(normalized.get("min_duration_seconds", 0.0)):
        return False
    for field_name in _CONDITION_TEXT_FIELDS:
        rule = normalized.get(field_name)
        if not isinstance(rule, Mapping):
            continue
        actual = str(values.get(field_name, "") or "")[:_MAX_MATCH_TEXT_CHARS]
        expected = str(rule.get("value", "") or "")
        mode = str(rule.get("mode", "exact") or "exact")
        folded_actual = actual.casefold()
        folded_expected = expected.casefold()
        if mode == "exact" and folded_actual != folded_expected:
            return False
        if mode == "prefix" and not folded_actual.startswith(folded_expected):
            return False
        if mode == "contains" and folded_expected not in folded_actual:
            return False
        if mode == "regex" and re.search(expected, actual, flags=re.IGNORECASE) is None:
            return False
    return True


@dataclass
class Trigger:
    trigger_id: str
    event_name: str
    action: Mapping[str, Any]
    owner: str
    debounce_seconds: float = 0.0
    last_fired_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class TriggerService:
    def __init__(
        self,
        *,
        action_runner: Callable[
            [Mapping[str, Any], Trigger, Mapping[str, Any]], Awaitable[Any] | Any
        ]
        | None = None,
        clock=time.time,
        max_triggers: int = 128,
        state_store: SchedulerStateStore | None = None,
    ) -> None:
        self._triggers: dict[str, Trigger] = {}
        self._action_runner = action_runner
        self._clock = clock
        self._max_triggers = max(1, min(int(max_triggers), 1024))
        self._last_error = ""
        self._last_event_name = ""
        self._last_trigger_id = ""
        self._last_status = "idle"
        self._last_duration_ms: float | None = None
        self._event_count = 0
        self._matched_count = 0
        self._completed_count = 0
        self._failed_count = 0
        self._skipped_count = 0
        self._state_store: SchedulerStateStore | None = None
        self._persistence_error = ""
        if state_store is not None:
            self.attach_state_store(state_store)

    @staticmethod
    def _validate_metadata(value: object) -> dict[str, Any]:
        """校验并复制触发器元数据。"""

        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("trigger metadata must be a mapping")
        metadata = dict(value)
        try:
            json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("trigger metadata must be JSON serializable") from exc
        if "conditions" in metadata:
            metadata["conditions"] = normalize_trigger_conditions(metadata["conditions"])
        return metadata

    def _now(self, value: object | None = None) -> float:
        """读取有限时间戳，避免坏时钟绕过触发器去抖。"""

        raw = self._clock() if value is None else value
        if isinstance(raw, bool):
            raise ValueError("trigger clock must return a finite timestamp")
        try:
            current = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("trigger clock must return a finite timestamp") from exc
        if not math.isfinite(current):
            raise ValueError("trigger clock must return a finite timestamp")
        return current

    def attach_state_store(
        self, state_store: SchedulerStateStore | None, *, restore: bool = True
    ) -> None:
        """绑定可选状态存储；恢复错误不会阻塞触发器启动。"""

        self._state_store = state_store
        self._persistence_error = ""
        if state_store is None or not restore:
            return
        loader = getattr(state_store, "load_triggers", None)
        if not callable(loader):
            self._persistence_error = "scheduler state store does not implement load_triggers"
            return
        try:
            snapshots = loader()
        except Exception as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"
            return
        for snapshot in snapshots or ():
            self._restore_snapshot(snapshot)

    def _restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        try:
            if not isinstance(snapshot, Mapping):
                raise ValueError("trigger snapshot must be a mapping")
            trigger_id = str(snapshot.get("trigger_id", "")).strip()
            event_name = str(snapshot.get("event_name", "")).strip().lower()
            owner = str(snapshot.get("owner", "")).strip()
            action = snapshot.get("action")
            if (
                not trigger_id
                or len(trigger_id) > 128
                or any(char in trigger_id for char in "\r\n\x00")
                or not event_name
                or len(event_name) > 128
                or any(char in event_name for char in "\r\n\x00")
                or not owner
                or len(owner) > 256
                or any(char in owner for char in "\r\n\x00")
                or not isinstance(action, Mapping)
                or not action
            ):
                raise ValueError("trigger snapshot identifiers are invalid")
            debounce = float(snapshot.get("debounce_seconds", 0.0))
            last_fired_at = float(snapshot.get("last_fired_at", 0.0))
            metadata = self._validate_metadata(snapshot.get("metadata", {}))
            if (
                not math.isfinite(debounce)
                or not 0 <= debounce <= 86400
                or not math.isfinite(last_fired_at)
                or last_fired_at < 0
            ):
                raise ValueError("trigger snapshot values are invalid")
            json.dumps(dict(action), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"
            return
        if trigger_id in self._triggers or len(self._triggers) >= self._max_triggers:
            return
        self._triggers[trigger_id] = Trigger(
            trigger_id=trigger_id,
            event_name=event_name,
            action=dict(action),
            owner=owner,
            debounce_seconds=debounce,
            last_fired_at=last_fired_at,
            metadata=dict(metadata),
        )

    @staticmethod
    def _snapshot(trigger: Trigger) -> dict[str, Any]:
        return {
            "trigger_id": trigger.trigger_id,
            "event_name": trigger.event_name,
            "action": dict(trigger.action),
            "owner": trigger.owner,
            "debounce_seconds": trigger.debounce_seconds,
            "last_fired_at": trigger.last_fired_at,
            "metadata": dict(trigger.metadata),
        }

    def _persist_trigger(self, trigger: Trigger) -> None:
        store = self._state_store
        saver = getattr(store, "save_trigger", None) if store is not None else None
        if not callable(saver):
            return
        try:
            saver(self._snapshot(trigger))
        except Exception as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"

    def _delete_persisted_trigger(self, trigger_id: str) -> None:
        store = self._state_store
        remover = getattr(store, "delete_trigger", None) if store is not None else None
        if not callable(remover):
            return
        try:
            remover(trigger_id)
        except Exception as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"

    def _record_run(
        self,
        trigger: Trigger,
        *,
        started_at: float,
        finished_at: float,
        status: str,
        error_text: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        store = self._state_store
        recorder = getattr(store, "record_run", None) if store is not None else None
        if not callable(recorder):
            return
        try:
            recorder(
                kind="trigger",
                item_id=trigger.trigger_id,
                owner=trigger.owner,
                started_at=started_at,
                finished_at=finished_at,
                status=status,
                error_text=error_text,
                payload={
                    "event_name": trigger.event_name,
                    "action_identity": trigger.action.get("identity", ""),
                    "payload_fields": len(payload or {}),
                },
            )
        except Exception as exc:
            self._persistence_error = f"{type(exc).__name__}: {exc}"

    def register(
        self,
        *,
        trigger_id: str,
        event_name: str,
        action: Mapping[str, Any],
        owner: str,
        debounce_seconds: float = 0.0,
        metadata: Mapping[str, Any] | None = None,
    ) -> Trigger:
        raw_trigger_id = str(trigger_id or "")
        raw_event_name = str(event_name or "")
        raw_owner = str(owner or "")
        if any(char in raw_trigger_id + raw_event_name + raw_owner for char in "\r\n\x00"):
            raise ValueError("trigger identifiers are invalid")
        trigger_id = raw_trigger_id.strip()
        event_name = raw_event_name.strip().lower()
        owner = raw_owner.strip()
        if (
            not trigger_id
            or len(trigger_id) > 128
            or not event_name
            or len(event_name) > 128
            or not owner
            or len(owner) > 256
        ):
            raise ValueError("trigger identifiers are invalid")
        if trigger_id not in self._triggers and len(self._triggers) >= self._max_triggers:
            raise ValueError("trigger limit exceeded")
        if not isinstance(action, Mapping) or not action:
            raise ValueError("trigger action must be a non-empty mapping")
        try:
            json.dumps(dict(action), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("trigger action must be JSON serializable") from exc
        supplied_metadata = metadata is not None
        normalized_metadata = self._validate_metadata(metadata)
        try:
            debounce = float(debounce_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("trigger debounce_seconds is invalid") from exc
        if not math.isfinite(debounce) or not 0 <= debounce <= 86400:
            raise ValueError("trigger debounce_seconds is out of range")
        previous = self._triggers.get(trigger_id)
        if previous is not None and previous.owner != owner:
            raise PermissionError("trigger owner mismatch")
        trigger = Trigger(
            trigger_id=trigger_id,
            event_name=event_name,
            action=dict(action),
            owner=owner,
            debounce_seconds=debounce,
            last_fired_at=previous.last_fired_at if previous else 0.0,
            metadata=(
                normalized_metadata
                if supplied_metadata
                else dict(previous.metadata)
                if previous
                else {}
            ),
        )
        self._triggers[trigger_id] = trigger
        self._persist_trigger(trigger)
        return trigger

    def remove(self, trigger_id: str, *, owner: str | None = None) -> bool:
        trigger = self._triggers.get(str(trigger_id or "").strip())
        if trigger is None or (owner is not None and trigger.owner != owner):
            return False
        del self._triggers[trigger.trigger_id]
        self._delete_persisted_trigger(trigger.trigger_id)
        return True

    async def emit(
        self, event_name: str, payload: Mapping[str, Any] | None = None
    ) -> tuple[str, ...]:
        raw_event_name = str(event_name or "")
        if any(char in raw_event_name for char in "\r\n\x00"):
            raise ValueError("trigger event name is invalid")
        event_name = raw_event_name.strip().lower()
        current = self._now()
        payload = dict(payload or {})
        self._event_count += 1
        self._last_event_name = event_name
        fired: list[str] = []
        for trigger in sorted(self._triggers.values(), key=lambda item: item.trigger_id):
            if trigger.event_name != event_name:
                continue
            conditions = trigger.metadata.get("conditions", {})
            if not trigger_conditions_match(
                conditions if isinstance(conditions, Mapping) else {}, payload
            ):
                self._skipped_count += 1
                log_event(
                    logger,
                    "scheduler.trigger.skipped",
                    component="scheduler.trigger",
                    status="skipped",
                    reason_code="condition",
                    fields={
                        "event_name": event_name,
                        "trigger_id": trigger.trigger_id,
                    },
                )
                continue
            cooldown = (
                float(conditions.get("cooldown_seconds", 0.0))
                if isinstance(conditions, Mapping)
                else 0.0
            )
            debounce = max(trigger.debounce_seconds, cooldown)
            if trigger.last_fired_at and current - trigger.last_fired_at < debounce:
                self._skipped_count += 1
                log_event(
                    logger,
                    "scheduler.trigger.skipped",
                    component="scheduler.trigger",
                    status="skipped",
                    reason_code="cooldown",
                    fields={
                        "event_name": event_name,
                        "trigger_id": trigger.trigger_id,
                        "cooldown_seconds": debounce,
                    },
                )
                continue
            trigger.last_fired_at = current
            self._persist_trigger(trigger)
            started_at = current
            self._matched_count += 1
            self._last_trigger_id = trigger.trigger_id
            self._last_status = "running"
            correlation_id = f"{event_name}:{trigger.trigger_id}:{started_at}"
            log_event(
                logger,
                "scheduler.trigger.started",
                component="scheduler.trigger",
                status="started",
                correlation_id=correlation_id,
                operation_id=trigger.trigger_id,
                fields={
                    "event_name": event_name,
                    "trigger_id": trigger.trigger_id,
                    "owner": trigger.owner,
                    "action_identity": trigger.action.get("identity", ""),
                    "payload_fields": len(payload),
                },
            )
            run_status = "completed"
            run_error = ""
            run_reason = "ok"
            try:
                if self._action_runner is not None:
                    result = self._action_runner(dict(trigger.action), trigger, payload)
                    if inspect.isawaitable(result):
                        await result
            except asyncio.CancelledError:
                finished_at = self._now()
                duration_ms = max(0.0, (finished_at - started_at) * 1000.0)
                self._last_status = "cancelled"
                self._last_duration_ms = duration_ms
                log_event(
                    logger,
                    "scheduler.trigger.cancelled",
                    component="scheduler.trigger",
                    status="cancelled",
                    level=20,
                    correlation_id=correlation_id,
                    operation_id=trigger.trigger_id,
                    duration_ms=duration_ms,
                    reason_code="cancelled",
                    fields={
                        "event_name": event_name,
                        "trigger_id": trigger.trigger_id,
                    },
                )
                self._record_run(
                    trigger,
                    started_at=started_at,
                    finished_at=finished_at,
                    status="cancelled",
                    payload=payload,
                )
                raise
            except Exception as exc:
                # 异常正文可能携带敏感上下文，不回传 UI 与诊断通道；
                # 只保存异常类名，完整堆栈由模块日志经脱敏层输出。
                logger.exception(
                    "trigger action failed trigger_id=%s event=%s",
                    trigger.trigger_id,
                    event_name,
                )
                self._last_error = type(exc).__name__
                run_status = "failed"
                run_reason = type(exc).__name__
                run_error = self._last_error
            finished_at = self._now()
            duration_ms = max(0.0, (finished_at - started_at) * 1000.0)
            self._last_duration_ms = duration_ms
            self._last_status = run_status
            if run_status == "completed":
                self._completed_count += 1
            else:
                self._failed_count += 1
            log_event(
                logger,
                f"scheduler.trigger.{run_status}",
                component="scheduler.trigger",
                status=run_status,
                level=30 if run_status == "failed" else 20,
                correlation_id=correlation_id,
                operation_id=trigger.trigger_id,
                duration_ms=duration_ms,
                reason_code=run_reason,
                fields={
                    "event_name": event_name,
                    "trigger_id": trigger.trigger_id,
                    "owner": trigger.owner,
                    "action_identity": trigger.action.get("identity", ""),
                },
            )
            self._record_run(
                trigger,
                started_at=started_at,
                finished_at=finished_at,
                status=run_status,
                error_text=run_error,
                payload=payload,
            )
            fired.append(trigger.trigger_id)
        return tuple(fired)

    async def emit_user_interaction(
        self, payload: Mapping[str, Any] | None = None
    ) -> tuple[str, ...]:
        """发出标准化用户交互事件。"""

        return await self.emit("user_interaction", payload)

    async def emit_idle(self, payload: Mapping[str, Any] | None = None) -> tuple[str, ...]:
        """发出标准化用户空闲事件。"""

        return await self.emit("idle", payload)

    def status(self) -> dict[str, object]:
        """返回触发器数量、最近运行状态及结构化计数。"""

        return {
            "trigger_count": len(self._triggers),
            "last_error": self._last_error,
            "last_event_name": self._last_event_name,
            "last_trigger_id": self._last_trigger_id,
            "last_status": self._last_status,
            "last_duration_ms": self._last_duration_ms,
            "event_count": self._event_count,
            "matched_count": self._matched_count,
            "completed_count": self._completed_count,
            "failed_count": self._failed_count,
            "skipped_count": self._skipped_count,
            "persistence": "attached" if self._state_store is not None else "detached",
            "persistence_error": self._persistence_error,
        }

    def list_triggers(self, *, owner: str | None = None) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "trigger_id": trigger.trigger_id,
                "event_name": trigger.event_name,
                "owner": trigger.owner,
                "debounce_seconds": trigger.debounce_seconds,
                "last_fired_at": trigger.last_fired_at,
                "metadata": dict(trigger.metadata),
            }
            for trigger in sorted(self._triggers.values(), key=lambda item: item.trigger_id)
            if owner is None or trigger.owner == owner
        )
