from __future__ import annotations

import asyncio
import inspect
import logging
import math
from collections import deque
from collections.abc import Callable, Mapping
from threading import Lock
from time import monotonic

from logger.events import log_event

from .triggers import TriggerService

logger = logging.getLogger(__name__)

MIN_IDLE_SECONDS = 1.0
MAX_IDLE_SECONDS = 7 * 24 * 60 * 60.0
MIN_POLL_SECONDS = 0.05
MAX_POLL_SECONDS = 60.0
MAX_PENDING_INTERACTIONS = 128
SYSTEM_IDLE_PROVIDER_DISABLED = "disabled"
SYSTEM_IDLE_PROVIDER_X11 = "x11"
SYSTEM_IDLE_PROVIDER_WINDOWS = "windows"
SYSTEM_IDLE_PROVIDERS = frozenset(
    {
        SYSTEM_IDLE_PROVIDER_DISABLED,
        SYSTEM_IDLE_PROVIDER_X11,
        SYSTEM_IDLE_PROVIDER_WINDOWS,
    }
)
MIN_SYSTEM_IDLE_THRESHOLD_SECONDS = 1.0
MAX_SYSTEM_IDLE_THRESHOLD_SECONDS = MAX_IDLE_SECONDS


def _finite_number(value: object, *, field_name: str, minimum: float, maximum: float) -> float:
    """解析有限范围内的浮点数。"""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ValueError(f"{field_name} is outside the allowed range")
    return result


def _source(value: object) -> str:
    """规范化事件来源，避免把任意文本写入状态或日志。"""

    result = str(value or "unknown").strip()
    if not result:
        return "unknown"
    if any(char in result for char in "\r\n\x00"):
        raise ValueError("activity source is invalid")
    return result[:64]


def _validate_system_idle_provider(value: object) -> str:
    """校验系统空闲 provider 名称；只接受公开配置中的精确值。"""

    if not isinstance(value, str):
        raise ValueError("system_idle_provider must be one of disabled, x11 or windows")
    result = value.strip()
    if result not in SYSTEM_IDLE_PROVIDERS:
        raise ValueError("system_idle_provider must be one of disabled, x11 or windows")
    return result


class UserActivityTracker:
    """跟踪用户最近一次交互，并在状态边沿投递调度事件。

    ``record_interaction`` 可以从 Qt 主线程或其他线程同步调用；真正的
    触发器执行始终回到 tracker 启动时的 asyncio 事件循环。该类不执行
    工具，所有动作仍由 ``TriggerService`` 的既有权限管线处理。系统空闲
    provider 是可选的，失败时始终按非活跃处理。
    """

    def __init__(
        self,
        *,
        triggers: TriggerService,
        enabled: bool = True,
        idle_seconds: float = 300.0,
        poll_seconds: float = 0.5,
        clock: Callable[[], float] = monotonic,
        max_pending_interactions: int = MAX_PENDING_INTERACTIONS,
        system_idle_provider: str = SYSTEM_IDLE_PROVIDER_DISABLED,
        system_idle_threshold_seconds: float = 300.0,
        system_idle_probe: Callable[[], object] | None = None,
    ) -> None:
        if not isinstance(triggers, TriggerService) and not callable(
            getattr(triggers, "emit", None)
        ):
            raise TypeError("triggers must expose emit")
        self._triggers = triggers
        self._enabled = bool(enabled)
        self._idle_seconds = _finite_number(
            idle_seconds,
            field_name="idle_seconds",
            minimum=MIN_IDLE_SECONDS,
            maximum=MAX_IDLE_SECONDS,
        )
        self._poll_seconds = _finite_number(
            poll_seconds,
            field_name="poll_seconds",
            minimum=MIN_POLL_SECONDS,
            maximum=MAX_POLL_SECONDS,
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._system_idle_provider = _validate_system_idle_provider(system_idle_provider)
        self._system_idle_threshold_seconds = _finite_number(
            system_idle_threshold_seconds,
            field_name="system_idle_threshold_seconds",
            minimum=MIN_SYSTEM_IDLE_THRESHOLD_SECONDS,
            maximum=MAX_SYSTEM_IDLE_THRESHOLD_SECONDS,
        )
        if system_idle_probe is not None and not callable(system_idle_probe):
            raise TypeError("system_idle_probe must be callable")
        self._system_idle_probe = system_idle_probe
        self._system_idle_status = (
            "disabled" if self._system_idle_provider == SYSTEM_IDLE_PROVIDER_DISABLED else "unknown"
        )
        self._last_system_idle_active = False
        self._last_system_idle_at: float | None = None
        try:
            pending_limit = int(max_pending_interactions)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_pending_interactions is invalid") from exc
        if pending_limit < 1:
            raise ValueError("max_pending_interactions is invalid")
        self._max_pending_interactions = min(pending_limit, MAX_PENDING_INTERACTIONS)
        self._lock = Lock()
        self._last_interaction_at: float | None = None
        self._state = "unknown"
        self._last_source = ""
        self._interaction_count = 0
        self._idle_count = 0
        self._last_error = ""
        self._last_event_at: float | None = None
        self._pending_interactions: deque[dict[str, object]] = deque(
            maxlen=self._max_pending_interactions
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._drain_task: asyncio.Task[None] | None = None

    @property
    def enabled(self) -> bool:
        """返回是否启用活跃状态观察。"""

        with self._lock:
            return self._enabled

    @property
    def idle_seconds(self) -> float:
        """返回进入空闲状态所需的无交互秒数。"""

        with self._lock:
            return self._idle_seconds

    @property
    def poll_seconds(self) -> float:
        """返回后台轮询间隔。"""

        with self._lock:
            return self._poll_seconds

    @property
    def system_idle_provider(self) -> str:
        """返回系统空闲探针名称。"""

        with self._lock:
            return self._system_idle_provider

    @property
    def system_idle_threshold_seconds(self) -> float:
        """返回系统空闲判定为活跃的阈值。"""

        with self._lock:
            return self._system_idle_threshold_seconds

    @property
    def running(self) -> bool:
        """返回后台循环是否仍在运行。"""

        task = self._loop_task
        return task is not None and not task.done()

    def _now(self, value: object | None = None) -> float:
        raw = self._clock() if value is None else value
        if isinstance(raw, bool):
            raise ValueError("activity clock must return a finite timestamp")
        try:
            result = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("activity clock must return a finite timestamp") from exc
        if not math.isfinite(result):
            raise ValueError("activity clock must return a finite timestamp")
        return result

    def _read_system_idle(self) -> bool:
        """读取可选系统空闲 provider，并将所有失败收敛为非活跃。"""

        with self._lock:
            provider_name = self._system_idle_provider
            probe = self._system_idle_probe
            threshold = self._system_idle_threshold_seconds
        if provider_name == SYSTEM_IDLE_PROVIDER_DISABLED:
            with self._lock:
                self._system_idle_status = "disabled"
                self._last_system_idle_active = False
            return False
        if provider_name not in {
            SYSTEM_IDLE_PROVIDER_X11,
            SYSTEM_IDLE_PROVIDER_WINDOWS,
        } or not callable(probe):
            with self._lock:
                self._system_idle_status = "unavailable"
                self._last_system_idle_active = False
            return False
        try:
            raw_idle = probe()
            if isinstance(raw_idle, bool):
                raise ValueError("system idle value is invalid")
            idle_seconds = float(raw_idle)
            if not math.isfinite(idle_seconds) or idle_seconds < 0.0:
                raise ValueError("system idle value is invalid")
        except Exception as exc:
            with self._lock:
                self._system_idle_status = "unavailable"
                self._last_system_idle_active = False
                self._last_system_idle_at = None
                self._last_error = f"{type(exc).__name__}: system idle probe failed"
            return False
        active = idle_seconds < threshold
        try:
            observed_at = self._now()
        except ValueError:
            observed_at = None
        with self._lock:
            self._system_idle_status = "ready"
            self._last_system_idle_active = active
            self._last_system_idle_at = observed_at
        return active

    def is_user_active(self, now: object | None = None) -> bool:
        """返回当前是否明确处于活跃窗口。

        未收到过交互、已关闭观察或时钟异常都返回 ``False``，供需要活跃
        用户的调度任务 fail-closed。
        """

        try:
            current = self._now(now)
        except ValueError:
            return False
        with self._lock:
            if not self._enabled:
                return False
            if (
                self._last_interaction_at is not None
                and current - self._last_interaction_at < self._idle_seconds
            ):
                # 显式宿主互动优先于系统探针；短暂的 X11 查询失败不会
                # 覆盖已经确认的活跃状态。
                return True
        return self._read_system_idle()

    def status(self) -> dict[str, object]:
        """返回可供控制台和诊断页使用的脱敏状态。"""

        try:
            current = self._now()
        except ValueError:
            current = None
        with self._lock:
            last = self._last_interaction_at
            if last is None or current is None:
                idle_for: float | None = None
            else:
                idle_for = max(0.0, current - last)
            state = self._state if self._enabled else "disabled"
            effective_active = bool(state == "active" or self._last_system_idle_active)
            return {
                "enabled": self._enabled,
                "running": self.running,
                "state": state,
                "active": effective_active,
                "idle_seconds": self._idle_seconds,
                "poll_seconds": self._poll_seconds,
                "idle_for_seconds": idle_for,
                "last_interaction_at": last,
                "last_source": self._last_source,
                "interaction_count": self._interaction_count,
                "idle_count": self._idle_count,
                "last_event_at": self._last_event_at,
                "pending_interactions": len(self._pending_interactions),
                "last_error": self._last_error,
                "system_idle_provider": self._system_idle_provider,
                "system_idle_threshold_seconds": self._system_idle_threshold_seconds,
                "system_idle_status": self._system_idle_status,
                "system_idle_available": self._system_idle_status == "ready",
                "last_system_idle_at": self._last_system_idle_at,
            }

    def _interaction_payload(self, *, at: float, source: str) -> dict[str, object]:
        return {
            "status": "available",
            "active": True,
            # 触发器条件使用统一的 ``user_active`` 字段；交互事件在
            # 记录时已经确认用户刚刚活跃，不能只依赖旧的 ``active`` 别名。
            "user_active": True,
            "state": "active",
            "source": source,
            "observed_at": at,
            "idle_for_seconds": 0.0,
        }

    def record_interaction(self, source: object = "unknown") -> Mapping[str, object]:
        """记录一次用户交互并异步触发 ``user_interaction``。

        该方法不接受任意 payload，避免键盘文本、窗口标题或其他敏感内容
        进入调度器事件；调用方只需提供有限长度的来源标签。
        """

        safe_source = _source(source)
        now = self._now()
        with self._lock:
            if not self._enabled:
                disabled = True
            else:
                disabled = False
                became_active = self._state != "active"
                self._last_interaction_at = now
                self._state = "active"
                self._last_source = safe_source
                self._interaction_count += 1
                self._last_event_at = now
                payload = self._interaction_payload(at=now, source=safe_source)
                payload["became_active"] = became_active
                self._pending_interactions.append(payload)
                loop = self._loop
        if disabled:
            return self.status()
        log_event(
            logger,
            "scheduler.activity.interaction",
            component="scheduler.activity",
            status="active",
            reason_code="interaction",
            fields={"source": safe_source, "interaction_count": self._interaction_count},
        )
        self._schedule_drain(loop)
        return self.status()

    def _schedule_drain(self, loop: asyncio.AbstractEventLoop | None) -> None:
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._ensure_drain_task)
        except RuntimeError:
            return

    def _ensure_drain_task(self) -> None:
        if self._drain_task is not None and not self._drain_task.done():
            return
        try:
            self._drain_task = asyncio.create_task(self._drain_pending())
        except RuntimeError:
            self._drain_task = None

    async def _drain_pending(self) -> None:
        while True:
            with self._lock:
                if not self._pending_interactions:
                    return
                payload = self._pending_interactions.popleft()
            try:
                result = self._triggers.emit("user_interaction", payload)
                if inspect.isawaitable(result):
                    await result
                if payload.get("became_active") is True:
                    active_result = self._triggers.emit("user_active", payload)
                    if inspect.isawaitable(active_result):
                        await active_result
            except asyncio.CancelledError:
                with self._lock:
                    self._pending_interactions.appendleft(payload)
                raise
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
                log_event(
                    logger,
                    "scheduler.activity.loop_failed",
                    component="scheduler.activity",
                    status="degraded",
                    level=logging.WARNING,
                    reason_code="poll_loop_failed",
                    fields={"error_type": type(exc).__name__},
                )
                log_event(
                    logger,
                    "scheduler.activity.emit_failed",
                    component="scheduler.activity",
                    status="failed",
                    level=logging.WARNING,
                    reason_code="trigger_emit_failed",
                    fields={"error_type": type(exc).__name__},
                )

    async def poll_once(self, *, now: object | None = None) -> Mapping[str, object]:
        """检查一次空闲边沿并在 active -> idle 时触发 ``idle``。"""

        current = self._now(now)
        # 后台轮询刷新系统 provider，使控制面能看到最新的可用/不可用状态；
        # 该调用只返回累计空闲时长，不携带输入内容。
        with self._lock:
            should_probe_system_idle = self._enabled and (
                self._system_idle_provider != SYSTEM_IDLE_PROVIDER_DISABLED
            )
        if should_probe_system_idle:
            self._read_system_idle()
        emit_payload: dict[str, object] | None = None
        with self._lock:
            if not self._enabled or self._last_interaction_at is None:
                elapsed = 0.0
            else:
                elapsed = max(0.0, current - self._last_interaction_at)
            if self._enabled and self._last_interaction_at is not None:
                if self._state == "active" and elapsed >= self._idle_seconds:
                    self._state = "idle"
                    self._idle_count += 1
                    source = self._last_source or "unknown"
                    emit_payload = {
                        "status": "available",
                        "active": False,
                        "state": "idle",
                        "source": source,
                        "observed_at": current,
                        "idle_for_seconds": elapsed,
                    }
                    self._last_event_at = current
        if emit_payload is None:
            return self.status()
        log_event(
            logger,
            "scheduler.activity.idle",
            component="scheduler.activity",
            status="idle",
            reason_code="idle_threshold",
            fields={"idle_count": self._idle_count},
        )
        try:
            result = self._triggers.emit("idle", emit_payload)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
            log_event(
                logger,
                "scheduler.activity.emit_failed",
                component="scheduler.activity",
                status="failed",
                level=logging.WARNING,
                reason_code="idle_emit_failed",
                fields={"error_type": type(exc).__name__},
            )
        return self.status()

    def set_system_idle_probe(self, probe: Callable[[], object] | None) -> Mapping[str, object]:
        """替换当前平台探针，供运行时热切换 provider 时重新接线。"""

        if probe is not None and not callable(probe):
            raise TypeError("system_idle_probe must be callable")
        with self._lock:
            self._system_idle_probe = probe
            if self._system_idle_provider == SYSTEM_IDLE_PROVIDER_DISABLED:
                self._system_idle_status = "disabled"
            else:
                self._system_idle_status = "unknown"
            self._last_system_idle_active = False
            self._last_system_idle_at = None
        return self.status()

    def reconfigure(
        self,
        *,
        enabled: bool | None = None,
        idle_seconds: float | None = None,
        poll_seconds: float | None = None,
        system_idle_provider: str | None = None,
        system_idle_threshold_seconds: float | None = None,
    ) -> Mapping[str, object]:
        """热更新观察参数；关闭后清除活跃状态并 fail-closed。"""

        new_idle = (
            self._idle_seconds
            if idle_seconds is None
            else _finite_number(
                idle_seconds,
                field_name="idle_seconds",
                minimum=MIN_IDLE_SECONDS,
                maximum=MAX_IDLE_SECONDS,
            )
        )
        new_poll = (
            self._poll_seconds
            if poll_seconds is None
            else _finite_number(
                poll_seconds,
                field_name="poll_seconds",
                minimum=MIN_POLL_SECONDS,
                maximum=MAX_POLL_SECONDS,
            )
        )
        new_system_provider = (
            self._system_idle_provider
            if system_idle_provider is None
            else _validate_system_idle_provider(system_idle_provider)
        )
        new_system_threshold = (
            self._system_idle_threshold_seconds
            if system_idle_threshold_seconds is None
            else _finite_number(
                system_idle_threshold_seconds,
                field_name="system_idle_threshold_seconds",
                minimum=MIN_SYSTEM_IDLE_THRESHOLD_SECONDS,
                maximum=MAX_SYSTEM_IDLE_THRESHOLD_SECONDS,
            )
        )
        with self._lock:
            was_enabled = self._enabled
            self._idle_seconds = new_idle
            self._poll_seconds = new_poll
            provider_changed = new_system_provider != self._system_idle_provider
            threshold_changed = new_system_threshold != self._system_idle_threshold_seconds
            self._system_idle_provider = new_system_provider
            self._system_idle_threshold_seconds = new_system_threshold
            if provider_changed or threshold_changed:
                self._system_idle_status = (
                    "disabled"
                    if new_system_provider == SYSTEM_IDLE_PROVIDER_DISABLED
                    else "unknown"
                )
                self._last_system_idle_active = False
                self._last_system_idle_at = None
            if enabled is not None:
                self._enabled = bool(enabled)
            if not self._enabled:
                self._state = "unknown"
                self._pending_interactions.clear()
                self._system_idle_status = "disabled"
                self._last_system_idle_active = False
                self._last_system_idle_at = None
            elif new_system_provider != SYSTEM_IDLE_PROVIDER_DISABLED and (
                provider_changed or threshold_changed or not was_enabled
            ):
                self._system_idle_status = "unknown"
        return self.status()

    async def start(self) -> Mapping[str, object]:
        """启动空闲轮询；重复启动保持幂等。"""

        if self.running:
            return self.status()
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._loop_task = asyncio.create_task(self._run_loop())
        self._ensure_drain_task()
        return self.status()

    async def _run_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass

    async def stop(self) -> Mapping[str, object]:
        """停止轮询并丢弃未投递的过期交互事件。"""

        if self._stop_event is not None:
            self._stop_event.set()
        task = self._loop_task
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        drain = self._drain_task
        if drain is not None and drain is not asyncio.current_task() and not drain.done():
            drain.cancel()
            try:
                await drain
            except asyncio.CancelledError:
                pass
        with self._lock:
            self._pending_interactions.clear()
            self._state = "unknown" if self._enabled else "disabled"
            self._last_system_idle_active = False
            self._last_system_idle_at = None
            self._system_idle_status = (
                "unknown"
                if self._enabled and self._system_idle_provider != SYSTEM_IDLE_PROVIDER_DISABLED
                else "disabled"
            )
        self._loop_task = None
        self._drain_task = None
        self._stop_event = None
        self._loop = None
        return self.status()


__all__ = [
    "MAX_IDLE_SECONDS",
    "MAX_PENDING_INTERACTIONS",
    "MAX_POLL_SECONDS",
    "MIN_IDLE_SECONDS",
    "MIN_POLL_SECONDS",
    "MAX_SYSTEM_IDLE_THRESHOLD_SECONDS",
    "MIN_SYSTEM_IDLE_THRESHOLD_SECONDS",
    "SYSTEM_IDLE_PROVIDER_DISABLED",
    "SYSTEM_IDLE_PROVIDER_X11",
    "SYSTEM_IDLE_PROVIDER_WINDOWS",
    "SYSTEM_IDLE_PROVIDERS",
    "UserActivityTracker",
]
