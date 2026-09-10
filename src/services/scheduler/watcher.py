from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

from gui.platforms.protocol import DesktopPlatform

from .triggers import TriggerService

MIN_POLL_SECONDS = 0.05
MAX_POLL_SECONDS = 60.0
MIN_POLL_TIMEOUT_SECONDS = 0.1
MAX_POLL_TIMEOUT_SECONDS = 60.0
SYNC_READ_WAIT_SECONDS = 0.01
MAX_STALE_READS = 2


@dataclass(slots=True)
class _ForegroundRead:
    """保存一次不可取消的同步前台窗口读取。"""

    call: Callable[[], object]
    generation: int
    done: Event = field(default_factory=Event)
    result: object = None
    error: BaseException | None = None
    timed_out: bool = False
    stale: bool = False

    def run(self) -> None:
        """在线程中执行同步调用，并始终发布完成信号。"""

        try:
            self.result = self.call()
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()


class DesktopWindowWatcher:
    """可取消轮询前台窗口并发出 ``window_changed`` 事件。"""

    def __init__(
        self,
        *,
        platform: DesktopPlatform,
        triggers: TriggerService,
        poll_seconds: float = 0.5,
        emit_initial: bool = False,
        poll_timeout_seconds: float = 5.0,
        activity_provider: Callable[[], object] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._platform = platform
        self._triggers = triggers
        self._poll_seconds = max(MIN_POLL_SECONDS, min(float(poll_seconds), MAX_POLL_SECONDS))
        self._poll_timeout_seconds = max(
            MIN_POLL_TIMEOUT_SECONDS,
            min(float(poll_timeout_seconds), MAX_POLL_TIMEOUT_SECONDS),
        )
        self._emit_initial = bool(emit_initial)
        self._activity_provider = activity_provider
        self._clock = clock or time.monotonic
        self._loop_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._poll_lock = asyncio.Lock()
        self._sync_read_lock = Lock()
        self._sync_read: _ForegroundRead | None = None
        self._stale_reads: list[_ForegroundRead] = []
        self._generation = 0
        self._running_generation: int | None = None
        self._last_identity: tuple[object, ...] | None = None
        self._last_identity_at: float | None = None
        self._last_snapshot: dict[str, object] | None = None
        self._status: dict[str, object] = {"status": "stopped", "error": ""}

    @property
    def poll_seconds(self) -> float:
        return self._poll_seconds

    @property
    def running(self) -> bool:
        return self._loop_task is not None and not self._loop_task.done()

    def reconfigure(
        self,
        *,
        poll_seconds: float | None = None,
        emit_initial: bool | None = None,
        poll_timeout_seconds: float | None = None,
    ) -> None:
        """在不重建后台任务的情况下刷新前台窗口观察参数。"""

        if poll_seconds is not None:
            try:
                value = float(poll_seconds)
            except (TypeError, ValueError, OverflowError):
                value = self._poll_seconds
            self._poll_seconds = max(MIN_POLL_SECONDS, min(value, MAX_POLL_SECONDS))
        if poll_timeout_seconds is not None:
            try:
                value = float(poll_timeout_seconds)
            except (TypeError, ValueError, OverflowError):
                value = self._poll_timeout_seconds
            self._poll_timeout_seconds = max(
                MIN_POLL_TIMEOUT_SECONDS,
                min(value, MAX_POLL_TIMEOUT_SECONDS),
            )
        if emit_initial is not None:
            self._emit_initial = bool(emit_initial)

    def set_platform(self, platform: DesktopPlatform) -> None:
        """替换轮询平台；用于宿主在 GUI 线程建立后绑定调度边界。"""

        self._platform = platform
        self._last_identity = None
        self._last_identity_at = None
        if self.running:
            self._invalidate_current_read()
            with self._sync_read_lock:
                self._generation += 1
                # 平台切换必须让后续读取绑定到新代际；否则仍会复用旧平台的读取函数。
                self._running_generation = self._generation
                self._prune_stale_reads_locked()
                stale_limit_reached = len(self._stale_reads) >= MAX_STALE_READS
            if stale_limit_reached:
                self._status = {
                    "status": "degraded",
                    "error": "too many previous foreground_window calls remain in flight",
                }

    def status(self) -> Mapping[str, object]:
        """返回观察器状态及最近一次平台摘要。"""

        result = dict(self._status)
        if self._last_snapshot is not None:
            result["snapshot"] = dict(self._last_snapshot)
        with self._sync_read_lock:
            self._prune_stale_reads_locked()
            read = self._sync_read
            current_in_flight = read is not None and not read.done.is_set()
            current_timed_out = bool(read is not None and read.timed_out and current_in_flight)
            stale_in_flight = len(self._stale_reads)
            in_flight = current_in_flight or stale_in_flight > 0
        if current_timed_out:
            read_status = "timed_out"
        elif current_in_flight:
            read_status = "in_flight"
        elif stale_in_flight:
            read_status = "stale_in_flight"
        else:
            read_status = "idle"
        result["foreground_window_read"] = {
            "status": read_status,
            "in_flight": in_flight,
            "recoverable": in_flight,
            "stale_in_flight": stale_in_flight,
        }
        return result

    async def start(self) -> Mapping[str, object]:
        """启动后台轮询；平台接口不可用时返回状态而不创建任务。"""

        if self.running:
            return self.status()
        self._invalidate_current_read()
        stale_limit_reached = False
        with self._sync_read_lock:
            self._prune_stale_reads_locked()
            if len(self._stale_reads) >= MAX_STALE_READS:
                self._status = {
                    "status": "degraded",
                    "error": "a previous foreground_window call remains in flight",
                }
                stale_limit_reached = True
        if stale_limit_reached:
            return self.status()
        with self._sync_read_lock:
            self._generation += 1
            self._running_generation = self._generation
        if not callable(getattr(self._platform, "foreground_window", None)):
            self._status = {"status": "unavailable", "error": "foreground_window is unavailable"}
            return self.status()
        self._stop_event = asyncio.Event()
        self._status = {"status": "running", "error": ""}
        self._loop_task = asyncio.create_task(self._run_loop())
        return self.status()

    async def stop(self) -> Mapping[str, object]:
        """请求取消轮询并等待后台任务结束。"""

        self._invalidate_current_read()
        with self._sync_read_lock:
            self._running_generation = None
            self._generation += 1
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
        if self._status.get("status") != "unavailable":
            self._status = {
                "status": "stopped",
                "error": str(self._status.get("error", "")),
            }
        return self.status()

    async def _run_loop(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # 触发器或第三方平台异常不能杀死长期观察循环；下一轮仍可恢复。
                self._status = {
                    "status": "degraded",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._poll_seconds)
            except TimeoutError:
                pass
        if self._status.get("status") == "running":
            self._status = {"status": "stopped", "error": ""}

    async def poll_once(self) -> Mapping[str, object]:
        """读取一次窗口摘要并在身份变化时发出事件。"""

        async with self._poll_lock:
            return await self._poll_once()

    async def _poll_once(self) -> Mapping[str, object]:
        """串行完成一次读取，避免并发轮询重复消费同一结果。"""

        read = self._get_sync_read()
        if read is None:
            return self.status()
        if not await self._wait_for_sync_read(read):
            if self._read_is_stale(read):
                return self.status()
            self._status = {
                "status": "degraded",
                "error": "foreground_window timed out; synchronous call remains in flight",
            }
            self._last_identity_at = None
            return self.status()
        if self._read_is_stale(read):
            self._clear_sync_read(read)
            return self.status()
        try:
            if read.error is not None:
                raise read.error
            result = read.result
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=self._poll_timeout_seconds)
        except Exception as exc:  # 平台不可用不应终止后台观察任务
            self._status = {"status": "unavailable", "error": str(exc)}
            self._last_identity_at = None
            return self.status()
        finally:
            self._clear_sync_read(read)
        if self._read_is_stale(read):
            return self.status()
        if not isinstance(result, Mapping):
            self._status = {
                "status": "unavailable",
                "error": "foreground_window returned non-mapping",
            }
            self._last_identity_at = None
            return self.status()
        snapshot = dict(result)
        self._last_snapshot = snapshot
        if str(snapshot.get("status", "")).strip().lower() != "available":
            self._status = {"status": "unavailable", "error": str(snapshot.get("reason", ""))}
            self._last_identity_at = None
            return self.status()
        identity = self._identity(snapshot)
        changed = self._last_identity is not None and identity != self._last_identity
        initial = self._last_identity is None
        try:
            observed_at = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            observed_at = time.monotonic()
        if initial or changed or self._last_identity_at is None:
            self._last_identity_at = observed_at
        self._last_identity = identity
        snapshot["active_for_seconds"] = max(0.0, observed_at - self._last_identity_at)
        provider = self._activity_provider
        if callable(provider):
            try:
                snapshot["user_active"] = bool(provider())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                snapshot["user_active"] = False
        else:
            snapshot["user_active"] = False
        self._last_snapshot = snapshot
        self._status = {"status": "running", "error": ""}
        if changed or (initial and self._emit_initial):
            await self._triggers.emit("window_changed", snapshot)
        await self._triggers.emit("window_active", snapshot)
        return self.status()

    def _get_sync_read(self) -> _ForegroundRead | None:
        """复用当前读取；不存在时才创建唯一的后台线程。"""

        with self._sync_read_lock:
            self._prune_stale_reads_locked()
            read = self._sync_read
            if read is not None:
                return read
            if len(self._stale_reads) >= MAX_STALE_READS:
                self._status = {
                    "status": "degraded",
                    "error": "too many previous foreground_window calls remain in flight",
                }
                return None
            generation = (
                self._running_generation
                if self._running_generation is not None
                else self._generation
            )
            read = _ForegroundRead(self._platform.foreground_window, generation)
            self._sync_read = read
        thread = Thread(
            target=read.run,
            name="meapet-window-watcher",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException as exc:
            read.error = exc
            read.done.set()
        return read

    async def _wait_for_sync_read(self, read: _ForegroundRead) -> bool:
        """等待同步读取；超时只标记状态，不再提交新的线程任务。"""

        if self._read_is_stale(read):
            return False
        if read.done.is_set():
            return True
        if read.timed_out:
            return False
        deadline = asyncio.get_running_loop().time() + self._poll_timeout_seconds
        while not read.done.is_set():
            if self._read_is_stale(read):
                return False
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                with self._sync_read_lock:
                    if not read.done.is_set() and not read.stale:
                        read.timed_out = True
                        return False
                break
            await asyncio.sleep(min(SYNC_READ_WAIT_SECONDS, remaining))
        return True

    def _read_is_stale(self, read: _ForegroundRead) -> bool:
        """判断读取是否属于已停止或已切换代际。"""

        with self._sync_read_lock:
            return read.stale or read.generation != self._generation

    def _invalidate_current_read(self) -> None:
        """停止当前代读取并把不可取消的线程保留为可恢复状态。"""

        with self._sync_read_lock:
            read = self._sync_read
            self._sync_read = None
            if read is None:
                return
            read.stale = True
            if not read.done.is_set():
                self._stale_reads.append(read)

    def _prune_stale_reads_locked(self) -> None:
        """丢弃已完成的旧代读取结果，不让其进入新代状态。"""

        self._stale_reads[:] = [read for read in self._stale_reads if not read.done.is_set()]

    def _clear_sync_read(self, read: _ForegroundRead) -> None:
        """仅清理当前已完成读取，避免误删随后创建的读取。"""

        with self._sync_read_lock:
            if self._sync_read is read and read.done.is_set():
                self._sync_read = None

    @staticmethod
    def _identity(snapshot: Mapping[str, object]) -> tuple[object, ...]:
        keys = ("backend", "window_id", "app_id", "pid", "process_name", "executable")
        return tuple(snapshot.get(key) for key in keys)


WindowWatcher = DesktopWindowWatcher

__all__ = [
    "DesktopWindowWatcher",
    "WindowWatcher",
    "MIN_POLL_SECONDS",
    "MAX_POLL_SECONDS",
    "MIN_POLL_TIMEOUT_SECONDS",
    "MAX_POLL_TIMEOUT_SECONDS",
    "SYNC_READ_WAIT_SECONDS",
    "MAX_STALE_READS",
]
