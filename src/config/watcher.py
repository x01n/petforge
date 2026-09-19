from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from logger.events import log_event

from .loader import (
    MAX_CONFIGURATION_BYTES,
    ConfigurationError,
    LoadedConfiguration,
    load_configuration,
)

logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

MIN_INTERVAL_SECONDS = 0.05
MAX_INTERVAL_SECONDS = 60.0
MIN_DEBOUNCE_SECONDS = 0.0
MAX_DEBOUNCE_SECONDS = 10.0
DEFAULT_STABLE_CHECKS = 2

ReloadCallback = Callable[[LoadedConfiguration], Awaitable[object] | object]

_SUCCESS_STATUSES = frozenset(
    {
        "ok",
        "accepted",
        "applied",
        "completed",
        "reloaded",
        "updated",
        "saved",
        "partial",
        # 某些不可安全热替换的区段可以由宿主接受并提示重启；文件仍应
        # 记为已观察，避免同一份文件在每轮轮询重复提交。
        "restart_required",
    }
)
_FAILURE_STATUSES = frozenset(
    {
        "failed",
        "error",
        "denied",
        "unavailable",
        "rejected",
        "cancelled",
        "timeout",
    }
)

_DETAIL_SECTIONS = frozenset(
    {
        "llm",
        "tts",
        "asr",
        "memory",
        "behavior",
        "proactive",
        "watcher",
        "config",
        "scheduler",
        "ui",
        "logging",
        "mcp",
        "plugins",
        "tools",
        "rendering",
        "storage",
        "app",
        "web",
    }
)
_DETAIL_STATUSES = frozenset(
    {
        "reloaded",
        "available",
        "unchanged",
        "pending",
        "requested",
        "busy",
        "failed",
        "unavailable",
        "restart_required",
        "degraded",
        "ready",
    }
)


@dataclass(frozen=True, slots=True)
class ConfigurationReload:
    """一次观察或重载结果的脱敏摘要。"""

    status: str
    generation: int
    path: Path
    changed: bool = False
    error: str = ""
    configuration: LoadedConfiguration | None = None

    def public(self) -> dict[str, object]:
        """返回可直接投影到日志/UI 的摘要，不携带配置值或密钥。"""

        return {
            "status": self.status,
            "generation": self.generation,
            "path": str(self.path),
            "changed": self.changed,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    """文件内容指纹；哈希用于过滤仅修改 mtime 的重复保存。"""

    digest: str
    size: int


class ConfigurationWatcher:
    """轮询 YAML 文件并串行提交校验后的配置。

    ``on_reload`` 在观察器所属事件循环中调用。回调返回 ``False``、带失败
    ``status`` 的映射或抛出异常时，当前配置保持不变；只有成功回执才会提升
    ``generation`` 并更新 ``current_configuration``。默认需要连续两次看到
    相同内容，能够避开非原子编辑器写入中的半份 YAML。
    """

    def __init__(
        self,
        path: str | Path,
        on_reload: ReloadCallback | None = None,
        *,
        callback: ReloadCallback | None = None,
        defaults: Mapping[str, Any] | None = None,
        environment: Mapping[str, str] | None = None,
        initial_configuration: LoadedConfiguration | None = None,
        interval_seconds: float = 0.5,
        debounce_seconds: float = 0.1,
        stable_checks: int = DEFAULT_STABLE_CHECKS,
        max_bytes: int = MAX_CONFIGURATION_BYTES,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        raw_path = str(path or "").strip()
        if not raw_path:
            raise ValueError("configuration path is required")
        resolved = Path(raw_path).expanduser().resolve()
        selected_callback = on_reload if on_reload is not None else callback
        if not callable(selected_callback):
            raise TypeError("configuration reload callback is required")
        try:
            interval = float(interval_seconds)
            debounce = float(debounce_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("configuration watcher timing is invalid") from exc
        if not MIN_INTERVAL_SECONDS <= interval <= MAX_INTERVAL_SECONDS:
            raise ValueError("configuration watcher interval is outside the allowed range")
        if not MIN_DEBOUNCE_SECONDS <= debounce <= MAX_DEBOUNCE_SECONDS:
            raise ValueError("configuration watcher debounce is outside the allowed range")
        if isinstance(stable_checks, bool):
            raise ValueError("configuration watcher stable_checks must be an integer")
        try:
            checks = int(stable_checks)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("configuration watcher stable_checks must be an integer") from exc
        if checks < 1 or checks > 8 or checks != stable_checks:
            raise ValueError("configuration watcher stable_checks is outside the allowed range")
        if isinstance(max_bytes, bool):
            raise ValueError("configuration watcher max_bytes must be an integer")
        try:
            byte_limit = int(max_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("configuration watcher max_bytes must be an integer") from exc
        if byte_limit < 1024 or byte_limit > 64 * 1024 * 1024 or byte_limit != max_bytes:
            raise ValueError("configuration watcher max_bytes is outside the allowed range")

        self.path = resolved
        self._callback = selected_callback
        self._defaults = (
            copy.deepcopy(dict(defaults)) if isinstance(defaults, Mapping) else defaults
        )
        self._environment = dict(environment) if environment is not None else None
        self._interval_seconds = interval
        self._debounce_seconds = debounce
        self._stable_checks = checks
        self._max_bytes = byte_limit
        self._clock = clock
        self._current_configuration = initial_configuration
        self._generation = 0
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state_lock = threading.RLock()
        # public poll_once() 也可能被测试或宿主并发调用；锁同时保护指纹和
        # callback，确保同一文件内容最多产生一次回调。
        self._poll_lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
        self._observed: _FileSnapshot | None = None
        self._pending: _FileSnapshot | None = None
        self._pending_seen = 0
        self._pending_since = 0.0
        self._deferred_digest = ""
        self._deferred_until = 0.0
        self._last_result = ConfigurationReload("stopped", 0, self.path)
        self._last_error = ""
        # 运行时回调的结构化结果不能塞进 ConfigurationReload.error：那会
        # 把“模块局部失败但整体已应用”误报成配置拒绝。这里仅保存经过白名单
        # 投影的公开字段，供 Qt/Web 控制面显示最近一次应用细节。
        self._last_details: dict[str, object] = {}

    @property
    def interval_seconds(self) -> float:
        return self._interval_seconds

    @property
    def running(self) -> bool:
        task = self._task
        return task is not None and not task.done()

    @property
    def current_configuration(self) -> LoadedConfiguration | None:
        return self._current_configuration

    @property
    def generation(self) -> int:
        return self._generation

    def reconfigure(
        self,
        *,
        interval_seconds: float | None = None,
        debounce_seconds: float | None = None,
        stable_checks: int | None = None,
    ) -> None:
        """更新轮询参数；调用方应先通过运行时配置校验。"""

        if interval_seconds is not None:
            interval = float(interval_seconds)
            if not MIN_INTERVAL_SECONDS <= interval <= MAX_INTERVAL_SECONDS:
                raise ValueError("configuration watcher interval is outside the allowed range")
            self._interval_seconds = interval
        if debounce_seconds is not None:
            debounce = float(debounce_seconds)
            if not MIN_DEBOUNCE_SECONDS <= debounce <= MAX_DEBOUNCE_SECONDS:
                raise ValueError("configuration watcher debounce is outside the allowed range")
            self._debounce_seconds = debounce
        if stable_checks is not None:
            if isinstance(stable_checks, bool):
                raise ValueError("configuration watcher stable_checks must be an integer")
            checks = int(stable_checks)
            if checks < 1 or checks > 8 or checks != stable_checks:
                raise ValueError("configuration watcher stable_checks is outside the allowed range")
            self._stable_checks = checks

    def acknowledge(self, configuration: LoadedConfiguration) -> bool:
        """确认宿主已应用文件内容，抑制 GUI 保存后的重复回调。

        该方法故意保持同步，供 ``Future.add_done_callback`` 调用；跨线程时
        通过观察器所属事件循环排队，绝不直接触碰 asyncio 锁或任务。
        """

        # if not isinstance(configuration, LoadedConfiguration):
        #     return False
        if configuration.path.resolve() != self.path:
            return False
        loop = self._loop
        if loop is None or loop.is_closed():
            self._acknowledge_local(configuration)
            return True
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is loop:
            self._acknowledge_local(configuration)
            return True
        try:
            loop.call_soon_threadsafe(self._acknowledge_local, configuration)
        except RuntimeError:
            self._acknowledge_local(configuration)
        return True

    def _acknowledge_local(self, configuration: LoadedConfiguration) -> None:
        """确认已应用的内容，同时保留并发外部写入的重新加载机会。

        GUI 保存完成与编辑器/站点再次写入可能交错。只有两次连续指纹一致且
        当前磁盘解析结果与宿主已应用配置相同，才允许推进观察基线；否则仍
        记录宿主当前配置，但让下一次轮询处理磁盘上的更新内容。
        """

        first_snapshot = self._read_snapshot()
        matching_snapshot: _FileSnapshot | None = None
        if first_snapshot is not None:
            try:
                disk_configuration = self._load()
            except Exception:
                disk_configuration = None
            second_snapshot = self._read_snapshot()
            if (
                disk_configuration is not None
                and second_snapshot is not None
                and first_snapshot.digest == second_snapshot.digest
                and disk_configuration.values == configuration.values
            ):
                matching_snapshot = second_snapshot

        with self._state_lock:
            self._current_configuration = configuration
            self._last_error = ""
            if matching_snapshot is not None:
                self._observed = matching_snapshot
                self._pending = None
                self._pending_seen = 0
                self._deferred_digest = ""
                self._deferred_until = 0.0
            self._last_result = ConfigurationReload(
                "running" if self.running else "stopped",
                self._generation,
                self.path,
                changed=True,
                configuration=configuration,
            )

    def status(self) -> Mapping[str, object]:
        """返回不含配置正文的生命周期和最近一次状态。"""

        result = self._last_result.public()
        result["running"] = self.running
        result["pending"] = self._pending is not None
        result["generation"] = self._generation
        result.update(self._last_details)
        return result

    async def start(self) -> Mapping[str, object]:
        """建立初始指纹并启动轮询；初始配置不会重复提交回调。"""

        if self.running:
            return self.status()
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._pending = None
        self._pending_seen = 0
        self._deferred_digest = ""
        self._deferred_until = 0.0
        self._last_error = ""
        self._last_details = {}
        snapshot = self._read_snapshot()
        initial_status = "running"
        if snapshot is not None:
            self._observed = snapshot
            if self._current_configuration is None:
                try:
                    self._current_configuration = self._load()
                except Exception as exc:
                    self._last_error = _safe_error(exc)
                    initial_status = "rejected"
        else:
            self._observed = None
            initial_status = "unavailable"
            if not self._last_error:
                self._last_error = "configuration file is unavailable"
        self._last_result = ConfigurationReload(
            initial_status,
            self._generation,
            self.path,
            error=self._last_error,
            configuration=self._current_configuration,
        )
        self._task = asyncio.create_task(self._run(), name="meapet-config-watcher")
        return self.status()

    async def stop(self) -> Mapping[str, object]:
        """停止轮询；未完成的 reload 回调会随事件循环取消。"""

        event = self._stop_event
        if event is not None:
            event.set()
        task = self._task
        current = asyncio.current_task()
        if task is not None and task is current:
            # reload 回调可能因为新配置把 watcher 自身关闭；不能在当前
            # 协程仍执行时清空 ``_stop_event``，否则 ``_run`` 收尾阶段会
            # 解引用 None 并留下“任务已完成但句柄仍运行”的假状态。
            self._pending = None
            self._pending_seen = 0
            self._last_result = ConfigurationReload(
                "stopped",
                self._generation,
                self.path,
                error=self._last_error,
                configuration=self._current_configuration,
            )
            self._last_details = {}
            return self.status()
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._task = None
        self._stop_event = None
        self._loop = None
        self._pending = None
        self._pending_seen = 0
        self._last_result = ConfigurationReload(
            "stopped",
            self._generation,
            self.path,
            error=self._last_error,
            configuration=self._current_configuration,
        )
        self._last_details = {}
        return self.status()

    async def poll_once(self) -> Mapping[str, object]:
        """主动执行一次检测；可用于 Qt/测试宿主的定时器。"""

        async with self._poll_lock:
            return await self._poll_once_locked()

    async def _run(self) -> None:
        assert self._stop_event is not None
        try:
            while not self._stop_event.is_set():
                try:
                    await self.poll_once()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    # 观察器必须长期存活；意外异常只进入诊断并等待下一轮恢复。
                    self._last_error = _safe_error(exc)
                    self._last_result = ConfigurationReload(
                        "degraded", self._generation, self.path, error=self._last_error
                    )
                    self._last_details = {}
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval_seconds)
                except TimeoutError:
                    pass
        finally:
            if self._task is asyncio.current_task():
                self._task = None
                self._stop_event = None
                self._loop = None

    async def _poll_once_locked(self) -> Mapping[str, object]:
        snapshot = self._read_snapshot()
        if snapshot is None:
            # 文件暂时不存在或超过大小上限时不清除旧配置；同一缺失状态只
            # 记录一次，文件恢复后下一次指纹变化会重新进入稳定检查。
            self._pending = None
            self._pending_seen = 0
            error = self._last_error or "configuration file is unavailable"
            self._last_result = ConfigurationReload(
                "unavailable", self._generation, self.path, error=error
            )
            self._last_details = {}
            return self.status()
        if self._observed is not None and snapshot.digest == self._observed.digest:
            if self._last_result.status == "unavailable":
                self._last_result = ConfigurationReload(
                    "running",
                    self._generation,
                    self.path,
                    error=self._last_error,
                    configuration=self._current_configuration,
                )
            return self.status()

        if snapshot.digest == self._deferred_digest:
            if self._clock() < self._deferred_until:
                self._last_result = ConfigurationReload(
                    "deferred", self._generation, self.path, changed=True
                )
                return self.status()
            self._deferred_digest = ""
            self._deferred_until = 0.0
            return await self._reload_snapshot(snapshot)
        self._deferred_digest = ""
        self._deferred_until = 0.0

        if self._pending is None or self._pending.digest != snapshot.digest:
            self._pending = snapshot
            self._pending_seen = 1
            self._pending_since = self._clock()
            if self._stable_checks == 1 and self._debounce_seconds == 0:
                # 单次稳定检查模式用于已知采用原子替换的写入器；首个完整
                # 内容即可提交，不额外制造一个 pending 轮询。
                self._pending = None
                self._pending_seen = 0
                return await self._reload_snapshot(snapshot)
            self._last_result = ConfigurationReload(
                "pending", self._generation, self.path, changed=True
            )
            self._last_details = {}
            return self.status()

        self._pending_seen += 1
        elapsed = max(0.0, self._clock() - self._pending_since)
        if self._pending_seen < self._stable_checks or elapsed < self._debounce_seconds:
            self._last_result = ConfigurationReload(
                "pending", self._generation, self.path, changed=True
            )
            self._last_details = {}
            return self.status()

        # 成功/失败的最终指纹由 ``_reload_snapshot`` 写入；若宿主明确要求
        # 延后，观察器会保留旧基线并在冷却后重试同一份内容。
        self._pending = None
        self._pending_seen = 0
        return await self._reload_snapshot(snapshot)

    async def _reload_snapshot(self, snapshot: _FileSnapshot) -> Mapping[str, object]:
        async with self._reload_lock:
            started_at = time.monotonic()
            try:
                configuration = self._load()
            except Exception as exc:
                self._last_error = _safe_error(exc)
                self._observed = snapshot
                self._last_result = ConfigurationReload(
                    "rejected",
                    self._generation,
                    self.path,
                    changed=True,
                    error=self._last_error,
                    configuration=self._current_configuration,
                )
                self._last_details = {}
                log_event(
                    logger,
                    "config.reload.failed",
                    component="config_watcher",
                    status="failed",
                    duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
                    reason_code="load_failed",
                )
                return self.status()

            # 文件在解析期间可能被编辑器再次替换；解析结果必须与应用前
            # 的最新指纹一致，否则本次配置不能提交给运行时。
            current_snapshot = self._read_snapshot()
            if current_snapshot is None:
                self._pending = None
                self._pending_seen = 0
                self._last_result = ConfigurationReload(
                    "unavailable",
                    self._generation,
                    self.path,
                    changed=True,
                    error=self._last_error or "configuration file is unavailable",
                )
                self._last_details = {}
                return self.status()
            if current_snapshot.digest != snapshot.digest:
                self._pending = current_snapshot
                self._pending_seen = 1
                self._pending_since = self._clock()
                self._last_result = ConfigurationReload(
                    "pending",
                    self._generation,
                    self.path,
                    changed=True,
                )
                self._last_details = {}
                return self.status()

            try:
                callback_result = self._callback(configuration)
                if inspect.isawaitable(callback_result):
                    callback_result = await callback_result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = _safe_error(exc)
                self._observed = snapshot
                self._last_result = ConfigurationReload(
                    "rejected",
                    self._generation,
                    self.path,
                    changed=True,
                    error=self._last_error,
                    configuration=self._current_configuration,
                )
                self._last_details = {}
                log_event(
                    logger,
                    "config.reload.failed",
                    component="config_watcher",
                    status="failed",
                    duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
                    reason_code="callback_failed",
                )
                return self.status()

            callback_status = _callback_status(callback_result)
            callback_details = _callback_details(callback_result)
            # ApplicationRuntime 保持整体状态为 ``reloaded``，但模块管理器
            # 可以在回执中报告局部失败；把这种结果投影为 partial，避免 Web
            # 控制面显示“运行中”而隐藏实际失败。
            if callback_status == "reloaded" and callback_details.get("modules_failed"):
                callback_status = "partial"
            if _callback_requests_retry(callback_result):
                self._deferred_digest = snapshot.digest
                self._deferred_until = self._clock() + self._interval_seconds
                self._last_result = ConfigurationReload(
                    "deferred",
                    self._generation,
                    self.path,
                    changed=True,
                    error="reload deferred by runtime",
                    configuration=self._current_configuration,
                )
                self._last_details = callback_details
                log_event(
                    logger,
                    "config.reload.deferred",
                    component="config_watcher",
                    status="deferred",
                    duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
                )
                return self.status()
            if callback_status in _FAILURE_STATUSES:
                self._observed = snapshot
                self._last_error = callback_status
                self._last_result = ConfigurationReload(
                    "rejected",
                    self._generation,
                    self.path,
                    changed=True,
                    error=callback_status,
                    configuration=self._current_configuration,
                )
                self._last_details = callback_details
                log_event(
                    logger,
                    "config.reload.failed",
                    component="config_watcher",
                    status="failed",
                    duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
                    reason_code=callback_status,
                )
                return self.status()

            self._generation += 1
            self._observed = snapshot
            self._current_configuration = configuration
            self._last_error = ""
            self._last_details = callback_details
            # 对外保持稳定的观察器状态；宿主的 ``ok/applied`` 等细分回执
            # 不应让 UI/调度器需要枚举一整套别名。只有明确的部分应用或
            # 需重启状态保留其语义。
            result_status = (
                callback_status
                if callback_status in {"partial", "restart_required"}
                else "reloaded"
            )
            self._last_result = ConfigurationReload(
                result_status,
                self._generation,
                self.path,
                changed=True,
                configuration=configuration,
            )
            log_event(
                logger,
                "config.reload.completed",
                component="config_watcher",
                status=result_status,
                duration_ms=max(0.0, (time.monotonic() - started_at) * 1000.0),
                fields={"generation": self._generation},
            )
            return self.status()

    def _load(self) -> LoadedConfiguration:
        return load_configuration(
            self.path,
            defaults=self._defaults,
            environment=self._environment,
        )

    def _read_snapshot(self) -> _FileSnapshot | None:
        try:
            if self.path.stat().st_size > self._max_bytes:
                self._last_error = "configuration file exceeds size limit"
                return None
            payload = self.path.read_bytes()
        except (OSError, UnicodeError) as exc:
            self._last_error = _safe_error(exc)
            return None
        if len(payload) > self._max_bytes:
            self._last_error = "configuration file exceeds size limit"
            return None
        self._last_error = ""
        return _FileSnapshot(hashlib.sha256(payload).hexdigest(), len(payload))


def _callback_status(value: object) -> str:
    """把宿主回执规整为短状态；未知状态按成功兼容。"""

    if value is False:
        return "rejected"
    if value is True or value is None:
        return "reloaded"
    if isinstance(value, Mapping):
        raw = value.get("status", value.get("state", ""))
        status = str(getattr(raw, "value", raw) or "").strip().lower()
        if status:
            if status in _SUCCESS_STATUSES or status in _FAILURE_STATUSES:
                return status
            return "reloaded"
    return "reloaded"


def _callback_details(value: object) -> dict[str, object]:
    """投影运行时回执的有限字段，避免把配置内容带入控制面。"""

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, object] = {}
    for key in ("changed_sections", "applied_sections", "restart_sections"):
        raw = value.get(key)
        if not isinstance(raw, (list, tuple, set, frozenset)):
            continue
        sections = tuple(
            item for item in (str(item).strip() for item in raw) if item in _DETAIL_SECTIONS
        )
        if sections:
            result[key] = sections
            result[f"{key}_count"] = len(sections)
    for key in ("modules_failed", "modules_reloaded"):
        raw = value.get(key)
        if not isinstance(raw, (list, tuple, set, frozenset)):
            continue
        # 模块标识是诊断所需的公开数据；仅保留短、非空字符串，避免回显
        # 任意对象的 repr 或异常正文。
        modules = tuple(
            item[:128] for item in (str(item).strip() for item in raw) if item and len(item) <= 128
        )
        if modules:
            result[key] = modules
            result[f"{key}_count"] = len(modules)
    for key in (
        "rendering_status",
        "plugins_status",
        "modules_status",
        "tools_status",
        "mcp_status",
    ):
        raw = str(value.get(key, "") or "").strip().lower()
        if raw in _DETAIL_STATUSES:
            result[key] = raw
    if value.get("cleanup") == "restart_required":
        result["cleanup"] = "restart_required"
    if value.get("retry") is True:
        result["retry"] = True
    return result


def _callback_requests_retry(value: object) -> bool:
    """识别宿主要求稍后重试的显式回执。"""

    if not isinstance(value, Mapping):
        return False
    retry = value.get("retry", value.get("deferred", False))
    if isinstance(retry, str):
        return retry.strip().lower() in {"true", "yes", "on", "1"}
    return retry is True


def _safe_error(exc: BaseException) -> str:
    """错误摘要不回显异常正文中的密钥、请求体或命令参数。"""

    if isinstance(exc, ConfigurationError):
        return str(exc)[:240]
    return type(exc).__name__


ConfigFileWatcher = ConfigurationWatcher
YamlConfigurationWatcher = ConfigurationWatcher

__all__ = [
    "ConfigFileWatcher",
    "ConfigurationReload",
    "ConfigurationWatcher",
    "YamlConfigurationWatcher",
]
