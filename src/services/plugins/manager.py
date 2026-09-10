"""桌宠插件发现、生命周期编排与失败隔离。"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import importlib.util
import inspect
import logging
import sys
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any

from .types import (
    PluginContext,
    PluginDescriptor,
    PluginSettings,
    PluginState,
    PluginStatus,
    plugin_source_label,
)

logger = logging.getLogger(__name__)

_MAX_PLUGIN_BYTES = 4 * 1024 * 1024
_PLUGIN_FACTORY_EXPORT = "create_plugin"
_PLUGIN_ID_EXPORT = "PLUGIN_ID"
_PLUGIN_VERSION_EXPORT = "PLUGIN_VERSION"
_PLUGIN_DEPENDENCIES_EXPORT = "PLUGIN_DEPENDENCIES"


class _PluginGenerationGate:
    """使旧插件代次的服务访问和已注册回调失效。"""

    def __init__(self) -> None:
        self._active = True
        self._lock = Lock()

    def require_active(self) -> None:
        with self._lock:
            active = self._active
        if not active:
            raise RuntimeError("plugin generation is inactive")

    def deactivate(self) -> None:
        with self._lock:
            self._active = False

    def guard_callback(self, callback: object) -> object:
        if not callable(callback):
            return callback
        if inspect.iscoroutinefunction(callback):

            async def guarded_async(*args: object, **kwargs: object) -> object:
                try:
                    self.require_active()
                except RuntimeError:
                    return None
                return await callback(*args, **kwargs)

            return guarded_async

        def guarded_sync(*args: object, **kwargs: object) -> object:
            try:
                self.require_active()
            except RuntimeError:
                return None
            return callback(*args, **kwargs)

        return guarded_sync


class _GenerationBoundCallable:
    """在服务方法调用边界检查代次，并保护传入的插件回调。"""

    def __init__(self, target: object, gate: _PluginGenerationGate) -> None:
        self._target = target
        self._gate = gate

    def __call__(self, *args: object, **kwargs: object) -> object:
        self._gate.require_active()
        guarded_args = tuple(self._gate.guard_callback(item) for item in args)
        guarded_kwargs = {key: self._gate.guard_callback(value) for key, value in kwargs.items()}
        result = self._target(*guarded_args, **guarded_kwargs)
        if not inspect.isawaitable(result):
            self._gate.require_active()
            return result

        async def wait_for_result() -> object:
            value = await result
            self._gate.require_active()
            return value

        return wait_for_result()


class _GenerationBoundService:
    """只暴露仍处于当前插件代次的组合根服务。"""

    def __init__(self, target: object, gate: _PluginGenerationGate) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_gate", gate)

    def __getattr__(self, name: str) -> object:
        gate = object.__getattribute__(self, "_gate")
        gate.require_active()
        target = object.__getattribute__(self, "_target")
        value = getattr(target, name)
        if callable(value):
            return _GenerationBoundCallable(value, gate)
        return value


@dataclass(slots=True)
class _PluginRecord:
    descriptor: PluginDescriptor
    state: PluginState = PluginState.DISCOVERED
    generation: int = 0
    instance: object | None = None
    failure_count: int = 0
    error_type: str = ""
    health: str = "unknown"
    generation_gate: _PluginGenerationGate | None = None

    def public(self) -> PluginStatus:
        return PluginStatus(
            plugin_id=self.descriptor.plugin_id,
            state=self.state,
            generation=self.generation,
            source=self.descriptor.source,
            version=self.descriptor.version,
            dependencies=self.descriptor.dependencies,
            failure_count=self.failure_count,
            error_type=self.error_type,
            health=self.health,
        )


class PluginManager:
    """管理内置和外部插件，并把单个插件故障隔离在其状态内。

    外部 Python 文件必须导出 ``PLUGIN_ID`` 和 ``create_plugin(context)``。
    可选导出 ``PLUGIN_VERSION``、``PLUGIN_DEPENDENCIES``。实例生命周期钩子
    的参数固定为 ``load(context)``、``start()``、``stop()``、``unload()``、
    ``reload(context)`` 和 ``health()``；未提供的钩子视为成功的空操作。
    """

    def __init__(
        self,
        *,
        settings: PluginSettings | None = None,
        configuration: Mapping[str, Any] | None = None,
        services: Mapping[str, object] | None = None,
        base_directory: str | Path | None = None,
    ) -> None:
        self.settings = settings or PluginSettings()
        self._base_directory = (
            Path(base_directory).expanduser().resolve()
            if base_directory is not None
            else Path.cwd()
        )
        self._services: dict[str, object] = dict(services or {})
        self._configuration: dict[str, Any] = deepcopy(dict(configuration or {}))
        self._records: dict[str, _PluginRecord] = {}
        self._source_ids: dict[str, set[str]] = {}
        self._source_modules: dict[str, str] = {}
        self._discovery_errors: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._watch_task: asyncio.Task[None] | None = None
        self._running = False
        self._closed = False
        self._generation = 0
        self._last_refresh_ms = 0.0
        self._resume_on_enable = False
        self._started = False

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        services: Mapping[str, object] | None = None,
        base_directory: str | Path | None = None,
    ) -> PluginManager:
        """按精确的 ``plugins`` 配置区构造管理器。"""

        return cls(
            settings=PluginSettings.from_mapping(value),
            configuration=value,
            services=services,
            base_directory=base_directory,
        )

    @property
    def running(self) -> bool:
        return self._running and not self._closed

    @property
    def generation(self) -> int:
        return self._generation

    def set_services(self, services: Mapping[str, object] | None) -> None:
        """更新后续插件上下文使用的服务映射。"""

        if services is not None and not isinstance(services, Mapping):
            raise TypeError("plugin services must be a mapping")
        self._services = dict(services or {})

    def register(self, descriptor: PluginDescriptor, *, replace: bool = False) -> PluginStatus:
        """登记一个显式插件描述；重复身份默认拒绝覆盖。"""

        if not isinstance(descriptor, PluginDescriptor):
            raise TypeError("plugin descriptor is required")
        current = self._records.get(descriptor.plugin_id)
        if current is not None and not replace:
            raise ValueError(f"duplicate plugin id: {descriptor.plugin_id}")
        if current is not None and current.state in {
            PluginState.RUNNING,
            PluginState.STARTING,
            PluginState.STOPPING,
            PluginState.LOADING,
            PluginState.UNLOADING,
        }:
            raise RuntimeError("running plugin must be reloaded asynchronously")
        if current is None:
            record = _PluginRecord(descriptor)
        else:
            record = _PluginRecord(
                descriptor,
                state=PluginState.DISCOVERED,
                generation=current.generation,
                failure_count=current.failure_count,
            )
        self._records[descriptor.plugin_id] = record
        return record.public()

    def unregister(self, plugin_id: str) -> bool:
        """移除尚未运行的显式描述；运行插件必须先调用异步卸载。"""

        key = str(plugin_id or "").strip()
        record = self._records.get(key)
        if record is None:
            return False
        if record.instance is not None or record.state in {
            PluginState.RUNNING,
            PluginState.STARTING,
            PluginState.STOPPING,
            PluginState.LOADING,
            PluginState.UNLOADING,
        }:
            raise RuntimeError("loaded plugin must be unloaded asynchronously")
        self._records.pop(key, None)
        return True

    def descriptors(self) -> tuple[PluginDescriptor, ...]:
        return tuple(record.descriptor for record in self._records.values())

    def status(self, plugin_id: str | None = None) -> Mapping[str, object]:
        """返回固定字段的脱敏状态，不执行插件代码。"""

        if plugin_id is not None:
            record = self._records.get(str(plugin_id or "").strip())
            if record is None:
                return {
                    "status": "unavailable",
                    "plugin_id": str(plugin_id or "").strip(),
                    "reason": "unknown_plugin",
                }
            return {"status": "available", **record.public().public()}
        records = tuple(record.public().public() for record in self._records.values())
        failed_discovery = tuple(
            {"source": plugin_source_label(source), "error_type": error_type[:64]}
            for source, error_type in sorted(self._discovery_errors.items())
        )
        return {
            "status": "closed" if self._closed else "running" if self._running else "stopped",
            "enabled": self.settings.enabled,
            "running": self._running,
            "generation": self._generation,
            "plugin_count": len(records),
            "loaded_count": sum(
                item["state"] in {"loaded", "running", "stopped"} for item in records
            ),
            "running_count": sum(item["state"] == "running" for item in records),
            "failed_count": sum(item["state"] == "failed" for item in records),
            "discovery_error_count": len(failed_discovery),
            "discovery_errors": failed_discovery,
            "last_refresh_ms": round(self._last_refresh_ms, 3),
            "plugins": records,
        }

    def discover(self) -> tuple[PluginDescriptor, ...]:
        """扫描配置目录与显式启用的 Python 入口点，单文件失败不影响其它文件。"""

        if self._closed or not self.settings.enabled:
            return ()
        discovered: list[PluginDescriptor] = []
        source_ids: dict[str, set[str]] = {}
        for configured_directory in self.settings.directories:
            directory = Path(configured_directory).expanduser()
            if not directory.is_absolute():
                directory = self._base_directory / directory
            directory = directory.resolve()
            source_key = str(directory)
            source_ids[source_key] = set()
            try:
                paths = tuple(sorted(directory.glob("*.py"))) if directory.is_dir() else ()
            except OSError as exc:
                self._discovery_errors[source_key] = type(exc).__name__
                continue
            for path in paths:
                descriptor = self._discover_file(path)
                if descriptor is None:
                    # 文件暂时不可解析时保留已运行代次；下一次轮询可在
                    # 用户完成写入后重试，避免半写入导致服务瞬时消失。
                    for record in self._records.values():
                        if record.descriptor.source == str(path):
                            source_ids[source_key].add(record.descriptor.plugin_id)
                    continue
                source_ids[source_key].add(descriptor.plugin_id)
                discovered.append(descriptor)

        if self.settings.entry_points_enabled:
            for descriptor in self._discover_entry_points():
                source_ids.setdefault("entry_points", set()).add(descriptor.plugin_id)
                discovered.append(descriptor)

        for descriptor in discovered:
            try:
                existing = self._records.get(descriptor.plugin_id)
                if existing is None:
                    self.register(descriptor)
                elif existing.descriptor.source != descriptor.source:
                    self._discovery_errors[descriptor.source] = "duplicate_plugin_id"
                elif existing.descriptor.fingerprint != descriptor.fingerprint:
                    # 热重载由 refresh() 执行；这里只更新未加载描述，避免同步
                    # 发现阶段中断正在使用的实例。
                    if existing.instance is None:
                        self.register(descriptor, replace=True)
            except (RuntimeError, TypeError, ValueError) as exc:
                self._discovery_errors[descriptor.source] = type(exc).__name__
        self._source_ids = source_ids
        return tuple(discovered)

    def _discover_file(self, path: Path) -> PluginDescriptor | None:
        source = str(path)
        try:
            size = path.stat().st_size
            if size > _MAX_PLUGIN_BYTES:
                raise ValueError("plugin file exceeds size limit")
            payload = path.read_bytes()
            if len(payload) > _MAX_PLUGIN_BYTES:
                raise ValueError("plugin file exceeds size limit")
            fingerprint = hashlib.sha256(payload).hexdigest()
            for record in self._records.values():
                if (
                    record.descriptor.source == source
                    and record.descriptor.fingerprint == fingerprint
                    and source in self._source_modules
                ):
                    self._discovery_errors.pop(source, None)
                    return record.descriptor
            source_key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
            # 内容指纹必须参与模块身份。同一路径在文件大小、mtime 甚至同一
            # 字节码缓存时间窗内更新时，也必须加载新源码而不是复用旧 pyc。
            module_name = f"meapet_plugin_{source_key}_{fingerprint[:16]}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("plugin module spec is unavailable")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                # SourceFileLoader 可能在同一秒、同尺寸改写时命中旧 pyc。
                # 这里直接编译已经计算过指纹的字节，保证发现与执行同一快照。
                source_code = compile(payload, source, "exec", dont_inherit=True)
                exec(source_code, module.__dict__)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
            plugin_id = str(getattr(module, _PLUGIN_ID_EXPORT, "") or "").strip()
            factory = getattr(module, _PLUGIN_FACTORY_EXPORT, None)
            if not plugin_id or not callable(factory):
                sys.modules.pop(module_name, None)
                raise ValueError("plugin module must export PLUGIN_ID and create_plugin")
            dependencies = getattr(module, _PLUGIN_DEPENDENCIES_EXPORT, ())
            descriptor = PluginDescriptor(
                plugin_id=plugin_id,
                factory=factory,
                source=source,
                version=str(getattr(module, _PLUGIN_VERSION_EXPORT, "") or ""),
                dependencies=tuple(dependencies),
                fingerprint=fingerprint,
            )
            previous_module = self._source_modules.get(source)
            if previous_module and previous_module != module_name:
                sys.modules.pop(previous_module, None)
            self._source_modules[source] = module_name
            self._discovery_errors.pop(source, None)
            return descriptor
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            self._discovery_errors[source] = type(exc).__name__
            logger.warning("插件发现失败：%s (%s)", path.name, type(exc).__name__)
            return None

    def _discover_entry_points(self) -> tuple[PluginDescriptor, ...]:
        discovered: list[PluginDescriptor] = []
        try:
            entries = importlib.metadata.entry_points(group=self.settings.entry_point_group)
        except (TypeError, ValueError, importlib.metadata.PackageNotFoundError):
            entries = ()
        for entry in entries:
            source = f"entry_point:{entry.name}"
            try:
                loaded = entry.load()
                descriptor = loaded() if isinstance(loaded, type) else loaded
                if isinstance(descriptor, PluginDescriptor):
                    discovered.append(descriptor)
                    continue
                plugin_id = str(getattr(loaded, _PLUGIN_ID_EXPORT, "") or "").strip()
                if not plugin_id or not callable(loaded):
                    raise ValueError(
                        "entry point must expose PluginDescriptor or factory with PLUGIN_ID"
                    )
                dependencies = getattr(loaded, _PLUGIN_DEPENDENCIES_EXPORT, ())
                discovered.append(
                    PluginDescriptor(
                        plugin_id=plugin_id,
                        factory=loaded,
                        source=source,
                        version=str(getattr(loaded, _PLUGIN_VERSION_EXPORT, "") or ""),
                        dependencies=tuple(dependencies),
                    )
                )
                self._discovery_errors.pop(source, None)
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                    raise
                self._discovery_errors[source] = type(exc).__name__
        return tuple(discovered)

    async def load(self, plugin_id: str) -> PluginStatus:
        """按依赖顺序加载单个插件并隔离失败。"""

        key = str(plugin_id or "").strip()
        async with self._lock:
            return await self._load_locked(key, stack=())

    async def _load_locked(self, plugin_id: str, *, stack: tuple[str, ...]) -> PluginStatus:
        record = self._records.get(plugin_id)
        if record is None:
            return PluginStatus(plugin_id, PluginState.FAILED, error_type="unknown_plugin")
        if not self.settings.enabled:
            record.state = PluginState.DISABLED
            return record.public()
        if record.state == PluginState.RUNNING:
            return record.public()
        if record.instance is not None and record.state in {
            PluginState.LOADED,
            PluginState.STOPPED,
        }:
            if self._running and record.state is PluginState.STOPPED:
                return await self._start_locked(record)
            return record.public()
        if plugin_id in stack:
            return self._fail(record, "dependency_cycle")
        for dependency in record.descriptor.dependencies:
            dependency_status = await self._load_locked(dependency, stack=(*stack, plugin_id))
            if dependency_status.state not in {
                PluginState.LOADED,
                PluginState.RUNNING,
                PluginState.STOPPED,
            }:
                return self._fail(record, "dependency_failed")

        record.state = PluginState.LOADING
        record.error_type = ""
        record.health = "unknown"
        self._generation += 1
        record.generation = self._generation
        record.generation_gate = _PluginGenerationGate()
        context = self._context(record)
        try:
            instance = record.descriptor.factory(context)
            if inspect.isawaitable(instance):
                instance = await instance
            record.instance = instance
            result = await self._call(instance, "load", context)
            del result
            record.state = PluginState.LOADED
            if self._running:
                return await self._start_locked(record)
            return record.public()
        except asyncio.CancelledError:
            await self._discard_failed_instance(record)
            if record.generation_gate is not None:
                record.generation_gate.deactivate()
            record.state = PluginState.FAILED
            record.failure_count += 1
            record.error_type = "cancelled"
            raise
        except BaseException as exc:
            await self._discard_failed_instance(record)
            if record.generation_gate is not None:
                record.generation_gate.deactivate()
            return self._fail(record, type(exc).__name__)

    async def load_all(self) -> tuple[PluginStatus, ...]:
        """加载全部已发现插件；一个插件失败不会阻断其它插件。"""

        async with self._lock:
            statuses = []
            for plugin_id in tuple(self._records):
                try:
                    statuses.append(await self._load_locked(plugin_id, stack=()))
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    record = self._records.get(plugin_id)
                    if record is not None:
                        statuses.append(self._fail(record, type(exc).__name__))
            return tuple(statuses)

    async def _start_locked(self, record: _PluginRecord) -> PluginStatus:
        if record.instance is None:
            return self._fail(record, "missing_instance")
        if record.state is PluginState.RUNNING:
            return record.public()
        record.state = PluginState.STARTING
        try:
            await self._call(record.instance, "start")
            record.state = PluginState.RUNNING
            record.health = "unknown"
            return record.public()
        except asyncio.CancelledError:
            await self._discard_failed_instance(record)
            if record.generation_gate is not None:
                record.generation_gate.deactivate()
            record.generation_gate = None
            record.state = PluginState.FAILED
            record.failure_count += 1
            record.error_type = "cancelled"
            raise
        except BaseException as exc:
            await self._discard_failed_instance(record)
            if record.generation_gate is not None:
                record.generation_gate.deactivate()
            record.generation_gate = None
            return self._fail(record, type(exc).__name__)

    async def start(self) -> Mapping[str, object]:
        """发现并加载插件，随后启动文件变更轮询。"""

        if self._closed:
            return dict(self.status())
        self._started = True
        self._running = bool(self.settings.enabled)
        self.discover()
        await self.load_all()
        if self.settings.reload_enabled and self.settings.directories and self._watch_task is None:
            self._watch_task = asyncio.create_task(self._watch_loop(), name="meapet-plugin-watcher")
        return dict(self.status())

    async def stop(self) -> Mapping[str, object]:
        """停止插件实例但保留描述，供运行时重新启动。"""

        self._started = False
        self._resume_on_enable = False
        self._running = False
        await self._stop_watcher()
        async with self._lock:
            for record in tuple(self._records.values())[::-1]:
                await self._stop_locked(record)
        return dict(self.status())

    async def unload(self, plugin_id: str) -> PluginStatus:
        """停止并卸载一个插件；卸载失败只影响该插件。"""

        key = str(plugin_id or "").strip()
        async with self._lock:
            record = self._records.get(key)
            if record is None:
                return PluginStatus(key, PluginState.FAILED, error_type="unknown_plugin")
            return await self._unload_locked(record)

    async def unload_all(self) -> tuple[PluginStatus, ...]:
        async with self._lock:
            statuses = []
            for record in tuple(self._records.values())[::-1]:
                statuses.append(await self._unload_locked(record))
            return tuple(statuses)

    async def _stop_locked(self, record: _PluginRecord) -> PluginStatus:
        if record.instance is None or record.state not in {
            PluginState.RUNNING,
            PluginState.STARTING,
        }:
            return record.public()
        record.state = PluginState.STOPPING
        try:
            await self._call(record.instance, "stop")
            record.state = PluginState.STOPPED
            return record.public()
        except asyncio.CancelledError:
            record.state = PluginState.FAILED
            record.failure_count += 1
            record.error_type = "cancelled"
            raise
        except BaseException as exc:
            return self._fail(record, type(exc).__name__)

    async def _unload_locked(self, record: _PluginRecord) -> PluginStatus:
        if record.instance is None:
            if record.generation_gate is not None:
                record.generation_gate.deactivate()
                record.generation_gate = None
            if record.state is not PluginState.FAILED:
                record.state = PluginState.UNLOADED
            return record.public()
        if record.state in {PluginState.RUNNING, PluginState.STARTING}:
            await self._stop_locked(record)
            if record.state is PluginState.FAILED:
                # stop() 失败后仍尝试 unload，避免插件持有进程/线程资源。
                pass
        record.state = PluginState.UNLOADING
        try:
            await self._call(record.instance, "unload")
            # 生命周期清理完成后、下一代实例创建前关闭组合根服务闸门。
            # 插件可在 stop/unload 中正常注销资源，旧实例保留的迟到回调
            # 则无法在新代次启动后继续修改共享服务。
            if record.generation_gate is not None:
                record.generation_gate.deactivate()
            record.instance = None
            record.generation_gate = None
            record.state = PluginState.UNLOADED
            record.health = "unknown"
            record.error_type = ""
            return record.public()
        except asyncio.CancelledError:
            if record.generation_gate is not None:
                record.generation_gate.deactivate()
            record.instance = None
            record.generation_gate = None
            record.state = PluginState.FAILED
            record.failure_count += 1
            record.error_type = "cancelled"
            raise
        except BaseException as exc:
            if record.generation_gate is not None:
                record.generation_gate.deactivate()
            # 丢弃失败实例引用，避免下一次 load_all 创建新实例时遗留旧资源。
            record.instance = None
            record.generation_gate = None
            return self._fail(record, type(exc).__name__)

    async def reload(
        self,
        plugin_id: str,
        descriptor: PluginDescriptor | None = None,
    ) -> PluginStatus:
        """以新代次替换单个插件，旧实例卸载完成后才创建新实例。"""

        key = str(plugin_id or "").strip()
        async with self._lock:
            current = self._records.get(key)
            if current is None:
                if descriptor is None:
                    return PluginStatus(key, PluginState.FAILED, error_type="unknown_plugin")
                self.register(descriptor)
                current = self._records[descriptor.plugin_id]
            if descriptor is not None and descriptor.plugin_id != key:
                return self._fail(current, "plugin_id_mismatch")
            was_running = self._running
            # 失败且实例已清理的记录可以从当前描述重新尝试；不能让
            # _unload_locked 保留 FAILED 状态阻断下一次热重载。
            if current.instance is None and current.state is PluginState.FAILED:
                current.state = PluginState.DISCOVERED
                current.error_type = ""
                current.health = "unknown"
                if descriptor is not None:
                    current.descriptor = descriptor
            else:
                unloaded = await self._unload_locked(current)
                if unloaded.state is PluginState.FAILED:
                    return unloaded
                if descriptor is not None:
                    current.descriptor = descriptor
                current.state = PluginState.DISCOVERED
            if was_running:
                return await self._load_locked(key, stack=())
            return current.public()

    async def refresh(self) -> Mapping[str, object]:
        """应用新增、删除和内容变更的文件插件。"""

        if self._closed or not self.settings.enabled:
            return dict(self.status())
        started = monotonic()
        previous_source_ids = {source: set(ids) for source, ids in self._source_ids.items()}
        discovered = self.discover()
        current_source_ids = {source: set(ids) for source, ids in self._source_ids.items()}
        removed_ids = {
            plugin_id
            for source, old_ids in previous_source_ids.items()
            for plugin_id in old_ids - current_source_ids.get(source, set())
        }
        for plugin_id in removed_ids:
            record = self._records.get(plugin_id)
            if record is not None:
                await self.unload(plugin_id)
                if record.instance is None:
                    self.unregister(plugin_id)
                    if record.descriptor.source in self._source_modules:
                        sys.modules.pop(self._source_modules.pop(record.descriptor.source), None)

        for descriptor in discovered:
            record = self._records.get(descriptor.plugin_id)
            if record is None:
                self.register(descriptor)
                if self._running:
                    await self.load(descriptor.plugin_id)
                continue
            if (
                record.descriptor.source == descriptor.source
                and record.descriptor.fingerprint != descriptor.fingerprint
            ):
                result = await self.reload(descriptor.plugin_id, descriptor)
                if result.state is PluginState.FAILED:
                    # 新文件失败时保留失败状态；旧实例已完成卸载，下一轮可重试。
                    self._discovery_errors[descriptor.source] = result.error_type or "reload_failed"
                continue
            if (
                self._running
                and record.instance is None
                and record.descriptor.source == descriptor.source
                and record.state
                in {
                    PluginState.DISCOVERED,
                    PluginState.UNLOADED,
                    PluginState.DISABLED,
                }
            ):
                await self.load(descriptor.plugin_id)
        self._last_refresh_ms = (monotonic() - started) * 1000.0
        return dict(self.status())

    async def reconfigure(
        self,
        value: Mapping[str, Any] | None,
        *,
        base_directory: str | Path | None = None,
    ) -> Mapping[str, object]:
        """热替换插件配置并立即应用目录变化。"""

        settings = PluginSettings.from_mapping(value)
        old_settings = self.settings
        old_configuration = deepcopy(self._configuration)
        if base_directory is not None:
            self._base_directory = Path(base_directory).expanduser().resolve()
        self.settings = settings
        self._configuration = deepcopy(dict(value or {}))
        if not settings.enabled:
            self._resume_on_enable = self._started
            self._running = False
            await self._stop_watcher()
            await self.unload_all()
            async with self._lock:
                for record in self._records.values():
                    if record.instance is None:
                        record.state = PluginState.DISABLED
            return dict(self.status())
        if self._resume_on_enable:
            self._resume_on_enable = False
            return await self.start()
        if not self._started:
            return dict(self.status())
        self._running = True
        if (
            old_settings.directories != settings.directories
            or old_settings.entry_points_enabled != settings.entry_points_enabled
        ):
            await self.refresh()
        if old_configuration != self._configuration:
            await self._reload_configured_plugins()
        if settings.reload_enabled and settings.directories and self._watch_task is None:
            self._watch_task = asyncio.create_task(self._watch_loop(), name="meapet-plugin-watcher")
        elif (
            not settings.reload_enabled or not settings.directories
        ) and self._watch_task is not None:
            await self._stop_watcher()
        await self.load_all()
        return dict(self.status())

    async def _reload_configured_plugins(self) -> None:
        """把新的插件配置快照交给已加载实例的 reload 钩子。"""

        async with self._lock:
            for record in tuple(self._records.values()):
                if record.instance is None or record.state not in {
                    PluginState.LOADED,
                    PluginState.RUNNING,
                    PluginState.STOPPED,
                }:
                    continue
                reload_hook = getattr(record.instance, "reload", None)
                if not callable(reload_hook):
                    unloaded = await self._unload_locked(record)
                    if unloaded.state is PluginState.FAILED:
                        continue
                    record.state = PluginState.DISCOVERED
                    if self._running:
                        await self._load_locked(record.descriptor.plugin_id, stack=())
                    continue
                previous_generation = record.generation
                previous_gate = record.generation_gate
                previous_state = record.state
                self._generation += 1
                record.generation = self._generation
                record.generation_gate = _PluginGenerationGate()
                context = self._context(record)
                try:
                    await self._call(record.instance, "reload", context)
                except asyncio.CancelledError:
                    if record.generation_gate is not None:
                        record.generation_gate.deactivate()
                    record.generation = previous_generation
                    record.generation_gate = previous_gate
                    record.state = previous_state
                    raise
                except BaseException as exc:
                    if record.generation_gate is not None:
                        record.generation_gate.deactivate()
                    record.generation = previous_generation
                    record.generation_gate = previous_gate
                    # 保留仍可工作的旧实例，避免后续 load_all 覆盖实例并遗漏
                    # 其异步资源；失败信息留在状态供诊断。
                    record.state = previous_state
                    record.failure_count += 1
                    record.error_type = type(exc).__name__[:64]
                    record.health = "failed"
                else:
                    if previous_gate is not None:
                        previous_gate.deactivate()
                    record.health = "unknown"
                    record.error_type = ""

    async def health(self) -> tuple[Mapping[str, object], ...]:
        """逐插件读取可选健康钩子；失败只更新该插件状态。"""

        async with self._lock:
            results: list[Mapping[str, object]] = []
            for record in self._records.values():
                if record.instance is None or record.state not in {
                    PluginState.RUNNING,
                    PluginState.LOADED,
                    PluginState.STOPPED,
                }:
                    results.append(record.public().public())
                    continue
                try:
                    value = await self._call(record.instance, "health")
                    record.health = self._health_token(value)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    record.failure_count += 1
                    record.error_type = type(exc).__name__
                    record.health = "failed"
                results.append(record.public().public())
            return tuple(results)

    async def close(self) -> Mapping[str, object]:
        """关闭观察器并卸载全部插件；关闭过程继续处理其它插件。"""

        if self._closed:
            return dict(self.status())
        self._closed = True
        self._started = False
        self._running = False
        await self._stop_watcher()
        async with self._lock:
            for record in tuple(self._records.values())[::-1]:
                try:
                    await self._unload_locked(record)
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    self._fail(record, type(exc).__name__)
        return dict(self.status())

    async def _watch_loop(self) -> None:
        try:
            while self._running and not self._closed and self.settings.reload_enabled:
                await asyncio.sleep(self.settings.reload_interval_seconds)
                try:
                    await self.refresh()
                except asyncio.CancelledError:
                    raise
                except BaseException as exc:
                    logger.warning("插件热重载失败：%s", type(exc).__name__)
        finally:
            if asyncio.current_task() is self._watch_task:
                self._watch_task = None

    async def _stop_watcher(self) -> None:
        task = self._watch_task
        if task is None:
            return
        self._watch_task = None
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def _context(self, record: _PluginRecord) -> PluginContext:
        gate = record.generation_gate
        if gate is None:
            gate = _PluginGenerationGate()
            record.generation_gate = gate
        services = {
            name: _GenerationBoundService(value, gate) if value is not None else value
            for name, value in self._services.items()
        }
        return PluginContext(
            record.descriptor.plugin_id,
            record.generation,
            configuration=deepcopy(self._configuration),
            services=services,
        )

    @staticmethod
    async def _call(instance: object, name: str, *args: object) -> object:
        method = getattr(instance, name, None)
        if not callable(method):
            return None
        value = method(*args)
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _health_token(value: object) -> str:
        if isinstance(value, Mapping):
            value = value.get("status", "unknown")
        token = str(value or "unknown").strip().lower()
        if token in {"ready", "ok", "healthy", "degraded", "failed", "unknown"}:
            return token
        return "unknown"

    @staticmethod
    async def _discard_failed_instance(record: _PluginRecord) -> None:
        """异步释放加载失败的实例，避免留下未执行的卸载协程。"""

        instance = record.instance
        record.instance = None
        if instance is None:
            return
        close = getattr(instance, "unload", None)
        if not callable(close):
            return
        try:
            value = close()
            if inspect.isawaitable(value):
                await value
        except BaseException:
            # 原始加载异常优先保留；清理失败只记录日志，不让实例残留。
            logger.debug("插件失败实例卸载异常", exc_info=True)

    @staticmethod
    def _fail(record: _PluginRecord, error_type: str) -> PluginStatus:
        record.state = PluginState.FAILED
        record.failure_count += 1
        record.error_type = str(error_type or "PluginError")[:64]
        record.health = "failed"
        return record.public()


__all__ = ["PluginManager"]
