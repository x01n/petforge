"""内置模块的依赖生命周期、热替换与故障隔离。"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from logger.events import log_event

from .types import (
    ModuleContext,
    ModuleDescriptor,
    ModuleGenerationGate,
    ModuleState,
    ModuleStatus,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Record:
    descriptor: ModuleDescriptor
    state: ModuleState = ModuleState.REGISTERED
    generation: int = 0
    instance: object | None = None
    gate: ModuleGenerationGate | None = None
    failure_count: int = 0
    error_type: str = ""
    health: str = "unknown"

    def public(self) -> dict[str, object]:
        return ModuleStatus(
            module_id=self.descriptor.module_id,
            state=self.state,
            generation=self.generation,
            source=self.descriptor.source,
            dependencies=self.descriptor.dependencies,
            adopted=self.descriptor.adopted,
            failure_count=self.failure_count,
            error_type=self.error_type,
            health=self.health,
        ).public()


class ModuleManager:
    """管理内置服务，单个模块失败不会阻断其它模块。"""

    def __init__(
        self,
        *,
        configuration: Mapping[str, Any] | None = None,
        services: Mapping[str, object] | None = None,
    ) -> None:
        self._configuration = deepcopy(dict(configuration or {}))
        self._services = dict(services or {})
        self._records: dict[str, _Record] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._closed = False

    @property
    def running(self) -> bool:
        return self._running and not self._closed

    def set_services(self, services: Mapping[str, object] | None) -> None:
        self._services = dict(services or {})

    def update_configuration(self, configuration: Mapping[str, Any] | None) -> None:
        """更新模块上下文配置；不会自动触发模块副作用。"""

        self._configuration = deepcopy(dict(configuration or {}))

    def register(self, descriptor: ModuleDescriptor, *, replace: bool = False) -> dict[str, object]:
        if not isinstance(descriptor, ModuleDescriptor):
            raise TypeError("module descriptor is required")
        current = self._records.get(descriptor.module_id)
        if current is not None and not replace:
            raise ValueError(f"duplicate module id: {descriptor.module_id}")
        if current is not None and current.state in {
            ModuleState.LOADING,
            ModuleState.STARTING,
            ModuleState.RUNNING,
            ModuleState.STOPPING,
            ModuleState.UNLOADING,
            ModuleState.RELOADING,
        }:
            raise RuntimeError("running module must be reloaded asynchronously")
        generation = current.generation if current is not None else 0
        self._records[descriptor.module_id] = _Record(
            descriptor=descriptor,
            state=ModuleState.DISABLED if not descriptor.enabled else ModuleState.REGISTERED,
            generation=generation,
            instance=descriptor.instance,
            gate=current.gate if current is not None else None,
        )
        return self._records[descriptor.module_id].public()

    def adopt(
        self,
        module_id: str,
        instance: object,
        *,
        dependencies: tuple[str, ...] = (),
        source: str = "runtime",
        enabled: bool = True,
        configuration: Mapping[str, Any] | None = None,
        replace: bool = False,
    ) -> dict[str, object]:
        """登记已由运行时负责启动的对象，管理器不会重复调用其钩子。"""

        return self.register(
            ModuleDescriptor(
                module_id,
                instance=instance,
                source=source,
                dependencies=dependencies,
                enabled=enabled,
                adopted=True,
                configuration=configuration or {},
            ),
            replace=replace,
        )

    def replace_adopted(self, module_id: str, instance: object) -> dict[str, object]:
        """替换由组合根持有的对象，不调用对象生命周期钩子。"""

        name = str(module_id or "").strip()
        record = self._records.get(name)
        if record is None:
            return self.adopt(name, instance)
        if not record.descriptor.adopted:
            raise RuntimeError("module is not adopted")
        if record.gate is not None:
            record.gate.deactivate()
        record.generation += 1
        record.gate = ModuleGenerationGate(record.generation)
        record.instance = instance
        if record.state == ModuleState.FAILED:
            record.state = ModuleState.RUNNING if self._running else ModuleState.REGISTERED
            record.error_type = ""
        return record.public()

    def status(self, module_id: str | None = None) -> dict[str, object]:
        if module_id is not None:
            key = str(module_id or "").strip()
            record = self._records.get(key)
            return (
                record.public()
                if record is not None
                else {
                    "module_id": key,
                    "state": "unavailable",
                    "generation": 0,
                    "source": "runtime",
                    "dependencies": (),
                    "adopted": False,
                    "failure_count": 0,
                    "error_type": "unknown_module",
                    "health": "unknown",
                }
            )
        return {
            "running": self.running,
            "closed": self._closed,
            "modules": {name: record.public() for name, record in self._records.items()},
        }

    def _order(self) -> tuple[str, ...]:
        records = self._records
        result: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited:
                return
            if name in visiting:
                raise ValueError("module dependency cycle")
            visiting.add(name)
            record = records[name]
            for dependency in record.descriptor.dependencies:
                if dependency in records:
                    visit(dependency)
            visiting.remove(name)
            visited.add(name)
            result.append(name)

        for name in records:
            visit(name)
        return tuple(result)

    async def _invoke(self, target: object, method: str, *args: object) -> object:
        callback = getattr(target, method, None)
        if not callable(callback):
            return None
        result = callback(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    def _context(self, record: _Record) -> ModuleContext:
        gate = record.gate
        configuration = dict(self._configuration)
        configuration.update(record.descriptor.configuration)
        return ModuleContext(
            record.descriptor.module_id,
            record.generation,
            configuration,
            self._services,
            gate,
        )

    async def _load_one(self, name: str) -> None:
        record = self._records[name]
        if record.state in {ModuleState.DISABLED, ModuleState.RUNNING}:
            return
        if any(
            self._records[dependency].state == ModuleState.FAILED
            for dependency in record.descriptor.dependencies
            if dependency in self._records
        ):
            record.state = ModuleState.FAILED
            record.error_type = "dependency_failed"
            record.failure_count += 1
            return
        record.state = ModuleState.LOADING
        try:
            record.generation += 1
            record.gate = ModuleGenerationGate(record.generation)
            if record.instance is None and record.descriptor.factory is not None:
                value = record.descriptor.factory(self._context(record))
                record.instance = await value if inspect.isawaitable(value) else value
            if not record.descriptor.adopted:
                await self._invoke(record.instance, "load", self._context(record))
            record.state = ModuleState.LOADED
            record.error_type = ""
        except Exception as exc:
            record.state = ModuleState.FAILED
            record.failure_count += 1
            record.error_type = type(exc).__name__

    async def _start_one(self, name: str) -> None:
        record = self._records[name]
        if record.state in {ModuleState.DISABLED, ModuleState.FAILED, ModuleState.RUNNING}:
            return
        if any(
            self._records[dependency].state in {ModuleState.FAILED, ModuleState.DISABLED}
            for dependency in record.descriptor.dependencies
            if dependency in self._records
        ):
            record.state = ModuleState.FAILED
            record.error_type = "dependency_failed"
            record.failure_count += 1
            return
        record.state = ModuleState.STARTING
        try:
            if not record.descriptor.adopted:
                await self._invoke(record.instance, "start")
            record.state = ModuleState.RUNNING
            record.error_type = ""
        except Exception as exc:
            record.state = ModuleState.FAILED
            record.failure_count += 1
            record.error_type = type(exc).__name__

    async def start(self) -> dict[str, object]:
        async with self._lock:
            started_at = asyncio.get_running_loop().time()
            if self._closed:
                raise RuntimeError("module manager is closed")
            order = self._order()
            for name in order:
                await self._load_one(name)
                await self._start_one(name)
            self._running = True
            result = self.status()
            log_event(
                logger,
                "module.lifecycle.started",
                component="modules",
                status="completed",
                duration_ms=(asyncio.get_running_loop().time() - started_at) * 1000.0,
                fields={"module_count": len(order)},
            )
            return result

    async def stop(self) -> dict[str, object]:
        async with self._lock:
            for name in reversed(self._order()):
                record = self._records[name]
                if record.state != ModuleState.RUNNING:
                    continue
                record.state = ModuleState.STOPPING
                try:
                    if not record.descriptor.adopted:
                        await self._invoke(record.instance, "stop")
                    record.state = ModuleState.STOPPED
                except Exception as exc:
                    record.state = ModuleState.FAILED
                    record.failure_count += 1
                    record.error_type = type(exc).__name__
            self._running = False
            return self.status()

    async def unload(self, module_id: str | None = None) -> dict[str, object]:
        async with self._lock:
            names = (
                (str(module_id or "").strip(),)
                if module_id is not None
                else tuple(reversed(self._order()))
            )
            for name in names:
                record = self._records.get(name)
                if record is None:
                    continue
                if record.state == ModuleState.RUNNING:
                    await self._stop_record(record)
                record.state = ModuleState.UNLOADING
                try:
                    if not record.descriptor.adopted:
                        await self._invoke(record.instance, "unload")
                    if record.gate is not None:
                        record.gate.deactivate()
                    record.instance = None if not record.descriptor.adopted else record.instance
                    record.state = ModuleState.UNLOADED
                except Exception as exc:
                    record.state = ModuleState.FAILED
                    record.failure_count += 1
                    record.error_type = type(exc).__name__
            return self.status()

    async def _stop_record(self, record: _Record) -> None:
        record.state = ModuleState.STOPPING
        if not record.descriptor.adopted:
            await self._invoke(record.instance, "stop")
        record.state = ModuleState.STOPPED

    async def reload(
        self,
        module_id: str,
        *,
        replacement: object | Callable[[ModuleContext], object | Awaitable[object]] | None = None,
        callback: Callable[
            [object | None, object | None, ModuleContext], object | Awaitable[object]
        ]
        | None = None,
    ) -> dict[str, object]:
        """热替换单个模块；失败或取消时恢复旧实例。"""

        async with self._lock:
            name = str(module_id or "").strip()
            record = self._records.get(name)
            if record is None:
                raise KeyError(name)

            old = record.instance
            old_generation = record.generation
            old_gate = record.gate
            adopted = record.descriptor.adopted
            record.state = ModuleState.RELOADING
            if old_gate is not None:
                old_gate.deactivate()
            record.generation = old_generation + 1
            record.gate = ModuleGenerationGate(record.generation)
            context = self._context(record)
            replacement_instance: object | None = None

            async def cleanup_replacement() -> None:
                """回收已经部分加载的新实例，不能遮蔽主故障。"""

                if replacement_instance is None or replacement_instance is old or adopted:
                    return
                for method in ("stop", "unload"):
                    try:
                        await self._invoke(replacement_instance, method)
                    except Exception:
                        pass

            async def restore_old() -> bool:
                """为旧实例分配新代次并尝试恢复其生命周期。"""

                record.instance = old
                record.generation = old_generation + 1
                record.gate = ModuleGenerationGate(record.generation)
                if old is None:
                    return False
                if adopted:
                    return True
                try:
                    await self._invoke(old, "load", self._context(record))
                    await self._invoke(old, "start")
                except Exception:
                    return False
                return True

            try:
                if old is not None and not adopted:
                    await self._invoke(old, "stop")
                    await self._invoke(old, "unload")
                if replacement is None:
                    value = record.descriptor.factory(context) if record.descriptor.factory else old
                elif callable(replacement) and not hasattr(replacement, "start"):
                    value = replacement(context)
                else:
                    value = replacement
                replacement_instance = await value if inspect.isawaitable(value) else value
                record.instance = replacement_instance
                if not adopted:
                    await self._invoke(record.instance, "load", context)
                    await self._invoke(record.instance, "start")
                if callback is not None:
                    value = callback(old, record.instance, context)
                    if inspect.isawaitable(value):
                        await value
                record.state = ModuleState.RUNNING
                record.error_type = ""
                result = record.public()
                result["reload_status"] = "reloaded"
                return result
            except asyncio.CancelledError:
                await cleanup_replacement()
                restored = await restore_old()
                record.state = ModuleState.RUNNING if restored else ModuleState.FAILED
                record.failure_count += 1
                record.error_type = "" if restored else "reload_restore_failed"
                raise
            except Exception as exc:
                await cleanup_replacement()
                restored = await restore_old()
                record.state = ModuleState.RUNNING if restored else ModuleState.FAILED
                record.failure_count += 1
                record.error_type = "" if restored else type(exc).__name__
                result = record.public()
                result["reload_status"] = "restored" if restored else "failed"
                result["reload_error"] = type(exc).__name__
                return result

    async def reconfigure(self, configuration: Mapping[str, Any] | None) -> dict[str, object]:
        """提交配置快照，并为支持 reload 的模块执行异步回调。"""

        async with self._lock:
            self.update_configuration(configuration)
            records = tuple(self._records.values())
        reloaded: list[str] = []
        failed: list[str] = []
        for record in records:
            callback = getattr(record.instance, "reload_configuration", None)
            callback_with_context = False
            if not callable(callback):
                callback = getattr(record.instance, "reload", None)
                callback_with_context = callable(callback)
            if not callable(callback):
                continue
            try:
                argument = self._context(record) if callback_with_context else self._configuration
                result = callback(argument)
                if inspect.isawaitable(result):
                    result = await result
                if callback_with_context and result is not None and result is not record.instance:
                    replacement = result

                    def factory(
                        _context: ModuleContext,
                        replacement_instance: object = replacement,
                    ) -> object:
                        return replacement_instance

                    status = await self.reload(
                        record.descriptor.module_id,
                        replacement=factory,
                    )
                    if (
                        status.get("state") != ModuleState.RUNNING.value
                        or status.get("reload_status") != "reloaded"
                    ):
                        failed.append(record.descriptor.module_id)
                        continue
                reloaded.append(record.descriptor.module_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                record.failure_count += 1
                record.error_type = type(exc).__name__
                record.state = ModuleState.FAILED
                failed.append(record.descriptor.module_id)
        return {
            "status": "reloaded" if not failed else "degraded",
            "reloaded": tuple(reloaded),
            "failed": tuple(failed),
        }

    async def health(self) -> tuple[dict[str, object], ...]:
        async with self._lock:
            result: list[dict[str, object]] = []
            for name in self._order():
                record = self._records[name]
                if record.instance is None:
                    continue
                try:
                    value = await self._invoke(record.instance, "health")
                    if isinstance(value, Mapping):
                        token = value.get("status", value.get("health", "unknown"))
                    else:
                        token = value
                    record.health = str(token or "unknown")[:32]
                except Exception as exc:
                    record.health = "failed"
                    record.error_type = type(exc).__name__
                result.append(record.public())
            return tuple(result)

    async def close(self) -> dict[str, object]:
        if self._closed:
            return self.status()
        await self.stop()
        await self.unload()
        self._closed = True
        return self.status()


__all__ = ["ModuleManager"]
