"""内置模块生命周期使用的稳定数据契约。"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock
from typing import Any, Protocol, TypeAlias


class ModuleState(StrEnum):
    """模块状态。"""

    REGISTERED = "registered"
    LOADING = "loading"
    LOADED = "loaded"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    RELOADING = "reloading"
    FAILED = "failed"
    DISABLED = "disabled"


def source_label(value: object) -> str:
    """把来源映射为不泄露路径的稳定标签。"""

    source = str(value or "runtime").strip() or "runtime"
    if source == "runtime":
        return source
    return "source:sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


class ModuleGenerationGate:
    """阻止旧代次回调在热替换后写入新实例。"""

    def __init__(self, generation: int) -> None:
        self.generation = int(generation)
        self._active = True
        self._lock = Lock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def require_active(self) -> None:
        if not self.active:
            raise RuntimeError("module generation is inactive")

    def deactivate(self) -> None:
        with self._lock:
            self._active = False

    def guard_callback(self, callback: object) -> object:
        """包装回调，旧代次调用会被丢弃。"""

        if not callable(callback):
            return callback
        if inspect.iscoroutinefunction(callback):

            async def guarded_async(*args: object, **kwargs: object) -> object:
                if not self.active:
                    return None
                return await callback(*args, **kwargs)

            return guarded_async

        def guarded_sync(*args: object, **kwargs: object) -> object:
            if not self.active:
                return None
            return callback(*args, **kwargs)

        return guarded_sync


@dataclass(frozen=True, slots=True)
class ModuleContext:
    """模块获得的最小运行时上下文。"""

    module_id: str
    generation: int
    configuration: Mapping[str, Any] = field(default_factory=dict)
    services: Mapping[str, object] = field(default_factory=dict)
    gate: ModuleGenerationGate | None = None

    def __post_init__(self) -> None:
        name = str(self.module_id or "").strip()
        if not name or len(name) > 128 or any(c in name for c in "\x00\r\n"):
            raise ValueError("module_id is invalid")
        if isinstance(self.generation, bool) or int(self.generation) != self.generation:
            raise ValueError("module generation is invalid")
        if self.generation < 0:
            raise ValueError("module generation is invalid")
        if not isinstance(self.configuration, Mapping) or not isinstance(self.services, Mapping):
            raise ValueError("module context mappings are invalid")
        object.__setattr__(self, "module_id", name)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "configuration", dict(self.configuration))
        object.__setattr__(self, "services", dict(self.services))


ModuleFactory: TypeAlias = Callable[[ModuleContext], object | Awaitable[object]]


@dataclass(frozen=True, slots=True)
class ModuleDescriptor:
    """模块描述；``instance`` 用于接管已有内置对象。"""

    module_id: str
    factory: ModuleFactory | None = None
    instance: object | None = None
    source: str = "runtime"
    dependencies: tuple[str, ...] = ()
    enabled: bool = True
    adopted: bool = False
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.module_id or "").strip()
        if not name or len(name) > 128 or any(c in name for c in "\x00\r\n"):
            raise ValueError("module_id is invalid")
        if self.factory is None and self.instance is None:
            raise ValueError("module factory or instance is required")
        if self.factory is not None and not callable(self.factory):
            raise TypeError("module factory must be callable")
        if isinstance(self.dependencies, (str, bytes, bytearray)):
            raise ValueError("module dependencies must be a sequence")
        dependencies: list[str] = []
        for item in self.dependencies:
            dependency = str(item or "").strip()
            if not dependency or dependency == name or dependency in dependencies:
                raise ValueError("module dependency is invalid")
            dependencies.append(dependency)
        object.__setattr__(self, "module_id", name)
        object.__setattr__(self, "source", str(self.source or "runtime").strip() or "runtime")
        object.__setattr__(self, "dependencies", tuple(dependencies))
        object.__setattr__(self, "configuration", dict(self.configuration))


@dataclass(frozen=True, slots=True)
class ModuleStatus:
    """可公开投影的模块状态。"""

    module_id: str
    state: ModuleState
    generation: int = 0
    source: str = "runtime"
    dependencies: tuple[str, ...] = ()
    adopted: bool = False
    failure_count: int = 0
    error_type: str = ""
    health: str = "unknown"

    def public(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "state": self.state.value,
            "generation": max(0, int(self.generation)),
            "source": source_label(self.source),
            "dependencies": self.dependencies,
            "adopted": bool(self.adopted),
            "failure_count": max(0, int(self.failure_count)),
            "error_type": self.error_type[:64],
            "health": self.health[:32],
        }


class ModuleLifecycle(Protocol):
    """模块实例可选实现的生命周期钩子。"""

    async def load(self, context: ModuleContext) -> object: ...

    async def start(self) -> object: ...

    async def stop(self) -> object: ...

    async def unload(self) -> object: ...

    async def reload(self, context: ModuleContext) -> object: ...

    async def health(self) -> object: ...


__all__ = [
    "ModuleContext",
    "ModuleDescriptor",
    "ModuleFactory",
    "ModuleGenerationGate",
    "ModuleLifecycle",
    "ModuleState",
    "ModuleStatus",
]
