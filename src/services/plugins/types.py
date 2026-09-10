"""桌宠插件生命周期的稳定数据契约。"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, TypeAlias


class PluginState(StrEnum):
    """插件在管理器中的可观察状态。"""

    DISCOVERED = "discovered"
    LOADING = "loading"
    LOADED = "loaded"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    FAILED = "failed"
    DISABLED = "disabled"


def plugin_source_label(value: object) -> str:
    """将插件来源缩减为可诊断但不泄漏本机路径的稳定标签。"""

    source = str(value or "runtime").strip() or "runtime"
    if source == "runtime":
        return source
    if source.startswith("entry_point:"):
        name = source.partition(":")[2].strip()
        return f"entry_point:{name[:128]}" if name else "entry_point"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return f"file:sha256:{digest}"


@dataclass(frozen=True, slots=True)
class PluginContext:
    """插件初始化时收到的运行时上下文。

    ``services`` 只包含组合根显式提供的对象；插件不能通过上下文隐式访问
    进程环境、配置文件路径或未声明的全局对象。
    """

    plugin_id: str
    generation: int
    configuration: Mapping[str, Any] = field(default_factory=dict)
    services: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        plugin_id = str(self.plugin_id or "").strip()
        if (
            not plugin_id
            or len(plugin_id) > 128
            or any(character in plugin_id for character in "\x00\r\n")
        ):
            raise ValueError("plugin_id is invalid")
        if isinstance(self.generation, bool) or int(self.generation) != self.generation:
            raise ValueError("plugin generation is invalid")
        if self.generation < 0:
            raise ValueError("plugin generation is invalid")
        if not isinstance(self.configuration, Mapping):
            raise ValueError("plugin configuration must be a mapping")
        if not isinstance(self.services, Mapping):
            raise ValueError("plugin services must be a mapping")
        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "configuration", dict(self.configuration))
        object.__setattr__(self, "services", dict(self.services))


class PluginFactory(Protocol):
    """从上下文创建一个插件实例。"""

    def __call__(self, context: PluginContext) -> object | Awaitable[object]: ...


PluginFactoryLike: TypeAlias = Callable[[PluginContext], object | Awaitable[object]]


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    """插件发现结果及其显式依赖。"""

    plugin_id: str
    factory: PluginFactoryLike
    source: str = "runtime"
    version: str = ""
    dependencies: tuple[str, ...] = ()
    fingerprint: str = ""

    def __post_init__(self) -> None:
        plugin_id = str(self.plugin_id or "").strip()
        if (
            not plugin_id
            or len(plugin_id) > 128
            or any(character in plugin_id for character in "\x00\r\n")
        ):
            raise ValueError("plugin_id is invalid")
        if not callable(self.factory):
            raise TypeError("plugin factory must be callable")
        source = str(self.source or "runtime").strip()
        if not source or len(source) > 1024 or any(character in source for character in "\x00\r\n"):
            raise ValueError("plugin source is invalid")
        version = str(self.version or "").strip()
        if len(version) > 128 or any(character in version for character in "\x00\r\n"):
            raise ValueError("plugin version is invalid")
        if isinstance(self.dependencies, (str, bytes, bytearray)):
            raise ValueError("plugin dependencies must be a sequence")
        dependencies: list[str] = []
        for dependency in self.dependencies:
            name = str(dependency or "").strip()
            if not name or name == plugin_id or len(name) > 128:
                raise ValueError("plugin dependency is invalid")
            if any(character in name for character in "\x00\r\n"):
                raise ValueError("plugin dependency is invalid")
            if name in dependencies:
                raise ValueError("plugin dependencies contain duplicates")
            dependencies.append(name)
        fingerprint = str(self.fingerprint or "").strip()
        if len(fingerprint) > 128 or any(character in fingerprint for character in "\x00\r\n"):
            raise ValueError("plugin fingerprint is invalid")
        object.__setattr__(self, "plugin_id", plugin_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "dependencies", tuple(dependencies))
        object.__setattr__(self, "fingerprint", fingerprint)


@dataclass(frozen=True, slots=True)
class PluginStatus:
    """可交给 UI、日志和诊断工具的插件状态。"""

    plugin_id: str
    state: PluginState
    generation: int = 0
    source: str = "runtime"
    version: str = ""
    dependencies: tuple[str, ...] = ()
    failure_count: int = 0
    error_type: str = ""
    health: str = "unknown"

    def public(self) -> dict[str, object]:
        """返回不包含异常正文、配置值或路径参数的状态。"""

        return {
            "plugin_id": self.plugin_id,
            "state": self.state.value,
            "generation": self.generation,
            "source": plugin_source_label(self.source),
            "version": self.version[:128],
            "dependencies": self.dependencies,
            "failure_count": max(0, int(self.failure_count)),
            "error_type": self.error_type[:64],
            "health": self.health[:32],
        }


@dataclass(frozen=True, slots=True)
class PluginSettings:
    """插件发现与热重载配置。"""

    enabled: bool = True
    directories: tuple[str, ...] = ()
    entry_points_enabled: bool = False
    entry_point_group: str = "meapet.plugins"
    reload_enabled: bool = True
    reload_interval_seconds: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PluginSettings:
        """从 ``plugins`` 配置区解析严格字段。"""

        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("plugins must be a mapping")

        defaults = cls()

        def boolean(name: str, default: bool) -> bool:
            raw = value.get(name, default)
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, int) and raw in {0, 1}:
                return bool(raw)
            if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
                return raw.strip().lower() == "true"
            raise ValueError(f"plugins.{name} must be a boolean")

        raw_directories = value.get("directories", ())
        if raw_directories is None:
            raw_directories = ()
        if isinstance(raw_directories, (str, bytes, bytearray)) or not isinstance(
            raw_directories, Sequence
        ):
            raise ValueError("plugins.directories must be a list")
        directories: list[str] = []
        for directory in raw_directories:
            path = str(directory or "").strip()
            if not path or len(path) > 4096 or any(character in path for character in "\x00\r\n"):
                raise ValueError("plugins.directories contains an invalid path")
            if path not in directories:
                directories.append(path)

        group = str(value.get("entry_point_group", defaults.entry_point_group) or "").strip()
        if not group or len(group) > 128 or any(character in group for character in "\x00\r\n"):
            raise ValueError("plugins.entry_point_group is invalid")

        raw_interval = value.get("reload_interval_seconds", defaults.reload_interval_seconds)
        if isinstance(raw_interval, bool):
            raise ValueError("plugins.reload_interval_seconds must be a number")
        try:
            interval = float(raw_interval)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("plugins.reload_interval_seconds must be a number") from exc
        if not 0.1 <= interval <= 60.0:
            raise ValueError("plugins.reload_interval_seconds is outside the allowed range")

        return cls(
            enabled=boolean("enabled", defaults.enabled),
            directories=tuple(directories),
            entry_points_enabled=boolean("entry_points_enabled", defaults.entry_points_enabled),
            entry_point_group=group,
            reload_enabled=boolean("reload_enabled", defaults.reload_enabled),
            reload_interval_seconds=interval,
        )


class PluginLifecycle(Protocol):
    """插件实例可选实现的异步生命周期。"""

    async def load(self, context: PluginContext) -> object: ...

    async def start(self) -> object: ...

    async def stop(self) -> object: ...

    async def unload(self) -> object: ...

    async def reload(self, context: PluginContext) -> object: ...

    async def health(self) -> object: ...


__all__ = [
    "PluginContext",
    "PluginDescriptor",
    "PluginFactory",
    "PluginFactoryLike",
    "PluginLifecycle",
    "PluginSettings",
    "PluginState",
    "PluginStatus",
]
