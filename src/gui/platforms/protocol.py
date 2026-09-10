from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from time import time
from typing import Protocol


class CapabilityState(StrEnum):
    """单项能力的探测状态。"""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlatformCapability:
    """平台能力探测结果。"""

    name: str
    state: CapabilityState | str
    detail: str = ""
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise ValueError("capability name is required")
        try:
            state = CapabilityState(self.state)
        except (TypeError, ValueError) as exc:
            raise ValueError("capability state is unsupported") from exc
        evidence = tuple(str(item) for item in (self.evidence or ()) if str(item))
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "detail", str(self.detail or ""))
        object.__setattr__(self, "evidence", evidence)

    @property
    def available(self) -> bool:
        """返回能力是否可以直接使用。"""

        return self.state is CapabilityState.AVAILABLE


# 兼容调用方更短的命名，同时保留一个唯一的实现类型。
Capability = PlatformCapability


@dataclass(frozen=True)
class PlatformSnapshot:
    """一次平台能力探测的不可变快照。"""

    backend: str
    capabilities: tuple[PlatformCapability, ...] = ()
    details: Mapping[str, object] = field(default_factory=dict)
    observed_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        backend = str(self.backend or "").strip().lower()
        if not backend:
            raise ValueError("platform backend is required")
        capabilities = tuple(self.capabilities or ())
        if any(not isinstance(item, PlatformCapability) for item in capabilities):
            raise TypeError("capabilities must contain PlatformCapability values")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "details", dict(self.details or {}))
        object.__setattr__(self, "observed_at", float(self.observed_at))

    def capability(self, name: str) -> PlatformCapability | None:
        """按精确名称查找能力。"""

        for item in self.capabilities:
            if item.name == name:
                return item
        return None

    def supports(self, name: str) -> bool:
        """返回指定能力是否可用。"""

        item = self.capability(name)
        return bool(item and item.available)


@dataclass(frozen=True)
class WindowContext:
    """当前前台窗口的最小上下文。

    标题、命令行和可执行文件都可能包含敏感信息。平台适配器应在授权后才
    填充这些字段；上层可以用 ``redacted`` 表示已经做过脱敏。
    """

    backend: str
    window_id: str = ""
    title: str = ""
    app_id: str = ""
    pid: int | None = None
    process_name: str = ""
    executable: str = ""
    command_line: tuple[str, ...] = ()
    geometry: tuple[int, int, int, int] | None = None
    source: str = ""
    redacted: bool = False
    observed_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        backend = str(self.backend or "").strip().lower()
        if not backend:
            raise ValueError("window backend is required")
        pid = self.pid
        if pid is not None:
            pid = int(pid)
            if pid <= 0:
                pid = None
        geometry = self.geometry
        if geometry is not None:
            geometry = tuple(int(value) for value in geometry)
            if len(geometry) != 4:
                raise ValueError("window geometry must contain four integers")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "window_id", str(self.window_id or ""))
        object.__setattr__(self, "title", str(self.title or ""))
        object.__setattr__(self, "app_id", str(self.app_id or ""))
        object.__setattr__(self, "pid", pid)
        object.__setattr__(self, "process_name", str(self.process_name or ""))
        object.__setattr__(self, "executable", str(self.executable or ""))
        object.__setattr__(
            self,
            "command_line",
            tuple(str(item) for item in (self.command_line or ())),
        )
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "source", str(self.source or ""))
        object.__setattr__(self, "redacted", bool(self.redacted))
        object.__setattr__(self, "observed_at", float(self.observed_at))


DesktopContext = WindowContext


class DesktopPlatform(Protocol):
    """桌面能力适配器的最小接口。"""

    @property
    def backend(self) -> str:
        """返回当前桌面后端名称。"""

    def probe(self) -> PlatformSnapshot:
        """探测平台能力。"""

    def active_window(self) -> WindowContext | None:
        """读取当前前台窗口；不可用时返回 ``None``。"""

    def cursor_position(self) -> Mapping[str, object]:
        """读取当前全局光标位置；不可用时返回明确的 fail-closed 结果。"""

    def system_idle_seconds(self) -> float | None:
        """读取可选的系统空闲秒数；未实现或不可用时返回 ``None``。"""

    def foreground_window(self) -> Mapping[str, object]:
        """返回适合工具层消费的前台窗口摘要。"""

    def list_processes(self, limit: int = 20) -> Mapping[str, object]:
        """返回当前用户可见的进程摘要。"""

    def capture_screen(
        self,
        *,
        scope: str = "screen",
        region: Mapping[str, object] | None = None,
        window_id: int | None = None,
    ) -> Mapping[str, object]:
        """捕获屏幕内容；窗口范围可接收已在后台解析的窗口 ID。"""

    def click_at(self, x: int, y: int, *, button: str = "left") -> Mapping[str, object]:
        """在明确授权后点击一个全局屏幕坐标；不可用时返回失败状态。"""

    def automation_batch(self, steps: object) -> Mapping[str, object]:
        """执行已审批的有限键盘、鼠标和窗口自动化步骤。"""

    def set_click_through(self, window: object, enabled: bool) -> PlatformCapability:
        """设置窗口点击穿透状态。"""

    def set_input_shape(self, window: object, region: object | None) -> PlatformCapability:
        """设置与视觉遮罩分离的原生输入区域；不可用时返回降级结果。"""

    def always_on_top_status(self, window: object) -> Mapping[str, object]:
        """回读窗口管理器实际保留的置顶状态。"""

    def move_overlay(self, window: object, x: int, y: int) -> PlatformCapability:
        """请求移动桌宠窗口。"""


__all__ = [
    "Capability",
    "CapabilityState",
    "DesktopContext",
    "DesktopPlatform",
    "PlatformCapability",
    "PlatformSnapshot",
    "WindowContext",
]
