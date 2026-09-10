"""桌面平台适配器工厂。"""

from __future__ import annotations

import os
import platform as _platform
import sys
from collections.abc import Mapping

from .linux import LinuxDesktopPlatform
from .protocol import (
    CapabilityState,
    DesktopPlatform,
    PlatformCapability,
    PlatformSnapshot,
    WindowContext,
)
from .windows import WindowsDesktopPlatform


class UnsupportedDesktopPlatform:
    """未实现平台的 fail-closed 适配器。"""

    _system: str

    def __init__(self, system: str) -> None:
        self._system = str(system or "unknown")

    @property
    def backend(self) -> str:
        return f"unsupported:{self._system.lower()}"

    def probe(self) -> PlatformSnapshot:
        capabilities = tuple(
            PlatformCapability(name=name, state=CapabilityState.UNAVAILABLE, detail="平台未实现")
            for name in (
                "active_window",
                "cursor_position",
                "process_context",
                "click_through",
                "always_on_top",
                "global_hotkeys",
                "overlay_position",
                "screen_capture",
                "input_control",
                "system_idle",
            )
        )
        return PlatformSnapshot(backend=self.backend, capabilities=capabilities)

    def active_window(self) -> WindowContext | None:
        return None

    def system_idle_seconds(self) -> float | None:
        return None

    def cursor_position(self) -> Mapping[str, object]:
        return {
            "status": "unavailable",
            "backend": self.backend,
            "reason": "平台未实现",
        }

    def foreground_window(self) -> Mapping[str, object]:
        return {"status": "unavailable", "backend": self.backend, "reason": "平台未实现"}

    def list_processes(self, limit: int = 20) -> Mapping[str, object]:
        del limit
        return {"status": "unavailable", "processes": [], "reason": "平台未实现"}

    def capture_screen(
        self,
        *,
        scope: str = "screen",
        region: Mapping[str, object] | None = None,
        window_id: int | None = None,
    ) -> Mapping[str, object]:
        del scope, region, window_id
        return {"status": "unavailable", "reason": "平台未实现"}

    def click_at(self, x: int, y: int, *, button: str = "left") -> Mapping[str, object]:
        del x, y, button
        return {"status": "unavailable", "reason": "平台未实现"}

    def automation_batch(self, steps: object) -> Mapping[str, object]:
        del steps
        return {"status": "unavailable", "reason": "平台未实现"}

    def set_click_through(self, window: object, enabled: bool) -> PlatformCapability:
        del window, enabled
        return PlatformCapability(
            name="click_through",
            state=CapabilityState.UNAVAILABLE,
            detail="平台未实现",
        )

    def set_input_shape(self, window: object, region: object | None) -> PlatformCapability:
        del window, region
        return PlatformCapability(
            name="input_shape",
            state=CapabilityState.UNAVAILABLE,
            detail="平台未实现",
        )

    def always_on_top_status(self, window: object) -> Mapping[str, object]:
        del window
        return {
            "status": "unavailable",
            "enabled": False,
            "confirmed": False,
            "detail": "平台未实现",
        }

    def move_overlay(self, window: object, x: int, y: int) -> PlatformCapability:
        del window, x, y
        return PlatformCapability(
            name="overlay_position",
            state=CapabilityState.UNAVAILABLE,
            detail="平台未实现",
        )


def _resolved_system(system: str | None) -> str:
    if system is not None:
        return str(system)
    if os.name == "nt" or sys.platform == "win32":
        return "Windows"
    if sys.platform == "linux":
        return "Linux"
    return str(_platform.system() or "unknown")


def create_desktop_platform(
    *,
    system: str | None = None,
    environ: Mapping[str, str] | None = None,
    include_command_line: bool = False,
) -> DesktopPlatform:
    """按运行操作系统创建桌面平台适配器。

    ``system`` 仅用于测试或宿主显式装配，正常运行时由 Python 运行环境
    自动解析。未知系统返回 fail-closed 适配器，不复用 Linux 实现。
    """

    resolved = _resolved_system(system).strip().lower()
    if resolved in {"windows", "win32"}:
        return WindowsDesktopPlatform(
            environ,
            include_command_line=include_command_line,
            is_windows=True,
        )
    if resolved in {"linux", "linux2"}:
        return LinuxDesktopPlatform(environ, include_command_line=include_command_line)
    return UnsupportedDesktopPlatform(resolved)


__all__ = ["UnsupportedDesktopPlatform", "create_desktop_platform"]
