"""桌面平台适配器。"""

from .factory import UnsupportedDesktopPlatform, create_desktop_platform
from .linux import (
    BACKEND_UNKNOWN,
    BACKEND_WAYLAND,
    BACKEND_X11,
    LinuxDesktopPlatform,
    X11WindowContextProvider,
    detect_linux_backend,
    probe_linux_platform,
    probe_x11_compositor,
    probe_x11_idle_seconds,
    probe_x11_input_control,
    set_window_click_through,
    set_window_input_shape,
)
from .protocol import (
    Capability,
    CapabilityState,
    DesktopContext,
    DesktopPlatform,
    PlatformCapability,
    PlatformSnapshot,
    WindowContext,
)
from .windows import (
    BACKEND_WINDOWS,
    WindowsDesktopPlatform,
    probe_windows_platform,
)

__all__ = [
    "BACKEND_UNKNOWN",
    "BACKEND_WAYLAND",
    "BACKEND_X11",
    "BACKEND_WINDOWS",
    "Capability",
    "CapabilityState",
    "DesktopContext",
    "DesktopPlatform",
    "LinuxDesktopPlatform",
    "WindowsDesktopPlatform",
    "UnsupportedDesktopPlatform",
    "create_desktop_platform",
    "PlatformCapability",
    "PlatformSnapshot",
    "WindowContext",
    "X11WindowContextProvider",
    "detect_linux_backend",
    "probe_linux_platform",
    "probe_x11_idle_seconds",
    "probe_x11_input_control",
    "probe_x11_compositor",
    "set_window_click_through",
    "set_window_input_shape",
    "probe_windows_platform",
]
