from __future__ import annotations

import importlib
import logging
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from threading import RLock
from time import sleep, time
from typing import Any, cast

from .protocol import (
    CapabilityState,
    PlatformCapability,
    PlatformSnapshot,
    WindowContext,
)

BACKEND_X11 = "x11"
BACKEND_WAYLAND = "wayland"
BACKEND_UNKNOWN = "unknown"

logger = logging.getLogger(__name__)


def detect_linux_backend(environ: Mapping[str, str] | None = None) -> str:
    """根据已定义的会话变量判断 Linux 图形后端。"""

    values = dict(os.environ if environ is None else environ)
    qt_platform = str(values.get("QT_QPA_PLATFORM", "")).strip().lower()
    if qt_platform == "xcb":
        return BACKEND_X11
    if qt_platform == "wayland":
        return BACKEND_WAYLAND

    session_type = str(values.get("XDG_SESSION_TYPE", "")).strip().lower()
    if session_type == BACKEND_X11:
        return BACKEND_X11
    if session_type == BACKEND_WAYLAND:
        return BACKEND_WAYLAND
    if values.get("WAYLAND_DISPLAY"):
        return BACKEND_WAYLAND
    if values.get("DISPLAY"):
        return BACKEND_X11
    return BACKEND_UNKNOWN


def _import_optional(module_name: str) -> Any | None:
    """动态导入可选依赖；失败时返回 ``None``。"""

    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        return None


def probe_x11_compositor(environ: Mapping[str, str] | None = None) -> bool | None:

    values = dict(os.environ if environ is None else environ)
    if detect_linux_backend(values) != BACKEND_X11:
        return None
    display_name = str(values.get("DISPLAY", "") or "").strip()
    if not display_name:
        return None
    module = _import_optional("Xlib.display")
    display_class = getattr(module, "Display", None) if module is not None else None
    if not callable(display_class):
        return None
    display = None
    try:
        display = display_class(display_name)
        atom = display.intern_atom("_NET_WM_CM_S0", only_if_exists=False)
        owner = display.get_selection_owner(atom)
        return bool(getattr(owner, "id", None))
    except Exception:
        return None
    finally:
        closer = getattr(display, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def probe_x11_idle_seconds(environ: Mapping[str, str] | None = None) -> float | None:
    """读取 X11 MIT-SCREEN-SAVER 提供的系统空闲秒数。

    该探针只在明确的 X11 会话中工作，并且仅读取扩展返回的累计空闲时长；
    不读取键盘内容、窗口标题、进程命令行或屏幕像素。扩展不存在、DISPLAY
    未设置、连接失败或返回值不合法时统一返回 ``None``，调用方必须按未知/非
    活跃处理。原生 Wayland 没有可移植的等价 API，因此不会尝试猜测门户接口。
    """

    values = dict(os.environ if environ is None else environ)
    if detect_linux_backend(values) != BACKEND_X11:
        return None
    display_name = str(values.get("DISPLAY", "") or "").strip()
    if not display_name:
        return None
    # python-xlib 只有在导入扩展模块后才会把 screensaver_query_info 方法
    # 注册到 Display/Window 对象；两个导入都保持可选，避免影响无 X11 启动。
    screensaver = _import_optional("Xlib.ext.screensaver")
    display_module = _import_optional("Xlib.display")
    display_class = getattr(display_module, "Display", None) if display_module else None
    if screensaver is None or not callable(display_class):
        return None
    display = None
    try:
        display = display_class(display_name)
        has_extension = getattr(display, "has_extension", None)
        if callable(has_extension) and not has_extension("MIT-SCREEN-SAVER"):
            return None
        root = display.screen().root
        query = getattr(root, "screensaver_query_info", None)
        if not callable(query):
            return None
        info = query()
        idle_ms = getattr(info, "idle", None)
        if isinstance(idle_ms, bool):
            return None
        idle_ms = float(idle_ms)
        if not math.isfinite(idle_ms) or idle_ms < 0.0:
            return None
        return idle_ms / 1000.0
    except Exception:
        return None
    finally:
        closer = getattr(display, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def probe_x11_input_control(environ: Mapping[str, str] | None = None) -> bool:
    """确认当前 X11 服务端和 python-xlib 都提供 XTEST。"""

    values = dict(os.environ if environ is None else environ)
    if detect_linux_backend(values) != BACKEND_X11:
        return False
    display_name = str(values.get("DISPLAY", "") or "").strip()
    display_module = _import_optional("Xlib.display") if display_name else None
    xtest_module = _import_optional("Xlib.ext.xtest") if display_module is not None else None
    display_class = getattr(display_module, "Display", None)
    if not callable(display_class) or xtest_module is None:
        return False
    display = None
    try:
        display = display_class(display_name)
        has_extension = getattr(display, "has_extension", None)
        return bool(callable(has_extension) and has_extension("XTEST"))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    finally:
        closer = getattr(display, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def _capability(
    name: str,
    state: CapabilityState,
    detail: str,
    *evidence: str,
) -> PlatformCapability:
    return PlatformCapability(name=name, state=state, detail=detail, evidence=tuple(evidence))


def _move_window(window: object, x: int, y: int, *, qt_core: object | None = None) -> None:
    """移动 QWidget 或 QWindow，并严格适配各自的位置参数契约。"""

    mover = getattr(window, "move", None)
    if callable(mover):
        mover(int(x), int(y))
        return
    setter = getattr(window, "setPosition", None)
    if not callable(setter):
        raise AttributeError("window does not expose a position setter")
    core = qt_core or _import_optional("PySide6.QtCore")
    point_type = getattr(core, "QPoint", None) if core is not None else None
    point = point_type(int(x), int(y)) if callable(point_type) else (int(x), int(y))
    try:
        setter(point)
    except TypeError:
        # 测试替身或少数旧 Qt 包装可能仍暴露双参数形式；只有在单参数
        # 调用明确失败后才尝试该兼容签名，避免把 QWindow 的真实契约
        # 错误地当成 QWidget。
        setter(int(x), int(y))


def _qt_topmost_target(window: object) -> object:
    """解析承载顶层标志的原生 QWindow，避免重建 QWidget。"""

    view = getattr(window, "view", None)
    source = view if view is not None else window
    handle_getter = getattr(source, "windowHandle", None)
    if callable(handle_getter):
        try:
            handle = handle_getter()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            handle = None
        if handle is not None:
            return handle
    return source


def _qt_topmost_flag(window: object) -> bool | None:
    """回读原生 Qt 窗口的 WindowStaysOnTopHint。"""

    qt_core = _import_optional("PySide6.QtCore")
    qt = getattr(qt_core, "Qt", None) if qt_core is not None else None
    window_type = getattr(qt, "WindowType", qt)
    flag = getattr(window_type, "WindowStaysOnTopHint", None)
    if flag is None:
        return None
    target = _qt_topmost_target(window)
    for getter_name in ("flags", "windowFlags"):
        getter = getattr(target, getter_name, None)
        if not callable(getter):
            continue
        try:
            return bool(getter() & flag)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return None


def _qt_topmost_window_id(window: object) -> int | None:
    """返回原生顶层窗口 ID；无窗口句柄时明确失败。"""

    target = _qt_topmost_target(window)
    getter = getattr(target, "winId", None)
    if not callable(getter):
        return None
    try:
        window_id = int(cast(Any, getter)())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return window_id if window_id > 0 else None


def _read_x11_topmost_state(
    window: object,
    environ: Mapping[str, str],
) -> tuple[bool | None, tuple[str, ...]]:
    """从 EWMH 属性读取窗口管理器实际保留的置顶状态。"""

    display_name = str(environ.get("DISPLAY", "") or "").strip()
    window_id = _qt_topmost_window_id(window)
    if not display_name or window_id is None:
        return None, ()
    display_module = _import_optional("Xlib.display")
    x_module = _import_optional("Xlib.X")
    display_class = getattr(display_module, "Display", None)
    if not callable(display_class) or x_module is None:
        return None, ()
    connection: Any = None
    try:
        connection = display_class(display_name)
        state_atom = connection.intern_atom("_NET_WM_STATE")
        above_atom = connection.intern_atom("_NET_WM_STATE_ABOVE")
        stays_on_top_atom = connection.intern_atom("_NET_WM_STATE_STAYS_ON_TOP")
        native_window = connection.create_resource_object("window", window_id)
        property_value = native_window.get_full_property(
            state_atom,
            getattr(x_module, "AnyPropertyType", 0),
        )
        values = {
            int(value)
            for value in (
                getattr(property_value, "value", ()) if property_value is not None else ()
            )
        }
        enabled = above_atom in values or stays_on_top_atom in values
        evidence = tuple(
            name
            for name, atom in (
                ("_NET_WM_STATE_ABOVE", above_atom),
                ("_NET_WM_STATE_STAYS_ON_TOP", stays_on_top_atom),
            )
            if atom in values
        )
        return enabled, evidence
    except Exception:
        # Xlib 的 BadWindow/BadAtom 等协议异常不继承 OSError；窗口可能在
        # 异步置顶回读期间被重建或关闭，必须收敛为“无法确认”而非崩溃。
        logger.debug("X11 topmost EWMH readback failed", exc_info=True)
        return None, ()
    finally:
        closer = getattr(connection, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def probe_linux_platform(environ: Mapping[str, str] | None = None) -> PlatformSnapshot:
    """探测 Linux 平台能力，不创建 Qt 窗口。"""

    values = dict(os.environ if environ is None else environ)
    backend = detect_linux_backend(values)
    details = {
        "session_type": str(values.get("XDG_SESSION_TYPE", "")),
        "desktop": str(values.get("XDG_CURRENT_DESKTOP", "")),
        "display": str(values.get("DISPLAY", "")),
        "wayland_display": str(values.get("WAYLAND_DISPLAY", "")),
    }

    if backend == BACKEND_WAYLAND:
        qt_core = _import_optional("PySide6.QtCore")
        capabilities = (
            _capability(
                "active_window",
                CapabilityState.UNAVAILABLE,
                "native Wayland has no portable global active-window API",
            ),
            _capability(
                "cursor_position",
                CapabilityState.UNAVAILABLE,
                "native Wayland has no portable global cursor-position API",
            ),
            _capability(
                "process_context",
                CapabilityState.UNAVAILABLE,
                "a compositor-specific permissioned adapter is required",
            ),
            _capability(
                "click_through",
                CapabilityState.DEGRADED if qt_core else CapabilityState.UNAVAILABLE,
                (
                    "Qt input transparency flag can be requested; compositor behavior "
                    "is not guaranteed"
                    if qt_core
                    else "native Wayland click-through requires PySide6"
                ),
                "Qt.WindowType.WindowTransparentForInput",
            ),
            _capability(
                "always_on_top",
                CapabilityState.DEGRADED if qt_core else CapabilityState.UNAVAILABLE,
                (
                    "Qt WindowStaysOnTopHint can be requested; compositor policy may override it"
                    if qt_core
                    else "native topmost flag requires PySide6"
                ),
                "Qt.WindowType.WindowStaysOnTopHint",
            ),
            _capability(
                "global_hotkeys",
                CapabilityState.UNAVAILABLE,
                "Wayland compositor or portal must provide global shortcut registration",
            ),
            _capability(
                "overlay_position",
                CapabilityState.UNAVAILABLE,
                "top-level client positioning is compositor-controlled",
            ),
            _capability(
                "screen_capture",
                CapabilityState.UNAVAILABLE,
                "native Wayland capture requires a portal adapter",
            ),
            _capability(
                "input_control",
                CapabilityState.UNAVAILABLE,
                "native Wayland input injection requires a permissioned compositor adapter",
            ),
        )
        return PlatformSnapshot(backend=backend, capabilities=capabilities, details=details)

    if backend == BACKEND_X11:
        display_configured = bool(str(values.get("DISPLAY", "")).strip())
        compositor = probe_x11_compositor(values)
        details["compositor"] = (
            "available"
            if compositor is True
            else "unavailable"
            if compositor is False
            else "unknown"
        )
        xlib = _import_optional("Xlib.display") if display_configured else None
        psutil = _import_optional("psutil")
        qt_core = _import_optional("PySide6.QtCore")
        qt_gui = _import_optional("PySide6.QtGui")
        xtest_available = probe_x11_input_control(values) if xlib else False
        xwayland_only = (
            str(values.get("XDG_SESSION_TYPE", "")).strip().lower() == BACKEND_WAYLAND
            and str(values.get("QT_QPA_PLATFORM", "")).strip().lower() == "xcb"
        )
        if xlib:
            x11_state = CapabilityState.DEGRADED if xwayland_only else CapabilityState.AVAILABLE
        else:
            x11_state = CapabilityState.UNAVAILABLE
        cursor_state = (
            CapabilityState.DEGRADED
            if xwayland_only
            else CapabilityState.AVAILABLE
            if xlib
            else CapabilityState.UNAVAILABLE
        )
        if not xlib:
            process_state = CapabilityState.UNAVAILABLE
        elif psutil:
            process_state = CapabilityState.DEGRADED if xwayland_only else CapabilityState.AVAILABLE
        else:
            process_state = CapabilityState.DEGRADED
        click_state = (
            CapabilityState.AVAILABLE
            if qt_core and display_configured
            else CapabilityState.UNAVAILABLE
        )
        if not qt_core or not display_configured:
            topmost_state = CapabilityState.UNAVAILABLE
        elif not xlib or xwayland_only:
            # Qt 旗标只能说明客户端提交了请求；缺少 EWMH 回读或运行在
            # XWayland 时，必须等具体窗口的 _NET_WM_STATE 才能确认置顶。
            topmost_state = CapabilityState.DEGRADED
        else:
            topmost_state = CapabilityState.AVAILABLE
        overlay_state = (
            CapabilityState.AVAILABLE
            if qt_core and display_configured
            else CapabilityState.UNAVAILABLE
        )
        if qt_gui and qt_core and display_configured:
            capture_state = CapabilityState.DEGRADED if xwayland_only else CapabilityState.AVAILABLE
        else:
            capture_state = CapabilityState.UNAVAILABLE
        input_state = (
            CapabilityState.DEGRADED
            if xwayland_only and xtest_available
            else CapabilityState.AVAILABLE
            if xtest_available
            else CapabilityState.UNAVAILABLE
        )
        if xwayland_only:
            details["scope"] = "xwayland"
        capabilities = (
            _capability(
                "active_window",
                x11_state,
                (
                    "EWMH active-window properties are readable through Xwayland"
                    if xwayland_only and xlib
                    else "EWMH active-window properties are readable"
                    if xlib
                    else "python-xlib or DISPLAY is unavailable"
                ),
                "_NET_ACTIVE_WINDOW",
            ),
            _capability(
                "cursor_position",
                cursor_state,
                (
                    "X11 QueryPointer is available through Xwayland"
                    if xwayland_only and xlib
                    else "X11 QueryPointer is available"
                    if xlib
                    else "python-xlib or DISPLAY is unavailable"
                ),
                "QueryPointer",
            ),
            _capability(
                "process_context",
                process_state,
                (
                    "_NET_WM_PID can be mapped with psutil through Xwayland"
                    if xwayland_only and xlib and psutil
                    else "_NET_WM_PID can be mapped with psutil"
                    if xlib and psutil
                    else "python-xlib or DISPLAY is unavailable"
                    if not xlib
                    else "process metadata is partial"
                ),
                "_NET_WM_PID",
            ),
            _capability(
                "click_through",
                click_state,
                (
                    "Qt runtime is available for WindowTransparentForInput"
                    if qt_core and display_configured
                    else "PySide6 is unavailable"
                    if not qt_core
                    else "DISPLAY is unavailable"
                ),
                "Qt.WindowType.WindowTransparentForInput",
            ),
            _capability(
                "always_on_top",
                topmost_state,
                (
                    "XWayland topmost requires per-window EWMH readback"
                    if xwayland_only and qt_core and display_configured
                    else "Qt topmost requests require EWMH readback, but python-xlib is unavailable"
                    if qt_core and display_configured and not xlib
                    else "Qt WindowStaysOnTopHint and EWMH readback are available"
                    if qt_core and display_configured
                    else "PySide6 or DISPLAY is unavailable"
                ),
                "Qt.WindowType.WindowStaysOnTopHint",
                "_NET_WM_STATE_ABOVE",
                "_NET_WM_STATE_STAYS_ON_TOP",
            ),
            _capability(
                "global_hotkeys",
                CapabilityState.AVAILABLE if xlib else CapabilityState.UNAVAILABLE,
                (
                    "X11 root-window key grabs are available"
                    if xlib
                    else "python-xlib or DISPLAY is unavailable"
                ),
                "XGrabKey",
            ),
            _capability(
                "overlay_position",
                overlay_state,
                "X11 permits a position request through QWindow"
                if qt_core and display_configured
                else "PySide6 is unavailable",
            ),
            _capability(
                "screen_capture",
                capture_state,
                (
                    "Qt screen grab covers the Xwayland surface"
                    if capture_state is CapabilityState.DEGRADED
                    else "Qt screen grab is available"
                    if capture_state is CapabilityState.AVAILABLE
                    else "PySide6 is unavailable"
                ),
            ),
            _capability(
                "input_control",
                input_state,
                (
                    "XTEST input injection is limited to Xwayland surfaces"
                    if input_state is CapabilityState.DEGRADED
                    else "XTEST input injection is available"
                    if input_state is CapabilityState.AVAILABLE
                    else "python-xlib XTEST support or DISPLAY is unavailable"
                ),
                "XTEST",
            ),
        )
        return PlatformSnapshot(backend=backend, capabilities=capabilities, details=details)

    capabilities = tuple(
        _capability(name, CapabilityState.UNAVAILABLE, "Linux display backend is unknown")
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
        )
    )
    return PlatformSnapshot(backend=backend, capabilities=capabilities, details=details)


def _decode_text(value: object) -> str:
    """解码 X11 属性中的文本值。"""

    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.rstrip("\x00")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return ""
        if all(isinstance(item, int) for item in value):
            try:
                return bytes(value).rstrip(b"\x00").decode("utf-8", errors="replace")
            except (ValueError, UnicodeError):
                return ""
        return _decode_text(value[0])
    return str(value or "")


def _first_int(value: object) -> int | None:
    """从 X11 CARDINAL/WINDOW 属性中读取首个整数。"""

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        value = value[0] if value else None
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed and parsed > 0 else None


def _read_window_geometry(window: object, root: object) -> tuple[int, int, int, int] | None:
    """读取 X11 窗口几何；窗口对象不支持时保持为空。"""

    try:
        geometry = window.get_geometry()
        x = int(getattr(geometry, "x"))
        y = int(getattr(geometry, "y"))
        width = int(getattr(geometry, "width"))
        height = int(getattr(geometry, "height"))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None

    # get_geometry() 的坐标通常是父窗口相对坐标；尝试转换到根窗口坐标。
    try:
        translated = window.translate_coords(root, 0, 0)
        x = int(getattr(translated, "x"))
        y = int(getattr(translated, "y"))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    return (x, y, width, height)


def _window_app_id(value: object) -> str:
    """从 WM_CLASS 的两个 NUL 分隔字段取 class 字段。"""

    if isinstance(value, bytes):
        fields = [field for field in value.split(b"\x00") if field]
        return _decode_text(fields[-1]) if fields else ""
    if isinstance(value, str):
        fields = [field for field in value.split("\x00") if field]
        return fields[-1] if fields else ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if all(isinstance(item, int) for item in value):
            return _window_app_id(bytes(value))
        return _decode_text(value[-1]) if value else ""
    return ""


def _capture_region(value: Mapping[str, object] | None) -> tuple[int, int, int, int]:
    """校验区域坐标，避免把任意对象传入 Qt 图像 API。"""

    if not isinstance(value, Mapping):
        raise ValueError("capture region must be a mapping")
    # ``bool`` 是 ``int`` 的子类；若直接调用 ``int(True)``，调用方传入的
    # JSON 布尔值会被静默解释为一个像素坐标，导致截取了错误区域。
    try:
        raw_values = tuple(value[key] for key in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("capture region requires x, y, width and height") from exc
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_values):
        raise ValueError("capture region values must be integers")
    x, y, width, height = cast(tuple[int, int, int, int], raw_values)
    if width <= 0 or height <= 0 or width > 8192 or height > 8192:
        raise ValueError("capture region dimensions are out of range")
    return x, y, width, height


def _screen_geometry(screen: object) -> tuple[int, int, int, int] | None:
    """读取 QScreen 的虚拟桌面几何，失败时返回 ``None``。"""

    getter = getattr(screen, "geometry", None)
    if not callable(getter):
        return None
    try:
        geometry = getter()
        fields = tuple(getattr(geometry, name, None) for name in ("x", "y", "width", "height"))
        if not all(callable(field) for field in fields):
            return None
        x, y, width, height = (int(field()) for field in fields)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _read_x11_cursor_position(environ: Mapping[str, str]) -> Mapping[str, object]:
    """通过 X11 QueryPointer 读取全局光标；协议或依赖不可用时拒绝返回坐标。"""

    if detect_linux_backend(environ) != BACKEND_X11:
        return {
            "status": "unavailable",
            "backend": detect_linux_backend(environ),
            "reason": "global cursor position requires X11",
        }
    display_name = str(environ.get("DISPLAY", "") or "").strip()
    if not display_name:
        return {"status": "unavailable", "backend": BACKEND_X11, "reason": "DISPLAY is unavailable"}
    display_module = _import_optional("Xlib.display")
    x_module = _import_optional("Xlib.X")
    display_class = getattr(display_module, "Display", None) if display_module is not None else None
    if not callable(display_class) or x_module is None:
        return {
            "status": "unavailable",
            "backend": BACKEND_X11,
            "reason": "python-xlib is unavailable",
        }
    display = None
    try:
        display = display_class(display_name)
        screen_getter = getattr(display, "screen", None)
        if not callable(screen_getter):
            return {
                "status": "unavailable",
                "backend": BACKEND_X11,
                "reason": "X11 screen is unavailable",
            }
        screen = screen_getter()
        root = getattr(screen, "root", None)
        query_pointer = getattr(root, "query_pointer", None)
        if not callable(query_pointer):
            return {
                "status": "unavailable",
                "backend": BACKEND_X11,
                "reason": "X11 pointer query is unavailable",
            }
        pointer = query_pointer()
        raw_x = getattr(pointer, "root_x", None)
        raw_y = getattr(pointer, "root_y", None)
        if isinstance(raw_x, bool) or isinstance(raw_y, bool):
            return {
                "status": "unavailable",
                "backend": BACKEND_X11,
                "reason": "X11 pointer coordinates are invalid",
            }
        x = int(raw_x)
        y = int(raw_y)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return {
            "status": "unavailable",
            "backend": BACKEND_X11,
            "reason": "X11 pointer query failed",
        }
    finally:
        closer = getattr(display, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
    return {
        "status": "available",
        "backend": BACKEND_X11,
        "x": x,
        "y": y,
        "source": "x11-query-pointer",
    }


def _screen_for_capture_region(
    application: object,
    primary_screen: object,
    region: tuple[int, int, int, int],
) -> tuple[object, int, int] | None:
    """把全局虚拟桌面区域映射到单个 QScreen 的局部坐标。

    X11 的 Qt ``QScreen.grabWindow`` 参数是目标屏幕局部坐标，而工具层的
    OCR/点击坐标使用虚拟桌面全局坐标。跨屏区域会被拒绝，避免把第二块屏幕
    的请求错误地裁剪到主屏左上角。
    """

    x, y, width, height = region
    right = x + width
    bottom = y + height
    screens_getter = getattr(application, "screens", None)
    if not callable(screens_getter):
        screens = (primary_screen,)
    else:
        try:
            screens = tuple(screens_getter()) or (primary_screen,)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            screens = (primary_screen,)
    for screen in screens:
        geometry = _screen_geometry(screen)
        if geometry is None:
            continue
        left, top, screen_width, screen_height = geometry
        if (
            x >= left
            and y >= top
            and right <= left + screen_width
            and bottom <= top + screen_height
        ):
            return screen, x - left, y - top
    return None


def _x11_window_geometry(
    environ: Mapping[str, str],
    window_id: int,
) -> tuple[int, int, int, int] | None:
    """按已解析的 X11 窗口 ID 读取根坐标，避免焦点切换导致原点错配。"""

    display_name = str(environ.get("DISPLAY", "") or "").strip()
    if not display_name or window_id <= 0:
        return None
    display_module = _import_optional("Xlib.display")
    display_class = getattr(display_module, "Display", None) if display_module else None
    if not callable(display_class):
        return None
    display = None
    try:
        display = display_class(display_name)
        root = display.screen().root
        window = display.create_resource_object("window", int(window_id))
        return _read_window_geometry(window, root)
    except Exception:
        logger.debug("X11 window geometry readback failed", exc_info=True)
        return None
    finally:
        closer = getattr(display, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass


def _screen_for_window_id(
    application: object,
    primary_screen: object,
    window_id: int,
) -> object:
    """按同一 X11 窗口句柄选择 Qt 所属屏幕，支持多屏截图。"""

    all_windows = getattr(application, "allWindows", None)
    if not callable(all_windows):
        return primary_screen
    try:
        windows = tuple(all_windows())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return primary_screen
    for candidate in windows:
        getter = getattr(candidate, "winId", None)
        if not callable(getter):
            continue
        try:
            if int(getter()) != window_id:
                continue
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
            continue
        screen_getter = getattr(candidate, "screen", None)
        if not callable(screen_getter):
            continue
        try:
            screen = screen_getter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            continue
        if screen is not None:
            return screen
    return primary_screen


class X11WindowContextProvider:
    """读取 X11 EWMH 前台窗口和对应进程信息。"""

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        include_command_line: bool = False,
        excluded_window_ids: Iterable[int] | None = None,
    ) -> None:
        self._environment = dict(os.environ if environ is None else environ)
        self._include_command_line = bool(include_command_line)
        self._excluded_window_ids = {int(value) for value in (excluded_window_ids or ())}

    def read(self) -> WindowContext | None:
        """读取一次快照；窗口在读取期间消失时返回 ``None``。"""

        if not self._environment.get("DISPLAY"):
            return None
        display_module = _import_optional("Xlib.display")
        x_module = _import_optional("Xlib.X")
        if display_module is None or x_module is None:
            return None
        display = None
        try:
            display = display_module.Display(self._environment.get("DISPLAY"))
            root = display.screen().root
            active_atom = display.intern_atom("_NET_ACTIVE_WINDOW", only_if_exists=True)
            if not active_atom:
                return None
            active_property = root.get_full_property(active_atom, x_module.AnyPropertyType)
            active_id = _first_int(getattr(active_property, "value", None))
            if active_id is None or active_id in self._excluded_window_ids:
                return None
            window = display.create_resource_object("window", active_id)

            def read_property(name: str) -> object | None:
                atom = display.intern_atom(name, only_if_exists=True)
                if not atom:
                    return None
                prop = window.get_full_property(atom, x_module.AnyPropertyType)
                return getattr(prop, "value", None) if prop is not None else None

            title = _decode_text(read_property("_NET_WM_NAME"))
            if not title:
                title = _decode_text(read_property("WM_NAME"))
            wm_class = read_property("WM_CLASS")
            app_id = _window_app_id(wm_class)
            pid = _first_int(read_property("_NET_WM_PID"))
            geometry = _read_window_geometry(window, root)

            process_name = ""
            executable = ""
            command_line: tuple[str, ...] = ()
            if pid is not None:
                psutil = _import_optional("psutil")
                if psutil is not None:
                    try:
                        process = psutil.Process(pid)
                        process_name = str(process.name() or "")
                        executable = str(process.exe() or "")
                        if self._include_command_line:
                            command_line = tuple(str(item) for item in (process.cmdline() or ()))
                    except Exception:
                        # 进程权限异常不应丢弃已经读取到的窗口信息。
                        pass

            return WindowContext(
                backend=BACKEND_X11,
                window_id=f"0x{active_id:x}",
                title=title,
                app_id=app_id,
                pid=pid,
                process_name=process_name,
                executable=executable,
                command_line=command_line,
                geometry=geometry,
                source="x11-ewmh",
                observed_at=time(),
            )
        except Exception:
            # 前台窗口可能在属性读取期间被销毁；观察失败时按不可用处理。
            return None
        finally:
            if display is not None:
                try:
                    display.close()
                except Exception:
                    pass

    get_active_window = read


def set_window_click_through(
    window: object,
    enabled: bool,
    *,
    backend: str | None = None,
) -> PlatformCapability:
    """请求 Qt 点击穿透，并按 X11/Wayland 能力边界返回状态。

    Wayland 会在 Qt 标志确实保留时返回 ``degraded``：客户端只能证明请求已
    提交，不能替 compositor 保证最终输入路由。
    """

    resolved_backend = (backend or detect_linux_backend()).strip().lower()
    if resolved_backend not in {BACKEND_X11, BACKEND_WAYLAND}:
        return _capability(
            "click_through",
            CapabilityState.UNAVAILABLE,
            "click-through requires an X11 or Wayland Qt backend",
        )
    qt_core = _import_optional("PySide6.QtCore")
    if qt_core is None:
        return _capability(
            "click_through",
            CapabilityState.UNAVAILABLE,
            "PySide6 is unavailable",
        )
    try:
        flag = qt_core.Qt.WindowType.WindowTransparentForInput
        visibility_getter = getattr(window, "isVisible", None)
        was_visible = None
        if callable(visibility_getter):
            was_visible = bool(visibility_getter())
        position: tuple[int, int] | None = None
        position_getter = getattr(window, "pos", None)
        if not callable(position_getter):
            position_getter = getattr(window, "position", None)
        if callable(position_getter):
            try:
                point = position_getter()
                position = (int(point.x()), int(point.y()))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                position = None
        # QWebEngineView.setWindowFlag() 会销毁并重建 Chromium 顶层原生
        # 表面。置顶确认正在等待 EWMH 回读时若这里更换 XID，旧事务将
        # 永远无法收敛。优先在承载 QWidget 的 QWindow 上原地切换输入
        # 透明旗标；只有尚未创建原生句柄时才退回 QWidget 路径。
        flag_target = _qt_topmost_target(window)
        setter = getattr(flag_target, "setFlag", None)
        if not callable(setter):
            setter = getattr(flag_target, "setWindowFlag", None)
        if not callable(setter):
            raise AttributeError("window does not expose a Qt flag setter")
        setter(flag, bool(enabled))
        # QWidget.setWindowFlag() 重建顶层原生窗口时会隐藏可见控件；先恢复
        # 可见性再报告标志状态，否则 WebEngine 桌宠每次切换都会消失。
        if (
            flag_target is window
            and was_visible
            and callable(visibility_getter)
            and not bool(visibility_getter())
        ):
            shower = getattr(window, "show", None)
            if not callable(shower):
                raise RuntimeError("window was hidden while applying the Qt flag")
            shower()
            if not bool(visibility_getter()):
                raise RuntimeError("window remained hidden after applying the Qt flag")

        def restore_position() -> None:
            if position is None:
                return
            try:
                _move_window(window, *position, qt_core=qt_core)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return

        restore_position()
        timer_class = getattr(qt_core, "QTimer", None)
        single_shot = getattr(timer_class, "singleShot", None) if timer_class else None
        if callable(single_shot):
            try:
                single_shot(0, restore_position)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        flags_getter = getattr(flag_target, "flags", None)
        if not callable(flags_getter):
            flags_getter = getattr(flag_target, "windowFlags", None)
        if not callable(flags_getter):
            raise AttributeError("window does not expose Qt flags")
        actual_flags = flags_getter()
        active = bool(actual_flags & flag)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Qt click-through flag update failed: %s", type(exc).__name__)
        return _capability(
            "click_through",
            CapabilityState.UNAVAILABLE,
            "Qt input transparency flag update failed",
        )
    if active != bool(enabled):
        return _capability(
            "click_through",
            CapabilityState.UNAVAILABLE,
            "window system did not retain the requested Qt flag",
            "Qt.WindowType.WindowTransparentForInput",
        )
    if resolved_backend == BACKEND_WAYLAND:
        return _capability(
            "click_through",
            CapabilityState.DEGRADED,
            "Qt input transparency flag applied; Wayland compositor behavior is not guaranteed",
            "Qt.WindowType.WindowTransparentForInput",
        )
    return _capability(
        "click_through",
        CapabilityState.AVAILABLE,
        "Qt input transparency flag applied",
        "Qt.WindowType.WindowTransparentForInput",
    )


def _x11_shape_device_scale(window: object) -> float:
    """读取 Qt 窗口的设备倍率并归一化为可用于 X11 的正数。"""

    getter = getattr(window, "devicePixelRatioF", None)
    if not callable(getter):
        getter = getattr(window, "devicePixelRatio", None)
    try:
        scale = float(getter()) if callable(getter) else 1.0
    except (TypeError, ValueError, ArithmeticError):
        scale = 1.0
    # Qt 的逻辑坐标需要转换为 X11 的设备像素。低于 1 的异常替身值
    # 不应把区域缩成不可见；真实 Qt DPR 通常为 1 或更高。
    if not (scale > 0.0) or scale != scale or scale in {float("inf"), float("-inf")}:
        return 1.0
    return max(1.0, scale)


def _x11_shape_rectangles(
    region: object | None,
    *,
    window: object,
    device_scale: float | None = None,
) -> list[tuple[int, int, int, int]]:
    """将 Qt ``QRegion`` 转换为 X11 Shape 的设备像素矩形。"""

    scale = _x11_shape_device_scale(window) if device_scale is None else float(device_scale)
    if not (scale > 0.0) or scale != scale or scale in {float("inf"), float("-inf")}:
        scale = 1.0
    scale = max(1.0, scale)
    if region is None:
        width_getter = getattr(window, "width", None)
        height_getter = getattr(window, "height", None)
        width = int(width_getter()) if callable(width_getter) else 1
        height = int(height_getter()) if callable(height_getter) else 1
        return [(0, 0, max(1, round(width * scale)), max(1, round(height * scale)))]
    try:
        items = tuple(region)
    except (TypeError, ValueError):
        items = ()
    rectangles: list[tuple[int, int, int, int]] = []
    for item in items:
        try:
            x = round(int(item.x()) * scale)
            y = round(int(item.y()) * scale)
            width = round(int(item.width()) * scale)
            height = round(int(item.height()) * scale)
        except (AttributeError, TypeError, ValueError, ArithmeticError):
            continue
        if width > 0 and height > 0:
            rectangles.append((x, y, width, height))
    if not rectangles:
        # XShape 的空矩形列表代表恢复默认区域，而不是禁止输入；窗口外
        # 的 1px 区域才表示真正的空 Input Shape。
        return [(-1, -1, 1, 1)]
    return rectangles


def set_window_input_shape(
    window: object,
    region: object | None,
    *,
    backend: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> PlatformCapability:
    """在 X11 上写入独立的原生 Input Shape。

    ``region`` 使用 Qt 逻辑坐标；设置 ``None`` 会恢复整窗输入区域，空区域
    会提交窗口外的哨兵矩形。Wayland/未知后端明确返回降级或不可用，不伪造
    分离成功。
    """

    values = dict(os.environ if environ is None else environ)
    resolved_backend = (backend or detect_linux_backend(values)).strip().lower()
    if resolved_backend == BACKEND_WAYLAND:
        return _capability(
            "input_shape",
            CapabilityState.DEGRADED,
            "Wayland 不提供原生 Input Shape；仅保留 Qt 视觉遮罩",
            "XShapeInput",
        )
    if resolved_backend != BACKEND_X11:
        return _capability(
            "input_shape",
            CapabilityState.UNAVAILABLE,
            "原生 Input Shape 需要 X11",
            "XShapeInput",
        )
    display_name = str(values.get("DISPLAY", "") or "").strip()
    if not display_name:
        return _capability(
            "input_shape",
            CapabilityState.UNAVAILABLE,
            "DISPLAY 未配置，无法写入 X11 Input Shape",
            "XShapeInput",
        )
    try:
        display_module = _import_optional("Xlib.display")
        error_module = _import_optional("Xlib.error")
        shape_module = _import_optional("Xlib.ext.shape")
        display_class = getattr(display_module, "Display", None)
        catch_error = getattr(error_module, "CatchError", None)
        bad_window = getattr(error_module, "BadWindow", None)
        shape_set = getattr(getattr(shape_module, "SO", None), "Set", None)
        shape_input = getattr(getattr(shape_module, "SK", None), "Input", None)
        if not callable(display_class) or not callable(catch_error) or bad_window is None:
            raise ImportError("python-xlib Shape support is unavailable")
        if shape_set is None or shape_input is None:
            raise ImportError("XShapeInput constants are unavailable")

        win_id = int(window.winId())  # type: ignore[attr-defined]
        if win_id <= 0:
            raise ValueError("window id is invalid")
        connection = display_class(display_name)
        try:
            # Qt 切换窗口标志时可能异步重建原生窗口；BadWindow 只在本次
            # 请求内收敛为不可用，避免 Xlib 把竞态 traceback 打到 stderr。
            x11_errors = catch_error(bad_window)
            connection.set_error_handler(x11_errors)
            native_window = connection.create_resource_object("window", win_id)
            rectangles = _x11_shape_rectangles(region, window=window)
            native_window.shape_rectangles(shape_set, shape_input, 0, 0, 0, rectangles)
            connection.sync()
            if x11_errors.get_error() is not None:
                return _capability(
                    "input_shape",
                    CapabilityState.UNAVAILABLE,
                    "X11 窗口已失效，Input Shape 未应用",
                    "XShapeInput",
                )
            return _capability(
                "input_shape",
                CapabilityState.AVAILABLE,
                "X11 Input Shape 已应用",
                "XShapeInput",
            )
        finally:
            try:
                connection.close()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("X11 input Shape update failed: %s", type(exc).__name__)
        return _capability(
            "input_shape",
            CapabilityState.UNAVAILABLE,
            "X11 Input Shape 暂不可用",
            "XShapeInput",
        )


def _linux_automation_key(value: object) -> tuple[str, bool] | None:
    """解析 X11 自动化按键，返回按键名称和是否需要 Shift。"""

    if not isinstance(value, str):
        return None
    key = value.strip()
    if not key:
        return None
    names = {
        "backspace": "BackSpace",
        "tab": "Tab",
        "enter": "Return",
        "shift": "Shift_L",
        "ctrl": "Control_L",
        "alt": "Alt_L",
        "escape": "Escape",
        "space": "space",
        "page_up": "Prior",
        "page_down": "Next",
        "end": "End",
        "home": "Home",
        "left": "Left",
        "up": "Up",
        "right": "Right",
        "down": "Down",
        "insert": "Insert",
        "delete": "Delete",
        "win": "Super_L",
    }
    normalized = key.lower()
    if normalized in names:
        return (names[normalized], False)
    if len(key) == 1 and key.isascii() and key.isprintable():
        shifted = {
            "!": "1",
            "@": "2",
            "#": "3",
            "$": "4",
            "%": "5",
            "^": "6",
            "&": "7",
            "*": "8",
            "(": "9",
            ")": "0",
            "_": "-",
            "+": "=",
            "{": "[",
            "}": "]",
            "|": "\\",
            ":": ";",
            '"': "'",
            "<": ",",
            ">": ".",
            "?": "/",
        }
        if key in shifted:
            return (shifted[key], True)
        return (key.lower(), key.isalpha() and key.isupper())
    if normalized.startswith("f") and normalized[1:].isdigit():
        number = int(normalized[1:])
        if 1 <= number <= 12:
            return (f"F{number}", False)
    return None


def _linux_automation_failure(
    reason: str, *, completed: list[dict[str, object]]
) -> dict[str, object]:
    """返回不含输入正文的 Linux 自动化失败结果。"""

    return {
        "status": "partial" if completed else "unavailable",
        "backend": BACKEND_X11,
        "reason": reason,
        "completed": tuple(completed),
    }


def _linux_automation_window_id(value: object) -> int | None:
    """按自动化契约解析 X11 窗口 ID。"""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if not isinstance(value, str) or not value.startswith("0x"):
        return None
    try:
        parsed = int(value, 16)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 else None


def _linux_automation_integer(value: object) -> int | None:
    """读取严格 JSON 整数，拒绝布尔值和隐式转换。"""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _linux_automation_keycode(
    display: object, keysym_module: object, value: object
) -> tuple[int, bool] | None:
    """把白名单按键解析为 X11 keycode。"""

    parsed = _linux_automation_key(value)
    if parsed is None:
        return None
    key_name, requires_shift = parsed
    string_to_keysym = getattr(keysym_module, "string_to_keysym", None)
    converter = getattr(display, "keysym_to_keycode", None)
    if not callable(string_to_keysym) or not callable(converter):
        return None
    keysym = string_to_keysym(str(key_name))
    if not keysym and len(str(key_name)) == 1:
        keysym = ord(str(key_name))
    if not keysym:
        return None
    try:
        keycode = int(converter(keysym))
    except (TypeError, ValueError, OverflowError):
        return None
    return (keycode, requires_shift) if keycode > 0 else None


def _linux_automation_send_key(
    display: object,
    xtest_module: object,
    x_module: object,
    keysym_module: object,
    value: object,
    modifiers: Sequence[object],
    repeat: int,
    modifier_names: Mapping[str, object],
) -> bool:
    """通过 XTEST 发送一个有界按键序列。"""

    key = _linux_automation_keycode(display, keysym_module, value)
    if key is None or repeat < 1 or repeat > 5:
        return False
    if isinstance(modifiers, (str, bytes, bytearray)) or len(modifiers) > 3:
        return False
    modifier_codes: list[int] = []
    for item in modifiers:
        name = str(item).strip().lower()
        if name not in modifier_names:
            return False
        parsed = _linux_automation_keycode(display, keysym_module, name)
        if parsed is None or parsed[0] in modifier_codes:
            return False
        modifier_codes.append(parsed[0])
    keycode, requires_shift = key
    if requires_shift:
        shift = _linux_automation_keycode(display, keysym_module, "shift")
        if shift is None or shift[0] in modifier_codes:
            return False
        modifier_codes.append(shift[0])
    fake_input = getattr(xtest_module, "fake_input", None)
    key_press = getattr(x_module, "KeyPress", None)
    key_release = getattr(x_module, "KeyRelease", None)
    if not callable(fake_input) or key_press is None or key_release is None:
        return False
    try:
        for modifier in modifier_codes:
            fake_input(display, key_press, modifier)
        for _ in range(repeat):
            fake_input(display, key_press, keycode)
            fake_input(display, key_release, keycode)
        for modifier in reversed(modifier_codes):
            fake_input(display, key_release, modifier)
        synchronizer = getattr(display, "sync", None)
        if not callable(synchronizer):
            return False
        synchronizer()
        return True
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _linux_automation_send_text(
    display: object,
    xtest_module: object,
    x_module: object,
    keysym_module: object,
    text: str,
    modifier_names: Mapping[str, object],
) -> bool:
    """通过 XTEST 发送可打印 ASCII 按键文本。"""

    if (
        not text
        or len(text) > 512
        or any(
            ord(character) < 32 or ord(character) == 127 or not character.isascii()
            for character in text
        )
    ):
        return False
    for character in text:
        if not _linux_automation_send_key(
            display,
            xtest_module,
            x_module,
            keysym_module,
            character,
            (),
            1,
            modifier_names,
        ):
            return False
    return True


class LinuxDesktopPlatform:
    """Linux 平台适配器门面。"""

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        include_command_line: bool = False,
    ) -> None:
        self._environment = dict(os.environ if environ is None else environ)
        self._include_command_line = bool(include_command_line)
        self._snapshot: PlatformSnapshot | None = None
        self._excluded_window_ids: set[int] = set()
        self._excluded_window_ids_lock = RLock()

    @property
    def backend(self) -> str:
        return detect_linux_backend(self._environment)

    def probe(self) -> PlatformSnapshot:
        self._snapshot = probe_linux_platform(self._environment)
        return self._snapshot

    def active_window(self) -> WindowContext | None:
        if self.backend != BACKEND_X11:
            return None
        # Web/原生渲染器可能在 Qt 主线程注册新窗口，而 watcher 在后台线程
        # 构造 provider；先复制快照，避免 provider 遍历可变集合时发生竞态。
        with self._excluded_window_ids_lock:
            excluded_window_ids = tuple(self._excluded_window_ids)
        return X11WindowContextProvider(
            self._environment,
            include_command_line=self._include_command_line,
            excluded_window_ids=excluded_window_ids,
        ).read()

    def system_idle_seconds(self) -> float | None:
        """读取可选 X11 系统空闲时长；Wayland 或异常时返回 ``None``。"""

        return probe_x11_idle_seconds(self._environment)

    def cursor_position(self) -> Mapping[str, object]:
        """读取全局光标位置；Wayland 或 X11 失败时明确返回不可用。"""

        return _read_x11_cursor_position(self._environment)

    get_active_window = active_window

    def foreground_window(self) -> Mapping[str, object]:
        """把窗口契约转换为不会暴露额外对象的工具结果。"""

        window = self.active_window()
        if window is None:
            return {
                "status": "unavailable",
                "backend": self.backend,
                "reason": "active window is unavailable",
            }
        return {
            "status": "available",
            "backend": window.backend,
            "window_id": window.window_id,
            "title": window.title,
            "app_id": window.app_id,
            "pid": window.pid,
            "process_name": window.process_name,
            "executable": window.executable,
            "command_line": list(window.command_line),
            "geometry": list(window.geometry) if window.geometry else None,
            "source": window.source,
            "redacted": window.redacted,
        }

    def resolve_capture_window_id(self) -> int | None:
        """在 Qt 调度边界外读取窗口 ID，供窗口截图的主线程阶段使用。"""

        window = self.active_window()
        if window is None or not window.window_id:
            return None
        try:
            return int(window.window_id, 16)
        except (TypeError, ValueError):
            return None

    def list_processes(self, limit: int = 20) -> Mapping[str, object]:
        """列出可读取的进程摘要，不返回环境变量或完整命令行。"""

        if isinstance(limit, bool) or not isinstance(limit, int):
            return {
                "status": "unavailable",
                "processes": [],
                "reason": "process limit is invalid",
            }
        safe_limit = max(1, min(limit, 100))
        psutil = _import_optional("psutil")
        if psutil is None:
            return {
                "status": "unavailable",
                "processes": [],
                "reason": "psutil is unavailable",
            }
        processes: list[dict[str, object]] = []
        current_uid = getattr(os, "getuid", lambda: None)()
        current_user = str(os.environ.get("USER") or "")
        try:
            iterator = psutil.process_iter(("pid", "name", "exe", "username"))
            for process in iterator:
                if len(processes) >= safe_limit:
                    break
                try:
                    info = process.info
                    username = str(info.get("username") or "")
                    if current_uid is not None:
                        try:
                            process_uids = process.uids()
                            known_uids = {
                                int(getattr(process_uids, name))
                                for name in ("real", "effective", "saved")
                                if getattr(process_uids, name, None) is not None
                            }
                        except (AttributeError, OSError, PermissionError, TypeError, ValueError):
                            known_uids = set()
                        if known_uids:
                            if int(current_uid) not in known_uids:
                                continue
                        elif not current_user or not username or username != current_user:
                            continue
                    elif not current_user or not username or username != current_user:
                        continue
                    processes.append(
                        {
                            "pid": int(info.get("pid") or 0),
                            "name": str(info.get("name") or ""),
                            "executable": str(info.get("exe") or ""),
                        }
                    )
                except (AttributeError, KeyError, PermissionError, OSError, ValueError):
                    continue
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {
                "status": "unavailable",
                "processes": [],
                "reason": "process enumeration failed",
            }
        return {"status": "available", "processes": processes}

    def capture_screen(
        self,
        *,
        scope: str = "screen",
        region: Mapping[str, object] | None = None,
        window_id: int | None = None,
    ) -> Mapping[str, object]:
        """通过 Qt 屏幕抓取提供有界 PNG；Wayland 明确返回不可用。"""

        normalized_scope = str(scope or "screen").strip().lower()
        if normalized_scope not in {"screen", "window", "region"}:
            return {"status": "unavailable", "reason": "unsupported capture scope"}
        if self.backend == BACKEND_WAYLAND:
            return {
                "status": "unavailable",
                "reason": "native Wayland screen capture requires a portal adapter",
            }
        if normalized_scope == "window" and window_id is None:
            return {
                "status": "unavailable",
                "reason": "window id must be resolved before Qt screen capture",
            }
        qt_gui = _import_optional("PySide6.QtGui")
        qt_core = _import_optional("PySide6.QtCore")
        if qt_gui is None or qt_core is None:
            return {"status": "unavailable", "reason": "PySide6 is unavailable"}
        try:
            application = getattr(qt_gui, "QGuiApplication").instance()
            if application is None:
                return {"status": "unavailable", "reason": "QGuiApplication is not running"}
            screen = application.primaryScreen()
            if screen is None:
                return {"status": "unavailable", "reason": "primary screen is unavailable"}
            if normalized_scope == "window":
                if isinstance(window_id, bool) or not isinstance(window_id, int):
                    return {
                        "status": "unavailable",
                        "reason": "active window id is invalid",
                    }
                safe_window_id = int(window_id)
                if safe_window_id <= 0:
                    return {
                        "status": "unavailable",
                        "reason": "active window id is invalid",
                    }
                screen = _screen_for_window_id(application, screen, safe_window_id)
                image = screen.grabWindow(safe_window_id)
                window_geometry = _x11_window_geometry(self._environment, safe_window_id)
            elif normalized_scope == "region":
                rect = _capture_region(region)
                selected = _screen_for_capture_region(application, screen, rect)
                if selected is None:
                    return {
                        "status": "unavailable",
                        "reason": "capture region must fit within a single screen",
                    }
                target_screen, local_x, local_y = selected
                # QScreen.grabWindow 的完整重载直接在目标屏幕上截取局部区域；
                # 测试替身及少数旧绑定只实现单参数重载时，回退到同一屏幕
                # 的图像裁剪，保持虚拟桌面坐标到局部坐标的一致性。
                try:
                    image = target_screen.grabWindow(0, local_x, local_y, rect[2], rect[3])
                except TypeError:
                    image = target_screen.grabWindow(0).copy(local_x, local_y, rect[2], rect[3])
            else:
                image = screen.grabWindow(0)
                window_geometry = None
            image_width = int(image.width())
            image_height = int(image.height())
            if image_width <= 0 or image_height <= 0:
                return {"status": "unavailable", "reason": "captured image is empty"}
            buffer = qt_core.QBuffer()
            buffer.open(qt_core.QIODevice.OpenModeFlag.WriteOnly)
            if not image.save(buffer, "PNG"):
                return {"status": "unavailable", "reason": "PNG encoding failed"}
            data = bytes(buffer.data())
            if not data:
                return {"status": "unavailable", "reason": "PNG encoding produced no data"}
            if len(data) > 5 * 1024 * 1024:
                return {"status": "unavailable", "reason": "capture exceeds size limit"}
            result: dict[str, object] = {
                "status": "available",
                "scope": normalized_scope,
                "format": "png",
                "data": data,
                "width": image_width,
                "height": image_height,
            }
            if normalized_scope == "window" and window_geometry is not None:
                result["origin"] = {
                    "x": window_geometry[0],
                    "y": window_geometry[1],
                }
            return result
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "screen capture failed"}

    def click_at(self, x: int, y: int, *, button: str = "left") -> Mapping[str, object]:
        """通过 XTEST 执行一次有界全局坐标点击。

        Wayland 和缺少 XTEST 的会话明确返回不可用；实现不调用 shell，也不
        接收任意事件编号，避免工具参数越过平台输入边界。
        """

        if self.backend != BACKEND_X11:
            return {
                "status": "unavailable",
                "reason": "global coordinate input is only available through X11 XTEST",
            }
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int)
            or not isinstance(y, int)
        ):
            return {"status": "unavailable", "reason": "click coordinates are invalid"}
        target_x = int(x)
        target_y = int(y)
        if abs(target_x) > 1_000_000 or abs(target_y) > 1_000_000:
            return {"status": "unavailable", "reason": "click coordinates are outside bounds"}
        if not isinstance(button, str):
            return {"status": "unavailable", "reason": "click button is unsupported"}
        normalized_button = button.strip().lower()
        button_codes = {"left": 1, "middle": 2, "right": 3}
        button_code = button_codes.get(normalized_button)
        if button_code is None:
            return {"status": "unavailable", "reason": "click button is unsupported"}

        display_module = _import_optional("Xlib.display")
        x_module = _import_optional("Xlib.X")
        xtest_module = _import_optional("Xlib.ext.xtest")
        display_class = getattr(display_module, "Display", None)
        if not callable(display_class) or x_module is None or xtest_module is None:
            return {"status": "unavailable", "reason": "python-xlib XTEST is unavailable"}
        display_name = str(self._environment.get("DISPLAY", "") or "").strip()
        if not display_name:
            return {"status": "unavailable", "reason": "DISPLAY is unavailable"}
        display = None
        try:
            display = display_class(display_name)
            has_extension = getattr(display, "has_extension", None)
            if not callable(has_extension) or not has_extension("XTEST"):
                return {"status": "unavailable", "reason": "XTEST extension is unavailable"}
            inject = getattr(display, "xtest_fake_input", None)
            if not callable(inject):
                return {"status": "unavailable", "reason": "XTEST input method is unavailable"}
            inject(getattr(x_module, "MotionNotify"), x=target_x, y=target_y)
            inject(getattr(x_module, "ButtonPress"), detail=button_code)
            inject(getattr(x_module, "ButtonRelease"), detail=button_code)
            display.sync()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "XTEST click failed"}
        finally:
            closer = getattr(display, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass
        return {
            "status": "completed",
            "backend": BACKEND_X11,
            "x": target_x,
            "y": target_y,
            "button": normalized_button,
        }

    def automation_batch(self, steps: object) -> Mapping[str, object]:
        """在 X11/XTEST 上执行有限的跨应用自动化步骤。"""

        if self.backend != BACKEND_X11:
            return {
                "status": "unavailable",
                "backend": self.backend,
                "reason": "desktop automation requires an X11 XTEST session",
            }
        if isinstance(steps, (str, bytes, bytearray)) or not isinstance(steps, (list, tuple)):
            return {
                "status": "unavailable",
                "backend": BACKEND_X11,
                "reason": "automation steps are invalid",
            }
        if not steps or len(steps) > 16:
            return {
                "status": "unavailable",
                "backend": BACKEND_X11,
                "reason": "automation step count is out of range",
            }

        display_module = _import_optional("Xlib.display")
        x_module = _import_optional("Xlib.X")
        xtest_module = _import_optional("Xlib.ext.xtest")
        keysym_module = _import_optional("Xlib.XK")
        display_class = getattr(display_module, "Display", None)
        fake_input = getattr(xtest_module, "fake_input", None) if xtest_module else None
        if (
            not callable(display_class)
            or x_module is None
            or xtest_module is None
            or not callable(fake_input)
            or keysym_module is None
        ):
            return {
                "status": "unavailable",
                "backend": BACKEND_X11,
                "reason": "python-xlib XTEST automation is unavailable",
            }

        display = None
        completed: list[dict[str, object]] = []
        total_wait_ms = 0
        modifier_names = {"ctrl": True, "shift": True, "alt": True, "win": True}
        try:
            display_name = str(self._environment.get("DISPLAY", "") or "").strip()
            if not display_name:
                return {
                    "status": "unavailable",
                    "backend": BACKEND_X11,
                    "reason": "DISPLAY is unavailable",
                }
            display = display_class(display_name)
            has_extension = getattr(display, "has_extension", None)
            if not callable(has_extension) or not has_extension("XTEST"):
                return {
                    "status": "unavailable",
                    "backend": BACKEND_X11,
                    "reason": "XTEST extension is unavailable",
                }

            for index, raw_step in enumerate(steps):
                if not isinstance(raw_step, Mapping):
                    return _linux_automation_failure(
                        "automation step is invalid", completed=completed
                    )
                step_type = str(raw_step.get("type", "")).strip().lower()

                if step_type == "wait":
                    duration = _linux_automation_integer(raw_step.get("duration_ms"))
                    if duration is None or duration < 0 or duration > 2_000:
                        return _linux_automation_failure(
                            "automation wait is out of range", completed=completed
                        )
                    total_wait_ms += duration
                    if total_wait_ms > 5_000:
                        return _linux_automation_failure(
                            "automation total wait is out of range", completed=completed
                        )
                    if duration:
                        sleep(duration / 1000.0)
                    completed.append({"index": index, "type": step_type, "status": "completed"})
                    continue

                if step_type == "key":
                    repeat = _linux_automation_integer(raw_step.get("repeat", 1))
                    modifiers = raw_step.get("modifiers", ())
                    if repeat is None or not isinstance(modifiers, (list, tuple)):
                        return _linux_automation_failure(
                            "automation key is invalid", completed=completed
                        )
                    if not _linux_automation_send_key(
                        display,
                        xtest_module,
                        x_module,
                        keysym_module,
                        raw_step.get("key"),
                        modifiers,
                        repeat,
                        modifier_names,
                    ):
                        return _linux_automation_failure(
                            "X11 key input failed", completed=completed
                        )
                    completed.append(
                        {
                            "index": index,
                            "type": step_type,
                            "status": "completed",
                            "repeat": repeat,
                        }
                    )
                    continue

                if step_type == "text":
                    text = raw_step.get("text")
                    if not isinstance(text, str) or not _linux_automation_send_text(
                        display,
                        xtest_module,
                        x_module,
                        keysym_module,
                        text,
                        modifier_names,
                    ):
                        return _linux_automation_failure(
                            "X11 text input requires printable ASCII", completed=completed
                        )
                    completed.append(
                        {
                            "index": index,
                            "type": step_type,
                            "status": "completed",
                            "characters": len(text),
                        }
                    )
                    continue

                if step_type in {"move_pointer", "click"}:
                    x = _linux_automation_integer(raw_step.get("x"))
                    y = _linux_automation_integer(raw_step.get("y"))
                    if x is None or y is None or abs(x) > 1_000_000 or abs(y) > 1_000_000:
                        return _linux_automation_failure(
                            "automation coordinates are invalid", completed=completed
                        )
                    if step_type == "move_pointer":
                        fake_input(
                            display,
                            getattr(x_module, "MotionNotify"),
                            x=x,
                            y=y,
                        )
                    else:
                        button = str(raw_step.get("button", "left") or "").strip().lower()
                        button_code = {"left": 1, "middle": 2, "right": 3}.get(button)
                        if button_code is None:
                            return _linux_automation_failure(
                                "automation click is invalid", completed=completed
                            )
                        fake_input(
                            display,
                            getattr(x_module, "MotionNotify"),
                            x=x,
                            y=y,
                        )
                        fake_input(
                            display,
                            getattr(x_module, "ButtonPress"),
                            detail=button_code,
                        )
                        fake_input(
                            display,
                            getattr(x_module, "ButtonRelease"),
                            detail=button_code,
                        )
                    display.sync()
                    completed.append({"index": index, "type": step_type, "status": "completed"})
                    continue

                if step_type in {"activate_window", "move_window"}:
                    window_id = _linux_automation_window_id(raw_step.get("window_id"))
                    if window_id is None:
                        return _linux_automation_failure(
                            "automation window id is invalid", completed=completed
                        )
                    window = display.create_resource_object("window", window_id)
                    if step_type == "activate_window":
                        setter = getattr(window, "set_input_focus", None)
                        if not callable(setter):
                            return _linux_automation_failure(
                                "X11 window activation is unavailable", completed=completed
                            )
                        setter(
                            getattr(x_module, "RevertToParent"),
                            getattr(x_module, "CurrentTime"),
                        )
                    else:
                        x = _linux_automation_integer(raw_step.get("x"))
                        y = _linux_automation_integer(raw_step.get("y"))
                        if x is None or y is None or abs(x) > 1_000_000 or abs(y) > 1_000_000:
                            return _linux_automation_failure(
                                "automation window position is invalid", completed=completed
                            )
                        configure = getattr(window, "configure", None)
                        if not callable(configure):
                            return _linux_automation_failure(
                                "X11 window movement is unavailable", completed=completed
                            )
                        configure(x=x, y=y)
                    display.sync()
                    completed.append({"index": index, "type": step_type, "status": "completed"})
                    continue

                return _linux_automation_failure(
                    "automation step type is unsupported", completed=completed
                )
            return {
                "status": "completed",
                "backend": BACKEND_X11,
                "steps": tuple(completed),
            }
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("X11 automation failed: %s", type(exc).__name__)
            return _linux_automation_failure("X11 automation failed", completed=completed)
        finally:
            closer = getattr(display, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    def exclude_window_id(self, window_id: int) -> None:
        """排除属于桌宠自身的 X11 窗口。"""

        if isinstance(window_id, bool):
            return
        try:
            value = int(window_id)
        except (TypeError, ValueError, OverflowError):
            return
        if value <= 0:
            return
        with self._excluded_window_ids_lock:
            self._excluded_window_ids.add(value)

    def set_click_through(self, window: object, enabled: bool) -> PlatformCapability:
        return set_window_click_through(window, enabled, backend=self.backend)

    def set_input_shape(self, window: object, region: object | None) -> PlatformCapability:
        """为原生窗口设置独立 Input Shape；视觉遮罩由 Qt ``setMask`` 保留。"""

        return set_window_input_shape(
            window,
            region,
            backend=self.backend,
            environ=self._environment,
        )

    def always_on_top_status(self, window: object) -> Mapping[str, object]:
        """回读 Qt 原生旗标，并在 X11/XWayland 上以 EWMH 为最终依据。"""

        qt_enabled = _qt_topmost_flag(window)
        if self.backend == BACKEND_WAYLAND:
            if qt_enabled is None:
                return {
                    "status": "unavailable",
                    "enabled": False,
                    "confirmed": False,
                    "detail": "无法回读 Wayland 原生窗口置顶标志",
                }
            return {
                "status": "degraded",
                "enabled": qt_enabled,
                "confirmed": False,
                "detail": "Wayland 仅能回读客户端旗标，最终层级由合成器策略决定",
            }
        if self.backend != BACKEND_X11:
            return {
                "status": "unavailable",
                "enabled": bool(qt_enabled),
                "confirmed": False,
                "detail": "当前 Linux 显示后端不支持验证窗口置顶",
            }
        native_enabled, evidence = _read_x11_topmost_state(window, self._environment)
        if native_enabled is None:
            if qt_enabled is None:
                return {
                    "status": "unavailable",
                    "enabled": False,
                    "confirmed": False,
                    "detail": "无法回读原生窗口或 EWMH 置顶属性",
                }
            return {
                "status": "degraded",
                "enabled": qt_enabled,
                "confirmed": False,
                "detail": "Qt 置顶旗标可读，但窗口管理器 EWMH 状态不可读",
            }
        detail = "窗口管理器已确认置顶" if native_enabled else "窗口管理器确认当前窗口未置顶"
        result: dict[str, object] = {
            "status": "available",
            "enabled": native_enabled,
            "confirmed": True,
            "detail": detail,
        }
        if evidence:
            result["evidence"] = evidence
        return result

    def move_overlay(self, window: object, x: int, y: int) -> PlatformCapability:
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int)
            or not isinstance(y, int)
        ):
            return _capability(
                "overlay_position",
                CapabilityState.UNAVAILABLE,
                "Linux position coordinates are invalid",
            )
        if self.backend == BACKEND_WAYLAND:
            # Wayland 不承诺客户端可定位顶层窗口，但 Qt 仍可能接受一次
            # best-effort 请求（例如特定 compositor 或 XWayland 兼容层）。
            # 请求结果必须标记为 degraded，并在可读取几何时核对是否真的生效，
            # 不能把 compositor 忽略请求误报为 available。
            try:
                target_x = x
                target_y = y
                _move_window(window, target_x, target_y)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Wayland position request failed: %s", type(exc).__name__)
                return _capability(
                    "overlay_position",
                    CapabilityState.UNAVAILABLE,
                    "Wayland position request failed",
                )
            getter = getattr(window, "position", None)
            if not callable(getter):
                getter = getattr(window, "pos", None)
            observed = None
            if callable(getter):
                try:
                    point = getter()
                    observed = (int(point.x()), int(point.y()))
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    logger.debug("Wayland position readback failed: %s", type(exc).__name__)
            if observed == (target_x, target_y):
                detail = (
                    "Qt window geometry reflects the requested position; "
                    "Wayland compositor controls final placement"
                )
            elif observed is None:
                detail = "Wayland position request submitted; compositor controls final placement"
            else:
                detail = (
                    "Qt window geometry did not reflect the requested position "
                    f"(observed {observed[0]},{observed[1]}); "
                    "Wayland compositor controls final placement"
                )
            return _capability(
                "overlay_position",
                CapabilityState.DEGRADED,
                detail,
                "QWindow.setPosition",
            )
        if self.backend != BACKEND_X11:
            return _capability(
                "overlay_position",
                CapabilityState.UNAVAILABLE,
                "X11 is unavailable",
            )
        try:
            target_x = x
            target_y = y
            _move_window(window, target_x, target_y)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("X11 position request failed: %s", type(exc).__name__)
            return _capability(
                "overlay_position",
                CapabilityState.UNAVAILABLE,
                "X11 position request failed",
            )
        getter = getattr(window, "position", None)
        if not callable(getter):
            getter = getattr(window, "pos", None)
        if callable(getter):
            try:
                point = getter()
                observed = (int(point.x()), int(point.y()))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("X11 position readback failed: %s", type(exc).__name__)
            else:
                if observed != (target_x, target_y):
                    return _capability(
                        "overlay_position",
                        CapabilityState.UNAVAILABLE,
                        "X11 window geometry did not retain the requested position",
                        "QWidget.move/QWindow.setPosition",
                    )
        return _capability(
            "overlay_position",
            CapabilityState.AVAILABLE,
            "position request submitted",
        )


__all__ = [
    "BACKEND_UNKNOWN",
    "BACKEND_WAYLAND",
    "BACKEND_X11",
    "LinuxDesktopPlatform",
    "X11WindowContextProvider",
    "detect_linux_backend",
    "probe_linux_platform",
    "probe_x11_input_control",
    "probe_x11_idle_seconds",
    "probe_x11_compositor",
    "set_window_click_through",
    "set_window_input_shape",
]
