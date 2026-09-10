"""Windows 桌面能力适配器。

该模块只在调用 Windows 能力时加载 Win32 动态库，因此 Linux/Wayland
环境可以安全导入平台工厂并继续使用精灵或 Web Live2D 回退。
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import math
import os
from collections.abc import Mapping
from ctypes import wintypes
from threading import RLock
from time import sleep, time
from typing import Any, cast

from .protocol import CapabilityState, PlatformCapability, PlatformSnapshot, WindowContext

BACKEND_WINDOWS = "windows"

_GWL_EXSTYLE = -20
_WS_EX_TOPMOST = 0x00000008
_WS_EX_TRANSPARENT = 0x00000020
_WS_EX_NOACTIVATE = 0x08000000
_WINDOW_INPUT_STYLE_BITS = _WS_EX_TRANSPARENT | _WS_EX_NOACTIVATE
_HWND_TOPMOST = -1
_HWND_NOTOPMOST = -2
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_FRAMECHANGED = 0x0020
_SWP_NOOWNERZORDER = 0x0200
_SWP_NOSENDCHANGING = 0x0400
_MOUSEEVENTF_LEFTDOWN = 0x0002
_MOUSEEVENTF_LEFTUP = 0x0004
_MOUSEEVENTF_MIDDLEDOWN = 0x0020
_MOUSEEVENTF_MIDDLEUP = 0x0040
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP = 0x0010
_INPUT_MOUSE = 0
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_TH32CS_SNAPPROCESS = 0x00000002
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_MAX_PATH = 260
_PLATFORM_CAPABILITY_NAMES = (
    "active_window",
    "process_context",
    "click_through",
    "always_on_top",
    "global_hotkeys",
    "overlay_position",
    "screen_capture",
    "input_control",
    "system_idle",
    "cursor_position",
)

_ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

logger = logging.getLogger(__name__)

# 仅记录本适配器为输入透明所引入的样式位。关闭穿透时恢复基线，
# 不会误删宿主或 Qt 在窗口创建前已设置的扩展样式。
_native_input_style_lock = RLock()
_native_input_style_baselines: dict[int, int] = {}


class _Rect(ctypes.Structure):
    _fields_ = (
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    )


class _Point(ctypes.Structure):
    _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class _LastInputInfo(ctypes.Structure):
    _fields_ = (("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD))


class _ProcessEntry32W(ctypes.Structure):
    _fields_ = (
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", _ULONG_PTR),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * _MAX_PATH),
    )


class _MouseInput(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _KeyboardInput(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", _ULONG_PTR),
    )


class _InputUnion(ctypes.Union):
    _fields_ = (("mi", _MouseInput), ("ki", _KeyboardInput))


class _Input(ctypes.Structure):
    _anonymous_ = ("data",)
    _fields_ = (("type", wintypes.DWORD), ("data", _InputUnion))


_AUTOMATION_MAX_STEPS = 16
_AUTOMATION_MAX_TEXT_LENGTH = 512
_AUTOMATION_MAX_WAIT_MS = 2_000
_AUTOMATION_MAX_WAIT_TOTAL_MS = 5_000
_AUTOMATION_COORDINATE_LIMIT = 1_000_000
_AUTOMATION_VIRTUAL_KEYS = {
    "backspace": 0x08,
    "tab": 0x09,
    "enter": 0x0D,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "escape": 0x1B,
    "space": 0x20,
    "page_up": 0x21,
    "page_down": 0x22,
    "end": 0x23,
    "home": 0x24,
    "left": 0x25,
    "up": 0x26,
    "right": 0x27,
    "down": 0x28,
    "insert": 0x2D,
    "delete": 0x2E,
    "win": 0x5B,
    **{f"f{index}": 0x6F + index for index in range(1, 13)},
}
_AUTOMATION_MODIFIERS = {
    key: _AUTOMATION_VIRTUAL_KEYS[key] for key in ("ctrl", "shift", "alt", "win")
}


def _import_optional(module_name: str) -> Any | None:
    """动态导入可选依赖；缺失时返回 ``None``。"""

    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        return None


def _load_library(name: str) -> object | None:
    """按需加载 Win32 动态库；非 Windows 环境始终返回 ``None``。"""

    if os.name != "nt":
        return None
    loader = getattr(ctypes, "WinDLL", None)
    if not callable(loader):
        return None
    try:
        library = loader(name, use_last_error=True)
        _configure_win32_api(library, name)
        return library
    except (OSError, TypeError, ValueError):
        return None


def _configure_win32_api(api: object, library_name: str) -> None:
    """声明 Win32 ABI，避免 64 位句柄被 ctypes 默认 ``c_int`` 截断。"""

    if str(library_name or "").strip().lower() == "user32":
        specifications = {
            "GetForegroundWindow": ((), wintypes.HWND),
            "GetCursorPos": ((ctypes.POINTER(_Point),), wintypes.BOOL),
            "GetWindowTextLengthW": ((wintypes.HWND,), ctypes.c_int),
            "GetWindowTextW": (
                (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int),
                ctypes.c_int,
            ),
            "GetClassNameW": (
                (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int),
                ctypes.c_int,
            ),
            "GetWindowThreadProcessId": (
                (wintypes.HWND, wintypes.LPDWORD),
                wintypes.DWORD,
            ),
            "GetWindowRect": ((wintypes.HWND, wintypes.LPRECT), wintypes.BOOL),
            "GetLastInputInfo": (
                (ctypes.POINTER(_LastInputInfo),),
                wintypes.BOOL,
            ),
            "SetCursorPos": ((ctypes.c_int, ctypes.c_int), wintypes.BOOL),
            "SendInput": (
                (wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int),
                wintypes.UINT,
            ),
            "SetForegroundWindow": ((wintypes.HWND,), wintypes.BOOL),
            "mouse_event": (
                (wintypes.DWORD, ctypes.c_int, ctypes.c_int, wintypes.DWORD, _ULONG_PTR),
                None,
            ),
            "SetWindowPos": (
                (
                    wintypes.HWND,
                    wintypes.HWND,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    wintypes.UINT,
                ),
                wintypes.BOOL,
            ),
            "GetWindowLongPtrW": ((wintypes.HWND, ctypes.c_int), ctypes.c_ssize_t),
            "SetWindowLongPtrW": (
                (wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t),
                ctypes.c_ssize_t,
            ),
            "GetWindowLongW": ((wintypes.HWND, ctypes.c_int), ctypes.c_long),
            "SetWindowLongW": (
                (wintypes.HWND, ctypes.c_int, ctypes.c_long),
                ctypes.c_long,
            ),
        }
    elif str(library_name or "").strip().lower() == "kernel32":
        specifications = {
            "GetTickCount64": ((), ctypes.c_ulonglong),
            "CreateToolhelp32Snapshot": (
                (wintypes.DWORD, wintypes.DWORD),
                wintypes.HANDLE,
            ),
            "Process32FirstW": (
                (wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)),
                wintypes.BOOL,
            ),
            "Process32NextW": (
                (wintypes.HANDLE, ctypes.POINTER(_ProcessEntry32W)),
                wintypes.BOOL,
            ),
            "CloseHandle": ((wintypes.HANDLE,), wintypes.BOOL),
        }
    else:
        return
    for function_name, (argtypes, restype) in specifications.items():
        function = getattr(api, function_name, None)
        if not callable(function):
            continue
        try:
            function.argtypes = list(argtypes)
            function.restype = restype
        except (AttributeError, TypeError):
            # Python 测试替身的 bound method 不支持 ctypes 属性。
            continue


def _load_user32() -> object | None:
    return _load_library("user32")


def _load_kernel32() -> object | None:
    return _load_library("kernel32")


def _load_gdi32() -> object | None:
    """按需加载 GDI32；非 Windows 环境始终返回 ``None``。"""

    return _load_library("gdi32")


def _as_int(value: object) -> int:
    """读取普通整数或 ctypes 标量，失败时返回 0。"""

    raw = getattr(value, "value", value)
    try:
        return int(raw or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _automation_integer(value: object) -> int | None:
    """读取自动化契约中的 JSON 整数，拒绝布尔值和隐式类型转换。"""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _automation_key_code(value: object) -> int | None:
    """将白名单按键名归一化为 Win32 virtual-key，拒绝任意数值事件码。"""

    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in _AUTOMATION_VIRTUAL_KEYS:
        return _AUTOMATION_VIRTUAL_KEYS[key]
    if len(key) == 1 and "a" <= key <= "z":
        return ord(key.upper())
    if len(key) == 1 and "0" <= key <= "9":
        return ord(key)
    return None


def _automation_window_id(value: object) -> int | None:
    """按自动化工具契约解析正整数 HWND，支持十进制整数和 ``0x`` 字符串。"""

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


def _keyboard_input(virtual_key: int, *, flags: int = 0, scan_code: int = 0) -> _Input:
    return _Input(
        type=_INPUT_KEYBOARD,
        ki=_KeyboardInput(
            wVk=int(virtual_key),
            wScan=int(scan_code),
            dwFlags=int(flags),
            time=0,
            dwExtraInfo=0,
        ),
    )


def _mouse_input(flags: int) -> _Input:
    """构造一个无坐标偏移的 Win32 鼠标输入事件。"""

    return _Input(
        type=_INPUT_MOUSE,
        mi=_MouseInput(
            dx=0,
            dy=0,
            mouseData=0,
            dwFlags=int(flags),
            time=0,
            dwExtraInfo=0,
        ),
    )


def _unicode_input(code_unit: int, *, key_up: bool = False) -> _Input:
    return _keyboard_input(
        0,
        flags=_KEYEVENTF_UNICODE | (_KEYEVENTF_KEYUP if key_up else 0),
        scan_code=code_unit,
    )


def _send_inputs(user32: object | None, inputs: list[_Input]) -> bool:
    """原子提交有限输入事件；部分提交即失败，不将未提交事件误报为成功。"""

    sender = getattr(user32, "SendInput", None) if user32 is not None else None
    if not callable(sender) or not inputs:
        return False
    array_type = _Input * len(inputs)
    values = array_type(*inputs)
    try:
        sent = _as_int(sender(len(inputs), values, ctypes.sizeof(_Input)))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return False
    return sent == len(inputs)


def _automation_failure(reason: str, *, completed: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "unavailable",
        "backend": BACKEND_WINDOWS,
        "reason": reason,
        "completed_steps": tuple(completed),
    }


def _is_windows(value: bool | None = None) -> bool:
    """在真实系统或测试替身模式下判断是否允许调用 Windows 能力。"""

    return os.name == "nt" if value is None else bool(value)


def _capability(
    name: str,
    state: CapabilityState,
    detail: str,
    *evidence: str,
) -> PlatformCapability:
    return PlatformCapability(name=name, state=state, detail=detail, evidence=tuple(evidence))


def _unavailable_snapshot(*, is_windows: bool) -> PlatformSnapshot:
    """构造非 Windows 或 Win32 不可用时的统一能力快照。"""

    detail = "Windows platform is unavailable" if not is_windows else "Win32 APIs are unavailable"
    return PlatformSnapshot(
        backend=BACKEND_WINDOWS,
        capabilities=tuple(
            _capability(name, CapabilityState.UNAVAILABLE, detail)
            for name in _PLATFORM_CAPABILITY_NAMES
        ),
        details={"platform": "Windows", "is_windows": bool(is_windows)},
    )


def _qt_window_target(window: object) -> object:
    """解析承载顶层标志的 Qt 原生窗口。"""

    view = getattr(window, "view", None)
    source = view if view is not None else window
    getter = getattr(source, "windowHandle", None)
    if callable(getter):
        try:
            handle = getter()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            handle = None
        if handle is not None:
            return handle
    return source


def _window_id(window: object) -> int | None:
    """读取 QWidget/QWindow 的 Win32 HWND。"""

    if isinstance(window, bool):
        return None
    if isinstance(window, int):
        return window if window > 0 else None
    target = _qt_window_target(window)
    getter = getattr(target, "winId", None)
    if not callable(getter):
        return None
    try:
        value = _as_int(cast(Any, getter)())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _qt_flag(window: object, name: str) -> object | None:
    qt_core = _import_optional("PySide6.QtCore")
    qt = getattr(qt_core, "Qt", None) if qt_core is not None else None
    window_type = getattr(qt, "WindowType", qt)
    return getattr(window_type, name, None)


def _qt_flag_state(window: object, flag: object) -> bool | None:
    target = _qt_window_target(window)
    for getter_name in ("flags", "windowFlags"):
        getter = getattr(target, getter_name, None)
        if not callable(getter):
            continue
        try:
            return bool(getter() & flag)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return None


def _extended_style_accessors(user32: object | None) -> tuple[object | None, object | None]:
    """返回当前 Win32 架构可用的扩展窗口样式读写函数。"""

    if user32 is None:
        return None, None
    getter = getattr(user32, "GetWindowLongPtrW", None)
    setter = getattr(user32, "SetWindowLongPtrW", None)
    if not callable(getter):
        getter = getattr(user32, "GetWindowLongW", None)
    if not callable(setter):
        setter = getattr(user32, "SetWindowLongW", None)
    return getter if callable(getter) else None, setter if callable(setter) else None


def _read_native_extended_style(user32: object | None, hwnd: int) -> int | None:
    """读取有效 HWND 的扩展样式，无法回读时返回空。"""

    if hwnd <= 0:
        return None
    getter, _ = _extended_style_accessors(user32)
    if not callable(getter):
        return None
    try:
        return _as_int(getter(hwnd, _GWL_EXSTYLE))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _commit_native_extended_style(
    user32: object | None,
    hwnd: int,
    desired: int,
) -> int | None:
    """写入扩展样式并以回读作为唯一成功依据。"""

    if hwnd <= 0:
        return None
    _, setter = _extended_style_accessors(user32)
    if not callable(setter):
        return None
    try:
        setter(hwnd, _GWL_EXSTYLE, int(desired))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return None
    actual = _read_native_extended_style(user32, hwnd)
    if actual is None:
        return None
    reposition = getattr(user32, "SetWindowPos", None) if user32 is not None else None
    if callable(reposition):
        flags = (
            _SWP_NOMOVE
            | _SWP_NOSIZE
            | _SWP_NOZORDER
            | _SWP_NOACTIVATE
            | _SWP_NOOWNERZORDER
            | _SWP_FRAMECHANGED
        )
        try:
            # 不改变坐标、大小或层级；仅通知系统重新计算非客户区样式。
            reposition(hwnd, 0, 0, 0, 0, 0, flags)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
            return None
    return actual


def _set_native_input_transparency(
    window: object,
    enabled: bool,
    *,
    user32: object | None,
    baseline_style: int | None = None,
) -> tuple[bool | None, str]:
    """通过受控 WS_EX 位提供 Qt 输入透明的 Win32 兜底。"""

    hwnd = _window_id(window)
    if hwnd is None:
        return None, "无法读取有效 Win32 窗口句柄"
    current = _read_native_extended_style(user32, hwnd)
    if current is None:
        return None, "Win32 扩展窗口样式不可读"
    with _native_input_style_lock:
        baseline = _native_input_style_baselines.get(hwnd)
        if bool(enabled):
            if baseline is None:
                baseline = current if baseline_style is None else int(baseline_style)
                _native_input_style_baselines[hwnd] = baseline
            desired = int(current) | _WINDOW_INPUT_STYLE_BITS
        else:
            if baseline is None:
                # 不主动清理非本适配器设置的样式，避免影响其他窗口管理代码。
                return bool(current & _WINDOW_INPUT_STYLE_BITS) is False, "未发现本适配器的输入样式"
            desired = (int(current) & ~_WINDOW_INPUT_STYLE_BITS) | (
                int(baseline) & _WINDOW_INPUT_STYLE_BITS
            )
        actual = _commit_native_extended_style(user32, hwnd, desired)
        if actual is None:
            return None, "Win32 扩展窗口样式写入或回读失败"
        active = bool(actual & _WINDOW_INPUT_STYLE_BITS) is bool(enabled)
        if active and not bool(enabled):
            _native_input_style_baselines.pop(hwnd, None)
        return active, "Win32 扩展窗口样式已回读"


def _set_native_topmost(
    window: object,
    enabled: bool,
    *,
    user32: object | None,
) -> tuple[bool | None, str]:
    """用 SetWindowPos 请求 Win32 顶层层级，并回读扩展样式状态。"""

    hwnd = _window_id(window)
    reposition = getattr(user32, "SetWindowPos", None) if user32 is not None else None
    if hwnd is None or not callable(reposition):
        return None, "有效 Win32 窗口句柄或 SetWindowPos 不可用"
    flags = _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE | _SWP_NOOWNERZORDER
    try:
        accepted = bool(
            reposition(hwnd, _HWND_TOPMOST if enabled else _HWND_NOTOPMOST, 0, 0, 0, 0, flags)
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return None, "Win32 置顶请求失败"
    if not accepted:
        return None, "Win32 未接受置顶请求"
    actual = _read_native_extended_style(user32, hwnd)
    if actual is None:
        return None, "Win32 置顶样式不可回读"
    return bool(actual & _WS_EX_TOPMOST) is bool(enabled), "Win32 SetWindowPos 与扩展样式已回读"


def _read_text(user32: object, function_name: str, window_id: int) -> str:
    length_getter = getattr(user32, f"{function_name}LengthW", None)
    getter = getattr(user32, f"{function_name}W", None)
    if not callable(getter):
        return ""
    try:
        length = int(length_getter(window_id)) if callable(length_getter) else 256
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        length = 256
    buffer = ctypes.create_unicode_buffer(max(1, min(length + 1, 32768)))
    try:
        result = getter(window_id, buffer, len(buffer))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    if isinstance(result, str):
        return result.rstrip("\x00")
    return str(getattr(buffer, "value", "") or "").rstrip("\x00")


def _window_geometry(user32: object, window_id: int) -> tuple[int, int, int, int] | None:
    getter = getattr(user32, "GetWindowRect", None)
    if not callable(getter):
        return None
    rect = _Rect()
    try:
        if not bool(getter(window_id, ctypes.byref(rect))):
            return None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    width = int(rect.right) - int(rect.left)
    height = int(rect.bottom) - int(rect.top)
    if width <= 0 or height <= 0:
        return None
    return int(rect.left), int(rect.top), width, height


def _window_process_id(user32: object, window_id: int) -> int | None:
    getter = getattr(user32, "GetWindowThreadProcessId", None)
    if not callable(getter):
        return None
    process_id = wintypes.DWORD()
    try:
        # API 的返回值是线程 ID；进程 ID 只从第二个参数的输出缓冲读取。
        getter(window_id, ctypes.byref(process_id))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    value = _as_int(process_id)
    return value if value > 0 else None


def _process_details(
    process_id: int | None,
    *,
    include_command_line: bool,
) -> tuple[str, str, tuple[str, ...]]:
    if process_id is None:
        return "", "", ()
    psutil = _import_optional("psutil")
    process_type = getattr(psutil, "Process", None) if psutil is not None else None
    if not callable(process_type):
        return "", "", ()
    try:
        process = process_type(process_id)
        name = str(process.name() or "")
        executable = str(process.exe() or "")
        command_line = (
            tuple(str(item) for item in (process.cmdline() or ())) if include_command_line else ()
        )
        return name, executable, command_line
    except Exception:
        return "", "", ()


def _current_user_name() -> str:
    get_user = _import_optional("getpass")
    getter = getattr(get_user, "getuser", None) if get_user is not None else None
    if not callable(getter):
        return ""
    try:
        return str(getter() or "").strip().casefold()
    except (OSError, RuntimeError, TypeError, ValueError):
        return ""


def _windows_identity_matches(
    process_user: object,
    current_user: object,
    *,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """在已知 Windows 域边界内比较 psutil 用户身份。

    ``psutil`` 可能返回 ``DOMAIN\\User`` 或 ``User@domain``，而
    ``getpass.getuser`` 通常只返回短用户名。只有当前环境明确提供同一
    ``USERDOMAIN``/``USERDNSDOMAIN`` 时才展开限定身份，避免把其他域中
    的同名账户误认为当前用户。
    """

    process = str(process_user or "").strip().casefold()
    current = str(current_user or "").strip().casefold()
    if not process or not current:
        return False
    if process == current:
        return True

    def split_identity(value: str) -> tuple[str, str, str]:
        if "\\" in value:
            qualifier, account = value.rsplit("\\", 1)
            return "qualified", qualifier, account
        if "@" in value:
            account, qualifier = value.rsplit("@", 1)
            return "upn", qualifier, account
        return "short", "", value

    process_kind, process_qualifier, process_account = split_identity(process)
    current_kind, current_qualifier, current_account = split_identity(current)
    if process_account != current_account:
        return False

    values = os.environ if environ is None else environ
    trusted_qualifiers = {
        str(values.get(name) or "").strip().casefold() for name in ("USERDOMAIN", "USERDNSDOMAIN")
    }
    trusted_qualifiers.discard("")
    if current_qualifier:
        trusted_qualifiers.add(current_qualifier)

    # 仅允许环境明确声明的域/UPN 后缀；短用户名不会匹配未知限定身份。
    if process_kind == "short":
        return current_kind != "short" and current_qualifier in trusted_qualifiers
    if current_kind == "short":
        return process_qualifier in trusted_qualifiers
    return process_qualifier == current_qualifier and process_qualifier in trusted_qualifiers


def _psutil_processes(
    limit: int,
    *,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, object]] | None:
    psutil = _import_optional("psutil")
    iterator_factory = getattr(psutil, "process_iter", None) if psutil is not None else None
    if not callable(iterator_factory):
        return None
    processes: list[dict[str, object]] = []
    current_user = _current_user_name()
    try:
        iterator = iterator_factory(("pid", "name", "exe", "username"))
        for process in iterator:
            if len(processes) >= limit:
                break
            try:
                info = getattr(process, "info", {})
                pid = int(info.get("pid") or 0)
                if pid <= 0:
                    continue
                username = info.get("username")
                if (
                    current_user
                    and username
                    and not _windows_identity_matches(
                        username,
                        current_user,
                        environ=environ,
                    )
                ):
                    continue
                processes.append(
                    {
                        "pid": pid,
                        "name": str(info.get("name") or ""),
                        "executable": str(info.get("exe") or ""),
                    }
                )
            except (AttributeError, KeyError, OSError, PermissionError, TypeError, ValueError):
                continue
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return processes


def _toolhelp_processes(kernel32: object, limit: int) -> list[dict[str, object]] | None:
    snapshotter = getattr(kernel32, "CreateToolhelp32Snapshot", None)
    first = getattr(kernel32, "Process32FirstW", None)
    next_process = getattr(kernel32, "Process32NextW", None)
    closer = getattr(kernel32, "CloseHandle", None)
    if not all(callable(item) for item in (snapshotter, first, next_process, closer)):
        return None
    try:
        handle = snapshotter(_TH32CS_SNAPPROCESS, 0)
        handle_value = _as_int(handle)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None
    if handle_value in {0, -1, int(_INVALID_HANDLE_VALUE or -1)}:
        return None
    entry = _ProcessEntry32W()
    entry.dwSize = ctypes.sizeof(_ProcessEntry32W)
    processes: list[dict[str, object]] = []
    try:
        try:
            ok = bool(first(handle, ctypes.byref(entry)))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            ok = False
        while ok and len(processes) < limit:
            processes.append(
                {
                    "pid": int(entry.th32ProcessID),
                    "name": str(entry.szExeFile or ""),
                    "executable": "",
                }
            )
            try:
                ok = bool(next_process(handle, ctypes.byref(entry)))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                ok = False
    finally:
        try:
            closer(handle)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
    return processes


def probe_windows_platform(
    environ: Mapping[str, str] | None = None,
    *,
    user32: object | None = None,
    kernel32: object | None = None,
    gdi32: object | None = None,
    is_windows: bool | None = None,
) -> PlatformSnapshot:
    """探测 Windows 平台能力，不创建 Qt 窗口。"""

    del environ
    enabled = _is_windows(is_windows)
    user = _load_user32() if user32 is None and enabled else user32
    kernel = _load_kernel32() if kernel32 is None and enabled else kernel32
    gdi = _load_gdi32() if gdi32 is None and enabled else gdi32
    if not enabled:
        return _unavailable_snapshot(is_windows=False)
    if user is not None:
        _configure_win32_api(user, "user32")
    if kernel is not None:
        _configure_win32_api(kernel, "kernel32")

    qt_core = _import_optional("PySide6.QtCore")
    qt_gui = _import_optional("PySide6.QtGui")
    qt = getattr(qt_core, "Qt", None) if qt_core is not None else None
    window_type = getattr(qt, "WindowType", qt)
    qt_input_flag_available = getattr(window_type, "WindowTransparentForInput", None) is not None
    qt_topmost_flag_available = getattr(window_type, "WindowStaysOnTopHint", None) is not None
    qt_available = qt_core is not None
    foreground_available = bool(user and callable(getattr(user, "GetForegroundWindow", None)))
    cursor_available = bool(user and callable(getattr(user, "GetCursorPos", None)))
    idle_available = bool(
        user
        and kernel
        and callable(getattr(user, "GetLastInputInfo", None))
        and callable(getattr(kernel, "GetTickCount64", None))
    )
    process_available = bool(
        _import_optional("psutil") is not None
        or (kernel and callable(getattr(kernel, "CreateToolhelp32Snapshot", None)))
    )
    position_available = bool(user and callable(getattr(user, "SetWindowPos", None)))
    style_available = bool(
        user
        and (
            callable(getattr(user, "GetWindowLongPtrW", None))
            or callable(getattr(user, "GetWindowLongW", None))
        )
        and (
            callable(getattr(user, "SetWindowLongPtrW", None))
            or callable(getattr(user, "SetWindowLongW", None))
        )
    )
    click_available = bool(
        user
        and callable(getattr(user, "SetCursorPos", None))
        and callable(getattr(user, "SendInput", None))
    )
    hotkey_available = bool(
        user
        and callable(getattr(user, "RegisterHotKey", None))
        and callable(getattr(user, "UnregisterHotKey", None))
    )
    capture_available = qt_core is not None and qt_gui is not None
    capabilities = (
        _capability(
            "active_window",
            CapabilityState.AVAILABLE if foreground_available else CapabilityState.UNAVAILABLE,
            "Win32 foreground-window API is available"
            if foreground_available
            else "user32 foreground-window API is unavailable",
            "GetForegroundWindow",
        ),
        _capability(
            "process_context",
            CapabilityState.AVAILABLE if process_available else CapabilityState.UNAVAILABLE,
            "psutil or Toolhelp process enumeration is available"
            if process_available
            else "psutil and Toolhelp process enumeration are unavailable",
            "Toolhelp32Snapshot",
        ),
        _capability(
            "click_through",
            CapabilityState.AVAILABLE
            if qt_input_flag_available and style_available
            else CapabilityState.DEGRADED
            if qt_input_flag_available
            else CapabilityState.UNAVAILABLE,
            "Qt input flag and Win32 extended-style APIs are available"
            if qt_input_flag_available and style_available
            else "Qt input flag is available; Win32 style readback is unavailable"
            if qt_input_flag_available
            else "Qt WindowTransparentForInput is unavailable",
            "Qt.WindowType.WindowTransparentForInput",
            "WS_EX_TRANSPARENT",
        ),
        _capability(
            "always_on_top",
            CapabilityState.AVAILABLE
            if qt_topmost_flag_available and style_available
            else CapabilityState.DEGRADED
            if qt_topmost_flag_available
            else CapabilityState.UNAVAILABLE,
            "Qt topmost flag and Win32 style readback are available"
            if qt_topmost_flag_available and style_available
            else "Qt topmost flag is available; native readback is unavailable"
            if qt_topmost_flag_available
            else "Qt WindowStaysOnTopHint is unavailable",
            "Qt.WindowType.WindowStaysOnTopHint",
            "WS_EX_TOPMOST",
        ),
        _capability(
            "global_hotkeys",
            CapabilityState.AVAILABLE
            if hotkey_available
            else CapabilityState.DEGRADED
            if qt_available
            else CapabilityState.UNAVAILABLE,
            "RegisterHotKey is available"
            if hotkey_available
            else "Qt local shortcuts are available; Win32 global registration is unavailable"
            if qt_available
            else "global hotkey APIs are unavailable",
            "RegisterHotKey",
        ),
        _capability(
            "overlay_position",
            CapabilityState.AVAILABLE if position_available else CapabilityState.UNAVAILABLE,
            "SetWindowPos is available" if position_available else "SetWindowPos is unavailable",
            "SetWindowPos",
        ),
        _capability(
            "screen_capture",
            CapabilityState.AVAILABLE if capture_available else CapabilityState.UNAVAILABLE,
            "Qt screen capture is available"
            if capture_available
            else "PySide6 QtGui/QtCore is unavailable",
            "QScreen.grabWindow",
        ),
        _capability(
            "input_control",
            CapabilityState.AVAILABLE if click_available else CapabilityState.UNAVAILABLE,
            "Win32 cursor and SendInput injection is available"
            if click_available
            else "user32 input injection is unavailable",
            "SetCursorPos",
            "SendInput",
        ),
        _capability(
            "system_idle",
            CapabilityState.AVAILABLE if idle_available else CapabilityState.UNAVAILABLE,
            "GetLastInputInfo and GetTickCount64 are available"
            if idle_available
            else "Win32 idle-time APIs are unavailable",
            "GetLastInputInfo",
            "GetTickCount64",
        ),
        _capability(
            "cursor_position",
            CapabilityState.AVAILABLE if cursor_available else CapabilityState.UNAVAILABLE,
            "Win32 GetCursorPos is available"
            if cursor_available
            else "user32 cursor-position API is unavailable",
            "GetCursorPos",
        ),
    )
    return PlatformSnapshot(
        backend=BACKEND_WINDOWS,
        capabilities=capabilities,
        details={
            "platform": "Windows",
            "is_windows": True,
            "user32": bool(user),
            "kernel32": bool(kernel),
            "gdi32": bool(gdi),
        },
    )


def set_window_click_through(
    window: object,
    enabled: bool,
    *,
    user32: object | None = None,
    is_windows: bool | None = None,
) -> PlatformCapability:
    """切换点击穿透；Qt 标志不可用或未保留时使用 Win32 样式兜底。"""

    if not _is_windows(is_windows):
        return _capability(
            "click_through",
            CapabilityState.UNAVAILABLE,
            "Windows platform is unavailable",
        )
    qt_core = _import_optional("PySide6.QtCore")
    flag = _qt_flag(window, "WindowTransparentForInput") if qt_core is not None else None
    target = _qt_window_target(window)
    setter = getattr(target, "setFlag", None)
    if not callable(setter):
        setter = getattr(target, "setWindowFlag", None)
    was_visible = None
    visible_getter = getattr(window, "isVisible", None)
    if callable(visible_getter):
        try:
            was_visible = bool(visible_getter())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            was_visible = None
    position = _read_qt_position(window)
    native_api = user32 if user32 is not None else _load_user32()
    native_hwnd = _window_id(window) if native_api is not None else None
    native_baseline = (
        _read_native_extended_style(native_api, native_hwnd)
        if native_api is not None and native_hwnd is not None
        else None
    )
    if flag is not None and callable(setter):
        try:
            setter(flag, bool(enabled))
            if was_visible and callable(visible_getter) and not bool(visible_getter()):
                shower = getattr(window, "show", None)
                if callable(shower):
                    shower()
            if position is not None and _read_qt_position(window) != position:
                _restore_qt_position(window, position)
            active = _qt_flag_state(window, flag)
            if active is bool(enabled) and native_hwnd is None:
                return _capability(
                    "click_through",
                    CapabilityState.AVAILABLE,
                    "Qt WindowTransparentForInput was retained",
                    "Qt.WindowType.WindowTransparentForInput",
                )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Windows click-through Qt flag update failed: %s", type(exc).__name__)

    native_active, native_detail = _set_native_input_transparency(
        window,
        bool(enabled),
        user32=native_api,
        baseline_style=native_baseline,
    )
    if native_active is True:
        if position is not None and _read_qt_position(window) != position:
            _restore_qt_position(window, position)
        return _capability(
            "click_through",
            CapabilityState.AVAILABLE,
            f"{native_detail}；已启用 WS_EX_TRANSPARENT/WS_EX_NOACTIVATE 兜底",
            "WS_EX_TRANSPARENT",
            "WS_EX_NOACTIVATE",
        )
    detail = native_detail if native_active is None else "Win32 输入透明样式未保留请求状态"
    if flag is None:
        detail = "Qt WindowTransparentForInput 不可用；" + detail
    return _capability("click_through", CapabilityState.UNAVAILABLE, detail)


def _read_qt_position(window: object) -> tuple[int, int] | None:
    for getter_name in ("pos", "position"):
        getter = getattr(window, getter_name, None)
        if not callable(getter):
            continue
        try:
            point = getter()
            return int(point.x()), int(point.y())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
    return None


def _restore_qt_position(window: object, position: tuple[int, int] | None) -> None:
    if position is None:
        return
    mover = getattr(window, "move", None)
    if callable(mover):
        mover(*position)
        return
    setter = getattr(window, "setPosition", None)
    if not callable(setter):
        return
    qt_core = _import_optional("PySide6.QtCore")
    point_type = getattr(qt_core, "QPoint", None) if qt_core is not None else None
    point = point_type(*position) if callable(point_type) else position
    try:
        setter(point)
    except TypeError:
        setter(*position)


def set_window_input_shape(
    window: object,
    region: object | None,
    *,
    gdi32: object | None = None,
    is_windows: bool | None = None,
) -> PlatformCapability:
    """应用 Windows Qt Shape；空区域交由点击穿透标志处理。"""

    del gdi32
    if not _is_windows(is_windows):
        return _capability(
            "input_shape",
            CapabilityState.UNAVAILABLE,
            "Windows platform is unavailable",
        )
    if region == ():
        return _capability(
            "input_shape",
            CapabilityState.AVAILABLE,
            "empty input shape is represented by WindowTransparentForInput",
            "Qt.WindowType.WindowTransparentForInput",
        )
    target = _qt_window_target(window)
    setter = getattr(target, "setMask", None)
    if not callable(setter):
        return _capability(
            "input_shape",
            CapabilityState.UNAVAILABLE,
            "window does not expose a Qt mask setter",
        )
    try:
        if region is None:
            clearer = getattr(target, "clearMask", None)
            if callable(clearer):
                clearer()
            else:
                setter(region)
        else:
            setter(region)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Windows input Shape update failed: %s", type(exc).__name__)
        return _capability(
            "input_shape", CapabilityState.UNAVAILABLE, "Qt input Shape update failed"
        )
    return _capability("input_shape", CapabilityState.AVAILABLE, "Qt window mask was applied")


def _move_window(
    window: object,
    x: int,
    y: int,
    *,
    user32: object | None = None,
) -> None:
    mover = getattr(window, "move", None)
    if callable(mover):
        mover(int(x), int(y))
        return
    setter = getattr(window, "setPosition", None)
    if callable(setter):
        qt_core = _import_optional("PySide6.QtCore")
        point_type = getattr(qt_core, "QPoint", None) if qt_core is not None else None
        point = point_type(int(x), int(y)) if callable(point_type) else (int(x), int(y))
        try:
            setter(point)
        except TypeError:
            setter(int(x), int(y))
        return
    hwnd = _window_id(window)
    user32 = _load_user32() if user32 is None else user32
    setter = getattr(user32, "SetWindowPos", None) if user32 is not None else None
    if hwnd is None or not callable(setter):
        raise AttributeError("window does not expose a position setter")
    flags = _SWP_NOSIZE | _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_NOOWNERZORDER
    if not bool(setter(hwnd, 0, int(x), int(y), 0, 0, flags)):
        raise OSError("SetWindowPos failed")


def _read_cursor_position(user32: object | None) -> Mapping[str, object]:
    """通过 Win32 GetCursorPos 读取全局光标；失败时不返回坐标。"""

    getter = getattr(user32, "GetCursorPos", None) if user32 is not None else None
    if not callable(getter):
        return {
            "status": "unavailable",
            "backend": BACKEND_WINDOWS,
            "reason": "user32 cursor-position API is unavailable",
        }
    point = _Point()
    try:
        result = getter(ctypes.byref(point))
        if not bool(result):
            return {
                "status": "unavailable",
                "backend": BACKEND_WINDOWS,
                "reason": "GetCursorPos failed",
            }
        x = int(point.x)
        y = int(point.y)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return {
            "status": "unavailable",
            "backend": BACKEND_WINDOWS,
            "reason": "GetCursorPos failed",
        }
    return {
        "status": "available",
        "backend": BACKEND_WINDOWS,
        "x": x,
        "y": y,
        "source": "win32-GetCursorPos",
    }


def _window_position(
    window: object,
    *,
    user32: object | None = None,
) -> tuple[int, int] | None:
    position = _read_qt_position(window)
    if position is not None:
        return position
    hwnd = _window_id(window)
    user32 = _load_user32() if user32 is None else user32
    if hwnd is None or user32 is None:
        return None
    geometry = _window_geometry(user32, hwnd)
    return geometry[:2] if geometry else None


def _screen_geometry(screen: object) -> tuple[int, int, int, int] | None:
    """读取 QScreen 的虚拟桌面几何；读取失败时返回空。"""

    getter = getattr(screen, "geometry", None)
    if not callable(getter):
        return None
    try:
        geometry = getter()
        x_getter = getattr(geometry, "x", None)
        y_getter = getattr(geometry, "y", None)
        width_getter = getattr(geometry, "width", None)
        height_getter = getattr(geometry, "height", None)
        if not all(callable(item) for item in (x_getter, y_getter, width_getter, height_getter)):
            return None
        x = int(x_getter())
        y = int(y_getter())
        width = int(width_getter())
        height = int(height_getter())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _screen_for_capture_region(
    application: object,
    primary_screen: object,
    region: tuple[int, int, int, int],
) -> tuple[object, int, int] | None:
    """把全局虚拟桌面区域映射到一个 QScreen 的局部坐标。"""

    x, y, width, height = region
    right = x + width
    bottom = y + height
    screens_getter = getattr(application, "screens", None)
    if not callable(screens_getter):
        geometry = _screen_geometry(primary_screen)
        if geometry is None:
            return primary_screen, x, y
        left, top, screen_width, screen_height = geometry
        if (
            x >= left
            and y >= top
            and right <= left + screen_width
            and bottom <= top + screen_height
        ):
            return primary_screen, x - left, y - top
        return None
    try:
        screens = tuple(screens_getter())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        screens = (primary_screen,)
    if not screens:
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


def _capture_region(value: Mapping[str, object] | None) -> tuple[int, int, int, int]:
    """严格校验虚拟桌面截图区域，拒绝 JSON 的隐式数字转换。"""

    if not isinstance(value, Mapping):
        raise ValueError("capture region must be a mapping")
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


def _screen_for_window_id(
    application: object,
    primary_screen: object,
    window_id: int,
) -> object:
    """按同一 HWND 找到 Qt 所属屏幕，避免多屏窗口总在主屏抓取。"""

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
            if _as_int(getter()) != window_id:
                continue
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
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


class WindowsDesktopPlatform:
    """Windows 平台适配器门面。"""

    def __init__(
        self,
        environ: Mapping[str, str] | None = None,
        *,
        include_command_line: bool = False,
        user32: object | None = None,
        kernel32: object | None = None,
        gdi32: object | None = None,
        is_windows: bool | None = None,
    ) -> None:
        self._environment = dict(os.environ if environ is None else environ)
        self._include_command_line = bool(include_command_line)
        self._is_windows = _is_windows(is_windows)
        self._user32 = user32
        self._kernel32 = kernel32
        self._gdi32 = gdi32
        self._snapshot: PlatformSnapshot | None = None
        self._excluded_window_ids: set[int] = set()
        self._excluded_window_ids_lock = RLock()

    def _user32_api(self) -> object | None:
        """返回实例注入或当前 Windows 会话的 user32。"""

        if not self._is_windows:
            return None
        return self._user32 if self._user32 is not None else _load_user32()

    def _kernel32_api(self) -> object | None:
        """返回实例注入或当前 Windows 会话的 kernel32。"""

        if not self._is_windows:
            return None
        return self._kernel32 if self._kernel32 is not None else _load_kernel32()

    def _gdi32_api(self) -> object | None:
        """返回实例注入或当前 Windows 会话的 gdi32。"""

        if not self._is_windows:
            return None
        return self._gdi32 if self._gdi32 is not None else _load_gdi32()

    @property
    def backend(self) -> str:
        return BACKEND_WINDOWS

    def probe(self) -> PlatformSnapshot:
        self._snapshot = probe_windows_platform(
            self._environment,
            user32=self._user32,
            kernel32=self._kernel32,
            gdi32=self._gdi32,
            is_windows=self._is_windows,
        )
        return self._snapshot

    def active_window(self) -> WindowContext | None:
        if not self._is_windows:
            return None
        user32 = self._user32_api()
        getter = getattr(user32, "GetForegroundWindow", None) if user32 else None
        if not callable(getter):
            return None
        try:
            window_id = _as_int(getter())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        if window_id <= 0:
            return None
        with self._excluded_window_ids_lock:
            if window_id in self._excluded_window_ids:
                return None
        title = _read_text(user32, "GetWindowText", window_id)
        app_id = _read_text(user32, "GetClassName", window_id)
        process_id = _window_process_id(user32, window_id)
        process_name, executable, command_line = _process_details(
            process_id,
            include_command_line=self._include_command_line,
        )
        return WindowContext(
            backend=BACKEND_WINDOWS,
            window_id=f"0x{window_id:x}",
            title=title,
            app_id=app_id,
            pid=process_id,
            process_name=process_name,
            executable=executable,
            command_line=command_line,
            geometry=_window_geometry(user32, window_id),
            source="win32-user32",
            observed_at=time(),
        )

    def cursor_position(self) -> Mapping[str, object]:
        """读取 Windows 全局光标位置；非 Windows 或 API 失败时 fail-closed。"""

        if not self._is_windows:
            return {
                "status": "unavailable",
                "backend": BACKEND_WINDOWS,
                "reason": "platform is not Windows",
            }
        return _read_cursor_position(self._user32_api())

    get_active_window = active_window

    def system_idle_seconds(self) -> float | None:
        if not self._is_windows:
            return None
        user32 = self._user32_api()
        kernel32 = self._kernel32_api()
        idle_getter = getattr(user32, "GetLastInputInfo", None) if user32 else None
        tick_getter = getattr(kernel32, "GetTickCount64", None) if kernel32 else None
        if not callable(idle_getter) or not callable(tick_getter):
            return None
        info = _LastInputInfo()
        info.cbSize = ctypes.sizeof(_LastInputInfo)
        try:
            if not bool(idle_getter(ctypes.byref(info))):
                return None
            now_ms = _as_int(tick_getter())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        elapsed_ms = (now_ms - _as_int(info.dwTime)) & 0xFFFFFFFF
        elapsed = elapsed_ms / 1000.0
        return elapsed if math.isfinite(elapsed) and elapsed >= 0.0 else None

    def foreground_window(self) -> Mapping[str, object]:
        window = self.active_window()
        if window is None:
            return {
                "status": "unavailable",
                "backend": BACKEND_WINDOWS,
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
        window = self.active_window()
        if window is None or not window.window_id:
            return None
        try:
            return int(window.window_id, 16)
        except (TypeError, ValueError):
            return None

    def list_processes(self, limit: int = 20) -> Mapping[str, object]:
        if not self._is_windows:
            return {
                "status": "unavailable",
                "processes": [],
                "reason": "Windows platform is unavailable",
            }
        if isinstance(limit, bool) or not isinstance(limit, int):
            return {
                "status": "unavailable",
                "processes": [],
                "reason": "process limit is invalid",
            }
        safe_limit = max(1, min(limit, 100))
        processes = _psutil_processes(safe_limit, environ=self._environment)
        if processes is None:
            kernel32 = self._kernel32_api()
            processes = _toolhelp_processes(kernel32, safe_limit) if kernel32 is not None else None
        if processes is None:
            return {
                "status": "unavailable",
                "processes": [],
                "reason": "Windows process enumeration is unavailable",
            }
        return {"status": "available", "processes": processes}

    def capture_screen(
        self,
        *,
        scope: str = "screen",
        region: Mapping[str, object] | None = None,
        window_id: int | None = None,
    ) -> Mapping[str, object]:
        if not self._is_windows:
            return {"status": "unavailable", "reason": "Windows platform is unavailable"}
        normalized_scope = str(scope or "screen").strip().lower()
        if normalized_scope not in {"screen", "window", "region"}:
            return {"status": "unavailable", "reason": "unsupported capture scope"}
        if normalized_scope == "window" and window_id is None:
            return {
                "status": "unavailable",
                "reason": "window id must be resolved before Qt screen capture",
            }
        safe_window_id: int | None = None
        if normalized_scope == "window":
            if isinstance(window_id, bool) or not isinstance(window_id, int) or window_id <= 0:
                return {"status": "unavailable", "reason": "active window id is invalid"}
            safe_window_id = int(window_id)

        capture_region: tuple[int, int, int, int] | None = None
        if normalized_scope == "region":
            try:
                capture_region = _capture_region(region)
            except ValueError as exc:
                reason = str(exc)
                return {
                    "status": "unavailable",
                    "reason": reason,
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
                assert safe_window_id is not None
                screen = _screen_for_window_id(application, screen, safe_window_id)
                image = screen.grabWindow(safe_window_id)
                window_geometry = _window_geometry(self._user32_api(), safe_window_id)
            elif capture_region is not None:
                x, y, width, height = capture_region
                selected = _screen_for_capture_region(application, screen, capture_region)
                if selected is None:
                    return {
                        "status": "unavailable",
                        "reason": "capture region must fit within a single screen",
                    }
                target_screen, local_x, local_y = selected
                try:
                    image = target_screen.grabWindow(0, local_x, local_y, width, height)
                except TypeError:
                    # 旧测试替身或非标准 Qt 绑定可能只接受 window 参数；
                    # 回退到同一目标屏幕的局部裁剪，不改变真实 Qt 路径。
                    image = target_screen.grabWindow(0).copy(local_x, local_y, width, height)
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
        if not self._is_windows:
            return {
                "status": "unavailable",
                "reason": "Windows input injection is unavailable",
            }
        if isinstance(x, bool) or isinstance(y, bool):
            return {"status": "unavailable", "reason": "click coordinates are invalid"}
        target_x = _automation_integer(x)
        target_y = _automation_integer(y)
        if target_x is None or target_y is None:
            return {"status": "unavailable", "reason": "click coordinates are invalid"}
        if (
            target_x < -1_000_000
            or target_y < -1_000_000
            or target_x > 1_000_000
            or target_y > 1_000_000
        ):
            return {"status": "unavailable", "reason": "click coordinates are outside bounds"}
        events = {
            "left": (_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
            "middle": (_MOUSEEVENTF_MIDDLEDOWN, _MOUSEEVENTF_MIDDLEUP),
            "right": (_MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP),
        }
        if not isinstance(button, str):
            return {"status": "unavailable", "reason": "click button is unsupported"}
        normalized_button = button.strip().lower()
        pair = events.get(normalized_button)
        if pair is None:
            return {"status": "unavailable", "reason": "click button is unsupported"}
        user32 = self._user32_api()
        move = getattr(user32, "SetCursorPos", None) if user32 else None
        if not callable(move) or not callable(getattr(user32, "SendInput", None)):
            return {
                "status": "unavailable",
                "reason": "Win32 input injection is unavailable",
            }
        try:
            if not bool(move(target_x, target_y)):
                return {"status": "unavailable", "reason": "cursor positioning failed"}
            if not _send_inputs(
                user32,
                [_mouse_input(pair[0]), _mouse_input(pair[1])],
            ):
                # 首个按下事件可能已经提交；尽力补发释放，避免失败后
                # 系统留下逻辑上仍处于按下状态的鼠标按钮。
                _send_inputs(user32, [_mouse_input(pair[1])])
                return {
                    "status": "unavailable",
                    "reason": "Win32 click input was not fully accepted",
                }
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "Win32 click injection failed"}
        return {
            "status": "completed",
            "backend": BACKEND_WINDOWS,
            "x": target_x,
            "y": target_y,
            "button": normalized_button,
        }

    def automation_batch(self, steps: object) -> Mapping[str, object]:
        """执行经过工具层确认的有限 Windows 自动化步骤。

        平台边界再次校验全部字段，避免直接平台调用绕过工具 JSON Schema。
        文本内容和窗口标题不会写入返回值；执行结果只保留步骤类型与数量。
        """

        if not self._is_windows:
            return {
                "status": "unavailable",
                "backend": BACKEND_WINDOWS,
                "reason": "Windows automation is unavailable",
            }
        if isinstance(steps, (str, bytes, bytearray)) or not isinstance(steps, (list, tuple)):
            return {
                "status": "unavailable",
                "backend": BACKEND_WINDOWS,
                "reason": "automation steps are invalid",
            }
        if not steps or len(steps) > _AUTOMATION_MAX_STEPS:
            return {
                "status": "unavailable",
                "backend": BACKEND_WINDOWS,
                "reason": "automation step count is out of range",
            }
        user32 = self._user32_api()
        if user32 is None:
            return {
                "status": "unavailable",
                "backend": BACKEND_WINDOWS,
                "reason": "Win32 automation APIs are unavailable",
            }

        completed: list[dict[str, object]] = []
        total_wait_ms = 0
        for index, raw_step in enumerate(steps):
            if not isinstance(raw_step, Mapping):
                return _automation_failure("automation step is invalid", completed=completed)
            step_type = str(raw_step.get("type", "")).strip().lower()
            if step_type == "wait":
                duration = raw_step.get("duration_ms")
                if isinstance(duration, bool):
                    return _automation_failure("automation wait is invalid", completed=completed)
                duration_ms = _automation_integer(duration)
                if duration_ms is None:
                    return _automation_failure("automation wait is invalid", completed=completed)
                if duration_ms < 0 or duration_ms > _AUTOMATION_MAX_WAIT_MS:
                    return _automation_failure(
                        "automation wait is out of range", completed=completed
                    )
                total_wait_ms += duration_ms
                if total_wait_ms > _AUTOMATION_MAX_WAIT_TOTAL_MS:
                    return _automation_failure(
                        "automation total wait is out of range", completed=completed
                    )
                if duration_ms:
                    sleep(duration_ms / 1000.0)
                completed.append({"index": index, "type": step_type, "status": "completed"})
                continue

            if step_type == "key":
                virtual_key = _automation_key_code(raw_step.get("key"))
                raw_modifiers = raw_step.get("modifiers", ())
                repeat = raw_step.get("repeat", 1)
                if (
                    virtual_key is None
                    or isinstance(raw_modifiers, (str, bytes, bytearray))
                    or not isinstance(raw_modifiers, (list, tuple))
                    or not all(isinstance(item, str) for item in raw_modifiers)
                ):
                    return _automation_failure("automation key is invalid", completed=completed)
                repeats = _automation_integer(repeat)
                if repeats is None:
                    return _automation_failure("automation key is invalid", completed=completed)
                if repeats < 1 or repeats > 5 or len(raw_modifiers) > 3:
                    return _automation_failure(
                        "automation key is out of range", completed=completed
                    )
                modifiers: list[int] = []
                for raw_modifier in raw_modifiers:
                    modifier = _AUTOMATION_MODIFIERS.get(raw_modifier.strip().lower())
                    if modifier is None or modifier in modifiers:
                        return _automation_failure(
                            "automation modifier is invalid", completed=completed
                        )
                    modifiers.append(modifier)
                inputs = [_keyboard_input(modifier) for modifier in modifiers]
                for _ in range(repeats):
                    inputs.extend(
                        (
                            _keyboard_input(virtual_key),
                            _keyboard_input(virtual_key, flags=_KEYEVENTF_KEYUP),
                        )
                    )
                inputs.extend(
                    _keyboard_input(modifier, flags=_KEYEVENTF_KEYUP)
                    for modifier in reversed(modifiers)
                )
                if not _send_inputs(user32, inputs):
                    # 主键或修饰键可能已经提交；尽力释放已可能按下的按键，
                    # 失败仍返回 fail-closed。
                    cleanup_inputs = [_keyboard_input(virtual_key, flags=_KEYEVENTF_KEYUP)]
                    cleanup_inputs.extend(
                        _keyboard_input(modifier, flags=_KEYEVENTF_KEYUP)
                        for modifier in reversed(modifiers)
                    )
                    _send_inputs(
                        user32,
                        cleanup_inputs,
                    )
                    return _automation_failure(
                        "Win32 key input was not fully accepted", completed=completed
                    )
                completed.append(
                    {"index": index, "type": step_type, "status": "completed", "repeat": repeats}
                )
                continue

            if step_type == "text":
                text = raw_step.get("text")
                if not isinstance(text, str) or not text or len(text) > _AUTOMATION_MAX_TEXT_LENGTH:
                    return _automation_failure("automation text is invalid", completed=completed)
                if any(ord(character) < 32 or ord(character) == 127 for character in text):
                    return _automation_failure(
                        "automation text contains control characters", completed=completed
                    )
                try:
                    encoded = text.encode("utf-16-le", "strict")
                except UnicodeError:
                    return _automation_failure(
                        "automation text encoding is invalid", completed=completed
                    )
                code_units = [
                    int.from_bytes(encoded[offset : offset + 2], "little")
                    for offset in range(0, len(encoded), 2)
                ]
                inputs = [
                    item
                    for unit in code_units
                    for item in (_unicode_input(unit), _unicode_input(unit, key_up=True))
                ]
                if not _send_inputs(user32, inputs):
                    # Unicode 输入由成对事件组成；补发所有释放事件，避免
                    # 部分提交后把输入状态留在按下态。
                    _send_inputs(
                        user32,
                        [_unicode_input(unit, key_up=True) for unit in code_units],
                    )
                    return _automation_failure(
                        "Win32 text input was not fully accepted", completed=completed
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

            if step_type == "click":
                result = self.click_at(
                    raw_step.get("x"),
                    raw_step.get("y"),
                    button=raw_step.get("button", "left"),
                )
                if result.get("status") != "completed":
                    return _automation_failure("Win32 click input failed", completed=completed)
                completed.append({"index": index, "type": step_type, "status": "completed"})
                continue

            if step_type == "move_pointer":
                x = raw_step.get("x")
                y = raw_step.get("y")
                if isinstance(x, bool) or isinstance(y, bool):
                    return _automation_failure(
                        "automation coordinates are invalid", completed=completed
                    )
                target_x = _automation_integer(x)
                target_y = _automation_integer(y)
                if target_x is None or target_y is None:
                    return _automation_failure(
                        "automation coordinates are invalid", completed=completed
                    )
                if (
                    abs(target_x) > _AUTOMATION_COORDINATE_LIMIT
                    or abs(target_y) > _AUTOMATION_COORDINATE_LIMIT
                ):
                    return _automation_failure(
                        "automation coordinates are outside bounds", completed=completed
                    )
                move = getattr(user32, "SetCursorPos", None)
                try:
                    accepted = bool(move(target_x, target_y)) if callable(move) else False
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    accepted = False
                if not accepted:
                    return _automation_failure(
                        "Win32 cursor positioning failed", completed=completed
                    )
                completed.append({"index": index, "type": step_type, "status": "completed"})
                continue

            if step_type == "activate_window":
                window_id = raw_step.get("window_id")
                hwnd = _automation_window_id(window_id)
                if hwnd is None:
                    return _automation_failure(
                        "automation window id is invalid", completed=completed
                    )
                activate = getattr(user32, "SetForegroundWindow", None)
                if hwnd <= 0 or not callable(activate):
                    return _automation_failure(
                        "Win32 window activation is unavailable", completed=completed
                    )
                try:
                    accepted = bool(activate(hwnd))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    accepted = False
                if not accepted:
                    return _automation_failure(
                        "Win32 window activation failed", completed=completed
                    )
                foreground = getattr(user32, "GetForegroundWindow", None)
                if not callable(foreground):
                    return _automation_failure(
                        "Win32 foreground readback is unavailable", completed=completed
                    )
                try:
                    if _as_int(foreground()) != hwnd:
                        return _automation_failure(
                            "Win32 foreground readback did not match", completed=completed
                        )
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    return _automation_failure(
                        "Win32 foreground readback failed", completed=completed
                    )
                completed.append({"index": index, "type": step_type, "status": "completed"})
                continue

            if step_type == "move_window":
                window_id = raw_step.get("window_id")
                x = raw_step.get("x")
                y = raw_step.get("y")
                if any(isinstance(value, bool) for value in (x, y)):
                    return _automation_failure(
                        "automation window position is invalid", completed=completed
                    )
                hwnd = _automation_window_id(window_id)
                target_x = _automation_integer(x)
                target_y = _automation_integer(y)
                if target_x is None or target_y is None:
                    return _automation_failure(
                        "automation window position is invalid", completed=completed
                    )
                if (
                    hwnd is None
                    or abs(target_x) > _AUTOMATION_COORDINATE_LIMIT
                    or abs(target_y) > _AUTOMATION_COORDINATE_LIMIT
                ):
                    return _automation_failure(
                        "automation window position is outside bounds", completed=completed
                    )
                reposition = getattr(user32, "SetWindowPos", None)
                if not callable(reposition):
                    return _automation_failure(
                        "Win32 window move is unavailable", completed=completed
                    )
                try:
                    accepted = bool(
                        reposition(
                            hwnd,
                            0,
                            target_x,
                            target_y,
                            0,
                            0,
                            _SWP_NOSIZE | _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_NOOWNERZORDER,
                        )
                    )
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    accepted = False
                if not accepted:
                    return _automation_failure("Win32 window move failed", completed=completed)
                geometry = _window_geometry(user32, hwnd)
                if geometry is not None and geometry[:2] != (target_x, target_y):
                    return _automation_failure(
                        "Win32 window position readback did not match", completed=completed
                    )
                completed.append({"index": index, "type": step_type, "status": "completed"})
                continue

            return _automation_failure("automation step type is unsupported", completed=completed)

        return {
            "status": "completed",
            "backend": BACKEND_WINDOWS,
            "steps": tuple(completed),
        }

    def exclude_window_id(self, window_id: int) -> None:
        if not self._is_windows:
            return
        try:
            value = int(window_id)
        except (TypeError, ValueError, OverflowError):
            return
        if value > 0:
            with self._excluded_window_ids_lock:
                self._excluded_window_ids.add(value)

    def set_click_through(self, window: object, enabled: bool) -> PlatformCapability:
        if not self._is_windows:
            return _capability(
                "click_through",
                CapabilityState.UNAVAILABLE,
                "Windows platform is unavailable",
            )
        return set_window_click_through(
            window,
            enabled,
            user32=self._user32_api(),
            is_windows=self._is_windows,
        )

    def set_input_shape(self, window: object, region: object | None) -> PlatformCapability:
        return set_window_input_shape(
            window,
            region,
            gdi32=self._gdi32_api(),
            is_windows=self._is_windows,
        )

    def always_on_top_status(self, window: object) -> Mapping[str, object]:
        if not self._is_windows:
            return {
                "status": "unavailable",
                "enabled": False,
                "confirmed": False,
                "detail": "Windows platform is unavailable",
            }
        user32 = self._user32_api()
        hwnd = _window_id(window)
        getter = getattr(user32, "GetWindowLongPtrW", None) if user32 else None
        if not callable(getter) and user32 is not None:
            getter = getattr(user32, "GetWindowLongW", None)
        native_enabled: bool | None = None
        if hwnd is not None and callable(getter):
            try:
                native_enabled = bool(_as_int(getter(hwnd, _GWL_EXSTYLE)) & _WS_EX_TOPMOST)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                native_enabled = None
        qt_flag = _qt_flag(window, "WindowStaysOnTopHint")
        qt_enabled = _qt_flag_state(window, qt_flag) if qt_flag is not None else None
        enabled = native_enabled if native_enabled is not None else qt_enabled
        if enabled is None:
            return {
                "status": "unavailable",
                "enabled": False,
                "confirmed": False,
                "detail": "无法回读 Windows 原生窗口置顶状态",
            }
        if native_enabled is not None:
            return {
                "status": "available",
                "enabled": native_enabled,
                "confirmed": True,
                "detail": "Win32 extended style confirms topmost state",
                "evidence": ("WS_EX_TOPMOST",),
            }
        return {
            "status": "degraded",
            "enabled": bool(enabled),
            "confirmed": False,
            "detail": "Qt topmost flag is readable; Win32 z-order readback is unavailable",
        }

    def set_always_on_top(self, window: object, enabled: bool) -> PlatformCapability:
        """通过 Win32 z-order API 设置置顶并回读原生状态。"""

        if not self._is_windows:
            return _capability(
                "always_on_top",
                CapabilityState.UNAVAILABLE,
                "Windows platform is unavailable",
            )
        result, detail = _set_native_topmost(
            window,
            bool(enabled),
            user32=self._user32_api(),
        )
        if result is True:
            return _capability(
                "always_on_top",
                CapabilityState.AVAILABLE,
                detail,
                "SetWindowPos",
                "WS_EX_TOPMOST",
            )
        return _capability(
            "always_on_top",
            CapabilityState.UNAVAILABLE,
            detail,
            "SetWindowPos",
            "WS_EX_TOPMOST",
        )

    def move_overlay(self, window: object, x: int, y: int) -> PlatformCapability:
        if not self._is_windows:
            return _capability(
                "overlay_position",
                CapabilityState.UNAVAILABLE,
                "Windows platform is unavailable",
            )
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int)
            or not isinstance(y, int)
        ):
            return _capability(
                "overlay_position",
                CapabilityState.UNAVAILABLE,
                "Windows position coordinates are invalid",
            )
        try:
            target_x = x
            target_y = y
            _move_window(
                window,
                target_x,
                target_y,
                user32=self._user32_api(),
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Windows overlay move failed: %s", type(exc).__name__)
            return _capability(
                "overlay_position",
                CapabilityState.UNAVAILABLE,
                "Windows position request failed",
            )
        observed = _window_position(
            window,
            user32=self._user32_api(),
        )
        if observed is not None and observed != (target_x, target_y):
            return _capability(
                "overlay_position",
                CapabilityState.UNAVAILABLE,
                "Windows window geometry did not retain the requested position",
                "QWidget.move/QWindow.setPosition/SetWindowPos",
            )
        return _capability(
            "overlay_position",
            CapabilityState.AVAILABLE,
            "Windows position request submitted",
            "SetWindowPos",
        )


__all__ = [
    "BACKEND_WINDOWS",
    "WindowsDesktopPlatform",
    "probe_windows_platform",
    "set_window_click_through",
    "set_window_input_shape",
]
