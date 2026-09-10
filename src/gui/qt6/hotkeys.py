from __future__ import annotations

import ctypes
import os
import re
import select
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from ctypes import wintypes
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Self


class HotkeyMode(StrEnum):
    """快捷键触发方式。"""

    PRESS = "press"
    HOLD = "hold"


DEFAULT_HOTKEYS: tuple[tuple[str, str, HotkeyMode], ...] = (
    ("open_input", "Ctrl+Return", HotkeyMode.HOLD),
    ("open_console", "Ctrl+Shift+M", HotkeyMode.PRESS),
    ("toggle_visibility", "Ctrl+Shift+H", HotkeyMode.PRESS),
    ("toggle_always_on_top", "Ctrl+Shift+T", HotkeyMode.PRESS),
    ("toggle_window_lock", "Ctrl+Shift+L", HotkeyMode.PRESS),
    ("restore_click_through", "Ctrl+Alt+Space", HotkeyMode.HOLD),
)

_ACTION_PATTERN = re.compile(r"^[a-z][a-z0-9_:-]{0,63}$")
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_<>]+(?:\+[A-Za-z0-9_<>]+)*$")
_MODIFIER_NAMES = frozenset({"CTRL", "CONTROL", "ALT", "SHIFT", "META", "SUPER", "KEYPAD"})
_KEY_NAME_MAP = {
    "RETURN": "Return",
    "ENTER": "Enter",
    "SPACE": "Space",
    "TAB": "Tab",
    "ESC": "Esc",
    "ESCAPE": "Escape",
    "BACKSPACE": "Backspace",
    "DELETE": "Delete",
    "INSERT": "Insert",
    "HOME": "Home",
    "END": "End",
    "PAGEUP": "PageUp",
    "PAGEDOWN": "PageDown",
    "UP": "Up",
    "DOWN": "Down",
    "LEFT": "Left",
    "RIGHT": "Right",
}


def _parse_bool(value: object, *, default: bool) -> bool:
    """解析快捷键配置中的严格布尔值。"""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError("hotkeys.enabled must be a boolean")


def normalize_key_sequence(value: object) -> str:
    """规范化并校验 Qt PortableText 形式的快捷键。"""

    text = str(value or "").strip()
    if not text or len(text) > 80 or not _KEY_PATTERN.fullmatch(text):
        raise ValueError("hotkey sequence must use Qt key names joined by '+'")
    parts = tuple(part.upper() for part in text.split("+"))
    if not parts or not parts[-1]:
        raise ValueError("hotkey sequence key is required")
    # 只允许一个非修饰键，避免按住配置退化为只有修饰键而无法收到释放事件。
    non_modifier_indexes = [
        index for index, part in enumerate(parts) if part not in _MODIFIER_NAMES
    ]
    if len(non_modifier_indexes) != 1 or non_modifier_indexes[0] != len(parts) - 1:
        raise ValueError("hotkey sequence must contain exactly one non-modifier key")
    return "+".join(
        {
            "CONTROL": "Ctrl",
            "CTRL": "Ctrl",
            "ALT": "Alt",
            "SHIFT": "Shift",
            "META": "Meta",
            "SUPER": "Meta",
            "KEYPAD": "Keypad",
        }.get(part, _KEY_NAME_MAP.get(part, part.title()))
        for part in parts
    )


@dataclass(frozen=True)
class HotkeyBinding:
    """一个动作的快捷键绑定。"""

    action: str
    sequence: str
    mode: HotkeyMode = HotkeyMode.PRESS
    enabled: bool = True

    def __post_init__(self) -> None:
        action = str(self.action or "").strip().lower()
        if not _ACTION_PATTERN.fullmatch(action):
            raise ValueError("hotkey action is invalid")
        sequence = normalize_key_sequence(self.sequence)
        try:
            mode = HotkeyMode(self.mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("hotkey mode must be press or hold") from exc
        if mode is HotkeyMode.HOLD and sequence.split("+")[-1] in _MODIFIER_NAMES:
            raise ValueError("hold hotkey requires a non-modifier key")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "enabled", bool(self.enabled))


@dataclass(frozen=True)
class HotkeyConfig:
    """界面快捷键配置。"""

    enabled: bool = True
    bindings: tuple[HotkeyBinding, ...] = ()

    @classmethod
    def defaults(cls) -> Self:
        return cls(
            enabled=True,
            bindings=tuple(
                HotkeyBinding(action, sequence, mode) for action, sequence, mode in DEFAULT_HOTKEYS
            ),
        )

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        """从 ``ui.hotkeys`` 映射构造配置。

        推荐使用 ``bindings`` 列表；为便于从旧配置迁移，也允许在映射顶层
        直接用动作名覆盖默认序列。显式空列表表示关闭所有快捷键绑定。
        """

        if value is None:
            return cls.defaults()
        if not isinstance(value, Mapping):
            raise ValueError("ui.hotkeys must be a mapping")
        enabled = _parse_bool(value.get("enabled"), default=True)
        rows: list[HotkeyBinding] = []
        if "bindings" in value:
            raw_bindings = value.get("bindings")
            if not isinstance(raw_bindings, Sequence) or isinstance(
                raw_bindings, (str, bytes, bytearray)
            ):
                raise ValueError("ui.hotkeys.bindings must be a sequence")
            for index, raw in enumerate(raw_bindings):
                if not isinstance(raw, Mapping):
                    raise ValueError(f"ui.hotkeys.bindings[{index}] must be a mapping")
                action = raw.get("action")
                sequence = raw.get("sequence")
                if action is None or sequence is None:
                    raise ValueError(f"ui.hotkeys.bindings[{index}] requires action and sequence")
                try:
                    mode = HotkeyMode(str(raw.get("mode", HotkeyMode.PRESS)).strip().lower())
                except ValueError as exc:
                    raise ValueError(f"ui.hotkeys.bindings[{index}].mode is invalid") from exc
                rows.append(
                    HotkeyBinding(
                        str(action),
                        str(sequence),
                        mode,
                        _parse_bool(raw.get("enabled"), default=True),
                    )
                )
        else:
            for action, sequence, mode in DEFAULT_HOTKEYS:
                raw = value.get(action)
                if raw is None:
                    rows.append(HotkeyBinding(action, sequence, mode))
                elif isinstance(raw, Mapping):
                    rows.append(
                        HotkeyBinding(
                            action,
                            str(raw.get("sequence", sequence)),
                            HotkeyMode(str(raw.get("mode", mode)).strip().lower()),
                            _parse_bool(raw.get("enabled"), default=True),
                        )
                    )
                else:
                    rows.append(HotkeyBinding(action, str(raw), mode))
        enabled_rows = tuple(row for row in rows if row.enabled)
        seen: set[str] = set()
        for row in enabled_rows:
            if row.sequence in seen:
                raise ValueError(f"duplicate hotkey sequence: {row.sequence}")
            seen.add(row.sequence)
        return cls(enabled=enabled, bindings=enabled_rows)

    def by_action(self, action: str) -> tuple[HotkeyBinding, ...]:
        """返回指定动作的启用绑定。"""

        name = str(action or "").strip().lower()
        return tuple(binding for binding in self.bindings if binding.action == name)


def validate_hotkey_config(value: object) -> HotkeyConfig:
    """配置校验入口，供启动阶段和控制台配置编辑器复用。"""

    return HotkeyConfig.from_mapping(value)


_IS_WINDOWS = os.name == "nt" or sys.platform.startswith("win")
_WIN32_MODIFIER_MASKS = {
    "ALT": 0x0001,
    "CTRL": 0x0002,
    "SHIFT": 0x0004,
    "META": 0x0008,
}
_WIN32_VIRTUAL_KEYS = {
    "BACKSPACE": 0x08,
    "TAB": 0x09,
    "RETURN": 0x0D,
    "ENTER": 0x0D,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "ALT": 0x12,
    "PAUSE": 0x13,
    "CAPSLOCK": 0x14,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "SPACE": 0x20,
    "PAGEUP": 0x21,
    "PAGEDOWN": 0x22,
    "END": 0x23,
    "HOME": 0x24,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "INSERT": 0x2D,
    "DELETE": 0x2E,
}
_WIN32_MODIFIER_VIRTUAL_KEYS = {
    "ALT": (0xA4, 0xA5),
    "CTRL": (0xA2, 0xA3),
    "SHIFT": (0xA0, 0xA1),
    "META": (0x5B, 0x5C),
}
_WM_QUIT = 0x0012
_WM_HOTKEY = 0x0312
_PM_REMOVE = 0x0001
_MOD_NOREPEAT = 0x4000


def _load_win32_library(name: str) -> object | None:
    """在 Windows 上延迟加载 Win32 DLL；其它平台不触碰 ``ctypes.WinDLL``。"""

    if not _IS_WINDOWS:
        return None
    try:
        loader = getattr(ctypes, "WinDLL", None)
        if not callable(loader):
            return None
        return loader(name, use_last_error=True)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


_WIN32_USER32 = _load_win32_library("user32.dll")
_WIN32_KERNEL32 = _load_win32_library("kernel32.dll")


def _configure_win32_api(api: object | None) -> None:
    """为 ctypes Win32 函数声明真实参数，测试替身保持原样。"""

    if api is None:
        return
    specifications = {
        "RegisterHotKey": (
            [wintypes.HWND, wintypes.INT, wintypes.UINT, wintypes.UINT],
            wintypes.BOOL,
        ),
        "UnregisterHotKey": ([wintypes.HWND, wintypes.INT], wintypes.BOOL),
        "PeekMessageW": (
            [
                ctypes.POINTER(wintypes.MSG),
                wintypes.HWND,
                wintypes.UINT,
                wintypes.UINT,
                wintypes.UINT,
            ],
            wintypes.BOOL,
        ),
        "GetMessageW": (
            [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT],
            wintypes.BOOL,
        ),
        "PostThreadMessageW": (
            [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM],
            wintypes.BOOL,
        ),
        "GetAsyncKeyState": ([wintypes.INT], wintypes.SHORT),
    }
    for name, (argtypes, restype) in specifications.items():
        function = getattr(api, name, None)
        if not callable(function):
            continue
        try:
            function.argtypes = argtypes
            function.restype = restype
        except (AttributeError, TypeError):
            # Python 测试替身的 bound method 不支持 ctypes 属性。
            continue


def _win32_virtual_key(key: str) -> int:
    """返回 RegisterHotKey/GetAsyncKeyState 使用的虚拟键码。"""

    token = str(key or "").strip().upper()
    virtual_key = _WIN32_VIRTUAL_KEYS.get(token)
    if virtual_key is not None:
        return virtual_key
    if len(token) == 1 and token.isalnum():
        return ord(token)
    if token.startswith("F") and token[1:].isdigit():
        function_number = int(token[1:])
        if 1 <= function_number <= 24:
            return 0x70 + function_number - 1
    return 0


def _win32_hotkey_parts(sequence: str) -> tuple[int, int, tuple[str, ...]]:
    """把已规范化的 Qt 组合键转换为 RegisterHotKey 参数。"""

    parts = tuple(item.upper() for item in str(sequence).split("+"))
    if not parts or not parts[-1]:
        raise ValueError("hotkey sequence key is required")
    modifiers = 0
    for part in parts[:-1]:
        mask = _WIN32_MODIFIER_MASKS.get(part)
        if mask is None:
            raise ValueError(f"unsupported Windows modifier: {part}")
        modifiers |= mask
    key = parts[-1]
    virtual_key = _win32_virtual_key(key)
    if not virtual_key:
        raise ValueError(f"unsupported Windows key: {key}")
    return modifiers, virtual_key, parts


def _win32_message_value(message: object, name: str, default: int = 0) -> int:
    """读取 ctypes 消息结构或测试替身中的整数消息字段。"""

    try:
        value = getattr(message, name, default)
        return int(getattr(value, "value", value) or 0)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return default


class _Win32GlobalHotkey:
    """使用 RegisterHotKey/WM_HOTKEY 的 Windows 全局快捷键桥接。

    ``RegisterHotKey`` 绑定到注册线程的消息队列，因此桥接器拥有独立的
    消息线程，不阻塞 Qt 主线程。Windows 没有 ``WM_HOTKEY`` 释放事件，
    ``hold`` 绑定由 ``GetAsyncKeyState`` 在同一线程轮询修饰键和主键状态。
    """

    def __init__(
        self,
        bindings: Sequence[HotkeyBinding],
        callback: Callable[[str, bool], None],
        *,
        user32: object | None = None,
        kernel32: object | None = None,
        is_windows: bool | None = None,
        startup_timeout: float = 2.0,
    ) -> None:
        self._bindings = tuple(bindings)
        self._callback = callback
        self._user32 = _WIN32_USER32 if user32 is None else user32
        self._kernel32 = _WIN32_KERNEL32 if kernel32 is None else kernel32
        _configure_win32_api(self._user32)
        self._is_windows = _IS_WINDOWS if is_windows is None else bool(is_windows)
        self._startup_timeout = max(0.1, float(startup_timeout))
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._registrations: list[tuple[int, HotkeyBinding]] = []
        self._registration_parts: dict[int, tuple[int, tuple[str, ...]]] = {}
        self._active_ids: set[int] = set()
        self._active_holds: set[int] = set()
        self._startup_result: tuple[str, tuple[str, ...], tuple[str, ...]] | None = None
        self.errors: list[str] = []

    @staticmethod
    def _return_code(value: object) -> int:
        try:
            return int(getattr(value, "value", value) or 0)
        except (TypeError, ValueError, OverflowError):
            return 0

    def _last_error(self) -> int:
        getter = getattr(self._user32, "get_last_error", None)
        if not callable(getter):
            try:
                import ctypes

                return int(ctypes.get_last_error())
            except (AttributeError, OSError, TypeError, ValueError):
                return 0
        try:
            return int(getter() or 0)
        except (TypeError, ValueError, OverflowError):
            return 0

    def _emit(self, action: str, active: bool) -> None:
        try:
            self._callback(action, active)
        except Exception:
            # 回调最终通过 Qt Signal 进入主线程；回调异常不能杀掉消息线程。
            return

    def _key_is_down(self, key: str) -> bool:
        getter = getattr(self._user32, "GetAsyncKeyState", None)
        if not callable(getter):
            return False
        virtual_keys = _WIN32_MODIFIER_VIRTUAL_KEYS.get(key)
        if virtual_keys is None:
            virtual_keys = (_win32_virtual_key(key),)
        for virtual_key in virtual_keys:
            if not virtual_key:
                continue
            try:
                if self._return_code(getter(virtual_key)) & 0x8000:
                    return True
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
        return False

    def _poll_active(self) -> None:
        """通过 GetAsyncKeyState 收敛按键释放并抑制系统自动重复。"""

        for hotkey_id in tuple(self._active_ids):
            parts = self._registration_parts.get(hotkey_id)
            if parts is None:
                self._active_ids.discard(hotkey_id)
                self._active_holds.discard(hotkey_id)
                continue
            _virtual_key, sequence_parts = parts
            if all(self._key_is_down(part) for part in sequence_parts):
                continue
            self._active_ids.discard(hotkey_id)
            was_hold = hotkey_id in self._active_holds
            self._active_holds.discard(hotkey_id)
            if not was_hold:
                continue
            binding = next(
                (item for item_id, item in self._registrations if item_id == hotkey_id),
                None,
            )
            if binding is not None:
                self._emit(binding.action, False)

    def _dispatch_message(self, message: object) -> bool:
        message_type = _win32_message_value(message, "message", -1)
        if message_type == _WM_QUIT:
            return False
        if message_type != _WM_HOTKEY:
            return True
        hotkey_id = _win32_message_value(message, "wParam")
        binding = next(
            (item for item_id, item in self._registrations if item_id == hotkey_id),
            None,
        )
        if binding is None:
            return True
        if hotkey_id in self._active_ids:
            return True
        self._active_ids.add(hotkey_id)
        if binding.mode is HotkeyMode.HOLD:
            self._active_holds.add(hotkey_id)
            self._emit(binding.action, True)
        else:
            self._emit(binding.action, True)
        return True

    def _run(self) -> None:
        user32 = self._user32
        kernel32 = self._kernel32
        thread_id_getter = getattr(kernel32, "GetCurrentThreadId", None)
        self._thread_id = None
        if callable(thread_id_getter):
            try:
                self._thread_id = self._return_code(thread_id_getter())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                self._thread_id = None
        if not self._thread_id:
            try:
                self._thread_id = int(threading.get_native_id())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self._thread_id = None
        registrations: list[tuple[int, HotkeyBinding]] = []
        try:
            peek_message = getattr(user32, "PeekMessageW", None)
            register = getattr(user32, "RegisterHotKey", None)
            get_message = getattr(user32, "GetMessageW", None)
            unregister = getattr(user32, "UnregisterHotKey", None)
            post_thread_message = getattr(user32, "PostThreadMessageW", None)
            if not callable(register) or not callable(unregister):
                self.errors.append("RegisterHotKey or UnregisterHotKey is unavailable")
                self._startup_result = "unavailable", (), tuple(self.errors)
                return
            if not callable(peek_message) and not callable(get_message):
                self.errors.append("Windows message-loop API is unavailable")
                self._startup_result = "unavailable", (), tuple(self.errors)
                return
            if not callable(post_thread_message):
                self.errors.append("PostThreadMessageW is unavailable")
                self._startup_result = "unavailable", (), tuple(self.errors)
                return
            if any(binding.mode is HotkeyMode.HOLD for binding in self._bindings) and not callable(
                getattr(user32, "GetAsyncKeyState", None)
            ):
                self.errors.append("GetAsyncKeyState is required for hold hotkeys")
            message = wintypes.MSG()
            if callable(peek_message):
                # 触发当前线程消息队列初始化；RegisterHotKey(NULL, ...) 依赖该队列。
                try:
                    peek_message(ctypes.byref(message), None, 0, 0, 0)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    self.errors.append("Windows message queue initialization failed")
            for index, binding in enumerate(self._bindings, start=1):
                if binding.mode is HotkeyMode.HOLD and not callable(
                    getattr(user32, "GetAsyncKeyState", None)
                ):
                    continue
                try:
                    modifiers, virtual_key, sequence_parts = _win32_hotkey_parts(binding.sequence)
                except ValueError as exc:
                    self.errors.append(f"{binding.action}: {exc}")
                    continue
                try:
                    if binding.mode is HotkeyMode.PRESS:
                        modifiers |= _MOD_NOREPEAT
                    result = register(None, index, modifiers, virtual_key)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    self.errors.append(f"{binding.action}: RegisterHotKey failed: {exc}")
                    continue
                if self._return_code(result) == 0:
                    self.errors.append(
                        f"{binding.action}: RegisterHotKey failed with error {self._last_error()}"
                    )
                    continue
                registrations.append((index, binding))
                self._registration_parts[index] = (virtual_key, sequence_parts)
            self._registrations = registrations
            if not registrations:
                self._startup_result = (
                    "degraded" if self.errors else "unavailable",
                    (),
                    tuple(self.errors),
                )
                return
            actions = tuple(binding.action for _hotkey_id, binding in registrations)
            status = "available" if not self.errors else "degraded"
            self._startup_result = status, actions, tuple(self.errors)
            self._ready.set()
            while not self._stop.is_set():
                handled_message = False
                if callable(peek_message):
                    try:
                        while self._return_code(
                            peek_message(ctypes.byref(message), None, 0, 0, _PM_REMOVE)
                        ):
                            handled_message = True
                            if not self._dispatch_message(message):
                                self._stop.set()
                                break
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        self.errors.append("Windows message polling failed")
                        break
                elif callable(get_message):
                    try:
                        result = self._return_code(get_message(ctypes.byref(message), None, 0, 0))
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        self.errors.append("Windows message retrieval failed")
                        break
                    if result <= 0 or not self._dispatch_message(message):
                        break
                    handled_message = True
                if self._active_ids:
                    self._poll_active()
                if not handled_message:
                    time.sleep(0.01)
        finally:
            for hotkey_id, _binding in tuple(registrations):
                try:
                    unregister(None, hotkey_id)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
            self._active_ids.clear()
            self._active_holds.clear()
            self._ready.set()

    def start(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """注册 Windows 全局快捷键并启动消息线程。"""

        if self._thread is not None or self._registrations:
            self.close()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread_id = None
        self._registrations.clear()
        self._registration_parts.clear()
        self._active_ids.clear()
        self._active_holds.clear()
        self._startup_result = None
        self.errors.clear()
        if not self._is_windows:
            self.errors.append("Windows global hotkeys are unavailable on this platform")
            return "unavailable", (), tuple(self.errors)
        if self._user32 is None:
            self.errors.append("user32.dll is unavailable")
            return "unavailable", (), tuple(self.errors)
        self._thread = threading.Thread(target=self._run, name="meapet-win32-hotkeys", daemon=True)
        self._thread.start()
        if not self._ready.wait(self._startup_timeout):
            self.errors.append("Windows global hotkey startup timed out")
            self.close()
            return "degraded", (), tuple(self.errors)
        result = self._startup_result
        if result is None:
            result = "unavailable", (), tuple(self.errors)
        return result

    def close(self) -> None:
        """停止消息线程并反注册所有 Windows 全局快捷键。"""

        self._stop.set()
        thread = self._thread
        thread_id = self._thread_id
        post_thread_message = getattr(self._user32, "PostThreadMessageW", None)
        if thread is not None and thread.is_alive() and callable(post_thread_message) and thread_id:
            try:
                post_thread_message(thread_id, _WM_QUIT, 0, 0)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        self._thread_id = None
        self._registrations.clear()
        self._registration_parts.clear()
        self._active_ids.clear()
        self._active_holds.clear()


class _X11GlobalHotkey:
    """可选的 X11 根窗口快捷键抓取器。

    Wayland 不允许客户端读取全局按键，因此该实现只在明确的 X11 会话中
    尝试注册；依赖缺失、冲突或 DISPLAY 不可用都会返回错误而不阻塞桌宠。
    """

    _KEYSYM_ALIASES = {
        "SPACE": "space",
        "RETURN": "Return",
        "ENTER": "Return",
        "ESC": "Escape",
        "ESCAPE": "Escape",
        "PAGEUP": "Prior",
        "PAGEDOWN": "Next",
    }

    def __init__(
        self, bindings: Sequence[HotkeyBinding], callback: Callable[[str, bool], None]
    ) -> None:
        self._bindings = tuple(bindings)
        self._callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._display: object | None = None
        self._root: object | None = None
        self._registrations: list[tuple[int, int, HotkeyBinding]] = []
        self._active: set[tuple[str, str]] = set()
        self.errors: list[str] = []

    @staticmethod
    def _modifier_mask(parts: Sequence[str], x_module: object) -> int:
        values = {
            "CTRL": int(getattr(x_module, "ControlMask", 4)),
            "CONTROL": int(getattr(x_module, "ControlMask", 4)),
            "SHIFT": int(getattr(x_module, "ShiftMask", 1)),
            "ALT": int(getattr(x_module, "Mod1Mask", 8)),
            "META": int(getattr(x_module, "Mod4Mask", 64)),
            "SUPER": int(getattr(x_module, "Mod4Mask", 64)),
        }
        mask = 0
        for part in parts:
            mask |= values.get(part.upper(), 0)
        return mask

    def start(self) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        """注册根窗口按键并启动监听线程。"""

        if self._thread is not None or self._display is not None or self._registrations:
            self.close()
        self._stop = threading.Event()
        self._active.clear()
        self.errors.clear()
        if not self._bindings:
            return "unavailable", (), ()
        try:
            from Xlib import XK, X, display
        except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as exc:
            self.errors.append(f"python-xlib unavailable: {exc}")
            return "degraded", (), tuple(self.errors)
        try:
            connection = display.Display()
            root = connection.screen().root
        except Exception as exc:
            self.errors.append(f"X11 display unavailable: {exc}")
            return "degraded", (), tuple(self.errors)
        self._display = connection
        self._root = root
        for binding in self._bindings:
            parts = binding.sequence.split("+")
            key_name = self._KEYSYM_ALIASES.get(parts[-1].upper(), parts[-1])
            try:
                keysym = XK.string_to_keysym(key_name)
                keycode = int(connection.keysym_to_keycode(keysym))
                if keycode <= 0:
                    raise ValueError(f"unknown X11 key: {parts[-1]}")
                modifier = self._modifier_mask(parts[:-1], X)
                root.grab_key(
                    keycode,
                    modifier,
                    True,
                    X.GrabModeAsync,
                    X.GrabModeAsync,
                )
                self._registrations.append((keycode, modifier, binding))
            except Exception as exc:
                self.errors.append(f"{binding.action}: {exc}")
        if not self._registrations:
            try:
                connection.close()
            except (AttributeError, OSError, RuntimeError):
                pass
            self._display = None
            self._root = None
            return "degraded", (), tuple(self.errors)
        try:
            connection.flush()
        except (AttributeError, OSError, RuntimeError):
            self.errors.append("X11 key grab flush failed")
        self._thread = threading.Thread(target=self._run, name="meapet-x11-hotkeys", daemon=True)
        self._thread.start()
        status = "available" if not self.errors else "degraded"
        return (
            status,
            tuple(binding.action for _key, _modifier, binding in self._registrations),
            tuple(self.errors),
        )

    def _run(self) -> None:
        display = self._display
        if display is None:
            return
        try:
            event_types = {2, 3}  # X.KeyPress / X.KeyRelease
            while not self._stop.is_set():
                try:
                    if not display.pending_events():
                        select.select([display.fileno()], [], [], 0.2)
                    if not display.pending_events():
                        continue
                    event = display.next_event()
                except Exception:
                    # Xwayland/Xvfb 退出时 python-xlib 抛出
                    # ``ConnectionClosedError``；快捷键线程必须静默收尾，
                    # 不能在 Qt 退出阶段把异常打印成未处理线程回溯。
                    break
                if int(getattr(event, "type", -1)) not in event_types:
                    continue
                keycode = int(getattr(event, "detail", 0) or 0)
                state = int(getattr(event, "state", 0) or 0)
                pressed = int(getattr(event, "type", -1)) == 2
                for registered_keycode, modifier, binding in self._registrations:
                    if registered_keycode != keycode:
                        continue
                    if pressed:
                        # KeyPress must include the complete modifier mask.
                        if (state & modifier) != modifier:
                            continue
                        state_key = (binding.action, binding.sequence)
                        if state_key in self._active:
                            continue
                        self._active.add(state_key)
                    else:
                        # Window managers may report KeyRelease after a modifier
                        # was released. Match an active registration without
                        # requiring the modifier bits to remain in the event.
                        state_key = (binding.action, binding.sequence)
                        if state_key not in self._active:
                            continue
                        self._active.discard(state_key)
                    try:
                        self._callback(binding.action, pressed)
                    except Exception:
                        continue
        finally:
            root = self._root
            connection = self._display
            if root is not None:
                for keycode, modifier, _binding in self._registrations:
                    try:
                        root.ungrab_key(keycode, modifier)
                    except Exception:
                        pass
            if connection is not None:
                try:
                    connection.flush()
                except Exception:
                    pass
                try:
                    connection.close()
                except Exception:
                    pass

    def close(self) -> None:
        """停止监听并释放 X11 根窗口抓取。"""

        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.8)
        self._thread = None
        self._registrations.clear()
        self._display = None
        self._root = None


try:  # Qt 依赖保持可选；无 GUI 环境仍可加载配置与运行测试。
    from PySide6.QtCore import QEvent, QObject, Qt, Signal
    from PySide6.QtGui import QKeySequence, QShortcut
    from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit, QTextEdit, QWidget

    pyside6_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_available = False


if pyside6_available:

    class _HotkeyEventFilter(QObject):
        """为按住型绑定转发按下/释放事件。"""

        def __init__(self, manager: HotkeyManager) -> None:
            super().__init__()
            self._manager = manager

        def eventFilter(self, watched: object, event: object) -> bool:  # noqa: N802
            del watched
            event_type = getattr(event, "type", lambda: None)()
            if event_type == QEvent.Type.KeyPress and self._manager._handle_hover_enter(event):
                return True
            if event_type == QEvent.Type.KeyPress:
                return self._manager._handle_key_event(event, pressed=True)
            elif event_type == QEvent.Type.KeyRelease:
                return self._manager._handle_key_event(event, pressed=False)
            return False

    class HotkeyManager(QObject):
        """注册窗口级快捷键，并报告局部/降级能力。"""

        globalHotkey = Signal(str, bool)

        def __init__(
            self,
            config: HotkeyConfig,
            callbacks: Mapping[str, Callable[..., Any]] | None = None,
            *,
            backend: str = "unknown",
        ) -> None:
            super().__init__()
            self.config = config
            self.callbacks = dict(callbacks or {})
            self.backend = str(backend or "unknown").strip().lower()
            self._filter = _HotkeyEventFilter(self)
            # 使用动作+序列作为按住状态键。一个动作可以配置多个不同
            # 组合键，不能只按动作名去重，否则释放其中一个组合键会错误地
            # 结束同动作的另一个绑定。
            self._active_holds: dict[str, HotkeyBinding] = {}
            self._watched_targets: list[object] = []
            self._shortcuts: list[object] = []
            self._shortcut_actions: set[str] = set()
            self._shortcut_sequences: set[str] = set()
            self._registered: list[HotkeyBinding] = []
            self._errors: list[str] = []
            self._installed = False
            self._global_bridge: _X11GlobalHotkey | _Win32GlobalHotkey | None = None
            self._global_actions: set[str] = set()
            self._global_sequences: set[str] = set()
            self._hover_enter_callback: Callable[[], Any] | None = None
            self._hover_provider: Callable[[], bool] | None = None
            self.globalHotkey.connect(self._on_global_hotkey)

        def install(self, target: object | None = None) -> Mapping[str, object]:
            """安装绑定；返回可展示给控制台的能力摘要。"""

            if self._installed:
                self.close()
            self._errors.clear()
            if not self.config.enabled:
                return {"status": "unavailable", "reason": "hotkeys are disabled", "bindings": []}
            app = QApplication.instance()
            if not isinstance(app, QApplication):
                return {
                    "status": "unavailable",
                    "reason": "QApplication is not running",
                    "bindings": [],
                }
            watched: list[object] = []
            if target is not None:
                installer = getattr(target, "installEventFilter", None)
                if callable(installer):
                    watched.append(target)
                viewport_factory = getattr(target, "viewport", None)
                if callable(viewport_factory):
                    try:
                        viewport = viewport_factory()
                    except (AttributeError, RuntimeError, TypeError):
                        viewport = None
                    if viewport is not None and all(viewport is not item for item in watched):
                        if callable(getattr(viewport, "installEventFilter", None)):
                            watched.append(viewport)
                window_factory = getattr(target, "windowHandle", None)
                if callable(window_factory):
                    try:
                        window = window_factory()
                    except (AttributeError, RuntimeError, TypeError):
                        window = None
                    if window is not None and all(window is not item for item in watched):
                        if callable(getattr(window, "installEventFilter", None)):
                            watched.append(window)
            for watched_target in watched:
                try:
                    watched_target.installEventFilter(self._filter)
                    self._watched_targets.append(watched_target)
                except (AttributeError, RuntimeError, TypeError) as exc:
                    self._errors.append(f"local hotkey target: {exc}")
            global_status = "unavailable"
            global_actions: tuple[str, ...] = ()
            if self.backend == "x11":
                self._global_bridge = _X11GlobalHotkey(
                    self.config.bindings,
                    lambda action, active: self.globalHotkey.emit(action, active),
                )
                global_status, global_actions, global_errors = self._global_bridge.start()
                self._global_actions.update(global_actions)
                # ``start`` 保留动作名作为公开摘要；从真实注册表中再读取
                # 序列，避免同一动作配置多个组合键时把未注册的本地绑定
                # 一并吞掉。测试替身没有注册表时才回退到动作名匹配。
                registrations = getattr(self._global_bridge, "_registrations", ())
                for registration in registrations:
                    binding = registration[-1] if registration else None
                    if isinstance(binding, HotkeyBinding):
                        self._global_sequences.add(binding.sequence)
                if not self._global_sequences:
                    self._global_sequences.update(
                        binding.sequence
                        for binding in self.config.bindings
                        if binding.action in self._global_actions
                    )
                self._errors.extend(global_errors)
            elif self.backend == "windows":
                self._global_bridge = _Win32GlobalHotkey(
                    self.config.bindings,
                    lambda action, active: self.globalHotkey.emit(action, active),
                )
                global_status, global_actions, global_errors = self._global_bridge.start()
                self._global_actions.update(global_actions)
                registrations = getattr(self._global_bridge, "_registrations", ())
                for registration in registrations:
                    binding = registration[-1] if registration else None
                    if isinstance(binding, HotkeyBinding):
                        self._global_sequences.add(binding.sequence)
                if not self._global_sequences:
                    self._global_sequences.update(
                        binding.sequence
                        for binding in self.config.bindings
                        if binding.action in self._global_actions
                    )
                self._errors.extend(global_errors)
            local_bindings = tuple(
                binding
                for binding in self.config.bindings
                if binding.sequence not in self._global_sequences
            )
            if local_bindings and not self._watched_targets:
                self._errors.append("local hotkey target is unavailable")
            if isinstance(target, QWidget):
                for binding in self.config.bindings:
                    if (
                        binding.mode is not HotkeyMode.PRESS
                        or binding.sequence in self._global_sequences
                    ):
                        continue
                    try:
                        shortcut = QShortcut(QKeySequence(binding.sequence), target)
                        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
                        # ``press`` 语义是一次按下触发；Qt 默认允许长按自动
                        # 重复，可能把显示/隐藏和置顶动作连续切换多次。
                        set_auto_repeat = getattr(shortcut, "setAutoRepeat", None)
                        if callable(set_auto_repeat):
                            set_auto_repeat(False)
                        shortcut.activated.connect(
                            lambda action=binding.action: self._invoke(action)
                        )
                        self._shortcuts.append(shortcut)
                        self._shortcut_actions.add(binding.action)
                        self._shortcut_sequences.add(binding.sequence)
                    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                        self._errors.append(f"{binding.action}: {exc}")
            for binding in self.config.bindings:
                self._registered.append(binding)
            if not self._registered:
                status = "unavailable"
            else:
                has_local_path = bool(self._watched_targets or self._shortcuts)
                has_global_path = bool(self._global_sequences)
                if not has_local_path and not has_global_path:
                    status = "unavailable"
                else:
                    status = "available" if not self._errors else "degraded"
            # X11 根窗口抓取或 Windows RegisterHotKey 成功时，快捷键才是
            # 操作系统级能力。Wayland、未知后端以及全局抓取失败时，Qt 局部监听仍可用，
            # 但必须明确报告降级，不能让调用方误以为窗口失焦后仍可触发。
            if self._registered and self.backend != "x11":
                status = "degraded" if status == "available" else status
            elif (
                self._registered
                and self.backend in {"x11", "windows"}
                and global_status != "available"
            ):
                status = "degraded" if status == "available" else status
            if global_status == "available" and self._registered and not self._errors:
                status = "available"
            self._installed = True
            return {
                "status": status,
                "backend": self.backend,
                "bindings": [binding.action for binding in self._registered],
                "global_bindings": list(global_actions),
                "global_status": global_status,
                "local_bindings": [binding.action for binding in local_bindings],
                "errors": tuple(self._errors),
            }

        def install_hover_enter(
            self,
            callback: Callable[[], Any] | None,
            active_provider: Callable[[], bool] | None,
        ) -> None:
            """通过已安装的桌宠窗口过滤器处理悬停 Enter。

            应用级 Python 过滤器会让 PySide 包装 WebEngine 内部 QObject；
            Qt 6.11 的动态属性事件可在包装期间重入并崩溃，故复用窗口级监听。
            """

            self._hover_enter_callback = callback
            self._hover_provider = active_provider

        def _handle_hover_enter(self, event: object) -> bool:
            callback = self._hover_enter_callback
            provider = self._hover_provider
            if not self.config.enabled or not callable(callback) or not callable(provider):
                return False
            try:
                if bool(getattr(event, "isAutoRepeat", lambda: False)()):
                    return False
                if event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier:
                    return False
                if event.key() not in {Qt.Key.Key_Return, Qt.Key.Key_Enter} or not bool(provider()):
                    return False
                focus = QApplication.focusWidget()
                if isinstance(focus, (QLineEdit, QTextEdit, QPlainTextEdit)):
                    return False
                callback()
                return True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False

        def _on_global_hotkey(self, action: str, active: bool) -> None:
            """在 Qt 主线程消费 X11 线程发来的全局按键事件。"""

            if action in self._global_actions:
                bindings = tuple(
                    binding for binding in self.config.bindings if binding.action == action
                )
                if not active and all(binding.mode is HotkeyMode.PRESS for binding in bindings):
                    return
                if any(binding.mode is HotkeyMode.HOLD for binding in bindings):
                    self._invoke(action, active)
                else:
                    self._invoke(action)

        @staticmethod
        def _binding_state_key(binding: HotkeyBinding) -> str:
            """返回一个不会和另一个序列冲突的按住状态键。"""

            return f"{binding.action}\x00{binding.sequence}"

        @staticmethod
        def _event_key_name(event: object) -> str:
            """读取事件本身的键名，兼容释放修饰键时没有完整组合键的情况。"""

            try:
                value = QKeySequence(int(event.key())).toString(
                    QKeySequence.SequenceFormat.PortableText
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                return ""
            token = str(value or "").strip()
            if "+" in token:
                token = token.rsplit("+", 1)[-1]
            return {
                "CONTROL": "CTRL",
                "CTRL": "CTRL",
                "SUPER": "META",
                "META": "META",
            }.get(token.upper(), token.upper())

        def _event_sequence(self, event: object) -> str:
            try:
                combination = getattr(event, "keyCombination", None)
                if callable(combination):
                    value = combination()
                else:
                    value = event.modifiers() | event.key()
                text = QKeySequence(value).toString(QKeySequence.SequenceFormat.PortableText)
                return normalize_key_sequence(text)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                return ""

        def _handle_key_event(self, event: object, *, pressed: bool) -> bool:
            sequence = self._event_sequence(event)
            if pressed:
                if bool(getattr(event, "isAutoRepeat", lambda: False)()):
                    return False
                handled = False
                for binding in self.config.bindings:
                    if binding.sequence != sequence:
                        continue
                    handled = True
                    if binding.sequence in self._global_sequences:
                        continue
                    if binding.mode is HotkeyMode.PRESS:
                        if binding.sequence in self._shortcut_sequences:
                            continue
                        self._invoke(binding.action)
                    elif binding.mode is HotkeyMode.HOLD:
                        state_key = self._binding_state_key(binding)
                        if state_key not in self._active_holds:
                            self._active_holds[state_key] = binding
                            self._invoke(binding.action, True)
                return handled
            if not self._active_holds:
                return False
            # 释放修饰键后 Qt 事件通常不再包含完整序列。优先按完整序列
            # 匹配，失败时按实际释放的键匹配主键或任一修饰键；无关按键
            # 的释放不能结束桌宠当前的临时状态。
            active_keys: list[str] = []
            if sequence:
                active_keys.extend(
                    state_key
                    for state_key, binding in self._active_holds.items()
                    if binding.sequence == sequence
                )
            if not active_keys:
                event_key = self._event_key_name(event)
                if event_key:
                    active_keys.extend(
                        state_key
                        for state_key, binding in self._active_holds.items()
                        if event_key in {part.upper() for part in binding.sequence.split("+")}
                    )
            if active_keys:
                for state_key in active_keys:
                    binding = self._active_holds.pop(state_key, None)
                    if binding is not None:
                        self._invoke(binding.action, False)
                return True
            return False

        def _invoke(self, action: str, *args: object) -> object | None:
            callback = self.callbacks.get(action)
            if not callable(callback):
                return None
            try:
                return callback(*args)
            except TypeError:
                # 兼容不需要 active 参数的旧回调。按住动作的释放事件不能
                # 再次调用无参回调，否则“打开输入框”等动作会重复触发。
                if args and args != (False,):
                    try:
                        return callback()
                    except Exception:
                        return None
                return None
            except Exception:
                return None

        def close(self) -> None:
            """解除事件过滤器、快捷键和隐藏宿主。"""

            for binding in tuple(self._active_holds.values()):
                self._invoke(binding.action, False)
            self._active_holds.clear()
            for shortcut in tuple(self._shortcuts):
                try:
                    shortcut.setEnabled(False)
                    shortcut.deleteLater()
                except (AttributeError, RuntimeError, TypeError):
                    pass
            self._shortcuts.clear()
            self._shortcut_actions.clear()
            self._shortcut_sequences.clear()
            for watched_target in tuple(self._watched_targets):
                try:
                    watched_target.removeEventFilter(self._filter)
                except (AttributeError, RuntimeError, TypeError):
                    pass
            self._watched_targets.clear()
            if self._global_bridge is not None:
                self._global_bridge.close()
                self._global_bridge = None
            self._global_actions.clear()
            self._global_sequences.clear()
            self._hover_enter_callback = None
            self._hover_provider = None
            self._registered.clear()
            self._installed = False


else:

    class HotkeyManager:  # pragma: no cover
        """无 Qt 环境下的显式不可用实现。"""

        def __init__(
            self,
            config: HotkeyConfig,
            callbacks: Mapping[str, Callable[..., Any]] | None = None,
            *,
            backend: str = "unknown",
        ) -> None:
            self.config = config
            self.callbacks = dict(callbacks or {})
            self.backend = backend

        def install(self, target: object | None = None) -> Mapping[str, object]:
            del target
            return {
                "status": "unavailable",
                "backend": self.backend,
                "reason": "PySide6 is unavailable",
                "bindings": [],
                "global_bindings": [],
                "global_status": "unavailable",
                "local_bindings": [],
                "errors": ("PySide6 is unavailable",),
            }

        def close(self) -> None:
            return None


__all__ = [
    "DEFAULT_HOTKEYS",
    "HotkeyBinding",
    "HotkeyConfig",
    "HotkeyManager",
    "HotkeyMode",
    "normalize_key_sequence",
    "pyside6_available",
    "validate_hotkey_config",
]
