from __future__ import annotations

import asyncio
import ctypes
import sys
from ctypes import wintypes
from types import SimpleNamespace

import gui.platforms.windows as windows_module
from app.runtime import build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from gui.platforms.factory import UnsupportedDesktopPlatform, create_desktop_platform
from gui.platforms.linux import LinuxDesktopPlatform
from gui.platforms.windows import (
    BACKEND_WINDOWS,
    WindowsDesktopPlatform,
    probe_windows_platform,
)


def test_platform_factory_selects_windows_linux_and_fail_closed_unknown() -> None:
    assert isinstance(create_desktop_platform(system="Windows"), WindowsDesktopPlatform)
    assert isinstance(create_desktop_platform(system="win32"), WindowsDesktopPlatform)
    assert isinstance(create_desktop_platform(system="Linux"), LinuxDesktopPlatform)
    assert isinstance(create_desktop_platform(system="linux"), LinuxDesktopPlatform)
    unknown = create_desktop_platform(system="Plan9")
    assert isinstance(unknown, UnsupportedDesktopPlatform)
    assert unknown.backend == "unsupported:plan9"


def test_windows_configures_last_input_info_pointer_abi() -> None:
    class Function:
        argtypes = None
        restype = None

        def __call__(self, *_args):
            return 1

    api = SimpleNamespace(GetLastInputInfo=Function())

    windows_module._configure_win32_api(api, "user32")

    assert api.GetLastInputInfo.argtypes == [ctypes.POINTER(windows_module._LastInputInfo)]
    assert api.GetLastInputInfo.restype is wintypes.BOOL


def test_windows_configures_cursor_position_pointer_abi() -> None:
    class Function:
        argtypes = None
        restype = None

        def __call__(self, *_args):
            return 1

    api = SimpleNamespace(GetCursorPos=Function())

    windows_module._configure_win32_api(api, "user32")

    assert api.GetCursorPos.argtypes == [ctypes.POINTER(windows_module._Point)]
    assert api.GetCursorPos.restype is wintypes.BOOL


def test_windows_cursor_position_uses_get_cursor_pos() -> None:
    class User32:
        def GetCursorPos(self, point):
            target = ctypes.cast(point, ctypes.POINTER(windows_module._Point)).contents
            target.x = -12
            target.y = 845
            return 1

    result = WindowsDesktopPlatform(is_windows=True, user32=User32()).cursor_position()
    assert result == {
        "status": "available",
        "backend": BACKEND_WINDOWS,
        "x": -12,
        "y": 845,
        "source": "win32-GetCursorPos",
    }


def test_windows_cursor_position_is_fail_closed_without_api() -> None:
    result = WindowsDesktopPlatform(is_windows=True, user32=SimpleNamespace()).cursor_position()
    assert result["status"] == "unavailable"
    assert "x" not in result
    assert "y" not in result


def test_build_runtime_uses_windows_factory_when_python_reports_win32(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    configuration = LoadedConfiguration(tmp_path / "config.yaml", {})
    runtime = build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    try:
        assert runtime.platform.backend == BACKEND_WINDOWS
        assert isinstance(runtime.platform._platform, WindowsDesktopPlatform)
        assert runtime.platform._platform._is_windows is True
    finally:
        asyncio.run(runtime.close())


def test_windows_probe_reports_win32_capabilities(monkeypatch) -> None:
    class User32:
        def GetForegroundWindow(self):
            return 1

        def GetLastInputInfo(self, _info):
            return 1

        def SetCursorPos(self, _x, _y):
            return 1

        def SendInput(self, count, _entries, _size):
            return count

        def mouse_event(self, *_args):
            return None

        def GetWindowLongPtrW(self, _window_id, _index):
            return 0

        def SetWindowLongPtrW(self, _window_id, _index, _value):
            return 0

        def RegisterHotKey(self, *_args):
            return 1

        def UnregisterHotKey(self, *_args):
            return 1

        def SetWindowPos(self, *_args):
            return 1

    class Kernel32:
        def GetTickCount64(self):
            return 1

        def CreateToolhelp32Snapshot(self, *_args):
            return 1

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(
            WindowType=SimpleNamespace(
                WindowTransparentForInput=1,
                WindowStaysOnTopHint=2,
            )
        )
    )
    monkeypatch.setattr(
        "gui.platforms.windows._import_optional",
        lambda name: (
            qt_core
            if name == "PySide6.QtCore"
            else object()
            if name in {"PySide6.QtGui", "psutil"}
            else None
        ),
    )
    snapshot = probe_windows_platform(user32=User32(), kernel32=Kernel32(), is_windows=True)
    assert snapshot.backend == BACKEND_WINDOWS
    assert snapshot.supports("active_window")
    assert snapshot.supports("process_context")
    assert snapshot.supports("click_through")
    assert snapshot.supports("always_on_top")
    assert snapshot.supports("overlay_position")
    assert snapshot.supports("screen_capture")
    assert snapshot.supports("input_control")
    assert snapshot.supports("system_idle")


def test_windows_active_window_and_idle_probe_use_user32(monkeypatch) -> None:
    class User32:
        def GetForegroundWindow(self):
            return 0x1234

        def GetWindowTextLengthW(self, _window_id):
            return 4

        def GetWindowTextW(self, _window_id, buffer, _size):
            buffer.value = "Demo"
            return 4

        def GetClassNameLengthW(self, _window_id):
            return 6

        def GetClassNameW(self, _window_id, buffer, _size):
            buffer.value = "Window"
            return 6

        def GetWindowThreadProcessId(self, _window_id, process_id):
            ctypes.cast(process_id, ctypes.POINTER(ctypes.c_ulong)).contents.value = 321
            return 1

        def GetWindowRect(self, _window_id, rect):
            target = ctypes.cast(rect, ctypes.POINTER(windows_module._Rect)).contents
            target.left = 10
            target.top = 20
            target.right = 210
            target.bottom = 140
            return 1

        def GetLastInputInfo(self, info):
            target = ctypes.cast(info, ctypes.POINTER(windows_module._LastInputInfo)).contents
            target.dwTime = 500
            return 1

    class Kernel32:
        def GetTickCount64(self):
            return 2_000

    monkeypatch.setattr("gui.platforms.windows._import_optional", lambda _name: None)
    platform = WindowsDesktopPlatform(is_windows=True, user32=User32(), kernel32=Kernel32())
    window = platform.active_window()
    assert window is not None
    assert window.window_id == "0x1234"
    assert window.title == "Demo"
    assert window.app_id == "Window"
    assert window.pid == 321
    assert window.geometry == (10, 20, 200, 120)
    assert platform.system_idle_seconds() == 1.5


def test_windows_click_move_and_shape_use_qt_and_win32_boundaries(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class Point:
        def __init__(self, x, y):
            self._x = x
            self._y = y

        def x(self):
            return self._x

        def y(self):
            return self._y

    class Window:
        def __init__(self):
            self._flags = 0
            self._position = Point(3, 4)
            self.mask = None

        def flags(self):
            return self._flags

        def setFlag(self, flag, enabled):
            self._flags = self._flags | flag if enabled else self._flags & ~flag
            calls.append(("flag", flag, enabled))

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = Point(x, y)
            calls.append(("move", x, y))

        def setMask(self, region):
            self.mask = region
            calls.append(("mask", region))

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(
            WindowType=SimpleNamespace(WindowTransparentForInput=1, WindowStaysOnTopHint=2)
        )
    )
    monkeypatch.setattr(
        "gui.platforms.windows._import_optional",
        lambda name: qt_core if name == "PySide6.QtCore" else None,
    )
    window = Window()
    through = WindowsDesktopPlatform(is_windows=True)
    through_result = through.set_click_through(window, True)
    assert through_result.available
    assert window.flags() & 1
    assert through.set_input_shape(window, {"shape": "model"}).available
    assert through.move_overlay(window, 40, 50).available
    assert window.pos().x() == 40
    assert window.pos().y() == 50
    assert any(call[0] == "mask" for call in calls)


def test_windows_click_at_uses_bounded_mouse_events(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    sent: list[list[tuple[int, int, int, int]]] = []

    class User32:
        def SetCursorPos(self, x, y):
            calls.append(("move", x, y))
            return 1

        def SendInput(self, count, entries, size):
            assert size == ctypes.sizeof(windows_module._Input)
            sent.append(
                [
                    (entry.type, entry.mi.dx, entry.mi.dy, entry.mi.dwFlags)
                    for entry in entries[:count]
                ]
            )
            return count

    platform = WindowsDesktopPlatform(is_windows=True, user32=User32())
    result = platform.click_at(12, 34, button="right")
    assert result == {
        "status": "completed",
        "backend": BACKEND_WINDOWS,
        "x": 12,
        "y": 34,
        "button": "right",
    }
    assert calls[0] == ("move", 12, 34)
    assert sent == [
        [
            (windows_module._INPUT_MOUSE, 0, 0, windows_module._MOUSEEVENTF_RIGHTDOWN),
            (windows_module._INPUT_MOUSE, 0, 0, windows_module._MOUSEEVENTF_RIGHTUP),
        ]
    ]


def test_windows_click_at_rejects_partial_sendinput_without_mouse_event_fallback() -> None:
    calls: list[tuple[object, ...]] = []
    sent: list[list[int]] = []

    class User32:
        def SetCursorPos(self, x, y):
            calls.append(("move", x, y))
            return 1

        def SendInput(self, count, entries, _size):
            sent.append([entry.mi.dwFlags for entry in entries[:count]])
            return count - 1 if len(sent) == 1 else count

        def mouse_event(self, *_args):
            raise AssertionError("mouse_event must not be used for a verified click")

    result = WindowsDesktopPlatform(is_windows=True, user32=User32()).click_at(5, 6)

    assert result == {
        "status": "unavailable",
        "reason": "Win32 click input was not fully accepted",
    }
    assert calls == [("move", 5, 6)]
    assert sent == [
        [windows_module._MOUSEEVENTF_LEFTDOWN, windows_module._MOUSEEVENTF_LEFTUP],
        [windows_module._MOUSEEVENTF_LEFTUP],
    ]


def test_windows_click_at_rejects_implicit_numeric_coercion() -> None:
    class User32:
        def SetCursorPos(self, *_args):
            raise AssertionError("invalid coordinates must not reach Win32")

    platform = WindowsDesktopPlatform(is_windows=True, user32=User32())

    assert platform.click_at(1.5, 2)["status"] == "unavailable"
    assert platform.click_at("1", 2)["status"] == "unavailable"
    assert platform.click_at(1, 2, button=None)["status"] == "unavailable"


def test_windows_automation_key_releases_main_key_after_partial_sendinput() -> None:
    sent: list[list[tuple[int, int]]] = []

    class User32:
        def SendInput(self, count, entries, _size):
            sent.append([(entry.ki.wVk, entry.ki.dwFlags) for entry in entries[:count]])
            return count - 1 if len(sent) == 1 else count

    result = WindowsDesktopPlatform(is_windows=True, user32=User32()).automation_batch(
        [{"type": "key", "key": "a"}]
    )

    assert result["status"] == "unavailable"
    assert sent == [
        [(ord("A"), 0), (ord("A"), windows_module._KEYEVENTF_KEYUP)],
        [(ord("A"), windows_module._KEYEVENTF_KEYUP)],
    ]


def test_windows_automation_batch_accepts_hex_window_id_from_tool_schema() -> None:
    calls: list[tuple[object, ...]] = []

    class User32:
        def SetForegroundWindow(self, hwnd):
            calls.append(("activate", hwnd))
            return 1

        def GetForegroundWindow(self):
            return 0x5B

    result = WindowsDesktopPlatform(is_windows=True, user32=User32()).automation_batch(
        [{"type": "activate_window", "window_id": "0x5b"}]
    )

    assert result["status"] == "completed"
    assert calls == [("activate", 0x5B)]


def test_windows_automation_batch_uses_sendinput_and_verifies_window_operations() -> None:
    sent: list[list[tuple[int, int, int, int]]] = []
    calls: list[tuple[object, ...]] = []

    class User32:
        def SendInput(self, count, entries, size):  # noqa: N802
            assert size == ctypes.sizeof(windows_module._Input)
            sent.append(
                [
                    (entry.type, entry.ki.wVk, entry.ki.wScan, entry.ki.dwFlags)
                    for entry in entries[:count]
                ]
            )
            return count

        def SetCursorPos(self, x, y):  # noqa: N802
            calls.append(("cursor", x, y))
            return 1

        def mouse_event(self, *args):
            calls.append(("mouse", *args))

        def SetForegroundWindow(self, hwnd):  # noqa: N802
            calls.append(("activate", hwnd))
            return 1

        def GetForegroundWindow(self):  # noqa: N802
            return 91

        def SetWindowPos(self, *args):  # noqa: N802
            calls.append(("move_window", *args))
            return 1

        def GetWindowRect(self, hwnd, rect):  # noqa: N802
            assert hwnd == 91
            target = ctypes.cast(rect, ctypes.POINTER(windows_module._Rect)).contents
            target.left, target.top, target.right, target.bottom = 12, 34, 112, 134
            return 1

    result = WindowsDesktopPlatform(is_windows=True, user32=User32()).automation_batch(
        [
            {"type": "key", "key": "a", "modifiers": ["ctrl"]},
            {"type": "text", "text": "你好"},
            {"type": "move_pointer", "x": -12, "y": 34},
            {"type": "click", "x": 12, "y": 34, "button": "right"},
            {"type": "activate_window", "window_id": 91},
            {"type": "move_window", "window_id": 91, "x": 12, "y": 34},
        ]
    )

    assert result["status"] == "completed"
    assert [entry["type"] for entry in result["steps"]] == [
        "key",
        "text",
        "move_pointer",
        "click",
        "activate_window",
        "move_window",
    ]
    assert sent[0] == [
        (windows_module._INPUT_KEYBOARD, 0x11, 0, 0),
        (windows_module._INPUT_KEYBOARD, ord("A"), 0, 0),
        (windows_module._INPUT_KEYBOARD, ord("A"), 0, windows_module._KEYEVENTF_KEYUP),
        (windows_module._INPUT_KEYBOARD, 0x11, 0, windows_module._KEYEVENTF_KEYUP),
    ]
    assert all(entry[1] == 0 and entry[3] & windows_module._KEYEVENTF_UNICODE for entry in sent[1])
    assert ("activate", 91) in calls
    assert any(item[0] == "move_window" for item in calls)


def test_windows_automation_batch_fails_closed_after_partial_step_failure() -> None:
    class User32:
        def SendInput(self, _count, _entries, _size):  # noqa: N802
            return 0

    result = WindowsDesktopPlatform(is_windows=True, user32=User32()).automation_batch(
        [{"type": "key", "key": "enter"}]
    )

    assert result["status"] == "unavailable"
    assert result["completed_steps"] == ()


def test_windows_automation_batch_rejects_unsafe_inputs_and_disabled_platform() -> None:
    class User32:
        def SendInput(self, count, _entries, _size):  # noqa: N802
            return count

        def SetCursorPos(self, _x, _y):  # noqa: N802
            raise AssertionError("invalid coordinates must not reach Win32")

    platform = WindowsDesktopPlatform(is_windows=True, user32=User32())
    assert (
        platform.automation_batch([{"type": "text", "text": "line\nbreak"}])["status"]
        == "unavailable"
    )
    assert (
        platform.automation_batch([{"type": "move_pointer", "x": 1_000_001, "y": 0}])["status"]
        == "unavailable"
    )
    assert (
        platform.automation_batch([{"type": "wait", "duration_ms": 2_001}])["status"]
        == "unavailable"
    )
    assert (
        platform.automation_batch([{"type": "wait", "duration_ms": 0}] * 17)["status"]
        == "unavailable"
    )
    assert (
        WindowsDesktopPlatform(is_windows=False, user32=User32()).automation_batch(
            [{"type": "key", "key": "enter"}]
        )["status"]
        == "unavailable"
    )


def test_windows_automation_batch_rejects_implicit_numeric_coercion() -> None:
    class User32:
        def SendInput(self, count, _entries, _size):  # noqa: N802
            return count

        def SetCursorPos(self, *_args):  # noqa: N802
            raise AssertionError("invalid automation input must not reach Win32")

        def SetWindowPos(self, *_args):  # noqa: N802
            raise AssertionError("invalid automation input must not reach Win32")

    platform = WindowsDesktopPlatform(is_windows=True, user32=User32())
    invalid_steps = (
        {"type": "wait", "duration_ms": "1"},
        {"type": "wait", "duration_ms": 1.5},
        {"type": "key", "key": "a", "repeat": "2"},
        {"type": "move_pointer", "x": 1.5, "y": 2},
        {"type": "move_pointer", "x": "1", "y": 2},
        {"type": "move_window", "window_id": 10, "x": 1.5, "y": 2},
        {"type": "move_window", "window_id": 10, "x": "1", "y": 2},
        {"type": "click", "x": 1, "y": 2, "button": None},
    )

    for step in invalid_steps:
        result = platform.automation_batch([step])
        assert result["status"] == "unavailable", step


def test_windows_platform_is_fail_closed_when_disabled(monkeypatch) -> None:
    """非 Windows 适配器不得因本机存在 Qt 或注入对象而执行副作用。"""

    class ExplodingAPI:
        def __getattr__(self, _name):
            raise AssertionError("disabled Windows adapter touched Win32 API")

    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("disabled Windows adapter imported optional dependency")
        ),
    )
    platform = WindowsDesktopPlatform(
        is_windows=False,
        user32=ExplodingAPI(),
        kernel32=ExplodingAPI(),
        gdi32=ExplodingAPI(),
    )
    snapshot = platform.probe()
    assert snapshot.details["is_windows"] is False
    assert all(item.state.value == "unavailable" for item in snapshot.capabilities)
    direct_snapshot = probe_windows_platform(
        user32=ExplodingAPI(),
        kernel32=ExplodingAPI(),
        gdi32=ExplodingAPI(),
        is_windows=False,
    )
    assert all(item.state.value == "unavailable" for item in direct_snapshot.capabilities)
    assert platform.active_window() is None
    assert platform.system_idle_seconds() is None
    assert platform.list_processes()["status"] == "unavailable"
    assert platform.capture_screen()["status"] == "unavailable"
    assert platform.click_at(1, 2)["status"] == "unavailable"
    assert platform.set_click_through(object(), True).state.value == "unavailable"
    assert platform.set_input_shape(object(), None).state.value == "unavailable"
    assert platform.always_on_top_status(object())["status"] == "unavailable"
    assert platform.move_overlay(object(), 1, 2).state.value == "unavailable"


def test_windows_probe_rejects_missing_qt_window_flags(monkeypatch) -> None:
    class User32:
        def GetForegroundWindow(self):
            return 1

        def GetLastInputInfo(self, _info):
            return 1

        def SetCursorPos(self, _x, _y):
            return 1

        def SendInput(self, count, _entries, _size):
            return count

        def mouse_event(self, *_args):
            return None

        def GetWindowLongPtrW(self, _hwnd, _index):
            return 0

        def SetWindowLongPtrW(self, _hwnd, _index, _value):
            return 0

        def SetWindowPos(self, *_args):
            return 1

        def RegisterHotKey(self, *_args):
            return 1

        def UnregisterHotKey(self, *_args):
            return 1

    class Kernel32:
        def GetTickCount64(self):
            return 1000

        def CreateToolhelp32Snapshot(self, *_args):
            return 1

    qt_core = SimpleNamespace(Qt=SimpleNamespace(WindowType=SimpleNamespace()))
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_core if name == "PySide6.QtCore" else None,
    )
    snapshot = probe_windows_platform(
        user32=User32(),
        kernel32=Kernel32(),
        is_windows=True,
    )
    assert not snapshot.supports("click_through")
    assert not snapshot.supports("always_on_top")


def test_windows_probe_accepts_injected_win32_apis_without_loading_dlls(monkeypatch) -> None:
    """能力探针可在非 Windows 主机用显式 Win32 替身验证。"""

    class User32:
        def GetForegroundWindow(self):
            return 1

        def GetLastInputInfo(self, _info):
            return 1

        def SetCursorPos(self, _x, _y):
            return 1

        def SendInput(self, count, _entries, _size):
            return count

        def mouse_event(self, *_args):
            return None

        def GetWindowLongPtrW(self, _hwnd, _index):
            return 0

        def SetWindowLongPtrW(self, _hwnd, _index, _value):
            return 0

        def SetWindowPos(self, *_args):
            return 1

        def RegisterHotKey(self, *_args):
            return 1

        def UnregisterHotKey(self, *_args):
            return 1

    class Kernel32:
        def GetTickCount64(self):
            return 1000

        def CreateToolhelp32Snapshot(self, *_args):
            return 1

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(
            WindowType=SimpleNamespace(
                WindowTransparentForInput=1,
                WindowStaysOnTopHint=2,
            )
        )
    )
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_core if name == "PySide6.QtCore" else None,
    )
    monkeypatch.setattr(
        windows_module,
        "_load_user32",
        lambda: (_ for _ in ()).throw(
            AssertionError("probe loaded instead of using injected user32")
        ),
    )
    snapshot = probe_windows_platform(
        user32=User32(),
        kernel32=Kernel32(),
        is_windows=True,
    )
    assert snapshot.supports("active_window")
    assert snapshot.supports("click_through")
    assert snapshot.supports("always_on_top")
    assert snapshot.supports("global_hotkeys")
    assert snapshot.supports("overlay_position")
    assert snapshot.supports("input_control")
    assert snapshot.supports("system_idle")


def test_windows_toolhelp_accepts_ctypes_handle_and_closes_snapshot(monkeypatch) -> None:
    """Toolhelp 枚举应兼容 ctypes 句柄，并始终关闭快照。"""

    class Kernel32:
        def __init__(self):
            self.closed = []
            self.index = 0

        def CreateToolhelp32Snapshot(self, _flags, _pid):
            return ctypes.c_void_p(123)

        def Process32FirstW(self, _handle, entry):
            target = ctypes.cast(entry, ctypes.POINTER(windows_module._ProcessEntry32W)).contents
            target.th32ProcessID = 456
            target.szExeFile = "demo.exe"
            return 1

        def Process32NextW(self, _handle, _entry):
            return 0

        def CloseHandle(self, handle):
            self.closed.append(handle)

    kernel = Kernel32()
    monkeypatch.setattr(windows_module, "_import_optional", lambda _name: None)
    platform = WindowsDesktopPlatform(is_windows=True, kernel32=kernel)
    result = platform.list_processes(limit=3)
    assert result == {
        "status": "available",
        "processes": [{"pid": 456, "name": "demo.exe", "executable": ""}],
    }
    assert len(kernel.closed) == 1


def test_windows_native_topmost_readback_uses_qwindow_handle(
    monkeypatch,
) -> None:
    """置顶回读应读取原生 QWindow 的 HWND 和 WS_EX_TOPMOST。"""

    calls: list[tuple[int, int]] = []

    class Handle:
        def __init__(self) -> None:
            self._flags = 0

        def winId(self):  # noqa: N802
            return ctypes.c_void_p(0x1_0000_0001)

        def flags(self):
            return self._flags

    class View:
        def __init__(self) -> None:
            self.native = Handle()

        def windowHandle(self):  # noqa: N802
            return self.native

    class User32:
        def GetWindowLongPtrW(self, hwnd, index):  # noqa: N802
            calls.append((int(hwnd), int(index)))
            return windows_module._WS_EX_TOPMOST

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(WindowType=SimpleNamespace(WindowStaysOnTopHint=2))
    )
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_core if name == "PySide6.QtCore" else None,
    )

    status = WindowsDesktopPlatform(is_windows=True, user32=User32()).always_on_top_status(View())

    assert status == {
        "status": "available",
        "enabled": True,
        "confirmed": True,
        "detail": "Win32 extended style confirms topmost state",
        "evidence": ("WS_EX_TOPMOST",),
    }
    assert calls == [(0x1_0000_0001, windows_module._GWL_EXSTYLE)]


def test_windows_native_click_through_targets_qwindow_and_preserves_position(
    monkeypatch,
) -> None:
    """QWidget 包装的 QWindow 切换输入透明时不应重建或丢失坐标。"""

    calls: list[tuple[object, ...]] = []

    class Point:
        def __init__(self, x: int, y: int) -> None:
            self._x = x
            self._y = y

        def x(self):
            return self._x

        def y(self):
            return self._y

    class Handle:
        def __init__(self) -> None:
            self._flags = 0

        def flags(self):
            return self._flags

        def setFlag(self, flag, enabled):  # noqa: N802
            self._flags = self._flags | flag if enabled else self._flags & ~flag
            calls.append(("native-flag", flag, bool(enabled)))

    class View:
        def __init__(self) -> None:
            self.native = Handle()
            self._position = Point(17, 23)
            self.visible = True

        def windowHandle(self):  # noqa: N802
            return self.native

        def isVisible(self):  # noqa: N802
            return self.visible

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = Point(int(x), int(y))
            calls.append(("widget-move", int(x), int(y)))

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(WindowType=SimpleNamespace(WindowTransparentForInput=1))
    )
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_core if name == "PySide6.QtCore" else None,
    )

    view = View()
    result = WindowsDesktopPlatform(is_windows=True).set_click_through(view, True)

    assert result.state is windows_module.CapabilityState.AVAILABLE
    assert view.native.flags() & 1
    assert (view.pos().x(), view.pos().y()) == (17, 23)
    assert calls == [("native-flag", 1, True)]
