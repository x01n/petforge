from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

import gui.platforms.windows as windows_module
from config import loader
from gui.platforms.windows import WindowsDesktopPlatform
from services import processes


def test_windows_toolhelp_closes_snapshot_when_first_probe_fails(monkeypatch) -> None:
    """Toolhelp 首次读取失败时仍必须关闭有效快照句柄。"""

    class Kernel32:
        def __init__(self) -> None:
            self.closed: list[object] = []

        def CreateToolhelp32Snapshot(self, _flags, _process_id):
            return ctypes.c_void_p(7)

        def Process32FirstW(self, _handle, _entry):
            return 0

        def Process32NextW(self, _handle, _entry):
            raise AssertionError("Process32NextW must not run after a failed first read")

        def CloseHandle(self, handle):
            self.closed.append(handle)

    kernel = Kernel32()
    monkeypatch.setattr(windows_module, "_import_optional", lambda _name: None)

    result = WindowsDesktopPlatform(is_windows=True, kernel32=kernel).list_processes(limit=2)

    assert result == {"status": "available", "processes": []}
    assert len(kernel.closed) == 1


def test_windows_toolhelp_rejects_invalid_snapshot_without_close(monkeypatch) -> None:
    """INVALID_HANDLE_VALUE 不是可关闭的快照句柄。"""

    class Kernel32:
        def __init__(self) -> None:
            self.closed: list[object] = []

        def CreateToolhelp32Snapshot(self, _flags, _process_id):
            return ctypes.c_void_p(-1)

        def Process32FirstW(self, _handle, _entry):
            raise AssertionError("invalid snapshot must stop before Process32FirstW")

        def Process32NextW(self, _handle, _entry):
            raise AssertionError("invalid snapshot must stop before Process32NextW")

        def CloseHandle(self, handle):
            self.closed.append(handle)

    kernel = Kernel32()
    monkeypatch.setattr(windows_module, "_import_optional", lambda _name: None)

    result = WindowsDesktopPlatform(is_windows=True, kernel32=kernel).list_processes(limit=2)

    assert result["status"] == "unavailable"
    assert result["processes"] == []
    assert kernel.closed == []


def test_windows_process_listing_filters_user_and_honors_limit(monkeypatch) -> None:
    """Windows 进程摘要只返回当前用户，并遵守有界数量。"""

    class Process:
        def __init__(self, info: dict[str, object]) -> None:
            self.info = info

    entries = [
        Process({"pid": 11, "name": "first.exe", "exe": "C:/first.exe", "username": "ALICE"}),
        Process({"pid": 12, "name": "other.exe", "exe": "C:/other.exe", "username": "Bob"}),
        Process({"pid": 13, "name": "second.exe", "exe": "C:/second.exe", "username": "alice"}),
    ]

    class Psutil:
        @staticmethod
        def process_iter(_fields):
            return iter(entries)

    class Getpass:
        @staticmethod
        def getuser():
            return "Alice"

    def optional(name: str):
        return {"psutil": Psutil, "getpass": Getpass}.get(name)

    monkeypatch.setattr(windows_module, "_import_optional", optional)
    result = WindowsDesktopPlatform(is_windows=True).list_processes(limit=1)

    assert result == {
        "status": "available",
        "processes": [{"pid": 11, "name": "first.exe", "executable": "C:/first.exe"}],
    }


def test_windows_process_listing_accepts_current_domain_and_upn_identity(monkeypatch) -> None:
    """域名/UPN 形式仅在当前环境域一致时归属于当前用户。"""

    class Process:
        def __init__(self, info: dict[str, object]) -> None:
            self.info = info

    entries = [
        Process(
            {
                "pid": 21,
                "name": "domain.exe",
                "exe": "C:/domain.exe",
                "username": "CONTOSO\\Alice",
            }
        ),
        Process(
            {
                "pid": 22,
                "name": "upn.exe",
                "exe": "C:/upn.exe",
                "username": "Alice@corp.example",
            }
        ),
        Process(
            {
                "pid": 23,
                "name": "foreign.exe",
                "exe": "C:/foreign.exe",
                "username": "EVIL\\Alice",
            }
        ),
        Process(
            {
                "pid": 24,
                "name": "other.exe",
                "exe": "C:/other.exe",
                "username": "CONTOSO\\Bob",
            }
        ),
    ]

    class Psutil:
        @staticmethod
        def process_iter(_fields):
            return iter(entries)

    class Getpass:
        @staticmethod
        def getuser():
            return "Alice"

    monkeypatch.setenv("USERDOMAIN", "CONTOSO")
    monkeypatch.setenv("USERDNSDOMAIN", "corp.example")
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: {"psutil": Psutil, "getpass": Getpass}.get(name),
    )

    result = windows_module._psutil_processes(10)

    assert result == [
        {"pid": 21, "name": "domain.exe", "executable": "C:/domain.exe"},
        {"pid": 22, "name": "upn.exe", "executable": "C:/upn.exe"},
    ]


def test_windows_process_listing_uses_adapter_environment_for_domain_filter(monkeypatch) -> None:
    """适配器显式环境中的域名应参与进程身份过滤。"""

    class Process:
        def __init__(self, info: dict[str, object]) -> None:
            self.info = info

    class Psutil:
        @staticmethod
        def process_iter(_fields):
            return iter(
                [
                    Process(
                        {
                            "pid": 31,
                            "name": "domain.exe",
                            "exe": "C:/domain.exe",
                            "username": "CONTOSO\\Alice",
                        }
                    ),
                    Process(
                        {
                            "pid": 32,
                            "name": "foreign.exe",
                            "exe": "C:/foreign.exe",
                            "username": "OTHER\\Alice",
                        }
                    ),
                ]
            )

    class Getpass:
        @staticmethod
        def getuser():
            return "Alice"

    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: {"psutil": Psutil, "getpass": Getpass}.get(name),
    )
    platform = WindowsDesktopPlatform(
        {"USERDOMAIN": "CONTOSO", "USERDNSDOMAIN": "corp.example"},
        is_windows=True,
    )

    assert platform.list_processes(limit=10) == {
        "status": "available",
        "processes": [{"pid": 31, "name": "domain.exe", "executable": "C:/domain.exe"}],
    }


def test_windows_process_listing_excludes_invalid_pid(monkeypatch) -> None:
    """进程摘要不应暴露缺失或非正数 PID。"""

    class Process:
        def __init__(self, info: dict[str, object]) -> None:
            self.info = info

    class Psutil:
        @staticmethod
        def process_iter(_fields):
            return iter(
                [
                    Process(
                        {
                            "pid": 0,
                            "name": "zero.exe",
                            "exe": "C:/zero.exe",
                            "username": "Alice",
                        }
                    ),
                    Process(
                        {
                            "pid": -1,
                            "name": "negative.exe",
                            "exe": "C:/negative.exe",
                            "username": "Alice",
                        }
                    ),
                    Process(
                        {
                            "pid": 25,
                            "name": "valid.exe",
                            "exe": "C:/valid.exe",
                            "username": "Alice",
                        }
                    ),
                ]
            )

    class Getpass:
        @staticmethod
        def getuser():
            return "Alice"

    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: {"psutil": Psutil, "getpass": Getpass}.get(name),
    )

    assert windows_module._psutil_processes(10) == [
        {"pid": 25, "name": "valid.exe", "executable": "C:/valid.exe"}
    ]


def test_windows_process_listing_fails_closed_when_username_is_unavailable(monkeypatch) -> None:
    """当前用户可知但进程身份缺失时，不能把未知进程列入桌面上下文。"""

    class Process:
        def __init__(self, info: dict[str, object]) -> None:
            self.info = info

    class Psutil:
        @staticmethod
        def process_iter(_fields):
            return iter(
                [
                    Process(
                        {
                            "pid": 41,
                            "name": "hidden.exe",
                            "exe": "C:/hidden.exe",
                            "username": None,
                        }
                    ),
                    Process(
                        {
                            "pid": 42,
                            "name": "empty.exe",
                            "exe": "C:/empty.exe",
                            "username": "",
                        }
                    ),
                    Process(
                        {
                            "pid": 43,
                            "name": "owned.exe",
                            "exe": "C:/owned.exe",
                            "username": "Alice",
                        }
                    ),
                ]
            )

    class Getpass:
        @staticmethod
        def getuser():
            return "Alice"

    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: {"psutil": Psutil, "getpass": Getpass}.get(name),
    )

    assert windows_module._psutil_processes(10) == [
        {"pid": 43, "name": "owned.exe", "executable": "C:/owned.exe"}
    ]


def test_windows_process_listing_rejects_implicit_limit_conversion() -> None:
    """平台边界不能把布尔值或字符串进程数量转换成整数。"""

    platform = WindowsDesktopPlatform(is_windows=True)
    for limit in (True, "10", 1.5):
        assert platform.list_processes(limit=limit) == {
            "status": "unavailable",
            "processes": [],
            "reason": "process limit is invalid",
        }


def test_windows_active_window_exclusion_is_fail_closed() -> None:
    """桌宠自身 HWND 被排除后不能继续暴露窗口上下文。"""

    class User32:
        @staticmethod
        def GetForegroundWindow():
            return 0x1234

    platform = WindowsDesktopPlatform(is_windows=True, user32=User32())
    platform.exclude_window_id(0x1234)

    assert platform.active_window() is None


def test_windows_capture_window_requires_explicit_id_before_qt(monkeypatch) -> None:
    """窗口截图必须先解析 HWND，不能在缺失 ID 时触碰 Qt。"""

    def unexpected_import(_name: str):
        raise AssertionError("Qt must not load before window id validation")

    monkeypatch.setattr(windows_module, "_import_optional", unexpected_import)

    result = WindowsDesktopPlatform(is_windows=True).capture_screen(scope="window")

    assert result == {
        "status": "unavailable",
        "reason": "window id must be resolved before Qt screen capture",
    }


def test_windows_capture_window_rejects_invalid_id_before_qt(monkeypatch) -> None:
    """窗口范围截图不得把 0 或非整数 HWND 解释为整屏截图。"""

    def unexpected_import(_name: str):
        raise AssertionError("invalid window id must be rejected before Qt capture")

    monkeypatch.setattr(windows_module, "_import_optional", unexpected_import)
    platform = WindowsDesktopPlatform(is_windows=True)

    for window_id in (0, -1, True, "0x10", 1.5):
        assert platform.capture_screen(scope="window", window_id=window_id) == {
            "status": "unavailable",
            "reason": "active window id is invalid",
        }


def test_windows_region_capture_rejects_implicit_numeric_coercion(monkeypatch) -> None:
    """区域截图不能把字符串或布尔值静默转换成坐标。"""

    def unexpected_import(_name: str):
        raise AssertionError("invalid capture region must be rejected before Qt capture")

    monkeypatch.setattr(windows_module, "_import_optional", unexpected_import)
    platform = WindowsDesktopPlatform(is_windows=True)

    assert platform.capture_screen(
        scope="region", region={"x": True, "y": 0, "width": 10, "height": 10}
    ) == {
        "status": "unavailable",
        "reason": "capture region values must be integers",
    }
    assert platform.capture_screen(
        scope="region", region={"x": 0, "y": 0, "width": "10", "height": 10}
    ) == {
        "status": "unavailable",
        "reason": "capture region values must be integers",
    }


def test_windows_window_capture_uses_hwnd_screen_and_returns_matching_origin(monkeypatch) -> None:
    """窗口截图必须按已解析 HWND 选择屏幕，并返回同一窗口的根坐标。"""

    class Buffer:
        def __init__(self) -> None:
            self.payload = b""

        def open(self, _mode):
            return True

        def data(self):
            return self.payload

    class Image:
        def width(self):
            return 20

        def height(self):
            return 30

        def save(self, buffer, _format):
            buffer.payload = b"PNG"
            return True

    class Screen:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def grabWindow(self, *args):  # noqa: N802
            self.calls.append(args)
            return Image()

    primary = Screen()
    secondary = Screen()

    class Window:
        def winId(self):  # noqa: N802
            return 0x123

        def screen(self):
            return secondary

    class Application:
        @staticmethod
        def instance():
            return application

        def primaryScreen(self):  # noqa: N802
            return primary

        def allWindows(self):  # noqa: N802
            return (Window(),)

    class User32:
        def GetWindowRect(self, hwnd, rect):  # noqa: N802
            assert hwnd == 0x123
            target = ctypes.cast(rect, ctypes.POINTER(windows_module._Rect)).contents
            target.left, target.top, target.right, target.bottom = 1600, 80, 1620, 110
            return 1

    qt_gui = SimpleNamespace(QGuiApplication=Application)
    qt_core = SimpleNamespace(
        QBuffer=Buffer,
        QIODevice=SimpleNamespace(OpenModeFlag=SimpleNamespace(WriteOnly=1)),
    )
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_gui if name == "PySide6.QtGui" else qt_core,
    )
    application = Application()

    result = WindowsDesktopPlatform(is_windows=True, user32=User32()).capture_screen(
        scope="window", window_id=0x123
    )

    assert result["status"] == "available"
    assert result["origin"] == {"x": 1600, "y": 80}
    assert result["width"] == 20
    assert result["height"] == 30
    assert primary.calls == []
    assert secondary.calls == [(0x123,)]


def test_windows_capture_rejects_empty_qt_image(monkeypatch) -> None:
    """Qt 返回空图像时不能伪造可用于 OCR 的截图。"""

    class Image:
        def width(self):
            return 0

        def height(self):
            return 0

    class Screen:
        def grabWindow(self, *_args):  # noqa: N802
            return Image()

    class Application:
        @staticmethod
        def instance():
            return application

        def primaryScreen(self):  # noqa: N802
            return Screen()

    qt_gui = SimpleNamespace(QGuiApplication=Application)
    qt_core = SimpleNamespace()
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_gui if name == "PySide6.QtGui" else qt_core,
    )
    application = Application()

    assert WindowsDesktopPlatform(is_windows=True).capture_screen() == {
        "status": "unavailable",
        "reason": "captured image is empty",
    }


def test_windows_region_capture_maps_negative_virtual_desktop_coordinates(
    monkeypatch,
) -> None:
    """区域截图应选择包含全局坐标的屏幕并转换为局部坐标。"""

    class Buffer:
        def __init__(self) -> None:
            self.payload = b""

        def open(self, _mode):
            return True

        def data(self):
            return self.payload

    class Image:
        def __init__(self, width: int, height: int) -> None:
            self._width = width
            self._height = height

        def save(self, buffer, _format):
            buffer.payload = b"PNG"
            return True

        def width(self):
            return self._width

        def height(self):
            return self._height

        def copy(self, *_args):
            raise AssertionError("known-screen capture must not crop a primary-screen image")

    class Geometry:
        def __init__(self, x: int, y: int, width: int, height: int) -> None:
            self._x = x
            self._y = y
            self._width = width
            self._height = height

        def x(self):
            return self._x

        def y(self):
            return self._y

        def width(self):
            return self._width

        def height(self):
            return self._height

    class Screen:
        def __init__(self, geometry: Geometry) -> None:
            self._geometry = geometry
            self.calls: list[tuple[object, ...]] = []

        def geometry(self):
            return self._geometry

        def grabWindow(self, *args):  # noqa: N802
            self.calls.append(args)
            return Image(int(args[-2]), int(args[-1]))

    primary = Screen(Geometry(0, 0, 1920, 1080))
    left = Screen(Geometry(-1280, 0, 1280, 1024))

    class Application:
        @staticmethod
        def instance():
            return application

        def primaryScreen(self):  # noqa: N802
            return primary

        def screens(self):
            return (primary, left)

    qt_gui = SimpleNamespace(QGuiApplication=Application)
    qt_core = SimpleNamespace(
        QBuffer=Buffer,
        QIODevice=SimpleNamespace(OpenModeFlag=SimpleNamespace(WriteOnly=1)),
    )
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_gui if name == "PySide6.QtGui" else qt_core,
    )
    application = Application()

    result = WindowsDesktopPlatform(is_windows=True).capture_screen(
        scope="region",
        region={"x": -1200, "y": 100, "width": 200, "height": 300},
    )

    assert result["status"] == "available"
    assert result["scope"] == "region"
    assert left.calls == [(0, 80, 100, 200, 300)]
    assert primary.calls == []


def test_windows_region_capture_rejects_cross_screen_rectangle(monkeypatch) -> None:
    """跨屏区域不应被错误地当作主屏局部坐标截图。"""

    class Screen:
        def geometry(self):
            return SimpleNamespace(x=lambda: 0, y=lambda: 0, width=lambda: 100, height=lambda: 100)

        def grabWindow(self, *_args):  # noqa: N802
            raise AssertionError("cross-screen capture must fail before grabWindow")

    screen = Screen()

    class Application:
        @staticmethod
        def instance():
            return application

        def primaryScreen(self):  # noqa: N802
            return screen

        def screens(self):
            return (screen,)

    qt_gui = SimpleNamespace(QGuiApplication=Application)
    qt_core = SimpleNamespace(
        QBuffer=object,
        QIODevice=SimpleNamespace(OpenModeFlag=SimpleNamespace(WriteOnly=1)),
    )
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_gui if name == "PySide6.QtGui" else qt_core,
    )
    application = Application()

    result = WindowsDesktopPlatform(is_windows=True).capture_screen(
        scope="region",
        region={"x": 80, "y": 0, "width": 30, "height": 20},
    )

    assert result == {
        "status": "unavailable",
        "reason": "capture region must fit within a single screen",
    }


def test_windows_click_validation_does_not_call_user32(monkeypatch) -> None:
    """布尔值和越界坐标在输入 API 之前拒绝。"""

    calls: list[tuple[object, ...]] = []

    class User32:
        def SetCursorPos(self, *args):
            calls.append(("move", *args))
            return 1

        def mouse_event(self, *args):
            calls.append(("mouse", *args))

    platform = WindowsDesktopPlatform(is_windows=True, user32=User32())

    assert platform.click_at(True, 1)["status"] == "unavailable"
    assert platform.click_at(-1_000_001, 1)["status"] == "unavailable"
    assert calls == []


def test_windows_move_overlay_uses_set_window_pos_fallback_and_readback() -> None:
    """没有 Qt 移动方法时，Win32 SetWindowPos 路径必须有几何回读。"""

    class View:
        def __init__(self) -> None:
            self.position = (0, 0)

        def winId(self):  # noqa: N802
            return 0x100000001

    class User32:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def SetWindowPos(self, hwnd, insert_after, x, y, width, height, flags):  # noqa: N802
            self.calls.append((hwnd, insert_after, x, y, width, height, flags))
            view.position = (x, y)
            return 1

        def GetWindowRect(self, _hwnd, rect):  # noqa: N802
            target = ctypes.cast(rect, ctypes.POINTER(windows_module._Rect)).contents
            target.left, target.top = view.position
            target.right = target.left + 20
            target.bottom = target.top + 20
            return 1

    view = View()
    user32 = User32()
    result = WindowsDesktopPlatform(is_windows=True, user32=user32).move_overlay(view, -20, 30)

    assert result.available
    assert view.position == (-20, 30)
    assert user32.calls == [
        (
            0x100000001,
            0,
            -20,
            30,
            0,
            0,
            windows_module._SWP_NOSIZE
            | windows_module._SWP_NOZORDER
            | windows_module._SWP_NOACTIVATE
            | windows_module._SWP_NOOWNERZORDER,
        )
    ]


def test_windows_move_overlay_rejects_boolean_coordinates() -> None:
    """窗口位置坐标不能把布尔值解释为 0/1 像素。"""

    platform = WindowsDesktopPlatform(is_windows=True)
    result = platform.move_overlay(object(), True, 20)

    assert result.state.value == "unavailable"
    assert result.detail == "Windows position coordinates are invalid"


def test_windows_spawn_flags_fail_closed_when_creation_flag_missing(monkeypatch) -> None:
    """缺少 CREATE_NEW_PROCESS_GROUP 时不能静默创建无隔离 worker。"""

    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(processes.subprocess, "CREATE_NEW_PROCESS_GROUP", None, raising=False)

    try:
        processes.process_group_spawn_kwargs()
    except RuntimeError as exc:
        assert str(exc) == "Windows process group creation is unavailable"
    else:
        raise AssertionError("missing Windows process-group flag must fail closed")


def test_windows_taskkill_rejects_nonpositive_pid_without_spawning(monkeypatch) -> None:
    """无效 PID 不能触发 taskkill，避免误伤无关进程。"""

    calls: list[object] = []
    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(
        processes.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert processes._windows_taskkill_tree(0, timeout=0.1) is False
    assert calls == []


def test_windows_topmost_uses_32bit_get_window_long_fallback(monkeypatch) -> None:
    """32 位 Windows 没有 GetWindowLongPtrW 时仍能回读置顶状态。"""

    class Handle:
        def winId(self):  # noqa: N802
            return 44

        def flags(self):
            return 0

    class View:
        def windowHandle(self):  # noqa: N802
            return Handle()

    class User32:
        def GetWindowLongW(self, hwnd, index):  # noqa: N802
            assert hwnd == 44
            assert index == windows_module._GWL_EXSTYLE
            return 0

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(WindowType=SimpleNamespace(WindowStaysOnTopHint=2))
    )
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_core if name == "PySide6.QtCore" else None,
    )

    result = WindowsDesktopPlatform(is_windows=True, user32=User32()).always_on_top_status(View())

    assert result == {
        "status": "available",
        "enabled": False,
        "confirmed": True,
        "detail": "Win32 extended style confirms topmost state",
        "evidence": ("WS_EX_TOPMOST",),
    }


def test_windows_click_through_native_style_fallback_preserves_baseline(monkeypatch) -> None:
    """Qt 标志无法回读时，Win32 输入样式应可启用并恢复原始位。"""

    class Handle:
        def winId(self):  # noqa: N802
            return 77

    class View:
        def windowHandle(self):  # noqa: N802
            return Handle()

    class User32:
        def __init__(self) -> None:
            self.style = 0x100
            self.calls: list[tuple[object, ...]] = []

        def GetWindowLongPtrW(self, hwnd, index):  # noqa: N802
            self.calls.append(("get", hwnd, index))
            return self.style

        def SetWindowLongPtrW(self, hwnd, index, value):  # noqa: N802
            self.calls.append(("set", hwnd, index, value))
            self.style = int(value)
            return self.style

        def SetWindowPos(self, *args):  # noqa: N802
            self.calls.append(("pos", *args))
            return 1

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(WindowType=SimpleNamespace(WindowTransparentForInput=1))
    )
    monkeypatch.setattr(
        windows_module,
        "_import_optional",
        lambda name: qt_core if name == "PySide6.QtCore" else None,
    )
    user32 = User32()
    platform = WindowsDesktopPlatform(is_windows=True, user32=user32)

    enabled = platform.set_click_through(View(), True)
    assert enabled.available
    assert user32.style & windows_module._WS_EX_TRANSPARENT
    assert user32.style & windows_module._WS_EX_NOACTIVATE

    disabled = platform.set_click_through(View(), False)
    assert disabled.available
    assert user32.style == 0x100


def test_windows_set_always_on_top_uses_native_z_order_and_readback() -> None:
    """Windows 置顶接口应使用 SetWindowPos 并验证 WS_EX_TOPMOST。"""

    class View:
        def winId(self):  # noqa: N802
            return 88

    class User32:
        def __init__(self) -> None:
            self.style = 0
            self.calls: list[tuple[object, ...]] = []

        def GetWindowLongPtrW(self, _hwnd, _index):  # noqa: N802
            return self.style

        def SetWindowPos(self, hwnd, insert_after, *_args):  # noqa: N802
            self.calls.append((hwnd, insert_after))
            self.style = windows_module._WS_EX_TOPMOST if insert_after == -1 else 0
            return 1

    user32 = User32()
    platform = WindowsDesktopPlatform(is_windows=True, user32=user32)
    result = platform.set_always_on_top(View(), True)

    assert result.available
    assert user32.calls == [(88, windows_module._HWND_TOPMOST)]


def test_windows_user_configuration_and_data_paths_use_appdata() -> None:
    """原生 Windows 配置和数据目录都应落在 APPDATA 下。"""

    environment = {"APPDATA": "C:/Users/demo/AppData/Roaming"}
    home = loader.Path("C:/Users/demo")

    configuration_paths = loader._user_configuration_paths(environment=environment, home=home)
    data_path = loader._user_data_directory(environment=environment, home=home)

    assert (
        configuration_paths[0]
        == loader.Path("C:/Users/demo/AppData/Roaming/MeaPet/config.yaml").resolve()
    )
    assert data_path == loader.Path("C:/Users/demo/AppData/Roaming/MeaPet/data").resolve()


def test_windows_environment_expansion_reuses_userprofile_as_home(
    monkeypatch,
) -> None:
    """配置写入前的展开路径必须与启动加载共享 USERPROFILE 兼容。"""

    monkeypatch.setattr(loader.sys, "platform", "win32")

    expanded = loader.expand_environment_values(
        {"asr": {"model_path": "${HOME}/.cache/modelscope"}},
        environment={"USERPROFILE": "C:/Users/demo"},
    )

    assert expanded == {"asr": {"model_path": "C:/Users/demo/.cache/modelscope"}}


def test_windows_environment_expansion_replaces_empty_home_with_userprofile(
    monkeypatch,
) -> None:
    """空的 HOME 也不能覆盖 Windows 的 USERPROFILE。"""

    monkeypatch.setattr(loader.sys, "platform", "win32")

    expanded = loader.expand_environment_values(
        {"path": "${HOME}/MeaPet"},
        environment={"HOME": "", "USERPROFILE": "C:/Users/demo"},
    )

    assert expanded == {"path": "C:/Users/demo/MeaPet"}


@pytest.mark.parametrize("rebuild", [False, True])
@pytest.mark.parametrize("noactivate", [False, True])
def test_windows_click_through_refreshes_hwnd_after_qt_flag_rebuild(
    monkeypatch, rebuild: bool, noactivate: bool
) -> None:
    """同时验证 Qt 标志、重建后的 HWND 和恢复非激活窗口原有样式。"""

    baseline = 0x100 | (windows_module._WS_EX_NOACTIVATE if noactivate else 0)
    styles = {77: baseline}
    writes = []
    monkeypatch.setattr(windows_module, "_native_input_style_baselines", {})

    class Window:
        hwnd = 77
        qt_flags = 0

        def winId(self):  # noqa: N802
            return self.hwnd

        def flags(self):
            return self.qt_flags

        def setFlag(self, flag, enabled):  # noqa: N802
            self.qt_flags = self.qt_flags | flag if enabled else self.qt_flags & ~flag
            previous_style = styles[self.hwnd]
            if rebuild:
                del styles[self.hwnd]
                self.hwnd += 1
            styles[self.hwnd] = (
                previous_style | windows_module._WS_EX_TRANSPARENT
                if enabled
                else previous_style & ~windows_module._WS_EX_TRANSPARENT
            )

    class User32:
        def GetWindowLongPtrW(self, hwnd, _index):  # noqa: N802
            assert hwnd in styles, "read from destroyed HWND"
            return styles[hwnd]

        def SetWindowLongPtrW(self, hwnd, _index, value):  # noqa: N802
            assert hwnd in styles, "write to destroyed HWND"
            writes.append(hwnd)
            styles[hwnd] = value
            return value

        def SetWindowPos(self, *_args):  # noqa: N802
            return 1

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(WindowType=SimpleNamespace(WindowTransparentForInput=1))
    )
    monkeypatch.setattr(windows_module, "_import_optional", lambda name: qt_core)
    window, user32 = Window(), User32()
    platform = WindowsDesktopPlatform(is_windows=True, user32=user32)

    assert platform.set_click_through(window, True).available
    assert window.qt_flags == 1
    assert styles[window.hwnd] & windows_module._WS_EX_TRANSPARENT
    assert platform.set_click_through(window, False).available
    assert window.qt_flags == 0
    assert styles[window.hwnd] == baseline
    assert writes == ([78, 79] if rebuild else [77, 77])
    assert windows_module._native_input_style_baselines == {}


def test_windows_noactivate_alone_is_not_click_through(monkeypatch) -> None:
    """桌宠禁止抢焦点不代表禁止点击；关闭恢复不得因此报告失败。"""

    monkeypatch.setattr(windows_module, "_native_input_style_baselines", {})
    user32 = SimpleNamespace(GetWindowLongPtrW=lambda hwnd, index: windows_module._WS_EX_NOACTIVATE)
    accepted, _ = windows_module._set_native_input_transparency(77, False, user32=user32)
    assert accepted is True
