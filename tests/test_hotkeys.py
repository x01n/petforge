from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.runtime import _validate_runtime_values
from config.loader import ConfigurationError
from gui.qt6.hotkeys import HotkeyConfig, HotkeyMode, normalize_key_sequence


def test_hotkey_defaults_and_yaml_overrides() -> None:
    defaults = HotkeyConfig.defaults()
    assert defaults.by_action("open_input")[0].mode is HotkeyMode.HOLD
    assert defaults.by_action("toggle_window_lock")[0].sequence == "Ctrl+Shift+L"
    config = HotkeyConfig.from_mapping(
        {
            "enabled": True,
            "bindings": [
                {"action": "open_input", "sequence": "Ctrl+Return"},
                {"action": "restore_click_through", "sequence": "Ctrl+Alt+Space", "mode": "hold"},
            ],
        }
    )
    assert config.enabled
    assert config.by_action("open_input")[0].mode is HotkeyMode.PRESS
    assert config.by_action("restore_click_through")[0].mode is HotkeyMode.HOLD
    assert normalize_key_sequence("control+shift+m") == "Ctrl+Shift+M"


def test_hold_noarg_callback_runs_only_on_press() -> None:
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    from PySide6.QtWidgets import QApplication

    import gui.qt6.hotkeys as hotkeys

    app = QApplication.instance() or QApplication([])
    assert app is not None
    events: list[str] = []
    config = HotkeyConfig.from_mapping(
        {"bindings": [{"action": "open_input", "sequence": "Ctrl+Return", "mode": "hold"}]}
    )
    manager = hotkeys.HotkeyManager(config, {"open_input": lambda: events.append("open")})
    # 直接验证回调适配层，不依赖当前显示服务器是否允许窗口事件。
    manager._invoke("open_input", True)
    manager._invoke("open_input", False)
    assert events == ["open"]
    manager.close()


def test_hotkey_manager_does_not_create_widget_for_qgui_application(tmp_path: Path) -> None:
    """仅有 QGuiApplication 时构造快捷键管理器不能创建 QWidget 或触发 Qt abort。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtGui import QGuiApplication
from gui.qt6.hotkeys import HotkeyConfig, HotkeyManager

app = QGuiApplication([])
manager = HotkeyManager(HotkeyConfig.from_mapping({"bindings": []}))
assert manager.install()["status"] == "unavailable"
manager.close()
print("hotkey-qgui-lifecycle-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "hotkey-qgui-lifecycle-ok" in result.stdout


def test_hotkey_config_rejects_duplicate_and_modifier_only() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        HotkeyConfig.from_mapping(
            {
                "bindings": [
                    {"action": "open_input", "sequence": "Ctrl+M"},
                    {"action": "open_console", "sequence": "Ctrl+M"},
                ]
            }
        )
    with pytest.raises(ValueError, match="exactly one"):
        HotkeyConfig.from_mapping(
            {
                "bindings": [
                    {"action": "restore_click_through", "sequence": "Ctrl+Alt", "mode": "hold"}
                ]
            }
        )


def test_hotkey_config_explicit_empty_bindings_disables_actions() -> None:
    config = HotkeyConfig.from_mapping({"enabled": False, "bindings": []})
    assert not config.enabled
    assert config.bindings == ()


def test_win32_global_hotkey_registers_dispatches_and_unregisters_with_simulated_api() -> None:
    """Win32 消息线程使用 RegisterHotKey 和 WM_HOTKEY 完成一次 press。"""

    import threading
    import time

    import gui.qt6.hotkeys as hotkeys

    class FakeUser32:
        def __init__(self) -> None:
            self.events: list[tuple[int, int]] = []
            self.registered: list[tuple[object, int, int, int]] = []
            self.unregistered: list[tuple[object, int]] = []
            self.thread_message_calls: list[tuple[int, int]] = []
            self.lock = threading.Lock()

        def RegisterHotKey(self, hwnd, hotkey_id, modifiers, virtual_key):  # noqa: N802
            self.registered.append((hwnd, hotkey_id, modifiers, virtual_key))
            return 1

        def UnregisterHotKey(self, hwnd, hotkey_id):  # noqa: N802
            self.unregistered.append((hwnd, hotkey_id))
            return 1

        def PeekMessageW(self, message, hwnd, minimum, maximum, remove):  # noqa: N802
            del hwnd, minimum, maximum
            with self.lock:
                if not self.events:
                    return 0
                message._obj.message, message._obj.wParam = self.events.pop(0)
            del remove
            return 1

        def PostThreadMessageW(self, thread_id, message, w_param, l_param):  # noqa: N802
            del w_param, l_param
            self.thread_message_calls.append((thread_id, message))
            with self.lock:
                self.events.append((message, 0))
            return 1

        def push(self, message, w_param):
            with self.lock:
                self.events.append((message, w_param))

        def GetAsyncKeyState(self, virtual_key):  # noqa: N802
            del virtual_key
            return 0

    class FakeKernel32:
        def GetCurrentThreadId(self):  # noqa: N802
            return 9001

    binding = hotkeys.HotkeyConfig.from_mapping(
        {"bindings": [{"action": "open_console", "sequence": "Ctrl+Shift+M"}]}
    ).bindings[0]
    api = FakeUser32()
    events: list[tuple[str, bool]] = []
    bridge = hotkeys._Win32GlobalHotkey(
        (binding,),
        lambda action, active: events.append((action, active)),
        user32=api,
        kernel32=FakeKernel32(),
        is_windows=True,
    )

    assert bridge.start() == ("available", ("open_console",), ())
    assert api.registered == [(None, 1, 0x4000 | 0x0002 | 0x0004, ord("M"))]
    api.push(hotkeys._WM_HOTKEY, 1)
    deadline = time.monotonic() + 1.0
    while not events and time.monotonic() < deadline:
        time.sleep(0.01)
    assert events == [("open_console", True)]
    bridge.close()
    assert api.unregistered == [(None, 1)]
    assert api.thread_message_calls == [(9001, hotkeys._WM_QUIT)]


def test_win32_hold_release_uses_simulated_async_key_state() -> None:
    """WM_HOTKEY 只负责按下，释放由 GetAsyncKeyState 收敛。"""

    import gui.qt6.hotkeys as hotkeys

    class FakeUser32:
        def __init__(self) -> None:
            self.down = {0xA2, 0xA4, hotkeys._win32_virtual_key("SPACE")}

        def GetAsyncKeyState(self, virtual_key):  # noqa: N802
            return 0x8000 if virtual_key in self.down else 0

    binding = hotkeys.HotkeyConfig.from_mapping(
        {
            "bindings": [
                {
                    "action": "restore_click_through",
                    "sequence": "Ctrl+Alt+Space",
                    "mode": "hold",
                }
            ]
        }
    ).bindings[0]
    api = FakeUser32()
    events: list[tuple[str, bool]] = []
    bridge = hotkeys._Win32GlobalHotkey(
        (binding,),
        lambda action, active: events.append((action, active)),
        user32=api,
        is_windows=True,
    )
    bridge._registrations = [(1, binding)]
    bridge._registration_parts[1] = hotkeys._win32_hotkey_parts(binding.sequence)[1:]
    message = type("Message", (), {"message": hotkeys._WM_HOTKEY, "wParam": 1})()
    assert bridge._dispatch_message(message)
    assert events == [("restore_click_through", True)]
    api.down.clear()
    bridge._poll_active()
    assert events == [("restore_click_through", True), ("restore_click_through", False)]


def test_win32_global_hotkey_is_explicitly_unavailable_off_windows() -> None:
    import gui.qt6.hotkeys as hotkeys

    bridge = hotkeys._Win32GlobalHotkey((), lambda _action, _active: None, is_windows=False)
    status, actions, errors = bridge.start()
    assert status == "unavailable"
    assert actions == ()
    assert errors == ("Windows global hotkeys are unavailable on this platform",)


def test_hotkey_manager_windows_uses_global_bridge_without_local_duplicate(monkeypatch) -> None:
    """Windows 全局注册成功时，Qt 局部路径不得再次注册同一组合键。"""

    try:
        from PySide6.QtWidgets import QApplication, QWidget

        import gui.qt6.hotkeys as hotkeys
    except (ImportError, ModuleNotFoundError, OSError):
        return

    app = QApplication.instance() or QApplication([])
    events: list[str] = []
    config = hotkeys.HotkeyConfig.from_mapping(
        {"bindings": [{"action": "open_console", "sequence": "Ctrl+Shift+M"}]}
    )
    binding = config.bindings[0]

    class FakeGlobal:
        def __init__(self, bindings, callback):
            assert tuple(bindings) == (binding,)
            self.callback = callback
            self._registrations = [(1, binding)]

        def start(self):
            return "available", ("open_console",), ()

        def close(self):
            return None

    monkeypatch.setattr(hotkeys, "_Win32GlobalHotkey", FakeGlobal)
    manager = hotkeys.HotkeyManager(
        config,
        {"open_console": lambda: events.append("console")},
        backend="windows",
    )
    host = QWidget()
    host.show()
    status = manager.install(host)
    assert status["status"] == "available"
    assert status["global_status"] == "available"
    assert status["global_bindings"] == ["open_console"]
    assert status["local_bindings"] == []
    manager.globalHotkey.emit("open_console", True)
    manager.globalHotkey.emit("open_console", False)
    assert events == ["console"]
    manager.close()
    host.close()
    app.processEvents()


def test_x11_global_binding_is_consumed_without_duplicate_local_callback(monkeypatch) -> None:
    try:
        from PySide6.QtWidgets import QApplication, QWidget

        import gui.qt6.hotkeys as hotkeys
    except (ImportError, ModuleNotFoundError, OSError):
        return

    app = QApplication.instance() or QApplication([])
    assert app is not None
    events: list[object] = []

    class FakeGlobal:
        def __init__(self, bindings, callback):
            del bindings
            self.callback = callback

        def start(self):
            return "available", ("open_console",), ()

        def close(self):
            return None

    monkeypatch.setattr(hotkeys, "_X11GlobalHotkey", FakeGlobal)
    config = HotkeyConfig.from_mapping(
        {"bindings": [{"action": "open_console", "sequence": "Ctrl+Shift+M"}]}
    )
    manager = hotkeys.HotkeyManager(
        config,
        {"open_console": lambda: events.append("console")},
        backend="x11",
    )
    host = QWidget()
    host.show()
    assert manager.install(host)["global_bindings"] == ["open_console"]
    # QApplication 事件过滤器消费了本地事件；全局线程事件只在主线程信号
    # 交付时执行一次，释放事件不会再次触发 press 动作。
    manager.globalHotkey.emit("open_console", True)
    manager.globalHotkey.emit("open_console", False)
    assert events == ["console"]
    manager.close()
    host.close()


def test_x11_release_after_modifier_release_is_delivered() -> None:
    import gui.qt6.hotkeys as hotkeys

    config = HotkeyConfig.from_mapping(
        {
            "bindings": [
                {
                    "action": "restore_click_through",
                    "sequence": "Ctrl+Alt+Space",
                    "mode": "hold",
                }
            ]
        }
    )
    binding = config.bindings[0]
    events: list[tuple[str, bool]] = []

    class FakeRoot:
        def ungrab_key(self, keycode: int, modifier: int) -> None:
            del keycode, modifier

    class FakeDisplay:
        def __init__(self) -> None:
            self.events = [
                type("Event", (), {"type": 2, "detail": 42, "state": 1})(),
                type("Event", (), {"type": 3, "detail": 42, "state": 0})(),
            ]
            self.root = FakeRoot()

        def pending_events(self) -> bool:
            return bool(self.events)

        def next_event(self) -> object:
            return self.events.pop(0)

        def fileno(self) -> int:
            return 0

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    bridge = hotkeys._X11GlobalHotkey(
        (binding,),
        lambda action, active: (
            events.append((action, active)),
            bridge._stop.set() if not active else None,
        ),
    )
    display = FakeDisplay()
    bridge._display = display
    bridge._root = display.root
    bridge._registrations = [(42, 1, binding)]
    bridge._run()

    assert events == [("restore_click_through", True), ("restore_click_through", False)]


def test_x11_hold_bindings_with_same_action_keep_independent_state() -> None:
    """同一动作的两组组合键不能互相提前结束。"""

    import gui.qt6.hotkeys as hotkeys

    config = HotkeyConfig.from_mapping(
        {
            "bindings": [
                {"action": "restore_click_through", "sequence": "Ctrl+Space", "mode": "hold"},
                {"action": "restore_click_through", "sequence": "Alt+Space", "mode": "hold"},
            ]
        }
    )
    first, second = config.bindings
    events: list[tuple[str, bool]] = []

    class FakeRoot:
        def ungrab_key(self, keycode: int, modifier: int) -> None:
            del keycode, modifier

    class FakeDisplay:
        def __init__(self) -> None:
            self.events = [
                type("Event", (), {"type": 2, "detail": 42, "state": 4})(),
                type("Event", (), {"type": 2, "detail": 43, "state": 8})(),
                type("Event", (), {"type": 3, "detail": 42, "state": 0})(),
                type("Event", (), {"type": 3, "detail": 43, "state": 0})(),
            ]
            self.root = FakeRoot()

        def pending_events(self) -> bool:
            return bool(self.events)

        def next_event(self) -> object:
            return self.events.pop(0)

        def fileno(self) -> int:
            return 0

        def flush(self) -> None:
            return None

        def close(self) -> None:
            return None

    display = FakeDisplay()
    bridge: hotkeys._X11GlobalHotkey

    def callback(action: str, active: bool) -> None:
        events.append((action, active))
        if len(events) == 4:
            bridge._stop.set()

    bridge = hotkeys._X11GlobalHotkey((first, second), callback)
    bridge._display = display
    bridge._root = display.root
    bridge._registrations = [(42, 4, first), (43, 8, second)]
    bridge._run()

    assert events == [
        ("restore_click_through", True),
        ("restore_click_through", True),
        ("restore_click_through", False),
        ("restore_click_through", False),
    ]


def test_runtime_validation_covers_hotkeys_and_topmost() -> None:
    values = {"ui": {"always_on_top": False, "hotkeys": {"bindings": []}}}
    _validate_runtime_values(values)
    invalid = {"ui": {"hotkeys": {"bindings": [{"action": "open_input", "sequence": "Ctrl"}]}}}
    with pytest.raises(ConfigurationError, match="hotkey configuration"):
        _validate_runtime_values(invalid)


def test_hotkey_manager_offscreen_press_and_hold(tmp_path: Path) -> None:
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget
from gui.qt6.hotkeys import HotkeyConfig, HotkeyManager

app = QApplication.instance() or QApplication([])
events = []
config = HotkeyConfig.from_mapping({"bindings": [
    {"action": "open_console", "sequence": "Ctrl+Shift+M", "mode": "press"},
    {"action": "restore_click_through", "sequence": "Ctrl+Alt+Space", "mode": "hold"},
]})
manager = HotkeyManager(config, {
    "open_console": lambda: events.append("console"),
    "restore_click_through": lambda active: events.append(("hold", active)),
})
host = QWidget()
host.show()
assert manager.install(host)["status"] == "degraded"
assert any(item is host for item in manager._watched_targets)
modifiers = Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier
press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Space, modifiers)
release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key_Space, modifiers)
app.sendEvent(host, press)
app.sendEvent(host, release)
assert events == [("hold", True), ("hold", False)]
manager.close()
host.close()
app.processEvents()
print("hotkey-offscreen-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "hotkey-offscreen-ok" in result.stdout


def test_local_wayland_bindings_report_degraded_global_scope() -> None:
    try:
        from PySide6.QtWidgets import QApplication, QWidget

        import gui.qt6.hotkeys as hotkeys
    except (ImportError, ModuleNotFoundError, OSError):
        return

    app = QApplication.instance() or QApplication([])
    host = QWidget()
    host.show()
    config = HotkeyConfig.from_mapping(
        {"bindings": [{"action": "toggle_visibility", "sequence": "Ctrl+Shift+H"}]}
    )
    manager = hotkeys.HotkeyManager(
        config,
        {"toggle_visibility": lambda: None},
        backend="wayland",
    )
    status = manager.install(host)
    assert status["status"] == "degraded"
    assert status["global_status"] == "unavailable"
    assert status["local_bindings"] == ["toggle_visibility"]
    manager.close()
    host.close()
    app.processEvents()


def test_local_hotkeys_without_target_are_unavailable() -> None:
    try:
        from PySide6.QtWidgets import QApplication

        import gui.qt6.hotkeys as hotkeys
    except (ImportError, ModuleNotFoundError, OSError):
        return

    app = QApplication.instance() or QApplication([])
    config = HotkeyConfig.from_mapping(
        {"bindings": [{"action": "toggle_visibility", "sequence": "Ctrl+Shift+H"}]}
    )
    manager = hotkeys.HotkeyManager(config, {"toggle_visibility": lambda: None}, backend="wayland")
    status = manager.install()
    assert status["status"] == "unavailable"
    assert "local hotkey target is unavailable" in status["errors"]
    manager.close()
    app.processEvents()


def test_hold_release_ignores_unrelated_key_and_keeps_bindings_separate() -> None:
    try:
        from PySide6.QtCore import QEvent, Qt
        from PySide6.QtGui import QKeyEvent
        from PySide6.QtWidgets import QApplication, QWidget

        import gui.qt6.hotkeys as hotkeys
    except (ImportError, ModuleNotFoundError, OSError):
        return

    app = QApplication.instance() or QApplication([])
    events: list[tuple[str, bool]] = []
    config = HotkeyConfig.from_mapping(
        {
            "bindings": [
                {"action": "first", "sequence": "Ctrl+Space", "mode": "hold"},
                {"action": "second", "sequence": "Alt+Space", "mode": "hold"},
            ]
        }
    )
    manager = hotkeys.HotkeyManager(
        config,
        {
            "first": lambda active: events.append(("first", active)),
            "second": lambda active: events.append(("second", active)),
        },
    )
    host = QWidget()
    host.show()
    manager.install(host)

    def send(event_type: QEvent.Type, key: int, modifiers: Qt.KeyboardModifiers) -> None:
        app.sendEvent(host, QKeyEvent(event_type, key, modifiers))
        app.processEvents()

    send(QEvent.Type.KeyPress, Qt.Key_Space, Qt.ControlModifier)
    send(QEvent.Type.KeyPress, Qt.Key_Space, Qt.AltModifier)
    send(QEvent.Type.KeyRelease, Qt.Key_X, Qt.NoModifier)
    assert events == [("first", True), ("second", True)]
    send(QEvent.Type.KeyRelease, Qt.Key_Space, Qt.ControlModifier)
    assert events[-1] == ("first", False)
    send(QEvent.Type.KeyRelease, Qt.Key_Alt, Qt.NoModifier)
    assert events[-1] == ("second", False)
    manager.close()
    host.close()
    app.processEvents()


def test_hover_enter_is_scoped_to_pet_and_preserves_modified_keys(monkeypatch) -> None:
    """悬停 Enter 只消费桌宠按键，组合键、长按及其它窗口继续接收输入。"""

    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication, QWidget

    import gui.qt6.hotkeys as hotkeys

    app = QApplication.instance() or QApplication([])
    events: list[str] = []

    class Host(QWidget):
        def keyPressEvent(self, event):  # noqa: N802
            events.append("forwarded")
            event.accept()

    host, other = Host(), Host()
    manager = hotkeys.HotkeyManager(HotkeyConfig(bindings=()))
    hovering = [True]
    monkeypatch.setattr(QApplication, "focusWidget", lambda: None)
    manager.install(host)
    manager.install_hover_enter(lambda: events.append("open"), lambda: hovering[0])

    def send(target, modifiers=Qt.KeyboardModifier.NoModifier, *, repeat=False) -> None:
        app.sendEvent(
            target,
            QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, modifiers, "", repeat),
        )

    try:
        send(host)
        assert events == ["open"]
        send(other)
        send(host, Qt.KeyboardModifier.ControlModifier)
        send(host, repeat=True)
        hovering[0] = False
        send(host)
        assert events == ["open", "forwarded", "forwarded", "forwarded", "forwarded"]
        hovering[0] = True
        manager.close()
        send(host)
        assert events[-1] == "forwarded"
    finally:
        manager.close()
        host.close()
        other.close()


@pytest.mark.parametrize("editor_name", ["QLineEdit", "QTextEdit", "QPlainTextEdit"])
def test_hover_enter_preserves_editor_subclass_input(monkeypatch, editor_name: str) -> None:
    """自定义输入控件获得焦点时，悬停桌宠也不能截走 Enter。"""

    from PySide6 import QtWidgets
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    import gui.qt6.hotkeys as hotkeys

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    editor = type("CustomEditor", (getattr(QtWidgets, editor_name),), {})()
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", lambda: editor)
    events: list[str] = []
    manager = hotkeys.HotkeyManager(HotkeyConfig(bindings=()))
    manager.install_hover_enter(lambda: events.append("open"), lambda: True)
    try:
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier)
        assert manager._handle_hover_enter(event) is False
        assert events == []
    finally:
        manager.close()
        editor.close()
