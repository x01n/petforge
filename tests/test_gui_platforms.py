from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from gui.platforms import CapabilityState
from gui.platforms import linux as linux_platform
from gui.qt6.app import (
    _click_through_operation_succeeded,
    _configure_qt_webengine_renderer,
    _pet_topmost_surface,
    _prepare_qt_platform,
    _surface_mask_required,
    _TopmostTransitionController,
    _tray_is_recoverable,
)
from gui.qt6.pet_interaction import PET_CLICK_DEBOUNCE_MS, classify_pet_click
from gui.qt6.web_host import WebPetHost
from gui.renderers.live2d import Live2DRenderer
from gui.renderers.web_live2d import WebLive2DRenderer, probe_web_live2d

# xvfb 子进程用例的固定超时，允许门禁或慢机器通过环境变量放宽。
_XVFB_TEST_TIMEOUT_SECONDS = float(os.getenv("MEAPET_XVFB_TEST_TIMEOUT", "20"))


def test_x11_automation_batch_supports_keys_text_pointer_click_and_window_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, object, object, dict[str, object]]] = []

    class FakeWindow:
        def set_input_focus(self, revert, timestamp):
            events.append(("focus", revert, timestamp, {}))

        def configure(self, **kwargs):
            events.append(("configure", None, None, dict(kwargs)))

    class FakeDisplay:
        def __init__(self, name):
            assert name == ":99"
            self.closed = False
            self.window = FakeWindow()

        def has_extension(self, name):
            return name == "XTEST"

        def keysym_to_keycode(self, keysym):
            return int(keysym) if isinstance(keysym, int) else len(str(keysym)) + 20

        def sync(self):
            events.append(("sync", None, None, {}))

        def create_resource_object(self, kind, identifier):
            assert kind == "window"
            assert identifier == 0x44
            return self.window

        def close(self):
            self.closed = True

    class FakeX:
        KeyPress = 2
        KeyRelease = 3
        MotionNotify = 6
        ButtonPress = 4
        ButtonRelease = 5
        RevertToParent = 0
        CurrentTime = 0

    class FakeXTest:
        @staticmethod
        def fake_input(display, event_type, detail=0, **kwargs):
            events.append(("input", event_type, detail, dict(kwargs)))

    class FakeXK:
        @staticmethod
        def string_to_keysym(value):
            return ord(value) if len(value) == 1 else len(value) + 100

    modules = {
        "Xlib.display": SimpleNamespace(Display=FakeDisplay),
        "Xlib.X": FakeX,
        "Xlib.ext.xtest": FakeXTest,
        "Xlib.XK": FakeXK,
    }
    monkeypatch.setattr(linux_platform, "_import_optional", modules.__getitem__)
    platform = linux_platform.LinuxDesktopPlatform(
        environ={"QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99"}
    )

    result = platform.automation_batch(
        [
            {"type": "key", "key": "a", "modifiers": ["ctrl"]},
            {"type": "text", "text": "Ab!"},
            {"type": "move_pointer", "x": 20, "y": 30},
            {"type": "click", "x": 20, "y": 30, "button": "left"},
            {"type": "wait", "duration_ms": 1},
            {"type": "activate_window", "window_id": "0x44"},
            {"type": "move_window", "window_id": "0x44", "x": 40, "y": 50},
        ]
    )

    assert result["status"] == "completed"
    assert [item["type"] for item in result["steps"]] == [
        "key",
        "text",
        "move_pointer",
        "click",
        "wait",
        "activate_window",
        "move_window",
    ]
    assert ("configure", None, None, {"x": 40, "y": 50}) in events
    assert any(item[0] == "focus" for item in events)
    assert any(item[0] == "input" and item[1] == FakeX.ButtonPress for item in events)


def test_x11_automation_batch_rejects_unicode_text_without_input_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Display:
        def __init__(self, _name):
            return None

        def has_extension(self, _name):
            return True

        def keysym_to_keycode(self, _keysym):
            return 24

        def sync(self):
            return None

        def close(self):
            return None

    class FakeX:
        KeyPress = 2
        KeyRelease = 3

    class FakeXTest:
        @staticmethod
        def fake_input(*_args, **_kwargs):
            return None

    class FakeXK:
        @staticmethod
        def string_to_keysym(value):
            return ord(value) if len(value) == 1 else 100

    modules = {
        "Xlib.display": SimpleNamespace(Display=Display),
        "Xlib.X": FakeX,
        "Xlib.ext.xtest": FakeXTest,
        "Xlib.XK": FakeXK,
    }
    monkeypatch.setattr(linux_platform, "_import_optional", modules.__getitem__)
    platform = linux_platform.LinuxDesktopPlatform({"DISPLAY": ":99"})

    result = platform.automation_batch([{"type": "text", "text": "中"}])

    assert result["status"] == "unavailable"
    assert "printable ASCII" in result["reason"]


def test_x11_automation_batch_fails_closed_on_wayland() -> None:
    platform = linux_platform.LinuxDesktopPlatform(
        environ={"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"}
    )
    result = platform.automation_batch([{"type": "wait", "duration_ms": 1}])
    assert result["status"] == "unavailable"
    assert "X11" in result["reason"]


def test_x11_idle_probe_uses_mit_screensaver_and_closes_display(monkeypatch) -> None:
    """X11 空闲探针只读取 MIT-SCREEN-SAVER 的累计毫秒数。"""

    class Info:
        idle = 2500

    class Root:
        def screensaver_query_info(self):
            return Info()

    class Display:
        closed = False

        def __init__(self, name):
            assert name == ":99"

        def has_extension(self, name):
            return name == "MIT-SCREEN-SAVER"

        def screen(self):
            return SimpleNamespace(root=Root())

        def close(self):
            self.closed = True

    display_module = SimpleNamespace(Display=Display)
    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda name: SimpleNamespace() if name == "Xlib.ext.screensaver" else display_module,
    )
    assert (
        linux_platform.probe_x11_idle_seconds({"QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99"}) == 2.5
    )


def test_x11_cursor_position_reads_query_pointer_and_closes_display(monkeypatch) -> None:
    class Pointer:
        root_x = -7
        root_y = 311

    class Root:
        def query_pointer(self):
            return Pointer()

    class Display:
        closed = False

        def __init__(self, name):
            assert name == ":99"

        def screen(self):
            return SimpleNamespace(root=Root())

        def close(self):
            self.closed = True

    display_module = SimpleNamespace(Display=Display)
    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda _name: display_module,
    )
    result = linux_platform.LinuxDesktopPlatform(
        {"QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99"}
    ).cursor_position()
    assert result == {
        "status": "available",
        "backend": "x11",
        "x": -7,
        "y": 311,
        "source": "x11-query-pointer",
    }


def test_linux_cursor_position_is_fail_closed_without_x11(monkeypatch) -> None:
    monkeypatch.setattr(linux_platform, "_import_optional", lambda _name: None)
    result = linux_platform.LinuxDesktopPlatform(
        {"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"}
    ).cursor_position()
    assert result["status"] == "unavailable"
    assert "x" not in result
    assert "y" not in result


def test_x11_idle_probe_is_fail_closed_for_wayland_and_bad_extension(monkeypatch) -> None:
    """Wayland 或扩展缺失时不得猜测系统活跃状态。"""

    assert linux_platform.probe_x11_idle_seconds({"WAYLAND_DISPLAY": "wayland-0"}) is None
    monkeypatch.setattr(linux_platform, "_import_optional", lambda _name: None)
    assert linux_platform.probe_x11_idle_seconds({"DISPLAY": ":99"}) is None


def test_unknown_x11_compositor_uses_conservative_shape_mask(monkeypatch) -> None:
    """合成器探测失败时不能把透明顶层误报为安全。"""

    monkeypatch.setattr("gui.qt6.app.probe_x11_compositor", lambda _values: None)
    assert _surface_mask_required(SimpleNamespace(backend="x11")) is True
    assert _surface_mask_required(SimpleNamespace(backend="wayland")) is True


def test_x11_shape_mask_also_closes_transparent_input_when_compositor_exists(monkeypatch) -> None:
    """视觉合成正常时，透明侧边仍必须收窄原生输入区域。"""

    monkeypatch.setattr("gui.qt6.app.probe_x11_compositor", lambda _values: True)
    assert _surface_mask_required(SimpleNamespace(backend="x11")) is True


def test_console_adjacent_placement_avoids_pet_and_respects_screen(tmp_path: Path) -> None:
    """控制台和模型向导应贴近桌宠，但不能覆盖模型或越过屏幕。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = r"""
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget
from gui.qt6.app import _place_widget_adjacent

app = QApplication.instance() or QApplication([])
screen = QGuiApplication.primaryScreen()
assert screen is not None
area = screen.availableGeometry()
pet = QWidget()
pet.setGeometry(area.left() + 8, area.top() + 30, 180, 160)
pet.show()
console = QWidget()
console.resize(300, 220)
console.show()
app.processEvents()
assert _place_widget_adjacent(console, pet, anchor=pet.frameGeometry().center()) is not None
app.processEvents()
assert not console.frameGeometry().intersects(pet.frameGeometry())
assert area.contains(console.frameGeometry().topLeft())
assert area.contains(console.frameGeometry().bottomRight())

pet.move(area.right() - pet.width() - 8, area.top() + 30)
app.processEvents()
assert _place_widget_adjacent(console, pet, anchor=pet.frameGeometry().center()) is not None
app.processEvents()
assert not console.frameGeometry().intersects(pet.frameGeometry())
assert area.contains(console.frameGeometry().topLeft())
assert area.contains(console.frameGeometry().bottomRight())

large = QWidget()
large.resize(760, 860)
large.show()
pet.move(area.center().x() - pet.width() // 2, area.top())
app.processEvents()
assert _place_widget_adjacent(large, pet, anchor=pet.frameGeometry().center()) is not None
app.processEvents()
assert not large.frameGeometry().intersects(pet.frameGeometry())
assert area.contains(large.frameGeometry().topLeft())
assert area.contains(large.frameGeometry().bottomRight())
large.close()

console.close()
pet.close()
app.processEvents()
print("qt-adjacent-placement-ok")
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
    assert "qt-adjacent-placement-ok" in result.stdout


@pytest.mark.xvfb
def test_x11_automation_reaches_qt_line_edit(tmp_path: Path) -> None:
    """XTEST 键盘事件必须真正到达 Qt 输入控件，而不只返回 completed。"""

    if shutil.which("xvfb-run") is None:
        pytest.skip("requires xvfb-run")
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = r"""
import json
import os
from PySide6.QtWidgets import QApplication, QLineEdit
from gui.platforms.linux import LinuxDesktopPlatform

app = QApplication.instance() or QApplication([])
line = QLineEdit()
line.resize(400, 60)
line.show()
line.activateWindow()
line.setFocus()
app.processEvents()
platform = LinuxDesktopPlatform(environ=dict(os.environ))
result = platform.automation_batch([{"type": "text", "text": "Ab!"}])
app.processEvents()
assert result["status"] == "completed", result
assert line.text() == "Ab!", json.dumps(result, ensure_ascii=False)
line.close()
print("x11-automation-qt-line-edit-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["XDG_SESSION_TYPE"] = "x11"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24",
            sys.executable,
            "-c",
            script,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=_XVFB_TEST_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "x11-automation-qt-line-edit-ok" in result.stdout


@pytest.mark.xvfb
def test_tiny_screen_uses_full_click_surface_instead_of_narrow_sidebar(
    tmp_path: Path,
) -> None:
    """极小屏侧栏不足时应暂时隐藏桌宠，并保留可点击的全宽控制台。"""

    if shutil.which("xvfb-run") is None:
        pytest.skip("requires xvfb-run")
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = r"""
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QScrollArea, QTabWidget, QWidget
from gui.qt6.app import _place_widget_adjacent
from gui.qt6.console import PetConsoleWindow
from gui.qt6.model_setup import ModelSetupDialog

app = QApplication.instance() or QApplication([])
screen = app.primaryScreen()
assert screen is not None
area = screen.availableGeometry()
pet = QWidget()
pet.resize(200, 200)
pet.move(area.center() - QPoint(pet.width() // 2, pet.height() // 2))
pet.show()
sent = []
console = PetConsoleWindow(
    callbacks={"submit": lambda value: sent.append(value) or {"status": "accepted"}},
    configuration={
        "llm": {"channels": []},
        "scheduler": {"enabled": True, "tasks": [], "triggers": []},
        "tools": {"active_groups": []},
    },
)
console.show()
app.processEvents()
def enter_tiny_mode():
    pet.hide()
    console.set_compact_layout(True)
    assert console._navigate_console_page("对话")
_place_widget_adjacent(
    console,
    pet,
    anchor=pet.frameGeometry().center(),
    on_tight_layout=enter_tiny_mode,
)
app.processEvents()
geometry = console.frameGeometry()
assert not pet.isVisible()
assert area.contains(geometry.topLeft())
assert area.contains(geometry.bottomRight())
assert geometry.width() >= area.width() - 16
input_line = console.findChild(QLineEdit, "conversationInput")
send = console.findChild(QPushButton, "sendMessageButton")
assert input_line is not None and input_line.isVisible()
assert send is not None and send.isVisible()
control_scroll = console.findChild(QScrollArea, "consoleControlScroll")
assert control_scroll is not None and control_scroll.viewport().isVisible()
input_top = input_line.mapToGlobal(QPoint(0, 0)).y()
input_bottom = input_line.mapToGlobal(input_line.rect().bottomRight()).y()
viewport_top = control_scroll.viewport().mapToGlobal(QPoint(0, 0)).y()
viewport_bottom = control_scroll.viewport().mapToGlobal(
    control_scroll.viewport().rect().bottomRight()
).y()
assert viewport_top <= input_top <= input_bottom <= viewport_bottom
assert input_bottom <= area.bottom()
input_line.setText("tiny-screen")
send.click()
assert sent == ["tiny-screen"]
assert "未提交" not in console.status_label.text()
assert input_line.mapToGlobal(input_line.rect().bottomRight()).y() <= area.bottom()
console.show_settings()
app.processEvents()
config_header = console._configuration_panel.findChild(QWidget, "configHeader")
section_picker = console._configuration_panel.findChild(QWidget, "configSectionPicker")
save_button = console._configuration_panel.findChild(QPushButton, "configPrimary")
rescue_card = console.findChild(QWidget, "consoleRescueCard")
overview_actions = [
    item
    for item in console._configuration_panel.findChildren(QPushButton, "configChoiceButton")
    if item.isVisible()
]
assert config_header is not None and not config_header.isVisible()
assert section_picker is not None and section_picker.isVisible()
assert save_button is not None and save_button.isVisible()
assert rescue_card is not None and not rescue_card.isVisible()
assert overview_actions
overview_bottom = overview_actions[0].mapToGlobal(
    overview_actions[0].rect().bottomRight()
).y()
assert overview_bottom <= area.bottom()
assert save_button.mapToGlobal(save_button.rect().bottomRight()).y() <= area.bottom()
console._configuration_panel._set_status("配置已保存；重新启动后完全生效。")
assert "重新启动后完全生效" in console.status_label.text()
tabs = console.findChild(QTabWidget, "consoleTabs")
assert tabs is not None
tabs.setCurrentIndex(0)
app.processEvents()
assert rescue_card.isVisible()
assert not console.findChild(QPushButton, "configCenterButton").isVisible()
assert console.findChild(QPushButton, "configureModelButton").isVisible()
console.grab().save("/tmp/meapet-console-tiny-640x480.png")
pet.show()
console.set_compact_layout(False)
assert pet.isVisible()
console.shutdown()
app.processEvents()
dialog = ModelSetupDialog()
dialog.show()
app.processEvents()
_place_widget_adjacent(
    dialog,
    pet,
    anchor=pet.frameGeometry().center(),
    on_tight_layout=pet.hide,
)
app.processEvents()
dialog_geometry = dialog.frameGeometry()
assert not pet.isVisible()
assert area.contains(dialog_geometry.topLeft())
assert area.contains(dialog_geometry.bottomRight())
assert dialog_geometry.width() >= area.width() - 16
assert dialog.save_button is not None and dialog.save_button.isVisible()
dialog.grab().save("/tmp/meapet-model-setup-tiny-640x480.png")
dialog.close()
pet.show()
assert pet.isVisible()
dialog.deleteLater()
app.processEvents()
print("tiny-screen-click-surface-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 640x480x24",
            sys.executable,
            "-c",
            script,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=_XVFB_TEST_TIMEOUT_SECONDS,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "tiny-screen-click-surface-ok" in result.stdout


def _write_model3_descriptor(model_root: Path, name: str = "demo") -> None:
    (model_root / f"{name}.moc3").write_bytes(b"moc")
    (model_root / f"{name}.png").write_bytes(b"texture")
    (model_root / f"{name}.model3.json").write_text(
        f'{{"FileReferences":{{"Moc":"{name}.moc3","Textures":["{name}.png"]}}}}',
        encoding="utf-8",
    )


def test_web_live2d_natural_layout_keeps_browser_canvas_scale_separate_from_default_fit() -> None:
    """桌面自然布局只在启动实例生效，通用渲染器仍保留动态拟合。"""

    default_html = WebLive2DRenderer("resources")._html_document()
    natural_html = WebLive2DRenderer("resources", natural_layout=True)._html_document()
    assert "const naturalLayout = false;" in default_html
    assert "const naturalLayout = true;" in natural_html
    assert "if (!naturalLayout && !base.visibleReferenceReady" in default_html
    assert "if (!naturalLayout && !base.visibleReferenceReady" in natural_html


def test_webengine_flags_keep_native_backend_by_default(monkeypatch) -> None:
    """默认不强制 SwiftShader；软件路径必须显式选择。"""

    monkeypatch.delenv("QTWEBENGINE_CHROMIUM_FLAGS", raising=False)
    monkeypatch.delenv("MEAPET_WEBENGINE_SOFTWARE", raising=False)
    monkeypatch.delenv("QT_QUICK_BACKEND", raising=False)
    _configure_qt_webengine_renderer()
    assert os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] == "--enable-transparent-visuals"

    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--foo")
    monkeypatch.setenv("MEAPET_WEBENGINE_SOFTWARE", "1")
    _configure_qt_webengine_renderer()
    flags = os.environ["QTWEBENGINE_CHROMIUM_FLAGS"].split()
    assert flags[:1] == ["--foo"]
    assert "--enable-transparent-visuals" in flags
    assert "--ignore-gpu-blocklist" not in flags
    assert "--use-gl=angle" in flags
    assert "--use-angle=swiftshader" in flags
    assert "--enable-unsafe-swiftshader" in flags
    assert "--disable-gpu-compositing" in flags
    assert os.environ["QT_QUICK_BACKEND"] == "software"

    # 用户已经选择其它 Qt Quick 后端时，软件 WebGL 配置不能静默覆盖它。
    monkeypatch.setenv("QT_QUICK_BACKEND", "custom")
    _configure_qt_webengine_renderer()
    assert os.environ["QT_QUICK_BACKEND"] == "custom"


def test_qt_platform_auto_prefers_xcb_controls_when_xwayland_is_available(monkeypatch) -> None:
    """存在 Xwayland 时 auto 保留绝对移动、置顶和输入标志的可控契约。"""

    from types import SimpleNamespace

    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    configuration = SimpleNamespace(values={"ui": {"qt_platform": "auto"}})

    assert _prepare_qt_platform(configuration) == "xcb"
    assert os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_qt_platform_explicit_xcb_is_not_rewritten_by_compositor_probe(monkeypatch) -> None:
    """显式 xcb 由 Shape 掩码回退处理透明面，不应悄悄改成另一个后端。"""

    from types import SimpleNamespace

    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")
    configuration = SimpleNamespace(values={"ui": {"qt_platform": "xcb"}})

    assert _prepare_qt_platform(configuration) == "xcb"
    assert os.environ["QT_QPA_PLATFORM"] == "xcb"


def test_web_live2d_action_reports_pending_before_model_ready(tmp_path: Path) -> None:
    """模型尚未完成加载时，动作不能向控制台伪装成已执行。"""

    renderer = WebLive2DRenderer(tmp_path)
    renderer._view = object()
    renderer._replay_pending_model_state = lambda: None  # type: ignore[method-assign]

    expression = renderer.set_expression("happy")
    motion = renderer.play_motion("wave")

    assert expression == {"status": "pending", "name": "happy"}
    assert motion == {"status": "pending", "name": "wave"}


def test_wayland_capabilities_report_compositor_limits() -> None:
    snapshot = linux_platform.probe_linux_platform(
        {
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-0",
            "DISPLAY": ":0",
        }
    )
    assert snapshot.backend == linux_platform.BACKEND_WAYLAND
    assert all(
        item.state is CapabilityState.UNAVAILABLE
        for item in snapshot.capabilities
        if item.name not in {"click_through", "always_on_top"}
    )
    # Qt 可以确认 WindowTransparentForInput 标志已提交，但 Wayland
    # compositor 是否真正路由穿透输入不由客户端决定，因此只能报告降级。
    click_through = snapshot.capability("click_through")
    if linux_platform._import_optional("PySide6.QtCore") is not None:
        assert click_through is not None
        assert click_through.state is CapabilityState.DEGRADED
    else:
        assert click_through is not None
        assert click_through.state is CapabilityState.UNAVAILABLE
    topmost = snapshot.capability("always_on_top")
    assert topmost is not None
    assert topmost.state is (
        CapabilityState.DEGRADED
        if linux_platform._import_optional("PySide6.QtCore") is not None
        else CapabilityState.UNAVAILABLE
    )
    assert snapshot.capability("global_hotkeys").state is CapabilityState.UNAVAILABLE


def test_xwayland_topmost_probe_requires_per_window_ewmh_readback(monkeypatch) -> None:
    """XWayland 不能仅凭 Qt 类型存在就把置顶能力报告为已确认。"""

    monkeypatch.setattr(linux_platform, "probe_x11_compositor", lambda _values: True)
    monkeypatch.setattr(linux_platform, "probe_x11_input_control", lambda _values: False)

    def fake_import(name: str):
        if name in {"PySide6.QtCore", "PySide6.QtGui", "Xlib.display", "psutil"}:
            return SimpleNamespace()
        return None

    monkeypatch.setattr(linux_platform, "_import_optional", fake_import)
    snapshot = linux_platform.probe_linux_platform(
        {
            "QT_QPA_PLATFORM": "xcb",
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-0",
            "DISPLAY": ":99",
        }
    )

    topmost = snapshot.capability("always_on_top")
    assert topmost is not None
    assert topmost.state is CapabilityState.DEGRADED
    assert "_NET_WM_STATE_ABOVE" in topmost.evidence
    assert "per-window EWMH readback" in topmost.detail


def test_linux_topmost_status_prefers_ewmh_over_qt_flag(monkeypatch) -> None:
    """Qt flag 为真而 KWin 未设置 ABOVE 时，平台回读必须报告未置顶。"""

    topmost_flag = 0x20
    wm_state = {"enabled": False, "closed": 0}

    class NativeHandle:
        def flags(self):
            return topmost_flag

        def winId(self):
            return 0x12345

    class View:
        def __init__(self) -> None:
            self.handle = NativeHandle()

        def windowHandle(self):
            return self.handle

    class NativeWindow:
        def get_full_property(self, _state_atom, _property_type):
            values = (11, 12) if wm_state["enabled"] else ()
            return SimpleNamespace(value=values)

    class Display:
        def __init__(self, name):
            assert name == ":99"

        def intern_atom(self, name):
            return {
                "_NET_WM_STATE": 10,
                "_NET_WM_STATE_ABOVE": 11,
                "_NET_WM_STATE_STAYS_ON_TOP": 12,
            }[name]

        def create_resource_object(self, kind, window_id):
            assert kind == "window"
            assert window_id == 0x12345
            return NativeWindow()

        def close(self):
            wm_state["closed"] += 1

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(
            WindowType=SimpleNamespace(WindowStaysOnTopHint=topmost_flag),
        )
    )

    def fake_import(name: str):
        return {
            "PySide6.QtCore": qt_core,
            "Xlib.display": SimpleNamespace(Display=Display),
            "Xlib.X": SimpleNamespace(AnyPropertyType=0),
        }.get(name)

    monkeypatch.setattr(linux_platform, "_import_optional", fake_import)
    platform = linux_platform.LinuxDesktopPlatform({"QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99"})

    dropped = platform.always_on_top_status(View())
    assert dropped == {
        "status": "available",
        "enabled": False,
        "confirmed": True,
        "detail": "窗口管理器确认当前窗口未置顶",
    }

    wm_state["enabled"] = True
    retained = platform.always_on_top_status(View())
    assert retained["status"] == "available"
    assert retained["enabled"] is True
    assert retained["confirmed"] is True
    assert retained["evidence"] == (
        "_NET_WM_STATE_ABOVE",
        "_NET_WM_STATE_STAYS_ON_TOP",
    )
    assert wm_state["closed"] == 2


def test_linux_topmost_status_contains_x11_protocol_error(monkeypatch) -> None:
    """窗口在 EWMH 回读中消失时必须降级，不能让 Xlib 异常终止应用。"""

    topmost_flag = 0x20
    closed = {"count": 0}

    class XProtocolError(Exception):
        pass

    class NativeHandle:
        def flags(self):
            return topmost_flag

        def winId(self):
            return 0x54321

    class View:
        def windowHandle(self):
            return NativeHandle()

    class NativeWindow:
        def get_full_property(self, _state_atom, _property_type):
            raise XProtocolError("window disappeared")

    class Display:
        def __init__(self, _name):
            return None

        def intern_atom(self, name):
            return {
                "_NET_WM_STATE": 10,
                "_NET_WM_STATE_ABOVE": 11,
                "_NET_WM_STATE_STAYS_ON_TOP": 12,
            }[name]

        def create_resource_object(self, _kind, _window_id):
            return NativeWindow()

        def close(self):
            closed["count"] += 1

    qt_core = SimpleNamespace(
        Qt=SimpleNamespace(
            WindowType=SimpleNamespace(WindowStaysOnTopHint=topmost_flag),
        )
    )

    def fake_import(name: str):
        return {
            "PySide6.QtCore": qt_core,
            "Xlib.display": SimpleNamespace(Display=Display),
            "Xlib.X": SimpleNamespace(AnyPropertyType=0),
        }.get(name)

    monkeypatch.setattr(linux_platform, "_import_optional", fake_import)
    platform = linux_platform.LinuxDesktopPlatform({"QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99"})

    status = platform.always_on_top_status(View())

    assert status["status"] == "degraded"
    assert status["enabled"] is True
    assert status["confirmed"] is False
    assert "EWMH" in str(status["detail"])
    assert closed["count"] == 1


def test_topmost_controller_uses_host_handler_and_bounds_native_readback() -> None:
    """组合根不能重复改旗标，且窗口管理器拒绝时必须有界结束 requested。"""

    class Host:
        def __init__(self) -> None:
            self.enabled = False
            self.pending = False
            self.calls: list[bool] = []

        def set_always_on_top(self, enabled: bool):
            self.calls.append(bool(enabled))
            self.pending = True
            return {"status": "requested", "enabled": bool(enabled)}

        def always_on_top_status(self):
            if self.pending:
                return {"status": "available", "enabled": True}
            return {"status": "available", "enabled": self.enabled}

    class Platform:
        def __init__(self) -> None:
            self.native_enabled = False
            self.set_calls = 0

        def set_always_on_top(self, _window, _enabled):
            self.set_calls += 1
            raise AssertionError("composition root must not submit a second native flag update")

        def always_on_top_status(self, _window):
            return {
                "status": "available",
                "enabled": self.native_enabled,
                "confirmed": True,
            }

    host = Host()
    platform = Platform()
    surface = object()
    scheduled: list[tuple[int, Callable[[], None]]] = []
    controller = _TopmostTransitionController(
        platform=platform,
        surface_provider=lambda: surface,
        host_provider=lambda: host,
        schedule=lambda delay, callback: scheduled.append((delay, callback)),
    )

    result = controller.toggle()
    assert result["status"] == "requested"
    assert host.calls == [True]
    assert platform.set_calls == 0
    assert [delay for delay, _callback in scheduled] == [0, 80, 240, 700, 1200]

    for _delay, callback in scheduled:
        callback()
    assert controller.status()["status"] == "unavailable"
    assert controller.status()["enabled"] is False
    assert "未确认" in str(controller.status().get("detail", ""))

    assert _pet_topmost_surface(surface, SimpleNamespace(view=object())) is surface
    web_view = object()
    assert _pet_topmost_surface(None, SimpleNamespace(view=web_view)) is web_view


def test_topmost_controller_publishes_confirmed_native_state() -> None:
    """宿主异步完成后，组合根应发布平台确认的最终置顶状态。"""

    class Host:
        def __init__(self) -> None:
            self.enabled = False

        def set_always_on_top(self, enabled: bool):
            self.enabled = bool(enabled)
            return {"status": "requested", "enabled": self.enabled}

        def always_on_top_status(self):
            return {"status": "available", "enabled": self.enabled}

    class Platform:
        def __init__(self) -> None:
            self.enabled = False

        def always_on_top_status(self, _window):
            return {
                "status": "available",
                "enabled": self.enabled,
                "confirmed": True,
                "detail": "窗口管理器已确认置顶",
            }

    host = Host()
    platform = Platform()
    scheduled: list[Callable[[], None]] = []
    controller = _TopmostTransitionController(
        platform=platform,
        surface_provider=object,
        host_provider=lambda: host,
        schedule=lambda _delay, callback: scheduled.append(callback),
    )
    result = controller.toggle()
    assert result["status"] == "requested"

    platform.enabled = True
    scheduled[0]()
    assert controller.status() == {
        "status": "available",
        "enabled": True,
        "confirmed": True,
        "detail": "窗口管理器已确认置顶",
    }


def test_x11_coordinate_click_uses_xtest_and_closes_display(monkeypatch) -> None:
    calls: list[tuple[int, int, int | None, int | None]] = []

    class Display:
        def __init__(self, name):
            assert name == ":99"

        def has_extension(self, name):
            return name == "XTEST"

        def xtest_fake_input(self, event_type, detail=0, *, x=None, y=None):
            calls.append((event_type, detail, x, y))

        def sync(self):
            return None

        def close(self):
            return None

    display_module = SimpleNamespace(Display=Display)
    x_module = SimpleNamespace(MotionNotify=6, ButtonPress=4, ButtonRelease=5)
    xtest_module = SimpleNamespace()

    def load(name):
        return {
            "Xlib.display": display_module,
            "Xlib.X": x_module,
            "Xlib.ext.xtest": xtest_module,
        }.get(name)

    monkeypatch.setattr(linux_platform, "_import_optional", load)
    platform = linux_platform.LinuxDesktopPlatform({"DISPLAY": ":99"})

    assert platform.probe().capability("input_control").state is CapabilityState.AVAILABLE

    result = platform.click_at(120, 240, button="right")

    assert result == {
        "status": "completed",
        "backend": "x11",
        "x": 120,
        "y": 240,
        "button": "right",
    }
    assert calls == [(6, 0, 120, 240), (4, 3, None, None), (5, 3, None, None)]


def test_x11_coordinate_click_accepts_negative_virtual_desktop_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object, object, object]] = []

    class Display:
        def __init__(self, name):
            assert name == ":99"

        def has_extension(self, name):
            return name == "XTEST"

        def xtest_fake_input(self, event_type, detail=0, *, x=None, y=None):
            calls.append((event_type, detail, x, y))

        def sync(self):
            return None

        def close(self):
            return None

    modules = {
        "Xlib.display": SimpleNamespace(Display=Display),
        "Xlib.X": SimpleNamespace(MotionNotify=6, ButtonPress=4, ButtonRelease=5),
        "Xlib.ext.xtest": SimpleNamespace(),
    }
    monkeypatch.setattr(linux_platform, "_import_optional", modules.__getitem__)
    platform = linux_platform.LinuxDesktopPlatform({"DISPLAY": ":99"})

    result = platform.click_at(-120, 240, button="left")

    assert result == {
        "status": "completed",
        "backend": "x11",
        "x": -120,
        "y": 240,
        "button": "left",
    }
    assert calls[0] == (6, 0, -120, 240)


def test_coordinate_click_fails_closed_outside_x11(monkeypatch) -> None:
    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda _name: (_ for _ in ()).throw(AssertionError("must not import XTEST")),
    )
    platform = linux_platform.LinuxDesktopPlatform(
        {"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"}
    )

    result = platform.click_at(10, 20)

    assert result["status"] == "unavailable"
    assert "X11 XTEST" in str(result["reason"])


def test_wayland_click_through_reports_degraded_when_qt_retains_flag(monkeypatch) -> None:
    flag = 8
    window_type = type("WindowType", (), {"WindowTransparentForInput": flag})
    core = type("Core", (), {"Qt": type("Qt", (), {"WindowType": window_type})})

    class Window:
        def __init__(self) -> None:
            self._flags = 0

        def setFlag(self, requested, enabled):  # noqa: N802
            self._flags = self._flags | requested if enabled else self._flags & ~requested

        def flags(self):
            return self._flags

    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda name: core if name == "PySide6.QtCore" else None,
    )
    result = linux_platform.set_window_click_through(Window(), True, backend="wayland")
    assert result.state is CapabilityState.DEGRADED
    assert "compositor" in result.detail


def test_click_through_reports_unavailable_when_qt_drops_flag(monkeypatch) -> None:
    """未保留 Qt 输入透明标志时不能把请求当作降级成功。"""

    flag = 8
    window_type = type("WindowType", (), {"WindowTransparentForInput": flag})
    core = type("Core", (), {"Qt": type("Qt", (), {"WindowType": window_type})})

    class Window:
        def setFlag(self, _requested, _enabled):  # noqa: N802
            return None

        def flags(self):
            return 0

    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda name: core if name == "PySide6.QtCore" else None,
    )
    for backend in ("x11", "wayland"):
        result = linux_platform.set_window_click_through(Window(), True, backend=backend)
        assert result.state is CapabilityState.UNAVAILABLE
        assert "did not retain" in result.detail


def test_click_through_failure_does_not_echo_exception_detail(monkeypatch) -> None:
    """Qt 失败回执只返回固定文案，不能把内部异常正文交给界面。"""

    flag = 8
    window_type = type("WindowType", (), {"WindowTransparentForInput": flag})
    core = type("Core", (), {"Qt": type("Qt", (), {"WindowType": window_type})})

    class Window:
        def setFlag(self, _requested, _enabled):  # noqa: N802
            raise RuntimeError("secret-window-state")

    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda name: core if name == "PySide6.QtCore" else None,
    )
    result = linux_platform.set_window_click_through(Window(), True, backend="x11")
    assert result.state is CapabilityState.UNAVAILABLE
    assert result.detail == "Qt input transparency flag update failed"
    assert "secret-window-state" not in result.detail


def test_click_through_supports_qwidget_window_flag_setter(monkeypatch) -> None:
    """WebEngine QWidget 没有 QWindow.setFlag，平台边界仍应可切换输入透明。"""

    flag = 8
    window_type = type("WindowType", (), {"WindowTransparentForInput": flag})
    core = type("Core", (), {"Qt": type("Qt", (), {"WindowType": window_type})})

    class Window:
        def __init__(self) -> None:
            self._flags = 0

        def setWindowFlag(self, requested, enabled):  # noqa: N802
            self._flags = self._flags | requested if enabled else self._flags & ~requested

        def windowFlags(self):  # noqa: N802
            return self._flags

    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda name: core if name == "PySide6.QtCore" else None,
    )
    window = Window()
    result = linux_platform.set_window_click_through(window, True, backend="x11")
    assert result.state is CapabilityState.AVAILABLE
    assert linux_platform.set_window_click_through(window, False, backend="x11").state is (
        CapabilityState.AVAILABLE
    )


def test_click_through_prefers_native_qwindow_and_preserves_topmost_flag(monkeypatch) -> None:
    """WebEngine 已有 QWindow 时必须原地改旗标，不能调用 QWidget 重建路径。"""

    transparent_flag = 8
    topmost_flag = 16
    window_type = type(
        "WindowType",
        (),
        {
            "WindowTransparentForInput": transparent_flag,
            "WindowStaysOnTopHint": topmost_flag,
        },
    )
    core = type("Core", (), {"Qt": type("Qt", (), {"WindowType": window_type})})

    class NativeWindow:
        def __init__(self) -> None:
            self._flags = topmost_flag
            self.calls: list[tuple[int, bool]] = []

        def setFlag(self, requested, enabled):  # noqa: N802
            self.calls.append((requested, bool(enabled)))
            self._flags = self._flags | requested if enabled else self._flags & ~requested

        def flags(self):
            return self._flags

    class Window:
        def __init__(self) -> None:
            self.native = NativeWindow()
            self.widget_calls: list[bool] = []

        def windowHandle(self):  # noqa: N802
            return self.native

        def setWindowFlag(self, _requested, enabled):  # noqa: N802
            self.widget_calls.append(bool(enabled))

    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda name: core if name == "PySide6.QtCore" else None,
    )
    window = Window()

    enabled = linux_platform.set_window_click_through(window, True, backend="x11")
    disabled = linux_platform.set_window_click_through(window, False, backend="x11")

    assert enabled.state is CapabilityState.AVAILABLE
    assert disabled.state is CapabilityState.AVAILABLE
    assert window.native.calls == [
        (transparent_flag, True),
        (transparent_flag, False),
    ]
    assert window.native.flags() & topmost_flag
    assert window.widget_calls == []


def test_x11_input_shape_normalizes_qt_region_to_device_pixels() -> None:
    """Input Shape 必须按窗口 DPR 将 Qt 逻辑矩形转换为 X11 设备像素。"""

    class Rect:
        def __init__(self, x: int, y: int, width: int, height: int) -> None:
            self._values = (x, y, width, height)

        def x(self) -> int:
            return self._values[0]

        def y(self) -> int:
            return self._values[1]

        def width(self) -> int:
            return self._values[2]

        def height(self) -> int:
            return self._values[3]

    class Window:
        def devicePixelRatioF(self) -> float:
            return 2.0

        def width(self) -> int:
            return 100

        def height(self) -> int:
            return 80

    assert linux_platform._x11_shape_rectangles([Rect(3, 4, 5, 6)], window=Window()) == [
        (6, 8, 10, 12)
    ]
    assert linux_platform._x11_shape_rectangles((), window=Window()) == [(-1, -1, 1, 1)]
    assert linux_platform._x11_shape_rectangles(None, window=Window()) == [(0, 0, 200, 160)]


def test_input_shape_reports_wayland_degraded_without_claiming_x11_support() -> None:
    """Wayland 没有原生 Input Shape 时必须显式返回降级。"""

    result = linux_platform.set_window_input_shape(
        object(),
        (),
        backend="wayland",
        environ={"WAYLAND_DISPLAY": "wayland-0"},
    )
    assert result.name == "input_shape"
    assert result.state is CapabilityState.DEGRADED
    assert "Wayland" in result.detail


def test_x11_input_shape_uses_dynamic_xlib_and_fails_closed_on_bad_window(monkeypatch) -> None:
    """XShape 请求应动态加载依赖，并把 BadWindow 竞态收敛为不可用。"""

    class Handler:
        def __init__(self) -> None:
            self.error = None

        def get_error(self):
            return self.error

    class NativeWindow:
        def __init__(self) -> None:
            self.calls = []

        def shape_rectangles(self, *args):
            self.calls.append(args)

    class Connection:
        def __init__(self) -> None:
            self.handler = None
            self.native = NativeWindow()
            self.fail_sync = False
            self.closed = False

        def set_error_handler(self, handler):
            self.handler = handler

        def create_resource_object(self, _kind, _window_id):
            return self.native

        def sync(self):
            if self.fail_sync and self.handler is not None:
                self.handler.error = object()

        def close(self):
            self.closed = True

    connection = Connection()
    error_module = SimpleNamespace(
        BadWindow=type("BadWindow", (Exception,), {}),
        CatchError=lambda _kind: Handler(),
    )
    shape_module = SimpleNamespace(
        SO=SimpleNamespace(Set=7),
        SK=SimpleNamespace(Input=8),
    )
    modules = {
        "Xlib.display": SimpleNamespace(Display=lambda _name: connection),
        "Xlib.error": error_module,
        "Xlib.ext.shape": shape_module,
    }
    monkeypatch.setattr(linux_platform, "_import_optional", modules.__getitem__)

    class Window:
        def winId(self) -> int:  # noqa: N802
            return 42

        def devicePixelRatioF(self) -> float:
            return 2.0

    class Rect:
        def x(self) -> int:
            return 1

        def y(self) -> int:
            return 2

        def width(self) -> int:
            return 3

        def height(self) -> int:
            return 4

    result = linux_platform.set_window_input_shape(
        Window(), [Rect()], backend="x11", environ={"DISPLAY": ":99"}
    )
    assert result.state is CapabilityState.AVAILABLE
    assert connection.native.calls[-1][-1] == [(2, 4, 6, 8)]
    connection.fail_sync = True
    failed = linux_platform.set_window_input_shape(
        Window(), [Rect()], backend="x11", environ={"DISPLAY": ":99"}
    )
    assert failed.state is CapabilityState.UNAVAILABLE
    assert "BadWindow" not in failed.detail
    assert connection.closed is True


def test_move_overlay_uses_qpoint_for_qwindow(monkeypatch) -> None:
    """QWindow.setPosition 的单 QPoint 契约不能按 QWidget 双参数调用。"""

    from PySide6.QtCore import QPoint

    class FakeWindow:
        def __init__(self) -> None:
            self.received = None

        def setPosition(self, point):  # noqa: N802
            assert isinstance(point, QPoint)
            self.received = (point.x(), point.y())

    window = FakeWindow()
    platform = linux_platform.LinuxDesktopPlatform({"QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99"})
    result = platform.move_overlay(window, 123, 456)
    assert result.state is CapabilityState.AVAILABLE
    assert window.received == (123, 456)


def test_qwidget_click_through_preserves_visible_window(tmp_path: Path) -> None:
    """QWidget 设置 WindowTransparentForInput 后仍必须保持可见。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication, QWidget
from gui.platforms.linux import LinuxDesktopPlatform

app = QApplication.instance() or QApplication([])
view = QWidget()
view.resize(120, 80)
view.show()
app.processEvents()
platform = LinuxDesktopPlatform({"DISPLAY": ":0"})
for enabled in (True, False):
    result = platform.set_click_through(view, enabled)
    app.processEvents()
    assert result.state.value == "available"
    assert view.isVisible()
view.close()
app.processEvents()
print("qwidget-click-through-visible-ok")
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
    assert "qwidget-click-through-visible-ok" in result.stdout


def test_click_through_recovery_requires_visible_tray() -> None:
    class Tray:
        def __init__(self, visible: bool):
            self.visible = visible

        def isVisible(self):
            return self.visible

    assert _tray_is_recoverable(Tray(True)) is True
    assert _tray_is_recoverable(Tray(False)) is False
    assert _tray_is_recoverable(object()) is False


def test_click_through_restore_does_not_hide_failed_state() -> None:
    assert _click_through_operation_succeeded({"status": "available"}, enabled=False)
    assert _click_through_operation_succeeded(
        {"status": "degraded", "enabled": False}, enabled=False
    )
    assert not _click_through_operation_succeeded(
        {"status": "unavailable", "enabled": False}, enabled=False
    )
    assert not _click_through_operation_succeeded(
        {"status": "degraded", "enabled": True}, enabled=False
    )


def test_qt_opengl_window_constructor_requests_alpha_without_set_color(tmp_path: Path) -> None:
    """在实际 PySide6 绑定中验证透明窗口构造不依赖不存在的 setColor。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    source_root = Path(__file__).resolve().parents[1] / "src"
    sprite_dir = Path(__file__).resolve().parents[1] / "resources" / "sprites"
    script = f"""
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from gui.platforms.linux import LinuxDesktopPlatform
from gui.qt6.opengl_host import PetOpenGLWindow
from gui.renderers.sprite import SpriteRenderer

app = QGuiApplication.instance() or QGuiApplication([])
window = PetOpenGLWindow(SpriteRenderer(Path({str(sprite_dir)!r})))
assert window.format().alphaBufferSize() >= 8
assert window.height() > window.width()
assert 1.8 < (window.height() - 50) / window.width() < 2.2
window.show()
app.processEvents()
platform = LinuxDesktopPlatform({{"QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99"}})
window.platform = platform
result = window.move_to(123, 145)
app.processEvents()
assert result["status"] == "available"
assert (window.position().x(), window.position().y()) == (123, 145)
window.close()
app.processEvents()
print('qt-opengl-window-ok')
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    python_path = str(source_root)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{python_path}{os.pathsep}{existing_python_path}" if existing_python_path else python_path
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
    assert "qt-opengl-window-ok" in result.stdout


def test_qt_opengl_bubble_preserves_head_and_tail_within_fixed_region(tmp_path: Path) -> None:
    """精灵回退气泡在固定区域内保留首尾，不静默裁掉长文本。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QApplication
from gui.qt6.opengl_host import _fit_bubble_text

app = QApplication.instance() or QApplication([])
font = QFont("LXGW WenKai", 15)
rect = QRectF(22, 18, 376, 36)
source = "开头提示：" + ("长文本内容" * 800) + "：结尾提示"
display = _fit_bubble_text(source, font, rect)
height = QFontMetrics(font).boundingRect(
    rect.toRect(),
    Qt.TextWordWrap,
    display,
).height()
assert "…" in display
assert display.startswith("开头提示：")
assert display.endswith("：结尾提示")
assert height <= rect.height()
print("qt-opengl-bubble-truncation-ok")
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
    assert "qt-opengl-bubble-truncation-ok" in result.stdout


def test_xwayland_scope_is_degraded(monkeypatch) -> None:
    def fake_import(name: str):
        return object() if name in {"Xlib.display", "psutil"} else None

    monkeypatch.setattr(linux_platform, "_import_optional", fake_import)
    snapshot = linux_platform.probe_linux_platform(
        {
            "QT_QPA_PLATFORM": "xcb",
            "XDG_SESSION_TYPE": "wayland",
            "DISPLAY": ":0",
        }
    )
    assert snapshot.backend == linux_platform.BACKEND_X11
    assert snapshot.details["scope"] == "xwayland"
    assert snapshot.capability("active_window").state is CapabilityState.DEGRADED
    assert snapshot.capability("process_context").state is CapabilityState.DEGRADED
    assert snapshot.capability("screen_capture").state is CapabilityState.UNAVAILABLE


def test_xcb_without_display_does_not_claim_window_controls(monkeypatch) -> None:
    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda name: object() if name in {"PySide6.QtCore", "PySide6.QtGui"} else None,
    )

    snapshot = linux_platform.probe_linux_platform(
        {
            "QT_QPA_PLATFORM": "xcb",
            "XDG_SESSION_TYPE": "wayland",
            "DISPLAY": "",
        }
    )

    assert snapshot.backend == linux_platform.BACKEND_X11
    assert snapshot.capability("click_through").state is CapabilityState.UNAVAILABLE
    assert snapshot.capability("overlay_position").state is CapabilityState.UNAVAILABLE


def test_wayland_window_controls_expose_degraded_state_for_both_hosts(tmp_path: Path) -> None:
    """OpenGL 与 WebEngine 宿主都必须保留 Wayland 的降级语义。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    sprite_dir = project_root / "resources" / "sprites"
    script = f"""
from pathlib import Path
from PySide6.QtWidgets import QApplication, QWidget
from gui.platforms.linux import LinuxDesktopPlatform
from gui.qt6.opengl_host import PetOpenGLWindow
from gui.qt6.web_host import WebPetHost
from gui.renderers.sprite import SpriteRenderer

app = QApplication.instance() or QApplication([])
platform = LinuxDesktopPlatform({{"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"}})
window = PetOpenGLWindow(SpriteRenderer(Path({str(sprite_dir)!r})), platform=platform)
window.show()
app.processEvents()
assert window.set_always_on_top(False)["status"] == "degraded"
assert window.set_click_through(True).state.value == "degraded"
assert window.begin_click_through_override().state.value == "degraded"
assert window.end_click_through_override().state.value == "degraded"
window.close()

view = QWidget()
view.show()
renderer = type("Renderer", (), {{"view": view, "shutdown": lambda self: None}})()
host = WebPetHost(renderer, platform=platform)
app.processEvents()
assert host.set_always_on_top(False)["status"] == "degraded"
assert host.set_click_through(True)["status"] == "degraded"
assert host.begin_click_through_override()["status"] == "degraded"
assert host.end_click_through_override()["status"] == "degraded"
host.shutdown()
view.close()
app.processEvents()
print("wayland-host-controls-ok")
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
    assert "wayland-host-controls-ok" in result.stdout


def test_window_capture_requires_pre_resolved_id(monkeypatch) -> None:
    platform = linux_platform.LinuxDesktopPlatform({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":99"})

    def forbidden_resolution() -> int:
        raise AssertionError("window lookup must stay outside the Qt capture stage")

    monkeypatch.setattr(platform, "resolve_capture_window_id", forbidden_resolution)
    assert platform.capture_screen(scope="window") == {
        "status": "unavailable",
        "reason": "window id must be resolved before Qt screen capture",
    }


def test_x11_region_capture_does_not_grab_full_primary_screen(monkeypatch) -> None:
    class Buffer:
        def __init__(self) -> None:
            self.payload = b""

        def open(self, _mode):
            return True

        def data(self):
            return self.payload

    class Image:
        def save(self, buffer, _format):
            buffer.payload = b"PNG"
            return True

        def width(self):
            return 30

        def height(self):
            return 40

    class Geometry:
        def x(self):
            return 0

        def y(self):
            return 0

        def width(self):
            return 1920

        def height(self):
            return 1080

    class Screen:
        def __init__(self) -> None:
            self.calls: list[tuple[object, ...]] = []

        def geometry(self):
            return Geometry()

        def grabWindow(self, *args):  # noqa: N802
            self.calls.append(args)
            return Image()

    primary = Screen()

    class Application:
        @staticmethod
        def instance():
            return application

        def primaryScreen(self):  # noqa: N802
            return primary

        def screens(self):
            return (primary,)

    application = Application()
    qt_gui = SimpleNamespace(QGuiApplication=Application)
    qt_core = SimpleNamespace(
        QBuffer=Buffer,
        QIODevice=SimpleNamespace(OpenModeFlag=SimpleNamespace(WriteOnly=1)),
    )
    monkeypatch.setattr(
        linux_platform,
        "_import_optional",
        lambda name: qt_gui if name == "PySide6.QtGui" else qt_core,
    )

    platform = linux_platform.LinuxDesktopPlatform({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":99"})
    result = platform.capture_screen(
        scope="region",
        region={"x": 10, "y": 20, "width": 30, "height": 40},
    )

    assert result["status"] == "available"
    assert primary.calls == [(0, 10, 20, 30, 40)]


def test_x11_capture_and_click_reject_implicit_numeric_coercion(monkeypatch) -> None:
    platform = linux_platform.LinuxDesktopPlatform({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":99"})

    assert platform.list_processes(limit="10") == {
        "status": "unavailable",
        "processes": [],
        "reason": "process limit is invalid",
    }
    assert platform.list_processes(limit=True) == {
        "status": "unavailable",
        "processes": [],
        "reason": "process limit is invalid",
    }
    assert platform.capture_screen(scope="window", window_id="10")["status"] == "unavailable"
    assert (
        platform.capture_screen(
            scope="region", region={"x": 0, "y": 0, "width": "10", "height": 10}
        )["status"]
        == "unavailable"
    )
    assert platform.click_at(1.5, 2)["status"] == "unavailable"
    assert platform.click_at("1", 2)["status"] == "unavailable"
    assert platform.click_at(1, 2, button=None)["status"] == "unavailable"


def test_x11_window_context_reads_ewmh_and_process(monkeypatch) -> None:
    class FakeWindow:
        def get_full_property(self, atom, _property_type):
            values = {
                "_NET_WM_NAME": [b"Editor"],
                "WM_NAME": [b"Fallback"],
                "WM_CLASS": [b"editor", b"org.Editor"],
                "_NET_WM_PID": [4242],
            }
            return SimpleNamespace(value=values.get(atom))

        def get_geometry(self):
            return SimpleNamespace(x=2, y=3, width=800, height=600)

        def translate_coords(self, _root, _x, _y):
            return SimpleNamespace(x=12, y=23)

    class FakeRoot:
        def get_full_property(self, _atom, _property_type):
            return SimpleNamespace(value=[0x123])

    class FakeDisplay:
        def __init__(self, _display_name):
            self.closed = False

        def screen(self):
            return SimpleNamespace(root=FakeRoot())

        def intern_atom(self, name, only_if_exists=True):
            return name if only_if_exists else name

        def create_resource_object(self, _kind, _window_id):
            return FakeWindow()

        def close(self):
            self.closed = True

    class FakeProcess:
        def __init__(self, _pid):
            pass

        def name(self):
            return "editor"

        def exe(self):
            return "/usr/bin/editor"

        def cmdline(self):
            return ("editor", "--safe")

    fake_display_module = SimpleNamespace(Display=FakeDisplay)
    fake_x_module = SimpleNamespace(AnyPropertyType=0)
    fake_psutil = SimpleNamespace(Process=FakeProcess)

    def fake_import(name: str):
        return {
            "Xlib.display": fake_display_module,
            "Xlib.X": fake_x_module,
            "psutil": fake_psutil,
        }.get(name)

    monkeypatch.setattr(linux_platform, "_import_optional", fake_import)
    context = linux_platform.X11WindowContextProvider(
        {"DISPLAY": ":99"}, include_command_line=True
    ).read()
    assert context is not None
    assert context.window_id == "0x123"
    assert context.title == "Editor"
    assert context.app_id == "org.Editor"
    assert context.pid == 4242
    assert context.process_name == "editor"
    assert context.command_line == ("editor", "--safe")
    assert context.geometry == (12, 23, 800, 600)


def test_x11_probe_without_qt_does_not_claim_overlay_support(monkeypatch) -> None:
    def fake_import(name: str):
        return object() if name in {"Xlib.display", "psutil"} else None

    monkeypatch.setattr(linux_platform, "_import_optional", fake_import)
    snapshot = linux_platform.probe_linux_platform({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":99"})
    assert snapshot.capability("active_window").available
    assert snapshot.capability("process_context").available
    assert snapshot.capability("click_through").state is CapabilityState.UNAVAILABLE
    assert snapshot.capability("overlay_position").state is CapabilityState.UNAVAILABLE


def test_linux_platform_snapshots_excluded_window_ids_before_provider(monkeypatch) -> None:
    platform = linux_platform.LinuxDesktopPlatform({"DISPLAY": ":99"})
    platform.exclude_window_id(101)
    observed: list[object] = []

    class FakeProvider:
        def __init__(self, _environment, *, include_command_line, excluded_window_ids):
            del _environment, include_command_line
            observed.append(excluded_window_ids)

        def read(self):
            return None

    monkeypatch.setattr(linux_platform, "X11WindowContextProvider", FakeProvider)
    assert platform.active_window() is None
    assert observed == [(101,)]


def test_linux_exclude_window_id_ignores_invalid_handles() -> None:
    platform = linux_platform.LinuxDesktopPlatform({"DISPLAY": ":99"})

    platform.exclude_window_id(True)
    platform.exclude_window_id("not-a-window")
    platform.exclude_window_id(0)
    platform.exclude_window_id(-1)

    assert platform._excluded_window_ids == set()


def test_web_live2d_probe_requires_model_and_qt(tmp_path: Path) -> None:
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True)
    for name in ("pixi.min.js", "pixi-live2d-display.min.js", "live2dcubismcore.min.js"):
        (javascript_root / name).write_text("/* test */", encoding="utf-8")

    modules = {
        "PySide6.QtCore": object(),
        "PySide6.QtWebEngineWidgets": object(),
        "PySide6.QtWebChannel": object(),
    }

    def fake_import(name: str):
        return modules[name]

    missing_model = probe_web_live2d(tmp_path, importer=fake_import)
    assert not missing_model.available
    assert "model" in missing_model.message.lower()

    model_root = tmp_path / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    (model_root / "legacy.model.json").write_text("{}", encoding="utf-8")
    legacy_probe = probe_web_live2d(tmp_path, importer=fake_import)
    assert not legacy_probe.available
    assert any(".model.json descriptors are unsupported" in item for item in legacy_probe.warnings)
    (model_root / "legacy.model.json").unlink()
    _write_model3_descriptor(model_root)
    available = probe_web_live2d(tmp_path, importer=fake_import)
    assert available.available
    assert available.model_path is not None
    assert available.warnings == ()

    renderer = WebLive2DRenderer(tmp_path, importer=fake_import)
    assert renderer.capabilities.available
    assert renderer.capabilities.expressions[:6] == (
        "neutral",
        "happy",
        "sad",
        "curious",
        "surprised",
        "shy",
    )
    assert renderer.capabilities.motions == ("idle", "blink", "wave", "walk")
    renderer._view = object()
    assert renderer.set_expression("happy")
    assert renderer.play_motion("wave")
    # 描述没有任何正式动作声明时，保留模型运行时自定义名称的 pending
    # 兼容路径；真正加载并声明动作后再由能力边界拒绝未知名称。
    assert renderer.set_expression("custom-expression")
    assert renderer.play_motion("custom-motion")
    html = renderer._html_document()
    assert "Live2DModel.from" in html
    assert "meapetBridge" in html
    assert 'src="live2d/js/pixi.min.js"' in html
    assert html.index('src="live2d/js/live2dcubismcore.min.js"') < html.index(
        'src="live2d/js/pixi-live2d-display.min.js"'
    )
    # Pixi 7 不再把旧的 ``transparent`` 选项映射到 WebGL alpha；
    # 保持默认不透明会在角色后面绘制黑色矩形。
    assert "backgroundAlpha: 0" in html
    assert "transparent: true" not in html
    assert "model.renderOrders" in html
    assert "function repairRenderOrders()" in html
    assert "typeof core.getRenderOrders === 'function'" in html
    assert "expected.length < count" in html
    assert "Live2D render order validation failed" in html
    assert "pixi-live2d-display-lipsyncpatch.min.js" not in html
    assert "__meapetBaseSize" in html
    assert "liveModel.rotation = 0;" in html
    assert "liveModel.scale.set(referenceScale, referenceScale)" in html
    assert "resize: function()" in html
    assert "proceduralMotions" in html
    assert "proceduralExpressions" in html
    assert "applyProceduralExpression" in html
    assert "applyProceduralMotion" in html
    assert "liveModel.scale.set(geometryFacingSign() * uniformScale, uniformScale)" in html
    assert "FACING_SWITCH_THRESHOLD" in html
    assert "facingSign: geometryFacingSign()" in html
    assert "setDirection" in html
    assert "resolution: Math.max(1, window.devicePixelRatio || 1)" in html
    assert "await PIXI.live2d.cubism4Ready()" in html
    assert "autoUpdate: false" in html
    assert "liveModel.internalModel.update(dt * 1000, performance.now())" in html
    assert "const externalClockHoldMs = 110" in html
    assert "advance: function(seconds, source)" in html
    assert "window.meapetLive2D.advance(dt, 'raf');" in html
    assert "window.meapetLive2D && window.meapetLive2D.advance" in html
    assert "window.requestAnimationFrame(animationTick)" in html
    assert "startAnimationLoop()" in html
    assert "pointerEvent" in html
    assert "installPointerBridge" in html
    assert "pointerdown" in html
    assert "event.key === 'Enter'" in html
    assert "enter_key" in html
    assert "pointercancel" in html
    assert "screen_x" in html
    assert "screen_y" in html
    assert "pressScreenX" in html
    assert "currentScreenX" in html
    assert "pressPayload.hit = true" in html
    assert "contextPayload.hit = true" in html
    assert "transparentPayload.hit = false" in html
    assert "hoverPayload.hit = nextHoverHit" in html
    assert "hoverProbeIntervalMs" in html
    assert "if (now - hoverProbeAt < hoverProbeIntervalMs) return;" in html
    assert "collectAll: false, sampleRenderedAlpha: false" in html
    # 页面事件跨越 WebChannel 到达 Qt 前可能已经经过数帧；拖动阈值一旦
    # 越过，必须先在页面本地冻结 rAF，再等待宿主确认，避免起始抽搐。
    assert "let localDragging = false;" in html
    assert "function setLocalDragging(value)" in html
    assert "setLocalDragging(true);" in html
    assert "setLocalDragging(false);" in html
    assert html.index("setLocalDragging(true);") < html.index("send('move', event);")
    assert "const localDragReleaseDelayMs = 90;" in html
    assert "function releaseLocalDragging()" in html
    assert "clockState.dragReleaseUntil" in html
    assert html.index("setLocalDragging(true);") < html.index("sendPayload(pressPayload);")
    assert html.index("send(type, event);") < html.index("releaseLocalDragging();")
    assert "drawableMinX" in html
    assert "triangleMinX" in html
    assert "if (!collectAll) break;" in html
    assert "if (!hitDetails.hit) return;" in html
    assert "setCursorTarget" in html
    assert "setInteractionLocked" in html
    assert "cursorState" in html
    assert "smoothTrackingAxis" in html
    assert "requestedDt" in html
    assert "largeDeltaClamps" in html
    assert "poseParameterLimit" in html
    assert "safeOffsetX" in html
    assert "referenceScale" in html
    assert "needsRepair" in html
    assert "Number.isInteger(order)" in html
    assert "expectedSet" in html
    assert "sequenceLike(actual || expected, expectedValues)" in html
    assert "facingState" in html
    assert "notifyWindowMoved" in html
    # 异步 exp3/motion3 加载必须由请求代次隔离；旧 Promise 不能覆盖后来的动作。
    assert "let expressionRequestGeneration = 0;" in html
    assert "let motionRequestGeneration = 0;" in html
    assert "const requestGeneration = ++expressionRequestGeneration;" in html
    assert "const requestGeneration = ++motionRequestGeneration;" in html
    assert "requestGeneration === expressionRequestGeneration" in html
    assert "requestGeneration === motionRequestGeneration" in html
    assert "lastGoodTransform" in html
    assert "enforceUniformTransform" in html
    assert "uniformScaleCorrections" in html
    assert "drawableGeometryBounds" in html
    assert "validateDrawableFrame" in html
    assert "const maskCounts = drawables.maskCounts" in html
    assert "maskIndex === drawable" in html
    assert "refreshCoreGeometry" in html
    assert "checkGeometryContinuity" in html
    assert "geometryContinuityRejects" in html
    assert "GEOMETRY_MAX_RATIO" in html
    assert "worldGeometryBounds" in html
    assert "const fitRatio = Math.min(" in html
    assert "fitScale = Math.min" not in html
    assert "reason: 'model-hidden'" in html
    assert "viewportStale" in html
    assert "ResizeObserver" in html
    assert "synchronizeViewport" in html
    assert "fitModelToViewport" in html
    assert "clampValue" in html
    assert "now - lastDoubleAt < doubleClickWindowMs" in html
    assert f"}}, {PET_CLICK_DEBOUNCE_MS});" in html
    assert "hitTestDrawableGeometry" in html
    assert "vertexPositions" in html
    assert "textureAlphaSource" in html
    assert "triangleBarycentric" in html
    assert "drawableGeometryDetailsWithTolerance" in html
    assert "getImageData(0, 0, width, height)" in html
    # 真实绑定探测使用 Cubism Core 的顶点快照；命中测试仍直接读取
    # raw vertexPositions，不调用旧版 containsPoint 封装。
    assert "getDrawableVertexPositions" in html
    assert "containsPoint" not in html
    assert "expression: expressionState.name" in html
    assert "motion: motionState.name" in html
    assert "max-height:min(48px, calc(100vh - 12px))" in html
    assert "overflow-y:auto" in html
    assert "bubble.classList.toggle('scrollable'" in html
    assert "box-sizing:border-box" in html
    assert "right:5px;top:50%" in html
    assert "lines.join('\\n')" in html
    renderer._set_unavailable("synthetic page failure", notify=False)
    assert renderer.capabilities.expressions
    assert renderer.capabilities.motions == ("idle", "blink", "wave", "walk")


def test_web_live2d_probe_prefers_declared_expression_and_motion_files(tmp_path: Path) -> None:
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True)
    for name in ("pixi.min.js", "pixi-live2d-display.min.js", "live2dcubismcore.min.js"):
        (javascript_root / name).write_text("/* test */", encoding="utf-8")
    model_root = tmp_path / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    _write_model3_descriptor(model_root)
    (model_root / "happy.exp3.json").write_text(
        json.dumps(
            {
                "Version": 3,
                "Parameters": [{"Id": "ParamEyeLOpen", "Value": 0.0, "Blend": "Overwrite"}],
            }
        ),
        encoding="utf-8",
    )
    motion = {
        "Version": 3,
        "Meta": {
            "Duration": 1.0,
            "Fps": 30.0,
            "Loop": False,
            "CurveCount": 1,
            "TotalSegmentCount": 1,
            "TotalPointCount": 2,
            "UserDataCount": 0,
            "TotalUserDataSize": 0,
        },
        "Curves": [
            {"Target": "Parameter", "Id": "ParamAngleX", "Segments": [0.0, 0.0, 0, 1.0, 10.0]}
        ],
    }
    (model_root / "idle.motion3.json").write_text(json.dumps(motion), encoding="utf-8")
    (model_root / "tap.motion3.json").write_text(json.dumps(motion), encoding="utf-8")
    descriptor = json.loads((model_root / "demo.model3.json").read_text(encoding="utf-8"))
    descriptor["FileReferences"].update(
        {
            "Expressions": [
                {"Name": "happy_real", "File": "happy.exp3.json"},
                {"Name": "happy", "File": "happy.exp3.json"},
                {
                    "Name": "absolute_must_be_ignored",
                    "File": str((model_root / "happy.exp3.json").resolve()),
                },
            ],
            "Motions": {
                "Idle": [{"File": "idle.motion3.json"}],
                "Tap": [{"File": "tap.motion3.json"}],
            },
        }
    )
    (model_root / "demo.model3.json").write_text(json.dumps(descriptor), encoding="utf-8")
    (model_root / "demo.actions.yaml").write_text(
        "version: 1\nmotions:\n  blink: Idle\n  invalid: Missing\n",
        encoding="utf-8",
    )
    modules = {
        "PySide6.QtCore": object(),
        "PySide6.QtWebEngineWidgets": object(),
        "PySide6.QtWebChannel": object(),
    }
    probe = probe_web_live2d(tmp_path, importer=lambda name: modules[name])
    assert probe.available
    assert probe.expressions == ("happy_real", "happy")
    assert probe.motions == ("Idle", "Tap")
    assert probe.motion_aliases == (("blink", "Idle"),)
    renderer = WebLive2DRenderer(tmp_path, importer=lambda name: modules[name])
    assert renderer.capabilities.expressions[:2] == ("happy_real", "happy")
    assert renderer.capabilities.motions[:3] == ("blink", "Idle", "Tap")
    html = renderer._html_document()
    # 模型运行期热切换需要替换这两个集合，因此它们是可变绑定。
    assert 'let declaredExpressions = new Set(["happy_real", "happy"]);' in html
    assert 'let declaredMotions = new Set(["Idle", "Tap"]);' in html
    assert 'let motionAliases = new Map(Object.entries({"blink": "Idle"}));' in html
    assert "if (!declaredExpressions.has(requested))" in html
    assert "if (!declaredMotions.has(resolvedMotion))" in html
    assert "motionAliases.get(requested.toLowerCase())" in html
    assert "liveModel.motion(resolvedMotion, 0, forcePriority)" in html
    assert "liveModel.expression(requested)" in html
    assert "if (liveModel && expressionState.procedural)" in html
    assert "if (motionState.procedural) applyProceduralMotion(dt);" in html
    assert "motionState.procedural = false" in html


@pytest.mark.xvfb
@pytest.mark.webengine
def test_web_live2d_long_bubble_is_scrollable_and_bounded(tmp_path: Path) -> None:
    """真实 WebEngine DOM 不得让长气泡覆盖整个模型视口。"""

    if shutil.which("xvfb-run") is None:
        pytest.skip("requires xvfb-run")
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = f"""
import json
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.renderers.web_live2d import WebLive2DRenderer

app = QApplication(sys.argv)
renderer = WebLive2DRenderer({str(project_root / "resources")!r})
if not renderer.initialize() or renderer.view is None:
    raise SystemExit("web-live2d-init-failed")
view = renderer.view
view.resize(420, 480)
view.show()

def finish(payload):
    print(json.dumps(payload, ensure_ascii=False))
    renderer.shutdown()
    view.close()
    app.quit()

def query():
    view.page().runJavaScript(
        '''(() => {{
          const bubble = document.getElementById('meapet-bubble');
          const debug = window.meapetLive2D && window.meapetLive2D.debug();
          return JSON.stringify({{
            rect: bubble.getBoundingClientRect().toJSON(),
            scrollHeight: bubble.scrollHeight,
            clientHeight: bubble.clientHeight,
            overflowY: getComputedStyle(bubble).overflowY,
            debug: debug
          }});
        }})()''',
        lambda value: finish(json.loads(value)),
    )

def wait_ready():
    if not renderer._model_ready:
        QTimer.singleShot(250, wait_ready)
        return
    renderer.set_speech("长文本测试" * 800)
    QTimer.singleShot(300, query)

QTimer.singleShot(1000, wait_ready)
QTimer.singleShot(12000, lambda: finish({{"timeout": True}}))
app.exec()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader "
        "--disable-gpu-compositing --disable-gpu-sandbox"
    )
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24",
            sys.executable,
            "-c",
            script,
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload.get("timeout") is not True, result.stderr
    assert payload["rect"]["height"] <= 48.0
    assert payload["scrollHeight"] > payload["clientHeight"]
    assert payload["overflowY"] == "auto"
    assert payload["debug"]["boundsHeight"] > 0


@pytest.mark.xvfb
@pytest.mark.webengine
def test_web_live2d_expression_and_motion_parameters_do_not_leak(tmp_path: Path) -> None:
    """真实帧只报告已探测的 Cubism 绑定，其余动作走页面变换。"""

    if shutil.which("xvfb-run") is None:
        pytest.skip("requires xvfb-run")
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = f"""
import json
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.renderers.web_live2d import WebLive2DRenderer

app = QApplication(sys.argv)
renderer = WebLive2DRenderer({str(project_root / "resources")!r})
original_html = renderer._html_document
renderer._html_document = lambda: original_html().replace(
    "liveModel = await loadModelResource(modelUrl);",
    "liveModel = await loadModelResource(modelUrl);"
    " window.__auditModel = liveModel;",
)
if not renderer.initialize() or renderer.view is None:
    raise SystemExit("web-live2d-init-failed")
view = renderer.view
view.resize(420, 480)
view.show()

def finish(payload):
    print(json.dumps(payload, ensure_ascii=False))
    renderer.shutdown()
    view.close()
    app.quit()

def query():
    view.page().runJavaScript(
        '''(() => {{
          const race = window.__asyncRace || {{}};
          const model = window.__auditModel;
          const core = model.internalModel.coreModel;
          const ids = [
            "ParamEyeBallX", "ParamEyeBallY",
            "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
            "ParamBreath", "ParamHairFront", "ParamHairSide",
            "ParamAngleX", "ParamAngleY"
          ];
          const values = () => Object.fromEntries(
            ids.map((id) => [id, core.getParameterValueById(id)])
          );
          const vertices = () => Array.from(
            {{length: core.getDrawableCount()}},
            (_unused, index) => Array.from(core.getDrawableVertexPositions(index))
          );
          const vertexDelta = (before, after) => {{
            let changed = 0;
            let total = 0;
            for (let index = 0; index < before.length; index += 1) {{
              for (let item = 0; item < before[index].length; item += 1) {{
                const delta = Math.abs(before[index][item] - after[index][item]);
                if (delta > 1e-7) changed += 1;
                total += delta;
              }}
            }}
            return {{changed, total}};
          }};
          const result = {{async_race: race}};
          let previousVertices = vertices();
          ["curious", "surprised", "shy", "neutral"].forEach((name) => {{
            window.meapetLive2D.setExpression(name);
            core.update();
            const currentVertices = vertices();
            result["vertex_expression_" + name] = vertexDelta(
              previousVertices, currentVertices
            );
            previousVertices = currentVertices;
            result["expression_" + name] = values();
          }});
          ["wave", "walk", "idle"].forEach((name) => {{
            window.meapetLive2D.playMotion(name);
            window.meapetLive2D.advance(0.2);
            core.update();
            const currentVertices = vertices();
            result["vertex_motion_" + name] = vertexDelta(
              previousVertices, currentVertices
            );
            previousVertices = currentVertices;
            result["motion_" + name] = values();
          }});
          const timelineBaseline = values().ParamAngleX;
          result.timelineAccepted = window.meapetLive2D.setExpressionRequest({{
            mode: "sequence", restore: "neutral", loop: false,
            expressions: [
              {{name: "happy", weight: 1, duration_seconds: 0.1,
                transition_seconds: 0.1, parameters: {{ParamAngleX: 4}}}},
              {{name: "sad", weight: 1, duration_seconds: 0.1,
                transition_seconds: 0.1, parameters: {{ParamAngleX: -4}}}}
            ]
          }});
          window.meapetLive2D.advance(0.05);
          result.timelineFirstHalf = values().ParamAngleX;
          window.meapetLive2D.advance(0.05);
          result.timelineFirstBoundary = values().ParamAngleX;
          window.meapetLive2D.advance(0.05);
          result.timelineSecondHalf = values().ParamAngleX;
          window.meapetLive2D.advance(0.05);
          result.timelineRestoreStart = values().ParamAngleX;
          window.meapetLive2D.advance(0.05);
          result.timelineRestoreHalf = values().ParamAngleX;
          window.meapetLive2D.advance(0.05);
          result.timelineRestored = values().ParamAngleX;
          result.timelineDebug = window.meapetLive2D.debug();
          result.timelineBaseline = timelineBaseline;
          result.motionRequestAccepted = window.meapetLive2D.playMotionRequest({{
            name: "wave", duration_seconds: 0.2, transition_seconds: 0.1,
            loop: true, parameters: {{ParamAngleX: 2}}
          }});
          window.meapetLive2D.advance(0.05);
          result.motionRequestHalf = values().ParamAngleX;
          window.meapetLive2D.advance(0.05);
          window.meapetLive2D.advance(0.05);
          window.meapetLive2D.advance(0.05);
          result.motionRequestLoopDebug = window.meapetLive2D.debug();
          window.meapetLive2D.playMotion("idle");
          result.motionRequestStoppedDebug = window.meapetLive2D.debug();
          result.motionRequestRestored = values().ParamAngleX;
          return JSON.stringify(result);
        }})()''',
        lambda value: finish(json.loads(value)),
    )

def run_async_race():
    # 用两个可控 Promise 模拟真实 exp3/motion3 的异步加载顺序：后发的 B
    # 先完成，旧的 A 再完成；最终状态必须仍然是 B。
    view.page().runJavaScript(
        '''(() => {{
          const model = window.__auditModel;
          const originalExpression = model.expression;
          const originalMotion = model.motion;
          const expressionResolvers = {{}};
          const motionResolvers = {{}};
          model.expression = (name) => new Promise((resolve) => {{
            expressionResolvers[name] = resolve;
          }});
          model.motion = (name) => new Promise((resolve) => {{
            motionResolvers[name] = resolve;
          }});
          window.meapetLive2D.setExpression("async-expression-A");
          window.meapetLive2D.setExpression("async-expression-B");
          window.meapetLive2D.playMotion("async-motion-A");
          window.meapetLive2D.playMotion("async-motion-B");
          expressionResolvers["async-expression-B"](true);
          motionResolvers["async-motion-B"](true);
          expressionResolvers["async-expression-A"](true);
          motionResolvers["async-motion-A"](true);
          setTimeout(() => {{
            const debug = window.meapetLive2D.debug();
            window.__asyncRace = {{
              expression: debug.expression,
              motion: debug.motion
            }};
            model.expression = originalExpression;
            model.motion = originalMotion;
          }}, 0);
          return true;
        }})()''',
        lambda _value: QTimer.singleShot(250, query),
    )

def wait_ready():
    if not renderer._model_ready:
        QTimer.singleShot(250, wait_ready)
        return
    QTimer.singleShot(300, run_async_race)

QTimer.singleShot(1000, wait_ready)
QTimer.singleShot(12000, lambda: finish({{"timeout": True}}))
app.exec()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader "
        "--disable-gpu-compositing --disable-gpu-sandbox"
    )
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24",
            sys.executable,
            "-c",
            script,
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert not isinstance(payload, dict) or payload.get("timeout") is not True, result.stderr
    assert isinstance(payload, dict)
    assert payload["async_race"] == {
        "expression": "async-expression-B",
        "motion": "async-motion-B",
    }
    assert abs(payload["expression_surprised"]["ParamEyeBallX"]) < 0.001
    assert abs(payload["expression_shy"]["ParamEyeBallX"]) < 0.001
    assert abs(payload["expression_neutral"]["ParamEyeBallY"]) < 0.001
    assert abs(payload["expression_curious"]["ParamAngleX"]) > 0.1
    assert abs(payload["expression_surprised"]["ParamAngleY"]) > 0.1
    assert payload["vertex_expression_curious"]["changed"] > 0
    assert payload["vertex_expression_surprised"]["changed"] > 0
    assert payload["vertex_expression_shy"]["changed"] > 0
    # 当前运行时探测未开放的身体/头发参数保持中性；兼容 wave/walk
    # 可以只使用外层统一平移/旋转，不能强制要求 Core 顶点变化。
    assert abs(payload["motion_wave"]["ParamBodyAngleZ"]) < 0.001
    assert abs(payload["motion_walk"]["ParamBodyAngleZ"]) < 0.001
    assert abs(payload["motion_idle"]["ParamBodyAngleX"]) < 0.001
    assert abs(payload["motion_idle"]["ParamBodyAngleY"]) < 0.001
    assert abs(payload["motion_idle"]["ParamBodyAngleZ"]) < 0.001
    assert abs(payload["motion_walk"]["ParamBodyAngleX"]) < 0.001
    assert abs(payload["motion_wave"]["ParamAngleX"]) > 0.1
    assert abs(payload["motion_walk"]["ParamAngleX"]) > 0.1
    assert payload["vertex_motion_wave"]["changed"] >= 0
    assert payload["vertex_motion_walk"]["changed"] >= 0
    assert payload["timelineAccepted"] is True
    timeline_baseline = payload["timelineBaseline"]
    assert payload["timelineFirstHalf"] == pytest.approx((timeline_baseline + 4.0) / 2.0, abs=0.15)
    assert payload["timelineFirstBoundary"] == pytest.approx(4.0, abs=0.15)
    assert payload["timelineSecondHalf"] == pytest.approx(0.0, abs=0.15)
    assert payload["timelineRestoreStart"] == pytest.approx(-4.0, abs=0.15)
    assert payload["timelineRestoreHalf"] == pytest.approx(
        (-4.0 + timeline_baseline) / 2.0, abs=0.15
    )
    assert payload["timelineRestored"] == pytest.approx(timeline_baseline, abs=0.15)
    assert payload["timelineDebug"]["expression"] == "neutral"
    assert payload["timelineDebug"]["expressionTimelineActive"] is False
    assert payload["motionRequestAccepted"] is True
    assert payload["motionRequestHalf"] == pytest.approx((timeline_baseline + 2.0) / 2.0, abs=0.2)
    assert payload["motionRequestLoopDebug"]["motionRequestActive"] is True
    assert payload["motionRequestLoopDebug"]["motionRequestLoop"] is True
    assert payload["motionRequestLoopDebug"]["motionRequestElapsed"] == pytest.approx(0.0)
    assert payload["motionRequestStoppedDebug"]["motionRequestActive"] is False
    assert payload["motionRequestRestored"] == pytest.approx(timeline_baseline, abs=0.15)


@pytest.mark.xvfb
@pytest.mark.webengine
def test_web_live2d_real_frame_keeps_alpha_and_uniform_scale(tmp_path: Path) -> None:
    """真实 WebEngine 合成帧不能退化成黑底或非等比缩放。"""

    if shutil.which("xvfb-run") is None:
        pytest.skip("requires xvfb-run")
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = f"""
import json
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.renderers.web_live2d import WebLive2DRenderer

app = QApplication(sys.argv)
renderer = WebLive2DRenderer({str(project_root / "resources")!r})
if not renderer.initialize() or renderer.view is None:
    raise SystemExit("web-live2d-init-failed")
view = renderer.view
view.resize(420, 480)
view.show()

def finish(payload):
    print(json.dumps(payload, ensure_ascii=False))
    renderer.shutdown()
    view.close()
    app.quit()

def capture(attempt=0):
    image = view.grab().toImage()
    left = image.width()
    top = image.height()
    right = bottom = -1
    opaque_black = 0
    for y in range(image.height()):
        for x in range(image.width()):
            color = image.pixelColor(x, y)
            alpha = color.alpha()
            if alpha <= 0:
                continue
            left = min(left, x)
            top = min(top, y)
            right = max(right, x)
            bottom = max(bottom, y)
            if alpha >= 250 and color.red() <= 2 and color.green() <= 2 and color.blue() <= 2:
                opaque_black += 1
    if right < 0 and attempt < 8:
        QTimer.singleShot(150, lambda: capture(attempt + 1))
        return
    view.page().runJavaScript(
        "JSON.stringify(window.meapetLive2D && window.meapetLive2D.debug())",
        lambda value: finish({{
            "alpha_bbox": None if right < 0 else [left, top, right + 1, bottom + 1],
            "opaque_black": opaque_black,
            "debug": json.loads(value),
        }}),
    )

def wait_ready():
    if not renderer._model_ready:
        QTimer.singleShot(100, wait_ready)
        return
    renderer.set_expression("curious")
    renderer.play_motion("wave")
    renderer.set_cursor_target(0, 480, 420, 480)
    QTimer.singleShot(350, capture)

QTimer.singleShot(1000, wait_ready)
QTimer.singleShot(12000, lambda: finish({{"timeout": True}}))
app.exec()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader "
        "--disable-gpu-compositing --disable-gpu-sandbox"
    )
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24",
            sys.executable,
            "-c",
            script,
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload.get("timeout") is not True, result.stderr
    assert payload["alpha_bbox"] is not None
    assert payload["opaque_black"] == 0
    assert abs(payload["debug"]["scaleX"]) == abs(payload["debug"]["scaleY"])
    assert payload["debug"]["boundsWidth"] > 0
    assert payload["debug"]["boundsHeight"] > 0
    assert payload["debug"]["uniformScaleCorrections"] == 0
    assert -8 <= payload["debug"]["angleX"] <= 8
    assert -7 <= payload["debug"]["angleY"] <= 7
    assert payload["debug"]["boundsX"] >= 0
    assert payload["debug"]["boundsY"] >= 0
    assert payload["debug"]["boundsX"] + payload["debug"]["boundsWidth"] <= 420
    assert payload["debug"]["boundsY"] + payload["debug"]["boundsHeight"] <= 480


@pytest.mark.xvfb
@pytest.mark.webengine
def test_web_live2d_resize_during_motion_keeps_scale_baseline(tmp_path: Path) -> None:
    """动作进行中调整窗口不能因瞬态 keyform 让模型缩放抽搐。"""

    if shutil.which("xvfb-run") is None:
        pytest.skip("requires xvfb-run")
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = f"""
import json
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.renderers.web_live2d import WebLive2DRenderer

app = QApplication(sys.argv)
renderer = WebLive2DRenderer({str(project_root / "resources")!r})
if not renderer.initialize() or renderer.view is None:
    raise SystemExit("web-live2d-init-failed")
view = renderer.view
view.resize(420, 480)
view.show()
actions = [("walk", 420, 480), ("blink", 421, 480), ("wave", 420, 481)] * 2
samples = []
index = 0

def finish(payload=None):
    if payload is None:
        payload = {{"timeout": True}}
    print(json.dumps(payload, ensure_ascii=False))
    renderer.shutdown()
    view.close()
    app.quit()

def step():
    global index
    if index >= len(actions):
        scales = [item["scale"] for item in samples]
        finish({{"scales": scales, "samples": len(samples)}})
        return
    motion, width, height = actions[index]
    index += 1
    renderer.play_motion(motion)
    view.resize(width, height)
    view.page().runJavaScript(
        f"window.meapetLive2D.resize({{width}},{{height}},true); "
        "JSON.stringify(window.meapetLive2D.debug())",
        lambda value: collect(motion, value),
    )

def collect(motion, value):
    try:
        debug = json.loads(str(value or "{{}}"))
    except (TypeError, ValueError):
        debug = {{}}
    if isinstance(debug, dict):
        samples.append({{
            "motion": motion,
            "scale": abs(float(debug.get("scaleX", 0) or 0)),
            "scale_y": abs(float(debug.get("scaleY", 0) or 0)),
            "bounds_width": float(debug.get("boundsWidth", 0) or 0),
            "bounds_height": float(debug.get("boundsHeight", 0) or 0),
            "geometry": debug.get("geometryContinuityValid"),
        }})
    QTimer.singleShot(160, step)

def wait_ready():
    if not renderer._model_ready:
        QTimer.singleShot(100, wait_ready)
        return
    QTimer.singleShot(200, step)

QTimer.singleShot(1000, wait_ready)
QTimer.singleShot(15000, finish)
app.exec()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader --disable-gpu-compositing --disable-gpu-sandbox"
    )
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24",
            sys.executable,
            "-c",
            script,
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload.get("timeout") is not True, result.stderr
    assert payload["samples"] == 6
    scales = payload["scales"]
    assert max(scales) - min(scales) < 0.002


@pytest.mark.xvfb
@pytest.mark.webengine
def test_web_live2d_hit_test_rejects_transparent_texture_meshes(tmp_path: Path) -> None:
    """点击命中必须遵循纹理 alpha，而不是把整块 ArtMesh 当成实体。"""

    if shutil.which("xvfb-run") is None:
        pytest.skip("requires xvfb-run")
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        pytest.skip("requires PySide6")

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    points_json = json.dumps([[5, 5], [80, 250], [210, 140], [210, 260], [200, 390], [270, 358]])
    script = f"""
import json
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.renderers.web_live2d import WebLive2DRenderer

app = QApplication(sys.argv)
renderer = WebLive2DRenderer({str(project_root / "resources")!r})
if not renderer.initialize() or renderer.view is None:
    raise SystemExit("web-live2d-init-failed")
view = renderer.view
view.resize(420, 480)
view.show()
def finish(payload):
    print(json.dumps(payload, ensure_ascii=False))
    renderer.shutdown()
    view.close()
    app.quit()

direction_results = {{}}

def query(direction):
    script = "window.meapetLive2D.setDirection(" + json.dumps(direction) + ");"
    script += "JSON.stringify(" + {points_json!r} + ".map(([x, y]) => {{"
    script += "const result = window.meapetLive2D.hitTest(x, y);"
    script += "return {{x, y, hit: Boolean(result && result.hit), part: result && result.part}};"
    script += "}}))"
    view.page().runJavaScript(
        script,
        lambda value: receive_direction(direction, json.loads(value)),
    )

def receive_direction(direction, payload):
    direction_results[direction] = payload
    if direction == "A":
        QTimer.singleShot(350, lambda: query("B"))
        return
    finish(direction_results)

def wait_ready():
    if not renderer._model_ready:
        QTimer.singleShot(100, wait_ready)
        return
    QTimer.singleShot(350, lambda: query("A"))

QTimer.singleShot(500, wait_ready)
QTimer.singleShot(12000, lambda: finish({{"timeout": True}}))
app.exec()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader "
        "--disable-gpu-compositing --disable-gpu-sandbox"
    )
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24",
            sys.executable,
            "-c",
            script,
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert isinstance(payload, dict)
    assert set(payload) == {"A", "B"}
    for direction in ("A", "B"):
        values = payload[direction]
        assert [item["hit"] for item in values] == [False, False, True, True, True, True]
        assert values[2]["part"] == "head"
        assert values[4]["part"] == "lower_left"
        assert values[5]["part"] == "lower_right"


def test_web_live2d_replays_actions_only_after_model_ready(tmp_path: Path) -> None:
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True)
    for name in ("pixi.min.js", "pixi-live2d-display.min.js", "live2dcubismcore.min.js"):
        (javascript_root / name).write_text("/* test */", encoding="utf-8")
    model_root = tmp_path / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    _write_model3_descriptor(model_root)

    def fake_import(_name: str):
        return object()

    renderer = WebLive2DRenderer(tmp_path, importer=fake_import)
    renderer._view = object()
    renderer._page_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append
    assert renderer.set_expression("happy")
    assert renderer.play_motion("wave")
    assert scripts == []
    renderer._on_model_ready()
    assert len(scripts) == 2
    assert "setExpression" in scripts[0]
    assert "playMotion" in scripts[1]


def test_web_live2d_interaction_feedback_is_click_safe_and_redacted(tmp_path: Path) -> None:
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True)
    for name in ("pixi.min.js", "pixi-live2d-display.min.js", "live2dcubismcore.min.js"):
        (javascript_root / name).write_text("/* test */", encoding="utf-8")
    model_root = tmp_path / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    _write_model3_descriptor(model_root)

    renderer = WebLive2DRenderer(tmp_path, importer=lambda _name: object())
    renderer._view = object()
    renderer._page_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append
    assert renderer.set_interaction_feedback(
        {
            "zone": "head",
            "phrase": "喵？摸摸头～",
            "affection": {"current": 6, "applied": 1, "secret": "hidden"},
        }
    )
    assert scripts and "setInteractionFeedback" in scripts[-1]
    assert "hidden" not in scripts[-1]
    assert renderer.set_interaction_feedback(None, visible=False)
    assert "false" in scripts[-1]


def test_web_live2d_keeps_bridge_ready_when_load_finished_arrives_late(tmp_path: Path) -> None:
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True)
    for name in ("pixi.min.js", "pixi-live2d-display.min.js", "live2dcubismcore.min.js"):
        (javascript_root / name).write_text("/* test */", encoding="utf-8")
    model_root = tmp_path / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    _write_model3_descriptor(model_root)

    renderer = WebLive2DRenderer(tmp_path, importer=lambda _name: object())
    renderer._view = object()
    renderer._on_model_ready()
    assert renderer.page_bridge_ready is True
    assert renderer._model_ready is True
    renderer._on_load_finished(True)
    assert renderer.page_bridge_ready is True
    renderer._on_load_finished(False)
    assert renderer.page_bridge_ready is False
    assert renderer._model_ready is False


def test_web_live2d_initialize_wires_qt_bridge_and_load_signal(tmp_path: Path) -> None:
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True)
    for name in ("pixi.min.js", "pixi-live2d-display.min.js", "live2dcubismcore.min.js"):
        (javascript_root / name).write_text("/* test */", encoding="utf-8")
    model_root = tmp_path / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    _write_model3_descriptor(model_root)

    class Signal:
        def __init__(self) -> None:
            self.callback = None

        def connect(self, callback) -> None:
            self.callback = callback

        def emit(self, value: bool) -> None:
            assert self.callback is not None
            self.callback(value)

    class Page:
        def __init__(self) -> None:
            self.channel = None
            self.background = None

        def setWebChannel(self, channel) -> None:  # noqa: N802
            self.channel = channel

        def setBackgroundColor(self, color) -> None:  # noqa: N802
            self.background = color

    class FakeView:
        instance = None

        def __init__(self, _parent=None) -> None:
            self.loadFinished = Signal()  # noqa: N815
            self._page = Page()
            self.html = ""
            self.base_url = None
            self.attributes = []
            self.stylesheet = ""
            self.auto_fill = None
            FakeView.instance = self

        def page(self):
            return self._page

        def setHtml(self, html, base_url) -> None:  # noqa: N802
            self.html = html
            self.base_url = base_url

        def setAttribute(self, attribute, enabled) -> None:  # noqa: N802
            self.attributes.append((attribute, enabled))

        def setAutoFillBackground(self, enabled) -> None:  # noqa: N802
            self.auto_fill = enabled

        def setStyleSheet(self, stylesheet) -> None:  # noqa: N802
            self.stylesheet = stylesheet

        def deleteLater(self) -> None:  # noqa: N802
            pass

    class FakeChannel:
        def __init__(self, _parent=None) -> None:
            self.objects = {}

        def registerObject(self, name, value) -> None:  # noqa: N802
            self.objects[name] = value

    class FakeUrl:
        @staticmethod
        def fromLocalFile(path: str):  # noqa: N802
            return path

    class FakeCore:
        QObject = type("QObject", (), {})
        QUrl = FakeUrl
        Qt = type(
            "Qt",
            (),
            {
                "WA_TranslucentBackground": "translucent",
                "WA_NoSystemBackground": "no-system",
                "WA_OpaquePaintEvent": "opaque",
            },
        )

        @staticmethod
        def Slot(*_types):
            return lambda function: function

    class FakeColor:
        def __init__(self, red, green, blue, alpha) -> None:
            self.rgba = (red, green, blue, alpha)

    modules = {
        "PySide6.QtCore": FakeCore,
        "PySide6.QtGui": type("FakeGui", (), {"QColor": FakeColor}),
        "PySide6.QtWebEngineWidgets": type("FakeWidgets", (), {"QWebEngineView": FakeView}),
        "PySide6.QtWebChannel": type("FakeWebChannel", (), {"QWebChannel": FakeChannel}),
    }

    renderer = WebLive2DRenderer(tmp_path, importer=modules.__getitem__)
    assert renderer.initialize(parent=object()) is True
    view = FakeView.instance
    assert view is not None
    assert view.page().channel is renderer._channel
    assert {attribute for attribute, enabled in view.attributes if enabled} == {
        "translucent",
        "no-system",
    }
    assert ("opaque", False) in view.attributes
    assert view.auto_fill is False
    assert view.stylesheet == "background: transparent;"
    assert view.page().background.rgba == (0, 0, 0, 0)
    assert renderer._channel.objects["meapetBridge"] is renderer._bridge
    assert "demo/demo.model3.json" in view.html
    assert view.base_url == str(tmp_path.resolve()) + "/"

    view.loadFinished.emit(True)
    assert renderer._page_ready is True
    renderer._bridge.modelReady()
    assert renderer._model_ready is True


def test_web_live2d_transparent_surface_clears_palette_and_stack_attribute() -> None:
    """透明 WebEngine 表面必须同时清理 Qt 调色板和层叠属性。"""

    class Color:
        def __init__(self, red: int, green: int, blue: int, alpha: int) -> None:
            self.rgba = (red, green, blue, alpha)

    class Palette:
        class ColorRole:
            Window = "window"
            Base = "base"

        def __init__(self) -> None:
            self.colors: dict[object, object] = {}

        def setColor(self, role: object, color: object) -> None:  # noqa: N802
            self.colors[role] = color

    class View:
        def __init__(self) -> None:
            self.attributes: list[tuple[object, bool]] = []
            self._palette = Palette()
            self.page_value = type("Page", (), {})()
            self.page_value.background = None
            self.quick_children: list[object] = []

            def set_background(color: object) -> None:
                self.page_value.background = color

            self.page_value.setBackgroundColor = set_background
            self.stylesheet = ""
            self.auto_fill = None

        def setAttribute(self, attribute: object, enabled: bool) -> None:  # noqa: N802
            self.attributes.append((attribute, enabled))

        def setAutoFillBackground(self, enabled: bool) -> None:  # noqa: N802
            self.auto_fill = enabled

        def setStyleSheet(self, stylesheet: str) -> None:  # noqa: N802
            self.stylesheet = stylesheet

        def palette(self) -> Palette:
            return self._palette

        def setPalette(self, palette: Palette) -> None:  # noqa: N802
            self._palette = palette

        def page(self):
            return self.page_value

        def findChildren(self, _klass):  # noqa: N802
            return tuple(self.quick_children)

    class Qt:
        WA_TranslucentBackground = "translucent"
        WA_NoSystemBackground = "no-system"
        WA_AlwaysStackOnTop = "stack"
        WA_OpaquePaintEvent = "opaque"

    Core = type("Core", (), {"Qt": Qt})
    Gui = type("Gui", (), {"QColor": Color, "QPalette": Palette})

    view = View()

    class Quick:
        def __init__(self) -> None:
            self.attributes: list[tuple[object, bool]] = []
            self._palette = Palette()
            self.auto_fill = None
            self.stylesheet = ""
            self.clear_color = None

        def setClearColor(self, color) -> None:  # noqa: N802
            self.clear_color = color

        def setAttribute(self, attribute, enabled: bool) -> None:  # noqa: N802
            self.attributes.append((attribute, enabled))

        def setAutoFillBackground(self, enabled: bool) -> None:  # noqa: N802
            self.auto_fill = enabled

        def setStyleSheet(self, stylesheet: str) -> None:  # noqa: N802
            self.stylesheet = stylesheet

        def palette(self) -> Palette:
            return self._palette

        def setPalette(self, palette: Palette) -> None:  # noqa: N802
            self._palette = palette

    quick = Quick()
    view.quick_children.append(quick)
    WebLive2DRenderer._configure_transparent_view(view, Core, Gui)

    assert {attribute for attribute, enabled in view.attributes if enabled} == {
        "translucent",
        "no-system",
        "stack",
    }
    assert ("opaque", False) in view.attributes
    assert view.auto_fill is False
    assert view.stylesheet == "background: transparent;"
    assert view._palette.colors["window"].rgba == (0, 0, 0, 0)
    assert view._palette.colors["base"].rgba == (0, 0, 0, 0)
    assert view.page_value.background.rgba == (0, 0, 0, 0)
    assert {attribute for attribute, enabled in quick.attributes if enabled} == {
        "translucent",
        "no-system",
        "stack",
    }
    assert ("opaque", False) in quick.attributes
    assert quick.clear_color.rgba == (0, 0, 0, 0)
    assert quick.auto_fill is False
    assert quick.stylesheet == "background: transparent;"
    assert quick._palette.colors["window"].rgba == (0, 0, 0, 0)
    assert quick._palette.colors["base"].rgba == (0, 0, 0, 0)


def test_web_live2d_html_has_bounded_model_load_failure_path() -> None:
    """WebGL 初始化卡住时页面必须向 Qt 桥报告可回退错误。"""

    renderer = WebLive2DRenderer("resources")
    html = renderer._html_document()

    assert "modelLoadTimeoutMs = 10000" in html
    assert "Live2D model load timed out" in html
    assert "Promise.race([observedModelPromise, timeoutPromise])" in html


def test_web_live2d_advance_clamps_invalid_frame_intervals() -> None:
    renderer = WebLive2DRenderer("resources")
    calls: list[str] = []
    renderer._model_ready = True
    renderer._view = object()
    renderer._run_javascript = calls.append  # type: ignore[method-assign]

    renderer.advance(float("nan"))
    renderer.advance(float("inf"))
    renderer.advance(-1.0)
    renderer.advance(1.0)

    assert renderer.state.elapsed == 0.05
    assert len(calls) == 4
    assert all("NaN" not in script and "Infinity" not in script for script in calls)


def test_web_live2d_replays_cdi_parameters_after_internal_update() -> None:
    """动作每帧重放时只允许使用当前 model3 DisplayInfo 声明的参数 ID。"""

    probe = probe_web_live2d("resources")
    assert probe.model_path is not None
    descriptor = json.loads(probe.model_path.read_text(encoding="utf-8"))
    cdi_path = probe.model_path.parent / descriptor["FileReferences"]["DisplayInfo"]
    cdi = json.loads(cdi_path.read_text(encoding="utf-8"))
    parameter_ids = {item["Id"] for item in cdi["Parameters"]}
    animated_ids = {
        "ParamBreath",
        "ParamHairFront",
        "ParamHairSide",
        "ParamBodyAngleX",
        "ParamBodyAngleY",
        "ParamBodyAngleZ",
        "ParamEyeLOpen",
        "ParamEyeROpen",
        "ParamEyeBallX",
        "ParamEyeBallY",
        "ParamAngleX",
        "ParamAngleY",
    }

    assert animated_ids <= parameter_ids
    html = WebLive2DRenderer("resources")._html_document()
    assert "applyProceduralExpression(expressionState.name)" in html
    assert "applyProceduralParameterMotion(seconds, advancePose = true)" in html
    assert "applyFormalParameterMotion(dt)" in html
    assert "frameValid = applyFormalParameterMotion(dt)" in html
    assert "frameValid = applyFormalParameterMotion(seconds)" not in html
    assert "detectParameterBindings()" in html
    assert "parameterBindingState" in html
    assert "parameterSafetyState" in html
    assert "poseParameterIds" in html
    assert "safeSpan" in html
    assert "parameterBindings" in html
    assert "getDrawableVertexPositions" in html
    assert all(parameter_id in html for parameter_id in animated_ids)
    assert "ParamEyeBallX: 0" in html
    assert "ParamEyeBallY: 0" in html
    assert "expressionAngles" in html
    assert 'setParameterValue("ParamAngleX", angleX * facing)' in html
    assert 'setParameterValue("ParamAngleY", angleY)' in html
    assert 'setParameterValue("ParamBodyAngleZ", 0)' in html
    assert 'setPartOpacity("Eye_L", openness)' in html
    assert 'setPartOpacity("Eye_R", openness)' in html
    assert 'id="meapet-face-overlay"' in html
    assert "drawFaceOverlay()" in html
    assert "speechActive" in html
    assert "speechElapsed" in html
    assert "setSpeech: function(text, mood, visible, speaking)" in html
    assert "setInteractionFeedback: function(value, visible)" in html
    assert 'id="meapet-interaction-toast"' in html


def test_web_live2d_cursor_target_rejects_invalid_coordinates() -> None:
    renderer = WebLive2DRenderer("resources")
    renderer._view = object()
    assert renderer.set_cursor_target(float("nan"), 0, 100, 100) is False
    assert renderer.set_cursor_target(999, 999, 100, 100) is True
    assert renderer._last_cursor_target == (99, 99, 100, 100)


def test_web_live2d_repeated_speech_snapshot_does_not_reset_active_mouth() -> None:
    renderer = WebLive2DRenderer("resources")
    renderer._view = object()
    if renderer._probe.motion_sync is None:
        return
    assert renderer.set_speech("同一句", speaking=True)
    renderer._pending_mouth_viseme = ("A", 1.0)
    assert renderer.set_speech("同一句", speaking=True)
    assert renderer._pending_mouth_viseme == ("A", 1.0)
    assert renderer.set_speech("同一句", speaking=False)
    assert renderer._pending_mouth_viseme == ("Silence", 0.0)


def test_web_live2d_bubble_never_blocks_model_pointer_input() -> None:
    """气泡是展示层，不能覆盖猫猫头部而吞掉 pointer 事件。"""

    renderer = WebLive2DRenderer("resources")
    html = renderer._html_document()
    assert "pointer-events:none" in html
    assert "canvas{display:block" in html


def test_static_live2d_debug_page_uses_current_model_descriptor() -> None:
    """资源目录里的独立调试页不能继续引用已移出的 Mare 描述。"""

    index_path = Path(__file__).resolve().parents[1] / "resources" / "live2d" / "index.html"
    source = index_path.read_text(encoding="utf-8")
    assert "model/mea_live2d/橙色猫猫.model3.json" in source
    assert "Mare.model3.json" not in source
    assert "model/mea_live2d/Mare" not in source
    assert 'id="mare-canvas"' not in source
    assert 'id="meapet-canvas"' in source
    assert "backgroundAlpha: 0" in source
    assert "autoUpdate: false" in source
    assert "window.requestAnimationFrame(animationTick)" in source
    assert "window.meapetLive2D" in source


def test_web_live2d_hit_feedback_is_part_aware_and_non_blocking() -> None:
    """逐部位反馈必须来自几何命中，并且展示层不能拦截后续指针。"""

    renderer = WebLive2DRenderer("resources")
    html = renderer._html_document()
    assert 'id="meapet-hit-feedback"' in html
    assert "pointer-events:none" in html
    assert "drawableGeometryDetails" in html
    assert "parentPartIndices" in html
    assert "feedback_part" in html
    assert "partNames: modelPartNames" in html
    part_names = renderer._model_part_names()
    assert part_names
    assert all(name in html for name in part_names[:8])
    part_labels = renderer._model_part_labels()
    assert part_labels
    assert "modelPartLabels" in html
    assert any(label in {"脸", "右眼", "左眼", "嘴", "眉毛"} for label in part_labels)
    # 当前模型的脸部覆盖层必须优先使用 DisplayInfo 语义部件的动态 bounds；
    # 没有语义部件的模型仍由前端的全身比例回退处理。
    assert "facePartPatterns" in html
    assert "faceFeatureBounds" in html
    assert "featurePoint('mouth'" in html
    # 浏览器的 dblclick 也会在透明画布区域产生；控制台入口只能由真实
    # 模型几何命中触发，不能被透明矩形抢走焦点。
    assert "const hitDetails = modelHitDetails(event, canvas);" in html
    assert "if (!hitDetails.hit)" in html
    # 方向 B 只镜像模型，部位名称仍按屏幕左右返回，保证反馈和用户直觉一致。
    assert "const screenNormalizedX = visibleWidth > 0" in html
    assert "part = screenNormalizedX < 0.5 ? 'lower_left' : 'lower_right';" in html
    assert "feedbackState.expiresAt = performance.now() + 1600" in html
    assert "}, 1640);" in html
    # 单击先显示半透明 pending marker，正式反馈仍与延迟 click 一起提交；
    # 双击时 pending marker 必须可撤销，避免打开控制台时留下假的最终反馈。
    assert "if (generation !== clickGeneration) return;" in html
    assert "showHitFeedback(hitDetails.part, hitDetails.canvasX, hitDetails.canvasY);" in html
    assert "provisionalMarker" in html
    assert "meapet-hit-marker.pending" in html
    assert "showPendingHitFeedback" in html
    assert "猫猫头 · 正在回应" in html
    assert "dragMoveIntervalMs = 16" in html
    assert "if (type === 'release' && pointerMoved) send('move', event);" in html
    assert "clearPendingHitFeedback" in html
    assert "pendingToastGeneration" in html
    assert "}, 900);" in html
    assert "CanvasOriginX" in html
    assert "inputMask: function()" in html
    assert "preserveDrawingBuffer: true" in html


def test_native_live2d_hit_test_uses_precise_runtime_part_api() -> None:
    from gui.renderers.live2d import Live2DRenderer

    renderer = Live2DRenderer("resources")
    calls: list[tuple[float, float, bool]] = []

    class Model:
        def HitPart(self, x, y, _include_hidden):  # noqa: N802
            calls.append((x, y, _include_hidden))
            return ["Body"] if x > 10 else []

    renderer._model = Model()
    assert renderer.hit_test(20, 30) is True
    assert renderer.hit_test(1, 30) is False
    assert calls == [(20.0, 30.0, False), (1.0, 30.0, False)]
    assert renderer.hit_parts(20, 30) == ("Body",)
    assert calls[-1] == (20.0, 30.0, False)

    renderer._model = object()
    assert renderer.hit_test(20, 30) is False


def test_native_live2d_applies_safe_cursor_pose_and_resize() -> None:
    from gui.renderers.live2d import Live2DRenderer

    renderer = Live2DRenderer("resources")
    calls: list[tuple[object, ...]] = []

    class Model:
        def Update(self):  # noqa: N802
            calls.append(("update",))

        def SetParameterValue(self, name, value, weight):  # noqa: N802
            calls.append((name, value, weight))

        def Resize(self, width, height):  # noqa: N802
            calls.append(("resize", width, height))

    renderer._model = Model()
    assert renderer.set_cursor_target(0, 0, 420, 480)
    renderer.resize(420, 480)
    renderer.advance(0.1)
    assert calls[0] == ("resize", 420, 480)
    assert calls[1] == ("update",)
    pose_calls = [item for item in calls if item[0] in {"ParamAngleX", "ParamAngleY"}]
    assert len(pose_calls) == 2
    assert all(abs(float(item[1])) <= 8.0 for item in pose_calls)


def test_native_live2d_tracking_is_damped_and_bounded() -> None:
    from gui.renderers.live2d import Live2DRenderer

    value, velocity = 0.0, 0.0
    samples: list[float] = []
    for _ in range(80):
        value, velocity = Live2DRenderer._smooth_axis(value, 2.0, velocity, 1 / 60)
        samples.append(value)
    assert all(-0.001 <= item <= 2.001 for item in samples)
    assert samples[-1] > 1.9
    assert all(abs(item) <= 12.0 for item in (value, velocity))


def test_native_live2d_direction_waits_for_neutral_crossing() -> None:
    from gui.renderers.live2d import Live2DRenderer

    renderer = Live2DRenderer("resources")
    directions: list[str] = []

    class Model:
        def SetDirection(self, value):  # noqa: N802
            directions.append(value)

        def Update(self):  # noqa: N802
            return None

        def SetParameterValue(self, *_values):  # noqa: N802
            return None

    renderer._model = Model()
    assert renderer.set_direction("B") is True
    assert directions == []
    for _ in range(20):
        renderer.advance(1 / 60)
    assert directions == ["B"]


def test_web_live2d_shutdown_closes_view_before_releasing_page() -> None:
    """关闭 WebEngine 时先停止顶层视图，再安排页面释放，避免 discarded 清理。"""

    class Page:
        def __init__(self, events: list[str]) -> None:
            self.deleted = False
            self.events = events

        def deleteLater(self) -> None:  # noqa: N802
            self.events.append("page")
            self.deleted = True

    class View:
        def __init__(self) -> None:
            self.events: list[str] = []
            self.page_value = Page(self.events)
            self.closed = False
            self.deleted = False

        def page(self) -> Page:
            return self.page_value

        def close(self) -> None:
            self.events.append("close")
            self.closed = True

        def deleteLater(self) -> None:  # noqa: N802
            self.events.append("view")
            self.deleted = True

    renderer = WebLive2DRenderer(".")
    view = View()
    renderer._view = view
    renderer.shutdown()

    assert view.page_value.deleted
    assert view.closed
    assert view.deleted
    assert view.events == ["close", "page", "view"]
    assert renderer.view is None


def test_web_pet_host_shutdown_owns_renderer_lifecycle_once() -> None:
    """宿主重复收到退出通知时不能再次关闭已废弃的 WebEngine 页面。"""

    class Renderer:
        view = object()

        def __init__(self) -> None:
            self.shutdown_count = 0
            self.interaction_callback = object()
            self.click_callback = object()

        def set_interaction_callback(self, callback):
            self.interaction_callback = callback

        def set_click_callback(self, callback):
            self.click_callback = callback

        def shutdown(self):
            self.shutdown_count += 1

    renderer = Renderer()
    host = WebPetHost(renderer)
    host.shutdown()
    host.shutdown()

    assert renderer.shutdown_count == 1
    assert renderer.interaction_callback is None
    assert renderer.click_callback is None


def test_web_live2d_failure_callback_is_called_once(tmp_path: Path) -> None:
    messages: list[str] = []
    renderer = WebLive2DRenderer(
        tmp_path,
        failure_callback=messages.append,
    )
    renderer._view = object()
    renderer._model_ready = True
    renderer._set_unavailable("model load failed")
    assert renderer._model_ready is False
    renderer._set_unavailable("model load failed again")
    assert messages == ["model load failed"]


def test_web_live2d_click_through_verifies_qt_flag_result(tmp_path: Path) -> None:
    flag = 4

    window_type = type("WindowType", (), {"WindowTransparentForInput": flag})
    core = type("Core", (), {"Qt": type("Qt", (), {"WindowType": window_type})})

    class View:
        def __init__(self, *, retain: bool = True) -> None:
            self.flags = 0
            self.retain = retain

        def setWindowFlag(self, requested: int, enabled: bool) -> None:  # noqa: N802
            if self.retain:
                self.flags = self.flags | requested if enabled else self.flags & ~requested

        def windowFlags(self) -> int:  # noqa: N802
            return self.flags

    renderer = WebLive2DRenderer(tmp_path, importer=lambda _name: core)
    renderer._view = View()
    assert renderer.set_click_through(True) == {"status": "available", "enabled": True}
    renderer._view = View(retain=False)
    result = renderer.set_click_through(True)
    assert result["status"] == "degraded"
    assert result["enabled"] is False


def test_native_live2d_import_failure_degrades_to_fallback(tmp_path: Path, monkeypatch) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    _write_model3_descriptor(model_root)

    def broken_import(_name: str):
        raise RuntimeError("native loader failure")

    monkeypatch.setattr("gui.renderers.live2d.importlib.import_module", broken_import)
    renderer = Live2DRenderer(model_root)
    assert not renderer.capabilities.available
    assert "unavailable" in renderer.capabilities.message


def test_web_pet_host_keeps_platform_position_boundary() -> None:
    class FakeRenderer:
        view = object()

        def set_expression(self, name):
            return name == "happy"

        def play_motion(self, name):
            return name == "wave"

        def set_speech(self, text, *, mood="neutral", visible=True):
            return {"text": text, "mood": mood, "visible": visible}

    class FakeCapability:
        state = CapabilityState.AVAILABLE
        detail = "ok"

    class FakePlatform:
        def move_overlay(self, _view, x, y):
            assert (x, y) == (12, 34)
            return FakeCapability()

    host = WebPetHost(FakeRenderer(), platform=FakePlatform())
    assert host.move_to(12, 34)["status"] == "available"
    assert host.set_expression("happy")
    assert host.play_motion("wave")
    assert host.set_speech("你好")["text"] == "你好"


def test_web_pet_host_display_size_presets_preserve_position() -> None:
    from PySide6.QtCore import QPoint

    class View:
        def __init__(self) -> None:
            self._position = QPoint(40, 50)
            self._size = (420, 480)

        def pos(self):
            return self._position

        def move(self, point):
            self._position = point

        def resize(self, width, height):
            self._size = (int(width), int(height))

    class Renderer:
        def __init__(self) -> None:
            self.view = View()

    host = WebPetHost(Renderer())
    result = host.set_display_size("large")
    assert result == {"status": "available", "preset": "large", "width": 520, "height": 620}
    assert host.view._size == (520, 620)
    assert host.view.pos() == QPoint(40, 50)


def test_web_pet_host_display_size_calls_synchronous_renderer_resize() -> None:
    """尺寸预设优先调用页面同步重排入口，避免连续点击滞后一档。"""

    from PySide6.QtCore import QPoint

    class Page:
        def __init__(self) -> None:
            self.scripts: list[str] = []

        def runJavaScript(self, script: str) -> None:  # noqa: N802
            self.scripts.append(script)

    class View:
        def __init__(self) -> None:
            self._position = QPoint(40, 50)
            self._size = (420, 480)
            self._page = Page()

        def pos(self):
            return self._position

        def move(self, point):
            self._position = point

        def resize(self, width, height):
            self._size = (int(width), int(height))

        def page(self):
            return self._page

    class Renderer:
        def __init__(self) -> None:
            self.view = View()
            self.page_ready = True

    renderer = Renderer()
    host = WebPetHost(renderer)
    result = host.set_display_size("large")
    assert result["status"] == "available"
    assert renderer.view._page.scripts
    assert "window.meapetLive2D.resize()" in renderer.view._page.scripts[-1]
    assert "window.meapetLive2D.resize(520,620)" in renderer.view._page.scripts[-1]


def test_web_pet_host_display_size_uses_actual_low_resolution_viewport(monkeypatch) -> None:
    """低分辨率下预设尺寸必须裁剪，并按实际外框同步页面。"""

    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    class Screen:
        def availableGeometry(self):
            return QRect(0, 0, 320, 240)

    class Page:
        def __init__(self) -> None:
            self.scripts: list[str] = []

        def runJavaScript(self, script: str) -> None:  # noqa: N802
            self.scripts.append(script)

    class View:
        def __init__(self) -> None:
            self._position = QPoint(20, 20)
            self._size = (420, 480)
            self._minimum = (420, 480)
            self._page = Page()

        def pos(self):
            return self._position

        def move(self, point):
            self._position = QPoint(point)

        def resize(self, width, height):
            self._size = (int(width), int(height))

        def width(self):
            return self._size[0]

        def height(self):
            return self._size[1]

        def minimumWidth(self):
            return self._minimum[0]

        def minimumHeight(self):
            return self._minimum[1]

        def setMinimumWidth(self, value):
            self._minimum = (int(value), self._minimum[1])

        def setMinimumHeight(self, value):
            self._minimum = (self._minimum[0], int(value))

        def page(self):
            return self._page

    class Renderer:
        def __init__(self) -> None:
            self.view = View()
            self.page_ready = True

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QGuiApplication, "primaryScreen", staticmethod(lambda: Screen()))
    renderer = Renderer()
    host = WebPetHost(renderer)
    result = host.set_display_size("large")
    assert result == {"status": "available", "preset": "large", "width": 320, "height": 240}
    assert renderer.view._size == (320, 240)
    assert renderer.view._minimum == (320, 240)
    assert "window.meapetLive2D.resize(320,240)" in renderer.view._page.scripts[-1]
    assert renderer.view.pos() == QPoint(20, 20)
    del app


def test_web_pet_host_syncs_external_resize_only_when_size_changes() -> None:
    """窗口管理器直接调整 QWidget 时，Chromium viewport 必须跟随且不重复刷脚本。"""

    class Page:
        def __init__(self) -> None:
            self.scripts: list[str] = []

        def runJavaScript(self, script: str) -> None:  # noqa: N802
            self.scripts.append(script)

    class View:
        def __init__(self) -> None:
            self.size = (420, 480)
            self._page = Page()

        def width(self):
            return self.size[0]

        def height(self):
            return self.size[1]

        def page(self):
            return self._page

    class Renderer:
        def __init__(self) -> None:
            self.view = View()
            self.page_ready = True

    renderer = Renderer()
    host = WebPetHost(renderer)
    host._sync_viewport_size()
    first_count = len(renderer.view._page.scripts)
    assert first_count >= 1
    assert "window.meapetLive2D.resize(420,480)" in renderer.view._page.scripts[-1]
    host._sync_viewport_size()
    assert len(renderer.view._page.scripts) == first_count
    renderer.view.size = (96, 480)
    host._sync_viewport_size()
    assert "window.meapetLive2D.resize(96,480)" in renderer.view._page.scripts[-1]


def test_web_pet_host_retries_initial_viewport_after_page_ready() -> None:
    """页面尚未就绪时读取的尺寸必须在握手后补发一次。"""

    class Page:
        def __init__(self) -> None:
            self.scripts: list[str] = []

        def runJavaScript(self, script: str) -> None:  # noqa: N802
            self.scripts.append(script)

    class View:
        def __init__(self) -> None:
            self._page = Page()
            self._size = (420, 480)

        def width(self) -> int:
            return self._size[0]

        def height(self) -> int:
            return self._size[1]

        def page(self) -> Page:
            return self._page

    renderer = SimpleNamespace(view=View(), page_ready=False)
    host = WebPetHost(renderer)

    host._sync_viewport_size()
    assert renderer.view._page.scripts == []
    assert host._last_viewport_size is None

    renderer.page_ready = True
    host._sync_viewport_size()

    assert len(renderer.view._page.scripts) == 1
    assert "window.meapetLive2D.resize(420,480)" in renderer.view._page.scripts[0]
    assert host._last_viewport_size == (420, 480)


def test_web_pet_host_retries_viewport_after_webengine_surface_rebuild() -> None:
    """原生子表面重建期间脚本提交失败时，下一轮必须继续同步。"""

    class Page:
        def __init__(self) -> None:
            self.calls = 0
            self.scripts: list[str] = []

        def runJavaScript(self, script: str) -> None:  # noqa: N802
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("surface is rebuilding")
            self.scripts.append(script)

    class View:
        def __init__(self) -> None:
            self._page = Page()

        def width(self) -> int:
            return 420

        def height(self) -> int:
            return 480

        def page(self) -> Page:
            return self._page

    renderer = SimpleNamespace(view=View(), page_ready=True)
    host = WebPetHost(renderer)

    host._sync_viewport_size()
    assert renderer.view._page.calls == 1
    assert host._last_viewport_size is None

    host._sync_viewport_size()
    assert renderer.view._page.calls == 2
    assert len(renderer.view._page.scripts) == 1
    assert host._last_viewport_size == (420, 480)


def test_web_pet_host_retries_deferred_viewport_after_surface_rebuild() -> None:
    """拖动释放时提交失败也不能丢失最后一个窗口尺寸。"""

    class Page:
        def __init__(self) -> None:
            self.calls = 0
            self.scripts: list[str] = []

        def runJavaScript(self, script: str) -> None:  # noqa: N802
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("surface is rebuilding")
            self.scripts.append(script)

    class View:
        def __init__(self) -> None:
            self._page = Page()
            self._size = (360, 260)

        def width(self) -> int:
            return self._size[0]

        def height(self) -> int:
            return self._size[1]

        def page(self) -> Page:
            return self._page

    renderer = SimpleNamespace(view=View(), page_ready=True)
    host = WebPetHost(renderer)
    host._renderer_dragging = True

    host._sync_viewport_size()
    assert host._deferred_page_resize == (360, 260)
    assert host._last_viewport_size == (360, 260)

    host._renderer_dragging = False
    host._flush_deferred_page_resize()
    assert host._deferred_page_resize == (360, 260)
    assert host._last_viewport_size is None

    host._sync_viewport_size()
    assert host._deferred_page_resize is None
    assert host._last_viewport_size == (360, 260)
    assert len(renderer.view._page.scripts) == 1
    assert "window.meapetLive2D.resize(360,260)" in renderer.view._page.scripts[0]


def test_web_pet_host_prefers_page_bridge_without_duplicate_qt_filter() -> None:
    class View:
        def installEventFilter(self, _filter):
            raise AssertionError("native filter should not be installed when page bridge is active")

    class Renderer:
        view = View()

        def set_interaction_callback(self, callback):
            self.callback = callback

        def set_click_callback(self, callback):
            self.click_callback = callback

    host = WebPetHost(Renderer())
    assert host.install_interaction()
    assert host._page_bridge_enabled is True
    assert host._interaction_filter is None


def test_web_pet_host_keeps_native_fallback_until_page_bridge_handshake(monkeypatch) -> None:
    """页面未握手时保留原生输入，握手后才移除以避免输入丢失。"""

    import gui.qt6.web_host as web_host_module

    class FakeFilter:
        def __init__(self, *_args):
            self.removed = False
            self._click_token = 0

        def install_on(self, target):
            target.installed = True

        def _clear_drag(self):
            return None

    monkeypatch.setattr(web_host_module, "_WebInteractionFilter", FakeFilter)

    class View:
        installed = False

        def findChildren(self, _type):  # noqa: N802
            return ()

        def removeEventFilter(self, _filter):  # noqa: N802
            self.installed = False

        def installEventFilter(self, _filter):  # noqa: N802
            self.installed = True

    class Renderer:
        def __init__(self):
            self.view = View()
            self.page_bridge_ready = False
            self.ready_callback = None

        def set_interaction_callback(self, callback):
            self.interaction_callback = callback

        def set_page_bridge_ready_callback(self, callback):
            self.ready_callback = callback

        def set_click_callback(self, callback):
            self.click_callback = callback

    renderer = Renderer()
    host = WebPetHost(renderer)
    assert host.install_interaction()
    assert host._page_bridge_enabled is False
    assert host._interaction_filter is not None
    native_filter = host._interaction_filter
    renderer.page_bridge_ready = True
    renderer.ready_callback(True)
    assert host._page_bridge_enabled is True
    assert host._interaction_filter is None
    assert native_filter._click_token == 1
    renderer.page_bridge_ready = False
    renderer.ready_callback(False)
    assert host._page_bridge_enabled is False
    assert host._interaction_filter is not None
    host.shutdown()


def test_web_pet_host_rebinds_interaction_after_topmost_flag_rebuild() -> None:
    """置顶切换重建 QWidget 原生窗口后，拖动过滤器必须重新挂载。"""

    class View:
        def __init__(self) -> None:
            self._flags = 0
            self._visible = True

        def isVisible(self):
            return self._visible

        def setWindowFlag(self, flag, enabled):  # noqa: N802
            self._flags = self._flags | flag if enabled else self._flags & ~flag
            # QWidget 在真实 Qt 后端可能暂时隐藏并创建新的原生窗口；
            # 这里保持可见，专门验证宿主触发重扫的契约。

        def show(self):
            self._visible = True

        def windowFlags(self):  # noqa: N802
            return self._flags

    class Renderer:
        def __init__(self) -> None:
            self.view = View()

    host = WebPetHost(Renderer())
    refreshes: list[str] = []
    scheduled: list[str] = []
    host._refresh_interaction_targets = lambda: refreshes.append("immediate")  # type: ignore[method-assign]
    host._schedule_interaction_refresh = lambda: scheduled.append("deferred")  # type: ignore[method-assign]

    result = host.set_always_on_top(True)

    assert result["status"] == "available"
    assert refreshes == ["immediate"]
    assert scheduled == ["deferred"]


def test_web_pet_host_handles_page_pointer_drag_and_callbacks() -> None:
    """Chromium 页面桥接的指针事件必须驱动窗口移动和控制入口。"""

    from PySide6.QtCore import QPoint

    class View:
        def __init__(self) -> None:
            self._position = QPoint(100, 120)

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = View()

    renderer = Renderer()
    host = WebPetHost(renderer)
    contexts: list[tuple[int, int]] = []
    doubles: list[bool] = []
    host._context_menu_callback = lambda point: contexts.append((point.x(), point.y()))
    host._double_click_callback = lambda: doubles.append(True)

    host._handle_page_interaction({"type": "press", "x": 10, "y": 20, "button": "left"})
    assert host._page_drag_origin == QPoint(10, 20)
    host._page_dragging = True
    host._handle_page_interaction({"type": "move", "x": 50, "y": 70, "button": "other"})
    assert renderer.view.pos() == QPoint(140, 170)
    host._handle_page_interaction({"type": "release", "x": 50, "y": 70, "button": "left"})
    assert host._page_drag_origin is None

    host._handle_page_interaction({"type": "context", "x": 12, "y": 14, "button": "right"})
    assert contexts == [(152, 184)]
    host._handle_page_interaction({"type": "double", "x": 12, "y": 14, "button": "left"})
    assert doubles == [True]


def test_web_pet_host_clears_page_drag_without_release_coordinates() -> None:
    """浏览器取消指针捕获时缺少坐标也不能遗留拖动状态。"""

    from PySide6.QtCore import QPoint

    class View:
        def __init__(self) -> None:
            self._position = QPoint(100, 120)

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = View()

    host = WebPetHost(Renderer())
    doubles: list[bool] = []
    host._double_click_callback = lambda: doubles.append(True)
    host._handle_page_interaction({"type": "press", "x": 10, "y": 20, "button": "left"})
    assert host._page_drag_origin == QPoint(10, 20)
    host._handle_page_interaction({"type": "release", "button": "left"})
    assert host._page_drag_origin is None
    assert host._page_window_origin is None
    host._handle_page_interaction({"type": "press", "x": 10, "y": 20, "button": "left"})
    host._handle_page_interaction({"type": "cancel"})
    assert host._page_drag_origin is None
    host._handle_page_interaction({"type": "double", "button": "left"})
    assert doubles == [True]


def test_web_pet_host_oversized_window_keeps_non_reversed_bounds() -> None:
    """窗口大于屏幕时，位置契约至少保留一个可放置的左上角。"""

    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    app = QApplication.instance() or QApplication([])
    screen = app.primaryScreen()
    assert screen is not None
    area = screen.availableGeometry()

    from PySide6.QtCore import QPoint

    class View:
        def __init__(self) -> None:
            self._position = QPoint(area.left(), area.top())

        def pos(self):
            return self._position

        def width(self):
            return area.width() + 100

        def height(self):
            return area.height() + 100

    class Renderer:
        def __init__(self) -> None:
            self.view = View()

    bounds = WebPetHost(Renderer()).movement_bounds()
    assert bounds is not None
    assert bounds[2] >= bounds[0]
    assert bounds[3] >= bounds[1]
    assert bounds[2] == bounds[0]
    assert bounds[3] == bounds[1]


def test_pet_click_zones_match_legacy_canvas_partition() -> None:
    """点击分区沿用旧项目规则，并拒绝越界/非有限坐标。"""

    assert classify_pet_click(10, 10, 100, 200) == "upper"
    assert classify_pet_click(10, 150, 100, 200) == "lower_left"
    assert classify_pet_click(90, 150, 100, 200) == "lower_right"
    assert classify_pet_click(100, 150, 100, 200) is None
    assert classify_pet_click(float("nan"), 10, 100, 200) is None


def test_web_pet_host_routes_page_click_to_safe_zone_callback() -> None:
    """页面普通点击应进入分区回调，同时保留悬停交互状态。"""

    from PySide6.QtCore import QPoint

    class View:
        def __init__(self) -> None:
            self._position = QPoint(100, 120)

        def pos(self):
            return self._position

        def width(self):
            return 200

        def height(self):
            return 100

    class Renderer:
        def __init__(self) -> None:
            self.view = View()

    renderer = Renderer()
    host = WebPetHost(renderer)
    clicks: list[dict[str, object]] = []
    host._click_callback = clicks.append
    host._handle_page_interaction({"type": "enter", "x": 20, "y": 20})
    assert host.is_user_interacting()
    host._handle_page_interaction({"type": "click", "x": 20, "y": 20, "button": "left"})
    host._handle_page_interaction({"type": "leave", "x": 20, "y": 20})
    assert not host.is_user_interacting()
    assert clicks == [{"type": "click", "zone": "upper", "x": 20, "y": 20, "button": "left"}]


def test_web_pet_host_projects_page_hover_coordinates_to_cursor_target() -> None:
    """页面悬停坐标应即时投影到注视目标，且不依赖对话模型。"""

    from PySide6.QtCore import QPoint

    class View:
        def pos(self):
            return QPoint(100, 120)

        def width(self):
            return 200

        def height(self):
            return 100

    class Renderer:
        def __init__(self) -> None:
            self.view = View()
            self.targets = []

        def set_cursor_target(self, x, y, width, height):
            self.targets.append((x, y, width, height))
            return True

    renderer = Renderer()
    host = WebPetHost(renderer)
    host._handle_page_interaction({"type": "hover", "hit": True, "screen_x": 160, "screen_y": 150})
    assert renderer.targets == [(60, 30, 200, 100)]
    assert host.is_user_interacting()


def test_web_pet_host_rejects_explicit_geometry_miss_before_business_feedback() -> None:
    """页面明确报告未命中时，透明画布不能触发猫猫头反馈。"""

    from PySide6.QtCore import QPoint

    class View:
        def pos(self):
            return QPoint(0, 0)

        def width(self):
            return 200

        def height(self):
            return 100

    class Renderer:
        view = View()

    host = WebPetHost(Renderer())
    clicks: list[dict[str, object]] = []
    host._click_callback = clicks.append
    host._handle_page_interaction(
        {"type": "click", "x": 20, "y": 20, "button": "left", "hit": False}
    )
    assert clicks == []


def test_web_live2d_pointer_payload_is_validated_and_forwarded(tmp_path: Path) -> None:
    renderer = WebLive2DRenderer(tmp_path)
    received: list[dict[str, object]] = []
    renderer.set_interaction_callback(received.append)
    renderer._on_web_pointer_event(json.dumps({"type": "press", "x": 2, "y": 3, "button": "left"}))
    renderer._on_web_pointer_event("not-json")
    renderer._on_web_pointer_event(json.dumps(["not", "an", "event"]))
    assert received == [{"type": "press", "x": 2, "y": 3, "button": "left"}]


def test_web_live2d_click_callback_is_optional_and_only_receives_click_payload() -> None:
    renderer = WebLive2DRenderer(".")
    received: list[dict[str, object]] = []
    renderer.set_click_callback(received.append)
    renderer._on_web_pointer_event(json.dumps({"type": "press", "x": 2, "y": 3, "button": "left"}))
    renderer._on_web_pointer_event(json.dumps({"type": "click", "x": 2, "y": 3, "button": "left"}))
    assert received == [{"type": "click", "x": 2, "y": 3, "button": "left"}]
    renderer.set_click_callback(None)


def test_web_live2d_click_and_pointer_callbacks_do_not_duplicate_host_click() -> None:
    """同一页面 click 同时经过指针桥和独立回调时，宿主各收一次。"""

    renderer = WebLive2DRenderer(".")
    pointer_events: list[dict[str, object]] = []
    click_events: list[dict[str, object]] = []
    renderer.set_interaction_callback(pointer_events.append)
    renderer.set_click_callback(click_events.append)
    payload = {"type": "click", "x": 8, "y": 9, "button": "left"}
    renderer._on_web_pointer_event(json.dumps(payload))
    assert pointer_events == [payload]
    assert click_events == [payload]
    renderer.set_interaction_callback(None)
    renderer.set_click_callback(None)


def test_web_pet_host_awaits_dispatched_platform_calls() -> None:
    class FakeRenderer:
        view = object()

    class FakeCapability:
        state = CapabilityState.AVAILABLE
        detail = "ok"

    class AsyncPlatform:
        def move_overlay(self, _view, _x, _y):
            async def resolve():
                return FakeCapability()

            return resolve()

        def set_click_through(self, _view, _enabled):
            async def resolve():
                return FakeCapability()

            return resolve()

    host = WebPetHost(FakeRenderer(), platform=AsyncPlatform())
    assert asyncio.run(host.move_to(12, 34))["status"] == "available"
    assert asyncio.run(host.set_click_through(True))["enabled"] is True


def test_web_pet_host_drops_stale_async_autonomous_move_before_platform_side_effect() -> None:
    """拖动在异步自主步首次执行前开始时，不应提交过期坐标。"""

    class FakeRenderer:
        view = object()

    class AsyncPlatform:
        def __init__(self) -> None:
            self.started = False

        def move_overlay(self, _view, _x, _y):
            async def resolve():
                self.started = True
                return {"status": "available"}

            return resolve()

    platform = AsyncPlatform()
    host = WebPetHost(FakeRenderer(), platform=platform)

    async def scenario() -> object:
        operation = host.move_to(80, 90, duration_ms=0)
        assert inspect.isawaitable(operation)
        # 模拟用户拖动开始：使旧几何代次失效后才让后台行为协程运行。
        host._invalidate_geometry()
        return await operation

    result = asyncio.run(scenario())
    assert result["status"] == "cancelled"
    assert platform.started is False


def test_web_pet_host_clears_hover_ids_when_replacing_native_targets() -> None:
    """原生子表面重建没有 Leave 事件时，不应残留悬停状态。"""

    class FakeRenderer:
        view = object()

    class InteractionFilter:
        def __init__(self) -> None:
            self._hover_targets = {101, 202}

        def _clear_drag(self) -> None:
            return None

    host = WebPetHost(FakeRenderer())
    interaction_filter = InteractionFilter()
    host._interaction_filter = interaction_filter
    host._interaction_targets = []
    host._remove_native_interaction_filter()
    assert interaction_filter._hover_targets == set()
    assert host._interaction_filter is None


def test_web_pet_host_retries_target_refresh_after_drag_finishes(monkeypatch) -> None:
    """重建期间若仍在拖动，过滤器刷新应延迟到 release 后完成。"""

    import gui.qt6.web_host as web_host_module

    class FakeRenderer:
        view = object()

    class InteractionFilter:
        pass

    callbacks: list[object] = []

    class Timer:
        @staticmethod
        def singleShot(_delay: int, callback) -> None:  # noqa: N802
            callbacks.append(callback)

    host = WebPetHost(FakeRenderer())
    host._interaction_filter = InteractionFilter()
    host._drag_transaction_active = lambda: True  # type: ignore[method-assign]
    refreshed: list[bool] = []
    host._refresh_interaction_targets = lambda: refreshed.append(True)  # type: ignore[method-assign]
    monkeypatch.setattr(web_host_module, "QTimer", Timer)
    host._schedule_interaction_refresh()
    assert len(callbacks) == 1
    # 拖动开始会推进 geometry_generation；旧回调仍需安排一次带新代次的
    # refresh，而不是因代次不匹配直接丢失过滤器重挂载。
    host._invalidate_geometry()
    callbacks.pop()()
    assert len(callbacks) == 1
    host._drag_transaction_active = lambda: False  # type: ignore[method-assign]
    callbacks.pop()()
    assert len(callbacks) == 1
    callbacks.pop()()
    assert refreshed == [True]


def test_web_live2d_move_uses_qwidget_move_fallback(tmp_path: Path) -> None:
    class FakeView:
        def __init__(self):
            self.position = None

        def move(self, x, y):
            self.position = (x, y)

    renderer = WebLive2DRenderer(tmp_path)
    view = FakeView()
    renderer._view = view
    result = renderer.move_to(12.4, 33.6)
    assert result == {"status": "available", "x": 12, "y": 34}
    assert view.position == (12, 34)


def test_linux_overlay_move_prefers_qwidget_move() -> None:
    class FakeWidget:
        def __init__(self):
            self.position = None

        def move(self, x, y):
            self.position = (x, y)

    widget = FakeWidget()
    platform = linux_platform.LinuxDesktopPlatform({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":99"})
    result = platform.move_overlay(widget, 12, 34)
    assert result.state is CapabilityState.AVAILABLE
    assert widget.position == (12, 34)


def test_linux_overlay_move_rejects_implicit_coordinate_conversion() -> None:
    """Linux 窗口位置不能把布尔值或字符串转换成像素。"""

    platform = linux_platform.LinuxDesktopPlatform({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":99"})
    for x, y in ((True, 34), (12, "34"), (1.5, 2)):
        result = platform.move_overlay(object(), x, y)
        assert result.state is CapabilityState.UNAVAILABLE
        assert result.detail == "Linux position coordinates are invalid"


def test_linux_overlay_move_rejects_x11_position_that_was_not_retained() -> None:
    """X11 窗口管理器未采纳位置时不能返回 available。"""

    class Point:
        def x(self):
            return 1

        def y(self):
            return 2

    class FakeWindow:
        def move(self, _x, _y):
            return None

        def pos(self):
            return Point()

    platform = linux_platform.LinuxDesktopPlatform({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":99"})
    result = platform.move_overlay(FakeWindow(), 12, 34)
    assert result.state is CapabilityState.UNAVAILABLE
    assert "did not retain" in result.detail


def test_linux_overlay_move_failure_does_not_echo_exception_detail() -> None:
    """X11 定位异常回执不泄露底层异常正文。"""

    class FakeWindow:
        def move(self, _x, _y):
            raise RuntimeError("secret-position-state")

    platform = linux_platform.LinuxDesktopPlatform({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":99"})
    result = platform.move_overlay(FakeWindow(), 12, 34)
    assert result.state is CapabilityState.UNAVAILABLE
    assert result.detail == "X11 position request failed"
    assert "secret-position-state" not in result.detail


def test_wayland_overlay_move_reports_best_effort_without_claiming_success() -> None:
    class FakeWindow:
        def __init__(self) -> None:
            self._position = (0, 0)

        def setPosition(self, x, y):  # noqa: N802
            # 模拟 compositor 忽略客户端定位请求。
            del x, y

        def position(self):
            class Point:
                def x(_self):
                    return self._position[0]

                def y(_self):
                    return self._position[1]

            return Point()

    window = FakeWindow()
    platform = linux_platform.LinuxDesktopPlatform(
        {
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-0",
            "DISPLAY": ":0",
        }
    )
    result = platform.move_overlay(window, 120, 340)
    assert result.state is CapabilityState.DEGRADED
    assert "did not reflect" in result.detail
    assert "compositor controls final placement" in result.detail


def test_wayland_overlay_move_failure_does_not_echo_exception_detail() -> None:
    """Wayland 定位异常回执不泄露底层异常正文。"""

    class FakeWindow:
        def setPosition(self, _x, _y):  # noqa: N802
            raise RuntimeError("secret-wayland-position")

    platform = linux_platform.LinuxDesktopPlatform(
        {"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"}
    )
    result = platform.move_overlay(FakeWindow(), 12, 34)
    assert result.state is CapabilityState.UNAVAILABLE
    assert result.detail == "Wayland position request failed"
    assert "secret-wayland-position" not in result.detail


def test_live2d_frame_audit_waits_for_fresh_composited_frame() -> None:
    """审计脚本必须等待合成帧并重试陈旧截图，不能把时序竞态当成动作重复。"""

    audit_path = Path(__file__).resolve().parents[1] / "scripts" / "live2d_frame_audit.py"
    source = audit_path.read_text(encoding="utf-8")
    assert "def save_capture()" in source
    assert "QTimer.singleShot(80, save_capture)" in source
    assert "previous_hashes" in source
    assert 'metrics.get("sha256") in previous_hashes' in source
    assert "finish() 的重复哈希检查" in source
