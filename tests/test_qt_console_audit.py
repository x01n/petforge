from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from gui.qt6.app import _public_logging_handler_status
from gui.qt6.console import _module_public_state


def _run_qt_script(tmp_path: Path, script: str) -> subprocess.CompletedProcess[str]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def test_module_pending_snapshot_is_not_reported_as_unavailable() -> None:
    assert _module_public_state({"pending": True, "available": False}) == "checking"
    assert (
        _module_public_state({"health": {"status": "loading", "pending": True, "available": False}})
        == "checking"
    )


def test_logging_module_status_distinguishes_configured_and_active_file_sink() -> None:
    """配置文件输出但 handler 创建失败时不得误报 ready。"""

    test_logger = logging.Logger("meapet-module-status-test", logging.INFO)
    console_handler = logging.StreamHandler()
    setattr(console_handler, "_meapet_logging_handler", "console")
    test_logger.addHandler(console_handler)
    values = {
        "level": "INFO",
        "console": {"enabled": True},
        "file": {
            "enabled": True,
            "rotation": "time",
            "path": "/private/never-expose.log",
        },
    }
    degraded = _public_logging_handler_status(values, test_logger)
    assert degraded == {
        "status": "degraded",
        "available": False,
        "ready": False,
        "level": "INFO",
        "handler_count": 1,
        "console_active": True,
        "file_configured": True,
        "file_active": False,
        "rotation_enabled": False,
    }
    assert "/private" not in repr(degraded)

    file_handler = logging.StreamHandler()
    setattr(file_handler, "_meapet_logging_handler", "file")
    test_logger.addHandler(file_handler)
    ready = _public_logging_handler_status(values, test_logger)
    assert ready["status"] == "ready"
    assert ready["file_active"] is True
    assert ready["rotation_enabled"] is True


def test_model_setup_requires_complete_fields_before_submit(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog

app = QApplication.instance() or QApplication([])
dialog = ModelSetupDialog()
dialog.show()
app.processEvents()
assert not dialog.save_button.isEnabled()
assert "不能为空" in dialog.card.status_label.text()
dialog.card.preset_combo.setCurrentIndex(dialog.card.preset_combo.findData("openai"))
app.processEvents()
assert dialog.save_button.isEnabled()
assert dialog.card.is_valid
dialog.close()
app.processEvents()
print("model-setup-validation-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "model-setup-validation-ok" in result.stdout


def test_model_setup_connection_test_is_async_and_returns_safe_status(tmp_path: Path) -> None:
    script = r"""
from concurrent.futures import Future
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
future = Future()
window = PetConsoleWindow(
    callbacks={"test_model": lambda payload: future},
    configuration={"llm": {"channels": []}},
)
assert window.show_model_settings()
dialog = window._model_setup_dialog
assert dialog is not None
dialog.card.preset_combo.setCurrentIndex(dialog.card.preset_combo.findData("openai"))
app.processEvents()
assert dialog.card.test_button.objectName() == "modelSetupTestButton"
dialog.card.test_button.click()
assert "正在测试" in dialog.card.status_label.text()
future.set_result({"status": "available", "message": "模型连接测试通过"})
QTest.qWait(120)
app.processEvents()
assert dialog.card.status_label.text() == "模型连接测试通过"
window.shutdown()
app.processEvents()
print("model-connection-async-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "model-connection-async-ok" in result.stdout


def test_configuration_panel_hides_raw_editors_until_explicit_switch(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({
    "llm": {"channels": [{"id": "primary", "api_key": "secret-value", "model": "demo"}]},
})
panel.show()
app.processEvents()
panel.navigation.setCurrentRow(panel.sections.index("llm"))
app.processEvents()
complex_editor = panel.complex_editors["llm"][("channels",)]
assert panel.advanced_editing is False
assert panel.model_setup_button is not None
assert complex_editor.isVisible() is False
assert panel.yaml_editor.isHidden()
panel.advanced_toggle.click()
app.processEvents()
assert panel.advanced_editing is True
assert complex_editor.isVisible() is True
assert not panel.yaml_editor.isHidden()
complex_editor.setPlainText("- api_key: secret-value\n  : broken\n")
assert "secret-value" not in panel.status_label.text()
panel.close()
app.processEvents()
print("config-advanced-switch-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "config-advanced-switch-ok" in result.stdout


def test_configuration_panel_defaults_to_clickable_cards_without_raw_fields(
    tmp_path: Path,
) -> None:
    """普通配置页不能让用户先看到 YAML、下拉框或数值微调框。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox, QTabWidget
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({
    "llm": {"channels": []},
    "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
    "logging": {"level": "INFO"},
})
panel.show()
app.processEvents()
assert panel.advanced_editing is False
assert panel._section_editors["overview"].isVisible() is False
assert all(
    not control.isVisible()
    for controls in panel.form_controls.values()
    for control in controls.values()
)
panel.navigation.setCurrentRow(panel.sections.index("tts"))
app.processEvents()
assert panel._friendly_controls["tts"][("enabled",)].isVisible()
assert all(
    not control.isVisible()
    for controls in panel.form_controls.values()
    for control in controls.values()
    if isinstance(control, (QComboBox, QSpinBox, QDoubleSpinBox))
)
toggle = panel._friendly_controls["tts"][("enabled",)]
toggle.click()
assert panel.draft_values()["tts"]["enabled"] is True
panel.navigation.setCurrentRow(panel.sections.index("ui"))
app.processEvents()
lock_toggle = panel._friendly_controls["ui"][("window_locked",)]
assert lock_toggle.isVisible()
lock_toggle.click()
assert panel.draft_values()["ui"]["window_locked"] is True
panel.navigation.setCurrentRow(panel.sections.index("tts"))
app.processEvents()
panel.advanced_toggle.click()
app.processEvents()
assert panel.advanced_editing is True
assert panel.form_controls["tts"][("backend",)].isVisible()
assert panel._section_editors["tts"].isVisible() is False
panel.navigation.setCurrentRow(panel.sections.index("advanced"))
app.processEvents()
assert panel.yaml_editor.isVisible()
panel.advanced_toggle.click()
assert panel.advanced_editing is False
panel.close()
app.processEvents()
print("config-click-surface-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "config-click-surface-ok" in result.stdout


def test_console_first_screen_exposes_rescue_actions_and_pet_feedback(tmp_path: Path) -> None:
    """首屏救援入口和猫猫头互动反馈必须不依赖滚动或原始参数。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QFrame, QPushButton
from gui.qt6.console import PetConsoleWindow, _action_label, _public_action_label

app = QApplication.instance() or QApplication([])
assert _action_label("expression", "custom_face") == "custom_face"
assert _action_label("motion", "IdleCustom") == "IdleCustom"
assert _public_action_label("expression", "custom_face") == "自定义表情"
assert _public_action_label("motion", "IdleCustom") == "自定义动作"
assert _public_action_label("motion", "A") == "动作 A"
calls = []
window = PetConsoleWindow(callbacks={
    "show_pet": lambda: calls.append("show") or {"status": "available", "visible": True},
    "restore_click_through": lambda: calls.append("restore") or {
        "status": "available", "enabled": False
    },
    "toggle_always_on_top": lambda: calls.append("topmost") or {
        "status": "updated", "enabled": True
    },
    "set_window_locked": lambda enabled: calls.append(("locked", bool(enabled))) or {
        "status": "available", "locked": bool(enabled), "enabled": bool(enabled)
    },
    "foreground_window": lambda: calls.append("foreground") or {
        "status": "available", "title": "编辑器", "process_name": "editor"
    },
    "list_processes": lambda: calls.append("processes") or {
        "status": "available", "processes": [{"name": "editor", "pid": 12}]
    },
    "movement_bounds": lambda: (0, 0, 400, 400),
    "move_to": lambda x, y: calls.append(("move", x, y)) or {"status": "available", "x": x, "y": y},
})
window.resize(620, 700)
window.show()
app.processEvents()
for object_name in (
    "headerShowPetButton",
    "headerRestoreClickThroughButton",
    "headerAlwaysOnTopButton",
    "headerCenterPetButton",
):
    button = window.findChild(QPushButton, object_name)
    assert button is not None and button.isVisible()
assert window._navigate_console_page("桌宠")
app.processEvents()
for object_name in (
    "foregroundQuickButton",
    "processesQuickButton",
    "windowLockQuickButton",
):
    button = window.findChild(QPushButton, object_name)
    assert button is not None and button.isVisible()
window.findChild(QPushButton, "headerShowPetButton").click()
window.findChild(QPushButton, "headerRestoreClickThroughButton").click()
window.findChild(QPushButton, "headerAlwaysOnTopButton").click()
window.findChild(QPushButton, "headerCenterPetButton").click()
window.findChild(QPushButton, "foregroundQuickButton").click()
window.findChild(QPushButton, "processesQuickButton").click()
window.findChild(QPushButton, "windowLockQuickButton").click()
assert calls[:3] == ["show", "restore", "topmost"]
assert "foreground" in calls and "processes" in calls
assert ("locked", True) in calls
assert any(item[0] == "move" for item in calls if isinstance(item, tuple))
assert window._navigate_console_page("概览")
app.processEvents()
window.set_compact_layout(True)
app.processEvents()
rescue_buttons = [
    window.findChild(QPushButton, name)
    for name in (
        "headerShowPetButton",
        "headerRestoreClickThroughButton",
        "headerAlwaysOnTopButton",
        "headerCenterPetButton",
        "headerWebConsoleButton",
    )
]
assert all(button is not None and button.isVisible() for button in rescue_buttons)
assert len({button.geometry().y() for button in rescue_buttons}) == 1
window.set_pet_feedback({
    "zone": "head",
    "phrase": "喵？摸摸头～",
    "mood": "happy",
    "affection": {"current": 12, "applied": 1},
})
assert "猫猫头" in window.findChild(type(window._pet_feedback), "petInteractionFeedback").text()
assert "好感度 12" in window.findChild(type(window._pet_feedback), "petInteractionFeedback").text()
window.clear_pet_feedback()
assert "点击猫猫头" in window._pet_feedback.text()
window.shutdown()
app.processEvents()
print("console-rescue-feedback-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-rescue-feedback-ok" in result.stdout


def test_console_reports_wayland_topmost_as_boundary_not_failure(tmp_path: Path) -> None:
    """Wayland 置顶请求已提交时，控制台应说明边界而不是报未完成。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QFrame, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(callbacks={
    "toggle_always_on_top": lambda: {
        "status": "degraded",
        "enabled": True,
        "detail": "Wayland compositor may override the topmost request",
    },
})
window.show()
app.processEvents()
button = window.findChild(QPushButton, "headerAlwaysOnTopButton")
assert button is not None
button.click()
assert "桌面合成器" in window.status_label.text()
assert "未完成" not in window.status_label.text()
assert button.text() == "取消置顶"
window.shutdown()
app.processEvents()
print("console-wayland-topmost-boundary-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-wayland-topmost-boundary-ok" in result.stdout


def test_console_topmost_button_reenables_after_async_terminal_state(tmp_path: Path) -> None:
    """Qt 控制台置顶按钮在异步终态前禁用，完成后恢复可操作。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(callbacks={
    "toggle_always_on_top": lambda: {"status": "requested", "enabled": True}
})
window.show()
app.processEvents()
button = window.findChild(QPushButton, "headerAlwaysOnTopButton")
assert button is not None
button.click()
assert button.isEnabled() is False
assert window.findChild(QPushButton, "toggleAlwaysOnTopButton").isEnabled() is False
window.set_always_on_top_status({"status": "available", "enabled": True})
assert button.isEnabled() is True
assert window.findChild(QPushButton, "toggleAlwaysOnTopButton").isEnabled() is True
assert button.text() == "取消置顶"
window.shutdown()
app.processEvents()
print("console-topmost-async-state-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-topmost-async-state-ok" in result.stdout


def test_configuration_panel_uses_compact_page_picker_on_small_width(tmp_path: Path) -> None:
    """窄屏配置页应改为上下布局，避免固定导航压缩表单。"""

    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({"tts": {"enabled": True, "language": "zh"}})
panel.resize(620, 680)
panel.show()
app.processEvents()
assert panel._responsive_narrow is True
assert panel._section_picker is not None and panel._section_picker.isVisible()
assert panel.navigation.isHidden()
panel._section_picker.setCurrentIndex(panel.sections.index("tts"))
app.processEvents()
assert panel.navigation.currentRow() == panel.sections.index("tts")
assert panel.stack.currentIndex() == panel.sections.index("tts")
panel._section_picker.setCurrentIndex(panel.sections.index("llm"))
app.processEvents()
assert panel.model_setup_button is not None and panel.model_setup_button.isVisible()
assert panel.model_setup_button.text() == "配置模型与 API 密钥"
panel.resize(900, 680)
app.processEvents()
assert panel._responsive_narrow is False
assert panel.navigation.isVisible()
assert panel._section_picker.isHidden()
panel.close()
app.processEvents()
print("config-responsive-picker-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "config-responsive-picker-ok" in result.stdout


def test_configuration_overview_reflows_cards_without_horizontal_clip(tmp_path: Path) -> None:
    """独立配置页缩到窄屏时，总览卡片应改为单列并保持可见。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QScrollArea
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({"llm": {"channels": []}})
panel.resize(420, 500)
panel.show()
app.processEvents()
assert panel._responsive_narrow is True
assert panel._overview_columns == 1
assert all(card.isVisible() for card in panel._overview_cards)
scroll = panel.findChild(QScrollArea, "friendlyOverviewScroll")
assert scroll is not None
assert all(card.geometry().right() <= scroll.widget().width() for card in panel._overview_cards)
panel.close()
app.processEvents()
print("config-overview-reflow-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "config-overview-reflow-ok" in result.stdout


def test_console_auto_compacts_and_wraps_rescue_actions_on_tiny_window(tmp_path: Path) -> None:
    """直接缩放到低分辨率时，控制台仍保留可点击的恢复入口。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QFrame, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow()
window.setMinimumSize(1, 1)
window.resize(420, 500)
window.show()
app.processEvents()
assert bool(window.property("compactLayout")) is True
buttons = [
    window.findChild(QPushButton, name)
    for name in (
        "headerShowPetButton",
        "headerRestoreClickThroughButton",
        "headerAlwaysOnTopButton",
        "headerCenterPetButton",
        "headerWebConsoleButton",
    )
]
assert all(button is not None and button.isVisible() for button in buttons)
assert len({button.geometry().y() for button in buttons}) >= 2
rescue_card = window.findChild(QFrame, "consoleRescueCard")
assert rescue_card is not None
assert all(rescue_card.rect().contains(button.geometry().center()) for button in buttons)
window.shutdown()
app.processEvents()
print("console-tiny-responsive-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-tiny-responsive-ok" in result.stdout


def test_console_extreme_width_reflows_input_and_quick_actions(tmp_path: Path) -> None:
    """支持的窄屏范围内，输入、观察和动作按钮必须留在窗口内。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QWidget
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
for width in (420, 680):
    window = PetConsoleWindow(configuration={"llm": {"channels": []}})
    window.setMinimumSize(1, 1)
    window.resize(width, 400)
    window.show()
    app.processEvents()
    assert window._navigate_console_page("对话")
    app.processEvents()
    assert getattr(window, "_console_page_columns", 2) == 1
    assert getattr(window, "_context_action_columns", 3) <= 2 if width == 420 else True
    assert getattr(window, "_quick_action_columns", 3) <= 2 if width == 420 else True
    for name in (
        "conversationInput",
        "sendMessageButton",
        "stopConversationButton",
    ):
        child = window.findChild(QWidget, name)
        assert child is not None and child.isVisible()
        bottom_right = child.mapTo(window, child.rect().bottomRight())
        assert bottom_right.x() <= window.rect().right()
    assert window._navigate_console_page("桌宠")
    app.processEvents()
    for name in (
        "foregroundQuickButton",
        "processesQuickButton",
        "windowLockQuickButton",
        "rendererModelSelector",
        "applyRendererModelButton",
    ):
        child = window.findChild(QWidget, name)
        assert child is not None and child.isVisible()
        bottom_right = child.mapTo(window, child.rect().bottomRight())
        assert bottom_right.x() <= window.rect().right()
    window.shutdown()
    app.processEvents()
print("console-extreme-width-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-extreme-width-ok" in result.stdout


def test_model_setup_dialog_uses_scrollable_low_resolution_bounds(tmp_path: Path) -> None:
    """模型向导在低分辨率屏仍保留固定的测试/提交底栏。"""

    script = r"""
from PySide6.QtCore import QRect
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog

app = QApplication.instance() or QApplication([])
dialog = ModelSetupDialog()
class Screen:
    def availableGeometry(self):
        return QRect(0, 0, 420, 500)
dialog.frameGeometry = lambda: QRect(0, 0, 420, 500)
from gui.qt6 import model_setup
original_screen_at = model_setup.QGuiApplication.screenAt
original_primary = model_setup.QGuiApplication.primaryScreen
model_setup.QGuiApplication.screenAt = staticmethod(lambda _point: Screen())
model_setup.QGuiApplication.primaryScreen = staticmethod(lambda: Screen())
try:
    dialog.show()
    app.processEvents()
    assert dialog.minimumWidth() <= 404
    assert dialog.minimumHeight() <= 484
    assert dialog.test_button.isVisible()
    assert dialog.save_button is not None and dialog.save_button.isVisible()
finally:
    model_setup.QGuiApplication.screenAt = original_screen_at
    model_setup.QGuiApplication.primaryScreen = original_primary
    dialog.close()
    app.processEvents()
print("model-setup-tiny-bounds-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "model-setup-tiny-bounds-ok" in result.stdout


def test_model_setup_provider_cards_complete_configuration_without_yaml(tmp_path: Path) -> None:
    """点击服务卡片即可填充可提交草稿，技术字段和 YAML 默认隐藏。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.model_setup import ModelSetupDialog

app = QApplication.instance() or QApplication([])
dialog = ModelSetupDialog()
dialog.show()
app.processEvents()
assert "可直接点选上方服务预设" in dialog.card.status_label.text()
openai = next(
    button for button in dialog.card.findChildren(QPushButton)
    if button.objectName() == "modelPresetButton" and button.text() == "OpenAI"
)
openai.click()
app.processEvents()
assert dialog.card.model_edit.text() == "gpt-4o-mini"
assert dialog.card.base_url_edit.text() == "https://api.openai.com/v1"
assert "云端密钥" in dialog.card._connection_summary.text()
assert "OPENAI_API_KEY" not in dialog.card._connection_summary.text()
assert dialog.card.is_valid
assert dialog.save_button.isEnabled()
assert dialog.card._advanced_group.isHidden()
assert not dialog.card.preview_edit.isVisible()
dialog.card._preview_toggle.click()
app.processEvents()
preview = dialog.card.preview_edit.toPlainText()
assert "服务：OpenAI" in preview
assert "模型：gpt-4o-mini" in preview
assert "base_url:" not in preview
assert "protocol:" not in preview
dialog.card._advanced_toggle.click()
app.processEvents()
assert dialog.card._advanced_group.isVisible()
dialog.close()
app.processEvents()
print("model-provider-card-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "model-provider-card-ok" in result.stdout


def test_console_model_status_and_failure_details_are_safe(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    configuration={"llm": {"channels": []}},
    callbacks={
        "expression": lambda _value: {
            "status": "failed",
            "detail": "api_key=secret-value",
            "arguments": {"token": "secret-value"},
        },
        "config_save": lambda _values: {
            "status": "failed",
            "detail": "api_key=secret-value",
        },
    },
)
window.show()
app.processEvents()
assert window.model_ready is False
assert bool(window.windowFlags() & Qt.WindowStaysOnTopHint)
assert "模型渠道未配置" in window.status_label.text()
window.set_status(
    "模型渠道未配置：运行 meapet-wizard channel add，或设置 "
    "MEAPET_API_BASE 与 MEAPET_MODEL"
)
assert "点击“配置模型”" in window.status_label.text()
assert "meapet-wizard" not in window.status_label.text()
window.expression_quick_buttons[0].click()
assert "secret-value" not in window.output_view.toPlainText()
assert "arguments" not in window.output_view.toPlainText()
assert "敏感详情已隐藏" in window.output_view.toPlainText()
assert window.show_model_settings()
dialog = window._model_setup_dialog
assert dialog is not None
assert dialog.windowModality() == Qt.WindowModality.NonModal
assert bool(dialog.windowFlags() & Qt.WindowStaysOnTopHint)
dialog.card.preset_combo.setCurrentIndex(dialog.card.preset_combo.findData("openai"))
assert dialog.card.submit()
assert dialog.isVisible()
assert "保存未完成" in dialog.card.status_label.text()
assert "secret-value" not in dialog.card.status_label.text()
window.shutdown()
app.processEvents()
print("console-model-status-safe-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-model-status-safe-ok" in result.stdout


def test_console_chat_click_reports_submitted_and_failed_states(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
submitted = PetConsoleWindow(callbacks={"submit": lambda _value: None})
submitted.input_line.setText("你好")
submitted.send_button.click()
assert "消息已提交" in submitted.status_label.text()
submitted.shutdown()

failed = PetConsoleWindow(callbacks={
    "submit": lambda _value: {"status": "failed", "detail": "api_key=secret-value"}
})
failed.input_line.setText("你好")
failed.send_button.click()
assert "消息未提交" in failed.status_label.text()
assert "secret-value" not in failed.output_view.toPlainText()
failed.shutdown()
pending = PetConsoleWindow(callbacks={"submit": lambda _value: {"status": "pending"}})
pending.input_line.setText("等待配置")
pending.send_button.click()
assert "消息已提交" in pending.status_label.text()
assert "消息未提交" not in pending.status_label.text()
pending.shutdown()
app.processEvents()
print("console-chat-state-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-chat-state-ok" in result.stdout


def test_console_approval_list_only_shows_safe_summary(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
window = PetConsoleWindow(callbacks={
    "approve_approval": lambda approval_id, _grant: calls.append(approval_id)
    or {"status": "available"},
})
window.set_pending_approvals([{
    "approval_id": "approval-1",
    "identity": "system:run_command",
    "safe_summary": "命令参数 token=secret-value",
}])
assert "secret-value" not in window.approval_list.currentItem().text()
window.findChild(QPushButton, "approveApprovalButton").click()
assert calls == ["approval-1"]
assert "secret-value" not in window.status_label.text()
window.shutdown()
app.processEvents()
print("console-approval-safe-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-approval-safe-ok" in result.stdout


def test_console_diagnostic_card_shows_safe_desktop_summary(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow()
observation = window.findChild(type(window._pet_feedback), "desktopObservationStatus")
assert observation is not None
window.set_diagnostic_result("前台窗口", {
    "status": "available",
    "title": "编辑器",
    "process_name": "editor",
    "pid": 42,
    "command_line": ["editor", "--token=secret-value"],
})
output = window.findChild(type(window._diagnostic_output), "diagnosticOutput").toPlainText()
assert "编辑器" in output
assert "编辑器" in observation.text()
assert "secret-value" not in output
window.set_diagnostic_result("进程列表", {
    "status": "available",
    "processes": [{"pid": 42, "name": "editor", "executable": "/usr/bin/editor"}],
})
assert "editor" in window.findChild(
    type(window._diagnostic_output), "diagnosticOutput"
).toPlainText()
assert "editor" in observation.text()
window.shutdown()
app.processEvents()
print("console-diagnostic-summary-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-diagnostic-summary-ok" in result.stdout


def test_console_display_size_is_preset_only(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
window = PetConsoleWindow(
    callbacks={
        "set_display_size": lambda preset: calls.append(preset)
        or {"status": "available", "preset": preset, "width": 420, "height": 480}
    }
)
for object_name, preset in (
    ("displaySizeSmallButton", "small"),
    ("displaySizeStandardButton", "standard"),
    ("displaySizeLargeButton", "large"),
):
    button = window.findChild(QPushButton, object_name)
    assert button is not None
    button.click()
assert calls == ["small", "standard", "large"]
window.shutdown()
app.processEvents()
print("console-display-size-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-display-size-ok" in result.stdout


def test_console_diagnostic_clicks_render_safe_summary(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(callbacks={
    "foreground_window": lambda: {
        "status": "available", "title": "编辑器", "process_name": "editor",
        "pid": 42, "command_line": ["--token=secret-value"],
    },
    "list_processes": lambda: {
        "status": "available", "processes": [
            {"pid": 42, "name": "editor", "command": ["--secret-value"]}
        ],
    },
})
window.findChild(QPushButton, "foregroundWindowButton").click()
assert "编辑器" in window._diagnostic_output.toPlainText()
assert "secret-value" not in window._diagnostic_output.toPlainText()
assert "前台窗口读取" in window.status_label.text()
window.findChild(QPushButton, "listProcessesButton").click()
assert "editor" in window._diagnostic_output.toPlainText()
assert "secret-value" not in window._diagnostic_output.toPlainText()
assert "进程读取" in window.status_label.text()
window.shutdown()
app.processEvents()
print("diagnostic-click-summary-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "diagnostic-click-summary-ok" in result.stdout


def test_console_api_audit_button_renders_timing_and_model_summary(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(callbacks={
    "api_audit": lambda: {
        "status": "available",
        "count": 1,
        "summary": {
            "total": 1,
            "by_status": [
                {
                    "key": "status:completed",
                    "count": 1,
                    "avg_first_ms": 42.0,
                    "avg_total_ms": 180.0,
                }
            ],
            "by_channel": [
                {
                    "key": "channel:primary",
                    "count": 1,
                    "avg_first_ms": 42.0,
                    "avg_total_ms": 180.0,
                }
            ],
            "latency_ms": {"avg_first": 42.0, "avg_total": 180.0, "max_total": 180.0},
        },
            "records": [{
                "status": "completed",
                "kind": "model",
                "request_id": "model-123",
                "provider": "openai",
                "protocol": "openai_chat",
                "channel_id": "primary",
                "channel_info": {"region": "local", "stream": True},
                "requested_model": "request-model",
                "response_model": "served-model",
                "time_to_first_token_ms": 42.0,
                "first_char": "当",
                "total_duration_ms": 180.0,
                "tool_execution_count": 2,
                "input_payload": {"messages": [{"role": "user", "content": "查询当前窗口"}]},
                "output_text": "当前窗口是编辑器。",
                "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
                "cache_read_tokens": 8,
                "cache_write_tokens": 3,
                "cache_read_duration_ms": 1.5,
                "cache_write_duration_ms": 2.5,
                "tool_calls": [{
                    "identity": "desktop:observe_foreground",
                    "arguments": "{\"include_title\":true}",
                }],
                "tool_executions": [{
                    "identity": "desktop:observe_foreground",
                    "status": "completed",
                    "duration_ms": 12.0,
                    "arguments": {"include_title": True},
                    "result": {"title": "编辑器"},
                    "phase": "execute",
                    "attempt": 1,
                    "plan_id": "plan-1",
                    "step_id": "step-1",
                }],
        }],
    },
    "log_records": lambda: {
        "status": "available",
        "count": 1,
        "records": [{
            "level": "INFO",
            "logger": "services.demo",
            "event": "demo.completed",
            "message": "结构化日志已脱敏",
        }],
    },
})
button = window.findChild(QPushButton, "apiAuditButton")
assert button is not None
button.click()
output = window.findChild(type(window._diagnostic_output), "diagnosticOutput").toPlainText()
assert "聚合摘要" in output
assert "status:completed x 1" in output
assert "channel:primary x 1" in output
assert "平均首字 42ms" in output
assert "primary" in output
assert "openai / openai_chat" in output
assert "渠道配置" in output
assert "request-model→served-model" in output
assert "首字 42ms" in output
assert "（当）" in output
assert "总计 180ms" in output
assert "工具 2" in output
assert "输入：查询当前窗口" in output
assert "输出：当前窗口是编辑器" in output
assert "用量：输入令牌 12，输出令牌 5，总令牌 17" in output
assert "缓存：读 8，写 3，读耗时 2ms，写耗时 2ms" in output
assert "调用：desktop:observe_foreground" in output
assert "参数：{\"include_title\":true}" in output
assert "结果：{\"title\":\"编辑器\"}" in output
assert "计划 plan-1" in output
assert "desktop:observe_foreground" in output
window.findChild(QPushButton, "logRecordsButton").click()
log_output = window.findChild(type(window._diagnostic_output), "diagnosticOutput").toPlainText()
assert "运行日志：1 条" in log_output
assert "services.demo / demo.completed" in log_output
assert "结构化日志已脱敏" in log_output
window.shutdown()
app.processEvents()
print("console-api-audit-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-api-audit-ok" in result.stdout


def test_console_log_table_filters_structured_rows(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QTableWidget
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow()
window.set_diagnostic_result("运行日志", {
    "status": "available",
    "count": 3,
    "records": [
        {
            "level": "INFO",
            "logger": "services.demo",
            "event": "demo.started",
            "status": "completed",
            "message": "启动完成",
        },
        {
            "level": "ERROR",
            "logger": "services.demo",
            "event": "demo.failed",
            "status": "failed",
            "message": "连接失败",
        },
        {
            "level": "WARNING",
            "logger": "services.other",
            "event": "other.warning",
            "status": "degraded",
            "message": "已降级",
        },
    ],
})
app.processEvents()
table = window.findChild(QTableWidget, "diagnosticLogTable")
search = window.findChild(QLineEdit, "logSearchInput")
level = window.findChild(QComboBox, "logLevelFilter")
assert table is not None and search is not None and level is not None
assert table.rowCount() == 3
search.setText("failed")
app.processEvents()
assert table.rowCount() == 1
assert table.item(0, 2).text() == "demo.failed"
search.clear()
level.setCurrentIndex(level.findData("WARNING"))
app.processEvents()
assert table.rowCount() == 1
assert table.item(0, 0).text() == "WARNING"
window.shutdown()
app.processEvents()
print("console-log-table-filter-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-log-table-filter-ok" in result.stdout


def test_configuration_panel_model_button_opens_clickable_dialog(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    configuration={"llm": {"channels": []}},
    callbacks={"config_save": lambda _values: {"status": "saved"}},
)
panel = window.configuration_panel
assert panel is not None
panel.navigation.setCurrentRow(panel.sections.index("llm"))
panel.model_setup_button.click()
app.processEvents()
assert window._model_setup_dialog is not None
assert window._model_setup_dialog.isVisible()
assert window._model_setup_dialog.card.preset_combo.count() >= 5
window._model_setup_dialog.close()
window.shutdown()
app.processEvents()
print("panel-model-button-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "panel-model-button-ok" in result.stdout


def test_console_secondary_actions_are_collapsed_until_needed(tmp_path: Path) -> None:
    """普通用户首屏不应直接看到坐标、诊断和审批参数区。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton, QWidget
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration={"llm": {"channels": []}})
window.resize(620, 700)
window.show()
app.processEvents()
panel = window.findChild(QWidget, "secondaryActionsPanel")
toggle = window.findChild(QPushButton, "moreActionsToggleButton")
developer_toggle = window.findChild(QPushButton, "customActionToggleButton")
developer_panel = window.findChild(QWidget, "advancedActionPanel")
assert panel is not None and toggle is not None
assert developer_toggle is not None and developer_panel is not None
assert window._navigate_console_page("桌宠")
app.processEvents()
assert developer_toggle.isVisible()
assert not developer_panel.isVisible()
window.configuration_panel.advanced_toggle.click()
app.processEvents()
assert developer_toggle.isVisible()
developer_toggle.click()
assert developer_panel.isVisible()
window.configuration_panel.advanced_toggle.click()
app.processEvents()
assert developer_toggle.isVisible()
assert developer_panel.isVisible()
developer_toggle.click()
assert not developer_panel.isVisible()
assert not panel.isVisible()
toggle.click()
assert panel.isVisible()
assert toggle.text() == "收起更多操作"
toggle.click()
assert not panel.isVisible()
window.shutdown()
app.processEvents()
print("console-secondary-actions-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-secondary-actions-ok" in result.stdout


def test_console_narrow_navigation_separates_chat_and_pet_actions(
    tmp_path: Path,
) -> None:
    """窄屏对话与桌宠操作应分页呈现，避免一个长页面互相挤压。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QFrame
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow()
window.resize(620, 700)
window.show()
app.processEvents()
conversation = window.findChild(QFrame, "conversationCard")
quick = window.findChild(QFrame, "quickActionsCard")
assert conversation is not None and quick is not None
assert not conversation.isVisible()
assert not quick.isVisible()
assert window._navigate_console_page("对话")
app.processEvents()
assert conversation.isVisible()
assert not quick.isVisible()
assert window.input_line.isVisible()
assert window._navigate_console_page("桌宠")
app.processEvents()
assert quick.isVisible()
assert not conversation.isVisible()
window.shutdown()
app.processEvents()
print("console-narrow-chat-first-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-narrow-chat-first-ok" in result.stdout


def test_console_renders_interaction_snapshot_actions_on_first_screen(tmp_path: Path) -> None:
    """交互快照的批准/拒绝动作应直接出现在首屏。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
window = PetConsoleWindow(callbacks={
    "approve_approval": lambda approval_id, grant: (
        calls.append(("approve", approval_id, grant)) or {"status": "requested"}
    ),
    "deny_approval": lambda approval_id: (
        calls.append(("deny", approval_id)) or {"status": "requested"}
    ),
})
window.set_interaction_actions([
    {"kind": "configure_model", "label": "配置模型"},
    {"kind": "approve", "label": "批准", "payload": {"approval_id": "a-1"}},
    {"kind": "grant_session", "label": "批准并允许本会话", "payload": {"approval_id": "a-1"}},
    {"kind": "deny", "label": "拒绝", "payload": {"approval_id": "a-1"}},
])
first_button = window.findChild(QPushButton, "stateAction_grant_session")
window.set_interaction_actions([
    {"kind": "configure_model", "label": "配置模型"},
    {"kind": "approve", "label": "批准", "payload": {"approval_id": "a-1"}},
    {"kind": "grant_session", "label": "批准并允许本会话", "payload": {"approval_id": "a-1"}},
    {"kind": "deny", "label": "拒绝", "payload": {"approval_id": "a-1"}},
])
assert window.findChild(QPushButton, "stateAction_grant_session") is first_button
assert window.findChild(QPushButton, "configureModelButton") is not None
assert window.findChild(QPushButton, "stateAction_configure_model") is None
assert window.findChild(QPushButton, "stateAction_approve") is not None
assert window.findChild(QPushButton, "stateAction_grant_session") is not None
assert window.findChild(QPushButton, "stateAction_deny") is not None
window.findChild(QPushButton, "stateAction_grant_session").click()
assert calls == [("approve", "a-1", True)]
window.set_interaction_actions(())
assert not window._state_actions_host.isVisible()
window.shutdown()
app.processEvents()
print("console-state-actions-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-state-actions-ok" in result.stdout


def test_console_action_surface_filters_internal_payload_and_approval_labels(
    tmp_path: Path,
) -> None:
    """动态按钮只保留审批标识，界面不显示内部身份或参数。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
window = PetConsoleWindow(callbacks={
    "approve_approval": lambda approval_id, grant: calls.append((approval_id, grant))
    or {"status": "requested"},
})
window.set_interaction_actions([{
    "kind": "approve",
    "label": "批准 api_key=secret-value",
    "payload": {
        "approval_id": "approval-safe",
        "arguments": {"token": "secret-value"},
        "command_line": "system:run_command --token=secret-value",
    },
}])
button = window.findChild(QPushButton, "stateAction_approve")
assert button is not None
assert button.text() == "批准"
assert "secret-value" not in repr(window._state_action_signature)
button.click()
assert calls == [("approval-safe", False)]
window.set_pending_approvals([{
    "approval_id": "approval-safe",
    "identity": "system:run_command",
    "safe_summary": "arguments: token=secret-value",
}])
item_text = window.approval_list.currentItem().text()
assert "system:run_command" not in item_text
assert "secret-value" not in item_text
window._approve_selected_approval()
assert "approval-safe" not in window.status_label.text()
window.shutdown()
app.processEvents()
print("console-safe-action-surface-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-safe-action-surface-ok" in result.stdout


def test_console_missing_callback_and_stop_failure_keep_failure_feedback(tmp_path: Path) -> None:
    """缺少宿主回调或停止失败时不能覆盖成“请求已提交”。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
failed = PetConsoleWindow(callbacks={"stop": lambda: {"status": "failed"}})
failed.findChild(QPushButton, "stopConversationButton").click()
assert "停止未提交" in failed.status_label.text()
missing = PetConsoleWindow()
missing.findChild(QPushButton, "showPetButton").click()
assert "功能不可用" in missing.status_label.text()
assert "请求已提交" not in missing.status_label.text()
failed.shutdown()
missing.shutdown()
app.processEvents()
print("console-failure-feedback-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-failure-feedback-ok" in result.stdout


def test_console_idle_stop_reports_no_active_message_without_noise(tmp_path: Path) -> None:
    """空闲时点击停止不能伪装成正在中断，也不能追加失败噪声。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
idle = PetConsoleWindow(callbacks={
    "stop": lambda: {
        "status": "unavailable",
        "reason": "当前没有正在生成的对话",
    }
})
idle.findChild(QPushButton, "stopConversationButton").click()
assert idle.status_label.text() == "当前没有正在生成的消息"
assert "停止未执行" not in idle.output_view.toPlainText()
queued = PetConsoleWindow(callbacks={
    "stop": lambda: {"status": "cancelled", "reason": "已取消等待中的消息"}
})
queued.findChild(QPushButton, "stopConversationButton").click()
assert queued.status_label.text() == "已取消等待中的消息"
idle.shutdown()
queued.shutdown()
app.processEvents()
print("console-idle-stop-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-idle-stop-ok" in result.stdout


def test_model_setup_hides_connection_details_until_more_settings(tmp_path: Path) -> None:
    """模型预设页默认只显示服务卡和模型名称，连接细节按需展开。"""

    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog

app = QApplication.instance() or QApplication([])
dialog = ModelSetupDialog()
dialog.show()
app.processEvents()
dialog.card.preset_combo.setCurrentIndex(dialog.card.preset_combo.findData("openai"))
app.processEvents()
assert not dialog.card.base_url_edit.isVisible()
assert not dialog.card.api_key_env_edit.isVisible()
assert "YAML" not in dialog.card._connection_summary.text()
dialog.card._advanced_toggle.click()
app.processEvents()
assert dialog.card.base_url_edit.isVisible()
assert dialog.card.api_key_env_edit.isVisible()
stylesheet = dialog.styleSheet().lower()
assert "#202020" in stylesheet
assert "#ff9dbe" not in stylesheet
dialog.close()
app.processEvents()
print("model-connection-details-hidden-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "model-connection-details-hidden-ok" in result.stdout


def test_model_setup_exposes_safe_api_key_action_without_revealing_secret(
    tmp_path: Path,
) -> None:
    """API 密钥入口应在首屏可见，点击后只打开环境变量字段。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QScrollArea
from gui.qt6.model_setup import ModelSetupDialog

app = QApplication.instance() or QApplication([])
dialog = ModelSetupDialog()
dialog.resize(520, 560)
dialog.show()
app.processEvents()
scroll = dialog.findChild(QScrollArea, "modelSetupScroll")
button_box = dialog.findChild(QDialogButtonBox)
assert scroll is not None and button_box is not None
assert scroll.verticalScrollBar().isVisible()
assert button_box.isVisible()
assert button_box.geometry().top() >= scroll.geometry().bottom()
assert dialog.test_button.isVisible()
action = dialog.card._api_key_action
assert action.isVisible()
assert "API 密钥" in action.text()
dialog.card.preset_combo.setCurrentIndex(dialog.card.preset_combo.findData("openai"))
app.processEvents()
assert action.isEnabled()
assert "云端密钥" in dialog.card._connection_summary.text()
assert "OPENAI_API_KEY" not in dialog.card._connection_summary.text()
action.click()
app.processEvents()
assert dialog.card._advanced_toggle.isChecked()
assert dialog.card.api_key_env_edit.isVisible()
assert dialog.card.api_key_value_edit.isVisible()
assert dialog.card.api_key_env_edit.hasFocus()
assert "密钥由启动环境提供" in action.toolTip()
dialog.card.preset_combo.setCurrentIndex(dialog.card.preset_combo.findData("ollama"))
app.processEvents()
assert not action.isEnabled()
assert not dialog.card.api_key_value_edit.isVisible()
assert "无需 API Key" in dialog.card._connection_summary.text()
dialog.close()
app.processEvents()
print("model-safe-api-action-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "model-safe-api-action-ok" in result.stdout


def test_model_setup_preserves_channel_priority_when_editing(tmp_path: Path) -> None:
    """编辑模型名称或预设时不能意外重置已有渠道优先级。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupCard, ModelSetupDraft

app = QApplication.instance() or QApplication([])
draft = ModelSetupDraft.from_preset(
    "openai", channel_id="priority-channel", priority=37
)
card = ModelSetupCard(draft, channel_drafts=(draft,))
assert card.draft().priority == 37
card.model_edit.setText("gpt-4.1-mini")
assert card.draft().priority == 37
card.preset_combo.setCurrentIndex(card.preset_combo.findData("custom"))
assert card.draft().priority == 37
card.close()
app.processEvents()
print("model-priority-preserved-ok")
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
    assert "model-priority-preserved-ok" in result.stdout


def test_console_window_lock_keeps_cursor_behavior_label_and_state(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
states = []
window = PetConsoleWindow(callbacks={
    "set_window_locked": lambda enabled: states.append(bool(enabled)) or {
        "status": "available", "locked": bool(enabled), "enabled": bool(enabled)
    }
})
assert window.window_lock_checkbox.toolTip()
window.window_lock_checkbox.setChecked(True)
assert states == [True]
assert window.window_lock_checkbox.isChecked()
assert not window.click_through_checkbox.isEnabled()
window.set_window_locked_checked(False)
assert not window.window_lock_checkbox.isChecked()
assert window.click_through_checkbox.isEnabled()
window.shutdown()
app.processEvents()
print("console-window-lock-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-window-lock-ok" in result.stdout


def test_runtime_console_hides_developer_surface(tmp_path: Path) -> None:
    """运行控制台默认只显示点击式配置入口，不展示完整配置开关或路径。"""

    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    configuration={"llm": {"channels": []}},
    configuration_path="/tmp/meapet-runtime-config.yaml",
    developer_mode=False,
)
window.show()
app.processEvents()
panel = window.configuration_panel
assert panel is not None
assert panel.advanced_toggle is not None
assert not panel.advanced_toggle.isVisible()
assert not panel._path_label.isVisible()
assert panel.advanced_editing is False
window.shutdown()
app.processEvents()
print("runtime-click-surface-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "runtime-click-surface-ok" in result.stdout


def test_console_tts_profile_and_language_controls_are_clickable(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
seen = []
window = PetConsoleWindow(callbacks={
    "select_tts_profile": lambda value: seen.append(("profile", value)) or {"status": "updated"},
    "select_tts_language": lambda value: seen.append(("language", value)) or {"status": "updated"},
})
window.set_tts_options(
    [{"id": "zh-main"}, {"id": "jp-main"}],
    language="jp",
    active_profile="jp-main",
)
app.processEvents()
assert window.findChild(QPushButton, "ttsLanguage_zh") is not None
assert window.findChild(QPushButton, "ttsProfile_0") is not None
window.findChild(QPushButton, "ttsLanguage_zh").click()
window.findChild(QPushButton, "ttsProfile_0").click()
assert ("language", "zh") in seen
assert ("profile", "zh-main") in seen
window.shutdown()
app.processEvents()
print("console-tts-controls-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-tts-controls-ok" in result.stdout


def test_console_structured_actions_and_tts_health_are_operable(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
seen = []
window = PetConsoleWindow(callbacks={
    "expression_request": (
        lambda value: seen.append(("expression", value)) or {"status": "completed"}
    ),
    "motion_request": lambda value: seen.append(("motion", value)) or {"status": "completed"},
})
window.findChild(QLineEdit, "expressionSequenceInput").setText("happy, curious")
window.findChild(QLineEdit, "expressionWeightsInput").setText("0.8, 0.2")
window.findChild(QLineEdit, "expressionParametersInput").setText("ParamAngleX=12")
window.findChild(QLineEdit, "expressionRestoreInput").setText("neutral")
window._expression_mode.setCurrentIndex(1)
window._expression_loop.setChecked(True)
window._play_expression_sequence()
window._motion.setCurrentText("wave")
window.findChild(QLineEdit, "motionParametersInput").setText("ParamBodyAngleX=4")
window._motion_loop.setChecked(True)
window._play_motion_request()
expression = seen[0][1]
assert expression["mode"] == "blend"
assert expression["loop"] is True
assert expression["restore"] == "neutral"
assert [row["weight"] for row in expression["expressions"]] == [0.8, 0.2]
assert expression["expressions"][0]["parameters"] == {"ParamAngleX": 12.0}
motion = seen[1][1]
assert motion["name"] == "wave" and motion["loop"] is True
assert motion["parameters"] == {"ParamBodyAngleX": 4.0}
window.set_tts_options(
    [],
    health={
        "available": False,
        "pending": True,
        "engine": "pending",
        "message": "TTS initialization is running",
    },
)
assert "语音后端初始化" in window._tts_status.text()
assert "初始化中" in window._tts_status.text()
window.set_tts_options(
    [
        {"id": "zh-main", "languages": ["zh"], "enabled": True},
        {"id": "jp-off", "languages": ["ja"], "enabled": False},
    ],
    language="zh",
    active_profile="zh-main",
    health={
        "available": True,
        "engine": "gpt-sovits-stdio",
        "message": "persistent worker is running",
    },
)
tts_status = window.findChild(QLabel, "consoleMetaLabel")
assert "GPT-SoVITS 标准输入" in window._tts_status.text()
assert "已就绪" in window._tts_status.text()
assert window.findChild(QPushButton, "ttsProfile_1").isEnabled() is False
window.shutdown()
app.processEvents()
print("console-structured-actions-tts-health-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-structured-actions-tts-health-ok" in result.stdout


def test_console_uses_fancyui_winui_navigation_and_components(tmp_path: Path) -> None:
    """主控制台必须使用左侧导航和真实 FancyUI 组件，而非顶部标签换色。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QFrame, QPushButton, QTabWidget, QToolButton
from gui.qt6.console import PetConsoleWindow, _CONSOLE_STYLE
from gui.qt6.fancyui import (
    FancyCard,
    FancyCommandBar,
    FancyInfoBar,
    FancyNavigationView,
    FancyStyleController,
)

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration={"llm": {"channels": []}})
window.show()
app.processEvents()
navigation = window.findChild(FancyNavigationView, "consoleTabs")
assert navigation is not None
assert isinstance(navigation, QTabWidget)
assert navigation.tabPosition() == QTabWidget.TabPosition.West
assert [navigation.tabText(index) for index in range(navigation.count())] == [
    "概览", "对话", "桌宠", "工具与安全", "模块中心", "配置中心"
]
assert navigation.currentWidget().objectName() == "consoleOverviewScroll"
assert isinstance(window.findChild(QFrame, "conversationCard"), FancyCard)
assert isinstance(window.findChild(QFrame, "quickActionsCard"), FancyCard)
assert isinstance(window.findChild(QFrame, "consoleHeaderCard"), FancyCard)
assert isinstance(window.findChild(QFrame, "consoleStatusInfoBar"), FancyInfoBar)
command_bar = window.findChild(QFrame, "consoleCommandBar")
assert isinstance(command_bar, FancyCommandBar)
assert isinstance(window._fancy_style_controller, FancyStyleController)
assert window._md3_theme.colors.primary == "#60CDFF"
assert "#ff9dbe" not in _CONSOLE_STYLE.lower()
assert "@primary@" in _CONSOLE_STYLE
assert "#ff9dbe" not in window.styleSheet().lower()
assert "@primary@" not in window.styleSheet()
window.resize(620, 780)
app.processEvents()
assert navigation.isCompact()
assert navigation.accessibleName() == "控制台功能导航"
assert navigation.tabBar().accessibleName() == "控制台功能导航"
for index in range(navigation.count()):
    assert navigation.tabToolTip(index) == navigation.tabText(index)
command_buttons = {
    button.defaultAction().text(): button
    for button in command_bar.findChildren(QToolButton)
    if button.defaultAction() is not None
}
expected_tooltips = {
    "刷新状态": "重新读取模型、语音、渲染与工具模块状态",
    "显示桌宠": "桌宠不可见时重新显示",
    "恢复鼠标": "关闭点击穿透并恢复桌宠交互",
}
for label, tooltip in expected_tooltips.items():
    assert command_buttons[label].accessibleName() == label
    assert command_buttons[label].toolTip() == tooltip
assert window.findChild(QPushButton, "configCenterButton").accessibleName() == "配置中心"
assert window.findChild(QPushButton, "configureModelButton").accessibleName() == "配置模型"
window.shutdown()
app.processEvents()
print("console-fancyui-navigation-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-fancyui-navigation-ok" in result.stdout


def test_console_page_switch_never_exposes_internal_labels_as_windows(tmp_path: Path) -> None:
    """页面切换和响应式重排不能把内部状态标签提升为独立窗口。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QLabel
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration={
    "llm": {
        "channels": [{
            "id": "main",
            "adapter": "openai",
            "protocol": "openai",
            "base_url": "http://127.0.0.1:8317",
            "model": "gpt-5.4-mini",
            "enabled": True,
        }],
    },
})
window.show()
app.processEvents()

for page in ("配置中心", "桌宠", "概览", "对话", "工具与安全", "配置中心", "桌宠"):
    assert window._navigate_console_page(page)
    for width, height in ((980, 760), (520, 560), (420, 500), (680, 760)):
        window.resize(width, height)
        app.processEvents()
        orphan_labels = [
            widget
            for widget in QApplication.topLevelWidgets()
            if isinstance(widget, QLabel) and widget.parent() is None
        ]
        assert not orphan_labels, [
            (widget.objectName(), widget.text(), widget.isVisible())
            for widget in orphan_labels
        ]
        assert window._status.parent() is window
        assert window._hint.parent() is window
        assert window._status.isHidden()
        assert window._hint.isHidden()

application_windows = [
    widget
    for widget in QApplication.topLevelWidgets()
    if widget.parent() is None and widget.isVisible()
]
assert application_windows == [window], [
    (type(widget).__name__, widget.objectName(), widget.windowTitle())
    for widget in application_windows
]
window.shutdown()
app.processEvents()
print("console-no-orphan-label-windows-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-no-orphan-label-windows-ok" in result.stdout


def test_console_fancy_pages_reflow_at_720_pixels(tmp_path: Path) -> None:
    """桌宠和工具页在宽屏双栏、窄屏单栏间重排并压缩左侧导航。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QFrame
from gui.qt6.console import PetConsoleWindow
from gui.qt6.fancyui import FancyNavigationView

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow()
window.resize(980, 760)
window.show()
app.processEvents()
navigation = window.findChild(FancyNavigationView, "consoleTabs")
quick = window.findChild(QFrame, "quickActionsCard")
window_card = window.findChild(QFrame, "windowActionsCard")
approval = window.findChild(QFrame, "approvalCard")
diagnostics = window.findChild(QFrame, "diagnosticsCard")
assert window._console_page_columns == 2
assert not navigation.isCompact()
quick_pos = window._pet_page_layout.getItemPosition(
    window._pet_page_layout.indexOf(quick)
)
window_pos = window._pet_page_layout.getItemPosition(
    window._pet_page_layout.indexOf(window_card)
)
assert quick_pos[:2] == (1, 0)
assert window_pos[:2] == (1, 1)
assert window._tools_page_layout.getItemPosition(
    window._tools_page_layout.indexOf(approval)
)[:2] == (1, 0)
assert window._tools_page_layout.getItemPosition(
    window._tools_page_layout.indexOf(diagnostics)
)[:2] == (1, 1)

window.resize(680, 760)
app.processEvents()
assert window._console_page_columns == 1
assert navigation.isCompact()
quick_pos = window._pet_page_layout.getItemPosition(
    window._pet_page_layout.indexOf(quick)
)
window_pos = window._pet_page_layout.getItemPosition(
    window._pet_page_layout.indexOf(window_card)
)
assert quick_pos[:2] == (1, 0)
assert window_pos[:2] == (2, 0)
assert window._tools_page_layout.getItemPosition(
    window._tools_page_layout.indexOf(approval)
)[:2] == (1, 0)
assert window._tools_page_layout.getItemPosition(
    window._tools_page_layout.indexOf(diagnostics)
)[:2] == (2, 0)
window.shutdown()
app.processEvents()
print("console-fancyui-responsive-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-fancyui-responsive-ok" in result.stdout


def test_console_configuration_page_preserves_usable_card_viewport(
    tmp_path: Path,
) -> None:
    """常用窗口高度下应收起重复页头，不能把设置卡压成不可读细条。"""

    script = r"""
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QScrollArea
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration={"llm": {"channels": []}})
window.resize(980, 760)
window.show()
app.processEvents()
assert window.show_settings()
app.processEvents()
panel = window._configuration_panel
header = panel.findChild(QFrame, "configHeader")
scroll = panel.findChild(QScrollArea, "friendlyOverviewScroll")
command = panel.findChild(QFrame, "configCommandArea")
title = window.findChild(QLabel, "consoleTitleLabel")
assert window._console_tabs.tabText(window._console_tabs.currentIndex()) == "配置中心"
assert title is not None and title.isVisible() and title.text() == "配置中心"
assert header is not None and not header.isVisible()
assert scroll is not None and scroll.viewport().height() >= 220
assert command is not None and command.isVisible()
scroll_bottom = scroll.mapTo(panel, scroll.rect().bottomRight()).y()
command_top = command.mapTo(panel, QPoint(0, 0)).y()
assert scroll_bottom < command_top
assert window._navigate_console_page("概览")
app.processEvents()
assert title.isVisible() and title.text() == "桌宠控制台"
window.shutdown()
app.processEvents()
print("console-config-viewport-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-config-viewport-ok" in result.stdout


def test_console_defers_configuration_tree_until_first_access(tmp_path: Path) -> None:
    """首屏不得为未访问的配置域创建数千个 Qt 对象。"""

    script = r"""
from time import perf_counter
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication, QCheckBox, QWidget
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
started = perf_counter()
window = PetConsoleWindow(configuration={
    "ui": {"theme": {"mode": "light"}},
    "llm": {"channels": []},
    "tts": {"enabled": True, "language": "zh"},
    "scheduler": {"tasks": [], "triggers": []},
})
constructed = perf_counter()
assert window._configuration_panel is None
assert window.findChild(QWidget, "configurationCenterHost") is not None
assert len(window.findChildren(QObject)) < 900
assert constructed - started < 2.0
window.show()
app.processEvents()
assert window._configuration_panel is None

config_started = perf_counter()
panel = window.configuration_panel
config_ready = perf_counter()
assert panel is not None
assert panel._built_sections == {"overview"}
assert config_ready - config_started < 1.5
panel.navigation.setCurrentRow(panel.sections.index("tts"))
app.processEvents()
assert panel._built_sections == {"overview", "tts"}
tts_enabled = panel.findChild(QCheckBox, "configFriendlyToggle_tts_enabled")
assert tts_enabled is not None and tts_enabled.isChecked()
tts_form = panel.findChild(QWidget, "configForm_tts")
assert tts_form is not None and tts_form.styleSheet() == ""
assert panel._md3_theme.mode.value == "light"
assert "scheduler" not in panel._built_sections
window.shutdown()
app.processEvents()
print("console-lazy-config-performance-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-lazy-config-performance-ok" in result.stdout


def test_console_module_center_is_lazy_sanitized_and_opens_exact_configuration(
    tmp_path: Path,
) -> None:
    """模块中心应懒加载、显示图标加文本，并拒绝回显敏感诊断字段。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
snapshot = {
    "status": "available",
    "adapters": ({"protocol": "openai"}, {"protocol": "claude"}),
    "tools": ("system:module_status", "desktop:observe_foreground"),
    "tts": {
        "available": True,
        "engine": "gpt-sovits-stdio",
        "endpoint": "http://127.0.0.1:9880/secret",
    },
    "runtime": {
        "asr": {
            "status": "ready",
            "available": True,
            "backend": "sensevoice",
            "model": "SenseVoiceSmall",
            "model_loaded": True,
            "language": "zh",
            "model_path": "/private/models/model.pt",
            "transcript": "不应展示的识别正文",
            "ipc": {"status": "ready", "running": True, "ready": True, "pid": 321},
        },
        "memory": {"enabled": True, "vector_index_size": 7, "lexical_index": "fts5"},
        "mcp": {"configured": 1, "ready": 1},
        "scheduler": {"running": True, "task_count": 2, "trigger_count": 3},
        "behavior": {"running": True, "phase": "idle", "movement_enabled": True},
        "proactive": {
            "status": "idle",
            "enabled": True,
            "running": False,
            "rule_count": 1,
        },
        "renderer": {"available": True, "backend": "web_live2d"},
        "watcher": {"running": True},
        "logging": {
            "status": "degraded",
            "level": "INFO",
            "file_configured": True,
            "file_active": False,
            "rotation_enabled": False,
        },
        "ipc": {"status": "ready", "worker_count": 2},
    },
}
window = PetConsoleWindow(
    callbacks={"module_status": lambda: snapshot},
    configuration={
        "asr": {
            "enabled": True,
            "backend": "sensevoice",
            "language": "auto",
            "private_extension": {"keep": True},
        }
    },
)
assert window._module_center_loaded is False
assert window.findChild(QLabel, "moduleState_asr") is None
assert window._navigate_console_page("模块中心")
app.processEvents()
assert window._module_center_loaded is True
assert len(window._module_cards) == 15
assert window.findChild(QLabel, "moduleState_asr").text() == "正常"
icon = window.findChild(QLabel, "moduleStatusIcon_asr")
assert icon.pixmap() is not None and not icon.pixmap().isNull()
assert icon.accessibleName() == "模块状态：正常"
meta = window.findChild(QLabel, "moduleMeta_asr").text()
assert "IPC：已就绪 · 进程 321" in meta
assert "SenseVoiceSmall · 已就绪" in meta
assert "文件输出不可用" in window.findChild(QLabel, "moduleDetail_logging").text()
visible_text = "\n".join(label.text() for label in window.findChildren(QLabel))
for forbidden in (
    "127.0.0.1",
    "/private/models",
    "不应展示的识别正文",
    "endpoint",
    "model_path",
):
    assert forbidden not in visible_text

config_button = window.findChild(QPushButton, "moduleConfig_asr")
config_button.click()
app.processEvents()
panel = window._configuration_panel
assert panel is not None
assert panel.navigation.currentRow() == panel.sections.index("asr")
assert panel.draft_values()["asr"]["private_extension"] == {"keep": True}
window.shutdown()
app.processEvents()
print("console-module-center-lazy-sanitized-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-module-center-lazy-sanitized-ok" in result.stdout


def test_console_module_test_future_is_non_blocking_busy_and_terminal(tmp_path: Path) -> None:
    """专用模块测试必须立即返回、拦截重复点击并恢复明确终态。"""

    script = r"""
from concurrent.futures import Future
from time import perf_counter
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
future = Future()
calls = []
window = PetConsoleWindow(
    callbacks={"module_test": lambda module_id: calls.append(module_id) or future},
)
assert window._navigate_console_page("模块中心")
window.show()
app.processEvents()
button = window.findChild(QPushButton, "moduleTest_asr")
started = perf_counter()
button.click()
elapsed = perf_counter() - started
assert elapsed < 0.2
assert calls == ["asr"]
assert not button.isEnabled()
assert button.text() == "测试中…"
assert window.findChild(QLabel, "moduleState_asr").text() == "加载中…"
first_generation = window._module_test_generations["asr"]
window._run_module_test("asr")
assert calls == ["asr"]
future.set_result({
    "status": "ready",
    "available": True,
    "backend": "sensevoice",
    "language": "ja",
    "model": "SenseVoiceSmall",
    "model_loaded": True,
    "latency_ms": 83.4,
    "transcript": "该文本不可进入模块中心",
    "model_path": "/private/model.pt",
    "ipc": {"status": "ready", "running": True, "ready": True, "pid": 456},
})
QTest.qWait(120)
app.processEvents()
assert button.isEnabled()
assert button.text() == "测试"
assert window.findChild(QLabel, "moduleState_asr").text() == "正常"
meta = window.findChild(QLabel, "moduleMeta_asr").text()
assert "延迟：83 ms" in meta
assert "SenseVoiceSmall" in meta
visible_text = "\n".join(label.text() for label in window.findChildren(QLabel))
assert "该文本不可进入模块中心" not in visible_text
assert "/private/model.pt" not in visible_text
future = Future()
button.click()
generation = window._module_test_generations["asr"]
assert generation > first_generation
window._poll_module_test("asr", generation, future, 0)
assert future.cancelled()
assert "asr" not in window._module_test_generations
assert button.isEnabled()
assert window.findChild(QLabel, "moduleState_asr").text() == "不可用"
window.shutdown()
app.processEvents()
print("console-module-test-future-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-module-test-future-ok" in result.stdout


def test_console_module_refresh_requested_state_finishes_from_runtime_projection(
    tmp_path: Path,
) -> None:
    """全局刷新应保持 busy，直到应用组合根投影终态后才重新启用。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QLabel, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(callbacks={"module_status": lambda: {"status": "requested"}})
assert window._navigate_console_page("模块中心")
window.show()
app.processEvents()
button = window.findChild(QPushButton, "moduleTest_tts")
assert window._module_check_busy is True
assert not button.isEnabled()
assert window.findChild(QLabel, "moduleState_tts").text() == "加载中…"
window.set_diagnostic_result("module_status", {
    "status": "available",
    "tts": {"available": True, "engine": "gpt-sovits-stdio"},
    "runtime": {"asr": {"status": "disabled", "available": False}},
})
app.processEvents()
assert window._module_check_busy is False
assert button.isEnabled()
assert window.findChild(QLabel, "moduleState_tts").text() == "正常"
assert window.findChild(QLabel, "moduleState_asr").text() == "降级"
window.shutdown()
app.processEvents()
print("console-module-refresh-terminal-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-module-refresh-terminal-ok" in result.stdout


def test_console_window_controls_keep_actual_state_from_failed_receipt(tmp_path: Path) -> None:
    """失败回执必须按实际字段回写穿透和锁定状态，不能按请求值复位。"""

    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
click_calls = []
lock_calls = []
window = PetConsoleWindow(callbacks={
    "click_through": lambda enabled: click_calls.append(enabled) or {
        "status": "unavailable",
        "enabled": True,
    },
    "set_window_locked": lambda enabled: lock_calls.append(enabled) or {
        "status": "unavailable",
        "locked": True,
        "enabled": True,
    },
})
window.set_click_through_checked(False)
window.click_through_checkbox.setChecked(True)
assert click_calls == [True]
assert window.click_through_checkbox.isChecked() is False
window.set_click_through_checked(True)
window.click_through_checkbox.setChecked(False)
assert click_calls == [True, False]
assert window.click_through_checkbox.isChecked() is True

window.set_window_locked_checked(False)
window.window_lock_checkbox.setChecked(True)
assert lock_calls == [True]
assert window.window_lock_checkbox.isChecked() is False
window.set_window_locked_checked(True)
window.window_lock_checkbox.setChecked(False)
assert lock_calls == [True, False]
assert window.window_lock_checkbox.isChecked() is True
assert not window.click_through_checkbox.isEnabled()
assert window._window_lock_quick_button.text() == "解除锁定"
window.shutdown()
app.processEvents()
print("console-window-receipt-state-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-window-receipt-state-ok" in result.stdout


def test_console_api_audit_filter_panel_and_grouped_table(tmp_path: Path) -> None:
    """筛选条与分组表：默认全量渲染、筛选行数与聚合摘要既有构造不变。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QTableWidget
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
payload = {
    "status": "available",
    "count": 2,
    "summary": {
        "total": 2,
        "by_status": [
            {"key": "status:completed", "count": 1, "avg_first_ms": 42.0, "avg_total_ms": 180.0},
            {"key": "status:failed", "count": 1, "avg_first_ms": None, "avg_total_ms": 240.0},
        ],
        "by_channel": [
            {"key": "channel:primary", "count": 1, "avg_first_ms": 42.0, "avg_total_ms": 180.0},
            {"key": "channel:secondary", "count": 1, "avg_first_ms": None, "avg_total_ms": 240.0},
        ],
        "latency_ms": {"avg_first": 42.0, "avg_total": 210.0, "max_total": 240.0},
    },
    "records": [
        {
            "status": "completed", "kind": "model", "channel_id": "primary",
            "channel_name": "主渠道", "provider": "openai", "protocol": "openai_chat",
            "requested_model": "req", "response_model": "served",
            "time_to_first_token_ms": 42.0, "total_duration_ms": 180.0,
        },
        {
            "status": "failed", "kind": "channel", "channel_id": "secondary",
            "provider": "mock", "protocol": "mock",
            "requested_model": "req2", "time_to_first_token_ms": None,
            "total_duration_ms": 240.0,
        },
    ],
}
window = PetConsoleWindow(callbacks={"api_audit": lambda: payload})
window.findChild(QPushButton, "apiAuditButton").click()
output = window._diagnostic_output.toPlainText()
assert "API 调用审计：2 条" in output
assert "聚合摘要" in output
assert "status:completed x 1" in output
assert "status:failed x 1" in output
assert "channel:primary x 1" in output
assert "channel:secondary x 1" in output
table = window.findChild(QTableWidget, "diagnosticAuditTable")
status_combo = window.findChild(QComboBox, "auditStatusFilter")
channel_combo = window.findChild(QComboBox, "auditChannelFilter")
refresh_button = window.findChild(QPushButton, "auditRefreshButton")
assert table is not None
assert status_combo is not None
assert channel_combo is not None
assert refresh_button is not None
assert not table.isHidden()
assert table.rowCount() == 2
status_items = [status_combo.itemData(i) for i in range(status_combo.count())]
channel_items = [channel_combo.itemData(i) for i in range(channel_combo.count())]
assert status_items == ["", "completed", "failed"]
assert channel_items == ["", "primary", "secondary"]
# 分组表行列只包含白名单摘要字段，不携带请求 ID、载荷与密钥类内容
headers = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]
assert headers == ["状态", "类型", "渠道", "协议", "请求模型", "首字", "总耗时"]
cell_texts = set()
for row_index in range(table.rowCount()):
    for column in range(table.columnCount()):
        item = table.item(row_index, column)
        if item is not None:
            cell_texts.add(item.text())
joined = " | ".join(sorted(cell_texts))
assert "model-123" not in joined
assert "密钥" not in joined
assert "completed" in joined
assert "主渠道" not in joined
assert "primary / openai" in joined
assert "req→served" in joined
assert "42 ms" in joined
assert "240 ms" in joined
system_lines = window._log_lines if hasattr(window, "_log_lines") else []
window.shutdown()
app.processEvents()
print("console-api-audit-filter-panel-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-api-audit-filter-panel-ok" in result.stdout


def test_console_api_audit_filter_refresh_routes_host_and_rejects_unknown(
    tmp_path: Path,
) -> None:
    """audit_refresh 回调：过滤键与客户端读取一致，回执映射更新表格且不越权。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QComboBox, QPushButton, QTableWidget
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
seen_filters = []

def audit_refresh(filters):
    seen_filters.append(dict(filters or {}))
    return {
        "status": "available",
        "count": 1,
        "filters": dict(filters or {}),
        "summary": {
            "total": 1,
            "by_status": [{"key": "status:failed", "count": 1}],
            "by_channel": [{"key": "channel:secondary", "count": 1}],
            "latency_ms": {"avg_first": None, "avg_total": 240.0, "max_total": 240.0},
        },
        "records": [
            {
                "status": "failed", "kind": "model", "channel_id": "secondary",
                "protocol": "openai_chat", "requested_model": "req2",
                "total_duration_ms": 240.0,
            }
        ],
    }

window = PetConsoleWindow(callbacks={
    "api_audit": lambda: {
        "status": "available",
        "count": 2,
        "summary": {
            "total": 2,
            "by_status": [
                {"key": "status:completed", "count": 1},
                {"key": "status:failed", "count": 1},
            ],
            "by_channel": [
                {"key": "channel:primary", "count": 1},
                {"key": "channel:secondary", "count": 1},
            ],
            "latency_ms": {"avg_first": None, "avg_total": None, "max_total": None},
        },
        "records": [
            {"status": "completed", "channel_id": "primary", "kind": "model"},
            {"status": "failed", "channel_id": "secondary", "kind": "model"},
        ],
    },
    "audit_refresh": audit_refresh,
})
window.findChild(QPushButton, "apiAuditButton").click()
status_combo = window.findChild(QComboBox, "auditStatusFilter")
channel_combo = window.findChild(QComboBox, "auditChannelFilter")
table = window.findChild(QTableWidget, "diagnosticAuditTable")
assert status_combo.itemData(status_combo.count() - 1) == "failed"
status_combo.setCurrentIndex(status_combo.findData("failed"))
channel_combo.setCurrentIndex(channel_combo.findData("secondary"))
table = window.findChild(QTableWidget, "diagnosticAuditTable")
assert table.rowCount() == 1
assert table.item(0, 0).text() == "failed"
assert table.item(0, 2).text() == "secondary"
assert seen_filters[-1] == {"status": "failed", "channel_id": "secondary"}
# 所有已见过滤调用都不能携带未知或可疑键
for received_filters in seen_filters:
    unknown_keys = set(received_filters) - {"status", "channel_id"}
    assert not unknown_keys, f"unexpected filter keys: {sorted(unknown_keys)}"
window.shutdown()
app.processEvents()
print("console-api-audit-filter-refresh-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "console-api-audit-filter-refresh-ok" in result.stdout


def test_api_audit_filter_sanitizer_and_repository_alignment() -> None:
    """白名单过滤键进入仓库层后行数对齐；未知/异常值被拒绝或归一。"""

    from db import ApiCallAuditRepository, Database
    from gui.qt6.app import _sanitize_api_audit_filters

    db = Database(":memory:")
    repository = ApiCallAuditRepository(db)
    for index, (status, channel_id) in enumerate(
        (
            ("completed", "primary"),
            ("completed", "primary"),
            ("failed", "secondary"),
            ("failed", "secondary"),
        )
    ):
        handle = repository.start(
            kind="model",
            status=status,
            channel_id=channel_id,
            provider="openai",
            started_at=float(index + 1),
        )
        handle.finish(status=status, completed_at=float(index + 1) + 0.25)
    assert repository.count() == 4
    assert repository.count(status="completed") == 2
    assert repository.count(channel_id="secondary") == 2

    unknown_like = {"session_id": "x", "limit": 999, "started_after": 0}
    cleaned = _sanitize_api_audit_filters(unknown_like)
    assert cleaned == {}
    polluted = {"status": "completed", "channel_id": "primary", "api_key": "leak"}
    cleaned = _sanitize_api_audit_filters(polluted)
    assert cleaned == {"status": "completed", "channel_id": "primary"}
    completed_records = repository.list_recent(limit=50, **cleaned)
    assert completed_records and all(item.status == "completed" for item in completed_records)
    assert repository.count(status="completed", channel_id="primary") == 2
