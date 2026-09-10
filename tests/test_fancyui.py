from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

from gui.qt6.fancyui import build_fancy_stylesheet
from gui.qt6.md3 import DARK_MD3_THEME

ROOT = Path(__file__).resolve().parents[1]


def _run_qt(code: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_fancy_stylesheet_uses_the_shared_md3_source_of_truth() -> None:
    stylesheet = build_fancy_stylesheet(DARK_MD3_THEME)

    assert 'QFrame[fancyRole="card"]' in stylesheet
    assert 'QFrame[fancyRole="command-bar"]' in stylesheet
    assert 'QFrame[fancyRole="info-bar"][fancySeverity="error"]' in stylesheet
    assert DARK_MD3_THEME.colors.primary in stylesheet
    assert "rgba(39, 39, 39, 230)" in stylesheet
    assert "rgba(" in stylesheet


def test_fancyui_public_widgets_complete_one_interactive_user_story() -> None:
    code = r"""
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QTabWidget, QVBoxLayout, QWidget
from gui.qt6 import (
    FancyCard,
    FancyCommandBar,
    FancyInfoBar,
    FancyNavigationView,
    FancyStyleController,
    fancy_icon,
)
from gui.qt6.md3 import MD3Theme

app = QApplication.instance() or QApplication([])
root = QWidget()
layout = QVBoxLayout(root)
controller = FancyStyleController()
controller.attach(root)

card = FancyCard("桌宠状态", "当前服务正常", icon="success", interactive=True)
clicked = []
card.clicked.connect(lambda: clicked.append(True))
card.addWidget(QLabel("实时状态"))
controller.register(card)
layout.addWidget(card)

navigation = FancyNavigationView()
assert issubclass(FancyNavigationView, QTabWidget)
assert navigation.addPage(QLabel("总览页面"), "总览", "home") == 0
navigation.addPage(QLabel("设置页面"), "设置", "file")
navigation.setCompact(True)
assert navigation.isCompact() is True
assert navigation.tabToolTip(0) == "总览"
navigation.setNavigationWidth(208)
assert navigation.navigationWidth() == 208
controller.register(navigation)
layout.addWidget(navigation)

actions = []
info = FancyInfoBar("模型已连接", "流式接口可以使用", severity="success")
info.setAction("查看", lambda: actions.append("info"))
controller.register(info)
layout.addWidget(info)

command_bar = FancyCommandBar()
save_action = command_bar.addCommand(
    "保存", lambda: actions.append("save"), icon="save", primary=True
)
refresh_action = command_bar.addCommand(
    "刷新", lambda: actions.append("refresh"), icon="refresh"
)
controller.register(command_bar)
layout.addWidget(command_bar)

root.resize(760, 620)
root.show()
app.processEvents()
QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=card.rect().center())
QTest.mouseClick(info._action_button, Qt.MouseButton.LeftButton)
save_action.trigger()
refresh_action.trigger()
app.processEvents()
assert clicked == [True]
assert actions == ["info", "save", "refresh"]
assert info.severity() == "success"
assert info.isClosable() is True
assert card.title() == "桌宠状态"
assert card.subtitle() == "当前服务正常"
assert len(command_bar.commands()) == 2
assert not fancy_icon("not-in-platform-theme").isNull()

light = MD3Theme.from_mapping({"mode": "light", "motion": {"reduced_motion": True}})
controller.updateTheme(light)
assert controller.theme.mode.value == "light"
assert controller.motionDuration() == 0
assert card._fancy_theme is light
info.reveal()
info.dismiss()
assert info.isHidden()
"""
    result = _run_qt(code)

    assert result.returncode == 0, result.stderr


def test_fancy_command_bar_moves_secondary_actions_to_overflow() -> None:
    code = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.fancyui import FancyCommandBar

app = QApplication.instance() or QApplication([])
bar = FancyCommandBar()
bar.addCommand("主要操作", primary=True)
bar.addCommand("重新加载")
bar.addCommand("导出诊断")
bar.resize(150, 48)
bar.show()
app.processEvents()
bar._sync_overflow()
assert bar._commands[0][1].isVisible()
assert not bar._commands[1][1].isVisible()
assert not bar._commands[2][1].isVisible()
assert bar._overflow_button.isVisible()
assert len(bar._overflow_menu.actions()) == 2
bar.clearCommands()
assert bar.commands() == ()
"""
    result = _run_qt(code)

    assert result.returncode == 0, result.stderr


def test_fancy_navigation_keyboard_skips_hidden_and_disabled_pages() -> None:
    code = r"""
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel
from gui.qt6.fancyui import FancyNavigationView

app = QApplication.instance() or QApplication([])
navigation = FancyNavigationView()
for index in range(5):
    navigation.addPage(QLabel(f"页面 {index}"), f"页面 {index}", "file")
navigation.setTabEnabled(1, False)
navigation.setTabVisible(3, False)
navigation.resize(640, 420)
navigation.show()
app.processEvents()
bar = navigation.tabBar()
bar.setFocus()
navigation.setCurrentIndex(2)

QTest.keyClick(bar, Qt.Key.Key_Down)
assert navigation.currentIndex() == 4
QTest.keyClick(bar, Qt.Key.Key_Down)
assert navigation.currentIndex() == 4
QTest.keyClick(bar, Qt.Key.Key_Up)
assert navigation.currentIndex() == 2
QTest.keyClick(bar, Qt.Key.Key_Left)
assert navigation.currentIndex() == 0
QTest.keyClick(bar, Qt.Key.Key_Right)
assert navigation.currentIndex() == 2
QTest.keyClick(bar, Qt.Key.Key_Home)
assert navigation.currentIndex() == 0
QTest.keyClick(bar, Qt.Key.Key_End)
assert navigation.currentIndex() == 4
"""
    result = _run_qt(code)

    assert result.returncode == 0, result.stderr


def test_fancyui_rejects_invalid_public_values() -> None:
    code = r"""
from PySide6.QtWidgets import QApplication, QLabel
from gui.qt6.fancyui import FancyInfoBar, FancyNavigationView, FancyStyleController, fancy_icon

app = QApplication.instance() or QApplication([])
checks = 0
for call in (
    lambda: fancy_icon("info", size=2),
    lambda: FancyInfoBar(severity="notice"),
    lambda: FancyInfoBar(duration_ms=-1),
    lambda: FancyNavigationView(navigation_width=80),
    lambda: FancyStyleController().register(QLabel("ok").text()),
):
    try:
        call()
    except (TypeError, ValueError):
        checks += 1
assert checks == 5
"""
    result = _run_qt(code)

    assert result.returncode == 0, result.stderr


def test_fancy_style_controller_coalesces_identical_root_style_passes() -> None:
    code = r"""
from PySide6.QtWidgets import QApplication, QWidget
from gui.qt6.fancyui import FancyStyleController

class CountingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.style_calls = 0

    def setStyleSheet(self, stylesheet):
        self.style_calls += 1
        super().setStyleSheet(stylesheet)

app = QApplication.instance() or QApplication([])
root = CountingWidget()
controller = FancyStyleController()
component_rules = 'QWidget#screenRoot { background: transparent; }'
controller.attach(root, extra_stylesheet=component_rules)
assert root.style_calls == 1
revision = controller.revision
controller.updateTheme(controller.theme, extra_stylesheet=component_rules)
assert root.style_calls == 1
assert controller.revision == revision
controller.updateTheme(
    controller.theme,
    extra_stylesheet=component_rules + ' QLabel { padding: 1px; }',
)
assert root.style_calls == 2
assert controller.revision == revision + 1
"""
    result = _run_qt(code)

    assert result.returncode == 0, result.stderr


def test_fancyui_keeps_exact_upstream_provenance() -> None:
    source = (ROOT / "src/gui/qt6/fancyui.py").read_text(encoding="utf-8")
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    license_bytes = (ROOT / "docs/licenses/QWidget-FancyUI-GPL-3.0.txt").read_bytes()

    assert "SPDX-License-Identifier: GPL-3.0-only" in source
    assert "e6b837a2f668fc0beed9463b6dc48d92f787f272" in source
    assert "Historical PySide6 reference: a03ae9c9e8ad98d79ded93ebc00a242602f11112" in source
    assert "e6b837a2f668fc0beed9463b6dc48d92f787f272" in notices
    assert "a03ae9c9e8ad98d79ded93ebc00a242602f11112" in notices
    assert hashlib.sha256(license_bytes).hexdigest() == (
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
    )
