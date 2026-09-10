from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


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
        timeout=30,
        check=False,
    )


def test_qt_console_visual_contract_generates_wide_and_narrow_captures(
    tmp_path: Path,
) -> None:
    """离屏抓取控制台宽窄布局，守住语义令牌和紧凑导航可读性。"""

    script = r"""
from pathlib import Path

from PySide6.QtWidgets import QApplication, QScrollArea
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration={"llm": {"channels": []}})
window.resize(980, 760)
window.show()
app.processEvents()
wide = Path("/tmp/meapet-console-visual-wide.png")
assert window.grab().save(str(wide))
assert wide.stat().st_size > 0
assert "#ff9dbe" not in window.styleSheet().lower()
assert "#60CDFF" in window.styleSheet()

window.resize(640, 500)
app.processEvents()
navigation = window._console_tabs
assert navigation.isCompact()
assert navigation.tabBar().tabSizeHint(0).width() == 132
assert navigation.tabText(0) == "概览"
scroll = window.findChild(QScrollArea, "consoleOverviewScroll")
assert scroll is not None
assert scroll.viewport().width() > 0

# 从非紧凑模式跨过 720px 断点时，导航栏必须立即采用 132px
# 实际宽度；否则正文会额外损失 56px，窄窗首屏会被错误压缩。
window.resize(719, 600)
app.processEvents()
assert navigation.isCompact()
assert navigation.tabBar().width() == 132
assert scroll.viewport().width() >= 553
narrow = Path("/tmp/meapet-console-visual-narrow.png")
assert window.grab().save(str(narrow))
assert narrow.stat().st_size > 0

window.resize(420, 500)
app.processEvents()
assert window._overview_action_columns == 1
assert all(button.width() > 0 for button in window._overview_action_items)
window.shutdown()
app.processEvents()
print("qt-console-visual-contract-ok")
"""
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "qt-console-visual-contract-ok" in result.stdout


def test_qt_console_adapts_minimum_size_after_show_on_small_screen(
    tmp_path: Path,
) -> None:
    """独立打开控制台时，低分辨率屏幕不应被 420×500 最小尺寸卡住。"""

    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration={"llm": {"channels": []}})
window.show()
app.processEvents()
screen = window.screen() or app.primaryScreen()
assert screen is not None
area = screen.availableGeometry()
assert window.minimumWidth() <= max(1, area.width() - 16)
assert window.minimumHeight() <= max(1, area.height() - 16)
window.resize(min(375, area.width()), min(420, area.height()))
app.processEvents()
# offscreen Qt 通常提供 800×600 的虚拟屏幕；此时 420×500 仍是正常的
# 首选最小尺寸，先显式切换到紧凑约束验证同一布局路径。真正小屏下
# ``showEvent`` 已会在这里之前自动放宽约束（由 Xvfb smoke 覆盖）。
if area.width() >= 436 and area.height() >= 516:
    window.setMinimumSize(1, 1)
    window.resize(min(375, area.width()), min(420, area.height()))
    app.processEvents()
assert window.width() <= min(375, area.width())
assert window.height() <= min(420, area.height())
assert bool(window.property("compactLayout")) is True
window.shutdown()
app.processEvents()
print("qt-console-small-screen-minimum-ok")
"""
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "qt-console-small-screen-minimum-ok" in result.stdout


def test_configuration_and_scheduler_visual_contracts_fit_a_normal_viewport(
    tmp_path: Path,
) -> None:
    """配置与调度页面在正常视口内保持首屏尺寸，不被隐藏表单撑开。"""

    script = r"""
from pathlib import Path

from PySide6.QtWidgets import QApplication, QScrollArea
from gui.qt6.config_panel import ConfigurationPanel
from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
configuration = ConfigurationPanel({"tts": {"enabled": True}, "scheduler": {}})
configuration.resize(1040, 760)
configuration.show()
app.processEvents()
assert configuration.size().toTuple() == (1040, 760)
assert configuration.property("responsiveMode") == "split"
assert "#ff9dbe" not in configuration.styleSheet().lower()
config_capture = Path("/tmp/meapet-config-visual-wide.png")
assert configuration.grab().save(str(config_capture))
assert config_capture.stat().st_size > 0

scheduler = SchedulerPanel({"tasks": [], "triggers": []})
scheduler.resize(1040, 760)
scheduler.show()
app.processEvents()
assert scheduler.size().toTuple() == (1040, 760)
assert scheduler.minimumSizeHint().height() <= 720
mode_scroll = scheduler.findChild(QScrollArea, "schedulerModeScroll")
assert mode_scroll is not None and mode_scroll.isVisible()
assert "#ff9dbe" not in scheduler.styleSheet().lower()
scheduler_capture = Path("/tmp/meapet-scheduler-visual-wide.png")
assert scheduler.grab().save(str(scheduler_capture))
assert scheduler_capture.stat().st_size > 0
configuration.close()
scheduler.close()
app.processEvents()
print("qt-config-scheduler-visual-contract-ok")
"""
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "qt-config-scheduler-visual-contract-ok" in result.stdout
