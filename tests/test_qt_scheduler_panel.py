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
        timeout=20,
        check=False,
    )


def test_scheduler_panel_preserves_unknown_entries_and_uses_safe_labels(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
values = {
    "enabled": True,
    "poll_seconds": 0.5,
    "custom_flag": {"keep": True},
    "tasks": [
        {
            "task_id": "task-1",
            "name": "眨眼",
            "expression": "every:5m",
            "action": {"identity": "pet:play_motion", "arguments": {"name": "blink"}},
            "owner": "config",
        },
        {
            "name": "高级任务",
            "expression": "every:1h",
            "action": {"identity": "custom:thing", "arguments": {"secret": "hidden"}},
            "unknown_field": {"keep": "exact"},
        },
        {
            "name": "带情绪语音",
            "expression": "every:2m",
            "action": {
                "identity": "pet:speak",
                "arguments": {"text": "你好", "mood": "happy"},
            },
        },
    ],
    "triggers": [
        {
            "trigger_id": "trigger-1",
            "event_name": "window_changed",
            "action": {"identity": "pet:play_motion", "arguments": {"name": "wave"}},
            "debounce_seconds": 30,
        },
        {
            "trigger_id": "trigger-advanced",
            "event_name": "future_event",
            "action": {"identity": "custom:thing", "arguments": {"x": 1}},
        },
    ],
}
panel = SchedulerPanel(values)
assert panel.task_list.count() == 1
assert panel.advanced_task_list.count() == 2
assert panel.trigger_list.count() == 1
assert panel.advanced_trigger_list.count() == 1
assert "高级任务" in panel.advanced_task_list.item(0).text()
assert "secret" not in panel.advanced_task_list.item(0).text()
assert panel.values() == values
panel.advanced_task_list.setCurrentRow(0)
panel.advanced_task_delete.click()
assert len(panel.values()["tasks"]) == 2
assert panel.values()["tasks"][1]["name"] == "带情绪语音"
panel.close()
print("scheduler-preserve-unknown-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "scheduler-preserve-unknown-ok" in result.stdout


def test_scheduler_panel_empty_lists_explain_next_click_and_disable_delete(tmp_path: Path) -> None:
    """空调度页应给出下一步提示，删除按钮不应伪装成可用。"""

    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
panel = SchedulerPanel({"enabled": True, "tasks": [], "triggers": []})
panel.show()
app.processEvents()
assert all(button.isVisible() for button in panel._task_interval_preset_buttons)
assert all(button is not None for button in panel._trigger_debounce_preset_buttons)
assert not panel._task_interval_host.isVisible()
assert not panel._trigger_debounce_host.isVisible()
panel._task_interval_preset_buttons[1].click()
panel.mode_tabs.setCurrentIndex(1)
app.processEvents()
assert all(button.isVisible() for button in panel._trigger_debounce_preset_buttons)
panel._trigger_debounce_preset_buttons[2].click()
assert panel.task_interval_spin.value() == 5.0
assert panel.task_interval_unit.currentData() == "m"
assert panel.trigger_debounce_spin.value() == 15.0
assert panel.task_list.count() == 1
assert "新增定时任务" in panel.task_list.item(0).text()
assert not panel.task_delete_button.isEnabled()
assert panel.trigger_list.count() == 1
assert "新增事件触发" in panel.trigger_list.item(0).text()
assert not panel.trigger_delete_button.isEnabled()
assert panel._advanced_card is not None and not panel._advanced_card.isVisible()
panel.close()
app.processEvents()
print("scheduler-empty-guidance-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    source_root = Path(__file__).resolve().parents[1] / "src"
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
    assert "scheduler-empty-guidance-ok" in result.stdout


def test_scheduler_panel_click_editors_build_exact_task_and_trigger_schema(tmp_path: Path) -> None:
    script = r"""
from pathlib import Path
from PySide6.QtWidgets import QApplication
from app.runtime import validate_runtime_configuration
from config.loader import LoadedConfiguration
from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
panel = SchedulerPanel({"enabled": False, "tasks": [], "triggers": []})
changes = []
panel.changed.connect(changes.append)
panel._new_task()
panel.task_name_edit.setText("每分钟挥手")
panel.task_interval_spin.setValue(1)
panel.task_interval_unit.setCurrentIndex(panel.task_interval_unit.findData("m"))
panel.task_action_combo.setCurrentIndex(panel.task_action_combo.findData("wave"))
panel.task_save_button.click()
task = panel.values()["tasks"][0]
assert task["name"] == "每分钟挥手"
assert task["expression"] == "every:1m"
assert task["action"] == {
    "identity": "pet:play_motion", "arguments": {"name": "wave"}
}
panel._new_task()
panel.task_name_edit.setText("说一句")
panel.task_action_combo.setCurrentIndex(panel.task_action_combo.findData("speak"))
panel.task_speak_edit.setText("你好，主人")
panel.task_save_button.click()
assert panel.values()["tasks"][1]["action"] == {
    "identity": "pet:speak", "arguments": {"text": "你好，主人"}
}
panel._new_trigger()
panel.trigger_event_combo.setCurrentIndex(panel.trigger_event_combo.findData("window_changed"))
assert "开启 watcher" in panel.trigger_hint.text()
assert "权限" in panel.trigger_hint.text()
panel.trigger_action_combo.setCurrentIndex(panel.trigger_action_combo.findData("happy"))
panel.trigger_debounce_spin.setValue(15.5)
panel.trigger_match_mode_combo.setCurrentIndex(panel.trigger_match_mode_combo.findData("contains"))
panel.trigger_process_edit.setText("code")
panel.trigger_title_edit.setText("main.py")
panel.trigger_duration_spin.setValue(30)
panel.trigger_require_active_check.setChecked(True)
panel.trigger_save_button.click()
trigger = panel.values()["triggers"][0]
assert trigger["event_name"] == "window_changed"
assert trigger["debounce_seconds"] == 15.5
assert trigger["action"] == {
    "identity": "pet:set_expression", "arguments": {"name": "happy"}
}
assert trigger["conditions"] == {
    "process_name": {"mode": "contains", "value": "code"},
    "title": {"mode": "contains", "value": "main.py"},
    "min_duration_seconds": 30.0,
    "require_user_active": True,
}
validate_runtime_configuration(
    LoadedConfiguration(Path("config.yaml"), {"scheduler": panel.values()})
)
assert changes
panel.trigger_list.setCurrentRow(0)
panel.trigger_delete_button.click()
assert panel.values()["triggers"] == []
panel.close()
print("scheduler-click-schema-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "scheduler-click-schema-ok" in result.stdout


def test_scheduler_panel_builds_daily_wall_clock_task(tmp_path: Path) -> None:
    """点击式调度器应能配置例如凌晨 3 点的本地每日提醒。"""

    script = r"""
from pathlib import Path
from PySide6.QtCore import QTime
from PySide6.QtWidgets import QApplication
from app.runtime import validate_runtime_configuration
from config.loader import LoadedConfiguration
from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
panel = SchedulerPanel({"enabled": True, "tasks": [], "triggers": []})
panel.activity_system_idle_combo.setCurrentIndex(
    panel.activity_system_idle_combo.findData("x11")
)
panel.activity_system_idle_threshold_spin.setValue(45.0)
activity = panel.values()["activity"]
assert activity["system_idle_provider"] == "x11"
assert activity["system_idle_threshold_seconds"] == 45.0
panel._new_task()
panel.task_name_edit.setText("凌晨提醒")
panel.task_schedule_mode.setCurrentIndex(panel.task_schedule_mode.findData("daily"))
panel.task_daily_time.setTime(QTime(3, 0))
panel.task_action_combo.setCurrentIndex(panel.task_action_combo.findData("speak"))
panel.task_speak_edit.setText("已经很晚了，请注意休息")
panel.task_save_button.click()
task = panel.values()["tasks"][0]
assert task["expression"] == "daily:03:00"
assert panel.task_list.count() == 1
validate_runtime_configuration(
    LoadedConfiguration(Path("config.yaml"), {"scheduler": panel.values()})
)
restored = SchedulerPanel(panel.values())
restored.show()
app.processEvents()
restored.task_list.setCurrentRow(0)
assert restored.task_schedule_mode.currentData() == "daily"
assert restored.task_daily_time.time().toString("HH:mm") == "03:00"
panel.close()
restored.close()
app.processEvents()
print("scheduler-daily-wall-clock-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "scheduler-daily-wall-clock-ok" in result.stdout


def test_scheduler_panel_exposes_windows_idle_provider(tmp_path: Path) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
panel = SchedulerPanel({"enabled": True, "tasks": [], "triggers": []})
index = panel.activity_system_idle_combo.findData("windows")
assert index >= 0
panel.activity_system_idle_combo.setCurrentIndex(index)
assert panel.values()["activity"]["system_idle_provider"] == "windows"
"""
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr or result.stdout


def test_scheduler_panel_exposes_activity_events_and_active_only_tasks(tmp_path: Path) -> None:
    """普通配置页应能设置活跃条件并选择 idle/user_interaction 事件。"""

    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
panel = SchedulerPanel({"enabled": True, "tasks": [], "triggers": []})
panel._new_task()
panel.task_name_edit.setText("活跃提醒")
panel.task_require_active.setChecked(True)
panel.task_save_button.click()
task = panel.values()["tasks"][0]
assert task["metadata"] == {"require_user_active": True}
panel._new_trigger()
panel.trigger_event_combo.setCurrentIndex(panel.trigger_event_combo.findData("idle"))
assert "空闲" in panel.trigger_hint.text()
panel.trigger_save_button.click()
assert panel.values()["triggers"][0]["event_name"] == "idle"
panel._new_trigger()
panel.trigger_event_combo.setCurrentIndex(panel.trigger_event_combo.findData("user_interaction"))
assert "互动" in panel.trigger_hint.text()
panel.close()
print("scheduler-activity-panel-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "scheduler-activity-panel-ok" in result.stdout


def test_configuration_panel_integrates_scheduler_editor_and_hides_yaml_by_default(
    tmp_path: Path,
) -> None:
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({
    "scheduler": {
        "enabled": True,
        "tasks": [{
            "name": "眨眼",
            "expression": "every:5m",
            "action": {"identity": "pet:play_motion", "arguments": {"name": "blink"}},
        }],
        "triggers": [],
    }
})
panel.show()
app.processEvents()
panel.navigation.setCurrentRow(panel.sections.index("scheduler"))
assert panel.scheduler_panel is not None
assert panel.scheduler_panel.isVisible()
assert panel.complex_editors["scheduler"][("tasks",)].isVisible() is False
panel.scheduler_panel._new_task()
panel.scheduler_panel.task_name_edit.setText("新任务")
panel.scheduler_panel.task_save_button.click()
assert panel.draft_values()["scheduler"]["tasks"][1]["name"] == "新任务"
panel.advanced_toggle.click()
app.processEvents()
assert panel.scheduler_panel.isVisible() is False
assert panel.complex_editors["scheduler"][("tasks",)].isVisible()
panel.close()
app.processEvents()
print("scheduler-config-integration-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "scheduler-config-integration-ok" in result.stdout


def test_scheduler_panel_reflows_activity_controls_on_narrow_width(tmp_path: Path) -> None:
    """窄屏调度页的活跃设置不能把宿主配置页横向撑开。"""

    script = r"""
from PySide6.QtWidgets import QApplication

from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
panel = SchedulerPanel(
    {
        "enabled": True,
        "activity": {
            "enabled": True,
            "idle_seconds": 300,
            "system_idle_provider": "x11",
            "system_idle_threshold_seconds": 300,
        },
        "tasks": [],
        "triggers": [],
    }
)
panel.resize(420, 700)
panel.show()
app.processEvents()
assert panel.width() <= 420
assert panel.minimumSizeHint().width() <= 420
assert panel.task_new_button.property("role") == "schedulerPrimary"
assert 'QPushButton[role="schedulerPrimary"]' in panel.styleSheet()
activity = panel.findChild(object, "schedulerActivityCard")
assert activity is not None
for name in (
    "schedulerActivityEnabled",
    "schedulerActivityIdleSeconds",
    "schedulerSystemIdleProvider",
    "schedulerSystemIdleThresholdSeconds",
):
    control = panel.findChild(object, name)
    assert control is not None and control.isVisible()
    right = control.mapTo(panel, control.rect().bottomRight()).x()
    assert right <= panel.rect().right()
panel.close()
app.processEvents()
print("scheduler-narrow-reflow-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "scheduler-narrow-reflow-ok" in result.stdout


def test_scheduler_panel_uses_fancyui_master_detail_layout(tmp_path: Path) -> None:
    """任务和触发器使用 FancyUI 主从双栏，并在窄屏可靠堆叠。"""

    script = r"""
from PySide6.QtWidgets import QApplication, QBoxLayout, QFrame
from gui.qt6.fancyui import FancyCard, FancyInfoBar, FancyStyleController
from gui.qt6.scheduler_panel import SchedulerPanel

app = QApplication.instance() or QApplication([])
panel = SchedulerPanel({"enabled": True, "tasks": [], "triggers": []})
panel.resize(1040, 760)
panel.show()
app.processEvents()
assert panel.mode_tabs.count() == 2
assert isinstance(panel.findChild(QFrame, "schedulerHeader"), FancyCard)
assert isinstance(panel.findChild(QFrame, "schedulerActivityCard"), FancyCard)
assert isinstance(panel.findChild(QFrame, "schedulerInfoBar"), FancyInfoBar)
assert isinstance(panel._fancy_style_controller, FancyStyleController)
assert panel.property("responsiveMode") == "split"
assert panel._task_master_detail_layout.direction() == QBoxLayout.Direction.LeftToRight
panel._new_task()
panel.task_name_edit.setText("测试任务")
panel.task_save_button.click()
assert panel.status_info.severity() == "warning"
panel.resize(620, 760)
app.processEvents()
assert panel.property("responsiveMode") == "stacked"
assert panel._task_master_detail_layout.direction() == QBoxLayout.Direction.TopToBottom
assert panel._trigger_master_detail_layout.direction() == QBoxLayout.Direction.TopToBottom
assert panel._task_master is not None and panel._task_master.maximumWidth() == 16777215
assert "#ff9dbe" not in panel.styleSheet().lower()
panel.close()
app.processEvents()
print("scheduler-fancyui-master-detail-ok")
"""
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    result = _run_qt_script(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert "scheduler-fancyui-master-detail-ok" in result.stdout
