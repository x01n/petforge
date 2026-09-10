from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_qt_console_configuration_panel_edits_all_domains_and_preserves_secret(
    tmp_path: Path,
) -> None:
    """无头 Qt 应能编辑业务域 YAML，并通过显式确认提交脱敏草稿。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
saved = []
values = {
    "llm": {"channels": [{"id": "primary", "api_key": "secret", "model": "old"}]},
    "tts": {"enabled": False, "language": "zh"},
    "logging": {"level": "INFO", "console": {"color": "auto"}},
    "tools": {"permissions": {"approval_ttl_seconds": 90}},
    "scheduler": {"tasks": [], "triggers": []},
    "rendering": {"backend": "auto", "sprite_scale": 0.6},
    "ui": {"stream": {"show_murmur": True}},
}
window = PetConsoleWindow(
    configuration=values,
    callbacks={"config_save": lambda draft: saved.append(draft) or {"status": "saved"}},
)
panel = window.configuration_panel
assert panel is not None
assert panel.navigation.count() == 17
assert panel.minimumWidth() <= 520
assert panel.findChild(type(panel._auto_save_toggle), "configAutoSaveToggle").isChecked()
assert panel.findChild(type(panel._status), "configEyebrow").text() == "MEAPET  ·  工作区设置"
panel.navigation.setCurrentRow(2)
editor = panel._section_editors["llm"]
editor.setPlainText("channels:\n  - id: primary\n    api_key: '***'\n    model: newer\n")
assert panel.save(confirmed=True)
assert saved[0]["llm"]["channels"][0]["api_key"] == "secret"
assert saved[0]["llm"]["channels"][0]["model"] == "newer"
panel.navigation.setCurrentRow(16)
assert "llm:" in panel.yaml_editor.toPlainText()
window.shutdown()
app.processEvents()
print("qt-console-config-ok")
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
    assert "qt-console-config-ok" in result.stdout


def test_qt_console_configuration_path_loads_raw_secret_before_atomic_save(tmp_path: Path) -> None:
    """仅提供配置路径时，控制台仍应保留未修改的密钥字段。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    path = tmp_path / "config.yaml"
    path.write_text(
        "llm:\n  channels:\n    - id: primary\n      api_key: ${MEAPET_TEST_KEY}\n"
        "      model: old\n",
        encoding="utf-8",
    )
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration_path={str(path)!r})
panel = window.configuration_panel
assert panel is not None
panel.navigation.setCurrentRow(2)
editor = panel._section_editors["llm"]
editor.setPlainText("channels:\\n  - id: primary\\n    api_key: '***'\\n    model: changed\\n")
assert panel.save(confirmed=True)
saved_text = open({str(path)!r}, encoding="utf-8").read()
assert "${{MEAPET_TEST_KEY}}" in saved_text
assert "changed" in saved_text
window.shutdown()
app.processEvents()
print("qt-console-config-file-ok")
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
    assert "qt-console-config-file-ok" in result.stdout


def test_qt_console_direct_file_save_reports_missing_channel_credentials(tmp_path: Path) -> None:
    """无宿主回调时，密钥引用可落盘但必须明确标记待补凭据。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    path = tmp_path / "config.yaml"
    path.write_text(
        "llm:\n  channels:\n    - id: primary\n      protocol: openai_chat\n"
        "      base_url: https://gateway.example.invalid/v1\n"
        "      model: demo\n      api_key: ${MEAPET_DIRECT_MISSING_KEY}\n",
        encoding="utf-8",
    )
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration_path={str(path)!r})
panel = window.configuration_panel
assert panel is not None
panel.navigation.setCurrentRow(panel.sections.index("llm"))
panel._section_editors["llm"].setPlainText(
    "channels:\\n  - id: primary\\n    protocol: openai_chat\\n"
    "    base_url: https://gateway.example.invalid/v1\\n"
    "    model: changed\\n    api_key: ${{MEAPET_DIRECT_MISSING_KEY}}\\n"
)
assert panel.save(confirmed=True)
assert "待补密钥" in panel.status_label.text()
window.shutdown()
app.processEvents()
print("qt-console-direct-missing-key-ok")
"""
    environment = os.environ.copy()
    environment.pop("MEAPET_DIRECT_MISSING_KEY", None)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = f"{source_root}{os.pathsep}{environment.get('PYTHONPATH', '')}"
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
    assert "qt-console-direct-missing-key-ok" in result.stdout


def test_configuration_overview_distinguishes_incomplete_model_channel(tmp_path: Path) -> None:
    """总览不能把缺少地址/模型的渠道误显示为可用。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({
    "llm": {
        "channels": [{"id": "primary", "protocol": "openai_chat", "model": ""}],
    },
})
panel.show()
app.processEvents()
status = panel._overview_status_labels["llm"].text()
assert "待补齐" in status
assert "可用模型渠道" not in status
panel.close()
app.processEvents()
print("config-model-readiness-summary-ok")
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
    assert "config-model-readiness-summary-ok" in result.stdout


def test_memory_configuration_controls_reflow_on_narrow_panel(tmp_path: Path) -> None:
    """记忆核心参数在窄屏仍可见、可编辑且不产生横向溢出。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication, QDoubleSpinBox, QSpinBox, QWidget
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({"memory": {"enabled": True, "recall_limit": 7}})
panel.setMinimumSize(1, 1)
panel.resize(320, 420)
panel.show()
app.processEvents()
panel.navigation.setCurrentRow(panel.sections.index("memory"))
app.processEvents()
for path in (("recall_limit",), ("context_max_chars",), ("decay_days",)):
    control = panel._friendly_controls["memory"][path]
    assert control.isVisible()
    assert isinstance(control, (QSpinBox, QDoubleSpinBox))
    bottom_right = control.mapTo(panel, control.rect().bottomRight())
    assert bottom_right.x() <= panel.rect().right()
panel._friendly_controls["memory"][("recall_limit",)].setValue(11)
assert panel.draft_values()["memory"]["recall_limit"] == 11
panel.close()
app.processEvents()
print("memory-config-narrow-ok")
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
    assert "memory-config-narrow-ok" in result.stdout


def test_qt_configuration_panel_exposes_modern_forms_and_complex_cards(tmp_path: Path) -> None:
    """分域表单、搜索导航和复杂 YAML 卡片应可在无头 Qt 中联动。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
)
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
values = {
    "app": {"name": "MeaPet"},
    "llm": {"channels": [{"id": "primary", "api_key": "secret", "model": "old"}]},
    "tts": {
        "enabled": False,
        "backend": "text_only",
        "language": "zh",
        "profiles": {"zh-main": {}, "jp-main": {"languages": ["ja"]}},
    },
    "logging": {"level": "INFO", "console": {"enabled": True, "color": "auto"}},
    "memory": {
        "enabled": True,
        "recall_limit": 7,
        "context_max_chars": 6000,
        "context_memory_item_chars": 500,
        "recent_exchange_limit": 10,
        "max_memories": 2000,
        "decay_days": 7,
        "prune_days": 30,
        "prune_importance_floor": 1,
        "consolidation_enabled": True,
        "consolidation_similarity": 0.85,
        "max_consolidation_memories": 200,
        "summarize_every_n": 20,
        "summary_chat_limit": 20,
        "exchange_min_chars": 20,
        "exchange_importance": 2,
        "exchange_decay_factor": 0.8,
        "semantic": {
            "enabled": False,
            "provider": "sentence_transformers",
            "model_path": "",
            "model_id": "",
            "model_revision": "",
            "dimension": 384,
            "batch_size": 16,
            "timeout_seconds": 5.0,
            "startup_timeout_seconds": 60.0,
            "shutdown_timeout_seconds": 2.0,
            "max_text_chars": 10000,
            "max_text_bytes": 40000,
            "device": "cpu",
            "backend": "torch",
            "index_dir": "",
            "hnsw_m": 16,
            "hnsw_ef_construction": 200,
            "hnsw_ef_search": 64,
        },
    },
    "tools": {"permissions": {"approval_ttl_seconds": 90}},
    "mcp": {"enabled": False, "servers": []},
    "watcher": {"enabled": False},
    "scheduler": {"enabled": True, "tasks": [], "triggers": []},
    "behavior": {"enabled": False, "probability": 0.35, "actions": []},
    "rendering": {"sprite_scale": 0.6},
    "storage": {"database": "pet.sqlite3"},
    "ui": {"always_on_top": True, "hotkeys": {"enabled": True, "bindings": []}},
}
panel = ConfigurationPanel(values)
panel.show()
app.processEvents()
assert panel.sections == (
    "overview", "app", "llm", "tts", "asr", "logging", "memory", "tools", "mcp", "watcher",
    "scheduler", "behavior", "proactive", "rendering", "storage", "ui", "advanced",
)
assert panel.navigation.count() == 17
assert isinstance(panel.form_controls["tts"][("enabled",)], QCheckBox)
assert "已配置 2 个音色" in panel._overview_status_labels["tts"].text()
assert isinstance(panel._friendly_controls["tts"][("role",)], QLineEdit)
panel._friendly_controls["tts"][("role",)].setText("assistant")
assert panel.draft_values()["tts"]["role"] == "assistant"
assert panel._friendly_controls["tts"][("health_probe",)].isChecked() is True
assert isinstance(
    panel._friendly_controls["tts"][("startup_timeout_seconds",)], QDoubleSpinBox
)
assert panel._friendly_controls["tts"][("startup_timeout_seconds",)].value() == 300.0
assert isinstance(
    panel._friendly_controls["tts"][("shutdown_timeout_seconds",)], QDoubleSpinBox
)
assert panel._friendly_controls["tts"][("shutdown_timeout_seconds",)].value() == 2.0
assert isinstance(panel.form_controls["rendering"][("sprite_scale",)], QDoubleSpinBox)
assert isinstance(panel.form_controls["rendering"][("frame_rate",)], QDoubleSpinBox)
assert isinstance(panel.form_controls["rendering"][("geometry_audit_hz",)], QDoubleSpinBox)
assert isinstance(panel.form_controls["rendering"][("backend",)], QComboBox)
assert panel.form_controls["rendering"][("backend",)].currentText() == "auto"
backend_buttons = panel._friendly_choice_buttons[("rendering", ("backend",))]
assert [button.text() for button in backend_buttons] == [
    "自动",
    "OpenGL",
    "Web Live2D",
    "精灵",
    "Vulkan",
]
assert all(isinstance(button, QPushButton) for button in backend_buttons)
assert backend_buttons[-1].isEnabled() is True
backend_buttons[2].click()
assert panel.draft_values()["rendering"]["backend"] == "web_live2d"
panel.navigation.setCurrentRow(panel.sections.index("memory"))
assert panel._friendly_controls["memory"][("enabled",)].isVisible()
assert isinstance(panel._friendly_controls["memory"][("recall_limit",)], QSpinBox)
assert isinstance(panel._friendly_controls["memory"][("context_max_chars",)], QSpinBox)
assert isinstance(panel._friendly_controls["memory"][("decay_days",)], QDoubleSpinBox)
assert isinstance(panel._friendly_controls["memory"][("semantic", "enabled")], QCheckBox)
assert isinstance(panel._friendly_controls["memory"][("semantic", "dimension")], QSpinBox)
assert panel._friendly_controls["memory"][("semantic", "dimension")].value() == 384
assert panel._overview_status_labels["memory"].text().startswith("长期记忆已开启")
panel._friendly_controls["memory"][("recall_limit",)].setValue(12)
panel._friendly_controls["memory"][("context_max_chars",)].setValue(8000)
panel._friendly_controls["memory"][("consolidation_enabled",)].setChecked(False)
assert panel.draft_values()["memory"]["recall_limit"] == 12
assert panel.draft_values()["memory"]["context_max_chars"] == 8000
assert panel.draft_values()["memory"]["consolidation_enabled"] is False
assert isinstance(panel.form_controls["app"][("name",)], QLineEdit)
panel.navigation.setCurrentRow(panel.sections.index("app"))
persona_traits = panel._friendly_controls["app"][("persona", "traits")]
dialogue_prompt = panel._friendly_controls["app"][("prompts", "dialogue")]
assert isinstance(persona_traits, QPlainTextEdit) and persona_traits.isVisible()
assert isinstance(dialogue_prompt, QPlainTextEdit) and dialogue_prompt.isVisible()
persona_traits.setPlainText("好奇\n可靠")
dialogue_prompt.setPlainText("保持桌宠语气，并根据工具回执回答。")
assert panel.draft_values()["app"]["persona"]["traits"] == ["好奇", "可靠"]
assert panel.draft_values()["app"]["prompts"]["dialogue"] == "保持桌宠语气，并根据工具回执回答。"
for path, selected in (
    (("movement", "moving_motion"), "walk"),
    (("movement", "idle_motion"), "idle"),
):
    control = panel.form_controls["behavior"][path]
    assert isinstance(control, QComboBox)
    assert control.count() == 4
    assert control.currentText() == selected
panel.form_controls["tts"][("enabled",)].setChecked(True)
panel.form_controls["rendering"][("sprite_scale",)].setValue(1.25)
assert panel.draft_values()["tts"]["enabled"] is True
assert panel.draft_values()["rendering"]["sprite_scale"] == 1.25
observation_toggle = panel.findChild(QCheckBox, "configToolGroup_desktop_observation")
assert observation_toggle is not None
assert observation_toggle.isChecked()
assert "active_groups" not in panel.draft_values()["tools"]
observation_toggle.setChecked(False)
assert "desktop_observation" not in panel.draft_values()["tools"]["active_groups"]
assert {
    "pet_control",
    "scheduler",
    "system",
}.issubset(panel.draft_values()["tools"]["active_groups"])
for group in ("desktop_control", "pet_control", "scheduler", "system"):
    control = panel.findChild(QCheckBox, f"configToolGroup_{group}")
    assert control is not None
    control.setChecked(False)
assert panel.draft_values()["tools"]["active_groups"] == ["mcp"]
system_toggle = panel.findChild(QCheckBox, "configToolGroup_system")
assert system_toggle is not None
assert "审批" in system_toggle.toolTip()
channels = panel.complex_editors["llm"][("channels",)]
channels.setPlainText("- id: primary\n  api_key: '***'\n  model: newer\n")
assert panel.draft_values()["llm"]["channels"][0]["model"] == "newer"
panel._search.setText("语音")
assert panel.navigation.item(3).isHidden() is False
assert panel.navigation.item(2).isHidden() is True
panel._search.clear()
assert panel.save_button.isEnabled()
print("qt-modern-config-ok")
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
    assert "qt-modern-config-ok" in result.stdout


def test_qt_console_without_callbacks_validates_before_direct_file_save(tmp_path: Path) -> None:
    """仅传入配置路径时也必须复用运行时校验，不能把坏任务直接写入文件。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    path = tmp_path / "config.yaml"
    original = "app:\n  name: MeaPet\nscheduler:\n  tasks: []\n"
    path.write_text(original, encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration_path={str(path)!r})
panel = window.configuration_panel
assert panel is not None
panel.navigation.setCurrentRow(panel.sections.index("scheduler"))
editor = panel._section_editors["scheduler"]
editor.setPlainText(
    "tasks:\\n  - name: broken\\n    expression: every:invalid\\n"
    "    action: {{identity: pet:play_motion, arguments: {{name: blink}}}}\\n"
)
assert panel.save(confirmed=True)
assert "配置保存失败" in panel.status_label.text()
assert open({str(path)!r}, encoding="utf-8").read() == {original!r}
window.shutdown()
app.processEvents()
print("qt-console-default-validation-ok")
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
    assert "qt-console-default-validation-ok" in result.stdout


def test_qt_console_direct_save_rejects_external_file_change(tmp_path: Path) -> None:
    """独立控制台保存不能覆盖打开编辑器后出现的外部 YAML 修改。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    path = tmp_path / "config.yaml"
    original = "app:\n  name: MeaPet\nscheduler:\n  tasks: []\n"
    external = "app:\n  name: External\nscheduler:\n  tasks: []\n"
    path.write_text(original, encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import copy
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(configuration_path={str(path)!r})
open({str(path)!r}, "w", encoding="utf-8").write({external!r})
draft = copy.deepcopy(window._configuration)
draft["app"]["name"] = "Editor"
assert window._save_configuration(draft, replace_values=True) is False
assert "已被外部修改" in window.status_label.text()
assert open({str(path)!r}, encoding="utf-8").read() == {external!r}
window.shutdown()
app.processEvents()
print("qt-console-config-conflict-ok")
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
    assert "qt-console-config-conflict-ok" in result.stdout


def test_qt_console_rejects_unknown_configuration_callback_result(tmp_path: Path) -> None:
    """配置回调误返回整数等未知类型时，控制台不能显示保存成功。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    configuration={"app": {"name": "MeaPet"}},
    config_callbacks={"validate": lambda _values: 1},
)
panel = window.configuration_panel
assert panel is not None
panel._validate_draft()
assert "配置校验失败" in panel.status_label.text()
window.shutdown()
app.processEvents()
print("qt-console-result-contract-ok")
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
    assert "qt-console-result-contract-ok" in result.stdout


def test_qt_configuration_enum_controls_do_not_display_blank_values(tmp_path: Path) -> None:
    """配置中的空枚举值不能让可视化下拉框呈现空白项。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication, QComboBox
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({
    "logging": {"level": "", "console": {"level": "", "color": ""}, "file": {"rotation": ""}},
    "tts": {"backend": "", "language": "", "media_type": ""},
})
assert panel.form_controls["tts"][("enabled",)].isChecked()
assert panel.form_controls["tts"][("backend",)].currentText() == "gpt_sovits_stdio"
for paths in panel.form_controls.values():
    for control in paths.values():
        if isinstance(control, QComboBox):
            assert control.count() > 0
            assert control.currentText().strip()
            assert all(control.itemText(i).strip() for i in range(control.count()))
draft = panel.draft_values()
assert draft["tts"] == {"backend": "gpt_sovits_stdio", "language": "zh", "media_type": "wav"}
print("qt-config-enum-options-ok")
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
    assert "qt-config-enum-options-ok" in result.stdout


def test_qt_configuration_save_is_single_flight_and_winui_styled(tmp_path: Path) -> None:
    """保存等待宿主回执期间禁用重复提交，并保持 WinUI 语义主题。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication, QLabel, QScrollArea
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({"tts": {"enabled": False, "language": "zh"}})
panel.show()
app.processEvents()
panel.navigation.setCurrentRow(panel.sections.index("tts"))
panel._friendly_controls["tts"][("enabled",)].click()
saves = []
panel.saveRequested.connect(lambda values: saves.append(values))
assert panel.save(confirmed=True)
assert panel.save_in_flight is True
assert panel.save_button.isEnabled() is False
assert panel.save(confirmed=True) is False
assert len(saves) == 1
assert "上一轮保存" in panel.status_label.text()
assert panel._md3_theme.colors.primary in panel.styleSheet()
assert "#ff9dbe" not in panel.styleSheet().lower()
assert "configFriendlyToolGroups" in panel.styleSheet()
overview_scroll = panel.findChild(QScrollArea, "friendlyOverviewScroll")
assert overview_scroll is not None
overview_color = overview_scroll.widget().palette().color(
    overview_scroll.widget().backgroundRole()
).name()
assert overview_color == panel._md3_theme.colors.surface.lower()
assert not any(
    "YAML" in label.text() for label in panel.findChildren(QLabel) if label.isVisible()
)
panel.set_save_result(True, "配置已保存。", values=panel.draft_values())
assert panel.save_in_flight is False
panel.close()
app.processEvents()
print("qt-config-single-flight-style-ok")
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
    assert "qt-config-single-flight-style-ok" in result.stdout


def test_qt_configuration_auto_save_is_debounced_and_failure_does_not_loop(
    tmp_path: Path,
) -> None:
    """自动保存只在编辑空闲后提交一次，失败后等待下一次用户编辑。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import time
from PySide6.QtWidgets import QApplication
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({"tts": {"enabled": False, "language": "zh"}}, auto_save=True)
saves = []
panel.saveRequested.connect(saves.append)
panel.navigation.setCurrentRow(panel.sections.index("tts"))
panel._friendly_controls["tts"][("enabled",)].click()
deadline = time.monotonic() + 1.15
while time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.02)
assert len(saves) == 1
panel.set_save_result(False, "保存失败")
saves.clear()
deadline = time.monotonic() + 0.9
while time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.02)
assert saves == []
panel._friendly_controls["tts"][("enabled",)].click()
panel._friendly_controls["tts"][("enabled",)].click()
deadline = time.monotonic() + 1.15
while time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.02)
assert len(saves) == 1
panel.close()
app.processEvents()
print("qt-config-auto-save-debounce-ok")
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
    assert "qt-config-auto-save-debounce-ok" in result.stdout


def test_qt_model_setup_persists_plaintext_key_to_config_file(tmp_path: Path) -> None:
    """勾选持久化后密钥写入配置文件并收紧权限，界面仍不回显。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import yaml
from pathlib import Path
from PySide6.QtWidgets import QApplication
from config.loader import default_configuration_values, load_configuration
from gui.qt6.console import PetConsoleWindow
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
config_path = Path({str(tmp_path / "persisted.yaml")!r})
defaults = default_configuration_values(
    resource_root=Path({str(Path("resources").resolve())!r}),
    database_path=Path({str(tmp_path / "db.sqlite3")!r}),
)
config_path.write_text(
    yaml.safe_dump(defaults, allow_unicode=True, sort_keys=False), encoding="utf-8"
)
loaded = load_configuration(config_path, defaults=defaults)
window = PetConsoleWindow(configuration=loaded.values, configuration_path=config_path)
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset(
        "openai",
        channel_id="primary",
        base_url="http://127.0.0.1:8317/v1",
        model="gpt-5.4-mini",
    ),
    on_submit=window._apply_model_setup_payload,
    on_delete=window._delete_model_channel,
    parent=window,
)
dialog.card.api_key_persist_check.setChecked(True)
dialog.card.api_key_value_edit.setText("persisted-test-secret")
assert dialog.card.submit()
raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
channel = raw["llm"]["channels"][0]
assert channel["api_key"] == "persisted-test-secret"
assert channel["api_key_env"] == ""
assert oct(config_path.stat().st_mode & 0o777) == "0o600"
assert "persisted-test-secret" not in dialog.yaml_text()
window.shutdown()
app.processEvents()
print("qt-config-plaintext-file-persist-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = f"{source_root}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-config-plaintext-file-persist-ok" in result.stdout


def test_qt_model_setup_second_edit_keeps_persisted_plaintext_key(tmp_path: Path) -> None:
    """重新编辑已保存明文渠道时，不得被同名环境变量引用覆盖。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = f"""
import os
import yaml
from pathlib import Path
from PySide6.QtWidgets import QApplication
from config.loader import default_configuration_values, load_configuration
from gui.qt6.console import PetConsoleWindow
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
os.environ["OPENAI_API_KEY"] = "environment-value-must-not-win"
config_path = Path({str(tmp_path / "persisted-twice.yaml")!r})
defaults = default_configuration_values(
    resource_root=Path({str(Path("resources").resolve())!r}),
    database_path=Path({str(tmp_path / "db.sqlite3")!r}),
)
config_path.write_text(
    yaml.safe_dump(defaults, allow_unicode=True, sort_keys=False), encoding="utf-8"
)
loaded = load_configuration(config_path, defaults=defaults)
window = PetConsoleWindow(configuration=loaded.values, configuration_path=config_path)
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset(
        "openai", channel_id="primary", base_url="http://127.0.0.1:8317/v1", model="demo"
    ),
    on_submit=window._apply_model_setup_payload,
    on_delete=window._delete_model_channel,
    parent=window,
)
dialog.card.api_key_persist_check.setChecked(True)
dialog.card.api_key_value_edit.setText("stored-twice-secret")
assert dialog.card.submit()
raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
assert raw["llm"]["channels"][0]["api_key"] == "stored-twice-secret"
dialog.card.model_edit.setText("edited-after-persist")
assert dialog.card.submit()
raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
assert raw["llm"]["channels"][0]["api_key"] == "stored-twice-secret"
assert raw["llm"]["channels"][0]["model"] == "edited-after-persist"
window.shutdown()
app.processEvents()
print("qt-config-plaintext-second-edit-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = f"{source_root}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-config-plaintext-second-edit-ok" in result.stdout


def test_configuration_panel_uses_fancyui_winui_settings_shell(tmp_path: Path) -> None:
    """配置中心使用真实 FancyUI 组件，并在窄宽度切换导航形态。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication, QFrame
from gui.qt6.config_panel import ConfigurationPanel
from gui.qt6.fancyui import (
    FancyCard,
    FancyCommandBar,
    FancyInfoBar,
    FancyNavigationView,
    FancyStyleController,
)

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({"tts": {"enabled": True}, "scheduler": {}})
panel.resize(1040, 760)
panel.show()
app.processEvents()
assert isinstance(panel.findChild(QFrame, "configHeader"), FancyCard)
navigation = panel.findChild(FancyNavigationView, "configFancyNavigation")
assert navigation is not None and navigation.count() == len(panel.sections)
assert isinstance(panel.findChild(QFrame, "configCommandBar"), FancyCommandBar)
assert isinstance(panel.findChild(QFrame, "configStatusInfoBar"), FancyInfoBar)
assert isinstance(panel._fancy_style_controller, FancyStyleController)
assert all(isinstance(card, FancyCard) for card in panel._overview_cards)
assert isinstance(panel.findChild(QFrame, "configPageHeader"), FancyCard)
panel.navigation.setCurrentRow(panel.sections.index("tts"))
assert navigation.currentIndex() == panel.sections.index("tts")
panel._set_status("配置已保存。")
assert panel._status_info.severity() == "success"
assert panel._status_info.message() == "配置已保存。"
assert panel.property("responsiveMode") == "split"
assert panel.navigation.isVisible()
assert panel._section_picker is not None and panel._section_picker.isHidden()
panel.resize(700, 720)
app.processEvents()
assert panel.property("responsiveMode") == "stacked"
assert panel.navigation.isHidden()
assert panel._section_picker.isVisible()
assert "#ff9dbe" not in panel.styleSheet().lower()
panel.close()
app.processEvents()
print("config-fancyui-winui-shell-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = f"{source_root}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "config-fancyui-winui-shell-ok" in result.stdout


def test_configuration_panel_asr_domain_preserves_exact_fields_and_unknown_keys(
    tmp_path: Path,
) -> None:
    """ASR 表单应使用最终字段契约，并继续保留扩展配置。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.config_panel import ConfigurationPanel

app = QApplication.instance() or QApplication([])
panel = ConfigurationPanel({
    "asr": {
        "enabled": True,
        "backend": "sensevoice",
        "python_executable": "/opt/asr/python",
        "model_path": "/models/SenseVoiceSmall",
        "device": "cpu",
        "language": "auto",
        "timeout_seconds": 60.0,
        "startup_timeout_seconds": 180.0,
        "max_audio_bytes": 16777216,
        "extension": {"preserve": True},
    }
})
panel.show()
app.processEvents()
assert "asr" in panel.sections
assert panel.open_section("asr")
app.processEvents()
assert panel._built_sections == {"overview", "asr"}
assert not panel.open_section("speech")
assert panel._friendly_controls["asr"][("enabled",)].isVisible()
for path in (("python_executable",), ("model_path",), ("timeout_seconds",),
             ("startup_timeout_seconds",), ("max_audio_bytes",)):
    assert not panel.form_controls["asr"][path].isVisible()
overview_text = "\n".join(label.text() for label in panel._overview_status_labels.values())
assert "/opt/asr/python" not in overview_text
assert "/models/SenseVoiceSmall" not in overview_text
panel.advanced_toggle.click()
app.processEvents()
for path in (("enabled",), ("backend",), ("python_executable",), ("model_path",),
             ("device",), ("language",), ("timeout_seconds",),
             ("startup_timeout_seconds",), ("max_audio_bytes",)):
    assert path in panel.form_controls["asr"]
assert panel.form_controls["asr"][("timeout_seconds",)].minimum() == 0.1
assert panel.form_controls["asr"][("timeout_seconds",)].maximum() == 600.0
assert panel.form_controls["asr"][("startup_timeout_seconds",)].minimum() == 0.1
assert panel.form_controls["asr"][("startup_timeout_seconds",)].maximum() == 600.0
assert panel.form_controls["asr"][("max_audio_bytes",)].minimum() == 1024
assert panel.form_controls["asr"][("max_audio_bytes",)].maximum() == 67108864
draft = panel.draft_values()
assert "timeout" not in draft["asr"]
assert draft["asr"]["timeout_seconds"] == 60.0
assert draft["asr"]["startup_timeout_seconds"] == 180.0
assert draft["asr"]["extension"] == {"preserve": True}
panel.close()
app.processEvents()
print("config-asr-domain-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = f"{source_root}{os.pathsep}{environment.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "config-asr-domain-ok" in result.stdout
