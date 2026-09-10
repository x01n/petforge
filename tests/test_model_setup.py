from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gui.qt6.model_setup import (
    ADAPTER_PROTOCOLS,
    MODEL_SETUP_PRESETS,
    ModelSetupDraft,
    adapter_for_protocol,
    build_model_setup_payload,
    build_model_test_payload,
    model_setup_drafts,
    openai_base_url_hint,
    parse_model_setup_yaml,
    preferred_api_key_env,
    protocols_for_adapter,
)


def test_adapter_protocols_are_explicit_two_level_choices() -> None:
    assert protocols_for_adapter("openai") == ("openai_chat", "openai_responses", "openai")
    assert protocols_for_adapter("anthropic") == ("anthropic_messages", "anthropic", "claude")
    assert protocols_for_adapter("ollama") == ("ollama_chat",)
    assert protocols_for_adapter("custom") == tuple(sorted(ADAPTER_PROTOCOLS["custom"]))


def test_openai_gateway_version_hint_is_advisory_only() -> None:
    assert openai_base_url_hint("openai_chat", "https://gateway.example.invalid")
    assert openai_base_url_hint("openai_responses", "https://gateway.example.invalid/api")
    assert openai_base_url_hint("openai_chat", "https://gateway.example.invalid/v1") == ""
    assert openai_base_url_hint("anthropic_messages", "https://gateway.example.invalid") == ""


def test_existing_protocol_without_preset_recovers_adapter_family() -> None:
    assert adapter_for_protocol("openai_responses") == "openai"
    assert adapter_for_protocol("anthropic_messages") == "anthropic"
    assert adapter_for_protocol("ollama_chat") == "ollama"
    draft = ModelSetupDraft.from_mapping(
        {
            "id": "primary",
            "protocol": "openai_responses",
            "base_url": "http://127.0.0.1:8317/v1",
            "model": "gpt-5.4-mini",
        }
    )
    assert draft.preset == "openai"
    assert draft.adapter == "openai"


def test_persisted_plaintext_channel_is_redacted_when_reopened() -> None:
    drafts = model_setup_drafts(
        {
            "llm": {
                "channels": [
                    {
                        "id": "primary",
                        "protocol": "openai_chat",
                        "base_url": "https://gateway.invalid/v1",
                        "model": "demo",
                        "api_key": "persisted-secret",
                        "api_key_env": "",
                    }
                ]
            }
        }
    )
    assert len(drafts) == 1
    assert drafts[0].api_key_env == ""
    assert "persisted-secret" not in repr(drafts[0])


def test_metadata_only_api_key_environment_reference_is_preserved() -> None:
    draft = ModelSetupDraft.from_mapping(
        {
            "id": "primary",
            "protocol": "openai_chat",
            "base_url": "https://gateway.example.invalid/v1",
            "model": "demo",
            "api_key": "",
            "api_key_env": "OPENAI_API_KEY",
        }
    )
    assert draft.api_key_env == "OPENAI_API_KEY"
    assert draft.clear_persisted_secret is False


def test_explicit_known_adapter_is_kept_when_protocol_is_compatible() -> None:
    draft = ModelSetupDraft.from_mapping(
        {
            "id": "primary",
            "adapter": "anthropic",
            "protocol": "anthropic_messages",
            "base_url": "https://gateway.example.invalid",
            "model": "demo-model",
        }
    )
    assert draft.adapter == "anthropic"


def test_explicit_adapter_rejects_protocol_from_another_family() -> None:
    with pytest.raises(ValueError, match="适配器与服务类型不匹配"):
        ModelSetupDraft.from_mapping(
            {
                "id": "primary",
                "preset": "openai",
                "protocol": "ollama_chat",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "demo-model",
            }
        )


def test_preferred_api_key_env_uses_existing_anthropic_auth_token(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "redacted-test-value")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert preferred_api_key_env("anthropic", "ANTHROPIC_API_KEY") == "ANTHROPIC_AUTH_TOKEN"


def test_preset_draft_uses_documented_protocol_and_environment_reference() -> None:
    draft = ModelSetupDraft.from_preset("openai")

    assert draft.protocol == "openai_chat"
    assert draft.base_url == "https://api.openai.com/v1"
    assert draft.model == "gpt-4o-mini"
    assert draft.api_key_env == "OPENAI_API_KEY"
    assert draft.api_key_reference == "${OPENAI_API_KEY}"


def test_callback_payload_matches_configuration_wizard_contract() -> None:
    draft = ModelSetupDraft(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://gateway.example.invalid/v1",
        model="demo-model",
        api_key_env="MEAPET_TEST_KEY",
    )

    payload = build_model_setup_payload(draft)

    assert payload == {
        "channel_id": "primary",
        "protocol": "openai_chat",
        "base_url": "https://gateway.example.invalid/v1",
        "model": "demo-model",
        "api_key_env": "MEAPET_TEST_KEY",
        "capabilities": ["streaming", "tools"],
        "enabled": True,
        "priority": 10,
    }
    assert "api_key" not in payload


def test_connection_test_payload_contains_only_safe_channel_fields() -> None:
    draft = ModelSetupDraft(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://gateway.example.invalid/v1",
        model="demo-model",
        api_key_env="MEAPET_TEST_KEY",
    )

    payload = build_model_test_payload(draft)

    assert payload == {
        "channel_id": "primary",
        "protocol": "openai_chat",
        "base_url": "https://gateway.example.invalid/v1",
        "model": "demo-model",
        "api_key_env": "MEAPET_TEST_KEY",
        "capabilities": ["streaming", "tools"],
        "enabled": True,
        "priority": 10,
    }
    assert "api_key" not in payload
    assert "secret-value" not in repr(payload)


def test_yaml_draft_contains_reference_but_never_a_secret() -> None:
    draft = ModelSetupDraft(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://gateway.example.invalid/v1",
        model="demo-model",
        api_key_env="MEAPET_TEST_KEY",
    )

    text = draft.to_yaml(validate=True)

    assert "${MEAPET_TEST_KEY}" in text
    assert "api_key_env" not in text
    assert "secret-value" not in text
    assert "preset: custom" in text
    restored = parse_model_setup_yaml(text)
    assert restored.api_key_env == "MEAPET_TEST_KEY"
    assert restored.model == "demo-model"


def test_raw_api_key_and_invalid_environment_name_are_rejected_without_echo() -> None:
    with pytest.raises(ValueError, match="环境变量引用") as raised:
        ModelSetupDraft.from_mapping(
            {
                "id": "primary",
                "protocol": "openai_chat",
                "base_url": "https://gateway.example.invalid/v1",
                "model": "demo-model",
                "api_key": "secret-value",
            }
        )
    assert "secret-value" not in str(raised.value)

    with pytest.raises(ValueError, match="大写环境变量名"):
        ModelSetupDraft(
            channel_id="primary",
            protocol="openai_chat",
            base_url="https://gateway.example.invalid/v1",
            model="demo-model",
            api_key_env="lowercase_key",
        )


def test_runtime_channel_validation_is_reused_for_url_and_protocol() -> None:
    with pytest.raises(ValueError, match="unsupported channel protocol"):
        ModelSetupDraft(
            channel_id="primary",
            protocol="unknown_protocol",
            base_url="https://gateway.example.invalid/v1",
            model="demo-model",
        )

    draft = ModelSetupDraft(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://gateway.example.invalid/v1?token=bad",
        model="demo-model",
    )
    with pytest.raises(ValueError, match="query"):
        draft.validate()

    credential_url = ModelSetupDraft(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://user:secret-value@gateway.example.invalid/v1",
        model="demo-model",
    )
    with pytest.raises(ValueError, match="credentials") as raised:
        credential_url.to_yaml()
    assert "secret-value" not in str(raised.value)


def test_all_builtin_presets_are_parseable_without_secret_values() -> None:
    for key, preset in MODEL_SETUP_PRESETS.items():
        draft = ModelSetupDraft.from_preset(key)
        assert draft.preset == key
        assert preset.api_key_env == draft.api_key_env
        assert "secret" not in draft.to_yaml().lower()


def test_model_setup_drafts_exposes_multiple_channels_without_secrets() -> None:
    drafts = model_setup_drafts(
        {
            "llm": {
                "channels": [
                    {
                        "id": "primary",
                        "protocol": "openai_chat",
                        "base_url": "https://gateway.example.invalid/v1",
                        "api_key": "${PRIMARY_KEY}",
                        "model": "demo-a",
                    },
                    {
                        "id": "backup",
                        "protocol": "ollama_chat",
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key": "secret-value",
                        "model": "demo-b",
                    },
                    {
                        "id": "metadata",
                        "protocol": "openai_chat",
                        "base_url": "https://gateway.example.invalid/v1",
                        "api_key": "",
                        "api_key_env": "META_KEY",
                        "model": "demo-c",
                    },
                ]
            }
        }
    )
    assert [item.channel_id for item in drafts] == ["primary", "backup", "metadata"]
    assert drafts[0].api_key_env == "PRIMARY_KEY"
    assert drafts[1].api_key_env == ""
    assert drafts[2].api_key_env == "META_KEY"
    assert "secret-value" not in repr(drafts)


def test_qt_dialog_emits_safe_payload_when_available(tmp_path: Path) -> None:
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
saved = []
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai", channel_id="primary"),
    channel_drafts=(
        ModelSetupDraft.from_preset("openai", channel_id="primary"),
        ModelSetupDraft.from_preset("ollama", channel_id="backup", model="llama3"),
    ),
)
dialog.saveRequested.connect(saved.append)
assert dialog.test_button.objectName() == "modelSetupTestButton"
assert dialog.card.test_button.objectName() == "modelSetupTestButton"
assert dialog.card.channel_selector.count() == 3
dialog.card.channel_selector.setCurrentIndex(2)
assert dialog.card.channel_id_edit.text() == "backup"
dialog.card.channel_selector.setCurrentIndex(0)
assert dialog.card.channel_id_edit.text() == "primary-2"
dialog.card.preset_combo.setCurrentIndex(dialog.card.preset_combo.findData("openai"))
assert dialog.card.model_edit.text() == "gpt-4o-mini"
assert dialog.card.channel_id_edit.text() == "primary-2"
assert dialog.card.submit()
assert saved[0]["api_key_env"] == "OPENAI_API_KEY"
assert "api_key" not in saved[0]
assert "${OPENAI_API_KEY}" in dialog.yaml_text()
dialog.close()
app.processEvents()
print("model-setup-qt-ok")
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
    assert "model-setup-qt-ok" in result.stdout


def test_qt_adapter_selection_filters_protocols_and_autonames_secret_env(tmp_path: Path) -> None:
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import os
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupCard, ModelSetupDraft

app = QApplication.instance() or QApplication([])
card = ModelSetupCard(
    ModelSetupDraft.from_preset(
        "custom", base_url="https://gateway.example.invalid/v1", model="demo"
    )
)
card.preset_combo.setCurrentIndex(card.preset_combo.findData("openai"))
assert card.adapter_combo is card.preset_combo
assert tuple(card.protocol_combo.itemData(i) for i in range(card.protocol_combo.count())) == (
    "openai_chat", "openai_responses", "openai"
)
card.preset_combo.setCurrentIndex(card.preset_combo.findData("ollama"))
assert tuple(
    card.protocol_combo.itemData(i) for i in range(card.protocol_combo.count())
) == ("ollama_chat",)
card.preset_combo.setCurrentIndex(card.preset_combo.findData("custom"))
card.base_url_edit.setText("https://gateway.example.invalid/v1")
card.model_edit.setText("demo")
card.api_key_env_edit.clear()
card.api_key_value_edit.setText("transient-secret")
assert card._apply_transient_api_key()
assert os.environ["MEAPET_API_KEY"] == "transient-secret"
assert card.api_key_value_edit.text() == ""
os.environ.pop("MEAPET_API_KEY", None)
card.close()
print("model-adapter-protocol-ok")
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
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model-adapter-protocol-ok" in result.stdout


def test_qt_model_setup_exposes_multi_channel_edit_controls(tmp_path: Path) -> None:
    """多渠道向导提供新增/删除入口并保留渠道优先级。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
drafts = (
    ModelSetupDraft.from_preset("openai", channel_id="primary", priority=3),
    ModelSetupDraft.from_preset("anthropic", channel_id="backup", priority=8),
)
dialog = ModelSetupDialog(
    drafts[0],
    channel_drafts=drafts,
    on_delete=lambda _id: {"status": "saved"},
)
assert dialog.card.channel_selector.count() == 3
assert dialog.card.delete_channel_button.isEnabled()
assert dialog.card.priority_spin.value() == 3
dialog.card.new_channel_button.click()
assert not dialog.card.delete_channel_button.isEnabled()
assert dialog.card.channel_id_edit.text() == "primary-2"
dialog.close()
app.processEvents()
print("model-multi-channel-controls-ok")
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
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model-multi-channel-controls-ok" in result.stdout


def test_qt_model_setup_auto_saves_channel_without_closing_dialog(tmp_path: Path) -> None:
    """渠道字段停止编辑后自动提交一次，向导保持可继续编辑。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import time
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
saved = []
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai", channel_id="primary"),
    on_submit=lambda values: saved.append(values) or {"status": "saved"},
)
dialog.show()
app.processEvents()
dialog.card.model_edit.setText("gpt-5.4-mini")
deadline = time.monotonic() + 1.15
while time.monotonic() < deadline:
    app.processEvents()
    time.sleep(0.02)
assert len(saved) == 1
assert dialog.isVisible()
dialog.close()
app.processEvents()
print("model-channel-auto-save-ok")
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
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model-channel-auto-save-ok" in result.stdout


def test_qt_model_setup_preserves_priority_and_refreshes_saved_channel(tmp_path: Path) -> None:
    """保存后选择器应保留优先级和最新草稿，便于继续编辑多个渠道。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
saved = []
draft = ModelSetupDraft.from_preset("openai", channel_id="primary", priority=3)
dialog = ModelSetupDialog(draft, on_submit=lambda value: saved.append(value) or {"status": "saved"})
assert dialog.card.submit()
assert saved[0]["llm"]["channels"][0]["priority"] == 3
assert [item.channel_id for item in dialog.card.channel_drafts] == ["primary"]
dialog.card.model_edit.setText("edited-model")
assert dialog.card.submit()
assert saved[-1]["llm"]["channels"][0]["model"] == "edited-model"
assert saved[-1]["llm"]["channels"][0]["priority"] == 3
assert dialog.card.channel_drafts[0].model == "edited-model"
dialog.close()
app.processEvents()
print("model-setup-priority-refresh-ok")
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
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model-setup-priority-refresh-ok" in result.stdout


def test_qt_model_setup_can_clear_saved_key_explicitly(tmp_path: Path) -> None:
    """清除密钥是显式操作，并在自动保存开启时要求手动提交。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
saved = []
draft = ModelSetupDraft.from_mapping({
    "id": "primary",
    "protocol": "openai_chat",
    "base_url": "https://gateway.example.invalid/v1",
    "model": "demo",
    "api_key": "${SAVED_KEY}",
    "api_key_env": "SAVED_KEY",
})
dialog = ModelSetupDialog(draft, on_submit=lambda value: saved.append(value) or {"status": "saved"})
dialog.card.clear_api_key_check.setChecked(True)
assert dialog.card.clear_persisted_secret_requested
dialog.card.model_edit.setText("changed-before-manual-save")
app.processEvents()
assert not saved
assert dialog.card.submit()
assert saved
channel = saved[0]["llm"]["channels"][0]
assert channel["api_key"] == ""
assert "api_key_env" not in channel
dialog.close()
app.processEvents()
print("model-setup-clear-key-ok")
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
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model-setup-clear-key-ok" in result.stdout


def test_qt_model_setup_blocks_stale_external_configuration_save(tmp_path: Path) -> None:
    """外部热重载后自动保存必须停止，避免旧渠道覆盖新配置。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
saved = []
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai", channel_id="primary"),
    on_submit=lambda value: saved.append(value) or {"status": "saved"},
)
dialog.mark_external_configuration_changed()
dialog.card.model_edit.setText("stale-edit")
app.processEvents()
dialog._submit()
assert saved == []
assert not dialog.auto_save_check.isEnabled()
dialog.close()
app.processEvents()
print("model-setup-stale-guard-ok")
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
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model-setup-stale-guard-ok" in result.stdout


def test_qt_transient_key_uses_selected_adapter_standard_environment(tmp_path: Path) -> None:
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import os
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupCard, ModelSetupDraft

app = QApplication.instance() or QApplication([])
for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
    os.environ.pop(name, None)
for adapter, expected in (("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")):
    card = ModelSetupCard(ModelSetupDraft.from_preset(adapter, api_key_env=""))
    card.api_key_env_edit.clear()
    card.api_key_value_edit.setText("transient-secret")
    assert card._apply_transient_api_key()
    assert card.draft().api_key_env == expected
    assert os.environ[expected] == "transient-secret"
    os.environ.pop(expected, None)
    card.close()
print("model-adapter-secret-default-ok")
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
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model-adapter-secret-default-ok" in result.stdout


def test_qt_custom_adapter_keeps_empty_environment_without_secret(tmp_path: Path) -> None:
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupCard, ModelSetupDraft

app = QApplication.instance() or QApplication([])
card = ModelSetupCard(ModelSetupDraft.from_preset("openai"))
card.preset_combo.setCurrentIndex(card.preset_combo.findData("custom"))
card.base_url_edit.setText("http://127.0.0.1:8317/v1")
card.model_edit.setText("demo")
assert card.api_key_env_edit.text() == ""
assert card.draft().api_key_reference == ""
assert card.is_valid
card.close()
print("model-custom-empty-key-ok")
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
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "model-custom-empty-key-ok" in result.stdout


def test_qt_transient_api_key_is_injected_and_erased_without_entering_payload(
    tmp_path: Path,
) -> None:
    """密码只在点击保存前注入当前进程，控件关闭后不保留且不进入脱敏结果。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import os
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
env_name = "MEAPET_TRANSIENT_UI_KEY"
secret = "transient-secret-value"
os.environ.pop(env_name, None)
saved = []
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai", channel_id="primary", api_key_env=env_name)
)
dialog.saveRequested.connect(saved.append)
dialog.card.api_key_value_edit.setText(secret)
assert dialog.card.api_key_value_edit.echoMode().name == "Password"
assert dialog.card.submit()
assert os.environ[env_name] == secret
assert dialog.card.api_key_value_edit.text() == ""
assert dialog.card.api_key_value_edit.isUndoAvailable() is False
assert saved and saved[0]["api_key_env"] == env_name
assert "api_key" not in saved[0]
assert secret not in repr(saved[0])
assert secret not in dialog.yaml_text()

second = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai", channel_id="secondary", api_key_env=env_name)
)
second.card.api_key_value_edit.setText(secret)
second.reject()
app.processEvents()
assert second.card.api_key_value_edit.text() == ""
assert second.card.api_key_value_edit.isUndoAvailable() is False
os.environ.pop(env_name, None)
print("model-transient-api-key-ok")
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
    assert "model-transient-api-key-ok" in result.stdout


def test_qt_persisted_api_key_reaches_only_controlled_submit_callback(tmp_path: Path) -> None:
    """勾选文件持久化时明文只交给受控保存回调，公开信号仍然脱敏。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
secret = "persisted-secret-value"
callbacks = []
signals = []
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai", channel_id="primary"),
    on_submit=lambda payload: callbacks.append(dict(payload)) or {"status": "saved"},
)
dialog.saveRequested.connect(signals.append)
dialog.card.api_key_persist_check.setChecked(True)
dialog.card.api_key_value_edit.setText(secret)
assert dialog.card.submit()
assert callbacks and callbacks[0]["api_key"] == secret
assert callbacks[0]["_persist_api_key"] is True
assert signals and "api_key" not in signals[0]
assert secret not in repr(signals[0])
assert dialog.card.api_key_value_edit.text() == ""
assert secret not in dialog.yaml_text()
dialog.close()
app.processEvents()
print("model-persisted-api-key-callback-ok")
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
    assert "model-persisted-api-key-callback-ok" in result.stdout


def test_qt_model_test_result_hides_internal_protocol_text(tmp_path: Path) -> None:
    """向导不应把宿主返回的协议赋值或密钥片段显示给用户。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai"),
    on_test=lambda _payload: {
        "status": "unavailable",
        "message": "api_key=super-secret status=failed",
    },
)
dialog.card.test_connection()
assert "super-secret" not in dialog.card.status_label.text()
assert "api_key" not in dialog.card.status_label.text().lower()
assert "状态" not in dialog.card.status_label.text()
dialog.close()
app.processEvents()
print("model-test-result-redacted-ok")
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
    assert "model-test-result-redacted-ok" in result.stdout


def test_model_test_result_hides_extended_credential_fields(tmp_path: Path) -> None:
    """模型探测回执中的扩展敏感键也必须使用统一占位文案。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
for message in (
    "credential=private-value",
    "access_key=private-value",
    "private_key=private-value",
):
    dialog = ModelSetupDialog(
        ModelSetupDraft.from_preset("openai"),
        on_test=lambda _payload, message=message: {
            "status": "unavailable",
            "message": message,
        },
    )
    dialog.card.test_connection()
    assert "private-value" not in dialog.card.status_label.text()
    assert "credential" not in dialog.card.status_label.text().lower()
    assert "access_key" not in dialog.card.status_label.text().lower()
    assert "private_key" not in dialog.card.status_label.text().lower()
    dialog.close()
    app.processEvents()
print("model-extended-credential-redacted-ok")
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
    assert "model-extended-credential-redacted-ok" in result.stdout


def test_qt_dialog_keeps_open_on_structured_save_failure(tmp_path: Path) -> None:
    """模型宿主返回失败状态时向导不能误报保存成功并关闭。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai"),
    on_submit=lambda _payload: {
        "status": "unavailable",
        "reason": "environment variable is not set: OPENAI_API_KEY",
    },
)
dialog.show()
app.processEvents()
dialog.card.submit()
app.processEvents()
assert dialog.isVisible()
assert "保存未完成" in dialog.card.status_label.text()
assert "OPENAI_API_KEY" not in dialog.card.status_label.text()
dialog.close()
app.processEvents()
print("model-setup-structured-failure-ok")
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
    assert "model-setup-structured-failure-ok" in result.stdout


def test_qt_dialog_explains_pending_runtime_application(tmp_path: Path) -> None:
    """宿主异步应用模型设置时，向导反馈应明确处于应用中。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai"),
    on_submit=lambda _payload: {"status": "saved", "runtime_status": "pending"},
)
dialog.show()
app.processEvents()
assert dialog.card.submit()
assert "正在应用" in dialog.card.status_label.text()
dialog.close()
app.processEvents()
print("model-setup-pending-feedback-ok")
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
    assert "model-setup-pending-feedback-ok" in result.stdout


def test_qt_dialog_exposes_click_test_connection_signal(tmp_path: Path) -> None:
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft

app = QApplication.instance() or QApplication([])
payloads = []
dialog = ModelSetupDialog(
    ModelSetupDraft.from_preset("openai"),
    on_test=lambda payload: {"status": "available", "message": "模型连接测试通过"},
)
dialog.testRequested.connect(payloads.append)
dialog.show()
app.processEvents()
assert dialog.card.test_connection()
app.processEvents()
assert payloads[0]["api_key_env"] == "OPENAI_API_KEY"
assert "api_key" not in payloads[0]
assert "模型连接测试通过" in dialog.card.status_label.text()
dialog.close()
app.processEvents()
print("model-setup-test-connection-ok")
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
    assert "model-setup-test-connection-ok" in result.stdout


def test_model_setup_uses_fancyui_step_wizard_and_responsive_layout(tmp_path: Path) -> None:
    """模型配置使用三步 FancyUI 向导，状态和窄屏布局可验证。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication, QFrame, QLabel
from gui.qt6.fancyui import FancyCard, FancyCommandBar, FancyInfoBar, FancyStyleController
from gui.qt6.model_setup import ModelSetupCard, ModelSetupDraft

app = QApplication.instance() or QApplication([])
card = ModelSetupCard(ModelSetupDraft.from_preset("openai"))
card.resize(920, 760)
card.show()
app.processEvents()
steps = card.findChildren(QLabel, "modelWizardStepItem")
assert [step.property("step") for step in steps] == ["1", "2", "3"]
assert len(card.findChildren(FancyCard, "modelWizardStepCard")) == 2
assert isinstance(card.findChild(QFrame, "modelWizardHeader"), FancyCard)
assert isinstance(card.findChild(QFrame, "modelSetupInfoBar"), FancyInfoBar)
assert isinstance(card._fancy_style_controller, FancyStyleController)
assert card.property("responsiveMode") == "split"
card.set_test_pending()
assert card.status_info.severity() == "warning"
card.set_test_result(True, "模型连接测试通过")
assert card.status_info.severity() == "success"
card.api_key_value_edit.setText("must-never-enter-status")
assert "must-never-enter-status" not in card.status_info.message()
card.resize(620, 760)
app.processEvents()
assert card.property("responsiveMode") == "stacked"
assert card._wizard_step_rail is not None and card._wizard_step_rail.maximumHeight() == 74
assert "#ff9dbe" not in card.styleSheet().lower()
card.close()
app.processEvents()
print("model-fancyui-step-wizard-ok")
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
    assert "model-fancyui-step-wizard-ok" in result.stdout
