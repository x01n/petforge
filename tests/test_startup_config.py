from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest
import yaml

import config.loader as loader
from app.__main__ import main as cli_main
from app.runtime import initialize_database
from config.loader import (
    MAX_CONFIGURATION_BYTES,
    ConfigurationError,
    configuration_source_digest,
    discover_configuration,
    expand_environment_values,
    load_configuration,
    redact_secrets,
)
from wizard.__main__ import main as wizard_main
from wizard.configuration import ConfigurationWizard


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def _environment(root: Path) -> dict[str, str]:
    return {
        "HOME": str(root / "home"),
        "XDG_CONFIG_HOME": str(root / "xdg-config"),
        "XDG_DATA_HOME": str(root / "xdg-data"),
    }


def test_loaded_configuration_records_exact_source_digest(tmp_path: Path) -> None:
    path = _write(tmp_path / "config.yaml", "app:\n  name: MeaPet\n")

    loaded = load_configuration(path)

    assert loaded.source_digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_configuration_load_and_revision_reject_oversized_file(tmp_path: Path) -> None:
    path = tmp_path / "oversized.yaml"
    path.write_bytes(b"x" * (MAX_CONFIGURATION_BYTES + 1))

    with pytest.raises(ConfigurationError, match="size limit"):
        load_configuration(path)
    with pytest.raises(ConfigurationError, match="size limit"):
        configuration_source_digest(path)


def test_configuration_load_rejects_unsupported_channel_protocol(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "unsupported-protocol.yaml",
        "llm:\n  channels:\n    - id: primary\n      protocol: unsupported_protocol\n",
    )

    with pytest.raises(ConfigurationError, match="unsupported protocol"):
        load_configuration(
            path,
            defaults=loader.default_configuration_values(
                resource_root=tmp_path / "resources", database_path=tmp_path / "db.sqlite3"
            ),
            environment={},
        )


def test_configuration_load_rejects_unknown_route_channel(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "unknown-route.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://primary.invalid/v1
      model: primary-model
  routing:
    dialogue:
      channel: missing
""",
    )

    with pytest.raises(ConfigurationError, match="unknown channel"):
        load_configuration(path, defaults={}, environment={})


def test_configuration_load_rejects_duplicate_or_primary_fallback(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "duplicate-fallback.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://primary.invalid/v1
      model: primary-model
    - id: backup
      protocol: openai_chat
      base_url: https://backup.invalid/v1
      model: backup-model
  routing:
    dialogue:
      channel: primary
      fallback_channels: [backup, backup]
""",
    )

    with pytest.raises(ConfigurationError, match="duplicates"):
        load_configuration(path, defaults={}, environment={})

    path.write_text(
        path.read_text(encoding="utf-8").replace("[backup, backup]", "[backup, primary]"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="primary channel"):
        load_configuration(path, defaults={}, environment={})


def test_configuration_load_rejects_route_without_eligible_model_or_capability(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "ineligible-route.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://primary.invalid/v1
      model: primary-model
      capabilities: [streaming]
  routing:
    dialogue:
      channel: primary
      model: missing-model
""",
    )

    with pytest.raises(ConfigurationError, match="model is not declared"):
        load_configuration(path, defaults={}, environment={})

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "model: missing-model", "required_capabilities: [vision]"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="required_capabilities"):
        load_configuration(path, defaults={}, environment={})


def test_configuration_load_accepts_boolean_vision_route_switch(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "vision-switch.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://primary.invalid/v1
      model: primary-model
  routing:
    vision: false
""",
    )

    loaded = load_configuration(path, defaults={}, environment={})

    assert loaded.values["llm"]["routing"]["vision"] is False


def test_configuration_load_accepts_vision_policy_without_vision_channel(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path / "vision-policy.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://primary.invalid/v1
      model: primary-model
  routing:
    vision:
      enabled: false
      max_tokens: 64
""",
    )

    loaded = load_configuration(path, defaults={}, environment={})

    assert loaded.values["llm"]["routing"]["vision"]["enabled"] is False


@pytest.mark.parametrize("field", ["adapter", "type"])
def test_configuration_load_accepts_legacy_channel_protocol_alias(
    tmp_path: Path, field: str
) -> None:
    path = _write(
        tmp_path / f"legacy-{field}.yaml",
        f"llm:\n  channels:\n    - id: primary\n      {field}: openai_chat\n"
        "      base_url: https://gateway.invalid/v1\n"
        "      model: demo-model\n",
    )

    loaded = load_configuration(
        path,
        defaults=loader.default_configuration_values(
            resource_root=tmp_path / "resources", database_path=tmp_path / "db.sqlite3"
        ),
        environment={},
    )

    channel = loaded.values["llm"]["channels"][0]
    assert channel[field] == "openai_chat"


def test_configuration_discovery_uses_explicit_then_environment_then_repository(
    tmp_path: Path,
) -> None:
    explicit = _write(tmp_path / "explicit.yaml", "app:\n  name: explicit\n")
    environment = _write(tmp_path / "env.yaml", "app:\n  name: environment\n")
    _write(tmp_path / "config.yml", "app:\n  name: repository\n")
    env = _environment(tmp_path)
    env["MEAPET_CONFIG"] = str(environment)

    selected = discover_configuration(
        explicit,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.path == explicit.resolve()
    assert selected.values["app"]["name"] == "explicit"

    selected = discover_configuration(
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.path == environment.resolve()
    assert selected.values["app"]["name"] == "environment"

    env.pop("MEAPET_CONFIG")
    selected = discover_configuration(
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.path == (tmp_path / "config.yml").resolve()
    assert selected.values["app"]["name"] == "repository"


def test_configuration_discovery_prefers_yaml_name_and_cwd_over_project_root(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "run"
    project = tmp_path / "project"
    cwd.mkdir()
    project.mkdir()
    _write(cwd / "config.yaml", "app:\n  name: cwd-yaml\n")
    _write(cwd / "config.yml", "app:\n  name: cwd-yml\n")
    _write(project / "config.yaml", "app:\n  name: project-yaml\n")
    selected = discover_configuration(
        environment=_environment(tmp_path),
        cwd=cwd,
        project_root=project,
        home=tmp_path / "home",
    )
    assert selected.path == (cwd / "config.yaml").resolve()
    assert selected.values["app"]["name"] == "cwd-yaml"


def test_configuration_discovery_checks_user_config_before_project_example(
    tmp_path: Path,
) -> None:
    env = _environment(tmp_path)
    user_path = _write(
        tmp_path / "xdg-config" / "meapet" / "config.yaml",
        "app:\n  name: user\n",
    )
    example_path = _write(
        tmp_path / "config" / "app.example.yaml",
        "app:\n  name: example\n",
    )
    selected = discover_configuration(
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.path == user_path.resolve()
    assert selected.values["app"]["name"] == "user"
    assert example_path.is_file()


def test_configuration_discovery_uses_example_without_overwriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    example_path = _write(
        tmp_path / "config" / "app.example.yaml",
        "app:\n  name: example\n",
    )
    selected = discover_configuration(
        environment=_environment(tmp_path),
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.path == example_path.resolve()
    assert not (tmp_path / "xdg-config" / "meapet" / "config.yaml").exists()


def test_configuration_discovery_creates_safe_default_without_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    env = _environment(tmp_path)
    selected = discover_configuration(
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.path == (tmp_path / "xdg-config" / "meapet" / "config.yaml").resolve()
    assert selected.path.is_file()
    assert selected.values["llm"]["channels"] == []
    assert selected.values["tts"]["enabled"] is True
    assert selected.values["tts"]["backend"] == "gpt_sovits_stdio"
    assert "api_key" not in selected.path.read_text(encoding="utf-8").lower()
    assert (
        Path(selected.values["storage"]["database"]).parent
        == (tmp_path / "xdg-data" / "meapet").resolve()
    )


def test_default_local_web_api_is_disabled_and_does_not_contain_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    selected = discover_configuration(
        environment=_environment(tmp_path),
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    local_api = selected.values["web"]["local_api"]
    assert local_api == {
        "enabled": False,
        "host": "127.0.0.1",
        "port": 0,
        "events": True,
        "event_limit": 64,
    }
    assert "capability" not in selected.path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("host", "0.0.0.0"),
        ("port", 65536),
        ("port", True),
        ("event_limit", 0),
        ("event_limit", 65),
    ),
)
def test_local_web_api_configuration_rejects_unsafe_values(
    tmp_path: Path, field: str, value: object
) -> None:
    config_path = _write(
        tmp_path / f"invalid-local-web-{field}.yaml",
        yaml.safe_dump({"web": {"local_api": {field: value}}}, allow_unicode=True),
    )
    with pytest.raises(ConfigurationError, match=f"web.local_api.{field}"):
        load_configuration(
            config_path,
            defaults=loader.default_configuration_values(
                resource_root=tmp_path / "resources", database_path=tmp_path / "db.sqlite3"
            ),
            environment={},
        )


def test_configuration_discovery_bootstraps_channel_from_explicit_environment(
    tmp_path: Path,
) -> None:
    config_path = _write(tmp_path / "config.yaml", "llm:\n  channels: []\n")
    env = _environment(tmp_path)
    env.update(
        {
            "MEAPET_API_BASE": "http://127.0.0.1:11434/v1",
            "MEAPET_MODEL": "demo-model",
            "MEAPET_API_KEY": "local-key",
        }
    )
    selected = discover_configuration(
        config_path,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    channel = selected.values["llm"]["channels"][0]
    assert channel["id"] == "primary"
    assert channel["base_url"] == "http://127.0.0.1:11434/v1"
    assert channel["model"] == "demo-model"
    assert channel["api_key"] == "local-key"
    assert channel["capabilities"] == ["streaming", "tools"]
    assert selected.values["llm"]["routing"]["dialogue"]["channel"] == "primary"

    env["MEAPET_CHANNEL_ID"] = "local-gateway"
    selected = discover_configuration(
        config_path,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.values["llm"]["routing"]["dialogue"]["channel"] == "local-gateway"


def test_configuration_discovery_bootstraps_anthropic_environment_without_secret_leak(
    tmp_path: Path,
) -> None:
    """Anthropic 标准环境变量只注入内存渠道，诊断快照不包含令牌。"""

    config_path = _write(tmp_path / "config.yaml", "llm:\n  channels: []\n")
    secret = "anthropic-secret-round117"
    env = _environment(tmp_path)
    env.update(
        {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:13000",
            "ANTHROPIC_AUTH_TOKEN": secret,
            "ANTHROPIC_MODEL": "claude-local-test",
        }
    )
    selected = discover_configuration(
        config_path,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    channel = selected.values["llm"]["channels"][0]
    assert channel["protocol"] == "anthropic_messages"
    assert channel["base_url"] == "http://127.0.0.1:13000"
    assert channel["model"] == "claude-local-test"
    assert channel["api_key"] == secret
    assert secret not in str(redact_secrets(selected.values))


def test_configuration_discovery_bootstraps_openai_standard_key_alias(
    tmp_path: Path,
) -> None:
    config_path = _write(tmp_path / "config.yaml", "llm:\n  channels: []\n")
    env = _environment(tmp_path)
    env.update(
        {
            "MEAPET_API_BASE": "http://127.0.0.1:8317/v1",
            "MEAPET_MODEL": "gpt-5.4-mini",
            "OPENAI_API_KEY": "openai-standard-secret",
        }
    )
    selected = discover_configuration(
        config_path,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    channel = selected.values["llm"]["channels"][0]
    assert channel["protocol"] == "openai_chat"
    assert channel["api_key"] == "openai-standard-secret"


def test_configuration_discovery_bootstraps_openai_standard_endpoint_alias(
    tmp_path: Path,
) -> None:
    config_path = _write(tmp_path / "config.yaml", "llm:\n  channels: []\n")
    env = _environment(tmp_path)
    env.update(
        {
            "OPENAI_BASE_URL": "http://127.0.0.1:8317/v1",
            "OPENAI_MODEL": "gpt-5.4-mini",
            "OPENAI_API_KEY": "openai-endpoint-secret",
        }
    )
    selected = discover_configuration(
        config_path,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    channel = selected.values["llm"]["channels"][0]
    assert channel["protocol"] == "openai_chat"
    assert channel["base_url"] == "http://127.0.0.1:8317/v1"
    assert channel["model"] == "gpt-5.4-mini"
    assert channel["api_key"] == "openai-endpoint-secret"


def test_explicit_api_key_env_hydrates_channel_without_writing_plaintext(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: http://127.0.0.1:8317/v1
      api_key_env: OPENAI_API_KEY
      api_key: ''
      model: demo
""",
    )
    env = _environment(tmp_path)
    env["OPENAI_API_KEY"] = "memory-only-secret"
    selected = load_configuration(config_path, defaults={}, environment=env)
    channel = selected.values["llm"]["channels"][0]
    assert channel["api_key"] == "memory-only-secret"
    assert "memory-only-secret" not in config_path.read_text(encoding="utf-8")
    assert (
        yaml.safe_load(config_path.read_text(encoding="utf-8"))["llm"]["channels"][0]["api_key_env"]
        == "OPENAI_API_KEY"
    )
    redacted = redact_secrets(selected.values)
    assert redacted["llm"]["channels"][0]["api_key_env"] == "OPENAI_API_KEY"
    assert redacted["llm"]["channels"][0]["api_key"] == "***"


def test_redact_secrets_covers_extended_credential_field_names() -> None:
    values = {
        "access_key": "access-secret",
        "private_key": "private-secret",
        "credential": "credential-secret",
        "api_key_env": "OPENAI_API_KEY",
    }
    redacted = redact_secrets(values)
    assert redacted == {
        "access_key": "***",
        "private_key": "***",
        "credential": "***",
        "api_key_env": "OPENAI_API_KEY",
    }


def test_configuration_environment_explicit_meapet_base_wins_over_anthropic_alias(
    tmp_path: Path,
) -> None:
    """显式通用地址优先，避免 shell 中常驻 Anthropic 变量改变协议。"""

    config_path = _write(tmp_path / "config.yaml", "llm:\n  channels: []\n")
    env = _environment(tmp_path)
    env.update(
        {
            "MEAPET_API_BASE": "http://127.0.0.1:11434/v1",
            "MEAPET_MODEL": "openai-local-test",
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:13000",
            "ANTHROPIC_MODEL": "claude-local-test",
        }
    )
    selected = discover_configuration(
        config_path,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.values["llm"]["channels"][0]["protocol"] == "openai_chat"


def test_configuration_yaml_anthropic_aliases_expand_only_in_memory(tmp_path: Path) -> None:
    """YAML 引用标准 Anthropic 变量，原始文件和脱敏输出均不落令牌。"""

    config_path = _write(
        tmp_path / "config.yaml",
        """llm:
  channels:
    - id: claude-local
      protocol: anthropic_messages
      base_url: ${ANTHROPIC_BASE_URL}
      api_key: ${ANTHROPIC_AUTH_TOKEN}
      model: ${ANTHROPIC_MODEL}
""",
    )
    secret = "anthropic-yaml-secret-round117"
    env = {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:13000",
        "ANTHROPIC_AUTH_TOKEN": secret,
        "ANTHROPIC_MODEL": "claude-yaml-test",
    }
    selected = load_configuration(config_path, defaults={}, environment=env)
    channel = selected.values["llm"]["channels"][0]
    assert channel["api_key"] == secret
    assert secret not in config_path.read_text(encoding="utf-8")
    assert secret not in str(redact_secrets(selected.values))


def test_configuration_environment_channel_updates_channel_id_route_alias(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        "llm:\n  channels: []\n  routing:\n    dialogue:\n      channel_id: primary\n",
    )
    env = _environment(tmp_path)
    env.update(
        {
            "MEAPET_API_BASE": "http://127.0.0.1:11434/v1",
            "MEAPET_MODEL": "demo-model",
            "MEAPET_CHANNEL_ID": "local-gateway",
        }
    )

    selected = discover_configuration(
        config_path,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )

    assert selected.values["llm"]["routing"]["dialogue"]["channel"] == "local-gateway"


def test_configuration_environment_channel_replaces_stale_empty_route(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        "llm:\n  channels: []\n  routing:\n    dialogue:\n      channel: stale-channel\n",
    )
    env = _environment(tmp_path)
    env.update(
        {
            "MEAPET_API_BASE": "http://127.0.0.1:11434/v1",
            "MEAPET_MODEL": "demo-model",
            "MEAPET_CHANNEL_ID": "local-gateway",
        }
    )

    selected = discover_configuration(
        config_path,
        environment=env,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )

    assert selected.values["llm"]["routing"]["dialogue"]["channel"] == "local-gateway"


def test_configuration_wizard_writes_api_key_reference_without_secret(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    wizard = ConfigurationWizard(config_path)
    wizard.configure_channel(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://api.example.invalid/v1",
        model="demo-model",
        api_key_env="MEAPET_TEST_KEY",
    )
    text = config_path.read_text(encoding="utf-8")
    assert "${MEAPET_TEST_KEY}" in text
    assert 'api_key: ""' not in text
    loaded = discover_configuration(
        config_path,
        environment={"MEAPET_TEST_KEY": "secret-value"},
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert loaded.values["llm"]["channels"][0]["api_key"] == "secret-value"
    assert loaded.values["llm"]["channels"][0]["capabilities"] == ["streaming", "tools"]


def test_configuration_wizard_can_explicitly_persist_plaintext_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    ConfigurationWizard(config_path).configure_channel(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://api.example.invalid/v1",
        model="demo-model",
        api_key="stored-secret",
        persist_api_key=True,
    )
    text = config_path.read_text(encoding="utf-8")
    assert "stored-secret" in text
    assert "${MEAPET_API_KEY}" not in text
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_configuration_wizard_edit_preserves_existing_plaintext_api_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    wizard = ConfigurationWizard(config_path)
    wizard.configure_channel(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://api.example.invalid/v1",
        model="old-model",
        api_key="stored-secret",
        persist_api_key=True,
    )
    wizard.configure_channel(
        channel_id="primary",
        protocol="openai_chat",
        base_url="https://api.example.invalid/v1",
        model="new-model",
    )
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["llm"]["channels"][0]["api_key"] == "stored-secret"
    assert saved["llm"]["channels"][0]["model"] == "new-model"


def test_configuration_wizard_switching_to_ollama_removes_old_credentials(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://gateway.example.invalid/v1
      model: old
      api_key: stored-secret
      api_key_env: OLD_KEY
""",
    )
    ConfigurationWizard(config_path).configure_channel(
        channel_id="primary",
        protocol="ollama_chat",
        base_url="http://127.0.0.1:11434/v1",
        model="llama3",
    )
    channel = yaml.safe_load(config_path.read_text(encoding="utf-8"))["llm"]["channels"][0]
    assert "api_key" not in channel
    assert "api_key_env" not in channel


def test_configuration_wizard_rejects_invalid_plaintext_persistence_flag(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="persist_api_key"):
        ConfigurationWizard(tmp_path / "config.yaml").configure_channel(
            channel_id="primary",
            protocol="openai_chat",
            base_url="https://api.example.invalid/v1",
            model="demo-model",
            api_key="stored-secret",
            persist_api_key="not-a-boolean",  # type: ignore[arg-type]
        )


def test_configuration_wizard_default_tts_points_to_project_resources(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    wizard = ConfigurationWizard(config_path)
    wizard.create()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["tts"]["enabled"] is True
    assert raw["tts"]["backend"] == "gpt_sovits_stdio"
    assert Path(raw["tts"]["ref_dir"]).is_absolute()


def test_example_channel_uses_runtime_vision_capability_key() -> None:
    loaded = load_configuration(Path("config/app.example.yaml"), defaults={})
    channels = loaded.values["llm"]["channels"]
    assert channels and "vision" in channels[0]["capabilities"]


def test_default_and_example_tool_groups_include_new_desktop_contracts() -> None:
    defaults = loader.default_configuration_values()
    groups = defaults["tools"]["groups"]
    assert groups["desktop_control"] == ["desktop:click_at"]
    assert groups["desktop_automation"] == ["desktop:automation_batch"]
    assert "system:module_status" in groups["system"]

    loaded = load_configuration(Path("config/app.example.yaml"), defaults={})
    example_tools = loaded.values["tools"]
    assert example_tools["groups"]["desktop_control"] == ["desktop:click_at"]
    assert example_tools["groups"]["desktop_automation"] == ["desktop:automation_batch"]
    assert "desktop:context_snapshot" in groups["desktop_observation"]
    assert "desktop:context_snapshot" in example_tools["groups"]["desktop_observation"]
    assert "desktop:ocr" in groups["desktop_observation"]
    assert "desktop:ocr" in example_tools["groups"]["desktop_observation"]
    assert "desktop_control" in example_tools["active_groups"]


def test_defaults_select_complete_owned_tts_layout(tmp_path: Path) -> None:
    resources = tmp_path / "resources"
    engine = tmp_path / "temp" / "third_party" / "GPT-SoVITS"
    python = engine / ".venv" / "bin" / "python"
    config = engine / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    gpt = resources / "models" / "GPT_weights" / "mea_pro-e50.ckpt"
    sovits = resources / "models" / "SoVITS_weights" / "mea_pro_e24_s13704.pth"
    for path in (python, config, gpt, sovits):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"available")
    python.chmod(0o755)

    defaults = loader.default_configuration_values(resource_root=resources)

    assert defaults["tts"]["engine_root"] == str(engine)
    assert defaults["tts"]["python_executable"] == str(python)
    assert defaults["tts"]["gpt_path"] == str(gpt)
    assert defaults["tts"]["sovits_path"] == str(sovits)
    assert defaults["tts"]["startup_timeout_seconds"] == 300.0


def test_defaults_select_windows_owned_tts_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows 独立环境使用 ``Scripts/python.exe`` 而不是 POSIX 入口。"""

    monkeypatch.setattr(sys, "platform", "win32")
    resources = tmp_path / "resources"
    engine = tmp_path / "temp" / "third_party" / "GPT-SoVITS"
    python = engine / ".venv" / "Scripts" / "python.exe"
    config = engine / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    for path in (
        python,
        config,
        resources / "models" / "GPT_weights" / "mea_pro-e50.ckpt",
        resources / "models" / "SoVITS_weights" / "mea_pro_e24_s13704.pth",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"available")

    defaults = loader.default_configuration_values(resource_root=resources)

    assert defaults["tts"]["engine_root"] == str(engine)
    assert defaults["tts"]["python_executable"] == str(python)
    assert defaults["asr"]["python_executable"] == str(python)


def test_defaults_do_not_use_windows_pythonw_or_posix_venv_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows worker 只能使用标准 ``Scripts/python.exe`` 入口。"""

    monkeypatch.setattr(sys, "platform", "win32")
    resources = tmp_path / "resources"
    engine = tmp_path / "temp" / "third_party" / "GPT-SoVITS"
    config = engine / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    pythonw = engine / ".venv" / "Scripts" / "pythonw.exe"
    posix_python = engine / ".venv" / "bin" / "python"
    for path in (config, pythonw, posix_python):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"available")

    defaults = loader.default_configuration_values(resource_root=resources)

    assert defaults["tts"]["engine_root"] == ""
    assert defaults["tts"]["python_executable"] == ""
    assert defaults["asr"]["python_executable"] == ""


def test_configuration_wizard_converts_legacy_api_key_to_transient_env_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    # 先登记一个空值，确保 configure_channel 的兼容入口在测试结束时
    # 不会把临时密钥泄漏到后续子进程环境。
    monkeypatch.setenv("MEAPET_API_KEY", "")
    ConfigurationWizard(config_path).configure_channel(
        channel_id="legacy",
        protocol="openai_chat",
        base_url="https://api.example.invalid/v1",
        model="demo-model",
        api_key="transient-secret",
    )
    text = config_path.read_text(encoding="utf-8")
    assert "${MEAPET_API_KEY}" in text
    assert "transient-secret" not in text


def test_configuration_wizard_patch_never_writes_expanded_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n  channels:\n    - id: primary\n      protocol: openai_chat\n"
        "      base_url: https://api.example.invalid/v1\n      model: old\n"
        "      api_key: ${MEAPET_PATCH_KEY}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MEAPET_PATCH_KEY", "expanded-only-at-runtime")
    ConfigurationWizard(config_path).patch(
        {
            "llm.channels": [
                {
                    "id": "primary",
                    "protocol": "openai_chat",
                    "base_url": "https://api.example.invalid/v1",
                    "model": "new",
                    "api_key": "expanded-only-at-runtime",
                }
            ]
        }
    )
    text = config_path.read_text(encoding="utf-8")
    assert "${MEAPET_PATCH_KEY}" in text
    assert "expanded-only-at-runtime" not in text


def test_configuration_wizard_patch_can_save_missing_api_key_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEAPET_PATCH_MISSING_KEY", raising=False)
    config_path = _write(
        tmp_path / "config.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://api.example.invalid/v1
      model: old
      api_key: ${MEAPET_PATCH_MISSING_KEY}
""",
    )
    loaded = ConfigurationWizard(config_path).patch(
        {
            "llm.channels": [
                {
                    "id": "primary",
                    "protocol": "openai_chat",
                    "base_url": "https://api.example.invalid/v1",
                    "model": "new",
                    "api_key": "${MEAPET_PATCH_MISSING_KEY}",
                }
            ]
        }
    )
    assert loaded.values["llm"]["channels"][0]["api_key"] == "${MEAPET_PATCH_MISSING_KEY}"
    assert loaded.source_digest == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert "model: new" in config_path.read_text(encoding="utf-8")


def test_configuration_wizard_channel_update_preserves_advanced_fields_and_key(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://gateway.example.invalid/v1
      model: old-model
      api_key: ${PRIMARY_KEY}
      endpoint: /responses
      headers:
        X-Trace: keep
      retry:
        max_attempts: 3
      request_body_template:
        temperature: 0.2
""",
    )
    wizard = ConfigurationWizard(config_path)
    wizard.configure_channel(
        channel_id="primary",
        protocol="openai_responses",
        base_url="https://gateway.example.invalid/v1",
        model="new-model",
    )
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    channel = saved["llm"]["channels"][0]
    assert channel["model"] == "new-model"
    assert channel["protocol"] == "openai_responses"
    assert channel["api_key"] == "${PRIMARY_KEY}"
    assert channel["endpoint"] == "/responses"
    assert channel["headers"] == {"X-Trace": "keep"}
    assert channel["retry"]["max_attempts"] == 3
    assert channel["request_body_template"] == {"temperature": 0.2}


def test_configuration_wizard_editing_backup_does_not_change_dialogue_route(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        """llm:
  channels:
    - id: primary
      protocol: openai_chat
      base_url: https://primary.example.invalid/v1
      model: primary-model
    - id: backup
      protocol: openai_chat
      base_url: https://backup.example.invalid/v1
      model: old-backup
  routing:
    dialogue:
      channel: primary
""",
    )
    ConfigurationWizard(config_path).configure_channel(
        channel_id="backup",
        protocol="openai_chat",
        base_url="https://backup.example.invalid/v1",
        model="new-backup",
    )
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["llm"]["routing"]["dialogue"]["channel"] == "primary"
    assert saved["llm"]["channels"][1]["model"] == "new-backup"


def test_configuration_wizard_protocol_change_removes_stale_adapter_metadata(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        """llm:
  channels:
    - id: primary
      preset: openai
      adapter: openai_chat
      protocol: openai_chat
      base_url: https://gateway.example.invalid/v1
      model: old-model
""",
    )
    ConfigurationWizard(config_path).configure_channel(
        channel_id="primary",
        protocol="anthropic_messages",
        base_url="https://gateway.example.invalid",
        model="claude-local",
    )
    channel = yaml.safe_load(config_path.read_text(encoding="utf-8"))["llm"]["channels"][0]
    assert channel["protocol"] == "anthropic_messages"
    assert "preset" not in channel
    assert "adapter" not in channel


def test_configuration_wizard_rejects_invalid_protocol_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    with pytest.raises(SystemExit) as raised:
        wizard_main(
            [
                "--config",
                str(config_path),
                "channel",
                "add",
                "--id",
                "primary",
                "--protocol",
                "invalid_protocol",
                "--base-url",
                "https://example.invalid/v1",
                "--model",
                "demo-model",
            ]
        )
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "unsupported channel protocol" in error
    assert "Traceback" not in error
    assert not config_path.exists()


def test_configuration_wizard_rejects_non_mapping_without_overwriting(tmp_path: Path) -> None:
    config_path = _write(tmp_path / "config.yaml", "- invalid-root\n")
    wizard = ConfigurationWizard(config_path)
    with pytest.raises(ValueError, match="configuration root must be a mapping"):
        wizard.configure_channel(
            channel_id="primary",
            protocol="openai_chat",
            base_url="https://example.invalid/v1",
            model="demo-model",
        )
    assert config_path.read_text(encoding="utf-8") == "- invalid-root\n"


def test_load_configuration_rejects_recursive_yaml_alias(tmp_path: Path) -> None:
    config_path = _write(tmp_path / "config.yaml", "value: &loop\n  nested: *loop\n")
    with pytest.raises(ConfigurationError, match="recursive"):
        load_configuration(config_path, defaults={})


def test_configuration_wizard_remove_clears_channel_id_route_alias(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        "\n".join(
            (
                "llm:",
                "  channels:",
                "    - id: primary",
                "      protocol: openai_chat",
                "      base_url: https://example.invalid/v1",
                "      model: demo",
                "  routing:",
                "    dialogue:",
                "      channel_id: primary",
                "",
            )
        ),
    )

    ConfigurationWizard(config_path).remove_channel("primary")

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert loaded["llm"]["channels"] == []
    assert "dialogue" not in loaded["llm"]["routing"]


def test_explicit_empty_environment_does_not_read_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MEAPET_TEST_HOST_SECRET", "must-not-be-read")
    config_path = _write(
        tmp_path / "config.yaml",
        "\n".join(
            (
                "llm:",
                "  channels:",
                "    - id: primary",
                "      protocol: openai_chat",
                "      base_url: https://example.invalid/v1",
                "      api_key: ${MEAPET_TEST_HOST_SECRET}",
                "      model: demo",
                "",
            )
        ),
    )

    with pytest.raises(ConfigurationError, match="environment variable is not set"):
        load_configuration(config_path, environment={})


def test_expand_environment_values_is_fail_closed_before_save() -> None:
    values = {"llm": {"channels": [{"api_key": "${MEAPET_TEST_KEY}"}]}}
    expanded = expand_environment_values(values, environment={"MEAPET_TEST_KEY": "resolved"})
    assert expanded["llm"]["channels"][0]["api_key"] == "resolved"
    with pytest.raises(ConfigurationError, match="environment variable is not set"):
        expand_environment_values(values, environment={})


def test_windows_configuration_expands_home_from_userprofile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows 只有 USERPROFILE 时兼容现有 HOME 示例变量。"""

    monkeypatch.setattr(sys, "platform", "win32")
    loaded = load_configuration(
        Path("config/app.example.yaml"),
        defaults={},
        environment={"HOME": "", "USERPROFILE": "C:/Users/demo"},
    )
    assert loaded.values["asr"]["model_path"].startswith("C:/Users/demo/")


def test_example_configuration_does_not_pin_posix_worker_entries() -> None:
    """示例配置把解释器入口交给平台默认发现器。"""

    raw = yaml.safe_load(Path("config/app.example.yaml").read_text(encoding="utf-8"))

    assert "python_executable" not in raw["tts"]
    assert "python_executable" not in raw["asr"]


def test_configuration_wizard_show_redacts_missing_environment_secret(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write(
        tmp_path / "config.yaml",
        "llm:\n  channels:\n    - id: primary\n      api_key: ${MEAPET_MISSING_KEY}\n",
    )

    assert wizard_main(["--config", str(config_path), "show"]) == 0
    output = capsys.readouterr().out
    assert "***" in output
    assert "MEAPET_MISSING_KEY" not in output


def test_default_resource_root_prefers_project_resources_over_legacy_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    (tmp_path / "resources").mkdir()
    (tmp_path / "temp" / "resources").mkdir(parents=True)
    selected = discover_configuration(
        environment=_environment(tmp_path),
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )
    assert selected.values["rendering"]["resource_root"] == str((tmp_path / "resources").resolve())


def test_default_resource_root_uses_project_root_when_cwd_is_external(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    (project / "resources").mkdir(parents=True)
    (project / "temp" / "resources").mkdir(parents=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(loader, "_module_project_root", lambda: project)
    selected = discover_configuration(
        environment=_environment(tmp_path),
        cwd=run_dir,
        home=tmp_path / "home",
    )
    assert selected.values["rendering"]["resource_root"] == str((project / "resources").resolve())


def test_relative_project_root_is_anchored_to_supplied_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    project = run_dir / "project"
    (project / "resources").mkdir(parents=True)
    monkeypatch.setattr(loader, "_module_project_root", lambda: None)
    selected = discover_configuration(
        environment=_environment(tmp_path),
        cwd=run_dir,
        project_root="project",
        home=tmp_path / "home",
    )
    assert selected.values["rendering"]["resource_root"] == str((project / "resources").resolve())


def test_initialize_database_creates_parent_and_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    database_path = tmp_path / "nested" / "state" / "meapet.sqlite3"
    _write(config_path, "storage:\n  database: nested/state/meapet.sqlite3\n")
    selected = discover_configuration(
        config_path,
        cwd=tmp_path,
        project_root=tmp_path,
        home=tmp_path / "home",
    )

    initialized = initialize_database(selected)
    assert initialized == database_path.resolve()
    assert database_path.is_file()
    import sqlite3

    connection = sqlite3.connect(database_path)
    try:
        assert connection.execute("SELECT version FROM memory_schema_version").fetchone()[0] == 7
    finally:
        connection.close()


def test_cli_validation_does_not_print_api_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "do-not-print-this-key"
    config_path = _write(
        tmp_path / "invalid.yaml",
        "\n".join(
            (
                "llm:",
                "  channels:",
                "    - id: main",
                "      protocol: unsupported",
                f"      api_key: {secret}",
                "",
            )
        ),
    )
    with pytest.raises(SystemExit):
        cli_main(["--config", str(config_path), "--validate"])
    output = capsys.readouterr()
    assert secret not in output.err
    assert secret not in output.out


def test_cli_validation_reports_resource_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resource_root = tmp_path / "resources"
    resource_root.mkdir()
    config_path = _write(
        tmp_path / "warnings.yaml",
        f"rendering:\n  resource_root: {resource_root}\n",
    )

    assert cli_main(["--config", str(config_path), "--validate"]) == 0
    output = capsys.readouterr()
    assert "resource_warning: no usable Live2D model3 descriptor was found" in output.err
    assert "resource_warning: no WebP sprite fallback was found" in output.err
    assert "model_channel_ready: False" in output.out
    assert "rendering_backend_requested: auto" in output.out
    assert "rendering_backend_selected: unavailable" in output.out
    assert "rendering_backend_available: False" in output.out
    assert "MEAPET_API_BASE" in output.err


def test_cli_validation_rejects_explicit_unavailable_renderer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    resource_root = tmp_path / "resources"
    resource_root.mkdir()
    config_path = _write(
        tmp_path / "vllank.yaml",
        f"rendering:\n  backend: vllank\n  resource_root: {resource_root}\n",
    )

    assert cli_main(["--config", str(config_path), "--validate"]) == 2
    output = capsys.readouterr()
    assert "rendering_backend_requested: vllank" in output.out
    assert "rendering_backend_available: False" in output.out
    assert "Vulkan rendering is unavailable" in output.out


def test_main_without_arguments_discovers_default_and_initializes_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _environment(tmp_path)
    monkeypatch.chdir(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MEAPET_CONFIG", raising=False)
    # 阻止源码工作树的示例模板参与本测试，模拟已安装运行环境中的空目录。
    monkeypatch.setattr(loader, "_module_project_root", lambda: None)

    calls: list[Path] = []

    def fake_run(configuration, _inventory):
        calls.append(configuration.path)
        return 0

    import gui.qt6.app as qt_app

    monkeypatch.setattr(qt_app, "run", fake_run)
    assert cli_main([]) == 0
    config_path = tmp_path / "xdg-config" / "meapet" / "config.yaml"
    database_path = tmp_path / "xdg-data" / "meapet" / "meapet.sqlite3"
    assert calls == [config_path.resolve()]
    assert config_path.is_file()
    assert database_path.is_file()
