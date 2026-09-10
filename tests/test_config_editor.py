from __future__ import annotations

from pathlib import Path

import pytest

from config.editor import (
    ConfigurationConflictError,
    ConfigurationEditSession,
    atomic_write_configuration,
    editor_snapshot,
    merge_editor_values,
    parse_editor_yaml,
    prepare_persisted_values,
)
from config.loader import ConfigurationError, configuration_source_digest


def test_editor_snapshot_preserves_secrets_during_merge() -> None:
    original = {
        "llm": {"channels": [{"id": "primary", "api_key": "keep-me", "model": "old"}]},
        "tts": {"enabled": False},
    }
    edited = editor_snapshot(original)
    edited["llm"]["channels"][0]["model"] = "new"
    merged = merge_editor_values(original, edited)
    assert merged["llm"]["channels"][0]["api_key"] == "keep-me"
    assert merged["llm"]["channels"][0]["model"] == "new"


def test_editor_snapshot_keeps_api_key_environment_reference_name_visible() -> None:
    original = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key_env": "OPENAI_API_KEY",
                    "api_key": "${OPENAI_API_KEY}",
                }
            ]
        }
    }
    snapshot = editor_snapshot(original)
    channel = snapshot["llm"]["channels"][0]
    assert channel["api_key_env"] == "OPENAI_API_KEY"
    assert channel["api_key"] == "***"


def test_editor_merge_matches_identity_after_channel_reorder_and_removal() -> None:
    """重排/删除渠道时，脱敏占位符必须跟随渠道身份而不是列表索引。"""

    original = {
        "llm": {
            "channels": [
                {"id": "alpha", "api_key": "alpha-secret", "model": "a"},
                {"id": "beta", "api_key": "beta-secret", "model": "b"},
            ]
        }
    }
    edited = {
        "llm": {
            "channels": [
                {"id": "beta", "api_key": "***", "model": "b-new"},
            ]
        }
    }

    merged = merge_editor_values(original, edited)

    assert merged["llm"]["channels"] == [{"id": "beta", "api_key": "beta-secret", "model": "b-new"}]


def test_preserve_secret_values_matches_identity_after_reorder() -> None:
    """运行时展开配置刷新到控制台时，环境引用也必须按渠道身份恢复。"""

    source = {
        "llm": {
            "channels": [
                {"id": "alpha", "api_key": "${ALPHA_KEY}"},
                {"id": "beta", "api_key": "${BETA_KEY}"},
            ]
        }
    }
    expanded = {
        "llm": {
            "channels": [
                {"id": "beta", "api_key": "beta-secret"},
                {"id": "alpha", "api_key": "alpha-secret"},
            ]
        }
    }

    from config.editor import preserve_secret_values

    persisted = preserve_secret_values(source, expanded)

    assert persisted["llm"]["channels"] == [
        {"id": "beta", "api_key": "${BETA_KEY}"},
        {"id": "alpha", "api_key": "${ALPHA_KEY}"},
    ]


def test_preserve_secret_values_keeps_environment_reference() -> None:
    from config.editor import preserve_secret_values

    source = {"llm": {"api_key": "${MEAPET_TEST_KEY}"}}
    expanded = {"llm": {"api_key": "plain-secret", "model": "new"}}
    persisted = preserve_secret_values(source, expanded)
    assert persisted == {"llm": {"api_key": "${MEAPET_TEST_KEY}", "model": "new"}}


def test_prepare_persisted_values_preserves_old_reference_and_accepts_new_reference() -> None:
    source = {"llm": {"channels": [{"id": "primary", "api_key": "${OLD_KEY}"}]}}
    expanded = {"llm": {"channels": [{"id": "primary", "api_key": "old-secret"}]}}
    persisted = prepare_persisted_values(source, expanded)
    assert persisted == source

    edited = {"llm": {"channels": [{"id": "primary", "api_key": "${NEW_KEY}"}]}}
    assert prepare_persisted_values(source, edited) == edited

    cleared = {"llm": {"channels": [{"id": "primary", "api_key": ""}]}}
    assert prepare_persisted_values(source, cleared) == {"llm": {"channels": [{"id": "primary"}]}}


def test_prepare_persisted_values_rejects_plaintext_secret_without_source_reference() -> None:
    with pytest.raises(ConfigurationError, match="environment variable references"):
        prepare_persisted_values({}, {"llm": {"api_key": "plain-secret"}})


def test_prepare_persisted_values_allows_explicit_plaintext_persistence() -> None:
    persisted = prepare_persisted_values(
        {},
        {"llm": {"channels": [{"id": "primary", "api_key": "plain-secret"}]}},
        allow_plaintext_secrets=True,
    )
    assert persisted["llm"]["channels"][0]["api_key"] == "plain-secret"


def test_prepare_persisted_values_rejects_placeholder_inside_plaintext_secret() -> None:
    with pytest.raises(ConfigurationError, match="environment placeholders"):
        prepare_persisted_values(
            {},
            {"llm": {"channels": [{"id": "primary", "api_key": "secret-${TOKEN}"}]}},
            allow_plaintext_secrets=True,
        )


def test_prepare_persisted_values_keeps_existing_plaintext_when_redacted() -> None:
    source = {"llm": {"api_key": "stored-secret"}}
    edited = {"llm": {"api_key": "***"}}
    persisted = prepare_persisted_values(
        source,
        edited,
        allow_plaintext_secrets=True,
    )
    assert persisted == source


def test_prepare_persisted_values_keeps_existing_plaintext_without_opt_in() -> None:
    source = {"llm": {"api_key": "stored-secret"}}
    assert prepare_persisted_values(source, {"llm": {"api_key": "***"}}) == source


def test_prepare_persisted_values_keeps_expanded_environment_reference_on_auto_save() -> None:
    source = {"llm": {"channels": [{"id": "primary", "api_key": "${PRIMARY_KEY}"}]}}
    expanded = {"llm": {"channels": [{"id": "primary", "api_key": "expanded-secret"}]}}
    persisted = prepare_persisted_values(
        source,
        expanded,
        allow_plaintext_secrets=True,
    )
    assert persisted == source


def test_prepare_persisted_values_replaces_reference_when_explicitly_requested() -> None:
    source = {"llm": {"channels": [{"id": "primary", "api_key": "${PRIMARY_KEY}"}]}}
    edited = {
        "llm": {"channels": [{"id": "primary", "api_key": "stored-secret", "api_key_env": ""}]}
    }
    persisted = prepare_persisted_values(
        source,
        edited,
        allow_plaintext_secrets=True,
    )
    assert persisted["llm"]["channels"][0]["api_key"] == "stored-secret"


def test_plaintext_persistence_marker_survives_later_automatic_save() -> None:
    source = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "stored-secret",
                    "api_key_env": "",
                    "model": "old",
                }
            ]
        }
    }
    edited = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "***",
                    "api_key_env": "",
                    "model": "new",
                }
            ]
        }
    }
    persisted = prepare_persisted_values(source, edited)
    assert persisted["llm"]["channels"][0]["api_key"] == "stored-secret"
    assert persisted["llm"]["channels"][0]["model"] == "new"


def test_channel_environment_metadata_follows_reference_change() -> None:
    source = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "${OLD_KEY}",
                    "api_key_env": "OLD_KEY",
                }
            ]
        }
    }
    edited = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "***",
                    "api_key_env": "NEW_KEY",
                }
            ]
        }
    }
    persisted = prepare_persisted_values(source, edited)
    channel = persisted["llm"]["channels"][0]
    assert channel["api_key"] == "${NEW_KEY}"
    assert channel["api_key_env"] == "NEW_KEY"


def test_channel_environment_metadata_never_coexists_with_plaintext_secret() -> None:
    source = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "stored-secret",
                    "api_key_env": "",
                }
            ]
        }
    }
    edited = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "***",
                    "api_key_env": "NEW_KEY",
                }
            ]
        }
    }
    channel = prepare_persisted_values(source, edited)["llm"]["channels"][0]
    assert channel["api_key"] == "${NEW_KEY}"
    assert channel["api_key_env"] == "NEW_KEY"


def test_clearing_channel_key_removes_environment_metadata() -> None:
    source = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "${OLD_KEY}",
                    "api_key_env": "OLD_KEY",
                }
            ]
        }
    }
    persisted = prepare_persisted_values(
        source,
        {"llm": {"channels": [{"id": "primary", "api_key": ""}]}},
    )
    channel = persisted["llm"]["channels"][0]
    assert "api_key" not in channel
    assert "api_key_env" not in channel


def test_metadata_only_channel_reference_round_trips_without_environment_value() -> None:
    source = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "",
                    "api_key_env": "META_KEY",
                    "model": "old",
                }
            ]
        }
    }
    edited = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "api_key": "",
                    "api_key_env": "META_KEY",
                    "model": "new",
                }
            ]
        }
    }
    persisted = prepare_persisted_values(source, edited)
    channel = persisted["llm"]["channels"][0]
    assert channel["api_key"] == "${META_KEY}"
    assert channel["api_key_env"] == "META_KEY"


def test_persisted_configuration_file_never_receives_expanded_secret(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    source = {
        "llm": {"channels": [{"id": "primary", "protocol": "openai_chat", "api_key": "${API_KEY}"}]}
    }
    expanded = {
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "protocol": "openai_chat",
                    "api_key": "expanded-secret-value",
                }
            ]
        }
    }
    atomic_write_configuration(path, prepare_persisted_values(source, expanded))
    text = path.read_text(encoding="utf-8")
    assert "${API_KEY}" in text
    assert "expanded-secret-value" not in text


def test_parse_editor_yaml_rejects_non_mapping_and_forbidden_keys() -> None:
    with pytest.raises(ConfigurationError, match="root must be a mapping"):
        parse_editor_yaml("- item")
    with pytest.raises(ConfigurationError, match="forbidden"):
        parse_editor_yaml("__proto__: bad")


def test_validate_editor_values_rejects_non_finite_numbers() -> None:
    from config.editor import validate_editor_values

    with pytest.raises(ConfigurationError, match="non-finite"):
        validate_editor_values({"rendering": {"sprite_scale": float("nan")}})


def test_atomic_write_configuration_replaces_file_and_keeps_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("old: true\n", encoding="utf-8")
    path.chmod(0o640)
    atomic_write_configuration(path, {"app": {"name": "MeaPet"}})
    assert path.read_text(encoding="utf-8") == "app:\n  name: MeaPet\n"
    assert path.stat().st_mode & 0o777 == 0o640
    assert not tuple(tmp_path.glob(".config.yaml.*.tmp"))


def test_atomic_write_configuration_restricts_secret_file_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  api_key: ${API_KEY}\n", encoding="utf-8")
    path.chmod(0o644)
    atomic_write_configuration(path, {"llm": {"api_key": "${API_KEY}"}})
    assert path.stat().st_mode & 0o777 == 0o600


def test_atomic_write_configuration_rejects_external_revision(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("app:\n  name: Original\n", encoding="utf-8")
    baseline = configuration_source_digest(path)
    path.write_text("app:\n  name: External\n", encoding="utf-8")

    with pytest.raises(ConfigurationConflictError, match="changed on disk"):
        atomic_write_configuration(
            path,
            {"app": {"name": "Editor"}},
            expected_digest=baseline,
        )

    assert path.read_text(encoding="utf-8") == "app:\n  name: External\n"
    assert not tuple(tmp_path.glob(".config.yaml.*.tmp"))


def test_parse_editor_yaml_rejects_recursive_aliases() -> None:
    with pytest.raises(ConfigurationError, match="recursive"):
        parse_editor_yaml("value: &loop\n  nested: *loop\n")


def test_edit_session_requires_explicit_confirmation(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    session = ConfigurationEditSession(path, {"app": {"name": "MeaPet"}})
    with pytest.raises(ConfigurationError, match="explicit confirmation"):
        session.save({"app": {"name": "New"}})
    session.save({"app": {"name": "New"}}, confirmed=True)
    assert "New" in path.read_text(encoding="utf-8")


def test_edit_session_restricts_mode_when_existing_plaintext_secret_is_kept(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    source = {"llm": {"api_key": "stored-secret", "model": "old"}}
    atomic_write_configuration(path, source, mode=0o644)
    session = ConfigurationEditSession(path, source)
    session.save(
        {"llm": {"api_key": "***", "model": "new"}},
        confirmed=True,
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert "stored-secret" in path.read_text(encoding="utf-8")


def test_edit_session_rejects_external_write_and_keeps_new_baseline(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    source = {"app": {"name": "Original"}}
    atomic_write_configuration(path, source)
    session = ConfigurationEditSession(path, source)
    path.write_text("app:\n  name: External\n", encoding="utf-8")

    with pytest.raises(ConfigurationConflictError, match="changed on disk"):
        session.save({"app": {"name": "Editor"}}, confirmed=True)

    assert path.read_text(encoding="utf-8") == "app:\n  name: External\n"


def test_new_edit_session_does_not_overwrite_externally_created_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    session = ConfigurationEditSession(path, {"app": {"name": "Draft"}})
    path.write_text("app:\n  name: External\n", encoding="utf-8")

    with pytest.raises(ConfigurationConflictError, match="changed on disk"):
        session.save({"app": {"name": "Editor"}}, confirmed=True)

    assert path.read_text(encoding="utf-8") == "app:\n  name: External\n"
