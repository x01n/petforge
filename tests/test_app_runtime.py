from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sqlite3
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from app.__main__ import main as cli_main
from app.loop import RuntimeLoop
from app.runtime import (
    DispatchingDesktopPlatform,
    MutablePetController,
    build_runtime,
    validate_runtime_configuration,
)
from config.loader import ConfigurationError, LoadedConfiguration, default_configuration_values
from config.resources import inspect_resources
from core.asr import ASRHealth
from core.rendering import RendererReadyState, RendererReadyStatus
from core.tts.contracts import EngineHealth
from gui.qt6.app import (
    _apply_conversation_operation_receipt,
    _build_model_test_channel,
    _configuration_restart_sections,
    _conversation_operation_receipt,
    _decode_public_expression_request,
    _decode_public_motion_request,
    _interaction_status_with_local_feedback,
    _is_missing_api_key_reference,
    _model_channel_diagnostic_summary,
    _model_channel_status,
    _prepare_persisted_configuration,
    _public_model_connection_future,
    _public_model_connection_result,
    _refresh_console_action_capabilities,
    _resolve_renderer_action,
    _restart_child_arguments,
    _restart_ready_receipt_matches,
    _runtime_module_test,
    _schedule_pet_affection,
    _speech_queue_blocked,
    _write_restart_ready_receipt,
)
from gui.qt6.dispatcher import _cancel_pending_futures
from services.tools.permissions import PermissionDecision
from services.tools.types import RiskLevel, ToolCallContext, ToolSpec


def test_missing_api_key_reference_is_the_only_unresolved_secret_case_allowed() -> None:
    values = {
        "llm": {
            "channels": [
                {"id": "primary", "api_key": "${MEAPET_TEST_KEY}"},
            ]
        }
    }
    error = ConfigurationError("environment variable is not set: MEAPET_TEST_KEY")
    assert _is_missing_api_key_reference(values, error) is True
    assert (
        _is_missing_api_key_reference(
            {"llm": {"channels": [{"id": "primary", "base_url": "${MEAPET_URL}"}]}},
            ConfigurationError("environment variable is not set: MEAPET_URL"),
        )
        is False
    )
    assert (
        _is_missing_api_key_reference(
            values,
            ConfigurationError("configuration is invalid"),
        )
        is False
    )
    assert (
        _is_missing_api_key_reference(
            {"llm": {"channels": [{"id": "primary", "token": "${MEAPET_TEST_KEY}"}]}},
            ConfigurationError("environment variable is not set: MEAPET_TEST_KEY"),
        )
        is True
    )


def test_restart_receipt_requires_matching_pid_and_renderer(tmp_path: Path) -> None:
    marker = tmp_path / "restart-ready.json"
    marker.write_bytes(b"")

    _write_restart_ready_receipt(
        marker,
        ready_status=RendererReadyStatus(
            backend="opengl",
            state=RendererReadyState.READY,
            model_ready=True,
            geometry_valid=True,
            frame_visible=True,
            alpha_nonempty=True,
            actual_api="opengl",
        ),
    )

    assert _restart_ready_receipt_matches(
        marker,
        process_id=os.getpid(),
        expected_backend="opengl",
    )
    assert _restart_ready_receipt_matches(
        marker,
        process_id=os.getpid(),
        expected_backend="auto",
    )
    assert not _restart_ready_receipt_matches(
        marker,
        process_id=os.getpid() + 1,
        expected_backend="opengl",
    )
    assert not _restart_ready_receipt_matches(
        marker,
        process_id=os.getpid(),
        expected_backend="vulkan",
    )


def test_restart_arguments_replace_previous_private_marker(tmp_path: Path) -> None:
    marker = tmp_path / "new.json"

    result = _restart_child_arguments(
        [
            "--config",
            "config/app.yaml",
            "--restart-ready-file",
            "/tmp/old.json",
            "--restart-ready-file=/tmp/older.json",
        ],
        marker,
    )

    assert result == [
        "--config",
        "config/app.yaml",
        "--restart-ready-file",
        str(marker),
    ]


def test_restart_receipt_refuses_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"")
    link = tmp_path / "link.json"
    link.symlink_to(target)

    with pytest.raises(OSError, match="regular file"):
        _write_restart_ready_receipt(
            link,
            ready_status=RendererReadyStatus(
                backend="opengl",
                state=RendererReadyState.READY,
                model_ready=True,
                geometry_valid=True,
                frame_visible=True,
                alpha_nonempty=True,
                actual_api="opengl",
            ),
        )


def test_web_structured_action_payloads_decode_capability_tokens() -> None:
    values = {
        ("expression", "cap-expression-0"): "happy",
        ("expression", "cap-expression-1"): "curious",
        ("motion", "cap-motion-0"): "wave",
    }

    def resolve(kind: str, token: object) -> str:
        return values.get((kind, str(token)), "")

    expression = _decode_public_expression_request(
        {
            "expressions": [
                {"name": "cap-expression-0", "weight": 0.7},
                {"name": "cap-expression-1", "weight": 0.3},
            ],
            "mode": "blend",
            "loop": True,
        },
        resolve,
    )
    motion = _decode_public_motion_request(
        {
            "name": "cap-motion-0",
            "duration_seconds": 1.2,
            "transition_seconds": 0.1,
            "parameters": {"ParamBodyAngleX": 4},
        },
        resolve,
    )

    assert [layer.name for layer in expression.expressions] == ["happy", "curious"]
    assert expression.mode == "blend"
    assert expression.loop is True
    assert motion.name == "wave"
    assert motion.parameters == {"ParamBodyAngleX": 4.0}

    with pytest.raises(ValueError, match="capability"):
        _decode_public_motion_request({"name": "cap-motion-9"}, resolve)


def test_console_persistence_allows_unset_api_key_reference_but_not_unset_url(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n  channels:\n    - id: primary\n"
        "      protocol: openai_chat\n      base_url: https://gateway.invalid/v1\n"
        "      model: demo\n      api_key: ${MEAPET_UNSET_KEY}\n",
        encoding="utf-8",
    )
    persisted, missing = _prepare_persisted_configuration(
        config_path,
        {
            "llm": {
                "channels": [
                    {
                        "id": "primary",
                        "protocol": "openai_chat",
                        "base_url": "https://gateway.invalid/v1",
                        "model": "demo",
                        "api_key": "${MEAPET_UNSET_KEY}",
                    }
                ]
            }
        },
    )
    assert missing is True
    assert persisted["llm"]["channels"][0]["api_key"] == "${MEAPET_UNSET_KEY}"
    with pytest.raises(ConfigurationError, match="environment variable is not set"):
        _prepare_persisted_configuration(
            config_path,
            {
                "llm": {
                    "channels": [
                        {
                            "id": "primary",
                            "protocol": "openai_chat",
                            "base_url": "${MEAPET_UNSET_URL}",
                            "model": "demo",
                            "api_key": "",
                        }
                    ]
                }
            },
        )


def test_console_persistence_does_not_mask_another_missing_environment_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MEAPET_MISSING_URL", raising=False)
    monkeypatch.delenv("MEAPET_MISSING_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "llm:\n  channels:\n    - id: primary\n"
        "      protocol: openai_chat\n      base_url: ${MEAPET_MISSING_URL}\n"
        "      model: demo\n      api_key: ${MEAPET_MISSING_KEY}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="environment variable is not set"):
        _prepare_persisted_configuration(
            config_path,
            {
                "llm": {
                    "channels": [
                        {
                            "id": "primary",
                            "protocol": "openai_chat",
                            "base_url": "${MEAPET_MISSING_URL}",
                            "model": "demo",
                            "api_key": "${MEAPET_MISSING_KEY}",
                        }
                    ]
                }
            },
        )


def test_model_test_channel_reuses_loaded_persisted_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = _build_model_test_channel(
        {
            "channel_id": "primary",
            "protocol": "openai_chat",
            "base_url": "https://gateway.example.invalid/v1",
            "model": "demo",
        },
        {
            "llm": {
                "channels": [
                    {
                        "id": "primary",
                        "protocol": "openai_chat",
                        "base_url": "https://gateway.example.invalid/v1",
                        "model": "demo",
                        "api_key": "stored-secret",
                    }
                ]
            }
        },
    )
    assert getattr(channel, "api_key", "") == "stored-secret"
    assert "stored-secret" not in repr(channel)
    monkeypatch.setenv("MEAPET_MODEL_TEST_KEY", "env-secret")
    referenced = _build_model_test_channel(
        {
            "channel_id": "primary",
            "protocol": "openai_chat",
            "base_url": "https://gateway.example.invalid/v1",
            "model": "demo",
            "api_key_env": "MEAPET_MODEL_TEST_KEY",
        },
        {"llm": {"channels": []}},
    )
    assert getattr(referenced, "api_key", "") == "env-secret"


@pytest.mark.parametrize(
    "phase",
    ("tool_running", "approval_required", "failed", "configuration_required"),
)
def test_speech_queue_yields_to_urgent_interaction_states(phase: str) -> None:
    assert _speech_queue_blocked(phase=phase, tool_status="") is True


def test_speech_queue_yields_to_tool_status_and_resumes_when_clear() -> None:
    assert _speech_queue_blocked(phase="streaming", tool_status="[tool] running") is True
    assert _speech_queue_blocked(phase="streaming", tool_status="") is False


def test_renderer_model_reload_refreshes_console_action_capabilities() -> None:
    class Capabilities:
        expressions = ("new-expression",)
        motions = ("new-motion",)

    class Renderer:
        capabilities = Capabilities()

    class Console:
        def __init__(self) -> None:
            self.values: tuple[tuple[str, ...], tuple[str, ...]] | None = None

        def set_action_capabilities(self, expressions, motions) -> None:
            self.values = (tuple(expressions), tuple(motions))

    console = Console()
    assert _refresh_console_action_capabilities(console, Renderer()) is True
    assert console.values == (("new-expression",), ("new-motion",))


def test_renderer_action_resolution_uses_safe_fallback_for_missing_formal_motion() -> None:
    class RendererController:
        def supports_motion(self, name: str) -> bool:
            return name in {"blink", "idle"}

        def supports_expression(self, _name: str) -> bool:
            return True

    controller = RendererController()
    assert _resolve_renderer_action(controller, "motion", "wave") == "blink"
    assert _resolve_renderer_action(controller, "motion", "blink") == "blink"


def test_renderer_action_resolution_keeps_unknown_capability_pending() -> None:
    class UnknownController:
        def supports_motion(self, _name: str) -> None:
            return None

    assert _resolve_renderer_action(UnknownController(), "motion", "wave") == "wave"


def test_configuration_restart_sections_allows_safe_runtime_hot_reload() -> None:
    before = {
        "llm": {"channels": []},
        "scheduler": {"enabled": True, "tasks": []},
        "tools": {"active_groups": ["pet_control"]},
    }
    assert (
        _configuration_restart_sections(
            before,
            {**before, "llm": {"channels": [{"id": "primary"}]}},
        )
        == ()
    )
    assert (
        _configuration_restart_sections(
            before,
            {**before, "scheduler": {"enabled": True, "tasks": [{"name": "blink"}]}},
        )
        == ()
    )
    assert _configuration_restart_sections(
        before,
        {**before, "tools": {"active_groups": ["pet_control", "scheduler"]}},
    ) == ("tools",)
    assert (
        _configuration_restart_sections(
            before,
            {**before, "rendering": {"model": "live2d/model/alternate.model3.json"}},
        )
        == ()
    )


def test_build_runtime_uses_direct_src_services(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    configuration = LoadedConfiguration(config_path, {})
    inventory = inspect_resources(tmp_path / "resources")
    runtime = build_runtime(configuration, inventory)
    try:
        assert "desktop:observe_foreground" in runtime.registry.identities()
        assert runtime.conversation.memory is runtime.memory
        assert runtime.conversation.affection is runtime.affection
        assert runtime.watcher.poll_seconds >= 0.05
        assert runtime.interaction_state is not None
        assert runtime.interaction_state.snapshot.model.ready is False
    finally:
        asyncio.run(runtime.close())


def test_runtime_wires_and_hot_reloads_memory_policy(tmp_path: Path) -> None:
    database_path = tmp_path / "memory-policy.sqlite3"
    initial = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "storage": {"database": str(database_path)},
            "memory": {
                "enabled": True,
                "recall_limit": 5,
                "context_max_chars": 1800,
                "auto_extract_enabled": True,
                "extract_max_items": 4,
            },
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))

    async def scenario() -> None:
        assert runtime.memory.settings.recall_limit == 5
        assert runtime.memory.settings.context_max_chars == 1800
        assert runtime.memory.settings.auto_extract_enabled is True
        assert runtime.memory.settings.extract_max_items == 4
        result = await runtime.apply_configuration(
            LoadedConfiguration(
                tmp_path / "config.yaml",
                {
                    "storage": {"database": str(database_path)},
                    "memory": {
                        "enabled": False,
                        "recall_limit": 2,
                        "context_max_chars": 900,
                        "auto_extract_enabled": False,
                        "extract_max_items": 0,
                    },
                },
            )
        )
        assert result["status"] == "reloaded"
        assert "memory" in result["applied_sections"]
        assert runtime.memory.enabled is False
        assert runtime.memory.settings.recall_limit == 2
        assert runtime.memory.settings.auto_extract_enabled is False
        assert runtime.memory.settings.extract_max_items == 0
        assert runtime.memory.build_context_prompt("不会读取") == ""

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(runtime.close())


def test_runtime_wires_optional_x11_system_idle_provider_and_hot_reloads(
    tmp_path: Path, monkeypatch
) -> None:
    from gui.platforms.linux import LinuxDesktopPlatform

    monkeypatch.setattr(
        "gui.platforms.linux.probe_x11_idle_seconds",
        lambda _environment: 2.0,
    )
    config_path = tmp_path / "config.yaml"
    database_path = tmp_path / "system-idle.sqlite3"
    platform = LinuxDesktopPlatform({"QT_QPA_PLATFORM": "xcb", "DISPLAY": ":99"})
    runtime = build_runtime(
        LoadedConfiguration(
            config_path,
            {
                "storage": {"database": str(database_path)},
                "scheduler": {
                    "activity": {
                        "system_idle_provider": "x11",
                        "system_idle_threshold_seconds": 5,
                    },
                    "tasks": [],
                    "triggers": [],
                },
            },
        ),
        inspect_resources(tmp_path / "resources"),
        platform=platform,
    )

    async def scenario() -> None:
        assert runtime.activity is not None
        assert runtime.activity.system_idle_provider == "x11"
        assert runtime.activity.is_user_active() is True
        result = await runtime.apply_configuration(
            LoadedConfiguration(
                config_path,
                {
                    "storage": {"database": str(database_path)},
                    "scheduler": {
                        "activity": {
                            "system_idle_provider": "disabled",
                            "system_idle_threshold_seconds": 5,
                        },
                        "tasks": [],
                        "triggers": [],
                    },
                },
            )
        )
        assert result["status"] == "reloaded"
        assert runtime.activity.system_idle_provider == "disabled"
        assert runtime.activity.is_user_active() is False

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(runtime.close())


def test_runtime_wires_windows_system_idle_provider(tmp_path: Path) -> None:
    class Platform:
        backend = "windows"

        def __init__(self) -> None:
            self.calls = 0

        def system_idle_seconds(self) -> float:
            self.calls += 1
            return 2.0

    platform = Platform()
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "windows-idle.yaml",
            {
                "storage": {"database": str(tmp_path / "windows-idle.sqlite3")},
                "scheduler": {
                    "activity": {
                        "system_idle_provider": "windows",
                        "system_idle_threshold_seconds": 5,
                    },
                    "tasks": [],
                    "triggers": [],
                },
            },
        ),
        inspect_resources(tmp_path / "resources"),
        platform=platform,
    )
    try:
        assert runtime.activity is not None
        assert runtime.activity.system_idle_provider == "windows"
        assert runtime.activity.is_user_active() is True
        assert platform.calls == 1
    finally:
        asyncio.run(runtime.close())


def test_runtime_does_not_wire_windows_provider_to_non_windows_backend(
    tmp_path: Path,
) -> None:
    class Platform:
        backend = "x11"

        def system_idle_seconds(self) -> float:
            raise AssertionError("windows provider must not call a non-Windows backend")

    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "windows-mismatch.yaml",
            {
                "storage": {"database": str(tmp_path / "windows-mismatch.sqlite3")},
                "scheduler": {
                    "activity": {
                        "system_idle_provider": "windows",
                        "system_idle_threshold_seconds": 5,
                    },
                    "tasks": [],
                    "triggers": [],
                },
            },
        ),
        inspect_resources(tmp_path / "resources"),
        platform=Platform(),
    )
    try:
        assert runtime.activity is not None
        assert runtime.activity.is_user_active() is False
        assert runtime.activity.status()["system_idle_status"] == "unavailable"
    finally:
        asyncio.run(runtime.close())


def test_runtime_hot_reloads_windows_system_idle_provider(tmp_path: Path) -> None:
    class Platform:
        backend = "windows"

        def __init__(self) -> None:
            self.calls = 0

        def system_idle_seconds(self) -> float:
            self.calls += 1
            return 2.0

    config_path = tmp_path / "windows-reload.yaml"
    database_path = tmp_path / "windows-reload.sqlite3"
    runtime = build_runtime(
        LoadedConfiguration(
            config_path,
            {
                "storage": {"database": str(database_path)},
                "scheduler": {
                    "activity": {"system_idle_provider": "disabled"},
                    "tasks": [],
                    "triggers": [],
                },
            },
        ),
        inspect_resources(tmp_path / "resources"),
        platform=Platform(),
    )
    try:

        async def scenario() -> None:
            assert runtime.activity is not None
            assert runtime.activity.is_user_active() is False
            result = await runtime.apply_configuration(
                LoadedConfiguration(
                    config_path,
                    {
                        "storage": {"database": str(database_path)},
                        "scheduler": {
                            "activity": {
                                "system_idle_provider": "windows",
                                "system_idle_threshold_seconds": 5,
                            },
                            "tasks": [],
                            "triggers": [],
                        },
                    },
                )
            )
            assert result["status"] == "reloaded"
            assert runtime.activity.system_idle_provider == "windows"
            assert runtime.activity.is_user_active() is True

        asyncio.run(scenario())
    finally:
        asyncio.run(runtime.close())


def test_build_runtime_persists_scheduler_tasks_across_restart(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    database_path = tmp_path / "state.sqlite3"
    configuration = LoadedConfiguration(
        config_path,
        {"storage": {"database": str(database_path)}, "scheduler": {"tasks": []}},
    )
    inventory = inspect_resources(tmp_path / "resources")

    first = build_runtime(configuration, inventory)
    try:
        assert first.scheduler.status()["persistence"] == "attached"
        first.scheduler.upsert(
            task_id="persisted-task",
            name="重启后提醒",
            expression="every:1h",
            action={"identity": "pet:speak", "arguments": {"text": "欢迎回来"}},
            owner="profile:session",
        )
    finally:
        asyncio.run(first.close())

    second = build_runtime(configuration, inventory)
    try:
        rows = second.scheduler.list_tasks(owner="profile:session")
        assert [row["task_id"] for row in rows] == ["persisted-task"]
        assert second.scheduler.status()["persistence"] == "attached"
    finally:
        asyncio.run(second.close())


def test_runtime_rejects_invalid_tts_segmentation_limit(tmp_path: Path) -> None:
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {"tts": {"segmentation": {"max_chars": 19}}},
    )
    with pytest.raises(ConfigurationError, match="tts.segmentation.max_chars"):
        validate_runtime_configuration(configuration)


def test_mutable_pet_controller_pauses_behavior_while_ui_is_locked() -> None:
    class Target:
        def is_user_interacting(self):
            return False

    controller = MutablePetController(Target())
    assert controller.is_user_interacting() is False
    controller.set_ui_interaction_locked(True)
    assert controller.is_user_interacting() is True
    controller.set_ui_interaction_locked(False)
    assert controller.is_user_interacting() is False


def test_mutable_pet_controller_invalidates_behavior_once_when_ui_lock_opens() -> None:
    class Target:
        def is_user_interacting(self):
            return False

    interrupts: list[str] = []
    controller = MutablePetController(Target())
    controller.set_interaction_interrupt(lambda: interrupts.append("interrupt"))

    controller.set_ui_interaction_locked(True)
    controller.set_ui_interaction_locked(True)
    assert interrupts == ["interrupt"]

    controller.set_ui_interaction_locked(False)
    controller.set_ui_interaction_locked(True)
    assert interrupts == ["interrupt", "interrupt"]


def test_mutable_pet_controller_advances_visual_generation_when_ui_lock_opens() -> None:
    controller = MutablePetController()
    before = controller.motion_generation()
    controller.set_ui_interaction_locked(True)
    assert controller.motion_generation() == before + 1


def test_mutable_pet_controller_propagates_interaction_interrupt_to_attached_target() -> None:
    class Target:
        def __init__(self) -> None:
            self.callback = None

        def set_interaction_interrupt(self, callback):
            self.callback = callback

    target = Target()
    controller = MutablePetController()
    calls: list[str] = []

    def callback() -> None:
        calls.append("interrupt")

    controller.set_interaction_interrupt(callback)
    controller.attach(target)
    assert target.callback is callback
    target.callback()
    assert calls == ["interrupt"]

    replacement = Target()
    controller.attach(replacement)
    assert target.callback is None
    assert replacement.callback is callback

    controller.set_interaction_interrupt(None)
    assert replacement.callback is None


def test_mutable_pet_controller_window_lock_does_not_pause_behavior() -> None:
    class Target:
        def __init__(self) -> None:
            self.locked = False

        def is_user_interacting(self):
            return False

        def set_window_locked(self, enabled):
            self.locked = bool(enabled)
            return {"status": "available", "locked": self.locked}

        def is_window_locked(self):
            return self.locked

    target = Target()
    controller = MutablePetController(target)
    assert controller.set_window_locked(True) == {"status": "available", "locked": True}
    assert controller.is_window_locked() is True
    assert controller.is_user_interacting() is False


def test_build_runtime_distinguishes_unset_and_explicit_empty_tool_groups(tmp_path: Path) -> None:
    """未配置工具组保持兼容全量；显式空列表必须关闭模型工具可见性。"""

    inventory = inspect_resources(tmp_path / "resources")

    def groups(values: dict[str, object]) -> tuple[object, tuple[str, ...]]:
        runtime = build_runtime(LoadedConfiguration(tmp_path / "config.yaml", values), inventory)
        try:
            visible = tuple(spec.identity for spec in runtime.conversation._tool_definitions())
            return runtime.conversation.tool_groups, visible
        finally:
            asyncio.run(runtime.close())

    unset_groups, unset_visible = groups({})
    assert unset_groups is None
    assert "pet:move" in unset_visible

    null_groups, null_visible = groups({"tools": {"active_groups": None}})
    assert null_groups is None
    assert "pet:move" in null_visible

    empty_groups, empty_visible = groups({"tools": {"active_groups": []}})
    assert empty_groups == ()
    assert empty_visible == ()

    pet_groups, pet_visible = groups({"tools": {"active_groups": ["pet_control"]}})
    assert pet_groups == ("pet_control",)
    assert "pet:move" in pet_visible
    assert "desktop:list_processes" not in pet_visible


def test_scheduler_and_trigger_reject_orphan_approval_actions(tmp_path: Path) -> None:
    """没有对话回合可续接时，审批动作必须撤销并记录失败。"""

    values = {
        "storage": {"database": str(tmp_path / "runtime.sqlite3")},
        "tools": {
            "active_groups": ["pet_control", "scheduler"],
            "permissions": {
                "auto_allow_low_risk": True,
                "bypass_approval": False,
            },
        },
        "scheduler": {
            "tasks": [
                {
                    "task_id": "approval-task",
                    "name": "需要确认的移动",
                    "expression": "every:1h",
                    "action": {
                        "identity": "pet:move",
                        "arguments": {"x": 100, "y": 100},
                    },
                    "owner": "config",
                }
            ],
            "triggers": [
                {
                    "trigger_id": "approval-trigger",
                    "event_name": "startup",
                    "action": {
                        "identity": "pet:move",
                        "arguments": {"x": 100, "y": 100},
                    },
                    "owner": "config",
                }
            ],
        },
    }
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", values),
        inspect_resources(tmp_path / "resources"),
    )

    async def scenario() -> None:
        task = runtime.scheduler._tasks["approval-task"]
        task.next_run_at = 0.0
        assert await runtime.scheduler.tick(now=1.0) == ("approval-task",)
        assert runtime.tools.pending_approvals() == ()
        assert "interactive approval" in str(runtime.scheduler.status()["last_error"])

        assert await runtime.triggers.emit("startup", {}) == ("approval-trigger",)
        assert runtime.tools.pending_approvals() == ()
        assert "interactive approval" in str(runtime.triggers.status()["last_error"])

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(runtime.close())


def test_model_channel_status_is_actionable_and_secret_free() -> None:
    class Router:
        def __init__(self, diagnostics):
            self._diagnostics = diagnostics

        def diagnostics(self, _task):
            return self._diagnostics

    empty = type("Runtime", (), {"router": Router({"ready": False, "channels": ()})})()
    ready = type(
        "Runtime",
        (),
        {
            "router": Router(
                {
                    "ready": True,
                    "channels": ({"id": "primary", "api_key_configured": True},),
                }
            )
        },
    )()
    assert _model_channel_status(empty) == (
        False,
        "模型渠道未配置：点击“配置模型”填写渠道",
    )
    assert _model_channel_status(ready) == (True, "模型渠道已就绪（1 个）")

    auth_missing = type(
        "Runtime",
        (),
        {
            "router": Router(
                {
                    "ready": True,
                    "channels": (
                        {
                            "id": "primary",
                            "protocol": "openai_chat",
                            "api_key_configured": False,
                        },
                    ),
                }
            )
        },
    )()
    auth_ready, auth_message = _model_channel_status(auth_missing)
    assert auth_ready is True
    assert "未检测到密钥" in auth_message
    assert "免密网关" in auth_message

    class BrokenRouter:
        def diagnostics(self, _task):
            raise RuntimeError("secret-value")

    broken = type("Runtime", (), {"router": BrokenRouter()})()
    assert "secret-value" not in _model_channel_status(broken)[1]


def test_model_channel_status_redacts_internal_diagnostic_reason() -> None:
    class Router:
        def diagnostics(self, _task):
            return {
                "ready": False,
                "reason": (
                    "no enabled channel satisfies task: dialogue; configure llm.channels "
                    "or set MEAPET_API_BASE and MEAPET_MODEL"
                ),
                "channels": (),
            }

    runtime = type("Runtime", (), {"router": Router()})()
    ready, message = _model_channel_status(runtime)
    assert ready is False
    assert message == "模型渠道未配置：点击“配置模型”填写渠道"
    assert "meapet-wizard" not in message
    assert "MEAPET_API_BASE" not in message
    summary = _model_channel_diagnostic_summary(runtime)
    assert "MEAPET_API_BASE" in summary
    assert "channels=0" in summary


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {
                "status": "available",
                "message": "Bearer secret-value https://private.invalid",
                "channel_id": "private-channel",
            },
            {"status": "available", "message": "模型连接测试通过"},
        ),
        (
            {
                "status": "unavailable",
                "reason": "authentication",
                "message": "api_key=secret-value",
            },
            {"status": "unavailable", "message": "模型连接失败，请点击“配置模型”检查密钥设置"},
        ),
        (
            {
                "status": "unavailable",
                "reason": "network",
                "message": "https://private.invalid/v1",
            },
            {"status": "unavailable", "message": "模型连接失败，请点击“配置模型”检查服务连接"},
        ),
        (
            {"status": "pending", "message": "internal operation id"},
            {"status": "pending", "message": "正在测试模型连接，请稍候…"},
        ),
    ],
)
def test_public_model_connection_result_is_fixed_and_secret_free(
    result: dict[str, object], expected: dict[str, str]
) -> None:
    assert _public_model_connection_result(result) == expected
    assert "secret-value" not in str(_public_model_connection_result(result))
    assert "private.invalid" not in str(_public_model_connection_result(result))


def test_public_model_connection_future_redacts_result_and_exceptions() -> None:
    source: concurrent.futures.Future[object] = concurrent.futures.Future()
    wrapped = _public_model_connection_future(source)
    source.set_result(
        {
            "status": "unavailable",
            "reason": "unknown",
            "message": "request headers Authorization=secret-value",
        }
    )
    assert wrapped.result(timeout=1) == {
        "status": "unavailable",
        "message": "模型连接测试未通过，请点击“配置模型”检查连接设置",
    }

    failed: concurrent.futures.Future[object] = concurrent.futures.Future()
    failed_wrapped = _public_model_connection_future(failed)
    failed.set_exception(RuntimeError("api_key=secret-value"))
    assert failed_wrapped.result(timeout=1) == {
        "status": "unavailable",
        "message": "模型连接测试未通过，请稍后重试",
    }
    assert "secret-value" not in str(failed_wrapped.result())


@pytest.mark.parametrize(
    ("status", "expected", "message"),
    [
        ("completed", "completed", "回复已完成"),
        ("approval_required", "approval_required", "等待你确认桌面操作"),
        ("cancelled", "cancelled", "对话已停止"),
        ("failed", "failed", "回复未完成，可重试"),
        ("tool_limit", "failed", "操作次数达到上限，回复未继续"),
    ],
)
def test_conversation_operation_receipt_is_public_and_terminal(
    status: str, expected: str, message: str
) -> None:
    """对话 Future 终态必须能投影为稳定的点击式回执。"""

    receipt = _conversation_operation_receipt({"status": status, "error": "secret-value"})
    assert receipt == {"status": expected, "message": message}
    assert "secret-value" not in str(receipt)


def test_conversation_operation_receipt_does_not_overwrite_a_newer_operation() -> None:
    current = {
        "id": "ui-2",
        "kind": "submit_text",
        "status": "requested",
        "message": "消息已提交",
    }
    stale = _apply_conversation_operation_receipt(current, "ui-1", {"status": "failed"})
    assert stale == current
    completed = _apply_conversation_operation_receipt(current, "ui-2", {"status": "completed"})
    assert completed["status"] == "completed"
    assert completed["message"] == "回复已完成"


def test_conversation_cancel_receipt_is_terminal() -> None:
    current = {
        "id": "ui-stop-1",
        "kind": "stop",
        "status": "requested",
        "message": "正在停止",
    }
    cancelled = _apply_conversation_operation_receipt(
        current,
        "ui-stop-1",
        {"status": "cancelled"},
    )
    assert cancelled["status"] == "cancelled"
    assert cancelled["message"] == "对话已停止"


def test_pet_click_affection_is_submitted_to_runtime_loop_and_status_hold_is_safe(
    tmp_path: Path,
) -> None:
    configuration = LoadedConfiguration(tmp_path / "config.yaml", {})
    runtime = build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    loop = RuntimeLoop(runtime)
    main_thread = threading.get_ident()
    pending: dict[int, tuple[object, str, str]] = {}

    class ProbeAffection:
        def __init__(self) -> None:
            self.thread_id: int | None = None
            self.delta: int | None = None

        def adjust(self, delta: int) -> dict[str, int]:
            self.thread_id = threading.get_ident()
            self.delta = delta
            return {"delta": delta}

    probe = ProbeAffection()
    loop.start(timeout=1.0)
    try:
        scheduled = _schedule_pet_affection(
            loop,
            probe,
            pending,
            generation=1,
            zone="upper",
            phrase="喵？摸摸头～",
        )
        assert scheduled == {"status": "pending"}
        future = pending[1][0]
        assert future.result(timeout=2.0) == {"delta": 1}
        assert probe.thread_id is not None
        assert probe.thread_id != main_thread
        assert probe.delta == 1

        assert (
            _interaction_status_with_local_feedback(
                phase="configuration_required",
                phase_message="模型未配置",
                feedback_notice="点击互动：喵？摸摸头～",
                feedback_active=True,
            )
            == "点击互动：喵？摸摸头～"
        )
        assert (
            _interaction_status_with_local_feedback(
                phase="approval_required",
                phase_message="等待你确认工具操作",
                feedback_notice="点击互动：喵？摸摸头～",
                feedback_active=True,
            )
            == "等待你确认工具操作"
        )
    finally:
        loop.stop(timeout=2.0)


def test_pet_click_affection_runtime_loop_updates_real_service(tmp_path: Path) -> None:
    configuration = LoadedConfiguration(tmp_path / "config.yaml", {})
    runtime = build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    loop = RuntimeLoop(runtime)
    main_thread = threading.get_ident()
    pending: dict[int, tuple[object, str, str]] = {}
    original_adjust = runtime.affection.adjust
    adjustment_thread: list[int] = []

    def record_adjust(delta: int):
        adjustment_thread.append(threading.get_ident())
        return original_adjust(delta)

    runtime.affection.adjust = record_adjust  # type: ignore[method-assign]
    loop.start(timeout=1.0)
    try:
        scheduled = _schedule_pet_affection(
            loop,
            runtime.affection,
            pending,
            generation=1,
            zone="upper",
            phrase="喵？摸摸头～",
        )
        assert scheduled == {"status": "pending"}
        change = pending[1][0].result(timeout=2.0)
        assert change.current == 6
        assert adjustment_thread and adjustment_thread[0] != main_thread
    finally:
        loop.stop(timeout=2.0)


def test_runtime_loop_starts_and_stops_background_services(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{scheduler: {enabled: true}}\n", encoding="utf-8")
    configuration = LoadedConfiguration(config_path, {"scheduler": {"enabled": True}})
    inventory = inspect_resources(tmp_path / "resources")
    runtime = build_runtime(configuration, inventory)
    loop = RuntimeLoop(runtime)
    loop.start()
    try:
        assert loop.running
        result = loop.submit(asyncio.sleep(0, result="ok")).result(timeout=2)
        assert result == "ok"
    finally:
        loop.stop()


def test_runtime_loop_stop_fails_fast_from_runtime_thread(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    runtime = build_runtime(
        LoadedConfiguration(config_path, {}), inspect_resources(tmp_path / "resources")
    )
    loop = RuntimeLoop(runtime)
    loop.start()

    async def invoke_stop() -> None:
        try:
            loop.stop()
        except RuntimeError as exc:
            assert "outside the runtime thread" in str(exc)
        else:
            raise AssertionError("stop unexpectedly blocked the runtime thread")

    try:
        loop.submit(invoke_stop()).result(timeout=2)
    finally:
        loop.stop()


def test_runtime_wires_stream_and_tts_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    values = {
        "tools": {"active_groups": ["pet_control"]},
        "ui": {
            "stream": {"show_murmur": False, "show_tool_status": False, "max_bubble_chars": 512}
        },
        "tts": {"language": "ja"},
    }
    configuration = LoadedConfiguration(config_path, values)
    inventory = inspect_resources(tmp_path / "resources")
    runtime = build_runtime(configuration, inventory)
    try:
        assert runtime.conversation.presentation.show_murmur is False
        assert runtime.conversation.presentation.show_tool_status is False
        assert runtime.conversation.presentation.max_text_length == 512
        assert runtime.conversation.tts_language == "ja"
        visible_groups = {
            spec.group for spec in runtime.registry.visible(runtime.conversation.tool_groups)
        }
        assert visible_groups == {"pet_control"}
    finally:
        asyncio.run(runtime.close())


def test_runtime_wires_subprocess_tts_reference_and_model_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    values = {
        "tts": {
            "enabled": True,
            "backend": "subprocess",
            "command": [sys.executable, "-c", "pass"],
            "ref_audio_path": "../voices/jp_normal.wav",
            "prompt_text": "こんにちは",
            "prompt_lang": "ja",
            "gpt_path": "../models/gpt.ckpt",
            "sovits_path": "../models/sovits.pth",
            "options": {"temperature": 0.7},
        }
    }
    runtime = build_runtime(
        LoadedConfiguration(config_path, values), inspect_resources(tmp_path / "resources")
    )
    try:
        backend = runtime.tts.backend
        assert backend.ref_audio_path == str((tmp_path / "../voices/jp_normal.wav").resolve())
        assert backend.prompt_text == "こんにちは"
        assert backend.prompt_lang == "ja"
        assert backend.default_options["gpt_path"] == str(
            (tmp_path / "../models/gpt.ckpt").resolve()
        )
        assert backend.default_options["sovits_path"] == str(
            (tmp_path / "../models/sovits.pth").resolve()
        )
        assert backend.default_options["temperature"] == 0.7
        assert backend.startup_timeout_seconds == 300.0
        assert backend.shutdown_timeout_seconds == 2.0
    finally:
        asyncio.run(runtime.close())


def test_runtime_strictly_parses_quoted_boolean_settings(tmp_path: Path) -> None:
    values = {
        "scheduler": {"enabled": "false"},
        "watcher": {"enabled": "false"},
        "behavior": {"enabled": "false"},
        "tools": {
            "permissions": {
                "bypass_approval": "false",
                "auto_allow_low_risk": "false",
            }
        },
    }
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", values),
        inspect_resources(tmp_path / "resources"),
    )

    async def start_and_close() -> None:
        await runtime.start_background()
        assert runtime.scheduler.status()["running"] is False
        assert runtime.watcher.status()["status"] == "stopped"
        assert runtime.behavior.status()["running"] is False
        spec = ToolSpec("system:test", "test", {"type": "object"}, lambda *_: None, RiskLevel.HIGH)
        decision = runtime.permissions.evaluate(
            spec,
            ToolCallContext("profile", "session", "turn"),
        )
        assert decision.decision is PermissionDecision.APPROVAL_REQUIRED
        await runtime.close()

    asyncio.run(start_and_close())


def test_invalid_runtime_boolean_does_not_create_database(tmp_path: Path) -> None:
    database_path = tmp_path / "created.sqlite3"
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "storage": {"database": str(database_path)},
            "tools": {"permissions": {"bypass_approval": "maybe"}},
        },
    )
    with pytest.raises(ConfigurationError):
        build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    assert not database_path.exists()


def test_invalid_logging_configuration_does_not_create_database(tmp_path: Path) -> None:
    database_path = tmp_path / "logging-invalid.sqlite3"
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {"storage": {"database": str(database_path)}, "logging": {"console": "invalid"}},
    )

    with pytest.raises(ConfigurationError, match="logging configuration is invalid"):
        validate_runtime_configuration(configuration)
    assert not database_path.exists()


def test_auto_model_setup_flag_uses_strict_boolean_contract(tmp_path: Path) -> None:
    valid = LoadedConfiguration(
        tmp_path / "valid.yaml",
        {"ui": {"auto_open_model_setup": False}},
    )
    validate_runtime_configuration(valid)

    invalid = LoadedConfiguration(
        tmp_path / "invalid.yaml",
        {"ui": {"auto_open_model_setup": "later"}},
    )
    with pytest.raises(ConfigurationError, match="auto_open_model_setup"):
        validate_runtime_configuration(invalid)


def test_runtime_validates_md3_theme_and_memory_summary_settings(tmp_path: Path) -> None:
    valid = LoadedConfiguration(
        tmp_path / "valid-theme.yaml",
        {
            "ui": {
                "theme": {
                    "mode": "dark",
                    "roles": {"primary": "#123456"},
                    "layout": {"density": -1},
                }
            },
            "memory": {
                "summarization_enabled": True,
                "summarize_daily": True,
                "summary_interval_minutes": 15,
                "summary_min_messages": 4,
                "recall_min_similarity": 0.1,
                "always_recall_priority": 8,
            },
        },
    )
    validate_runtime_configuration(valid)

    invalid_theme = LoadedConfiguration(
        tmp_path / "invalid-theme.yaml",
        {"ui": {"theme": {"roles": {"primary": "pink"}}}},
    )
    with pytest.raises(ConfigurationError, match="ui.theme"):
        validate_runtime_configuration(invalid_theme)

    invalid_memory = LoadedConfiguration(
        tmp_path / "invalid-memory.yaml",
        {"memory": {"summary_min_messages": 1}},
    )
    with pytest.raises(ConfigurationError, match="summary_min_messages"):
        validate_runtime_configuration(invalid_memory)


def test_runtime_module_diagnostics_are_content_free(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {}),
        inspect_resources(tmp_path / "resources"),
    )
    try:
        diagnostics = runtime.module_diagnostics()

        assert diagnostics["memory"]["enabled"] is True
        assert diagnostics["memory"]["lexical_index"] in {"fts5", "sparse_fallback"}
        assert diagnostics["memory_summary"]["running"] is False
        assert diagnostics["modules"]["modules"]["plugins"]["adopted"] is True
        assert diagnostics["modules"]["modules"]["api_audit"]["adopted"] is True
        assert diagnostics["renderer"] == {
            "available": False,
            "backend": "unattached",
        }
        encoded = repr(diagnostics)
        assert "config.yaml" not in encoded
        assert "sqlite" not in encoded
        assert "api_key" not in encoded
        tool_result = asyncio.run(runtime.execute_console_tool("system:module_status", {}))
        assert tool_result["status"] == "available"
        assert tool_result["runtime"]["memory"]["enabled"] is True
        assert tool_result["asr"]["status"] == "disabled"
        assert "python_executable" not in repr(tool_result["asr"])
        assert "model_path" not in repr(tool_result["asr"])
        assert tool_result["runtime"]["ipc"] == {
            "worker_count": 1,
            "ready_count": 0,
            "status": "unavailable",
            "workers": (
                {
                    "id": "asr",
                    "status": "unavailable",
                    "running": False,
                    "ready": False,
                    "pid": None,
                },
            ),
        }
    finally:
        asyncio.run(runtime.close())


def test_runtime_api_audit_public_records_returns_bounded_dict_snapshots(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {}),
        inspect_resources(tmp_path / "resources"),
    )
    try:
        assert runtime.api_call_audit_public_records(limit=50) == ()
        assert runtime.api_call_audit_count() == 0
    finally:
        asyncio.run(runtime.close())


def test_runtime_module_tests_return_only_the_selected_public_status(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {}),
        inspect_resources(tmp_path / "resources"),
    )
    expected_keys = {
        "model": {
            "module",
            "status",
            "available",
            "ready",
            "channel_count",
            "ready_count",
            "adapter_count",
            "latency_ms",
        },
        "memory": {
            "module",
            "status",
            "available",
            "ready",
            "enabled",
            "summarization_enabled",
            "vector_index_size",
            "vector_index_features",
            "lexical_index",
            "latency_ms",
        },
        "diary": {
            "module",
            "status",
            "available",
            "ready",
            "enabled",
            "public_entry_count",
            "latency_ms",
        },
        "tools": {
            "module",
            "status",
            "available",
            "ready",
            "tool_count",
            "public_tool_count",
            "latency_ms",
        },
        "mcp": {
            "module",
            "status",
            "available",
            "configured",
            "ready",
            "ready_count",
            "latency_ms",
        },
        "scheduler": {
            "module",
            "status",
            "available",
            "ready",
            "running",
            "task_count",
            "trigger_count",
            "persistence",
            "latency_ms",
        },
        "behavior": {
            "module",
            "status",
            "available",
            "ready",
            "running",
            "phase",
            "movement_enabled",
            "action_count",
            "movement_count",
            "latency_ms",
        },
        "renderer": {
            "module",
            "status",
            "available",
            "ready",
            "backend",
            "expression_count",
            "motion_count",
            "latency_ms",
        },
        "watcher": {
            "module",
            "status",
            "available",
            "ready",
            "running",
            "foreground_window_read",
            "latency_ms",
        },
        "ipc": {
            "module",
            "status",
            "available",
            "ready",
            "worker_count",
            "ready_count",
            "workers",
            "latency_ms",
        },
    }

    async def scenario() -> None:
        for module_id, allowed in expected_keys.items():
            result = await asyncio.wait_for(runtime.test_module(module_id), timeout=0.05)
            assert result["module"] == module_id
            assert set(result) == allowed
            assert "runtime" not in result
            assert "adapters" not in result
            assert "tools" not in result

        unknown = await runtime.test_module("../../private/secret")
        assert unknown == {
            "module": "unknown",
            "status": "unavailable",
            "available": False,
            "ready": False,
            "reason_code": "unknown_module",
            "latency_ms": 0.0,
        }
        assert "private" not in repr(unknown)
        assert "secret" not in repr(unknown)
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_module_tests_do_not_queue_or_cancel_loading_workers(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {}),
        inspect_resources(tmp_path / "resources"),
    )

    async def scenario() -> None:
        blocker = asyncio.Event()
        tts_task = asyncio.create_task(blocker.wait())
        asr_task = asyncio.create_task(blocker.wait())
        runtime._tts_start_task = tts_task
        runtime._asr_start_task = asr_task

        async def forbidden_tts_probe() -> Mapping[str, object]:
            raise AssertionError("unrelated module test must not probe TTS")

        async def forbidden_asr_probe() -> Mapping[str, object]:
            raise AssertionError("unrelated module test must not probe ASR")

        runtime.test_tts_module = forbidden_tts_probe  # type: ignore[method-assign]
        runtime.test_asr = forbidden_asr_probe  # type: ignore[method-assign]
        for module_id in (
            "model",
            "memory",
            "diary",
            "tools",
            "mcp",
            "scheduler",
            "behavior",
            "renderer",
            "watcher",
            "ipc",
        ):
            result = await asyncio.wait_for(runtime.test_module(module_id), timeout=0.05)
            assert result["module"] == module_id
            assert runtime._tts_start_task is tts_task
            assert runtime._asr_start_task is asr_task
            assert not tts_task.done()
            assert not asr_task.done()

        tts_task.cancel()
        asr_task.cancel()
        await asyncio.gather(tts_task, asr_task, return_exceptions=True)
        runtime._tts_start_task = None
        runtime._asr_start_task = None
        await runtime.close()

    asyncio.run(scenario())


def test_console_runtime_module_test_never_falls_back_to_global_status_tool() -> None:
    calls: list[str] = []

    class Runtime:
        async def test_module(self, module_id: str) -> Mapping[str, object]:
            calls.append(module_id)
            return {"module": module_id, "status": "ready", "available": True}

        async def execute_console_tool(self, *_args: object) -> Mapping[str, object]:
            raise AssertionError("module test must not call system:module_status")

    result = asyncio.run(_runtime_module_test(Runtime(), "memory"))  # type: ignore[arg-type]

    assert result == {"module": "memory", "status": "ready", "available": True}
    assert calls == ["memory"]


def test_runtime_asr_background_lifecycle_is_non_blocking_and_managed(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {"asr": {"enabled": False}}),
        inspect_resources(tmp_path / "resources"),
    )
    calls: list[str] = []

    class ASR:
        enabled = True

        async def start(self):
            calls.append("start")
            await asyncio.sleep(0)
            return ASRHealth("ready", True, "sensevoice", "SenseVoiceSmall", True, "cpu", "auto")

        async def health(self, *, probe=False):
            calls.append(f"health:{probe}")
            return ASRHealth("ready", True, "sensevoice", "SenseVoiceSmall", True, "cpu", "auto")

        def diagnostics(self):
            return {
                "status": "ready",
                "available": True,
                "ready": True,
                "running": True,
                "backend": "sensevoice",
                "model": "SenseVoiceSmall",
                "model_loaded": True,
                "device": "cpu",
                "language": "auto",
                "ipc": {"status": "ready", "running": True, "ready": True, "pid": 123},
            }

        def has_active_tasks(self):
            return False

        async def cancel(self, request_id):
            return False

        async def aclose(self):
            calls.append("close")

    runtime.asr = ASR()

    async def scenario() -> None:
        await runtime.start_background()
        assert runtime._asr_start_task is not None
        health = await runtime.wait_asr_initialization()
        assert health["ready"] is True
        assert runtime.module_diagnostics()["asr"]["model_loaded"] is True
        probe = await runtime.test_asr()
        assert probe["status"] == "ready"
        await runtime.close()

    asyncio.run(scenario())
    assert calls == ["start", "health:True", "close"]


def test_runtime_background_start_is_idempotent_and_rejects_closed_runtime(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {"enabled": False},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )

    async def scenario() -> None:
        await runtime.start_background()
        tts_task = runtime._tts_start_task
        asr_task = runtime._asr_start_task
        await runtime.start_background()
        assert runtime._tts_start_task is tts_task
        assert runtime._asr_start_task is asr_task
        await runtime.close()
        closed_tts_task = runtime._tts_start_task
        closed_asr_task = runtime._asr_start_task
        await runtime.start_background()
        assert runtime._tts_start_task is closed_tts_task
        assert runtime._asr_start_task is closed_asr_task

    asyncio.run(scenario())


def test_runtime_background_start_can_retry_after_cancelled_setup(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {"enabled": False},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original_start = runtime.memory_summarizer.start

    async def blocked_start() -> None:
        entered.set()
        await release.wait()

    async def scenario() -> None:
        runtime.memory_summarizer.start = blocked_start  # type: ignore[method-assign]
        startup = asyncio.create_task(runtime.start_background())
        await entered.wait()
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup
        assert runtime._background_started is False
        runtime.memory_summarizer.start = original_start  # type: ignore[method-assign]
        await runtime.start_background()
        assert runtime._background_started is True
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_mcp_refresh_failure_closes_bridge_and_client(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {"enabled": False},
                "watcher": {"enabled": False},
                "behavior": {"enabled": False},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    events: list[str] = []

    class Client:
        server_name = "failed-mcp"

        async def close(self) -> None:
            events.append("client:close")

    class Bridge:
        async def refresh_into(self, _registry) -> None:
            raise RuntimeError("refresh failed")

        async def close(self) -> None:
            events.append("bridge:close")

    runtime.mcp_clients = (Client(),)
    runtime.mcp_bridges = (Bridge(),)

    async def scenario() -> None:
        await runtime.start_background()
        assert runtime.mcp_status == {"failed-mcp": "unavailable"}
        assert events == ["bridge:close", "client:close"]
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_mcp_configuration_reload_switches_clients_and_rolls_back_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial = LoadedConfiguration(
        tmp_path / "initial.yaml",
        {
            "mcp": {"enabled": False, "servers": []},
            "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
            "asr": {"enabled": False},
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    events: list[str] = []

    class Client:
        def __init__(self, name: str) -> None:
            self.server_name = name

        async def close(self) -> None:
            events.append(f"client:{self.server_name}:close")

    class Bridge:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def _withdraw_bindings(self, *, keep_registries: bool) -> None:
            assert keep_registries is True
            events.append(f"bridge:{self.name}:withdraw")

        async def discover(self) -> tuple[object, ...]:
            events.append(f"bridge:{self.name}:discover")
            return ()

        def register_into(self, _registry) -> tuple[object, ...]:
            events.append(f"bridge:{self.name}:register")
            if self.fail:
                raise RuntimeError("register failed")
            return ()

        async def refresh_into(self, _registry) -> None:
            events.append(f"bridge:{self.name}:refresh")

        async def close(self) -> None:
            events.append(f"bridge:{self.name}:close")

    old_client = Client("old")
    old_bridge = Bridge("old")
    runtime.mcp_clients = (old_client,)
    runtime.mcp_bridges = (old_bridge,)
    runtime.mcp_status = {"old": "ready"}
    runtime._background_started = True
    runtime._mcp_started = True
    runtime._register_optional_modules = lambda: None  # type: ignore[method-assign]
    new_client = Client("new")
    new_bridge = Bridge("new")
    monkeypatch.setattr(
        runtime_module,
        "_build_mcp",
        lambda *_args, **_kwargs: ((new_client,), (new_bridge,)),
    )

    async def scenario() -> None:
        updated = LoadedConfiguration(
            tmp_path / "updated.yaml",
            {
                **initial.values,
                "mcp": {
                    "enabled": True,
                    "servers": [{"name": "new", "command": [sys.executable, "-c", "pass"]}],
                },
            },
        )
        result = await runtime.apply_configuration(updated)
        assert result["status"] == "reloaded", result
        assert result["mcp_status"] == "reloaded"
        assert runtime.mcp_clients == (new_client,)
        assert runtime.mcp_bridges == (new_bridge,)
        assert runtime.mcp_status == {"new": "ready"}
        assert events[:3] == [
            "bridge:new:discover",
            "bridge:old:withdraw",
            "bridge:new:register",
        ]
        assert "bridge:old:close" in events
        assert "client:old:close" in events

        failing_client = Client("failed")
        failing_bridge = Bridge("failed", fail=True)
        monkeypatch.setattr(
            runtime_module,
            "_build_mcp",
            lambda *_args, **_kwargs: ((failing_client,), (failing_bridge,)),
        )
        failed = LoadedConfiguration(
            tmp_path / "failed.yaml",
            {
                **updated.values,
                "mcp": {
                    "enabled": True,
                    "servers": [{"name": "failed", "command": [sys.executable, "-c", "pass"]}],
                },
            },
        )
        failure = await runtime.apply_configuration(failed)
        assert failure["status"] == "unavailable"
        assert runtime.mcp_clients == (new_client,)
        assert runtime.mcp_bridges == (new_bridge,)
        assert runtime.mcp_status == {"new": "ready"}
        assert "bridge:new:refresh" in events
        assert "bridge:failed:close" in events
        assert "client:failed:close" in events
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_mcp_refresh_cleanup_propagates_cancellation(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {"enabled": False},
                "watcher": {"enabled": False},
                "behavior": {"enabled": False},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    events: list[str] = []

    class Client:
        server_name = "cancelled-mcp"

        async def close(self) -> None:
            events.append("client:close")

    class Bridge:
        async def refresh_into(self, _registry) -> None:
            raise RuntimeError("refresh failed")

        async def close(self) -> None:
            events.append("bridge:close")
            raise asyncio.CancelledError

    runtime.mcp_clients = (Client(),)
    runtime.mcp_bridges = (Bridge(),)

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await runtime.start_background()
        assert events == ["bridge:close", "client:close"]
        assert runtime.mcp_status == {}
        assert runtime._mcp_started is False
        assert runtime._background_started is False
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_mcp_cancellation_keeps_other_resources_cleanup(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {"enabled": False},
                "watcher": {"enabled": False},
                "behavior": {"enabled": False},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    events: list[str] = []

    class Client:
        def __init__(self, name: str) -> None:
            self.server_name = name

        async def close(self) -> None:
            events.append(f"{self.server_name}:client")

    class Bridge:
        def __init__(self, name: str, cancel_on_close: bool) -> None:
            self.name = name
            self.cancel_on_close = cancel_on_close

        async def refresh_into(self, _registry) -> None:
            raise RuntimeError(f"{self.name} refresh failed")

        async def close(self) -> None:
            events.append(f"{self.name}:bridge")
            if self.cancel_on_close:
                raise asyncio.CancelledError

    runtime.mcp_clients = (Client("first"), Client("second"))
    runtime.mcp_bridges = (Bridge("first", True), Bridge("second", False))

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await runtime.start_background()
        assert events == [
            "first:bridge",
            "second:bridge",
            "first:client",
            "second:client",
        ]
        assert runtime.mcp_status == {}
        assert runtime._mcp_started is False
        assert runtime._background_started is False
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_mcp_cleanup_retries_unconfirmed_close(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {"enabled": False},
                "watcher": {"enabled": False},
                "behavior": {"enabled": False},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    events: list[str] = []

    class Client:
        server_name = "retry-mcp"

        async def close(self) -> None:
            events.append("client:close")

    class Bridge:
        def __init__(self) -> None:
            self.calls = 0

        async def refresh_into(self, _registry) -> None:
            raise RuntimeError("refresh failed")

        async def close(self) -> None:
            self.calls += 1
            events.append(f"bridge:close:{self.calls}")
            if self.calls == 1:
                raise RuntimeError("close not confirmed")

    bridge = Bridge()
    runtime.mcp_clients = (Client(),)
    runtime.mcp_bridges = (bridge,)

    async def scenario() -> None:
        await runtime.start_background()
        assert events == ["bridge:close:1", "bridge:close:2", "client:close"]
        assert runtime.mcp_status == {"retry-mcp": "unavailable"}
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_mcp_cleanup_failure_enters_startup_rollback(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {"enabled": False},
                "watcher": {"enabled": False},
                "behavior": {"enabled": False},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    events: list[str] = []

    class Client:
        server_name = "persistent-failure-mcp"

        async def close(self) -> None:
            events.append("client:close")

    class Bridge:
        def __init__(self) -> None:
            self.calls = 0

        async def refresh_into(self, _registry) -> None:
            raise RuntimeError("refresh failed")

        async def close(self) -> None:
            self.calls += 1
            events.append(f"bridge:close:{self.calls}")
            if self.calls <= 3:
                raise RuntimeError("close failed")

    runtime.mcp_clients = (Client(),)
    runtime.mcp_bridges = (Bridge(),)

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="close failed"):
            await runtime.start_background()
        assert events == [
            "bridge:close:1",
            "bridge:close:2",
            "bridge:close:3",
            "client:close",
        ]
        assert runtime.mcp_status == {}
        assert runtime._mcp_started is False
        assert runtime._background_started is False
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_background_failure_rolls_back_mcp_resources(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {"enabled": False},
                "watcher": {"enabled": True, "interval_seconds": 0.05},
                "behavior": {"enabled": False},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    events: list[str] = []

    class Client:
        server_name = "ready-mcp"

        async def close(self) -> None:
            events.append("client:close")

    class Bridge:
        async def refresh_into(self, _registry) -> None:
            events.append("bridge:refresh")

        async def close(self) -> None:
            events.append("bridge:close")

    runtime.mcp_clients = (Client(),)
    runtime.mcp_bridges = (Bridge(),)

    async def failing_watcher_start() -> None:
        raise RuntimeError("watcher startup failed")

    runtime.watcher.start = failing_watcher_start  # type: ignore[method-assign]

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="watcher startup failed"):
            await runtime.start_background()
        assert events == ["bridge:refresh", "bridge:close", "client:close"]
        assert runtime._mcp_started is False
        assert runtime._background_started is False
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_background_failure_rolls_back_restartable_services(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
                "scheduler": {
                    "enabled": True,
                    "poll_seconds": 0.05,
                    "activity": {"enabled": True, "idle_seconds": 60, "poll_seconds": 0.05},
                },
                "watcher": {"enabled": True, "interval_seconds": 0.05},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    original_watcher_start = runtime.watcher.start
    attempts = 0

    async def failing_watcher_start() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("watcher startup failed")

    async def scenario() -> None:
        runtime.watcher.start = failing_watcher_start  # type: ignore[method-assign]
        try:
            with pytest.raises(RuntimeError, match="watcher startup failed"):
                await runtime.start_background()
            assert attempts == 1
            assert runtime._background_started is False
            assert runtime._tts_start_task is None
            assert runtime._asr_start_task is None
            assert runtime.memory_summarizer is not None
            assert runtime.memory_summarizer.running is False
            assert runtime.activity is not None
            assert runtime.activity.running is False
            assert runtime.scheduler.status()["running"] is False
            assert runtime.watcher.running is False

            runtime.watcher.start = original_watcher_start  # type: ignore[method-assign]
            await runtime.start_background()
            assert runtime._background_started is True
            assert runtime.memory_summarizer.running is True
            assert runtime.activity.running is True
            assert runtime.scheduler.status()["running"] is True
            assert runtime.watcher.running is True
        finally:
            runtime.watcher.start = original_watcher_start  # type: ignore[method-assign]
            await runtime.close()

    asyncio.run(scenario())


def test_runtime_asr_configuration_is_strict_and_hot_reloadable(tmp_path: Path) -> None:
    invalid = LoadedConfiguration(
        tmp_path / "invalid.yaml",
        {
            "asr": {
                "enabled": True,
                "backend": "sensevoice",
                "python_executable": sys.executable,
                "model_path": str(tmp_path / "model"),
                "device": "cpu",
                "language": "auto",
                "timeout_seconds": 0,
                "startup_timeout_seconds": 180,
                "max_audio_bytes": 4096,
            }
        },
    )
    with pytest.raises(ConfigurationError, match="asr.timeout_seconds"):
        validate_runtime_configuration(invalid)

    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "initial.yaml",
            {"asr": {"enabled": False, "backend": "sensevoice", "language": "auto"}},
        ),
        inspect_resources(tmp_path / "resources"),
    )

    async def scenario() -> None:
        previous = runtime.asr
        result = await runtime.apply_configuration(
            LoadedConfiguration(
                tmp_path / "updated.yaml",
                {"asr": {"enabled": False, "backend": "sensevoice", "language": "zh"}},
            )
        )
        assert result["status"] == "reloaded"
        assert result["applied_sections"] == ("asr",)
        assert runtime.asr is not previous
        assert runtime.asr.language == "zh"
        assert previous.closed is True
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_asr_enabled_without_python_uses_project_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """示例配置省略解释器时，ASR 构造应自动复用当前项目环境。"""

    import app.runtime as runtime_module

    captured: dict[str, object] = {}

    class StubASR:
        def __init__(self, command, **kwargs):
            captured["command"] = tuple(command)
            captured.update(kwargs)

    monkeypatch.setattr(runtime_module, "ASRService", StubASR)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    model = tmp_path / "model"
    model.mkdir()
    configuration = LoadedConfiguration(
        config_dir / "app.yaml",
        {
            "asr": {
                "enabled": True,
                "backend": "sensevoice",
                "model_path": str(model),
                "language": "zh",
            }
        },
    )

    runtime_module._build_asr(configuration)

    command = captured["command"]
    assert isinstance(command, tuple)
    assert command[0]
    assert command[1:4] == ("-u", "-m", "services.asr.worker")


def test_validate_runtime_asr_enabled_requires_worker_configuration(tmp_path: Path) -> None:
    """生产配置校验仍拒绝显式启用但缺少 ASR worker 字段。"""

    configuration = LoadedConfiguration(
        tmp_path / "invalid-asr.yaml",
        {
            "asr": {
                "enabled": True,
                "backend": "sensevoice",
                "model_path": str(tmp_path / "model"),
                "language": "zh",
            }
        },
    )

    with pytest.raises(ConfigurationError, match="asr.python_executable"):
        validate_runtime_configuration(configuration)


def test_runtime_asr_defaults_to_zh_and_preserves_explicit_auto(tmp_path: Path) -> None:
    defaults = default_configuration_values(resource_root=Path("resources").resolve())
    assert defaults["asr"]["language"] == "zh"

    default_runtime = build_runtime(
        LoadedConfiguration(tmp_path / "default.yaml", {}),
        inspect_resources(tmp_path / "default-resources"),
    )
    explicit_runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "explicit.yaml",
            {"asr": {"enabled": False, "backend": "sensevoice", "language": "auto"}},
        ),
        inspect_resources(tmp_path / "explicit-resources"),
    )
    try:
        assert default_runtime.asr.language == "zh"
        assert explicit_runtime.asr.language == "auto"
    finally:
        asyncio.run(default_runtime.close())
        asyncio.run(explicit_runtime.close())


def test_runtime_asr_worker_environment_does_not_inherit_model_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-asr")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-reach-asr")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/inherited/path")
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {"asr": {"enabled": False}}),
        inspect_resources(tmp_path / "resources"),
    )
    try:
        environment = runtime.asr._client.env
        assert environment is not None
        assert "OPENAI_API_KEY" not in environment
        assert "ANTHROPIC_AUTH_TOKEN" not in environment
        assert "/untrusted/inherited/path" not in environment["PYTHONPATH"]
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert environment["PYTHONUNBUFFERED"] == "1"
    finally:
        asyncio.run(runtime.close())


def test_runtime_asr_module_test_does_not_queue_behind_model_loading(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {"asr": {"enabled": False}}),
        inspect_resources(tmp_path / "resources"),
    )

    async def scenario() -> None:
        blocker = asyncio.Event()
        runtime._asr_start_task = asyncio.create_task(blocker.wait())
        runtime.asr_startup_health = {
            "status": "loading",
            "available": False,
            "ready": False,
            "pending": True,
        }
        result = await asyncio.wait_for(runtime.test_asr(), timeout=0.05)
        assert result["status"] == "loading"
        assert result["available"] is False
        assert result["ready"] is False
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_transcription_waits_for_ordered_background_asr_start(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {"tts": {"enabled": False, "backend": "text_only", "language": "zh"}},
        ),
        inspect_resources(tmp_path / "resources"),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    class ASR:
        async def transcribe(self, _audio, **_kwargs):
            assert release.is_set()
            calls.append("transcribe")
            return {"text": "ok"}

        async def aclose(self) -> None:
            calls.append("close")

    async def initialize() -> None:
        started.set()
        await release.wait()

    async def scenario() -> None:
        await runtime.asr.aclose()
        runtime.asr = ASR()  # type: ignore[assignment]
        runtime._asr_start_task = asyncio.create_task(initialize())
        operation = asyncio.create_task(
            runtime.transcribe_audio(
                b"\x00\x00" * 32,
                audio_format="pcm_s16le",
                sample_rate=16000,
                channels=1,
            )
        )
        await started.wait()
        await asyncio.sleep(0)
        assert calls == []
        release.set()
        assert await operation == {"text": "ok"}
        assert calls == ["transcribe"]
        runtime._asr_start_task = None
        await runtime.close()

    asyncio.run(scenario())
    assert calls == ["transcribe", "close"]


def test_runtime_preloads_tts_before_asr_without_blocking_background_start(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    tts_release = asyncio.Event()
    events: list[str] = []

    class TTS:
        async def start(self):
            events.append("tts:start")
            await tts_release.wait()
            events.append("tts:ready")
            return EngineHealth("text-only", True, "ready")

        async def aclose(self):
            events.append("tts:close")

    class ASR:
        async def start(self):
            events.append("asr:start")
            return ASRHealth("ready", True, "sensevoice", "SenseVoiceSmall", True, "cpu", "auto")

        def diagnostics(self):
            return {
                "status": "ready",
                "available": True,
                "ready": True,
                "running": True,
                "backend": "sensevoice",
                "model": "SenseVoiceSmall",
                "model_loaded": True,
                "device": "cpu",
                "language": "auto",
                "ipc": {"status": "ready", "running": True, "ready": True, "pid": 123},
            }

        async def health(self, *, probe=False):
            return ASRHealth("ready", True, "sensevoice", "SenseVoiceSmall", True, "cpu", "auto")

        def has_active_tasks(self):
            return False

        async def cancel(self, request_id):
            return False

        async def aclose(self):
            events.append("asr:close")

    runtime.tts = TTS()
    runtime.asr = ASR()

    async def scenario() -> None:
        await asyncio.wait_for(runtime.start_background(), timeout=0.1)
        await asyncio.sleep(0)
        assert events == ["tts:start"]
        assert runtime.asr_startup_health["status"] == "queued"
        assert runtime.module_diagnostics()["asr"]["status"] == "queued"
        tts_release.set()
        await runtime.wait_asr_initialization()
        assert events[:3] == ["tts:start", "tts:ready", "asr:start"]
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_close_cancels_queued_asr_before_tts_startup(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
                "asr": {"enabled": False},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    blocker = asyncio.Event()
    events: list[str] = []

    class TTS:
        async def start(self):
            events.append("tts:start")
            await blocker.wait()
            return EngineHealth("text-only", True, "ready")

        async def aclose(self):
            events.append("tts:close")

    class ASR:
        async def start(self):
            events.append("asr:start")
            return ASRHealth("ready", True, "sensevoice", "SenseVoiceSmall", True, "cpu", "auto")

        async def aclose(self):
            events.append("asr:close")

    runtime.tts = TTS()
    runtime.asr = ASR()

    async def scenario() -> None:
        await runtime.start_background()
        await asyncio.sleep(0)
        assert events == ["tts:start"]
        await asyncio.wait_for(runtime.close(), timeout=0.2)

    asyncio.run(scenario())
    assert "asr:start" not in events
    assert events == ["tts:start", "tts:close", "asr:close"]


def test_runtime_background_lifecycle_initializes_and_unloads_tts(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {"tts": {"enabled": False, "backend": "text_only", "language": "zh"}},
        ),
        inspect_resources(tmp_path / "resources"),
    )

    async def scenario() -> None:
        await runtime.start_background()
        await runtime.wait_tts_initialization()
        assert runtime.tts_startup_health["available"] is True
        assert runtime.tts_startup_health["engine"] == "text-only"
        assert runtime.tts_diagnostics()["health"]["available"] is True
        await runtime.close()
        assert runtime.tts.closed is True

    asyncio.run(scenario())


def test_runtime_tts_module_probe_uses_health_without_synthesis(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {"tts": {"enabled": False, "backend": "text_only", "language": "zh"}},
        ),
        inspect_resources(tmp_path / "resources"),
    )
    calls: list[str] = []

    class Backend:
        async def health(self):
            calls.append("health")
            return EngineHealth("gpt-sovits-stdio", True, "ready")

        def diagnostics(self):
            return {
                "backend": "gpt-sovits-stdio",
                "mode": "built_in",
                "model_ready": True,
                "ipc": {"status": "ready", "running": True, "ready": True, "pid": 123},
            }

        def stream(self, _request):
            calls.append("synthesize")
            raise AssertionError("module probe must not synthesize")

        async def aclose(self):
            calls.append("close")

    runtime.tts.backend = Backend()

    async def scenario() -> None:
        result = await runtime.test_tts_module()
        assert result["status"] == "ready"
        assert result["ready"] is True
        assert result["mode"] == "built_in"
        assert result["model_ready"] is True
        assert result["ipc"]["pid"] == 123
        assert calls == ["health"]
        await runtime.close()

    asyncio.run(scenario())
    assert calls == ["health", "close"]


def test_runtime_tts_module_probe_returns_cached_busy_without_waiting_for_backend(
    tmp_path: Path,
) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {"tts": {"enabled": False, "backend": "text_only", "language": "zh"}},
        ),
        inspect_resources(tmp_path / "resources"),
    )
    calls: list[str] = []

    class Backend:
        engine_name = "busy-backend"

        async def health(self):
            calls.append("health")
            raise AssertionError("busy module probe must not queue behind backend health")

        def diagnostics(self):
            return {
                "backend": "busy-backend",
                "mode": "built_in",
                "model_ready": True,
                "ipc": {"status": "ready", "running": True, "ready": True, "pid": 123},
            }

        async def aclose(self):
            calls.append("close")

    runtime.tts.backend = Backend()
    runtime.tts.has_active_tasks = lambda: True  # type: ignore[method-assign]
    runtime.tts_startup_health = {
        "status": "ready",
        "available": True,
        "ready": True,
        "pending": False,
    }

    async def scenario() -> None:
        result = await asyncio.wait_for(runtime.test_tts_module(), timeout=0.05)
        assert result["status"] == "busy"
        assert result["available"] is True
        assert result["ready"] is False
        assert result["model_ready"] is True
        assert calls == []
        await runtime.close()

    asyncio.run(scenario())
    assert calls == ["close"]


def test_runtime_background_does_not_block_on_slow_tts_initialization(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only"},
                "scheduler": {"enabled": False, "activity": {"enabled": False}},
                "watcher": {"enabled": False},
                "behavior": {"enabled": False},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    health_calls = 0

    class SlowBackend:
        async def start(self) -> EngineHealth:
            entered.set()
            await release.wait()
            return EngineHealth("slow", True, "ready")

        async def health(self) -> EngineHealth:
            nonlocal health_calls
            health_calls += 1
            return EngineHealth("slow", True, "ready")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    runtime.tts.backend = SlowBackend()

    async def scenario() -> None:
        await asyncio.wait_for(runtime.start_background(), timeout=0.2)
        await entered.wait()
        assert runtime.tts_startup_health["pending"] is True
        probe = await asyncio.wait_for(runtime.test_tts_module(), timeout=0.05)
        assert probe["status"] == "loading"
        assert probe["available"] is False
        assert probe["ready"] is False
        assert probe["latency_ms"] < 50.0
        assert health_calls == 0
        release.set()
        health = await runtime.wait_tts_initialization()
        assert health["available"] is True
        assert health["engine"] == "slow"
        await runtime.close()

    asyncio.run(scenario())


def test_tts_hot_reload_cancels_pending_start_before_backend_swap(tmp_path: Path) -> None:
    initial = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
            "scheduler": {"enabled": False, "activity": {"enabled": False}},
            "watcher": {"enabled": False},
            "behavior": {"enabled": False},
            "config": {"reload": {"enabled": False}},
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    class SlowBackend:
        async def start(self) -> EngineHealth:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def health(self) -> EngineHealth:
            return EngineHealth("slow", False, "pending")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    runtime.tts.backend = SlowBackend()
    updated = LoadedConfiguration(
        initial.path,
        {
            **initial.values,
            "tts": {
                "enabled": False,
                "backend": "text_only",
                "language": "ja",
            },
        },
    )

    async def scenario() -> None:
        await runtime.start_background()
        await entered.wait()
        result = await asyncio.wait_for(runtime.apply_configuration(updated), timeout=1.0)
        assert result["status"] == "reloaded"
        assert cancelled.is_set()
        assert runtime.tts.started is True
        assert runtime.tts_startup_health["engine"] == "text-only"
        assert runtime.tts_startup_health["available"] is True
        await runtime.close()

    asyncio.run(scenario())


def test_tts_hot_reload_keeps_available_backend_when_replacement_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module
    from services.tts.coordinator import TTSCoordinator

    initial = LoadedConfiguration(
        tmp_path / "config.yaml",
        {"tts": {"enabled": False, "backend": "text_only", "language": "zh"}},
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    previous_backend = runtime.tts.backend
    replacement_closed = False

    class UnavailableBackend:
        async def start(self) -> EngineHealth:
            return EngineHealth("unavailable-test", False, "unavailable")

        async def health(self) -> EngineHealth:
            return EngineHealth("unavailable-test", False, "unavailable")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            nonlocal replacement_closed
            replacement_closed = True

    replacement = TTSCoordinator(UnavailableBackend())
    monkeypatch.setattr(runtime_module, "_build_tts", lambda _configuration: replacement)
    updated = LoadedConfiguration(
        initial.path,
        {"tts": {"enabled": False, "backend": "text_only", "language": "ja"}},
    )

    async def scenario() -> None:
        initial_health = await runtime.tts.start()
        runtime._set_tts_health(initial_health)
        health_before = dict(runtime.tts_startup_health)
        result = await runtime.apply_configuration(updated)
        assert result["status"] == "unavailable"
        assert result["reason"] == "tts backend is unavailable"
        assert runtime.tts.backend is previous_backend
        assert runtime.configuration is initial
        assert runtime.tts_startup_health == health_before
        assert replacement_closed is True
        await runtime.close()

    asyncio.run(scenario())


def test_tts_hot_reload_waits_for_old_tts_and_queued_asr_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module
    from services.tts.coordinator import TTSCoordinator

    initial = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
            "asr": {"enabled": False},
            "scheduler": {"enabled": False, "activity": {"enabled": False}},
            "watcher": {"enabled": False},
            "behavior": {"enabled": False},
            "config": {"reload": {"enabled": False}},
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    old_tts_entered = asyncio.Event()
    old_tts_cancelled = asyncio.Event()
    old_asr_task: asyncio.Task[object] | None = None

    class SlowOldBackend:
        async def start(self) -> EngineHealth:
            old_tts_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                old_tts_cancelled.set()
                raise

        async def health(self) -> EngineHealth:
            return EngineHealth("slow-old", False, "loading")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    class ReadyReplacementBackend:
        async def start(self) -> EngineHealth:
            assert old_tts_cancelled.is_set()
            assert old_asr_task is not None and old_asr_task.done()
            return EngineHealth("replacement", True, "ready")

        async def health(self) -> EngineHealth:
            return EngineHealth("replacement", True, "ready")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            return None

    runtime.tts.backend = SlowOldBackend()
    replacement = TTSCoordinator(ReadyReplacementBackend())
    monkeypatch.setattr(runtime_module, "_build_tts", lambda _configuration: replacement)
    updated = LoadedConfiguration(
        initial.path,
        {
            **initial.values,
            "tts": {"enabled": False, "backend": "text_only", "language": "ja"},
        },
    )

    async def scenario() -> None:
        nonlocal old_asr_task
        await runtime.start_background()
        await old_tts_entered.wait()
        old_asr_task = runtime._asr_start_task
        assert old_asr_task is not None and not old_asr_task.done()
        result = await runtime.apply_configuration(updated)
        assert result["status"] == "reloaded"
        assert runtime.tts.backend is replacement.backend
        await runtime.close()

    asyncio.run(scenario())


def test_asr_hot_reload_waits_for_old_model_start_tasks_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.runtime as runtime_module

    initial = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
            "asr": {"enabled": False, "backend": "sensevoice", "language": "auto"},
            "scheduler": {"enabled": False, "activity": {"enabled": False}},
            "watcher": {"enabled": False},
            "behavior": {"enabled": False},
            "config": {"reload": {"enabled": False}},
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    old_tts_entered = asyncio.Event()
    old_tts_cancelled = asyncio.Event()
    old_asr_task: asyncio.Task[object] | None = None

    class SlowOldTTS:
        async def start(self) -> EngineHealth:
            old_tts_entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                old_tts_cancelled.set()
                raise

        async def aclose(self) -> None:
            return None

    class ReplacementASR:
        enabled = False
        closed = False

        async def start(self) -> ASRHealth:
            assert old_tts_cancelled.is_set()
            assert old_asr_task is not None and old_asr_task.done()
            return ASRHealth("disabled", False, "sensevoice", "SenseVoiceSmall", False, "cpu", "zh")

        def has_active_tasks(self) -> bool:
            return False

        async def aclose(self) -> None:
            self.closed = True

    replacement = ReplacementASR()
    runtime.tts = SlowOldTTS()
    monkeypatch.setattr(runtime_module, "_build_asr", lambda _configuration: replacement)
    updated = LoadedConfiguration(
        initial.path,
        {
            **initial.values,
            "asr": {"enabled": False, "backend": "sensevoice", "language": "zh"},
        },
    )

    async def scenario() -> None:
        nonlocal old_asr_task
        await runtime.start_background()
        await old_tts_entered.wait()
        old_asr_task = runtime._asr_start_task
        assert old_asr_task is not None and not old_asr_task.done()
        result = await runtime.apply_configuration(updated)
        assert result["status"] == "reloaded"
        assert runtime.asr is replacement
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_close_cancels_pending_tts_initialization_and_unloads(tmp_path: Path) -> None:
    runtime = build_runtime(
        LoadedConfiguration(
            tmp_path / "config.yaml",
            {
                "tts": {"enabled": False, "backend": "text_only"},
                "scheduler": {"enabled": False, "activity": {"enabled": False}},
                "watcher": {"enabled": False},
                "behavior": {"enabled": False},
                "config": {"reload": {"enabled": False}},
            },
        ),
        inspect_resources(tmp_path / "resources"),
    )
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    closes = 0

    class SlowBackend:
        async def start(self) -> EngineHealth:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def health(self) -> EngineHealth:
            return EngineHealth("slow", False, "pending")

        async def stream(self, _request):
            if False:
                yield None

        async def aclose(self) -> None:
            nonlocal closes
            closes += 1

    runtime.tts.backend = SlowBackend()

    async def scenario() -> None:
        await runtime.start_background()
        await entered.wait()
        await asyncio.wait_for(runtime.close(), timeout=0.5)
        assert cancelled.is_set()
        assert closes == 1
        assert runtime.tts.closed is True
        assert runtime._tts_start_task is None

    asyncio.run(scenario())


def test_runtime_hot_reloads_persona_and_memory_prompts(tmp_path: Path) -> None:
    initial = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "app": {
                "persona": {"name": "Mea", "proactive_enabled": True},
                "prompts": {"dialogue": "旧对话提示。"},
            }
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    updated = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "app": {
                "persona": {
                    "name": "米娅",
                    "role": "桌面猫娘助手",
                    "user_address": "主人",
                    "traits": ["可靠"],
                    "proactive_enabled": False,
                },
                "prompts": {
                    "dialogue": "新对话提示。",
                    "memory_summary": "自定义总结提示。",
                    "memory_extract": "自定义提取提示。",
                },
            }
        },
    )
    try:
        result = asyncio.run(runtime.apply_configuration(updated))

        assert result["status"] == "reloaded"
        assert result["applied_sections"] == ("app",)
        assert runtime.persona_prompts.persona.name == "米娅"
        assert "新对话提示。" in runtime.conversation.system_prompt
        assert "名字：米娅" in runtime.conversation.system_prompt
        assert runtime.memory_summarizer._summary_prompt == "自定义总结提示。"
        assert runtime.memory_summarizer._extraction_prompt_text == "自定义提取提示。"
    finally:
        asyncio.run(runtime.close())


def test_invalid_scheduler_configuration_does_not_create_database(tmp_path: Path) -> None:
    database_path = tmp_path / "invalid-scheduler.sqlite3"
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "storage": {"database": str(database_path)},
            "scheduler": {
                "tasks": [
                    {
                        "name": "broken",
                        "expression": "every:0s",
                        "action": {"identity": "pet:move", "arguments": {}},
                    }
                ]
            },
        },
    )
    with pytest.raises(ConfigurationError):
        build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    assert not database_path.exists()


def test_cli_validate_rejects_invalid_runtime_without_creating_database(tmp_path: Path) -> None:
    database_path = tmp_path / "cli-invalid.sqlite3"
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "\n".join(
            (
                "storage:",
                f"  database: {database_path}",
                "tools:",
                "  permissions:",
                "    bypass_approval: maybe",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        cli_main(["--config", str(config_path), "--validate"])

    assert error.value.code == 2
    assert not database_path.exists()


def test_cli_validate_rejects_oversized_approval_ttl_without_creating_database(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "cli-invalid-ttl.sqlite3"
    config_path = tmp_path / "invalid-ttl.yaml"
    config_path.write_text(
        "\n".join(
            (
                "storage:",
                f"  database: {database_path}",
                "tools:",
                "  permissions:",
                "    approval_ttl_seconds: 3600.1",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        cli_main(["--config", str(config_path), "--validate"])

    assert error.value.code == 2
    assert not database_path.exists()


def test_cli_validate_rejects_invalid_channel_without_creating_database(tmp_path: Path) -> None:
    database_path = tmp_path / "cli-invalid-channel.sqlite3"
    config_path = tmp_path / "invalid-channel.yaml"
    config_path.write_text(
        "\n".join(
            (
                "storage:",
                f"  database: {database_path}",
                "llm:",
                "  channels:",
                "    - id: primary",
                "      protocol: unknown_protocol",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as error:
        cli_main(["--config", str(config_path), "--validate"])

    assert error.value.code == 2
    assert not database_path.exists()


def test_build_runtime_wraps_invalid_channel_without_creating_database(tmp_path: Path) -> None:
    database_path = tmp_path / "runtime-invalid-channel.sqlite3"
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "storage": {"database": str(database_path)},
            "llm": {"channels": [{"id": "primary", "protocol": "unknown_protocol"}]},
        },
    )

    with pytest.raises(
        ConfigurationError,
        match=r"runtime configuration is invalid: unsupported channel protocol: unknown_protocol",
    ):
        build_runtime(configuration, inspect_resources(tmp_path / "resources"))

    assert not database_path.exists()


def test_cli_validate_rejects_invalid_resource_path(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-resource.yaml"
    config_path.write_text("rendering:\n  resource_root: 42\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        cli_main(["--config", str(config_path), "--validate"])

    assert error.value.code == 2


def test_cli_validate_rejects_invalid_sprite_scale(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid-sprite-scale.yaml"
    config_path.write_text("rendering:\n  sprite_scale: 3\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        cli_main(["--config", str(config_path), "--validate"])

    assert error.value.code == 2


@pytest.mark.parametrize(
    ("field", "value"),
    (("frame_rate", 10), ("frame_rate", 121), ("geometry_audit_hz", 0), ("geometry_audit_hz", 61)),
)
def test_cli_validate_rejects_invalid_rendering_performance_settings(
    tmp_path: Path, field: str, value: int
) -> None:
    config_path = tmp_path / f"invalid-{field}-{value}.yaml"
    config_path.write_text(f"rendering:\n  {field}: {value}\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        cli_main(["--config", str(config_path), "--validate"])

    assert error.value.code == 2


def test_validate_runtime_configuration_checks_scheduler_domain_values(tmp_path: Path) -> None:
    database_path = tmp_path / "scheduler-invalid.sqlite3"
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "storage": {"database": str(database_path)},
            "scheduler": {
                "tasks": [
                    {
                        "task_id": "same",
                        "name": "first",
                        "expression": "every:1s",
                        "action": {"identity": "pet:move", "arguments": {}},
                        "owner": "first-owner",
                    },
                    {
                        "task_id": "same",
                        "name": "second",
                        "expression": "every:1s",
                        "action": {"identity": "pet:move", "arguments": {}},
                        "owner": "second-owner",
                    },
                ]
            },
        },
    )

    with pytest.raises(ConfigurationError):
        validate_runtime_configuration(configuration)
    assert not database_path.exists()


def test_validate_runtime_configuration_bounds_random_motion_duration(tmp_path: Path) -> None:
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "behavior": {
                "actions": [
                    {
                        "identity": "pet:play_motion",
                        "arguments": {"name": "blink"},
                        "duration_seconds": 30.1,
                    }
                ]
            }
        },
    )

    with pytest.raises(ConfigurationError, match="duration_seconds"):
        validate_runtime_configuration(configuration)


def test_runtime_wires_gsv_audio_and_health_settings(tmp_path: Path) -> None:
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "tts": {
                "enabled": True,
                "backend": "gpt_sovits_http",
                "endpoint": "https://tts.invalid/tts",
                "ref_audio_path": "/voice.wav",
                "health_probe": True,
                "raw_sample_rate": 16000,
                "raw_channels": 2,
                "expected_sample_rate": 32000,
                "max_total_bytes": 1024 * 1024,
            }
        },
    )
    runtime = build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    try:
        backend = runtime.tts.backend
        assert backend.health_probe is True
        assert backend.raw_format.sample_rate == 16000
        assert backend.raw_format.channels == 2
        assert backend.expected_sample_rate == 32000
        assert backend.max_total_bytes == 1024 * 1024
    finally:
        asyncio.run(runtime.close())


def test_runtime_stdio_backend_maps_pcm_alias_to_raw_worker_media(tmp_path: Path) -> None:
    reference = tmp_path / "zh_normal.wav"
    reference.write_bytes(b"reference")
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "tts": {
                "enabled": True,
                "backend": "gpt_sovits_stdio",
                "endpoint": "http://127.0.0.1:9880/tts",
                "ref_audio_path": str(reference),
                "media_type": "pcm",
            }
        },
    )
    runtime = build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    try:
        backend = runtime.tts.backend
        media_index = backend.command.index("--media-type") + 1
        assert backend.command[media_index] == "raw"
    finally:
        asyncio.run(runtime.close())


def test_runtime_preserves_owned_tts_virtualenv_python_symlink(tmp_path: Path) -> None:
    engine_root = tmp_path / "engine"
    python_link = engine_root / ".venv" / "bin" / "python"
    python_link.parent.mkdir(parents=True)
    python_link.symlink_to(sys.executable)
    engine_config = engine_root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    engine_config.parent.mkdir(parents=True)
    engine_config.write_text("custom: {}\n", encoding="utf-8")
    reference = tmp_path / "zh_normal.wav"
    reference.write_bytes(b"reference")
    configuration = LoadedConfiguration(
        tmp_path / "config" / "app.yaml",
        {
            "tts": {
                "enabled": True,
                "backend": "gpt_sovits_stdio",
                "engine_root": "../engine",
                "python_executable": "../engine/.venv/bin/python",
                "ref_audio_path": str(reference),
            }
        },
    )

    runtime = build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    try:
        assert Path(runtime.tts.backend.command[0]) == python_link
        assert Path(runtime.tts.backend.command[0]).is_symlink()
    finally:
        asyncio.run(runtime.close())


def test_runtime_rejects_disabling_stdio_ready_handshake(tmp_path: Path) -> None:
    reference = tmp_path / "zh_normal.wav"
    reference.write_bytes(b"reference")
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "tts": {
                "enabled": True,
                "backend": "gpt_sovits_stdio",
                "endpoint": "http://127.0.0.1:9880/tts",
                "ref_audio_path": str(reference),
                "startup_handshake": False,
            }
        },
    )

    with pytest.raises(ConfigurationError, match="cannot be disabled"):
        validate_runtime_configuration(configuration)


@pytest.mark.parametrize("value", (True, 32000.5, "32000", 384001))
def test_runtime_rejects_invalid_tts_expected_sample_rate(tmp_path: Path, value: object) -> None:
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {"tts": {"expected_sample_rate": value}},
    )
    with pytest.raises(ConfigurationError, match="expected_sample_rate"):
        validate_runtime_configuration(configuration)


@pytest.mark.parametrize("value", (7, "speaker\nid", "speaker\x00id", "x" * 1025))
def test_runtime_rejects_invalid_tts_output_device_id(tmp_path: Path, value: object) -> None:
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {"tts": {"output_device_id": value}},
    )
    with pytest.raises(ConfigurationError, match="output_device_id"):
        validate_runtime_configuration(configuration)


def test_runtime_wires_pcm_audio_feature_analyzer_and_allows_opt_out(tmp_path: Path) -> None:
    enabled_configuration = LoadedConfiguration(
        tmp_path / "enabled.yaml",
        {"tts": {"enabled": False, "audio_features": {"enabled": True}}},
    )
    enabled_runtime = build_runtime(
        enabled_configuration,
        inspect_resources(tmp_path / "enabled-resources"),
    )
    try:
        assert enabled_runtime.tts._feature_analyzer is not None
    finally:
        asyncio.run(enabled_runtime.close())

    disabled_configuration = LoadedConfiguration(
        tmp_path / "disabled.yaml",
        {"tts": {"enabled": False, "audio_features": {"enabled": False}}},
    )
    disabled_runtime = build_runtime(
        disabled_configuration,
        inspect_resources(tmp_path / "disabled-resources"),
    )
    try:
        assert disabled_runtime.tts._feature_analyzer is None
    finally:
        asyncio.run(disabled_runtime.close())


def test_runtime_rejects_invalid_audio_feature_thresholds(tmp_path: Path) -> None:
    configuration = LoadedConfiguration(
        tmp_path / "invalid-audio-features.yaml",
        {
            "tts": {
                "enabled": False,
                "audio_features": {"silence_threshold": 0.8, "open_threshold": 0.2},
            }
        },
    )
    with pytest.raises(ConfigurationError, match="audio_features thresholds"):
        validate_runtime_configuration(configuration)


def test_runtime_rejects_enabled_gsv_without_reference_audio(tmp_path: Path) -> None:
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "tts": {
                "enabled": True,
                "backend": "gpt_sovits_http",
                "endpoint": "https://tts.invalid/tts",
            }
        },
    )
    with pytest.raises(ConfigurationError, match="reference"):
        validate_runtime_configuration(configuration)


def test_dispatching_platform_and_pet_controller_use_ui_dispatcher() -> None:
    class Platform:
        backend = "test"

        def capture_screen(self, *, scope="screen", region=None):
            return {"status": "available", "scope": scope, "region": region}

        def set_click_through(self, window, enabled):
            return {"window": window, "enabled": enabled}

        def move_overlay(self, window, x, y):
            return {"window": window, "x": x, "y": y}

    class Pet:
        def move_to(self, x, y, *, duration_ms=800):
            return {"x": x, "y": y, "duration_ms": duration_ms}

        def set_click_through(self, enabled):
            return {"enabled": enabled}

    class Dispatcher:
        def __init__(self):
            self.calls = []

        def invoke(self, function, *args, **kwargs):
            async def run():
                self.calls.append(function.__name__)
                return function(*args, **kwargs)

            return run()

    dispatcher = Dispatcher()
    platform = DispatchingDesktopPlatform(Platform(), dispatcher)
    pet = MutablePetController(Pet(), dispatcher)
    capture = asyncio.run(platform.capture_screen(scope="region", region={"x": 1}))
    moved = asyncio.run(pet.move_to(12, 34, duration_ms=100))
    click_through = asyncio.run(pet.set_click_through(True))
    assert capture["scope"] == "region"
    assert moved == {"x": 12, "y": 34, "duration_ms": 100}
    assert click_through == {"enabled": True}
    assert dispatcher.calls == ["capture_screen", "move_to", "set_click_through"]


def test_pet_controller_flattens_nested_async_host_result() -> None:
    """Qt 调度器和宿主各返回一层协程时，行为工具仍应得到最终回执。"""

    class Dispatcher:
        def invoke(self, function, *args, **kwargs):
            async def dispatch():
                return function(*args, **kwargs)

            return dispatch()

    class Pet:
        def move_to(self, x, y, *, duration_ms=800):
            async def move():
                return {"status": "available", "x": x, "y": y, "duration_ms": duration_ms}

            return move()

    controller = MutablePetController(Pet(), Dispatcher())

    result = asyncio.run(controller.move_to(12, 34, duration_ms=0))

    assert result == {"status": "available", "x": 12, "y": 34, "duration_ms": 0}


def test_pet_controller_invalidates_visual_actions_for_same_host_reload() -> None:
    class Target:
        def play_motion(self, _name: str):
            return {"status": "available"}

    interruptions: list[bool] = []
    controller = MutablePetController(Target())
    controller.set_interaction_interrupt(lambda: interruptions.append(True))
    controller.play_motion_internal("blink")
    before = controller.motion_generation()

    controller.invalidate_visual_actions()

    assert controller.motion_generation() == before + 1
    assert controller.motion_owner_generation() is None
    assert interruptions == [True]


def test_dispatching_window_capture_requires_background_window_resolution() -> None:
    class Platform:
        backend = "x11"

        def active_window(self):
            raise AssertionError("X11 window lookup must not run in the Qt dispatcher")

        def capture_screen(self, *, scope="screen", region=None, window_id=None):
            return {"status": "available", "scope": scope, "region": region, "window_id": window_id}

    class Dispatcher:
        def invoke(self, function, *args, **kwargs):
            return function(*args, **kwargs)

    platform = DispatchingDesktopPlatform(Platform(), Dispatcher())
    result = platform.capture_screen(scope="window")
    assert result == {
        "status": "unavailable",
        "reason": "window id must be resolved before Qt screen capture",
    }


def test_dispatching_platform_forwards_native_input_shape() -> None:
    """原生 Input Shape 请求必须经过与其它 Qt 操作相同的调度边界。"""

    class Platform:
        backend = "x11"

        def set_input_shape(self, window, region):
            return {"window": window, "region": region, "status": "available"}

    class Dispatcher:
        def __init__(self):
            self.calls = []

        def invoke(self, function, *args, **kwargs):
            self.calls.append(function.__name__)
            return function(*args, **kwargs)

    dispatcher = Dispatcher()
    platform = DispatchingDesktopPlatform(Platform(), dispatcher)
    region = object()
    result = platform.set_input_shape("window", region)
    assert result == {"window": "window", "region": region, "status": "available"}
    assert dispatcher.calls == ["set_input_shape"]


def test_click_through_guard_fails_closed_without_recovery_entry() -> None:
    controller = MutablePetController()
    controller.set_click_through_guard(lambda: False)
    assert controller.set_click_through(True) == {
        "status": "unavailable",
        "reason": "click-through recovery entry is unavailable",
    }

    def broken_guard() -> bool:
        raise RuntimeError("tray error")

    controller.set_click_through_guard(broken_guard)
    assert controller.set_click_through(True)["status"] == "unavailable"


def test_dispatcher_pending_cancellation_handles_done_callbacks() -> None:
    import concurrent.futures

    pending: set[concurrent.futures.Future[object]] = set()
    first: concurrent.futures.Future[object] = concurrent.futures.Future()
    second: concurrent.futures.Future[object] = concurrent.futures.Future()
    pending.update((first, second))
    first.add_done_callback(pending.discard)
    second.add_done_callback(pending.discard)
    _cancel_pending_futures(pending)
    assert first.cancelled() and second.cancelled()
    assert not pending


def test_runtime_binds_dispatcher_and_closes_when_loop_never_started(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    configuration = LoadedConfiguration(config_path, {})
    inventory = inspect_resources(tmp_path / "resources")
    runtime = build_runtime(configuration, inventory)

    class Dispatcher:
        def invoke(self, function, *args, **kwargs):
            return function(*args, **kwargs)

    runtime.bind_ui_dispatcher(Dispatcher())
    loop = RuntimeLoop(runtime)
    loop.stop()
    try:
        runtime.database.connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        pass
    else:
        raise AssertionError("database remained open after unstarted loop stop")


def test_runtime_loop_closes_coroutine_when_event_loop_closes_during_submit(
    tmp_path: Path,
) -> None:
    configuration = LoadedConfiguration(tmp_path / "config.yaml", {})
    inventory = inspect_resources(tmp_path / "resources")
    runtime = build_runtime(configuration, inventory)
    loop = RuntimeLoop(runtime)
    closed_loop = asyncio.new_event_loop()
    closed_loop.close()

    class AliveThread:
        def is_alive(self):
            return True

    loop._loop = closed_loop
    loop._thread = AliveThread()
    coroutine = asyncio.sleep(0)
    try:
        loop.submit(coroutine)
    except RuntimeError:
        pass
    else:
        raise AssertionError("closed event loop accepted a coroutine")
    finally:
        asyncio.run(runtime.close())


def test_startup_trigger_does_not_block_runtime_loop_start(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    values = {
        "tools": {"permissions": {"bypass_approval": True}},
        "scheduler": {
            "enabled": False,
            "triggers": [
                {
                    "trigger_id": "startup-move",
                    "event_name": "startup",
                    "action": {
                        "identity": "pet:move",
                        "arguments": {"x": 1, "y": 2},
                    },
                    "owner": "config",
                }
            ],
        },
    }
    configuration = LoadedConfiguration(config_path, values)
    inventory = inspect_resources(tmp_path / "resources")

    class Pet:
        def move_to(self, *_args, **_kwargs):
            return {"status": "moved"}

    class StuckDispatcher:
        def invoke(self, _function, *_args, **_kwargs):
            async def wait_forever():
                await asyncio.Event().wait()

            return wait_forever()

    runtime = build_runtime(
        configuration,
        inventory,
        pet_controller=MutablePetController(Pet()),
        ui_dispatcher=StuckDispatcher(),
    )
    loop = RuntimeLoop(runtime)
    loop.start(timeout=0.5)
    try:
        assert loop.running
    finally:
        loop.stop(timeout=0.5)
