from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

from app.loop import RuntimeLoop
from app.runtime import build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from core.adapters.direct import ProviderAdapter, ToolCallDelta
from core.events.types import TextDelta, TurnFinished
from services.model_routing import ChannelConfig, ModelRouter
from services.tools.types import RiskLevel, ToolCallContext, ToolSpec


class _ApprovalAdapter(ProviderAdapter):
    provider = "reload-test"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming", "tools"})

    async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
        del request, cancel_event
        yield ToolCallDelta(context, 0, "call-reload", "system:reload-danger", '{"value": 1}')
        yield TurnFinished(context, "tool_calls")


class _BlockingAdapter(ProviderAdapter):
    provider = "reload-blocking-test"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming"})

    async def stream(self, request, *, context=None, cancel_event=None) -> AsyncIterator:
        del request, cancel_event
        yield TextDelta(context, "部分输出")
        await asyncio.Event().wait()


def _configuration(tmp_path: Path, values: dict[str, object]) -> LoadedConfiguration:
    return LoadedConfiguration(tmp_path / "config.yaml", values)


def _ready_values(secret: str = "reload-secret") -> dict[str, object]:
    return {
        "llm": {
            "channels": [
                {
                    "id": "reload-primary",
                    "base_url": "https://reload.invalid/v1",
                    "model": "reload-model",
                    "api_key": secret,
                }
            ]
        }
    }


def test_model_reload_replaces_dialogue_router_when_idle(tmp_path: Path) -> None:
    initial = _configuration(tmp_path, {})
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    updated = _configuration(tmp_path, _ready_values())
    old_router = runtime.router

    try:
        result = asyncio.run(runtime.apply_model_configuration(updated))

        assert result["status"] == "reloaded"
        assert runtime.router is runtime.conversation.router
        assert runtime.router is not old_router
        assert runtime.configuration == updated
        assert runtime.interaction_state is not None
        assert runtime.interaction_state.snapshot.model.ready is True
        assert "reload-secret" not in repr(result)
        assert "reload-secret" not in repr(runtime.interaction_state.snapshot)
    finally:
        asyncio.run(runtime.close())


def test_model_reload_keeps_old_router_during_active_conversation(tmp_path: Path) -> None:
    initial = _configuration(tmp_path, {})
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    updated = _configuration(tmp_path, _ready_values("active-secret"))
    old_router = runtime.router
    old_configuration = runtime.configuration

    async def apply_while_active() -> dict[str, object]:
        task = asyncio.current_task()
        assert task is not None
        runtime.conversation._active_tasks.add(task)
        try:
            return dict(await runtime.apply_model_configuration(updated))
        finally:
            runtime.conversation._active_tasks.discard(task)

    try:
        result = asyncio.run(apply_while_active())

        assert result["status"] == "restart_required"
        assert runtime.router is old_router
        assert runtime.conversation.router is old_router
        assert runtime.configuration is old_configuration
        assert "active-secret" not in repr(result)
    finally:
        asyncio.run(runtime.close())


def test_model_reload_rejects_invalid_route_without_echoing_secret(tmp_path: Path) -> None:
    initial = _configuration(tmp_path, {})
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    old_router = runtime.router
    updated = _configuration(
        tmp_path,
        {
            "llm": {
                "channels": [
                    {
                        "id": "invalid",
                        "protocol": "unsupported-protocol",
                        "api_key": "invalid-secret",
                    }
                ]
            }
        },
    )

    try:
        result = asyncio.run(runtime.apply_model_configuration(updated))

        assert result["status"] == "unavailable"
        assert runtime.router is old_router
        assert runtime.configuration is initial
        assert "invalid-secret" not in repr(result)
    finally:
        asyncio.run(runtime.close())


def test_model_reload_runs_on_runtime_loop_without_gui_wait(tmp_path: Path) -> None:
    initial = _configuration(tmp_path, {})
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    loop = RuntimeLoop(runtime)
    updated = _configuration(tmp_path, _ready_values("loop-secret"))

    loop.start(timeout=1.0)
    try:
        result = loop.submit(runtime.apply_model_configuration(updated)).result(timeout=2.0)
        assert result["status"] == "reloaded"
        assert runtime.router is runtime.conversation.router
        assert runtime.configuration == updated
        assert "loop-secret" not in repr(result)
    finally:
        loop.stop(timeout=2.0)


def test_full_configuration_reload_updates_tts_behavior_and_stream_ui(tmp_path: Path) -> None:
    initial = _configuration(tmp_path, {})
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    updated = _configuration(
        tmp_path,
        {
            "tts": {
                "enabled": False,
                "backend": "text_only",
                "language": "ja",
                "queue_size": 8,
            },
            "behavior": {
                "enabled": False,
                "min_interval_seconds": 2,
                "max_interval_seconds": 4,
                "probability": 0.1,
                "movement": {"enabled": False},
            },
            "ui": {
                "stream": {
                    "show_reasoning": True,
                    "show_murmur": False,
                    "show_tool_status": False,
                    "max_bubble_chars": 800,
                }
            },
        },
    )
    try:
        result = asyncio.run(runtime.apply_configuration(updated))
        assert result["status"] == "reloaded"
        assert set(result["applied_sections"]) == {"tts", "behavior", "ui"}
        assert runtime.configuration == updated
        assert runtime.conversation.tts_language == "ja"
        assert runtime.tts.queue_size == 8
        assert runtime.conversation.presentation.show_reasoning is True
        assert runtime.conversation.presentation.show_murmur is False
        assert runtime.conversation.presentation.show_tool_status is False
        assert runtime.conversation.presentation.max_text_length == 800
        assert runtime.interaction_state.max_text_length == 800
    finally:
        asyncio.run(runtime.close())


def test_tts_voice_profiles_hot_reload_and_update_sanitized_diagnostics(tmp_path: Path) -> None:
    initial = _configuration(
        tmp_path,
        {
            "tts": {
                "enabled": True,
                "backend": "text_only",
                "language": "zh",
                "profiles": {"voice-a": {"backend": "text_only", "languages": ["zh"]}},
                "routing": {"default_profile": "voice-a"},
            }
        },
    )
    updated = _configuration(
        tmp_path,
        {
            "tts": {
                "enabled": True,
                "backend": "text_only",
                "language": "zh",
                "queue_size": 9,
                "profiles": {
                    "voice-b": {"backend": "text_only", "languages": ["zh"]},
                    "voice-c": {"backend": "text_only", "languages": ["jp"]},
                },
                "routing": {
                    "default_profile": "voice-b",
                    "fallback_profiles": ["voice-c"],
                },
            }
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    try:
        assert runtime.tts_diagnostics()["voices"] == ("voice-a",)

        result = asyncio.run(runtime.apply_configuration(updated))

        assert result["status"] == "reloaded"
        assert result["applied_sections"] == ("tts",)
        diagnostics = runtime.tts_diagnostics()
        assert diagnostics["voices"] == ("voice-b", "voice-c")
        assert diagnostics["default_profile"] == "voice-b"
        assert diagnostics["queue_size"] == 9
        assert "endpoint" not in diagnostics
        assert "api_key" not in diagnostics
    finally:
        asyncio.run(runtime.close())


def test_rendering_model_change_requires_attached_reload_host(tmp_path: Path) -> None:
    """没有 Qt 渲染宿主时保留旧运行配置并明确要求重启。"""

    initial = _configuration(tmp_path, {})
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    old_router = runtime.router
    old_tts = runtime.tts
    updated = _configuration(
        tmp_path,
        {"rendering": {"model": "live2d/model/alternate.model3.json"}},
    )
    try:
        result = asyncio.run(runtime.apply_configuration(updated))

        assert result["status"] == "restart_required"
        assert result["changed_sections"] == ("rendering",)
        assert result["restart_sections"] == ("rendering",)
        assert result["rendering_status"] == "unavailable"
        # 非渲染服务保持原实例，避免一次保存导致对话或语音中断。
        assert runtime.router is old_router
        assert runtime.tts is old_tts
        assert runtime.configuration is initial
    finally:
        asyncio.run(runtime.close())


def test_rendering_backend_preference_change_keeps_matching_live_host(
    tmp_path: Path,
) -> None:
    class RendererHost:
        renderer = SimpleNamespace(
            capabilities=SimpleNamespace(
                backend="web_live2d",
                available=True,
                expressions=(),
                motions=(),
                message="synthetic Web Live2D host",
            )
        )

        def __init__(self) -> None:
            self.reload_calls = 0
            self.performance_calls: list[dict[str, float]] = []

        def reload_resources(self, *_args, **_kwargs):
            self.reload_calls += 1
            return {"status": "reloaded"}

        def set_rendering_performance(self, **kwargs):
            self.performance_calls.append(dict(kwargs))
            return {"status": "available", "applied": dict(kwargs), "failed": ()}

    initial = _configuration(
        tmp_path,
        {"rendering": {"backend": "auto", "resource_root": str(tmp_path / "resources")}},
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    host = RendererHost()
    runtime.pet_controller.attach(host)
    updated = _configuration(
        tmp_path,
        {
            "rendering": {
                "backend": "web_live2d",
                "resource_root": str(tmp_path / "resources"),
                "frame_rate": 48,
                "geometry_audit_hz": 12,
            }
        },
    )

    try:
        result = asyncio.run(runtime.apply_configuration(updated))

        assert result["status"] == "reloaded"
        assert result["applied_sections"] == ("rendering",)
        assert result["rendering_status"] == "unchanged"
        assert result["rendering_model"] == ""
        assert host.reload_calls == 0
        assert host.performance_calls == [{"frame_rate": 48.0, "geometry_audit_hz": 12.0}]
        assert runtime.configuration == updated
    finally:
        asyncio.run(runtime.close())


def test_rendering_backend_preference_change_restarts_for_different_live_host(
    tmp_path: Path,
) -> None:
    class RendererHost:
        renderer = SimpleNamespace(
            capabilities=SimpleNamespace(
                backend="sprite",
                available=True,
                expressions=(),
                motions=(),
                message="synthetic sprite host",
            )
        )

    initial = _configuration(
        tmp_path,
        {"rendering": {"backend": "auto", "resource_root": str(tmp_path / "resources")}},
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    runtime.pet_controller.attach(RendererHost())
    updated = _configuration(
        tmp_path,
        {
            "rendering": {
                "backend": "web_live2d",
                "resource_root": str(tmp_path / "resources"),
            }
        },
    )

    try:
        result = asyncio.run(runtime.apply_configuration(updated))

        assert result["status"] == "restart_required"
        assert result["current_backend"] == "sprite"
        assert result["requested_backend"] == "web_live2d"
        assert runtime.configuration is initial
    finally:
        asyncio.run(runtime.close())


def test_tools_policy_and_visible_groups_hot_reload_without_restart(tmp_path: Path) -> None:
    initial = _configuration(
        tmp_path,
        {
            "tools": {
                "active_groups": ["pet_control"],
                "permissions": {"auto_allow_low_risk": True},
                "command_allowlist": [],
            }
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    updated = _configuration(
        tmp_path,
        {
            "tools": {
                "active_groups": ["desktop_observation"],
                "permissions": {
                    "auto_allow_low_risk": False,
                    "deny": ["pet:move"],
                },
                "command_allowlist": [sys.executable],
            }
        },
    )
    try:
        result = asyncio.run(runtime.apply_configuration(updated))

        assert result["status"] == "reloaded"
        assert result["changed_sections"] == ("tools",)
        assert result["applied_sections"] == ("tools",)
        assert result["tools_status"] == "reloaded"
        assert runtime.configuration == updated
        assert runtime.conversation.tool_groups == ("desktop_observation",)
        visible = tuple(spec.identity for spec in runtime.conversation._tool_definitions())
        assert "desktop:observe_foreground" in visible
        assert "pet:move" not in visible
        decision = runtime.permissions.evaluate(
            runtime.registry.require("pet:move"),
            ToolCallContext("profile", "session", "turn"),
        )
        assert decision.decision.value == "deny"
        command_handler = runtime.registry.require("system:run_command").handler
        command_result = asyncio.run(
            command_handler(
                {"command": [sys.executable, "-c", "pass"]},
                ToolCallContext("profile", "session", "command"),
            )
        )
        assert command_result["status"] == "completed"
    finally:
        asyncio.run(runtime.close())


def test_rendering_resources_hot_reload_through_attached_host(tmp_path: Path) -> None:
    class RendererHost:
        def __init__(self) -> None:
            self.calls = []
            self.performance_calls = []

        def reload_resources(self, resource_root, *, model_path=None, sprite_scale=None):
            self.calls.append((Path(resource_root), Path(model_path), sprite_scale))
            return {"status": "reloaded", "frame_count": 4}

        def set_rendering_performance(self, **kwargs):
            self.performance_calls.append(dict(kwargs))
            return {"status": "available", "applied": dict(kwargs), "failed": ()}

    initial_root = tmp_path / "initial-resources"
    updated_root = tmp_path / "updated-resources"
    initial = _configuration(
        tmp_path,
        {"rendering": {"backend": "auto", "resource_root": str(initial_root)}},
    )
    runtime = build_runtime(initial, inspect_resources(initial_root))
    host = RendererHost()
    runtime.pet_controller.attach(host)
    updated = _configuration(
        tmp_path,
        {
            "rendering": {
                "backend": "auto",
                "resource_root": str(updated_root),
                "model": "live2d/model/alternate.model3.json",
                "sprite_scale": 0.75,
            }
        },
    )
    try:
        result = asyncio.run(runtime.apply_configuration(updated))

        assert result["status"] == "reloaded"
        assert result["applied_sections"] == ("rendering",)
        assert result["rendering_status"] == "reloaded"
        assert result["rendering_model"] == "live2d/model/alternate.model3.json"
        assert host.calls == [
            (
                updated_root.resolve(),
                (updated_root / "live2d/model/alternate.model3.json").resolve(),
                0.75,
            )
        ]
        assert host.performance_calls == [{"frame_rate": 60.0, "geometry_audit_hz": 30.0}]
        assert runtime.configuration == updated
        assert runtime.inventory.root == updated_root.resolve()
    finally:
        asyncio.run(runtime.close())


def test_rendering_performance_only_reload_does_not_rebuild_resources(tmp_path: Path) -> None:
    class RendererHost:
        def __init__(self) -> None:
            self.reload_count = 0
            self.performance_calls: list[dict[str, float]] = []

        def reload_resources(self, *_args, **_kwargs):
            self.reload_count += 1
            return {"status": "reloaded"}

        def set_rendering_performance(self, **kwargs):
            self.performance_calls.append(dict(kwargs))
            return {"status": "available", "applied": dict(kwargs), "failed": ()}

    root = tmp_path / "resources"
    initial = _configuration(
        tmp_path,
        {"rendering": {"backend": "auto", "resource_root": str(root), "sprite_scale": 0.6}},
    )
    runtime = build_runtime(initial, inspect_resources(root))
    host = RendererHost()
    runtime.pet_controller.attach(host)
    updated = _configuration(
        tmp_path,
        {
            "rendering": {
                "backend": "auto",
                "resource_root": str(root),
                "sprite_scale": 0.6,
                "frame_rate": 45,
                "geometry_audit_hz": 15,
            }
        },
    )
    try:
        result = asyncio.run(runtime.apply_configuration(updated))
        assert result["status"] == "reloaded"
        assert host.reload_count == 0
        assert host.performance_calls == [{"frame_rate": 45.0, "geometry_audit_hz": 15.0}]
    finally:
        asyncio.run(runtime.close())


def test_full_configuration_reload_starts_and_stops_behavior_loop(tmp_path: Path) -> None:
    runtime = build_runtime(_configuration(tmp_path, {}), inspect_resources(tmp_path / "resources"))
    enabled = _configuration(
        tmp_path,
        {
            "behavior": {
                "enabled": True,
                "min_interval_seconds": 2,
                "max_interval_seconds": 2,
                "probability": 0,
                "movement": {"enabled": False},
            }
        },
    )
    disabled = _configuration(tmp_path, {"behavior": {"enabled": False}})

    async def scenario() -> None:
        await runtime.start_background()
        result = await runtime.apply_configuration(enabled)
        assert result["status"] == "reloaded"
        assert runtime.behavior.status()["running"] is True
        result = await runtime.apply_configuration(disabled)
        assert result["status"] == "reloaded"
        assert runtime.behavior.status()["running"] is False
        await runtime.close()

    asyncio.run(scenario())


def test_full_configuration_reload_keeps_old_runtime_when_tts_is_invalid(tmp_path: Path) -> None:
    initial = _configuration(tmp_path, {})
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    old_configuration = runtime.configuration
    old_backend = runtime.tts.backend
    updated = _configuration(tmp_path, {"tts": {"enabled": True, "backend": "unknown"}})
    try:
        result = asyncio.run(runtime.apply_configuration(updated))
        assert result["status"] == "unavailable"
        assert runtime.configuration is old_configuration
        assert runtime.tts.backend is old_backend
    finally:
        asyncio.run(runtime.close())


def test_full_configuration_reload_commits_model_after_other_sections_succeed(
    tmp_path: Path, monkeypatch
) -> None:
    initial = _configuration(tmp_path, _ready_values("old-secret"))
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    old_router = runtime.router
    updated_values = _ready_values("new-secret")
    updated_values["behavior"] = {"enabled": False}
    updated = _configuration(tmp_path, updated_values)

    async def fail_behavior(_values: object, *, on_commit=None) -> None:
        del on_commit
        raise RuntimeError("simulated behavior setter failure")

    monkeypatch.setattr(runtime, "_apply_behavior_configuration", fail_behavior)
    try:
        result = asyncio.run(runtime.apply_configuration(updated))
        assert result["status"] == "unavailable"
        assert runtime.router is old_router
        assert runtime.conversation.router is old_router
        assert runtime.configuration is initial
    finally:
        asyncio.run(runtime.close())


def test_runtime_configuration_watcher_applies_atomic_file_change(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "config:\n  reload:\n    enabled: true\n    interval_seconds: 0.05\n"
        "    debounce_seconds: 0\n    stable_checks: 1\n"
        "ui:\n  stream:\n    show_murmur: true\n",
        encoding="utf-8",
    )
    from config.loader import default_configuration_values, load_configuration

    loaded = load_configuration(
        path,
        defaults=default_configuration_values(
            resource_root=tmp_path / "resources", database_path=tmp_path / "db.sqlite3"
        ),
        environment={},
    )
    runtime = build_runtime(loaded, inspect_resources(tmp_path / "resources"))
    loop = RuntimeLoop(runtime)
    loop.start(timeout=2.0)
    try:
        path.write_text(
            "config:\n  reload:\n    enabled: true\n    interval_seconds: 0.05\n"
            "    debounce_seconds: 0\n    stable_checks: 1\n"
            "ui:\n  stream:\n    show_murmur: false\n",
            encoding="utf-8",
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and runtime.configuration_watcher.generation == 0:
            time.sleep(0.02)
        assert runtime.configuration_watcher.generation == 1
        assert runtime.conversation.presentation.show_murmur is False
    finally:
        loop.stop(timeout=3.0)


def test_full_configuration_reload_replaces_only_config_owned_schedule_items(
    tmp_path: Path,
) -> None:
    initial = _configuration(
        tmp_path,
        {
            "scheduler": {
                "tasks": [
                    {
                        "task_id": "config-old",
                        "name": "旧任务",
                        "expression": "every:1h",
                        "action": {"identity": "pet:play_motion", "arguments": {"name": "blink"}},
                    }
                ],
                "triggers": [],
            }
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    try:
        runtime.scheduler.upsert(
            task_id="user-task",
            name="用户任务",
            expression="every:1h",
            action={"identity": "pet:play_motion", "arguments": {"name": "blink"}},
            owner="user",
        )
        updated = _configuration(
            tmp_path,
            {
                "scheduler": {
                    "tasks": [
                        {
                            "task_id": "config-new",
                            "name": "新任务",
                            "expression": "every:2h",
                            "action": {
                                "identity": "pet:play_motion",
                                "arguments": {"name": "blink"},
                            },
                        }
                    ],
                    "triggers": [],
                }
            },
        )
        result = asyncio.run(runtime.apply_configuration(updated))
        assert result["status"] == "reloaded"
        task_ids = {item["task_id"] for item in runtime.scheduler.list_tasks()}
        assert task_ids == {"config-new", "user-task"}
    finally:
        asyncio.run(runtime.close())


def test_runtime_root_approval_and_reload_are_serialized(tmp_path: Path) -> None:
    runtime = build_runtime(
        _configuration(tmp_path, {}),
        inspect_resources(tmp_path / "resources"),
    )
    initial_router = ModelRouter(
        [ChannelConfig("initial", base_url="https://initial.invalid/v1", model="initial")],
        adapter_factory=lambda _channel: _ApprovalAdapter(),
    )
    runtime.router = initial_router
    runtime.conversation.router = initial_router

    async def dangerous_tool(arguments, _context):
        return {"ok": arguments["value"]}

    runtime.registry.register(
        ToolSpec(
            "system:reload-danger",
            "需要确认的测试工具",
            {"type": "object"},
            dangerous_tool,
            RiskLevel.HIGH,
        )
    )
    loop = RuntimeLoop(runtime)
    updated = _configuration(
        tmp_path,
        {
            "llm": {
                "channels": [
                    {
                        "id": "updated",
                        "base_url": "https://updated.invalid/v1",
                        "model": "updated",
                    }
                ]
            }
        },
    )
    loop.start(timeout=1.0)
    try:
        result = loop.submit(runtime.conversation.complete("需要确认")).result(timeout=3.0)
        snapshot = runtime.interaction_state.snapshot
        assert result.status == "approval_required"
        assert snapshot.approval is not None
        assert snapshot.busy is True
        assert runtime.conversation.has_active_conversation() is True

        blocked = loop.submit(runtime.apply_model_configuration(updated)).result(timeout=3.0)
        assert blocked["status"] == "restart_required"
        assert runtime.router is initial_router

        resumed = loop.submit(
            runtime.conversation.resume_approval(snapshot.approval.approval_id)
        ).result(timeout=3.0)
        assert resumed.status == "completed"
        assert runtime.interaction_state.snapshot.busy is False
        assert runtime.conversation.has_active_conversation() is False

        reloaded = loop.submit(runtime.apply_model_configuration(updated)).result(timeout=3.0)
        assert reloaded["status"] == "reloaded"
        assert runtime.router is runtime.conversation.router
        assert runtime.router is not initial_router
    finally:
        loop.stop(timeout=3.0)


def test_runtime_root_cancel_clears_streaming_snapshot(tmp_path: Path) -> None:
    runtime = build_runtime(
        _configuration(tmp_path, {}),
        inspect_resources(tmp_path / "resources"),
    )
    router = ModelRouter(
        [ChannelConfig("blocking", base_url="https://blocking.invalid/v1", model="blocking")],
        adapter_factory=lambda _channel: _BlockingAdapter(),
    )
    runtime.router = router
    runtime.conversation.router = router
    runtime.interaction_state.set_model_diagnostics(router.diagnostics())
    loop = RuntimeLoop(runtime)
    loop.start(timeout=1.0)
    context = runtime.conversation.begin_context(turn_id="cancel-root")
    future = loop.submit(runtime.conversation.complete("停止", context=context))
    try:
        deadline = time.monotonic() + 2.0
        while not runtime.interaction_state.snapshot.busy and time.monotonic() < deadline:
            time.sleep(0.01)
        assert runtime.interaction_state.snapshot.busy is True

        loop.submit(runtime.cancel_conversation(context)).result(timeout=2.0)
        try:
            future.result(timeout=2.0)
        except Exception:
            pass
        snapshot = runtime.interaction_state.snapshot
        assert snapshot.busy is False
        assert snapshot.streaming is False
        assert snapshot.finish_reason == "cancelled"
        assert snapshot.actions == ()
        assert runtime.conversation.presentation.snapshot.rendered_text == ""
    finally:
        future.cancel()
        loop.stop(timeout=3.0)
