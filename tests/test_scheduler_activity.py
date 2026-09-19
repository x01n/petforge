from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.runtime import validate_runtime_configuration
from config.loader import ConfigurationError, LoadedConfiguration
from services.scheduler import SchedulerService, TriggerService, UserActivityTracker


def test_mutable_controller_keeps_behavior_interrupt_separate_from_activity_notifier() -> None:
    from app.runtime import MutablePetController

    callbacks: dict[str, object] = {}

    class Target:
        def set_interaction_interrupt(self, callback):
            callbacks["interrupt"] = callback

        def set_activity_notifier(self, callback):
            callbacks["activity"] = callback

    controller = MutablePetController(Target())

    def interrupt() -> None:
        return None

    def activity() -> None:
        return None

    controller.set_interaction_interrupt(interrupt)
    controller.set_activity_notifier(activity)
    assert callbacks == {"interrupt": interrupt, "activity": activity}


def test_activity_tracker_emits_interaction_and_one_idle_edge() -> None:
    now = [0.0]
    seen: list[tuple[str, dict[str, object]]] = []

    async def run(_action, trigger, payload):
        seen.append((trigger.event_name, dict(payload)))

    triggers = TriggerService(action_runner=run, clock=lambda: now[0])
    triggers.register(
        trigger_id="interaction",
        event_name="user_interaction",
        action={"identity": "pet:play_motion"},
        owner="test",
    )
    triggers.register(
        trigger_id="idle",
        event_name="idle",
        action={"identity": "pet:play_motion"},
        owner="test",
    )
    activity = UserActivityTracker(
        triggers=triggers,
        idle_seconds=10,
        poll_seconds=1,
        clock=lambda: now[0],
    )

    async def scenario() -> None:
        await activity.start()
        assert activity.is_user_active() is False
        activity.record_interaction("test")
        await asyncio.sleep(0)
        assert activity.is_user_active() is True
        now[0] = 11
        await activity.poll_once()
        await activity.poll_once()
        assert [name for name, _payload in seen] == ["user_interaction", "idle"]
        assert seen[-1][1]["state"] == "idle"
        activity.record_interaction("again")
        await asyncio.sleep(0)
        assert [name for name, _payload in seen] == [
            "user_interaction",
            "idle",
            "user_interaction",
        ]
        await activity.stop()

    asyncio.run(scenario())


def test_activity_interaction_satisfies_user_active_trigger_condition() -> None:
    seen: list[str] = []

    async def run(_action, trigger, _payload):
        seen.append(trigger.trigger_id)

    triggers = TriggerService(action_runner=run, clock=lambda: 100.0)
    triggers.register(
        trigger_id="active-interaction",
        event_name="user_interaction",
        action={"identity": "pet:ping"},
        owner="test",
        metadata={"conditions": {"require_user_active": True}},
    )
    activity = UserActivityTracker(
        triggers=triggers,
        idle_seconds=10.0,
        poll_seconds=1.0,
        clock=lambda: 100.0,
    )

    async def scenario() -> None:
        await activity.start()
        activity.record_interaction("test")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert seen == ["active-interaction"]
        await activity.stop()

    asyncio.run(scenario())


def test_scheduler_require_user_active_skips_once_then_runs_when_active() -> None:
    now = [100.0]
    active = [False]
    calls: list[str] = []

    async def run(_action, task):
        calls.append(task.task_id)

    scheduler = SchedulerService(
        clock=lambda: now[0],
        action_runner=run,
        activity_provider=lambda: active[0],
    )
    scheduler.upsert(
        task_id="active-only",
        name="仅活跃",
        expression="every:1s",
        action={"identity": "pet:speak", "arguments": {"text": "hi"}},
        owner="test",
        metadata={"require_user_active": True},
    )
    now[0] = 102.0
    assert asyncio.run(scheduler.tick()) == ("active-only",)
    assert calls == []
    assert scheduler.status()["last_skip"] == {
        "task_id": "active-only",
        "reason": "user_inactive",
        "at": 102.0,
    }
    assert scheduler.list_tasks()[0]["metadata"] == {"require_user_active": True}
    active[0] = True
    now[0] = 103.0
    assert asyncio.run(scheduler.tick()) == ("active-only",)
    assert calls == ["active-only"]


def test_scheduler_activity_metadata_is_strict_and_configuration_is_validated() -> None:
    scheduler = SchedulerService()
    with pytest.raises(ValueError, match="require_user_active"):
        scheduler.upsert(
            task_id="bad",
            name="坏",
            expression="every:1s",
            action={"identity": "pet:play_motion", "arguments": {"name": "wave"}},
            owner="test",
            metadata={"require_user_active": "true"},
        )
    with pytest.raises(ConfigurationError, match="scheduler.activity.idle_seconds"):
        validate_runtime_configuration(
            LoadedConfiguration(
                Path("config.yaml"),
                {
                    "scheduler": {
                        "activity": {"idle_seconds": 0},
                        "tasks": [],
                        "triggers": [],
                    }
                },
            )
        )
    validate_runtime_configuration(
        LoadedConfiguration(
            Path("config.yaml"),
            {
                "scheduler": {
                    "activity": {"enabled": True, "idle_seconds": 10, "poll_seconds": 1},
                    "tasks": [
                        {
                            "name": "活跃提醒",
                            "expression": "daily:03:00",
                            "metadata": {"require_user_active": True},
                            "action": {"identity": "pet:speak", "arguments": {"text": "hi"}},
                        }
                    ],
                    "triggers": [
                        {
                            "trigger_id": "idle",
                            "event_name": "idle",
                            "action": {
                                "identity": "pet:play_motion",
                                "arguments": {"name": "blink"},
                            },
                        }
                    ],
                }
            },
        )
    )


def test_system_idle_provider_is_optional_and_explicit_interaction_wins() -> None:
    """启用 X11 provider 后仅低于阈值判定活跃，异常按非活跃处理。"""

    probes = [2.0]
    activity = UserActivityTracker(
        triggers=TriggerService(),
        idle_seconds=10,
        system_idle_provider="x11",
        system_idle_threshold_seconds=5,
        system_idle_probe=lambda: probes[0],
    )
    assert activity.status()["state"] == "unknown"
    assert activity.is_user_active() is True
    assert activity.status()["system_idle_status"] == "ready"
    probes[0] = 5.0
    assert activity.is_user_active() is False
    probes[0] = None
    assert activity.is_user_active() is False
    assert activity.status()["system_idle_status"] == "unavailable"

    now = [0.0]
    explicit = UserActivityTracker(
        triggers=TriggerService(),
        idle_seconds=10,
        clock=lambda: now[0],
        system_idle_provider="x11",
        system_idle_threshold_seconds=1,
        system_idle_probe=lambda: 100.0,
    )
    explicit.record_interaction("message")
    now[0] = 9.0
    assert explicit.is_user_active() is True


def test_windows_system_idle_provider_uses_injected_probe_and_reloads() -> None:
    """Windows provider 使用平台注入的空闲秒数，并保持严格阈值语义。"""

    probes = [2.0]
    activity = UserActivityTracker(
        triggers=TriggerService(),
        idle_seconds=10,
        system_idle_provider="windows",
        system_idle_threshold_seconds=5,
        system_idle_probe=lambda: probes[0],
    )
    assert activity.is_user_active() is True
    assert activity.status()["system_idle_provider"] == "windows"
    assert activity.status()["system_idle_status"] == "ready"

    probes[0] = 5.0
    assert activity.is_user_active() is False
    assert activity.status()["system_idle_available"] is True

    activity.reconfigure(system_idle_provider="disabled")
    assert activity.status()["system_idle_provider"] == "disabled"
    assert activity.status()["system_idle_status"] == "disabled"
    assert activity.is_user_active() is False

    activity.reconfigure(system_idle_provider="windows")
    assert activity.status()["system_idle_status"] == "unknown"
    probes[0] = 1.0
    assert activity.is_user_active() is True


def test_system_idle_provider_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="disabled, x11 or windows"):
        UserActivityTracker(triggers=TriggerService(), system_idle_provider="win")


def test_system_idle_configuration_is_strict_and_hot_reloadable() -> None:
    with pytest.raises(ConfigurationError, match="system_idle_provider"):
        validate_runtime_configuration(
            LoadedConfiguration(
                Path("config.yaml"),
                {
                    "scheduler": {
                        "activity": {"system_idle_provider": "X11"},
                        "tasks": [],
                        "triggers": [],
                    }
                },
            )
        )
    with pytest.raises(ConfigurationError, match="system_idle_threshold_seconds"):
        validate_runtime_configuration(
            LoadedConfiguration(
                Path("config.yaml"),
                {
                    "scheduler": {
                        "activity": {"system_idle_threshold_seconds": 0},
                        "tasks": [],
                        "triggers": [],
                    }
                },
            )
        )
    validate_runtime_configuration(
        LoadedConfiguration(
            Path("config.yaml"),
            {
                "scheduler": {
                    "activity": {
                        "system_idle_provider": "x11",
                        "system_idle_threshold_seconds": 30,
                    },
                    "tasks": [],
                    "triggers": [],
                }
            },
        )
    )
