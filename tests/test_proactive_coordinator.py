from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.runtime import build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from core.adapters.direct import ProviderAdapter, ToolCallDelta
from core.events.types import TextDelta, TurnFinished
from services.conversation import ConversationService
from services.model_routing import ChannelConfig, ModelRouter
from services.proactive import ProactiveCoordinator, ProactiveSettings
from services.scheduler import (
    DesktopWindowWatcher,
    SchedulerService,
    TriggerService,
    normalize_trigger_conditions,
    trigger_conditions_match,
)
from services.tools import PermissionService, ToolExecutionService, ToolRegistry
from services.tools.types import RiskLevel, ToolSpec


class _Memory:
    def build_context_prompt(self, query: str, *, max_chars: int) -> str:
        return f"相关记忆：{query}"[:max_chars]


class _Affection:
    def get_affection(self) -> int:
        return 42


class _Conversation:
    def __init__(self, *, release: asyncio.Event | None = None) -> None:
        self.release = release
        self.calls: list[str] = []
        self.active = False
        self.max_active = 0
        self._active_count = 0
        self.cancelled = False
        self.pending = []

    def has_active_conversation(self) -> bool:
        return self.active or bool(self.pending)

    async def complete(self, prompt: str, **_kwargs: object):
        self.calls.append(prompt)
        self.active = True
        self._active_count += 1
        self.max_active = max(self.max_active, self._active_count)
        try:
            if self.release is not None:
                await self.release.wait()
            return SimpleNamespace(status="completed")
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        finally:
            self._active_count -= 1
            self.active = False

    def pending_approvals_for_ui(self):
        return tuple(self.pending)


def _settings(*rules: dict[str, object], **overrides: object) -> ProactiveSettings:
    values: dict[str, object] = {
        "enabled": True,
        "hourly_budget": 6,
        "daily_budget": 24,
        "global_cooldown_seconds": 0,
        "dedupe_seconds": 0,
        "rules": list(rules),
    }
    values.update(overrides)
    return ProactiveSettings.from_mapping(values)


def _coordinator(
    conversation: object,
    settings: ProactiveSettings,
    *,
    clock=lambda: 1_000.0,
    gate=lambda: {},
) -> ProactiveCoordinator:
    return ProactiveCoordinator(
        conversation,
        memory=_Memory(),
        affection=_Affection(),
        settings=settings,
        activity_provider=lambda: True,
        gate_provider=gate,
        clock=clock,
    )


def test_trigger_conditions_cover_modes_duration_activity_and_safe_regex() -> None:
    payload = {
        "process_name": "Code",
        "app_id": "org.code.Editor",
        "title": "MeaPet - main.py",
        "active_for_seconds": 600,
        "user_active": True,
    }
    conditions = normalize_trigger_conditions(
        {
            "process_name": {"mode": "exact", "value": "code"},
            "app_id": {"mode": "prefix", "value": "org.code"},
            "title": {"mode": "contains", "value": "main.py"},
            "min_duration_seconds": 300,
            "require_user_active": True,
            "cooldown_seconds": 60,
        }
    )
    assert trigger_conditions_match(conditions, payload) is True
    regex = normalize_trigger_conditions(
        {"title": {"mode": "regex", "value": r"^MeaPet - [a-z.]+$"}}
    )
    assert trigger_conditions_match(regex, payload) is True
    with pytest.raises(ValueError, match="unsupported construct"):
        normalize_trigger_conditions({"title": {"mode": "regex", "value": r"^(a+)+$"}})


def test_window_active_duration_condition_queues_one_proactive_turn() -> None:
    current = [0.0]

    class Platform:
        def foreground_window(self):
            return {
                "status": "available",
                "window_id": "1",
                "process_name": "code",
                "app_id": "org.code.Editor",
                "title": "project",
            }

    conversation = _Conversation()
    coordinator = _coordinator(conversation, _settings(), clock=lambda: current[0])

    async def run_action(action, _trigger, payload):
        return await coordinator.notify(
            "window_active",
            payload,
            instruction=str(action["arguments"]["instruction"]),
        )

    triggers = TriggerService(action_runner=run_action, clock=lambda: current[0])
    triggers.register(
        trigger_id="editor-stable",
        event_name="window_active",
        action={
            "identity": "proactive:run",
            "arguments": {"instruction": "简短关心用户。"},
        },
        owner="config",
        metadata={
            "conditions": {
                "process_name": {"mode": "exact", "value": "code"},
                "min_duration_seconds": 10,
                "require_user_active": True,
            }
        },
    )
    watcher = DesktopWindowWatcher(
        platform=Platform(),
        triggers=triggers,
        activity_provider=lambda: True,
        clock=lambda: current[0],
    )

    async def scenario() -> None:
        await watcher.poll_once()
        current[0] = 9
        await watcher.poll_once()
        assert conversation.calls == []
        current[0] = 11
        await watcher.poll_once()
        await coordinator.wait_idle()
        assert len(conversation.calls) == 1
        assert "project" in conversation.calls[0]
        assert "相关记忆" in conversation.calls[0]
        assert "好感度：42" in conversation.calls[0]
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_queue_recovers_if_turn_finished_wake_arrives_too_early() -> None:
    class BusyConversation:
        def __init__(self) -> None:
            self.active = True
            self.calls: list[str] = []

        def has_active_conversation(self) -> bool:
            return self.active

        async def complete(self, prompt: str, **_kwargs: object):
            self.calls.append(prompt)
            return SimpleNamespace(status="completed")

        def pending_approvals_for_ui(self):
            return ()

    conversation = BusyConversation()
    coordinator = _coordinator(
        conversation,
        _settings({"id": "early", "event": "custom", "instruction": "稍后提醒。"}),
    )

    async def scenario() -> None:
        queued = await coordinator.notify("custom", {"label": "early"})
        assert queued["status"] == "queued"
        await asyncio.sleep(0)
        assert coordinator.diagnostics()["status"] == "waiting_for_conversation"
        coordinator.wake()
        conversation.active = False
        await coordinator.wait_idle()
        assert len(conversation.calls) == 1
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_queue_waits_for_conversation_end_and_wakes() -> None:
    class BusyConversation:
        def __init__(self) -> None:
            self.active = True
            self.calls: list[str] = []

        def has_active_conversation(self) -> bool:
            return self.active

        async def complete(self, prompt: str, **_kwargs: object):
            self.calls.append(prompt)
            return SimpleNamespace(status="completed")

        def pending_approvals_for_ui(self):
            return ()

    conversation = BusyConversation()
    coordinator = _coordinator(
        conversation,
        _settings({"id": "wait", "event": "custom", "instruction": "稍后提醒。"}),
    )

    async def scenario() -> None:
        queued = await coordinator.notify("custom", {"label": "wait"})
        assert queued["status"] == "queued"
        await asyncio.sleep(0)
        assert coordinator.diagnostics()["status"] == "waiting_for_conversation"
        assert conversation.calls == []
        conversation.active = False
        coordinator.wake()
        await coordinator.wait_idle()
        assert len(conversation.calls) == 1
        assert coordinator.diagnostics()["completed"] == 1
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_event_queues_all_matching_rules_in_declaration_order() -> None:
    conversation = _Conversation()
    coordinator = _coordinator(
        conversation,
        _settings(
            {"id": "first", "event": "custom", "instruction": "第一条。"},
            {"id": "second", "event": "custom", "instruction": "第二条。"},
        ),
    )

    async def scenario() -> None:
        result = await coordinator.notify("custom", {"label": "both"})
        assert result == {"status": "queued", "pending": 2, "accepted": 2}
        await coordinator.wait_idle()
        assert len(conversation.calls) == 2
        assert "第一条。" in conversation.calls[0]
        assert "第二条。" in conversation.calls[1]
        await coordinator.close()

    asyncio.run(scenario())


def test_terminal_event_queues_while_finish_callback_still_active() -> None:
    class BusyConversation:
        def __init__(self) -> None:
            self.active = True
            self.calls: list[str] = []

        def has_active_conversation(self) -> bool:
            return self.active

        async def complete(self, prompt: str, **_kwargs: object):
            self.calls.append(prompt)
            return SimpleNamespace(status="completed")

        def pending_approvals_for_ui(self):
            return ()

    conversation = BusyConversation()
    coordinator = _coordinator(
        conversation,
        _settings(
            {"id": "finished", "event": "conversation_finished", "instruction": "完成后提醒。"}
        ),
        gate=lambda: {"dialogue_active": conversation.active},
    )

    async def scenario() -> None:
        queued = await coordinator.notify("conversation_finished", {"status": "completed"})
        assert queued["status"] == "queued"
        await asyncio.sleep(0)
        assert coordinator.diagnostics()["status"] == "waiting_for_conversation"
        conversation.active = False
        coordinator.wake()
        await coordinator.wait_idle()
        assert len(conversation.calls) == 1
        assert coordinator.diagnostics()["completed"] == 1
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_reconfigure_consumes_pending_approval_and_stops_worker() -> None:
    class ApprovalConversation:
        def __init__(self) -> None:
            self.pending = [SimpleNamespace(approval_id="approval-reconfigure")]
            self.denied: list[str] = []

        def has_active_conversation(self) -> bool:
            return bool(self.pending)

        async def complete(self, _prompt: str, **_kwargs: object):
            return SimpleNamespace(status="completed")

        def pending_approvals_for_ui(self):
            return tuple(self.pending)

        def deny_approval(self, approval_id: str) -> bool:
            key = str(approval_id)
            self.denied.append(key)
            self.pending = [item for item in self.pending if item.approval_id != key]
            return True

    conversation = ApprovalConversation()
    coordinator = _coordinator(
        conversation,
        _settings({"id": "approval", "event": "custom", "instruction": "执行。"}),
    )

    async def scenario() -> None:
        queued = await coordinator.notify("custom", {"label": "approval"})
        assert queued["status"] == "queued"
        await asyncio.sleep(0)
        assert coordinator.running is True
        coordinator.reconfigure(ProactiveSettings(enabled=False))
        await coordinator.wait_idle()
        assert conversation.denied == ["approval-reconfigure"]
        assert coordinator.diagnostics()["status"] == "disabled"
        assert coordinator.diagnostics()["pending_events"] == 0
        assert coordinator.running is False
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_reconfigure_enabled_resets_cancelled_status() -> None:
    conversation = _Conversation()
    coordinator = _coordinator(
        conversation,
        _settings({"id": "custom", "event": "custom", "instruction": "执行。"}),
    )

    async def scenario() -> None:
        coordinator.interrupt()
        assert coordinator.diagnostics()["status"] == "interrupted"
        coordinator.reconfigure(
            _settings({"id": "new", "event": "custom", "instruction": "重新执行。"})
        )
        assert coordinator.diagnostics()["status"] == "idle"
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_gate_provider_failure_fails_closed_without_worker() -> None:
    conversation = _Conversation()

    def broken_gate() -> Mapping[str, object]:
        raise RuntimeError("gate unavailable")

    coordinator = _coordinator(
        conversation,
        _settings({"id": "custom", "event": "custom", "instruction": "执行。"}),
        gate=broken_gate,
    )

    async def scenario() -> None:
        result = await coordinator.notify("custom", {"label": "gate"})
        assert result == {"status": "blocked", "reason": "gate"}
        assert coordinator.running is False
        assert coordinator.diagnostics()["dropped"] == 1
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_clock_failure_returns_unavailable_without_queueing() -> None:
    conversation = _Conversation()
    coordinator = _coordinator(
        conversation,
        _settings({"id": "clock", "event": "custom", "instruction": "执行。"}),
        clock=lambda: (_ for _ in ()).throw(RuntimeError("clock unavailable")),
    )

    async def scenario() -> None:
        result = await coordinator.notify("custom", {})
        assert result == {"status": "unavailable", "reason": "clock"}
        assert coordinator.diagnostics()["pending_events"] == 0
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_interrupt_from_foreign_thread_is_loop_safe() -> None:
    release = asyncio.Event()
    conversation = _Conversation(release=release)
    coordinator = _coordinator(
        conversation,
        _settings({"id": "custom", "event": "custom", "instruction": "执行。"}),
    )

    async def scenario() -> None:
        queued = await coordinator.notify("custom", {"label": "thread"})
        assert queued["status"] == "queued"
        await asyncio.sleep(0)
        assert conversation._active_count == 1
        thread = threading.Thread(target=coordinator.interrupt)
        thread.start()
        thread.join()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert conversation.cancelled is True
        assert coordinator.diagnostics()["pending_events"] == 0
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_wait_idle_observes_worker_restarted_by_finally() -> None:
    class Conversation:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def has_active_conversation(self) -> bool:
            return False

        async def complete(self, prompt: str, **_kwargs: object):
            self.calls.append(prompt)
            await asyncio.sleep(0)
            return SimpleNamespace(status="completed")

        def pending_approvals_for_ui(self):
            return ()

    conversation = Conversation()
    coordinator = _coordinator(
        conversation,
        _settings({"id": "restart", "event": "custom", "instruction": "执行。"}),
    )

    async def scenario() -> None:
        first = await coordinator.notify("custom", {"label": "first"})
        assert first["status"] == "queued"
        await asyncio.sleep(0)
        second = await coordinator.notify("custom", {"label": "second"})
        assert second["status"] == "queued"
        await coordinator.wait_idle()
        assert len(conversation.calls) == 2
        assert coordinator.diagnostics()["pending_events"] == 0
        await coordinator.close()

    asyncio.run(scenario())


def test_proactive_dedupe_budget_single_flight_and_gates() -> None:
    now = [1_000.0]
    release = asyncio.Event()
    conversation = _Conversation(release=release)
    settings = _settings(
        {"id": "custom", "event": "custom", "instruction": "处理自定义事件。"},
        hourly_budget=1,
        daily_budget=1,
        dedupe_seconds=60,
    )
    coordinator = _coordinator(conversation, settings, clock=lambda: now[0])

    async def scenario() -> None:
        first = await coordinator.notify("custom", {"label": "one"})
        assert first["status"] == "queued"
        await asyncio.sleep(0)
        duplicate = await coordinator.notify("custom", {"label": "one"})
        assert duplicate["status"] == "duplicate"
        second = await coordinator.notify("custom", {"label": "two"})
        assert second["status"] == "queued"
        assert conversation.max_active == 1
        release.set()
        await coordinator.wait_idle()
        diagnostics = coordinator.diagnostics()
        assert diagnostics["completed"] == 1
        assert diagnostics["budget_rejections"] == 1
        assert conversation.max_active == 1
        await coordinator.close()

    asyncio.run(scenario())

    blocked = _coordinator(
        _Conversation(),
        settings,
        gate=lambda: {"dialogue_active": True},
    )
    assert asyncio.run(blocked.notify("custom", {"label": "blocked"}))["status"] == "blocked"


def test_daily_three_scheduler_runs_only_when_user_active() -> None:
    before = datetime(2026, 8, 28, 2, 59).timestamp()
    due = datetime(2026, 8, 28, 3, 0).timestamp()
    current = [before]
    active = [False]
    conversation = _Conversation()
    coordinator = _coordinator(
        conversation,
        _settings(),
        clock=lambda: current[0],
    )

    async def run(action, task):
        return await coordinator.notify(
            "daily",
            {
                "task_id": task.task_id,
                "name": task.name,
                "expression": task.expression,
                "user_active": active[0],
            },
            instruction=str(action["arguments"]["instruction"]),
        )

    scheduler = SchedulerService(
        clock=lambda: current[0],
        action_runner=run,
        activity_provider=lambda: active[0],
    )
    scheduler.upsert(
        task_id="three-am",
        name="凌晨提醒",
        expression="daily:03:00",
        action={
            "identity": "proactive:run",
            "arguments": {"instruction": "提醒用户休息。"},
        },
        owner="config",
        metadata={"require_user_active": True},
    )

    async def scenario() -> None:
        current[0] = due
        await scheduler.tick()
        assert conversation.calls == []
        active[0] = True
        current[0] = datetime(2026, 8, 29, 3, 0).timestamp()
        await scheduler.tick()
        await coordinator.wait_idle()
        assert len(conversation.calls) == 1
        await coordinator.close()

    asyncio.run(scenario())


class _ApprovalAdapter(ProviderAdapter):
    provider = "proactive-test"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming", "tools"})

    def __init__(self) -> None:
        self.rounds = 0

    async def stream(self, request, *, context=None, cancel_event=None):
        del request, cancel_event
        self.rounds += 1
        if self.rounds == 1:
            yield ToolCallDelta(context, 0, "danger-call", "system:danger", "{}")
            yield TurnFinished(context, "tool_calls")
        else:
            yield TextDelta(context, "已完成。")
            yield TurnFinished(context, "stop")


def test_proactive_high_risk_tool_uses_existing_approval_continuation() -> None:
    adapter = _ApprovalAdapter()
    router = ModelRouter(
        [
            ChannelConfig(
                "proactive",
                base_url="https://proactive.invalid/v1",
                model="proactive-model",
                capabilities=("streaming", "tools"),
            )
        ],
        routes={"proactive": {"channel": "proactive"}},
        adapter_factory=lambda _channel: adapter,
    )
    called = 0

    async def danger(_arguments, _context):
        nonlocal called
        called += 1
        return {"status": "ok"}

    registry = ToolRegistry()
    registry.register(
        ToolSpec("system:danger", "危险操作", {"type": "object"}, danger, RiskLevel.HIGH)
    )
    executor = ToolExecutionService(registry, PermissionService())
    conversation = ConversationService(
        router,
        tools=executor,
        task="proactive",
        max_tool_rounds=2,
    )
    coordinator = _coordinator(
        conversation,
        _settings({"id": "event", "event": "custom", "instruction": "执行操作。"}),
    )

    async def scenario() -> None:
        await coordinator.notify("custom", {"label": "approval"})
        await coordinator.wait_idle()
        assert called == 0
        assert coordinator.diagnostics()["status"] == "approval_required"
        pending = coordinator.pending_approvals_for_ui()
        assert len(pending) == 1
        await coordinator.continue_approval(pending[0].approval_id)
        assert called == 1
        assert adapter.rounds == 2
        await coordinator.close()
        await router.aclose()

    asyncio.run(scenario())


def test_runtime_projects_proactive_approval_and_routes_continuation(tmp_path: Path) -> None:
    configuration = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "llm": {
                "channels": [
                    {
                        "id": "proactive",
                        "base_url": "https://proactive.invalid/v1",
                        "model": "proactive-model",
                        "capabilities": ["streaming", "tools"],
                    }
                ],
                "routing": {"proactive": {"channel": "proactive"}},
            },
            "proactive": {
                "enabled": True,
                "global_cooldown_seconds": 0,
                "dedupe_seconds": 0,
                "rules": [
                    {
                        "id": "approval",
                        "event": "custom",
                        "instruction": "执行需要确认的操作。",
                    }
                ],
            },
        },
    )
    runtime = build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    adapter = _ApprovalAdapter()
    router = ModelRouter(
        [
            ChannelConfig(
                "proactive",
                base_url="https://proactive.invalid/v1",
                model="proactive-model",
                capabilities=("streaming", "tools"),
            )
        ],
        routes={"proactive": {"channel": "proactive"}},
        adapter_factory=lambda _channel: adapter,
    )
    runtime.router = router
    runtime.conversation.router = router
    assert runtime.proactive is not None
    runtime.proactive.conversation.router = router
    runtime.pet_controller.attach(
        SimpleNamespace(
            is_user_interacting=lambda: False,
            is_window_locked=lambda: False,
        )
    )
    called = 0

    async def danger(_arguments, _context):
        nonlocal called
        called += 1
        return {"status": "ok"}

    runtime.registry.register(
        ToolSpec("system:danger", "危险操作", {"type": "object"}, danger, RiskLevel.HIGH)
    )

    async def scenario() -> None:
        queued = await runtime.notify_proactive_event("custom", {"label": "approval"})
        assert queued["status"] == "queued"
        await runtime.proactive.wait_idle()
        snapshot = runtime.interaction_state.snapshot
        assert snapshot.phase.value == "approval_required"
        assert snapshot.approval is not None
        pending = runtime.pending_approvals_for_ui()
        assert len(pending) == 1
        assert snapshot.approval.approval_id == pending[0].approval_id
        await runtime.continue_approval(pending[0].approval_id)
        assert called == 1
        await runtime.close()

    asyncio.run(scenario())


def test_proactive_close_cancels_active_model_and_logs_no_window_title(caplog) -> None:
    release = asyncio.Event()
    conversation = _Conversation(release=release)
    coordinator = _coordinator(
        conversation,
        _settings({"id": "window", "event": "window_active", "instruction": "观察。"}),
    )
    secret_title = "private-window-title-must-not-log"

    async def scenario() -> None:
        with caplog.at_level(logging.INFO):
            await coordinator.notify(
                "window_active",
                {"title": secret_title, "process_name": "code", "active_for_seconds": 60},
            )
            await asyncio.sleep(0)
            await coordinator.close()
        assert conversation.cancelled is True
        assert coordinator.diagnostics()["status"] == "closed"

    asyncio.run(scenario())
    assert secret_title not in caplog.text


def test_runtime_proactive_configuration_hot_reload_and_diagnostics(tmp_path: Path) -> None:
    initial = LoadedConfiguration(
        tmp_path / "config.yaml",
        {
            "llm": {
                "channels": [
                    {
                        "id": "primary",
                        "base_url": "https://proactive.invalid/v1",
                        "model": "model",
                        "capabilities": ["streaming", "tools"],
                    }
                ],
                "routing": {"proactive": {"channel": "primary"}},
            },
            "proactive": {"enabled": False, "rules": []},
        },
    )
    runtime = build_runtime(initial, inspect_resources(tmp_path / "resources"))
    updated = LoadedConfiguration(
        initial.path,
        {
            **initial.values,
            "proactive": {
                "enabled": True,
                "hourly_budget": 3,
                "daily_budget": 9,
                "rules": [{"id": "custom", "event": "custom", "instruction": "简短回应。"}],
            },
        },
    )

    async def scenario() -> None:
        result = await runtime.apply_configuration(updated)
        assert result["status"] == "reloaded"
        assert "proactive" in result["applied_sections"]
        diagnostics = runtime.module_diagnostics()["proactive"]
        assert diagnostics["enabled"] is True
        assert diagnostics["rule_count"] == 1
        probe = await runtime.test_module("proactive")
        assert probe["ready"] is True
        await runtime.close()

    asyncio.run(scenario())
