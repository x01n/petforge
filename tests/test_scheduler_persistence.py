from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from db import Database, SchedulerRepository
from db.database import SCHEMA_VERSION
from services.scheduler import (
    SchedulerService,
    TriggerService,
    next_schedule_at,
    validate_schedule_expression,
)


def test_scheduler_repository_is_idempotent_without_changing_schema_version() -> None:
    database = Database(":memory:")
    first = SchedulerRepository(database, max_run_rows=2)
    second = SchedulerRepository(database, max_run_rows=2)

    assert first.load_tasks() == ()
    assert second.load_triggers() == ()
    version = database.connection.execute(
        "SELECT version FROM memory_schema_version ORDER BY version DESC LIMIT 1"
    ).fetchone()[0]
    assert version == SCHEMA_VERSION
    tables = {
        row[0]
        for row in database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'scheduler_%'"
        )
    }
    assert {"scheduler_tasks", "scheduler_triggers", "scheduler_runs"} <= tables


def test_scheduler_task_state_restores_and_records_failure() -> None:
    database = Database(":memory:")
    repository = SchedulerRepository(database)
    now = [100.0]
    scheduler = SchedulerService(clock=lambda: now[0], state_store=repository)
    scheduler.upsert(
        task_id="task-a",
        name="提醒",
        expression="every:1s",
        action={"identity": "pet:speak", "arguments": {"text": "hi"}},
        owner="profile:session",
    )

    now[0] = 102.0

    async def broken(_action, _task):
        raise RuntimeError("action failed")

    scheduler._action_runner = broken
    assert asyncio.run(scheduler.tick()) == ("task-a",)
    restored = SchedulerService(clock=lambda: now[0], state_store=repository)
    task = restored.list_tasks()[0]
    assert task["run_count"] == 1
    assert task["next_run_at"] == 103.0
    runs = repository.list_runs(kind="task", item_id="task-a")
    assert runs[0]["status"] == "failed"
    assert "action failed" in runs[0]["error_text"]


def test_scheduler_owner_isolation_and_json_action_validation() -> None:
    scheduler = SchedulerService()
    scheduler.upsert(
        task_id="owned",
        name="任务",
        expression="every:1s",
        action={"identity": "pet:ping"},
        owner="owner-a",
    )
    with pytest.raises(PermissionError):
        scheduler.upsert(
            task_id="owned",
            name="覆盖",
            expression="every:1s",
            action={"identity": "pet:ping"},
            owner="owner-b",
        )
    with pytest.raises(ValueError, match="JSON"):
        scheduler.upsert(
            task_id="bad",
            name="坏任务",
            expression="every:1s",
            action={"value": object()},
            owner="owner-a",
        )


def test_scheduler_rejects_unbounded_identifiers_and_non_finite_clock() -> None:
    scheduler = SchedulerService()
    with pytest.raises(ValueError, match="invalid"):
        scheduler.upsert(
            task_id="bad-owner",
            name="任务",
            expression="every:1s",
            action={"identity": "pet:ping"},
            owner="line\nbreak",
        )
    with pytest.raises(ValueError, match="invalid"):
        scheduler.upsert(
            task_id="bad-expression",
            name="任务",
            expression="every:1s\n",
            action={"identity": "pet:ping"},
            owner="owner",
        )
    broken_clock = SchedulerService(clock=lambda: float("nan"))
    with pytest.raises(ValueError, match="finite"):
        broken_clock.upsert(
            task_id="bad-clock",
            name="任务",
            expression="every:1s",
            action={"identity": "pet:ping"},
            owner="owner",
        )
    with pytest.raises(ValueError, match="finite"):
        asyncio.run(broken_clock.tick(now=float("inf")))


def test_scheduler_supports_local_daily_wall_clock_expression() -> None:
    assert validate_schedule_expression(" daily:03:00 ") == "daily:03:00"
    with pytest.raises(ValueError, match="daily"):
        validate_schedule_expression("daily:24:00")
    with pytest.raises(ValueError, match="daily"):
        validate_schedule_expression("daily:3:00")

    # 2026-08-28 02:59:00 local time must schedule the same day's 03:00.
    before = datetime(2026, 8, 28, 2, 59).timestamp()
    same_day = datetime(2026, 8, 28, 3, 0).timestamp()
    after = datetime(2026, 8, 28, 3, 1).timestamp()
    next_day = datetime(2026, 8, 29, 3, 0).timestamp()
    assert next_schedule_at("daily:03:00", before) == same_day
    assert next_schedule_at("daily:03:00", after) == next_day


def test_scheduler_daily_task_runs_once_and_rolls_to_next_local_day() -> None:
    current = [datetime(2026, 8, 28, 2, 59).timestamp()]
    calls: list[str] = []

    async def run(_action, task):
        calls.append(task.task_id)

    scheduler = SchedulerService(clock=lambda: current[0], action_runner=run)
    scheduler.upsert(
        task_id="daily-reminder",
        name="凌晨提醒",
        expression="daily:03:00",
        action={"identity": "pet:speak", "arguments": {"text": "休息一下"}},
        owner="owner",
    )
    assert scheduler.list_tasks()[0]["next_run_at"] == datetime(2026, 8, 28, 3, 0).timestamp()
    current[0] = datetime(2026, 8, 28, 3, 1).timestamp()
    assert asyncio.run(scheduler.tick()) == ("daily-reminder",)
    assert calls == ["daily-reminder"]
    assert scheduler.list_tasks()[0]["next_run_at"] == datetime(2026, 8, 29, 3, 0).timestamp()


def test_trigger_state_restores_debounce_and_records_success() -> None:
    database = Database(":memory:")
    repository = SchedulerRepository(database)
    now = [50.0]
    calls: list[str] = []

    async def run(_action, trigger, _payload):
        calls.append(trigger.trigger_id)

    triggers = TriggerService(clock=lambda: now[0], action_runner=run, state_store=repository)
    triggers.register(
        trigger_id="window",
        event_name="window_changed",
        action={"identity": "pet:look"},
        owner="profile:session",
        debounce_seconds=10,
    )
    assert asyncio.run(triggers.emit("window_changed", {"window_id": 1})) == ("window",)
    now[0] = 55.0
    assert asyncio.run(triggers.emit("window_changed", {"window_id": 2})) == ()
    restored = TriggerService(clock=lambda: now[0], state_store=repository)
    assert restored.list_triggers()[0]["last_fired_at"] == 50.0
    assert calls == ["window"]
    runs = repository.list_runs(kind="trigger", item_id="window")
    assert runs[0]["status"] == "completed"


def test_trigger_rejects_non_finite_clock() -> None:
    triggers = TriggerService(clock=lambda: float("nan"))
    triggers.register(
        trigger_id="broken-clock",
        event_name="startup",
        action={"identity": "pet:ping"},
        owner="owner",
    )
    with pytest.raises(ValueError, match="finite"):
        asyncio.run(triggers.emit("startup", {}))


def test_trigger_rejects_non_finite_debounce() -> None:
    triggers = TriggerService()
    with pytest.raises(ValueError, match="out of range"):
        triggers.register(
            trigger_id="bad-debounce",
            event_name="startup",
            action={"identity": "pet:ping"},
            owner="owner",
            debounce_seconds=float("nan"),
        )


def test_scheduler_start_rejects_non_finite_poll_interval() -> None:
    scheduler = SchedulerService()
    with pytest.raises(ValueError, match="poll_seconds"):
        asyncio.run(scheduler.start(poll_seconds=float("inf")))


def test_scheduler_repository_skips_corrupt_json_and_bounds_runs() -> None:
    database = Database(":memory:")
    repository = SchedulerRepository(database, max_run_rows=2)
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO scheduler_tasks (
                task_id, name, expression, action, owner, next_run_at,
                enabled, last_run_at, run_count, metadata, created_at, updated_at
            ) VALUES ('bad', '坏', 'every:1s', ?, 'owner', 1, 1, 0, 0, '{}', 1, 1)
            """,
            (json.dumps(["not-a-mapping"]),),
        )
    assert repository.load_tasks() == ()
    for index in range(3):
        repository.record_run(
            kind="task",
            item_id="task",
            owner="owner",
            started_at=float(index),
            finished_at=float(index),
            status="completed",
        )
    assert len(repository.list_runs(limit=100)) == 2


def test_scheduler_repository_skips_non_finite_snapshot_values() -> None:
    database = Database(":memory:")
    repository = SchedulerRepository(database)
    with database.transaction() as connection:
        connection.execute(
            """
            INSERT INTO scheduler_tasks (
                task_id, name, expression, action, owner, next_run_at,
                enabled, last_run_at, run_count, metadata, created_at, updated_at
            ) VALUES ('nan', '坏时间', 'every:1s', ?, 'owner', 1, 1, 0, 0, '{}', 1, 1)
            """,
            (json.dumps({"value": float("nan")}),),
        )
    assert repository.load_tasks() == ()


def test_persistence_failure_does_not_block_memory_scheduler() -> None:
    class BrokenStore:
        def load_tasks(self):
            raise RuntimeError("storage offline")

    scheduler = SchedulerService(state_store=BrokenStore())
    scheduler.upsert(
        task_id="memory-only",
        name="内存任务",
        expression="every:1s",
        action={"identity": "pet:ping"},
        owner="owner",
    )
    assert scheduler.list_tasks()[0]["task_id"] == "memory-only"
    assert "storage offline" in str(scheduler.status()["persistence_error"])


def test_scheduler_restore_rejects_non_boolean_enabled_snapshot() -> None:
    """自定义状态存储的字符串 false 不能被宽松 bool() 恢复为启用。"""

    class Store:
        def load_tasks(self):
            return (
                {
                    "task_id": "malformed-enabled",
                    "name": "坏快照",
                    "expression": "every:1s",
                    "action": {"identity": "pet:ping"},
                    "owner": "owner",
                    "next_run_at": 10.0,
                    "enabled": "false",
                },
            )

    scheduler = SchedulerService(state_store=Store())
    assert scheduler.list_tasks() == ()
    assert "enabled must be a boolean" in str(scheduler.status()["persistence_error"])
