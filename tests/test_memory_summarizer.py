from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest

from app.runtime import build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from db.database import Database
from services.memory.service import MemoryService, MemorySettings
from services.memory.summarizer import MemorySummaryCoordinator


class _SummaryRouter:
    def __init__(self, *, text: str = "用户喜欢猫咪。", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.requests = []
        self.resolved_tasks: list[str] = []

    def resolve(self, task: str):
        self.resolved_tasks.append(task)
        assert task == "memory"
        return SimpleNamespace(primary=SimpleNamespace(selected_model="summary-model"))

    async def complete(self, request, *, task: str, context):
        self.requests.append((request, task, context))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


def _memory() -> tuple[Database, MemoryService]:
    database = Database(":memory:")
    service = MemoryService(
        database,
        settings=MemorySettings(
            summarize_every_n=2,
            summary_min_messages=4,
            summary_chat_limit=20,
        ),
    )
    for role, content in (
        ("user", "我喜欢猫咪"),
        ("assistant", "我记住了"),
        ("user", "我习惯晚上听音乐"),
        ("assistant", "之后会照顾这个习惯"),
    ):
        service.add_chat(role, content)
    service.increment_message_counter()
    service.increment_message_counter()
    return database, service


def test_summary_coordinator_consumes_due_batch_without_tools() -> None:
    database, memory = _memory()
    router = _SummaryRouter()
    coordinator = MemorySummaryCoordinator(memory, lambda: router, clock=lambda: 2_000_000.0)

    result = asyncio.run(coordinator.run_once())

    assert result["status"] == "completed"
    assert result["reason"] == "message_count"
    assert result["source_count"] == 4
    assert len(router.requests) == 1
    request, task, context = router.requests[0]
    assert task == "memory"
    assert router.resolved_tasks == ["memory"]
    assert request.model == "summary-model"
    assert request.tools == ()
    assert request.tool_choice == "none"
    assert request.metadata == {"purpose": "memory_summary", "trigger": "message_count"}
    assert context.session_id == "memory-summary"
    summaries = memory.list(memory_type="summary")
    assert len(summaries) == 1
    assert summaries[0].content == "用户喜欢猫咪。"
    assert memory.prepare_summarization_batch(force=True) is None
    assert coordinator.status().completed == 1
    database.close()


def test_summary_coordinator_failure_keeps_source_batch_for_retry(caplog) -> None:
    database, memory = _memory()
    router = _SummaryRouter(error=RuntimeError("provider details must stay private"))
    coordinator = MemorySummaryCoordinator(memory, lambda: router, clock=lambda: 2_000_000.0)
    before = memory.prepare_summarization_batch(force=True)
    assert before is not None

    with caplog.at_level(logging.WARNING, logger="services.memory.summarizer"):
        result = asyncio.run(coordinator.run_once())

    assert result == {
        "status": "failed",
        "reason": "message_count",
        "error": "RuntimeError",
    }
    after = memory.prepare_summarization_batch(force=True)
    assert after is not None
    assert after.source_ids == before.source_ids
    assert after.messages == before.messages
    assert memory.list(memory_type="summary") == ()
    assert coordinator.status().failures == 1
    public_status = coordinator.status().public()
    assert "provider details must stay private" not in caplog.text
    assert "我喜欢猫咪" not in caplog.text
    assert "provider details must stay private" not in repr(public_status)
    assert "我喜欢猫咪" not in repr(public_status)
    assert "RuntimeError" in caplog.text
    database.close()


def test_summary_coordinator_start_stop_is_idempotent() -> None:
    async def scenario() -> None:
        database = Database(":memory:")
        memory = MemoryService(database)
        coordinator = MemorySummaryCoordinator(
            memory,
            lambda: _SummaryRouter(),
            poll_seconds=1,
        )
        await coordinator.start()
        await coordinator.start()
        assert coordinator.running is True
        await coordinator.pause()
        await coordinator.pause()
        assert coordinator.running is False
        assert coordinator.wake() is True
        await coordinator.start()
        assert coordinator.running is True
        await coordinator.stop()
        await coordinator.stop()
        assert coordinator.running is False
        assert coordinator.wake() is False
        with pytest.raises(RuntimeError, match="closed"):
            await coordinator.start()
        database.close()

    asyncio.run(scenario())


def test_summary_coordinator_wake_from_foreign_thread_is_loop_safe() -> None:
    async def scenario() -> None:
        database = Database(":memory:")
        memory = MemoryService(database)
        coordinator = MemorySummaryCoordinator(memory, lambda: _SummaryRouter(), poll_seconds=60)
        await coordinator.start()
        coordinator._wake.clear()
        thread = threading.Thread(target=coordinator.wake)
        thread.start()
        thread.join()
        for _ in range(10):
            await asyncio.sleep(0)
            if coordinator._wake.is_set():
                break
        assert coordinator._wake.is_set() is True
        await coordinator.stop()
        database.close()

    asyncio.run(scenario())


def test_summary_coordinator_stop_cancels_active_model_without_consuming_sources() -> None:
    async def scenario() -> None:
        database, memory = _memory()
        entered = asyncio.Event()

        class BlockingRouter(_SummaryRouter):
            async def complete(self, request, *, task: str, context):
                self.requests.append((request, task, context))
                entered.set()
                await asyncio.Event().wait()

        router = BlockingRouter()
        coordinator = MemorySummaryCoordinator(
            memory,
            lambda: router,
            poll_seconds=1,
            clock=lambda: 2_000_000.0,
        )
        before = memory.prepare_summarization_batch(force=True)
        assert before is not None

        await coordinator.start()
        await asyncio.wait_for(entered.wait(), timeout=1)
        await coordinator.stop()

        after = memory.prepare_summarization_batch(force=True)
        assert after is not None
        assert after.source_ids == before.source_ids
        assert memory.list(memory_type="summary") == ()
        assert coordinator.running is False
        assert coordinator.status().busy is False
        assert coordinator.status().last_status == "cancelled"
        database.close()

    asyncio.run(scenario())


def test_summary_coordinator_storage_failure_keeps_exact_source_batch(monkeypatch) -> None:
    database, memory = _memory()
    router = _SummaryRouter()
    coordinator = MemorySummaryCoordinator(memory, lambda: router, clock=lambda: 2_000_000.0)
    before = memory.prepare_summarization_batch(force=True)
    assert before is not None

    def fail_store(*_args, **_kwargs):
        raise RuntimeError("storage internals must stay private")

    monkeypatch.setattr(memory, "store_summary", fail_store)
    result = asyncio.run(coordinator.run_once())

    after = memory.prepare_summarization_batch(force=True)
    assert result["status"] == "failed"
    assert result["error"] == "RuntimeError"
    assert after is not None
    assert after.source_ids == before.source_ids
    assert memory.list(memory_type="summary") == ()
    database.close()


def test_application_runtime_owns_summary_lifecycle_and_live_router_provider(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"storage:\n  database: {tmp_path / 'runtime.sqlite3'}\n",
        encoding="utf-8",
    )
    runtime = build_runtime(
        LoadedConfiguration(config_path, {"storage": {"database": "runtime.sqlite3"}}),
        inspect_resources(tmp_path / "resources"),
    )
    coordinator = runtime.memory_summarizer
    assert coordinator is not None
    original_router = runtime.router
    replacement_router = object()
    runtime.router = replacement_router  # type: ignore[assignment]
    assert coordinator._router_provider() is replacement_router
    runtime.router = original_router

    async def scenario() -> None:
        await runtime.start_background()
        await runtime.start_background()
        assert coordinator.running is True
        await runtime.close()
        await runtime.close()
        assert coordinator.running is False
        assert coordinator.wake() is False

    asyncio.run(scenario())


def test_background_batch_preparation_failure_keeps_task_alive_and_logs_only_type(
    monkeypatch,
    caplog,
) -> None:
    async def scenario() -> None:
        database, memory = _memory()
        before = memory.prepare_summarization_batch(force=True)
        assert before is not None
        entered = asyncio.Event()
        original_preparation = memory.prepare_summarization_batch

        def fail_preparation(*_args, **_kwargs):
            entered.set()
            raise RuntimeError("database path and chat secret must stay private")

        monkeypatch.setattr(memory, "prepare_summarization_batch", fail_preparation)
        coordinator = MemorySummaryCoordinator(
            memory,
            lambda: _SummaryRouter(),
            poll_seconds=1,
            clock=lambda: 2_000_000.0,
        )
        await coordinator.start()
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.sleep(0)

        assert coordinator.running is True
        assert coordinator.status().failures == 1
        assert coordinator.status().last_reason == "prepare"
        await coordinator.stop()
        monkeypatch.setattr(memory, "prepare_summarization_batch", original_preparation)
        after = memory.prepare_summarization_batch(force=True)
        assert after is not None
        assert after.source_ids == before.source_ids
        database.close()

    with caplog.at_level(logging.WARNING, logger="services.memory.summarizer"):
        asyncio.run(scenario())

    assert "RuntimeError" in caplog.text
    assert "database path and chat secret must stay private" not in caplog.text


def test_clock_failure_returns_safe_status_without_touching_source_batch(caplog) -> None:
    database, memory = _memory()
    before = memory.prepare_summarization_batch(force=True)
    assert before is not None

    def fail_clock() -> float:
        raise RuntimeError("clock configuration secret must stay private")

    coordinator = MemorySummaryCoordinator(memory, lambda: _SummaryRouter(), clock=fail_clock)
    with caplog.at_level(logging.WARNING, logger="services.memory.summarizer"):
        result = asyncio.run(coordinator.run_once())

    after = memory.prepare_summarization_batch(force=True)
    assert result == {
        "status": "failed",
        "reason": "coordinator",
        "error": "RuntimeError",
    }
    assert after is not None
    assert after.source_ids == before.source_ids
    assert coordinator.status().failures == 1
    assert "RuntimeError" in caplog.text
    assert "clock configuration secret must stay private" not in caplog.text
    database.close()


def test_model_extraction_microbatch_uses_strict_json_without_tools() -> None:
    database = Database(":memory:")
    memory = MemoryService(
        database,
        settings=MemorySettings(extract_max_items=3, extract_max_chars=240),
    )
    router = _SummaryRouter(
        text=(
            '[{"content":"用户职业：工程师","priority":8,'
            '"confidence":0.95,"tags":["profile"],"rule":"model"},'
            '{"content":"api_key=secret-value","priority":10,"confidence":1.0}]'
        )
    )
    coordinator = MemorySummaryCoordinator(memory, lambda: router, clock=lambda: 2_000_000.0)
    assert coordinator.enqueue_extraction("我是工程师", 11)
    assert coordinator.enqueue_extraction("我偏好简洁界面", 12)
    assert coordinator.enqueue_extraction("请记住这个习惯", 13)

    result = asyncio.run(coordinator.run_extraction_once())

    assert result == {
        "status": "completed",
        "source_count": 3,
        "promoted": 1,
        "pending": 0,
    }
    assert len(router.requests) == 1
    request, task, context = router.requests[0]
    assert task == "memory"
    assert request.tools == ()
    assert request.tool_choice == "none"
    assert request.metadata == {"purpose": "memory_extract", "batch_size": 3}
    assert context.session_id == "memory-extract"
    facts = memory.list(memory_type="fact")
    assert [item.content for item in facts] == ["用户职业：工程师"]
    assert facts[0].source_ids == (11, 12, 13)
    assert coordinator.status().extraction_completed == 1
    database.close()


def test_model_extraction_invalid_json_retries_once_without_logging_body(caplog) -> None:
    database = Database(":memory:")
    memory = MemoryService(database)
    router = _SummaryRouter(text="不是 JSON，包含用户正文秘密")
    coordinator = MemorySummaryCoordinator(memory, lambda: router, clock=lambda: 2_000_000.0)
    coordinator.enqueue_extraction("用户正文秘密", 21)

    with caplog.at_level(logging.WARNING, logger="services.memory.summarizer"):
        first = asyncio.run(coordinator.run_extraction_once())
        second = asyncio.run(coordinator.run_extraction_once(force=True))

    assert first["status"] == "failed"
    assert first["pending"] == 1
    assert second["status"] == "failed"
    assert second["pending"] == 0
    assert len(router.requests) == 2
    assert coordinator.status().extraction_failures == 2
    assert coordinator.status().extraction_dropped == 1
    assert "JSONDecodeError" in caplog.text
    assert "用户正文秘密" not in caplog.text
    assert "不是 JSON" not in caplog.text
    assert memory.list(memory_type="fact") == ()
    database.close()


def test_model_extraction_queue_is_bounded_and_disable_clears_pending() -> None:
    database = Database(":memory:")
    memory = MemoryService(database, settings=MemorySettings(extract_max_items=1))
    coordinator = MemorySummaryCoordinator(memory, lambda: _SummaryRouter())

    for index in range(20):
        assert coordinator.enqueue_extraction(f"完整回合 {index}", index + 1)

    assert coordinator.status().extraction_pending == 16
    assert coordinator.status().extraction_dropped == 4
    memory.configure(MemorySettings(auto_extract_enabled=False))
    result = asyncio.run(coordinator.run_extraction_once())
    assert result == {"status": "idle"}
    assert coordinator.status().extraction_pending == 0
    assert coordinator.status().extraction_dropped == 20
    database.close()


def test_model_extraction_cancellation_restores_pending_batch() -> None:
    async def scenario() -> None:
        database = Database(":memory:")
        memory = MemoryService(database)
        entered = asyncio.Event()

        class BlockingRouter(_SummaryRouter):
            async def complete(self, request, *, task: str, context):
                self.requests.append((request, task, context))
                entered.set()
                await asyncio.Event().wait()

        coordinator = MemorySummaryCoordinator(memory, lambda: BlockingRouter())
        coordinator.enqueue_extraction("需要后台提取的完整回合", 31)
        task = asyncio.create_task(coordinator.run_extraction_once())
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert coordinator.status().extraction_pending == 1
        assert coordinator.status().extraction_last_status == "cancelled"
        await coordinator.stop()
        assert coordinator.status().extraction_pending == 0
        database.close()

    asyncio.run(scenario())


def test_background_coordinator_automatically_consumes_extraction_queue() -> None:
    async def scenario() -> None:
        database = Database(":memory:")
        memory = MemoryService(database)
        router = _SummaryRouter(
            text='[{"content":"用户使用 Linux","priority":7,"confidence":0.95}]'
        )
        coordinator = MemorySummaryCoordinator(
            memory,
            lambda: router,
            poll_seconds=1,
            clock=lambda: 2_000_000.0,
        )
        await coordinator.start()
        assert coordinator.enqueue_extraction("我使用 Linux", 41)
        for _ in range(100):
            if coordinator.status().extraction_completed == 1:
                break
            await asyncio.sleep(0)

        assert coordinator.status().extraction_completed == 1
        assert memory.search("Linux", limit=1)[0].content == "用户使用 Linux"
        assert len(router.requests) == 1
        await coordinator.stop()
        database.close()

    asyncio.run(scenario())
