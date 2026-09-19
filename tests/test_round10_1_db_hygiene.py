"""第 10 轮方向 1：SQLite 性能与卫生收尾回归
（A 修剪节流 / B WAL 告警 / C 版本门控 / D 部分列更新 / E 汇总缓存）。

全部用例只依赖 src/db 与内存/临时 SQLite，属于快层；不使用 sleep，
时钟一律注入或 monkeypatch。
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable

import pytest

from db import ApiCallAuditRepository, Database, SchedulerRepository
from db.api_call_audit_repository import (
    SUMMARY_CACHE_TTL_SECONDS,
)
from db.database import SchemaMigrator
from db.pet_diary_repository import (
    AUDIT_PRUNE_INTERVAL as DIARY_AUDIT_PRUNE_INTERVAL,
)
from db.pet_diary_repository import (
    PetDiaryRepository,
)


def _record_runs(repository: SchedulerRepository, count: int) -> None:
    for index in range(count):
        repository.record_run(
            kind="task",
            item_id="task",
            owner="owner",
            started_at=float(index),
            finished_at=float(index),
            status="completed",
        )


class TestDiaryAuditTrimThrottle:
    """A 项：日记审计修剪按写计数抽样，超限时立即收敛、主写不受修剪失败影响。"""

    def test_no_trim_when_audit_below_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        database = Database(":memory:")
        repository = PetDiaryRepository(database, clock=lambda: 100.0)
        access = repository._issue_internal_access("agent:test")
        trims = 0
        original = repository._trim_audit

        def counting_trim(connection) -> None:
            nonlocal trims
            trims += 1
            original(connection)

        monkeypatch.setattr(repository, "_trim_audit", counting_trim)
        writes = DIARY_AUDIT_PRUNE_INTERVAL * 2 + 5
        for index in range(writes):
            repository._create(
                access,
                f"第 {index} 条",
                visibility="public",
                priority=5,
                tags=(),
                retention_until=None,
            )
        # 未达上限：只有写计数窗口的两次必然修剪，无超限触发的额外修剪，
        # 70 次写收敛为 2 次修剪而非 70 次。
        assert trims == writes // DIARY_AUDIT_PRUNE_INTERVAL

        # 计数越过窗口后的下一次写入也不修剪；超限探测才触发。
        repository._create(
            access,
            "结束条目",
            visibility="public",
            priority=5,
            tags=(),
            retention_until=None,
        )
        assert trims == writes // DIARY_AUDIT_PRUNE_INTERVAL

    def test_trim_fires_once_above_cap_and_converges(self) -> None:
        database = Database(":memory:")
        repository = PetDiaryRepository(database, clock=lambda: 200.0, max_audit_rows=3)
        access = repository._issue_internal_access("agent:test")
        for index in range(DIARY_AUDIT_PRUNE_INTERVAL + 2):
            repository._create(
                access,
                f"小上限 {index}",
                visibility="public",
                priority=5,
                tags=(),
                retention_until=None,
            )
        count = database.connection.execute("SELECT COUNT(*) FROM pet_diary_audit").fetchone()[0]
        assert count == 3

    def test_trim_failure_does_not_break_main_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        database = Database(":memory:")
        repository = PetDiaryRepository(database, clock=lambda: 300.0, max_audit_rows=1)
        access = repository._issue_internal_access("agent:test")

        def explode(connection) -> None:
            raise sqlite3.OperationalError("injected trim failure")

        monkeypatch.setattr(repository, "_trim_audit", explode)
        for index in range(DIARY_AUDIT_PRUNE_INTERVAL + 1):
            entry = repository._create(
                access,
                f"失败隔离 {index}",
                visibility="public",
                priority=5,
                tags=(),
                retention_until=None,
            )
            assert entry is not None and entry.id == index + 1


class TestSchedulerRunTrimThrottle:
    """A 项：调度执行记录修剪按写计数抽样，超限时立即收敛、主写不受修剪失败影响。"""

    def test_trim_fires_once_above_cap_and_converges(self) -> None:
        database = Database(":memory:")
        repository = SchedulerRepository(database, max_run_rows=2)
        _record_runs(repository, 6)
        count = database.connection.execute("SELECT COUNT(*) FROM scheduler_runs").fetchone()[0]
        assert count == 2

    def test_trim_failure_does_not_break_main_write(self, monkeypatch: pytest.MonkeyPatch) -> None:
        database = Database(":memory:")
        repository = SchedulerRepository(database, max_run_rows=1)

        def explode(connection) -> None:
            raise sqlite3.OperationalError("injected trim failure")

        monkeypatch.setattr(repository, "_trim_runs", explode)
        _record_runs(repository, 4)
        count = database.connection.execute("SELECT COUNT(*) FROM scheduler_runs").fetchone()[0]
        assert count == 4


class TestWalEnableWarning:
    """B 项：PRAGMA journal_mode 失败时记录 warning 并保留降级语义。"""

    def test_wal_failure_logs_warning_and_still_opens(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[logging.LogRecord] = []

        class ProbeHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record)

        handler = ProbeHandler()
        logger = logging.getLogger("db.database")
        logger.addHandler(handler)
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        original_connect = sqlite3.connect

        class FailingWalConnection:
            """代理 sqlite3.Connection：第一次 journal_mode PRAGMA 抛错误。"""

            def __init__(self, real):
                self._real = real
                self._failed = False

            def __getattr__(self, name):
                return getattr(self._real, name)

            def execute(self, statement, parameters=()):
                if not self._failed and "journal_mode" in str(statement):
                    self._failed = True
                    raise sqlite3.DatabaseError("injected PRAGMA failure")
                return self._real.execute(statement, parameters)

        def failing_connect(*args, **kwargs):
            return FailingWalConnection(original_connect(*args, **kwargs))

        monkeypatch.setattr(sqlite3, "connect", failing_connect)
        try:
            database = Database(tmp_path / "wal_warning.sqlite3")
            warnings = [
                str(record.getMessage()) for record in captured if record.levelno == logging.WARNING
            ]
            assert any("WAL" in message for message in warnings)
            assert database.connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
            database.close()
        finally:
            logger.removeHandler(handler)
            logger.setLevel(original_level)


class TestRepairEmbeddingsGatedByVersion:
    """C 项：嵌入修复只在 schema 版本实际变化时执行一次。"""

    def test_same_version_second_open_skips_embedding_scan(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "gated.sqlite3"
        initial = Database(path)
        initial.connection.execute(
            "INSERT INTO memories (content, embedding, created, updated) VALUES (?, ?, 1, 1)",
            ("首条记忆", '["bad"]'),
        )
        initial.connection.commit()
        assert (
            initial.connection.execute("SELECT MAX(version) FROM memory_schema_version").fetchone()[
                0
            ]
            == 7
        )
        initial.close()

        calls = 0
        original = SchemaMigrator._repair_memory_embeddings

        def counting(cls, connection, current_version) -> None:
            nonlocal calls
            calls += 1
            original(connection, current_version)

        monkeypatch.setattr(SchemaMigrator, "_repair_memory_embeddings", classmethod(counting))
        reopened = Database(path)
        reopened.close()
        assert calls == 0


class TestAuditRepositoryUpdateColumns:
    """D 项：update 只写变更列与 updated_at，返回形状与 _encode_record 语义不变。"""

    def _make_repository(self) -> tuple:
        class CountingConnection:
            """委托真实 sqlite3.Connection，仅记录 UPDATE 语句文本。"""

            def __init__(self, real):
                self._real = real
                self.update_statements: list[str] = []

            def __getattr__(self, name):
                return getattr(self._real, name)

            def execute(self, statement, parameters=()):
                if str(statement).strip().upper().startswith("UPDATE"):
                    self.update_statements.append(str(statement))
                return self._real.execute(statement, parameters)

        database = Database(":memory:")
        wrapper = CountingConnection(database.connection)
        database.connection = wrapper
        repository = ApiCallAuditRepository(database, clock=lambda: 500.0)
        record = repository.start(
            request_id="req-update",
            kind="model",
            channel_id="primary",
            input_payload={"nested": {"body": "large"}},
            started_at=10.0,
        )
        record = repository.get(record.record_id)
        assert record is not None
        return repository, record, wrapper

    def test_status_only_update_sql_excludes_payload_columns(self) -> None:
        repository, record, wrapper = self._make_repository()
        updated = repository.update(record.id, status="completed")
        assert updated is not None
        assert updated.status == "completed"
        sqls = [sql for sql in wrapper.update_statements if "api_call_audits" in sql]
        assert sqls
        assert "payload" not in " ".join(sqls)
        assert "updated_at" in sqls[0]

    def test_update_without_writable_columns_returns_readback_without_sql(self) -> None:
        repository, record, wrapper = self._make_repository()
        before = len(wrapper.update_statements)
        returned = repository.update(record.id)
        assert returned == record
        assert len(wrapper.update_statements) == before
        assert repository.get(record.id) == record


class TestSummaryCache:
    """E 项：summary 短 TTL 缓存，写入失效、过期重算、返回形状不变。"""

    def _make_repository(self) -> tuple[ApiCallAuditRepository, Callable[[], list[str]]]:
        database = Database(":memory:")
        repository = ApiCallAuditRepository(database, clock=lambda: 900.0)
        handle = repository.start(
            request_id="req-1",
            kind="model",
            channel_id="primary",
            requested_model="chat",
            started_at=10.0,
        )
        handle.finish(completed_at=11.0, response_model="chat-v2")
        repository.start(
            request_id="req-2",
            kind="model",
            channel_id="primary",
            requested_model="vision",
            started_at=20.0,
        ).fail(error_type="outcome", error_message="中断", status="failed")
        repository.start(
            request_id="req-3",
            kind="model",
            requested_model="chat",
            started_at=30.0,
        ).fail(error_type="network", error_message="超时")
        calls: list[str] = []
        original = repository.aggregate

        def counting(**filters):
            calls.append(",".join(sorted(filters)))
            return original(**filters)

        repository.aggregate = counting
        return repository, lambda: calls

    def test_same_params_same_window_hits_cache(self) -> None:
        repository, get_calls = self._make_repository()
        first = repository.summary()
        second = repository.summary()
        assert first == second
        assert get_calls() == [""]

    def test_different_params_use_separate_cache_entries(self) -> None:
        repository, get_calls = self._make_repository()
        repository.summary()
        repository.summary()
        repository.summary(status="failed")
        repository.summary(status="failed")
        assert get_calls() == ["", "status"]

    def test_write_invalidates_cache(self) -> None:
        repository, get_calls = self._make_repository()
        before_total = repository.summary()["total"]
        assert before_total == 3
        repository.start(
            request_id="req-new",
            kind="tool",
            started_at=40.0,
        )
        repository._clock = lambda: 900.0
        after = repository.summary()
        assert after["total"] == before_total + 1
        assert len(get_calls()) == 2

    def test_ttl_expiry_recomputes(self) -> None:
        repository, get_calls = self._make_repository()
        repository.summary()
        repository._clock = lambda: 900.0 + SUMMARY_CACHE_TTL_SECONDS
        repository.summary()
        assert get_calls() == ["", ""]

    def test_return_shape_stable(self) -> None:
        repository, _ = self._make_repository()
        summary = repository.summary(status="failed")
        assert set(summary) == {"total", "by_status", "by_channel", "latency_ms"}
        assert summary["total"] == 2
        assert summary["latency_ms"] == {
            "avg_first": None,
            "avg_total": None,
            "max_total": None,
        }
        assert any(row["key"] == "status:failed" for row in summary["by_status"])
