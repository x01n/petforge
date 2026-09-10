import sqlite3
from datetime import datetime, timedelta
from typing import cast

import pytest

from core.memory_embedding import SparseVectorIndex, compute_embedding
from db import Database, SchemaMigrator
from services.memory import MemoryService, SummarizationBatch


def _store_exchange(
    service: MemoryService,
    database: Database,
    *,
    timestamp: float,
    index: int,
) -> None:
    user_id = service.add_chat("user", f"用户消息 {index}")
    assistant_id = service.add_chat("assistant", f"桌宠回复 {index}")
    with database.transaction() as connection:
        connection.executemany(
            "UPDATE chat_history SET timestamp = ? WHERE id = ?",
            ((timestamp, user_id), (timestamp + 1, assistant_id)),
        )
    service.increment_message_counter()


def test_sparse_vector_index_replaces_updates_and_removes_documents() -> None:
    index = SparseVectorIndex()
    index.replace(
        (
            (1, compute_embedding("红茶")),
            (2, compute_embedding("咖啡")),
        )
    )

    assert index.size == 2
    assert index.search(compute_embedding("红茶"), limit=1)[0][0] == 1

    index.upsert(1, compute_embedding("抹茶"))
    assert index.search(compute_embedding("抹茶"), limit=1)[0][0] == 1
    index.remove(1)
    assert all(memory_id != 1 for memory_id, _ in index.search(compute_embedding("抹茶"), limit=5))


def test_memory_search_uses_bounded_vector_index_when_fts5_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    service = MemoryService(
        Database(tmp_path / "rag.sqlite3"),
        settings={"max_memories": 500},
    )
    for index in range(250):
        service.create(f"普通记录 {index}", priority=1)
    target = service.create("用户喜欢海盐焦糖", priority=7)
    service._fts_available = False

    search_calls: list[int] = []
    list_limits: list[int] = []
    original_search = service._vector_index.search
    original_list = service.list

    def traced_search(embedding, *, limit):
        search_calls.append(limit)
        return original_search(embedding, limit=limit)

    def traced_list(*, limit=100, tag=None, memory_type=None, tags=None):
        list_limits.append(limit)
        return original_list(limit=limit, tag=tag, memory_type=memory_type, tags=tags)

    monkeypatch.setattr(service._vector_index, "search", traced_search)
    monkeypatch.setattr(service, "list", traced_list)

    result = service.search("海盐焦糖", limit=1)

    assert result[0].id == target.id
    assert search_calls == [64]
    assert max(list_limits) < 250
    assert service.status()["lexical_index"] == "sparse_fallback"


def test_memory_search_uses_fts5_candidates_when_available(tmp_path, monkeypatch) -> None:
    service = MemoryService(Database(tmp_path / "fts-search.sqlite3"))
    if service.status()["lexical_index"] != "fts5":
        pytest.skip("SQLite runtime does not provide FTS5")
    target = service.create("用户爱好星空摄影", priority=5)
    monkeypatch.setattr(service._vector_index, "search", lambda _embedding, *, limit: ())
    monkeypatch.setattr(service, "list", lambda **_kwargs: ())

    result = service.search("星空摄影", limit=1)

    assert result[0].id == target.id
    service.update(target.id, content="用户爱好水下摄影")
    assert service.search("水下摄影", limit=1)[0].id == target.id
    service.delete(target.id)
    assert service.search("水下摄影", limit=1) == ()


def test_compatibility_search_serializes_loaded_memories_without_n_plus_one_reads(
    tmp_path, monkeypatch
) -> None:
    service = MemoryService(Database(tmp_path / "compat-search.sqlite3"))
    service.create("主人喜欢海盐焦糖", priority=8)
    service.create("主人喜欢乌龙茶", priority=7)

    def unexpected_lookup(_memory_id: int):
        raise AssertionError("search_memories must not re-read each selected memory")

    monkeypatch.setattr(service, "get_memory", unexpected_lookup)

    result = service.search_memories("主人喜欢", limit=2)

    assert {item["content"] for item in result} == {"主人喜欢海盐焦糖", "主人喜欢乌龙茶"}
    assert all("last_decay" in item for item in result)


def test_get_memory_reads_the_row_once(tmp_path) -> None:
    database = Database(tmp_path / "compat-get.sqlite3")
    service = MemoryService(database)
    memory = service.create("只读取一次")
    statements: list[str] = []
    database.connection.set_trace_callback(statements.append)
    try:
        value = service.get_memory(memory.id)
    finally:
        database.connection.set_trace_callback(None)

    assert value is not None
    assert sum("SELECT * FROM memories WHERE id =" in statement for statement in statements) == 1


def test_schema_migrator_treats_missing_fts5_as_optional() -> None:
    class _EmptyResult:
        @staticmethod
        def fetchone():
            return None

    class _NoFtsConnection:
        @staticmethod
        def execute(statement, _params=()):
            if "CREATE VIRTUAL TABLE" in statement:
                raise sqlite3.OperationalError("no such module: fts5")
            return _EmptyResult()

    connection = cast(sqlite3.Connection, _NoFtsConnection())
    assert SchemaMigrator()._ensure_memory_fts(connection) is False


def test_summary_message_threshold_boundary_and_batch_storage(tmp_path) -> None:
    database = Database(tmp_path / "message-summary.sqlite3")
    service = MemoryService(
        database,
        settings={
            "summarize_every_n": 2,
            "summary_min_messages": 4,
            "summarize_daily": False,
            "summary_interval_minutes": 0,
        },
    )
    _store_exchange(service, database, timestamp=1_000, index=1)
    assert service.summarization_reason(now=1_001) is None

    _store_exchange(service, database, timestamp=1_010, index=2)
    assert service.summarization_reason(now=1_011) == "message_count"
    batch = service.prepare_summarization_batch(now=1_011)
    assert isinstance(batch, SummarizationBatch)
    assert batch.reason == "message_count"
    assert len(batch.messages) == 4

    memory_id = service.store_summary(
        "用户完成了两轮对话。",
        batch.source_ids,
        trigger_reason=batch.reason,
        summarized_at=batch.created_at,
    )
    summary = service.get(memory_id)
    assert summary is not None
    assert summary.metadata["summary_reason"] == "message_count"
    assert service.summarization_reason(now=1_012) is None
    summarized = database.connection.execute(
        "SELECT COUNT(*) FROM chat_history WHERE summarized = 1"
    ).fetchone()[0]
    assert summarized == 4


def test_summary_daily_boundary_is_deterministic(tmp_path) -> None:
    database = Database(tmp_path / "daily-summary.sqlite3")
    service = MemoryService(
        database,
        settings={
            "summarize_every_n": 100,
            "summary_min_messages": 4,
            "summarize_daily": True,
            "summary_interval_minutes": 0,
        },
    )
    first_day = datetime(2026, 8, 29, 12, 0)
    for index in range(2):
        _store_exchange(
            service,
            database,
            timestamp=(first_day + timedelta(minutes=index)).timestamp(),
            index=index,
        )

    assert service.summarization_reason(now=datetime(2026, 8, 29, 23, 59).timestamp()) is None
    assert service.summarization_reason(now=datetime(2026, 8, 30, 0, 0).timestamp()) == "daily"


def test_summary_interval_boundary_and_trigger_precedence(tmp_path) -> None:
    database = Database(tmp_path / "interval-summary.sqlite3")
    service = MemoryService(
        database,
        settings={
            "summarize_every_n": 100,
            "summary_min_messages": 4,
            "summarize_daily": False,
            "summary_interval_minutes": 60,
        },
    )
    for index in range(2):
        _store_exchange(service, database, timestamp=1_000 + index, index=index)

    assert service.summarization_reason(now=4_599) is None
    assert service.summarization_reason(now=4_600) == "interval"

    service.configure(
        {
            "summarize_every_n": 2,
            "summary_min_messages": 4,
            "summarize_daily": True,
            "summary_interval_minutes": 1,
        }
    )
    assert service.summarization_reason(now=90_000) == "message_count"


def test_summary_batch_uses_configured_minimum_message_threshold(tmp_path) -> None:
    database = Database(tmp_path / "summary-minimum.sqlite3")
    service = MemoryService(
        database,
        settings={
            "summarize_every_n": 1,
            "summary_min_messages": 2,
            "summarize_daily": False,
            "summary_interval_minutes": 0,
        },
    )
    _store_exchange(service, database, timestamp=1_000, index=1)

    assert service.summarization_reason(now=1_001) == "message_count"
    batch = service.prepare_summarization_batch(now=1_001)
    assert isinstance(batch, SummarizationBatch)
    assert len(batch.messages) == 2
    assert len(batch.source_ids) == 2


def test_dynamic_identity_supersession_and_reinforcement_take_effect_immediately(tmp_path) -> None:
    service = MemoryService(Database(tmp_path / "dynamic-memory.sqlite3"))
    first = service.extract_and_promote("我叫小林")[0]
    reinforced = service.extract_and_promote("我叫小林")[0]
    second = service.extract_and_promote("我叫小周")[0]

    assert reinforced.priority > first.priority
    previous = service.get(first.id)
    assert previous is not None
    assert previous.metadata["superseded_by"] == second.id
    recalled = service.search("用户称呼", limit=5)
    assert [item.id for item in recalled] == [second.id]


def test_model_extraction_batch_is_bounded_safe_and_immediately_searchable(tmp_path) -> None:
    service = MemoryService(
        Database(tmp_path / "model-extract.sqlite3"),
        settings={"extract_max_items": 2},
    )
    promoted = service.promote_extracted_batch(
        (
            {"content": "用户偏好低糖饮品", "priority": 8, "confidence": 0.95},
            {"content": "用户习惯早起", "priority": 6, "confidence": 0.90},
            {"content": "用户喜欢第三条", "priority": 5, "confidence": 1.0},
            {"content": "api_key=secret-value", "priority": 10, "confidence": 1.0},
        )
    )

    assert len(promoted) == 2
    assert service.status()["vector_index_size"] == 2
    assert service.search("低糖饮品", limit=1)[0].id == promoted[0].id
