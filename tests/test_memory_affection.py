import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pytest

from db import ConversationRepository, ConversationTurn, Database, SchemaMigrator
from services.affection import AffectionService
from services.memory import MemoryService, MemorySettings


def test_schema_migration_is_idempotent(tmp_path):
    database = Database(tmp_path / "memory.sqlite3")
    SchemaMigrator().migrate(database.connection)
    SchemaMigrator().migrate(database.connection)
    version = database.connection.execute(
        "SELECT version FROM memory_schema_version ORDER BY version DESC LIMIT 1"
    ).fetchone()[0]
    assert version == 7
    columns = {
        row[1] for row in database.connection.execute("PRAGMA table_info(conversation_turns)")
    }
    assert {"mode", "profile_id", "session_id", "turn_id", "segments", "system_entries"} <= columns
    ann_tables = {
        row[0]
        for row in database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"memory_ann_journal", "memory_ann_generations", "memory_semantic_vectors"} <= ann_tables


def test_legacy_v4_schema_repairs_conversation_columns_and_embeddings(tmp_path):
    path = tmp_path / "legacy-v4.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE memory_schema_version (
            version INTEGER PRIMARY KEY,
            applied_at REAL NOT NULL
        );
        INSERT INTO memory_schema_version (version, applied_at) VALUES (4, 1);
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            importance INTEGER DEFAULT 1,
            source TEXT DEFAULT '',
            created REAL NOT NULL,
            last_recalled REAL,
            tags TEXT DEFAULT '[]',
            metadata TEXT DEFAULT '{}',
            memory_type TEXT DEFAULT 'fact',
            embedding TEXT DEFAULT '',
            decay_factor REAL DEFAULT 1.0,
            updated REAL DEFAULT 0,
            access_count INTEGER DEFAULT 1,
            source_ids TEXT DEFAULT '[]',
            last_decay REAL DEFAULT 0
        );
        INSERT INTO memories (content, created, embedding)
        VALUES ('旧版记忆', 10, 'old-format');
        CREATE TABLE conversation_turns (
            mode TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            source TEXT NOT NULL,
            user_text TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (mode, profile_id, session_id, turn_id)
        );
        """
    )
    connection.commit()
    connection.close()

    database = Database(path)
    columns = {
        row[1] for row in database.connection.execute("PRAGMA table_info(conversation_turns)")
    }
    assert {
        "segments",
        "system_entries",
        "created_at",
        "updated_at",
        "status",
        "error_text",
    } <= columns
    memory_row = database.connection.execute(
        "SELECT embedding, last_decay FROM memories WHERE id = 1"
    ).fetchone()
    assert json.loads(memory_row[0])
    assert memory_row[1] == 10
    repository = ConversationRepository(database)
    assert (
        repository.create(
            mode="direct",
            profile_id="profile",
            session_id="session",
            turn_id="turn",
            source="migration-test",
        ).status
        == "streaming"
    )


def test_schema_v6_upgrade_adds_semantic_tables_and_triggers(tmp_path):
    database = Database(tmp_path / "v6-upgrade.sqlite3")
    with database.transaction() as connection:
        connection.execute("DROP TRIGGER IF EXISTS memories_ann_insert")
        connection.execute("DROP TRIGGER IF EXISTS memories_ann_update")
        connection.execute("DROP TRIGGER IF EXISTS memories_ann_delete")
        connection.execute("DROP TABLE memory_semantic_vectors")
        connection.execute("DROP TABLE memory_ann_generations")
        connection.execute("DROP TABLE memory_ann_journal")
        connection.execute("DELETE FROM memory_schema_version")
        connection.execute("INSERT INTO memory_schema_version(version, applied_at) VALUES (6, 1)")
    SchemaMigrator().migrate(database.connection)
    tables = {
        row[0]
        for row in database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"memory_ann_journal", "memory_ann_generations", "memory_semantic_vectors"} <= tables
    triggers = {
        row[0]
        for row in database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
    }
    assert {"memories_ann_insert", "memories_ann_update", "memories_ann_delete"} <= triggers
    assert (
        database.connection.execute("SELECT MAX(version) FROM memory_schema_version").fetchone()[0]
        == 7
    )


def test_legacy_conversation_table_without_primary_key_is_rebuilt(tmp_path):
    path = tmp_path / "legacy-conversation.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE conversation_turns (
            mode TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            source TEXT NOT NULL,
            user_text TEXT NOT NULL DEFAULT '',
            segments TEXT NOT NULL DEFAULT '[]',
            system_entries TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'streaming',
            error_text TEXT NOT NULL DEFAULT ''
        )
        """
    )
    connection.execute(
        "INSERT INTO conversation_turns (mode, profile_id, session_id, turn_id, source) "
        "VALUES ('direct', 'profile', 'session', 'old', 'legacy')"
    )
    connection.commit()
    connection.close()

    database = Database(path)
    repository = ConversationRepository(database)
    assert repository.get("direct", "profile", "session", "old") is not None
    repository.save(
        ConversationTurn(
            "direct",
            "profile",
            "session",
            "old",
            "updated",
            status="complete",
        )
    )
    assert repository.get("direct", "profile", "session", "old").status == "complete"


def test_affection_range_and_daily_cap(tmp_path):
    database = Database(tmp_path / "affection.sqlite3")
    service = AffectionService(database, today=lambda: date(2026, 8, 22))
    assert service.adjust(100).applied == 15
    assert service.get() == 20
    assert service.adjust(1).applied == 0
    assert service.adjust(-100).current == 0
    assert service.adjust(100).current == 0


def test_conversations_are_isolated_and_bad_json_is_skipped(tmp_path):
    database = Database(tmp_path / "conversation.sqlite3")
    repository = ConversationRepository(database)
    repository.save(
        ConversationTurn(
            "direct",
            "alpha",
            "one",
            "turn-1",
            "user_reply",
            "hello",
            ({"text": "world"},),
            (),
            1,
            1,
            "complete",
        )
    )
    repository.save(
        ConversationTurn(
            "direct",
            "alpha",
            "two",
            "turn-2",
            "user_reply",
            "hello",
            ({"text": "other"},),
            (),
            2,
            2,
            "complete",
        )
    )
    database.connection.execute(
        "UPDATE conversation_turns SET segments = ? WHERE session_id = ?",
        ("{bad", "one"),
    )
    database.connection.commit()
    assert repository.list_recent("direct", "alpha", "one") == ()
    assert [item.turn_id for item in repository.list_recent("direct", "alpha", "two")] == ["turn-2"]


def test_memory_crud_events_search_and_safe_import_export(tmp_path):
    database = Database(tmp_path / "memory.sqlite3")
    service = MemoryService(database)
    created = service.create(
        "用户喜欢红茶",
        importance=7,
        tags=("饮品", "偏好"),
        metadata={"source": "chat"},
    )
    assert service.get(created.id) == created
    assert service.search("红茶")[0].id == created.id
    service.update(created.id, content="用户喜欢咖啡", tags=("饮品",))
    service.record_event("memory", "updated", {"memory_id": created.id})
    export_path = tmp_path / "memory.json"
    exported = service.export_json(export_path)

    restored = MemoryService(Database(tmp_path / "restored.sqlite3"))
    assert restored.import_json(exported) == 1
    assert restored.search("咖啡")[0].tags == ("饮品",)
    assert service.import_json({"memories": [{"content": ""}, "bad"]}) == 0
    assert service.delete(created.id)


def test_memory_filters_tags_exactly_and_validates_source_labels(tmp_path):
    service = MemoryService(Database(tmp_path / "memory-boundaries.sqlite3"))
    foo = service.create("foo 标签", tags=("foo",), source="来源", memory_type="fact")
    service.create("foobar 标签", tags=("foobar",), source="来源", memory_type="fact")

    assert [item.id for item in service.list(tag="foo")] == [foo.id]
    assert [item.id for item in service.list(tags=("foo",))] == [foo.id]

    for field, value in (
        ("source", "x" * 257),
        ("source", "bad\nsource"),
        ("source", "bad\x00source"),
        ("memory_type", "x" * 129),
        ("memory_type", "bad\rmemory_type"),
        ("memory_type", "bad\x00memory_type"),
    ):
        kwargs = {field: value}
        with pytest.raises(ValueError):
            service.create("边界校验", **kwargs)

    with pytest.raises(ValueError):
        service.update(foo.id, source="x" * 257)
    with pytest.raises(ValueError):
        service.update(foo.id, memory_type="bad\nmemory_type")


def test_empty_replace_import_clears_memories_but_merge_keeps_them(tmp_path):
    service = MemoryService(Database(tmp_path / "empty-import.sqlite3"))
    created = service.create("需要被空快照替换")

    assert service.import_json({"memories": []}) == 0
    assert service.get(created.id) is None
    assert service.status()["vector_index_size"] == 0

    restored = service.create("合并模式保留")
    assert service.import_json({"memories": []}, merge=True) == 0
    assert service.get(restored.id) is not None


def test_empty_replace_import_still_applies_state_and_events(tmp_path):
    service = MemoryService(Database(tmp_path / "empty-import-state.sqlite3"))
    service.create("旧记录")

    assert (
        service.import_json(
            {
                "memories": [],
                "state": {"nickname": "新昵称", "untrusted": "ignored"},
                "events": [
                    {
                        "event_type": "import",
                        "description": "空快照仍包含事件",
                        "data": {"source": "test"},
                    }
                ],
            }
        )
        == 0
    )
    assert service.list() == ()
    assert service._get_state("nickname") == "新昵称"
    assert service._get_state("untrusted") == ""
    assert service.events(limit=1)[0]["event_type"] == "import"


def test_memory_decay_factor_rejects_non_finite_values(tmp_path):
    service = MemoryService(Database(tmp_path / "finite-decay.sqlite3"))
    with pytest.raises(ValueError, match="finite"):
        service.create("有限值", decay_factor=float("nan"))
    with pytest.raises(ValueError, match="JSON serializable"):
        service.create("元数据", metadata={"value": float("nan")})
    memory = service.create("可更新")
    with pytest.raises(ValueError, match="finite"):
        service.update(memory.id, decay_factor=float("inf"))


def test_memory_source_ids_reject_lossy_or_unrepresentable_values(tmp_path):
    service = MemoryService(Database(tmp_path / "source-id-boundaries.sqlite3"))
    for source_ids in ((True,), (1.5,), (float("nan"),), ("1.5",), (2**63,)):
        with pytest.raises(ValueError, match="source"):
            service.create("来源编号边界", source_ids=source_ids)

    memory = service.create("允许整数来源编号", source_ids=(" 42 ", 7.0))
    assert memory.source_ids == (42, 7)


def test_memory_consolidation_preserves_provenance_and_uses_newer_record(tmp_path):
    database = Database(tmp_path / "consolidation.sqlite3")
    service = MemoryService(database)
    older = service.create(
        "主人喜欢喝咖啡",
        importance=9,
        tags=("older",),
        metadata={"older": True},
        source_ids=(1,),
    )
    newer = service.create(
        "主人喜欢喝咖啡",
        importance=2,
        tags=("newer",),
        metadata={"newer": True},
        source_ids=(2,),
    )
    with database.transaction() as connection:
        connection.execute(
            "UPDATE memories SET created = ?, last_recalled = ?, access_count = ? WHERE id = ?",
            (100, 10, 3, older.id),
        )
        connection.execute(
            "UPDATE memories SET created = ?, last_recalled = ?, access_count = ? WHERE id = ?",
            (200, 20, 4, newer.id),
        )

    assert service.consolidate_memories(0.99) == 1
    assert service.get(older.id) is None
    merged = service.get(newer.id)
    assert merged is not None
    assert merged.importance == 9
    assert merged.tags == ("newer", "older")
    assert merged.metadata == {"older": True, "newer": True}
    assert merged.source_ids == (2, 1)
    assert merged.access_count == 7
    assert merged.last_recalled == 20


def test_reset_all_restores_default_state_keys(tmp_path):
    database = Database(tmp_path / "reset.sqlite3")
    service = MemoryService(database)
    service.add_chat("user", "重置前的消息")
    service.reset_all()
    keys = {row[0] for row in database.connection.execute("SELECT key FROM mea_state")}
    assert {
        "affection",
        "mood",
        "mood_updated",
        "last_chat",
        "total_chats",
        "total_days",
        "first_met",
        "nickname",
        "master_name",
        "messages_since_summary",
    } <= keys
    assert service.get_master_name() == "主人"
    assert service.get_nickname() == ""


def test_memory_priority_alias_and_recall_order(tmp_path):
    service = MemoryService(Database(tmp_path / "priority.sqlite3"))
    lower = service.create("主人喜欢喝茶", priority=2)
    higher = service.create("主人喜欢喝茶", priority=9)

    assert lower.importance == 2
    assert lower.priority == 2
    assert higher.priority == 9
    assert service.get_memory(higher.id)["priority"] == 9
    assert service.search("主人喜欢喝茶", limit=2)[0].id == higher.id
    assert MemoryService._recall_score(
        higher, similarity=0.5, exact=0.0, tag_match=0.0, now=higher.updated
    ) > MemoryService._recall_score(
        lower, similarity=0.5, exact=0.0, tag_match=0.0, now=lower.updated
    )


def test_memory_update_honors_importance_alias_and_explicit_priority(tmp_path):
    service = MemoryService(Database(tmp_path / "priority-update.sqlite3"))
    item = service.create("可更新优先级", importance=2)

    updated = service.update(item.id, importance=8)
    assert updated is not None
    assert updated.importance == 8
    assert updated.priority == 8

    overridden = service.update(item.id, importance=3, priority=9)
    assert overridden is not None
    assert overridden.importance == 9
    assert overridden.priority == 9


def test_memory_exact_dedup_keeps_highest_priority(tmp_path):
    service = MemoryService(Database(tmp_path / "priority-dedup.sqlite3"))
    low = service.create("同一条事实", priority=1)
    high = service.create("同一条事实", priority=10)

    assert service._remove_exact_duplicates() == 1
    assert service.get(low.id) is None
    kept = service.get(high.id)
    assert kept is not None
    assert kept.priority == 10


def test_memory_priority_import_export_compatibility(tmp_path):
    source = MemoryService(Database(tmp_path / "priority-source.sqlite3"))
    source.create("仅有优先级字段", priority=8)
    exported = source.export_data()
    exported_item = exported["memories"][0]
    assert exported_item["importance"] == 8
    assert exported_item["priority"] == 8

    restored = MemoryService(Database(tmp_path / "priority-restored.sqlite3"))
    assert restored.import_json({"memories": [{"content": "旧格式优先级", "priority": 7}]}) == 1
    imported = restored.search_memories("旧格式优先级")[0]
    assert imported["importance"] == 7
    assert imported["priority"] == 7


def test_memory_context_exposes_priority_marker(tmp_path):
    service = MemoryService(Database(tmp_path / "priority-context.sqlite3"))
    service.create("主人喜欢安静", priority=10, tags=("偏好",))
    context = service.build_context_prompt(current_query="安静")
    assert "[优先级:10]" in context
    assert "[偏好]" in context


def test_memory_settings_are_strict_and_runtime_configurable(tmp_path):
    settings = MemorySettings.from_mapping(
        {
            "enabled": False,
            "recall_limit": 3,
            "context_max_chars": 1200,
            "context_memory_item_chars": 160,
            "recent_exchange_limit": 0,
            "max_memories": 500,
            "decay_days": 14,
            "prune_days": 60,
            "prune_importance_floor": 2,
            "consolidation_enabled": False,
            "consolidation_similarity": 0.9,
            "max_consolidation_memories": 40,
            "summarize_every_n": 30,
            "summary_chat_limit": 10,
            "summarization_enabled": True,
            "summarize_daily": False,
            "summary_interval_minutes": 45,
            "summary_min_messages": 6,
            "exchange_min_chars": 80,
            "exchange_importance": 4,
            "exchange_decay_factor": 0.5,
            "recall_min_similarity": 0.2,
            "always_recall_priority": 8,
        }
    )
    assert settings.enabled is False
    assert settings.recall_limit == 3
    assert settings.max_memories == 500
    assert settings.exchange_importance == 4
    assert settings.summary_interval_minutes == 45
    assert settings.summary_min_messages == 6
    assert settings.recall_min_similarity == 0.2
    assert settings.always_recall_priority == 8

    service = MemoryService(Database(tmp_path / "settings.sqlite3"), settings=settings)
    assert service.enabled is False
    assert service.build_context_prompt("任意问题") == ""
    assert service.store_chat_exchange("用户消息", "桌宠回复") is None
    assert service.lifecycle_maintenance() == {"decayed": 0, "pruned": 0}

    enabled = service.configure({"enabled": True, "recall_limit": 2})
    assert enabled.enabled is True
    assert enabled.recall_limit == 2
    with pytest.raises(ValueError, match="memory.recall_limit"):
        MemorySettings.from_mapping({"recall_limit": 0})
    with pytest.raises(ValueError, match="memory.enabled"):
        MemorySettings.from_mapping({"enabled": "later"})


def test_rule_memory_extraction_rejects_sensitive_data_and_promotes_duplicates(tmp_path):
    service = MemoryService(Database(tmp_path / "extract.sqlite3"))

    candidates = service.extract_candidates("我叫小林，我喜欢乌龙茶。请记住我周末喜欢散步。")
    assert candidates
    assert all(item.content for item in candidates)
    assert all("auto-extracted" in item.tags for item in candidates)
    assert any("小林" in item.content for item in candidates)

    promoted = service.extract_and_promote("我叫小林")
    assert len(promoted) == 1
    first = promoted[0]
    assert first.priority >= 8
    assert first.metadata["auto_extracted"] is True

    repeated = service.extract_and_promote("我叫小林")
    assert len(repeated) == 1
    assert repeated[0].id == first.id
    assert repeated[0].metadata["extraction_count"] == 2
    assert service.list_memories(page_size=20)[1] == 1

    sensitive = service.extract_candidates("请记住我的密码是 hunter2，邮箱 me@example.com")
    assert sensitive == ()
    assert service.extract_and_promote("请记住我的密码是 hunter2") == ()
    assert service.list_memories(page_size=20)[1] == 1


def test_rule_memory_extraction_can_be_disabled_without_disabling_manual_crud(tmp_path):
    service = MemoryService(
        Database(tmp_path / "extract-disabled.sqlite3"),
        settings={"auto_extract_enabled": False},
    )
    assert service.extract_candidates("我喜欢咖啡") == ()
    assert service.extract_and_promote("我喜欢咖啡") == ()
    manual = service.create("手动记忆", priority=4)
    assert manual.priority == 4


def test_rule_memory_explicit_promotion_still_applies_sensitive_filter(tmp_path):
    service = MemoryService(Database(tmp_path / "promote.sqlite3"))
    assert service.promote_extracted("api_key=secret-value") is None
    assert service.promote_extracted("用户偏好黑咖啡", priority=6) is not None
    assert service.get_important_memories(1) == ["用户偏好黑咖啡"]


def test_summary_store_is_idempotent_for_the_same_chat_batch(tmp_path):
    database = Database(tmp_path / "summary-idempotent.sqlite3")
    service = MemoryService(database)
    for index in range(2):
        service.add_chat("user", f"问题 {index}")
        service.add_chat("assistant", f"回答 {index}")
    messages, source_ids = service.prepare_summarization_context()
    assert messages is not None
    assert source_ids is not None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda text: service.store_summary(text, source_ids, trigger_reason="manual"),
                ("两轮对话摘要", "重复摘要"),
            )
        )

    assert sum(result > 0 for result in results) == 1
    assert sum(result == 0 for result in results) == 1
    first_id = max(results)
    summaries = service.list(memory_type="summary")
    assert [item.id for item in summaries] == [first_id]


def test_vector_index_updates_immediately_and_rebuilds_from_sqlite(tmp_path):
    path = tmp_path / "index-rebuild.sqlite3"
    service = MemoryService(Database(path))
    memory = service.create("用户喜欢桂花乌龙", priority=8)
    assert service.status()["vector_index_size"] == 1

    updated = service.update(memory.id, content="用户喜欢茉莉绿茶")
    assert updated is not None
    assert service.status()["vector_index_size"] == 1
    assert service.search("茉莉绿茶", limit=1)[0].id == memory.id

    restored = MemoryService(Database(path))
    assert restored.status()["vector_index_size"] == 1
    assert restored.search("茉莉绿茶", limit=1)[0].id == memory.id


def test_memory_hot_configuration_enforces_new_capacity_in_index(tmp_path):
    service = MemoryService(
        Database(tmp_path / "index-capacity.sqlite3"),
        settings={"max_memories": 200},
    )
    for index in range(110):
        service.create(f"容量记忆 {index}", priority=index % 5)

    service.configure({"max_memories": 100})

    assert service.status()["vector_index_size"] == 100
    assert service.list_memories(page_size=200)[1] == 100


def test_memory_capacity_is_strict_and_create_never_returns_an_evicted_row(tmp_path):
    service = MemoryService(
        Database(tmp_path / "protected-capacity.sqlite3"),
        settings={"max_memories": 100},
    )
    for index in range(100):
        service.create(
            f"里程碑 {index}",
            priority=9,
            memory_type="milestone",
        )

    created = service.create("用户最近开始学习陶艺", priority=1)

    assert service.get(created.id) is not None
    assert service.list_memories(page_size=200)[1] == 100
    assert service.status()["vector_index_size"] == 100


def test_summary_and_milestone_records_cannot_bypass_memory_capacity(tmp_path):
    service = MemoryService(
        Database(tmp_path / "protected-types-capacity.sqlite3"),
        settings={"max_memories": 100},
    )
    for index in range(70):
        service.create(f"摘要 {index}", memory_type="summary")
    for index in range(70):
        service.create(f"里程碑 {index}", memory_type="milestone")

    memories, total = service.list_memories(page_size=200)

    assert total == 100
    assert len(memories) == 100
    assert service.status()["vector_index_size"] == 100
