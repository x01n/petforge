from __future__ import annotations

import asyncio
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from db import Database
from gui.web.control_surface import public_control_state
from services.diary import PRIVATE_VISIBILITY, PUBLIC_VISIBILITY, PetDiaryService
from services.memory import MemoryService
from services.tools import PermissionService, ToolCallContext, ToolExecutionService, ToolRegistry
from services.tools.builtins import register_builtin_tools


def test_pet_diary_uses_independent_tables_and_default_list_hides_private_entries(tmp_path) -> None:
    database = Database(tmp_path / "diary.sqlite3")
    service = PetDiaryService(database, clock=lambda: 100.0)
    internal = service.open_internal_session("agent:diary")

    hidden = internal.create(
        "今天偷偷把主人的闹钟提前了五分钟。",
        tags=("日常", "秘密"),
        priority=9,
    )
    public = internal.create(
        "今天一起看了晚霞。",
        visibility=PUBLIC_VISIBILITY,
        tags=("日常",),
        priority=7,
    )

    assert hidden.visibility == PRIVATE_VISIBILITY
    assert service.list_entries() == (public,)
    assert service.get_entry(hidden.id) is None
    assert service.get_entry(public.id) == public
    assert internal.list_entries() == (public,)
    assert {entry.id for entry in internal.list_entries(include_private=True)} == {
        hidden.id,
        public.id,
    }

    tables = {
        row[0]
        for row in database.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    assert {"pet_diary_entries", "pet_diary_audit"} <= tables
    assert database.connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


def test_pet_diary_private_content_never_enters_memory_recall_or_public_state(tmp_path) -> None:
    database = Database(tmp_path / "isolated.sqlite3")
    diary = PetDiaryService(database, clock=lambda: 200.0)
    internal = diary.open_internal_session("agent:private-diary")
    secret = "只有桌宠知道的蓝色钥匙藏在第七码头。"
    internal.create(secret, tags=("隐藏",), priority=10)

    memory = MemoryService(database)
    try:
        assert memory.search("蓝色钥匙", limit=10) == ()
        assert secret not in memory.build_context_prompt("蓝色钥匙")
        assert memory.list() == ()
        projected = public_control_state(
            {
                "memory": memory.status(),
                "diary": {
                    "content": secret,
                    "private_entry_count": 1,
                    "tags": ["隐藏"],
                },
            }
        )
        assert "diary" not in projected
        assert secret not in repr(projected)
        assert "private_entry_count" not in repr(projected)
        assert diary.status() == {"enabled": True, "public_entry_count": 0}
        assert secret not in repr(diary.status())
    finally:
        memory.close()


def test_internal_diary_crud_archival_retention_and_audit_are_transactional(tmp_path) -> None:
    now = [1_000.0]
    database = Database(tmp_path / "audit.sqlite3")
    service = PetDiaryService(database, clock=lambda: now[0])
    internal = service.open_internal_session("agent:journal")

    entry = internal.create(
        "准备学习新的舞蹈。",
        priority=8,
        tags=("计划", "计划"),
        retention_until=2_000.0,
    )
    assert entry.tags == ("计划",)
    assert internal.get(entry.id) == entry

    updated = internal.update(
        entry.id,
        content="已经学会新的舞蹈。",
        visibility=PUBLIC_VISIBILITY,
        priority=10,
        tags=("成长",),
    )
    assert updated is not None
    assert updated.visibility == PUBLIC_VISIBILITY
    assert updated.priority == 10
    assert service.get_entry(entry.id) == updated

    archived = internal.set_archived(entry.id)
    assert archived is not None and archived.is_archived
    assert service.list_entries() == ()
    restored = internal.set_archived(entry.id, archived=False)
    assert restored is not None and not restored.is_archived

    with pytest.raises(PermissionError, match="retention_until"):
        internal.delete(entry.id)
    assert internal.get(entry.id) is not None
    assert internal.delete(entry.id, force=True) is True
    assert internal.get(entry.id) is None

    audit = internal.audit_log(entry_id=entry.id, limit=100)
    actions = [record.action for record in reversed(audit)]
    assert actions == [
        "create",
        "read",
        "update",
        "archive",
        "restore",
        "delete",
        "read",
        "delete",
        "read",
    ]
    assert any(record.result == "retention_blocked" for record in audit)
    assert any(record.private_access for record in audit)
    serialized_audit = json.dumps(
        [record.details for record in audit], ensure_ascii=False, sort_keys=True
    )
    assert "准备学习新的舞蹈" not in serialized_audit
    assert "已经学会新的舞蹈" not in serialized_audit
    assert "成长" not in serialized_audit
    assert database.connection.execute(
        "SELECT COUNT(*) FROM pet_diary_audit WHERE entry_id = ?", (entry.id,)
    ).fetchone()[0] == len(audit)


def test_diary_input_validation_and_access_capability_fail_closed(tmp_path) -> None:
    database = Database(tmp_path / "validation.sqlite3")
    service = PetDiaryService(database, clock=lambda: 300.0)
    internal = service.open_internal_session("agent:validator")

    with pytest.raises(ValueError, match="visibility"):
        internal.create("内容", visibility="PRIVATE")
    with pytest.raises(ValueError, match="priority"):
        internal.create("内容", priority=11)
    with pytest.raises(ValueError, match="tags"):
        internal.create("内容", tags="错误")
    with pytest.raises(ValueError, match="actor"):
        service.open_internal_session("bad\nactor")
    with pytest.raises(PermissionError, match="internal diary access"):
        service._repository._get(object(), 1, include_archived=True)


def test_pet_diary_schema_initialization_is_idempotent_without_memory_schema_change() -> None:
    database = Database(":memory:")
    before = database.connection.execute(
        "SELECT MAX(version) FROM memory_schema_version"
    ).fetchone()[0]

    first = PetDiaryService(database, clock=lambda: 10.0)
    applied_at = database.connection.execute(
        "SELECT applied_at FROM pet_diary_schema_version WHERE version = 1"
    ).fetchone()[0]
    second = PetDiaryService(database, clock=lambda: 20.0)

    assert first.list_entries() == ()
    assert second.list_entries() == ()
    assert applied_at == 10.0
    assert (
        database.connection.execute(
            "SELECT applied_at FROM pet_diary_schema_version WHERE version = 1"
        ).fetchone()[0]
        == applied_at
    )
    after = database.connection.execute(
        "SELECT MAX(version) FROM memory_schema_version"
    ).fetchone()[0]
    assert after == before


def test_public_diary_projection_hides_expired_entries_before_maintenance(tmp_path) -> None:
    now = [400.0]
    database = Database(tmp_path / "retention.sqlite3")
    service = PetDiaryService(database, clock=lambda: now[0])
    internal = service.open_internal_session("maintenance:diary")
    entry = internal.create(
        "这条公开日记只展示十秒。",
        visibility=PUBLIC_VISIBILITY,
        retention_until=410.0,
    )

    assert service.list_entries() == (entry,)
    now[0] = 410.0
    assert service.list_entries() == ()
    assert service.get_entry(entry.id) is None
    assert service.status() == {"enabled": True, "public_entry_count": 0}

    assert internal.archive_expired() == (entry.id,)
    archived = internal.get(entry.id)
    assert archived is not None and archived.is_archived
    audit = internal.audit_log(entry_id=entry.id)
    retention_audit = next(
        record for record in audit if record.details.get("reason") == "retention_expired"
    )
    assert retention_audit.action == "archive"
    assert "这条公开日记" not in repr(retention_audit.details)


def test_diary_repository_serializes_concurrent_creates_and_audits(tmp_path) -> None:
    database = Database(tmp_path / "concurrent.sqlite3")
    service = PetDiaryService(database, clock=lambda: 500.0)

    def create_batch(worker: int) -> tuple[int, ...]:
        session = service.open_internal_session(f"agent:worker-{worker}")
        return tuple(
            session.create(f"worker={worker}; item={item}", priority=worker % 11).id
            for item in range(20)
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        batches = tuple(executor.map(create_batch, range(8)))

    entry_ids = tuple(entry_id for batch in batches for entry_id in batch)
    assert len(entry_ids) == 160
    assert len(set(entry_ids)) == 160
    inspector = service.open_internal_session("audit:inspector")
    entries = inspector.list_entries(include_private=True, limit=500)
    assert len(entries) == 160
    assert (
        database.connection.execute(
            "SELECT COUNT(*) FROM pet_diary_audit WHERE action = 'create'"
        ).fetchone()[0]
        == 160
    )


def test_diary_schema_rejects_missing_columns_and_unknown_versions(tmp_path) -> None:
    malformed_database = Database(tmp_path / "malformed.sqlite3")
    malformed_database.connection.execute(
        """
        CREATE TABLE pet_diary_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL
        )
        """
    )
    malformed_database.connection.commit()
    with pytest.raises(sqlite3.OperationalError, match="retention_until"):
        PetDiaryService(malformed_database)

    database = Database(tmp_path / "future.sqlite3")
    PetDiaryService(database, clock=lambda: 600.0)
    database.connection.execute("DELETE FROM pet_diary_schema_version")
    database.connection.execute(
        "INSERT INTO pet_diary_schema_version (version, applied_at) VALUES (?, ?)",
        (999, 600.0),
    )
    database.connection.commit()
    with pytest.raises(sqlite3.OperationalError, match="schema.*999"):
        PetDiaryService(database, clock=lambda: 600.0)


def test_diary_reopens_without_mixing_private_entries_into_default_list(tmp_path) -> None:
    path = tmp_path / "persistent-diary.sqlite3"
    first_database = Database(path)
    first_service = PetDiaryService(first_database, clock=lambda: 700.0)
    first_internal = first_service.open_internal_session("agent:first-run")
    public = first_internal.create("公开日记", visibility=PUBLIC_VISIBILITY)
    private = first_internal.create("私密日记")
    first_database.close()

    restored_database = Database(path)
    restored_service = PetDiaryService(restored_database, clock=lambda: 701.0)
    restored_internal = restored_service.open_internal_session("agent:restored-run")
    assert restored_service.list_entries() == (public,)
    assert restored_service.get_entry(private.id) is None
    assert {entry.id for entry in restored_internal.list_entries(include_private=True)} == {
        public.id,
        private.id,
    }


def test_diary_audit_retention_is_bounded_without_losing_entries(tmp_path) -> None:
    database = Database(tmp_path / "bounded-audit.sqlite3")
    service = PetDiaryService(database, clock=lambda: 800.0, max_audit_rows=3)
    internal = service.open_internal_session("agent:bounded-audit")

    created = tuple(internal.create(f"日记 {index}") for index in range(5))

    assert len(internal.audit_log(limit=100)) == 3
    assert database.connection.execute("SELECT COUNT(*) FROM pet_diary_entries").fetchone()[0] == 5
    assert service.list_entries() == ()
    assert internal.get(created[-1].id) == created[-1]


def test_pet_diary_tools_write_and_recall_private_entries_without_public_projection(
    tmp_path,
) -> None:
    database = Database(tmp_path / "tool-diary.sqlite3")
    diary = PetDiaryService(database, clock=lambda: 500.0)
    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=object(),
        pet_controller=object(),
        diary_provider=lambda: diary,
    )
    executor = ToolExecutionService(registry, PermissionService())
    context = ToolCallContext("profile", "session", "turn")

    async def scenario():
        written = await executor.execute(
            call_id="write-diary",
            identity="pet:diary_write",
            arguments={
                "content": "偷偷练习了新的欢迎动作。",
                "priority": 8,
                "tags": ["成长"],
            },
            context=context,
        )
        recalled = await executor.execute(
            call_id="recall-diary",
            identity="pet:diary_recall",
            arguments={"include_private": True, "limit": 3},
            context=context,
        )
        return written, recalled

    written, recalled = asyncio.run(scenario())
    assert written.status == "completed"
    assert written.content == {
        "status": "completed",
        "entry_id": 1,
        "visibility": "private",
        "priority": 8,
        "tag_count": 1,
    }
    assert "偷偷练习" not in repr(written.content)
    assert recalled.status == "completed"
    assert recalled.content["entries"][0]["content"] == "偷偷练习了新的欢迎动作。"
    assert diary.list_entries() == ()
    assert registry.require("pet:diary_recall").read_only is True
