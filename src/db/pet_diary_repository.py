"""独立桌宠日记与访问审计仓储。"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .database import Database

PUBLIC_VISIBILITY = "public"
PRIVATE_VISIBILITY = "private"
DIARY_VISIBILITIES = frozenset({PUBLIC_VISIBILITY, PRIVATE_VISIBILITY})
PET_DIARY_SCHEMA_VERSION = 1
MAX_DIARY_AUDIT_ROWS = 8192

MAX_DIARY_CONTENT_LENGTH = 10_000
MAX_DIARY_TAGS = 16
MAX_DIARY_TAG_LENGTH = 64


@dataclass(frozen=True)
class PetDiaryEntry:
    """与 ``pet_diary_entries`` 表一一对应的不可变日记条目。"""

    id: int
    content: str
    visibility: str
    priority: int
    tags: tuple[str, ...]
    retention_until: float | None
    archived_at: float | None
    created_at: float
    updated_at: float

    @property
    def is_private(self) -> bool:
        """返回条目是否仅允许内部受控会话读取。"""

        return self.visibility == PRIVATE_VISIBILITY

    @property
    def is_archived(self) -> bool:
        """返回条目是否已归档。"""

        return self.archived_at is not None


@dataclass(frozen=True)
class PetDiaryAuditRecord:
    """不含日记正文和标签的访问审计记录。"""

    id: int
    entry_id: int | None
    action: str
    actor: str
    result: str
    private_access: bool
    occurred_at: float
    details: dict[str, Any]


@dataclass(frozen=True)
class _InternalDiaryAccess:
    key: object
    actor: str


class PetDiaryRepository:
    """持久化日记，并在同一事务中记录内部读写审计。"""

    _MAX_AUDIT_ROWS = MAX_DIARY_AUDIT_ROWS
    _ACTIONS = frozenset({"create", "read", "list", "update", "archive", "restore", "delete"})
    _RESULTS = frozenset({"ok", "not_found", "hidden", "retention_blocked", "invalid_record"})
    _UPDATE_COLUMNS = {
        "content": "content",
        "visibility": "visibility",
        "priority": "priority",
        "tags": "tags",
        "retention_until": "retention_until",
    }
    _REQUIRED_COLUMNS = {
        "pet_diary_schema_version": frozenset({"version", "applied_at"}),
        "pet_diary_entries": frozenset(
            {
                "id",
                "content",
                "visibility",
                "priority",
                "tags",
                "retention_until",
                "archived_at",
                "created_at",
                "updated_at",
            }
        ),
        "pet_diary_audit": frozenset(
            {
                "id",
                "entry_id",
                "action",
                "actor",
                "result",
                "private_access",
                "occurred_at",
                "details",
            }
        ),
    }
    _PRIMARY_KEYS = {
        "pet_diary_schema_version": ("version",),
        "pet_diary_entries": ("id",),
        "pet_diary_audit": ("id",),
    }

    def __init__(
        self,
        database: Database,
        *,
        clock: Callable[[], float] | None = None,
        max_audit_rows: int = _MAX_AUDIT_ROWS,
    ) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        self._database = database
        self._clock = clock or time.time
        self._max_audit_rows = max(1, min(int(max_audit_rows), 100_000))
        self._access_key = object()
        self._ensure_schema()

    @staticmethod
    def _table_columns(connection: sqlite3.Connection, table: str) -> frozenset[str]:
        """读取固定日记表列名；表名只来自仓储内部常量。"""

        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return frozenset(str(row[1]) for row in rows)

    @staticmethod
    def _table_primary_key(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
        """读取固定日记表主键列，防止旧表身份结构被静默接受。"""

        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        keyed = sorted(
            ((int(row[5]), str(row[1])) for row in rows if int(row[5]) > 0),
            key=lambda item: item[0],
        )
        return tuple(name for _, name in keyed)

    @classmethod
    def _validate_schema(cls, connection: sqlite3.Connection) -> None:
        """拒绝缺列或主键不一致的残缺日记表，避免猜测式迁移。"""

        for table, required in cls._REQUIRED_COLUMNS.items():
            columns = cls._table_columns(connection, table)
            missing = required - columns
            if missing:
                names = ", ".join(sorted(missing))
                raise sqlite3.OperationalError(f"{table} 缺少不可推断的列: {names}")
            primary_key = cls._table_primary_key(connection, table)
            if primary_key != cls._PRIMARY_KEYS[table]:
                expected = ", ".join(cls._PRIMARY_KEYS[table])
                actual = ", ".join(primary_key) or "无"
                raise sqlite3.OperationalError(
                    f"{table} 主键结构不兼容: 期望 {expected}，实际 {actual}"
                )

    def _ensure_schema(self) -> None:
        """创建独立日记表，并对已存在的结构执行严格版本校验。"""

        with self._database.transaction() as connection:
            statements = (
                """
                CREATE TABLE IF NOT EXISTS pet_diary_schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS pet_diary_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    visibility TEXT NOT NULL DEFAULT 'private'
                        CHECK (visibility IN ('public', 'private')),
                    priority INTEGER NOT NULL DEFAULT 5
                        CHECK (priority BETWEEN 0 AND 10),
                    tags TEXT NOT NULL DEFAULT '[]',
                    retention_until REAL,
                    archived_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS pet_diary_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id INTEGER,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    result TEXT NOT NULL,
                    private_access INTEGER NOT NULL DEFAULT 0
                        CHECK (private_access IN (0, 1)),
                    occurred_at REAL NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}'
                )
                """,
            )
            for statement in statements:
                connection.execute(statement)
            self._validate_schema(connection)
            version_row = connection.execute(
                "SELECT MAX(version) FROM pet_diary_schema_version"
            ).fetchone()
            version = int(version_row[0] or 0) if version_row else 0
            if version not in {0, PET_DIARY_SCHEMA_VERSION}:
                raise sqlite3.OperationalError(f"不支持的桌宠日记 schema 版本: {version}")
            if version == 0:
                connection.execute(
                    """
                    INSERT INTO pet_diary_schema_version (version, applied_at)
                    VALUES (?, ?)
                    """,
                    (PET_DIARY_SCHEMA_VERSION, self._now()),
                )
            index_statements = (
                """
                CREATE INDEX IF NOT EXISTS idx_pet_diary_public_recent
                    ON pet_diary_entries (visibility, archived_at, priority DESC, updated_at DESC)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_pet_diary_audit_entry
                    ON pet_diary_audit (entry_id, occurred_at DESC, id DESC)
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_pet_diary_audit_recent
                    ON pet_diary_audit (occurred_at DESC, id DESC)
                """,
            )
            for statement in index_statements:
                connection.execute(statement)

    def _now(self) -> float:
        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("diary clock must return a finite timestamp") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("diary clock must return a finite non-negative timestamp")
        return value

    @staticmethod
    def _entry_id(value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("entry_id must be a positive integer")
        try:
            result = int(str(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("entry_id must be a positive integer") from exc
        if result <= 0:
            raise ValueError("entry_id must be a positive integer")
        return result

    @staticmethod
    def _actor(value: object) -> str:
        actor = str(value or "").strip()
        if not actor or len(actor) > 128 or any(char in actor for char in "\r\n\x00"):
            raise ValueError("diary actor is invalid")
        return actor

    @staticmethod
    def _content(value: object) -> str:
        content = str(value or "").strip()
        if not content or len(content) > MAX_DIARY_CONTENT_LENGTH or "\x00" in content:
            raise ValueError("diary content must contain 1 to 10000 characters")
        return content

    @staticmethod
    def _visibility(value: object) -> str:
        visibility = str(value or "")
        if visibility not in DIARY_VISIBILITIES:
            raise ValueError("diary visibility must be public or private")
        return visibility

    @staticmethod
    def _priority(value: object) -> int:
        if isinstance(value, bool):
            raise ValueError("diary priority must be an integer from 0 to 10")
        try:
            priority = int(str(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("diary priority must be an integer from 0 to 10") from exc
        if priority < 0 or priority > 10:
            raise ValueError("diary priority must be an integer from 0 to 10")
        return priority

    @staticmethod
    def _tags(value: object) -> tuple[str, ...]:
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise ValueError("diary tags must be a sequence")
        result: list[str] = []
        seen: set[str] = set()
        for item in value:
            tag = str(item or "").strip()
            if (
                not tag
                or len(tag) > MAX_DIARY_TAG_LENGTH
                or any(char in tag for char in "\r\n\x00")
            ):
                raise ValueError("diary tag is invalid")
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
            if len(result) > MAX_DIARY_TAGS:
                raise ValueError("diary tags exceed the limit")
        return tuple(result)

    @staticmethod
    def _retention_until(value: object) -> float | None:
        if value is None:
            return None
        try:
            retention_until = float(str(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("retention_until must be a finite non-negative timestamp") from exc
        if not math.isfinite(retention_until) or retention_until < 0:
            raise ValueError("retention_until must be a finite non-negative timestamp")
        return retention_until

    @staticmethod
    def _decode_tags(value: object) -> tuple[str, ...] | None:
        try:
            parsed = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(parsed, list):
            return None
        try:
            return PetDiaryRepository._tags(parsed)
        except ValueError:
            return None

    @classmethod
    def _entry_from_row(cls, row: Any) -> PetDiaryEntry | None:
        if row is None:
            return None
        tags = cls._decode_tags(row["tags"])
        if tags is None:
            return None
        try:
            return PetDiaryEntry(
                id=cls._entry_id(row["id"]),
                content=cls._content(row["content"]),
                visibility=cls._visibility(row["visibility"]),
                priority=cls._priority(row["priority"]),
                tags=tags,
                retention_until=cls._retention_until(row["retention_until"]),
                archived_at=cls._retention_until(row["archived_at"]),
                created_at=float(row["created_at"]),
                updated_at=float(row["updated_at"]),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _audit_from_row(row: Any) -> PetDiaryAuditRecord | None:
        try:
            details = json.loads(str(row["details"]))
            if not isinstance(details, Mapping):
                return None
            raw_entry_id = row["entry_id"]
            entry_id = int(raw_entry_id) if raw_entry_id is not None else None
            return PetDiaryAuditRecord(
                id=int(row["id"]),
                entry_id=entry_id,
                action=str(row["action"]),
                actor=str(row["actor"]),
                result=str(row["result"]),
                private_access=bool(int(row["private_access"])),
                occurred_at=float(row["occurred_at"]),
                details=dict(details),
            )
        except (KeyError, TypeError, ValueError, OverflowError, json.JSONDecodeError):
            return None

    def _issue_internal_access(self, actor: object) -> _InternalDiaryAccess:
        return _InternalDiaryAccess(self._access_key, self._actor(actor))

    def _require_access(self, access: object) -> _InternalDiaryAccess:
        if not isinstance(access, _InternalDiaryAccess) or access.key is not self._access_key:
            raise PermissionError("internal diary access is required")
        return access

    def _record_audit(
        self,
        connection: Any,
        *,
        access: _InternalDiaryAccess,
        entry_id: int | None,
        action: str,
        result: str,
        private_access: bool,
        occurred_at: float,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if action not in self._ACTIONS or result not in self._RESULTS:
            raise ValueError("diary audit value is invalid")
        payload = json.dumps(
            dict(details or {}),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        connection.execute(
            """
            INSERT INTO pet_diary_audit (
                entry_id, action, actor, result, private_access, occurred_at, details
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                action,
                access.actor,
                result,
                1 if private_access else 0,
                occurred_at,
                payload,
            ),
        )
        connection.execute(
            """
            DELETE FROM pet_diary_audit
            WHERE id NOT IN (
                SELECT id FROM pet_diary_audit ORDER BY id DESC LIMIT ?
            )
            """,
            (self._max_audit_rows,),
        )

    @staticmethod
    def _select_entry(connection: Any, entry_id: int) -> Any:
        return connection.execute(
            "SELECT * FROM pet_diary_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()

    def list_public(self, *, limit: int = 100) -> tuple[PetDiaryEntry, ...]:
        """返回公开、未归档且仍在保留期内的条目。"""

        safe_limit = max(0, min(int(limit), 500))
        if safe_limit == 0:
            return ()
        now = self._now()
        with self._database._lock:
            rows = self._database.connection.execute(
                """
                SELECT * FROM pet_diary_entries
                WHERE visibility = 'public'
                  AND archived_at IS NULL
                  AND (retention_until IS NULL OR retention_until > ?)
                ORDER BY priority DESC, updated_at DESC, id DESC
                LIMIT ?
                """,
                (now, safe_limit),
            ).fetchall()
        return tuple(entry for row in rows if (entry := self._entry_from_row(row)) is not None)

    def get_public(self, entry_id: object) -> PetDiaryEntry | None:
        """按编号读取公开、未归档且仍在保留期内的条目。"""

        safe_id = self._entry_id(entry_id)
        now = self._now()
        with self._database._lock:
            row = self._database.connection.execute(
                """
                SELECT * FROM pet_diary_entries
                WHERE id = ?
                  AND visibility = 'public'
                  AND archived_at IS NULL
                  AND (retention_until IS NULL OR retention_until > ?)
                """,
                (safe_id, now),
            ).fetchone()
        return self._entry_from_row(row)

    def public_count(self) -> int:
        """返回公开、未归档且仍在保留期内的条目数。"""

        now = self._now()
        with self._database._lock:
            row = self._database.connection.execute(
                """
                SELECT COUNT(*) FROM pet_diary_entries
                WHERE visibility = 'public'
                  AND archived_at IS NULL
                  AND (retention_until IS NULL OR retention_until > ?)
                """,
                (now,),
            ).fetchone()
        return int(row[0]) if row is not None else 0

    def _create(
        self,
        access: object,
        content: object,
        *,
        visibility: object,
        priority: object,
        tags: object,
        retention_until: object,
    ) -> PetDiaryEntry:
        authorized = self._require_access(access)
        safe_content = self._content(content)
        safe_visibility = self._visibility(visibility)
        safe_priority = self._priority(priority)
        safe_tags = self._tags(tags)
        safe_retention = self._retention_until(retention_until)
        now = self._now()
        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO pet_diary_entries (
                    content, visibility, priority, tags, retention_until,
                    archived_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    safe_content,
                    safe_visibility,
                    safe_priority,
                    json.dumps(safe_tags, ensure_ascii=False, separators=(",", ":")),
                    safe_retention,
                    now,
                    now,
                ),
            )
            raw_entry_id = cursor.lastrowid
            if raw_entry_id is None:
                raise RuntimeError("created diary entry did not return an identifier")
            entry_id = int(raw_entry_id)
            row = self._select_entry(connection, entry_id)
            entry = self._entry_from_row(row)
            if entry is None:
                raise RuntimeError("created diary entry could not be restored")
            self._record_audit(
                connection,
                access=authorized,
                entry_id=entry_id,
                action="create",
                result="ok",
                private_access=entry.is_private,
                occurred_at=now,
                details={"visibility": entry.visibility, "priority": entry.priority},
            )
        return entry

    def _get(
        self,
        access: object,
        entry_id: object,
        *,
        include_archived: bool,
    ) -> PetDiaryEntry | None:
        authorized = self._require_access(access)
        safe_id = self._entry_id(entry_id)
        now = self._now()
        with self._database.transaction() as connection:
            row = self._select_entry(connection, safe_id)
            entry = self._entry_from_row(row)
            if row is not None and entry is None:
                result = "invalid_record"
            elif entry is None:
                result = "not_found"
            elif entry.is_archived and not include_archived:
                result = "hidden"
            else:
                result = "ok"
            private_access = bool(entry and entry.is_private)
            if result == "hidden":
                entry = None
            self._record_audit(
                connection,
                access=authorized,
                entry_id=safe_id,
                action="read",
                result=result,
                private_access=private_access,
                occurred_at=now,
                details={"include_archived": bool(include_archived)},
            )
        return entry

    def _list(
        self,
        access: object,
        *,
        include_private: bool,
        include_archived: bool,
        limit: int,
    ) -> tuple[PetDiaryEntry, ...]:
        authorized = self._require_access(access)
        safe_limit = max(0, min(int(limit), 500))
        clauses: list[str] = []
        if not include_private:
            clauses.append("visibility = 'public'")
        if not include_archived:
            clauses.append("archived_at IS NULL")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        now = self._now()
        with self._database.transaction() as connection:
            rows = (
                connection.execute(
                    f"""
                    SELECT * FROM pet_diary_entries {where}
                    ORDER BY priority DESC, updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
                if safe_limit
                else ()
            )
            entries = tuple(
                entry for row in rows if (entry := self._entry_from_row(row)) is not None
            )
            self._record_audit(
                connection,
                access=authorized,
                entry_id=None,
                action="list",
                result="ok",
                private_access=bool(include_private),
                occurred_at=now,
                details={
                    "include_private": bool(include_private),
                    "include_archived": bool(include_archived),
                    "result_count": len(entries),
                },
            )
        return entries

    def _update(
        self,
        access: object,
        entry_id: object,
        changes: Mapping[str, object],
    ) -> PetDiaryEntry | None:
        authorized = self._require_access(access)
        safe_id = self._entry_id(entry_id)
        unknown = set(changes) - set(self._UPDATE_COLUMNS)
        if unknown:
            raise ValueError(f"unsupported diary fields: {', '.join(sorted(unknown))}")
        if not changes:
            raise ValueError("at least one diary field is required")
        normalized: dict[str, object] = {}
        if "content" in changes:
            normalized["content"] = self._content(changes["content"])
        if "visibility" in changes:
            normalized["visibility"] = self._visibility(changes["visibility"])
        if "priority" in changes:
            normalized["priority"] = self._priority(changes["priority"])
        if "tags" in changes:
            safe_tags = self._tags(changes["tags"])
            normalized["tags"] = json.dumps(safe_tags, ensure_ascii=False, separators=(",", ":"))
        if "retention_until" in changes:
            normalized["retention_until"] = self._retention_until(changes["retention_until"])
        now = self._now()
        with self._database.transaction() as connection:
            previous = self._entry_from_row(self._select_entry(connection, safe_id))
            if previous is None:
                self._record_audit(
                    connection,
                    access=authorized,
                    entry_id=safe_id,
                    action="update",
                    result="not_found",
                    private_access=False,
                    occurred_at=now,
                    details={"fields": sorted(normalized)},
                )
                return None
            assignments = [f"{self._UPDATE_COLUMNS[name]} = ?" for name in normalized]
            values = [normalized[name] for name in normalized]
            assignments.append("updated_at = ?")
            values.extend((now, safe_id))
            connection.execute(
                f"UPDATE pet_diary_entries SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            entry = self._entry_from_row(self._select_entry(connection, safe_id))
            if entry is None:
                raise RuntimeError("updated diary entry could not be restored")
            self._record_audit(
                connection,
                access=authorized,
                entry_id=safe_id,
                action="update",
                result="ok",
                private_access=previous.is_private or entry.is_private,
                occurred_at=now,
                details={"fields": sorted(normalized)},
            )
        return entry

    def _set_archived(
        self,
        access: object,
        entry_id: object,
        *,
        archived: bool,
    ) -> PetDiaryEntry | None:
        authorized = self._require_access(access)
        safe_id = self._entry_id(entry_id)
        now = self._now()
        action = "archive" if archived else "restore"
        with self._database.transaction() as connection:
            previous = self._entry_from_row(self._select_entry(connection, safe_id))
            if previous is None:
                self._record_audit(
                    connection,
                    access=authorized,
                    entry_id=safe_id,
                    action=action,
                    result="not_found",
                    private_access=False,
                    occurred_at=now,
                )
                return None
            archived_at = now if archived else None
            connection.execute(
                "UPDATE pet_diary_entries SET archived_at = ?, updated_at = ? WHERE id = ?",
                (archived_at, now, safe_id),
            )
            entry = self._entry_from_row(self._select_entry(connection, safe_id))
            if entry is None:
                raise RuntimeError("archived diary entry could not be restored")
            self._record_audit(
                connection,
                access=authorized,
                entry_id=safe_id,
                action=action,
                result="ok",
                private_access=entry.is_private,
                occurred_at=now,
            )
        return entry

    def _delete(self, access: object, entry_id: object, *, force: bool) -> bool:
        authorized = self._require_access(access)
        safe_id = self._entry_id(entry_id)
        now = self._now()
        blocked = False
        with self._database.transaction() as connection:
            entry = self._entry_from_row(self._select_entry(connection, safe_id))
            if entry is None:
                self._record_audit(
                    connection,
                    access=authorized,
                    entry_id=safe_id,
                    action="delete",
                    result="not_found",
                    private_access=False,
                    occurred_at=now,
                    details={"force": bool(force)},
                )
                return False
            blocked = (
                entry.retention_until is not None
                and entry.retention_until > now
                and not bool(force)
            )
            if not blocked:
                connection.execute("DELETE FROM pet_diary_entries WHERE id = ?", (safe_id,))
            self._record_audit(
                connection,
                access=authorized,
                entry_id=safe_id,
                action="delete",
                result="retention_blocked" if blocked else "ok",
                private_access=entry.is_private,
                occurred_at=now,
                details={"force": bool(force)},
            )
        if blocked:
            raise PermissionError("diary entry is protected by retention_until")
        return True

    def _archive_expired(self, access: object) -> tuple[int, ...]:
        """归档已过保留期限的条目，并为每个条目保留审计记录。"""

        authorized = self._require_access(access)
        now = self._now()
        with self._database.transaction() as connection:
            rows = connection.execute(
                """
                SELECT id, visibility FROM pet_diary_entries
                WHERE archived_at IS NULL
                  AND retention_until IS NOT NULL
                  AND retention_until <= ?
                ORDER BY id
                """,
                (now,),
            ).fetchall()
            entry_ids = tuple(int(row["id"]) for row in rows)
            for row in rows:
                entry_id = int(row["id"])
                connection.execute(
                    "UPDATE pet_diary_entries SET archived_at = ?, updated_at = ? WHERE id = ?",
                    (now, now, entry_id),
                )
                self._record_audit(
                    connection,
                    access=authorized,
                    entry_id=entry_id,
                    action="archive",
                    result="ok",
                    private_access=str(row["visibility"]) == PRIVATE_VISIBILITY,
                    occurred_at=now,
                    details={"reason": "retention_expired"},
                )
        return entry_ids

    def _list_audit(
        self,
        access: object,
        *,
        limit: int,
        entry_id: object | None,
    ) -> tuple[PetDiaryAuditRecord, ...]:
        _ = self._require_access(access)
        safe_limit = max(0, min(int(limit), 1000))
        if safe_limit == 0:
            return ()
        clause = ""
        parameters: tuple[object, ...]
        if entry_id is None:
            parameters = (safe_limit,)
        else:
            safe_id = self._entry_id(entry_id)
            clause = "WHERE entry_id = ?"
            parameters = (safe_id, safe_limit)
        with self._database._lock:
            rows = self._database.connection.execute(
                f"""
                SELECT * FROM pet_diary_audit {clause}
                ORDER BY id DESC LIMIT ?
                """,
                parameters,
            ).fetchall()
        return tuple(record for row in rows if (record := self._audit_from_row(row)) is not None)


__all__ = [
    "DIARY_VISIBILITIES",
    "MAX_DIARY_AUDIT_ROWS",
    "MAX_DIARY_CONTENT_LENGTH",
    "PET_DIARY_SCHEMA_VERSION",
    "MAX_DIARY_TAG_LENGTH",
    "MAX_DIARY_TAGS",
    "PRIVATE_VISIBILITY",
    "PUBLIC_VISIBILITY",
    "PetDiaryAuditRecord",
    "PetDiaryEntry",
    "PetDiaryRepository",
]
