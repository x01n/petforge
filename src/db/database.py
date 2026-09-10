"""线程安全的 SQLite 连接与 schema v7 迁移。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from core.memory_embedding import compute_embedding, valid_embedding_item

SCHEMA_VERSION = 7


class SchemaMigrator:
    """创建并补齐与旧实现兼容的 v7 数据表。"""

    _CREATE_STATEMENTS = (
        """
        CREATE TABLE IF NOT EXISTS memory_schema_version (
            version INTEGER PRIMARY KEY,
            applied_at REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            mood TEXT DEFAULT 'neutral',
            timestamp REAL NOT NULL,
            summarized INTEGER DEFAULT 0
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS mea_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS master_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            description TEXT NOT NULL,
            data TEXT DEFAULT '{}',
            timestamp REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memories (
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
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS conversation_turns (
            mode TEXT NOT NULL,
            profile_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            source TEXT NOT NULL,
            user_text TEXT NOT NULL DEFAULT '',
            segments TEXT NOT NULL DEFAULT '[]',
            system_entries TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            status TEXT NOT NULL,
            error_text TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (mode, profile_id, session_id, turn_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_ann_journal (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
            created_at REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_ann_generations (
            generation TEXT PRIMARY KEY,
            settings_fingerprint TEXT NOT NULL,
            provider TEXT NOT NULL,
            package_version TEXT NOT NULL,
            model_id TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            dimension INTEGER NOT NULL CHECK (dimension > 0),
            index_revision TEXT NOT NULL,
            index_path TEXT NOT NULL DEFAULT '',
            item_count INTEGER NOT NULL CHECK (item_count >= 0),
            applied_sequence INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 0 CHECK (active IN (0, 1))
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS memory_semantic_vectors (
            generation TEXT NOT NULL,
            memory_id INTEGER NOT NULL,
            content_digest TEXT NOT NULL,
            vector BLOB NOT NULL,
            dimension INTEGER NOT NULL CHECK (dimension > 0),
            tombstone INTEGER NOT NULL DEFAULT 0 CHECK (tombstone IN (0, 1)),
            updated_at REAL NOT NULL,
            PRIMARY KEY (generation, memory_id),
            FOREIGN KEY (generation) REFERENCES memory_ann_generations(generation)
                ON DELETE CASCADE
        )
        """,
    )

    _INDEX_STATEMENTS = (
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_turns_recent
        ON conversation_turns (mode, profile_id, session_id, updated_at DESC)
        """,
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_ann_one_active
        ON memory_ann_generations (active) WHERE active = 1
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_semantic_memory
        ON memory_semantic_vectors (memory_id, tombstone)
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_memory_ann_journal_memory
        ON memory_ann_journal (memory_id, sequence)
        """,
    )

    _ANN_TRIGGER_STATEMENTS = (
        """
        CREATE TRIGGER IF NOT EXISTS memories_ann_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memory_ann_journal(memory_id, operation, created_at)
            VALUES (new.id, 'upsert', CAST(strftime('%s', 'now') AS REAL));
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS memories_ann_update
        AFTER UPDATE OF content ON memories BEGIN
            INSERT INTO memory_ann_journal(memory_id, operation, created_at)
            VALUES (new.id, 'upsert', CAST(strftime('%s', 'now') AS REAL));
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS memories_ann_delete AFTER DELETE ON memories BEGIN
            INSERT INTO memory_ann_journal(memory_id, operation, created_at)
            VALUES (old.id, 'delete', CAST(strftime('%s', 'now') AS REAL));
        END
        """,
    )

    _FTS_STATEMENTS = (
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            content,
            tags,
            content='memories',
            content_rowid='id',
            tokenize='trigram'
        )
        """,
        """
        CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
            INSERT INTO memory_fts(rowid, content, tags)
            VALUES (new.id, new.content, new.tags);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
        END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS memories_fts_update
        AFTER UPDATE OF content, tags ON memories BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content, tags)
            VALUES ('delete', old.id, old.content, old.tags);
            INSERT INTO memory_fts(rowid, content, tags)
            VALUES (new.id, new.content, new.tags);
        END
        """,
    )

    _REQUIRED_COLUMNS = {
        "chat_history": {
            "summarized": "INTEGER DEFAULT 0",
        },
        "memories": {
            "source": "TEXT DEFAULT ''",
            "created": "REAL DEFAULT 0",
            "last_recalled": "REAL DEFAULT 0",
            "tags": "TEXT DEFAULT '[]'",
            "metadata": "TEXT DEFAULT '{}'",
            "memory_type": "TEXT DEFAULT 'fact'",
            "embedding": "TEXT DEFAULT ''",
            "decay_factor": "REAL DEFAULT 1.0",
            "updated": "REAL DEFAULT 0",
            "access_count": "INTEGER DEFAULT 1",
            "source_ids": "TEXT DEFAULT '[]'",
            "last_decay": "REAL DEFAULT 0",
        },
        "memory_ann_generations": {
            "index_path": "TEXT NOT NULL DEFAULT ''",
        },
        "conversation_turns": {
            "source": "TEXT NOT NULL DEFAULT 'system'",
            "user_text": "TEXT NOT NULL DEFAULT ''",
            "segments": "TEXT NOT NULL DEFAULT '[]'",
            "system_entries": "TEXT NOT NULL DEFAULT '[]'",
            "created_at": "REAL NOT NULL DEFAULT 0",
            "updated_at": "REAL NOT NULL DEFAULT 0",
            "status": "TEXT NOT NULL DEFAULT 'streaming'",
            "error_text": "TEXT NOT NULL DEFAULT ''",
        },
    }

    _CONVERSATION_KEY_COLUMNS = ("mode", "profile_id", "session_id", "turn_id")
    _CONVERSATION_TABLE = "conversation_turns"

    @staticmethod
    def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row[1]) for row in rows}

    @staticmethod
    def _primary_key_columns(connection: sqlite3.Connection, table: str) -> tuple[str, ...]:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        keyed = sorted(
            ((int(row[5]), str(row[1])) for row in rows if int(row[5]) > 0),
            key=lambda item: item[0],
        )
        return tuple(name for _, name in keyed)

    @contextmanager
    def _atomic(self, connection: sqlite3.Connection) -> Iterator[None]:
        """在独立事务或调用方事务的保存点中执行迁移。"""

        owns_transaction = not connection.in_transaction
        savepoint: str | None = None
        if owns_transaction:
            connection.execute("BEGIN")
        else:
            savepoint = f"meapet_schema_{time.monotonic_ns()}"
            connection.execute(f"SAVEPOINT {savepoint}")
        try:
            yield
        except BaseException:
            if owns_transaction:
                connection.rollback()
            elif savepoint is not None:
                connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")
            raise
        else:
            if owns_transaction:
                connection.commit()
            elif savepoint is not None:
                connection.execute(f"RELEASE SAVEPOINT {savepoint}")

    @staticmethod
    def _create_conversation_table(connection: sqlite3.Connection, table: str) -> None:
        """创建完整会话表；表名只来自本类的固定迁移名称。"""

        if table == "conversation_turns":
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
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    error_text TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (mode, profile_id, session_id, turn_id)
                )
                """
            )
            return
        raise ValueError("unsupported schema table")

    def _repair_conversation_table(self, connection: sqlite3.Connection) -> None:
        """补齐旧会话表，并在复合主键缺失时重建而不丢弃可识别轮次。"""

        table = self._CONVERSATION_TABLE
        columns = self._columns(connection, table)
        missing_keys = [name for name in self._CONVERSATION_KEY_COLUMNS if name not in columns]
        if missing_keys:
            joined = ", ".join(missing_keys)
            raise sqlite3.OperationalError(f"conversation_turns 缺少不可推断的身份列: {joined}")

        primary_key = self._primary_key_columns(connection, table)
        if primary_key == self._CONVERSATION_KEY_COLUMNS:
            return

        connection.execute("DROP INDEX IF EXISTS idx_conversation_turns_recent")
        legacy_table = "conversation_turns_legacy"
        if self._columns(connection, legacy_table):
            raise sqlite3.OperationalError("conversation_turns 迁移残留表已存在")
        connection.execute(f"ALTER TABLE {table} RENAME TO {legacy_table}")
        self._create_conversation_table(connection, table)
        all_columns = (
            "mode",
            "profile_id",
            "session_id",
            "turn_id",
            "source",
            "user_text",
            "segments",
            "system_entries",
            "created_at",
            "updated_at",
            "status",
            "error_text",
        )
        copy_columns = tuple(name for name in all_columns if name in columns)
        if copy_columns:
            names = ", ".join(copy_columns)
            connection.execute(
                f"INSERT OR REPLACE INTO {table} ({names}) SELECT {names} FROM {legacy_table}"
            )
        connection.execute(f"DROP TABLE {legacy_table}")

    @staticmethod
    def _embedding_is_valid(value: object, content: object) -> bool:
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return False
        else:
            parsed = value
        if not isinstance(parsed, list):
            return False
        if not parsed and str(content or "").strip():
            return False
        return all(valid_embedding_item(item) for item in parsed)

    @classmethod
    def _repair_memory_embeddings(
        cls,
        connection: sqlite3.Connection,
        current_version: int,
    ) -> None:
        rows = connection.execute("SELECT id, content, embedding FROM memories").fetchall()
        for row in rows:
            if current_version < 5 or not cls._embedding_is_valid(row["embedding"], row["content"]):
                embedding = compute_embedding(str(row["content"] or ""))
                connection.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (json.dumps(embedding, ensure_ascii=False), row["id"]),
                )
        connection.execute(
            """
            UPDATE memories
            SET last_decay = COALESCE(last_recalled, created, 0)
            WHERE last_decay IS NULL OR last_decay = 0
            """
        )

    def _ensure_memory_fts(self, connection: sqlite3.Connection) -> bool:
        """创建可选 FTS5 词法索引；运行库缺失模块时保持核心迁移可用。"""

        existed = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
        ).fetchone()
        try:
            for statement in self._FTS_STATEMENTS:
                connection.execute(statement)
            if not existed:
                connection.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
        except sqlite3.OperationalError as exc:
            message = str(exc).lower()
            if (
                "no such module: fts5" not in message
                and "error in tokenizer constructor" not in message
                and "no such tokenizer: trigram" not in message
            ):
                raise
            return False
        return True

    def migrate(self, connection: sqlite3.Connection) -> None:
        """幂等创建表并把旧表补齐到 schema v7。"""

        if not isinstance(connection, sqlite3.Connection) and hasattr(connection, "connection"):
            connection = connection.connection
        with self._atomic(connection):
            for statement in self._CREATE_STATEMENTS:
                connection.execute(statement)
            current_row = connection.execute(
                "SELECT MAX(version) FROM memory_schema_version"
            ).fetchone()
            current_version = int(current_row[0] or 0) if current_row else 0
            for table, columns in self._REQUIRED_COLUMNS.items():
                existing = self._columns(connection, table)
                for name, definition in columns.items():
                    if name not in existing:
                        connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            self._repair_conversation_table(connection)
            for statement in self._INDEX_STATEMENTS:
                connection.execute(statement)
            for statement in self._ANN_TRIGGER_STATEMENTS:
                connection.execute(statement)
            self._repair_memory_embeddings(connection, current_version)
            self._ensure_memory_fts(connection)
            connection.execute(
                """
                INSERT OR REPLACE INTO memory_schema_version (version, applied_at)
                VALUES (?, ?)
                """,
                (SCHEMA_VERSION, time.time()),
            )


class Database:
    """拥有单个 SQLite 连接，并串行化事务。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        raw_path = str(path)
        if raw_path != ":memory:":
            file_path = Path(path).expanduser()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path = str(file_path)
        self.path = raw_path
        self._lock = threading.RLock()
        self._savepoint_counter = 0
        self.connection = sqlite3.connect(raw_path, check_same_thread=False)
        self.conn = self.connection
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 3000")
        if raw_path != ":memory:":
            try:
                self.connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.DatabaseError:
                pass
        with self._lock:
            SchemaMigrator().migrate(self.connection)
            self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        defaults = {
            "affection": "5",
            "mood": "平静",
            "mood_updated": str(time.time()),
            "last_chat": str(time.time()),
            "total_chats": "0",
            "total_days": "0",
            "first_met": str(time.time()),
            "nickname": "",
            "master_name": "主人",
            "messages_since_summary": "0",
        }
        self.connection.executemany(
            "INSERT OR IGNORE INTO mea_state (key, value) VALUES (?, ?)",
            defaults.items(),
        )
        self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """执行一个可嵌套使用的原子事务。"""

        with self._lock:
            outermost = not self.connection.in_transaction
            savepoint: str | None = None
            if outermost:
                self.connection.execute("BEGIN IMMEDIATE")
            else:
                self._savepoint_counter += 1
                savepoint = f"meapet_sp_{self._savepoint_counter}"
                self.connection.execute(f"SAVEPOINT {savepoint}")
            try:
                yield self.connection
            except BaseException:
                if outermost:
                    self.connection.rollback()
                elif savepoint is not None:
                    self.connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                    self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise
            else:
                if outermost:
                    self.connection.commit()
                elif savepoint is not None:
                    self.connection.execute(f"RELEASE SAVEPOINT {savepoint}")

    def close(self) -> None:
        """关闭连接；重复关闭保持幂等。"""

        with self._lock:
            try:
                self.connection.close()
            except sqlite3.ProgrammingError:
                pass

    def __enter__(self) -> Database:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()
