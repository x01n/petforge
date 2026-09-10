"""按会话复合键持久化对话轮次。"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from core.conversation.runtime import ConversationKey

from .database import Database


def _json_default(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _json_value(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


@dataclass(frozen=True)
class ConversationTurn:
    """与 `conversation_turns` 表一一对应的稳定数据对象。"""

    mode: str
    profile_id: str
    session_id: str
    turn_id: str
    source: str
    user_text: str = ""
    segments: tuple[dict[str, Any], ...] = ()
    system_entries: tuple[dict[str, Any], ...] = ()
    created_at: float = 0.0
    updated_at: float = 0.0
    status: str = "streaming"
    error_text: str = ""

    def __post_init__(self) -> None:
        values = {
            "mode": str(self.mode or "").strip().lower(),
            "profile_id": str(self.profile_id or "").strip(),
            "session_id": str(self.session_id or "").strip(),
            "turn_id": str(self.turn_id or "").strip(),
            "source": str(self.source or "system").strip() or "system",
            "user_text": str(self.user_text or ""),
            "status": str(self.status or "streaming").strip() or "streaming",
            "error_text": str(self.error_text or ""),
        }
        if values["mode"] not in {"direct", "agent"}:
            raise ValueError("conversation mode is unsupported")
        if not all(values[key] for key in ("mode", "profile_id", "session_id", "turn_id")):
            raise ValueError("conversation turn identifiers are required")
        for key in ("profile_id", "session_id", "turn_id"):
            if len(values[key]) > 256 or any(char in values[key] for char in "\r\n\x00"):
                raise ValueError("conversation identifier is invalid")
        object.__setattr__(self, "mode", values["mode"])
        object.__setattr__(self, "profile_id", values["profile_id"])
        object.__setattr__(self, "session_id", values["session_id"])
        object.__setattr__(self, "turn_id", values["turn_id"])
        object.__setattr__(self, "source", values["source"])
        object.__setattr__(self, "user_text", values["user_text"])
        object.__setattr__(self, "status", values["status"])
        object.__setattr__(self, "error_text", values["error_text"])
        object.__setattr__(self, "segments", self._normalize_entries(self.segments))
        object.__setattr__(self, "system_entries", self._normalize_entries(self.system_entries))
        object.__setattr__(self, "created_at", float(self.created_at or 0.0))
        object.__setattr__(self, "updated_at", float(self.updated_at or 0.0))

    @staticmethod
    def _normalize_entries(value: object) -> tuple[dict[str, Any], ...]:
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ValueError("conversation entries must be a sequence")
        normalized: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                normalized.append(dict(item))
            elif is_dataclass(item):
                converted = asdict(item)
                if not isinstance(converted, dict):
                    raise ValueError("conversation entry must be a mapping")
                normalized.append(converted)
            else:
                raise ValueError("conversation entries must contain mappings")
        return tuple(normalized)

    @property
    def key(self) -> ConversationKey:
        return ConversationKey(self.mode, self.profile_id, self.session_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "source": self.source,
            "user_text": self.user_text,
            "segments": self.segments,
            "system_entries": self.system_entries,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "error_text": self.error_text,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]


class ConversationRepository:
    """使用 `(mode, profile_id, session_id, turn_id)` 完整隔离轮次。"""

    def __init__(self, database: Database) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        self._database = database

    @staticmethod
    def _from_row(row: Any) -> ConversationTurn | None:
        if row is None:
            return None
        try:
            segments = json.loads(row["segments"])
            system_entries = json.loads(row["system_entries"])
            if not isinstance(segments, list) or not isinstance(system_entries, list):
                return None
            return ConversationTurn(
                row["mode"],
                row["profile_id"],
                row["session_id"],
                row["turn_id"],
                row["source"],
                row["user_text"],
                tuple(segments),
                tuple(system_entries),
                row["created_at"],
                row["updated_at"],
                row["status"],
                row["error_text"],
            )
        except (TypeError, ValueError, KeyError, OverflowError, json.JSONDecodeError):
            return None

    def save(self, turn: ConversationTurn) -> None:
        if not isinstance(turn, ConversationTurn):
            raise TypeError("turn must be a ConversationTurn")
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO conversation_turns (
                    mode, profile_id, session_id, turn_id, source, user_text,
                    segments, system_entries, created_at, updated_at, status, error_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mode, profile_id, session_id, turn_id) DO UPDATE SET
                    source = excluded.source,
                    user_text = excluded.user_text,
                    segments = excluded.segments,
                    system_entries = excluded.system_entries,
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at,
                    status = excluded.status,
                    error_text = excluded.error_text
                """,
                (
                    turn.mode,
                    turn.profile_id,
                    turn.session_id,
                    turn.turn_id,
                    turn.source,
                    turn.user_text,
                    _json_value(turn.segments),
                    _json_value(turn.system_entries),
                    turn.created_at,
                    turn.updated_at,
                    turn.status,
                    turn.error_text,
                ),
            )

    def create(
        self,
        *,
        mode: str,
        profile_id: str,
        session_id: str,
        turn_id: str,
        source: str,
        user_text: str = "",
        status: str = "streaming",
    ) -> ConversationTurn:
        now = time.time()
        turn = ConversationTurn(
            mode,
            profile_id,
            session_id,
            turn_id,
            source,
            user_text,
            (),
            (),
            now,
            now,
            status,
        )
        self.save(turn)
        return turn

    def get(
        self,
        mode: str,
        profile_id: str,
        session_id: str,
        turn_id: str,
    ) -> ConversationTurn | None:
        with self._database._lock:
            row = self._database.connection.execute(
                """
                SELECT * FROM conversation_turns
                WHERE mode = ? AND profile_id = ? AND session_id = ? AND turn_id = ?
                """,
                (mode, profile_id, session_id, turn_id),
            ).fetchone()
        return self._from_row(row)

    def list_recent(
        self,
        mode: str,
        profile_id: str,
        session_id: str,
        limit: int = 50,
    ) -> tuple[ConversationTurn, ...]:
        safe_limit = max(0, min(int(limit), 500))
        if safe_limit == 0:
            return ()
        with self._database._lock:
            rows = self._database.connection.execute(
                """
                SELECT * FROM conversation_turns
                WHERE mode = ? AND profile_id = ? AND session_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT ?
                """,
                (mode, profile_id, session_id, safe_limit),
            ).fetchall()
        result: list[ConversationTurn] = []
        for row in reversed(rows):
            turn = self._from_row(row)
            if turn is not None:
                result.append(turn)
        return tuple(result)

    def list_for_key(self, key: ConversationKey, limit: int = 50) -> tuple[ConversationTurn, ...]:
        if not isinstance(key, ConversationKey):
            raise TypeError("key must be a ConversationKey")
        return self.list_recent(key.mode, key.profile_id, key.session_id, limit)

    def delete(self, turn: ConversationTurn | tuple[str, str, str, str]) -> bool:
        if isinstance(turn, ConversationTurn):
            values = (turn.mode, turn.profile_id, turn.session_id, turn.turn_id)
        else:
            if len(turn) != 4:
                raise ValueError("turn key must contain four values")
            values = tuple(str(item) for item in turn)
        with self._database.transaction() as connection:
            cursor = connection.execute(
                """
                DELETE FROM conversation_turns
                WHERE mode = ? AND profile_id = ? AND session_id = ? AND turn_id = ?
                """,
                values,
            )
        return cursor.rowcount > 0

    def clear(self, key: ConversationKey | None = None) -> int:
        with self._database.transaction() as connection:
            if key is None:
                cursor = connection.execute("DELETE FROM conversation_turns")
            else:
                if not isinstance(key, ConversationKey):
                    raise TypeError("key must be a ConversationKey")
                cursor = connection.execute(
                    """
                    DELETE FROM conversation_turns
                    WHERE mode = ? AND profile_id = ? AND session_id = ?
                    """,
                    (key.mode, key.profile_id, key.session_id),
                )
        return cursor.rowcount

    def save_conversation_turn(self, turn: ConversationTurn, *, max_turns: int = 5) -> None:
        """兼容旧调用名称。"""

        self.save(turn)
        limit = max(0, min(int(max_turns), 500))
        if limit == 0:
            self.clear(turn.key)
            return
        with self._database.transaction() as connection:
            connection.execute(
                """
                DELETE FROM conversation_turns
                WHERE mode = ? AND profile_id = ? AND session_id = ?
                  AND turn_id NOT IN (
                    SELECT turn_id FROM conversation_turns
                    WHERE mode = ? AND profile_id = ? AND session_id = ?
                    ORDER BY updated_at DESC, turn_id DESC
                    LIMIT ?
                  )
                """,
                (
                    turn.mode,
                    turn.profile_id,
                    turn.session_id,
                    turn.mode,
                    turn.profile_id,
                    turn.session_id,
                    limit,
                ),
            )

    def load_conversation_turns(
        self,
        key: ConversationKey | None = None,
        *,
        max_total: int = 500,
    ) -> tuple[ConversationTurn, ...]:
        safe_limit = max(0, min(int(max_total), 5000))
        if key is not None:
            return self.list_for_key(key, safe_limit)
        if safe_limit == 0:
            return ()
        with self._database._lock:
            rows = self._database.connection.execute(
                "SELECT * FROM conversation_turns ORDER BY updated_at DESC, turn_id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        result: list[ConversationTurn] = []
        for row in reversed(rows):
            turn = self._from_row(row)
            if turn is not None:
                result.append(turn)
        return tuple(result)
