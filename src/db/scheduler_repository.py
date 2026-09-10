"""SQLite 调度任务、触发器快照和有界执行审计仓储。"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from typing import Any

from .database import Database


def _reject_json_constant(value: str) -> None:
    """拒绝 JSON 扩展常量 NaN/Infinity，避免坏快照进入调度器。"""

    raise ValueError(f"unsupported JSON constant: {value}")


class SchedulerRepository:
    """为调度服务提供可选持久化；表通过幂等 DDL 按需创建。

    该仓储不修改现有 ``memory_schema_version``，因此可独立部署到已有数据库。
    快照和执行记录均只接受 JSON 安全映射，损坏的历史行在读取时跳过。
    """

    _MAX_RUN_ROWS = 2048

    def __init__(self, database: Database, *, max_run_rows: int = _MAX_RUN_ROWS) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        self._database = database
        self._max_run_rows = max(1, min(int(max_run_rows), 100_000))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._database.transaction() as connection:
            statements = (
                """
                CREATE TABLE IF NOT EXISTS scheduler_tasks (
                    task_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    expression TEXT NOT NULL,
                    action TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    next_run_at REAL NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
                    last_run_at REAL NOT NULL DEFAULT 0,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS scheduler_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    event_name TEXT NOT NULL,
                    action TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    debounce_seconds REAL NOT NULL DEFAULT 0,
                    last_fired_at REAL NOT NULL DEFAULT 0,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS scheduler_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    finished_at REAL NOT NULL,
                    status TEXT NOT NULL,
                    error_text TEXT NOT NULL DEFAULT '',
                    payload TEXT NOT NULL DEFAULT '{}'
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_scheduler_runs_recent
                    ON scheduler_runs (finished_at DESC, id DESC);
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_scheduler_runs_item
                    ON scheduler_runs (kind, item_id, finished_at DESC, id DESC);
                """,
            )
            for statement in statements:
                connection.execute(statement)

    @staticmethod
    def _text(value: object, *, field_name: str, max_length: int) -> str:
        result = str(value or "").strip()
        if not result or len(result) > max_length or any(char in result for char in "\r\n\x00"):
            raise ValueError(f"{field_name} is invalid")
        return result

    @staticmethod
    def _number(value: object, *, field_name: str, minimum: float | None = None) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} is invalid") from exc
        if not math.isfinite(result) or (minimum is not None and result < minimum):
            raise ValueError(f"{field_name} is invalid")
        return result

    @staticmethod
    def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_name} must be a mapping")
        return dict(value)

    @classmethod
    def _json(cls, value: Mapping[str, Any], *, field_name: str) -> str:
        try:
            return json.dumps(
                dict(value), ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be JSON serializable") from exc

    @staticmethod
    def _decode_mapping(value: object) -> dict[str, Any] | None:
        try:
            parsed = json.loads(str(value), parse_constant=_reject_json_constant)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        return dict(parsed) if isinstance(parsed, Mapping) else None

    def load_tasks(self) -> tuple[dict[str, Any], ...]:
        rows = self._database.connection.execute(
            """
            SELECT task_id, name, expression, action, owner, next_run_at,
                   enabled, last_run_at, run_count, metadata, created_at, updated_at
            FROM scheduler_tasks ORDER BY next_run_at, task_id
            """
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            action = self._decode_mapping(row["action"])
            metadata = self._decode_mapping(row["metadata"])
            if action is None or metadata is None:
                continue
            try:
                task_id = self._text(row["task_id"], field_name="task_id", max_length=128)
                name = self._text(row["name"], field_name="name", max_length=128)
                expression = self._text(row["expression"], field_name="expression", max_length=128)
                owner = self._text(row["owner"], field_name="owner", max_length=256)
                next_run_at = self._number(row["next_run_at"], field_name="next_run_at")
                last_run_at = self._number(row["last_run_at"], field_name="last_run_at", minimum=0)
                created_at = self._number(row["created_at"], field_name="created_at", minimum=0)
                updated_at = self._number(row["updated_at"], field_name="updated_at", minimum=0)
                run_count = int(row["run_count"])
                if run_count < 0:
                    raise ValueError("run_count is invalid")
                enabled_value = int(row["enabled"])
                if enabled_value not in {0, 1}:
                    raise ValueError("enabled is invalid")
            except (TypeError, ValueError, OverflowError):
                continue
            result.append(
                {
                    "task_id": task_id,
                    "name": name,
                    "expression": expression,
                    "action": action,
                    "owner": owner,
                    "next_run_at": next_run_at,
                    "enabled": bool(enabled_value),
                    "last_run_at": last_run_at,
                    "run_count": run_count,
                    "metadata": metadata,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
        return tuple(result)

    def save_task(self, task: Mapping[str, Any]) -> None:
        task_id = self._text(task.get("task_id"), field_name="task_id", max_length=128)
        name = self._text(task.get("name"), field_name="name", max_length=128)
        expression = self._text(task.get("expression"), field_name="expression", max_length=128)
        owner = self._text(task.get("owner"), field_name="owner", max_length=256)
        action = self._mapping(task.get("action"), field_name="action")
        metadata = self._mapping(task.get("metadata", {}), field_name="metadata")
        next_run_at = self._number(task.get("next_run_at"), field_name="next_run_at")
        last_run_at = self._number(task.get("last_run_at", 0), field_name="last_run_at", minimum=0)
        try:
            run_count = int(task.get("run_count", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("run_count is invalid") from exc
        if run_count < 0:
            raise ValueError("run_count is invalid")
        enabled = 1 if bool(task.get("enabled", True)) else 0
        now = time.time()
        created_at = self._number(task.get("created_at", now), field_name="created_at")
        updated_at = self._number(task.get("updated_at", now), field_name="updated_at")
        action_json = self._json(action, field_name="action")
        metadata_json = self._json(metadata, field_name="metadata")
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO scheduler_tasks (
                    task_id, name, expression, action, owner, next_run_at, enabled,
                    last_run_at, run_count, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    name = excluded.name,
                    expression = excluded.expression,
                    action = excluded.action,
                    owner = excluded.owner,
                    next_run_at = excluded.next_run_at,
                    enabled = excluded.enabled,
                    last_run_at = excluded.last_run_at,
                    run_count = excluded.run_count,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    name,
                    expression,
                    action_json,
                    owner,
                    next_run_at,
                    enabled,
                    last_run_at,
                    run_count,
                    metadata_json,
                    created_at,
                    updated_at,
                ),
            )

    def delete_task(self, task_id: str) -> None:
        safe_id = self._text(task_id, field_name="task_id", max_length=128)
        with self._database.transaction() as connection:
            connection.execute("DELETE FROM scheduler_tasks WHERE task_id = ?", (safe_id,))

    def load_triggers(self) -> tuple[dict[str, Any], ...]:
        rows = self._database.connection.execute(
            """
            SELECT trigger_id, event_name, action, owner, debounce_seconds,
                   last_fired_at, metadata, created_at, updated_at
            FROM scheduler_triggers ORDER BY trigger_id
            """
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            action = self._decode_mapping(row["action"])
            metadata = self._decode_mapping(row["metadata"])
            if action is None or metadata is None:
                continue
            try:
                trigger_id = self._text(row["trigger_id"], field_name="trigger_id", max_length=128)
                event_name = self._text(row["event_name"], field_name="event_name", max_length=128)
                owner = self._text(row["owner"], field_name="owner", max_length=256)
                debounce_seconds = self._number(
                    row["debounce_seconds"], field_name="debounce_seconds", minimum=0
                )
                last_fired_at = self._number(
                    row["last_fired_at"], field_name="last_fired_at", minimum=0
                )
                created_at = self._number(row["created_at"], field_name="created_at", minimum=0)
                updated_at = self._number(row["updated_at"], field_name="updated_at", minimum=0)
            except (TypeError, ValueError, OverflowError):
                continue
            result.append(
                {
                    "trigger_id": trigger_id,
                    "event_name": event_name,
                    "action": action,
                    "owner": owner,
                    "debounce_seconds": debounce_seconds,
                    "last_fired_at": last_fired_at,
                    "metadata": metadata,
                    "created_at": created_at,
                    "updated_at": updated_at,
                }
            )
        return tuple(result)

    def save_trigger(self, trigger: Mapping[str, Any]) -> None:
        trigger_id = self._text(trigger.get("trigger_id"), field_name="trigger_id", max_length=128)
        event_name = self._text(trigger.get("event_name"), field_name="event_name", max_length=128)
        owner = self._text(trigger.get("owner"), field_name="owner", max_length=256)
        action = self._mapping(trigger.get("action"), field_name="action")
        metadata = self._mapping(trigger.get("metadata", {}), field_name="metadata")
        debounce_seconds = self._number(
            trigger.get("debounce_seconds", 0), field_name="debounce_seconds", minimum=0
        )
        if debounce_seconds > 86400:
            raise ValueError("debounce_seconds is invalid")
        last_fired_at = self._number(
            trigger.get("last_fired_at", 0), field_name="last_fired_at", minimum=0
        )
        now = time.time()
        created_at = self._number(trigger.get("created_at", now), field_name="created_at")
        updated_at = self._number(trigger.get("updated_at", now), field_name="updated_at")
        action_json = self._json(action, field_name="action")
        metadata_json = self._json(metadata, field_name="metadata")
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO scheduler_triggers (
                    trigger_id, event_name, action, owner, debounce_seconds,
                    last_fired_at, metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trigger_id) DO UPDATE SET
                    event_name = excluded.event_name,
                    action = excluded.action,
                    owner = excluded.owner,
                    debounce_seconds = excluded.debounce_seconds,
                    last_fired_at = excluded.last_fired_at,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    trigger_id,
                    event_name,
                    action_json,
                    owner,
                    debounce_seconds,
                    last_fired_at,
                    metadata_json,
                    created_at,
                    updated_at,
                ),
            )

    def delete_trigger(self, trigger_id: str) -> None:
        safe_id = self._text(trigger_id, field_name="trigger_id", max_length=128)
        with self._database.transaction() as connection:
            connection.execute("DELETE FROM scheduler_triggers WHERE trigger_id = ?", (safe_id,))

    def record_run(
        self,
        *,
        kind: str,
        item_id: str,
        owner: str,
        started_at: float,
        finished_at: float,
        status: str,
        error_text: str = "",
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        safe_kind = self._text(kind, field_name="kind", max_length=32)
        if safe_kind not in {"task", "trigger"}:
            raise ValueError("kind is invalid")
        safe_item_id = self._text(item_id, field_name="item_id", max_length=128)
        safe_owner = self._text(owner, field_name="owner", max_length=256)
        safe_status = self._text(status, field_name="status", max_length=32)
        start = self._number(started_at, field_name="started_at")
        finish = self._number(finished_at, field_name="finished_at")
        safe_error = str(error_text or "")[:2000]
        payload_mapping = {} if payload is None else self._mapping(payload, field_name="payload")
        payload_json = self._json(payload_mapping, field_name="payload")
        with self._database.transaction() as connection:
            connection.execute(
                """
                INSERT INTO scheduler_runs (
                    kind, item_id, owner, started_at, finished_at, status, error_text, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    safe_kind,
                    safe_item_id,
                    safe_owner,
                    start,
                    finish,
                    safe_status,
                    safe_error,
                    payload_json,
                ),
            )
            connection.execute(
                """
                DELETE FROM scheduler_runs
                WHERE id NOT IN (
                    SELECT id FROM scheduler_runs ORDER BY id DESC LIMIT ?
                )
                """,
                (self._max_run_rows,),
            )

    def list_runs(
        self,
        *,
        limit: int = 100,
        kind: str | None = None,
        item_id: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        safe_limit = max(1, min(int(limit), self._max_run_rows))
        clauses: list[str] = []
        parameters: list[object] = []
        if kind is not None:
            safe_kind = self._text(kind, field_name="kind", max_length=32)
            if safe_kind not in {"task", "trigger"}:
                raise ValueError("kind is invalid")
            clauses.append("kind = ?")
            parameters.append(safe_kind)
        if item_id is not None:
            clauses.append("item_id = ?")
            parameters.append(self._text(item_id, field_name="item_id", max_length=128))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._database.connection.execute(
            f"""
            SELECT id, kind, item_id, owner, started_at, finished_at,
                   status, error_text, payload
            FROM scheduler_runs {where}
            ORDER BY id DESC LIMIT ?
            """,
            (*parameters, safe_limit),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = self._decode_mapping(row["payload"])
            if payload is None:
                payload = {}
            result.append(
                {
                    "id": int(row["id"]),
                    "kind": str(row["kind"]),
                    "item_id": str(row["item_id"]),
                    "owner": str(row["owner"]),
                    "started_at": float(row["started_at"]),
                    "finished_at": float(row["finished_at"]),
                    "status": str(row["status"]),
                    "error_text": str(row["error_text"] or ""),
                    "payload": payload,
                }
            )
        return tuple(result)


__all__ = ["SchedulerRepository"]
