"""API 调用的可查询审计记录。

该仓储把一次模型请求的生命周期压缩成一条稳定记录：请求开始时写入输入与渠道，
流式响应出现首个 token 时补齐首字耗时，结束或失败时补齐输出、缓存、工具和总耗时。
写入接口保留同步版本供线程/后台任务使用，同时提供 ``asyncio.to_thread`` 包装，
避免事件循环直接执行 SQLite 写入。
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any

from core.json_values import validate_json_value
from logger.events import sanitize_log_text

from .database import Database

API_CALL_AUDIT_SCHEMA_VERSION = 2
MAX_AUDIT_IDENTIFIER_LENGTH = 256
MAX_AUDIT_LABEL_LENGTH = 128
MAX_AUDIT_TEXT_LENGTH = 1_000_000
MAX_AUDIT_JSON_LENGTH = 4_000_000
MAX_AUDIT_LIST_ITEMS = 4096

# 行数上限：与 prune 的默认 max_records 一致。每次存储写操作后自增
# 本地计数器，每满 100 次写执行一次修剪，把过期审计收敛到该上限。
AUDIT_MAX_ROWS = 10_000
AUDIT_PRUNE_INTERVAL = 100

# summary 被状态轮询与控制台刷新高频调用（同参同秒多拍）；在此窗口内
# 缓存聚合结果，把每拍全表 GROUP BY 收敛为每秒一次；写入路径立即失效。
SUMMARY_CACHE_TTL_SECONDS = 1.0


def sanitize_audit_value(value: object, *, max_text_length: int = 32_768) -> object:
    """保留审计结构，同时遮蔽凭据并限制文本、二进制和集合大小。"""

    secret_parts = (
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
    )

    def visit(item: object, key: str = "", depth: int = 0) -> object:
        normalized = key.strip().lower().replace("-", "_")
        token_secret = normalized in {
            "token",
            "access_token",
            "refresh_token",
            "id_token",
            "auth_token",
            "bearer_token",
        } or (normalized.endswith("_token") and not normalized.endswith("_tokens"))
        if normalized and (any(part in normalized for part in secret_parts) or token_secret):
            return "[redacted]"
        if depth >= 8:
            return "[depth-limited]"
        if isinstance(item, (bytes, bytearray, memoryview)):
            return {"kind": "bytes", "size": len(item)}
        if isinstance(item, Mapping):
            return {
                raw_key: visit(nested, str(raw_key), depth + 1)
                for raw_key, nested in list(item.items())[:MAX_AUDIT_LIST_ITEMS]
            }
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return [visit(nested, key, depth + 1) for nested in item[:MAX_AUDIT_LIST_ITEMS]]
        if isinstance(item, str):
            limit = max(0, min(int(max_text_length), MAX_AUDIT_TEXT_LENGTH))
            return sanitize_log_text(item, limit=limit)
        if item is None or isinstance(item, (bool, int, float)):
            return item
        return str(item)[: max(0, min(int(max_text_length), MAX_AUDIT_TEXT_LENGTH))]

    return visit(value)


def _json_ready(value: object, *, label: str) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value), label=label)
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} object keys must be strings")
            result[key] = _json_ready(nested, label=label)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(nested, label=label) for nested in value]
    return value


def _json_value(value: object, *, label: str, maximum: int = MAX_AUDIT_JSON_LENGTH) -> str:
    try:
        value = _json_ready(value, label=label)
        validate_json_value(value, label=label)
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be JSON serializable") from exc
    if len(rendered.encode("utf-8")) > maximum:
        raise ValueError(f"{label} exceeds the audit size limit")
    return rendered


def _decode_json(value: object, *, label: str) -> object:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"stored {label} is invalid JSON") from exc
    return parsed


def _identifier(value: object, *, field_name: str, required: bool = False) -> str:
    rendered = str(value or "").strip()
    if required and not rendered:
        raise ValueError(f"{field_name} is required")
    if len(rendered) > MAX_AUDIT_IDENTIFIER_LENGTH or any(
        character in rendered for character in "\r\n\x00"
    ):
        raise ValueError(f"{field_name} is invalid")
    return rendered


def _label(value: object, *, field_name: str) -> str:
    rendered = _identifier(value, field_name=field_name)
    if len(rendered) > MAX_AUDIT_LABEL_LENGTH:
        raise ValueError(f"{field_name} is too long")
    return rendered


def _finite_number(value: object, *, field_name: str, minimum: float = 0.0) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return result


def _optional_number(value: object, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, field_name=field_name)


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return result


def _optional_non_negative_int(value: object, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return result


def _mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} keys must be strings")
    result = dict(value)
    if any(not key for key in result):
        raise ValueError(f"{field_name} contains an empty key")
    return result


def _mapping_sequence(value: object, *, field_name: str) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{field_name} must be a sequence")
    if len(value) > MAX_AUDIT_LIST_ITEMS:
        raise ValueError(f"{field_name} contains too many items")
    result: list[dict[str, Any]] = []
    for item in value:
        result.append(_mapping(item, field_name=field_name))
    return tuple(result)


def _text(value: object, *, field_name: str) -> str:
    rendered = str(value or "")
    if "\x00" in rendered or len(rendered.encode("utf-8")) > MAX_AUDIT_TEXT_LENGTH:
        raise ValueError(f"{field_name} exceeds the audit size limit")
    limit = min(len(rendered), 16_384)
    pieces = re.split(r"(\s+)", rendered)
    sanitized = "".join(
        piece if piece.isspace() else sanitize_log_text(piece, limit=limit) for piece in pieces
    )
    return sanitized[:limit]


@dataclass(frozen=True, slots=True)
class ApiCallAuditRecord:
    """一条完整的模型/API 调用审计记录。"""

    request_id: str
    id: int | None = None
    mode: str = ""
    profile_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    generation_id: int = 0
    attempt: int = 1
    kind: str = "model"
    operation: str = "model.stream"
    status: str = "started"
    started_at: float = 0.0
    first_token_at: float | None = None
    first_char: str = ""
    completed_at: float | None = None
    total_duration_ms: float | None = None
    time_to_first_token_ms: float | None = None
    provider: str = ""
    protocol: str = ""
    channel_id: str = ""
    channel_name: str = ""
    channel_info: Mapping[str, Any] = field(default_factory=dict)
    requested_model: str = ""
    response_model: str = ""
    input_payload: Any = field(default_factory=dict)
    output_text: str = ""
    output_payload: Any = field(default_factory=dict)
    usage: Mapping[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cache_read_duration_ms: float | None = None
    cache_write_duration_ms: float | None = None
    cache_info: Mapping[str, Any] = field(default_factory=dict)
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    tool_executions: tuple[Mapping[str, Any], ...] = ()
    error_type: str = ""
    error_message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _identifier(self.request_id, field_name="request_id", required=True),
        )
        for field_name in ("mode", "profile_id", "session_id", "turn_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("provider", "protocol", "channel_id", "channel_name"):
            object.__setattr__(
                self,
                field_name,
                _label(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(self, "status", _label(self.status, field_name="status") or "started")
        object.__setattr__(self, "attempt", _positive_int(self.attempt, field_name="attempt"))
        object.__setattr__(self, "kind", _label(self.kind, field_name="kind") or "model")
        object.__setattr__(
            self,
            "operation",
            _label(self.operation, field_name="operation") or "model.stream",
        )
        generation = _optional_non_negative_int(self.generation_id, field_name="generation_id")
        object.__setattr__(self, "generation_id", generation or 0)
        started_at = _finite_number(self.started_at, field_name="started_at")
        object.__setattr__(self, "started_at", started_at)
        for field_name in ("first_token_at", "completed_at"):
            value = getattr(self, field_name)
            if value is not None:
                value = _finite_number(value, field_name=field_name)
                if value < started_at:
                    raise ValueError(f"{field_name} cannot precede started_at")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "total_duration_ms",
            "time_to_first_token_ms",
            "cache_read_duration_ms",
            "cache_write_duration_ms",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_number(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "channel_info",
            _mapping(sanitize_audit_value(self.channel_info), field_name="channel_info"),
        )
        object.__setattr__(
            self,
            "usage",
            _mapping(sanitize_audit_value(self.usage), field_name="usage"),
        )
        object.__setattr__(
            self,
            "cache_info",
            _mapping(sanitize_audit_value(self.cache_info), field_name="cache_info"),
        )
        object.__setattr__(
            self,
            "metadata",
            _mapping(sanitize_audit_value(self.metadata), field_name="metadata"),
        )
        object.__setattr__(
            self,
            "tool_calls",
            _mapping_sequence(sanitize_audit_value(self.tool_calls), field_name="tool_calls"),
        )
        object.__setattr__(
            self,
            "tool_executions",
            _mapping_sequence(
                sanitize_audit_value(self.tool_executions), field_name="tool_executions"
            ),
        )
        object.__setattr__(
            self,
            "requested_model",
            _label(self.requested_model, field_name="requested_model"),
        )
        object.__setattr__(
            self,
            "response_model",
            _label(self.response_model, field_name="response_model"),
        )
        object.__setattr__(
            self,
            "finish_reason",
            _label(self.finish_reason, field_name="finish_reason"),
        )
        object.__setattr__(self, "error_type", _label(self.error_type, field_name="error_type"))
        object.__setattr__(self, "output_text", _text(self.output_text, field_name="output_text"))
        object.__setattr__(self, "first_char", _text(self.first_char, field_name="first_char")[:1])
        object.__setattr__(
            self, "error_message", _text(self.error_message, field_name="error_message")
        )
        for field_name in ("cache_read_tokens", "cache_write_tokens"):
            object.__setattr__(
                self,
                field_name,
                _optional_non_negative_int(getattr(self, field_name), field_name=field_name),
            )
        if self.id is not None:
            object.__setattr__(self, "id", _positive_int(self.id, field_name="id"))
        created_at = _finite_number(self.created_at or started_at, field_name="created_at")
        updated_at = _finite_number(self.updated_at or created_at, field_name="updated_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "input_payload", sanitize_audit_value(self.input_payload))
        object.__setattr__(self, "output_payload", sanitize_audit_value(self.output_payload))
        _json_value(self.input_payload, label="input_payload")
        _json_value(self.output_payload, label="output_payload")
        _json_value(self.channel_info, label="channel_info")
        _json_value(self.usage, label="usage")
        _json_value(self.cache_info, label="cache_info")
        _json_value(self.tool_calls, label="tool_calls")
        _json_value(self.tool_executions, label="tool_executions")
        _json_value(self.metadata, label="metadata")

    @property
    def model_id(self) -> str:
        """返回供应商最终响应的模型 ID；没有响应 ID 时返回请求模型。"""

        return self.response_model or self.requested_model

    @property
    def time_to_first_char_ms(self) -> float | None:
        """兼容“首字耗时”命名；流式文本首事件即首字计时。"""

        return self.time_to_first_token_ms

    @property
    def cache_read(self) -> bool:
        """是否存在缓存读取 usage 或明确的读取标记。"""

        return self.cache_read_tokens is not None or self.cache_info.get("read") is True

    @property
    def cache_write(self) -> bool:
        """是否存在缓存写入 usage 或明确的写入标记。"""

        return self.cache_write_tokens is not None or self.cache_info.get("write") is True

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def tool_execution_count(self) -> int:
        return len(self.tool_executions)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "request_id": self.request_id,
            "mode": self.mode,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "generation_id": self.generation_id,
            "attempt": self.attempt,
            "kind": self.kind,
            "operation": self.operation,
            "status": self.status,
            "started_at": self.started_at,
            "first_token_at": self.first_token_at,
            "first_char": self.first_char,
            "completed_at": self.completed_at,
            "total_duration_ms": self.total_duration_ms,
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "time_to_first_char_ms": self.time_to_first_char_ms,
            "provider": self.provider,
            "protocol": self.protocol,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "channel_info": dict(self.channel_info),
            "requested_model": self.requested_model,
            "response_model": self.response_model,
            "model_id": self.model_id,
            "input_payload": self.input_payload,
            "output_text": self.output_text,
            "output_payload": self.output_payload,
            "usage": dict(self.usage),
            "finish_reason": self.finish_reason,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "cache_read": self.cache_read,
            "cache_write": self.cache_write,
            "cache_read_duration_ms": self.cache_read_duration_ms,
            "cache_write_duration_ms": self.cache_write_duration_ms,
            "cache_info": dict(self.cache_info),
            "tool_calls": self.tool_calls,
            "tool_executions": self.tool_executions,
            "tool_call_count": self.tool_call_count,
            "tool_execution_count": self.tool_execution_count,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ApiCallAuditHandle:
    """一次流式 API 调用的增量更新句柄。"""

    def __init__(self, repository: ApiCallAuditRepository, record: ApiCallAuditRecord) -> None:
        self._repository = repository
        self._record = record
        self._lock = threading.RLock()

    @property
    def record(self) -> ApiCallAuditRecord:
        with self._lock:
            return self._record

    @property
    def record_id(self) -> int:
        record_id = self.record.id
        if record_id is None:
            raise RuntimeError("audit record has not been persisted")
        return record_id

    def update(self, **changes: Any) -> ApiCallAuditRecord:
        with self._lock:
            updated = self._repository.update(self.record_id, **changes)
            self._record = updated
            return updated

    async def aupdate(self, **changes: Any) -> ApiCallAuditRecord:
        return await asyncio.to_thread(self.update, **changes)

    def mark_first_token(
        self, *, occurred_at: float | None = None, first_char: object = ""
    ) -> ApiCallAuditRecord:
        with self._lock:
            # 终态之后可能仍会收到迟到的供应商事件；不能让它改写首字时间。
            if self._record.first_token_at is not None or self._record.completed_at is not None:
                return self._record
            timestamp = self._repository.now() if occurred_at is None else occurred_at
            duration = max(0.0, (float(timestamp) - self._record.started_at) * 1000.0)
            return self.update(
                first_token_at=timestamp,
                time_to_first_token_ms=duration,
                first_char=str(first_char or "")[:1],
            )

    async def amark_first_token(
        self, *, occurred_at: float | None = None, first_char: object = ""
    ) -> ApiCallAuditRecord:
        return await asyncio.to_thread(
            self.mark_first_token, occurred_at=occurred_at, first_char=first_char
        )

    def append_output(self, delta: object) -> ApiCallAuditRecord:
        with self._lock:
            if self._record.completed_at is not None:
                return self._record
            return self.update(
                output_text=self._record.output_text + _text(delta, field_name="output_delta")
            )

    async def aappend_output(self, delta: object) -> ApiCallAuditRecord:
        return await asyncio.to_thread(self.append_output, delta)

    def add_tool_call(self, detail: Mapping[str, Any]) -> ApiCallAuditRecord:
        with self._lock:
            calls = tuple(self._record.tool_calls) + (_mapping(detail, field_name="tool_call"),)
            return self.update(tool_calls=calls)

    async def aadd_tool_call(self, detail: Mapping[str, Any]) -> ApiCallAuditRecord:
        return await asyncio.to_thread(self.add_tool_call, detail)

    def add_tool_execution(self, detail: Mapping[str, Any]) -> ApiCallAuditRecord:
        with self._lock:
            executions = tuple(self._record.tool_executions) + (
                _mapping(detail, field_name="tool_execution"),
            )
            return self.update(tool_executions=executions)

    async def aadd_tool_execution(self, detail: Mapping[str, Any]) -> ApiCallAuditRecord:
        return await asyncio.to_thread(self.add_tool_execution, detail)

    def finish(
        self,
        *,
        status: str = "completed",
        completed_at: float | None = None,
        output_text: object | None = None,
        output_payload: object | None = None,
        response_model: object | None = None,
        usage: Mapping[str, Any] | None = None,
        finish_reason: object | None = None,
        cache_read_tokens: object | None = None,
        cache_write_tokens: object | None = None,
        cache_read_duration_ms: object | None = None,
        cache_write_duration_ms: object | None = None,
        cache_info: Mapping[str, Any] | None = None,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
    ) -> ApiCallAuditRecord:
        with self._lock:
            # ``stream`` 的取消、异常和收尾路径可能在边界上同时到达。
            # 一次持久化终态只能由第一次收尾决定，后续路径保持原记录。
            if self._record.completed_at is not None:
                return self._record
            ended = self._repository.now() if completed_at is None else completed_at
            total = max(0.0, (float(ended) - self._record.started_at) * 1000.0)
            changes: dict[str, Any] = {
                "status": status,
                "completed_at": ended,
                "total_duration_ms": total,
            }
            if output_text is not None:
                changes["output_text"] = output_text
            if output_payload is not None:
                changes["output_payload"] = output_payload
            if response_model is not None:
                changes["response_model"] = response_model
            if usage is not None:
                changes["usage"] = usage
            if finish_reason is not None:
                changes["finish_reason"] = finish_reason
            for key, value in {
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
                "cache_read_duration_ms": cache_read_duration_ms,
                "cache_write_duration_ms": cache_write_duration_ms,
            }.items():
                if value is not None:
                    changes[key] = value
            if cache_info is not None:
                changes["cache_info"] = cache_info
            if tool_calls is not None:
                changes["tool_calls"] = tool_calls
            return self.update(**changes)

    async def afinish(self, **kwargs: Any) -> ApiCallAuditRecord:
        return await asyncio.to_thread(self.finish, **kwargs)

    def fail(
        self,
        *,
        error_type: object = "unknown",
        error_message: object = "",
        status: str = "failed",
        completed_at: float | None = None,
    ) -> ApiCallAuditRecord:
        with self._lock:
            if self._record.completed_at is not None:
                return self._record
            ended = self._repository.now() if completed_at is None else completed_at
            total = max(0.0, (float(ended) - self._record.started_at) * 1000.0)
            return self.update(
                status=status,
                error_type=error_type,
                error_message=error_message,
                completed_at=ended,
                total_duration_ms=total,
            )

    async def afail(self, **kwargs: Any) -> ApiCallAuditRecord:
        return await asyncio.to_thread(self.fail, **kwargs)


class ApiCallAuditRepository:
    """统一 API 调用审计仓储，和现有 ``Database`` 共用连接与事务锁。"""

    _TABLE = "api_call_audits"
    _REQUIRED_COLUMNS = frozenset(
        {
            "id",
            "request_id",
            "mode",
            "profile_id",
            "session_id",
            "turn_id",
            "generation_id",
            "attempt",
            "kind",
            "operation",
            "status",
            "started_at",
            "first_token_at",
            "first_char",
            "completed_at",
            "total_duration_ms",
            "time_to_first_token_ms",
            "provider",
            "protocol",
            "channel_id",
            "channel_name",
            "channel_info",
            "requested_model",
            "response_model",
            "input_payload",
            "output_text",
            "output_payload",
            "usage",
            "finish_reason",
            "cache_read_tokens",
            "cache_write_tokens",
            "cache_read_duration_ms",
            "cache_write_duration_ms",
            "cache_info",
            "tool_calls",
            "tool_executions",
            "error_type",
            "error_message",
            "metadata",
            "created_at",
            "updated_at",
        }
    )
    _COLUMN_DEFINITIONS = {
        "mode": "TEXT NOT NULL DEFAULT ''",
        "profile_id": "TEXT NOT NULL DEFAULT ''",
        "session_id": "TEXT NOT NULL DEFAULT ''",
        "turn_id": "TEXT NOT NULL DEFAULT ''",
        "generation_id": "INTEGER NOT NULL DEFAULT 0",
        "attempt": "INTEGER NOT NULL DEFAULT 1",
        "kind": "TEXT NOT NULL DEFAULT 'model'",
        "operation": "TEXT NOT NULL DEFAULT 'model.stream'",
        "status": "TEXT NOT NULL DEFAULT 'started'",
        "started_at": "REAL NOT NULL DEFAULT 0",
        "first_token_at": "REAL",
        "first_char": "TEXT NOT NULL DEFAULT ''",
        "completed_at": "REAL",
        "total_duration_ms": "REAL",
        "time_to_first_token_ms": "REAL",
        "provider": "TEXT NOT NULL DEFAULT ''",
        "protocol": "TEXT NOT NULL DEFAULT ''",
        "channel_id": "TEXT NOT NULL DEFAULT ''",
        "channel_name": "TEXT NOT NULL DEFAULT ''",
        "channel_info": "TEXT NOT NULL DEFAULT '{}'",
        "requested_model": "TEXT NOT NULL DEFAULT ''",
        "response_model": "TEXT NOT NULL DEFAULT ''",
        "input_payload": "TEXT NOT NULL DEFAULT '{}'",
        "output_text": "TEXT NOT NULL DEFAULT ''",
        "output_payload": "TEXT NOT NULL DEFAULT '{}'",
        "usage": "TEXT NOT NULL DEFAULT '{}'",
        "finish_reason": "TEXT NOT NULL DEFAULT ''",
        "cache_read_tokens": "INTEGER",
        "cache_write_tokens": "INTEGER",
        "cache_read_duration_ms": "REAL",
        "cache_write_duration_ms": "REAL",
        "cache_info": "TEXT NOT NULL DEFAULT '{}'",
        "tool_calls": "TEXT NOT NULL DEFAULT '[]'",
        "tool_executions": "TEXT NOT NULL DEFAULT '[]'",
        "error_type": "TEXT NOT NULL DEFAULT ''",
        "error_message": "TEXT NOT NULL DEFAULT ''",
        "metadata": "TEXT NOT NULL DEFAULT '{}'",
        "created_at": "REAL NOT NULL DEFAULT 0",
        "updated_at": "REAL NOT NULL DEFAULT 0",
    }

    def __init__(self, database: Database, *, clock: Any = time.time) -> None:
        if not isinstance(database, Database):
            raise TypeError("database must be a Database")
        self._database = database
        self._clock = clock
        self._write_count = 0
        self._prune_lock = threading.Lock()
        self._summary_cache: tuple[tuple[object, object], dict[str, object]] | None = None
        self._ensure_schema()

    def now(self) -> float:
        return _finite_number(self._clock(), field_name="audit clock")

    @staticmethod
    def _table_columns(connection: Any, table: str) -> frozenset[str]:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return frozenset(str(row[1]) for row in rows)

    @staticmethod
    def _table_primary_key(connection: Any, table: str) -> tuple[str, ...]:
        rows = connection.execute(f"PRAGMA table_info({table})").fetchall()
        keyed = sorted(
            ((int(row[5]), str(row[1])) for row in rows if int(row[5]) > 0),
            key=lambda item: item[0],
        )
        return tuple(name for _, name in keyed)

    def _ensure_schema(self) -> None:
        with self._database.transaction() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS api_call_audit_schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at REAL NOT NULL
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS api_call_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT '',
                    profile_id TEXT NOT NULL DEFAULT '',
                    session_id TEXT NOT NULL DEFAULT '',
                    turn_id TEXT NOT NULL DEFAULT '',
                    generation_id INTEGER NOT NULL DEFAULT 0,
                    attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
                    kind TEXT NOT NULL DEFAULT 'model',
                    operation TEXT NOT NULL DEFAULT 'model.stream',
                    status TEXT NOT NULL DEFAULT 'started',
                    started_at REAL NOT NULL,
                    first_token_at REAL,
                    first_char TEXT NOT NULL DEFAULT '',
                    completed_at REAL,
                    total_duration_ms REAL,
                    time_to_first_token_ms REAL,
                    provider TEXT NOT NULL DEFAULT '',
                    protocol TEXT NOT NULL DEFAULT '',
                    channel_id TEXT NOT NULL DEFAULT '',
                    channel_name TEXT NOT NULL DEFAULT '',
                    channel_info TEXT NOT NULL DEFAULT '{}',
                    requested_model TEXT NOT NULL DEFAULT '',
                    response_model TEXT NOT NULL DEFAULT '',
                    input_payload TEXT NOT NULL DEFAULT '{}',
                    output_text TEXT NOT NULL DEFAULT '',
                    output_payload TEXT NOT NULL DEFAULT '{}',
                    usage TEXT NOT NULL DEFAULT '{}',
                    finish_reason TEXT NOT NULL DEFAULT '',
                    cache_read_tokens INTEGER,
                    cache_write_tokens INTEGER,
                    cache_read_duration_ms REAL,
                    cache_write_duration_ms REAL,
                    cache_info TEXT NOT NULL DEFAULT '{}',
                    tool_calls TEXT NOT NULL DEFAULT '[]',
                    tool_executions TEXT NOT NULL DEFAULT '[]',
                    error_type TEXT NOT NULL DEFAULT '',
                    error_message TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """)
            columns = self._table_columns(connection, self._TABLE)
            missing = self._REQUIRED_COLUMNS - columns
            if missing:
                for name in sorted(missing):
                    definition = self._COLUMN_DEFINITIONS.get(name)
                    if definition is None:
                        raise RuntimeError(f"{self._TABLE} 缺少不可推断的列: {name}")
                    connection.execute(f"ALTER TABLE {self._TABLE} ADD COLUMN {name} {definition}")
            if self._table_primary_key(connection, self._TABLE) != ("id",):
                raise RuntimeError(f"{self._TABLE} 主键结构不兼容")
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_call_audits_recent
                    ON api_call_audits (started_at DESC, id DESC)
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_call_audits_request
                    ON api_call_audits (request_id, started_at DESC, id DESC)
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_call_audits_context
                    ON api_call_audits (profile_id, session_id, started_at DESC, id DESC)
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_call_audits_channel_model
                    ON api_call_audits (channel_id, response_model, started_at DESC, id DESC)
                """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_call_audits_kind_operation
                    ON api_call_audits (kind, operation, started_at DESC, id DESC)
                """)
            connection.execute(
                """
                INSERT OR REPLACE INTO api_call_audit_schema_version (version, applied_at)
                VALUES (?, ?)
                """,
                (API_CALL_AUDIT_SCHEMA_VERSION, self.now()),
            )

    # 与 _encode_record 返回值顺序一一对应的列名清单。
    _ENCODE_COLUMNS = (
        "request_id",
        "mode",
        "profile_id",
        "session_id",
        "turn_id",
        "generation_id",
        "attempt",
        "kind",
        "operation",
        "status",
        "started_at",
        "first_token_at",
        "first_char",
        "completed_at",
        "total_duration_ms",
        "time_to_first_token_ms",
        "provider",
        "protocol",
        "channel_id",
        "channel_name",
        "channel_info",
        "requested_model",
        "response_model",
        "input_payload",
        "output_text",
        "output_payload",
        "usage",
        "finish_reason",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_read_duration_ms",
        "cache_write_duration_ms",
        "cache_info",
        "tool_calls",
        "tool_executions",
        "error_type",
        "error_message",
        "metadata",
        "created_at",
        "updated_at",
    )

    @staticmethod
    def _encode_record(record: ApiCallAuditRecord) -> tuple[Any, ...]:
        return (
            record.request_id,
            record.mode,
            record.profile_id,
            record.session_id,
            record.turn_id,
            record.generation_id,
            record.attempt,
            record.kind,
            record.operation,
            record.status,
            record.started_at,
            record.first_token_at,
            record.first_char,
            record.completed_at,
            record.total_duration_ms,
            record.time_to_first_token_ms,
            record.provider,
            record.protocol,
            record.channel_id,
            record.channel_name,
            _json_value(record.channel_info, label="channel_info"),
            record.requested_model,
            record.response_model,
            _json_value(record.input_payload, label="input_payload"),
            record.output_text,
            _json_value(record.output_payload, label="output_payload"),
            _json_value(record.usage, label="usage"),
            record.finish_reason,
            record.cache_read_tokens,
            record.cache_write_tokens,
            record.cache_read_duration_ms,
            record.cache_write_duration_ms,
            _json_value(record.cache_info, label="cache_info"),
            _json_value(record.tool_calls, label="tool_calls"),
            _json_value(record.tool_executions, label="tool_executions"),
            record.error_type,
            record.error_message,
            _json_value(record.metadata, label="metadata"),
            record.created_at,
            record.updated_at,
        )

    @staticmethod
    def _from_row(row: Any) -> ApiCallAuditRecord:
        def sequence(field_name: str) -> tuple[dict[str, Any], ...]:
            value = _decode_json(row[field_name], label=field_name)
            return _mapping_sequence(value, field_name=field_name)

        return ApiCallAuditRecord(
            request_id=row["request_id"],
            id=row["id"],
            mode=row["mode"],
            profile_id=row["profile_id"],
            session_id=row["session_id"],
            turn_id=row["turn_id"],
            generation_id=row["generation_id"],
            attempt=row["attempt"],
            kind=row["kind"],
            operation=row["operation"],
            status=row["status"],
            started_at=row["started_at"],
            first_token_at=row["first_token_at"],
            first_char=row["first_char"],
            completed_at=row["completed_at"],
            total_duration_ms=row["total_duration_ms"],
            time_to_first_token_ms=row["time_to_first_token_ms"],
            provider=row["provider"],
            protocol=row["protocol"],
            channel_id=row["channel_id"],
            channel_name=row["channel_name"],
            channel_info=_mapping(
                _decode_json(row["channel_info"], label="channel_info"),
                field_name="channel_info",
            ),
            requested_model=row["requested_model"],
            response_model=row["response_model"],
            input_payload=_decode_json(row["input_payload"], label="input_payload"),
            output_text=row["output_text"],
            output_payload=_decode_json(row["output_payload"], label="output_payload"),
            usage=_mapping(_decode_json(row["usage"], label="usage"), field_name="usage"),
            finish_reason=row["finish_reason"],
            cache_read_tokens=row["cache_read_tokens"],
            cache_write_tokens=row["cache_write_tokens"],
            cache_read_duration_ms=row["cache_read_duration_ms"],
            cache_write_duration_ms=row["cache_write_duration_ms"],
            cache_info=_mapping(
                _decode_json(row["cache_info"], label="cache_info"),
                field_name="cache_info",
            ),
            tool_calls=sequence("tool_calls"),
            tool_executions=sequence("tool_executions"),
            error_type=row["error_type"],
            error_message=row["error_message"],
            metadata=_mapping(
                _decode_json(row["metadata"], label="metadata"), field_name="metadata"
            ),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save(self, record: ApiCallAuditRecord) -> ApiCallAuditRecord:
        if not isinstance(record, ApiCallAuditRecord):
            raise TypeError("record must be an ApiCallAuditRecord")
        now = self.now()
        record = replace(record, created_at=record.created_at or now, updated_at=now)
        values = self._encode_record(record)
        columns = ", ".join(self._ENCODE_COLUMNS)
        placeholders = ", ".join("?" for _ in range(len(self._ENCODE_COLUMNS)))
        with self._database.transaction() as connection:
            cursor = connection.execute(
                f"INSERT INTO {self._TABLE} ({columns}) VALUES ({placeholders})", values
            )
            if cursor.lastrowid is None:
                raise RuntimeError("audit record insert did not return an id")
            record = replace(record, id=int(cursor.lastrowid))
        self._invalidate_summary_cache()
        self._maybe_prune()
        return record

    def _maybe_prune(self) -> None:
        """按写入次数抽样执行自动修剪，把审计表收敛到 AUDIT_MAX_ROWS。"""

        with self._prune_lock:
            self._write_count += 1
            if self._write_count < AUDIT_PRUNE_INTERVAL:
                return
            self._write_count = 0
        try:
            self.prune(max_records=AUDIT_MAX_ROWS)
        except (sqlite3.DatabaseError, OSError, RuntimeError, TypeError, ValueError):
            # 审计写入主路径不能让修剪失败回滚或中断业务调用。
            pass

    async def asave(self, record: ApiCallAuditRecord) -> ApiCallAuditRecord:
        return await asyncio.to_thread(self.save, record)

    def start(self, **kwargs: Any) -> ApiCallAuditHandle:
        now = self.now()
        request_id = str(kwargs.pop("request_id", "") or uuid.uuid4().hex)
        record = ApiCallAuditRecord(
            request_id=request_id,
            started_at=kwargs.pop("started_at", now),
            created_at=now,
            updated_at=now,
            **kwargs,
        )
        return ApiCallAuditHandle(self, self.save(record))

    async def astart(self, **kwargs: Any) -> ApiCallAuditHandle:
        return await asyncio.to_thread(self.start, **kwargs)

    def get(self, record_id: object) -> ApiCallAuditRecord | None:
        safe_id = _positive_int(record_id, field_name="record_id")
        with self._database._lock:
            row = self._database.connection.execute(
                f"SELECT * FROM {self._TABLE} WHERE id = ?", (safe_id,)
            ).fetchone()
        return None if row is None else self._from_row(row)

    async def aget(self, record_id: object) -> ApiCallAuditRecord | None:
        return await asyncio.to_thread(self.get, record_id)

    def update(self, record_id: object, **changes: Any) -> ApiCallAuditRecord:
        current = self.get(record_id)
        if current is None:
            raise KeyError(f"audit record not found: {record_id}")
        changed = set(changes)
        allowed = set(ApiCallAuditRecord.__dataclass_fields__) - {
            "id",
            "created_at",
            "updated_at",
        }
        unknown = changed - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported audit fields: {names}")
        self._invalidate_summary_cache()
        updated = replace(current, **changes, updated_at=self.now())
        if not changed:
            # 没有可写列时直接读回原记录返回，不发 UPDATE。
            return current
        # 只写变更列与 updated_at，避免整行 SELECT 之外再全列 UPDATE 大 JSON 列。
        encoded = dict(zip(self._ENCODE_COLUMNS, self._encode_record(updated)))
        assignments = [f"{column} = ?" for column in sorted(changed)]
        assignments.append("updated_at = ?")
        values = [encoded[column] for column in sorted(changed)]
        values.extend((updated.updated_at, updated.id))
        with self._database.transaction() as connection:
            connection.execute(
                f"UPDATE {self._TABLE} SET {', '.join(assignments)} WHERE id = ?", values
            )
        self._maybe_prune()
        return updated

    async def aupdate(self, record_id: object, **changes: Any) -> ApiCallAuditRecord:
        return await asyncio.to_thread(self.update, record_id, **changes)

    @staticmethod
    def _query_parts(
        *,
        request_id: object | None,
        profile_id: object | None,
        session_id: object | None,
        turn_id: object | None,
        status: object | None,
        provider: object | None,
        protocol: object | None,
        channel_id: object | None,
        kind: object | None,
        operation: object | None,
        requested_model: object | None,
        response_model: object | None,
        started_after: object | None,
        started_before: object | None,
    ) -> tuple[str, tuple[object, ...]]:
        filters: list[str] = []
        parameters: list[object] = []
        for field_name, value in (
            ("request_id", request_id),
            ("profile_id", profile_id),
            ("session_id", session_id),
            ("turn_id", turn_id),
            ("status", status),
            ("provider", provider),
            ("protocol", protocol),
            ("channel_id", channel_id),
            ("kind", kind),
            ("operation", operation),
            ("requested_model", requested_model),
            ("response_model", response_model),
        ):
            if value is not None:
                filters.append(f"{field_name} = ?")
                parameters.append(_identifier(value, field_name=field_name))
        if started_after is not None:
            filters.append("started_at >= ?")
            parameters.append(_finite_number(started_after, field_name="started_after"))
        if started_before is not None:
            filters.append("started_at <= ?")
            parameters.append(_finite_number(started_before, field_name="started_before"))
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        return where, tuple(parameters)

    def list_recent(
        self,
        *,
        limit: int = 100,
        request_id: object | None = None,
        profile_id: object | None = None,
        session_id: object | None = None,
        turn_id: object | None = None,
        status: object | None = None,
        provider: object | None = None,
        protocol: object | None = None,
        channel_id: object | None = None,
        kind: object | None = None,
        operation: object | None = None,
        requested_model: object | None = None,
        response_model: object | None = None,
        started_after: object | None = None,
        started_before: object | None = None,
    ) -> tuple[ApiCallAuditRecord, ...]:
        try:
            safe_limit = max(0, min(int(limit), 1000))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("limit must be an integer") from exc
        if safe_limit == 0:
            return ()
        where, filter_parameters = self._query_parts(
            request_id=request_id,
            profile_id=profile_id,
            session_id=session_id,
            turn_id=turn_id,
            status=status,
            provider=provider,
            protocol=protocol,
            channel_id=channel_id,
            kind=kind,
            operation=operation,
            requested_model=requested_model,
            response_model=response_model,
            started_after=started_after,
            started_before=started_before,
        )
        parameters = list(filter_parameters)
        parameters.append(safe_limit)
        with self._database._lock:
            rows = self._database.connection.execute(
                f"SELECT * FROM {self._TABLE} {where} ORDER BY started_at DESC, id DESC LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return tuple(self._from_row(row) for row in rows)

    async def alist_recent(self, **kwargs: Any) -> tuple[ApiCallAuditRecord, ...]:
        return await asyncio.to_thread(self.list_recent, **kwargs)

    def count(self, **filters: Any) -> int:
        where, parameters = self._query_parts(
            request_id=filters.pop("request_id", None),
            profile_id=filters.pop("profile_id", None),
            session_id=filters.pop("session_id", None),
            turn_id=filters.pop("turn_id", None),
            status=filters.pop("status", None),
            provider=filters.pop("provider", None),
            protocol=filters.pop("protocol", None),
            channel_id=filters.pop("channel_id", None),
            kind=filters.pop("kind", None),
            operation=filters.pop("operation", None),
            requested_model=filters.pop("requested_model", None),
            response_model=filters.pop("response_model", None),
            started_after=filters.pop("started_after", None),
            started_before=filters.pop("started_before", None),
        )
        if filters:
            names = ", ".join(sorted(filters))
            raise ValueError(f"unsupported audit query fields: {names}")
        with self._database._lock:
            row = self._database.connection.execute(
                f"SELECT COUNT(*) FROM {self._TABLE} {where}", parameters
            ).fetchone()
        return int(row[0]) if row is not None else 0

    async def acount(self, **filters: Any) -> int:
        return await asyncio.to_thread(self.count, **filters)

    def _aggregate_parts(self, **filters: Any) -> tuple[str, tuple[object, ...]]:
        """复用 count() 的过滤键白名单，未知键以 ValueError 拒绝。"""

        where, parameters = self._query_parts(
            request_id=filters.pop("request_id", None),
            profile_id=filters.pop("profile_id", None),
            session_id=filters.pop("session_id", None),
            turn_id=filters.pop("turn_id", None),
            status=filters.pop("status", None),
            provider=filters.pop("provider", None),
            protocol=filters.pop("protocol", None),
            channel_id=filters.pop("channel_id", None),
            kind=filters.pop("kind", None),
            operation=filters.pop("operation", None),
            requested_model=filters.pop("requested_model", None),
            response_model=filters.pop("response_model", None),
            started_after=filters.pop("started_after", None),
            started_before=filters.pop("started_before", None),
        )
        if filters:
            names = ", ".join(sorted(filters))
            raise ValueError(f"unsupported audit query fields: {names}")
        return where, parameters

    def aggregate(self, **filters: Any) -> tuple[dict[str, object], ...]:
        """只读聚合：按 status 与 COALESCE(channel_id,'') 两组返回计数与耗时统计。

        每组条目键固定为 {key,count,avg_first_ms,avg_total_ms,max_total_ms}；
        AVG/MAX 只在有限数值上计算，结果四舍五入到毫秒级，无样本时为 null。
        """

        where, parameters = self._aggregate_parts(**filters)
        rows: list[dict[str, object]] = []
        with self._database._lock:
            for column, prefix in (
                ("status", "status"),
                ("COALESCE(channel_id, '')", "channel"),
            ):
                cursor = self._database.connection.execute(
                    f"""
                    SELECT {column} AS bucket,
                           COUNT(*) AS total,
                           AVG(CAST(time_to_first_token_ms AS REAL)) AS avg_first,
                           AVG(CAST(total_duration_ms AS REAL)) AS avg_total,
                           MAX(CAST(total_duration_ms AS REAL)) AS max_total
                    FROM {self._TABLE} {where}
                    GROUP BY {column}
                    """,
                    parameters,
                )
                for row in cursor.fetchall():
                    bucket = row["bucket"]
                    if bucket is None:
                        bucket = ""
                    rows.append(
                        {
                            "key": f"{prefix}:{bucket}",
                            "count": 0 if row["total"] is None else int(row["total"]),
                            "avg_first_ms": (
                                None
                                if row["avg_first"] is None
                                else round(max(0.0, float(row["avg_first"])), 3)
                            ),
                            "avg_total_ms": (
                                None
                                if row["avg_total"] is None
                                else round(max(0.0, float(row["avg_total"])), 3)
                            ),
                            "max_total_ms": (
                                None
                                if row["max_total"] is None
                                else round(max(0.0, float(row["max_total"])), 3)
                            ),
                        }
                    )
        return tuple(rows)

    async def aaggregate(self, **filters: Any) -> tuple[dict[str, object], ...]:
        return await asyncio.to_thread(self.aggregate, **filters)

    def _invalidate_summary_cache(self) -> None:
        """写入路径调用：清空 summary 的短 TTL 缓存槽。"""

        self._summary_cache = None

    def summary(self, **filters: Any) -> dict[str, object]:
        """把聚合结果整理成 {total,by_status,by_channel,latency_ms} 的安全汇总。

        状态轮询与控制台刷新高频调用；同参同 TTL 窗口（
        SUMMARY_CACHE_TTL_SECONDS）内复用聚合结果，未知字段沿用
        aggregate 的键白名单校验（ValueError）。
        """

        signature: tuple[object, ...] = tuple(sorted(filters.items()))
        window = int(self.now() // SUMMARY_CACHE_TTL_SECONDS)
        cached = self._summary_cache
        if cached is not None and cached[0] == (signature, window):
            return cached[1].copy()
        groups = self.aggregate(**filters)
        by_status = tuple(item for item in groups if str(item["key"]).startswith("status:"))
        by_channel = tuple(item for item in groups if str(item["key"]).startswith("channel:"))
        total = sum(int(item["count"]) for item in groups if str(item["key"]).startswith("status:"))
        result = {
            "total": total,
            "by_status": by_status,
            "by_channel": by_channel,
            "latency_ms": {
                "avg_first": None,
                "avg_total": None,
                "max_total": None,
            },
        }
        self._summary_cache = ((signature, window), result)
        return result.copy()

    async def asummary(self, **filters: Any) -> dict[str, object]:
        return await asyncio.to_thread(self.summary, **filters)

    def prune(self, *, max_records: int = 10_000) -> int:
        """保留最新的有界记录，返回删除数量。"""

        self._invalidate_summary_cache()
        if isinstance(max_records, bool):
            raise ValueError("max_records must be an integer")
        try:
            limit = int(max_records)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_records must be an integer") from exc
        if limit < 1 or limit > 1_000_000:
            raise ValueError("max_records is outside the allowed range")
        with self._database.transaction() as connection:
            cursor = connection.execute(
                f"""
                DELETE FROM {self._TABLE}
                WHERE id NOT IN (
                    SELECT id FROM {self._TABLE}
                    ORDER BY started_at DESC, id DESC
                    LIMIT ?
                )
                """,
                (limit,),
            )
        return int(cursor.rowcount)

    async def aprune(self, *, max_records: int = 10_000) -> int:
        return await asyncio.to_thread(self.prune, max_records=max_records)


__all__ = [
    "API_CALL_AUDIT_SCHEMA_VERSION",
    "ApiCallAuditHandle",
    "ApiCallAuditRecord",
    "ApiCallAuditRepository",
    "MAX_AUDIT_IDENTIFIER_LENGTH",
    "MAX_AUDIT_JSON_LENGTH",
    "MAX_AUDIT_TEXT_LENGTH",
    "sanitize_audit_value",
]
