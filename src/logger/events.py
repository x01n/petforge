"""统一、脱敏的结构化运行事件日志。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

_SAFE_KEY = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|token|secret|password|authorization|cookie|credential|content|text|prompt)",
    re.IGNORECASE,
)
_BLOCKED_RAW_KEY = re.compile(
    r"(?:^|[._-])(?:path|cwd|url|uri|endpoint|command|argv|stdout|stderr|message|reason|error)"
    r"(?:$|[._-])",
    re.IGNORECASE,
)
_SAFE_METADATA_SUFFIX = (
    "_bytes",
    "_code",
    "_count",
    "_fingerprint",
    "_length",
    "_lines",
    "_type",
)
_FINGERPRINT_KEY = re.compile(
    r"(?:^|[._-])(?:operation|request)(?:_id)?(?:$|[._-])",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?i)(?:\bsk-[a-z0-9_-]{8,}\b|"
    r"(?:x-goog-api[_-]?key|session[_-]?key)[\"']?\s*[:=]\s*[\"']?[^\s,;\"']+[\"']?|"
    r"\b(?:api[_-]?key|token|secret|password|authorization|cookie|credential)\b"
    r"\s*[:=]\s*[^\s,;]+)"
)
_URL_VALUE = re.compile(r"(?i)\b(?:https?|wss?)://[^\s\]\[(){}<>\"']+")
_POSIX_PATH_VALUE = re.compile(
    r"(?<![a-zA-Z0-9_.-])/(?:[^\s\]\[(){}<>\"':;,]+/)+"
    r"[^\s\]\[(){}<>\"':;,]*"
)
_WINDOWS_PATH_VALUE = re.compile(r"(?i)\b[a-z]:\\(?:[^\s\"'<>|]+\\)*[^\s\"'<>|]*")
_LABEL = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,95}\Z")


def event_fingerprint(value: object) -> str:
    """返回不可逆的短关联标识，不把原始 ID 写入日志。"""

    rendered = str(value or "")
    if not rendered:
        return ""
    return hashlib.sha256(rendered.encode("utf-8", "replace")).hexdigest()[:16]


def sanitize_log_text(value: object, *, limit: int = 2048) -> str:
    """压平并脱敏日志文本，不保留凭据、网络地址或隐私路径。"""

    try:
        safe_limit = max(0, min(int(limit), 16_384))
    except (TypeError, ValueError, OverflowError):
        safe_limit = 2048
    rendered = " ".join(str(value or "").replace("\x00", " ").split())
    rendered = _SECRET_VALUE.sub("[redacted]", rendered)
    rendered = _URL_VALUE.sub("[url]", rendered)
    rendered = _WINDOWS_PATH_VALUE.sub("[path]", rendered)
    rendered = _POSIX_PATH_VALUE.sub("[path]", rendered)
    return rendered[:safe_limit]


def _safe_label(value: object, *, fallback: str, limit: int) -> str:
    rendered = str(value or "").strip()
    if not _LABEL.fullmatch(rendered):
        return fallback
    return rendered[:limit]


def _safe_value(key: str, value: object) -> object:
    normalized_key = str(key or "").strip().lower()
    if _SECRET_KEY.search(normalized_key):
        return "[redacted]"
    metadata_only = normalized_key.endswith(_SAFE_METADATA_SUFFIX)
    if _BLOCKED_RAW_KEY.search(normalized_key) and not metadata_only:
        return "[redacted]"
    if _FINGERPRINT_KEY.search(normalized_key) and not normalized_key.endswith("_fingerprint"):
        return event_fingerprint(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 3) if math.isfinite(value) else None
    if isinstance(value, str):
        return sanitize_log_text(value, limit=160)
    if isinstance(value, Mapping):
        return {"kind": "mapping", "size": min(len(value), 10_000)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {"kind": "sequence", "size": min(len(value), 10_000)}
    return type(value).__name__


def log_event(
    target: logging.Logger,
    event: str,
    *,
    component: str,
    status: str,
    level: int = logging.INFO,
    correlation_id: object = None,
    operation_id: object = None,
    request_id: object = None,
    duration_ms: float | None = None,
    reason_code: str | None = None,
    detail: str | None = None,
    fields: Mapping[str, Any] | None = None,
) -> None:
    """输出稳定 JSON 事件；标识符只写指纹，正文和敏感位置统一脱敏。"""

    if not isinstance(target, logging.Logger):
        raise TypeError("target must be a logging.Logger")
    payload: dict[str, object] = {
        "event": _safe_label(event, fallback="unknown", limit=96),
        "component": _safe_label(component, fallback="unknown", limit=64),
        "status": _safe_label(status, fallback="unknown", limit=32),
    }
    fingerprint = event_fingerprint(correlation_id)
    if fingerprint:
        payload["correlation"] = fingerprint
    operation_fingerprint = event_fingerprint(operation_id)
    if operation_fingerprint:
        payload["operation"] = operation_fingerprint
    request_fingerprint = event_fingerprint(request_id)
    if request_fingerprint:
        payload["request"] = request_fingerprint
    if duration_ms is not None:
        try:
            duration = float(duration_ms)
        except (TypeError, ValueError, OverflowError):
            duration = -1.0
        if math.isfinite(duration) and duration >= 0:
            payload["duration_ms"] = round(duration, 3)
    if reason_code is not None:
        payload["reason_code"] = _safe_label(
            reason_code,
            fallback="unknown",
            limit=64,
        ).lower()
    if detail is not None:
        safe_detail = sanitize_log_text(detail, limit=160)
        if safe_detail:
            payload["detail"] = safe_detail
    for raw_key, raw_value in (fields or {}).items():
        key = str(raw_key or "").strip().lower()
        if not _SAFE_KEY.fullmatch(key) or key in payload:
            continue
        payload[key] = _safe_value(key, raw_value)
    target.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True))


__all__ = ["event_fingerprint", "log_event", "sanitize_log_text"]
