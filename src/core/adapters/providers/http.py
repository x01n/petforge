from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from core.adapters.direct.errors import (
    AdapterConfigurationError,
    AdapterProtocolError,
    ProviderAdapterError,
    classify_exception,
)

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


def endpoint_url(channel: Any, *, default_path: str, endpoint_name: str = "chat") -> str:
    base_url = str(getattr(channel, "base_url", "") or "").strip().rstrip("/")
    if not base_url:
        raise AdapterConfigurationError("channel base_url is required")
    endpoint_getter = getattr(channel, "endpoint_path", None)
    if callable(endpoint_getter):
        endpoint = endpoint_getter(endpoint_name)
    else:
        endpoint = getattr(channel, "endpoint", default_path)
    endpoint = str(endpoint or default_path).strip()
    if endpoint.lower().startswith(("http://", "https://")):
        return endpoint
    if not endpoint.startswith("/"):
        endpoint = "/" + endpoint
    # Channels often store a provider base path (e.g. /v1). Do not duplicate it.
    return join_base_path(base_url, endpoint)


def join_base_path(base_url: str, endpoint: str) -> str:
    """拼接 URL 路径并避免 ``/v1/v1`` 这类重复版本段。"""

    base = str(base_url or "").rstrip("/")
    path = str(endpoint or "").strip()
    if not base:
        raise AdapterConfigurationError("channel base_url is required")
    if path.lower().startswith(("http://", "https://")):
        return path
    parsed = urlsplit(base)
    endpoint_parts = urlsplit(path)
    endpoint_path = endpoint_parts.path or "/"
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path
    base_path = parsed.path.rstrip("/")
    if endpoint_path == base_path and base_path:
        joined_path = base_path
    elif base_path and endpoint_path.startswith(base_path + "/"):
        joined_path = base_path + endpoint_path[len(base_path) :]
    elif base_path and base_path.endswith(endpoint_path):
        # 允许 base_url 已经是完整端点，避免再次拼接同一路径。
        joined_path = base_path
    else:
        joined_path = base_path + (
            endpoint_path if endpoint_path.startswith("/") else "/" + endpoint_path
        )
    # 端点查询参数是协议的一部分（例如 Gemini 的 ``alt=sse``），不能在
    # 拼接版本路径时丢失；base URL 的查询参数则不继承，避免意外泄漏密钥。
    return urlunsplit(
        (parsed.scheme, parsed.netloc, joined_path, endpoint_parts.query, endpoint_parts.fragment)
    )


def configured_headers(
    channel: Any, *, authorization: str | None = None, accept: str = "text/event-stream"
) -> dict[str, str]:
    configured = getattr(channel, "headers", {}) or {}
    if not isinstance(configured, Mapping):
        raise AdapterConfigurationError("channel headers must be a mapping")
    result = {str(key): str(value) for key, value in configured.items()}
    protected = {
        "authorization",
        "content-type",
        "accept",
        "x-api-key",
        "x-goog-api-key",
        "anthropic-version",
    }
    for key in tuple(result):
        if key.lower() in protected:
            result.pop(key, None)
    if authorization:
        result["Authorization"] = authorization
    result["Accept"] = accept
    result["Content-Type"] = "application/json"
    return result


def status_error(status_code: int, *, retry_after: object = None) -> ProviderAdapterError:
    code = int(status_code)
    if code in {401, 403}:
        category = "authentication" if code == 401 else "authorization"
        message = (
            "provider authentication failed" if code == 401 else "provider authorization failed"
        )
        retryable = False
    elif code == 429:
        category, message, retryable = "rate_limit", "provider rate limit exceeded", True
    elif code in {408, 425}:
        category, message, retryable = "timeout", "provider request timed out", True
    elif code >= 500:
        category, message, retryable = "server", "provider service is unavailable", True
    else:
        category, message, retryable = "protocol", f"provider returned HTTP {code}", False
    error = ProviderAdapterError(message, category=category, status_code=code, retryable=retryable)
    if retry_after is not None:
        try:
            value = float(retry_after)
            if value >= 0:
                error.retry_after_seconds = value  # type: ignore[attr-defined]
        except (TypeError, ValueError, OverflowError):
            pass
    return error


def provider_error_details(value: object) -> tuple[str, str, bool]:
    """把供应商流内错误映射为安全类别和固定提示。

    只读取错误类别字段，不把供应商返回的正文、密钥或请求标识带入提示。
    """

    fields: list[str] = []

    def collect(item: object) -> None:
        if not isinstance(item, Mapping):
            return
        for key in ("code", "type", "status", "reason"):
            raw = item.get(key)
            if isinstance(raw, (str, int)) and not isinstance(raw, bool):
                fields.append(str(raw).lower())
        nested = item.get("error")
        if isinstance(nested, Mapping):
            collect(nested)

    collect(value)
    code = " ".join(fields)
    if any(
        token in code
        for token in (
            "auth",
            "api_key",
            "apikey",
            "credential",
            "missing_key",
            "key_invalid",
            "unauthenticated",
            "invalid_key",
            "invalidkey",
            "unauthorized",
        )
    ):
        return "authentication", "provider authentication failed; check the API key", False
    if any(
        token in code for token in ("permission", "forbidden", "access_denied", "permission_denied")
    ):
        return "authorization", "provider authorization failed; check API permissions", False
    if any(
        token in code
        for token in ("rate", "limit", "resource_exhausted", "quota", "too_many_requests")
    ):
        return "rate_limit", "provider rate limit reached", True
    return "server", "provider returned an error payload", True


async def open_stream(client: Any, method: str, url: str, **kwargs: Any) -> Any:
    try:
        stream_cm = client.stream(method, url, **kwargs)
        if inspect.isawaitable(stream_cm):
            stream_cm = await stream_cm
        return stream_cm
    except asyncio_cancelled_types():
        raise
    except Exception as exc:
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            raise ProviderAdapterError(
                "provider request timed out", category="timeout", retryable=True, cause=exc
            ) from exc
        if httpx is not None and isinstance(exc, httpx.TransportError):
            raise ProviderAdapterError(
                "provider network request failed", category="network", retryable=True, cause=exc
            ) from exc
        raise


def asyncio_cancelled_types() -> tuple[type[BaseException], ...]:
    # Avoid importing asyncio at module import time in embedders that replace its policy.
    import asyncio

    return (asyncio.CancelledError,)


async def iter_sse_json(response: Any) -> AsyncIterator[tuple[str, Mapping[str, Any]]]:
    """解析 SSE 或 JSONL；多行 data 会在空行处合并。"""

    event_name = "message"
    data_lines: list[str] = []

    async def flush() -> tuple[str, Mapping[str, Any]] | None:
        nonlocal event_name
        if not data_lines:
            event_name = "message"
            return None
        raw = "\n".join(data_lines).strip()
        data_lines.clear()
        current_event = event_name
        event_name = "message"
        if not raw or raw == "[DONE]":
            return None
        try:
            value = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise AdapterProtocolError("provider stream payload is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise AdapterProtocolError("provider stream payload must be an object")
        return current_event, value

    async for raw_line in response.aiter_lines():
        line = str(raw_line)
        if not line:
            item = await flush()
            if item is not None:
                yield item
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip() or "message"
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        if line == "data":
            data_lines.append("")
            continue
        # SSE 的 id/retry 及其他字段不属于数据载荷，应忽略而不是拼入 JSON。
        if line.startswith("id:") or line.startswith("retry:") or ":" in line:
            continue
        # Gemini proxies and test transports sometimes return JSONL without data:.
        if not data_lines and line.lstrip().startswith("{"):
            data_lines.append(line)
            item = await flush()
            if item is not None:
                yield item
            continue
        data_lines.append(line)
    item = await flush()
    if item is not None:
        yield item


def response_status(response: Any) -> None:
    status_code = int(getattr(response, "status_code", 200))
    if status_code < 200 or status_code >= 300:
        headers = getattr(response, "headers", {}) or {}
        raise status_error(status_code, retry_after=headers.get("retry-after"))


def normalize_stream_error(
    error: ProviderAdapterError, *, before_first_event: bool
) -> ProviderAdapterError:
    """按已发布事件数修正错误边界，同时保留安全分类和重试信息。"""

    if error.before_first_event == before_first_event:
        return error
    normalized = ProviderAdapterError(
        error.safe_message,
        category=error.category,
        status_code=error.status_code,
        retryable=error.retryable,
        before_first_event=before_first_event,
        cause=error,
    )
    retry_after = getattr(error, "retry_after_seconds", None)
    if retry_after is not None:
        normalized.retry_after_seconds = retry_after  # type: ignore[attr-defined]
    return normalized


def classify_stream_exception(
    error: BaseException, *, before_first_event: bool
) -> ProviderAdapterError:
    """把流上下文和迭代阶段的底层异常归一到公共错误分类。"""

    # httpx 的部分传输异常名称（例如 ReadError）不含网络相关关键字，
    # 优先按类型判断，避免流迭代阶段被误标为 unknown。
    if httpx is not None:
        if isinstance(error, httpx.TimeoutException):
            return ProviderAdapterError(
                "provider request timed out",
                category="timeout",
                retryable=True,
                before_first_event=before_first_event,
                cause=error,
            )
        if isinstance(error, httpx.TransportError):
            return ProviderAdapterError(
                "provider network request failed",
                category="network",
                retryable=True,
                before_first_event=before_first_event,
                cause=error,
            )
    return classify_exception(error, before_first_event=before_first_event)


__all__ = [
    "configured_headers",
    "classify_stream_exception",
    "endpoint_url",
    "iter_sse_json",
    "join_base_path",
    "open_stream",
    "response_status",
    "status_error",
    "normalize_stream_error",
]
