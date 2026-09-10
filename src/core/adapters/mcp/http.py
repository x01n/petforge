"""MCP Streamable HTTP 与旧式 SSE 传输客户端。

客户端只接受 JSON-RPC 2.0 对象，不把 URL、请求头或响应正文写入异常消息。
``transport=http`` 直接向端点发送 POST；``transport=sse`` 先订阅 SSE，
从 ``endpoint`` 事件取得 message endpoint，再通过 POST 发送请求。两种传输
共享 ``list_tools``/``call_tool`` 契约，因而可以直接交给 ``MCPToolBridge``。
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

try:  # httpx 是核心依赖；保留导入降级以便文档扫描器工作。
    import httpx
except ImportError:  # pragma: no cover - 运行时构造客户端时给出明确错误
    httpx = None  # type: ignore[assignment]

from .content import (
    MCPContentLimits,
    MCPGetPromptResult,
    MCPPrompt,
    MCPReadResourceResult,
    MCPResource,
    get_mcp_prompt,
    list_mcp_prompts,
    list_mcp_resources,
    read_mcp_resource,
)
from .protocol import (
    MCP_PROTOCOL_VERSION,
    MCPCloseHandler,
    MCPConnectionHandler,
    MCPNotificationDispatcher,
    MCPNotificationHandler,
    MCPProtocolError,
    MCPServerCapabilities,
    MCPServerError,
    MCPTimeoutError,
    MCPTool,
    MCPTransportError,
    negotiate_protocol_version,
    reject_json_constant,
    validate_json_value,
)

_TRANSPORT_ALIASES = {
    "http": "http",
    "post": "http",
    "streamable_http": "http",
    "streamable-http": "http",
    "sse": "sse",
    "eventsource": "sse",
    "event_source": "sse",
}


def normalize_mcp_transport(value: object, *, default: str = "http") -> str:
    """规范化 MCP 网络传输名称；未知值直接拒绝。"""

    raw = str(value if value is not None else default).strip().lower()
    normalized = _TRANSPORT_ALIASES.get(raw)
    if normalized is None:
        raise ValueError("MCP HTTP transport must be http or sse")
    return normalized


def _strict_bool(value: object, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "on", "1"}:
        return True
    if text in {"false", "no", "off", "0"}:
        return False
    raise ValueError(f"{field_name} must be a boolean")


def _validated_url(value: object, field_name: str = "MCP URL") -> str:
    url = str(value or "").strip()
    if not url or len(url) > 4096 or any(char in url for char in "\r\n\x00"):
        raise ValueError(f"{field_name} is invalid")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must use http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} must not contain credentials")
    if parsed.fragment:
        raise ValueError(f"{field_name} must not contain a fragment")
    try:
        # 触发端口解析，拒绝非法端口而不是等到请求时才失败。
        _ = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field_name} has an invalid port") from exc
    if not parsed.hostname or len(parsed.hostname) > 255:
        raise ValueError(f"{field_name} has an invalid host")
    return url


def _validated_relative_or_url(value: object, field_name: str) -> str:
    raw = str(value or "").strip()
    if not raw or len(raw) > 4096 or any(char in raw for char in "\r\n\x00"):
        raise ValueError(f"{field_name} is invalid")
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc:
        return _validated_url(raw, field_name)
    if not raw.startswith("/"):
        raw = "/" + raw
    return raw


def _validated_headers(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("MCP headers must be a mapping")
    if len(value) > 64:
        raise ValueError("MCP headers contain too many entries")
    result: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            raise ValueError("MCP header names and values must be strings")
        key = raw_key.strip()
        if (
            not key
            or len(key) > 256
            or any(char in key for char in "\r\n\x00")
            or not all(char.isalnum() or char in "!#$%&'*+-.^_`|~" for char in key)
        ):
            raise ValueError("MCP header name is invalid")
        if len(raw_value) > 8192 or any(char in raw_value for char in "\r\n\x00"):
            raise ValueError("MCP header value is invalid")
        # HTTP field names are case-insensitive.  Reject duplicates so that a
        # hidden duplicate cannot override a visible authorization policy.
        lowered = key.lower()
        if any(existing.lower() == lowered for existing in result):
            raise ValueError("MCP headers contain duplicate names")
        result[key] = raw_value
    return result


def _timeout(value: object, field_name: str, default: float) -> float:
    try:
        number = float(default if value is None else value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field_name} is invalid") from exc
    if not 0.1 <= number <= 600 or number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{field_name} is outside the allowed range")
    return number


@dataclass(frozen=True, repr=False)
class HttpMCPServerConfig:
    """一个 MCP HTTP/SSE 服务器的安全连接配置。

    ``headers`` 明确设置为不参与 ``repr``，避免 API 密钥进入日志、调试器
    或配置诊断摘要。调用方仍可通过 ``header_names`` 获取无值的审计信息。
    """

    name: str
    url: str
    transport: str = "http"
    headers: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)
    message_url: str | None = field(default=None, repr=False, compare=False)
    request_timeout_seconds: float = 30.0
    startup_timeout_seconds: float = 10.0
    max_message_bytes: int = 4 * 1024 * 1024
    initialize: bool = True
    verify_tls: bool = True
    close_session: bool = True

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name or len(name) > 128 or any(char in name for char in "\r\n\x00"):
            raise ValueError("MCP server name is invalid")
        url = _validated_url(self.url)
        transport = normalize_mcp_transport(self.transport)
        headers = _validated_headers(self.headers)
        message_url = (
            None
            if self.message_url is None or not str(self.message_url).strip()
            else _validated_relative_or_url(self.message_url, "MCP message URL")
        )
        request_timeout = _timeout(
            self.request_timeout_seconds, "MCP request_timeout_seconds", 30.0
        )
        startup_timeout = _timeout(
            self.startup_timeout_seconds, "MCP startup_timeout_seconds", 10.0
        )
        try:
            max_bytes = int(self.max_message_bytes)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("MCP max_message_bytes is invalid") from exc
        if not 1024 <= max_bytes <= 64 * 1024 * 1024:
            raise ValueError("MCP max_message_bytes is outside the allowed range")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "transport", transport)
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "message_url", message_url)
        object.__setattr__(self, "request_timeout_seconds", request_timeout)
        object.__setattr__(self, "startup_timeout_seconds", startup_timeout)
        object.__setattr__(self, "max_message_bytes", max_bytes)
        object.__setattr__(
            self, "initialize", _strict_bool(self.initialize, "MCP initialize", True)
        )
        object.__setattr__(
            self, "verify_tls", _strict_bool(self.verify_tls, "MCP verify_tls", True)
        )
        object.__setattr__(
            self, "close_session", _strict_bool(self.close_session, "MCP close_session", True)
        )

    @property
    def header_names(self) -> tuple[str, ...]:
        """返回不含值的请求头名称，供诊断和 UI 使用。"""

        return tuple(sorted(self.headers, key=str.lower))

    def __repr__(self) -> str:
        """返回不含查询参数、请求头值和 message endpoint 的调试摘要。"""

        parsed = urlsplit(self.url)
        safe_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        if parsed.query:
            safe_url += "?<redacted>"
        return (
            "HttpMCPServerConfig("
            f"name={self.name!r}, url={safe_url!r}, transport={self.transport!r}, "
            f"header_names={self.header_names!r}, "
            f"request_timeout_seconds={self.request_timeout_seconds!r}, "
            f"startup_timeout_seconds={self.startup_timeout_seconds!r}, "
            f"max_message_bytes={self.max_message_bytes!r}, initialize={self.initialize!r}, "
            f"verify_tls={self.verify_tls!r}, close_session={self.close_session!r})"
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HttpMCPServerConfig:
        if not isinstance(value, Mapping):
            raise ValueError("MCP HTTP server must be a mapping")
        return cls(
            name=str(value.get("name", "default")),
            url=value.get("url", value.get("endpoint", "")),
            transport=value.get("transport", "http"),
            headers=value.get("headers", {}),
            message_url=value.get("message_url", value.get("message_endpoint")),
            request_timeout_seconds=value.get(
                "request_timeout_seconds", value.get("timeout_seconds", 30.0)
            ),
            startup_timeout_seconds=value.get("startup_timeout_seconds", 10.0),
            max_message_bytes=value.get("max_message_bytes", 4 * 1024 * 1024),
            initialize=value.get("initialize", True),
            verify_tls=value.get("verify_tls", True),
            close_session=value.get("close_session", True),
        )


def _same_origin(base: str, target: str) -> bool:
    left = urlsplit(base)
    right = urlsplit(target)
    return (left.scheme.lower(), left.hostname, left.port or _default_port(left.scheme)) == (
        right.scheme.lower(),
        right.hostname,
        right.port or _default_port(right.scheme),
    )


def _default_port(scheme: str) -> int | None:
    return 443 if scheme.lower() == "https" else 80 if scheme.lower() == "http" else None


def _safe_message(value: object, secrets: Sequence[str] = ()) -> str:
    text = str(value or "MCP server error").strip() or "MCP server error"
    for secret in secrets:
        if secret:
            text = text.replace(secret, "<redacted>")
    return text[:256]


class _BoundedHTTPResponse:
    """为 HTTP MCP 响应提供流式、有界读取和统一关闭。"""

    def __init__(
        self,
        response: Any,
        *,
        context: Any | None = None,
        max_bytes: int,
    ) -> None:
        self._response = response
        self._context = context
        self._max_bytes = int(max_bytes)
        self._read_bytes = 0
        self._content: bytes | None = None
        self._closed = False
        self.status_code = int(getattr(response, "status_code", 0) or 0)
        self.headers = getattr(response, "headers", {})
        raw_length = None
        try:
            raw_length = self.headers.get("content-length")
        except (AttributeError, TypeError):
            pass
        if raw_length is not None:
            try:
                content_length = int(str(raw_length).strip())
            except (TypeError, ValueError, OverflowError) as exc:
                raise MCPProtocolError("MCP HTTP Content-Length is invalid") from exc
            if content_length < 0 or content_length > self._max_bytes:
                raise MCPProtocolError("MCP HTTP response exceeds the message limit")

    def _consume(self, amount: int) -> None:
        if amount < 0 or self._read_bytes + amount > self._max_bytes:
            raise MCPProtocolError("MCP HTTP response exceeds the message limit")
        self._read_bytes += amount

    async def aiter_lines(self) -> AsyncIterator[str]:
        iterator = getattr(self._response, "aiter_lines", None)
        if callable(iterator):
            async for line in iterator():
                # SSE 的消息上限由 ``iter_sse_events`` 按事件边界计算；
                # 长连接本身不能累计成一个总字节上限。
                yield str(line)
            return
        raw = await self.read_content()
        for line in raw.decode("utf-8", "replace").splitlines():
            yield line

    async def read_content(self) -> bytes:
        if self._content is not None:
            return self._content
        iterator = getattr(self._response, "aiter_bytes", None)
        if callable(iterator):
            chunks: list[bytes] = []
            async for chunk in iterator():
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                if not isinstance(chunk, (bytes, bytearray)):
                    raise MCPTransportError("MCP HTTP response could not be read")
                payload = bytes(chunk)
                self._consume(len(payload))
                chunks.append(payload)
            self._content = b"".join(chunks)
            return self._content
        try:
            raw = getattr(self._response, "content", b"")
            if inspect.isawaitable(raw):
                raw = await raw
        except Exception as exc:
            raise MCPTransportError("MCP HTTP response could not be read") from exc
        if isinstance(raw, str):
            payload = raw.encode("utf-8")
        elif isinstance(raw, (bytes, bytearray)):
            payload = bytes(raw)
        else:
            raise MCPTransportError("MCP HTTP response could not be read")
        self._consume(len(payload))
        self._content = payload
        return payload

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._response, "aclose", None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
        context = self._context
        self._context = None
        if context is not None:
            try:
                result = context.__aexit__(None, None, None)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)


async def iter_sse_events(
    response: Any, *, max_message_bytes: int = 4 * 1024 * 1024
) -> AsyncIterator[tuple[str, str]]:
    """按 SSE 事件边界读取文本数据。

    ``data:`` 多行会按 SSE 规范用换行合并；``id``、``retry`` 和注释不会
    进入载荷。该函数只负责 framing，不解析 JSON，便于两个 HTTP 入口共用。
    """

    event_name = "message"
    data_lines: list[str] = []
    data_bytes = 0

    async def flush() -> tuple[str, str] | None:
        nonlocal event_name, data_bytes
        if not data_lines:
            event_name = "message"
            data_bytes = 0
            return None
        data = "\n".join(data_lines)
        current = event_name.strip().lower() or "message"
        data_lines.clear()
        event_name = "message"
        data_bytes = 0
        return current, data

    async for raw_line in response.aiter_lines():
        line = str(raw_line)
        data_bytes += len(line.encode("utf-8", "replace")) + 1
        if data_bytes > max_message_bytes:
            raise MCPProtocolError("MCP SSE message exceeds the message limit")
        if not line:
            item = await flush()
            if item is not None:
                yield item
            continue
        if line.startswith("\ufeff"):
            line = line.lstrip("\ufeff")
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
        if line.startswith("id:") or line.startswith("retry:") or ":" in line:
            continue
    item = await flush()
    if item is not None:
        yield item


class HttpMCPClient:
    """异步 MCP HTTP/SSE 客户端。

    传入 ``client`` 时客户端生命周期由调用方管理，便于测试和复用连接池；
    未传入时创建一个关闭代理环境变量的 ``httpx.AsyncClient``。
    """

    def __init__(
        self,
        config: HttpMCPServerConfig | Mapping[str, Any],
        *,
        client: Any | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, HttpMCPServerConfig)
            else HttpMCPServerConfig.from_mapping(config)
        )
        self._owns_client = client is None
        if client is None:
            if httpx is None:  # pragma: no cover
                raise MCPTransportError("httpx is required for MCP HTTP transport")
            client = httpx.AsyncClient(trust_env=False, verify=self.config.verify_tls)
        self._client = client
        self._closed = False
        self._started = False
        self._initialized = False
        self._failure: BaseException | None = None
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[Mapping[str, Any]]] = {}
        # 超时/取消后仍可能从 SSE 收到旧回包；有限放弃 ID 防止它破坏
        # 同一 HTTP/SSE 会话中的其他请求。
        self._abandoned_ids: deque[int] = deque(maxlen=256)
        self._next_id = 0
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._message_url: str | None = self.config.message_url
        self._sse_task: asyncio.Task[None] | None = None
        self._sse_cm: Any | None = None
        self._sse_response: Any | None = None
        self._sse_ready = asyncio.Event()
        self._event_task: asyncio.Task[None] | None = None
        self._event_cm: Any | None = None
        self._event_response: Any | None = None
        self._event_stream_supported: bool | None = None
        self._notifications = MCPNotificationDispatcher()

    def _reset_message_endpoint(self) -> None:
        """丢弃当前会话发现的 endpoint，保留配置中的显式 endpoint。"""

        self._message_url = self.config.message_url

    def _invalidate_closed_sse(self) -> None:
        """将已结束的旧式 SSE 主流转换为下一次 start 的重连状态。"""

        if self.config.transport != "sse" or self._failure is not None:
            return
        task = self._sse_task
        if not self._initialized or task is None or not task.done():
            return
        self._failure = MCPTransportError("MCP SSE stream closed")
        self._started = False
        self._session_id = None
        self._reset_message_endpoint()
        self._notifications.set_capabilities(MCPServerCapabilities())
        self._notifications.notify_connection("disconnected")

    def _abandon_request(self, request_id: int) -> None:
        """记录已经取消或超时的请求，允许迟到 SSE 回包安全丢弃。"""

        if request_id not in self._abandoned_ids:
            self._abandoned_ids.append(request_id)

    @property
    def server_name(self) -> str:
        return self.config.name

    @property
    def capabilities(self) -> MCPServerCapabilities:
        return self._notifications.capabilities

    @property
    def connection_state(self) -> str:
        return self._notifications.connection_state

    def add_notification_handler(self, handler: MCPNotificationHandler) -> Callable[[], None]:
        return self._notifications.add_handler(handler)

    def add_close_handler(self, handler: MCPCloseHandler) -> Callable[[], None]:
        return self._notifications.add_close_handler(handler)

    def add_connection_handler(self, handler: MCPConnectionHandler) -> Callable[[], None]:
        return self._notifications.add_connection_handler(handler)

    @property
    def session_id(self) -> str | None:
        """返回会话标识本身；该值不会进入日志或异常。"""

        return self._session_id

    def _headers(self, *, accept: str = "application/json, text/event-stream") -> dict[str, str]:
        result = dict(self.config.headers)
        # 这些字段由传输层控制，避免配置中的重复值改变协议语义。
        for key in tuple(result):
            if key.lower() in {
                "content-length",
                "content-type",
                "accept",
                "mcp-session-id",
                "mcp-protocol-version",
            }:
                result.pop(key, None)
        result["Accept"] = accept
        result["Content-Type"] = "application/json"
        if self._protocol_version:
            result["MCP-Protocol-Version"] = self._protocol_version
        if self._session_id:
            result["Mcp-Session-Id"] = self._session_id
        return result

    def _capture_session(self, response: Any) -> None:
        headers = getattr(response, "headers", {})
        try:
            for key, value in headers.items():
                if str(key).lower() != "mcp-session-id":
                    continue
                candidate = str(value).strip()
                if (
                    not candidate
                    or len(candidate) > 256
                    or any(ord(char) < 0x21 or ord(char) > 0x7E for char in candidate)
                ):
                    return
                self._session_id = candidate
                return
        except (AttributeError, TypeError, ValueError):
            return

    async def _open_sse(self) -> None:
        if self.config.transport != "sse":
            self._sse_ready.set()
            return
        if self._message_url is not None:
            # 显式 message endpoint 仍要求与 SSE 端点同源，防止把认证头发送
            # 到配置之外的主机；SSE 服务返回的 endpoint 也遵循同一边界。
            resolved = urljoin(self.config.url, self._message_url)
            if not _same_origin(self.config.url, resolved):
                raise ValueError("MCP message URL must use the same origin")
            self._message_url = resolved
            self._sse_ready.set()
        stream = getattr(self._client, "stream", None)
        if not callable(stream):
            raise MCPTransportError("MCP SSE client does not support streaming")
        context: Any | None = None
        try:
            context = stream(
                "GET",
                self.config.url,
                headers=self._headers(accept="text/event-stream"),
                timeout=self.config.startup_timeout_seconds,
            )
            if inspect.isawaitable(context):
                context = await context
            response = await context.__aenter__()
        except asyncio.CancelledError:
            # 取消可能发生在上下文管理器的 ``__aenter__`` 期间；此时
            # ``_sse_cm`` 尚未接管资源，必须主动退出临时上下文，避免
            # HTTP 连接在热重载/重试后继续存活。
            if context is not None:
                try:
                    close_context = getattr(context, "__aexit__", None)
                    if callable(close_context):
                        result = close_context(None, None, None)
                        if inspect.isawaitable(result):
                            await result
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
            raise
        except Exception as exc:
            if context is not None:
                try:
                    close_context = getattr(context, "__aexit__", None)
                    if callable(close_context):
                        result = close_context(None, None, None)
                        if inspect.isawaitable(result):
                            await result
                except Exception:
                    pass
            raise self._transport_exception(exc, startup=True) from exc
        self._sse_cm = context
        self._sse_response = response
        status = int(getattr(response, "status_code", 0) or 0)
        if status == 404 and self._session_id:
            self._session_id = None
            self._reset_message_endpoint()
            self._started = False
            self._notifications.set_capabilities(MCPServerCapabilities())
            self._notifications.notify_connection("disconnected")
            raise MCPTransportError("MCP HTTP session expired")
        if status < 200 or status >= 300:
            await self._close_sse_stream()
            raise MCPTransportError("MCP SSE endpoint returned an invalid HTTP status")
        self._capture_session(response)
        self._sse_task = asyncio.create_task(self._read_sse(), name=f"mcp-sse-{self.server_name}")
        try:
            await asyncio.wait_for(self._sse_ready.wait(), self.config.startup_timeout_seconds)
        except TimeoutError as exc:
            raise MCPTimeoutError("MCP SSE endpoint did not publish a message endpoint") from exc
        if self._failure is not None and self._message_url is None:
            raise MCPTransportError("MCP SSE endpoint is unavailable") from self._failure
        if self._message_url is None:
            raise MCPProtocolError("MCP SSE endpoint did not publish a message endpoint")

    async def _read_sse(self) -> None:
        response = self._sse_response
        if response is None:
            return
        failure: BaseException | None = None
        try:
            async for current_event, raw in iter_sse_events(
                response, max_message_bytes=self.config.max_message_bytes
            ):
                raw = raw.strip()
                if not raw:
                    continue
                if current_event in {"endpoint", "message_endpoint"}:
                    try:
                        parsed: object = json.loads(raw)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        parsed = raw
                    if isinstance(parsed, Mapping):
                        parsed = parsed.get("url", parsed.get("endpoint", ""))
                    endpoint = str(parsed or "").strip()
                    if not endpoint:
                        raise MCPProtocolError("MCP SSE endpoint event is empty")
                    resolved = urljoin(self.config.url, endpoint)
                    if not _same_origin(self.config.url, resolved):
                        raise MCPProtocolError("MCP SSE message endpoint has a different origin")
                    self._message_url = resolved
                    self._sse_ready.set()
                    continue
                try:
                    value = json.loads(raw, parse_constant=reject_json_constant)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise MCPProtocolError("MCP SSE message is not valid JSON") from exc
                await self._deliver_response(value)
        except asyncio.CancelledError:
            return
        except BaseException as exc:
            failure = exc
        if failure is not None:
            self._notifications.set_capabilities(MCPServerCapabilities())
            self._notifications.notify_connection("disconnected")
            self._failure = failure
            self._sse_ready.set()
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(
                        failure
                        if isinstance(failure, MCPErrorTypes)
                        else MCPTransportError("MCP SSE stream failed")
                    )
            self._pending.clear()
        # 初始握手前，服务端发送 endpoint 后结束响应仍允许 POST 继续完成；
        # 已初始化会话的 EOF 则必须标记断线，否则下一次请求会误复用已关闭
        # 的 SSE 连接而不会进入 start() 的重连路径。
        if failure is None and self._initialized:
            failure = MCPTransportError("MCP SSE stream closed")
            self._notifications.set_capabilities(MCPServerCapabilities())
            self._notifications.notify_connection("disconnected")
            self._failure = failure
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(failure)
            self._pending.clear()

    async def _close_sse_stream(self) -> None:
        task = self._sse_task
        self._sse_task = None
        current = asyncio.current_task()
        if task is not None and task is not current and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        context = self._sse_cm
        self._sse_cm = None
        self._sse_response = None
        if context is not None:
            try:
                result = context.__aexit__(None, None, None)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass

    def _start_http_event_stream(self) -> None:
        """为 Streamable HTTP 打开可选 GET 通知流；旧 SSE 复用既有读取器。"""

        capabilities = self._notifications.capabilities
        list_changed = any(
            (
                capabilities.tools_list_changed,
                capabilities.resources_list_changed,
                capabilities.prompts_list_changed,
            )
        )
        if (
            self.config.transport != "http"
            or not list_changed
            or not self._notifications.has_handlers
            or self._closed
            or self._event_stream_supported is False
            or (self._event_task is not None and not self._event_task.done())
        ):
            return
        self._event_task = asyncio.create_task(
            self._http_event_loop(), name=f"mcp-http-events-{self.server_name}"
        )

    async def _http_event_loop(self) -> None:
        """有界重连 Streamable HTTP GET 流，只接收协商后的固定通知。"""

        stream = getattr(self._client, "stream", None)
        if not callable(stream):
            return
        for attempt in range(3):
            if self._closed:
                return
            context: Any | None = None
            try:
                context = stream(
                    "GET",
                    self.config.url,
                    headers=self._headers(accept="text/event-stream"),
                    timeout=self.config.startup_timeout_seconds,
                )
                if inspect.isawaitable(context):
                    context = await context
                response = await context.__aenter__()
                self._event_cm = context
                self._event_response = response
                status = int(getattr(response, "status_code", 0) or 0)
                if status == 405:
                    self._event_stream_supported = False
                    return
                if status == 404 and self._session_id:
                    # 2025-06-18 要求带会话请求收到 404 后重新 initialize。
                    # 先撤销协商状态并通知桥接器撤下旧工具；恢复任务会使用
                    # 无旧 session 的新 InitializeRequest 重建完整列表。
                    self._session_id = None
                    self._reset_message_endpoint()
                    self._started = False
                    self._notifications.set_capabilities(MCPServerCapabilities())
                    self._notifications.notify_connection("disconnected")
                    return
                if status < 200 or status >= 300:
                    raise MCPTransportError("MCP HTTP event stream returned an invalid status")
                headers = getattr(response, "headers", {})
                content_type = str(headers.get("content-type", "")).lower()
                if "text/event-stream" not in content_type:
                    self._event_stream_supported = False
                    return
                self._capture_session(response)
                self._event_stream_supported = True
                async for _event_name, raw in iter_sse_events(
                    response, max_message_bytes=self.config.max_message_bytes
                ):
                    raw = raw.strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        value = json.loads(raw, parse_constant=reject_json_constant)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        # 非法通知不进入业务层；结束本流后按有界策略重连。
                        break
                    try:
                        await self._deliver_response(value)
                    except MCPProtocolError:
                        # 独立 GET 流只接受通知；请求、响应和坏信封均忽略。
                        continue
            except asyncio.CancelledError:
                return
            except Exception:
                pass
            finally:
                if self._event_cm is context:
                    self._event_cm = None
                    self._event_response = None
                if context is not None:
                    try:
                        result = context.__aexit__(None, None, None)
                        if inspect.isawaitable(result):
                            await result
                    except Exception:
                        pass
            if attempt < 2 and not self._closed:
                await asyncio.sleep((0.1, 0.3)[attempt])

    async def _close_http_event_stream(self) -> None:
        task = self._event_task
        self._event_task = None
        current = asyncio.current_task()
        if task is not None and task is not current and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        context = self._event_cm
        self._event_cm = None
        self._event_response = None
        if context is not None:
            try:
                result = context.__aexit__(None, None, None)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass

    def _transport_exception(
        self, exc: BaseException, *, startup: bool = False
    ) -> MCPTransportError:
        if isinstance(exc, MCPTransportError):
            return exc
        if httpx is not None and isinstance(exc, httpx.TimeoutException):
            return MCPTimeoutError("MCP startup timed out" if startup else "MCP request timed out")
        return MCPTransportError("MCP HTTP transport is unavailable")

    async def start(self) -> None:
        self._invalidate_closed_sse()
        if self._started and self._failure is None:
            self._start_http_event_stream()
            self._notifications.notify_connection("connected")
            return
        async with self._start_lock:
            self._invalidate_closed_sse()
            if self._started and self._failure is None:
                self._start_http_event_stream()
                self._notifications.notify_connection("connected")
                return
            if self._closed:
                raise MCPTransportError("MCP client is closed")
            await self._close_http_event_stream()
            self._event_stream_supported = None
            self._notifications.set_capabilities(MCPServerCapabilities())
            self._failure = None
            self._protocol_version = None
            self._initialized = False
            self._sse_ready = asyncio.Event()
            try:
                await self._open_sse()
                self._started = True
                if self.config.initialize:
                    await asyncio.wait_for(
                        self._initialize(), timeout=self.config.startup_timeout_seconds
                    )
                self._start_http_event_stream()
                self._notifications.notify_connection("connected")
            except asyncio.CancelledError:
                # 取消可能发生在握手请求已经提交之后；此时不能保留
                # ``_started=True``，否则下一次热重载/重试会误复用未完成的
                # 会话。清理当前代次的流和会话状态，但保留客户端可重试。
                self._started = False
                await self._close_sse_stream()
                await self._close_http_event_stream()
                self._session_id = None
                self._reset_message_endpoint()
                self._initialized = False
                self._failure = None
                raise
            except BaseException:
                self._started = False
                await self._close_sse_stream()
                # 初始化失败后不得把上一代会话标识或发现的 message endpoint
                # 带入下一次握手；显式配置的 endpoint 会在下一代重新解析。
                self._session_id = None
                self._reset_message_endpoint()
                self._initialized = False
                raise

    async def _initialize(self) -> None:
        result = await self._request_raw(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "meapet", "version": "0.1.0"},
            },
            ensure_started=False,
        )
        version = result.get("protocolVersion")
        self._protocol_version = negotiate_protocol_version(version)
        self._notifications.set_capabilities(MCPServerCapabilities.from_initialize_result(result))
        await self._notify("notifications/initialized", {})
        self._initialized = True

    async def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        payload = {"jsonrpc": "2.0", "method": method, "params": dict(params)}
        response = await self._send_post(payload, notification=True)
        close_response = getattr(response, "aclose", None)
        if callable(close_response):
            try:
                closed = close_response()
                if inspect.isawaitable(closed):
                    await closed
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    async def _send_post(self, payload: Mapping[str, Any], *, notification: bool = False) -> Any:
        target = self._message_url if self.config.transport == "sse" else self.config.url
        if not target:
            raise MCPTransportError("MCP message endpoint is unavailable")
        response: _BoundedHTTPResponse | None = None
        context: Any | None = None
        try:
            async with self._write_lock:
                stream = getattr(self._client, "stream", None)
                if callable(stream):
                    context = stream(
                        "POST",
                        target,
                        headers=self._headers(),
                        json=dict(payload),
                        timeout=self.config.request_timeout_seconds,
                    )
                    if inspect.isawaitable(context):
                        context = await context
                    enter = getattr(context, "__aenter__", None)
                    exit_context = getattr(context, "__aexit__", None)
                    if callable(enter) and callable(exit_context):
                        raw_response = enter()
                        if inspect.isawaitable(raw_response):
                            raw_response = await raw_response
                        response = _BoundedHTTPResponse(
                            raw_response,
                            context=context,
                            max_bytes=self.config.max_message_bytes,
                        )
                        context = None
                    else:
                        context = None
                        request = self._client.request(
                            "POST",
                            target,
                            headers=self._headers(),
                            json=dict(payload),
                            timeout=self.config.request_timeout_seconds,
                        )
                        if inspect.isawaitable(request):
                            request = await request
                        response = _BoundedHTTPResponse(
                            request,
                            max_bytes=self.config.max_message_bytes,
                        )
                else:
                    request = self._client.request(
                        "POST",
                        target,
                        headers=self._headers(),
                        json=dict(payload),
                        timeout=self.config.request_timeout_seconds,
                    )
                    if inspect.isawaitable(request):
                        request = await request
                    response = _BoundedHTTPResponse(
                        request,
                        max_bytes=self.config.max_message_bytes,
                    )
        except MCPProtocolError:
            if response is not None:
                await response.aclose()
            if context is not None:
                try:
                    result = context.__aexit__(None, None, None)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
            raise
        except asyncio.CancelledError:
            if response is not None:
                await response.aclose()
            if context is not None:
                try:
                    result = context.__aexit__(None, None, None)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
            raise
        except Exception as exc:
            if response is not None:
                await response.aclose()
            if context is not None:
                try:
                    result = context.__aexit__(None, None, None)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    pass
            raise self._transport_exception(exc) from exc
        if response is None:
            raise MCPTransportError("MCP HTTP response is unavailable")
        status = response.status_code

        async def close_failed_response() -> None:
            close = getattr(response, "aclose", None)
            if not callable(close):
                return
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass

        if status == 404 and self._session_id:
            self._session_id = None
            self._reset_message_endpoint()
            self._started = False
            self._notifications.set_capabilities(MCPServerCapabilities())
            self._notifications.notify_connection("disconnected")
            await close_failed_response()
            raise MCPTransportError("MCP HTTP session expired")
        if status < 200 or status >= 300:
            await close_failed_response()
            raise MCPTransportError("MCP HTTP endpoint returned an invalid HTTP status")
        self._capture_session(response)
        if notification:
            return response
        return response

    async def _response_messages(self, response: Any) -> AsyncIterator[Mapping[str, Any]]:
        headers = getattr(response, "headers", {})
        content_type = ""
        try:
            content_type = str(headers.get("content-type", "")).lower()
        except (AttributeError, TypeError):
            pass
        if "text/event-stream" in content_type:
            async for _event_name, raw in iter_sse_events(
                response, max_message_bytes=self.config.max_message_bytes
            ):
                raw = raw.strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    value = json.loads(raw, parse_constant=reject_json_constant)
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise MCPProtocolError("MCP HTTP stream is not valid JSON") from exc
                if isinstance(value, Mapping):
                    yield dict(value)
                else:
                    raise MCPProtocolError("MCP HTTP stream message must be an object")
            return
        reader = getattr(response, "read_content", None)
        if callable(reader):
            raw_bytes = await reader()
        else:
            try:
                raw = getattr(response, "content", b"")
                if inspect.isawaitable(raw):
                    raw = await raw
                if isinstance(raw, str):
                    raw_bytes = raw.encode("utf-8")
                elif isinstance(raw, (bytes, bytearray)):
                    raw_bytes = bytes(raw)
                else:
                    raw_bytes = b""
            except Exception as exc:
                raise MCPTransportError("MCP HTTP response could not be read") from exc
            if len(raw_bytes) > self.config.max_message_bytes:
                raise MCPProtocolError("MCP response exceeds the message limit")
        if not raw_bytes.strip():
            return
        try:
            value = json.loads(raw_bytes.decode("utf-8"), parse_constant=reject_json_constant)
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MCPProtocolError("MCP HTTP response is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise MCPProtocolError("MCP HTTP response must be a JSON object")
        yield dict(value)

    async def _deliver_response(self, value: object) -> None:
        if not isinstance(value, Mapping):
            raise MCPProtocolError("MCP response must be a JSON object")
        if value.get("jsonrpc") != "2.0":
            raise MCPProtocolError("MCP response jsonrpc must be 2.0")
        raw_id = value.get("id")
        if raw_id is None or isinstance(raw_id, bool):
            if "result" in value or "error" in value:
                raise MCPProtocolError("MCP response id is missing")
            self._notifications.dispatch(value.get("method"), value.get("params"))
            return
        if not isinstance(raw_id, int):
            raise MCPProtocolError("MCP response id is invalid")
        future = self._pending.get(raw_id)
        if future is None:
            if raw_id in self._abandoned_ids:
                self._abandoned_ids.remove(raw_id)
                return
            raise MCPProtocolError("MCP response id does not match a pending request")
        if ("result" in value) == ("error" in value):
            raise MCPProtocolError("MCP response must contain exactly one result or error")
        self._pending.pop(raw_id, None)
        if not future.done():
            future.set_result(dict(value))

    async def _request_raw(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        ensure_started: bool = True,
    ) -> Mapping[str, Any]:
        if ensure_started:
            await self.start()
        if self._failure is not None:
            raise MCPTransportError("MCP HTTP transport is unavailable") from self._failure
        try:
            validate_json_value(params or {})
        except (TypeError, ValueError) as exc:
            raise MCPProtocolError("MCP request contains invalid JSON values") from exc
        self._next_id += 1
        request_id = self._next_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Mapping[str, Any]] = loop.create_future()
        self._pending[request_id] = future
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": dict(params or {}),
        }
        try:
            response = await self._send_post(payload)
            try:
                async for value in self._response_messages(response):
                    await self._deliver_response(value)
            finally:
                # httpx 的普通 request 会缓冲响应，但自定义客户端/流式
                # 测试替身可能仍持有连接；无论 JSON-RPC 是否有效都应关闭
                # 该响应，避免每个工具调用泄漏连接。
                close_response = getattr(response, "aclose", None)
                if callable(close_response):
                    try:
                        closed = close_response()
                        if inspect.isawaitable(closed):
                            await closed
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        pass
            # Legacy SSE message endpoints return 202/empty; the response is
            # delivered asynchronously through the long-lived GET stream.
            try:
                result = await asyncio.wait_for(future, self.config.request_timeout_seconds)
            except TimeoutError as exc:
                self._abandon_request(request_id)
                self._pending.pop(request_id, None)
                raise MCPTimeoutError(f"MCP request timed out: {method}") from exc
        except asyncio.CancelledError:
            self._abandon_request(request_id)
            self._pending.pop(request_id, None)
            raise
        except MCPErrorTypes:
            self._abandon_request(request_id)
            self._pending.pop(request_id, None)
            raise
        except BaseException:
            self._abandon_request(request_id)
            self._pending.pop(request_id, None)
            raise
        if "error" in result:
            error = result.get("error")
            if not isinstance(error, Mapping):
                raise MCPProtocolError("MCP JSON-RPC error must be an object")
            code = error.get("code", -32603)
            if isinstance(code, bool) or not isinstance(code, int):
                raise MCPProtocolError("MCP JSON-RPC error code is invalid")
            message = _safe_message(error.get("message"), tuple(self.config.headers.values()))
            raise MCPServerError(code, message)
        result_value = result.get("result")
        if not isinstance(result_value, Mapping):
            raise MCPProtocolError("MCP JSON-RPC result must be an object")
        return dict(result_value)

    async def request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        name = str(method or "").strip()
        if not name or any(char in name for char in "\r\n\x00"):
            raise ValueError("MCP method is invalid")
        return await self._request_raw(name, params)

    async def list_tools(self) -> tuple[MCPTool, ...]:
        tools: list[MCPTool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            result = await self.request("tools/list", {} if cursor is None else {"cursor": cursor})
            raw_tools = result.get("tools", ())
            if not isinstance(raw_tools, list):
                raise MCPProtocolError("MCP tools/list result.tools must be a list")
            for raw_tool in raw_tools:
                if not isinstance(raw_tool, Mapping):
                    raise MCPProtocolError("MCP tools/list tool entry must be an object")
                try:
                    tools.append(MCPTool.from_mapping(raw_tool, server=self.server_name))
                except (TypeError, ValueError) as exc:
                    raise MCPProtocolError("MCP tools/list contains an invalid tool") from exc
            next_cursor = result.get("nextCursor", result.get("next_cursor"))
            if next_cursor is None:
                break
            if not isinstance(next_cursor, str) or not next_cursor.strip():
                raise MCPProtocolError("MCP tools/list nextCursor must be a non-empty string")
            if next_cursor in seen_cursors:
                raise MCPProtocolError("MCP tools/list pagination repeated a cursor")
            seen_cursors.add(next_cursor)
            cursor = next_cursor
        return tuple(tools)

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any] | None = None
    ) -> Mapping[str, Any]:
        remote_name = str(name or "").strip()
        if not remote_name:
            raise ValueError("MCP tool name is required")
        try:
            remote_name = MCPTool(name=remote_name, server=self.server_name).name
        except (TypeError, ValueError) as exc:
            raise MCPProtocolError("MCP tool name is invalid") from exc
        values = dict(arguments or {})
        try:
            validate_json_value(values)
        except (TypeError, ValueError) as exc:
            raise MCPProtocolError("MCP request contains invalid JSON values") from exc
        result = await self.request("tools/call", {"name": remote_name, "arguments": values})
        if "content" in result:
            content = result["content"]
            if not isinstance(content, list) or any(
                not isinstance(item, Mapping) for item in content
            ):
                raise MCPProtocolError("MCP tools/call content must be a list of objects")
        if "structuredContent" in result and not isinstance(result["structuredContent"], Mapping):
            raise MCPProtocolError("MCP tools/call structuredContent must be an object")
        for key in ("isError", "is_error"):
            if key in result and not isinstance(result[key], bool):
                raise MCPProtocolError(f"MCP tools/call {key} must be a boolean")
        return dict(result)

    def _content_limits(self) -> MCPContentLimits:
        return MCPContentLimits.for_transport(
            max_message_bytes=self.config.max_message_bytes,
            request_timeout_seconds=self.config.request_timeout_seconds,
        )

    async def list_resources(self) -> tuple[MCPResource, ...]:
        await self.start()
        if not self.capabilities.resources:
            raise MCPProtocolError("MCP server did not declare the resources capability")
        return await list_mcp_resources(
            self.request,
            server=self.server_name,
            limits=self._content_limits(),
        )

    async def read_resource(self, uri: str) -> MCPReadResourceResult:
        await self.start()
        if not self.capabilities.resources:
            raise MCPProtocolError("MCP server did not declare the resources capability")
        return await read_mcp_resource(
            self.request,
            uri,
            server=self.server_name,
            limits=self._content_limits(),
        )

    async def list_prompts(self) -> tuple[MCPPrompt, ...]:
        await self.start()
        if not self.capabilities.prompts:
            raise MCPProtocolError("MCP server did not declare the prompts capability")
        return await list_mcp_prompts(
            self.request,
            server=self.server_name,
            limits=self._content_limits(),
        )

    async def get_prompt(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> MCPGetPromptResult:
        await self.start()
        if not self.capabilities.prompts:
            raise MCPProtocolError("MCP server did not declare the prompts capability")
        return await get_mcp_prompt(
            self.request,
            name,
            arguments,
            server=self.server_name,
            limits=self._content_limits(),
        )

    async def close(self) -> None:
        async with self._start_lock:
            if self._closed:
                return
            self._closed = True
            await self._notifications.close()
            for future in tuple(self._pending.values()):
                if not future.done():
                    future.set_exception(MCPTransportError("MCP client is closing"))
            self._pending.clear()
            self._abandoned_ids.clear()
            # DELETE 会话前先关闭 SSE 读取任务，避免同一连接上的并发读写。
            await self._close_sse_stream()
            await self._close_http_event_stream()
            if self.config.close_session and self._session_id and self.config.transport == "sse":
                try:
                    target = self.config.url
                    request = self._client.request(
                        "DELETE",
                        target,
                        headers=self._headers(),
                        timeout=min(self.config.request_timeout_seconds, 2.0),
                    )
                    if inspect.isawaitable(request):
                        await request
                except BaseException:
                    pass
            self._session_id = None
            self._reset_message_endpoint()
            self._started = False
            self._initialized = False
            if self._owns_client:
                close = getattr(self._client, "aclose", None)
                if callable(close):
                    result = close()
                    if inspect.isawaitable(result):
                        await result

    async def __aenter__(self) -> HttpMCPClient:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


MCPErrorTypes = (MCPProtocolError, MCPServerError, MCPTimeoutError, MCPTransportError)

# 兼容常见命名习惯，并让配置 UI/第三方扩展无需区分大小写缩写。
HTTPMCPClient = HttpMCPClient
SSEMCPClient = HttpMCPClient
HTTPMCPServerConfig = HttpMCPServerConfig
SSEMCPServerConfig = HttpMCPServerConfig
MCPHttpClient = HttpMCPClient
MCPHttpServerConfig = HttpMCPServerConfig
SseMCPClient = HttpMCPClient
SseMCPServerConfig = HttpMCPServerConfig


__all__ = [
    "HTTPMCPClient",
    "HTTPMCPServerConfig",
    "MCPHttpClient",
    "MCPHttpServerConfig",
    "HttpMCPClient",
    "HttpMCPServerConfig",
    "SSEMCPClient",
    "SSEMCPServerConfig",
    "SseMCPClient",
    "SseMCPServerConfig",
    "iter_sse_events",
    "normalize_mcp_transport",
]
