from __future__ import annotations

import hmac
import ipaddress
import json
import secrets
import threading
from collections import deque
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import isfinite
from typing import Any
from urllib.parse import parse_qs, urlsplit

from .control_surface import friendly_public_text, public_control_state, safe_public_text

__all__ = [
    "LOCAL_CAPABILITY_HEADER",
    "PUBLIC_ACTION_KINDS",
    "LocalWebServer",
    "LocalWebApiServer",
]


LOCAL_CAPABILITY_HEADER = "X-MeaPet-Capability"
"""本地控制面 capability token 的请求头名称。"""


PUBLIC_ACTION_KINDS = frozenset(
    {
        "open_input",
        "open_config",
        "configure_model",
        "select_model_channel",
        "submit_text",
        "stop",
        "retry",
        "approve",
        "grant_session",
        "deny",
        "pet_part",
        "expression",
        "motion",
        "show_pet",
        "toggle_visibility",
        "center_pet",
        "nudge_pet",
        "set_display_size",
        "toggle_window_lock",
        "toggle_always_on_top",
        "toggle_click_through",
        "restore_click_through",
        "read_foreground_window",
        "read_processes",
    }
)

_ACTION_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "open_input": frozenset(),
    "select_model_channel": frozenset({"channel_id", "model"}),
    "submit_text": frozenset({"text"}),
    "pet_part": frozenset({"part"}),
    "expression": frozenset({"name"}),
    "motion": frozenset({"name"}),
    "nudge_pet": frozenset({"direction"}),
    "set_display_size": frozenset({"preset"}),
}
_PUBLIC_PARTS = frozenset({"head", "body", "lower_left", "lower_right"})
_PUBLIC_DIRECTIONS = frozenset({"left", "right", "up", "down"})
_PUBLIC_PRESETS = frozenset({"small", "standard", "large"})
_CAPABILITY_PREFIXES = {"expression": "cap-expression-", "motion": "cap-motion-"}
_ALLOWED_QUERY_KEYS = frozenset({"since", "limit", "stream"})
_MAX_BODY_BYTES = 16 * 1024
_MAX_EVENT_LIMIT = 64
_MAX_EVENT_MESSAGE = 400
_MAX_EVENT_TYPE = 48
_SENSITIVE_RESULT_KEYS = frozenset(
    {
        "operation_id",
        "call_id",
        "approval_id",
        "identity",
        "token",
        "api_key",
        "authorization",
        "command_line",
        "arguments",
        "path",
        "executable",
    }
)


class _InvalidRequest(ValueError):
    """请求未通过公开协议校验。"""


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _json_bytes(value: object) -> bytes:
    """使用严格 JSON 编码；失败时不把内部对象错误返回给客户端。"""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _safe_status(value: object, default: str = "completed") -> str:
    status = safe_public_text(value, limit=32).strip().lower()
    if not status:
        return default
    allowed = {
        "idle",
        "requested",
        "pending",
        "started",
        "accepted",
        "completed",
        "available",
        "updated",
        "saved",
        "validated",
        "reloaded",
        "approval_required",
        "tool_limit",
        "degraded",
        "failed",
        "error",
        "unavailable",
        "cancelled",
        "canceled",
        "denied",
        "rejected",
        "timeout",
        "restart_required",
        "ok",
    }
    return status if status in allowed else "unavailable"


def _safe_event(event_id: int, value: Mapping[str, Any]) -> dict[str, object]:
    """把宿主事件压缩为不会泄露调用参数的公开事件。"""

    raw_type = safe_public_text(value.get("type", "state"), limit=_MAX_EVENT_TYPE)
    event_type = "".join(char for char in raw_type if char.isalnum() or char in "._-")
    event_type = event_type[:_MAX_EVENT_TYPE] or "state"
    result: dict[str, object] = {
        "id": event_id,
        "type": event_type,
        "updated_at": 0.0,
    }
    raw_status = value.get("status", value.get("state_status"))
    if raw_status is not None:
        result["status"] = _safe_status(raw_status)
    message = friendly_public_text(value.get("message"), limit=_MAX_EVENT_MESSAGE)
    if message:
        result["message"] = message
    for key in ("part", "zone"):
        item = safe_public_text(value.get(key), limit=32).strip().lower()
        if item in _PUBLIC_PARTS:
            result[key] = item
    for key in ("mood", "expression", "motion"):
        item = safe_public_text(value.get(key), limit=64)
        if item:
            result[key] = item
    raw_revision = value.get("revision")
    if isinstance(raw_revision, bool):
        raw_revision = None
    if isinstance(raw_revision, (int, float)) and isfinite(float(raw_revision)):
        result["revision"] = max(0, int(raw_revision))
    raw_updated = value.get("updated_at", value.get("at"))
    if isinstance(raw_updated, (int, float)) and not isinstance(raw_updated, bool):
        if isfinite(float(raw_updated)):
            result["updated_at"] = float(raw_updated)
    state = value.get("state")
    if isinstance(state, Mapping):
        # 状态事件必须走同一公开投影，不能把宿主原始快照嵌入事件。
        result["state"] = public_control_state(state)
    data = value.get("data")
    if isinstance(data, Mapping) and "state" not in result:
        # 一般事件只允许有限的用户可见标量；复杂数据以状态投影处理。
        safe_data: dict[str, object] = {}
        for key in ("part", "zone", "mood", "expression", "motion", "status", "message"):
            if key in data:
                safe_data[key] = data[key]
        if safe_data:
            result["data"] = _safe_event_data(safe_data)
    return result


def _safe_event_data(value: Mapping[str, Any]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key in ("part", "zone"):
        item = safe_public_text(value.get(key), limit=32).strip().lower()
        if item in _PUBLIC_PARTS:
            result[key] = item
    for key in ("mood", "expression", "motion"):
        item = safe_public_text(value.get(key), limit=64)
        if item:
            result[key] = item
    if value.get("status") is not None:
        result["status"] = _safe_status(value.get("status"))
    message = friendly_public_text(value.get("message"), limit=_MAX_EVENT_MESSAGE)
    if message:
        result["message"] = message
    return result


def _safe_action_result(value: object) -> dict[str, object]:
    """过滤宿主回执，确保 HTTP 层不会重新暴露内部调用标识。"""

    if not isinstance(value, Mapping):
        if isinstance(value, bool):
            return {
                "status": "completed" if value else "unavailable",
                "message": "操作已完成" if value else "操作暂时不可用",
            }
        return {"status": "completed", "message": "操作已提交"}
    result: dict[str, object] = {}
    raw_status = value.get("status", value.get("state", "completed"))
    result["status"] = _safe_status(raw_status)
    for key in ("status_label", "message", "reason", "detail"):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) and key not in _SENSITIVE_RESULT_KEYS:
            text = friendly_public_text(item, limit=400)
            if text:
                result[key] = text
    for key in ("part", "zone"):
        item = safe_public_text(value.get(key), limit=32).strip().lower()
        if item in _PUBLIC_PARTS:
            result[key] = item
    for key in ("mood", "expression", "motion"):
        item = safe_public_text(value.get(key), limit=64)
        if item:
            result[key] = item
    affection = value.get("affection")
    if isinstance(affection, Mapping):
        result["affection"] = {
            key: friendly_public_text(affection.get(key), limit=64)
            for key in ("current", "applied", "tier", "status")
            if friendly_public_text(affection.get(key), limit=64)
        }
    if "message" not in result and "reason" in result:
        result["message"] = result["reason"]
    return result


def _validate_capability_token(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("capability token must be text")
    token = value.strip()
    # 明确拒绝短测试令牌，避免宿主误把可猜字符串用于真实监听。
    if len(token) < 32 or len(token) > 256:
        raise ValueError("capability token length is invalid")
    if any(char.isspace() or ord(char) < 0x21 or ord(char) > 0x7E for char in token):
        raise ValueError("capability token contains unsupported characters")
    return token


def _validate_action_payload(kind: str, payload: object) -> dict[str, object]:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise _InvalidRequest("payload must be an object")
    expected = _ACTION_PAYLOAD_KEYS.get(kind, frozenset())
    if any(not isinstance(key, str) or key not in expected for key in payload):
        raise _InvalidRequest("payload contains unsupported fields")
    result: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(value, (str, int, float, bool)) or isinstance(value, bool):
            raise _InvalidRequest("payload value is invalid")
        if isinstance(value, float) and not isfinite(value):
            raise _InvalidRequest("payload value is invalid")
        result[key] = value
    if kind == "submit_text":
        text = result.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise _InvalidRequest("text is invalid")
        result["text"] = text.strip()
    elif kind == "select_model_channel":
        channel_id = str(result.get("channel_id", "")).strip()
        model = str(result.get("model", "")).strip()
        if not channel_id and not model:
            raise _InvalidRequest("channel_id or model is required")
        if channel_id:
            if len(channel_id) > 128 or any(
                char.isspace() or ord(char) < 0x20 for char in channel_id
            ):
                raise _InvalidRequest("channel_id is invalid")
            result["channel_id"] = channel_id
        if "model" in result:
            model = str(result["model"]).strip()
            if (
                not model
                or len(model) > 128
                or any(char.isspace() or ord(char) < 0x20 for char in model)
            ):
                raise _InvalidRequest("model is invalid")
            result["model"] = model
    elif kind == "pet_part":
        part = str(result.get("part", "")).strip().lower()
        if part not in _PUBLIC_PARTS:
            raise _InvalidRequest("part is invalid")
        result["part"] = part
    elif kind in {"expression", "motion"}:
        name = str(result.get("name", "")).strip().lower()
        prefix = _CAPABILITY_PREFIXES[kind]
        suffix = name[len(prefix) :] if name.startswith(prefix) else ""
        if not suffix.isdigit() or len(suffix) > 2 or int(suffix) >= 32:
            raise _InvalidRequest("capability is invalid")
        result["name"] = name
    elif kind == "nudge_pet":
        direction = str(result.get("direction", "")).strip().lower()
        if direction not in _PUBLIC_DIRECTIONS:
            raise _InvalidRequest("direction is invalid")
        result["direction"] = direction
    elif kind == "set_display_size":
        preset = str(result.get("preset", "")).strip().lower()
        if preset not in _PUBLIC_PRESETS:
            raise _InvalidRequest("preset is invalid")
        result["preset"] = preset
    return result


class _RequestHandler(BaseHTTPRequestHandler):
    """单次请求处理器；不保留客户端连接。"""

    protocol_version = "HTTP/1.0"
    server_version = "MeaPetLocal/1"
    sys_version = ""

    @property
    def api(self) -> LocalWebServer:
        return self.server.api  # type: ignore[attr-defined, no-any-return]

    def log_message(self, _format: str, *args: object) -> None:
        # 不记录完整 URL，避免未来误把查询内容或令牌写入日志。
        try:
            self.api._logger(
                "local web request method=%s path=%s",
                self.command,
                self.path.split("?", 1)[0],
            )
        except Exception:
            return

    def _respond(
        self,
        status: int,
        value: object,
        *,
        content_type: str = "application/json",
    ) -> None:
        try:
            body = (
                _json_bytes(value) if content_type.startswith("application/json") else bytes(value)
            )
        except (TypeError, ValueError, OverflowError):
            status = int(HTTPStatus.INTERNAL_SERVER_ERROR)
            body = _json_bytes(
                {
                    "ok": False,
                    "error": {
                        "code": "serialization_error",
                        "message": "响应暂时不可用",
                    },
                }
            )
            content_type = "application/json; charset=utf-8"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return

    def _error(self, status: int, code: str, message: str) -> None:
        self._respond(status, {"ok": False, "error": {"code": code, "message": message}})

    def _security_check(self) -> bool:
        client_host = str(self.client_address[0] if self.client_address else "")
        try:
            is_loopback = ipaddress.ip_address(client_host).is_loopback
        except ValueError:
            is_loopback = False
        if not is_loopback:
            self._error(HTTPStatus.FORBIDDEN, "local_only", "只允许本机访问")
            return False
        if any(self.headers.get(name) for name in ("Forwarded", "X-Forwarded-For", "X-Real-IP")):
            self._error(HTTPStatus.FORBIDDEN, "forwarded_source_denied", "不接受代理转发来源")
            return False
        host_header = self.headers.get("Host", "").strip()
        if host_header and not self.api._host_header_allowed(host_header):
            self._error(HTTPStatus.FORBIDDEN, "host_denied", "主机来源不受支持")
            return False
        origin = self.headers.get("Origin", "").strip()
        if origin and not self.api._origin_allowed(origin):
            self._error(HTTPStatus.FORBIDDEN, "origin_denied", "页面来源不受支持")
            return False
        fetch_site = self.headers.get("Sec-Fetch-Site", "").strip().lower()
        if fetch_site == "cross-site":
            self._error(HTTPStatus.FORBIDDEN, "cross_site_denied", "跨站请求已拒绝")
            return False
        return True

    def _authenticate(self) -> bool:
        values = self.headers.get_all(LOCAL_CAPABILITY_HEADER, [])
        authorization = self.headers.get_all("Authorization", [])
        supplied: list[str] = []
        supplied.extend(item.strip() for item in values if item.strip())
        for item in authorization:
            prefix, separator, token = item.partition(" ")
            if prefix.lower() != "bearer" or not separator or not token.strip():
                self._error(HTTPStatus.UNAUTHORIZED, "invalid_token", "访问令牌无效")
                return False
            supplied.append(token.strip())
        if len(supplied) != 1 or not hmac.compare_digest(supplied[0], self.api.capability_token):
            self._error(HTTPStatus.UNAUTHORIZED, "invalid_token", "访问令牌无效")
            return False
        return True

    def _parse_path(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlsplit(self.path)
        if parsed.scheme or parsed.netloc or not parsed.path.startswith("/"):
            raise _InvalidRequest("absolute URL is not allowed")
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
        if any(key not in _ALLOWED_QUERY_KEYS for key in query):
            raise _InvalidRequest("query field is not allowed")
        return parsed.path, query

    def _query_int(
        self,
        query: Mapping[str, list[str]],
        key: str,
        default: int,
        maximum: int,
    ) -> int:
        values = query.get(key, [])
        if not values:
            return default
        if len(values) != 1 or not values[0].isdigit():
            raise _InvalidRequest("query value is invalid")
        value = int(values[0])
        if value < 0 or value > maximum:
            raise _InvalidRequest("query value is invalid")
        return value

    def do_GET(self) -> None:  # noqa: N802
        if not self._security_check() or not self._authenticate():
            return
        try:
            path, query = self._parse_path()
            if path == "/api/state":
                if query:
                    raise _InvalidRequest("state query is not allowed")
                self._respond(HTTPStatus.OK, self.api.public_state())
                return
            if path == "/api/events":
                if not self.api.events_enabled:
                    self._error(HTTPStatus.NOT_FOUND, "events_disabled", "事件通道未启用")
                    return
                since = self._query_int(query, "since", 0, 2**63 - 1)
                limit = self._query_int(query, "limit", 32, _MAX_EVENT_LIMIT)
                stream = query.get("stream", [""])[0].strip().lower()
                if stream not in {"", "poll", "sse", "1", "true"}:
                    raise _InvalidRequest("stream value is invalid")
                events, cursor = self.api.events_snapshot(since=since, limit=limit)
                if (
                    stream in {"sse", "1", "true"}
                    or "text/event-stream" in self.headers.get("Accept", "").lower()
                ):
                    self._respond_sse(events, cursor)
                else:
                    self._respond(
                        HTTPStatus.OK,
                        {"events": events, "cursor": cursor, "has_more": len(events) >= limit},
                    )
                return
            self._error(HTTPStatus.NOT_FOUND, "not_found", "接口不存在")
        except _InvalidRequest as exc:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
        except Exception:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "unavailable", "控制通道暂时不可用")

    def _read_json_body(self) -> Mapping[str, object]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise _InvalidRequest("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdigit():
            raise _InvalidRequest("Content-Length is required")
        length = int(raw_length)
        if length < 2 or length > _MAX_BODY_BYTES:
            raise _InvalidRequest("request body is too large")
        body = self.rfile.read(length)
        if len(body) != length:
            raise _InvalidRequest("request body is incomplete")
        try:
            value = json.loads(body.decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            raise _InvalidRequest("request body is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise _InvalidRequest("request body must be an object")
        return value

    def do_POST(self) -> None:  # noqa: N802
        if not self._security_check() or not self._authenticate():
            return
        try:
            path, query = self._parse_path()
            if query:
                raise _InvalidRequest("action query is not allowed")
            if path != "/api/action":
                self._error(HTTPStatus.NOT_FOUND, "not_found", "接口不存在")
                return
            body = self._read_json_body()
            if set(body) != {"kind", "payload"} and set(body) != {"kind"}:
                raise _InvalidRequest("action body fields are invalid")
            raw_kind = body.get("kind")
            if not isinstance(raw_kind, str):
                raise _InvalidRequest("action kind is invalid")
            kind = raw_kind.strip().lower()
            if kind not in PUBLIC_ACTION_KINDS:
                raise _InvalidRequest("action is not allowed")
            payload = _validate_action_payload(kind, body.get("payload", {}))
            result = self.api.invoke_action(kind, payload)
            self._respond(HTTPStatus.OK, _safe_action_result(result))
        except _InvalidRequest as exc:
            self._error(HTTPStatus.BAD_REQUEST, "invalid_request", str(exc))
        except Exception:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, "unavailable", "操作暂时不可用")

    def _respond_sse(self, events: list[dict[str, object]], cursor: int) -> None:
        rows: list[str] = []
        if not events:
            rows.append(f'event: keepalive\ndata: {{"events":[],"cursor":{cursor}}}\n\n')
        else:
            for event in events:
                event_name = str(event.get("type", "state"))
                payload = _json_bytes(event).decode("utf-8")
                rows.append(f"id: {event['id']}\nevent: {event_name}\ndata: {payload}\n\n")
        body = "".join(rows).encode("utf-8")
        self._respond(HTTPStatus.OK, body, content_type="text/event-stream; charset=utf-8")


class _ApiHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], api: LocalWebServer) -> None:
        self.api = api
        super().__init__(address, _RequestHandler)


class LocalWebServer:
    """可选、本机限定的公开控制面服务器。

    ``start`` 之前不会打开端口；``port=0`` 让操作系统分配临时端口，适合测试。
    ``action_callback`` 只负责把已验证的公开动作交给 Qt/runtime 组合根，不能绕过
    现有审批和权限服务。
    """

    def __init__(
        self,
        state_provider: Callable[[], Mapping[str, Any] | None],
        action_callback: Callable[[str, Mapping[str, object]], object],
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        capability_token: str | None = None,
        enable_events: bool = True,
        event_limit: int = 64,
        logger: Callable[..., object] | None = None,
    ) -> None:
        if host != "127.0.0.1":
            raise ValueError("local web server must bind exactly to 127.0.0.1")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if not callable(state_provider) or not callable(action_callback):
            raise TypeError("state_provider and action_callback must be callable")
        if isinstance(event_limit, bool) or not isinstance(event_limit, int):
            raise ValueError("event_limit must be a positive integer")
        if event_limit < 1 or event_limit > _MAX_EVENT_LIMIT:
            raise ValueError(f"event_limit must be between 1 and {_MAX_EVENT_LIMIT}")
        token = (
            secrets.token_urlsafe(32)
            if capability_token is None
            else _validate_capability_token(capability_token)
        )
        self._state_provider = state_provider
        self._action_callback = action_callback
        self._host = host
        self._requested_port = port
        self._capability_token = token
        self.events_enabled = bool(enable_events)
        self._event_limit = event_limit
        self._events: deque[dict[str, object]] = deque(maxlen=event_limit)
        self._event_sequence = 0
        self._lock = threading.RLock()
        self._httpd: _ApiHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._logger = logger if callable(logger) else lambda *_args, **_kwargs: None

    @property
    def capability_token(self) -> str:
        """返回当前进程的 capability token；服务器不会通过 HTTP 回显它。"""

        return self._capability_token

    @property
    def token(self) -> str:
        """``capability_token`` 的简短只读别名，便于本地宿主接入。"""

        return self._capability_token

    @property
    def address(self) -> tuple[str, int] | None:
        with self._lock:
            if self._httpd is None:
                return None
            host, port = self._httpd.server_address[:2]
            return str(host), int(port)

    @property
    def base_url(self) -> str | None:
        address = self.address
        return f"http://{address[0]}:{address[1]}" if address is not None else None

    @property
    def url(self) -> str | None:
        """``base_url`` 的只读别名。"""

        return self.base_url

    @property
    def is_running(self) -> bool:
        with self._lock:
            return bool(
                self._httpd is not None and self._thread is not None and self._thread.is_alive()
            )

    def start(self) -> LocalWebServer:
        with self._lock:
            if self._httpd is not None:
                return self
            httpd = _ApiHTTPServer((self._host, self._requested_port), self)
            thread = threading.Thread(
                target=httpd.serve_forever,
                name="meapet-local-web",
                daemon=True,
            )
            self._httpd = httpd
            self._thread = thread
            thread.start()
        return self

    def stop(self, *, timeout: float = 2.0) -> None:
        with self._lock:
            httpd = self._httpd
            thread = self._thread
            self._httpd = None
            self._thread = None
        if httpd is None:
            return
        try:
            httpd.shutdown()
        finally:
            httpd.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, float(timeout)))

    def __enter__(self) -> LocalWebServer:
        return self.start()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.stop()

    def public_state(self) -> dict[str, object]:
        try:
            value = self._state_provider()
            return public_control_state(value)
        except Exception:
            raise RuntimeError("state provider unavailable") from None

    def invoke_action(self, kind: str, payload: Mapping[str, object]) -> object:
        return self._action_callback(kind, dict(payload))

    def publish_event(self, event: Mapping[str, Any]) -> int:
        """加入一个脱敏事件并返回单调递增事件号。"""

        if not isinstance(event, Mapping):
            raise TypeError("event must be a mapping")
        with self._lock:
            self._event_sequence += 1
            item = _safe_event(self._event_sequence, event)
            self._events.append(item)
            return self._event_sequence

    def publish_state_event(self, *, event_type: str = "state") -> int:
        """读取当前公开状态并加入事件队列。"""

        state = self.public_state()
        return self.publish_event({"type": event_type, "state": state})

    def events_snapshot(
        self,
        *,
        since: int = 0,
        limit: int = 32,
    ) -> tuple[list[dict[str, object]], int]:
        if isinstance(since, bool) or not isinstance(since, int) or since < 0:
            raise ValueError("since must be a non-negative integer")
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= _MAX_EVENT_LIMIT
        ):
            raise ValueError("limit is invalid")
        with self._lock:
            rows = [dict(item) for item in self._events if int(item["id"]) > since]
            cursor = self._event_sequence
        return rows[:limit], cursor

    def _host_header_allowed(self, value: str) -> bool:
        host = value.strip().lower()
        if host.startswith("["):
            return False
        if ":" in host:
            hostname, separator, port = host.rpartition(":")
            if not separator or not port.isdigit():
                return False
            host = hostname
            if self.address is not None and int(port) != self.address[1]:
                return False
        return host in {"127.0.0.1", "localhost"}

    @staticmethod
    def _origin_allowed(value: str) -> bool:
        origin = value.strip().lower().rstrip("/")
        return (
            origin in {"null", "http://127.0.0.1", "http://localhost"}
            or (origin.startswith("http://127.0.0.1:") and origin.rsplit(":", 1)[-1].isdigit())
            or (origin.startswith("http://localhost:") and origin.rsplit(":", 1)[-1].isdigit())
        )


LocalWebApiServer = LocalWebServer
