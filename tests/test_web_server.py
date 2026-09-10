from __future__ import annotations

import http.client
import json
from collections.abc import Mapping

import pytest

from gui.web.server import (
    LOCAL_CAPABILITY_HEADER,
    LocalWebServer,
    _safe_action_result,
    _safe_event,
)


def _request(
    server: LocalWebServer,
    method: str,
    path: str,
    *,
    token: str | None = None,
    headers: Mapping[str, str] | None = None,
    payload: object | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    address = server.address
    assert address is not None
    connection = http.client.HTTPConnection(address[0], address[1], timeout=3)
    request_headers = dict(headers or {})
    if token is not None:
        request_headers[LOCAL_CAPABILITY_HEADER] = token
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
        request_headers["Content-Length"] = str(len(body))
    connection.request(method, path, body=body, headers=request_headers)
    response = connection.getresponse()
    result = (response.status, response.read(), dict(response.headers.items()))
    connection.close()
    return result


@pytest.fixture
def running_server():
    calls: list[tuple[str, dict[str, object]]] = []

    def state_provider() -> dict[str, object]:
        return {
            "renderer": {"backend": "web_live2d", "available": True},
            "interaction": {"phase": "idle"},
            "feedback": {"part": "head", "phrase": "摸摸头～"},
        }

    def action_callback(kind: str, payload: Mapping[str, object]) -> dict[str, object]:
        calls.append((kind, dict(payload)))
        return {
            "status": "completed",
            "message": "部位反馈已完成",
            "operation_id": "must-not-cross-http-boundary",
            "part": payload.get("part"),
        }

    server = LocalWebServer(state_provider, action_callback, port=0, event_limit=3)
    server.start()
    try:
        yield server, calls
    finally:
        server.stop()


def test_server_is_opt_in_and_binds_loopback_with_random_token() -> None:
    server = LocalWebServer(lambda: {}, lambda _kind, _payload: {})
    assert not server.is_running
    assert server.address is None
    assert server.base_url is None
    assert len(server.capability_token) >= 32
    assert server.capability_token == server.token
    with pytest.raises(ValueError, match="127.0.0.1"):
        LocalWebServer(lambda: {}, lambda _kind, _payload: {}, host="0.0.0.0")


def test_state_requires_header_token_and_returns_public_projection(running_server) -> None:
    server, _calls = running_server
    status, body, headers = _request(server, "GET", "/api/state")
    assert status == 401
    assert json.loads(body)["error"]["code"] == "invalid_token"
    assert headers["Cache-Control"] == "no-store"

    status, body, _headers = _request(server, "GET", "/api/state", token=server.token)
    assert status == 200
    state = json.loads(body)
    assert state["renderer"]["label"] == "Web Live2D"
    assert state["feedback"]["part"] == "head"
    assert "operation_id" not in body.decode("utf-8")


def test_action_whitelist_validates_payload_and_redacts_callback_result(running_server) -> None:
    server, calls = running_server
    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={"kind": "pet_part", "payload": {"part": "head"}},
    )
    assert status == 200
    result = json.loads(body)
    assert result == {"status": "completed", "message": "部位反馈已完成", "part": "head"}
    assert calls == [("pet_part", {"part": "head"})]

    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={"kind": "open_input"},
    )
    assert status == 200
    assert calls[-1] == ("open_input", {})

    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={"kind": "expression", "payload": {"name": "ParamAngleX"}},
    )
    assert status == 400
    assert json.loads(body)["error"]["code"] == "invalid_request"
    assert len(calls) == 2

    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={"kind": "open_input"},
    )
    assert status == 200
    assert calls[-1] == ("open_input", {})

    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={"kind": "select_model_channel", "payload": {"channel_id": "backup"}},
    )
    assert status == 200
    assert calls[-1] == ("select_model_channel", {"channel_id": "backup"})
    assert json.loads(body)["status"] == "completed"

    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={
            "kind": "select_model_channel",
            "payload": {"channel_id": "backup", "model": "backup-model"},
        },
    )
    assert status == 200
    assert calls[-1] == (
        "select_model_channel",
        {"channel_id": "backup", "model": "backup-model"},
    )
    assert json.loads(body)["status"] == "completed"

    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={
            "kind": "select_model_channel",
            "payload": {"channel_id": "backup", "api_key": "private"},
        },
    )
    assert status == 400
    assert "private" not in body.decode("utf-8")

    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={
            "kind": "open_config",
            "payload": {},
        },
    )
    assert status == 200
    assert calls[-1] == ("open_config", {})
    assert "/private" not in body.decode("utf-8")

    status, body, _headers = _request(
        server,
        "POST",
        "/api/action",
        token=server.token,
        payload={
            "kind": "pet_part",
            "payload": {"part": "head", "approval_id": "private"},
        },
    )
    assert status == 400
    assert "private" not in body.decode("utf-8")


def test_events_are_bounded_and_support_polling_and_one_shot_sse(running_server) -> None:
    server, _calls = running_server
    for index in range(5):
        server.publish_event(
            {
                "type": "feedback",
                "part": "head",
                "message": f"第 {index} 次",
                "operation_id": "private",
            }
        )
    status, body, _headers = _request(
        server,
        "GET",
        "/api/events?since=0&limit=64",
        token=server.token,
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["cursor"] == 5
    assert [item["id"] for item in payload["events"]] == [3, 4, 5]
    assert "operation_id" not in body.decode("utf-8")

    status, body, headers = _request(
        server,
        "GET",
        "/api/events?since=3&stream=sse",
        token=server.token,
        headers={"Accept": "text/event-stream"},
    )
    assert status == 200
    assert headers["Content-Type"].startswith("text/event-stream")
    text = body.decode("utf-8")
    assert "event: feedback" in text
    assert "id: 4" in text and "id: 5" in text


def test_source_and_query_boundaries_are_rejected(running_server) -> None:
    server, _calls = running_server
    status, body, _headers = _request(
        server,
        "GET",
        "/api/state?token=" + server.token,
        token=server.token,
    )
    assert status == 400
    assert "token" not in body.decode("utf-8").lower()

    status, body, _headers = _request(
        server,
        "GET",
        "/api/state",
        token=server.token,
        headers={"Origin": "https://outside.invalid"},
    )
    assert status == 403
    assert json.loads(body)["error"]["code"] == "origin_denied"


def test_events_can_be_disabled() -> None:
    server = LocalWebServer(lambda: {}, lambda _kind, _payload: {}, enable_events=False)
    server.start()
    try:
        status, body, _headers = _request(
            server,
            "GET",
            "/api/events",
            token=server.token,
        )
        assert status == 404
        assert json.loads(body)["error"]["code"] == "events_disabled"
    finally:
        server.stop()


def test_server_stop_is_idempotent_and_releases_listener() -> None:
    server = LocalWebServer(lambda: {}, lambda _kind, _payload: {}, port=0)
    assert not server.is_running
    server.start()
    address = server.address
    assert address is not None
    assert server.is_running
    server.stop(timeout=2)
    assert server.address is None
    assert not server.is_running
    server.stop(timeout=2)

    # 同一端口可重新绑定，证明 stop 已关闭底层监听套接字；新的实例仍
    # 生成独立 capability，旧令牌不会被生命周期复用。
    replacement = LocalWebServer(lambda: {}, lambda _kind, _payload: {}, port=address[1])
    try:
        replacement.start()
        assert replacement.address == address
        assert replacement.capability_token != server.capability_token
    finally:
        replacement.stop(timeout=2)


def test_server_rejects_duplicate_capability_headers(running_server) -> None:
    server, _calls = running_server
    status, body, _headers = _request(
        server,
        "GET",
        "/api/state",
        token=server.token,
        headers={"Authorization": f"Bearer {server.token}"},
    )
    assert status == 401
    assert json.loads(body)["error"]["code"] == "invalid_token"


def test_http_projection_translates_stream_markers_before_returning_messages() -> None:
    result = _safe_action_result(
        {
            "status": "failed",
            "message": "[failed] 操作未完成",
            "detail": "[timeout] 请求超时",
            "api_key": "must-not-cross-boundary",
        }
    )
    assert result["message"] == "未完成：操作未完成"
    assert result["detail"] == "响应超时：请求超时"
    assert "api_key" not in json.dumps(result, ensure_ascii=False)

    event = _safe_event(
        1,
        {"type": "operation", "message": "[approval_required] 请确认", "call_id": "private"},
    )
    assert event["message"] == "等待确认：请确认"
    assert "call_id" not in json.dumps(event, ensure_ascii=False)
