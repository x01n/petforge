"""第 9 轮方向 4（桌面生命周期与测试门禁）E 项：关闭期守卫回归测试。

三个 GUI 文件属于真实 Qt 区，构造窗口需要显示资源；此处用源码结构断言
加无绑定方法调用验证语义，保持在快层（no xvfb/webengine）可执行。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _PROJECT_ROOT / "src" / "gui" / "qt6"


def _function_body_source(source: str, function_name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            body = ast.get_source_segment(source, node) or ""
            return body
    raise AssertionError(f"function {function_name} not found")


def test_e_web_host_relay_forwards_failure_even_when_closing() -> None:
    """关闭阶段仍必须转达 renderer 失败终态，避免调用方永久 pending。"""

    from gui.qt6.web_host import WebPetHost

    source = (_SOURCE_ROOT / "web_host.py").read_text(encoding="utf-8")
    body = _function_body_source(source, "_on_renderer_model_reload")
    relay_predicate = (
        'callback is None or ((self._closing or self._shutdown) and status == "available")'
    )
    assert relay_predicate in body

    received: list[dict[str, Any]] = []

    def fake_callback(value: object) -> None:
        received.append(dict(value))  # type: ignore[arg-type]

    closing_host = SimpleNamespace(
        _closing=True,
        _shutdown=False,
        _model_reload_callback=fake_callback,
        _invalidate_geometry=lambda: received.append({"geometry": True}),
        _surface_mask_capture_generation=0,
        _surface_mask_capture_pending=False,
        _surface_mask_capture_failures=0,
        _last_cursor_target=None,
        _last_window_position=None,
        _deferred_page_resize=None,
        _schedule_interaction_refresh=lambda: received.append({"refresh": True}),
        _surface_mask_enabled=False,
        _schedule_surface_mask_retries=lambda: received.append({"mask": True}),
    )
    WebPetHost._on_renderer_model_reload(  # type: ignore[attr-defined]
        closing_host, {"status": "failed", "reason": "model reload request failed"}
    )
    assert len(received) == 1
    assert received[0].get("status") == "failed"
    assert "geometry" not in received[0]

    normal_host = SimpleNamespace(
        _closing=False,
        _shutdown=False,
        _model_reload_callback=fake_callback,
        _invalidate_geometry=lambda: None,
        _surface_mask_capture_generation=0,
        _surface_mask_capture_pending=False,
        _surface_mask_capture_failures=0,
        _last_cursor_target=None,
        _last_window_position=None,
        _deferred_page_resize=None,
        _schedule_interaction_refresh=lambda: None,
        _surface_mask_enabled=False,
        _schedule_surface_mask_retries=lambda: None,
    )
    received.clear()
    WebPetHost._on_renderer_model_reload(  # type: ignore[attr-defined]
        normal_host, {"status": "available"}
    )
    assert any(item.get("status") == "available" for item in received)


def test_e_opengl_shutdown_explicitly_stops_drag_watchdog() -> None:
    """shutdown 必须显式停止并清空 _drag_watchdog_timer。"""

    source = (_SOURCE_ROOT / "opengl_host.py").read_text(encoding="utf-8")
    body = _function_body_source(source, "shutdown")
    assert "_drag_watchdog_timer" in body
    assert "watchdog.stop()" in body or ".stop()" in body


def test_e_web_console_restore_retry_checks_shutdown_before_requeue() -> None:
    """恢复重试不再裸 lambda 捕获 self；重排前必须检查 _shutdown。"""

    source = (_SOURCE_ROOT / "web_console.py").read_text(encoding="utf-8")
    assert "retry_restore_after_focus" in source
    body = _function_body_source(source, "_restore_after_focus")
    assert "if self._shutdown" in body


def test_e_console_input_visibility_deferred_callback_checks_allow_close() -> None:
    """挂在事件队列里的输入区回调首行必须检查关闭标志。"""

    source = (_SOURCE_ROOT / "console.py").read_text(encoding="utf-8")
    body = _function_body_source(source, "_ensure_conversation_input_visible")
    index = body.find("if self._allow_close:")
    assert index >= 0
    assert index < body.find("self.findChild")
