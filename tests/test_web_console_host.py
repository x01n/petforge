from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_web_bridge_sanitizes_operation_results() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda _kind, _payload: {
            "status": "requested",
            "approval_id": "private-approval",
            "call_id": "private-call",
            "reason": "等待确认",
        },
    )
    result = json.loads(bridge.invoke("approve", "{}"))
    encoded = json.dumps(result, ensure_ascii=False)
    assert result["status"] == "requested"
    assert result["message"] == "等待确认"
    assert result["status_label"] == "已提交"
    assert "private-approval" not in encoded
    assert "private-call" not in encoded
    assert "operation_id" not in result


def test_web_bridge_redacts_sensitive_result_fields_and_hides_operation_id() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda _kind, _payload: {
            "status": "failed",
            "detail": "api_key=private-value",
            "message": "Authorization: Bearer private-token",
            "operation_id": "approval-internal-id",
            "name": "ParamAngleX",
        },
    )
    result = json.loads(bridge.invoke("expression", "{}"))
    encoded = json.dumps(result, ensure_ascii=False)
    for marker in (
        "api_key",
        "Authorization",
        "private-token",
        "approval-internal-id",
        "ParamAngleX",
    ):
        assert marker not in encoded
    assert result["detail"] == "敏感详情已隐藏"
    assert result["message"] == "敏感详情已隐藏"
    assert result["status_label"] == "执行失败"
    assert "operation_id" not in result
    assert "name" not in result


def test_web_bridge_keeps_only_click_route_direction_payload() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    seen: list[tuple[str, dict[str, object]]] = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda kind, payload: seen.append((kind, dict(payload))) or {"status": "completed"},
    )
    bridge.invoke(
        "nudge_pet",
        json.dumps(
            {
                "direction": "left",
                "x": 100,
                "command_line": "secret",
                "name": "ParamAngleX",
            }
        ),
    )
    assert seen == [("nudge_pet", {"direction": "left"})]


def test_web_bridge_opens_config_center_without_accepting_payload() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    seen: list[tuple[str, dict[str, object]]] = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda kind, payload: (
            seen.append((kind, dict(payload)))
            or {"status": "completed", "message": "已打开配置中心"}
        ),
    )
    result = json.loads(
        bridge.invoke(
            "open_config",
            json.dumps({"path": "/private/config.yaml", "api_key": "private"}),
        )
    )
    assert result["status"] == "completed"
    assert seen == [("open_config", {})]
    assert "/private" not in json.dumps(result, ensure_ascii=False)


def test_web_bridge_keeps_only_safe_model_channel_selector() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    seen: list[tuple[str, dict[str, object]]] = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda kind, payload: seen.append((kind, dict(payload))) or {"status": "updated"},
    )
    bridge.invoke(
        "select_model_channel",
        json.dumps({"channel_id": "backup", "api_key": "private", "base_url": "https://private"}),
    )
    assert seen == [("select_model_channel", {"channel_id": "backup"})]


def test_web_bridge_accepts_a_bounded_model_selector() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    seen: list[tuple[str, dict[str, object]]] = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda kind, payload: seen.append((kind, dict(payload))) or {"status": "updated"},
    )
    bridge.invoke(
        "select_model_channel",
        json.dumps({"channel_id": "backup", "model": "model-v2", "token": "secret"}),
    )
    assert seen == [("select_model_channel", {"channel_id": "backup", "model": "model-v2"})]


def test_web_bridge_rejects_cross_kind_capability_and_action_fields() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    seen: list[tuple[str, dict[str, object]]] = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda kind, payload: seen.append((kind, dict(payload))) or {"status": "completed"},
    )

    expression_result = json.loads(
        bridge.invoke(
            "expression",
            json.dumps(
                {
                    "name": "cap-motion-1",
                    "text": "must be dropped",
                    "direction": "left",
                }
            ),
        )
    )
    motion_result = json.loads(
        bridge.invoke(
            "motion",
            json.dumps({"name": "cap-expression-0", "preset": "large"}),
        )
    )
    bridge.invoke(
        "read_processes",
        json.dumps({"target": "model_setup", "mode": "click", "text": "must be dropped"}),
    )

    assert expression_result["status"] == "completed"
    assert motion_result["status"] == "completed"
    assert seen == [
        ("expression", {}),
        ("motion", {}),
        ("read_processes", {}),
    ]


def test_web_bridge_validates_structured_actions_and_renderer_selection() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    seen: list[tuple[str, dict[str, object]]] = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda kind, payload: seen.append((kind, dict(payload))) or {"status": "completed"},
    )
    bridge.invoke(
        "expression_request",
        json.dumps(
            {
                "mode": "blend",
                "loop": True,
                "expressions": [
                    {
                        "name": "cap-expression-0",
                        "weight": 0.75,
                        "duration_seconds": 2,
                        "transition_seconds": 0.3,
                        "parameters": {
                            "ParamAngleX": 12,
                            "bad key": 1,
                            "ParamUnsafe": 10001,
                        },
                    },
                    {"name": "cap-motion-0", "weight": 0.25},
                ],
                "api_key": "private",
            }
        ),
    )
    bridge.invoke(
        "motion_request",
        json.dumps(
            {
                "name": "cap-motion-1",
                "duration_seconds": 4,
                "transition_seconds": 0.1,
                "loop": True,
                "parameters": {"ParamBodyAngleX": 4},
                "command_line": "private",
            }
        ),
    )
    bridge.invoke("select_renderer_backend", json.dumps({"backend": "vulkan"}))
    bridge.invoke("select_renderer_backend", json.dumps({"backend": "private"}))
    assert seen == [
        (
            "expression_request",
            {
                "expressions": [
                    {
                        "name": "cap-expression-0",
                        "weight": 0.75,
                        "duration_seconds": 2.0,
                        "transition_seconds": 0.3,
                        "parameters": {"ParamAngleX": 12.0},
                    }
                ],
                "mode": "blend",
                "loop": True,
            },
        ),
        (
            "motion_request",
            {
                "name": "cap-motion-1",
                "duration_seconds": 4.0,
                "transition_seconds": 0.1,
                "loop": True,
                "parameters": {"ParamBodyAngleX": 4.0},
            },
        ),
        ("select_renderer_backend", {"backend": "vulkan"}),
        ("select_renderer_backend", {}),
    ]


def test_web_bridge_rejects_unknown_action_before_callback() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    called = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda kind, payload: called.append((kind, payload)) or {"status": "completed"},
    )
    result = json.loads(bridge.invoke("system:run_command", "{}"))
    assert result == {"status": "unavailable", "reason": "操作不可用"}
    assert called == []


def test_web_console_state_revision_helper_is_monotonic_and_safe() -> None:
    try:
        from gui.qt6.web_console import _state_revision
    except ImportError:
        return
    assert _state_revision({"revision": 12}) == 12
    assert _state_revision({"revision": -3}) == 0
    assert _state_revision({"revision": "bad"}) == 0
    assert _state_revision({"revision": True}) == 0
    assert _state_revision(None) == 0


def test_web_console_html_rejects_late_state_snapshots() -> None:
    from gui.web.control_surface import control_surface_html

    html = control_surface_html({"revision": 3})
    assert "function stateRevision(value)" in html
    assert "if (stateRevision(candidate) < stateRevision(state)) return false;" in html
    assert "applyState(JSON.parse(next||'{}'))" in html


def test_web_bridge_allows_observation_buttons_without_arguments() -> None:
    try:
        from gui.qt6.web_console import _ControlSurfaceBridge
    except ImportError:
        return
    seen: list[tuple[str, dict[str, object]]] = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda kind, payload: (
            seen.append((kind, dict(payload)))
            or {
                "status": "available",
                "title": "编辑器",
                "process_name": "editor",
                "pid": 42,
                "command_line": "--secret",
            }
        ),
    )
    for kind in ("read_foreground_window", "read_processes"):
        result = json.loads(
            bridge.invoke(
                kind,
                json.dumps({"limit": 99, "command_line": "--secret"}),
            )
        )
        assert result["status"] == "available"
        assert "command_line" not in json.dumps(result, ensure_ascii=False)
    assert seen == [("read_foreground_window", {}), ("read_processes", {})]


def test_web_control_surface_host_renders_and_routes_clicks(tmp_path: Path) -> None:
    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = """
import json
import sys
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QApplication
from gui.qt6.app import _configure_qt_webengine_renderer
from gui.qt6.web_console import WebControlSurfaceWindow
_configure_qt_webengine_renderer()
app = QApplication(sys.argv)
seen = []
state = {
    "interaction": {
        "phase": "approval_required",
        "text": "你好",
        "actions": [
            {"kind": "approve", "label": "允许", "enabled": True, "payload": {}},
            {"kind": "deny", "label": "拒绝", "enabled": True, "payload": {}},
        ],
        "approval": {
            "display_name": "查看当前窗口",
            "safe_summary": "读取当前前台窗口标题和进程名称",
            "expires_at": 4000000000,
        },
    },
    "renderer": {"backend": "web_live2d", "available": True},
    "affection": {"current": "6", "tier": "认识"},
    "capabilities": {"expressions": ["happy"], "motions": ["wave"]},
}
window = WebControlSurfaceWindow(
    lambda: state,
    lambda kind, payload: seen.append((kind, dict(payload))) or {"status": "completed"},
)
window.show()
window.showMinimized()
window.show_and_focus()
assert not bool(window.windowState() & Qt.WindowState.WindowMinimized)

def finish(value):
    probe = json.loads(str(value or "{}"))
    print(json.dumps({"probe": probe, "seen": seen}, ensure_ascii=False))
    window.shutdown()
    app.quit()

def wait_ready(attempt=0):
    if not window.page_ready:
        if attempt < 80:
            QTimer.singleShot(100, lambda: wait_ready(attempt + 1))
            return
        finish("{}")
        return
    window.set_state(state)

    def probe():
        window.view.page().runJavaScript(
            "(()=>{const b=document.querySelector('#parts button');b?.click();"
            "document.querySelector('[data-action=read_foreground_window]')?.click();"
            "return JSON.stringify({title:document.querySelector('h2')?.textContent,"
            "renderer:document.getElementById('renderer')?.textContent,"
            "observation:[...document.querySelectorAll('#observationCard button')]"
            ".map(x=>x.textContent),"
            "state_actions:[...document.querySelectorAll('#stateActions button')]"
            ".map(x=>x.textContent),"
            "approval_actions:[...document.querySelectorAll('#approvalActions button')]"
            ".map(x=>x.textContent)});})()",
            finish,
        )

    QTimer.singleShot(250, probe)

QTimer.singleShot(500, wait_ready)
QTimer.singleShot(12000, app.quit)
app.exec()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader "
        "--disable-gpu-compositing --disable-gpu-sandbox"
    )
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24",
            sys.executable,
            "-c",
            script,
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["probe"]["title"] == "桌宠控制台"
    assert "Web Live2D" in payload["probe"]["renderer"]
    assert payload["probe"]["observation"][1] == "查看运行中的程序"
    assert payload["probe"]["observation"][0] in {"查看前台窗口", "处理中…"}
    assert payload["probe"]["state_actions"] == []
    assert payload["probe"]["approval_actions"] == ["允许", "本次会话允许", "拒绝"]
    assert payload["seen"] == [
        ["pet_part", {"part": "head"}],
        ["read_foreground_window", {}],
    ]
