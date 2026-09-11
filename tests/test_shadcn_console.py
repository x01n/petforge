from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("kind", "payload", "expected"),
    [
        (
            "select_tts_profile",
            {"profile": "voice-one", "language": "zh"},
            {"profile": "voice-one", "language": "zh"},
        ),
        ("select_tts_language", {"language": "ja"}, {"language": "ja"}),
        ("select_renderer_model", {"model": "橙色猫猫 model3"}, {"model": "橙色猫猫 model3"}),
        ("select_tts_profile", {"profile": "bad\nvalue", "language": True}, {}),
    ],
)
def test_shadcn_selectors_use_existing_runtime_bridge(kind, payload, expected) -> None:
    from gui.qt6.web_console import _ControlSurfaceBridge

    calls = []
    bridge = _ControlSurfaceBridge(
        lambda: {},
        lambda action, values: calls.append((action, values)) or {"status": "updated"},
    )
    result = json.loads(bridge.invoke(kind, json.dumps({**payload, "api_key": "do-not-forward"})))
    assert result["status"] == "updated"
    assert calls == [(kind, expected)]


def test_built_console_is_available_as_an_offline_package_resource() -> None:
    from gui.web.console import console_html

    html = console_html()
    assert '<html lang="zh-CN"' in html
    assert 'id="root"' in html
    assert "meapetControlSurfaceBridge" in html
    assert "meapetControlSurface" in html
    assert 'src="https://' not in html
    assert 'href="https://' not in html


def test_shadcn_console_loads_and_invokes_runtime_over_qwebchannel() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        r"""
import json
import sys
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.qt6.app import _configure_qt_webengine_renderer
from gui.qt6.web_console import WebControlSurfaceWindow

_configure_qt_webengine_renderer()
app = QApplication([])
calls = []
state = {"revision": 3, "connection_state": "local",
         "interaction": {"phase": "idle", "text": "最新回复", "model": {"ready": False}},
         "renderer": {"backend": "web_live2d", "available": True}}
window = WebControlSurfaceWindow(
    lambda: state,
    lambda kind, payload: calls.append([kind, dict(payload)]) or {"status": "completed"},
)
window.show()
ticks = 0
clicked = False
report = {}
def finish(ok, value):
    report.update(ok=ok, view=value, calls=calls)
    timer.stop()
    window.shutdown()
    app.quit()

def checked(raw):
    global clicked
    value = json.loads(raw or "{}")
    if not value.get("ready"):
        return
    if not clicked:
        clicked = True
        window.set_state({**state, "revision": 2, "interaction": {"text": "过期回复"}})
        window.view.page().runJavaScript(
            'document.querySelector("button[data-action=configure_model]").click()')
    elif calls:
        finish(True, value)

def poll():
    global ticks
    ticks += 1
    if ticks > 100:
        finish(False, {"reason": "frontend timeout"})
        return
    if not window.page_ready:
        return
    window.view.page().runJavaScript("""
        + '"""'
        + r"""
        JSON.stringify({
            ready: !!document.querySelector('button[data-action="configure_model"]:not(:disabled)'),
            title: document.title,
            text: document.body.innerText,
            width: document.documentElement.clientWidth,
            scrollWidth: document.documentElement.scrollWidth,
            react: !!document.querySelector('#root [data-slot]')
        })
    """
        + '"""'
        + r""", checked)

timer = QTimer()
timer.timeout.connect(poll)
timer.start(100)
app.exec()
print(json.dumps(report, ensure_ascii=False))
"""
    )
    environment = dict(os.environ, PYTHONPATH=str(root / "src"), MEAPET_WEBENGINE_SOFTWARE="1")
    command = [sys.executable, "-c", script]
    if sys.platform.startswith("linux"):
        if shutil.which("xvfb-run") is None:
            pytest.skip("requires Xvfb")
        command = ["xvfb-run", "-a", "-s", "-screen 0 1280x900x24", *command]
        environment.update(QT_QPA_PLATFORM="xcb", QT_QUICK_BACKEND="software")
        environment.pop("WAYLAND_DISPLAY", None)
    result = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True, timeout=25
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["ok"], report
    assert report["view"]["react"]
    assert "过期回复" not in report["view"]["text"]
    assert report["view"]["scrollWidth"] <= report["view"]["width"]
    assert report["calls"] == [["configure_model", {}]]
