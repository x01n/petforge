"""当前橙色猫猫 MotionSync 资源与 Web 口型帧回归。"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from core.rendering.resources import validate_motionsync3_description
from gui.renderers.web_live2d import WebLive2DRenderer

_PROFILE = Path("resources/live2d/model/mea_live2d/橙色猫猫.motionsync3.json")
_VISEMES = ("Silence", "A", "I", "U", "E", "O")


def test_current_motionsync3_profile_is_strictly_valid() -> None:
    profile = validate_motionsync3_description(_PROFILE)
    assert profile is not None
    assert profile.setting_id == "MotionSyncSetting"
    assert profile.visemes == _VISEMES
    assert profile.parameters == (
        ("ParamMouthForm", -1.0, 1.0),
        ("ParamMouthOpenY", 0.0, 1.0),
    )
    mappings = dict(profile.mappings)
    assert mappings["A"] == (
        ("ParamMouthForm", 1.0),
        ("ParamMouthOpenY", 1.0),
    )
    assert mappings["U"] == (
        ("ParamMouthForm", -1.0),
        ("ParamMouthOpenY", 0.4000000059604645),
    )
    assert profile.sample_rate == 30.0
    assert profile.blend_ratio == pytest.approx(0.4)


def test_malformed_motionsync3_profile_is_rejected(tmp_path: Path) -> None:
    value = json.loads(_PROFILE.read_text(encoding="utf-8"))
    value["Settings"][0]["Mappings"] = [
        item for item in value["Settings"][0]["Mappings"] if item["Id"] != "O"
    ]
    path = tmp_path / "invalid.motionsync3.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert validate_motionsync3_description(path) is None


def test_motionsync3_unhashable_identifiers_fail_closed(tmp_path: Path) -> None:
    for field, replacement in (
        (("Settings", 0, "AudioParameters", 0, "Id"), []),
        (("Settings", 0, "Mappings", 0, "Targets", 0, "Id"), {}),
    ):
        value = json.loads(_PROFILE.read_text(encoding="utf-8"))
        cursor: object = value
        for key in field[:-1]:
            cursor = cursor[key]  # type: ignore[index]
        cursor[field[-1]] = replacement  # type: ignore[index]
        path = tmp_path / ("unhashable-" + str(field[-1]) + ".motionsync3.json")
        path.write_text(json.dumps(value), encoding="utf-8")
        assert validate_motionsync3_description(path) is None


def test_web_html_injects_current_motionsync_profile_and_safe_api() -> None:
    renderer = WebLive2DRenderer("resources")
    assert renderer.probe.motion_sync is not None
    html = renderer._html_document()
    assert 'let motionSyncProfile = {"version": 1' in html
    assert '"Silence"' in html
    assert '"ParamMouthForm"' in html
    assert "setMouthViseme: function(name, intensity)" in html
    assert "motionSyncAppliedParameters" in html
    assert renderer.set_mouth_viseme("A") is False


@pytest.mark.skipif(shutil.which("xvfb-run") is None, reason="requires Xvfb")
def test_current_motionsync_visemes_render_with_geometry_guard() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    script = f"""
import json
import sys
from pathlib import Path
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.qt6.app import _configure_qt_webengine_renderer
from gui.renderers.web_live2d import WebLive2DRenderer

_configure_qt_webengine_renderer()
app = QApplication(sys.argv)
renderer = WebLive2DRenderer({str(project_root / "resources")!r})
if not renderer.initialize() or renderer.view is None:
    print(json.dumps({{'status': 'unavailable'}}))
    raise SystemExit(1)
view = renderer.view
view.resize(420, 480)
view.show()
labels = {list(_VISEMES)!r}
samples = []
finished = False

def finish():
    global finished
    if finished:
        return
    finished = True
    print(json.dumps({{'status': 'ok', 'samples': samples}}, ensure_ascii=False))
    renderer.shutdown()
    view.close()
    app.quit()

def step(index=0):
    if index >= len(labels):
        finish()
        return
    label = labels[index]
    view.page().runJavaScript(
        "JSON.stringify((() => {{"
        f"const label = {{json.dumps(label)}};"
        "window.meapetLive2D.setSpeech('口型测试', 'neutral', true);"
        "const accepted = window.meapetLive2D.setMouthViseme(label, 1);"
        "window.meapetLive2D.advance(0.05);"
        "const debug = window.meapetLive2D.debug();"
        "const canvas = document.getElementById('meapet-face-overlay');"
        "const context = canvas && canvas.getContext ? canvas.getContext('2d') : null;"
        "let alpha = 0;"
        "if (canvas && context) {{"
        "  const data = context.getImageData(0, 0, canvas.width, canvas.height).data;"
        "  for (let i = 3; i < data.length; i += 4) if (data[i] > 0) alpha += 1;"
        "}}"
        "return {{accepted, debug, alpha}};"
        "}})())",
        lambda value: receive(index, value),
    )

def receive(index, value):
    try:
        payload = json.loads(str(value or '{{}}'))
    except (TypeError, ValueError):
        payload = {{}}
    debug = payload.get('debug') if isinstance(payload, dict) else {{}}
    samples.append({{
        'label': labels[index],
        'accepted': bool(payload.get('accepted')),
        'viseme': debug.get('motionSyncViseme'),
        'form': debug.get('motionSyncMouthForm'),
        'open': debug.get('motionSyncMouthOpenY'),
        'applied': debug.get('motionSyncAppliedParameters'),
        'available': debug.get('motionSyncAvailable'),
        'geometry': debug.get('geometryValid'),
        'continuity': debug.get('geometryContinuityValid'),
        'uniform': abs(abs(float(debug.get('scaleX', 0) or 0)) -
                       abs(float(debug.get('scaleY', 0) or 0))) <= 1e-6,
        'alpha': int(payload.get('alpha', 0) or 0),
    }})
    QTimer.singleShot(120, lambda: step(index + 1))

def wait_ready():
    if not renderer.model_ready:
        QTimer.singleShot(80, wait_ready)
        return
    step()

QTimer.singleShot(700, wait_ready)
QTimer.singleShot(12000, finish)
app.exec()
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["XDG_SESSION_TYPE"] = "x11"
    environment.pop("WAYLAND_DISPLAY", None)
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1280x900x24",
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
    assert payload["status"] == "ok"
    samples = payload["samples"]
    assert [item["label"] for item in samples] == list(_VISEMES)
    assert all(item["accepted"] and item["available"] for item in samples)
    assert all(item["geometry"] and item["continuity"] and item["uniform"] for item in samples)
    assert all(item["applied"] == 0 for item in samples)
    assert samples[0]["form"] == pytest.approx(0)
    assert samples[0]["open"] == pytest.approx(0)
    assert all(item["alpha"] > 0 for item in samples[1:])
