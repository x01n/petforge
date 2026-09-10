from __future__ import annotations

import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    shutil.which("xvfb-run") is None or shutil.which("xdotool") is None,
    reason="X11 physical input smoke requires xvfb-run and xdotool",
)
@pytest.mark.parametrize("scale_factor", ("1", "2"))
def test_physical_input_smoke_routes_transparent_and_click_through_events(
    scale_factor: str,
) -> None:
    """真实 X11 事件不能只依赖页面内合成 PointerEvent。"""

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QT_SCALE_FACTOR"] = scale_factor
    source_root = project_root / "src"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1280x720x24",
            sys.executable,
            str(project_root / "scripts" / "physical_input_smoke.py"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok", payload
    assert payload["background_clicks"] >= 2
    assert payload["pet_clicks"] >= 1


def test_physical_input_smoke_selects_xcb_only_when_qpa_is_unset(monkeypatch) -> None:
    """Xvfb 探针不应被宿主 Wayland 环境悄悄切换到错误的 QPA。"""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "physical_input_smoke.py"
    script_namespace = runpy.run_path(str(script_path), run_name="physical_input_smoke_test")
    prepare_x11_qpa = script_namespace["_prepare_x11_qpa"]

    monkeypatch.setenv("DISPLAY", ":99")
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    assert prepare_x11_qpa() is True
    assert os.environ["QT_QPA_PLATFORM"] == "xcb"

    monkeypatch.setenv("QT_QPA_PLATFORM", "wayland-egl")
    assert prepare_x11_qpa() is False
    assert os.environ["QT_QPA_PLATFORM"] == "wayland-egl"
