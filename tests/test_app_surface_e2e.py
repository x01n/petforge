from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("xvfb-run") is None, reason="requires Xvfb")
def test_real_app_web_surface_has_no_opaque_black_rectangle() -> None:
    """真实 app.run 顶层 WebEngine 表面不能退化成黑色矩形。"""

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
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
            "-screen 0 1440x900x24",
            sys.executable,
            str(project_root / "scripts" / "app_surface_e2e_smoke.py"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok", payload
    assert float(payload.get("raw_black_ratio", 1.0)) < 0.20, payload
    if payload.get("compositor_unavailable"):
        assert payload.get("screen_capture") == "unavailable", payload
    else:
        assert float(payload.get("black_ratio", 1.0)) < 0.20, payload
