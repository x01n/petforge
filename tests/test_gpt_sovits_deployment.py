"""GPT-SoVITS 外置部署脚本的可复现入口测试。"""

from __future__ import annotations

import subprocess
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parents[1]
_BOOTSTRAP = _PROJECT_ROOT / "scripts" / "bootstrap_gpt_sovits.sh"
_START = _PROJECT_ROOT / "scripts" / "start_gpt_sovits.sh"
_SMOKE = _PROJECT_ROOT / "scripts" / "gpt_sovits_smoke.py"
_PINNED_COMMIT = "48b1a0169a28582a8984402f82cf438d3bfa6aca"


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_gpt_sovits_shell_scripts_are_syntactically_valid() -> None:
    for script in (_BOOTSTRAP, _START):
        result = _run("bash", "-n", str(script))
        assert result.returncode == 0, result.stderr


def test_bootstrap_help_documents_pinned_source_and_uv_install() -> None:
    result = _run("bash", str(_BOOTSTRAP), "--help")
    assert result.returncode == 0, result.stderr
    assert _PINNED_COMMIT in result.stdout
    assert "--install" in result.stdout
    assert "--update" in result.stdout


def test_start_help_documents_probe_and_background_readiness() -> None:
    result = _run("bash", str(_START), "--help")
    assert result.returncode == 0, result.stderr
    assert "--probe" in result.stdout
    assert "--background" in result.stdout


def test_real_smoke_cli_exposes_both_supported_languages() -> None:
    result = _run("uv", "run", "python", str(_SMOKE), "--help")
    assert result.returncode == 0, result.stderr
    assert "zh" in result.stdout
    assert "ja" in result.stdout
    assert "--expected-sample-rate" in result.stdout
