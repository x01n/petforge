from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from gui.web.server import LocalWebServer

TRIGGER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "meapet_trigger.py"


def _run_trigger(args: list[str], env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TRIGGER_SCRIPT), *args],
        env=dict(env),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=8,
        check=False,
    )


@pytest.fixture
def trigger_server():
    calls: list[tuple[str, dict[str, object]]] = []
    response = {"status": "completed", "message": "done"}

    def action_callback(kind: str, payload: Mapping[str, object]) -> dict[str, str]:
        calls.append((kind, dict(payload)))
        return response

    with LocalWebServer(lambda: {}, action_callback, port=0) as server:
        assert server.base_url is not None
        env = {
            **os.environ,
            "MEAPET_LOCAL_API_URL": server.base_url,
            "MEAPET_LOCAL_API_TOKEN": server.capability_token,
            "NO_PROXY": "127.0.0.1",
            "no_proxy": "127.0.0.1",
        }
        yield env, calls, response


def test_open_input_reaches_local_server_from_subprocess(trigger_server) -> None:
    env, calls, _response = trigger_server

    result = _run_trigger(["--open-input"], env)

    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "completed"
    assert calls == [("open_input", {})]


def test_submit_text_preserves_content_and_requested_success(trigger_server) -> None:
    env, calls, response = trigger_server
    response["status"] = "requested"
    text = '\u7b2c\u4e00\u884c\n\u7b2c\u4e8c\u884c "quoted" \\slash'

    result = _run_trigger(["--text", f"  {text}  "], env)

    assert result.returncode == 0
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "requested"
    assert calls == [("submit_text", {"text": text})]


def test_action_failure_has_distinct_exit_code(trigger_server) -> None:
    env, calls, response = trigger_server
    response["status"] = "unavailable"

    result = _run_trigger(["--open-input"], env)

    assert result.returncode == 1
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "unavailable"
    assert calls == [("open_input", {})]


@pytest.mark.parametrize("text", ["", "   ", "a" * 2001])
def test_invalid_text_is_rejected_before_callback(trigger_server, text: str) -> None:
    env, calls, _response = trigger_server

    result = _run_trigger(["--text", text], env)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "MeaPet local action failed: HTTPError"
    assert calls == []


def test_invalid_capability_is_rejected_before_callback(trigger_server) -> None:
    env, calls, _response = trigger_server
    env["MEAPET_LOCAL_API_TOKEN"] = "invalid"

    result = _run_trigger(["--open-input"], env)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "MeaPet local action failed: HTTPError"
    assert calls == []


@pytest.mark.parametrize("missing", ["MEAPET_LOCAL_API_URL", "MEAPET_LOCAL_API_TOKEN"])
def test_missing_environment_has_configuration_exit_code(trigger_server, missing: str) -> None:
    env, calls, _response = trigger_server
    env.pop(missing)

    result = _run_trigger(["--open-input"], env)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == ("MEAPET_LOCAL_API_URL and MEAPET_LOCAL_API_TOKEN are required")
    assert calls == []


def test_malformed_url_has_configuration_exit_code_without_traceback(
    trigger_server,
) -> None:
    env, calls, _response = trigger_server
    env["MEAPET_LOCAL_API_URL"] = "invalid"

    result = _run_trigger(["--open-input"], env)

    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.strip() == "MeaPet local action failed: ValueError"
    assert calls == []
