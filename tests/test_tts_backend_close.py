from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

from core.tts.contracts import SpeechRequest
from services.tts.backend import SubprocessTTSBackend

_WORKER = r"""
import json
import pathlib
import sys
import time

mode = sys.argv[1]
marker = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else None
capabilities = [] if mode == "plain" else ["cancel", "shutdown", "streaming_synthesis"]
print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
    "capabilities": capabilities,
}), flush=True)

for raw in sys.stdin:
    message = json.loads(raw)
    message_type = message["type"]
    request_id = message.get("request_id", "")
    if message_type == "synthesize":
        continue
    if message_type == "cancel":
        if marker is not None:
            marker.write_text("cancelled", encoding="utf-8")
        print(json.dumps({
            "type": "cancelled",
            "protocol": "meapet.tts.jsonl",
            "version": 1,
            "request_id": request_id,
        }), flush=True)
        continue
    if message_type == "shutdown":
        if mode == "slow-shutdown":
            time.sleep(0.2)
        print(json.dumps({
            "type": "shutdown",
            "protocol": "meapet.tts.jsonl",
            "version": 1,
            "request_id": request_id,
        }), flush=True)
        break
"""


def _backend(
    mode: str,
    *,
    marker: Path | None = None,
    timeout_seconds: float = 180.0,
    shutdown_timeout_seconds: float = 0.5,
) -> SubprocessTTSBackend:
    command = (sys.executable, "-u", "-c", _WORKER, mode)
    if marker is not None:
        command += (str(marker),)
    return SubprocessTTSBackend(
        command,
        timeout_seconds=timeout_seconds,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )


def _process_is_gone(process_id: int) -> bool:
    if sys.platform.startswith("linux"):
        try:
            state = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        except OSError:
            return True
        return state.rsplit(") ", 1)[-1][:1] == "Z"
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return True
    except (OSError, PermissionError):
        return False
    return False


def test_cancelled_close_waiter_does_not_cancel_shared_worker_cleanup() -> None:
    async def scenario() -> tuple[int, object]:
        backend = _backend("slow-shutdown", shutdown_timeout_seconds=1.0)
        assert (await backend.start()).available is True
        process = backend._process
        assert process is not None

        first_waiter = asyncio.create_task(backend.aclose())
        await asyncio.sleep(0.03)
        shared_cleanup = backend._close_task
        assert shared_cleanup is not None
        first_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        second_waiter = asyncio.create_task(backend.aclose())
        await second_waiter
        assert backend._close_task is shared_cleanup
        assert backend._closed is True
        assert backend._process is None
        assert process.returncode is not None
        return process.pid, shared_cleanup

    process_id, cleanup = asyncio.run(scenario())
    assert cleanup.done() is True
    assert _process_is_gone(process_id)


def test_cancelled_only_close_waiter_still_reaps_worker() -> None:
    async def scenario() -> int:
        backend = _backend("slow-shutdown", shutdown_timeout_seconds=1.0)
        assert (await backend.start()).available is True
        process = backend._process
        assert process is not None

        waiter = asyncio.create_task(backend.aclose())
        await asyncio.sleep(0.03)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert backend._closed is False

        deadline = time.monotonic() + 2.0
        while not backend._closed and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert backend._closed is True
        assert backend._process is None
        assert process.returncode is not None
        await backend.aclose()
        return process.pid

    process_id = asyncio.run(scenario())
    assert _process_is_gone(process_id)


def test_failed_close_can_be_retried_to_compensate_worker_cleanup(monkeypatch) -> None:
    async def scenario() -> int:
        backend = _backend("plain")
        assert (await backend.start()).available is True
        process = backend._process
        assert process is not None
        original_dispose = backend._dispose_process
        dispose_calls = 0

        async def fail_first_dispose(process_arg, *, graceful=False):
            nonlocal dispose_calls
            dispose_calls += 1
            if dispose_calls == 1:
                raise RuntimeError("injected cleanup failure")
            await original_dispose(process_arg, graceful=graceful)

        monkeypatch.setattr(backend, "_dispose_process", fail_first_dispose)
        with pytest.raises(RuntimeError, match="injected cleanup failure"):
            await backend.aclose()
        assert backend._closed is False
        first_cleanup = backend._close_task

        await backend.aclose()
        assert backend._close_task is not first_cleanup
        assert backend._closed is True
        assert backend._process is None
        assert dispose_calls == 2
        return process.pid

    process_id = asyncio.run(scenario())
    assert _process_is_gone(process_id)


def test_close_cancels_long_stream_via_ipc_with_bounded_latency(tmp_path) -> None:
    cancel_marker = tmp_path / "cancelled"

    async def scenario() -> tuple[float, int]:
        backend = _backend(
            "stream",
            marker=cancel_marker,
            timeout_seconds=180.0,
            shutdown_timeout_seconds=0.2,
        )
        stream = backend.stream(SpeechRequest("long-stream", "长时间推理"))
        stream_task = asyncio.create_task(anext(stream))
        deadline = time.monotonic() + 2.0
        while backend._active_request_id != "long-stream" and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert backend._active_request_id == "long-stream"
        process = backend._process
        assert process is not None

        started = time.monotonic()
        await backend.aclose()
        elapsed = time.monotonic() - started
        with pytest.raises(asyncio.CancelledError):
            await stream_task
        await stream.aclose()
        assert backend._closed is True
        assert backend._process is None
        assert process.returncode is not None
        return elapsed, process.pid

    elapsed, process_id = asyncio.run(scenario())
    assert elapsed < 1.0
    assert cancel_marker.read_text(encoding="utf-8") == "cancelled"
    assert _process_is_gone(process_id)


@pytest.mark.parametrize(
    ("command", "expected"),
    (
        ("health", "health"),
        ("load_model", "model_loaded"),
        ("shutdown", "shutdown"),
    ),
)
def test_ipc_business_ack_requires_exact_nonempty_request_id(
    command: str,
    expected: str,
) -> None:
    worker = f"""
import json
import sys

print(json.dumps({{
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
    "capabilities": ["health", "model_load", "shutdown"],
}}), flush=True)
for raw in sys.stdin:
    json.loads(raw)
    print(json.dumps({{
        "type": {expected!r},
        "protocol": "meapet.tts.jsonl",
        "version": 1,
    }}), flush=True)
"""

    async def scenario() -> None:
        backend = SubprocessTTSBackend(
            (sys.executable, "-u", "-c", worker),
            timeout_seconds=1.0,
            shutdown_timeout_seconds=0.1,
        )
        assert (await backend.start()).available is True
        async with backend._request_lock:
            with pytest.raises(RuntimeError, match="request_id is required"):
                await backend._invoke_ipc(
                    command,
                    request_id="strict-business-ack",
                    expected=frozenset({expected}),
                    timeout=0.5,
                )
        await backend.aclose()

    asyncio.run(scenario())


def test_zero_exit_after_nonfinal_audio_is_not_reported_as_completed() -> None:
    worker = r"""
import base64
import json
import sys

print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
request = json.loads(sys.stdin.readline())
print(json.dumps({
    "type": "audio",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "request_id": request["request_id"],
    "data": base64.b64encode(b"\x00\x00").decode("ascii"),
    "sample_rate": 24000,
    "channels": 1,
    "final": False,
}), flush=True)
"""

    async def scenario() -> None:
        backend = SubprocessTTSBackend(
            (sys.executable, "-u", "-c", worker),
            require_final=False,
            timeout_seconds=1.0,
            shutdown_timeout_seconds=0.1,
        )
        with pytest.raises(RuntimeError, match="terminal stream event"):
            async for _chunk in backend.stream(SpeechRequest("truncated", "截断流")):
                pass
        assert backend._process is None
        await backend.aclose()

    asyncio.run(scenario())


def test_diagnostics_distinguishes_loading_ready_and_stopped_ipc() -> None:
    worker = r"""
import json
import sys
import time

time.sleep(0.2)
print(json.dumps({
    "type": "ready",
    "protocol": "meapet.tts.jsonl",
    "version": 1,
    "streaming": True,
}), flush=True)
for _line in sys.stdin:
    pass
"""

    async def scenario() -> None:
        backend = SubprocessTTSBackend(
            (sys.executable, "-u", "-c", worker),
            startup_timeout_seconds=1.0,
            shutdown_timeout_seconds=0.1,
        )
        assert backend.diagnostics()["ipc"] == {
            "status": "stopped",
            "running": False,
            "ready": False,
            "pid": None,
        }

        start_task = asyncio.create_task(backend.start())
        deadline = time.monotonic() + 1.0
        while backend._process is None and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        loading = backend.diagnostics()["ipc"]
        assert loading["status"] == "loading"
        assert loading["running"] is True
        assert loading["ready"] is False

        assert (await start_task).available is True
        ready = backend.diagnostics()["ipc"]
        assert ready["status"] == "ready"
        assert ready["running"] is True
        assert ready["ready"] is True

        await backend.aclose()
        stopped = backend.diagnostics()["ipc"]
        assert stopped["status"] == "stopped"
        assert stopped["running"] is False
        assert stopped["ready"] is False

    asyncio.run(scenario())
