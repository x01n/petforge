from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

from core.adapters.mcp import stdio as mcp_stdio
from core.memory_semantic import SentenceTransformerProcess
from services import processes


class _AsyncProcess:
    def __init__(self) -> None:
        self.pid = 417
        self.returncode: int | None = None
        self.signals: list[int] = []
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def send_signal(self, value: int) -> None:
        self.signals.append(value)

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        while self.returncode is None:
            await asyncio.sleep(10)
        return self.returncode


class _SyncProcess:
    def __init__(self) -> None:
        self.pid = 733
        self.returncode: int | None = None
        self.stdin = None
        self.stdout = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, *, timeout: float) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            raise subprocess.TimeoutExpired("worker", timeout)
        return self.returncode


def test_windows_spawn_uses_stdlib_process_group_flag(monkeypatch) -> None:
    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(processes.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)

    assert processes.process_group_spawn_kwargs() == {"creationflags": 0x200}


def test_windows_taskkill_invocation_is_tree_scoped_and_shell_free(monkeypatch) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(processes.subprocess, "CREATE_NO_WINDOW", 0x8000000, raising=False)

    def run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(processes.subprocess, "run", run)

    assert processes._windows_taskkill_tree(512, timeout=0.25) is True
    assert calls == [
        (
            ("taskkill", "/PID", "512", "/T", "/F"),
            {
                "shell": False,
                "stdin": processes.subprocess.DEVNULL,
                "stdout": processes.subprocess.DEVNULL,
                "stderr": processes.subprocess.DEVNULL,
                "check": False,
                "timeout": 0.25,
                "creationflags": 0x8000000,
            },
        )
    ]


def test_windows_async_tree_cleanup_prefers_ctrl_break_then_taskkill(monkeypatch) -> None:
    process = _AsyncProcess()
    calls: list[tuple[int, float]] = []
    control_break = 12345

    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(processes.signal, "CTRL_BREAK_EVENT", control_break, raising=False)

    def taskkill(process_id: int, *, timeout: float) -> bool:
        calls.append((process_id, timeout))
        process.returncode = 0
        return True

    monkeypatch.setattr(processes, "_windows_taskkill_tree", taskkill)

    asyncio.run(processes.terminate_process_tree(process, timeout=0.1))

    assert process.signals == [control_break]
    assert calls == [(417, 0.1)]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.returncode == 0


def test_windows_async_tree_cleanup_falls_back_to_root_kill(monkeypatch) -> None:
    process = _AsyncProcess()
    control_break = 54321

    monkeypatch.setattr(processes.os, "name", "nt")
    monkeypatch.setattr(processes.signal, "CTRL_BREAK_EVENT", control_break, raising=False)
    monkeypatch.setattr(processes, "_windows_taskkill_tree", lambda *_args, **_kwargs: False)

    asyncio.run(processes.terminate_process_tree(process, timeout=0.1))

    assert process.signals == [control_break]
    assert process.terminate_calls == 0
    assert process.kill_calls == 1
    assert process.returncode == -9


def test_windows_sync_cleanup_uses_taskkill_for_process_tree(monkeypatch) -> None:
    process = _SyncProcess()
    calls: list[tuple[int, float]] = []

    monkeypatch.setattr(processes.os, "name", "nt")

    def taskkill(process_id: int, *, timeout: float) -> bool:
        calls.append((process_id, timeout))
        process.returncode = 0
        return True

    monkeypatch.setattr(processes, "_windows_taskkill_tree", taskkill)
    processes.terminate_process_sync(process, timeout=0.2)

    assert calls == [(733, 0.2)]
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert process.wait_calls == 2
    assert process.returncode == 0


def test_mcp_stdio_uses_shared_tree_cleanup(monkeypatch) -> None:
    client = mcp_stdio.StdioMCPClient(
        {
            "name": "cleanup-test",
            "command": ("worker",),
            "initialize": False,
        }
    )
    process = _AsyncProcess()
    calls: list[tuple[object, float]] = []

    async def cleanup(process_arg, *, timeout: float) -> None:
        calls.append((process_arg, timeout))

    monkeypatch.setattr(mcp_stdio, "terminate_process_tree", cleanup)
    asyncio.run(client._terminate_process_unlocked(process, timeout=0.3))

    assert calls == [(process, 0.3)]


def test_semantic_worker_uses_shared_sync_cleanup(monkeypatch) -> None:
    process = _SyncProcess()
    settings = SimpleNamespace(shutdown_timeout_seconds=0.4)
    worker = object.__new__(SentenceTransformerProcess)
    worker._settings = settings
    worker._process = process
    worker._spec = object()
    worker._read_buffer = bytearray(b"stale")
    calls: list[tuple[object, float]] = []

    def cleanup(process_arg, *, timeout: float) -> None:
        calls.append((process_arg, timeout))

    monkeypatch.setattr("core.memory_semantic.terminate_process_sync", cleanup)
    worker._terminate()

    assert calls == [(process, 0.4)]
    assert worker._process is None
    assert worker._spec is None
    assert worker._read_buffer == bytearray()
