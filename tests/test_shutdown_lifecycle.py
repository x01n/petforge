from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

from app.loop import RuntimeLoop
from app.runtime import build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from gui.qt6 import app as qt_app


def test_shutdown_signal_handlers_request_qt_quit_and_restore(monkeypatch) -> None:
    calls: list[tuple[object, object]] = []
    handlers: dict[object, object] = {}

    class FakeApplication:
        def __init__(self) -> None:
            self.quit_count = 0

        def quit(self) -> None:
            self.quit_count += 1

    application = FakeApplication()

    def fake_getsignal(signum):
        return f"old-{signum}"

    def fake_signal(signum, handler):
        calls.append((signum, handler))
        handlers[signum] = handler
        return handler

    monkeypatch.setattr(qt_app.signal, "getsignal", fake_getsignal)
    monkeypatch.setattr(qt_app.signal, "signal", fake_signal)
    restore = qt_app._install_shutdown_signal_handlers(application)

    assert signal.SIGINT in handlers
    handlers[signal.SIGINT](signal.SIGINT, None)
    if hasattr(signal, "SIGTERM"):
        handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert application.quit_count == 1

    restore()
    restore()
    restored = {signum: handler for signum, handler in calls[-2:]}
    assert restored[signal.SIGINT] == f"old-{signal.SIGINT}"
    if hasattr(signal, "SIGTERM"):
        assert restored[signal.SIGTERM] == f"old-{signal.SIGTERM}"


def test_shutdown_signal_handlers_skip_non_main_thread(monkeypatch) -> None:
    installed: list[object] = []

    def fake_signal(signum, _handler):
        installed.append(signum)

    monkeypatch.setattr(qt_app.signal, "signal", fake_signal)
    result: list[object] = []

    def worker() -> None:
        result.append(qt_app._install_shutdown_signal_handlers(object()))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert installed == []
    assert len(result) == 1
    result[0]()


def test_qt_signal_shutdown_wakes_event_loop(tmp_path: Path) -> None:
    """真实 Qt 事件循环收到 SIGTERM 后应在短时间内退出。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
import os
import signal
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.qt6.app import _install_shutdown_signal_handlers

app = QApplication([])
restore = _install_shutdown_signal_handlers(app)
QTimer.singleShot(100, lambda: os.kill(os.getpid(), signal.SIGTERM))
code = app.exec()
restore()
assert code == 0
print("qt-signal-shutdown-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{environment['PYTHONPATH']}"
        if environment.get("PYTHONPATH")
        else str(source_root)
    )
    result = subprocess.run(
        ["xvfb-run", "-a", "-s", "-screen 0 800x600x24", sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-signal-shutdown-ok" in result.stdout


def test_runtime_loop_stop_is_idempotent_after_thread_shutdown(tmp_path: Path) -> None:
    configuration = LoadedConfiguration(tmp_path / "config.yaml", {})
    runtime = build_runtime(configuration, inspect_resources(tmp_path / "resources"))
    loop = RuntimeLoop(runtime)
    loop.start()
    loop.stop(timeout=2)
    loop.stop(timeout=2)
    assert runtime.database.connection is not None
    try:
        runtime.database.connection.execute("SELECT 1")
    except Exception:
        # SQLite 在关闭连接后应拒绝查询；这里仅确认第二次 stop 没有
        # 改变已经完成的关闭状态。
        return
    raise AssertionError("database remained open after idempotent RuntimeLoop.stop")


def test_runtime_loop_stop_requested_during_startup_does_not_enter_forever() -> None:
    class Database:
        def close(self) -> None:
            return None

    class Runtime:
        def __init__(self) -> None:
            self.database = Database()
            self.started = threading.Event()
            self.closed = 0

        async def start_background(self) -> None:
            self.started.set()
            await asyncio.Event().wait()

        async def close(self) -> None:
            self.closed += 1

    runtime = Runtime()
    loop = RuntimeLoop(runtime)
    loop._thread = threading.Thread(target=loop._thread_main, daemon=True)
    loop._thread.start()
    assert runtime.started.wait(2)
    stopper = threading.Thread(target=lambda: loop.stop(timeout=2))
    stopper.start()
    stopper.join(3)
    assert not stopper.is_alive()
    assert runtime.closed == 1
