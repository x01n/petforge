from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("backend", ["sprite", "web_live2d"])
def test_app_tray_hide_keeps_hotkey_conversation_and_streaming_audio(backend: str) -> None:
    """真实应用窗口隐藏后，快捷键对话与 TTS 消费仍贯通。"""

    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONUNBUFFERED="1")
    command = [sys.executable, str(Path(__file__).resolve()), backend]
    if sys.platform.startswith("linux"):
        if shutil.which("xvfb-run") is None:
            pytest.skip("requires Xvfb")
        command = ["xvfb-run", "-a", "-s", "-screen 0 1440x900x24", *command]
        environment.update(
            QT_QPA_PLATFORM="xcb",
            QT_QUICK_BACKEND="software",
            QT_OPENGL="software",
            XDG_SESSION_TYPE="x11",
            MEAPET_WEBENGINE_SOFTWARE="1",
        )
        environment.pop("WAYLAND_DISPLAY", None)
    result = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True, timeout=65
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["status"] == "ok", report
    assert report["audio_chunks_while_hidden"] >= 3
    assert report["model_calls"] == 2
    assert report["recovery_requests"] == 0
    assert report["tray_restored"] is True


def test_web_host_import_without_qt() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['PySide6'] = None; "
            "from gui.qt6 import web_host; "
            "assert not web_host._qt_available; "
            "assert web_host._WebCloseEventFilter is None",
        ],
        env=dict(os.environ, PYTHONPATH=str(root / "src")),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def _run_app_scenario(backend: str) -> None:
    import asyncio
    import tempfile
    import threading
    import time
    import traceback

    from PySide6.QtCore import QObject, QThread, QTimer, Signal
    from PySide6.QtWidgets import QApplication, QInputDialog, QSystemTrayIcon

    import gui.qt6.app as gui_app
    import gui.qt6.hotkeys as hotkeys
    from app.runtime import build_runtime
    from config.loader import LoadedConfiguration
    from config.resources import inspect_resources
    from core.adapters.direct import ProviderAdapter
    from core.events.types import TextDelta, TurnFinished
    from core.tts.contracts import EngineHealth, SpeechChunk
    from services.model_routing import ChannelConfig, ModelRouter

    gui_app._configure_qt_webengine_renderer()
    app = QApplication.instance() or QApplication([])
    captures = {}
    audio = []
    resume = threading.Event()
    report = {"status": "failed", "tray_restored": False, "recovery_requests": 0}

    class Tray(QObject):
        activated = Signal(object)
        ActivationReason = QSystemTrayIcon.ActivationReason

        @staticmethod
        def isSystemTrayAvailable():  # noqa: N802
            return True

        def __init__(self, parent):
            super().__init__(parent)
            self.visible = False
            captures["tray"] = self

        def setIcon(self, value):  # noqa: N802
            self.icon = value

        def setToolTip(self, value):  # noqa: N802
            self.tooltip = value

        def setContextMenu(self, menu):  # noqa: N802
            self.menu = menu

        def show(self):
            self.visible = True

        def hide(self):
            self.visible = False

        def isVisible(self):  # noqa: N802
            return self.visible

    class KeyBridge:
        def __init__(self, bindings, callback):
            self.bindings = bindings

        def start(self):
            return "available", tuple(binding.action for binding in self.bindings), ()

        def close(self):
            pass

    class Keys(hotkeys.HotkeyManager):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captures["keys"] = self

    class Console(gui_app.PetConsoleWindow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captures["console"] = self

    class WebConsole(gui_app.WebControlSurfaceWindow):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captures["web_console"] = self

    class Player(QObject):
        audioReceived = Signal(object)
        playbackStateChanged = Signal(bool)
        playing = False

        def __init__(self):
            super().__init__()
            self.closed = False
            self.audioReceived.connect(self.receive)
            captures["player"] = self

        def __call__(self, chunk):
            self.audioReceived.emit(chunk)

        def receive(self, chunk):
            assert QThread.currentThread() == app.thread()
            if chunk.data:
                assert not self.closed
                audio.append((chunk.data, not captures["surface"].isVisible()))

        def diagnostics(self):
            return {"status": "idle", "available": True, "active": False}

        def stop_all(self):
            self.closed = True

        def close(self):
            self.closed = True

    class Speech:
        async def start(self):
            return EngineHealth("tray-test", True)

        health = start

        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"\x01\x00" * 80)
            while not resume.is_set():
                await asyncio.sleep(0.01)
            yield SpeechChunk(request.request_id, b"\x02\x00" * 80, is_final=True)

        async def aclose(self):
            pass

    class Adapter(ProviderAdapter):
        provider = "tray-test"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming"})
        calls = 0

        async def stream(self, request, *, context=None, cancel_event=None):
            if request.metadata.get("purpose") == "memory_extract":
                yield TextDelta(context, "[]")
            else:
                self.calls += 1
                yield TextDelta(context, "回复仍在继续。")
            yield TurnFinished(context, "stop")

    import PySide6.QtWidgets as widgets

    widgets.QSystemTrayIcon = Tray
    hotkeys._Win32GlobalHotkey = KeyBridge
    hotkeys._X11GlobalHotkey = KeyBridge
    gui_app.HotkeyManager = Keys
    gui_app.PetConsoleWindow = Console
    gui_app.WebControlSurfaceWindow = WebConsole
    gui_app.QtAudioPlayer = Player

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="petforge-tray-") as scratch:
        configuration = LoadedConfiguration(
            Path(scratch) / "config.yaml",
            {
                "storage": {"database": str(Path(scratch) / "state.sqlite3")},
                "rendering": {"backend": backend, "resource_root": str(root / "resources")},
                "tts": {"enabled": True, "backend": "text_only"},
                "asr": {"enabled": False, "capture": {"enabled": False}},
                "behavior": {"enabled": False},
                "watcher": {"enabled": False},
                "scheduler": {"enabled": False, "activity": {"enabled": False}},
                "config": {"reload": {"enabled": False}},
                "ui": {"auto_open_model_setup": False},
            },
        )
        runtime = build_runtime(configuration, inspect_resources(root / "resources"))
        runtime.tts.backend = Speech()
        adapter = Adapter()
        router = ModelRouter(
            [ChannelConfig("tray", base_url="https://tray.invalid/v1", model="test")],
            adapter_factory=lambda channel: adapter,
        )
        runtime.router = runtime.conversation.router = router
        runtime.interaction_state.set_model_diagnostics(router.diagnostics("dialogue"))
        deadline = time.monotonic() + 45
        stage = 0

        def action(label):
            return next(item for item in captures["tray"].menu.actions() if item.text() == label)

        def close_surface(surface):
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes

                post = ctypes.WinDLL("user32", use_last_error=True).PostMessageW
                post.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                post.restype = wintypes.BOOL
                assert post(int(surface.winId()), 0x0010, 0, 0)
            else:
                surface.close()

        def tick():
            nonlocal stage
            try:
                if time.monotonic() > deadline:
                    raise AssertionError(f"tray lifecycle timed out at stage {stage}")
                if QThread.currentThread().loopLevel() < 1:
                    return
                if stage == 0:
                    if "keys" not in captures:
                        return
                    target = runtime.pet_controller._target
                    surface = getattr(target, "view", target)
                    if backend == "web_live2d" and not target.renderer.model_ready:
                        return
                    captures["surface"] = surface
                    captures["target"] = target

                    # 关闭控制台不应触发点击恢复；即使恢复不可用，也不应自动弹回。
                    def reject_recovery(enabled):
                        report["recovery_requests"] += 1
                        return {"status": "unavailable", "enabled": True}

                    target.set_click_through = reject_recovery
                    action("打开控制台").trigger()
                    console = captures["console"]
                    console.input_line.setText("第一条消息")
                    console._submit()
                    stage = 1
                elif stage == 1:
                    if not audio:
                        return
                    close_surface(captures["console"])
                    close_surface(captures["surface"])
                    stage = 2
                elif stage == 2:
                    if captures["console"].isVisible() or captures["surface"].isVisible():
                        return
                    assert not captures["player"].closed
                    assert not captures["target"]._shutdown
                    assert captures["tray"].isVisible()
                    resume.set()
                    stage = 3
                elif stage == 3:
                    if (
                        len(audio) < 2
                        or runtime.interaction_state.snapshot.phase.value != "completed"
                    ):
                        return
                    assert audio[-1][1] is True
                    stage = 4
                    captures["keys"]._on_global_hotkey("open_input", True)
                elif stage == 4:
                    if adapter.calls < 2 or len(audio) < 4:
                        return
                    assert not captures["surface"].isVisible()
                    assert not captures["console"].isVisible()
                    captures["tray"].activated.emit(Tray.ActivationReason.DoubleClick)
                    assert captures["console"].isVisible()
                    assert "回复仍在继续" in captures["console"].output_view.toPlainText()
                    action("打开网页控制台").trigger()
                    close_surface(captures["web_console"])
                    stage = 5
                elif stage == 5:
                    if captures["web_console"].isVisible():
                        return
                    assert captures["console"].isVisible()
                    close_surface(captures["console"])
                    action("显示桌宠").trigger()
                    assert captures["surface"].isVisible()
                    action("隐藏桌宠").trigger()
                    assert not captures["surface"].isVisible()
                    report.update(
                        status="ok",
                        tray_restored=True,
                        model_calls=adapter.calls,
                        audio_chunks_while_hidden=sum(hidden for data, hidden in audio),
                    )
                    timer.stop()
                    app.quit()
            except Exception:
                traceback.print_exc()
                timer.stop()
                app.exit(1)

        # 使用真实 QInputDialog；在其嵌套事件循环中填入快捷键消息并确认。
        def fill_dialog():
            for widget in app.topLevelWidgets():
                if isinstance(widget, QInputDialog) and widget.isVisible():
                    widget.setTextValue("隐藏后快捷键消息")
                    widget.accept()

        dialog_timer = QTimer(app)
        dialog_timer.timeout.connect(fill_dialog)
        dialog_timer.start(50)
        timer = QTimer(app)
        timer.timeout.connect(tick)
        timer.start(100)
        code = gui_app.run(configuration, inspect_resources(root / "resources"), runtime=runtime)
        timer.stop()
        dialog_timer.stop()
        print(json.dumps(report, ensure_ascii=False))
        raise SystemExit(code if report["status"] == "ok" else 1)


if __name__ == "__main__":
    _run_app_scenario(sys.argv[1])
