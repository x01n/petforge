"""真实 Qt/Xvfb 组合根回归。"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# 本文件九个用例全部在子进程内经 xvfb-run 启动真实 Xvfb 窗口，且除
# 纯信号收尾用例外均创建真实 WebEngine 渲染器。模块级统一打标，保持
# 与「真实 Qt/Xvfb 组合根回归」的文件定位一致。
pytestmark = [pytest.mark.xvfb, pytest.mark.webengine]


def test_virtual_window_software_webgl_keeps_live2d_ready_with_qt_quick_software() -> None:
    """显式软件路径会自动同步 Qt Quick software 并创建真实模型。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    # 不预设 Qt Quick 后端；应用入口应在显式软件 WebGL 开关下自动补齐，
    # 避免窗口重建时回到 Mesa/llvmpipe 的不稳定路径。
    environment.pop("QT_QUICK_BACKEND", None)
    environment["XDG_SESSION_TYPE"] = "x11"
    environment.pop("WAYLAND_DISPLAY", None)
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    # 让脚本内的 `_configure_qt_webengine_renderer()` 负责完整拼接软件
    # 参数，直接覆盖旧环境不会漏掉 unsafe SwiftShader 修复。
    environment.pop("QTWEBENGINE_CHROMIUM_FLAGS", None)
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
            "-screen 0 1280x900x24 -nolisten tcp",
            sys.executable,
            str(project_root / "scripts" / "virtual_window_smoke.py"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["model_ready"] is True
    assert payload["page_bridge_ready"] is True
    assert payload["renderer_ready"]["state"] == "ready"
    assert payload["renderer_ready"]["geometry_valid"] is True
    assert payload["renderer_ready"]["alpha_nonempty"] is True
    assert payload["click_parts"] == ["head", "body", "lower_left", "lower_right"]
    assert payload["shape_mask"]["input_ready"] is True
    smoke_sequence = payload["smoke_sequence"]
    assert smoke_sequence["done"] is True
    assert smoke_sequence["timed_out"] is False
    assert smoke_sequence["error"] == ""
    assert smoke_sequence["double_gesture_count"] == 2
    assert smoke_sequence["drag_move_count"] == 5
    assert smoke_sequence["expected_position"] == payload["position"]
    assert smoke_sequence["actual_position"] == payload["position"]
    assert smoke_sequence["pointer_detail_values"] == [0]
    assert smoke_sequence["pointermove_button_values"] == [-1]
    assert smoke_sequence["pointer_id_values"] == [1]
    assert smoke_sequence["screen_coordinate_events"] >= 8
    assert payload["page_event_types"].count("double") == 1
    last_press = max(
        index
        for index, event_type in enumerate(payload["page_event_types"])
        if event_type == "press"
    )
    drag_tail = payload["page_event_types"][last_press:]
    assert drag_tail[0] == "press"
    assert drag_tail[-1] == "release"
    # 页面在 pointerup 前会补发最后一个 move，故桥接侧至少有合成的
    # 五次 move，允许多出该终态补发，但中间不能混入其他事件。
    assert len(drag_tail) >= 7
    assert set(drag_tail[1:-1]) == {"move"}


def test_drag_autonomous_competition_keeps_latest_pointer_target() -> None:
    """高频页面拖动不能被自主漫游回弹或遗留动作状态抢占。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["QT_QUICK_BACKEND"] = "software"
    environment["XDG_SESSION_TYPE"] = "x11"
    environment.pop("WAYLAND_DISPLAY", None)
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader --disable-gpu-compositing --disable-gpu-sandbox"
    )
    source_root = project_root / "src"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "timeout",
            "--foreground",
            "-k",
            "5s",
            "35s",
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1440x900x24 -nolisten tcp",
            sys.executable,
            str(project_root / "scripts" / "drag_autonomous_competition_smoke.py"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok", payload
    assert payload["model_ready"] is True
    assert payload["page_bridge_ready"] is True
    assert payload["autonomous_before_drag"] > 0
    assert payload["autonomous_accepted_during_drag"] == 0
    assert payload["autonomous_after_release"] > 0
    assert payload["guard_probe"]["status"] == "cancelled"
    assert payload["release_error_px"] <= 18
    assert payload["drag_targets"] >= 20
    assert payload["max_drag_jump"] <= 36
    assert payload["visual_drag_samples"] >= 3
    assert payload["visual_dragging_false"] == 0
    assert payload["visual_continuity_rejects"] == 0
    assert payload["visual_scale_jump"] <= 0.02


def test_runtime_continuous_submit_waits_for_cancel_before_new_generation() -> None:
    """连续发送时新回合不能被旧回合的异步取消标记为 stale。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["XDG_SESSION_TYPE"] = "x11"
    environment.pop("WAYLAND_DISPLAY", None)
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader "
        "--disable-gpu-compositing --disable-gpu-sandbox"
    )
    old_python_path = environment.get("PYTHONPATH")
    source_root = project_root / "src"
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
            str(project_root / "scripts" / "runtime_submission_smoke.py"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payloads = [
        ast.literal_eval(line)
        for line in result.stdout.splitlines()
        if line.lstrip().startswith("{")
    ]
    assert payloads, result.stdout
    payload = next(item for item in payloads if "calls" in item)
    assert payload["calls"] == 2
    assert payload["memory_calls"] == 1
    assert payload["generations"][1] > payload["generations"][0]
    assert payload["inputs"] == ["第一条", "第三条"]
    assert "旧回合首段" in payload["texts"]
    assert "新回合首段" in payload["texts"]
    assert payload["phase"] == "completed"


def test_app_run_signal_closes_runtime() -> None:
    """真实 app.run 收到 SIGTERM 后必须完成 RuntimeLoop 收尾并返回零。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    script = r"""
import os
import signal
import tempfile
from pathlib import Path

from PySide6.QtCore import QTimer

from config.loader import LoadedConfiguration, default_configuration_values
from config.resources import inspect_resources
import gui.qt6.app as app_module

root = Path(os.environ["MEAPET_TEST_ROOT"])
values = default_configuration_values(resource_root=root / "resources")
values["tts"] = {"enabled": False, "backend": "text_only", "language": "zh"}
values["asr"] = {"enabled": False, "capture": {"enabled": False}}
values["rendering"] = {
    "backend": "sprite",
    "resource_root": str(root / "resources"),
    "model": "",
    "sprite_scale": 0.6,
}
values["behavior"] = {"enabled": False}
values["watcher"] = {"enabled": False}
values["scheduler"] = {"enabled": False, "activity": {"enabled": False}}
values["config"] = {"reload": {"enabled": False}}
values["ui"] = {
    "qt_platform": "xcb",
    "auto_open_model_setup": False,
    "always_on_top": False,
    "window_locked": False,
}
database = Path(tempfile.mkdtemp(prefix="meapet-app-run-signal-")) / "signal.sqlite3"
values["storage"] = {"database": str(database)}
configuration = LoadedConfiguration(root / "config/app.example.yaml", values)
inventory = inspect_resources(root / "resources")
original_install = app_module._install_shutdown_signal_handlers
def install_and_schedule_shutdown(app):
    restore = original_install(app)
    QTimer.singleShot(500, lambda: os.kill(os.getpid(), signal.SIGTERM))
    return restore
app_module._install_shutdown_signal_handlers = install_and_schedule_shutdown
code = app_module.run(configuration, inventory)
if code != 0:
    raise SystemExit(code)
print("app-run-signal-ok")
"""
    environment = os.environ.copy()
    environment.update(
        {
            "MEAPET_TEST_ROOT": str(project_root),
            "QT_QPA_PLATFORM": "xcb",
            "XDG_SESSION_TYPE": "x11",
            "QT_OPENGL": "software",
            "PYTHONPATH": str(project_root / "src"),
        }
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24 -nolisten tcp",
            sys.executable,
            "-c",
            script,
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "app-run-signal-ok" in result.stdout
    assert "Traceback" not in result.stderr


def test_app_run_signal_closes_active_ptt_future() -> None:
    """app.run 退出时必须先关闭活动 PTT Future，再回收播放器与运行时。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    script = r"""
import concurrent.futures
import json
import os
import signal
import tempfile
from pathlib import Path

from PySide6.QtCore import QTimer

from config.loader import LoadedConfiguration, default_configuration_values
from config.resources import inspect_resources
import gui.qt6.app as app_module

events = []

class Signal:
    def __init__(self):
        self._callbacks = []
    def connect(self, callback):
        self._callbacks.append(callback)
    def emit(self, *args):
        for callback in tuple(self._callbacks):
            callback(*args)

class Capture:
    def __init__(self, **_kwargs):
        self.state = "idle"
        self.recording = False
        self.inputDevicesChanged = Signal()
        self.stateChanged = Signal()
        self.captureReady = Signal()
        self.input_device_choices = ()
        self.input_devices_loaded = False
        self.closed = False
    def public_status(self):
        return {"enabled": True, "recording": False}
    def load_input_devices(self):
        self.input_devices_loaded = True
        return ()
    def configure(self, **_kwargs):
        return None
    def close(self):
        if self.closed:
            return
        self.closed = True
        events.append("capture_close")

class Player:
    def __init__(self):
        self.playbackStateChanged = Signal()
        self.playing = False
        self.closed = False
    def __call__(self, _chunk):
        return None
    def diagnostics(self):
        return {"status": "idle", "available": True, "active": False}
    def stop_all(self):
        return None
    def close(self):
        if self.closed:
            return
        self.closed = True
        events.append("audio_close")

class Controller:
    instance = None
    def __init__(self, capture, **_kwargs):
        type(self).instance = self
        self.capture = capture
        self.state = "idle"
        self.transcribing = False
        self.blocks_playback = False
        self.stateChanged = Signal()
        self.transcriptionReady = Signal()
        self.future = None
        self.closed = False
    def toggle(self):
        return {"status": self.state}
    def activate(self):
        self.future = concurrent.futures.Future()
        self.state = "transcribing"
        self.transcribing = True
        self.blocks_playback = True
        self.stateChanged.emit(self.state, "testing")
    def close(self):
        if self.closed:
            return
        self.closed = True
        events.append("controller_close")
        if self.future is not None:
            self.future.cancel()
        self.capture.close()
        self.state = "closed"
        self.transcribing = False
        self.blocks_playback = False
        self.stateChanged.emit(self.state, "closed")

app_module.QtAudioPlayer = Player
app_module.QtMicrophoneCapture = Capture
app_module.QtPushToTalkController = Controller

root = Path(os.environ["MEAPET_TEST_ROOT"])
values = default_configuration_values(resource_root=root / "resources")
values["tts"] = {"enabled": False, "backend": "text_only", "language": "zh"}
values["asr"] = {"enabled": False, "capture": {"enabled": True}}
values["rendering"] = {
    "backend": "sprite",
    "resource_root": str(root / "resources"),
    "model": "",
    "sprite_scale": 0.6,
}
values["behavior"] = {"enabled": False}
values["watcher"] = {"enabled": False}
values["scheduler"] = {"enabled": False, "activity": {"enabled": False}}
values["config"] = {"reload": {"enabled": False}}
values["ui"] = {
    "qt_platform": "xcb",
    "auto_open_model_setup": False,
    "always_on_top": False,
    "window_locked": False,
}
database = Path(tempfile.mkdtemp(prefix="meapet-app-run-ptt-")) / "ptt.sqlite3"
values["storage"] = {"database": str(database)}
configuration = LoadedConfiguration(root / "config/app.example.yaml", values)
inventory = inspect_resources(root / "resources")
original_install = app_module._install_shutdown_signal_handlers
def install_and_schedule_shutdown(app):
    restore = original_install(app)
    QTimer.singleShot(300, lambda: Controller.instance.activate())
    QTimer.singleShot(700, lambda: os.kill(os.getpid(), signal.SIGTERM))
    return restore
app_module._install_shutdown_signal_handlers = install_and_schedule_shutdown
code = app_module.run(configuration, inventory)
payload = {
    "code": code,
    "events": events,
    "future_cancelled": bool(
        Controller.instance
        and Controller.instance.future
        and Controller.instance.future.cancelled()
    ),
}
print(json.dumps(payload, sort_keys=True))
if (
    code != 0
    or payload["events"] != ["controller_close", "capture_close", "audio_close"]
    or not payload["future_cancelled"]
):
    raise SystemExit(1)
"""
    environment = os.environ.copy()
    environment.update(
        {
            "MEAPET_TEST_ROOT": str(project_root),
            "QT_QPA_PLATFORM": "xcb",
            "XDG_SESSION_TYPE": "x11",
            "QT_OPENGL": "software",
            "PYTHONPATH": str(project_root / "src"),
        }
    )
    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1024x768x24 -nolisten tcp",
            sys.executable,
            "-c",
            script,
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "code": 0,
        "events": ["controller_close", "capture_close", "audio_close"],
        "future_cancelled": True,
    }
    assert "Traceback" not in result.stderr


def test_app_local_web_sink_starts_dispatches_and_closes() -> None:
    """显式启用的本地 API 必须经 sink 持有、主线程调度并在退出时回收。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["QT_OPENGL"] = "software"
    environment["XDG_SESSION_TYPE"] = "x11"
    environment.pop("WAYLAND_DISPLAY", None)
    source_root = project_root / "src"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "timeout",
            "--foreground",
            "-k",
            "5s",
            "20s",
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1440x900x24",
            sys.executable,
            str(project_root / "scripts" / "local_web_lifecycle_smoke.py"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=28,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["lifecycle"] == ["started", "closed"]
    assert payload["state_status"] == 200
    assert payload["action_status"] == 200
    assert payload["events_status"] == 200


def test_app_local_web_enabled_without_sink_keeps_zero_listener() -> None:
    """仅 YAML 开关不能开放无 capability 持有者的 HTTP 监听。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["QT_OPENGL"] = "software"
    environment["XDG_SESSION_TYPE"] = "x11"
    environment.pop("WAYLAND_DISPLAY", None)
    source_root = project_root / "src"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "timeout",
            "--foreground",
            "-k",
            "5s",
            "20s",
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1440x900x24",
            sys.executable,
            str(project_root / "scripts" / "local_web_lifecycle_smoke.py"),
            "--without-sink",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=28,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["listener"] == "absent"
    assert payload["lifecycle"] == []


def test_app_console_feedback_shutdown_does_not_touch_discarded_page() -> None:
    """点击反馈的延迟清理不能在 WebEngine 退出阶段再次执行 JavaScript。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    # Xvfb 没有真实 GPU/合成器；Qt Quick 的默认图形线程在整套回归中
    # 可能让 WebEngine 首帧延迟到点击脚本超时。与虚拟窗口其它入口一致，
    # 显式使用 Qt Quick software，测试关注的是页面/窗口协议而非 GPU 驱动。
    environment["QT_QUICK_BACKEND"] = "software"
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader "
        "--disable-gpu-compositing --disable-gpu-sandbox"
    )
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
            str(project_root / "scripts" / "app_console_e2e_smoke.py"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        # 整套 Qt/WebEngine 回归同时运行时，首帧编译和窗口合成可能延迟；
        # smoke 采用 45 秒有界窗口，这里留出进程清理余量，避免偶发误报。
        timeout=75,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\\nstderr={result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["model_setup_non_modal"] is True
    assert payload["model_setup_visible"] is True or payload["model_channel_ready"] is True
    assert payload["model_setup_topmost"] is True
    assert payload["console_topmost"] is True
    assert payload["console_pet_overlap"] is False
    assert payload["model_setup_pet_overlap"] is False
    assert payload["layout_in_screen"] is True
    assert payload["layout_screenshot"] == "/tmp/meapet-console-layout-e2e.png"
    assert "disabled in Discarded state" not in result.stderr


def test_app_parts_feedback_reaches_console_and_web_toast(tmp_path: Path) -> None:
    """真实 app.run 的头部、身体和左右腿反馈不能只停留在 renderer 层。"""

    if shutil.which("xvfb-run") is None:
        return
    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "xcb"
    environment["XDG_SESSION_TYPE"] = "x11"
    environment.pop("WAYLAND_DISPLAY", None)
    environment["MEAPET_WEBENGINE_SOFTWARE"] = "1"
    environment["QTWEBENGINE_CHROMIUM_FLAGS"] = (
        "--enable-transparent-visuals --use-gl=angle --use-angle=swiftshader "
        "--enable-unsafe-swiftshader "
        "--disable-gpu-compositing --disable-gpu-sandbox"
    )
    source_root = project_root / "src"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [
            "timeout",
            "--foreground",
            "-k",
            "5s",
            "55s",
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1440x900x24",
            sys.executable,
            str(project_root / "scripts" / "app_parts_e2e_smoke.py"),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=65,
        check=False,
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert [item["part"] for item in payload["parts"]] == [
        "head",
        "body",
        "lower_left",
        "lower_right",
    ]
    assert all(item["feedback_ok"] and item["toast_ok"] for item in payload["parts"])
    assert all(
        "happy" not in str(item["toast"]).lower() and "wave" not in str(item["toast"]).lower()
        for item in payload["parts"]
    )
