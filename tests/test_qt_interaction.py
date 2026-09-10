from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

from gui.qt6.app import _tts_speaking_for_snapshot, _TTSActivityTracker


def test_pet_window_focus_probe_fails_closed_for_non_focusable_or_invalid_windows() -> None:
    """桌宠展示窗口不应因恢复入口触发隐式 requestActivate。"""

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.app import _window_accepts_focus

    app = QApplication.instance() or QApplication([])
    blocked = QWidget()
    blocked.setWindowFlags(Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
    focusable = QWidget()
    focusable.setWindowFlags(Qt.WindowType.Tool)

    assert _window_accepts_focus(blocked) is False
    assert _window_accepts_focus(focusable) is True
    assert _window_accepts_focus(None) is False

    class Invalid:
        def windowFlags(self):
            raise RuntimeError("window has already been destroyed")

    assert _window_accepts_focus(Invalid()) is False
    blocked.close()
    focusable.close()
    app.processEvents()


def test_opengl_host_resource_reload_clears_sprite_caches(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    source = Path(__file__).resolve().parents[1] / "resources" / "sprites" / "mea01A_001.webp"
    sprite_root = tmp_path / "sprites"
    sprite_root.mkdir()
    (sprite_root / "frame.webp").write_bytes(source.read_bytes())
    renderer = SpriteRenderer(sprite_root)
    resized: list[tuple[int, int]] = []
    updated: list[bool] = []
    host = SimpleNamespace(
        _shutdown=False,
        _renderer_dragging=False,
        _drag_transaction_active=lambda: False,
        renderer=renderer,
        _sprite_scale=0.6,
        _sprite_image_path=Path("old.webp"),
        _sprite_image=object(),
        _sprite_scaled_image=object(),
        _sprite_scaled_path=Path("old.webp"),
        _sprite_scaled_area=(1, 1),
        _sprite_previous_image_path=Path("previous.webp"),
        _sprite_previous_image=object(),
        _sprite_previous_scaled_image=object(),
        _sprite_previous_scaled_path=Path("previous.webp"),
        _sprite_previous_scaled_area=(1, 1),
        _sprite_reference_size=(1, 1),
        _preferred_size=lambda: (320, 440),
        resize=lambda width, height: resized.append((width, height)),
        _invalidate_geometry=lambda: 1,
        _surface_mask_enabled=False,
        update=lambda: updated.append(True),
    )

    invalid_root = tmp_path / "invalid"
    invalid_root.mkdir()
    cached_image = host._sprite_image
    rejected = PetOpenGLWindow.reload_resources(host, invalid_root, sprite_scale=0.8)
    assert rejected["status"] == "unavailable"
    assert host._sprite_image is cached_image
    assert resized == []

    result = PetOpenGLWindow.reload_resources(host, tmp_path, sprite_scale=0.8)

    assert result["status"] == "reloaded"
    assert host._sprite_scale == 0.8
    assert host._sprite_image_path is None
    assert host._sprite_image is None
    assert host._sprite_scaled_image is None
    assert host._sprite_previous_image is None
    assert host._sprite_previous_scaled_image is None
    assert host._sprite_reference_size is None
    assert resized == [(320, 440)]
    assert updated == [True]

    host._renderer_dragging = True
    assert PetOpenGLWindow.reload_resources(host, tmp_path)["status"] == "busy"


def test_opengl_host_crossfades_cached_previous_and_current_sprite_frames() -> None:
    from types import SimpleNamespace

    import pytest

    from core.rendering.actions import ExpressionRequest
    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    sprite_root = Path(__file__).resolve().parents[1] / "resources" / "sprites"
    renderer = SpriteRenderer(sprite_root)
    request = ExpressionRequest.from_arguments(
        {
            "name": "happy",
            "duration_seconds": 0.5,
            "transition_seconds": 0.2,
        }
    )
    assert renderer.set_expression_request(request)["status"] == "started"
    renderer.advance(0.1)

    class Frame:
        def width(self) -> int:
            return 100

        def height(self) -> int:
            return 100

    class Painter:
        def __init__(self) -> None:
            self.opacities: list[float] = []
            self.draws: list[object] = []
            self.ended = False

        def setOpacity(self, value: float) -> None:  # noqa: N802
            self.opacities.append(float(value))

        def drawImage(self, _target, image) -> None:  # noqa: N802
            self.draws.append(image)

        def end(self) -> None:
            self.ended = True

    frame = Frame()
    painter = Painter()
    host = SimpleNamespace(
        renderer=renderer,
        width=lambda: 200,
        height=lambda: 250,
        _sprite_image_path=None,
        _sprite_image=None,
        _sprite_scaled_image=None,
        _sprite_scaled_path=None,
        _sprite_scaled_area=None,
        _sprite_previous_image_path=None,
        _sprite_previous_image=None,
        _sprite_previous_scaled_image=None,
        _sprite_previous_scaled_path=None,
        _sprite_previous_scaled_area=None,
        _sprite_reference_size=(100, 100),
        _scaled_sprite_frame=lambda *_args, **_kwargs: (frame, frame),
        _clear_sprite_previous_cache=lambda: None,
        _paint_device=lambda: (painter, object()),
    )

    PetOpenGLWindow._paint_sprite(host)

    assert painter.opacities == pytest.approx([0.5, 0.5, 1.0])
    assert painter.draws == [frame, frame]
    assert painter.ended is True


def test_opengl_host_reuses_decoded_and_scaled_sprite_cache() -> None:
    from types import MethodType, SimpleNamespace

    from gui.qt6.opengl_host import PetOpenGLWindow

    frame_path = Path(__file__).resolve().parents[1] / "resources" / "sprites" / "mea01A_001.webp"
    host = SimpleNamespace(
        _sprite_image_path=None,
        _sprite_image=None,
        _sprite_scaled_image=None,
        _sprite_scaled_path=None,
        _sprite_scaled_area=None,
        _sprite_previous_image_path=None,
        _sprite_previous_image=None,
        _sprite_previous_scaled_image=None,
        _sprite_previous_scaled_path=None,
        _sprite_previous_scaled_area=None,
        _sprite_reference_size=None,
    )
    host._sprite_frame_image = MethodType(PetOpenGLWindow._sprite_frame_image, host)

    decoded_first = PetOpenGLWindow._sprite_frame_image(host, frame_path)
    decoded_second = PetOpenGLWindow._sprite_frame_image(host, frame_path)
    scaled_first = PetOpenGLWindow._scaled_sprite_frame(host, frame_path, 240, 320)
    scaled_second = PetOpenGLWindow._scaled_sprite_frame(host, frame_path, 240, 320)

    assert decoded_first is not None
    assert decoded_second is decoded_first
    assert scaled_first is not None
    assert scaled_second is not None
    assert scaled_second[1] is scaled_first[1]


def test_opengl_host_forwards_structured_expression_and_motion_requests() -> None:
    from types import SimpleNamespace

    from core.rendering.actions import ExpressionRequest, MotionRequest
    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    class Signal:
        def __init__(self) -> None:
            self.values: list[str] = []

        def emit(self, value: str) -> None:
            self.values.append(str(value))

    sprite_root = Path(__file__).resolve().parents[1] / "resources" / "sprites"
    renderer = SpriteRenderer(sprite_root)
    expression_signal = Signal()
    motion_signal = Signal()
    updates: list[bool] = []
    host = SimpleNamespace(
        renderer=renderer,
        expressionChanged=expression_signal,
        motionChanged=motion_signal,
        update=lambda: updates.append(True),
    )

    expression_result = PetOpenGLWindow.set_expression_request(
        host,
        ExpressionRequest.from_arguments({"name": "happy"}),
    )
    motion_result = PetOpenGLWindow.play_motion_request(
        host,
        MotionRequest.from_arguments({"name": "wave", "duration_seconds": 0.2}),
    )

    assert expression_result["status"] == "started"
    assert motion_result["status"] == "started"
    assert expression_signal.values == ["happy"]
    assert motion_signal.values == ["wave"]
    assert updates == [True, True]


def test_tts_activity_tracker_does_not_silence_between_serial_sentences() -> None:
    """同一回合的首句完成后，下一句开始会使旧 Silence 回调失效。"""

    tracker = _TTSActivityTracker()
    context = object()

    tracker.started(context)
    first_silence = tracker.finished(context)
    assert first_silence is not None

    tracker.started(context)
    assert tracker.can_silence(first_silence) is False
    second_silence = tracker.finished(context)
    assert second_silence is not None
    assert tracker.can_silence(first_silence) is False
    assert tracker.can_silence(second_silence) is True


def test_tts_activity_tracker_waits_for_all_contexts_before_silence() -> None:
    tracker = _TTSActivityTracker()
    first = object()
    second = object()

    tracker.started(first)
    tracker.started(second)
    assert tracker.finished(first) is None
    final_generation = tracker.finished(second)
    assert final_generation is not None
    assert tracker.can_silence(final_generation) is True


def test_tts_speaking_ignores_stale_sentence_after_terminal_state() -> None:
    class Snapshot:
        speech_text = "上一句。"
        phase = "completed"

    assert _tts_speaking_for_snapshot(Snapshot(), active=True) is False
    assert _tts_speaking_for_snapshot(Snapshot(), active=False) is False
    assert _tts_speaking_for_snapshot(Snapshot(), active=False, pending_sentence=True) is False

    class StreamingSnapshot:
        phase = "streaming"

    assert (
        _tts_speaking_for_snapshot(StreamingSnapshot(), active=False, pending_sentence=True) is True
    )


def test_opengl_window_drag_and_context_signal(tmp_path: Path) -> None:
    """实际 Qt 事件链应移动窗口并向宿主发出右键菜单请求。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    sprite_dir = project_root / "resources" / "sprites"
    script = f"""
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QGuiApplication, QKeyEvent, QMouseEvent
from PySide6.QtTest import QTest
from gui.qt6.opengl_host import PetOpenGLWindow
from gui.renderers.sprite import SpriteRenderer

app = QGuiApplication.instance() or QGuiApplication([])
window = PetOpenGLWindow(SpriteRenderer({str(sprite_dir)!r}))
window.setPosition(100, 120)
seen = []
console_seen = []
clicks = []
window.set_click_callback(lambda payload: clicks.append(dict(payload)))
window.contextMenuRequested.connect(lambda point: seen.append((point.x(), point.y())))
window.consoleRequested.connect(lambda: console_seen.append(True))
window.show()
app.processEvents()
assert window.flags() & Qt.WindowDoesNotAcceptFocus
mask_result = window.set_surface_mask_enabled(True)
mask_rect = window.mask().boundingRect()
assert mask_result["enabled"] is True
assert not window.mask().isEmpty()
assert mask_rect.left() > 0
assert mask_rect.top() >= 50
assert mask_rect.right() < window.width()
assert mask_rect.bottom() < window.height()
window.set_speech("气泡测试")
window._refresh_surface_mask(force=True)
speech_mask_rect = window.mask().boundingRect()
assert speech_mask_rect.left() <= 8
assert speech_mask_rect.top() <= 8
assert speech_mask_rect.width() >= window.width() - 16
window.set_speech("", visible=False)

def send(kind, x, y, button, buttons, require_accept=True):
    event = QMouseEvent(
        kind,
        QPointF(x, y),
        QPointF(x, y),
        QPointF(x, y),
        button,
        buttons,
        Qt.NoModifier,
    )
    app.sendEvent(window, event)
    app.processEvents()
    if require_accept:
        assert event.isAccepted()

# 拖动必须从真实精灵像素启动；窗口透明角落不应抢走桌面拖动状态。
send(QEvent.Type.MouseButtonPress, 180, 260, Qt.LeftButton, Qt.LeftButton)
send(QEvent.Type.MouseMove, 210, 290, Qt.NoButton, Qt.LeftButton)
assert window.position() == (QPointF(130, 150).toPoint())
send(QEvent.Type.MouseButtonRelease, 210, 290, Qt.LeftButton, Qt.NoButton)
send(QEvent.Type.MouseButtonPress, 166, 83, Qt.LeftButton, Qt.LeftButton)
send(QEvent.Type.MouseButtonRelease, 166, 83, Qt.LeftButton, Qt.NoButton)
QTest.qWait(800)
assert clicks and clicks[-1]["zone"] == "upper"
transparent_clicks = len(clicks)
before_transparent_drag = window.position()
send(QEvent.Type.MouseButtonPress, 12, 14, Qt.LeftButton, Qt.LeftButton, False)
send(QEvent.Type.MouseMove, 42, 49, Qt.NoButton, Qt.LeftButton)
assert window.position() == before_transparent_drag
send(QEvent.Type.MouseButtonRelease, 42, 49, Qt.LeftButton, Qt.NoButton)
send(QEvent.Type.MouseButtonPress, 12, 14, Qt.LeftButton, Qt.LeftButton, False)
send(QEvent.Type.MouseButtonRelease, 12, 14, Qt.LeftButton, Qt.NoButton)
QTest.qWait(500)
assert len(clicks) == transparent_clicks
transparent_double_count = len(console_seen)
send(QEvent.Type.MouseButtonDblClick, 12, 14, Qt.LeftButton, Qt.LeftButton)
assert len(console_seen) == transparent_double_count
send(QEvent.Type.MouseButtonDblClick, 166, 83, Qt.LeftButton, Qt.LeftButton)
assert len(console_seen) == transparent_double_count + 1
send(QEvent.Type.MouseButtonPress, 42, 49, Qt.RightButton, Qt.RightButton, False)
assert seen == []
send(QEvent.Type.MouseButtonPress, 180, 260, Qt.RightButton, Qt.RightButton)
assert seen == [(180, 260)]
locked = window.set_window_locked(True)
assert locked["locked"] is True
assert window.is_user_interacting() is False
locked_clicks = len(clicks)
send(QEvent.Type.MouseButtonPress, 166, 83, Qt.LeftButton, Qt.LeftButton, False)
send(QEvent.Type.MouseButtonRelease, 166, 83, Qt.LeftButton, Qt.NoButton, False)
QTest.qWait(500)
assert len(clicks) == locked_clicks
assert window.set_window_locked(False)["locked"] is False
key = QKeyEvent(
    QEvent.Type.KeyPress,
    Qt.Key_M,
    Qt.ControlModifier | Qt.ShiftModifier,
)
app.sendEvent(window, key)
assert key.isAccepted()
assert len(console_seen) == 2
window.close()
app.processEvents()
print("qt-interaction-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-interaction-ok" in result.stdout


def test_opengl_drag_coalesces_burst_and_flushes_latest_target() -> None:
    """QOpenGLWindow 高频拖动只提交最新目标，释放前可强制冲刷。"""

    try:
        from PySide6.QtCore import QPoint
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    calls: list[tuple[int, int]] = []
    apply = window._apply_drag_target

    def record(target: QPoint) -> None:
        calls.append((target.x(), target.y()))
        apply(target)

    window._apply_drag_target = record  # type: ignore[method-assign]
    window._move_drag_target(QPoint(100, 100))
    window._move_drag_target(QPoint(101, 101))
    window._move_drag_target(QPoint(102, 102))
    assert calls == [(100, 100)]
    assert window._pending_drag_target == (102, 102)
    window._flush_drag_target(force=True)
    assert calls == [(100, 100), (102, 102)]
    assert window._pending_drag_target is None
    window.shutdown()
    window.close()
    app.processEvents()


def test_opengl_press_interrupts_behavior_once_per_gesture() -> None:
    """原生 OpenGL 按下边沿应立即停止后台行为规划。"""

    try:
        from PySide6.QtCore import QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    window.setPosition(100, 120)
    window._model_hit_test = lambda _point: True  # type: ignore[method-assign]
    interruptions: list[str] = []
    window.set_interaction_interrupt(lambda: interruptions.append("interrupt"))

    def event(kind, x, y, button, buttons):
        return QMouseEvent(
            kind,
            QPointF(x, y),
            QPointF(x + 100, y + 120),
            QPointF(x + 100, y + 120),
            button,
            buttons,
            Qt.NoModifier,
        )

    try:
        window.mousePressEvent(
            event(
                QMouseEvent.Type.MouseButtonPress,
                180,
                260,
                Qt.LeftButton,
                Qt.LeftButton,
            )
        )
        assert interruptions == ["interrupt"]
        window.mouseReleaseEvent(
            event(
                QMouseEvent.Type.MouseButtonRelease,
                180,
                260,
                Qt.LeftButton,
                Qt.NoButton,
            )
        )
        window.mousePressEvent(
            event(
                QMouseEvent.Type.MouseButtonPress,
                180,
                260,
                Qt.LeftButton,
                Qt.LeftButton,
            )
        )
        assert interruptions == ["interrupt", "interrupt"]
    finally:
        window.shutdown()
        window.close()
        app.processEvents()


def test_opengl_autonomous_step_does_not_reset_renderer_tracking() -> None:
    """自主小步移动不应把 Live2D 注视目标重置为中性。"""

    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    notifications: list[bool] = []

    class Platform:
        def move_overlay(self, _window, x, y):
            return {"status": "available", "x": x, "y": y}

    window.renderer.notify_window_moved = lambda: notifications.append(True)  # type: ignore[attr-defined]
    window.platform = Platform()
    result = window.move_to(120, 140, duration_ms=0)

    assert result["status"] == "available"
    assert notifications == []
    window.shutdown()
    window.close()
    app.processEvents()


def test_opengl_cursor_tracking_ignores_expected_autonomous_move(monkeypatch) -> None:
    """光标轮询不能把自主窗口小步误报成外部位移。"""

    try:
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    window.setPosition(20, 30)
    window._last_window_position = (20, 30)
    notifications: list[bool] = []
    window.renderer.notify_window_moved = lambda: notifications.append(True)  # type: ignore[attr-defined]
    window.renderer.set_cursor_target = lambda *_values: True  # type: ignore[attr-defined]

    class Platform:
        def move_overlay(self, target, x, y):
            target.setPosition(x, y)
            return {"status": "available"}

    window.platform = Platform()
    result = window.move_to(48, 60, duration_ms=0)
    assert result["status"] == "available"
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: window.position()))
    assert window.update_cursor_tracking() is True
    assert notifications == []
    window.shutdown()
    window.close()
    app.processEvents()


def test_sprite_window_uses_reduced_idle_render_rate() -> None:
    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    try:
        assert window._render_interval_ms == 33
        assert window._timer.interval() == 33
    finally:
        window.shutdown()
        window.close()
        app.processEvents()


def test_opengl_release_establishes_new_cursor_tracking_baseline() -> None:
    """拖动释放后的首个追踪 tick 不应重复清空模型速度。"""

    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    window.setPosition(20, 30)
    notifications: list[bool] = []
    window.renderer.notify_window_moved = lambda: notifications.append(True)  # type: ignore[attr-defined]
    window.renderer.set_cursor_target = lambda *_values: True  # type: ignore[attr-defined]
    window._renderer_dragging = True
    window.setPosition(80, 90)
    window._set_renderer_dragging(False)
    assert window.is_user_interacting() is True

    # 位置已被 _set_renderer_dragging(False) 记录为新基线，追踪只更新
    # 当前目标，不再额外触发一次窗口位移通知。
    assert window.update_cursor_tracking() is True
    assert notifications == []
    window.shutdown()
    window.close()
    app.processEvents()


def test_opengl_cursor_tracking_honors_renderer_dragging_flag(monkeypatch) -> None:
    """渲染器拖动态是最后一道追踪闸门。"""

    try:
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    targets: list[tuple[object, ...]] = []
    window.renderer.set_cursor_target = lambda *values: targets.append(values)  # type: ignore[attr-defined]
    window._renderer_dragging = True
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: window.position()))
    assert window.update_cursor_tracking() is True
    assert targets == []
    window.shutdown()
    window.close()
    app.processEvents()


def test_opengl_cursor_tracking_retries_after_renderer_rejects_target(monkeypatch) -> None:
    """原生渲染器拒绝注视坐标后，下一 tick 必须再次提交。"""

    try:
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    window.setPosition(20, 30)
    calls: list[tuple[object, ...]] = []

    def setter(*values):
        calls.append(values)
        return False if len(calls) == 1 else None

    window.renderer.set_cursor_target = setter  # type: ignore[attr-defined]
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(40, 50)))

    assert window.update_cursor_tracking() is False
    assert window._last_cursor_target is None
    assert window.update_cursor_tracking() is True
    assert len(calls) == 2
    assert window._last_cursor_target == (20, 20, window.width(), window.height())
    window.shutdown()
    window.close()
    app.processEvents()


def test_opengl_delayed_position_restore_cannot_overwrite_new_drag_geometry() -> None:
    """窗口标志重建的零延迟恢复不能把用户的新位置拉回旧坐标。"""

    try:
        from PySide6.QtCore import QPoint
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    generation = window._geometry_generation
    window._restore_position(QPoint(40, 50), geometry_generation=generation)
    window.setPosition(QPoint(180, 210))
    window._invalidate_geometry()
    QTest.qWait(20)

    assert window.position() == QPoint(180, 210)
    window.shutdown()
    window.close()
    app.processEvents()


def test_opengl_async_move_result_is_discarded_after_geometry_superseded() -> None:
    """异步平台位置回执不能覆盖开始后的用户拖动事务。"""

    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])

    class Platform:
        def __init__(self) -> None:
            self.ready = asyncio.Event()

        def move_overlay(self, _window, _x, _y):
            async def wait_for_result():
                await self.ready.wait()
                return {"status": "available"}

            return wait_for_result()

    async def run() -> dict[str, object]:
        platform = Platform()
        window = PetOpenGLWindow(
            SpriteRenderer(project_root / "resources" / "sprites"),
            platform=platform,
        )
        operation = window.move_to(80, 90)
        task = asyncio.create_task(operation)
        await asyncio.sleep(0)
        window._invalidate_geometry()
        platform.ready.set()
        result = await task
        window.shutdown()
        window.close()
        app.processEvents()
        return result

    result = asyncio.run(run())
    assert result["status"] == "cancelled"
    assert result["detail"] == "geometry transaction superseded"


def test_opengl_autonomous_step_holds_surface_mask_timer_until_settled() -> None:
    """自主步进稳定窗口内不应调用 Qt setMask 或原生输入 Shape。"""

    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])
    window = PetOpenGLWindow(SpriteRenderer(project_root / "resources" / "sprites"))
    window._surface_mask_enabled = True
    window._autonomous_geometry_until = time.monotonic() + 1.0
    calls: list[str] = []
    window.setMask = lambda _region: calls.append("mask")  # type: ignore[method-assign]
    window._set_platform_input_shape = lambda _region: calls.append("input")  # type: ignore[method-assign]

    assert window._refresh_surface_mask(force=True) is False
    assert calls == []
    window.shutdown()
    window.close()
    app.processEvents()


def test_opengl_autonomous_move_cancellation_releases_geometry_reservation() -> None:
    """OpenGL 自主移动的同步、未启动和已启动取消都不能泄漏 pending。"""

    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])

    class ViewPlatform:
        def move_overlay(self, *_args):
            raise asyncio.CancelledError

    window = PetOpenGLWindow(
        SpriteRenderer(project_root / "resources" / "sprites"),
        platform=ViewPlatform(),
    )
    try:
        try:
            window.move_to(80, 90, duration_ms=0)
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("move_to should propagate asyncio.CancelledError")
        assert window._autonomous_geometry_pending == 0
        assert window._autonomous_geometry_active() is True
    finally:
        window.shutdown()
        window.close()
        app.processEvents()


def test_opengl_autonomous_move_prestart_cancellation_does_not_claim_pending() -> None:
    """OpenGL 返回协程在首次调度前取消时不应遗留 pending。"""

    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])

    class Deferred:
        def __await__(self):
            if False:
                yield None
            return {"status": "available"}

    class Platform:
        def move_overlay(self, *_args):
            return Deferred()

    async def run() -> int:
        window = PetOpenGLWindow(
            SpriteRenderer(project_root / "resources" / "sprites"),
            platform=Platform(),
        )
        try:
            operation = window.move_to(80, 90, duration_ms=0)
            assert window._autonomous_geometry_pending == 0
            task = asyncio.create_task(operation)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("pre-start cancellation should propagate")
            return window._autonomous_geometry_pending
        finally:
            window.shutdown()
            window.close()
            app.processEvents()

    assert asyncio.run(run()) == 0


def test_opengl_autonomous_move_started_cancellation_releases_pending() -> None:
    """OpenGL 协程开始等待平台后取消，也必须归还 pending。"""

    try:
        from PySide6.QtWidgets import QApplication
    except (ImportError, ModuleNotFoundError, OSError):
        return

    from gui.qt6.opengl_host import PetOpenGLWindow
    from gui.renderers.sprite import SpriteRenderer

    project_root = Path(__file__).resolve().parents[1]
    app = QApplication.instance() or QApplication([])

    class Deferred:
        def __init__(self, event: asyncio.Event) -> None:
            self._event = event

        def __await__(self):
            return self._event.wait().__await__()

    class Platform:
        def __init__(self, event: asyncio.Event) -> None:
            self._event = event

        def move_overlay(self, *_args):
            return Deferred(self._event)

    async def run() -> int:
        event = asyncio.Event()
        window = PetOpenGLWindow(
            SpriteRenderer(project_root / "resources" / "sprites"),
            platform=Platform(event),
        )
        try:
            task = asyncio.create_task(window.move_to(80, 90, duration_ms=0))
            await asyncio.sleep(0)
            assert window._autonomous_geometry_pending == 1
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return window._autonomous_geometry_pending
        finally:
            window.shutdown()
            window.close()
            app.processEvents()

    assert asyncio.run(run()) == 0


def test_opengl_visual_mask_keeps_bubble_out_of_native_input_shape(tmp_path: Path) -> None:
    """OpenGL 视觉遮罩可包含气泡，但原生 Input Shape 只覆盖精灵。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    sprite_dir = project_root / "resources" / "sprites"
    script = f"""
from pathlib import Path
from PySide6.QtCore import QRect, Qt
from PySide6.QtWidgets import QApplication
from gui.qt6.opengl_host import PetOpenGLWindow
from gui.renderers.sprite import SpriteRenderer

app = QApplication.instance() or QApplication([])
events = []
class Platform:
    def __init__(self):
        self.click_through = False

    def set_click_through(self, window, enabled):
        self.click_through = bool(enabled)
        window.setFlag(Qt.WindowTransparentForInput, bool(enabled))
        return {{"status": "available", "enabled": bool(enabled)}}

    def set_input_shape(self, window, region):
        if region is None:
            events.append(None)
        elif isinstance(region, tuple) and not region:
            events.append(QRect(-1, -1, 1, 1))
        else:
            events.append(region.boundingRect())
        return {{"status": "available", "detail": "test"}}

platform = Platform()
window = PetOpenGLWindow(
    SpriteRenderer(Path({str(sprite_dir)!r})),
    platform=platform,
)
window._input_transparency_active = lambda: platform.click_through
window.show()
app.processEvents()
assert window.set_surface_mask_enabled(True)["enabled"] is True
status = window.surface_mask_status()
assert status["ready"] is True
assert status["input_ready"] is True
window.set_speech("气泡不会拦截桌面点击")
window._refresh_surface_mask(force=True)
visual = window.mask().boundingRect()
input_rect = events[-1]
assert input_rect is not None
assert visual.top() <= 8
assert input_rect.top() >= 50
assert input_rect != visual
assert window.set_click_through(True)["enabled"] is True
app.processEvents()
assert events[-1].top() == -1
assert window.set_click_through(False)["enabled"] is False
app.processEvents()
assert events[-1].top() >= 50
window.set_interaction_feedback({{"zone": "head", "phrase": "反馈牌"}})
window._refresh_surface_mask(force=True)
assert events[-1] is not None
assert events[-1].top() >= 50
disabled = window.set_surface_mask_enabled(False)
assert disabled["enabled"] is False
assert events[-1] is None
assert window.surface_mask_status()["ready"] is False
# 关闭视觉遮罩后再开启/解除点击穿透，输入区必须先变为空哨兵、再恢复整窗。
assert window.set_click_through(True)["enabled"] is True
assert events[-1].top() == -1
assert window.set_click_through(False)["enabled"] is False
assert events[-1] is None
window.close()
app.processEvents()
print("qt-opengl-input-shape-separation-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-opengl-input-shape-separation-ok" in result.stdout


def test_opengl_sprite_parts_follow_drawn_model_coordinates(tmp_path: Path) -> None:
    """精灵回退的部位反馈应以实际绘制内容坐标为准。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    sprite_dir = project_root / "resources" / "sprites"
    script = f"""
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication
from gui.qt6.opengl_host import PetOpenGLWindow
from gui.renderers.sprite import SpriteRenderer

app = QApplication.instance() or QApplication([])
window = PetOpenGLWindow(SpriteRenderer({str(sprite_dir)!r}), size=(420, 480))
window.show()
app.processEvents()
assert window._model_part(QPoint(210, 110)) == "head"
assert window._model_part(QPoint(210, 250)) == "body"
assert window._model_part(QPoint(190, 420)) == "lower_left"
assert window._model_part(QPoint(220, 420)) == "lower_right"
window.close()
app.processEvents()
print("qt-opengl-sprite-parts-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-opengl-sprite-parts-ok" in result.stdout
