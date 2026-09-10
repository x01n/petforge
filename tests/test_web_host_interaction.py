from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path


def test_web_host_qwebengine_topmost_is_deferred_and_preserves_hidden_state() -> None:
    """真实 WebEngine 置顶请求不能同步阻塞，也不能把隐藏窗口意外显示。"""

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint
            self._visible = False
            self._position = QPoint(80, 90)
            self.set_flag_calls: list[bool] = []

        def isVisible(self):
            return self._visible

        def show(self):
            self._visible = True

        def windowFlags(self):
            return self._flags

        def setWindowFlag(self, flag, enabled):
            self.set_flag_calls.append(bool(enabled))
            self._flags = self._flags | flag if enabled else self._flags & ~flag

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    renderer = Renderer()
    host = WebPetHost(renderer)
    first = host.set_always_on_top(True)
    second = host.set_always_on_top(False)
    assert first["status"] == "cancelled"
    assert "更新的窗口事务" in first["detail"]
    assert second["status"] == "requested"
    assert renderer.view.set_flag_calls == []
    assert host.is_always_on_top() is False

    loop = QEventLoop()
    QTimer.singleShot(0, loop.quit)
    loop.exec()
    assert renderer.view.set_flag_calls == [False]
    assert renderer.view.isVisible() is False
    assert host.is_always_on_top() is False

    host.shutdown()
    app.processEvents()


def test_web_host_qwebengine_topmost_reports_wayland_boundary_after_async_submit() -> None:
    """Wayland 置顶保持非阻塞，并在异步回读后保留降级边界。"""

    from time import monotonic

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from gui.platforms.linux import LinuxDesktopPlatform
    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint
            self._position = QPoint(40, 50)
            self._visible = True

        def isVisible(self):
            return self._visible

        def show(self):
            self._visible = True

        def windowFlags(self):
            return self._flags

        def setWindowFlag(self, flag, enabled):
            self._flags = self._flags | flag if enabled else self._flags & ~flag

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    platform = LinuxDesktopPlatform({"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0"})
    host = WebPetHost(Renderer(), platform=platform)
    started = monotonic()
    result = host.set_always_on_top(True)
    elapsed = monotonic() - started

    assert elapsed < 0.2
    assert result["status"] == "degraded"
    detail = str(result.get("detail", "")).casefold()
    assert "wayland" in detail
    assert "compositor" in detail
    assert host.is_always_on_top() is True

    loop = QEventLoop()
    QTimer.singleShot(80, loop.quit)
    loop.exec()
    assert result["status"] == "degraded"
    assert host.always_on_top_status()["status"] == "degraded"
    assert host.always_on_top_status()["enabled"] is True

    host.shutdown()
    app.processEvents()


def test_web_host_qwebengine_topmost_async_readback_rejects_dropped_flag() -> None:
    """异步置顶回读发现 Qt 丢旗标时应收敛为 unavailable。"""

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint
            self._position = QPoint(10, 20)
            self._visible = True

        def isVisible(self):
            return self._visible

        def show(self):
            self._visible = True

        def windowFlags(self):
            return self._flags

        def setWindowFlag(self, _flag, _enabled):
            # 模拟窗口系统忽略客户端的置顶请求。
            return None

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    host = WebPetHost(Renderer())
    result = host.set_always_on_top(True)
    assert result["status"] == "requested"

    loop = QEventLoop()
    QTimer.singleShot(80, loop.quit)
    loop.exec()
    assert result["status"] == "unavailable"
    assert result["enabled"] is False
    assert host.always_on_top_status()["status"] == "unavailable"
    assert host.is_always_on_top() is False

    host.shutdown()
    app.processEvents()


def test_web_host_qwebengine_topmost_async_accepts_window_flags_fallback() -> None:
    """旧 Qt 替身只有 setWindowFlags 时仍能完成异步置顶回读。"""

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint
            self._position = QPoint(5, 6)
            self._visible = True
            self.calls: list[object] = []

        def isVisible(self):
            return self._visible

        def show(self):
            self._visible = True

        def windowFlags(self):
            return self._flags

        def setWindowFlags(self, flags):
            self.calls.append(flags)
            self._flags = flags

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    renderer = Renderer()
    host = WebPetHost(renderer)
    result = host.set_always_on_top(True)
    assert result["status"] == "requested"

    loop = QEventLoop()
    QTimer.singleShot(80, loop.quit)
    loop.exec()
    assert result["status"] == "available"
    assert result["enabled"] is True
    assert renderer.view.calls
    assert host.always_on_top_status() == {"status": "available", "enabled": True}

    host.shutdown()
    app.processEvents()


def test_web_host_qwebengine_topmost_uses_native_qwindow_and_ewmh_readback(
    monkeypatch,
) -> None:
    """置顶切换必须保留 XID，并等待 EWMH 属性确认后才报告成功。"""

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class NativeWindow:
        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint
            self._win_id = 0x120034
            self.calls: list[bool] = []

        def flags(self):
            return self._flags

        def setFlag(self, flag, enabled):  # noqa: N802
            self.calls.append(bool(enabled))
            self._flags = self._flags | flag if enabled else self._flags & ~flag

        def winId(self):  # noqa: N802
            return self._win_id

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self.native = NativeWindow()
            self._position = QPoint(30, 40)
            self.widget_calls: list[bool] = []

        def windowHandle(self):  # noqa: N802
            return self.native

        def windowFlags(self):  # noqa: N802
            return Qt.WindowType.FramelessWindowHint

        def setWindowFlag(self, _flag, enabled):  # noqa: N802
            self.widget_calls.append(bool(enabled))

        def isVisible(self):  # noqa: N802
            return True

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    renderer = Renderer()
    host = WebPetHost(renderer)
    states = iter((False, True))
    monkeypatch.setattr(host, "_qt_uses_x11", lambda: True)
    monkeypatch.setattr(host, "_read_x11_topmost_state", lambda _view: next(states, True))

    original_id = renderer.view.native.winId()
    result = host.set_always_on_top(True)
    assert result["status"] == "requested"
    assert renderer.view.native.calls == []

    loop = QEventLoop()
    QTimer.singleShot(120, loop.quit)
    loop.exec()

    assert renderer.view.native.winId() == original_id
    assert renderer.view.native.calls == [True]
    assert renderer.view.widget_calls == []
    assert result == {"status": "available", "enabled": True}
    assert host.is_always_on_top() is True

    host.shutdown()
    app.processEvents()


def test_web_host_qwebengine_topmost_rejects_qt_flag_without_ewmh(
    monkeypatch,
) -> None:
    """Qt flag 为真但 EWMH 仍为空时，终态必须明确报告未置顶。"""

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    import gui.qt6.web_host as web_host_module
    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(web_host_module, "_X11_TOPMOST_VERIFY_DELAYS_MS", (0, 10, 20))

    class NativeWindow:
        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint

        def flags(self):
            return self._flags

        def setFlag(self, flag, enabled):  # noqa: N802
            self._flags = self._flags | flag if enabled else self._flags & ~flag

        def winId(self):  # noqa: N802
            return 0x120035

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self.native = NativeWindow()
            self._position = QPoint(10, 20)

        def windowHandle(self):  # noqa: N802
            return self.native

        def windowFlags(self):  # noqa: N802
            return Qt.WindowType.FramelessWindowHint

        def setWindowFlag(self, _flag, _enabled):  # noqa: N802
            raise AssertionError("QWidget 置顶路径会重建 QWebEngine 窗口")

        def isVisible(self):  # noqa: N802
            return True

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    host = WebPetHost(Renderer())
    monkeypatch.setattr(host, "_qt_uses_x11", lambda: True)
    monkeypatch.setattr(host, "_read_x11_topmost_state", lambda _view: False)

    result = host.set_always_on_top(True)
    loop = QEventLoop()
    QTimer.singleShot(100, loop.quit)
    loop.exec()

    assert result["status"] == "unavailable"
    assert result["enabled"] is False
    assert "_NET_WM_STATE_ABOVE" in str(result.get("detail", ""))
    assert host._read_qt_topmost_flag(host.view) is True
    assert host.is_always_on_top() is False

    host.shutdown()
    app.processEvents()


def test_web_host_qwebengine_topmost_waits_for_an_active_drag() -> None:
    """置顶重建不能插入拖动事务，释放后应自动继续提交。"""

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint
            self._position = QPoint(10, 20)
            self.set_flag_calls: list[bool] = []

        def isVisible(self):
            return True

        def windowFlags(self):
            return self._flags

        def setWindowFlag(self, flag, enabled):
            self.set_flag_calls.append(bool(enabled))
            self._flags = self._flags | flag if enabled else self._flags & ~flag

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    renderer = Renderer()
    host = WebPetHost(renderer)
    result = host.set_always_on_top(True)
    assert result["status"] == "requested"
    host._page_dragging = True
    app.processEvents()
    assert renderer.view.set_flag_calls == []

    host._page_dragging = False
    loop = QEventLoop()
    QTimer.singleShot(180, loop.quit)
    loop.exec()
    assert renderer.view.set_flag_calls == [True]
    host.shutdown()
    app.processEvents()


def test_web_host_qwebengine_topmost_drag_watchdog_cancels_stuck_request(monkeypatch) -> None:
    """丢失 release 时置顶等待应有界收敛且不清理用户拖动态。"""

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    import gui.qt6.web_host as web_host_module
    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(web_host_module, "_WINDOW_FLAG_DRAG_WAIT_TIMEOUT_SECONDS", 0.05)

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint
            self._position = QPoint(10, 20)
            self._visible = True

        def isVisible(self):
            return self._visible

        def show(self):
            self._visible = True

        def windowFlags(self):
            return self._flags

        def setWindowFlag(self, flag, enabled):
            self._flags = self._flags | flag if enabled else self._flags & ~flag

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    host = WebPetHost(Renderer())
    result = host.set_always_on_top(True)
    host._page_dragging = True
    loop = QEventLoop()
    QTimer.singleShot(180, loop.quit)
    loop.exec()

    assert result["status"] == "cancelled"
    assert "超时" in str(result.get("detail", ""))
    assert host._page_dragging is True
    assert host._window_flag_transition_pending is None
    assert host.always_on_top_status()["enabled"] is False

    host.shutdown()
    app.processEvents()


def test_web_host_page_drag_watchdog_clears_missing_release(monkeypatch) -> None:
    """页面桥丢失 release 时不能永久阻断自主行为和置顶。"""

    from PySide6.QtCore import QEventLoop, QPoint, QTimer
    from PySide6.QtWidgets import QApplication, QWidget

    import gui.qt6.web_host as web_host_module
    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(web_host_module, "_DRAG_TRANSACTION_TIMEOUT_MS", 30)
    view = QWidget()
    view.resize(180, 140)
    view.show()
    renderer = type("Renderer", (), {"view": view, "set_dragging": lambda self, _value: None})()
    host = WebPetHost(renderer)
    host._page_drag_origin = QPoint(10, 10)
    host._page_drag_global_origin = QPoint(10, 10)
    host._page_window_origin = QPoint(20, 30)
    host._page_dragging = True
    host._renderer_dragging = True
    host._arm_page_drag_watchdog()
    loop = QEventLoop()
    QTimer.singleShot(500, loop.quit)
    loop.exec()
    assert host._page_drag_origin is None
    assert host._page_dragging is False
    assert host._renderer_dragging is False
    assert host.is_user_interacting() is False
    host.shutdown()
    view.close()
    app.processEvents()


def test_web_native_filter_watchdog_clears_press_before_drag_threshold(monkeypatch) -> None:
    """原生过滤器在按下后丢失 release 也必须释放拖动原点。"""

    from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
    from PySide6.QtWidgets import QApplication, QWidget

    import gui.qt6.web_host as web_host_module
    from gui.qt6.web_host import _WebInteractionFilter

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(web_host_module, "_DRAG_TRANSACTION_TIMEOUT_MS", 30)

    class View(QWidget):
        def grabMouse(self):
            return None

        def releaseMouse(self):
            return None

    view = View()
    view.resize(180, 140)
    view.move(20, 30)
    view.show()
    filter_object = _WebInteractionFilter(view, None, None)

    class PressEvent:
        def type(self):
            return QEvent.Type.MouseButtonPress

        def button(self):
            return Qt.MouseButton.LeftButton

        def globalPosition(self):
            return QPointF(60, 70)

    assert filter_object.eventFilter(view, PressEvent()) is True
    assert filter_object._drag_origin == QPoint(60, 70)
    filter_object._drag_watchdog_expired()
    assert filter_object._drag_origin is None
    assert filter_object._dragging is False
    view.close()
    app.processEvents()


def test_web_native_filter_transparent_press_clears_child_focus() -> None:
    """原生过滤器命中透明区域时必须清除 Chromium 子表面焦点。"""

    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import _WebInteractionFilter

    app = QApplication.instance() or QApplication([])

    class View(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.clear_focus_calls = 0

        def clearFocus(self):  # noqa: N802
            self.clear_focus_calls += 1
            return super().clearFocus()

    class PressEvent:
        def type(self):
            return QEvent.Type.MouseButtonPress

        def button(self):
            return Qt.MouseButton.LeftButton

        def globalPosition(self):  # noqa: N802
            return QPointF(40, 50)

    view = View()
    view.resize(120, 90)
    view.show()
    filter_object = _WebInteractionFilter(
        view,
        None,
        None,
        hit_test_callback=lambda _point: False,
    )

    assert filter_object.eventFilter(view, PressEvent()) is False
    assert view.clear_focus_calls == 1
    assert filter_object._drag_origin is None
    view.close()
    app.processEvents()


def test_web_host_click_through_clears_stale_page_hover_state() -> None:
    """开启点击穿透后没有 leave 事件时，自主行为不能永久停在悬停态。"""

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class Renderer:
        def __init__(self) -> None:
            self.view = QWidget()

        def set_click_through(self, enabled: bool):
            return {"status": "available", "enabled": bool(enabled)}

    renderer = Renderer()
    host = WebPetHost(renderer)
    host._page_pointer_over = True
    assert host.is_user_interacting() is True
    result = host.set_click_through(True)
    assert result == {"status": "available", "enabled": True}
    assert host._page_pointer_over is False
    assert host.is_user_interacting() is False
    host.shutdown()
    renderer.view.close()
    app.processEvents()


def test_web_host_visibility_toggle_clears_stale_hover_state() -> None:
    """隐藏桌宠没有 leave 事件时，恢复显示不能遗留悬停锁。"""

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])
    view = QWidget()
    view.show()
    app.processEvents()
    renderer = type("Renderer", (), {"view": view})()
    host = WebPetHost(renderer)
    host._page_pointer_over = True
    result = host.toggle_visibility()
    assert result == {"status": "available", "visible": False}
    assert host._page_pointer_over is False
    assert host.is_user_interacting() is False
    host.toggle_visibility()
    host.shutdown()
    view.close()
    app.processEvents()


def test_web_host_surface_mask_skips_capture_while_hidden_after_first_show() -> None:
    """隐藏桌宠期间不应继续高频请求 WebEngine alpha 帧。"""

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])
    view = QWidget()
    view.resize(200, 120)
    view.show()
    app.processEvents()
    renderer = type("Renderer", (), {"view": view, "page_ready": True})()
    host = WebPetHost(renderer)
    host._surface_mask_enabled = True
    host._surface_mask_view_visible = True
    captures: list[object] = []
    host._request_surface_mask_capture = (  # type: ignore[method-assign]
        lambda _view: captures.append("capture") or True
    )
    host._set_x11_input_shapes = lambda *_args, **_kwargs: None  # type: ignore[method-assign]

    view.hide()
    assert host._refresh_surface_mask() is False
    assert captures == []
    view.show()
    assert host._refresh_surface_mask() is False
    assert captures == ["capture"]
    host.shutdown()
    view.close()
    app.processEvents()


def test_web_host_drag_and_context_menu_filter(tmp_path: Path) -> None:
    """Web Live2D 宿主路径也应支持拖动和右键控制入口。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget
from gui.qt6.web_host import WebPetHost

app = QApplication.instance() or QApplication([])
view = QWidget()
view.resize(220, 180)
view.move(100, 120)
child = QWidget(view)
child.setGeometry(0, 0, 220, 180)
origin = view.pos()
seen = []
opened = []
host = WebPetHost(type("Renderer", (), {"view": view, "shutdown": lambda self: None})())
assert host.install_interaction(
    lambda point: seen.append((point.x(), point.y())), lambda: opened.append(True)
)
view.show()
child.show()
app.processEvents()

def send(kind, x, y, button, buttons, target=child):
    event = QMouseEvent(
        kind,
        QPointF(x, y),
        QPointF(x, y),
        QPointF(x, y),
        button,
        buttons,
        Qt.NoModifier,
    )
    app.sendEvent(target, event)
    app.processEvents()
    assert event.isAccepted() or kind == QEvent.Type.MouseMove

send(QEvent.Type.MouseButtonPress, 12, 14, Qt.LeftButton, Qt.LeftButton)
send(QEvent.Type.MouseMove, 42, 49, Qt.NoButton, Qt.LeftButton)
assert view.pos() == origin + QPoint(30, 35)
send(QEvent.Type.MouseButtonRelease, 42, 49, Qt.LeftButton, Qt.NoButton)
send(QEvent.Type.MouseButtonPress, 42, 49, Qt.RightButton, Qt.RightButton)
assert seen
send(QEvent.Type.MouseButtonDblClick, 42, 49, Qt.LeftButton, Qt.LeftButton)
assert opened
# Chromium 会在页面加载后继续创建子 QWidget；递归过滤器必须覆盖新对象。
dynamic = QWidget(child)
dynamic.setGeometry(0, 0, 220, 180)
dynamic.show()
app.processEvents()
before_dynamic = view.pos()
send(QEvent.Type.MouseButtonPress, 12, 14, Qt.LeftButton, Qt.LeftButton, dynamic)
send(QEvent.Type.MouseMove, 52, 64, Qt.NoButton, Qt.LeftButton, dynamic)
assert view.pos() == before_dynamic + QPoint(40, 50)
send(QEvent.Type.MouseButtonRelease, 52, 64, Qt.LeftButton, Qt.NoButton, dynamic)
host.shutdown()
view.close()
app.processEvents()
print("web-host-interaction-ok")
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
    assert "web-host-interaction-ok" in result.stdout


def test_web_host_enter_opens_input_when_pointer_is_over_pet(tmp_path: Path) -> None:
    """鼠标停在桌宠区域时，Enter 应调用输入入口；离开后不应触发。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QWidget
from gui.qt6.web_host import WebPetHost

app = QApplication.instance() or QApplication([])
view = QWidget()
view.resize(220, 180)
child = QWidget(view)
child.setGeometry(0, 0, 220, 180)
opened = []
renderer = type("Renderer", (), {"view": view, "shutdown": lambda self: None})()
host = WebPetHost(renderer)
assert host.install_interaction(enter_callback=lambda: opened.append(True))
view.show()
child.show()
app.processEvents()

app.sendEvent(child, QEvent(QEvent.Type.Enter))
app.processEvents()
event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Return, Qt.NoModifier)
app.sendEvent(child, event)
app.processEvents()
assert opened == [True]
assert event.isAccepted()

for modifiers, repeat in [(Qt.ControlModifier, False), (Qt.NoModifier, True)]:
    app.sendEvent(child, QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Return, modifiers, "", repeat))
assert opened == [True]

app.sendEvent(child, QEvent(QEvent.Type.Leave))
app.sendEvent(view, QEvent(QEvent.Type.Leave))
app.sendEvent(child, QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Return, Qt.NoModifier))
assert opened == [True]
host.shutdown()
view.close()
app.processEvents()
print("web-host-enter-input-ok")
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
    assert "web-host-enter-input-ok" in result.stdout


def test_web_host_rejects_explicit_transparent_press_and_context(tmp_path: Path) -> None:
    """页面几何命中为 false 时不能抢焦点、拖动或弹出右键菜单。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication, QWidget
from gui.qt6.web_host import WebPetHost

app = QApplication.instance() or QApplication([])
view = QWidget()
view.resize(220, 180)
view.move(100, 120)
opened = []
renderer = type(
    "Renderer",
    (),
    {"view": view, "page_bridge_ready": True, "shutdown": lambda self: None},
)()
host = WebPetHost(renderer)
assert host.install_interaction(lambda _point: opened.append(True))
view.show()
app.processEvents()
origin = view.pos()
host._handle_page_interaction({"type": "press", "button": "left", "x": 5, "y": 5, "hit": False})
host._handle_page_interaction({"type": "move", "button": "left", "x": 90, "y": 90, "hit": False})
host._handle_page_interaction({"type": "release", "button": "left", "x": 90, "y": 90, "hit": False})
assert view.pos() == origin
host._handle_page_interaction({"type": "context", "button": "right", "x": 5, "y": 5, "hit": False})
assert opened == []
host._handle_page_interaction({"type": "press", "button": "left", "x": 5, "y": 5})
host._handle_page_interaction({"type": "move", "button": "left", "x": 90, "y": 90})
assert view.pos() == origin
host.shutdown()
view.close()
app.processEvents()
print("web-transparent-input-guard-ok")
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
    assert "web-transparent-input-guard-ok" in result.stdout


def test_web_host_forwards_model_reload_terminal_receipt() -> None:
    """宿主转发模型切换终态，同时保留自身几何失效处理。"""

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    QApplication.instance() or QApplication([])

    class Renderer:
        page_ready = True
        model_ready = True

        def __init__(self) -> None:
            self.view = QWidget()
            self.view.resize(120, 90)
            self.callback = None

        def reload_model(self, _path: object) -> dict[str, object]:
            return {"status": "pending", "request_id": 4}

        def set_model_reload_callback(self, callback: object) -> None:
            self.callback = callback

        def shutdown(self) -> None:
            return None

    renderer = Renderer()
    host = WebPetHost(renderer)
    receipts: list[dict[str, object]] = []
    host.set_model_reload_callback(lambda value: receipts.append(dict(value)))
    assert callable(renderer.callback)
    result = host.reload_model("live2d/model/second.model3.json")
    assert result == {"status": "pending", "request_id": 4}
    assert receipts == []
    renderer.callback({"status": "failed", "request_id": 4, "reason": "页面尚未就绪"})
    assert receipts == [{"status": "failed", "request_id": 4, "reason": "页面尚未就绪"}]
    before = host._geometry_generation
    renderer.callback({"status": "available", "request_id": 5})
    assert host._geometry_generation > before
    assert receipts[-1] == {"status": "available", "request_id": 5}
    host.shutdown()
    renderer.view.close()


def test_web_host_resource_reload_delegates_and_is_busy_during_drag() -> None:
    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    QApplication.instance() or QApplication([])

    class Renderer:
        page_ready = True
        model_ready = True

        def __init__(self) -> None:
            self.view = QWidget()
            self.calls: list[tuple[object, object, object]] = []

        def reload_resources(self, root, *, model_path=None, sprite_scale=None):
            self.calls.append((root, model_path, sprite_scale))
            return {"status": "pending", "request_id": 8}

        def shutdown(self) -> None:
            return None

    renderer = Renderer()
    host = WebPetHost(renderer)
    host._page_dragging = True
    assert host.reload_resources("/resources", model_path="model3.json")["status"] == "busy"
    assert renderer.calls == []

    host._page_dragging = False
    result = host.reload_resources("/resources", model_path="model3.json", sprite_scale=0.6)
    assert result == {"status": "pending", "request_id": 8}
    assert renderer.calls == [("/resources", "model3.json", 0.6)]
    host.shutdown()
    renderer.view.close()


def test_web_host_forwards_pending_reload_failure_during_shutdown() -> None:
    """关闭宿主时，未完成的模型切换不能让控制台永久停在 pending。"""

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    QApplication.instance() or QApplication([])

    class Renderer:
        page_ready = True
        model_ready = True

        def __init__(self) -> None:
            self.view = QWidget()
            self.callback = None

        def set_model_reload_callback(self, callback: object) -> None:
            self.callback = callback

        def reload_model(self, _path: object) -> dict[str, object]:
            return {"status": "pending", "request_id": 3}

        def shutdown(self) -> None:
            if callable(self.callback):
                self.callback(
                    {
                        "status": "failed",
                        "request_id": 3,
                        "reason": "renderer closed",
                    }
                )

    renderer = Renderer()
    host = WebPetHost(renderer)
    receipts: list[dict[str, object]] = []
    host.set_model_reload_callback(lambda value: receipts.append(dict(value)))
    assert host.reload_model("second.model3.json")["status"] == "pending"
    host.shutdown()
    assert receipts == [{"status": "failed", "request_id": 3, "reason": "renderer closed"}]
    renderer.view.close()


def test_web_host_blocks_autonomy_while_model_reload_is_pending() -> None:
    """候选模型加载期间，行为服务不能继续提交旧模型动作。"""

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    QApplication.instance() or QApplication([])

    class Renderer:
        page_ready = True
        model_ready = True

        def __init__(self) -> None:
            self.view = QWidget()
            self.model_reload_status = {"status": "pending", "request_id": 1}

    renderer = Renderer()
    host = WebPetHost(renderer)
    assert host.is_user_interacting() is True
    renderer.model_reload_status = {"status": "available"}
    assert host.is_user_interacting() is False
    renderer.view.close()


def test_web_native_wayland_system_move_release_clears_drag_state(monkeypatch) -> None:
    """Wayland 系统拖动不建立局部原点，release 仍必须清理全部状态。"""

    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6 import web_host

    app = QApplication.instance() or QApplication([])
    if web_host._WebInteractionFilter is None:
        return

    class Handle:
        def startSystemMove(self):  # noqa: N802
            return True

    class View(QWidget):
        def windowHandle(self):  # noqa: N802
            return Handle()

        def mapFromGlobal(self, point):  # noqa: N802
            return point

    class Event:
        def __init__(self, event_type):
            self._event_type = event_type

        def type(self):
            return self._event_type

        def button(self):
            return Qt.LeftButton

        def globalPosition(self):  # noqa: N802
            return QPointF(20, 30)

    view = View()
    view.resize(120, 90)
    view.show()
    app.processEvents()
    states: list[bool] = []
    interaction_filter = web_host._WebInteractionFilter(
        view,
        None,
        None,
        dragging_callback=states.append,
        hit_test_callback=lambda _point: True,
    )
    monkeypatch.setattr(
        web_host._WebInteractionFilter,
        "_is_wayland",
        staticmethod(lambda: True),
    )

    assert interaction_filter.eventFilter(view, Event(QEvent.Type.MouseButtonPress)) is True
    assert interaction_filter._system_move_active is True
    assert interaction_filter._drag_origin is None
    assert states == [True]

    assert interaction_filter.eventFilter(view, Event(QEvent.Type.MouseButtonRelease)) is True
    assert interaction_filter._system_move_active is False
    assert interaction_filter._drag_origin is None
    assert interaction_filter._dragging is False
    assert states == [True, False]

    view.close()
    app.processEvents()


def test_web_host_window_lock_keeps_cursor_tracking_without_user_input() -> None:
    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class Renderer:
        def __init__(self) -> None:
            self.view = QWidget()
            self.view.resize(200, 120)
            self.view.move(20, 30)
            self.locked: list[bool] = []
            self.targets: list[tuple[int, int, int, int]] = []

        def set_click_through(self, enabled: bool):
            return {"status": "available", "enabled": bool(enabled)}

        def set_interaction_locked(self, enabled: bool):
            self.locked.append(bool(enabled))

        def set_cursor_target(self, x: int, y: int, width: int, height: int):
            self.targets.append((x, y, width, height))

    renderer = Renderer()
    host = WebPetHost(renderer)
    renderer.view.show()
    app.processEvents()
    assert host.set_window_locked(True) == {"status": "available", "locked": True, "enabled": True}
    assert host.is_window_locked() is True
    assert host.set_window_locked(True) == {"status": "available", "locked": True, "enabled": True}
    assert host.is_user_interacting() is False
    host._handle_page_interaction(
        {"type": "press", "x": 20, "y": 20, "button": "left", "hit": True}
    )
    assert host._page_drag_origin is None
    assert host.update_cursor_tracking() is True
    assert renderer.targets
    assert host.set_window_locked(False) == {
        "status": "available",
        "locked": False,
        "enabled": False,
    }
    assert renderer.locked == [True, False]
    host._click_through = True
    assert host.set_window_locked(False) == {
        "status": "available",
        "locked": False,
        "enabled": True,
    }
    host.shutdown()
    renderer.view.close()
    app.processEvents()


def test_surface_mask_reports_input_disabled_while_window_locked() -> None:
    """锁定时实际清空 Input Shape，状态不能继续报告可点击。"""

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class Renderer:
        page_ready = True
        model_ready = True

        def __init__(self) -> None:
            self.view = QWidget()
            self.view.resize(200, 120)

        def set_interaction_locked(self, _enabled: bool) -> None:
            return None

    renderer = Renderer()
    host = WebPetHost(renderer)
    host._set_x11_input_shapes = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    host._surface_mask_enabled = True
    host._surface_mask_region = object()
    host._surface_mask_input_region = object()
    renderer.view.show()
    app.processEvents()

    host._window_locked = True
    host._refresh_surface_mask()
    state = host.surface_mask_status()
    assert state["ready"] is True
    assert state["input_ready"] is False
    assert state["status"] == "pending"

    host.shutdown()
    renderer.view.close()
    app.processEvents()


def test_web_host_surface_mask_reports_async_pending_state() -> None:
    """首帧 alpha 尚未回读时应返回 pending，而不是误报永久不可用。"""

    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    QApplication.instance() or QApplication([])

    class Page:
        def runJavaScript(self, _script, _callback=None):  # noqa: N802
            return None

    class View(QWidget):
        def __init__(self):
            super().__init__()
            self.resize(220, 180)

        def page(self):
            return Page()

        def findChildren(self, *_args):  # noqa: N802
            return ()

    renderer = SimpleNamespace(view=View(), page_ready=True)
    host = WebPetHost(renderer)
    host._set_x11_input_shapes = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    initial = host.set_surface_mask_enabled(True)
    assert initial["status"] == "pending"
    assert initial["requested"] is True
    assert host.surface_mask_status()["status"] == "pending"

    host._surface_mask_region = object()
    host._surface_mask_input_region = object()
    assert host.surface_mask_status() == {
        "status": "available",
        "enabled": True,
        "ready": True,
        "input_ready": True,
        "capture_pending": True,
    }
    host._surface_mask_capture_pending = False
    assert host.surface_mask_status() == {
        "status": "available",
        "enabled": True,
        "ready": True,
        "input_ready": True,
        "capture_pending": False,
    }
    host.shutdown()
    renderer.view.close()


def test_web_host_surface_mask_recovers_after_single_window_flag_switch() -> None:
    """单次窗口标志切换后，异步 Shape 必须收敛为可点击。"""

    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class Platform:
        backend = "x11"

        @staticmethod
        def set_click_through(_view: object, enabled: bool) -> dict[str, object]:
            return {"status": "available", "enabled": bool(enabled)}

    class Renderer:
        page_ready = True

        def __init__(self) -> None:
            self.view = QWidget()
            self.view.resize(200, 120)

    renderer = Renderer()
    renderer.view.show()
    host = WebPetHost(renderer, platform=Platform())
    host._surface_mask_enabled = True
    host._surface_mask_region = object()
    host._surface_mask_input_region = object()
    host._set_x11_input_shapes = lambda *_args, **_kwargs: None  # type: ignore[method-assign]
    host._refresh_surface_mask = lambda: True  # type: ignore[method-assign]

    result = host.set_click_through(False)
    assert result["status"] == "available"
    assert host.surface_mask_status()["status"] == "pending"

    loop = QEventLoop()
    QTimer.singleShot(500, loop.quit)
    loop.exec()
    state = host.surface_mask_status()
    assert state["status"] == "available"
    assert state["ready"] is True
    assert state["input_ready"] is True

    host.shutdown()
    renderer.view.close()
    app.processEvents()


def test_web_host_restores_input_shape_when_mask_disabled_during_click_through() -> None:
    """关闭视觉遮罩后解除穿透不能遗留空输入区域。"""

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])
    events: list[object] = []

    class Platform:
        @staticmethod
        def set_click_through(_view: object, enabled: bool) -> dict[str, object]:
            return {"status": "available", "enabled": bool(enabled)}

    class Renderer:
        page_ready = True

        def __init__(self) -> None:
            self.view = QWidget()
            self.view.resize(200, 120)

    renderer = Renderer()
    host = WebPetHost(renderer, platform=Platform())
    host._set_x11_input_shapes = (  # type: ignore[method-assign]
        lambda _view, region, **_kwargs: events.append(region)
    )
    renderer.view.show()
    app.processEvents()
    host._click_through = True
    host.set_surface_mask_enabled(False)
    assert events[-1] == ()
    result = host.set_click_through(False)
    assert result["enabled"] is False
    assert events[-1] is None
    host.shutdown()
    renderer.view.close()
    app.processEvents()


def test_web_host_shutdown_invalidates_large_surface_callbacks() -> None:
    """退出阶段的迟到大图回调不能继续占用 Qt 主线程。"""

    from types import SimpleNamespace

    from PySide6.QtWidgets import QApplication, QWidget

    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])
    view = QWidget()
    view.resize(200, 120)
    renderer = SimpleNamespace(view=view, page_ready=True)
    host = WebPetHost(renderer)
    host._surface_mask_enabled = True
    host._surface_mask_capture_generation = 1
    view.show()
    app.processEvents()
    payload = "data:image/png;base64," + ("A" * (16 * 1024 * 1024 + 1))
    host.prepare_shutdown()
    host._finish_surface_mask_capture(view, 1, payload)
    assert host._surface_mask_capture_failures == 0
    assert host._surface_mask_enabled is False
    host.shutdown()
    view.close()
    app.processEvents()


def test_web_host_async_window_lock_resolves_without_leaking_coroutine() -> None:
    """异步平台切换输入时，锁定状态必须等待结果再收敛。"""

    from gui.qt6.web_host import WebPetHost

    class Renderer:
        page_ready = False
        model_ready = False
        view = object()

        def set_interaction_locked(self, _enabled: bool) -> None:
            return None

    class Platform:
        def set_click_through(self, _view: object, enabled: bool):
            async def resolve():
                return {"status": "available", "enabled": bool(enabled)}

            return resolve()

    host = WebPetHost(Renderer(), platform=Platform())
    locked = asyncio.run(host.set_window_locked(True))
    assert locked == {"status": "available", "locked": True, "enabled": True}
    assert host.is_window_locked() is True
    unlocked = asyncio.run(host.set_window_locked(False))
    assert unlocked == {"status": "available", "locked": False, "enabled": False}
    assert host.is_window_locked() is False


def test_web_host_serializes_topmost_and_native_click_through_without_rebuilding_xid(
    monkeypatch,
    caplog,
) -> None:
    """置顶与穿透共享 QWindow；最新置顶请求收敛且不会留下 Shape 暂停。"""

    import logging

    from PySide6.QtCore import QEventLoop, QPoint, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    from gui.platforms.linux import set_window_click_through
    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])

    class NativeWindow:
        def __init__(self) -> None:
            self._flags = Qt.WindowType.FramelessWindowHint
            self._win_id = 0x240031
            self.calls: list[tuple[object, bool]] = []

        def flags(self):
            return self._flags

        def setFlag(self, flag, enabled):  # noqa: N802
            self.calls.append((flag, bool(enabled)))
            self._flags = self._flags | flag if enabled else self._flags & ~flag

        def winId(self):  # noqa: N802
            return self._win_id

    class FakeWebEngineView:
        __module__ = "PySide6.QtWebEngineWidgets"

        def __init__(self) -> None:
            self.native = NativeWindow()
            self._position = QPoint(30, 40)
            self.widget_calls: list[bool] = []

        def windowHandle(self):  # noqa: N802
            return self.native

        def windowFlags(self):  # noqa: N802
            return Qt.WindowType.FramelessWindowHint

        def setWindowFlag(self, _flag, enabled):  # noqa: N802
            self.widget_calls.append(bool(enabled))

        def isVisible(self):  # noqa: N802
            return True

        def pos(self):
            return self._position

        def move(self, x, y):
            self._position = QPoint(int(x), int(y))

    class Renderer:
        def __init__(self) -> None:
            self.view = FakeWebEngineView()

        def shutdown(self) -> None:
            return None

    class Platform:
        backend = "x11"

        @staticmethod
        def set_click_through(view, enabled):
            return set_window_click_through(view, enabled, backend="x11")

    renderer = Renderer()
    host = WebPetHost(renderer, platform=Platform())
    host._surface_mask_enabled = True
    monkeypatch.setattr(host, "_qt_uses_x11", lambda: True)
    monkeypatch.setattr(host, "_x11_topmost_window_id", lambda _view: 0x240031)
    monkeypatch.setattr(
        host,
        "_read_x11_topmost_state",
        lambda _view: bool(renderer.view.native.flags() & Qt.WindowType.WindowStaysOnTopHint),
    )
    caplog.set_level(logging.INFO, logger="gui.qt6.web_host")

    original_xid = renderer.view.native.winId()
    first = host.set_always_on_top(True)
    click_enabled = host.set_click_through(True)
    second = host.set_always_on_top(False)
    click_disabled = host.set_click_through(False)
    final = host.set_always_on_top(True)

    loop = QEventLoop()
    QTimer.singleShot(160, loop.quit)
    loop.exec()

    assert first["status"] == "cancelled"
    assert second["status"] == "cancelled"
    assert click_enabled["status"] == "available"
    assert click_disabled["status"] == "available"
    assert final == {"status": "available", "enabled": True}
    assert host.always_on_top_status()["enabled"] is True
    assert host._input_transparency_active() is False
    assert renderer.view.native.winId() == original_xid
    assert renderer.view.widget_calls == []
    assert host._window_flag_transition_pending is None
    assert host._window_flag_transition_apply is None
    assert host._window_flag_transition_cleanup is None
    assert host._surface_mask_rebuild_pending is False
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "gui.window_flags.transaction.start" in messages
    assert "gui.window_flags.transaction.merged" in messages
    assert "gui.window_flags.transaction.terminal" in messages
    assert "0x240031" not in messages

    host.shutdown()
    app.processEvents()


def test_web_host_async_click_through_timeout_releases_transaction(monkeypatch, caplog) -> None:
    """平台 awaitable 不返回时，输入事务必须超时并释放 pending 状态。"""

    import logging

    import gui.qt6.web_host as web_host_module
    from gui.qt6.web_host import WebPetHost

    class Renderer:
        page_ready = False
        model_ready = False
        view = object()

    class Platform:
        @staticmethod
        def set_click_through(_view: object, _enabled: bool):
            async def never_finishes():
                await asyncio.sleep(60)

            return never_finishes()

    monkeypatch.setattr(web_host_module, "_WINDOW_FLAG_DISPATCH_TIMEOUT_SECONDS", 0.01)
    caplog.set_level(logging.INFO, logger="gui.qt6.web_host")
    host = WebPetHost(Renderer(), platform=Platform())

    result = asyncio.run(host.set_click_through(True))

    assert result["status"] == "unavailable"
    assert result["enabled"] is False
    assert host._click_through_transition_pending is None
    assert host._surface_mask_rebuild_pending is False
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "gui.window_flags.transaction.timeout" in messages
    assert '"reason_code": "dispatch_timeout"' in messages


def test_web_host_latest_async_window_lock_request_wins() -> None:
    """连续锁定/解锁时旧异步结果不能把最新窗口状态写回。"""

    from gui.qt6.web_host import WebPetHost

    class Renderer:
        page_ready = False
        model_ready = False
        view = object()

        def set_interaction_locked(self, _enabled: bool) -> None:
            return None

    class Platform:
        def __init__(self) -> None:
            self.input_transparent = False

        def set_click_through(self, _view: object, enabled: bool):
            async def resolve():
                await asyncio.sleep(0.03 if enabled else 0)
                self.input_transparent = bool(enabled)
                return {"status": "available", "enabled": bool(enabled)}

            return resolve()

    async def exercise(host: WebPetHost):
        lock = host.set_window_locked(True)
        unlock = host.set_window_locked(False)
        return await asyncio.gather(lock, unlock)

    platform = Platform()
    host = WebPetHost(Renderer(), platform=platform)
    host._input_transparency_active = lambda: platform.input_transparent  # type: ignore[method-assign]
    locked_result, unlocked_result = asyncio.run(exercise(host))

    assert locked_result["status"] == "cancelled"
    assert unlocked_result == {
        "status": "available",
        "locked": False,
        "enabled": False,
    }
    assert host.is_window_locked() is False
    assert host._window_lock_transition_pending is None
    assert host._click_through_transition_pending is None
    assert platform.input_transparent is False


def test_web_host_reapplies_latest_native_input_after_stale_async_dispatch() -> None:
    """旧平台协程迟到时，原生输入旗标和置顶恢复都必须服从最新意图。"""

    from gui.qt6.web_host import WebPetHost

    class Renderer:
        page_ready = False
        model_ready = False
        view = object()

    class Platform:
        backend = "x11"

        def __init__(self) -> None:
            self.input_transparent = False

        def set_click_through(self, _view: object, enabled: bool):
            async def resolve():
                await asyncio.sleep(0.03 if enabled else 0.001)
                self.input_transparent = bool(enabled)
                return {"status": "available", "enabled": bool(enabled)}

            return resolve()

    async def exercise(host: WebPetHost):
        stale = host.set_click_through(True)
        await asyncio.sleep(0)
        host._topmost_confirmed = False
        latest = host.set_click_through(False)
        return await asyncio.gather(stale, latest)

    platform = Platform()
    host = WebPetHost(Renderer(), platform=platform)
    host._topmost_confirmed = True
    host._input_transparency_active = lambda: platform.input_transparent  # type: ignore[method-assign]
    topmost_requests: list[bool] = []
    host.set_always_on_top = (  # type: ignore[method-assign]
        lambda enabled: (
            topmost_requests.append(bool(enabled))
            or {"status": "available", "enabled": bool(enabled)}
        )
    )

    stale_result, latest_result = asyncio.run(exercise(host))

    assert stale_result["status"] == "cancelled"
    assert latest_result == {"status": "available", "enabled": False, "detail": ""}
    assert platform.input_transparent is False
    assert host._click_through is False
    assert host._click_through_transition_pending is None
    assert True not in topmost_requests
