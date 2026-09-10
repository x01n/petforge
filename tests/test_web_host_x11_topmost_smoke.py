"""真实 X11 窗口管理器下的 WebEngine 置顶终态冒烟测试。"""

from __future__ import annotations

import os
from time import monotonic

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("MEAPET_X11_TOPMOST_SMOKE") != "1",
    reason="仅在隔离 X11/窗口管理器冒烟环境中运行",
)


def test_webengine_topmost_keeps_xid_and_confirms_ewmh() -> None:
    """QWindow 切换不得更换 XID，且开关都必须由 EWMH 回读确认。"""

    from PySide6.QtCore import QEventLoop, Qt, QTimer
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QApplication

    from gui.platforms.linux import LinuxDesktopPlatform
    from gui.qt6.web_host import WebPetHost

    app = QApplication.instance() or QApplication([])
    view = QWebEngineView()
    view.setWindowFlags(
        Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
    )
    view.resize(320, 240)
    view.show()

    class Renderer:
        def __init__(self) -> None:
            self.view = view

        def shutdown(self) -> None:
            return None

    host = WebPetHost(Renderer(), platform=LinuxDesktopPlatform())

    def process_for(milliseconds: int) -> None:
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()

    def wait_ewmh(expected: bool, timeout: float = 2.0) -> bool | None:
        deadline = monotonic() + timeout
        actual = host._read_x11_topmost_state(view)
        while actual is not expected and monotonic() < deadline:
            process_for(20)
            actual = host._read_x11_topmost_state(view)
        return actual

    def wait_result(result: dict[str, object], timeout: float = 2.5) -> None:
        deadline = monotonic() + timeout
        while result.get("status") == "requested" and monotonic() < deadline:
            process_for(20)

    def read_state_names() -> tuple[str, ...]:
        from Xlib import X, display

        connection = display.Display(os.environ["DISPLAY"])
        try:
            window = connection.create_resource_object("window", int(view.windowHandle().winId()))
            state_atom = connection.intern_atom("_NET_WM_STATE", only_if_exists=False)
            value = window.get_full_property(state_atom, X.AnyPropertyType)
            return tuple(
                str(connection.get_atom_name(int(item)))
                for item in tuple(getattr(value, "value", ()) or ())
            )
        finally:
            connection.close()

    process_for(200)
    original_id = int(view.windowHandle().winId())
    assert wait_ewmh(True) is True

    disabled = host.set_always_on_top(False)
    assert wait_ewmh(False) is False
    wait_result(disabled)
    disabled_id = int(view.windowHandle().winId())
    disabled_ewmh = host._read_x11_topmost_state(view)
    disabled_names = read_state_names()

    enabled = host.set_always_on_top(True)
    assert wait_ewmh(True) is True
    wait_result(enabled)
    enabled_id = int(view.windowHandle().winId())
    enabled_ewmh = host._read_x11_topmost_state(view)
    enabled_names = read_state_names()

    click_enabled = host.set_click_through(True)
    process_for(120)
    click_enabled_id = int(view.windowHandle().winId())
    click_enabled_ewmh = host._read_x11_topmost_state(view)
    click_disabled = host.set_click_through(False)
    process_for(120)
    click_disabled_id = int(view.windowHandle().winId())
    click_disabled_ewmh = host._read_x11_topmost_state(view)

    superseded_disable = host.set_always_on_top(False)
    rapid_click_enabled = host.set_click_through(True)
    superseded_enable = host.set_always_on_top(True)
    rapid_click_disabled = host.set_click_through(False)
    final_enable = host.set_always_on_top(True)
    wait_result(final_enable)
    process_for(120)
    final_id = int(view.windowHandle().winId())
    final_ewmh = host._read_x11_topmost_state(view)

    evidence = {
        "original_xid": hex(original_id),
        "disabled_xid": hex(disabled_id),
        "enabled_xid": hex(enabled_id),
        "disabled_ewmh": disabled_ewmh,
        "disabled_names": disabled_names,
        "enabled_ewmh": enabled_ewmh,
        "enabled_names": enabled_names,
        "disabled_result": dict(disabled),
        "enabled_result": dict(enabled),
        "click_enabled_xid": hex(click_enabled_id),
        "click_disabled_xid": hex(click_disabled_id),
        "click_enabled_ewmh": click_enabled_ewmh,
        "click_disabled_ewmh": click_disabled_ewmh,
        "click_enabled_result": dict(click_enabled),
        "click_disabled_result": dict(click_disabled),
        "rapid_click_enabled_result": dict(rapid_click_enabled),
        "rapid_click_disabled_result": dict(rapid_click_disabled),
        "superseded_disable_result": dict(superseded_disable),
        "superseded_enable_result": dict(superseded_enable),
        "final_enable_result": dict(final_enable),
        "final_xid": hex(final_id),
        "final_ewmh": final_ewmh,
        "shape_rebuild_pending": host._surface_mask_rebuild_pending,
        "topmost_pending": host._window_flag_transition_pending,
    }
    print(evidence)

    assert disabled_id == original_id
    assert enabled_id == original_id
    assert disabled_ewmh is False
    assert enabled_ewmh is True
    assert click_enabled_id == original_id
    assert click_disabled_id == original_id
    assert click_enabled_ewmh is True
    assert click_disabled_ewmh is True
    assert click_enabled["status"] == "available"
    assert click_disabled["status"] == "available"
    assert superseded_disable["status"] == "cancelled"
    assert superseded_enable["status"] == "cancelled"
    assert rapid_click_enabled["status"] == "available"
    assert rapid_click_disabled["status"] == "available"
    assert final_id == original_id
    assert final_ewmh is True
    assert final_enable["status"] == "available"
    assert final_enable["enabled"] is True
    assert "确认置顶" in str(final_enable.get("detail", ""))
    assert host._surface_mask_rebuild_pending is False
    assert host._window_flag_transition_pending is None
    assert disabled["status"] == "available"
    assert disabled["enabled"] is False
    assert enabled["status"] == "available"
    assert enabled["enabled"] is True

    host.shutdown()
    view.close()
    app.processEvents()
