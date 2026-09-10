from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PySide6.QtCore import QPoint

from gui.qt6.web_host import WebPetHost
from gui.renderers.sprite import SpriteRenderer
from gui.renderers.web_live2d import WebLive2DRenderer


def test_web_renderer_dragging_freezes_external_clock(monkeypatch) -> None:
    """拖动期间不应继续向页面提交外部动画时间步。"""

    renderer = WebLive2DRenderer("resources")
    renderer._view = object()  # type: ignore[assignment]
    renderer._page_ready = True
    renderer._model_ready = True
    scripts: list[str] = []
    monkeypatch.setattr(renderer, "_run_javascript", scripts.append)

    renderer.set_dragging(True)
    renderer.advance(0.03)

    assert renderer.state.elapsed == 0.0
    assert any("setDragging(true)" in script for script in scripts)
    assert not any(".advance(" in script for script in scripts)

    renderer.set_dragging(False)
    renderer.advance(0.03)

    assert renderer.state.elapsed == 0.03
    assert any("setDragging(false)" in script for script in scripts)
    assert any(".advance(" in script for script in scripts)


def test_sprite_renderer_dragging_preserves_current_animation_frame() -> None:
    """精灵回退在拖动期间保持当前帧，释放后再继续动作。"""

    resource_root = Path(__file__).resolve().parents[1] / "resources" / "sprites"
    renderer = SpriteRenderer(resource_root)
    assert renderer.play_motion("walk")
    renderer.set_dragging(True)
    renderer.advance(0.2)

    assert renderer.state.elapsed == 0.0

    renderer.set_dragging(False)
    renderer.advance(0.2)

    assert renderer.state.elapsed == 0.2


def test_web_host_dragging_bridge_is_idempotent() -> None:
    """宿主只在状态边沿通知渲染器冻结或恢复。"""

    states: list[bool] = []
    renderer = SimpleNamespace(
        view=object(),
        set_dragging=lambda value: states.append(bool(value)),
    )
    host = WebPetHost(renderer)

    host._set_renderer_dragging(True)
    host._set_renderer_dragging(True)
    host._set_renderer_dragging(False)
    host._set_renderer_dragging(False)

    assert states == [True, False]


def test_web_host_forces_page_release_when_threshold_was_not_crossed() -> None:
    """短点击也必须解除页面先行冻结的本地动作时钟。"""

    class View:
        def pos(self):
            return QPoint(100, 120)

    renderer = SimpleNamespace(
        view=View(),
        page_bridge_ready=True,
        set_dragging=lambda value: calls.append(bool(value)),
    )
    calls: list[bool] = []
    host = WebPetHost(renderer)

    host._handle_page_interaction(
        {"type": "press", "button": "left", "x": 20, "y": 30, "hit": True}
    )
    assert host._page_drag_origin == QPoint(20, 30)
    assert calls == []

    host._handle_page_interaction(
        {"type": "release", "button": "left", "x": 20, "y": 30, "hit": True}
    )

    assert calls == [False]
    assert host._page_drag_origin is None
    assert host._page_dragging is False


def test_web_renderer_html_defers_resize_until_drag_release() -> None:
    """ResizeObserver/窗口 resize 在拖动期间只能记录最后尺寸。"""

    html = WebLive2DRenderer("resources")._html_document()

    assert "const resizeState =" in html
    assert "function flushDeferredResize()" in html
    assert "performance.now() < Number(clockState.dragReleaseUntil || 0)" in html
    assert "resizeState.pendingWidth = width" in html
    assert "resizeState.appliedCount += 1" in html
    assert "function scheduleDeferredResizeFlush" in html
    assert "resizeState.flushRetries += 1" in html
    assert "localPointerActive" in html
    assert "hostDragging" in html
    assert "!clockState.hostDragging" in html
    assert "clockState.dragging = false;" in html
    assert "if (resizeState.pending) scheduleDeferredResizeFlush(32);" in html
    assert "flushDeferredResize();" in html
    assert "new ResizeObserver" in html


def test_web_renderer_release_guard_clears_host_dragging_for_throttled_pages() -> None:
    """宿主释放回调被页面稳定窗拦截时，host 标记仍须可收敛。"""

    html = WebLive2DRenderer("resources")._html_document()
    guard_start = html.index(
        "if (!requested && performance.now() < Number(clockState.dragReleaseUntil || 0))"
    )
    normal_update = html.index("clockState.hostDragging = requested;", guard_start)
    release_guard = html[guard_start:normal_update]

    assert "clockState.hostDragging = false;" in release_guard
    assert release_guard.index("clockState.hostDragging = false;") < release_guard.index(
        "return true;"
    )


def test_web_renderer_drag_release_has_rAF_fallback_and_reasserts_next_press() -> None:
    """页面定时器被节流时由 rAF 收敛，下一笔按下重新声明拖动态。"""

    html = WebLive2DRenderer("resources")._html_document()
    assert "let reassertHostDragging = false;" in html
    assert "reassertHostDragging = true;" in html
    assert "if (reassertHostDragging)" in html
    local_drag_start = html.index("function setLocalDragging(value)")
    local_drag_end = html.index("function releaseLocalDragging()", local_drag_start)
    local_drag = html[local_drag_start:local_drag_end]
    assert "api.setDragging(true);" in local_drag
    # 兜底必须位于 animationTick 的 dragging 分支，而不是 resize pending
    # 分支；没有尺寸变化时也必须恢复动作时钟。
    tick_start = html.index("function animationTick(timestamp)")
    tick_end = html.index("const dt = lastFrameTime", tick_start)
    dragging_tick = html[tick_start:tick_end]
    assert "!clockState.localPointerActive && !clockState.hostDragging" in dragging_tick
    assert "clockState.dragging = false;" in dragging_tick
    assert "clockState.dragReleaseUntil = 0;" in dragging_tick


def test_web_host_queues_latest_resize_while_dragging(monkeypatch) -> None:
    """Qt 宿主在拖动事务中不应向页面重复提交中间尺寸。"""

    class Page:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def runJavaScript(self, script, _callback=None):  # noqa: N802
            self.calls.append(str(script))

    class View:
        def __init__(self) -> None:
            self._page = Page()

        def page(self):
            return self._page

        def width(self):
            return 420

        def height(self):
            return 480

        def pos(self):
            return SimpleNamespace(x=lambda: 10, y=lambda: 20)

    view = View()
    states: list[bool] = []
    renderer = SimpleNamespace(
        view=view,
        page_ready=True,
        set_dragging=lambda value: states.append(bool(value)),
    )
    host = WebPetHost(renderer)
    monkeypatch.setattr("gui.qt6.web_host.QTimer.singleShot", lambda *_args: None)

    host._renderer_dragging = True
    host._request_page_resize(320, 240)
    host._request_page_resize(360, 260)
    assert view._page.calls == []
    assert host._deferred_page_resize == (360, 260)

    host._set_renderer_dragging(False)
    assert len(view._page.calls) == 1
    assert "window.meapetLive2D.resize(360,260)" in view._page.calls[0]
    assert host._deferred_page_resize is None
    assert states == [False]
