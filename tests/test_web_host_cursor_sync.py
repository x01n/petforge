from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from PySide6.QtCore import QPoint
from PySide6.QtGui import QCursor

from gui.qt6.web_host import WebPetHost


def test_webengine_focus_proxy_is_forced_to_no_focus() -> None:
    """展示视图及其 focusProxy 都不能触发 Chromium requestActivate。"""

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QLineEdit, QWidget

    from gui.qt6.web_host import _set_no_focus

    app = QApplication.instance() or QApplication([])
    view = QWidget()
    proxy = QLineEdit(view)
    view.setFocusProxy(proxy)
    _set_no_focus(view)
    no_focus = getattr(getattr(Qt, "FocusPolicy", Qt), "NoFocus", None)
    if no_focus is None:
        no_focus = getattr(Qt, "NoFocus", None)
    assert no_focus is not None
    assert view.focusPolicy() == no_focus
    assert proxy.focusPolicy() == no_focus
    view.close()
    app.processEvents()


def test_global_cursor_mapping_prefers_qt_local_coordinates() -> None:
    """窗口有边框或被重建后，局部坐标应以 Qt 映射结果为准。"""

    class View:
        def mapFromGlobal(self, _point):  # noqa: N802
            return SimpleNamespace(x=lambda: 7, y=lambda: 11)

    assert WebPetHost._map_global_cursor(View(), object(), (100, 200)) == (7, 11)


def test_global_cursor_mapping_falls_back_to_position() -> None:
    point = SimpleNamespace(x=lambda: 107, y=lambda: 211)
    assert WebPetHost._map_global_cursor(object(), point, (100, 200)) == (7, 11)


def test_move_invalidates_cached_cursor_target(monkeypatch) -> None:
    class View:
        def pos(self):
            return SimpleNamespace(x=lambda: 20, y=lambda: 30)

    class Platform:
        def move_overlay(self, _view, _x, _y):
            return {"status": "available"}

    host = WebPetHost(SimpleNamespace(view=View()), platform=Platform())
    move_mask_events: list[str] = []
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: move_mask_events.append("begin"))
    monkeypatch.setattr(
        host,
        "_schedule_surface_mask_after_move",
        lambda: move_mask_events.append("finish"),
    )
    host._last_cursor_target = (10, 10, 100, 100)
    assert host.move_to(50, 60)["status"] == "available"
    assert host._last_cursor_target is None
    assert move_mask_events == ["begin", "finish"]


def test_autonomous_step_does_not_reset_tracking_or_rebuild_shape(monkeypatch) -> None:
    """自主漫游的小步移动不应反复清空注视速度或暂停 Shape。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(20, 30)

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))

        def pos(self):
            return self.origin

        def width(self):
            return 200

        def height(self):
            return 160

        def mapFromGlobal(self, point):  # noqa: N802
            return QPoint(point.x() - self.origin.x(), point.y() - self.origin.y())

    notifications: list[bool] = []
    events: list[str] = []

    class Platform:
        def move_overlay(self, view, x, y):
            view.move(x, y)
            return {"status": "available"}

    renderer = SimpleNamespace(
        view=View(),
        notify_window_moved=lambda: notifications.append(True),
        set_cursor_target=lambda *_values: True,
    )
    host = WebPetHost(renderer, platform=Platform())
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: events.append("begin"))
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: events.append("finish"))

    result = host.move_to(48, 60, duration_ms=0)

    assert result["status"] == "available"
    assert renderer.view.pos() == QPoint(48, 60)
    assert notifications == []
    assert events == []

    # 位置提交后下一次光标轮询仍可能看到新窗口坐标；该变化属于自主
    # 几何事务，不应被误报为外部移动并重置 Live2D 注视速度。
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(60, 70)))
    assert host.update_cursor_tracking() is True
    assert notifications == []


def test_autonomous_step_holds_surface_mask_timer_until_geometry_settles(monkeypatch) -> None:
    """自主小步期间 Shape 定时器不能抓图或提交新区域。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(20, 30)

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))

        def pos(self):
            return self.origin

        def width(self):
            return 220

        def height(self):
            return 180

    class Platform:
        def move_overlay(self, view, x, y):
            view.move(x, y)
            return {"status": "available"}

    view = View()
    host = WebPetHost(SimpleNamespace(view=view, page_ready=True), platform=Platform())
    host._surface_mask_enabled = True
    host._surface_mask_region = object()
    requests: list[bool] = []
    monkeypatch.setattr(host, "_request_surface_mask_capture", lambda _view: requests.append(True))

    result = host.move_to(48, 60, duration_ms=0)
    assert result["status"] == "available"
    assert host._refresh_surface_mask() is True
    assert requests == []


def test_release_establishes_new_cursor_tracking_baseline(monkeypatch) -> None:
    """拖动释放后的首个追踪 tick 不应重复清空模型速度。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(20, 30)

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))

        def pos(self):
            return self.origin

        def width(self):
            return 200

        def height(self):
            return 160

        def mapFromGlobal(self, point):  # noqa: N802
            return QPoint(point.x() - self.origin.x(), point.y() - self.origin.y())

    notifications: list[bool] = []
    renderer = SimpleNamespace(
        view=View(),
        set_dragging=lambda _value: None,
        notify_window_moved=lambda: notifications.append(True),
        set_cursor_target=lambda *_values: None,
    )
    host = WebPetHost(renderer)
    host._last_window_position = (20, 30)
    host._renderer_dragging = True
    renderer.view.move(80, 90)
    host._set_renderer_dragging(False)
    assert host.is_user_interacting() is True

    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(120, 130)))
    assert host.update_cursor_tracking() is True
    assert notifications == []


def test_cursor_tracking_is_suspended_after_press_before_drag_threshold(monkeypatch) -> None:
    """按下已建立交互事务后，越过阈值前也不能写入注视目标。"""

    class View:
        def pos(self):
            return QPoint(20, 30)

        def width(self):
            return 200

        def height(self):
            return 160

        def mapFromGlobal(self, point):  # noqa: N802
            return QPoint(point.x() - 20, point.y() - 30)

    targets: list[tuple[object, ...]] = []
    renderer = SimpleNamespace(
        view=View(),
        set_cursor_target=lambda *values: targets.append(values),
    )
    host = WebPetHost(renderer)
    host._page_drag_origin = QPoint(100, 80)
    host._page_window_origin = QPoint(20, 30)
    host._page_dragging = False
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(150, 120)))

    assert host.update_cursor_tracking() is True
    assert targets == []


def test_locking_window_clears_an_inflight_drag_before_input_rebuild(monkeypatch) -> None:
    """锁定快捷键抢在 release 前触发时，不应留下旧拖动原点。"""

    renderer = SimpleNamespace(view=object(), set_dragging=lambda _value: None)
    host = WebPetHost(renderer)
    host._page_drag_origin = QPoint(1, 2)
    host._page_window_origin = QPoint(10, 20)
    host._page_dragging = True
    host._renderer_dragging = True
    monkeypatch.setattr(
        host,
        "set_click_through",
        lambda _enabled: {"status": "available", "enabled": True},
    )

    result = host.set_window_locked(True)

    assert result["locked"] is True
    assert host._page_drag_origin is None
    assert host._page_window_origin is None
    assert host._page_dragging is False
    assert host._renderer_dragging is False


def test_cursor_tracking_clamps_to_last_valid_pixel(monkeypatch) -> None:
    class View:
        def pos(self):
            return QPoint(100, 200)

        def width(self):
            return 100

        def height(self):
            return 80

        def mapFromGlobal(self, _point):  # noqa: N802
            return QPoint(100, 80)

    targets: list[tuple[int, int, int, int]] = []
    renderer = SimpleNamespace(
        view=View(),
        set_cursor_target=lambda *values: targets.append(values),
    )
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(200, 280)))

    assert WebPetHost(renderer).update_cursor_tracking() is True
    assert targets == [(99, 79, 100, 80)]


def test_web_cursor_tracking_retries_after_renderer_rejects_target(monkeypatch) -> None:
    """渲染器暂时拒绝坐标时，下一次相同坐标仍应重试。"""

    class View:
        def pos(self):
            return QPoint(10, 20)

        def width(self):
            return 100

        def height(self):
            return 80

        def mapFromGlobal(self, point):  # noqa: N802
            return QPoint(point.x() - 10, point.y() - 20)

    calls: list[tuple[object, ...]] = []

    def setter(*values):
        calls.append(values)
        return False if len(calls) == 1 else None

    renderer = SimpleNamespace(view=View(), set_cursor_target=setter)
    host = WebPetHost(renderer)
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(30, 40)))

    assert host.update_cursor_tracking() is False
    assert host._last_cursor_target is None
    assert host.update_cursor_tracking() is True
    assert len(calls) == 2
    assert host._last_cursor_target == (20, 20, 100, 80)


def test_page_cursor_tracking_retries_after_renderer_rejects_target() -> None:
    """页面桥坐标被渲染器拒绝后不能被失败缓存吞掉。"""

    class View:
        def pos(self):
            return QPoint(10, 20)

        def width(self):
            return 100

        def height(self):
            return 80

    calls: list[tuple[object, ...]] = []

    def setter(*values):
        calls.append(values)
        return False if len(calls) == 1 else None

    renderer = SimpleNamespace(view=View(), set_cursor_target=setter)
    host = WebPetHost(renderer)
    payload = {"type": "hover", "screen_x": 30, "screen_y": 40, "hit": True}

    assert host._update_cursor_tracking_from_page(payload) is False
    assert host._last_cursor_target is None
    assert host._update_cursor_tracking_from_page(payload) is True
    assert len(calls) == 2
    assert host._last_cursor_target == (20, 20, 100, 80)


def test_cursor_tracking_notifies_renderer_after_window_move(monkeypatch) -> None:
    """顶层窗口位移时清空旧速度，避免模型向旧目标过冲。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(100, 200)

        def pos(self):
            return self.origin

        def width(self):
            return 100

        def height(self):
            return 80

        def mapFromGlobal(self, point):  # noqa: N802
            return QPoint(point.x() - self.origin.x(), point.y() - self.origin.y())

    view = View()
    targets: list[tuple[int, int, int, int]] = []
    notifications: list[bool] = []
    renderer = SimpleNamespace(
        view=view,
        set_cursor_target=lambda *values: targets.append(values),
        notify_window_moved=lambda: notifications.append(True),
    )
    host = WebPetHost(renderer)
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(150, 240)))
    assert host.update_cursor_tracking() is True
    view.origin = QPoint(110, 205)
    assert host.update_cursor_tracking() is True
    assert notifications == [True]
    assert len(targets) == 2


def test_page_drag_notifies_renderer_before_cursor_resync(monkeypatch) -> None:
    """页面拖动路径也必须清空旧追踪速度。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(10, 20)

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))

        def pos(self):
            return self.origin

    notifications: list[bool] = []
    renderer = SimpleNamespace(
        view=View(),
        notify_window_moved=lambda: notifications.append(True),
    )
    host = WebPetHost(renderer)
    move_mask_events: list[str] = []
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: move_mask_events.append("begin"))
    monkeypatch.setattr(
        host,
        "_schedule_surface_mask_after_move",
        lambda: move_mask_events.append("finish"),
    )
    host._page_drag_origin = QPoint(2, 3)
    host._page_window_origin = QPoint(10, 20)
    host._move_page_drag_target(QPoint(30, 40))
    assert renderer.view.pos() == QPoint(30, 40)
    assert notifications == [True]
    # 高频小步拖动不应每帧清零追踪速度；跨过明显距离时仍会复位。
    host._move_page_drag_target(QPoint(32, 42))
    assert notifications == [True]
    host._move_page_drag_target(QPoint(40, 50))
    host._flush_page_drag_target(force=True)
    assert notifications == [True, True]
    assert move_mask_events == ["begin", "finish", "begin", "begin", "finish"]


def test_page_drag_does_not_submit_duplicate_window_moves(monkeypatch) -> None:
    """平台适配器已移动窗口时，页面拖动不应再次调用 QWidget.move。"""

    class View:
        def __init__(self) -> None:
            self._position = QPoint(100, 120)
            self.move_calls = 0

        def pos(self):
            return self._position

        def move(self, x, y):
            self.move_calls += 1
            self._position = QPoint(int(x), int(y))

    class Platform:
        def move_overlay(self, view, x, y):
            view.move(x, y)
            return {"status": "available"}

    view = View()
    host = WebPetHost(SimpleNamespace(view=view), platform=Platform())
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: None)
    host._move_page_drag_target(QPoint(140, 160))
    assert view.pos() == QPoint(140, 160)
    assert view.move_calls == 1


def test_cursor_tracking_is_paused_while_page_dragging(monkeypatch) -> None:
    """拖动期间不应由定时器重置注视平滑速度。"""

    class View:
        def pos(self):
            return QPoint(100, 120)

        def width(self):
            return 100

        def height(self):
            return 80

    notifications: list[bool] = []
    renderer = SimpleNamespace(
        view=View(),
        set_cursor_target=lambda *_values: None,
        notify_window_moved=lambda: notifications.append(True),
    )
    host = WebPetHost(renderer)
    host._page_dragging = True
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(150, 160)))
    assert host.update_cursor_tracking() is True
    assert notifications == []


def test_cursor_tracking_is_paused_when_renderer_dragging_flag_is_set(monkeypatch) -> None:
    """渲染器已进入拖动态时，即使宿主原点刚清理也不能更新注视。"""

    class View:
        def pos(self):
            return QPoint(100, 120)

        def width(self):
            return 100

        def height(self):
            return 80

    targets: list[tuple[object, ...]] = []
    renderer = SimpleNamespace(
        view=View(),
        set_cursor_target=lambda *values: targets.append(values),
    )
    host = WebPetHost(renderer)
    host._renderer_dragging = True
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(150, 160)))
    assert host.update_cursor_tracking() is True
    assert targets == []


def test_page_drag_coalesces_burst_and_flushes_latest_target(monkeypatch) -> None:
    """页面桥突发 move 只提交首个和最新坐标。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(10, 20)
            self.moves: list[tuple[int, int]] = []

        def move(self, x, y):
            self.moves.append((int(x), int(y)))
            self.origin = QPoint(int(x), int(y))

        def pos(self):
            return self.origin

    view = View()
    host = WebPetHost(SimpleNamespace(view=view))
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_page_drag_flush", lambda: None)

    host._move_page_drag_target(QPoint(100, 100))
    host._move_page_drag_target(QPoint(101, 101))
    host._move_page_drag_target(QPoint(102, 102))
    assert view.moves == [(100, 100)]
    assert host._pending_drag_target == (102, 102)

    host._flush_page_drag_target(force=True)
    assert view.moves == [(100, 100), (102, 102)]
    assert host._pending_drag_target is None


def test_surface_mask_is_not_rescheduled_while_page_drag_is_active(monkeypatch) -> None:
    """持续页面拖动不能让 Shape 防抖定时器中途恢复并触发重绘竞态。"""

    renderer = SimpleNamespace(view=object())
    host = WebPetHost(renderer)
    host._surface_mask_enabled = True
    host._page_drag_origin = QPoint(1, 1)
    host._page_window_origin = QPoint(10, 10)
    host._page_dragging = True
    scheduled: list[bool] = []
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: scheduled.append(True))

    host._apply_page_drag_target = lambda _target: scheduled.append(False)  # type: ignore[method-assign]
    host._move_page_drag_target(QPoint(20, 20))
    host._flush_page_drag_target(force=True)

    assert scheduled == [False]
    # 清理释放状态后才允许安排一次恢复。
    host._clear_page_drag()
    host._schedule_surface_mask_after_move()
    # _clear_page_drag 与 renderer release 都会安排一次下一事件循环的
    # 防抖恢复；两次调用必须幂等合并，测试不应依赖内部调用次数。
    assert scheduled[0] is False
    assert scheduled[-1] is True
    assert len(scheduled) >= 2


def test_stale_surface_mask_timer_cannot_finish_during_drag(monkeypatch) -> None:
    """已排队的旧 Shape 定时器在拖动开始后也必须失效。"""

    host = WebPetHost(SimpleNamespace(view=object()))
    host._surface_mask_enabled = True
    host._surface_mask_move_active = True
    host._surface_mask_move_generation = 3
    host._page_drag_origin = QPoint(2, 2)
    host._page_dragging = True
    refreshed: list[bool] = []
    monkeypatch.setattr(host, "_refresh_surface_mask", lambda: refreshed.append(True))

    host._finish_surface_mask_move(3)

    assert refreshed == []
    assert host._surface_mask_move_active is True


def test_stale_surface_capture_callback_cannot_commit_during_geometry_transaction(
    monkeypatch,
) -> None:
    """异步 canvas 回读不能在新拖动事务中覆盖旧 Shape。"""

    view = SimpleNamespace(width=lambda: 220, height=lambda: 180)
    host = WebPetHost(SimpleNamespace(view=view, page_ready=True))
    host._surface_mask_enabled = True
    host._surface_mask_capture_generation = 4
    host._surface_mask_capture_pending = True
    host._geometry_generation = 9
    committed: list[bool] = []
    monkeypatch.setattr(
        host,
        "_commit_surface_mask_regions",
        lambda *_args, **_kwargs: committed.append(True),
    )

    host._surface_mask_move_active = True
    host._finish_surface_mask_capture(view, 4, {}, 9)

    assert committed == []
    assert host._surface_mask_capture_pending is False

    host._surface_mask_capture_pending = True
    host._finish_surface_mask_capture(view, 4, {}, 8)
    assert committed == []
    assert host._surface_mask_capture_pending is False


def test_async_move_result_is_discarded_after_new_drag_transaction() -> None:
    """平台异步移动回执迟到时不能把窗口状态写回旧事务。"""

    class View:
        def pos(self):
            return QPoint(20, 30)

    class Platform:
        def __init__(self) -> None:
            self.ready = asyncio.Event()

        def move_overlay(self, _view, _x, _y):
            async def wait_for_result():
                await self.ready.wait()
                return {"status": "available"}

            return wait_for_result()

    async def run() -> dict[str, object]:
        platform = Platform()
        host = WebPetHost(SimpleNamespace(view=View()), platform=platform)
        operation = host.move_to(80, 90)
        task = asyncio.create_task(operation)
        await asyncio.sleep(0)
        host._invalidate_geometry()
        platform.ready.set()
        return await task

    result = asyncio.run(run())
    assert result["status"] == "cancelled"
    assert result["detail"] == "geometry transaction superseded"


def test_delayed_page_resize_cannot_run_after_drag_starts(monkeypatch) -> None:
    """旧 viewport resize 回调不能在新拖动事务中改写页面布局。"""

    class Page:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def runJavaScript(self, script, _callback=None):  # noqa: N802
            self.calls.append(str(script))

    class View:
        page_ready = True

        def __init__(self) -> None:
            self._page = Page()

        def page(self):
            return self._page

    view = View()
    host = WebPetHost(SimpleNamespace(view=view, page_ready=True))
    callbacks: list[object] = []
    monkeypatch.setattr(
        "gui.qt6.web_host.QTimer.singleShot",
        lambda _delay, callback: callbacks.append(callback),
    )

    host._request_page_resize(320, 240)
    assert len(view._page.calls) == 1
    assert len(callbacks) == 1
    host._invalidate_geometry()
    callbacks[0]()
    assert len(view._page.calls) == 1


def test_page_drag_platform_unavailable_keeps_qt_fallback(monkeypatch) -> None:
    """平台明确返回 unavailable 时仍保留一次 Qt 回退。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(0, 0)
            self.moves = 0

        def pos(self):
            return self.origin

        def move(self, x, y):
            self.moves += 1
            self.origin = QPoint(int(x), int(y))

    class Platform:
        def move_overlay(self, _view, _x, _y):
            return {"status": "unavailable"}

    view = View()
    host = WebPetHost(SimpleNamespace(view=view), platform=Platform())
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: None)
    host._move_page_drag_target(QPoint(40, 50))
    assert view.moves == 1
    assert view.pos() == QPoint(40, 50)


def test_page_drag_uses_screen_coordinates_when_window_follows_pointer(monkeypatch) -> None:
    """窗口跟随指针后，页面局部坐标变化不能抵消真实屏幕位移。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(100, 120)
            self.moves: list[tuple[int, int]] = []

        def pos(self):
            return self.origin

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))
            self.moves.append((int(x), int(y)))

    view = View()
    host = WebPetHost(SimpleNamespace(view=view, page_bridge_ready=True))
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_page_drag_flush", lambda: None)
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: None)

    # 初始局部点为 (20, 30)，全局点为 (120, 150)。窗口移动后，
    # 页面局部点反而回到 (30, 40)，但全局点继续向右下移动。
    host._handle_page_interaction(
        {
            "type": "press",
            "button": "left",
            "x": 20,
            "y": 30,
            "screen_x": 120,
            "screen_y": 150,
            "hit": True,
        }
    )
    host._handle_page_interaction(
        {
            "type": "move",
            "button": "other",
            "x": 30,
            "y": 40,
            "screen_x": 160,
            "screen_y": 190,
            "hit": True,
        }
    )
    assert view.pos() == QPoint(140, 160)
    # 第二个事件的局部坐标继续变化，但全局位移仍按 (80, 100) 计算；
    # 旧实现会把已移动窗口的局部坐标误当成固定原点，产生错误回弹。
    host._handle_page_interaction(
        {
            "type": "move",
            "button": "other",
            "x": 35,
            "y": 45,
            "screen_x": 200,
            "screen_y": 240,
            "hit": True,
        }
    )
    host._handle_page_interaction(
        {
            "type": "release",
            "button": "left",
            "x": 40,
            "y": 50,
            "screen_x": 220,
            "screen_y": 270,
            "hit": True,
        }
    )
    assert view.pos() == QPoint(200, 240)
    assert host._page_drag_origin is None


def test_page_press_interrupts_behavior_once_per_gesture() -> None:
    """页面按下边沿立即中断后台行为，移动/释放不重复触发。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(100, 120)

        def pos(self):
            return self.origin

    host = WebPetHost(SimpleNamespace(view=View(), page_bridge_ready=True))
    interruptions: list[str] = []
    host.set_interaction_interrupt(lambda: interruptions.append("interrupt"))

    host._handle_page_interaction(
        {"type": "press", "button": "left", "x": 20, "y": 30, "hit": True}
    )
    host._handle_page_interaction(
        {"type": "move", "button": "other", "x": 40, "y": 50, "hit": True}
    )
    assert interruptions == ["interrupt"]
    host._handle_page_interaction(
        {"type": "release", "button": "left", "x": 40, "y": 50, "hit": True}
    )
    host._handle_page_interaction(
        {"type": "press", "button": "left", "x": 20, "y": 30, "hit": True}
    )
    assert interruptions == ["interrupt", "interrupt"]


def test_page_drag_null_screen_coordinates_keep_legacy_local_path(monkeypatch) -> None:
    """旧浏览器发送 null 屏幕坐标时仍可用局部坐标拖动。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(100, 120)

        def pos(self):
            return self.origin

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))

    view = View()
    host = WebPetHost(SimpleNamespace(view=view, page_bridge_ready=True))
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_page_drag_flush", lambda: None)
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: None)

    host._handle_page_interaction(
        {
            "type": "press",
            "button": "left",
            "x": 20,
            "y": 30,
            "screen_x": None,
            "screen_y": None,
            "hit": True,
        }
    )
    host._handle_page_interaction(
        {
            "type": "move",
            "button": "other",
            "x": 40,
            "y": 50,
            "screen_x": None,
            "screen_y": None,
            "hit": True,
        }
    )
    host._handle_page_interaction(
        {
            "type": "release",
            "button": "left",
            "x": 40,
            "y": 50,
            "screen_x": None,
            "screen_y": None,
            "hit": True,
        }
    )
    assert view.pos() == QPoint(120, 140)


def test_page_drag_callback_is_ignored_during_shutdown() -> None:
    """WebChannel 迟到拖动事件不能在关闭阶段再次移动窗口。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(0, 0)
            self.moves = 0

        def pos(self):
            return self.origin

        def move(self, x, y):
            self.moves += 1
            self.origin = QPoint(int(x), int(y))

    view = View()
    host = WebPetHost(SimpleNamespace(view=view))
    host.prepare_shutdown()
    host._page_drag_origin = QPoint(0, 0)
    host._page_window_origin = QPoint(0, 0)
    host._page_dragging = True
    host._move_page_drag_target(QPoint(20, 20))
    assert view.moves == 0


def test_page_events_are_replayed_after_native_surface_rebuild(monkeypatch) -> None:
    """置顶/穿透重建期间的页面拖动事件按顺序回放，不能丢失拖动终点。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(100, 120)

        def pos(self):
            return self.origin

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))

    view = View()
    host = WebPetHost(SimpleNamespace(view=view, page_bridge_ready=True))
    monkeypatch.setattr(host, "_schedule_deferred_page_events", lambda: None)
    host._interaction_rebuild_until = time.monotonic() + 0.5

    host._handle_page_interaction(
        {"type": "press", "button": "left", "x": 10, "y": 20, "hit": True}
    )
    host._handle_page_interaction({"type": "move", "button": "left", "x": 60, "y": 70, "hit": True})
    host._handle_page_interaction(
        {"type": "release", "button": "left", "x": 60, "y": 70, "hit": True}
    )

    assert view.pos() == QPoint(100, 120)
    assert [item["type"] for item in host._deferred_page_events] == [
        "press",
        "move",
        "release",
    ]

    host._interaction_rebuild_until = 0.0
    host._drain_deferred_page_events()

    assert view.pos() == QPoint(150, 170)
    assert host._deferred_page_events == []
    assert host._page_drag_origin is None
    assert host._page_dragging is False


def test_page_rebuild_event_queue_keeps_terminal_release_bounded(monkeypatch) -> None:
    """连续重建不会无限积压 move，终态 release 必须留在有界队列中。"""

    class View:
        def pos(self):
            return QPoint(0, 0)

    host = WebPetHost(SimpleNamespace(view=View(), page_bridge_ready=True))
    monkeypatch.setattr(host, "_schedule_deferred_page_events", lambda: None)
    host._interaction_rebuild_until = time.monotonic() + 1.0

    for index in range(32):
        host._handle_page_interaction(
            {
                "type": "move",
                "button": "left",
                "x": index,
                "y": index,
                "hit": True,
            }
        )
    host._handle_page_interaction(
        {"type": "release", "button": "left", "x": 31, "y": 31, "hit": True}
    )

    assert len(host._deferred_page_events) <= 8
    assert host._deferred_page_events[-1]["type"] == "release"


def test_page_release_commits_drag_when_intermediate_move_is_lost(monkeypatch) -> None:
    """WebChannel 丢失中间 move 时，release 仍提交超阈值的最终坐标。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(100, 120)
            self.moves: list[tuple[int, int]] = []

        def pos(self):
            return self.origin

        def move(self, x, y):
            self.moves.append((int(x), int(y)))
            self.origin = QPoint(int(x), int(y))

    dragging_states: list[bool] = []
    view = View()
    host = WebPetHost(
        SimpleNamespace(
            view=view,
            page_bridge_ready=True,
            set_dragging=lambda value: dragging_states.append(bool(value)),
        )
    )
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: None)

    host._handle_page_interaction(
        {
            "type": "press",
            "button": "left",
            "x": 20,
            "y": 30,
            "screen_x": 120,
            "screen_y": 150,
            "hit": True,
        }
    )
    # 模拟重建/节流期间唯一的 pointermove 没有抵达宿主，release 仍携带
    # 浏览器的最终屏幕坐标。
    host._handle_page_interaction(
        {
            "type": "release",
            "button": "left",
            "x": 70,
            "y": 70,
            "screen_x": 170,
            "screen_y": 190,
            "hit": True,
        }
    )

    assert view.pos() == QPoint(150, 160)
    assert view.moves == [(150, 160)]
    assert dragging_states == [True, False]
    assert host._page_drag_origin is None
    assert host._page_dragging is False


def test_deferred_page_queue_preserves_lone_active_drag_move(monkeypatch) -> None:
    """满队列时必须保留活动 press 后唯一的 move 和最终 release。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(100, 120)

        def pos(self):
            return self.origin

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))

    view = View()
    host = WebPetHost(SimpleNamespace(view=view, page_bridge_ready=True))
    monkeypatch.setattr(host, "_schedule_deferred_page_events", lambda: None)
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: None)
    host._interaction_rebuild_until = time.monotonic() + 1.0

    # 四段已结束点击填满队列；随后只有一个活动 move 与最终 release。
    for _index in range(4):
        host._handle_page_interaction(
            {"type": "press", "button": "left", "x": 10, "y": 10, "hit": True}
        )
        host._handle_page_interaction(
            {"type": "release", "button": "left", "x": 10, "y": 10, "hit": True}
        )
    host._handle_page_interaction(
        {
            "type": "press",
            "button": "left",
            "x": 20,
            "y": 30,
            "screen_x": 120,
            "screen_y": 150,
            "hit": True,
        }
    )
    host._handle_page_interaction(
        {
            "type": "move",
            "button": "other",
            "x": 70,
            "y": 70,
            "screen_x": 170,
            "screen_y": 190,
            "hit": True,
        }
    )
    host._handle_page_interaction(
        {
            "type": "release",
            "button": "left",
            "x": 70,
            "y": 70,
            "screen_x": 170,
            "screen_y": 190,
            "hit": True,
        }
    )

    active_press = max(
        index
        for index, payload in enumerate(host._deferred_page_events)
        if payload["type"] == "press"
    )
    active_events = host._deferred_page_events[active_press + 1 :]
    assert any(payload["type"] == "move" for payload in active_events)
    assert host._deferred_page_events[-1]["type"] == "release"

    host._interaction_rebuild_until = 0.0
    host._drain_deferred_page_events()

    assert view.pos() == QPoint(150, 160)
    assert host._page_drag_origin is None


def test_page_rebuild_event_queue_preserves_press_through_move_overflow(monkeypatch) -> None:
    """页面重建期间高频 move 溢出时仍须保留同一手势的 press。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(100, 120)
            self.moves: list[tuple[int, int]] = []

        def pos(self):
            return self.origin

        def move(self, x, y):
            self.origin = QPoint(int(x), int(y))
            self.moves.append((int(x), int(y)))

    view = View()
    host = WebPetHost(SimpleNamespace(view=view, page_bridge_ready=True))
    monkeypatch.setattr(host, "_schedule_deferred_page_events", lambda: None)
    host._interaction_rebuild_until = time.monotonic() + 1.0

    host._handle_page_interaction(
        {"type": "press", "button": "left", "x": 10, "y": 20, "hit": True}
    )
    for index in range(1, 33):
        host._handle_page_interaction(
            {
                "type": "move",
                "button": "left",
                "x": 10 + index * 2,
                "y": 20 + index * 2,
                "hit": True,
            }
        )
    host._handle_page_interaction(
        {"type": "release", "button": "left", "x": 74, "y": 84, "hit": True}
    )

    assert len(host._deferred_page_events) <= 8
    assert host._deferred_page_events[0]["type"] == "press"
    assert host._deferred_page_events[-1]["type"] == "release"

    host._interaction_rebuild_until = 0.0
    host._drain_deferred_page_events()

    assert view.pos() == QPoint(164, 184)
    assert view.moves[-1] == (164, 184)
    assert host._page_drag_origin is None
    assert host._page_dragging is False


def test_autonomous_move_cancelled_error_releases_geometry_reservation() -> None:
    """自主移动同步抛出取消时必须归还几何 pending。"""

    class View:
        def pos(self):
            return QPoint(20, 30)

    class Platform:
        def move_overlay(self, *_args):
            raise asyncio.CancelledError

    host = WebPetHost(SimpleNamespace(view=View()), platform=Platform())
    generation = host._geometry_generation
    try:
        host.move_to(48, 60, duration_ms=0)
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("move_to should propagate asyncio.CancelledError")

    assert host._autonomous_geometry_pending == 0
    assert host._autonomous_geometry_active() is True
    assert host._geometry_generation == generation + 1


def test_autonomous_move_unexpected_exception_releases_geometry_reservation() -> None:
    """未列举的平台异常向上传递前也必须归还 pending。"""

    class View:
        def pos(self):
            return QPoint(20, 30)

    class Platform:
        def move_overlay(self, *_args):
            raise LookupError("platform probe failed")

    host = WebPetHost(SimpleNamespace(view=View()), platform=Platform())
    try:
        host.move_to(48, 60, duration_ms=0)
    except LookupError:
        pass
    else:
        raise AssertionError("unexpected platform exception should propagate")

    assert host._autonomous_geometry_pending == 0


def test_autonomous_move_prestart_cancellation_does_not_claim_pending() -> None:
    """返回协程在首次调度前取消时不应遗留 pending。"""

    class Deferred:
        def __await__(self):
            if False:
                yield None
            return {"status": "available"}

    class View:
        def pos(self):
            return QPoint(20, 30)

    class Platform:
        def move_overlay(self, *_args):
            return Deferred()

    async def run() -> tuple[int, bool]:
        host = WebPetHost(SimpleNamespace(view=View()), platform=Platform())
        operation = host.move_to(48, 60, duration_ms=0)
        assert host._autonomous_geometry_pending == 0
        task = asyncio.create_task(operation)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("pre-start cancellation should propagate")
        return host._autonomous_geometry_pending, host._autonomous_geometry_active()

    pending, active = asyncio.run(run())
    assert pending == 0
    assert active is True


def test_renderer_autonomous_move_prestart_cancellation_releases_geometry() -> None:
    """Web 渲染器回退路径也必须采用延迟 pending 归属。"""

    class Deferred:
        def __await__(self):
            if False:
                yield None
            return {"status": "available"}

    class View:
        def pos(self):
            return QPoint(20, 30)

    class Renderer:
        view = View()

        def move_to(self, *_args, **_kwargs):
            return Deferred()

    async def run() -> int:
        host = WebPetHost(Renderer())
        operation = host.move_to(48, 60, duration_ms=0)
        assert host._autonomous_geometry_pending == 0
        task = asyncio.create_task(operation)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return host._autonomous_geometry_pending

    assert asyncio.run(run()) == 0


def test_autonomous_move_started_cancellation_releases_pending() -> None:
    """协程已经开始等待平台时取消，也必须归还 pending。"""

    class Deferred:
        def __init__(self, event: asyncio.Event) -> None:
            self._event = event

        def __await__(self):
            async def wait():
                await self._event.wait()
                return {"status": "available"}

            return wait().__await__()

    class View:
        def pos(self):
            return QPoint(20, 30)

    class Platform:
        def __init__(self, event: asyncio.Event) -> None:
            self._event = event

        def move_overlay(self, *_args):
            return Deferred(self._event)

    async def run() -> int:
        event = asyncio.Event()
        platform = Platform(event)
        host = WebPetHost(SimpleNamespace(view=View()), platform=platform)
        task = asyncio.create_task(host.move_to(48, 60, duration_ms=0))
        await asyncio.sleep(0)
        assert host._autonomous_geometry_pending == 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return host._autonomous_geometry_pending

    assert asyncio.run(run()) == 0


def test_autonomous_pending_survives_drag_superseding_inflight_moves() -> None:
    """拖动抢占时旧异步移动的归属不能误扣新移动 pending。"""

    class Deferred:
        def __init__(self, event: asyncio.Event) -> None:
            self._event = event

        def __await__(self):
            async def wait():
                await self._event.wait()
                return {"status": "available"}

            return wait().__await__()

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(20, 30)

        def pos(self):
            return self.origin

    class Platform:
        def __init__(self, values: list[Deferred]) -> None:
            self._values = values

        def move_overlay(self, *_args):
            return self._values.pop(0)

    async def run() -> int:
        event = asyncio.Event()
        platform = Platform([Deferred(event), Deferred(event)])
        renderer = SimpleNamespace(view=View(), set_dragging=lambda _value: None)
        host = WebPetHost(renderer, platform=platform)

        first = asyncio.create_task(host.move_to(48, 60, duration_ms=0))
        await asyncio.sleep(0)
        assert host._autonomous_geometry_pending == 1

        host._set_renderer_dragging(True)
        assert host._autonomous_geometry_pending == 1
        host._set_renderer_dragging(False)
        host._autonomous_block_until = 0.0

        second = asyncio.create_task(host.move_to(80, 90, duration_ms=0))
        await asyncio.sleep(0)
        assert host._autonomous_geometry_pending == 2

        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        assert host._autonomous_geometry_pending == 1

        second.cancel()
        try:
            await second
        except asyncio.CancelledError:
            pass
        return host._autonomous_geometry_pending

    assert asyncio.run(run()) == 0


def test_autonomous_step_is_rejected_during_high_frequency_page_drag(monkeypatch) -> None:
    """高频拖动期间的自主步进不得改变窗口位置或遗留 pending 坐标。"""

    class View:
        def __init__(self) -> None:
            self.origin = QPoint(80, 90)
            self.moves: list[tuple[int, int]] = []

        def pos(self):
            return self.origin

        def move(self, x, y):
            point = (int(x), int(y))
            self.moves.append(point)
            self.origin = QPoint(*point)

    view = View()
    host = WebPetHost(SimpleNamespace(view=view, page_bridge_ready=True))
    monkeypatch.setattr(host, "_schedule_page_drag_flush", lambda: None)
    monkeypatch.setattr(host, "_begin_surface_mask_move", lambda: None)
    monkeypatch.setattr(host, "_schedule_surface_mask_after_move", lambda: None)

    host._handle_page_interaction(
        {"type": "press", "button": "left", "x": 40, "y": 40, "hit": True}
    )
    for offset in range(4, 204, 4):
        host._handle_page_interaction(
            {
                "type": "move",
                "button": "left",
                "x": 40 + offset,
                "y": 40 + offset // 2,
                "hit": True,
            }
        )
        if offset == 100:
            before = view.pos()
            rejected = host.move_to(1200, 700, duration_ms=0)
            assert rejected["status"] == "cancelled"
            assert view.pos() == before

    host._handle_page_interaction(
        {"type": "release", "button": "left", "x": 240, "y": 140, "hit": True}
    )

    assert host._pending_drag_target is None
    assert host._page_dragging is False
    assert view.pos() == QPoint(280, 190)
    assert len(view.moves) < 60
