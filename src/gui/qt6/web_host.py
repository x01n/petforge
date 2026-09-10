"""Qt WebEngine Live2D 窗口的桌宠控制边界。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import inspect
import json
import logging
import math
import os
import weakref
from collections.abc import Callable, Mapping
from time import monotonic
from typing import Any

from core.rendering.actions import ExpressionRequest, MotionRequest
from gui.qt6.display_size import display_size_preset
from gui.qt6.pet_interaction import (
    PET_CLICK_DEBOUNCE_MS,
    classify_pet_part,
    make_pet_click_payload,
)
from logger.events import event_fingerprint, log_event

logger = logging.getLogger(__name__)

_SURFACE_MASK_REFRESH_INTERVAL_MS = 160
# 高频 Shape 重建会和 X11/Chromium 原生子表面的移动提交竞争，导致
# 合成器短暂显示黑色旧帧。输入命中由页面 hitTest 独立保证，因此动作期间
# 使用 100ms 的有界刷新，保留可见区域更新又避免每个动画 tick 重建 native mask。
_SURFACE_MASK_MOTION_INTERVAL_MS = 100
_SURFACE_MASK_CAPTURE_INTERVAL_SECONDS = 0.1
_MAX_SURFACE_MASK_DATA_BYTES = 16 * 1024 * 1024
# 鼠标/触控板事件可能远高于渲染帧率。拖动提交以 120Hz 为上限，
# 并在释放时补交最后坐标；这样不会牺牲跟手性，也不会让 Qt/窗口管理器
# 的几何请求队列积压旧坐标。
_DRAG_MOVE_INTERVAL_SECONDS = 1.0 / 120.0
_AUTONOMOUS_RESUME_GRACE_SECONDS = 0.35
# 自主游走的连续小步不需要重建相对窗口的 Shape；留出短稳定窗口，
# 让定时器不会在 QWidget.move 与 Chromium 帧提交之间抓取/写入旧区域。
_AUTONOMOUS_GEOMETRY_SETTLE_SECONDS = 0.08
# 置顶切换不能插入正在进行的拖动事务；若 compositor 丢失 release，
# 仍需要一个有界等待窗口来释放 pending/Shape 状态。15 秒足够覆盖正常
# 的长按拖动，同时避免窗口控制永久停留在“正在应用”。
_WINDOW_FLAG_DRAG_RETRY_INTERVAL_MS = 120
_WINDOW_FLAG_DRAG_WAIT_TIMEOUT_SECONDS = 15.0
# xcb 在 QWindow.setFlag() 返回后异步提交 EWMH 属性；以短、有界回读确认
# 窗口管理器真实状态，不能把 Qt 客户端旗标当作已经置顶。
_X11_TOPMOST_VERIFY_DELAYS_MS = (0, 40, 120, 320, 700)
# 平台实现允许返回 awaitable，但窗口控制不能无限等待 IPC/线程调度；
# 超时后必须释放 Shape 暂停状态，让恢复入口仍然可用。
_WINDOW_FLAG_DISPATCH_TIMEOUT_SECONDS = 3.0
# 任何拖动事务都必须有界收敛；窗口失焦或合成器丢失 release 时，不能
# 永久阻断光标追踪、自主行动和窗口控制。
_DRAG_TRANSACTION_TIMEOUT_MS = 15_000


def _safe_close_awaitable(value: object) -> None:
    """回收未被外层协程消费的内部 awaitable，避免退出时产生警告。"""

    closer = getattr(value, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:
        # 对象可能已经由任务取消路径关闭；回收阶段不能再传播异常。
        return


def _attach_awaitable_finalizer(owner: object, value: object) -> None:
    """让外层协程被取消前未启动时仍能关闭内部协程。"""

    if not callable(getattr(value, "close", None)):
        return
    try:
        weakref.finalize(owner, _safe_close_awaitable, value)
    except (TypeError, ValueError):
        return


def _qt_object_alive(value: object | None) -> bool:
    """检查 Qt 包装对象是否仍指向有效的原生对象。"""

    if value is None:
        return False
    if not _qt_available:
        return True
    try:
        import shiboken6

        return bool(shiboken6.isValid(value))
    except (ImportError, OSError, RuntimeError, TypeError, ValueError):
        # 纯 Python 测试替身没有 Shiboken 类型；交由各调用点的异常
        # 防护处理，不能因为可选检查不可用而屏蔽兼容对象。
        return True


def _set_cursor_shape(target: object, shape_name: str) -> None:
    """给桌宠窗口设置拖动光标提示；失败时不影响输入链路。"""

    setter = getattr(target, "setCursor", None)
    if not callable(setter):
        return
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QCursor

        cursor_shape = getattr(getattr(Qt, "CursorShape", Qt), shape_name)
        setter(QCursor(cursor_shape))
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return


def _set_no_focus(target: object) -> None:
    """让展示用 WebEngine/Quick 子窗口不参与应用焦点竞争。"""

    if target is None:
        return
    try:
        from PySide6.QtCore import Qt

        focus_policy = getattr(getattr(Qt, "FocusPolicy", Qt), "NoFocus", None)
        if focus_policy is None:
            focus_policy = getattr(Qt, "NoFocus", None)
        setter = getattr(target, "setFocusPolicy", None)
        if callable(setter) and focus_policy is not None:
            setter(focus_policy)
        proxy_getter = getattr(target, "focusProxy", None)
        proxy = proxy_getter() if callable(proxy_getter) else None
        if proxy is not None and proxy is not target:
            proxy_setter = getattr(proxy, "setFocusPolicy", None)
            if callable(proxy_setter) and focus_policy is not None:
                proxy_setter(focus_policy)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return


try:  # WebEngine 是可选依赖，核心服务不能因为 Qt 缺失而无法导入。
    from PySide6.QtCore import QCoreApplication, QEvent, QObject, QPoint, Qt, QTimer

    _qt_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    _qt_available = False


if _qt_available:

    class _WebInteractionFilter(QObject):
        """为 WebEngine 视图提供桌宠拖动和右键菜单事件。"""

        def __init__(
            self,
            view: object,
            context_menu_callback: Callable[[QPoint], None] | None,
            double_click_callback: Callable[[], None] | None,
            target_callback: Callable[[object], None] | None = None,
            click_callback: Callable[[Mapping[str, object]], None] | None = None,
            hit_test_callback: Callable[[QPoint], bool | None] | None = None,
            dragging_callback: Callable[[bool], object] | None = None,
            interaction_started_callback: Callable[[], object] | None = None,
            enter_callback: Callable[[], None] | None = None,
        ) -> None:
            super().__init__(view)
            self._view = view
            self._context_menu_callback = context_menu_callback
            self._double_click_callback = double_click_callback
            self._target_callback = target_callback
            self._click_callback = click_callback
            self._hit_test_callback = hit_test_callback
            self._dragging_callback = dragging_callback
            self._interaction_started_callback = interaction_started_callback
            self._enter_callback = enter_callback
            self._drag_origin: QPoint | None = None
            self._window_origin: QPoint | None = None
            self._system_move_active = False
            self._dragging = False
            self._last_move_target: tuple[int, int] | None = None
            self._pending_move_target: tuple[int, int] | None = None
            self._last_move_at = 0.0
            self._hover_targets: set[int] = set()
            self._click_token = 0
            self._drag_watchdog_timer: QTimer | None = None

        def _notify_interaction_started(self) -> None:
            """在原生按下边沿立即通知行为服务。"""

            callback = self._interaction_started_callback
            if not callable(callback):
                return
            try:
                result = callback()
                if inspect.isawaitable(result):
                    _safe_close_awaitable(result)
            except Exception:
                logger.debug("failed to interrupt behavior on native press", exc_info=True)

        def install_on(self, target: object) -> None:
            """把过滤器安装到 WebEngine 当前及后续创建的子对象。"""

            _set_no_focus(target)
            installer = getattr(target, "installEventFilter", None)
            if callable(installer):
                try:
                    installer(self)
                    if self._target_callback is not None:
                        self._target_callback(target)
                except (AttributeError, RuntimeError, TypeError):
                    return
            finder = getattr(target, "findChildren", None)
            if not callable(finder):
                return
            try:
                children = tuple(finder(QObject))
            except (AttributeError, RuntimeError, TypeError):
                return
            for child in children:
                if child is target or child is self:
                    continue
                _set_no_focus(child)
                child_installer = getattr(child, "installEventFilter", None)
                if callable(child_installer):
                    try:
                        child_installer(self)
                        if self._target_callback is not None:
                            self._target_callback(child)
                    except (AttributeError, RuntimeError, TypeError):
                        continue

        def _start_system_move(self) -> bool:
            """在 Wayland 上让合成器接管 QWidget 顶层拖动。"""

            # X11 保留手动拖动路径，这样双击事件仍能到达过滤器；部分
            # X11 窗口管理器会在 startSystemMove() 返回后吞掉后续双击。
            if not self._is_wayland():
                return False

            handle_getter = getattr(self._view, "windowHandle", None)
            handle = handle_getter() if callable(handle_getter) else None
            starter = getattr(handle, "startSystemMove", None)
            if not callable(starter):
                return False
            try:
                accepted = bool(starter())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False
            self._system_move_active = accepted
            if accepted:
                callback = self._dragging_callback
                if callable(callback):
                    try:
                        callback(True)
                    except Exception:
                        logger.debug("failed to set Web native dragging state", exc_info=True)
            return accepted

        @staticmethod
        def _is_wayland() -> bool:
            try:
                from PySide6.QtGui import QGuiApplication

                return "wayland" in str(QGuiApplication.platformName()).lower()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False

        @staticmethod
        def _global_point(event: object) -> QPoint | None:
            getter = getattr(event, "globalPosition", None)
            if callable(getter):
                try:
                    value = getter()
                    to_point = getattr(value, "toPoint", None)
                    if callable(to_point):
                        return to_point()
                    if isinstance(value, QPoint):
                        return value
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            getter = getattr(event, "globalPos", None)
            if callable(getter):
                try:
                    value = getter()
                    return value if isinstance(value, QPoint) else QPoint(value)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            return None

        def _local_point(self, point: QPoint | None) -> QPoint | None:
            """把原生过滤器的全局坐标转换为 Web 视图局部坐标。"""

            if point is None:
                return None
            mapper = getattr(self._view, "mapFromGlobal", None)
            if not callable(mapper):
                return point
            try:
                value = mapper(point)
                return value if isinstance(value, QPoint) else QPoint(value)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return None

        def _arm_drag_watchdog(self) -> None:
            """为原生过滤器拖动事务设置有界收敛计时器。"""

            try:
                timer = self._drag_watchdog_timer
                if timer is None:
                    timer = QTimer(self)
                    timer.setSingleShot(True)
                    timer.timeout.connect(self._drag_watchdog_expired)
                    self._drag_watchdog_timer = timer
                timer.start(_DRAG_TRANSACTION_TIMEOUT_MS)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return

        def _drag_watchdog_expired(self) -> None:
            """清理丢失 release 的原生拖动，不触碰已结束的手势。"""

            if not (self._drag_origin is not None or self._dragging or self._system_move_active):
                return
            self._clear_drag()
            _set_cursor_shape(self._view, "ArrowCursor")

        def _clear_drag(self) -> None:
            # release 事件会在清理前显式冲刷 pending；其它取消路径直接丢弃
            # 旧坐标，避免下一次按下沿用上一段拖动的尾部请求。
            self._drag_origin = None
            self._window_origin = None
            self._system_move_active = False
            self._dragging = False
            self._last_move_target = None
            self._pending_move_target = None
            self._last_move_at = 0.0
            watchdog = self._drag_watchdog_timer
            if watchdog is not None:
                try:
                    watchdog.stop()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            release = getattr(self._view, "releaseMouse", None)
            if callable(release):
                try:
                    release()
                except (AttributeError, RuntimeError):
                    pass
            # 先清除过滤器自身的拖动字段，再通知宿主恢复 Shape；否则宿主
            # 的恢复回调会看到旧 `_drag_origin`，误以为拖动仍在进行而把
            # Shape 暂停永久延长。
            callback = self._dragging_callback
            if callable(callback):
                try:
                    callback(False)
                except Exception:
                    logger.debug("failed to clear Web native dragging state", exc_info=True)
            self._reconcile_hover_state()

        def _reconcile_hover_state(self) -> None:
            """拖动结束后清理窗口外遗留的 hover 标志。"""

            if not self._hover_targets:
                return
            try:
                from PySide6.QtGui import QCursor

                global_point = QCursor.pos()
                local = self._local_point(global_point)
                width_getter = getattr(self._view, "width", None)
                height_getter = getattr(self._view, "height", None)
                width = int(width_getter()) if callable(width_getter) else 0
                height = int(height_getter()) if callable(height_getter) else 0
                inside = (
                    local is not None
                    and 0 <= int(local.x()) < max(1, width)
                    and 0 <= int(local.y()) < max(1, height)
                )
                if not inside or self._model_hit(local) is False:
                    self._hover_targets.clear()
            except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
                # 无法读取光标时保留 hover，等待真实 Leave；不以猜测覆盖
                # 用户仍在窗口内的状态。
                return

        def _model_hit(self, point: QPoint | None) -> bool | None:
            """返回原生回退路径的精确命中结果。

            ``None`` 表示旧测试替身/旧渲染器没有命中能力，此时保留兼容
            行为；真实 Web Live2D 在页面桥尚未握手时必须 fail-closed，避免
            原生矩形视口先抢焦点再把透明点当成拖动起点。
            """

            callback = self._hit_test_callback
            if callback is None:
                return None
            try:
                return callback(point) if point is not None else False
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False

        def _schedule_click(self, payload: Mapping[str, object]) -> None:
            """延迟普通点击，给双击事件一个取消窗口。"""

            callback = self._click_callback
            if callback is None:
                return
            self._click_token += 1
            token = self._click_token

            def emit() -> None:
                if token != self._click_token:
                    return
                try:
                    callback(dict(payload))
                except Exception:
                    logger.exception("Web Live2D native click callback failed")

            try:
                QTimer.singleShot(PET_CLICK_DEBOUNCE_MS, emit)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                emit()

        def eventFilter(self, watched: object, event: object) -> bool:  # noqa: N802
            event_type = getattr(event, "type", lambda: None)()
            key_press = getattr(QEvent.Type, "KeyPress", None)
            if event_type == key_press and self._hover_targets:
                if event.isAutoRepeat() or event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier:
                    return False
                from PySide6.QtWidgets import QApplication, QLineEdit, QPlainTextEdit, QTextEdit

                if isinstance(QApplication.focusWidget(), (QLineEdit, QTextEdit, QPlainTextEdit)):
                    return False
                key = getattr(event, "key", lambda: None)()
                enter_keys = {
                    getattr(Qt, "Key_Return", None),
                    getattr(Qt, "Key_Enter", None),
                    getattr(getattr(Qt, "Key", object), "Key_Return", None),
                    getattr(getattr(Qt, "Key", object), "Key_Enter", None),
                }
                if key in enter_keys:
                    callback = self._enter_callback
                    if callable(callback):
                        try:
                            callback()
                        except Exception:
                            logger.exception("Web Live2D enter callback failed")
                        return True
            enter_event = getattr(QEvent.Type, "Enter", None)
            leave_event = getattr(QEvent.Type, "Leave", None)
            point = self._global_point(event)
            if event_type == enter_event:
                if self._model_hit(self._local_point(point)) is False:
                    _set_cursor_shape(self._view, "ArrowCursor")
                    return False
                self._hover_targets.add(id(watched))
                _set_cursor_shape(self._view, "OpenHandCursor")
                return False
            if event_type == leave_event:
                self._hover_targets.discard(id(watched))
                if not self._dragging:
                    _set_cursor_shape(self._view, "ArrowCursor")
                return False
            child_added = getattr(QEvent.Type, "ChildAdded", None)
            if event_type == child_added:
                # QWebEngineView 在页面加载后才创建 Chromium 视口；如果只
                # 过滤初始化时已有的 viewport，真实鼠标事件会绕过宿主。
                child_getter = getattr(event, "child", None)
                child = child_getter() if callable(child_getter) else None
                if child is not None:
                    self.install_on(child)
                # 继续让 Qt 完成子对象挂载。
                return False
            mouse_press = getattr(QEvent.Type, "MouseButtonPress", None)
            mouse_double_click = getattr(QEvent.Type, "MouseButtonDblClick", None)
            mouse_move = getattr(QEvent.Type, "MouseMove", None)
            mouse_release = getattr(QEvent.Type, "MouseButtonRelease", None)
            global_point = point
            if event_type == mouse_press:
                button = getattr(event, "button", lambda: Qt.NoButton)()
                if button == Qt.LeftButton and global_point is not None:
                    local_point = self._local_point(global_point)
                    if self._model_hit(local_point) is False:
                        self._clear_drag()
                        _set_cursor_shape(self._view, "ArrowCursor")
                        clear_focus = getattr(self._view, "clearFocus", None)
                        if callable(clear_focus):
                            try:
                                clear_focus()
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                pass
                        return False
                    _set_cursor_shape(self._view, "ClosedHandCursor")
                    self._notify_interaction_started()
                    if self._start_system_move():
                        self._arm_drag_watchdog()
                        return True
                    self._drag_origin = global_point
                    self._dragging = False
                    position = getattr(self._view, "pos", None)
                    self._window_origin = position() if callable(position) else QPoint()
                    # 按下边沿立即抓取鼠标，确保尚未越过拖动阈值时即使指针
                    # 离开 WebEngine 视口，release 仍能回到过滤器。
                    grab = getattr(self._view, "grabMouse", None)
                    if callable(grab):
                        try:
                            grab()
                        except (AttributeError, RuntimeError):
                            pass
                    self._arm_drag_watchdog()
                    # 过滤器必须吞掉按下事件才能在 WebEngine 的内部视口
                    # 重定向后继续收到 move/release；拖动阈值仍避免轻触
                    # 改变窗口位置。Web 页面本身不依赖该顶层鼠标事件来
                    # 驱动桌宠交互，双击/右键由宿主显式转发。
                    return True
                if button == Qt.RightButton:
                    if self._model_hit(self._local_point(global_point)) is False:
                        self._clear_drag()
                        return False
                    _set_cursor_shape(self._view, "ArrowCursor")
                    self._notify_interaction_started()
                    self._clear_drag()
                    callback = self._context_menu_callback
                    if callback is not None and global_point is not None:
                        try:
                            callback(global_point)
                        except Exception:
                            logger.exception("Web Live2D context menu callback failed")
                    return True
            elif event_type == mouse_double_click:
                button = getattr(event, "button", lambda: Qt.NoButton)()
                callback = self._double_click_callback
                if button == Qt.LeftButton and callback is not None and point is not None:
                    if self._model_hit(self._local_point(point)) is False:
                        self._click_token += 1
                        self._clear_drag()
                        return False
                    _set_cursor_shape(self._view, "OpenHandCursor")
                    self._click_token += 1
                    self._clear_drag()
                    try:
                        callback()
                    except Exception:
                        logger.exception("Web Live2D double-click callback failed")
                    return True
            elif event_type == mouse_move and self._system_move_active:
                return True
            elif event_type == mouse_move and self._drag_origin is not None:
                # XTest、部分 Wayland 输入桥和触控板驱动会把拖动中的
                # MouseMove 报告为 NoButton；按下事件已经建立了拖动状态，
                # 这里等 MouseButtonRelease 清理，不能因按钮位缺失而中断。
                if global_point is not None and self._window_origin is not None:
                    self._arm_drag_watchdog()
                    if not self._dragging:
                        try:
                            from PySide6.QtWidgets import QApplication

                            threshold = int(QApplication.startDragDistance())
                        except (ImportError, RuntimeError, TypeError, ValueError):
                            threshold = 4
                        if (global_point - self._drag_origin).manhattanLength() < max(1, threshold):
                            return False
                        self._dragging = True
                        callback = self._dragging_callback
                        if callable(callback):
                            try:
                                callback(True)
                            except Exception:
                                logger.debug(
                                    "failed to set Web native dragging state", exc_info=True
                                )
                        grab = getattr(self._view, "grabMouse", None)
                        if callable(grab):
                            try:
                                grab()
                            except (AttributeError, RuntimeError):
                                pass
                    target = self._window_origin + global_point - self._drag_origin
                    target_tuple = (int(target.x()), int(target.y()))
                    if target_tuple == self._last_move_target:
                        return True
                    self._pending_move_target = target_tuple
                    now = monotonic()
                    previous_target = self._last_move_target
                    distance = (
                        math.hypot(
                            target_tuple[0] - previous_target[0],
                            target_tuple[1] - previous_target[1],
                        )
                        if previous_target is not None
                        else float("inf")
                    )
                    if (
                        self._last_move_at <= 0.0
                        or now - self._last_move_at >= _DRAG_MOVE_INTERVAL_SECONDS
                        or distance >= 8.0
                    ):
                        self._flush_move()
                    return True
            elif event_type == mouse_release:
                button = getattr(event, "button", lambda: Qt.NoButton)()
                # Wayland 的 ``startSystemMove()`` 成功后不会建立本地
                # ``_drag_origin``，但会把 ``_system_move_active`` 置真。
                # 只检查局部拖动原点会漏掉 release，导致过滤器和宿主永久
                # 保持 dragging 状态，后续光标追踪/Shape 恢复都会被阻断。
                if button == Qt.LeftButton and (
                    self._drag_origin is not None or self._system_move_active or self._dragging
                ):
                    system_move_active = self._system_move_active
                    dragging = self._dragging or system_move_active
                    click_payload = None
                    if not dragging and self._drag_origin is not None and global_point is not None:
                        origin_getter = getattr(self._view, "mapFromGlobal", None)
                        local = None
                        if callable(origin_getter):
                            try:
                                local = origin_getter(global_point)
                            except (AttributeError, RuntimeError, TypeError, ValueError):
                                local = None
                        if local is not None:
                            if self._model_hit(local) is False:
                                local = None
                        if local is not None:
                            width_getter = getattr(self._view, "width", None)
                            height_getter = getattr(self._view, "height", None)
                            width = width_getter() if callable(width_getter) else 0
                            height = height_getter() if callable(height_getter) else 0
                            click_payload = make_pet_click_payload(
                                {"x": local.x(), "y": local.y(), "button": "left"},
                                width=width,
                                height=height,
                            )
                    if dragging:
                        if (
                            global_point is not None
                            and self._window_origin is not None
                            and self._drag_origin is not None
                        ):
                            final_target = self._window_origin + global_point - self._drag_origin
                            self._pending_move_target = (
                                int(final_target.x()),
                                int(final_target.y()),
                            )
                        self._flush_move()
                    self._clear_drag()
                    _set_cursor_shape(self._view, "OpenHandCursor")
                    if click_payload is not None:
                        self._schedule_click(click_payload)
                    return dragging or system_move_active
            window_deactivate = getattr(QEvent.Type, "WindowDeactivate", None)
            focus_out = getattr(QEvent.Type, "FocusOut", None)
            if event_type in {window_deactivate, focus_out} and (
                self._drag_origin is not None or self._dragging or self._system_move_active
            ):
                self._clear_drag()
                _set_cursor_shape(self._view, "ArrowCursor")
                return False
            return False

        def _flush_move(self) -> None:
            """提交原生过滤器拖动队列中的最新窗口坐标。"""

            target_tuple = self._pending_move_target
            if target_tuple is None:
                return
            self._pending_move_target = None
            if target_tuple == self._last_move_target:
                return
            move = getattr(self._view, "move", None)
            target = QPoint(*target_tuple)
            if not callable(move):
                return
            try:
                move(target)
            except TypeError:
                try:
                    move(target.x(), target.y())
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    return
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return
            self._last_move_target = target_tuple
            self._last_move_at = monotonic()


else:
    _WebInteractionFilter = None  # type: ignore[assignment]


class WebPetHost:
    """把平台窗口能力与 ``WebLive2DRenderer`` 组合为桌宠控制器。"""

    def __init__(self, renderer: object, *, platform: object | None = None) -> None:
        self.renderer = renderer
        self.platform = platform
        self._interaction_filter: object | None = None
        self._interaction_targets: list[object] = []
        # QWebEngine 页面存在 QWebChannel 指针桥时优先使用页面事件；原生
        # Qt 过滤器仍保留给旧 viewport、测试替身和页面桥未覆盖的事件路径。
        self._page_bridge_enabled = False
        self._click_through = False
        self._click_through_override_active = False
        self._click_through_before_override = False
        self._window_locked = False
        self._click_through_before_lock = False
        self._last_cursor_target: tuple[int, int, int, int] | None = None
        self._last_window_position: tuple[int, int] | None = None
        # QWebEngine 的 Chromium 视口不会把按下/释放事件转成顶层
        # QWidget 事件；页面通过 QWebChannel 回传这些状态，独立于 Qt
        # 原生过滤器保留一份拖动状态。
        self._page_drag_origin: QPoint | None = None
        # 保留页面局部原点用于旧 WebView/测试载荷兼容；新页面同时保存
        # 屏幕全局原点，窗口跟随指针时不再因 client 坐标变化而回弹。
        self._page_drag_global_origin: QPoint | None = None
        self._page_drag_uses_global = False
        self._page_window_origin: QPoint | None = None
        self._page_dragging = False
        self._page_system_move_active = False
        self._page_drag_watchdog_generation = 0
        self._page_pointer_over = False
        # 拖动期间窗口位置会高频变化；追踪复位只需在跨过明显距离或
        # 时间窗口后执行，不能每个 move 都清零速度，否则注视会像“卡住”。
        self._last_drag_target: tuple[int, int] | None = None
        self._last_drag_notify_at = 0.0
        # 页面桥的回调可能在 Chromium 消息队列中突发到达；只保留尚未提交
        # 的最后坐标，按统一频率提交，避免窗口追赶过期位置。
        self._pending_drag_target: tuple[int, int] | None = None
        self._last_drag_submit_at = 0.0
        self._drag_flush_timer: QTimer | None = None
        self._renderer_dragging = False
        self._autonomous_block_until = 0.0
        self._page_bridge_ready_setter: Callable[[Callable[[bool], object] | None], None] | None = (
            None
        )
        self._surface_mask_enabled = False
        self._surface_mask_timer: QTimer | None = None
        self._surface_mask_region: object | None = None
        self._surface_mask_input_region: object | None = None
        self._surface_mask_view_visible: bool | None = None
        # QWebEngineView 的可见内容实际由 QQuickWidget 子表面提交；仅给
        # 顶层 QWidget 设置 Shape 时，子表面仍会以完整矩形拦截透明角点。
        self._surface_mask_targets: list[object] = []
        self._x11_input_shape_last_update = 0.0
        self._surface_mask_capture_pending = False
        self._surface_mask_capture_generation = 0
        self._surface_mask_capture_last_request = 0.0
        self._surface_mask_capture_failures = 0
        # ``None`` 表示平台没有独立 X11 Input Shape（Wayland/兼容替身），
        # ``False`` 才表示本次明确写入失败；状态卡不能把后者伪装成 ready。
        self._x11_input_shape_last_result: bool | None = None
        # aboutToQuit 先于 renderer/view 的最终销毁触发；单独的 closing
        # 墓碑让迟到的 WebChannel 回调在解码大块 PNG 之前立即返回，而不
        # 提前把 _shutdown 置真（后者会跳过真正的 renderer 清理）。
        self._closing = False
        # 顶层 WebEngine 原生子表面在 move() 期间可能先提交位置、再提交
        # Chromium 帧；此时重建 Qt mask 会把旧清屏色短暂合成到新位置。
        # 移动期间保留上一份 visual/input Shape，待窗口稳定后再单次刷新。
        self._surface_mask_move_active = False
        self._surface_mask_move_generation = 0
        self._surface_mask_move_timer: QTimer | None = None
        # 所有会改变顶层几何或 Shape 的操作共享一个单调代次。页面/平台
        # 回调是异步的，旧回调即使在用户开始拖动后抵达，也不得把旧区域或
        # 旧坐标写回当前窗口。
        self._geometry_generation = 0
        self._autonomous_geometry_until = 0.0
        self._autonomous_geometry_pending = 0
        # Qt 切换顶层窗口标志会重建 QWebEngine 原生子表面；在重建窗口
        # 的短窗口内暂停 Shape 定时器和异步 canvas 回读，避免旧 drawable
        # 继续向已销毁的 QRhi surface 提交请求。
        self._surface_mask_rebuild_pending = False
        self._surface_mask_rebuild_generation = 0
        # QWebEngineView.setWindowFlag 可能同步重建 Chromium 原生子表面。
        # 控制台按钮回调不能阻塞在该调用上；待处理请求在下一轮 Qt 事件
        # 循环执行，后续同线程请求会覆盖旧回调而不会同步执行它。
        self._window_flag_transition_generation = 0
        self._window_flag_transition_pending: bool | None = None
        self._window_flag_transition_apply: Callable[[], None] | None = None
        self._window_flag_transition_cleanup: Callable[[], None] | None = None
        self._topmost_capability_cache: tuple[str, str, str, float] | None = None
        # 保存最近一次置顶请求的可读回执；xcb 会在 QWindow.setFlag
        # 返回后异步提交 EWMH，调用方可在确认窗口管理器终态后回读。
        self._window_flag_transition_result: dict[str, object] | None = None
        self._topmost_confirmed: bool | None = None
        self._click_through_transition_generation = 0
        self._click_through_transition_pending: bool | None = None
        self._window_lock_transition_generation = 0
        self._window_lock_transition_pending: bool | None = None
        self._x11_compositor_state: bool | None = None
        # 点击穿透和没有 QWindow 的置顶回退会重建原生子表面。页面桥
        # 事件可能恰好在重建窗口抵达；暂存一小段有界事件序列，待新 viewport
        # 稳定后按原顺序回放，避免用户刚恢复点击就丢失拖动起点。
        self._interaction_rebuild_until = 0.0
        self._deferred_page_events: list[dict[str, object]] = []
        self._deferred_page_event_timer: QTimer | None = None
        self._draining_deferred_page_events = False
        self._viewport_sync_timer: QTimer | None = None
        self._last_viewport_size: tuple[int, int] | None = None
        self._resize_generation = 0
        # 拖动期间 Chromium/Qt 可能短暂报告中间尺寸。先只保留最新的
        # 逻辑尺寸，释放后由同一事务一次性提交，避免页面每次回调都重置
        # Live2D 基准缩放和动作姿态。
        self._deferred_page_resize: tuple[int, int] | None = None
        self._context_menu_callback: Callable[[object], None] | None = None
        self._double_click_callback: Callable[[], None] | None = None
        self._enter_callback: Callable[[], None] | None = None
        self._click_callback: Callable[[Mapping[str, object]], None] | None = None
        self._renderer_click_callback_active = False
        self._renderer_model_reload_callback_active = False
        self._model_reload_callback: Callable[[Mapping[str, object]], object] | None = None
        self._interaction_interrupt: Callable[[], object] | None = None
        self._activity_notifier: Callable[[], object] | None = None
        self._interaction_interrupt_notified = False
        self._shutdown = False

    def set_interaction_interrupt(self, callback: Callable[[], object] | None) -> None:
        """设置用户按下时立即使后台行为代际失效的同步回调。"""

        self._interaction_interrupt = callback

    def set_activity_notifier(self, callback: Callable[[], object] | None) -> None:
        """设置实际用户手势的活跃状态通知器。"""

        self._activity_notifier = callback

    def set_model_reload_callback(
        self,
        callback: Callable[[Mapping[str, object]], object] | None,
    ) -> None:
        """设置模型切换终态回执回调，供 Qt 控制台更新状态。"""

        self._model_reload_callback = callback
        # 模型切换回执与页面指针桥是两条独立能力链。即使旧 WebEngine
        # 无法安装完整交互过滤器，控制台仍应收到 reload 的终态；重复设置
        # 同一宿主回调是幂等的，不改变拖动路径。
        setter = getattr(self.renderer, "set_model_reload_callback", None)
        if not callable(setter):
            self._renderer_model_reload_callback_active = False
            return
        try:
            setter(self._on_renderer_model_reload if callback is not None else None)
            self._renderer_model_reload_callback_active = callback is not None
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._renderer_model_reload_callback_active = False

    def reload_model(self, model_path: object | None) -> object:
        """请求渲染器在当前 WebEngine 页面内原子切换模型。"""

        handler = getattr(self.renderer, "reload_model", None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "model reload is unavailable"}
        if self._drag_transaction_active() or self._renderer_dragging:
            return {"status": "busy", "reason": "model reload is blocked while dragging"}
        self._invalidate_geometry()
        try:
            result = handler(model_path)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Web host model reload request failed: %s", type(exc).__name__)
            return {"status": "failed", "reason": "model reload request failed"}
        if inspect.isawaitable(result):
            return result
        if isinstance(result, Mapping) and str(result.get("status", "")).lower() == "available":
            self._on_renderer_model_reload(dict(result))
        return result

    def reload_resources(
        self,
        resource_root: object,
        *,
        model_path: object | None = None,
        sprite_scale: float | None = None,
    ) -> object:
        """转发统一资源重载契约；拖动和关闭阶段严格拒绝。"""

        if self._closing or self._shutdown:
            return {"status": "unavailable", "reason": "renderer host is closed"}
        handler = getattr(self.renderer, "reload_resources", None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "resource reload is unavailable"}
        if self._drag_transaction_active() or self._renderer_dragging:
            return {"status": "busy", "reason": "resource reload is blocked while dragging"}
        try:
            result = handler(
                resource_root,
                model_path=model_path,
                sprite_scale=sprite_scale,
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Web host resource reload request failed: %s", type(exc).__name__)
            return {"status": "failed", "reason": "resource reload request failed"}
        if inspect.isawaitable(result):
            return result
        if isinstance(result, Mapping):
            status = str(result.get("status", "") or "").strip().lower()
            if status in {"pending", "available", "reloaded"}:
                self._invalidate_geometry()
            if status in {"available", "reloaded"}:
                self._on_renderer_model_reload(dict(result))
        return result

    def _on_renderer_model_reload(self, result: Mapping[str, object]) -> None:
        """模型交换成功后使旧 Shape/命中回读代次失效并安排刷新。"""

        status = str(result.get("status", "") or "").strip().lower()
        if status == "available" and not self._closing and not self._shutdown:
            self._invalidate_geometry()
            self._surface_mask_capture_generation += 1
            self._surface_mask_capture_pending = False
            self._surface_mask_capture_failures = 0
            self._last_cursor_target = None
            self._last_window_position = None
            self._deferred_page_resize = None
            self._schedule_interaction_refresh()
            if self._surface_mask_enabled:
                self._schedule_surface_mask_retries()
        callback = self._model_reload_callback
        # 关闭阶段不再触碰 Qt/Shape，但仍转发 renderer 的失败终态；否则
        # 控制台/调用方会永久保留 pending 状态。
        if callback is None or ((self._closing or self._shutdown) and status == "available"):
            return
        try:
            callback(dict(result))
        except Exception:
            logger.exception("Web host model reload callback failed")

    def _notify_interaction_started(self) -> None:
        """在一次用户手势开始时只通知一次行为服务。"""

        if self._interaction_interrupt_notified:
            return
        self._interaction_interrupt_notified = True
        callback = self._interaction_interrupt
        if callable(callback):
            try:
                result = callback()
                if inspect.isawaitable(result):
                    _safe_close_awaitable(result)
            except Exception:
                logger.debug("failed to interrupt behavior on page press", exc_info=True)
        notifier = self._activity_notifier
        if callable(notifier):
            try:
                result = notifier()
                if inspect.isawaitable(result):
                    _safe_close_awaitable(result)
            except Exception:
                logger.debug("failed to record Web user activity", exc_info=True)

    def _invalidate_geometry(self, *, cancel_capture: bool = True) -> int:
        """开启新的几何事务并使旧的异步回调失效。"""

        self._geometry_generation += 1
        if cancel_capture:
            self._surface_mask_capture_generation += 1
            self._surface_mask_capture_pending = False
        return self._geometry_generation

    def _geometry_is_current(self, generation: int | None) -> bool:
        """检查异步几何回调是否仍属于当前窗口事务。"""

        return (
            (generation is None or generation == self._geometry_generation)
            and not self._closing
            and not self._shutdown
        )

    def _drag_transaction_active(self) -> bool:
        """返回页面桥或原生过滤器是否仍持有用户拖动事务。"""

        interaction = self._interaction_filter
        return bool(
            self._page_drag_origin is not None
            or self._page_dragging
            or self._page_system_move_active
            or (
                interaction is not None
                and (
                    getattr(interaction, "_drag_origin", None) is not None
                    or bool(getattr(interaction, "_dragging", False))
                    or bool(getattr(interaction, "_system_move_active", False))
                )
            )
        )

    def _autonomous_geometry_active(self) -> bool:
        """返回自主步进后的 Shape 稳定窗口是否仍未结束。"""

        return (
            bool(self._autonomous_geometry_pending) or monotonic() < self._autonomous_geometry_until
        )

    def _begin_autonomous_geometry(self) -> int:
        """合并自主小步的 Shape 暂停，而不影响光标追踪状态。"""

        generation = self._invalidate_geometry()
        self._activate_autonomous_geometry()
        return generation

    def _activate_autonomous_geometry(self) -> None:
        """在自主移动协程真正开始后接管一次 pending 计数。"""

        self._autonomous_geometry_pending += 1
        self._autonomous_geometry_until = max(
            self._autonomous_geometry_until,
            monotonic() + _AUTONOMOUS_GEOMETRY_SETTLE_SECONDS,
        )

    def _end_autonomous_geometry(self) -> None:
        """结束一个自主位置请求并保留短暂的 Shape 稳定窗口。"""

        self._autonomous_geometry_pending = max(0, self._autonomous_geometry_pending - 1)
        if self._autonomous_geometry_pending == 0:
            self._autonomous_geometry_until = max(
                self._autonomous_geometry_until,
                monotonic() + _AUTONOMOUS_GEOMETRY_SETTLE_SECONDS,
            )

    @property
    def view(self) -> Any | None:
        return getattr(self.renderer, "view", None)

    def move_to(self, x: float, y: float, *, duration_ms: int = 800) -> object:
        view = self.view
        # ``duration_ms=0`` 是自主漫游的连续步进标记。步进期间保持
        # Shape 和注视时钟连续，避免每个坐标都触发一次重建/复位。
        try:
            autonomous_step = int(duration_ms) <= 0
        except (TypeError, ValueError, OverflowError):
            autonomous_step = False
        if self._drag_transaction_active():
            # 用户拖动拥有窗口几何的最高优先级；迟到的自主/模型移动
            # 不能覆盖当前指针目标，否则会出现回弹和视觉抽搐。
            return {"status": "cancelled", "reason": "user interaction active"}
        if autonomous_step and monotonic() < self._autonomous_block_until:
            return {"status": "cancelled", "reason": "user interaction settling"}
        # 记录本次几何事务；异步平台回执回来时必须确认代次仍然有效。
        move_generation = self._geometry_generation
        if autonomous_step:
            move_generation = self._begin_autonomous_geometry()
        else:
            move_generation = self._invalidate_geometry()
        # 移动期间暂停 Shape 回读，避免抓取旧位置的 Chromium 子表面后
        # 把默认清屏色提交到新位置；完成后由短防抖恢复当前帧。
        if not autonomous_step:
            self._begin_surface_mask_move()
            move_generation = self._geometry_generation
        # 位置请求一旦提交，旧局部坐标就不能继续抑制下一次同步；Wayland
        # 或异步平台适配器可能在稍后才真正改变顶层窗口位置。
        self._last_cursor_target = None
        self._last_window_position = None
        self._last_drag_target = None
        self._last_drag_notify_at = 0.0
        if not autonomous_step:
            notifier = getattr(self.renderer, "notify_window_moved", None)
            if callable(notifier):
                try:
                    notifier()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to reset Live2D tracking before move", exc_info=True)
        mover = getattr(self.platform, "move_overlay", None) if self.platform else None
        if view is not None and callable(mover):
            try:
                capability = mover(view, int(round(float(x))), int(round(float(y))))
                if inspect.isawaitable(capability):
                    # 仅在返回的协程真正开始执行时持有 pending；调用方若在
                    # 首次调度前取消/丢弃协程，不会把 Shape 永久锁在稳定窗口。
                    if autonomous_step:
                        self._end_autonomous_geometry()
                    operation = self._move_after_dispatch(
                        capability,
                        x,
                        y,
                        autonomous_step=autonomous_step,
                        geometry_generation=move_generation,
                        autonomous_geometry_deferred=autonomous_step,
                    )
                    _attach_awaitable_finalizer(operation, capability)
                    return operation
                state = getattr(getattr(capability, "state", None), "value", None)
                detail = getattr(capability, "detail", "")
                if isinstance(capability, Mapping):
                    value = capability.get("status", capability.get("state", "unavailable"))
                    state = getattr(value, "value", value)
                    detail = capability.get("detail", capability.get("reason", ""))
                result = {
                    "status": str(state or "unavailable"),
                    "x": int(round(float(x))),
                    "y": int(round(float(y))),
                    "detail": str(detail or ""),
                }
                if not autonomous_step:
                    self._schedule_surface_mask_after_move()
                if autonomous_step:
                    self._end_autonomous_geometry()
                return result
            except asyncio.CancelledError:
                if not autonomous_step:
                    self._schedule_surface_mask_after_move()
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                if not autonomous_step:
                    self._schedule_surface_mask_after_move()
                if autonomous_step:
                    self._end_autonomous_geometry()
                logger.debug("Web host move failed: %s", type(exc).__name__)
                return {
                    "status": "unavailable",
                    "reason": "桌宠位置暂时无法调整，请稍后重试",
                }
            except BaseException:
                if not autonomous_step:
                    self._schedule_surface_mask_after_move()
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
        handler = getattr(self.renderer, "move_to", None)
        if callable(handler):
            try:
                result = handler(x, y, duration_ms=duration_ms)
                if inspect.isawaitable(result):
                    # 保持同步返回 awaitable 的契约；真正的 pending 归属
                    # 延后到该 awaitable 首次执行，避免任务在调度前取消时
                    # 留下永久的几何稳定状态。
                    if autonomous_step:
                        self._end_autonomous_geometry()
                    operation = self._move_after_dispatch(
                        result,
                        x,
                        y,
                        autonomous_step=autonomous_step,
                        geometry_generation=move_generation,
                        autonomous_geometry_deferred=autonomous_step,
                    )
                    _attach_awaitable_finalizer(operation, result)
                    return operation
                if not autonomous_step:
                    self._schedule_surface_mask_after_move()
                if autonomous_step:
                    self._end_autonomous_geometry()
                return result
            except asyncio.CancelledError:
                if not autonomous_step:
                    self._schedule_surface_mask_after_move()
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                if not autonomous_step:
                    self._schedule_surface_mask_after_move()
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
            except BaseException:
                if not autonomous_step:
                    self._schedule_surface_mask_after_move()
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
        if not autonomous_step:
            self._schedule_surface_mask_after_move()
        if autonomous_step:
            self._end_autonomous_geometry()
        return {"status": "unavailable", "reason": "web renderer is not movable"}

    def position(self) -> object:
        """读取 WebEngine 顶层窗口位置。"""

        view = self.view
        getter = getattr(view, "pos", None) if view is not None else None
        if not callable(getter):
            return {"status": "unavailable", "reason": "web view position is unavailable"}
        try:
            return getter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "web view position is unavailable"}

    def movement_bounds(self) -> tuple[int, int, int, int] | None:
        """返回 WebEngine 窗口左上角可移动的屏幕范围。"""

        view = self.view
        if view is None:
            return None
        try:
            from PySide6.QtGui import QGuiApplication

            handle_getter = getattr(view, "windowHandle", None)
            handle = handle_getter() if callable(handle_getter) else None
            screen = getattr(handle, "screen", lambda: None)() if handle is not None else None
            if screen is None:
                position = self.position()
                screen = QGuiApplication.screenAt(position) if position is not None else None
            if screen is None:
                screen = QGuiApplication.primaryScreen()
            if screen is None:
                return None
            area = screen.availableGeometry()
            left = int(area.left())
            top = int(area.top())
            width = max(1, int(view.width()))
            height = max(1, int(view.height()))
            # 小屏或高 DPI 下窗口可能大于可用区域；仍返回一个可用的
            # 单点边界，让“移到中央”和自主行动把窗口放在屏幕左上角，
            # 而不是把 right/bottom 反转成“位置不可用”。
            right = max(left, int(area.right()) - width + 1)
            bottom = max(top, int(area.bottom()) - height + 1)
            return left, top, right, bottom
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def set_display_size(self, preset: str) -> dict[str, object]:
        """按小/标准/大预设调整 WebEngine 窗口，模型保持等比缩放。"""

        selected = display_size_preset(preset)
        if selected is None:
            return {"status": "unavailable", "reason": "display size preset is invalid"}
        view = self.view
        if view is None:
            return {"status": "unavailable", "reason": "web view is unavailable"}
        key, (width, height) = selected
        geometry_generation = self._invalidate_geometry()
        self._begin_surface_mask_move()
        geometry_generation = self._geometry_generation
        try:
            position = self._read_position(view)
            # 显示预设是逻辑尺寸；低分辨率/高 DPI 桌面可能无法容纳
            # 原始高度。先按当前屏幕可用区域裁剪，并暂时降低 Qt 的
            # 最小尺寸约束，否则 resize() 会被静默拒绝而页面仍按预设
            # 尺寸绘制，最终出现外框与 Live2D 视口不一致。
            bounds_before_resize = self.movement_bounds()
            if bounds_before_resize is not None:
                left, top, right, bottom = bounds_before_resize
                # ``movement_bounds`` 将过大窗口的右/下界压到左/上界，
                # 所以不能由 bounds 反推屏幕尺寸。直接读取窗口所属屏幕
                # 的 availableGeometry，缺少屏幕对象时才保留原预设。
                screen = None
                handle_getter = getattr(view, "windowHandle", None)
                handle = handle_getter() if callable(handle_getter) else None
                screen_getter = getattr(handle, "screen", None)
                if callable(screen_getter):
                    screen = screen_getter()
                if screen is None:
                    try:
                        from PySide6.QtGui import QGuiApplication

                        screen = QGuiApplication.primaryScreen()
                    except (ImportError, OSError, RuntimeError):
                        screen = None
                area = getattr(screen, "availableGeometry", lambda: None)() if screen else None
                screen_width = int(area.width()) if area is not None else right - left + 1
                screen_height = int(area.height()) if area is not None else bottom - top + 1
                width = min(int(width), max(1, screen_width))
                height = min(int(height), max(1, screen_height))
            min_width_getter = getattr(view, "minimumWidth", None)
            min_height_getter = getattr(view, "minimumHeight", None)
            min_width_setter = getattr(view, "setMinimumWidth", None)
            min_height_setter = getattr(view, "setMinimumHeight", None)
            if callable(min_width_getter) and callable(min_width_setter):
                if width < int(min_width_getter()):
                    min_width_setter(width)
            if callable(min_height_getter) and callable(min_height_setter):
                if height < int(min_height_getter()):
                    min_height_setter(height)
            view.resize(width, height)
            width_getter = getattr(view, "width", None)
            height_getter = getattr(view, "height", None)
            actual_width = max(1, int(width_getter())) if callable(width_getter) else int(width)
            actual_height = max(1, int(height_getter())) if callable(height_getter) else int(height)
            bounds = self.movement_bounds()
            if position is not None and bounds is not None:
                left, top, right, bottom = bounds
                position = (max(left, min(position[0], right)), max(top, min(position[1], bottom)))
            self._restore_position(
                view,
                position,
                geometry_generation=geometry_generation,
            )
            # 以 Qt 实际接受的外框尺寸同步页面，不能继续使用可能被
            # 最小尺寸或窗口管理器调整过的请求值。
            self._request_page_resize(actual_width, actual_height)
            self._schedule_interaction_refresh()
            self._schedule_surface_mask_after_move()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self._schedule_surface_mask_after_move()
            logger.debug("Web host display size update failed: %s", type(exc).__name__)
            return {
                "status": "unavailable",
                "reason": "桌宠大小暂时无法调整，请稍后重试",
            }
        return {
            "status": "available",
            "preset": key,
            "width": actual_width,
            "height": actual_height,
        }

    def _send_page_resize(self, width: int, height: int) -> bool:
        """向页面提交一次已归并的尺寸，并安排一轮布局后校正。"""

        view = self.view
        if view is None or not bool(getattr(self.renderer, "page_ready", True)):
            return False
        page_getter = getattr(view, "page", None)
        page = page_getter() if callable(page_getter) else None
        run_javascript = getattr(page, "runJavaScript", None)
        if not callable(run_javascript):
            return False
        width = max(1, int(width))
        height = max(1, int(height))
        explicit_call = f"window.meapetLive2D.resize({width},{height})"
        script = (
            "(window.meapetLive2D && typeof window.meapetLive2D.resize === 'function' "
            f"? {explicit_call} : window.dispatchEvent(new Event('resize')));\n"
            "/* window.meapetLive2D.resize() */"
        )
        geometry_generation = self._geometry_generation
        self._resize_generation += 1
        generation = self._resize_generation
        try:
            run_javascript(script)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False

        def refresh_after_layout() -> None:
            if (
                generation != self._resize_generation
                or not self._geometry_is_current(geometry_generation)
                or self._drag_transaction_active()
                or self.view is not view
            ):
                return
            if not bool(getattr(self.renderer, "page_ready", True)):
                return
            try:
                run_javascript(script)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return

        # QWebEngineView.resize() 先改变 QWidget 外框，再由 Chromium 更新
        # documentElement/clientWidth；第二次调用必须落在那一轮事件之后。
        try:
            QTimer.singleShot(70, refresh_after_layout)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        return True

    def _request_page_resize(self, width: int | None = None, height: int | None = None) -> bool:
        """在布局提交后同步页面；拖动期间只保留最新尺寸。"""

        view = self.view
        if view is None or not bool(getattr(self.renderer, "page_ready", True)):
            return False
        if width is None or height is None:
            try:
                width_getter = getattr(view, "width", None)
                height_getter = getattr(view, "height", None)
                width_value = width_getter() if callable(width_getter) else width_getter
                height_value = height_getter() if callable(height_getter) else height_getter
                width = int(width_value or 0)
                height = int(height_value or 0)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
        dimensions = (max(1, int(width)), max(1, int(height)))
        if self._drag_transaction_active() or self._renderer_dragging:
            # 使拖动开始前已经安排的布局回调失效；它们即使晚到也不能在
            # 指针仍按下时重置 Live2D 基准缩放。
            self._resize_generation += 1
            self._deferred_page_resize = dimensions
            return True
        sent = self._send_page_resize(*dimensions)
        if sent:
            # 直接请求已经提交了最新尺寸，清理释放拖动时遗留的旧待处理值。
            self._deferred_page_resize = None
        return sent

    def _flush_deferred_page_resize(self) -> None:
        """释放拖动后只提交一次最后尺寸。"""

        if self._drag_transaction_active() or self._renderer_dragging:
            return
        dimensions = self._deferred_page_resize
        if dimensions is None:
            return
        if self._send_page_resize(*dimensions):
            self._deferred_page_resize = None
        else:
            # 释放瞬间可能仍处于 QWebEngine 原生子表面重建窗口；
            # 保留待处理尺寸并清除去重快照，下一次视口定时器才能重试。
            self._last_viewport_size = None

    def _sync_viewport_size(self) -> None:
        """把外层 QWidget 尺寸及时同步到 Chromium 视口。"""

        view = self.view
        if view is None or self._shutdown:
            return
        try:
            width = int(view.width())
            height = int(view.height())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return
        if width <= 0 or height <= 0:
            return
        size = (width, height)
        # 页面加载/模型握手完成前不能执行 JavaScript。若此时先缓存尺寸，
        # 页面就绪后尺寸没有变化时会被去重逻辑永久跳过，导致 Chromium
        # 继续使用首个默认 viewport。保持“未同步”标记，下一轮定时器在
        # page_ready 后必然补发一次真实尺寸。
        if not bool(getattr(self.renderer, "page_ready", True)):
            self._last_viewport_size = None
            return
        if size == self._last_viewport_size:
            return
        if self._request_page_resize(width, height):
            self._last_viewport_size = size
        else:
            # runJavaScript 可能在 QWebEngine 原生子表面重建期间暂时不可用；
            # 不能把这次失败记成已同步，下一次定时器才能再次提交。
            self._last_viewport_size = None

    def _start_viewport_sync(self) -> None:
        """监听窗口管理器/用户拖动造成的尺寸变化。"""

        if not _qt_available or self.view is None or self._viewport_sync_timer is not None:
            return
        self._sync_viewport_size()
        try:
            timer = QTimer(self.view)
            timer.setInterval(50)
            timer.timeout.connect(self._sync_viewport_size)
            timer.start()
            self._viewport_sync_timer = timer
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            self._viewport_sync_timer = None

    def set_surface_mask_enabled(self, enabled: bool) -> dict[str, object]:
        """按 WebEngine alpha 设置精确可见/输入 Shape 掩码。"""

        view = self.view
        if self._closing or self._shutdown or view is None or not _qt_object_alive(view):
            return {"status": "unavailable", "enabled": False, "reason": "web view is unavailable"}
        self._invalidate_geometry()
        self._surface_mask_move_generation += 1
        self._surface_mask_move_active = False
        move_timer = self._surface_mask_move_timer
        if move_timer is not None:
            try:
                move_timer.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        # 正在等待的平台移动仍拥有自己的 pending；这里只清除旧稳定
        # 时间窗，不能直接归零，否则迟到的 finally 会误扣后续移动。
        self._autonomous_geometry_until = 0.0
        self._surface_mask_enabled = bool(enabled)
        try:
            visible_getter = getattr(view, "isVisible", None)
            visible = bool(visible_getter()) if callable(visible_getter) else None
            # 首次装配时窗口可能尚未 show；保留 ``None``，让页面首帧
            # 仍可建立 pending Shape。只有确认过一次可见后，后续 hide
            # 才视为隐藏态并暂停高频回读。
            if visible is True:
                self._surface_mask_view_visible = True
            elif self._surface_mask_view_visible is not True:
                self._surface_mask_view_visible = None
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            if self._surface_mask_view_visible is not True:
                self._surface_mask_view_visible = None
        if not self._surface_mask_enabled:
            self._surface_mask_rebuild_pending = False
            self._surface_mask_rebuild_generation += 1
            self._surface_mask_capture_generation += 1
            self._surface_mask_capture_pending = False
            self._surface_mask_capture_failures = 0
            timer = self._surface_mask_timer
            self._surface_mask_timer = None
            if timer is not None:
                timer.stop()
            self._surface_mask_region = None
            self._surface_mask_input_region = None
            self._set_x11_input_shapes(
                view,
                () if self._click_through else None,
                force=True,
            )
            if not self._clear_surface_masks(view):
                return {"status": "unavailable", "enabled": False}
            return {"status": "available", "enabled": False, "requested": False}
        applied = self._refresh_surface_mask()
        if not applied and self._surface_mask_region is None:
            # 模型首帧尚未可抓取时，先让透明顶层完全交还底层窗口；否则
            # QQuickWidget 会在 Shape 建立前以完整矩形拦截桌面点击。
            self._set_x11_input_shapes(view, (), force=True)
        if self._surface_mask_timer is None:
            timer = QTimer(view)
            # 动作/追踪会改变 alpha 包围区域；移动期间由 move 防抖负责，
            # walk/wave/blink 使用 80ms 有界刷新，避免和原生子表面竞争。
            timer.setInterval(self._surface_mask_interval_ms())
            timer.timeout.connect(self._refresh_surface_mask)
            self._surface_mask_timer = timer
            timer.start()
            self._schedule_surface_mask_retries()
        state = self.surface_mask_status()
        state["requested"] = True
        # 首帧 alpha 通过 WebChannel 异步回读；调用方不能把“尚未回读”
        # 当成永久不可用，否则控制台会错误提示用户关闭精确输入掩码。
        if not applied and state["status"] == "unavailable":
            state["status"] = "pending" if callable(getattr(view, "page", None)) else "unavailable"
        return state

    def surface_mask_status(self) -> dict[str, object]:
        """回读当前 Shape 状态，区分已应用、等待首帧和不可用。"""

        view = self.view
        enabled = bool(self._surface_mask_enabled)
        view_alive = _qt_object_alive(view)
        visual_ready = self._surface_mask_region is not None
        if visual_ready:
            empty_checker = getattr(self._surface_mask_region, "isEmpty", None)
            if callable(empty_checker):
                try:
                    visual_ready = not bool(empty_checker())
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    visual_ready = False
        # 点击穿透、窗口锁定和原生子表面重建期间，X11 Input Shape 已被
        # 主动清空；即使上一帧区域对象仍保留用于恢复，也不能把它报告为
        # 当前可点击。恢复定时器完成后再由同一判定重新报告 ready。
        input_ready = (
            self._surface_mask_input_region is not None
            and not self._click_through
            and not self._window_locked
            and not self._surface_mask_rebuild_pending
            and self._x11_input_shape_last_result is not False
        )
        capture_pending = bool(self._surface_mask_capture_pending)
        if not enabled:
            status = "available" if view_alive else "unavailable"
        elif visual_ready and input_ready:
            status = "available"
        elif visual_ready and self._x11_input_shape_last_result is False:
            status = "degraded"
        elif capture_pending:
            status = "pending"
        elif not view_alive:
            status = "unavailable"
        else:
            status = "pending"
        return {
            "status": status,
            "enabled": bool(visual_ready and enabled),
            "ready": visual_ready,
            "input_ready": bool(input_ready),
            "capture_pending": capture_pending,
        }

    def _surface_mask_widgets(self, view: object) -> tuple[object, ...]:
        """返回顶层视图和覆盖完整视口的原生子表面。"""

        targets: list[object] = [view]
        finder = getattr(view, "findChildren", None)
        width_getter = getattr(view, "width", None)
        height_getter = getattr(view, "height", None)
        view_width = int(width_getter()) if callable(width_getter) else 0
        view_height = int(height_getter()) if callable(height_getter) else 0
        if not callable(finder):
            return tuple(targets)
        try:
            children = tuple(finder(QObject))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return tuple(targets)
        for child in children:
            setter = getattr(child, "setMask", None)
            geometry_getter = getattr(child, "geometry", None)
            if not callable(setter) or not callable(geometry_getter):
                continue
            try:
                geometry = geometry_getter()
                if int(geometry.width()) >= max(1, view_width - 1) and int(
                    geometry.height()
                ) >= max(1, view_height - 1):
                    targets.append(child)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return tuple(targets)

    def _clear_surface_masks(self, view: object) -> bool:
        """清除顶层和原生子表面的旧 Shape。"""

        candidates = (*self._surface_mask_targets, *self._surface_mask_widgets(view))
        targets: list[object] = []
        for candidate in candidates:
            if not any(existing is candidate for existing in targets):
                targets.append(candidate)
        ok = True
        for target in targets:
            clearer = getattr(target, "clearMask", None)
            if not callable(clearer):
                continue
            try:
                clearer()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                ok = False
        self._surface_mask_targets = []
        return ok

    def _set_x11_input_shape(self, target: object, region: object | None) -> bool:
        """在 X11 上直接设置原生窗口的 Input Shape。

        QWebEngine 的 QQuickWidget 是独立原生子窗口，Qt 的 QWidget Shape
        有时只约束可见区域而不约束子窗口输入；这里补写 X11 SHAPE 的
        Input kind。Wayland 或缺少 python-xlib 时保持 Qt 路径，不伪造成功。
        """

        if str(os.environ.get("QT_QPA_PLATFORM", "") or "").strip().lower() == "wayland":
            return False
        display_name = str(os.environ.get("DISPLAY", "") or "").strip()
        if not display_name:
            return False
        try:
            from Xlib import display, error
            from Xlib.ext import shape

            win_id = int(target.winId())  # type: ignore[attr-defined]
            scale_getter = getattr(target, "devicePixelRatioF", None)
            if not callable(scale_getter):
                scale_getter = getattr(target, "devicePixelRatio", None)
            try:
                device_scale = float(scale_getter()) if callable(scale_getter) else 1.0
            except (TypeError, ValueError):
                device_scale = 1.0
            device_scale = max(1.0, device_scale)
            connection = display.Display(display_name)
            try:
                # Qt 切换窗口标志时会异步销毁/重建 QWebEngine 原生子表面。
                # 旧窗口在 Shape 请求排队期间可能已经失效；Python-Xlib 的
                # 默认错误处理器会把 BadWindow 直接打印到 stderr，污染用户
                # 日志并掩盖真正的渲染结果。仅对本次连接捕获该竞态，其他
                # X11 错误仍按失败路径处理。
                x11_errors = error.CatchError(error.BadWindow)
                connection.set_error_handler(x11_errors)
                window = connection.create_resource_object("window", win_id)
                if region is None:
                    width_getter = getattr(target, "width", None)
                    height_getter = getattr(target, "height", None)
                    rectangles = [
                        (
                            0,
                            0,
                            max(1, round(int(width_getter()) * device_scale))
                            if callable(width_getter)
                            else 1,
                            max(1, round(int(height_getter()) * device_scale))
                            if callable(height_getter)
                            else 1,
                        )
                    ]
                else:
                    rectangles = [
                        (
                            round(int(rect.x()) * device_scale),
                            round(int(rect.y()) * device_scale),
                            max(1, round(int(rect.width()) * device_scale)),
                            max(1, round(int(rect.height()) * device_scale)),
                        )
                        for rect in region
                        if int(rect.width()) > 0 and int(rect.height()) > 0
                    ]
                    if not rectangles:
                        # X11 SHAPE 的空矩形列表表示“恢复默认输入区域”，
                        # 不是“禁止输入”；放置一个窗口外的 1px 区域才是真正
                        # 的空 Input Shape。
                        rectangles = [(-1, -1, 1, 1)]
                # python-xlib 的 Rectangles 请求参数依次是 operation、kind、
                # ordering、x_offset、y_offset、rectangles；两个偏移量都必须
                # 保留，否则会在 Xlib 层直接报缺少参数。
                window.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, rectangles)
                connection.sync()
                if x11_errors.get_error() is not None:
                    logger.debug("X11 input Shape target disappeared during update")
                    return False
                return True
            finally:
                connection.close()
        except Exception:
            # Xlib 在 DISPLAY/Xvfb 提前退出时会抛出专用的
            # DisplayConnectionError/ConnectionClosedError；关闭竞态不应
            # 把 traceback 泄露到用户日志，也不能阻断 Qt 退出。
            logger.debug("X11 input Shape is unavailable")
            return False

    def _set_x11_input_shapes(
        self, view: object, region: object | None, *, force: bool = False
    ) -> bool:
        """把同一 Input Shape 写入顶层和覆盖完整视口的子表面。"""

        if not self._x11_input_shape_supported():
            self._x11_input_shape_last_result = None
            return False

        empty_region = region is None or not callable(getattr(region, "__iter__", None))
        if (
            not force
            and not empty_region
            and monotonic() - self._x11_input_shape_last_update < 0.08
        ):
            return True
        if not force and region == () and monotonic() - self._x11_input_shape_last_update < 0.08:
            return True
        applied = False
        for target in self._surface_mask_widgets(view):
            local_region = region
            if (
                region is not None
                and target is not view
                and callable(getattr(region, "translated", None))
            ):
                try:
                    geometry = target.geometry()
                    local_region = region.translated(-int(geometry.x()), -int(geometry.y()))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    continue
            applied = self._set_x11_input_shape(target, local_region) or applied
        if applied:
            self._x11_input_shape_last_update = monotonic()
        self._x11_input_shape_last_result = applied
        return applied

    def _apply_surface_masks(
        self, view: object, visual_region: object, input_region: object | None = None
    ) -> bool:
        """分别应用可见 Shape 与 X11 输入 Shape。"""

        targets = self._surface_mask_widgets(view)
        applied: list[object] = []
        for target in targets:
            setter = getattr(target, "setMask", None)
            if not callable(setter):
                continue
            local_region = visual_region
            if target is not view and callable(getattr(visual_region, "translated", None)):
                try:
                    geometry = target.geometry()
                    local_region = visual_region.translated(-int(geometry.x()), -int(geometry.y()))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    continue
            try:
                setter(local_region)
                applied.append(target)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
        if not applied:
            return False
        force_input_shape = self._surface_mask_region is None or not self._surface_mask_targets
        self._surface_mask_targets = applied
        effective_input = visual_region if input_region is None else input_region
        self._set_x11_input_shapes(
            view,
            effective_input,
            force=force_input_shape,
        )
        self._surface_mask_input_region = effective_input
        return True

    def _surface_mask_interval_ms(self) -> int:
        """动作期间提高 Shape 刷新频率，降低发梢/脚部被旧掩码裁掉的窗口。"""

        state = getattr(self.renderer, "state", None)
        motion = str(getattr(state, "motion", "") or "").strip().lower()
        if motion in {"walk", "wave", "blink"}:
            return _SURFACE_MASK_MOTION_INTERVAL_MS
        return _SURFACE_MASK_REFRESH_INTERVAL_MS

    def _begin_surface_mask_move(self) -> None:
        """暂停移动期间的原生 Shape 重建，避免 Chromium 子表面旧帧泄漏。"""

        if not self._surface_mask_enabled or self._shutdown:
            return
        # 高频拖动会反复进入此方法；同一段拖动保持一个 generation，
        # 由下面的单次 QTimer 延后到最后一次 move 后再恢复 Shape。
        if not self._surface_mask_move_active:
            self._invalidate_geometry()
            self._surface_mask_move_generation += 1
            self._surface_mask_move_active = True
        timer = self._surface_mask_move_timer
        if timer is None:
            view = self.view
            try:
                if _qt_available and view is not None:
                    timer = QTimer(view)
                    timer.setSingleShot(True)
                    timer.timeout.connect(
                        lambda: self._finish_surface_mask_move(self._surface_mask_move_generation)
                    )
                    self._surface_mask_move_timer = timer
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                timer = None

    def _finish_surface_mask_move(self, generation: int) -> None:
        """在移动提交稳定后恢复一次 Shape 读取。"""

        if generation != self._surface_mask_move_generation or self._shutdown:
            return
        # 已经进入新的用户拖动时，旧的 programmatic/move 防抖回调不能在
        # 指针仍按下期间恢复 Shape；释放路径会用当前代次再次安排。
        if self._drag_transaction_active():
            return
        self._surface_mask_move_active = False
        self._invalidate_geometry()
        timer = self._surface_mask_move_timer
        if timer is not None:
            try:
                timer.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._refresh_surface_mask()

    def _schedule_surface_mask_after_move(self) -> None:
        """以短防抖延迟结束移动暂停，合并连续自主步进。"""

        if not self._surface_mask_enabled or self._shutdown:
            return
        # 页面桥的 move 事件可能比 Qt 主线程快；持续拖动期间不能恢复
        # Shape。否则 36ms 防抖定时器会在指针仍按下时触发 canvas 回读，
        # 与窗口几何提交/Chromium 重绘交错，造成黑帧和模型抽搐。释放或
        # 取消路径会先清理拖动状态，再调用本方法。
        if self._drag_transaction_active():
            return
        generation = self._surface_mask_move_generation
        # 行走步进默认约 80ms；延迟必须短于下一步，否则连续移动会让
        # visual Shape 永久停留在旧帧并在无合成器环境显示黑色残留。
        delay_ms = 36
        try:
            timer = self._surface_mask_move_timer
            if timer is not None and QCoreApplication.instance() is not None:
                timer.start(delay_ms)
                return
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        try:
            if _qt_available and QCoreApplication.instance() is not None:
                QTimer.singleShot(delay_ms, lambda: self._finish_surface_mask_move(generation))
                return
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        self._finish_surface_mask_move(generation)

    def _pause_surface_mask_for_window_rebuild(
        self, view: object, *, defer_input_shape: bool = False
    ) -> bool:
        """暂停窗口标志切换期间的 Shape 回读，避免访问旧原生子表面。"""

        if not self._surface_mask_enabled or self._shutdown:
            return False
        self._invalidate_geometry()
        self._surface_mask_rebuild_pending = True
        self._surface_mask_rebuild_generation += 1
        self._surface_mask_capture_generation += 1
        self._surface_mask_capture_pending = False
        self._surface_mask_capture_failures = 0
        timer = self._surface_mask_timer
        if timer is not None:
            try:
                timer.stop()
            except (AttributeError, RuntimeError):
                pass
        # 先清空 X11 Input Shape；窗口重建后由延迟回读重新提交当前帧，
        # 不能让旧子窗口在重建期间继续拦截桌面点击。
        if defer_input_shape and _qt_available and QCoreApplication.instance() is not None:
            try:
                QTimer.singleShot(
                    0,
                    lambda: (
                        self._set_x11_input_shapes(view, (), force=True)
                        if self.view is view and not self._closing and not self._shutdown
                        else None
                    ),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self._set_x11_input_shapes(view, (), force=True)
        else:
            self._set_x11_input_shapes(view, (), force=True)
        return True

    def _mark_interaction_rebuild(self, delay_seconds: float = 0.2) -> None:
        """标记原生子表面重建窗口，并安排页面事件回放。"""

        self._invalidate_geometry()
        self._interaction_rebuild_until = max(
            self._interaction_rebuild_until,
            monotonic() + max(0.05, min(float(delay_seconds), 1.0)),
        )
        self._schedule_deferred_page_events()

    def _schedule_deferred_page_events(self) -> None:
        if not self._deferred_page_events or self._shutdown or self._closing:
            return
        try:
            view = self.view
            if _qt_available and view is not None and QCoreApplication.instance() is not None:
                remaining = max(0.01, self._interaction_rebuild_until - monotonic())
                delay_ms = max(1, int(math.ceil(remaining * 1000.0)))
                timer = self._deferred_page_event_timer
                if timer is None:
                    timer = QTimer(view)
                    timer.setSingleShot(True)
                    try:
                        timer.setTimerType(Qt.TimerType.PreciseTimer)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                    timer.timeout.connect(self._drain_deferred_page_events)
                    self._deferred_page_event_timer = timer
                timer.start(delay_ms)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass

    def _drain_deferred_page_events(self) -> None:
        """在 viewport 重建稳定后按顺序重放暂存页面事件。"""

        if self._shutdown or self._closing or not self._deferred_page_events:
            return
        remaining = self._interaction_rebuild_until - monotonic()
        if remaining > 0:
            self._schedule_deferred_page_events()
            return
        events = tuple(self._deferred_page_events)
        self._deferred_page_events.clear()
        self._draining_deferred_page_events = True
        try:
            for payload in events:
                self._handle_page_interaction(payload)
        finally:
            self._draining_deferred_page_events = False

    def _resume_surface_mask_after_window_rebuild(self, view: object, paused: bool) -> None:
        """在 QWebEngine 子表面稳定后恢复 Shape 定时刷新。"""

        if not paused:
            return
        if (
            not self._surface_mask_enabled
            or self._shutdown
            or self.view is not view
            or not _qt_object_alive(view)
        ):
            self._surface_mask_rebuild_pending = False
            return
        generation = self._surface_mask_rebuild_generation

        def resume() -> None:
            if (
                generation != self._surface_mask_rebuild_generation
                or self._shutdown
                or self.view is not view
                or not _qt_object_alive(view)
                or not self._surface_mask_enabled
            ):
                return
            self._surface_mask_rebuild_pending = False
            timer = self._surface_mask_timer
            if timer is not None:
                try:
                    timer.setInterval(self._surface_mask_interval_ms())
                    timer.start()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            self._refresh_surface_mask()
            # 某些 Qt/Chromium 版本在第一次重建后仍会丢弃一帧；补一组
            # 有界重试，确保异步 inputMask 回读完成后用户可以重新点击。
            self._schedule_surface_mask_retries()

        # 延迟到原生子表面完成重建后再请求 canvas；取消旧代次的回调，
        # 避免连续置顶/穿透操作让多个 timer 同时访问 discarded drawable。
        if not _qt_available:
            self._surface_mask_rebuild_pending = False
            return
        try:
            if QCoreApplication.instance() is None:
                # 无事件循环的契约测试/无头核心不能可靠地执行 singleShot；
                # 立即清除 pending，避免 API 状态永久卡住。
                resume()
                return
            QTimer.singleShot(320, resume)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # QTimer 在 teardown 竞态下可能拒绝注册；状态必须收敛，且
            # 不再访问已经销毁的 WebEngine 对象。
            self._surface_mask_rebuild_pending = False

    def _schedule_surface_mask_retries(self) -> None:
        """在 QQuickWidget 首次创建/模型首帧到达后重试 Shape。"""

        if (
            not _qt_available
            or QCoreApplication.instance() is None
            or not self._surface_mask_enabled
            or self._surface_mask_rebuild_pending
            or self.view is None
        ):
            return
        generation = self._surface_mask_rebuild_generation

        def retry() -> None:
            if generation != self._surface_mask_rebuild_generation:
                return
            if self._surface_mask_rebuild_pending or not self._surface_mask_enabled:
                return
            self._refresh_surface_mask()

        for delay in (0, 50, 120, 240, 500, 900):
            try:
                QTimer.singleShot(delay, retry)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return

    def _request_surface_mask_capture(self, view: object) -> bool | None:
        """请求页面返回只包含 Live2D canvas 的 alpha 图。

        ``view.grab()`` 会把气泡、反馈牌和点击 marker 一并纳入 alpha；
        这些元素虽然在 DOM 中 ``pointer-events:none``，但 Qt/X11 Shape
        仍会把它们当作输入区域。页面 canvas 数据路径能从根源上排除
        非交互 overlay；不支持该入口的测试替身才回退到普通抓图。
        """

        if (
            self._closing
            or self._shutdown
            or self._surface_mask_move_active
            or self._drag_transaction_active()
            or self._autonomous_geometry_active()
            or not _qt_object_alive(view)
        ):
            return None
        if not bool(getattr(self.renderer, "page_ready", True)):
            return None
        page_getter = getattr(view, "page", None)
        page = page_getter() if callable(page_getter) else None
        run_javascript = getattr(page, "runJavaScript", None)
        if not callable(run_javascript):
            return None
        now = monotonic()
        if self._surface_mask_capture_pending:
            if now - self._surface_mask_capture_last_request < 0.5:
                return True
            # 页面 discarded/IPC 丢包时不能永久卡在 pending；让旧回调
            # 失效并重新请求，首帧没有输入区时继续 fail-open。
            self._surface_mask_capture_pending = False
            self._surface_mask_capture_generation += 1
            if self._surface_mask_region is None:
                self._set_x11_input_shapes(view, (), force=True)
        if now - self._surface_mask_capture_last_request < _SURFACE_MASK_CAPTURE_INTERVAL_SECONDS:
            return True
        self._surface_mask_capture_pending = True
        self._surface_mask_capture_last_request = now
        self._surface_mask_capture_generation += 1
        generation = self._surface_mask_capture_generation
        geometry_generation = self._geometry_generation
        # 保留上一份 visual mask 直到异步页面回读完成。提前清除会让
        # QQuickWidget 在几十毫秒内退回矩形 surface，产生黑面/闪烁；
        # 页面回读同时携带 overlay 矩形，回调到达后一次性提交新区域。

        def receive(value: object) -> None:
            self._finish_surface_mask_capture(view, generation, value, geometry_generation)

        script = (
            "JSON.stringify(window.meapetLive2D && "
            "typeof window.meapetLive2D.inputMask === 'function' "
            "? window.meapetLive2D.inputMask() : {status:'unavailable'})"
        )
        try:
            run_javascript(script, receive)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            self._surface_mask_capture_pending = False
            return None
        return True

    def _finish_surface_mask_capture(
        self,
        view: object,
        generation: int,
        value: object,
        geometry_generation: int | None = None,
    ) -> None:
        """解码页面 canvas alpha 并提交新的可见/输入 Shape。"""

        if self._closing or self._shutdown or generation != self._surface_mask_capture_generation:
            return
        self._surface_mask_capture_pending = False
        # canvas 回读跨越 Chromium/Qt 消息队列；拖动、窗口标志切换或
        # 其它几何事务开始后，旧截图不能再覆盖当前 Shape。保留上一份
        # 有效区域，待当前事务结束后由定时器重新请求。
        if (
            not self._geometry_is_current(geometry_generation)
            or self._surface_mask_move_active
            or self._drag_transaction_active()
            or self._autonomous_geometry_active()
        ):
            return
        if (
            not self._surface_mask_enabled
            or self._shutdown
            or self.view is not view
            or not _qt_object_alive(view)
        ):
            return
        visual_rects: object = ()
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except (TypeError, ValueError):
                decoded = value
            value = decoded
        if isinstance(value, Mapping):
            status = str(value.get("status", "") or "").strip().lower()
            if status == "hidden":
                self._surface_mask_capture_failures = 0
                self._surface_mask_region = None
                self._surface_mask_input_region = None
                self._clear_surface_masks(view)
                self._set_x11_input_shapes(view, (), force=True)
                return
            visual_rects = value.get("visual_rects", ())
            value = value.get("data")
        image = None
        if isinstance(value, str) and "," in value:
            encoded = value.split(",", 1)[1]
            # 页面回调属于不可信输入；在 base64 解码前限制大小，避免
            # 退出阶段或异常页面返回超大字符串阻塞 Qt 主线程。
            if len(encoded) > _MAX_SURFACE_MASK_DATA_BYTES:
                self._surface_mask_capture_failures += 1
                return
            try:
                from PySide6.QtGui import QImage

                image = QImage.fromData(base64.b64decode(encoded, validate=True))
            except (binascii.Error, ImportError, OSError, RuntimeError, TypeError, ValueError):
                image = None
        if image is not None and not image.isNull():
            normalized = self._region_from_image(view, image)
            if normalized is not None:
                _model_image, input_region = normalized
                visual_region = self._visual_region_for_page_capture(
                    view, input_region, visual_rects
                )
                if self._commit_surface_mask_regions(view, visual_region, input_region):
                    self._surface_mask_image = _model_image
                    self._surface_mask_capture_failures = 0
                    return
        self._surface_mask_capture_failures += 1
        if self._surface_mask_capture_failures >= 3:
            self._surface_mask_capture_failures = 0
            self._fallback_surface_mask_capture(view)
        # 模型尚未绘制或 WebGL 暂时不能读回时保留上一份有效 Shape；
        # 下一次定时器/重试会再次请求 canvas，而不是把 overlay 抓图误当输入区。
        if self._surface_mask_enabled and self._surface_mask_region is None:
            try:
                QTimer.singleShot(50, self._refresh_surface_mask)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    def _visual_region_for_page_capture(
        self, view: object, fallback: object, visual_rects: object = ()
    ) -> object:
        """为 canvas 输入图选择不裁剪气泡的可见区域。"""

        if self._x11_input_shape_supported() and self._x11_compositor_available():
            try:
                from PySide6.QtCore import QRect
                from PySide6.QtGui import QRegion

                return QRegion(QRect(0, 0, max(1, int(view.width())), max(1, int(view.height()))))
            except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
                return fallback
        if self._x11_input_shape_supported():
            # 无 X11 合成器时，view.grab() 的 native 子表面可能返回整块
            # 不透明黑色；不能再用它生成视觉 Shape。模型 alpha 加上已知
            # overlay 矩形足以保留可见内容并裁掉黑面。
            overlay_region = self._overlay_visual_region(view, visual_rects)
            if overlay_region is None:
                overlay_region = self._known_overlay_visual_region(view)
            if overlay_region is not None:
                try:
                    return fallback.united(overlay_region)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            return fallback
        overlay_region = self._overlay_visual_region(view, visual_rects)
        if overlay_region is not None:
            try:
                return fallback.united(overlay_region)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        try:
            image = view.grab().toImage()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return fallback
        normalized = self._region_from_image(view, image)
        return normalized[1] if normalized is not None else fallback

    def _known_overlay_visual_region(self, view: object) -> object | None:
        """在无合成器回退时按宿主状态保留气泡/反馈的可见区域。"""

        try:
            from PySide6.QtCore import QRect
            from PySide6.QtGui import QRegion

            width = max(1, int(view.width()))
            height = max(1, int(view.height()))
            region = QRegion()
            if bool(getattr(self.renderer, "_speech_visible", False)) and str(
                getattr(self.renderer, "_speech_text", "") or ""
            ):
                region = region.united(QRegion(QRect(8, 0, max(1, width - 16), min(56, height))))
            if bool(getattr(self.renderer, "_interaction_feedback_visible", False)):
                toast_width = min(96, max(1, width - 10))
                toast_height = min(112, max(1, height - 20))
                region = region.united(
                    QRegion(
                        QRect(
                            max(0, width - toast_width - 8),
                            max(0, (height - toast_height) // 2 - 4),
                            toast_width + 12,
                            toast_height + 8,
                        )
                    )
                )
            return region if not region.isEmpty() else None
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _overlay_visual_region(view: object, value: object) -> object | None:
        """把页面返回的展示层矩形转换为逻辑像素区域。"""

        if isinstance(value, (str, bytes, bytearray, Mapping)):
            return None
        try:
            from PySide6.QtCore import QRect
            from PySide6.QtGui import QRegion

            region = QRegion()
            width = max(1, int(view.width()))
            height = max(1, int(view.height()))
            for item in value or ():
                if not isinstance(item, Mapping):
                    continue
                x = max(0, min(width, round(float(item.get("x", 0) or 0))))
                y = max(0, min(height, round(float(item.get("y", 0) or 0))))
                right = max(x, min(width, round(x + float(item.get("width", 0) or 0))))
                bottom = max(y, min(height, round(y + float(item.get("height", 0) or 0))))
                if right > x and bottom > y:
                    region = region.united(QRegion(QRect(x, y, right - x, bottom - y)))
            return region if not region.isEmpty() else None
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def _x11_compositor_available(self) -> bool:
        """判断 X11 是否有合成器；无合成器时视觉 Shape 必须收窄以裁掉黑面。"""

        if not self._x11_input_shape_supported():
            return False
        if self._x11_compositor_state is not None:
            return self._x11_compositor_state
        try:
            from gui.platforms.linux import probe_x11_compositor

            value = probe_x11_compositor()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError):
            value = None
        # 无法确认时选择保守的 alpha 视觉区域，避免把未知环境的黑色
        # 原生子表面提交到屏幕；这只增加一次低频 grab，不改变 Input Shape。
        self._x11_compositor_state = value is True
        return self._x11_compositor_state

    def _fallback_surface_mask_capture(self, view: object) -> None:
        """在页面 canvas 读回连续失败时使用受控的 Qt 抓图回退。"""

        if (
            self._closing
            or self._shutdown
            or self._surface_mask_move_active
            or self._autonomous_geometry_active()
        ):
            return
        if self._drag_transaction_active():
            return
        try:
            image = view.grab().toImage()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return
        if image.isNull():
            return
        normalized = self._region_from_image(view, image)
        if normalized is None:
            return
        image, visual_region = normalized
        previous_input = self._surface_mask_input_region
        if (
            self._x11_input_shape_supported()
            and previous_input is not None
            and callable(getattr(previous_input, "isEmpty", None))
            and not previous_input.isEmpty()
        ):
            self._surface_mask_image = image
            self._commit_surface_mask_regions(view, visual_region, previous_input)
            return
        self._commit_surface_mask_image(view, image, exclude_overlays=True)

    @staticmethod
    def _x11_input_shape_supported() -> bool:
        """返回当前进程是否可以把输入 Shape 与可见 Shape 分离。"""

        if (
            not os.environ.get("DISPLAY")
            or str(os.environ.get("QT_QPA_PLATFORM", "") or "").strip().lower() == "wayland"
        ):
            return False
        try:
            from Xlib import display  # noqa: F401
            from Xlib.ext import shape  # noqa: F401

            return bool(getattr(shape, "SK", None) and getattr(shape, "SO", None))
        except (ImportError, OSError, RuntimeError):
            return False

    def _region_from_image(self, view: object, image: object) -> tuple[object, object] | None:
        """把截图归一化到 Qt 逻辑像素并返回图像/alpha 区域。"""

        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QBitmap, QRegion

            width_getter = getattr(view, "width", None)
            height_getter = getattr(view, "height", None)
            logical_width = max(1, int(width_getter())) if callable(width_getter) else image.width()
            logical_height = (
                max(1, int(height_getter())) if callable(height_getter) else image.height()
            )
            if image.width() != logical_width or image.height() != logical_height:
                image = image.scaled(
                    logical_width,
                    logical_height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            image.setDevicePixelRatio(1.0)
            return image, QRegion(QBitmap.fromImage(image.createAlphaMask()))
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def _commit_surface_mask_regions(
        self, view: object, visual_region: object, input_region: object
    ) -> bool:
        """提交可见区域和独立输入区域。"""

        if (
            self._surface_mask_move_active
            or self._drag_transaction_active()
            or self._autonomous_geometry_active()
        ):
            return False
        if callable(getattr(visual_region, "isEmpty", None)) and visual_region.isEmpty():
            return False
        if not self._apply_surface_masks(view, visual_region, input_region):
            return False
        self._surface_mask_region = visual_region
        self._surface_mask_input_region = input_region
        return True

    def _commit_surface_mask_image(
        self, view: object, image: object, *, exclude_overlays: bool = False
    ) -> bool:
        """把一张已选定来源的 alpha 图转换为 Qt/X11 Shape。"""

        normalized = self._region_from_image(view, image)
        if normalized is None:
            return False
        image, visual_region = normalized
        input_region = (
            self._exclude_noninteractive_overlays(view, visual_region)
            if exclude_overlays
            else visual_region
        )
        # X11 可把可见气泡保留在 Qt mask 中，再用 SHAPE Input 只接收模型；
        # Wayland 没有该分离入口，只能让 Qt 的可见 Shape 同时承担输入。
        if not self._x11_input_shape_supported():
            input_region = visual_region
        self._surface_mask_image = image
        return self._commit_surface_mask_regions(view, visual_region, input_region)

    def _exclude_noninteractive_overlays(self, view: object, region: object) -> object:
        """从抓图回退区域中移除已知的非交互 DOM overlay。"""

        try:
            from PySide6.QtCore import QRect
            from PySide6.QtGui import QRegion

            width = max(1, int(view.width()))
            height = max(1, int(view.height()))
            exclusions = QRegion()
            speech_visible = bool(getattr(self.renderer, "_speech_visible", False))
            speech_text = str(getattr(self.renderer, "_speech_text", "") or "")
            if speech_visible and speech_text:
                exclusions = exclusions.united(
                    QRegion(QRect(8, 0, max(1, width - 16), min(56, height)))
                )
            feedback_visible = bool(getattr(self.renderer, "_interaction_feedback_visible", False))
            if feedback_visible:
                toast_width = min(96, max(1, width - 10))
                toast_height = min(112, max(1, height - 20))
                exclusions = exclusions.united(
                    QRegion(
                        QRect(
                            max(0, width - toast_width - 8),
                            max(0, (height - toast_height) // 2 - 4),
                            toast_width + 12,
                            toast_height + 8,
                        )
                    )
                )
            return region.subtracted(exclusions)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return region

    def _refresh_surface_mask(self) -> bool:
        """按当前 WebEngine 合成帧的 alpha 更新窗口 Shape。"""

        if (
            self._closing
            or self._shutdown
            or not self._surface_mask_enabled
            or self._surface_mask_rebuild_pending
        ):
            return False
        if (
            self._surface_mask_move_active
            or self._drag_transaction_active()
            or self._autonomous_geometry_active()
        ):
            # 移动期间保留已提交的区域；清空或重建 Shape 会让原生
            # QQuickWidget 在新位置短暂显示默认清屏色。
            return bool(self._surface_mask_region is not None)
        view = self.view
        if view is None or not _qt_object_alive(view):
            return False
        try:
            visible_getter = getattr(view, "isVisible", None)
            visible = bool(visible_getter()) if callable(visible_getter) else None
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            visible = None
        if visible is False and self._surface_mask_view_visible is True:
            # 隐藏桌宠不会再产生页面 leave/resize；暂停异步 alpha 回读，
            # 否则控制台打开期间仍会高频 grab/写 Shape 并占用主线程。
            if self._surface_mask_view_visible is not False:
                self._surface_mask_capture_generation += 1
                self._surface_mask_capture_pending = False
                self._set_x11_input_shapes(view, (), force=True)
                # 旧区域仅用于视觉保留；隐藏期间不能继续把它报告成
                # 可点击输入区，恢复显示后由当前帧重新建立。
                self._surface_mask_input_region = None
            self._surface_mask_view_visible = False
            return False
        if visible is True:
            self._surface_mask_view_visible = True
        timer = self._surface_mask_timer
        if timer is not None:
            try:
                timer.setInterval(self._surface_mask_interval_ms())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        if self._click_through or self._window_locked:
            # Qt 的 WindowTransparentForInput 对 QQuickWidget 子表面并不总是
            # 同步；直接清空 X11 Input Shape，确保锁定/穿透状态立即把点击
            # 交还底层窗口，同时保留 Qt 的可见 Shape。
            self._set_x11_input_shapes(view, (), force=True)
            return True
        # 页面提供模型 canvas alpha 时优先走该路径，避免把气泡/Toast/marker
        # 的可见像素错误地变成 Qt/X11 输入区域。没有页面桥的纯测试替身
        # 才继续使用下面的同步抓图回退。
        capture_requested = self._request_surface_mask_capture(view)
        if capture_requested is not None:
            if self._surface_mask_region is None:
                # 首帧/页面重建期间必须 fail-open；否则原生子表面会以
                # 完整矩形拦截气泡外的底层点击，直到异步 canvas alpha 到达。
                self._set_x11_input_shapes(view, (), force=True)
            return bool(self._surface_mask_region is not None)
        previous_region = self._surface_mask_region
        had_previous_region = bool(
            previous_region is not None
            and callable(getattr(previous_region, "isEmpty", None))
            and not previous_region.isEmpty()
        )
        applied = False
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtGui import QBitmap, QRegion

            # setMask 同时裁剪绘制和输入。先暂时移除旧掩码再抓取完整
            # WebEngine 帧，否则气泡/反馈在旧掩码外出现时永远无法进入
            # 下一份截图，动态显示会被永久裁掉。
            if had_previous_region:
                self._clear_surface_masks(view)
            image = view.grab().toImage()
            if image.isNull():
                self._set_x11_input_shapes(view, (), force=True)
                return False
            # QWindow/QScreen 在高 DPI 下返回物理像素截图，而 Shape
            # 区域使用顶层窗口的逻辑像素。直接把 2x/1.5x 图片交给
            # QRegion 会把掩码放大到窗口外，产生裁切、黑边和点击坐标
            # 偏移；先归一化到 view 的逻辑尺寸再创建位图。
            width_getter = getattr(view, "width", None)
            height_getter = getattr(view, "height", None)
            logical_width = max(1, int(width_getter())) if callable(width_getter) else image.width()
            logical_height = (
                max(1, int(height_getter())) if callable(height_getter) else image.height()
            )
            if image.width() != logical_width or image.height() != logical_height:
                image = image.scaled(
                    logical_width,
                    logical_height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.FastTransformation,
                )
            image.setDevicePixelRatio(1.0)
            # WebEngine 在模型尚未完成首帧、GPU surface 短暂失效或页面切换
            # 时可能返回尺寸正常但全透明的截图。不能把这张空图提交为新
            # Shape，否则窗口会被裁成 0x0，随后页面即使恢复也无法接收
            # 输入。保留上一份有效掩码，下一次定时刷新再重试。
            mask = QBitmap.fromImage(image.createAlphaMask())
            region = QRegion(mask)
            if region.isEmpty():
                self._set_x11_input_shapes(view, (), force=True)
                return False
            self._surface_mask_image = image
            if not self._apply_surface_masks(view, region):
                return False
            self._surface_mask_region = region
            applied = True
            return True
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        finally:
            if not applied and had_previous_region:
                self._apply_surface_masks(view, previous_region)

    def is_user_interacting(self) -> bool:
        """返回 WebEngine 拖动或鼠标悬停是否正在处理用户交互。"""

        # 页面/模型尚未完成握手时，禁止自主行为先移动正在建立的
        # Chromium surface；否则首帧尺寸和窗口位置会同时变化，容易出现
        # 黑帧、缩放跳变或“刚启动就乱跑”。模型失败后由上层切换精灵，
        # 不再满足该阻塞条件。
        model_ready = getattr(self.renderer, "model_ready", None)
        if isinstance(model_ready, bool) and not model_ready:
            return True
        reload_status = getattr(self.renderer, "model_reload_status", None)
        if isinstance(reload_status, Mapping):
            if str(reload_status.get("status", "") or "").strip().lower() == "pending":
                # 候选模型正在异步预加载；行为不能在旧模型仍可见、新
                # 模型即将交换的窗口提交动作，否则交换后会出现迟到姿态。
                return True
        if self._window_locked:
            return False
        # 拖动释放后的短稳定窗口也视为交互中。行为循环可能比 Qt 的
        # 50ms 轮询更快，单靠 release 边沿会漏掉短按并让随机动作立即
        # 抢占渲染器；该门只影响自主行为，锁定窗口仍保持可自由漫游。
        if monotonic() < self._autonomous_block_until:
            return True
        interaction = self._interaction_filter
        return bool(
            self._page_pointer_over
            or self._page_drag_origin is not None
            or self._page_system_move_active
            or (
                interaction is not None
                and (
                    getattr(interaction, "_drag_origin", None) is not None
                    or bool(getattr(interaction, "_system_move_active", False))
                    or bool(getattr(interaction, "_hover_targets", ()))
                )
            )
        )

    def _start_page_system_move(self) -> bool:
        """在原生 Wayland 上优先让 compositor 处理页面拖动。"""

        backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
        if backend != "wayland":
            return False
        view = self.view
        handle_getter = getattr(view, "windowHandle", None) if view is not None else None
        handle = handle_getter() if callable(handle_getter) else None
        starter = getattr(handle, "startSystemMove", None)
        if not callable(starter):
            return False
        try:
            accepted = bool(starter())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            accepted = False
        self._page_system_move_active = accepted
        if accepted:
            self._set_renderer_dragging(True)
            self._arm_page_drag_watchdog()
        return accepted

    def _arm_page_drag_watchdog(self) -> None:
        """为页面桥拖动建立有界释放兜底。"""

        self._page_drag_watchdog_generation += 1
        generation = self._page_drag_watchdog_generation

        def expire() -> None:
            if generation != self._page_drag_watchdog_generation:
                return
            if self._shutdown or self._closing:
                return
            if not (
                self._page_drag_origin is not None
                or self._page_dragging
                or self._page_system_move_active
            ):
                return
            logger.debug("Web page drag release was not observed; clearing stale transaction")
            self._clear_page_drag()

        try:
            if _qt_available and QCoreApplication.instance() is not None:
                QTimer.singleShot(_DRAG_TRANSACTION_TIMEOUT_MS, expire)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

    def _clear_page_drag(self) -> None:
        """清理页面指针拖动状态。"""

        had_drag = self._drag_transaction_active() or self._renderer_dragging
        self._page_drag_watchdog_generation += 1
        # 页面按下会先在 Chromium 内部冻结本地时钟，只有越过拖动阈值
        # 后宿主才会把 `_renderer_dragging` 置真。短点击/快速 release
        # 因而可能在宿主标志仍为 false 时留下页面 dragging；此处仍强制
        # 发一次 false，让页面稳定窗和 watchdog 能够收敛，即使其本地
        # release timer 被后台节流或丢弃。
        self._set_renderer_dragging(False, force=had_drag and not self._renderer_dragging)
        self._page_drag_origin = None
        self._page_drag_global_origin = None
        self._page_drag_uses_global = False
        self._page_window_origin = None
        self._page_dragging = False
        self._page_system_move_active = False
        self._interaction_interrupt_notified = False
        self._last_drag_target = None
        self._last_drag_notify_at = 0.0
        self._pending_drag_target = None
        self._last_drag_submit_at = 0.0
        timer = self._drag_flush_timer
        if timer is not None:
            try:
                timer.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        # 拖动期间暂停光标追踪后，下一次定时更新必须重新提交局部坐标。
        self._last_cursor_target = None
        if had_drag:
            self._invalidate_geometry()
            self._schedule_surface_mask_after_move()
        self._flush_deferred_page_resize()

    def _set_renderer_dragging(self, dragging: bool, *, force: bool = False) -> None:
        """同步拖动状态到渲染器，冻结动作时钟和注视速度。"""

        value = bool(dragging)
        if not value:
            # 原生过滤器的短点击不会把 renderer_dragging 置 True；即使
            # 回调是一次 false→false，也必须结束本次手势的通知去重状态。
            self._interaction_interrupt_notified = False
        if self._renderer_dragging == value and not force:
            return
        if value:
            self._notify_interaction_started()
            # 原生过滤器路径不会经过页面 `press` 回调；在它越过拖动阈值
            # 的瞬间建立同一几何事务，阻止正在途中的 canvas 回读提交旧
            # Shape。页面桥路径重复调用是幂等的。
            self._invalidate_geometry()
            self._begin_surface_mask_move()
            # 运行中的异步自主移动仍需由自身 finally 归还 pending；拖动
            # 事务通过 generation/drag 状态隔离，不在这里篡改计数。
            self._autonomous_geometry_until = 0.0
        if not value:
            self._autonomous_block_until = max(
                self._autonomous_block_until,
                monotonic() + _AUTONOMOUS_RESUME_GRACE_SECONDS,
            )
            # release/cancel 路径会先冲刷最后一个目标；以当前顶层位置
            # 建立新的追踪基线，避免释放后的下一次定时器把拖动前坐标
            # 误判成外部窗口跳变，再次清空 Live2D 注视速度。
            position = self._read_position(self.view)
            self._last_window_position = position
            self._invalidate_geometry()
        self._renderer_dragging = value
        setter = getattr(self.renderer, "set_dragging", None)
        if callable(setter):
            try:
                setter(value)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("failed to update Web Live2D dragging state", exc_info=True)
        if not value:
            # 原生过滤器在回调返回后才清除自身字段；排到下一事件循环
            # 的恢复检查可确保它不会被旧 `_drag_origin` 阻断。
            try:
                if _qt_available and QCoreApplication.instance() is not None:
                    QTimer.singleShot(0, self._schedule_surface_mask_after_move)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            self._flush_deferred_page_resize()

    @staticmethod
    def _page_point(payload: Mapping[str, object]) -> QPoint | None:
        """把页面 CSS 坐标转换为有限的 Qt 局部坐标。"""

        try:
            raw_x = payload.get("x")
            raw_y = payload.get("y")
            # bool 虽然可以被 ``float`` 转换，但它不是合法的像素坐标；
            # 拒绝这类载荷，避免 WebChannel 异常数据被解释为 (0, 1)。
            if isinstance(raw_x, bool) or isinstance(raw_y, bool):
                return None
            x = float(raw_x)
            y = float(raw_y)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return QPoint(round(x), round(y))

    @staticmethod
    def _page_screen_point(payload: Mapping[str, object]) -> QPoint | None:
        """读取页面提供的屏幕全局坐标；缺失时返回 ``None``。

        页面局部坐标在窗口拖动期间不是稳定参考系。只有两个坐标同时为
        有限数值时才启用全局路径，避免旧页面/测试载荷的半截字段被误当
        成原点。
        """

        try:
            raw_x = payload.get("screen_x")
            raw_y = payload.get("screen_y")
            # 与局部坐标相同，bool 不是屏幕坐标。不能让 True/False
            # 经由 float() 静默变成 1/0 后启用全局拖动路径。
            if isinstance(raw_x, bool) or isinstance(raw_y, bool):
                return None
            x = float(raw_x)
            y = float(raw_y)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        return QPoint(round(x), round(y))

    def _page_global_point(
        self,
        payload: Mapping[str, object],
        local_point: QPoint | None,
    ) -> QPoint | None:
        """把页面事件转换为不随窗口移动的屏幕坐标。

        新页面直接传 ``screen_x/screen_y``。旧页面没有该字段时，用事件
        到达时窗口当前原点加局部坐标重建全局点；这比直接累加局部增量
        稳定，因为窗口已经移动后局部坐标的变化会抵消窗口位移。
        """

        screen_point = self._page_screen_point(payload)
        if screen_point is not None:
            return screen_point
        if local_point is None:
            return None
        position = self._read_position(self.view)
        if position is not None:
            return self._map_local_to_global(self.view, local_point, position)
        # 没有位置回读能力的兼容替身沿用旧的局部坐标路径。
        return QPoint(local_point)

    @staticmethod
    def _page_has_screen_coordinates(payload: Mapping[str, object]) -> bool:
        """判断页面是否使用新版屏幕坐标载荷。"""

        # 页面脚本在旧浏览器没有 ``screenX/screenY`` 时会显式发送 null；
        # 仅检查键是否存在会把拖动原点置为空并清掉整段手势。只有两个
        # 值都能通过有限数值校验时才切换到全局坐标路径。
        return WebPetHost._page_screen_point(payload) is not None

    @staticmethod
    def _map_local_to_global(
        view: object,
        local_point: QPoint,
        fallback_position: tuple[int, int],
    ) -> QPoint:
        """将视图局部点转换为屏幕坐标，并兼容无 Qt 映射的替身。

        ``pos()`` 对带父窗口的 QWidget 是父坐标，而页面事件和
        ``QCursor.pos()`` 使用屏幕坐标。优先调用 Qt 的 ``mapToGlobal``
        处理窗口边框、嵌套窗口和平台装饰；测试替身或旧 WebView 没有该
        方法时才回退到顶层位置加局部坐标。
        """

        mapper = getattr(view, "mapToGlobal", None)
        if callable(mapper):
            try:
                mapped = mapper(local_point)
                return QPoint(round(float(mapped.x())), round(float(mapped.y())))
            except (AttributeError, OSError, OverflowError, RuntimeError, TypeError, ValueError):
                pass
        try:
            return QPoint(
                int(fallback_position[0]) + int(local_point.x()),
                int(fallback_position[1]) + int(local_point.y()),
            )
        except (AttributeError, IndexError, OSError, OverflowError, TypeError, ValueError):
            # fallback_position 由 _read_position 产生，只有不完整的测试
            # 替身才会到达这里；返回局部点仍比丢弃拖动事件可恢复。
            return QPoint(int(local_point.x()), int(local_point.y()))

    def _move_page_drag_target(self, target: QPoint) -> None:
        """合并页面拖动坐标，并按受控频率提交最新位置。"""

        if self._closing or self._shutdown:
            return
        self._begin_surface_mask_move()
        target_tuple = (int(target.x()), int(target.y()))
        if target_tuple == self._pending_drag_target or target_tuple == self._last_drag_target:
            return
        self._pending_drag_target = target_tuple
        now = monotonic()
        if (
            self._last_drag_submit_at <= 0.0
            or now - self._last_drag_submit_at >= _DRAG_MOVE_INTERVAL_SECONDS
        ):
            self._flush_page_drag_target()
            return
        self._schedule_page_drag_flush()

    def _schedule_page_drag_flush(self) -> None:
        """在下一可用拖动时间片冲刷页面坐标。"""

        if self._pending_drag_target is None:
            return
        view = self.view
        try:
            if _qt_available and view is not None and QCoreApplication.instance() is not None:
                timer = self._drag_flush_timer
                if timer is None:
                    timer = QTimer(view)
                    timer.setSingleShot(True)
                    try:
                        timer.setTimerType(Qt.TimerType.PreciseTimer)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                    timer.timeout.connect(self._flush_page_drag_target)
                    self._drag_flush_timer = timer
                remaining = max(
                    0.001,
                    _DRAG_MOVE_INTERVAL_SECONDS - (monotonic() - self._last_drag_submit_at),
                )
                timer.start(max(1, int(math.ceil(remaining * 1000.0))))
                return
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        # 无事件循环的测试替身或已销毁的 Qt 对象保留 pending；下一次
        # move 或 release(force=True) 会冲刷它，不能在这里递归重试。

    def _flush_page_drag_target(self, *, force: bool = False) -> None:
        """只提交队列中的最后坐标，避免窗口追赶过期 move。"""

        if self._closing or self._shutdown:
            self._pending_drag_target = None
            return
        target_tuple = self._pending_drag_target
        if target_tuple is None:
            return
        if (
            not force
            and self._last_drag_submit_at > 0.0
            and monotonic() - self._last_drag_submit_at < _DRAG_MOVE_INTERVAL_SECONDS
        ):
            self._schedule_page_drag_flush()
            return
        self._pending_drag_target = None
        if target_tuple == self._last_drag_target:
            return
        self._apply_page_drag_target(QPoint(*target_tuple))

    @staticmethod
    def _move_result_accepted(result: object) -> bool:
        """判断平台移动回执是否明确表示请求已提交。"""

        if isinstance(result, Mapping):
            raw = result.get("status", result.get("state", ""))
        else:
            raw = getattr(result, "state", getattr(result, "status", ""))
        value = getattr(raw, "value", raw)
        return str(value or "").strip().lower() in {
            "available",
            "degraded",
            "requested",
            "completed",
        }

    def _apply_page_drag_target(self, target: QPoint) -> None:
        """提交一个已合并的页面拖动目标。"""

        if self._closing or self._shutdown:
            return
        view = self.view
        if view is None:
            return
        # 平台适配器本身通常也会调用 QWidget.move/QWindow.setPosition。
        # 先调用本地 move 再调用适配器会对同一目标提交两次几何请求；在
        # QWebEngine 原生子表面异步提交时，这些请求会交错触发重绘与 Shape
        # 回读。平台返回能力对象后视为请求已提交，不再因瞬时回读滞后重复
        # 调用 QWidget.move；只有适配器异常/未提供结果时才走 Qt 回退。
        moved = False
        mover = getattr(self.platform, "move_overlay", None) if self.platform else None
        if callable(mover):
            try:
                result = mover(view, target.x(), target.y())
                if inspect.iscoroutine(result):
                    # 拖动回调不能等待异步平台结果；关闭未执行协程，
                    # 让一次 Qt 回退承担当前坐标提交。
                    result.close()
                else:
                    observed = self._read_position(view)
                    moved = observed == (int(target.x()), int(target.y()))
                    # 只有明确的成功/降级回执才抑制 Qt 回退；例如
                    # ``status=unavailable`` 仍需保留一次本地移动机会。
                    moved = moved or self._move_result_accepted(result)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("Web Live2D platform move failed during page drag", exc_info=True)
        if not moved:
            move = getattr(view, "move", None)
            if callable(move):
                try:
                    move(target.x(), target.y())
                    moved = True
                except TypeError:
                    try:
                        move(target)
                        moved = True
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        pass
                except (AttributeError, OSError, RuntimeError, ValueError):
                    pass
        if not moved:
            setter = getattr(view, "setPosition", None)
            if callable(setter):
                try:
                    setter(target)
                except TypeError:
                    try:
                        setter(target.x(), target.y())
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        pass
                except (AttributeError, OSError, RuntimeError, ValueError):
                    pass
        self._last_drag_submit_at = monotonic()
        # 窗口位置改变后，旧的局部光标目标不能无限期复用；但拖动
        # 每帧都会进这里，不能每次都清零临界阻尼速度，否则模型注视会
        # 在拖动期间像冻结一样。按 8px/120ms 去抖复位，仍能覆盖跨屏和
        # 合成器坐标重排，同时保留连续追踪。
        target_tuple = (int(target.x()), int(target.y()))
        previous_target = self._last_drag_target
        self._last_drag_target = target_tuple
        now = monotonic()
        distance = (
            math.hypot(
                target_tuple[0] - previous_target[0],
                target_tuple[1] - previous_target[1],
            )
            if previous_target is not None
            else float("inf")
        )
        should_notify = (
            previous_target is None or distance >= 8.0 or now - self._last_drag_notify_at >= 0.12
        )
        if should_notify:
            self._last_cursor_target = None
            self._last_window_position = None
            self._last_drag_notify_at = now
            notifier = getattr(self.renderer, "notify_window_moved", None)
            if callable(notifier):
                try:
                    notifier()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to reset Live2D tracking after page drag", exc_info=True)
        # 持续拖动期间保持上一份 Shape；释放/取消路径会在清理拖动状态后
        # 单次恢复，避免高频窗口移动与 canvas 回读交错。
        if not self._page_dragging and not self._page_system_move_active:
            self._schedule_surface_mask_after_move()

    def _handle_page_interaction(self, payload: Mapping[str, object]) -> None:
        """处理 Web 页面回传的拖动、菜单、双击和悬停事件。"""

        view = self.view
        if self._closing or self._shutdown or view is None or not isinstance(payload, Mapping):
            return
        event_type = str(payload.get("type", "") or "").strip().lower()
        if (
            not self._draining_deferred_page_events
            and event_type
            in {"press", "move", "release", "cancel", "context", "double", "transparent"}
            and monotonic() < self._interaction_rebuild_until
        ):
            # 重建期间只保留有界事件序列。活动手势的 press、最新 move 和
            # release/cancel 属于不可丢弃的骨架；旧手势只保留到有空间为止。
            # 若先丢掉活动 press，重建结束后的 replay 会把 move/release
            # 当成无原点事件，表现为“拖动完全没有反应”。
            queued = dict(payload)
            events = self._deferred_page_events
            if len(events) >= 8:
                active_press_index: int | None = None
                for index, item in enumerate(events):
                    queued_type = str(item.get("type", "") or "").strip().lower()
                    if queued_type == "press":
                        active_press_index = index
                    elif queued_type in {"release", "cancel"}:
                        active_press_index = None

                # 优先淘汰活动手势之前的旧事件；这不会改变当前手势的
                # press→move→release 顺序，也能压缩连续点击产生的历史。
                discard_index = None
                if active_press_index is not None:
                    discard_index = next(
                        (
                            index
                            for index in range(active_press_index)
                            if str(events[index].get("type", "") or "").strip().lower()
                            not in {"press", "move", "release", "cancel"}
                        ),
                        None,
                    )
                    if discard_index is None and active_press_index > 0:
                        discard_index = 0

                # 活动手势之前没有可淘汰项时，move 只保留最新一个；
                # release/cancel 到达时保留已有 move，让最终坐标和顺序
                # 都能回放。
                if (
                    discard_index is None
                    and event_type == "move"
                    and active_press_index is not None
                ):
                    discard_index = next(
                        (
                            index
                            for index in range(active_press_index + 1, len(events))
                            if str(events[index].get("type", "") or "").strip().lower() == "move"
                        ),
                        None,
                    )
                if discard_index is None:
                    discard_index = next(
                        (
                            index
                            for index, item in enumerate(events)
                            if str(item.get("type", "") or "").strip().lower() == "move"
                            and index != active_press_index
                        ),
                        None,
                    )
                if discard_index is None:
                    discard_index = next(
                        (index for index in range(len(events)) if index != active_press_index),
                        0,
                    )
                del events[discard_index]
            events.append(queued)
            self._schedule_deferred_page_events()
            return
        point = self._page_point(payload)
        global_point = self._page_global_point(payload, point)
        page_bridge_ready = bool(getattr(self.renderer, "page_bridge_ready", False))
        if event_type in {"press", "context"} and page_bridge_ready:
            # 当前页面桥必须携带几何命中结果；缺失/false 都按透明区域
            # 处理，避免旧脚本或伪造载荷重新打开矩形拖动入口。
            if payload.get("hit") is not True:
                event_type = "transparent"
        if event_type == "enter":
            hit = payload.get("hit")
            pointer_over = hit is not False
            if self._page_pointer_over != pointer_over:
                _set_cursor_shape(view, "OpenHandCursor" if pointer_over else "ArrowCursor")
            self._page_pointer_over = pointer_over
            if not self._update_cursor_tracking_from_page(payload):
                self.update_cursor_tracking()
            return
        if event_type == "leave":
            self._page_pointer_over = False
            if not self._page_dragging and not self._page_system_move_active:
                _set_cursor_shape(view, "ArrowCursor")
            return
        if event_type == "enter_key":
            if payload.get("hit") is not True or self._window_locked:
                return
            callback = self._enter_callback
            if callable(callback):
                try:
                    callback()
                except Exception:
                    logger.exception("Web Live2D enter callback failed")
            return
        # Chromium 在失去指针捕获、窗口切换或合成器取消系统移动时，
        # 可能发送不带坐标的 release/cancel。清理必须先于坐标校验，
        # 否则下一次点击会继续沿用旧的拖动原点，表现为桌宠无法再次移动。
        if event_type in {"release", "cancel"}:
            # release 可能先于 move 抵达；只要按下到释放的位移已越过
            # Qt 阈值，就在这里补建拖动状态，保证最终坐标仍会提交。
            if event_type == "release" and not self._page_dragging:
                origin = (
                    self._page_drag_global_origin
                    if self._page_drag_uses_global
                    else self._page_drag_origin
                )
                release_point = global_point if self._page_drag_uses_global else point
                if origin is not None and release_point is not None:
                    try:
                        from PySide6.QtWidgets import QApplication

                        threshold = int(QApplication.startDragDistance())
                    except (ImportError, RuntimeError, TypeError, ValueError):
                        threshold = 4
                    if (release_point - origin).manhattanLength() >= max(1, threshold):
                        self._page_dragging = True
                        self._set_renderer_dragging(True)
            if self._page_dragging:
                # pointerup 会携带最后一个页面坐标；即便此前的 move
                # 回调还在 WebChannel 队列中，也先用 release 坐标覆盖
                # pending，确保窗口不会停在倒数第二个位置。
                if event_type == "release" and point is not None:
                    origin = (
                        self._page_drag_global_origin
                        if self._page_drag_uses_global
                        else self._page_drag_origin
                    )
                    window_origin = self._page_window_origin
                    release_point = global_point if self._page_drag_uses_global else point
                    if (
                        origin is not None
                        and window_origin is not None
                        and release_point is not None
                    ):
                        self._pending_drag_target = (
                            int(window_origin.x() + release_point.x() - origin.x()),
                            int(window_origin.y() + release_point.y() - origin.y()),
                        )
                self._flush_page_drag_target(force=True)
            self._clear_page_drag()
            # 未越过拖动阈值的普通点击也在按下时暂停过 Shape；清理完
            # 拖动状态后再结束短暂停顿，避免恢复函数看到 active 状态而跳过。
            self._schedule_surface_mask_after_move()
            _set_cursor_shape(
                view,
                "OpenHandCursor" if self._page_pointer_over else "ArrowCursor",
            )
            return
        if event_type == "hover":
            pointer_over = payload.get("hit") is True
            if self._page_pointer_over != pointer_over:
                _set_cursor_shape(view, "OpenHandCursor" if pointer_over else "ArrowCursor")
            self._page_pointer_over = pointer_over
            # 页面桥提供了节流后的坐标时，立即刷新一次注视目标；这让锁定
            # 或点击穿透状态下仍能视觉跟随划过位置，而不依赖下一次全局
            # 光标轮询。该路径只写入渲染器姿态，不触发对话/工具调用。
            if not self._update_cursor_tracking_from_page(payload):
                self.update_cursor_tracking()
            return
        if self._window_locked:
            self._clear_page_drag()
            return
        if event_type == "transparent":
            self._clear_page_drag()
            self._schedule_surface_mask_after_move()
            self._page_pointer_over = False
            _set_cursor_shape(view, "ArrowCursor")
            clear_focus = getattr(view, "clearFocus", None)
            if callable(clear_focus):
                try:
                    clear_focus()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            return
        # 双击只需要打开控制台，不依赖页面坐标；即使浏览器没有提供
        # clientX/clientY，也不能让控制台入口失效。
        if event_type == "double":
            callback = getattr(self, "_double_click_callback", None)
            self._clear_page_drag()
            self._schedule_surface_mask_after_move()
            if "hit" in payload and not bool(payload.get("hit")):
                return
            if callable(callback):
                try:
                    callback()
                except Exception:
                    logger.exception("Web Live2D double-click callback failed")
            return
        if point is None:
            return
        button = str(payload.get("button", "") or "").strip().lower()
        if event_type == "press":
            if button == "left":
                if "hit" in payload and not bool(payload.get("hit")):
                    # 页面桥对透明区域显式 fail-closed；不能让空点建立
                    # 拖动原点、抓走焦点或改变桌宠窗口位置。
                    return
                _set_cursor_shape(view, "ClosedHandCursor")
                position = self._read_position(view)
                if position is None:
                    return
                # 页面按下即取得几何事务所有权；即使尚未越过拖动阈值，
                # 旧的异步 move/Shape 回调也不能在这段手势中回写窗口。
                self._invalidate_geometry()
                self._begin_surface_mask_move()
                self._notify_interaction_started()
                if self._start_page_system_move():
                    return
                # 新版页面携带屏幕全局坐标；旧页面没有该字段时保留局部
                # 坐标契约，避免第三方 WebView 的兼容测试/集成被破坏。
                self._page_drag_origin = point
                self._page_drag_uses_global = self._page_has_screen_coordinates(payload)
                self._page_drag_global_origin = (
                    global_point if self._page_drag_uses_global else point
                )
                if self._page_drag_origin is None or self._page_drag_global_origin is None:
                    self._clear_page_drag()
                    return
                self._page_window_origin = QPoint(*position)
                self._page_dragging = False
                self._arm_page_drag_watchdog()
                return
            if button == "right":
                if "hit" in payload and not bool(payload.get("hit")):
                    return
                _set_cursor_shape(view, "ArrowCursor")
                callback = getattr(self, "_context_menu_callback", None)
                if callable(callback):
                    position = self._read_position(view)
                    if position is not None:
                        try:
                            callback(self._map_local_to_global(view, point, position))
                        except Exception:
                            logger.exception("Web Live2D context menu callback failed")
                return
        if event_type == "context":
            if "hit" in payload and not bool(payload.get("hit")):
                return
            callback = getattr(self, "_context_menu_callback", None)
            position = self._read_position(view)
            if callable(callback) and position is not None:
                try:
                    callback(self._map_local_to_global(view, point, position))
                except Exception:
                    logger.exception("Web Live2D context menu callback failed")
            return
        if event_type == "click":
            # 新版 WebLive2DRenderer 会通过独立 click callback 转发；没有
            # 该扩展时仍在宿主侧处理，保证旧渲染器/测试替身可用。
            if not self._renderer_click_callback_active:
                self._emit_click(payload)
            return
        if event_type == "move":
            if self._page_system_move_active:
                return
            origin = (
                self._page_drag_global_origin
                if self._page_drag_uses_global
                else self._page_drag_origin
            )
            window_origin = self._page_window_origin
            drag_point = global_point if self._page_drag_uses_global else point
            if origin is None or window_origin is None or drag_point is None:
                return
            self._arm_page_drag_watchdog()
            if not self._page_dragging:
                try:
                    from PySide6.QtWidgets import QApplication

                    threshold = int(QApplication.startDragDistance())
                except (ImportError, RuntimeError, TypeError, ValueError):
                    threshold = 4
                if (drag_point - origin).manhattanLength() < max(1, threshold):
                    return
                self._page_dragging = True
                self._set_renderer_dragging(True)
            self._move_page_drag_target(window_origin + drag_point - origin)
            return

    def _emit_click(self, payload: Mapping[str, object]) -> None:
        """把已校验的页面点击转换为分区回调。"""

        callback = self._click_callback
        view = self.view
        if callback is None or view is None:
            return
        # WebEngine 握手前的原生回退只用于拖动/菜单，不应在模型尚未首帧
        # 或页面已报错时按整块画布触发猫猫反馈；真实模型就绪后页面桥会
        # 接管并提供带纹理 alpha 的精确命中结果。旧测试替身没有该属性，
        # 继续保留兼容回退。
        model_ready = getattr(self.renderer, "model_ready", None)
        if isinstance(model_ready, bool) and not model_ready:
            return
        button = str(payload.get("button", "left") or "").strip().lower()
        if button not in {"", "left"}:
            return
        # 页面几何命中结果是业务反馈的前置条件；缺失字段兼容原生 Qt
        # 回退，但显式 ``hit=false`` 必须 fail-closed，避免透明画布或伪造
        # WebChannel 载荷触发猫猫头反馈。
        if "hit" in payload and not bool(payload.get("hit")):
            return
        width_getter = getattr(view, "width", None)
        height_getter = getattr(view, "height", None)
        width = width_getter() if callable(width_getter) else 0
        height = height_getter() if callable(height_getter) else 0
        normalized = make_pet_click_payload(payload, width=width, height=height)
        if normalized is None:
            return
        if (
            isinstance(model_ready, bool)
            and model_ready
            and not str(normalized.get("part", "") or "").strip()
        ):
            part = classify_pet_part(normalized.get("x"), normalized.get("y"), width, height)
            if part is not None:
                normalized["part"] = part
        try:
            callback(normalized)
        except Exception:
            logger.exception("Web Live2D click callback failed")

    def _handle_renderer_click(self, payload: Mapping[str, object]) -> None:
        """接收渲染器独立 click callback，沿用宿主分区契约。"""

        self._emit_click(payload)

    async def _move_after_dispatch(
        self,
        capability: object,
        x: float,
        y: float,
        *,
        autonomous_step: bool = False,
        geometry_generation: int | None = None,
        autonomous_geometry_deferred: bool = False,
    ) -> dict[str, object]:
        geometry_owned = bool(autonomous_step and not autonomous_geometry_deferred)
        try:
            # 调度器返回的 awaitable 尚未必真正进入 Qt/平台线程。若用户在
            # ``move_to`` 返回后、该协程首次运行前开始拖动，旧的自主目标
            # 不能再被 await；否则平台仍会提交过期坐标，随后新拖动位置会
            # 被旧请求回弹。先在 await 前做一次代次/交互检查，并回收未
            # 启动的协程，避免既产生位置跳变又留下 RuntimeWarning。
            if (
                not self._geometry_is_current(geometry_generation)
                or self._drag_transaction_active()
            ):
                _safe_close_awaitable(capability)
                return {
                    "status": "cancelled",
                    "x": int(round(float(x))),
                    "y": int(round(float(y))),
                    "detail": "geometry transaction superseded",
                }
            if autonomous_step and autonomous_geometry_deferred:
                # move_to 已释放调用期 reservation；只有该协程实际进入
                # 执行后才增加 pending，因而 pre-start cancellation 不泄漏。
                self._activate_autonomous_geometry()
                geometry_owned = True
            result = await capability
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Web host async move failed: %s", type(exc).__name__)
            return {
                "status": "unavailable",
                "x": int(round(float(x))),
                "y": int(round(float(y))),
                "detail": "桌宠位置暂时无法调整，请稍后重试",
            }
        finally:
            try:
                if (
                    not autonomous_step
                    and self._geometry_is_current(geometry_generation)
                    and not self._drag_transaction_active()
                ):
                    self._schedule_surface_mask_after_move()
            finally:
                if geometry_owned:
                    self._end_autonomous_geometry()
        if not self._geometry_is_current(geometry_generation):
            # 事务已被新的拖动/窗口操作取代；保留返回契约，但不要把旧
            # 回执继续当作当前几何状态使用。
            return {
                "status": "cancelled",
                "x": int(round(float(x))),
                "y": int(round(float(y))),
                "detail": "geometry transaction superseded",
            }
        self._last_cursor_target = None
        self._last_window_position = None
        state = getattr(getattr(result, "state", None), "value", None)
        detail = getattr(result, "detail", "")
        if isinstance(result, Mapping):
            value = result.get("status", result.get("state", "unavailable"))
            state = getattr(value, "value", value)
            detail = result.get("detail", result.get("reason", ""))
        return {
            "status": str(state or "unavailable"),
            "x": int(round(float(x))),
            "y": int(round(float(y))),
            "detail": str(detail or ""),
        }

    def set_rendering_performance(
        self,
        *,
        frame_rate: float | None = None,
        geometry_audit_hz: float | None = None,
    ) -> dict[str, object]:
        """热更新 Web Live2D 的绘制与几何审计频率。"""

        if self._closing or self._shutdown:
            return {"status": "unavailable", "reason": "renderer host is closed"}
        applied: dict[str, float] = {}
        failures: list[str] = []
        for name, value, setter in (
            ("frame_rate", frame_rate, getattr(self.renderer, "set_frame_rate", None)),
            (
                "geometry_audit_hz",
                geometry_audit_hz,
                getattr(self.renderer, "set_geometry_audit_hz", None),
            ),
        ):
            if value is None:
                continue
            if not callable(setter):
                failures.append(name)
                continue
            try:
                if setter(value):
                    applied[name] = float(value)
                else:
                    failures.append(name)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                failures.append(name)
        if failures and not applied:
            return {
                "status": "unavailable",
                "reason": "renderer performance settings are unavailable",
                "failed": tuple(failures),
            }
        return {
            "status": "available" if not failures else "degraded",
            "applied": dict(applied),
            "failed": tuple(failures),
        }

    def set_expression(self, name: str) -> object:
        handler = getattr(self.renderer, "set_expression", None)
        return handler(name) if callable(handler) else False

    def set_expression_request(self, request: ExpressionRequest) -> object:
        """把结构化表情编排原样交给 Web Live2D 渲染器。"""

        if not isinstance(request, ExpressionRequest):
            return {"status": "unavailable", "reason": "expression request is invalid"}
        handler = getattr(self.renderer, "set_expression_request", None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "expression sequencing is unavailable"}
        return handler(request)

    def play_motion(self, name: str) -> object:
        handler = getattr(self.renderer, "play_motion", None)
        return handler(name) if callable(handler) else False

    def play_motion_request(self, request: MotionRequest) -> object:
        """把结构化动作编排原样交给 Web Live2D 渲染器。"""

        if not isinstance(request, MotionRequest):
            return {"status": "unavailable", "reason": "motion request is invalid"}
        handler = getattr(self.renderer, "play_motion_request", None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "motion sequencing is unavailable"}
        return handler(request)

    def set_direction(self, direction: str) -> object:
        """把朝向交给 Web Live2D/精灵渲染器。"""

        handler = getattr(self.renderer, "set_direction", None)
        return handler(direction) if callable(handler) else False

    def set_speech(
        self,
        text: str,
        *,
        mood: str = "neutral",
        visible: bool = True,
        speaking: bool = True,
    ) -> object:
        handler = getattr(self.renderer, "set_speech", None)
        if not callable(handler):
            return False
        try:
            return handler(text, mood=mood, visible=visible, speaking=speaking)
        except TypeError:
            # 兼容仍实现旧三参数契约的测试/外部渲染器；旧渲染器无法区分
            # 气泡与口型时沿用其原有可见即说话行为。
            return handler(text, mood=mood, visible=visible)

    def set_interaction_feedback(
        self,
        value: Mapping[str, object] | None,
        *,
        visible: bool = True,
    ) -> object:
        """把部位互动结果转发给渲染层的非阻塞反馈层。"""

        handler = getattr(self.renderer, "set_interaction_feedback", None)
        if not callable(handler):
            return False
        return handler(value, visible=visible)

    def set_click_through(self, enabled: bool) -> object:
        requested = bool(enabled)
        if self._window_locked and not requested:
            return {
                "status": "unavailable",
                "enabled": True,
                "locked": True,
                "reason": "window is locked; unlock it before restoring input",
            }
        self._click_through_transition_generation += 1
        transition_generation = self._click_through_transition_generation
        self._click_through_transition_pending = requested
        transition_started_at = monotonic()
        desired_topmost, topmost_was_pending = self._topmost_intent()
        self._log_window_flag_transaction(
            "start",
            generation=transition_generation,
            status="started",
            requested_topmost=desired_topmost,
            requested_click_through=requested,
            requested_lock=self._window_locked,
            reason_code="click_through_request",
        )
        handler = getattr(self.platform, "set_click_through", None) if self.platform else None
        view = self.view
        if view is not None and callable(handler):
            backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
            native_window = self._native_topmost_window(view)
            native_input_path = bool(
                backend in {"x11", "wayland", "windows"}
                and self._is_qwebengine_view(view)
                and callable(getattr(native_window, "setFlag", None))
            )
            if not native_input_path:
                self._mark_interaction_rebuild()
            mask_paused = (
                False if native_input_path else self._pause_surface_mask_for_window_rebuild(view)
            )
            geometry_generation = self._geometry_generation
            position = self._read_position(view)
            try:
                capability = handler(view, requested)
            except asyncio.CancelledError:
                self._resume_surface_mask_after_window_rebuild(view, mask_paused)
                if transition_generation == self._click_through_transition_generation:
                    self._click_through_transition_pending = None
                self._log_window_flag_transaction(
                    "terminal",
                    generation=transition_generation,
                    status="cancelled",
                    requested_topmost=desired_topmost,
                    requested_click_through=requested,
                    requested_lock=self._window_locked,
                    verified=False,
                    reason_code="dispatch_cancelled",
                    started_at=transition_started_at,
                )
                self._reconcile_topmost_after_input_transition(
                    desired_topmost,
                    had_pending_request=topmost_was_pending,
                    generation=transition_generation,
                )
                raise
            except Exception as exc:
                self._resume_surface_mask_after_window_rebuild(view, mask_paused)
                logger.debug("Web host click-through request failed: %s", type(exc).__name__)
                result = {
                    "status": "unavailable",
                    "enabled": bool(self._click_through),
                    "detail": "点击穿透暂时无法切换，请使用控制台恢复入口",
                }
                if transition_generation == self._click_through_transition_generation:
                    self._click_through_transition_pending = None
                self._log_window_flag_transaction(
                    "terminal",
                    generation=transition_generation,
                    status="unavailable",
                    requested_topmost=desired_topmost,
                    requested_click_through=requested,
                    requested_lock=self._window_locked,
                    verified=False,
                    reason_code="dispatch_failed",
                    started_at=transition_started_at,
                )
                self._reconcile_topmost_after_input_transition(
                    desired_topmost,
                    had_pending_request=topmost_was_pending,
                    generation=transition_generation,
                )
                return result
            if inspect.isawaitable(capability):
                operation = self._click_through_after_dispatch(
                    capability,
                    requested,
                    position,
                    surface_mask_paused=mask_paused,
                    geometry_generation=geometry_generation,
                    transition_generation=transition_generation,
                    transition_started_at=transition_started_at,
                    desired_topmost=desired_topmost,
                    topmost_was_pending=topmost_was_pending,
                    native_input_path=native_input_path,
                )
                _attach_awaitable_finalizer(operation, capability)
                return operation
            state = getattr(getattr(capability, "state", None), "value", None)
            detail = getattr(capability, "detail", "")
            if isinstance(capability, Mapping):
                value = capability.get("status", capability.get("state", "unavailable"))
                state = getattr(value, "value", value)
                detail = capability.get("detail", capability.get("reason", ""))
            result = {
                "status": str(state or "unavailable"),
                "enabled": requested,
                "detail": str(detail or ""),
            }
            actual = self._input_transparency_active()
            if actual is not None:
                result["enabled"] = actual
                if actual != requested:
                    result["status"] = "unavailable"
                    result["detail"] = (
                        "window system did not retain the requested input transparency flag"
                    )
            accepted = result["status"] in {
                "available",
                "degraded",
                "requested",
                "completed",
            }
            if transition_generation != self._click_through_transition_generation:
                accepted = False
                result = {
                    "status": "cancelled",
                    "enabled": bool(self._click_through),
                    "detail": "window flag transaction superseded",
                }
            if accepted:
                self._click_through = bool(result["enabled"])
                if self._click_through:
                    self._clear_page_hover_state(view)
                if not self._surface_mask_enabled:
                    # 遮罩关闭时仍要让原生 Input Shape 与点击穿透状态
                    # 一致；否则先穿透再关闭遮罩会留下旧的整窗/空区。
                    self._set_x11_input_shapes(
                        view,
                        () if self._click_through else None,
                        force=True,
                    )
                self._restore_position(
                    view,
                    position,
                    geometry_generation=geometry_generation,
                )
                # Qt 切换 WindowTransparentForInput 可能重建 QWebEngineView
                # 的原生子窗口；重新遍历可避免恢复点击后拖动过滤器仍挂在旧
                # 的 Chromium viewport 上。
                if not native_input_path:
                    self._refresh_interaction_targets()
                    self._schedule_interaction_refresh()
            self._resume_surface_mask_after_window_rebuild(view, mask_paused)
            if transition_generation == self._click_through_transition_generation:
                self._click_through_transition_pending = None
            self._log_window_flag_transaction(
                "terminal",
                generation=transition_generation,
                status=str(result["status"]),
                requested_topmost=desired_topmost,
                requested_click_through=requested,
                requested_lock=self._window_locked,
                verified=accepted and bool(result["enabled"]) == requested,
                reason_code=("applied" if accepted else "not_applied"),
                started_at=transition_started_at,
            )
            self._reconcile_topmost_after_input_transition(
                desired_topmost,
                had_pending_request=topmost_was_pending,
                generation=transition_generation,
            )
            return result
        renderer_handler = getattr(self.renderer, "set_click_through", None)
        if not callable(renderer_handler):
            self._click_through_transition_pending = None
            self._log_window_flag_transaction(
                "terminal",
                generation=transition_generation,
                status="unavailable",
                requested_topmost=desired_topmost,
                requested_click_through=requested,
                requested_lock=self._window_locked,
                verified=False,
                reason_code="handler_unavailable",
                started_at=transition_started_at,
            )
            return {"status": "unavailable", "reason": "click-through is unavailable"}
        self._invalidate_geometry()
        try:
            result = renderer_handler(requested)
        except asyncio.CancelledError:
            self._click_through_transition_pending = None
            raise
        except Exception as exc:
            logger.debug("Web renderer click-through request failed: %s", type(exc).__name__)
            self._click_through_transition_pending = None
            self._log_window_flag_transaction(
                "terminal",
                generation=transition_generation,
                status="unavailable",
                requested_topmost=desired_topmost,
                requested_click_through=requested,
                requested_lock=self._window_locked,
                verified=False,
                reason_code="renderer_failed",
                started_at=transition_started_at,
            )
            return {
                "status": "unavailable",
                "reason": "点击穿透暂时无法切换，请使用控制台恢复入口",
            }
        if isinstance(result, Mapping):
            status = str(result.get("status", result.get("state", "")) or "").lower()
            if status in {"available", "degraded", "requested", "completed"}:
                self._click_through = bool(result.get("enabled", requested))
                if self._click_through:
                    self._clear_page_hover_state(self.view)
                self._refresh_interaction_targets()
                self._schedule_interaction_refresh()
        if transition_generation == self._click_through_transition_generation:
            self._click_through_transition_pending = None
        final_status = (
            str(result.get("status", result.get("state", "unknown")))
            if isinstance(result, Mapping)
            else "unknown"
        )
        self._log_window_flag_transaction(
            "terminal",
            generation=transition_generation,
            status=final_status,
            requested_topmost=desired_topmost,
            requested_click_through=requested,
            requested_lock=self._window_locked,
            verified=bool(self._click_through) == requested,
            reason_code="renderer_completed",
            started_at=transition_started_at,
        )
        self._reconcile_topmost_after_input_transition(
            desired_topmost,
            had_pending_request=topmost_was_pending,
            generation=transition_generation,
        )
        return result

    def is_window_locked(self) -> bool:
        """返回锁定窗口状态。"""

        return bool(self._window_locked)

    def _finish_window_lock_async(
        self,
        requested: bool,
        operation: object,
        *,
        transition_generation: int,
    ) -> object:
        """等待异步输入切换并原子收敛锁定状态。"""

        async def resolve() -> dict[str, object]:
            try:
                result = await operation  # type: ignore[misc]
            except asyncio.CancelledError:
                if transition_generation == self._window_lock_transition_generation:
                    self._window_lock_transition_pending = None
                    if requested:
                        self._window_locked = False
                        self._set_renderer_interaction_locked(False)
                    else:
                        self._window_locked = True
                        self._set_renderer_interaction_locked(True)
                raise
            except Exception as exc:
                logger.debug("Web host async window lock update failed: %s", type(exc).__name__)
                if transition_generation != self._window_lock_transition_generation:
                    return {
                        "status": "cancelled",
                        "locked": bool(self._window_locked),
                        "enabled": bool(self._click_through),
                        "reason": "window lock transaction superseded",
                    }
                self._window_lock_transition_pending = None
                if requested:
                    self._window_locked = False
                    self._set_renderer_interaction_locked(False)
                    return {
                        "status": "unavailable",
                        "locked": False,
                        "enabled": bool(self._click_through),
                        "reason": "窗口锁定暂时无法切换，请稍后重试",
                    }
                self._window_locked = True
                self._set_renderer_interaction_locked(True)
                return {
                    "status": "unavailable",
                    "locked": True,
                    "enabled": bool(self._click_through),
                    "reason": "窗口锁定暂时无法解除，请稍后重试",
                }
            if transition_generation != self._window_lock_transition_generation:
                return {
                    "status": "cancelled",
                    "locked": bool(self._window_locked),
                    "enabled": bool(self._click_through),
                    "reason": "window lock transaction superseded",
                }
            self._window_lock_transition_pending = None
            if requested:
                if not self._operation_succeeded(result, enabled=True):
                    self._window_locked = False
                    self._set_renderer_interaction_locked(False)
                    return {
                        "status": "unavailable",
                        "locked": False,
                        "enabled": bool(self._click_through),
                        "reason": "窗口锁定无法启用输入保护",
                    }
                return {"status": "available", "locked": True, "enabled": True}
            if not self._operation_succeeded(result, enabled=bool(self._click_through_before_lock)):
                self._window_locked = True
                self._set_renderer_interaction_locked(True)
                return {
                    "status": "unavailable",
                    "locked": True,
                    "enabled": bool(self._click_through),
                    "reason": "窗口锁定无法恢复之前的输入状态",
                }
            self._click_through_before_lock = False
            self._set_renderer_interaction_locked(False)
            return {
                "status": "available",
                "locked": False,
                "enabled": bool(self._click_through),
            }

        return resolve()

    def set_window_locked(self, enabled: bool) -> object:
        """锁定/解锁窗口；锁定不暂停自主行为和全局光标追踪。"""

        requested = bool(enabled)
        if requested == self._window_locked:
            # 锁定状态与点击穿透状态并非同一个字段；Wayland/异步平台可能
            # 已记录锁定但输入标志尚未落地，不能用 requested 覆盖真实值。
            pending = self._window_lock_transition_pending == requested
            return {
                "status": "requested" if pending else "available",
                "locked": requested,
                "enabled": bool(self._click_through),
            }
        self._window_lock_transition_generation += 1
        transition_generation = self._window_lock_transition_generation
        self._window_lock_transition_pending = requested
        if requested:
            # 快捷键/控制台可能在 pointer release 丢失时锁定窗口；清理
            # 页面桥和原生过滤器的拖动原点，防止解锁后迟到坐标回弹。
            self._clear_page_drag()
            interaction = self._interaction_filter
            clear_drag = getattr(interaction, "_clear_drag", None)
            if callable(clear_drag):
                try:
                    clear_drag()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            self._click_through_before_lock = bool(self._click_through)
            self._window_locked = True
            self._set_renderer_interaction_locked(True)
            result = self.set_click_through(True)
            if inspect.isawaitable(result):
                return self._finish_window_lock_async(
                    True,
                    result,
                    transition_generation=transition_generation,
                )
            if not self._operation_succeeded(result, enabled=True):
                self._window_lock_transition_pending = None
                self._window_locked = False
                self._set_renderer_interaction_locked(False)
                return {
                    "status": "unavailable",
                    "locked": False,
                    "enabled": bool(self._click_through),
                    "reason": "window lock could not enable input transparency",
                }
            self._window_lock_transition_pending = None
            return {"status": "available", "locked": True, "enabled": True}
        self._window_locked = False
        restore = bool(self._click_through_before_lock)
        result = self.set_click_through(restore)
        if inspect.isawaitable(result):
            return self._finish_window_lock_async(
                False,
                result,
                transition_generation=transition_generation,
            )
        if not self._operation_succeeded(result, enabled=restore):
            self._window_lock_transition_pending = None
            self._window_locked = True
            self._set_renderer_interaction_locked(True)
            return {
                "status": "unavailable",
                "locked": True,
                "enabled": bool(self._click_through),
                "reason": "window lock could not restore the previous input state",
            }
        self._click_through_before_lock = False
        self._window_lock_transition_pending = None
        self._set_renderer_interaction_locked(False)
        return {"status": "available", "locked": False, "enabled": restore}

    def toggle_window_locked(self) -> dict[str, object]:
        """切换锁定状态。"""

        return self.set_window_locked(not self._window_locked)

    @staticmethod
    def _operation_succeeded(result: object, *, enabled: bool) -> bool:
        if isinstance(result, Mapping):
            status = str(result.get("status", result.get("state", "")) or "").lower()
            reported = result.get("enabled")
            return status in {"available", "degraded", "requested", "completed"} and (
                not isinstance(reported, bool) or reported == enabled
            )
        state = str(getattr(getattr(result, "state", None), "value", "") or "").lower()
        return state in {"available", "degraded", "requested", "completed"}

    def _set_renderer_interaction_locked(self, locked: bool) -> None:
        setter = getattr(self.renderer, "set_interaction_locked", None)
        if callable(setter):
            try:
                setter(bool(locked))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("failed to update Web Live2D interaction lock", exc_info=True)

    def update_cursor_tracking(self) -> bool:
        """读取全局光标并投影到 Web Live2D，不依赖窗口输入事件。"""

        view = self.view
        if view is None:
            return False
        interaction = self._interaction_filter
        if (
            self._page_drag_origin is not None
            or self._page_system_move_active
            or self._page_dragging
            or self._renderer_dragging
            or (
                interaction is not None
                and (
                    getattr(interaction, "_drag_origin", None) is not None
                    or bool(getattr(interaction, "_system_move_active", False))
                )
            )
        ):
            # 窗口跟随指针移动时，屏幕光标映射到模型的局部坐标本应保持
            # 稳定；同时读取位置并 notify_window_moved 会清空平滑速度，
            # 与页面拖动高频 move 交错后造成注视抖动。拖动结束时由
            # _clear_page_drag 清除缓存，下一次 tick 再恢复追踪。
            return True
        try:
            from PySide6.QtGui import QCursor

            global_point = QCursor.pos()
            position = self._read_position(view)
            if position is None:
                return False
            previous_position = self._last_window_position
            self._last_window_position = position
            # 自主游走的窗口位置变化是已知几何事务；在稳定窗内不能把
            # 每一个小步误判成外部窗口跳变并清空注视速度，否则模型会在
            # 行走期间反复回中性姿态，表现为头部/发梢抽搐。稳定窗结束
            # 后仍会正常检测真实的外部移动。
            if (
                previous_position is not None
                and position != previous_position
                and not self._autonomous_geometry_active()
            ):
                notifier = getattr(self.renderer, "notify_window_moved", None)
                if callable(notifier):
                    try:
                        notifier()
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        logger.debug("failed to notify Live2D window movement", exc_info=True)
            width = max(1, int(view.width()))
            height = max(1, int(view.height()))
            local = self._map_global_cursor(view, global_point, position)
            if local is None:
                return False
            local_x = max(0, min(width - 1, local[0]))
            local_y = max(0, min(height - 1, local[1]))
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return False
        target = (local_x, local_y, width, height)
        if target == self._last_cursor_target:
            return True
        setter = getattr(self.renderer, "set_cursor_target", None)
        if not callable(setter):
            return False
        try:
            result = setter(local_x, local_y, width, height)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        # 渲染器可能在页面/模型重建期间显式返回 False；该坐标不能标记为
        # 已提交，否则后续定时 tick 会因命中缓存而永远不再重试。None 是
        # 旧渲染器的合法“无返回值”契约，仍按成功处理。
        if result is False:
            return False
        self._last_cursor_target = target
        return True

    def _clear_page_hover_state(self, view: object | None = None) -> None:
        """点击穿透后清除不会再收到 ``leave`` 的页面悬停状态。"""

        self._page_pointer_over = False
        interaction = self._interaction_filter
        hover_targets = getattr(interaction, "_hover_targets", None)
        if isinstance(hover_targets, set):
            hover_targets.clear()
        target = view if view is not None else self.view
        if target is not None:
            _set_cursor_shape(target, "ArrowCursor")

    def _update_cursor_tracking_from_page(self, payload: Mapping[str, object]) -> bool:
        """按页面桥携带的屏幕坐标立即刷新注视目标。

        页面 pointermove 已按 50ms 限频并带有 ``screen_x/screen_y``；直接
        使用该坐标可覆盖浏览器与系统光标短暂不同步的窗口，尤其适用于
        锁定/点击穿透模式。该路径只调用渲染器姿态接口，不触发对话或
        工具请求；字段缺失时交给全局光标轮询回退。
        """

        view = self.view
        if view is None or not isinstance(payload, Mapping):
            return False
        if (
            self._page_dragging
            or self._page_system_move_active
            or self._renderer_dragging
            or self._drag_transaction_active()
        ):
            return False
        try:
            raw_x = payload.get("screen_x")
            raw_y = payload.get("screen_y")
            if raw_x is None or raw_y is None:
                return False
            screen_x = float(raw_x)
            screen_y = float(raw_y)
            if not math.isfinite(screen_x) or not math.isfinite(screen_y):
                return False
            position = self._read_position(view)
            if position is None:
                return False
            width = max(1, int(view.width()))
            height = max(1, int(view.height()))
            local_x = max(0, min(width - 1, int(round(screen_x - position[0]))))
            local_y = max(0, min(height - 1, int(round(screen_y - position[1]))))
        except (AttributeError, OSError, OverflowError, TypeError, ValueError):
            return False
        target = (local_x, local_y, width, height)
        if target == self._last_cursor_target:
            return True
        setter = getattr(self.renderer, "set_cursor_target", None)
        if not callable(setter):
            return False
        try:
            result = setter(local_x, local_y, width, height)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        if result is False:
            return False
        self._last_cursor_target = target
        return True

    @staticmethod
    def _map_global_cursor(
        view: object,
        global_point: object,
        fallback_position: tuple[int, int],
    ) -> tuple[int, int] | None:
        """将屏幕光标转换为视图局部坐标，优先使用 Qt 的真实映射。"""

        mapper = getattr(view, "mapFromGlobal", None)
        if callable(mapper):
            try:
                local = mapper(global_point)
                return int(local.x()), int(local.y())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        try:
            return (
                int(global_point.x()) - int(fallback_position[0]),
                int(global_point.y()) - int(fallback_position[1]),
            )
        except (AttributeError, IndexError, OSError, TypeError, ValueError):
            return None

    @staticmethod
    def _read_position(view: object) -> tuple[int, int] | None:
        """读取顶层窗口坐标，供 Qt 原生窗口重建后恢复。"""

        getter = getattr(view, "pos", None)
        if not callable(getter):
            getter = getattr(view, "position", None)
        if not callable(getter):
            return None
        try:
            point = getter()
            return int(point.x()), int(point.y())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def _restore_position(
        self,
        view: object,
        position: tuple[int, int] | None,
        *,
        geometry_generation: int | None = None,
    ) -> None:
        """恢复切换窗口标志时被 Qt 重置的桌宠位置。"""

        if view is None or position is None:
            return

        def apply() -> None:
            if not self._geometry_is_current(geometry_generation):
                return
            if self._drag_transaction_active():
                return
            mover = getattr(view, "move", None)
            if callable(mover):
                try:
                    mover(*position)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                return
            setter = getattr(view, "setPosition", None)
            if not callable(setter):
                return
            try:
                setter(QPoint(*position))
            except TypeError:
                try:
                    setter(*position)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass

        apply()
        try:
            # 纯契约测试/无头核心可能只导入 PySide6 而没有事件循环；
            # 此时注册 Qt 定时器会在解释器退出阶段触发段错误。
            if _qt_available and QCoreApplication.instance() is not None:
                QTimer.singleShot(0, apply)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    def _log_window_flag_transaction(
        self,
        phase: str,
        *,
        generation: int,
        status: str,
        requested_topmost: bool | None = None,
        requested_click_through: bool | None = None,
        requested_lock: bool | None = None,
        verified: bool | None = None,
        reason_code: str = "",
        started_at: float | None = None,
    ) -> None:
        """输出不含窗口标题、路径和原始 XID 的窗口事务日志。"""

        fields: dict[str, object] = {
            "generation": max(0, int(generation)),
            "window_fingerprint": event_fingerprint(f"web-window:{id(self)}"),
        }
        if requested_topmost is not None:
            fields["requested_topmost"] = bool(requested_topmost)
        if requested_click_through is not None:
            fields["requested_click_through"] = bool(requested_click_through)
        if requested_lock is not None:
            fields["requested_lock"] = bool(requested_lock)
        if verified is not None:
            fields["verified"] = bool(verified)
        duration_ms = None
        if started_at is not None:
            duration_ms = max(0.0, (monotonic() - float(started_at)) * 1000.0)
        log_event(
            logger,
            f"gui.window_flags.transaction.{phase}",
            component="gui.qt6.web_host",
            status=status,
            operation_id=f"web-window:{id(self)}",
            request_id=f"web-window:{id(self)}:{int(generation)}",
            duration_ms=duration_ms,
            reason_code=str(reason_code or "none")[:64],
            fields=fields,
        )

    def _topmost_intent(self) -> tuple[bool, bool]:
        """返回当前期望置顶值，以及该值是否来自尚未完成的请求。"""

        pending = self._window_flag_transition_pending
        if pending is not None:
            return bool(pending), True
        if self._topmost_confirmed is not None:
            return bool(self._topmost_confirmed), False
        view = self.view
        actual = self._read_qt_topmost_flag(view) if view is not None else None
        return bool(actual), False

    def _cancel_pending_topmost_transition(self, *, reason_code: str, detail: str) -> bool:
        """终止旧置顶请求并释放它持有的 Shape/事件事务。"""

        pending = self._window_flag_transition_pending
        apply = self._window_flag_transition_apply
        cleanup = self._window_flag_transition_cleanup
        if pending is None and apply is None and cleanup is None:
            return False
        generation = self._window_flag_transition_generation
        view = self.view
        actual = self._read_qt_topmost_flag(view) if view is not None else None
        enabled = bool(actual) if actual is not None else bool(self._topmost_confirmed)
        result = self._window_flag_transition_result
        if isinstance(result, dict):
            result.clear()
            result.update({"status": "cancelled", "enabled": enabled, "detail": detail})
        self._window_flag_transition_generation += 1
        self._window_flag_transition_pending = None
        self._window_flag_transition_apply = None
        self._window_flag_transition_cleanup = None
        if callable(cleanup):
            try:
                cleanup()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._log_window_flag_transaction(
            "merged",
            generation=generation,
            status="cancelled",
            requested_topmost=bool(pending) if pending is not None else None,
            verified=False,
            reason_code=reason_code,
        )
        return True

    def _reconcile_topmost_after_input_transition(
        self,
        desired: bool,
        *,
        had_pending_request: bool,
        generation: int,
    ) -> None:
        """点击穿透切换后重新提交置顶并等待 EWMH 终态。"""

        if self._closing or self._shutdown or self.view is None:
            return
        if not desired and not had_pending_request:
            return
        result = self.set_always_on_top(bool(desired))
        status = str(result.get("status", "unknown") or "unknown")
        self._log_window_flag_transaction(
            "recovery",
            generation=generation,
            status=status,
            requested_topmost=bool(desired),
            requested_click_through=self._click_through_transition_pending,
            verified=status == "available" and bool(result.get("enabled")) == bool(desired),
            reason_code=(
                "topmost_requeued_after_input_transition"
                if had_pending_request
                else "topmost_reverified_after_input_transition"
            ),
        )

    def _input_transparency_active(self) -> bool | None:
        """读取 WebEngine 顶层窗口当前输入透明标志。"""

        view = self.view
        if view is None:
            return None
        try:
            from PySide6.QtCore import Qt

            flag = Qt.WindowType.WindowTransparentForInput
            handle = self._native_topmost_window(view)
            native_flags = getattr(handle, "flags", None)
            if callable(native_flags):
                return bool(native_flags() & flag)
            widget_flags = getattr(view, "windowFlags", None)
            if callable(widget_flags):
                return bool(widget_flags() & flag)
            return None
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return None

    @staticmethod
    def _is_qwebengine_view(view: object) -> bool:
        """识别真实 QWebEngineView，以便把原生子表面扫描移出按钮回调。"""

        try:
            return type(view).__module__ == "PySide6.QtWebEngineWidgets"
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _native_topmost_window(view: object) -> object | None:
        """返回承载 QWidget 的原生 QWindow；测试替身或未创建窗口时返回空。"""

        getter = getattr(view, "windowHandle", None)
        if not callable(getter):
            return None
        try:
            handle = getter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        return handle if handle is not None and _qt_object_alive(handle) else None

    @classmethod
    def _read_qt_topmost_flag(cls, view: object) -> bool | None:
        """优先回读原生 QWindow 旗标，兼容没有 windowHandle 的旧替身。"""

        try:
            from PySide6.QtCore import Qt

            flag = Qt.WindowType.WindowStaysOnTopHint
            handle = cls._native_topmost_window(view)
            native_flags = getattr(handle, "flags", None)
            if callable(native_flags):
                return bool(native_flags() & flag)
            widget_flags = getattr(view, "windowFlags", None)
            if callable(widget_flags):
                return bool(widget_flags() & flag)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            return None
        return None

    @classmethod
    def _set_qt_topmost_flag(
        cls,
        view: object,
        requested: bool,
        *,
        require_native: bool = False,
    ) -> str:
        """不重建 WebEngine QWidget 地切换置顶，返回实际采用的 Qt 路径。"""

        from PySide6.QtCore import Qt

        flag = Qt.WindowType.WindowStaysOnTopHint
        handle = cls._native_topmost_window(view)
        native_setter = getattr(handle, "setFlag", None)
        if callable(native_setter):
            native_setter(flag, requested)
            return "qwindow"
        if require_native:
            raise RuntimeError("native QWindow disappeared before topmost transition")

        single_setter = getattr(view, "setWindowFlag", None)
        if callable(single_setter):
            single_setter(flag, requested)
            return "qwidget-single"

        flags_setter = getattr(view, "setWindowFlags", None)
        flags_getter = getattr(view, "windowFlags", None)
        if not callable(flags_setter) or not callable(flags_getter):
            raise AttributeError("web view does not expose a window flag setter")
        current_flags = flags_getter()
        flags_setter((current_flags | flag) if requested else (current_flags & ~flag))
        return "qwidget-flags"

    def _qt_uses_x11(self) -> bool:
        """确认当前 Qt 顶层窗口是否由 xcb 后端承载。"""

        try:
            from PySide6.QtGui import QGuiApplication

            application = QGuiApplication.instance()
            platform_name = (
                str(application.platformName() or "").strip().lower()
                if application is not None
                else ""
            )
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            platform_name = ""

        if platform_name:
            return platform_name == "xcb"
        configured = str(os.environ.get("QT_QPA_PLATFORM", "") or "").strip().lower()
        if configured:
            return configured.split(":", 1)[0] == "xcb"
        backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
        return backend == "x11" and bool(os.environ.get("DISPLAY"))

    @classmethod
    def _x11_topmost_window_id(cls, view: object) -> int | None:
        """读取 QWindow 的 XID，避免 QWidget 重建路径创建另一个顶层窗口。"""

        handle = cls._native_topmost_window(view)
        for target in (handle, view):
            getter = getattr(target, "winId", None)
            if not callable(getter):
                continue
            try:
                value = int(getter())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def _read_x11_topmost_state(self, view: object) -> bool | None:
        """回读 EWMH 置顶属性；无法连接或窗口已销毁时返回未知。"""

        if not self._qt_uses_x11():
            return None
        display_name = str(os.environ.get("DISPLAY", "") or "").strip()
        window_id = self._x11_topmost_window_id(view)
        if not display_name or window_id is None:
            return None
        connection = None
        try:
            from Xlib import X, display, error

            connection = display.Display(display_name)
            errors = error.CatchError(error.BadWindow)
            connection.set_error_handler(errors)
            window = connection.create_resource_object("window", window_id)
            state_atom = connection.intern_atom("_NET_WM_STATE", only_if_exists=False)
            above_atom = connection.intern_atom("_NET_WM_STATE_ABOVE", only_if_exists=False)
            stays_atom = connection.intern_atom("_NET_WM_STATE_STAYS_ON_TOP", only_if_exists=False)
            value = window.get_full_property(state_atom, X.AnyPropertyType)
            connection.sync()
            if errors.get_error() is not None:
                return None
            atoms = {
                int(item)
                for item in tuple(getattr(value, "value", ()) or ())
                if isinstance(item, int)
            }
            return int(above_atom) in atoms or int(stays_atom) in atoms
        except Exception:
            logger.debug("X11 topmost EWMH readback is unavailable")
            return None
        finally:
            closer = getattr(connection, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    def _schedule_qwebengine_topmost(
        self,
        view: object,
        requested: bool,
        was_visible: bool | None,
        position: tuple[int, int] | None,
        mask_paused: bool,
        geometry_generation: int,
        native_window_path: bool,
    ) -> dict[str, object]:
        """异步切换原生 QWindow 置顶，并确认窗口管理器的最终状态。"""

        self._cancel_pending_topmost_transition(
            reason_code="superseded_by_new_topmost_request",
            detail="置顶请求已由更新的窗口事务替代",
        )
        self._window_flag_transition_generation += 1
        generation = self._window_flag_transition_generation
        self._window_flag_transition_pending = requested
        transition_started_at = monotonic()

        def platform_result(*, probe: bool = True) -> tuple[str, str]:
            backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
            if probe:
                readback = getattr(self.platform, "always_on_top_status", None)
                native_status = None
                if callable(readback):
                    try:
                        native_status = readback(view)
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        native_status = None
                if isinstance(native_status, Mapping):
                    normalized = str(native_status.get("status", "") or "").strip().lower()
                    if normalized in {"available", "degraded", "unavailable"}:
                        status = normalized
                        raw_detail = native_status.get("detail", "")
                    else:
                        status, raw_detail = self._platform_capability_status("always_on_top")
                else:
                    status, raw_detail = self._platform_capability_status("always_on_top")
            elif backend == "wayland":
                status, raw_detail = (
                    "degraded",
                    "Wayland compositor may override the topmost request",
                )
            else:
                status, raw_detail = "available", ""
            detail = str(raw_detail or "").strip()
            if status == "degraded":
                if backend == "wayland" and "wayland" not in detail.casefold():
                    detail = "Wayland compositor may override the topmost request" + (
                        f"; {detail}" if detail else ""
                    )
                elif not detail:
                    detail = "窗口置顶请求已提交，最终状态由桌面合成器决定"
            return status, detail

        platform_status, detail = platform_result(probe=False)
        drag_wait_deadline = monotonic() + _WINDOW_FLAG_DRAG_WAIT_TIMEOUT_SECONDS
        result: dict[str, object] = {
            "status": "requested",
            "enabled": requested,
            "detail": "窗口置顶切换已提交，正在等待原生窗口确认",
        }
        if platform_status != "available":
            result["status"] = platform_status
            if detail:
                result["detail"] = detail
        self._window_flag_transition_result = result

        cleanup_finished = False

        def release_cleanup() -> None:
            nonlocal cleanup_finished
            if cleanup_finished:
                return
            cleanup_finished = True
            if self._window_flag_transition_cleanup is release_cleanup:
                self._window_flag_transition_cleanup = None
            self._resume_surface_mask_after_window_rebuild(view, mask_paused)

        self._window_flag_transition_cleanup = release_cleanup
        self._log_window_flag_transaction(
            "start",
            generation=generation,
            status="started",
            requested_topmost=requested,
            requested_click_through=self._click_through_transition_pending,
            requested_lock=self._window_locked,
            reason_code="topmost_request",
        )

        def read_actual_flag() -> bool | None:
            return self._read_qt_topmost_flag(view)

        def finish(
            status: str,
            enabled: bool,
            message: str = "",
            *,
            reason_code: str = "completed",
        ) -> None:
            if generation != self._window_flag_transition_generation:
                return
            release_cleanup()
            confirmed = bool(enabled)
            result.clear()
            result.update({"status": status, "enabled": confirmed})
            if message:
                result["detail"] = message
            self._topmost_confirmed = confirmed
            self._window_flag_transition_pending = None
            self._window_flag_transition_result = result
            self._log_window_flag_transaction(
                "terminal",
                generation=generation,
                status=status,
                requested_topmost=requested,
                requested_click_through=self._click_through_transition_pending,
                requested_lock=self._window_locked,
                verified=(status == "available" and confirmed == requested),
                reason_code=reason_code,
                started_at=transition_started_at,
            )

        def still_current() -> bool:
            return bool(
                not self._shutdown
                and self.view is view
                and generation == self._window_flag_transition_generation
                and _qt_object_alive(view)
            )

        def finish_platform(actual: bool) -> None:
            status, current_detail = platform_result()
            finish(
                status if status != "available" else "available",
                actual,
                current_detail,
                reason_code="platform_confirmed",
            )

        def verify_x11(attempt: int = 0) -> None:
            if not still_current():
                return
            actual = self._read_x11_topmost_state(view)
            if actual is not None and actual == requested:
                finish_platform(actual)
                return
            next_attempt = attempt + 1
            if next_attempt < len(_X11_TOPMOST_VERIFY_DELAYS_MS):
                delay = int(_X11_TOPMOST_VERIFY_DELAYS_MS[next_attempt])
                try:
                    QTimer.singleShot(delay, lambda: verify_x11(next_attempt))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    fallback = self._read_qt_topmost_flag(view)
                    finish(
                        "unavailable",
                        bool(fallback),
                        "无法继续安排窗口管理器置顶回读",
                        reason_code="ewmh_verify_schedule_failed",
                    )
                return
            self._log_window_flag_transaction(
                "timeout",
                generation=generation,
                status="unavailable",
                requested_topmost=requested,
                requested_click_through=self._click_through_transition_pending,
                requested_lock=self._window_locked,
                verified=False,
                reason_code="ewmh_verify_timeout",
                started_at=transition_started_at,
            )
            if actual is None:
                finish(
                    "unavailable",
                    False,
                    "无法回读 X11 窗口的 _NET_WM_STATE_ABOVE，置顶状态未确认",
                    reason_code="ewmh_unreadable",
                )
            else:
                finish(
                    "unavailable",
                    actual,
                    "窗口管理器未应用请求的 _NET_WM_STATE_ABOVE 状态",
                    reason_code="ewmh_not_applied",
                )

        def refresh_after_apply(path: str) -> None:
            if not still_current():
                return
            try:
                if (
                    was_visible is True
                    and callable(getattr(view, "isVisible", None))
                    and not view.isVisible()
                ):
                    view.show()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
            if path != "qwindow":
                self._refresh_interaction_targets()
                self._schedule_interaction_refresh()

        def apply() -> None:
            if self._window_flag_transition_apply is not apply:
                return
            if not still_current():
                if generation == self._window_flag_transition_generation:
                    finish(
                        "cancelled",
                        self.is_always_on_top(),
                        "窗口已关闭或置顶请求已取消",
                        reason_code="host_closed",
                    )
                self._window_flag_transition_apply = None
                return
            if self._drag_transaction_active():
                if monotonic() >= drag_wait_deadline:
                    self._window_flag_transition_apply = None
                    actual = read_actual_flag()
                    self._log_window_flag_transaction(
                        "timeout",
                        generation=generation,
                        status="cancelled",
                        requested_topmost=requested,
                        requested_click_through=self._click_through_transition_pending,
                        requested_lock=self._window_locked,
                        verified=False,
                        reason_code="drag_wait_timeout",
                        started_at=transition_started_at,
                    )
                    finish(
                        "cancelled",
                        actual if actual is not None else False,
                        "等待拖动释放超时，已取消置顶切换",
                        reason_code="drag_wait_timeout",
                    )
                    return
                try:
                    QTimer.singleShot(_WINDOW_FLAG_DRAG_RETRY_INTERVAL_MS, apply)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    self._window_flag_transition_apply = None
                    finish(
                        "unavailable",
                        self.is_always_on_top(),
                        "窗口置顶暂时无法切换，请稍后重试",
                        reason_code="drag_retry_schedule_failed",
                    )
                return

            self._window_flag_transition_apply = None
            try:
                path = self._set_qt_topmost_flag(
                    view,
                    requested,
                    require_native=native_window_path,
                )
                backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
                if backend == "windows":
                    native_setter = getattr(self.platform, "set_always_on_top", None)
                    if callable(native_setter):
                        native_result = native_setter(view, requested)
                        if inspect.isawaitable(native_result):
                            closer = getattr(native_result, "close", None)
                            if callable(closer):
                                closer()
                            native_result = {
                                "status": "unavailable",
                                "detail": "Windows 原生置顶请求未在 Qt 主线程完成",
                            }
                        native_state = getattr(
                            getattr(native_result, "state", None),
                            "value",
                            None,
                        )
                        native_detail = getattr(native_result, "detail", "")
                        if isinstance(native_result, Mapping):
                            native_state = native_result.get(
                                "status",
                                native_result.get("state", "unavailable"),
                            )
                            native_detail = native_result.get(
                                "detail",
                                native_result.get("reason", ""),
                            )
                        native_state = str(native_state or "unavailable").strip().lower()
                        if native_state not in {
                            "available",
                            "degraded",
                            "requested",
                            "completed",
                        }:
                            finish(
                                "unavailable",
                                bool(read_actual_flag()),
                                str(native_detail or "Windows 原生置顶请求失败"),
                                reason_code="win32_topmost_failed",
                            )
                            return
                actual = read_actual_flag()
                if actual is None:
                    finish(
                        "unavailable",
                        False,
                        "无法回读原生 QWindow 置顶旗标，置顶状态未确认",
                        reason_code="qt_flag_unreadable",
                    )
                elif actual != requested:
                    finish(
                        "unavailable",
                        actual,
                        "Qt 原生窗口未保留请求的置顶旗标",
                        reason_code="qt_flag_not_retained",
                    )
                else:
                    self._restore_position(
                        view,
                        position,
                        geometry_generation=geometry_generation,
                    )
                    if self._qt_uses_x11() and self._x11_topmost_window_id(view) is not None:
                        delay = int(_X11_TOPMOST_VERIFY_DELAYS_MS[0])
                        try:
                            QTimer.singleShot(delay, verify_x11)
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            finish(
                                "unavailable",
                                actual,
                                "无法安排窗口管理器置顶回读",
                                reason_code="ewmh_verify_schedule_failed",
                            )
                    else:
                        finish_platform(actual)
                try:
                    QTimer.singleShot(0, lambda: refresh_after_apply(path))
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    refresh_after_apply(path)
            except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
                finish(
                    "unavailable",
                    bool(read_actual_flag()),
                    "窗口置顶暂时无法切换，请稍后重试",
                    reason_code="qt_flag_update_failed",
                )
                logger.debug("QWebEngine topmost transition failed", exc_info=True)
            finally:
                release_cleanup()

        self._window_flag_transition_apply = apply
        try:
            QTimer.singleShot(0, apply)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            apply()
        return result

    def set_always_on_top(self, enabled: bool) -> dict[str, object]:
        """切换 WebEngine 顶层窗口的置顶标志并回读结果。"""

        requested = bool(enabled)
        view = self.view
        if view is None:
            self._cancel_pending_topmost_transition(
                reason_code="view_unavailable",
                detail="Web 视图不可用，已终止置顶事务",
            )
            result = {
                "status": "unavailable",
                "enabled": False,
                "reason": "web view is unavailable",
            }
            self._window_flag_transition_pending = None
            self._window_flag_transition_result = result
            return result
        if self._drag_transaction_active():
            # 拖动期间拒绝新的置顶请求时，同时使已排队的旧回调失效；
            # 否则旧 QTimer 仍可能在释放后写回过期窗口旗标。
            self._cancel_pending_topmost_transition(
                reason_code="user_interaction_active",
                detail="用户拖动期间已取消置顶事务",
            )
            result = {
                "status": "cancelled",
                "enabled": self.is_always_on_top(),
                "reason": "user interaction active",
            }
            self._window_flag_transition_pending = None
            self._window_flag_transition_result = result
            return result
        try:
            from PySide6.QtCore import Qt

            flag = Qt.WindowType.WindowStaysOnTopHint
            was_visible = (
                bool(view.isVisible()) if callable(getattr(view, "isVisible", None)) else None
            )
            position = self._read_position(view)
            native_webengine = (
                self._is_qwebengine_view(view)
                and _qt_available
                and QCoreApplication.instance() is not None
            )
            native_window = self._native_topmost_window(view) if native_webengine else None
            native_setter = getattr(native_window, "setFlag", None)
            setter = getattr(view, "setWindowFlag", None)
            set_single_flag = callable(setter)
            if not set_single_flag:
                setter = getattr(view, "setWindowFlags", None)
            if not callable(native_setter) and not callable(setter):
                raise AttributeError("web view does not expose a window flag setter")

            # QWebEngineView.setWindowFlag() 会销毁并重建 Chromium 顶层，
            # KWin/Xwayland 中新 XID 可能只保留 Qt flag 而丢失 EWMH 置顶。
            # 已有 QWindow 时直接切换其原生旗标，不暂停 Shape、不重挂事件。
            native_window_path = native_webengine and callable(native_setter)
            mask_paused = (
                False
                if native_window_path
                else self._pause_surface_mask_for_window_rebuild(
                    view,
                    defer_input_shape=native_webengine,
                )
            )
            if not native_window_path:
                self._mark_interaction_rebuild()
            geometry_generation = self._geometry_generation
            if native_webengine:
                return self._schedule_qwebengine_topmost(
                    view,
                    requested,
                    was_visible,
                    position,
                    mask_paused,
                    geometry_generation,
                    native_window_path,
                )
            if not set_single_flag:
                try:
                    current = view.windowFlags()
                    setter((current | flag) if requested else (current & ~flag))
                except Exception:
                    self._resume_surface_mask_after_window_rebuild(view, mask_paused)
                    raise
            else:
                try:
                    setter(flag, requested)
                except Exception:
                    self._resume_surface_mask_after_window_rebuild(view, mask_paused)
                    raise
            if was_visible and callable(getattr(view, "isVisible", None)) and not view.isVisible():
                view.show()
            flags = getattr(view, "windowFlags", None)
            if not callable(flags):
                raise AttributeError("web view does not expose window flags")
            actual = bool(flags() & flag)
            self._restore_position(
                view,
                position,
                geometry_generation=geometry_generation,
            )
            if actual != requested:
                self._resume_surface_mask_after_window_rebuild(view, mask_paused)
                result = {
                    "status": "unavailable",
                    "enabled": actual,
                    "reason": "window system did not retain the requested topmost flag",
                }
                self._window_flag_transition_pending = None
                self._window_flag_transition_result = result
                return result
            # QWidget.setWindowFlag() 可以重建 WebEngine 的原生 viewport；旧
            # 的事件过滤器会随之失效，导致置顶切换后拖动和右键菜单同时失灵。
            # 立即重扫一次，并在 Qt 处理完重建事件后再补扫一次。
            if self._is_qwebengine_view(view) and _qt_available:
                # findChildren()/事件过滤器重挂载可能等待 Chromium 子表面
                # 销毁；不要在控制台点击回调中同步执行，避免窗口短暂无响应。
                try:
                    QTimer.singleShot(
                        0,
                        lambda: (
                            self._refresh_interaction_targets(),
                            self._schedule_interaction_refresh(),
                        ),
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    self._refresh_interaction_targets()
                    self._schedule_interaction_refresh()
            else:
                self._refresh_interaction_targets()
                self._schedule_interaction_refresh()
            self._resume_surface_mask_after_window_rebuild(view, mask_paused)
            platform_status, detail = self._platform_capability_status("always_on_top")
            detail = str(detail or "").strip()
            backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
            if (
                platform_status == "degraded"
                and backend == "wayland"
                and "wayland" not in detail.casefold()
            ):
                detail = "Wayland compositor may override the topmost request" + (
                    f"; {detail}" if detail else ""
                )
            result = (
                {"status": platform_status, "enabled": actual, "detail": detail}
                if platform_status != "available"
                else {"status": "available", "enabled": actual}
            )
            self._topmost_confirmed = actual
            self._window_flag_transition_pending = None
            self._window_flag_transition_result = result
            return result
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if "mask_paused" in locals():
                self._resume_surface_mask_after_window_rebuild(view, mask_paused)
            logger.debug("Web host topmost update failed: %s", type(exc).__name__)
            result = {
                "status": "unavailable",
                "enabled": False,
                "reason": "窗口置顶暂时无法切换，请稍后重试",
            }
            self._window_flag_transition_pending = None
            self._window_flag_transition_result = result
            return result

    def _platform_capability_status(self, name: str, *, probe: bool = True) -> tuple[str, str]:
        """把窗口平台能力映射为宿主可展示的状态。"""

        platform = self.platform
        if platform is None:
            return "available", ""
        backend = str(getattr(platform, "backend", "") or "").strip().lower()
        now = monotonic()
        cached = self._topmost_capability_cache
        if name == "always_on_top" and cached is not None:
            cached_backend, cached_status, cached_detail, cached_at = cached
            if cached_backend == backend and now - cached_at < 1.0:
                return cached_status, cached_detail
        probe_handler = getattr(platform, "probe", None)
        if probe and callable(probe_handler):
            try:
                snapshot = probe_handler()
                capability_getter = getattr(snapshot, "capability", None)
                capability = capability_getter(name) if callable(capability_getter) else None
                if capability is not None:
                    state = getattr(getattr(capability, "state", None), "value", None)
                    detail = str(getattr(capability, "detail", "") or "")
                    normalized = str(state or "unknown").strip().lower()
                    if normalized in {"available", "degraded", "unavailable"}:
                        if name == "always_on_top":
                            self._topmost_capability_cache = (backend, normalized, detail, now)
                        return normalized, detail
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        if backend == "wayland":
            status, detail = "degraded", "Wayland compositor may override the topmost request"
        else:
            status, detail = "available", ""
        if name == "always_on_top":
            self._topmost_capability_cache = (backend, status, detail, now)
        return status, detail

    def is_always_on_top(self) -> bool:
        """返回已确认的窗口管理器状态，切换中返回本次请求值。"""

        if self._window_flag_transition_pending is not None:
            return bool(self._window_flag_transition_pending)
        if self._topmost_confirmed is not None:
            return bool(self._topmost_confirmed)
        view = self.view
        actual = self._read_qt_topmost_flag(view) if view is not None else None
        return bool(actual) if actual is not None else False

    def always_on_top_status(self) -> dict[str, object]:
        """回读置顶状态及平台边界，避免异步请求长期停留在 requested。"""

        pending = self._window_flag_transition_pending
        cached = self._window_flag_transition_result
        if pending is not None and isinstance(cached, Mapping):
            return dict(cached)
        actual = self.is_always_on_top()
        # 状态轮询运行在 Qt 主线程，不能每 40ms 触发 Xlib/平台探针；
        # 置顶切换完成时已在异步回调缓存真实能力，未缓存时使用明确的
        # Wayland/X11 默认边界，下一次切换再刷新探针。
        status, detail = self._platform_capability_status("always_on_top", probe=False)
        text = str(detail or "").strip()
        backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
        if status == "degraded" and backend == "wayland" and "wayland" not in text.casefold():
            text = "Wayland compositor may override the topmost request" + (
                f"; {text}" if text else ""
            )
        if isinstance(cached, Mapping):
            cached_enabled = cached.get("enabled")
            cached_status = str(cached.get("status", "") or "").strip().lower()
            if cached_enabled == actual and cached_status in {
                "available",
                "degraded",
                "unavailable",
            }:
                return dict(cached)
        result: dict[str, object] = {"status": status, "enabled": actual}
        if text:
            result["detail"] = text
        self._window_flag_transition_result = result
        return dict(result)

    def toggle_always_on_top(self) -> dict[str, object]:
        """切换 WebEngine 顶层窗口置顶状态。"""

        return self.set_always_on_top(not self.is_always_on_top())

    def toggle_visibility(self) -> dict[str, object]:
        """切换 WebEngine 桌宠可见性。"""

        view = self.view
        if view is None:
            return {"status": "unavailable", "visible": False, "reason": "web view is unavailable"}
        try:
            if view.isVisible():
                # 隐藏窗口不会可靠地产生页面 ``leave``；先清理页面/原生
                # 悬停状态和未完成拖动，避免恢复显示后自主行为永久停住。
                self._clear_page_drag()
                self._clear_page_hover_state(view)
                view.hide()
            else:
                view.show()
                self._schedule_interaction_refresh()
            visible = bool(view.isVisible())
            return {"status": "available", "visible": visible}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Web host visibility update failed: %s", type(exc).__name__)
            return {
                "status": "unavailable",
                "visible": False,
                "reason": "桌宠显示状态暂时无法切换，请稍后重试",
            }

    def begin_click_through_override(self) -> object:
        """按住恢复快捷键时取消 WebEngine 点击穿透。"""

        previous = bool(getattr(self, "_click_through", False))
        if getattr(self, "_click_through_override_active", False):
            return {"status": "available", "enabled": False, "temporary": True}
        self._click_through_override_active = True
        self._click_through_before_override = previous
        if not previous:
            return {"status": "available", "enabled": False, "temporary": True}
        result = self.set_click_through(False)
        status = str(result.get("status", "")) if isinstance(result, Mapping) else ""
        if status in {"unavailable", "unknown"}:
            self._click_through_override_active = False
            self._click_through_before_override = False
        return result

    def end_click_through_override(self) -> object:
        """释放恢复快捷键时恢复 WebEngine 点击穿透状态。"""

        if not getattr(self, "_click_through_override_active", False):
            return {"status": "available", "enabled": False}
        previous = bool(getattr(self, "_click_through_before_override", False))
        self._click_through_override_active = False
        self._click_through_before_override = False
        return (
            self.set_click_through(True) if previous else {"status": "available", "enabled": False}
        )

    def install_interaction(
        self,
        context_menu_callback: Callable[[object], None] | None = None,
        double_click_callback: Callable[[], None] | None = None,
        click_callback: Callable[[Mapping[str, object]], None] | None = None,
        enter_callback: Callable[[], None] | None = None,
    ) -> bool:
        """安装 WebEngine 拖动、菜单、双击和普通点击互动。"""

        view = self.view
        filter_type = _WebInteractionFilter
        callback_setter = getattr(self.renderer, "set_interaction_callback", None)
        if view is None or (filter_type is None and not callable(callback_setter)):
            return False
        if self._interaction_filter is not None:
            return True
        try:
            self._context_menu_callback = context_menu_callback
            self._double_click_callback = double_click_callback
            self._click_callback = click_callback
            self._enter_callback = enter_callback
            page_bridge_available = callable(callback_setter)
            self._page_bridge_ready_setter = None
            if page_bridge_available:
                callback_setter(self._handle_page_interaction)
                ready_setter = getattr(self.renderer, "set_page_bridge_ready_callback", None)
                if callable(ready_setter):
                    self._page_bridge_ready_setter = ready_setter
                    ready_setter(self._handle_page_bridge_ready)
                    page_bridge_ready = bool(getattr(self.renderer, "page_bridge_ready", False))
                else:
                    # 旧渲染器没有握手状态契约时，沿用原有行为，避免
                    # 把兼容测试替身误判为未就绪而重复安装过滤器。
                    page_bridge_ready = True
                self._page_bridge_enabled = page_bridge_ready
            else:
                self._page_bridge_enabled = False
            click_setter = getattr(self.renderer, "set_click_callback", None)
            if callable(click_setter):
                click_setter(self._handle_renderer_click)
                self._renderer_click_callback_active = True
            else:
                self._renderer_click_callback_active = False
            reload_setter = getattr(self.renderer, "set_model_reload_callback", None)
            if callable(reload_setter):
                reload_setter(self._on_renderer_model_reload)
                self._renderer_model_reload_callback_active = True
            else:
                self._renderer_model_reload_callback_active = False
            # 页面桥可用时让 Chromium 直接接收完整 press/move/release 序列；
            # 同时在 Qt 过滤器上拦截会造成重复 click，并吞掉拖动事件。过滤器
            # 只作为页面桥不可用、旧 WebEngine viewport 或测试替身的回退。
            if filter_type is not None and not self._page_bridge_enabled:
                self._install_native_interaction_filter(filter_type, view)
            # WebEngine 的 ResizeObserver 只能看到 Chromium 已提交的布局；
            # 外层窗口被窗口管理器直接调整时，先由 Qt 尺寸回读主动同步，
            # 避免短暂沿用上一档 viewport 造成模型缩放/命中错位。
            self._start_viewport_sync()
            return True
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            if self._viewport_sync_timer is not None:
                try:
                    self._viewport_sync_timer.stop()
                except (AttributeError, RuntimeError):
                    pass
                self._viewport_sync_timer = None
            self._last_viewport_size = None
            self._interaction_filter = None
            self._interaction_targets = []
            self._page_bridge_enabled = False
            self._context_menu_callback = None
            self._double_click_callback = None
            self._click_callback = None
            self._enter_callback = None
            self._renderer_click_callback_active = False
            self._renderer_model_reload_callback_active = False
            if callable(callback_setter):
                try:
                    callback_setter(None)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            ready_setter = self._page_bridge_ready_setter
            self._page_bridge_ready_setter = None
            if ready_setter is not None:
                try:
                    ready_setter(None)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            click_setter = getattr(self.renderer, "set_click_callback", None)
            if callable(click_setter):
                try:
                    click_setter(None)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            reload_setter = getattr(self.renderer, "set_model_reload_callback", None)
            if callable(reload_setter):
                try:
                    reload_setter(None)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            return False

    def _install_native_interaction_filter(self, filter_type: object, view: object) -> None:
        """在页面桥未握手时挂载 Qt 原生输入回退。"""

        if self._interaction_filter is not None or not callable(filter_type):
            return
        interaction_filter = filter_type(
            view,
            self._context_menu_callback,
            self._double_click_callback,
            self._register_interaction_target,
            self._emit_click,
            self._native_model_hit_test,
            self._set_renderer_dragging,
            self._notify_interaction_started,
            self._enter_callback,
        )
        self._interaction_filter = interaction_filter
        self._refresh_interaction_targets()

    def _native_model_hit_test(self, point: QPoint | None) -> bool | None:
        """为 WebEngine 原生回退提供 fail-closed 的模型命中结果。

        页面桥接成功后不会走该路径。桥接尚未握手时，真实 Web Live2D
        没有同步的 Python 命中 API，宁可不拦截鼠标，也不能把整个透明视口
        当成可拖动区域；旧渲染器/测试替身没有 ``model_ready`` 属性则保留
        兼容的 ``None`` 结果。
        """

        renderer = self.renderer
        if self._window_locked:
            return False
        hit_test = getattr(renderer, "hit_test", None)
        if callable(hit_test) and point is not None:
            try:
                return bool(hit_test(point.x(), point.y()))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
        if isinstance(getattr(renderer, "model_ready", None), bool):
            return False
        return None

    def _remove_native_interaction_filter(self) -> None:
        """切换到页面桥时移除旧过滤器，防止重复 click 和拖动竞争。"""

        interaction_filter = self._interaction_filter
        if interaction_filter is None:
            return
        # 原生过滤器的单击回调有 450ms 延迟；桥接握手恰好发生在这段窗口
        # 内时必须使旧 token 失效，否则页面桥的 click 会与旧回调重复。
        click_token = getattr(interaction_filter, "_click_token", None)
        if isinstance(click_token, int):
            try:
                setattr(interaction_filter, "_click_token", click_token + 1)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        clear_drag = getattr(interaction_filter, "_clear_drag", None)
        if callable(clear_drag):
            try:
                clear_drag()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        for target in tuple(self._interaction_targets):
            try:
                target.removeEventFilter(interaction_filter)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        # 过滤器即将被彻底移除，旧对象不会再发送 ``Leave``；其 id 若继续
        # 留在 ``_hover_targets`` 会让行为服务永久认为用户正在悬停，导致
        # 自主行为卡在 paused。切换到页面桥时丢弃旧视口快照。
        hover_targets = getattr(interaction_filter, "_hover_targets", None)
        if isinstance(hover_targets, set):
            hover_targets.clear()
        self._interaction_targets = []
        self._interaction_filter = None

    def _handle_page_bridge_ready(self, ready: bool) -> None:
        """按 WebChannel 握手状态在原生过滤器与页面桥之间切换。"""

        self._invalidate_geometry()
        self._apply_no_focus_to_view_tree()
        if bool(ready):
            self._page_bridge_enabled = True
            self._remove_native_interaction_filter()
            self._clear_page_drag()
            self._schedule_surface_mask_retries()
            return
        self._page_bridge_enabled = False
        self._clear_page_drag()
        if self.view is not None and _WebInteractionFilter is not None:
            try:
                self._install_native_interaction_filter(_WebInteractionFilter, self.view)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("failed to install native interaction fallback", exc_info=True)

    def _refresh_interaction_targets(self) -> None:
        """递归覆盖 WebEngine 动态创建的 Chromium 视口。"""

        interaction_filter = self._interaction_filter
        view = self.view
        if interaction_filter is None or view is None:
            self._apply_no_focus_to_view_tree()
            return
        self._apply_no_focus_to_view_tree()
        # 先清理旧目标上的过滤器；QWebEngine 切换原生窗口时旧对象可能
        # 仍暂存一小段时间，反复安装会导致一次事件被处理多次。
        for target in tuple(self._interaction_targets):
            try:
                target.removeEventFilter(interaction_filter)
            except (AttributeError, RuntimeError, TypeError):
                pass
        # Qt 重建 Chromium 子表面时通常不会补发 Leave。先保留悬停集合，
        # 待新的目标列表建立后只剔除真正销毁的对象；同一 viewport 重挂载
        # 时保留 hover，避免指针未移动却错误恢复自主行为。
        self._interaction_targets = []
        interaction_filter.install_on(view)
        targets: list[object] = [view]
        finder = getattr(view, "findChildren", None)
        if callable(finder):
            try:
                targets.extend(tuple(finder(QObject)))
            except (AttributeError, RuntimeError, TypeError):
                pass
        # 去重但保持稳定顺序，便于 shutdown 和测试检查。
        seen: set[int] = {id(target) for target in self._interaction_targets}
        for target in targets:
            marker = id(target)
            if marker in seen or target is interaction_filter:
                continue
            seen.add(marker)
            self._interaction_targets.append(target)
        hover_targets = getattr(interaction_filter, "_hover_targets", None)
        if isinstance(hover_targets, set):
            active_ids = {id(target) for target in self._interaction_targets}
            hover_targets.intersection_update(active_ids)

    def _apply_no_focus_to_view_tree(self) -> None:
        """在页面桥模式下也持续清理动态 Chromium 子表面的焦点能力。"""

        if not _qt_available:
            return
        view = self.view
        if view is None:
            return
        _set_no_focus(view)
        finder = getattr(view, "findChildren", None)
        if not callable(finder):
            return
        try:
            children = tuple(finder(QObject))
        except (AttributeError, RuntimeError, TypeError):
            return
        for child in children:
            if child is not view:
                _set_no_focus(child)

    def _schedule_interaction_refresh(self) -> None:
        """在 Qt 完成顶层窗口重建后重新挂载 Chromium 事件过滤器。"""

        if not _qt_available or self._interaction_filter is None or self.view is None:
            return
        geometry_generation = self._geometry_generation

        def refresh() -> None:
            if self._drag_transaction_active():
                # 顶层标志切换与用户拖动可能同时发生；当前手势期间不能
                # 重挂载 Chromium 过滤器，否则会丢失 release。不要一次性
                # 放弃刷新，延迟到手势结束后再以“当前”几何代次重新
                # 调度；拖动开始本身会 invalidate 旧代次，不能继续复用
                # 这里捕获的 stale generation。
                try:
                    QTimer.singleShot(120, self._schedule_interaction_refresh)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                return
            if not self._geometry_is_current(geometry_generation):
                return
            self._refresh_interaction_targets()

        try:
            QTimer.singleShot(0, refresh)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # 无事件循环或测试替身没有 QTimer 时，立即扫描已经足够。
            return

    def _register_interaction_target(self, target: object) -> None:
        """记录动态 Chromium 子对象，确保销毁时能移除过滤器。"""

        if target is self._interaction_filter or any(
            item is target for item in self._interaction_targets
        ):
            return
        self._interaction_targets.append(target)

    async def _click_through_after_dispatch(
        self,
        capability: object,
        enabled: bool,
        position: tuple[int, int] | None = None,
        *,
        surface_mask_paused: bool = False,
        geometry_generation: int | None = None,
        transition_generation: int,
        transition_started_at: float,
        desired_topmost: bool,
        topmost_was_pending: bool,
        native_input_path: bool,
    ) -> dict[str, object]:
        timed_out = False
        superseded = False
        try:
            result = await asyncio.wait_for(  # type: ignore[arg-type]
                capability,
                timeout=_WINDOW_FLAG_DISPATCH_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            current = transition_generation == self._click_through_transition_generation
            if current:
                self._click_through_transition_pending = None
            self._log_window_flag_transaction(
                "terminal",
                generation=transition_generation,
                status="cancelled",
                requested_topmost=desired_topmost,
                requested_click_through=enabled,
                requested_lock=self._window_locked,
                verified=False,
                reason_code="dispatch_cancelled",
                started_at=transition_started_at,
            )
            if current:
                self._reconcile_topmost_after_input_transition(
                    desired_topmost,
                    had_pending_request=topmost_was_pending,
                    generation=transition_generation,
                )
            else:
                await self._restore_latest_click_through_after_stale_dispatch(
                    stale_generation=transition_generation,
                )
            raise
        except TimeoutError:
            timed_out = True
            superseded = transition_generation != self._click_through_transition_generation
            payload: dict[str, object] = {
                "status": "cancelled" if superseded else "unavailable",
                "enabled": bool(self._click_through),
                "detail": (
                    "window flag transaction superseded"
                    if superseded
                    else "窗口输入切换超时，已恢复可操作状态"
                ),
            }
            self._log_window_flag_transaction(
                "timeout",
                generation=transition_generation,
                status="unavailable",
                requested_topmost=desired_topmost,
                requested_click_through=enabled,
                requested_lock=self._window_locked,
                verified=False,
                reason_code="dispatch_timeout",
                started_at=transition_started_at,
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Web host async click-through update failed: %s", type(exc).__name__)
            superseded = transition_generation != self._click_through_transition_generation
            payload = {
                "status": "cancelled" if superseded else "unavailable",
                "enabled": bool(self._click_through),
                "detail": (
                    "window flag transaction superseded"
                    if superseded
                    else "点击穿透暂时无法切换，请使用控制台恢复入口"
                ),
            }
        else:
            if transition_generation != self._click_through_transition_generation:
                superseded = True
                payload = {
                    "status": "cancelled",
                    "enabled": bool(self._click_through),
                    "detail": "window flag transaction superseded",
                }
            else:
                state = getattr(getattr(result, "state", None), "value", None)
                detail = getattr(result, "detail", "")
                if isinstance(result, Mapping):
                    value = result.get("status", result.get("state", "unavailable"))
                    state = getattr(value, "value", value)
                    detail = result.get("detail", result.get("reason", ""))
                payload = {
                    "status": str(state or "unavailable"),
                    "enabled": bool(enabled),
                    "detail": str(detail or ""),
                }
                accepted = payload["status"] in {
                    "available",
                    "degraded",
                    "requested",
                    "completed",
                }
                if accepted:
                    actual = self._input_transparency_active()
                    if actual is not None:
                        payload["enabled"] = actual
                        if actual != bool(enabled):
                            payload["status"] = "unavailable"
                            payload["detail"] = (
                                "window system did not retain the requested input transparency flag"
                            )
                            accepted = False
                if accepted:
                    self._click_through = bool(payload["enabled"])
                    if self._click_through:
                        self._clear_page_hover_state(self.view)
                    if not self._surface_mask_enabled and self.view is not None:
                        self._set_x11_input_shapes(
                            self.view,
                            () if self._click_through else None,
                            force=True,
                        )
                    if (
                        not self._geometry_is_current(geometry_generation)
                        or self._drag_transaction_active()
                    ):
                        payload = {
                            "status": "cancelled",
                            "enabled": bool(self._click_through),
                            "detail": "geometry transaction superseded",
                        }
                    else:
                        self._restore_position(
                            self.view,
                            position,
                            geometry_generation=geometry_generation,
                        )
                        if not native_input_path:
                            self._refresh_interaction_targets()
                            self._schedule_interaction_refresh()
        finally:
            self._resume_surface_mask_after_window_rebuild(
                self.view,
                surface_mask_paused,
            )
        if superseded:
            await self._restore_latest_click_through_after_stale_dispatch(
                stale_generation=transition_generation,
            )
        if transition_generation == self._click_through_transition_generation:
            self._click_through_transition_pending = None
        self._log_window_flag_transaction(
            "terminal",
            generation=transition_generation,
            status=str(payload["status"]),
            requested_topmost=desired_topmost,
            requested_click_through=enabled,
            requested_lock=self._window_locked,
            verified=(
                not timed_out
                and payload["status"] in {"available", "degraded", "completed"}
                and bool(payload["enabled"]) == bool(enabled)
            ),
            reason_code=("dispatch_timeout" if timed_out else "dispatch_completed"),
            started_at=transition_started_at,
        )
        if transition_generation == self._click_through_transition_generation:
            self._reconcile_topmost_after_input_transition(
                desired_topmost,
                had_pending_request=topmost_was_pending,
                generation=transition_generation,
            )
        return payload

    async def _restore_latest_click_through_after_stale_dispatch(
        self,
        *,
        stale_generation: int,
    ) -> None:
        """迟到的平台回执若改写原生旗标，重新提交当前输入意图。

        代次检查只能阻止旧协程写回宿主字段，无法撤销平台协程已经发生的
        原生窗口副作用。恢复过程每次提交后重新核对代次；用户在恢复期间
        再次切换时，会继续以最新意图补偿，且整体受同一窗口事务超时约束。
        """

        deadline = monotonic() + _WINDOW_FLAG_DISPATCH_TIMEOUT_SECONDS
        while monotonic() < deadline:
            if self._closing or self._shutdown or self.view is None:
                return
            generation = self._click_through_transition_generation
            pending = self._click_through_transition_pending
            desired = bool(pending) if pending is not None else bool(self._click_through)
            handler = getattr(self.platform, "set_click_through", None) if self.platform else None
            uses_platform_handler = callable(handler)
            if not uses_platform_handler:
                handler = getattr(self.renderer, "set_click_through", None)
            if not callable(handler):
                return
            try:
                recovery = (
                    handler(self.view, desired) if uses_platform_handler else handler(desired)
                )
                if inspect.isawaitable(recovery):
                    remaining = max(0.001, deadline - monotonic())
                    recovery = await asyncio.wait_for(recovery, timeout=remaining)
            except asyncio.CancelledError:
                raise
            except (AttributeError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
                self._log_window_flag_transaction(
                    "recovery",
                    generation=stale_generation,
                    status="unavailable",
                    requested_click_through=desired,
                    verified=False,
                    reason_code="stale_dispatch_recovery_failed",
                )
                return
            if generation != self._click_through_transition_generation:
                continue
            actual = self._input_transparency_active()
            recovery_status = "available"
            recovery_enabled: bool | None = None
            if isinstance(recovery, Mapping):
                recovery_status = str(
                    recovery.get("status", recovery.get("state", "unavailable")) or "unavailable"
                ).lower()
                if "enabled" in recovery:
                    recovery_enabled = bool(recovery.get("enabled"))
            else:
                state = getattr(getattr(recovery, "state", None), "value", None)
                if state is not None:
                    recovery_status = str(state).lower()
            accepted = recovery_status in {
                "available",
                "degraded",
                "requested",
                "completed",
            }
            verified = accepted and (
                bool(actual) == desired
                if actual is not None
                else recovery_enabled is None or recovery_enabled == desired
            )
            if not self._surface_mask_enabled:
                self._set_x11_input_shapes(
                    self.view,
                    () if desired else None,
                    force=True,
                )
            self._log_window_flag_transaction(
                "recovery",
                generation=stale_generation,
                status="available" if verified else "unavailable",
                requested_click_through=desired,
                verified=verified,
                reason_code=(
                    "stale_dispatch_reapplied_latest"
                    if verified
                    else "stale_dispatch_recovery_not_retained"
                ),
            )
            latest_topmost, latest_topmost_pending = self._topmost_intent()
            self._reconcile_topmost_after_input_transition(
                latest_topmost,
                had_pending_request=latest_topmost_pending,
                generation=generation,
            )
            return
        self._log_window_flag_transaction(
            "timeout",
            generation=stale_generation,
            status="unavailable",
            requested_click_through=(
                bool(self._click_through_transition_pending)
                if self._click_through_transition_pending is not None
                else bool(self._click_through)
            ),
            verified=False,
            reason_code="stale_dispatch_recovery_timeout",
        )

    def shutdown(self) -> None:
        self.prepare_shutdown()
        if self._shutdown:
            return
        self._shutdown = True
        self._window_locked = False
        self._click_through_before_lock = False
        self._clear_page_drag()
        timer = self._surface_mask_timer
        self._surface_mask_timer = None
        if timer is not None:
            try:
                timer.stop()
            except (AttributeError, RuntimeError):
                pass
        viewport_timer = self._viewport_sync_timer
        self._viewport_sync_timer = None
        if viewport_timer is not None:
            try:
                viewport_timer.stop()
            except (AttributeError, RuntimeError):
                pass
        self._surface_mask_move_generation += 1
        self._surface_mask_move_active = False
        self._surface_mask_move_timer = None
        drag_timer = self._drag_flush_timer
        self._drag_flush_timer = None
        if drag_timer is not None:
            try:
                drag_timer.stop()
                drag_timer.deleteLater()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._last_viewport_size = None
        self._deferred_page_resize = None
        self._last_cursor_target = None
        self._last_window_position = None
        self._surface_mask_enabled = False
        self._surface_mask_rebuild_pending = False
        self._surface_mask_rebuild_generation += 1
        self._surface_mask_capture_generation += 1
        self._surface_mask_capture_pending = False
        self._surface_mask_capture_failures = 0
        # 使尚未投递的窗口旗标回调失效；否则退出后 Qt 仍可能执行旧
        # QTimer，重新触碰已经销毁的 QWebEngine 原生子表面。
        self._cancel_pending_topmost_transition(
            reason_code="host_shutdown",
            detail="窗口宿主关闭，置顶事务已取消",
        )
        self._window_flag_transition_generation += 1
        self._window_flag_transition_pending = None
        self._window_flag_transition_apply = None
        self._window_flag_transition_cleanup = None
        self._window_flag_transition_result = None
        self._topmost_confirmed = None
        self._topmost_capability_cache = None
        self._click_through_transition_generation += 1
        self._click_through_transition_pending = None
        self._window_lock_transition_generation += 1
        self._window_lock_transition_pending = None
        self._deferred_page_events.clear()
        self._interaction_rebuild_until = 0.0
        self._autonomous_geometry_until = 0.0
        self._autonomous_geometry_pending = 0
        deferred_timer = self._deferred_page_event_timer
        self._deferred_page_event_timer = None
        if deferred_timer is not None:
            try:
                deferred_timer.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._surface_mask_region = None
        self._surface_mask_input_region = None
        self._surface_mask_view_visible = None
        view = self.view
        if view is not None:
            self._set_x11_input_shapes(view, None, force=True)
        ready_setter = self._page_bridge_ready_setter
        self._page_bridge_ready_setter = None
        if ready_setter is not None:
            try:
                ready_setter(None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        callback_setter = getattr(self.renderer, "set_interaction_callback", None)
        if callable(callback_setter):
            try:
                callback_setter(None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._context_menu_callback = None
        self._double_click_callback = None
        self._click_callback = None
        self._renderer_click_callback_active = False
        # 保留模型回执回调到 renderer.shutdown() 完成。若页面在关闭阶段
        # 仍有一笔候选加载，renderer 会先发出 failed 终态，调用方不能永久
        # 卡在“切换中”。
        self._page_pointer_over = False
        self._page_bridge_enabled = False
        click_setter = getattr(self.renderer, "set_click_callback", None)
        if callable(click_setter):
            try:
                click_setter(None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._remove_native_interaction_filter()
        handler = getattr(self.renderer, "shutdown", None)
        if callable(handler):
            try:
                handler()
            except Exception as exc:
                logger.debug("Web Live2D renderer cleanup failed: %s", type(exc).__name__)
        self._model_reload_callback = None
        self._renderer_model_reload_callback_active = False
        reload_setter = getattr(self.renderer, "set_model_reload_callback", None)
        if callable(reload_setter):
            try:
                reload_setter(None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    def prepare_shutdown(self) -> None:
        """在 Qt ``aboutToQuit`` 阶段停止异步 Shape 回调。

        该方法只封锁新的页面读回和定时器，不释放 renderer/view；真正的
        生命周期清理仍由 :meth:`shutdown` 完成，因而可安全重复调用。
        """

        if self._shutdown or self._closing:
            return
        self._closing = True
        self._surface_mask_enabled = False
        self._surface_mask_view_visible = None
        self._surface_mask_rebuild_pending = False
        self._surface_mask_capture_pending = False
        self._surface_mask_capture_generation += 1
        self._cancel_pending_topmost_transition(
            reason_code="host_closing",
            detail="窗口宿主正在关闭，置顶事务已取消",
        )
        self._window_flag_transition_generation += 1
        self._window_flag_transition_pending = None
        self._window_flag_transition_apply = None
        self._window_flag_transition_cleanup = None
        self._window_flag_transition_result = None
        self._topmost_confirmed = None
        self._topmost_capability_cache = None
        self._click_through_transition_generation += 1
        self._click_through_transition_pending = None
        self._window_lock_transition_generation += 1
        self._window_lock_transition_pending = None
        self._deferred_page_events.clear()
        self._deferred_page_resize = None
        self._interaction_rebuild_until = 0.0
        self._autonomous_geometry_until = 0.0
        self._autonomous_geometry_pending = 0
        deferred_timer = self._deferred_page_event_timer
        self._deferred_page_event_timer = None
        if deferred_timer is not None:
            try:
                deferred_timer.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        self._surface_mask_rebuild_generation += 1
        timer = self._surface_mask_timer
        self._surface_mask_timer = None
        if timer is not None:
            try:
                timer.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        drag_timer = self._drag_flush_timer
        self._drag_flush_timer = None
        if drag_timer is not None:
            try:
                drag_timer.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass


__all__ = ["WebPetHost"]
