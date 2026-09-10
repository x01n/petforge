from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path

from core.rendering.actions import ExpressionRequest, MotionRequest
from gui.qt6.display_size import display_size_preset
from gui.qt6.pet_interaction import (
    PET_CLICK_DEBOUNCE_MS,
    classify_pet_part,
    make_pet_click_payload,
)
from gui.renderers.live2d import Live2DRenderer
from gui.renderers.protocol import Renderer
from gui.renderers.sprite import SpriteRenderer

logger = logging.getLogger(__name__)

# 高刷新率输入设备可能在一帧内产生多个 MouseMove；限制几何提交频率，
# 释放时再补齐最后位置，避免窗口管理器与 OpenGL 重绘队列互相追赶。
_DRAG_MOVE_INTERVAL_SECONDS = 1.0 / 120.0
# 释放拖动后给窗口管理器和渲染线程一个稳定窗口；期间丢弃已经排队的
# 自主步进，避免旧目标在用户松手后把窗口拉回去。
_AUTONOMOUS_RESUME_GRACE_SECONDS = 0.35
_AUTONOMOUS_GEOMETRY_SETTLE_SECONDS = 0.08
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


try:  # GUI 依赖保持可选，核心合约测试不需要 Qt。
    from PySide6.QtCore import QPoint, QRectF, QSize, Qt, QTimer, Signal
    from PySide6.QtGui import (
        QBitmap,
        QColor,
        QCursor,
        QFont,
        QFontMetrics,
        QImage,
        QLinearGradient,
        QOpenGLContext,
        QPainter,
        QPainterPath,
        QRegion,
        QSurfaceFormat,
    )
    from PySide6.QtOpenGL import QOpenGLPaintDevice, QOpenGLWindow

    pyside6_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_available = False


if pyside6_available:

    def _fit_bubble_text(text: str, font: QFont, text_rect: QRectF) -> str:
        """在固定气泡区域内保留回答首尾，避免长文本被静默裁掉。"""

        metrics = QFontMetrics(font)
        display_text = str(text or "")
        while (
            len(display_text) > 16
            and metrics.boundingRect(
                text_rect.toRect(),
                Qt.TextWordWrap,
                display_text,
            ).height()
            > text_rect.height()
        ):
            keep = max(12, int(len(display_text) * 0.72))
            head = max(4, keep // 2)
            tail = max(4, keep - head)
            display_text = f"{str(text or '')[:head]}…{str(text or '')[-tail:]}"
        return display_text

    class PetOpenGLWindow(QOpenGLWindow):
        expressionChanged = Signal(str)
        motionChanged = Signal(str)
        textSubmitted = Signal(str)
        consoleRequested = Signal()
        # 右键菜单由上层宿主创建；窗口只负责提供可靠的全局坐标。
        contextMenuRequested = Signal(QPoint)
        closeRequested = Signal()

        def __init__(
            self,
            renderer: Renderer,
            *,
            platform: object | None = None,
            size: tuple[int, int] | None = None,
            fallback_renderer: Renderer | None = None,
            sprite_scale: float = 0.6,
            always_on_top: bool = True,
        ) -> None:
            fmt = QSurfaceFormat()
            fmt.setAlphaBufferSize(8)
            # 使用 Qt 的 alpha 混合更新路径，避免部分 X11/Wayland 合成器将
            # 未覆盖的 framebuffer 区域当作不透明黑色提交。每帧仍会先清空
            # 透明颜色，因此不会保留上一帧的残影。
            super().__init__(QOpenGLWindow.UpdateBehavior.PartialUpdateBlend)
            self.setFormat(fmt)
            self._always_on_top = bool(always_on_top)
            self._topmost_capability_cache: tuple[str, str, str, float] | None = None
            # 展示层不应因透明区域点击而抢走当前应用焦点；控制台和模型
            # 向导使用独立 QWidget，需要输入时仍可正常激活。
            self.setFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
            self.setFlag(Qt.WindowStaysOnTopHint, self._always_on_top)
            self.renderer = renderer
            self._fallback_renderer = fallback_renderer
            self._sprite_scale = max(0.1, min(2.0, float(sprite_scale)))
            self.platform = platform
            self._click_through = False
            self._click_through_override_active = False
            self._click_through_before_override: bool | None = None
            self._window_locked = False
            self._click_through_before_lock = False
            self._last_cursor_target: tuple[int, int, int, int] | None = None
            self._last_window_position: tuple[int, int] | None = None
            self._legacy_shortcuts_enabled = True
            self._speech_text = ""
            self._speech_mood = "neutral"
            self._speech_visible = True
            self._interaction_feedback_text = ""
            self._interaction_feedback_visible = False
            self._interaction_feedback_until = 0.0
            self._last_frame_at = time.monotonic()
            self._drag_origin: QPoint | None = None
            self._window_origin: QPoint | None = None
            self._dragging = False
            self._system_move_active = False
            self._pending_drag_target: tuple[int, int] | None = None
            self._last_drag_target: tuple[int, int] | None = None
            self._last_drag_submit_at = 0.0
            self._drag_flush_timer: QTimer | None = None
            self._drag_watchdog_timer: QTimer | None = None
            self._renderer_dragging = False
            self._autonomous_block_until = 0.0
            # 顶层窗口位置、输入 Shape 和异步恢复回调共享单调几何代次。
            # 新的用户拖动一旦开始，旧的窗口标志/行为回调不得再写回坐标。
            self._geometry_generation = 0
            self._autonomous_geometry_until = 0.0
            self._autonomous_geometry_pending = 0
            self._pointer_over = False
            self._click_callback: Callable[[Mapping[str, object]], None] | None = None
            self._interaction_interrupt: Callable[[], object] | None = None
            self._activity_notifier: Callable[[], object] | None = None
            self._interaction_interrupt_notified = False
            self._click_token = 0
            self._sprite_image_path = None
            self._sprite_image = None
            self._sprite_scaled_image = None
            self._sprite_scaled_path = None
            self._sprite_scaled_area = None
            self._sprite_previous_image_path = None
            self._sprite_previous_image = None
            self._sprite_previous_scaled_image = None
            self._sprite_previous_scaled_path = None
            self._sprite_previous_scaled_area = None
            # 精灵帧的透明画布是动作/表情的共同坐标系。以首帧尺寸作为
            # 稳定参考，不让某一张差分图片的边界改变窗口比例或脚底锚点。
            self._sprite_reference_size: tuple[int, int] | None = None
            self._surface_mask_enabled = False
            self._surface_mask_last_refresh = 0.0
            # Qt ``setMask`` 同时影响视觉和默认输入区域；X11 上由平台
            # 适配器额外写入仅覆盖模型像素的 Input Shape。该状态只用于
            # 诊断和控制台反馈，不能把 Wayland 的降级结果伪装成成功。
            self._surface_mask_input_shape_status: dict[str, object] = {
                "status": "unavailable",
                "detail": "平台未提供原生 Input Shape",
            }
            self._shutdown = False
            # QWindow 没有 QWidget 的 setMouseTracking；在按下时抓取鼠标，
            # 确保拖动不会因为帧窗口或指针离开边界而中断。
            window_size = size if size is not None else self._preferred_size()
            self.resize(*window_size)
            self._timer = QTimer(self)
            # 原生 Live2D 需要约 60 FPS 才能保持参数插值；静态 WebP 精灵
            # 只有显式动作和气泡变化，30 FPS 已足够且可把空闲绘制次数减半。
            self._render_interval_ms = 16 if isinstance(renderer, Live2DRenderer) else 33
            self._timer.setInterval(self._render_interval_ms)
            self._timer.timeout.connect(self._tick)
            self._timer.start()

        def set_interaction_interrupt(self, callback: Callable[[], object] | None) -> None:
            """设置用户按下时立即使后台行为代际失效的同步回调。"""

            self._interaction_interrupt = callback

        def set_activity_notifier(self, callback: Callable[[], object] | None) -> None:
            """设置实际用户手势的活跃状态通知器。"""

            self._activity_notifier = callback

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
                    logger.debug("failed to interrupt behavior on OpenGL press", exc_info=True)
            notifier = self._activity_notifier
            if callable(notifier):
                try:
                    result = notifier()
                    if inspect.isawaitable(result):
                        _safe_close_awaitable(result)
                except Exception:
                    logger.debug("failed to record OpenGL user activity", exc_info=True)

        def _invalidate_geometry(self) -> int:
            """开启新的几何事务，使延迟回调失效。"""

            self._geometry_generation += 1
            return self._geometry_generation

        def _geometry_is_current(self, generation: int | None) -> bool:
            """检查延迟几何回调是否仍属于当前窗口。"""

            return (
                generation is None or generation == self._geometry_generation
            ) and not self._shutdown

        def _drag_transaction_active(self) -> bool:
            """返回用户拖动或 Wayland 系统移动是否仍在进行。"""

            return bool(self._drag_origin is not None or self._dragging or self._system_move_active)

        def reload_resources(
            self,
            resource_root: object,
            *,
            model_path: object | None = None,
            sprite_scale: float | None = None,
        ) -> Mapping[str, object]:
            """在 OpenGL 宿主边界请求原子资源重载并清除精灵图像缓存。"""

            if self._shutdown:
                return {"status": "unavailable", "reason": "renderer host is closed"}
            if self._drag_transaction_active() or self._renderer_dragging:
                return {"status": "busy", "reason": "resource reload is blocked while dragging"}
            handler = getattr(self.renderer, "reload_resources", None)
            if not callable(handler):
                return {"status": "unavailable", "reason": "resource reload is unavailable"}
            try:
                result = handler(
                    resource_root,
                    model_path=model_path,
                    sprite_scale=sprite_scale,
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("OpenGL resource reload failed: %s", type(exc).__name__)
                return {"status": "failed", "reason": "resource reload failed"}
            if not isinstance(result, Mapping):
                return {"status": "failed", "reason": "resource reload returned an invalid result"}
            response = dict(result)
            status = str(response.get("status", "") or "").strip().lower()
            if status not in {"available", "reloaded", "unchanged"}:
                return response
            if isinstance(self.renderer, SpriteRenderer):
                applied_scale = response.get("sprite_scale")
                if applied_scale is not None:
                    self._sprite_scale = float(applied_scale)
                self._sprite_image_path = None
                self._sprite_image = None
                self._sprite_scaled_image = None
                self._sprite_scaled_path = None
                self._sprite_scaled_area = None
                self._sprite_previous_image_path = None
                self._sprite_previous_image = None
                self._sprite_previous_scaled_image = None
                self._sprite_previous_scaled_path = None
                self._sprite_previous_scaled_area = None
                self._sprite_reference_size = None
                self.resize(*self._preferred_size())
            self._invalidate_geometry()
            self._last_cursor_target = None
            self._last_window_position = None
            if self._surface_mask_enabled:
                self._refresh_surface_mask(force=True)
            self.update()
            return response

        def _autonomous_geometry_active(self) -> bool:
            """返回自主步进后的 Shape 稳定窗口是否仍未结束。"""

            return (
                bool(self._autonomous_geometry_pending)
                or time.monotonic() < self._autonomous_geometry_until
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
                time.monotonic() + _AUTONOMOUS_GEOMETRY_SETTLE_SECONDS,
            )

        def _end_autonomous_geometry(self) -> None:
            """结束一个自主位置请求并保留短暂的 Shape 稳定窗口。"""

            self._autonomous_geometry_pending = max(0, self._autonomous_geometry_pending - 1)
            if self._autonomous_geometry_pending == 0:
                self._autonomous_geometry_until = max(
                    self._autonomous_geometry_until,
                    time.monotonic() + _AUTONOMOUS_GEOMETRY_SETTLE_SECONDS,
                )

        def initializeGL(self) -> None:  # noqa: N802
            if isinstance(self.renderer, Live2DRenderer):
                if not self.renderer.initialize() and self._fallback_renderer is not None:
                    self._activate_fallback()
                else:
                    resizer = getattr(self.renderer, "resize", None)
                    if callable(resizer):
                        try:
                            resizer(self.width(), self.height(), self.devicePixelRatio())
                        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                            logger.debug("renderer initial viewport resize failed", exc_info=True)

        def resizeGL(self, width: int, height: int) -> None:  # noqa: N802
            resizer = getattr(self.renderer, "resize", None)
            if callable(resizer):
                try:
                    resizer(width, height, self.devicePixelRatio())
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("renderer viewport resize failed", exc_info=True)

        def paintGL(self) -> None:  # noqa: N802
            try:
                from OpenGL.GL import (
                    GL_COLOR_BUFFER_BIT,
                    GL_DEPTH_BUFFER_BIT,
                    glClear,
                    glClearColor,
                )

                glClearColor(0.0, 0.0, 0.0, 0.0)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
                pass
            if isinstance(self.renderer, Live2DRenderer) and self.renderer.model is not None:
                try:
                    self.renderer.draw()
                except Exception as exc:
                    logger.warning(
                        "Live2D draw failed; using sprite fallback: %s", type(exc).__name__
                    )
                    self._activate_fallback()
            if isinstance(self.renderer, SpriteRenderer):
                self._paint_sprite()
            self._paint_bubble()
            self._paint_interaction_feedback()

        def _paint_device(self) -> tuple[QPainter, QOpenGLPaintDevice] | None:
            dpr = max(1.0, float(self.devicePixelRatio()))
            device = QOpenGLPaintDevice(
                QSize(round(self.width() * dpr), round(self.height() * dpr))
            )
            device.setDevicePixelRatio(dpr)
            painter = QPainter(device)
            if not painter.isActive():
                return None
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
            return painter, device

        def _clear_sprite_previous_cache(self) -> None:
            """释放表情过渡完成后不再使用的前一帧解码和缩放缓存。"""

            self._sprite_previous_image_path = None
            self._sprite_previous_image = None
            self._sprite_previous_scaled_image = None
            self._sprite_previous_scaled_path = None
            self._sprite_previous_scaled_area = None

        def _sprite_frame_image(self, path: Path, *, previous: bool = False) -> QImage | None:
            """读取当前或前一精灵帧，并按路径复用已解码的 QImage。"""

            prefix = "_sprite_previous" if previous else "_sprite"
            path_attribute = f"{prefix}_image_path"
            image_attribute = f"{prefix}_image"
            if getattr(self, path_attribute) == path:
                cached = getattr(self, image_attribute)
                return cached if cached is not None and not cached.isNull() else None
            image = QImage(str(path))
            setattr(self, path_attribute, path)
            if not image.isNull():
                try:
                    image.setDevicePixelRatio(1.0)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                if self._sprite_reference_size is None:
                    self._sprite_reference_size = (image.width(), image.height())
            cached = image if not image.isNull() else None
            setattr(self, image_attribute, cached)
            setattr(self, f"{prefix}_scaled_image", None)
            setattr(self, f"{prefix}_scaled_path", None)
            setattr(self, f"{prefix}_scaled_area", None)
            return cached

        def _scaled_sprite_frame(
            self,
            path: Path,
            area_width: int,
            area_height: int,
            *,
            previous: bool = False,
        ) -> tuple[QImage, QImage] | None:
            """返回按稳定透明画布等比缩放后的帧，并缓存缩放结果。"""

            image = self._sprite_frame_image(path, previous=previous)
            if image is None:
                return None
            prefix = "_sprite_previous" if previous else "_sprite"
            area_key = (area_width, area_height, self._sprite_reference_size)
            scaled = getattr(self, f"{prefix}_scaled_image")
            if (
                scaled is None
                or getattr(self, f"{prefix}_scaled_path") != path
                or getattr(self, f"{prefix}_scaled_area") != area_key
            ):
                reference_width, reference_height = self._sprite_reference_size or (
                    image.width(),
                    image.height(),
                )
                uniform_scale = min(
                    area_width / max(1, int(reference_width)),
                    area_height / max(1, int(reference_height)),
                )
                scaled = image.scaled(
                    max(1, round(image.width() * uniform_scale)),
                    max(1, round(image.height() * uniform_scale)),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                setattr(self, f"{prefix}_scaled_image", scaled)
                setattr(self, f"{prefix}_scaled_path", path)
                setattr(self, f"{prefix}_scaled_area", area_key)
            if scaled is None or scaled.isNull():
                return None
            return image, scaled

        def _paint_sprite(self) -> None:
            if not isinstance(self.renderer, SpriteRenderer):
                return
            path = self.renderer.current_frame
            if path is None:
                return
            area_top = 50
            area_width = max(1, self.width())
            area_height = max(1, self.height() - area_top)
            previous_path = self.renderer.previous_frame
            progress = self.renderer.expression_transition_progress
            if (
                previous_path is not None
                and previous_path == self._sprite_image_path
                and path != self._sprite_image_path
            ):
                # 当前帧恰好是时间线要求的前一帧时直接转交缓存，避免在表情
                # 切换边沿重新解码同一张 WebP。
                self._sprite_previous_image_path = self._sprite_image_path
                self._sprite_previous_image = self._sprite_image
                self._sprite_previous_scaled_image = self._sprite_scaled_image
                self._sprite_previous_scaled_path = self._sprite_scaled_path
                self._sprite_previous_scaled_area = self._sprite_scaled_area
            elif previous_path is None or progress >= 1.0:
                self._clear_sprite_previous_cache()
            current_frame = self._scaled_sprite_frame(path, area_width, area_height)
            if current_frame is None:
                return
            image, scaled = current_frame
            previous_frame = (
                self._scaled_sprite_frame(
                    previous_path,
                    area_width,
                    area_height,
                    previous=True,
                )
                if previous_path is not None and previous_path != path and progress < 1.0
                else None
            )
            pair = self._paint_device()
            if pair is None:
                return
            painter, _device = pair
            reference_width, reference_height = self._sprite_reference_size or (
                image.width(),
                image.height(),
            )
            uniform_scale = min(
                area_width / max(1, reference_width),
                area_height / max(1, reference_height),
            )
            canvas_width = max(1, round(reference_width * uniform_scale))
            canvas_height = max(1, round(reference_height * uniform_scale))
            # 透明画布以底边为锚点，差分帧即使内容包围盒不同也不会上下跳动。
            target_x = (area_width - canvas_width) / 2.0 + (canvas_width - scaled.width()) / 2.0
            target_y = area_top + area_height - canvas_height + (canvas_height - scaled.height())
            transform = getattr(self.renderer, "visual_transform", (0.0, 0.0))
            try:
                offset_x = float(transform[0])
                offset_y = float(transform[1])
            except (IndexError, TypeError, ValueError):
                offset_x = offset_y = 0.0
            target_x += offset_x
            target_y += offset_y
            target = QRectF(target_x, target_y, scaled.width(), scaled.height())
            if previous_frame is not None:
                _previous_image, previous_scaled = previous_frame
                previous_x = (
                    (area_width - canvas_width) / 2.0
                    + (canvas_width - previous_scaled.width()) / 2.0
                    + offset_x
                )
                previous_y = (
                    area_top
                    + area_height
                    - canvas_height
                    + (canvas_height - previous_scaled.height())
                    + offset_y
                )
                previous_target = QRectF(
                    previous_x,
                    previous_y,
                    previous_scaled.width(),
                    previous_scaled.height(),
                )
                painter.setOpacity(1.0 - progress)
                painter.drawImage(previous_target, previous_scaled)
                painter.setOpacity(progress)
                painter.drawImage(target, scaled)
                painter.setOpacity(1.0)
            else:
                painter.drawImage(target, scaled)
            painter.end()

        def _preferred_size(self) -> tuple[int, int]:
            """根据首帧尺寸计算自然窗口大小，避免精灵被固定矩形压扁。"""

            if isinstance(self.renderer, SpriteRenderer):
                path = self.renderer.current_frame
                if path is not None:
                    image = QImage(str(path))
                    if not image.isNull():
                        source_width = max(1, image.width())
                        source_height = max(1, image.height())
                        # 最小可见尺寸也通过同一比例推导，避免分别 clamp
                        # 宽高导致窗口和精灵的参考比例失真。
                        effective_scale = max(
                            self._sprite_scale,
                            160.0 / source_width,
                            240.0 / source_height,
                        )
                        return (
                            max(1, round(source_width * effective_scale)),
                            max(1, round(source_height * effective_scale)) + 50,
                        )
            return 420, 520

        def resizeEvent(self, event) -> None:  # noqa: N802
            self._sprite_scaled_image = None
            self._sprite_scaled_area = None
            self._sprite_previous_scaled_image = None
            self._sprite_previous_scaled_area = None
            super().resizeEvent(event)

        def _paint_bubble(self) -> None:
            if not self._speech_visible or not self._speech_text:
                return
            pair = self._paint_device()
            if pair is None:
                return
            painter, _device = pair
            rect = QRectF(8, 8, self.width() - 16, min(48, max(32, self.height() - 16)))
            path = QPainterPath()
            path.addRoundedRect(rect, 20, 20)
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0.0, QColor("#2E2440"))
            gradient.setColorAt(0.45, QColor("#221A2E"))
            gradient.setColorAt(1.0, QColor("#1A1426"))
            painter.setPen(QColor("#0B0713"))
            painter.setBrush(gradient)
            painter.drawPath(path)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QColor("#FF9DBE"))
            painter.drawPath(path)
            painter.setPen(QColor("#FAF6FB"))
            font = QFont("LXGW WenKai", 15)
            painter.setFont(font)
            text_rect = rect.adjusted(14, 10, -14, -10)
            # 固定气泡区域不能把长流式回答直接裁成不可读的前缀。
            display_text = _fit_bubble_text(self._speech_text, font, text_rect)
            painter.drawText(text_rect, Qt.TextWordWrap, display_text)
            painter.end()

        def _paint_interaction_feedback(self) -> None:
            """在 QOpenGL/精灵路径绘制不遮挡脚部的侧边反馈牌。"""

            if (
                not self._interaction_feedback_visible
                or not self._interaction_feedback_text
                or time.monotonic() >= self._interaction_feedback_until
            ):
                return
            pair = self._paint_device()
            if pair is None:
                return
            painter, _device = pair
            width = min(96.0, max(72.0, float(self.width() - 10)))
            height = min(112.0, max(54.0, float(self.height() - 20)))
            rect = QRectF(
                float(self.width()) - width - 5.0, (self.height() - height) / 2.0, width, height
            )
            path = QPainterPath()
            path.addRoundedRect(rect, 12, 12)
            painter.setPen(QColor("#FFB6CE"))
            painter.setBrush(QColor(34, 26, 46, 240))
            painter.drawPath(path)
            painter.setPen(QColor("#FFF5FA"))
            painter.setFont(QFont("LXGW WenKai", 11))
            painter.drawText(
                rect.adjusted(8, 7, -8, -7), Qt.TextWordWrap, self._interaction_feedback_text
            )
            painter.end()

        def set_surface_mask_enabled(self, enabled: bool) -> dict[str, object]:
            """按实际 alpha 设置精确可见/输入 Shape 掩码。"""

            self._invalidate_geometry()
            self._surface_mask_enabled = bool(enabled)
            # 不清零运行中移动的 pending；每个请求由自身 finally 归还，
            # 避免遮罩切换后旧回执误扣新的自主步骤。
            self._autonomous_geometry_until = 0.0
            if not self._surface_mask_enabled:
                # 点击穿透期间保持真正的空 Input Shape；否则清除视觉
                # 遮罩时写入整窗区域会短暂恢复对底层应用的拦截。
                input_shape = self._set_platform_input_shape(() if self._click_through else None)
                try:
                    self.clearMask()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    return {
                        "status": "unavailable",
                        "enabled": False,
                        "ready": False,
                        "input_ready": False,
                        "capture_pending": False,
                        "input_shape": input_shape,
                    }
                state = self.surface_mask_status()
                state["input_shape"] = input_shape
                return state

            applied = self._refresh_surface_mask(force=True)
            state = self.surface_mask_status()
            # 保留调用时刚刚得到的原生结果，即使平台异步重建窗口后状态
            # 尚未再次回写，也不能把上一次的状态带回控制台。
            state["enabled"] = bool(applied and state.get("enabled", False))
            state["input_shape"] = dict(self._surface_mask_input_shape_status)
            return state

        def surface_mask_status(self) -> dict[str, object]:
            """返回 OpenGL 视觉遮罩与原生输入区域的当前状态。

            ``QWindow.setMask`` 负责视觉裁剪，而 X11 Input Shape 由平台适配器
            单独维护；控制台需要同时知道两者，不能只依据 ``setMask`` 是否成功
            就把透明区域报告为可点击。Wayland/无平台适配器时保留明确降级状态。
            """

            enabled = bool(self._surface_mask_enabled)
            visual_ready = False
            if enabled:
                try:
                    mask = self.mask()
                    empty_checker = getattr(mask, "isEmpty", None)
                    visual_ready = not bool(empty_checker()) if callable(empty_checker) else True
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    visual_ready = False
            input_shape = dict(self._surface_mask_input_shape_status)
            input_state = str(input_shape.get("status", "unavailable") or "").lower()
            input_ready = bool(
                enabled
                and visual_ready
                and not self._click_through
                and not self._window_locked
                and input_state in {"available", "degraded"}
            )
            if not enabled:
                status = "available"
            elif visual_ready and input_ready:
                status = "available"
            elif visual_ready and input_state == "degraded":
                status = "degraded"
            elif visual_ready and input_state == "unavailable":
                status = "degraded"
            else:
                status = "pending"
            return {
                "status": status,
                "enabled": bool(enabled and visual_ready),
                "ready": visual_ready,
                "input_ready": input_ready,
                "capture_pending": False,
                "input_shape": input_shape,
            }

        @staticmethod
        def _input_shape_status(result: object) -> dict[str, object]:
            """将平台能力结果收敛为无对象引用的 UI 状态。"""

            if isinstance(result, Mapping):
                raw_status = result.get("status", result.get("state", "unavailable"))
                detail = result.get("detail", result.get("reason", ""))
            else:
                raw_status = getattr(getattr(result, "state", None), "value", "unavailable")
                detail = getattr(result, "detail", "")
            status = str(getattr(raw_status, "value", raw_status) or "unavailable").strip().lower()
            if status not in {"available", "degraded", "unavailable"}:
                status = "unavailable"
            return {"status": status, "detail": str(detail or "")}

        def _set_platform_input_shape(self, region: object | None) -> dict[str, object]:
            """通过可选平台适配器设置原生 Input Shape。"""

            setter = getattr(self.platform, "set_input_shape", None)
            if not callable(setter):
                result = {
                    "status": "unavailable",
                    "detail": "当前平台未提供原生 Input Shape",
                }
                self._surface_mask_input_shape_status = result
                return dict(result)
            try:
                result = setter(self, region)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("OpenGL native input Shape update failed: %s", type(exc).__name__)
                result = {
                    "status": "unavailable",
                    "detail": "原生 Input Shape 暂不可用",
                }
            status = self._input_shape_status(result)
            self._surface_mask_input_shape_status = status
            return dict(status)

        def _schedule_surface_input_shape_restore(self) -> None:
            """顶层标志重建后重新写入模型 Input Shape。"""

            if self._shutdown:
                return
            geometry_generation = self._geometry_generation

            def apply() -> None:
                if (
                    not self._geometry_is_current(geometry_generation)
                    or self._drag_transaction_active()
                    or self._autonomous_geometry_active()
                ):
                    return
                if self._surface_mask_enabled:
                    self._refresh_surface_mask(force=True)
                else:
                    # 遮罩被关闭后没有模型区域可提交；根据当前穿透状态
                    # 恢复整窗或保持窗口外哨兵，避免先开启穿透再关闭遮罩
                    # 的组合把窗口永久留在空输入区。
                    self._set_platform_input_shape(() if self._click_through else None)

            apply()
            try:
                # setFlag 可能异步替换原生窗口；下一轮事件循环拿到新 winId
                # 后再写一次，避免气泡区域在切换置顶/穿透后重新拦截桌面。
                QTimer.singleShot(0, apply)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        def _refresh_surface_mask(self, *, force: bool = False) -> bool:
            """按当前精灵 alpha 更新窗口 Shape，避免透明区域显示黑面。"""

            if not self._surface_mask_enabled:
                return False
            if self._drag_transaction_active() or self._autonomous_geometry_active():
                # 拖动期间保留上一份视觉/输入区域；重建 Shape 会和窗口
                # 几何提交交错，导致 OpenGL 合成器出现黑帧或跳帧。
                try:
                    mask = self.mask()
                    checker = getattr(mask, "isEmpty", None)
                    return not bool(checker()) if callable(checker) else True
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    return False
            if not isinstance(self.renderer, SpriteRenderer):
                # 渲染器切换后，不能让上一个精灵留下的原生 Input Shape
                # 继续拦截桌面输入；视觉遮罩也一并恢复为整窗默认值。
                self._set_platform_input_shape(() if self._click_through else None)
                try:
                    self.clearMask()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                return False
            now = time.monotonic()
            motion = str(getattr(getattr(self.renderer, "state", None), "motion", "") or "")
            refresh_interval = (
                0.024 if motion.strip().lower() in {"walk", "wave", "blink"} else 0.05
            )
            if not force and now - self._surface_mask_last_refresh < refresh_interval:
                return True
            path = self.renderer.current_frame
            if path is None:
                return False
            image = QImage(str(path))
            if image.isNull():
                return False
            try:
                image.setDevicePixelRatio(1.0)
                reference_width, reference_height = self._sprite_reference_size or (
                    image.width(),
                    image.height(),
                )
                area_top = 50
                area_width = max(1, self.width())
                area_height = max(1, self.height() - area_top)
                uniform_scale = min(
                    area_width / max(1, int(reference_width)),
                    area_height / max(1, int(reference_height)),
                )
                scaled = image.scaled(
                    max(1, round(image.width() * uniform_scale)),
                    max(1, round(image.height() * uniform_scale)),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                mask = QBitmap.fromImage(scaled.createAlphaMask())
                region = QRegion(mask)
                # 资源/解码器可能短暂返回全透明帧；不要把空区域提交为
                # 顶层 Shape，否则窗口会变成 0x0 且后续鼠标无法恢复。
                if region.isEmpty():
                    return False
                canvas_width = max(1, round(int(reference_width) * uniform_scale))
                canvas_height = max(1, round(int(reference_height) * uniform_scale))
                target_x = (area_width - canvas_width) / 2.0 + (canvas_width - scaled.width()) / 2.0
                target_y = (
                    area_top + area_height - canvas_height + (canvas_height - scaled.height())
                )
                transform = getattr(self.renderer, "visual_transform", (0.0, 0.0))
                target_x += float(transform[0])
                target_y += float(transform[1])
                region.translate(round(target_x), round(target_y))
                # 保留一份仅模型像素的区域给 X11 Input Shape；气泡/反馈牌
                # 只并入视觉 Shape，避免展示层覆盖底层应用的点击。Qt 的
                # setMask 仍负责无合成器时的可见裁剪，原生 Input Shape
                # 由平台适配器在后面独立写入。
                input_region = QRegion(region)
                visual_region = QRegion(region)
                if self._speech_visible and self._speech_text:
                    bubble_height = min(48, max(32, self.height() - 16))
                    visual_region = visual_region.united(
                        QRegion(QRectF(8, 8, self.width() - 16, bubble_height).toRect())
                    )
                if (
                    self._interaction_feedback_visible
                    and self._interaction_feedback_text
                    and time.monotonic() < self._interaction_feedback_until
                ):
                    toast_width = min(96, max(72, self.width() - 10))
                    toast_height = min(112, max(54, self.height() - 20))
                    visual_region = visual_region.united(
                        QRegion(
                            QRectF(
                                self.width() - toast_width - 5,
                                (self.height() - toast_height) / 2,
                                toast_width,
                                toast_height,
                            ).toRect()
                        )
                    )
                self.setMask(visual_region)
                # Qt 的 WindowTransparentForInput 在不同窗口管理器上可能
                # 仅是提示；点击穿透状态下强制提交空原生区域，解除后再
                # 恢复模型区域，避免气泡或整窗区域抢走底层点击。
                self._set_platform_input_shape(() if self._click_through else input_region)
                self._surface_mask_last_refresh = now
                return True
            except (
                AttributeError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
                ZeroDivisionError,
            ):
                return False

        def _tick(self) -> None:
            now = time.monotonic()
            if not self._renderer_dragging:
                try:
                    self.renderer.advance(now - self._last_frame_at)
                except Exception as exc:
                    logger.warning(
                        "Live2D update failed; using sprite fallback: %s", type(exc).__name__
                    )
                    self._activate_fallback()
            self._last_frame_at = now
            if self._interaction_feedback_visible and now >= self._interaction_feedback_until:
                self._interaction_feedback_visible = False
                self._interaction_feedback_text = ""
            # 拖动时窗口几何由鼠标事件连续提交；此时反复重建 alpha
            # Shape 会和 OpenGL/窗口管理器重绘交错，造成模型短暂跳帧。
            # 保留上一份区域，释放后再强制采集一次当前帧。
            if self._drag_origin is None and not self._dragging and not self._system_move_active:
                self._refresh_surface_mask()
            # 拖动期间保留最后一帧，不主动排入新的 OpenGL 绘制请求；窗口
            # 几何本身仍由鼠标事件提交。这样可以避免窗口管理器移动、部分
            # 更新合成和渲染线程同时争抢 framebuffer，释放时再补一帧。
            if not self._renderer_dragging:
                self.update()

        def _activate_fallback(self) -> None:
            fallback = self._fallback_renderer
            if fallback is None or self.renderer is fallback:
                return
            previous = self.renderer
            self.renderer = fallback
            try:
                self._release_renderer(previous)
            except Exception as exc:
                logger.warning("failed to release Live2D renderer: %s", type(exc).__name__)

        def _release_renderer(self, renderer: Renderer) -> None:
            """确保渲染器销毁发生在该窗口的 OpenGL context 中。"""

            acquired = False
            try:
                current_context = QOpenGLContext.currentContext()
                window_context = self.context()
                if window_context is not None and current_context != window_context:
                    self.makeCurrent()
                    acquired = True
            except (AttributeError, RuntimeError):
                # 窗口尚未创建 context 时，渲染器仍需获得一次释放机会。
                acquired = False
            try:
                renderer.shutdown()
            finally:
                if acquired:
                    try:
                        self.doneCurrent()
                    except (AttributeError, RuntimeError):
                        pass

        def set_speech(
            self,
            text: str,
            *,
            mood: str = "neutral",
            visible: bool = True,
            speaking: bool = True,
        ) -> None:
            self._speech_text = str(text or "")
            self._speech_mood = str(mood or "neutral")
            self._speech_visible = bool(visible)
            if not speaking:
                # 文案气泡与实际语音是两条状态；空闲提示/点击提示
                # 不能沿用上一段音频的张嘴参数。
                setter = getattr(self.renderer, "set_mouth_viseme", None)
                if callable(setter):
                    try:
                        setter("Silence", intensity=0.0)
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        pass
            self.update()

        def set_interaction_feedback(
            self,
            value: Mapping[str, object] | None,
            *,
            visible: bool = True,
        ) -> bool:
            """精灵/OpenGL 路径复用气泡作为部位互动结果展示。"""

            if not visible or not isinstance(value, Mapping):
                self._interaction_feedback_visible = False
                self._interaction_feedback_text = ""
                self._refresh_surface_mask(force=True)
                self.update()
                return True
            labels = {
                "head": "猫猫头",
                "upper": "上半身",
                "body": "身体",
                "lower_left": "左腿",
                "lower_right": "右腿",
            }
            zone = str(value.get("zone", "") or "").strip().lower()
            phrase = str(value.get("phrase", "") or "").strip()
            affection = value.get("affection")
            current = ""
            applied = ""
            if isinstance(affection, Mapping):
                current = str(affection.get("current", "") or "").strip()
                applied = str(affection.get("applied", "") or "").strip()
            lines = [item for item in (labels.get(zone, ""), phrase) if item]
            visual = " · ".join(
                str(value.get(key, "") or "").strip()
                for key in ("expression", "motion")
                if str(value.get(key, "") or "").strip()
            )
            if visual:
                lines.append(visual)
            if current:
                lines.append(f"好感度 {current}" + (f"（本次 {applied}）" if applied else ""))
            self._interaction_feedback_text = "\n".join(lines)
            self._interaction_feedback_visible = bool(self._interaction_feedback_text)
            self._interaction_feedback_until = time.monotonic() + 1.8
            if phrase:
                # 互动气泡不等于正在播放音频；TTS 状态由宿主单独投影，
                # 这里先保持闭嘴，避免无 TTS/降级时首屏一直张嘴。
                self.set_speech(
                    phrase,
                    mood=str(value.get("mood", "neutral") or "neutral"),
                    speaking=False,
                )
            self._refresh_surface_mask(force=True)
            self.update()
            return True

        def set_legacy_shortcuts_enabled(self, enabled: bool) -> None:
            """控制内置按键回退；正式运行由 YAML 快捷键管理器接管。"""

            self._legacy_shortcuts_enabled = bool(enabled)

        def set_click_callback(
            self,
            callback: Callable[[Mapping[str, object]], None] | None,
        ) -> None:
            """设置普通单击回调；双击仍保留打开控制台语义。"""

            self._click_callback = callback

        def _event_local_point(self, event: object) -> QPoint | None:
            """读取鼠标事件相对桌宠窗口的整数坐标。"""

            position_getter = getattr(event, "position", None)
            if callable(position_getter):
                try:
                    position = position_getter()
                    to_point = getattr(position, "toPoint", None)
                    if callable(to_point):
                        return to_point()
                    if isinstance(position, QPoint):
                        return position
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            pos_getter = getattr(event, "pos", None)
            if callable(pos_getter):
                try:
                    point = pos_getter()
                    return point if isinstance(point, QPoint) else QPoint(point)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            return None

        def _cursor_model_hit(self) -> bool:
            """读取当前光标是否位于可见模型像素，供悬停状态使用。"""

            try:
                global_point = QCursor.pos()
                local_point = global_point - self.position()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
            return self._model_hit_test(local_point)

        def _model_hit_test(self, point: QPoint | None) -> bool:
            """按当前后端的真实可见内容判定交互命中。"""

            if point is None:
                return False
            if isinstance(self.renderer, SpriteRenderer):
                return self._sprite_hit_test(point)
            if isinstance(self.renderer, Live2DRenderer):
                hit_test = getattr(self.renderer, "hit_test", None)
                if callable(hit_test):
                    try:
                        return bool(hit_test(point.x(), point.y()))
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        return False
            # 未声明精确命中能力的后端不能把透明窗口当作可点击模型。
            return False

        def _model_part(self, point: QPoint | None) -> str | None:
            """优先使用原生部件 ID，缺失时回退到稳定画布部位。"""

            if point is None:
                return None
            if isinstance(self.renderer, SpriteRenderer):
                # 精灵窗口的上方 50px 是气泡区，不能直接用整窗中线划分
                # 部位；按与绘制/alpha 命中完全相同的内容坐标分类，模型
                # 中部才会稳定落到“身体”，而不是被误判成右腿。
                sprite_part = self._sprite_part(point)
                if sprite_part is not None:
                    return sprite_part
            fallback = classify_pet_part(point.x(), point.y(), self.width(), self.height())
            if not isinstance(self.renderer, Live2DRenderer):
                return fallback
            getter = getattr(self.renderer, "hit_parts", None)
            if not callable(getter):
                return fallback
            try:
                parts = tuple(str(item).strip().lower() for item in getter(point.x(), point.y()))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return fallback
            if fallback in {"lower_left", "lower_right"}:
                return fallback
            if any(
                any(marker in item for marker in ("head", "face", "eye", "mouth", "brow"))
                for item in parts
            ):
                return "head"
            if any("body" in item for item in parts):
                return "body"
            return fallback

        def _sprite_part(self, point: QPoint) -> str | None:
            """按精灵实际绘制坐标返回头、身体或左右下肢。"""

            renderer = self.renderer
            if not isinstance(renderer, SpriteRenderer):
                return None
            path = renderer.current_frame
            if path is None:
                return None
            image = self._sprite_image
            if image is None or self._sprite_image_path != path or image.isNull():
                image = QImage(str(path))
            if image.isNull():
                return None
            try:
                image.setDevicePixelRatio(1.0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            reference_width, reference_height = self._sprite_reference_size or (
                image.width(),
                image.height(),
            )
            reference_width = max(1, int(reference_width))
            reference_height = max(1, int(reference_height))
            area_top = 50
            area_width = max(1, self.width())
            area_height = max(1, self.height() - area_top)
            uniform_scale = min(area_width / reference_width, area_height / reference_height)
            if not math.isfinite(uniform_scale) or uniform_scale <= 0:
                return None
            canvas_width = max(1, round(reference_width * uniform_scale))
            canvas_height = max(1, round(reference_height * uniform_scale))
            scaled_width = max(1, round(image.width() * uniform_scale))
            scaled_height = max(1, round(image.height() * uniform_scale))
            target_x = (area_width - canvas_width) / 2.0 + (canvas_width - scaled_width) / 2.0
            target_y = area_top + area_height - canvas_height + (canvas_height - scaled_height)
            transform = getattr(renderer, "visual_transform", (0.0, 0.0))
            try:
                target_x += float(transform[0])
                target_y += float(transform[1])
            except (IndexError, TypeError, ValueError, OverflowError):
                return None
            relative_y = (float(point.y()) - target_y) / max(1.0, float(scaled_height))
            if not math.isfinite(relative_y) or relative_y < 0.0 or relative_y >= 1.0:
                return None
            if relative_y < 0.30:
                return "head"
            if relative_y < 0.72:
                return "body"
            relative_x = (float(point.x()) - target_x) / max(1.0, float(scaled_width))
            if not math.isfinite(relative_x) or relative_x < 0.0 or relative_x >= 1.0:
                return "body"
            return "lower_left" if relative_x < 0.5 else "lower_right"

        def _schedule_click(self, event: object) -> None:
            """延迟普通点击，避免双击同时触发一次分区互动。"""

            callback = self._click_callback
            if callback is None:
                return
            point = self._event_local_point(event)
            if point is None:
                return
            # QOpenGLWindow 透明区域仍属于顶层窗口；先按当前精灵帧的
            # alpha 做命中，避免点击桌面空白或气泡旁边也触发猫猫反馈。
            if not self._model_hit_test(point):
                return
            payload = make_pet_click_payload(
                {"x": point.x(), "y": point.y(), "button": "left"},
                width=self.width(),
                height=self.height(),
            )
            if payload is None:
                return
            part = self._model_part(point)
            if part is not None:
                payload["part"] = part
            self._click_token += 1
            token = self._click_token

            def emit() -> None:
                if token != self._click_token:
                    return
                try:
                    callback(payload)
                except Exception:
                    logger.exception("OpenGL pet click callback failed")

            try:
                QTimer.singleShot(PET_CLICK_DEBOUNCE_MS, emit)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                emit()

        def _sprite_hit_test(self, point: QPoint) -> bool:
            """按与绘制相同的等比变换检查精灵像素透明度。"""

            renderer = self.renderer
            if not isinstance(renderer, SpriteRenderer):
                return True
            path = renderer.current_frame
            if path is None:
                return False
            image = self._sprite_image
            if image is None or self._sprite_image_path != path or image.isNull():
                image = QImage(str(path))
            if image.isNull():
                return False
            try:
                image.setDevicePixelRatio(1.0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            reference_width, reference_height = self._sprite_reference_size or (
                image.width(),
                image.height(),
            )
            reference_width = max(1, int(reference_width))
            reference_height = max(1, int(reference_height))
            area_top = 50
            area_width = max(1, self.width())
            area_height = max(1, self.height() - area_top)
            uniform_scale = min(
                area_width / reference_width,
                area_height / reference_height,
            )
            if not math.isfinite(uniform_scale) or uniform_scale <= 0:
                return False
            canvas_width = max(1, round(reference_width * uniform_scale))
            canvas_height = max(1, round(reference_height * uniform_scale))
            scaled_width = max(1, round(image.width() * uniform_scale))
            scaled_height = max(1, round(image.height() * uniform_scale))
            target_x = (area_width - canvas_width) / 2.0 + (canvas_width - scaled_width) / 2.0
            target_y = area_top + area_height - canvas_height + (canvas_height - scaled_height)
            transform = getattr(renderer, "visual_transform", (0.0, 0.0))
            try:
                target_x += float(transform[0])
                target_y += float(transform[1])
            except (IndexError, TypeError, ValueError):
                pass
            local_x = float(point.x()) - target_x
            local_y = float(point.y()) - target_y
            if local_x < 0 or local_y < 0 or local_x >= scaled_width or local_y >= scaled_height:
                return False
            source_x = min(image.width() - 1, max(0, int(local_x / uniform_scale)))
            source_y = min(image.height() - 1, max(0, int(local_y / uniform_scale)))
            try:
                return image.pixelColor(source_x, source_y).alpha() > 8
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # 旧 Qt 绑定没有 pixelColor 时保守采用绘制包围盒，仍不影响
                # Web Live2D 的 raw drawable 命中路径。
                return True

        def set_always_on_top(self, enabled: bool) -> dict[str, object]:
            """切换窗口置顶标志，并回读 Qt 实际保留的状态。"""

            requested = bool(enabled)
            if self._drag_transaction_active():
                return {
                    "status": "cancelled",
                    "enabled": self.is_always_on_top(),
                    "reason": "user interaction active",
                }
            was_visible = bool(self.isVisible())
            position = self.position()
            geometry_generation = self._invalidate_geometry()
            try:
                self.setFlag(Qt.WindowStaysOnTopHint, requested)
                if was_visible and not self.isVisible():
                    self.show()
                actual = bool(self.flags() & Qt.WindowStaysOnTopHint)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("OpenGL topmost update failed: %s", type(exc).__name__)
                return {
                    "status": "unavailable",
                    "enabled": self._always_on_top,
                    "reason": "窗口置顶暂时无法切换，请稍后重试",
                }
            self._restore_position(position, geometry_generation=geometry_generation)
            self._schedule_surface_input_shape_restore()
            self._always_on_top = actual
            if actual != requested:
                return {
                    "status": "unavailable",
                    "enabled": actual,
                    "reason": "window system did not retain the requested topmost flag",
                }
            platform_status, detail = self._platform_capability_status("always_on_top")
            if platform_status != "available":
                return {
                    "status": platform_status,
                    "enabled": actual,
                    "detail": detail,
                }
            return {"status": "available", "enabled": actual}

        def _restore_position(
            self,
            position: QPoint | None,
            *,
            geometry_generation: int | None = None,
        ) -> None:
            """恢复切换顶层窗口标志时可能被窗口系统重置的坐标。"""

            if position is None:
                return

            def apply() -> None:
                if (
                    not self._geometry_is_current(geometry_generation)
                    or self._drag_transaction_active()
                ):
                    return
                try:
                    self.setPosition(position)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass

            apply()
            try:
                QTimer.singleShot(0, apply)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        def _platform_capability_status(self, name: str) -> tuple[str, str]:
            """把窗口平台能力映射为宿主可展示的状态。"""

            platform = self.platform
            if platform is None:
                return "available", ""
            backend = str(getattr(platform, "backend", "") or "").strip().lower()
            now = time.monotonic()
            cached = self._topmost_capability_cache
            if name == "always_on_top" and cached is not None:
                cached_backend, cached_status, cached_detail, cached_at = cached
                if cached_backend == backend and now - cached_at < 1.0:
                    return cached_status, cached_detail
            probe = getattr(platform, "probe", None)
            if callable(probe):
                try:
                    snapshot = probe()
                    capability_getter = getattr(snapshot, "capability", None)
                    capability = capability_getter(name) if callable(capability_getter) else None
                    if capability is not None:
                        state = getattr(getattr(capability, "state", None), "value", None)
                        detail = str(getattr(capability, "detail", "") or "")
                        normalized = str(state or "unknown").strip().lower()
                        if normalized in {"available", "degraded", "unavailable"}:
                            if name == "always_on_top":
                                self._topmost_capability_cache = (
                                    backend,
                                    normalized,
                                    detail,
                                    now,
                                )
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
            """返回窗口当前置顶状态。"""

            try:
                return bool(self.flags() & Qt.WindowStaysOnTopHint)
            except (AttributeError, RuntimeError, TypeError):
                return bool(self._always_on_top)

        def always_on_top_status(self) -> dict[str, object]:
            """返回置顶实际旗标及平台能力边界。"""

            actual = self.is_always_on_top()
            status, detail = self._platform_capability_status("always_on_top")
            result: dict[str, object] = {"status": status, "enabled": actual}
            if detail:
                result["detail"] = str(detail)[:180]
            return result

        def toggle_always_on_top(self) -> dict[str, object]:
            """在当前状态基础上切换置顶。"""

            return self.set_always_on_top(not self.is_always_on_top())

        def toggle_visibility(self) -> dict[str, object]:
            """切换桌宠可见性。"""

            try:
                if self.isVisible():
                    # 隐藏窗口不会再收到 leave；清理悬停和拖动状态，避免
                    # 恢复显示后行为服务继续误判用户正在交互。
                    self._clear_drag()
                    self._pointer_over = False
                    self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                    self.hide()
                    visible = False
                else:
                    self.show()
                    visible = bool(self.isVisible())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("OpenGL visibility update failed: %s", type(exc).__name__)
                return {
                    "status": "unavailable",
                    "visible": False,
                    "reason": "桌宠显示状态暂时无法切换，请稍后重试",
                }
            return {
                "status": "available",
                "visible": visible,
            }

        def move_to(self, x: float, y: float, *, duration_ms: int = 800) -> dict[str, object]:
            # BehaviorService 使用 ``duration_ms=0`` 提交连续自主步进。
            # 这些步进已经由行为层限速，不应每一步都清空 Live2D 注视
            # 速度，否则走路会在中性姿态和光标姿态之间反复跳变。
            try:
                autonomous_step = int(duration_ms) <= 0
            except (TypeError, ValueError, OverflowError):
                autonomous_step = False
            if self._drag_transaction_active():
                # 用户拖动拥有窗口几何的最高优先级；迟到的自主/模型移动
                # 不能覆盖当前指针目标，否则会出现回弹和视觉抽搐。
                return {"status": "cancelled", "reason": "user interaction active"}
            if autonomous_step and time.monotonic() < self._autonomous_block_until:
                return {"status": "cancelled", "reason": "user interaction settling"}
            geometry_generation = self._geometry_generation
            if autonomous_step:
                geometry_generation = self._begin_autonomous_geometry()
            else:
                geometry_generation = self._invalidate_geometry()
            self._last_cursor_target = None
            self._last_window_position = None
            if not autonomous_step:
                notifier = getattr(self.renderer, "notify_window_moved", None)
                if callable(notifier):
                    try:
                        notifier()
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        logger.debug("failed to reset Live2D tracking before move", exc_info=True)
            try:
                target_x = int(round(float(x)))
                target_y = int(round(float(y)))
            except (TypeError, ValueError, OverflowError):
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
            if self.platform is not None:
                try:
                    capability = self.platform.move_overlay(self, target_x, target_y)
                    if inspect.isawaitable(capability):
                        # 调用期 reservation 已经完成代次失效；pending 交给
                        # 实际运行的协程，避免 task 尚未开始就取消时泄漏。
                        if autonomous_step:
                            self._end_autonomous_geometry()
                        operation = self._move_after_dispatch(
                            capability,
                            target_x,
                            target_y,
                            autonomous_step=autonomous_step,
                            geometry_generation=geometry_generation,
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
                        "x": target_x,
                        "y": target_y,
                        "detail": str(detail or ""),
                    }
                    if autonomous_step:
                        self._end_autonomous_geometry()
                    return result
                except asyncio.CancelledError:
                    if autonomous_step:
                        self._end_autonomous_geometry()
                    raise
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    if autonomous_step:
                        self._end_autonomous_geometry()
                    raise
                except BaseException:
                    if autonomous_step:
                        self._end_autonomous_geometry()
                    raise
            # QWindow 的位置契约以 QPoint 为主；PySide 某些版本额外暴露
            # 双整数重载，但不依赖该实现差异，保证无平台适配器时也能移动。
            try:
                self.setPosition(QPoint(target_x, target_y))
            except asyncio.CancelledError:
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
            except BaseException:
                if autonomous_step:
                    self._end_autonomous_geometry()
                raise
            if autonomous_step:
                self._end_autonomous_geometry()
            return {"status": "available", "x": target_x, "y": target_y}

        def movement_bounds(self) -> tuple[int, int, int, int] | None:
            """返回桌宠左上角可移动的屏幕范围。"""

            try:
                from PySide6.QtGui import QGuiApplication

                screen = self.screen()
                if screen is None:
                    screen = QGuiApplication.screenAt(self.position())
                if screen is None:
                    screen = QGuiApplication.primaryScreen()
                if screen is None:
                    return None
                area = screen.availableGeometry()
                left = int(area.left())
                top = int(area.top())
                width = max(1, int(self.width()))
                height = max(1, int(self.height()))
                # 小屏或高 DPI 下窗口可能大于可用区域；仍返回一个可用的
                # 单点边界，避免控制台和自主行动收到反向范围而放弃移动。
                right = max(left, int(area.right()) - width + 1)
                bottom = max(top, int(area.bottom()) - height + 1)
                return left, top, right, bottom
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return None

        def set_display_size(self, preset: str) -> dict[str, object]:
            """按小/标准/大预设调整窗口，保持渲染器等比绘制。"""

            selected = display_size_preset(preset)
            if selected is None:
                return {"status": "unavailable", "reason": "display size preset is invalid"}
            key, (width, height) = selected
            try:
                position = self.position()
                self.resize(width, height)
                bounds = self.movement_bounds()
                if position is not None and bounds is not None:
                    position = QPoint(
                        max(bounds[0], min(position.x(), bounds[2])),
                        max(bounds[1], min(position.y(), bounds[3])),
                    )
                if position is not None:
                    self.setPosition(position)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("OpenGL display size update failed: %s", type(exc).__name__)
                return {
                    "status": "unavailable",
                    "reason": "桌宠大小暂时无法调整，请稍后重试",
                }
            return {"status": "available", "preset": key, "width": width, "height": height}

        def is_user_interacting(self) -> bool:
            """返回拖动、系统移动或鼠标悬停是否正在进行。"""

            if isinstance(self.renderer, Live2DRenderer) and self.renderer.model is None:
                # 原生模型尚未完成初始化时不要让自主行为移动窗口；若宿主
                # 已切换到精灵回退，renderer 不再是 Live2DRenderer，行为会
                # 在下一轮正常恢复。
                return True
            if self._window_locked:
                return False
            # 拖动释放后的短稳定窗口也视为交互中。行为循环可能比 Qt 的
            # 50ms 轮询更快，单靠 release 边沿会漏掉短按并让随机动作立即
            # 抢占渲染器；锁定窗口仍保持可自由漫游。
            if time.monotonic() < self._autonomous_block_until:
                return True
            return bool(
                self._drag_origin is not None or self._system_move_active or self._pointer_over
            )

        def set_expression(self, name: str) -> bool:
            changed = self.renderer.set_expression(name)
            if changed:
                self.expressionChanged.emit(str(name))
                self.update()
            return changed

        def set_expression_request(self, request: ExpressionRequest) -> Mapping[str, object]:
            """向当前原生渲染器提交结构化表情请求并刷新首个过渡帧。"""

            handler = getattr(self.renderer, "set_expression_request", None)
            if not callable(handler):
                return {"status": "unavailable", "reason": "expression request is unsupported"}
            result = handler(request)
            if not isinstance(result, Mapping):
                return {"status": "failed", "reason": "expression request result is invalid"}
            response = dict(result)
            if str(response.get("status", "")).strip().lower() == "started":
                self.expressionChanged.emit(self.renderer.state.expression)
                self.update()
            return response

        def play_motion(self, name: str) -> bool:
            changed = self.renderer.play_motion(name)
            if changed:
                self.motionChanged.emit(str(name))
            return changed

        def play_motion_request(self, request: MotionRequest) -> Mapping[str, object]:
            """向当前原生渲染器提交结构化动作请求并同步公开状态。"""

            handler = getattr(self.renderer, "play_motion_request", None)
            if not callable(handler):
                return {"status": "unavailable", "reason": "motion request is unsupported"}
            result = handler(request)
            if not isinstance(result, Mapping):
                return {"status": "failed", "reason": "motion request result is invalid"}
            response = dict(result)
            if str(response.get("status", "")).strip().lower() == "started":
                self.motionChanged.emit(self.renderer.state.motion)
                self.update()
            return response

        def set_direction(self, direction: str) -> object:
            """更新当前渲染器朝向，不改变窗口尺寸或位置。"""

            setter = getattr(self.renderer, "set_direction", None)
            if callable(setter):
                try:
                    changed = bool(setter(direction))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    changed = False
                if changed:
                    self.update()
                return changed
            return False

        def _input_transparency_active(self) -> bool | None:
            """读取 Qt 当前输入透明标志；读取失败时返回 ``None``。"""

            try:
                window_type = getattr(Qt, "WindowType", Qt)
                flag = window_type.WindowTransparentForInput
                getter = getattr(self, "flags", None)
                if not callable(getter):
                    getter = getattr(self, "windowFlags", None)
                if not callable(getter):
                    return None
                return bool(getter() & flag)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return None

        def set_click_through(self, enabled: bool) -> object:
            requested = bool(enabled)
            if self._window_locked and not requested:
                return {
                    "status": "unavailable",
                    "enabled": True,
                    "locked": True,
                    "reason": "window is locked; unlock it before restoring input",
                }
            if self.platform is not None:
                try:
                    result = self.platform.set_click_through(self, requested)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("OpenGL click-through request failed: %s", type(exc).__name__)
                    return {
                        "status": "unavailable",
                        "enabled": bool(self._click_through),
                        "reason": "点击穿透暂时无法切换，请使用控制台恢复入口",
                    }
                state = getattr(getattr(result, "state", None), "value", None)
                if state is None and isinstance(result, Mapping):
                    value = result.get("status", result.get("state", "unavailable"))
                    state = getattr(value, "value", value)
                state = str(state or "unavailable")
                # Wayland 只能确认 Qt 已保留该标志，不能确认 compositor
                # 的最终输入路由；degraded 仍记录为已请求，便于控制台
                # 提供恢复入口并避免状态与实际 Qt 标志脱节。
                actual = self._input_transparency_active()
                self._click_through = (
                    actual
                    if actual is not None
                    else requested and state in {"available", "degraded"}
                )
                self._schedule_surface_input_shape_restore()
                return result
            self._click_through = requested
            window_type = getattr(Qt, "WindowType", Qt)
            self.setFlag(window_type.WindowTransparentForInput, requested)
            self._schedule_surface_input_shape_restore()
            return {"state": "requested", "enabled": requested}

        def is_window_locked(self) -> bool:
            """返回锁定窗口状态；锁定只禁止鼠标操作，不暂停渲染和行为。"""

            return bool(self._window_locked)

        def _finish_window_lock_async(self, requested: bool, operation: object) -> object:
            """等待异步输入切换并原子收敛锁定状态。"""

            async def resolve() -> dict[str, object]:
                try:
                    result = await operation  # type: ignore[misc]
                except asyncio.CancelledError:
                    if requested:
                        self._window_locked = False
                    else:
                        self._window_locked = True
                    raise
                except Exception as exc:
                    logger.debug("OpenGL async window lock update failed: %s", type(exc).__name__)
                    if requested:
                        self._window_locked = False
                        return {
                            "status": "unavailable",
                            "locked": False,
                            "enabled": bool(self._click_through),
                            "reason": "窗口锁定暂时无法切换，请稍后重试",
                        }
                    self._window_locked = True
                    return {
                        "status": "unavailable",
                        "locked": True,
                        "enabled": bool(self._click_through),
                        "reason": "窗口锁定暂时无法解除，请稍后重试",
                    }
                if requested:
                    if not self._operation_succeeded(result, enabled=True):
                        self._window_locked = False
                        return {
                            "status": "unavailable",
                            "locked": False,
                            "enabled": bool(self._click_through),
                            "reason": "窗口锁定无法启用输入保护",
                        }
                    setter = getattr(self.renderer, "set_interaction_locked", None)
                    if callable(setter):
                        setter(True)
                    return {"status": "available", "locked": True, "enabled": True}
                if not self._operation_succeeded(
                    result, enabled=bool(self._click_through_before_lock)
                ):
                    self._window_locked = True
                    return {
                        "status": "unavailable",
                        "locked": True,
                        "enabled": bool(self._click_through),
                        "reason": "窗口锁定无法恢复之前的输入状态",
                    }
                self._click_through_before_lock = False
                setter = getattr(self.renderer, "set_interaction_locked", None)
                if callable(setter):
                    setter(False)
                return {
                    "status": "available",
                    "locked": False,
                    "enabled": bool(self._click_through),
                }

            return resolve()

        def set_window_locked(self, enabled: bool) -> object:
            """切换桌宠锁定。

            锁定会让窗口进入点击穿透，但不会调用行为服务的交互暂停标志；
            光标追踪由宿主定时器继续驱动，解锁时恢复锁定前的穿透状态。
            """

            requested = bool(enabled)
            if requested == self._window_locked:
                # 锁定和点击穿透是独立状态；重复读取时必须回报实际输入
                # 标志，不能把 requested 当成 enabled。
                return {
                    "status": "available",
                    "locked": requested,
                    "enabled": bool(self._click_through),
                }
            if requested:
                # 锁定可以在用户拖动尚未收到 release 时由快捷键/控制台
                # 触发；先结束旧拖动，避免解锁后沿用过期原点把窗口跳走。
                if self._drag_origin is not None or self._dragging or self._system_move_active:
                    self._clear_drag()
                self._click_through_before_lock = bool(self._click_through)
                self._window_locked = True
                result = self.set_click_through(True)
                if inspect.isawaitable(result):
                    return self._finish_window_lock_async(True, result)
                if not self._operation_succeeded(result, enabled=True):
                    self._window_locked = False
                    return {
                        "status": "unavailable",
                        "locked": False,
                        "enabled": bool(self._click_through),
                        "reason": "window lock could not enable input transparency",
                    }
                setter = getattr(self.renderer, "set_interaction_locked", None)
                if callable(setter):
                    setter(True)
                return {"status": "available", "locked": True, "enabled": True}
            self._window_locked = False
            restore = bool(self._click_through_before_lock)
            result = self._set_click_through_unlocked(restore)
            if inspect.isawaitable(result):
                return self._finish_window_lock_async(False, result)
            if not self._operation_succeeded(result, enabled=restore):
                self._window_locked = True
                setter = getattr(self.renderer, "set_interaction_locked", None)
                if callable(setter):
                    try:
                        setter(True)
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        logger.debug("failed to restore OpenGL interaction lock", exc_info=True)
                return {
                    "status": "unavailable",
                    "locked": True,
                    "enabled": bool(self._click_through),
                    "reason": "window lock could not restore the previous input state",
                }
            self._click_through_before_lock = False
            setter = getattr(self.renderer, "set_interaction_locked", None)
            if callable(setter):
                setter(False)
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

        def _set_click_through_unlocked(self, enabled: bool) -> object:
            """在已解除锁定的内部路径中恢复点击穿透状态。"""

            locked = self._window_locked
            self._window_locked = False
            try:
                return self.set_click_through(enabled)
            finally:
                self._window_locked = locked

        def update_cursor_tracking(self) -> bool:
            """读取全局光标并投影到渲染器，不依赖窗口是否接收鼠标。"""

            if (
                self._drag_origin is not None
                or self._dragging
                or self._system_move_active
                or self._renderer_dragging
            ):
                # 拖动期间窗口跟随指针移动，局部坐标保持稳定；暂停
                # 定时器追踪可避免反复清空注视平滑速度造成抽搐。
                return True
            try:
                global_point = QCursor.pos()
                position = self.position()
                current_position = (int(position.x()), int(position.y()))
                previous_position = self._last_window_position
                self._last_window_position = current_position
                # 自主游走已经由 move_to 建立几何稳定窗；不要把它的每个
                # 小步当成外部窗口跳变并重置注视速度，否则行走时会出现
                # 姿态反复回中性的视觉抽搐。稳定窗外仍保留真实外部移动
                # 的通知路径。
                if (
                    previous_position is not None
                    and current_position != previous_position
                    and not self._autonomous_geometry_active()
                ):
                    notifier = getattr(self.renderer, "notify_window_moved", None)
                    if callable(notifier):
                        notifier()
                width = max(1, int(self.width()))
                height = max(1, int(self.height()))
                local_x = max(0, min(width - 1, global_point.x() - position.x()))
                local_y = max(0, min(height - 1, global_point.y() - position.y()))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
            target = (local_x, local_y, width, height)
            if target == self._last_cursor_target:
                return True
            setter = getattr(self.renderer, "set_cursor_target", None)
            if callable(setter):
                try:
                    result = setter(local_x, local_y, width, height)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    return False
                # 页面/模型尚未就绪时渲染器可以显式拒绝坐标；保留缓存
                # 未命中状态，下一次全局追踪才能重试。同样兼容旧渲染器
                # 不返回值（None）的实现。
                if result is False:
                    return False
                self._last_cursor_target = target
                return True
            # 精灵或旧原生运行时没有注视参数时，至少保持朝向跟随光标。
            direction_setter = getattr(self.renderer, "set_direction", None)
            if callable(direction_setter):
                try:
                    result = direction_setter("A" if local_x < width / 2 else "B")
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    return False
                if result is False:
                    return False
                self._last_cursor_target = target
            return True

        def begin_click_through_override(self) -> object:
            """按住恢复快捷键时暂时取消点击穿透。"""

            if self._click_through_override_active:
                return {"status": "available", "enabled": False, "temporary": True}
            self._click_through_before_override = bool(self._click_through)
            self._click_through_override_active = True
            if not self._click_through_before_override:
                return {"status": "available", "enabled": False, "temporary": True}
            result = self.set_click_through(False)
            state = getattr(getattr(result, "state", None), "value", None)
            if isinstance(result, Mapping):
                state = result.get("status", result.get("state", state or ""))
            if str(getattr(state, "value", state) or "").strip().lower() in {
                "unavailable",
                "unknown",
            }:
                self._click_through_override_active = False
                self._click_through_before_override = None
            return result

        def end_click_through_override(self) -> object:
            """释放恢复快捷键时恢复按住前的点击穿透状态。"""

            if not self._click_through_override_active:
                return {"status": "available", "enabled": bool(self._click_through)}
            previous = bool(self._click_through_before_override)
            self._click_through_override_active = False
            self._click_through_before_override = None
            if not previous:
                return {"status": "available", "enabled": False}
            return self.set_click_through(True)

        @staticmethod
        def _event_global_point(event) -> QPoint | None:
            """兼容 Qt6 鼠标事件的浮点和整数全局坐标接口。"""

            getter = getattr(event, "globalPosition", None)
            if callable(getter):
                try:
                    position = getter()
                    to_point = getattr(position, "toPoint", None)
                    if callable(to_point):
                        return to_point()
                    if isinstance(position, QPoint):
                        return position
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            getter = getattr(event, "globalPos", None)
            if callable(getter):
                try:
                    point = getter()
                    return point if isinstance(point, QPoint) else QPoint(point)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            return None

        def _clear_drag(self) -> None:
            had_drag = self._drag_transaction_active() or self._renderer_dragging
            watchdog = self._drag_watchdog_timer
            if watchdog is not None:
                try:
                    watchdog.stop()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            self._set_renderer_dragging(False)
            self._drag_origin = None
            self._window_origin = None
            self._dragging = False
            self._system_move_active = False
            self._interaction_interrupt_notified = False
            self._pending_drag_target = None
            self._last_drag_target = None
            self._last_drag_submit_at = 0.0
            timer = self._drag_flush_timer
            if timer is not None:
                try:
                    timer.stop()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            self._last_cursor_target = None
            try:
                self.setMouseGrabEnabled(False)
            except (AttributeError, RuntimeError):
                pass
            if had_drag:
                self._invalidate_geometry()

        def _arm_drag_watchdog(self) -> None:
            """为原生/Wayland 拖动建立有界释放兜底。"""

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
            """清理丢失 release 的拖动事务。"""

            if not self._drag_transaction_active() and not self._renderer_dragging:
                return
            self._clear_drag()
            self._pointer_over = False
            try:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        def _start_compositor_move(self) -> bool:
            """在 Wayland 上优先交给合成器处理顶层拖动。"""

            backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
            if backend != "wayland":
                return False
            starter = getattr(self, "startSystemMove", None)
            if not callable(starter):
                return False
            try:
                accepted = bool(starter())
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False
            self._system_move_active = accepted
            if accepted:
                self._set_renderer_dragging(True)
                self._arm_drag_watchdog()
            return accepted

        def _move_drag_target(self, target: QPoint) -> None:
            """合并高频拖动坐标，并按受控频率提交最新位置。"""

            if self._shutdown:
                return
            target_tuple = (int(target.x()), int(target.y()))
            if target_tuple == self._pending_drag_target or target_tuple == self._last_drag_target:
                return
            self._pending_drag_target = target_tuple
            now = time.monotonic()
            if (
                self._last_drag_submit_at <= 0.0
                or now - self._last_drag_submit_at >= _DRAG_MOVE_INTERVAL_SECONDS
            ):
                self._flush_drag_target()
            else:
                self._schedule_drag_flush()

        def _schedule_drag_flush(self) -> None:
            """在下一可用拖动时间片冲刷最新坐标。"""

            if self._pending_drag_target is None:
                return
            try:
                timer = self._drag_flush_timer
                if timer is None:
                    timer = QTimer(self)
                    timer.setSingleShot(True)
                    try:
                        timer.setTimerType(Qt.TimerType.PreciseTimer)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass
                    timer.timeout.connect(self._flush_drag_target)
                    self._drag_flush_timer = timer
                remaining = max(
                    0.001,
                    _DRAG_MOVE_INTERVAL_SECONDS - (time.monotonic() - self._last_drag_submit_at),
                )
                timer.start(max(1, int(math.ceil(remaining * 1000.0))))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                # 没有活动 Qt 事件循环时，释放事件仍会强制冲刷队列。
                return

        def _flush_drag_target(self, *, force: bool = False) -> None:
            """只提交队列中的最后坐标，避免窗口追赶过期 move。"""

            if self._shutdown:
                self._pending_drag_target = None
                return
            target_tuple = self._pending_drag_target
            if target_tuple is None:
                return
            if (
                not force
                and self._last_drag_submit_at > 0.0
                and time.monotonic() - self._last_drag_submit_at < _DRAG_MOVE_INTERVAL_SECONDS
            ):
                self._schedule_drag_flush()
                return
            self._pending_drag_target = None
            if target_tuple == self._last_drag_target:
                return
            self._apply_drag_target(QPoint(*target_tuple))

        def _set_renderer_dragging(self, dragging: bool) -> None:
            """同步拖动状态到渲染器，冻结动作时钟和注视速度。"""

            value = bool(dragging)
            if not value:
                self._interaction_interrupt_notified = False
            if self._renderer_dragging == value:
                return
            self._invalidate_geometry()
            self._autonomous_geometry_until = 0.0
            if not value:
                self._autonomous_block_until = max(
                    self._autonomous_block_until,
                    time.monotonic() + _AUTONOMOUS_RESUME_GRACE_SECONDS,
                )
                # 拖动释放前已经冲刷了最后一个窗口坐标；把它作为新的
                # 追踪基线，避免下一次 16ms 光标定时器仅因“旧坐标→新坐标”
                # 再次调用 notify_window_moved，造成模型刚松手就回中性抖动。
                try:
                    position = self.position()
                    self._last_window_position = (int(position.x()), int(position.y()))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    self._last_window_position = None
            self._renderer_dragging = value
            if not value:
                # 释放后从当前时刻继续计时，避免把拖动期间的暂停时间
                # 作为一个超大 dt 交给动作/物理更新。
                self._last_frame_at = time.monotonic()
            setter = getattr(self.renderer, "set_dragging", None)
            if callable(setter):
                try:
                    setter(value)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to update OpenGL dragging state", exc_info=True)
            if not value:
                self._schedule_surface_input_shape_restore()

        async def _move_after_dispatch(
            self,
            capability: object,
            x: int,
            y: int,
            *,
            autonomous_step: bool,
            geometry_generation: int,
            autonomous_geometry_deferred: bool = False,
        ) -> dict[str, object]:
            """等待异步平台移动，并拒绝已被新拖动取代的回执。"""

            geometry_owned = bool(autonomous_step and not autonomous_geometry_deferred)
            try:
                if autonomous_step and autonomous_geometry_deferred:
                    # move_to 已释放调用期 reservation；只有协程开始运行才
                    # 持有 pending，从而覆盖 pre-start cancellation。
                    self._activate_autonomous_geometry()
                    geometry_owned = True
                result = await capability
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("OpenGL async move failed: %s", type(exc).__name__)
                if (
                    not autonomous_step
                    and self._geometry_is_current(geometry_generation)
                    and not self._drag_transaction_active()
                ):
                    self._schedule_surface_input_shape_restore()
                return {
                    "status": "unavailable",
                    "x": int(x),
                    "y": int(y),
                    "detail": "桌宠位置暂时无法调整，请稍后重试",
                }
            finally:
                if geometry_owned:
                    self._end_autonomous_geometry()
            if not self._geometry_is_current(geometry_generation):
                return {
                    "status": "cancelled",
                    "x": int(x),
                    "y": int(y),
                    "detail": "geometry transaction superseded",
                }
            if not autonomous_step:
                self._schedule_surface_input_shape_restore()
            state = getattr(getattr(result, "state", None), "value", None)
            detail = getattr(result, "detail", "")
            if isinstance(result, Mapping):
                value = result.get("status", result.get("state", "unavailable"))
                state = getattr(value, "value", value)
                detail = result.get("detail", result.get("reason", ""))
            return {
                "status": str(state or "unavailable"),
                "x": int(x),
                "y": int(y),
                "detail": str(detail or ""),
            }

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

        def _apply_drag_target(self, target: QPoint) -> None:
            """提交一个已合并的拖动目标。"""

            if self._shutdown:
                return
            # Linux 平台适配器内部已经调用 QWindow.setPosition。先执行
            # Qt 原生移动再调用适配器会在拖动高频事件中重复提交几何，
            # 触发 OpenGL 重绘竞态并造成画面抽搐；平台返回能力对象后
            # 视为请求已提交，不再因瞬时回读滞后重复调用 Qt 移动。
            moved = False
            mover = getattr(self.platform, "move_overlay", None)
            if callable(mover):
                try:
                    result = mover(self, target.x(), target.y())
                    # 鼠标事件处理器不能阻塞等待后台协程。调度器在 Qt 主线程
                    # 通常会同步返回能力对象；若外部适配器返回协程，主动关闭
                    # 未执行的协程，避免退出时出现未等待警告。
                    if inspect.iscoroutine(result):
                        try:
                            result.close()
                        except (AttributeError, RuntimeError, TypeError):
                            pass
                    else:
                        observed = self.position()
                        moved = observed == QPoint(target) or self._move_result_accepted(result)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("overlay move adapter failed during drag", exc_info=True)
            if not moved:
                try:
                    self.setPosition(target)
                    moved = True
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("Qt window move failed during drag", exc_info=True)
            self._last_drag_submit_at = time.monotonic()
            self._last_drag_target = (int(target.x()), int(target.y()))

        def mousePressEvent(self, event) -> None:  # noqa: N802
            if self._window_locked:
                self._clear_drag()
                event.ignore()
                return
            point = self._event_global_point(event)
            button = event.button()
            if button == Qt.LeftButton and point is not None:
                local_point = self._event_local_point(event)
                # QOpenGLWindow 的透明承载面覆盖整个窗口；只有模型/精灵
                # 实际像素才允许启动拖动。否则点击桌面空白会抢走拖动状态，
                # 并在下一次 move 时把桌宠带走，表现为“透明区域也能拖动”。
                if not self._model_hit_test(local_point):
                    self._drag_origin = None
                    self._window_origin = None
                    self._dragging = False
                    self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                    event.ignore()
                    return
                self._invalidate_geometry()
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
                self._notify_interaction_started()
                if self._start_compositor_move():
                    event.accept()
                    return
                self._drag_origin = point
                self._window_origin = self.position()
                self._dragging = False
                # Wayland 上优先使用 startSystemMove；若 compositor 拒绝该
                # 请求，仍尝试 Qt 的鼠标抓取作为兼容回退。部分 compositor
                # 会忽略它，但不会影响已建立的系统移动路径。
                try:
                    self.setMouseGrabEnabled(True)
                except (AttributeError, RuntimeError):
                    pass
                self._arm_drag_watchdog()
            elif button == Qt.RightButton:
                local_point = self._event_local_point(event)
                if not self._model_hit_test(local_point):
                    self._clear_drag()
                    self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                    event.ignore()
                    return
                # 右键同时关闭穿透并交给宿主显示控制菜单；即使菜单不可用，
                # 这一操作也为用户留下恢复输入的机会。
                self._notify_interaction_started()
                self._clear_drag()
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                try:
                    self.set_click_through(False)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to restore input before context menu", exc_info=True)
                if point is not None:
                    self.contextMenuRequested.emit(point)
            event.accept()

        def enterEvent(self, event) -> None:  # noqa: N802
            """鼠标悬停时暂停自主漫游，避免桌宠追着指针移动。"""

            if self._window_locked:
                self._pointer_over = False
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                accept = getattr(event, "accept", None)
                if callable(accept):
                    accept()
                return
            self._pointer_over = self._cursor_model_hit()
            self.setCursor(
                QCursor(
                    Qt.CursorShape.OpenHandCursor
                    if self._pointer_over
                    else Qt.CursorShape.ArrowCursor
                )
            )
            accept = getattr(event, "accept", None)
            if callable(accept):
                accept()

        def leaveEvent(self, event) -> None:  # noqa: N802
            self._pointer_over = False
            if not self._dragging and not self._system_move_active:
                self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            accept = getattr(event, "accept", None)
            if callable(accept):
                accept()

        def mouseMoveEvent(self, event) -> None:  # noqa: N802
            if self._window_locked:
                self._clear_drag()
                event.ignore()
                return
            if self._system_move_active:
                event.accept()
                return
            local_point = self._event_local_point(event)
            if self._drag_origin is None or self._window_origin is None:
                self._pointer_over = self._model_hit_test(local_point)
                self.setCursor(
                    QCursor(
                        Qt.CursorShape.OpenHandCursor
                        if self._pointer_over
                        else Qt.CursorShape.ArrowCursor
                    )
                )
                # 即使透明区域不启动拖动，也消费 move，避免它继续冒泡到
                # 其它顶层窗口；位置保持不变且不会触发点击回调。
                event.accept()
                return
            # XTest、部分 Wayland 输入桥和触控板驱动会把拖动中的
            # MouseMove 报告为 NoButton；按下事件已经建立了拖动状态，
            # 这里等 MouseButtonRelease 清理，不能因按钮位缺失而中断。
            global_point = self._event_global_point(event)
            if global_point is None:
                event.ignore()
                return
            self._arm_drag_watchdog()
            self._pointer_over = True
            if not self._dragging:
                try:
                    from PySide6.QtWidgets import QApplication

                    threshold = int(QApplication.startDragDistance())
                except (ImportError, RuntimeError, TypeError, ValueError):
                    threshold = 4
                if (global_point - self._drag_origin).manhattanLength() < max(1, threshold):
                    event.accept()
                    return
                self._dragging = True
                self._set_renderer_dragging(True)
            delta = global_point - self._drag_origin
            target = self._window_origin + delta
            self._move_drag_target(target)
            event.accept()

        def mouseReleaseEvent(self, event) -> None:  # noqa: N802
            if self._window_locked:
                self._clear_drag()
                event.ignore()
                return
            if event.button() == Qt.LeftButton:
                was_dragging = self._dragging
                if was_dragging:
                    # 释放事件是最后一个可靠的全局坐标；补入 pending
                    # 后强制冲刷，避免高刷新率输入在最后一帧停在旧位置。
                    point = self._event_global_point(event)
                    if (
                        point is not None
                        and self._drag_origin is not None
                        and self._window_origin is not None
                    ):
                        self._pending_drag_target = (
                            int(self._window_origin.x() + point.x() - self._drag_origin.x()),
                            int(self._window_origin.y() + point.y() - self._drag_origin.y()),
                        )
                    self._flush_drag_target(force=True)
                self._clear_drag()
                self._pointer_over = self._cursor_model_hit()
                self.setCursor(
                    QCursor(
                        Qt.CursorShape.OpenHandCursor
                        if self._pointer_over
                        else Qt.CursorShape.ArrowCursor
                    )
                )
                if not was_dragging:
                    self._schedule_click(event)
                elif self._surface_mask_enabled:
                    self._refresh_surface_mask(force=True)
                if was_dragging:
                    self.update()
            event.accept()

        def focusOutEvent(self, event) -> None:  # noqa: N802
            """窗口失焦时收敛可能丢失 release 的拖动状态。"""

            if self._drag_transaction_active() or self._renderer_dragging:
                self._clear_drag()
            try:
                super().focusOutEvent(event)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

        def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
            """双击桌宠打开控制台，给无托盘环境提供可发现的入口。"""

            if self._window_locked:
                self._click_token += 1
                self._clear_drag()
                event.ignore()
                return
            if event.button() == Qt.LeftButton:
                # 顶层窗口是透明矩形，Qt 仍会把桌面空白区域报告为双击。
                # 只有真实模型/精灵像素命中才打开控制台，避免用户双击桌面
                # 其它位置时被桌宠抢走焦点或误弹控制台。
                if not self._model_hit_test(self._event_local_point(event)):
                    self._click_token += 1
                    self._clear_drag()
                    event.accept()
                    return
                self._click_token += 1
                self._clear_drag()
                self.consoleRequested.emit()
                event.accept()
                return
            super().mouseDoubleClickEvent(event)

        def keyPressEvent(self, event) -> None:  # noqa: N802
            """用回车打开最小输入框，保持对话入口不依赖额外 QWidget。"""

            if not self._legacy_shortcuts_enabled:
                return super().keyPressEvent(event)
            modifiers = event.modifiers()
            if (
                event.key() == Qt.Key_M
                and bool(modifiers & Qt.ControlModifier)
                and bool(modifiers & Qt.ShiftModifier)
            ):
                self.consoleRequested.emit()
                event.accept()
                return
            if event.key() in {Qt.Key_Return, Qt.Key_Enter}:
                try:
                    from PySide6.QtWidgets import QInputDialog

                    text, accepted = QInputDialog.getText(
                        None,
                        "MeaPet 对话",
                        "输入消息：",
                    )
                except (ImportError, RuntimeError):
                    text, accepted = "", False
                if accepted and str(text).strip():
                    self.textSubmitted.emit(str(text).strip())
                    return
            super().keyPressEvent(event)

        def shutdown(self) -> None:
            """停止帧计时器并释放当前渲染器；可重复调用。"""

            if self._shutdown:
                return
            self._shutdown = True
            self._window_locked = False
            self._click_through_before_lock = False
            self._last_cursor_target = None
            self._last_window_position = None
            self._click_token += 1
            self._click_callback = None
            self._timer.stop()
            drag_timer = self._drag_flush_timer
            self._drag_flush_timer = None
            if drag_timer is not None:
                try:
                    drag_timer.stop()
                    drag_timer.deleteLater()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            watchdog = self._drag_watchdog_timer
            self._drag_watchdog_timer = None
            if watchdog is not None:
                try:
                    watchdog.stop()
                    watchdog.deleteLater()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
            self._pending_drag_target = None
            self._autonomous_geometry_until = 0.0
            self._autonomous_geometry_pending = 0
            self._set_renderer_dragging(False)
            self._release_renderer(self.renderer)

        def closeEvent(self, event) -> None:  # noqa: N802
            # 用户关闭只隐藏到托盘；应用退出时由 shutdown() 统一释放资源。
            if not self._shutdown:
                self.closeRequested.emit()
                event.ignore()
                self.hide()
                return
            self._timer.stop()
            self.shutdown()
            super().closeEvent(event)

else:

    class PetOpenGLWindow:  # pragma: no cover
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("PySide6 is not installed")

        def reload_resources(
            self,
            resource_root: object,
            *,
            model_path: object | None = None,
            sprite_scale: float | None = None,
        ) -> Mapping[str, object]:
            del resource_root, model_path, sprite_scale
            return {"status": "unavailable", "reason": "PySide6 is not installed"}
