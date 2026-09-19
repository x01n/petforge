from __future__ import annotations

import inspect
import logging
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from core.rendering.actions import ExpressionRequest, MotionRequest
from gui.qt6.display_size import display_size_preset
from gui.qt6.pet_interaction import classify_pet_part, make_pet_click_payload
from gui.renderers.protocol import (
    RendererBackend,
    RendererCapabilities,
    RendererLifecycleState,
    RendererLifecycleStatus,
    RendererState,
    normalize_renderer_backend,
    renderer_backend_compatibility_alias,
)
from gui.renderers.sprite import SpriteRenderer

logger = logging.getLogger(__name__)

_VULKAN_API_NAME = "vulkan"
_INITIALIZATION_TIMEOUT_MS = 5_000
_FRAME_INTERVAL_MS = 33
_BUBBLE_HEIGHT = 58
_QML_DOCUMENT = """
import QtQuick

Item {
    id: root
    objectName: "vulkanPetRoot"
    property url frameSource
    property real frameOffsetX: 0
    property real frameOffsetY: 0
    property string speechText: ""
    property bool speechVisible: false
    property string feedbackText: ""
    property bool feedbackVisible: false
    readonly property bool frameReady: petImage.status === Image.Ready

    Item {
        id: stage
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.top: parent.top
        anchors.topMargin: 58
        anchors.bottom: parent.bottom

        Image {
            id: petImage
            objectName: "petImage"
            anchors.fill: parent
            source: root.frameSource
            fillMode: Image.PreserveAspectFit
            verticalAlignment: Image.AlignBottom
            horizontalAlignment: Image.AlignHCenter
            asynchronous: false
            cache: false
            mipmap: true
            smooth: true
            transform: Translate {
                x: root.frameOffsetX
                y: root.frameOffsetY
            }
        }
    }

    Rectangle {
        id: speechBubble
        objectName: "speechBubble"
        visible: root.speechVisible && root.speechText.length > 0
        x: 8
        y: 6
        width: Math.max(1, parent.width - 16)
        height: 48
        radius: 18
        color: "#EE221A2E"
        border.width: 1
        border.color: "#FFFF9DBE"

        Text {
            anchors.fill: parent
            anchors.margins: 10
            text: root.speechText
            color: "#FFFAF6FB"
            font.family: "LXGW WenKai"
            font.pixelSize: 15
            wrapMode: Text.Wrap
            elide: Text.ElideRight
            maximumLineCount: 2
        }
    }

    Rectangle {
        id: feedbackPanel
        objectName: "feedbackPanel"
        visible: root.feedbackVisible && root.feedbackText.length > 0
        width: Math.min(112, Math.max(80, parent.width * 0.28))
        height: Math.min(120, Math.max(60, parent.height * 0.24))
        x: parent.width - width - 6
        y: Math.max(62, (parent.height - height) / 2)
        radius: 12
        color: "#F0221A2E"
        border.width: 1
        border.color: "#FFFFB6CE"

        Text {
            anchors.fill: parent
            anchors.margins: 8
            text: root.feedbackText
            color: "#FFFFF5FA"
            font.family: "LXGW WenKai"
            font.pixelSize: 12
            wrapMode: Text.Wrap
            elide: Text.ElideRight
            maximumLineCount: 6
        }
    }
}
""".strip()


try:  # Qt 保持可选，核心模块和无 GUI 测试仍可导入。
    from PySide6.QtCore import (
        QCoreApplication,
        QEventLoop,
        QObject,
        QPoint,
        QRect,
        Qt,
        QTimer,
        QUrl,
        Signal,
    )
    from PySide6.QtGui import (
        QBitmap,
        QColor,
        QCursor,
        QGuiApplication,
        QImage,
        QRegion,
        QSurfaceFormat,
    )
    from PySide6.QtQml import QQmlComponent
    from PySide6.QtQuick import QQuickView, QQuickWindow, QSGRendererInterface

    pyside6_vulkan_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_vulkan_available = False


def _enum_name(value: object) -> str:
    """返回 Qt 枚举的稳定小写名称。"""

    name = getattr(value, "name", None)
    if name:
        return str(name).strip().lower()
    text = str(value or "").strip()
    return text.rsplit(".", 1)[-1].lower() if text else ""


def _finite_scale(value: object, *, default: float = 0.6) -> float:
    """把精灵缩放限制在配置契约范围内。"""

    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(scale):
        return default
    return max(0.1, min(2.0, scale))


def _finite_dimension(value: object, *, default: int = 1) -> int:
    """把外部尺寸限制为 Qt 可接受的有限正整数。"""

    try:
        dimension = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(dimension) or dimension <= 0.0:
        return default
    return max(1, min(16_384, int(round(dimension))))


def _fit_display_size_to_area(
    width: object,
    height: object,
    area: object | None,
) -> tuple[int, int]:
    """将显示预设限制在屏幕可用区域内，保证 Qt Quick 视口完整可见。"""

    safe_width = _finite_dimension(width)
    safe_height = _finite_dimension(height)
    if area is None:
        return safe_width, safe_height
    try:
        area_width = _finite_dimension(area.width())
        area_height = _finite_dimension(area.height())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return safe_width, safe_height
    return min(safe_width, area_width), min(safe_height, area_height)


def _visible_snapshot(image: object) -> bool:
    """确认场景图抓帧包含至少一个非透明像素。"""

    if not pyside6_vulkan_available or not isinstance(image, QImage) or image.isNull():
        return False
    try:
        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        size = int(converted.sizeInBytes())
        data = bytes(converted.constBits())[:size]
    except (AttributeError, BufferError, RuntimeError, TypeError, ValueError):
        return False
    return any(data[3::4])


if pyside6_vulkan_available:

    def prepare_vulkan_scenegraph() -> tuple[bool, str]:
        """在首个 Qt Quick 窗口前选择 Vulkan，并拒绝覆盖活动场景图。"""

        app = QGuiApplication.instance()
        if app is None:
            return False, "QGuiApplication must exist before creating the Vulkan host"
        existing = tuple(
            window for window in QGuiApplication.allWindows() if isinstance(window, QQuickWindow)
        )
        if existing:
            actual = _enum_name(existing[0].rendererInterface().graphicsApi())
            if actual != _VULKAN_API_NAME:
                return (
                    False,
                    "an existing Qt Quick window uses a non-Vulkan scenegraph; "
                    "close it before switching",
                )
        try:
            # Qt Quick 的 alpha buffer 与 graphics API 都是首窗前的全局选择；
            # 仅在 QQuickView 构造后设置局部 QSurfaceFormat，部分驱动会先
            # 创建不带 alpha 的 QRhi swapchain，从而把透明区域合成为黑底。
            QQuickWindow.setDefaultAlphaBuffer(True)
            QQuickWindow.setGraphicsApi(QSGRendererInterface.GraphicsApi.Vulkan)
            requested = _enum_name(QQuickWindow.graphicsApi())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"Qt Quick rejected Vulkan graphics API: {type(exc).__name__}"
        if requested != _VULKAN_API_NAME:
            return False, f"Qt Quick retained graphics API '{requested or 'unknown'}'"
        if not QQuickWindow.hasDefaultAlphaBuffer():
            return False, "Qt Quick rejected the default alpha buffer request"
        return True, "Qt Quick Vulkan graphics API and alpha buffer requested"

    class VulkanPetHost(QQuickView):
        """兼容宿主；没有 Live2D Vulkan 提供者时拒绝初始化。"""

        expressionChanged = Signal(str)
        motionChanged = Signal(str)
        textSubmitted = Signal(str)
        consoleRequested = Signal()
        contextMenuRequested = Signal(QPoint)
        closeRequested = Signal()

        def __init__(
            self,
            sprite_renderer: SpriteRenderer,
            *,
            platform: object | None = None,
            size: tuple[int, int] | None = None,
            sprite_scale: float = 0.6,
            always_on_top: bool = True,
            compatibility_alias: str = "",
        ) -> None:
            if isinstance(sprite_renderer, SpriteRenderer):
                raise RuntimeError("Vulkan Live2D host does not accept a sprite renderer")
            raise TypeError("VulkanPetHost requires a Live2D Vulkan provider")
            prepared, reason = prepare_vulkan_scenegraph()
            if not prepared:
                raise RuntimeError(reason)
            super().__init__()
            self._sprite_renderer = sprite_renderer
            self.platform = platform
            self._sprite_scale = _finite_scale(sprite_scale)
            self._always_on_top = bool(always_on_top)
            self._compatibility_alias = renderer_backend_compatibility_alias(compatibility_alias)
            self._lifecycle = RendererLifecycleStatus(
                RendererBackend.VULKAN.value,
                RendererLifecycleState.CREATED,
                _VULKAN_API_NAME,
                reason="scenegraph has not been initialized",
            )
            self._scenegraph_initialized = False
            self._scenegraph_error = ""
            self._frame_verified = False
            self._frame_presented = False
            self._shutdown = False
            self._dragging = False
            self._press_global: QPoint | None = None
            self._window_origin: QPoint | None = None
            self._pointer_over = False
            self._window_locked = False
            self._click_through = False
            self._click_through_before_lock = False
            self._click_through_override_active = False
            self._click_through_before_override = False
            self._surface_mask_enabled = False
            self._click_callback: Callable[[Mapping[str, object]], None] | None = None
            self._interaction_interrupt: Callable[[], object] | None = None
            self._activity_notifier: Callable[[], object] | None = None
            self._speech_text = ""
            self._speech_visible = False
            self._feedback_text = ""
            self._feedback_visible = False
            self._feedback_until = 0.0
            self._last_frame_path: Path | None = None
            self._last_frame_image: QImage | None = None
            self._last_cursor_target: tuple[int, int, int, int] | None = None
            self._last_size: tuple[int, int, float] | None = None
            self._last_frame_at = time.monotonic()

            fmt = QSurfaceFormat()
            fmt.setAlphaBufferSize(8)
            self.setFormat(fmt)
            self.setColor(QColor(0, 0, 0, 0))
            flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
            flags |= Qt.WindowType.WindowDoesNotAcceptFocus
            if self._always_on_top:
                flags |= Qt.WindowType.WindowStaysOnTopHint
            self.setFlags(flags)
            self.setResizeMode(QQuickView.ResizeMode.SizeRootObjectToView)
            self.setPersistentGraphics(True)
            self.setPersistentSceneGraph(True)
            qml_url = QUrl("qrc:/meapet/vulkan_host.qml")
            component = QQmlComponent(self.engine())
            component.setData(_QML_DOCUMENT.encode("utf-8"), qml_url)
            component_deadline = time.monotonic() + 2.0
            while (
                component.status() is QQmlComponent.Status.Loading
                and time.monotonic() < component_deadline
            ):
                QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 25)
                time.sleep(0.005)
            if component.status() is QQmlComponent.Status.Error:
                errors = "; ".join(str(item) for item in component.errors())
                raise RuntimeError("Vulkan host QML failed to load: " + (errors or "unknown error"))
            root = component.create()
            if root is None:
                raise RuntimeError("Vulkan host QML did not create a root item")
            self._qml_component = component
            self.setContent(qml_url, component, root)
            self._root: QObject = root
            image_item = root.findChild(QObject, "petImage")
            if image_item is None:
                raise RuntimeError("Vulkan host QML is missing the pet image item")
            self._image_item: QObject = image_item
            self.sceneGraphInitialized.connect(
                self._on_scenegraph_initialized,
                Qt.ConnectionType.DirectConnection,
            )
            self.sceneGraphInvalidated.connect(
                self._on_scenegraph_invalidated,
                Qt.ConnectionType.DirectConnection,
            )
            self.sceneGraphError.connect(self._on_scenegraph_error)
            self.frameSwapped.connect(self._on_frame_swapped)
            self.resize(*(size or self._preferred_size()))
            self._sync_frame(force=True)

            self._timer = QTimer(self)
            self._timer.setInterval(_FRAME_INTERVAL_MS)
            self._timer.timeout.connect(self._tick)

        @property
        def renderer(self) -> VulkanPetHost:
            """让控制器读取 Vulkan 宿主本身的真实能力。"""

            return self

        @property
        def view(self) -> VulkanPetHost:
            """与 Web 宿主保持一致的窗口访问接口。"""

            return self

        @property
        def capabilities(self) -> RendererCapabilities:
            source = self._sprite_renderer.capabilities
            return RendererCapabilities(
                backend=RendererBackend.VULKAN.value,
                available=self._lifecycle.available,
                expressions=source.expressions,
                motions=source.motions,
                message=(
                    "Qt Quick Vulkan scenegraph is initialized and frame-verified"
                    if self._lifecycle.available
                    else self._lifecycle.reason
                ),
            )

        @property
        def state(self) -> RendererState:
            return self._sprite_renderer.state

        @property
        def lifecycle_status(self) -> RendererLifecycleStatus:
            return self._lifecycle

        @property
        def initialization_pending(self) -> bool:
            return self._lifecycle.state in {
                RendererLifecycleState.CREATED,
                RendererLifecycleState.INITIALIZING,
            }

        @property
        def initialized(self) -> bool:
            return self._lifecycle.available

        @property
        def compatibility_alias(self) -> str:
            return self._compatibility_alias

        @property
        def actual_graphics_api(self) -> str:
            return self._lifecycle.actual_api

        @property
        def last_size(self) -> tuple[int, int, float] | None:
            return self._last_size

        def _set_lifecycle(
            self,
            state: RendererLifecycleState,
            *,
            actual_api: str = "",
            reason: str = "",
        ) -> None:
            self._lifecycle = RendererLifecycleStatus(
                RendererBackend.VULKAN.value,
                state,
                _VULKAN_API_NAME,
                actual_api,
                reason,
            )

        def _on_scenegraph_initialized(self) -> None:
            try:
                actual = _enum_name(self.rendererInterface().graphicsApi())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                actual = ""
            self._scenegraph_initialized = True
            self._scenegraph_error = ""
            self._frame_verified = False
            self._frame_presented = False
            self._set_lifecycle(
                RendererLifecycleState.INITIALIZING,
                actual_api=actual,
                reason="scenegraph initialized; frame verification is pending",
            )

        def _on_scenegraph_invalidated(self) -> None:
            self._scenegraph_initialized = False
            self._frame_verified = False
            self._frame_presented = False
            if not self._shutdown:
                self._set_lifecycle(
                    RendererLifecycleState.INITIALIZING,
                    reason="scenegraph is rebuilding",
                )

        def _on_scenegraph_error(self, _error: object, message: str) -> None:
            self._scenegraph_error = str(message or "Qt Quick scenegraph initialization failed")
            self._scenegraph_initialized = False
            self._frame_verified = False
            self._frame_presented = False
            self._set_lifecycle(
                RendererLifecycleState.FAILED,
                reason=self._scenegraph_error,
            )

        def _on_frame_swapped(self) -> None:
            self._frame_presented = True

        def _image_ready(self) -> bool:
            try:
                ready = bool(self._root.property("frameReady"))
                painted_width = float(self._image_item.property("paintedWidth") or 0.0)
                painted_height = float(self._image_item.property("paintedHeight") or 0.0)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return False
            return ready and painted_width > 0.0 and painted_height > 0.0

        def _verify_scenegraph_frame(self) -> bool:
            if not self._scenegraph_initialized or self._scenegraph_error:
                return False
            actual = self._lifecycle.actual_api
            if actual != _VULKAN_API_NAME:
                self._set_lifecycle(
                    RendererLifecycleState.FAILED,
                    actual_api=actual,
                    reason=f"scenegraph selected '{actual or 'unknown'}' instead of Vulkan",
                )
                return False
            if not self._frame_presented or not self._image_ready():
                return False
            self._frame_verified = True
            self._set_lifecycle(
                RendererLifecycleState.READY,
                actual_api=actual,
                reason="scenegraph API readback and visible frame verification succeeded",
            )
            return True

        def frame_diagnostics(self) -> Mapping[str, object]:
            """按需抓取当前帧，供控制台和验证流程审计实际像素。"""

            if not self._lifecycle.available:
                return {
                    "available": False,
                    "backend": RendererBackend.VULKAN.value,
                    "graphics_api": self.actual_graphics_api,
                    "reason": self._lifecycle.reason,
                }
            try:
                snapshot = self.grabWindow()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                return {
                    "available": False,
                    "backend": RendererBackend.VULKAN.value,
                    "graphics_api": self.actual_graphics_api,
                    "reason": f"frame capture failed: {type(exc).__name__}",
                }
            visible = _visible_snapshot(snapshot)
            return {
                "available": visible,
                "backend": RendererBackend.VULKAN.value,
                "graphics_api": self.actual_graphics_api,
                "width": int(snapshot.width()),
                "height": int(snapshot.height()),
                "alpha": bool(snapshot.hasAlphaChannel()),
                "visible_pixels": visible,
            }

        def _await_ready(self, timeout_ms: int) -> bool:
            timeout = max(100, min(15_000, int(timeout_ms))) / 1000.0
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not self._shutdown:
                QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 25)
                if self._scenegraph_error:
                    break
                if self._lifecycle.state is RendererLifecycleState.FAILED:
                    break
                if self._lifecycle.available:
                    return True
                if self._scenegraph_initialized and self._verify_scenegraph_frame():
                    return True
                time.sleep(0.005)
            if not self._lifecycle.available:
                reason = (
                    self._scenegraph_error
                    or self._lifecycle.reason
                    or "Vulkan initialization timed out"
                )
                if self._scenegraph_initialized and self._lifecycle.actual_api == _VULKAN_API_NAME:
                    reason = "Vulkan Live2D provider did not produce a visible frame before timeout"
                self._set_lifecycle(
                    RendererLifecycleState.FAILED,
                    actual_api=self._lifecycle.actual_api,
                    reason=reason,
                )
            return self._lifecycle.available

        def initialize(
            self,
            surface: object | None = None,
            *,
            timeout_ms: int = _INITIALIZATION_TIMEOUT_MS,
        ) -> bool:
            """显示窗口并等待真实 Vulkan 场景图和可见帧验证。"""

            if surface not in {None, self}:
                self._set_lifecycle(
                    RendererLifecycleState.FAILED,
                    reason="VulkanPetHost owns its QQuickWindow surface",
                )
                return False
            if self._shutdown:
                return False
            if self._lifecycle.available:
                return True
            self._set_lifecycle(
                RendererLifecycleState.INITIALIZING,
                reason="waiting for Qt Quick Vulkan scenegraph",
            )
            self.show()
            if not self._await_ready(timeout_ms):
                self._timer.stop()
                try:
                    self.hide()
                    self.releaseResources()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                return False
            self._last_frame_at = time.monotonic()
            self._timer.start()
            return True

        def _preferred_size(self) -> tuple[int, int]:
            path = self._sprite_renderer.current_frame
            if path is None:
                return 420, 520
            image = QImage(str(path))
            if image.isNull():
                return 420, 520
            width = max(1, image.width())
            height = max(1, image.height())
            scale = max(self._sprite_scale, 160.0 / width, 240.0 / height)
            return max(1, round(width * scale)), max(1, round(height * scale)) + _BUBBLE_HEIGHT

        def _sync_frame(self, *, force: bool = False) -> None:
            path = self._sprite_renderer.current_frame
            if not force and path == self._last_frame_path:
                transform = self._sprite_renderer.visual_transform
                self._root.setProperty("frameOffsetX", float(transform[0]))
                self._root.setProperty("frameOffsetY", float(transform[1]))
                return
            self._last_frame_path = path
            self._last_frame_image = QImage(str(path)) if path is not None else None
            if self._last_frame_image is not None and self._last_frame_image.isNull():
                self._last_frame_image = None
            self._root.setProperty("frameSource", QUrl.fromLocalFile(str(path)) if path else QUrl())
            transform = self._sprite_renderer.visual_transform
            self._root.setProperty("frameOffsetX", float(transform[0]))
            self._root.setProperty("frameOffsetY", float(transform[1]))
            if self._surface_mask_enabled:
                self._refresh_surface_mask()

        def _tick(self) -> None:
            if self._shutdown:
                return
            now = time.monotonic()
            elapsed = max(0.0, min(0.25, now - self._last_frame_at))
            self._last_frame_at = now
            self._sprite_renderer.advance(elapsed)
            if self._feedback_visible and now >= self._feedback_until:
                self._feedback_visible = False
                self._feedback_text = ""
                self._root.setProperty("feedbackVisible", False)
                self._root.setProperty("feedbackText", "")
            self._sync_frame()
            self.update()

        def advance(self, elapsed_seconds: float) -> None:
            self._sprite_renderer.advance(elapsed_seconds)
            self._sync_frame()
            self.update()

        def draw(self) -> None:
            self.update()

        def resize(
            self,
            width: object,
            height: object | None = None,
            device_pixel_ratio: float = 1.0,
        ) -> None:
            """兼容 QWindow 与 Renderer3D 的尺寸入口。"""

            if height is None:
                super().resize(width)
                try:
                    size = self.size()
                    self._last_size = (
                        max(1, int(size.width())),
                        max(1, int(size.height())),
                        max(0.1, min(8.0, float(self.devicePixelRatio()))),
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    self._last_size = None
                return
            safe_width = _finite_dimension(width)
            safe_height = _finite_dimension(height)
            try:
                safe_dpr = max(0.1, min(8.0, float(device_pixel_ratio)))
            except (TypeError, ValueError, OverflowError):
                safe_dpr = 1.0
            if not math.isfinite(safe_dpr):
                safe_dpr = 1.0
            self._last_size = (safe_width, safe_height, safe_dpr)
            super().resize(safe_width, safe_height)

        def supports_expression(self, name: str) -> bool:
            return self._sprite_renderer.supports_expression(name)

        def supports_motion(self, name: str) -> bool:
            return self._sprite_renderer.supports_motion(name)

        def set_expression(self, name: str) -> bool:
            changed = self._sprite_renderer.set_expression(name)
            if changed:
                self._sync_frame(force=True)
                self.expressionChanged.emit(str(name))
                self.update()
            return changed

        def set_expression_request(self, request: ExpressionRequest) -> Mapping[str, object]:
            result = self._sprite_renderer.set_expression_request(request)
            if str(result.get("status", "")) == "started":
                self._sync_frame(force=True)
                self.expressionChanged.emit(self._sprite_renderer.state.expression)
                self.update()
            return result

        def play_motion(self, name: str) -> bool:
            changed = self._sprite_renderer.play_motion(name)
            if changed:
                self.motionChanged.emit(str(name))
                self.update()
            return changed

        def play_motion_request(self, request: MotionRequest) -> Mapping[str, object]:
            result = self._sprite_renderer.play_motion_request(request)
            if str(result.get("status", "")) == "started":
                self.motionChanged.emit(self._sprite_renderer.state.motion)
                self.update()
            return result

        def set_direction(self, direction: str) -> bool:
            changed = self._sprite_renderer.set_direction(direction)
            if changed:
                self._sync_frame(force=True)
                self.update()
            return changed

        def set_cursor_target(self, x: int, _y: int, width: int, _height: int) -> bool:
            return self.set_direction("A" if int(x) < max(1, int(width)) / 2 else "B")

        def set_mouth_viseme(self, _name: str, *, intensity: float = 1.0) -> bool:
            del intensity
            return False

        def set_dragging(self, dragging: bool) -> bool:
            self._dragging = bool(dragging)
            return self._sprite_renderer.set_dragging(dragging)

        def set_interaction_locked(self, locked: bool) -> bool:
            result = self.set_window_locked(locked)
            return str(result.get("status", "")) == "available"

        def reload_resources(
            self,
            resource_root: str | Path,
            *,
            model_path: str | Path | None = None,
            sprite_scale: float | None = None,
        ) -> Mapping[str, object]:
            if self._shutdown:
                return {"status": "unavailable", "reason": "renderer host is closed"}
            if self._dragging or self._press_global is not None:
                return {"status": "busy", "reason": "resource reload is blocked while dragging"}
            result = dict(
                self._sprite_renderer.reload_resources(
                    resource_root,
                    model_path=model_path,
                    sprite_scale=sprite_scale,
                )
            )
            if str(result.get("status", "")).lower() not in {"reloaded", "available", "unchanged"}:
                return result
            if sprite_scale is not None:
                self._sprite_scale = _finite_scale(sprite_scale, default=self._sprite_scale)
            self._last_frame_path = None
            self._last_frame_image = None
            self._root.setProperty("frameSource", QUrl())
            QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
            self._sync_frame(force=True)
            self.resize(*self._preferred_size())
            self.update()
            result["backend"] = RendererBackend.VULKAN.value
            result["graphics_api"] = self.actual_graphics_api
            return result

        def set_speech(
            self,
            text: str,
            *,
            mood: str = "neutral",
            visible: bool = True,
            speaking: bool = True,
        ) -> None:
            del mood, speaking
            self._speech_text = str(text or "")
            self._speech_visible = bool(visible and self._speech_text)
            self._root.setProperty("speechText", self._speech_text)
            self._root.setProperty("speechVisible", self._speech_visible)
            if self._surface_mask_enabled:
                self._refresh_surface_mask()
            self.update()

        def set_interaction_feedback(
            self,
            value: Mapping[str, object] | None,
            *,
            visible: bool = True,
        ) -> bool:
            if not visible or not isinstance(value, Mapping):
                self._feedback_text = ""
                self._feedback_visible = False
            else:
                labels = {
                    "head": "猫猫头",
                    "upper": "上半身",
                    "body": "身体",
                    "lower_left": "左腿",
                    "lower_right": "右腿",
                }
                zone = str(value.get("zone", "") or "").strip().lower()
                parts = [
                    labels.get(zone, ""),
                    str(value.get("phrase", "") or "").strip(),
                    " · ".join(
                        str(value.get(key, "") or "").strip()
                        for key in ("expression", "motion")
                        if str(value.get(key, "") or "").strip()
                    ),
                ]
                self._feedback_text = "\n".join(item for item in parts if item)
                self._feedback_visible = bool(self._feedback_text)
                self._feedback_until = time.monotonic() + 1.8
            self._root.setProperty("feedbackText", self._feedback_text)
            self._root.setProperty("feedbackVisible", self._feedback_visible)
            if self._surface_mask_enabled:
                self._refresh_surface_mask()
            self.update()
            return True

        def _model_rect(self) -> QRect | None:
            image = self._last_frame_image
            if image is None or image.isNull():
                return None
            area_width = max(1, self.width())
            area_height = max(1, self.height() - _BUBBLE_HEIGHT)
            scale = min(area_width / image.width(), area_height / image.height())
            width = max(1, round(image.width() * scale))
            height = max(1, round(image.height() * scale))
            transform = self._sprite_renderer.visual_transform
            x = round((area_width - width) / 2 + float(transform[0]))
            y = round(_BUBBLE_HEIGHT + area_height - height + float(transform[1]))
            return QRect(x, y, width, height)

        def _model_hit(self, point: QPoint | None) -> bool:
            rect = self._model_rect()
            image = self._last_frame_image
            if point is None or rect is None or image is None or not rect.contains(point):
                return False
            source_x = min(
                image.width() - 1,
                max(0, int((point.x() - rect.x()) * image.width() / rect.width())),
            )
            source_y = min(
                image.height() - 1,
                max(0, int((point.y() - rect.y()) * image.height() / rect.height())),
            )
            try:
                return image.pixelColor(source_x, source_y).alpha() > 8
            except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
                return False

        def _refresh_surface_mask(self) -> bool:
            if not self._surface_mask_enabled:
                self.clearMask()
                return True
            rect = self._model_rect()
            image = self._last_frame_image
            if rect is None or image is None:
                return False
            try:
                scaled = image.scaled(
                    rect.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                alpha = scaled.createAlphaMask()
                region = QRegion(QBitmap.fromImage(alpha))
                region.translate(rect.x(), rect.y())
                if self._speech_visible:
                    region |= QRegion(QRect(8, 6, max(1, self.width() - 16), 48))
                if self._feedback_visible:
                    width = min(112, max(80, round(self.width() * 0.28)))
                    height = min(120, max(60, round(self.height() * 0.24)))
                    region |= QRegion(
                        QRect(
                            self.width() - width - 6,
                            max(62, round((self.height() - height) / 2)),
                            width,
                            height,
                        )
                    )
                self.setMask(region)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
            return not self.mask().isEmpty()

        def set_surface_mask_enabled(self, enabled: bool) -> dict[str, object]:
            self._surface_mask_enabled = bool(enabled)
            ready = self._refresh_surface_mask()
            return {
                "status": "available" if ready else "unavailable",
                "enabled": bool(self._surface_mask_enabled and ready),
                "ready": ready,
                "input_ready": ready,
            }

        def surface_mask_status(self) -> dict[str, object]:
            ready = not self.mask().isEmpty() if self._surface_mask_enabled else True
            return {
                "status": "available" if ready else "unavailable",
                "enabled": bool(self._surface_mask_enabled),
                "ready": ready,
                "input_ready": ready,
            }

        def set_click_callback(
            self,
            callback: Callable[[Mapping[str, object]], None] | None,
        ) -> None:
            self._click_callback = callback

        def set_interaction_interrupt(self, callback: Callable[[], object] | None) -> None:
            self._interaction_interrupt = callback

        def set_activity_notifier(self, callback: Callable[[], object] | None) -> None:
            self._activity_notifier = callback

        @staticmethod
        def _close_unconsumed_awaitable(value: object) -> None:
            if not inspect.isawaitable(value):
                return
            closer = getattr(value, "close", None)
            if callable(closer):
                closer()

        def _notify_interaction(self) -> None:
            for callback in (self._interaction_interrupt, self._activity_notifier):
                if not callable(callback):
                    continue
                try:
                    self._close_unconsumed_awaitable(callback())
                except Exception:
                    logger.debug("Vulkan interaction callback failed", exc_info=True)

        @staticmethod
        def _event_global_point(event: object) -> QPoint | None:
            getter = getattr(event, "globalPosition", None)
            if callable(getter):
                try:
                    value = getter()
                    return value.toPoint()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    return None
            return None

        @staticmethod
        def _event_local_point(event: object) -> QPoint | None:
            getter = getattr(event, "position", None)
            if callable(getter):
                try:
                    value = getter()
                    return value.toPoint()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    return None
            return None

        def mousePressEvent(self, event: object) -> None:  # noqa: N802
            if self._window_locked:
                event.ignore()
                return
            point = self._event_local_point(event)
            if event.button() == Qt.MouseButton.RightButton and self._model_hit(point):
                global_point = self._event_global_point(event)
                if global_point is not None:
                    self.contextMenuRequested.emit(global_point)
                event.accept()
                return
            if event.button() != Qt.MouseButton.LeftButton or not self._model_hit(point):
                event.ignore()
                return
            self._notify_interaction()
            self._press_global = self._event_global_point(event)
            self._window_origin = self.position()
            self._dragging = False
            self._pointer_over = True
            try:
                self.setMouseGrabEnabled(True)
            except (AttributeError, RuntimeError):
                pass
            event.accept()

        def mouseMoveEvent(self, event: object) -> None:  # noqa: N802
            point = self._event_local_point(event)
            self._pointer_over = self._model_hit(point)
            if self._window_locked or self._press_global is None or self._window_origin is None:
                event.accept() if self._pointer_over else event.ignore()
                return
            global_point = self._event_global_point(event)
            if global_point is None:
                event.ignore()
                return
            delta = global_point - self._press_global
            if not self._dragging and delta.manhattanLength() < 4:
                event.accept()
                return
            if not self._dragging:
                self._dragging = True
                self._sprite_renderer.set_dragging(True)
            self.setPosition(self._window_origin + delta)
            event.accept()

        def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802
            if event.button() != Qt.MouseButton.LeftButton:
                event.ignore()
                return
            was_dragging = self._dragging
            point = self._event_local_point(event)
            self._press_global = None
            self._window_origin = None
            self._dragging = False
            self._sprite_renderer.set_dragging(False)
            try:
                self.setMouseGrabEnabled(False)
            except (AttributeError, RuntimeError):
                pass
            if not was_dragging and self._model_hit(point) and point is not None:
                payload = make_pet_click_payload(
                    {
                        "x": point.x(),
                        "y": point.y(),
                        "part": classify_pet_part(
                            point.x(), point.y(), self.width(), self.height()
                        ),
                    },
                    width=self.width(),
                    height=self.height(),
                )
                if payload is not None and callable(self._click_callback):
                    try:
                        self._click_callback(payload)
                    except Exception:
                        logger.debug("Vulkan pet click callback failed", exc_info=True)
            event.accept()

        def mouseDoubleClickEvent(self, event: object) -> None:  # noqa: N802
            if (
                not self._window_locked
                and event.button() == Qt.MouseButton.LeftButton
                and self._model_hit(self._event_local_point(event))
            ):
                self.consoleRequested.emit()
                event.accept()
                return
            event.ignore()

        def update_cursor_tracking(self) -> bool:
            try:
                local = QCursor.pos() - self.position()
                width = max(1, self.width())
                height = max(1, self.height())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
            target = (local.x(), local.y(), width, height)
            self._pointer_over = self._model_hit(local)
            if target == self._last_cursor_target:
                return True
            self._last_cursor_target = target
            return self.set_cursor_target(*target)

        def is_user_interacting(self) -> bool:
            return False if self._window_locked else bool(self._dragging or self._pointer_over)

        def move_to(self, x: float, y: float, *, duration_ms: int = 800) -> object:
            del duration_ms
            if self._dragging or self._press_global is not None:
                return {"status": "cancelled", "reason": "user interaction active"}
            target_x = int(round(float(x)))
            target_y = int(round(float(y)))
            handler = getattr(self.platform, "move_overlay", None) if self.platform else None
            if callable(handler):
                return handler(self, target_x, target_y)
            self.setPosition(QPoint(target_x, target_y))
            return {"status": "available", "x": target_x, "y": target_y}

        def movement_bounds(self) -> tuple[int, int, int, int] | None:
            screen = self.screen() or QGuiApplication.primaryScreen()
            if screen is None:
                return None
            area = screen.availableGeometry()
            return (
                area.left(),
                area.top(),
                max(area.left(), area.right() - self.width() + 1),
                max(area.top(), area.bottom() - self.height() + 1),
            )

        def set_display_size(self, preset: str) -> dict[str, object]:
            selected = display_size_preset(preset)
            if selected is None:
                return {"status": "unavailable", "reason": "display size preset is invalid"}
            key, (width, height) = selected
            try:
                position = self.position()
                screen = self.screen() or QGuiApplication.primaryScreen()
                area = screen.availableGeometry() if screen is not None else None
                width, height = _fit_display_size_to_area(width, height, area)

                # QQuickView 可能保留之前预设留下的最小尺寸；先降低约束，
                # 否则小屏幕下 resize 会被静默拒绝，QML 视口与控制台回执不一致。
                min_width_getter = getattr(self, "minimumWidth", None)
                min_height_getter = getattr(self, "minimumHeight", None)
                min_width_setter = getattr(self, "setMinimumWidth", None)
                min_height_setter = getattr(self, "setMinimumHeight", None)
                if callable(min_width_getter) and callable(min_width_setter):
                    if width < int(min_width_getter()):
                        min_width_setter(width)
                if callable(min_height_getter) and callable(min_height_setter):
                    if height < int(min_height_getter()):
                        min_height_setter(height)

                self.resize(width, height)
                actual_width = max(1, int(self.width()))
                actual_height = max(1, int(self.height()))
                bounds = self.movement_bounds()
                if position is not None and bounds is not None:
                    position = QPoint(
                        max(bounds[0], min(position.x(), bounds[2])),
                        max(bounds[1], min(position.y(), bounds[3])),
                    )
                    self.setPosition(position)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug("Vulkan display size update failed: %s", type(exc).__name__)
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

        def _apply_window_flag(self, flag: Qt.WindowType, enabled: bool) -> bool:
            was_visible = self.isVisible()
            position = self.position()
            self.setFlag(flag, bool(enabled))
            self.setPosition(position)
            if was_visible:
                self.show()
            actual = bool(self.flags() & flag)
            if self._lifecycle.state is RendererLifecycleState.INITIALIZING:
                self._await_ready(3_000)
            return actual == bool(enabled) and self._lifecycle.available

        def set_click_through(self, enabled: bool) -> dict[str, object]:
            requested = bool(enabled)
            if self._window_locked and not requested:
                return {
                    "status": "unavailable",
                    "enabled": True,
                    "locked": True,
                    "reason": "window is locked; unlock it before restoring input",
                }
            applied = self._apply_window_flag(Qt.WindowType.WindowTransparentForInput, requested)
            if applied:
                self._click_through = requested
            return {
                "status": "available" if applied else "unavailable",
                "enabled": bool(self.flags() & Qt.WindowType.WindowTransparentForInput),
                "locked": self._window_locked,
            }

        def begin_click_through_override(self) -> Mapping[str, object]:
            if self._click_through_override_active:
                return {"status": "available", "enabled": False, "temporary": True}
            self._click_through_override_active = True
            self._click_through_before_override = self._click_through
            return (
                self.set_click_through(False)
                if self._click_through
                else {
                    "status": "available",
                    "enabled": False,
                    "temporary": True,
                }
            )

        def end_click_through_override(self) -> Mapping[str, object]:
            if not self._click_through_override_active:
                return {"status": "available", "enabled": self._click_through}
            restore = self._click_through_before_override
            self._click_through_override_active = False
            self._click_through_before_override = False
            return self.set_click_through(restore)

        def set_window_locked(self, locked: bool) -> dict[str, object]:
            requested = bool(locked)
            if requested == self._window_locked:
                return {
                    "status": "available",
                    "locked": requested,
                    "enabled": self._click_through,
                }
            if requested:
                self._click_through_before_lock = self._click_through
                result = self.set_click_through(True)
                if str(result.get("status", "")) != "available":
                    return dict(result)
                self._window_locked = True
                return {"status": "available", "locked": True, "enabled": True}
            self._window_locked = False
            result = self.set_click_through(self._click_through_before_lock)
            if str(result.get("status", "")) != "available":
                self._window_locked = True
                return dict(result)
            self._click_through_before_lock = False
            return {"status": "available", "locked": False, "enabled": self._click_through}

        def toggle_window_locked(self) -> dict[str, object]:
            return self.set_window_locked(not self._window_locked)

        def is_window_locked(self) -> bool:
            return self._window_locked

        def set_always_on_top(self, enabled: bool) -> dict[str, object]:
            applied = self._apply_window_flag(Qt.WindowType.WindowStaysOnTopHint, bool(enabled))
            if applied:
                self._always_on_top = bool(enabled)
            return {
                "status": "available" if applied else "unavailable",
                "enabled": bool(self.flags() & Qt.WindowType.WindowStaysOnTopHint),
            }

        def toggle_always_on_top(self) -> dict[str, object]:
            return self.set_always_on_top(not self.is_always_on_top())

        def is_always_on_top(self) -> bool:
            return bool(self.flags() & Qt.WindowType.WindowStaysOnTopHint)

        def always_on_top_status(self) -> dict[str, object]:
            return {"status": "available", "enabled": self.is_always_on_top()}

        def toggle_visibility(self) -> dict[str, object]:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self._await_ready(3_000)
            return {"status": "available", "visible": self.isVisible()}

        def set_legacy_shortcuts_enabled(self, _enabled: bool) -> None:
            return None

        def shutdown(self) -> None:
            """停止场景图和精灵生命周期；重复调用安全。"""

            if self._shutdown:
                return
            self._shutdown = True
            self._timer.stop()
            self._click_callback = None
            self._interaction_interrupt = None
            self._activity_notifier = None
            self._sprite_renderer.set_dragging(False)
            self._sprite_renderer.shutdown()
            try:
                self.hide()
                self.releaseResources()
                self.close()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("Vulkan host resource release failed", exc_info=True)
            self._set_lifecycle(
                RendererLifecycleState.CLOSED,
                actual_api=self._lifecycle.actual_api,
                reason="renderer host is closed",
            )

        def prepare_shutdown(self) -> None:
            self.shutdown()

        def closeEvent(self, event: object) -> None:  # noqa: N802
            # 用户关闭只隐藏到托盘；应用退出时由 shutdown() 统一释放资源。
            if not self._shutdown:
                self.closeRequested.emit()
                event.ignore()
                self.hide()
                return
            self.shutdown()
            event.accept()

    def create_vulkan_host(selection: object, **kwargs: object) -> VulkanPetHost:
        """从统一选择结果创建 Live2D Vulkan 宿主。"""

        del kwargs
        backend = normalize_renderer_backend(getattr(selection, "backend", ""))
        if backend != RendererBackend.VULKAN.value:
            raise ValueError("Vulkan host factory requires a Vulkan renderer selection")
        raise RuntimeError(
            "Vulkan Live2D renderer provider is unavailable; "
            "Qt Quick Vulkan alone cannot render Cubism models"
        )


else:

    def prepare_vulkan_scenegraph() -> tuple[bool, str]:  # pragma: no cover
        return False, "PySide6 Qt Quick is unavailable"

    class VulkanPetHost:  # pragma: no cover
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 Qt Quick is unavailable")

    def create_vulkan_host(  # pragma: no cover
        _selection: object,
        **_kwargs: object,
    ) -> VulkanPetHost:
        raise RuntimeError("PySide6 Qt Quick is unavailable")


__all__ = [
    "VulkanPetHost",
    "create_vulkan_host",
    "prepare_vulkan_scenegraph",
    "pyside6_vulkan_available",
]
