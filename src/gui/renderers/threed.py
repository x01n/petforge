"""Vulkan 场景图运行时探测与旧版失败关闭兼容对象。"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import (
    RendererBackend,
    RendererCapabilities,
    RendererLifecycleState,
    RendererLifecycleStatus,
    normalize_renderer_backend,
    renderer_backend_compatibility_alias,
)

_RESERVED_BACKEND = RendererBackend.VULKAN.value


@dataclass(frozen=True, slots=True)
class VulkanRuntimeProbe:
    """Qt Vulkan Python 绑定的只读探测结果。"""

    available: bool
    detail: str
    evidence: tuple[str, ...] = ()


def probe_qt_vulkan_runtime(
    importer: Callable[[str], Any] | None = None,
    *,
    library_finder: Callable[[str], str | None] | None = None,
) -> VulkanRuntimeProbe:
    """检查 Qt Quick Vulkan 场景图所需的精确 Python API 和加载器。

    PySide6 并不需要导出 ``QVulkanWindow`` 才能使用 Vulkan。Qt 6 的
    公共入口是 ``QQuickWindow.setGraphicsApi(Vulkan)``，最终可用性仍由
    宿主在 ``sceneGraphInitialized`` 后回读 ``graphicsApi`` 确认。
    """

    loader = importer or importlib.import_module
    try:
        qt_quick = loader("PySide6.QtQuick")
    except (ImportError, ModuleNotFoundError) as exc:
        return VulkanRuntimeProbe(
            False,
            f"PySide6.QtQuick is unavailable: {type(exc).__name__}",
            ("missing:PySide6.QtQuick",),
        )
    required = ("QQuickWindow", "QQuickView", "QSGRendererInterface")
    missing = tuple(name for name in required if getattr(qt_quick, name, None) is None)
    if missing:
        return VulkanRuntimeProbe(
            False,
            "Qt Quick Vulkan scenegraph bindings are unavailable: " + ", ".join(missing),
            tuple(f"missing:PySide6.QtQuick.{name}" for name in missing),
        )
    quick_window = qt_quick.QQuickWindow
    renderer_interface = qt_quick.QSGRendererInterface
    graphics_api = getattr(renderer_interface, "GraphicsApi", None)
    vulkan = getattr(graphics_api, "Vulkan", None)
    required_methods = (
        "setGraphicsApi",
        "graphicsApi",
        "setDefaultAlphaBuffer",
        "hasDefaultAlphaBuffer",
    )
    missing_methods = tuple(
        name for name in required_methods if not callable(getattr(quick_window, name, None))
    )
    if missing_methods or vulkan is None:
        evidence = tuple(f"missing:PySide6.QtQuick.QQuickWindow.{name}" for name in missing_methods)
        if vulkan is None:
            evidence += ("missing:QSGRendererInterface.GraphicsApi.Vulkan",)
        return VulkanRuntimeProbe(
            False,
            "Qt Quick does not expose the required Vulkan graphics API",
            evidence,
        )
    locate = library_finder or find_library
    try:
        vulkan_library = locate("vulkan")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return VulkanRuntimeProbe(
            False,
            f"Vulkan loader lookup failed: {type(exc).__name__}",
            ("missing:libvulkan",),
        )
    if not vulkan_library:
        return VulkanRuntimeProbe(
            False,
            "Vulkan loader is unavailable",
            ("missing:libvulkan",),
        )
    return VulkanRuntimeProbe(
        True,
        "Qt Quick Vulkan API and loader are available; scenegraph initialization is still required",
        tuple(f"present:PySide6.QtQuick.{name}" for name in required)
        + tuple(f"present:QQuickWindow.{name}" for name in required_methods)
        + ("present:QSGRendererInterface.GraphicsApi.Vulkan", f"loader:{vulkan_library}"),
    )


def _safe_dimension(value: object, *, default: int = 1) -> int:
    """把生命周期输入限制为有限、正的逻辑像素尺寸。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(numeric) or numeric <= 0.0:
        return default
    return max(1, min(16_384, int(round(numeric))))


def _safe_device_pixel_ratio(value: object, *, default: float = 1.0) -> float:
    """把设备像素比限制在三维占位生命周期可记录的安全范围。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(numeric) or numeric <= 0.0:
        return default
    return max(0.1, min(8.0, numeric))


class Reserved3DRenderer:
    """旧 API 的失败关闭对象；默认注册表不再使用该类型。"""

    def __init__(self, backend: str = _RESERVED_BACKEND) -> None:
        normalized = normalize_renderer_backend(backend or _RESERVED_BACKEND)
        if normalized != _RESERVED_BACKEND:
            raise ValueError("Reserved3DRenderer backend must be vulkan or a supported alias")
        self._compatibility_alias = renderer_backend_compatibility_alias(backend)
        self._capabilities = RendererCapabilities(
            backend=_RESERVED_BACKEND,
            available=False,
            message="旧 Reserved3DRenderer 已停用，请使用已注册的 VulkanPetHost 工厂",
        )
        self._last_size: tuple[int, int, float] | None = None
        self._initialized = False
        self._lifecycle = RendererLifecycleStatus(
            backend=_RESERVED_BACKEND,
            state=RendererLifecycleState.FAILED,
            requested_api="vulkan",
            reason="legacy unavailable object; use the registered VulkanPetHost factory",
        )

    @property
    def last_size(self) -> tuple[int, int, float] | None:
        """返回最近一次 ``resize`` 的安全化尺寸，便于诊断而不暴露 surface。"""

        return self._last_size

    @property
    def initialized(self) -> bool:
        """占位后端始终不会进入已初始化状态。"""

        return self._initialized

    @property
    def compatibility_alias(self) -> str:
        """返回构造时使用的旧拼写；规范键返回空串。"""

        return self._compatibility_alias

    @property
    def capabilities(self) -> RendererCapabilities:
        return self._capabilities

    @property
    def lifecycle_status(self) -> RendererLifecycleStatus:
        """返回旧对象的明确失败状态，禁止被切换流程当成真实宿主。"""

        return self._lifecycle

    def initialize(self, surface: object | None = None) -> bool:
        del surface
        # 明确拒绝任何 surface，避免调用方误以为预留实现已经接管了
        # Qt/Vulkan 资源；重复调用保持幂等且不会保存外部对象引用。
        self._initialized = False
        return False

    def resize(self, width: int, height: int, device_pixel_ratio: float = 1.0) -> None:
        self._last_size = (
            _safe_dimension(width),
            _safe_dimension(height),
            _safe_device_pixel_ratio(device_pixel_ratio),
        )

    def draw(self) -> None:
        return None

    def reload_resources(
        self,
        resource_root: str | Path,
        *,
        model_path: str | Path | None = None,
        sprite_scale: float | None = None,
    ) -> dict[str, object]:
        del resource_root, model_path, sprite_scale
        return {
            "status": "unavailable",
            "reason": "Vulkan renderer has no initialized resource lifecycle",
        }

    def shutdown(self) -> None:
        self._initialized = False
        self._last_size = None
        self._lifecycle = RendererLifecycleStatus(
            backend=_RESERVED_BACKEND,
            state=RendererLifecycleState.CLOSED,
            requested_api="vulkan",
            reason="legacy unavailable object is closed",
        )


__all__ = ["Reserved3DRenderer", "VulkanRuntimeProbe", "probe_qt_vulkan_runtime"]
