"""不依赖 Qt 的渲染器状态契约。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from core.rendering.actions import ExpressionRequest, MotionRequest

# 这些是宿主控制台可以安全展示的通用名称。后端若能从模型描述读取更
# 精确的能力，应在 ``RendererCapabilities`` 中覆盖；没有描述时也不能
# 让用户面对一个空的下拉框。
DEFAULT_EXPRESSION_NAMES: tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "curious",
    "surprised",
    "shy",
)
DEFAULT_MOTION_NAMES: tuple[str, ...] = ("idle", "blink", "wave", "walk")


class RendererBackend(StrEnum):
    """用户可选择的渲染后端。"""

    AUTO = "auto"
    OPENGL = "opengl"
    WEB_LIVE2D = "web_live2d"
    SPRITE = "sprite"
    VULKAN = "vulkan"
    UNAVAILABLE = "unavailable"


# 用户曾使用过的两个明确拼写只作为输入兼容别名，内部能力、工厂和实例
# 一律使用规范键 ``vulkan``，避免一个后端出现三套身份。
VULKAN_COMPATIBILITY_ALIASES: tuple[str, ...] = ("vllank", "vllakn")
CANONICAL_RENDERER_BACKEND_CHOICES: tuple[str, ...] = tuple(
    item.value
    for item in (
        RendererBackend.AUTO,
        RendererBackend.OPENGL,
        RendererBackend.WEB_LIVE2D,
        RendererBackend.SPRITE,
        RendererBackend.VULKAN,
    )
)
RENDERER_BACKEND_INPUTS: tuple[str, ...] = (
    CANONICAL_RENDERER_BACKEND_CHOICES + VULKAN_COMPATIBILITY_ALIASES
)
# UI 和外部枚举只暴露规范键；旧拼写仅在配置输入边界兼容。
RENDERER_BACKEND_CHOICES: tuple[str, ...] = CANONICAL_RENDERER_BACKEND_CHOICES


def normalize_renderer_backend(value: object) -> str:
    """规范化渲染后端键；旧拼写统一映射为 ``vulkan``。"""

    normalized = str(value or RendererBackend.AUTO.value).strip().lower()
    if normalized in VULKAN_COMPATIBILITY_ALIASES:
        return RendererBackend.VULKAN.value
    if normalized not in RENDERER_BACKEND_INPUTS:
        choices = ", ".join(CANONICAL_RENDERER_BACKEND_CHOICES)
        raise ValueError(f"rendering.backend must be one of: {choices}")
    return normalized


def renderer_backend_compatibility_alias(value: object) -> str:
    """返回输入使用的 Vulkan 兼容拼写；规范键和其他后端返回空串。"""

    normalized = str(value or "").strip().lower()
    return normalized if normalized in VULKAN_COMPATIBILITY_ALIASES else ""


@dataclass(frozen=True)
class RendererBackendState:
    """后端选择状态，区分用户请求和实际可用后端。"""

    requested: str
    selected: str
    available: bool
    reason: str = ""
    compatibility_alias: str = ""


class RendererInitializationState(StrEnum):
    """渲染器实例化阶段的结果状态。

    ``UNAVAILABLE`` 表示选择阶段已经确认后端没有运行能力；``FAILED``
    表示能力探测通过但工厂或实例初始化失败。两者都不会触发隐式回退。
    """

    READY = "ready"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class RendererLifecycleState(StrEnum):
    """图形宿主的真实生命周期状态。"""

    CREATED = "created"
    INITIALIZING = "initializing"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class RendererLifecycleStatus:
    """记录请求图形 API 与场景图实际 API 的可审计结果。"""

    backend: str
    state: RendererLifecycleState
    requested_api: str
    actual_api: str = ""
    reason: str = ""

    @property
    def available(self) -> bool:
        """仅场景图已就绪且 API 回读一致时返回可用。"""

        return self.state is RendererLifecycleState.READY


@dataclass(frozen=True)
class RendererInitializationResult:
    """渲染器工厂的可审计初始化结果。"""

    backend: str
    state: RendererInitializationState
    renderer: object | None = None
    reason: str = ""

    @property
    def available(self) -> bool:
        """返回实例是否已经完成初始化并可绘制。"""

        return self.state is RendererInitializationState.READY

    @property
    def succeeded(self) -> bool:
        """返回初始化是否成功。"""

        return self.available

    @property
    def ok(self) -> bool:
        """返回 ``succeeded`` 的简短别名，便于宿主处理结果。"""

        return self.succeeded

    @property
    def error(self) -> str:
        """返回可展示的失败原因；成功时为空字符串。"""

        return "" if self.succeeded else self.reason


# 便于调用方按“结果”或“初始化”两种命名习惯读取同一契约。
RendererInitialization = RendererInitializationResult


@dataclass(frozen=True)
class RendererCapabilities:
    backend: str
    available: bool
    model_path: Path | None = None
    expressions: tuple[str, ...] = ()
    motions: tuple[str, ...] = ()
    message: str = ""


@dataclass
class RendererState:
    expression: str = "neutral"
    motion: str = "idle"
    frame_index: int = 0
    elapsed: float = 0.0


class Renderer(Protocol):
    @property
    def capabilities(self) -> RendererCapabilities: ...

    @property
    def state(self) -> RendererState: ...

    def advance(self, elapsed_seconds: float) -> None: ...

    def set_expression(self, name: str) -> bool: ...

    def set_expression_request(self, request: ExpressionRequest) -> Mapping[str, object]: ...

    def play_motion(self, name: str) -> bool: ...

    def play_motion_request(self, request: MotionRequest) -> Mapping[str, object]: ...

    def supports_expression(self, name: str) -> bool | None: ...

    def supports_motion(self, name: str) -> bool | None: ...

    def set_mouth_viseme(self, name: str, *, intensity: float = 1.0) -> bool: ...

    def set_direction(self, direction: str) -> bool: ...

    def set_cursor_target(self, x: int, y: int, width: int, height: int) -> bool: ...

    def set_dragging(self, dragging: bool) -> bool: ...

    def reload_resources(
        self,
        resource_root: str | Path,
        *,
        model_path: str | Path | None = None,
        sprite_scale: float | None = None,
    ) -> Mapping[str, object]: ...

    def shutdown(self) -> None: ...


class Renderer3D(Protocol):
    """可切换三维后端的可审计生命周期契约。"""

    @property
    def capabilities(self) -> RendererCapabilities: ...

    @property
    def lifecycle_status(self) -> RendererLifecycleStatus: ...

    def initialize(self, surface: object | None = None) -> bool: ...

    def resize(self, width: int, height: int, device_pixel_ratio: float = 1.0) -> None: ...

    def draw(self) -> None: ...

    def reload_resources(
        self,
        resource_root: str | Path,
        *,
        model_path: str | Path | None = None,
        sprite_scale: float | None = None,
    ) -> Mapping[str, object]: ...

    def shutdown(self) -> None: ...
