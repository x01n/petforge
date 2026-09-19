"""渲染器能力边界与 Live2D/精灵实现。"""

from .assets import (
    RenderAssetInventory,
    RendererRegistration,
    RendererRegistry,
    RuntimeCapability,
    RuntimeCapabilityState,
    default_renderer_registry,
    probe_live2d_runtime,
    probe_render_assets,
    probe_web_live2d_runtime,
    select_renderer,
)
from .live2d import Live2DRenderer
from .model_catalog import (
    Live2DModelCatalog,
    Live2DModelEntry,
    Live2DModelSelection,
    ModelCatalog,
)
from .protocol import (
    DEFAULT_EXPRESSION_NAMES,
    DEFAULT_MOTION_NAMES,
    RENDERER_BACKEND_CHOICES,
    Renderer3D,
    RendererBackend,
    RendererBackendState,
    RendererCapabilities,
    RendererInitialization,
    RendererInitializationResult,
    RendererInitializationState,
    RendererLifecycleState,
    RendererLifecycleStatus,
    RendererState,
    normalize_renderer_backend,
)
from .sprite import SpriteRenderer
from .threed import probe_live2d_vulkan_runtime, probe_qt_vulkan_runtime
from .web_live2d import WebLive2DProbe, WebLive2DRenderer, probe_web_live2d

__all__ = [
    "Live2DRenderer",
    "Live2DModelCatalog",
    "Live2DModelEntry",
    "Live2DModelSelection",
    "ModelCatalog",
    "DEFAULT_EXPRESSION_NAMES",
    "DEFAULT_MOTION_NAMES",
    "RENDERER_BACKEND_CHOICES",
    "RendererBackend",
    "RendererBackendState",
    "RendererInitialization",
    "RendererInitializationResult",
    "RendererInitializationState",
    "RendererLifecycleState",
    "RendererLifecycleStatus",
    "RenderAssetInventory",
    "RendererRegistration",
    "RendererRegistry",
    "RendererCapabilities",
    "Renderer3D",
    "RendererState",
    "RuntimeCapability",
    "RuntimeCapabilityState",
    "SpriteRenderer",
    "WebLive2DProbe",
    "WebLive2DRenderer",
    "probe_live2d_runtime",
    "probe_live2d_vulkan_runtime",
    "probe_qt_vulkan_runtime",
    "probe_render_assets",
    "probe_web_live2d_runtime",
    "probe_web_live2d",
    "select_renderer",
    "default_renderer_registry",
    "normalize_renderer_backend",
]
