from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from time import time
from typing import Any

from core.rendering.resources import is_valid_webp, validate_model3_description

from .model_catalog import Live2DModelCatalog, Live2DModelSelection
from .protocol import (
    RendererBackend,
    RendererBackendState,
    RendererInitializationResult,
    RendererInitializationState,
    normalize_renderer_backend,
    renderer_backend_compatibility_alias,
)
from .threed import probe_qt_vulkan_runtime


class RuntimeCapabilityState(StrEnum):
    """Live2D 运行时探测状态。"""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class RuntimeCapability:
    """运行库能力，而不是渲染器当前帧状态。"""

    name: str
    state: RuntimeCapabilityState
    detail: str = ""
    evidence: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.state is RuntimeCapabilityState.AVAILABLE


@dataclass(frozen=True)
class RenderAssetInventory:
    """渲染所需文件的实际清单。"""

    root: Path
    live2d_models: tuple[Path, ...] = ()
    sprite_frames: tuple[Path, ...] = ()
    warnings: tuple[str, ...] = ()
    observed_at: float = field(default_factory=time)

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        models = tuple(Path(path) for path in (self.live2d_models or ()))
        sprites = tuple(Path(path) for path in (self.sprite_frames or ()))
        warnings = tuple(str(item) for item in (self.warnings or ()) if str(item))
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "live2d_models", models)
        object.__setattr__(self, "sprite_frames", sprites)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "observed_at", float(self.observed_at))

    @property
    def live2d_available(self) -> bool:
        """返回是否存在通过 model3 文件引用校验的描述。"""

        return bool(self.live2d_models)

    @property
    def sprite_available(self) -> bool:
        """返回是否存在通过 RIFF/WEBP 头校验的精灵帧。"""

        return bool(self.sprite_frames)


def _unique_sorted(paths: Iterable[Path]) -> tuple[Path, ...]:
    """按绝对路径去重并排序。"""

    return tuple(sorted({Path(path).resolve() for path in paths}))


def _path_is_relative_to(path: str | Path, root: str | Path) -> bool:
    """判断路径是否位于根目录内，兼容 Python 3.11 的 ``Path`` API。"""

    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def probe_render_assets(root: str | Path) -> RenderAssetInventory:
    """扫描资源根目录并返回 Live2D/精灵能力清单。"""

    resource_root = Path(root).expanduser().resolve()
    warnings: list[str] = []
    if not resource_root.is_dir():
        return RenderAssetInventory(
            resource_root,
            warnings=("resource root is unavailable",),
        )

    model_root = resource_root / "live2d" / "model"
    model3_paths = _unique_sorted(model_root.rglob("*.model3.json")) if model_root.is_dir() else ()
    legacy_model_paths = (
        _unique_sorted(model_root.rglob("*.model.json")) if model_root.is_dir() else ()
    )
    models = tuple(path for path in model3_paths if validate_model3_description(path) is not None)
    invalid_models = len(model3_paths) - len(models)
    if legacy_model_paths:
        warnings.append("Live2D .model.json descriptors are unsupported without an exact schema")
    if invalid_models:
        warnings.append(f"{invalid_models} Live2D .model3.json descriptor(s) are invalid")
    if not models:
        warnings.append("no usable Live2D model3 descriptor was found; sprite fallback is required")

    sprite_root = resource_root / "sprites"
    sprite_paths = _unique_sorted(sprite_root.glob("*.webp")) if sprite_root.is_dir() else ()
    sprites = tuple(path for path in sprite_paths if is_valid_webp(path))
    if len(sprites) != len(sprite_paths):
        warnings.append("invalid WebP sprite files were ignored")
    if not sprites:
        warnings.append("no WebP sprite fallback was found")

    return RenderAssetInventory(
        root=resource_root,
        live2d_models=models,
        sprite_frames=sprites,
        warnings=tuple(warnings),
    )


def _import_live2d(importer: Callable[[str], Any] | None = None) -> Any:
    """只在调用时导入 live2d-py 的当前 Python 模块路径。"""

    load = importlib.import_module if importer is None else importer
    return load("live2d.v3")


def _probe_live2d_model_loader(module: Any) -> tuple[bool, str, tuple[str, ...]]:
    """确认 ``LAppModel`` 可以实例化并暴露模型描述加载入口。

    模块级函数存在并不代表当前发行包真的提供可用的 v3 模型对象。把这
    个检查放在能力探测阶段，可以避免 ``--validate`` 先把一个随后必然在
    ``initialize`` 阶段失败的原生 OpenGL 后端报告为可用。
    """

    model_class = getattr(module, "LAppModel", None)
    if not callable(model_class):
        return False, "live2d-py is missing the LAppModel class", ("LAppModel",)
    try:
        model = model_class()
    except Exception as exc:
        return (
            False,
            "live2d-py LAppModel cannot be instantiated",
            ("LAppModel", type(exc).__name__),
        )
    if not callable(getattr(model, "LoadModelJson", None)):
        return (
            False,
            "live2d-py LAppModel is missing LoadModelJson",
            ("LAppModel.LoadModelJson",),
        )
    return True, "live2d-py v3 model loader is available", ("LAppModel", "LoadModelJson")


def probe_live2d_runtime(
    *,
    importer: Callable[[str], Any] | None = None,
) -> RuntimeCapability:
    """探测当前 Python 环境是否暴露所需的 live2d-py v3 API。"""

    try:
        module = _import_live2d(importer)
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        SystemError,
    ) as exc:
        return RuntimeCapability(
            name="live2d_runtime",
            state=RuntimeCapabilityState.UNAVAILABLE,
            detail="live2d-py is not importable on this platform",
            evidence=(type(exc).__name__,),
        )

    required = ("init", "glInit", "clearBuffer", "dispose", "LAppModel")
    missing = tuple(name for name in required if not callable(getattr(module, name, None)))
    if missing:
        return RuntimeCapability(
            name="live2d_runtime",
            state=RuntimeCapabilityState.UNAVAILABLE,
            detail="live2d-py is missing required v3 APIs",
            evidence=missing,
        )
    loader_available, loader_detail, loader_evidence = _probe_live2d_model_loader(module)
    if not loader_available:
        return RuntimeCapability(
            name="live2d_runtime",
            state=RuntimeCapabilityState.UNAVAILABLE,
            detail=loader_detail,
            evidence=loader_evidence,
        )
    return RuntimeCapability(
        name="live2d_runtime",
        state=RuntimeCapabilityState.AVAILABLE,
        detail="live2d-py v3 runtime and model loader are available",
        evidence=required + loader_evidence[1:],
    )


@dataclass(frozen=True)
class RendererSelection:
    """根据资源和运行时能力选择的后端。"""

    backend: str
    reason: str
    inventory: RenderAssetInventory
    live2d_runtime: RuntimeCapability
    requested_backend: str = RendererBackend.AUTO.value
    web_live2d_runtime: RuntimeCapability | None = None
    compatibility_alias: str = ""
    model_selection: Live2DModelSelection | None = None

    @property
    def available(self) -> bool:
        """返回实际选择是否可创建渲染窗口。"""

        return self.backend != RendererBackend.UNAVAILABLE.value

    @property
    def allows_runtime_fallback(self) -> bool:
        """仅自动选择允许运行期降级到精灵。

        显式后端是用户的可验证选择；即使运行期加载失败，也不能静默把
        另一种渲染器当成已经启用的结果。
        """

        return self.requested_backend == RendererBackend.AUTO.value

    @property
    def model_path(self) -> Path | None:
        """返回当前选择的 Live2D model3 描述路径。"""

        selection = self.model_selection
        if selection is not None:
            return selection.path
        return self.inventory.live2d_models[0] if self.inventory.live2d_models else None

    @property
    def model_choices(self) -> tuple[str, ...]:
        """返回可写入 ``rendering.model`` 的资源相对路径。"""

        selection = self.model_selection
        if selection is not None:
            return selection.choices
        choices: list[str] = []
        for path in self.inventory.live2d_models:
            if not _path_is_relative_to(path, self.inventory.root):
                continue
            try:
                choices.append(Path(path).resolve().relative_to(self.inventory.root).as_posix())
            except (OSError, RuntimeError, ValueError):
                continue
        return tuple(sorted(set(choices)))

    @property
    def state(self) -> RendererBackendState:
        """返回适合日志/控制台展示的选择状态。"""

        return RendererBackendState(
            requested=self.requested_backend,
            selected=self.backend,
            available=self.available,
            reason=self.reason,
            compatibility_alias=self.compatibility_alias,
        )


def probe_web_live2d_runtime(
    resource_root: str | Path,
    *,
    importer: Callable[[str], Any] | None = None,
) -> RuntimeCapability:
    """探测 Web Live2D 模型、前端资源和 Qt bridge。"""

    try:
        from .web_live2d import probe_web_live2d

        probe = probe_web_live2d(resource_root, importer=importer)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return RuntimeCapability(
            name="web_live2d",
            state=RuntimeCapabilityState.UNAVAILABLE,
            detail="Web Live2D probe failed",
            evidence=(type(exc).__name__,),
        )
    evidence = tuple(path.name for path in probe.javascript_assets) + probe.missing_qt_modules
    return RuntimeCapability(
        name="web_live2d",
        state=(
            RuntimeCapabilityState.AVAILABLE
            if probe.available
            else RuntimeCapabilityState.UNAVAILABLE
        ),
        detail=probe.message,
        evidence=evidence,
    )


def _default_opengl_factory(selection: RendererSelection) -> Any:
    """创建原生 OpenGL Live2D 实例；导入延迟到真正初始化阶段。"""

    from .live2d import Live2DRenderer

    return Live2DRenderer(
        selection.inventory.root / "live2d" / "model",
        model_path=selection.model_path,
    )


def _default_web_live2d_factory(selection: RendererSelection) -> Any:
    """创建 Web Live2D 实例；Qt 对象仍只在 ``initialize`` 中创建。"""

    from .web_live2d import WebLive2DRenderer

    return WebLive2DRenderer(
        selection.inventory.root,
        model_path=selection.model_path,
    )


def _default_sprite_factory(selection: RendererSelection) -> Any:
    """创建精灵实例。"""

    from .sprite import SpriteRenderer

    return SpriteRenderer(selection.inventory.root / "sprites")


def _default_vulkan_factory(selection: RendererSelection, **host_options: Any) -> Any:
    """创建拥有独立 QQuickWindow surface 的 Vulkan 场景图宿主。"""

    from gui.qt6.vulkan_host import create_vulkan_host

    return create_vulkan_host(selection, **host_options)


def _default_vulkan_probe(inventory: RenderAssetInventory) -> RuntimeCapability:
    """探测 Qt Vulkan 场景图入口及当前后端所需精灵资源。"""

    if not inventory.sprite_available:
        return RuntimeCapability(
            name="vulkan_qt_runtime",
            state=RuntimeCapabilityState.UNAVAILABLE,
            detail=(
                "Vulkan rendering is unavailable: scenegraph requires at least one valid "
                "WebP sprite frame"
            ),
            evidence=("missing:sprite_frame",),
        )

    probe = probe_qt_vulkan_runtime()
    return RuntimeCapability(
        name="vulkan_qt_runtime",
        state=(
            RuntimeCapabilityState.AVAILABLE
            if probe.available
            else RuntimeCapabilityState.UNAVAILABLE
        ),
        detail=probe.detail,
        evidence=probe.evidence,
    )


def _runtime_not_required(name: str, backend: str) -> RuntimeCapability:
    """记录显式后端选择跳过的无关探针。"""

    return RuntimeCapability(
        name=name,
        state=RuntimeCapabilityState.UNAVAILABLE,
        detail=f"not probed for explicit {backend} selection",
        evidence=("probe_not_required",),
    )


@dataclass(frozen=True)
class RendererRegistration:
    """一个后端的注册信息。

    工厂接收 ``RendererSelection`` 作为第一个参数，返回具体渲染器；零
    参数工厂也受 ``RendererRegistry.initialize`` 支持。工厂可以为空，表示
    该后端只有能力/诊断预留，尚未接入实例化代码。三维后端还可提供独立
    ``capability_probe``，只有探针和工厂都可用时才会被选择。
    """

    backend: str
    factory: Callable[..., Any] | None = None
    dimension: str = "2d"
    auto_priority: int | None = None
    capability_probe: Callable[[RenderAssetInventory], RuntimeCapability | bool] | None = None

    def __post_init__(self) -> None:
        normalized = normalize_renderer_backend(self.backend)
        if normalized in {
            RendererBackend.AUTO.value,
            RendererBackend.UNAVAILABLE.value,
        }:
            raise ValueError(f"renderer registration cannot use backend: {normalized}")
        dimension = str(self.dimension).strip().lower()
        if dimension not in {"2d", "3d"}:
            raise ValueError("renderer registration dimension must be 2d or 3d")
        if self.auto_priority is not None:
            int(self.auto_priority)
        if self.capability_probe is not None and not callable(self.capability_probe):
            raise TypeError("renderer registration capability_probe must be callable")
        object.__setattr__(self, "backend", normalized)
        object.__setattr__(self, "dimension", dimension)


class RendererRegistry:
    """集中管理渲染后端选择、注册和实例初始化。

    选择阶段只回答“哪个后端具备实际能力”；初始化阶段再验证工厂和实例
    生命周期。显式后端初始化失败始终返回 ``FAILED``，不会在注册表内部
    偷换为精灵。只有 ``requested_backend=auto`` 的调用方才可以根据选择
    结果自行执行既有回退策略。
    """

    def __init__(self, registrations: Iterable[RendererRegistration] = ()) -> None:
        self._registrations: dict[str, RendererRegistration] = {}
        for registration in registrations:
            self.register(registration)

    @classmethod
    def default(cls) -> RendererRegistry:
        """返回当前发行版的默认后端注册表。

        ``vulkan`` 工厂创建真实 Qt Quick 场景图宿主；选择阶段只验证
        入口和资源，实例仍须通过场景图 API 回读及可见帧验证。
        """

        return cls(
            (
                RendererRegistration(
                    RendererBackend.OPENGL.value,
                    factory=_default_opengl_factory,
                    dimension="2d",
                    auto_priority=10,
                ),
                RendererRegistration(
                    RendererBackend.WEB_LIVE2D.value,
                    factory=_default_web_live2d_factory,
                    dimension="2d",
                    auto_priority=20,
                ),
                RendererRegistration(
                    RendererBackend.SPRITE.value,
                    factory=_default_sprite_factory,
                    dimension="2d",
                    auto_priority=30,
                ),
                RendererRegistration(
                    RendererBackend.VULKAN.value,
                    factory=_default_vulkan_factory,
                    dimension="3d",
                    auto_priority=None,
                    capability_probe=_default_vulkan_probe,
                ),
            )
        )

    @property
    def backends(self) -> tuple[str, ...]:
        """按注册顺序返回可识别的后端键。"""

        return tuple(self._registrations)

    def register(
        self,
        registration: RendererRegistration | str,
        *,
        factory: Callable[..., Any] | None = None,
        dimension: str = "2d",
        auto_priority: int | None = None,
        capability_probe: Callable[[RenderAssetInventory], RuntimeCapability | bool] | None = None,
        replace: bool = False,
    ) -> RendererRegistration:
        """注册一个后端，并拒绝意外覆盖已有工厂。"""

        item = (
            registration
            if isinstance(registration, RendererRegistration)
            else RendererRegistration(
                registration,
                factory=factory,
                dimension=dimension,
                auto_priority=auto_priority,
                capability_probe=capability_probe,
            )
        )
        if item.backend in self._registrations and not replace:
            raise ValueError(f"renderer backend is already registered: {item.backend}")
        self._registrations[item.backend] = item
        return item

    def registration(self, backend: object) -> RendererRegistration | None:
        """读取一个后端注册项；未知键返回 ``None``。"""

        try:
            normalized = normalize_renderer_backend(backend)
        except (TypeError, ValueError):
            return None
        return self._registrations.get(normalized)

    def select(
        self,
        inventory_or_root: RenderAssetInventory | str | Path,
        *,
        requested_backend: str = RendererBackend.AUTO.value,
        requested_model: object = None,
        runtime: RuntimeCapability | None = None,
        web_runtime: RuntimeCapability | None = None,
        importer: Callable[[str], Any] | None = None,
    ) -> RendererSelection:
        """基于注册表和实际能力生成不可变选择结果。"""

        inventory = (
            inventory_or_root
            if isinstance(inventory_or_root, RenderAssetInventory)
            else probe_render_assets(inventory_or_root)
        )
        requested_text = str(requested_backend or RendererBackend.AUTO.value).strip().lower()
        requested = normalize_renderer_backend(requested_text)
        compatibility_alias = renderer_backend_compatibility_alias(requested_text)
        public_request = compatibility_alias or requested
        needs_model_selection = requested in {
            RendererBackend.AUTO.value,
            RendererBackend.OPENGL.value,
            RendererBackend.WEB_LIVE2D.value,
        }
        if needs_model_selection:
            catalog = Live2DModelCatalog.from_descriptors(
                inventory.root,
                inventory.live2d_models,
            )
            model_selection = (
                catalog.select(requested_model)
                if catalog.models or requested_model is not None
                else None
            )
        else:
            model_selection = None

        needs_native_probe = requested in {
            RendererBackend.AUTO.value,
            RendererBackend.OPENGL.value,
        }
        needs_web_probe = requested in {
            RendererBackend.AUTO.value,
            RendererBackend.WEB_LIVE2D.value,
        }
        live2d_runtime = (
            runtime
            if runtime is not None
            else (
                probe_live2d_runtime(importer=importer)
                if needs_native_probe
                else _runtime_not_required("live2d_runtime", public_request)
            )
        )
        web_live2d_runtime = (
            web_runtime
            if web_runtime is not None
            else (
                probe_web_live2d_runtime(inventory.root, importer=importer)
                if needs_web_probe
                else _runtime_not_required("web_live2d", public_request)
            )
        )

        def unavailable(reason: str, *, alias: str = "") -> RendererSelection:
            return RendererSelection(
                backend=RendererBackend.UNAVAILABLE.value,
                reason=reason,
                inventory=inventory,
                live2d_runtime=live2d_runtime,
                requested_backend=public_request,
                web_live2d_runtime=web_live2d_runtime,
                compatibility_alias=alias or compatibility_alias,
                model_selection=model_selection,
            )

        def selected(backend: RendererBackend, reason: str) -> RendererSelection:
            registration = self.registration(backend.value)
            if registration is None:
                return unavailable(f"renderer backend is not registered: {backend.value}")
            if registration.factory is None:
                return unavailable(f"renderer backend has no factory: {backend.value}")
            return RendererSelection(
                backend=backend.value,
                reason=reason,
                inventory=inventory,
                live2d_runtime=live2d_runtime,
                requested_backend=public_request,
                web_live2d_runtime=web_live2d_runtime,
                compatibility_alias=compatibility_alias,
                model_selection=model_selection,
            )

        # 显式模型键不能在自动模式下静默退回另一个模型；精灵后端不读取
        # Live2D 描述，因此用户明确选择精灵时保留其独立启动能力。
        if (
            model_selection is not None
            and model_selection.explicit
            and not model_selection.available
            and requested != RendererBackend.SPRITE.value
        ):
            return unavailable(model_selection.reason or "requested Live2D model is unavailable")

        if requested == RendererBackend.VULKAN.value:
            registration = self.registration(RendererBackend.VULKAN.value)
            if registration is None:
                return unavailable("Vulkan rendering is unavailable: backend is not registered")
            if registration.factory is None:
                return unavailable("Vulkan rendering is unavailable: no initialization factory")
            probe = registration.capability_probe
            if probe is None:
                return unavailable("Vulkan rendering is unavailable: no bound capability probe")
            try:
                probe_result = probe(inventory)
            except Exception as exc:
                return unavailable("Vulkan capability probe failed: " + type(exc).__name__)
            probe_available = (
                probe_result.available
                if isinstance(probe_result, RuntimeCapability)
                else bool(probe_result)
            )
            if not probe_available:
                detail = (
                    probe_result.detail
                    if isinstance(probe_result, RuntimeCapability)
                    else "Vulkan capability probe reported unavailable"
                )
                return unavailable(detail)
            detail = (
                probe_result.detail
                if isinstance(probe_result, RuntimeCapability)
                else "registered Vulkan renderer and capability probe are available"
            )
            return selected(RendererBackend.VULKAN, detail)
        if requested == RendererBackend.OPENGL.value:
            if inventory.live2d_available and live2d_runtime.available:
                return selected(
                    RendererBackend.OPENGL,
                    "requested OpenGL Live2D backend is available",
                )
            return unavailable("requested OpenGL Live2D backend is unavailable")
        if requested == RendererBackend.WEB_LIVE2D.value:
            if inventory.live2d_available and web_live2d_runtime.available:
                return selected(
                    RendererBackend.WEB_LIVE2D,
                    "requested Web Live2D backend is available",
                )
            return unavailable("requested Web Live2D backend is unavailable")
        if requested == RendererBackend.SPRITE.value:
            if inventory.sprite_available:
                return selected(
                    RendererBackend.SPRITE,
                    "requested sprite backend is available",
                )
            return unavailable("requested sprite backend has no valid WebP frames")

        if inventory.live2d_available and web_live2d_runtime.available:
            # Web Live2D 保留为 auto 的稳定默认路径：它与当前已验收的
            # 透明画布、动作、换模和尺寸适配一致。原生 OpenGL 仍可由
            # 用户显式选择，避免不同平台的 live2d-py/Cubism 组合在
            # 未经完整视觉验收时改变桌宠外观。
            return selected(
                RendererBackend.WEB_LIVE2D,
                "auto selected the stable Web Live2D backend",
            )
        if inventory.live2d_available and live2d_runtime.available:
            return selected(
                RendererBackend.OPENGL,
                "auto selected the native OpenGL Live2D backend because Web Live2D is unavailable",
            )
        if inventory.sprite_available:
            return selected(RendererBackend.SPRITE, "auto selected the sprite fallback backend")
        return unavailable("auto found no usable rendering backend")

    def initialize(
        self,
        selection: RendererSelection,
        *,
        factory: Callable[..., Any] | None = None,
        surface: object | None = None,
        **factory_kwargs: Any,
    ) -> RendererInitializationResult:
        """按选择结果创建并初始化实例，不执行隐式后端替换。"""

        if not isinstance(selection, RendererSelection):
            raise TypeError("selection must be a RendererSelection")
        backend = str(selection.backend)
        if backend == RendererBackend.UNAVAILABLE.value:
            return RendererInitializationResult(
                backend=backend,
                state=RendererInitializationState.UNAVAILABLE,
                reason=selection.reason or "selected renderer is unavailable",
            )
        registration = self.registration(backend)
        if registration is None:
            return RendererInitializationResult(
                backend=backend,
                state=RendererInitializationState.UNAVAILABLE,
                reason=f"renderer backend is not registered: {backend}",
            )
        creator = factory or registration.factory
        if creator is None:
            return RendererInitializationResult(
                backend=backend,
                state=RendererInitializationState.UNAVAILABLE,
                reason=f"renderer backend has no factory: {backend}",
            )
        instance: Any | None = None

        def close_instance(value: Any | None) -> None:
            """释放工厂已经创建、但未进入可用状态的实例。"""

            shutdown = getattr(value, "shutdown", None)
            if not callable(shutdown):
                return
            try:
                shutdown()
            except Exception:
                # 初始化错误本身比清理错误更具诊断价值；清理失败只进入
                # debug 日志，不能覆盖明确的 unavailable/failed 结果。
                logging.getLogger(__name__).debug("renderer cleanup failed", exc_info=True)

        try:
            # 注册契约优先传入完整选择对象；零参数工厂也被明确支持，方便
            # 三维适配器在不依赖资源清单时独立构造。通过签名绑定区分两者，
            # 不会把工厂内部抛出的 TypeError 误判为另一种调用形式。
            try:
                signature = inspect.signature(creator)
            except (TypeError, ValueError):
                signature = None
            if signature is not None:
                try:
                    signature.bind(selection, **factory_kwargs)
                except TypeError:
                    signature.bind(**factory_kwargs)
                    instance = creator(**factory_kwargs)
                else:
                    instance = creator(selection, **factory_kwargs)
            else:
                instance = creator(selection, **factory_kwargs)
            if instance is None:
                raise RuntimeError("renderer factory returned no instance")
            capabilities = getattr(instance, "capabilities", None)
            if capabilities is not None:
                capability_backend = str(getattr(capabilities, "backend", backend))
                if capability_backend != backend:
                    raise RuntimeError(
                        f"renderer capability backend mismatch: expected {backend}, "
                        f"got {capability_backend}"
                    )
                initialization_pending = bool(getattr(instance, "initialization_pending", False))
                if (
                    not bool(getattr(capabilities, "available", True))
                    and not initialization_pending
                ):
                    reason = str(getattr(capabilities, "message", "renderer is unavailable"))
                    close_instance(instance)
                    return RendererInitializationResult(
                        backend=backend,
                        state=RendererInitializationState.UNAVAILABLE,
                        renderer=None,
                        reason=reason,
                    )
            initializer = getattr(instance, "initialize", None)
            if callable(initializer):
                initialized = initializer() if surface is None else initializer(surface)
                if initialized is False:
                    raise RuntimeError("renderer initialize returned false")
            capabilities = getattr(instance, "capabilities", None)
            if capabilities is not None and not bool(getattr(capabilities, "available", True)):
                raise RuntimeError(
                    str(getattr(capabilities, "message", "renderer became unavailable"))
                )
            return RendererInitializationResult(
                backend=backend,
                state=RendererInitializationState.READY,
                renderer=instance,
                reason="renderer initialized",
            )
        except Exception as exc:
            close_instance(instance)
            return RendererInitializationResult(
                backend=backend,
                state=RendererInitializationState.FAILED,
                renderer=None,
                reason=f"renderer initialization failed: {type(exc).__name__}: {exc}",
            )

    # ``create`` 是初始化契约的语义别名，保留同一返回类型和失败行为。
    create = initialize


def default_renderer_registry() -> RendererRegistry:
    """返回新的默认注册表，避免调用方共享可变注册状态。"""

    return RendererRegistry.default()


def select_renderer(
    inventory_or_root: RenderAssetInventory | str | Path,
    *,
    requested_backend: str = RendererBackend.AUTO.value,
    requested_model: object = None,
    runtime: RuntimeCapability | None = None,
    web_runtime: RuntimeCapability | None = None,
    importer: Callable[[str], Any] | None = None,
) -> RendererSelection:
    """兼容旧调用方的默认注册表选择入口。"""

    return RendererRegistry.default().select(
        inventory_or_root,
        requested_backend=requested_backend,
        requested_model=requested_model,
        runtime=runtime,
        web_runtime=web_runtime,
        importer=importer,
    )


__all__ = [
    "RenderAssetInventory",
    "Live2DModelCatalog",
    "Live2DModelSelection",
    "RuntimeCapability",
    "RuntimeCapabilityState",
    "RendererSelection",
    "RendererRegistration",
    "RendererRegistry",
    "default_renderer_registry",
    "probe_web_live2d_runtime",
    "probe_live2d_runtime",
    "probe_render_assets",
    "select_renderer",
]
