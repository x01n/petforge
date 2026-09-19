from __future__ import annotations

from pathlib import Path

import pytest

from gui.renderers.assets import (
    RenderAssetInventory,
    RendererRegistry,
    RendererSelection,
    RuntimeCapability,
    RuntimeCapabilityState,
)
from gui.renderers.protocol import (
    RendererCapabilities,
    RendererInitializationState,
    RendererState,
)
from gui.renderers.threed import Reserved3DRenderer


def _selection(
    *,
    backend: str = "opengl",
    available: bool = True,
) -> RendererSelection:
    inventory = RenderAssetInventory(
        Path("/tmp/meapet-registry-test"),
        live2d_models=(Path("/tmp/meapet-registry-test/demo.model3.json"),),
    )
    return RendererSelection(
        backend=backend if available else "unavailable",
        reason="synthetic unavailable" if not available else "synthetic available",
        inventory=inventory,
        live2d_runtime=RuntimeCapability(
            "live2d_runtime",
            RuntimeCapabilityState.AVAILABLE,
            "synthetic runtime",
        ),
        requested_backend=backend,
    )


def test_default_registry_requires_live2d_vulkan_provider() -> None:
    registry = RendererRegistry.default()

    assert registry.backends == ("opengl", "web_live2d", "sprite", "vulkan")
    registration = registry.registration("vulkan")
    assert registration is not None
    assert registration.dimension == "3d"
    assert callable(registration.factory)
    assert callable(registration.capability_probe)

    selection = registry.select(
        RenderAssetInventory(Path("/tmp/meapet-empty")),
        requested_backend="vllank",
        runtime=RuntimeCapability(
            "live2d_runtime", RuntimeCapabilityState.UNAVAILABLE, "not installed"
        ),
        web_runtime=RuntimeCapability(
            "web_live2d", RuntimeCapabilityState.UNAVAILABLE, "not installed"
        ),
    )
    result = registry.initialize(selection)
    assert selection.backend == "unavailable"
    assert selection.requested_backend == "vllank"
    assert selection.compatibility_alias == "vllank"
    assert result.state is RendererInitializationState.UNAVAILABLE
    assert result.renderer is None
    assert "Vulkan rendering is unavailable" in result.reason


def test_reserved_vulkan_resource_reload_is_explicitly_unavailable(tmp_path: Path) -> None:
    renderer = Reserved3DRenderer()

    result = renderer.reload_resources(tmp_path, model_path="model.glb", sprite_scale=1.0)

    assert result["status"] == "unavailable"
    assert "no initialized resource lifecycle" in str(result["reason"])


def test_registry_can_activate_a_real_3d_registration_without_changing_default() -> None:
    registry = RendererRegistry.default()

    class Fake3D:
        capabilities = RendererCapabilities("vulkan", True, message="synthetic 3D")

        def initialize(self) -> bool:
            return True

    registry.register(
        "vulkan",
        factory=lambda: Fake3D(),
        dimension="3d",
        capability_probe=lambda _inventory: True,
        replace=True,
    )
    inventory = RenderAssetInventory(Path("/tmp/meapet-3d"))
    selection = registry.select(
        inventory,
        requested_backend="vllakn",
        runtime=RuntimeCapability(
            "live2d_runtime", RuntimeCapabilityState.UNAVAILABLE, "not installed"
        ),
        web_runtime=RuntimeCapability(
            "web_live2d", RuntimeCapabilityState.UNAVAILABLE, "not installed"
        ),
    )
    result = registry.initialize(selection)

    assert selection.backend == "vulkan"
    assert selection.requested_backend == "vllakn"
    assert selection.compatibility_alias == "vllakn"
    assert selection.available is True
    assert result.state is RendererInitializationState.READY
    assert result.available is True


def test_registry_initialization_success_is_distinct_from_selection() -> None:
    registry = RendererRegistry()
    registry.register("opengl")

    class FakeRenderer:
        capabilities = RendererCapabilities("opengl", True)
        state = RendererState()

        def __init__(self) -> None:
            self.initialized = False

        def initialize(self) -> bool:
            self.initialized = True
            return True

    renderer = FakeRenderer()
    selection = _selection()
    result = registry.initialize(selection, factory=lambda selected: renderer)

    assert result.state is RendererInitializationState.READY
    assert result.available is True
    assert result.renderer is renderer
    assert renderer.initialized is True


def test_registry_factory_failure_is_terminal_and_does_not_switch_backend() -> None:
    registry = RendererRegistry()
    registry.register("opengl")
    selection = _selection()
    calls: list[str] = []

    def broken_factory(selected: RendererSelection) -> object:
        calls.append(selected.backend)
        raise RuntimeError("native load failed")

    result = registry.initialize(selection, factory=broken_factory)

    assert calls == ["opengl"]
    assert result.state is RendererInitializationState.FAILED
    assert result.renderer is None
    assert "native load failed" in result.reason
    assert result.backend == "opengl"


def test_registry_unavailable_selection_does_not_invoke_factory() -> None:
    registry = RendererRegistry()
    registry.register("opengl")
    selection = _selection(available=False)
    called = False

    def factory(selected: RendererSelection) -> object:
        nonlocal called
        called = True
        return object()

    result = registry.initialize(selection, factory=factory)

    assert called is False
    assert result.state is RendererInitializationState.UNAVAILABLE
    assert result.backend == "unavailable"


def test_registry_rejects_post_factory_capability_mismatch() -> None:
    registry = RendererRegistry()
    registry.register("opengl")

    class WrongRenderer:
        capabilities = RendererCapabilities("sprite", True)

    result = registry.initialize(_selection(), factory=lambda selected: WrongRenderer())

    assert result.state is RendererInitializationState.FAILED
    assert "backend mismatch" in result.reason


def test_registry_cleans_up_instance_that_reports_unavailable_capabilities() -> None:
    registry = RendererRegistry()
    registry.register("vulkan", factory=lambda: None, dimension="3d", replace=False)
    selection = _selection(backend="vulkan")

    class UnavailableRenderer:
        capabilities = RendererCapabilities("vulkan", False, message="device unavailable")

        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    instance = UnavailableRenderer()
    result = registry.initialize(selection, factory=lambda: instance)

    assert result.state is RendererInitializationState.UNAVAILABLE
    assert result.renderer is None
    assert instance.shutdown_calls == 1
    assert result.reason == "device unavailable"


def test_qt_app_selection_uses_the_shared_renderer_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gui.qt6 import app as qt_app

    calls: list[tuple[object, object]] = []
    expected = _selection(backend="sprite")

    class FakeRegistry:
        def select(self, root: object, *, requested_backend: object) -> RendererSelection:
            calls.append((root, requested_backend))
            return expected

    monkeypatch.setattr(qt_app, "default_renderer_registry", lambda: FakeRegistry())
    result = qt_app._select_renderer_for_run("/tmp/meapet-resources", "sprite")

    assert result is expected
    assert calls == [("/tmp/meapet-resources", "sprite")]
