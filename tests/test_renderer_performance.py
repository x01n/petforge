from __future__ import annotations

from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

from gui.renderers import assets
from gui.renderers.assets import RenderAssetInventory, RendererRegistry
from gui.renderers.protocol import CANONICAL_RENDERER_BACKEND_CHOICES, normalize_renderer_backend
from gui.renderers.threed import probe_qt_vulkan_runtime


def test_explicit_sprite_selection_skips_live2d_runtime_probes(monkeypatch) -> None:
    calls = {"native": 0, "web": 0}

    def native_probe(*_args, **_kwargs):
        calls["native"] += 1
        raise AssertionError("native probe is unrelated to explicit sprite selection")

    def web_probe(*_args, **_kwargs):
        calls["web"] += 1
        raise AssertionError("web probe is unrelated to explicit sprite selection")

    monkeypatch.setattr(assets, "probe_live2d_runtime", native_probe)
    monkeypatch.setattr(assets, "probe_web_live2d_runtime", web_probe)
    inventory = RenderAssetInventory(
        Path("/tmp/meapet-render-probe-budget"),
        sprite_frames=(Path("/tmp/meapet-render-probe-budget/frame.webp"),),
    )

    selection = RendererRegistry.default().select(inventory, requested_backend="sprite")

    assert selection.backend == "sprite"
    assert calls == {"native": 0, "web": 0}
    assert selection.live2d_runtime.evidence == ("probe_not_required",)
    assert selection.web_live2d_runtime is not None
    assert selection.web_live2d_runtime.evidence == ("probe_not_required",)


def test_explicit_sprite_selection_microbenchmark_has_constant_probe_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        assets,
        "probe_live2d_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected probe")),
    )
    monkeypatch.setattr(
        assets,
        "probe_web_live2d_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected probe")),
    )
    inventory = RenderAssetInventory(
        Path("/tmp/meapet-render-selection-benchmark"),
        sprite_frames=(Path("/tmp/meapet-render-selection-benchmark/frame.webp"),),
    )
    registry = RendererRegistry.default()

    started = perf_counter()
    for _ in range(2_000):
        assert registry.select(inventory, requested_backend="sprite").backend == "sprite"
    elapsed = perf_counter() - started

    assert elapsed < 5.0


def test_explicit_vulkan_alias_skips_unrelated_live2d_probes(monkeypatch) -> None:
    def unexpected_probe(*_args, **_kwargs):
        raise AssertionError("Live2D probe is unrelated to explicit Vulkan selection")

    monkeypatch.setattr(assets, "probe_live2d_runtime", unexpected_probe)
    monkeypatch.setattr(assets, "probe_web_live2d_runtime", unexpected_probe)

    selection = RendererRegistry.default().select(
        RenderAssetInventory(Path("/tmp/meapet-vulkan-probe-budget")),
        requested_backend="vllakn",
    )

    assert selection.backend == "unavailable"
    assert selection.requested_backend == "vllakn"
    assert selection.compatibility_alias == "vllakn"
    assert selection.live2d_runtime.evidence == ("probe_not_required",)


def test_vulkan_aliases_normalize_to_one_backend_identity() -> None:
    assert "vulkan" in CANONICAL_RENDERER_BACKEND_CHOICES
    assert "vllank" not in CANONICAL_RENDERER_BACKEND_CHOICES
    assert "vllakn" not in CANONICAL_RENDERER_BACKEND_CHOICES
    assert normalize_renderer_backend("vulkan") == "vulkan"
    assert normalize_renderer_backend("vllank") == "vulkan"
    assert normalize_renderer_backend("vllakn") == "vulkan"


def test_qt_vulkan_probe_requires_exact_scenegraph_bindings_and_loader() -> None:
    class GraphicsApi:
        Vulkan = object()

    class QuickWindow:
        setGraphicsApi = staticmethod(lambda _api: None)
        graphicsApi = staticmethod(lambda: GraphicsApi.Vulkan)
        setDefaultAlphaBuffer = staticmethod(lambda _enabled: None)
        hasDefaultAlphaBuffer = staticmethod(lambda: True)

    incomplete = probe_qt_vulkan_runtime(
        importer=lambda _name: SimpleNamespace(QQuickWindow=QuickWindow),
        library_finder=lambda _name: "libvulkan.so.1",
    )
    complete = probe_qt_vulkan_runtime(
        importer=lambda _name: SimpleNamespace(
            QQuickWindow=QuickWindow,
            QQuickView=object,
            QSGRendererInterface=SimpleNamespace(GraphicsApi=GraphicsApi),
        ),
        library_finder=lambda _name: "libvulkan.so.1",
    )

    assert incomplete.available is False
    assert "QQuickView" in incomplete.detail
    assert complete.available is True
    assert complete.evidence == (
        "present:PySide6.QtQuick.QQuickWindow",
        "present:PySide6.QtQuick.QQuickView",
        "present:PySide6.QtQuick.QSGRendererInterface",
        "present:QQuickWindow.setGraphicsApi",
        "present:QQuickWindow.graphicsApi",
        "present:QQuickWindow.setDefaultAlphaBuffer",
        "present:QQuickWindow.hasDefaultAlphaBuffer",
        "present:QSGRendererInterface.GraphicsApi.Vulkan",
        "loader:libvulkan.so.1",
    )


def test_vulkan_factory_without_bound_probe_never_reports_available() -> None:
    registry = RendererRegistry.default()
    registry.register(
        "vulkan",
        factory=lambda: object(),
        dimension="3d",
        capability_probe=None,
        replace=True,
    )

    selection = registry.select(
        RenderAssetInventory(Path("/tmp/meapet-vulkan-no-probe")),
        requested_backend="vulkan",
    )

    assert selection.backend == "unavailable"
    assert selection.available is False
    assert selection.reason == "Vulkan rendering is unavailable: no bound capability probe"
