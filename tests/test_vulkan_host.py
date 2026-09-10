from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gui.renderers.assets import RendererRegistry
from gui.renderers.protocol import RendererCapabilities, RendererInitializationState


def test_registry_initializes_a_pending_renderer_before_accepting_capabilities() -> None:
    class PendingRenderer:
        initialization_pending = True

        def __init__(self) -> None:
            self.ready = False

        @property
        def capabilities(self) -> RendererCapabilities:
            return RendererCapabilities("vulkan", self.ready)

        def initialize(self) -> bool:
            self.ready = True
            self.initialization_pending = False
            return True

    registry = RendererRegistry()
    registry.register(
        "vulkan",
        factory=lambda: PendingRenderer(),
        dimension="3d",
        capability_probe=lambda _inventory: True,
    )
    from gui.renderers.assets import RenderAssetInventory

    selection = registry.select(
        RenderAssetInventory(Path("/tmp/meapet-vulkan-pending")),
        requested_backend="vulkan",
    )
    result = registry.initialize(selection)

    assert result.state is RendererInitializationState.READY
    assert result.available is True
    assert result.renderer is not None
    assert result.renderer.capabilities.available is True


def test_default_vulkan_factory_accepts_host_options(monkeypatch, tmp_path: Path) -> None:
    from gui.renderers import assets

    received: dict[str, object] = {}
    sentinel = object()

    def create(selection, **options):
        received["selection"] = selection
        received.update(options)
        return sentinel

    monkeypatch.setattr("gui.qt6.vulkan_host.create_vulkan_host", create)
    selection = RendererRegistry.default().select(
        assets.probe_render_assets(tmp_path),
        requested_backend="vulkan",
    )
    # 空资源会在选择阶段拒绝 Vulkan；构造一份只用于工厂契约的精确选择。
    selection = selection.__class__(
        requested_backend="vulkan",
        backend="vulkan",
        reason="test",
        inventory=assets.RenderAssetInventory(tmp_path),
        live2d_runtime=selection.live2d_runtime,
    )
    result = RendererRegistry.default().initialize(
        selection,
        factory=assets._default_vulkan_factory,
        platform="platform",
        sprite_scale=0.75,
        always_on_top=False,
    )

    assert result.succeeded is True
    assert result.renderer is sentinel
    assert received == {
        "selection": selection,
        "platform": "platform",
        "sprite_scale": 0.75,
        "always_on_top": False,
    }


def test_offscreen_scenegraph_readback_fails_closed_in_isolated_process() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
from PySide6.QtWidgets import QApplication
from gui.renderers.assets import RendererRegistry, probe_render_assets

app = QApplication([])
registry = RendererRegistry.default()
selection = registry.select(probe_render_assets('temp/resources'), requested_backend='vulkan')
result = registry.initialize(selection)
print(result.state.value, result.available, result.renderer is None)
raise SystemExit(0 if not result.available and result.renderer is None else 3)
"""
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "failed False True" in completed.stdout


def test_vulkan_preparation_requests_global_alpha_before_first_quick_window() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
from PySide6.QtQuick import QQuickWindow
from PySide6.QtWidgets import QApplication
from gui.qt6.vulkan_host import prepare_vulkan_scenegraph

app = QApplication([])
ok, reason = prepare_vulkan_scenegraph()
print(ok, QQuickWindow.hasDefaultAlphaBuffer(), reason)
raise SystemExit(0 if ok and QQuickWindow.hasDefaultAlphaBuffer() else 4)
"""
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert completed.returncode == 0, completed.stderr
    assert "True True" in completed.stdout
