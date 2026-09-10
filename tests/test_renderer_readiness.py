from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.rendering import RendererReadyState, RendererReadyStatus
from gui.qt6.app import (
    _renderer_ready_status,
    _restart_ready_receipt_matches,
    _write_restart_ready_receipt,
)
from gui.renderers.model_catalog import Live2DModelCatalog
from gui.renderers.web_live2d import WebLive2DRenderer


def _write_model(root: Path, name: str, *, texture: bytes = b"texture") -> Path:
    directory = root / "live2d" / "model" / name
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = directory / f"{name}.model3.json"
    (directory / f"{name}.moc3").write_bytes(b"moc")
    (directory / "texture.png").write_bytes(texture)
    descriptor.write_text(
        json.dumps(
            {
                "FileReferences": {
                    "Moc": f"{name}.moc3",
                    "Textures": ["texture.png"],
                }
            }
        ),
        encoding="utf-8",
    )
    return descriptor


def _write_web_assets(root: Path) -> None:
    javascript_root = root / "live2d" / "js"
    javascript_root.mkdir(parents=True, exist_ok=True)
    for name in ("pixi.min.js", "live2dcubismcore.min.js", "pixi-live2d-display.min.js"):
        (javascript_root / name).write_text("// test asset", encoding="utf-8")


class _FakePage:
    def __init__(self) -> None:
        self.callback = None
        self.scripts: list[str] = []

    def runJavaScript(self, script: str, callback=None) -> None:  # noqa: N802
        self.scripts.append(script)
        self.callback = callback


class _FakeView:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    def page(self) -> _FakePage:
        return self._page


class _FakeImage:
    class Format:
        Format_RGBA8888 = 1

    def __init__(self, *, alpha: int = 255) -> None:
        self._alpha = alpha

    def isNull(self) -> bool:  # noqa: N802
        return False

    def width(self) -> int:
        return 1

    def height(self) -> int:
        return 1

    def convertToFormat(self, _value: object) -> _FakeImage:  # noqa: N802
        return self

    def sizeInBytes(self) -> int:  # noqa: N802
        return 4

    def constBits(self) -> bytes:  # noqa: N802
        return bytes((0, 0, 0, self._alpha))


def _ready_probe_payload(*, alpha_nonempty: bool = True) -> str:
    return json.dumps(
        {
            "debug": {
                "geometryValid": True,
                "renderOrderValid": True,
                "viewportStale": False,
                "modelVisible": True,
                "width": 100,
                "height": 200,
            },
            "mask": {
                "status": "ok",
                "data": "data:image/png;base64,AA==",
                "raw_alpha_nonempty": alpha_nonempty,
            },
        }
    )


def _ready_web_status() -> RendererReadyStatus:
    return RendererReadyStatus(
        backend="web_live2d",
        state=RendererReadyState.READY,
        bridge_ready=True,
        model_ready=True,
        geometry_valid=True,
        frame_visible=True,
        alpha_nonempty=True,
        generation=3,
    )


def test_renderer_ready_contract_is_fixed_sanitized_and_strict() -> None:
    status = _ready_web_status()

    assert status.public() == {
        "backend": "web_live2d",
        "state": "ready",
        "ready": True,
        "terminal": True,
        "bridge_ready": True,
        "model_ready": True,
        "geometry_valid": True,
        "frame_visible": True,
        "alpha_nonempty": True,
        "actual_api": "",
        "reason_code": "",
        "generation": 3,
    }
    assert "path" not in status.public()
    assert "url" not in status.public()
    with pytest.raises(ValueError, match="evidence"):
        RendererReadyStatus(
            backend="web_live2d",
            state=RendererReadyState.READY,
            bridge_ready=True,
        )


def test_web_ready_requires_bridge_geometry_visible_model_and_raw_alpha(tmp_path: Path) -> None:
    model = _write_model(tmp_path, "first")
    _write_web_assets(tmp_path)
    renderer = WebLive2DRenderer(tmp_path, model_path=model, importer=lambda _name: object())
    page = _FakePage()
    renderer._view = _FakeView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True

    pending = renderer.request_ready_status()

    assert pending.state is RendererReadyState.PENDING
    assert page.callback is not None
    page.callback(
        json.dumps(
            {
                "debug": {
                    "geometryValid": True,
                    "renderOrderValid": True,
                    "viewportStale": False,
                    "modelVisible": True,
                    "width": 100,
                    "height": 200,
                },
                "mask": {
                    "status": "ok",
                    "data": "data:image/png;base64,AA==",
                    "raw_alpha_nonempty": True,
                },
            }
        )
    )
    assert renderer.ready_status.ready is True
    assert renderer.ready_status.public()["alpha_nonempty"] is True

    renderer._invalidate_ready_status("frame_pending")
    renderer.request_ready_status()
    assert page.callback is not None
    page.callback(
        {
            "debug": {
                "geometryValid": True,
                "renderOrderValid": True,
                "viewportStale": False,
                "modelVisible": True,
                "width": 100,
                "height": 200,
            },
            "mask": {
                "status": "ok",
                "data": "data:image/png;base64,AA==",
                "raw_alpha_nonempty": False,
            },
        }
    )
    assert renderer.ready_status.state is RendererReadyState.PENDING
    assert renderer.ready_status.alpha_nonempty is False


def test_web_model_reload_timeout_releases_transaction_and_keeps_old_model(
    tmp_path: Path,
) -> None:
    first = _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")
    _write_web_assets(tmp_path)
    renderer = WebLive2DRenderer(tmp_path, model_path=first, importer=lambda _name: object())
    renderer._view = object()
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append  # type: ignore[method-assign]
    now = [10.0]
    renderer._clock = lambda: now[0]
    renderer._model_reload_timeout_seconds = 2.0
    receipts: list[dict[str, object]] = []
    renderer.set_model_reload_callback(lambda value: receipts.append(dict(value)))
    old_probe = renderer.probe

    request = renderer.reload_model(second)
    now[0] = 12.1
    expired = renderer.model_reload_status

    assert request["status"] == "pending"
    assert expired["status"] == "failed"
    assert renderer._pending_model_reload is None
    assert renderer.probe is old_probe
    assert renderer.capabilities.model_path == first.resolve()
    assert receipts[-1]["request_id"] == request["request_id"]
    assert any("rollbackModelReload" in script for script in scripts)
    assert renderer.reload_model(second)["status"] == "pending"


def test_web_model_reload_uses_prepared_accept_finalize_handshake(tmp_path: Path) -> None:
    first = _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")
    _write_web_assets(tmp_path)
    renderer = WebLive2DRenderer(tmp_path, model_path=first, importer=lambda _name: object())
    page = _FakePage()
    renderer._view = _FakeView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append  # type: ignore[method-assign]

    request = renderer.reload_model(second)
    request_id = request["request_id"]
    renderer._on_model_reload_finished(json.dumps({"request_id": request_id, "status": "prepared"}))

    assert renderer._pending_model_reload is not None
    assert renderer.model_reload_status["phase"] == "prepared"
    assert "acceptModelReload" in scripts[-1]
    assert renderer.probe.model_path == first.resolve()

    renderer._on_model_reload_finished(
        json.dumps({"request_id": request_id, "status": "available"})
    )
    assert renderer._pending_model_reload is not None
    assert renderer.model_reload_status["phase"] == "verifying"
    assert renderer.probe.model_path == first.resolve()
    assert page.callback is not None
    page.callback(_ready_probe_payload())
    assert renderer._pending_model_reload is None
    assert "finalizeModelReload" in scripts[-1]
    assert renderer.probe.model_path == second.resolve()
    html = renderer._html_document()
    assert "function rollbackModelReload(requestId)" in html
    assert "function finalizeModelReload(requestId)" in html


def test_web_runtime_avoids_duplicate_vertex_scan_and_limits_alpha_probe_to_readiness(
    tmp_path: Path,
) -> None:
    model = _write_model(tmp_path, "performance")
    _write_web_assets(tmp_path)
    renderer = WebLive2DRenderer(tmp_path, model_path=model, importer=lambda _name: object())
    page = _FakePage()
    renderer._view = _FakeView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True

    html = renderer._html_document()
    renderer.request_ready_status()

    assert "mask: api.inputMask({probeAlpha: true})" in page.scripts[-1]
    assert "if (options.probeAlpha === true && gl" in html
    assert "const fitted = geometryReady && fitModelToViewport();" in html
    assert "const frameGeometryValid = validateDrawableFrame();" not in html
    assert "force || vertexDirty || structuralDirty" in html
    assert "now - renderOrderAuditState.lastAuditAt < 500" in html


def test_web_runtime_flushes_parameter_geometry_after_post_update_pose_writes(
    tmp_path: Path,
) -> None:
    """姿态参数在 internalModel.update 后写入时必须当帧重算 Core 顶点。"""

    model = _write_model(tmp_path, "parameter-dirty")
    _write_web_assets(tmp_path)
    renderer = WebLive2DRenderer(
        tmp_path,
        model_path=model,
        importer=lambda _name: object(),
    )
    html = renderer._html_document()

    assert "let coreParameterDirty = false;" in html
    assert "coreParameterDirty = true;" in html
    assert "function flushCoreParameterGeometry()" in html
    # 最终刷新位于两个时间线分支之后，程序化注视/口型没有时间线变化时
    # 也能消费脏标记；刷新失败会保留标记供下一帧重试。
    flush_start = html.index("function flushCoreParameterGeometry()")
    assert "if (!refreshCoreGeometry()) return false;" in html[flush_start : flush_start + 700]
    advance_start = html.index("const expressionRequestChanged = advanceExpressionTimeline(dt);")
    assert "if (coreParameterDirty)" in html[advance_start:]


def test_same_path_content_change_and_explicit_reload_bypass_unchanged(tmp_path: Path) -> None:
    first = _write_model(tmp_path, "first", texture=b"old")
    _write_web_assets(tmp_path)
    renderer = WebLive2DRenderer(tmp_path, model_path=first, importer=lambda _name: object())
    page = _FakePage()
    renderer._view = _FakeView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append  # type: ignore[method-assign]
    old_version = renderer.probe.resource_version

    (first.parent / "texture.png").write_bytes(b"new-content")
    changed = renderer.reload_model(first)

    assert changed["status"] == "pending"
    assert "meapet_resource_version=" in scripts[-1]
    renderer._on_model_reload_finished(
        json.dumps({"request_id": changed["request_id"], "status": "prepared"})
    )
    renderer._on_model_reload_finished(
        json.dumps({"request_id": changed["request_id"], "status": "available"})
    )
    assert page.callback is not None
    page.callback(_ready_probe_payload())
    assert renderer.probe.resource_version != old_version
    assert renderer.reload_model(first)["status"] == "unchanged"
    forced = renderer.reload_model(first, force=True)
    assert forced["status"] == "pending"


def test_web_model_reload_rolls_back_when_new_frame_never_has_alpha(tmp_path: Path) -> None:
    first = _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")
    _write_web_assets(tmp_path)
    renderer = WebLive2DRenderer(tmp_path, model_path=first, importer=lambda _name: object())
    page = _FakePage()
    renderer._view = _FakeView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append  # type: ignore[method-assign]
    now = [10.0]
    renderer._clock = lambda: now[0]
    renderer._model_reload_timeout_seconds = 1.0

    request = renderer.reload_model(second)
    renderer._on_model_reload_finished(
        json.dumps({"request_id": request["request_id"], "status": "prepared"})
    )
    renderer._on_model_reload_finished(
        json.dumps({"request_id": request["request_id"], "status": "available"})
    )
    assert page.callback is not None
    page.callback(_ready_probe_payload(alpha_nonempty=False))
    assert renderer.model_reload_status["phase"] == "verifying"
    assert renderer.probe.model_path == first.resolve()

    now[0] = 11.1
    renderer.advance(0.01)
    assert renderer.model_reload_status["status"] == "failed"
    assert renderer.probe.model_path == first.resolve()
    assert any("rollbackModelReload" in script for script in scripts)


def test_web_renderer_discarded_lifecycle_closes_pending_reload_without_rollback_dispatch(
    tmp_path: Path,
) -> None:
    """页面进入 Discarded 时应先封锁回滚脚本，并收敛为 CLOSED。"""

    first = _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")
    _write_web_assets(tmp_path)
    renderer = WebLive2DRenderer(tmp_path, model_path=first, importer=lambda _name: object())

    class DiscardedPage(_FakePage):
        def __init__(self) -> None:
            super().__init__()
            self.state_name = "Active"

        def lifecycleState(self):  # noqa: N802
            return SimpleNamespace(name=self.state_name)

    page = DiscardedPage()
    renderer._view = _FakeView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True

    request = renderer.reload_model(second)
    assert request["status"] == "pending"
    assert len(page.scripts) == 1
    assert "reloadModel" in page.scripts[0]

    page.state_name = "Discarded"
    renderer._on_page_lifecycle_state_changed(SimpleNamespace(name="Discarded"))

    assert renderer.page_ready is False
    assert renderer.model_ready is False
    assert renderer.ready_status.state is RendererReadyState.CLOSED
    assert renderer.ready_status.reason_code == "page_closed"
    assert renderer._pending_model_reload is None
    assert renderer.model_reload_status["status"] == "failed"
    # discarded 页面不能再接收 rollbackModelReload；初始 reload 请求是唯一脚本。
    assert len(page.scripts) == 1


def test_model_catalog_explicit_rescan_tracks_add_remove_and_rejects_traversal(
    tmp_path: Path,
) -> None:
    first = _write_model(tmp_path, "first")
    initial = Live2DModelCatalog.scan(tmp_path)
    second = _write_model(tmp_path, "second")
    expanded = Live2DModelCatalog.scan(tmp_path)
    first.unlink()
    reduced = Live2DModelCatalog.scan(tmp_path)

    assert len(initial.choices) == 1
    assert len(expanded.choices) == 2
    assert expanded.select("live2d/model/second/second.model3.json").available is True
    assert reduced.choices == ("live2d/model/second/second.model3.json",)
    assert reduced.select("../second.model3.json").available is False
    assert second.resolve().is_relative_to((tmp_path / "live2d" / "model").resolve())


def test_model_catalog_scan_rejects_descriptor_symlink_outside_model_root(
    tmp_path: Path,
) -> None:
    outside = _write_model(tmp_path / "detached", "outside")
    model_root = tmp_path / "live2d" / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "linked.model3.json").symlink_to(outside)

    assert Live2DModelCatalog.scan(tmp_path).choices == ()


def test_restart_receipt_rejects_pending_and_requires_real_web_evidence(tmp_path: Path) -> None:
    marker = tmp_path / "ready.json"
    marker.write_bytes(b"")
    with pytest.raises(ValueError, match="READY"):
        _write_restart_ready_receipt(
            marker,
            ready_status=RendererReadyStatus(
                backend="web_live2d",
                state=RendererReadyState.PENDING,
            ),
        )
    assert marker.read_bytes() == b""

    _write_restart_ready_receipt(marker, ready_status=_ready_web_status())
    assert _restart_ready_receipt_matches(
        marker,
        process_id=os.getpid(),
        expected_backend="web_live2d",
    )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    payload["renderer"]["alpha_nonempty"] = False
    marker.write_text(json.dumps(payload), encoding="utf-8")
    assert not _restart_ready_receipt_matches(
        marker,
        process_id=os.getpid(),
        expected_backend="web_live2d",
    )


def test_backend_ready_sequence_web_vulkan_web_uses_current_backend_evidence() -> None:
    web = SimpleNamespace(renderer=SimpleNamespace(request_ready_status=_ready_web_status))
    lifecycle = SimpleNamespace(
        available=True,
        actual_api="vulkan",
        state=SimpleNamespace(value="ready"),
    )
    vulkan = SimpleNamespace(
        renderer=None,
        lifecycle_status=lifecycle,
        frame_diagnostics=lambda: {
            "available": True,
            "visible_pixels": True,
        },
    )
    vulkan.renderer = vulkan

    assert _renderer_ready_status(web, "web_live2d").backend == "web_live2d"
    assert _renderer_ready_status(vulkan, "vulkan").public() == {
        "backend": "vulkan",
        "state": "ready",
        "ready": True,
        "terminal": True,
        "bridge_ready": False,
        "model_ready": True,
        "geometry_valid": True,
        "frame_visible": True,
        "alpha_nonempty": True,
        "actual_api": "vulkan",
        "reason_code": "",
        "generation": 0,
    }
    assert _renderer_ready_status(web, "web_live2d").backend == "web_live2d"


def test_sprite_and_opengl_ready_require_nonempty_captured_alpha(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.webp"
    frame_path.write_bytes(b"frame")
    sprite_renderer = SimpleNamespace(current_frame=frame_path)
    sprite_target = SimpleNamespace(
        renderer=sprite_renderer,
        grabFramebuffer=lambda: _FakeImage(alpha=255),
    )
    opengl_renderer = SimpleNamespace(model=object(), last_transform_valid=True)
    opengl_target = SimpleNamespace(
        renderer=opengl_renderer,
        grabFramebuffer=lambda: _FakeImage(alpha=255),
    )

    assert _renderer_ready_status(sprite_target, "sprite").ready is True
    assert _renderer_ready_status(opengl_target, "opengl").ready is True
    sprite_target.grabFramebuffer = lambda: _FakeImage(alpha=0)
    assert _renderer_ready_status(sprite_target, "sprite").ready is False
