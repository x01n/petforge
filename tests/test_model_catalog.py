from __future__ import annotations

import json
from pathlib import Path

from gui.renderers.assets import RuntimeCapability, RuntimeCapabilityState, select_renderer
from gui.renderers.model_catalog import Live2DModelCatalog
from gui.renderers.web_live2d import WebLive2DRenderer, probe_web_live2d


class _ReloadPage:
    def __init__(self) -> None:
        self.callback = None

    def runJavaScript(self, _script: str, callback=None) -> None:  # noqa: N802
        self.callback = callback


class _ReloadView:
    def __init__(self, page: _ReloadPage) -> None:
        self._page = page

    def page(self) -> _ReloadPage:
        return self._page


def _complete_reload(renderer: WebLive2DRenderer, page: _ReloadPage, request_id: int) -> None:
    renderer._on_model_reload_finished(json.dumps({"request_id": request_id, "status": "prepared"}))
    renderer._on_model_reload_finished(
        json.dumps({"request_id": request_id, "status": "available"})
    )
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


def _write_model(root: Path, name: str, *, valid: bool = True) -> Path:
    directory = root / "live2d" / "model" / name
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = directory / f"{name}.model3.json"
    if valid:
        (directory / f"{name}.moc3").write_bytes(b"moc")
        (directory / "texture.png").write_bytes(b"texture")
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
    else:
        descriptor.write_text('{"FileReferences":{"Moc":"../outside.moc3"}}', encoding="utf-8")
    return descriptor


def test_model_catalog_scans_valid_model3_and_exposes_exact_keys(tmp_path: Path) -> None:
    _write_model(tmp_path, "zeta")
    _write_model(tmp_path, "alpha")
    _write_model(tmp_path, "broken", valid=False)

    catalog = Live2DModelCatalog.scan(tmp_path)

    assert catalog.choices == (
        "live2d/model/alpha/alpha.model3.json",
        "live2d/model/zeta/zeta.model3.json",
    )
    selected = catalog.select("")
    assert selected.available
    assert selected.selected is not None
    assert selected.selected.key == catalog.choices[0]
    assert (
        catalog.select(catalog.choices[1]).path
        == (tmp_path / "live2d/model/zeta/zeta.model3.json").resolve()
    )


def test_model_catalog_rejects_path_traversal_and_unknown_keys(tmp_path: Path) -> None:
    _write_model(tmp_path, "demo")
    catalog = Live2DModelCatalog.scan(tmp_path)

    outside = catalog.select("live2d/model/demo/../../outside.model3.json")
    assert not outside.available
    assert "outside" in outside.reason
    assert not catalog.select("/tmp/demo.model3.json").available
    assert not catalog.select("live2d\\model\\demo\\demo.model3.json").available
    assert not catalog.select("live2d/model/missing/missing.model3.json").available


def test_renderer_selection_uses_requested_model_and_rejects_invalid_model(tmp_path: Path) -> None:
    first = _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")
    runtime = RuntimeCapability(
        "live2d_runtime",
        RuntimeCapabilityState.AVAILABLE,
        "test runtime",
    )

    selected = select_renderer(
        tmp_path,
        requested_backend="opengl",
        requested_model="live2d/model/second/second.model3.json",
        runtime=runtime,
    )
    assert selected.available
    assert selected.model_path == second.resolve()
    assert selected.model_choices == (
        "live2d/model/first/first.model3.json",
        "live2d/model/second/second.model3.json",
    )

    invalid = select_renderer(
        tmp_path,
        requested_backend="opengl",
        requested_model="live2d/model/first/../../outside.model3.json",
        runtime=runtime,
    )
    assert not invalid.available
    assert "outside" in invalid.reason
    assert first.exists()


def test_web_probe_honors_explicit_model_path(tmp_path: Path) -> None:
    _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")

    probe = probe_web_live2d(tmp_path, model_path=second)

    assert probe.model_path == second.resolve()


def test_web_renderer_model_reload_dispatches_atomic_page_protocol(tmp_path: Path) -> None:
    """候选模型切换只返回 pending，回执成功后才提交 Python probe。"""

    first = _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True, exist_ok=True)
    for name in ("pixi.min.js", "live2dcubismcore.min.js", "pixi-live2d-display.min.js"):
        (javascript_root / name).write_text("// test asset", encoding="utf-8")

    renderer = WebLive2DRenderer(
        tmp_path,
        model_path=first,
        importer=lambda _name: object(),
    )
    old_probe = renderer._probe
    page = _ReloadPage()
    renderer._view = _ReloadView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append  # type: ignore[method-assign]

    result = renderer.reload_model(second)

    assert result["status"] == "pending"
    assert isinstance(result["request_id"], int)
    assert result["model"] == second.resolve()
    assert renderer._probe is old_probe
    assert renderer.capabilities.model_path == first.resolve()
    assert scripts and "reloadModel" in scripts[-1]
    request_id = result["request_id"]
    renderer._on_model_reload_finished(
        json.dumps({"request_id": request_id, "status": "failed", "reason": "test"})
    )
    assert renderer._probe is old_probe
    assert renderer.model_reload_status["status"] == "failed"

    success = renderer.reload_model(second)
    assert success["status"] == "pending"
    _complete_reload(renderer, page, int(success["request_id"]))
    assert renderer._probe.model_path == second.resolve()
    assert renderer.capabilities.model_path == second.resolve()
    assert renderer.model_reload_status["status"] == "available"

    invalid = renderer.reload_model("live2d/model/missing/missing.model3.json")
    assert invalid["status"] == "unavailable"
    assert renderer._probe.model_path == second.resolve()


def test_web_renderer_model_reload_rejects_stale_and_concurrent_receipts(tmp_path: Path) -> None:
    """旧代次和并发请求不能覆盖当前模型切换状态。"""

    first = _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")
    third = _write_model(tmp_path, "third")
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True, exist_ok=True)
    for name in ("pixi.min.js", "live2dcubismcore.min.js", "pixi-live2d-display.min.js"):
        (javascript_root / name).write_text("// test asset", encoding="utf-8")
    renderer = WebLive2DRenderer(tmp_path, model_path=first, importer=lambda _name: object())
    page = _ReloadPage()
    renderer._view = _ReloadView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True
    renderer._run_javascript = lambda _script: None  # type: ignore[method-assign]

    pending = renderer.reload_model(second)
    assert pending["status"] == "pending"
    busy = renderer.reload_model(third)
    assert busy["status"] == "busy"
    renderer._on_model_reload_finished(
        json.dumps({"request_id": int(pending["request_id"]) + 1, "status": "available"})
    )
    assert renderer._pending_model_reload is not None
    assert renderer._probe.model_path == first.resolve()
    _complete_reload(renderer, page, int(pending["request_id"]))
    assert renderer._pending_model_reload is None
    assert renderer._probe.model_path == second.resolve()


def test_web_renderer_resource_reload_reuses_same_root_and_rejects_root_change(
    tmp_path: Path,
) -> None:
    first = _write_model(tmp_path, "first")
    renderer = WebLive2DRenderer(tmp_path, model_path=first, importer=lambda _name: object())
    calls: list[object] = []
    renderer.reload_model = lambda path: (
        calls.append(path)
        or {  # type: ignore[method-assign]
            "status": "pending",
            "request_id": 1,
        }
    )

    same_root = renderer.reload_resources(tmp_path, model_path=first, sprite_scale=0.8)

    assert same_root == {
        "status": "pending",
        "request_id": 1,
        "sprite_scale_applied": False,
    }
    assert calls == [first]

    different_root = renderer.reload_resources(tmp_path / "other", model_path=first)
    assert different_root["status"] == "restart_required"
    assert renderer.model_reload_status["status"] == "restart_required"
    assert calls == [first]

    renderer.set_dragging(True)
    busy = renderer.reload_resources(tmp_path, model_path=first)
    assert busy["status"] == "busy"


def test_web_model_loader_destroys_late_timeout_result() -> None:
    """模型加载超时后，迟到的 Live2D 实例必须被主动销毁。"""

    renderer = WebLive2DRenderer("resources")
    html = renderer._html_document()
    start = html.index("async function loadModelResource(url)")
    end = html.index("async function loadModel()", start)
    loader = html[start:end]

    assert "let timedOut = false;" in loader
    assert "const observedModelPromise = modelPromise.then(" in loader
    assert "if (timedOut) destroyCandidateModel(model);" in loader
    assert "if (timedOut) return null;" in loader


def test_web_renderer_failed_reload_restores_public_model_metadata(tmp_path: Path) -> None:
    """候选预检失败后，页面公开 URL 也必须仍指向旧模型。"""

    renderer = WebLive2DRenderer(tmp_path)
    html = renderer._html_document()
    restore_start = html.index("function restoreRuntimeObject(snapshot)")
    restore_end = html.index("function captureCandidateBindings()", restore_start)
    restore = html[restore_start:restore_end]
    assert "window.meapetLive2D.modelUrl = modelUrl;" in restore
    assert "window.meapetLive2D.partNames = modelPartNames;" in restore


def test_web_renderer_shutdown_completes_pending_reload_callback(tmp_path: Path) -> None:
    """关闭页面时不能让模型切换调用方永久等待。"""

    renderer = WebLive2DRenderer(tmp_path)
    callbacks: list[dict[str, object]] = []
    renderer.set_model_reload_callback(lambda value: callbacks.append(dict(value)))
    renderer._pending_model_reload = (7, renderer._probe)
    renderer.shutdown()
    assert callbacks == [
        {
            "status": "failed",
            "request_id": 7,
            "model": renderer._probe.model_path,
            "reason": "Web Live2D renderer was closed during model reload",
        }
    ]


def test_web_renderer_reload_dispatch_failure_is_reported_immediately(tmp_path: Path) -> None:
    first = _write_model(tmp_path, "first")
    second = _write_model(tmp_path, "second")
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True, exist_ok=True)
    for name in ("pixi.min.js", "live2dcubismcore.min.js", "pixi-live2d-display.min.js"):
        (javascript_root / name).write_text("// test asset", encoding="utf-8")
    renderer = WebLive2DRenderer(tmp_path, model_path=first, importer=lambda _name: object())
    renderer._view = object()
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True
    renderer._run_javascript = lambda _script: False  # type: ignore[method-assign]

    result = renderer.reload_model(second)

    assert result["status"] == "failed"
    assert renderer._pending_model_reload is None
    assert renderer.model_reload_status["status"] == "failed"


def test_web_model_loader_caps_resize_edge(tmp_path: Path) -> None:
    renderer = WebLive2DRenderer(tmp_path)
    html = renderer._html_document()
    start = html.index("function resize(requestedWidth, requestedHeight, force = false)")
    end = html.index("function applyModelMetadata", start)
    resize = html[start:end]

    assert "const maxViewportEdge = 16384;" in resize
    assert "Math.min(maxViewportEdge, Math.max(" in resize
