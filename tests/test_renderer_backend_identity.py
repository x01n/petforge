from __future__ import annotations

import types
from pathlib import Path

from gui.renderers.live2d import Live2DRenderer


def test_native_live2d_reports_public_opengl_backend(tmp_path: Path, monkeypatch) -> None:
    """原生 Live2D 的能力标识必须与用户可选的 OpenGL 后端一致。"""

    model_dir = tmp_path / "live2d" / "model" / "demo"
    model_dir.mkdir(parents=True)
    (model_dir / "demo.moc3").write_bytes(b"moc3")
    (model_dir / "texture.png").write_bytes(b"texture")
    (model_dir / "demo.model3.json").write_text(
        '{"FileReferences":{"Moc":"demo.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )

    class FakeModel:
        def LoadModelJson(self, _path: str) -> None:
            return None

    fake_module = types.SimpleNamespace(
        init=lambda: None,
        glInit=lambda: None,
        clearBuffer=lambda: None,
        dispose=lambda: None,
        LAppModel=FakeModel,
    )
    monkeypatch.setattr(
        "gui.renderers.live2d.importlib.import_module",
        lambda _name: fake_module,
    )

    renderer = Live2DRenderer(model_dir)

    assert renderer.capabilities.backend == "opengl"
    assert renderer.capabilities.available is True


def test_native_live2d_resource_reload_requires_restart_and_honors_drag_busy(
    tmp_path: Path,
) -> None:
    renderer = Live2DRenderer(tmp_path)
    renderer.set_dragging(True)
    assert renderer.reload_resources(tmp_path)["status"] == "busy"

    renderer.set_dragging(False)
    result = renderer.reload_resources(tmp_path, model_path="demo.model3.json")
    assert result == {
        "status": "restart_required",
        "reason": "native Live2D resources are owned by the current OpenGL context",
    }
