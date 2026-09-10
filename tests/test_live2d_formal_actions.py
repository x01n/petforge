"""Cubism exp3/motion3 文件契约和当前资源能力边界。"""

from __future__ import annotations

import json
from pathlib import Path

from core.rendering.resources import (
    validate_expression3_description,
    validate_motion3_description,
)
from gui.renderers.web_live2d import WebLive2DRenderer, probe_web_live2d


def _valid_expression() -> dict[str, object]:
    return {
        "Version": 3,
        "Parameters": [{"Id": "ParamEyeLOpen", "Value": 0.0, "Blend": "Overwrite"}],
    }


def _valid_motion() -> dict[str, object]:
    return {
        "Version": 3,
        "Meta": {
            "Duration": 1.0,
            "Fps": 30.0,
            "Loop": False,
            "CurveCount": 1,
            "TotalSegmentCount": 1,
            "TotalPointCount": 2,
            "UserDataCount": 0,
            "TotalUserDataSize": 0,
        },
        "Curves": [
            {
                "Target": "Parameter",
                "Id": "ParamAngleX",
                "Segments": [0.0, 0.0, 0, 1.0, 10.0],
            }
        ],
    }


def test_formal_action_validators_reject_placeholder_json(tmp_path: Path) -> None:
    expression = tmp_path / "empty.exp3.json"
    motion = tmp_path / "empty.motion3.json"
    huge = tmp_path / "huge.exp3.json"
    expression.write_text("{}", encoding="utf-8")
    motion.write_text("{}", encoding="utf-8")
    huge.write_text(
        json.dumps({"Parameters": [{"Id": "ParamAngleX", "Value": 10**1000}]}),
        encoding="utf-8",
    )

    assert not validate_expression3_description(expression)
    assert not validate_motion3_description(motion)
    assert not validate_expression3_description(huge)


def test_formal_action_validators_accept_cubism_runtime_shapes(tmp_path: Path) -> None:
    expression = tmp_path / "happy.exp3.json"
    motion = tmp_path / "wave.motion3.json"
    expression.write_text(json.dumps(_valid_expression()), encoding="utf-8")
    motion.write_text(json.dumps(_valid_motion()), encoding="utf-8")

    assert validate_expression3_description(expression)
    assert validate_motion3_description(motion)


def test_current_runtime_descriptor_exposes_valid_formal_motion_files() -> None:
    root = Path(__file__).resolve().parents[1]
    descriptor_path = root / "resources/live2d/model/mea_live2d/橙色猫猫.model3.json"
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    references = descriptor["FileReferences"]

    assert not references.get("Expressions")
    assert tuple(references["Motions"]) == ("A", "B")
    assert not list(descriptor_path.parent.glob("*.exp3.json"))
    motion_files = {
        group: descriptor_path.parent / entries[0]["File"]
        for group, entries in references["Motions"].items()
    }
    assert all(validate_motion3_description(path) for path in motion_files.values())
    probe = probe_web_live2d(root / "resources")
    assert probe.model_path == descriptor_path
    assert probe.motions == ("A", "B")
    assert probe.motion_aliases == (("angry", "B"), ("blink", "A"))
    assert probe.motion_durations == (("A", 6.1), ("B", 6.1))

    renderer = WebLive2DRenderer(root / "resources")
    renderer._view = object()
    renderer._model_ready = True
    assert {"idle", "wave", "walk"}.issubset(renderer.capabilities.motions)
    assert renderer.supports_motion("blink") is True
    assert renderer.supports_motion("wave") is True
    assert renderer.supports_motion("spin") is False
    html = renderer._html_document()
    assert 'let motionDurations = new Map(Object.entries({"A": 6.1, "B": 6.1}));' in html
    assert "stopFormalMotionOnce" in html
    assert "motionFinished" in html


def test_web_formal_motion_completion_updates_public_state() -> None:
    """页面正式动作收尾后，Python 状态卡也必须回到 idle。"""

    root = Path(__file__).resolve().parents[1]
    renderer = WebLive2DRenderer(root / "resources")
    renderer._view = object()
    renderer._model_ready = True

    assert renderer.play_motion("blink")["status"] == "available"
    assert renderer.state.motion == "blink"
    assert renderer._formal_motion_name == "A"

    renderer._on_formal_motion_finished("B")
    assert renderer.state.motion == "blink"
    assert renderer._formal_motion_name == "A"

    renderer._on_formal_motion_finished("A")
    assert renderer.state.motion == "idle"
    assert renderer.state.elapsed == 0.0
    assert renderer._formal_motion_name is None


def test_probe_keeps_invalid_declared_actions_on_procedural_fallback(tmp_path: Path) -> None:
    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True)
    for name in ("pixi.min.js", "pixi-live2d-display.min.js", "live2dcubismcore.min.js"):
        (javascript_root / name).write_text("/* test */", encoding="utf-8")
    model_root = tmp_path / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    (model_root / "demo.moc3").write_bytes(b"MOC3-runtime-test")
    (model_root / "texture.png").write_bytes(b"texture")
    (model_root / "empty.exp3.json").write_text("{}", encoding="utf-8")
    (model_root / "empty.motion3.json").write_text("{}", encoding="utf-8")
    (model_root / "demo.model3.json").write_text(
        json.dumps(
            {
                "Version": 3,
                "FileReferences": {
                    "Moc": "demo.moc3",
                    "Textures": ["texture.png"],
                    "Expressions": [{"Name": "bad", "File": "empty.exp3.json"}],
                    "Motions": {"Bad": [{"File": "empty.motion3.json"}]},
                },
            }
        ),
        encoding="utf-8",
    )
    modules = {
        "PySide6.QtCore": object(),
        "PySide6.QtWebEngineWidgets": object(),
        "PySide6.QtWebChannel": object(),
    }

    probe = probe_web_live2d(tmp_path, importer=lambda name: modules[name])

    assert probe.expressions == ()
    assert probe.motions == ()
    assert probe.as_capabilities().expressions
    assert probe.as_capabilities().motions


def test_web_renderer_rejects_unknown_action_after_formal_model_is_ready(tmp_path: Path) -> None:
    """正式动作清单就绪后，未知名称不能再回报 pending/available。"""

    javascript_root = tmp_path / "live2d" / "js"
    javascript_root.mkdir(parents=True)
    for name in ("pixi.min.js", "pixi-live2d-display.min.js", "live2dcubismcore.min.js"):
        (javascript_root / name).write_text("/* test */", encoding="utf-8")
    model_root = tmp_path / "live2d" / "model" / "demo"
    model_root.mkdir(parents=True)
    (model_root / "demo.moc3").write_bytes(b"MOC3-runtime-test")
    (model_root / "texture.png").write_bytes(b"texture")
    (model_root / "happy.exp3.json").write_text(json.dumps(_valid_expression()), encoding="utf-8")
    (model_root / "tap.motion3.json").write_text(json.dumps(_valid_motion()), encoding="utf-8")
    (model_root / "demo.model3.json").write_text(
        json.dumps(
            {
                "Version": 3,
                "FileReferences": {
                    "Moc": "demo.moc3",
                    "Textures": ["texture.png"],
                    "Expressions": [{"Name": "happy_real", "File": "happy.exp3.json"}],
                    "Motions": {"Tap": [{"File": "tap.motion3.json"}]},
                },
            }
        ),
        encoding="utf-8",
    )
    modules = {
        "PySide6.QtCore": object(),
        "PySide6.QtWebEngineWidgets": object(),
        "PySide6.QtWebChannel": object(),
    }
    renderer = WebLive2DRenderer(tmp_path, importer=lambda name: modules[name])
    renderer._view = object()
    renderer._model_ready = True
    renderer._replay_pending_model_state = lambda: None  # type: ignore[method-assign]

    assert renderer.set_expression("happy_real")["status"] == "available"
    assert renderer.play_motion("Tap")["status"] == "available"
    assert renderer.set_expression("unknown-expression") is False
    assert renderer.play_motion("unknown") is False
