from __future__ import annotations

from pathlib import Path

import pytest

from core.rendering.actions import (
    ExpressionRequest,
    ExpressionTimeline,
    MotionRequest,
)
from gui.renderers.sprite import SpriteRenderer
from gui.renderers.web_live2d import WebLive2DRenderer

RESOURCE_SPRITES = Path(__file__).resolve().parents[1] / "resources" / "sprites"


def test_expression_request_preserves_single_name_compatibility() -> None:
    request = ExpressionRequest.from_arguments({"name": "happy"})

    assert request.mode == "sequence"
    assert request.loop is False
    assert request.restore == "neutral"
    assert len(request.expressions) == 1
    assert request.expressions[0].name == "happy"


def test_expression_request_supports_sequence_parameters_and_duration() -> None:
    request = ExpressionRequest.from_arguments(
        {
            "mode": "sequence",
            "restore": "neutral",
            "expressions": [
                {
                    "name": "happy",
                    "duration_seconds": 0.5,
                    "transition_seconds": 0.2,
                    "parameters": {"ParamAngleX": 12.0},
                },
                {
                    "name": "curious",
                    "duration_seconds": 0.8,
                    "transition_seconds": 0.1,
                    "parameters": {"ParamAngleY": -6.0},
                },
            ],
        }
    )
    timeline = ExpressionTimeline()

    first = timeline.start(request, current="neutral")
    assert first.current == "happy"
    assert first.changed is True
    assert timeline.advance(0.25).progress == pytest.approx(1.0)
    second = timeline.advance(0.25)
    assert second.current == "curious"
    assert second.previous == "happy"
    assert second.changed is True
    restoring = timeline.advance(0.8)
    assert restoring.current == "neutral"
    assert restoring.finished is False
    restored = timeline.advance(0.1)
    assert restored.current == "neutral"
    assert restored.finished is True
    assert restored.parameters == {}
    assert timeline.active is False


def test_expression_request_blends_parameter_layers_by_weight() -> None:
    request = ExpressionRequest.from_arguments(
        {
            "mode": "blend",
            "expressions": [
                {"name": "happy", "weight": 0.75, "parameters": {"ParamAngleX": 20}},
                {"name": "shy", "weight": 0.25, "parameters": {"ParamAngleX": -4}},
            ],
        }
    )
    frame = ExpressionTimeline().start(request, current="neutral")

    assert frame.current == "happy"
    assert frame.parameters["ParamAngleX"] == pytest.approx(14.0)


def test_expression_timeline_fades_parameters_out_before_restore_finishes() -> None:
    request = ExpressionRequest.from_arguments(
        {
            "name": "happy",
            "duration_seconds": 0.1,
            "transition_seconds": 0.2,
            "parameters": {"ParamAngleX": 10},
        }
    )
    timeline = ExpressionTimeline()
    timeline.start(request, current="neutral")

    restoring = timeline.advance(0.1)
    assert restoring.current == "neutral"
    assert restoring.parameters["ParamAngleX"] == pytest.approx(10.0)
    halfway = timeline.advance(0.1)
    assert halfway.parameters["ParamAngleX"] == pytest.approx(5.0)
    finished = timeline.advance(0.1)
    assert finished.finished is True
    assert finished.parameters == {}


def test_motion_request_accepts_parameters_duration_and_transition() -> None:
    request = MotionRequest.from_arguments(
        {
            "name": "wave",
            "duration_seconds": 2.5,
            "transition_seconds": 0.3,
            "loop": True,
            "parameters": {"ParamBodyAngleX": 8},
        }
    )

    assert request.name == "wave"
    assert request.duration_seconds == 2.5
    assert request.transition_seconds == 0.3
    assert request.loop is True
    assert request.parameters == {"ParamBodyAngleX": 8.0}


@pytest.mark.parametrize(
    "arguments",
    [
        {"expressions": []},
        {"name": "happy", "duration_seconds": 0},
        {"name": "happy", "parameters": {"bad parameter": 1}},
        {"mode": "unknown", "name": "happy"},
        {"mode": "blend", "expressions": [{"name": "happy", "weight": 0}]},
    ],
)
def test_expression_request_rejects_invalid_boundaries(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ExpressionRequest.from_arguments(arguments)


@pytest.mark.parametrize(
    "arguments",
    (
        {},
        {"name": "happy", "expressions": [{"name": "neutral"}]},
        {"expressions": [{"name": "happy"}], "weight": 0.5},
        {"expressions": [{"name": "happy"}], "parameters": {"ParamAngleX": 1}},
    ),
)
def test_expression_request_requires_one_unambiguous_shape(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="exactly one|single-layer"):
        ExpressionRequest.from_arguments(arguments)


def test_web_renderer_dispatches_sequence_blend_and_single_expression_compatibility() -> None:
    renderer = WebLive2DRenderer("resources")
    renderer._view = object()
    renderer._page_ready = True
    renderer._model_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append  # type: ignore[method-assign]

    sequence = ExpressionRequest.from_arguments(
        {
            "mode": "sequence",
            "restore": "neutral",
            "expressions": [
                {
                    "name": "happy",
                    "duration_seconds": 0.2,
                    "transition_seconds": 0.1,
                    "parameters": {"ParamAngleX": 4},
                },
                {
                    "name": "curious",
                    "duration_seconds": 0.3,
                    "transition_seconds": 0.15,
                    "parameters": {"ParamAngleY": -2},
                },
            ],
        }
    )
    result = renderer.set_expression_request(sequence)

    assert result["status"] == "started"
    assert result["duration_seconds"] == pytest.approx(0.5)
    assert scripts and "setExpressionRequest" in scripts[-1]
    assert '"mode": "sequence"' in scripts[-1]
    assert '"ParamAngleX": 4.0' in scripts[-1]
    assert '"restore": "neutral"' in scripts[-1]

    blend = ExpressionRequest.from_arguments(
        {
            "mode": "blend",
            "expressions": [
                {"name": "happy", "weight": 0.75, "parameters": {"ParamCheek": 1}},
                {"name": "shy", "weight": 0.25, "parameters": {"ParamCheek": 0}},
            ],
        }
    )
    result = renderer.set_expression_request(blend)
    assert result["status"] == "started"
    assert result["duration_seconds"] == pytest.approx(1.5)
    assert '"mode": "blend"' in scripts[-1]
    assert '"weight": 0.75' in scripts[-1]

    assert renderer.set_expression("sad")
    assert "setExpression(" in scripts[-1]
    assert "setExpressionRequest" not in scripts[-1]
    assert renderer._expression_timeline.active is False


def test_web_renderer_dispatches_motion_parameters_duration_transition_and_loop() -> None:
    renderer = WebLive2DRenderer("resources")
    renderer._view = object()
    renderer._page_ready = True
    renderer._model_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append  # type: ignore[method-assign]
    request = MotionRequest.from_arguments(
        {
            "name": "wave",
            "duration_seconds": 0.6,
            "transition_seconds": 0.2,
            "loop": True,
            "parameters": {"ParamBodyAngleX": 3},
        }
    )

    result = renderer.play_motion_request(request)

    assert result == {
        "status": "started",
        "name": "wave",
        "duration_seconds": 0.6,
        "transition_seconds": 0.2,
        "loop": True,
        "parameters_applied": True,
    }
    assert scripts and "playMotionRequest" in scripts[-1]
    assert '"duration_seconds": 0.6' in scripts[-1]
    assert '"transition_seconds": 0.2' in scripts[-1]
    assert '"loop": true' in scripts[-1]
    assert '"ParamBodyAngleX": 3.0' in scripts[-1]


def test_web_renderer_replays_pending_structured_requests_and_cleans_shutdown() -> None:
    renderer = WebLive2DRenderer("resources")
    renderer._view = object()
    renderer._page_ready = True
    scripts: list[str] = []
    renderer._run_javascript = scripts.append  # type: ignore[method-assign]
    expression = ExpressionRequest.from_arguments({"name": "happy"})
    motion = MotionRequest.from_arguments({"name": "wave", "loop": True})

    assert renderer.set_expression_request(expression)["status"] == "pending"
    assert renderer.play_motion_request(motion)["status"] == "pending"
    assert scripts == []

    renderer._on_model_ready()
    assert "setExpressionRequest" in scripts[0]
    assert "playMotionRequest" in scripts[1]
    renderer.shutdown()
    assert renderer._pending_expression_request is None
    assert renderer._pending_motion_request is None
    assert renderer._expression_timeline.active is False
    assert renderer._motion_request is None


def test_web_renderer_html_executes_timeline_after_internal_model_update() -> None:
    html = WebLive2DRenderer("resources")._html_document()

    assert "setExpressionRequest: function(value)" in html
    assert "playMotionRequest: function(value)" in html
    assert "function advanceExpressionTimeline(seconds)" in html
    assert "function advanceMotionRequest(seconds)" in html
    assert "configureExpressionParameterStage" in html
    assert "motionRequestDuration" in html
    assert html.index("liveModel.internalModel.update(dt * 1000") < html.index(
        "advanceExpressionTimeline(dt)"
    )


def test_sprite_expression_request_crossfades_into_expression_and_restore() -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)
    request = ExpressionRequest.from_arguments(
        {
            "name": "happy",
            "duration_seconds": 0.2,
            "transition_seconds": 0.1,
            "restore": "neutral",
        }
    )

    assert renderer.set_expression_request(request)["status"] == "started"
    assert renderer.previous_frame is not None
    assert renderer.previous_frame.name == "mea01A_001.webp"
    assert renderer.current_frame.name == "mea01A_182.webp"
    assert renderer.expression_transition_progress == 0.0

    renderer.advance(0.1)
    assert renderer.previous_frame is None
    assert renderer.current_frame.name == "mea01A_182.webp"
    renderer.advance(0.1)
    assert renderer.previous_frame is not None
    assert renderer.previous_frame.name == "mea01A_182.webp"
    assert renderer.current_frame.name == "mea01A_001.webp"
    assert renderer.expression_transition_progress == 0.0
    renderer.advance(0.05)
    assert renderer.expression_transition_progress == pytest.approx(0.5)
    renderer.advance(0.05)
    assert renderer.previous_frame is None
    assert renderer.current_frame.name == "mea01A_001.webp"


def test_sprite_motion_request_loops_using_natural_frame_duration() -> None:
    renderer = SpriteRenderer(RESOURCE_SPRITES)
    request = MotionRequest.from_arguments({"name": "blink", "loop": True})

    assert renderer.play_motion_request(request)["status"] == "started"
    renderer.advance(renderer.frame_interval * 2)

    assert renderer.state.motion == "blink"
    assert renderer.current_frame.name == "mea01A_011.webp"
    assert renderer._motion_request is request
