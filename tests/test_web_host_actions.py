from __future__ import annotations

from core.rendering.actions import ExpressionRequest, MotionRequest
from gui.qt6.web_host import WebPetHost


def test_web_host_forwards_structured_visual_action_requests() -> None:
    calls: list[object] = []

    class Renderer:
        def set_expression_request(self, request: ExpressionRequest):
            calls.append(request)
            return {"status": "started", "kind": "expression"}

        def play_motion_request(self, request: MotionRequest):
            calls.append(request)
            return {"status": "started", "kind": "motion"}

    host = WebPetHost(Renderer())
    expression = ExpressionRequest.from_arguments(
        {
            "expressions": [
                {"name": "happy", "duration_seconds": 0.4},
                {"name": "curious", "duration_seconds": 0.6},
            ]
        }
    )
    motion = MotionRequest.from_arguments(
        {"name": "wave", "duration_seconds": 1.2, "transition_seconds": 0.15}
    )

    assert host.set_expression_request(expression)["kind"] == "expression"
    assert host.play_motion_request(motion)["kind"] == "motion"
    assert calls == [expression, motion]


def test_web_host_rejects_invalid_or_unsupported_structured_actions() -> None:
    host = WebPetHost(object())

    assert host.set_expression_request(object())["status"] == "unavailable"
    assert host.play_motion_request(object())["status"] == "unavailable"
    assert (
        host.set_expression_request(ExpressionRequest.from_arguments({"name": "happy"}))["status"]
        == "unavailable"
    )
    assert (
        host.play_motion_request(MotionRequest.from_arguments({"name": "wave"}))["status"]
        == "unavailable"
    )
