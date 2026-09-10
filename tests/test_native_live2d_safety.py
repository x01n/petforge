from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.rendering.actions import ExpressionRequest, MotionRequest
from gui.renderers.live2d import Live2DRenderer


class _Parameter:
    def __init__(self, minimum: float, maximum: float, default: float = 0.0) -> None:
        self.min = minimum
        self.max = maximum
        self.default = default


class _BoundedModel:
    def __init__(self) -> None:
        self.values: list[tuple[str, float, float]] = []
        self.updated = 0

    def GetParamIds(self):  # noqa: N802
        return ["ParamAngleX", "ParamAngleY"]

    def GetParameter(self, index):  # noqa: N802
        return (_Parameter(-2.0, 2.0), _Parameter(-1.0, 3.0, 1.0))[index]

    def Update(self):  # noqa: N802
        self.updated += 1

    def SetParameterValue(self, name, value, weight):  # noqa: N802
        self.values.append((name, value, weight))


def _renderer_with(model: object) -> Live2DRenderer:
    renderer = Live2DRenderer("resources")
    renderer._model = model
    return renderer


def test_native_initialize_is_idempotent_for_an_existing_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "demo.moc3").write_bytes(b"moc3")
    (model_dir / "texture.png").write_bytes(b"texture")
    descriptor = model_dir / "demo.model3.json"
    descriptor.write_text(
        '{"FileReferences":{"Moc":"demo.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )

    init_calls: list[str] = []
    gl_init_calls: list[str] = []
    load_calls: list[str] = []

    class FakeModel:
        def LoadModelJson(self, path: str) -> None:  # noqa: N802
            load_calls.append(path)

    fake_module = SimpleNamespace(
        init=lambda: init_calls.append("init"),
        glInit=lambda: gl_init_calls.append("glInit"),
        clearBuffer=lambda: None,
        dispose=lambda: None,
        LAppModel=FakeModel,
    )
    monkeypatch.setattr(
        "gui.renderers.live2d.importlib.import_module",
        lambda _name: fake_module,
    )

    renderer = Live2DRenderer(model_dir, model_path=descriptor)
    assert renderer.initialize() is True
    assert renderer.initialize() is True
    assert init_calls == ["init"]
    assert gl_init_calls == ["glInit"]
    assert load_calls == [str(descriptor)]


def test_native_initialize_failure_can_be_retried_without_reprobing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "demo.moc3").write_bytes(b"moc3")
    (model_dir / "texture.png").write_bytes(b"texture")
    descriptor = model_dir / "demo.model3.json"
    descriptor.write_text(
        '{"FileReferences":{"Moc":"demo.moc3","Textures":["texture.png"]}}',
        encoding="utf-8",
    )
    attempts = 0

    class FlakyModel:
        def LoadModelJson(self, _path: str) -> None:  # noqa: N802
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary loader failure")

    fake_module = SimpleNamespace(
        init=lambda: None,
        glInit=lambda: None,
        clearBuffer=lambda: None,
        dispose=lambda: None,
        LAppModel=FlakyModel,
    )
    monkeypatch.setattr(
        "gui.renderers.live2d.importlib.import_module",
        lambda _name: fake_module,
    )

    renderer = Live2DRenderer(model_dir, model_path=descriptor)
    assert renderer.initialize() is False
    assert renderer.capabilities.available is False
    assert renderer.initialize() is True
    assert renderer.capabilities.available is True
    assert attempts == 2


def test_native_resize_propagates_model_rejection_and_keeps_previous_viewport() -> None:
    class RejectingModel(_BoundedModel):
        def Resize(self, width, height):  # noqa: N802
            return False

    renderer = _renderer_with(RejectingModel())
    assert renderer.viewport_size == (1, 1)
    assert renderer.resize(420, 480, 2.0) is False
    assert renderer.viewport_size == (1, 1)


def test_native_tracking_rejects_non_finite_input_and_stays_finite() -> None:
    model = _BoundedModel()
    renderer = _renderer_with(model)

    assert renderer.set_cursor_target(float("nan"), 10, 420, 480) is False
    assert renderer.set_cursor_target(10, float("inf"), 420, 480) is False
    assert renderer.set_cursor_target(10**20, 10**20, 420, 480) is True
    renderer.advance(float("nan"))
    renderer.advance(float("inf"))

    assert model.updated == 2
    assert model.values
    assert all(math.isfinite(value) for _name, value, _weight in model.values)
    assert all(abs(value) <= 2.0 for _name, value, _weight in model.values)


def test_native_tracking_respects_model_limits_with_safety_margin() -> None:
    model = _BoundedModel()
    renderer = _renderer_with(model)
    assert renderer.set_cursor_target(0, 0, 420, 480)
    for _ in range(120):
        renderer.advance(1 / 60)

    values = {name: value for name, value, _weight in model.values[-2:]}
    # ParamAngleX 的安全范围是 [-1.56, 1.56]；ParamAngleY 的模型默认值
    # 为 1，安全范围是 [-0.56, 2.56]，且注视预算仍会继续收窄。
    assert -1.56 <= values["ParamAngleX"] <= 1.56
    assert -0.56 <= values["ParamAngleY"] <= 2.56


def test_native_window_move_returns_tracking_target_to_neutral() -> None:
    renderer = _renderer_with(_BoundedModel())
    assert renderer.set_cursor_target(0, 0, 420, 480)
    renderer.advance(1 / 60)
    renderer.notify_window_moved()

    assert renderer._cursor_velocity == (0.0, 0.0)
    assert renderer._cursor_goal == (0.0, 0.0)
    for _ in range(30):
        renderer.advance(1 / 60)
    assert all(abs(value) < 0.2 for value in renderer._cursor_current)


def test_native_turn_retries_direction_if_runtime_rejects_first_commit() -> None:
    directions: list[str] = []

    class Model(_BoundedModel):
        def __init__(self) -> None:
            super().__init__()
            self.rejections = 1

        def SetDirection(self, value):  # noqa: N802
            if self.rejections:
                self.rejections -= 1
                raise RuntimeError("transient direction failure")
            directions.append(value)

    model = Model()
    renderer = _renderer_with(model)
    assert renderer.set_direction("B") is True
    for _ in range(80):
        renderer.advance(1 / 60)

    assert directions == ["B"]
    assert renderer._pending_direction is None
    assert renderer._facing_sign == -1


def test_native_invalid_mvp_is_fail_closed_before_draw() -> None:
    class Model(_BoundedModel):
        def __init__(self) -> None:
            super().__init__()
            self.drawn = 0
            self.invalid = True

        def GetMvp(self):  # noqa: N802
            return [float("nan")] * 16 if self.invalid else [1.0] * 16

        def Draw(self):  # noqa: N802
            self.drawn += 1

    model = Model()
    renderer = _renderer_with(model)
    with pytest.raises(RuntimeError, match="MVP transform"):
        renderer.draw()
    assert model.drawn == 0
    assert renderer.last_transform_valid is False

    model.invalid = False
    renderer.draw()
    assert model.drawn == 1
    assert renderer.last_transform_valid is True


def test_native_resize_rejects_zero_and_keeps_logical_aspect_contract() -> None:
    model = _BoundedModel()
    model.resize_calls = []

    def resize(width, height):
        model.resize_calls.append((width, height))

    model.Resize = resize
    renderer = _renderer_with(model)
    assert renderer.resize(0, 480) is False
    assert renderer.viewport_size == (1, 1)
    assert renderer.resize(420, 480, float("nan")) is True
    assert renderer.viewport_size == (420, 480)
    assert model.resize_calls == [(420, 480)]


def test_native_pcm_viseme_alias_updates_bounded_mouth_parameters() -> None:
    class MouthModel(_BoundedModel):
        def GetParamIds(self):  # noqa: N802
            return ["ParamMouthForm", "ParamMouthOpenY"]

        def GetParameter(self, index):  # noqa: N802
            return (_Parameter(-1.0, 1.0), _Parameter(0.0, 1.0))[index]

    model = MouthModel()
    renderer = _renderer_with(model)
    renderer._refresh_parameter_limits(model)
    assert renderer.set_mouth_viseme("Neutral", intensity=0.5) is True
    assert renderer.set_mouth_viseme("Open", intensity=1.0) is True
    mouth_values = {
        name: value
        for name, value, _weight in model.values
        if name in {"ParamMouthForm", "ParamMouthOpenY"}
    }
    assert mouth_values["ParamMouthForm"] == pytest.approx(0.78)
    assert mouth_values["ParamMouthOpenY"] == pytest.approx(0.78)
    assert renderer.set_mouth_viseme("unsupported", intensity=1.0) is False


def test_native_action_contract_accepts_procedural_fallback_and_rejects_unknown() -> None:
    class ActionModel(_BoundedModel):
        def __init__(self) -> None:
            super().__init__()
            self.expression_calls: list[str] = []
            self.motion_calls: list[tuple[str, int, int]] = []
            self.stop_calls = 0

        def StartMotion(self, name, group, priority):  # noqa: N802
            self.motion_calls.append((name, group, priority))

        def StopAllMotions(self):  # noqa: N802
            self.stop_calls += 1

    class Module:
        class MotionPriority:
            NORMAL = 2

    renderer = _renderer_with(ActionModel())
    renderer._module = Module()
    model = renderer.model
    assert model is not None

    # 当前资源没有 exp3，语义表情走已绑定参数的有限回退；未知参数名
    # 不能冒充成功。配置中的 angry 别名应解析到正式 B 动作组。
    assert renderer.supports_expression("happy") is True
    assert renderer.set_expression("happy") is True
    assert renderer.supports_expression("ParamAngleX") is False
    assert renderer.set_expression("ParamAngleX") is False
    assert renderer.supports_motion("angry") is True
    assert renderer.play_motion("angry") is True
    assert model.motion_calls == [("B", 0, 2)]
    renderer._motion_durations = {"B": 0.1}
    renderer.advance(0.05)
    renderer.advance(0.05)
    assert model.stop_calls == 1
    assert renderer.state.motion == "idle"
    assert renderer.play_motion("unknown") is False


def test_native_formal_expression_clears_previous_procedural_pose() -> None:
    """切换到正式 exp3 后不能继续叠加旧的程序化角度。"""

    class FormalModel(_BoundedModel):
        def __init__(self) -> None:
            super().__init__()
            self.expression_calls: list[str] = []

        def SetExpression(self, name):  # noqa: N802
            self.expression_calls.append(str(name))

    renderer = _renderer_with(FormalModel())
    renderer._declared_expressions = ("formal",)
    renderer._procedural_expression = "happy"
    renderer._procedural_expression_active = True

    assert renderer.set_expression("formal") is True
    assert renderer._procedural_expression_active is False
    renderer.set_cursor_target(210, 240, 420, 480)
    renderer.advance(1 / 60)

    angle_values = [
        value
        for name, value, _weight in renderer.model.values
        if name in {"ParamAngleX", "ParamAngleY"}
    ]
    assert angle_values
    assert max(abs(value) for value in angle_values[-2:]) < 0.5
    assert renderer.model.expression_calls == ["formal"]


class _ActionParameterModel:
    def __init__(self) -> None:
        self.values: list[tuple[str, float, float]] = []
        self.updated = 0
        self.stop_calls = 0

    def GetParamIds(self):  # noqa: N802
        return ["ParamAngleX", "ParamAngleY", "ParamBodyAngleX", "ParamBodyAngleZ"]

    def GetParameter(self, _index):  # noqa: N802
        return _Parameter(-100.0, 100.0, 0.0)

    def Update(self):  # noqa: N802
        self.updated += 1

    def SetParameterValue(self, name, value, weight):  # noqa: N802
        self.values.append((str(name), float(value), float(weight)))

    def StopAllMotions(self):  # noqa: N802
        self.stop_calls += 1


def _last_parameter(model: _ActionParameterModel, name: str) -> float:
    return next(value for parameter, value, _weight in reversed(model.values) if parameter == name)


def test_native_expression_parameters_transition_and_restore_without_leak() -> None:
    model = _ActionParameterModel()
    renderer = _renderer_with(model)
    request = ExpressionRequest.from_arguments(
        {
            "name": "happy",
            "duration_seconds": 0.1,
            "transition_seconds": 0.1,
            "restore": "neutral",
            "parameters": {"ParamBodyAngleX": 4},
        }
    )

    result = renderer.set_expression_request(request)
    assert result["status"] == "started"
    assert _last_parameter(model, "ParamBodyAngleX") == pytest.approx(0.0)

    renderer.advance(0.05)
    assert _last_parameter(model, "ParamBodyAngleX") == pytest.approx(2.0)
    renderer.advance(0.05)
    assert renderer.state.expression == "neutral"
    assert _last_parameter(model, "ParamBodyAngleX") == pytest.approx(4.0)
    renderer.advance(0.05)
    assert _last_parameter(model, "ParamBodyAngleX") == pytest.approx(2.0)
    renderer.advance(0.05)
    assert _last_parameter(model, "ParamBodyAngleX") == pytest.approx(0.0)
    assert renderer._expression_parameter_values == {}


def test_native_motion_uses_natural_duration_loop_and_shutdown_cleans_requests() -> None:
    model = _ActionParameterModel()
    dispose_calls: list[bool] = []

    class Module:
        class MotionPriority:
            NORMAL = 2

        @staticmethod
        def dispose() -> None:
            dispose_calls.append(True)

    renderer = _renderer_with(model)
    renderer._module = Module()
    request = MotionRequest.from_arguments(
        {
            "name": "wave",
            "transition_seconds": 0.1,
            "loop": True,
            "parameters": {"ParamBodyAngleZ": 6},
        }
    )

    result = renderer.play_motion_request(request)
    assert result["status"] == "started"
    for _ in range(18):
        renderer.advance(0.05)

    assert renderer.state.motion == "wave"
    assert renderer._motion_request is request
    assert renderer._motion_request_elapsed == pytest.approx(0.0)
    renderer.advance(0.05)
    assert _last_parameter(model, "ParamBodyAngleZ") == pytest.approx(3.0)

    renderer.shutdown()
    renderer.shutdown()

    assert renderer.model is None
    assert renderer._module is None
    assert renderer._motion_request is None
    assert renderer._motion_parameter_baseline == {}
    assert renderer._expression_parameter_values == {}
    assert model.stop_calls == 1
    assert dispose_calls == [True]
