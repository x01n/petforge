"""跨渲染后端的表情序列、参数和过渡契约。"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

_ACTION_NAME = re.compile(r"[^\x00\r\n]{1,64}\Z")
_PARAMETER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")


def _finite(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} is outside the allowed range")
    return parsed


def _name(value: object, *, field_name: str) -> str:
    rendered = str(value or "").strip()
    if not _ACTION_NAME.fullmatch(rendered):
        raise ValueError(f"{field_name} is invalid")
    return rendered


def _parameters(value: object, *, field_name: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 32:
        raise ValueError(f"{field_name} must be a mapping with at most 32 items")
    result: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name or "").strip()
        if not _PARAMETER_NAME.fullmatch(name):
            raise ValueError(f"{field_name} contains an invalid parameter name")
        result[name] = _finite(
            raw_value,
            name=f"{field_name}.{name}",
            minimum=-10_000.0,
            maximum=10_000.0,
        )
    return result


@dataclass(frozen=True, slots=True)
class ExpressionLayer:
    """一个表情步骤或同时混合层。"""

    name: str
    weight: float = 1.0
    duration_seconds: float = 1.5
    transition_seconds: float = 0.2
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="expression.name"))
        object.__setattr__(
            self,
            "weight",
            _finite(self.weight, name="expression.weight", minimum=0.0, maximum=1.0),
        )
        object.__setattr__(
            self,
            "duration_seconds",
            _finite(
                self.duration_seconds,
                name="expression.duration_seconds",
                minimum=0.05,
                maximum=120.0,
            ),
        )
        object.__setattr__(
            self,
            "transition_seconds",
            _finite(
                self.transition_seconds,
                name="expression.transition_seconds",
                minimum=0.0,
                maximum=10.0,
            ),
        )
        object.__setattr__(
            self,
            "parameters",
            _parameters(self.parameters, field_name="expression.parameters"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ExpressionLayer:
        if not isinstance(value, Mapping):
            raise ValueError("expression layer must be a mapping")
        allowed = {
            "name",
            "weight",
            "duration_seconds",
            "transition_seconds",
            "parameters",
        }
        if set(value) - allowed:
            raise ValueError("expression layer contains unsupported keys")
        return cls(
            name=value.get("name", ""),
            weight=value.get("weight", 1.0),
            duration_seconds=value.get("duration_seconds", 1.5),
            transition_seconds=value.get("transition_seconds", 0.2),
            parameters=value.get("parameters", {}),
        )


@dataclass(frozen=True, slots=True)
class ExpressionRequest:
    """单表情、序列或多层混合请求。"""

    expressions: tuple[ExpressionLayer, ...]
    mode: Literal["sequence", "blend"] = "sequence"
    loop: bool = False
    restore: str = "neutral"

    def __post_init__(self) -> None:
        expressions = tuple(self.expressions or ())
        if not expressions or len(expressions) > 8:
            raise ValueError("expressions must contain 1 to 8 layers")
        if any(not isinstance(item, ExpressionLayer) for item in expressions):
            raise ValueError("expressions must contain ExpressionLayer values")
        mode = str(self.mode or "").strip().lower()
        if mode not in {"sequence", "blend"}:
            raise ValueError("expression mode must be sequence or blend")
        if mode == "blend" and sum(item.weight for item in expressions) <= 0:
            raise ValueError("blend expressions require a positive total weight")
        object.__setattr__(self, "expressions", expressions)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "loop", bool(self.loop))
        object.__setattr__(self, "restore", _name(self.restore, field_name="expression.restore"))

    @classmethod
    def from_arguments(cls, value: Mapping[str, Any]) -> ExpressionRequest:
        if not isinstance(value, Mapping):
            raise ValueError("expression request must be a mapping")
        allowed = {
            "name",
            "expressions",
            "parameters",
            "weight",
            "duration_seconds",
            "transition_seconds",
            "mode",
            "loop",
            "restore",
        }
        if set(value) - allowed:
            raise ValueError("expression request contains unsupported keys")
        has_name = "name" in value
        has_expressions = "expressions" in value
        if has_name == has_expressions:
            raise ValueError("expression request requires exactly one of name or expressions")
        single_layer_fields = {
            "name",
            "parameters",
            "weight",
            "duration_seconds",
            "transition_seconds",
        }
        if has_expressions and single_layer_fields.intersection(value):
            raise ValueError("expression sequence cannot include single-layer fields")
        if has_name:
            layers = (
                ExpressionLayer(
                    name=value.get("name", ""),
                    weight=value.get("weight", 1.0),
                    duration_seconds=value.get("duration_seconds", 1.5),
                    transition_seconds=value.get("transition_seconds", 0.2),
                    parameters=value.get("parameters", {}),
                ),
            )
        else:
            raw_expressions = value.get("expressions")
            if isinstance(raw_expressions, (str, bytes, bytearray)) or not isinstance(
                raw_expressions, Sequence
            ):
                raise ValueError("expressions must be a list")
            layers = tuple(ExpressionLayer.from_mapping(item) for item in raw_expressions)
        return cls(
            expressions=layers,
            mode=str(value.get("mode", "sequence")),
            loop=value.get("loop", False),
            restore=value.get("restore", "neutral"),
        )


@dataclass(frozen=True, slots=True)
class MotionRequest:
    """带参数、持续时间与过渡时间的动作请求。"""

    name: str
    duration_seconds: float | None = None
    transition_seconds: float = 0.2
    loop: bool = False
    parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="motion.name"))
        if self.duration_seconds is not None:
            object.__setattr__(
                self,
                "duration_seconds",
                _finite(
                    self.duration_seconds,
                    name="motion.duration_seconds",
                    minimum=0.05,
                    maximum=300.0,
                ),
            )
        object.__setattr__(
            self,
            "transition_seconds",
            _finite(
                self.transition_seconds,
                name="motion.transition_seconds",
                minimum=0.0,
                maximum=10.0,
            ),
        )
        object.__setattr__(self, "loop", bool(self.loop))
        object.__setattr__(
            self,
            "parameters",
            _parameters(self.parameters, field_name="motion.parameters"),
        )

    @classmethod
    def from_arguments(cls, value: Mapping[str, Any]) -> MotionRequest:
        if not isinstance(value, Mapping):
            raise ValueError("motion request must be a mapping")
        allowed = {"name", "duration_seconds", "transition_seconds", "loop", "parameters"}
        if set(value) - allowed:
            raise ValueError("motion request contains unsupported keys")
        return cls(
            name=value.get("name", ""),
            duration_seconds=value.get("duration_seconds"),
            transition_seconds=value.get("transition_seconds", 0.2),
            loop=value.get("loop", False),
            parameters=value.get("parameters", {}),
        )


@dataclass(frozen=True, slots=True)
class ExpressionFrame:
    """时间线在当前帧的后端无关投影。"""

    previous: str
    current: str
    progress: float
    parameters: Mapping[str, float]
    changed: bool = False
    finished: bool = False


class ExpressionTimeline:
    """确定性推进表情序列或多层参数混合。"""

    def __init__(self) -> None:
        self._request: ExpressionRequest | None = None
        self._index = 0
        self._elapsed = 0.0
        self._previous = "neutral"
        self._finished = True
        self._restoring = False
        self._restore_elapsed = 0.0
        self._restore_transition = 0.0
        self._restore_parameters: dict[str, float] = {}

    @property
    def active(self) -> bool:
        return self._request is not None and not self._finished

    def start(self, request: ExpressionRequest, *, current: str) -> ExpressionFrame:
        if not isinstance(request, ExpressionRequest):
            raise TypeError("request must be an ExpressionRequest")
        self._request = request
        self._index = 0
        self._elapsed = 0.0
        self._previous = _name(current or request.restore, field_name="current expression")
        self._finished = False
        self._restoring = False
        self._restore_elapsed = 0.0
        self._restore_transition = 0.0
        self._restore_parameters.clear()
        return self._frame(changed=True)

    def _blend_parameters(self) -> dict[str, float]:
        request = self._request
        if request is None:
            return {}
        layers = (
            request.expressions if request.mode == "blend" else (request.expressions[self._index],)
        )
        total = sum(layer.weight for layer in layers) or 1.0
        result: dict[str, float] = {}
        for layer in layers:
            for name, value in layer.parameters.items():
                result[name] = result.get(name, 0.0) + value * (layer.weight / total)
        return result

    def _frame(self, *, changed: bool = False, finished: bool = False) -> ExpressionFrame:
        request = self._request
        if request is None:
            return ExpressionFrame(self._previous, self._previous, 1.0, {}, changed, True)
        if self._restoring:
            progress = (
                1.0
                if self._restore_transition <= 0
                else min(1.0, self._restore_elapsed / self._restore_transition)
            )
            parameters = (
                {}
                if finished
                else {
                    name: value * (1.0 - progress)
                    for name, value in self._restore_parameters.items()
                }
            )
            return ExpressionFrame(
                self._previous,
                request.restore,
                progress,
                parameters,
                changed,
                finished,
            )
        layer = request.expressions[self._index]
        transition = layer.transition_seconds
        progress = 1.0 if transition <= 0 else min(1.0, self._elapsed / transition)
        current = (
            max(request.expressions, key=lambda item: item.weight).name
            if request.mode == "blend"
            else layer.name
        )
        return ExpressionFrame(
            self._previous,
            request.restore if finished else current,
            progress,
            self._blend_parameters(),
            changed,
            finished,
        )

    def advance(self, elapsed_seconds: float) -> ExpressionFrame | None:
        request = self._request
        if request is None or self._finished:
            return None
        elapsed = max(0.0, _finite(elapsed_seconds, name="elapsed", minimum=0, maximum=60))
        if self._restoring:
            self._restore_elapsed += elapsed
            completed = (
                self._restore_transition <= 0 or self._restore_elapsed >= self._restore_transition
            )
            if completed:
                self._finished = True
            return self._frame(finished=completed)
        self._elapsed += elapsed
        if request.mode == "blend":
            duration = max(item.duration_seconds for item in request.expressions)
        else:
            duration = request.expressions[self._index].duration_seconds
        if self._elapsed < duration:
            return self._frame()
        if request.mode == "sequence" and self._index + 1 < len(request.expressions):
            self._previous = request.expressions[self._index].name
            self._index += 1
            self._elapsed = 0.0
            return self._frame(changed=True)
        if request.loop:
            self._previous = request.expressions[self._index].name
            self._index = 0
            self._elapsed = 0.0
            return self._frame(changed=True)
        self._previous = (
            max(request.expressions, key=lambda item: item.weight).name
            if request.mode == "blend"
            else request.expressions[self._index].name
        )
        self._restoring = True
        self._restore_elapsed = 0.0
        self._restore_transition = request.expressions[self._index].transition_seconds
        self._restore_parameters = self._blend_parameters()
        if self._restore_transition <= 0:
            self._finished = True
            return self._frame(changed=True, finished=True)
        return self._frame(changed=True)

    def cancel(self) -> None:
        self._request = None
        self._finished = True
        self._restoring = False
        self._elapsed = 0.0
        self._restore_elapsed = 0.0
        self._restore_parameters.clear()


__all__ = [
    "ExpressionFrame",
    "ExpressionLayer",
    "ExpressionRequest",
    "ExpressionTimeline",
    "MotionRequest",
]
