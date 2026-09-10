"""Live2D 能力探测与动态适配器；模型/运行库缺失时明确不可用。"""

from __future__ import annotations

import importlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from core.rendering.actions import ExpressionRequest, ExpressionTimeline, MotionRequest
from core.rendering.resources import (
    validate_expression3_description,
    validate_model3_description,
    validate_motion3_description,
)

from .assets import _probe_live2d_model_loader
from .protocol import (
    DEFAULT_EXPRESSION_NAMES,
    DEFAULT_MOTION_NAMES,
    RendererCapabilities,
    RendererState,
)

# 原生运行时的输入来自 Qt 事件、窗口管理器和第三方绑定。所有进入
# Live2D Core 的数值都要经过同一组上限，避免跨屏/拖动时的瞬时大步进把
# 参数或 MVP 矩阵推到 NaN/Inf。这里的角度上限只是注视预算，不是模型的
# 原始参数范围；真实范围在模型加载后再取 min/max 并留出安全余量。
_MAX_FRAME_SECONDS = 0.05
_MIN_FRAME_SECONDS = 1.0 / 240.0
_MAX_TRACKING_SPEED = 12.0
_TARGET_SLEW_RATE = 10.0
_TRACKING_DEAD_ZONE = 0.08
_POSE_SAFETY_RATIO = 0.78
_MAX_VIEWPORT_EDGE = 16_384
_MAX_DEVICE_PIXEL_RATIO = 8.0
_DEFAULT_POSE_LIMITS: dict[str, tuple[float, float]] = {
    "ParamAngleX": (-8.0, 8.0),
    "ParamAngleY": (-7.0, 7.0),
}
_ACTION_ALIAS_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_PROCEDURAL_EXPRESSION_POSES: dict[str, tuple[tuple[str, float], ...]] = {
    "neutral": (("ParamAngleX", 0.0), ("ParamAngleY", 0.0)),
    "happy": (("ParamAngleX", -6.0), ("ParamAngleY", 2.0)),
    "sad": (("ParamAngleX", 6.0), ("ParamAngleY", -7.0)),
    "curious": (("ParamAngleX", 8.0), ("ParamAngleY", 5.0)),
    "surprised": (("ParamAngleX", 0.0), ("ParamAngleY", 8.0)),
    "shy": (("ParamAngleX", -7.0), ("ParamAngleY", -3.0)),
}
_PROCEDURAL_MOTION_DURATIONS = {"blink": 0.45, "wave": 0.9}


def _nonempty_file(path: Path) -> bool:
    """返回普通非空文件状态；资源探测失败时不抛出到 GUI 线程。"""

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _declared_actions(model_path: Path | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """读取 model3 中通过 exp3/motion3 校验的原生动作名称。"""

    if model_path is None:
        return (), ()
    try:
        value = json.loads(model_path.read_text(encoding="utf-8"))
        references = value.get("FileReferences") if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return (), ()
    if not isinstance(references, Mapping):
        return (), ()

    def child(reference: object) -> Path | None:
        if not isinstance(reference, str) or not reference.strip():
            return None
        candidate = Path(reference)
        if candidate.is_absolute() or candidate.anchor or "\x00" in reference:
            return None
        try:
            resolved = (model_path.parent / candidate).resolve()
            resolved.relative_to(model_path.parent.resolve())
        except (OSError, RuntimeError, ValueError):
            return None
        return resolved if _nonempty_file(resolved) else None

    expressions: list[str] = []
    raw_expressions = references.get("Expressions")
    if isinstance(raw_expressions, list):
        for item in raw_expressions:
            if not isinstance(item, Mapping):
                continue
            name = str(item.get("Name", "") or "").strip()
            path = child(item.get("File"))
            if (
                name
                and path is not None
                and validate_expression3_description(path)
                and name not in expressions
            ):
                expressions.append(name)

    motions: list[str] = []
    raw_motions = references.get("Motions")
    if isinstance(raw_motions, Mapping):
        for group, entries in raw_motions.items():
            group_name = str(group or "").strip()
            if not group_name or not isinstance(entries, list):
                continue
            if any(
                isinstance(item, Mapping)
                and (path := child(item.get("File"))) is not None
                and validate_motion3_description(path)
                for item in entries
            ):
                motions.append(group_name)
    return tuple(expressions), tuple(motions)


def _declared_motion_durations(model_path: Path | None) -> dict[str, float]:
    """读取正式 motion3 的单循环时长，供循环动作一次性收尾。"""

    if model_path is None:
        return {}
    try:
        descriptor = json.loads(model_path.read_text(encoding="utf-8"))
        references = descriptor.get("FileReferences") if isinstance(descriptor, Mapping) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    raw_motions = references.get("Motions") if isinstance(references, Mapping) else None
    if not isinstance(raw_motions, Mapping):
        return {}
    result: dict[str, float] = {}
    for group, entries in raw_motions.items():
        group_name = str(group or "").strip()
        if not group_name or not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, Mapping) or not isinstance(item.get("File"), str):
                continue
            path = (model_path.parent / str(item["File"])).resolve()
            try:
                path.relative_to(model_path.parent.resolve())
                motion = json.loads(path.read_text(encoding="utf-8"))
                meta = motion.get("Meta") if isinstance(motion, Mapping) else None
                raw_duration: object = meta.get("Duration") if isinstance(meta, Mapping) else None
                if isinstance(raw_duration, bool) or not isinstance(
                    raw_duration, (int, float, str)
                ):
                    continue
                duration = float(raw_duration)
            except (
                OSError,
                UnicodeError,
                ValueError,
                TypeError,
                OverflowError,
                json.JSONDecodeError,
            ):
                continue
            if math.isfinite(duration) and duration > 0.0:
                result[group_name] = duration
                break
    return result


def _motion_aliases(
    model_path: Path | None, declared: tuple[str, ...]
) -> tuple[tuple[str, str], ...]:
    """读取与原生 model3 对应的显式 YAML 动作别名。"""

    if model_path is None:
        return ()
    prefix = (
        model_path.name[: -len(".model3.json")] if model_path.name.endswith(".model3.json") else ""
    )
    if not prefix:
        return ()
    sidecar = model_path.with_name(f"{prefix}.actions.yaml")
    value: object = None
    if _nonempty_file(sidecar):
        try:
            value = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            value = None
    else:
        # 运行时传入的是 resources/live2d/model；项目级配置位于根目录
        # config，只有在该精确布局存在时才读取，不从其它目录猜测。
        try:
            resource_root = model_path.parents[3]
            project_root = model_path.parents[4]
        except IndexError:
            return ()
        config_path = project_root / "config" / "live2d-actions.yaml"
        if _nonempty_file(config_path):
            try:
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, yaml.YAMLError):
                config = None
            try:
                relative_model = model_path.relative_to(resource_root).as_posix()
            except ValueError:
                relative_model = ""
            models = config.get("models") if isinstance(config, Mapping) else None
            value = models.get(relative_model) if isinstance(models, Mapping) else None
    if not isinstance(value, Mapping) or value.get("version") != 1:
        return ()
    motions = value.get("motions")
    if not isinstance(motions, Mapping):
        return ()
    declared_set = set(declared)
    result: list[tuple[str, str]] = []
    for raw_alias, raw_target in motions.items():
        alias = str(raw_alias or "").strip().lower()
        target = str(raw_target or "").strip()
        if (
            _ACTION_ALIAS_PATTERN.fullmatch(alias)
            and target in declared_set
            and alias not in {item[0] for item in result}
        ):
            result.append((alias, target))
    return tuple(sorted(result))


def _known_action(
    value: str,
    declared: tuple[str, ...],
    aliases: tuple[tuple[str, str], ...] = (),
) -> bool:
    """只接受精确正式名称或显式别名；无 model3 时由调用方保留兼容。"""

    requested = str(value or "").strip()
    if not requested:
        return False
    if requested in declared:
        return True
    folded = requested.casefold()
    if any(alias.casefold() == folded for alias, _target in aliases):
        return True
    return False


def _finite_float(value: object, default: float = 0.0) -> float:
    """将外部数值归一化为有限浮点；非法值使用明确的安全默认值。"""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _clamp(value: float, lower: float, upper: float) -> float:
    """在已知有限边界内钳位数值。"""

    if lower > upper:
        lower, upper = upper, lower
    return max(lower, min(upper, value))


def _bounded_dimension(value: object) -> int | None:
    """校验逻辑视口尺寸并限制异常超大输入。"""

    numeric = _finite_float(value, -1.0)
    if numeric <= 0.0:
        return None
    integer = int(round(numeric))
    if integer <= 0:
        return None
    return min(_MAX_VIEWPORT_EDGE, integer)


def _shape_tracking_axis(value: float) -> float:
    """对注视目标应用死区和 smoothstep，消除中心抖动与边缘折线。"""

    sign = -1.0 if value < 0.0 else 1.0
    magnitude = max(0.0, abs(value))
    if magnitude <= _TRACKING_DEAD_ZONE:
        return 0.0
    denominator = max(1e-6, 1.0 - _TRACKING_DEAD_ZONE)
    normalized = _clamp((magnitude - _TRACKING_DEAD_ZONE) / denominator, 0.0, 1.0)
    shaped = normalized * normalized * (3.0 - 2.0 * normalized)
    return sign * shaped


def _find_model(model_dir: Path) -> Path | None:
    model_paths = sorted(model_dir.rglob("*.model3.json"))
    for path in model_paths:
        if validate_model3_description(path) is not None:
            return path
    return None


def _resolve_selected_model(model_dir: Path, model_path: str | Path | None) -> Path | None:
    """在模型目录内解析显式描述，拒绝绝对路径和路径穿越。"""

    if model_path is None:
        return _find_model(model_dir)
    try:
        root = model_dir.expanduser().resolve()
        raw = Path(model_path).expanduser()
        candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
        candidate.relative_to(root)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
    if not candidate.name.endswith(".model3.json"):
        return None
    return candidate if validate_model3_description(candidate) is not None else None


class Live2DRenderer:
    def __init__(self, model_dir: str | Path, *, model_path: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self._model_path = (
            _resolve_selected_model(self.model_dir, model_path) if self.model_dir.is_dir() else None
        )
        self._declared_expressions, self._declared_motions = _declared_actions(self._model_path)
        self._motion_aliases = _motion_aliases(self._model_path, self._declared_motions)
        self._motion_durations = _declared_motion_durations(self._model_path)
        self._module: Any | None = None
        self._model: Any | None = None
        self._state = RendererState()
        self._viewport_size = (1, 1)
        self._device_pixel_ratio = 1.0
        self._cursor_target = (0.0, 0.0)
        self._cursor_goal = (0.0, 0.0)
        self._cursor_current = (0.0, 0.0)
        self._cursor_velocity = (0.0, 0.0)
        self._facing = 1.0
        self._facing_target = 1.0
        self._facing_sign = 1
        self._pending_direction: str | None = None
        self._facing_pose = 1.0
        self._interaction_locked = False
        # 拖动窗口时冻结模型时间步，保持当前姿态和网格不变；窗口几何
        # 仍由宿主更新，释放后再从新的时间基线继续播放，避免拖动期间
        # 动作/物理与合成器重绘交错造成视觉抽搐。
        self._dragging = False
        self._mouth_viseme = "Silence"
        self._mouth_intensity = 0.0
        self._procedural_expression = "neutral"
        self._procedural_expression_active = False
        self._procedural_motion = "idle"
        self._procedural_motion_active = False
        self._formal_motion_elapsed = 0.0
        self._formal_motion_name: str | None = None
        self._expression_timeline = ExpressionTimeline()
        self._expression_frame_parameters: dict[str, float] = {}
        self._expression_parameter_starts: dict[str, float] = {}
        self._expression_parameter_values: dict[str, float] = {}
        self._expression_frame_progress = 1.0
        self._motion_request: MotionRequest | None = None
        self._motion_request_elapsed = 0.0
        self._motion_parameter_baseline: dict[str, float] = {}
        self._motion_parameter_values: dict[str, float] = {}
        self._parameter_limits: dict[str, tuple[float, float, float]] = {}
        self._parameter_ids: frozenset[str] = frozenset()
        self._parameter_limits_model: Any | None = None
        self._parameter_limits_loaded = False
        self._initialization_error = ""
        self._last_transform_valid = True
        self._transform_invalid_frames = 0
        self._capabilities = self._probe()

    def _probe(self) -> RendererCapabilities:
        if self._model_path is None:
            return RendererCapabilities(
                "opengl", False, message="no usable Live2D model3 descriptor was found"
            )
        try:
            module = importlib.import_module("live2d.v3")
        except (
            ImportError,
            ModuleNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            SystemError,
        ):
            return RendererCapabilities(
                "opengl", False, self._model_path, message="live2d-py is unavailable"
            )
        required = ("init", "glInit", "clearBuffer", "dispose", "LAppModel")
        if any(not callable(getattr(module, name, None)) for name in required):
            return RendererCapabilities(
                "opengl",
                False,
                self._model_path,
                message="live2d-py v3 API is incomplete",
            )
        loader_available, loader_message, _evidence = _probe_live2d_model_loader(module)
        if not loader_available:
            return RendererCapabilities(
                "opengl",
                False,
                self._model_path,
                message=loader_message,
            )
        self._module = module
        expression_names: list[str] = []
        expression_keys: set[str] = set()
        for name in (*self._declared_expressions, *DEFAULT_EXPRESSION_NAMES):
            text = str(name or "").strip()
            key = text.casefold()
            if text and key not in expression_keys:
                expression_names.append(text)
                expression_keys.add(key)
        motion_names: list[str] = []
        motion_keys: set[str] = set()
        for name in (
            *(alias for alias, _target in self._motion_aliases),
            *self._declared_motions,
            *DEFAULT_MOTION_NAMES,
        ):
            text = str(name or "").strip()
            key = text.casefold()
            if text and key not in motion_keys:
                motion_names.append(text)
                motion_keys.add(key)
        return RendererCapabilities(
            "opengl",
            True,
            self._model_path,
            expressions=tuple(expression_names),
            motions=tuple(motion_names),
            message="Live2D runtime and model loader are available",
        )

    @property
    def capabilities(self) -> RendererCapabilities:
        if self._initialization_error and self._capabilities.available:
            return replace(
                self._capabilities,
                available=False,
                message=self._initialization_error,
            )
        return self._capabilities

    def supports_expression(self, name: str) -> bool:
        """判断表情是否为正式声明或明确的程序化兼容动作。"""

        if self._model_path is None:
            # 没有模型描述时保留旧测试替身的宽松兼容语义；真实实例一旦
            # 发现 model3，就必须走精确动作清单。
            return bool(str(name or "").strip())
        requested = str(name or "").strip()
        if _known_action(requested, self._declared_expressions):
            return True
        folded = requested.casefold()
        if any(item.casefold() == folded for item in self._declared_expressions):
            return False
        return folded in {item.casefold() for item in DEFAULT_EXPRESSION_NAMES}

    def supports_motion(self, name: str) -> bool:
        """判断动作是否为正式组名、显式 YAML 别名或程序化动作。"""

        if self._model_path is None:
            return bool(str(name or "").strip())
        requested = str(name or "").strip()
        if _known_action(requested, self._declared_motions, self._motion_aliases):
            return True
        folded = requested.casefold()
        if any(item.casefold() == folded for item in self._declared_motions):
            return False
        return folded in {item.casefold() for item in DEFAULT_MOTION_NAMES}

    @property
    def state(self) -> RendererState:
        return self._state

    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def viewport_size(self) -> tuple[int, int]:
        """返回当前有效的逻辑视口尺寸。"""

        return self._viewport_size

    @property
    def last_transform_valid(self) -> bool:
        """返回最近一次绘制前的 MVP 有限性校验结果。"""

        return self._last_transform_valid

    def _refresh_parameter_limits(self, model: Any | None = None) -> None:
        """读取 LAppModel 参数边界并缓存安全写入范围。

        ``live2d-py`` 的不同发行版分别暴露 ``GetParamIds`` 或
        ``GetParameterIds``，参数边界则可能通过 ``GetParameter`` 对象或
        按索引读取。这里只使用实际存在的接口；读取失败时保留空缓存，
        由注视预算的保守默认值接管，不把猜测的字段写回模型。
        """

        target = self._model if model is None else model
        if target is None or self._parameter_limits_model is target:
            return
        self._parameter_limits_model = target
        self._parameter_limits_loaded = True
        self._parameter_limits = {}
        self._parameter_ids = frozenset()

        ids_getter = getattr(target, "GetParamIds", None)
        if not callable(ids_getter):
            ids_getter = getattr(target, "GetParameterIds", None)
        ids: tuple[str, ...] = ()
        if callable(ids_getter):
            try:
                raw_ids = ids_getter()
                if isinstance(raw_ids, str):
                    raw_ids = (raw_ids,)
                if raw_ids is None:
                    raw_ids = ()
                ids = tuple(
                    value for value in (str(item or "").strip() for item in tuple(raw_ids)) if value
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                ids = ()
        self._parameter_ids = frozenset(ids)

        parameter_getter = getattr(target, "GetParameter", None)
        minimum_getter = getattr(target, "GetParameterMinimumValue", None)
        maximum_getter = getattr(target, "GetParameterMaximumValue", None)
        default_getter = getattr(target, "GetParameterDefaultValue", None)
        for index, parameter_id in enumerate(ids):
            raw_min: object | None = None
            raw_max: object | None = None
            raw_default: object | None = None
            if callable(parameter_getter):
                try:
                    parameter = parameter_getter(index)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    parameter = None
                if parameter is not None:
                    raw_min = getattr(parameter, "min", None)
                    raw_max = getattr(parameter, "max", None)
                    raw_default = getattr(parameter, "default", None)
            if raw_min is None and callable(minimum_getter):
                try:
                    raw_min = minimum_getter(index)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    raw_min = None
            if raw_max is None and callable(maximum_getter):
                try:
                    raw_max = maximum_getter(index)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    raw_max = None
            if raw_default is None and callable(default_getter):
                try:
                    raw_default = default_getter(index)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    raw_default = None
            minimum = _finite_float(raw_min, math.nan)
            maximum = _finite_float(raw_max, math.nan)
            if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
                continue
            default = _finite_float(raw_default, (minimum + maximum) / 2.0)
            default = _clamp(default, minimum, maximum)
            self._parameter_limits[parameter_id] = (minimum, maximum, default)

    def _parameter_bounds(
        self,
        parameter_id: str,
        fallback: tuple[float, float],
    ) -> tuple[float, float]:
        """返回留出物理/动作余量后的参数写入边界。"""

        lower, upper = fallback
        raw = self._parameter_limits.get(parameter_id)
        if raw is not None:
            raw_minimum, raw_maximum, default = raw
            # 以模型默认值为中心保留 22% 原始范围，避免把动作/物理
            # 的边界值当作稳定注视姿态。随后再与注视预算取交集。
            safe_minimum = default - (default - raw_minimum) * _POSE_SAFETY_RATIO
            safe_maximum = default + (raw_maximum - default) * _POSE_SAFETY_RATIO
            lower = max(lower, safe_minimum)
            upper = min(upper, safe_maximum)
        if lower > upper:
            # 极窄或非对称模型范围仍必须得到一个确定值，优先使用默认
            # 范围的中点而不是把越界值送到 Core。
            midpoint = (fallback[0] + fallback[1]) / 2.0
            lower = upper = midpoint
        return float(lower), float(upper)

    def _set_safe_parameter(
        self,
        parameter_id: str,
        value: object,
        fallback: tuple[float, float],
    ) -> bool:
        """以有限值、模型边界和保守注视预算写入单个参数。"""

        model = self._model
        setter = getattr(model, "SetParameterValue", None) if model is not None else None
        if not callable(setter):
            return False
        # 只有在模型明确报告了参数 ID 列表时才拒绝未知参数；旧绑定
        # 没有列表接口时仍允许其自身处理兼容 ID。
        if self._parameter_ids and parameter_id not in self._parameter_ids:
            return False
        lower, upper = self._parameter_bounds(parameter_id, fallback)
        safe_value = _clamp(_finite_float(value), lower, upper)
        try:
            setter(parameter_id, safe_value, 1.0)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return False
        return True

    def _parameter_default(self, parameter_id: str) -> float:
        """返回已探测模型默认值；旧绑定没有边界接口时使用中性零值。"""

        limits = self._parameter_limits.get(parameter_id)
        return float(limits[2]) if limits is not None else 0.0

    def _configure_expression_parameters(self, parameters: Mapping[str, float]) -> None:
        """为下一表情阶段捕获连续起点并补齐旧参数的恢复目标。"""

        requested = dict(parameters)
        parameter_ids = set(self._expression_parameter_values) | set(requested)
        starts: dict[str, float] = {}
        targets: dict[str, float] = {}
        for parameter_id in parameter_ids:
            default = self._parameter_default(parameter_id)
            starts[parameter_id] = self._expression_parameter_values.get(parameter_id, default)
            targets[parameter_id] = requested.get(parameter_id, default)
        self._expression_parameter_starts = starts
        self._expression_frame_parameters = targets

    def _apply_expression_parameters(self, progress: float) -> None:
        """按时间线进度写入表情参数，避免阶段边界从零值瞬跳。"""

        amount = _clamp(_finite_float(progress, 1.0), 0.0, 1.0)
        for parameter_id, target in self._expression_frame_parameters.items():
            start = self._expression_parameter_starts.get(
                parameter_id,
                self._parameter_default(parameter_id),
            )
            value = start + (target - start) * amount
            if self._set_safe_parameter(parameter_id, value, (-10_000.0, 10_000.0)):
                self._expression_parameter_values[parameter_id] = value

    def _reset_expression_parameters(self, *, apply: bool = True) -> None:
        """清理表情参数缓存，并按需把已写入值恢复到模型默认值。"""

        if apply:
            for parameter_id in self._expression_parameter_values:
                self._set_safe_parameter(
                    parameter_id,
                    self._parameter_default(parameter_id),
                    (-10_000.0, 10_000.0),
                )
        self._expression_frame_parameters.clear()
        self._expression_parameter_starts.clear()
        self._expression_parameter_values.clear()
        self._expression_frame_progress = 1.0

    def _configure_motion_parameters(self, request: MotionRequest) -> None:
        """记录动作参数的中性基线，供进入和退出过渡复用。"""

        self._reset_motion_parameters()
        for parameter_id in request.parameters:
            baseline = self._parameter_default(parameter_id)
            self._motion_parameter_baseline[parameter_id] = baseline
            self._motion_parameter_values[parameter_id] = baseline

    def _motion_request_duration(self, request: MotionRequest) -> float | None:
        """解析显式时长、正式 motion3 时长或既有程序化动作时长。"""

        if request.duration_seconds is not None:
            return request.duration_seconds
        resolved = next(
            (
                target
                for alias, target in self._motion_aliases
                if alias.casefold() == request.name.casefold()
            ),
            request.name,
        )
        declared = self._motion_durations.get(resolved)
        if declared is not None and declared > 0:
            return declared
        return _PROCEDURAL_MOTION_DURATIONS.get(request.name.casefold())

    def _apply_motion_parameters(self, request: MotionRequest, duration: float | None) -> None:
        """按动作进入/退出包络写入参数，非循环动作结束前回到基线。"""

        transition = request.transition_seconds
        progress = 1.0 if transition <= 0 else min(1.0, self._motion_request_elapsed / transition)
        if duration is not None and transition > 0:
            remaining = max(0.0, duration - self._motion_request_elapsed)
            progress = min(progress, remaining / transition)
        for parameter_id, target in request.parameters.items():
            baseline = self._motion_parameter_baseline.get(
                parameter_id,
                self._parameter_default(parameter_id),
            )
            value = baseline + (target - baseline) * progress
            if self._set_safe_parameter(parameter_id, value, (-10_000.0, 10_000.0)):
                self._motion_parameter_values[parameter_id] = value

    def _reset_motion_parameters(self, *, apply: bool = True) -> None:
        """清理动作参数缓存，并按需恢复其进入请求前的中性值。"""

        if apply:
            for parameter_id, baseline in self._motion_parameter_baseline.items():
                self._set_safe_parameter(parameter_id, baseline, (-10_000.0, 10_000.0))
        self._motion_parameter_baseline.clear()
        self._motion_parameter_values.clear()

    def _stop_all_motions(self) -> None:
        """通过绑定实际暴露的停止入口终止当前正式动作。"""

        model = self._model
        stopper = getattr(model, "StopAllMotions", None) if model is not None else None
        if not callable(stopper):
            manager = getattr(model, "motionManager", None) if model is not None else None
            stopper = getattr(manager, "stopAllMotions", None)
            if not callable(stopper):
                stopper = getattr(manager, "StopAllMotions", None)
        if callable(stopper):
            try:
                stopper()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass

    def _validate_transform(self) -> bool:
        """校验原生 MVP，拒绝 NaN/Inf 或异常长度的矩阵。"""

        model = self._model
        if model is None:
            return False
        getter = getattr(model, "GetMvp", None)
        if not callable(getter):
            # 当前 LAppModel 包装器没有公开 GetMvp，但其底层 Model 有；
            # 仅在该私有桥实际存在时读取，旧版本仍保持兼容。
            core_model = getattr(model, "_model", None)
            getter = getattr(core_model, "GetMvp", None) if core_model is not None else None
        if not callable(getter):
            self._last_transform_valid = True
            return True
        try:
            raw_values = getter()
            values = () if raw_values is None else tuple(float(item) for item in tuple(raw_values))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError, OverflowError):
            values = ()
        valid = len(values) >= 16 and all(math.isfinite(item) for item in values[:16])
        # 过大的有限值同样会把顶点投射到视口之外，视为坏帧而不是绘制。
        valid = valid and max((abs(item) for item in values[:16]), default=0.0) <= 1e6
        self._last_transform_valid = valid
        if valid:
            self._transform_invalid_frames = 0
        else:
            self._transform_invalid_frames += 1
        return valid

    def initialize(self) -> bool:
        if self._module is None or self._model_path is None:
            return False
        # Qt 在 OpenGL context 重建或宿主重复装配时可能再次调用
        # initializeGL；已有模型继续复用当前生命周期，不能重复加载
        # MOC3 或再次执行模块级初始化。
        if self._model is not None:
            return True
        self._initialization_error = ""
        try:
            init = getattr(self._module, "init", None)
            gl_init = getattr(self._module, "glInit", None)
            if callable(init):
                init()
            if callable(gl_init):
                gl_init()
            model_class = getattr(self._module, "LAppModel", None)
            if model_class is None:
                raise RuntimeError("live2d-py LAppModel is unavailable")
            model = model_class()
            load = getattr(model, "LoadModelJson", None)
            if not callable(load):
                raise RuntimeError("live2d model loader is unavailable")
            load(str(self._model_path))
            self._model = model
            self._initialization_error = ""
            self._parameter_limits_model = None
            self._refresh_parameter_limits(model)
            return True
        except Exception as exc:
            self._initialization_error = f"Live2D initialization failed: {type(exc).__name__}"
            self._model = None
            return False

    def advance(self, elapsed_seconds: float) -> None:
        """推进原生模型并在 Update 后应用受限光标姿态。"""

        if self._dragging:
            return
        elapsed = _clamp(_finite_float(elapsed_seconds), 0.0, _MAX_FRAME_SECONDS)
        self._state.elapsed += elapsed
        expression_frame = self._expression_timeline.advance(elapsed)
        if expression_frame is not None:
            if expression_frame.changed:
                restoring = bool(getattr(self._expression_timeline, "_restoring", False))
                if restoring:
                    # 时间步恰好越过保持时长时，先收敛上一阶段的目标值，再以
                    # 该值作为恢复过渡起点，避免半程参数在切回 neutral 时跳变。
                    self._apply_expression_parameters(1.0)
                self._set_expression_name(expression_frame.current)
                self._configure_expression_parameters(
                    {} if restoring else expression_frame.parameters
                )
            self._expression_frame_progress = expression_frame.progress
        motion_request = self._motion_request
        motion_duration: float | None = None
        motion_should_finish = False
        if motion_request is not None:
            self._motion_request_elapsed += elapsed
            motion_duration = self._motion_request_duration(motion_request)
            if motion_duration is not None and self._motion_request_elapsed >= motion_duration:
                if motion_request.loop:
                    self._motion_request_elapsed %= motion_duration
                    self._play_motion_name(motion_request.name)
                else:
                    motion_should_finish = True
        model = self._model
        if model is None:
            return
        self._refresh_parameter_limits(model)
        update = getattr(model, "Update", None)
        if callable(update):
            try:
                update()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return
        if self._formal_motion_name is not None and self._motion_request is None:
            self._formal_motion_elapsed += elapsed
            duration = self._motion_durations.get(self._formal_motion_name, 0.0)
            if duration > 0.0 and self._formal_motion_elapsed >= duration:
                self._stop_all_motions()
                self._formal_motion_name = None
                self._formal_motion_elapsed = 0.0
                self._state.motion = "idle"
                self._state.elapsed = 0.0
        dt = max(_MIN_FRAME_SECONDS, min(_MAX_FRAME_SECONDS, elapsed))
        # 目标先经过有界 slew，再进入临界阻尼；跨屏/窗口移动造成的
        # 单帧大跳不会直接变成模型参数的大跳。
        self._cursor_target = tuple(
            self._slew_value(current, goal, dt, _TARGET_SLEW_RATE)
            for current, goal in zip(self._cursor_target, self._cursor_goal)
        )
        pairs = tuple(
            self._smooth_axis(current, target, velocity, dt)
            for current, target, velocity in zip(
                self._cursor_current, self._cursor_target, self._cursor_velocity
            )
        )
        self._cursor_current = tuple(pair[0] for pair in pairs)
        self._cursor_velocity = tuple(pair[1] for pair in pairs)
        self._facing_pose = _clamp(_finite_float(self._facing_pose, 1.0), -1.0, 1.0)
        self._facing_target = -1.0 if self._facing_target < 0.0 else 1.0
        facing_step = min(0.18, max(0.04, dt * 6.0))
        self._facing_pose += max(
            -facing_step, min(facing_step, self._facing_target - self._facing_pose)
        )
        if abs(self._facing_target - self._facing_pose) < 0.01:
            self._facing_pose = self._facing_target
        if self._facing_sign > 0 and self._facing_pose <= -0.12:
            if self._pending_direction is None or self._apply_pending_direction():
                self._facing_sign = -1
        elif self._facing_sign < 0 and self._facing_pose >= 0.12:
            if self._pending_direction is None or self._apply_pending_direction():
                self._facing_sign = 1
        # LAppModel.Update 可能恢复动作管理器保存的参数；程序化表情的角度
        # 在最终注视写入时叠加一次。正式 exp3 表情完全交给 Cubism 管理器，
        # 不能把上一个程序化表情残值带入正式动作。
        expression_pose = (
            dict(_PROCEDURAL_EXPRESSION_POSES.get(self._procedural_expression, ()))
            if self._procedural_expression_active
            else {}
        )
        self._set_safe_parameter(
            "ParamAngleX",
            self._cursor_current[0] * 8.0 * _finite_float(self._facing_pose, 1.0)
            + expression_pose.get("ParamAngleX", 0.0),
            _DEFAULT_POSE_LIMITS["ParamAngleX"],
        )
        self._set_safe_parameter(
            "ParamAngleY",
            self._cursor_current[1] * 7.0 + expression_pose.get("ParamAngleY", 0.0),
            _DEFAULT_POSE_LIMITS["ParamAngleY"],
        )
        self._apply_expression_parameters(self._expression_frame_progress)
        if expression_frame is not None and expression_frame.finished:
            self._reset_expression_parameters(apply=False)
        # 程序化动作只写入经过同一安全边界的附加参数；正式 motion3 不
        # 进入此分支，避免动作管理器的 keyform 被覆盖。
        self._apply_procedural_motion(elapsed)
        # 显式 request.parameters 是最终覆盖层，必须位于程序化动作之后；
        # 否则 wave/blink 对同一参数的内部写入会吞掉调用方要求的渐变值。
        if motion_request is not None:
            self._apply_motion_parameters(motion_request, motion_duration)
        if motion_should_finish:
            self._reset_motion_parameters()
            self._motion_request = None
            self._motion_request_elapsed = 0.0
            if self._formal_motion_name is not None:
                self._stop_all_motions()
            self._play_motion_name("idle")
        self._apply_mouth_pose()

    def set_mouth_viseme(self, name: str, *, intensity: float = 1.0) -> bool:
        """设置有限口型形状，并将通用 PCM 标签映射到模型参数。"""

        requested = str(name or "").strip().upper()
        aliases = {"NEUTRAL": "I", "OPEN": "A"}
        viseme = aliases.get(requested, requested)
        if viseme not in {"SILENCE", "A", "I", "U", "E", "O"}:
            return False
        amount = _finite_float(intensity, math.nan)
        if not math.isfinite(amount):
            return False
        self._mouth_viseme = viseme.title() if viseme != "SILENCE" else "Silence"
        self._mouth_intensity = _clamp(amount, 0.0, 1.0)
        if self._model is None:
            return False
        return self._apply_mouth_pose()

    def _apply_mouth_pose(self) -> bool:
        """在模型 Update 后写入口型参数；缺少参数时安全返回失败。"""

        shapes: dict[str, tuple[float, float]] = {
            "Silence": (0.0, 0.0),
            "A": (1.0, 1.0),
            "I": (1.0, 0.4),
            "U": (-1.0, 0.4),
            "E": (1.0, 0.7),
            "O": (-1.0, 1.0),
        }
        form, opening = shapes.get(self._mouth_viseme, shapes["Silence"])
        amount = _clamp(_finite_float(self._mouth_intensity), 0.0, 1.0)
        form_ok = self._set_safe_parameter("ParamMouthForm", form * amount, (-1.0, 1.0))
        open_ok = self._set_safe_parameter("ParamMouthOpenY", opening * amount, (0.0, 1.0))
        return form_ok or open_ok

    def set_cursor_target(self, x: int, y: int, width: int, height: int) -> bool:
        """把全局光标投影到原生模型的有限注视目标。"""

        normalized_width = _bounded_dimension(width)
        normalized_height = _bounded_dimension(height)
        raw_x = _finite_float(x, math.nan)
        raw_y = _finite_float(y, math.nan)
        if normalized_width is None or normalized_height is None:
            return False
        if not math.isfinite(raw_x) or not math.isfinite(raw_y):
            return False
        pixel_width = max(1, normalized_width - 1)
        pixel_height = max(1, normalized_height - 1)
        normalized_x = _clamp(raw_x / pixel_width, 0.0, 1.0)
        normalized_y = _clamp(raw_y / pixel_height, 0.0, 1.0)
        self._viewport_size = (normalized_width, normalized_height)
        next_target = (
            _shape_tracking_axis((normalized_x - 0.5) * 2.0),
            _shape_tracking_axis((0.5 - normalized_y) * 2.0),
        )
        target_jump = math.hypot(
            next_target[0] - self._cursor_goal[0],
            next_target[1] - self._cursor_goal[1],
        )
        if target_jump > 1.1:
            self._cursor_velocity = (0.0, 0.0)
        self._cursor_goal = next_target
        return True

    def set_interaction_locked(self, locked: bool) -> bool:
        """锁定原生窗口输入但保留光标追踪。"""

        self._interaction_locked = bool(locked)
        return True

    def set_dragging(self, dragging: bool) -> bool:
        """冻结/恢复拖动期间的模型时间步。"""

        value = bool(dragging)
        if value and not self._dragging:
            self._cursor_velocity = (0.0, 0.0)
        self._dragging = value
        if not value:
            # 释放后让新一轮注视从稳定速度开始，避免把拖动前的速度
            # 带到当前窗口位置。
            self._cursor_velocity = (0.0, 0.0)
        return True

    def reload_resources(
        self,
        resource_root: str | Path,
        *,
        model_path: str | Path | None = None,
        sprite_scale: float | None = None,
    ) -> Mapping[str, object]:
        """原生 Live2D 资源由当前 OpenGL 生命周期持有，只能重启交换。"""

        del resource_root, model_path, sprite_scale
        if self._dragging:
            return {"status": "busy", "reason": "resource reload is blocked while dragging"}
        return {
            "status": "restart_required",
            "reason": "native Live2D resources are owned by the current OpenGL context",
        }

    def notify_window_moved(self) -> bool:
        """清除窗口位移前的追踪速度，避免转场时向旧方向过冲。"""

        self._cursor_velocity = (0.0, 0.0)
        # 先把目标收回中性点；宿主随后会用新的全局坐标覆盖 goal，
        # 没有新坐标时也不会继续追逐窗口移动前的旧位置。
        self._cursor_goal = (0.0, 0.0)
        return True

    def resize(self, width: int, height: int, device_pixel_ratio: float = 1.0) -> bool:
        """把原生模型画布同步到逻辑窗口尺寸，始终由模型负责等比适配。"""

        normalized_width = _bounded_dimension(width)
        normalized_height = _bounded_dimension(height)
        if normalized_width is None or normalized_height is None:
            # Qt 在窗口刚创建/销毁时可能回调 0×0；保留上一份有效视口，
            # 不把零尺寸写入 Core，避免下一帧出现无穷比例或拉伸。
            return False
        dpr = _finite_float(device_pixel_ratio, 1.0)
        dpr = _clamp(dpr, 1.0, _MAX_DEVICE_PIXEL_RATIO)
        changed = (normalized_width, normalized_height) != self._viewport_size
        resize = getattr(self._model, "Resize", None)
        if callable(resize):
            try:
                result = resize(normalized_width, normalized_height)
                if result is False:
                    return False
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
        # 只有模型接受新画布后才提交 Python 侧视口，避免失败的 Resize
        # 让后续命中/追踪误以为新尺寸已经生效而停止重试。
        self._viewport_size = (normalized_width, normalized_height)
        self._device_pixel_ratio = dpr
        if changed:
            self.notify_window_moved()
        return True

    def draw(self) -> None:
        if self._model is None:
            return
        if not self._validate_transform():
            # QOpenGLWindow 会把异常交给宿主的精灵回退；不要把坏矩阵
            # 继续提交到 OpenGL，否则会在窗口移动/恢复时留下穿模帧。
            raise RuntimeError("Live2D MVP transform is not finite")
        draw = getattr(self._model, "Draw", None)
        if callable(draw):
            draw()

    def _apply_procedural_expression(self, name: str) -> bool:
        """使用已探测参数提供无 exp3 时的有限语义表情回退。"""

        profile = _PROCEDURAL_EXPRESSION_POSES.get(str(name or "").strip().casefold())
        if profile is None:
            return False
        self._refresh_parameter_limits(self._model)
        changed = False
        for parameter_id, value in profile:
            changed = (
                self._set_safe_parameter(
                    parameter_id,
                    value,
                    _DEFAULT_POSE_LIMITS.get(parameter_id, (-1.0, 1.0)),
                )
                or changed
            )
        return changed

    def _apply_procedural_motion(self, elapsed: float) -> bool:
        """以统一参数平移实现无 motion3 时的轻量动作回退。"""

        if not self._procedural_motion_active:
            return False
        name = self._procedural_motion
        phase = self._state.elapsed
        if name == "blink":
            cycle = phase % 0.45 if self._motion_request is not None else min(0.45, phase)
            progress = cycle / 0.45
            openness = (
                1.0 - progress / 0.35
                if progress < 0.35
                else 0.0
                if progress < 0.72
                else (progress - 0.72) / 0.28
            )
            changed = self._set_safe_parameter("ParamEyeLOpen", openness, (0.0, 1.0))
            changed = self._set_safe_parameter("ParamEyeROpen", openness, (0.0, 1.0)) or changed
            if self._motion_request is None and phase >= 0.45:
                self._procedural_motion_active = False
                self._state.motion = "idle"
                self._state.elapsed = 0.0
            return changed
        if name == "wave":
            cycle = phase % 0.9 if self._motion_request is not None else min(0.9, phase)
            angle = math.sin(math.pi * max(0.0, cycle / 0.9))
            changed = self._set_safe_parameter("ParamBodyAngleZ", 6.0 * angle, (-8.0, 8.0))
            if self._motion_request is None and phase >= 0.9:
                self._procedural_motion_active = False
                self._state.motion = "idle"
                self._state.elapsed = 0.0
            return changed
        if name == "walk":
            angle = math.sin(phase * 12.0)
            return self._set_safe_parameter("ParamBodyAngleZ", 3.0 * angle, (-8.0, 8.0))
        return False

    def _set_expression_name(self, name: str) -> bool:
        if self._model is None:
            return False
        if not self.supports_expression(name):
            return False
        requested = str(name).strip()
        if requested in self._declared_expressions or self._model_path is None:
            setter = getattr(self._model, "SetExpression", None)
            if not callable(setter):
                return False
            try:
                result = setter(requested)
            except Exception:
                return False
            if result is False:
                return False
            self._procedural_expression_active = False
        else:
            if not self._apply_procedural_expression(requested):
                return False
            self._procedural_expression = requested.casefold()
            self._procedural_expression_active = True
        self._state.expression = requested
        return True

    def set_expression(self, name: str) -> bool:
        """立即切换单一表情，并取消尚未完成的表情序列。"""

        self._expression_timeline.cancel()
        self._reset_expression_parameters()
        return self._set_expression_name(name)

    def set_expression_request(self, request: ExpressionRequest) -> Mapping[str, object]:
        """启动多表情序列或参数混合。"""

        if not isinstance(request, ExpressionRequest):
            return {"status": "unavailable", "reason": "expression request is invalid"}
        unsupported = tuple(
            layer.name for layer in request.expressions if not self.supports_expression(layer.name)
        )
        if unsupported or not self.supports_expression(request.restore):
            return {
                "status": "unavailable",
                "reason": "one or more expressions are unsupported",
            }
        self._reset_expression_parameters()
        frame = self._expression_timeline.start(request, current=self._state.expression)
        if not self._set_expression_name(frame.current):
            self._expression_timeline.cancel()
            self._reset_expression_parameters()
            return {"status": "unavailable", "reason": "expression could not start"}
        self._configure_expression_parameters(frame.parameters)
        self._expression_frame_progress = frame.progress
        self._apply_expression_parameters(frame.progress)
        duration = (
            max(item.duration_seconds for item in request.expressions)
            if request.mode == "blend"
            else sum(item.duration_seconds for item in request.expressions)
        )
        return {
            "status": "started",
            "mode": request.mode,
            "expression_count": len(request.expressions),
            "duration_seconds": duration,
            "parameters_applied": True,
        }

    def _play_motion_name(self, name: str) -> bool:
        if self._model is None or self._module is None:
            return False
        if not self.supports_motion(name):
            return False
        requested = str(name).strip()
        resolved = next(
            (
                target
                for alias, target in self._motion_aliases
                if alias.casefold() == requested.casefold()
            ),
            requested,
        )
        if resolved in self._declared_motions:
            start = getattr(self._model, "StartMotion", None)
            priority = getattr(getattr(self._module, "MotionPriority", None), "NORMAL", 2)
            if not callable(start):
                return False
            try:
                result = start(resolved, 0, priority)
            except Exception:
                return False
            if result is False:
                return False
            self._procedural_motion_active = False
            self._formal_motion_name = resolved
            self._formal_motion_elapsed = 0.0
        else:
            self._formal_motion_name = None
            self._formal_motion_elapsed = 0.0
            self._procedural_motion = requested.casefold()
            self._procedural_motion_active = requested.casefold() != "idle"
            # idle 是明确的可执行基线，即使绑定只提供 Update 而没有参数写入。
            if requested.casefold() != "idle" and not callable(
                getattr(self._model, "SetParameterValue", None)
            ):
                return False
            self._state.elapsed = 0.0
            self._apply_procedural_motion(0.0)
        self._state.motion = requested
        return True

    def play_motion(self, name: str) -> bool:
        """立即播放单一动作，并取消带时长的旧动作请求。"""

        self._reset_motion_parameters()
        self._motion_request = None
        self._motion_request_elapsed = 0.0
        return self._play_motion_name(name)

    def play_motion_request(self, request: MotionRequest) -> Mapping[str, object]:
        """播放带持续时间、参数和循环策略的动作。"""

        if not isinstance(request, MotionRequest) or not self.supports_motion(request.name):
            return {"status": "unavailable", "reason": "motion request is unsupported"}
        self._configure_motion_parameters(request)
        if not self._play_motion_name(request.name):
            self._reset_motion_parameters()
            return {"status": "unavailable", "reason": "motion could not start"}
        self._motion_request = request
        self._motion_request_elapsed = 0.0
        return {
            "status": "started",
            "name": request.name,
            "duration_seconds": request.duration_seconds,
            "transition_seconds": request.transition_seconds,
            "loop": request.loop,
            "parameters_applied": True,
        }

    def set_direction(self, direction: str) -> bool:
        """向原生运行时转发可选朝向接口。"""

        if self._model is None:
            return False
        normalized = str(direction or "").strip().upper()
        if normalized not in {"A", "B"}:
            return False
        setter = getattr(self._model, "SetDirection", None)
        if not callable(setter):
            return False
        previous = (self._facing_target, self._pending_direction, self._facing)
        self._facing_target = -1.0 if normalized == "B" else 1.0
        self._facing = self._facing_target
        self._pending_direction = normalized
        if (self._facing_sign < 0) == (self._facing_target < 0):
            if not self._apply_pending_direction():
                self._facing_target, self._pending_direction, self._facing = previous
                return False
        return True

    def _apply_pending_direction(self) -> bool:
        """在朝向姿态经过中性区后提交一次几何转身。"""

        direction = self._pending_direction
        model = self._model
        setter = getattr(model, "SetDirection", None) if model is not None else None
        if not direction or not callable(setter):
            return False
        try:
            setter(direction)
        except Exception:
            return False
        self._pending_direction = None
        return True

    @staticmethod
    def _slew_value(current: float, target: float, seconds: float, max_rate: float) -> float:
        """以有限速率逼近目标，避免跨屏输入产生瞬时目标跳变。"""

        now = _finite_float(current)
        goal = _finite_float(target)
        dt = _clamp(_finite_float(seconds), _MIN_FRAME_SECONDS, _MAX_FRAME_SECONDS)
        rate = max(_MIN_FRAME_SECONDS, _finite_float(max_rate, _TARGET_SLEW_RATE))
        step = rate * dt
        return now + _clamp(goal - now, -step, step)

    @staticmethod
    def _smooth_axis(
        current: float,
        target: float,
        velocity: float,
        seconds: float,
    ) -> tuple[float, float]:
        """用临界阻尼推进单轴注视并返回值/速度。"""

        dt = _clamp(_finite_float(seconds), _MIN_FRAME_SECONDS, _MAX_FRAME_SECONDS)
        current = _finite_float(current)
        # 该数学辅助函数也由回归测试/旧宿主直接调用，允许一般的有限
        # 目标；真正进入模型的光标目标已在 set_cursor_target 中限制为
        # [-1, 1]，参数写入还会经过 _set_safe_parameter。
        target = _clamp(_finite_float(target), -_MAX_TRACKING_SPEED, _MAX_TRACKING_SPEED)
        velocity = _clamp(_finite_float(velocity), -_MAX_TRACKING_SPEED, _MAX_TRACKING_SPEED)
        smooth_time = 0.18
        omega = 2.0 / smooth_time
        x = omega * dt
        decay = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
        change = current - target
        max_change = 12.0 * smooth_time
        change = _clamp(change, -max_change, max_change)
        adjusted_target = current - change
        temporary = (velocity + omega * change) * dt
        next_velocity = (velocity - omega * temporary) * decay
        next_value = adjusted_target + (change + temporary) * decay
        if (target - current) * (next_value - target) < 0:
            next_value = target
            next_velocity = 0.0
        next_value = _clamp(_finite_float(next_value), -_MAX_TRACKING_SPEED, _MAX_TRACKING_SPEED)
        return next_value, _clamp(
            _finite_float(next_velocity), -_MAX_TRACKING_SPEED, _MAX_TRACKING_SPEED
        )

    def hit_parts(self, x: float, y: float, *, include_hidden: bool = False) -> tuple[str, ...]:
        """返回原生运行时命中的部件 ID；缺少精确 API 时 fail-closed。"""

        if self._interaction_locked:
            return ()
        point_x = _finite_float(x, math.nan)
        point_y = _finite_float(y, math.nan)
        if not math.isfinite(point_x) or not math.isfinite(point_y):
            return ()
        model = self._model
        hit_part = getattr(model, "HitPart", None) if model is not None else None
        if not callable(hit_part):
            # 不能把整个透明窗口当作角色；由 Web alpha/精灵像素命中接管。
            return ()
        try:
            raw = hit_part(point_x, point_y, bool(include_hidden))
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return ()
        if isinstance(raw, str):
            values = (raw,)
        else:
            try:
                values = tuple(raw)
            except TypeError:
                return ()
        return tuple(text for text in (str(value or "").strip() for value in values) if text)

    def hit_test(self, x: float, y: float) -> bool:
        """使用 live2d-py 的精确部件命中，拒绝透明画布点击。"""

        return bool(self.hit_parts(x, y))

    def shutdown(self) -> None:
        self._expression_timeline.cancel()
        self._reset_expression_parameters()
        self._reset_motion_parameters()
        self._motion_request = None
        self._motion_request_elapsed = 0.0
        self._stop_all_motions()
        dispose = getattr(self._module, "dispose", None) if self._module is not None else None
        if callable(dispose):
            try:
                dispose()
            except Exception:
                pass
        self._module = None
        self._model = None
        self._parameter_limits = {}
        self._parameter_ids = frozenset()
        self._parameter_limits_model = None
        self._parameter_limits_loaded = False
        self._cursor_target = (0.0, 0.0)
        self._cursor_goal = (0.0, 0.0)
        self._cursor_current = (0.0, 0.0)
        self._cursor_velocity = (0.0, 0.0)
        self._pending_direction = None
        self._facing = 1.0
        self._facing_target = 1.0
        self._facing_sign = 1
        self._facing_pose = 1.0
        self._dragging = False
        self._interaction_locked = False
        self._procedural_expression = "neutral"
        self._procedural_expression_active = False
        self._procedural_motion = "idle"
        self._procedural_motion_active = False
        self._mouth_viseme = "Silence"
        self._mouth_intensity = 0.0
        self._formal_motion_name = None
        self._formal_motion_elapsed = 0.0
        self._state.expression = "neutral"
        self._state.motion = "idle"
        self._state.elapsed = 0.0
        self._state.frame_index = 0
        self._last_transform_valid = True
