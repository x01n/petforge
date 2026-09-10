from __future__ import annotations

import importlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any

import yaml

from core.rendering.actions import ExpressionRequest, ExpressionTimeline, MotionRequest
from core.rendering.readiness import RendererReadyState, RendererReadyStatus
from core.rendering.resources import (
    Live2DMotionSyncProfile,
    live2d_model_resource_version,
    validate_expression3_description,
    validate_model3_description,
    validate_motion3_description,
    validate_motionsync3_description,
)

from .protocol import (
    DEFAULT_EXPRESSION_NAMES,
    DEFAULT_MOTION_NAMES,
    RendererCapabilities,
    RendererState,
)

logger = logging.getLogger(__name__)

_REQUIRED_JAVASCRIPT = (
    "pixi.min.js",
    "live2dcubismcore.min.js",
    "pixi-live2d-display.min.js",
)
# 该补丁包包含旧的双运行时入口，会在仅加载 Cubism4 构建时重新注册不兼容的运行库。
_OPTIONAL_JAVASCRIPT: tuple[str, ...] = ()
_QT_MODULES = (
    "PySide6.QtCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebChannel",
)
_PROCEDURAL_EXPRESSIONS = DEFAULT_EXPRESSION_NAMES
_PROCEDURAL_MOTIONS = DEFAULT_MOTION_NAMES
_PROCEDURAL_MOTION_DURATIONS = {"blink": 0.45, "wave": 0.9}
_ACTION_ALIAS_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")


_WEB_LIVE2D_DEFAULT_FRAME_RATE = 60.0
_WEB_LIVE2D_MIN_FRAME_RATE = 15.0
_WEB_LIVE2D_MAX_FRAME_RATE = 120.0
_WEB_LIVE2D_DEFAULT_GEOMETRY_AUDIT_HZ = 30.0
_WEB_LIVE2D_MIN_GEOMETRY_AUDIT_HZ = 1.0
_WEB_LIVE2D_MAX_GEOMETRY_AUDIT_HZ = 60.0
_WEB_LIVE2D_MAX_VIEWPORT_EDGE = 16_384


def _normalize_render_setting(
    value: object,
    *,
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """校验渲染性能设置，拒绝布尔值、非有限值和越界值。"""

    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite number")
    if normalized < minimum or normalized > maximum:
        raise ValueError(f"{name} must be between {minimum:g} and {maximum:g}")
    return normalized


def _expression_request_payload(request: ExpressionRequest) -> dict[str, object]:
    """把已校验的表情请求转换为页面可执行的纯 JSON 数据。"""

    return {
        "expressions": [
            {
                "name": layer.name,
                "weight": layer.weight,
                "duration_seconds": layer.duration_seconds,
                "transition_seconds": layer.transition_seconds,
                "parameters": dict(layer.parameters),
            }
            for layer in request.expressions
        ],
        "mode": request.mode,
        "loop": request.loop,
        "restore": request.restore,
    }


def _expression_request_duration(request: ExpressionRequest) -> float:
    """返回序列或混合模式的实际保持时长。"""

    durations = tuple(layer.duration_seconds for layer in request.expressions)
    return max(durations) if request.mode == "blend" else sum(durations)


def _positive_finite(value: object) -> bool:
    """只接受大于零的有限数值，不允许布尔值伪装为尺寸。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and number > 0


def _motion_request_payload(request: MotionRequest) -> dict[str, object]:
    """把已校验的动作请求转换为页面可执行的纯 JSON 数据。"""

    return {
        "name": request.name,
        "duration_seconds": request.duration_seconds,
        "transition_seconds": request.transition_seconds,
        "loop": request.loop,
        "parameters": dict(request.parameters),
    }


def _known_action_name(
    value: str,
    declared: tuple[str, ...],
    procedural: tuple[str, ...],
    aliases: tuple[tuple[str, str], ...] = (),
) -> bool:
    """判断动作请求是否属于当前模型或明确的程序化兼容集合。

    正式 Cubism 名称必须精确匹配；只有显式 YAML 别名和内置语义词才
    允许大小写归一化。这样不会把任意参数名/拼写错误当成成功动作。
    """

    requested = str(value or "").strip()
    if not requested:
        return False
    if requested in declared:
        return True
    folded = requested.casefold()
    if any(alias.casefold() == folded for alias, _target in aliases):
        return True
    # 若正式名称只在大小写上不同，必须要求其精确拼写，不能被过程动作
    # 抢先接管；这也避免同一动作在控制台出现两个看似相同的入口。
    if any(name.casefold() == folded for name in declared):
        return False
    return folded in {name.casefold() for name in procedural}


def _load_module(
    name: str,
    importer: Callable[[str], Any] | None = None,
) -> Any | None:
    """动态加载可选模块；导入失败只影响对应能力。"""

    load = importlib.import_module if importer is None else importer
    try:
        return load(name)
    except (
        ImportError,
        ModuleNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        SystemError,
    ):
        return None


def _find_valid_model(
    model_root: Path,
    model_path: str | Path | None = None,
) -> Path | None:
    """选择模型目录内的显式描述，未指定时使用排序后的首个描述。"""

    if not model_root.is_dir():
        return None
    if model_path is not None:
        try:
            # 配置和模型目录选择器都使用资源根相对路径（例如
            # ``live2d/model/foo/foo.model3.json``）。绝对路径只在资源根
            # 内有效；不能把输入拼到 model_root 后再“猜”另一种路径语义。
            resource_root = model_root.parent.parent.resolve()
            raw = Path(model_path).expanduser()
            candidate = raw.resolve() if raw.is_absolute() else (resource_root / raw).resolve()
            candidate.relative_to(model_root.resolve())
        except (OSError, RuntimeError, TypeError, ValueError):
            return None
        if not candidate.name.endswith(".model3.json"):
            return None
        return candidate if validate_model3_description(candidate) is not None else None
    model_paths = sorted(model_root.rglob("*.model3.json"))
    for path in model_paths:
        if validate_model3_description(path) is not None:
            return path.resolve()
    return None


def _declared_model_actions(model_path: Path | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """读取 model3 中通过 exp3/motion3 契约校验的动作声明。"""

    if model_path is None:
        return (), ()
    try:
        value = json.loads(model_path.read_text(encoding="utf-8"))
        references = value.get("FileReferences") if isinstance(value, dict) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return (), ()
    if not isinstance(references, Mapping):
        return (), ()

    def action_path(reference: object) -> Path | None:
        if not isinstance(reference, str) or not reference.strip():
            return None
        reference_path = Path(reference)
        if reference_path.is_absolute() or reference_path.anchor or "\x00" in reference:
            return None
        try:
            resolved = (model_path.parent / reference_path).resolve()
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
            expression_path = action_path(item.get("File"))
            if (
                name
                and expression_path is not None
                and validate_expression3_description(expression_path)
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
                and (motion_path := action_path(item.get("File"))) is not None
                and validate_motion3_description(motion_path)
                for item in entries
            ):
                motions.append(group_name)
    return tuple(expressions), tuple(motions)


def _declared_motion_durations(model_path: Path | None) -> tuple[tuple[str, float], ...]:
    """读取正式 motion3 的单循环时长，供显式动作自动收尾。"""

    if model_path is None:
        return ()
    try:
        descriptor = json.loads(model_path.read_text(encoding="utf-8"))
        references = descriptor.get("FileReferences") if isinstance(descriptor, Mapping) else None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return ()
    raw_motions = references.get("Motions") if isinstance(references, Mapping) else None
    if not isinstance(raw_motions, Mapping):
        return ()
    durations: list[tuple[str, float]] = []
    for group, entries in raw_motions.items():
        group_name = str(group or "").strip()
        if not group_name or not isinstance(entries, list):
            continue
        duration: float | None = None
        for item in entries:
            if not isinstance(item, Mapping):
                continue
            reference = item.get("File")
            if not isinstance(reference, str) or not reference.strip():
                continue
            path = (model_path.parent / reference).resolve()
            try:
                path.relative_to(model_path.parent.resolve())
                motion = json.loads(path.read_text(encoding="utf-8"))
                meta = motion.get("Meta") if isinstance(motion, Mapping) else None
                raw_duration: object = meta.get("Duration") if isinstance(meta, Mapping) else None
                if isinstance(raw_duration, bool) or not isinstance(
                    raw_duration, (int, float, str)
                ):
                    continue
                value = float(raw_duration)
            except (
                OSError,
                UnicodeError,
                ValueError,
                TypeError,
                OverflowError,
                json.JSONDecodeError,
            ):
                continue
            if math.isfinite(value) and value > 0.0:
                duration = value
                break
        if duration is not None:
            durations.append((group_name, duration))
    return tuple(durations)


def _model_motion_aliases(
    resource_root: Path,
    model_path: Path | None,
    declared_motions: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """读取当前模型的显式语义动作映射。"""

    if model_path is None or not model_path.name.endswith(".model3.json"):
        return ()
    prefix = model_path.name[: -len(".model3.json")]
    sidecar = model_path.with_name(f"{prefix}.actions.yaml")
    value: object = None
    if _nonempty_file(sidecar):
        try:
            value = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, yaml.YAMLError):
            value = None
    else:
        config_path = resource_root.parent / "config" / "live2d-actions.yaml"
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
    declared = set(declared_motions)
    aliases: list[tuple[str, str]] = []
    for raw_alias, raw_target in motions.items():
        alias = str(raw_alias or "").strip().lower()
        target = str(raw_target or "").strip()
        if (
            _ACTION_ALIAS_PATTERN.fullmatch(alias)
            and target in declared
            and alias not in {item[0] for item in aliases}
        ):
            aliases.append((alias, target))
    return tuple(sorted(aliases))


def _model_motion_sync_profile(
    model_path: Path | None,
) -> Live2DMotionSyncProfile | None:
    """读取与当前 model3 同名且通过严格校验的 MotionSync 配置。"""

    if model_path is None or not model_path.name.endswith(".model3.json"):
        return None
    prefix = model_path.name[: -len(".model3.json")]
    return validate_motionsync3_description(model_path.with_name(f"{prefix}.motionsync3.json"))


def _web_assets(resource_root: Path) -> tuple[Path, ...]:
    javascript_root = resource_root / "live2d" / "js"
    return tuple(
        path
        for name in (*_REQUIRED_JAVASCRIPT, *_OPTIONAL_JAVASCRIPT)
        for path in (javascript_root / name,)
        if _nonempty_file(path)
    )


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


@dataclass(frozen=True)
class WebLive2DProbe:
    """Web Live2D 后端探测结果。"""

    resource_root: Path
    model_path: Path | None
    javascript_assets: tuple[Path, ...]
    qt_modules: tuple[str, ...]
    missing_qt_modules: tuple[str, ...]
    message: str
    warnings: tuple[str, ...] = ()
    expressions: tuple[str, ...] = ()
    motions: tuple[str, ...] = ()
    motion_aliases: tuple[tuple[str, str], ...] = ()
    motion_sync: Live2DMotionSyncProfile | None = None
    motion_durations: tuple[tuple[str, float], ...] = ()
    resource_version: str = ""

    @property
    def available(self) -> bool:
        """返回是否可以创建 WebEngine Live2D 视图。"""

        asset_names = {path.name for path in self.javascript_assets}
        return bool(
            self.model_path
            and all(name in asset_names for name in _REQUIRED_JAVASCRIPT)
            and not self.missing_qt_modules
        )

    def as_capabilities(self) -> RendererCapabilities:
        """转换为现有渲染器协议的能力对象。"""

        motion_names: list[str] = []
        motion_keys: set[str] = set()
        for name in (
            *(alias for alias, _target in self.motion_aliases),
            *self.motions,
            *_PROCEDURAL_MOTIONS,
        ):
            text = str(name or "").strip()
            key = text.casefold()
            if text and key not in motion_keys:
                motion_names.append(text)
                motion_keys.add(key)
        # 程序化动作是正式 model3 存在时仍可用的安全回退（例如 walk/wave），
        # 也必须出现在控制台能力列表中；否则下拉框会只显示原始 A/B 组，
        # 用户看不到实际可执行的语义动作。
        expression_names: list[str] = []
        expression_keys: set[str] = set()
        for name in (*self.expressions, *_PROCEDURAL_EXPRESSIONS):
            text = str(name or "").strip()
            key = text.casefold()
            if text and key not in expression_keys:
                expression_names.append(text)
                expression_keys.add(key)
        return RendererCapabilities(
            backend="web_live2d",
            available=self.available,
            model_path=self.model_path,
            expressions=tuple(expression_names) or _PROCEDURAL_EXPRESSIONS,
            motions=tuple(motion_names) or _PROCEDURAL_MOTIONS,
            message=self.message,
        )


def probe_web_live2d(
    resource_root: str | Path,
    *,
    model_path: str | Path | None = None,
    importer: Callable[[str], Any] | None = None,
) -> WebLive2DProbe:
    """探测模型、前端资源和 Qt WebEngine/WebChannel。"""

    root = Path(resource_root).expanduser().resolve()
    # 只验证当前前端实际使用的 model3 字段，避免把目录存在误认为可用模型。
    model_root = root / "live2d" / "model"
    model3_paths = sorted(model_root.rglob("*.model3.json")) if model_root.is_dir() else ()
    legacy_model_paths = sorted(model_root.rglob("*.model.json")) if model_root.is_dir() else ()
    selected_model_path = _find_valid_model(model_root, model_path)
    warnings: list[str] = []
    if legacy_model_paths:
        warnings.append("Live2D .model.json descriptors are unsupported without an exact schema")
    invalid_model_count = sum(validate_model3_description(path) is None for path in model3_paths)
    if invalid_model_count:
        warnings.append(f"{invalid_model_count} Live2D .model3.json descriptor(s) are invalid")
    javascript_assets = _web_assets(root)
    declared_expressions, declared_motions = _declared_model_actions(selected_model_path)
    motion_durations = _declared_motion_durations(selected_model_path)
    motion_aliases = _model_motion_aliases(root, selected_model_path, declared_motions)
    motion_sync = _model_motion_sync_profile(selected_model_path)
    model_files = (
        validate_model3_description(selected_model_path)
        if selected_model_path is not None
        else None
    )
    resource_version = live2d_model_resource_version(model_files) if model_files is not None else ""
    if selected_model_path is not None and selected_model_path.name.endswith(".model3.json"):
        prefix = selected_model_path.name[: -len(".model3.json")]
        motion_sync_path = selected_model_path.with_name(f"{prefix}.motionsync3.json")
        if _nonempty_file(motion_sync_path) and motion_sync is None:
            warnings.append("Live2D motionsync3 descriptor is invalid and was ignored")
    asset_names = {path.name for path in javascript_assets}
    modules = tuple(name for name in _QT_MODULES if _load_module(name, importer))
    missing_qt_modules = tuple(name for name in _QT_MODULES if name not in modules)

    if selected_model_path is None:
        message = warnings[0] if warnings else "usable Live2D model3 descriptor is unavailable"
    elif not all(name in asset_names for name in _REQUIRED_JAVASCRIPT):
        message = "Live2D JavaScript assets are incomplete"
    elif missing_qt_modules:
        message = "Qt WebEngine/WebChannel is unavailable"
    else:
        message = "Web Live2D model, JavaScript assets, and Qt bridge are available"

    return WebLive2DProbe(
        resource_root=root,
        model_path=selected_model_path,
        javascript_assets=javascript_assets,
        qt_modules=modules,
        missing_qt_modules=missing_qt_modules,
        message=message,
        warnings=tuple(warnings),
        expressions=declared_expressions,
        motions=declared_motions,
        motion_aliases=motion_aliases,
        motion_durations=motion_durations,
        motion_sync=motion_sync,
        resource_version=resource_version,
    )


def qt_webengine_available(
    *,
    importer: Callable[[str], Any] | None = None,
) -> bool:
    """只探测 Qt WebEngine 相关模块，不创建 GUI 对象。"""

    return all(_load_module(name, importer) is not None for name in _QT_MODULES)


class WebLive2DRenderer:
    """将 Python 状态桥接到 QWebEngineView 中的 Live2D 页面。

    ``initialize`` 必须在 Qt GUI 线程中调用。模型声明的 ``exp3``/
    ``motion3`` 由 Cubism 动作管理器执行；缺失的语义动作才使用经过当前
    MOC3 顶点探测的兼容参数和页面变换。可选的同名 ``actions.yaml`` 只做
    显式语义别名，不从文件名猜测动作含义。桌面启动可用 ``natural_layout``
    沿用独立浏览器页面的画布比例；默认测试/嵌入实例保持动态网格拟合。
    """

    def __init__(
        self,
        resource_root: str | Path,
        *,
        model_path: str | Path | None = None,
        importer: Callable[[str], Any] | None = None,
        failure_callback: Callable[[str], None] | None = None,
        natural_layout: bool = False,
        frame_rate: float = _WEB_LIVE2D_DEFAULT_FRAME_RATE,
        geometry_audit_hz: float = _WEB_LIVE2D_DEFAULT_GEOMETRY_AUDIT_HZ,
    ) -> None:
        self.resource_root = Path(resource_root).expanduser().resolve()
        self._importer = importer
        self._natural_layout = bool(natural_layout)
        self._frame_rate = _normalize_render_setting(
            frame_rate,
            name="frame_rate",
            default=_WEB_LIVE2D_DEFAULT_FRAME_RATE,
            minimum=_WEB_LIVE2D_MIN_FRAME_RATE,
            maximum=_WEB_LIVE2D_MAX_FRAME_RATE,
        )
        self._geometry_audit_hz = _normalize_render_setting(
            geometry_audit_hz,
            name="geometry_audit_hz",
            default=_WEB_LIVE2D_DEFAULT_GEOMETRY_AUDIT_HZ,
            minimum=_WEB_LIVE2D_MIN_GEOMETRY_AUDIT_HZ,
            maximum=_WEB_LIVE2D_MAX_GEOMETRY_AUDIT_HZ,
        )
        self._probe = probe_web_live2d(
            self.resource_root,
            model_path=model_path,
            importer=importer,
        )
        self._capabilities = self._probe.as_capabilities()
        self._state = RendererState()
        self._view: Any | None = None
        self._channel: Any | None = None
        self._bridge: Any | None = None
        self._failure_callback = failure_callback
        self._failure_reported = False
        self._speech_text = ""
        self._speech_mood = "neutral"
        self._speech_visible = False
        self._pending_interaction_feedback: tuple[dict[str, str], bool] | None = None
        self._interaction_feedback_visible = False
        self._page_ready = False
        self._model_ready = False
        self._page_discarded = False
        # 页面只有在 QWebChannel 已完成握手并报告真实模型就绪后，才可以
        # 取代 Qt 原生事件过滤器。初始化/加载失败期间保留原生回退，避免
        # 仅凭 ``set_interaction_callback`` 方法存在就误判桥接已连通。
        self._page_bridge_ready = False
        self._page_bridge_ready_callback: Callable[[bool], object] | None = None
        self._pending_expression = ""
        self._pending_expression_request: dict[str, object] | None = None
        self._expression_timeline = ExpressionTimeline()
        self._pending_motion = ""
        self._pending_motion_request: dict[str, object] | None = None
        self._motion_request: MotionRequest | None = None
        self._motion_request_elapsed = 0.0
        self._pending_direction = ""
        # 页面正式 motion3 到时会通过 WebChannel 回报；保留当前正式组
        # 用于同步 Python 状态卡，避免画面已回到 idle 而控制台仍显示旧动作。
        self._formal_motion_name: str | None = None
        self._pending_speech: tuple[str, str, bool, bool] | None = None
        self._speech_speaking = False
        self._pending_mouth_viseme: tuple[str, float] | None = None
        # QWebEngine 的 Chromium 视口会吞掉顶层 QWidget 的按下/释放事件。
        # WebPetHost 在安装交互时注入回调，页面通过 QWebChannel 把指针事件
        # 转回 Qt 宿主；没有宿主时保持空回调，不改变渲染器的独立可测试性。
        self._interaction_callback: Callable[[Mapping[str, object]], object] | None = None
        # 普通单击与拖动/双击分开暴露；宿主负责把局部坐标转换为稳定的
        # ``upper``/``lower_left``/``lower_right`` 分区，再决定低风险反应。
        self._click_callback: Callable[[Mapping[str, object]], object] | None = None
        self._interaction_locked = False
        # 拖动期间冻结页面动作/物理时钟；窗口位置仍由 Qt 宿主合并更新，
        # 释放后从新的时间基线继续，避免 WebGL 帧与原生窗口移动交错。
        self._dragging = False
        self._last_cursor_target: tuple[int, int, int, int] | None = None
        # 模型运行期切换采用“候选页内预加载 -> 成功回执 -> Python 提交”
        # 协议。未收到与当前代次匹配的回执前，旧 probe/能力/Shape 均保持
        # 不变；这样异步加载失败或迟到回调不会污染当前桌宠。
        self._model_reload_generation = 0
        self._pending_model_reload: tuple[int, WebLive2DProbe] | None = None
        self._model_reload_phase = "idle"
        self._model_reload_deadline = 0.0
        self._model_reload_timeout_seconds = 15.0
        self._clock: Callable[[], float] = monotonic
        self._model_reload_status: dict[str, object] = {
            "status": "idle",
            "model": self._probe.model_path,
        }
        self._model_reload_callback: Callable[[Mapping[str, object]], object] | None = None
        self._ready_probe_generation = 0
        self._ready_probe_pending = False
        self._ready_probe_next_at = 0.0
        self._ready_status = RendererReadyStatus(
            backend="web_live2d",
            state=RendererReadyState.PENDING,
            reason_code="page_pending",
        )

    @property
    def capabilities(self) -> RendererCapabilities:
        return self._capabilities

    @property
    def state(self) -> RendererState:
        return self._state

    @property
    def probe(self) -> WebLive2DProbe:
        return self._probe

    @property
    def view(self) -> Any | None:
        return self._view

    @property
    def page_bridge_ready(self) -> bool:
        """返回 WebChannel 指针桥是否已完成真实页面握手。"""

        return self._page_bridge_ready

    @property
    def model_ready(self) -> bool:
        """返回页面是否已经加载真实 Cubism 模型。"""

        return self._model_ready

    @property
    def page_ready(self) -> bool:
        """返回 Web 页面是否仍可接收 JavaScript 调用。"""

        return self._page_ready

    @property
    def frame_rate(self) -> float:
        """返回页面绘制时钟的目标帧率。"""

        return self._frame_rate

    @property
    def geometry_audit_hz(self) -> float:
        """返回动态几何审计的目标频率。"""

        return self._geometry_audit_hz

    @property
    def performance_settings(self) -> Mapping[str, float]:
        """返回可热更新的渲染性能设置快照。"""

        return {
            "frame_rate": self._frame_rate,
            "geometry_audit_hz": self._geometry_audit_hz,
        }

    def set_frame_rate(self, value: float) -> bool:
        """更新页面绘制帧率，并让下一次时钟 tick 立即生效。"""

        try:
            normalized = _normalize_render_setting(
                value,
                name="frame_rate",
                default=self._frame_rate,
                minimum=_WEB_LIVE2D_MIN_FRAME_RATE,
                maximum=_WEB_LIVE2D_MAX_FRAME_RATE,
            )
        except ValueError:
            return False
        self._frame_rate = normalized
        if self._view is not None:
            self._run_javascript(
                f"window.meapetLive2D && window.meapetLive2D.setFrameRate({normalized!r});"
            )
        return True

    def set_geometry_audit_hz(self, value: float) -> bool:
        """更新动态几何审计频率，并使缓存按新间隔重新校验。"""

        try:
            normalized = _normalize_render_setting(
                value,
                name="geometry_audit_hz",
                default=self._geometry_audit_hz,
                minimum=_WEB_LIVE2D_MIN_GEOMETRY_AUDIT_HZ,
                maximum=_WEB_LIVE2D_MAX_GEOMETRY_AUDIT_HZ,
            )
        except ValueError:
            return False
        self._geometry_audit_hz = normalized
        if self._view is not None:
            self._run_javascript(
                f"window.meapetLive2D && window.meapetLive2D.setGeometryAuditHz({normalized!r});"
            )
        return True

    @property
    def model_reload_status(self) -> Mapping[str, object]:
        """返回最近一次模型切换的不可变快照。"""

        self._expire_pending_model_reload()
        return dict(self._model_reload_status)

    @property
    def ready_status(self) -> RendererReadyStatus:
        """返回固定字段、无本地路径的真实帧就绪状态。"""

        return self._ready_status

    def request_ready_status(self) -> RendererReadyStatus:
        """异步探测桥接、几何与当前 WebGL 原始 alpha 帧。"""

        if self._page_discarded or self._view is None:
            return self._set_ready_status(
                RendererReadyState.CLOSED if self._page_discarded else RendererReadyState.PENDING,
                reason_code="page_closed" if self._page_discarded else "view_pending",
            )
        if not self._page_ready:
            return self._set_ready_status(
                RendererReadyState.PENDING,
                reason_code="page_pending",
            )
        if not self._page_bridge_ready:
            return self._set_ready_status(
                RendererReadyState.PENDING,
                reason_code="bridge_pending",
            )
        if not self._model_ready:
            return self._set_ready_status(
                RendererReadyState.PENDING,
                reason_code="model_pending",
            )
        if self._ready_status.ready or self._ready_probe_pending:
            return self._ready_status
        if self._clock() < self._ready_probe_next_at:
            return self._ready_status

        self._ready_probe_generation += 1
        generation = self._ready_probe_generation
        self._ready_probe_pending = True
        self._ready_probe_next_at = self._clock() + 0.2
        self._ready_status = RendererReadyStatus(
            backend="web_live2d",
            state=RendererReadyState.PENDING,
            bridge_ready=True,
            model_ready=True,
            reason_code="frame_pending",
            generation=generation,
        )
        script = """(function() {
          const api = window.meapetLive2D;
          if (!api || typeof api.debug !== 'function' ||
              typeof api.inputMask !== 'function') return null;
          return JSON.stringify({debug: api.debug(), mask: api.inputMask({probeAlpha: true})});
        })()"""
        try:
            page = self._view.page()
            page.runJavaScript(
                script,
                lambda value, current=generation: self._on_ready_probe_result(current, value),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._ready_probe_pending = False
            return self._set_ready_status(
                RendererReadyState.PENDING,
                reason_code="probe_dispatch_failed",
                generation=generation,
            )
        return self._ready_status

    def _set_ready_status(
        self,
        state: RendererReadyState,
        *,
        geometry_valid: bool = False,
        frame_visible: bool = False,
        alpha_nonempty: bool = False,
        reason_code: str = "",
        generation: int | None = None,
    ) -> RendererReadyStatus:
        """提交不包含资源路径或异常正文的 Web READY 状态。"""

        self._ready_status = RendererReadyStatus(
            backend="web_live2d",
            state=state,
            bridge_ready=self._page_bridge_ready,
            model_ready=self._model_ready,
            geometry_valid=geometry_valid,
            frame_visible=frame_visible,
            alpha_nonempty=alpha_nonempty,
            reason_code=reason_code,
            generation=(self._ready_probe_generation if generation is None else max(0, generation)),
        )
        return self._ready_status

    def _invalidate_ready_status(self, reason_code: str) -> None:
        """使迟到探针失效并回到可重试的 pending 状态。"""

        self._ready_probe_generation += 1
        self._ready_probe_pending = False
        self._ready_probe_next_at = 0.0
        self._set_ready_status(RendererReadyState.PENDING, reason_code=reason_code)

    def _on_ready_probe_result(self, generation: int, payload: object) -> None:
        """只接受当前代次且同时满足四项 Web 真实帧证据的回执。"""

        if generation != self._ready_probe_generation or not self._ready_probe_pending:
            return
        self._ready_probe_pending = False
        value: object = payload
        if isinstance(payload, str):
            try:
                value = json.loads(payload)
            except (TypeError, ValueError, json.JSONDecodeError):
                value = None
        debug = value.get("debug") if isinstance(value, Mapping) else None
        mask = value.get("mask") if isinstance(value, Mapping) else None
        geometry_valid = bool(
            isinstance(debug, Mapping)
            and debug.get("geometryValid") is True
            and debug.get("renderOrderValid") is True
            and debug.get("viewportStale") is False
        )
        frame_visible = bool(
            isinstance(debug, Mapping)
            and debug.get("modelVisible") is True
            and _positive_finite(debug.get("width"))
            and _positive_finite(debug.get("height"))
        )
        data = mask.get("data") if isinstance(mask, Mapping) else None
        alpha_nonempty = bool(
            isinstance(mask, Mapping)
            and mask.get("status") == "ok"
            and mask.get("raw_alpha_nonempty") is True
            and isinstance(data, str)
            and data.startswith("data:image/png;base64,")
            and len(data) > len("data:image/png;base64,")
        )
        ready = bool(
            self._page_bridge_ready
            and self._model_ready
            and geometry_valid
            and frame_visible
            and alpha_nonempty
        )
        self._set_ready_status(
            RendererReadyState.READY if ready else RendererReadyState.PENDING,
            geometry_valid=geometry_valid,
            frame_visible=frame_visible,
            alpha_nonempty=alpha_nonempty,
            reason_code="" if ready else "frame_evidence_pending",
            generation=generation,
        )
        if ready:
            self._commit_verified_model_reload()

    def set_model_reload_callback(
        self,
        callback: Callable[[Mapping[str, object]], object] | None,
    ) -> None:
        """设置模型切换回执回调；宿主销毁时传入 ``None``。"""

        self._model_reload_callback = callback

    def set_page_bridge_ready_callback(
        self,
        callback: Callable[[bool], object] | None,
    ) -> None:
        """设置页面桥就绪状态回调；宿主销毁时传入 ``None``。"""

        self._page_bridge_ready_callback = callback

    def _set_page_bridge_ready(self, ready: bool) -> None:
        value = bool(ready)
        if self._page_bridge_ready == value:
            return
        self._page_bridge_ready = value
        if not value:
            if self._page_discarded:
                self._ready_probe_generation += 1
                self._ready_probe_pending = False
                self._set_ready_status(RendererReadyState.CLOSED, reason_code="page_closed")
            else:
                self._invalidate_ready_status("bridge_pending")
        callback = self._page_bridge_ready_callback
        if callback is None:
            return
        try:
            callback(value)
        except Exception:
            logger.exception("Web Live2D page bridge state callback failed")

    def set_interaction_callback(
        self,
        callback: Callable[[Mapping[str, object]], object] | None,
    ) -> None:
        """设置页面指针事件回调；宿主销毁时传入 ``None`` 清理引用。"""

        self._interaction_callback = callback

    def set_click_callback(
        self,
        callback: Callable[[Mapping[str, object]], object] | None,
    ) -> None:
        """设置普通单击回调；传入 ``None`` 可在宿主销毁时解除引用。"""

        self._click_callback = callback

    def set_interaction_locked(self, locked: bool) -> bool:
        """锁定页面鼠标互动，但不停止渲染和光标目标更新。"""

        self._interaction_locked = bool(locked)
        if self._view is None:
            return True
        self._run_javascript(
            "window.meapetLive2D && window.meapetLive2D.setInteractionLocked(%s);"
            % ("true" if self._interaction_locked else "false")
        )
        return True

    def set_dragging(self, dragging: bool) -> bool:
        """冻结/恢复页面动作时钟，不改变窗口锁定语义。"""

        value = bool(dragging)
        self._dragging = value
        if self._view is None:
            return True
        self._run_javascript(
            "window.meapetLive2D && window.meapetLive2D.setDragging(%s);"
            % ("true" if value else "false")
        )
        return True

    def reload_model(
        self,
        model_path: str | Path | None,
        *,
        force: bool = False,
    ) -> Mapping[str, object]:
        """请求页面内原子切换到另一个已校验的 model3。

        目标文件先在 Python 侧执行完整引用校验，再交给页面预加载。页面
        只有在候选模型已创建、几何/绘制顺序通过检查后才交换 ``liveModel``；
        失败回执不会改动当前模型。调用期间禁止拖动和并发请求，避免 Shape
        回读、窗口几何和 WebGL 代次交叉写回。
        """

        probe = probe_web_live2d(
            self.resource_root,
            model_path=model_path,
            importer=self._importer,
        )
        if not probe.available or probe.model_path is None:
            result: dict[str, object] = {
                "status": "unavailable",
                "reason": probe.message or "requested Live2D model is unavailable",
            }
            self._model_reload_status = dict(result)
            return result
        self._expire_pending_model_reload()
        current = self._probe.model_path
        same_path = current is not None and probe.model_path.resolve() == current.resolve()
        same_version = bool(
            self._probe.resource_version
            and probe.resource_version
            and self._probe.resource_version == probe.resource_version
        )
        if same_path and same_version and not force:
            result = {"status": "unchanged", "model": probe.model_path}
            self._model_reload_status = dict(result)
            return result
        if self._pending_model_reload is not None:
            result = {
                "status": "busy",
                "model": probe.model_path,
                "reason": "another Live2D model reload is in progress",
            }
            self._model_reload_status = dict(result)
            return result
        if self._view is None or not self._page_ready or not self._model_ready:
            result = {
                "status": "unavailable",
                "model": probe.model_path,
                "reason": "Web Live2D page is not ready for model reload",
            }
            self._model_reload_status = dict(result)
            return result
        if self._dragging:
            result = {
                "status": "busy",
                "model": probe.model_path,
                "reason": "model reload is blocked while dragging",
            }
            self._model_reload_status = dict(result)
            return result

        self._model_reload_generation += 1
        request_id = self._model_reload_generation
        self._pending_model_reload = (request_id, probe)
        self._model_reload_phase = "loading"
        self._model_reload_deadline = self._clock() + self._model_reload_timeout_seconds
        result = {
            "status": "pending",
            "request_id": request_id,
            "model": probe.model_path,
        }
        self._model_reload_status = dict(result)
        payload = self._model_reload_payload(probe)
        script = (
            "window.meapetLive2D && typeof window.meapetLive2D.reloadModel === 'function' "
            "&& window.meapetLive2D.reloadModel("
            f"{json.dumps(payload['url'], ensure_ascii=False)},"
            f"{json.dumps(payload, ensure_ascii=False)},"
            f"{request_id});"
        )
        try:
            if self._run_javascript(script) is False:
                raise RuntimeError("failed to dispatch model reload")
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Web Live2D model reload dispatch failed: %s", type(exc).__name__)
            self._pending_model_reload = None
            self._model_reload_phase = "idle"
            self._model_reload_deadline = 0.0
            result = {
                "status": "failed",
                "request_id": request_id,
                "model": probe.model_path,
                "reason": "failed to dispatch model reload",
            }
            self._model_reload_status = dict(result)
            return result
        return result

    def _expire_pending_model_reload(self) -> None:
        """在页面漏回执时按单调时钟结束事务并恢复自治。"""

        if self._pending_model_reload is None or self._model_reload_deadline <= 0:
            return
        if self._clock() < self._model_reload_deadline:
            return
        self._fail_pending_model_reload("Live2D model reload timed out")

    def reload_resources(
        self,
        resource_root: str | Path,
        *,
        model_path: str | Path | None = None,
        sprite_scale: float | None = None,
        force: bool = False,
    ) -> Mapping[str, object]:
        """在当前资源根内复用页内模型交换；资源根变化要求重启宿主。"""

        self._expire_pending_model_reload()
        if self._dragging:
            result: dict[str, object] = {
                "status": "busy",
                "reason": "resource reload is blocked while dragging",
            }
            self._model_reload_status = dict(result)
            return result
        if self._pending_model_reload is not None:
            result = {
                "status": "busy",
                "reason": "another Live2D model reload is in progress",
            }
            self._model_reload_status = dict(result)
            return result
        try:
            target_root = Path(resource_root).expanduser().resolve()
        except (OSError, RuntimeError, TypeError, ValueError):
            result = {"status": "unavailable", "reason": "resource_root is invalid"}
            self._model_reload_status = dict(result)
            return result
        if target_root != self.resource_root:
            result = {
                "status": "restart_required",
                "reason": "Web Live2D resource_root change requires a new page lifecycle",
            }
            self._model_reload_status = dict(result)
            return result
        selected_model = model_path if model_path is not None else self._probe.model_path
        reload_result = (
            self.reload_model(selected_model, force=True)
            if force
            else self.reload_model(selected_model)
        )
        if isinstance(reload_result, Mapping):
            value = dict(reload_result)
            if sprite_scale is not None:
                value["sprite_scale_applied"] = False
            return value
        return {"status": "failed", "reason": "model reload returned an invalid result"}

    def _model_reload_payload(self, probe: WebLive2DProbe) -> dict[str, object]:
        """构造页面候选模型元数据；所有路径和动作来自已校验 probe。"""

        descriptor = probe.model_path
        model_url = ""
        if descriptor is not None:
            try:
                model_url = descriptor.relative_to(self.resource_root).as_posix()
            except ValueError:
                model_url = ""
        if model_url and probe.resource_version:
            model_url = f"{model_url}?meapet_resource_version={probe.resource_version}"
        return {
            "url": model_url,
            "part_names": list(self._model_part_names(descriptor)),
            "part_labels": list(self._model_part_labels(descriptor)),
            "expressions": list(probe.expressions),
            "motions": list(probe.motions),
            "motion_aliases": dict(probe.motion_aliases),
            "motion_durations": dict(probe.motion_durations),
            "motion_sync": probe.motion_sync.as_mapping() if probe.motion_sync else None,
        }

    def _on_model_reload_finished(self, payload: str) -> None:
        """仅接受当前请求代次的页面回执，并原子提交 Python probe。"""

        try:
            value = json.loads(str(payload or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.debug("invalid Web Live2D model reload payload")
            return
        if not isinstance(value, Mapping):
            return
        raw_request_id = value.get("request_id")
        if isinstance(raw_request_id, bool) or not isinstance(raw_request_id, int):
            return
        pending = self._pending_model_reload
        if pending is None or pending[0] != raw_request_id:
            # 旧页面或旧加载事务的迟到回执必须丢弃，不能覆盖当前能力和 Shape。
            return
        _request_id, probe = pending
        status = str(value.get("status", "") or "").strip().lower()
        if status == "prepared" and self._model_reload_phase == "loading":
            self._model_reload_phase = "prepared"
            result: dict[str, object] = {
                "status": "pending",
                "phase": "prepared",
                "request_id": raw_request_id,
                "model": probe.model_path,
            }
            self._model_reload_status = dict(result)
            self._run_javascript(
                "window.meapetLive2D && "
                "typeof window.meapetLive2D.acceptModelReload === 'function' && "
                f"window.meapetLive2D.acceptModelReload({raw_request_id});"
            )
            return
        if status == "available" and self._model_reload_phase == "prepared":
            # 页面已经显示新模型，但仍保留旧模型和完整运行时快照。只有新的
            # WebGL 帧同时通过几何、可见性与原始 alpha 证据后，Python 才
            # finalize 并向控制台发出成功终态；超时路径仍可完整 rollback。
            self._model_reload_phase = "verifying"
            self._model_reload_status = {
                "status": "pending",
                "phase": "verifying",
                "request_id": raw_request_id,
                "model": probe.model_path,
            }
            self._invalidate_ready_status("model_reload_frame_pending")
            self.request_ready_status()
            return

        self._fail_pending_model_reload(
            str(value.get("reason", "Live2D model reload was rejected") or "")
        )

    def _commit_verified_model_reload(self) -> None:
        """在新模型真实帧 READY 后提交 probe，并释放旧页面模型。"""

        pending = self._pending_model_reload
        if pending is None or self._model_reload_phase != "verifying":
            return
        request_id, probe = pending
        self._run_javascript(
            "window.meapetLive2D && "
            "typeof window.meapetLive2D.finalizeModelReload === 'function' && "
            f"window.meapetLive2D.finalizeModelReload({request_id});"
        )
        self._pending_model_reload = None
        self._model_reload_phase = "idle"
        self._model_reload_deadline = 0.0
        self._probe = probe
        self._capabilities = probe.as_capabilities()
        self._model_ready = True
        self._failure_reported = False
        # 新模型从 neutral/idle 开始；方向、光标、气泡等宿主状态由页面保留。
        self._formal_motion_name = None
        self._pending_expression = ""
        self._pending_expression_request = None
        self._expression_timeline.cancel()
        self._pending_motion = ""
        self._pending_motion_request = None
        self._motion_request = None
        self._motion_request_elapsed = 0.0
        self._state.expression = "neutral"
        self._state.motion = "idle"
        self._state.elapsed = 0.0
        result: dict[str, object] = {
            "status": "available",
            "request_id": request_id,
            "model": probe.model_path,
        }
        self._model_reload_status = dict(result)
        callback = self._model_reload_callback
        if callback is None:
            return
        try:
            callback(dict(result))
        except Exception:
            logger.exception("Web Live2D model reload callback failed")

    def _fail_pending_model_reload(self, reason: str) -> None:
        """页面失效时收敛未完成切换，避免宿主永久停留在 pending。"""

        pending = self._pending_model_reload
        if pending is None:
            return
        request_id, probe = pending
        self._run_javascript(
            "window.meapetLive2D && "
            "typeof window.meapetLive2D.rollbackModelReload === 'function' && "
            f"window.meapetLive2D.rollbackModelReload({request_id});"
        )
        self._pending_model_reload = None
        self._model_reload_phase = "idle"
        self._model_reload_deadline = 0.0
        self._invalidate_ready_status("frame_pending")
        result: dict[str, object] = {
            "status": "failed",
            "request_id": request_id,
            "model": probe.model_path,
            "reason": str(reason or "Web Live2D page became unavailable"),
        }
        self._model_reload_status = dict(result)
        callback = self._model_reload_callback
        if callback is None:
            return
        try:
            callback(dict(result))
        except Exception:
            logger.exception("Web Live2D pending model reload failure callback failed")

    def set_cursor_target(self, x: int, y: int, width: int, height: int) -> bool:
        """把全局光标映射为模型注视目标；不依赖窗口是否接收输入。"""

        if self._view is None:
            return False
        try:
            target_width = max(1, int(width))
            target_height = max(1, int(height))
            target = (
                max(0, min(target_width - 1, int(x))),
                max(0, min(target_height - 1, int(y))),
                target_width,
                target_height,
            )
        except (OverflowError, TypeError, ValueError):
            return False
        if target == self._last_cursor_target:
            return True
        self._last_cursor_target = target
        self._run_javascript(
            "window.meapetLive2D && window.meapetLive2D.setCursorTarget("
            f"{target[0]},{target[1]},{target[2]},{target[3]});"
        )
        return True

    def supports_expression(self, name: str) -> bool:
        """判断表情是否已声明或属于安全的程序化回退。"""

        normalized = str(name or "").strip()
        if not normalized or self._view is None:
            return False
        has_formal_actions = bool(self._probe.expressions or self._probe.motions)
        if (
            self._probe.model_path is not None
            and self._model_ready
            and has_formal_actions
            and not _known_action_name(
                normalized,
                self._probe.expressions,
                _PROCEDURAL_EXPRESSIONS,
            )
        ):
            return False
        return True

    def supports_motion(self, name: str) -> bool:
        """判断动作是否已声明、具有显式 alias 或属于程序化回退。"""

        normalized = str(name or "").strip()
        if not normalized or self._view is None:
            return False
        has_formal_actions = bool(self._probe.expressions or self._probe.motions)
        if (
            self._probe.model_path is not None
            and self._model_ready
            and has_formal_actions
            and not _known_action_name(
                normalized,
                self._probe.motions,
                _PROCEDURAL_MOTIONS,
                self._probe.motion_aliases,
            )
        ):
            return False
        return True

    def notify_window_moved(self) -> bool:
        """通知页面顶层窗口刚发生位移，清除旧追踪速度。"""

        if self._view is None:
            return False
        self._run_javascript("window.meapetLive2D && window.meapetLive2D.notifyWindowMoved();")
        return True

    def _on_web_pointer_event(self, payload: str) -> None:
        """校验并转发页面通过 WebChannel 发送的指针事件。"""

        try:
            value = json.loads(str(payload or ""))
        except (TypeError, ValueError):
            logger.debug("invalid Web Live2D pointer payload")
            return
        if not isinstance(value, Mapping):
            return
        callback = self._interaction_callback
        if callback is not None:
            try:
                callback(dict(value))
            except Exception:
                # 页面事件不能让 Qt WebEngine 的回调线程退出；宿主会自行记录
                # 不可用状态，渲染循环继续工作。
                logger.exception("Web Live2D pointer callback failed")
        if str(value.get("type", "") or "").strip().lower() == "click":
            click_callback = self._click_callback
            if click_callback is not None:
                try:
                    click_callback(dict(value))
                except Exception:
                    logger.exception("Web Live2D click callback failed")

    def initialize(self, parent: Any | None = None) -> bool:
        """在 GUI 线程创建 WebEngine 视图和 QWebChannel 桥。"""

        if not self._capabilities.available or self._probe.model_path is None:
            return False
        if self._view is not None:
            return True
        self._page_discarded = False
        self._invalidate_ready_status("page_pending")

        core = _load_module("PySide6.QtCore", self._importer)
        gui = _load_module("PySide6.QtGui", self._importer)
        widgets = _load_module("PySide6.QtWebEngineWidgets", self._importer)
        web_channel = _load_module("PySide6.QtWebChannel", self._importer)
        if core is None or widgets is None or web_channel is None:
            self._set_unavailable("Qt WebEngine/WebChannel is unavailable")
            return False

        view_class = getattr(widgets, "QWebEngineView", None)
        channel_class = getattr(web_channel, "QWebChannel", None)
        object_class = getattr(core, "QObject", None)
        if view_class is None or channel_class is None or object_class is None:
            self._set_unavailable("Qt WebEngine bridge classes are unavailable")
            return False

        try:
            self._view = view_class(parent) if parent is not None else view_class()
            # 必须在 setHtml() 创建 Chromium 渲染面之前配置透明属性。若等到
            # 顶层宿主显示前才设置，部分 X11/Wayland 合成器会把首个不透明
            # WebEngine surface 固定为黑色，后续 CSS/页面 alpha 无法修正。
            self._configure_transparent_view(self._view, core, gui)
            self._channel = channel_class(self._view)
            bridge_class = self._make_bridge(core, object_class)
            self._bridge = bridge_class()
            self._channel.registerObject("meapetBridge", self._bridge)
            self._view.page().setWebChannel(self._channel)
            self._page_ready = False
            self._model_ready = False
            self._invalidate_ready_status("page_pending")
            load_finished = getattr(self._view, "loadFinished", None)
            connect = getattr(load_finished, "connect", None)
            if callable(connect):
                connect(self._on_load_finished)
            else:
                self._page_ready = True
            lifecycle_signal = getattr(self._view.page(), "lifecycleStateChanged", None)
            lifecycle_connect = getattr(lifecycle_signal, "connect", None)
            if callable(lifecycle_connect):
                lifecycle_connect(self._on_page_lifecycle_state_changed)
            self._view.setHtml(self._html_document(), self._base_url(core))
            # QWebEngineView 创建其内部 QQuickWidget/Chromium surface 的时机
            # 随 Qt 版本而变；在构造 view 时查找通常还没有子承载对象。setHtml
            # 后立刻再应用一次，避免内部场景沿用不透明黑色清屏，造成用户看到
            # 的黑色矩形或黑边。loadFinished 时还会再校正一次，覆盖异步创建。
            self._configure_transparent_view(self._view, core, gui)
            return True
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            view, self._view = self._view, None
            self._channel = None
            self._bridge = None
            self._page_ready = False
            self._model_ready = False
            self._page_discarded = True
            if view is not None:
                try:
                    view.deleteLater()
                except (AttributeError, RuntimeError):
                    pass
            self._set_unavailable("Qt WebEngine view initialization failed", notify=False)
            return False

    @staticmethod
    def _configure_transparent_view(view: Any, core: Any, gui: Any) -> None:
        """在 WebEngine 创建页面前申请带 alpha 的 QWidget/页面表面。"""

        qt = getattr(core, "Qt", None)
        color_class = getattr(gui, "QColor", None)
        palette_class = getattr(gui, "QPalette", None)
        if qt is None:
            return
        setter = getattr(view, "setAttribute", None)
        for name in (
            "WA_TranslucentBackground",
            "WA_NoSystemBackground",
            "WA_AlwaysStackOnTop",
        ):
            attribute = getattr(qt, name, None)
            if attribute is None or not callable(setter):
                continue
            try:
                setter(attribute, True)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        opaque_attribute = getattr(qt, "WA_OpaquePaintEvent", None)
        if opaque_attribute is not None and callable(setter):
            try:
                setter(opaque_attribute, False)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        auto_fill = getattr(view, "setAutoFillBackground", None)
        if callable(auto_fill):
            try:
                auto_fill(False)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        style = getattr(view, "setStyleSheet", None)
        if callable(style):
            try:
                style("background: transparent;")
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        # QWebEngineView 继承 QAbstractScrollArea，默认 frame 会在透明
        # surface 周围绘制一圈平台色边框；它不是模型像素，必须显式移除。
        try:
            frame_shape = getattr(view, "setFrameShape", None)
            widgets = _load_module("PySide6.QtWidgets")
            frame_class = getattr(widgets, "QFrame", None)
            shape_enum = getattr(frame_class, "Shape", frame_class)
            no_frame = getattr(shape_enum, "NoFrame", None)
            if callable(frame_shape) and no_frame is not None:
                frame_shape(no_frame)
            line_width = getattr(view, "setLineWidth", None)
            if callable(line_width):
                line_width(0)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            pass
        if not callable(color_class):
            return
        transparent = color_class(0, 0, 0, 0)
        # 展示窗口不应参与应用焦点竞争；QWebEngineView 的 focusProxy 和
        # 动态 Quick 子表面也要同步设为 NoFocus，否则 Chromium 重建时会
        # 触发 requestActivate 警告并把控制台输入焦点抢走。
        try:
            focus_policy = getattr(getattr(qt, "FocusPolicy", qt), "NoFocus", None)
            if focus_policy is None:
                focus_policy = getattr(qt, "NoFocus", None)
            focus_setter = getattr(view, "setFocusPolicy", None)
            if callable(focus_setter) and focus_policy is not None:
                focus_setter(focus_policy)
            proxy_getter = getattr(view, "focusProxy", None)
            proxy = proxy_getter() if callable(proxy_getter) else None
            proxy_setter = getattr(proxy, "setFocusPolicy", None)
            if proxy is not None and callable(proxy_setter) and focus_policy is not None:
                proxy_setter(focus_policy)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass
        if callable(palette_class):
            palette_getter = getattr(view, "palette", None)
            palette_setter = getattr(view, "setPalette", None)
            if callable(palette_getter) and callable(palette_setter):
                try:
                    palette = palette_getter()
                    role_class = getattr(palette_class, "ColorRole", palette_class)
                    set_color = getattr(palette, "setColor", None)
                    if callable(set_color):
                        for role_name in ("Window", "Base"):
                            role = getattr(role_class, role_name, None)
                            if role is not None:
                                set_color(role, transparent)
                        palette_setter(palette)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
        try:
            page = view.page()
            set_background = getattr(page, "setBackgroundColor", None)
            if callable(set_background):
                set_background(transparent)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass

        # Qt WebEngine Widgets 在 Qt 6 中通过内部 QQuickWidget 承载
        # Chromium 表面。只给外层 QWebEngineView 设置透明属性时，某些
        # 平台仍会让这个内部场景以默认黑色清屏；QQuickWidget 官方契约
        # 要求同时调用 setClearColor(Qt.transparent)。这里通过动态查找
        # 处理可选模块，不把 Qt QuickWidgets 变成核心导入依赖。
        try:
            quick_widgets = _load_module("PySide6.QtQuickWidgets")
            quick_class = getattr(quick_widgets, "QQuickWidget", None)
            finder = getattr(view, "findChildren", None)
            if callable(quick_class) and callable(finder):
                quick_children = tuple(finder(quick_class))
            else:
                quick_children = ()
            for quick in quick_children:
                clear_color = getattr(quick, "setClearColor", None)
                if callable(clear_color):
                    clear_color(transparent)
                quick_setter = getattr(quick, "setAttribute", None)
                if callable(quick_setter):
                    for name in (
                        "WA_TranslucentBackground",
                        "WA_NoSystemBackground",
                        "WA_AlwaysStackOnTop",
                    ):
                        attribute = getattr(qt, name, None)
                        if attribute is not None:
                            quick_setter(attribute, True)
                    if opaque_attribute is not None:
                        quick_setter(opaque_attribute, False)
                quick_auto_fill = getattr(quick, "setAutoFillBackground", None)
                if callable(quick_auto_fill):
                    quick_auto_fill(False)
                quick_style = getattr(quick, "setStyleSheet", None)
                if callable(quick_style):
                    quick_style("background: transparent;")
                quick_palette_getter = getattr(quick, "palette", None)
                quick_palette_setter = getattr(quick, "setPalette", None)
                if callable(quick_palette_getter) and callable(quick_palette_setter):
                    quick_palette = quick_palette_getter()
                    role_class = getattr(palette_class, "ColorRole", palette_class)
                    quick_set_color = getattr(quick_palette, "setColor", None)
                    if callable(quick_set_color):
                        for role_name in ("Window", "Base"):
                            role = getattr(role_class, role_name, None)
                            if role is not None:
                                quick_set_color(role, transparent)
                        quick_palette_setter(quick_palette)
            # 不同 Qt 版本可能使用其它 QWidget 作为 Chromium focusProxy；
            # 对当前树统一设 NoFocus，后续 ChildAdded 由宿主过滤器再次补齐。
            widgets = _load_module("PySide6.QtWidgets")
            widget_class = getattr(widgets, "QWidget", None)
            if callable(widget_class) and callable(finder):
                focus_policy = getattr(getattr(qt, "FocusPolicy", qt), "NoFocus", None)
                if focus_policy is None:
                    focus_policy = getattr(qt, "NoFocus", None)
                for child in tuple(finder(widget_class)):
                    setter = getattr(child, "setFocusPolicy", None)
                    if callable(setter) and focus_policy is not None:
                        setter(focus_policy)
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            # 内部承载实现随 Qt 版本变化；外层 QWidget/page 的透明配置
            # 已经完成时，找不到 QQuickWidget 不能阻止 Live2D 启动。
            pass

    def _motion_request_duration(self, request: MotionRequest) -> float | None:
        """解析页面动作请求的显式、motion3 或程序化自然时长。"""

        if request.duration_seconds is not None:
            return request.duration_seconds
        resolved = next(
            (
                target
                for alias, target in self._probe.motion_aliases
                if alias.casefold() == request.name.casefold()
            ),
            request.name,
        )
        declared = dict(self._probe.motion_durations).get(resolved)
        if declared is not None and declared > 0:
            return declared
        return _PROCEDURAL_MOTION_DURATIONS.get(request.name.casefold())

    def advance(self, elapsed_seconds: float) -> None:
        self._expire_pending_model_reload()
        if (
            self._pending_model_reload is not None
            and self._model_reload_phase == "verifying"
            and not self._dragging
        ):
            self.request_ready_status()
        if self._dragging:
            return
        try:
            seconds = float(elapsed_seconds)
        except (TypeError, ValueError, OverflowError):
            seconds = 0.0
        # 页面脚本和 Qt 定时器都可能在窗口切换/恢复时收到异常间隔；
        # 不能把 NaN/Inf 或一次性的大间隔拼进 JavaScript，否则会生成
        # 无效脚本或让动作跨越安全连续性护栏。
        if not math.isfinite(seconds):
            seconds = 0.0
        seconds = max(0.0, min(0.05, seconds))
        self._state.elapsed += seconds
        expression_frame = self._expression_timeline.advance(seconds)
        if expression_frame is not None and expression_frame.changed:
            self._state.expression = expression_frame.current
        motion_request = self._motion_request
        if motion_request is not None:
            self._motion_request_elapsed += seconds
            motion_duration = self._motion_request_duration(motion_request)
            if motion_duration is not None and self._motion_request_elapsed >= motion_duration:
                if motion_request.loop:
                    self._motion_request_elapsed %= motion_duration
                else:
                    self._motion_request = None
                    self._motion_request_elapsed = 0.0
                    self._state.motion = "idle"
        if not self._model_ready:
            return
        # 页面自身保留 requestAnimationFrame 作为无宿主时钟；外部 Qt
        # 驱动到达时标记来源，页面会在短窗口内跳过 rAF 更新，避免同一帧
        # 同时推进两次导致动作加速和物理穿模。
        self._run_javascript(
            f"window.meapetLive2D && window.meapetLive2D.advance({seconds!r}, 'external');"
        )

    def set_expression(self, name: str) -> bool | dict[str, object]:
        normalized = str(name or "").strip()
        if not normalized or self._view is None:
            return False
        # 至少有一个正式动作/表情声明时，模型已具备可验证的能力边界；
        # 完全没有声明的最小测试描述保留旧 pending 兼容行为。页面层仍
        # 会把非过程自定义名称交给 Cubism manager，以兼容运行时扩展。
        has_formal_actions = bool(self._probe.expressions or self._probe.motions)
        if (
            self._probe.model_path is not None
            and self._model_ready
            and has_formal_actions
            and not _known_action_name(normalized, self._probe.expressions, _PROCEDURAL_EXPRESSIONS)
        ):
            return False
        self._expression_timeline.cancel()
        self._pending_expression_request = None
        self._state.expression = normalized
        self._pending_expression = normalized
        self._replay_pending_model_state()
        return {
            "status": "available" if self._model_ready else "pending",
            "name": normalized,
        }

    def set_expression_request(self, request: ExpressionRequest) -> Mapping[str, object]:
        """启动由页面渲染时钟执行的表情序列或参数混合。"""

        if not isinstance(request, ExpressionRequest) or self._view is None:
            return {"status": "unavailable", "reason": "expression request is invalid"}
        if any(not self.supports_expression(layer.name) for layer in request.expressions):
            return {"status": "unavailable", "reason": "one or more expressions are unsupported"}
        if not self.supports_expression(request.restore):
            return {"status": "unavailable", "reason": "restore expression is unsupported"}
        frame = self._expression_timeline.start(request, current=self._state.expression)
        self._state.expression = frame.current
        self._pending_expression = ""
        self._pending_expression_request = _expression_request_payload(request)
        self._replay_pending_model_state()
        return {
            "status": "started" if self._model_ready else "pending",
            "mode": request.mode,
            "expression_count": len(request.expressions),
            "duration_seconds": _expression_request_duration(request),
            "parameters_applied": True,
        }

    def play_motion(self, name: str) -> bool | dict[str, object]:
        normalized = str(name or "").strip()
        if not normalized or self._view is None:
            return False
        has_formal_actions = bool(self._probe.expressions or self._probe.motions)
        if (
            self._probe.model_path is not None
            and self._model_ready
            and has_formal_actions
            and not _known_action_name(
                normalized,
                self._probe.motions,
                _PROCEDURAL_MOTIONS,
                self._probe.motion_aliases,
            )
        ):
            return False
        self._motion_request = None
        self._motion_request_elapsed = 0.0
        self._pending_motion_request = None
        resolved = next(
            (
                target
                for alias, target in self._probe.motion_aliases
                if alias.casefold() == normalized.casefold()
            ),
            normalized,
        )
        self._formal_motion_name = resolved if resolved in self._probe.motions else None
        self._state.motion = normalized
        self._pending_motion = normalized
        self._replay_pending_model_state()
        return {
            "status": "available" if self._model_ready else "pending",
            "name": normalized,
        }

    def play_motion_request(self, request: MotionRequest) -> Mapping[str, object]:
        """启动由页面渲染时钟执行的动作参数、时长和循环策略。"""

        if (
            not isinstance(request, MotionRequest)
            or self._view is None
            or not self.supports_motion(request.name)
        ):
            return {"status": "unavailable", "reason": "motion request is unsupported"}
        normalized = request.name
        resolved = next(
            (
                target
                for alias, target in self._probe.motion_aliases
                if alias.casefold() == normalized.casefold()
            ),
            normalized,
        )
        self._formal_motion_name = resolved if resolved in self._probe.motions else None
        self._state.motion = normalized
        self._motion_request = request
        self._motion_request_elapsed = 0.0
        self._pending_motion = ""
        self._pending_motion_request = _motion_request_payload(request)
        self._replay_pending_model_state()
        return {
            "status": "started" if self._model_ready else "pending",
            "name": request.name,
            "duration_seconds": request.duration_seconds,
            "transition_seconds": request.transition_seconds,
            "loop": request.loop,
            "parameters_applied": True,
        }

    def set_direction(self, direction: str) -> bool:
        """设置 A/B 朝向；Web 页面会以等比镜像实现无专属侧面模型的转身。"""

        normalized = str(direction or "").strip().upper()
        if normalized not in {"A", "B"} or self._view is None:
            return False
        self._pending_direction = normalized
        self._replay_pending_model_state()
        return True

    def set_speech(
        self,
        text: str,
        *,
        mood: str = "neutral",
        visible: bool = True,
        speaking: bool = True,
    ) -> bool:
        """更新 Web 页面中的动态气泡，并独立控制口型动画。"""

        previous = (
            self._speech_text,
            self._speech_mood,
            self._speech_visible,
            self._speech_speaking,
        )
        self._speech_text = str(text or "")
        self._speech_mood = str(mood or "neutral")
        self._speech_visible = bool(visible and self._speech_text)
        self._speech_speaking = bool(self._speech_visible and speaking)
        if self._view is None:
            return False
        self._pending_speech = (
            self._speech_text,
            self._speech_mood,
            bool(visible),
            self._speech_speaking,
        )
        if self._probe.motion_sync is not None and (
            not self._speech_speaking
            or previous[0] != self._speech_text
            or previous[1] != self._speech_mood
            or previous[2] != self._speech_visible
            or previous[3] != self._speech_speaking
        ):
            # 相同句子的重复 UI 快照不应覆盖正在播放的音频口型；
            # 新句、隐藏或 speaking 状态变化才回到静音基线。
            self._pending_mouth_viseme = ("Silence", 0.0)
        self._replay_pending_speech()
        self._replay_pending_mouth_viseme()
        return True

    def set_mouth_viseme(self, name: str, *, intensity: float = 1.0) -> bool:
        """提交 MotionSync 口型标签；没有安全 profile 时明确拒绝。"""

        profile = self._probe.motion_sync
        if profile is None or self._view is None:
            return False
        label_by_upper = {label.upper(): label for label in profile.visemes}
        requested = str(name or "").strip().upper()
        # PCM RMS 分析器只输出有限的通用标签；映射到当前模型已有的
        # MotionSync 形状，避免分析器事件被安全 profile 无声丢弃。
        requested = {"NEUTRAL": "I", "OPEN": "A"}.get(requested, requested)
        normalized = label_by_upper.get(requested)
        try:
            resolved_intensity = float(intensity)
        except (TypeError, ValueError):
            return False
        if normalized is None or not math.isfinite(resolved_intensity):
            return False
        resolved_intensity = max(0.0, min(1.0, resolved_intensity))
        self._pending_mouth_viseme = (normalized, resolved_intensity)
        self._replay_pending_mouth_viseme()
        return True

    def set_interaction_feedback(
        self,
        value: Mapping[str, object] | None,
        *,
        visible: bool = True,
    ) -> bool:
        """在模型窗口内显示不拦截鼠标的部位互动结果。"""

        payload: dict[str, str] = {}
        if isinstance(value, Mapping):
            for key in (
                "zone",
                "part",
                "phrase",
                "mood",
                "expression",
                "motion",
                "tier",
                "current",
                "applied",
            ):
                raw = value.get(key, "")
                if key in {"current", "applied"} and isinstance(value.get("affection"), Mapping):
                    raw = value["affection"].get(key, "")  # type: ignore[index]
                text = str(raw or "").strip().replace("\r", " ").replace("\n", " ")
                if text:
                    payload[key] = text[:160]
        if self._view is None:
            return False
        self._interaction_feedback_visible = bool(visible and payload)
        self._pending_interaction_feedback = (payload, self._interaction_feedback_visible)
        self._replay_pending_interaction_feedback()
        return True

    def move_to(self, x: float, y: float, *, duration_ms: int = 800) -> dict[str, object]:
        """请求 WebEngine 顶层窗口移动；平台层负责返回准确能力状态。"""

        del duration_ms
        view = self._view
        if view is None:
            return {"status": "unavailable", "reason": "web view is not initialized"}
        try:
            target_x = int(round(float(x)))
            target_y = int(round(float(y)))
            mover = getattr(view, "move", None)
            if callable(mover):
                mover(target_x, target_y)
            else:
                set_position = getattr(view, "setPosition", None)
                if not callable(set_position):
                    raise AttributeError("web view does not expose a position setter")
                set_position(target_x, target_y)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Web Live2D move failed: %s", type(exc).__name__)
            return {
                "status": "unavailable",
                "reason": "桌宠位置暂时无法调整，请稍后重试",
            }
        return {"status": "available", "x": target_x, "y": target_y}

    def set_click_through(self, enabled: bool) -> dict[str, object]:
        """在 Qt 支持时切换 WebEngine 顶层窗口的输入透明标志。"""

        view = self._view
        core = _load_module("PySide6.QtCore", self._importer)
        if view is None or core is None:
            return {"status": "unavailable", "reason": "web view or Qt Core is unavailable"}
        try:
            flag = core.Qt.WindowType.WindowTransparentForInput
            visibility_getter = getattr(view, "isVisible", None)
            was_visible = bool(visibility_getter()) if callable(visibility_getter) else None
            view.setWindowFlag(flag, bool(enabled))
            if was_visible and callable(visibility_getter) and not bool(visibility_getter()):
                shower = getattr(view, "show", None)
                if not callable(shower):
                    return {
                        "status": "unavailable",
                        "reason": "web view was hidden while applying the Qt flag",
                    }
                shower()
                if not bool(visibility_getter()):
                    return {
                        "status": "unavailable",
                        "reason": "web view remained hidden after applying the Qt flag",
                    }
            flags = getattr(view, "windowFlags", None)
            if not callable(flags):
                return {
                    "status": "unavailable",
                    "reason": "web view does not expose window flags for verification",
                }
            actual_flags = flags()
            active = bool(actual_flags & flag)
            if active != bool(enabled):
                return {
                    "status": "degraded",
                    "enabled": active,
                    "reason": "window system did not retain the requested Qt flag",
                }
            return {"status": "available", "enabled": bool(enabled)}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("Web Live2D click-through update failed: %s", type(exc).__name__)
            return {
                "status": "unavailable",
                "reason": "点击穿透暂时无法切换，请使用控制台恢复入口",
            }

    def shutdown(self) -> None:
        # 先让异步 Qt 定时器看到页面已失效，再关闭 QWebEngineView；否则
        # 退出窗口期间迟到的 setSpeech/runJavaScript 会在 Chromium discarded
        # 状态下打印警告并污染用户日志。
        self._page_ready = False
        self._model_ready = False
        self._page_discarded = True
        self._ready_probe_generation += 1
        self._ready_probe_pending = False
        self._set_ready_status(RendererReadyState.CLOSED, reason_code="page_closed")
        self._model_reload_generation += 1
        self._fail_pending_model_reload("Web Live2D renderer was closed during model reload")
        self._model_reload_status = {"status": "closed"}
        self._set_page_bridge_ready(False)
        view, self._view = self._view, None
        self._interaction_callback = None
        self._click_callback = None
        self._channel = None
        self._bridge = None
        self._page_bridge_ready_callback = None
        self._failure_reported = True
        self._pending_expression = ""
        self._pending_expression_request = None
        self._expression_timeline.cancel()
        self._pending_motion = ""
        self._pending_motion_request = None
        self._motion_request = None
        self._motion_request_elapsed = 0.0
        self._pending_direction = ""
        self._formal_motion_name = None
        self._pending_speech = None
        self._pending_mouth_viseme = None
        self._pending_interaction_feedback = None
        self._speech_visible = False
        self._interaction_feedback_visible = False
        self._interaction_locked = False
        self._dragging = False
        self._last_cursor_target = None
        if view is not None:
            try:
                # 先关闭顶层视图，让 Chromium 停止向 WebChannel 派发任务；
                # 页面对象过早 deleteLater 会在 discarded 状态下触发内部
                # JavaScript 清理，表现为退出日志污染。
                closer = getattr(view, "close", None)
                if callable(closer):
                    closer()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            try:
                page_getter = getattr(view, "page", None)
                page = page_getter() if callable(page_getter) else None
                delete_page = getattr(page, "deleteLater", None)
                if callable(delete_page):
                    delete_page()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            try:
                view.deleteLater()
            except (AttributeError, RuntimeError):
                pass

    def _set_unavailable(self, message: str, *, notify: bool = True) -> None:
        # 页面/模型报告错误后，原生 Qt 输入回退不能继续把已失效模型当作
        # 可命中对象；清掉就绪标志让宿主暂停部位反馈，等待精灵回退或重载。
        self._fail_pending_model_reload(message)
        self._model_ready = False
        self._set_page_bridge_ready(False)
        self._ready_probe_generation += 1
        self._ready_probe_pending = False
        self._set_ready_status(RendererReadyState.FAILED, reason_code="renderer_unavailable")
        self._capabilities = RendererCapabilities(
            backend="web_live2d",
            available=False,
            model_path=self._probe.model_path,
            expressions=_PROCEDURAL_EXPRESSIONS,
            motions=_PROCEDURAL_MOTIONS,
            message=message,
        )
        callback = self._failure_callback
        if (
            notify
            and self._view is not None
            and callback is not None
            and not self._failure_reported
        ):
            self._failure_reported = True
            try:
                callback(message)
            except Exception as exc:
                logger.warning("Web Live2D failure callback failed: %s", type(exc).__name__)

    def _run_javascript(self, script: str) -> bool:
        if self._view is None or not self._page_ready or self._page_discarded:
            return False
        try:
            page = self._view.page()
            lifecycle_getter = getattr(page, "lifecycleState", None)
            if callable(lifecycle_getter):
                state = lifecycle_getter()
                state_name = str(getattr(state, "name", state) or "").lower()
                if "discarded" in state_name:
                    self._on_page_lifecycle_state_changed(state)
                    return False
            page.runJavaScript(script)
            return True
        except (AttributeError, RuntimeError, TypeError, OSError, ValueError):
            return False

    def _on_page_lifecycle_state_changed(self, state: object) -> None:
        """页面进入 Discarded 后立即封锁所有迟到的渲染调用。"""

        state_name = str(getattr(state, "name", state) or "").lower()
        if "discarded" not in state_name:
            return
        # 先封锁页面，再结束模型切换事务。否则 _fail_pending_model_reload
        # 的回滚脚本会再次读取 discarded 生命周期并递归进入本方法，最终
        # 在 Qt 主线程中触发递归错误或把 READY 状态留在 pending。
        self._page_discarded = True
        self._page_ready = False
        self._model_ready = False
        self._set_page_bridge_ready(False)
        self._fail_pending_model_reload("Web Live2D page was discarded during model reload")
        # _fail_pending_model_reload 会使 READY 探针失效；终态必须重新明确
        # 固定为 CLOSED，避免页面已销毁却被报告成可重试的 pending。
        self._set_ready_status(RendererReadyState.CLOSED, reason_code="page_closed")

    def _on_load_finished(self, ok: bool = True) -> None:
        # 页面加载会触发/重建内部 QQuickWidget；每次页面完成后重新申请
        # alpha，不能只依赖 initialize() 构造阶段的属性设置。
        if self._view is not None:
            core = _load_module("PySide6.QtCore", self._importer)
            gui = _load_module("PySide6.QtGui", self._importer)
            if core is not None and gui is not None:
                self._configure_transparent_view(self._view, core, gui)
        self._page_ready = bool(ok)
        if self._page_ready:
            self._page_discarded = False
        if not self._page_ready:
            self._model_ready = False
            self._set_page_bridge_ready(False)
            self._set_unavailable("Web Live2D page failed to load")
            return
        # Chromium 的 loadFinished 与页面脚本的 modelReady 顺序并不固定；
        # 如果模型已经通过 QWebChannel 报告就绪，不能被晚到的 loadFinished
        # 信号重新降级为“未握手”。
        self._set_page_bridge_ready(self._model_ready)
        self._replay_pending_speech()
        self._replay_pending_interaction_feedback()
        self._replay_pending_model_state()
        self.request_ready_status()

    def _on_model_ready(self) -> None:
        """接收页面桥接的真实模型就绪信号后重放动作和表情。"""

        self._model_ready = True
        self._set_page_bridge_ready(True)
        self._replay_pending_interaction_feedback()
        self._replay_pending_model_state()
        self.request_ready_status()

    def _on_formal_motion_finished(self, name: str) -> None:
        """同步页面已完成的 motion3 或限时请求，收敛公开状态到 idle。"""

        normalized = str(name or "").strip()
        request = self._motion_request
        if (
            normalized
            and request is not None
            and not request.loop
            and normalized.casefold() == request.name.casefold()
        ):
            self._motion_request = None
            self._motion_request_elapsed = 0.0
            self._pending_motion_request = None
            self._formal_motion_name = None
            self._state.motion = "idle"
            self._state.elapsed = 0.0
            return
        current = self._formal_motion_name
        if not normalized or current is None:
            return
        if normalized.casefold() != current.casefold():
            return
        self._formal_motion_name = None
        self._state.motion = "idle"
        self._state.elapsed = 0.0

    def _replay_pending_model_state(self) -> None:
        if self._view is None or not self._page_ready or not self._model_ready:
            return
        expression, self._pending_expression = self._pending_expression, ""
        expression_request, self._pending_expression_request = (
            self._pending_expression_request,
            None,
        )
        motion, self._pending_motion = self._pending_motion, ""
        motion_request, self._pending_motion_request = self._pending_motion_request, None
        direction, self._pending_direction = self._pending_direction, ""
        if expression_request is not None:
            self._run_javascript(
                "window.meapetLive2D && window.meapetLive2D.setExpressionRequest("
                f"{json.dumps(expression_request, ensure_ascii=False)});"
            )
        elif expression:
            self._run_javascript(
                "window.meapetLive2D && window.meapetLive2D.setExpression("
                f"{json.dumps(expression, ensure_ascii=False)});"
            )
        if motion_request is not None:
            self._run_javascript(
                "window.meapetLive2D && window.meapetLive2D.playMotionRequest("
                f"{json.dumps(motion_request, ensure_ascii=False)});"
            )
        elif motion:
            self._run_javascript(
                "window.meapetLive2D && window.meapetLive2D.playMotion("
                f"{json.dumps(motion, ensure_ascii=False)});"
            )
        if direction:
            self._run_javascript(
                "window.meapetLive2D && window.meapetLive2D.setDirection("
                f"{json.dumps(direction, ensure_ascii=False)});"
            )
        if self._interaction_locked:
            self._run_javascript(
                "window.meapetLive2D && window.meapetLive2D.setInteractionLocked(true);"
            )
        if self._dragging:
            self._run_javascript("window.meapetLive2D && window.meapetLive2D.setDragging(true);")
        if self._last_cursor_target is not None:
            x, y, width, height = self._last_cursor_target
            self._run_javascript(
                "window.meapetLive2D && window.meapetLive2D.setCursorTarget("
                f"{x},{y},{width},{height});"
            )
        self._replay_pending_mouth_viseme()

    def _replay_pending_speech(self) -> None:
        if self._view is None or not self._page_ready:
            return
        speech, self._pending_speech = self._pending_speech, None
        if speech is not None:
            text, mood, visible, speaking = speech
            self._run_javascript(
                "window.meapetLive2D && window.meapetLive2D.setSpeech("
                f"{json.dumps(text, ensure_ascii=False)},"
                f"{json.dumps(mood, ensure_ascii=False)},"
                f"{str(visible).lower()},"
                f"{str(speaking).lower()});"
            )

    def _replay_pending_mouth_viseme(self) -> None:
        if self._view is None or not self._page_ready or self._pending_mouth_viseme is None:
            return
        name, intensity = self._pending_mouth_viseme
        self._run_javascript(
            "window.meapetLive2D && window.meapetLive2D.setMouthViseme("
            f"{json.dumps(name, ensure_ascii=False)},{intensity!r});"
        )

    def _replay_pending_interaction_feedback(self) -> None:
        if self._view is None or not self._page_ready:
            return
        value, self._pending_interaction_feedback = self._pending_interaction_feedback, None
        if value is None:
            return
        payload, visible = value
        self._run_javascript(
            "window.meapetLive2D && window.meapetLive2D.setInteractionFeedback("
            f"{json.dumps(payload, ensure_ascii=False)},{str(visible).lower()});"
        )

    def _base_url(self, core: Any) -> Any:
        qurl = getattr(core, "QUrl", None)
        if qurl is None:
            return None
        factory = getattr(qurl, "fromLocalFile", None)
        if not callable(factory):
            return None
        return factory(str(self.resource_root) + "/")

    def _model_part_names(self, model_path: Path | None = None) -> tuple[str, ...]:
        """读取当前 model3 DisplayInfo 的 Parts 顺序，供几何命中标注使用。"""

        descriptor = model_path if model_path is not None else self._probe.model_path
        if descriptor is None:
            return ()
        try:
            model = json.loads(descriptor.read_text(encoding="utf-8"))
            references = model.get("FileReferences")
            if not isinstance(references, dict):
                return ()
            display_reference = references.get("DisplayInfo")
            if not isinstance(display_reference, str) or not display_reference.strip():
                return ()
            display_path = (descriptor.parent / display_reference).resolve()
            display_path.relative_to(descriptor.parent.resolve())
            value = json.loads(display_path.read_text(encoding="utf-8"))
            parts = value.get("Parts")
            if not isinstance(parts, list):
                return ()
            names: list[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    names.append("")
                    continue
                identifier = part.get("Id")
                if isinstance(identifier, str) and identifier.strip():
                    names.append(identifier.strip())
                else:
                    names.append("")
            return tuple(names)
        except (
            OSError,
            RuntimeError,
            UnicodeError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return ()

    def _model_part_labels(self, model_path: Path | None = None) -> tuple[str, ...]:
        """读取 DisplayInfo 中与部件 ID 对齐的显示名称。"""

        descriptor = model_path if model_path is not None else self._probe.model_path
        if descriptor is None:
            return ()
        try:
            model = json.loads(descriptor.read_text(encoding="utf-8"))
            references = model.get("FileReferences")
            if not isinstance(references, dict):
                return ()
            display_reference = references.get("DisplayInfo")
            if not isinstance(display_reference, str) or not display_reference.strip():
                return ()
            display_path = (descriptor.parent / display_reference).resolve()
            display_path.relative_to(descriptor.parent.resolve())
            value = json.loads(display_path.read_text(encoding="utf-8"))
            parts = value.get("Parts")
            if not isinstance(parts, list):
                return ()
            labels: list[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    labels.append("")
                    continue
                label = part.get("Name")
                identifier = part.get("Id")
                if isinstance(label, str) and label.strip():
                    labels.append(label.strip())
                elif isinstance(identifier, str) and identifier.strip():
                    labels.append(identifier.strip())
                else:
                    labels.append("")
            return tuple(labels)
        except (
            OSError,
            RuntimeError,
            UnicodeError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return ()

    def _make_bridge(self, core: Any, object_class: Any) -> type:
        slot = getattr(core, "Slot", None)

        def decorate_string(function: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(slot):
                return function
            try:
                return slot(str)(function)
            except (TypeError, ValueError):
                return function

        def decorate_empty(function: Callable[..., Any]) -> Callable[..., Any]:
            if not callable(slot):
                return function
            try:
                return slot()(function)
            except (TypeError, ValueError):
                return function

        renderer = self

        class Bridge(object_class):
            @decorate_string
            def setExpression(self, name: str) -> bool:  # noqa: N802
                return renderer.set_expression(name)

            @decorate_string
            def playMotion(self, name: str) -> bool:  # noqa: N802
                return renderer.play_motion(name)

            @decorate_string
            def motionFinished(self, name: str) -> None:  # noqa: N802
                renderer._on_formal_motion_finished(name)

            @decorate_string
            def modelReloadFinished(self, payload: str) -> None:  # noqa: N802
                renderer._on_model_reload_finished(payload)

            @decorate_empty
            def modelReady(self) -> None:  # noqa: N802
                renderer._on_model_ready()

            @decorate_string
            def pointerEvent(self, payload: str) -> None:  # noqa: N802
                renderer._on_web_pointer_event(payload)

            @decorate_string
            def reportError(self, message: str) -> None:  # noqa: N802
                renderer._set_unavailable(str(message or "Web Live2D page reported an error"))

        return Bridge

    def _html_document(self) -> str:
        scripts = "\n".join(
            f'<script src="{path.relative_to(self.resource_root).as_posix()}"></script>'
            for path in self._probe.javascript_assets
        )
        model_url = ""
        if self._probe.model_path is not None:
            try:
                model_url = self._probe.model_path.relative_to(self.resource_root).as_posix()
            except ValueError:
                model_url = self._probe.model_path.name
        if model_url and self._probe.resource_version:
            model_url = f"{model_url}?meapet_resource_version={self._probe.resource_version}"
        model_json = json.dumps(model_url, ensure_ascii=False)
        part_names_json = json.dumps(self._model_part_names(), ensure_ascii=False)
        part_labels_json = json.dumps(self._model_part_labels(), ensure_ascii=False)
        # ``expressions``/``motions`` 只包含当前 model3 精确声明且文件真实存在的
        # 动作；语义别名单独注入，避免同名过程回退拦截正式动作。
        declared_expressions_json = json.dumps(self._probe.expressions, ensure_ascii=False)
        declared_motions_json = json.dumps(self._probe.motions, ensure_ascii=False)
        motion_aliases_json = json.dumps(dict(self._probe.motion_aliases), ensure_ascii=False)
        motion_durations_json = json.dumps(dict(self._probe.motion_durations), ensure_ascii=False)
        motion_sync_json = json.dumps(
            self._probe.motion_sync.as_mapping() if self._probe.motion_sync else None,
            ensure_ascii=False,
        )
        frame_rate_json = format(self._frame_rate, ".12g")
        geometry_audit_hz_json = format(self._geometry_audit_hz, ".12g")
        return (
            f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
html,body{{margin:0;width:100%;height:100%;background:transparent;overflow:hidden;line-height:0}}
canvas{{display:block;margin:0;width:100%;height:100%;background:transparent}}
#meapet-face-overlay{{position:fixed;inset:0;z-index:4;pointer-events:none}}
#meapet-bubble{{box-sizing:border-box;position:fixed;left:8px;right:8px;top:6px;z-index:10;
  max-height:min(48px, calc(100vh - 12px));overflow-y:hidden;overflow-x:hidden;
  overscroll-behavior:contain;
  padding:8px 12px;border:1px solid #ff9dbe;border-radius:18px;
  background:rgba(34,26,46,.94);color:#faf6fb;
  font:15px sans-serif;line-height:1.35;white-space:pre-wrap;word-break:break-word;
  display:none;pointer-events:none}}
#meapet-bubble.scrollable{{overflow-y:auto}}
#meapet-bubble::-webkit-scrollbar{{width:6px}}
#meapet-bubble:not(.scrollable)::-webkit-scrollbar{{width:0}}
#meapet-interaction-toast{{box-sizing:border-box;position:fixed;right:5px;top:50%;
  z-index:9;width:min(96px,calc(100% - 10px));max-height:112px;overflow:hidden;
  padding:7px 8px;border:1px solid #ffb6ce;border-radius:12px;
  background:rgba(34,26,46,.94);color:#fff5fa;font:11px sans-serif;line-height:1.3;
  white-space:pre-wrap;word-break:break-word;transform:translateY(-50%);
  display:none;pointer-events:none;text-align:left}}
#meapet-interaction-toast[data-zone="body"]{{border-color:#c9b5ff}}
#meapet-interaction-toast[data-zone="lower_left"]{{border-color:#9ddcff}}
#meapet-interaction-toast[data-zone="lower_right"]{{border-color:#ffd28f}}
#meapet-interaction-toast.pending{{opacity:.82;border-style:dashed}}
#meapet-hit-feedback{{position:fixed;inset:0;z-index:5;pointer-events:none;overflow:visible}}
.meapet-hit-marker{{position:absolute;width:30px;height:30px;transform:translate(-50%,-50%);
  pointer-events:none;background:transparent;border-radius:50%;
  animation:meapet-hit-pop 1.25s ease-out forwards}}
.meapet-hit-marker.pending{{opacity:.62;animation-duration:.52s}}
.meapet-hit-marker.pending::before{{border-style:dashed;animation:none}}
.meapet-hit-marker::before{{content:"";position:absolute;inset:1px;border:2px solid #ff9dbe;
  border-radius:50%;box-shadow:0 0 0 0 rgba(255,157,190,.52);
  animation:meapet-hit-ring 1.25s ease-out forwards}}
.meapet-hit-marker::after{{content:"";position:absolute;left:50%;top:calc(100% + 4px);
  transform:translateX(-50%);padding:2px 7px;border:1px solid rgba(255,157,190,.7);
  border-radius:9px;background:rgba(34,26,46,.84);color:#fff5fa;
  font:700 11px sans-serif;text-shadow:0 1px 3px #2b1737;white-space:nowrap}}
.meapet-hit-marker.head::after{{left:calc(100% + 7px);top:50%;transform:translateY(-50%)}}
.meapet-hit-marker.lower_left::after{{left:auto;right:calc(100% + 7px);top:50%;
  transform:translateY(-50%)}}
.meapet-hit-marker.lower_right::after{{left:calc(100% + 7px);top:50%;transform:translateY(-50%)}}
.meapet-hit-marker.head::after{{content:"摸摸头"}}
.meapet-hit-marker.body::after{{content:"拍拍身体"}}
.meapet-hit-marker.lower_left::after{{content:"左边"}}
.meapet-hit-marker.lower_right::after{{content:"右边"}}
.meapet-hit-marker.body::before{{border-color:#c9b5ff}}
.meapet-hit-marker.lower_left::before{{border-color:#9ddcff}}
.meapet-hit-marker.lower_right::before{{border-color:#ffd28f}}
@keyframes meapet-hit-pop{{0%{{opacity:.95;transform:translate(-50%,-50%) scale(.55)}}
  100%{{opacity:0;transform:translate(-50%,-50%) scale(1.2)}}}}
@keyframes meapet-hit-ring{{0%{{box-shadow:0 0 0 0 rgba(255,157,190,.58)}}
  100%{{box-shadow:0 0 0 9px rgba(255,157,190,0)}}}}
*{{user-select:none;-webkit-user-select:none}}
canvas{{touch-action:none}}
</style>
{scripts}
<script>
(function() {{
  const coreModel = window.Live2DCubismCore && window.Live2DCubismCore.Model;
  if (!coreModel || typeof coreModel.fromMoc !== 'function') return;
  const fromMoc = coreModel.fromMoc;
  function sequenceLike(source, values) {{
    const constructor = source && source.constructor;
    if (constructor && typeof constructor.from === 'function') {{
      try {{ return constructor.from(values); }} catch (error) {{}}
    }}
    return values;
  }}
  coreModel.fromMoc = function(moc) {{
    const model = fromMoc.call(this, moc);
    if (model && model.drawables) {{
      // 某些 Cubism Core/WebGL 组合会保留一个空或旧长度的 drawable
      // renderOrders。渲染器按这个数组排序；不校正时头发、手和身体会
      // 在动作/镜像切换时互相穿过。只在内容确实不同的情况下替换，
      // 不改变 Core 对数组类型和所有权的处理。
      // Cubism Core 6 的 MOC3 模型通常只在 getRenderOrders() 方法中
      // 暴露顺序；旧版桥接直接读取 model.renderOrders 会静默跳过修复，
      // 导致运行时 drawable 顺序被旧缓存覆盖时出现头发/身体穿模。
      const expected = typeof model.getRenderOrders === 'function'
        ? model.getRenderOrders() : model.renderOrders;
      if (!expected || typeof expected.length !== 'number') return model;
      const drawableCount = Number(model.drawables.count) || 0;
      const expectedValues = Array.from(expected, Number);
      const orderSet = new Set();
      let validOrder = Number.isInteger(drawableCount) && drawableCount > 0 &&
        expectedValues.length >= drawableCount;
      for (let index = 0; validOrder && index < drawableCount; index += 1) {{
        const order = expectedValues[index];
        if (!Number.isInteger(order) || order < 0 || order >= drawableCount ||
            orderSet.has(order)) {{
          validOrder = false;
          break;
        }}
        orderSet.add(order);
      }}
      for (let index = drawableCount; validOrder && index < expectedValues.length; index += 1) {{
        if (!Number.isFinite(expectedValues[index])) validOrder = false;
      }}
      if (!validOrder) return model;
      const actual = model.drawables.renderOrders;
      let needsRepair = !actual || actual.length !== expectedValues.length;
      if (!needsRepair) {{
        for (let index = 0; index < expectedValues.length; index += 1) {{
          if (Number(actual[index]) !== expectedValues[index]) {{
            needsRepair = true;
            break;
          }}
        }}
      }}
      if (needsRepair) {{
        try {{
          model.drawables.renderOrders = sequenceLike(actual || expected, expectedValues);
        }} catch (error) {{
          // 某些 Core 包装把 renderOrders 暴露为只读；让后续运行时
          // 校验决定是否 fail-closed，而不是在模型加载阶段抛出异常。
        }}
      }}
    }}
    return model;
  }};
}})();
</script>
<script src="qrc:///qtwebchannel/qwebchannel.js"></script>
</head><body><canvas id="meapet-face-overlay"></canvas><div id="meapet-bubble"></div>
<div id="meapet-interaction-toast" role="status" aria-live="polite" aria-atomic="true"></div>
<canvas id="meapet-live2d"></canvas><div id="meapet-hit-feedback"></div>
<script>
(function() {{
  let modelUrl = __MODEL_URL__;
  let modelPartNames = __PART_NAMES__;
  let modelPartLabels = __PART_LABELS__;
  let declaredExpressions = new Set(__DECLARED_EXPRESSIONS__);
  let declaredMotions = new Set(__DECLARED_MOTIONS__);
  let motionAliases = new Map(Object.entries(__MOTION_ALIASES__));
  let motionDurations = new Map(Object.entries(__MOTION_DURATIONS__));
  let motionSyncProfile = __MOTION_SYNC_PROFILE__;
  let liveModel = null;
  let pixiApp = null;
  let refreshHoverHook = null;
  let animationFrame = 0;
  let lastFrameTime = 0;
  // WebEngine 页面需要 rAF 作为独立运行时钟，但 Qt 宿主/测试也可能
  // 显式调用 renderer.advance。外部驱动活跃时短暂暂停 rAF，避免同一
  // 时间片重复推进动作、物理和几何拟合。
  const clockState = {{
    externalUntil: 0, skippedRafFrames: 0, dragging: false, dragReleaseUntil: 0,
    localPointerActive: false, hostDragging: false
  }};
  // 拖动期间 Qt/WebEngine 可能连续报告中间 viewport 尺寸。保留最新的
  // 有效尺寸但延迟真正的 Pixi resize，避免每次 ResizeObserver 回调都重置
  // Live2D 基准缩放、锚点和当前动作。释放后的恢复路径只消费一次 pending。
  const resizeState = {{
    pending: false, pendingWidth: 0, pendingHeight: 0,
    applied: false, appliedWidth: 0, appliedHeight: 0,
    appliedModel: null,
    deferredCount: 0, appliedCount: 0,
    flushTimer: null, flushRetries: 0
  }};
  const externalClockHoldMs = 110;
  // 绘制频率与几何审计频率独立：低帧率只减少 Core/绘制更新，
  // 几何审计仍按自己的周期运行，并复用最近一次合法结果。
  const renderPerformanceState = {{
    targetFrameRate: Math.max(15, Math.min(120, Number(__FRAME_RATE__) || 60)),
    geometryAuditHz: Math.max(1, Math.min(60, Number(__GEOMETRY_AUDIT_HZ__) || 30)),
    lastRenderAt: 0,
    pendingSeconds: 0,
    renderedFrames: 0,
    throttledFrames: 0,
    geometryAudits: 0,
    geometryAuditCacheHits: 0,
    geometryAuditCacheMisses: 0,
    geometryVertexDirtyFrames: 0,
    geometryStructuralDirtyFrames: 0
  }};
  renderPerformanceState.renderIntervalMs =
    1000 / renderPerformanceState.targetFrameRate;
  renderPerformanceState.geometryAuditIntervalMs =
    1000 / renderPerformanceState.geometryAuditHz;
  const geometryAuditState = {{
    cacheValid: false,
    force: true,
    lastAuditAt: 0,
    lastDynamicFlags: null
  }};
  let renderOrderErrorReported = false;
  const renderOrderAuditState = {{ valid: false, force: true, lastAuditAt: 0 }};
  const proceduralExpressions = new Set(["neutral", "happy", "sad", "curious", "surprised", "shy"]);
  const proceduralMotions = new Set(["idle", "blink", "wave", "walk"]);
  const expressionState = {{
    name: "neutral",
    procedural: true,
    targetAngle: {{ x: 0, y: 0 }},
    currentAngle: {{ x: 0, y: 0 }}
  }};
  const cursorState = {{
    targetX: 0, targetY: 0, currentX: 0, currentY: 0,
    velocityX: 0, velocityY: 0, lastInputX: 0, lastInputY: 0,
    targetGeneration: 0, locked: false
  }};
  // 兼容表情只写入当前 MOC3 顶点探测确认有效的标准参数；正式 exp3
  // 若存在则由模型动作管理器优先接管。
  const expressionProfiles = {{
    neutral: {{ ParamEyeLOpen: 1, ParamEyeROpen: 1, ParamEyeLSmile: 0,
      ParamEyeRSmile: 0, ParamMouthForm: 0, ParamMouthOpenY: 0,
      ParamCheek: 0, ParamBrowLY: 0, ParamBrowRY: 0,
      ParamEyeBallX: 0, ParamEyeBallY: 0 }},
    happy: {{ ParamEyeLOpen: 0.9, ParamEyeROpen: 0.9, ParamEyeLSmile: 1,
      ParamEyeRSmile: 1, ParamMouthForm: 0.7, ParamMouthOpenY: 0.2,
      ParamCheek: 0.45, ParamBrowLY: 0.15, ParamBrowRY: 0.15,
      ParamEyeBallX: 0, ParamEyeBallY: 0 }},
    sad: {{ ParamEyeLOpen: 0.75, ParamEyeROpen: 0.75, ParamEyeLSmile: 0,
      ParamEyeRSmile: 0, ParamMouthForm: -0.8, ParamMouthOpenY: 0.05,
      ParamCheek: 0, ParamBrowLY: -0.5, ParamBrowRY: -0.5,
      ParamEyeBallX: 0, ParamEyeBallY: 0 }},
    curious: {{ ParamEyeLOpen: 0.95, ParamEyeROpen: 0.95, ParamEyeLSmile: 0.3,
      ParamEyeRSmile: 0.3, ParamMouthForm: 0.2, ParamMouthOpenY: 0.1,
      ParamCheek: 0.1, ParamBrowLY: 0.5, ParamBrowRY: 0.2,
      ParamEyeBallX: 0.35, ParamEyeBallY: 0 }},
    surprised: {{ ParamEyeLOpen: 1, ParamEyeROpen: 1, ParamEyeLSmile: 0,
      ParamEyeRSmile: 0, ParamMouthForm: 0, ParamMouthOpenY: 0.9,
      ParamCheek: 0.2, ParamBrowLY: 0.7, ParamBrowRY: 0.7,
      ParamEyeBallX: 0, ParamEyeBallY: 0 }},
    shy: {{ ParamEyeLOpen: 0.8, ParamEyeROpen: 0.8, ParamEyeLSmile: 0.5,
      ParamEyeRSmile: 0.5, ParamMouthForm: 0.2, ParamMouthOpenY: 0.05,
      ParamCheek: 1, ParamBrowLY: -0.2, ParamBrowRY: -0.2,
      ParamEyeBallX: 0, ParamEyeBallY: 0 }}
  }};
  // 角度兼容值保持在小范围内，并继续受当前模型 min/max 与顶点绑定探测
  // 约束，避免把脸部提示放大成整个人物扭转。
  const expressionAngles = {{
    neutral: {{ x: 0, y: 0 }},
    // 保留可读的姿态词汇；真实脸部参数越完整，运行时缩放越小。
    happy: {{ x: -6, y: 2 }},
    sad: {{ x: 6, y: -7 }},
    curious: {{ x: 8, y: 5 }},
    surprised: {{ x: 0, y: 8 }},
    shy: {{ x: -7, y: -3 }}
  }};
  const motionState = {{ name: "idle", elapsed: 0, procedural: true }};
  const expressionTimelineState = {{
    active: false, phase: 'idle', request: null, index: 0, elapsed: 0,
    baseline: Object.create(null), current: Object.create(null),
    from: Object.create(null), target: Object.create(null)
  }};
  const motionRequestState = {{
    active: false, request: null, elapsed: 0,
    baseline: Object.create(null), current: Object.create(null)
  }};
  // formal exp3/motion3 加载可能异步完成；请求代次保证旧请求不能在
  // 新请求之后回写状态，避免实际 Cubism 动作与 debug/限幅状态分离。
  let expressionRequestGeneration = 0;
  let motionRequestGeneration = 0;
  // 注视角度和外层行走位移使用独立的安全状态。不能直接把光标瞬时值
  // 写入 Cubism 参数：窗口移动、光标跨屏或合成器丢帧时会产生大跳变，
  // 进而让发丝/手脚越过相邻 ArtMesh。这里用有速度上限的平滑和最终姿态
  // 限幅，保证“看向光标”不会牺牲模型拓扑稳定性。
  const poseState = {{
    currentX: 0, currentY: 0,
    desiredX: 0, desiredY: 0, appliedCursorX: 0, appliedCursorY: 0,
    facingCurrent: 1, mode: "procedural",
    formalAppliedX: 0, formalAppliedY: 0,
    formalWrittenX: null, formalWrittenY: null
  }};
  const trackingGuard = {{
    largeDeltaClamps: 0
  }};
  // 外部 motion3、窗口重排或 Pixi 的异步 transform 更新不能把模型写成
  // 非等比缩放。护栏只在检测到非有限/非等比值时收敛到较小的统一比例，
  // 以宁可短暂缩小也不把一帧拉伸矩阵提交给 WebGL。
  const transformGuard = {{ uniformScaleCorrections: 0 }};
  // 每帧都在 Core 更新后做一次轻量几何审计。静态 Pixi bounds 只反映
  // 模型原始矩形，动作/keyform 可能把发梢、裙摆推到矩形外；动态顶点
  // bounds 用于拟合和 fail-closed，避免“看起来等比但实际穿出窗口”。
  const geometryGuard = {{
    invalidFrames: 0,
    dynamicBoundsFallbacks: 0,
    geometryValid: true,
    continuityValid: true,
    continuityRejects: 0,
    continuityResets: 0,
    continuityResetFrames: 0,
    lastLocalBounds: null,
    dynamicBounds: null,
    worldBounds: null,
    faceBounds: null,
    faceFeatureBounds: {{ eyes: null, mouth: null, brows: null, face: null }},
    faceAnchorPoints: null
  }};
  // 动态 keyform 在正常表情/行走时只会产生小幅变化。若某一帧的局部
  // drawable 包围盒突然缩放或跳跃，通常意味着 Core/Pixi 网格上传不同步，
  // 继续绘制会表现为穿模或拉伸。该帧直接隔离，下一帧重新尝试；resize、
  // 转身和动作切换会显式放宽两帧，避免把合法的重排误判为坏帧。
  const GEOMETRY_MIN_RATIO = 0.55;
  const GEOMETRY_MAX_RATIO = 1.8;
  function resetGeometryContinuity(frames = 2) {{
    geometryAuditState.cacheValid = false;
    geometryAuditState.force = true;
    geometryGuard.lastLocalBounds = null;
    geometryGuard.continuityValid = true;
    geometryGuard.continuityResetFrames = Math.max(
      geometryGuard.continuityResetFrames,
      Math.max(1, Number(frames) || 1)
    );
    geometryGuard.continuityResets += 1;
  }}
  function checkGeometryContinuity(bounds) {{
    if (!bounds) {{
      geometryGuard.continuityValid = true;
      return true;
    }}
    const current = {{
      x: Number(bounds.x), y: Number(bounds.y),
      width: Number(bounds.width), height: Number(bounds.height)
    }};
    if (!finiteBounds(current)) {{
      geometryGuard.continuityValid = false;
      geometryGuard.continuityRejects += 1;
      return false;
    }}
    const previous = geometryGuard.lastLocalBounds;
    if (!previous || geometryGuard.continuityResetFrames > 0) {{
      geometryGuard.lastLocalBounds = current;
      geometryGuard.continuityResetFrames = Math.max(
        0, geometryGuard.continuityResetFrames - 1
      );
      geometryGuard.continuityValid = true;
      return true;
    }}
    const ratioX = current.width / Math.max(0.001, Number(previous.width));
    const ratioY = current.height / Math.max(0.001, Number(previous.height));
    const previousCenterX = Number(previous.x) + Number(previous.width) / 2;
    const previousCenterY = Number(previous.y) + Number(previous.height) / 2;
    const currentCenterX = current.x + current.width / 2;
    const currentCenterY = current.y + current.height / 2;
    const centerDistance = Math.hypot(
      currentCenterX - previousCenterX, currentCenterY - previousCenterY
    );
    const extent = Math.max(
      Number(previous.width), Number(previous.height), current.width, current.height
    );
    const centerLimit = Math.max(96, extent * 0.45);
    const valid = [ratioX, ratioY, centerDistance].every(Number.isFinite) &&
      ratioX >= GEOMETRY_MIN_RATIO && ratioX <= GEOMETRY_MAX_RATIO &&
      ratioY >= GEOMETRY_MIN_RATIO && ratioY <= GEOMETRY_MAX_RATIO &&
      centerDistance <= centerLimit;
    if (!valid) {{
      geometryGuard.continuityValid = false;
      geometryGuard.continuityRejects += 1;
      return false;
    }}
    geometryGuard.lastLocalBounds = current;
    geometryGuard.continuityValid = true;
    return true;
  }}
  function enforceUniformTransform() {{
    if (!liveModel || !liveModel.scale || typeof liveModel.scale.set !== 'function') return false;
    const scaleX = Number(liveModel.scale.x);
    const scaleY = Number(liveModel.scale.y);
    const rotation = Number(liveModel.rotation);
    if (![scaleX, scaleY, rotation].every(Number.isFinite)) return false;
    const magnitudeX = Math.abs(scaleX);
    const magnitudeY = Math.abs(scaleY);
    if (!(magnitudeX > 0) || !(magnitudeY > 0)) return false;
    const uniform = Math.min(magnitudeX, magnitudeY);
    const sign = scaleX < 0 ? -1 : 1;
    if (Math.abs(magnitudeX - magnitudeY) > 0.000001) {{
      liveModel.scale.set(sign * uniform, uniform);
      const baseTransform = liveModel.__meapetBaseTransform;
      if (baseTransform && Number.isFinite(Number(baseTransform.scale))) {{
        baseTransform.scale = uniform;
      }}
      transformGuard.uniformScaleCorrections += 1;
    }}
    return true;
  }}
  let speechActive = false;
  let speechElapsed = 0;
  const motionSyncState = {{
    viseme: "Silence", intensity: 0, mouthForm: 0, mouthOpenY: 0,
    appliedParameters: 0
  }};
  const modelLoadTimeoutMs = 10000;
  // 外层镜像可以立即切换，但横向姿态不能同步瞬时变号。否则自主移动
  // 在 A/B 方向之间切换时，发丝/手臂会在单帧内跨过另一层 ArtMesh。
  let facing = 1;
  let facingTarget = 1;
  // 几何镜像不能在目标方向写入的同一帧翻转。先让姿态经过中性区，
  // 再切换 scale.x 的符号；这样转身时不会把发丝、手臂和身体以相反
  // 的 keyform 同时提交给 WebGL。
  const facingState = {{ target: 1, current: 1, sign: 1 }};
  const FACING_SWITCH_THRESHOLD = 0.12;
  let pendingModelReady = false;
  let pendingError = '';
  // 模型切换回执带有严格代次；页面只保留一笔候选加载，旧请求完成
  // 后也不能覆盖新模型。回执在 WebChannel 尚未握手时进入有界队列。
  let modelReloadGeneration = 0;
  let pendingModelReload = null;
  const pendingModelReloadEvents = [];
  const requestedViewport = {{ width: 0, height: 0 }};
  const feedbackState = {{
    part: null, count: 0, expiresAt: 0, anchorX: 0, anchorY: 0,
    provisionalMarker: null, pendingToastGeneration: 0
  }};
  const faceOverlay = document.getElementById('meapet-face-overlay');
  const faceOverlayContext = faceOverlay && faceOverlay.getContext
    ? faceOverlay.getContext('2d', {{ willReadFrequently: true }}) : null;
  let faceOverlayEnabled = true;
  let eyeOverlayEnabled = true;
  let mouthOverlayEnabled = true;
  let expressionOverlayEnabled = true;
  function flushBridgeEvents() {{
    const bridge = window.meapetBridge;
    if (!bridge) return;
    if (pendingError) {{
      const message = pendingError;
      pendingError = '';
      if (bridge.reportError) bridge.reportError(message);
    }}
    if (pendingModelReady) {{
      pendingModelReady = false;
      if (bridge.modelReady) bridge.modelReady();
    }}
    while (pendingModelReloadEvents.length && bridge.modelReloadFinished) {{
      const event = pendingModelReloadEvents.shift();
      bridge.modelReloadFinished(JSON.stringify(event));
    }}
  }}
  function reportError(error) {{
    const message = String(error && error.message ? error.message : error);
    if (window.meapetBridge && window.meapetBridge.reportError) {{
      window.meapetBridge.reportError(message);
    }} else {{
      pendingError = message;
    }}
  }}
  function reportModelReady() {{
    if (window.meapetBridge && window.meapetBridge.modelReady) {{
      window.meapetBridge.modelReady();
    }} else {{
      pendingModelReady = true;
    }}
  }}
  function reportModelReload(requestId, status, model, reason) {{
    const event = {{
      request_id: Number(requestId) || 0,
      status: String(status || 'failed'),
      model: String(model || ''),
      reason: String(reason || '')
    }};
    const bridge = window.meapetBridge;
    if (bridge && typeof bridge.modelReloadFinished === 'function') {{
      try {{ bridge.modelReloadFinished(JSON.stringify(event)); }} catch (error) {{}}
    }} else if (pendingModelReloadEvents.length < 4) {{
      pendingModelReloadEvents.push(event);
    }}
  }}
  function clearPendingHitFeedback() {{
    feedbackState.pendingToastGeneration += 1;
    const toast = document.getElementById('meapet-interaction-toast');
    if (!toast || toast.dataset.pending !== 'true') return;
    toast.dataset.pending = 'false';
    toast.classList.remove('pending');
    toast.textContent = '';
    toast.style.display = 'none';
  }}
  function showPendingHitFeedback(part) {{
    const normalized = String(part || 'body').trim().toLowerCase();
    const toast = document.getElementById('meapet-interaction-toast');
    if (!toast) return;
    const labels = {{
      head: '猫猫头 · 正在回应…', body: '身体 · 正在回应…',
      lower_left: '左边 · 正在回应…', lower_right: '右边 · 正在回应…'
    }};
    const generation = ++feedbackState.pendingToastGeneration;
    toast.dataset.zone = normalized;
    toast.dataset.pending = 'true';
    toast.classList.add('pending');
    toast.textContent = labels[normalized] || labels.body;
    toast.style.display = 'block';
    setTimeout(function() {{
      if (feedbackState.pendingToastGeneration !== generation) return;
      clearPendingHitFeedback();
    }}, 900);
  }}
  function showHitFeedback(part, canvasX, canvasY, provisional = false) {{
    const normalized = String(part || 'body').trim().toLowerCase();
    const layer = document.getElementById('meapet-hit-feedback');
    if (!layer) return;
    if (!provisional) clearPendingHitFeedback();
    if (feedbackState.provisionalMarker && feedbackState.provisionalMarker.parentNode === layer) {{
      layer.removeChild(feedbackState.provisionalMarker);
    }}
    feedbackState.provisionalMarker = null;
    const offsets = {{
      head: {{ x: 34, y: -18 }},
      body: {{ x: 24, y: -14 }},
      lower_left: {{ x: -24, y: -12 }},
      lower_right: {{ x: 24, y: -12 }}
    }};
    const offset = offsets[normalized] || offsets.body;
    const maxX = Math.max(16, Number(layer.clientWidth || window.innerWidth) - 16);
    const maxY = Math.max(16, Number(layer.clientHeight || window.innerHeight) - 16);
    const anchorX = Math.max(16, Math.min(maxX, (Number(canvasX) || 0) + offset.x));
    const anchorY = Math.max(16, Math.min(maxY, (Number(canvasY) || 0) + offset.y));
    const marker = document.createElement('span');
    marker.className = 'meapet-hit-marker ' + (provisional ? 'pending ' : '') +
      (['head', 'body', 'lower_left', 'lower_right'].includes(normalized)
        ? normalized : 'body');
    marker.dataset.part = normalized;
    // 标记略微移到点击点外侧，环心不压住眼睛、嘴或身体主体。
    marker.style.left = `${{anchorX}}px`;
    marker.style.top = `${{anchorY}}px`;
    marker.setAttribute('aria-hidden', 'true');
    layer.appendChild(marker);
    if (provisional) {{
      feedbackState.provisionalMarker = marker;
      setTimeout(function() {{
        if (feedbackState.provisionalMarker === marker) feedbackState.provisionalMarker = null;
        if (marker.parentNode === layer) layer.removeChild(marker);
      }}, 520);
      return;
    }}
    feedbackState.part = normalized;
    feedbackState.count += 1;
    feedbackState.anchorX = anchorX;
    feedbackState.anchorY = anchorY;
    feedbackState.expiresAt = performance.now() + 1600;
    setTimeout(function() {{
      if (marker.parentNode === layer) layer.removeChild(marker);
    }}, 1640);
  }}
  // 只把经过真实顶点探测的 CDI 参数当作 Cubism 变形入口。不同模型的
  // 参数绑定差异很大；运行时按当前 MOC3 实测开放，未探测到的 ID 永远
  // 不会被写入并伪装成动作证据。
  const parameterBindingState = Object.create(null);
  // 记录真实模型参数的保守安全范围。只对会改变姿态/骨骼的参数启用，
  // 眼睛透明度等离散反馈仍允许完整取值，避免把眨眼夹成半睁。
  const parameterSafetyState = Object.create(null);
  // Core 参数写入发生在 internalModel.update() 之后时，Cubism 不会自动
  // 重算 drawable 顶点；记录一次脏标记，由 advance() 在本帧所有姿态写入
  // 完成后合并执行一次 Core update，避免使用上一帧网格布局或命中区域。
  let coreParameterDirty = false;
  const poseParameterIds = new Set([
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
    "ParamHairFront", "ParamHairSide", "ParamHairBack"
  ]);
  const probeParameterIds = [
    "ParamAngleX", "ParamAngleY", "ParamAngleZ",
    "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeLSmile", "ParamEyeRSmile",
    "ParamEyeBallX", "ParamEyeBallY", "ParamBrowLY", "ParamBrowRY",
    "ParamMouthForm", "ParamMouthOpenY", "ParamCheek",
    "ParamBodyAngleX", "ParamBodyAngleY", "ParamBodyAngleZ",
    "ParamBreath", "ParamHairFront", "ParamHairSide", "ParamHairBack"
  ];
  const facialParameterIds = [
    "ParamEyeLOpen", "ParamEyeROpen", "ParamEyeLSmile", "ParamEyeRSmile",
    "ParamEyeBallX", "ParamEyeBallY", "ParamBrowLY", "ParamBrowRY",
    "ParamMouthForm", "ParamMouthOpenY", "ParamCheek"
  ];
  const expressionParameterIds = [
    "ParamEyeLSmile", "ParamEyeRSmile", "ParamBrowLY", "ParamBrowRY",
    "ParamMouthForm", "ParamMouthOpenY", "ParamCheek"
  ];
  let facialBindingCount = 0;
  let expressionBindingCount = 0;
  let expressionPoseScale = 1;
  function drawableVertexSnapshot(core) {{
    if (!core || typeof core.getDrawableCount !== 'function' ||
        typeof core.getDrawableVertexPositions !== 'function') return null;
    const vertices = [];
    for (let index = 0; index < Number(core.getDrawableCount()) || 0; index += 1) {{
      const positions = core.getDrawableVertexPositions(index);
      vertices.push(positions ? Array.from(positions) : []);
    }}
    const raw = typeof core.getModel === 'function' ? core.getModel() : null;
    const opacities = raw && raw.drawables && raw.drawables.opacities
      ? Array.from(raw.drawables.opacities) : [];
    return {{ vertices, opacities }};
  }}
  function drawableVertexChanged(before, after) {{
    if (!before || !after || before.vertices.length !== after.vertices.length) return false;
    for (let drawable = 0; drawable < before.vertices.length; drawable += 1) {{
      const previous = before.vertices[drawable];
      const current = after.vertices[drawable];
      if (previous.length !== current.length) return true;
      for (let item = 0; item < previous.length; item += 1) {{
        if (Math.abs(Number(previous[item]) - Number(current[item])) > 1e-7) return true;
      }}
    }}
    if (before.opacities.length !== after.opacities.length) return true;
    for (let drawable = 0; drawable < before.opacities.length; drawable += 1) {{
      if (
        Math.abs(Number(before.opacities[drawable]) - Number(after.opacities[drawable])) > 1e-7
      ) {{
        return true;
      }}
    }}
    return false;
  }}
  function detectParameterBindings() {{
    const internal = liveModel && liveModel.internalModel;
    const core = internal && internal.coreModel;
    probeParameterIds.forEach(function(id) {{
      parameterBindingState[id] = false;
      delete parameterSafetyState[id];
    }});
    if (!core || typeof core.getParameterIndex !== 'function' ||
        typeof core.getParameterValueByIndex !== 'function' ||
        typeof core.setParameterValueByIndex !== 'function' ||
        typeof core.getParameterMinimumValue !== 'function' ||
        typeof core.getParameterMaximumValue !== 'function') return;
    for (const id of probeParameterIds) {{
      try {{
        const index = Number(core.getParameterIndex(id));
        if (index < 0) continue;
        const current = Number(core.getParameterValueByIndex(index));
        const minimum = Number(core.getParameterMinimumValue(index));
        const maximum = Number(core.getParameterMaximumValue(index));
        if (!Number.isFinite(current) || !Number.isFinite(minimum) ||
            !Number.isFinite(maximum) || minimum === maximum) continue;
        const probe = Math.abs(current - minimum) >= Math.abs(maximum - current)
          ? maximum : minimum;
        if (poseParameterIds.has(id)) {{
          // 以当前值为中心只开放约三分之一的原始范围，并保留模型自身
          // 的 min/max；范围来自真实 core 参数，不依赖某个导出器的固定数值。
          const span = Math.max(Math.abs(maximum - current), Math.abs(current - minimum));
          const safeSpan = Math.max(0.5, span * 0.34);
          parameterSafetyState[id] = {{
            min: Math.max(minimum, current - safeSpan),
            max: Math.min(maximum, current + safeSpan),
            sourceMin: minimum,
            sourceMax: maximum
          }};
        }}
        const before = drawableVertexSnapshot(core);
        core.setParameterValueByIndex(index, probe);
        const updater = internal && typeof internal.update === 'function'
          ? function() {{ internal.update(0, performance.now()); }}
          : typeof core.update === 'function' ? function() {{ core.update(); }} : null;
        if (updater) updater();
        const after = drawableVertexSnapshot(core);
        parameterBindingState[id] = drawableVertexChanged(before, after);
        core.setParameterValueByIndex(index, current);
        if (updater) updater();
      }} catch (error) {{
        parameterBindingState[id] = false;
      }}
    }}
    facialBindingCount = facialParameterIds.filter(
      id => parameterBindingState[id] === true
    ).length;
    expressionBindingCount = expressionParameterIds.filter(
      id => parameterBindingState[id] === true
    ).length;
    // 眼睛、嘴型和装饰表情分别决定回退层，不能用任意几个面部绑定
    // 一次性关闭整张 overlay。完整模型可直接使用自己的 keyform；缺少
    // 眼开闭或口型绑定时仍保留相应的局部视觉反馈。
    eyeOverlayEnabled = !(
      parameterBindingState.ParamEyeLOpen && parameterBindingState.ParamEyeROpen
    );
    mouthOverlayEnabled = !(
      parameterBindingState.ParamMouthForm || parameterBindingState.ParamMouthOpenY
    );
    expressionOverlayEnabled = expressionBindingCount < 2;
    faceOverlayEnabled = eyeOverlayEnabled || mouthOverlayEnabled || expressionOverlayEnabled;
    expressionPoseScale = facialBindingCount >= 3 ? 0.25 : 1;
  }}
  function setParameterValue(id, value, respectSafety = true) {{
    if (Object.prototype.hasOwnProperty.call(parameterBindingState, id) &&
        parameterBindingState[id] !== true) return false;
    const internal = liveModel && liveModel.internalModel;
    const model = internal && internal.coreModel;
    if (!model || typeof model.setParameterValueById !== 'function') return false;
    let safeValue = Number(value);
    if (!Number.isFinite(safeValue)) safeValue = 0;
    const safety = parameterSafetyState[id];
    if (respectSafety && safety && poseParameterIds.has(id)) {{
      safeValue = clampValue(safeValue, Number(safety.min), Number(safety.max));
    }}
    try {{
      model.setParameterValueById(id, safeValue);
      coreParameterDirty = true;
      return true;
    }} catch (error) {{
      return false;
    }}
  }}
  function readParameterValue(id, fallback = 0) {{
    const internal = liveModel && liveModel.internalModel;
    const model = internal && internal.coreModel;
    if (!model || typeof model.getParameterValueById !== 'function') return fallback;
    try {{
      const value = Number(model.getParameterValueById(id));
      return Number.isFinite(value) ? value : fallback;
    }} catch (error) {{
      return fallback;
    }}
  }}
  function setPartOpacity(id, value) {{
    const internal = liveModel && liveModel.internalModel;
    const model = internal && internal.coreModel;
    if (!model) return false;
    const normalized = Math.max(0, Math.min(1, Number(value) || 0));
    let changed = false;
    try {{
      if (typeof model.setPartOpacityById === 'function') {{
        model.setPartOpacityById(id, normalized);
        changed = true;
      }}
      if (typeof model.getPartIndex === 'function' &&
          typeof model.setPartOpacityByIndex === 'function') {{
        const index = model.getPartIndex(id);
        if (Number(index) >= 0) {{
          model.setPartOpacityByIndex(index, normalized);
          changed = true;
        }}
      }}
    }} catch (error) {{
      // 某些 Core 版本只暴露 drawable 数组；继续尝试直接更新子 ArtMesh。
    }}
    try {{
      const raw = typeof model.getModel === 'function' ? model.getModel() : null;
      const parts = raw && raw.parts;
      const drawables = raw && raw.drawables;
      const partIds = parts && parts.ids;
      const parentIndices = drawables && drawables.parentPartIndices;
      const opacities = drawables && drawables.opacities;
      const partIndex = partIds && typeof partIds.indexOf === 'function'
        ? partIds.indexOf(id) : -1;
      if (partIndex >= 0 && parentIndices && opacities) {{
        for (let index = 0; index < parentIndices.length; index += 1) {{
          if (Number(parentIndices[index]) === partIndex) opacities[index] = normalized;
        }}
        changed = true;
      }}
    }} catch (error) {{
      if (changed) coreParameterDirty = true;
      return changed;
    }}
    if (changed) coreParameterDirty = true;
    return changed;
  }}
  function motionSyncMapping(name) {{
    if (!motionSyncProfile || !motionSyncProfile.mappings) return null;
    const mapping = motionSyncProfile.mappings[String(name || '').trim()];
    return mapping && Array.isArray(mapping.targets) ? mapping : null;
  }}
  function motionSyncTargetValue(mapping, id, fallback) {{
    if (!mapping || !Array.isArray(mapping.targets)) return fallback;
    for (const target of mapping.targets) {{
      if (target && target.id === id && Number.isFinite(Number(target.value))) {{
        return Number(target.value);
      }}
    }}
    return fallback;
  }}
  function applyMotionSyncViseme(name, intensity) {{
    const requested = String(name || '').trim();
    const mapping = motionSyncMapping(requested);
    const silence = motionSyncMapping('Silence');
    if (!mapping || !silence) return false;
    const audio = Array.isArray(motionSyncProfile.audio_parameters)
      ? motionSyncProfile.audio_parameters.find(item => item && item.id === requested)
      : null;
    if (!audio || audio.enabled !== true) return false;
    // intensity 已由上游音素输入归一化；资源中的 Scale、BlendRatio、Smoothing
    // 和 SampleRate 属于音频分析阶段，不能在这里重复放大或平滑，否则会把
    // A/O 等标签的显式强度改变成不可预测的值。
    const amount = Math.max(0, Math.min(1, Number(intensity) || 0));
    const formTarget = motionSyncTargetValue(mapping, 'ParamMouthForm', 0);
    const openTarget = motionSyncTargetValue(mapping, 'ParamMouthOpenY', 0);
    const formSilence = motionSyncTargetValue(silence, 'ParamMouthForm', 0);
    const openSilence = motionSyncTargetValue(silence, 'ParamMouthOpenY', 0);
    const form = formSilence + (formTarget - formSilence) * amount;
    const open = openSilence + (openTarget - openSilence) * amount;
    let applied = 0;
    const targetIds = new Set();
    for (const target of mapping.targets) {{
      if (!target || typeof target.id !== 'string' || targetIds.has(target.id)) continue;
      targetIds.add(target.id);
      const base = motionSyncTargetValue(silence, target.id, 0);
      const targetValue = Number(target.value);
      if (!Number.isFinite(targetValue)) continue;
      if (parameterBindingState[target.id] === true &&
          setParameterValue(target.id, base + (targetValue - base) * amount, false)) {{
        applied += 1;
      }}
    }}
    motionSyncState.viseme = requested;
    motionSyncState.intensity = amount;
    motionSyncState.mouthForm = Number.isFinite(form) ? form : 0;
    motionSyncState.mouthOpenY = Number.isFinite(open) ? open : 0;
    motionSyncState.appliedParameters = applied;
    return true;
  }}
  function facePoint(normalizedX, normalizedY) {{
    if (!liveModel || !window.PIXI || !PIXI.Point || typeof liveModel.toGlobal !== 'function') {{
      return null;
    }}
    const bounds = finiteBounds(geometryGuard.dynamicBounds)
      ? geometryGuard.dynamicBounds
      : (typeof liveModel.getLocalBounds === 'function' ? liveModel.getLocalBounds() : null);
    if (!bounds) return null;
    return liveModel.toGlobal(new PIXI.Point(
      bounds.x + bounds.width * normalizedX,
      bounds.y + bounds.height * normalizedY
    ));
  }}
  function featureBoundsFor(name) {{
    const features = geometryGuard.faceFeatureBounds;
    const value = features && features[name];
    return finiteBounds(value) ? value : null;
  }}
  function featurePoint(name, normalizedX, normalizedY, fallbackX, fallbackY) {{
    const bounds = featureBoundsFor(name);
    if (!bounds) return facePoint(fallbackX, fallbackY);
    if (!liveModel || !window.PIXI || !PIXI.Point || typeof liveModel.toGlobal !== 'function') {{
      return null;
    }}
    return liveModel.toGlobal(new PIXI.Point(
      bounds.x + bounds.width * normalizedX,
      bounds.y + bounds.height * normalizedY
    ));
  }}
  function pointRecord(point) {{
    if (!point) return null;
    const x = Number(point.x);
    const y = Number(point.y);
    return Number.isFinite(x) && Number.isFinite(y) ? {{ x, y }} : null;
  }}
  function drawFaceCurve(context, point, width, height, start, end) {{
    if (!point) return;
    context.beginPath();
    context.moveTo(point.x - width / 2, point.y);
    context.quadraticCurveTo(point.x, point.y + height, point.x + width / 2, point.y);
    context.stroke();
  }}
  function drawFaceOverlay() {{
    if (!faceOverlayContext || !faceOverlay) return;
    const context = faceOverlayContext;
    context.clearRect(0, 0, faceOverlay.width, faceOverlay.height);
    if (!faceOverlayEnabled || !liveModel) return;
    const bounds = finiteBounds(geometryGuard.dynamicBounds)
      ? geometryGuard.dynamicBounds
      : (typeof liveModel.getLocalBounds === 'function' ? liveModel.getLocalBounds() : null);
    if (!bounds) return;
    // 优先使用 DisplayInfo 中脸/眼/嘴/眉部件的当前顶点 bounds，而不是
    // 把整个人物的归一化比例当成脸部坐标。不同立绘的头身比差异很大，
    // 固定比例会把嘴画到额头或围巾上；没有语义部件时才回退到可见全身。
    const eyeWidth = Math.max(8, bounds.width * 0.07 * Math.abs(Number(liveModel.scale.x) || 1));
    const eyeHeight = Math.max(3, bounds.height * 0.018 * Math.abs(Number(liveModel.scale.y) || 1));
    const mouthFeature = featureBoundsFor('mouth');
    const mouthWidth = mouthFeature
      ? Math.max(4, Math.min(10,
          Number(mouthFeature.width) * Math.abs(Number(liveModel.scale.x) || 1) * 0.72))
      : eyeWidth * 0.32;
    const mouthHeight = mouthFeature
      ? Math.max(3, Math.min(10,
          Number(mouthFeature.height) * Math.abs(Number(liveModel.scale.y) || 1) * 0.45))
      : eyeHeight * 0.8;
    const leftEye = featurePoint('eyes', 0.25, 0.5, 0.45, 0.18);
    const rightEye = featurePoint('eyes', 0.75, 0.5, 0.55, 0.18);
    const cheekLeft = featurePoint('face', 0.34, 0.68, 0.45, 0.23);
    const cheekRight = featurePoint('face', 0.66, 0.68, 0.55, 0.23);
    const mouth = featurePoint('mouth', 0.5, 0.5, 0.49, 0.245);
    geometryGuard.faceAnchorPoints = {{
      leftEye: pointRecord(leftEye), rightEye: pointRecord(rightEye),
      cheekLeft: pointRecord(cheekLeft), cheekRight: pointRecord(cheekRight),
      mouth: pointRecord(mouth)
    }};
    const expression = expressionState.name;
    const blinking = motionState.name === "blink";
    context.lineCap = 'round';
    if (blinking && eyeOverlayEnabled) {{
      // 眼睛 ArtMesh 的透明度负责真实闭眼，但虹膜仍有一部分烘焙在
      // Face ArtMesh；用紧贴眼眶的小椭圆遮住虹膜，再画轻量闭眼线。
      context.fillStyle = 'rgba(251,245,237,.9)';
      [leftEye, rightEye].forEach(function(point) {{
        if (!point) return;
        context.beginPath();
        context.ellipse(
          point.x,
          point.y + eyeHeight * 0.16,
          eyeWidth / 2,
          eyeHeight * 1.3,
          0,
          0,
          Math.PI * 2
        );
        context.fill();
      }});
      context.strokeStyle = 'rgba(91,60,75,.92)';
      context.lineWidth = Math.max(1.2, eyeHeight * 0.24);
      drawFaceCurve(context, leftEye, eyeWidth * 0.86, -eyeHeight * 0.48, 0, 0);
      drawFaceCurve(context, rightEye, eyeWidth * 0.86, -eyeHeight * 0.48, 0, 0);
      return;
    }}
    if (speechActive && mouthOverlayEnabled && mouth) {{
      const pulse = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(speechElapsed * 18));
      const syncActive = Boolean(
        motionSyncProfile && motionSyncState.viseme !== 'Silence' &&
        motionSyncState.intensity > 0
      );
      const syncOpen = syncActive
        ? Math.max(0, Math.min(1, motionSyncState.mouthOpenY))
        : pulse;
      const syncForm = syncActive
        ? Math.max(-1, Math.min(1, motionSyncState.mouthForm)) : 0;
      context.fillStyle = 'rgba(83,40,68,.64)';
      context.beginPath();
      context.ellipse(
        mouth.x,
        mouth.y,
        mouthWidth * (syncActive ? 0.42 + 0.16 * (1 - Math.abs(syncForm)) : 0.5),
        mouthHeight * (0.32 + 0.45 * syncOpen),
        0,
        0,
        Math.PI * 2
      );
      context.fill();
    }}
    if (expressionOverlayEnabled && (expression === "happy" || expression === "shy")) {{
      context.fillStyle = expression === "shy"
        ? 'rgba(255,125,157,.22)' : 'rgba(255,154,169,.16)';
      [cheekLeft, cheekRight].forEach(function(point) {{
        if (!point) return;
        context.beginPath();
        context.ellipse(point.x, point.y, eyeWidth * 0.45, eyeHeight * 0.42, 0, 0, Math.PI * 2);
        context.fill();
      }});
    }}
    if (expressionOverlayEnabled && expression === "surprised" && mouth) {{
      context.fillStyle = 'rgba(83,40,68,.72)';
      context.beginPath();
      context.ellipse(mouth.x, mouth.y, mouthWidth * 0.65, mouthHeight * 0.72, 0, 0, Math.PI * 2);
      context.fill();
    }}
    if (expressionOverlayEnabled && (expression === "sad" || expression === "curious")) {{
      context.strokeStyle = 'rgba(91,60,75,.76)';
      context.lineWidth = Math.max(1.2, eyeHeight * 0.24);
      const browLeft = featurePoint('brows', 0.25, 0.5, 0.45, 0.145);
      const browRight = featurePoint('brows', 0.75, 0.5, 0.55, 0.145);
      const browHeight = expression === "sad" ? -eyeHeight * 0.45 : eyeHeight * 0.45;
      drawFaceCurve(context, browLeft, eyeWidth * 0.68, browHeight, 0, 0);
      drawFaceCurve(context, browRight, eyeWidth * 0.68, browHeight, 0, 0);
    }}
  }}
  function applyProceduralExpression(name, immediate) {{
    const normalized = String(name || '').trim().toLowerCase();
    const profile = expressionProfiles[normalized];
    if (!profile) return false;
    if (!expressionState.procedural) {{
      // 从正式 exp3 切回过程表情时，以当前 Core 参数作为插值起点，
      // 不把旧的过程角度直接带入新表情。
      expressionState.currentAngle.x = readParameterValue("ParamAngleX", 0);
      expressionState.currentAngle.y = readParameterValue("ParamAngleY", 0);
    }}
    let changed = false;
    Object.keys(profile).forEach(function(id) {{
      changed = setParameterValue(id, profile[id]) || changed;
    }});
    // 只有当前模型没有真实眼睛参数绑定时才使用旧部件透明度回退；完整
    // 模型应保留自身眼睛 keyform 和物理，不再写死某个资源的 Part ID。
    const eyeOpen = Math.max(
      0.05,
      Math.min(1, (Number(profile.ParamEyeLOpen) + Number(profile.ParamEyeROpen)) / 2)
    );
    if (!parameterBindingState.ParamEyeLOpen && !parameterBindingState.ParamEyeROpen) {{
      changed = setPartOpacity("Eye_L", eyeOpen) || changed;
      changed = setPartOpacity("Eye_R", eyeOpen) || changed;
    }}
    const angles = expressionAngles[normalized] || expressionAngles.neutral;
    expressionState.targetAngle = {{
      x: (Number(angles.x) || 0) * expressionPoseScale,
      y: (Number(angles.y) || 0) * expressionPoseScale
    }};
    // 只更新目标，不在调用线程里按固定比例推进。固定比例会让低帧率/窗口
    // 移动时出现与帧率相关的角度跳变；实际插值统一由渲染时钟驱动。
    if (immediate) {{
      updateExpressionPose(1 / 60);
      changed = setParameterValue("ParamAngleX", expressionState.currentAngle.x) || changed;
      changed = setParameterValue("ParamAngleY", expressionState.currentAngle.y) || changed;
    }}
    if (!changed && liveModel && typeof liveModel.expression === 'function') {{
      try {{
        const result = liveModel.expression(normalized);
        changed = result !== false;
      }} catch (error) {{
        changed = false;
      }}
    }}
    if (changed) {{
      expressionState.name = normalized;
      expressionState.procedural = true;
    }}
    return changed;
  }}
  function updateExpressionPose(seconds) {{
    const dt = Math.max(1 / 240, Math.min(0.05, Number(seconds) || 0));
    // 指数形式的阻尼对 30/60/144Hz 都保持相近的实际响应时间。
    const blend = 1 - Math.exp(-dt * 9.0);
    expressionState.currentAngle.x +=
      (expressionState.targetAngle.x - expressionState.currentAngle.x) * blend;
    expressionState.currentAngle.y +=
      (expressionState.targetAngle.y - expressionState.currentAngle.y) * blend;
  }}
  function setCursorTarget(x, y, width, height) {{
    const normalizedWidth = Math.max(1, Number(width) || 1);
    const normalizedHeight = Math.max(1, Number(height) || 1);
    const rawX = Number(x);
    const rawY = Number(y);
    if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) return false;
    // 传入的是像素索引而不是长度；使用 width-1/height-1 让右下角也能
    // 到达精确边界，同时保留对极小窗口的除零保护。
    const pixelWidth = Math.max(1, normalizedWidth - 1);
    const pixelHeight = Math.max(1, normalizedHeight - 1);
    const normalizedX = Math.max(0, Math.min(1, rawX / pixelWidth));
    const normalizedY = Math.max(0, Math.min(1, rawY / pixelHeight));
    // 采用死区+有界增益，避免光标在中心附近抖动，也避免把光标目标
    // 叠加到 AngleX/AngleY 后超过当前模型探测出的安全姿态范围。
    const deadZone = 0.08;
    const horizontal = normalizedX - 0.5;
    const vertical = 0.5 - normalizedY;
    const remap = function(value) {{
      const sign = value < 0 ? -1 : 1;
      const magnitude = Math.max(0, Math.abs(value) - deadZone) / (0.5 - deadZone);
      // smoothstep 让接近中心和屏幕边缘的速度都连续，避免目标角度在
      // 死区边缘产生折线跳变。
      const shaped = magnitude * magnitude * (3 - 2 * magnitude);
      return sign * Math.min(1, shaped);
    }};
    const nextTargetX = remap(horizontal) * 1.55;
    const nextTargetY = remap(vertical) * 1.05;
    // 窗口移动时局部坐标可能在一帧内跨过整个视口；清空旧速度，
    // 让模型从当前姿态平滑追上新的屏幕位置而不是先向错误方向扭转。
    if (Math.hypot(nextTargetX - cursorState.targetX, nextTargetY - cursorState.targetY) > 1.1) {{
      // 窗口移动/跨屏时旧速度的方向没有参考价值；完全清零比保留一部分
      // 速度更可靠，避免新目标尚未到达前先向错误方向过冲。
      cursorState.velocityX = 0;
      cursorState.velocityY = 0;
      cursorState.targetGeneration += 1;
    }}
    cursorState.targetX = nextTargetX;
    cursorState.targetY = nextTargetY;
    cursorState.lastInputX = normalizedX;
    cursorState.lastInputY = normalizedY;
    return true;
  }}
  function clampValue(value, minimum, maximum) {{
    return Math.max(minimum, Math.min(maximum, Number(value) || 0));
  }}
  function smoothTrackingAxis(current, target, velocity, seconds) {{
    // 临界阻尼追踪：比单纯 lerp 更不容易在跨屏/窗口移动时过冲。
    const dt = Math.max(1 / 240, Math.min(0.05, Number(seconds) || 0));
    const smoothTime = 0.18;
    const maxSpeed = 12;
    const omega = 2 / smoothTime;
    const x = omega * dt;
    const decay = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
    const originalTarget = Number(target) || 0;
    let change = (Number(current) || 0) - originalTarget;
    const maxChange = maxSpeed * smoothTime;
    change = clampValue(change, -maxChange, maxChange);
    const adjustedTarget = (Number(current) || 0) - change;
    const temporary = ((Number(velocity) || 0) + omega * change) * dt;
    let nextVelocity = ((Number(velocity) || 0) - omega * temporary) * decay;
    let nextValue = adjustedTarget + (change + temporary) * decay;
    // 过冲时直接落在目标上，并清零速度；这一步对窗口拖动尤其重要，
    // 否则模型会在光标离开后继续摆动并短暂穿过安全姿态。
    if ((originalTarget - (Number(current) || 0)) * (nextValue - originalTarget) < 0) {{
      nextValue = originalTarget;
      nextVelocity = 0;
    }}
    return {{ value: nextValue, velocity: clampValue(nextVelocity, -maxSpeed, maxSpeed) }};
  }}
  function poseParameterLimit(id, fallback) {{
    const safety = parameterSafetyState[id];
    if (!safety) return fallback;
    const sourceLimit = Math.min(
      Math.abs(Number(safety.min) || 0), Math.abs(Number(safety.max) || 0)
    );
    if (!(sourceLimit > 0)) return fallback;
    // 留出约 22% 的原始范围给模型自身的物理/动作叠加，避免把 CDI
    // 的边界值当成可长期稳定的注视姿态。
    return Math.max(0.5, Math.min(fallback, sourceLimit * 0.78));
  }}
  function slewPose(current, target, seconds, maxRate) {{
    const dt = Math.max(1 / 240, Math.min(0.05, Number(seconds) || 0));
    const step = Math.max(0.05, Number(maxRate) * dt);
    return (Number(current) || 0) + clampValue(
      (Number(target) || 0) - (Number(current) || 0), -step, step
    );
  }}
  function updateFacingPose(seconds) {{
    const dt = Math.max(1 / 240, Math.min(0.05, Number(seconds) || 0));
    // 姿态先经过中性区，再提交几何镜像。若 scale.x 先翻转而 ParamAngleX
    // 仍处于旧方向，发丝/手臂会在同一帧跨过相邻 ArtMesh，表现为穿模。
    const step = Math.min(0.18, Math.max(0.04, dt * 6.0));
    const delta = Number(facingState.target) - Number(facingState.current);
    facingState.current += clampValue(delta, -step, step);
    if (Math.abs(facingState.target - facingState.current) < 0.01) {{
      facingState.current = facingState.target;
    }}
    if (facingState.sign > 0 && facingState.current <= -FACING_SWITCH_THRESHOLD) {{
      facingState.sign = -1;
      resetGeometryContinuity(2);
    }} else if (facingState.sign < 0 && facingState.current >= FACING_SWITCH_THRESHOLD) {{
      facingState.sign = 1;
      resetGeometryContinuity(2);
    }}
    facing = facingState.current;
    poseState.facingCurrent = facingState.current;
    return facing;
  }}
  function geometryFacingSign() {{
    return Number(facingState.sign) < 0 ? -1 : 1;
  }}
  function applyFormalParameterMotion(seconds) {{
    if (!liveModel) return false;
    const dt = Math.max(0, Number(seconds) || 0);
    if (poseState.mode !== 'formal') {{
      poseState.currentX = 0;
      poseState.currentY = 0;
      poseState.formalAppliedX = 0;
      poseState.formalAppliedY = 0;
      poseState.formalWrittenX = null;
      poseState.formalWrittenY = null;
      poseState.mode = 'formal';
    }}
    if (dt > 0) updateFacingPose(dt);
    const safetyX = parameterSafetyState.ParamAngleX;
    const safetyY = parameterSafetyState.ParamAngleY;
    const minX = safetyX && Number.isFinite(Number(safetyX.sourceMin))
      ? Number(safetyX.sourceMin) : -10.2;
    const maxX = safetyX && Number.isFinite(Number(safetyX.sourceMax))
      ? Number(safetyX.sourceMax) : 10.2;
    const minY = safetyY && Number.isFinite(Number(safetyY.sourceMin))
      ? Number(safetyY.sourceMin) : -10.2;
    const maxY = safetyY && Number.isFinite(Number(safetyY.sourceMax))
      ? Number(safetyY.sourceMax) : 10.2;
    let baseX = readParameterValue("ParamAngleX", 0);
    let baseY = readParameterValue("ParamAngleY", 0);
    // 某些 motion manager 不会在每个 update 完整覆盖角度参数；若读到的
    // 值仍是上一帧写入值，先减去上一份注视增量，避免 M+C→M+2C 漂移。
    if (poseState.formalWrittenX !== null &&
        Math.abs(baseX - poseState.formalWrittenX) < 0.001) {{
      baseX -= poseState.formalAppliedX;
    }}
    if (poseState.formalWrittenY !== null &&
        Math.abs(baseY - poseState.formalWrittenY) < 0.001) {{
      baseY -= poseState.formalAppliedY;
    }}
    baseX = clampValue(baseX, minX, maxX);
    baseY = clampValue(baseY, minY, maxY);
    const budgetX = Math.max(0, Math.min(7.4, Math.max(maxX - baseX, baseX - minX)));
    const budgetY = Math.max(0, Math.min(6.4, Math.max(maxY - baseY, baseY - minY)));
    const targetX = clampValue(cursorState.currentX, -budgetX, budgetX);
    const targetY = clampValue(cursorState.currentY, -budgetY, budgetY);
    poseState.appliedCursorX = targetX;
    poseState.appliedCursorY = targetY;
    poseState.desiredX = targetX;
    poseState.desiredY = targetY;
    if (dt > 0) {{
      poseState.currentX = slewPose(poseState.currentX, targetX, dt, 14);
      poseState.currentY = slewPose(poseState.currentY, targetY, dt, 13.5);
    }}
    const outputX = clampValue(baseX + poseState.currentX, minX, maxX);
    const outputY = clampValue(baseY + poseState.currentY, minY, maxY);
    poseState.formalAppliedX = outputX - baseX;
    poseState.formalAppliedY = outputY - baseY;
    poseState.formalWrittenX = outputX;
    poseState.formalWrittenY = outputY;
    if (parameterBindingState.ParamAngleX) {{
      setParameterValue("ParamAngleX", outputX, false);
    }}
    if (parameterBindingState.ParamAngleY) {{
      setParameterValue("ParamAngleY", outputY, false);
    }}
    return true;
  }}
  function applyProceduralParameterMotion(seconds, advancePose = true) {{
    if (!liveModel) return false;
    if (poseState.mode !== 'procedural') {{
      poseState.currentX = 0;
      poseState.currentY = 0;
      poseState.formalAppliedX = 0;
      poseState.formalAppliedY = 0;
      poseState.formalWrittenX = null;
      poseState.formalWrittenY = null;
    }}
    poseState.mode = 'procedural';
    const dt = Math.max(0, Number(seconds) || 0);
    const elapsed = Math.max(0, Number(motionState.elapsed) || 0);
    // 指令函数会用 seconds=0 预览下一状态；预览不能偷偷推进时间，
    // 否则快速 A/B 切换会在一个渲染 tick 内累积多次朝向步进。
    const poseDt = advancePose ? dt : 0;
    if (poseDt > 0) {{
      updateExpressionPose(poseDt);
      updateFacingPose(poseDt);
    }}
    let motionAngleX = 0;
    let motionAngleY = 0;
    const procedural = motionState.procedural && proceduralMotions.has(motionState.name);
    const proceduralExpression = expressionState.procedural;
    // 动作之间共享同一组 CDI 参数；每帧先清理上一个动作的残值，
    // 再叠加当前动作，避免 wave 的身体旋转泄漏到 walk/idle。
    if (procedural) {{
      setParameterValue("ParamBreath", 0);
      setParameterValue("ParamHairFront", 0);
      setParameterValue("ParamHairSide", 0);
      setParameterValue("ParamBodyAngleX", 0);
      setParameterValue("ParamBodyAngleY", 0);
      setParameterValue("ParamBodyAngleZ", 0);
    }}
    if (procedural && motionState.name === "idle") {{
      const phase = elapsed * 2 * Math.PI;
      setParameterValue("ParamBreath", 0.5 + 0.15 * Math.sin(phase / 3.2));
      setParameterValue("ParamHairFront", 0.08 * Math.sin(phase / 2.4));
      setParameterValue("ParamHairSide", 0.06 * Math.sin(phase / 2.8 + 0.7));
    }} else if (procedural && motionState.name === "walk") {{
      const phase = elapsed * 12;
      setParameterValue("ParamBodyAngleX", 2.5 * Math.sin(phase));
      setParameterValue("ParamBodyAngleY", 1.2 * Math.cos(phase));
      setParameterValue("ParamHairSide", 0.15 * Math.sin(phase));
      // BodyAngle/Hair 未绑定时仍用受限 AngleX/Y 做小幅摆动；绑定存在时
      // 参数本身也会生效，外层变换始终不改变 x/y 两轴缩放比例。
      motionAngleX += 2.6 * Math.sin(phase);
      motionAngleY += 0.8 * Math.cos(phase);
    }} else if (procedural && motionState.name === "wave") {{
      const phase = elapsed * Math.PI * 2 / 0.9;
      setParameterValue("ParamBodyAngleZ", 1.2 * Math.sin(phase));
      setParameterValue("ParamHairFront", 0.1 * Math.sin(phase));
      motionAngleX += 2.8 * Math.sin(phase);
      motionAngleY += 0.9 * Math.cos(phase);
    }} else if (procedural && motionState.name === "blink") {{
      const phase = Math.min(1, elapsed / 0.45);
      // 让闭眼阶段保持足够长，避免 180ms 的截图/真实帧只采到半透明
      // 眼睛，看起来像眼睛闪烁而不是一次完整眨眼。
      const openness = phase < 0.35
        ? 1 - phase / 0.35
        : phase < 0.72
          ? 0
          : (phase - 0.72) / 0.28;
      setParameterValue("ParamEyeLOpen", openness);
      setParameterValue("ParamEyeROpen", openness);
      if (!parameterBindingState.ParamEyeLOpen && !parameterBindingState.ParamEyeROpen) {{
        setPartOpacity("Eye_L", openness);
        setPartOpacity("Eye_R", openness);
      }}
      // 闭眼参数在当前 MOC3 中没有可见绑定，用极轻的点头作为可见回馈。
      motionAngleY -= 1.5 * Math.sin(Math.PI * phase);
    }}
    // 先给表达式/正式动作留出余量，再把光标贡献限制在剩余预算内。
    // 这样不会出现“表情一切换，光标角度把脸拧过头”的穿模瞬间。
    const limitX = poseParameterLimit("ParamAngleX", 7.4);
    const limitY = poseParameterLimit("ParamAngleY", 6.4);
    // 正式 motion3/exp3 由 Cubism 管理器写入参数；兼容过程动作才使用
    // 本地表达式角度。正式分支只在安全范围内叠加注视，不覆盖动作自身的
    // keyform，避免动作被冻结或在一帧内跳到错误姿态。
    const formalAngleX = readParameterValue("ParamAngleX", 0);
    const formalAngleY = readParameterValue("ParamAngleY", 0);
    const baseAngleX = proceduralExpression
      ? clampValue(Number(expressionState.currentAngle.x) + motionAngleX, -limitX, limitX)
      : clampValue(formalAngleX + motionAngleX, -limitX, limitX);
    const baseAngleY = proceduralExpression
      ? clampValue(Number(expressionState.currentAngle.y) + motionAngleY, -limitY, limitY)
      : clampValue(formalAngleY + motionAngleY, -limitY, limitY);
    const cursorBudgetX = Math.max(0, limitX - Math.abs(baseAngleX));
    const cursorBudgetY = Math.max(0, limitY - Math.abs(baseAngleY));
    poseState.appliedCursorX = clampValue(cursorState.currentX, -cursorBudgetX, cursorBudgetX);
    poseState.appliedCursorY = clampValue(cursorState.currentY, -cursorBudgetY, cursorBudgetY);
    poseState.desiredX = clampValue(baseAngleX + poseState.appliedCursorX, -limitX, limitX);
    poseState.desiredY = clampValue(baseAngleY + poseState.appliedCursorY, -limitY, limitY);
    poseState.currentX = poseDt > 0
      ? slewPose(poseState.currentX, poseState.desiredX, poseDt, 14)
      : poseState.currentX;
    poseState.currentY = poseDt > 0
      ? slewPose(poseState.currentY, poseState.desiredY, poseDt, 13.5)
      : poseState.currentY;
    // 动作与光标叠加后再次限幅，避免姿态参数超出当前导出模型的安全范围
    // 造成关节穿模、局部拉伸或模型边界突然跳出窗口。
    const angleX = clampValue(poseState.currentX, -limitX, limitX);
    const angleY = clampValue(poseState.currentY, -limitY, limitY);
    // 镜像朝向时反转横向角度，避免角色转身后姿态方向相反。
    if (procedural || proceduralExpression || parameterBindingState.ParamAngleX) {{
      setParameterValue("ParamAngleX", angleX * facing);
    }}
    if (procedural || proceduralExpression || parameterBindingState.ParamAngleY) {{
      setParameterValue("ParamAngleY", angleY);
    }}
    if (proceduralExpression && motionState.name !== "blink") {{
      const profile = expressionProfiles[expressionState.name] || expressionProfiles.neutral;
      const eyeOpen = Math.max(
        0.05,
        Math.min(1, (Number(profile.ParamEyeLOpen) + Number(profile.ParamEyeROpen)) / 2)
      );
      if (!parameterBindingState.ParamEyeLOpen && !parameterBindingState.ParamEyeROpen) {{
        setPartOpacity("Eye_L", eyeOpen);
        setPartOpacity("Eye_R", eyeOpen);
      }}
    }}
    const geometryReady = refreshCoreGeometry();
    // fitModelToViewport 内部执行按 dynamicFlags 缓存的几何审计。这里不再
    // 额外全量扫描一次全部顶点，避免每帧重复做相同的 O(vertex) 校验。
    const fitted = geometryReady && fitModelToViewport();
    const renderOrderReady = fitted && repairRenderOrders(false);
    if (!fitted || !renderOrderReady) {{
      // 视口或模型矩阵出现 NaN/越界时宁可暂时隐藏当前帧，也不把坏矩阵
      // 交给 WebGL；下一帧仍会重新测量并尝试恢复。
      liveModel.visible = false;
      return false;
    }}
    liveModel.visible = true;
    drawFaceOverlay();
    return true;
  }}
  function viewportMargin(width, height) {{
    return Math.max(8, Math.min(16, Math.min(width, height) * 0.04));
  }}
  function refreshCoreGeometry() {{
    const internal = liveModel && liveModel.internalModel;
    const core = internal && internal.coreModel;
    if (!core) return true;
    try {{
      // 自定义表情/注视写入发生在 internalModel.update() 之后；立即重算
      // Core 顶点，保证本帧 fit/hit/截图看到的是同一份姿态，而不是上一帧。
      // 优先调用原始 Core 模型，保留 vertexPositionsDidChange 标志，让
      // Pixi 下一次绘制真正上传新网格；包装层 update() 会立即清除这些标志。
      const raw = typeof core.getModel === 'function' ? core.getModel() : null;
      if (raw && typeof raw.update === 'function') raw.update();
      else if (typeof core.update === 'function') core.update();
      else return false;
      return true;
    }} catch (error) {{
      geometryGuard.geometryValid = false;
      geometryGuard.invalidFrames += 1;
      return false;
    }}
  }}
  function flushCoreParameterGeometry() {{
    if (!coreParameterDirty) return true;
    // 合并同一帧中的表情、注视、口型和动作参数写入；失败时保留脏标记，
    // 下一帧仍会重试并由几何护栏隔离坏帧。
    if (!refreshCoreGeometry()) return false;
    coreParameterDirty = false;
    return true;
  }}
  const facePartPatterns = {{
    eyes: /眼|睫|眼线|眼影|瞳|eye/i,
    mouth: /嘴|口|唇|mouth/i,
    brows: /眉|brow/i,
    face: /脸|五官|鼻|face/i
  }};
  function newBoundsAccumulator() {{
    return {{ minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity }};
  }}
  function addBoundsPoint(accumulator, x, y) {{
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    accumulator.minX = Math.min(accumulator.minX, x);
    accumulator.minY = Math.min(accumulator.minY, y);
    accumulator.maxX = Math.max(accumulator.maxX, x);
    accumulator.maxY = Math.max(accumulator.maxY, y);
  }}
  function finishBoundsAccumulator(accumulator) {{
    if (!accumulator || !Number.isFinite(accumulator.minX) ||
        !Number.isFinite(accumulator.minY) ||
        !(accumulator.maxX > accumulator.minX) ||
        !(accumulator.maxY > accumulator.minY)) return null;
    return {{
      x: accumulator.minX, y: accumulator.minY,
      width: accumulator.maxX - accumulator.minX,
      height: accumulator.maxY - accumulator.minY
    }};
  }}
  function drawableGeometryBounds() {{
    const internal = liveModel && liveModel.internalModel;
    const core = internal && internal.coreModel;
    const raw = core && typeof core.getModel === 'function' ? core.getModel() : null;
    const drawables = raw && raw.drawables;
    const canvasInfo = raw && raw.canvasinfo;
    const pixelsPerUnit = Number(internal && internal.pixelsPerUnit) ||
      Number(canvasInfo && canvasInfo.PixelsPerUnit);
    const canvasWidth = Number(internal && internal.originalWidth) ||
      Number(canvasInfo && canvasInfo.CanvasWidth);
    const canvasHeight = Number(internal && internal.originalHeight) ||
      Number(canvasInfo && canvasInfo.CanvasHeight);
    const originX = Number(canvasInfo && canvasInfo.CanvasOriginX);
    const originY = Number(canvasInfo && canvasInfo.CanvasOriginY);
    geometryGuard.faceBounds = null;
    geometryGuard.faceFeatureBounds = {{ eyes: null, mouth: null, brows: null, face: null }};
    if (!drawables || !(pixelsPerUnit > 0) || !(canvasWidth > 0) || !(canvasHeight > 0))
      return null;
    const resolvedOriginX = Number.isFinite(originX) ? originX : canvasWidth / 2;
    const resolvedOriginY = Number.isFinite(originY) ? originY : canvasHeight / 2;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let vertexCount = 0;
    const featureAccumulators = {{
      eyes: newBoundsAccumulator(),
      mouth: newBoundsAccumulator(),
      brows: newBoundsAccumulator(),
      face: newBoundsAccumulator()
    }};
    const parentPartIndices = drawables.parentPartIndices;
    const partIds = raw && raw.parts && raw.parts.ids;
    const count = Number(drawables.count) || 0;
    for (let drawable = 0; drawable < count; drawable += 1) {{
      if (drawables.opacities) {{
        const opacity = Number(drawables.opacities[drawable]);
        if (!Number.isFinite(opacity)) return null;
        if (opacity <= 0.001) continue;
      }}
      const positions = drawables.vertexPositions && drawables.vertexPositions[drawable];
      if (!positions || typeof positions.length !== 'number') continue;
      const declaredCount = Number(drawables.vertexCounts && drawables.vertexCounts[drawable]);
      const limit = Number.isInteger(declaredCount) && declaredCount > 0
        ? Math.min(positions.length, declaredCount * 2) : positions.length;
      if (limit < 2 || limit % 2 !== 0) continue;
      const parentIndex = Number(parentPartIndices && parentPartIndices[drawable]);
      const hasParent = Number.isInteger(parentIndex) && parentIndex >= 0 &&
        parentIndex < modelPartLabels.length;
      const partId = partIds && hasParent ? partIds[parentIndex] : '';
      const partLabel = String(hasParent ? (modelPartLabels[parentIndex] || partId || '') : '');
      const featureNames = Object.keys(facePartPatterns).filter(function(name) {{
        return facePartPatterns[name].test(partLabel);
      }});
      for (let item = 0; item + 1 < limit; item += 2) {{
        const rawX = Number(positions[item]);
        const rawY = Number(positions[item + 1]);
        if (!Number.isFinite(rawX) || !Number.isFinite(rawY)) return null;
        const x = rawX * pixelsPerUnit + resolvedOriginX;
        const y = -rawY * pixelsPerUnit + resolvedOriginY;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
        vertexCount += 1;
        for (const featureName of featureNames) {{
          addBoundsPoint(featureAccumulators[featureName], x, y);
        }}
      }}
    }}
    if (!Number.isFinite(minX) || !Number.isFinite(minY) ||
        !Number.isFinite(maxX) || !Number.isFinite(maxY) ||
        !(maxX > minX) || !(maxY > minY) || vertexCount < 3) return null;
    const bounds = {{ x: minX, y: minY, width: maxX - minX, height: maxY - minY,
      vertexCount }};
    const featureBounds = {{
      eyes: finishBoundsAccumulator(featureAccumulators.eyes),
      mouth: finishBoundsAccumulator(featureAccumulators.mouth),
      brows: finishBoundsAccumulator(featureAccumulators.brows),
      face: finishBoundsAccumulator(featureAccumulators.face)
    }};
    geometryGuard.faceFeatureBounds = featureBounds;
    geometryGuard.faceBounds = featureBounds.face || featureBounds.eyes ||
      featureBounds.mouth || featureBounds.brows || null;
    if (featureBounds.eyes || featureBounds.mouth || featureBounds.brows) {{
      const combined = newBoundsAccumulator();
      for (const feature of Object.values(featureBounds)) {{
        if (!feature) continue;
        addBoundsPoint(combined, Number(feature.x), Number(feature.y));
        addBoundsPoint(
          combined,
          Number(feature.x) + Number(feature.width),
          Number(feature.y) + Number(feature.height)
        );
      }}
      geometryGuard.faceBounds = featureBounds.face || finishBoundsAccumulator(combined);
    }}
    if (!checkGeometryContinuity(bounds)) return null;
    return bounds;
  }}
  function worldBoundsFromLocal(localBounds) {{
    if (!finiteBounds(localBounds) || !liveModel ||
        typeof liveModel.toGlobal !== 'function' || !window.PIXI || !PIXI.Point) return null;
    const right = Number(localBounds.x) + Number(localBounds.width);
    const bottom = Number(localBounds.y) + Number(localBounds.height);
    const corners = [
      [Number(localBounds.x), Number(localBounds.y)],
      [right, Number(localBounds.y)], [Number(localBounds.x), bottom], [right, bottom]
    ];
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    try {{
      for (const corner of corners) {{
        const point = liveModel.toGlobal(new PIXI.Point(corner[0], corner[1]));
        const x = Number(point && point.x);
        const y = Number(point && point.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }}
    }} catch (error) {{
      return null;
    }}
    if (!(maxX > minX) || !(maxY > minY)) return null;
    return {{ x: minX, y: minY, width: maxX - minX, height: maxY - minY }};
  }}
  function drawableDynamicFlags() {{
    const internal = liveModel && liveModel.internalModel;
    const core = internal && internal.coreModel;
    const raw = core && typeof core.getModel === 'function' ? core.getModel() : null;
    const drawables = raw && raw.drawables;
    const flags = drawables && drawables.dynamicFlags;
    const utils = window.Live2DCubismCore && window.Live2DCubismCore.Utils;
    if (!drawables || !flags || typeof flags.length !== 'number' || !utils) return null;
    const methods = {{
      vertexPositionsDidChange: 'hasVertexPositionsDidChangeBit',
      visibilityDidChange: 'hasVisibilityDidChangeBit',
      opacityDidChange: 'hasOpacityDidChangeBit',
      renderOrderDidChange: 'hasRenderOrderDidChangeBit'
    }};
    const result = {{}};
    for (const [name, method] of Object.entries(methods)) {{
      const check = utils[method];
      if (typeof check !== 'function') return null;
      let changed = false;
      for (let index = 0; index < flags.length; index += 1) {{
        try {{
          if (Boolean(check.call(utils, Number(flags[index])))) {{
            changed = true;
            break;
          }}
        }} catch (error) {{
          return null;
        }}
      }}
      result[name] = changed;
    }}
    return result;
  }}
  function auditGeometry(force = false) {{
    const now = performance.now();
    const flags = drawableDynamicFlags();
    geometryAuditState.lastDynamicFlags = flags;
    if (flags && flags.vertexPositionsDidChange) {{
      renderPerformanceState.geometryVertexDirtyFrames += 1;
    }}
    const vertexDirty = Boolean(flags && flags.vertexPositionsDidChange);
    const structuralDirty = Boolean(flags && (
      flags.visibilityDidChange || flags.opacityDidChange ||
      flags.renderOrderDidChange
    ));
    if (structuralDirty) renderPerformanceState.geometryStructuralDirtyFrames += 1;
    const due = Boolean(
      force || vertexDirty || structuralDirty || geometryAuditState.force ||
      !geometryAuditState.cacheValid ||
      !geometryAuditState.lastAuditAt ||
      now - geometryAuditState.lastAuditAt >=
        renderPerformanceState.geometryAuditIntervalMs
    );

    if (!due) {{
      renderPerformanceState.geometryAuditCacheHits += 1;
      return Boolean(geometryGuard.geometryValid);
    }}
    renderPerformanceState.geometryAuditCacheMisses += 1;
    renderPerformanceState.geometryAudits += 1;
    geometryAuditState.force = false;
    geometryAuditState.lastAuditAt = now;
    const frameValid = validateDrawableFrame();
    const bounds = drawableGeometryBounds();
    geometryGuard.dynamicBounds = finiteBounds(bounds) ? bounds : null;
    geometryGuard.worldBounds = null;
    if (!frameValid || !geometryGuard.continuityValid) {{
      geometryAuditState.cacheValid = false;
      geometryAuditState.force = true;
      return false;
    }}
    geometryAuditState.cacheValid = true;
    return true;
  }}
  function validateDrawableFrame() {{
    const internal = liveModel && liveModel.internalModel;
    const core = internal && internal.coreModel;
    const raw = core && typeof core.getModel === 'function' ? core.getModel() : null;
    const drawables = raw && raw.drawables;
    if (!drawables) {{
      geometryGuard.geometryValid = true;
      return true;
    }}
    const count = Number(drawables.count) || 0;
    const maskCounts = drawables.maskCounts;
    const masks = drawables.masks;
    if (!Number.isInteger(count) || count <= 0) {{
      geometryGuard.geometryValid = false;
      geometryGuard.invalidFrames += 1;
      return false;
    }}
    for (let drawable = 0; drawable < count; drawable += 1) {{
      const positions = drawables.vertexPositions && drawables.vertexPositions[drawable];
      const indices = drawables.indices && drawables.indices[drawable];
      const vertexCount = Number(drawables.vertexCounts && drawables.vertexCounts[drawable]) || 0;
      if (!positions || !indices || vertexCount < 3 ||
          positions.length < vertexCount * 2 || indices.length < 3) continue;
      for (let item = 0; item < vertexCount * 2; item += 1) {{
        if (!Number.isFinite(Number(positions[item]))) {{
          geometryGuard.geometryValid = false;
          geometryGuard.invalidFrames += 1;
          return false;
        }}
      }}
      for (let item = 0; item < indices.length; item += 1) {{
        const index = Number(indices[item]);
        if (!Number.isInteger(index) || index < 0 || index >= vertexCount) {{
          geometryGuard.geometryValid = false;
          geometryGuard.invalidFrames += 1;
          return false;
        }}
      }}
      if (drawables.opacities && !Number.isFinite(Number(drawables.opacities[drawable]))) {{
        geometryGuard.geometryValid = false;
        geometryGuard.invalidFrames += 1;
        return false;
      }}
      // 裁剪网格索引同样属于 Core 几何契约。索引越界或 mask 数量与
      // 数组不一致时，Pixi 可能把头发/身体采样到错误的遮罩，表现为
      // 局部穿模；可验证时直接拒绝当前帧，缺少该字段的旧 Core 继续
      // 使用其自身的裁剪实现。
      if (maskCounts && masks) {{
        const maskCount = Number(maskCounts[drawable]);
        const maskList = masks[drawable];
        if (!Number.isInteger(maskCount) || maskCount < 0 ||
            (maskCount > 0 && (!maskList || maskList.length < maskCount))) {{
          geometryGuard.geometryValid = false;
          geometryGuard.invalidFrames += 1;
          return false;
        }}
        for (let mask = 0; mask < maskCount; mask += 1) {{
          const maskIndex = Number(maskList[mask]);
          if (!Number.isInteger(maskIndex) || maskIndex < 0 || maskIndex >= count ||
              maskIndex === drawable) {{
            geometryGuard.geometryValid = false;
            geometryGuard.invalidFrames += 1;
            return false;
          }}
        }}
      }}
    }}
    geometryGuard.geometryValid = true;
    return true;
  }}
  function finiteBounds(bounds) {{
    if (!bounds) return false;
    return [bounds.x, bounds.y, bounds.width, bounds.height].every(function(value) {{
      return Number.isFinite(Number(value));
    }}) && Number(bounds.width) > 0 && Number(bounds.height) > 0;
  }}
  function fitModelToViewport() {{
    if (!liveModel || !pixiApp) return false;
    if (!enforceUniformTransform()) return false;
    const canvas = document.getElementById('meapet-live2d');
    const width = Math.max(1, Number(canvas && canvas.clientWidth) || window.innerWidth || 1);
    const height = Math.max(1, Number(canvas && canvas.clientHeight) || window.innerHeight || 1);
    const margin = viewportMargin(width, height);
    const getBounds = function() {{
      return typeof liveModel.getBounds === 'function' ? liveModel.getBounds() : null;
    }};
    // 顶点只在 Core 更新后扫描一次；之后仅把四个角映射到世界坐标，
    // 避免为了 Shape/fit 在每个 pass 反复遍历纹理网格。
    if (!auditGeometry(false)) return false;
    const dynamicLocalBounds = finiteBounds(geometryGuard.dynamicBounds)
      ? geometryGuard.dynamicBounds : null;
    // ``geometryGuard.dynamicBounds`` 返回 null 时可能是资源缺失，也可能是
    // 连续性护栏拒绝了当前帧。后者不能落回静态 bounds 继续绘制，否则
    // 坏网格仍会进入 WebGL，只是布局矩形看起来正常。
    if (!geometryGuard.continuityValid) return false;
    const getLayoutBounds = function() {{
      const staticBounds = getBounds();
      const dynamicBounds = worldBoundsFromLocal(dynamicLocalBounds);
      if (finiteBounds(dynamicBounds)) {{
        geometryGuard.dynamicBounds = dynamicLocalBounds;
        geometryGuard.worldBounds = dynamicBounds;
      }} else {{
        geometryGuard.dynamicBoundsFallbacks += 1;
        geometryGuard.worldBounds = null;
      }}
      // 动态网格用于越界检查和真实命中区域；布局的首次基准仍由
      // 模型画布边界决定，避免动作或透明留白变化造成角色突然放大。
      return finiteBounds(dynamicBounds) ? dynamicBounds : staticBounds;
    }};
    const base = liveModel.__meapetBaseTransform;
    const restoreLastGood = function() {{
    const saved = base && base.lastGoodTransform;
      if (!saved) return;
      liveModel.x = saved.x;
      liveModel.y = saved.y;
      liveModel.rotation = saved.rotation;
      const restoreScale = Math.max(
        0.001,
        Math.max(Math.abs(Number(saved.scaleX) || 0), Math.abs(Number(saved.scaleY) || 0))
      );
      liveModel.scale.set(geometryFacingSign() * restoreScale, restoreScale);
      base.scale = saved.scale;
      base.safeOffsetX = saved.safeOffsetX;
      base.safeOffsetY = saved.safeOffsetY;
    }};
    const fail = function() {{
      restoreLastGood();
      return false;
    }};
    let bounds = getLayoutBounds();
    if (!base) return false;
    if (!finiteBounds(bounds)) return fail();
    const availableWidth = Math.max(1, width - margin * 2);
    const availableHeight = Math.max(1, height - margin * 2);
    const naturalLayout = __NATURAL_LAYOUT__;
    // 自然布局沿用模型画布基准；默认渲染器仍保留动态可见网格拟合。
    if (!naturalLayout && !base.visibleReferenceReady && finiteBounds(dynamicLocalBounds)) {{
      const canvasReferenceScale = Math.max(
        0.001,
        Number(base.canvasReferenceScale) || Number(base.referenceScale) ||
          Math.abs(Number(liveModel.scale.x)) || 1
      );
      const visibleFitScale = Math.max(
        0.001,
        Math.min(
          availableWidth / Number(dynamicLocalBounds.width),
          availableHeight / Number(dynamicLocalBounds.height)
        ) * 0.92
      );
      const visibleReferenceScale = Math.max(
        canvasReferenceScale,
        Math.min(canvasReferenceScale * 2, visibleFitScale)
      );
      base.referenceScale = visibleReferenceScale;
      base.scale = visibleReferenceScale;
      base.visibleReferenceReady = true;
      base.visibleReferenceLocalBounds = {{
        x: Number(dynamicLocalBounds.x), y: Number(dynamicLocalBounds.y),
        width: Number(dynamicLocalBounds.width), height: Number(dynamicLocalBounds.height)
      }};
      liveModel.scale.set(
        geometryFacingSign() * visibleReferenceScale,
        visibleReferenceScale
      );
      const visibleBounds = worldBoundsFromLocal(base.visibleReferenceLocalBounds);
      if (finiteBounds(visibleBounds)) {{
        const alignX = width / 2 -
          (Number(visibleBounds.x) + Number(visibleBounds.width) / 2);
        const alignY = height - margin -
          (Number(visibleBounds.y) + Number(visibleBounds.height));
        if (Number.isFinite(alignX)) base.x = Number(liveModel.x) + alignX;
        if (Number.isFinite(alignY)) base.y = Number(liveModel.y) + alignY;
        liveModel.x = Number(base.x);
        liveModel.y = Number(base.y);
      }}
    }}
    // 每次拟合前先移除上一帧的安全位移。这样修正量不会累积，窗口
    // 自主移动/行走结束后模型会回到稳定锚点，而不是逐帧漂移到边缘。
    const motionX = Number(base.motionOffsetX) || 0;
    const motionY = Number(base.motionOffsetY) || 0;
    liveModel.x = Number(base.x) + motionX;
    liveModel.y = Number(base.y) + motionY;
    base.safeOffsetX = 0;
    base.safeOffsetY = 0;
    bounds = getLayoutBounds();
    if (!finiteBounds(bounds)) return fail();
    const referenceScale = Math.max(
      0.001,
      Number(base.referenceScale) || Math.abs(Number(liveModel.scale.x)) || 1
    );
    let scale = Math.max(0.001, Math.abs(Number(liveModel.scale.x)) || Number(base.scale) || 1);
    // 最多迭代两次：第一次处理姿态导致的溢出并保持严格等比，第二次
    // 重新取 bounds 后处理缩放引起的边界变化。
    for (let pass = 0; pass < 2; pass += 1) {{
      const fitRatio = Math.min(
        availableWidth / Math.max(1, Number(bounds.width)),
        availableHeight / Math.max(1, Number(bounds.height))
      );
      let nextScale = scale;
      if (fitRatio < 0.995) {{
        // 溢出立即收缩，并额外留 0.5% 的抗抖余量，避免刚好贴边时
        // 在相邻帧之间反复放大/缩小。
        nextScale = Math.max(0.001, scale * Math.min(1, fitRatio) * 0.995);
      }} else if (fitRatio > 1.015 && scale < referenceScale) {{
        // 有余量时慢慢恢复到窗口基准，不让一次大姿态跳变永久缩小模型。
        const recovery = Math.min(0.02, (fitRatio - 1) * 0.12);
        nextScale = Math.min(referenceScale, scale * (1 + recovery));
      }}
      nextScale = Math.min(referenceScale, Math.max(0.001, nextScale));
      const desiredScaleX = geometryFacingSign() * nextScale;
      const scaleSignChanged = Math.sign(Number(liveModel.scale.x) || 1) !==
        Math.sign(desiredScaleX);
      if (Math.abs(nextScale - scale) > 0.00005 || scaleSignChanged) {{
        scale = nextScale;
        base.scale = scale;
        // x/y 使用同一绝对缩放值；只在安全朝向切换时更新符号，
        // 防止 sign 状态与实际 transform 错开一帧。
        liveModel.scale.set(geometryFacingSign() * scale, scale);
        if (!enforceUniformTransform()) return fail();
        bounds = getLayoutBounds();
        if (!finiteBounds(bounds)) return fail();
      }}
      const left = Number(bounds.x) || 0;
      const top = Number(bounds.y) || 0;
      const right = left + Math.max(0, Number(bounds.width) || 0);
      const bottom = top + Math.max(0, Number(bounds.height) || 0);
      let offsetX = 0;
      let offsetY = 0;
      if (Number(bounds.width) >= availableWidth) {{
        offsetX = width / 2 - (left + Number(bounds.width) / 2);
      }} else if (left < margin) {{
        offsetX = margin - left;
      }} else if (right > width - margin) {{
        offsetX = width - margin - right;
      }}
      if (Number(bounds.height) >= availableHeight) {{
        offsetY = height / 2 - (top + Number(bounds.height) / 2);
      }} else if (top < margin) {{
        offsetY = margin - top;
      }} else if (bottom > height - margin) {{
        offsetY = height - margin - bottom;
      }}
      if (Number.isFinite(offsetX)) {{
        base.safeOffsetX = offsetX;
        liveModel.x = Number(base.x) + motionX + offsetX;
      }}
      if (Number.isFinite(offsetY)) {{
        base.safeOffsetY = offsetY;
        liveModel.y = Number(base.y) + motionY + offsetY;
      }}
      bounds = getLayoutBounds();
      if (!finiteBounds(bounds)) return fail();
      const safe = Number(bounds.x) >= margin - 0.5 &&
        Number(bounds.y) >= margin - 0.5 &&
        Number(bounds.x) + Number(bounds.width) <= width - margin + 0.5 &&
        Number(bounds.y) + Number(bounds.height) <= height - margin + 0.5;
      if (safe) break;
    }}
    base.lastBounds = {{
      x: Number(bounds.x) || 0, y: Number(bounds.y) || 0,
      width: Number(bounds.width) || 0, height: Number(bounds.height) || 0
    }};
    const safeScale = Number.isFinite(Number(scale)) && scale > 0 &&
      Math.abs(Number(liveModel.scale.x)) <= referenceScale + 0.0001 &&
      Math.abs(Number(liveModel.scale.y)) <= referenceScale + 0.0001;
    const safeBounds = finiteBounds(bounds) &&
      Number(bounds.x) >= margin - 0.5 && Number(bounds.y) >= margin - 0.5 &&
      Number(bounds.x) + Number(bounds.width) <= width - margin + 0.5 &&
      Number(bounds.y) + Number(bounds.height) <= height - margin + 0.5;
    if (!safeScale || !safeBounds) return fail();
    base.lastGoodTransform = {{
      x: Number(liveModel.x) || 0,
      y: Number(liveModel.y) || 0,
      rotation: Number(liveModel.rotation) || 0,
      scaleX: Number(liveModel.scale.x) || 0,
      scaleY: Number(liveModel.scale.y) || 0,
      scale,
      safeOffsetX: Number(base.safeOffsetX) || 0,
      safeOffsetY: Number(base.safeOffsetY) || 0
    }};
    return true;
  }}
  function resize(requestedWidth, requestedHeight, force = false) {{
    if (!pixiApp) return false;
    const maxViewportEdge = __MAX_VIEWPORT_EDGE__;
    const canvas = document.getElementById('meapet-live2d');
    // Pixi 会在 Application 初始化时给 canvas 写入默认的 800x600
    // 内联尺寸；读取 canvas.clientWidth 会把这个默认值误当成窗口尺寸，
    // 结果是模型被布局到可视区域右侧并被裁切。尺寸基准必须来自
    // HTML 宿主/窗口，再把同一尺寸写回 renderer 和 canvas 样式。
    const host = canvas.parentElement || document.documentElement;
    const documentWidth = document.documentElement.clientWidth || window.innerWidth || 1;
    const documentHeight = document.documentElement.clientHeight || window.innerHeight || 1;
    const explicitWidth = Number(requestedWidth);
    const explicitHeight = Number(requestedHeight);
    const width = Math.min(maxViewportEdge, Math.max(
      1,
      Number.isFinite(explicitWidth) && explicitWidth > 0
        ? explicitWidth : host.clientWidth || documentWidth
    ));
    const height = Math.min(maxViewportEdge, Math.max(
      1,
      Number.isFinite(explicitHeight) && explicitHeight > 0
        ? explicitHeight : host.clientHeight || documentHeight
    ));
    const resizeDeferred = clockState.dragging ||
      performance.now() < Number(clockState.dragReleaseUntil || 0);
    if (!force && resizeDeferred) {{
      if (!resizeState.pending || resizeState.pendingWidth !== width ||
          resizeState.pendingHeight !== height) resizeState.deferredCount += 1;
      resizeState.pending = true;
      resizeState.pendingWidth = width;
      resizeState.pendingHeight = height;
      requestedViewport.width = width;
      requestedViewport.height = height;
      if (!clockState.dragging) scheduleDeferredResizeFlush(32);
      return false;
    }}
    // 相同 CSS 尺寸无需再次重建 Pixi 基准；ResizeObserver 可能在释放后
    // 因 overlay/字体布局再触发一次，但不能把同一尺寸重复应用成视觉跳帧。
    if (!force && resizeState.applied && liveModel &&
        resizeState.appliedModel === liveModel &&
        resizeState.appliedWidth === width && resizeState.appliedHeight === height) {{
      resizeState.pending = false;
      return true;
    }}
    resizeState.pending = false;
    resizeState.pendingWidth = 0;
    resizeState.pendingHeight = 0;
    requestedViewport.width = width;
    requestedViewport.height = height;
    if (faceOverlay) {{
      faceOverlay.width = width;
      faceOverlay.height = height;
      faceOverlay.style.width = `${{width}}px`;
      faceOverlay.style.height = `${{height}}px`;
    }}
    canvas.style.width = `${{width}}px`;
    canvas.style.height = `${{height}}px`;
    pixiApp.renderer.resize(width, height);
    if (liveModel) {{
      // ``model.width``/``model.height`` 已经包含当前 scale。直接用它们
      // 重算会在每次 resize 时累乘缩放，最终造成模型越来越小或比例异常。
      // 首次布局前固定记录未缩放的局部边界，后续始终以同一基准计算一个
      // 标量并同时写入 x/y，保证动作和窗口尺寸变化不会拉伸模型。
      // getLocalBounds() 不应受上一次动作的外层旋转污染；重置后由
      // applyProceduralMotion(0) 按当前动作重新应用旋转，避免窗口调整把
      // 旧动作角度带入新的基准布局。
      liveModel.rotation = 0;
      if (!liveModel.__meapetBaseSize) {{
        liveModel.scale.set(1, 1);
        const bounds = typeof liveModel.getLocalBounds === 'function'
          ? liveModel.getLocalBounds()
          : null;
        liveModel.__meapetBaseSize = {{
          width: Math.max(1, Number(bounds && bounds.width) || Number(liveModel.width) || 1),
          height: Math.max(1, Number(bounds && bounds.height) || Number(liveModel.height) || 1)
        }};
      }}
      const base = liveModel.__meapetBaseSize;
      const scale = Math.max(0.001, Math.min(width / base.width, height / base.height) * 0.9);
      const margin = viewportMargin(width, height);
      // 视口调整不应重新以当前动作的可见网格建立拟合基准。动作中的
      // keyform（尤其 blink/换装）会暂时改变动态 bounds；若在 resize 时
      // 重新采样该 bounds，模型比例会随动作突然缩放。首次布局得到的
      // 可见局部边界与模型本体绑定，后续窗口调整只按新视口重算标量。
      const previousTransform = liveModel.__meapetBaseTransform;
      const retainedVisibleReference = previousTransform &&
        previousTransform.visibleReferenceReady &&
        finiteBounds(previousTransform.visibleReferenceLocalBounds)
        ? {{
            x: Number(previousTransform.visibleReferenceLocalBounds.x),
            y: Number(previousTransform.visibleReferenceLocalBounds.y),
            width: Number(previousTransform.visibleReferenceLocalBounds.width),
            height: Number(previousTransform.visibleReferenceLocalBounds.height)
          }}
        : null;
      const retainedVisibleScaleRatio = retainedVisibleReference
        ? Math.max(
            1,
            Math.min(
              2,
              (Number(previousTransform.referenceScale) || scale) /
                Math.max(0.001, Number(previousTransform.canvasReferenceScale) || scale)
            )
          )
        : 1;
      const referenceScale = scale * retainedVisibleScaleRatio;
      liveModel.scale.set(referenceScale, referenceScale);
      liveModel.x = width / 2;
      liveModel.y = height - margin;
      liveModel.__meapetBaseTransform = {{
        scale: referenceScale, referenceScale: referenceScale, canvasReferenceScale: scale,
        visibleReferenceReady: Boolean(retainedVisibleReference),
        visibleReferenceLocalBounds: retainedVisibleReference,
        x: width / 2, y: height - margin,
        rotation: 0, motionOffsetX: 0, motionOffsetY: 0,
        safeOffsetX: 0, safeOffsetY: 0, lastBounds: null
      }};
      resetGeometryContinuity(3);
      poseState.currentX = 0;
      poseState.currentY = 0;
      if (retainedVisibleReference) {{
        // resize 期间沿用上一帧已经提交的 Core 顶点，只重算外层布局。
        // 这里不能用 seconds=0 重放当前动作：部分 Cubism 模型会在
        // internalModel.update 之前短暂暴露默认 keyform，形成一帧巨大的
        // 动态 bounds，随后 fit 失败并隐藏模型，肉眼表现为缩放抽搐。
        fitModelToViewport();
      }} else {{
        // 首次加载没有稳定顶点基准，需要完成一次参数重放和 Core 更新。
        applyProceduralMotion(0);
      }}
    }}
    resizeState.applied = true;
    resizeState.appliedWidth = width;
    resizeState.appliedHeight = height;
    resizeState.appliedModel = liveModel;
    resizeState.appliedCount += 1;
    return true;
  }}
  function scheduleDeferredResizeFlush(delayMs = 32) {{
    if (!resizeState.pending || resizeState.flushTimer !== null) return false;
    const now = performance.now();
    const releaseAt = Number(clockState.dragReleaseUntil || 0);
    const wait = Math.max(
      Math.max(1, Number(delayMs) || 1),
      releaseAt > now ? releaseAt - now + 1 : 0
    );
    resizeState.flushTimer = setTimeout(function() {{
      resizeState.flushTimer = null;
      if (!resizeState.pending) return;
      if (clockState.dragging || performance.now() < Number(clockState.dragReleaseUntil || 0)) {{
        const releaseAtNow = performance.now();
        const releaseDeadline = Number(clockState.dragReleaseUntil || 0);
        if (clockState.dragging && !clockState.localPointerActive &&
            !clockState.hostDragging && releaseAtNow >= releaseDeadline) {{
          // 宿主已发出 release，但页面本地 timer 可能被后台节流；没有
          // 活跃指针时可以安全收敛状态，避免 pending viewport 永久悬挂。
          clockState.dragging = false;
          clockState.dragReleaseUntil = 0;
          lastFrameTime = releaseAtNow;
          resetGeometryContinuity(1);
          resizeState.flushRetries = 0;
          flushDeferredResize();
          return;
        }}
        // 宿主可能比页面本地 pointer release 早一个消息队列解除 dragging；
        // 有界重试避免 pending 永久悬挂，同时不在真实拖动期间强行恢复。
        resizeState.flushRetries += 1;
        if (resizeState.flushRetries <= 8) scheduleDeferredResizeFlush(32);
        return;
      }}
      resizeState.flushRetries = 0;
      flushDeferredResize();
    }}, Math.min(250, Math.max(1, wait)));
    return true;
  }}
  function flushDeferredResize() {{
    if (!resizeState.pending) return false;
    if (clockState.dragging ||
        performance.now() < Number(clockState.dragReleaseUntil || 0)) {{
      scheduleDeferredResizeFlush(32);
      return false;
    }}
    const width = resizeState.pendingWidth;
    const height = resizeState.pendingHeight;
    resizeState.pending = false;
    resizeState.pendingWidth = 0;
    resizeState.pendingHeight = 0;
    if (!(width > 0) || !(height > 0)) return false;
    // 强制消费 release 前记录的最后尺寸；后续重复的 ResizeObserver 回调
    // 会由 appliedWidth/appliedHeight 去重，不再重置模型姿态。
    return resize(width, height, true);
  }}
  function synchronizeViewport() {{
    if (!pixiApp) return;
    const canvas = document.getElementById('meapet-live2d');
    const host = canvas && canvas.parentElement;
    const width = Math.max(1, Number(host && host.clientWidth) || window.innerWidth || 1);
    const height = Math.max(1, Number(host && host.clientHeight) || window.innerHeight || 1);
    // Pixi 的 renderer.width/height 在高 DPI 下是物理像素；Shape、命中
    // 和窗口布局使用 CSS 逻辑像素。优先比较 screen 尺寸，避免 QT_SCALE_FACTOR=2
    // 时每帧误判 stale 并反复重建基准变换。
    const screen = pixiApp.renderer && pixiApp.renderer.screen;
    const resolution = Math.max(1, Number(pixiApp.renderer && pixiApp.renderer.resolution) ||
      window.devicePixelRatio || 1);
    const rendererWidth = Number(screen && screen.width) ||
      (Number(pixiApp.renderer && pixiApp.renderer.width) || 0) / resolution;
    const rendererHeight = Number(screen && screen.height) ||
      (Number(pixiApp.renderer && pixiApp.renderer.height) || 0) / resolution;
    if (Math.abs(rendererWidth - width) > 0.5 || Math.abs(rendererHeight - height) > 0.5) {{
      // 动作/表情指令可能在 QWidget.resize() 和 Chromium 布局事件之间
      // 到达；在下一帧主动补一次 resize，阻止旧视口参与拟合。
      resize();
    }}
  }}
  function applyProceduralMotion(seconds) {{
    if (!liveModel || !liveModel.__meapetBaseTransform || !motionState.procedural ||
        !proceduralMotions.has(motionState.name)) return false;
    const dt = Math.max(0, Number(seconds) || 0);
    motionState.elapsed += dt;
    if (!motionRequestState.active && motionState.name === "wave" &&
        motionState.elapsed >= 0.9) {{
      motionState.name = "idle";
      motionState.elapsed = 0;
    }} else if (!motionRequestState.active && motionState.name === "blink" &&
               motionState.elapsed >= 0.45) {{
      motionState.name = "idle";
      motionState.elapsed = 0;
    }}
    const base = liveModel.__meapetBaseTransform;
    let phase = 0;
    let offsetX = 0;
    let offsetY = 0;
    let rotation = 0;
    let pulse = 1;
    if (motionState.name === "idle") {{
      // ParamBreath 未绑定时保持等比缩放，只用极小统一垂直位移表现待机；
      // 已绑定模型仍由参数本身提供呼吸细节。
      phase = motionState.elapsed * 2 * Math.PI;
      offsetY = -0.8 * Math.sin(phase / 3.2);
    }} else if (motionState.name === "walk") {{
      phase = motionState.elapsed * 12;
      // 行走不使用非等比缩放；角度和统一平移负责动作感。
      offsetX = 3.5 * Math.sin(phase / 2);
      offsetY = -1.8 * Math.abs(Math.sin(phase));
      rotation = 0.008 * Math.sin(phase);
    }} else if (motionState.name === "wave") {{
      phase = motionState.elapsed * Math.PI * 2 / 0.9;
      offsetX = 1.2 * Math.sin(phase);
      rotation = 0.012 * Math.sin(phase);
    }} else if (motionState.name === "blink") {{
      phase = Math.min(1, motionState.elapsed / 0.45);
      offsetY = -1.0 * Math.sin(Math.PI * phase);
    }}
    const uniformScale = Math.max(
      0.001, Math.min(base.referenceScale || base.scale, base.scale * pulse)
    );
    // x/y 使用同一绝对缩放值；几何镜像只使用经过中性区确认的符号，
    // 不把连续的 facing 值直接当作 scale.x，避免转身期间产生非等比缩放。
    liveModel.scale.set(geometryFacingSign() * uniformScale, uniformScale);
    base.motionOffsetX = offsetX;
    base.motionOffsetY = offsetY;
    liveModel.x = base.x + base.safeOffsetX + offsetX;
    liveModel.y = base.y + base.safeOffsetY + offsetY;
    liveModel.rotation = base.rotation + (facing < 0 ? -rotation : rotation);
    return applyProceduralParameterMotion(dt);
  }}
  async function loadModelResource(url) {{
    if (!window.PIXI || !window.PIXI.live2d || !url) {{
      throw new Error('Pixi Live2D runtime or model URL is unavailable');
    }}
    const modelPromise = PIXI.live2d.Live2DModel.from(url, {{
      autoInteract: false,
      autoUpdate: false
    }});
    // Promise.race 超时只会结束等待方，底层模型加载仍可能在网络/解码
    // 完成后回调。观察该迟到结果并主动销毁，避免模型切换或首屏超时
    // 累积 WebGL/Cubism 资源；超时后的迟到拒绝也必须被消费。
    let timedOut = false;
    let timeoutId = 0;
    const observedModelPromise = modelPromise.then(
      function(model) {{
        if (timedOut) destroyCandidateModel(model);
        return model;
      }},
      function(error) {{
        if (timedOut) return null;
        throw error;
      }}
    );
    const timeoutPromise = new Promise(function(_resolve, reject) {{
      timeoutId = setTimeout(function() {{
        timedOut = true;
        reject(new Error('Live2D model load timed out'));
      }}, modelLoadTimeoutMs);
    }});
    try {{
      return await Promise.race([observedModelPromise, timeoutPromise]);
    }} finally {{
      clearTimeout(timeoutId);
    }}
  }}
  function destroyCandidateModel(model) {{
    if (!model) return;
    try {{
      if (pixiApp && pixiApp.stage && typeof pixiApp.stage.removeChild === 'function') {{
        pixiApp.stage.removeChild(model);
      }}
    }} catch (error) {{}}
    try {{
      if (typeof model.destroy === 'function') model.destroy();
    }} catch (error) {{}}
  }}
  function applyModelMetadata(metadata) {{
    if (!metadata || typeof metadata !== 'object') return false;
    const url = String(metadata.url || '').trim();
    if (!url) return false;
    modelUrl = url;
    modelPartNames = Array.isArray(metadata.part_names) ? metadata.part_names.slice() : [];
    modelPartLabels = Array.isArray(metadata.part_labels) ? metadata.part_labels.slice() : [];
    declaredExpressions = new Set(
      Array.isArray(metadata.expressions) ? metadata.expressions.map(String) : []
    );
    declaredMotions = new Set(
      Array.isArray(metadata.motions) ? metadata.motions.map(String) : []
    );
    motionAliases = new Map(
      Object.entries(metadata.motion_aliases && typeof metadata.motion_aliases === 'object'
        ? metadata.motion_aliases : {{}})
    );
    motionDurations = new Map(
      Object.entries(metadata.motion_durations && typeof metadata.motion_durations === 'object'
        ? metadata.motion_durations : {{}})
    );
    motionSyncProfile = metadata.motion_sync && typeof metadata.motion_sync === 'object'
      ? metadata.motion_sync : null;
    if (window.meapetLive2D) {{
      window.meapetLive2D.modelUrl = modelUrl;
      window.meapetLive2D.partNames = modelPartNames;
    }}
    return true;
  }}
  function validateCandidateModel(model) {{
    if (!model || !model.internalModel || typeof model.getLocalBounds !== 'function') return false;
    let bounds = null;
    try {{ bounds = model.getLocalBounds(); }} catch (error) {{ return false; }}
    if (!finiteBounds(bounds)) return false;
    const scale = model.scale;
    return Boolean(scale) && Number.isFinite(Number(scale.x)) &&
      Number.isFinite(Number(scale.y));
  }}
  async function loadModel() {{
    if (!window.PIXI || !window.PIXI.live2d || !modelUrl) {{
      throw new Error('Pixi Live2D runtime or model URL is unavailable');
    }}
    const canvas = document.getElementById('meapet-live2d');
    installPointerBridge(canvas);
    const viewportDocument = document.documentElement;
    const viewportWidth = Math.max(1, viewportDocument.clientWidth || window.innerWidth || 1);
    const viewportHeight = Math.max(1, viewportDocument.clientHeight || window.innerHeight || 1);
    pixiApp = new PIXI.Application({{
      view: canvas, width: viewportWidth, height: viewportHeight,
      backgroundAlpha: 0, backgroundColor: 0x000000, antialias: true,
      resolution: Math.max(1, window.devicePixelRatio || 1), autoDensity: true,
      // Qt 的原生输入 Shape 需要读取“只有模型 canvas”的 alpha；保留
      // 当前绘制缓冲后，inputMask() 不会把气泡/Toast/点击反馈算进输入区。
      preserveDrawingBuffer: true
    }});
    if (typeof PIXI.live2d.cubism4Ready === 'function') {{
      await PIXI.live2d.cubism4Ready();
    }}
    // 渲染宿主以统一时钟调用 ``advance``；关闭 pixi 共享 ticker，
    // 否则 Cubism 会被自动 ticker 和宿主各更新一次，动作速度会异常。
    liveModel = await loadModelResource(modelUrl);
    liveModel.anchor.set(0.5, 1.0);
    // 在首次布局/动画前完成真实 keyform 探测，避免初始化帧把无效
    // 参数写入模型；新导出的可见参数会在此自动进入白名单。
    detectParameterBindings();
    if (!repairRenderOrders()) {{
      throw new Error('Live2D render order validation failed');
    }}
    pixiApp.stage.addChild(liveModel);
    resize();
  }}
  function copyRuntimeObject(value) {{
    return value && typeof value === 'object' ? Object.assign(Object.create(null), value) : {{}};
  }}
  function snapshotModelRuntime() {{
    return {{
      liveModel, modelUrl, modelPartNames, modelPartLabels,
      liveModelVisible: liveModel ? liveModel.visible !== false : false,
      declaredExpressions, declaredMotions, motionAliases, motionDurations,
      motionSyncProfile,
      parameterBindings: copyRuntimeObject(parameterBindingState),
      parameterSafety: copyRuntimeObject(parameterSafetyState),
      facialBindingCount, expressionBindingCount, expressionPoseScale,
      eyeOverlayEnabled, mouthOverlayEnabled, expressionOverlayEnabled, faceOverlayEnabled,
      resize: copyRuntimeObject(resizeState),
      geometry: copyRuntimeObject(geometryGuard),
      geometryAudit: copyRuntimeObject(geometryAuditState),
      renderPerformance: copyRuntimeObject(renderPerformanceState),
      expression: copyRuntimeObject(expressionState),
      motion: copyRuntimeObject(motionState),
      pose: copyRuntimeObject(poseState)
    }};
  }}
  function restoreRuntimeObject(snapshot) {{
    if (!snapshot) return;
    liveModel = snapshot.liveModel;
    modelUrl = snapshot.modelUrl;
    modelPartNames = snapshot.modelPartNames;
    modelPartLabels = snapshot.modelPartLabels;
    declaredExpressions = snapshot.declaredExpressions;
    declaredMotions = snapshot.declaredMotions;
    motionAliases = snapshot.motionAliases;
    motionDurations = snapshot.motionDurations;
    motionSyncProfile = snapshot.motionSyncProfile;
    Object.keys(parameterBindingState).forEach(key => delete parameterBindingState[key]);
    Object.assign(parameterBindingState, snapshot.parameterBindings || {{}});
    Object.keys(parameterSafetyState).forEach(key => delete parameterSafetyState[key]);
    Object.assign(parameterSafetyState, snapshot.parameterSafety || {{}});
    facialBindingCount = Number(snapshot.facialBindingCount) || 0;
    expressionBindingCount = Number(snapshot.expressionBindingCount) || 0;
    expressionPoseScale = Number(snapshot.expressionPoseScale) || 1;
    eyeOverlayEnabled = Boolean(snapshot.eyeOverlayEnabled);
    mouthOverlayEnabled = Boolean(snapshot.mouthOverlayEnabled);
    expressionOverlayEnabled = Boolean(snapshot.expressionOverlayEnabled);
    faceOverlayEnabled = Boolean(snapshot.faceOverlayEnabled);
    Object.assign(resizeState, snapshot.resize || {{}});
    Object.assign(geometryGuard, snapshot.geometry || {{}});
    Object.assign(geometryAuditState, snapshot.geometryAudit || {{}});
    Object.assign(renderPerformanceState, snapshot.renderPerformance || {{}});
    Object.assign(expressionState, snapshot.expression || {{}});
    Object.assign(motionState, snapshot.motion || {{}});
    Object.assign(poseState, snapshot.pose || {{}});
    // 预检失败时 Python 仍保留旧模型；公开对象也必须恢复旧元数据，
    // 否则后续命中/动作请求会把候选 URL 当成当前模型而产生状态分裂。
    if (window.meapetLive2D) {{
      window.meapetLive2D.modelUrl = modelUrl;
      window.meapetLive2D.partNames = modelPartNames;
    }}
  }}
  function captureCandidateBindings() {{
    return {{
      parameterBindings: copyRuntimeObject(parameterBindingState),
      parameterSafety: copyRuntimeObject(parameterSafetyState),
      facialBindingCount, expressionBindingCount, expressionPoseScale,
      eyeOverlayEnabled, mouthOverlayEnabled, expressionOverlayEnabled, faceOverlayEnabled
    }};
  }}
  function applyCandidateBindings(value) {{
    Object.keys(parameterBindingState).forEach(key => delete parameterBindingState[key]);
    Object.assign(parameterBindingState, value.parameterBindings || {{}});
    Object.keys(parameterSafetyState).forEach(key => delete parameterSafetyState[key]);
    Object.assign(parameterSafetyState, value.parameterSafety || {{}});
    facialBindingCount = Number(value.facialBindingCount) || 0;
    expressionBindingCount = Number(value.expressionBindingCount) || 0;
    expressionPoseScale = Number(value.expressionPoseScale) || 1;
    eyeOverlayEnabled = Boolean(value.eyeOverlayEnabled);
    mouthOverlayEnabled = Boolean(value.mouthOverlayEnabled);
    expressionOverlayEnabled = Boolean(value.expressionOverlayEnabled);
    faceOverlayEnabled = Boolean(value.faceOverlayEnabled);
  }}
  async function reloadModel(url, metadata, requestId) {{
    const id = Number(requestId);
    if (!Number.isInteger(id) || id <= 0 || !metadata || typeof metadata !== 'object' ||
        String(metadata.url || '') !== String(url || '')) {{
      reportModelReload(id, 'failed', url, 'invalid model reload request');
      return false;
    }}
    if (!pixiApp || !liveModel || clockState.dragging || clockState.localPointerActive ||
        clockState.hostDragging) {{
      reportModelReload(id, 'failed', url, 'model reload is blocked while the pet is busy');
      return false;
    }}
    if (pendingModelReload !== null) {{
      reportModelReload(id, 'failed', url, 'another model reload is in progress');
      return false;
    }}
    modelReloadGeneration = Math.max(modelReloadGeneration, id);
    const request = {{
      id, url: String(url), metadata, phase: 'loading', candidate: null, snapshot: null
    }};
    pendingModelReload = request;
    let candidate = null;
    const snapshot = snapshotModelRuntime();
    try {{
      candidate = await loadModelResource(request.url);
      if (pendingModelReload !== request || modelReloadGeneration !== id) {{
        destroyCandidateModel(candidate);
        return false;
      }}
      // 候选模型加载期间仍可能收到新的指针/宿主拖动事件；交换必须在
      // 同一空闲边界完成，否则旧模型的输入 Shape 与新模型首帧会交错。
      // 保留旧模型并报告失败，调用方可在手势结束后重新发起请求。
      if (clockState.dragging || clockState.localPointerActive || clockState.hostDragging) {{
        throw new Error('model reload was superseded by pet interaction');
      }}
      if (!validateCandidateModel(candidate)) {{
        throw new Error('candidate Live2D model geometry is invalid');
      }}
      candidate.anchor.set(0.5, 1.0);
      candidate.visible = false;
      pixiApp.stage.addChild(candidate);

      // 预检阶段临时切换全局引用，只执行不会触碰窗口/Shape 的 Core
      // 参数和绘制顺序检查；完成后立即恢复旧模型及全部运行时状态。
      if (!applyModelMetadata(request.metadata)) {{
        throw new Error('candidate model metadata is invalid');
      }}
      liveModel = candidate;
      detectParameterBindings();
      if (!repairRenderOrders(true) || !validateDrawableFrame()) {{
        throw new Error('candidate Live2D geometry validation failed');
      }}
      const candidateBindings = captureCandidateBindings();
      restoreRuntimeObject(snapshot);

      // 单个 JavaScript 任务内完成交换，旧模型在成功回执前保持可见；
      // 新模型先完成等比布局和首帧安全审计，任何失败都回滚到 snapshot。
      applyModelMetadata(request.metadata);
      liveModel = candidate;
      applyCandidateBindings(candidateBindings);
      resetExpressionTimeline(false);
      resetMotionRequest(false);
      expressionState.name = 'neutral';
      expressionState.procedural = true;
      expressionState.targetAngle = {{ x: 0, y: 0 }};
      expressionState.currentAngle = {{ x: 0, y: 0 }};
      motionState.name = 'idle';
      motionState.elapsed = 0;
      motionState.procedural = true;
      resetGeometryContinuity(4);
      if (!resize(0, 0, true) || !validateDrawableFrame() || !repairRenderOrders(true) ||
          !fitModelToViewport() || !enforceUniformTransform()) {{
        throw new Error('candidate Live2D first frame validation failed');
      }}
      candidate.visible = true;
      if (snapshot.liveModel) snapshot.liveModel.visible = false;
      // 页面只进入 prepared，不释放旧模型。Python 收到同代次回执后先
      // accept，页面再报告 available；Python 最终 finalize 才销毁旧 Core。
      // 任一回执丢失时，Python 有界超时会调用 rollback 恢复 snapshot。
      request.phase = 'prepared';
      request.candidate = candidate;
      request.snapshot = snapshot;
      reportModelReload(id, 'prepared', request.url, '');
      return true;
    }} catch (error) {{
      if (candidate) destroyCandidateModel(candidate);
      restoreRuntimeObject(snapshot);
      if (snapshot.liveModel) snapshot.liveModel.visible = Boolean(snapshot.liveModelVisible);
      pendingModelReload = null;
      reportModelReload(
        id, 'failed', request.url,
        String(error && error.message ? error.message : 'candidate Live2D model rejected')
      );
      return false;
    }}
  }}
  function acceptModelReload(requestId) {{
    const id = Number(requestId);
    const request = pendingModelReload;
    if (!request || request.id !== id || request.phase !== 'prepared') return false;
    request.phase = 'accepted';
    reportModelReload(id, 'available', request.url, '');
    return true;
  }}
  function finalizeModelReload(requestId) {{
    const id = Number(requestId);
    const request = pendingModelReload;
    if (!request || request.id !== id || request.phase !== 'accepted') return false;
    pendingModelReload = null;
    const oldModel = request.snapshot && request.snapshot.liveModel;
    if (oldModel) window.requestAnimationFrame(function() {{ destroyCandidateModel(oldModel); }});
    request.snapshot = null;
    request.candidate = null;
    return true;
  }}
  function rollbackModelReload(requestId) {{
    const id = Number(requestId);
    const request = pendingModelReload;
    if (!request || request.id !== id) return false;
    pendingModelReload = null;
    if (request.snapshot) {{
      restoreRuntimeObject(request.snapshot);
      if (request.snapshot.liveModel) {{
        request.snapshot.liveModel.visible = Boolean(request.snapshot.liveModelVisible);
      }}
    }}
    if (request.candidate) destroyCandidateModel(request.candidate);
    request.snapshot = null;
    request.candidate = null;
    return true;
  }}
  function installPointerBridge(canvas) {{
    if (!canvas || canvas.__meapetPointerBridgeInstalled) return;
    canvas.__meapetPointerBridgeInstalled = true;
    let activePointerId = null;
    let pressX = 0;
    let pressY = 0;
    let pressScreenX = 0;
    let pressScreenY = 0;
    let pointerMoved = false;
    // 页面事件跨越 WebChannel 到达 Python/Qt 前通常会延迟数帧。
    // 若只等待宿主回调设置 dragging，窗口已开始移动时 rAF 仍会推进
    // Cubism，表现为拖动起始处的姿态抽搐。页面本地先冻结时钟，宿主
    // 回调随后幂等确认；释放/取消也先在页面恢复，避免尾帧继续漂移。
    let localDragging = false;
    let localDragReleaseTimer = null;
    const localDragReleaseDelayMs = 90;
    function setLocalDragging(value) {{
      const next = Boolean(value);
      let reassertHostDragging = false;
      if (next && localDragReleaseTimer !== null) {{
        clearTimeout(localDragReleaseTimer);
        localDragReleaseTimer = null;
        clockState.dragReleaseUntil = 0;
        // 新一笔按下可能紧接在上一笔 release 的稳定窗内到达；即使
        // localDragging 仍为 true，也要把 host 标记重新写回 true，避免
        // 迟到的宿主 false 让 debug/兜底路径误判当前手势已结束。
        reassertHostDragging = true;
      }}
      if (localDragging === next) {{
        if (reassertHostDragging) {{
          const api = window.meapetLive2D;
          if (api && typeof api.setDragging === 'function') api.setDragging(true);
        }}
        return;
      }}
      localDragging = next;
      if (next) clockState.dragReleaseUntil = 0;
      const api = window.meapetLive2D;
      if (api && typeof api.setDragging === 'function') api.setDragging(next);
    }}
    function releaseLocalDragging() {{
      if (!localDragging) return;
      if (localDragReleaseTimer !== null) clearTimeout(localDragReleaseTimer);
      clockState.dragReleaseUntil = performance.now() + localDragReleaseDelayMs;
      localDragReleaseTimer = setTimeout(function() {{
        localDragReleaseTimer = null;
        clockState.dragReleaseUntil = 0;
        setLocalDragging(false);
      }}, localDragReleaseDelayMs);
    }}
    let clickTimer = null;
    let clickGeneration = 0;
    let lastClickAt = 0;
    let lastClickX = 0;
    let lastClickY = 0;
    let lastDoubleAt = 0;
    let lastPointerClientX = NaN;
    let lastPointerClientY = NaN;
    let lastPointerScreenX = NaN;
    let lastPointerScreenY = NaN;
    let hoverHit = false;
    // 命中测试会遍历当前模型的三角形并可能读回 WebGL alpha；光标移动
    // 事件远高于渲染帧率时，不能让每个事件都同步执行完整探测。
    let hoverProbeAt = -Infinity;
    const hoverProbeIntervalMs = 50;
    let dragMoveAt = -Infinity;
    const dragMoveIntervalMs = 16;
    const doubleClickWindowMs = 420;
    function makePayload(type, event) {{
      const rect = canvas.getBoundingClientRect();
      const button = event.button === 0 ? 'left' : event.button === 2 ? 'right' : 'other';
      // clientX/clientY 是窗口局部 CSS 坐标。窗口跟随指针移动时，
      // 该坐标会随窗口原点变化，宿主若用它计算位移会出现回弹/抽搐。
      // 同时传递屏幕坐标，宿主优先使用稳定的全局坐标；旧 WebView
      // 没有 screenX/screenY 时仍可回退到局部坐标。
      const clientX = Number(event.clientX);
      const clientY = Number(event.clientY);
      const rawScreenX = Number(event.screenX);
      const rawScreenY = Number(event.screenY);
      // PointerEvent 构造器在自动化/兼容 WebView 中会把缺失的屏幕坐标
      // 默认填成 (0, 0)。窗口不在屏幕原点时，这个哨兵值会把拖动原点
      // 错当成全局左上角；只有当局部坐标也在原点时才接受该值。
      const missingScreenOrigin = rawScreenX === 0 && rawScreenY === 0 &&
        (clientX !== 0 || clientY !== 0);
      const screenX = missingScreenOrigin ? NaN : rawScreenX;
      const screenY = missingScreenOrigin ? NaN : rawScreenY;
      return {{
        type: type,
        x: clientX - Number(rect.left),
        y: clientY - Number(rect.top),
        screen_x: Number.isFinite(screenX) ? screenX : null,
        screen_y: Number.isFinite(screenY) ? screenY : null,
        button: button,
        pointer_id: Number(event.pointerId) || 0,
        buttons: Number(event.buttons) || 0,
        detail: Number(event.detail) || 0
      }};
    }}
    function sendPayload(payload) {{
      const bridge = window.meapetBridge;
      if (!bridge || typeof bridge.pointerEvent !== 'function') return false;
      try {{
        bridge.pointerEvent(JSON.stringify(payload));
        return true;
      }} catch (error) {{
        return false;
      }}
    }}
    function send(type, event) {{
      return sendPayload(makePayload(type, event));
    }}
    function cancelPendingClick() {{
      clickGeneration += 1;
      if (clickTimer !== null) {{
        clearTimeout(clickTimer);
        clickTimer = null;
      }}
      const marker = feedbackState.provisionalMarker;
      const layer = document.getElementById('meapet-hit-feedback');
      if (marker && layer && marker.parentNode === layer) layer.removeChild(marker);
      feedbackState.provisionalMarker = null;
      clearPendingHitFeedback();
    }}
    function scheduleClick(event) {{
      const hitDetails = modelHitDetails(event, canvas);
      if (!hitDetails.hit) return;
      const now = performance.now();
      const pointX = Number(event.clientX);
      const pointY = Number(event.clientY);
      const closeToPrevious = Math.hypot(pointX - lastClickX, pointY - lastClickY) <= 18;
      if (lastClickAt > 0 && now - lastClickAt <= doubleClickWindowMs && closeToPrevious) {{
        cancelPendingClick();
        lastClickAt = 0;
        lastDoubleAt = now;
        sendPayload(makePayload('double', event));
        return;
      }}
      lastClickAt = now;
      lastClickX = pointX;
      lastClickY = pointY;
      const payload = makePayload('click', event);
      payload.hit = true;
      payload.feedback_part = feedbackState.part;
      payload.feedback_count = feedbackState.count;
      if (typeof hitDetails.part === 'string' && hitDetails.part) {{
        payload.part = hitDetails.part;
      }}
      if (Array.isArray(hitDetails.drawableIds)) {{
        payload.drawable_ids = hitDetails.drawableIds.slice(0, 8);
      }}
      if (Array.isArray(hitDetails.partNames)) {{
        payload.part_names = hitDetails.partNames.slice(0, 8);
      }}
      // 先给用户即时的视觉回馈；好感度、控制台状态和正式 click 仍在
      // 双击窗口结束后提交，双击时会撤销这个临时 marker。
      showHitFeedback(hitDetails.part, hitDetails.canvasX, hitDetails.canvasY, true);
      showPendingHitFeedback(hitDetails.part);
      const generation = ++clickGeneration;
      if (clickTimer !== null) clearTimeout(clickTimer);
      clickTimer = setTimeout(function() {{
        clickTimer = null;
        if (generation !== clickGeneration) return;
        // 反馈和业务 click 必须在同一去抖分支提交。此前 marker 在检测
        // 双击前就显示，用户双击打开控制台时会短暂看到一次并未真正
        // 触发的“摸摸头”反馈，造成好感度/视觉状态不一致。
        showHitFeedback(hitDetails.part, hitDetails.canvasX, hitDetails.canvasY);
        sendPayload(payload);
      }}, 450);
    }}
    canvas.addEventListener('pointerenter', function(event) {{
      lastPointerClientX = Number(event.clientX);
      lastPointerClientY = Number(event.clientY);
      lastPointerScreenX = Number(event.screenX);
      lastPointerScreenY = Number(event.screenY);
      hoverProbeAt = performance.now();
      // hover 只需要布尔命中；跳过全量 drawable 收集与 framebuffer
      // readback，避免光标移动阻塞渲染。点击路径仍使用默认精确模式。
      const hitDetails = modelHitDetails(event, canvas, {{
        collectAll: false, sampleRenderedAlpha: false
      }});
      const payload = makePayload('enter', event);
      payload.hit = Boolean(hitDetails.hit);
      hoverHit = payload.hit;
      sendPayload(payload);
    }});
    canvas.addEventListener('pointerleave', function(event) {{
      lastPointerClientX = NaN;
      lastPointerClientY = NaN;
      lastPointerScreenX = NaN;
      lastPointerScreenY = NaN;
      hoverProbeAt = -Infinity;
      const payload = makePayload('leave', event);
      payload.hit = false;
      hoverHit = false;
      sendPayload(payload);
    }});
    window.addEventListener('keydown', function(event) {{
      // 光标停在模型上时，Enter 直接打开输入框；输入框/编辑器自身的
      // Enter 不被桌宠拦截，透明区域也不会触发该快捷入口。
      if ((event.key === 'Enter' || event.key === 'Return') && hoverHit &&
          !event.repeat && !event.isComposing && !event.ctrlKey && !event.altKey &&
          !event.shiftKey && !event.metaKey && !cursorState.locked &&
          !(document.activeElement && document.activeElement.isContentEditable) &&
          !(document.activeElement && document.activeElement.tagName === 'INPUT') &&
          !(document.activeElement && document.activeElement.tagName === 'TEXTAREA')) {{
        const payload = makePayload('enter_key', event);
        payload.hit = true;
        sendPayload(payload);
        event.preventDefault();
      }}
    }});
    canvas.addEventListener('pointerdown', function(event) {{
      lastPointerClientX = Number(event.clientX);
      lastPointerClientY = Number(event.clientY);
      lastPointerScreenX = Number(event.screenX);
      lastPointerScreenY = Number(event.screenY);
      if (cursorState.locked) return;
      if (event.button === 0) {{
        // 透明画布覆盖整个窗口，但透明区域不能建立拖动/指针捕获；
        // 否则一次点空会先抢走焦点，再让宿主把桌宠窗口拖走。
        const hitDetails = modelHitDetails(event, canvas);
        if (!hitDetails.hit) {{
          const transparentPayload = makePayload('transparent', event);
          transparentPayload.hit = false;
          sendPayload(transparentPayload);
          // 不阻止浏览器默认事件；若平台没有 Shape 输入区域，底层应用
          // 仍有机会接收这次透明点点击。
          return;
        }}
        activePointerId = event.pointerId;
        clockState.localPointerActive = true;
        pressX = Number(event.clientX);
        pressY = Number(event.clientY);
        const initialScreenX = Number(event.screenX);
        const initialScreenY = Number(event.screenY);
        const missingScreenOrigin = initialScreenX === 0 && initialScreenY === 0 &&
          (pressX !== 0 || pressY !== 0);
        pressScreenX = !missingScreenOrigin && Number.isFinite(initialScreenX)
          ? initialScreenX : pressX;
        pressScreenY = !missingScreenOrigin && Number.isFinite(initialScreenY)
          ? initialScreenY : pressY;
        pointerMoved = false;
        dragMoveAt = -Infinity;
        // 先冻结页面自己的 rAF，再跨 WebChannel 通知宿主；普通点击只会
        // 短暂冻结一帧，不会改变点击去抖或双击语义。
        setLocalDragging(true);
        try {{ canvas.setPointerCapture(event.pointerId); }} catch (error) {{}}
        const pressPayload = makePayload('press', event);
        pressPayload.hit = true;
        sendPayload(pressPayload);
        event.preventDefault();
      }} else if (event.button === 2) {{
        const hitDetails = modelHitDetails(event, canvas);
        if (!hitDetails.hit) {{
          const transparentPayload = makePayload('transparent', event);
          transparentPayload.hit = false;
          sendPayload(transparentPayload);
          return;
        }}
        const contextPayload = makePayload('context', event);
        contextPayload.hit = true;
        sendPayload(contextPayload);
        event.preventDefault();
      }}
    }});
    canvas.addEventListener('pointermove', function(event) {{
      lastPointerClientX = Number(event.clientX);
      lastPointerClientY = Number(event.clientY);
      lastPointerScreenX = Number(event.screenX);
      lastPointerScreenY = Number(event.screenY);
      if (activePointerId === null) {{
        // 浏览器可能以数百 Hz 派发 pointermove；当前帧的 hover 只需有限
        // 频率更新，按下/释放路径仍始终执行精确命中，不能被此节流影响。
        const now = performance.now();
        if (now - hoverProbeAt < hoverProbeIntervalMs) return;
        hoverProbeAt = now;
        const hitDetails = modelHitDetails(event, canvas, {{
          collectAll: false, sampleRenderedAlpha: false
        }});
        const nextHoverHit = Boolean(hitDetails.hit);
        // 即使命中状态没有改变，也要按 50ms 节流发送坐标。宿主可据此
        // 立即刷新光标注视目标，锁定/点击穿透时仍能视觉跟随；该事件不
        // 触发模型调用或好感度变化，因此不会造成行为请求风暴。
        if (nextHoverHit !== hoverHit) hoverHit = nextHoverHit;
        const hoverPayload = makePayload('hover', event);
        hoverPayload.hit = nextHoverHit;
        sendPayload(hoverPayload);
        return;
      }}
      if (activePointerId !== event.pointerId) return;
      const rawCurrentScreenX = Number(event.screenX);
      const rawCurrentScreenY = Number(event.screenY);
      const currentClientX = Number(event.clientX);
      const currentClientY = Number(event.clientY);
      const missingScreenOrigin = rawCurrentScreenX === 0 && rawCurrentScreenY === 0 &&
        (currentClientX !== 0 || currentClientY !== 0);
      const currentScreenX = !missingScreenOrigin && Number.isFinite(rawCurrentScreenX)
        ? rawCurrentScreenX : currentClientX;
      const currentScreenY = !missingScreenOrigin && Number.isFinite(rawCurrentScreenY)
        ? rawCurrentScreenY : currentClientY;
      if (Math.hypot(currentScreenX - pressScreenX, currentScreenY - pressScreenY) >= 8) {{
        pointerMoved = true;
        // 在发送跨进程 move 回调之前冻结页面自身的 rAF；否则 Qt 侧
        // ``setDragging(true)`` 到达前仍会执行动作/物理更新。
        setLocalDragging(true);
      }}
      // QWebChannel 的 move 回调跨越 Chromium/Qt 边界；触控板或高刷新率
      // 鼠标可能每秒产生数百个事件。限制传输到约 60Hz，避免消息队列积压
      // 造成窗口追赶旧坐标而抖动；释放时仍会补发最后一个位置。
      const now = performance.now();
      if (now - dragMoveAt >= dragMoveIntervalMs) {{
        dragMoveAt = now;
        send('move', event);
      }}
      event.preventDefault();
    }});
    function finish(type, event) {{
      if (activePointerId !== event.pointerId) return;
      const shouldClick = type === 'release' && event.button === 0 && !pointerMoved;
      if (type === 'release' && pointerMoved) send('move', event);
      send(type, event);
      clockState.localPointerActive = false;
      // 窗口管理器收到最后一个坐标后仍可能需要一轮合成；延迟恢复
      // 页面时钟，避免拖动尾帧在新位置尚未稳定时出现跳变。
      releaseLocalDragging();
      try {{ canvas.releasePointerCapture(event.pointerId); }} catch (error) {{}}
      activePointerId = null;
      if (shouldClick) scheduleClick(event);
      event.preventDefault();
    }}
    function cancelActivePointer() {{
      // 窗口切换/浏览器失焦时不一定会派发 pointerup。发送不带坐标的
      // cancel，让宿主丢弃拖动事务而不把窗口回写到旧按下位置。
      if (activePointerId === null) {{
        if (localDragging) setLocalDragging(false);
        return;
      }}
      const bridge = window.meapetBridge;
      if (bridge && typeof bridge.pointerEvent === 'function') {{
        try {{
          bridge.pointerEvent(JSON.stringify({{
            type: 'cancel', pointer_id: Number(activePointerId) || 0,
            button: 'left', buttons: 0
          }}));
        }} catch (error) {{}}
      }}
      clockState.localPointerActive = false;
      releaseLocalDragging();
      try {{ canvas.releasePointerCapture(activePointerId); }} catch (error) {{}}
      activePointerId = null;
      pointerMoved = false;
    }}
    function refreshHoverFromLastPointer() {{
      // 模型异步加载完成时，指针可能一直停在画布上，没有新的
      // pointermove；主动重算一次，保证悬停反馈和光标追踪不丢失。
      if (!Number.isFinite(lastPointerClientX) || !Number.isFinite(lastPointerClientY)) return;
      const synthetic = {{
        clientX: lastPointerClientX,
        clientY: lastPointerClientY,
        screenX: lastPointerScreenX,
        screenY: lastPointerScreenY,
        pointerId: 0,
        button: -1,
        buttons: 0,
        detail: 0
      }};
      const details = modelHitDetails(synthetic, canvas, {{
        collectAll: false, sampleRenderedAlpha: false
      }});
      const payload = makePayload('hover', synthetic);
      payload.hit = Boolean(details.hit);
      hoverHit = payload.hit;
      sendPayload(payload);
    }}
    refreshHoverHook = refreshHoverFromLastPointer;
    canvas.addEventListener('pointerup', function(event) {{ finish('release', event); }});
    canvas.addEventListener('pointercancel', function(event) {{ finish('cancel', event); }});
    window.addEventListener('blur', cancelActivePointer);
    window.addEventListener('pagehide', cancelActivePointer);
    document.addEventListener('visibilitychange', function() {{
      if (document.hidden) cancelActivePointer();
    }});
    canvas.addEventListener('dblclick', function(event) {{
      if (cursorState.locked) {{
        event.preventDefault();
        return;
      }}
      // canvas 是覆盖整个透明窗口的矩形；浏览器的 dblclick 事件本身
      // 不区分模型和透明区域。只有几何命中才允许打开控制台，避免在
      // 桌面空白处双击时误触发桌宠控制入口。
      const hitDetails = modelHitDetails(event, canvas);
      if (!hitDetails.hit) {{
        event.preventDefault();
        return;
      }}
      const now = performance.now();
      // Chromium 可能把原生 dblclick 事件排到第二次 pointerup 之后
      // 超过 200ms；沿用同一去抖窗口可避免页面手势和 pointer 手势各自
      // 触发一次控制台，且不会影响两个独立双击之间的正常间隔。
      if (now - lastDoubleAt < doubleClickWindowMs) {{
        event.preventDefault();
        return;
      }}
      lastDoubleAt = now;
      lastClickAt = 0;
      cancelPendingClick();
      send('double', event);
      event.preventDefault();
    }});
    // 由宿主负责模型区域的右键菜单；浏览器默认菜单不能在透明边缘
    // 意外弹出并再次抢走当前应用焦点。
    canvas.addEventListener('contextmenu', function(event) {{ event.preventDefault(); }});
  }}
  function pointInTriangle(px, py, ax, ay, bx, by, cx, cy) {{
    const ab = (px - ax) * (by - ay) - (py - ay) * (bx - ax);
    const bc = (px - bx) * (cy - by) - (py - by) * (cx - bx);
    const ca = (px - cx) * (ay - cy) - (py - cy) * (ax - cx);
    const hasNegative = ab < -0.0001 || bc < -0.0001 || ca < -0.0001;
    const hasPositive = ab > 0.0001 || bc > 0.0001 || ca > 0.0001;
    return !(hasNegative && hasPositive);
  }}
  // Cubism drawable 网格通常是矩形；只按三角形命中会把纹理透明区域
  // 当成可点击内容，用户点到头发/身体旁的空白也会触发反馈。缓存每张
  // 本地纹理的 alpha 平面，并在命中的三角形内插值 UV，命中条件与最终
  // 渲染像素保持一致。跨域或无法读取纹理时保守回退到几何命中，不能
  // 因可选的 alpha 采样让整个模型失去交互。
  const textureAlphaCache = new Map();
  function textureAlphaSource(textureIndex) {{
    const index = Number(textureIndex);
    if (textureAlphaCache.has(index)) return textureAlphaCache.get(index);
    try {{
      const texture = liveModel && liveModel.textures && liveModel.textures[index];
      const source = texture && texture.baseTexture && texture.baseTexture.resource &&
        texture.baseTexture.resource.source;
      const width = Number(source && (source.naturalWidth || source.width));
      const height = Number(source && (source.naturalHeight || source.height));
      if (!source || !(width > 0) || !(height > 0)) {{
        textureAlphaCache.set(index, null);
        return null;
      }}
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext('2d', {{ willReadFrequently: true }});
      if (!context) {{
        textureAlphaCache.set(index, null);
        return null;
      }}
      context.drawImage(source, 0, 0, width, height);
      const pixels = context.getImageData(0, 0, width, height).data;
      const value = {{ width, height, pixels }};
      textureAlphaCache.set(index, value);
      return value;
    }} catch (error) {{
      // file:// 资源通常同源；若目标 WebView 禁止读取图像像素，保留几何
      // 回退即可，不能把安全策略错误升级为模型加载失败。
      textureAlphaCache.set(index, null);
      return null;
    }}
  }}
  function triangleBarycentric(px, py, ax, ay, bx, by, cx, cy) {{
    const denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    if (Math.abs(denominator) < 1e-9) return null;
    const first = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denominator;
    const second = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denominator;
    const third = 1 - first - second;
    if (first < -0.0001 || second < -0.0001 || third < -0.0001) return null;
    return [first, second, third];
  }}
  function sampleTextureAlpha(source, u, v) {{
    if (!source) return null;
    const x = Math.max(0, Math.min(source.width - 1, Math.round(u * (source.width - 1))));
    // Cubism UV 的原点在左下，而 Canvas ImageData 的原点在左上。
    const y = Math.max(0, Math.min(source.height - 1,
      Math.round((1 - v) * (source.height - 1))));
    let maximum = 0;
    // 纹理边缘的抗锯齿像素可能落在相邻采样点；小范围最大值不会把
    // 完全透明的大块区域误判为可点击，同时保留细发丝/鞋带的交互。
    for (let offsetY = -1; offsetY <= 1; offsetY += 1) {{
      for (let offsetX = -1; offsetX <= 1; offsetX += 1) {{
        const sampleX = Math.max(0, Math.min(source.width - 1, x + offsetX));
        const sampleY = Math.max(0, Math.min(source.height - 1, y + offsetY));
        const alpha = Number(source.pixels[(sampleY * source.width + sampleX) * 4 + 3]) || 0;
        maximum = Math.max(maximum, alpha);
      }}
    }}
    return maximum;
  }}
  function drawableGeometryDetails(localX, localY, collectAll = true) {{
    const core = liveModel && liveModel.internalModel && liveModel.internalModel.coreModel;
    if (!core || typeof core.getModel !== 'function') return null;
    const rawModel = core.getModel();
    const drawables = rawModel && rawModel.drawables;
    const canvasInfo = rawModel && rawModel.canvasinfo;
    const internal = liveModel && liveModel.internalModel;
    const pixelsPerUnit = Number(internal && internal.pixelsPerUnit) ||
      Number(canvasInfo && canvasInfo.PixelsPerUnit);
    const originalWidth = Number(internal && internal.originalWidth) ||
      Number(canvasInfo && canvasInfo.CanvasWidth);
    const originalHeight = Number(internal && internal.originalHeight) ||
      Number(canvasInfo && canvasInfo.CanvasHeight);
    const canvasOriginX = Number(canvasInfo && canvasInfo.CanvasOriginX);
    const canvasOriginY = Number(canvasInfo && canvasInfo.CanvasOriginY);
    const originX = Number.isFinite(canvasOriginX) ? canvasOriginX : originalWidth / 2;
    const originY = Number.isFinite(canvasOriginY) ? canvasOriginY : originalHeight / 2;
    if (
      !drawables || !(pixelsPerUnit > 0) || !(originalWidth > 0) ||
      !(originalHeight > 0)
    ) return null;
    let geometryAvailable = false;
    const hits = [];
    const count = Number(drawables.count) || 0;
    for (let drawable = 0; drawable < count; drawable += 1) {{
      const positions = drawables.vertexPositions && drawables.vertexPositions[drawable];
      const uvs = drawables.vertexUvs && drawables.vertexUvs[drawable];
      const indices = drawables.indices && drawables.indices[drawable];
      const vertexCount = Number(drawables.vertexCounts && drawables.vertexCounts[drawable]) || 0;
      if (!positions || !indices || vertexCount < 3 || indices.length < 3) continue;
      if (drawables.opacities && Number(drawables.opacities[drawable]) <= 0.001) continue;
      // 先做 Drawable 级局部包围盒筛选。多数触点位于模型外或只落在
      // 单个部位；不应为每个事件都对全部三角形计算重心坐标。
      const drawableLimit = Math.min(positions.length, vertexCount * 2);
      let drawableMinX = Infinity;
      let drawableMinY = Infinity;
      let drawableMaxX = -Infinity;
      let drawableMaxY = -Infinity;
      let drawableGeometryFinite = true;
      for (let vertex = 0; vertex + 1 < drawableLimit; vertex += 2) {{
        const vertexX = Number(positions[vertex]) * pixelsPerUnit + originX;
        const vertexY = -Number(positions[vertex + 1]) * pixelsPerUnit + originY;
        if (!Number.isFinite(vertexX) || !Number.isFinite(vertexY)) {{
          drawableGeometryFinite = false;
          break;
        }}
        drawableMinX = Math.min(drawableMinX, vertexX);
        drawableMinY = Math.min(drawableMinY, vertexY);
        drawableMaxX = Math.max(drawableMaxX, vertexX);
        drawableMaxY = Math.max(drawableMaxY, vertexY);
      }}
      if (!drawableGeometryFinite || !Number.isFinite(drawableMinX) ||
          !Number.isFinite(drawableMinY) || !Number.isFinite(drawableMaxX) ||
          !Number.isFinite(drawableMaxY) ||
          localX < drawableMinX - 0.0001 || localX > drawableMaxX + 0.0001 ||
          localY < drawableMinY - 0.0001 || localY > drawableMaxY + 0.0001) continue;
      geometryAvailable = true;
      const textureSource = uvs && drawables.textureIndices
        ? textureAlphaSource(drawables.textureIndices[drawable]) : null;
      let drawableHit = false;
      for (let index = 0; index + 2 < indices.length; index += 3) {{
        const a = Number(indices[index]) * 2;
        const b = Number(indices[index + 1]) * 2;
        const c = Number(indices[index + 2]) * 2;
        if (
          a + 1 >= positions.length || b + 1 >= positions.length ||
          c + 1 >= positions.length
        ) continue;
        const ax = Number(positions[a]) * pixelsPerUnit + originX;
        const ay = -Number(positions[a + 1]) * pixelsPerUnit + originY;
        const bx = Number(positions[b]) * pixelsPerUnit + originX;
        const by = -Number(positions[b + 1]) * pixelsPerUnit + originY;
        const cx = Number(positions[c]) * pixelsPerUnit + originX;
        const cy = -Number(positions[c + 1]) * pixelsPerUnit + originY;
        // 三角形 AABB 是廉价的第二级筛选，避免对同一 Drawable 中
        // 大量不相交面片执行除法/重心坐标和纹理采样。
        const triangleMinX = Math.min(ax, bx, cx);
        const triangleMaxX = Math.max(ax, bx, cx);
        const triangleMinY = Math.min(ay, by, cy);
        const triangleMaxY = Math.max(ay, by, cy);
        if (localX < triangleMinX - 0.0001 || localX > triangleMaxX + 0.0001 ||
            localY < triangleMinY - 0.0001 || localY > triangleMaxY + 0.0001) continue;
        const barycentric = triangleBarycentric(
          localX, localY, ax, ay, bx, by, cx, cy
        );
        if (!barycentric) continue;
        // 没有 UV/纹理 alpha 时使用几何回退；纹理可读时只有实际不透明
        // 像素才算命中，避免透明画布区域触发猫猫头反馈。
        if (!textureSource || !uvs ||
            a + 1 >= uvs.length || b + 1 >= uvs.length || c + 1 >= uvs.length) {{
          drawableHit = true;
          break;
        }}
        const u = barycentric[0] * Number(uvs[a]) +
          barycentric[1] * Number(uvs[b]) + barycentric[2] * Number(uvs[c]);
        const v = barycentric[0] * Number(uvs[a + 1]) +
          barycentric[1] * Number(uvs[b + 1]) + barycentric[2] * Number(uvs[c + 1]);
        const alpha = sampleTextureAlpha(textureSource, u, v);
        if (alpha === null || alpha > 8) {{
          drawableHit = true;
          break;
        }}
      }}
      if (drawableHit) {{
        hits.push(drawable);
        if (!collectAll) break;
      }}
    }}
    const ids = drawables.ids && typeof drawables.ids.length === 'number'
      ? Array.from(drawables.ids) : [];
    const parentPartIndices = drawables.parentPartIndices &&
      typeof drawables.parentPartIndices.length === 'number'
      ? Array.from(drawables.parentPartIndices) : [];
    return {{
      available: geometryAvailable,
      hit: hits.length > 0,
      drawableIndices: hits,
      drawableIds: hits.map(index => ids[index]).filter(value => value !== undefined),
      parentPartIndices: hits.map(index => parentPartIndices[index])
        .filter(value => value !== undefined)
    }};
  }}
  function drawableGeometryDetailsWithTolerance(localX, localY, collectAll = true) {{
    const exact = drawableGeometryDetails(localX, localY, collectAll);
    if (exact === null || exact.hit || !liveModel) return exact;
    // 触点落在抗锯齿边缘时，动作造成的 2~4px 屏幕位移不应让腿部
    // 点击完全失效；只在模型附近做极小的视觉像素容差，不能把整块
    // 透明画布重新变成可点击区域。模型缩放越小，局部坐标容差相应增大。
    // 动态可见网格比静态模型矩形更紧；优先用它做容差外的快速拒绝，
    // 避免透明画布角落触发八次全量三角形扫描。
    const localBounds = finiteBounds(geometryGuard.dynamicBounds)
      ? geometryGuard.dynamicBounds
      : (typeof liveModel.getLocalBounds === 'function' ? liveModel.getLocalBounds() : null);
    const scale = Math.max(0.001, Math.abs(Number(liveModel.scale && liveModel.scale.x) || 1));
    const radius = Math.max(1.5, Math.min(6, 1.5 / scale));
    if (localBounds && (
      localX < Number(localBounds.x) - radius ||
      localY < Number(localBounds.y) - radius ||
      localX > Number(localBounds.x) + Number(localBounds.width) + radius ||
      localY > Number(localBounds.y) + Number(localBounds.height) + radius
    )) return exact;
    const offsets = [
      [-radius, 0], [radius, 0], [0, -radius], [0, radius],
      [-radius, -radius], [-radius, radius], [radius, -radius], [radius, radius]
    ];
    for (const offset of offsets) {{
      const nearby = drawableGeometryDetails(
        localX + offset[0], localY + offset[1], collectAll
      );
      if (nearby && nearby.hit) return nearby;
    }}
    return exact;
  }}
  function hitTestDrawableGeometry(localX, localY) {{
    const details = drawableGeometryDetailsWithTolerance(localX, localY);
    return details === null ? null : Boolean(details.hit);
  }}
  function drawableDebugSummary() {{
    const core = liveModel && liveModel.internalModel && liveModel.internalModel.coreModel;
    const rawModel = core && typeof core.getModel === 'function' ? core.getModel() : null;
    const drawables = rawModel && rawModel.drawables;
    if (!drawables) return null;
    const summary = {{ count: Number(drawables.count) || 0, keys: Object.keys(drawables) }};
    for (const key of [
      'ids', 'drawableIds', 'parentPartIndices', 'renderOrders', 'vertexCounts'
    ]) {{
      const value = drawables[key];
      if (value && typeof value.length === 'number') summary[key] = Array.from(value);
    }}
    const canvasInfo = rawModel && rawModel.canvasinfo;
    if (canvasInfo) summary.canvasinfo = {{
      keys: Object.keys(canvasInfo),
      values: Object.fromEntries(Object.keys(canvasInfo).map(key => [key, canvasInfo[key]]))
    }};
    return summary;
  }}
  function repairRenderOrders() {{
    // Core 6 在每次 update 后可能重新暴露 drawable 顺序；不能只在
    // fromMoc() 初始化时修复，否则动作或窗口移动期间仍可能重新出现穿模。
    const core = liveModel && liveModel.internalModel && liveModel.internalModel.coreModel;
    const rawModel = core && typeof core.getModel === 'function' ? core.getModel() : null;
    const drawables = rawModel && rawModel.drawables;
    if (!core || !drawables) return false;
    const force = arguments.length > 0 ? Boolean(arguments[0]) : false;
    const flags = geometryAuditState.lastDynamicFlags;
    const now = performance.now();
    const changed = Boolean(flags && flags.renderOrderDidChange);
    if (!force && !changed && !renderOrderAuditState.force &&
        renderOrderAuditState.valid && renderOrderAuditState.lastAuditAt &&
        now - renderOrderAuditState.lastAuditAt < 500) return true;
    renderOrderAuditState.force = false;
    renderOrderAuditState.valid = false;
    renderOrderAuditState.lastAuditAt = now;
    let expected = null;
    try {{
      expected = typeof core.getRenderOrders === 'function'
        ? core.getRenderOrders()
        : rawModel.renderOrders;
    }} catch (error) {{
      return false;
    }}
    const count = Number(drawables.count) || 0;
    // Core 的 renderOrders 还可能包含 clipping/offscreen drawable 的顺序，
    // 因此合法长度可以大于 drawableCount；只能拒绝不足以覆盖真实 drawable
    // 的结果，并在写回时保留完整 expected 数组。
    if (!expected || typeof expected.length !== 'number' ||
        !Number.isInteger(count) || count <= 0 || expected.length < count) return false;
    const expectedValues = Array.from(expected, Number);
    const expectedSet = new Set();
    for (let index = 0; index < count; index += 1) {{
      const order = expectedValues[index];
      // Pixi 的排序表必须是 drawable 索引的全排列；仅检查有限值会让
      // 重复/越界顺序把某个 ArtMesh 绘制两次、另一个直接跳过。
      if (!Number.isInteger(order) || order < 0 || order >= count ||
          expectedSet.has(order)) return false;
      expectedSet.add(order);
    }}
    for (let index = count; index < expectedValues.length; index += 1) {{
      if (!Number.isFinite(expectedValues[index])) return false;
    }}
    const actual = drawables.renderOrders;
    const actualValues = actual && typeof actual.length === 'number'
      ? Array.from(actual, Number) : null;
    let healthy = Boolean(actualValues && actualValues.length === expectedValues.length);
    if (healthy) {{
      for (let index = 0; index < expectedValues.length; index += 1) {{
        if (actualValues[index] !== expectedValues[index]) {{
          healthy = false;
          break;
        }}
      }}
    }}
    if (!healthy) {{
      try {{
        const constructor = (actual || expected) && (actual || expected).constructor;
        const repairedValues = constructor && typeof constructor.from === 'function'
          ? constructor.from(expectedValues) : expectedValues;
        drawables.renderOrders = repairedValues;
      }} catch (error) {{
        return false;
      }}
      const repaired = drawables.renderOrders;
      if (!repaired || repaired.length !== expectedValues.length) return false;
      for (let index = 0; index < expectedValues.length; index += 1) {{
        if (Number(repaired[index]) !== expectedValues[index]) return false;
      }}
    }}
    renderOrderAuditState.valid = true;
    return true;
  }}
  function renderedCanvasAlpha(canvas, canvasX, canvasY) {{
    // 几何网格可能仍覆盖被裁剪/透明的 ArtMesh；只有当前 WebGL 帧真正
    // 写入了 alpha，才允许把点击转成部位反馈。直接读取默认 framebuffer
    // 比异步截图更适合 pointer 事件，也能随动作/表情/窗口移动同步更新。
    if (!canvas || !pixiApp) return null;
    const renderer = pixiApp.renderer;
    const context = renderer && (renderer.gl || (renderer.context && renderer.context.gl));
    if (!context || typeof context.readPixels !== 'function') return null;
    const rect = typeof canvas.getBoundingClientRect === 'function'
      ? canvas.getBoundingClientRect() : null;
    const cssWidth = Math.max(1, Number(rect && rect.width) || Number(canvas.clientWidth) || 1);
    const cssHeight = Math.max(1, Number(rect && rect.height) || Number(canvas.clientHeight) || 1);
    const physicalWidth = Math.max(1, Number(renderer.width) ||
      Math.round(cssWidth * (Number(renderer.resolution) || window.devicePixelRatio || 1)));
    const physicalHeight = Math.max(1, Number(renderer.height) ||
      Math.round(cssHeight * (Number(renderer.resolution) || window.devicePixelRatio || 1)));
    const normalizedX = Number(canvasX) / cssWidth;
    const normalizedY = Number(canvasY) / cssHeight;
    if (!Number.isFinite(normalizedX) || !Number.isFinite(normalizedY) ||
        normalizedX < 0 || normalizedY < 0 || normalizedX > 1 || normalizedY > 1) return 0;
    const centerX = Math.max(0, Math.min(physicalWidth - 1,
      Math.round(normalizedX * (physicalWidth - 1))));
    const centerY = Math.max(0, Math.min(physicalHeight - 1,
      Math.round(normalizedY * (physicalHeight - 1))));
    const radius = Math.max(1, Math.min(3,
      Math.ceil(Math.max(physicalWidth / cssWidth, physicalHeight / cssHeight) * 0.75)));
    const pixel = new Uint8Array(4);
    let maximum = 0;
    try {{
      if (typeof context.flush === 'function') context.flush();
      const rgba = Number(context.RGBA) || 0x1908;
      const unsignedByte = Number(context.UNSIGNED_BYTE) || 0x1401;
      for (let offsetY = -radius; offsetY <= radius; offsetY += 1) {{
        for (let offsetX = -radius; offsetX <= radius; offsetX += 1) {{
          const sampleX = Math.max(0, Math.min(physicalWidth - 1, centerX + offsetX));
          const sampleY = Math.max(0, Math.min(physicalHeight - 1, centerY + offsetY));
          context.readPixels(
            sampleX,
            physicalHeight - 1 - sampleY,
            1,
            1,
            rgba,
            unsignedByte,
            pixel
          );
          maximum = Math.max(maximum, Number(pixel[3]) || 0);
        }}
      }}
      return maximum;
    }} catch (error) {{
      // 某些 Chromium/GPU 组合禁止读回默认 framebuffer；此时保留几何
      // 回退，不把一次可选的观测失败升级成模型不可用。
      return null;
    }}
  }}
  function modelHitDetails(event, canvas, options = {{}}) {{
    if (!liveModel || !canvas) return {{ hit: false, reason: 'model-unavailable' }};
    if (liveModel.visible === false) return {{ hit: false, reason: 'model-hidden' }};
    const rect = canvas.getBoundingClientRect();
    const canvasX = Number(event && event.clientX) - Number(rect.left);
    const canvasY = Number(event && event.clientY) - Number(rect.top);
    if (!Number.isFinite(canvasX) || !Number.isFinite(canvasY)) {{
      return {{ hit: false, reason: 'invalid-point' }};
    }}
    try {{
      let localX = canvasX;
      let localY = canvasY;
      if (typeof liveModel.toLocal === 'function' && window.PIXI && PIXI.Point) {{
        const local = liveModel.toLocal(new PIXI.Point(canvasX, canvasY));
        localX = Number(local && local.x);
        localY = Number(local && local.y);
      }}
      if (!Number.isFinite(localX) || !Number.isFinite(localY)) {{
        return {{ hit: false, reason: 'invalid-local-point', canvasX, canvasY }};
      }}
      const collectAll = options.collectAll !== false;
      const sampleRenderedAlpha = options.sampleRenderedAlpha !== false;
      const geometryDetails = drawableGeometryDetailsWithTolerance(
        localX, localY, collectAll
      );
      const geometryHit = geometryDetails === null ? null : Boolean(geometryDetails.hit);
      const bounds = finiteBounds(geometryGuard.dynamicBounds)
        ? geometryGuard.dynamicBounds
        : (typeof liveModel.getLocalBounds === 'function'
          ? liveModel.getLocalBounds()
          : (typeof liveModel.getBounds === 'function' ? liveModel.getBounds() : null));
      const width = Number(bounds && bounds.width) || 0;
      const height = Number(bounds && bounds.height) || 0;
      const originX = Number(bounds && bounds.x) || 0;
      const originY = Number(bounds && bounds.y) || 0;
      const normalizedX = width > 0 ? (localX - originX) / width : 0;
      const normalizedY = height > 0 ? (localY - originY) / height : 0;
      const insideBounds = normalizedX >= 0 && normalizedX <= 1 &&
        normalizedY >= 0 && normalizedY <= 1;
      const renderedAlpha = sampleRenderedAlpha
        ? renderedCanvasAlpha(canvas, canvasX, canvasY) : null;
      const renderedHit = renderedAlpha === null ? null : renderedAlpha > 8;
      const finalHit = geometryHit === false
        ? false
        : renderedHit === null
          ? (geometryHit === null ? insideBounds : geometryHit)
          : Boolean(renderedHit && (geometryHit === null || geometryHit));
      // 下半身的“左/右”是用户在屏幕上看到的左右，而不是模型局部坐标
      // 的左右。方向 B 会把模型沿 X 轴镜像；若继续使用 normalizedX，
      // 用户点击屏幕左侧时会收到 lower_right，反馈标记和动作也会反向。
      const baseTransform = liveModel.__meapetBaseTransform;
      const visibleBounds = finiteBounds(baseTransform && baseTransform.lastBounds)
        ? baseTransform.lastBounds
        : (typeof liveModel.getBounds === 'function' ? liveModel.getBounds() : null);
      const visibleWidth = Number(visibleBounds && visibleBounds.width) || 0;
      const visibleLeft = Number(visibleBounds && visibleBounds.x) || 0;
      // 左右部位以模型当前屏幕 bounds 为基准，而不是整个窗口宽度；
      // 自主漫游或动作平移后，模型靠近窗口边缘仍应保持用户看到的左右。
      const screenNormalizedX = visibleWidth > 0
        ? (canvasX - visibleLeft) / visibleWidth : 0.5;
      const parentParts = geometryDetails && geometryDetails.parentPartIndices || [];
      const drawableIds = geometryDetails && geometryDetails.drawableIds || [];
      const partNames = parentParts.map(index =>
        Array.isArray(window.meapetLive2D && window.meapetLive2D.partNames)
          ? window.meapetLive2D.partNames[index] : undefined
      ).filter(value => typeof value === 'string' && value);
      const normalizedPartNames = partNames.map(value => String(value).toLowerCase());
      const hasFacePart = normalizedPartNames.some(value =>
        value.includes('face') || value.includes('eye') || value.includes('mouth') ||
        value.includes('brow')
      );
      const hasBodyPart = normalizedPartNames.some(value => value.includes('body'));
      const hasSidePart = normalizedPartNames.some(value =>
        value.includes('hair') || value.includes('accessor')
      );
      let part = null;
      if (hasFacePart || normalizedY < 0.34) part = 'head';
      // 身体/后发网格在腿部经常重叠；先按稳定的脚部纵向区域分类，
      // 避免表情姿态或高 DPI 浮点误差把左/右下点击重新归为 body。
      else if ((normalizedY >= 0.74 &&
                (screenNormalizedX < 0.5 || screenNormalizedX > 0.58)) ||
               (hasSidePart && normalizedY >= 0.62 &&
                (screenNormalizedX < 0.38 || screenNormalizedX > 0.62))) {{
        part = screenNormalizedX < 0.5 ? 'lower_left' : 'lower_right';
      }} else if (hasBodyPart) part = 'body';
      else part = 'body';
      return {{
        hit: finalHit,
        geometry: geometryHit,
        renderedAlpha,
        part: finalHit ? part : null,
        drawableIndices: geometryDetails && geometryDetails.drawableIndices || [],
        drawableIds,
        parentPartIndices: parentParts,
        partNames,
        canvasX, canvasY, localX, localY,
        normalizedX, normalizedY, screenNormalizedX,
        boundsX: originX, boundsY: originY, boundsWidth: width, boundsHeight: height
      }};
    }} catch (error) {{
      return {{ hit: false, reason: 'hit-test-error' }};
    }}
  }}
  function isModelHit(event, canvas) {{
    return Boolean(modelHitDetails(event, canvas).hit);
  }}
  function emptyParameterMap() {{
    return Object.create(null);
  }}
  function copyParameterMap(value) {{
    const result = emptyParameterMap();
    if (!value || typeof value !== 'object' || Array.isArray(value)) return result;
    Object.keys(value).forEach(function(id) {{ result[id] = Number(value[id]); }});
    return result;
  }}
  function normalizeParameterMap(value) {{
    if (value === null || typeof value === 'undefined') return emptyParameterMap();
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const entries = Object.entries(value);
    if (entries.length > 32) return null;
    const result = emptyParameterMap();
    for (const entry of entries) {{
      const id = String(entry[0] || '').trim();
      if (typeof entry[1] === 'boolean') return null;
      const numeric = Number(entry[1]);
      if (!/^[A-Za-z][A-Za-z0-9_.:-]{{0,127}}$/.test(id) ||
          !Number.isFinite(numeric) || Math.abs(numeric) > 10000) return null;
      result[id] = numeric;
    }}
    return result;
  }}
  function normalizeActionName(value) {{
    const name = String(value || '').trim();
    return name && name.length <= 64 && !/[\\u0000\\r\\n]/.test(name) ? name : '';
  }}
  function boundedRequestNumber(value, minimum, maximum, fallback) {{
    if (typeof value === 'boolean') return null;
    if (value === null || typeof value === 'undefined') return fallback;
    const numeric = Number(value);
    return Number.isFinite(numeric) && numeric >= minimum && numeric <= maximum
      ? numeric : null;
  }}
  function normalizeExpressionRequest(value) {{
    if (!value || typeof value !== 'object' || Array.isArray(value) ||
        !Array.isArray(value.expressions) || value.expressions.length < 1 ||
        value.expressions.length > 8) return null;
    const mode = String(value.mode || 'sequence').trim().toLowerCase();
    if (mode !== 'sequence' && mode !== 'blend') return null;
    const expressions = [];
    for (const raw of value.expressions) {{
      if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
      const name = normalizeActionName(raw.name);
      const weight = boundedRequestNumber(raw.weight, 0, 1, 1);
      const duration = boundedRequestNumber(raw.duration_seconds, 0.05, 120, 1.5);
      const transition = boundedRequestNumber(raw.transition_seconds, 0, 10, 0.2);
      const parameters = normalizeParameterMap(raw.parameters);
      if (!name || weight === null || duration === null || transition === null ||
          parameters === null) return null;
      expressions.push({{
        name, weight, duration_seconds: duration,
        transition_seconds: transition, parameters
      }});
    }}
    if (mode === 'blend' && expressions.reduce(
      function(total, layer) {{ return total + layer.weight; }}, 0
    ) <= 0) return null;
    const restore = normalizeActionName(value.restore || 'neutral');
    if (!restore) return null;
    return {{ expressions, mode, loop: Boolean(value.loop), restore }};
  }}
  function expressionStageLayer(request, index) {{
    return request.mode === 'blend' ? request.expressions[0] : request.expressions[index];
  }}
  function expressionStageName(request, index) {{
    if (request.mode !== 'blend') return request.expressions[index].name;
    return request.expressions.reduce(function(selected, layer) {{
      return layer.weight > selected.weight ? layer : selected;
    }}, request.expressions[0]).name;
  }}
  function expressionStageParameters(request, index) {{
    if (request.mode !== 'blend') return copyParameterMap(request.expressions[index].parameters);
    const result = emptyParameterMap();
    const total = request.expressions.reduce(
      function(sum, layer) {{ return sum + layer.weight; }}, 0
    ) || 1;
    request.expressions.forEach(function(layer) {{
      Object.keys(layer.parameters).forEach(function(id) {{
        result[id] = Number(result[id] || 0) + Number(layer.parameters[id]) * layer.weight / total;
      }});
    }});
    return result;
  }}
  function configureExpressionParameterStage(target) {{
    const next = copyParameterMap(target);
    const from = emptyParameterMap();
    const ids = new Set(Object.keys(expressionTimelineState.current).concat(Object.keys(next)));
    ids.forEach(function(id) {{
      if (!Object.prototype.hasOwnProperty.call(expressionTimelineState.baseline, id)) {{
        expressionTimelineState.baseline[id] = readParameterValue(id, 0);
      }}
      if (!Object.prototype.hasOwnProperty.call(expressionTimelineState.current, id)) {{
        expressionTimelineState.current[id] = expressionTimelineState.baseline[id];
      }}
      from[id] = expressionTimelineState.current[id];
      if (!Object.prototype.hasOwnProperty.call(next, id)) {{
        next[id] = expressionTimelineState.baseline[id];
      }}
    }});
    expressionTimelineState.from = from;
    expressionTimelineState.target = next;
  }}
  function applyExpressionParameterProgress(progress) {{
    const amount = clampValue(Number(progress) || 0, 0, 1);
    let changed = false;
    Object.keys(expressionTimelineState.target).forEach(function(id) {{
      const start = Number(expressionTimelineState.from[id]);
      const target = Number(expressionTimelineState.target[id]);
      const value = start + (target - start) * amount;
      expressionTimelineState.current[id] = value;
      changed = setParameterValue(id, value) || changed;
    }});
    return changed;
  }}
  function resetExpressionTimeline(restoreParameters) {{
    if (restoreParameters) {{
      Object.keys(expressionTimelineState.baseline).forEach(function(id) {{
        setParameterValue(id, expressionTimelineState.baseline[id]);
      }});
    }}
    expressionTimelineState.active = false;
    expressionTimelineState.phase = 'idle';
    expressionTimelineState.request = null;
    expressionTimelineState.index = 0;
    expressionTimelineState.elapsed = 0;
    expressionTimelineState.baseline = emptyParameterMap();
    expressionTimelineState.current = emptyParameterMap();
    expressionTimelineState.from = emptyParameterMap();
    expressionTimelineState.target = emptyParameterMap();
  }}
  function beginExpressionStage(index) {{
    const request = expressionTimelineState.request;
    if (!request) return false;
    expressionTimelineState.phase = 'active';
    expressionTimelineState.index = index;
    expressionTimelineState.elapsed = 0;
    configureExpressionParameterStage(expressionStageParameters(request, index));
    const accepted = window.meapetLive2D.setExpression(
      expressionStageName(request, index), true
    );
    if (accepted !== false) applyExpressionParameterProgress(0);
    return accepted !== false;
  }}
  function beginExpressionRestore() {{
    const request = expressionTimelineState.request;
    if (!request) return false;
    // 在最后一层结束的边界保留目标参数，随后再从该目标平滑回到
    // baseline。正式/过程表情切换可能会同步改写 Core 参数，先保存
    // current 并在 setExpression 后恢复 from，避免恢复阶段第一帧
    // 被 neutral 的即时姿态覆盖。
    const restoreFrom = copyParameterMap(expressionTimelineState.current);
    expressionTimelineState.phase = 'restore';
    expressionTimelineState.elapsed = 0;
    configureExpressionParameterStage(expressionTimelineState.baseline);
    const accepted = window.meapetLive2D.setExpression(request.restore, true);
    if (accepted !== false) {{
      expressionTimelineState.from = restoreFrom;
      expressionTimelineState.current = copyParameterMap(restoreFrom);
      applyExpressionParameterProgress(0);
    }}
    return accepted !== false;
  }}
  function advanceExpressionTimeline(seconds) {{
    if (!expressionTimelineState.active || !expressionTimelineState.request) return false;
    const request = expressionTimelineState.request;
    const dt = Math.max(0, Number(seconds) || 0);
    expressionTimelineState.elapsed += dt;
    const layer = expressionStageLayer(request, expressionTimelineState.index);
    const transition = Number(layer.transition_seconds) || 0;
    const progress = transition <= 0
      ? 1 : Math.min(1, expressionTimelineState.elapsed / transition);
    const changed = applyExpressionParameterProgress(progress);
    if (expressionTimelineState.phase === 'restore') {{
      if (progress >= 1) resetExpressionTimeline(false);
      return changed;
    }}
    const duration = request.mode === 'blend'
      ? Math.max.apply(null, request.expressions.map(function(item) {{
          return Number(item.duration_seconds) || 0;
        }}))
      : Number(layer.duration_seconds) || 0;
    if (expressionTimelineState.elapsed < duration) return changed;
    if (request.mode === 'sequence' &&
        expressionTimelineState.index + 1 < request.expressions.length) {{
      if (!beginExpressionStage(expressionTimelineState.index + 1)) beginExpressionRestore();
      return changed;
    }}
    if (request.loop) {{
      if (!beginExpressionStage(0)) beginExpressionRestore();
      return changed;
    }}
    beginExpressionRestore();
    return changed;
  }}
  function normalizeMotionRequest(value) {{
    if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
    const name = normalizeActionName(value.name);
    const duration = value.duration_seconds === null ||
      typeof value.duration_seconds === 'undefined'
      ? null : boundedRequestNumber(value.duration_seconds, 0.05, 300, null);
    const transition = boundedRequestNumber(value.transition_seconds, 0, 10, 0.2);
    const parameters = normalizeParameterMap(value.parameters);
    if (!name || (value.duration_seconds !== null &&
        typeof value.duration_seconds !== 'undefined' && duration === null) ||
        transition === null || parameters === null) return null;
    return {{
      name, duration_seconds: duration, transition_seconds: transition,
      loop: Boolean(value.loop), parameters
    }};
  }}
  function motionRequestDuration(request) {{
    if (Number(request.duration_seconds) > 0) return Number(request.duration_seconds);
    const alias = motionAliases.get(String(request.name).toLowerCase());
    const resolved = String(alias || request.name);
    const declared = Number(motionDurations.get(resolved) || 0);
    if (declared > 0) return declared;
    const normalized = String(request.name).toLowerCase();
    if (normalized === 'blink') return 0.45;
    if (normalized === 'wave') return 0.9;
    return 0;
  }}
  function resetMotionRequest(restoreParameters) {{
    if (restoreParameters) {{
      Object.keys(motionRequestState.baseline).forEach(function(id) {{
        setParameterValue(id, motionRequestState.baseline[id]);
      }});
    }}
    motionRequestState.active = false;
    motionRequestState.request = null;
    motionRequestState.elapsed = 0;
    motionRequestState.baseline = emptyParameterMap();
    motionRequestState.current = emptyParameterMap();
  }}
  function advanceMotionRequest(seconds) {{
    if (!motionRequestState.active || !motionRequestState.request) return false;
    const request = motionRequestState.request;
    const dt = Math.max(0, Number(seconds) || 0);
    motionRequestState.elapsed += dt;
    const transition = Number(request.transition_seconds) || 0;
    const duration = motionRequestDuration(request);
    let progress = transition <= 0 ? 1 : Math.min(1, motionRequestState.elapsed / transition);
    if (duration > 0 && transition > 0) {{
      progress = Math.min(
        progress,
        Math.max(0, (duration - motionRequestState.elapsed) / transition)
      );
    }}
    let changed = false;
    Object.keys(request.parameters).forEach(function(id) {{
      const start = Number(motionRequestState.baseline[id]);
      const target = Number(request.parameters[id]);
      const value = start + (target - start) * progress;
      motionRequestState.current[id] = value;
      changed = setParameterValue(id, value) || changed;
    }});
    if (!(duration > 0) || motionRequestState.elapsed < duration) return changed;
    if (request.loop) {{
      motionRequestState.elapsed %= duration;
      Object.keys(motionRequestState.baseline).forEach(function(id) {{
        motionRequestState.current[id] = motionRequestState.baseline[id];
        setParameterValue(id, motionRequestState.baseline[id]);
      }});
      window.meapetLive2D.playMotion(request.name, true);
      return changed;
    }}
    const finishedName = request.name;
    resetMotionRequest(true);
    window.meapetLive2D.playMotion('idle', true);
    const bridge = window.meapetBridge;
    if (bridge && typeof bridge.motionFinished === 'function') {{
      try {{ bridge.motionFinished(finishedName); }} catch (error) {{}}
    }}
    return changed;
  }}
  function stopFormalMotionOnce() {{
    if (motionRequestState.active) return false;
    const duration = Number(motionDurations.get(motionState.name) || 0);
    if (motionState.procedural || duration <= 0 || motionState.elapsed < duration) return false;
    const finishedName = motionState.name;
    const manager = liveModel && liveModel.internalModel && liveModel.internalModel.motionManager;
    if (manager && typeof manager.stopAllMotions === 'function') manager.stopAllMotions();
    motionState.name = 'idle';
    motionState.elapsed = 0;
    motionState.procedural = true;
    resetGeometryContinuity(2);
    const bridge = window.meapetBridge;
    if (bridge && typeof bridge.motionFinished === 'function') {{
      try {{ bridge.motionFinished(finishedName); }} catch (error) {{}}
    }}
    return true;
  }}
  window.meapetLive2D = {{
    modelUrl: modelUrl,
    reloadModel: function(url, metadata, requestId) {{
      return reloadModel(url, metadata, requestId);
    }},
    acceptModelReload: function(requestId) {{
      return acceptModelReload(requestId);
    }},
    finalizeModelReload: function(requestId) {{
      return finalizeModelReload(requestId);
    }},
    rollbackModelReload: function(requestId) {{
      return rollbackModelReload(requestId);
    }},
    partNames: modelPartNames,
    setFrameRate: function(value) {{
      return setFrameRate(value);
    }},
    setGeometryAuditHz: function(value) {{
      return setGeometryAuditHz(value);
    }},
    // Qt 在调整顶层窗口尺寸后直接调用这个入口，避免只派发 resize
    // 事件时与 Chromium 的异步布局队列发生一帧错位，导致模型暂时沿用
    // 上一个窗口尺寸并出现裁切或缩放滞后。
    resize: function() {{
      resize(arguments[0], arguments[1]);
      return Boolean(pixiApp);
    }},
    advance: function(seconds, source) {{
      const clockNow = performance.now();
      const sourceName = String(source || '').toLowerCase();
      const externalClock = sourceName === 'external';
      // 只有页面自己的 rAF 时钟需要按墙钟限频。宿主手动 advance 和
      // 测试/动作控制调用表示明确的逻辑时间步，不能因连续调用间隔
      // 小于一帧而静默丢弃，否则时间线和口型会停在旧状态。
      const throttleByWallClock = sourceName === 'raf';
      if (externalClock) {{
        clockState.externalUntil = clockNow + externalClockHoldMs;
      }} else if (clockNow < clockState.externalUntil) {{
        clockState.skippedRafFrames += 1;
        return;
      }}
      const requestedDt = Math.max(0, Number(seconds) || 0);
      // 拖动时保留当前画面，并丢弃暂停期间的时间；释放后从新的时间基线
      // 继续，避免动作/物理在下一帧一次性追赶导致几何跳变。
      if (clockState.dragging) return;
      renderPerformanceState.pendingSeconds = Math.min(
        0.25, renderPerformanceState.pendingSeconds + requestedDt
      );
      const elapsedSinceRender = renderPerformanceState.lastRenderAt
        ? Math.max(0, clockNow - renderPerformanceState.lastRenderAt)
        : Infinity;
      if (throttleByWallClock &&
        renderPerformanceState.lastRenderAt &&
        elapsedSinceRender + 0.25 < renderPerformanceState.renderIntervalMs
      ) {{
        renderPerformanceState.throttledFrames += 1;
        return;
      }}
      const dt = Math.min(0.05, renderPerformanceState.pendingSeconds);
      renderPerformanceState.pendingSeconds = Math.max(
        0, renderPerformanceState.pendingSeconds - dt
      );
      renderPerformanceState.lastRenderAt = clockNow;
      renderPerformanceState.renderedFrames += 1;
      // WebEngine/合成器在窗口移动、切屏或短暂阻塞后可能一次性回调
      // 很大的时间差。把整段差值限制在 50ms，后续帧再自然追上。
      if (requestedDt > 0.05) {{
        cursorState.velocityX = 0;
        cursorState.velocityY = 0;
        trackingGuard.largeDeltaClamps += 1;
        if (requestedDt > 0.2) resetGeometryContinuity(2);
      }}
      speechElapsed += dt;
      synchronizeViewport();
      const trackingDt = Math.min(0.05, dt);
      const trackingX = smoothTrackingAxis(
        cursorState.currentX, cursorState.targetX, cursorState.velocityX, trackingDt
      );
      const trackingY = smoothTrackingAxis(
        cursorState.currentY, cursorState.targetY, cursorState.velocityY, trackingDt
      );
      cursorState.currentX = trackingX.value;
      cursorState.currentY = trackingY.value;
      cursorState.velocityX = trackingX.velocity;
      cursorState.velocityY = trackingY.velocity;
      if (liveModel && liveModel.internalModel && liveModel.internalModel.update) {{
        // pixi-live2d-display 的内部更新接口使用毫秒，而宿主渲染协议
        // 使用秒；同时关闭模型自带 ticker，避免每帧重复推进两次。
        liveModel.internalModel.update(dt * 1000, performance.now());
      }}
      if (!repairRenderOrders()) {{
        // 顺序来源或写回校验失败时宁可隐藏模型并结束页面，也不能让
        // 未验证的 drawable 顺序继续进入 WebGL 帧，避免动作期间穿模。
        liveModel.visible = false;
        if (!renderOrderErrorReported) {{
          renderOrderErrorReported = true;
          reportError(new Error('Live2D render order validation failed'));
        }}
        return;
      }}
      liveModel.visible = true;
      // Cubism 的内部更新会重新装载保存的参数；只有兼容状态才重放 CDI
      // 表情/过程动作，真实 model3 动作交给自身管理器，不能再被默认状态覆盖。
      if (liveModel && expressionState.procedural) applyProceduralExpression(expressionState.name);
      let frameValid = true;
      if (motionState.procedural) applyProceduralMotion(dt);
      frameValid = liveModel.visible !== false;
      if (!motionState.procedural && liveModel) {{
        motionState.elapsed += dt;
        if (liveModel.__meapetBaseTransform) {{
          liveModel.__meapetBaseTransform.motionOffsetX = 0;
          liveModel.__meapetBaseTransform.motionOffsetY = 0;
        }}
        frameValid = applyFormalParameterMotion(dt) !== false && frameValid;
        stopFormalMotionOnce();
        frameValid = refreshCoreGeometry() && repairRenderOrders() && frameValid;
        frameValid = fitModelToViewport() && frameValid;
        drawFaceOverlay();
      }}
      const expressionRequestChanged = advanceExpressionTimeline(dt);
      const motionRequestChanged = advanceMotionRequest(dt);
      if (expressionRequestChanged || motionRequestChanged) {{
        const refreshed = refreshCoreGeometry();
        if (refreshed) coreParameterDirty = false;
        frameValid = refreshed && repairRenderOrders() &&
          fitModelToViewport() && frameValid;
        drawFaceOverlay();
      }}
      // 程序化姿态/口型/注视通常不会触发表情时间线分支；在所有本帧
      // 参数写入完成后至少刷新一次 Core，确保绘制、拟合和命中使用同一
      // 份最新顶点，而不是延迟到下一帧才生效。
      if (coreParameterDirty) {{
        const refreshed = flushCoreParameterGeometry();
        frameValid = refreshed && repairRenderOrders() &&
          fitModelToViewport() && frameValid;
        drawFaceOverlay();
      }}
      if (!enforceUniformTransform()) liveModel.visible = false;
      if (!frameValid) liveModel.visible = false;
    }},
    setCursorTarget: function(x, y, width, height) {{
      return setCursorTarget(x, y, width, height);
    }},
    notifyWindowMoved: function() {{
      cursorState.velocityX = 0;
      cursorState.velocityY = 0;
      cursorState.targetGeneration += 1;
      return true;
    }},
    setInteractionLocked: function(locked) {{
      cursorState.locked = Boolean(locked);
      return true;
    }},
    setDragging: function(dragging) {{
      const requested = Boolean(dragging);
      // 页面本地释放保护优先于跨 WebChannel 的宿主回调；否则宿主的
      // ``setDragging(false)`` 可能先于最后一个 move/release 到达，重新
      // 开启 rAF 并在窗口移动尾帧造成抽搐。
      if (!requested && performance.now() < Number(clockState.dragReleaseUntil || 0)) {{
        // 宿主的 false 已经确认指针事务结束；即使本地 release timer
        // 因页面节流/销毁没有继续回调，也必须清除 host 标记，否则下面
        // 的兜底条件会永远认为仍在拖动，动作时钟和 pending viewport 无法恢复。
        clockState.hostDragging = false;
        // 保留 dragging 到稳定窗口结束，避免最后一个 move/release 尚未
        // 合成时提前推进一帧；真正仍有本地指针时，后续 true 请求会覆盖该标记。
        if (resizeState.pending) scheduleDeferredResizeFlush(32);
        return true;
      }}
      clockState.hostDragging = requested;
      clockState.dragging = requested;
      if (clockState.dragging) {{
        cursorState.velocityX = 0;
        cursorState.velocityY = 0;
        renderPerformanceState.pendingSeconds = 0;
        renderPerformanceState.lastRenderAt = performance.now();
        geometryAuditState.force = true;
        if (resizeState.flushTimer !== null) {{
          clearTimeout(resizeState.flushTimer);
          resizeState.flushTimer = null;
        }}
        resizeState.flushRetries = 0;
      }} else {{
        // 释放后从当前时间重新开始，避免把拖动期间停留的时间一次性
        // 计入物理/动作；下一次外部或 rAF tick 会自然继续。
        lastFrameTime = performance.now();
        resetGeometryContinuity(1);
        // 时钟已恢复后只应用拖动期间记录的最后 viewport，随后由
        // ResizeObserver 的同尺寸回调去重，避免释放瞬间连续重建缩放。
        scheduleDeferredResizeFlush(32);
      }}
      return true;
    }},
    setExpressionRequest: function(value) {{
      if (!liveModel) return false;
      const request = normalizeExpressionRequest(value);
      if (!request) return false;
      resetExpressionTimeline(true);
      expressionTimelineState.active = true;
      expressionTimelineState.request = request;
      request.expressions.forEach(function(layer) {{
        Object.keys(layer.parameters).forEach(function(id) {{
          if (!Object.prototype.hasOwnProperty.call(expressionTimelineState.baseline, id)) {{
            const baseline = readParameterValue(id, 0);
            expressionTimelineState.baseline[id] = baseline;
            expressionTimelineState.current[id] = baseline;
          }}
        }});
      }});
      if (!beginExpressionStage(0)) {{
        resetExpressionTimeline(true);
        return false;
      }}
      return true;
    }},
    setExpression: function(name, preserveTimeline) {{
      if (!liveModel) return false;
      const requested = String(name || '').trim();
      if (!requested) return false;
      // Python 桥接层已校验一次；页面公开对象仍需拒绝只差大小写的
      // 正式名称，避免把它误交给过程回退。其它自定义名称交给 Cubism
      // manager，由运行时自己的动作清单决定是否接受。
      if (modelUrl) {{
        const folded = requested.toLowerCase();
        const declaredCaseCollision = Array.from(declaredExpressions).some(
          value => String(value).toLowerCase() === folded && String(value) !== requested
        );
        if (declaredCaseCollision) return false;
      }}
      if (!preserveTimeline) resetExpressionTimeline(true);
      const requestGeneration = ++expressionRequestGeneration;
      // model3 中声明的精确名称优先于兼容名称；这样名为 ``happy`` 或
      // ``neutral`` 的真实表情不会被同名过程回退拦截。
      if (!declaredExpressions.has(requested)) {{
        const normalized = requested.toLowerCase();
        if (proceduralExpressions.has(normalized)) {{
          resetGeometryContinuity(2);
          return applyProceduralExpression(normalized, true);
        }}
      }}
      if (typeof liveModel.expression === 'function') {{
        try {{
          const result = liveModel.expression(requested);
          const markAccepted = function(value) {{
            if (requestGeneration === expressionRequestGeneration && value !== false) {{
              expressionState.name = requested;
              expressionState.procedural = false;
              resetGeometryContinuity(2);
            }}
          }};
          if (result && typeof result.then === 'function') {{
            // pixi-live2d-display 的 expression() 返回 Promise；显式消费
            // 拒绝分支，避免资源加载失败变成 Chromium 未处理异常。
            result.then(markAccepted, function() {{}});
          }} else {{
            markAccepted(result);
          }}
          return result;
        }} catch (error) {{
          return false;
        }}
      }}
      const manager = liveModel.internalModel && liveModel.internalModel.motionManager &&
        liveModel.internalModel.motionManager.expressionManager;
      let expressionResult = false;
      if (manager && manager.setExpression) {{
        expressionResult = manager.setExpression(requested);
        if (expressionResult && typeof expressionResult.then === 'function') {{
          expressionResult.then(function() {{}}, function() {{}});
        }}
      }}
      const accepted = Boolean(
        expressionResult
      );
      if (accepted) {{
        expressionState.name = requested;
        expressionState.procedural = false;
        resetGeometryContinuity(2);
      }}
      return accepted;
    }},
    playMotionRequest: function(value) {{
      if (!liveModel) return false;
      const request = normalizeMotionRequest(value);
      if (!request) return false;
      resetMotionRequest(true);
      Object.keys(request.parameters).forEach(function(id) {{
        const baseline = readParameterValue(id, 0);
        motionRequestState.baseline[id] = baseline;
        motionRequestState.current[id] = baseline;
      }});
      motionRequestState.active = true;
      motionRequestState.request = request;
      motionRequestState.elapsed = 0;
      const accepted = window.meapetLive2D.playMotion(request.name, true);
      if (accepted === false) {{
        resetMotionRequest(true);
        return false;
      }}
      return true;
    }},
    playMotion: function(name, preserveRequest) {{
      if (!liveModel) return false;
      const requested = String(name || '').trim();
      if (!requested) return false;
      const aliasedMotion = motionAliases.get(requested.toLowerCase());
      const resolvedMotion = String(aliasedMotion || requested).trim();
      if (modelUrl) {{
        const folded = requested.toLowerCase();
        const declaredCaseCollision = Array.from(declaredMotions).some(
          value => String(value).toLowerCase() === String(resolvedMotion).toLowerCase() &&
            String(value) !== resolvedMotion
        );
        if (declaredCaseCollision && !motionAliases.has(folded)) return false;
      }}
      if (!preserveRequest) resetMotionRequest(true);
      const requestGeneration = ++motionRequestGeneration;
      const activeMotionManager = liveModel.internalModel &&
        liveModel.internalModel.motionManager;
      // model3 中声明的精确动作组优先于过程兼容词汇；例如真实的
      // ``Idle`` 动作组必须进入 Cubism 动作管理器。
      if (!declaredMotions.has(resolvedMotion)) {{
        const normalized = requested.toLowerCase();
        if (proceduralMotions.has(normalized)) {{
          if (activeMotionManager && typeof activeMotionManager.stopAllMotions === 'function') {{
            activeMotionManager.stopAllMotions();
          }}
          motionState.name = normalized;
          motionState.elapsed = 0;
          motionState.procedural = true;
          resetGeometryContinuity(2);
          return true;
        }}
      }}
      try {{
        if (activeMotionManager && typeof activeMotionManager.stopAllMotions === 'function') {{
          activeMotionManager.stopAllMotions();
        }}
        if (typeof liveModel.motion === 'function') {{
          const priorityTable = window.PIXI && PIXI.live2d && PIXI.live2d.MotionPriority;
          const forcePriority = Number(priorityTable && priorityTable.FORCE) || 3;
          // play_motion 表示用户/模型的显式切换，必须覆盖仍在播放的上一
          // 动作。单参数 motion() 会走随机动作并以普通优先级保留旧动作；
          // 当前组使用确定的首个 motion3 和 FORCE 优先级，A→B 切换才能
          // 与控制台状态保持一致。
          const result = liveModel.motion(resolvedMotion, 0, forcePriority);
          const markAccepted = function(value) {{
            if (requestGeneration === motionRequestGeneration && value !== false) {{
              motionState.name = resolvedMotion;
              motionState.elapsed = 0;
              motionState.procedural = false;
              resetGeometryContinuity(2);
            }}
          }};
          if (result && typeof result.then === 'function') {{
            result.then(markAccepted, function() {{}});
          }} else {{
            markAccepted(result);
          }}
          return result !== false;
        }}
        const manager = activeMotionManager;
        let motionResult = false;
        if (manager && manager.startMotion) {{
          const priorityTable = window.PIXI && PIXI.live2d && PIXI.live2d.MotionPriority;
          const forcePriority = Number(priorityTable && priorityTable.FORCE) || 3;
          motionResult = manager.startMotion(resolvedMotion, 0, forcePriority);
          if (motionResult && typeof motionResult.then === 'function') {{
            motionResult.then(function() {{}}, function() {{}});
          }}
        }}
        const accepted = Boolean(motionResult);
        if (accepted) {{
          motionState.name = resolvedMotion;
          motionState.elapsed = 0;
          motionState.procedural = false;
          resetGeometryContinuity(2);
        }}
        return accepted;
      }} catch (error) {{
        return false;
      }}
    }},
    setDirection: function(name) {{
      const normalized = String(name || '').trim().toUpperCase();
      if (normalized !== 'A' && normalized !== 'B') return false;
      facingTarget = normalized === 'B' ? -1 : 1;
      facingState.target = facingTarget;
      resetGeometryContinuity(2);
      return true;
    }},
    setPartOpacity: function(name, value) {{
      return setPartOpacity(String(name || ''), Number(value));
    }},
    debug: function() {{
      if (!liveModel) return null;
      const debugCanvas = document.getElementById('meapet-live2d');
      const debugHost = debugCanvas && debugCanvas.parentElement;
      const expectedViewportWidth = Math.max(
        1, Number(debugHost && debugHost.clientWidth) || window.innerWidth || 1
      );
      const expectedViewportHeight = Math.max(
        1, Number(debugHost && debugHost.clientHeight) || window.innerHeight || 1
      );
      const debugRenderer = pixiApp && pixiApp.renderer;
      const debugScreen = debugRenderer && debugRenderer.screen;
      const debugResolution = Math.max(
        1, Number(debugRenderer && debugRenderer.resolution) || window.devicePixelRatio || 1
      );
      const actualViewportWidth = Number(debugScreen && debugScreen.width) ||
        (Number(debugRenderer && debugRenderer.width) || 0) / debugResolution;
      const actualViewportHeight = Number(debugScreen && debugScreen.height) ||
        (Number(debugRenderer && debugRenderer.height) || 0) / debugResolution;
      const staticBounds = typeof liveModel.getBounds === 'function'
        ? liveModel.getBounds() : null;
      const baseTransform = liveModel.__meapetBaseTransform;
      const bounds = finiteBounds(baseTransform && baseTransform.lastBounds)
        ? baseTransform.lastBounds : staticBounds;
      const core = liveModel.internalModel && liveModel.internalModel.coreModel;
      const rawModel = core && typeof core.getModel === 'function' ? core.getModel() : null;
      const rawDrawables = rawModel && rawModel.drawables;
      const rawParts = rawModel && rawModel.parts;
      const rawPartIds = rawParts && rawParts.ids;
      const rawParentIndices = rawDrawables && rawDrawables.parentPartIndices;
      const rawOpacities = rawDrawables && rawDrawables.opacities;
      function drawableOpacityForPart(partId) {{
        if (!rawPartIds || !rawParentIndices || !rawOpacities) return [];
        const partIndex = rawPartIds.indexOf(partId);
        if (partIndex < 0) return [];
        const result = [];
        for (let index = 0; index < rawParentIndices.length; index += 1) {{
          if (Number(rawParentIndices[index]) === partIndex) {{
            result.push(Number(rawOpacities[index]));
          }}
        }}
        return result;
      }}
      return {{
        modelUrl: modelUrl,
        modelReloadGeneration: Number(modelReloadGeneration) || 0,
        modelReloadPending: pendingModelReload
          ? Number(pendingModelReload.id) || 0 : null,
        modelVisible: liveModel.visible !== false,
        renderOrderValid: repairRenderOrders(),
        expression: expressionState.name,
        expressionProcedural: Boolean(expressionState.procedural),
        expressionTimelineActive: Boolean(expressionTimelineState.active),
        expressionTimelineMode: expressionTimelineState.request
          ? expressionTimelineState.request.mode : null,
        expressionTimelinePhase: expressionTimelineState.phase,
        expressionTimelineIndex: Number(expressionTimelineState.index) || 0,
        expressionTimelineElapsed: Number(expressionTimelineState.elapsed) || 0,
        expressionTimelineParameterCount: Object.keys(expressionTimelineState.current).length,
        motion: motionState.name,
        motionProcedural: Boolean(motionState.procedural),
        motionRequestActive: Boolean(motionRequestState.active),
        motionRequestElapsed: Number(motionRequestState.elapsed) || 0,
        motionRequestLoop: Boolean(
          motionRequestState.request && motionRequestState.request.loop
        ),
        motionRequestParameterCount: Object.keys(motionRequestState.current).length,
        width: Math.abs(Number(liveModel.width) || 0),
        height: Number(liveModel.height) || 0,
        x: Number(liveModel.x) || 0,
        y: Number(liveModel.y) || 0,
        scaleX: Number(liveModel.scale && liveModel.scale.x) || 0,
        scaleY: Number(liveModel.scale && liveModel.scale.y) || 0,
        boundsX: Number(bounds && bounds.x) || 0,
        boundsY: Number(bounds && bounds.y) || 0,
        boundsWidth: Number(bounds && bounds.width) || 0,
        boundsHeight: Number(bounds && bounds.height) || 0,
        staticBoundsX: Number(staticBounds && staticBounds.x) || 0,
        staticBoundsY: Number(staticBounds && staticBounds.y) || 0,
        staticBoundsWidth: Number(staticBounds && staticBounds.width) || 0,
        staticBoundsHeight: Number(staticBounds && staticBounds.height) || 0,
        angleX: parameterBindingState.ParamAngleX
          ? Number(core && core.getParameterValueById("ParamAngleX")) || 0 : null,
        angleY: parameterBindingState.ParamAngleY
          ? Number(core && core.getParameterValueById("ParamAngleY")) || 0 : null,
        parameterBindings: Object.assign({{}}, parameterBindingState),
        parameterSafety: Object.assign({{}}, parameterSafetyState),
        facialBindingCount: Number(facialBindingCount) || 0,
        expressionBindingCount: Number(expressionBindingCount) || 0,
        expressionPoseScale: Number(expressionPoseScale) || 0,
        faceOverlayEnabled: Boolean(faceOverlayEnabled),
        eyeOverlayEnabled: Boolean(eyeOverlayEnabled),
        mouthOverlayEnabled: Boolean(mouthOverlayEnabled),
        expressionOverlayEnabled: Boolean(expressionOverlayEnabled),
        motionSyncAvailable: Boolean(motionSyncProfile),
        motionSyncViseme: motionSyncState.viseme,
        motionSyncIntensity: Number(motionSyncState.intensity) || 0,
        motionSyncMouthForm: Number(motionSyncState.mouthForm) || 0,
        motionSyncMouthOpenY: Number(motionSyncState.mouthOpenY) || 0,
        motionSyncAppliedParameters: Number(motionSyncState.appliedParameters) || 0,
        motionSyncVisemes: motionSyncProfile && Array.isArray(motionSyncProfile.audio_parameters)
          ? motionSyncProfile.audio_parameters.map(item => item && item.id).filter(Boolean)
          : [],
        eyeParameterOpenL: Number(
          core && core.getParameterValueById && core.getParameterValueById("ParamEyeLOpen")
        ) || 0,
        eyeParameterOpenR: Number(
          core && core.getParameterValueById && core.getParameterValueById("ParamEyeROpen")
        ) || 0,
        eyeOpacityL: Number(
          core && core.getPartOpacityById && core.getPartOpacityById("Eye_L")
        ) || 0,
        eyeOpacityR: Number(
          core && core.getPartOpacityById && core.getPartOpacityById("Eye_R")
        ) || 0,
        eyeDrawableOpacitiesL: drawableOpacityForPart("Eye_L"),
        eyeDrawableOpacitiesR: drawableOpacityForPart("Eye_R"),
        speechActive,
        speechElapsed: Number(speechElapsed) || 0,
        targetAngleX: Number(expressionState.targetAngle.x) || 0,
        targetAngleY: Number(expressionState.targetAngle.y) || 0,
        cursorTargetX: Number(cursorState.targetX) || 0,
        cursorTargetY: Number(cursorState.targetY) || 0,
        cursorCurrentX: Number(cursorState.currentX) || 0,
        cursorCurrentY: Number(cursorState.currentY) || 0,
        cursorVelocityX: Number(cursorState.velocityX) || 0,
        cursorVelocityY: Number(cursorState.velocityY) || 0,
        cursorTargetGeneration: Number(cursorState.targetGeneration) || 0,
        largeDeltaClamps: Number(trackingGuard.largeDeltaClamps) || 0,
        externalClockActive: performance.now() < clockState.externalUntil,
        skippedRafFrames: Number(clockState.skippedRafFrames) || 0,
        targetFrameRate: Number(renderPerformanceState.targetFrameRate) || 0,
        renderIntervalMs: Number(renderPerformanceState.renderIntervalMs) || 0,
        renderedFrames: Number(renderPerformanceState.renderedFrames) || 0,
        throttledFrames: Number(renderPerformanceState.throttledFrames) || 0,
        geometryAuditHz: Number(renderPerformanceState.geometryAuditHz) || 0,
        geometryAuditIntervalMs: Number(renderPerformanceState.geometryAuditIntervalMs) || 0,
        geometryAudits: Number(renderPerformanceState.geometryAudits) || 0,
        geometryAuditCacheValid: Boolean(geometryAuditState.cacheValid),
        geometryAuditCacheHits: Number(renderPerformanceState.geometryAuditCacheHits) || 0,
        geometryAuditCacheMisses: Number(renderPerformanceState.geometryAuditCacheMisses) || 0,
        geometryVertexDirtyFrames: Number(renderPerformanceState.geometryVertexDirtyFrames) || 0,
        geometryStructuralDirtyFrames: Number(
          renderPerformanceState.geometryStructuralDirtyFrames
        ) || 0,
        geometryAuditDynamicFlags: geometryAuditState.lastDynamicFlags,
        appliedCursorX: Number(poseState.appliedCursorX) || 0,
        appliedCursorY: Number(poseState.appliedCursorY) || 0,
        desiredPoseX: Number(poseState.desiredX) || 0,
        desiredPoseY: Number(poseState.desiredY) || 0,
        appliedPoseX: Number(poseState.currentX) || 0,
        appliedPoseY: Number(poseState.currentY) || 0,
        facing: Number(facing) || 1,
        facingPose: Number(facingState.current) || 1,
        facingSign: geometryFacingSign(),
        safeScale: Number(liveModel.__meapetBaseTransform &&
          liveModel.__meapetBaseTransform.scale) || 0,
        safeOffsetX: Number(liveModel.__meapetBaseTransform &&
          liveModel.__meapetBaseTransform.safeOffsetX) || 0,
        safeOffsetY: Number(liveModel.__meapetBaseTransform &&
          liveModel.__meapetBaseTransform.safeOffsetY) || 0,
        uniformScaleCorrections: Number(transformGuard.uniformScaleCorrections) || 0,
        geometryValid: Boolean(geometryGuard.geometryValid),
        geometryInvalidFrames: Number(geometryGuard.invalidFrames) || 0,
        geometryContinuityValid: Boolean(geometryGuard.continuityValid),
        geometryContinuityRejects: Number(geometryGuard.continuityRejects) || 0,
        geometryContinuityResets: Number(geometryGuard.continuityResets) || 0,
        dynamicBoundsFallbacks: Number(geometryGuard.dynamicBoundsFallbacks) || 0,
        dynamicBounds: geometryGuard.dynamicBounds
          ? {{ x: Number(geometryGuard.dynamicBounds.x) || 0,
              y: Number(geometryGuard.dynamicBounds.y) || 0,
              width: Number(geometryGuard.dynamicBounds.width) || 0,
              height: Number(geometryGuard.dynamicBounds.height) || 0 }}
          : null,
        worldGeometryBounds: geometryGuard.worldBounds
          ? {{ x: Number(geometryGuard.worldBounds.x) || 0,
              y: Number(geometryGuard.worldBounds.y) || 0,
              width: Number(geometryGuard.worldBounds.width) || 0,
              height: Number(geometryGuard.worldBounds.height) || 0 }}
          : null,
        faceFeatureBounds: geometryGuard.faceFeatureBounds,
        faceAnchorPoints: geometryGuard.faceAnchorPoints,
        interactionLocked: Boolean(cursorState.locked),
        dragging: Boolean(clockState.dragging),
        localPointerActive: Boolean(clockState.localPointerActive),
        hostDragging: Boolean(clockState.hostDragging),
        resizePending: Boolean(resizeState.pending),
        resizePendingViewport: resizeState.pending
          ? [Number(resizeState.pendingWidth) || 0, Number(resizeState.pendingHeight) || 0]
          : null,
        resizeAppliedViewport: resizeState.applied
          ? [Number(resizeState.appliedWidth) || 0, Number(resizeState.appliedHeight) || 0]
          : null,
        resizeDeferredCount: Number(resizeState.deferredCount) || 0,
        resizeAppliedCount: Number(resizeState.appliedCount) || 0,
        feedbackPart: feedbackState.part,
        feedbackCount: feedbackState.count,
        feedbackAnchorX: Number(feedbackState.anchorX) || 0,
        feedbackAnchorY: Number(feedbackState.anchorY) || 0,
        feedbackActive: performance.now() < feedbackState.expiresAt,
        viewportStale: Math.abs(actualViewportWidth - expectedViewportWidth) > 0.5 ||
          Math.abs(actualViewportHeight - expectedViewportHeight) > 0.5
      }};
    }},
    inputMask: function() {{
      const options = arguments.length > 0 && arguments[0] &&
        typeof arguments[0] === 'object' ? arguments[0] : {{}};
      const canvas = document.getElementById('meapet-live2d');
      if (!canvas || !liveModel || typeof canvas.toDataURL !== 'function')
        return {{ status: 'unavailable' }};
      if (liveModel.visible === false) return {{ status: 'hidden' }};
      try {{
        const canvasRect = canvas.getBoundingClientRect();
        let rawAlphaNonempty = null;
        const renderer = pixiApp && pixiApp.renderer;
        const gl = renderer && renderer.gl;
        const rawWidth = Math.floor(Number(renderer && renderer.width) || canvas.width || 0);
        const rawHeight = Math.floor(Number(renderer && renderer.height) || canvas.height || 0);
        try {{
          if (options.probeAlpha === true && gl && rawWidth > 0 && rawHeight > 0 &&
              rawWidth * rawHeight <= 4194304) {{
            const pixels = new Uint8Array(rawWidth * rawHeight * 4);
            gl.readPixels(0, 0, rawWidth, rawHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
            for (let index = 3; index < pixels.length; index += 4) {{
              if (pixels[index] > 0) {{ rawAlphaNonempty = true; break; }}
            }}
          }}
        }} catch (error) {{}}
        const visualRects = [];
        ['meapet-bubble', 'meapet-interaction-toast'].forEach(function(id) {{
          const node = document.getElementById(id);
          if (!node || getComputedStyle(node).display === 'none') return;
          const rect = node.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) visualRects.push({{
            x: rect.left - canvasRect.left, y: rect.top - canvasRect.top,
            width: rect.width, height: rect.height
          }});
        }});
        document.querySelectorAll(
          '#meapet-hit-feedback .meapet-hit-marker'
        ).forEach(function(node) {{
          const rect = node.getBoundingClientRect();
          if (rect.width > 0 && rect.height > 0) visualRects.push({{
            x: rect.left - canvasRect.left, y: rect.top - canvasRect.top,
            width: rect.width, height: rect.height
          }});
        }});
        return {{
          status: 'ok', data: canvas.toDataURL('image/png'),
          raw_alpha_nonempty: rawAlphaNonempty, visual_rects: visualRects
        }};
      }} catch (error) {{
        return {{ status: 'unavailable' }};
      }}
    }},
    hitTest: function(x, y, options) {{
      // 调试/命中扫描可显式关闭 framebuffer alpha 读回；正式点击路径
      // 仍使用默认精确 alpha 校验。这样动作/高 DPI 压力下的部位探针
      // 不会阻塞渲染线程，也不改变公开 hitTest(x, y) 的旧契约。
      const details = modelHitDetails({{clientX: Number(x), clientY: Number(y)}},
        document.getElementById('meapet-live2d'),
        options && typeof options === 'object' ? options : {{}});
      details.drawableSummary = drawableDebugSummary();
      return details;
    }},
    setSpeech: function(text, mood, visible, speaking) {{
      const bubble = document.getElementById('meapet-bubble');
      if (!bubble) return;
      bubble.textContent = text || '';
      bubble.dataset.mood = mood || 'neutral';
      const shown = Boolean(visible && text);
      speechActive = Boolean(shown && speaking !== false);
      if (!shown) speechElapsed = 0;
      bubble.style.display = shown ? 'block' : 'none';
      const refreshScrollState = function() {{
        const scrollable = shown && bubble.scrollHeight > bubble.clientHeight + 1;
        bubble.classList.toggle('scrollable', scrollable);
      }};
      refreshScrollState();
      if (shown) window.requestAnimationFrame(refreshScrollState);
      if (!shown && motionSyncProfile) applyMotionSyncViseme('Silence', 0);
    }},
    setMouthViseme: function(name, intensity) {{
      return applyMotionSyncViseme(name, intensity);
    }},
    setInteractionFeedback: function(value, visible) {{
      const toast = document.getElementById('meapet-interaction-toast');
      if (!toast) return;
      clearPendingHitFeedback();
      toast.dataset.pending = 'false';
      toast.classList.remove('pending');
      const data = value && typeof value === 'object' ? value : {{}};
      const labels = {{
        head: '猫猫头', upper: '上半身', body: '身体',
        lower_left: '左腿', lower_right: '右腿'
      }};
      const zone = String(data.zone || '').trim().toLowerCase();
      const lines = [];
      if (labels[zone]) lines.push(labels[zone]);
      if (data.phrase) lines.push(String(data.phrase));
      if (data.expression || data.motion) {{
        const visual = [data.expression, data.motion].filter(Boolean).join(' · ');
        if (visual) lines.push(visual);
      }}
      const current = String(data.current || '').trim();
      const applied = String(data.applied || '').trim();
      if (current) lines.push(`好感度 ${{current}}${{applied ? `（本次 ${{applied}}）` : ''}}`);
      toast.textContent = lines.join('\\n');
      toast.dataset.zone = zone;
      toast.style.display = Boolean(visible && lines.length) ? 'block' : 'none';
    }}
  }};
  function setFrameRate(value) {{
    const normalized = Number(value);
    if (!Number.isFinite(normalized) || normalized < 15 || normalized > 120) return false;
    renderPerformanceState.targetFrameRate = normalized;
    renderPerformanceState.renderIntervalMs = 1000 / normalized;
    return true;
  }}
  function setGeometryAuditHz(value) {{
    const normalized = Number(value);
    if (!Number.isFinite(normalized) || normalized < 1 || normalized > 60) return false;
    renderPerformanceState.geometryAuditHz = normalized;
    renderPerformanceState.geometryAuditIntervalMs = 1000 / normalized;
    geometryAuditState.force = true;
    return true;
  }}
  function animationTick(timestamp) {{
    if (!liveModel) {{
      animationFrame = 0;
      return;
    }}
    const now = Number(timestamp) || performance.now();
    if (clockState.dragging) {{
      // 保持时间基线跟随真实 rAF，但不推进模型；释放时不会产生大 dt。
      lastFrameTime = now;
      const releaseAt = Number(clockState.dragReleaseUntil || 0);
      if (!clockState.localPointerActive && !clockState.hostDragging &&
          releaseAt > 0 && now >= releaseAt) {{
        // 页面本地 release timer 可能因后台节流/销毁没有执行；host
        // 标记已清除且没有活跃指针时，rAF 是独立的最终收敛路径，不能
        // 让动作和追踪永久停在 dragging 状态。
        clockState.dragging = false;
        clockState.dragReleaseUntil = 0;
        resetGeometryContinuity(1);
        if (resizeState.pending) flushDeferredResize();
      }}
      animationFrame = window.requestAnimationFrame(animationTick);
      return;
    }}
    const dt = lastFrameTime ? Math.min(0.1, Math.max(0, (now - lastFrameTime) / 1000)) : 0;
    lastFrameTime = now;
    if (window.meapetLive2D && window.meapetLive2D.advance) {{
      window.meapetLive2D.advance(dt, 'raf');
    }}
    animationFrame = window.requestAnimationFrame(animationTick);
  }}
  function startAnimationLoop() {{
    if (!animationFrame) animationFrame = window.requestAnimationFrame(animationTick);
  }}
  window.addEventListener('resize', resize);
  // QWidget 尺寸变化和 Chromium 的 document/layout 尺寸更新并非同一事件；
  // ResizeObserver 让模型在异步布局完成后再次以真实 clientWidth/clientHeight
  // 拟合，避免连续调整窗口时短暂沿用上一档比例。
  if (typeof ResizeObserver === 'function') {{
    const resizeHost = document.body || document.documentElement;
    if (resizeHost) {{
      const resizeObserver = new ResizeObserver(function() {{ resize(); }});
      resizeObserver.observe(resizeHost);
    }}
  }}
  if (window.qt && window.qt.webChannelTransport) {{
    try {{
      new QWebChannel(qt.webChannelTransport, function(channel) {{
        window.meapetBridge = channel.objects.meapetBridge;
        flushBridgeEvents();
      }});
    }} catch (error) {{
      reportError(error);
    }}
  }} else {{
    reportError(new Error('Qt WebChannel transport is unavailable'));
  }}
  loadModel().then(function() {{
    if (typeof refreshHoverHook === 'function') refreshHoverHook();
    reportModelReady();
    startAnimationLoop();
  }}).catch(reportError);
}})();
  </script></body></html>""".replace("__MODEL_URL__", model_json)
            .replace("__PART_NAMES__", part_names_json)
            .replace("__PART_LABELS__", part_labels_json)
            .replace("__DECLARED_EXPRESSIONS__", declared_expressions_json)
            .replace("__DECLARED_MOTIONS__", declared_motions_json)
            .replace("__MOTION_ALIASES__", motion_aliases_json)
            .replace("__MOTION_DURATIONS__", motion_durations_json)
            .replace("__MOTION_SYNC_PROFILE__", motion_sync_json)
            .replace("__FRAME_RATE__", frame_rate_json)
            .replace("__GEOMETRY_AUDIT_HZ__", geometry_audit_hz_json)
            .replace("__MAX_VIEWPORT_EDGE__", str(_WEB_LIVE2D_MAX_VIEWPORT_EDGE))
            .replace("__NATURAL_LAYOUT__", "true" if self._natural_layout else "false")
        )


__all__ = [
    "WebLive2DProbe",
    "WebLive2DRenderer",
    "probe_web_live2d",
    "qt_webengine_available",
]
