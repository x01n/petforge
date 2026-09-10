from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from typing import Any


def _safe_public_text(value: object, *, limit: int = 240) -> str:
    """延迟导入 Web 投影，避免 ``gui.qt6`` 包初始化形成循环。"""

    from gui.web.control_surface import safe_public_text

    return safe_public_text(value, limit=limit)


def _public_control_state(value: Mapping[str, Any] | None) -> dict[str, object]:
    """延迟生成公开状态，保持 Web 控制面可独立导入。"""

    from gui.web.control_surface import public_control_state

    return public_control_state(value)


def _control_surface_html() -> str:
    """延迟构造页面，Qt 包加载阶段不反向导入自身。"""

    from gui.web.control_surface import control_surface_html

    return control_surface_html()


def _state_revision(value: Mapping[str, object] | None) -> int:
    """读取公开快照版本；无效版本按零处理。"""

    if not isinstance(value, Mapping):
        return 0
    raw = value.get("revision", 0)
    if isinstance(raw, bool):
        return 0
    try:
        revision = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, revision)


try:  # WebEngine 仅作为可选桌面能力，不影响无 GUI 核心。
    from PySide6.QtCore import QObject, Qt, QUrl, Signal, Slot
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    pyside6_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_available = False


if pyside6_available:
    _PUBLIC_INPUT_KEYS = frozenset(
        {
            "name",
            "part",
            "target",
            "mode",
            "text",
            "preset",
            "direction",
            "channel_id",
            "model",
            "backend",
            "expressions",
            "weight",
            "duration_seconds",
            "transition_seconds",
            "parameters",
            "loop",
        }
    )
    # 每类公开动作只接收其点击路径需要的字段；额外字段即使是标量也不
    # 进入宿主回调，避免把调试参数或配置片段伪装成公开路由载荷。
    _PUBLIC_ACTION_PAYLOAD_KEYS = {
        "open_config": frozenset(),
        "restart_application": frozenset(),
        "select_renderer_backend": frozenset({"backend"}),
        "configure_model": frozenset({"target", "mode"}),
        "select_model_channel": frozenset({"channel_id", "model"}),
        "submit_text": frozenset({"text"}),
        "pet_part": frozenset({"part"}),
        "expression": frozenset({"name"}),
        "motion": frozenset({"name"}),
        "expression_request": frozenset({"expressions", "mode", "loop"}),
        "motion_request": frozenset(
            {"name", "duration_seconds", "transition_seconds", "loop", "parameters"}
        ),
        "nudge_pet": frozenset({"direction"}),
        "set_display_size": frozenset({"preset"}),
    }
    _PUBLIC_ACTION_KINDS = frozenset(
        {
            "open_config",
            "restart_application",
            "select_renderer_backend",
            "configure_model",
            "select_model_channel",
            "submit_text",
            "stop",
            "retry",
            "approve",
            "grant_session",
            "deny",
            "pet_part",
            "expression",
            "motion",
            "expression_request",
            "motion_request",
            "show_pet",
            "toggle_visibility",
            "center_pet",
            "nudge_pet",
            "set_display_size",
            "toggle_window_lock",
            "toggle_always_on_top",
            "toggle_click_through",
            "restore_click_through",
            "read_foreground_window",
            "read_processes",
        }
    )
    _PUBLIC_CAPABILITY_NAME = re.compile(r"^cap-(?:expression|motion)-\d{1,2}$")
    _PUBLIC_PARAMETER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    _PUBLIC_PARTS = frozenset({"head", "body", "lower_left", "lower_right"})
    _PUBLIC_DIRECTIONS = frozenset({"left", "right", "up", "down"})
    _PUBLIC_PRESETS = frozenset({"small", "standard", "large"})
    _PUBLIC_TARGETS = frozenset({"model_setup"})
    _PUBLIC_MODES = frozenset({"click"})
    _PUBLIC_EXPRESSION_MODES = frozenset({"sequence", "blend"})
    _PUBLIC_RENDERER_BACKENDS = frozenset({"auto", "opengl", "vulkan", "web_live2d", "sprite"})

    def _bounded_number(
        value: object,
        *,
        minimum: float,
        maximum: float,
        default: float,
    ) -> float:
        """把公开动作数值限制在渲染契约允许范围内。"""

        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return default
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
            return default
        return parsed

    def _public_parameters(value: object) -> dict[str, float]:
        """过滤网页动作参数，只保留有限命名和有限数值。"""

        if not isinstance(value, Mapping) or len(value) > 32:
            return {}
        result: dict[str, float] = {}
        for raw_name, raw_value in value.items():
            name = str(raw_name or "").strip()
            if not _PUBLIC_PARAMETER_NAME.fullmatch(name) or isinstance(raw_value, bool):
                continue
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(parsed) and -10_000.0 <= parsed <= 10_000.0:
                result[name] = parsed
        return result

    def _public_capability_token(value: object, kind: str) -> str:
        """校验公开能力令牌并绑定到表情或动作类别。"""

        token = str(value or "").strip().lower()
        if _PUBLIC_CAPABILITY_NAME.fullmatch(token) and token.startswith(f"cap-{kind}-"):
            return token
        return ""

    _PUBLIC_RESULT_STATUSES = frozenset(
        {
            "idle",
            "requested",
            "completed",
            "available",
            "updated",
            "started",
            "accepted",
            "ok",
            "saved",
            "validated",
            "reloaded",
            "pending",
            "approval_required",
            "tool_limit",
            "degraded",
            "failed",
            "error",
            "unavailable",
            "cancelled",
            "canceled",
            "denied",
            "rejected",
            "timeout",
            "restart_required",
        }
    )
    _PUBLIC_RESULT_LABELS = {
        "idle": "等待处理",
        "requested": "已提交",
        "pending": "等待处理",
        "started": "已开始",
        "accepted": "已接受",
        "completed": "已完成",
        "available": "已就绪",
        "updated": "已更新",
        "saved": "已保存",
        "validated": "已校验",
        "reloaded": "已刷新",
        "approval_required": "等待确认",
        "tool_limit": "需要确认",
        "degraded": "降级运行",
        "failed": "执行失败",
        "error": "暂时不可用",
        "unavailable": "暂时不可用",
        "cancelled": "已停止",
        "canceled": "已停止",
        "denied": "已拒绝",
        "rejected": "已拒绝",
        "timeout": "响应超时",
        "restart_required": "需要重启",
    }

    def _safe_result(value: object) -> dict[str, object]:
        """过滤动作回执，避免审批编号、调用参数再次回到网页。"""

        if not isinstance(value, Mapping):
            return {"status": "completed", "message": "操作已提交"}
        result: dict[str, object] = {}
        raw_status = value.get("status", value.get("state", "completed"))
        raw_status = getattr(raw_status, "value", raw_status)
        status = _safe_public_text(raw_status, limit=32).strip().lower()
        result["status"] = status if status in _PUBLIC_RESULT_STATUSES else "unavailable"
        result["status_label"] = _PUBLIC_RESULT_LABELS.get(status, "暂时不可用")
        for key in ("reason", "detail", "message", "part"):
            item = value.get(key)
            if not isinstance(item, (str, int, float, bool)):
                continue
            if key in {"reason", "detail", "message"}:
                result[key] = _safe_public_text(item, limit=400)
            else:
                part = _safe_public_text(item, limit=80).strip().lower()
                if part in {"head", "body", "lower_left", "lower_right"}:
                    result[key] = part
        if "reason" in result and "message" not in result:
            result["message"] = result["reason"]
        return result

    class _ControlSurfaceBridge(QObject):
        """把页面动作限制在公开类型和有限载荷内。"""

        def __init__(
            self,
            state_provider: Callable[[], Mapping[str, Any] | None],
            action_callback: Callable[[str, Mapping[str, object]], object],
        ) -> None:
            super().__init__()
            self._state_provider = state_provider
            self._action_callback = action_callback

        @Slot(result=str)
        def getState(self) -> str:  # noqa: N802
            try:
                value = _public_control_state(self._state_provider())
                return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                return "{}"

        @Slot(str, str, result=str)
        def invoke(self, kind: str, payload: str = "{}") -> str:
            normalized = str(kind or "").strip().lower()[:64]
            if not normalized:
                return json.dumps({"status": "unavailable", "reason": "操作不可用"})
            if normalized not in _PUBLIC_ACTION_KINDS:
                return json.dumps({"status": "unavailable", "reason": "操作不可用"})
            try:
                value = json.loads(str(payload or "{}"))
            except (TypeError, ValueError):
                value = {}
            mapping = dict(value) if isinstance(value, Mapping) else {}
            # 页面只传当前动作需要的公开路由字段；宿主按当前交互状态解析审批动作。
            # 未登记的字段即使是标量也必须丢弃，避免调试参数进入回调。
            allowed_payload_keys = _PUBLIC_ACTION_PAYLOAD_KEYS.get(normalized, frozenset())
            safe_payload = {
                str(key): item
                for key, item in mapping.items()
                if str(key) in allowed_payload_keys
                and str(key) in _PUBLIC_INPUT_KEYS
                and isinstance(item, (str, int, float, bool))
            }
            if normalized == "select_renderer_backend":
                backend = str(mapping.get("backend", "") or "").strip().lower()
                safe_payload = {"backend": backend} if backend in _PUBLIC_RENDERER_BACKENDS else {}
            elif normalized == "expression_request":
                rows: list[dict[str, object]] = []
                raw_rows = mapping.get("expressions")
                if isinstance(raw_rows, list):
                    for raw_row in raw_rows[:8]:
                        if not isinstance(raw_row, Mapping):
                            continue
                        name = _public_capability_token(raw_row.get("name"), "expression")
                        if not name:
                            continue
                        rows.append(
                            {
                                "name": name,
                                "weight": _bounded_number(
                                    raw_row.get("weight", 1.0),
                                    minimum=0.0,
                                    maximum=1.0,
                                    default=1.0,
                                ),
                                "duration_seconds": _bounded_number(
                                    raw_row.get("duration_seconds", 1.5),
                                    minimum=0.05,
                                    maximum=120.0,
                                    default=1.5,
                                ),
                                "transition_seconds": _bounded_number(
                                    raw_row.get("transition_seconds", 0.2),
                                    minimum=0.0,
                                    maximum=10.0,
                                    default=0.2,
                                ),
                                "parameters": _public_parameters(raw_row.get("parameters")),
                            }
                        )
                mode = str(mapping.get("mode", "sequence") or "sequence").strip().lower()
                safe_payload = {
                    "expressions": rows,
                    "mode": mode if mode in _PUBLIC_EXPRESSION_MODES else "sequence",
                    "loop": (
                        mapping.get("loop", False)
                        if isinstance(mapping.get("loop", False), bool)
                        else False
                    ),
                }
            elif normalized == "motion_request":
                name = _public_capability_token(mapping.get("name"), "motion")
                safe_payload = {
                    "name": name,
                    "duration_seconds": _bounded_number(
                        mapping.get("duration_seconds", 1.5),
                        minimum=0.05,
                        maximum=300.0,
                        default=1.5,
                    ),
                    "transition_seconds": _bounded_number(
                        mapping.get("transition_seconds", 0.2),
                        minimum=0.0,
                        maximum=10.0,
                        default=0.2,
                    ),
                    "loop": (
                        mapping.get("loop", False)
                        if isinstance(mapping.get("loop", False), bool)
                        else False
                    ),
                    "parameters": _public_parameters(mapping.get("parameters")),
                }
            if "name" in safe_payload:
                name = str(safe_payload["name"]).strip().lower()
                capability_kind = (
                    "expression"
                    if normalized in {"expression", "expression_request"}
                    else "motion"
                    if normalized in {"motion", "motion_request"}
                    else ""
                )
                expected_prefix = f"cap-{capability_kind}-"
                if (
                    not capability_kind
                    or not _PUBLIC_CAPABILITY_NAME.fullmatch(name)
                    or not name.startswith(expected_prefix)
                ):
                    safe_payload.pop("name", None)
                else:
                    safe_payload["name"] = name
            if "part" in safe_payload:
                part = str(safe_payload["part"]).strip().lower()
                if part in _PUBLIC_PARTS:
                    safe_payload["part"] = part
                else:
                    safe_payload.pop("part", None)
            if "direction" in safe_payload:
                direction = str(safe_payload["direction"]).strip().lower()
                if direction in _PUBLIC_DIRECTIONS:
                    safe_payload["direction"] = direction
                else:
                    safe_payload.pop("direction", None)
            if "preset" in safe_payload:
                preset = str(safe_payload["preset"]).strip().lower()
                if preset in _PUBLIC_PRESETS:
                    safe_payload["preset"] = preset
                else:
                    safe_payload.pop("preset", None)
            if "target" in safe_payload:
                target = str(safe_payload["target"]).strip().lower()
                if target in _PUBLIC_TARGETS:
                    safe_payload["target"] = target
                else:
                    safe_payload.pop("target", None)
            if "mode" in safe_payload:
                mode = str(safe_payload["mode"]).strip().lower()
                allowed_modes = (
                    _PUBLIC_EXPRESSION_MODES
                    if normalized == "expression_request"
                    else _PUBLIC_MODES
                )
                if mode in allowed_modes:
                    safe_payload["mode"] = mode
                else:
                    safe_payload.pop("mode", None)
            if "text" in safe_payload:
                safe_payload["text"] = str(safe_payload["text"])[:2000]
            if "channel_id" in safe_payload:
                channel_id = str(safe_payload["channel_id"]).strip()
                if (
                    not channel_id
                    or len(channel_id) > 128
                    or any(char.isspace() or ord(char) < 0x20 for char in channel_id)
                ):
                    safe_payload.pop("channel_id", None)
                else:
                    safe_payload["channel_id"] = channel_id
            if "model" in safe_payload:
                model = str(safe_payload["model"]).strip()
                if (
                    not model
                    or len(model) > 128
                    or any(char.isspace() or ord(char) < 0x20 for char in model)
                ):
                    safe_payload.pop("model", None)
                else:
                    safe_payload["model"] = model
            try:
                result = self._action_callback(normalized, safe_payload)
            except Exception:
                result = {"status": "unavailable", "reason": "操作暂时不可用"}
            safe_result = _safe_result(result)
            # 页面只需要可读回执；审批编号、回合编号和 operation_id 均不返回。
            return json.dumps(safe_result, ensure_ascii=False, default=str)

    class WebControlSurfaceWindow(QWidget):
        """显示公开状态和点击动作的独立网页控制台。"""

        hidden = Signal()
        closeRequested = Signal()

        def __init__(
            self,
            state_provider: Callable[[], Mapping[str, Any] | None],
            action_callback: Callable[[str, Mapping[str, object]], object],
            *,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.setWindowTitle("MeaPet 网页控制台")
            self.setWindowFlags(
                Qt.WindowType.Window | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
            )
            self.resize(720, 760)
            self._state_provider = state_provider
            self._page_ready = False
            self._shutdown = False
            self._pending_state: dict[str, object] | None = None
            self._last_state_revision = -1
            self._channel = QWebChannel(self)
            self._bridge = _ControlSurfaceBridge(state_provider, action_callback)
            self._channel.registerObject("meapetControlSurfaceBridge", self._bridge)
            self._view = QWebEngineView(self)
            self._view.page().setWebChannel(self._channel)
            self._view.page().lifecycleStateChanged.connect(self._on_lifecycle_state_changed)
            self._view.setHtml(_control_surface_html(), QUrl("about:blank"))
            self._view.loadFinished.connect(self._on_load_finished)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._view)

        @property
        def view(self) -> QWebEngineView:
            return self._view

        @property
        def page_ready(self) -> bool:
            return self._page_ready

        def _on_load_finished(self, ok: bool) -> None:
            self._page_ready = bool(ok)
            if not self._page_ready:
                return
            self._run_state(self._pending_state or _public_control_state(self._state_provider()))
            self._pending_state = None

        def _on_lifecycle_state_changed(self, state: QWebEnginePage.LifecycleState) -> None:
            """停止向已被 Qt 丢弃的页面投递 JavaScript。"""

            if state == QWebEnginePage.LifecycleState.Discarded:
                self._page_ready = False
                self._pending_state = None

        def _page_is_discarded(self) -> bool:
            try:
                return self._view.page().lifecycleState() == QWebEnginePage.LifecycleState.Discarded
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return True

        def _accept_state(self, value: Mapping[str, object]) -> bool:
            """拒绝晚到的旧快照，避免状态回退。"""

            revision = _state_revision(value)
            if revision < self._last_state_revision:
                return False
            self._last_state_revision = revision
            return True

        def _run_state(self, state: Mapping[str, object]) -> None:
            if self._shutdown or not self._page_ready or self._page_is_discarded():
                self._page_ready = False
                return
            if not self._accept_state(state):
                return
            payload = json.dumps(
                _public_control_state(state), ensure_ascii=False, separators=(",", ":")
            )
            try:
                self._view.page().runJavaScript(
                    "window.meapetControlSurface && "
                    f"window.meapetControlSurface.setState({payload});"
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                self._page_ready = False

        def set_state(self, state: Mapping[str, Any] | None = None) -> None:
            if self._shutdown:
                return
            value = _public_control_state(state if state is not None else self._state_provider())
            if not self._page_ready:
                if self._pending_state is None or self._accept_state(value):
                    self._pending_state = value
                return
            self._run_state(value)

        def show_and_focus(self) -> None:
            # 恢复被用户最小化的控制台；仅调用 show() 时 Qt 可能保留
            # WindowMinimized 状态，导致救援入口看似已打开但仍不可见。
            self.showNormal()
            self.raise_()
            self.activateWindow()

        def hideEvent(self, event: object) -> None:  # noqa: N802
            super().hideEvent(event)
            if not self._shutdown:
                self.hidden.emit()

        def closeEvent(self, event: object) -> None:  # noqa: N802
            if self._shutdown:
                event.accept()
                return
            event.ignore()
            self.closeRequested.emit()
            self.hide()

        def shutdown(self) -> None:
            if self._shutdown:
                return
            self._shutdown = True
            self._page_ready = False
            self._pending_state = None
            try:
                self._view.page().setWebChannel(None)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            self._view.close()
            self.close()


else:

    class WebControlSurfaceWindow:  # pragma: no cover
        """无 WebEngine 环境下的明确不可用占位。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 WebEngine is not installed")


__all__ = ["WebControlSurfaceWindow", "pyside6_available"]
