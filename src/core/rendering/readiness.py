"""Qt 宿主与控制面共用的渲染器就绪契约。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_BACKENDS = frozenset({"opengl", "web_live2d", "sprite", "vulkan", "unavailable"})


class RendererReadyState(StrEnum):
    """渲染器终态与非终态就绪状态。"""

    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


def _token(value: object, *, fallback: str = "") -> str:
    rendered = str(value or "").strip().casefold()
    return rendered if _TOKEN.fullmatch(rendered) else fallback


@dataclass(frozen=True, slots=True)
class RendererReadyStatus:
    """不包含资源内容与本地路径的渲染器就绪快照。

    契约不允许携带模型路径、URL、异常正文或平台对象标识。各后端只能通过
    固定布尔字段和长度受限的令牌暴露就绪证据。
    """

    backend: str
    state: RendererReadyState
    bridge_ready: bool = False
    model_ready: bool = False
    geometry_valid: bool = False
    frame_visible: bool = False
    alpha_nonempty: bool = False
    actual_api: str = ""
    reason_code: str = ""
    generation: int = 0

    def __post_init__(self) -> None:
        backend = _token(self.backend)
        if backend not in _BACKENDS:
            raise ValueError("renderer readiness backend is invalid")
        try:
            state = (
                self.state
                if isinstance(self.state, RendererReadyState)
                else RendererReadyState(self.state)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("renderer readiness state is invalid") from exc
        actual_api = _token(self.actual_api)
        reason_code = _token(self.reason_code)
        evidence = {
            "bridge_ready": self.bridge_ready,
            "model_ready": self.model_ready,
            "geometry_valid": self.geometry_valid,
            "frame_visible": self.frame_visible,
            "alpha_nonempty": self.alpha_nonempty,
        }
        if any(not isinstance(value, bool) for value in evidence.values()):
            raise ValueError("renderer readiness evidence must be boolean")
        if not isinstance(self.generation, int) or isinstance(self.generation, bool):
            raise ValueError("renderer readiness generation is invalid")
        if self.generation < 0:
            raise ValueError("renderer readiness generation is invalid")
        required = {
            "web_live2d": (
                "bridge_ready",
                "model_ready",
                "geometry_valid",
                "frame_visible",
                "alpha_nonempty",
            ),
            "opengl": ("model_ready", "geometry_valid", "frame_visible", "alpha_nonempty"),
            "sprite": ("model_ready", "geometry_valid", "frame_visible", "alpha_nonempty"),
            "vulkan": ("model_ready", "geometry_valid", "frame_visible"),
        }.get(backend, ())
        if state is RendererReadyState.READY:
            if not required or any(not evidence[name] for name in required):
                raise ValueError("renderer READY evidence is incomplete")
            if backend == "vulkan" and actual_api != "vulkan":
                raise ValueError("Vulkan READY requires the Vulkan graphics API")
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "actual_api", actual_api)
        object.__setattr__(self, "reason_code", reason_code)

    @property
    def ready(self) -> bool:
        return self.state is RendererReadyState.READY

    @property
    def terminal(self) -> bool:
        return self.state in {
            RendererReadyState.READY,
            RendererReadyState.FAILED,
            RendererReadyState.CLOSED,
        }

    def public(self) -> dict[str, object]:
        """返回重启回执使用的固定公开结构。"""

        return {
            "backend": self.backend,
            "state": self.state.value,
            "ready": self.ready,
            "terminal": self.terminal,
            "bridge_ready": bool(self.bridge_ready),
            "model_ready": bool(self.model_ready),
            "geometry_valid": bool(self.geometry_valid),
            "frame_visible": bool(self.frame_visible),
            "alpha_nonempty": bool(self.alpha_nonempty),
            "actual_api": self.actual_api,
            "reason_code": self.reason_code,
            "generation": self.generation,
        }


__all__ = ["RendererReadyState", "RendererReadyStatus"]
