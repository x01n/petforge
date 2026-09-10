from __future__ import annotations

import math
import re
from pathlib import Path

from core.rendering.actions import ExpressionRequest, ExpressionTimeline, MotionRequest
from core.rendering.resources import is_valid_webp

from .protocol import (
    DEFAULT_EXPRESSION_NAMES,
    DEFAULT_MOTION_NAMES,
    RendererCapabilities,
    RendererState,
)

_SPRITE_NAME = re.compile(
    r"^mea(?P<outfit>[0-9]{2})(?P<direction>[A-E])_"
    r"(?P<code>[0-9]{3})(?P<variant>a)?\.webp$"
)

DEFAULT_OUTFIT = "01"
DEFAULT_DIRECTION = "A"
DEFAULT_EXPRESSION_CODE = "001"
BLINK_CODES = ("011", "012")

# 这些语义名称对应资源中实际存在的差分编号。映射是固定的，不能在每一帧
# 或每次调用时随机选择，否则模型会表现为表情持续抖动。011/012 是闭眼
# 眨眼帧，只能由 blink 动作使用，不能再误当作 happy/content 表情。
EXPRESSION_CODES = {
    "default": "001",
    "neutral": "001",
    "happy": "182",
    "talking": "192",
    "sad": "701",
    "curious": "191",
    "surprised": "192",
    "shy": "171",
    "melancholy": "702",
    "content": "181",
    "peaceful": "181",
    "innocent": "102",
    "teary": "302",
    "intrigued": "191",
    "gentle": "601",
    "embarrassed": "171",
    "annoyed": "611",
    "wistful": "701",
    "pensive": "702",
}

# 旧实现只明确把 011/012 定义为闭眼差分，没有为其它编号提供动作
# 语义。因此 wave/walk 只记录动作状态，精灵回退保持当前表情；Live2D
# 后端仍可执行模型自身的动作。这样不会把未定义的编号误当作动作帧。
MOTION_CODES = {
    "idle": (),
    "blink": BLINK_CODES,
    "wave": (),
    "walk": (),
}


class SpriteRenderer:
    """管理固定服装/方向下的表情差分和显式动作。"""

    def __init__(
        self,
        sprite_dir: str | Path,
        *,
        frame_interval: float = 0.12,
        outfit: str = DEFAULT_OUTFIT,
        direction: str = DEFAULT_DIRECTION,
    ) -> None:
        self.sprite_dir = Path(sprite_dir).expanduser().resolve()
        self.frames: tuple[Path, ...] = ()
        self.frame_interval = max(0.02, float(frame_interval))
        self._groups: dict[tuple[str, str, str], Path] = {}
        self._unstructured_frames: tuple[Path, ...] = ()
        self._index_frames()

        # 方向/服装会被自由行动控制器在运行时切换。使用属性而不是让调用方
        # 直接改内部字段，切换时可以清理上一组动作帧，避免动作从旧朝向跳到
        # 新朝向的错误帧。
        self._outfit = self._normalize_outfit(outfit)
        self._direction = self._normalize_direction(direction)
        self._expression_code = DEFAULT_EXPRESSION_CODE
        self._motion_sequence: tuple[Path, ...] = ()
        self._motion_index = 0
        self._motion_active = False
        self._motion_fallback = False
        self._dragging = False
        self._expression_timeline = ExpressionTimeline()
        self._previous_expression_frame: Path | None = None
        self._expression_transition_progress = 1.0
        self._motion_request: MotionRequest | None = None
        self._motion_request_elapsed = 0.0
        self._state = RendererState(expression="neutral")
        self._capabilities = RendererCapabilities(
            backend="sprite",
            available=bool(self.frames),
            expressions=DEFAULT_EXPRESSION_NAMES,
            motions=DEFAULT_MOTION_NAMES,
            message="WebP sprite fallback" if self.frames else "no valid WebP sprite frame",
        )

    def _index_frames(self) -> None:
        """建立差分索引，并为测试用的小型无命名资源保留顺序回退。"""

        self.frames, self._groups, self._unstructured_frames = self._scan_frames(self.sprite_dir)

    @staticmethod
    def _scan_frames(
        sprite_dir: Path,
    ) -> tuple[tuple[Path, ...], dict[tuple[str, str, str], Path], tuple[Path, ...]]:
        """完整验证候选目录，在调用方提交前不修改当前渲染状态。"""

        frames = tuple(path for path in sorted(sprite_dir.glob("*.webp")) if is_valid_webp(path))
        groups: dict[tuple[str, str, str], Path] = {}
        unstructured: list[Path] = []
        for path in frames:
            match = _SPRITE_NAME.fullmatch(path.name)
            if match is None:
                unstructured.append(path)
                continue
            key = (match["outfit"], match["direction"], match["code"])
            # 排序后先遇到无 a 后缀的文件；重复变体不参与动画。
            groups.setdefault(key, path)
        return frames, groups, tuple(unstructured)

    def _normalize_outfit(self, value: str) -> str:
        normalized = str(value or DEFAULT_OUTFIT).strip()
        available = {key[0] for key in self._groups}
        if normalized in available:
            return normalized
        return DEFAULT_OUTFIT if DEFAULT_OUTFIT in available else normalized

    def _normalize_direction(self, value: str) -> str:
        normalized = str(value or DEFAULT_DIRECTION).strip().upper()
        available = {key[1] for key in self._groups if key[0] == self.outfit}
        if normalized in available:
            return normalized
        return DEFAULT_DIRECTION if DEFAULT_DIRECTION in available else normalized

    def _available_expression_names(self) -> tuple[str, ...]:
        """返回当前服装/朝向确实有对应帧的语义表情。"""

        if not self.frames or not self._groups:
            return ()
        return tuple(
            name
            for name in DEFAULT_EXPRESSION_NAMES
            if (code := EXPRESSION_CODES.get(name)) is not None
            and self._path_for_code(code) is not None
        )

    def _available_motion_names(self) -> tuple[str, ...]:
        """返回当前精灵能执行的动作；无帧动作不会伪报成功。"""

        if not self.frames:
            return ()
        names: list[str] = ["idle"]
        if self._motion_paths("blink"):
            names.append("blink")
        # wave/walk 的回退是显式的统一平移补偿，不依赖不存在的图片帧。
        names.extend(("wave", "walk"))
        return tuple(names)

    def _refresh_capabilities(self) -> None:
        """在资源组切换后同步能力快照。"""

        self._capabilities = RendererCapabilities(
            backend="sprite",
            available=bool(self.frames),
            expressions=self._available_expression_names(),
            motions=self._available_motion_names(),
            message="WebP sprite fallback" if self.frames else "no valid WebP sprite frame",
        )

    @property
    def capabilities(self) -> RendererCapabilities:
        self._refresh_capabilities()
        return self._capabilities

    @property
    def outfit(self) -> str:
        """返回当前服装组。"""

        return self._outfit

    @outfit.setter
    def outfit(self, value: str) -> None:
        normalized = self._normalize_outfit(value)
        if getattr(self, "_outfit", None) == normalized:
            return
        self._outfit = normalized
        if hasattr(self, "_direction"):
            self._direction = self._normalize_direction(self._direction)
        self._sync_expression_after_geometry_change()
        self._reset_motion_after_geometry_change()

    @property
    def direction(self) -> str:
        """返回当前朝向组。"""

        return self._direction

    @direction.setter
    def direction(self, value: str) -> None:
        normalized = self._normalize_direction(value)
        if getattr(self, "_direction", None) == normalized:
            return
        self._direction = normalized
        self._sync_expression_after_geometry_change()
        self._reset_motion_after_geometry_change()

    def _sync_expression_after_geometry_change(self) -> None:
        """朝向/服装缺少旧差分时回到真实存在的中性帧。"""

        if not hasattr(self, "_expression_code"):
            return
        if self._groups and self._path_for_code(self._expression_code) is not None:
            return
        if self._groups and self._path_for_code(DEFAULT_EXPRESSION_CODE) is not None:
            self._expression_code = DEFAULT_EXPRESSION_CODE
            self._state.expression = "neutral"

    def set_direction(self, direction: str) -> bool:
        """设置可用朝向；非法值不会悄悄替换当前帧。"""

        normalized = str(direction or "").strip().upper()
        available = {key[1] for key in self._groups if key[0] == self.outfit}
        if normalized not in available:
            return False
        self.direction = normalized
        return True

    def supports_expression(self, name: str) -> bool:
        """判断精灵是否有可绘制的目标表情帧。"""

        if not self.frames:
            return False
        normalized = str(name or "").strip().lower()
        code = EXPRESSION_CODES.get(normalized, normalized)
        if not re.fullmatch(r"[0-9]{3}", code):
            return False
        return bool(self._groups) and self._path_for_code(code) is not None

    def supports_motion(self, name: str) -> bool:
        """判断精灵是否能执行目标动作。"""

        normalized = str(name or "").strip().lower()
        return normalized in self._available_motion_names()

    @property
    def state(self) -> RendererState:
        return self._state

    @property
    def visual_transform(self) -> tuple[float, float]:
        """返回无动作帧时的安全视觉补偿 ``(x, y)``（逻辑像素）。

        旧项目的精灵资源只有表情/眨眼差分，没有 ``wave``/``walk`` 的
        独立图片。为了让工具调用仍然有可见反馈，这两类动作使用一个很小
        的平移呼吸补偿；不改变图片宽高，也不会引入拉伸。真实动作帧存在时
        返回零偏移。
        """

        if not self._motion_active or self._motion_sequence or not self._motion_fallback:
            return 0.0, 0.0
        duration = 0.72
        phase = min(1.0, max(0.0, self._state.elapsed / duration))
        angle = math.pi * phase
        if self._state.motion == "walk":
            return 6.0 * math.sin(angle * 2.0), -4.0 * abs(math.sin(angle))
        return 4.0 * math.sin(angle), -4.0 * math.sin(angle)

    @property
    def expression_code(self) -> str:
        """返回当前表情对应的三位差分编号。"""

        return self._expression_code

    @property
    def current_frame(self) -> Path | None:
        """返回当前应该绘制的单帧，不遍历无关资源。"""

        if not self.frames:
            return None
        if self._motion_active and self._motion_sequence:
            return self._motion_sequence[self._motion_index]
        return self._base_path() or (
            self._unstructured_frames[0] if self._unstructured_frames else self.frames[0]
        )

    @property
    def previous_frame(self) -> Path | None:
        """返回表情过渡前一帧；过渡结束后为空。"""

        return self._previous_expression_frame

    @property
    def expression_transition_progress(self) -> float:
        """返回当前表情交叉淡化进度。"""

        return max(0.0, min(1.0, float(self._expression_transition_progress)))

    def _reset_motion_after_geometry_change(self) -> None:
        """朝向/服装变化后清理旧路径，保持当前表情和稳定锚点。"""

        if not hasattr(self, "_motion_sequence"):
            return
        self._motion_sequence = ()
        self._motion_index = 0
        self._motion_active = False
        self._motion_fallback = False
        self._state.motion = "idle"
        self._state.frame_index = 0
        self._state.elapsed = 0.0

    def _index_key(self, code: str) -> tuple[str, str, str]:
        return self.outfit, self.direction, code

    def _path_for_code(self, code: str) -> Path | None:
        return self._groups.get(self._index_key(code))

    def _expression_path(self) -> Path | None:
        return self._path_for_code(self._expression_code)

    def _base_path(self) -> Path | None:
        return self._expression_path() or self._path_for_code(DEFAULT_EXPRESSION_CODE)

    def _motion_paths(self, name: str) -> tuple[Path, ...]:
        return tuple(
            path for code in MOTION_CODES[name] if (path := self._path_for_code(code)) is not None
        )

    def _motion_request_duration(self, request: MotionRequest) -> float | None:
        """返回显式时长或当前精灵动作自身的完整播放时长。"""

        if request.duration_seconds is not None:
            return request.duration_seconds
        paths = self._motion_paths(request.name)
        if paths:
            return len(paths) * self.frame_interval
        if request.name.casefold() in {"wave", "walk"}:
            return 0.72
        return None

    def advance(self, elapsed_seconds: float) -> None:
        """推进显式动作；空闲状态始终停在当前表情。"""

        if self._dragging:
            return
        try:
            elapsed = float(elapsed_seconds)
        except (TypeError, ValueError, OverflowError):
            elapsed = 0.0
        if not math.isfinite(elapsed):
            elapsed = 0.0
        elapsed = max(0.0, elapsed)
        expression_frame = self._expression_timeline.advance(elapsed)
        if expression_frame is not None:
            if expression_frame.changed:
                previous = self.current_frame
                self._set_expression_name(expression_frame.current)
                current = self.current_frame
                self._previous_expression_frame = previous if previous != current else None
            self._expression_transition_progress = expression_frame.progress
            if expression_frame.finished or expression_frame.progress >= 1.0:
                self._previous_expression_frame = None
        motion_request = self._motion_request
        animation_elapsed = elapsed
        if motion_request is not None:
            self._motion_request_elapsed += elapsed
            duration = self._motion_request_duration(motion_request)
            if duration is not None and self._motion_request_elapsed >= duration:
                if motion_request.loop:
                    self._motion_request_elapsed %= duration
                    self._play_motion_name(motion_request.name)
                    animation_elapsed = self._motion_request_elapsed
                else:
                    self._motion_request = None
                    self._motion_request_elapsed = 0.0
                    self._play_motion_name("idle")
                    animation_elapsed = 0.0
        self._state.elapsed += animation_elapsed
        if not self._motion_active or not self._motion_sequence:
            if self._motion_active and self._motion_fallback:
                # 无独立动作帧的兼容动作只持续一个短周期，之后回到当前表情。
                if self._state.elapsed >= 0.72:
                    self._motion_active = False
                    self._motion_fallback = False
                    self._state.motion = "idle"
                    self._state.elapsed = 0.0
            return
        while self._state.elapsed >= self.frame_interval and self._motion_active:
            self._state.elapsed -= self.frame_interval
            if self._motion_index + 1 < len(self._motion_sequence):
                self._motion_index += 1
                self._state.frame_index = self._motion_index
            else:
                # 动作只播放一次；结束后回到明确设置的表情。
                self._motion_active = False
                self._motion_sequence = ()
                self._motion_index = 0
                self._state.frame_index = 0
                self._state.motion = "idle"

    def set_dragging(self, dragging: bool) -> bool:
        """冻结/恢复精灵动作，保持拖动期间的帧和锚点稳定。"""

        self._dragging = bool(dragging)
        return True

    def reload_resources(
        self,
        resource_root: str | Path,
        *,
        model_path: str | Path | None = None,
        sprite_scale: float | None = None,
    ) -> dict[str, object]:
        """原子重扫精灵目录；失败时保留旧帧和动作状态。"""

        del model_path
        if self._dragging:
            return {"status": "busy", "reason": "resource reload is blocked while dragging"}
        scale: float | None = None
        if sprite_scale is not None:
            try:
                scale = float(sprite_scale)
            except (TypeError, ValueError, OverflowError):
                return {"status": "unavailable", "reason": "sprite_scale is invalid"}
            if not math.isfinite(scale) or scale < 0.1 or scale > 2.0:
                return {"status": "unavailable", "reason": "sprite_scale is outside bounds"}
        try:
            requested_root = Path(resource_root).expanduser().resolve()
            nested_sprite_dir = requested_root / "sprites"
            target = nested_sprite_dir if nested_sprite_dir.is_dir() else requested_root
            frames, groups, unstructured = self._scan_frames(target)
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "sprite resource scan failed"}
        if not frames:
            return {"status": "unavailable", "reason": "no valid WebP sprite frame"}

        # 所有文件校验完成后一次性提交新索引；同目录也必须提交，以便宿主
        # 清除已经解码的 QImage 并读取原地替换后的文件内容。
        self.sprite_dir = target
        self.frames = frames
        self._groups = groups
        self._unstructured_frames = unstructured
        self._outfit = self._normalize_outfit(self._outfit)
        self._direction = self._normalize_direction(self._direction)
        self._expression_timeline.cancel()
        self._previous_expression_frame = None
        self._expression_transition_progress = 1.0
        self._motion_request = None
        self._motion_request_elapsed = 0.0
        self._sync_expression_after_geometry_change()
        self._reset_motion_after_geometry_change()
        self._refresh_capabilities()
        return {
            "status": "reloaded",
            "resource_root": requested_root,
            "sprite_dir": target,
            "frame_count": len(frames),
            "sprite_scale": scale,
        }

    def _set_expression_name(self, name: str) -> bool:
        """设置语义表情或资源中的三位差分编号。"""

        normalized = str(name or "").strip().lower()
        code = EXPRESSION_CODES.get(normalized, normalized)
        if not re.fullmatch(r"[0-9]{3}", code):
            return False
        if not self.supports_expression(normalized):
            return False

        self._expression_code = code
        self._state.expression = normalized if normalized in EXPRESSION_CODES else code
        self._state.motion = "idle"
        self._state.frame_index = 0
        self._state.elapsed = 0.0
        self._motion_sequence = ()
        self._motion_index = 0
        self._motion_active = False
        self._motion_fallback = False
        return True

    def set_expression(self, name: str) -> bool:
        """立即设置单一表情，并取消旧表情序列。"""

        self._expression_timeline.cancel()
        self._previous_expression_frame = None
        self._expression_transition_progress = 1.0
        return self._set_expression_name(name)

    def set_expression_request(self, request: ExpressionRequest) -> dict[str, object]:
        """启动带持续时间的表情序列；Sprite 使用帧交叉淡化。"""

        if not isinstance(request, ExpressionRequest):
            return {"status": "unavailable", "reason": "expression request is invalid"}
        if any(not self.supports_expression(item.name) for item in request.expressions):
            return {"status": "unavailable", "reason": "one or more expressions are unsupported"}
        if not self.supports_expression(request.restore):
            return {"status": "unavailable", "reason": "restore expression is unsupported"}
        previous = self.current_frame
        frame = self._expression_timeline.start(request, current=self._state.expression)
        if not self._set_expression_name(frame.current):
            self._expression_timeline.cancel()
            return {"status": "unavailable", "reason": "expression could not start"}
        current = self.current_frame
        self._previous_expression_frame = previous if previous != current else None
        self._expression_transition_progress = frame.progress
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
            "parameters_applied": False,
        }

    def _play_motion_name(self, name: str) -> bool:
        """播放一次已知动作；缺少动作差分时保持当前表情。"""

        normalized = str(name or "").strip().lower()
        if not self.supports_motion(normalized):
            return False
        self._state.motion = normalized
        self._state.elapsed = 0.0
        self._motion_index = 0
        paths = self._motion_paths(normalized)
        if normalized == "idle" or not paths:
            self._motion_sequence = ()
            self._motion_fallback = normalized != "idle"
            self._motion_active = self._motion_fallback
            self._state.frame_index = 0
            return True
        self._motion_sequence = paths
        self._motion_active = True
        self._motion_fallback = False
        self._state.frame_index = 0
        return True

    def play_motion(self, name: str) -> bool:
        """立即播放动作，并取消旧的持续时间请求。"""

        self._motion_request = None
        self._motion_request_elapsed = 0.0
        return self._play_motion_name(name)

    def play_motion_request(self, request: MotionRequest) -> dict[str, object]:
        """播放带持续时间和循环策略的 Sprite 动作。"""

        if not isinstance(request, MotionRequest) or not self.supports_motion(request.name):
            return {"status": "unavailable", "reason": "motion request is unsupported"}
        if not self._play_motion_name(request.name):
            return {"status": "unavailable", "reason": "motion could not start"}
        self._motion_request = request
        self._motion_request_elapsed = 0.0
        return {
            "status": "started",
            "name": request.name,
            "duration_seconds": request.duration_seconds,
            "transition_seconds": request.transition_seconds,
            "loop": request.loop,
            "parameters_applied": False,
        }

    def shutdown(self) -> None:
        self._expression_timeline.cancel()
        self._motion_request = None
        self._motion_request_elapsed = 0.0
        self._dragging = False
        self._motion_sequence = ()
        self._motion_index = 0
        self._motion_active = False
        self._motion_fallback = False
        self._previous_expression_frame = None
        self._expression_transition_progress = 1.0
        self._state.expression = "neutral"
        self._state.motion = "idle"
        self._state.elapsed = 0.0
        self._state.frame_index = 0
