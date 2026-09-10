"""模型主动行为与随机行为的隔离调度。"""

from __future__ import annotations

import asyncio
import inspect
import math
import random
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from services.tools.types import BEHAVIOR_INTERNAL_METADATA_KEY, ToolCallContext

_FAILURE_STATUSES = frozenset(
    {
        "denied",
        "error",
        "failed",
        "unavailable",
        "approval_required",
        "duplicate",
        "cancelled",
        "timeout",
        "false",
        "none",
        "null",
    }
)
_MOVEMENT_INTERACTION_POLL_SECONDS = 0.1
# 随机 motion 不能无限占用正式 Cubism 动作管理器。部分 model3 的动作组
# 明确声明 ``Loop: true``，因此成功提交后必须经过一个短暂、可取消的展示
# 窗口再回到 idle；否则一次随机触发会让模型持续循环，看起来像行为失控。
_DEFAULT_ACTION_DURATION_SECONDS = 0.8
_MAX_ACTION_DURATION_SECONDS = 30.0
# Qt 宿主退出后不再处理主线程 dispatcher 事件；动作收尾若无限等待
# ``play_motion`` 会阻塞 RuntimeLoop.stop。正常运行时该窗口足够完成一次
# idle 回执，超时只用于关闭/桌面线程已停止的边界。
_MOTION_RESET_TIMEOUT_SECONDS = 0.5


@dataclass(frozen=True)
class BehaviorAction:
    identity: str
    arguments: Mapping[str, Any]
    weight: float = 1.0
    duration_seconds: float = _DEFAULT_ACTION_DURATION_SECONDS

    def __post_init__(self) -> None:
        """把随机动作保持时长规整到有限范围。"""

        try:
            duration = float(self.duration_seconds)
        except (TypeError, ValueError, OverflowError):
            duration = _DEFAULT_ACTION_DURATION_SECONDS
        if not math.isfinite(duration):
            duration = _DEFAULT_ACTION_DURATION_SECONDS
        object.__setattr__(
            self,
            "duration_seconds",
            max(0.0, min(_MAX_ACTION_DURATION_SECONDS, duration)),
        )


class BehaviorPhase(StrEnum):
    """自主行为状态机的可观察阶段。"""

    STOPPED = "stopped"
    IDLE = "idle"
    PLANNING = "planning"
    MOVING = "moving"
    ACTING = "acting"
    PAUSED = "paused"


@dataclass(frozen=True)
class AutonomousMovementConfig:
    """自主游走参数。

    ``bounds`` 不在服务内缓存。每次规划前都重新读取宿主窗口的可用区域，
    因此多屏、分辨率变化和窗口尺寸变化不会把桌宠带出屏幕。
    """

    enabled: bool = False
    min_distance_px: float = 80.0
    max_distance_px: float = 280.0
    step_pixels: float = 36.0
    step_interval_seconds: float = 0.08
    moving_motion: str = "walk"
    idle_motion: str = "idle"
    movement_identity: str = "pet:autonomous_move"
    flip_facing: bool = True
    max_steps: int = 64
    max_speed_px_per_second: float = 240.0

    def __post_init__(self) -> None:
        try:
            min_distance = float(self.min_distance_px)
        except (TypeError, ValueError, OverflowError):
            min_distance = 80.0
        if not math.isfinite(min_distance):
            min_distance = 80.0
        min_distance = max(1.0, min_distance)
        try:
            max_distance = float(self.max_distance_px)
        except (TypeError, ValueError, OverflowError):
            max_distance = 280.0
        if not math.isfinite(max_distance):
            max_distance = 280.0
        max_distance = max(min_distance, max_distance)
        try:
            step_pixels = float(self.step_pixels)
        except (TypeError, ValueError, OverflowError):
            step_pixels = 36.0
        if not math.isfinite(step_pixels):
            step_pixels = 36.0
        step_pixels = max(4.0, step_pixels)
        try:
            interval = float(self.step_interval_seconds)
        except (TypeError, ValueError, OverflowError):
            interval = 0.08
        if not math.isfinite(interval):
            interval = 0.08
        interval = max(0.02, min(1.0, interval))
        try:
            max_speed = float(self.max_speed_px_per_second)
        except (TypeError, ValueError, OverflowError):
            max_speed = 240.0
        if not math.isfinite(max_speed):
            max_speed = 240.0
        max_speed = max(20.0, min(1200.0, max_speed))
        try:
            max_steps = int(self.max_steps)
        except (TypeError, ValueError, OverflowError):
            max_steps = 64
        max_steps = max(1, min(64, max_steps))
        object.__setattr__(self, "enabled", bool(self.enabled))
        object.__setattr__(self, "min_distance_px", min_distance)
        object.__setattr__(self, "max_distance_px", max_distance)
        object.__setattr__(self, "step_pixels", step_pixels)
        object.__setattr__(self, "step_interval_seconds", interval)
        object.__setattr__(self, "max_speed_px_per_second", max_speed)
        object.__setattr__(self, "moving_motion", str(self.moving_motion or "").strip())
        object.__setattr__(self, "idle_motion", str(self.idle_motion or "").strip())
        object.__setattr__(
            self,
            "movement_identity",
            str(self.movement_identity or "pet:autonomous_move").strip(),
        )
        object.__setattr__(self, "flip_facing", bool(self.flip_facing))
        object.__setattr__(self, "max_steps", max_steps)


class BehaviorService:
    """随机策略只能通过 ToolExecutionService 调用工具，不能直达 GUI。

    自主游走与模型行为共享同一权限管线，但游走使用隐藏的低风险
    ``pet:autonomous_move`` 工具。模型仍使用 ``pet:move``，不会因为开启
    自主行为而绕过模型工具的审批策略。
    """

    def __init__(
        self,
        tool_executor: object,
        *,
        actions: tuple[BehaviorAction, ...] = (),
        interval_seconds: tuple[float, float] = (8.0, 20.0),
        probability: float = 0.35,
        random_source: random.Random | None = None,
        context_factory: Callable[[], ToolCallContext] | None = None,
        movement: AutonomousMovementConfig | None = None,
        position_provider: Callable[[], object] | None = None,
        bounds_provider: Callable[[], object] | None = None,
        interaction_provider: Callable[[], object] | None = None,
        motion_generation_provider: Callable[[], object] | None = None,
        direction_setter: Callable[[str], object] | None = None,
        interaction_cooldown_seconds: float = 0.35,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._executor = tool_executor
        # 配置通常已由运行时校验，但服务也可能被无配置文件的宿主直接
        # 构造。过滤非有限/非正权重，避免 NaN 让随机游标永远选中末项，
        # 或让无效动作在后台循环中反复提交。
        valid_actions: list[BehaviorAction] = []
        for action in actions:
            try:
                weight = float(action.weight)
            except (AttributeError, TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(weight) and weight > 0:
                try:
                    valid_actions.append(
                        BehaviorAction(
                            action.identity,
                            action.arguments,
                            weight,
                            getattr(action, "duration_seconds", _DEFAULT_ACTION_DURATION_SECONDS),
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    continue
        self._actions = tuple(valid_actions)
        try:
            low, high = interval_seconds
        except (TypeError, ValueError):
            low, high = 8.0, 20.0
        try:
            low_value = float(low)
            high_value = float(high)
        except (TypeError, ValueError, OverflowError):
            low_value, high_value = 8.0, 20.0
        if not math.isfinite(low_value):
            low_value = 8.0
        if not math.isfinite(high_value):
            high_value = 20.0
        low_value = max(0.5, low_value)
        self._interval = (low_value, max(low_value, high_value))
        try:
            probability_value = float(probability)
        except (TypeError, ValueError, OverflowError):
            probability_value = 0.35
        if not math.isfinite(probability_value):
            probability_value = 0.35
        self._probability = max(0.0, min(probability_value, 1.0))
        self._random = random_source or random.Random()
        self._context_factory = context_factory or (
            lambda: ToolCallContext("default", "local", "behavior", source="random")
        )
        self._movement = movement or AutonomousMovementConfig()
        self._position_provider = position_provider
        self._bounds_provider = bounds_provider
        self._interaction_provider = interaction_provider
        self._motion_generation_provider = motion_generation_provider
        self._direction_setter = direction_setter
        try:
            cooldown = float(interaction_cooldown_seconds)
        except (TypeError, ValueError, OverflowError):
            cooldown = 0.35
        if not math.isfinite(cooldown):
            cooldown = 0.35
        self._interaction_cooldown_seconds = max(0.0, min(10.0, cooldown))
        self._clock = clock or time.monotonic
        self._interaction_cooldown_until = 0.0
        self._interaction_active = False
        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._last_error = ""
        self._phase = BehaviorPhase.STOPPED
        self._target: tuple[int, int] | None = None
        self._last_position: tuple[int, int] | None = None
        self._movement_count = 0
        self._interrupt_generation = 0
        # ``tick`` 可能由运行循环和测试/控制台入口同时触发；移动工具
        # 不是可重入操作。串行化后不会出现两个随机目标交错提交位置。
        self._tick_lock = asyncio.Lock()
        self._facing: str | None = None
        self._movement_motion_active = False
        self._movement_motion_idle_name: str | None = None
        self._movement_motion_generation: int | None = None
        # 记录正在执行的随机动作；交互/取消发生在工具 await 期间时，
        # tick 返回前仍可把已经提交的非 idle 动作收敛回基线。
        self._active_action_identity: str | None = None
        self._active_action_motion_name: str | None = None
        self._active_action_idle_motion: str | None = None
        self._active_action_motion_generation: int | None = None
        self._active_action_duration_seconds = _DEFAULT_ACTION_DURATION_SECONDS
        self._busy_tick_count = 0

    def choose_action(self) -> BehaviorAction | None:
        """按一次触发抽样和权重选择一个随机动作。"""

        unit = self._sample_behavior_unit()
        if unit is None:
            return None
        return self._choose_weighted_action(unit)

    def _sample_behavior_unit(self) -> float | None:
        """把 ``probability`` 转换为一次统一的行为抽样。

        以前漫游分支和随机动作各自做一次概率抽样，开启两者时实际触发
        率变成 ``2p-p²``，而且固定随机源会让动作序列难以预测。现在一次
        ``random()`` 同时决定“本轮是否行动”和条件分布中的位置；返回的
        值在行为已触发时均匀落在 ``[0, 1]``。
        """

        if self._probability <= 0.0:
            return None
        try:
            roll = float(self._random.random())
        except (AttributeError, TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(roll):
            return None
        roll = max(0.0, min(1.0, roll))
        if roll > self._probability:
            return None
        # ``probability`` 已在构造/重载阶段限制到正数；这里仍保留有限
        # 保护，避免外部直接篡改字段后产生 NaN/无穷游标。
        unit = roll / self._probability
        return unit if math.isfinite(unit) else None

    def _choose_weighted_action(self, unit: float) -> BehaviorAction | None:
        """按已归一化游标选择动作，不再次读取概率或随机源。"""

        if not self._actions or not math.isfinite(unit):
            return None
        total = sum(action.weight for action in self._actions)
        if not math.isfinite(total) or total <= 0:
            return None
        cursor = max(0.0, min(1.0, unit)) * total
        for action in self._actions:
            cursor -= action.weight
            if cursor <= 0:
                return action
        return self._actions[-1]

    def _choose_behavior(self) -> tuple[bool, BehaviorAction | None]:
        """一次决定本轮执行漫游、随机动作或空闲。

        漫游使用隐含权重 ``1.0``，随机动作使用各自配置权重。这样
        ``probability`` 表示整个行为周期的总触发率，而不是在漫游和动作
        分支上重复计算；两类行为仍能在同一配置中按权重交替出现。
        """

        unit = self._sample_behavior_unit()
        if unit is None:
            return False, None
        movement_available = bool(self._movement.enabled)
        action_total = sum(action.weight for action in self._actions)
        if not math.isfinite(action_total) or action_total < 0:
            action_total = 0.0
        total = (1.0 if movement_available else 0.0) + action_total
        if not math.isfinite(total) or total <= 0:
            return False, None
        cursor = unit * total
        if movement_available:
            if cursor < 1.0 or action_total <= 0.0:
                return True, None
            cursor -= 1.0
        if action_total <= 0.0:
            return False, None
        action_unit = cursor / action_total
        return False, self._choose_weighted_action(action_unit)

    def reconfigure(
        self,
        *,
        actions: tuple[BehaviorAction, ...] | None = None,
        interval_seconds: tuple[float, float] | None = None,
        probability: float | None = None,
        movement: AutonomousMovementConfig | None = None,
        interaction_cooldown_seconds: float | None = None,
    ) -> None:
        """在运行时线程内替换行为参数，不重建工具或随机源。

        调用方应先完成配置校验；这里仍复用构造阶段的有限值处理，确保
        外部宿主直接调用时不会把非有限权重或间隔写入后台循环。移动参数
        发生变化时推进中断代际，让旧目标不会继续提交位置。
        """

        if actions is not None:
            valid_actions: list[BehaviorAction] = []
            for action in actions:
                try:
                    weight = float(action.weight)
                except (AttributeError, TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(weight) and weight > 0:
                    valid_actions.append(
                        BehaviorAction(
                            action.identity,
                            action.arguments,
                            weight,
                            getattr(action, "duration_seconds", _DEFAULT_ACTION_DURATION_SECONDS),
                        )
                    )
            self._actions = tuple(valid_actions)
        if interval_seconds is not None:
            try:
                low, high = interval_seconds
                low_value = max(0.5, float(low))
                high_value = max(low_value, float(high))
                if math.isfinite(low_value) and math.isfinite(high_value):
                    self._interval = (low_value, high_value)
            except (TypeError, ValueError, OverflowError):
                pass
        if probability is not None:
            try:
                value = float(probability)
                if math.isfinite(value):
                    self._probability = max(0.0, min(value, 1.0))
            except (TypeError, ValueError, OverflowError):
                pass
        if interaction_cooldown_seconds is not None:
            try:
                cooldown = float(interaction_cooldown_seconds)
                if math.isfinite(cooldown):
                    self._interaction_cooldown_seconds = max(0.0, min(10.0, cooldown))
            except (TypeError, ValueError, OverflowError):
                pass
        if movement is not None:
            changed = movement != self._movement
            self._movement = movement
            if changed and self._phase in {
                BehaviorPhase.PLANNING,
                BehaviorPhase.MOVING,
                BehaviorPhase.ACTING,
            }:
                self.interrupt()

    async def _provider_value(self, provider: Callable[[], object] | None) -> object:
        if provider is None:
            return None
        try:
            result = provider()
            return await result if inspect.isawaitable(result) else result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None

    async def _read_motion_generation(self) -> int | None:
        """读取当前仍由行为服务持有的动作代次。"""

        provider = self._motion_generation_provider
        if provider is None:
            return None
        value = await self._provider_value(provider)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        # 允许测试替身返回整数浮点，拒绝截断任意小数或非有限值。
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        return None

    async def _motion_owner_matches(self, generation: int | None) -> bool:
        """仅在动作代次仍由本行为持有时允许发送 idle。"""

        # 没有代数探针的旧宿主沿用既有收尾行为；生产控制器会提供
        # ``motion_owner_generation``，外部 motion 替换动作后返回 None；
        # 单纯移动/表情仍允许旧 motion 按所有权正常收尾。
        if self._motion_generation_provider is None:
            return True
        if generation is None:
            return False
        current = await self._read_motion_generation()
        return current == generation

    @staticmethod
    def _coerce_xy(value: object) -> tuple[int, int] | None:
        """兼容 Qt QPoint、二元序列和 x/y 映射。"""

        if isinstance(value, Mapping):
            if "x" not in value or "y" not in value:
                return None
            value = (value.get("x"), value.get("y"))
        x_getter = getattr(value, "x", None)
        y_getter = getattr(value, "y", None)
        if callable(x_getter) and callable(y_getter):
            try:
                return int(round(float(x_getter()))), int(round(float(y_getter())))
            except (TypeError, ValueError, OverflowError):
                return None
        if isinstance(value, (str, bytes, bytearray)):
            return None
        try:
            return int(round(float(value[0]))), int(round(float(value[1])))  # type: ignore[index]
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def _coerce_bounds(cls, value: object) -> tuple[int, int, int, int] | None:
        """规整为可放置窗口左上角的 ``left, top, right, bottom``。"""

        if isinstance(value, Mapping):
            if all(key in value for key in ("left", "top", "right", "bottom")):
                value = (value["left"], value["top"], value["right"], value["bottom"])
            elif all(key in value for key in ("x", "y", "width", "height")):
                value = (
                    value["x"],
                    value["y"],
                    float(value["x"]) + float(value["width"]) - 1,
                    float(value["y"]) + float(value["height"]) - 1,
                )
            else:
                return None
        else:
            # 生产 Qt 宿主通常已转换为元组；直接传入 QRect/QRectF 时也
            # 使用 Qt 的闭区间边界，避免把窗口尺寸误当作右下坐标。
            edge_getters = tuple(
                getattr(value, name, None) for name in ("left", "top", "right", "bottom")
            )
            if all(callable(getter) for getter in edge_getters):
                try:
                    value = tuple(getter() for getter in edge_getters)  # type: ignore[misc]
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    return None
            else:
                geometry_getters = tuple(
                    getattr(value, name, None) for name in ("x", "y", "width", "height")
                )
                if all(callable(getter) for getter in geometry_getters):
                    try:
                        x, y, width, height = (getter() for getter in geometry_getters)  # type: ignore[misc]
                        value = (x, y, float(x) + float(width) - 1, float(y) + float(height) - 1)
                    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
                        return None
        if isinstance(value, (str, bytes, bytearray)):
            return None
        try:
            left, top, right, bottom = (int(round(float(value[index]))) for index in range(4))  # type: ignore[index]
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            return None
        if right < left or bottom < top:
            return None
        return left, top, right, bottom

    async def _is_interacting(self) -> bool:
        # 没有探针时保持向后兼容；探针存在但读取失败时按“正在交互”
        # 处理，避免在窗口状态不确定时后台移动覆盖用户操作。
        if self._interaction_provider is None:
            # 即使宿主没有交互探针，显式 ``interrupt()`` 也必须保留
            # 一个短暂安全窗；否则快速拖动/配置锁定结束后下一次 tick
            # 会立即重新规划，旧窗口坐标可能再次回弹。
            return self._record_interaction_state(False)
        value = await self._provider_value(self._interaction_provider)
        if value is None:
            return self._record_interaction_state(True)
        active = False
        if isinstance(value, Mapping):
            if "interacting" in value or "active" in value:
                raw = value.get("interacting", value.get("active", False))
                if isinstance(raw, str):
                    normalized = raw.strip().lower()
                    if normalized in {"true", "1", "yes", "on"}:
                        active = True
                    elif normalized in {"false", "0", "no", "off", ""}:
                        active = False
                    else:
                        # 未知字符串按探针失败闭锁，不能被 Python 的 bool()
                        # 静默解释为“可自由移动”。
                        active = True
                else:
                    active = bool(raw)
                return self._record_interaction_state(active)
            status = getattr(value.get("status"), "value", value.get("status"))
            # MutablePetController 在宿主未挂载或 Qt 调度器关闭时会返回
            # 不可用回执；这属于探针失败，必须闭锁自主行为，而不是把
            # 协议状态映射成“未交互”后继续移动。
            normalized_status = str(status or "").strip().lower()
            if normalized_status in _FAILURE_STATUSES:
                return self._record_interaction_state(True)
            # 没有明确 interacting/active 字段的未知回执不能被当成“空闲”
            # 放行；只有已知的成功状态才允许继续自主行为。
            active = normalized_status not in {"available", "ready", "completed", "idle"}
            return self._record_interaction_state(active)
        return self._record_interaction_state(bool(value))

    def _record_interaction_state(self, active: bool) -> bool:
        """记录交互边沿，并在释放后保留一个短暂的安全冷却。"""

        now = self._safe_now()
        if active:
            self._interaction_active = True
            # 交互持续期间不断后移截止时间；释放后至少完整等待一段
            # 冷却，避免旧的漫游步进/动作在鼠标松开瞬间重新抢占渲染器。
            self._interaction_cooldown_until = max(
                self._interaction_cooldown_until,
                now + self._interaction_cooldown_seconds,
            )
            return True
        if self._interaction_active:
            self._interaction_active = False
        return now < self._interaction_cooldown_until

    def _safe_now(self) -> float:
        """读取有限单调时钟；自定义时钟异常时回退到系统时钟。"""

        try:
            value = float(self._clock())
        except (TypeError, ValueError, OverflowError, OSError, RuntimeError):
            value = time.monotonic()
        return value if math.isfinite(value) else time.monotonic()

    async def _set_facing_for_target(
        self,
        current: tuple[int, int],
        target: tuple[int, int],
        *,
        generation: int | None = None,
    ) -> None:
        """在移动开始前更新渲染方向；方向切换不改变窗口几何。"""

        setter = self._direction_setter
        if not self._movement.flip_facing or setter is None:
            return
        delta_x = target[0] - current[0]
        if delta_x == 0:
            return
        # 资源约定与旧项目一致：A 面向左，B 面向右。方向只由本次
        # 位移的横向符号决定，不使用屏幕绝对坐标，否则桌宠在屏幕
        # 左右两侧会出现反向行走的视觉反馈。
        direction = "A" if delta_x < 0 else "B"
        # 同一段连续游走保持朝向，避免每次随机规划都重复触发渲染器
        # 的整帧重绘，表现为拖动或漫游时模型抽搐。
        if direction == self._facing:
            return
        try:
            result = setter(direction)
            if inspect.isawaitable(result):
                # 异步 setter 的最终值才代表 Qt/渲染线程是否接受了方向；
                # 不能把 coroutine 对象本身误当作成功。
                result = await result
            accepted = self._result_succeeded(result)
            # 方向 setter 可能跨线程等待；用户在等待期间开始拖动时，
            # 这次旧规划即使收到成功回执也不能继续缓存为当前朝向。否则
            # 释放后下一轮会把旧目标方向当成有效状态，表现为镜像抽动。
            stale = generation is not None and generation != self._interrupt_generation
            if not stale:
                try:
                    stale = await self._is_interacting()
                except asyncio.CancelledError:
                    raise
            if accepted and not stale:
                self._facing = direction
            elif stale:
                self._facing = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 转身是视觉增强，不能让平台不支持方向切换时阻断位置移动。
            self._last_error = f"direction update failed: {type(exc).__name__}: {exc}"

    def _identity_is_safe(self, identity: str) -> bool:
        """随机行为拒绝中/高风险工具，避免后台循环产生审批风暴。"""

        registry = getattr(self._executor, "registry", None)
        getter = getattr(registry, "get", None) if registry is not None else None
        if not callable(getter):
            return True
        spec = getter(identity)
        if spec is None:
            return False
        try:
            return int(getattr(spec, "risk", 0)) <= 0
        except (TypeError, ValueError):
            return False

    def _new_context(self) -> ToolCallContext:
        context = self._context_factory()
        metadata = dict(context.metadata)
        # 行为自身的动作/移动通过同一工具权限管线执行，但不能把“调用动作”
        # 当成新的用户交互再次中断当前 tick。该标志只在内部上下文中传播，
        # 外部模型、控制台和调度器上下文不会携带它。
        metadata[BEHAVIOR_INTERNAL_METADATA_KEY] = True
        return ToolCallContext(
            context.profile_id,
            context.session_id,
            f"{context.turn_id[:220]}-{secrets.token_urlsafe(8)}",
            source=context.source,
            metadata=metadata,
        )

    async def _execute(self, identity: str, arguments: Mapping[str, Any]) -> object:
        if not self._identity_is_safe(identity):
            self._last_error = f"blocked unsafe behavior tool: {identity}"
            return {"status": "denied", "reason": "behavior only permits low-risk tools"}
        execute = getattr(self._executor, "execute", None)
        if not callable(execute):
            raise RuntimeError("behavior tool executor is unavailable")
        context = self._new_context()
        try:
            result = execute(
                call_id=f"behavior-{secrets.token_urlsafe(8)}",
                identity=identity,
                arguments=dict(arguments),
                context=context,
            )
            return await result if inspect.isawaitable(result) else result
        finally:
            clear_turn = getattr(self._executor, "clear_turn", None)
            if callable(clear_turn):
                result = clear_turn(context)
                if inspect.isawaitable(result):
                    await result

    async def _movement_cancelled(self, *, reset_motion: bool) -> dict[str, str]:
        """收尾被用户交互打断的漫游，并停止已经开始的走动动作。"""

        # 本地 movement_started 与渲染器回执之间可能存在一个 await；
        # 只要服务字段表明 walk 已经生效，就必须执行一次 idle 复位。
        await self._reset_movement_motion(force=reset_motion or self._movement_motion_active)
        self._phase = BehaviorPhase.PAUSED
        return {"status": "cancelled", "reason": "user interaction interrupted movement"}

    async def _wait_between_movement_steps(
        self,
        delay_seconds: float,
        *,
        generation: int,
        reset_motion: bool,
    ) -> dict[str, str] | None:
        """以短时间片等待，避免低速长路径延迟响应拖动/锁定。"""

        remaining = max(0.0, float(delay_seconds))
        while remaining > 0.0:
            slice_seconds = min(remaining, _MOVEMENT_INTERACTION_POLL_SECONDS)
            await asyncio.sleep(slice_seconds)
            remaining = max(0.0, remaining - slice_seconds)
            if generation != self._interrupt_generation or await self._is_interacting():
                return await self._movement_cancelled(reset_motion=reset_motion)
        return None

    async def _reset_movement_motion(self, *, force: bool = False) -> None:
        """停止已经接受的走动动作；动作不支持时只记录错误。"""

        active = self._movement_motion_active
        idle_name = self._movement_motion_idle_name
        owner_generation = self._movement_motion_generation
        self._movement_motion_active = False
        self._movement_motion_idle_name = None
        self._movement_motion_generation = None
        if not (force or active):
            return
        idle_name = str(idle_name or self._movement.idle_motion or "").strip()
        if not idle_name:
            return
        if not await self._motion_owner_matches(owner_generation):
            return
        try:
            await asyncio.wait_for(
                self._play_motion(idle_name), timeout=_MOTION_RESET_TIMEOUT_SECONDS
            )
        except TimeoutError:
            if not self._last_error:
                self._last_error = "idle motion reset timed out"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._last_error:
                self._last_error = f"idle motion reset failed: {type(exc).__name__}"

    @staticmethod
    def _result_status(result: object) -> str:
        if result is True:
            return "completed"
        if result is False:
            return "failed"

        outer_status = getattr(result, "status", None)
        outer_status = getattr(outer_status, "value", outer_status)
        outer = str(outer_status or "").strip().lower()

        content = getattr(result, "content", None)
        if isinstance(content, Mapping):
            embedded = getattr(content.get("status"), "value", content.get("status"))
            embedded_text = str(embedded or "").strip().lower()
            if outer in _FAILURE_STATUSES:
                return outer
            if embedded_text:
                return embedded_text
            if content.get("ok") is False or content.get("success") is False:
                return "failed"

        if isinstance(result, Mapping):
            status = getattr(result.get("status"), "value", result.get("status"))
            text = str(status or "").strip().lower()
            if text:
                return text
            if result.get("ok") is False or result.get("success") is False:
                return "failed"
            return ""
        return outer

    @classmethod
    def _result_succeeded(cls, result: object) -> bool:
        """把工具/Qt setter 的多种返回形态归一为是否接受。"""

        if result is False:
            return False
        if isinstance(result, Mapping):
            if result.get("ok") is False or result.get("success") is False:
                return False
        else:
            if getattr(result, "ok", True) is False or getattr(result, "success", True) is False:
                return False
        return cls._result_status(result) not in _FAILURE_STATUSES

    async def _play_motion(self, name: str) -> bool:
        """播放动作并识别不支持/失败结果；失败时不阻断窗口移动。"""

        result = await self._execute("pet:play_motion", {"name": name})
        if self._result_succeeded(result):
            return True
        status = self._result_status(result) or "unavailable"
        if not self._last_error:
            self._last_error = f"motion failed: {name} ({status})"
        return False

    @staticmethod
    def _motion_name_from_action(action: BehaviorAction) -> str | None:
        """从随机动作中提取可在中断时复位的动作名称。"""

        if str(action.identity or "").strip() != "pet:play_motion":
            return None
        arguments = action.arguments
        if not isinstance(arguments, Mapping):
            return None
        name = str(arguments.get("name", "") or "").strip()
        if not name or name.casefold() == "idle":
            return None
        return name

    def _begin_action(self, action: BehaviorAction) -> None:
        """登记动作边界；登记在 await 前，覆盖工具已提交后才抛错的情况。"""

        self._active_action_identity = str(action.identity or "").strip() or None
        self._active_action_motion_name = self._motion_name_from_action(action)
        # 动作开始时固定复位动作；配置热重载可能在工具 await 期间替换
        # ``movement.idle_motion``，不能让新配置的空值跳过旧动作收尾。
        self._active_action_idle_motion = str(self._movement.idle_motion or "").strip() or None
        try:
            duration = float(action.duration_seconds)
        except (AttributeError, TypeError, ValueError, OverflowError):
            duration = _DEFAULT_ACTION_DURATION_SECONDS
        self._active_action_duration_seconds = (
            duration if math.isfinite(duration) else _DEFAULT_ACTION_DURATION_SECONDS
        )
        self._active_action_duration_seconds = max(
            0.0,
            min(_MAX_ACTION_DURATION_SECONDS, self._active_action_duration_seconds),
        )

    def _clear_action(self) -> None:
        self._active_action_identity = None
        self._active_action_motion_name = None
        self._active_action_idle_motion = None
        self._active_action_motion_generation = None
        self._active_action_duration_seconds = _DEFAULT_ACTION_DURATION_SECONDS

    async def _settle_random_action(
        self,
        *,
        generation: int,
        succeeded: bool,
    ) -> dict[str, str] | None:
        """让随机 motion 展示有限时长后回到 idle。

        该路径只处理 ``BehaviorAction`` 登记的随机动作，不影响模型在对话
        或点击回调中主动发起的长期表情/动作。等待期间按短时间片检查代际
        和用户交互，保证拖动、锁定或停止都能先清理正式 motion。
        """

        if self._active_action_motion_name is None:
            return None
        if not succeeded:
            await self._cleanup_interrupted_action()
            return None

        remaining = max(0.0, float(self._active_action_duration_seconds))
        while remaining > 0.0:
            slice_seconds = min(remaining, _MOVEMENT_INTERACTION_POLL_SECONDS)
            await asyncio.sleep(slice_seconds)
            remaining = max(0.0, remaining - slice_seconds)
            if generation != self._interrupt_generation or await self._is_interacting():
                await self._cleanup_interrupted_action()
                self._phase = BehaviorPhase.PAUSED
                self._target = None
                return {
                    "status": "cancelled",
                    "reason": "user interaction interrupted action",
                }
        await self._cleanup_interrupted_action()
        return None

    async def _cleanup_interrupted_action(self) -> None:
        """交互/取消时停止已登记的随机动作。"""

        motion_name = self._active_action_motion_name
        idle_name = self._active_action_idle_motion
        owner_generation = self._active_action_motion_generation
        self._active_action_motion_name = None
        self._active_action_idle_motion = None
        self._active_action_motion_generation = None
        if motion_name is None:
            return
        idle_name = str(idle_name or self._movement.idle_motion or "").strip()
        if not idle_name or idle_name.casefold() == motion_name.casefold():
            return
        if not await self._motion_owner_matches(owner_generation):
            return
        try:
            await asyncio.wait_for(
                self._play_motion(idle_name), timeout=_MOTION_RESET_TIMEOUT_SECONDS
            )
        except TimeoutError:
            if not self._last_error:
                self._last_error = "interrupted motion reset timed out"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._last_error:
                self._last_error = f"interrupted motion reset failed: {type(exc).__name__}"

    def _plan_target(
        self,
        current: tuple[int, int],
        bounds: tuple[int, int, int, int],
    ) -> tuple[int, int] | None:
        left, top, right, bottom = bounds
        start_x = max(left, min(current[0], right))
        start_y = max(top, min(current[1], bottom))
        for _ in range(12):
            distance = self._random.uniform(
                self._movement.min_distance_px, self._movement.max_distance_px
            )
            angle = self._random.uniform(0.0, math.tau)
            target_x = round(start_x + math.cos(angle) * distance)
            target_y = round(start_y + math.sin(angle) * distance)
            target = (
                max(left, min(int(target_x), right)),
                max(top, min(int(target_y), bottom)),
            )
            if math.hypot(target[0] - start_x, target[1] - start_y) >= min(
                self._movement.min_distance_px, max(right - left, bottom - top)
            ):
                return target
        # 小屏或边界很窄时，选离当前点最远的角，仍然保持边界安全。
        corners = ((left, top), (left, bottom), (right, top), (right, bottom))
        target = max(corners, key=lambda point: math.hypot(point[0] - start_x, point[1] - start_y))
        return target if target != (start_x, start_y) else None

    async def _wander(self) -> object | None:
        """执行一次漫游，并保证异常不会把状态机卡在中间阶段。"""

        try:
            return await self._wander_impl()
        except asyncio.CancelledError:
            # stop() 会在取消任务前设置 stop_event；其它外部取消仍保留
            # 暂停态，下一轮不会误认为漫游已经正常完成。若走路动作已经
            # 被渲染器接受，先尽力切回 idle；否则程序化 walk 可能在移动
            # 协程取消后继续写入参数，表现为桌宠“自己乱动”。
            motion_was_active = self._movement_motion_active
            if motion_was_active:
                try:
                    await self._reset_movement_motion(force=True)
                except asyncio.CancelledError:
                    # 外部可能连续取消；状态标志仍必须清零，避免下一轮
                    # 把已经失效的动作当成活动动作再次复位。
                    self._movement_motion_active = False
                except Exception as exc:
                    self._last_error = f"idle motion reset failed: {type(exc).__name__}"
            self._movement_motion_active = False
            stop_event = self._stop_event
            self._phase = (
                BehaviorPhase.STOPPED
                if stop_event is not None and stop_event.is_set()
                else BehaviorPhase.PAUSED
            )
            raise
        except Exception as exc:
            # 移动动作已经提交但位置工具随后失败时，也要恢复待机动作，
            # 否则程序化 walk 会继续写参数，造成桌宠自行抖动。
            try:
                await self._reset_movement_motion()
            except asyncio.CancelledError:
                raise
            except Exception as reset_exc:
                self._last_error = f"idle motion reset failed: {type(reset_exc).__name__}"
            self._last_error = f"autonomous movement failed: {type(exc).__name__}"
            # interrupt() 已将阶段标记为 PAUSED 时，异常只是迟到的
            # 工具回执，不得把用户交互状态覆盖回 IDLE。
            if self._phase is not BehaviorPhase.PAUSED:
                self._phase = BehaviorPhase.IDLE
            return {"status": "failed", "reason": "autonomous movement failed"}
        finally:
            # 目标只在 MOVING 阶段有意义；清除后控制台不会展示已经结束
            # 的旧目标，也不会让下一次规划误用过期几何。
            if self._phase in {BehaviorPhase.PLANNING, BehaviorPhase.MOVING}:
                self._phase = BehaviorPhase.IDLE
            self._target = None

    async def _wander_impl(self) -> object | None:
        self._phase = BehaviorPhase.PLANNING
        # 在读取 Qt 几何之前记录代际；用户若在规划阶段开始拖动，
        # 即使交互探针随后短暂恢复，也不能让这次旧规划继续执行。
        generation = self._interrupt_generation
        if self._movement.movement_identity != "pet:autonomous_move":
            self._phase = BehaviorPhase.IDLE
            self._last_error = "movement identity is not the protected autonomous tool"
            return {"status": "denied", "reason": "protected autonomous tool is required"}
        current = self._coerce_xy(await self._provider_value(self._position_provider))
        bounds = self._coerce_bounds(await self._provider_value(self._bounds_provider))
        if current is None or bounds is None:
            self._phase = BehaviorPhase.IDLE
            self._last_error = "movement geometry is unavailable"
            return None
        target = self._plan_target(current, bounds)
        if target is None:
            self._phase = BehaviorPhase.IDLE
            return None
        self._target = target
        self._phase = BehaviorPhase.MOVING
        last_result: object | None = None
        if generation != self._interrupt_generation or await self._is_interacting():
            return await self._movement_cancelled(reset_motion=False)
        await self._set_facing_for_target(current, target, generation=generation)
        if generation != self._interrupt_generation or await self._is_interacting():
            return await self._movement_cancelled(reset_motion=False)
        movement_started = False
        self._movement_motion_active = False
        self._movement_motion_generation = None
        # 先登记本次游走对应的待机动作，再等待渲染器确认 walk。
        # 渲染器可能在 await 期间已经接受动作、随后因拖动/关闭取消
        # 请求；若只在 await 返回后登记，取消路径无法发出 idle 收尾。
        self._movement_motion_idle_name = str(self._movement.idle_motion or "").strip() or None
        if self._movement.moving_motion:
            # 动作是可选视觉效果；模型未声明该动作时只记录失败并继续
            # 位置移动，不能把“动作不支持”误当成移动已经开始。
            self._movement_motion_active = True
            try:
                movement_started = await self._play_motion(self._movement.moving_motion)
            except asyncio.CancelledError:
                # ``_play_motion`` 的副作用可能早于其回执；保留 active
                # 标志交给 ``_wander`` 的取消收尾，确保 walk 不会残留。
                raise
            except Exception:
                # 普通异常也走 ``_wander`` 的收尾；不要在这里提前清掉
                # 可能已经提交到渲染器的动作标志。
                raise
            self._movement_motion_active = movement_started
            if movement_started:
                self._movement_motion_generation = await self._read_motion_generation()
        # 用欧氏距离计算步数，并在 max_steps 触顶时拉长步间隔；仅把
        # 步数截断而继续使用原 interval 会让长路径瞬间超出
        # max_speed_px_per_second，表现为窗口突然跳动。
        distance = math.hypot(target[0] - current[0], target[1] - current[1])
        nominal_step = min(
            self._movement.step_pixels,
            max(1.0, self._movement.max_speed_px_per_second * self._movement.step_interval_seconds),
        )
        steps = max(
            1,
            min(
                self._movement.max_steps,
                math.ceil(distance / nominal_step),
            ),
        )
        step_distance = distance / steps if steps else distance
        step_interval = max(
            self._movement.step_interval_seconds,
            step_distance / self._movement.max_speed_px_per_second,
        )
        previous_point = current
        reached_target = False
        for index in range(1, steps + 1):
            if generation != self._interrupt_generation or await self._is_interacting():
                return await self._movement_cancelled(reset_motion=movement_started)
            ratio = index / steps
            point = (
                round(current[0] + (target[0] - current[0]) * ratio),
                round(current[1] + (target[1] - current[1]) * ratio),
            )
            # 四舍五入后相邻插值点可能相同；跳过重复位置，避免向 Qt
            # 连续提交同一个窗口坐标造成合成器重绘抖动。
            if point == previous_point:
                continue
            last_result = await self._execute(
                self._movement.movement_identity,
                {"x": point[0], "y": point[1], "duration_ms": 0},
            )
            status = self._result_status(last_result)
            if status in {"cancelled", "canceled"}:
                # 宿主在拖动/锁定切换期间会拒绝迟到的自主步进；这不是
                # 普通工具失败，必须保留 PAUSED 状态并复位 walk，避免
                # 下一轮把旧目标当成已完成后继续乱动。
                return await self._movement_cancelled(reset_motion=movement_started)
            if not self._result_succeeded(last_result):
                self._last_error = f"movement failed: {status}"
                break
            # 移动工具可能在等待平台/Qt 调度时才收到用户拖动；执行返回
            # 后再次检查代际和交互状态，不把这一个迟到步进当作仍属
            # 自主游走，也不继续提交后续坐标。
            if generation != self._interrupt_generation or await self._is_interacting():
                return await self._movement_cancelled(reset_motion=movement_started)
            self._last_position = point
            previous_point = point
            reached_target = point == target
            if index < steps:
                interrupted = await self._wait_between_movement_steps(
                    step_interval,
                    generation=generation,
                    reset_motion=movement_started,
                )
                if interrupted is not None:
                    return interrupted
        # 只有 walk 确实被渲染器接受时才切回待机；无 moving_motion 或
        # 不支持该动作时不额外提交 idle，避免每次漫游结束都覆盖其它
        # 表情/动作并造成“乱动”。helper 会先清除 active 标志，因而
        # idle setter 抛错时 wrapper 不会重复提交同一动作。
        await self._reset_movement_motion(force=movement_started)
        self._phase = BehaviorPhase.IDLE
        if reached_target:
            self._movement_count += 1
        return last_result

    async def tick(self) -> object | None:
        # ``asyncio.Lock`` 默认会把并发调用排队。行为 tick 可能包含数十个
        # 移动步进，排队的旧 tick 在用户拖动后仍会继续规划，表现为回弹和
        # “自己乱动”。只接受当前时间片的首个 tick，其余调用直接丢弃；
        # ``locked()`` 与随后的 ``async with`` 之间没有 await，在同一事件
        # 循环内是原子的，也避免跨 ``asyncio.run`` 测试绑定锁到错误循环。
        if self._tick_lock.locked():
            self._busy_tick_count += 1
            return None
        async with self._tick_lock:
            try:
                if await self._is_interacting():
                    # 交互可能在上一轮工具返回之后才被宿主观测到；先
                    # 收走残留 motion，再把本轮状态置为暂停，不能只清理
                    # target 而让渲染器继续沿用 walk/随机动作。
                    await self._reset_movement_motion(force=self._movement_motion_active)
                    await self._cleanup_interrupted_action()
                    self._phase = BehaviorPhase.PAUSED
                    self._target = None
                    return None

                # 新一轮行为开始后，上一轮的瞬时错误不应一直污染状态卡；
                # 当前轮的工具/几何错误会在下面重新写入。
                self._last_error = ""

                # 一次 tick 只允许一种主动行为。漫游会产生多个位置步进，
                # 若其后立即播放随机动作，渲染状态会在 walk/idle/动作之间
                # 连续覆盖，表现为桌宠乱动。统一决策器只读取一次概率，
                # 再按漫游/动作权重选择当前类别，避免双重概率放大。
                movement_selected, action = self._choose_behavior()
                if movement_selected:
                    return await self._wander()

                if action is None:
                    self._phase = BehaviorPhase.IDLE
                    self._target = None
                    return None

                generation = self._interrupt_generation
                # 用户交互可能在抽样和动作提交之间开始；再次确认后才
                # 提交随机动作，避免控制台/拖动期间抢占模型状态。
                if generation != self._interrupt_generation or await self._is_interacting():
                    self._phase = BehaviorPhase.PAUSED
                    return None
                self._phase = BehaviorPhase.ACTING
                self._begin_action(action)
                try:
                    action_result = await self._execute(action.identity, action.arguments)
                    succeeded = self._result_succeeded(action_result)
                    if succeeded and self._active_action_motion_name is not None:
                        self._active_action_motion_generation = await self._read_motion_generation()
                    if not succeeded:
                        # 明确失败的动作通常没有可复位状态，但登记值先
                        # 保留到交互检查之后；工具可能在返回失败前已经
                        # 把 motion 写入渲染器，取消路径仍要尽力发出 idle。
                        if not self._last_error:
                            status = self._result_status(action_result) or "failed"
                            self._last_error = (
                                f"behavior action failed: {action.identity} ({status})"
                            )
                    settled = await self._settle_random_action(
                        generation=generation,
                        succeeded=succeeded,
                    )
                    if settled is not None:
                        return settled
                    interrupted = (
                        generation != self._interrupt_generation or await self._is_interacting()
                    )
                    if interrupted:
                        await self._cleanup_interrupted_action()
                        self._phase = BehaviorPhase.PAUSED
                        self._target = None
                        return {
                            "status": "cancelled",
                            "reason": "user interaction interrupted action",
                        }
                    self._phase = BehaviorPhase.IDLE
                    self._target = None
                    return action_result
                except asyncio.CancelledError:
                    # 取消可能发生在工具已经把 motion 交给渲染器、但还
                    # 没有返回回执的窗口；清理登记值后再向上保留取消。
                    try:
                        await self._cleanup_interrupted_action()
                    except asyncio.CancelledError:
                        self._active_action_motion_name = None
                    raise
                except Exception:
                    # 工具异常同样可能发生在副作用已经提交之后；不要在
                    # finally 清空登记前跳过一次 idle 收尾。
                    await self._cleanup_interrupted_action()
                    raise
                finally:
                    self._clear_action()
            except asyncio.CancelledError:
                try:
                    await self._cleanup_interrupted_action()
                except asyncio.CancelledError:
                    self._active_action_motion_name = None
                stop_event = self._stop_event
                self._phase = (
                    BehaviorPhase.STOPPED
                    if stop_event is not None and stop_event.is_set()
                    else BehaviorPhase.PAUSED
                )
                self._target = None
                raise
            except Exception as exc:
                # 手动 tick 与后台循环都经过这里；将异常转成可观察结果，
                # 同时收敛 phase/target，避免下一轮继续沿用 MOVING/ACTING。
                self._last_error = f"behavior tick failed: {type(exc).__name__}"
                if self._phase is not BehaviorPhase.PAUSED:
                    self._phase = BehaviorPhase.IDLE
                self._target = None
                return {"status": "failed", "reason": "behavior tick failed"}

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._phase = BehaviorPhase.IDLE
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
            delay = self._random.uniform(*self._interval)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            except TimeoutError:
                pass

    def status(self) -> dict[str, object]:
        """返回随机行为循环状态。"""

        return {
            "running": self._task is not None and not self._task.done(),
            "action_count": len(self._actions),
            "last_error": self._last_error,
            "phase": self._phase.value,
            "target": self._target,
            "position": self._last_position,
            "movement_count": self._movement_count,
            "movement_enabled": self._movement.enabled,
            "step_interval_seconds": self._movement.step_interval_seconds,
            "max_speed_px_per_second": self._movement.max_speed_px_per_second,
            "interaction_cooldown_seconds": self._interaction_cooldown_seconds,
            "busy_tick_count": self._busy_tick_count,
            "active_action": self._active_action_identity,
        }

    def interrupt(self) -> None:
        """取消当前移动；下一轮在用户交互结束后重新规划。"""

        self._interrupt_generation += 1
        self._facing = None
        # 显式中断通常来自拖动、控制台锁定或窗口状态切换。把它记录为
        # 一次交互边沿，即使宿主没有交互探针或探针很快恢复空闲，也不会
        # 在同一时间片重新抢占渲染器。
        now = self._safe_now()
        self._interaction_active = True
        self._interaction_cooldown_until = max(
            self._interaction_cooldown_until,
            now + self._interaction_cooldown_seconds,
        )
        # 交互开始时立即撤掉公开目标，避免控制台在漫游协程尚未返回的
        # 短窗口继续显示一个已经失效的位置。
        self._target = None
        if self._phase in {
            BehaviorPhase.PLANNING,
            BehaviorPhase.MOVING,
            BehaviorPhase.ACTING,
        }:
            self._phase = BehaviorPhase.PAUSED

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        task = self._task
        if task is not None:
            if task is not asyncio.current_task():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # 手动 tick 可能不属于后台任务；即使没有可取消的 ``_task``，
        # 也不要把上一次走路/动作的状态带到下一次启动。
        try:
            await self._reset_movement_motion(force=self._movement_motion_active)
            await self._cleanup_interrupted_action()
        except asyncio.CancelledError:
            self._movement_motion_active = False
            self._active_action_motion_name = None
            raise
        self._task = None
        self._stop_event = None
        self._target = None
        self._facing = None
        self._movement_motion_active = False
        self._clear_action()
        self._phase = BehaviorPhase.STOPPED
