from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pytest

from app.runtime import MutablePetController, build_runtime, validate_runtime_configuration
from config.loader import ConfigurationError, LoadedConfiguration
from config.resources import inspect_resources
from services.scheduler.behavior import AutonomousMovementConfig, BehaviorAction, BehaviorService
from services.tools.builtins import register_builtin_tools
from services.tools.executor import ToolExecutionService
from services.tools.permissions import PermissionService
from services.tools.registry import ToolRegistry
from services.tools.types import (
    BEHAVIOR_INTERNAL_METADATA_KEY,
    RiskLevel,
    ToolCallContext,
    ToolSpec,
)


class _Executor:
    def __init__(self, *, risk: RiskLevel = RiskLevel.LOW) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.registry = type("Registry", (), {})()
        specs = {
            "pet:autonomous_move": ToolSpec(
                "pet:autonomous_move",
                "move",
                {"type": "object"},
                lambda *_args: None,
                RiskLevel.LOW,
            ),
            "pet:play_motion": ToolSpec(
                "pet:play_motion",
                "motion",
                {"type": "object"},
                lambda *_args: None,
                RiskLevel.LOW,
            ),
            "pet:unsafe": ToolSpec(
                "pet:unsafe",
                "unsafe",
                {"type": "object"},
                lambda *_args: None,
                risk,
            ),
        }
        self.registry.get = specs.get  # type: ignore[attr-defined]

    async def execute(self, *, identity, arguments, **_kwargs):
        self.calls.append((identity, dict(arguments)))
        return {"status": "completed", "identity": identity}

    def clear_turn(self, _context: ToolCallContext) -> None:
        return None


class _HangingMotionExecutor(_Executor):
    """模拟 Qt 事件循环已停止后永远收不到动作回执的执行器。"""

    async def execute(self, *, identity, arguments, **_kwargs):
        self.calls.append((identity, dict(arguments)))
        await asyncio.Event().wait()


def _movement(**kwargs) -> AutonomousMovementConfig:
    values = {
        "enabled": True,
        "min_distance_px": 80,
        "max_distance_px": 80,
        "step_pixels": 20,
        "step_interval_seconds": 0.02,
        "moving_motion": "",
        "idle_motion": "",
    }
    values.update(kwargs)
    return AutonomousMovementConfig(**values)


def test_stop_bounds_idle_motion_cleanup_when_dispatcher_is_closed() -> None:
    """宿主退出时动作收尾超时也不能阻塞 RuntimeLoop.stop。"""

    behavior = BehaviorService(_HangingMotionExecutor(), movement=_movement())
    behavior._movement_motion_active = True
    behavior._movement_motion_idle_name = "idle"

    async def scenario() -> None:
        await asyncio.wait_for(behavior.stop(), timeout=1.0)

    asyncio.run(scenario())
    assert behavior.status()["phase"] == "stopped"


def test_autonomous_wander_interpolates_and_clamps_to_bounds() -> None:
    executor = _Executor()
    directions: list[str] = []
    behavior = BehaviorService(
        executor,
        probability=1.0,
        random_source=random.Random(7),
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
        direction_setter=directions.append,
    )

    result = asyncio.run(behavior.tick())

    status = result.get("status") if isinstance(result, dict) else getattr(result, "status", "")
    assert status == "completed"
    moves = [
        arguments for identity, arguments in executor.calls if identity == "pet:autonomous_move"
    ]
    assert len(moves) >= 2
    assert moves[0] != moves[-1]
    assert all(0 <= int(item["x"]) <= 180 and 0 <= int(item["y"]) <= 180 for item in moves)
    assert behavior.status()["movement_count"] == 1
    assert behavior.status()["phase"] == "idle"
    assert directions and directions[0] == "A"


def test_autonomous_facing_uses_left_a_and_right_b() -> None:
    executor = _Executor()
    directions: list[str] = []
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
        direction_setter=directions.append,
    )

    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]
    asyncio.run(behavior.tick())
    assert directions == ["A"]

    directions.clear()
    behavior._plan_target = lambda _current, _bounds: (170, 100)  # type: ignore[method-assign]
    asyncio.run(behavior.tick())
    assert directions == ["B"]


def test_autonomous_async_facing_uses_final_setter_result() -> None:
    """异步方向 setter 拒绝时不能缓存方向，也不能阻断位置移动。"""

    executor = _Executor()
    directions: list[str] = []

    async def reject(_direction: str) -> bool:
        directions.append(_direction)
        await asyncio.sleep(0)
        return False

    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
        direction_setter=reject,
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    asyncio.run(behavior.tick())

    assert directions == ["A"]
    assert behavior._facing is None
    assert any(identity == "pet:autonomous_move" for identity, _ in executor.calls)


def test_autonomous_async_facing_is_not_cached_after_interrupt() -> None:
    """异步转身等待期间被打断时，旧方向不能写回行为状态。"""

    executor = _Executor()
    started = asyncio.Event()
    release = asyncio.Event()
    directions: list[str] = []

    async def delayed(direction: str) -> bool:
        directions.append(direction)
        started.set()
        await release.wait()
        return True

    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
        direction_setter=delayed,
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    async def scenario() -> object:
        task = asyncio.create_task(behavior.tick())
        await started.wait()
        behavior.interrupt()
        release.set()
        return await task

    result = asyncio.run(scenario())
    assert isinstance(result, dict)
    assert result["status"] == "cancelled"
    assert directions == ["A"]
    assert behavior._facing is None


def test_autonomous_motion_failure_does_not_block_movement() -> None:
    """模型不支持 configured moving_motion 时仍可安全完成窗口移动。"""

    class MotionRejectingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            self.calls.append((identity, dict(arguments)))
            if identity == "pet:play_motion":
                return {"status": "unavailable", "reason": "unsupported motion"}
            return {"status": "completed", "identity": identity}

    executor = MotionRejectingExecutor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(moving_motion="unsupported", idle_motion=""),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    result = asyncio.run(behavior.tick())

    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert any(identity == "pet:autonomous_move" for identity, _ in executor.calls)
    assert behavior.status()["movement_count"] == 1
    assert "motion failed" in str(behavior.status()["last_error"])


def test_autonomous_tick_does_not_stack_action_after_movement() -> None:
    """一次 tick 只提交漫游或随机动作，避免渲染状态连续覆盖。"""

    class FixedRandom:
        def random(self) -> float:
            return 0.0

        def uniform(self, low: float, high: float) -> float:
            return (low + high) / 2

    executor = _Executor()
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:play_motion", {"name": "blink"}, 1.0),),
        probability=1.0,
        random_source=FixedRandom(),  # type: ignore[arg-type]
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )

    asyncio.run(behavior.tick())

    identities = [identity for identity, _ in executor.calls]
    assert "pet:autonomous_move" in identities
    assert identities.count("pet:play_motion") == 0


def test_autonomous_tick_uses_one_probability_gate_for_both_behavior_kinds() -> None:
    """漫游和动作共用一次触发概率，不再把概率放大为两次抽样。"""

    class SequenceRandom:
        def __init__(self) -> None:
            self.calls = 0

        def random(self) -> float:
            self.calls += 1
            # p=0.5 时 unit=0.8；在漫游权重 1、动作权重 1 的条件
            # 分布中应选择动作。第二个值若被读取会暴露旧的双重抽样。
            return 0.4 if self.calls == 1 else 0.99

        def uniform(self, low: float, high: float) -> float:
            return (low + high) / 2

    executor = _Executor()
    source = SequenceRandom()
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:play_motion", {"name": "blink"}, 1.0),),
        probability=0.5,
        random_source=source,  # type: ignore[arg-type]
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )

    result = asyncio.run(behavior.tick())

    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert [identity for identity, _ in executor.calls] == ["pet:play_motion"]
    assert source.calls == 1


def test_successful_random_motion_returns_to_idle_after_bounded_hold() -> None:
    """成功的随机 motion 不能因正式 Loop 动作而永久循环。"""

    executor = _Executor()
    behavior = BehaviorService(
        executor,
        actions=(
            BehaviorAction(
                "pet:play_motion",
                {"name": "blink"},
                1.0,
                duration_seconds=0.0,
            ),
        ),
        probability=1.0,
        random_source=random.Random(0),
        movement=_movement(enabled=False, idle_motion="idle"),
    )

    result = asyncio.run(behavior.tick())

    assert isinstance(result, dict)
    assert result["status"] == "completed"
    motions = [
        arguments["name"] for identity, arguments in executor.calls if identity == "pet:play_motion"
    ]
    assert motions == ["blink", "idle"]
    assert behavior.status()["active_action"] is None


def test_random_motion_hold_is_bounded_and_configurable() -> None:
    """动作保持时长被规整到 0～30 秒，避免配置拖住行为循环。"""

    short = BehaviorAction("pet:play_motion", {"name": "wave"}, duration_seconds=-2)
    long = BehaviorAction("pet:play_motion", {"name": "wave"}, duration_seconds=999)
    assert short.duration_seconds == 0.0
    assert long.duration_seconds == 30.0


def test_autonomous_tick_drops_a_stale_concurrent_call() -> None:
    """长漫游期间到达的第二个 tick 不能排队到下一次继续乱动。"""

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            if identity == "pet:autonomous_move":
                started.set()
                await release.wait()
            return await super().execute(identity=identity, arguments=arguments, **kwargs)

    executor = BlockingExecutor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(step_pixels=20),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    async def scenario():
        first = asyncio.create_task(behavior.tick())
        await started.wait()
        second = await behavior.tick()
        release.set()
        return await first, second

    first, second = asyncio.run(scenario())

    assert isinstance(first, dict)
    assert second is None
    assert behavior.status()["busy_tick_count"] == 1
    moves = [identity for identity, _ in executor.calls if identity == "pet:autonomous_move"]
    # 80px 目标在速度上限下会拆成多个步进；关键断言是只有首个
    # tick 产生这一条步进序列，第二个并发 tick 没有追加另一条规划。
    assert moves


def test_autonomous_exception_converges_to_idle_without_stale_target() -> None:
    class FailingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            if identity == "pet:autonomous_move":
                raise RuntimeError("window move failed")
            return await super().execute(identity=identity, arguments=arguments, **kwargs)

    executor = FailingExecutor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    result = asyncio.run(behavior.tick())

    assert isinstance(result, dict)
    assert result["status"] == "failed"
    assert behavior.status()["phase"] == "idle"
    assert behavior.status()["target"] is None


def test_autonomous_interaction_probe_failure_pauses_fail_closed() -> None:
    executor = _Executor()

    def broken_probe() -> bool:
        raise RuntimeError("probe unavailable")

    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
        interaction_provider=broken_probe,
    )

    assert asyncio.run(behavior.tick()) is None
    assert behavior.status()["phase"] == "paused"
    assert executor.calls == []


def test_autonomous_interaction_release_keeps_a_short_cooldown() -> None:
    """交互刚结束时不应立即重新抢占动作或窗口位置。"""

    executor = _Executor()
    now = [100.0]
    interacting = [True]
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 300, 300),
        interaction_provider=lambda: interacting[0],
        interaction_cooldown_seconds=0.35,
        clock=lambda: now[0],
    )

    assert asyncio.run(behavior.tick()) is None
    interacting[0] = False
    # 释放后的短窗口仍然闭锁，不能产生任何移动调用。
    assert asyncio.run(behavior.tick()) is None
    assert executor.calls == []

    now[0] += 0.36
    result = asyncio.run(behavior.tick())
    assert isinstance(result, dict)
    assert result.get("status") == "completed"
    assert any(identity == "pet:autonomous_move" for identity, _ in executor.calls)


def test_explicit_interrupt_keeps_cooldown_without_interaction_probe() -> None:
    """没有宿主探针时，显式中断也不能让行为立即重新抢占。"""

    executor = _Executor()
    now = [10.0]
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 300, 300),
        interaction_cooldown_seconds=0.35,
        clock=lambda: now[0],
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    behavior.interrupt()
    assert asyncio.run(behavior.tick()) is None
    assert executor.calls == []

    now[0] += 0.36
    result = asyncio.run(behavior.tick())
    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert any(identity == "pet:autonomous_move" for identity, _ in executor.calls)


def test_autonomous_unavailable_interaction_probe_pauses_fail_closed() -> None:
    """宿主返回不可用回执时不能被解释成未交互。"""

    executor = _Executor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 300, 300),
        interaction_provider=lambda: {"status": "unavailable"},
    )

    assert asyncio.run(behavior.tick()) is None
    assert behavior.status()["phase"] == "paused"
    assert executor.calls == []


def test_autonomous_movement_config_normalizes_non_finite_values() -> None:
    movement = AutonomousMovementConfig(
        enabled=True,
        min_distance_px=float("inf"),
        max_distance_px=float("nan"),
        step_pixels=float("inf"),
        step_interval_seconds=float("nan"),
        max_steps=float("inf"),  # type: ignore[arg-type]
    )

    assert movement.min_distance_px == 80.0
    assert movement.max_distance_px == 280.0
    assert movement.step_pixels == 36.0
    assert movement.step_interval_seconds == 0.08
    assert movement.max_steps == 64


def test_autonomous_movement_caps_step_speed_to_a_finite_rate() -> None:
    movement = AutonomousMovementConfig(
        enabled=True,
        step_pixels=100,
        step_interval_seconds=0.1,
        max_speed_px_per_second=180,
    )
    assert movement.max_speed_px_per_second == 180.0

    executor = _Executor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=movement,
        position_provider=lambda: (0, 0),
        bounds_provider=lambda: (0, 0, 200, 200),
    )
    behavior._plan_target = lambda _current, _bounds: (100, 0)  # type: ignore[method-assign]
    asyncio.run(behavior.tick())
    points = [
        (int(arguments["x"]), int(arguments["y"]))
        for identity, arguments in executor.calls
        if identity == "pet:autonomous_move"
    ]
    assert points
    assert (
        max(
            (points[0][0], points[0][1]),
        )
        <= 18
    )


def test_autonomous_speed_cap_survives_max_step_limit(monkeypatch) -> None:
    """达到 max_steps 上限时通过延长间隔维持速度上限。"""

    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(float(delay))

    monkeypatch.setattr("services.scheduler.behavior.asyncio.sleep", record_sleep)
    movement = AutonomousMovementConfig(
        enabled=True,
        min_distance_px=1000,
        max_distance_px=1000,
        step_pixels=100,
        step_interval_seconds=0.02,
        max_speed_px_per_second=20,
        max_steps=4,
        moving_motion="",
        idle_motion="",
    )
    executor = _Executor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=movement,
        position_provider=lambda: (0, 0),
        bounds_provider=lambda: (0, 0, 1200, 100),
    )
    behavior._plan_target = lambda _current, _bounds: (1000, 0)  # type: ignore[method-assign]

    asyncio.run(behavior.tick())

    points = [
        (int(arguments["x"]), int(arguments["y"]))
        for identity, arguments in executor.calls
        if identity == "pet:autonomous_move"
    ]
    assert points == [(250, 0), (500, 0), (750, 0), (1000, 0)]
    assert max(delays) <= 0.1
    assert sum(delays) == pytest.approx(37.5)


def test_autonomous_wait_checks_interrupt_before_the_next_speed_limited_step(monkeypatch) -> None:
    """低速长路径的等待窗口最多延迟一个短轮询片才响应用户交互。"""

    sleep_started = asyncio.Event()
    release_sleep = asyncio.Event()

    async def controlled_sleep(_delay: float) -> None:
        sleep_started.set()
        await release_sleep.wait()

    monkeypatch.setattr("services.scheduler.behavior.asyncio.sleep", controlled_sleep)
    executor = _Executor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(
            step_pixels=100,
            step_interval_seconds=0.1,
            max_speed_px_per_second=100,
        ),
        position_provider=lambda: (0, 0),
        bounds_provider=lambda: (0, 0, 200, 100),
    )
    behavior._plan_target = lambda _current, _bounds: (100, 0)  # type: ignore[method-assign]

    async def scenario():
        task = asyncio.create_task(behavior.tick())
        await sleep_started.wait()
        behavior.interrupt()
        release_sleep.set()
        return await task

    result = asyncio.run(scenario())

    assert isinstance(result, dict)
    assert result["status"] == "cancelled"
    moves = [identity for identity, _ in executor.calls if identity == "pet:autonomous_move"]
    assert len(moves) == 1


def test_behavior_action_exception_converges_to_idle() -> None:
    class FailingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            if identity == "pet:play_motion":
                raise RuntimeError("renderer failed")
            return await super().execute(identity=identity, arguments=arguments, **kwargs)

    behavior = BehaviorService(
        FailingExecutor(),
        actions=(BehaviorAction("pet:play_motion", {"name": "blink"}, 1.0),),
        probability=1.0,
        random_source=random.Random(0),
    )

    result = asyncio.run(behavior.tick())

    assert isinstance(result, dict)
    assert result["status"] == "failed"
    assert behavior.status()["phase"] == "idle"
    assert behavior.status()["target"] is None


def test_autonomous_exception_resets_accepted_moving_motion() -> None:
    class FailingAfterMotionExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            self.calls.append((identity, dict(arguments)))
            if identity == "pet:autonomous_move":
                raise RuntimeError("window move failed")
            return {"status": "completed", "identity": identity}

    executor = FailingAfterMotionExecutor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(moving_motion="walk", idle_motion="idle"),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    result = asyncio.run(behavior.tick())

    assert isinstance(result, dict)
    assert result["status"] == "failed"
    motions = [
        arguments["name"] for identity, arguments in executor.calls if identity == "pet:play_motion"
    ]
    assert motions == ["walk", "idle"]
    assert behavior._movement_motion_active is False


def test_autonomous_interruption_clears_target_after_inflight_move() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            if identity == "pet:autonomous_move":
                started.set()
                await release.wait()
            return await super().execute(identity=identity, arguments=arguments, **kwargs)

    executor = BlockingExecutor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    async def scenario():
        task = asyncio.create_task(behavior.tick())
        await started.wait()
        behavior.interrupt()
        release.set()
        return await task

    result = asyncio.run(scenario())

    assert isinstance(result, dict)
    assert result["status"] == "cancelled"
    assert behavior.status()["phase"] == "paused"
    assert behavior.status()["target"] is None


def test_autonomous_interruption_resets_an_inflight_random_motion() -> None:
    """用户交互打断随机动作时，已提交的 motion 必须回到 idle。"""

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            self.calls.append((identity, dict(arguments)))
            if identity == "pet:play_motion" and arguments.get("name") == "wave":
                started.set()
                await release.wait()
            return {"status": "completed", "identity": identity}

    executor = BlockingExecutor()
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:play_motion", {"name": "wave"}, 1.0),),
        probability=1.0,
        random_source=random.Random(0),
        movement=_movement(enabled=False, idle_motion="idle"),
    )

    async def scenario():
        task = asyncio.create_task(behavior.tick())
        await started.wait()
        behavior.interrupt()
        release.set()
        return await task

    result = asyncio.run(scenario())

    assert isinstance(result, dict)
    assert result["status"] == "cancelled"
    motions = [
        arguments["name"] for identity, arguments in executor.calls if identity == "pet:play_motion"
    ]
    assert motions == ["wave", "idle"]
    assert behavior.status()["phase"] == "paused"
    assert behavior.status()["active_action"] is None


def test_autonomous_interruption_keeps_action_idle_snapshot_after_reconfigure() -> None:
    """动作等待期间热重载时仍按启动时的 idle 名称收尾。"""

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            self.calls.append((identity, dict(arguments)))
            if identity == "pet:play_motion" and arguments.get("name") == "wave":
                started.set()
                await release.wait()
            return {"status": "completed", "identity": identity}

    executor = BlockingExecutor()
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:play_motion", {"name": "wave"}, 1.0),),
        probability=1.0,
        random_source=random.Random(0),
        movement=_movement(enabled=False, idle_motion="idle-before"),
    )

    async def scenario() -> object:
        task = asyncio.create_task(behavior.tick())
        await started.wait()
        behavior.reconfigure(movement=_movement(enabled=False, idle_motion=""))
        release.set()
        return await task

    result = asyncio.run(scenario())

    assert isinstance(result, dict)
    assert result["status"] == "cancelled"
    motions = [
        arguments["name"] for identity, arguments in executor.calls if identity == "pet:play_motion"
    ]
    assert motions == ["wave", "idle-before"]


def test_autonomous_cancelling_a_random_motion_resets_before_propagating() -> None:
    """外部取消随机动作时先收尾，再把 CancelledError 交给调用方。"""

    started = asyncio.Event()

    class BlockingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            self.calls.append((identity, dict(arguments)))
            if identity == "pet:play_motion" and arguments.get("name") == "wave":
                started.set()
                await asyncio.Event().wait()
            return {"status": "completed", "identity": identity}

    executor = BlockingExecutor()
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:play_motion", {"name": "wave"}, 1.0),),
        probability=1.0,
        random_source=random.Random(0),
        movement=_movement(enabled=False, idle_motion="idle"),
    )

    async def scenario():
        task = asyncio.create_task(behavior.tick())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    motions = [
        arguments["name"] for identity, arguments in executor.calls if identity == "pet:play_motion"
    ]
    assert motions == ["wave", "idle"]
    assert behavior.status()["active_action"] is None


def test_autonomous_cancellation_resets_an_accepted_walking_motion() -> None:
    """取消漫游时必须停止已启动的 walk，不能留下持续抖动的动作状态。"""

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            self.calls.append((identity, dict(arguments)))
            if identity == "pet:autonomous_move":
                started.set()
                await release.wait()
            return {"status": "completed", "identity": identity}

    executor = BlockingExecutor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(moving_motion="walk", idle_motion="idle"),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    async def scenario() -> None:
        task = asyncio.create_task(behavior.tick())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    motions = [
        arguments["name"] for identity, arguments in executor.calls if identity == "pet:play_motion"
    ]
    assert motions == ["walk", "idle"]
    assert behavior._movement_motion_active is False


def test_autonomous_cancellation_during_walking_motion_resets_idle() -> None:
    """walk 在回执前被取消时也必须回到 idle。"""

    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingMotionExecutor(_Executor):
        async def execute(self, *, identity, arguments, **kwargs):
            self.calls.append((identity, dict(arguments)))
            if identity == "pet:play_motion" and arguments.get("name") == "walk":
                started.set()
                await release.wait()
            return {"status": "completed", "identity": identity}

    executor = BlockingMotionExecutor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        movement=_movement(moving_motion="walk", idle_motion="idle"),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 180, 180),
    )
    behavior._plan_target = lambda _current, _bounds: (20, 100)  # type: ignore[method-assign]

    async def scenario() -> None:
        task = asyncio.create_task(behavior.tick())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    motions = [
        arguments["name"] for identity, arguments in executor.calls if identity == "pet:play_motion"
    ]
    assert motions == ["walk", "idle"]
    assert behavior._movement_motion_active is False
    assert behavior._movement_motion_idle_name is None


def test_autonomous_tick_serializes_concurrent_plans() -> None:
    """并发触发行为时，两个游走目标不得交错提交窗口坐标。"""

    class SlowExecutor(_Executor):
        active = 0
        max_active = 0

        async def execute(self, *, identity, arguments, **kwargs):
            if identity == "pet:autonomous_move":
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                await asyncio.sleep(0)
                self.active -= 1
            return await super().execute(identity=identity, arguments=arguments, **kwargs)

    executor = SlowExecutor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        random_source=random.Random(3),
        movement=_movement(step_pixels=12),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 300, 300),
    )

    async def scenario():
        return await asyncio.gather(behavior.tick(), behavior.tick())

    outcomes = asyncio.run(scenario())
    assert len(outcomes) == 2
    assert executor.max_active == 1


def test_autonomous_bounds_accept_qt_style_geometry() -> None:
    class Rect:
        def left(self):
            return 10

        def top(self):
            return 20

        def right(self):
            return 110

        def bottom(self):
            return 220

    assert BehaviorService._coerce_bounds(Rect()) == (10, 20, 110, 220)


def test_autonomous_wander_is_cancelled_by_user_interaction() -> None:
    executor = _Executor()
    behavior = BehaviorService(
        executor,
        probability=1.0,
        random_source=random.Random(1),
        movement=_movement(step_pixels=4),
        position_provider=lambda: (100, 100),
        bounds_provider=lambda: (0, 0, 300, 300),
        interaction_provider=lambda: len(executor.calls) >= 1,
    )

    result = asyncio.run(behavior.tick())

    assert isinstance(result, dict)
    assert result["status"] == "cancelled"
    assert behavior.status()["movement_count"] == 0
    assert len(executor.calls) == 1


def test_behavior_rejects_medium_or_high_risk_random_actions() -> None:
    executor = _Executor(risk=RiskLevel.MEDIUM)
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:unsafe", {}, 1.0),),
        probability=1.0,
        random_source=random.Random(0),
    )

    result = asyncio.run(behavior.tick())

    assert isinstance(result, dict)
    assert result["status"] == "denied"
    assert executor.calls == []
    assert "blocked unsafe behavior tool" in str(behavior.status()["last_error"])


def test_pet_controller_geometry_delegates_are_backward_compatible() -> None:
    class Target:
        def position(self):
            return (12, 34)

        def movement_bounds(self):
            return (0, 0, 100, 100)

        def is_user_interacting(self):
            return True

    controller = MutablePetController(Target())
    assert controller.position() == (12, 34)
    assert controller.movement_bounds() == (0, 0, 100, 100)
    assert controller.is_user_interacting() is True


def test_pet_controller_external_actions_interrupt_but_behavior_actions_do_not() -> None:
    """外部动作抢占自主行为，行为内部转发不应再次触发中断。"""

    class Target:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def position(self):
            return (100, 100)

        def movement_bounds(self):
            return (0, 0, 180, 180)

        def move_to(self, x, y, *, duration_ms=800):
            self.calls.append(("move", (x, y, duration_ms)))
            return {"status": "available", "x": x, "y": y}

        def set_expression(self, name):
            self.calls.append(("expression", name))
            return {"status": "available", "name": name}

        def play_motion(self, name):
            self.calls.append(("motion", name))
            return {"status": "available", "name": name}

    target = Target()
    interruptions: list[str] = []
    controller = MutablePetController(target)
    controller.set_interaction_interrupt(lambda: interruptions.append("interrupt"))

    assert controller.play_motion("angry")["status"] == "available"
    assert controller.set_expression("happy")["status"] == "available"
    assert controller.move_to(12, 34, duration_ms=0)["status"] == "available"
    assert interruptions == ["interrupt", "interrupt", "interrupt"]

    # 内部入口仍调用同一个目标，但不把动作/移动误判为新的用户交互。
    assert controller.play_motion_internal("blink")["status"] == "available"
    assert controller.set_expression_internal("neutral")["status"] == "available"
    assert controller.move_to_internal(56, 78, duration_ms=0)["status"] == "available"
    assert interruptions == ["interrupt", "interrupt", "interrupt"]


def test_behavior_tool_context_uses_internal_pet_forwarders() -> None:
    """行为服务经工具管线调用动作/移动时不会自我中断。"""

    class Target:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def position(self):
            return (100, 100)

        def movement_bounds(self):
            return (0, 0, 180, 180)

        def move_to(self, x, y, *, duration_ms=800):
            self.calls.append(("move", (x, y, duration_ms)))
            return {"status": "available", "x": x, "y": y}

        def set_expression(self, name):
            self.calls.append(("expression", name))
            return {"status": "available", "name": name}

        def play_motion(self, name):
            self.calls.append(("motion", name))
            return {"status": "available", "name": name}

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    target = Target()
    interruptions: list[str] = []
    controller = MutablePetController(target)
    controller.set_interaction_interrupt(lambda: interruptions.append("interrupt"))
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=controller)

    class RecordingExecutor(ToolExecutionService):
        def __init__(self) -> None:
            super().__init__(registry, PermissionService(bypass_approval=True))
            self.contexts: list[ToolCallContext] = []

        async def execute(self, **kwargs):
            self.contexts.append(kwargs["context"])
            return await super().execute(**kwargs)

    executor = RecordingExecutor()

    async def scenario() -> None:
        action_behavior = BehaviorService(
            executor,
            actions=(BehaviorAction("pet:play_motion", {"name": "blink"}, 1.0, 0.0),),
            probability=1.0,
            movement=AutonomousMovementConfig(enabled=False),
        )
        action_result = await action_behavior.tick()
        assert getattr(action_result, "status", "") == "completed"

        movement_behavior = BehaviorService(
            executor,
            probability=1.0,
            movement=AutonomousMovementConfig(
                enabled=True,
                min_distance_px=80,
                max_distance_px=80,
                step_pixels=20,
                step_interval_seconds=0.02,
                moving_motion="",
                idle_motion="",
            ),
            position_provider=controller.position,
            bounds_provider=controller.movement_bounds,
        )
        movement_result = await movement_behavior.tick()
        assert movement_result is not None

    asyncio.run(scenario())
    assert interruptions == []
    assert executor.contexts
    assert all(
        context.metadata.get(BEHAVIOR_INTERNAL_METADATA_KEY) is True
        for context in executor.contexts
    )
    assert any(kind == "motion" and value == "blink" for kind, value in target.calls)
    assert any(kind == "move" and value[2] == 0 for kind, value in target.calls)

    # 普通控制器入口代表模型/控制台/用户动作，必须抢占当前自主行为。
    assert controller.play_motion("angry")["status"] == "available"
    assert controller.set_expression("happy")["status"] == "available"
    assert controller.move_to(12, 34, duration_ms=0)["status"] == "available"
    assert interruptions == ["interrupt", "interrupt", "interrupt"]


@pytest.mark.parametrize("external_action", ("motion", "expression", "move"))
def test_external_action_does_not_get_overwritten_by_stale_behavior_idle(
    external_action: str,
) -> None:
    """外部动作抢占时，旧行为收尾不能再发送 idle 覆盖新状态。"""

    class Target:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []
            self.motion_started = asyncio.Event()

        def play_motion(self, name):
            self.calls.append(("motion", name))
            if name == "blink":
                self.motion_started.set()
            return {"status": "available", "name": name}

        def set_expression(self, name):
            self.calls.append(("expression", name))
            return {"status": "available", "name": name}

        def move_to(self, x, y, *, duration_ms=800):
            self.calls.append(("move", (x, y, duration_ms)))
            return {"status": "available", "x": x, "y": y}

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    target = Target()
    controller = MutablePetController(target)
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=controller)
    executor = ToolExecutionService(registry, PermissionService(bypass_approval=True))
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:play_motion", {"name": "blink"}, 1.0, 0.4),),
        probability=1.0,
        movement=AutonomousMovementConfig(enabled=False, idle_motion="idle"),
        motion_generation_provider=controller.motion_owner_generation,
    )
    controller.set_interaction_interrupt(behavior.interrupt)

    async def scenario() -> object:
        task = asyncio.create_task(behavior.tick())
        await target.motion_started.wait()
        if external_action == "motion":
            external = controller.play_motion("angry")
        elif external_action == "expression":
            external = controller.set_expression("happy")
        else:
            external = controller.move_to(12, 34, duration_ms=0)
        result = await asyncio.wait_for(task, timeout=1.0)
        return external, result

    external, result = asyncio.run(scenario())
    assert external["status"] == "available"
    assert isinstance(result, dict)
    assert result["status"] == "cancelled"
    if external_action == "motion":
        # 外部 motion 已替换行为动作，旧收尾不得再覆盖 angry。
        assert not any(kind == "motion" and value == "idle" for kind, value in target.calls)
        assert target.calls == [("motion", "blink"), ("motion", "angry")]
    elif external_action == "expression":
        # 表情没有替换当前 motion；渲染器仍由行为持有时，正常收尾仍应回到 idle。
        assert target.calls[-1] == ("motion", "idle")
        assert target.calls == [
            ("motion", "blink"),
            ("expression", "happy"),
            ("motion", "idle"),
        ]
    else:
        # 位置移动同样不替换 motion；行为所有权仍在时允许 idle 收尾。
        assert target.calls[-1] == ("motion", "idle")
        assert target.calls == [
            ("motion", "blink"),
            ("move", (12.0, 34.0, 0)),
            ("motion", "idle"),
        ]


def test_behavior_owned_motion_still_resets_to_idle_with_generation_probe() -> None:
    """没有外部抢占时，代次匹配的行为动作仍正常回到 idle。"""

    class Target:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def play_motion(self, name):
            self.calls.append(name)
            return {"status": "available", "name": name}

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    target = Target()
    controller = MutablePetController(target)
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=controller)
    executor = ToolExecutionService(registry, PermissionService(bypass_approval=True))
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:play_motion", {"name": "blink"}, 1.0, 0.0),),
        probability=1.0,
        movement=AutonomousMovementConfig(enabled=False, idle_motion="idle"),
        motion_generation_provider=controller.motion_owner_generation,
    )

    result = asyncio.run(behavior.tick())

    assert getattr(result, "status", "") == "completed"
    assert target.calls == ["blink", "idle"]


def test_queued_internal_motion_is_dropped_after_external_preemption() -> None:
    """Qt 队列中的旧行为动作不能在外部动作之后迟到写回。"""

    class Target:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def play_motion(self, name):
            self.calls.append(str(name))
            return {"status": "available", "name": name}

    class QueueDispatcher:
        def __init__(self) -> None:
            self.pending: list[tuple[object, tuple[object, ...], dict[str, object], object]] = []
            self.executing = False

        def invoke(self, function, *args, **kwargs):
            if self.executing:
                return function(*args, **kwargs)
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            self.pending.append((function, args, kwargs, future))

            async def wait_result():
                return await future

            return wait_result()

        def flush(self, index: int) -> None:
            function, args, kwargs, future = self.pending.pop(index)
            self.executing = True
            try:
                result = function(*args, **kwargs)
            finally:
                self.executing = False
            future.set_result(result)

    target = Target()
    dispatcher = QueueDispatcher()
    controller = MutablePetController(target, dispatcher)

    async def scenario() -> tuple[object, object]:
        old_internal = controller.play_motion_internal("blink")
        external = controller.play_motion("angry")
        # 外部动作先在宿主队列中落地，旧内部调用随后才有机会执行。
        dispatcher.flush(1)
        external_result = await external
        dispatcher.flush(0)
        internal_result = await old_internal
        return external_result, internal_result

    external_result, internal_result = asyncio.run(scenario())
    assert external_result["status"] == "available"
    assert internal_result["status"] == "cancelled"
    assert target.calls == ["angry"]


def test_autonomous_move_tool_is_hidden_and_low_risk() -> None:
    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    class Pet:
        def move_to(self, x, y, *, duration_ms=800):
            return {"status": "available", "x": x, "y": y, "duration_ms": duration_ms}

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=Pet())
    spec = registry.require("pet:autonomous_move")
    assert spec.public is False
    assert spec.risk is RiskLevel.LOW
    assert "pet:autonomous_move" not in {
        item.identity for item in registry.visible(("pet_control",))
    }
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="auto-move",
            identity="pet:autonomous_move",
            arguments={"x": 10, "y": 20, "duration_ms": 0},
            context=ToolCallContext("p", "s", "t", source="behavior"),
        )
    )
    assert outcome.status == "completed"


def test_runtime_wires_autonomous_movement_config(tmp_path: Path) -> None:
    values = {
        "behavior": {
            "enabled": True,
            "movement": {
                "enabled": True,
                "min_distance_px": 100,
                "max_distance_px": 220,
                "step_pixels": 24,
                "step_interval_seconds": 0.05,
                "movement_identity": "pet:autonomous_move",
                "max_steps": 32,
            },
        }
    }
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", values),
        inspect_resources(tmp_path / "resources"),
    )
    try:
        assert runtime.behavior.status()["movement_enabled"] is True
        assert "pet:autonomous_move" in runtime.registry.identities()
        assert callable(runtime.behavior._motion_generation_provider)
    finally:
        asyncio.run(runtime.close())


def test_runtime_rejects_invalid_autonomous_movement_bounds(tmp_path: Path) -> None:
    values = {"behavior": {"movement": {"min_distance_px": 300, "max_distance_px": 100}}}
    try:
        validate_runtime_configuration(LoadedConfiguration(tmp_path / "config.yaml", values))
    except ConfigurationError as exc:
        assert "max_distance_px" in str(exc)
    else:
        raise AssertionError("invalid autonomous movement bounds were accepted")


def test_runtime_rejects_fractional_autonomous_max_steps(tmp_path: Path) -> None:
    values = {"behavior": {"movement": {"max_steps": 1.5}}}
    try:
        validate_runtime_configuration(LoadedConfiguration(tmp_path / "config.yaml", values))
    except ConfigurationError as exc:
        assert "max_steps" in str(exc)
    else:
        raise AssertionError("fractional autonomous max_steps were accepted")


def test_runtime_rejects_rebinding_the_protected_movement_tool(tmp_path: Path) -> None:
    values = {"behavior": {"movement": {"movement_identity": "pet:move"}}}
    try:
        validate_runtime_configuration(LoadedConfiguration(tmp_path / "config.yaml", values))
    except ConfigurationError as exc:
        assert "pet:autonomous_move" in str(exc)
    else:
        raise AssertionError("autonomous movement tool rebinding was accepted")
