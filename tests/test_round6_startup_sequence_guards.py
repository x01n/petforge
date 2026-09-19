from __future__ import annotations

import asyncio
from pathlib import Path

from app.runtime import ApplicationRuntime, build_runtime
from config.loader import LoadedConfiguration
from config.resources import inspect_resources
from core.asr import ASRHealth
from core.tts.contracts import EngineHealth

# 全部等待都带超时上界；关键时序用锁步事件，不依赖固定 sleep。
_TIMEOUT = 2.0


def _runtime(tmp_path: Path) -> ApplicationRuntime:
    """用关闭全部后台服务的配置构建运行时，避免触碰真实 TTS/ASR 资源。"""
    values = {
        "tts": {"enabled": False, "backend": "text_only", "language": "zh"},
        "asr": {"enabled": False},
        "scheduler": {"enabled": False, "activity": {"enabled": False}},
        "watcher": {"enabled": False},
        "behavior": {"enabled": False},
        "config": {"reload": {"enabled": False}},
    }
    configuration = LoadedConfiguration(tmp_path / "config.yaml", values)
    return build_runtime(configuration, inspect_resources(tmp_path / "resources"))


def _install_fakes(runtime: ApplicationRuntime, tts: FakeTTS, asr: FakeASR) -> None:
    """把 TTS/ASR 替换为可控 fake；启动路径只消费这两个对象。"""
    runtime.tts = tts
    runtime.asr = asr


class FakeTTS:
    """可控 TTS fake：start() 进入后阻塞于 release，允许测试锁步观察。"""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.start_calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.failure: BaseException | None = None

    async def start(self) -> EngineHealth:
        self.start_calls += 1
        self.order.append("tts:enter")
        self.entered.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        self.order.append("tts:return")
        return EngineHealth("tts", True, "ready")

    async def aclose(self) -> None:
        return None


class FakeASR:
    """可控 ASR fake：与 FakeTTS 共享 order 列表以断言串行启动顺序。"""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.start_calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def start(self) -> ASRHealth:
        self.start_calls += 1
        self.order.append("asr:enter")
        self.entered.set()
        await self.release.wait()
        self.order.append("asr:return")
        self.finished.set()
        return ASRHealth("ready", True, "sensevoice", "SenseVoiceSmall", True, "cpu", "zh", "ready")

    async def aclose(self) -> None:
        return None


def test_background_start_runs_tts_before_asr(tmp_path: Path) -> None:
    """TTS 完成前 ASR 不得开始；顺序在共享事件序列中严格可观察。"""

    runtime = _runtime(tmp_path)
    order: list[str] = []
    tts = FakeTTS(order)
    asr = FakeASR(order)
    _install_fakes(runtime, tts, asr)

    async def scenario() -> None:
        await runtime.start_background()
        await asyncio.wait_for(tts.entered.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter"]
        assert asr.start_calls == 0

        tts.release.set()
        await asyncio.wait_for(asr.entered.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter", "tts:return", "asr:enter"]
        assert order.index("tts:return") < order.index("asr:enter")
        assert order.index("tts:enter") < order.index("asr:enter")

        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter", "tts:return", "asr:enter", "asr:return"]
        await runtime.close()

    asyncio.run(scenario())


def test_asr_startup_health_queued_then_loading_then_ready(tmp_path: Path) -> None:
    """asr_startup_health 状态机：queued -> loading -> ready 的成功路径。"""

    runtime = _runtime(tmp_path)
    tts = FakeTTS([])
    asr = FakeASR([])
    _install_fakes(runtime, tts, asr)

    async def scenario() -> None:
        await runtime.start_background()
        await asyncio.wait_for(tts.entered.wait(), timeout=_TIMEOUT)
        queued = dict(runtime.asr_startup_health)
        assert queued["status"] == "queued"
        assert queued["pending"] is True
        assert queued["message"] == "ASR is queued after TTS initialization"
        assert runtime.tts_startup_health["status"] == "loading"

        tts.release.set()
        await asyncio.wait_for(asr.entered.wait(), timeout=_TIMEOUT)
        loading = dict(runtime.asr_startup_health)
        assert loading["status"] == "loading"
        assert loading["pending"] is True
        assert loading["ready"] is False
        assert runtime.tts_startup_health["status"] == "ready"
        assert runtime.tts_startup_health["pending"] is False

        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)
        ready = dict(runtime.asr_startup_health)
        assert ready["status"] == "ready"
        assert ready["available"] is True
        assert ready["ready"] is True
        assert ready["model_loaded"] is True
        assert "pending" not in ready
        await runtime.close()

    asyncio.run(scenario())


def test_tts_start_failure_still_releases_asr_to_ready(tmp_path: Path) -> None:
    """TTS 启动异常转为 unavailable 健康回执后，ASR 照常完成加载。"""

    runtime = _runtime(tmp_path)
    order: list[str] = []
    tts = FakeTTS(order)
    asr = FakeASR(order)
    tts.failure = RuntimeError("backend exploded")
    _install_fakes(runtime, tts, asr)

    async def scenario() -> None:
        await runtime.start_background()
        await asyncio.wait_for(tts.entered.wait(), timeout=_TIMEOUT)
        tts.release.set()
        await asyncio.wait_for(asr.entered.wait(), timeout=_TIMEOUT)
        tts_health = dict(runtime.tts_startup_health)
        assert tts_health["status"] == "unavailable"
        assert tts_health["available"] is False
        assert "TTS initialization failed: RuntimeError" in str(tts_health["message"])

        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter", "asr:enter", "asr:return"]
        assert dict(runtime.asr_startup_health)["status"] == "ready"
        await runtime.close()

    asyncio.run(scenario())


def test_tts_start_cancellation_releases_asr(tmp_path: Path) -> None:
    """TTS 启动代次被取消时，未被取消的 ASR 等待者吞掉取消并继续加载。

    对应 _initialize_asr_after_tts 的实现语义：TTS 任务以取消终态结束时，
    shield 会把 CancelledError 传播给 ASR wrapper；只有 wrapper 自身正在取消
    或运行时已关闭才 re-raise，否则放行并进入 ASR 队列阶段。
    """

    runtime = _runtime(tmp_path)
    order: list[str] = []
    tts = FakeTTS(order)
    asr = FakeASR(order)
    _install_fakes(runtime, tts, asr)

    async def scenario() -> None:
        await runtime.start_background()
        await asyncio.wait_for(tts.entered.wait(), timeout=_TIMEOUT)
        assert dict(runtime.asr_startup_health)["status"] == "queued"

        # 走生产语义：close/热替换路径通过 _cancel_tts_initialization 取消代次。
        await runtime._cancel_tts_initialization()
        assert runtime._tts_start_task is None
        # 被取消的 TTS 代次不重写健康字段，保留启动事务写入的 loading。
        assert dict(runtime.tts_startup_health)["status"] == "loading"

        await asyncio.wait_for(asr.entered.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter", "asr:enter"]
        assert dict(runtime.asr_startup_health)["status"] == "loading"

        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)
        assert dict(runtime.asr_startup_health)["status"] == "ready"
        await runtime.close()

    asyncio.run(scenario())


def test_start_background_is_single_flight(tmp_path: Path) -> None:
    """重复调用 start_background 不重复创建 TTS/ASR 启动任务。"""

    runtime = _runtime(tmp_path)
    order: list[str] = []
    tts = FakeTTS(order)
    asr = FakeASR(order)
    _install_fakes(runtime, tts, asr)

    async def scenario() -> None:
        await runtime.start_background()
        await asyncio.wait_for(tts.entered.wait(), timeout=_TIMEOUT)
        tts_task = runtime._tts_start_task
        asr_task = runtime._asr_start_task
        assert tts.start_calls == 1

        # 首次启动事务仍进行中时重复调用：直接返回，不创建新代次。
        await runtime.start_background()
        assert tts.start_calls == 1
        assert asr.start_calls == 0
        assert runtime._tts_start_task is tts_task
        assert runtime._asr_start_task is asr_task

        tts.release.set()
        await asyncio.wait_for(asr.entered.wait(), timeout=_TIMEOUT)
        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)

        # 全部完成后的重复调用同样不重建任务。
        await runtime.start_background()
        assert tts.start_calls == 1
        assert asr.start_calls == 1
        assert runtime._tts_start_task is tts_task
        assert runtime._asr_start_task is asr_task
        await runtime.close()

    asyncio.run(scenario())
