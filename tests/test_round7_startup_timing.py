from __future__ import annotations

import asyncio
import math
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


def _bounded_finite(value: object) -> float | None:
    """返回有限数值本身，非数值或 NaN/inf 返回 None。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class FakeTTS:
    """可控 TTS fake：start() 进入后阻塞于 release，允许测试锁步观察。"""

    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.failure: BaseException | None = None

    async def start(self) -> EngineHealth:
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
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def start(self) -> ASRHealth:
        self.order.append("asr:enter")
        self.entered.set()
        await self.release.wait()
        self.order.append("asr:return")
        self.finished.set()
        return ASRHealth("ready", True, "sensevoice", "SenseVoiceSmall", True, "cpu", "zh", "ready")

    async def aclose(self) -> None:
        return None


def test_startup_health_carries_timing_fields_on_all_phases(tmp_path: Path) -> None:
    """loading/queued 阶段按键齐全且耗时空值，且串行顺序不受观测影响。"""

    runtime = _runtime(tmp_path)
    order: list[str] = []
    tts = FakeTTS(order)
    asr = FakeASR(order)
    _install_fakes(runtime, tts, asr)

    async def scenario() -> None:
        await runtime.start_background()
        await asyncio.wait_for(tts.entered.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter"]
        assert asr.entered.is_set() is False

        tts_loading = dict(runtime.tts_startup_health)
        assert tts_loading["status"] == "loading"
        assert "started_at_ms" in tts_loading
        assert _bounded_finite(tts_loading["started_at_ms"]) is not None
        assert tts_loading["start_duration_ms"] is None

        asr_queued = dict(runtime.asr_startup_health)
        assert asr_queued["status"] == "queued"
        assert "started_at_ms" in asr_queued
        assert _bounded_finite(asr_queued["started_at_ms"]) is not None
        assert asr_queued["start_duration_ms"] is None

        tts.release.set()
        await asyncio.wait_for(asr.entered.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter", "tts:return", "asr:enter"]
        assert order.index("tts:return") < order.index("asr:enter")
        assert order.index("tts:enter") < order.index("asr:enter")

        tts_ready = dict(runtime.tts_startup_health)
        assert tts_ready["status"] == "ready"
        assert _bounded_finite(tts_ready["started_at_ms"]) is not None
        duration = _bounded_finite(tts_ready["start_duration_ms"])
        assert duration is not None
        assert duration > 0
        # 与 backend 自带 latency 同形：要么有限非负，要么干脆缺省。
        kept_latency = tts_ready.get("latency_ms")
        assert kept_latency is None or _bounded_finite(kept_latency) is not None

        asr_loading = dict(runtime.asr_startup_health)
        assert asr_loading["status"] == "loading"
        assert _bounded_finite(asr_loading["started_at_ms"]) is not None
        assert asr_loading["start_duration_ms"] is None

        asr_queued_origin = asr_queued["started_at_ms"]
        assert _bounded_finite(asr_loading["started_at_ms"]) >= asr_queued_origin

        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter", "tts:return", "asr:enter", "asr:return"]

        asr_ready = dict(runtime.asr_startup_health)
        assert asr_ready["status"] == "ready"
        assert _bounded_finite(asr_ready["started_at_ms"]) is not None
        asr_duration = _bounded_finite(asr_ready["start_duration_ms"])
        assert asr_duration is not None
        assert asr_duration > 0
        await runtime.close()

    asyncio.run(scenario())


def test_start_duration_is_bounded_in_public_projection(tmp_path: Path) -> None:
    """tts/asr 公开快照的 start_duration_ms 有界化，非法值收敛为 None。"""

    runtime = _runtime(tmp_path)
    order: list[str] = []
    tts = FakeTTS(order)
    asr = FakeASR(order)
    _install_fakes(runtime, tts, asr)

    async def scenario() -> None:
        runtime.tts_startup_health = {
            "status": "ready",
            "available": True,
            "engine": "canvas",
            "message": "rich start_duration_ms 2e9",
            "pending": False,
            "started_at_ms": 100.0,
            "start_duration_ms": 2_000_000.0,
        }
        assert runtime.tts_diagnostics()["health"]["start_duration_ms"] == 1_000_000.0

        runtime.tts_startup_health = {
            "status": "ready",
            "available": True,
            "engine": "canvas",
            "message": "negative start_duration_ms -5",
            "pending": False,
            "started_at_ms": 100.0,
            "start_duration_ms": -5.0,
        }
        assert runtime.tts_diagnostics()["health"]["start_duration_ms"] == 0.0

        runtime.tts_startup_health = {
            "status": "ready",
            "available": True,
            "engine": "canvas",
            "message": "NaN start_duration_ms",
            "pending": False,
            "started_at_ms": 100.0,
            "start_duration_ms": float("nan"),
        }
        assert runtime.tts_diagnostics()["health"]["start_duration_ms"] is None

        runtime.tts_startup_health = {
            "status": "ready",
            "available": True,
            "engine": "canvas",
            "message": "None start_duration_ms",
            "pending": False,
            "started_at_ms": 100.0,
            "start_duration_ms": None,
        }
        assert runtime.tts_diagnostics()["health"]["start_duration_ms"] is None

        runtime.asr_startup_health = {
            "status": "ready",
            "available": True,
            "ready": True,
            "message": "asr inf",
            "started_at_ms": 100.0,
            "start_duration_ms": float("inf"),
        }
        assert runtime.asr_diagnostics()["health"]["start_duration_ms"] is None

        runtime.asr_startup_health = {
            "status": "ready",
            "available": True,
            "ready": True,
            "message": "asr 42",
            "started_at_ms": 100.0,
            "start_duration_ms": 42,
        }
        assert runtime.asr_diagnostics()["health"]["start_duration_ms"] == 42.0

        await runtime.start_background()
        await asyncio.wait_for(tts.entered.wait(), timeout=_TIMEOUT)
        tts.release.set()
        await asyncio.wait_for(asr.entered.wait(), timeout=_TIMEOUT)
        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)

        public_tts = dict(runtime.tts_diagnostics()["health"])
        public_asr = dict(runtime.asr_diagnostics()["health"])
        raw_tts = dict(runtime.tts_startup_health)
        raw_asr = dict(runtime.asr_startup_health)
        for public, raw in ((public_tts, raw_tts), (public_asr, raw_asr)):
            public_duration = _bounded_finite(public.get("start_duration_ms"))
            raw_duration = _bounded_finite(raw.get("start_duration_ms"))
            assert raw_duration is not None
            assert raw_duration > 0
            assert public_duration == raw_duration

        module_diagnostics = runtime.module_diagnostics()
        assert (
            module_diagnostics["tts"]["health"]["start_duration_ms"]
            == public_tts["start_duration_ms"]
        )
        assert (
            module_diagnostics["asr"]["health"]["start_duration_ms"]
            == public_asr["start_duration_ms"]
        )
        await runtime.close()

    asyncio.run(scenario())


def test_tts_failure_path_keeps_bounded_duration_and_pending_false(tmp_path: Path) -> None:
    """TTS 启动异常转 unavailable 后 duration 有限为正，pending 变为 False。"""

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

        raw = dict(runtime.tts_startup_health)
        assert raw["status"] == "unavailable"
        assert raw["available"] is False
        assert raw["pending"] is False
        assert "TTS initialization failed: RuntimeError" in str(raw["message"])
        assert _bounded_finite(raw["started_at_ms"]) is not None
        failed_duration = _bounded_finite(raw["start_duration_ms"])
        assert failed_duration is not None
        assert failed_duration > 0

        public = dict(runtime.tts_diagnostics()["health"])
        assert public["start_duration_ms"] == failed_duration

        wait_snapshot = await runtime.wait_tts_initialization()
        assert _bounded_finite(wait_snapshot["start_duration_ms"]) == failed_duration

        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)
        assert dict(runtime.asr_startup_health)["status"] == "ready"
        assert _bounded_finite(dict(runtime.asr_startup_health)["start_duration_ms"]) > 0
        await runtime.close()

    asyncio.run(scenario())


def test_tts_cancellation_leaves_duration_none_and_stale_keys_intact(
    tmp_path: Path,
) -> None:
    """取消代次不重写健康回执：loading 保持、耗时空值、lated_at 键不出现。"""

    runtime = _runtime(tmp_path)
    order: list[str] = []
    tts = FakeTTS(order)
    asr = FakeASR(order)
    _install_fakes(runtime, tts, asr)

    async def scenario() -> None:
        await runtime.start_background()
        await asyncio.wait_for(tts.entered.wait(), timeout=_TIMEOUT)

        # 走生产语义：close/热替换路径通过 _cancel_tts_initialization 取消代次。
        await runtime._cancel_tts_initialization()
        assert runtime._tts_start_task is None
        # 被取消的 TTS 代次不重写健康字段，保留启动事务写入的 loading。
        assert dict(runtime.tts_startup_health)["status"] == "loading"
        stale = dict(runtime.tts_startup_health)
        assert stale["start_duration_ms"] is None
        assert "lated_at_ms" not in stale
        assert "started_at_ms" in stale
        assert _bounded_finite(stale["started_at_ms"]) is not None

        await asyncio.wait_for(asr.entered.wait(), timeout=_TIMEOUT)
        assert order == ["tts:enter", "asr:enter"]
        assert dict(runtime.asr_startup_health)["status"] == "loading"

        asr.release.set()
        await asyncio.wait_for(asr.finished.wait(), timeout=_TIMEOUT)
        assert dict(runtime.asr_startup_health)["status"] == "ready"
        assert _bounded_finite(dict(runtime.asr_startup_health)["start_duration_ms"]) > 0
        await runtime.close()

    asyncio.run(scenario())


def test_text_only_tts_background_lifecycle_emits_bounded_duration(tmp_path: Path) -> None:
    """真实 text_only 后端走成功后 duration 有限为正，公开投影保持 None 或有限。"""

    runtime = _runtime(tmp_path)

    async def scenario() -> None:
        await runtime.start_background()
        health = await runtime.wait_tts_initialization()
        assert health["status"] == "ready"
        assert health["available"] is True
        assert health["engine"] == "text-only"
        duration = _bounded_finite(health.get("start_duration_ms"))
        assert duration is not None
        assert duration > 0
        latency = health.get("latency_ms")
        assert latency is None or _bounded_finite(latency) is not None

        public = dict(runtime.tts_diagnostics()["health"])
        assert _bounded_finite(public.get("start_duration_ms")) is not None
        await runtime.close()

    asyncio.run(scenario())
