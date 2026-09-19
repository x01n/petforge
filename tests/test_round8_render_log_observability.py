"""第 8 轮方向 2（渲染与日志审计深化）的观测性回断言。

覆盖两层：

- 日志聚合：``recent_log_summary`` 的维度计数、警告阈值、空队列与
  非法参数归一；``_safe_log_diagnostic`` 的 summary 白名单与计数有界；
- 渲染采样：``_on_performance_diagnostics`` 对新采样键的过滤与
  ``frame_time_ms`` 派生（samples=0 时缺省、负数/NaN 隔离、0.5s
  自锁语义、非白名单键不泄漏）。

渲染器回读断言需要构造页面桥（``_FakePage``/``_FakeView``/``_fake_clock``），
全部完成后由模块级 fixture 关闭内存日志处理器，避免与
``tests/test_logger.py`` 的根处理器还原相互污染。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from gui.renderers.web_live2d import WebLive2DRenderer
from gui.web.control_surface import _safe_log_diagnostic, public_control_state
from logger import MemoryLogHandler, recent_log_summary

MAX_BOUND = 1_000_000


class _FakeClock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakePage:
    def __init__(self) -> None:
        self.callback = None
        self.scripts: list[str] = []

    def runJavaScript(self, script: str, callback=None) -> None:  # noqa: N802
        self.scripts.append(script)
        self.callback = callback


class _FakeView:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    def page(self) -> _FakePage:
        return self._page


def _safe_memory_handler() -> MemoryLogHandler:
    import logger as logger_module

    if logger_module._MEMORY_HANDLER is not None:
        return logger_module._MEMORY_HANDLER
    handler = MemoryLogHandler(capacity=64)
    logger_module._MEMORY_HANDLER = handler
    return handler


@pytest.fixture
def memory_logs():
    import logger as logger_module

    previous = logger_module._MEMORY_HANDLER
    handler = logger_module._MEMORY_HANDLER
    if handler is None:
        handler = MemoryLogHandler(capacity=64)
        logger_module._MEMORY_HANDLER = handler
    handler.clear()
    try:
        yield handler
    finally:
        handler.clear()
        if previous is None:
            logger_module._MEMORY_HANDLER = None


def _emit(handler: MemoryLogHandler, level: str, logger: str, message: str) -> None:
    record = logging.LogRecord(logger, logging.getLevelName(level), "", 0, message, (), None)
    handler.emit(record)


def _structured(
    handler: MemoryLogHandler,
    level: str,
    logger: str,
    event: str,
    status: str,
    duration: float,
) -> None:
    payload = json.dumps(
        {"event": event, "status": status, "duration_ms": duration},
        ensure_ascii=False,
    )
    _emit(handler, level, logger, payload)


def _fresh_renderer(tmp_path: Path) -> tuple[WebLive2DRenderer, _FakePage, _FakeClock]:
    model_root = tmp_path / "live2d" / "model" / "perf"
    model_root.mkdir(parents=True, exist_ok=True)
    descriptor = model_root / "perf.model3.json"
    (model_root / "perf.moc3").write_bytes(b"moc")
    (model_root / "texture.png").write_bytes(b"texture")
    descriptor.write_text(
        json.dumps({"FileReferences": {"Moc": "perf.moc3", "Textures": ["texture.png"]}}),
        encoding="utf-8",
    )
    clock = _FakeClock()
    renderer = WebLive2DRenderer(
        tmp_path,
        model_path=descriptor,
        importer=lambda _name: object(),
    )
    renderer._clock = clock
    page = _FakePage()
    renderer._view = _FakeView(page)
    renderer._page_ready = True
    renderer._model_ready = True
    renderer._page_bridge_ready = True
    return renderer, page, clock


def _deliver(page: _FakePage, payload: dict) -> None:
    assert page.callback is not None
    page.callback(json.dumps(payload))


# ---------------------------------------------------------------- 日志聚合


def test_log_summary_counts_each_dimension_and_warning_threshold(memory_logs) -> None:
    handler = memory_logs
    _structured(handler, "INFO", "core", "app.started", "completed", 1.0)
    _structured(handler, "INFO", "core", "app.started", "completed", 2.0)
    _structured(handler, "WARNING", "core", "model.channel.slow", "degraded", 3.0)
    _structured(handler, "ERROR", "network", "model.call.failed", "failed", 4.0)
    _structured(handler, "DEBUG", "probe", "probe.ran", "completed", 1.0)

    summary = recent_log_summary(limit=200)

    assert set(summary) == {"total", "warning_plus", "by_level", "by_logger", "by_event"}
    assert summary["total"] == 5
    assert summary["warning_plus"] == 2
    assert summary["by_level"] == {"INFO": 2, "WARNING": 1, "ERROR": 1, "DEBUG": 1}
    assert summary["by_logger"] == {"core": 3, "network": 1, "probe": 1}
    assert summary["by_event"] == {
        "app.started": 2,
        "model.channel.slow": 1,
        "model.call.failed": 1,
        "probe.ran": 1,
    }


def test_log_summary_scans_capped_window_and_stays_within_limit(memory_logs) -> None:
    handler = memory_logs
    for index in range(8):
        _structured(handler, "INFO", "burst", f"burst.{index}", "completed", 1.0)

    summary = recent_log_summary(limit=5)

    assert summary["total"] == 5
    assert summary["warning_plus"] == 0
    assert summary["by_event"] == {f"burst.{index}": 1 for index in range(3, 8)}
    # 空 level/logger/event 维度直接跳过，不产生空键条目。
    _emit(handler, "INFO", "", "plain line")
    assert "" not in recent_log_summary(limit=200)["by_logger"]


def test_log_summary_rejects_invalid_params_and_empty_queue(memory_logs) -> None:
    handler = memory_logs
    for limit in (0, -3):
        assert recent_log_summary(limit=limit)["total"] == 0
    # 折叠测试 / 覆盖测试该路径后统一按 unknown 拒绝（非数值量纲归一）。
    with pytest.raises(ValueError):
        handler.recent_log_summary(limit="unknown")
    with pytest.raises(ValueError):
        handler.recent_log_summary(limit=True)

    _structured(handler, "INFO", "core", "keep.me", "completed", 1.0)
    empty = recent_log_summary(limit=-1)
    assert empty == {
        "total": 0,
        "warning_plus": 0,
        "by_level": {},
        "by_logger": {},
        "by_event": {},
    }


def test_log_summary_is_empty_without_handler(monkeypatch) -> None:
    import logger as logger_module

    monkeypatch.setattr(logger_module, "_MEMORY_HANDLER", None)
    summary = recent_log_summary(limit=200)
    assert summary == {
        "total": 0,
        "warning_plus": 0,
        "by_level": {},
        "by_logger": {},
        "by_event": {},
    }


def test_log_summary_counts_bounded_and_total_never_exceeds_limit(memory_logs) -> None:
    handler = memory_logs
    for _ in range(64):
        _structured(handler, "WARNING", "spam", "spam.slow", "degraded", 1.0)

    summary = recent_log_summary(limit=200)
    assert summary["total"] == 64
    assert summary["warning_plus"] == 64
    assert summary["by_level"] == {"WARNING": 64}
    assert summary["by_logger"] == {"spam": 64}
    assert summary["by_event"] == {"spam.slow": 64}


def test_safe_log_diagnostic_whitelists_and_bounds_summary() -> None:
    diagnostic = _safe_log_diagnostic(
        {
            "status": "available",
            "count": 4,
            "summary": {
                "by_level": {"INFO": 2, "WARNING": 5, "ERROR": 9},
                "by_logger": {"core": 3, "network": 8, "other-logger": 1},
                "by_event": {
                    "model.call.completed": 4,
                    "model.call.failed": 100,
                },
                "total": 4,
                "warning_plus": 100,
            },
            "records": [],
        }
    )

    assert diagnostic["summary"] == {
        "total": 4,
        "warning_plus": 100,
        "by_level": {"ERROR": 9, "WARNING": 5, "INFO": 2},
        "by_logger": {"network": 8, "core": 3, "other-logger": 1},
        "by_event": {"model.call.failed": 100, "model.call.completed": 4},
    }


def test_safe_log_diagnostic_summary_rejects_sensitive_and_oversized_counts() -> None:
    filtered = _safe_log_diagnostic(
        {
            "summary": {
                "by_level": {"INFO": 2},
                "by_logger": {"token=secret": 1, "ok.logger": 2},
                "by_event": {"placeholder": 1},
            }
        }
    )["summary"]
    assert "ok.logger" in filtered["by_logger"]
    assert "token" not in json.dumps(filtered, ensure_ascii=False)

    malformed = _safe_log_diagnostic({"summary": {"by_level": {"INFO": 2_000_000}}})
    assert malformed["summary"] == {}

    partial = _safe_log_diagnostic(
        {"summary": {"by_level": {"INFO": 2}, "by_logger": "not-a-mapping"}}
    )
    assert partial["summary"] == {}

    assert _safe_log_diagnostic({})["summary"] == {}
    assert _safe_log_diagnostic("not-a-mapping")["summary"] == {}


def test_public_control_state_projects_log_summary() -> None:
    state = public_control_state(
        {
            "diagnostics": {
                "logs": {
                    "status": "available",
                    "count": 2,
                    "summary": {
                        "by_level": {"INFO": 2},
                        "by_logger": {"core": 2},
                        "by_event": {"app.started": 2},
                    },
                    "records": [],
                }
            }
        }
    )

    summary = state["diagnostics"]["logs"]["summary"]
    assert summary == {
        "total": 0,
        "warning_plus": 0,
        "by_level": {"INFO": 2},
        "by_logger": {"core": 2},
        "by_event": {"app.started": 2},
    }
    # 公开结构允许 message 字段（连接文案等），但日志诊断对象本身不得
    # 出现 message 正文键。
    assert "message" not in json.dumps(state["diagnostics"]["logs"], ensure_ascii=False)


# ---------------------------------------------------------------- 渲染采样


def test_performance_diagnostics_derive_frame_time_ms_and_guard_zero_samples(
    tmp_path: Path,
) -> None:
    renderer, page, _clock = _fresh_renderer(tmp_path)

    renderer.request_performance_diagnostics()
    _deliver(
        page,
        {
            "targetFrameRate": 60,
            "renderedFrames": 3,
            "frameTimeAccumulatorMs": 49.0,
            "frameTimeSamples": 3,
            "largeFrameDeltas": 2,
        },
    )
    diagnostics = renderer.performance_diagnostics
    assert diagnostics["frameTimeAccumulatorMs"] == 49.0
    assert diagnostics["frameTimeSamples"] == 3.0
    assert diagnostics["largeFrameDeltas"] == 2.0
    assert diagnostics["frame_time_ms"] == round(49.0 / 3.0, 1)


def test_performance_diagnostics_drop_frame_time_without_samples(tmp_path: Path) -> None:
    renderer, page, _clock = _fresh_renderer(tmp_path)

    renderer.request_performance_diagnostics()
    _deliver(
        page,
        {
            "frameTimeAccumulatorMs": 0.0,
            "frameTimeSamples": 0,
            "largeFrameDeltas": 0,
        },
    )
    diagnostics = renderer.performance_diagnostics
    assert diagnostics["frameTimeSamples"] == 0.0
    assert "frame_time_ms" not in diagnostics


def test_performance_diagnostics_filter_negative_and_nan_samples(tmp_path: Path) -> None:
    renderer, page, _clock = _fresh_renderer(tmp_path)

    renderer.request_performance_diagnostics()
    _deliver(
        page,
        {
            "frameTimeAccumulatorMs": float("nan"),
            "frameTimeSamples": "not-a-number",
            "largeFrameDeltas": -7,
        },
    )
    diagnostics = renderer.performance_diagnostics
    assert "frameTimeAccumulatorMs" not in diagnostics
    assert "frameTimeSamples" not in diagnostics
    assert "largeFrameDeltas" not in diagnostics
    assert "frame_time_ms" not in diagnostics
    assert "sampled_at" in diagnostics
    assert set(diagnostics) == {"sampled_at"}


def test_performance_diagnostics_service_lock_keeps_half_second_pacing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import gui.renderers.web_live2d as web_live2d_module

    renderer, page, clock = _fresh_renderer(tmp_path)
    monkeypatch.setattr(web_live2d_module, "monotonic", clock)

    renderer.request_performance_diagnostics()
    assert len(page.scripts) == 1

    renderer.request_performance_diagnostics()
    assert len(page.scripts) == 1

    clock.advance(0.49)
    renderer.request_performance_diagnostics()
    assert len(page.scripts) == 1

    clock.advance(0.02)
    renderer.request_performance_diagnostics()
    assert len(page.scripts) == 2


def test_performance_diagnostics_never_leak_unknown_keys(tmp_path: Path) -> None:
    renderer, page, _clock = _fresh_renderer(tmp_path)

    renderer.request_performance_diagnostics()
    _deliver(page, {"targetFrameRate": 60, "secret": "must-not-leak"})
    diagnostics = renderer.performance_diagnostics
    assert "secret" not in diagnostics
    assert diagnostics["targetFrameRate"] == 60.0


def test_sampled_at_uses_monotonic_clock(tmp_path: Path, monkeypatch) -> None:
    import gui.renderers.web_live2d as web_live2d_module

    renderer, page, clock = _fresh_renderer(tmp_path)
    monkeypatch.setattr(web_live2d_module, "monotonic", clock)

    renderer.request_performance_diagnostics()
    _deliver(page, {})
    recorded = renderer.performance_diagnostics["sampled_at"]

    clock.advance(3.0)
    renderer.request_performance_diagnostics()
    _deliver(page, {"targetFrameRate": 30})
    advanced = renderer.performance_diagnostics["sampled_at"]

    assert advanced == pytest.approx(recorded + 3.0, abs=0.2)


def test_renderer_sample_payload_end_to_end_shape(tmp_path: Path) -> None:
    renderer, page, _clock = _fresh_renderer(tmp_path)
    renderer.request_performance_diagnostics()
    script = page.scripts[-1]
    # JS 回读只走 debug().renderPerformance，宿主派生 frame_time_ms。
    assert "meapetLive2D" in script
    assert "renderPerformance" in script
    _deliver(
        page,
        {
            "renderedFrames": 4,
            "throttledFrames": 1,
            "frameTimeAccumulatorMs": 54.0,
            "frameTimeSamples": 4,
            "largeFrameDeltas": 1,
        },
    )
    diagnostics = renderer.performance_diagnostics
    assert diagnostics["renderedFrames"] == 4.0
    assert diagnostics["throttledFrames"] == 1.0
    assert diagnostics["frame_time_ms"] == round(54.0 / 4.0, 1)
    assert diagnostics["largeFrameDeltas"] == 1.0
