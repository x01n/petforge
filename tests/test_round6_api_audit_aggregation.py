"""Round6 A 模块：API 调用审计聚合的仓库/运行时/Web 投影契约。"""

from __future__ import annotations

import math

import pytest

from app.runtime import (
    ApplicationRuntime,
    LoadedConfiguration,
    build_runtime,
    inspect_resources,
)
from db import ApiCallAuditRepository, Database
from gui.web.control_surface import _safe_api_audit_diagnostic


def _runtime_with_audit(tmp_path):
    runtime = build_runtime(
        LoadedConfiguration(tmp_path / "config.yaml", {}),
        inspect_resources(tmp_path / "resources"),
    )
    return runtime


def test_repository_aggregate_contract_keys_and_none_durations() -> None:
    """聚合 SQL 只输出白名单键；无耗时样本时 JSON 兼容为 null。"""

    repository = ApiCallAuditRepository(Database(":memory:"))
    handle = repository.start(
        request_id="round6-a", kind="model", channel_id="primary", started_at=1.0
    )
    handle.finish(completed_at=1.25, response_model="served")
    row = repository.aggregate()[0]
    assert set(row) == {"key", "count", "avg_first_ms", "avg_total_ms", "max_total_ms"}
    # 无首字样本：AVG(CAST(NULL AS REAL)) 为 NULL -> null 语义。
    assert row["avg_first_ms"] is None
    assert row["avg_total_ms"] == pytest.approx(250.0)
    assert row["max_total_ms"] == pytest.approx(250.0)
    # 非有限值向外输出时保持 JSON 语义（float('nan') 输出 null）。
    assert math.isnan(float("nan")) is True


def test_runtime_api_call_audit_summary_bounds_values(tmp_path) -> None:
    """runtime 汇总入口对 count/耗时做 max(0, …) 有界化，None 保持 None。"""

    runtime = _runtime_with_audit(tmp_path)
    try:
        summary = runtime.api_call_audit_summary()
        assert set(summary) == {"total", "by_status", "by_channel", "latency_ms"}
        assert summary["total"] == 0
        assert summary["by_status"] == ()
        assert summary["by_channel"] == ()
        assert summary["latency_ms"] == {
            "avg_first": None,
            "avg_total": None,
            "max_total": None,
        }
        with pytest.raises(ValueError, match="unsupported audit query fields"):
            runtime.api_call_audit_summary(unknown_field=1)
        assert ApplicationRuntime._bounded_audit_total(9999999999) == 1_000_000
        assert ApplicationRuntime._bounded_audit_duration(-5) == 0.0
        assert ApplicationRuntime._bounded_audit_duration(True) is None
        assert ApplicationRuntime._bounded_audit_duration(1_000_000_000) == 1_000_000.0
    finally:
        import asyncio

        asyncio.run(runtime.close())


def test_web_projection_sanitizes_aggregate_rows() -> None:
    """Web 投影只允许白名单键集合；非法键、越界数值与异常行被剔除或截断。"""

    projected = _safe_api_audit_diagnostic(
        {
            "status": "available",
            "count": 7,
            "summary": {
                "total": 9_000_000,
                "by_status": [
                    {
                        "key": "status:completed",
                        "count": 5,
                        "avg_first_ms": 12.5,
                        "avg_total_ms": 50.0,
                        "extra": "not-allowed",
                    },
                    {
                        "key": "status:injected",
                        "count": -10,
                        "avg_first_ms": -1,
                        "avg_total_ms": 1_000_000_000,
                    },
                    {"key": "status<evil", "count": 1, "avg_first_ms": None},
                ],
                "by_channel": [
                    {
                        "key": "channel:primary",
                        "count": 5,
                        "avg_first_ms": 12.5,
                        "avg_total_ms": 50.0,
                    },
                ],
                "latency_ms": {"avg_first": 3.25, "avg_total": 12.25, "max_total": 1_000_000_000},
            },
            "records": [
                {
                    "kind": "model",
                    "status": "completed",
                    "channel_id": "primary",
                    "response_model": "served-model",
                    "time_to_first_token_ms": 12.5,
                    "total_duration_ms": 50.0,
                    "tool_execution_count": 2,
                    "input_payload": {"secret": "hidden"},
                    "output_text": "private response",
                }
            ],
        }
    )
    summary = projected["summary"]
    assert summary["total"] == 1_000_000
    allowed_keys = {"key", "count", "avg_first_ms", "avg_total_ms"}
    for row in summary["by_status"]:
        assert set(row) == allowed_keys
    assert summary["by_status"][0] == {
        "key": "status:completed",
        "count": 5,
        "avg_first_ms": 12.5,
        "avg_total_ms": 50.0,
    }
    # 非法键与越界行被丢弃或修正。count 为负时夹到 0；越界耗时保留原始值。
    assert summary["by_status"][1]["count"] == 0
    assert summary["by_status"][1]["avg_first_ms"] is None  # -1 负值判为 None
    assert "status<evil" not in {row["key"] for row in summary["by_status"]}
    assert summary["latency_ms"]["max_total"] == 1_000_000_000.0
    assert projected["count"] == 7
    assert projected["records"][0]["first_ms"] == 12.5
