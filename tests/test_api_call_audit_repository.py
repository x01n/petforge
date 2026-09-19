"""API 调用审计仓储聚合（按状态/渠道/耗时）回归测试。"""

from __future__ import annotations

import asyncio

import pytest

from db import ApiCallAuditRepository, Database


def _seeded_repository() -> ApiCallAuditRepository:
    """构造三条覆盖完成/失败/进行中与空渠道的审计记录。"""

    repository = ApiCallAuditRepository(Database(":memory:"))
    completed = repository.start(
        request_id="req-1",
        kind="model",
        channel_id="primary",
        requested_model="chat",
        started_at=10.0,
    )
    completed.mark_first_token(occurred_at=10.2, first_char="答")
    completed.finish(completed_at=11.0, response_model="chat-v2")

    repository.start(
        request_id="req-2",
        kind="model",
        channel_id="psql",
        requested_model="vision",
        started_at=20.0,
    ).fail(error_type="timeout", completed_at=22.0)

    pending = repository.start(
        request_id="req-3",
        kind="model",
        requested_model="gemini-2",
        started_at=30.0,
    )
    pending.mark_first_token(occurred_at=30.5)
    return repository


def test_aggregate_groups_by_status_and_channel_with_none_tolerated_durations() -> None:
    """SQL GROUP BY 正确性：状态/渠道分桶、AVG/MAX 与 None 样本处理。"""

    repository = _seeded_repository()
    groups = repository.aggregate()

    status_rows = {item["key"]: item for item in groups if str(item["key"]).startswith("status:")}
    assert set(status_rows) == {"status:completed", "status:failed", "status:started"}
    assert status_rows["status:completed"]["count"] == 1
    # 首字耗时样本只有 completed(200ms) 与 started(500ms)；failed 无样本发为 null。
    assert status_rows["status:completed"]["avg_first_ms"] == pytest.approx(200.0)
    assert status_rows["status:failed"]["avg_first_ms"] is None
    assert status_rows["status:completed"]["avg_total_ms"] == pytest.approx(1000.0)
    assert status_rows["status:failed"]["max_total_ms"] == pytest.approx(2000.0)
    # 进行中的调用尚无总耗时，均值保持 null。
    assert status_rows["status:started"]["avg_total_ms"] is None

    channel_rows = {item["key"]: item for item in groups if str(item["key"]).startswith("channel:")}
    assert set(channel_rows) == {"channel:primary", "channel:psql", "channel:"}
    assert channel_rows["channel:primary"]["count"] == 1
    assert channel_rows["channel:psql"]["count"] == 1
    # channel_id 空值按空串分组。
    assert channel_rows["channel:"]["count"] == 1

    summary = repository.summary()
    assert summary["total"] == 3
    assert set(summary) == {"total", "by_status", "by_channel", "latency_ms"}
    assert set(summary["latency_ms"]) == {"avg_first", "avg_total", "max_total"}
    # 仓库层汇总保持结构占位；全局加权耗时在 app.runtime 的有界投影中计算。
    assert summary["latency_ms"]["avg_first"] is None
    assert summary["latency_ms"]["max_total"] is None


def test_aggregate_respects_query_filters_and_rejects_unknown_fields() -> None:
    """过滤键语义与 count() 一致；未知键拒绝并抛 ValueError。"""

    repository = _seeded_repository()
    assert repository.aggregate(kind="tool") == ()
    assert repository.summary(kind="tool")["total"] == 0
    filtered = {item["key"]: item["count"] for item in repository.aggregate(status="completed")}
    assert filtered == {"status:completed": 1, "channel:primary": 1}

    with pytest.raises(ValueError, match="unsupported audit query fields"):
        repository.aggregate(limit=5)
    with pytest.raises(ValueError, match="unsupported audit query fields"):
        repository.summary(request="anything")


def test_aggregate_async_wrappers_keep_same_contract() -> None:
    async def scenario() -> None:
        repository = ApiCallAuditRepository(Database(":memory:"))
        handle = await repository.astart(request_id="async-agg", channel_id="primary")
        await handle.afinish(completed_at=handle.record.started_at + 0.25)
        groups = await repository.aaggregate()
        assert {item["key"] for item in groups} == {"status:completed", "channel:primary"}
        completed = next(item for item in groups if item["key"] == "status:completed")
        assert completed["max_total_ms"] == pytest.approx(250.0)
        assert (await repository.asummary())["total"] == 1

    asyncio.run(scenario())
