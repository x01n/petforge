from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from core.adapters.direct.base import ProviderAdapter
from core.adapters.direct.events import AdapterMetadata, ToolCallDelta
from core.contracts.chat import ChatMessage, ChatRequest
from core.events.types import ConversationContext, TextDelta, TurnFinished
from db import (
    API_CALL_AUDIT_SCHEMA_VERSION,
    ApiCallAuditRecord,
    ApiCallAuditRepository,
    Database,
)
from services.conversation.service import ConversationService
from services.model_routing.channels import ChannelConfig
from services.model_routing.router import ModelRouter, _audit_cache_fields
from services.tools import PermissionService, RiskLevel, ToolExecutionService, ToolRegistry
from services.tools.types import ToolSpec


class _AuditedAdapter(ProviderAdapter):
    provider = "audit-provider"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming", "tools"})

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request, *, context=None, cancel_event=None):
        del request, cancel_event
        self.calls += 1
        if self.calls == 1:
            yield ToolCallDelta(context, 0, "call-1", "pet:ping", '{"value":1}')
            yield TurnFinished(context, "tool_calls")
            return
        yield TextDelta(context, "完成。")
        yield TurnFinished(
            context,
            response_model="served-model",
            usage={"prompt_tokens": 3, "completion_tokens": 1, "cached_tokens": 2},
        )


class _EmptyThenTextAdapter(ProviderAdapter):
    provider = "empty-then-text"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming"})

    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, request, *, context=None, cancel_event=None):
        del request, cancel_event
        self.calls += 1
        if self.calls == 1:
            yield TextDelta(context, " \n")
            yield TurnFinished(context, "stop")
            return
        yield TextDelta(context, "备用渠道已响应")
        yield TurnFinished(context, "stop", response_model="served-fallback")


class _LateMetadataAdapter(ProviderAdapter):
    provider = "late-metadata"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming"})

    async def stream(self, request, *, context=None, cancel_event=None):
        del request, cancel_event
        yield TextDelta(context, "带元数据的响应")
        yield AdapterMetadata(
            context,
            {
                "response_model": "served-late-metadata",
                "cache_read_duration_ms": 2.5,
                "cache_write_duration_ms": 3.5,
            },
        )
        yield TurnFinished(
            context,
            usage={"cached_tokens": 1, "cache_creation_tokens": 2},
        )


def test_model_router_persists_model_and_tool_audits() -> None:
    async def scenario() -> None:
        database = Database(":memory:")
        repository = ApiCallAuditRepository(database)
        adapter = _AuditedAdapter()
        router = ModelRouter(
            [
                ChannelConfig(
                    "audit",
                    base_url="https://audit.invalid/v1",
                    model="requested-model",
                    capabilities=("streaming", "tools"),
                )
            ],
            adapter_factory=lambda _channel: adapter,
            audit_repository=repository,
        )
        registry = ToolRegistry()

        async def ping(arguments, _context):
            return {"pong": arguments["value"]}

        registry.register(
            ToolSpec(
                "pet:ping",
                "ping",
                {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
                ping,
                RiskLevel.LOW,
                read_only=True,
            )
        )
        tools = ToolExecutionService(
            registry,
            PermissionService(auto_allow_low_risk=True),
            execution_record_sink=router.record_tool_execution,
        )
        service = ConversationService(router, tools=tools)
        result = await service.complete("调用工具")
        assert result.text == "完成。"
        rows = repository.list_recent(limit=10)
        model_rows = [row for row in rows if row.kind == "model"]
        tool_rows = [row for row in rows if row.kind == "tool"]
        completed_model = next(row for row in model_rows if row.output_text == "完成。")
        assert completed_model.response_model == "served-model"
        assert completed_model.requested_model == "requested-model"
        assert completed_model.input_payload["model"] == "requested-model"
        assert completed_model.input_payload["messages"][-1]["role"] == "tool"
        assert completed_model.cache_read_tokens == 2
        assert completed_model.first_char == "完"
        assert completed_model.time_to_first_token_ms is not None
        tool_model = next(row for row in model_rows if row.tool_call_count == 1)
        assert tool_model.tool_calls[0]["arguments"] == '{"value":1}'
        assert any(row.tool_execution_count == 1 for row in tool_rows)
        assert tool_model.tool_execution_count == 1
        assert tool_model.tool_executions[0]["arguments"] == {"value": 1}
        assert tool_model.tool_executions[0]["result"]["pong"] == 1
        assert tool_model.tool_executions[0]["status"] == "completed"
        await service.aclose()
        await router.aclose()

    asyncio.run(scenario())


def test_model_router_does_not_lock_fallback_on_empty_text_event() -> None:
    async def scenario() -> None:
        database = Database(":memory:")
        repository = ApiCallAuditRepository(database)
        adapter = _EmptyThenTextAdapter()
        router = ModelRouter(
            [
                ChannelConfig(
                    "primary",
                    base_url="https://primary.invalid/v1",
                    model="chat-primary",
                    capabilities=("streaming",),
                    priority=0,
                ),
                ChannelConfig(
                    "fallback",
                    base_url="https://fallback.invalid/v1",
                    model="chat-fallback",
                    capabilities=("streaming",),
                    priority=1,
                ),
            ],
            adapter_factory=lambda _channel: adapter,
            audit_repository=repository,
        )
        request = ChatRequest(
            model="chat-primary",
            messages=(ChatMessage("user", "测试备用渠道"),),
        )
        context = ConversationContext("pet", "session", "turn-empty", 0)
        response = await router.complete(request, context=context)
        assert response.text == "备用渠道已响应"
        assert adapter.calls == 2
        rows = repository.list_recent(kind="model", limit=10)
        assert any(
            row.response_model == "served-fallback" and row.status == "completed" for row in rows
        )
        assert {row.attempt for row in rows} == {1, 2}
        first_attempt = next(row for row in rows if row.attempt == 1)
        assert first_attempt.status == "failed"
        assert first_attempt.first_token_at is None
        await router.aclose()

    asyncio.run(scenario())


def test_model_router_persists_metadata_emitted_after_first_text() -> None:
    async def scenario() -> None:
        database = Database(":memory:")
        repository = ApiCallAuditRepository(database)
        router = ModelRouter(
            [
                ChannelConfig(
                    "late",
                    base_url="https://late.invalid/v1",
                    model="late-model",
                    capabilities=("streaming",),
                )
            ],
            adapter_factory=lambda _channel: _LateMetadataAdapter(),
            audit_repository=repository,
        )
        context = ConversationContext("pet", "session", "turn-late-metadata", 0)
        response = await router.complete(
            ChatRequest("late-model", (ChatMessage("user", "测试迟到元数据"),)),
            context=context,
        )
        assert response.text == "带元数据的响应"
        row = repository.list_recent(kind="model", limit=1)[0]
        assert row.response_model == "served-late-metadata"
        assert row.cache_read_tokens == 1
        assert row.cache_write_tokens == 2
        assert row.cache_read_duration_ms == pytest.approx(2.5)
        assert row.cache_write_duration_ms == pytest.approx(3.5)
        assert row.first_char == "带"
        assert row.time_to_first_token_ms is not None
        await router.aclose()

    asyncio.run(scenario())


def test_api_call_audit_lifecycle_round_trips_all_observability_fields() -> None:
    database = Database(":memory:")
    repository = ApiCallAuditRepository(database, clock=lambda: 100.0)

    handle = repository.start(
        request_id="req-1",
        mode="agent",
        profile_id="pet",
        session_id="session",
        turn_id="turn",
        generation_id=4,
        attempt=2,
        provider="openai",
        protocol="openai_chat",
        channel_id="primary",
        channel_name="主渠道",
        channel_info={"region": "local", "stream": True},
        requested_model="model-requested",
        input_payload={"messages": [{"role": "user", "content": "hello"}]},
        metadata={"retry": True},
        started_at=90.0,
    )

    handle.mark_first_token(occurred_at=90.25)
    handle.append_output("hello ")
    handle.append_output("world")
    handle.add_tool_call(
        {"call_id": "call-1", "identity": "desktop:click_at", "arguments": {"x": 3}}
    )
    handle.add_tool_execution(
        {
            "call_id": "call-1",
            "status": "completed",
            "duration_ms": 12.5,
            "result": {"clicked": True},
        }
    )
    record = handle.finish(
        completed_at=91.0,
        response_model="model-served",
        output_payload={"finish_reason": "stop"},
        usage={"prompt_tokens": 8, "completion_tokens": 2},
        finish_reason="stop",
        cache_read_tokens=5,
        cache_write_tokens=2,
        cache_read_duration_ms=1.25,
        cache_write_duration_ms=0.75,
        cache_info={"read": True, "write": True},
    )

    assert record.id == handle.record_id
    assert record.model_id == "model-served"
    assert record.time_to_first_token_ms == pytest.approx(250.0)
    assert record.total_duration_ms == pytest.approx(1000.0)
    assert record.output_text == "hello world"
    assert record.cache_read_tokens == 5
    assert record.cache_write_tokens == 2
    assert record.tool_call_count == 1
    assert record.tool_execution_count == 1

    restored = repository.get(handle.record_id)
    assert restored == record
    assert restored is not None
    assert restored.input_payload["messages"][0]["content"] == "hello"
    assert restored.tool_executions[0]["result"]["clicked"] is True


def test_api_call_audit_terminal_state_is_idempotent_but_tool_details_can_arrive_late() -> None:
    repository = ApiCallAuditRepository(Database(":memory:"))
    handle = repository.start(request_id="terminal", started_at=10.0)
    completed = handle.finish(completed_at=11.0, response_model="served")

    assert (
        handle.fail(error_type="late", error_message="late failure", completed_at=12.0) == completed
    )
    assert handle.append_output("迟到正文") == completed
    assert handle.record.error_type == ""
    assert handle.record.output_text == ""

    updated = handle.add_tool_execution(
        {"call_id": "tool-1", "status": "completed", "result": {"ok": True}}
    )
    assert updated.status == "completed"
    assert updated.tool_execution_count == 1
    assert repository.get(handle.record_id).tool_execution_count == 1


def test_cache_audit_fields_choose_the_first_valid_source_and_fallback_duration() -> None:
    fields = _audit_cache_fields(
        {
            "cached_tokens": "invalid",
            "cache_read_tokens": 7,
            "cache_write_tokens": 3,
            "cache_read_duration_ms": 2.5,
        },
        {"cache_read_duration_ms": None, "cache_write_duration_ms": 1.25},
    )
    assert fields == (
        7,
        3,
        2.5,
        1.25,
        {
            "read": True,
            "write": True,
            "read_source": "cache_read_tokens",
            "write_source": "cache_write_tokens",
        },
    )


def test_legacy_openai_sse_usage_preserves_nested_cache_tokens() -> None:
    from core.adapters.direct.openai_chat_sse import _usage_payload

    assert _usage_payload(
        {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 5},
        }
    ) == {
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
        "prompt_tokens_details.cached_tokens": 5,
    }


def test_api_call_audit_failure_and_query_filters(tmp_path: Path) -> None:
    path = tmp_path / "audit.sqlite3"
    repository = ApiCallAuditRepository(Database(path))
    first = repository.start(
        request_id="req-failed",
        channel_id="fallback",
        requested_model="vision",
        started_at=10.0,
    )
    failed = first.fail(error_type="timeout", error_message="provider timed out", completed_at=11.0)
    second = repository.start(
        request_id="req-ok",
        channel_id="primary",
        requested_model="chat",
        started_at=20.0,
    )
    second.finish(response_model="chat-v2", completed_at=21.0)

    assert failed.status == "failed"
    assert repository.list_recent(status="failed") == (failed,)
    assert repository.list_recent(channel_id="primary")[0].model_id == "chat-v2"
    assert repository.list_recent(request_id="req-ok")[0].request_id == "req-ok"
    assert repository.list_recent(started_after=15.0, started_before=25.0)[0].request_id == "req-ok"

    reopened = ApiCallAuditRepository(Database(path))
    assert reopened.get(failed.id).error_type == "timeout"
    assert reopened.list_recent(limit=1)[0].request_id == "req-ok"


def test_api_call_audit_async_wrappers_keep_same_contract() -> None:
    async def scenario() -> None:
        repository = ApiCallAuditRepository(Database(":memory:"))
        handle = await repository.astart(
            request_id="async-req",
            provider="gemini",
            requested_model="gemini-2",
            input_payload={"stream": True},
        )
        await handle.amark_first_token(occurred_at=handle.record.started_at + 0.1)
        await handle.aappend_output("chunk")
        record = await handle.afinish(response_model="gemini-2.5")
        assert (await repository.aget(record.id)).output_text == "chunk"
        assert (await repository.alist_recent(request_id="async-req"))[0].model_id == "gemini-2.5"
        assert await repository.acount(request_id="async-req") == 1

    asyncio.run(scenario())


def test_api_call_audit_rejects_invalid_payload_and_identifiers() -> None:
    with pytest.raises(ValueError, match="request_id is required"):
        ApiCallAuditRecord(request_id="", started_at=1.0)
    with pytest.raises(ValueError, match="input_payload must be JSON serializable"):
        ApiCallAuditRecord(request_id="req", started_at=1.0, input_payload={1: "bad"})
    with pytest.raises(ValueError, match="attempt must be a positive integer"):
        ApiCallAuditRecord(request_id="req", started_at=1.0, attempt=0)


def test_api_call_audit_schema_is_independent_and_versioned() -> None:
    database = Database(":memory:")
    ApiCallAuditRepository(database)
    version = database.connection.execute(
        "SELECT MAX(version) FROM api_call_audit_schema_version"
    ).fetchone()[0]
    assert version == API_CALL_AUDIT_SCHEMA_VERSION
    columns = {
        row[1]
        for row in database.connection.execute("PRAGMA table_info(api_call_audits)").fetchall()
    }
    assert {
        "request_id",
        "response_model",
        "time_to_first_token_ms",
        "tool_executions",
    } <= columns
