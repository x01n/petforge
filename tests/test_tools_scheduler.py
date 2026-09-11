from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import pytest

from app.runtime import MutablePetController
from logger.events import event_fingerprint
from services.conversation import PresentationService
from services.scheduler.behavior import BehaviorAction, BehaviorService
from services.scheduler.scheduler import SchedulerService
from services.tools.builtins import register_builtin_tools
from services.tools.command import (
    normalize_command_allowlist,
    normalize_command_argv,
    normalize_command_timeout,
)
from services.tools.executor import ToolExecutionService, ToolPlanResult
from services.tools.permissions import PermissionService
from services.tools.registry import ToolRegistry
from services.tools.signature import ToolInvocationSignature, canonical_tool_arguments
from services.tools.types import (
    RiskLevel,
    ToolCallContext,
    ToolKind,
    ToolPlanStep,
    ToolSpec,
    tool_user_label,
)
from services.tools.validation import validate_arguments


def test_tool_signature_uses_canonical_json_and_call_id() -> None:
    left = ToolInvocationSignature.create(
        identity="system:inspect",
        call_id="call-one",
        arguments={"nested": {"b": 2, "a": 1}, "enabled": True},
    )
    right = ToolInvocationSignature.create(
        identity="system:inspect",
        call_id="call-two",
        arguments={"enabled": True, "nested": {"a": 1, "b": 2}},
    )

    assert left.canonical_arguments == right.canonical_arguments
    assert left.digest != right.digest
    assert canonical_tool_arguments({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    with pytest.raises(ValueError, match="canonical JSON"):
        canonical_tool_arguments({1: "invalid"})


def test_system_command_boundary_requires_argv_allowlist_and_finite_timeout() -> None:
    allowed = normalize_command_allowlist(("/usr/bin/printf",))

    assert normalize_command_argv(
        ["/usr/bin/printf", "%s", "$(touch /tmp/never-run)"],
        allowlist=allowed,
    ) == ("/usr/bin/printf", "%s", "$(touch /tmp/never-run)")
    with pytest.raises(ValueError, match="argv list"):
        normalize_command_argv("/usr/bin/printf unsafe", allowlist=allowed)
    with pytest.raises(PermissionError, match="allowlist"):
        normalize_command_argv(["/bin/sh", "-c", "true"], allowlist=allowed)
    with pytest.raises(ValueError, match="finite"):
        normalize_command_timeout(float("nan"))


def test_private_asr_tool_transfers_audio_bytes_without_exposing_it_to_models() -> None:
    class Platform:
        pass

    class Result:
        def public(self):
            return {
                "request_id": "asr-result",
                "text": "转写结果",
                "language": "zh",
                "confidence": None,
                "confidence_available": False,
                "duration_ms": 5,
            }

    class ASR:
        max_audio_bytes = 4096
        language = "auto"

        async def transcribe(self, audio, **kwargs):
            assert audio.startswith(b"RIFF")
            assert kwargs == {
                "audio_format": "wav",
                "sample_rate": 16000,
                "channels": 1,
                "language": "zh",
            }
            return Result()

        def diagnostics(self):
            return {"status": "ready", "available": True, "ready": True}

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        asr=ASR(),
    )
    wav = b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + b"\x00" * 32
    spec = registry.require("system:transcribe_audio")
    result = asyncio.run(
        spec.handler(
            {
                "audio_base64": base64.b64encode(wav).decode("ascii"),
                "audio_format": "wav",
                "language": "zh",
            },
            ToolCallContext("profile", "session", "turn"),
        )
    )

    assert result["status"] == "completed"
    assert result["text"] == "转写结果"
    assert "system:transcribe_audio" in registry.identities()
    assert "system:transcribe_audio" not in {row["function"]["name"] for row in registry.schemas()}


def test_private_asr_tool_rejects_encoded_audio_before_base64_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Platform:
        pass

    class ASR:
        max_audio_bytes = 4
        language = "auto"

    decoded = False

    def forbidden_decode(*args, **kwargs):
        nonlocal decoded
        decoded = True
        raise AssertionError("oversized payload must be rejected before decode")

    monkeypatch.setattr("services.tools.builtins.base64.b64decode", forbidden_decode)
    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        asr=ASR(),
    )
    spec = registry.require("system:transcribe_audio")
    result = asyncio.run(
        spec.handler(
            {"audio_base64": "A" * 9, "audio_format": "wav"},
            ToolCallContext("profile", "session", "turn"),
        )
    )

    assert result == {"status": "denied", "reason": "audio exceeds configured byte limit"}
    assert decoded is False


def test_system_run_command_executes_literal_argv_without_shell_expansion() -> None:
    class Platform:
        pass

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        command_allowlist=(sys.executable,),
    )
    literal = "$(printf should-not-expand)"
    outcome = asyncio.run(
        ToolExecutionService(
            registry,
            PermissionService(bypass_approval=True),
        ).execute(
            call_id="literal-argv",
            identity="system:run_command",
            arguments={
                "command": [
                    sys.executable,
                    "-c",
                    "import sys; print(sys.argv[1])",
                    literal,
                ],
                "timeout_seconds": 3,
            },
            context=ToolCallContext("profile", "session", "literal-argv-turn"),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["return_code"] == 0
    assert outcome.content["stdout"].strip() == literal


def test_tool_execution_deduplicates_only_exact_three_part_signature() -> None:
    calls: list[dict[str, int]] = []

    async def handler(arguments, _context):
        calls.append(dict(arguments))
        return {"value": arguments["value"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:dedupe",
            "dedupe",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler,
            RiskLevel.LOW,
            kind=ToolKind.SYSTEM,
        )
    )
    registry.register(
        ToolSpec(
            "system:dedupe_other",
            "dedupe other",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            handler,
            RiskLevel.LOW,
            kind=ToolKind.SYSTEM,
        )
    )
    executor = ToolExecutionService(registry, PermissionService(), max_calls_per_turn=5)
    context = ToolCallContext("profile", "session", "dedupe-turn")

    async def scenario():
        first = await executor.execute(
            call_id="one", identity="system:dedupe", arguments={"value": 1}, context=context
        )
        same_call_id_different_arguments = await executor.execute(
            call_id="one", identity="system:dedupe", arguments={"value": 99}, context=context
        )
        different_call_id_same_arguments = await executor.execute(
            call_id="two", identity="system:dedupe", arguments={"value": 1}, context=context
        )
        exact_duplicate = await executor.execute(
            call_id="one", identity="system:dedupe", arguments={"value": 1}, context=context
        )
        different = await executor.execute(
            call_id="three", identity="system:dedupe", arguments={"value": 2}, context=context
        )
        different_identity = await executor.execute(
            call_id="one",
            identity="system:dedupe_other",
            arguments={"value": 1},
            context=context,
        )
        over_limit = await executor.execute(
            call_id="four", identity="system:dedupe", arguments={"value": 3}, context=context
        )
        return (
            first,
            same_call_id_different_arguments,
            different_call_id_same_arguments,
            exact_duplicate,
            different,
            different_identity,
            over_limit,
        )

    (
        first,
        same_call_id_different_arguments,
        different_call_id_same_arguments,
        exact_duplicate,
        different,
        different_identity,
        over_limit,
    ) = asyncio.run(scenario())
    assert first.status == "completed"
    assert same_call_id_different_arguments.status == "completed"
    assert different_call_id_same_arguments.status == "completed"
    assert exact_duplicate.status == "duplicate"
    assert exact_duplicate.content["duplicate_reason"] == "exact_signature"
    assert exact_duplicate.content["original_result"] == {"value": 1}
    assert different.status == "completed"
    assert different_identity.status == "completed"
    assert over_limit.status == "denied"
    assert calls == [{"value": 1}, {"value": 99}, {"value": 1}, {"value": 2}, {"value": 1}]


def test_execute_batch_parallelizes_safe_reads_and_preserves_order() -> None:
    active = 0
    maximum_active = 0
    release = asyncio.Event()

    async def read(arguments, _context):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await release.wait()
            return {"value": arguments["value"]}
        finally:
            active -= 1

    async def scenario():
        registry = ToolRegistry()
        for identity in ("system:batch_a", "system:batch_b"):
            registry.register(
                ToolSpec(
                    identity,
                    identity,
                    {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                    },
                    read,
                    RiskLevel.LOW,
                    read_only=True,
                )
            )
        executor = ToolExecutionService(
            registry,
            PermissionService(auto_allow_low_risk=True),
        )
        task = asyncio.create_task(
            executor.execute_batch(
                (
                    {"call_id": "a", "identity": "system:batch_a", "arguments": {"value": 1}},
                    {"call_id": "b", "identity": "system:batch_b", "arguments": {"value": 2}},
                ),
                context=ToolCallContext("profile", "session", "batch"),
            )
        )
        while maximum_active < 2:
            await asyncio.sleep(0)
        release.set()
        return await task

    outcomes = asyncio.run(scenario())
    assert maximum_active == 2
    assert [item.call_id for item in outcomes] == ["a", "b"]
    assert [item.content["value"] for item in outcomes] == [1, 2]


def test_execute_batch_stops_after_approval_and_returns_safe_skips() -> None:
    calls: list[str] = []

    async def side_effect(_arguments, _context):
        calls.append("first")
        return {"ok": True}

    async def second(_arguments, _context):
        calls.append("second")
        return {"ok": True}

    async def scenario():
        registry = ToolRegistry()
        registry.register(
            ToolSpec("system:batch_first", "first", {"type": "object"}, side_effect, RiskLevel.HIGH)
        )
        registry.register(
            ToolSpec("system:batch_second", "second", {"type": "object"}, second, RiskLevel.HIGH)
        )
        executor = ToolExecutionService(registry, PermissionService())
        outcomes = await executor.execute_batch(
            (
                {"call_id": "first", "identity": "system:batch_first", "arguments": {}},
                {"call_id": "second", "identity": "system:batch_second", "arguments": {}},
            ),
            context=ToolCallContext("profile", "session", "batch-approval"),
        )
        assert len(outcomes) == 2
        assert outcomes[0].status == "approval_required"
        assert outcomes[1].status == "denied"
        assert outcomes[1].content == {"error": "前一项操作等待确认，后续操作未执行"}
        assert len(executor.pending_approvals()) == 1
        resumed = await executor.approve_and_execute(outcomes[0].approval.approval_id)
        return outcomes, resumed

    outcomes, resumed = asyncio.run(scenario())
    assert resumed.status == "completed"
    assert calls == ["first"]


def test_execute_plan_resolves_dependency_layers_and_parallelizes_safe_reads() -> None:
    active = 0
    maximum_active = 0
    release = asyncio.Event()
    calls: list[str] = []

    async def read(arguments, _context):
        nonlocal active, maximum_active
        calls.append(arguments["name"])
        active += 1
        maximum_active = max(maximum_active, active)
        try:
            await release.wait()
            return {"value": arguments["name"]}
        finally:
            active -= 1

    async def scenario() -> ToolPlanResult:
        registry = ToolRegistry()
        for identity in ("system:plan_a", "system:plan_b", "system:plan_c"):
            registry.register(
                ToolSpec(
                    identity,
                    identity,
                    {"type": "object", "properties": {"name": {"type": "string"}}},
                    read,
                    RiskLevel.LOW,
                    read_only=True,
                )
            )
        executor = ToolExecutionService(registry, PermissionService(auto_allow_low_risk=True))
        task = asyncio.create_task(
            executor.execute_plan(
                (
                    ToolPlanStep("a", "call-a", "system:plan_a", {"name": "a"}),
                    ToolPlanStep("b", "call-b", "system:plan_b", {"name": "b"}),
                    ToolPlanStep("c", "call-c", "system:plan_c", {"name": "c"}, ("a", "b")),
                ),
                context=ToolCallContext("profile", "session", "plan"),
            )
        )
        while maximum_active < 2:
            await asyncio.sleep(0)
        release.set()
        return await task

    result = asyncio.run(scenario())
    assert isinstance(result, ToolPlanResult)
    assert maximum_active == 2
    assert calls[:2] == ["a", "b"]
    assert calls[2:] == ["c"]
    assert [outcome.call_id for outcome in result.outcomes] == ["call-a", "call-b", "call-c"]
    assert [item.parallel_group for item in result.audit] == [1, 1, 2]


def test_execute_plan_stops_before_unstarted_dependent_steps_on_approval() -> None:
    calls: list[str] = []

    async def write(_arguments, _context):
        calls.append("write")
        return {"ok": True}

    async def scenario() -> ToolPlanResult:
        registry = ToolRegistry()
        registry.register(
            ToolSpec("system:plan_write", "write", {"type": "object"}, write, RiskLevel.HIGH)
        )
        registry.register(
            ToolSpec("system:plan_after", "after", {"type": "object"}, write, RiskLevel.HIGH)
        )
        executor = ToolExecutionService(registry, PermissionService())
        return await executor.execute_plan(
            (
                ToolPlanStep("write", "call-write", "system:plan_write"),
                ToolPlanStep("after", "call-after", "system:plan_after", depends_on=("write",)),
            ),
            context=ToolCallContext("profile", "session", "plan-approval"),
        )

    result = asyncio.run(scenario())
    assert result.stopped is True
    assert [item.status for item in result.outcomes] == ["approval_required", "denied"]
    assert result.audit[1].reason == "approval_pending"
    assert calls == []
    assert len(result.outcomes[0].approval and result.outcomes[0].approval.approval_id or "") > 0


def test_execute_plan_rejects_unknown_and_cyclic_dependencies() -> None:
    async def handler(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(ToolSpec("system:plan", "plan", {"type": "object"}, handler, RiskLevel.LOW))
    executor = ToolExecutionService(registry, PermissionService(auto_allow_low_risk=True))
    context = ToolCallContext("profile", "session", "plan-invalid")

    with pytest.raises(ValueError, match="unknown dependency"):
        asyncio.run(
            executor.execute_plan(
                (ToolPlanStep("a", "a", "system:plan", depends_on=("missing",)),),
                context=context,
            )
        )
    with pytest.raises(ValueError, match="dependency cycle"):
        asyncio.run(
            executor.execute_plan(
                (
                    ToolPlanStep("a", "a", "system:plan", depends_on=("b",)),
                    ToolPlanStep("b", "b", "system:plan", depends_on=("a",)),
                ),
                context=context,
            )
        )


def test_execute_plan_resolves_result_reference_and_resumes_after_approval() -> None:
    clicks: list[tuple[int, int]] = []

    async def ocr(_arguments, _context):
        return {
            "coordinate_space": "screen",
            "boxes": ({"text": "确定", "x": 120, "y": 44},),
        }

    async def click(arguments, _context):
        clicks.append((arguments["x"], arguments["y"]))
        return {"status": "completed"}

    async def scenario() -> tuple[ToolPlanResult, ToolPlanResult]:
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "desktop:ocr",
                "ocr",
                {"type": "object"},
                ocr,
                RiskLevel.LOW,
                read_only=True,
            )
        )
        registry.register(
            ToolSpec(
                "desktop:click_at",
                "click",
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
                click,
                RiskLevel.HIGH,
            )
        )
        executor = ToolExecutionService(
            registry,
            PermissionService(auto_allow_low_risk=True),
        )
        context = ToolCallContext("profile", "session", "plan-reference")
        first = await executor.execute_plan(
            (
                ToolPlanStep("ocr", "call-ocr", "desktop:ocr"),
                ToolPlanStep(
                    "click",
                    "call-click",
                    "desktop:click_at",
                    {
                        "x": {
                            "$step_result": {
                                "step_id": "ocr",
                                "path": ["boxes", 0, "x"],
                            }
                        },
                        "y": {
                            "$step_result": {
                                "step_id": "ocr",
                                "path": ["boxes", 0, "y"],
                            }
                        },
                    },
                    ("ocr",),
                ),
            ),
            context=context,
        )
        assert first.checkpoint is not None
        approval = first.pending_approval
        assert approval is not None
        approved = await executor.approve_and_execute(
            approval.approval_id,
            context=context,
        )
        second = await executor.resume_plan(
            first.checkpoint,
            approved,
            context=context,
        )
        return first, second

    first, second = asyncio.run(scenario())
    assert [item.status for item in first.outcomes] == [
        "completed",
        "approval_required",
    ]
    assert [item.status for item in second.outcomes] == ["completed", "completed"]
    assert second.stopped is False
    assert clicks == [(120, 44)]


def test_execute_plan_retries_only_low_risk_reads_and_bounds_timeout() -> None:
    calls = 0

    async def flaky(_arguments, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient")
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:flaky_read",
            "read",
            {"type": "object"},
            flaky,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    registry.register(
        ToolSpec(
            "system:write",
            "write",
            {"type": "object"},
            flaky,
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(auto_allow_low_risk=True),
    )
    result = asyncio.run(
        executor.execute_plan(
            (
                ToolPlanStep(
                    "read",
                    "read-call",
                    "system:flaky_read",
                    max_attempts=2,
                ),
            ),
            context=ToolCallContext("profile", "session", "plan-retry"),
        )
    )
    assert result.outcomes[0].status == "completed"
    assert result.audit[0].attempts == 2
    with pytest.raises(ValueError, match="retries require"):
        asyncio.run(
            executor.execute_plan(
                (
                    ToolPlanStep(
                        "write",
                        "write-call",
                        "system:write",
                        max_attempts=2,
                    ),
                ),
                context=ToolCallContext("profile", "session", "plan-retry-invalid"),
            )
        )
    with pytest.raises(ValueError, match="timeout_seconds"):
        asyncio.run(
            executor.execute_plan(
                (),
                context=ToolCallContext("profile", "session", "plan-timeout-invalid"),
                timeout_seconds=float("inf"),
            )
        )


def test_execute_plan_cancellation_cleans_plan_approvals() -> None:
    cancel_event = asyncio.Event()

    class CancellingPermission(PermissionService):
        def evaluate(self, spec, context, *, call_id, safe_summary):
            result = super().evaluate(
                spec,
                context,
                call_id=call_id,
                safe_summary=safe_summary,
            )
            cancel_event.set()
            return result

    async def write(_arguments, _context):
        raise AssertionError("approval must prevent handler execution")

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:cancelled_plan_write",
            "write",
            {"type": "object"},
            write,
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(registry, CancellingPermission())

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await executor.execute_plan(
                (
                    ToolPlanStep("first", "first-call", "system:cancelled_plan_write"),
                    ToolPlanStep("second", "second-call", "system:cancelled_plan_write"),
                ),
                context=ToolCallContext("profile", "session", "cancelled-plan"),
                stop_on_approval=False,
                cancel_event=cancel_event,
            )
        assert executor.pending_approvals() == ()

    asyncio.run(scenario())


def test_execute_plan_honors_cancellation_and_step_timeout() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked(_arguments, _context):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:blocked",
            "blocked",
            {"type": "object"},
            blocked,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(auto_allow_low_risk=True),
    )

    async def scenario() -> None:
        cancel_event = asyncio.Event()
        task = asyncio.create_task(
            executor.execute_plan(
                (
                    ToolPlanStep(
                        "blocked",
                        "blocked-call",
                        "system:blocked",
                        timeout_seconds=5.0,
                    ),
                ),
                context=ToolCallContext("profile", "session", "plan-cancel"),
                cancel_event=cancel_event,
            )
        )
        await started.wait()
        cancel_event.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

        result = await executor.execute_plan(
            (
                ToolPlanStep(
                    "timeout",
                    "timeout-call",
                    "system:blocked",
                    timeout_seconds=0.05,
                ),
            ),
            context=ToolCallContext("profile", "session", "plan-timeout"),
        )
        assert result.outcomes[0].status == "failed"
        assert result.audit[0].reason == "step_timeout"

    asyncio.run(scenario())


def test_execute_plan_preserves_unexpected_parallel_approval_checkpoint() -> None:
    """权限在探针后发生变化时，审批结果仍需可恢复。"""

    class FlappingPermission(PermissionService):
        def allows_without_approval(self, spec, context):
            del spec, context
            return True

    async def read(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:flapping_read",
            "read",
            {"type": "object"},
            read,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    executor = ToolExecutionService(registry, FlappingPermission(auto_allow_low_risk=False))
    result = asyncio.run(
        executor.execute_plan(
            (
                ToolPlanStep("one", "one-call", "system:flapping_read"),
                ToolPlanStep("two", "two-call", "system:flapping_read"),
            ),
            context=ToolCallContext("profile", "session", "plan-flapping"),
        )
    )

    assert result.stopped is True
    assert result.pending_approval is not None
    assert result.checkpoint is not None
    assert result.audit[0].status == "approval_required"
    assert result.audit[1].status == "denied"
    assert len(executor.pending_approvals()) == 1


def test_tool_execution_single_flight_reuses_concurrent_result() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def handler(arguments, _context):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return {"value": arguments["value"]}

    async def scenario():
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "system:single_flight",
                "single flight",
                {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
                handler,
                RiskLevel.LOW,
                read_only=True,
                kind=ToolKind.SYSTEM,
            )
        )
        executor = ToolExecutionService(registry, PermissionService())
        context = ToolCallContext("profile", "session", "single-flight-turn")
        first_task = asyncio.create_task(
            executor.execute(
                call_id="first",
                identity="system:single_flight",
                arguments={"value": 7},
                context=context,
            )
        )
        await started.wait()
        duplicate_task = asyncio.create_task(
            executor.execute(
                call_id="first",
                identity="system:single_flight",
                arguments={"value": 7},
                context=context,
            )
        )
        await asyncio.sleep(0)
        release.set()
        return await asyncio.gather(first_task, duplicate_task)

    first, duplicate = asyncio.run(scenario())
    assert calls == 1
    assert first.status == "completed"
    assert duplicate.status == "duplicate"
    assert duplicate.content["original_result"] == {"value": 7}


def test_tool_duplicate_approval_reuses_final_result() -> None:
    calls = 0

    async def handler(arguments, _context):
        nonlocal calls
        calls += 1
        return {"value": arguments["value"]}

    async def scenario():
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "system:approval_dedupe",
                "approval dedupe",
                {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
                handler,
                RiskLevel.HIGH,
                kind=ToolKind.SYSTEM,
            )
        )
        executor = ToolExecutionService(registry, PermissionService())
        context = ToolCallContext("profile", "session", "approval-dedupe-turn")
        first = await executor.execute(
            call_id="first",
            identity="system:approval_dedupe",
            arguments={"value": 4},
            context=context,
        )
        duplicate_pending = await executor.execute(
            call_id="first",
            identity="system:approval_dedupe",
            arguments={"value": 4},
            context=context,
        )
        resumed = await executor.approve_and_execute(first.approval.approval_id)
        duplicate_completed = await executor.execute(
            call_id="first",
            identity="system:approval_dedupe",
            arguments={"value": 4},
            context=context,
        )
        return executor, first, duplicate_pending, resumed, duplicate_completed

    executor, first, duplicate_pending, resumed, duplicate_completed = asyncio.run(scenario())
    assert first.status == "approval_required"
    assert duplicate_pending.status == "duplicate"
    assert duplicate_pending.content["original_status"] == "approval_required"
    assert len(executor.pending_approvals()) == 0
    assert resumed.status == "completed"
    assert duplicate_completed.status == "duplicate"
    assert duplicate_completed.content["original_status"] == "completed"
    assert duplicate_completed.content["original_result"] == {"value": 4}
    assert calls == 1


def test_clear_turn_preserves_pending_signature_for_duplicate_approval() -> None:
    calls = 0

    async def handler(_arguments, _context):
        nonlocal calls
        calls += 1
        return {"value": 1}

    async def scenario():
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "system:clear_turn_dedupe",
                "clear turn dedupe",
                {"type": "object"},
                handler,
                RiskLevel.HIGH,
            )
        )
        executor = ToolExecutionService(registry, PermissionService())
        context = ToolCallContext(
            "profile",
            "session",
            "clear-turn",
            metadata={"mode": "agent", "generation_id": 3},
        )
        first = await executor.execute(
            call_id="same-call",
            identity="system:clear_turn_dedupe",
            arguments={},
            context=context,
        )
        assert first.status == "approval_required"
        # 对话 finally 会清理普通调用记录，但不能使仍待审批的签名失效。
        executor.clear_turn(context)
        duplicate = await executor.execute(
            call_id="same-call",
            identity="system:clear_turn_dedupe",
            arguments={},
            context=context,
        )
        assert duplicate.status == "duplicate"
        assert duplicate.content["original_status"] == "approval_required"
        resumed = await executor.approve_and_execute(first.approval.approval_id, context=context)
        duplicate_after = await executor.execute(
            call_id="same-call",
            identity="system:clear_turn_dedupe",
            arguments={},
            context=context,
        )
        return resumed, duplicate_after

    resumed, duplicate_after = asyncio.run(scenario())
    assert resumed.status == "completed"
    assert duplicate_after.status == "duplicate"
    assert duplicate_after.content["original_result"] == {"value": 1}
    assert calls == 1


def test_evicted_approval_forgets_deduplication_state() -> None:
    async def handler(_arguments, _context):
        return {"ok": True}

    async def scenario():
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                "system:bounded_approval",
                "bounded approval",
                {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
                handler,
                RiskLevel.HIGH,
                kind=ToolKind.SYSTEM,
            )
        )
        executor = ToolExecutionService(registry, PermissionService())
        executor._PENDING_APPROVAL_LIMIT = 2
        first_context = ToolCallContext("profile", "session", "approval-0")
        first = await executor.execute(
            call_id="call-0",
            identity="system:bounded_approval",
            arguments={"value": 0},
            context=first_context,
        )
        for index in range(1, 3):
            outcome = await executor.execute(
                call_id=f"call-{index}",
                identity="system:bounded_approval",
                arguments={"value": index},
                context=ToolCallContext("profile", "session", f"approval-{index}"),
            )
            assert outcome.status == "approval_required"
        retried = await executor.execute(
            call_id="call-0",
            identity="system:bounded_approval",
            arguments={"value": 0},
            context=first_context,
        )
        return executor, first, retried

    executor, first, retried = asyncio.run(scenario())

    assert first.approval is not None
    assert retried.status == "approval_required"
    assert retried.approval is not None
    assert retried.approval.approval_id != first.approval.approval_id
    assert len(executor.pending_approvals()) == executor._PENDING_APPROVAL_LIMIT


def test_tool_logs_use_signature_without_argument_payload(caplog) -> None:
    secret = "do-not-log-this-secret"

    async def handler(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:logged",
            "logged",
            {
                "type": "object",
                "properties": {"secret": {"type": "string"}},
                "required": ["secret"],
            },
            handler,
            RiskLevel.LOW,
            kind=ToolKind.SYSTEM,
        )
    )
    caplog.set_level("INFO", logger="services.tools.executor")
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="logged-call",
            identity="system:logged",
            arguments={"secret": secret},
            context=ToolCallContext("profile", "session", "logged-turn"),
        )
    )

    assert outcome.status == "completed"
    payloads = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "services.tools.executor"
    ]
    started = next(payload for payload in payloads if payload["event"] == "tool.call.started")
    completed = next(payload for payload in payloads if payload["event"] == "tool.call.completed")
    assert started["component"] == "tool"
    assert started["identity"] == "system:logged"
    assert started["risk"] == "low"
    assert started["correlation"] == event_fingerprint("logged-call")
    assert started["operation"] == event_fingerprint("logged-call")
    assert started["reason_code"] == "running"
    assert len(started["signature"]) == 64
    assert completed["status"] == "completed"
    assert completed["reason_code"] == "completed"
    assert completed["duration_ms"] >= 0
    assert "logged-call" not in caplog.text
    assert secret not in caplog.text

    def failing_sink(_state, _outcome):
        raise RuntimeError(secret)

    second = asyncio.run(
        ToolExecutionService(
            registry,
            PermissionService(),
            event_sink=failing_sink,
        ).execute(
            call_id="sink-failure",
            identity="system:logged",
            arguments={"secret": secret},
            context=ToolCallContext("profile", "session", "sink-failure-turn"),
        )
    )
    assert second.status == "completed"
    payloads = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "services.tools.executor"
    ]
    assert any(
        payload["event"] == "tool.call.notification_failed"
        and payload["status"] == "degraded"
        and payload["reason_code"] == "event_sink_failed"
        and payload["error_type"] == "RuntimeError"
        for payload in payloads
    )
    assert secret not in caplog.text


def test_tool_log_durations_isolate_concurrent_identical_signatures(
    caplog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同调用 ID 的相同工具参数必须分别保留生命周期计时。"""

    executor = ToolExecutionService(ToolRegistry(), PermissionService())
    ticks = iter((1.0, 2.0, 4.0, 7.0))
    monkeypatch.setattr(
        "services.tools.executor.time.monotonic",
        lambda: next(ticks),
    )
    caplog.set_level("INFO", logger="services.tools.executor")
    signature = "a" * 64

    for event, call_id in (
        ("started", "call-a"),
        ("started", "call-b"),
        ("completed", "call-a"),
        ("completed", "call-b"),
    ):
        executor._log_tool_event(
            event,
            identity="system:same_signature",
            signature_digest=signature,
            status="running" if event == "started" else "completed",
            call_id=call_id,
        )

    completed = [
        json.loads(record.message)
        for record in caplog.records
        if '"event": "tool.call.completed"' in record.message
    ]
    assert [item["operation"] for item in completed] == [
        event_fingerprint("call-a"),
        event_fingerprint("call-b"),
    ]
    assert [item["duration_ms"] for item in completed] == [3000.0, 5000.0]


def test_tool_structured_logs_cover_approval_and_duplicate_lifecycle(caplog) -> None:
    secret = "approval-secret-must-not-appear"

    async def handler(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:approval_log",
            "approval log",
            {
                "type": "object",
                "properties": {"secret": {"type": "string"}},
                "required": ["secret"],
            },
            handler,
            RiskLevel.HIGH,
            kind=ToolKind.SYSTEM,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    context = ToolCallContext("profile", "session", "approval-log-turn")
    caplog.set_level("INFO", logger="services.tools.executor")

    async def scenario() -> None:
        pending = await executor.execute(
            call_id="approval-call",
            identity="system:approval_log",
            arguments={"secret": secret},
            context=context,
        )
        assert pending.approval is not None
        duplicate = await executor.execute(
            call_id="approval-call",
            identity="system:approval_log",
            arguments={"secret": secret},
            context=context,
        )
        assert duplicate.status == "duplicate"
        await executor.approve_and_execute(pending.approval.approval_id)
        denied = await executor.execute(
            call_id="denied-call",
            identity="system:approval_log",
            arguments={"secret": "different"},
            context=context,
        )
        assert denied.approval is not None
        executor.deny_approval(denied.approval.approval_id)

    asyncio.run(scenario())

    payloads = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "services.tools.executor"
    ]
    events = {payload["event"] for payload in payloads}
    assert {
        "tool.call.started",
        "tool.call.approval_required",
        "tool.call.duplicate",
        "tool.call.approved",
        "tool.call.completed",
        "tool.call.denied",
    } <= events
    assert all(payload["component"] == "tool" for payload in payloads)
    assert all(payload["identity"] == "system:approval_log" for payload in payloads)
    assert all(payload["risk"] == "high" for payload in payloads)
    assert secret not in caplog.text
    assert "approval-call" not in caplog.text


def test_tool_pending_shutdown_emits_terminal_without_approval_or_call_id(
    caplog,
) -> None:
    async def handler(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:pending_log",
            "pending log",
            {"type": "object", "properties": {}},
            handler,
            RiskLevel.HIGH,
            kind=ToolKind.SYSTEM,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    caplog.set_level("INFO", logger="services.tools.executor")

    pending = asyncio.run(
        executor.execute(
            call_id="private-pending-call",
            identity="system:pending_log",
            arguments={},
            context=ToolCallContext("profile", "session", "pending-log-turn"),
        )
    )
    assert pending.status == "approval_required"
    assert pending.approval is not None
    approval_id = pending.approval.approval_id
    executor.clear_pending()

    payloads = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "services.tools.executor"
    ]
    cancelled = next(payload for payload in payloads if payload["event"] == "tool.call.cancelled")
    assert cancelled["status"] == "shutdown"
    assert cancelled["reason_code"] == "runtime_shutdown"
    assert cancelled["operation"] == event_fingerprint("private-pending-call")
    assert cancelled["duration_ms"] >= 0
    assert "private-pending-call" not in caplog.text
    assert approval_id not in caplog.text


def test_tool_runtime_validates_and_requires_approval() -> None:
    calls: list[dict] = []

    async def handler(arguments, _context):
        calls.append(dict(arguments))
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "danger",
            {
                "type": "object",
                "properties": {"value": {"type": "integer", "minimum": 1}},
                "required": ["value"],
            },
            handler,
            RiskLevel.HIGH,
        )
    )
    permissions = PermissionService()
    executor = ToolExecutionService(registry, permissions)
    context = ToolCallContext("p", "s", "t")
    outcome = asyncio.run(
        executor.execute(
            call_id="c1", identity="system:danger", arguments={"value": 1}, context=context
        )
    )
    assert outcome.status == "approval_required"
    assert not calls
    assert permissions.pending()


def test_tool_execution_fails_closed_for_invalid_context_and_non_json_snapshot() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "test:echo",
            "echo",
            {"type": "object"},
            lambda arguments, _context: {"value": arguments.get("value")},
            RiskLevel.LOW,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())

    async def run() -> None:
        invalid_context = await executor.execute(
            call_id="bad-context",
            identity="test:echo",
            arguments={},
            context=object(),  # type: ignore[arg-type]
        )
        assert invalid_context.status == "denied"
        assert invalid_context.content == {"error": "invalid tool context"}

        invalid_arguments = await executor.execute(
            call_id="bad-arguments",
            identity="test:echo",
            arguments=[("value", "not-a-mapping")],  # type: ignore[arg-type]
            context=ToolCallContext("p", "s", "t"),
        )
        assert invalid_arguments.status == "denied"
        assert invalid_arguments.content == {"error": "invalid arguments"}

        non_json = await executor.execute(
            call_id="non-json",
            identity="test:echo",
            arguments={"value": object()},
            context=ToolCallContext("p", "s", "non-json"),
        )
        assert non_json.status == "denied"
        assert non_json.content == {"error": "tool invocation snapshot is invalid"}

    asyncio.run(run())


def test_tool_context_rejects_control_characters_and_non_mapping_metadata() -> None:
    with pytest.raises(ValueError, match="profile_id"):
        ToolCallContext("profile\n", "session", "turn")
    with pytest.raises(ValueError, match="source"):
        ToolCallContext("profile", "session", "turn", source="model\n")
    with pytest.raises(ValueError, match="metadata"):
        ToolCallContext("profile", "session", "turn", metadata=[("key", "value")])  # type: ignore[arg-type]


def test_scheduler_runs_due_task() -> None:
    now = [100.0]
    executed: list[str] = []

    async def run(action, task):
        executed.append(str(action["identity"]))

    scheduler = SchedulerService(action_runner=run, clock=lambda: now[0])
    scheduler.upsert(
        task_id="task-1",
        name="tick",
        expression="every:1s",
        action={"identity": "pet:play_motion"},
        owner="p:s",
    )
    now[0] = 102.0
    asyncio.run(scheduler.tick())
    assert executed == ["pet:play_motion"]


def test_scheduler_records_action_error_without_stopping_tick() -> None:
    async def run(_action, _task):
        raise RuntimeError("transient")

    now = [100.0]
    scheduler = SchedulerService(action_runner=run, clock=lambda: now[0])
    scheduler.upsert(
        task_id="task-1",
        name="tick",
        expression="every:1s",
        action={"identity": "pet:play_motion"},
        owner="test",
    )
    now[0] = 102.0
    assert asyncio.run(scheduler.tick()) == ("task-1",)
    assert "transient" in str(scheduler.status()["last_error"])


def test_builtin_speak_passes_conversation_context_to_tts() -> None:
    class Pet:
        def __init__(self):
            self.messages = []

        def speak(self, text, *, mood="neutral"):
            self.messages.append((text, mood))
            return {"status": "rendered"}

        def move_to(self, *_args, **_kwargs):
            return {"status": "moved"}

        def set_expression(self, _name):
            return True

        def play_motion(self, _name):
            return True

        def set_click_through(self, enabled):
            return {"status": "available", "enabled": enabled}

    class TTS:
        def __init__(self):
            self.calls = []

        async def enqueue_text(self, context, text, **kwargs):
            self.calls.append((context, text, kwargs))
            return (text,)

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    pet = Pet()
    tts = TTS()
    presentation = PresentationService()
    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=pet,
        tts=tts,
        presentation=presentation,
        default_language="ja",
    )
    executor = ToolExecutionService(registry, PermissionService())
    outcome = asyncio.run(
        executor.execute(
            call_id="speak-1",
            identity="pet:speak",
            arguments={"text": "你好", "mood": "happy"},
            context=ToolCallContext("p", "s", "t", metadata={"mode": "agent", "generation_id": 7}),
        )
    )
    assert outcome.status == "completed"
    assert pet.messages == [("你好", "happy")]
    assert tts.calls and tts.calls[0][1] == "你好"
    assert tts.calls[0][0].generation_id == 7
    assert tts.calls[0][0].mode == "agent"
    assert tts.calls[0][2]["language"] == "ja"
    assert presentation.snapshot.rendered_text == "你好"
    assert presentation.snapshot.rendered_mood == "happy"


def test_builtin_context_snapshot_aggregates_desktop_state_without_capture() -> None:
    class Platform:
        backend = "test-desktop"

        def foreground_window(self):
            return {"status": "available", "title": "编辑器", "pid": 42}

        def list_processes(self, limit=20):
            return {
                "status": "available",
                "processes": [{"pid": 42, "name": "editor", "executable": "/secret/editor"}][
                    :limit
                ],
            }

        def system_idle_seconds(self):
            return 12.5

        def cursor_position(self):
            return {"status": "available", "backend": "test-desktop", "x": -3, "y": 420}

        def capture_screen(self, **_kwargs):
            raise AssertionError("context snapshot must not capture the screen")

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=object())
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="context-snapshot",
            identity="desktop:context_snapshot",
            arguments={"process_limit": 5},
            context=ToolCallContext("p", "s", "context-snapshot"),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["status"] == "available"
    assert outcome.content["foreground_window"]["title"] == "编辑器"
    assert outcome.content["processes"]["processes"] == [{"pid": 42, "name": "editor"}]
    assert "executable" not in outcome.content["processes"]["processes"][0]
    assert outcome.content["cursor_position"] == {
        "status": "available",
        "backend": "test-desktop",
        "x": -3,
        "y": 420,
    }
    assert outcome.content["system_idle_seconds"] == 12.5


def test_builtin_cursor_position_is_read_only_and_preserves_negative_coordinates() -> None:
    class Platform:
        backend = "test-desktop"

        def cursor_position(self):
            return {
                "status": "available",
                "backend": "test-desktop",
                "x": -10,
                "y": 20,
                "source": "test",
            }

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=object())
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="cursor-position",
            identity="desktop:cursor_position",
            arguments={},
            context=ToolCallContext("p", "s", "cursor-position"),
        )
    )
    assert outcome.status == "completed"
    assert outcome.content == {
        "status": "available",
        "backend": "test-desktop",
        "x": -10,
        "y": 20,
        "source": "test",
    }


def test_builtin_context_snapshot_degrades_when_one_desktop_read_is_unavailable() -> None:
    class Platform:
        backend = "partial-desktop"

        def foreground_window(self):
            return {"status": "unavailable", "reason": "no active window"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def system_idle_seconds(self):
            return None

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=object())
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="context-partial",
            identity="desktop:context_snapshot",
            arguments={},
            context=ToolCallContext("p", "s", "context-partial"),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["status"] == "degraded"
    assert outcome.content["processes"]["status"] == "available"


def test_builtin_speak_voice_matches_exact_tts_profile() -> None:
    class Pet:
        def speak(self, _text, *, mood="neutral"):
            return {"status": "rendered", "mood": mood}

    class Backend:
        def profile_ids(self):
            return ("voice-a", "voice-b")

    class TTS:
        backend = Backend()

        def __init__(self):
            self.calls = []

        async def enqueue_text(self, context, text, **kwargs):
            self.calls.append((context, text, kwargs))
            return (text,)

    class Platform:
        pass

    tts = TTS()
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=Pet(), tts=tts)
    executor = ToolExecutionService(registry, PermissionService())
    accepted = asyncio.run(
        executor.execute(
            call_id="voice-exact",
            identity="pet:speak",
            arguments={"text": "音色测试", "voice": "voice-b"},
            context=ToolCallContext("p", "s", "voice-exact"),
        )
    )
    rejected = asyncio.run(
        executor.execute(
            call_id="voice-missing",
            identity="pet:speak",
            arguments={"text": "不会合成", "voice": "VOICE-B"},
            context=ToolCallContext("p", "s", "voice-missing"),
        )
    )

    assert accepted.status == "completed"
    assert tts.calls[0][2]["profile_id"] == "voice-b"
    # 音色无效时仍保留桌宠气泡；只拒绝 TTS 入队。
    assert rejected.status == "completed"
    assert rejected.content["status"] == "unavailable"
    assert len(tts.calls) == 1


def test_builtin_speak_passes_explicit_tts_role() -> None:
    class Pet:
        def speak(self, _text, *, mood="neutral"):
            return {"status": "rendered", "mood": mood}

    class TTS:
        async def enqueue_text(self, _context, _text, **kwargs):
            self.role = kwargs.get("role")
            return ("角色语音",)

    registry = ToolRegistry()
    tts = TTS()
    register_builtin_tools(registry, platform=object(), pet_controller=Pet(), tts=tts)
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="role-speak",
            identity="pet:speak",
            arguments={"text": "请用女仆角色说话", "role": "maid"},
            context=ToolCallContext("p", "s", "role-speak"),
        )
    )

    assert outcome.status == "completed"
    assert tts.role == "maid"


def test_builtin_speak_keeps_visual_output_when_tts_is_unavailable() -> None:
    class Pet:
        def __init__(self):
            self.messages = []

        def speak(self, text, *, mood="neutral"):
            self.messages.append((text, mood))
            return {"status": "rendered"}

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    pet = Pet()
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=pet)
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="speak-text-only",
            identity="pet:speak",
            arguments={"text": "仍然显示", "mood": "happy"},
            context=ToolCallContext("p", "s", "t"),
        )
    )
    assert outcome.status == "completed"
    assert outcome.content["status"] == "unavailable"
    assert outcome.content["rendered"] is True
    assert pet.messages == [("仍然显示", "happy")]


def test_builtin_expression_preserves_live2d_case() -> None:
    class Pet:
        def __init__(self):
            self.names: list[str] = []

        def set_expression(self, name):
            self.names.append(name)
            return True

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    pet = Pet()
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=pet)
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="expression-case",
            identity="pet:set_expression",
            arguments={"name": "SmileLeft"},
            context=ToolCallContext("p", "s", "t"),
        )
    )

    assert outcome.status == "completed"
    assert pet.names == ["SmileLeft"]


def test_builtin_unavailable_pet_action_is_not_reported_as_completed() -> None:
    class Pet:
        def set_expression(self, _name):
            return False

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=Pet())
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="expression-unavailable",
            identity="pet:set_expression",
            arguments={"name": "happy"},
            context=ToolCallContext("p", "s", "t"),
        )
    )

    assert outcome.status == "failed"
    assert outcome.content["status"] == "unavailable"


def test_builtin_rejects_action_before_renderer_when_capability_probe_denies() -> None:
    class Pet:
        def __init__(self) -> None:
            self.called = False

        def supports_expression(self, _name):
            return False

        def set_expression(self, _name):
            self.called = True
            return True

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    pet = Pet()
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=pet)
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="expression-probe-denied",
            identity="pet:set_expression",
            arguments={"name": "unknown"},
            context=ToolCallContext("p", "s", "t"),
        )
    )

    assert outcome.status == "failed"
    assert outcome.content == {"status": "unavailable", "reason": "当前渲染器不支持此表情"}
    assert pet.called is False


def test_capture_window_resolves_x11_context_outside_qt_dispatch() -> None:
    caller_thread = threading.current_thread().name
    resolver_threads: list[str] = []
    captured: list[dict[str, object]] = []

    class Platform:
        def resolve_capture_window_id(self):
            resolver_threads.append(threading.current_thread().name)
            time.sleep(0.02)
            return 0x123

        def capture_screen(self, **kwargs):
            captured.append(dict(kwargs))
            return {
                "status": "available",
                "scope": kwargs.get("scope"),
                "window_id": kwargs.get("window_id"),
                "data": b"png",
            }

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=MutablePetController())
    assert registry.require("desktop:click_at").risk is RiskLevel.HIGH
    assert tool_user_label("desktop:click_at") == "点击屏幕坐标"
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService(bypass_approval=True)).execute(
            call_id="capture-window",
            identity="desktop:capture_screen",
            arguments={"scope": "window"},
            context=ToolCallContext("p", "s", "capture-window"),
        )
    )
    assert outcome.status == "completed"
    assert resolver_threads and resolver_threads[0] != caller_thread
    assert captured == [{"scope": "window", "region": None, "window_id": 0x123}]


def test_builtin_ocr_uses_tsv_and_translates_region_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "tesseract"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    raise SystemExit(0)\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.write(\n"
        "    'level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\t'\n"
        "    'left\\ttop\\twidth\\theight\\tconf\\ttext\\n'\n"
        "    '5\\t1\\t1\\t1\\t1\\t1\\t4\\t6\\t20\\t10\\t96.5\\tHello\\n'\n"
        "    '5\\t1\\t1\\t1\\t1\\t2\\t28\\t6\\t24\\t10\\t90\\tworld\\n'\n"
        ")\n",
        encoding="ascii",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(
        "services.tools.builtins.shutil.which",
        lambda name: str(executable) if name == "tesseract" else None,
    )

    class Platform:
        def capture_screen(self, **kwargs):
            assert kwargs == {
                "scope": "region",
                "region": {"x": 100, "y": 200, "width": 80, "height": 60},
            }
            return {
                "status": "available",
                "scope": "region",
                "format": "png",
                "data": b"png",
                "width": 80,
                "height": 60,
            }

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=MutablePetController())
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService(bypass_approval=True)).execute(
            call_id="ocr-region",
            identity="desktop:ocr",
            arguments={
                "scope": "region",
                "region": {"x": 100, "y": 200, "width": 80, "height": 60},
            },
            context=ToolCallContext("p", "s", "ocr-region"),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["text"] == "Hello world"
    assert outcome.content["coordinate_space"] == "screen"
    assert outcome.content["origin"] == {"x": 100, "y": 200}
    assert outcome.content["boxes"] == [
        {
            "x": 104,
            "y": 206,
            "width": 20,
            "height": 10,
            "text": "Hello",
            "confidence": 96.5,
        },
        {
            "x": 128,
            "y": 206,
            "width": 24,
            "height": 10,
            "text": "world",
            "confidence": 90.0,
        },
    ]
    assert "data" not in outcome.content


def test_builtin_ocr_translates_window_coordinates_when_geometry_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "tesseract"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    raise SystemExit(0)\n"
        "sys.stdin.buffer.read()\n"
        "print('level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\t'\n"
        "      'left\\ttop\\twidth\\theight\\tconf\\ttext')\n"
        "print('5\\t1\\t1\\t1\\t1\\t1\\t4\\t6\\t20\\t10\\t96.5\\t确定')\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(
        "services.tools.builtins.shutil.which",
        lambda name: str(executable) if name == "tesseract" else None,
    )

    class Window:
        geometry = (300, 400, 800, 600)

    class Platform:
        backend = "windows"

        def resolve_capture_window_id(self):
            return 123

        def active_window(self):
            return Window()

        def capture_screen(self, **kwargs):
            assert kwargs == {"scope": "window", "region": None, "window_id": 123}
            return {
                "status": "available",
                "scope": "window",
                "format": "png",
                "data": b"png",
                "width": 800,
                "height": 600,
            }

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=MutablePetController())
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService(bypass_approval=True)).execute(
            call_id="ocr-window",
            identity="desktop:ocr",
            arguments={"scope": "window"},
            context=ToolCallContext("p", "s", "ocr-window"),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["coordinate_space"] == "screen"
    assert outcome.content["origin"] == {"x": 300, "y": 400}
    assert outcome.content["boxes"][0]["x"] == 304
    assert outcome.content["boxes"][0]["y"] == 406


def test_builtin_ocr_reports_unavailable_without_tesseract_before_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Platform:
        def capture_screen(self, **_kwargs):
            raise AssertionError("capture must not run when OCR backend is unavailable")

    monkeypatch.setattr("services.tools.builtins.shutil.which", lambda _name: None)
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=MutablePetController())
    outcome = asyncio.run(
        registry.require("desktop:ocr").handler(
            {"scope": "screen"}, ToolCallContext("p", "s", "ocr-unavailable")
        )
    )

    assert outcome == {
        "status": "unavailable",
        "reason": "tesseract executable is unavailable",
    }


def test_builtin_ocr_bounds_text_and_box_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = tmp_path / "tesseract"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if '--version' in sys.argv:\n"
        "    raise SystemExit(0)\n"
        "sys.stdin.buffer.read()\n"
        "print('level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\t'\n"
        "      'left\\ttop\\twidth\\theight\\tconf\\ttext')\n"
        "for index in range(300):\n"
        "    print(f'5\\t1\\t1\\t1\\t{index}\\t1\\t0\\t{index}\\t1\\t1\\t80\\t' + 'x' * 64)\n",
        encoding="ascii",
    )
    executable.chmod(0o755)
    monkeypatch.setattr(
        "services.tools.builtins.shutil.which",
        lambda name: str(executable) if name == "tesseract" else None,
    )

    class Platform:
        def capture_screen(self, **_kwargs):
            return {
                "status": "available",
                "scope": "screen",
                "format": "png",
                "data": b"png",
                "width": 1000,
                "height": 1000,
            }

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=MutablePetController())
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService(bypass_approval=True)).execute(
            call_id="ocr-bounds",
            identity="desktop:ocr",
            arguments={"scope": "screen"},
            context=ToolCallContext("p", "s", "ocr-bounds"),
        )
    )

    assert outcome.status == "completed"
    assert len(outcome.content["boxes"]) == 256
    assert len(outcome.content["text"]) <= 8192
    assert outcome.content["truncated"] is True


def test_builtin_click_through_reaches_pet_controller() -> None:
    class Pet:
        def set_click_through(self, enabled):
            return {"status": "available", "enabled": enabled}

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=Pet())
    executor = ToolExecutionService(registry, PermissionService(bypass_approval=True))
    outcome = asyncio.run(
        executor.execute(
            call_id="click-through",
            identity="pet:set_click_through",
            arguments={"enabled": True},
            context=ToolCallContext("p", "s", "t"),
        )
    )
    assert outcome.status == "completed"
    assert outcome.content == {"status": "available", "enabled": True}


def test_builtin_coordinate_click_requires_approval_and_reuses_snapshot() -> None:
    clicks: list[tuple[int, int, str]] = []

    class Platform:
        def click_at(self, x, y, *, button="left"):
            clicks.append((x, y, button))
            return {"status": "completed", "x": x, "y": y, "button": button}

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=MutablePetController())
    spec = registry.require("desktop:automation_batch")
    assert spec.risk is RiskLevel.HIGH
    assert spec.group == "desktop_automation"
    assert spec.parameters["properties"]["steps"]["maxItems"] == 16
    executor = ToolExecutionService(registry, PermissionService())
    context = ToolCallContext("p", "s", "coordinate-click")

    async def scenario():
        pending = await executor.execute(
            call_id="click-at",
            identity="desktop:click_at",
            arguments={"x": 120, "y": 240, "button": "right"},
            context=context,
        )
        assert pending.status == "approval_required"
        assert pending.approval is not None
        assert clicks == []
        return await executor.approve_and_execute(
            pending.approval.approval_id,
            context=context,
        )

    outcome = asyncio.run(scenario())
    assert outcome.status == "completed"
    assert clicks == [(120, 240, "right")]


def test_builtin_automation_batch_requires_one_approval_and_deduplicates_exact_sequence() -> None:
    batches: list[object] = []

    class Platform:
        def automation_batch(self, steps):
            batches.append(steps)
            return {"status": "completed", "steps": ()}

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=MutablePetController())
    executor = ToolExecutionService(registry, PermissionService())
    context = ToolCallContext("p", "s", "automation")
    arguments = {"steps": [{"type": "key", "key": "enter"}]}

    async def scenario():
        pending = await executor.execute(
            call_id="automation-batch",
            identity="desktop:automation_batch",
            arguments=arguments,
            context=context,
        )
        assert pending.status == "approval_required"
        assert pending.approval is not None
        completed = await executor.approve_and_execute(
            pending.approval.approval_id,
            context=context,
        )
        duplicate = await executor.execute(
            call_id="automation-batch",
            identity="desktop:automation_batch",
            arguments=arguments,
            context=context,
        )
        return completed, duplicate

    completed, duplicate = asyncio.run(scenario())
    assert completed.status == "completed"
    assert duplicate.status == "duplicate"
    assert batches == [[{"type": "key", "key": "enter"}]]


def test_builtin_automation_batch_rejects_unknown_step_and_excessive_steps_before_approval() -> (
    None
):
    class Platform:
        def automation_batch(self, _steps):
            raise AssertionError("invalid schema must not reach platform")

    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=MutablePetController())
    executor = ToolExecutionService(registry, PermissionService())
    context = ToolCallContext("p", "s", "automation-invalid")

    invalid_kind = asyncio.run(
        executor.execute(
            call_id="invalid-kind",
            identity="desktop:automation_batch",
            arguments={"steps": [{"type": "shell", "command": "bad"}]},
            context=context,
        )
    )
    excessive = asyncio.run(
        executor.execute(
            call_id="too-many",
            identity="desktop:automation_batch",
            arguments={"steps": [{"type": "wait", "duration_ms": 0}] * 17},
            context=context,
        )
    )

    assert invalid_kind.status == "denied"
    assert excessive.status == "denied"


def test_module_status_reports_sanitized_runtime_capabilities() -> None:
    class Capability:
        name = "input_control"
        state = "available"

    class Platform:
        backend = "x11"

        def probe(self):
            return type("Snapshot", (), {"capabilities": (Capability(),)})()

    class Health:
        engine = "test-tts"
        available = True
        message = "ready"
        latency_ms = 1.5

    class TTS:
        async def health(self):
            return Health()

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        tts=TTS(),
    )
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="module-status",
            identity="system:module_status",
            arguments={},
            context=ToolCallContext("p", "s", "module-status"),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["platform"]["capabilities"] == (
        {"name": "input_control", "state": "available"},
    )
    assert outcome.content["tts"]["engine"] == "test-tts"
    assert any(item["protocol"] == "openai" for item in outcome.content["adapters"])
    assert "desktop:click_at" in outcome.content["tools"]
    assert registry.require("system:module_status").kind is ToolKind.SYSTEM
    assert registry.require("system:run_command").kind is ToolKind.SYSTEM
    assert tool_user_label("system:module_status") == "检查模块状态"


def test_module_status_does_not_probe_tts_health_while_speech_is_active() -> None:
    class Platform:
        backend = "x11"

        def probe(self):
            return type("Snapshot", (), {"capabilities": ()})()

    class Backend:
        engine_name = "busy-backend"

    class TTS:
        backend = Backend()

        def has_active_tasks(self) -> bool:
            return True

        async def health(self):
            raise AssertionError("active TTS health must not be queued")

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        tts=TTS(),
    )
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="module-status-busy",
            identity="system:module_status",
            arguments={},
            context=ToolCallContext("p", "s", "module-status-busy"),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["tts"]["status"] == "busy"
    assert outcome.content["tts"]["engine"] == "busy-backend"


def test_module_status_reuses_runtime_tts_snapshot_without_active_probe() -> None:
    class Platform:
        backend = "x11"

        def probe(self):
            return type("Snapshot", (), {"capabilities": ()})()

    class TTS:
        def has_active_tasks(self) -> bool:
            return False

        async def health(self):
            raise AssertionError("runtime snapshot must replace duplicate TTS health probe")

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        tts=TTS(),
        module_status_provider=lambda: {
            "tts": {
                "backend": "snapshot-backend",
                "health": {
                    "status": "loading",
                    "available": False,
                    "latency_ms": 2.5,
                },
            }
        },
    )
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="module-status-snapshot",
            identity="system:module_status",
            arguments={},
            context=ToolCallContext("p", "s", "module-status-snapshot"),
        )
    )

    assert outcome.status == "completed"
    assert outcome.content["tts"] == {
        "status": "loading",
        "available": False,
        "engine": "snapshot-backend",
        "latency_ms": 2.5,
    }


def test_boolean_schema_accepts_both_boolean_values() -> None:
    schema = {"type": "object", "properties": {"enabled": {"type": "boolean"}}}
    assert not validate_arguments(schema, {"enabled": True})
    assert not validate_arguments(schema, {"enabled": False})
    assert validate_arguments(schema, {"enabled": "true"})


def test_number_schema_rejects_non_finite_values() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "number"}}}
    assert validate_arguments(schema, {"x": float("nan")})
    assert validate_arguments(schema, {"x": float("inf")})


def test_expression_and_motion_tool_schemas_match_runtime_action_contracts() -> None:
    class Platform:
        pass

    class Pet:
        def __init__(self) -> None:
            self.expressions = []
            self.motions = []

        def supports_expression(self, _name):
            return True

        def supports_motion(self, _name):
            return True

        def set_expression_request(self, request):
            self.expressions.append(request)
            return {"status": "updated"}

        def play_motion_request(self, request):
            self.motions.append(request)
            return {"status": "started"}

    pet = Pet()
    registry = ToolRegistry()
    register_builtin_tools(registry, platform=Platform(), pet_controller=pet)
    expression_spec = registry.require("pet:set_expression")
    motion_spec = registry.require("pet:play_motion")

    assert validate_arguments(expression_spec.parameters, {})
    assert validate_arguments(
        expression_spec.parameters,
        {"name": "happy", "expressions": [{"name": "neutral"}]},
    )
    assert validate_arguments(
        expression_spec.parameters,
        {"expressions": [{"name": "happy"}], "parameters": {"ParamAngleX": 1}},
    )
    assert validate_arguments(
        expression_spec.parameters,
        {"name": "happy", "parameters": {"bad key": 1}},
    )
    assert validate_arguments(
        expression_spec.parameters,
        {"name": "happy", "parameters": {"ParamAngleX": "1"}},
    )
    assert validate_arguments(
        expression_spec.parameters,
        {"name": "happy", "parameters": {f"P{index}": 1 for index in range(33)}},
    )
    assert not validate_arguments(
        expression_spec.parameters,
        {"name": "happy", "parameters": {"ParamAngleX": 1.5}},
    )
    assert not validate_arguments(
        expression_spec.parameters,
        {
            "expressions": [
                {"name": "happy", "weight": 0.7, "parameters": {"ParamEyeLOpen": 1}},
                {"name": "neutral", "weight": 0.3},
            ],
            "mode": "blend",
        },
    )
    assert validate_arguments(
        motion_spec.parameters,
        {"name": "wave", "parameters": {"ParamBodyAngleX": float("nan")}},
    )

    async def scenario() -> None:
        context = ToolCallContext("profile", "session", "action-contract-turn")
        direct_invalid = await expression_spec.handler({}, context)
        assert direct_invalid["status"] == "denied"
        executor = ToolExecutionService(registry, PermissionService())
        invalid = await executor.execute(
            call_id="invalid-expression-shape",
            identity="pet:set_expression",
            arguments={"expressions": [{"name": "happy"}], "weight": 0.5},
            context=context,
        )
        assert invalid.status == "denied"
        valid_expression = await executor.execute(
            call_id="valid-expression-shape",
            identity="pet:set_expression",
            arguments={
                "expressions": [
                    {"name": "happy", "weight": 0.6},
                    {"name": "neutral", "weight": 0.4},
                ],
                "mode": "blend",
            },
            context=context,
        )
        valid_motion = await executor.execute(
            call_id="valid-motion-shape",
            identity="pet:play_motion",
            arguments={"name": "wave", "parameters": {"ParamBodyAngleX": 2.0}},
            context=context,
        )
        assert valid_expression.status == "completed"
        assert valid_motion.status == "completed"

    asyncio.run(scenario())
    assert len(pet.expressions) == 1
    assert len(pet.motions) == 1


def test_permission_ttl_rejects_non_finite_values_and_caps_upper_bound() -> None:
    with pytest.raises(ValueError, match="finite"):
        PermissionService(approval_ttl_seconds=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        PermissionService(approval_ttl_seconds=float("inf"))

    now = [100.0]
    permissions = PermissionService(approval_ttl_seconds=99999, clock=lambda: now[0])
    spec = ToolSpec("system:danger", "danger", {"type": "object"}, lambda *_: None, RiskLevel.HIGH)
    result = permissions.evaluate(spec, ToolCallContext("p", "s", "t"))

    assert result.approval is not None
    assert result.approval.expires_at == 3700.0


def test_permission_policy_reconfigure_is_atomic_and_preserves_existing_approvals() -> None:
    now = [100.0]
    permissions = PermissionService(auto_allow_low_risk=True, clock=lambda: now[0])
    low = ToolSpec("system:observe", "observe", {"type": "object"}, lambda *_: None, RiskLevel.LOW)
    danger = ToolSpec(
        "system:danger", "danger", {"type": "object"}, lambda *_: None, RiskLevel.HIGH
    )
    context = ToolCallContext("profile", "session", "turn")

    assert permissions.evaluate(low, context).decision.value == "allow"
    pending = permissions.evaluate(danger, context)
    assert pending.approval is not None

    permissions.reconfigure(
        deny=("system:danger",),
        auto_allow_low_risk=False,
        approval_ttl_seconds=5.0,
    )

    assert permissions.evaluate(danger, context).decision.value == "deny"
    assert permissions.evaluate(low, context).decision.value == "approval_required"
    assert pending.approval in permissions.pending()


def test_bypass_emits_parameter_free_audit_before_tool_execution() -> None:
    calls: list[dict[str, object]] = []
    events: list[tuple[str, str, dict[str, object]]] = []
    sequence: list[str] = []

    async def handler(arguments, _context):
        sequence.append("handler")
        calls.append(dict(arguments))
        return {"ok": True}

    async def sink(state, outcome):
        sequence.append(state)
        events.append((state, outcome.status, dict(outcome.content)))

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "danger",
            {
                "type": "object",
                "properties": {"secret": {"type": "string"}},
                "required": ["secret"],
            },
            handler,
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(bypass_approval=True),
        event_sink=sink,
    )
    outcome = asyncio.run(
        executor.execute(
            call_id="bypass-call",
            identity="system:danger",
            arguments={"secret": "not-in-audit"},
            context=ToolCallContext("profile", "session", "turn"),
        )
    )

    assert outcome.status == "completed"
    assert calls == [{"secret": "not-in-audit"}]
    assert sequence[:2] == ["audit", "handler"]
    assert events[0] == (
        "audit",
        "audit",
        {"event": "approval_bypassed", "authorization": "bypass_approval"},
    )
    assert events[1][0] == "completed"
    records = executor.audit_events()
    assert len(records) == 1
    assert records[0].event == "approval_bypassed"
    assert records[0].identity == "system:danger"
    assert records[0].call_id == "bypass-call"
    assert all("not-in-audit" not in str(value) for value in events[0][2].values())


def test_bypass_fails_closed_when_audit_sink_fails() -> None:
    calls: list[str] = []

    async def handler(_arguments, _context):
        calls.append("called")
        return {"ok": True}

    def sink(_state, _outcome):
        raise RuntimeError("audit storage unavailable")

    registry = ToolRegistry()
    registry.register(
        ToolSpec("system:danger", "danger", {"type": "object"}, handler, RiskLevel.HIGH)
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(bypass_approval=True),
        event_sink=sink,
    )
    outcome = asyncio.run(
        executor.execute(
            call_id="audit-failure",
            identity="system:danger",
            arguments={},
            context=ToolCallContext("profile", "session", "turn"),
        )
    )

    assert outcome.status == "denied"
    assert outcome.content == {"error": "authorization audit failed"}
    assert not calls
    assert executor.audit_events()[0].event == "approval_bypassed"


def test_builtin_speak_awaits_dispatched_visual_update() -> None:
    class Pet:
        def __init__(self):
            self.messages = []

        def set_speech(self, text, *, mood="neutral"):
            self.messages.append((text, mood))
            return {"status": "available"}

    class Dispatcher:
        def __init__(self):
            self.calls = 0

        def invoke(self, function, *args, **kwargs):
            async def run():
                self.calls += 1
                return function(*args, **kwargs)

            return run()

    class TTS:
        async def enqueue_text(self, _context, text, **_kwargs):
            return (text,)

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    pet = Pet()
    dispatcher = Dispatcher()
    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(pet, dispatcher),
        tts=TTS(),
    )
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService()).execute(
            call_id="speak-dispatched",
            identity="pet:speak",
            arguments={"text": "你好"},
            context=ToolCallContext("p", "s", "t"),
        )
    )
    assert outcome.status == "completed"
    assert outcome.content["rendered"] is True
    assert dispatcher.calls == 1
    assert pet.messages == [("你好", "neutral")]


def test_behavior_reuses_no_turn_quota_between_ticks() -> None:
    async def handler(_arguments, _context):
        return {"status": "completed"}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "pet:ping",
            "ping",
            {"type": "object", "properties": {}, "additionalProperties": False},
            handler,
            RiskLevel.LOW,
        )
    )
    executor = ToolExecutionService(registry, PermissionService(), max_calls_per_turn=8)
    behavior = BehaviorService(
        executor,
        actions=(BehaviorAction("pet:ping", {}, 1.0),),
        probability=1.0,
        random_source=random.Random(0),
    )

    async def scenario():
        return [await behavior.tick() for _ in range(10)]

    outcomes = asyncio.run(scenario())
    assert all(getattr(outcome, "status", "") == "completed" for outcome in outcomes)


def test_session_grant_is_scoped_to_profile() -> None:
    async def handler(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(ToolSpec("pet:move", "move", {"type": "object"}, handler, RiskLevel.MEDIUM))
    permissions = PermissionService()
    executor = ToolExecutionService(registry, permissions)
    profile_one = ToolCallContext("profile-one", "session", "turn-one")
    first = asyncio.run(
        executor.execute(
            call_id="first",
            identity="pet:move",
            arguments={},
            context=profile_one,
        )
    )
    assert first.approval is not None
    permissions.approve(first.approval.approval_id, grant_session=True)
    granted = asyncio.run(
        executor.execute(
            call_id="second",
            identity="pet:move",
            arguments={},
            context=ToolCallContext("profile-one", "session", "turn-two"),
        )
    )
    isolated = asyncio.run(
        executor.execute(
            call_id="third",
            identity="pet:move",
            arguments={},
            context=ToolCallContext("profile-two", "session", "turn-one"),
        )
    )
    assert granted.status == "completed"
    assert isolated.status == "approval_required"


def test_pending_approvals_exposes_only_immutable_requests_in_expiry_order() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec("pet:move", "move", {"type": "object"}, lambda *_: {"ok": True}, RiskLevel.MEDIUM)
    )
    executor = ToolExecutionService(registry, PermissionService())

    async def scenario() -> tuple[str, str]:
        first = await executor.execute(
            call_id="approval-later",
            identity="pet:move",
            arguments={},
            context=ToolCallContext("p", "s", "turn-later"),
        )
        second = await executor.execute(
            call_id="approval-earlier",
            identity="pet:move",
            arguments={},
            context=ToolCallContext("p", "s", "turn-earlier"),
        )
        assert first.approval is not None
        assert second.approval is not None
        requests = executor.pending_approvals()
        assert {request.call_id for request in requests} == {
            "approval-later",
            "approval-earlier",
        }
        # 不可变审批请求只暴露展示字段，执行器内部参数快照不能从列表取得。
        assert all(request.identity == "pet:move" for request in requests)
        return first.approval.approval_id, second.approval.approval_id

    first_id, second_id = asyncio.run(scenario())
    assert executor.deny_approval(first_id) is not None
    assert [request.approval_id for request in executor.pending_approvals()] == [second_id]


def test_approval_terminal_states_are_detailed_execution_records() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:audit_terminal",
            "audit terminal",
            {"type": "object", "properties": {"value": {"type": "string"}}},
            lambda *_: {"ok": True},
            RiskLevel.HIGH,
            kind=ToolKind.SYSTEM,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    context = ToolCallContext("profile", "session", "audit-terminal-turn")

    async def request(call_id: str):
        return await executor.execute(
            call_id=call_id,
            identity="system:audit_terminal",
            arguments={"value": "kept-in-audit"},
            context=context,
        )

    pending = asyncio.run(request("deny-call"))
    assert pending.approval is not None
    assert executor.deny_approval(pending.approval.approval_id) is not None
    denied = executor.execution_records()[-1]
    assert denied.phase == "approval_denied"
    assert denied.status == "denied"
    assert denied.arguments == {"value": "kept-in-audit"}
    assert denied.error == "approval denied by user"

    shutdown_pending = asyncio.run(request("shutdown-call"))
    assert shutdown_pending.approval is not None
    executor.clear_pending()
    shutdown = executor.execution_records()[-1]
    assert shutdown.phase == "runtime_shutdown"
    assert shutdown.status == "denied"
    assert shutdown.call_id == "shutdown-call"
    assert shutdown.error == "tool call cancelled during shutdown"


def test_pending_approvals_can_be_filtered_to_one_conversation_context() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "danger",
            {"type": "object"},
            lambda *_: {"ok": True},
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())

    async def scenario():
        first = await executor.execute(
            call_id="approval-one",
            identity="system:danger",
            arguments={},
            context=ToolCallContext(
                "profile", "session", "turn-one", metadata={"mode": "direct", "generation_id": 1}
            ),
        )
        second = await executor.execute(
            call_id="approval-two",
            identity="system:danger",
            arguments={},
            context=ToolCallContext(
                "profile", "session", "turn-two", metadata={"mode": "direct", "generation_id": 2}
            ),
        )
        assert first.approval is not None and second.approval is not None
        filtered = executor.pending_approvals_for_context(
            ToolCallContext(
                "profile", "session", "turn-one", metadata={"mode": "direct", "generation_id": 1}
            )
        )
        assert [item.call_id for item in filtered] == ["approval-one"]

    asyncio.run(scenario())


def test_approval_resume_uses_immutable_pending_arguments_and_is_single_use() -> None:
    calls: list[tuple[dict, ToolCallContext]] = []

    async def handler(arguments, context):
        calls.append((dict(arguments), context))
        return {"value": arguments["value"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "danger",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            handler,
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    original = ToolCallContext(
        "profile",
        "session",
        "turn",
        source="model",
        metadata={"mode": "agent", "generation_id": 4},
    )

    async def scenario():
        first = await executor.execute(
            call_id="approval-call",
            identity="system:danger",
            arguments={"value": 7},
            context=original,
        )
        assert first.status == "approval_required"
        assert first.approval is not None
        original.metadata["generation_id"] = 999
        # 对话 finally 会清理调用计数，但不能丢弃等待审批的不可变快照。
        executor.clear_turn(original)
        wrong_context = ToolCallContext(
            "other-profile",
            "session",
            "turn",
            source="model",
            metadata={"mode": "agent", "generation_id": 4},
        )
        rejected = await executor.approve_and_execute(
            first.approval.approval_id,
            context=wrong_context,
        )
        assert rejected.status == "denied"
        assert not calls
        resumed = await executor.approve_and_execute(
            first.approval.approval_id,
            context=ToolCallContext(
                "profile",
                "session",
                "turn",
                source="model",
                metadata={"mode": "agent", "generation_id": 4},
            ),
        )
        assert resumed.status == "completed"
        assert resumed.content == {"value": 7}
        assert calls[0][0] == {"value": 7}
        assert calls[0][1].metadata["generation_id"] == 4
        duplicate = await executor.approve_and_execute(first.approval.approval_id)
        assert duplicate.status == "denied"
        assert len(calls) == 1

    asyncio.run(scenario())


def test_approval_snapshot_is_deep_and_binds_the_context_source() -> None:
    calls: list[tuple[dict[str, object], ToolCallContext]] = []

    async def handler(arguments, context):
        calls.append((dict(arguments), context))
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "danger",
            {
                "type": "object",
                "properties": {
                    "payload": {
                        "type": "object",
                        "properties": {"target": {"type": "string"}},
                        "required": ["target"],
                    }
                },
                "required": ["payload"],
            },
            handler,
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    arguments = {"payload": {"target": "original"}}
    context = ToolCallContext(
        "profile",
        "session",
        "turn",
        source="model",
        metadata={"mode": "agent", "generation_id": 3, "nested": {"state": "original"}},
    )

    async def scenario() -> None:
        outcome = await executor.execute(
            call_id="deep-snapshot",
            identity="system:danger",
            arguments=arguments,
            context=context,
        )
        assert outcome.approval is not None
        arguments["payload"]["target"] = "modified"
        context.metadata["nested"]["state"] = "modified"
        wrong_source = ToolCallContext(
            "profile",
            "session",
            "turn",
            source="scheduler",
            metadata={"mode": "agent", "generation_id": 3},
        )
        rejected = await executor.approve_and_execute(
            outcome.approval.approval_id,
            context=wrong_source,
        )
        assert rejected.status == "denied"
        resumed = await executor.approve_and_execute(
            outcome.approval.approval_id,
            context=ToolCallContext(
                "profile",
                "session",
                "turn",
                source="model",
                metadata={"mode": "agent", "generation_id": 3},
            ),
        )
        assert resumed.status == "completed"

    asyncio.run(scenario())
    assert calls[0][0] == {"payload": {"target": "original"}}
    assert calls[0][1].metadata["nested"] == {"state": "original"}


def test_approval_expiry_uses_permission_service_clock() -> None:
    permission_now = [100.0]
    execution_now = [10_000.0]
    calls: list[str] = []

    async def handler(_arguments, _context):
        calls.append("called")
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec("system:danger", "danger", {"type": "object"}, handler, RiskLevel.HIGH)
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(clock=lambda: permission_now[0]),
        clock=lambda: execution_now[0],
    )

    async def scenario() -> None:
        outcome = await executor.execute(
            call_id="separate-clocks",
            identity="system:danger",
            arguments={},
            context=ToolCallContext("p", "s", "t"),
        )
        assert outcome.approval is not None
        resumed = await executor.approve_and_execute(outcome.approval.approval_id)
        assert resumed.status == "completed"

    asyncio.run(scenario())
    assert calls == ["called"]


def test_approval_resume_rejects_expired_snapshot() -> None:
    now = [100.0]
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:danger",
            "danger",
            {"type": "object"},
            lambda _arguments, _context: {"ok": True},
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(registry, PermissionService(clock=lambda: now[0]))
    context = ToolCallContext("p", "s", "t")

    async def scenario():
        outcome = await executor.execute(
            call_id="expired",
            identity="system:danger",
            arguments={},
            context=context,
        )
        assert outcome.approval is not None
        now[0] += 1000
        retried = await executor.execute(
            call_id="expired-retry",
            identity="system:danger",
            arguments={},
            context=context,
        )
        assert retried.status == "approval_required"
        assert retried.approval is not None
        resumed = await executor.approve_and_execute(outcome.approval.approval_id)
        assert resumed.status == "denied"
        assert "missing or expired" in resumed.content["error"]

    asyncio.run(scenario())


def test_expired_approval_is_a_detailed_terminal_execution_record() -> None:
    now = [100.0]
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:expiring",
            "expiring",
            {"type": "object"},
            lambda _arguments, _context: {"ok": True},
            RiskLevel.HIGH,
        )
    )
    permissions = PermissionService(clock=lambda: now[0])
    executor = ToolExecutionService(registry, permissions)
    context = ToolCallContext("profile", "session", "expiry-audit")

    async def scenario() -> None:
        pending = await executor.execute(
            call_id="expiry-call",
            identity="system:expiring",
            arguments={"secret": "must stay bounded"},
            context=context,
        )
        assert pending.status == "approval_required"
        assert pending.approval is not None
        now[0] += 1000.0
        assert executor.pending_approvals() == ()
        # 重复读取不会重复写入同一个过期终态。
        assert executor.pending_approvals() == ()

    asyncio.run(scenario())
    records = [item for item in executor.execution_records() if item.call_id == "expiry-call"]
    assert len(records) == 2
    assert records[0].phase == "awaiting_approval"
    assert records[1].phase == "approval_expired"
    assert records[1].status == "denied"
    assert records[1].error == "approval request expired"


def test_scheduler_stop_cancels_blocked_action() -> None:
    now = [0.0]
    started = asyncio.Event()

    async def run(_action, _task):
        started.set()
        await asyncio.Event().wait()

    scheduler = SchedulerService(action_runner=run, clock=lambda: now[0])
    scheduler.upsert(
        task_id="blocked",
        name="blocked",
        expression="every:1s",
        action={"identity": "pet:ping"},
        owner="test",
    )
    now[0] = 2.0

    async def scenario():
        await scheduler.start(poll_seconds=0.01)
        await started.wait()
        await asyncio.wait_for(scheduler.stop(), timeout=0.2)

    asyncio.run(scenario())
    assert not scheduler.status()["running"]


def test_behavior_stop_cancels_blocked_action() -> None:
    started = asyncio.Event()

    class Executor:
        async def execute(self, **_kwargs):
            started.set()
            await asyncio.Event().wait()

        def clear_turn(self, _context):
            return None

    behavior = BehaviorService(
        Executor(),
        actions=(BehaviorAction("pet:ping", {}, 1.0),),
        probability=1.0,
        random_source=random.Random(0),
    )

    async def scenario():
        await behavior.start()
        await started.wait()
        await asyncio.wait_for(behavior.stop(), timeout=0.2)

    asyncio.run(scenario())
    assert not behavior.status()["running"]


def test_run_command_cancellation_terminates_subprocess(tmp_path: Path) -> None:
    marker = tmp_path / "command-finished"
    script = (
        "import pathlib,time; time.sleep(30); "
        "pathlib.Path(__import__('sys').argv[1]).write_text('done')"
    )

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        command_allowlist=(sys.executable,),
    )
    executor = ToolExecutionService(registry, PermissionService(bypass_approval=True))
    context = ToolCallContext("p", "s", "command-cancel")

    async def scenario():
        task = asyncio.create_task(
            executor.execute(
                call_id="command-cancel",
                identity="system:run_command",
                arguments={
                    "command": [sys.executable, "-c", script, str(marker)],
                    "timeout_seconds": 30,
                },
                context=context,
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("command task completed unexpectedly")

    asyncio.run(scenario())
    assert not marker.exists()


def test_run_command_cancellation_terminates_child_process_group(tmp_path: Path) -> None:
    if os.name != "posix":
        return
    child_pid_path = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
    )

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        command_allowlist=(sys.executable,),
    )
    executor = ToolExecutionService(registry, PermissionService(bypass_approval=True))
    context = ToolCallContext("p", "s", "command-child-cancel")

    async def scenario() -> int:
        task = asyncio.create_task(
            executor.execute(
                call_id="command-child-cancel",
                identity="system:run_command",
                arguments={
                    "command": [sys.executable, "-c", script, str(child_pid_path)],
                    "timeout_seconds": 30,
                },
                context=context,
            )
        )
        deadline = time.monotonic() + 2.0
        child_pid_text = ""
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            if child_pid_path.exists():
                child_pid_text = child_pid_path.read_text(encoding="utf-8").strip()
                if child_pid_text:
                    break
        assert child_pid_text
        child_pid = int(child_pid_text)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return child_pid

    child_pid = asyncio.run(scenario())
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("child process survived command cancellation")


def test_run_command_reaps_child_holding_pipes_after_parent_exit(tmp_path: Path) -> None:
    if os.name != "posix":
        return
    child_pid_path = tmp_path / "pipe-child.pid"
    script = (
        "import pathlib, subprocess, sys; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid))"
    )

    class Platform:
        def foreground_window(self):
            return {"status": "unavailable"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=Platform(),
        pet_controller=MutablePetController(),
        command_allowlist=(sys.executable,),
    )
    outcome = asyncio.run(
        ToolExecutionService(registry, PermissionService(bypass_approval=True)).execute(
            call_id="command-pipe-child",
            identity="system:run_command",
            arguments={
                "command": [sys.executable, "-c", script, str(child_pid_path)],
                "timeout_seconds": 0.2,
            },
            context=ToolCallContext("p", "s", "command-pipe-child"),
        )
    )
    assert outcome.status == "completed"
    assert child_pid_path.exists()
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.02)
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("child holding output pipes survived parent exit")


def test_foreground_observation_cancellation_keeps_one_recoverable_read() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    class BlockingPlatform:
        def foreground_window(self):
            nonlocal calls
            calls += 1
            started.set()
            release.wait(timeout=2)
            return {"status": "available", "window_id": "one"}

        def list_processes(self, limit=20):
            return {"status": "available", "processes": []}

        def capture_screen(self, **_kwargs):
            return {"status": "unavailable"}

    registry = ToolRegistry()
    register_builtin_tools(
        registry,
        platform=BlockingPlatform(),
        pet_controller=MutablePetController(),
    )
    executor = ToolExecutionService(registry, PermissionService())
    context = ToolCallContext("p", "s", "observe-cancel")

    async def scenario():
        first = asyncio.create_task(
            executor.execute(
                call_id="observe-one",
                identity="desktop:observe_foreground",
                arguments={},
                context=context,
            )
        )
        await asyncio.to_thread(started.wait, 1)
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        second = await executor.execute(
            call_id="observe-two",
            identity="desktop:observe_foreground",
            arguments={},
            context=context,
        )
        assert second.content["status"] == "degraded"
        assert second.content["recoverable"] is True
        assert calls == 1

    try:
        asyncio.run(scenario())
    finally:
        release.set()


def test_tool_execution_records_capture_redacted_arguments_results_and_timing() -> None:
    received = []

    async def handler(arguments, _context):
        return {
            "ok": True,
            "value": arguments["value"],
            "secret": "result-secret",
        }

    async def record_sink(record):
        received.append(record)

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:detailed_record",
            "detailed record",
            {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "secret": {"type": "string"},
                },
                "required": ["value", "secret"],
            },
            handler,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(auto_allow_low_risk=True),
        execution_record_sink=record_sink,
    )

    outcome = asyncio.run(
        executor.execute(
            call_id="detailed-call",
            identity="system:detailed_record",
            arguments={"value": 7, "secret": "argument-secret"},
            context=ToolCallContext("profile", "session", "detailed-turn"),
        )
    )

    assert outcome.status == "completed"
    records = executor.execution_records()
    assert len(records) == 1
    record = records[0]
    assert record.identity == "system:detailed_record"
    assert record.call_id == "detailed-call"
    assert record.arguments == {"value": 7, "secret": "[redacted]"}
    assert record.result == {
        "ok": True,
        "value": 7,
        "secret": "[redacted]",
    }
    assert record.error == ""
    assert record.duration_ms is not None and record.duration_ms >= 0
    assert record.started_at is not None
    assert record.finished_at is not None
    assert record.attempt == 1
    assert record.retry_count == 0
    assert record.sequence == 1
    assert received == [record]
    assert record.as_dict()["depends_on"] == []
    assert "argument-secret" not in repr(record)
    assert "result-secret" not in repr(record)


def test_tool_execution_records_plan_retry_and_parallel_relationships() -> None:
    calls = 0

    async def flaky(_arguments, _context):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("transient")
        return {"ok": True}

    async def read(arguments, _context):
        return {"name": arguments["name"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:retry_record",
            "retry record",
            {"type": "object"},
            flaky,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    registry.register(
        ToolSpec(
            "system:parallel_record",
            "parallel record",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            read,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(auto_allow_low_risk=True),
    )
    context = ToolCallContext("profile", "session", "record-plan")
    result = asyncio.run(
        executor.execute_plan(
            (
                ToolPlanStep(
                    "retry",
                    "retry-call",
                    "system:retry_record",
                    max_attempts=2,
                ),
                ToolPlanStep(
                    "parallel-a",
                    "parallel-a-call",
                    "system:parallel_record",
                    {"name": "a"},
                    depends_on=("retry",),
                ),
                ToolPlanStep(
                    "parallel-b",
                    "parallel-b-call",
                    "system:parallel_record",
                    {"name": "b"},
                    depends_on=("retry",),
                ),
            ),
            context=context,
        )
    )

    assert result.outcomes[0].status == "completed"
    records = executor.execution_records()
    retry_records = [item for item in records if item.step_id == "retry"]
    assert [item.attempt for item in retry_records] == [1, 2]
    assert [item.retry_count for item in retry_records] == [0, 1]
    assert retry_records[0].status == "failed"
    assert retry_records[-1].status == "completed"
    assert retry_records[0].plan_id
    assert retry_records[0].plan_id == retry_records[-1].plan_id
    assert retry_records[0].parallel_group == 1
    assert retry_records[0].source == "agent"
    assert retry_records[0].depends_on == ()
    parallel_records = [item for item in records if item.step_id in {"parallel-a", "parallel-b"}]
    assert len(parallel_records) == 2
    assert {item.parallel_group for item in parallel_records} == {2}
    assert {item.plan_id for item in parallel_records} == {retry_records[0].plan_id}


def test_tool_execution_records_batch_membership_and_skipped_calls() -> None:
    async def read(arguments, _context):
        return {"value": arguments["value"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:batch_record",
            "batch record",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            read,
            RiskLevel.LOW,
            read_only=True,
        )
    )
    executor = ToolExecutionService(
        registry,
        PermissionService(auto_allow_low_risk=True),
    )
    outcomes = asyncio.run(
        executor.execute_batch(
            (
                {
                    "call_id": "batch-a",
                    "identity": "system:batch_record",
                    "arguments": {"value": 1},
                },
                {
                    "call_id": "batch-b",
                    "identity": "system:batch_record",
                    "arguments": {"value": 2},
                },
            ),
            context=ToolCallContext("profile", "session", "batch-record"),
        )
    )

    assert [item.status for item in outcomes] == ["completed", "completed"]
    records = executor.execution_records()
    assert len(records) == 2
    assert {item.batch_index for item in records} == {0, 1}
    assert {item.parallel_group for item in records} == {1}
    assert len({item.batch_id for item in records}) == 1
    assert [item.arguments["value"] for item in records] == [1, 2]


def test_tool_execution_records_batch_skips_after_approval() -> None:
    async def write(arguments, _context):
        return {"value": arguments["value"]}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:batch_high_record",
            "batch high record",
            {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
            write,
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    outcomes = asyncio.run(
        executor.execute_batch(
            (
                {
                    "call_id": "batch-high-a",
                    "identity": "system:batch_high_record",
                    "arguments": {"value": 1},
                },
                {
                    "call_id": "batch-high-b",
                    "identity": "system:batch_high_record",
                    "arguments": {"value": 2},
                },
            ),
            context=ToolCallContext("profile", "session", "batch-skip-record"),
        )
    )

    assert [item.status for item in outcomes] == ["approval_required", "denied"]
    records = executor.execution_records()
    assert len(records) == 2
    skipped = next(item for item in records if item.call_id == "batch-high-b")
    assert skipped.phase == "skipped"
    assert skipped.arguments == {"value": 2}
    assert skipped.batch_index == 1
    assert skipped.batch_id == records[0].batch_id


def test_tool_execution_records_plan_skips_with_dependencies() -> None:
    async def action(_arguments, _context):
        return {"ok": True}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "system:plan_skip_record",
            "plan skip record",
            {"type": "object"},
            action,
            RiskLevel.HIGH,
        )
    )
    executor = ToolExecutionService(registry, PermissionService())
    result = asyncio.run(
        executor.execute_plan(
            (
                ToolPlanStep("first", "plan-first", "system:plan_skip_record"),
                ToolPlanStep(
                    "second",
                    "plan-second",
                    "system:plan_skip_record",
                    depends_on=("first",),
                ),
            ),
            context=ToolCallContext(
                "profile",
                "session",
                "plan-skip-record",
                metadata={"mode": "agent", "generation_id": 9},
            ),
        )
    )

    assert [item.status for item in result.outcomes] == ["approval_required", "denied"]
    records = executor.execution_records()
    assert len(records) == 2
    first, second = records
    assert first.phase == "awaiting_approval"
    assert first.risk == "high"
    assert first.mode == "agent"
    assert first.generation_id == 9
    assert second.phase == "skipped"
    assert second.step_id == "second"
    assert second.depends_on == ("first",)
    assert second.plan_id == first.plan_id
    assert second.tool_kind == "local"
    assert second.risk == "high"
