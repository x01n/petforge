"""工具调用去重、限额、权限和串行策略。"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from concurrent.futures import Future as ConcurrentFuture
from copy import deepcopy
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, cast

from logger.events import event_fingerprint, log_event

from .permissions import MAX_PENDING_APPROVALS, PermissionDecision, PermissionService
from .registry import ToolRegistry
from .signature import ToolInvocationSignature, canonical_tool_arguments
from .types import (
    MAX_TOOL_PLAN_STEPS,
    ApprovalRequest,
    RiskLevel,
    ToolCallContext,
    ToolPlanStep,
    ToolSpec,
)
from .validation import validate_arguments

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolOutcome:
    status: str
    identity: str
    call_id: str
    content: Mapping[str, Any]
    approval: ApprovalRequest | None = None


@dataclass(frozen=True)
class ToolExecutionRecord:
    """一次工具调用的详细、脱敏且有界的执行记录。"""

    identity: str
    call_id: str
    status: str
    tool_kind: str = ""
    risk: str = ""
    read_only: bool = False
    arguments: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] = field(default_factory=dict)
    error: str = ""
    error_type: str = ""
    signature_digest: str = ""
    profile_id: str = ""
    session_id: str = ""
    turn_id: str = ""
    source: str = ""
    mode: str = ""
    generation_id: int = 0
    context: ToolCallContext | None = field(default=None, repr=False, compare=False)
    started_at: float | None = None
    finished_at: float | None = None
    duration_ms: float | None = None
    attempt: int = 1
    max_attempts: int = 1
    retry_count: int = 0
    plan_id: str = ""
    step_id: str = ""
    depends_on: tuple[str, ...] = ()
    parallel_group: int = 0
    phase: str = "execute"
    batch_id: str = ""
    batch_index: int = -1
    sequence: int = 0

    def as_dict(self) -> dict[str, Any]:
        """返回可序列化的详细审计快照副本。"""

        return {
            "identity": self.identity,
            "call_id": self.call_id,
            "status": self.status,
            "tool_kind": self.tool_kind,
            "risk": self.risk,
            "read_only": self.read_only,
            "arguments": deepcopy(dict(self.arguments)),
            "result": deepcopy(dict(self.result)),
            "error": self.error,
            "error_type": self.error_type,
            "signature_digest": self.signature_digest,
            "profile_id": self.profile_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "source": self.source,
            "mode": self.mode,
            "generation_id": self.generation_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "retry_count": self.retry_count,
            "plan_id": self.plan_id,
            "step_id": self.step_id,
            "depends_on": list(self.depends_on),
            "parallel_group": self.parallel_group,
            "phase": self.phase,
            "batch_id": self.batch_id,
            "batch_index": self.batch_index,
            "sequence": self.sequence,
        }

    to_dict = as_dict


def _audit_text(value: object, *, limit: int = 256) -> str:
    rendered = str(value or "").replace("\x00", "").replace("\r", " ").replace("\n", " ")
    return rendered[: max(0, int(limit))]


def _audit_value(
    value: object,
    *,
    key: str = "",
    depth: int = 0,
) -> object:
    """递归复制审计值；秘密字段脱敏并限制大小，避免审计自身造成泄漏。"""

    normalized_key = str(key or "").strip().lower()
    secret_parts = (
        "secret",
        "password",
        "passwd",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "audio_base64",
        "image_base64",
        "access_key",
        "private_key",
    )
    if any(part in normalized_key for part in secret_parts):
        return "[redacted]"
    if depth >= 8:
        return "[depth-limited]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return _audit_text(value, limit=4096)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for index, (raw_key, nested) in enumerate(value.items()):
            if index >= 128:
                result["__truncated__"] = True
                break
            rendered_key = _audit_text(raw_key, limit=256)
            result[rendered_key] = _audit_value(
                nested,
                key=rendered_key,
                depth=depth + 1,
            )
        return result
    if isinstance(value, (list, tuple)):
        result_list = [_audit_value(item, depth=depth + 1) for item in value[:128]]
        if len(value) > 128:
            result_list.append("[truncated]")
        return result_list
    return type(value).__name__


def _audit_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    sanitized = _audit_value(value)
    return dict(sanitized) if isinstance(sanitized, Mapping) else {}


@dataclass(frozen=True)
class ToolAuditEvent:
    """不含工具参数的授权审计记录。"""

    event: str
    identity: str
    call_id: str
    profile_id: str
    session_id: str
    turn_id: str
    occurred_at: float
    signature_digest: str = ""


@dataclass(frozen=True)
class ToolPlanAudit:
    """计划步骤的脱敏执行摘要。"""

    step_id: str
    status: str
    depends_on: tuple[str, ...] = ()
    parallel_group: int = 0
    reason: str = ""
    attempts: int = 0


@dataclass(frozen=True)
class ToolPlanCheckpoint:
    """审批暂停时保留的不可变计划执行状态。"""

    steps: tuple[ToolPlanStep, ...]
    context: ToolCallContext
    outcomes: tuple[tuple[str, ToolOutcome], ...]
    audit: tuple[tuple[str, ToolPlanAudit], ...]
    pending_step_id: str
    parallel_group: int
    deadline: float
    parallel_read_only: bool
    stop_on_approval: bool


@dataclass(frozen=True)
class ToolPlanResult:
    """计划按声明顺序返回的结果、审计摘要和可续接检查点。"""

    outcomes: tuple[ToolOutcome, ...]
    audit: tuple[ToolPlanAudit, ...]
    stopped: bool = False
    steps: tuple[ToolPlanStep, ...] = ()
    pending_approval: ApprovalRequest | None = None
    checkpoint: ToolPlanCheckpoint | None = field(default=None, repr=False, compare=False)


@dataclass(slots=True)
class _InvocationRecord:
    """同一回合内一次唯一调用及其 single-flight 结果。"""

    signature: ToolInvocationSignature
    future: ConcurrentFuture[ToolOutcome] = field(default_factory=ConcurrentFuture)
    outcome: ToolOutcome | None = None
    context: ToolCallContext | None = None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    started_monotonic: float | None = None


@dataclass(slots=True)
class _TurnInvocationState:
    """按身份、调用 ID 和规范参数三元组索引唯一调用。"""

    by_signature: dict[ToolInvocationSignature, _InvocationRecord] = field(default_factory=dict)
    call_count: int = 0


@dataclass(frozen=True)
class _PendingInvocation:
    """审批暂停期间保存的不可变工具调用快照。"""

    spec: ToolSpec
    values: Mapping[str, Any]
    context: ToolCallContext
    call_id: str
    approval: ApprovalRequest
    signature: ToolInvocationSignature
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    started_monotonic: float | None = None


class ToolExecutionService:
    _AUDIT_EVENT_LIMIT = 1024
    _PENDING_APPROVAL_LIMIT = MAX_PENDING_APPROVALS

    def __init__(
        self,
        registry: ToolRegistry,
        permissions: PermissionService,
        *,
        max_calls_per_turn: int = 8,
        clock=time.time,
        event_sink: Callable[[str, ToolOutcome], Awaitable[None] | None] | None = None,
        execution_record_sink: (
            Callable[[ToolExecutionRecord], Awaitable[None] | None] | None
        ) = None,
    ) -> None:
        self.registry = registry
        self.permissions = permissions
        self.max_calls_per_turn = max(1, min(int(max_calls_per_turn), 64))
        self._clock = clock
        self._event_sink = event_sink
        self._execution_record_sink = execution_record_sink
        self._turn_invocations: dict[tuple[str, str, str, str, str], _TurnInvocationState] = {}
        # 这些状态既会在运行时事件循环中访问，也会被无 Qt 的同步审批入口读取。
        # 临界区内不等待协程，因此使用线程锁避免跨线程读写竞态。
        self._state_lock = RLock()
        self._side_effect_lock = asyncio.Lock()
        self._read_lock = asyncio.Semaphore(4)
        self._approval_lock = RLock()
        self._pending_approvals: dict[str, _PendingInvocation] = {}
        self._audit_lock = RLock()
        self._audit_events: list[ToolAuditEvent] = []
        self._execution_records: list[ToolExecutionRecord] = []
        self._execution_sequence = 0
        # 生命周期计时只按不可逆签名索引，不保存工具参数或原始 call_id。
        self._operation_started: dict[str, float] = {}

    async def _notify(self, state: str, outcome: ToolOutcome) -> None:
        if self._event_sink is None:
            return
        result = self._event_sink(state, outcome)
        if inspect.isawaitable(result):
            await result

    def _log_tool_event(
        self,
        event: str,
        *,
        identity: str,
        signature_digest: str,
        status: str = "",
        call_id: str = "",
        risk: object = None,
        error_type: str = "",
        reason_code: str = "",
        error_fingerprint: str = "",
    ) -> None:
        """记录成对工具生命周期；参数、结果和原始调用 ID 均不进入日志。"""

        normalized_event = str(event or "unknown").strip().lower()
        event_name = (
            normalized_event
            if normalized_event.startswith("tool.")
            else f"tool.call.{normalized_event}"
        )
        level = logging.INFO
        if normalized_event == "denied":
            level = logging.WARNING
        elif normalized_event in {"failed", "notification_failed"}:
            level = logging.ERROR
        spec = self.registry.get(identity)
        resolved_risk = risk if risk is not None else getattr(spec, "risk", None)
        risk_name = str(getattr(resolved_risk, "name", resolved_risk) or "unknown").lower()
        fields: dict[str, object] = {
            "identity": identity or "unknown",
            "risk": risk_name,
            "signature": signature_digest or "unavailable",
        }
        if error_type:
            fields["error_type"] = error_type
        if error_fingerprint:
            fields["error_fingerprint"] = error_fingerprint
        duration_ms: float | None = None
        if signature_digest:
            operation_fingerprint = event_fingerprint(call_id) or "anonymous"
            timer_key = f"{operation_fingerprint}:{signature_digest}"
            with self._state_lock:
                if normalized_event == "started":
                    self._operation_started[timer_key] = time.monotonic()
                elif normalized_event in {"completed", "denied", "failed", "cancelled"}:
                    started_at = self._operation_started.pop(timer_key, None)
                    if started_at is not None:
                        duration_ms = max(0.0, (time.monotonic() - started_at) * 1000.0)
        log_event(
            logger,
            event_name,
            component="tool",
            status=status or "unknown",
            level=level,
            correlation_id=call_id,
            operation_id=call_id,
            duration_ms=duration_ms,
            reason_code=reason_code or status or normalized_event,
            fields=fields,
        )

    async def _emit_outcome(
        self,
        state: str,
        outcome: ToolOutcome,
        *,
        signature_digest: str,
    ) -> None:
        embedded_status = str(outcome.content.get("status", "") or "").strip().lower()
        safe_embedded_status = (
            embedded_status
            if embedded_status
            and len(embedded_status) <= 64
            and all(character.isalnum() or character in "_.-" for character in embedded_status)
            else ""
        )
        raw_error = outcome.content.get("error")
        self._log_tool_event(
            state,
            identity=outcome.identity,
            signature_digest=signature_digest,
            status=outcome.status,
            call_id=outcome.call_id,
            reason_code=safe_embedded_status or outcome.status,
            error_fingerprint=event_fingerprint(raw_error) if raw_error else "",
        )
        await self._notify(state, outcome)

    def _claim_invocation(
        self,
        turn_key: tuple[str, str, str, str, str],
        signature: ToolInvocationSignature,
    ) -> tuple[_InvocationRecord, str | None, bool]:
        """原子登记唯一调用，返回记录、重复原因和是否超过回合限额。"""

        with self._state_lock:
            state = self._turn_invocations.get(turn_key)
            if state is None:
                if len(self._turn_invocations) >= 1024:
                    for old_key, old_state in tuple(self._turn_invocations.items()):
                        records = tuple(old_state.by_signature.values())
                        if all(
                            record.future.done()
                            and (
                                record.outcome is None
                                or record.outcome.status != "approval_required"
                            )
                            for record in records
                        ):
                            self._turn_invocations.pop(old_key, None)
                            break
                state = _TurnInvocationState()
                self._turn_invocations[turn_key] = state
            duplicate = state.by_signature.get(signature)
            if duplicate is not None:
                return duplicate, "exact_signature", False
            record = _InvocationRecord(signature)
            state.by_signature[signature] = record
            state.call_count += 1
            return record, None, state.call_count > self.max_calls_per_turn

    def _complete_invocation(self, record: _InvocationRecord, outcome: ToolOutcome) -> None:
        with self._state_lock:
            record.outcome = outcome
            if not record.future.done():
                record.future.set_result(outcome)

    def _abandon_invocation(
        self,
        *,
        turn_key: tuple[str, str, str, str, str],
        record: _InvocationRecord,
        outcome: ToolOutcome,
    ) -> None:
        """取消未完成调用后唤醒现有等待者，并允许后续显式重试。"""

        with self._state_lock:
            record.outcome = outcome
            if not record.future.done():
                record.future.set_result(outcome)
            state = self._turn_invocations.get(turn_key)
            if state is None:
                return
            if state.by_signature.get(record.signature) is record:
                state.by_signature.pop(record.signature, None)
            state.call_count = max(0, state.call_count - 1)
            if not state.by_signature:
                self._turn_invocations.pop(turn_key, None)

    def _replace_completed_outcome(
        self,
        *,
        context: ToolCallContext,
        signature: ToolInvocationSignature,
        outcome: ToolOutcome,
    ) -> None:
        """审批恢复后让后续重复调用取得最终执行结果。"""

        with self._state_lock:
            state = self._turn_invocations.get(self._turn_key(context))
            record = state.by_signature.get(signature) if state is not None else None
            if record is not None and record.signature == signature:
                record.outcome = outcome

    def _forget_invocation(
        self,
        *,
        context: ToolCallContext,
        signature: ToolInvocationSignature,
    ) -> None:
        """审批到期后删除去重索引，使用户能够显式重新发起审批。"""

        with self._state_lock:
            turn_key = self._turn_key(context)
            state = self._turn_invocations.get(turn_key)
            if state is None:
                return
            record = state.by_signature.get(signature)
            if record is None or record.signature != signature:
                return
            state.by_signature.pop(signature, None)
            state.call_count = max(0, state.call_count - 1)
            if not state.by_signature:
                self._turn_invocations.pop(turn_key, None)

    @staticmethod
    def _duplicate_outcome(
        signature: ToolInvocationSignature,
        original: ToolOutcome,
        *,
        reason: str,
    ) -> ToolOutcome:
        return ToolOutcome(
            "duplicate",
            signature.identity,
            signature.call_id,
            {
                "status": "duplicate",
                "duplicate_reason": reason,
                "duplicate_of_call_id": original.call_id,
                "original_status": original.status,
                "original_result": dict(original.content),
                "signature_digest": signature.digest,
            },
            original.approval,
        )

    async def _await_duplicate(
        self,
        signature: ToolInvocationSignature,
        record: _InvocationRecord,
        *,
        reason: str,
    ) -> ToolOutcome:
        with self._state_lock:
            original = record.outcome
        if original is None:
            original = await asyncio.shield(asyncio.wrap_future(record.future))
        outcome = self._duplicate_outcome(signature, original, reason=reason)
        await self._publish_outcome("duplicate", outcome, signature_digest=signature.digest)
        return outcome

    async def _publish_outcome(
        self,
        state: str,
        outcome: ToolOutcome,
        *,
        signature_digest: str,
    ) -> None:
        """事件接收器故障只降级通知；授权绕过审计仍由调用方单独 fail-close。"""

        try:
            await self._emit_outcome(state, outcome, signature_digest=signature_digest)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_tool_event(
                "notification_failed",
                identity=outcome.identity,
                signature_digest=signature_digest,
                status="degraded",
                call_id=outcome.call_id,
                error_type=type(exc).__name__,
                reason_code="event_sink_failed",
            )

    def audit_events(self) -> tuple[ToolAuditEvent, ...]:
        """返回有界、参数脱敏的授权审计快照。"""

        with self._audit_lock:
            return tuple(self._audit_events)

    def set_execution_record_sink(
        self,
        sink: Callable[[ToolExecutionRecord], Awaitable[None] | None] | None,
    ) -> None:
        """热替换详细执行记录接收器，不改变当前工具执行状态。"""

        if sink is not None and not callable(sink):
            raise TypeError("execution record sink must be callable or None")
        self._execution_record_sink = sink

    def execution_records(self) -> tuple[ToolExecutionRecord, ...]:
        """返回有界的详细工具执行快照，参数和结果已按字段脱敏。"""

        with self._audit_lock:
            return tuple(self._execution_records)

    def detailed_audit_events(self) -> tuple[ToolExecutionRecord, ...]:
        """返回详细执行记录；作为审计查询接口的语义化别名。"""

        return self.execution_records()

    def clear_execution_records(self) -> None:
        """清空详细执行快照，不影响授权审计或回合去重状态。"""

        with self._audit_lock:
            self._execution_records.clear()

    def _append_execution_record_sync(
        self,
        *,
        outcome: ToolOutcome,
        pending: _PendingInvocation,
        error_type: str = "",
        phase: str = "approval_denied",
    ) -> ToolExecutionRecord:
        """同步登记审批终态，再把持久化接收器交给当前事件循环。"""

        finished_monotonic = time.monotonic()
        try:
            finished_at = self._now()
        except RuntimeError:
            finished_at = None
        duration_ms = None
        if pending.started_monotonic is not None:
            duration_ms = max(
                0.0,
                (finished_monotonic - pending.started_monotonic) * 1000.0,
            )
        metadata = dict(pending.audit_metadata)
        context = pending.context
        context_metadata = getattr(context, "metadata", {})
        if not isinstance(context_metadata, Mapping):
            context_metadata = {}
        try:
            generation_id = max(
                0,
                min(
                    int(
                        metadata.get(
                            "generation_id",
                            context_metadata.get("generation_id", 0),
                        )
                        or 0
                    ),
                    1_000_000_000,
                ),
            )
        except (TypeError, ValueError, OverflowError):
            generation_id = 0
        try:
            attempt = max(1, min(int(metadata.get("attempt", 1)), 64))
        except (TypeError, ValueError, OverflowError):
            attempt = 1
        try:
            max_attempts = max(attempt, min(int(metadata.get("max_attempts", 1)), 64))
        except (TypeError, ValueError, OverflowError):
            max_attempts = attempt
        try:
            retry_count = max(0, min(int(metadata.get("retry_count", attempt - 1)), 63))
        except (TypeError, ValueError, OverflowError):
            retry_count = max(0, attempt - 1)
        raw_dependencies = metadata.get("depends_on", ())
        if isinstance(raw_dependencies, (str, bytes, bytearray)):
            dependencies: tuple[str, ...] = ()
        else:
            try:
                dependencies = tuple(_audit_text(value, limit=256) for value in raw_dependencies)
            except TypeError:
                dependencies = ()
        try:
            parallel_group = max(
                0,
                min(int(metadata.get("parallel_group", 0) or 0), 1_000_000),
            )
        except (TypeError, ValueError, OverflowError):
            parallel_group = 0
        try:
            batch_index = max(
                -1,
                min(int(metadata.get("batch_index", -1) or -1), 1_000_000),
            )
        except (TypeError, ValueError, OverflowError):
            batch_index = -1
        spec = pending.spec
        record = ToolExecutionRecord(
            identity=_audit_text(spec.identity, limit=256),
            call_id=_audit_text(pending.call_id, limit=256),
            status=_audit_text(outcome.status, limit=64),
            tool_kind=_audit_text(
                getattr(getattr(spec, "kind", None), "value", ""),
                limit=32,
            ),
            risk=_audit_text(
                getattr(getattr(spec, "risk", None), "name", ""),
                limit=32,
            ).lower(),
            read_only=bool(getattr(spec, "read_only", False)),
            arguments=_audit_mapping(pending.values),
            result=_audit_mapping(outcome.content),
            error=_audit_text(outcome.content.get("error", ""), limit=1024),
            error_type=_audit_text(error_type, limit=128),
            signature_digest=_audit_text(pending.signature.digest, limit=128),
            profile_id=_audit_text(getattr(context, "profile_id", ""), limit=256),
            session_id=_audit_text(getattr(context, "session_id", ""), limit=256),
            turn_id=_audit_text(getattr(context, "turn_id", ""), limit=256),
            source=_audit_text(
                metadata.get("source", getattr(context, "source", "")),
                limit=256,
            ),
            mode=_audit_text(
                metadata.get("mode", context_metadata.get("mode", "")),
                limit=32,
            ),
            generation_id=generation_id,
            context=context,
            started_at=pending.started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            attempt=attempt,
            max_attempts=max_attempts,
            retry_count=retry_count,
            plan_id=_audit_text(metadata.get("plan_id", ""), limit=256),
            step_id=_audit_text(metadata.get("step_id", ""), limit=256),
            depends_on=dependencies,
            parallel_group=parallel_group,
            phase=_audit_text(phase, limit=64),
            batch_id=_audit_text(metadata.get("batch_id", ""), limit=256),
            batch_index=batch_index,
            sequence=0,
        )
        with self._audit_lock:
            self._execution_sequence += 1
            record = replace(record, sequence=self._execution_sequence)
            self._execution_records.append(record)
            if len(self._execution_records) > self._AUDIT_EVENT_LIMIT:
                del self._execution_records[
                    : len(self._execution_records) - self._AUDIT_EVENT_LIMIT
                ]
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return record
        loop.create_task(
            self._emit_execution_record(record),
            name="meapet-tool-execution-audit",
        )
        return record

    async def _emit_execution_record(self, record: ToolExecutionRecord) -> None:
        """异步投递同步登记的执行记录；接收器故障只降低审计状态。"""

        sink = self._execution_record_sink
        if sink is None:
            return
        try:
            result = sink(record)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._log_tool_event(
                "notification_failed",
                identity=record.identity,
                signature_digest=record.signature_digest,
                status="degraded",
                call_id=record.call_id,
                error_type=type(exc).__name__,
                reason_code="execution_record_sink_failed",
            )

    async def _record_execution(
        self,
        *,
        outcome: ToolOutcome,
        arguments: Mapping[str, Any] | None,
        context: ToolCallContext | None,
        signature_digest: str = "",
        started_monotonic: float | None = None,
        started_at: float | None = None,
        error_type: str = "",
        audit_metadata: Mapping[str, Any] | None = None,
        status_override: str = "",
    ) -> ToolExecutionRecord:
        """保存一次工具执行的详细快照，并把记录交给可选审计接收器。"""

        finished_monotonic = time.monotonic()
        try:
            finished_at = self._now()
        except RuntimeError:
            finished_at = None
        duration_ms = None
        if started_monotonic is not None:
            duration_ms = max(0.0, (finished_monotonic - started_monotonic) * 1000.0)
        metadata = dict(audit_metadata) if isinstance(audit_metadata, Mapping) else {}
        raw_attempt = metadata.get("attempt", 1)
        raw_max_attempts = metadata.get("max_attempts", 1)
        try:
            attempt = int(raw_attempt)
        except (TypeError, ValueError, OverflowError):
            attempt = 1
        try:
            max_attempts = int(raw_max_attempts)
        except (TypeError, ValueError, OverflowError):
            max_attempts = 1
        attempt = max(1, min(attempt, 64))
        max_attempts = max(attempt, min(max_attempts, 64))
        raw_retry_count = metadata.get("retry_count", attempt - 1)
        try:
            retry_count = int(raw_retry_count)
        except (TypeError, ValueError, OverflowError):
            retry_count = attempt - 1
        retry_count = max(0, min(retry_count, 63))
        raw_parallel_group = metadata.get("parallel_group", 0)
        try:
            parallel_group = int(raw_parallel_group)
        except (TypeError, ValueError, OverflowError):
            parallel_group = 0
        parallel_group = max(0, min(parallel_group, 1_000_000))
        raw_batch_index = metadata.get("batch_index", -1)
        try:
            batch_index = int(raw_batch_index)
        except (TypeError, ValueError, OverflowError):
            batch_index = -1
        batch_index = max(-1, min(batch_index, 1_000_000))
        raw_dependencies = metadata.get("depends_on", ())
        if isinstance(raw_dependencies, (str, bytes, bytearray)):
            dependencies = ()
        else:
            try:
                dependencies = tuple(_audit_text(value, limit=256) for value in raw_dependencies)
            except TypeError:
                dependencies = ()
        raw_generation_id = metadata.get(
            "generation_id",
            getattr(context, "metadata", {}).get("generation_id", 0),
        )
        try:
            generation_id = int(raw_generation_id or 0)
        except (TypeError, ValueError, OverflowError):
            generation_id = 0
        generation_id = max(0, min(generation_id, 1_000_000_000))
        raw_mode = metadata.get(
            "mode",
            getattr(context, "metadata", {}).get("mode", ""),
        )
        mode = _audit_text(raw_mode, limit=32)
        spec = self.registry.get(outcome.identity)
        tool_kind = _audit_text(
            metadata.get(
                "tool_kind",
                getattr(getattr(spec, "kind", None), "value", ""),
            ),
            limit=32,
        )
        risk = _audit_text(
            metadata.get(
                "risk",
                getattr(getattr(spec, "risk", None), "name", ""),
            ),
            limit=32,
        ).lower()
        read_only = bool(metadata.get("read_only", getattr(spec, "read_only", False)))
        content = _audit_mapping(outcome.content)
        error_value = content.get("error", "")
        error = _audit_text(error_value, limit=1024) if error_value else ""
        record = ToolExecutionRecord(
            identity=_audit_text(outcome.identity, limit=256),
            call_id=_audit_text(outcome.call_id, limit=256),
            status=_audit_text(status_override or outcome.status, limit=64),
            tool_kind=tool_kind,
            risk=risk,
            read_only=read_only,
            arguments=_audit_mapping(arguments),
            result=content,
            error=error,
            error_type=_audit_text(error_type, limit=128),
            signature_digest=_audit_text(signature_digest, limit=128),
            profile_id=_audit_text(getattr(context, "profile_id", ""), limit=256),
            session_id=_audit_text(getattr(context, "session_id", ""), limit=256),
            turn_id=_audit_text(getattr(context, "turn_id", ""), limit=256),
            source=_audit_text(
                metadata.get("source", getattr(context, "source", "")),
                limit=256,
            ),
            mode=mode,
            generation_id=generation_id,
            context=context,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            attempt=attempt,
            max_attempts=max_attempts,
            retry_count=retry_count,
            plan_id=_audit_text(metadata.get("plan_id", ""), limit=256),
            step_id=_audit_text(metadata.get("step_id", ""), limit=256),
            depends_on=dependencies,
            parallel_group=parallel_group,
            phase=_audit_text(
                metadata.get(
                    "phase",
                    "awaiting_approval" if outcome.status == "approval_required" else "execute",
                ),
                limit=64,
            ),
            batch_id=_audit_text(metadata.get("batch_id", ""), limit=256),
            batch_index=batch_index,
            sequence=0,
        )
        with self._audit_lock:
            self._execution_sequence += 1
            record = replace(record, sequence=self._execution_sequence)
            self._execution_records.append(record)
            if len(self._execution_records) > self._AUDIT_EVENT_LIMIT:
                del self._execution_records[
                    : len(self._execution_records) - self._AUDIT_EVENT_LIMIT
                ]
        sink = self._execution_record_sink
        if sink is not None:
            try:
                result = sink(record)
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_tool_event(
                    "notification_failed",
                    identity=outcome.identity,
                    signature_digest=signature_digest,
                    status="degraded",
                    call_id=outcome.call_id,
                    error_type=type(exc).__name__,
                    reason_code="execution_record_sink_failed",
                )
        return record

    async def _record_audit(
        self,
        *,
        event: str,
        identity: str,
        call_id: str,
        context: ToolCallContext,
        signature_digest: str,
    ) -> None:
        """先记录授权事件，再允许后续副作用执行。"""

        occurred_at = self._now()
        record = ToolAuditEvent(
            event=event,
            identity=identity,
            call_id=call_id,
            profile_id=context.profile_id,
            session_id=context.session_id,
            turn_id=context.turn_id,
            occurred_at=occurred_at,
            signature_digest=signature_digest,
        )
        with self._audit_lock:
            self._audit_events.append(record)
            if len(self._audit_events) > self._AUDIT_EVENT_LIMIT:
                del self._audit_events[: len(self._audit_events) - self._AUDIT_EVENT_LIMIT]
        audit_outcome = ToolOutcome(
            "audit",
            identity,
            call_id,
            {"event": event, "authorization": "bypass_approval"},
        )
        self._log_tool_event(
            "approved" if event == "approval_bypassed" else event,
            identity=identity,
            signature_digest=signature_digest,
            status="bypassed" if event == "approval_bypassed" else "audit",
            call_id=call_id,
        )
        await self._notify("audit", audit_outcome)

    def _now(self) -> float:
        value = self._clock()
        if isinstance(value, bool):
            raise RuntimeError("tool execution clock must return a finite timestamp")
        try:
            current = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("tool execution clock must return a finite timestamp") from exc
        if not math.isfinite(current):
            raise RuntimeError("tool execution clock must return a finite timestamp")
        return current

    async def _invoke_authorized(
        self,
        *,
        spec: ToolSpec,
        values: Mapping[str, Any],
        context: ToolCallContext,
        call_id: str,
    ) -> ToolOutcome:
        """执行已经通过权限检查的调用并统一处理并发锁。"""

        async def invoke() -> Any:
            result = spec.handler(values, context)
            if inspect.isawaitable(result):
                return await result
            return result

        try:
            lock = self._read_lock if spec.read_only else self._side_effect_lock
            async with lock:
                raw_result = await invoke()
            content = raw_result if isinstance(raw_result, Mapping) else {"value": raw_result}
            content = dict(content)
            embedded_status = str(content.get("status", "")).strip().lower()
            failure_states = {"denied", "error", "failed", "unavailable"}
            if (
                not spec.read_only
                and embedded_status in failure_states
                and content.get("rendered") is not True
            ):
                outcome_status = "denied" if embedded_status == "denied" else "failed"
            else:
                outcome_status = "completed"
            return ToolOutcome(outcome_status, spec.identity, call_id, content)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolOutcome(
                "failed",
                spec.identity,
                call_id,
                {"error": f"tool failed: {type(exc).__name__}"},
            )

    async def _evaluate_unique_invocation(
        self,
        *,
        spec: ToolSpec,
        signature: ToolInvocationSignature,
        values: Mapping[str, Any],
        context: ToolCallContext,
        audit_metadata: Mapping[str, Any] | None = None,
        started_at: float | None = None,
        started_monotonic: float | None = None,
    ) -> ToolOutcome:
        """对已经完成 single-flight 登记的调用执行校验、授权和处理器。"""

        errors = validate_arguments(spec.parameters, values)
        if errors:
            return ToolOutcome(
                "denied",
                signature.identity,
                signature.call_id,
                {"error": "invalid arguments", "fields": [error.path for error in errors]},
            )
        safe_summary = f"需要确认：{spec.user_label}"
        try:
            permission = self.permissions.evaluate(
                spec,
                context,
                call_id=signature.call_id,
                safe_summary=safe_summary,
            )
        except (RuntimeError, TypeError, ValueError):
            return ToolOutcome(
                "denied",
                signature.identity,
                signature.call_id,
                {"error": "permission evaluation failed"},
            )
        if permission.decision is PermissionDecision.DENY:
            return ToolOutcome(
                "denied",
                signature.identity,
                signature.call_id,
                {"error": permission.reason},
            )
        if permission.decision is PermissionDecision.APPROVAL_REQUIRED:
            approval = permission.approval
            if approval is None:
                return ToolOutcome(
                    "denied",
                    signature.identity,
                    signature.call_id,
                    {"error": "approval request was not created"},
                )
            with self._approval_lock:
                self._prune_pending_locked()
                if len(self._pending_approvals) >= self._PENDING_APPROVAL_LIMIT:
                    oldest_id = next(iter(self._pending_approvals))
                    evicted = self._pending_approvals.pop(oldest_id)
                    self._forget_invocation(
                        context=evicted.context,
                        signature=evicted.signature,
                    )
                    try:
                        self.permissions.deny(oldest_id)
                    except (KeyError, RuntimeError, TypeError, ValueError):
                        pass
                    self._log_tool_event(
                        "denied",
                        identity=evicted.spec.identity,
                        signature_digest=evicted.signature.digest,
                        status="evicted",
                        call_id=evicted.call_id,
                        risk=evicted.spec.risk,
                    )
                self._pending_approvals[approval.approval_id] = _PendingInvocation(
                    spec=spec,
                    values=values,
                    context=context,
                    call_id=signature.call_id,
                    approval=approval,
                    signature=signature,
                    audit_metadata=dict(audit_metadata or {}),
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                )
            return ToolOutcome(
                "approval_required",
                signature.identity,
                signature.call_id,
                {"status": "waiting for confirmation"},
                approval,
            )
        if permission.bypassed:
            try:
                await self._record_audit(
                    event="approval_bypassed",
                    identity=signature.identity,
                    call_id=signature.call_id,
                    context=context,
                    signature_digest=signature.digest,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                return ToolOutcome(
                    "denied",
                    signature.identity,
                    signature.call_id,
                    {"error": "authorization audit failed"},
                )
        return await self._invoke_authorized(
            spec=spec,
            values=values,
            context=context,
            call_id=signature.call_id,
        )

    async def execute(
        self,
        *,
        call_id: str,
        identity: str,
        arguments: Mapping[str, Any] | None,
        context: ToolCallContext,
        audit_metadata: Mapping[str, Any] | None = None,
    ) -> ToolOutcome:
        """执行工具调用并保存详细的脱敏执行记录。"""

        call_id = str(call_id or "").strip()
        identity = str(identity or "").strip()
        started_monotonic = time.monotonic()
        try:
            started_at = self._now()
        except RuntimeError:
            started_at = None
        metadata = dict(audit_metadata) if isinstance(audit_metadata, Mapping) else {}
        snapshot_values: Mapping[str, Any] = (
            dict(arguments) if isinstance(arguments, Mapping) else {}
        )
        snapshot_context = context if isinstance(context, ToolCallContext) else None
        signature_digest = ""

        async def finish(
            outcome: ToolOutcome,
            *,
            publish: bool = True,
            error_type: str = "",
        ) -> ToolOutcome:
            await self._record_execution(
                outcome=outcome,
                arguments=snapshot_values,
                context=snapshot_context,
                signature_digest=signature_digest,
                started_monotonic=started_monotonic,
                started_at=started_at,
                error_type=error_type,
                audit_metadata=metadata,
            )
            if publish:
                await self._publish_outcome(
                    outcome.status,
                    outcome,
                    signature_digest=signature_digest,
                )
            return outcome

        if not call_id or not identity:
            return await finish(
                ToolOutcome("denied", identity, call_id, {"error": "invalid tool call"}),
            )
        if not isinstance(context, ToolCallContext):
            return await finish(
                ToolOutcome("denied", identity, call_id, {"error": "invalid tool context"}),
            )
        snapshot_context = self._snapshot_context(context)
        if arguments is not None and not isinstance(arguments, Mapping):
            return await finish(
                ToolOutcome("denied", identity, call_id, {"error": "invalid arguments"}),
            )
        spec = self.registry.get(identity)
        if spec is None:
            return await finish(
                ToolOutcome("denied", identity, call_id, {"error": "unknown tool"}),
            )
        metadata.setdefault("tool_kind", str(spec.kind.value))
        metadata.setdefault("risk", spec.risk.name.lower())
        metadata.setdefault("read_only", spec.read_only)
        try:
            snapshot_values = self._snapshot_mapping(dict(arguments or {}))
            signature = ToolInvocationSignature.create(
                identity=identity,
                call_id=call_id,
                arguments=snapshot_values,
            )
            signature_digest = signature.digest
        except ValueError:
            return await finish(
                ToolOutcome(
                    "denied",
                    identity,
                    call_id,
                    {"error": "tool invocation snapshot is invalid"},
                ),
            )
        with self._approval_lock:
            self._prune_pending_locked()
        turn_key = self._turn_key(snapshot_context)
        record, duplicate_reason, over_limit = self._claim_invocation(turn_key, signature)
        if duplicate_reason is not None:
            outcome = await self._await_duplicate(
                signature,
                record,
                reason=duplicate_reason,
            )
            return await finish(outcome, publish=False)
        with self._state_lock:
            record.context = snapshot_context
            record.arguments = dict(snapshot_values)
            record.started_at = started_at
            record.started_monotonic = started_monotonic
        self._log_tool_event(
            "started",
            identity=identity,
            signature_digest=signature.digest,
            status="running",
            call_id=call_id,
            risk=spec.risk,
        )
        if over_limit:
            outcome = ToolOutcome(
                "denied", identity, call_id, {"error": "tool call limit exceeded"}
            )
            self._complete_invocation(record, outcome)
            return await finish(outcome)
        try:
            outcome = await self._evaluate_unique_invocation(
                spec=spec,
                signature=signature,
                values=snapshot_values,
                context=snapshot_context,
                audit_metadata=metadata,
                started_at=started_at,
                started_monotonic=started_monotonic,
            )
        except asyncio.CancelledError:
            cancelled = ToolOutcome(
                "failed",
                identity,
                call_id,
                {"error": "tool call cancelled"},
            )
            self._abandon_invocation(turn_key=turn_key, record=record, outcome=cancelled)
            self._log_tool_event(
                "failed",
                identity=identity,
                signature_digest=signature.digest,
                status="cancelled",
                call_id=call_id,
                risk=spec.risk,
            )
            await self._record_execution(
                outcome=cancelled,
                arguments=snapshot_values,
                context=snapshot_context,
                signature_digest=signature.digest,
                started_monotonic=started_monotonic,
                started_at=started_at,
                error_type="CancelledError",
                audit_metadata=metadata,
            )
            raise
        except BaseException as exc:
            failed = ToolOutcome(
                "failed",
                identity,
                call_id,
                {"error": f"tool aborted: {type(exc).__name__}"},
            )
            self._abandon_invocation(turn_key=turn_key, record=record, outcome=failed)
            self._log_tool_event(
                "failed",
                identity=identity,
                signature_digest=signature.digest,
                status="aborted",
                call_id=call_id,
                risk=spec.risk,
                error_type=type(exc).__name__,
            )
            await self._record_execution(
                outcome=failed,
                arguments=snapshot_values,
                context=snapshot_context,
                signature_digest=signature.digest,
                started_monotonic=started_monotonic,
                started_at=started_at,
                error_type=type(exc).__name__,
                audit_metadata=metadata,
            )
            raise
        self._complete_invocation(record, outcome)
        return await finish(outcome)

    async def execute_batch(
        self,
        calls: Sequence[object],
        *,
        context: ToolCallContext,
        parallel_read_only: bool = True,
        stop_on_approval: bool = True,
    ) -> tuple[ToolOutcome, ...]:
        """按输入顺序执行一批工具调用，并安全合并可并行的只读调用。"""

        if isinstance(calls, (str, bytes, bytearray)) or not isinstance(calls, Sequence):
            raise ValueError("tool calls must be a sequence")
        batch = tuple(calls)
        if not batch:
            return ()
        prepared: list[tuple[object, object, object]] = []
        for call in batch:
            if isinstance(call, Mapping):
                call_id = call.get("call_id", "")
                identity = call.get("identity", "")
                arguments = call.get("arguments", {})
            else:
                call_id = getattr(call, "call_id", "")
                identity = getattr(call, "identity", "")
                arguments = getattr(call, "arguments", {})
            prepared.append((call_id, identity, arguments))

        batch_id = event_fingerprint(
            "|".join(
                (
                    str(getattr(context, "profile_id", "")),
                    str(getattr(context, "session_id", "")),
                    str(getattr(context, "turn_id", "")),
                    str(time.monotonic_ns()),
                )
            )
        )
        can_parallel = bool(parallel_read_only) and len(prepared) >= 2
        if can_parallel:
            probe = getattr(self.permissions, "allows_without_approval", None)
            if not callable(probe):
                can_parallel = False
            else:
                for _call_id, raw_identity, _arguments in prepared:
                    identity = str(raw_identity or "").strip()
                    spec = self.registry.get(identity)
                    if spec is None or not spec.read_only or spec.risk is not RiskLevel.LOW:
                        can_parallel = False
                        break
                    try:
                        if not bool(probe(spec, context)):
                            can_parallel = False
                            break
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        can_parallel = False
                        break
        parallel_group = 1 if can_parallel else 0
        if can_parallel:
            return tuple(
                await asyncio.gather(
                    *(
                        self.execute(
                            call_id=str(call_id or ""),
                            identity=str(identity or ""),
                            arguments=(arguments if isinstance(arguments, Mapping) else None),
                            context=context,
                            audit_metadata={
                                "batch_id": batch_id,
                                "batch_index": index,
                                "parallel_group": parallel_group,
                            },
                        )
                        for index, (call_id, identity, arguments) in enumerate(prepared)
                    )
                )
            )

        outcomes: list[ToolOutcome] = []
        for index, (call_id, identity, arguments) in enumerate(prepared):
            outcome = await self.execute(
                call_id=str(call_id or ""),
                identity=str(identity or ""),
                arguments=arguments if isinstance(arguments, Mapping) else None,
                context=context,
                audit_metadata={
                    "batch_id": batch_id,
                    "batch_index": index,
                    "parallel_group": parallel_group,
                },
            )
            outcomes.append(outcome)
            if not stop_on_approval or outcome.status != "approval_required":
                continue
            for skipped_index, (skipped_call_id, skipped_identity, skipped_arguments) in enumerate(
                prepared[index + 1 :],
                start=index + 1,
            ):
                skipped = ToolOutcome(
                    "denied",
                    str(skipped_identity or ""),
                    str(skipped_call_id or ""),
                    {"error": "前一项操作等待确认，后续操作未执行"},
                )
                await self._record_execution(
                    outcome=skipped,
                    arguments=(skipped_arguments if isinstance(skipped_arguments, Mapping) else {}),
                    context=context,
                    audit_metadata={
                        "batch_id": batch_id,
                        "batch_index": skipped_index,
                        "parallel_group": parallel_group,
                        "phase": "skipped",
                    },
                )
                outcomes.append(skipped)
            break
        return tuple(outcomes)

    def _normalize_plan_steps(
        self,
        steps: Sequence[ToolPlanStep],
    ) -> tuple[ToolPlanStep, ...]:
        """规范化并验证有限严格 DAG；不执行任何工具。"""

        if isinstance(steps, (str, bytes, bytearray)) or not isinstance(steps, Sequence):
            raise ValueError("tool plan steps must be a sequence")
        if len(steps) > MAX_TOOL_PLAN_STEPS:
            raise ValueError("tool plan contains too many steps")
        normalized: list[ToolPlanStep] = []
        allowed_fields = {
            "step_id",
            "call_id",
            "identity",
            "arguments",
            "depends_on",
            "timeout_seconds",
            "max_attempts",
            "on_error",
        }
        for raw_step in steps:
            if isinstance(raw_step, ToolPlanStep):
                step = raw_step
            else:
                if not isinstance(raw_step, Mapping):
                    raise ValueError("tool plan steps must contain ToolPlanStep values")
                if set(raw_step) - allowed_fields:
                    raise ValueError("tool plan step contains unknown fields")
                try:
                    step = ToolPlanStep(
                        step_id=raw_step["step_id"],
                        call_id=raw_step["call_id"],
                        identity=raw_step["identity"],
                        arguments=raw_step.get("arguments", {}),
                        depends_on=raw_step.get("depends_on", ()),
                        timeout_seconds=raw_step.get("timeout_seconds", 15.0),
                        max_attempts=raw_step.get("max_attempts", 1),
                        on_error=raw_step.get("on_error", "skip_dependents"),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("invalid tool plan step") from exc
            if step.max_attempts > 1:
                spec = self.registry.get(step.identity)
                if spec is None or not spec.read_only or spec.risk is not RiskLevel.LOW:
                    raise ValueError("tool plan retries require a low-risk read-only tool")
            normalized.append(step)

        by_id: dict[str, ToolPlanStep] = {}
        call_ids: set[str] = set()
        for step in normalized:
            if step.step_id in by_id:
                raise ValueError("tool plan contains duplicate step_id")
            if step.call_id in call_ids:
                raise ValueError("tool plan contains duplicate call_id")
            by_id[step.step_id] = step
            call_ids.add(step.call_id)
        for step in normalized:
            if len(set(step.depends_on)) != len(step.depends_on):
                raise ValueError("tool plan contains duplicate dependency")
            if step.step_id in step.depends_on:
                raise ValueError("tool plan contains self dependency")
            if any(dependency not in by_id for dependency in step.depends_on):
                raise ValueError("tool plan contains an unknown dependency")

        indegree = {step.step_id: len(step.depends_on) for step in normalized}
        dependents: dict[str, list[str]] = {step.step_id: [] for step in normalized}
        for step in normalized:
            for dependency in step.depends_on:
                dependents[dependency].append(step.step_id)
        frontier = [step.step_id for step in normalized if indegree[step.step_id] == 0]
        visited = 0
        while frontier:
            current = frontier.pop()
            visited += 1
            for dependent in dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    frontier.append(dependent)
        if visited != len(normalized):
            raise ValueError("tool plan contains a dependency cycle")
        return tuple(normalized)

    @staticmethod
    def _resolve_plan_arguments(
        step: ToolPlanStep,
        outcomes: Mapping[str, ToolOutcome],
    ) -> dict[str, Any]:
        """只从已声明且成功的直接依赖中解析结果路径。"""

        visited = 0

        def resolve(value: object, depth: int) -> object:
            nonlocal visited
            visited += 1
            if visited > 4096 or depth > 32:
                raise ValueError("tool plan arguments are too deeply nested")
            if isinstance(value, Mapping):
                if "$step_result" in value:
                    if set(value) != {"$step_result"}:
                        raise ValueError("tool plan result reference must be standalone")
                    reference = value["$step_result"]
                    if not isinstance(reference, Mapping) or set(reference) != {
                        "step_id",
                        "path",
                    }:
                        raise ValueError("tool plan result reference is invalid")
                    step_id = str(reference.get("step_id") or "").strip()
                    if step_id not in step.depends_on:
                        raise ValueError("tool plan result reference is not a declared dependency")
                    outcome = outcomes.get(step_id)
                    if outcome is None or outcome.status not in {"completed", "duplicate"}:
                        raise ValueError("tool plan result reference is unavailable")
                    current: object = outcome.content
                    if outcome.status == "duplicate":
                        original = outcome.content.get("original_result")
                        if isinstance(original, Mapping):
                            current = original
                    path = reference.get("path")
                    if isinstance(path, (str, bytes, bytearray)) or not isinstance(path, Sequence):
                        raise ValueError("tool plan result reference path is invalid")
                    if len(path) > 16:
                        raise ValueError("tool plan result reference path is too long")
                    for part in path:
                        if isinstance(part, bool):
                            raise ValueError("tool plan result reference path is invalid")
                        if isinstance(part, str):
                            if not isinstance(current, Mapping) or part not in current:
                                raise ValueError("tool plan result reference path is missing")
                            current = current[part]
                            continue
                        if isinstance(part, int):
                            if (
                                part < 0
                                or isinstance(current, (str, bytes, bytearray))
                                or not isinstance(current, Sequence)
                                or part >= len(current)
                            ):
                                raise ValueError("tool plan result reference index is missing")
                            current = cast(Sequence[object], current)[part]
                            continue
                        raise ValueError("tool plan result reference path is invalid")
                    return deepcopy(current)
                rendered: dict[str, object] = {}
                for key, nested in value.items():
                    if not isinstance(key, str):
                        raise ValueError("tool plan argument keys must be strings")
                    rendered[key] = resolve(nested, depth + 1)
                return rendered
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return [resolve(nested, depth + 1) for nested in value]
            return value

        resolved = resolve(step.arguments, 0)
        if not isinstance(resolved, dict):
            raise ValueError("tool plan arguments must resolve to a mapping")
        canonical_tool_arguments(resolved)
        return resolved

    @staticmethod
    def _plan_retry_call_id(call_id: str, attempt: int) -> str:
        suffix = f"-retry-{attempt}"
        return f"{call_id[: max(1, 256 - len(suffix))]}{suffix}"

    async def _execute_plan_step(
        self,
        step: ToolPlanStep,
        arguments: Mapping[str, Any],
        *,
        context: ToolCallContext,
        remaining_timeout_seconds: float,
        cancel_event: asyncio.Event | None,
        parallel_group: int = 0,
        plan_id: str = "",
    ) -> tuple[ToolOutcome, int]:
        """执行一个步骤，并在安全范围内应用超时、取消和重试。"""

        async def invoke(
            call_id: str,
            timeout_seconds: float,
            metadata: Mapping[str, Any],
        ) -> ToolOutcome:
            execution = asyncio.create_task(
                self.execute(
                    call_id=call_id,
                    identity=step.identity,
                    arguments=arguments,
                    context=context,
                    audit_metadata=metadata,
                )
            )
            cancellation = (
                asyncio.create_task(cancel_event.wait()) if cancel_event is not None else None
            )

            async def wait_for_result() -> ToolOutcome:
                if cancellation is None:
                    return await execution
                done, _pending = await asyncio.wait(
                    (execution, cancellation),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancellation in done and cancellation.result():
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)
                    raise asyncio.CancelledError
                return await execution

            try:
                return await asyncio.wait_for(wait_for_result(), timeout=timeout_seconds)
            finally:
                if cancellation is not None:
                    cancellation.cancel()
                    await asyncio.gather(cancellation, return_exceptions=True)
                if not execution.done():
                    execution.cancel()
                    await asyncio.gather(execution, return_exceptions=True)

        last_outcome: ToolOutcome | None = None
        for attempt in range(1, step.max_attempts + 1):
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError
            call_id = (
                step.call_id if attempt == 1 else self._plan_retry_call_id(step.call_id, attempt)
            )
            timeout_seconds = min(
                step.timeout_seconds,
                max(0.0, float(remaining_timeout_seconds)),
            )
            metadata = {
                "attempt": attempt,
                "max_attempts": step.max_attempts,
                "retry_count": attempt - 1,
                "step_id": step.step_id,
                "depends_on": step.depends_on,
                "parallel_group": parallel_group,
                "plan_id": plan_id,
                "source": "agent",
            }
            try:
                attempt_started_monotonic = time.monotonic()
                attempt_started_at = self._now()
            except RuntimeError:
                attempt_started_monotonic = time.monotonic()
                attempt_started_at = None
            if timeout_seconds <= 0:
                outcome = ToolOutcome(
                    "failed",
                    step.identity,
                    call_id,
                    {"error": "tool plan timeout exceeded"},
                )
                await self._record_execution(
                    outcome=outcome,
                    arguments=arguments,
                    context=context,
                    started_monotonic=attempt_started_monotonic,
                    started_at=attempt_started_at,
                    error_type="TimeoutError",
                    audit_metadata=metadata,
                )
                return outcome, attempt
            try:
                outcome = await invoke(call_id, timeout_seconds, metadata)
            except TimeoutError:
                outcome = ToolOutcome(
                    "failed",
                    step.identity,
                    call_id,
                    {"error": "tool step timed out"},
                )
                await self._record_execution(
                    outcome=outcome,
                    arguments=arguments,
                    context=context,
                    started_monotonic=attempt_started_monotonic,
                    started_at=attempt_started_at,
                    error_type="TimeoutError",
                    audit_metadata=metadata,
                )
            last_outcome = outcome
            if outcome.status != "failed" or attempt >= step.max_attempts:
                return outcome, attempt
        if last_outcome is None:
            raise RuntimeError("tool plan step did not produce an outcome")
        return last_outcome, step.max_attempts

    async def _execute_plan_normalized(
        self,
        steps: tuple[ToolPlanStep, ...],
        *,
        context: ToolCallContext,
        timeout_seconds: float,
        parallel_read_only: bool,
        stop_on_approval: bool,
        cancel_event: asyncio.Event | None,
        initial_outcomes: Mapping[str, ToolOutcome] | None = None,
        initial_audit: Mapping[str, ToolPlanAudit] | None = None,
        initial_parallel_group: int = 0,
        force_stop: bool = False,
        deadline: float | None = None,
    ) -> ToolPlanResult:
        """执行已验证计划，并在审批点生成可恢复检查点。"""

        if not steps:
            return ToolPlanResult((), (), False, ())
        loop = asyncio.get_running_loop()
        plan_deadline = loop.time() + timeout_seconds if deadline is None else float(deadline)
        by_id = {step.step_id: step for step in steps}
        declaration_index = {step.step_id: index for index, step in enumerate(steps)}
        plan_id = event_fingerprint(
            "|".join(f"{step.step_id}:{step.call_id}:{step.identity}" for step in steps)
            + context.turn_id
        )
        outcomes_by_id = dict(initial_outcomes or {})
        audit_by_id = dict(initial_audit or {})
        unresolved = set(by_id) - set(outcomes_by_id)
        success_statuses = {"completed", "duplicate"}
        stopped = bool(force_stop)
        stop_reason = "error_stop" if force_stop else ""
        pending_step_id = ""
        parallel_group = max(0, int(initial_parallel_group))

        def remaining_timeout() -> float:
            return max(0.0, plan_deadline - loop.time())

        def record(
            step: ToolPlanStep,
            outcome: ToolOutcome,
            *,
            reason: str = "",
            attempts: int = 0,
        ) -> None:
            outcomes_by_id[step.step_id] = outcome
            audit_by_id[step.step_id] = ToolPlanAudit(
                step.step_id,
                outcome.status,
                step.depends_on,
                parallel_group,
                reason,
                attempts,
            )

        async def record_synthetic(
            step: ToolPlanStep,
            outcome: ToolOutcome,
            *,
            phase: str,
        ) -> None:
            await self._record_execution(
                outcome=outcome,
                arguments=step.arguments,
                context=context,
                audit_metadata={
                    "source": "agent",
                    "mode": getattr(context, "metadata", {}).get("mode", "agent"),
                    "generation_id": getattr(context, "metadata", {}).get(
                        "generation_id",
                        0,
                    ),
                    "plan_id": plan_id,
                    "step_id": step.step_id,
                    "depends_on": step.depends_on,
                    "parallel_group": parallel_group,
                    "phase": phase,
                    "attempt": 1,
                    "max_attempts": step.max_attempts,
                },
            )

        def outcome_reason(outcome: ToolOutcome) -> str:
            error = str(outcome.content.get("error", "")).strip()
            if error == "tool step timed out":
                return "step_timeout"
            if error == "tool plan timeout exceeded":
                return "plan_timeout"
            if outcome.status == "failed":
                return "step_failed"
            if outcome.status == "denied":
                return "step_denied"
            return ""

        while unresolved and not stopped:
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError
            if remaining_timeout() <= 0:
                stopped = True
                stop_reason = "plan_timeout"
                break
            ready = [
                by_id[step_id]
                for step_id in unresolved
                if all(dependency in outcomes_by_id for dependency in by_id[step_id].depends_on)
            ]
            if not ready:
                raise ValueError("tool plan cannot resolve dependencies")
            parallel_group += 1
            executable: list[tuple[ToolPlanStep, dict[str, Any]]] = []
            for step in sorted(ready, key=lambda item: declaration_index[item.step_id]):
                failed_dependencies = [
                    dependency
                    for dependency in step.depends_on
                    if outcomes_by_id[dependency].status not in success_statuses
                ]
                if failed_dependencies:
                    record(
                        step,
                        ToolOutcome(
                            "denied",
                            step.identity,
                            step.call_id,
                            {"error": "依赖步骤未完成，未执行"},
                        ),
                        reason="dependency_failed",
                    )
                    await record_synthetic(
                        step,
                        outcomes_by_id[step.step_id],
                        phase="dependency_failed",
                    )
                    unresolved.remove(step.step_id)
                    continue
                try:
                    resolved_arguments = self._resolve_plan_arguments(
                        step,
                        outcomes_by_id,
                    )
                except (OverflowError, TypeError, ValueError):
                    record(
                        step,
                        ToolOutcome(
                            "failed",
                            step.identity,
                            step.call_id,
                            {"error": "tool plan result reference is invalid"},
                        ),
                        reason="result_reference_failed",
                    )
                    await record_synthetic(
                        step,
                        outcomes_by_id[step.step_id],
                        phase="result_reference_failed",
                    )
                    unresolved.remove(step.step_id)
                    if step.on_error == "stop":
                        stopped = True
                        stop_reason = "error_stop"
                        break
                    continue
                executable.append((step, resolved_arguments))
            if stopped or not executable:
                continue

            probe = getattr(self.permissions, "allows_without_approval", None)

            def is_safe_read(item: tuple[ToolPlanStep, Mapping[str, Any]]) -> bool:
                step = item[0]
                spec = self.registry.get(step.identity)
                if not (
                    bool(parallel_read_only)
                    and callable(probe)
                    and spec is not None
                    and spec.read_only
                    and spec.risk is RiskLevel.LOW
                ):
                    return False
                try:
                    return bool(probe(spec, context))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    return False

            if len(executable) > 1 and all(is_safe_read(item) for item in executable):
                step_results = await asyncio.gather(
                    *(
                        self._execute_plan_step(
                            step,
                            arguments,
                            context=context,
                            remaining_timeout_seconds=remaining_timeout(),
                            cancel_event=cancel_event,
                            parallel_group=parallel_group,
                            plan_id=plan_id,
                        )
                        for step, arguments in executable
                    )
                )
                parallel_approval_seen = False
                for (step, _arguments), (outcome, attempts) in zip(
                    executable,
                    step_results,
                    strict=True,
                ):
                    if (
                        stop_on_approval
                        and outcome.status == "approval_required"
                        and outcome.approval is not None
                        and parallel_approval_seen
                    ):
                        # 并行启动后若权限热更新导致多个调用同时进入审批，
                        # 只保留声明顺序中的第一个检查点；消费并拒绝
                        # 其余请求，避免隐藏的审批卡片和不可恢复快照泄漏。
                        self.deny_approval(outcome.approval.approval_id)
                        outcome = ToolOutcome(
                            "denied",
                            outcome.identity,
                            outcome.call_id,
                            {"error": "并行批次已有操作等待确认，当前步骤未执行"},
                        )
                    record(
                        step,
                        outcome,
                        reason=outcome_reason(outcome),
                        attempts=attempts,
                    )
                    unresolved.remove(step.step_id)
                    if outcome.status == "approval_required" and outcome.approval is not None:
                        # 权限探针与实际 evaluate 之间可能发生配置热更新；
                        # 即使并行批次已启动，也必须保留审批检查点，不能
                        # 让 approval_required 结果静默落入“已完成”计划。
                        if not pending_step_id:
                            pending_step_id = step.step_id
                        if stop_on_approval:
                            parallel_approval_seen = True
                            stopped = True
                            stop_reason = "approval_pending"
                    elif outcome.status not in success_statuses and step.on_error == "stop":
                        stopped = True
                        stop_reason = "error_stop"
                continue

            for step, arguments in executable:
                outcome, attempts = await self._execute_plan_step(
                    step,
                    arguments,
                    context=context,
                    remaining_timeout_seconds=remaining_timeout(),
                    cancel_event=cancel_event,
                    parallel_group=parallel_group,
                    plan_id=plan_id,
                )
                record(
                    step,
                    outcome,
                    reason=outcome_reason(outcome),
                    attempts=attempts,
                )
                unresolved.remove(step.step_id)
                if outcome.status == "approval_required":
                    pending_step_id = step.step_id
                    if stop_on_approval:
                        stopped = True
                        stop_reason = "approval_pending"
                        break
                elif outcome.status not in success_statuses and step.on_error == "stop":
                    stopped = True
                    stop_reason = "error_stop"
                    break

        checkpoint: ToolPlanCheckpoint | None = None
        pending_approval: ApprovalRequest | None = None
        if pending_step_id:
            pending_outcome = outcomes_by_id[pending_step_id]
            pending_approval = pending_outcome.approval
            if pending_approval is not None and stop_on_approval:
                checkpoint = ToolPlanCheckpoint(
                    steps,
                    self._snapshot_context(context),
                    tuple(
                        (step.step_id, outcomes_by_id[step.step_id])
                        for step in steps
                        if step.step_id in outcomes_by_id
                    ),
                    tuple(
                        (step.step_id, audit_by_id[step.step_id])
                        for step in steps
                        if step.step_id in audit_by_id
                    ),
                    pending_step_id,
                    parallel_group,
                    plan_deadline,
                    bool(parallel_read_only),
                    bool(stop_on_approval),
                )

        result_outcomes = dict(outcomes_by_id)
        result_audit = dict(audit_by_id)
        if unresolved:
            filler_reason = stop_reason or "not_started"
            message = (
                "前一项操作等待确认，后续步骤未执行"
                if filler_reason == "approval_pending"
                else "计划已停止，步骤未执行"
            )
            for step in steps:
                if step.step_id not in unresolved:
                    continue
                skipped = ToolOutcome(
                    "denied",
                    step.identity,
                    step.call_id,
                    {"error": message},
                )
                result_outcomes[step.step_id] = skipped
                result_audit[step.step_id] = ToolPlanAudit(
                    step.step_id,
                    skipped.status,
                    step.depends_on,
                    parallel_group,
                    filler_reason,
                    0,
                )
                await record_synthetic(
                    step,
                    skipped,
                    phase="skipped",
                )
        return ToolPlanResult(
            outcomes=tuple(result_outcomes[step.step_id] for step in steps),
            audit=tuple(result_audit[step.step_id] for step in steps),
            stopped=stopped,
            steps=steps,
            pending_approval=pending_approval,
            checkpoint=checkpoint,
        )

    async def execute_plan(
        self,
        steps: Sequence[ToolPlanStep],
        *,
        context: ToolCallContext,
        parallel_read_only: bool = True,
        stop_on_approval: bool = True,
        timeout_seconds: float = 30.0,
        cancel_event: asyncio.Event | None = None,
    ) -> ToolPlanResult:
        """验证并执行受限 DAG，在审批点返回不可变续接检查点。"""

        if not isinstance(context, ToolCallContext):
            raise ValueError("tool plan context is invalid")
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
            raise ValueError("tool plan timeout_seconds must be a number")
        bounded_timeout = float(timeout_seconds)
        if not math.isfinite(bounded_timeout) or not 0.05 <= bounded_timeout <= 120.0:
            raise ValueError("tool plan timeout_seconds is outside the allowed range")
        normalized = self._normalize_plan_steps(steps)
        return await self._execute_plan_normalized(
            normalized,
            context=context,
            timeout_seconds=bounded_timeout,
            parallel_read_only=bool(parallel_read_only),
            stop_on_approval=bool(stop_on_approval),
            cancel_event=cancel_event,
        )

    async def resume_plan(
        self,
        checkpoint: ToolPlanCheckpoint,
        approved_outcome: ToolOutcome,
        *,
        context: ToolCallContext,
        cancel_event: asyncio.Event | None = None,
    ) -> ToolPlanResult:
        """用刚批准的原始步骤结果继续同一计划，不重放已完成步骤。"""

        if not isinstance(checkpoint, ToolPlanCheckpoint):
            raise ValueError("tool plan checkpoint is invalid")
        if not isinstance(approved_outcome, ToolOutcome):
            raise ValueError("approved tool outcome is invalid")
        if not isinstance(context, ToolCallContext) or not self._contexts_match(
            checkpoint.context,
            context,
        ):
            raise ValueError("tool plan continuation context does not match")
        outcomes = dict(checkpoint.outcomes)
        audit = dict(checkpoint.audit)
        pending = outcomes.get(checkpoint.pending_step_id)
        if (
            pending is None
            or pending.status != "approval_required"
            or pending.approval is None
            or approved_outcome.call_id != pending.call_id
            or approved_outcome.identity != pending.identity
            or approved_outcome.status == "approval_required"
        ):
            raise ValueError("approved outcome does not match the paused plan step")
        outcomes[checkpoint.pending_step_id] = approved_outcome
        previous_audit = audit[checkpoint.pending_step_id]
        audit[checkpoint.pending_step_id] = ToolPlanAudit(
            checkpoint.pending_step_id,
            approved_outcome.status,
            previous_audit.depends_on,
            previous_audit.parallel_group,
            "approval_resumed",
            max(1, previous_audit.attempts),
        )
        pending_step = next(
            step for step in checkpoint.steps if step.step_id == checkpoint.pending_step_id
        )
        force_stop = (
            approved_outcome.status not in {"completed", "duplicate"}
            and pending_step.on_error == "stop"
        )
        return await self._execute_plan_normalized(
            checkpoint.steps,
            context=context,
            timeout_seconds=max(
                0.05,
                checkpoint.deadline - asyncio.get_running_loop().time(),
            ),
            parallel_read_only=checkpoint.parallel_read_only,
            stop_on_approval=checkpoint.stop_on_approval,
            cancel_event=cancel_event,
            initial_outcomes=outcomes,
            initial_audit=audit,
            initial_parallel_group=checkpoint.parallel_group,
            force_stop=force_stop,
            deadline=checkpoint.deadline,
        )

    def pending_approval(self, approval_id: str) -> ApprovalRequest | None:
        """读取仍在等待的审批快照；过期或不存在时返回 None。"""

        key = str(approval_id or "").strip()
        with self._approval_lock:
            self._prune_pending_locked()
            pending = self._pending_approvals.get(key)
            return pending.approval if pending is not None else None

    def pending_approvals(self) -> tuple[ApprovalRequest, ...]:
        """返回当前仍有效的审批请求，供受控界面展示。

        返回值只包含不可变的 ``ApprovalRequest``，不会暴露待执行工具的参数
        或可替换的执行上下文。实际批准仍必须经过
        ``ConversationService.resume_approval`` 的代际校验。
        """

        with self._approval_lock:
            self._prune_pending_locked()
            return tuple(
                pending.approval
                for pending in sorted(
                    self._pending_approvals.values(),
                    key=lambda item: (item.approval.expires_at, item.approval.approval_id),
                )
            )

    def pending_approvals_for_context(
        self, context: ToolCallContext
    ) -> tuple[ApprovalRequest, ...]:
        """只返回与给定回合完全匹配的审批，避免跨会话按钮串线。"""

        if not isinstance(context, ToolCallContext):
            raise TypeError("context must be a ToolCallContext")
        with self._approval_lock:
            self._prune_pending_locked()
            return tuple(
                pending.approval
                for pending in sorted(
                    self._pending_approvals.values(),
                    key=lambda item: (item.approval.expires_at, item.approval.approval_id),
                )
                if self._contexts_match(pending.context, context)
            )

    def approval_context(self, approval_id: str) -> ToolCallContext | None:
        """返回审批快照绑定的上下文，供无 Qt 的恢复入口做代际校验。"""

        key = str(approval_id or "").strip()
        with self._approval_lock:
            self._prune_pending_locked()
            pending = self._pending_approvals.get(key)
            return self._snapshot_context(pending.context) if pending is not None else None

    async def approve_and_execute(
        self,
        approval_id: str,
        *,
        grant_session: bool = False,
        context: ToolCallContext | None = None,
    ) -> ToolOutcome:
        """原子消费审批并执行原始参数；调用方不能替换工具或参数。"""

        key = str(approval_id or "").strip()
        pending: _PendingInvocation | None = None
        outcome: ToolOutcome | None = None
        permission_approved = False
        with self._approval_lock:
            self._prune_pending_locked()
            pending = self._pending_approvals.get(key)
            if pending is None:
                outcome = ToolOutcome(
                    "denied",
                    "",
                    "",
                    {"error": "approval request is missing or expired"},
                )
            elif context is not None and not self._contexts_match(context, pending.context):
                outcome = ToolOutcome(
                    "denied",
                    pending.spec.identity,
                    pending.call_id,
                    {"error": "approval context does not match the pending call"},
                    pending.approval,
                )
            else:
                try:
                    self.permissions.approve(key, grant_session=grant_session)
                except (KeyError, RuntimeError, TypeError, ValueError):
                    self._pending_approvals.pop(key, None)
                    outcome = ToolOutcome(
                        "denied",
                        pending.spec.identity,
                        pending.call_id,
                        {"error": "approval request is missing or expired"},
                    )
                else:
                    self._pending_approvals.pop(key, None)
                    permission_approved = True
        if outcome is not None:
            if pending is not None:
                self._log_tool_event(
                    "approval_rejected",
                    identity=pending.spec.identity,
                    signature_digest=pending.signature.digest,
                    status="context_mismatch"
                    if "context" in str(outcome.content.get("error", ""))
                    else outcome.status,
                    call_id=pending.call_id,
                    risk=pending.spec.risk,
                    reason_code="approval_context_mismatch"
                    if "context" in str(outcome.content.get("error", ""))
                    else "approval_missing",
                )
                await self._record_execution(
                    outcome=outcome,
                    arguments=pending.values,
                    context=pending.context,
                    signature_digest=pending.signature.digest,
                    started_monotonic=pending.started_monotonic,
                    started_at=pending.started_at,
                    audit_metadata=pending.audit_metadata,
                )
            else:
                self._log_tool_event(
                    "denied",
                    identity="",
                    signature_digest="",
                    status=outcome.status,
                )
                await self._record_execution(
                    outcome=outcome,
                    arguments={},
                    context=None,
                )
            return outcome
        if pending is None or not permission_approved:
            raise RuntimeError("approval state did not produce an outcome")
        try:
            resumed_started_monotonic = time.monotonic()
            resumed_started_at = self._now()
        except RuntimeError:
            resumed_started_monotonic = time.monotonic()
            resumed_started_at = None
        resumed_metadata = dict(pending.audit_metadata)
        resumed_metadata["phase"] = "approval_resume"
        self._log_tool_event(
            "approved",
            identity=pending.spec.identity,
            signature_digest=pending.signature.digest,
            status="approved",
            call_id=pending.call_id,
            risk=pending.spec.risk,
        )
        try:
            outcome = await self._invoke_authorized(
                spec=pending.spec,
                values=dict(pending.values),
                context=pending.context,
                call_id=pending.call_id,
            )
        except asyncio.CancelledError:
            cancelled = ToolOutcome(
                "failed",
                pending.spec.identity,
                pending.call_id,
                {"error": "tool call cancelled"},
            )
            self._log_tool_event(
                "failed",
                identity=pending.spec.identity,
                signature_digest=pending.signature.digest,
                status="cancelled",
                call_id=pending.call_id,
                risk=pending.spec.risk,
            )
            await self._record_execution(
                outcome=cancelled,
                arguments=pending.values,
                context=pending.context,
                signature_digest=pending.signature.digest,
                started_monotonic=resumed_started_monotonic,
                started_at=resumed_started_at,
                error_type="CancelledError",
                audit_metadata=resumed_metadata,
            )
            raise
        self._replace_completed_outcome(
            context=pending.context,
            signature=pending.signature,
            outcome=outcome,
        )
        await self._record_execution(
            outcome=outcome,
            arguments=pending.values,
            context=pending.context,
            signature_digest=pending.signature.digest,
            started_monotonic=resumed_started_monotonic,
            started_at=resumed_started_at,
            audit_metadata=resumed_metadata,
        )
        await self._publish_outcome(
            outcome.status,
            outcome,
            signature_digest=pending.signature.digest,
        )
        return outcome

    def deny_approval(self, approval_id: str) -> ApprovalRequest | None:
        """消费并拒绝一个待审批调用，拒绝后不可再次执行。"""

        key = str(approval_id or "").strip()
        with self._approval_lock:
            self._prune_pending_locked()
            pending = self._pending_approvals.pop(key, None)
            if pending is None:
                self._log_tool_event(
                    "denied", identity="", signature_digest="", status="approval_missing"
                )
                return None
            try:
                self.permissions.deny(key)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                outcome = ToolOutcome(
                    "denied",
                    pending.spec.identity,
                    pending.call_id,
                    {"error": "approval deny failed"},
                    pending.approval,
                )
                self._replace_completed_outcome(
                    context=pending.context,
                    signature=pending.signature,
                    outcome=outcome,
                )
                self._log_tool_event(
                    "failed",
                    identity=pending.spec.identity,
                    signature_digest=pending.signature.digest,
                    status="approval_deny_failed",
                    call_id=pending.call_id,
                    risk=pending.spec.risk,
                    error_type=type(exc).__name__,
                    reason_code="approval_deny_failed",
                )
                self._append_execution_record_sync(
                    outcome=outcome,
                    pending=pending,
                    error_type=type(exc).__name__,
                    phase="approval_deny_failed",
                )
                return None
            outcome = ToolOutcome(
                "denied",
                pending.spec.identity,
                pending.call_id,
                {"error": "approval denied by user"},
                pending.approval,
            )
            self._replace_completed_outcome(
                context=pending.context,
                signature=pending.signature,
                outcome=outcome,
            )
            self._log_tool_event(
                "denied",
                identity=pending.spec.identity,
                signature_digest=pending.signature.digest,
                status="approval_denied",
                call_id=pending.call_id,
                risk=pending.spec.risk,
            )
            self._append_execution_record_sync(
                outcome=outcome,
                pending=pending,
            )
            return pending.approval

    def _prune_pending_locked(self) -> None:
        # 到期语义由 PermissionService 的时钟统一决定。执行器可使用独立时钟
        # 记录审计时间，不能据此提前丢弃仍有效的审批快照。
        previous = dict(self._pending_approvals)
        try:
            active_ids = {request.approval_id for request in self.permissions.pending()}
        except (RuntimeError, TypeError, ValueError) as exc:
            self._pending_approvals.clear()
            for pending in previous.values():
                self._forget_invocation(context=pending.context, signature=pending.signature)
                self._log_tool_event(
                    "denied",
                    identity=pending.spec.identity,
                    signature_digest=pending.signature.digest,
                    status="permission_state_unavailable",
                    call_id=pending.call_id,
                    risk=pending.spec.risk,
                    error_type=type(exc).__name__,
                    reason_code="permission_state_unavailable",
                )
                # 权限状态不可用时，审批快照已经不可执行；必须把这个终态
                # 写入详细记录，否则控制台只能看到普通事件而无法关联原始调用。
                self._append_execution_record_sync(
                    outcome=ToolOutcome(
                        "denied",
                        pending.spec.identity,
                        pending.call_id,
                        {"error": "permission state unavailable"},
                        pending.approval,
                    ),
                    pending=pending,
                    error_type=type(exc).__name__,
                    phase="approval_state_unavailable",
                )
            return
        self._pending_approvals = {
            key: pending for key, pending in previous.items() if key in active_ids
        }
        for key, pending in previous.items():
            if key in active_ids:
                continue
            self._forget_invocation(context=pending.context, signature=pending.signature)
            self._log_tool_event(
                "denied",
                identity=pending.spec.identity,
                signature_digest=pending.signature.digest,
                status="expired",
                call_id=pending.call_id,
                risk=pending.spec.risk,
            )
            self._append_execution_record_sync(
                outcome=ToolOutcome(
                    "denied",
                    pending.spec.identity,
                    pending.call_id,
                    {"error": "approval request expired"},
                    pending.approval,
                ),
                pending=pending,
                phase="approval_expired",
            )

    def clear_pending(self) -> None:
        """运行时关闭时终止所有未消费审批，并写入对应终态。"""

        with self._approval_lock:
            pending_items = tuple(self._pending_approvals.items())
            self._pending_approvals.clear()
            for approval_id, pending in pending_items:
                self._forget_invocation(context=pending.context, signature=pending.signature)
                try:
                    self.permissions.deny(approval_id)
                except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                    error_type = type(exc).__name__
                else:
                    error_type = ""
                outcome = ToolOutcome(
                    "denied",
                    pending.spec.identity,
                    pending.call_id,
                    {"error": "tool call cancelled during shutdown"},
                    pending.approval,
                )
                self._log_tool_event(
                    "cancelled",
                    identity=pending.spec.identity,
                    signature_digest=pending.signature.digest,
                    status="shutdown",
                    call_id=pending.call_id,
                    risk=pending.spec.risk,
                    error_type=error_type,
                    reason_code="runtime_shutdown",
                )
                self._append_execution_record_sync(
                    outcome=outcome,
                    pending=pending,
                    error_type=error_type,
                    phase="runtime_shutdown",
                )

    def clear_turn(
        self,
        context: ToolCallContext,
        *,
        clear_pending: bool = False,
    ) -> None:
        turn_key = self._turn_key(context)
        # 审批快照由 PermissionService 统一计时；先在审批锁内清理
        # 已失效请求并捕获仍有效的签名，再更新回合去重索引。这样
        # ConversationService 的 finally 可以清理普通调用计数，同时保留
        # 等待审批的 single-flight 记录，避免同一调用在审批期间再次执行。
        with self._approval_lock:
            self._prune_pending_locked()
            preserved_signatures: set[ToolInvocationSignature] = set()
            if not clear_pending:
                preserved_signatures = {
                    pending.signature
                    for pending in self._pending_approvals.values()
                    if self._turn_key(pending.context) == turn_key
                }
            else:
                removed = tuple(
                    (key, pending)
                    for key, pending in self._pending_approvals.items()
                    if self._contexts_match(pending.context, context)
                )
                for approval_id, pending in removed:
                    self._pending_approvals.pop(approval_id, None)
                    try:
                        self.permissions.deny(approval_id)
                    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                        error_type = type(exc).__name__
                    else:
                        error_type = ""
                    self._forget_invocation(
                        context=pending.context,
                        signature=pending.signature,
                    )
                    self._log_tool_event(
                        "cancelled",
                        identity=pending.spec.identity,
                        signature_digest=pending.signature.digest,
                        status="turn_cleared",
                        call_id=pending.call_id,
                        risk=pending.spec.risk,
                        error_type=error_type,
                        reason_code="turn_cleared",
                    )
            with self._state_lock:
                state = self._turn_invocations.get(turn_key)
                if state is None:
                    return
                if not preserved_signatures:
                    self._turn_invocations.pop(turn_key, None)
                    return
                state.by_signature = {
                    signature: record
                    for signature, record in state.by_signature.items()
                    if signature in preserved_signatures
                }
                state.call_count = len(state.by_signature)
                if not state.by_signature:
                    self._turn_invocations.pop(turn_key, None)

    @staticmethod
    def _snapshot_context(context: ToolCallContext) -> ToolCallContext:
        """复制审批所需上下文，避免调用方修改原对象影响恢复代际。"""

        return ToolCallContext(
            context.profile_id,
            context.session_id,
            context.turn_id,
            source=context.source,
            metadata=ToolExecutionService._snapshot_mapping(context.metadata),
        )

    @staticmethod
    def _snapshot_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
        """递归复制 JSON 风格参数，防止审批期间的外部可变引用被复用。"""

        try:
            copied = deepcopy(dict(values))
            # 工具参数最终会进入 JSON-RPC/模型消息；审批快照不能保存
            # 非 JSON 对象或 NaN/Infinity，否则批准后可能执行不同语义。
            canonical_tool_arguments(copied)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("tool invocation snapshot is invalid") from exc
        if not isinstance(copied, dict):
            raise ValueError("tool invocation snapshot is invalid")
        return copied

    @staticmethod
    def _contexts_match(left: ToolCallContext, right: ToolCallContext) -> bool:
        for field_name in ("profile_id", "session_id", "turn_id", "source"):
            if getattr(left, field_name) != getattr(right, field_name):
                return False
        left_metadata = left.metadata
        right_metadata = right.metadata
        left_generation = ToolExecutionService._generation_id(left_metadata)
        right_generation = ToolExecutionService._generation_id(right_metadata)
        if left_generation is None or right_generation is None:
            return False
        return (
            str(left_metadata.get("mode", "direct")).strip().lower()
            == str(right_metadata.get("mode", "direct")).strip().lower()
            and left_generation == right_generation
        )

    @staticmethod
    def _generation_id(metadata: Mapping[str, Any]) -> int | None:
        """只接受有限、非负且不截断的审批代际号。"""

        value = metadata.get("generation_id", 0)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer() or value < 0:
                return None
            return int(value)
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or not normalized.isdecimal() or len(normalized) > 20:
                return None
            try:
                return int(normalized)
            except ValueError:
                return None
        return None

    @staticmethod
    def _turn_key(context: ToolCallContext) -> tuple[str, str, str, str, str]:
        mode = str(context.metadata.get("mode", "direct")).strip().lower()
        if mode not in {"direct", "agent"}:
            mode = "direct"
        return (mode, context.profile_id, context.session_id, context.turn_id, context.source)
