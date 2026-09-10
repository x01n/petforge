"""模型可见的受限多步工具计划协议。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.contracts.chat import ToolCall, ToolDefinition

from .executor import ToolOutcome, ToolPlanResult
from .types import MAX_TOOL_PLAN_STEPS, ToolPlanStep

AGENT_PLAN_TOOL_IDENTITY = "agent:execute_plan"
AGENT_PLAN_PROTOCOL_VERSION = 1
DEFAULT_TOOL_PLAN_TIMEOUT_SECONDS = 30.0
MAX_TOOL_PLAN_TIMEOUT_SECONDS = 120.0
MAX_TOOL_PLAN_ARGUMENT_BYTES = 65_536
_MAX_REFERENCE_PATH_ITEMS = 16
_MAX_ARGUMENT_DEPTH = 32


@dataclass(frozen=True)
class AgentToolPlan:
    """一次通过模型工具调用提交的已验证计划。"""

    steps: tuple[ToolPlanStep, ...]
    timeout_seconds: float = DEFAULT_TOOL_PLAN_TIMEOUT_SECONDS


def agent_plan_tool_definition() -> ToolDefinition:
    """返回所有供应商共享的 v1 计划工具定义。"""

    reference_description = (
        "参数值可使用严格结果引用对象："
        '{"$step_result":{"step_id":"步骤ID","path":["字段",0]}}；'
        "被引用步骤必须同时列入 depends_on。"
    )
    step_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "step_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "identity": {"type": "string", "minLength": 3, "maxLength": 256},
            "arguments": {
                "type": "object",
                "description": reference_description,
            },
            "depends_on": {
                "type": "array",
                "maxItems": MAX_TOOL_PLAN_STEPS,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
            },
            "timeout_seconds": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 60.0,
            },
            "max_attempts": {"type": "integer", "minimum": 1, "maximum": 3},
            "on_error": {
                "type": "string",
                "enum": ["stop", "skip_dependents"],
            },
        },
        "required": ["step_id", "identity"],
        "additionalProperties": False,
    }
    return ToolDefinition(
        AGENT_PLAN_TOOL_IDENTITY,
        ("按严格依赖图执行有限工具步骤；同层低风险只读步骤可并行，高风险步骤仍逐项请求用户确认。"),
        {
            "type": "object",
            "properties": {
                "version": {"type": "integer", "enum": [AGENT_PLAN_PROTOCOL_VERSION]},
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 0.05,
                    "maximum": MAX_TOOL_PLAN_TIMEOUT_SECONDS,
                },
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_TOOL_PLAN_STEPS,
                    "items": step_schema,
                },
            },
            "required": ["version", "steps"],
            "additionalProperties": False,
        },
    )


def parse_agent_plan_call(
    call: ToolCall,
    *,
    allowed_identities: frozenset[str],
) -> AgentToolPlan:
    """严格解析计划调用；任何未声明字段都会使整份计划失效。"""

    if call.identity != AGENT_PLAN_TOOL_IDENTITY:
        raise ValueError("tool call is not an agent plan")
    arguments = call.arguments
    if any(not isinstance(key, str) for key in arguments):
        raise ValueError("agent plan argument keys must be strings")
    if set(arguments) - {"version", "timeout_seconds", "steps"}:
        raise ValueError("agent plan contains unknown fields")
    version = arguments.get("version")
    if isinstance(version, bool) or version != AGENT_PLAN_PROTOCOL_VERSION:
        raise ValueError("agent plan version is unsupported")
    timeout_seconds = _finite_number(
        arguments.get("timeout_seconds", DEFAULT_TOOL_PLAN_TIMEOUT_SECONDS),
        name="agent plan timeout_seconds",
        minimum=0.05,
        maximum=MAX_TOOL_PLAN_TIMEOUT_SECONDS,
    )
    raw_steps = arguments.get("steps")
    if isinstance(raw_steps, (str, bytes, bytearray)) or not isinstance(raw_steps, Sequence):
        raise ValueError("agent plan steps must be a sequence")
    if not 1 <= len(raw_steps) <= MAX_TOOL_PLAN_STEPS:
        raise ValueError("agent plan step count is outside the allowed range")
    try:
        serialized = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("agent plan arguments must be JSON compatible") from exc
    if len(serialized.encode("utf-8")) > MAX_TOOL_PLAN_ARGUMENT_BYTES:
        raise ValueError("agent plan arguments exceed the size limit")

    steps: list[ToolPlanStep] = []
    seen_step_ids: set[str] = set()
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError("agent plan step must be a mapping")
        allowed_fields = {
            "step_id",
            "identity",
            "arguments",
            "depends_on",
            "timeout_seconds",
            "max_attempts",
            "on_error",
        }
        if set(raw_step) - allowed_fields:
            raise ValueError("agent plan step contains unknown fields")
        if "step_id" not in raw_step or "identity" not in raw_step:
            raise ValueError("agent plan step is missing required fields")
        raw_step_id = raw_step.get("step_id")
        raw_identity = raw_step.get("identity")
        if not isinstance(raw_step_id, str) or not isinstance(raw_identity, str):
            raise ValueError("agent plan identifiers must be strings")
        identity = raw_identity.strip()
        if identity == AGENT_PLAN_TOOL_IDENTITY or identity not in allowed_identities:
            raise ValueError("agent plan step uses an unavailable tool")
        raw_arguments = raw_step.get("arguments", {})
        if not isinstance(raw_arguments, Mapping):
            raise ValueError("agent plan step arguments must be a mapping")
        raw_dependencies = raw_step.get("depends_on", ())
        if isinstance(raw_dependencies, (str, bytes, bytearray)) or not isinstance(
            raw_dependencies, Sequence
        ):
            raise ValueError("agent plan depends_on must be a sequence")
        if any(not isinstance(item, str) for item in raw_dependencies):
            raise ValueError("agent plan dependencies must be strings")
        max_attempts = raw_step.get("max_attempts", 1)
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise ValueError("agent plan max_attempts must be an integer")
        step = ToolPlanStep(
            step_id=raw_step_id,
            call_id=_step_call_id(call.call_id, index),
            identity=identity,
            arguments=dict(raw_arguments),
            depends_on=tuple(raw_dependencies),
            timeout_seconds=_finite_number(
                raw_step.get("timeout_seconds", 15.0),
                name="agent plan step timeout_seconds",
                minimum=0.05,
                maximum=60.0,
            ),
            max_attempts=max_attempts,
            on_error=raw_step.get("on_error", "skip_dependents"),
        )
        if step.step_id in seen_step_ids:
            raise ValueError("agent plan contains duplicate step_id")
        seen_step_ids.add(step.step_id)
        referenced_steps = _validate_result_references(step.arguments)
        if not referenced_steps.issubset(set(step.depends_on)):
            raise ValueError("agent plan result reference is not declared in depends_on")
        steps.append(step)
    return AgentToolPlan(tuple(steps), timeout_seconds)


def agent_plan_outcome(call_id: str, result: ToolPlanResult) -> ToolOutcome:
    """把内部逐步结果折叠为一个符合供应商工具消息约束的结果。"""

    approval = result.pending_approval
    statuses = {item.status for item in result.outcomes}
    if approval is not None:
        status = "approval_required"
    elif "failed" in statuses:
        status = "failed"
    elif "denied" in statuses:
        status = "denied"
    else:
        status = "completed"
    audit_by_step = {item.step_id: item for item in result.audit}
    steps: list[dict[str, object]] = []
    for step, outcome in zip(result.steps, result.outcomes, strict=True):
        audit = audit_by_step[step.step_id]
        steps.append(
            {
                "step_id": step.step_id,
                "identity": step.identity,
                "status": outcome.status,
                "result": _safe_result_value(outcome.content),
                "reason": audit.reason,
                "attempts": audit.attempts,
            }
        )
    return ToolOutcome(
        status,
        AGENT_PLAN_TOOL_IDENTITY,
        str(call_id or "").strip(),
        {
            "version": AGENT_PLAN_PROTOCOL_VERSION,
            "status": status,
            "stopped": result.stopped,
            "steps": steps,
        },
        approval,
    )


def _step_call_id(outer_call_id: str, index: int) -> str:
    digest = hashlib.sha256(str(outer_call_id).encode("utf-8")).hexdigest()[:20]
    return f"agent-plan-{digest}-{index + 1}"


def _finite_number(value: object, *, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} is outside the allowed range")
    return number


def _validate_result_references(value: object) -> frozenset[str]:
    references: set[str] = set()
    visited = 0

    def walk(item: object, depth: int) -> None:
        nonlocal visited
        visited += 1
        if visited > 4096 or depth > _MAX_ARGUMENT_DEPTH:
            raise ValueError("agent plan arguments are too deeply nested")
        if isinstance(item, Mapping):
            if "$step_result" in item:
                if set(item) != {"$step_result"}:
                    raise ValueError("agent plan result reference must be a standalone object")
                reference = item["$step_result"]
                if not isinstance(reference, Mapping) or set(reference) != {"step_id", "path"}:
                    raise ValueError("agent plan result reference is invalid")
                raw_step_id = reference.get("step_id")
                if not isinstance(raw_step_id, str):
                    raise ValueError("agent plan result reference step_id is invalid")
                step_id = raw_step_id.strip()
                path = reference.get("path")
                if not step_id or len(step_id) > 256:
                    raise ValueError("agent plan result reference step_id is invalid")
                if isinstance(path, (str, bytes, bytearray)) or not isinstance(path, Sequence):
                    raise ValueError("agent plan result reference path must be a sequence")
                if len(path) > _MAX_REFERENCE_PATH_ITEMS:
                    raise ValueError("agent plan result reference path is too long")
                for part in path:
                    if isinstance(part, bool) or not isinstance(part, (str, int)):
                        raise ValueError("agent plan result reference path item is invalid")
                    if isinstance(part, int) and part < 0:
                        raise ValueError("agent plan result reference index is invalid")
                    if isinstance(part, str) and (not part or len(part) > 256):
                        raise ValueError("agent plan result reference key is invalid")
                references.add(step_id)
                return
            for key, nested in item.items():
                if not isinstance(key, str):
                    raise ValueError("agent plan argument keys must be strings")
                walk(nested, depth + 1)
            return
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            for nested in item:
                walk(nested, depth + 1)

    walk(value, 0)
    return frozenset(references)


def _safe_result_value(value: object, *, depth: int = 0) -> object:
    if depth > 24:
        return {"truncated": True}
    if value is None or isinstance(value, (bool, int, str)):
        return value if not isinstance(value, str) or len(value) <= 16_384 else value[:16_384]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _safe_result_value(nested, depth=depth + 1)
            for key, nested in list(value.items())[:256]
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        rendered = [_safe_result_value(nested, depth=depth + 1) for nested in list(value)[:256]]
        if len(value) > 256:
            rendered.append({"truncated_items": len(value) - 256})
        return rendered
    return {"type": type(value).__name__}


__all__ = [
    "AGENT_PLAN_PROTOCOL_VERSION",
    "AGENT_PLAN_TOOL_IDENTITY",
    "AgentToolPlan",
    "agent_plan_outcome",
    "agent_plan_tool_definition",
    "parse_agent_plan_call",
]
