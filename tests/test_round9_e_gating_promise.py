"""第 9 轮方向 2 的 E 项：XML approve 放行窗口的高危身份门禁。

调度器/主动行为身份（scheduler:/proactive:）即使本回合模型输出
<approve> 配套窗口也不放行；普通 MEDIUM 身份保持原有窗口语义。
本地假时钟与独立注册表，不发网络、不触 UI。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from services.tools.permissions import (
    XML_APPROVE_WINDOW_REASON,
    PermissionDecision,
    PermissionService,
)
from services.tools.types import RiskLevel, ToolCallContext, ToolSpec


class _FakeClock:
    """可推进的单调时钟，供 PermissionService 使用。"""

    def __init__(self, start: float = 1_000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _handler(*_args: object, **_kwargs: object) -> Mapping[str, Any]:
    return {"ok": True}


def _spec(identity: str, risk: RiskLevel) -> ToolSpec:
    return ToolSpec(
        identity,
        f"测试工具 {identity}",
        {"type": "object"},
        _handler,
        risk=risk,
    )


def _context(turn: str = "turn-9", session: str = "session-9") -> ToolCallContext:
    return ToolCallContext("profile-9", session, turn)


def _evaluate(
    permissions: PermissionService,
    spec: ToolSpec,
    context: ToolCallContext,
) -> object:
    return permissions.evaluate(
        spec,
        context,
        call_id="call-9",
        safe_summary="需要确认",
    )


def test_xml_approve_rejects_scheduler_action_still_requires_approval() -> None:
    """scheduler:upsert 即使拿到 approve 窗口也不放行。"""

    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    context = _context()
    spec = _spec("scheduler:upsert", RiskLevel.MEDIUM)

    assert _evaluate(permissions, spec, context).decision is PermissionDecision.APPROVAL_REQUIRED
    # 前缀被过滤：声明本身不建立窗口。
    assert permissions.accept_xml_approve(context, identities=("scheduler:upsert",)) is False
    result = _evaluate(permissions, spec, context)
    assert result.decision is not PermissionDecision.ALLOW
    assert result.reason != XML_APPROVE_WINDOW_REASON


def test_xml_approve_rejects_proactive_action() -> None:
    """proactive: 前缀动作同样不进入窗口放行。"""

    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    context = _context()
    spec = _spec("proactive:unknown_high_freq", RiskLevel.MEDIUM)

    assert _evaluate(permissions, spec, context).decision is PermissionDecision.APPROVAL_REQUIRED
    assert (
        permissions.accept_xml_approve(
            context,
            identities=("proactive:unknown_high_freq",),
        )
        is False
    )
    result = _evaluate(permissions, spec, context)
    assert result.decision is not PermissionDecision.ALLOW
    assert result.reason != XML_APPROVE_WINDOW_REASON


def test_xml_approve_literal_play_motion_action_still_allowed() -> None:
    """普通桌面动作保持上游许可语义，纯激活放行行为不破坏。"""

    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    context = _context()
    spec = _spec("pet:play_motion", RiskLevel.MEDIUM)

    assert _evaluate(permissions, spec, context).decision is PermissionDecision.APPROVAL_REQUIRED
    assert permissions.accept_xml_approve(context, identities=("pet:play_motion",)) is True
    result = _evaluate(permissions, spec, context)
    assert result.decision is PermissionDecision.ALLOW
    assert result.reason == XML_APPROVE_WINDOW_REASON


def test_xml_approve_window_scoped_to_declared_turn() -> None:
    """跨回合窗口与 E 项防线组合：低风险身份走正常放行语义。"""

    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    spec = _spec("pet:list_models", RiskLevel.LOW)

    context_one = _context(turn="turn-one")
    # LOW 与窗口无关，走 auto_low 分支。
    assert _evaluate(permissions, spec, context_one).decision is PermissionDecision.ALLOW
    assert (
        permissions.accept_xml_approve(
            context_one,
            identities=("pet:list_models",),
        )
        is True
    )
