"""第 6 轮 B 模块：XML approve 指令与工具权限门禁的有界放行接入测试。

覆盖安全语义：默认关闭零副作用、启用后窗口内放行、HIGH 永不生效、
deny 名单永远优先、窗口过期回到审批、热重载清除窗口、会话/回合
不匹配保持审批、提示注入文本、严格布尔解析、执行器桥接入口以及
对话服务的窗口提升接线。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from services.conversation.service import ConversationService
from services.directives import DirectivesError, directives_output_hint, extract_xml_directives
from services.model_routing.channels import ChannelConfig
from services.model_routing.router import ModelRouter
from services.tools.executor import ToolExecutionService
from services.tools.permissions import (
    XML_APPROVE_WINDOW_REASON,
    PermissionDecision,
    PermissionService,
)
from services.tools.registry import ToolRegistry
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


def _context(turn: str = "turn-1", session: str = "session-1") -> ToolCallContext:
    return ToolCallContext("profile-1", session, turn)


def _evaluate(permissions: PermissionService, spec: ToolSpec, context: ToolCallContext):
    return permissions.evaluate(spec, context, call_id="call-1", safe_summary="需要确认")


def test_xml_approve_disabled_by_default_keeps_approval_flow() -> None:
    """默认关闭：accept 声明不产生窗口，决策仍是审批，审计 reason 不变。"""
    clock = _FakeClock()
    permissions = PermissionService(clock=clock)
    context = _context()
    medium = _spec("pet:move", RiskLevel.MEDIUM)

    assert permissions.xml_approve_enabled is False
    assert permissions.accept_xml_approve(context, identities=("pet:move",)) is False
    # 未启用时 evaluate 仍回到审批路径，且不携带窗口 reason。
    assert _evaluate(permissions, medium, context).decision is PermissionDecision.APPROVAL_REQUIRED
    assert not permissions.appliance_active(context, "pet:move")


def test_xml_approve_window_allows_blocked_medium_identity_with_audit_reason() -> None:
    """启用后：声明把被拦截的 MEDIUM 身份提为短窗口放行，reason 唯一可测。"""
    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    context = _context()
    medium = _spec("pet:move", RiskLevel.MEDIUM)

    assert _evaluate(permissions, medium, context).decision is PermissionDecision.APPROVAL_REQUIRED
    assert permissions.accept_xml_approve(context, identities=("pet:move",)) is True
    assert permissions.appliance_active(context, "pet:move") is True

    result = _evaluate(permissions, medium, context)
    assert result.decision is PermissionDecision.ALLOW
    assert result.reason == XML_APPROVE_WINDOW_REASON
    assert result.bypassed is False


def test_xml_approve_never_releases_high_risk() -> None:
    """HIGH 风险即使被声明也永不进入窗口，评估仍要求审批。"""
    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    context = _context()
    high = _spec("desktop:automation_batch", RiskLevel.HIGH)

    assert permissions.accept_xml_approve(context, identities=("desktop:automation_batch",)) is True
    result = _evaluate(permissions, high, context)
    assert result.decision is PermissionDecision.APPROVAL_REQUIRED
    assert result.reason != XML_APPROVE_WINDOW_REASON


def test_xml_approve_window_deny_list_stays_priority() -> None:
    """deny 名单命中时即使窗口活跃也保持拒绝，deny 永远优先。"""
    clock = _FakeClock()
    permissions = PermissionService(
        clock=clock,
        xml_approve_enabled=True,
        deny=("pet:move",),
    )
    context = _context()
    medium = _spec("pet:move", RiskLevel.MEDIUM)

    # 声明被 deny 过滤，窗口甚至不会真正建立。
    assert permissions.accept_xml_approve(context, identities=("pet:move",)) is False
    result = _evaluate(permissions, medium, context)
    assert result.decision is PermissionDecision.DENY
    assert result.reason != XML_APPROVE_WINDOW_REASON


def test_xml_approve_window_expires_back_to_approval() -> None:
    """窗口过期后同会话同身份重新回到审批路径。"""
    clock = _FakeClock()
    permissions = PermissionService(
        clock=clock,
        xml_approve_enabled=True,
        xml_approve_window_seconds=15.0,
    )
    context = _context()
    medium = _spec("pet:move", RiskLevel.MEDIUM)

    assert permissions.accept_xml_approve(context, identities=("pet:move",)) is True
    assert _evaluate(permissions, medium, context).decision is PermissionDecision.ALLOW

    clock.advance(30.0)
    assert not permissions.appliance_active(context, "pet:move")
    assert _evaluate(permissions, medium, context).decision is PermissionDecision.APPROVAL_REQUIRED


def test_xml_approve_window_is_scoped_to_session_and_turn() -> None:
    """会话或回合不匹配时窗口失效，审批照旧。"""
    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    medium = _spec("pet:move", RiskLevel.MEDIUM)

    assert permissions.accept_xml_approve(_context(turn="turn-1"), identities=("pet:move",)) is True
    other_turn = _context(turn="turn-2")
    assert not permissions.appliance_active(other_turn, "pet:move")
    assert (
        _evaluate(permissions, medium, other_turn).decision is PermissionDecision.APPROVAL_REQUIRED
    )

    other_session = ToolCallContext("profile-1", "session-other", "turn-1")
    assert not permissions.appliance_active(other_session, "pet:move")
    assert (
        _evaluate(permissions, medium, other_session).decision
        is PermissionDecision.APPROVAL_REQUIRED
    )


def test_reconfigure_clears_runtime_xml_windows() -> None:
    """热重载后运行期放行状态清除，即使参数完全相同也要回到审批。"""
    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    context = _context()
    medium = _spec("pet:move", RiskLevel.MEDIUM)

    assert permissions.accept_xml_approve(context, identities=("pet:move",)) is True
    permissions.reconfigure(xml_approve_enabled=True)
    assert not permissions.appliance_active(context, "pet:move")
    assert _evaluate(permissions, medium, context).decision is PermissionDecision.APPROVAL_REQUIRED


def test_allowlist_and_low_risk_do_not_interact_with_window() -> None:
    """allowlist 收紧对新身份立即生效；auto_low 的 LOW 始终直通。"""
    clock = _FakeClock()
    permissions = PermissionService(
        clock=clock,
        xml_approve_enabled=True,
        allow=("pet:move",),
    )
    context = _context()
    medium = _spec("pet:move", RiskLevel.MEDIUM)
    outside = _spec("pet:set_click_through", RiskLevel.MEDIUM)
    low = _spec("pet:list_models", RiskLevel.LOW)

    assert permissions.accept_xml_approve(context, identities=("pet:move",)) is True
    assert _evaluate(permissions, medium, context).decision is PermissionDecision.ALLOW
    # 不在 allowlist 的身份永远走拒绝分支，窗口不会覆盖 allowlist。
    assert _evaluate(permissions, outside, context).decision is PermissionDecision.DENY
    # LOW 依然受 allowlist 约束：allowlist 判定先于 auto_low（窗口无关）。
    assert _evaluate(permissions, low, context).decision is PermissionDecision.DENY


def test_reconfigure_unknown_keys_default_to_false_and_bounded_ttl() -> None:
    """reconfigure 不传新键时门控回落到关闭，超界 TTL 被钳制后可建立窗口。"""
    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    context = _context()
    medium = _spec("pet:move", RiskLevel.MEDIUM)

    # 超界 TTL 被钳制到窗口上限而不是报错，窗口仍可建立。
    assert (
        permissions.accept_xml_approve(context, identities=("pet:move",), ttl_seconds=999_999.0)
        is True
    )
    assert permissions.appliance_active(context, "pet:move") is True

    permissions.reconfigure()
    assert permissions.xml_approve_enabled is False
    assert permissions.accept_xml_approve(context, identities=("pet:move",)) is False
    assert _evaluate(permissions, medium, context).decision is PermissionDecision.APPROVAL_REQUIRED


def test_directives_output_hint_mentions_approve_gating() -> None:
    """提示注入文本包含 approve 描述，且不改变既有行。"""
    hint = directives_output_hint("zh-CN")
    assert "&lt;approve&gt;true&lt;/approve&gt;" in hint
    assert "仅当你在同一回合内发起的工具调用被拦截时可请求放行；高风险操作不适用。" in hint
    assert "<approve>" not in hint or "&lt;approve&gt;" in hint


def test_xml_approve_strict_boolean_parsing_unchanged() -> None:
    """approve 字段仍是严格布尔：非法值报 DirectivesError，不落默认。"""
    assert extract_xml_directives("<meapet><approve>true</approve></meapet>").approve is True
    assert extract_xml_directives("<meapet><approve>false</approve></meapet>").approve is False
    try:
        extract_xml_directives("<meapet><approve>maybe</approve></meapet>")
    except DirectivesError:
        pass
    else:
        raise AssertionError("非法布尔值必须抛出 DirectivesError")


def test_executor_promote_xml_approve_bridges_to_permissions() -> None:
    """执行器桥接入口只转发，启用状态下透传返回并建立窗口。"""
    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    registry = ToolRegistry()
    registry.register(_spec("pet:move", RiskLevel.MEDIUM))
    executor = ToolExecutionService(registry, permissions)
    context = _context()

    assert executor.promote_xml_approve(context, identities=("pet:move",)) is True
    assert permissions.appliance_active(context, "pet:move") is True

    # 未启用时转发返回 False，无副作用。
    disabled = PermissionService(clock=_FakeClock())
    disabled_executor = ToolExecutionService(registry, disabled)
    assert disabled_executor.promote_xml_approve(context, identities=("pet:move",)) is False


def test_conversation_promote_wiring_skips_high_risk_and_uses_executor() -> None:
    """对话服务接线经由执行器桥接；HIGH 拦截记录阻止整体窗口建立。"""
    clock = _FakeClock()
    permissions = PermissionService(clock=clock, xml_approve_enabled=True)
    registry = ToolRegistry()
    registry.register(_spec("pet:move", RiskLevel.MEDIUM))
    registry.register(_spec("desktop:automation_batch", RiskLevel.HIGH))
    executor = ToolExecutionService(registry, permissions)
    service = ConversationService(_router(), tools=executor)
    context = service.begin_context()

    from services.tools.executor import ToolOutcome

    # 只有 MEDIUM 被拦截时，窗口可以建立。
    service._promote_xml_approve_window(
        context,
        ("pet:move",),
        (ToolOutcome("denied", "pet:move", "call-1", {"error": "tool is denied by policy"}),),
    )
    assert (
        permissions.appliance_active(
            ToolCallContext(context.profile_id, context.session_id, context.turn_id),
            "pet:move",
        )
        is True
    )

    # HIGH 拦截记录也在本回合时整体降级，不建立窗口。
    permissions.cancel_xml_approve_window(
        ToolCallContext(context.profile_id, context.session_id, context.turn_id)
    )
    service._promote_xml_approve_window(
        context,
        ("desktop:automation_batch",),
        (
            ToolOutcome(
                "denied",
                "desktop:automation_batch",
                "call-2",
                {"error": "tool is denied by policy"},
            ),
        ),
    )
    assert not permissions.appliance_active(
        ToolCallContext(context.profile_id, context.session_id, context.turn_id)
    )


def _router() -> ModelRouter:
    channel = ChannelConfig("fake", base_url="https://fake.invalid/v1", model="demo")

    class _Adapter:
        provider = "fake"
        protocol = "openai_chat"
        capabilities = frozenset({"streaming"})

        async def stream(self, request, *, context=None, cancel_event=None):
            del request, cancel_event
            from core.events.types import TextDelta, TurnFinished

            yield TextDelta(context, "好的")
            yield TurnFinished(context)

    return ModelRouter([channel], adapter_factory=lambda _channel: _Adapter())
