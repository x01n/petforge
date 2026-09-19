"""默认拒绝的工具风险评估与带 TTL 的审批。"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from threading import RLock

from .types import ApprovalRequest, RiskLevel, ToolCallContext, ToolSpec


class PermissionDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    APPROVAL_REQUIRED = "approval_required"


_MIN_APPROVAL_TTL_SECONDS = 5.0
_MAX_APPROVAL_TTL_SECONDS = 3600.0
MAX_PENDING_APPROVALS = 1024
_XML_APPROVE_NEVER_PREFIXES = ("scheduler:", "proactive:")

XML_APPROVE_WINDOW_REASON = "xml approve window active"
_XML_WINDOW_TTL_MIN_SECONDS = 5.0
_XML_WINDOW_TTL_MAX_SECONDS = 300.0
_DEFAULT_XML_WINDOW_TTL_SECONDS = 15.0


@dataclass(frozen=True)
class _XmlApproveWindow:
    profile_id: str
    session_id: str
    turn_id: str
    expires_at: float
    approved_identities: frozenset[str]


@dataclass(frozen=True)
class PermissionResult:
    decision: PermissionDecision
    reason: str
    approval: ApprovalRequest | None = None
    bypassed: bool = False


class PermissionService:
    """公开集与执行权限分离：模型看得见不代表一定能执行。"""

    def __init__(
        self,
        *,
        allow: Iterable[str] = (),
        deny: Iterable[str] = (),
        bypass_approval: bool = False,
        auto_allow_low_risk: bool = True,
        approval_ttl_seconds: float = 90.0,
        xml_approve_enabled: bool = False,
        xml_approve_window_seconds: float = 15.0,
        clock=time.time,
    ) -> None:
        self._allow = {str(item).strip() for item in allow if str(item).strip()}
        self._deny = {str(item).strip() for item in deny if str(item).strip()}
        self._bypass = bool(bypass_approval)
        self._auto_low = bool(auto_allow_low_risk)
        self._ttl = self._bounded_ttl(approval_ttl_seconds)
        # XML approve 门控默认关闭；只有配置显式开启并且窗口 TTL 有限时，
        # accept_xml_approve 才可能有副作用。未启用时 approve=true 与现状
        # 完全一致（零回归）。
        self._xml_approve_enabled = bool(xml_approve_enabled)
        self._xml_window_ttl = self._bounded_xml_window_ttl(xml_approve_window_seconds)
        self._clock = clock
        self._lock = RLock()
        self._session_grants: dict[tuple[str, str, str], float] = {}
        self._pending: dict[str, ApprovalRequest] = {}
        self._xml_windows: dict[tuple[str, str], _XmlApproveWindow] = {}

    def reconfigure(
        self,
        *,
        allow: Iterable[str] = (),
        deny: Iterable[str] = (),
        bypass_approval: bool = False,
        auto_allow_low_risk: bool = True,
        approval_ttl_seconds: float = 90.0,
        xml_approve_enabled: bool = False,
        xml_approve_window_seconds: float = 15.0,
    ) -> None:
        """原子替换运行期权限策略，不中断已执行的工具调用。

        已经创建的审批请求属于已经发起的调用，继续保留其原始快照；
        后续新调用立即使用新策略。策略切换会撤销旧的会话授权，避免
        新的 allow/deny 规则被旧授权绕过。
        """

        next_allow = {str(item).strip() for item in allow if str(item).strip()}
        next_deny = {str(item).strip() for item in deny if str(item).strip()}
        next_bypass = bool(bypass_approval)
        next_auto_low = bool(auto_allow_low_risk)
        next_ttl = self._bounded_ttl(approval_ttl_seconds)
        next_xml_enabled = bool(xml_approve_enabled)
        next_xml_window_ttl = self._bounded_xml_window_ttl(xml_approve_window_seconds)
        with self._lock:
            changed = (
                next_allow != self._allow
                or next_deny != self._deny
                or next_bypass != self._bypass
                or next_auto_low != self._auto_low
                or next_ttl != self._ttl
                or next_xml_enabled != self._xml_approve_enabled
                or next_xml_window_ttl != self._xml_window_ttl
            )
            self._allow = next_allow
            self._deny = next_deny
            self._bypass = next_bypass
            self._auto_low = next_auto_low
            self._ttl = next_ttl
            self._xml_approve_enabled = next_xml_enabled
            self._xml_window_ttl = next_xml_window_ttl
            # 热重载无论参数是否变化都必须清除 XML 运行期放行窗口：
            # 旧策略下授权的窗口不允许跨配置存续，避免 deny 收紧被窗口绕过。
            self._xml_windows.clear()
            if changed:
                self._session_grants.clear()

    @staticmethod
    def _bounded_ttl(value: object) -> float:
        """返回受限 TTL；非有限值不能进入审批状态。"""

        if isinstance(value, bool):
            raise ValueError("approval_ttl_seconds must be a finite number")
        try:
            ttl = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("approval_ttl_seconds must be a finite number") from exc
        if not isfinite(ttl):
            raise ValueError("approval_ttl_seconds must be a finite number")
        return max(_MIN_APPROVAL_TTL_SECONDS, min(ttl, _MAX_APPROVAL_TTL_SECONDS))

    @staticmethod
    def _bounded_xml_window_ttl(value: object) -> float:
        """返回受限 XML 放行窗口 TTL；布尔值与非有限值都不接受。"""

        if isinstance(value, bool):
            raise ValueError("xml_approve_window_seconds must be a finite number")
        try:
            ttl = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("xml_approve_window_seconds must be a finite number") from exc
        if not isfinite(ttl):
            raise ValueError("xml_approve_window_seconds must be a finite number")
        return max(_XML_WINDOW_TTL_MIN_SECONDS, min(ttl, _XML_WINDOW_TTL_MAX_SECONDS))

    @property
    def xml_approve_enabled(self) -> bool:
        """XML approve 门控是否启用；只读，供编排层/审计层观察。"""

        return self._xml_approve_enabled

    def _now(self, now: object | None = None) -> float:
        value = self._clock() if now is None else now
        if isinstance(value, bool):
            raise RuntimeError("permission clock must return a finite timestamp")
        try:
            current = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("permission clock must return a finite timestamp") from exc
        if not isfinite(current):
            raise RuntimeError("permission clock must return a finite timestamp")
        return current

    def _purge(self, now: object | None = None) -> None:
        current = self._now(now)
        self._session_grants = {
            key: expiry for key, expiry in self._session_grants.items() if expiry > current
        }
        self._pending = {
            key: request for key, request in self._pending.items() if request.expires_at > current
        }
        self._xml_windows = {
            key: window for key, window in self._xml_windows.items() if window.expires_at > current
        }

    def evaluate(
        self,
        spec: ToolSpec,
        context: ToolCallContext,
        *,
        call_id: str = "",
        safe_summary: str = "",
    ) -> PermissionResult:
        with self._lock:
            now = self._now()
            self._purge(now)
            if spec.identity in self._deny:
                return PermissionResult(PermissionDecision.DENY, "tool is denied by policy")
            if self._allow and spec.identity not in self._allow:
                return PermissionResult(PermissionDecision.DENY, "tool is not in the allowlist")
            if spec.risk == RiskLevel.LOW and self._auto_low:
                return PermissionResult(PermissionDecision.ALLOW, "low-risk tool")
            # XML approve 有界放行：仅当门控启用、窗口未过期、身份在本回
            # 合被拦截过、风险不高于 MEDIUM 时生效。deny 与 allowlist 判定
            # 在前，deny 名单永远优先；HIGH/LOW 分支不与窗口交互，LOW 走
            # 上方 auto_low，HIGH 落到下方审批流程。
            prefixes_denied = spec.identity.lower().startswith(_XML_APPROVE_NEVER_PREFIXES)
            window = self._xml_windows.get((context.profile_id, context.session_id))
            if (
                window is not None
                and window.expires_at > now
                and context.turn_id == window.turn_id
                and spec.identity in window.approved_identities
                and spec.risk == RiskLevel.MEDIUM
                and not prefixes_denied
            ):
                return PermissionResult(PermissionDecision.ALLOW, XML_APPROVE_WINDOW_REASON)
            if self._bypass and (not self._allow or spec.identity in self._allow):
                return PermissionResult(
                    PermissionDecision.ALLOW,
                    "approval bypass is enabled",
                    bypassed=True,
                )
            expiry = self._session_grants.get(
                (context.profile_id, context.session_id, spec.identity), 0.0
            )
            if expiry > now:
                return PermissionResult(PermissionDecision.ALLOW, "session grant is active")
            approval_id = f"approval-{secrets.token_urlsafe(12)}"
            expires_at = now + self._ttl
            if not isfinite(expires_at) or expires_at <= now:
                raise RuntimeError("permission clock cannot create a future approval expiry")
            if len(self._pending) >= MAX_PENDING_APPROVALS:
                self._pending.pop(next(iter(self._pending)))
            request = ApprovalRequest(
                approval_id=approval_id,
                call_id=str(call_id or "").strip() or f"pending-{secrets.token_urlsafe(8)}",
                identity=spec.identity,
                session_id=context.session_id,
                safe_summary=str(safe_summary or "tool requires confirmation").strip(),
                expires_at=expires_at,
                profile_id=context.profile_id,
                display_name=spec.user_label,
            )
            self._pending[approval_id] = request
            return PermissionResult(
                PermissionDecision.APPROVAL_REQUIRED, "confirmation is required", request
            )

    def evaluate_without_approval(
        self,
        spec: ToolSpec,
        context: ToolCallContext,
        *,
        call_id: str = "",
        safe_summary: str = "",
    ) -> PermissionDecision:
        """按与 :meth:`evaluate` 相同的决策序裁决策略，但不创建审批。

        与 :meth:`allows_without_approval` 的区别：本探针把“没有放行依据”
        与“被 deny 拒绝”区分开，分别返回 APPROVAL_REQUIRED 与 DENY，
        且不产生任何审批副作用。窗口放行分支在这里同样生效。
        """

        del call_id, safe_summary
        if not isinstance(spec, ToolSpec) or not isinstance(context, ToolCallContext):
            return PermissionDecision.APPROVAL_REQUIRED
        with self._lock:
            now = self._now()
            self._purge(now)
            if spec.identity in self._deny:
                return PermissionDecision.DENY
            if self._allow and spec.identity not in self._allow:
                return PermissionDecision.DENY
            if spec.risk == RiskLevel.LOW and self._auto_low:
                return PermissionDecision.ALLOW
            # XML 放行窗口与 evaluate 守住同一前缀防线：
            # scheduler:/proactive: 动作不进入窗口放行。
            prefixes_denied = spec.identity.lower().startswith(_XML_APPROVE_NEVER_PREFIXES)
            window = self._xml_windows.get((context.profile_id, context.session_id))
            if (
                window is not None
                and window.expires_at > now
                and context.turn_id == window.turn_id
                and spec.identity in window.approved_identities
                and spec.risk == RiskLevel.MEDIUM
                and not prefixes_denied
            ):
                return PermissionDecision.ALLOW
            if self._bypass and (not self._allow or spec.identity in self._allow):
                return PermissionDecision.ALLOW
            expiry = self._session_grants.get(
                (context.profile_id, context.session_id, spec.identity), 0.0
            )
            if expiry > now:
                return PermissionDecision.ALLOW
            return PermissionDecision.APPROVAL_REQUIRED

    def accept_xml_approve(
        self,
        context: ToolCallContext,
        *,
        identities: Iterable[object] = (),
        ttl_seconds: float | None = None,
    ) -> bool:
        """按声明身份提升一次有界放行窗口；窗口外或越权声明返回 False。

        仅在门控启用、身份是本会话本回合确实被拦截/拒绝过的假性操作、
        且归属盘与风险等级通过时才提升。调用方必须先经过
        :meth:`appliance_active` 校验避免打扰；会话与回合在这里最终复核。

        Args:
            context: 声明身份所属的工具调用上下文。
            identities: 被放行的工具身份集合；高风险身份会被过滤。
            ttl_seconds: 覆盖构造期窗口 TTL；None 时沿用构造期值。

        Returns:
            True 表示新窗口已生效、False 表示未启用或没有可放行的身份。
        """

        if not isinstance(context, ToolCallContext):
            return False
        if not self._xml_approve_enabled:
            return False
        self._discard_windows_for(context)
        eligible_identities: set[str] = set()
        with self._lock:
            now = self._now()
            self._purge(now)
            for item in identities:
                identity = str(item or "").strip()
                if not identity or identity in self._deny:
                    continue
                if self._allow and identity not in self._allow:
                    continue
                if identity.lower().startswith(_XML_APPROVE_NEVER_PREFIXES):
                    continue
                eligible_identities.add(identity)
            if not eligible_identities:
                return False
            window_ttl = (
                self._xml_window_ttl
                if ttl_seconds is None
                else self._bounded_xml_window_ttl(ttl_seconds)
            )
            expires_at = now + window_ttl
            if not isfinite(expires_at) or expires_at <= now:
                raise RuntimeError("permission clock cannot create a future approval window expiry")
            self._xml_windows[(context.profile_id, context.session_id)] = _XmlApproveWindow(
                profile_id=context.profile_id,
                session_id=context.session_id,
                turn_id=context.turn_id,
                expires_at=expires_at,
                approved_identities=frozenset(eligible_identities),
            )
            return True

    def appliance_active(self, context: ToolCallContext, identity: object = "") -> bool:
        """返回当前会话是否存在覆盖该身份的未过期 XML 放行窗口。"""

        if not isinstance(context, ToolCallContext):
            return False
        identity = str(identity or "").strip()
        with self._lock:
            now = self._now()
            self._purge(now)
            window = self._xml_windows.get((context.profile_id, context.session_id))
            if window is None or window.expires_at <= now or window.turn_id != context.turn_id:
                return False
            if not identity:
                return True
            return identity in window.approved_identities and window.expires_at > now

    def cancel_xml_approve_window(self, context: ToolCallContext) -> None:
        """撤销当前会话的 XML 放行窗口；无窗口时静默返回。"""

        self._discard_windows_for(context)

    def _discard_windows_for(self, context: ToolCallContext | None) -> None:
        if not isinstance(context, ToolCallContext):
            return
        with self._lock:
            self._xml_windows.pop((context.profile_id, context.session_id), None)

    def allows_without_approval(self, spec: ToolSpec, context: ToolCallContext) -> bool:
        """判断调用是否已经具备无审批执行资格，不创建审批记录。

        对话编排器用这个只读探针决定能否并行执行多个观察工具；探针必须
        与 :meth:`evaluate` 使用同一策略和锁，不能通过读取内部字段猜测权限。
        """

        if not isinstance(spec, ToolSpec) or not isinstance(context, ToolCallContext):
            return False
        with self._lock:
            now = self._now()
            self._purge(now)
            if spec.identity in self._deny:
                return False
            if self._allow and spec.identity not in self._allow:
                return False
            if spec.risk == RiskLevel.LOW and self._auto_low:
                return True
            if self._bypass and (not self._allow or spec.identity in self._allow):
                return True
            expiry = self._session_grants.get(
                (context.profile_id, context.session_id, spec.identity), 0.0
            )
            return expiry > now

    def approve(self, approval_id: str, *, grant_session: bool = False) -> ApprovalRequest:
        with self._lock:
            now = self._now()
            self._purge(now)
            session_expiry = now + self._ttl
            if grant_session and (not isfinite(session_expiry) or session_expiry <= now):
                raise RuntimeError("permission clock cannot create a future session grant expiry")
            request = self._pending.pop(str(approval_id or "").strip(), None)
            if request is None:
                raise KeyError("approval request is missing or expired")
            if grant_session:
                self._session_grants[(request.profile_id, request.session_id, request.identity)] = (
                    session_expiry
                )
            return request

    def deny(self, approval_id: str) -> ApprovalRequest:
        with self._lock:
            self._purge()
            request = self._pending.pop(str(approval_id or "").strip(), None)
            if request is None:
                raise KeyError("approval request is missing or expired")
            return request

    def pending(self) -> tuple[ApprovalRequest, ...]:
        with self._lock:
            self._purge()
            return tuple(sorted(self._pending.values(), key=lambda item: item.expires_at))
