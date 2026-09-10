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
        clock=time.time,
    ) -> None:
        self._allow = {str(item).strip() for item in allow if str(item).strip()}
        self._deny = {str(item).strip() for item in deny if str(item).strip()}
        self._bypass = bool(bypass_approval)
        self._auto_low = bool(auto_allow_low_risk)
        self._ttl = self._bounded_ttl(approval_ttl_seconds)
        self._clock = clock
        self._lock = RLock()
        self._session_grants: dict[tuple[str, str, str], float] = {}
        self._pending: dict[str, ApprovalRequest] = {}

    def reconfigure(
        self,
        *,
        allow: Iterable[str] = (),
        deny: Iterable[str] = (),
        bypass_approval: bool = False,
        auto_allow_low_risk: bool = True,
        approval_ttl_seconds: float = 90.0,
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
        with self._lock:
            changed = (
                next_allow != self._allow
                or next_deny != self._deny
                or next_bypass != self._bypass
                or next_auto_low != self._auto_low
                or next_ttl != self._ttl
            )
            self._allow = next_allow
            self._deny = next_deny
            self._bypass = next_bypass
            self._auto_low = next_auto_low
            self._ttl = next_ttl
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
