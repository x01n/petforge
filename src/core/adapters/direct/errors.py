from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ErrorCategory = Literal[
    "authentication",
    "authorization",
    "network",
    "timeout",
    "rate_limit",
    "server",
    "protocol",
    "configuration",
    "cancelled",
    "unknown",
]


@dataclass(frozen=True)
class ErrorContext:
    """用于日志和路由决策的最小错误信息。"""

    category: ErrorCategory = "unknown"
    status_code: int | None = None
    retryable: bool = False
    before_first_event: bool = True


class ProviderAdapterError(RuntimeError):
    """提供方请求失败，消息保持为可安全展示的文本。"""

    def __init__(
        self,
        safe_message: str,
        *,
        category: ErrorCategory = "unknown",
        status_code: int | None = None,
        retryable: bool = False,
        before_first_event: bool = True,
        cause: BaseException | None = None,
    ) -> None:
        message = str(safe_message or "provider request failed").strip()
        if not message:
            message = "provider request failed"
        super().__init__(message)
        self.safe_message = message
        self.category: ErrorCategory = category
        self.status_code = int(status_code) if status_code is not None else None
        self.retryable = bool(retryable)
        self.before_first_event = bool(before_first_event)
        self.cause = cause

    @property
    def context(self) -> ErrorContext:
        return ErrorContext(
            category=self.category,
            status_code=self.status_code,
            retryable=self.retryable,
            before_first_event=self.before_first_event,
        )


class AdapterConfigurationError(ProviderAdapterError):
    """渠道配置不满足安全契约。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, category="configuration", retryable=False)


class AdapterProtocolError(ProviderAdapterError):
    """响应无法解析为已知流协议。"""

    def __init__(self, message: str, *, before_first_event: bool = True) -> None:
        super().__init__(
            message,
            category="protocol",
            retryable=False,
            before_first_event=before_first_event,
        )


class AdapterCancelled(ProviderAdapterError):
    """调用由上层取消。"""

    def __init__(self, message: str = "provider request cancelled") -> None:
        super().__init__(message, category="cancelled", retryable=False)


def classify_exception(
    error: BaseException,
    *,
    before_first_event: bool = True,
) -> ProviderAdapterError:
    """将底层异常转换为不含敏感信息的适配器错误。"""

    if isinstance(error, ProviderAdapterError):
        if error.before_first_event == before_first_event:
            return error
        return ProviderAdapterError(
            error.safe_message,
            category=error.category,
            status_code=error.status_code,
            retryable=error.retryable,
            before_first_event=before_first_event,
            cause=error,
        )

    name = type(error).__name__.lower()
    if "timeout" in name:
        return ProviderAdapterError(
            "provider request timed out",
            category="timeout",
            retryable=True,
            before_first_event=before_first_event,
            cause=error,
        )
    if any(
        token in name for token in ("connect", "network", "transport", "remoteprotocol", "oserror")
    ):
        return ProviderAdapterError(
            "provider network request failed",
            category="network",
            retryable=True,
            before_first_event=before_first_event,
            cause=error,
        )
    return ProviderAdapterError(
        "provider request failed",
        category="unknown",
        retryable=False,
        before_first_event=before_first_event,
        cause=error,
    )


__all__ = [
    "AdapterCancelled",
    "AdapterConfigurationError",
    "AdapterProtocolError",
    "ErrorCategory",
    "ErrorContext",
    "ProviderAdapterError",
    "classify_exception",
]
