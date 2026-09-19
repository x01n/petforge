"""直连运行时使用的显式重试策略。"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import ProviderAdapterError, classify_exception

_DEFAULT_CATEGORIES = frozenset({"network", "timeout", "rate_limit", "server"})
_DEFAULT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """总尝试次数、退避和可重试错误分类均显式配置。"""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.0
    retryable_categories: frozenset[str] = field(default_factory=lambda: _DEFAULT_CATEGORIES)
    retryable_status_codes: frozenset[int] = field(default_factory=lambda: _DEFAULT_STATUS_CODES)
    respect_retry_after: bool = True
    # 渠道级优雅降级缩放：热度内突发错误时，在 1.0~1.8 之间放大退避，
    # 避免多个并发回合在同一个故障窗口内同步重试同一渠道。
    backoff_scale_warmup: float = 1.0
    backoff_scale_max: float = 1.8
    fr2_allow_after_first_event: bool = False
    """首事件后发生网络错误时是否允许在同渠道内重试；默认保持首事件前
    重试语义，流中出现网络错误由调用方决定继续消费还是换渠道。"""

    def __post_init__(self) -> None:
        raw_attempts = self.max_attempts
        if isinstance(raw_attempts, bool) or not isinstance(raw_attempts, int):
            raise ValueError("retry max_attempts must be an integer")
        attempts = raw_attempts
        initial = float(self.initial_delay_seconds)
        maximum = float(self.max_delay_seconds)
        multiplier = float(self.backoff_multiplier)
        jitter = float(self.jitter_ratio)
        if not 1 <= attempts <= 10:
            raise ValueError("retry max_attempts must be between 1 and 10")
        if not all(math.isfinite(value) for value in (initial, maximum, multiplier, jitter)):
            raise ValueError("retry numeric values must be finite")
        if initial < 0 or maximum < initial:
            raise ValueError("retry delays are invalid")
        if multiplier < 1:
            raise ValueError("retry backoff_multiplier must be at least 1")
        if not 0 <= jitter <= 1:
            raise ValueError("retry jitter_ratio must be between 0 and 1")
        categories = frozenset(
            str(item).strip().lower() for item in self.retryable_categories if str(item).strip()
        )
        try:
            raw_statuses = tuple(self.retryable_status_codes)
        except TypeError as exc:
            raise ValueError("retryable_status_codes must be an iterable of integers") from exc
        if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_statuses):
            raise ValueError("retryable_status_codes must contain only integers")
        statuses = frozenset(raw_statuses)
        if not isinstance(self.respect_retry_after, bool):
            raise ValueError("respect_retry_after must be a boolean")
        if not isinstance(self.fr2_allow_after_first_event, bool):
            raise ValueError("fr2_allow_after_first_event must be a boolean")
        warmup = float(self.backoff_scale_warmup)
        ceiling = float(self.backoff_scale_max)
        if not all(math.isfinite(value) for value in (warmup, ceiling)):
            raise ValueError("retry backoff scale values must be finite")
        if not 1.0 <= warmup <= ceiling <= 5.0:
            raise ValueError("retry backoff scale values are invalid")
        object.__setattr__(self, "max_attempts", attempts)
        object.__setattr__(self, "initial_delay_seconds", initial)
        object.__setattr__(self, "max_delay_seconds", maximum)
        object.__setattr__(self, "backoff_multiplier", multiplier)
        object.__setattr__(self, "jitter_ratio", jitter)
        object.__setattr__(self, "retryable_categories", categories)
        object.__setattr__(self, "retryable_status_codes", statuses)
        object.__setattr__(self, "respect_retry_after", self.respect_retry_after)
        object.__setattr__(self, "fr2_allow_after_first_event", self.fr2_allow_after_first_event)
        object.__setattr__(self, "backoff_scale_warmup", warmup)
        object.__setattr__(self, "backoff_scale_max", ceiling)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> RetryPolicy:
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ValueError("retry policy must be a mapping")
        aliases = {
            "attempts": "max_attempts",
            "retries": "max_attempts",
            "initial_delay": "initial_delay_seconds",
            "max_delay": "max_delay_seconds",
            "multiplier": "backoff_multiplier",
            "jitter": "jitter_ratio",
            "categories": "retryable_categories",
            "statuses": "retryable_status_codes",
        }
        allowed = {
            "max_attempts",
            "initial_delay_seconds",
            "max_delay_seconds",
            "backoff_multiplier",
            "jitter_ratio",
            "retryable_categories",
            "retryable_status_codes",
            "respect_retry_after",
            "backoff_scale_warmup",
            "backoff_scale_max",
            "fr2_allow_after_first_event",
        }
        values: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = aliases.get(str(raw_key).strip(), str(raw_key).strip())
            if key not in allowed:
                raise ValueError(f"unsupported retry policy field: {key}")
            if isinstance(raw_value, str) and key in {
                "retryable_categories",
                "retryable_status_codes",
            }:
                raw_value = [item.strip() for item in raw_value.split(",") if item.strip()]
            values[key] = raw_value
        return cls(**values)

    def should_retry(
        self, error: BaseException, *, attempt: int, before_first_event: bool = True
    ) -> bool:
        if not before_first_event and not self.fr2_allow_after_first_event:
            return False
        if int(attempt) < 1 or int(attempt) >= self.max_attempts:
            return False
        classified = (
            error
            if isinstance(error, ProviderAdapterError)
            else classify_exception(
                error,
                before_first_event=before_first_event,
            )
        )
        if not classified.retryable or classified.category not in self.retryable_categories:
            return False
        if (
            classified.status_code is not None
            and classified.status_code not in self.retryable_status_codes
        ):
            return False
        return True

    def delay_for(
        self,
        attempt: int,
        *,
        retry_after_seconds: float | None = None,
        warmup_ratio: float = 0.0,
    ) -> float:
        """计算第 ``attempt`` 次重试前的退避时长。

        ``warmup_ratio`` 在 0~1 之间时按热度在 ``backoff_scale_warmup`` 与
        ``backoff_scale_max`` 之间线性放大基础退避，用来在短时间突发错误时
        给渠道散热；为 0 时行为与旧版本完全一致。
        """

        number = max(1, int(attempt))
        try:
            scaled = self.initial_delay_seconds * self.backoff_multiplier ** (number - 1)
        except OverflowError:
            # 极端参数组合（如 multiplier 接近 1e300）下指数先溢出，
            # 直接按上限钳制而不是向上抛。
            scaled = self.max_delay_seconds
        delay = min(self.max_delay_seconds, scaled)
        ratio = min(1.0, max(0.0, float(warmup_ratio or 0.0)))
        if ratio > 0:
            scale = (
                self.backoff_scale_warmup
                + (self.backoff_scale_max - self.backoff_scale_warmup) * ratio
            )
            delay = min(self.max_delay_seconds, delay * scale)
        if self.respect_retry_after and retry_after_seconds is not None:
            delay = max(delay, min(self.max_delay_seconds, max(0.0, float(retry_after_seconds))))
        if self.jitter_ratio:
            delay *= 1 + random.uniform(-self.jitter_ratio, self.jitter_ratio)
        return max(0.0, min(self.max_delay_seconds, delay))

    def delays(self) -> Iterable[float]:
        for attempt in range(1, self.max_attempts):
            yield self.delay_for(attempt)


__all__ = ["RetryPolicy"]
