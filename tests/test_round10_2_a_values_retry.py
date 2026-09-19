"""第 10 轮方向 2：values 工具函数与直连重试策略的覆盖补漏。

覆盖对象：
- ``src/core/util/values.py``：主路径与非法输入/类型归一边界；
- ``src/core/adapters/direct/retry.py``：退避有界性、尝试次数硬上限、
  抖动范围、首事件后重试门控与 from_mapping 别名/拒绝。

全部为纯内存单元测试，无网络、无模型、无 GUI、无真实数据库文件。
"""

from __future__ import annotations

import random

import pytest

from core.adapters.direct.errors import ProviderAdapterError
from core.adapters.direct.retry import RetryPolicy
from core.util.values import as_mapping, as_sequence, finite_float, safe_int


def test_as_mapping_returns_mappings_unchanged_and_empty_mapping_otherwise() -> None:
    source = {"a": 1}
    assert as_mapping(source) is source
    assert as_mapping([("a", 1)]) == {}
    assert as_mapping("text") == {}
    assert as_mapping(None) == {}
    assert as_mapping(3) == {}


def test_as_sequence_rejects_text_bytes_mappings_and_scalars() -> None:
    assert as_sequence("abc") == ()
    assert as_sequence(b"abc") == ()
    assert as_sequence(bytearray(b"abc")) == ()
    assert as_sequence({"a": 1}) == ()
    assert as_sequence(3) == ()
    values = [1, 2]
    assert as_sequence(values) is values
    assert as_sequence((1, 2)) == (1, 2)


def test_finite_float_parses_and_uses_default_for_non_finite_or_invalid() -> None:
    assert finite_float("12.5") == 12.5
    assert finite_float(3) == 3.0
    assert finite_float(float("nan"), default=-1.0) == -1.0
    assert finite_float(float("inf"), default=7.0) == 7.0
    assert finite_float("-inf") == 0.0
    assert finite_float("bad", default=4.0) == 4.0
    assert finite_float(object(), default=9.0) == 9.0
    assert finite_float(None) == 0.0


def test_safe_int_rejects_bool_and_uses_default_for_invalid() -> None:
    assert safe_int("42") == 42
    assert safe_int(2.9) == 2
    assert safe_int(True) == 0
    assert safe_int(False, default=5) == 5
    assert safe_int("not-an-int", default=-1) == -1
    assert safe_int(float("nan")) == 0
    assert safe_int(object()) == 0
    assert safe_int("0x10") == 0


def test_retry_policy_rejects_out_of_range_control_values() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=11)
    with pytest.raises(ValueError, match="delays"):
        RetryPolicy(initial_delay_seconds=-1.0)
    with pytest.raises(ValueError, match="delays"):
        RetryPolicy(initial_delay_seconds=3.0, max_delay_seconds=2.0)
    with pytest.raises(ValueError, match="multiplier"):
        RetryPolicy(backoff_multiplier=0.5)
    with pytest.raises(ValueError, match="jitter"):
        RetryPolicy(jitter_ratio=1.5)
    with pytest.raises(ValueError, match="backoff scale"):
        RetryPolicy(backoff_scale_warmup=0.5)
    with pytest.raises(ValueError, match="backoff scale"):
        RetryPolicy(backoff_scale_warmup=2.0, backoff_scale_max=1.9)


def test_retry_policy_from_mapping_applies_aliases_and_rejects_invalid_shapes() -> None:
    policy = RetryPolicy.from_mapping(
        {"attempts": 5, "initial_delay": 0.25, "categories": "network, timeout"}
    )
    assert policy.max_attempts == 5
    assert policy.initial_delay_seconds == 0.25
    assert policy.retryable_categories == frozenset({"network", "timeout"})
    assert RetryPolicy.from_mapping(None).max_attempts == 3
    with pytest.raises(ValueError, match="unsupported retry policy field"):
        RetryPolicy.from_mapping({"unknown_option": 1})
    with pytest.raises(ValueError, match="must be a mapping"):
        RetryPolicy.from_mapping([])


def test_retry_policy_delays_are_bounded_capped_and_monotonic() -> None:
    policy = RetryPolicy(
        max_attempts=5,
        initial_delay_seconds=1.0,
        backoff_multiplier=4.0,
        max_delay_seconds=3.0,
        jitter_ratio=0.0,
    )
    delays = tuple(policy.delays())
    assert len(delays) == policy.max_attempts - 1
    assert delays == tuple(sorted(delays))
    assert all(0.0 <= delay <= policy.max_delay_seconds for delay in delays)
    assert delays == (1.0, 3.0, 3.0, 3.0)


def test_retry_policy_delay_for_applies_retry_after_cap_and_warmup_scale() -> None:
    policy = RetryPolicy(
        initial_delay_seconds=1.0,
        max_delay_seconds=100.0,
        backoff_multiplier=1.0,
        jitter_ratio=0.0,
        backoff_scale_warmup=1.0,
        backoff_scale_max=2.0,
    )
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(1, retry_after_seconds=4.0) == 4.0
    assert policy.delay_for(1, retry_after_seconds=0.1) == 1.0
    assert policy.delay_for(1, retry_after_seconds=999.0) == 100.0
    assert policy.delay_for(1, warmup_ratio=0.5) == 1.5
    assert policy.delay_for(1, warmup_ratio=1.0) == 2.0
    assert policy.delay_for(1, warmup_ratio=2.0) == 2.0
    assert policy.delay_for(1, warmup_ratio=-1.0) == 1.0


def test_retry_policy_jitter_stays_within_ratio_bounds(monkeypatch) -> None:
    policy = RetryPolicy(
        initial_delay_seconds=1.0,
        max_delay_seconds=100.0,
        backoff_multiplier=1.0,
        jitter_ratio=0.5,
    )
    monkeypatch.setattr(random, "uniform", lambda _low, high: high)
    assert policy.delay_for(1) == 1.5
    monkeypatch.setattr(random, "uniform", lambda low, _high: low)
    assert policy.delay_for(1) == 0.5


def test_retry_policy_max_attempts_is_a_hard_upper_bound() -> None:
    policy = RetryPolicy(max_attempts=3)
    assert policy.should_retry(TimeoutError(), attempt=1)
    assert policy.should_retry(TimeoutError(), attempt=2)
    assert not policy.should_retry(TimeoutError(), attempt=3)
    assert not policy.should_retry(TimeoutError(), attempt=0)
    assert not policy.should_retry(TimeoutError(), attempt=-4)


def test_retry_policy_after_first_event_gating() -> None:
    default = RetryPolicy()
    assert default.should_retry(TimeoutError(), attempt=1, before_first_event=True)
    assert not default.should_retry(TimeoutError(), attempt=1, before_first_event=False)
    permissive = RetryPolicy(fr2_allow_after_first_event=True)
    assert permissive.should_retry(TimeoutError(), attempt=1, before_first_event=False)


def test_retry_policy_should_retry_requires_retryable_category_and_status() -> None:
    policy = RetryPolicy(
        retryable_categories=frozenset({"server"}),
        retryable_status_codes=frozenset({503}),
    )
    accepted = ProviderAdapterError(
        "gateway down", category="server", status_code=503, retryable=True
    )
    assert policy.should_retry(accepted, attempt=1)
    wrong_category = ProviderAdapterError(
        "limited", category="rate_limit", status_code=503, retryable=True
    )
    assert not policy.should_retry(wrong_category, attempt=1)
    wrong_status = ProviderAdapterError(
        "bad request", category="server", status_code=400, retryable=True
    )
    assert not policy.should_retry(wrong_status, attempt=1)
    assert not policy.should_retry(ValueError("unrelated"), attempt=1)


def test_retry_policy_extreme_multiplier_clamps_instead_of_overflow() -> None:
    policy = RetryPolicy(
        max_attempts=6,
        initial_delay_seconds=1.0,
        backoff_multiplier=1e300,
        max_delay_seconds=8.0,
        jitter_ratio=0.0,
    )
    assert tuple(policy.delays()) == (1.0, 8.0, 8.0, 8.0, 8.0)
