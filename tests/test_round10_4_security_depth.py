"""第 10 轮方向 4：安全纵深与既定留项的锁定测试。

覆盖：调度动作身份白名单、审计正文渠道级截断阈值、fr2 首事件后重试
锁定、ASR cancel 后进程复用、proactive payload 控制符清洗，以及
scheduler 持久化异常摘要的兜底脱敏。
"""

from __future__ import annotations

import pytest

from core.contracts.chat import ChatMessage, ChatRequest
from services.model_routing.channels import ChannelConfig, channel_from_mapping
from services.model_routing.router import (
    _AUDIT_MESSAGE_TEXT_LIMIT,
    _AUDIT_TRUNCATED_MARK,
    _audit_request_payload,
)
from services.proactive.coordinator import ProactiveCoordinator
from services.scheduler.scheduler import (
    _SCHEDULER_ALLOWED_ACTION_IDENTITIES,
    SchedulerService,
)


class _BrokenStore:
    """总是在持久化时抛出含密钥文本的存储，验证摘要兜底脱敏。"""

    def load_tasks(self):
        raise RuntimeError("storage offline sk-round10-abcdef")

    def save_task(self, _task):
        return None

    def delete_task(self, _task_id):
        return None


def _play_motion_action() -> dict[str, object]:
    return {"identity": "pet:play_motion", "arguments": {"name": "wave"}}


# ---------------------------------------------------------------- A：白名单


def test_scheduler_allows_sample_preset_identities() -> None:
    """A：示例配置 GUI 预设身份与安全 pet 身份照常入队。"""

    scheduler = SchedulerService()
    for identity in ("pet:play_motion", "pet:set_expression", "pet:speak"):
        scheduler.upsert(
            task_id=f"allowed-{identity.split(':')[1]}",
            name="允许",
            expression="every:5m",
            action={"identity": identity, "arguments": {}},
            owner="p:s",
        )
    assert len(scheduler._tasks) == 3


def test_scheduler_rejects_self_reference_and_approval_identities() -> None:
    """A：scheduler 自指与 HIGH/MEDIUM 审批身份一律拒绝，含持久化前拦截。"""

    scheduler = SchedulerService(state_store=_BrokenStore())
    denied = (
        "scheduler:upsert",
        "scheduler:set_trigger",
        "system:run_command",
        "desktop:click_at",
        "desktop:automation_batch",
        "desktop:capture_screen",
        "desktop:ocr",
        "pet:set_click_through",
        "pet:switch_model",
        "system:transcribe_audio",
        "scheduler:remove",
        "pet:diary_write",
        "pet:move",
        "pet:ping",
    )
    for identity in denied:
        with pytest.raises(ValueError, match="not allowed"):
            scheduler.upsert(
                task_id=f"denied-{identity.split(':')[1]}",
                name="拒绝",
                expression="every:5m",
                action={"identity": identity},
                owner="p:s",
            )
    assert scheduler._tasks == {}
    assert identity not in _SCHEDULER_ALLOWED_ACTION_IDENTITIES


def test_scheduler_rejects_missing_or_boolean_identity() -> None:
    """A：缺失或非字符串身份在允许集判定前被明确拒绝。"""

    scheduler = SchedulerService()
    for broken in ({"identity": "   "}, {"identity": True}):
        with pytest.raises(ValueError, match="non-empty string"):
            scheduler.upsert(
                task_id="broken-identity",
                name="坏身份",
                expression="every:5m",
                action=broken,
                owner="p:s",
            )


# ------------------------------------------------- B：审计正文渠道配置


def test_audit_text_limit_default_and_channel_override() -> None:
    """B：默认阈值保持 4000 不变；渠道显式配置更大阈值时保留更长正文。"""

    short_text = "x" * (_AUDIT_MESSAGE_TEXT_LIMIT - 100)
    long_text = "x" * 5000
    request = ChatRequest(
        model="demo",
        messages=(ChatMessage("user", short_text), ChatMessage("assistant", long_text)),
    )
    default_payload = _audit_request_payload(request)
    assert default_payload["messages"][1]["content"].endswith(_AUDIT_TRUNCATED_MARK)

    channel = channel_from_mapping(
        {
            "id": "primary",
            "base_url": "https://primary.invalid/v1",
            "model": "demo",
            "audit_text_limit": 6000,
        }
    )
    wider_payload = _audit_request_payload(request, channel=channel)
    assert "x" * 5000 in wider_payload["messages"][1]["content"]
    assert not wider_payload["messages"][1]["content"].endswith(_AUDIT_TRUNCATED_MARK)


def test_channel_audit_text_limit_below_default_is_rejected() -> None:
    """B：低于全局默认的截断阈值属于配置错误，不能缩小审计口径。"""

    from core.adapters.direct.errors import AdapterConfigurationError

    with pytest.raises(AdapterConfigurationError, match="audit_text_limit"):
        channel_from_mapping(
            {
                "id": "primary",
                "base_url": "https://primary.invalid/v1",
                "model": "demo",
                "audit_text_limit": 1000,
            }
        )
    with pytest.raises(AdapterConfigurationError, match="audit_text_limit"):
        ChannelConfig("primary", base_url="https://x.invalid/v1", audit_text_limit="wide")


# --------------------------- C：fr2 首事件后重试（锁定与渠道级可配置）


def test_fr2_allow_after_first_event_is_channel_configurable() -> None:
    """C：单渠道开启后该渠道首事件后仍可重试该渠道；默认全域保持关闭。"""

    from core.adapters.direct.retry import RetryPolicy

    default_channel = channel_from_mapping(
        {"id": "primary", "base_url": "https://primary.invalid/v1", "model": "demo"}
    )
    assert default_channel.retry.fr2_allow_after_first_event is False
    assert not default_channel.retry.should_retry(
        TimeoutError(), attempt=1, before_first_event=False
    )

    permissive_channel = channel_from_mapping(
        {
            "id": "primary",
            "base_url": "https://primary.invalid/v1",
            "model": "demo",
            "retry": {"fr2_allow_after_first_event": True},
        }
    )
    assert permissive_channel.retry.fr2_allow_after_first_event is True
    assert permissive_channel.retry.should_retry(
        TimeoutError(), attempt=1, before_first_event=False
    )
    assert isinstance(permissive_channel.retry, RetryPolicy)


# ----------------------- D：ASR cancel 后进程保留（worker 常驻语义）


def test_asr_worker_cancel_keeps_process_resident_for_next_request() -> None:
    """D：取消一次转录不清 worker 停止标记，模型保留可服务下一次转录。"""

    from services.asr import worker as asr_worker

    fake_model = object()
    worker = asr_worker.SenseVoiceWorker(
        backend="sensevoice",
        model_path="unused-in-residency-test",
        device="cpu",
        language="auto",
        max_audio_bytes=4096,
    )
    worker.model = fake_model
    emitted: list[tuple[str, object]] = []
    original_emit = asr_worker._emit

    def capture(message_type, *, request_id="", fields=None):
        emitted.append((message_type, fields))

    asr_worker._emit = capture  # type: ignore[assignment]
    try:
        worker.cancel("residency-request")
        asr_worker._emit = original_emit
    finally:
        asr_worker._emit = original_emit
    assert worker._stopping is False
    assert worker.model is fake_model
    assert emitted and emitted[0][0] == "cancelled"
    assert emitted[0][1] == {"cancelled": False}


# ----------------------- E：proactive payload 控制符清洗


def test_proactive_safe_payload_strips_control_characters() -> None:
    """E：含 \x00/\r\n/ESC 的标题进入 payload 前被清洗，随后才截断。"""

    dirty = "干净标题\x00第一段\r\n第二段\x1b[31m结尾"
    cleaned = ProactiveCoordinator._safe_payload({"title": dirty})
    assert cleaned["title"] == "干净标题第一段第二段[31m结尾"
    assert "\x00" not in cleaned["title"]
    assert "\r" not in cleaned["title"] and "\n" not in cleaned["title"]
    assert "\x1b" not in cleaned["title"]

    long_dirty = "标题\x00" + "长" * 300
    result = ProactiveCoordinator._safe_payload({"title": long_dirty})
    assert "\x00" not in result["title"]
    assert len(result["title"]) <= 256

    numeric = ProactiveCoordinator._safe_payload(
        {"pid": 12, "user_active": True, "geometry": [1, 2, 3]}
    )
    assert numeric == {"pid": 12, "user_active": True, "geometry": [1, 2, 3]}


# ----------------------- F：持久化异常摘要兜底脱敏


def test_scheduler_persistence_error_text_is_bounded() -> None:
    """F：持久化异常进入状态接口前经过类名加有界清洗摘要。"""

    scheduler = SchedulerService(state_store=_BrokenStore())
    scheduler.upsert(
        task_id="memory-fallback",
        name="内存",
        expression="every:5m",
        action=_play_motion_action(),
        owner="p:s",
    )
    assert scheduler._tasks
    status = scheduler.status()
    assert "RuntimeError: storage offline" in status["persistence_error"]
    assert len(status["persistence_error"]) <= 192
    assert status["persistence_error"].count("\n") == 0
