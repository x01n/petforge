"""第 2 轮新增能力回归：XML 静默门禁、换模入口与主题令牌。"""

from __future__ import annotations

import asyncio

from core.conversation.persona import PersonaPromptBundle
from core.events.types import ConversationContext, TextDelta, TurnFinished
from core.tts.contracts import EngineHealth, SpeechChunk
from services.conversation.presentation import PresentationService
from services.conversation.service import ConversationService
from services.directives import extract_xml_directives, validate_directives
from services.model_routing.channels import ChannelConfig
from services.model_routing.router import ModelRouter
from services.tts.coordinator import TTSCoordinator


class _SpeakableAdapter:
    """返回一段带静默指令的回答。"""

    provider = "silent"
    protocol = "openai_chat"
    capabilities = frozenset({"streaming"})

    async def stream(self, request, *, context=None, cancel_event=None):
        del request, cancel_event
        yield TextDelta(context, "我不会说话。")
        yield TextDelta(context, "<meapet><silent>true</silent></meapet>")
        yield TurnFinished(context)


class _RoutingAdapter(_SpeakableAdapter):
    pass


class _FakeSpeechBackend:
    async def health(self) -> EngineHealth:
        return EngineHealth("fake", True)

    async def stream(self, request):
        yield SpeechChunk(request.request_id, b"pcm", 24000, 1)
        yield SpeechChunk(request.request_id, b"", 24000, 1, is_final=True)


def _router() -> ModelRouter:
    channel = ChannelConfig("fake", base_url="https://fake.invalid/v1", model="demo")
    return ModelRouter(
        [channel],
        adapter_factory=lambda _channel: _RoutingAdapter(),
    )


class _DiagnosticsTTS(TTSCoordinator):
    """记录回合是否收到过文本。"""

    def __init__(self) -> None:
        super().__init__(_FakeSpeechBackend())
        self.enqueued: list[tuple[ConversationContext, str]] = []

    async def enqueue_text(
        self, context: ConversationContext, text: str, **kwargs: object
    ) -> tuple[str, ...]:
        flush = bool(kwargs.get("flush", False))
        self.enqueued.append((context, text))
        return await super().enqueue_text(context, text, language="zh", mood="neutral", flush=flush)


def test_silent_directive_blocks_tts_after_manifest() -> None:
    tts = _DiagnosticsTTS()
    service = ConversationService(_router(), tts=tts)

    result = asyncio.run(service.complete("说点什么"))

    assert result.status == "completed"
    assert result.text == "我不会说话。"
    assert "<meapet>" not in result.text
    # 指令在回合收尾生效：正文前半句仍被表达，但指令标记后的排队
    # 语音被 silent 门禁拦住；记忆/历史只存净正文。
    assert not any("<meapet>" in text for _ctx, text in tts.enqueued)


def test_xml_directive_gate_never_leaks_full_tag() -> None:
    from services.directives import strip_xml_control

    assert strip_xml_control("我不会说话。<meapet><silent>true</silent></meapet>") == "我不会说话。"


def test_persona_bundle_guidance_includes_mood_and_motion_names() -> None:
    bundle = PersonaPromptBundle.from_configuration(
        {"app": {"prompts": {"tool_guidance": "只依据工具真实回执。"}}}
    )
    prompt = bundle.dialogue_system_prompt()
    assert "【工具规则】" in prompt
    assert "只依据工具真实回执。" in prompt


def test_channel_health_dynamic_sort_keeps_low_failures_first() -> None:
    good = ChannelConfig("good", base_url="https://fake.invalid/v1", model="demo")
    bad = ChannelConfig("bad", base_url="https://fake.invalid/v1", model="demo")
    router = ModelRouter(
        [good, bad],
        adapter_factory=lambda _channel: _RoutingAdapter(),
    )
    router._health["bad"].failures = 4
    options = router._healthy_options((good, bad))
    assert options[0].id == "good"


def test_validate_directives_drops_unknown_motion_without_error() -> None:
    directives = extract_xml_directives("<meapet><motion>walk</motion></meapet>")
    validated = validate_directives(directives, allowed_motions=("wave",))
    assert validated.motion == ""


def test_silent_directive_gate_flags_and_release_clears() -> None:
    """silent 置位使门禁为真；silent=False/approve 不误伤；refuse 独立置位。"""
    service = ConversationService(_router(), tts=None)
    context = service.begin_context()

    assert not service._directive_rejected(context)
    assert service.apply_answer_directive(context, silent=True) is True
    assert service._directive_rejected(context)
    service._release_tts_context(context)
    assert not service._directive_rejected(context)

    assert service.apply_answer_directive(context, silent=False) is False
    assert not service._directive_rejected(context)
    assert service.apply_answer_directive(context, approve=True) is True
    assert not service._directive_rejected(context)
    assert service.apply_answer_directive(context, refuse=True) is True
    assert service._directive_rejected(context)
    service._release_tts_context(context)
    assert not service._directive_rejected(context)


def test_apply_answer_directive_silent_returns_true_only_for_silent() -> None:
    """返回值语义：silent/refuse 返回 True，普通或 approve 只透传 approve。"""
    service = ConversationService(_router(), tts=None)
    context = service.begin_context()

    assert service.apply_answer_directive(context, silent=True) is True
    service._release_tts_context(context)
    assert service.apply_answer_directive(context, silent=False) is False
    assert service.apply_answer_directive(context, approve=False) is False
    assert service.apply_answer_directive(context, approve=True, silent=False) is True
    assert service.apply_answer_directive(context, refuse=True) is True


def test_strip_pending_xml_applies_silent_gate_and_keeps_plain_body() -> None:
    """完整指令块立即剥离并置位 silent 门禁，正文保留。"""
    service = ConversationService(_router(), tts=None)
    context = service.begin_context()

    plain = service._strip_pending_xml(context, "正文<meapet><silent>true</silent></meapet>")

    assert plain == "正文"
    assert service._directive_rejected(context)
    assert context in service._silent_directives


def test_strip_pending_xml_malformed_prefix_never_raises() -> None:
    """未闭合指令不抛异常，剥离前文后保留原文其余部分。"""
    service = ConversationService(_router(), tts=None)
    context = service.begin_context()

    for raw in ("正文<meapet><silent", "<meapet><silent", "正文<meapet><refuse"):
        service._silent_directives.discard(context)
        service._rejected_directives.pop(context, None)
        plain = service._strip_pending_xml(context, raw)
        assert plain == raw
        assert not service._directive_rejected(context)


def test_strip_pending_xml_rejects_until_round_finish() -> None:
    """整条无效布尔指令留在正文中，仅由回合收尾整体拒绝。"""
    service = ConversationService(_router(), tts=None)
    context = service.begin_context()

    plain = service._strip_pending_xml(context, "正文<meapet><silent>maybe</silent></meapet>")

    assert plain == "正文"
    assert not service._directive_rejected(context)
    assert context not in service._silent_directives


def test_approve_directive_does_not_reject_the_turn() -> None:
    """approve 是放行信号，绝不把回合置入拒绝门禁。"""
    service = ConversationService(_router(), tts=None)
    context = service.begin_context()

    assert service.apply_answer_directive(context, approve=True) is True
    assert not service._directive_rejected(context)
    assert service._strip_pending_xml(context, "<meapet><approve>true</approve></meapet>") == ""
    assert not service._directive_rejected(context)


class _RefuseAdapter(_SpeakableAdapter):
    """返回一整条带 refuse 指令的回答。"""

    async def stream(self, request, *, context=None, cancel_event=None):
        del request, cancel_event
        yield TextDelta(context, "回答<meapet><refuse>true</refuse></meapet>")
        yield TurnFinished(context)


def _refuse_router() -> ModelRouter:
    channel = ChannelConfig("refuse", base_url="https://fake.invalid/v1", model="demo")
    return ModelRouter([channel], adapter_factory=lambda _channel: _RefuseAdapter())


def test_refuse_turn_end_to_end_skips_tts_and_keeps_plain_text() -> None:
    """refuse 回合：语音队列为空，结果正文为剥离指令后的净正文。"""
    tts = _DiagnosticsTTS()
    service = ConversationService(_refuse_router(), tts=tts)

    result = asyncio.run(service.complete("说点什么"))

    assert result.status == "completed"
    assert result.text == "回答"
    assert "<meapet>" not in result.text
    assert not any("<meapet>" in text for _ctx, text in tts.enqueued)


def test_presentation_consume_strips_xml_marker_delta() -> None:
    """展示层增量含指令标记时，快照正文不出现 <meapet。"""
    presentation = PresentationService()
    context = ConversationContext("default", "local", "turn-pres", 1)
    presentation.reset(context=context)

    presentation.consume(TextDelta(context, "回答<meapet><silent>true</silent></meapet>"))

    assert "<meapet" not in presentation.snapshot.text
    assert presentation.snapshot.text == "回答"


def test_extract_xml_directives_approve_and_refuse_parsed() -> None:
    """approve/refuse 严格布尔解析且计入控制字段。"""
    approved = extract_xml_directives("<meapet><approve>true</approve></meapet>")
    assert approved.approve is True
    assert approved.any_control

    refused = extract_xml_directives("<meapet><refuse>1</refuse></meapet>")
    assert refused.refuse is True
    assert refused.any_control
