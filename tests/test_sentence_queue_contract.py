from __future__ import annotations

from core.events import ConversationContext, SentenceReady, TextDelta, ToolStatusChanged
from services.conversation import PresentationService


def test_sentence_queue_preserves_submission_order() -> None:
    context = ConversationContext("p", "s", "queue", 1)
    presentation = PresentationService()

    presentation.commit_sentence(SentenceReady(context, "第一句。", 1))
    presentation.commit_sentence(SentenceReady(context, "第二句。", 2))

    assert presentation.pending_sentence_count == 2
    first = presentation.pop_next_sentence()
    second = presentation.pop_next_sentence()
    assert first is not None and first.text == "第一句。"
    assert second is not None and second.text == "第二句。"
    assert presentation.pop_next_sentence() is None
    assert presentation.pending_sentence_count == 0


def test_sentence_queue_peek_requires_ack_and_preserves_retry() -> None:
    context = ConversationContext("p", "s", "queue-retry", 1)
    presentation = PresentationService()
    event = SentenceReady(context, "重试句。", 1)
    presentation.commit_sentence(event)

    peeked = presentation.peek_next_sentence()
    assert peeked == event
    assert presentation.pending_sentence_count == 1
    assert presentation.ack_sentence(event) is True
    assert presentation.pending_sentence_count == 0
    assert presentation.ack_sentence(event) is False


def test_sentence_queue_reset_discards_previous_generation() -> None:
    first_context = ConversationContext("p", "s", "queue-reset-a", 1)
    second_context = ConversationContext("p", "s", "queue-reset-b", 2)
    presentation = PresentationService()
    presentation.commit_sentence(SentenceReady(first_context, "旧句。", 1))

    presentation.reset(context=second_context)
    assert presentation.pop_next_sentence() is None
    presentation.commit_sentence(SentenceReady(second_context, "新句。", 1))
    current = presentation.pop_next_sentence()
    assert current is not None and current.text == "新句。"


def test_sentence_event_does_not_duplicate_raw_output() -> None:
    context = ConversationContext("p", "s", "queue-dual-channel", 1)
    presentation = PresentationService()

    presentation.consume(TextDelta(context, "你好。"))
    presentation.consume(SentenceReady(context, "你好。", 1))

    assert presentation.snapshot.text == "你好。"
    assert presentation.snapshot.speech_text == "你好。"
    assert presentation.snapshot.rendered_text == "你好。"


def test_new_model_text_clears_previous_tool_status() -> None:
    context = ConversationContext("p", "s", "queue-status", 1)
    presentation = PresentationService()
    presentation.consume(ToolStatusChanged(context, "running", "正在执行", "call-1"))
    assert presentation.snapshot.tool_status
    presentation.consume(TextDelta(context, "结果。"))
    assert presentation.snapshot.tool_status == ""
