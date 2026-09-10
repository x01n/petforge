from __future__ import annotations

from core.contracts.chat import ChatMessage, ChatRequest, ToolDefinition
from core.conversation.runtime import ConversationKey, GenerationGate
from core.events.types import TextDelta


def test_chat_request_normalizes_messages_and_tools() -> None:
    request = ChatRequest(
        model="demo",
        messages=(ChatMessage("user", "你好"),),
        tools=(ToolDefinition("pet:move", "move", {"type": "object"}),),
    )
    assert request.messages[0].as_mapping()["content"] == "你好"
    assert request.tools[0].identity == "pet:move"


def test_generation_gate_rejects_late_events() -> None:
    gate = GenerationGate()
    key = ConversationKey("direct", "profile", "session")
    old = gate.begin(key, "turn-1")
    new = gate.begin(key, "turn-2")
    assert not gate.accepts(old)
    assert gate.accepts(new)
    assert TextDelta(new, "ok").delta == "ok"
