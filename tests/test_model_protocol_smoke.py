from __future__ import annotations

import argparse
import asyncio
import importlib.util
from pathlib import Path

from core.events.types import ConversationContext, TextDelta, TurnFinished

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "model_protocol_smoke.py"
_SPEC = importlib.util.spec_from_file_location("model_protocol_smoke", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


class _FakeAdapter:
    def __init__(self, channel, *, client=None) -> None:
        del client
        self.channel = channel
        self.closed = False

    def _url(self) -> str:
        return f"https://example.invalid{self.channel.endpoint}"

    async def stream(self, request):
        del request
        context = ConversationContext("p", "s", "t", 0)
        yield TextDelta(context, "你好。")
        yield TurnFinished(context, "end_turn", {"prompt_tokens": 1})

    async def aclose(self) -> None:
        self.closed = True


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "base_url": "http://127.0.0.1:13000",
        "model": "explicit-model",
        "claude_model": "explicit-claude-model",
        "openai_model": "explicit-openai-model",
        "token": "secret-token",
        "openai_chat_endpoint": "/v1/chat/completions",
        "openai_responses_endpoint": "/v1/responses",
        "openai_omit_temperature": False,
        "message": "只回复OK",
        "max_tokens": 64,
        "timeout_seconds": 5.0,
        "fail_on_error": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_protocol_smoke_uses_explicit_paths_and_never_returns_token(monkeypatch) -> None:
    monkeypatch.setattr(smoke, "ClaudeProviderAdapter", _FakeAdapter)
    monkeypatch.setattr(smoke, "OpenAIProviderAdapter", _FakeAdapter)
    monkeypatch.setattr(smoke, "OpenAIResponsesAdapter", _FakeAdapter)
    result = asyncio.run(smoke._run(_args()))
    assert result["status"] == "ok"
    assert [item["protocol"] for item in result["protocols"]] == [
        "anthropic_messages",
        "openai_chat",
        "openai_responses",
    ]
    assert [item["path"] for item in result["protocols"]] == [
        "/chat/completions",
        "/v1/chat/completions",
        "/v1/responses",
    ]
    assert [item["model"] for item in result["protocols"]] == [
        "explicit-claude-model",
        "explicit-openai-model",
        "explicit-openai-model",
    ]
    assert all(item["status"] == "ok" for item in result["protocols"])
    assert all(item["first_event_ms"] is not None for item in result["protocols"])
    assert all(item["text_delta_count"] == 1 for item in result["protocols"])
    assert "secret-token" not in repr(result)


def test_protocol_smoke_requires_model_without_guessing(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["model_protocol_smoke.py", "--base-url", "http://127.0.0.1:13000"],
    )
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MEAPET_OPENAI_MODEL", raising=False)
    monkeypatch.delenv("MEAPET_MODEL", raising=False)
    try:
        smoke._arguments()
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - parser must reject missing model.
        raise AssertionError("missing model was accepted")
