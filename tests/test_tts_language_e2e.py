"""TTS 语言、参考音频和流式产物的端到端契约测试。

测试使用本地 HTTP 假服务模拟 GPT-SoVITS 推理端点，不访问网络；参考音频和文本
直接读取交付资源，确保请求语言、参考音频语言与输出 PCM 格式保持一致。
"""

from __future__ import annotations

import asyncio
import io
import struct
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from core.events.types import AudioChunk, ConversationContext
from core.tts.contracts import SpeechRequest
from core.tts.language import canonical_tts_language, resolve_mood_reference
from services.tts.backend import GptSovitsHttpBackend
from services.tts.coordinator import SpeechStatus, TTSCoordinator

_RESOURCE_ROOT = Path(__file__).parents[1] / "resources" / "GPT-Sovits"
_REFERENCE_CASES = tuple(
    (language, mood, _RESOURCE_ROOT / mood / f"{language}_{mood}.wav")
    for language in ("zh", "jp")
    for mood in ("normal", "soft", "clam")
)


def _has_japanese_script(value: str) -> bool:
    return any("\u3040" <= char <= "\u30ff" or "\u31f0" <= char <= "\u31ff" for char in value)


def _has_cjk_script(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _reference_text(path: Path) -> str:
    text_path = path.with_suffix(".txt")
    text = text_path.read_text(encoding="utf-8").strip()
    assert text, f"参考文本为空: {text_path}"
    return text


def _reference_format(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as stream:
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
        frame_count = stream.getnframes()
    assert frame_count > 0
    return sample_rate, channels, sample_width


def _pcm_wav(*, sample_rate: int, channels: int, frames: int = 32) -> bytes:
    """生成带确定性非静音脉冲的 16 位 PCM WAV 响应。"""

    samples = bytearray()
    for index in range(frames * channels):
        value = 1200 if index % 8 == 0 else -1200
        samples.extend(struct.pack("<h", value))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(bytes(samples))
    return buffer.getvalue()


def test_language_and_mood_tokens_cannot_escape_reference_directory(tmp_path: Path) -> None:
    ref_dir = tmp_path / "refs"
    (ref_dir / "normal").mkdir(parents=True)
    (tmp_path / "outside_normal.wav").write_bytes(b"not-audio")

    assert canonical_tts_language("../../outside") == ""
    assert (
        resolve_mood_reference(
            ref_dir,
            language="../../outside",
            mood="normal",
        )
        is None
    )
    assert resolve_mood_reference(ref_dir, language="zh", mood="../outside") is None
    with pytest.raises(ValueError, match="language"):
        SpeechRequest("unsafe-language", "你好", language="../../outside")
    with pytest.raises(ValueError, match="mood"):
        SpeechRequest("unsafe-mood", "你好", mood="../outside")
    for language in ("ja", "jp", "日语", "ja-JP"):
        assert (
            SpeechRequest(f"safe-{language}", "こんにちは", language=language).language == language
        )


def test_reference_prompt_text_symlink_cannot_escape_reference_directory(tmp_path: Path) -> None:
    ref_dir = tmp_path / "refs"
    normal_dir = ref_dir / "normal"
    normal_dir.mkdir(parents=True)
    wav_path = normal_dir / "jp_normal.wav"
    wav_path.write_bytes(b"not-audio")
    outside_text = tmp_path / "outside.txt"
    outside_text.write_text("OUTSIDE_SECRET", encoding="utf-8")
    (normal_dir / "jp_normal.txt").symlink_to(outside_text)

    profile = resolve_mood_reference(ref_dir, language="ja", mood="normal")

    assert profile is not None
    assert profile["path"] == str(wav_path.resolve())
    assert profile["text"] == ""


class _Response:
    status_code = 200

    def __init__(self, body: bytes) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        # 故意把 RIFF 头和 data 块拆在不同网络分片中。
        for offset in range(0, len(self._body), 11):
            yield self._body[offset : offset + 11]


class _Stream:
    def __init__(self, response: _Response) -> None:
        self.response = response

    async def __aenter__(self) -> _Response:
        return self.response

    async def __aexit__(self, *_args: object) -> None:
        return None


class _Client:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests: list[tuple[str, str, object]] = []

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def stream(self, method: str, endpoint: str, *, json: object) -> _Stream:
        self.requests.append((method, endpoint, json))
        return _Stream(self.response)


@pytest.mark.parametrize("language,mood,path", _REFERENCE_CASES)
def test_delivered_reference_voice_matches_language_and_pcm_contract(
    language: str, mood: str, path: Path
) -> None:
    if not path.is_file():
        pytest.skip("交付资源目录未复制到工作区")
    text = _reference_text(path)
    sample_rate, channels, sample_width = _reference_format(path)
    assert path.name == f"{language}_{mood}.wav"
    assert sample_rate == (32000 if language == "zh" else 44100)
    assert channels == 1
    assert sample_width == 2
    if language == "jp":
        assert _has_japanese_script(text)
    else:
        assert _has_cjk_script(text)
        assert not _has_japanese_script(text)


@pytest.mark.parametrize(
    "request_language,resource_language,mood",
    (("zh", "zh", "normal"), ("ja", "jp", "normal")),
)
def test_gpt_sovits_language_request_generates_audio_and_coordinator_event(
    request_language: str, resource_language: str, mood: str
) -> None:
    reference_path = _RESOURCE_ROOT / mood / f"{resource_language}_{mood}.wav"
    if not reference_path.is_file():
        pytest.skip("交付资源目录未复制到工作区")
    prompt_text = _reference_text(reference_path)
    sample_rate, channels, sample_width = _reference_format(reference_path)
    assert sample_width == 2
    wav_response = _pcm_wav(sample_rate=sample_rate, channels=channels)
    clients: list[_Client] = []

    def factory(**_kwargs: object) -> _Client:
        client = _Client(_Response(wav_response))
        clients.append(client)
        return client

    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        client_factory=factory,
        ref_dir=str(_RESOURCE_ROOT),
        prompt_lang=request_language,
        media_type="wav",
    )
    request = SpeechRequest(
        f"language-{request_language}",
        "你好。" if request_language == "zh" else "こんにちは。",
        language=request_language,
    )

    async def collect_backend() -> list:
        return [chunk async for chunk in backend.stream(request)]

    chunks = asyncio.run(collect_backend())
    assert chunks[-1].is_final
    audio_data = b"".join(chunk.data for chunk in chunks[:-1])
    assert audio_data
    assert any(byte != 0 for byte in audio_data)
    assert all(chunk.sample_rate == sample_rate for chunk in chunks)
    assert all(chunk.channels == channels for chunk in chunks)
    assert all(len(chunk.data) % (channels * 2) == 0 for chunk in chunks)

    payload = clients[0].requests[0][2]
    assert isinstance(payload, dict)
    assert payload["text_lang"] == request_language
    assert payload["prompt_lang"] == request_language
    assert payload["ref_audio_path"] == str(reference_path)
    assert payload["prompt_text"] == prompt_text
    assert payload["streaming_mode"] == 3

    audio_events: list[AudioChunk] = []
    statuses: list[SpeechStatus] = []
    coordinator = TTSCoordinator(
        backend,
        audio_sink=audio_events.append,
        status_sink=statuses.append,
    )
    context = ConversationContext("default", "local", f"tts-{request_language}", 1)

    async def run_coordinator() -> None:
        await coordinator.enqueue_text(
            context,
            request.text,
            language=request_language,
            mood=mood,
            flush=True,
        )
        await coordinator.aclose()

    asyncio.run(run_coordinator())
    assert audio_events
    assert any(event.data for event in audio_events)
    assert audio_events[-1].is_final
    assert all(event.context == context for event in audio_events)
    assert all(event.sample_rate == sample_rate for event in audio_events)
    assert all(len(event.data) % (channels * 2) == 0 for event in audio_events)
    assert statuses[-1].state == "completed"
    assert len(clients) == 2
    coordinator_payload = clients[1].requests[0][2]
    assert isinstance(coordinator_payload, dict)
    assert coordinator_payload["text_lang"] == request_language
    assert coordinator_payload["prompt_lang"] == request_language


def test_gpt_sovits_http_generator_close_is_logged_as_cancellation(tmp_path: Path, caplog) -> None:
    class PartialResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self) -> AsyncIterator[bytes]:
            yield b"\x01\x00\x02\x00"
            await asyncio.Event().wait()

    class PartialStream:
        async def __aenter__(self) -> PartialResponse:
            return PartialResponse()

        async def __aexit__(self, *_args: object) -> None:
            return None

    class PartialClient:
        async def __aenter__(self) -> PartialClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def stream(self, *_args: object, **_kwargs: object) -> PartialStream:
            return PartialStream()

    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        client_factory=lambda **_kwargs: PartialClient(),
        ref_audio_path=str(tmp_path / "zh_normal.wav"),
        media_type="raw",
    )

    async def scenario() -> None:
        stream = backend.stream(SpeechRequest("generator-close", "你好。"))
        chunk = await anext(stream)
        assert chunk.data == b"\x01\x00\x02\x00"
        await stream.aclose()

    with caplog.at_level("INFO", logger="services.tts.backend"):
        asyncio.run(scenario())

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert '"event": "tts.request.cancel"' in rendered
    assert '"event": "tts.request.failed"' not in rendered


def test_gpt_sovits_rejects_unsupported_korean_language_before_network() -> None:
    """GPT-SoVITS 没有韩语参考/协议契约时必须在联网前明确失败。"""

    backend = GptSovitsHttpBackend("http://tts.invalid/tts", client_factory=lambda **_: None)
    request = SpeechRequest("unsupported-ko", "안녕하세요", language="ko")

    async def collect() -> None:
        async for _chunk in backend.stream(request):
            pass

    with pytest.raises(RuntimeError, match="does not provide"):
        asyncio.run(collect())
