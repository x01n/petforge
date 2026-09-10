from __future__ import annotations

import asyncio
import io
import json
import struct
import threading
import wave
from collections.abc import AsyncIterator

import pytest

from core.events.types import AudioChunk, ConversationContext
from core.tts.audio import AudioProtocolError, PcmFormat, StreamingPcmValidator, StreamingWavDecoder
from core.tts.contracts import SpeechChunk, SpeechMethodCall, SpeechRequest
from core.tts.ipc import (
    TTS_IPC_COMMANDS,
    build_ipc_message,
    parse_ipc_message,
)
from services.tts.backend import GptSovitsHttpBackend, SubprocessTTSBackend
from services.tts.coordinator import TTSCoordinator


def test_tts_ipc_contract_round_trips_health_and_rejects_protocol_drift() -> None:
    payload = build_ipc_message(
        "health",
        request_id="health-1",
        fields={"detail": False},
    )
    message = parse_ipc_message(
        json.dumps(payload),
        max_bytes=4096,
        allowed_types=TTS_IPC_COMMANDS,
    )

    assert message.type == "health"
    assert message.request_id == "health-1"
    assert message.payload["detail"] is False

    payload["version"] = 99
    with pytest.raises(ValueError, match="version"):
        parse_ipc_message(
            json.dumps(payload),
            max_bytes=4096,
            allowed_types=TTS_IPC_COMMANDS,
        )


def test_tts_ipc_contract_bounds_messages_and_ids() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        parse_ipc_message("{}" * 100, max_bytes=32)
    with pytest.raises(ValueError, match="request_id"):
        build_ipc_message("cancel", request_id="bad\nrequest")


def _wav_bytes(
    pcm: bytes,
    *,
    sample_rate: int = 24000,
    channels: int = 1,
    include_odd_junk: bool = False,
    data_size: int | None = None,
) -> bytes:
    block_align = channels * 2
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        16,
    )
    chunks = b""
    if include_odd_junk:
        chunks += b"JUNK\x01\x00\x00\x00x\x00"
    chunks += b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(pcm) if data_size is None else data_size) + pcm
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def test_streaming_wav_decoder_handles_split_header_extensions_and_padding() -> None:
    raw = _wav_bytes(b"\x01\x00\x02\x00", include_odd_junk=True)
    decoder = StreamingWavDecoder()
    pieces: list[bytes] = []
    for index in range(0, len(raw), 3):
        pieces.extend(decoder.feed(raw[index : index + 3]))
    assert decoder.finish() == PcmFormat(24000, 1)
    assert b"".join(pieces) == b"\x01\x00\x02\x00"
    assert all(len(piece) % 2 == 0 for piece in pieces)


def test_streaming_wav_decoder_accepts_unknown_length_data_chunk() -> None:
    raw = _wav_bytes(b"\x01\x00\x02\x00", data_size=0)
    decoder = StreamingWavDecoder()
    pieces = decoder.feed(raw[:40])
    pieces += decoder.feed(raw[40:])
    assert decoder.finish() == PcmFormat(24000, 1)
    assert b"".join(pieces) == b"\x01\x00\x02\x00"


def test_streaming_wav_decoder_can_enforce_expected_sample_rate() -> None:
    raw = _wav_bytes(b"\x01\x00", sample_rate=32000)
    decoder = StreamingWavDecoder(expected_sample_rate=32000)
    pieces = decoder.feed(raw)
    assert b"".join(pieces) == b"\x01\x00"
    assert decoder.finish() == PcmFormat(32000, 1)

    wrong_rate = StreamingWavDecoder(expected_sample_rate=32000)
    with pytest.raises(AudioProtocolError, match="sample rate"):
        wrong_rate.feed(_wav_bytes(b"\x01\x00", sample_rate=24000))


@pytest.mark.parametrize("value", (True, 32000.5, "32000"))
def test_streaming_wav_decoder_rejects_non_integer_expected_sample_rate(value: object) -> None:
    with pytest.raises(AudioProtocolError, match="expected WAV sample rate"):
        StreamingWavDecoder(expected_sample_rate=value)  # type: ignore[arg-type]


def test_streaming_wav_decoder_rejects_missing_odd_chunk_padding() -> None:
    """奇数长度 RIFF 元数据块缺少对齐字节时必须拒绝。"""

    fmt = struct.pack("<HHIIHH", 1, 1, 24000, 48000, 2, 16)
    pcm = b"\x01\x00"
    chunks = (
        b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(pcm))
        + pcm
        + b"JUNK"
        + struct.pack("<I", 1)
        + b"x"
    )
    raw = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
    decoder = StreamingWavDecoder()
    decoder.feed(raw)
    with pytest.raises(AudioProtocolError, match="padding"):
        decoder.finish()
    with pytest.raises(AudioProtocolError, match="padding"):
        decoder.finish()


def test_streaming_wav_decoder_rejects_partial_pcm_frame() -> None:
    decoder = StreamingWavDecoder()
    decoder.feed(_wav_bytes(b"\x01"))
    with pytest.raises(AudioProtocolError, match="partial audio frame"):
        decoder.finish()


def test_streaming_pcm_validator_rejects_empty_and_odd_audio() -> None:
    validator = StreamingPcmValidator(24000, 2)
    assert validator.feed(b"\x01\x00") == ()
    with pytest.raises(AudioProtocolError, match="partial audio frame"):
        validator.finish()
    empty = StreamingPcmValidator()
    with pytest.raises(AudioProtocolError, match="no audio"):
        empty.finish()


class _FakeResponse:
    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        self.chunks = chunks
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    async def __aenter__(self) -> _FakeResponse:
        return self.response

    async def __aexit__(self, *_args: object) -> None:
        return None


class _FakeHttpClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.requests: list[tuple[str, str, object]] = []

    async def __aenter__(self) -> _FakeHttpClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def stream(self, method: str, endpoint: str, *, json: object) -> _FakeStream:
        self.requests.append((method, endpoint, json))
        return _FakeStream(self.response)

    async def get(self, _endpoint: str) -> _FakeResponse:
        return self.response


def test_gpt_sovits_http_backend_decodes_streaming_wav_with_fake_client() -> None:
    raw = _wav_bytes(b"\x01\x00\x02\x00", include_odd_junk=True)
    response = _FakeResponse([raw[:2], raw[2:17], raw[17:44], raw[44:]])
    clients: list[_FakeHttpClient] = []

    def factory(**_kwargs: object) -> _FakeHttpClient:
        client = _FakeHttpClient(response)
        clients.append(client)
        return client

    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        client_factory=factory,
        ref_audio_path="/voice.wav",
        prompt_text="参考文本",
    )

    async def scenario() -> list:
        return [chunk async for chunk in backend.stream(SpeechRequest("req", "你好。"))]

    chunks = asyncio.run(scenario())
    assert b"".join(chunk.data for chunk in chunks[:-1]) == b"\x01\x00\x02\x00"
    assert chunks[-1].is_final
    assert clients[0].requests[0][0:2] == ("POST", "http://tts.invalid/tts")
    payload = clients[0].requests[0][2]
    assert isinstance(payload, dict)
    assert payload["ref_audio_path"] == "/voice.wav"
    assert payload["streaming_mode"] == 3


def test_gpt_sovits_http_backend_accepts_official_streaming_header_and_raw_chunks() -> None:
    """官方 api_v2 的流式 WAV 是空 data 头后接裸 PCM，不能按完整 WAV 分片解析。"""

    header_buffer = io.BytesIO()
    with wave.open(header_buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(32000)
        output.writeframes(b"")
    body = header_buffer.getvalue() + b"\x10\x00\x20\x00" * 64
    response = _FakeResponse([body[index : index + 5] for index in range(0, len(body), 5)])
    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        expected_sample_rate=32000,
        ref_audio_path="/voice.wav",
        client_factory=lambda **_kwargs: _FakeHttpClient(response),
    )

    async def scenario() -> list[SpeechChunk]:
        return [chunk async for chunk in backend.stream(SpeechRequest("official", "你好。"))]

    chunks = asyncio.run(scenario())
    assert chunks[-1].is_final
    assert b"".join(chunk.data for chunk in chunks[:-1]) == b"\x10\x00\x20\x00" * 64
    assert all(chunk.sample_rate == 32000 for chunk in chunks)


def test_gpt_sovits_http_backend_raw_pcm_validates_frame_boundaries() -> None:
    response = _FakeResponse([b"\x01", b"\x00\x02", b"\x00"])
    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        media_type="raw",
        ref_audio_path="/voice.wav",
        client_factory=lambda **_kwargs: _FakeHttpClient(response),
    )

    async def scenario() -> list:
        return [chunk async for chunk in backend.stream(SpeechRequest("req", "你好。"))]

    chunks = asyncio.run(scenario())
    assert [chunk.data for chunk in chunks] == [b"\x01\x00", b"\x02\x00", b""]
    assert chunks[-1].is_final


def test_gpt_sovits_http_backend_expected_sample_rate_labels_raw_pcm() -> None:
    response = _FakeResponse([b"\x01\x00\x02\x00"])
    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        media_type="raw",
        expected_sample_rate=32000,
        ref_audio_path="/voice.wav",
        client_factory=lambda **_kwargs: _FakeHttpClient(response),
    )

    async def scenario() -> list[SpeechChunk]:
        return [chunk async for chunk in backend.stream(SpeechRequest("raw-rate", "你好。"))]

    chunks = asyncio.run(scenario())
    assert chunks[-1].is_final
    assert all(chunk.sample_rate == 32000 for chunk in chunks)
    assert backend.raw_format.sample_rate == 32000


def test_gpt_sovits_http_backend_rejects_unexpected_wav_sample_rate() -> None:
    response = _FakeResponse([_wav_bytes(b"\x01\x00", sample_rate=24000)])
    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        expected_sample_rate=32000,
        ref_audio_path="/voice.wav",
        client_factory=lambda **_kwargs: _FakeHttpClient(response),
    )

    async def scenario() -> None:
        async for _chunk in backend.stream(SpeechRequest("wrong-rate", "你好。")):
            pass

    with pytest.raises(RuntimeError, match="invalid wav audio"):
        asyncio.run(scenario())


def test_gpt_sovits_http_backend_accepts_pcm_display_alias() -> None:
    backend = GptSovitsHttpBackend(
        "http://127.0.0.1:9880/tts",
        media_type="pcm",
    )

    assert backend.media_type == "raw"


def test_gpt_sovits_adapter_builds_default_streaming_method_call(tmp_path) -> None:
    reference = tmp_path / "zh_voice.wav"
    reference.write_bytes(b"reference")
    backend = GptSovitsHttpBackend(
        "http://127.0.0.1:9880/tts",
        ref_audio_path=str(reference),
        prompt_text="参考文本",
        prompt_lang="zh",
    )
    request = SpeechRequest(
        "method-call",
        "你好。",
        language="zh",
        mood="soft",
        profile_id="voice-a",
    )

    call = backend.build_method_call(request)

    assert isinstance(call, SpeechMethodCall)
    assert call.request_id == "method-call"
    assert call.method == "POST"
    assert call.endpoint == "http://127.0.0.1:9880/tts"
    assert call.streaming is True
    assert call.payload["streaming_mode"] == 3
    assert call.payload["text"] == "你好。"
    assert call.payload["text_lang"] == "zh"


@pytest.mark.parametrize("value", (True, 32000.5, "32000"))
def test_gpt_sovits_http_backend_rejects_non_integer_expected_sample_rate(value: object) -> None:
    with pytest.raises(ValueError, match="expected_sample_rate"):
        GptSovitsHttpBackend("http://127.0.0.1:9880/tts", expected_sample_rate=value)  # type: ignore[arg-type]


def test_gpt_sovits_http_backend_rejects_truncated_wav() -> None:
    response = _FakeResponse([b"RIFF\x00\x00WAVE"])
    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        ref_audio_path="/voice.wav",
        client_factory=lambda **_kwargs: _FakeHttpClient(response),
    )

    async def scenario() -> None:
        async for _chunk in backend.stream(SpeechRequest("req", "你好。")):
            pass

    with pytest.raises(RuntimeError, match="invalid wav audio"):
        asyncio.run(scenario())


def test_gpt_sovits_http_backend_requires_reference_audio_before_network() -> None:
    calls: list[object] = []

    def factory(**_kwargs: object) -> _FakeHttpClient:
        calls.append(True)
        return _FakeHttpClient(_FakeResponse([]))

    backend = GptSovitsHttpBackend("http://tts.invalid/tts", client_factory=factory)

    async def scenario() -> None:
        async for _chunk in backend.stream(SpeechRequest("req", "你好。")):
            pass

    with pytest.raises(RuntimeError, match="reference audio path is required"):
        asyncio.run(scenario())
    assert not calls


def test_gpt_sovits_http_backend_normalizes_explicit_japanese_profile_language() -> None:
    response = _FakeResponse([_wav_bytes(b"\x01\x00")])
    clients: list[_FakeHttpClient] = []

    def factory(**_kwargs: object) -> _FakeHttpClient:
        client = _FakeHttpClient(response)
        clients.append(client)
        return client

    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        reference_audios={
            "ja": {"path": "/voice/jp_normal.wav"},
        },
        prompt_lang="zh",
        options={"top_k": 9, "unknown_field": "ignored"},
        client_factory=factory,
    )

    async def scenario() -> list:
        return [
            chunk
            async for chunk in backend.stream(
                SpeechRequest("ja-profile", "こんにちは", language="ja")
            )
        ]

    chunks = asyncio.run(scenario())
    assert chunks[-1].is_final
    payload = clients[0].requests[0][2]
    assert isinstance(payload, dict)
    assert payload["text_lang"] == "ja"
    assert payload["prompt_lang"] == "ja"
    assert payload["ref_audio_path"] == "/voice/jp_normal.wav"
    assert payload["top_k"] == 9
    assert "unknown_field" not in payload


def test_gpt_sovits_health_probe_reports_endpoint_status() -> None:
    response = _FakeResponse([], status_code=405)
    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        health_probe=True,
        client_factory=lambda **_kwargs: _FakeHttpClient(response),
    )
    health = asyncio.run(backend.health())
    assert not health.available
    assert health.latency_ms is not None

    malformed = GptSovitsHttpBackend("not-a-url")
    assert not asyncio.run(malformed.health()).available


def test_gpt_sovits_health_probe_accepts_api_validation_error() -> None:
    class _PostProbeClient(_FakeHttpClient):
        async def post(self, endpoint: str, *, json: object) -> _FakeResponse:
            self.requests.append(("POST", endpoint, json))
            return self.response

    response = _FakeResponse([], status_code=400)
    clients: list[_PostProbeClient] = []

    def factory(**_kwargs: object) -> _PostProbeClient:
        client = _PostProbeClient(response)
        clients.append(client)
        return client

    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        health_probe=True,
        client_factory=factory,
    )
    health = asyncio.run(backend.health())
    assert health.available
    assert health.message == "endpoint probe returned HTTP 400"
    assert clients[0].requests == [("POST", "http://tts.invalid/tts", {})]


def test_gpt_sovits_health_probe_has_an_async_deadline_for_custom_clients() -> None:
    class _HangingProbeClient(_FakeHttpClient):
        async def post(self, _endpoint: str, *, json: object) -> _FakeResponse:
            del json
            await asyncio.sleep(1.0)
            return self.response

    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        timeout_seconds=0.02,
        health_probe=True,
        client_factory=lambda **_kwargs: _HangingProbeClient(_FakeResponse([])),
    )

    health = asyncio.run(backend.health())
    assert health.available is False
    assert health.message == "endpoint probe timed out"


def test_subprocess_worker_rejects_mismatched_request_id(tmp_path) -> None:
    import base64
    import json
    import sys

    message = json.dumps(
        {
            "type": "audio",
            "protocol": "meapet.tts.jsonl",
            "version": 1,
            "request_id": "other",
            "data": base64.b64encode(b"\x00\x00").decode(),
        }
    )
    backend = SubprocessTTSBackend(
        (
            sys.executable,
            "-c",
            "import sys; sys.stdin.readline(); print(sys.argv[1], flush=True)",
            message,
        ),
        require_ready=False,
    )

    async def scenario() -> None:
        stream = backend.stream(SpeechRequest("req", "你好"))
        with pytest.raises(RuntimeError, match="request_id"):
            await anext(stream)
        await stream.aclose()

    asyncio.run(scenario())


def test_tts_coordinator_rejects_mismatched_request_id() -> None:
    from core.tts.contracts import EngineHealth, SpeechChunk
    from services.tts.coordinator import SpeechStatus

    class Backend:
        async def health(self) -> EngineHealth:
            return EngineHealth("fake", True)

        async def stream(self, _request):
            yield SpeechChunk("other", b"\x00\x00", 24000, 1, is_final=True)

    audio: list[AudioChunk] = []
    statuses: list[SpeechStatus] = []
    coordinator = TTSCoordinator(Backend(), audio_sink=audio.append, status_sink=statuses.append)
    context = ConversationContext("p", "s", "mismatched-id", 1)

    async def scenario() -> None:
        await coordinator.enqueue_text(context, "你好。", flush=True)

    asyncio.run(scenario())
    assert audio == []
    assert statuses[-1].state == "degraded"


def test_coordinator_does_not_hold_audio_lock_during_blocking_sink() -> None:
    entered = threading.Event()
    release = threading.Event()
    sink_returned = threading.Event()

    def sink(_chunk: AudioChunk) -> None:
        entered.set()
        release.wait(timeout=2)
        sink_returned.set()

    class Backend:
        async def stream(self, request):
            from core.tts.contracts import SpeechChunk

            yield SpeechChunk(request.request_id, b"\x00\x00", 24000, 1, is_final=True)

    coordinator = TTSCoordinator(Backend(), audio_sink=sink)
    context = ConversationContext("p", "s", "lock", 1)

    async def scenario() -> None:
        task = asyncio.create_task(coordinator.enqueue_text(context, "你好。", flush=True))
        assert await asyncio.to_thread(entered.wait, 1)
        finished = threading.Event()

        def cancel() -> None:
            coordinator.cancel(context)
            finished.set()

        thread = threading.Thread(target=cancel)
        thread.start()
        assert finished.wait(0.5)
        release.set()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert sink_returned.is_set()
