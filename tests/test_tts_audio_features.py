from __future__ import annotations

import asyncio
import struct

import pytest

from core.events.types import AudioChunk, AudioFeature, ConversationContext
from core.tts.contracts import EngineHealth, SpeechChunk
from services.tts.audio_features import PcmAudioFeatureAnalyzer
from services.tts.coordinator import TTSCoordinator


def _context(turn_id: str = "feature") -> ConversationContext:
    return ConversationContext("p", "s", turn_id, 1)


def test_audio_feature_requires_explicit_confirmed_source_and_bounded_intensity() -> None:
    feature = AudioFeature(
        _context(),
        request_id="speech-1",
        viseme="A",
        intensity=0.75,
        start_ms=10,
        duration_ms=40,
        source="verified-analyzer",
    )
    assert feature.viseme == "A"
    assert feature.intensity == 0.75
    assert feature.confirmed is True

    with pytest.raises(ValueError, match="explicitly confirmed"):
        AudioFeature(_context(), "speech-1", "A", 0.5, source="analyzer", confirmed=False)
    with pytest.raises(ValueError, match="between 0 and 1"):
        AudioFeature(_context(), "speech-1", "A", 1.1, source="analyzer")
    with pytest.raises(ValueError, match="source is invalid"):
        AudioFeature(_context(), "speech-1", "A", 0.5, source="")


def test_tts_feature_analyzer_dispatches_only_confirmed_events_without_text_guessing() -> None:
    class Backend:
        async def health(self):
            return EngineHealth("fake", True)

        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"\x00\x00", 24000, 1, is_final=True)

    audio: list[AudioChunk] = []
    features: list[AudioFeature] = []

    class Analyzer:
        def analyze(self, chunk: AudioChunk):
            return (
                AudioFeature(
                    chunk.context,
                    request_id=chunk.request_id,
                    viseme="A",
                    intensity=0.6,
                    source="verified-analyzer",
                ),
            )

    coordinator = TTSCoordinator(
        Backend(),
        audio_sink=audio.append,
        feature_analyzer=Analyzer(),
        feature_sink=features.append,
    )
    asyncio.run(coordinator.enqueue_text(_context(), "中文文本。", flush=True))
    assert len(audio) == 1
    assert len(features) == 1
    assert features[0].request_id == audio[0].request_id
    assert features[0].viseme == "A"
    assert features[0].intensity == 0.6


def test_missing_analyzer_keeps_audio_and_emits_no_automatic_feature() -> None:
    class Backend:
        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    audio: list[AudioChunk] = []
    features: list[AudioFeature] = []
    coordinator = TTSCoordinator(Backend(), audio_sink=audio.append, feature_sink=features.append)
    asyncio.run(coordinator.enqueue_text(_context("fallback"), "文本。", flush=True))
    assert [item.data for item in audio] == [b"pcm"]
    assert features == []


def test_stale_feature_is_dropped_without_affecting_audio() -> None:
    class Backend:
        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    features: list[AudioFeature] = []

    class Analyzer:
        def analyze(self, chunk: AudioChunk):
            return (
                AudioFeature(
                    chunk.context,
                    request_id="different-request",
                    viseme="I",
                    intensity=1.0,
                    source="verified-analyzer",
                ),
            )

    coordinator = TTSCoordinator(
        Backend(), feature_analyzer=Analyzer(), feature_sink=features.append
    )
    asyncio.run(coordinator.enqueue_text(_context("stale"), "文本。", flush=True))
    assert features == []


def test_feature_analyzer_failure_preserves_tts_completion() -> None:
    class Backend:
        async def stream(self, request):
            yield SpeechChunk(request.request_id, b"pcm", 24000, 1, is_final=True)

    class BrokenAnalyzer:
        def analyze(self, _chunk: AudioChunk):
            raise RuntimeError("model is not installed")

    statuses = []
    audio: list[AudioChunk] = []
    coordinator = TTSCoordinator(
        Backend(),
        status_sink=statuses.append,
        audio_sink=audio.append,
        feature_analyzer=BrokenAnalyzer(),
        feature_sink=lambda _feature: None,
    )
    asyncio.run(coordinator.enqueue_text(_context("broken"), "文本。", flush=True))
    assert [item.data for item in audio] == [b"pcm"]
    assert [item.state for item in statuses] == ["started", "completed"]


def _pcm_chunk(
    context: ConversationContext,
    request_id: str,
    sample: int,
    *,
    frames: int = 800,
    sample_rate: int = 8000,
    channels: int = 1,
) -> AudioChunk:
    values = [sample] * frames * channels
    return AudioChunk(
        context=context,
        request_id=request_id,
        data=struct.pack("<" + "h" * len(values), *values),
        sample_rate=sample_rate,
        channels=channels,
    )


def test_pcm_audio_feature_analyzer_emits_bounded_acoustic_labels_and_timing() -> None:
    analyzer = PcmAudioFeatureAnalyzer()
    silence = analyzer.analyze(_pcm_chunk(_context("silence"), "speech-silence", 0))
    neutral = analyzer.analyze(_pcm_chunk(_context("neutral"), "speech-neutral", 4096))
    opened = analyzer.analyze(_pcm_chunk(_context("open"), "speech-open", 20000))

    assert silence is not None
    assert neutral is not None
    assert opened is not None
    silence_feature = tuple(silence)[0]
    neutral_feature = tuple(neutral)[0]
    open_feature = tuple(opened)[0]
    assert (silence_feature.viseme, silence_feature.intensity, silence_feature.duration_ms) == (
        "Silence",
        0.0,
        100,
    )
    assert neutral_feature.viseme == "Neutral"
    assert 0.0 < neutral_feature.intensity < 1.0
    assert open_feature.viseme == "Open"
    assert 0.0 < open_feature.intensity <= 1.0
    assert all(
        feature.start_ms == 0 for feature in (silence_feature, neutral_feature, open_feature)
    )


def test_pcm_audio_feature_analyzer_rejects_unknown_data_without_cross_request_state() -> None:
    analyzer = PcmAudioFeatureAnalyzer()
    first = _pcm_chunk(_context("first"), "speech-first", 4096)
    second = _pcm_chunk(_context("second"), "speech-second", 20000)
    object.__setattr__(second, "sample_format", "f32le")

    assert analyzer.analyze(second) is None
    first_feature = tuple(analyzer.analyze(first) or ())[0]
    assert first_feature.context == first.context
    assert first_feature.request_id == first.request_id
    assert not hasattr(first_feature, "data")


def test_pcm_audio_feature_analyzer_dispatches_through_tts_coordinator() -> None:
    class Backend:
        async def stream(self, request):
            yield SpeechChunk(
                request.request_id,
                struct.pack("<800h", *([20000] * 800)),
                8000,
                1,
                is_final=True,
            )

    features: list[AudioFeature] = []
    coordinator = TTSCoordinator(
        Backend(),
        feature_analyzer=PcmAudioFeatureAnalyzer(),
        feature_sink=features.append,
    )
    asyncio.run(coordinator.enqueue_text(_context("pcm-coordinator"), "音频。", flush=True))

    assert len(features) == 1
    assert features[0].viseme == "Open"
    assert features[0].request_id.startswith("speech-")


def test_pcm_audio_feature_analyzer_sequences_chunks_and_releases_request_state() -> None:
    analyzer = PcmAudioFeatureAnalyzer()
    context = _context("sequence")
    first = _pcm_chunk(context, "speech-sequence", 20000, frames=800, sample_rate=8000)
    second = _pcm_chunk(context, "speech-sequence", 0, frames=400, sample_rate=8000)
    object.__setattr__(second, "is_final", True)

    first_feature = tuple(analyzer.analyze(first) or ())[0]
    second_feature = tuple(analyzer.analyze(second) or ())[0]

    assert (first_feature.start_ms, first_feature.duration_ms) == (0, 100)
    assert (second_feature.start_ms, second_feature.duration_ms) == (100, 50)
    assert analyzer._offsets == {}


def test_tts_coordinator_resets_feature_state_when_stream_ends_without_final() -> None:
    class BrokenBackend:
        async def stream(self, request):
            yield SpeechChunk(
                request.request_id,
                struct.pack("<800h", *([20000] * 800)),
                8000,
                1,
                is_final=False,
            )
            raise RuntimeError("stream interrupted")

    analyzer = PcmAudioFeatureAnalyzer()
    context = _context("interrupted")
    coordinator = TTSCoordinator(
        BrokenBackend(),
        feature_analyzer=analyzer,
        feature_sink=lambda _feature: None,
    )

    asyncio.run(coordinator.enqueue_text(context, "中断。", flush=True))

    assert analyzer._offsets == {}
