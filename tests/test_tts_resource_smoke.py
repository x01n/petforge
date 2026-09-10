"""使用交付资源复核 TTS 参考音频与 HTTP 音频协议；不访问网络。"""

from __future__ import annotations

import asyncio
import io
import os
import struct
import sys
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from core.tts.audio import StreamingWavDecoder
from core.tts.contracts import SpeechRequest
from services.tts.backend import GptSovitsHttpBackend, GptSovitsStdioBackend

_RESOURCE_ROOT = Path(__file__).parents[1] / "resources"
_REFERENCE_WAV = _RESOURCE_ROOT / "GPT-Sovits" / "normal" / "zh_normal.wav"
_PROJECT_ROOT = Path(__file__).parents[1]


class _Response:
    status_code = 200

    def __init__(self, body: bytes) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        for offset in range(0, len(self._body), 17):
            yield self._body[offset : offset + 17]


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


def _wav_with_reference_format(reference: bytes) -> bytes:
    """构造小型 WAV 响应，复用参考音频的 PCM 参数但不发送完整语音文件。"""

    with wave.open(io.BytesIO(reference), "rb") as stream:
        sample_rate = stream.getframerate()
        channels = stream.getnchannels()
        sample_width = stream.getsampwidth()
    assert sample_width == 2
    pcm = b"\x00\x00" * channels * 8
    block_align = channels * sample_width
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        sample_width * 8,
    )
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(pcm)) + pcm
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


@pytest.mark.skipif(not _REFERENCE_WAV.is_file(), reason="resources/ 未复制到工作区")
def test_gpt_sovits_resource_reference_and_stream_smoke() -> None:
    reference = _REFERENCE_WAV.read_bytes()
    assert reference[:4] == b"RIFF"
    clients: list[_Client] = []

    def factory(**_kwargs: object) -> _Client:
        client = _Client(_Response(_wav_with_reference_format(reference)))
        clients.append(client)
        return client

    backend = GptSovitsHttpBackend(
        "http://tts.invalid/tts",
        ref_audio_path=str(_REFERENCE_WAV),
        client_factory=factory,
    )

    async def run() -> list:
        return [chunk async for chunk in backend.stream(SpeechRequest("smoke", "你好。"))]

    chunks = asyncio.run(run())
    assert chunks[-1].is_final
    assert b"".join(chunk.data for chunk in chunks[:-1])
    assert clients[0].requests[0][0:2] == ("POST", "http://tts.invalid/tts")
    payload = clients[0].requests[0][2]
    assert isinstance(payload, dict)
    assert payload["ref_audio_path"] == str(_REFERENCE_WAV)
    assert payload["streaming_mode"] == 3


@pytest.mark.parametrize(
    "reference_path",
    sorted(_RESOURCE_ROOT.glob("GPT-Sovits/*/*.wav")),
)
def test_delivered_reference_wavs_decode_across_network_boundaries(reference_path: Path) -> None:
    """交付的中日语情绪参考均满足流式 WAV 解码和 PCM 帧对齐契约。"""

    decoder = StreamingWavDecoder()
    chunks: list[bytes] = []
    raw = reference_path.read_bytes()
    for offset in range(0, len(raw), 17):
        chunks.extend(decoder.feed(raw[offset : offset + 17]))
    audio_format = decoder.finish()
    assert audio_format.channels == 1
    assert audio_format.sample_rate in {32000, 44100}
    assert b"".join(chunks)


@pytest.mark.skipif(
    os.environ.get("MEAPET_REAL_TTS_SMOKE") != "1",
    reason="设置 MEAPET_REAL_TTS_SMOKE=1 后执行真实本地模型验收",
)
def test_owned_gpt_sovits_zh_ja_stream_and_worker_exit() -> None:
    """真实预载中日模型流并验证 owned worker 在关闭后已退出。"""

    engine_root = _PROJECT_ROOT / "temp" / "third_party" / "GPT-SoVITS"
    python_executable = engine_root / ".venv" / "bin" / "python"
    engine_config = engine_root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    gpt_path = _RESOURCE_ROOT / "models" / "GPT_weights" / "mea_pro-e50.ckpt"
    sovits_path = _RESOURCE_ROOT / "models" / "SoVITS_weights" / "mea_pro_e24_s13704.pth"
    required = (
        python_executable,
        engine_config,
        gpt_path,
        sovits_path,
        _REFERENCE_WAV,
        _RESOURCE_ROOT / "GPT-Sovits" / "normal" / "jp_normal.wav",
    )
    assert all(path.is_file() for path in required)
    backend = GptSovitsStdioBackend(
        engine_root=engine_root,
        engine_config="GPT_SoVITS/configs/tts_infer.yaml",
        python_executable=python_executable,
        ref_dir=_RESOURCE_ROOT / "GPT-Sovits",
        startup_language="zh",
        expected_sample_rate=32000,
        startup_timeout_seconds=300.0,
        shutdown_timeout_seconds=3.0,
        timeout_seconds=180.0,
        default_options={"gpt_path": str(gpt_path), "sovits_path": str(sovits_path)},
    )

    async def scenario() -> tuple[int, dict[str, tuple[int, int, bool]]]:
        health = await backend.start()
        assert health.available is True
        diagnostics = backend.diagnostics()
        assert diagnostics["mode"] == "built_in"
        assert diagnostics["model_ready"] is True
        process = backend._process
        assert process is not None
        process_id = int(process.pid)
        results: dict[str, tuple[int, int, bool]] = {}
        for language, text in (("zh", "你好。"), ("ja", "こんにちは。")):
            chunks = [
                chunk
                async for chunk in backend.stream(
                    SpeechRequest(f"real-{language}", text, language=language)
                )
            ]
            audio = b"".join(chunk.data for chunk in chunks)
            results[language] = (
                len(audio),
                len(chunks),
                bool(chunks and chunks[-1].is_final),
            )
        await backend.aclose()
        return process_id, results

    process_id, results = asyncio.run(scenario())
    assert results.keys() == {"zh", "ja"}
    assert all(
        byte_count > 0 and chunk_count > 0 and final
        for byte_count, chunk_count, final in results.values()
    )
    assert backend._process is None
    if sys.platform.startswith("linux"):
        assert not Path(f"/proc/{process_id}").exists()
