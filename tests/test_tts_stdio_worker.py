from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from core.tts.contracts import SpeechRequest
from services.tts.backend import GptSovitsStdioBackend


def _wav_bytes() -> tuple[bytes, bytes]:
    pcm = b"\x10\x00\x20\x00" * 120
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(24000)
        target.writeframes(pcm)
    return buffer.getvalue(), pcm


def test_gpt_sovits_stdio_worker_streams_two_requests_and_unloads(tmp_path: Path) -> None:
    wav_data, expected_pcm = _wav_bytes()
    requests: list[dict[str, object]] = []
    model_requests: list[tuple[str, dict[str, list[str]]]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            requests.append(payload)
            if not payload:
                body = b'{"detail":"missing text"}'
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_data)))
            self.end_headers()
            for offset in range(0, len(wav_data), 37):
                self.wfile.write(wav_data[offset : offset + 37])
                self.wfile.flush()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            model_requests.append((parsed.path, parse_qs(parsed.query)))
            body = b'{"message":"success"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    zh_reference = tmp_path / "zh_normal.wav"
    jp_reference = tmp_path / "jp_normal.wav"
    zh_reference.write_bytes(b"reference-zh")
    jp_reference.write_bytes(b"reference-jp")
    backend = GptSovitsStdioBackend(
        f"http://127.0.0.1:{server.server_port}/tts",
        reference_audios={
            "zh": {"path": str(zh_reference), "text": "你好", "prompt_lang": "zh"},
            "jp": {
                "path": str(jp_reference),
                "text": "こんにちは",
                "prompt_lang": "ja",
            },
        },
        expected_sample_rate=24000,
        timeout_seconds=2.0,
        startup_timeout_seconds=10.0,
        shutdown_timeout_seconds=1.0,
        max_total_bytes=4096,
        default_options={
            "gpt_path": "/models/mea.ckpt",
            "sovits_path": "/models/mea.pth",
        },
    )
    max_total_index = backend.command.index("--max-total-bytes") + 1
    assert int(backend.command[max_total_index]) == backend.max_total_bytes

    async def scenario() -> tuple[int, list[bytes], list[bytes]]:
        health = await backend.start()
        assert health.available is True
        process = backend._process
        assert process is not None
        process_id = int(process.pid)
        first = [
            chunk.data
            async for chunk in backend.stream(SpeechRequest("first", "第一句。", language="zh"))
        ]
        second = [
            chunk.data
            async for chunk in backend.stream(SpeechRequest("second", "二番目。", language="ja"))
        ]
        assert backend._process is process
        await backend.aclose()
        assert backend._process is None
        assert (await backend.health()).available is False
        return process_id, first, second

    try:
        process_id, first, second = asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert process_id > 0
    assert b"".join(first) == expected_pcm
    assert b"".join(second) == expected_pcm
    assert len(requests) == 3
    assert requests[0] == {}
    assert requests[1]["text"] == "第一句。"
    assert requests[2]["text"] == "二番目。"
    assert requests[1]["text_lang"] == "zh"
    assert requests[1]["prompt_lang"] == "zh"
    assert requests[1]["ref_audio_path"] == str(zh_reference)
    assert requests[2]["text_lang"] == "ja"
    assert requests[2]["prompt_lang"] == "ja"
    assert requests[2]["ref_audio_path"] == str(jp_reference)
    assert requests[1]["streaming_mode"] == 3
    assert requests[1]["media_type"] == "wav"
    assert model_requests == [
        ("/set_gpt_weights", {"weights_path": ["/models/mea.ckpt"]}),
        ("/set_sovits_weights", {"weights_path": ["/models/mea.pth"]}),
    ]


def test_stdio_startup_health_rejects_missing_default_language_reference(tmp_path: Path) -> None:
    backend = GptSovitsStdioBackend(
        "http://127.0.0.1:9880/tts",
        ref_dir=str(tmp_path / "missing"),
        startup_language="ja",
        health_probe=False,
    )

    health = asyncio.run(backend.start())

    assert health.available is False
    assert health.engine == "gpt-sovits-stdio"
    assert "jp" in health.message
    assert backend._process is None


def test_owned_stdio_worker_loads_models_routes_languages_and_unloads(tmp_path: Path) -> None:
    engine_root = tmp_path / "engine"
    package = engine_root / "GPT_SoVITS" / "TTS_infer_pack"
    package.mkdir(parents=True)
    (engine_root / "GPT_SoVITS" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    engine_config = engine_root / "fake.yaml"
    engine_config.write_text("fake: true\n", encoding="utf-8")
    gpt_model = engine_root / "mea.ckpt"
    sovits_model = engine_root / "mea.pth"
    gpt_model.write_bytes(b"gpt-model")
    sovits_model.write_bytes(b"sovits-model")
    reloaded_gpt_model = engine_root / "mea-reloaded.ckpt"
    reloaded_sovits_model = engine_root / "mea-reloaded.pth"
    reloaded_gpt_model.write_bytes(b"gpt-model-reloaded")
    reloaded_sovits_model.write_bytes(b"sovits-model-reloaded")
    (package / "TTS.py").write_text(
        """
import json
import time
from pathlib import Path


class TTS_Config:
    def __init__(self, path):
        self.path = path


class Audio:
    def __init__(self, data):
        self.data = data

    def tobytes(self):
        return self.data


class TTS:
    def __init__(self, config):
        self.events = Path(config.path).parent / "events.jsonl"
        self._write({"event": "initialized"})

    def _write(self, value):
        with self.events.open("a", encoding="utf-8") as target:
            target.write(json.dumps(value, ensure_ascii=False) + "\\n")

    def init_t2s_weights(self, path):
        self._write({"event": "gpt", "path": path})

    def init_vits_weights(self, path):
        self._write({"event": "sovits", "path": path})

    def run(self, request):
        self._write({
            "event": "synthesize",
            "text_lang": request["text_lang"],
            "prompt_lang": request["prompt_lang"],
            "reference": request["ref_audio_path"],
        })
        if request["text"] == "取消。":
            for _index in range(100):
                time.sleep(0.01)
                yield 24000, Audio(b"\\x10\\x00")
            return
        yield 24000, Audio(b"\\x10\\x00\\x20\\x00")

    def __del__(self):
        try:
            self._write({"event": "unloaded"})
        except Exception:
            pass
""",
        encoding="utf-8",
    )
    zh_reference = tmp_path / "zh_owned.wav"
    jp_reference = tmp_path / "jp_owned.wav"
    zh_reference.write_bytes(b"reference-zh")
    jp_reference.write_bytes(b"reference-jp")
    backend = GptSovitsStdioBackend(
        engine_root=engine_root,
        engine_config="fake.yaml",
        python_executable=sys.executable,
        reference_audios={
            "zh": {"path": str(zh_reference), "prompt_lang": "zh"},
            "jp": {"path": str(jp_reference), "prompt_lang": "ja"},
        },
        startup_language="zh",
        expected_sample_rate=24000,
        startup_timeout_seconds=2.0,
        shutdown_timeout_seconds=1.0,
        default_options={"gpt_path": str(gpt_model), "sovits_path": str(sovits_model)},
    )

    async def scenario() -> tuple[int, bytes, bytes]:
        health = await backend.start()
        assert health.available is True
        ipc_health = await backend.health()
        assert ipc_health.available is True
        assert "built_in" in ipc_health.message
        diagnostics = backend.diagnostics()
        assert diagnostics["mode"] == "built_in"
        assert diagnostics["model_ready"] is True
        assert diagnostics["ipc"]["ready"] is True
        process = backend._process
        assert process is not None
        process_id = int(process.pid)
        zh_audio = b"".join(
            [
                chunk.data
                async for chunk in backend.stream(
                    SpeechRequest("owned-zh", "你好。", language="zh")
                )
            ]
        )
        reload_health = await backend.reload_models(
            gpt_path=str(reloaded_gpt_model),
            sovits_path=str(reloaded_sovits_model),
        )
        assert reload_health.available is True
        jp_audio = b"".join(
            [
                chunk.data
                async for chunk in backend.stream(
                    SpeechRequest("owned-jp", "こんにちは。", language="ja")
                )
            ]
        )
        cancelled_stream = backend.stream(SpeechRequest("owned-cancel", "取消。", language="zh"))
        first_cancelled_chunk = await anext(cancelled_stream)
        assert first_cancelled_chunk.data == b"\x10\x00"
        await cancelled_stream.aclose()
        assert backend._process is process
        assert (await backend.health()).available is True
        assert backend._process is process
        await backend.aclose()
        return process_id, zh_audio, jp_audio

    process_id, zh_audio, jp_audio = asyncio.run(scenario())
    events = [
        json.loads(line)
        for line in (engine_root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert process_id > 0
    assert zh_audio == b"\x10\x00\x20\x00"
    assert jp_audio == b"\x10\x00\x20\x00"
    assert events[:3] == [
        {"event": "initialized"},
        {"event": "gpt", "path": str(gpt_model)},
        {"event": "sovits", "path": str(sovits_model)},
    ]
    assert {"event": "gpt", "path": str(reloaded_gpt_model)} in events
    assert {"event": "sovits", "path": str(reloaded_sovits_model)} in events
    syntheses = [event for event in events if event["event"] == "synthesize"]
    assert [(event["text_lang"], event["prompt_lang"]) for event in syntheses] == [
        ("zh", "zh"),
        ("ja", "ja"),
        ("zh", "zh"),
    ]
    assert events[-1] == {"event": "unloaded"}
    assert backend._process is None
