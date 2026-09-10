"""由桌宠进程持有的 GPT-SoVITS JSONL IPC worker。"""

from __future__ import annotations

import argparse
import base64
import contextlib
import gc
import json
import os
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

# 外部 GPT-SoVITS Python 可直接执行本文件；显式加入当前项目 src，保证
# JSONL/PCM 契约代码仍来自同一实现，而不要求外部环境安装 meapet wheel。
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.tts.audio import AudioProtocolError, PcmFormat, StreamingPcmValidator, StreamingWavDecoder
from core.tts.ipc import (
    TTS_IPC_CAPABILITIES,
    TTS_IPC_COMMANDS,
    TTS_IPC_PROTOCOL,
    TTS_IPC_VERSION,
    parse_ipc_message,
    safe_ipc_id,
)

_PROTOCOL = TTS_IPC_PROTOCOL
_VERSION = TTS_IPC_VERSION
_MAX_TEXT_CHARS = 20_000
_MAX_AUDIO_LINE_BYTES = 256 * 1024
_OPTION_KEYS = frozenset(
    {
        "aux_ref_audio_paths",
        "top_k",
        "top_p",
        "temperature",
        "text_split_method",
        "batch_size",
        "batch_threshold",
        "split_bucket",
        "fragment_interval",
        "seed",
        "parallel_infer",
        "repetition_penalty",
        "sample_steps",
        "super_sampling",
        "overlap_length",
        "min_chunk_length",
    }
)
_OUTPUT_LOCK = threading.Lock()


class _SynthesisCancelled(RuntimeError):
    """表示当前推理已由 IPC cancel 指令终止。"""


def _emit(value: Mapping[str, object]) -> None:
    payload = dict(value)
    payload.setdefault("protocol", _PROTOCOL)
    payload.setdefault("version", _VERSION)
    rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _OUTPUT_LOCK:
        sys.stdout.write(rendered)
        sys.stdout.flush()


def _safe_request_id(value: object) -> str:
    return safe_ipc_id(value, required=True)


def _request_payload(
    value: object,
    *,
    media_type: str,
    streaming_mode: int,
    speed: float,
) -> tuple[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise ValueError("request must be an object")
    if value.get("type") != "synthesize" or value.get("streaming") is not True:
        raise ValueError("request type is invalid")
    request_id = _safe_request_id(value.get("request_id"))
    text = str(value.get("text") or "").strip()
    if not text or len(text) > _MAX_TEXT_CHARS or "\x00" in text:
        raise ValueError("request text is invalid")
    reference_audio = str(value.get("reference_audio") or "").strip()
    if not reference_audio or any(char in reference_audio for char in "\x00\r\n"):
        raise ValueError("reference audio is invalid")
    language = str(value.get("language") or "").strip()
    if not language or len(language) > 16:
        raise ValueError("request language is invalid")
    options = value.get("options")
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("request options must be an object")
    payload = {str(key): item for key, item in (options or {}).items() if str(key) in _OPTION_KEYS}
    payload.update(
        {
            "text": text,
            "text_lang": language,
            "ref_audio_path": reference_audio,
            "prompt_text": str(value.get("reference_text") or ""),
            "prompt_lang": str((options or {}).get("prompt_language") or language),
            "streaming_mode": streaming_mode,
            "media_type": media_type,
            "speed_factor": speed,
        }
    )
    return request_id, payload


def _emit_audio(request_id: str, data: bytes, audio_format: PcmFormat) -> int:
    emitted = 0
    frame_bytes = audio_format.channels * 2
    chunk_limit = _MAX_AUDIO_LINE_BYTES - (_MAX_AUDIO_LINE_BYTES % frame_bytes)
    for offset in range(0, len(data), chunk_limit):
        chunk = data[offset : offset + chunk_limit]
        if not chunk:
            continue
        _emit(
            {
                "type": "audio",
                "request_id": request_id,
                "data": base64.b64encode(chunk).decode("ascii"),
                "sample_rate": audio_format.sample_rate,
                "channels": audio_format.channels,
                "sample_format": audio_format.sample_format,
                "final": False,
            }
        )
        emitted += len(chunk)
    return emitted


def _probe(client: Any, endpoint: str) -> None:
    response = client.post(endpoint, json={})
    status = int(response.status_code)
    if not (200 <= status < 300 or status in {400, 422}):
        raise RuntimeError("GPT-SoVITS endpoint is unavailable")


def _model_endpoint(endpoint: str, action: str) -> str:
    parsed = urlparse(endpoint)
    parent = parsed.path.rsplit("/", 1)[0].rstrip("/")
    path = f"{parent}/{action}" if parent else f"/{action}"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _load_external_models(
    client: Any,
    endpoint: str,
    *,
    gpt_path: str,
    sovits_path: str,
) -> None:
    for action, parameter, path in (
        ("set_gpt_weights", "weights_path", gpt_path),
        ("set_sovits_weights", "weights_path", sovits_path),
    ):
        if not path:
            continue
        response = client.get(_model_endpoint(endpoint, action), params={parameter: path})
        if not 200 <= int(response.status_code) < 300:
            raise RuntimeError("GPT-SoVITS model selection failed")


def _synthesize(
    client: Any,
    endpoint: str,
    request_id: str,
    payload: Mapping[str, object],
    *,
    media_type: str,
    raw_format: PcmFormat,
    expected_sample_rate: int | None,
    max_total_bytes: int,
    cancel_event: threading.Event,
) -> None:
    decoder = (
        StreamingWavDecoder(
            max_total_bytes=max_total_bytes,
            expected_sample_rate=expected_sample_rate,
        )
        if media_type == "wav"
        else None
    )
    validator = (
        None
        if decoder is not None
        else StreamingPcmValidator(
            raw_format.sample_rate,
            raw_format.channels,
            max_total_bytes=max_total_bytes,
        )
    )
    audio_bytes = 0
    if cancel_event.is_set():
        raise _SynthesisCancelled("request cancelled before synthesis")
    with client.stream("POST", endpoint, json=dict(payload)) as response:
        response.raise_for_status()
        for data in response.iter_bytes():
            if cancel_event.is_set():
                raise _SynthesisCancelled("request cancelled during synthesis")
            if not data:
                continue
            pieces = decoder.feed(data) if decoder is not None else validator.feed(data)  # type: ignore[union-attr]
            audio_format = decoder.format if decoder is not None else validator.format  # type: ignore[union-attr]
            if audio_format is None:
                continue
            for piece in pieces:
                audio_bytes += _emit_audio(request_id, piece, audio_format)
    audio_format = decoder.finish() if decoder is not None else validator.finish()  # type: ignore[union-attr]
    if audio_bytes <= 0:
        raise AudioProtocolError("GPT-SoVITS returned no audio")
    _emit(
        {
            "type": "done",
            "request_id": request_id,
            "sample_rate": audio_format.sample_rate,
            "channels": audio_format.channels,
        }
    )


def _load_direct_pipeline(
    engine_root: str,
    *,
    engine_config: str,
    gpt_path: str,
    sovits_path: str,
) -> object:
    """在 worker 进程内加载 GPT-SoVITS，并应用精确权重路径。"""

    root = Path(engine_root).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError("GPT-SoVITS engine root is unavailable")
    config = Path(engine_config).expanduser()
    if not config.is_absolute():
        config = root / config
    config = config.resolve()
    if not config.is_file():
        raise RuntimeError("GPT-SoVITS engine config is unavailable")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "GPT_SoVITS"))
    os.chdir(root)
    with contextlib.redirect_stdout(sys.stderr):
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

        pipeline = TTS(TTS_Config(str(config)))
        if gpt_path:
            pipeline.init_t2s_weights(gpt_path)
        if sovits_path:
            pipeline.init_vits_weights(sovits_path)
    return pipeline


def _synthesize_direct(
    pipeline: object,
    request_id: str,
    payload: Mapping[str, object],
    *,
    expected_sample_rate: int | None,
    max_total_bytes: int,
    cancel_event: threading.Event,
) -> None:
    """把模型生成器的 int16 PCM 分片直接写回 JSONL stdout。"""

    request = dict(payload)
    mode = int(request.pop("streaming_mode", 3))
    request.pop("media_type", None)
    request["streaming_mode"] = mode in {2, 3}
    request["return_fragment"] = mode == 1
    request["fixed_length_chunk"] = mode == 3
    runner = getattr(pipeline, "run", None)
    if not callable(runner):
        raise RuntimeError("GPT-SoVITS pipeline has no run method")
    total = 0
    last_format: PcmFormat | None = None
    iterator = iter(runner(request))
    while True:
        if cancel_event.is_set():
            raise _SynthesisCancelled("request cancelled during synthesis")
        try:
            # 第三方推理代码会向 stdout 打印 token 诊断；只在推进其生成器
            # 时重定向，JSONL audio/done 事件必须回到原始 stdout。
            with contextlib.redirect_stdout(sys.stderr):
                sample_rate, audio = next(iterator)
        except StopIteration:
            break
        audio_format = PcmFormat(int(sample_rate), 1)
        if expected_sample_rate is not None and audio_format.sample_rate != expected_sample_rate:
            raise AudioProtocolError("GPT-SoVITS sample rate does not match")
        if isinstance(audio, (bytes, bytearray, memoryview)):
            data = bytes(audio)
        else:
            converter = getattr(audio, "tobytes", None)
            if not callable(converter):
                raise AudioProtocolError("GPT-SoVITS returned unsupported audio")
            data = bytes(converter())
        if len(data) % 2:
            raise AudioProtocolError("GPT-SoVITS returned an incomplete PCM frame")
        total += len(data)
        if total > max_total_bytes:
            raise AudioProtocolError("GPT-SoVITS audio exceeds the configured limit")
        _emit_audio(request_id, data, audio_format)
        last_format = audio_format
    if total <= 0 or last_format is None:
        raise AudioProtocolError("GPT-SoVITS returned no audio")
    _emit(
        {
            "type": "done",
            "request_id": request_id,
            "sample_rate": last_format.sample_rate,
            "channels": last_format.channels,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meapet-tts-worker")
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--engine-root", default="")
    parser.add_argument(
        "--engine-config",
        default="GPT_SoVITS/configs/tts_infer.yaml",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--media-type", choices=("wav", "raw"), default="wav")
    parser.add_argument("--streaming-mode", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--speed-factor", type=float, default=1.0)
    parser.add_argument("--raw-sample-rate", type=int, default=24000)
    parser.add_argument("--raw-channels", type=int, default=1)
    parser.add_argument("--expected-sample-rate", type=int, default=0)
    parser.add_argument("--max-total-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--gpt-path", default="")
    parser.add_argument("--sovits-path", default="")
    parser.add_argument("--no-verify-tls", action="store_true")
    parser.add_argument("--skip-health-probe", action="store_true")
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=2.0)
    return parser


def _safe_model_path(value: object) -> str:
    """校验 IPC 模型路径；路径只用于 worker 内部，不回显到 stdout。"""

    path = str(value or "").strip()
    if len(path) > 4096 or any(char in path for char in "\x00\r\n"):
        raise ValueError("model path is invalid")
    return path


@dataclass
class _WorkerRuntime:
    """管理模型、活动推理和标准输入命令的单进程生命周期。"""

    args: argparse.Namespace
    endpoint: str
    client: Any = None
    pipeline: object | None = None
    mode: str = "external"
    gpt_path: str = ""
    sovits_path: str = ""
    _active_request_id: str = ""
    _active_cancel: threading.Event | None = None
    _active_thread: threading.Thread | None = None
    _state_lock: threading.Lock = field(default_factory=threading.Lock)

    def _active(self) -> bool:
        thread = self._active_thread
        return thread is not None and thread.is_alive()

    def emit_health(self, request_id: str) -> None:
        with self._state_lock:
            active = self._active()
            loaded = self.pipeline is not None or self.client is not None
        _emit(
            {
                "type": "health",
                "request_id": request_id,
                "available": loaded,
                "mode": self.mode,
                "model_loaded": loaded,
                "active": active,
            }
        )

    def load_models(self, request_id: str, gpt_path: str, sovits_path: str) -> None:
        with self._state_lock:
            if self._active():
                _emit(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "message": "model load is unavailable while synthesis is active",
                        "reason_code": "worker_busy",
                    }
                )
                return
        try:
            normalized_gpt = _safe_model_path(gpt_path)
            normalized_sovits = _safe_model_path(sovits_path)
            if self.pipeline is not None:
                with contextlib.redirect_stdout(sys.stderr):
                    if normalized_gpt:
                        loader = getattr(self.pipeline, "init_t2s_weights", None)
                        if not callable(loader):
                            raise RuntimeError("GPT model loader is unavailable")
                        loader(normalized_gpt)
                    if normalized_sovits:
                        loader = getattr(self.pipeline, "init_vits_weights", None)
                        if not callable(loader):
                            raise RuntimeError("SoVITS model loader is unavailable")
                        loader(normalized_sovits)
            elif self.client is not None:
                _load_external_models(
                    self.client,
                    self.endpoint,
                    gpt_path=normalized_gpt,
                    sovits_path=normalized_sovits,
                )
            else:
                raise RuntimeError("TTS engine is unavailable")
        except Exception as exc:
            _emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": "model load failed",
                    "reason_code": "model_load_failed",
                    "error_type": type(exc).__name__,
                }
            )
            return
        self.gpt_path = normalized_gpt or self.gpt_path
        self.sovits_path = normalized_sovits or self.sovits_path
        _emit(
            {
                "type": "model_loaded",
                "request_id": request_id,
                "mode": self.mode,
                "model_ready": True,
                "gpt_loaded": bool(self.gpt_path),
                "sovits_loaded": bool(self.sovits_path),
            }
        )

    def synthesize(self, value: Mapping[str, object]) -> None:
        request_id, payload = _request_payload(
            value,
            media_type=self.args.media_type,
            streaming_mode=self.args.streaming_mode,
            speed=float(self.args.speed_factor),
        )
        with self._state_lock:
            if self._active():
                _emit(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "message": "worker is already synthesizing",
                        "reason_code": "worker_busy",
                    }
                )
                return
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._run_synthesis,
                args=(request_id, payload, cancel_event),
                name="meapet-tts-inference",
                daemon=True,
            )
            self._active_request_id = request_id
            self._active_cancel = cancel_event
            self._active_thread = thread
            thread.start()

    def _run_synthesis(
        self,
        request_id: str,
        payload: Mapping[str, object],
        cancel_event: threading.Event,
    ) -> None:
        try:
            expected_sample_rate = (
                int(self.args.expected_sample_rate) if self.args.expected_sample_rate else None
            )
            if self.pipeline is not None:
                _synthesize_direct(
                    self.pipeline,
                    request_id,
                    payload,
                    expected_sample_rate=expected_sample_rate,
                    max_total_bytes=int(self.args.max_total_bytes),
                    cancel_event=cancel_event,
                )
            else:
                _synthesize(
                    self.client,
                    self.endpoint,
                    request_id,
                    payload,
                    media_type=self.args.media_type,
                    raw_format=PcmFormat(self.args.raw_sample_rate, self.args.raw_channels),
                    expected_sample_rate=expected_sample_rate,
                    max_total_bytes=int(self.args.max_total_bytes),
                    cancel_event=cancel_event,
                )
        except _SynthesisCancelled:
            _emit(
                {
                    "type": "cancelled",
                    "request_id": request_id,
                    "reason_code": "request_cancelled",
                }
            )
        except Exception as exc:
            _emit(
                {
                    "type": "error",
                    "request_id": request_id,
                    "message": "synthesis failed",
                    "reason_code": "synthesis_failed",
                    "error_type": type(exc).__name__,
                }
            )
        finally:
            with self._state_lock:
                if self._active_request_id == request_id:
                    self._active_request_id = ""
                    self._active_cancel = None
                    self._active_thread = None

    def cancel(self, request_id: str) -> None:
        with self._state_lock:
            active_id = self._active_request_id
            cancel_event = self._active_cancel
            active = self._active()
        if active and cancel_event is not None and active_id == request_id:
            cancel_event.set()
            return
        _emit(
            {
                "type": "cancelled",
                "request_id": request_id,
                "reason_code": "request_not_active",
            }
        )

    def shutdown(self, request_id: str) -> None:
        with self._state_lock:
            cancel_event = self._active_cancel
            thread = self._active_thread
        if cancel_event is not None:
            cancel_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.05, float(self.args.shutdown_timeout_seconds)))
        _emit(
            {
                "type": "shutdown",
                "request_id": request_id,
                "clean": thread is None or not thread.is_alive(),
            }
        )

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
        self.pipeline = None
        gc.collect()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    endpoint = str(args.endpoint).strip()
    engine_root = str(args.engine_root or "").strip()
    runtime = _WorkerRuntime(args=args, endpoint=endpoint)
    stage = "configuration"
    shutdown_requested = False
    try:
        PcmFormat(args.raw_sample_rate, args.raw_channels)
        if not 0.1 <= float(args.speed_factor) <= 4.0:
            raise ValueError("speed factor is invalid")
        if args.max_total_bytes < 1024:
            raise ValueError("audio limit is invalid")
        if not 0.05 <= float(args.shutdown_timeout_seconds) <= 30.0:
            raise ValueError("shutdown timeout is invalid")
        if engine_root:
            stage = "engine_load"
            runtime.pipeline = _load_direct_pipeline(
                engine_root,
                engine_config=str(args.engine_config or ""),
                gpt_path=str(args.gpt_path or "").strip(),
                sovits_path=str(args.sovits_path or "").strip(),
            )
            runtime.mode = "built_in"
        else:
            stage = "endpoint_configuration"
            parsed = urlparse(endpoint)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError("endpoint is invalid")
            import httpx

            runtime.client = httpx.Client(
                timeout=httpx.Timeout(float(args.timeout_seconds)),
                verify=not bool(args.no_verify_tls),
            )
            if not args.skip_health_probe:
                stage = "endpoint_health"
                _probe(runtime.client, endpoint)
            stage = "model_load"
            _load_external_models(
                runtime.client,
                endpoint,
                gpt_path=str(args.gpt_path or "").strip(),
                sovits_path=str(args.sovits_path or "").strip(),
            )
        runtime.gpt_path = str(args.gpt_path or "").strip()
        runtime.sovits_path = str(args.sovits_path or "").strip()
    except Exception as exc:
        _emit(
            {
                "type": "error",
                "message": "worker initialization failed",
                "reason_code": "initialization_failed",
                "stage": stage,
                "error_type": type(exc).__name__,
            }
        )
        runtime.close()
        return 2

    _emit(
        {
            "type": "model_loaded",
            "mode": runtime.mode,
            "model_ready": True,
            "gpt_loaded": bool(runtime.gpt_path),
            "sovits_loaded": bool(runtime.sovits_path),
        }
    )
    _emit(
        {
            "type": "ready",
            "protocol": _PROTOCOL,
            "version": _VERSION,
            "streaming": True,
            "mode": runtime.mode,
            "capabilities": list(TTS_IPC_CAPABILITIES),
        }
    )
    try:
        for line in sys.stdin:
            request_id = ""
            try:
                message = parse_ipc_message(
                    line,
                    max_bytes=256 * 1024,
                    allowed_types=TTS_IPC_COMMANDS,
                )
                request_id = message.request_id
                if message.type == "health":
                    runtime.emit_health(request_id)
                elif message.type == "load_model":
                    runtime.load_models(
                        request_id,
                        str(message.payload.get("gpt_path") or ""),
                        str(message.payload.get("sovits_path") or ""),
                    )
                elif message.type == "synthesize":
                    runtime.synthesize(message.payload)
                elif message.type == "cancel":
                    runtime.cancel(_safe_request_id(request_id))
                elif message.type == "shutdown":
                    runtime.shutdown(request_id)
                    shutdown_requested = True
                    break
            except Exception as exc:
                _emit(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "message": "IPC command failed",
                        "reason_code": "invalid_command",
                        "error_type": type(exc).__name__,
                    }
                )
    finally:
        if not shutdown_requested:
            runtime.shutdown("")
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
