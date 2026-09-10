"""SenseVoice 常驻 JSONL IPC worker；stdout 只允许协议事件。"""

from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import json
import math
import re
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from core.asr import ASR_IPC_SPEC, SUPPORTED_ASR_LANGUAGES
from core.ipc import build_jsonl_message, parse_jsonl_message

_PROTOCOL_STDOUT = sys.stdout
_WRITE_LOCK = threading.Lock()
_TAG_PATTERN = re.compile(r"<\|([^|]{1,32})\|>")


class _SenseVoiceModel(Protocol):
    """隔离可选 FunASR 依赖，只描述 worker 实际调用的方法。"""

    def generate(
        self,
        *,
        input: object,
        cache: dict[str, object],
        language: str,
        use_itn: bool,
        batch_size_s: int,
    ) -> object: ...


def _emit(
    message_type: str,
    *,
    request_id: str = "",
    fields: Mapping[str, object] | None = None,
) -> None:
    message = build_jsonl_message(
        ASR_IPC_SPEC,
        message_type,
        request_id=request_id,
        fields=fields,
    )
    rendered = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    with _WRITE_LOCK:
        _PROTOCOL_STDOUT.write(rendered + "\n")
        _PROTOCOL_STDOUT.flush()


def _error(request_id: str, reason_code: str) -> None:
    _emit("error", request_id=request_id, fields={"reason_code": str(reason_code)[:64]})


def _model_root(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    required = ("model.pt", "config.yaml", "tokens.json", "am.mvn")
    if not path.is_dir() or not all((path / filename).is_file() for filename in required):
        raise RuntimeError("model_unavailable")
    return path


def _pcm_waveform(audio: bytes, *, sample_rate: int, channels: int) -> object:
    with contextlib.redirect_stdout(sys.stderr):
        np = importlib.import_module("numpy")

    samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape((-1, channels)).mean(axis=1)
    if sample_rate != 16000 and samples.size:
        output_size = max(1, round(samples.size * 16000 / sample_rate))
        source = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
        target = np.linspace(0.0, 1.0, num=output_size, endpoint=False)
        samples = np.interp(target, source, samples).astype(np.float32)
    return samples


def _language_from_text(raw_text: str, requested: str) -> str:
    match = _TAG_PATTERN.match(raw_text)
    if match is not None:
        language = match.group(1).strip().lower()
        if language in SUPPORTED_ASR_LANGUAGES:
            return language
    return requested if requested != "auto" else "unknown"


def _confidence_from_result(result: Mapping[str, object]) -> float | None:
    value = result.get("confidence")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


class SenseVoiceWorker:
    """串行持有一个 SenseVoice 模型，并允许主循环接收取消。"""

    def __init__(
        self,
        *,
        backend: str,
        model_path: str,
        device: str,
        language: str,
        max_audio_bytes: int,
    ) -> None:
        self.backend = backend
        self.model_path = model_path
        self.device = device
        self.language = language
        self.max_audio_bytes = max_audio_bytes
        self.model: _SenseVoiceModel | None = None
        self._active_lock = threading.Lock()
        self._active_request_id = ""
        self._active_cancel: threading.Event | None = None
        self._active_thread: threading.Thread | None = None
        self._stopping = False

    def load_model(self, request_id: str) -> None:
        if self.model is not None:
            _emit(
                "model_loaded",
                request_id=request_id,
                fields={"loaded": True, "backend": self.backend, "model": "SenseVoiceSmall"},
            )
            return
        try:
            root = _model_root(self.model_path)
            _emit("status", request_id=request_id, fields={"state": "loading"})
            with contextlib.redirect_stdout(sys.stderr):
                auto_model = getattr(importlib.import_module("funasr"), "AutoModel")

                self.model = auto_model(
                    model=str(root),
                    trust_remote_code=False,
                    device=self.device,
                    disable_pbar=True,
                    disable_update=True,
                    disable_log=True,
                )
        except Exception:
            self.model = None
            _error(request_id, "model_load_failed")
            return
        _emit(
            "model_loaded",
            request_id=request_id,
            fields={"loaded": True, "backend": self.backend, "model": "SenseVoiceSmall"},
        )

    def health(self, request_id: str) -> None:
        loaded = self.model is not None
        with self._active_lock:
            busy = bool(self._active_request_id)
        _emit(
            "health",
            request_id=request_id,
            fields={
                "ready": loaded and not self._stopping,
                "model_loaded": loaded,
                "busy": busy,
                "backend": self.backend,
                "model": "SenseVoiceSmall",
            },
        )

    def transcribe(self, request_id: str, payload: Mapping[str, object]) -> None:
        if self.model is None:
            _error(request_id, "model_unavailable")
            return
        with self._active_lock:
            if self._active_request_id:
                _error(request_id, "worker_busy")
                return
            cancel = threading.Event()
            self._active_request_id = request_id
            self._active_cancel = cancel
        thread = threading.Thread(
            target=self._run_transcription,
            args=(request_id, payload, cancel),
            name="meapet-asr-transcribe",
            daemon=True,
        )
        with self._active_lock:
            self._active_thread = thread
        thread.start()

    def cancel(self, request_id: str) -> None:
        with self._active_lock:
            active = self._active_request_id == request_id
            cancel = self._active_cancel if active else None
        if cancel is None:
            _emit("cancelled", request_id=request_id, fields={"cancelled": False})
            return
        cancel.set()
        _emit("cancelled", request_id=request_id, fields={"cancelled": True})
        # FunASR 的同步 generate 无可中断句柄。确认取消后退出 worker，
        # 由框架在下次调用时干净重载，绝不让旧推理与新请求并发访问模型。
        self._stopping = True

    def shutdown(self, request_id: str) -> None:
        self._stopping = True
        with self._active_lock:
            cancel = self._active_cancel
        if cancel is not None:
            cancel.set()
        _emit("shutdown", request_id=request_id, fields={"stopped": True})

    def _run_transcription(
        self,
        request_id: str,
        payload: Mapping[str, object],
        cancel: threading.Event,
    ) -> None:
        started = time.monotonic()
        try:
            encoded = payload.get("audio_base64")
            if not isinstance(encoded, str):
                raise ValueError("audio_invalid")
            # 标准 Base64 对 N 个原始字节最多使用 ceil(N / 3) * 4 个字符。
            # 先在字符串边界拒绝明显超限输入，避免为攻击性负载分配完整
            # 解码缓冲；解码后的精确字节上限检查仍保留以覆盖填充分组。
            max_encoded_chars = ((self.max_audio_bytes + 2) // 3) * 4
            if len(encoded) > max_encoded_chars:
                raise ValueError("audio_limit")
            audio = base64.b64decode(encoded.encode("ascii"), validate=True)
            if not audio or len(audio) > self.max_audio_bytes:
                raise ValueError("audio_limit")
            audio_format = str(payload.get("audio_format") or "").strip().lower()
            sample_rate_value = payload.get("sample_rate", 16000)
            channels_value = payload.get("channels", 1)
            if isinstance(sample_rate_value, bool) or not isinstance(sample_rate_value, int):
                raise ValueError("pcm_metadata_invalid")
            if isinstance(channels_value, bool) or not isinstance(channels_value, int):
                raise ValueError("pcm_metadata_invalid")
            sample_rate = sample_rate_value
            channels = channels_value
            language = str(payload.get("language") or self.language).strip().lower()
            if language not in SUPPORTED_ASR_LANGUAGES:
                raise ValueError("language_invalid")
            if audio_format == "wav":
                if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
                    raise ValueError("wav_invalid")
                model_input: object = audio
            elif audio_format == "pcm_s16le":
                if not 8000 <= sample_rate <= 192000 or not 1 <= channels <= 8:
                    raise ValueError("pcm_metadata_invalid")
                if len(audio) % (2 * channels):
                    raise ValueError("pcm_alignment_invalid")
                model_input = _pcm_waveform(audio, sample_rate=sample_rate, channels=channels)
            else:
                raise ValueError("format_invalid")
            if cancel.is_set():
                return
            model = self.model
            if model is None:
                raise RuntimeError("model_unavailable")
            with contextlib.redirect_stdout(sys.stderr):
                result_value = model.generate(
                    input=model_input,
                    cache={},
                    language=language,
                    use_itn=True,
                    batch_size_s=60,
                )
            if cancel.is_set():
                return
            if not isinstance(result_value, list) or not result_value:
                raise RuntimeError("result_invalid")
            result = result_value[0]
            if not isinstance(result, Mapping):
                raise RuntimeError("result_invalid")
            raw_text = str(result.get("text") or "")
            detected_language = _language_from_text(raw_text, language)
            with contextlib.redirect_stdout(sys.stderr):
                postprocess_module = importlib.import_module("funasr.utils.postprocess_utils")
                rich_transcription_postprocess = getattr(
                    postprocess_module, "rich_transcription_postprocess"
                )

                text = str(rich_transcription_postprocess(raw_text)).strip()
            confidence = _confidence_from_result(result)
            if cancel.is_set():
                return
            _emit(
                "transcript",
                request_id=request_id,
                fields={
                    "text": text,
                    "language": detected_language,
                    "confidence": confidence,
                    "confidence_available": confidence is not None,
                    "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
                },
            )
        except Exception as exc:
            if not cancel.is_set():
                reason = str(exc) if isinstance(exc, ValueError) else "transcription_failed"
                _error(request_id, reason)
        finally:
            with self._active_lock:
                if self._active_request_id == request_id:
                    self._active_request_id = ""
                    self._active_cancel = None
                    self._active_thread = None


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", required=True, choices=("sensevoice",))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--language", required=True, choices=tuple(sorted(SUPPORTED_ASR_LANGUAGES)))
    parser.add_argument("--max-audio-bytes", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not 1024 <= args.max_audio_bytes <= 64 * 1024 * 1024:
        return 2
    worker = SenseVoiceWorker(
        backend=args.backend,
        model_path=args.model_path,
        device=args.device,
        language=args.language,
        max_audio_bytes=args.max_audio_bytes,
    )
    _emit(
        "ready",
        fields={
            "backend": args.backend,
            "capabilities": list(ASR_IPC_SPEC.capabilities),
            "model_loaded": False,
        },
    )
    max_line_bytes = min(96 * 1024 * 1024, (args.max_audio_bytes * 4 // 3) + 65536)
    while not worker._stopping:
        line = sys.stdin.buffer.readline(max_line_bytes + 1)
        if not line:
            break
        if len(line) > max_line_bytes or not line.endswith(b"\n"):
            return 2
        try:
            message = parse_jsonl_message(
                ASR_IPC_SPEC,
                line,
                max_bytes=max_line_bytes,
                allowed_types=ASR_IPC_SPEC.commands,
            )
        except ValueError:
            _error("", "invalid_envelope")
            continue
        if message.type == "startup":
            _emit("started", request_id=message.request_id, fields={"accepted": True})
        elif message.type == "load_model":
            worker.load_model(message.request_id)
        elif message.type == "health":
            worker.health(message.request_id)
        elif message.type == "transcribe":
            worker.transcribe(message.request_id, message.payload)
        elif message.type == "cancel":
            worker.cancel(message.request_id)
        elif message.type == "shutdown":
            worker.shutdown(message.request_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SenseVoiceWorker", "main"]
