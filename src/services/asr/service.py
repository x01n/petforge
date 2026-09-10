"""由应用框架管理的常驻语音识别 IPC 服务。"""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import os
import secrets
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from core.asr import (
    ASR_IPC_SPEC,
    SUPPORTED_ASR_BACKENDS,
    SUPPORTED_ASR_LANGUAGES,
    ASRHealth,
    TranscriptionRequest,
    TranscriptionResult,
)
from core.ipc import safe_ipc_id
from logger.events import log_event
from services.ipc import IPCProcessError, JSONLSubprocessClient

logger = logging.getLogger(__name__)


class ASRService:
    def __init__(
        self,
        command: Sequence[str],
        *,
        enabled: bool,
        backend: str,
        model_name: str,
        device: str,
        language: str,
        timeout_seconds: float,
        startup_timeout_seconds: float,
        max_audio_bytes: int,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        backend_value = str(backend or "").strip().lower()
        if backend_value not in SUPPORTED_ASR_BACKENDS:
            raise ValueError("ASR backend is invalid")
        language_value = str(language or "").strip().lower()
        if language_value not in SUPPORTED_ASR_LANGUAGES:
            raise ValueError("ASR language is invalid")
        timeout_value = float(timeout_seconds)
        startup_timeout_value = float(startup_timeout_seconds)
        if not 0.1 <= timeout_value <= 600.0:
            raise ValueError("ASR request timeout is invalid")
        if not 0.1 <= startup_timeout_value <= 600.0:
            raise ValueError("ASR startup timeout is invalid")
        if isinstance(max_audio_bytes, bool) or not (
            1024 <= int(max_audio_bytes) <= 64 * 1024 * 1024
        ):
            raise ValueError("ASR audio byte limit is invalid")
        device_value = str(device or "").strip()
        if (
            not device_value
            or len(device_value) > 64
            or any(char in device_value for char in "\x00\r\n")
        ):
            raise ValueError("ASR device is invalid")
        self.enabled = bool(enabled)
        self.backend = backend_value
        self.model_name = str(model_name or "SenseVoiceSmall")[:128]
        self.device = device_value
        self.language = language_value
        self.timeout_seconds = timeout_value
        self.startup_timeout_seconds = startup_timeout_value
        self.max_audio_bytes = int(max_audio_bytes)
        line_limit = min(96 * 1024 * 1024, (self.max_audio_bytes * 4 // 3) + 65536)
        self._client = JSONLSubprocessClient(
            ASR_IPC_SPEC,
            command,
            cwd=cwd,
            env=env,
            component="asr.ipc",
            backend=backend_value,
            max_line_bytes=line_limit,
            startup_timeout_seconds=startup_timeout_value,
            request_timeout_seconds=timeout_value,
            shutdown_timeout_seconds=min(timeout_value, 3.0),
            startup_passthrough_events=frozenset({"status"}),
        )
        self._health = ASRHealth(
            "disabled" if not self.enabled else "unavailable",
            False,
            self.backend,
            self.model_name,
            False,
            self.device,
            self.language,
            "ASR is disabled" if not self.enabled else "ASR has not started",
        )
        self._start_lock = asyncio.Lock()
        self._active: set[str] = set()
        self._transcribe_claimed = False
        self._transcription_task: asyncio.Task[object] | None = None
        self._pending_request_id = ""
        self._health_probe_claimed = False
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._started = False

    @property
    def running(self) -> bool:
        return self._client.running

    @property
    def started(self) -> bool:
        return self._started

    @property
    def closed(self) -> bool:
        return self._closed

    def has_active_tasks(self) -> bool:
        return self._transcribe_claimed or bool(self._active)

    async def start(self) -> ASRHealth:

        async with self._start_lock:
            if self._closing or self._closed:
                return self._set_unavailable("closed", "ASR is closed")
            if not self.enabled:
                return self._health
            if self._health.ready and self.running:
                return self._health
            self._health = ASRHealth(
                "loading",
                False,
                self.backend,
                self.model_name,
                False,
                self.device,
                self.language,
                "ASR model is loading",
            )
            load_started = time.monotonic()
            progress_task = asyncio.create_task(
                self._load_progress(load_started),
                name="meapet-asr-model-progress",
            )
            log_event(
                logger,
                "asr.model.load.start",
                component="asr.service",
                status="started",
                fields={"backend": self.backend, "model": self.model_name},
            )
            try:
                await self._client.start()
                startup = await self._client.request(
                    "startup",
                    request_id=self._request_id("startup"),
                    expected_events=frozenset({"started"}),
                    timeout_seconds=min(self.startup_timeout_seconds, 30.0),
                )
                if not bool(startup.payload.get("accepted", False)):
                    raise IPCProcessError("startup_rejected")
                loaded = await self._client.request(
                    "load_model",
                    request_id=self._request_id("load"),
                    expected_events=frozenset({"model_loaded"}),
                    timeout_seconds=self.startup_timeout_seconds,
                    passthrough_events=frozenset({"status"}),
                )
                if not bool(loaded.payload.get("loaded", False)):
                    raise IPCProcessError("model_unavailable")
            except asyncio.CancelledError:
                self._started = False
                self._set_unavailable("unavailable", "ASR initialization was cancelled")
                log_event(
                    logger,
                    "asr.model.load.cancel",
                    component="asr.service",
                    status="cancelled",
                    fields={"backend": self.backend, "model": self.model_name},
                )
                raise
            except (IPCProcessError, OSError, RuntimeError, TypeError, ValueError) as exc:
                self._started = False
                log_event(
                    logger,
                    "asr.model.load.failed",
                    component="asr.service",
                    status="failed",
                    fields={
                        "backend": self.backend,
                        "model": self.model_name,
                        "error_type": type(exc).__name__,
                    },
                )
                return self._set_unavailable(
                    "unavailable",
                    f"ASR initialization failed: {type(exc).__name__}",
                )
            finally:
                progress_task.cancel()
                await asyncio.gather(progress_task, return_exceptions=True)
            self._started = True
            self._health = ASRHealth(
                "ready",
                True,
                self.backend,
                self.model_name,
                True,
                self.device,
                self.language,
                "ASR is ready",
            )
            log_event(
                logger,
                "asr.model.load.complete",
                component="asr.service",
                status="completed",
                duration_ms=(time.monotonic() - load_started) * 1000.0,
                fields={"backend": self.backend, "model": self.model_name},
            )
            return self._health

    async def _load_progress(self, started_at: float) -> None:
        """在大型本地模型加载期间更新可见状态，避免日志长时间无变化。"""

        try:
            while True:
                await asyncio.sleep(5.0)
                elapsed_ms = max(0.0, (time.monotonic() - started_at) * 1000.0)
                self._health = ASRHealth(
                    "loading",
                    False,
                    self.backend,
                    self.model_name,
                    False,
                    self.device,
                    self.language,
                    f"ASR model is loading ({elapsed_ms / 1000.0:.0f}s)",
                )
                log_event(
                    logger,
                    "asr.model.load.progress",
                    component="asr.service",
                    status="loading",
                    duration_ms=elapsed_ms,
                    fields={"backend": self.backend, "model": self.model_name},
                )
        except asyncio.CancelledError:
            raise

    async def health(self, *, probe: bool = False) -> ASRHealth:
        """返回缓存健康；显式 probe 从不排队到另一条 IPC 请求之后。"""

        if (
            self._start_lock.locked()
            or self.has_active_tasks()
            or self._health_probe_claimed
            or not probe
            or not self.running
            or not self.enabled
        ):
            return self._health
        self._health_probe_claimed = True
        try:
            event = await self._client.request(
                "health",
                request_id=self._request_id("health"),
                expected_events=frozenset({"health"}),
                timeout_seconds=min(self.timeout_seconds, 10.0),
            )
        except asyncio.CancelledError:
            if not self.running:
                self._set_unavailable("unavailable", "ASR health probe was cancelled")
            raise
        except (IPCProcessError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._set_unavailable("unavailable", f"ASR health failed: {type(exc).__name__}")
        finally:
            self._health_probe_claimed = False
        ready = bool(event.payload.get("ready", False))
        loaded = bool(event.payload.get("model_loaded", False))
        self._health = ASRHealth(
            "ready" if ready and loaded else "unavailable",
            ready and loaded,
            self.backend,
            self.model_name,
            loaded,
            self.device,
            self.language,
            "ASR is ready" if ready and loaded else "ASR worker is not ready",
        )
        return self._health

    async def transcribe(
        self,
        audio: bytes,
        *,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        channels: int = 1,
        language: str | None = None,
        request_id: str | None = None,
    ) -> TranscriptionResult:
        """通过 IPC 转写有界内存音频，不向 worker 暴露文件路径。"""

        request = TranscriptionRequest(
            audio,
            audio_format,
            sample_rate,
            channels,
            language or self.language,
        )
        if len(request.audio) > self.max_audio_bytes:
            raise ValueError("ASR audio exceeds configured byte limit")
        if self._closing or self._closed:
            raise RuntimeError("ASR is closed")
        normalized_id = (
            safe_ipc_id(request_id) if request_id is not None else self._request_id("transcribe")
        )
        if not normalized_id:
            raise ValueError("ASR request id is invalid")
        if self._transcribe_claimed:
            raise RuntimeError("ASR transcription is already running")
        self._transcribe_claimed = True
        current_task = asyncio.current_task()
        self._transcription_task = current_task
        self._pending_request_id = normalized_id
        try:
            health = await self.start()
            if not health.ready:
                raise RuntimeError("ASR is unavailable")
            self._active.add(normalized_id)
            self._pending_request_id = ""
            try:
                event = await self._client.request(
                    "transcribe",
                    request_id=normalized_id,
                    fields={
                        "audio_base64": base64.b64encode(request.audio).decode("ascii"),
                        "audio_format": request.audio_format,
                        "sample_rate": request.sample_rate,
                        "channels": request.channels,
                        "language": request.language,
                    },
                    expected_events=frozenset({"transcript", "cancelled"}),
                    timeout_seconds=self.timeout_seconds,
                    passthrough_events=frozenset({"status"}),
                )
            finally:
                self._active.discard(normalized_id)
            if event.type == "cancelled":
                process = self._client.process
                if process is not None and process.returncode is None:
                    try:
                        await asyncio.wait_for(process.wait(), timeout=2.0)
                    except TimeoutError:
                        pass
                self._started = False
                self._set_unavailable("unavailable", "ASR request was cancelled")
                raise asyncio.CancelledError
            confidence_value = event.payload.get("confidence")
            confidence = (
                float(confidence_value)
                if isinstance(confidence_value, (int, float))
                and not isinstance(confidence_value, bool)
                else None
            )
            duration_value = event.payload.get("duration_ms", 0)
            duration_ms = (
                int(duration_value)
                if isinstance(duration_value, (int, float)) and not isinstance(duration_value, bool)
                else 0
            )
            return TranscriptionResult(
                normalized_id,
                str(event.payload.get("text") or ""),
                str(event.payload.get("language") or "unknown"),
                confidence,
                bool(event.payload.get("confidence_available", confidence is not None)),
                duration_ms,
            )
        finally:
            if self._pending_request_id == normalized_id:
                self._pending_request_id = ""
            if self._transcription_task is current_task:
                self._transcription_task = None
            self._transcribe_claimed = False

    async def cancel(self, request_id: str) -> bool:
        """取消模型加载中的调用，或向正在推理的 worker 发送协议中断。"""

        normalized = safe_ipc_id(request_id)
        if not normalized:
            return False
        if normalized == self._pending_request_id:
            task = self._transcription_task
            if task is not None and task is not asyncio.current_task() and not task.done():
                task.cancel()
                return True
            return False
        if normalized not in self._active or not self.running:
            return False
        sender = getattr(self._client, "send_interrupt", None)
        if not callable(sender):
            return False
        sent = sender("cancel", request_id=normalized)
        if not inspect.isawaitable(sent):
            return False
        await sent
        return True

    async def cancel_all(self) -> int:
        request_ids = set(self._active)
        if self._pending_request_id:
            request_ids.add(self._pending_request_id)
        cancelled = 0
        for request_id in tuple(request_ids):
            if await self.cancel(request_id):
                cancelled += 1
        return cancelled

    async def aclose(self) -> None:
        """共享实际清理任务；取消当前等待不会取消 worker 回收。"""

        if self._closed:
            return
        close_task = self._close_task
        if close_task is not None and close_task.done() and not close_task.cancelled():
            close_task.exception()
        if close_task is None or close_task.done():
            self._closing = True
            self._started = False
            close_task = asyncio.create_task(
                self._close_once(),
                name="meapet-asr-service-close",
            )
            self._close_task = close_task
        await asyncio.shield(close_task)

    async def _close_once(self) -> None:
        """按取消业务、等待任务、关闭 IPC 的顺序完成一次真实清理。"""

        transcription_task = self._transcription_task
        try:
            try:
                await asyncio.wait_for(self.cancel_all(), timeout=min(self.timeout_seconds, 1.0))
            except (IPCProcessError, OSError, RuntimeError, TimeoutError, ValueError):
                pass
            if (
                transcription_task is not None
                and transcription_task is not asyncio.current_task()
                and not transcription_task.done()
            ):
                try:
                    await asyncio.wait_for(
                        asyncio.gather(transcription_task, return_exceptions=True),
                        timeout=3.0,
                    )
                except TimeoutError:
                    transcription_task.cancel()
        finally:
            await self._client.shutdown()
        self._active.clear()
        self._pending_request_id = ""
        self._transcription_task = None
        self._transcribe_claimed = False
        self._started = False
        self._health = ASRHealth(
            "closed",
            False,
            self.backend,
            self.model_name,
            False,
            self.device,
            self.language,
            "ASR is closed",
        )
        # closed 只在协议/进程树清理真正结束之后置真。
        self._closed = True
        self._closing = False

    def diagnostics(self) -> dict[str, object]:
        """返回无路径、音频、文本和 stderr 的固定诊断字段。"""

        process = self._client.process
        pid = int(process.pid) if process is not None and process.returncode is None else None
        return {
            "status": self._health.status,
            "available": self._health.available,
            "ready": self._health.ready,
            "message": self._health.message[:240],
            "running": self.running,
            "backend": self.backend,
            "model": self.model_name,
            "model_loaded": self._health.model_loaded,
            "device": self.device,
            "language": self.language,
            "ipc": {
                "status": "ready" if self.running and self._health.ready else "unavailable",
                "running": self.running,
                "ready": self._health.ready,
                "pid": pid,
            },
        }

    def _set_unavailable(self, status: str, message: str) -> ASRHealth:
        self._health = ASRHealth(
            status,
            False,
            self.backend,
            self.model_name,
            False,
            self.device,
            self.language,
            message,
        )
        return self._health

    @staticmethod
    def _request_id(prefix: str) -> str:
        return f"asr-{prefix}-{os.getpid()}-{secrets.token_hex(8)}"


__all__ = ["ASRService"]
