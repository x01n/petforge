"""可取消的 GPT-SoVITS HTTP/本地子进程边界和文本降级。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import os
import shutil
import sys
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import Any
from urllib.parse import urlparse

from core.tts.audio import (
    AudioProtocolError,
    PcmFormat,
    StreamingPcmValidator,
    StreamingWavDecoder,
)
from core.tts.contracts import EngineHealth, SpeechChunk, SpeechMethodCall, SpeechRequest
from core.tts.ipc import (
    TTS_IPC_CAPABILITIES,
    TTS_IPC_PROTOCOL,
    TTS_IPC_SPEC,
    TTS_IPC_VERSION,
    build_ipc_message,
    parse_ipc_message,
)
from core.tts.language import (
    canonical_tts_language,
    normalize_reference_audios,
    protocol_tts_language,
    resolve_mood_reference,
)
from logger.events import event_fingerprint, log_event
from services.ipc import drain_bounded_stderr, write_jsonl_message
from services.processes import (
    close_process_transport,
    process_group_id,
    process_group_spawn_kwargs,
    terminate_process_tree,
)

logger = logging.getLogger(__name__)

_TTS_WORKER_PROTOCOL = TTS_IPC_PROTOCOL
_TTS_WORKER_PROTOCOL_VERSION = TTS_IPC_VERSION
_GPT_SOVITS_REFERENCE_LANGUAGES = frozenset({"zh", "jp", "en"})

_WORKER_ENVIRONMENT_KEYS = (
    "HOME",
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "TMPDIR",
    "XDG_CACHE_HOME",
)


def _isolated_worker_environment() -> dict[str, str]:
    """构造本地模型 worker 的最小环境，不继承模型渠道凭据。"""

    environment = {key: os.environ[key] for key in _WORKER_ENVIRONMENT_KEYS if key in os.environ}
    environment.update(
        {
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def _protocol_bool(value: object, field_name: str) -> bool:
    """解析 worker JSON 中的布尔字段，拒绝任意非布尔文本。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
    raise RuntimeError(f"TTS worker {field_name} must be boolean")


def _config_bool(value: object, field_name: str) -> bool:
    """解析后端关键字布尔值，避免字符串 ``false`` 变成真值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def _merge_options(*values: Mapping[str, object] | None) -> dict[str, object]:
    """按后者覆盖前者合并 worker 选项。"""

    merged: dict[str, object] = {}
    for value in values:
        if value is not None:
            merged.update({str(key): item for key, item in value.items()})
    return merged


_GPT_SOVITS_HTTP_OPTION_KEYS = frozenset(
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


def _gpt_sovits_http_options(value: Mapping[str, object] | None) -> dict[str, object]:
    """仅转发 api_v2 明确支持的推理字段，协议核心字段由后端固定。"""

    if value is None:
        return {}
    return {
        str(key): item for key, item in value.items() if str(key) in _GPT_SOVITS_HTTP_OPTION_KEYS
    }


def _reference_profile(
    profiles: Mapping[str, Mapping[str, str]],
    *,
    request: SpeechRequest,
    ref_dir: str,
    default_path: str,
    default_text: str,
    default_prompt_lang: str,
) -> dict[str, str]:
    """为请求选择同语言参考音频，不跨语言静默回退。"""

    requested = canonical_tts_language(request.language)
    if requested not in _GPT_SOVITS_REFERENCE_LANGUAGES:
        raise RuntimeError(
            f"GPT-SoVITS does not provide a reference/protocol language: {request.language}"
        )
    profile = profiles.get(requested)
    explicit_path = bool(request.reference_audio)
    mood_profile = resolve_mood_reference(
        ref_dir,
        language=request.language,
        mood=request.mood,
    )
    if profiles and profile is None and mood_profile is None and not explicit_path:
        raise RuntimeError(f"no TTS reference audio is configured for language: {request.language}")

    # 固定 profile 只覆盖自己提供的字段。这样可为某一语言指定文本或固定音色，
    # 同时让其它字段继续使用 ref_dir 的同语言情绪选择。
    selected: dict[str, str] = dict(mood_profile or {})
    if profile is not None:
        for key, value in profile.items():
            normalized = str(value or "").strip()
            if normalized:
                selected[key] = normalized
    path = request.reference_audio or selected.get("path", "")
    if not path and default_path and _path_matches_language(default_path, requested):
        path = default_path
    elif not path and default_path and not _reference_path_has_known_language(default_path):
        # 无法从自定义文件名判断语言时，只有未启用目录路由才允许沿用显式默认路径。
        if not ref_dir and not profiles:
            path = default_path
    text = request.reference_text or selected.get("text", "") or default_text
    prompt_value = selected.get("prompt_lang", "") or default_prompt_lang
    if not selected.get("prompt_lang"):
        path_language = _reference_path_language(selected.get("path", "") or path)
        configured_prompt_language = canonical_tts_language(default_prompt_lang)
        if path_language and path_language != configured_prompt_language:
            # 固定参考音频的文件名前缀是更可靠的语种证据；避免全局中文默认值
            # 覆盖日语/英语 profile，除非调用方在 profile 中显式指定 prompt_lang。
            prompt_value = path_language
    prompt_lang = protocol_tts_language(prompt_value or request.language)
    text_lang = protocol_tts_language(selected.get("text_lang", "") or request.language)
    if not prompt_lang or not text_lang:
        raise RuntimeError(f"invalid TTS language tag for request: {request.language}")
    routed_reference = (
        profiles or ref_dir or (default_path and _reference_path_has_known_language(default_path))
    )
    if routed_reference and not path:
        raise RuntimeError(
            f"no TTS reference audio path is configured for language: {request.language}"
        )
    return {
        "path": str(path or "").strip(),
        "text": str(text or "").strip(),
        "prompt_lang": str(prompt_lang or "").strip(),
        "text_lang": str(text_lang or "").strip(),
    }


def _path_matches_language(path: str, language: str) -> bool:
    """仅依据交付资源的 ``<lang>_*.wav`` 命名判断默认参考是否同语种。"""

    return bool(language and _reference_path_language(path) == language)


def _reference_path_language(path: str) -> str:
    """从交付参考文件名提取规范化语言桶。"""

    name = Path(str(path)).stem.lower()
    prefix = name.split("_", 1)[0]
    language = canonical_tts_language(prefix)
    return language if language in {"zh", "jp", "en"} else ""


def _reference_path_has_known_language(path: str) -> bool:
    """判断交付参考音频命名是否含有已知语言前缀。"""

    return bool(_reference_path_language(path))


class TextOnlyBackend:
    # 文字后端有意不产生 PCM；profile 路由不能把它当作失败的空流。
    requires_audio = False

    async def start(self) -> EngineHealth:
        health = await self.health()
        log_event(
            logger,
            "tts.backend.initialize",
            component="tts.text_only",
            status="ready",
            fields={"backend": "text-only", "available": health.available},
        )
        return health

    async def health(self) -> EngineHealth:
        return EngineHealth("text-only", True, "audio is disabled; text remains available")

    async def stream(self, request: SpeechRequest) -> AsyncIterator[SpeechChunk]:
        log_event(
            logger,
            "tts.request.skipped",
            component="tts.text_only",
            status="disabled",
            level=logging.DEBUG,
            correlation_id=request.request_id,
            fields={"backend": "text-only", "language": request.language},
        )
        if False:
            yield SpeechChunk(request.request_id, b"")

    async def aclose(self) -> None:
        log_event(
            logger,
            "tts.backend.unload",
            component="tts.text_only",
            status="completed",
            fields={"backend": "text-only"},
        )


async def _readline_until_process_exit(
    process: asyncio.subprocess.Process,
    wait_task: asyncio.Task[int],
    *,
    deadline: float | None = None,
) -> bytes:
    """读取 worker 一行；父进程先退出且子进程持有管道时及时返回错误。"""

    assert process.stdout is not None
    read_task = asyncio.create_task(process.stdout.readline())
    try:
        while True:
            if deadline is not None:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise RuntimeError("TTS worker output timed out")
                wait_timeout = min(0.25, remaining)
            else:
                wait_timeout = 0.25
            done, _ = await asyncio.wait(
                (read_task, wait_task), timeout=wait_timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if read_task in done:
                return read_task.result()
            if wait_task in done:
                try:
                    # 正常退出时给继承管道的收尾留出短暂时间。
                    return await asyncio.wait_for(asyncio.shield(read_task), timeout=0.5)
                except TimeoutError as exc:
                    raise RuntimeError("TTS worker exited while a child kept stdout open") from exc
            if process.returncode is not None or _posix_process_gone(process):
                raise RuntimeError("TTS worker exited while a child kept stdout open")
            if deadline is not None and monotonic() >= deadline:
                raise RuntimeError("TTS worker output timed out")
    finally:
        if not read_task.done():
            read_task.cancel()
        await asyncio.gather(read_task, return_exceptions=True)


def _posix_process_gone(process: asyncio.subprocess.Process) -> bool:
    """在 asyncio transport 尚未收到 EOF 时识别已消失的 POSIX 父进程。"""

    if os.name != "posix":
        return False
    try:
        process_id = int(process.pid)
    except (TypeError, ValueError):
        return True
    if sys.platform.startswith("linux"):
        try:
            stat = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        except (OSError, UnicodeError):
            # /proc 可能未挂载、被 hidepid 隐藏或受容器权限限制；读取失败
            # 不是进程退出证据，继续使用可移植的 signal 0 存活探测。
            pass
        else:
            state = stat.rsplit(") ", 1)[-1][:1]
            if state == "Z":
                return True
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False
    return False


class SubprocessTTSBackend:
    """持久 JSONL 标准输入 TTS worker。

    worker 在 start 时启动，并在整个应用生命周期内复用。每个请求通过一行
    JSON 写入 stdin；stdout 逐行返回 audio、done 或 error 事件。旧的单请求
    worker 退出后会在下一次调用前自动重启。
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        max_line_bytes: int = 2 * 1024 * 1024,
        max_chunk_bytes: int = 2 * 1024 * 1024,
        max_total_bytes: int = 64 * 1024 * 1024,
        require_final: bool = False,
        require_audio: bool = True,
        validate_pcm: bool = False,
        timeout_seconds: float = 60.0,
        startup_timeout_seconds: float = 300.0,
        shutdown_timeout_seconds: float = 2.0,
        require_ready: bool = True,
        ref_dir: str | Path = "",
        reference_audios: Mapping[str, object] | None = None,
        ref_audio_path: str = "",
        prompt_text: str = "",
        prompt_lang: str = "",
        default_options: Mapping[str, object] | None = None,
        model_options: Mapping[str, object] | None = None,
        gpt_path: str = "",
        sovits_path: str = "",
        engine_name: str = "subprocess",
        log_component: str = "tts.subprocess",
    ) -> None:
        self.command = tuple(str(part) for part in command)
        if not self.command or any(not part for part in self.command):
            raise ValueError("TTS worker command is required")
        self.cwd = str(cwd) if cwd is not None else None
        self.engine_name = str(engine_name or "").strip()
        self.log_component = str(log_component or "").strip()
        if not self.engine_name or len(self.engine_name) > 128:
            raise ValueError("TTS worker engine_name is invalid")
        if not self.log_component or len(self.log_component) > 128:
            raise ValueError("TTS worker log_component is invalid")
        if any(char in "\x00\r\n" for char in self.engine_name + self.log_component):
            raise ValueError("TTS worker logging identity is invalid")
        self.max_line_bytes = max(1024, int(max_line_bytes))
        requested_total_bytes = max(2, int(max_total_bytes))
        self.max_chunk_bytes = min(max(2, int(max_chunk_bytes)), requested_total_bytes)
        self.max_total_bytes = requested_total_bytes
        self.require_final = _config_bool(require_final, "TTS worker require_final")
        self.require_audio = _config_bool(require_audio, "TTS worker require_audio")
        self.validate_pcm = _config_bool(validate_pcm, "TTS worker validate_pcm")
        try:
            timeout = float(timeout_seconds)
            startup_timeout = float(startup_timeout_seconds)
            shutdown_timeout = float(shutdown_timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("TTS worker timeouts must be numbers") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("TTS worker timeout_seconds must be positive and finite")
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("TTS worker startup_timeout_seconds must be positive and finite")
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("TTS worker shutdown_timeout_seconds must be positive and finite")
        self.timeout_seconds = timeout
        self.startup_timeout_seconds = startup_timeout
        self.shutdown_timeout_seconds = shutdown_timeout
        self.require_ready = _config_bool(require_ready, "TTS worker require_ready")
        self.ref_dir = str(ref_dir or "").strip()
        try:
            self.reference_audios = normalize_reference_audios(reference_audios)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        self.ref_audio_path = str(ref_audio_path or "").strip()
        self.prompt_text = str(prompt_text or "").strip()
        self.prompt_lang = str(prompt_lang or "").strip()
        self.default_options = _merge_options(
            default_options,
            model_options,
            {
                key: value
                for key, value in {
                    "gpt_path": str(gpt_path or "").strip(),
                    "sovits_path": str(sovits_path or "").strip(),
                }.items()
                if value
            },
        )
        self._process: asyncio.subprocess.Process | None = None
        self._wait_task: asyncio.Task[int] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._group_id: int | None = None
        self._process_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._dispose_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._startup_tasks: set[asyncio.Task[Any]] = set()
        self._active_stream_task: asyncio.Task[Any] | None = None
        self._active_request_id = ""
        self._worker_capabilities: frozenset[str] = frozenset()
        self._worker_ready = False
        self._configured_worker_mode = "legacy"
        self._worker_mode = self._configured_worker_mode
        self._worker_model_loaded = False
        self._startup_error: dict[str, str] = {}
        self._worker_environment = _isolated_worker_environment()
        # 某些外部 worker 会在 ``audio(final=true)`` 后仍发送 ``done``。
        # 生成器已按 final 返回时，这个终态会留在常驻 stdout；只记录有限的
        # 已完成请求，下一次读取时可严格区分迟到 done 与真正的 request_id 错配。
        self._completed_request_ids: deque[str] = deque(maxlen=256)
        self._completed_request_set: set[str] = set()

    def _remember_completed_request(self, request_id: str) -> None:
        """记录有限数量的已完成请求，供常驻 worker 清理迟到 done。"""

        normalized = str(request_id or "").strip()
        if not normalized or normalized in self._completed_request_set:
            return
        if len(self._completed_request_ids) >= self._completed_request_ids.maxlen:
            expired = self._completed_request_ids.popleft()
            self._completed_request_set.discard(expired)
        self._completed_request_ids.append(normalized)
        self._completed_request_set.add(normalized)

    def _operation(self, command: str, request_id: object = "") -> str:
        """返回不含原始 ID、命令路径或正文的稳定操作指纹。"""

        return event_fingerprint(f"{self.engine_name}:{command}:{request_id}")

    def _log_ipc(
        self,
        command: str,
        status: str,
        *,
        request_id: object = "",
        reason_code: str = "",
        return_code: int | None = None,
        level: int = logging.INFO,
    ) -> None:
        terminal = status in {"completed", "failed", "cancelled", "unavailable"}
        event = (
            "tts.worker.ipc.complete"
            if status == "completed"
            else "tts.worker.ipc.cancelled"
            if status == "cancelled"
            else "tts.worker.ipc.failed"
            if terminal
            else "tts.worker.ipc.start"
        )
        log_event(
            logger,
            event,
            component=self.log_component,
            status=status,
            level=level,
            operation_id=f"{self.engine_name}:{command}:{request_id}",
            request_id=request_id,
            reason_code=reason_code or None,
            fields={
                "backend": self.engine_name,
                "ipc_action": command,
                "return_code": return_code,
            },
        )

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        """有界消费第三方 stderr，只记录摘要指纹和长度。"""

        await drain_bounded_stderr(
            process.stderr,
            max_line_bytes=self.max_line_bytes,
            component=self.log_component,
            backend=self.engine_name,
            target_logger=logger,
        )

    def _configuration_health(self) -> EngineHealth:
        if not self.command:
            return EngineHealth(self.engine_name, False, "worker command is not configured")
        executable = self.command[0]
        if any(separator in executable for separator in (os.sep, os.altsep) if separator):
            available = Path(executable).is_file() and os.access(executable, os.X_OK)
        else:
            available = shutil.which(executable) is not None
        if not available:
            return EngineHealth(self.engine_name, False, "worker executable is not available")
        if self.cwd is not None and not Path(self.cwd).is_dir():
            return EngineHealth(self.engine_name, False, "worker cwd does not exist")
        return EngineHealth(self.engine_name, True, "configured persistent JSONL worker")

    async def start(self) -> EngineHealth:
        """跟踪启动调用，确保关闭可打断尚未完成的模型握手。"""

        if self._closing or self._closed:
            return EngineHealth(self.engine_name, False, "worker backend is closed")
        task = asyncio.current_task()
        if task is not None:
            self._startup_tasks.add(task)
        try:
            return await self._start_once()
        finally:
            if task is not None:
                self._startup_tasks.discard(task)

    async def _start_once(self) -> EngineHealth:
        """启动并保留 worker；重复调用不会创建第二个进程。"""

        configured = self._configuration_health()
        if not configured.available:
            log_event(
                logger,
                "tts.backend.initialize",
                component=self.log_component,
                status="unavailable",
                level=logging.WARNING,
                fields={"backend": self.engine_name, "reason": configured.message},
            )
            return configured
        async with self._process_lock:
            if self._closing or self._closed:
                return EngineHealth(self.engine_name, False, "worker backend is closed")
            process = self._process
            if process is not None and process.returncode is None:
                return EngineHealth(self.engine_name, True, "persistent JSONL worker is running")
            if process is not None:
                await self._dispose_process(process)
            self._log_ipc("start", "started")
            try:
                process = await asyncio.create_subprocess_exec(
                    *self.command,
                    cwd=self.cwd,
                    env=self._worker_environment,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=self.max_line_bytes + 1,
                    **process_group_spawn_kwargs(),
                )
            except (OSError, RuntimeError) as exc:
                self._log_ipc(
                    "start",
                    "failed",
                    reason_code="spawn_failed",
                    level=logging.WARNING,
                )
                log_event(
                    logger,
                    "tts.backend.initialize",
                    component=self.log_component,
                    status="failed",
                    level=logging.WARNING,
                    reason_code="spawn_failed",
                    fields={"backend": self.engine_name, "error_type": type(exc).__name__},
                )
                return EngineHealth(
                    self.engine_name,
                    False,
                    f"worker startup failed: {type(exc).__name__}",
                )
            self._process = process
            self._wait_task = asyncio.create_task(process.wait())
            self._stderr_task = asyncio.create_task(self._drain_stderr(process))
            self._group_id = process_group_id(process)
            self._worker_ready = not self.require_ready
            try:
                if self.require_ready:
                    await self._await_ready(process, self._wait_task)
                else:
                    done, _ = await asyncio.wait((self._wait_task,), timeout=0.05)
                    if done or process.returncode is not None:
                        raise RuntimeError("TTS worker exited during startup")
            except asyncio.CancelledError:
                self._log_ipc(
                    "start",
                    "cancelled",
                    reason_code="startup_cancelled",
                    return_code=process.returncode,
                    level=logging.INFO,
                )
                await self._dispose_process(process)
                raise
            except (RuntimeError, ValueError) as exc:
                return_code = process.returncode
                startup_error = dict(self._startup_error)
                if not startup_error:
                    startup_error = {
                        "reason_code": "handshake_failed",
                        "stage": "handshake",
                        "error_type": type(exc).__name__,
                    }
                    self._startup_error = dict(startup_error)
                await self._dispose_process(process, graceful=process.returncode is not None)
                reason_code = startup_error.get("reason_code", "handshake_failed")
                self._log_ipc(
                    "start",
                    "failed",
                    reason_code=reason_code,
                    return_code=return_code,
                    level=logging.WARNING,
                )
                log_event(
                    logger,
                    "tts.backend.initialize",
                    component=self.log_component,
                    status="failed",
                    level=logging.WARNING,
                    reason_code=reason_code,
                    fields={
                        "backend": self.engine_name,
                        "error_type": type(exc).__name__,
                        "stage": startup_error.get("stage", "handshake"),
                        "return_code": return_code,
                    },
                )
                return EngineHealth(
                    self.engine_name,
                    False,
                    self._startup_health_message(),
                )
            self._log_ipc("start", "completed", return_code=process.returncode)
            log_event(
                logger,
                "tts.backend.initialize",
                component=self.log_component,
                status="ready",
                fields={
                    "backend": self.engine_name,
                    "available": True,
                    "pid": process.pid,
                },
            )
            return EngineHealth(self.engine_name, True, "persistent JSONL worker is running")

    def _startup_health_message(self) -> str:
        """生成不含路径、端点和第三方原文的可诊断启动结果。"""

        if not self._startup_error:
            return "worker initialization handshake failed"
        reason = self._startup_error.get("reason_code", "handshake_failed")[:64]
        stage = self._startup_error.get("stage", "handshake")[:64]
        error_type = self._startup_error.get("error_type", "RuntimeError")[:64]
        return f"worker initialization failed: {reason} at {stage} ({error_type})"

    async def _await_ready(
        self,
        process: asyncio.subprocess.Process,
        wait_task: asyncio.Task[int],
    ) -> None:
        """消费有界启动事件，直到 worker 明确确认模型和音色已就绪。"""

        self._startup_error = {}
        self._worker_ready = False
        deadline = monotonic() + self.startup_timeout_seconds
        for _index in range(256):
            try:
                line = await _readline_until_process_exit(
                    process,
                    wait_task,
                    deadline=deadline,
                )
            except ValueError as exc:
                raise RuntimeError("TTS worker startup line exceeds limit") from exc
            if not line:
                raise RuntimeError("TTS worker exited before ready")
            if len(line) > self.max_line_bytes:
                raise RuntimeError("TTS worker startup line exceeds limit")
            try:
                parsed = parse_ipc_message(
                    line,
                    max_bytes=self.max_line_bytes,
                    allowed_types=TTS_IPC_SPEC.events,
                )
            except ValueError as exc:
                raise RuntimeError("TTS worker returned an invalid startup event") from exc
            message = parsed.payload
            event_type = parsed.type
            if event_type in {"status", "log"}:
                continue
            if event_type == "model_loaded":
                mode = str(message.get("mode") or "").strip().lower()
                if mode in {"built_in", "external"}:
                    self._worker_mode = mode
                self._worker_model_loaded = bool(message.get("model_ready", False))
                continue
            if event_type == "error":
                self._startup_error = {
                    "reason_code": str(message.get("reason_code") or "worker_error")[:64],
                    "stage": str(message.get("stage") or "initialization")[:64],
                    "error_type": str(message.get("error_type") or "RuntimeError")[:64],
                }
                raise RuntimeError("TTS worker reported an initialization error")
            if event_type != "ready":
                raise RuntimeError("TTS worker returned an unexpected startup event")
            if message.get("streaming") is not True:
                raise RuntimeError("TTS worker does not support streaming")
            raw_capabilities = message.get("capabilities", ())
            if isinstance(raw_capabilities, list):
                self._worker_capabilities = frozenset(
                    str(item).strip()
                    for item in raw_capabilities
                    if str(item).strip() in TTS_IPC_CAPABILITIES
                )
            mode = str(message.get("mode") or "").strip().lower()
            if mode in {"built_in", "external"}:
                self._worker_mode = mode
            self._worker_ready = True
            return
        raise RuntimeError("TTS worker emitted too many startup events")

    async def _write_ipc_command(
        self,
        process: asyncio.subprocess.Process,
        command: str,
        *,
        request_id: str = "",
        fields: Mapping[str, object] | None = None,
    ) -> None:
        if process.stdin is None or process.stdin.is_closing():
            raise RuntimeError("TTS worker stdin is unavailable")
        payload = build_ipc_message(command, request_id=request_id, fields=fields)
        await write_jsonl_message(
            process.stdin,
            payload,
            max_line_bytes=self.max_line_bytes,
        )

    async def _await_ipc_event(
        self,
        process: asyncio.subprocess.Process,
        wait_task: asyncio.Task[int],
        *,
        request_id: str,
        expected: frozenset[str],
        timeout: float,
    ) -> Mapping[str, object]:
        deadline = monotonic() + timeout
        for _index in range(4096):
            try:
                line = await _readline_until_process_exit(process, wait_task, deadline=deadline)
            except ValueError as exc:
                raise RuntimeError("TTS worker IPC line exceeds limit") from exc
            if not line:
                raise RuntimeError("TTS worker exited before IPC acknowledgement")
            if len(line) > self.max_line_bytes:
                raise RuntimeError("TTS worker IPC line exceeds limit")
            try:
                parsed = parse_ipc_message(
                    line,
                    max_bytes=self.max_line_bytes,
                    allowed_types=TTS_IPC_SPEC.events,
                )
            except ValueError as exc:
                raise RuntimeError("TTS worker returned an invalid IPC event") from exc
            message = parsed.payload
            event_type = parsed.type
            response_request_id = parsed.request_id
            if event_type in {"audio", "done", "error", "cancelled"}:
                if not response_request_id:
                    raise RuntimeError("TTS worker IPC request_id is required")
                if response_request_id != request_id:
                    raise RuntimeError("TTS worker IPC request_id does not match")
            if response_request_id and response_request_id != request_id:
                raise RuntimeError("TTS worker IPC request_id does not match")
            if event_type in {"status", "log", "audio"}:
                continue
            if event_type == "error":
                reason_code = str(message.get("reason_code") or "worker_error")[:64]
                raise RuntimeError(f"TTS worker IPC failed: {reason_code}")
            if event_type in expected:
                if not response_request_id:
                    raise RuntimeError("TTS worker IPC request_id is required")
                if response_request_id != request_id:
                    raise RuntimeError("TTS worker IPC request_id does not match")
                return dict(message)
        raise RuntimeError("TTS worker emitted too many IPC events")

    async def _invoke_ipc(
        self,
        command: str,
        *,
        request_id: str,
        expected: frozenset[str],
        fields: Mapping[str, object] | None = None,
        timeout: float | None = None,
    ) -> Mapping[str, object]:
        process = self._process
        wait_task = self._wait_task
        if process is None or wait_task is None or process.returncode is not None:
            raise RuntimeError("TTS worker is not running")
        operation_timeout = float(timeout or self.timeout_seconds)
        self._log_ipc(command, "started", request_id=request_id)
        try:
            await self._write_ipc_command(
                process,
                command,
                request_id=request_id,
                fields=fields,
            )
            message = await self._await_ipc_event(
                process,
                wait_task,
                request_id=request_id,
                expected=expected,
                timeout=operation_timeout,
            )
        except asyncio.CancelledError:
            self._log_ipc(
                command,
                "cancelled",
                request_id=request_id,
                reason_code="operation_cancelled",
                level=logging.INFO,
            )
            raise
        except (BrokenPipeError, ConnectionError, OSError, RuntimeError, TimeoutError) as exc:
            reason_code = (
                str(exc).rsplit(": ", 1)[-1]
                if str(exc).startswith("TTS worker IPC failed:")
                else "ipc_failed"
            )
            self._log_ipc(
                command,
                "failed",
                request_id=request_id,
                reason_code=reason_code,
                return_code=process.returncode,
                level=logging.WARNING,
            )
            raise
        self._log_ipc(command, "completed", request_id=request_id)
        event_type = str(message.get("type") or "").strip().lower()
        if event_type == "model_loaded":
            self._worker_model_loaded = bool(message.get("model_ready", False))
            mode = str(message.get("mode") or "").strip().lower()
            if mode in {"built_in", "external"}:
                self._worker_mode = mode
        return message

    async def health(self) -> EngineHealth:
        if self._closing or self._closed:
            return EngineHealth(self.engine_name, False, "worker backend is closed")
        configured = self._configuration_health()
        if not configured.available:
            return configured
        process = self._process
        if process is None:
            if self._startup_error:
                return EngineHealth(
                    self.engine_name,
                    False,
                    self._startup_health_message(),
                )
            return EngineHealth(self.engine_name, True, "persistent JSONL worker is configured")
        if process.returncode is None:
            if "health" not in self._worker_capabilities:
                return EngineHealth(self.engine_name, True, "persistent JSONL worker is running")
            operation_id = self._operation("health", process.pid)
            async with self._request_lock:
                try:
                    message = await self._invoke_ipc(
                        "health",
                        request_id=operation_id,
                        expected=frozenset({"health"}),
                        timeout=min(self.timeout_seconds, 2.0),
                    )
                except (OSError, RuntimeError, TimeoutError) as exc:
                    return EngineHealth(
                        self.engine_name,
                        False,
                        f"worker health IPC failed: {type(exc).__name__}",
                    )
            available = _protocol_bool(message.get("available", False), "available")
            mode = str(message.get("mode") or self._worker_mode)[:32]
            return EngineHealth(
                self.engine_name,
                available,
                f"persistent JSONL worker is running ({mode})",
            )
        return EngineHealth(
            self.engine_name,
            False,
            f"worker exited with status {process.returncode}",
        )

    def diagnostics(self) -> Mapping[str, object]:
        """返回不含命令、端点、路径和请求正文的 worker 状态。"""

        process = self._process
        running = process is not None and process.returncode is None
        ipc_ready = bool(running and self._worker_ready)
        return {
            "backend": self.engine_name,
            "mode": self._worker_mode,
            "model_ready": bool(running and self._worker_model_loaded),
            "ipc": {
                "status": "ready" if ipc_ready else "loading" if running else "stopped",
                "running": running,
                "ready": ipc_ready,
                "pid": int(process.pid) if running else None,
            },
            "capabilities": tuple(sorted(self._worker_capabilities)),
        }

    async def reload_models(self, *, gpt_path: str = "", sovits_path: str = "") -> EngineHealth:
        """通过 IPC 在空闲 worker 内加载明确的 GPT/SoVITS 权重。"""

        if "model_load" not in self._worker_capabilities:
            return EngineHealth(self.engine_name, False, "worker model-load IPC is unavailable")
        for path in (gpt_path, sovits_path):
            if len(str(path)) > 4096 or any(char in str(path) for char in "\x00\r\n"):
                raise ValueError("TTS model path is invalid")
        async with self._request_lock:
            operation_id = self._operation("load_model", f"{gpt_path}\0{sovits_path}")
            try:
                await self._invoke_ipc(
                    "load_model",
                    request_id=operation_id,
                    expected=frozenset({"model_loaded"}),
                    fields={"gpt_path": str(gpt_path), "sovits_path": str(sovits_path)},
                    timeout=self.startup_timeout_seconds,
                )
            except (OSError, RuntimeError, TimeoutError) as exc:
                return EngineHealth(
                    self.engine_name,
                    False,
                    f"worker model load failed: {type(exc).__name__}",
                )
        return EngineHealth(self.engine_name, True, "worker models loaded")

    def build_call(self, request: SpeechRequest) -> Mapping[str, object]:
        """把调用信息转换为 worker JSON；不包含命令、cwd 或其它启动配置。"""

        reference = _reference_profile(
            self.reference_audios,
            request=request,
            ref_dir=self.ref_dir,
            default_path=self.ref_audio_path,
            default_text=self.prompt_text,
            default_prompt_lang=self.prompt_lang,
        )
        options = _merge_options(
            self.default_options,
            request.options,
            {
                "text_language": reference["text_lang"],
                "prompt_language": reference["prompt_lang"],
                "ref_wav": reference["path"],
            },
        )
        return {
            "type": "synthesize",
            "protocol": _TTS_WORKER_PROTOCOL,
            "version": _TTS_WORKER_PROTOCOL_VERSION,
            "request_id": request.request_id,
            "text": request.text,
            "language": reference["text_lang"],
            "mood": request.mood,
            "voice": request.voice or request.profile_id,
            "reference_audio": reference["path"],
            "reference_text": reference["text"],
            "options": options,
            "streaming": True,
        }

    async def _running_process(
        self,
    ) -> tuple[asyncio.subprocess.Process, asyncio.Task[int]]:
        health = await self.start()
        if not health.available:
            raise RuntimeError(f"TTS worker is unavailable: {health.message}")
        process = self._process
        wait_task = self._wait_task
        if process is None or wait_task is None or process.returncode is not None:
            raise RuntimeError("TTS worker failed to stay running")
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("TTS worker standard streams are unavailable")
        return process, wait_task

    async def _dispose_process(
        self,
        process: asyncio.subprocess.Process,
        *,
        graceful: bool = False,
    ) -> None:
        """串行化同一 worker 的回收，避免流取消与后端关闭重复操作传输。"""

        async with self._dispose_lock:
            await self._dispose_process_unlocked(process, graceful=graceful)

    async def _dispose_process_unlocked(
        self,
        process: asyncio.subprocess.Process,
        *,
        graceful: bool = False,
    ) -> None:
        """关闭 stdin 后有界等待，超时则终止整个进程组。"""

        wait_task = self._wait_task if self._process is process else None
        stderr_task = self._stderr_task if self._process is process else None
        group_id = self._group_id if self._process is process else process_group_id(process)
        if process.stdin is not None and not process.stdin.is_closing():
            process.stdin.close()
            wait_closed = getattr(process.stdin, "wait_closed", None)
            if callable(wait_closed):
                try:
                    await asyncio.wait_for(
                        wait_closed(),
                        timeout=min(self.shutdown_timeout_seconds, 0.5),
                    )
                except (BrokenPipeError, ConnectionError, TimeoutError):
                    pass

        force = not graceful
        if graceful and process.returncode is None:
            try:
                waiter = wait_task if wait_task is not None else asyncio.create_task(process.wait())
                await asyncio.wait_for(
                    asyncio.shield(waiter),
                    timeout=self.shutdown_timeout_seconds,
                )
            except TimeoutError:
                force = True
        elif graceful and wait_task is not None and not wait_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(wait_task),
                    timeout=min(self.shutdown_timeout_seconds, 0.5),
                )
            except TimeoutError:
                force = True

        if force:
            await terminate_process_tree(
                process,
                timeout=min(self.shutdown_timeout_seconds, 0.5),
                group_id=group_id,
            )
        if wait_task is not None and not wait_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(wait_task), timeout=0.5)
            except TimeoutError:
                wait_task.cancel()
                await asyncio.gather(wait_task, return_exceptions=True)
        close_process_transport(process)
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            await asyncio.gather(stderr_task, return_exceptions=True)
        if self._process is process:
            if process.returncode is None and not _posix_process_gone(process):
                # 保留 process 引用供下一次 aclose 补偿；不能把仅关闭了
                # asyncio transport 的存活进程提交为 closed 终态。
                raise RuntimeError("TTS worker process did not exit during cleanup")
            self._process = None
            self._wait_task = None
            self._stderr_task = None
            self._group_id = None
            self._worker_capabilities = frozenset()
            self._worker_ready = False
            self._worker_mode = self._configured_worker_mode
            self._worker_model_loaded = False
            self._completed_request_ids.clear()
            self._completed_request_set.clear()

    async def _cancel_active_request(
        self,
        process: asyncio.subprocess.Process,
        wait_task: asyncio.Task[int],
        request_id: str,
    ) -> bool:
        """优先使用 cancel IPC 保留已加载模型；无确认时由调用方回收进程。"""

        if "cancel" not in self._worker_capabilities or process.returncode is not None:
            return False
        operation = asyncio.create_task(
            self._invoke_ipc(
                "cancel",
                request_id=request_id,
                expected=frozenset({"cancelled"}),
                timeout=min(self.shutdown_timeout_seconds, 1.0),
            )
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(operation),
                timeout=min(self.shutdown_timeout_seconds, 1.0) + 0.1,
            )
        except (asyncio.CancelledError, OSError, RuntimeError, TimeoutError):
            if not operation.done():
                operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
            return False
        return process.returncode is None

    async def stream(self, request: SpeechRequest) -> AsyncIterator[SpeechChunk]:
        if self._closing or self._closed:
            raise RuntimeError("TTS worker backend is closed")
        async with self._request_lock:
            if self._closing or self._closed:
                raise RuntimeError("TTS worker backend is closed")
            process, wait_task = await self._running_process()
            deadline = monotonic() + self.timeout_seconds
            failed = False
            cancelled = False
            completed = False
            disposed = False
            cancel_acknowledged = False
            final_seen = False
            audio_seen = False
            total_audio_bytes = 0
            chunk_count = 0
            stream_format: PcmFormat | None = None
            pcm_validator: StreamingPcmValidator | None = None
            started = monotonic()
            payload = self.build_call(request)
            operation = self._operation("synthesize", request.request_id)
            log_event(
                logger,
                "tts.request.start",
                component=self.log_component,
                status="started",
                correlation_id=request.request_id,
                fields={
                    "backend": self.engine_name,
                    "language": request.language,
                    "mood": request.mood,
                    "voice": request.voice or request.profile_id or "default",
                    "streaming": True,
                    "operation": operation,
                },
            )
            active_task = asyncio.current_task()
            self._active_stream_task = active_task
            self._active_request_id = request.request_id
            try:
                self._log_ipc("synthesize", "started", request_id=request.request_id)
                assert process.stdin is not None
                process.stdin.write(
                    (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                )
                await process.stdin.drain()
                while True:
                    try:
                        line = await _readline_until_process_exit(
                            process,
                            wait_task,
                            deadline=deadline,
                        )
                    except ValueError as exc:
                        raise RuntimeError("TTS worker line exceeds limit") from exc
                    if not line:
                        raise RuntimeError("TTS worker exited before a terminal stream event")
                    if len(line) > self.max_line_bytes:
                        raise RuntimeError("TTS worker line exceeds limit")
                    try:
                        parsed = parse_ipc_message(
                            line,
                            max_bytes=self.max_line_bytes,
                            allowed_types=TTS_IPC_SPEC.events,
                        )
                    except ValueError as exc:
                        raise RuntimeError("TTS worker returned an invalid event") from exc
                    message = parsed.payload
                    event_type = parsed.type
                    response_request_id = parsed.request_id
                    # 允许一个已完成请求的迟到 ``done``，它可能紧随
                    # ``audio(final=true)`` 出现在同一常驻 stdout 管道中。
                    # 其它请求 ID 或其它事件仍按协议错误处理，避免掩盖
                    # worker 串流错乱。
                    if (
                        event_type == "done"
                        and response_request_id in self._completed_request_set
                        and response_request_id != request.request_id
                    ):
                        continue
                    if event_type in {"audio", "done", "error", "cancelled"}:
                        if not response_request_id:
                            raise RuntimeError("TTS worker response request_id is required")
                        if response_request_id != request.request_id:
                            raise RuntimeError("TTS worker response request_id does not match")
                    elif response_request_id and response_request_id != request.request_id:
                        raise RuntimeError("TTS worker response request_id does not match")
                    if event_type == "error":
                        detail = str(message.get("message") or "").strip()
                        reason_code = str(message.get("reason_code") or "worker_error")[:64]
                        raise RuntimeError(
                            f"TTS worker reported an error: {reason_code}"
                            + (f" ({detail[:120]})" if detail else "")
                        )
                    if event_type == "cancelled":
                        raise RuntimeError("TTS worker cancelled an active synthesis")
                    if event_type in {"ready", "status", "log"}:
                        continue
                    if event_type == "done":
                        completed = True
                        final_seen = True
                        if self.require_audio and not audio_seen:
                            raise RuntimeError("TTS worker ended without audio data")
                        if stream_format is None:
                            stream_format = PcmFormat(24000, 1)
                        self._remember_completed_request(request.request_id)
                        yield SpeechChunk(
                            request.request_id,
                            b"",
                            stream_format.sample_rate,
                            stream_format.channels,
                            stream_format.sample_format,
                            is_final=True,
                        )
                        break
                    if event_type != "audio":
                        continue
                    if final_seen:
                        raise RuntimeError("TTS worker emitted audio after final chunk")
                    encoded = message.get("data", "")
                    if not isinstance(encoded, str):
                        raise RuntimeError("TTS worker audio data must be base64 text")
                    if len(encoded) > ((self.max_chunk_bytes + 2) * 4 // 3 + 8):
                        raise RuntimeError("TTS worker audio chunk exceeds limit")
                    try:
                        data = base64.b64decode(encoded, validate=True)
                    except (ValueError, binascii.Error) as exc:
                        raise RuntimeError("TTS worker audio is not valid base64") from exc
                    if len(data) > self.max_chunk_bytes:
                        raise RuntimeError("TTS worker audio chunk exceeds limit")
                    total_audio_bytes += len(data)
                    if total_audio_bytes > self.max_total_bytes:
                        raise RuntimeError("TTS worker audio exceeds the configured size limit")
                    try:
                        sample_rate = int(message.get("sample_rate", 24000))
                        channels = int(message.get("channels", 1))
                        sample_format = str(message.get("sample_format", "s16le"))
                        pcm_format = PcmFormat(sample_rate, channels, sample_format)
                    except (AudioProtocolError, TypeError, ValueError) as exc:
                        raise RuntimeError("TTS worker returned an invalid PCM format") from exc
                    if stream_format is not None and pcm_format != stream_format:
                        raise RuntimeError("TTS worker changed PCM format mid-stream")
                    stream_format = pcm_format
                    if self.validate_pcm and pcm_validator is None:
                        pcm_validator = StreamingPcmValidator(
                            pcm_format.sample_rate,
                            pcm_format.channels,
                            max_total_bytes=self.max_total_bytes,
                        )
                    if self.validate_pcm and pcm_validator is not None:
                        try:
                            pcm_validator.feed(data)
                        except AudioProtocolError as exc:
                            raise RuntimeError("TTS worker returned unaligned PCM audio") from exc
                    is_final = _protocol_bool(message.get("final", False), "final")
                    final_seen = is_final
                    audio_seen = audio_seen or bool(data)
                    chunk_count += 1
                    if is_final:
                        completed = True
                        self._remember_completed_request(request.request_id)
                    yield SpeechChunk(
                        request_id=request.request_id,
                        data=data,
                        sample_rate=pcm_format.sample_rate,
                        channels=pcm_format.channels,
                        sample_format=pcm_format.sample_format,
                        is_final=is_final,
                    )
                    if is_final:
                        break
                if self.require_final and not final_seen:
                    raise RuntimeError("TTS worker ended without a final audio chunk")
                if self.require_audio and not audio_seen:
                    raise RuntimeError("TTS worker ended without audio data")
                if self.validate_pcm and pcm_validator is not None:
                    try:
                        pcm_validator.finish()
                    except AudioProtocolError as exc:
                        raise RuntimeError("TTS worker returned incomplete PCM audio") from exc
                if final_seen and process.returncode is None:
                    await asyncio.wait((wait_task,), timeout=0.05)
                if process.returncode not in (0, None):
                    raise RuntimeError("TTS worker exited unsuccessfully")
                if wait_task.done() or process.returncode is not None:
                    await self._dispose_process(process, graceful=True)
                    disposed = True
                log_event(
                    logger,
                    "tts.request.complete",
                    component=self.log_component,
                    status="completed",
                    correlation_id=request.request_id,
                    duration_ms=(monotonic() - started) * 1000,
                    fields={
                        "backend": self.engine_name,
                        "chunks": chunk_count,
                        "bytes": total_audio_bytes,
                        "operation": operation,
                    },
                )
                self._log_ipc("synthesize", "completed", request_id=request.request_id)
            except asyncio.CancelledError:
                cancelled = True
                cancel_acknowledged = await self._cancel_active_request(
                    process,
                    wait_task,
                    request.request_id,
                )
                self._log_ipc(
                    "synthesize",
                    "cancelled",
                    request_id=request.request_id,
                    reason_code=("request_cancelled" if cancel_acknowledged else "worker_recycled"),
                    level=logging.INFO,
                )
                log_event(
                    logger,
                    "tts.request.cancel",
                    component=self.log_component,
                    status="cancelled",
                    correlation_id=request.request_id,
                    duration_ms=(monotonic() - started) * 1000,
                    fields={"backend": self.engine_name},
                )
                raise
            except GeneratorExit:
                cancelled = not final_seen
                if cancelled:
                    cancel_acknowledged = await self._cancel_active_request(
                        process,
                        wait_task,
                        request.request_id,
                    )
                    self._log_ipc(
                        "synthesize",
                        "cancelled",
                        request_id=request.request_id,
                        reason_code=(
                            "request_cancelled" if cancel_acknowledged else "worker_recycled"
                        ),
                        level=logging.INFO,
                    )
                    log_event(
                        logger,
                        "tts.request.cancel",
                        component=self.log_component,
                        status="cancelled",
                        correlation_id=request.request_id,
                        duration_ms=(monotonic() - started) * 1000,
                        fields={"backend": self.engine_name},
                    )
                raise
            except BaseException as exc:
                failed = True
                self._log_ipc(
                    "synthesize",
                    "failed",
                    request_id=request.request_id,
                    reason_code="synthesis_failed",
                    return_code=process.returncode,
                    level=logging.WARNING,
                )
                log_event(
                    logger,
                    "tts.request.failed",
                    component=self.log_component,
                    status="failed",
                    level=logging.WARNING,
                    correlation_id=request.request_id,
                    duration_ms=(monotonic() - started) * 1000,
                    fields={"backend": self.engine_name, "error": type(exc).__name__},
                )
                raise
            finally:
                try:
                    if disposed:
                        pass
                    elif failed or (cancelled and not cancel_acknowledged):
                        await self._dispose_process(process)
                    elif process.returncode is not None:
                        await self._dispose_process(process, graceful=True)
                    elif not completed and not cancel_acknowledged:
                        await self._dispose_process(process)
                finally:
                    if self._active_stream_task is active_task:
                        self._active_stream_task = None
                        self._active_request_id = ""

    def _close_wait_timeout(self) -> float:
        """返回关闭锁等待上限；推理超时不参与关闭预算。"""

        return min(max(self.shutdown_timeout_seconds + 0.25, 0.25), 3.0)

    async def _cancel_startup_calls(
        self,
        *,
        initiator: asyncio.Task[Any] | None,
    ) -> None:
        """取消仍在握手的启动调用，并有界等待其自身 finally 回收子进程。"""

        tasks = tuple(
            task for task in self._startup_tasks if task is not initiator and not task.done()
        )
        for task in tasks:
            task.cancel()
        if not tasks:
            return
        waiter = asyncio.gather(*tasks, return_exceptions=True)
        try:
            await asyncio.wait_for(
                asyncio.shield(waiter),
                timeout=self._close_wait_timeout(),
            )
        except TimeoutError:
            # gather 保留对任务的引用并消费其终态；后续强制进程回收不依赖
            # 第三方模型初始化协程是否及时响应取消。
            pass

    async def _close_process(self) -> None:
        """在请求与进程锁均已取得时请求正常卸载并回收 worker。"""

        process = self._process
        if process is None:
            return
        if "shutdown" in self._worker_capabilities and process.returncode is None:
            operation_id = self._operation("shutdown", process.pid)
            try:
                await self._invoke_ipc(
                    "shutdown",
                    request_id=operation_id,
                    expected=frozenset({"shutdown"}),
                    timeout=min(self.shutdown_timeout_seconds, self._close_wait_timeout()),
                )
            except (
                BrokenPipeError,
                ConnectionError,
                OSError,
                RuntimeError,
                TimeoutError,
            ):
                pass
        await self._dispose_process(process, graceful=True)

    async def _close_once(
        self,
        *,
        initiator: asyncio.Task[Any] | None,
    ) -> None:
        """执行一次可补偿的真实清理；只在传输已回收后提交关闭终态。"""

        started = monotonic()
        active_task = self._active_stream_task
        if active_task is not None and active_task is not initiator and not active_task.done():
            # stream 的 CancelledError 分支会先走 cancel IPC；未确认时其 finally
            # 会终止进程组，因此无需等待完整的推理超时。
            active_task.cancel()
        elif (
            active_task is initiator
            and self._active_request_id
            and "cancel" in self._worker_capabilities
        ):
            # 调用方可能在收到一个音频块后直接关闭后端，此时生成器仍持有
            # request lock，但当前任务已在生成器之外，不能取消调用方自身。
            process = self._process
            if process is not None and process.returncode is None:
                try:
                    await asyncio.wait_for(
                        self._write_ipc_command(
                            process,
                            "cancel",
                            request_id=self._active_request_id,
                        ),
                        timeout=min(self._close_wait_timeout(), 0.5),
                    )
                except (BrokenPipeError, ConnectionError, OSError, RuntimeError, TimeoutError):
                    pass

        request_lock_acquired = False
        try:
            try:
                await asyncio.wait_for(
                    self._request_lock.acquire(),
                    timeout=self._close_wait_timeout(),
                )
                request_lock_acquired = True
            except TimeoutError:
                await self._cancel_startup_calls(initiator=initiator)
                process = self._process
                if process is not None:
                    await self._dispose_process(process)
            else:
                process_lock_acquired = False
                try:
                    try:
                        await asyncio.wait_for(
                            self._process_lock.acquire(),
                            timeout=self._close_wait_timeout(),
                        )
                        process_lock_acquired = True
                    except TimeoutError:
                        await self._cancel_startup_calls(initiator=initiator)
                        await asyncio.wait_for(
                            self._process_lock.acquire(),
                            timeout=self._close_wait_timeout(),
                        )
                        process_lock_acquired = True
                    await self._close_process()
                finally:
                    if process_lock_acquired:
                        self._process_lock.release()

            # 取消与关闭 IPC 中的任何异常都不得留下未回收的 owned worker。
            remaining = self._process
            if remaining is not None:
                await self._dispose_process(remaining)
            if self._process is not None:
                raise RuntimeError("TTS worker cleanup did not release the process")
        except BaseException as exc:
            self._closed = False
            log_event(
                logger,
                "tts.backend.unload",
                component=self.log_component,
                status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
                level=logging.WARNING,
                duration_ms=(monotonic() - started) * 1000,
                reason_code=(
                    "cleanup_cancelled"
                    if isinstance(exc, asyncio.CancelledError)
                    else "cleanup_failed"
                ),
                fields={"backend": self.engine_name, "error_type": type(exc).__name__},
            )
            raise
        finally:
            if request_lock_acquired:
                self._request_lock.release()

        self._closed = True
        self._closing = False
        log_event(
            logger,
            "tts.backend.unload",
            component=self.log_component,
            status="completed",
            duration_ms=(monotonic() - started) * 1000,
            fields={"backend": self.engine_name},
        )

    @staticmethod
    def _consume_close_task(task: asyncio.Task[None]) -> None:
        """消费无人继续等待时的异常，同时保留 task 供下一次关闭判断。"""

        if not task.cancelled():
            task.exception()

    async def aclose(self) -> None:
        """共享真实清理任务；取消任一等待者不会中断 worker 回收。"""

        if self._closed:
            return
        close_task = self._close_task
        if close_task is None or close_task.done():
            if close_task is not None and not close_task.cancelled():
                close_task.exception()
            self._closing = True
            close_task = asyncio.create_task(
                self._close_once(initiator=asyncio.current_task()),
                name="meapet-tts-backend-close",
            )
            close_task.add_done_callback(self._consume_close_task)
            self._close_task = close_task
        await asyncio.shield(close_task)


class GptSovitsHttpBackend:
    """GPT-SoVITS api_v2 `/tts` 流式适配器。

    WAV 和裸 PCM 都通过增量校验器处理，网络分片边界不会影响 RIFF 块或音频帧
    的解析。服务协议未确认时，默认健康检查只验证配置；显式启用 ``health_probe``
    才会发起不触发推理的轻量 HTTP 探测。``expected_sample_rate`` 可用于把
    WAV 或裸 PCM 输出锁定到部署模型声明的采样率；留空时保持对其它合法 PCM
    采样率的兼容。
    """

    def __init__(
        self,
        endpoint: str = "",
        *,
        engine_root: str | Path = "",
        engine_config: str | Path = "GPT_SoVITS/configs/tts_infer.yaml",
        python_executable: str | Path = "",
        timeout_seconds: float = 60.0,
        ref_audio_path: str = "",
        prompt_text: str = "",
        prompt_lang: str = "",
        ref_dir: str | Path = "",
        reference_audios: Mapping[str, object] | None = None,
        options: Mapping[str, object] | None = None,
        media_type: str = "wav",
        streaming_mode: int = 3,
        speed_factor: float = 1.0,
        verify_tls: bool = True,
        raw_sample_rate: int = 24000,
        raw_channels: int = 1,
        expected_sample_rate: int | None = None,
        max_total_bytes: int = 64 * 1024 * 1024,
        health_probe: bool = False,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.endpoint = str(endpoint or "").strip()
        try:
            timeout = float(timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise ValueError("GPT-SoVITS timeout_seconds must be a number") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("GPT-SoVITS timeout_seconds must be positive and finite")
        self.timeout_seconds = timeout
        self.ref_audio_path = str(ref_audio_path or "").strip()
        self.prompt_text = str(prompt_text or "").strip()
        self.prompt_lang = str(prompt_lang or "").strip()
        self.ref_dir = str(ref_dir or "").strip()
        try:
            self.reference_audios = normalize_reference_audios(reference_audios)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        self.options = _gpt_sovits_http_options(options)
        self.media_type = str(media_type or "wav").strip().lower()
        # 配置中心历史上把裸音频显示为 ``pcm``；协议实现统一使用
        # ``raw``，这里保留明确别名，避免用户切换下拉项后保存失败。
        if self.media_type == "pcm":
            self.media_type = "raw"
        if self.media_type not in {"wav", "raw"}:
            raise ValueError("GPT-SoVITS media_type must be wav or raw")
        if isinstance(streaming_mode, bool):
            raise ValueError("GPT-SoVITS streaming_mode must be an integer from 0 to 3")
        try:
            normalized_streaming_mode = int(streaming_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("GPT-SoVITS streaming_mode must be an integer from 0 to 3") from exc
        if (
            normalized_streaming_mode not in {0, 1, 2, 3}
            or normalized_streaming_mode != streaming_mode
        ):
            raise ValueError("GPT-SoVITS streaming_mode must be an integer from 0 to 3")
        self.streaming_mode = normalized_streaming_mode
        try:
            speed = float(speed_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError("GPT-SoVITS speed_factor must be a number") from exc
        if not math.isfinite(speed) or not 0.1 <= speed <= 4.0:
            raise ValueError("GPT-SoVITS speed_factor must be between 0.1 and 4.0")
        self.speed_factor = speed
        self.verify_tls = _config_bool(verify_tls, "GPT-SoVITS verify_tls")
        if expected_sample_rate is not None:
            if isinstance(expected_sample_rate, bool):
                raise ValueError("GPT-SoVITS expected_sample_rate must be an integer")
            try:
                normalized_expected_sample_rate = int(expected_sample_rate)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("GPT-SoVITS expected_sample_rate must be an integer") from exc
            if normalized_expected_sample_rate != expected_sample_rate:
                raise ValueError("GPT-SoVITS expected_sample_rate must be an integer")
            try:
                PcmFormat(normalized_expected_sample_rate, 1)
            except (AudioProtocolError, TypeError, ValueError) as exc:
                raise ValueError("GPT-SoVITS expected_sample_rate is invalid") from exc
            self.expected_sample_rate: int | None = normalized_expected_sample_rate
        else:
            self.expected_sample_rate = None
        # 裸 PCM 没有容器元数据；显式 expected_sample_rate 必须同时成为
        # 播放标签，否则官方 32000 Hz 输出会被默认 24000 Hz 播放而变调。
        raw_rate = (
            self.expected_sample_rate
            if self.media_type == "raw" and self.expected_sample_rate is not None
            else raw_sample_rate
        )
        try:
            self.raw_format = PcmFormat(raw_rate, raw_channels)
        except (AudioProtocolError, TypeError, ValueError) as exc:
            raise ValueError("GPT-SoVITS raw PCM format is invalid") from exc
        self.max_total_bytes = max(1024, int(max_total_bytes))
        self.health_probe = _config_bool(health_probe, "GPT-SoVITS health_probe")
        self.client_factory = client_factory

    async def start(self) -> EngineHealth:
        """初始化 HTTP 适配器并执行配置指定的健康检查。"""

        health = await self.health()
        log_event(
            logger,
            "tts.backend.initialize",
            component="tts.gpt_sovits",
            status="ready" if health.available else "unavailable",
            level=logging.INFO if health.available else logging.WARNING,
            duration_ms=health.latency_ms,
            fields={"backend": "gpt-sovits", "available": health.available},
        )
        return health

    def build_method_call(self, request: SpeechRequest) -> SpeechMethodCall:
        """由适配器把纯调用信息转换为 GPT-SoVITS HTTP 方法调用。"""

        issue = self._endpoint_issue()
        if issue is not None:
            raise RuntimeError(f"GPT-SoVITS {issue}")
        reference = _reference_profile(
            self.reference_audios,
            request=request,
            ref_dir=self.ref_dir,
            default_path=self.ref_audio_path,
            default_text=self.prompt_text,
            default_prompt_lang=self.prompt_lang,
        )
        if not reference["path"]:
            raise RuntimeError("GPT-SoVITS reference audio path is required")
        payload = _merge_options(
            self.options,
            _gpt_sovits_http_options(request.options),
            {
                "text": request.text,
                "text_lang": reference["text_lang"],
                "ref_audio_path": reference["path"],
                "prompt_text": reference["text"],
                "prompt_lang": reference["prompt_lang"],
                "streaming_mode": self.streaming_mode,
                "media_type": self.media_type,
                "speed_factor": self.speed_factor,
            },
        )
        return SpeechMethodCall(
            request_id=request.request_id,
            method="POST",
            endpoint=self.endpoint,
            payload=payload,
            streaming=True,
        )

    def _endpoint_issue(self) -> str | None:
        if not self.endpoint:
            return "endpoint is not configured"
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "endpoint must be an http or https URL"
        if parsed.username is not None or parsed.password is not None:
            return "endpoint must not contain user information"
        return None

    def _client(self, httpx: Any, *, timeout_seconds: float | None = None) -> Any:
        timeout = httpx.Timeout(timeout_seconds or self.timeout_seconds)
        factory = self.client_factory or httpx.AsyncClient
        return factory(timeout=timeout, verify=self.verify_tls)

    async def health(self) -> EngineHealth:
        issue = self._endpoint_issue()
        if issue is not None:
            return EngineHealth("gpt-sovits", False, issue)
        if not self.health_probe:
            return EngineHealth("gpt-sovits", True, "endpoint is configured")
        return await self.probe_health()

    async def probe_health(self) -> EngineHealth:
        """执行一次轻量 HTTP 探测；不会发送合成文本或暴露参考音频。"""

        issue = self._endpoint_issue()
        if issue is not None:
            return EngineHealth("gpt-sovits", False, issue)
        try:
            import httpx
        except ImportError:
            return EngineHealth("gpt-sovits", False, "httpx is not installed")
        started = monotonic()
        probe_timeout = min(self.timeout_seconds, 5.0)
        try:
            # httpx 的传输超时无法约束自定义 client_factory；再加一层协程
            # 截止时间，避免配置中心探测被异常客户端永久挂起。
            async with asyncio.timeout(probe_timeout):
                async with self._client(httpx, timeout_seconds=probe_timeout) as client:
                    post = getattr(client, "post", None)
                    if callable(post):
                        # GPT-SoVITS api_v2 在缺少必填字段时会返回 400，
                        # 这已足以证明路由可达且不会触发推理。
                        response = await post(self.endpoint, json={})
                    else:
                        response = await client.get(self.endpoint)
            status_code = int(getattr(response, "status_code", 0))
            # 只接受成功或 api_v2/校验器已确认的缺参状态；鉴权失败、方法
            # 错误和路径错误都不能被报告为模型已就绪。
            available = 200 <= status_code < 300 or status_code in {400, 422}
            message = f"endpoint probe returned HTTP {status_code}"
            return EngineHealth(
                "gpt-sovits",
                available,
                message,
                latency_ms=(monotonic() - started) * 1000,
            )
        except TimeoutError:
            return EngineHealth(
                "gpt-sovits",
                False,
                "endpoint probe timed out",
                latency_ms=(monotonic() - started) * 1000,
            )
        except Exception as exc:
            return EngineHealth(
                "gpt-sovits",
                False,
                f"endpoint probe failed: {type(exc).__name__}",
                latency_ms=(monotonic() - started) * 1000,
            )

    async def stream(self, request: SpeechRequest) -> AsyncIterator[SpeechChunk]:
        call = self.build_method_call(request)
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for the GPT-SoVITS HTTP backend") from exc

        decoder: StreamingWavDecoder | None = None
        validator: StreamingPcmValidator | None = None
        if self.media_type == "wav":
            decoder = StreamingWavDecoder(
                max_total_bytes=self.max_total_bytes,
                expected_sample_rate=self.expected_sample_rate,
            )
        else:
            validator = StreamingPcmValidator(
                self.raw_format.sample_rate,
                self.raw_format.channels,
                max_total_bytes=self.max_total_bytes,
            )

        started = monotonic()
        chunk_count = 0
        total_audio_bytes = 0
        completed = False
        log_event(
            logger,
            "tts.request.start",
            component="tts.gpt_sovits",
            status="started",
            correlation_id=request.request_id,
            fields={
                "backend": "gpt-sovits",
                "language": request.language,
                "mood": request.mood,
                "voice": request.voice or request.profile_id or "default",
                "streaming": call.streaming,
            },
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                async with self._client(httpx) as client:
                    stream_kwargs: dict[str, object] = {"json": dict(call.payload)}
                    if call.headers:
                        stream_kwargs["headers"] = dict(call.headers)
                    async with client.stream(
                        call.method,
                        call.endpoint,
                        **stream_kwargs,
                    ) as response:
                        response.raise_for_status()
                        async for data in response.aiter_bytes():
                            if not data:
                                continue
                            try:
                                pieces = (
                                    decoder.feed(data)
                                    if decoder is not None
                                    else validator.feed(data)  # type: ignore[union-attr]
                                )
                            except AudioProtocolError as exc:
                                raise RuntimeError(
                                    f"GPT-SoVITS returned invalid {self.media_type} audio"
                                ) from exc
                            audio_format = (
                                decoder.format if decoder is not None else validator.format
                            )  # type: ignore[union-attr]
                            if audio_format is None:
                                continue
                            for piece in pieces:
                                chunk_count += 1
                                total_audio_bytes += len(piece)
                                yield SpeechChunk(
                                    request.request_id,
                                    piece,
                                    audio_format.sample_rate,
                                    audio_format.channels,
                                    audio_format.sample_format,
                                )
                        try:
                            audio_format = (
                                decoder.finish() if decoder is not None else validator.finish()  # type: ignore[union-attr]
                            )
                        except AudioProtocolError as exc:
                            raise RuntimeError(
                                f"GPT-SoVITS returned invalid {self.media_type} audio"
                            ) from exc
                        completed = True
                        yield SpeechChunk(
                            request.request_id,
                            b"",
                            audio_format.sample_rate,
                            audio_format.channels,
                            audio_format.sample_format,
                            is_final=True,
                        )
            log_event(
                logger,
                "tts.request.complete",
                component="tts.gpt_sovits",
                status="completed",
                correlation_id=request.request_id,
                duration_ms=(monotonic() - started) * 1000,
                fields={
                    "backend": "gpt-sovits",
                    "chunks": chunk_count,
                    "bytes": total_audio_bytes,
                },
            )
        except asyncio.CancelledError:
            log_event(
                logger,
                "tts.request.cancel",
                component="tts.gpt_sovits",
                status="cancelled",
                correlation_id=request.request_id,
                duration_ms=(monotonic() - started) * 1000,
                fields={"backend": "gpt-sovits"},
            )
            raise
        except TimeoutError as exc:
            log_event(
                logger,
                "tts.request.failed",
                component="tts.gpt_sovits",
                status="failed",
                level=logging.WARNING,
                correlation_id=request.request_id,
                duration_ms=(monotonic() - started) * 1000,
                fields={"backend": "gpt-sovits", "error": "TimeoutError"},
            )
            raise RuntimeError("GPT-SoVITS TTS request timed out") from exc
        except GeneratorExit:
            # 调用方在收到首个音频后主动关闭异步生成器是正常的流式
            # 取消，不应被记录为失败或触发错误级告警。
            log_event(
                logger,
                "tts.request.cancel",
                component="tts.gpt_sovits",
                status="cancelled",
                correlation_id=request.request_id,
                duration_ms=(monotonic() - started) * 1000,
                fields={"backend": "gpt-sovits", "reason": "generator_closed"},
            )
            raise
        except BaseException as exc:
            log_event(
                logger,
                "tts.request.failed",
                component="tts.gpt_sovits",
                status="failed",
                level=logging.WARNING,
                correlation_id=request.request_id,
                duration_ms=(monotonic() - started) * 1000,
                fields={"backend": "gpt-sovits", "error": type(exc).__name__},
            )
            raise
        finally:
            if not completed:
                log_event(
                    logger,
                    "tts.request.incomplete",
                    component="tts.gpt_sovits",
                    status="incomplete",
                    level=logging.DEBUG,
                    correlation_id=request.request_id,
                    fields={"backend": "gpt-sovits"},
                )

    async def aclose(self) -> None:
        """HTTP client 为逐请求上下文；卸载时只记录生命周期终点。"""

        log_event(
            logger,
            "tts.backend.unload",
            component="tts.gpt_sovits",
            status="completed",
            fields={"backend": "gpt-sovits"},
        )


class GptSovitsStdioBackend(SubprocessTTSBackend):
    """通过内置持久 JSONL worker 调用 GPT-SoVITS 流式接口。"""

    def __init__(
        self,
        endpoint: str = "",
        *,
        engine_root: str | Path = "",
        engine_config: str | Path = "GPT_SoVITS/configs/tts_infer.yaml",
        python_executable: str | Path = "",
        media_type: str = "wav",
        streaming_mode: int = 3,
        speed_factor: float = 1.0,
        verify_tls: bool = True,
        raw_sample_rate: int = 24000,
        raw_channels: int = 1,
        expected_sample_rate: int | None = 32000,
        health_probe: bool = True,
        startup_language: str = "zh",
        **kwargs: Any,
    ) -> None:
        root_text = str(engine_root or "").strip()
        self.direct_engine = bool(root_text)
        self.engine_root = str(Path(root_text).expanduser().resolve()) if root_text else ""
        self.engine_config = str(engine_config or "").strip()
        selected_python = str(python_executable or sys.executable).strip()
        if not selected_python:
            raise ValueError("GPT-SoVITS stdio python_executable is required")
        endpoint_text = str(endpoint or "").strip()
        verify_tls_enabled = _config_bool(verify_tls, "GPT-SoVITS stdio verify_tls")
        health_probe_enabled = _config_bool(health_probe, "GPT-SoVITS stdio health_probe")
        if not self.direct_engine:
            parsed = urlparse(endpoint_text)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError("GPT-SoVITS stdio endpoint must be an http or https URL")
        if isinstance(streaming_mode, bool) or streaming_mode not in {0, 1, 2, 3}:
            raise ValueError("GPT-SoVITS stdio streaming_mode must be an integer from 0 to 3")
        speed = float(speed_factor)
        if not math.isfinite(speed) or not 0.1 <= speed <= 4.0:
            raise ValueError("GPT-SoVITS stdio speed_factor must be between 0.1 and 4.0")
        selected_media = str(media_type or "").strip().lower()
        if selected_media == "pcm":
            selected_media = "raw"
        if selected_media not in {"wav", "raw"}:
            raise ValueError("GPT-SoVITS stdio media_type must be wav or raw")
        raw_format = PcmFormat(raw_sample_rate, raw_channels)
        language = canonical_tts_language(startup_language)
        if not language:
            raise ValueError("GPT-SoVITS stdio startup_language is invalid")
        self.startup_language = language
        default_options = kwargs.get("default_options")
        configured_options = default_options if isinstance(default_options, Mapping) else {}
        configured_gpt_path = str(configured_options.get("gpt_path", "") or "").strip()
        configured_sovits_path = str(configured_options.get("sovits_path", "") or "").strip()
        self.configured_gpt_path = configured_gpt_path
        self.configured_sovits_path = configured_sovits_path
        normalized_total_bytes = max(
            2,
            int(kwargs.get("max_total_bytes", 64 * 1024 * 1024)),
        )
        worker_script = Path(__file__).resolve().with_name("worker.py")
        command = [
            selected_python,
            str(worker_script),
            "--media-type",
            selected_media,
            "--streaming-mode",
            str(streaming_mode),
            "--speed-factor",
            str(speed),
            "--raw-sample-rate",
            str(raw_format.sample_rate),
            "--raw-channels",
            str(raw_format.channels),
            "--timeout-seconds",
            str(float(kwargs.get("timeout_seconds", 60.0))),
            "--max-total-bytes",
            str(normalized_total_bytes),
            "--shutdown-timeout-seconds",
            str(float(kwargs.get("shutdown_timeout_seconds", 2.0))),
        ]
        if self.direct_engine:
            command.extend(("--engine-root", self.engine_root))
            command.extend(("--engine-config", self.engine_config))
        else:
            command.extend(("--endpoint", endpoint_text))
        if expected_sample_rate is not None:
            expected = PcmFormat(int(expected_sample_rate), 1).sample_rate
            command.extend(("--expected-sample-rate", str(expected)))
        if configured_gpt_path:
            command.extend(("--gpt-path", configured_gpt_path))
        if configured_sovits_path:
            command.extend(("--sovits-path", configured_sovits_path))
        if not self.direct_engine and not verify_tls_enabled:
            command.append("--no-verify-tls")
        if not self.direct_engine and not health_probe_enabled:
            command.append("--skip-health-probe")
        super().__init__(
            tuple(command),
            require_ready=True,
            engine_name="gpt-sovits-stdio",
            log_component="tts.gpt_sovits_stdio",
            **kwargs,
        )
        self._configured_worker_mode = "built_in" if self.direct_engine else "external"
        self._worker_mode = self._configured_worker_mode

    def _configuration_health(self) -> EngineHealth:
        health = super()._configuration_health()
        if not health.available:
            return health
        if self.direct_engine and not Path(self.engine_root).is_dir():
            return EngineHealth(
                "gpt-sovits-stdio",
                False,
                "GPT-SoVITS engine root is unavailable",
            )
        if self.direct_engine:
            engine_config = Path(self.engine_config).expanduser()
            if not engine_config.is_absolute():
                engine_config = Path(self.engine_root) / engine_config
            if not engine_config.is_file():
                return EngineHealth(
                    "gpt-sovits-stdio",
                    False,
                    "GPT-SoVITS engine config is unavailable",
                )
            for label, model_path in (
                ("GPT", self.configured_gpt_path),
                ("SoVITS", self.configured_sovits_path),
            ):
                if model_path and not Path(model_path).is_file():
                    return EngineHealth(
                        "gpt-sovits-stdio",
                        False,
                        f"GPT-SoVITS {label} model is unavailable",
                    )
        try:
            reference = _reference_profile(
                self.reference_audios,
                request=SpeechRequest(
                    "startup-reference-check",
                    "health check",
                    language=self.startup_language,
                    mood="normal",
                ),
                ref_dir=self.ref_dir,
                default_path=self.ref_audio_path,
                default_text=self.prompt_text,
                default_prompt_lang=self.prompt_lang,
            )
            path = Path(reference["path"])
            if not path.is_file() or path.stat().st_size <= 0:
                raise OSError("reference audio is unavailable")
        except (OSError, RuntimeError, TypeError, ValueError):
            return EngineHealth(
                "gpt-sovits-stdio",
                False,
                f"reference audio is unavailable for language: {self.startup_language}",
            )
        return EngineHealth(
            "gpt-sovits-stdio",
            True,
            (
                "owned GPT-SoVITS model worker is configured"
                if self.direct_engine
                else "external GPT-SoVITS adapter worker is configured"
            ),
        )
