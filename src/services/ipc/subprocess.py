"""可由 TTS、ASR 等模型 worker 复用的有界 JSONL 子进程客户端。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from time import monotonic
from typing import Any

from core.ipc import (
    IPCProtocolSpec,
    JSONLMessage,
    build_jsonl_message,
    parse_jsonl_message,
    safe_ipc_id,
)
from logger.events import event_fingerprint, log_event
from services.processes import (
    close_process_transport,
    process_group_id,
    process_group_spawn_kwargs,
    terminate_process_tree,
)

logger = logging.getLogger(__name__)


class IPCProcessError(RuntimeError):
    """本地 IPC worker 没有完成明确的协议终态。"""

    def __init__(self, reason_code: str, message: str = "IPC worker operation failed") -> None:
        self.reason_code = str(reason_code or "ipc_failed")[:64]
        super().__init__(message)


def encode_jsonl_message(value: Mapping[str, object], *, max_line_bytes: int) -> bytes:
    """编码一行有界 JSON，拒绝会导致 worker 管道失控的超大消息。"""

    payload = (json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    if len(payload) > int(max_line_bytes):
        raise IPCProcessError("request_too_large", "IPC request exceeds line limit")
    return payload


def decode_jsonl_mapping(payload: bytes, *, max_line_bytes: int) -> dict[str, object]:
    """解码一行有界 JSON 对象，不解释具体模型协议。"""

    if len(payload) > int(max_line_bytes):
        raise IPCProcessError("response_too_large", "IPC response exceeds line limit")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IPCProcessError("invalid_json", "IPC response is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise IPCProcessError("invalid_envelope", "IPC response must be an object")
    return dict(value)


async def write_jsonl_message(
    stream: asyncio.StreamWriter | Any,
    value: Mapping[str, object],
    *,
    max_line_bytes: int,
) -> None:
    """向 asyncio stdin 写入并排空一条 JSONL 消息。"""

    if stream is None or stream.is_closing():
        raise IPCProcessError("stdin_unavailable", "IPC worker stdin is unavailable")
    stream.write(encode_jsonl_message(value, max_line_bytes=max_line_bytes))
    try:
        await stream.drain()
    except (BrokenPipeError, ConnectionError, OSError) as exc:
        raise IPCProcessError("stdin_failed", "IPC worker stdin write failed") from exc


async def drain_bounded_stderr(
    stream: asyncio.StreamReader | None,
    *,
    max_line_bytes: int,
    component: str,
    backend: str,
    target_logger: logging.Logger = logger,
    sink: Callable[[Mapping[str, object]], Awaitable[None] | None] | None = None,
) -> None:
    """持续排空 stderr，只传播不可逆指纹、字节数和截断状态。"""

    if stream is None:
        return
    while True:
        try:
            line = await stream.readline()
        except ValueError:
            log_event(
                target_logger,
                "ipc.worker.stderr",
                component=component,
                status="observed",
                level=logging.DEBUG,
                fields={
                    "backend": backend,
                    "stderr_bytes": max_line_bytes + 1,
                    "stderr_fingerprint": event_fingerprint("oversized-stderr-line"),
                    "stderr_lines": 1,
                    "truncated": True,
                },
            )
            continue
        except (OSError, RuntimeError):
            return
        if not line:
            return
        bounded = bytes(line[:max_line_bytes])
        summary: dict[str, object] = {
            "backend": backend,
            "stderr_bytes": len(line),
            "stderr_fingerprint": event_fingerprint(bounded),
            "stderr_lines": 1,
            "truncated": len(line) > len(bounded),
        }
        log_event(
            target_logger,
            "ipc.worker.stderr",
            component=component,
            status="observed",
            level=logging.DEBUG,
            fields=summary,
        )
        if sink is not None:
            result = sink(dict(summary))
            if inspect.isawaitable(result):
                await result


class JSONLSubprocessClient:
    """串行请求、显式握手且拥有子进程组的 JSONL IPC 客户端。"""

    def __init__(
        self,
        spec: IPCProtocolSpec,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        component: str = "ipc.worker",
        backend: str = "worker",
        max_line_bytes: int = 2 * 1024 * 1024,
        startup_timeout_seconds: float = 60.0,
        request_timeout_seconds: float = 60.0,
        shutdown_timeout_seconds: float = 2.0,
        ready_event: str = "ready",
        shutdown_command: str = "shutdown",
        shutdown_event: str = "shutdown",
        startup_passthrough_events: frozenset[str] = frozenset({"status", "log", "model_loaded"}),
    ) -> None:
        self.spec = spec
        self.command = tuple(str(item) for item in command)
        if not self.command or any(not item for item in self.command):
            raise ValueError("IPC worker command is required")
        self.cwd = str(cwd) if cwd is not None else None
        self.env = None if env is None else {str(key): str(value) for key, value in env.items()}
        self.component = str(component or "ipc.worker")
        self.backend = str(backend or "worker")
        self.max_line_bytes = max(1024, int(max_line_bytes))
        self.startup_timeout_seconds = _positive_timeout(
            startup_timeout_seconds, "startup_timeout_seconds"
        )
        self.request_timeout_seconds = _positive_timeout(
            request_timeout_seconds, "request_timeout_seconds"
        )
        self.shutdown_timeout_seconds = _positive_timeout(
            shutdown_timeout_seconds, "shutdown_timeout_seconds"
        )
        if ready_event not in spec.events:
            raise ValueError("IPC ready event is invalid")
        if shutdown_command not in spec.commands or shutdown_event not in spec.events:
            raise ValueError("IPC shutdown contract is invalid")
        self.ready_event = ready_event
        self.shutdown_command = shutdown_command
        self.shutdown_event = shutdown_event
        self.startup_passthrough_events = frozenset(startup_passthrough_events)
        self.startup_events: list[JSONLMessage] = []
        self._process: asyncio.subprocess.Process | None = None
        self._wait_task: asyncio.Task[int] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._group_id: int | None = None
        self._process_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._dispose_lock = asyncio.Lock()
        self._submissions: dict[str, asyncio.Event] = {}
        self._shutdown_task: asyncio.Task[None] | None = None
        self._disposing = False
        self._poisoned = False
        self._closed = False

    @property
    def process(self) -> asyncio.subprocess.Process | None:
        return self._process

    @property
    def running(self) -> bool:
        process = self._process
        return (
            not self._disposing
            and not self._poisoned
            and process is not None
            and process.returncode is None
        )

    def _configuration_error(self) -> IPCProcessError | None:
        executable = self.command[0]
        if any(separator in executable for separator in (os.sep, os.altsep) if separator):
            available = Path(executable).is_file() and os.access(executable, os.X_OK)
        else:
            available = shutil.which(executable) is not None
        if not available:
            return IPCProcessError("executable_unavailable")
        if self.cwd is not None and not Path(self.cwd).is_dir():
            return IPCProcessError("cwd_unavailable")
        return None

    async def start(self) -> JSONLMessage:
        """启动一次 worker，并等待带正确协议头的 ready 事件。"""

        async with self._process_lock:
            if self._closed:
                raise IPCProcessError("client_closed")
            if self.running:
                return JSONLMessage(
                    self.ready_event,
                    "",
                    {
                        "type": self.ready_event,
                        "protocol": self.spec.protocol,
                        "version": self.spec.version,
                    },
                )
            issue = self._configuration_error()
            if issue is not None:
                raise issue
            await self._dispose()
            log_event(
                logger,
                "ipc.worker.start",
                component=self.component,
                status="started",
                operation_id=f"{self.backend}:start",
                fields={"backend": self.backend},
            )
            try:
                process = await asyncio.create_subprocess_exec(
                    *self.command,
                    cwd=self.cwd,
                    env=self.env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=self.max_line_bytes + 1,
                    **process_group_spawn_kwargs(),
                )
            except (OSError, RuntimeError) as exc:
                log_event(
                    logger,
                    "ipc.worker.start",
                    component=self.component,
                    status="failed",
                    level=logging.WARNING,
                    operation_id=f"{self.backend}:start",
                    reason_code="spawn_failed",
                    fields={"backend": self.backend, "error_type": type(exc).__name__},
                )
                raise IPCProcessError("spawn_failed") from exc
            self._process = process
            self._wait_task = asyncio.create_task(process.wait())
            self._group_id = process_group_id(process)
            self._stderr_task = asyncio.create_task(
                drain_bounded_stderr(
                    process.stderr,
                    max_line_bytes=self.max_line_bytes,
                    component=self.component,
                    backend=self.backend,
                )
            )
            # shutdown 可能发生在 create_subprocess_exec 正在等待操作系统
            # 创建进程期间。关闭任务当时看不到 PID，因此新进程一旦出现
            # 必须由启动路径自行回收，不能重新进入 ready 握手。
            if self._closed:
                await self._dispose()
                raise IPCProcessError("client_closed")
            self.startup_events.clear()
            try:
                ready = await self._read_until(
                    request_id="",
                    expected=frozenset({self.ready_event}),
                    timeout=self.startup_timeout_seconds,
                    passthrough=self.startup_passthrough_events,
                )
            except asyncio.CancelledError:
                log_event(
                    logger,
                    "ipc.worker.start",
                    component=self.component,
                    status="cancelled",
                    level=logging.INFO,
                    operation_id=f"{self.backend}:start",
                    reason_code="startup_cancelled",
                    fields={"backend": self.backend},
                )
                await self._dispose()
                raise
            except BaseException as exc:
                reason_code = (
                    exc.reason_code if isinstance(exc, IPCProcessError) else "startup_failed"
                )
                log_event(
                    logger,
                    "ipc.worker.start",
                    component=self.component,
                    status="failed",
                    level=logging.WARNING,
                    operation_id=f"{self.backend}:start",
                    reason_code=reason_code,
                    fields={"backend": self.backend, "error_type": type(exc).__name__},
                )
                await self._dispose()
                raise
            log_event(
                logger,
                "ipc.worker.start",
                component=self.component,
                status="completed",
                operation_id=f"{self.backend}:start",
                fields={"backend": self.backend, "pid": process.pid},
            )
            return ready

    async def request(
        self,
        command: str,
        *,
        request_id: str,
        fields: Mapping[str, object] | None = None,
        expected_events: frozenset[str],
        timeout_seconds: float | None = None,
        passthrough_events: frozenset[str] | None = None,
    ) -> JSONLMessage:
        """串行发送命令并按 request_id 等待明确终态。"""

        normalized_command = str(command or "").strip().lower()
        if normalized_command not in self.spec.commands:
            raise ValueError("IPC request command is invalid")
        try:
            normalized_id = safe_ipc_id(request_id, required=True)
        except ValueError as exc:
            raise ValueError("IPC request_id is invalid") from exc
        try:
            expected = frozenset(expected_events)
            passthrough = (
                frozenset(item for item in ("status", "log") if item in self.spec.events)
                if passthrough_events is None
                else frozenset(passthrough_events)
            )
        except TypeError as exc:
            raise ValueError("IPC request event contract is invalid") from exc
        if not expected or not expected.issubset(self.spec.events):
            raise ValueError("IPC expected events are invalid")
        if not passthrough.issubset(self.spec.events):
            raise ValueError("IPC passthrough events are invalid")
        timeout = (
            self.request_timeout_seconds
            if timeout_seconds is None
            else _positive_timeout(timeout_seconds, "timeout_seconds")
        )
        if normalized_id in self._submissions:
            raise IPCProcessError("duplicate_request_id")
        submitted = asyncio.Event()
        self._submissions[normalized_id] = submitted
        operation_id = f"{self.backend}:{normalized_command}:{normalized_id}"
        log_event(
            logger,
            "ipc.request",
            component=self.component,
            status="started",
            operation_id=operation_id,
            request_id=normalized_id,
            fields={"backend": self.backend, "ipc_action": normalized_command},
        )
        try:
            await self.start()
            async with self._request_lock:
                if self._closed:
                    raise IPCProcessError("client_closed")
                if not self.running:
                    raise IPCProcessError("worker_unavailable")
                process = self._process
                assert process is not None
                async with self._write_lock:
                    # 从这里开始，即使 drain 被取消也可能已有部分或整行命令
                    # 进入 worker。先标记已提交，取消路径才能回收该 worker，
                    # 避免迟到终态污染下一次串行业务读取。
                    submitted.set()
                    await write_jsonl_message(
                        process.stdin,
                        build_jsonl_message(
                            self.spec,
                            normalized_command,
                            request_id=normalized_id,
                            fields=fields,
                        ),
                        max_line_bytes=self.max_line_bytes,
                    )
                result = await self._read_until(
                    request_id=normalized_id,
                    expected=expected,
                    timeout=timeout,
                    passthrough=passthrough,
                )
                terminal_status = "cancelled" if result.type == "cancelled" else "completed"
                log_event(
                    logger,
                    "ipc.request",
                    component=self.component,
                    status=terminal_status,
                    level=logging.WARNING if terminal_status == "cancelled" else logging.INFO,
                    operation_id=operation_id,
                    request_id=normalized_id,
                    reason_code="worker_cancelled" if terminal_status == "cancelled" else None,
                    fields={"backend": self.backend, "ipc_action": normalized_command},
                )
                return result
        except asyncio.CancelledError:
            log_event(
                logger,
                "ipc.request",
                component=self.component,
                status="cancelled",
                level=logging.WARNING,
                operation_id=operation_id,
                request_id=normalized_id,
                reason_code="request_cancelled",
                fields={"backend": self.backend, "ipc_action": normalized_command},
            )
            # 仍在串行锁队列中的请求没有接触 stdin，不得误杀另一个请求
            # 正在使用的共享 worker。已进入写入阶段则必须回收，确保迟到
            # 回执不能留在 stdout 中成为下一次请求的首条消息。
            if submitted.is_set():
                self._poisoned = True
                await self._dispose()
            raise
        except BaseException as exc:
            reason_code = exc.reason_code if isinstance(exc, IPCProcessError) else "request_failed"
            log_event(
                logger,
                "ipc.request",
                component=self.component,
                status="failed",
                level=logging.WARNING,
                operation_id=operation_id,
                request_id=normalized_id,
                reason_code=reason_code,
                fields={"backend": self.backend, "error_type": type(exc).__name__},
            )
            if submitted.is_set():
                self._poisoned = True
                await self._dispose()
            raise
        finally:
            self._submissions.pop(normalized_id, None)

    async def send_interrupt(
        self,
        command: str,
        *,
        request_id: str,
        fields: Mapping[str, object] | None = None,
    ) -> None:
        """并发写入 cancel 等中断命令；原 request 协程负责读取终态。"""

        normalized_command = str(command or "").strip().lower()
        if normalized_command not in self.spec.commands:
            raise ValueError("IPC interrupt command is invalid")
        try:
            normalized_id = safe_ipc_id(request_id, required=True)
        except ValueError as exc:
            raise ValueError("IPC request_id is invalid") from exc
        submitted = self._submissions.get(normalized_id)
        if submitted is None:
            raise IPCProcessError("request_not_submitted")
        try:
            await asyncio.wait_for(
                submitted.wait(),
                timeout=min(self.startup_timeout_seconds, self.request_timeout_seconds),
            )
        except TimeoutError as exc:
            raise IPCProcessError("request_not_submitted") from exc
        process = self._process
        if process is None or process.returncode is not None:
            raise IPCProcessError("worker_unavailable")
        async with self._write_lock:
            if process is not self._process or process.returncode is not None:
                raise IPCProcessError("worker_unavailable")
            await write_jsonl_message(
                process.stdin,
                build_jsonl_message(
                    self.spec,
                    normalized_command,
                    request_id=normalized_id,
                    fields=fields,
                ),
                max_line_bytes=self.max_line_bytes,
            )
        log_event(
            logger,
            "ipc.interrupt",
            component=self.component,
            status="submitted",
            operation_id=f"{self.backend}:{normalized_command}:{normalized_id}",
            request_id=normalized_id,
            fields={"backend": self.backend, "ipc_action": normalized_command},
        )

    async def _read_until(
        self,
        *,
        request_id: str,
        expected: frozenset[str],
        timeout: float,
        passthrough: frozenset[str],
    ) -> JSONLMessage:
        process = self._process
        if process is None or process.stdout is None:
            raise IPCProcessError("stdout_unavailable")
        deadline = monotonic() + timeout
        for _index in range(4096):
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise IPCProcessError("response_timeout")
            try:
                async with asyncio.timeout(remaining):
                    line = await process.stdout.readline()
            except TimeoutError as exc:
                raise IPCProcessError("response_timeout") from exc
            except ValueError as exc:
                raise IPCProcessError("response_too_large") from exc
            if not line:
                raise IPCProcessError("worker_exited")
            try:
                message = parse_jsonl_message(
                    self.spec,
                    line,
                    max_bytes=self.max_line_bytes,
                    allowed_types=self.spec.events,
                )
            except ValueError as exc:
                raise IPCProcessError("invalid_envelope") from exc
            if message.type == "error":
                if message.request_id != request_id:
                    raise IPCProcessError("request_id_mismatch")
                reason_code = str(message.payload.get("reason_code") or "worker_error")[:64]
                raise IPCProcessError(reason_code)
            if message.type in expected:
                # 业务终态必须携带并精确匹配 ID。启动握手以空字符串
                # 调用本方法，因此仍明确允许不带 request_id 的 ready。
                if message.request_id != request_id:
                    raise IPCProcessError("request_id_mismatch")
                return message
            if message.type in passthrough:
                if message.request_id and message.request_id != request_id:
                    raise IPCProcessError("request_id_mismatch")
                self.startup_events.append(message)
                continue
            raise IPCProcessError("unexpected_event")
        raise IPCProcessError("event_limit_exceeded")

    async def shutdown(self) -> None:
        """有界关闭 worker；取消调用方不会取消正在执行的实际回收。"""

        self._closed = True
        task = self._shutdown_task
        retry_cleanup = False
        if task is not None and task.done():
            if task.cancelled():
                retry_cleanup = True
            else:
                retry_cleanup = task.exception() is not None
        if task is None or (task.done() and (retry_cleanup or self._process is not None)):
            task = asyncio.create_task(
                self._shutdown_once(),
                name=f"meapet-ipc-shutdown-{self.backend}",
            )
            self._shutdown_task = task
        await asyncio.shield(task)

    async def _shutdown_once(self) -> None:
        """优先协议关闭；活动长请求存在时立即进入进程树回收。"""

        started_at = monotonic()
        operation_id = f"{self.backend}:shutdown"
        log_event(
            logger,
            "ipc.worker.shutdown",
            component=self.component,
            status="started",
            operation_id=operation_id,
            fields={"backend": self.backend},
        )
        graceful = False
        shutdown_lock_acquired = False
        reason_code = ""
        initial_process = self._process
        had_running_process = initial_process is not None and initial_process.returncode is None
        try:
            # 不在业务锁后等待完整请求超时。短窗口只用于让已排队但尚未
            # 写入的请求观察 client_closed 并退出；长推理直接回收进程树。
            lock_wait = min(0.05, self.shutdown_timeout_seconds)
            try:
                await asyncio.wait_for(self._request_lock.acquire(), timeout=lock_wait)
                shutdown_lock_acquired = True
            except TimeoutError:
                reason_code = "active_request"
            process = self._process
            if shutdown_lock_acquired and process is not None and process.returncode is None:
                request_id = event_fingerprint(f"{self.backend}:shutdown:{process.pid}")
                try:
                    async with self._write_lock:
                        await write_jsonl_message(
                            process.stdin,
                            build_jsonl_message(
                                self.spec,
                                self.shutdown_command,
                                request_id=request_id,
                            ),
                            max_line_bytes=self.max_line_bytes,
                        )
                    await self._read_until(
                        request_id=request_id,
                        expected=frozenset({self.shutdown_event}),
                        timeout=self.shutdown_timeout_seconds,
                        passthrough=frozenset(
                            item for item in ("status", "log") if item in self.spec.events
                        ),
                    )
                    graceful = True
                except (IPCProcessError, OSError, RuntimeError, TimeoutError) as exc:
                    reason_code = (
                        exc.reason_code if isinstance(exc, IPCProcessError) else "shutdown_failed"
                    )
            await self._dispose(graceful=graceful)
            log_event(
                logger,
                "ipc.worker.shutdown",
                component=self.component,
                status="completed",
                operation_id=operation_id,
                reason_code=reason_code or None,
                duration_ms=(monotonic() - started_at) * 1000.0,
                fields={
                    "backend": self.backend,
                    "shutdown_mode": (
                        "not_running"
                        if not had_running_process
                        else "graceful"
                        if graceful
                        else "forced"
                    ),
                },
            )
        except BaseException as exc:
            log_event(
                logger,
                "ipc.worker.shutdown",
                component=self.component,
                status="failed",
                level=logging.WARNING,
                operation_id=operation_id,
                reason_code="cleanup_failed",
                duration_ms=(monotonic() - started_at) * 1000.0,
                fields={"backend": self.backend, "error_type": type(exc).__name__},
            )
            raise
        finally:
            if shutdown_lock_acquired:
                self._request_lock.release()

    async def _dispose(self, *, graceful: bool = False) -> None:
        async with self._dispose_lock:
            process = self._process
            wait_task = self._wait_task
            stderr_task = self._stderr_task
            group_id = self._group_id
            if process is None:
                self._poisoned = False
                return
            # 串行化实际回收，但在清理完成前保留 PID/传输引用。若本次清理
            # 意外中止，后续 shutdown 才能看到同一进程并执行补偿回收。
            self._disposing = True
            self._poisoned = True
            cleanup_complete = False
            try:
                if process.stdin is not None and not process.stdin.is_closing():
                    try:
                        process.stdin.close()
                    except (BrokenPipeError, ConnectionError, OSError, RuntimeError):
                        pass
                if graceful and process.returncode is None:
                    try:
                        await asyncio.wait_for(
                            process.wait(), timeout=self.shutdown_timeout_seconds
                        )
                    except TimeoutError:
                        pass
                if process.returncode is None:
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
                if stderr_task is not None and not stderr_task.done():
                    stderr_task.cancel()
                    await asyncio.gather(stderr_task, return_exceptions=True)
                close_process_transport(process)
                if self._process is process:
                    self._process = None
                    self._wait_task = None
                    self._stderr_task = None
                    self._group_id = None
                cleanup_complete = True
            finally:
                self._disposing = False
                if cleanup_complete:
                    self._poisoned = False


def _positive_timeout(value: object, field_name: str) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"IPC {field_name} must be a number") from exc
    if not 0 < timeout <= 3600:
        raise ValueError(f"IPC {field_name} is outside the allowed range")
    return timeout


__all__ = [
    "IPCProcessError",
    "JSONLSubprocessClient",
    "decode_jsonl_mapping",
    "drain_bounded_stderr",
    "encode_jsonl_message",
    "write_jsonl_message",
]
