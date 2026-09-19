"""桌宠首批内置工具；所有桌面副作用仍经过统一权限服务。"""

from __future__ import annotations

import asyncio
import base64
import binascii
import csv
import inspect
import io
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Any

from core.adapters.providers import provider_registry
from core.events.types import ConversationContext
from core.rendering.actions import ExpressionRequest, MotionRequest
from services.processes import (
    close_process_transport,
    process_group_id,
    process_group_spawn_kwargs,
    terminate_process_tree,
)

from .command import (
    normalize_command_allowlist,
    normalize_command_argv,
    normalize_command_timeout,
)
from .registry import ToolRegistry
from .types import (
    BEHAVIOR_INTERNAL_METADATA_KEY,
    RiskLevel,
    ToolCallContext,
    ToolKind,
    ToolSpec,
)

_DESKTOP_READ_TIMEOUT_SECONDS = 5.0
_ACTION_PARAMETER_NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$"
_OCR_IMAGE_MAX_BYTES = 5 * 1024 * 1024
_OCR_TSV_MAX_BYTES = 2 * 1024 * 1024
_OCR_TEXT_MAX_CHARS = 8 * 1024
_OCR_MAX_BOXES = 256
_OCR_MAX_COORDINATE = 1_000_000
_OCR_PROCESS_TIMEOUT_SECONDS = 15.0
_OCR_PROBE_TIMEOUT_SECONDS = 2.0
_OCR_LANGUAGE_PATTERN = re.compile(r"^[A-Za-z0-9_.+-]{1,32}$")


def _probe_ocr_backend() -> tuple[str | None, str]:
    """探测固定的 Tesseract 可执行文件，不把路径或命令输出交给模型。"""

    executable = shutil.which("tesseract")
    if not executable:
        return None, "tesseract executable is unavailable"
    try:
        result = subprocess.run(
            (executable, "--version"),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=_OCR_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, "tesseract backend probe failed"
    if result.returncode != 0:
        return None, "tesseract backend probe failed"
    return executable, ""


def _parse_ocr_tsv(
    payload: bytes,
    *,
    width: int | None = None,
    height: int | None = None,
    origin_x: int = 0,
    origin_y: int = 0,
) -> dict[str, object]:
    """解析受限 TSV，只保留文字和可用于后续点击的安全坐标。"""

    if len(payload) > _OCR_TSV_MAX_BYTES:
        payload = payload[:_OCR_TSV_MAX_BYTES]
        truncated = True
    else:
        truncated = False
    try:
        source = payload.decode("utf-8", errors="replace")
    except (AttributeError, UnicodeError):
        return {"text": "", "boxes": [], "truncated": True}

    rows = csv.DictReader(io.StringIO(source), delimiter="\t")
    boxes: list[dict[str, object]] = []
    line_words: dict[tuple[str, str, str, str], list[str]] = {}
    line_order: list[tuple[str, str, str, str]] = []

    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("level", "")) != "5":
            continue
        raw_text = str(row.get("text", "") or "")
        word = " ".join(raw_text.split())
        if not word:
            continue
        word = word[:512]
        try:
            left = int(row.get("left", ""))
            top = int(row.get("top", ""))
            box_width = int(row.get("width", ""))
            box_height = int(row.get("height", ""))
        except (TypeError, ValueError, OverflowError):
            continue
        if (
            left < 0
            or top < 0
            or box_width <= 0
            or box_height <= 0
            or left > _OCR_MAX_COORDINATE
            or top > _OCR_MAX_COORDINATE
            or box_width > _OCR_MAX_COORDINATE
            or box_height > _OCR_MAX_COORDINATE
        ):
            continue
        if width is not None and (left >= width or left + box_width > width):
            continue
        if height is not None and (top >= height or top + box_height > height):
            continue
        if len(boxes) >= _OCR_MAX_BOXES:
            truncated = True
            break

        confidence: float | None = None
        try:
            parsed_confidence = float(row.get("conf", ""))
        except (TypeError, ValueError, OverflowError):
            parsed_confidence = -1.0
        if parsed_confidence >= 0.0:
            confidence = max(0.0, min(100.0, parsed_confidence))
        screen_x = left + origin_x
        screen_y = top + origin_y
        if (
            screen_x < -_OCR_MAX_COORDINATE
            or screen_x > _OCR_MAX_COORDINATE
            or screen_y < -_OCR_MAX_COORDINATE
            or screen_y > _OCR_MAX_COORDINATE
        ):
            continue
        box = {
            "x": screen_x,
            "y": screen_y,
            "width": box_width,
            "height": box_height,
            "text": word,
            "confidence": confidence,
        }
        boxes.append(box)
        line_key = tuple(
            str(row.get(name, "") or "")
            for name in ("page_num", "block_num", "par_num", "line_num")
        )
        if line_key not in line_words:
            line_words[line_key] = []
            line_order.append(line_key)
        line_words[line_key].append(word)

    text = "\n".join(" ".join(line_words[key]) for key in line_order)
    if len(text) > _OCR_TEXT_MAX_CHARS:
        text = text[:_OCR_TEXT_MAX_CHARS]
        truncated = True
    return {"text": text, "boxes": boxes, "truncated": truncated}


def _automation_step_schema() -> dict[str, object]:
    """返回受控桌面自动化单步契约；不允许透传 Win32 事件码。"""

    common = {
        "key": {"type": "string", "minLength": 1, "maxLength": 32},
        "modifiers": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "string",
                "enum": ["ctrl", "shift", "alt", "win"],
            },
        },
        "repeat": {"type": "integer", "minimum": 1, "maximum": 5},
        "text": {"type": "string", "minLength": 1, "maxLength": 512},
        "x": {"type": "integer", "minimum": -1_000_000, "maximum": 1_000_000},
        "y": {"type": "integer", "minimum": -1_000_000, "maximum": 1_000_000},
        "window_id": {
            "oneOf": [
                {"type": "integer", "minimum": 1, "maximum": 2**63 - 1},
                {"type": "string", "pattern": r"^0x[0-9a-fA-F]+$", "maxLength": 18},
            ]
        },
        "button": {"type": "string", "enum": ["left", "middle", "right"]},
        "duration_ms": {"type": "integer", "minimum": 0, "maximum": 2_000},
    }

    def branch(properties: tuple[str, ...], required: tuple[str, ...]) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {name: common[name] for name in properties},
            "required": ["type", *required],
            "additionalProperties": False,
        }

    common["type"] = {"type": "string"}
    branches = [
        branch(("type", "key", "modifiers", "repeat"), ("key",)),
        branch(("type", "text"), ("text",)),
        branch(("type", "x", "y", "button"), ("x", "y")),
        branch(("type", "x", "y"), ("x", "y")),
        branch(("type", "window_id"), ("window_id",)),
        branch(("type", "window_id", "x", "y"), ("window_id", "x", "y")),
        branch(("type", "duration_ms"), ("duration_ms",)),
    ]
    for item, step_type in zip(
        branches,
        (
            "key",
            "text",
            "click",
            "move_pointer",
            "activate_window",
            "move_window",
            "wait",
        ),
        strict=True,
    ):
        item["properties"]["type"] = {"type": "string", "enum": [step_type]}
    return {"oneOf": branches}


def _action_parameters_schema() -> dict[str, object]:
    """返回与渲染动作契约一致的有限数值参数 Schema。"""

    return {
        "type": "object",
        "maxProperties": 32,
        "propertyNames": {"pattern": _ACTION_PARAMETER_NAME_PATTERN},
        "additionalProperties": {
            "type": "number",
            "minimum": -10_000,
            "maximum": 10_000,
        },
    }


@dataclass(slots=True)
class _DesktopRead:
    """保存一个不可取消的同步桌面读取，避免取消后重复创建线程。"""

    call: Any
    done: Event = field(default_factory=Event)
    result: object = None
    error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = self.call()
        except BaseException as exc:
            self.error = exc
        finally:
            self.done.set()


class _DesktopReadGate:
    """每类只读桌面能力最多保留一个进行中的同步调用。"""

    def __init__(self, label: str) -> None:
        self._label = label
        self._lock = Lock()
        self._active: _DesktopRead | None = None

    async def invoke(self, call: Any) -> object:
        with self._lock:
            active = self._active
            if active is not None and not active.done.is_set():
                return {
                    "status": "degraded",
                    "reason": f"{self._label} call remains in flight",
                    "recoverable": True,
                }
            read = _DesktopRead(call)
            self._active = read
        thread = Thread(target=read.run, name=f"meapet-{self._label}-read", daemon=True)
        try:
            thread.start()
        except BaseException as exc:
            read.error = exc
            read.done.set()
        deadline = asyncio.get_running_loop().time() + _DESKTOP_READ_TIMEOUT_SECONDS
        while not read.done.is_set():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return {
                    "status": "degraded",
                    "reason": f"{self._label} timed out; call remains in flight",
                    "recoverable": True,
                }
            await asyncio.sleep(min(0.01, remaining))
        with self._lock:
            if self._active is read:
                self._active = None
        if read.error is not None:
            raise read.error
        result = read.result
        if inspect.isawaitable(result):
            return await asyncio.wait_for(result, timeout=_DESKTOP_READ_TIMEOUT_SECONDS)
        return result


def _call(target: object, method: str, *args: object, **kwargs: object) -> Any:
    handler = getattr(target, method, None)
    if not callable(handler):
        raise RuntimeError(f"desktop capability is unavailable: {method}")
    return handler(*args, **kwargs)


def _call_pet_method(
    target: object,
    method: str,
    *args: object,
    context: ToolCallContext,
    force_internal: bool = False,
    **kwargs: object,
) -> Any:
    """按工具上下文选择外部或行为内部的桌宠转发入口。"""

    metadata = context.metadata
    internal = force_internal or metadata.get(BEHAVIOR_INTERNAL_METADATA_KEY) is True
    if internal:
        internal_handler = getattr(target, f"{method}_internal", None)
        if callable(internal_handler):
            return internal_handler(*args, **kwargs)
    # 旧控制器只实现公开方法；没有内部入口时保留原始签名兼容。
    return _call(target, method, *args, **kwargs)


async def _pet_action_supported(target: object, kind: str, name: str) -> bool | None:
    """读取桌宠渲染器的可选动作探针。

    ``None`` 表示旧控制器没有能力探针，保留鸭子类型兼容；明确返回
    ``False`` 时在调用渲染器前拒绝，避免工具把未声明名称记录成成功。
    """

    checker = getattr(target, f"supports_{kind}", None)
    if not callable(checker):
        return None
    try:
        result = checker(name)
        if inspect.isawaitable(result):
            result = await result
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    if result is None:
        return None
    return bool(result)


async def _read_process_output(
    stream: asyncio.StreamReader | None,
    limit: int = 64 * 1024,
) -> bytes:
    """持续排空子进程管道，同时限制保留在内存中的输出。"""

    if stream is None:
        return b""
    buffer = bytearray()
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return bytes(buffer)
        if len(buffer) < limit:
            buffer.extend(chunk[: limit - len(buffer)])


async def _cancel_reader_tasks(*tasks: asyncio.Task[bytes]) -> None:
    """取消并回收 stdout/stderr 读取任务，避免管道任务悬挂。"""

    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _completed_output(task: asyncio.Task[bytes]) -> bytes:
    """只读取已经完成且返回 bytes 的管道任务结果。"""

    if not task.done() or task.cancelled():
        return b""
    try:
        result = task.result()
    except BaseException:
        return b""
    return result if isinstance(result, bytes) else b""


async def _terminate_process(
    process: asyncio.subprocess.Process, *, group_id: int | None = None
) -> None:
    """终止并回收子进程；正常取消不得留下命令继续运行。"""

    await terminate_process_tree(process, timeout=1.0, group_id=group_id)


async def _run_ocr_process(
    executable: str,
    image_data: bytes,
    language: str,
) -> tuple[str, bytes]:
    """把截图通过标准输入送入 Tesseract，并限制进程输出与生命周期。"""

    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "stdin",
            "stdout",
            "--psm",
            "6",
            "-l",
            language,
            "tsv",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_group_spawn_kwargs(),
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return "unavailable", b""

    group_id = process_group_id(process)
    stdout_task = asyncio.create_task(_read_process_output(process.stdout, _OCR_TSV_MAX_BYTES))
    stderr_task = asyncio.create_task(_read_process_output(process.stderr, 16 * 1024))
    timed_out = False
    stdout = b""
    try:
        if process.stdin is None:
            return "unavailable", b""
        try:
            process.stdin.write(image_data)
            await process.stdin.drain()
            process.stdin.close()
        except (BrokenPipeError, ConnectionError, OSError, RuntimeError):
            await _terminate_process(process, group_id=group_id)
            return "unavailable", b""
        try:
            await asyncio.wait_for(
                asyncio.shield(process.wait()), timeout=_OCR_PROCESS_TIMEOUT_SECONDS
            )
        except TimeoutError:
            timed_out = True
            await _terminate_process(process, group_id=group_id)
        try:
            stdout, _stderr = await asyncio.wait_for(
                asyncio.gather(stdout_task, stderr_task), timeout=1.0
            )
        except TimeoutError:
            timed_out = True
            await _terminate_process(process, group_id=group_id)
            await _cancel_reader_tasks(stdout_task, stderr_task)
            stdout = _completed_output(stdout_task)
    except asyncio.CancelledError:
        await _terminate_process(process, group_id=group_id)
        await _cancel_reader_tasks(stdout_task, stderr_task)
        raise
    finally:
        if process.returncode is None:
            await _terminate_process(process, group_id=group_id)
        await _cancel_reader_tasks(stdout_task, stderr_task)
        close_process_transport(process)
    if timed_out:
        return "timeout", b""
    if process.returncode != 0:
        return "unavailable", b""
    return "completed", stdout


def register_builtin_tools(
    registry: ToolRegistry,
    *,
    platform: object,
    pet_controller: object,
    scheduler: object | None = None,
    trigger_service: object | None = None,
    tts: object | None = None,
    asr: object | None = None,
    asr_provider: Callable[[], object | None] | None = None,
    presentation: object | None = None,
    module_status_provider: Callable[[], object] | None = None,
    diary_provider: Callable[[], object | None] | None = None,
    default_language: str = "zh",
    command_allowlist: tuple[str, ...] = (),
) -> None:
    """注册工具并保持平台对象只通过鸭子类型边界接入。"""

    foreground_gate = _DesktopReadGate("foreground-window")
    process_gate = _DesktopReadGate("process-list")
    capture_window_gate = _DesktopReadGate("capture-window")
    platform_probe_gate = _DesktopReadGate("platform-probe")
    configured_language = str(default_language or "zh").strip() or "zh"
    allowed_commands = normalize_command_allowlist(command_allowlist)

    def set_command_allowlist(values: Iterable[str]) -> None:
        """原子替换系统命令白名单，供运行期配置热重载使用。"""

        nonlocal allowed_commands
        allowed_commands = normalize_command_allowlist(values)

    def current_asr() -> object | None:
        """返回运行时当前 ASR，避免热替换后闭包继续持有旧 worker。"""

        return asr_provider() if callable(asr_provider) else asr

    def current_diary() -> object | None:
        """返回组合根当前的日记服务，避免热替换后持有旧仓储。"""

        return diary_provider() if callable(diary_provider) else None

    async def foreground(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        result = await foreground_gate.invoke(lambda: _call(platform, "foreground_window"))
        if inspect.isawaitable(result):
            result = await result
        return dict(result) if isinstance(result, Mapping) else {"status": "unknown"}

    async def cursor_position(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """读取全局光标位置，不触发输入或屏幕捕获副作用。"""

        del arguments, context
        result = await platform_probe_gate.invoke(lambda: _call(platform, "cursor_position"))
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Mapping):
            return dict(result)
        return {"status": "unavailable", "reason": "cursor position returned non-mapping"}

    async def processes(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        raw_limit = arguments.get("limit", 20)
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            return {"status": "denied", "reason": "process limit is invalid"}
        limit = raw_limit
        result = await process_gate.invoke(
            lambda: _call(platform, "list_processes", limit=max(1, min(limit, 100)))
        )
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Mapping):
            return dict(result)
        return {"processes": list(result or ())}

    async def context_snapshot(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """一次获取脱敏桌面上下文，不触发截图或输入副作用。"""

        del context
        raw_limit = arguments.get("process_limit", 20)
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            return {"status": "denied", "reason": "process limit is invalid"}
        limit = raw_limit
        safe_limit = max(1, min(limit, 100))

        async def read_idle() -> object:
            idle_getter = getattr(platform, "system_idle_seconds", None)
            if not callable(idle_getter):
                return None
            return await platform_probe_gate.invoke(idle_getter)

        foreground_result, process_result, idle_result, cursor_result = await asyncio.gather(
            foreground_gate.invoke(lambda: _call(platform, "foreground_window")),
            process_gate.invoke(lambda: _call(platform, "list_processes", limit=safe_limit)),
            read_idle(),
            platform_probe_gate.invoke(lambda: _call(platform, "cursor_position")),
            return_exceptions=True,
        )

        def mapping_result(value: object, label: str) -> dict[str, object]:
            if isinstance(value, asyncio.CancelledError):
                raise value
            if isinstance(value, Exception):
                return {"status": "unavailable", "reason": f"{label} read failed"}
            if inspect.isawaitable(value):
                raise RuntimeError(f"{label} read returned an awaitable")
            return dict(value) if isinstance(value, Mapping) else {"status": "unknown"}

        raw_foreground = mapping_result(foreground_result, "foreground window")
        raw_processes = mapping_result(process_result, "process list")
        raw_cursor = mapping_result(cursor_result, "cursor position")
        cursor_position: dict[str, object] = {
            "status": raw_cursor.get("status", "unknown"),
        }
        if str(raw_cursor.get("status", "")).strip().lower() == "available":
            raw_x = raw_cursor.get("x")
            raw_y = raw_cursor.get("y")
            if (
                isinstance(raw_x, int)
                and not isinstance(raw_x, bool)
                and isinstance(raw_y, int)
                and not isinstance(raw_y, bool)
            ):
                cursor_position.update({"x": int(raw_x), "y": int(raw_y)})
                for key in ("backend", "source"):
                    if key in raw_cursor:
                        cursor_position[key] = str(raw_cursor[key] or "")[:80]
            else:
                cursor_position = {
                    "status": "unavailable",
                    "reason": "cursor coordinates are invalid",
                }
        elif "reason" in raw_cursor:
            cursor_position["reason"] = str(raw_cursor["reason"] or "")[:160]
        foreground = {
            key: raw_foreground[key]
            for key in (
                "status",
                "backend",
                "window_id",
                "title",
                "app_id",
                "pid",
                "process_name",
                "geometry",
                "source",
                "redacted",
            )
            if key in raw_foreground
        }
        if "status" not in foreground:
            foreground["status"] = "unknown"

        process_rows = raw_processes.get("processes", ())
        safe_process_rows: list[dict[str, object]] = []
        if isinstance(process_rows, (list, tuple)):
            for row in process_rows[:safe_limit]:
                if not isinstance(row, Mapping):
                    continue
                safe_row = {key: row[key] for key in ("pid", "name") if key in row}
                if safe_row:
                    safe_process_rows.append(safe_row)
        processes = {
            "status": raw_processes.get("status", "unknown"),
            "processes": safe_process_rows,
        }
        if "reason" in raw_processes:
            processes["reason"] = str(raw_processes["reason"] or "")[:160]

        if isinstance(idle_result, asyncio.CancelledError):
            raise idle_result
        idle_seconds: float | None = None
        if isinstance(idle_result, (int, float)) and not isinstance(idle_result, bool):
            idle_seconds = max(0.0, float(idle_result))

        foreground_available = str(foreground.get("status", "")).strip().lower() == "available"
        processes_available = str(raw_processes.get("status", "")).strip().lower() == "available"
        if foreground_available and processes_available:
            status = "available"
        elif foreground_available or processes_available:
            status = "degraded"
        else:
            status = "unavailable"
        return {
            "status": status,
            "backend": str(getattr(platform, "backend", "unknown"))[:64],
            "foreground_window": foreground,
            "processes": processes,
            "cursor_position": cursor_position,
            "system_idle_seconds": idle_seconds,
        }

    async def capture(arguments: Mapping[str, Any], context: ToolCallContext) -> Mapping[str, Any]:
        scope = str(arguments.get("scope", "screen")).strip().lower()
        if scope not in {"screen", "window", "region"}:
            raise ValueError("capture scope is unsupported")
        capture_kwargs: dict[str, object] = {
            "scope": scope,
            "region": arguments.get("region"),
        }
        if scope == "window":
            resolver = getattr(platform, "resolve_capture_window_id", None)
            if callable(resolver):
                window_id = await capture_window_gate.invoke(resolver)
                if inspect.isawaitable(window_id):
                    window_id = await window_id
                if window_id is None:
                    return {
                        "status": "unavailable",
                        "reason": "active window id is unavailable",
                    }
                capture_kwargs["window_id"] = int(window_id)
        result = _call(platform, "capture_screen", **capture_kwargs)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            return {"status": "unavailable"}
        data = result.get("data", b"")
        if isinstance(data, bytes) and len(data) > 5 * 1024 * 1024:
            raise ValueError("capture exceeds the maximum size")
        captured = dict(result)
        # 窗口截图的像素原点是窗口左上角；补充几何后，OCR 坐标才能安全
        # 传给全局 click_at。读取失败时保留 capture 坐标，不伪造屏幕坐标。
        if scope == "window" and captured.get("status") == "available":
            read_window = getattr(platform, "active_window", None)
            if callable(read_window) and "origin" not in captured:
                try:
                    window = await capture_window_gate.invoke(read_window)
                    if inspect.isawaitable(window):
                        window = await window
                    geometry = getattr(window, "geometry", None)
                    if isinstance(geometry, (tuple, list)) and len(geometry) == 4:
                        raw_x, raw_y = geometry[0], geometry[1]
                        if (
                            isinstance(raw_x, int)
                            and not isinstance(raw_x, bool)
                            and isinstance(raw_y, int)
                            and not isinstance(raw_y, bool)
                        ):
                            captured["origin"] = {"x": raw_x, "y": raw_y}
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
        return captured

    async def ocr(arguments: Mapping[str, Any], context: ToolCallContext) -> Mapping[str, Any]:
        """捕获屏幕并用本地 OCR 识别文字；截图始终停留在进程边界内。"""

        raw_language = arguments.get("language", "eng")
        if not isinstance(raw_language, str):
            return {"status": "denied", "reason": "OCR language is invalid"}
        language = raw_language.strip()
        if _OCR_LANGUAGE_PATTERN.fullmatch(language) is None:
            return {"status": "denied", "reason": "OCR language is invalid"}
        executable, probe_reason = await asyncio.to_thread(_probe_ocr_backend)
        if executable is None:
            return {"status": "unavailable", "reason": probe_reason}

        capture_result = await capture(arguments, context)
        if not isinstance(capture_result, Mapping):
            return {"status": "unavailable", "reason": "screen capture is unavailable"}
        if str(capture_result.get("status", "")).strip().lower() != "available":
            return {"status": "unavailable", "reason": "screen capture is unavailable"}
        image_data = capture_result.get("data")
        if not isinstance(image_data, bytes) or not image_data:
            return {"status": "unavailable", "reason": "screen capture data is unavailable"}
        if len(image_data) > _OCR_IMAGE_MAX_BYTES:
            return {"status": "unavailable", "reason": "screen capture exceeds OCR size limit"}

        process_status, tsv = await _run_ocr_process(executable, image_data, language)
        if process_status == "timeout":
            return {"status": "timeout", "reason": "OCR backend timed out"}
        if process_status != "completed":
            return {"status": "unavailable", "reason": "OCR backend is unavailable"}

        width = capture_result.get("width")
        height = capture_result.get("height")
        image_width = int(width) if isinstance(width, int) and not isinstance(width, bool) else None
        image_height = (
            int(height) if isinstance(height, int) and not isinstance(height, bool) else None
        )
        origin_x = 0
        origin_y = 0
        coordinate_space = "capture"
        scope = str(arguments.get("scope", "screen")).strip().lower()
        region = arguments.get("region")
        if scope == "region" and isinstance(region, Mapping):
            raw_x = region.get("x")
            raw_y = region.get("y")
            if isinstance(raw_x, int) and not isinstance(raw_x, bool):
                origin_x = raw_x
            if isinstance(raw_y, int) and not isinstance(raw_y, bool):
                origin_y = raw_y
            coordinate_space = "screen"
        elif scope == "window":
            window_origin = capture_result.get("origin")
            if isinstance(window_origin, Mapping):
                raw_x = window_origin.get("x")
                raw_y = window_origin.get("y")
                if (
                    isinstance(raw_x, int)
                    and not isinstance(raw_x, bool)
                    and isinstance(raw_y, int)
                    and not isinstance(raw_y, bool)
                ):
                    origin_x = raw_x
                    origin_y = raw_y
                    coordinate_space = "screen"
        parsed = _parse_ocr_tsv(
            tsv,
            width=image_width,
            height=image_height,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        return {
            "status": "completed",
            "scope": scope,
            "language": language,
            "coordinate_space": coordinate_space,
            "origin": {"x": origin_x, "y": origin_y},
            "width": image_width,
            "height": image_height,
            **parsed,
        }

    async def click_at(arguments: Mapping[str, Any], context: ToolCallContext) -> Mapping[str, Any]:
        raw_x = arguments.get("x")
        raw_y = arguments.get("y")
        if (
            isinstance(raw_x, bool)
            or isinstance(raw_y, bool)
            or not isinstance(raw_x, int)
            or not isinstance(raw_y, int)
        ):
            return {"status": "denied", "reason": "click coordinates are invalid"}
        result = _call(
            platform,
            "click_at",
            raw_x,
            raw_y,
            button=arguments.get("button", "left"),
        )
        if inspect.isawaitable(result):
            result = await result
        return dict(result) if isinstance(result, Mapping) else {"status": "unavailable"}

    async def automation_batch(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """将完整自动化序列作为一次高风险、可去重的工具调用执行。"""

        del context
        steps = arguments.get("steps")
        # Windows 的等待步骤和 SendInput 均不依赖 Qt 对象，放到工作线程，
        # 防止有限的用户可见等待阻塞运行时事件循环或控制台重绘。
        result = await asyncio.to_thread(_call, platform, "automation_batch", steps)
        if inspect.isawaitable(result):
            result = await result
        return dict(result) if isinstance(result, Mapping) else {"status": "unavailable"}

    async def module_status(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        del arguments, context
        snapshot = await platform_probe_gate.invoke(lambda: _call(platform, "probe"))
        if inspect.isawaitable(snapshot):
            snapshot = await snapshot
        ocr_executable, ocr_reason = await asyncio.to_thread(_probe_ocr_backend)
        ocr_status: dict[str, object] = {
            "status": "available" if ocr_executable is not None else "unavailable",
            "available": ocr_executable is not None,
            "backend": "tesseract" if ocr_executable is not None else "unavailable",
        }
        if ocr_reason:
            ocr_status["reason"] = ocr_reason
        platform_status: dict[str, object] = {
            "backend": str(getattr(platform, "backend", "unknown"))[:64],
            "capabilities": (),
        }
        capabilities = getattr(snapshot, "capabilities", ())
        if not isinstance(capabilities, (str, bytes, bytearray, Mapping)):
            platform_status["capabilities"] = tuple(
                {
                    "name": str(getattr(item, "name", ""))[:64],
                    "state": str(getattr(item, "state", "unknown"))[:32],
                }
                for item in capabilities
                if str(getattr(item, "name", "")).strip()
            )[:64]

        runtime_status: Mapping[str, object] = {}
        if callable(module_status_provider):
            try:
                value = module_status_provider()
                if inspect.isawaitable(value):
                    value = await value
                if isinstance(value, Mapping):
                    runtime_status = dict(value)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                runtime_status = {"status": "unavailable"}

        runtime_tts_value = runtime_status.get("tts")
        runtime_tts = runtime_tts_value if isinstance(runtime_tts_value, Mapping) else None
        tts_status: dict[str, object] = {"available": False, "engine": "unconfigured"}
        health = getattr(tts, "health", None) if tts is not None else None
        has_active_tasks = getattr(tts, "has_active_tasks", None) if tts is not None else None
        tts_busy = bool(callable(has_active_tasks) and has_active_tasks())
        if tts_busy:
            backend = getattr(tts, "backend", None)
            tts_status = {
                "status": "busy",
                "available": True,
                "engine": str(getattr(backend, "engine_name", "busy") or "busy")[:128],
                "message": "TTS is busy",
            }
        elif runtime_tts is not None:
            runtime_health_value = runtime_tts.get("health")
            runtime_health = (
                runtime_health_value if isinstance(runtime_health_value, Mapping) else {}
            )
            tts_status = {
                "status": str(
                    runtime_health.get("status", runtime_tts.get("status", "unknown")) or "unknown"
                )[:32],
                "available": bool(
                    runtime_health.get("available", runtime_tts.get("available", False))
                ),
                "engine": str(
                    runtime_health.get("engine", runtime_tts.get("backend", "unknown")) or "unknown"
                )[:128],
                "latency_ms": runtime_health.get("latency_ms"),
            }
        elif callable(health):
            try:
                value = health()
                if inspect.isawaitable(value):
                    value = await value
                tts_status = {
                    "available": bool(getattr(value, "available", False)),
                    "engine": str(getattr(value, "engine", "unknown"))[:128],
                    "message": str(getattr(value, "message", ""))[:256],
                    "latency_ms": getattr(value, "latency_ms", None),
                }
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                tts_status = {
                    "available": False,
                    "engine": "unknown",
                    "message": "TTS health probe failed",
                }
        asr_status: Mapping[str, object] = {
            "status": "unconfigured",
            "available": False,
            "ready": False,
        }
        active_asr = current_asr()
        asr_diagnostics = (
            getattr(active_asr, "diagnostics", None) if active_asr is not None else None
        )
        if callable(asr_diagnostics):
            try:
                value = asr_diagnostics()
                if isinstance(value, Mapping):
                    asr_status = dict(value)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                asr_status = {"status": "unavailable", "available": False, "ready": False}
        return {
            "status": "available",
            "platform": platform_status,
            "ocr": ocr_status,
            "tts": tts_status,
            "asr": asr_status,
            "adapters": provider_registry.diagnostics(),
            "tools": registry.identities(),
            "runtime": runtime_status,
        }

    async def transcribe_audio(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """转写调用方提供的有界字节，不允许把用户文件路径交给 worker。"""

        del context
        active_asr = current_asr()
        if active_asr is None:
            return {"status": "unavailable", "reason": "ASR is not configured"}
        encoded = arguments.get("audio_base64")
        if not isinstance(encoded, str) or not encoded:
            return {"status": "denied", "reason": "audio_base64 is invalid"}
        byte_limit = int(getattr(active_asr, "max_audio_bytes", 0) or 0)
        encoded_limit = ((byte_limit + 2) // 3) * 4 if byte_limit > 0 else 0
        if encoded_limit <= 0 or len(encoded) > encoded_limit:
            return {"status": "denied", "reason": "audio exceeds configured byte limit"}
        try:
            audio = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (UnicodeEncodeError, binascii.Error, ValueError):
            return {"status": "denied", "reason": "audio_base64 is invalid"}
        if not audio or len(audio) > byte_limit:
            return {"status": "denied", "reason": "audio exceeds configured byte limit"}
        transcribe = getattr(active_asr, "transcribe", None)
        if not callable(transcribe):
            return {"status": "unavailable", "reason": "ASR transcribe is unavailable"}
        try:
            result = transcribe(
                audio,
                audio_format=str(arguments.get("audio_format", "wav")),
                sample_rate=int(arguments.get("sample_rate", 16000)),
                channels=int(arguments.get("channels", 1)),
                language=str(arguments.get("language") or getattr(active_asr, "language", "auto")),
            )
            if inspect.isawaitable(result):
                result = await result
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return {
                "status": "unavailable",
                "reason": f"ASR transcription failed: {type(exc).__name__}",
            }
        public = getattr(result, "public", None)
        if callable(public):
            value = public()
        elif isinstance(result, Mapping):
            value = result
        else:
            return {"status": "unavailable", "reason": "ASR result is invalid"}
        if not isinstance(value, Mapping):
            return {"status": "unavailable", "reason": "ASR result is invalid"}
        confidence_value = value.get("confidence")
        confidence = (
            float(confidence_value)
            if isinstance(confidence_value, (int, float)) and not isinstance(confidence_value, bool)
            else None
        )
        duration_value = value.get("duration_ms", 0)
        duration_ms = (
            int(duration_value)
            if isinstance(duration_value, (int, float)) and not isinstance(duration_value, bool)
            else 0
        )
        return {
            "status": "completed",
            "request_id": str(value.get("request_id") or "")[:256],
            "text": str(value.get("text") or ""),
            "language": str(value.get("language") or "unknown")[:16],
            "confidence": confidence,
            "confidence_available": bool(value.get("confidence_available", False)),
            "duration_ms": max(0, duration_ms),
        }

    async def move(
        arguments: Mapping[str, Any],
        context: ToolCallContext,
        *,
        force_internal: bool = False,
    ) -> Mapping[str, Any]:
        result = _call_pet_method(
            pet_controller,
            "move_to",
            float(arguments["x"]),
            float(arguments["y"]),
            duration_ms=int(arguments.get("duration_ms", 800)),
            context=context,
            force_internal=force_internal,
        )
        if inspect.isawaitable(result):
            result = await result
        return dict(result) if isinstance(result, Mapping) else {"status": "moved"}

    async def autonomous_move(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """自主游走专用移动入口。

        该入口不向模型公开，风险级别为低；行为服务已经在每个插值点做了
        边界约束和用户交互检查。模型需要移动时仍必须调用 ``pet:move``，
        继续遵守中风险审批策略。
        """

        return await move(arguments, context, force_internal=True)

    async def list_live2d_models(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """列出当前资源树可用 Live2D 模型，只暴露资源相对键。"""

        del arguments, context
        available = _call(pet_controller, "available_models")
        if inspect.isawaitable(available):
            available = await available
        if not isinstance(available, (list, tuple)):
            return {"status": "unavailable", "reason": "模型目录暂不可用"}
        models: list[str] = []
        for item in available:
            text = str(item or "").strip()
            if text and text not in models:
                models.append(text)
        return {
            "status": "completed",
            "models": models,
            "count": len(models),
        }

    async def switch_live2d_model(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """切换当前 Live2D 模型；回显只包含资源相对键与状态。"""

        del context
        model_key = str(arguments.get("model_key", "") or "").strip()
        if not model_key or len(model_key) > 512 or any(char in model_key for char in "\r\n\x00"):
            return {"status": "failed", "reason": "模型键不可用"}
        result = _call(pet_controller, "switch_live2d_model", model_key)
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            return {"status": "failed", "reason": "模型切换回执不可用"}
        safe: dict[str, object] = {
            "status": str(result.get("status", "unavailable") or "unavailable"),
            "model_key": model_key,
        }
        # 宿主回执可能携带绝对路径；只透传状态与确认后的资源键。
        reason = result.get("reason")
        if isinstance(reason, str) and reason.strip():
            safe["reason"] = reason[:256]
        if str(safe.get("status", "")).lower() not in {
            "unavailable",
            "failed",
            "denied",
        }:
            safe["persistence_status"] = str(
                result.get("persistence_status", "pending") or "pending"
            )
        return safe

    async def expression(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        try:
            request = ExpressionRequest.from_arguments(arguments)
        except (TypeError, ValueError):
            return {"status": "denied", "reason": "表情请求参数无效"}
        for layer in request.expressions:
            supported = await _pet_action_supported(pet_controller, "expression", layer.name)
            if supported is False:
                return {"status": "unavailable", "reason": "当前渲染器不支持此表情"}
        extended = any(key != "name" for key in arguments)
        result = _call_pet_method(
            pet_controller,
            "set_expression_request" if extended else "set_expression",
            request if extended else request.expressions[0].name,
            context=context,
        )
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Mapping):
            return dict(result)
        return {
            "status": "updated" if result is not False else "unavailable",
            "name": request.expressions[0].name,
            "expression_count": len(request.expressions),
        }

    async def motion(arguments: Mapping[str, Any], context: ToolCallContext) -> Mapping[str, Any]:
        try:
            request = MotionRequest.from_arguments(arguments)
        except (TypeError, ValueError):
            return {"status": "denied", "reason": "动作请求参数无效"}
        supported = await _pet_action_supported(pet_controller, "motion", request.name)
        if supported is False:
            return {"status": "unavailable", "reason": "当前渲染器不支持此动作"}
        extended = any(key != "name" for key in arguments)
        result = _call_pet_method(
            pet_controller,
            "play_motion_request" if extended else "play_motion",
            request if extended else request.name,
            context=context,
        )
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Mapping):
            return dict(result)
        return {
            "status": "started" if result is not False else "unavailable",
            "name": request.name,
        }

    async def speak(arguments: Mapping[str, Any], context: ToolCallContext) -> Mapping[str, Any]:
        text = str(arguments["text"]).strip()
        if not text or len(text) > 2000:
            raise ValueError("speech text is invalid")
        mood = str(arguments.get("mood", "neutral"))
        silent = bool(arguments.get("silent", False))
        visual_result = _call(pet_controller, "speak", text, mood=mood)
        if inspect.isawaitable(visual_result):
            visual_result = await visual_result
        rendered = True
        if isinstance(visual_result, Mapping):
            rendered = str(visual_result.get("status", "")).strip().lower() not in {
                "unavailable",
                "denied",
                "failed",
            }
        elif visual_result is False:
            rendered = False
        set_direct_speech = getattr(presentation, "set_direct_speech", None)
        if callable(set_direct_speech):
            try:
                set_direct_speech(text, mood=mood)
            except Exception:
                pass
        if tts is None:
            return {
                "status": "unavailable",
                "text": text,
                "rendered": rendered,
                "segments": [],
            }
        metadata = context.metadata
        mode = str(metadata.get("mode", "direct")).strip().lower()
        if mode not in {"direct", "agent"}:
            mode = "direct"
        try:
            generation_id = max(0, int(metadata.get("generation_id", 0)))
        except (TypeError, ValueError):
            generation_id = 0
        voice = str(arguments.get("voice", "") or "").strip()
        role = str(arguments.get("role", "") or "").strip()
        if len(role) > 128 or any(char in role for char in "\r\n\x00"):
            return {
                "status": "unavailable",
                "text": text,
                "rendered": rendered,
                "reason": "requested TTS role is invalid",
                "segments": [],
            }
        if voice:
            backend = getattr(tts, "backend", None)
            profile_ids = getattr(backend, "profile_ids", None)
            if not callable(profile_ids) or voice not in tuple(profile_ids()):
                return {
                    "status": "unavailable",
                    "text": text,
                    "rendered": rendered,
                    "reason": "requested TTS voice is unavailable",
                    "segments": [],
                }
        speech_context = ConversationContext(
            context.profile_id,
            context.session_id,
            context.turn_id,
            generation_id,
            mode,
        )
        if silent:
            # 模型显式要求本次不发声：正文由视觉/气泡展示，语音队列跳过。
            return {"status": "silent", "rendered": rendered, "segments": []}
        result = _call(
            tts,
            "enqueue_text",
            speech_context,
            text,
            language=str(arguments.get("language") or configured_language),
            mood=mood,
            profile_id=voice,
            voice=voice,
            role=role,
            flush=True,
        )
        if inspect.isawaitable(result):
            result = await result
        return {
            "status": "queued",
            "rendered": rendered,
            "segments": list(result) if isinstance(result, (list, tuple)) else [],
        }

    async def set_click_through(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        enabled = bool(arguments["enabled"])
        result = _call(pet_controller, "set_click_through", enabled)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Mapping):
            return dict(result)
        state = str(getattr(getattr(result, "state", None), "value", "unavailable"))
        return {"status": state, "enabled": enabled}

    async def diary_write(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """写入独立日记；工具结果永远不回显正文。"""

        diary = current_diary()
        open_session = getattr(diary, "open_internal_session", None)
        if not callable(open_session):
            return {"status": "unavailable", "reason": "pet diary is unavailable"}
        actor = f"model:{context.profile_id}"[:128]
        session = open_session(actor)
        create = getattr(session, "create", None)
        if not callable(create):
            return {"status": "unavailable", "reason": "pet diary is unavailable"}
        entry = create(
            arguments["content"],
            visibility=arguments.get("visibility", "private"),
            priority=arguments.get("priority", 5),
            tags=arguments.get("tags", ()),
            retention_until=arguments.get("retention_until"),
        )
        return {
            "status": "completed",
            "entry_id": int(getattr(entry, "id")),
            "visibility": str(getattr(entry, "visibility")),
            "priority": int(getattr(entry, "priority")),
            "tag_count": len(tuple(getattr(entry, "tags", ()) or ())),
        }

    async def diary_recall(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """只向当前模型回合返回有界日记内容，不进入普通状态投影。"""

        diary = current_diary()
        open_session = getattr(diary, "open_internal_session", None)
        if not callable(open_session):
            return {"status": "unavailable", "reason": "pet diary is unavailable"}
        actor = f"model:{context.profile_id}"[:128]
        session = open_session(actor)
        list_entries = getattr(session, "list_entries", None)
        if not callable(list_entries):
            return {"status": "unavailable", "reason": "pet diary is unavailable"}
        requested_limit = max(1, min(int(arguments.get("limit", 5)), 8))
        entries = list_entries(
            include_private=bool(arguments.get("include_private", True)),
            include_archived=False,
            limit=requested_limit,
        )
        rows: list[dict[str, object]] = []
        character_budget = 16_000
        truncated = False
        for entry in entries:
            content = str(getattr(entry, "content", ""))
            if len(content) > character_budget:
                truncated = True
                break
            character_budget -= len(content)
            rows.append(
                {
                    "entry_id": int(getattr(entry, "id")),
                    "content": content,
                    "visibility": str(getattr(entry, "visibility")),
                    "priority": int(getattr(entry, "priority")),
                    "tags": tuple(str(item) for item in getattr(entry, "tags", ()) or ()),
                    "created_at": float(getattr(entry, "created_at")),
                    "updated_at": float(getattr(entry, "updated_at")),
                }
            )
        return {
            "status": "completed",
            "entries": tuple(rows),
            "count": len(rows),
            "truncated": truncated or len(rows) < len(entries),
            "confidential": bool(arguments.get("include_private", True)),
        }

    async def schedule(arguments: Mapping[str, Any], context: ToolCallContext) -> Mapping[str, Any]:
        if scheduler is None:
            return {"status": "unavailable"}
        schedule_kwargs: dict[str, object] = {
            "task_id": str(arguments.get("task_id") or "").strip() or None,
            "name": str(arguments["name"]).strip(),
            "expression": str(arguments["expression"]).strip(),
            "action": dict(arguments.get("action") or {}),
            "owner": f"{context.profile_id}:{context.session_id}",
        }
        # 只有调用方显式提供元数据时才传入新参数，保持旧宿主的
        # ``upsert`` 鸭子类型兼容；当前 SchedulerService 会进一步严格
        # 校验 ``require_user_active`` 的布尔值。
        # 频率下限在 SchedulerService 内生效；低于下限的表达式会以
        # 明确 reason 返回，分支返回值保持与其他入口一致的 Mapping。
        if "metadata" in arguments:
            metadata = arguments.get("metadata")
            schedule_kwargs["metadata"] = (
                dict(metadata) if isinstance(metadata, Mapping) else metadata
            )
        result = _call(scheduler, "upsert", **schedule_kwargs)
        if inspect.isawaitable(result):
            result = await result
        output = dict(result) if isinstance(result, Mapping) else {"status": "scheduled"}
        # 描述性回执给模型足够信息续写下一步；元数据保持可读但不含密钥，
        # 已由 SchedulerService 严格校验 require_user_active。
        if "owner" in output:
            output["owner"] = context.profile_id if str(output.get("owner", "")) else ""
        return output

    async def list_schedules(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """列出当前会话创建的定时任务，供模型回读自己的提醒。"""

        del arguments
        if scheduler is None:
            return {"status": "unavailable", "reason": "调度器未启用"}
        owner = f"{context.profile_id}:{context.session_id}"
        list_call = _call(scheduler, "list_tasks", owner=owner)
        if inspect.isawaitable(list_call):
            list_call = await list_call
        tasks = list(list_call) if isinstance(list_call, (list, tuple)) else []
        rows = []
        for item in tasks:
            if not isinstance(item, Mapping):
                continue
            rows.append(
                {
                    "task_id": str(item.get("task_id", ""))[:128],
                    "name": str(item.get("name", ""))[:128],
                    "expression": str(item.get("expression", ""))[:128],
                    "enabled": bool(item.get("enabled", True)),
                    "next_run_at": float(item.get("next_run_at", 0.0) or 0.0),
                    "metadata": dict(item.get("metadata") or {}),
                }
            )
        return {"status": "completed", "tasks": tuple(rows), "count": len(rows)}

    async def remove_schedule(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """删除当前会话创建的定时任务；只能删除自己属主的任务。"""

        task_id = str(arguments.get("task_id") or "").strip()
        if not task_id or len(task_id) > 128 or any(char in task_id for char in "\r\n\x00"):
            return {"status": "failed", "reason": "任务标识不可用"}
        if scheduler is None:
            return {"status": "unavailable", "reason": "调度器未启用"}
        owner = f"{context.profile_id}:{context.session_id}"
        result = _call(scheduler, "remove", task_id=task_id, owner=owner)
        if inspect.isawaitable(result):
            result = await result
        return {"status": "removed" if result else "unchanged"}

    async def define_trigger(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        """创建窗口停留等事件触发器；动作执行前仍走统一审批管线。"""

        if trigger_service is None:
            return {"status": "unavailable", "reason": "触发器服务未启用"}
        owner = f"{context.profile_id}:{context.session_id}"
        requested_id = str(arguments.get("trigger_id") or "").strip()
        trigger_id = requested_id or f"trigger-{uuid.uuid4().hex[:12]}"
        if len(trigger_id) > 128 or any(char in trigger_id for char in "\r\n\x00"):
            return {"status": "failed", "reason": "触发器标识不可用"}
        register_kwargs: dict[str, object] = {
            "trigger_id": trigger_id,
            "event_name": str(arguments.get("event_name") or "").strip(),
            "action": dict(arguments.get("action") or {}),
            "owner": owner,
        }
        conditions_mapping = dict(arguments.get("conditions") or {})
        if conditions_mapping:
            register_kwargs["metadata"] = {"conditions": conditions_mapping}
        result = _call(trigger_service, "register", **register_kwargs)
        if inspect.isawaitable(result):
            result = await result
        return {"status": "registered", **(dict(result) if isinstance(result, Mapping) else {})}

    async def run_command(
        arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Mapping[str, Any]:
        argv = normalize_command_argv(arguments.get("command"), allowlist=allowed_commands)
        timeout = normalize_command_timeout(arguments.get("timeout_seconds", 10.0))
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_group_spawn_kwargs(),
        )
        group_id = process_group_id(process)
        stdout_task = asyncio.create_task(_read_process_output(process.stdout))
        stderr_task = asyncio.create_task(_read_process_output(process.stderr))
        timed_out = False
        stdout = b""
        stderr = b""
        try:
            try:
                await asyncio.wait_for(asyncio.shield(process.wait()), timeout=timeout)
            except TimeoutError:
                timed_out = True
                await _terminate_process(process, group_id=group_id)
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task), timeout=1.0
                )
            except TimeoutError:
                # 父进程可能已退出但孙进程仍持有管道；回收整个进程组后再取消读取器。
                timed_out = True
                await _terminate_process(process, group_id=group_id)
                await _cancel_reader_tasks(stdout_task, stderr_task)
                stdout = _completed_output(stdout_task)
                stderr = _completed_output(stderr_task)
        except asyncio.CancelledError:
            await _terminate_process(process, group_id=group_id)
            await _cancel_reader_tasks(stdout_task, stderr_task)
            raise
        finally:
            if process.returncode is None:
                await _terminate_process(process, group_id=group_id)
            await _cancel_reader_tasks(stdout_task, stderr_task)
            close_process_transport(process)
        return {
            "status": "timeout" if timed_out else "completed",
            "return_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace")[-4000:],
            "stderr": stderr.decode("utf-8", errors="replace")[-4000:],
        }

    # ToolSpec 保持既有不可变契约；将白名单 setter 挂在内置处理器上，
    # 使运行时可以在不重建 registry 或正在执行的工具的前提下替换策略。
    setattr(run_command, "set_command_allowlist", set_command_allowlist)

    registry.register_many(
        (
            ToolSpec(
                "desktop:observe_foreground",
                "读取当前前台窗口的最小公开信息",
                {"type": "object", "properties": {}, "additionalProperties": False},
                foreground,
                RiskLevel.LOW,
                "desktop_observation",
                read_only=True,
            ),
            ToolSpec(
                "desktop:cursor_position",
                "读取当前全局光标位置，不执行任何输入操作",
                {"type": "object", "properties": {}, "additionalProperties": False},
                cursor_position,
                RiskLevel.LOW,
                "desktop_observation",
                read_only=True,
            ),
            ToolSpec(
                "desktop:list_processes",
                "读取当前用户可见的进程摘要",
                {
                    "type": "object",
                    "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                    "additionalProperties": False,
                },
                processes,
                RiskLevel.LOW,
                "desktop_observation",
                read_only=True,
            ),
            ToolSpec(
                "desktop:context_snapshot",
                "一次读取前台窗口、当前用户进程摘要、全局光标位置和系统空闲状态",
                {
                    "type": "object",
                    "properties": {
                        "process_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                        }
                    },
                    "additionalProperties": False,
                },
                context_snapshot,
                RiskLevel.LOW,
                "desktop_observation",
                read_only=True,
            ),
            ToolSpec(
                "desktop:capture_screen",
                "在用户确认后捕获指定范围的屏幕画面",
                {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "enum": ["screen", "window", "region"]},
                        "region": {"type": "object"},
                    },
                    "required": ["scope"],
                    "additionalProperties": False,
                },
                capture,
                RiskLevel.HIGH,
                "desktop_observation",
                read_only=True,
            ),
            ToolSpec(
                "desktop:ocr",
                "在用户确认后识别指定屏幕范围中的文字和坐标",
                {
                    "type": "object",
                    "properties": {
                        "scope": {"type": "string", "enum": ["screen", "window", "region"]},
                        "region": {
                            "type": "object",
                            "properties": {
                                "x": {
                                    "type": "integer",
                                    "minimum": -1_000_000,
                                    "maximum": 1_000_000,
                                },
                                "y": {
                                    "type": "integer",
                                    "minimum": -1_000_000,
                                    "maximum": 1_000_000,
                                },
                                "width": {"type": "integer", "minimum": 1, "maximum": 8192},
                                "height": {"type": "integer", "minimum": 1, "maximum": 8192},
                            },
                            "required": ["x", "y", "width", "height"],
                            "additionalProperties": False,
                        },
                        "language": {
                            "type": "string",
                            "pattern": r"^[A-Za-z0-9_.+-]{1,32}$",
                        },
                    },
                    "additionalProperties": False,
                },
                ocr,
                RiskLevel.HIGH,
                "desktop_observation",
                read_only=True,
            ),
            ToolSpec(
                "desktop:click_at",
                "在用户确认后点击指定的全局屏幕坐标",
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                        "y": {"type": "integer", "minimum": -1000000, "maximum": 1000000},
                        "button": {
                            "type": "string",
                            "enum": ["left", "middle", "right"],
                        },
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
                click_at,
                RiskLevel.HIGH,
                "desktop_control",
            ),
            ToolSpec(
                "desktop:automation_batch",
                "在用户确认后执行有限的键盘、鼠标与窗口自动化步骤",
                {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 16,
                            "items": _automation_step_schema(),
                        }
                    },
                    "required": ["steps"],
                    "additionalProperties": False,
                },
                automation_batch,
                RiskLevel.HIGH,
                "desktop_automation",
            ),
            ToolSpec(
                "system:module_status",
                "读取平台、模型适配器、语音和工具模块的脱敏能力状态",
                {"type": "object", "properties": {}, "additionalProperties": False},
                module_status,
                RiskLevel.LOW,
                "system",
                read_only=True,
                kind=ToolKind.SYSTEM,
            ),
            ToolSpec(
                "system:transcribe_audio",
                "通过内置 ASR IPC 转写调用方提供的有界 WAV 或 PCM 字节",
                {
                    "type": "object",
                    "properties": {
                        "audio_base64": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 89478488,
                        },
                        "audio_format": {"type": "string", "enum": ["wav", "pcm_s16le"]},
                        "sample_rate": {
                            "type": "integer",
                            "minimum": 8000,
                            "maximum": 192000,
                        },
                        "channels": {"type": "integer", "minimum": 1, "maximum": 8},
                        "language": {
                            "type": "string",
                            "enum": ["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                        },
                    },
                    "required": ["audio_base64", "audio_format"],
                    "additionalProperties": False,
                },
                transcribe_audio,
                RiskLevel.MEDIUM,
                "system",
                kind=ToolKind.SYSTEM,
                public=False,
            ),
            ToolSpec(
                "pet:move",
                "让桌宠移动到屏幕上的安全位置",
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "duration_ms": {"type": "integer", "minimum": 0, "maximum": 10000},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
                move,
                RiskLevel.MEDIUM,
                "pet_control",
            ),
            ToolSpec(
                "pet:autonomous_move",
                "自主行为内部使用的安全移动步进（不向模型公开）",
                {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "duration_ms": {"type": "integer", "minimum": 0, "maximum": 1000},
                    },
                    "required": ["x", "y"],
                    "additionalProperties": False,
                },
                autonomous_move,
                RiskLevel.LOW,
                "pet_control",
                public=False,
            ),
            ToolSpec(
                "pet:set_expression",
                "设置单个表情，或按持续时间和参数播放多个表情",
                {
                    "type": "object",
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["name"],
                            "not": {"type": "object", "required": ["expressions"]},
                        },
                        {
                            "type": "object",
                            "required": ["expressions"],
                            "not": {
                                "anyOf": [
                                    {"type": "object", "required": ["name"]},
                                    {"type": "object", "required": ["parameters"]},
                                    {"type": "object", "required": ["weight"]},
                                    {"type": "object", "required": ["duration_seconds"]},
                                    {"type": "object", "required": ["transition_seconds"]},
                                ]
                            },
                        },
                    ],
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 64},
                        "expressions": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 64,
                                    },
                                    "weight": {"type": "number", "minimum": 0, "maximum": 1},
                                    "duration_seconds": {
                                        "type": "number",
                                        "minimum": 0.05,
                                        "maximum": 120,
                                    },
                                    "transition_seconds": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 10,
                                    },
                                    "parameters": _action_parameters_schema(),
                                },
                                "required": ["name"],
                                "additionalProperties": False,
                            },
                        },
                        "parameters": _action_parameters_schema(),
                        "weight": {"type": "number", "minimum": 0, "maximum": 1},
                        "duration_seconds": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 120,
                        },
                        "transition_seconds": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 10,
                        },
                        "mode": {"type": "string", "enum": ["sequence", "blend"]},
                        "loop": {"type": "boolean"},
                        "restore": {"type": "string", "minLength": 1, "maxLength": 64},
                    },
                    "additionalProperties": False,
                },
                expression,
                RiskLevel.LOW,
                "pet_control",
            ),
            ToolSpec(
                "pet:play_motion",
                "播放带参数、持续时间、过渡和循环策略的动作",
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "maxLength": 64},
                        "duration_seconds": {
                            "type": "number",
                            "minimum": 0.05,
                            "maximum": 300,
                        },
                        "transition_seconds": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 10,
                        },
                        "loop": {"type": "boolean"},
                        "parameters": _action_parameters_schema(),
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
                motion,
                RiskLevel.LOW,
                "pet_control",
            ),
            ToolSpec(
                "pet:speak",
                "将短句排入流式语音队列；本次不想发声时设置 silent: true，正文仍显示",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "minLength": 1, "maxLength": 2000},
                        "mood": {"type": "string", "maxLength": 32},
                        "language": {"type": "string", "maxLength": 16},
                        "voice": {"type": "string", "minLength": 1, "maxLength": 128},
                        "role": {"type": "string", "maxLength": 128},
                        "silent": {"type": "boolean"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
                speak,
                RiskLevel.LOW,
                "pet_control",
            ),
            ToolSpec(
                "pet:diary_write",
                "记录一条桌宠自己的日记；私密条目不会进入普通记忆或界面状态",
                {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1, "maxLength": 10000},
                        "visibility": {"type": "string", "enum": ["public", "private"]},
                        "priority": {"type": "integer", "minimum": 0, "maximum": 10},
                        "tags": {
                            "type": "array",
                            "maxItems": 16,
                            "items": {"type": "string", "minLength": 1, "maxLength": 64},
                        },
                        "retention_until": {
                            "oneOf": [
                                {"type": "number", "minimum": 0},
                                {"type": "null"},
                            ],
                        },
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
                diary_write,
                RiskLevel.LOW,
                "pet_diary",
            ),
            ToolSpec(
                "pet:diary_recall",
                "回想桌宠自己的日记；private 内容属于内部思考，不得向主人逐字转述",
                {
                    "type": "object",
                    "properties": {
                        "include_private": {"type": "boolean"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 8},
                    },
                    "additionalProperties": False,
                },
                diary_recall,
                RiskLevel.LOW,
                "pet_diary",
                read_only=True,
            ),
            ToolSpec(
                "pet:list_models",
                "列出当前资源目录中可用 Live2D 模型，只暴露资源相对键",
                {"type": "object", "properties": {}, "additionalProperties": False},
                list_live2d_models,
                RiskLevel.LOW,
                "pet_control",
                read_only=True,
            ),
            ToolSpec(
                "pet:switch_model",
                "把桌宠切换到指定的 Live2D 模型资源键",
                {
                    "type": "object",
                    "properties": {
                        "model_key": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 512,
                        }
                    },
                    "required": ["model_key"],
                    "additionalProperties": False,
                },
                switch_live2d_model,
                RiskLevel.MEDIUM,
                "pet_control",
            ),
            ToolSpec(
                "pet:set_click_through",
                "切换桌宠窗口的点击穿透状态；高风险状态需要确认",
                {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}},
                    "required": ["enabled"],
                    "additionalProperties": False,
                },
                set_click_through,
                RiskLevel.HIGH,
                "pet_control",
            ),
            ToolSpec(
                "scheduler:upsert",
                (
                    "创建或更新桌宠定时任务；每天同一时刻的提醒用 daily:HH:MM"
                    "（24 小时制），周期用 every:<数值><ms、s、m 或 h>；"
                    "action 是工具身份与参数的结构；凌晨提醒配合 "
                    "metadata.require_user_active 只在本机活跃时执行"
                ),
                {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "maxLength": 128},
                        "name": {"type": "string", "minLength": 1, "maxLength": 128},
                        "expression": {"type": "string", "minLength": 1, "maxLength": 128},
                        "action": {"type": "object"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["name", "expression", "action"],
                    "additionalProperties": False,
                },
                schedule,
                RiskLevel.MEDIUM,
                "scheduler",
            ),
            ToolSpec(
                "scheduler:list",
                "列出模型自己创建的定时任务，用于回读提醒并清理过期项目",
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                list_schedules,
                RiskLevel.LOW,
                "scheduler",
            ),
            ToolSpec(
                "scheduler:remove",
                "删除模型自己创建的一个定时任务",
                {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    },
                    "required": ["task_id"],
                    "additionalProperties": False,
                },
                remove_schedule,
                RiskLevel.LOW,
                "scheduler",
            ),
            ToolSpec(
                "scheduler:set_trigger",
                "创建窗口停留、用户活跃等事件触发器；动作执行前仍走统一审批管线",
                {
                    "type": "object",
                    "properties": {
                        "trigger_id": {"type": "string", "maxLength": 128},
                        "event_name": {"type": "string", "minLength": 1, "maxLength": 64},
                        "conditions": {"type": "object"},
                        "action": {"type": "object"},
                    },
                    "required": ["event_name", "action"],
                    "additionalProperties": False,
                },
                define_trigger,
                RiskLevel.MEDIUM,
                "scheduler",
            ),
            ToolSpec(
                "system:run_command",
                "在明确允许的命令白名单内执行一次非 shell 命令",
                {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 16,
                        },
                        "timeout_seconds": {"type": "number"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
                run_command,
                RiskLevel.HIGH,
                "system",
                kind=ToolKind.SYSTEM,
            ),
        )
    )
