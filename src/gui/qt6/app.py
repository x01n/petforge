from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import logging
import os
import re
import signal
import stat
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from time import monotonic
from time import time as wall_time
from typing import Any, cast

from app.loop import RuntimeLoop
from app.runtime import (
    ApplicationRuntime,
    asr_capture_configuration,
    build_runtime,
    validate_runtime_configuration,
)
from config.editor import (
    ConfigurationConflictError,
    assert_configuration_revision,
    atomic_write_configuration,
    parse_editor_yaml,
    prepare_persisted_values,
)
from config.loader import (
    ConfigurationError,
    LoadedConfiguration,
    configuration_source_digest,
    default_configuration_values,
    expand_environment_values,
    load_configuration,
    parse_bool,
)
from config.resources import ResourceInventory
from core.adapters.direct.errors import AdapterConfigurationError
from core.events.types import AudioFeature
from core.rendering.actions import ExpressionRequest, MotionRequest
from core.rendering.readiness import RendererReadyState, RendererReadyStatus
from gui.platforms.linux import probe_x11_compositor
from gui.qt6.audio import QtAudioPlayer
from gui.qt6.console import (
    PetConsoleWindow,
    _friendly_stream_text,
    _public_action_label,
    _safe_result_summary,
    _safe_ui_text,
)
from gui.qt6.dispatcher import QtMainThreadDispatcher
from gui.qt6.hotkeys import HotkeyConfig, HotkeyManager
from gui.qt6.mcp_content_panel import MCPContentWindow
from gui.qt6.md3 import md3_web_tokens, theme_from_configuration
from gui.qt6.microphone import CapturedPcm, QtMicrophoneCapture, QtPushToTalkController
from gui.qt6.opengl_host import PetOpenGLWindow, pyside6_available
from gui.qt6.web_console import WebControlSurfaceWindow
from gui.qt6.web_console import pyside6_available as web_console_available
from gui.qt6.web_host import WebPetHost
from gui.renderers.assets import RendererSelection, default_renderer_registry
from gui.renderers.live2d import Live2DRenderer
from gui.renderers.model_catalog import Live2DModelCatalog
from gui.renderers.protocol import normalize_renderer_backend
from gui.renderers.sprite import SpriteRenderer
from gui.renderers.web_live2d import WebLive2DRenderer
from gui.web.server import LocalWebServer
from logger import recent_log_records
from services.conversation.interaction_contract import pet_feedback
from services.model_routing.channels import channel_from_mapping
from services.tts.coordinator import SpeechStatus

logger = logging.getLogger(__name__)

# 连续输入在取消旧回合后短暂合并，避免高负载 Qt 事件循环先启动中间消息，
# 随后又启动最新消息，导致模型请求数量和桌宠动作出现多余一轮。
_RESUBMIT_DEBOUNCE_MS = 120
# 多个句段属于同一回合时，TTSCoordinator 会分别发出 started/completed。
# 首段 completed 与下一段 started 之间可能存在一个很短的空窗；在该窗口内
# 不应立刻把口型切回 Silence，否则连续语音会出现关嘴/张嘴抖动。
_TTS_MOUTH_SILENCE_DEBOUNCE_MS = 120
_RESTART_READY_ARGUMENT = "--restart-ready-file"
_RESTART_READY_TIMEOUT_SECONDS = 20.0
_RESTART_RENDER_READY_TIMEOUT_SECONDS = 15.0
_CONSOLE_MODULE_IDS = frozenset(
    {
        "model",
        "tts",
        "asr",
        "memory",
        "tools",
        "mcp",
        "scheduler",
        "behavior",
        "proactive",
        "renderer",
        "window",
        "watcher",
        "logging",
        "ipc",
    }
)


def _public_logging_handler_status(
    logging_values: Mapping[str, object],
    root_logger: logging.Logger,
) -> Mapping[str, object]:
    """从 MeaPet 自有 handler 生成公开状态，不读取或返回日志路径。"""

    level = str(logging_values.get("level", logging.getLevelName(root_logger.level)) or "")
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        level = "INFO"
    file_values = logging_values.get("file")
    file_values = file_values if isinstance(file_values, Mapping) else {}
    file_configured = bool(file_values.get("enabled", False))
    rotation = str(file_values.get("rotation", "none") or "none").strip().lower()
    handler_kinds = tuple(
        str(getattr(handler, "_meapet_logging_handler", "") or "")
        for handler in root_logger.handlers
    )
    console_values = logging_values.get("console")
    console_values = console_values if isinstance(console_values, Mapping) else {}
    console_configured = bool(console_values.get("enabled", True))
    memory_values = logging_values.get("memory")
    memory_values = memory_values if isinstance(memory_values, Mapping) else {}
    memory_configured = "memory" in logging_values and bool(memory_values.get("enabled", True))
    console_active = "console" in handler_kinds
    file_active = "file" in handler_kinds
    memory_active = "memory" in handler_kinds
    degraded = (
        (file_configured and not file_active)
        or (console_configured and not console_active)
        or (memory_configured and not memory_active)
    )
    result: dict[str, object] = {
        "status": "degraded" if degraded else "ready",
        "available": not degraded,
        "ready": not degraded,
        "level": level,
        "handler_count": len(handler_kinds),
        "console_active": console_active,
        "file_configured": file_configured,
        "file_active": file_active,
        "rotation_enabled": file_active and rotation in {"size", "time"},
    }
    if "memory" in logging_values:
        result.update(
            {
                "memory_configured": memory_configured,
                "memory_active": memory_active,
            }
        )
    return result


def _restart_child_arguments(argv: list[str], marker_path: Path) -> list[str]:
    """移除旧握手参数并为新子进程附加唯一 marker。"""

    result: list[str] = []
    skip_next = False
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item == _RESTART_READY_ARGUMENT:
            skip_next = True
            continue
        if item.startswith(_RESTART_READY_ARGUMENT + "="):
            continue
        result.append(item)
    result.extend((_RESTART_READY_ARGUMENT, str(marker_path)))
    return result


def _write_restart_ready_receipt(
    path: str | Path,
    *,
    ready_status: RendererReadyStatus,
) -> None:
    """只向父进程预建的普通文件写入最小 READY 回执。"""

    if not isinstance(ready_status, RendererReadyStatus) or not ready_status.ready:
        raise ValueError("restart receipt requires a terminal renderer READY status")

    marker = Path(os.path.abspath(os.fspath(path)))
    metadata = os.lstat(marker)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise OSError("restart marker is not a private regular file")
    getuid = getattr(os, "getuid", None)
    if callable(getuid) and metadata.st_uid != getuid():
        raise OSError("restart marker owner does not match")
    flags = os.O_WRONLY | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(marker, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_ino != metadata.st_ino
            or opened.st_dev != metadata.st_dev
        ):
            raise OSError("restart marker changed before write")
        payload = json.dumps(
            {
                "status": "ready",
                "pid": os.getpid(),
                "backend": normalize_renderer_backend(ready_status.backend),
                "renderer": ready_status.public(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _restart_ready_receipt_matches(
    path: str | Path,
    *,
    process_id: int,
    expected_backend: str,
) -> bool:
    """验证子进程 PID、状态及实际渲染后端均与本次重启一致。"""

    marker = Path(os.path.abspath(os.fspath(path)))
    try:
        descriptor = os.open(marker, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            return False
        if metadata.st_size <= 0 or metadata.st_size > 4096:
            return False
        raw = os.read(descriptor, 4097)
        if len(raw) > 4096:
            return False
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False
    finally:
        os.close(descriptor)
    if not isinstance(value, Mapping) or value.get("status") != "ready":
        return False
    pid = value.get("pid")
    if isinstance(pid, bool) or pid != process_id:
        return False
    renderer = value.get("renderer")
    if not isinstance(renderer, Mapping):
        return False
    if renderer.get("state") != RendererReadyState.READY.value or renderer.get("ready") is not True:
        return False
    try:
        actual = normalize_renderer_backend(value.get("backend"))
        expected = normalize_renderer_backend(expected_backend)
    except ValueError:
        return False
    if renderer.get("backend") != actual:
        return False
    required = {
        "web_live2d": (
            "bridge_ready",
            "model_ready",
            "geometry_valid",
            "frame_visible",
            "alpha_nonempty",
        ),
        "opengl": ("model_ready", "geometry_valid", "frame_visible", "alpha_nonempty"),
        "sprite": ("model_ready", "geometry_valid", "frame_visible", "alpha_nonempty"),
        "vulkan": ("model_ready", "geometry_valid", "frame_visible"),
    }.get(actual, ())
    if not required or any(renderer.get(name) is not True for name in required):
        return False
    if actual == "vulkan" and renderer.get("actual_api") != "vulkan":
        return False
    if expected == "auto":
        return actual in {"opengl", "web_live2d", "sprite", "vulkan"}
    return actual == expected


def _image_frame_evidence(image: object) -> tuple[bool, bool]:
    """返回 Qt 抓帧是否有尺寸、是否包含非透明像素。"""

    try:
        image_value: Any = image
        is_null = getattr(image_value, "isNull", None)
        width_getter = getattr(image_value, "width", None)
        height_getter = getattr(image_value, "height", None)
        converter = getattr(image_value, "convertToFormat", None)
        image_format_owner = getattr(image_value.__class__, "Format", None)
        rgba_format = getattr(image_format_owner, "Format_RGBA8888", None)
        if (
            image is None
            or not callable(is_null)
            or bool(is_null())
            or not callable(width_getter)
            or not callable(height_getter)
            or not callable(converter)
            or rgba_format is None
        ):
            return False, False
        width = int(cast(Any, width_getter)())
        height = int(cast(Any, height_getter)())
        if width <= 0 or height <= 0:
            return False, False
        converted: Any = cast(Any, converter)(rgba_format)
        size = int(converted.sizeInBytes())
        data = bytes(converted.constBits())[:size]
    except (AttributeError, BufferError, RuntimeError, TypeError, ValueError):
        return False, False
    return True, bool(data) and any(data[3::4])


def _renderer_ready_status(target: object, backend: object) -> RendererReadyStatus:
    """按真实后端证据生成统一且脱敏的 READY 状态。"""

    try:
        normalized = normalize_renderer_backend(backend)
    except ValueError:
        return RendererReadyStatus(
            backend="unavailable",
            state=RendererReadyState.FAILED,
            reason_code="backend_invalid",
        )
    renderer = getattr(target, "renderer", target)
    request_status = getattr(renderer, "request_ready_status", None)
    if callable(request_status):
        try:
            status = request_status()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            status = None
        if isinstance(status, RendererReadyStatus) and status.backend == normalized:
            return status

    if normalized == "vulkan":
        lifecycle = getattr(target, "lifecycle_status", None)
        diagnostics_getter = getattr(target, "frame_diagnostics", None)
        try:
            diagnostics = diagnostics_getter() if callable(diagnostics_getter) else {}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            diagnostics = {}
        actual_api = str(getattr(lifecycle, "actual_api", "") or "").strip().lower()
        lifecycle_ready = bool(getattr(lifecycle, "available", False))
        visible = bool(
            isinstance(diagnostics, Mapping)
            and diagnostics.get("available") is True
            and diagnostics.get("visible_pixels") is True
        )
        ready = lifecycle_ready and actual_api == "vulkan" and visible
        lifecycle_state = str(getattr(getattr(lifecycle, "state", None), "value", "") or "")
        terminal_failure = lifecycle_state in {"failed", "closed"}
        return RendererReadyStatus(
            backend="vulkan",
            state=(
                RendererReadyState.READY
                if ready
                else RendererReadyState.FAILED
                if terminal_failure
                else RendererReadyState.PENDING
            ),
            model_ready=lifecycle_ready,
            geometry_valid=lifecycle_ready and actual_api == "vulkan",
            frame_visible=visible,
            alpha_nonempty=visible,
            actual_api=actual_api,
            reason_code="" if ready else "vulkan_frame_pending",
        )

    frame = None
    for method_name in ("grabFramebuffer", "grabWindow"):
        getter = getattr(target, method_name, None)
        if not callable(getter):
            continue
        try:
            frame = getter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            frame = None
        break
    frame_visible, alpha_nonempty = _image_frame_evidence(frame)
    if normalized == "opengl":
        model_ready = getattr(renderer, "model", None) is not None
        geometry_valid = bool(getattr(renderer, "last_transform_valid", False))
        actual_api = "opengl"
    elif normalized == "sprite":
        current_frame = getattr(renderer, "current_frame", None)
        try:
            model_ready = bool(
                isinstance(current_frame, Path)
                and current_frame.is_file()
                and current_frame.stat().st_size > 0
            )
        except OSError:
            model_ready = False
        geometry_valid = model_ready
        actual_api = "raster"
    else:
        return RendererReadyStatus(
            backend="unavailable",
            state=RendererReadyState.FAILED,
            reason_code="backend_unavailable",
        )
    ready = model_ready and geometry_valid and frame_visible and alpha_nonempty
    return RendererReadyStatus(
        backend=normalized,
        state=RendererReadyState.READY if ready else RendererReadyState.PENDING,
        model_ready=model_ready,
        geometry_valid=geometry_valid,
        frame_visible=frame_visible,
        alpha_nonempty=alpha_nonempty,
        actual_api=actual_api,
        reason_code="" if ready else "frame_pending",
    )


def _decode_public_expression_request(
    payload: Mapping[str, object],
    resolver: Callable[[str, object], str],
) -> ExpressionRequest:
    """把 Web capability token 转为领域表情请求。"""

    raw_rows = payload.get("expressions")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("expression sequence is empty")
    rows: list[dict[str, object]] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("expression sequence is invalid")
        name = resolver("expression", raw_row.get("name"))
        if not name:
            raise ValueError("expression capability is unavailable")
        row = dict(raw_row)
        row["name"] = name
        rows.append(row)
    return ExpressionRequest.from_arguments(
        {
            "expressions": rows,
            "mode": payload.get("mode", "sequence"),
            "loop": payload.get("loop", False),
        }
    )


def _decode_public_motion_request(
    payload: Mapping[str, object],
    resolver: Callable[[str, object], str],
) -> MotionRequest:
    """把 Web capability token 转为领域动作请求。"""

    name = resolver("motion", payload.get("name"))
    if not name:
        raise ValueError("motion capability is unavailable")
    arguments = dict(payload)
    arguments["name"] = name
    return MotionRequest.from_arguments(arguments)


_TOPMOST_VERIFY_DELAYS_MS = (0, 80, 240, 700, 1200)


def _pet_topmost_surface(window: object | None, web_host: object | None) -> object | None:
    """返回真正承载桌宠顶层标志的 Qt surface。"""

    if window is not None:
        return window
    return getattr(web_host, "view", None) if web_host is not None else None


class _TopmostTransitionController:
    """统一提交置顶切换，并把 Qt 请求收敛为窗口管理器实际状态。"""

    def __init__(
        self,
        *,
        platform: object,
        surface_provider: Callable[[], object | None],
        host_provider: Callable[[], object | None],
        schedule: Callable[[int, Callable[[], None]], object],
    ) -> None:
        self._platform = platform
        self._surface_provider = surface_provider
        self._host_provider = host_provider
        self._schedule = schedule
        self._status: dict[str, object] = {}
        self._generation = 0
        self._pending_request: bool | None = None

    @staticmethod
    def _normalize(
        value: object,
        *,
        fallback_enabled: bool = False,
    ) -> dict[str, object]:
        if not isinstance(value, Mapping):
            return {
                "status": "unavailable",
                "enabled": bool(fallback_enabled),
                "detail": "窗口置顶操作没有返回可验证状态",
            }
        status = str(value.get("status", "") or "").strip().lower()
        if status not in {
            "available",
            "degraded",
            "unavailable",
            "cancelled",
            "requested",
            "pending",
        }:
            status = "unavailable"
        result: dict[str, object] = {
            "status": status,
            "enabled": bool(value.get("enabled", fallback_enabled)),
        }
        if isinstance(value.get("confirmed"), bool):
            result["confirmed"] = bool(value["confirmed"])
        detail = _safe_ui_text(value.get("detail") or value.get("reason"), limit=180)
        if detail:
            result["detail"] = detail
        evidence = value.get("evidence")
        if isinstance(evidence, (tuple, list)):
            result["evidence"] = tuple(str(item) for item in evidence[:8] if str(item))
        return result

    def _read(self) -> dict[str, object]:
        host = self._host_provider()
        host_observed: dict[str, object] | None = None
        host_reader = getattr(host, "always_on_top_status", None)
        if callable(host_reader):
            try:
                host_observed = self._normalize(host_reader())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("host topmost readback failed", exc_info=True)
            else:
                if str(host_observed.get("status", "")) in {"requested", "pending"}:
                    return host_observed

        # 宿主负责修改自己的窗口和交互状态；平台层只补充原生窗口管理器
        # 回读，使 OpenGL/Vulkan 与 WebEngine 使用同一个真实性边界。
        surface = self._surface_provider()
        reader = getattr(self._platform, "always_on_top_status", None)
        if surface is not None and callable(reader):
            try:
                platform_observed = self._normalize(reader(surface))
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("platform topmost readback failed", exc_info=True)
            else:
                if host_observed is None or str(host_observed.get("status", "")) not in {
                    "cancelled",
                    "unavailable",
                }:
                    return platform_observed
        if host_observed is not None:
            return host_observed

        getter = getattr(host, "is_always_on_top", None)
        try:
            enabled = bool(getter()) if callable(getter) else False
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            enabled = False
        return {
            "status": "unavailable",
            "enabled": enabled,
            "detail": "桌宠窗口没有可用的置顶状态回读接口",
        }

    def status(self, *, refresh: bool = False) -> dict[str, object]:
        """返回统一状态；高频 UI 轮询默认只读缓存。"""

        if refresh or not self._status:
            observed = self._read()
            if self._pending_request is None or refresh:
                self._status = observed
        return dict(self._status)

    def _finish(self, value: Mapping[str, object]) -> None:
        self._status = dict(value)
        self._pending_request = None

    def _verify(self, generation: int, requested: bool, attempt: int) -> None:
        if generation != self._generation or self._pending_request is None:
            return
        observed = self._read()
        status = str(observed.get("status", "") or "").strip().lower()
        enabled = bool(observed.get("enabled", False))
        final_attempt = attempt == len(_TOPMOST_VERIFY_DELAYS_MS) - 1
        if status == "available" and enabled == requested:
            observed["confirmed"] = bool(observed.get("confirmed", True))
            self._finish(observed)
            return
        if status == "degraded":
            # Wayland 无可移植的全局堆叠回读；客户端旗标只能作为明确降级终态，
            # 不能让按钮永远停留在 requested。
            self._finish(observed)
            return
        if status in {"cancelled", "unavailable"} and final_attempt:
            self._finish(observed)
            return
        if final_attempt:
            self._finish(
                {
                    "status": "unavailable",
                    "enabled": enabled,
                    "confirmed": bool(observed.get("confirmed", False)),
                    "detail": (
                        "窗口管理器未确认请求的置顶状态"
                        if enabled != requested
                        else "窗口管理器置顶状态无法确认"
                    ),
                }
            )
            return
        self._status = {
            "status": "requested",
            "enabled": requested,
            "confirmed": False,
            "detail": "正在等待窗口管理器确认置顶状态",
        }

    def _schedule_readback(self, requested: bool) -> None:
        self._generation += 1
        generation = self._generation
        self._pending_request = requested
        try:
            for attempt, delay_ms in enumerate(_TOPMOST_VERIFY_DELAYS_MS):
                self._schedule(
                    delay_ms,
                    lambda generation=generation, requested=requested, attempt=attempt: (
                        self._verify(generation, requested, attempt)
                    ),
                )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            self._finish(
                {
                    "status": "unavailable",
                    "enabled": bool(self._status.get("enabled", False)),
                    "confirmed": False,
                    "detail": "无法调度窗口管理器置顶状态回读",
                }
            )

    def toggle(self) -> dict[str, object]:
        """通过当前渲染宿主切换置顶，并有界轮询统一状态。"""

        current = self.status(refresh=True)
        requested = not bool(current.get("enabled", False))
        host = self._host_provider()
        setter = getattr(host, "set_always_on_top", None)
        handler = setter if callable(setter) else getattr(host, "toggle_always_on_top", None)
        if not callable(handler):
            return {
                "status": "unavailable",
                "enabled": bool(current.get("enabled", False)),
                "detail": "桌宠窗口不可用",
            }
        try:
            raw_result = setter(requested) if callable(setter) else handler()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug("host topmost update failed", exc_info=True)
            raw_result = {
                "status": "unavailable",
                "enabled": bool(current.get("enabled", False)),
                "detail": "窗口置顶请求失败",
            }
        result = self._normalize(raw_result, fallback_enabled=requested)
        self._status = result
        if str(result.get("status", "")) not in {"cancelled", "unavailable"}:
            self._schedule_readback(requested)
        else:
            self._generation += 1
            self._pending_request = None
        return dict(result)


class _TTSActivityTracker:
    """按语音请求计数并为终态 Silence 提供代次校验。

    ``SpeechStatus`` 只携带会话上下文，而一个回合可能包含多个串行句段。
    这里按 context 记录活动请求数，并用全局代次使延迟 Silence 在下一段
    started 后失效。该类不依赖 Qt，便于在无桌面环境中回归竞态语义。
    """

    def __init__(self) -> None:
        self._counts: dict[object, int] = {}
        self._generation = 0
        self._lock = threading.RLock()

    def started(self, context: object) -> None:
        """登记一个新的 TTS 句段。"""

        try:
            hash(context)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._counts[context] = self._counts.get(context, 0) + 1
            self._generation += 1

    def finished(self, context: object) -> int | None:
        """登记句段终态，返回需要延迟检查的 Silence 代次。"""

        try:
            hash(context)
        except (TypeError, ValueError):
            with self._lock:
                self._generation += 1
                return self._generation
        with self._lock:
            count = self._counts.get(context, 0)
            if count <= 1:
                self._counts.pop(context, None)
            else:
                self._counts[context] = count - 1
            self._generation += 1
            if not self._counts:
                return self._generation
            return None

    def can_silence(self, generation: int) -> bool:
        """判断延迟回调是否仍对应“所有句段已结束”的状态。"""

        with self._lock:
            return generation == self._generation and not self._counts

    @property
    def active(self) -> bool:
        """返回是否仍有任意句段处于活动状态。"""

        with self._lock:
            return bool(self._counts)

    def clear(self) -> None:
        """清空活动请求并推进代次，使已排队回调失效。"""

        with self._lock:
            self._counts.clear()
            self._generation += 1


def _tts_speaking_for_snapshot(
    snapshot: object,
    *,
    active: bool,
    pending_sentence: bool = False,
) -> bool:
    """按真实语音/句子状态决定是否保持口型动画。

    ``BubbleSnapshot.speech_text`` 是最近一句的历史值，不能单独作为张嘴
    条件。终态或配置错误快照即使残留旧句，也必须强制关闭；队首句子在
    尚未收到 TTS started 状态时仍允许短暂保持张嘴，等待音频状态接管。
    """

    phase = getattr(snapshot, "phase", "")
    phase = str(getattr(phase, "value", phase) or "").strip().lower()
    # 终态优先于队列状态：旧队列项不能在已完成/失败/配置错误的回合中
    # 重新触发口型。只有仍处于生成或工具执行阶段时，待发句/活动音频
    # 才能保持张嘴；缺少 phase 的旧快照按活动音频兼容处理。
    if phase in {"completed", "failed", "configuration_required"}:
        return False
    if pending_sentence:
        return True
    if not active:
        return False
    if phase and phase not in {"streaming", "tool_running"}:
        return False
    return True


_PUBLIC_CHANNEL_PROTOCOL_LABELS = {
    "openai": "OpenAI",
    "openai_chat": "OpenAI",
    "openai_responses": "OpenAI",
    "anthropic": "Claude",
    "anthropic_messages": "Claude",
    "claude": "Claude",
    "gemini": "Gemini",
    "gemini_generate": "Gemini",
    "google_gemini": "Gemini",
    "ollama_chat": "Ollama",
}
_PUBLIC_CHANNEL_STATUSES = frozenset(
    {"已停用", "未配置", "暂时冷却", "已就绪", "连接已恢复", "未就绪"}
)
_PUBLIC_MODEL_NOT_CONFIGURED = "模型渠道未配置：点击“配置模型”填写渠道"
_PUBLIC_MODEL_NOT_READY = "模型渠道未就绪：点击“配置模型”检查服务连接"
_PUBLIC_MODEL_COOLDOWN = "模型渠道暂时冷却，请稍后重试或切换渠道"
_PUBLIC_MODEL_DISABLED = "模型渠道已停用，请点击“配置模型”启用服务"


def _resolve_renderer_action(
    controller: object,
    kind: str,
    requested: str,
) -> str:
    """把点击反馈动作解析为当前渲染器确实接受的名称。

    交互契约保留语义化的 ``wave`` 等默认动作，但某些正式 model3 只声明
    了别的动作组（例如当前模型通过 alias 提供 ``blink``/``angry``）。
    点击时先询问渲染器能力；若语义动作不可用，使用同类安全回退，避免
    气泡显示“已挥手”而实际动作被静默拒绝。
    """

    value = str(requested or "").strip()
    if not value:
        return ""
    checker = getattr(controller, f"supports_{kind}", None)
    if not callable(checker):
        return value

    def accepted(name: str) -> bool | None:
        try:
            result = checker(name)
            if inspect.isawaitable(result):
                closer = getattr(result, "close", None)
                if callable(closer):
                    closer()
                return None
            if result is None:
                return None
            return bool(result)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None

    support = accepted(value)
    if support is not False:
        return value
    fallbacks = ("blink", "idle") if kind == "motion" else ("happy", "neutral")
    for fallback in fallbacks:
        if fallback.casefold() == value.casefold():
            continue
        if accepted(fallback) is True:
            return fallback
    return ""


def _refresh_console_action_capabilities(console: object | None, renderer: object | None) -> bool:
    """把当前渲染器的表情/动作声明同步到控制台。"""

    if console is None or renderer is None:
        return False
    refresh = getattr(console, "set_action_capabilities", None)
    if not callable(refresh):
        return False
    capabilities = getattr(renderer, "capabilities", None)
    try:
        refresh(
            getattr(capabilities, "expressions", ()),
            getattr(capabilities, "motions", ()),
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        logger.debug("failed to refresh console action capabilities", exc_info=True)
        return False
    return True


def _public_model_channels(router: object | None) -> tuple[dict[str, object], ...]:
    """把路由器渠道健康压缩为公开控制台可见的最小状态。"""

    channels_getter = getattr(router, "channels", None) if router is not None else None
    if not callable(channels_getter):
        return ()
    try:
        channels = tuple(channels_getter())
    except Exception:
        return ()
    health_getter = getattr(router, "health", None)
    try:
        raw_health = health_getter() if callable(health_getter) else {}
    except Exception:
        raw_health = {}
    health = raw_health if isinstance(raw_health, Mapping) else {}
    active_channel_id = ""
    routed_model = ""
    route_getter = getattr(router, "route", None)
    if callable(route_getter):
        try:
            route = route_getter("dialogue")
        except Exception:
            route = None
        active_channel_id = str(getattr(route, "channel_id", "") or "").strip()
        routed_model = str(getattr(route, "model", "") or "").strip()
    if not active_channel_id:
        # 没有显式 route 时，显示路由器实际会选中的首个渠道，避免界面把
        # 用户真正使用的渠道误报为“未选择”。
        resolve = getattr(router, "resolve", None)
        if callable(resolve):
            try:
                selection = resolve("dialogue")
                primary = getattr(selection, "primary", None)
                active_channel_id = str(getattr(primary, "id", "") or "").strip()
                routed_model = str(
                    getattr(selection, "requested_model", "")
                    or getattr(primary, "selected_model", "")
                    or ""
                ).strip()
            except Exception:
                active_channel_id = ""
    now = wall_time()
    result: list[dict[str, object]] = []
    for channel in channels:
        channel_id = str(getattr(channel, "id", "") or "").strip()
        if not channel_id:
            continue
        protocol_key = str(getattr(channel, "protocol", "") or "").strip().lower()
        protocol = _PUBLIC_CHANNEL_PROTOCOL_LABELS.get(protocol_key, "自定义渠道")
        model_configured = bool(str(getattr(channel, "selected_model", "") or "").strip())
        selected_model = str(getattr(channel, "selected_model", "") or "").strip()[:128]
        display_model = (
            routed_model[:128]
            if channel_id == active_channel_id and routed_model
            else selected_model
        )
        raw_models = getattr(channel, "models", ())
        try:
            models = tuple(
                str(item or "").strip()[:128] for item in raw_models if str(item or "").strip()
            )[:32]
        except TypeError:
            models = ()
        if selected_model and selected_model not in models:
            models = (selected_model, *models)[:32]
        enabled = bool(getattr(channel, "enabled", False))
        ready = bool(getattr(channel, "is_ready", False))
        raw_row = health.get(channel_id, {}) if isinstance(health, Mapping) else {}
        row = raw_row if isinstance(raw_row, Mapping) else {}
        try:
            failures = max(0, min(2**31 - 1, int(row.get("failures", 0) or 0)))
        except (TypeError, ValueError, OverflowError):
            failures = 0
        try:
            cooldown_until = float(row.get("cooldown_until", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            cooldown_until = 0.0
        if not enabled:
            status = "已停用"
        elif not model_configured or not bool(getattr(channel, "base_url", "")):
            status = "未配置"
        elif cooldown_until > now:
            status = "暂时冷却"
        elif ready and failures:
            status = "连接已恢复"
        elif ready:
            status = "已就绪"
        else:
            status = "未就绪"
        result.append(
            {
                "id": channel_id,
                "protocol": protocol,
                "model": display_model,
                "models": models,
                "model_configured": model_configured,
                "ready": ready,
                "failures": failures,
                "active": channel_id == active_channel_id,
                "selectable": bool(enabled and ready and cooldown_until <= now),
                "status": status if status in _PUBLIC_CHANNEL_STATUSES else "未就绪",
            }
        )
    return tuple(result)


def _public_model_image_status(router: object | None) -> dict[str, object]:
    """把图片处理能力投影为固定用户状态，不暴露路由细节。"""

    getter = getattr(router, "image_capability_status", None) if router is not None else None
    if not callable(getter):
        return {"ready": False, "mode": "unavailable", "label": "图片理解未配置"}
    try:
        value = getter("dialogue")
    except Exception:
        return {"ready": False, "mode": "unavailable", "label": "图片理解暂不可用"}
    if not isinstance(value, Mapping):
        return {"ready": False, "mode": "unavailable", "label": "图片理解未配置"}
    mode = str(value.get("mode", "") or "").strip().lower()
    if mode not in {"direct", "summary", "unavailable"}:
        mode = "unavailable"
    labels = {
        "direct": "主模型直接识图",
        "summary": "视觉模型摘要回退",
        "unavailable": "图片理解未配置",
    }
    ready = bool(value.get("ready", False)) and mode != "unavailable"
    return {"ready": ready, "mode": mode, "label": labels[mode]}


def _select_renderer_for_run(
    resource_root: str | Path,
    requested_backend: object,
    *,
    requested_model: object = None,
) -> RendererSelection:
    """通过统一注册表解析启动配置，保持 Qt 装配层只消费选择结果。"""

    registry = default_renderer_registry()
    if requested_model is None:
        return registry.select(resource_root, requested_backend=requested_backend)
    return registry.select(
        resource_root,
        requested_backend=requested_backend,
        requested_model=requested_model,
    )


def _configure_qt_webengine_renderer() -> None:
    """只请求透明画布；软件 WebGL 作为显式回退而不是默认强制。"""

    required_flags = ["--enable-transparent-visuals"]
    software = str(os.environ.get("MEAPET_WEBENGINE_SOFTWARE", "") or "").strip().lower()
    if software in {"1", "true", "yes", "on"}:
        # 仅在目标环境明确要求时启用。SwiftShader 在 Chromium 新版中默认
        # 会被 WebGL 安全策略拒绝；没有 ``--enable-unsafe-swiftshader`` 时，
        # Pixi 会报告 ``Unable to auto-detect a suitable renderer``，导致
        # 虚拟窗口中模型永远停在未就绪。该开关只在用户显式请求软件路径时
        # 注入，不影响默认硬件/系统后端。
        # ``--disable-gpu-compositing`` 保留给 Xvfb 的透明 Qt surface，避免
        # DMA-BUF 合成黑块；unsafe SwiftShader 让 WebGL 本身仍可创建上下文。
        required_flags.extend(
            (
                "--use-gl=angle",
                "--use-angle=swiftshader",
                "--enable-unsafe-swiftshader",
                "--disable-gpu-compositing",
            )
        )
        # Chromium 的 WebGL 后端与 Qt WebEngine 的 QQuickWidget 场景图必须
        # 使用同一条软件路径。只切换 Chromium 而保留 Mesa/llvmpipe 时，
        # 置顶、穿透等窗口标志重建可能在虚拟显示中触发原生段错误；用户若
        # 已显式选择其它 Qt Quick 后端则尊重其选择。
        if not str(os.environ.get("QT_QUICK_BACKEND", "") or "").strip():
            os.environ["QT_QUICK_BACKEND"] = "software"
    current = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "").split()
    for flag in required_flags:
        if flag not in current:
            current.append(flag)
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(current)


def _install_shutdown_signal_handlers(app: Any) -> Callable[[], None]:
    """把终端中断转成 Qt 退出请求，并返回一次性恢复函数。

    Python 只允许主线程安装信号处理器。Qt 事件循环运行期间，处理器只
    调用线程安全的 ``QCoreApplication.quit``，真正的资源释放仍由 ``run``
    的统一 ``finally`` 路径完成。这样 SIGINT、SIGTERM、托盘退出和窗口
    关闭共享同一套 RuntimeLoop、音频、调度器及数据库收尾逻辑。
    """

    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    previous: dict[int, Any] = {}
    signals: list[int] = [signal.SIGINT]
    sigterm = getattr(signal, "SIGTERM", None)
    if sigterm is not None:
        signals.append(sigterm)
    shutdown_requested = False
    shutdown_timer: Any = None

    # PySide 在 Python 信号处理器中直接调用 ``QApplication.quit`` 时，
    # 某些 Qt 版本只设置了退出标志却没有唤醒正在运行的 C++ 事件循环。
    # 用一个短周期 Qt 定时器在事件循环线程再次调用 ``exit``，避免
    # ``timeout 12s uv run meapet`` 或终端 SIGTERM 留下不退出的进程。
    try:
        from PySide6.QtCore import QTimer

        shutdown_timer = QTimer(app)
        shutdown_timer.setInterval(50)
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, TypeError, ValueError):
        shutdown_timer = None

    def request_shutdown(signum: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.debug("duplicate shutdown signal ignored: %s", signum)
            return
        shutdown_requested = True
        logger.info("received shutdown signal: %s", signum)
        quit_method = getattr(app, "quit", None)
        if not callable(quit_method):
            return
        try:
            quit_method()
        except (AttributeError, RuntimeError, TypeError):
            logger.exception("Qt application rejected shutdown request")

    def poll_shutdown() -> None:
        """在 Qt 事件循环线程内再次提交退出请求。"""

        if not shutdown_requested:
            return
        if shutdown_timer is not None:
            try:
                shutdown_timer.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        exit_method = getattr(app, "exit", None)
        if callable(exit_method):
            try:
                exit_method(0)
                return
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        # 非 Qt 测试替身没有 exit()；真实 QApplication 会优先走上面的
        # 明确退出码，保留原有 quit 兼容路径。
        quit_method = getattr(app, "quit", None)
        if callable(quit_method):
            try:
                quit_method()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass

    if shutdown_timer is not None:
        try:
            shutdown_timer.timeout.connect(poll_shutdown)
            shutdown_timer.start()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            shutdown_timer = None

    try:
        for signum in signals:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_shutdown)
    except (OSError, RuntimeError, ValueError):
        if shutdown_timer is not None:
            try:
                shutdown_timer.stop()
                shutdown_timer.deleteLater()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, RuntimeError, ValueError):
                pass
        return lambda: None

    restored = False

    def restore() -> None:
        nonlocal restored
        if restored:
            return
        restored = True
        if shutdown_timer is not None:
            try:
                shutdown_timer.stop()
                shutdown_timer.deleteLater()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        for signum, handler in previous.items():
            try:
                signal.signal(signum, handler)
            except (OSError, RuntimeError, ValueError):
                logger.debug("failed to restore signal handler: %s", signum)

    return restore


def _tray_is_recoverable(tray: object) -> bool:
    """只有托盘对象确认可见时才允许把点击穿透视为可恢复。"""

    visible = getattr(tray, "isVisible", None)
    if not callable(visible):
        return False
    try:
        return bool(visible())
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _click_through_operation_succeeded(result: object, *, enabled: bool) -> bool:
    """判断点击穿透切换是否已被窗口层接受。

    恢复入口不能只依赖 Qt 控件的视觉状态：窗口标志更新失败时仍把复选框
    清掉，会让用户误以为输入已经恢复。Wayland 的 ``degraded`` 只表示 Qt
    已提交请求，因此在有明确状态时同样视为一次已接受的切换。
    """

    if result is None:
        return False
    if isinstance(result, bool):
        return bool(result)
    if isinstance(result, Mapping):
        raw_status = result.get("status", result.get("state", ""))
        status = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
        reported = result.get("enabled")
        if isinstance(reported, bool) and status in {
            "available",
            "degraded",
            "requested",
            "completed",
        }:
            return reported == bool(enabled)
        return status in {"available", "degraded", "requested", "completed"}
    raw_state = getattr(result, "state", "")
    state = str(getattr(raw_state, "value", raw_state) or "").strip().lower()
    return state in {"available", "degraded", "requested", "completed"}


_CRITICAL_INTERACTION_PHASES = frozenset(
    {"streaming", "tool_running", "approval_required", "failed"}
)
_SPEECH_QUEUE_BLOCKING_PHASES = frozenset(
    {"tool_running", "approval_required", "failed", "configuration_required"}
)


def _speech_queue_blocked(*, phase: object, tool_status: object) -> bool:
    """工具、审批、失败和配置入口状态优先于待显示句子。"""

    phase_value = str(getattr(phase, "value", phase) or "").strip().lower()
    return phase_value in _SPEECH_QUEUE_BLOCKING_PHASES or bool(str(tool_status or "").strip())


_HOT_RELOAD_CONFIGURATION_ROOTS = frozenset(
    {
        "llm",
        "tts",
        "memory",
        "behavior",
        "watcher",
        "config",
        "ui",
        "scheduler",
        "logging",
        "rendering",
        "app",
    }
)

_ENVIRONMENT_REFERENCE_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _environment_reference_locations(values: object) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """收集配置中所有环境变量引用及其精确字段路径。"""

    locations: list[tuple[str, tuple[object, ...]]] = []
    active: set[int] = set()

    def visit(value: object, path: tuple[object, ...]) -> None:
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                raise ConfigurationError("configuration contains recursive aliases")
            active.add(identity)
            try:
                for raw_key, nested in value.items():
                    visit(nested, (*path, str(raw_key)))
            finally:
                active.remove(identity)
            return
        if isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in active:
                raise ConfigurationError("configuration contains recursive aliases")
            active.add(identity)
            try:
                for index, nested in enumerate(value):
                    visit(nested, (*path, index))
            finally:
                active.remove(identity)
            return
        if not isinstance(value, str):
            return
        for match in _ENVIRONMENT_REFERENCE_PATTERN.finditer(value):
            locations.append((match.group(1), path))

    visit(values, ())
    return tuple(locations)


def _is_channel_api_key_path(path: tuple[object, ...]) -> bool:
    """判断路径是否严格指向渠道的密钥字段。"""

    return (
        len(path) == 4
        and path[0] == "llm"
        and path[1] == "channels"
        and isinstance(path[2], int)
        and path[3] in {"api_key", "token"}
    )


def _configuration_restart_sections(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> tuple[str, ...]:
    """返回无法由当前运行时热替换、必须重启才生效的根配置域。"""

    keys = {str(key) for key in before} | {str(key) for key in after}
    return tuple(
        sorted(
            key
            for key in keys
            if key not in _HOT_RELOAD_CONFIGURATION_ROOTS and before.get(key) != after.get(key)
        )
    )


def _is_missing_api_key_reference(values: Mapping[str, Any], error: object) -> bool:
    """判断环境展开失败是否只涉及渠道 API Key 引用。

    配置文件可以安全保存 ``${ENV_NAME}`` 引用，即使当前进程尚未提供该
    变量；真正启动/热加载时仍必须展开成功。其它字段（例如服务地址）
    的缺失变量不能放宽，否则会把不可用配置误写入磁盘。
    """

    message = str(error or "")
    prefix = "environment variable is not set:"
    if not message.startswith(prefix):
        return False
    environment_name = message[len(prefix) :].strip()
    if not environment_name:
        return False
    locations = _environment_reference_locations(values)
    if not locations:
        return False
    # 只有所有未提供的引用都落在渠道 API Key 字段时，才允许配置文件先落盘。
    # 非密钥字段（例如 base_url）缺失时必须继续 fail-closed。
    missing_locations = [(name, path) for name, path in locations if name not in os.environ]
    if not missing_locations:
        missing_locations = [(name, path) for name, path in locations if name == environment_name]
    return bool(missing_locations) and all(
        name == environment_name and _is_channel_api_key_path(path)
        for name, path in missing_locations
    )


def _prepare_persisted_configuration(
    path: Path,
    values: Mapping[str, Any],
    *,
    allow_plaintext_secrets: bool = False,
) -> tuple[dict[str, Any], bool]:
    """读取原始 YAML 并准备安全落盘值。

    返回值第二项表示仅有 API Key 环境变量尚未提供；此状态允许保存引用，
    但调用方不能把未展开的配置交给运行时适配器。
    """

    raw_source = parse_editor_yaml(path.read_text(encoding="utf-8"))
    persisted = prepare_persisted_values(
        raw_source,
        values,
        allow_plaintext_secrets=allow_plaintext_secrets,
    )
    if not isinstance(persisted, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    missing_locations = tuple(
        (name, path)
        for name, path in _environment_reference_locations(persisted)
        if name not in os.environ
    )
    missing_api_key_reference = bool(missing_locations) and all(
        _is_channel_api_key_path(path) for _name, path in missing_locations
    )
    if not missing_api_key_reference:
        # 让配置加载器继续生成原始的缺失变量错误；这里不能吞掉地址、模型
        # 或其它普通字段的环境变量缺失。
        expand_environment_values(persisted)
    return dict(persisted), missing_api_key_reference


def _build_model_test_channel(
    payload: Mapping[str, object], configured_values: Mapping[str, Any]
) -> object:
    """从安全向导载荷和已加载配置构造一次性连接测试渠道。"""

    if not isinstance(payload, Mapping):
        raise ConfigurationError("model test payload must be a mapping")
    channel_values = dict(payload)
    channel_id = str(channel_values.pop("channel_id", "") or "").strip()
    api_key_env = str(channel_values.pop("api_key_env", "") or "").strip()
    # 测试载荷不接受任何原始凭据；已有运行配置才是密钥来源。
    channel_values.pop("api_key", None)
    channel_values.pop("token", None)
    if not channel_id:
        raise ConfigurationError("model test channel id is required")
    configured_llm = configured_values.get("llm")
    configured_channels = (
        configured_llm.get("channels") if isinstance(configured_llm, Mapping) else None
    )
    existing_channel: Mapping[str, Any] | None = None
    if isinstance(configured_channels, list):
        for item in configured_channels:
            if isinstance(item, Mapping) and str(item.get("id", "") or "").strip() == channel_id:
                existing_channel = item
                break
    if existing_channel is not None:
        merged_values = dict(existing_channel)
        merged_values.update(channel_values)
        channel_values = merged_values
    channel_values["id"] = channel_id
    protocol = str(channel_values.get("protocol", "") or "").strip().lower()
    if api_key_env:
        if not os.environ.get(api_key_env):
            raise ConfigurationError("model test API key environment is unavailable")
        channel_values["api_key"] = "${" + api_key_env + "}"
        channel_values["api_key_env"] = api_key_env
    elif protocol == "ollama_chat":
        channel_values.pop("api_key", None)
        channel_values.pop("token", None)
        channel_values.pop("api_key_env", None)
        channel_values.pop("api_key_environment", None)
    expanded = expand_environment_values({"llm": {"channels": [channel_values]}})
    channels = expanded.get("llm", {}).get("channels", [])
    if not isinstance(channels, list) or not channels:
        raise ConfigurationError("model test channel is unavailable")
    return channel_from_mapping(channels[0])


async def _adjust_pet_affection(affection: object) -> object:
    """在 RuntimeLoop 协程边界内执行一次点击好感度变更。"""

    adjust = getattr(affection, "adjust", None)
    if not callable(adjust):
        raise TypeError("affection service does not expose adjust")
    return adjust(1)


def _schedule_pet_affection(
    runtime_loop: RuntimeLoop | None,
    affection: object,
    pending: dict[int, tuple[Any, str, str]],
    *,
    generation: int,
    zone: str,
    phrase: str,
) -> dict[str, str]:
    """把点击好感度写入投递到 RuntimeLoop，不在 Qt 回调中访问数据库。"""

    if runtime_loop is None or not runtime_loop.running:
        return {"status": "unavailable"}
    try:
        future = runtime_loop.submit(_adjust_pet_affection(affection))
    except RuntimeError:
        return {"status": "unavailable"}
    pending[generation] = (future, zone, phrase)
    return {"status": "pending"}


def _interaction_status_with_local_feedback(
    *,
    phase: str,
    phase_message: str,
    feedback_notice: str,
    feedback_active: bool,
) -> str:
    """短时保留本地点击反馈，但不遮蔽审批、工具、生成或失败状态。"""

    if feedback_active and phase not in _CRITICAL_INTERACTION_PHASES:
        return str(feedback_notice or "").strip()
    return str(phase_message or "").strip()


def _model_channel_diagnostics(runtime: ApplicationRuntime) -> Mapping[str, object] | None:
    """读取路由诊断；失败时只记录异常类型，不把异常正文交给界面。"""

    router = getattr(runtime, "router", None)
    getter = getattr(router, "diagnostics", None)
    if not callable(getter):
        return None
    try:
        diagnostics = getter("dialogue")
    except Exception as exc:
        logger.debug("model channel diagnostics unavailable: %s", type(exc).__name__)
        return None
    return diagnostics if isinstance(diagnostics, Mapping) else None


async def _runtime_module_test(
    runtime: ApplicationRuntime,
    module_id: str,
) -> Mapping[str, object]:
    """调用运行时逐模块探针，不回退到全量模块状态工具。"""

    tester = getattr(runtime, "test_module", None)
    if not callable(tester):
        return {
            "status": "unavailable",
            "reason_code": "module_probe_unavailable",
        }
    try:
        value = tester(module_id)
        if inspect.isawaitable(value):
            value = await value
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("runtime module test unavailable: %s", type(exc).__name__)
        return {
            "status": "unavailable",
            "reason_code": "module_probe_failed",
        }
    if not isinstance(value, Mapping):
        return {
            "status": "unavailable",
            "reason_code": "invalid_module_probe_result",
        }
    return dict(value)


def _model_channel_diagnostic_summary(runtime: ApplicationRuntime) -> str:
    """生成仅供日志使用的路由摘要，保留诊断原因但不返回给公开控制面。"""

    diagnostics = _model_channel_diagnostics(runtime)
    if diagnostics is None:
        return "diagnostics_unavailable"
    rows = diagnostics.get("channels", ())
    try:
        channel_rows = tuple(rows)
    except TypeError:
        channel_rows = ()
    # ModelRouter 的 diagnostics 契约不含密钥；日志仍限制长度并折叠换行，
    # 避免异常或供应商正文把日志格式污染成多行可执行内容。
    reason = _safe_ui_text(diagnostics.get("reason", ""), limit=240)
    return (
        f"ready={bool(diagnostics.get('ready'))} "
        f"channels={len(channel_rows)} reason={reason or 'unspecified'}"
    )


def _model_channel_status(runtime: ApplicationRuntime) -> tuple[bool, str]:
    """返回脱敏的对话渠道状态和点击式下一步提示。"""

    diagnostics = _model_channel_diagnostics(runtime)
    if diagnostics is None:
        return False, _PUBLIC_MODEL_NOT_READY
    try:
        rows = tuple(diagnostics.get("channels", ()) or ())
    except TypeError:
        rows = ()
    if bool(diagnostics.get("ready")):
        # ready 只代表地址/模型可路由；免密本地网关是合法场景，因此
        # 不把缺少密钥直接判为不可用，但在已知云端协议上明确提醒用户
        # 通过环境变量或安全密钥入口完成认证。
        missing_auth = any(
            isinstance(row, Mapping)
            and "api_key_configured" in row
            and not bool(row.get("api_key_configured", False))
            and str(row.get("protocol", "") or "").strip().lower() != "ollama_chat"
            for row in rows
        )
        if missing_auth:
            return True, (
                f"模型渠道已就绪（{len(rows)} 个）；未检测到密钥，免密网关可继续，"
                "需要鉴权的服务请点击“配置模型”"
            )
        return True, f"模型渠道已就绪（{len(rows)} 个）"
    if not rows:
        return False, _PUBLIC_MODEL_NOT_CONFIGURED

    # 仅按路由器公开的有限状态分类；绝不把 reason 原文（其中可能包含
    # CLI、环境变量、路径或请求细节）拼接到用户可见文案。
    normalized_reason = " ".join(str(diagnostics.get("reason", "") or "").split()).casefold()
    if "disabled" in normalized_reason or "停用" in normalized_reason:
        return False, _PUBLIC_MODEL_DISABLED
    if "cooldown" in normalized_reason or "冷却" in normalized_reason:
        return False, _PUBLIC_MODEL_COOLDOWN
    return False, _PUBLIC_MODEL_NOT_READY


_PUBLIC_CONNECTION_TEST_MESSAGES = {
    "authentication": "模型连接失败，请点击“配置模型”检查密钥设置",
    "authorization": "模型连接失败，请点击“配置模型”检查访问权限",
    "network": "模型连接失败，请点击“配置模型”检查服务连接",
    "timeout": "模型连接超时，请稍后重试",
    "rate_limit": "模型连接受限，请稍后重试",
    "server": "模型服务暂时不可用，请稍后重试",
    "protocol": "模型响应格式不受支持，请检查连接设置",
    "configuration": "模型连接配置不完整，请点击“配置模型”补齐设置",
    "cancelled": "模型连接测试已停止",
}
_PUBLIC_CONNECTION_SUCCESS_STATUSES = frozenset(
    {"available", "ready", "ok", "completed", "success"}
)
_PUBLIC_CONNECTION_PENDING_STATUSES = frozenset({"pending", "requested", "started"})


def _public_model_connection_result(result: object) -> dict[str, str]:
    """将连接探测结果压缩为固定状态和文案，拒绝回显内部字段。"""

    if isinstance(result, Mapping):
        raw_status = result.get("status", result.get("state", ""))
        raw_reason = result.get("reason", "")
    else:
        raw_status = getattr(result, "status", "")
        raw_reason = getattr(result, "reason", "")
    status = str(getattr(raw_status, "value", raw_status) or "").strip().casefold()
    reason = str(getattr(raw_reason, "value", raw_reason) or "").strip().casefold()
    if status in _PUBLIC_CONNECTION_SUCCESS_STATUSES:
        return {"status": "available", "message": "模型连接测试通过"}
    if status in _PUBLIC_CONNECTION_PENDING_STATUSES:
        return {"status": "pending", "message": "正在测试模型连接，请稍候…"}
    if status in {"cancelled", "canceled"} or reason == "cancelled":
        return {"status": "unavailable", "message": _PUBLIC_CONNECTION_TEST_MESSAGES["cancelled"]}
    message = _PUBLIC_CONNECTION_TEST_MESSAGES.get(reason)
    if message is None:
        message = "模型连接测试未通过，请点击“配置模型”检查连接设置"
    return {"status": "unavailable", "message": message}


def _public_model_connection_future(
    source: object,
) -> concurrent.futures.Future[dict[str, str]]:
    """包装运行时连接探测 Future，保证 Qt 向导只收到脱敏回执。"""

    target: concurrent.futures.Future[dict[str, str]] = concurrent.futures.Future()
    done = getattr(source, "done", None)
    result_getter = getattr(source, "result", None)
    add_done_callback = getattr(source, "add_done_callback", None)
    if not callable(done) or not callable(result_getter) or not callable(add_done_callback):
        target.set_result(_public_model_connection_result(source))
        return target

    def complete(future: object) -> None:
        try:
            outcome = result_getter()
        except concurrent.futures.CancelledError:
            safe = {
                "status": "unavailable",
                "message": _PUBLIC_CONNECTION_TEST_MESSAGES["cancelled"],
            }
        except BaseException as exc:
            logger.warning("model connection test failed: %s", type(exc).__name__)
            safe = {"status": "unavailable", "message": "模型连接测试未通过，请稍后重试"}
        else:
            safe = _public_model_connection_result(outcome)
        if not target.done():
            target.set_result(safe)

    try:
        add_done_callback(complete)
    except Exception as exc:
        logger.warning("model connection callback setup failed: %s", type(exc).__name__)
        target.set_result(
            {"status": "unavailable", "message": "模型连接测试暂时不可用，请稍后重试"}
        )
    return target


def _conversation_operation_receipt(result: object) -> dict[str, str]:
    """把对话 Future 的终态压缩成点击式控制台可读回执。

    ``ConversationResult`` 只在运行时线程可见，供应商异常也不能直接回显到
    Web/Qt。该函数只接受公开的 ``status``/``finish_reason``，统一映射成有限
    的状态和固定文案，供 Qt 主线程更新最近一次操作。
    """

    raw_status = ""
    if isinstance(result, Mapping):
        raw_status = (
            str(result.get("status", result.get("finish_reason", "")) or "").strip().lower()
        )
    else:
        raw_status = str(getattr(result, "status", "") or "").strip().lower()
    if raw_status in {"completed", "complete", "stop"}:
        return {"status": "completed", "message": "回复已完成"}
    if raw_status in {"approval_required", "awaiting_approval"}:
        return {"status": "approval_required", "message": "等待你确认桌面操作"}
    if raw_status in {"cancelled", "canceled"}:
        return {"status": "cancelled", "message": "对话已停止"}
    if raw_status in {"failed", "error", "tool_limit"}:
        if raw_status == "tool_limit":
            return {"status": "failed", "message": "操作次数达到上限，回复未继续"}
        return {"status": "failed", "message": "回复未完成，可重试"}
    return {"status": "completed", "message": "回复已结束"}


def _apply_conversation_operation_receipt(
    current: Mapping[str, object], operation_id: str, result: object
) -> dict[str, object]:
    """仅在操作编号仍是最新值时更新对话回执。"""

    updated = dict(current)
    if str(current.get("id", "") or "") != str(operation_id or ""):
        return updated
    receipt = _conversation_operation_receipt(result)
    updated.update(
        {
            "status": receipt["status"],
            "message": receipt["message"],
            "updated_at": monotonic(),
        }
    )
    return updated


def _configure_web_view(view: Any, *, always_on_top: bool = True) -> None:
    """把 WebEngine 视图配置为透明、置顶的桌宠窗口。"""

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QFrame

    # 桌宠是展示层，不应因为点击透明区域把当前工作的窗口抢成前台。
    # 控制台/模型向导是独立窗口，仍然可以正常获取焦点。
    flags = Qt.FramelessWindowHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus
    if always_on_top:
        flags |= Qt.WindowStaysOnTopHint
    try:
        focus_policy = getattr(getattr(Qt, "FocusPolicy", Qt), "NoFocus", None)
        if focus_policy is None:
            focus_policy = getattr(Qt, "NoFocus", None)
        set_focus_policy = getattr(view, "setFocusPolicy", None)
        if callable(set_focus_policy) and focus_policy is not None:
            set_focus_policy(focus_policy)
        focus_proxy_getter = getattr(view, "focusProxy", None)
        focus_proxy = focus_proxy_getter() if callable(focus_proxy_getter) else None
        if focus_proxy is not None and focus_proxy is not view:
            proxy_setter = getattr(focus_proxy, "setFocusPolicy", None)
            if callable(proxy_setter) and focus_policy is not None:
                proxy_setter(focus_policy)
        view.setAttribute(Qt.WA_TranslucentBackground, True)
        view.setAttribute(Qt.WA_NoSystemBackground, True)
        view.setAttribute(Qt.WA_OpaquePaintEvent, False)
        # QWebEngineView 的 Chromium viewport 是原生子表面；让 Qt 采用
        # 与 QOpenGLWidget 相同的栈序和透明调色板，避免 Xwayland 下把
        # 未覆盖区域填成不透明黑色。
        always_stack = getattr(Qt, "WA_AlwaysStackOnTop", None)
        if always_stack is not None:
            view.setAttribute(always_stack, True)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    view.setWindowFlags(flags)
    try:
        palette = view.palette()
        transparent = QColor(0, 0, 0, 0)
        palette.setColor(QPalette.ColorRole.Window, transparent)
        palette.setColor(QPalette.ColorRole.Base, transparent)
        view.setPalette(palette)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    try:
        view.setAutoFillBackground(False)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    try:
        view.setFrameShape(QFrame.Shape.NoFrame)
        view.setLineWidth(0)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    try:
        view.setStyleSheet("background: transparent;")
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    try:
        page = view.page()
        set_background = getattr(page, "setBackgroundColor", None)
        if callable(set_background):
            set_background(QColor(0, 0, 0, 0))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    # QWebEngineView 的 focusProxy/Quick 子表面可能在 setWindowFlags 或
    # setHtml 后重新创建；重新遍历现有 QWidget，避免内部窗口再次抢焦点。
    try:
        focus_policy = getattr(getattr(Qt, "FocusPolicy", Qt), "NoFocus", None)
        if focus_policy is None:
            focus_policy = getattr(Qt, "NoFocus", None)
        finder = getattr(view, "findChildren", None)
        from PySide6.QtWidgets import QWidget

        children = tuple(finder(QWidget)) if callable(finder) else ()
        for child in children:
            setter = getattr(child, "setFocusPolicy", None)
            if callable(setter) and focus_policy is not None:
                setter(focus_policy)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        pass


def _window_accepts_focus(target: object | None) -> bool:
    """读取桌宠窗口的焦点能力；读取失败时按不接收焦点处理。"""

    if target is None:
        return False
    try:
        from PySide6.QtCore import Qt

        flags_getter = getattr(target, "windowFlags", None)
        if not callable(flags_getter):
            return False
        return not bool(flags_getter() & Qt.WindowType.WindowDoesNotAcceptFocus)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return False


def _clamp_widget_to_screen(widget: Any, anchor: object | None = None) -> tuple[int, int] | None:
    """把控制台放入屏幕可用区域，避免跟随桌宠位置移出屏幕。"""

    try:
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QGuiApplication

        point: QPoint | None = None
        if isinstance(anchor, QPoint):
            point = anchor
        elif isinstance(anchor, (tuple, list)) and len(anchor) >= 2:
            point = QPoint(int(anchor[0]), int(anchor[1]))
        screen = QGuiApplication.screenAt(point) if point is not None else None
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        area = screen.availableGeometry()
        width_getter = getattr(widget, "width", None)
        height_getter = getattr(widget, "height", None)
        width = max(1, int(width_getter())) if callable(width_getter) else 1
        height = max(1, int(height_getter())) if callable(height_getter) else 1
        # 窄屏/低分辨率上优先保留滚动区域，而不是让窗口越过屏幕边界。
        max_width = max(1, int(area.width()) - 16)
        max_height = max(1, int(area.height()) - 16)
        if width > max_width or height > max_height:
            resizer = getattr(widget, "resize", None)
            if callable(resizer):
                width = min(width, max_width)
                height = min(height, max_height)
                # Qt 会拒绝小于 minimumSize 的 resize；低分辨率下必须先
                # 降低该约束，否则控制台/向导仍会越过屏幕边界。
                min_width_getter = getattr(widget, "minimumWidth", None)
                min_height_getter = getattr(widget, "minimumHeight", None)
                min_width_setter = getattr(widget, "setMinimumWidth", None)
                min_height_setter = getattr(widget, "setMinimumHeight", None)
                try:
                    current_min_width = int(min_width_getter()) if callable(min_width_getter) else 0
                    current_min_height = (
                        int(min_height_getter()) if callable(min_height_getter) else 0
                    )
                    if callable(min_width_setter) and width < current_min_width:
                        min_width_setter(width)
                    if callable(min_height_setter) and height < current_min_height:
                        min_height_setter(height)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                resizer(width, height)
        x = int(area.left())
        y = int(area.top())
        if point is not None:
            x = int(point.x())
            y = int(point.y())
        x = max(int(area.left()), min(x, int(area.right()) - width + 1))
        y = max(int(area.top()), min(y, int(area.bottom()) - height + 1))
        mover = getattr(widget, "move", None)
        if not callable(mover):
            return None
        mover(x, y)
        return x, y
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _place_widget_adjacent(
    widget: Any,
    avoid: object | None = None,
    *,
    anchor: object | None = None,
    gap: int = 12,
    on_tight_layout: Callable[[], None] | None = None,
) -> tuple[int, int] | None:
    """把辅助窗口放在桌宠邻近区域，优先避免遮挡桌宠。

    候选位置按右、左、下、上排序；每个候选都会先限制到当前屏幕可用区域，
    再按与 ``avoid`` 的重叠面积和距离选择。侧边不足以承载可点击窗口时，
    传入 ``on_tight_layout`` 可切换为占满屏幕的滚动布局，并在回调中暂时
    隐藏桌宠，避免生成几百像素宽的不可用参数栏。
    """

    try:
        from PySide6.QtCore import QPoint, QRect
        from PySide6.QtGui import QGuiApplication

        def rect_value(value: object | None) -> QRect | None:
            if isinstance(value, QRect):
                return QRect(value)
            getter = getattr(value, "frameGeometry", None)
            if callable(getter):
                try:
                    result = getter()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    result = None
                if isinstance(result, QRect):
                    return QRect(result)
            getter = getattr(value, "geometry", None)
            if callable(getter):
                try:
                    result = getter()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    result = None
                if isinstance(result, QRect):
                    return QRect(result)
            return None

        def point_value(value: object | None) -> QPoint | None:
            if isinstance(value, QPoint):
                return QPoint(value)
            if isinstance(value, (tuple, list)) and len(value) >= 2:
                return QPoint(int(value[0]), int(value[1]))
            return None

        avoid_rect = rect_value(avoid)
        anchor_point = point_value(anchor)
        if anchor_point is None and avoid_rect is not None and avoid_rect.isValid():
            anchor_point = avoid_rect.center()
        screen = QGuiApplication.screenAt(anchor_point) if anchor_point is not None else None
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return _clamp_widget_to_screen(widget, anchor)
        area = screen.availableGeometry()
        width_getter = getattr(widget, "width", None)
        height_getter = getattr(widget, "height", None)
        width = max(1, int(width_getter())) if callable(width_getter) else 1
        height = max(1, int(height_getter())) if callable(height_getter) else 1
        frame_rect = rect_value(widget)
        frame_width = max(width, frame_rect.width()) if frame_rect is not None else width
        frame_height = max(height, frame_rect.height()) if frame_rect is not None else height
        max_width = max(1, int(area.width()) - 16)
        max_height = max(1, int(area.height()) - 16)
        frame_extra_width = max(0, frame_width - width)
        frame_extra_height = max(0, frame_height - height)
        max_width = max(1, max_width - frame_extra_width)
        max_height = max(1, max_height - frame_extra_height)
        if width > max_width or height > max_height:
            resizer = getattr(widget, "resize", None)
            if callable(resizer):
                width = min(width, max_width)
                height = min(height, max_height)
                # Qt 会拒绝小于 minimumSize 的 resize；低分辨率下必须先
                # 降低该约束，否则控制台/向导仍会越过屏幕边界。
                min_width_getter = getattr(widget, "minimumWidth", None)
                min_height_getter = getattr(widget, "minimumHeight", None)
                min_width_setter = getattr(widget, "setMinimumWidth", None)
                min_height_setter = getattr(widget, "setMinimumHeight", None)
                try:
                    current_min_width = int(min_width_getter()) if callable(min_width_getter) else 0
                    current_min_height = (
                        int(min_height_getter()) if callable(min_height_getter) else 0
                    )
                    if callable(min_width_setter) and width < current_min_width:
                        min_width_setter(width)
                    if callable(min_height_setter) and height < current_min_height:
                        min_height_setter(height)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                resizer(width, height)
                frame_rect = rect_value(widget)
                frame_width = max(width, frame_rect.width()) if frame_rect is not None else width
                frame_height = (
                    max(height, frame_rect.height()) if frame_rect is not None else height
                )

        if avoid_rect is None or not avoid_rect.isValid():
            return _clamp_widget_to_screen(widget, anchor)

        # 低分辨率下，单纯把 760px 控制台夹到屏幕边缘仍可能盖住桌宠。
        # 若左右任一侧有足够空间，临时降低窗口最小宽度并收窄到该侧，
        # 内容由内部滚动区域承载；这样 1024px 屏幕仍能保留桌宠可点击区。
        side_right = max(0, int(area.right()) - avoid_rect.right() - int(gap) + 1)
        side_left = max(0, avoid_rect.left() - int(area.left()) - int(gap))
        side_width = max(side_right, side_left)
        vertical_bottom = max(0, int(area.bottom()) - avoid_rect.bottom() - int(gap) + 1)
        vertical_top = max(0, avoid_rect.top() - int(area.top()) - int(gap))
        vertical_height = max(vertical_bottom, vertical_top)
        min_width_setter = getattr(widget, "setMinimumWidth", None)
        min_height_setter = getattr(widget, "setMinimumHeight", None)
        resizer = getattr(widget, "resize", None)

        def refresh_frame_size() -> None:
            nonlocal width, height, frame_width, frame_height, frame_rect
            width = max(1, int(width_getter())) if callable(width_getter) else width
            height = max(1, int(height_getter())) if callable(height_getter) else height
            frame_rect = rect_value(widget)
            frame_width = max(width, frame_rect.width()) if frame_rect is not None else width
            frame_height = max(height, frame_rect.height()) if frame_rect is not None else height

        needs_full_tight = callable(on_tight_layout) and side_width < max(360, frame_width)
        if needs_full_tight and side_width >= 360:
            # 控制台先尝试自身的紧凑布局；很多 1280px 桌面其实可以容纳
            # 约 620px 的滚动控制台，不应因为默认 760px frame 放不下就
            # 直接隐藏桌宠。若紧凑布局仍超过侧边空间，再进入全屏避让。
            compact_setter = getattr(widget, "set_compact_layout", None)
            if callable(compact_setter):
                try:
                    compact_setter(True)
                    refresh_frame_size()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
            needs_full_tight = side_width < max(360, frame_width)
        if needs_full_tight:
            # 侧边不足以承载当前窗口的实际 frame 宽度时，不能只把窗口
            # 夹到屏幕边缘：那会让控制台与桌宠保留一小段重叠，造成桌宠
            # 点击/焦点被辅助窗口吞掉。由组合根暂时隐藏桌宠，本窗口占满
            # 可用区域并依靠内部滚动；仅在确实没有恢复回调时保留缩窄回退。
            try:
                on_tight_layout()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
            target_width = max(
                1,
                int(area.width()) - 16 - max(0, frame_width - width),
            )
            target_height = max(
                1,
                int(area.height()) - 16 - max(0, frame_height - height),
            )
            if callable(resizer):
                try:
                    current_min_width = (
                        int(getattr(widget, "minimumWidth")())
                        if callable(getattr(widget, "minimumWidth", None))
                        else 0
                    )
                    current_min_height = (
                        int(getattr(widget, "minimumHeight")())
                        if callable(getattr(widget, "minimumHeight", None))
                        else 0
                    )
                    if callable(min_width_setter) and target_width < current_min_width:
                        min_width_setter(target_width)
                    if callable(min_height_setter) and target_height < current_min_height:
                        min_height_setter(target_height)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                resizer(target_width, target_height)
                refresh_frame_size()
            mover = getattr(widget, "move", None)
            if not callable(mover):
                return None
            x = int(area.left()) + 8
            y = int(area.top()) + 8
            try:
                mover(x, y)
            except (TypeError, ValueError):
                mover(QPoint(x, y))
            return x, y

        try:
            if (
                side_width < 360
                and vertical_height >= 240
                and frame_height > vertical_height
                and callable(min_height_setter)
                and callable(resizer)
            ):
                target_client_height = max(1, vertical_height - max(0, frame_height - height))
                min_height_setter(target_client_height)
                resizer(width, target_client_height)
                refresh_frame_size()
            elif (
                side_width > 0
                and frame_width > side_width
                and callable(min_width_setter)
                and callable(resizer)
            ):
                target_client_width = max(1, side_width - max(0, frame_width - width))
                min_width_setter(target_client_width)
                resizer(target_client_width, height)
                refresh_frame_size()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass

        left = int(area.left())
        top = int(area.top())
        right = int(area.right()) - frame_width + 1
        bottom = int(area.bottom()) - frame_height + 1

        candidates = (
            (avoid_rect.right() + int(gap), avoid_rect.top()),
            (avoid_rect.left() - frame_width - int(gap), avoid_rect.top()),
            (avoid_rect.left(), avoid_rect.bottom() + int(gap)),
            (avoid_rect.left(), avoid_rect.top() - frame_height - int(gap)),
        )
        avoid_center = avoid_rect.center()
        scored: list[tuple[int, int, int, int, int]] = []
        for order, (candidate_x, candidate_y) in enumerate(candidates):
            x = max(left, min(int(candidate_x), right))
            y = max(top, min(int(candidate_y), bottom))
            candidate_rect = QRect(x, y, frame_width, frame_height)
            intersection = candidate_rect.intersected(avoid_rect)
            overlap = intersection.width() * intersection.height() if intersection.isValid() else 0
            distance = abs(candidate_rect.center().x() - avoid_center.x()) + abs(
                candidate_rect.center().y() - avoid_center.y()
            )
            # 重叠面积优先；没有重叠时严格保留右、左、下、上的候选顺序。
            scored.append((int(overlap), order, int(distance), x, y))
        _overlap, _order, _distance, x, y = min(scored)
        mover = getattr(widget, "move", None)
        if not callable(mover):
            return None
        try:
            mover(x, y)
        except (TypeError, ValueError):
            mover(QPoint(x, y))
        # Qt 的 client geometry 与 frameGeometry 在 X11/Wayland 上可能相差
        # 不同边框宽度；候选评分使用的是移动前快照，最终必须用移动后的
        # 实际矩形再做一次不相交校验，否则会留下几个像素的焦点/点击遮挡。
        actual_avoid = rect_value(avoid) or avoid_rect
        actual_widget = rect_value(widget)
        if (
            actual_avoid is not None
            and actual_widget is not None
            and actual_widget.intersects(actual_avoid)
        ):
            current_width = max(1, actual_widget.width())
            current_height = max(1, actual_widget.height())
            safe_candidates = (
                (actual_avoid.right() + int(gap), actual_avoid.top()),
                (actual_avoid.left() - current_width - int(gap), actual_avoid.top()),
                (actual_avoid.left(), actual_avoid.bottom() + int(gap)),
                (actual_avoid.left(), actual_avoid.top() - current_height - int(gap)),
            )
            for safe_x, safe_y in safe_candidates:
                safe_x = max(
                    int(area.left()),
                    min(int(safe_x), int(area.right()) - current_width + 1),
                )
                safe_y = max(
                    int(area.top()),
                    min(int(safe_y), int(area.bottom()) - current_height + 1),
                )
                try:
                    mover(safe_x, safe_y)
                except (TypeError, ValueError):
                    mover(QPoint(safe_x, safe_y))
                refreshed = rect_value(widget)
                if refreshed is not None and not refreshed.intersects(actual_avoid):
                    return safe_x, safe_y
            if callable(on_tight_layout):
                try:
                    on_tight_layout()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
                try:
                    mover(int(area.left()) + 8, int(area.top()) + 8)
                except (TypeError, ValueError):
                    mover(QPoint(int(area.left()) + 8, int(area.top()) + 8))
                return int(area.left()) + 8, int(area.top()) + 8
        return x, y
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
        return _clamp_widget_to_screen(widget, anchor)


def _prepare_qt_platform(configuration: LoadedConfiguration) -> str:
    """选择 Qt 图形后端；auto 优先保留桌宠的可控窗口契约。

    Wayland 原生 surface 的透明合成通常更直接，但绝对定位、置顶和输入
    透明最终由 compositor 决定。当前项目已经为无合成器 X11 提供 alpha
    Shape 掩码，因此在同时存在 Xwayland ``DISPLAY`` 和 Wayland socket 时，
    ``auto`` 选择 xcb 可以保留拖动、坐标移动和窗口标志的可回读能力；用户
    需要纯 Wayland 行为时仍可显式设置 ``ui.qt_platform=wayland``。
    """

    configured = "auto"
    values = configuration.values.get("ui", {})
    if isinstance(values, Mapping):
        configured = str(values.get("qt_platform", "auto") or "auto").strip().lower()
    environment_override = str(os.environ.get("MEAPET_QT_PLATFORM", "") or "").strip().lower()
    if environment_override:
        configured = environment_override
    if configured not in {"auto", "xcb", "wayland"}:
        raise ConfigurationError("ui.qt_platform must be auto, xcb or wayland")

    # 用户显式设置 QT_QPA_PLATFORM 时优先级最高，避免启动器覆盖调试或
    # 打包环境的明确选择。
    session = str(os.environ.get("XDG_SESSION_TYPE", "") or "").strip().lower()
    wayland_display = str(os.environ.get("WAYLAND_DISPLAY", "") or "").strip()
    x11_display = str(os.environ.get("DISPLAY", "") or "").strip()

    explicit = str(os.environ.get("QT_QPA_PLATFORM", "") or "").strip().lower()
    if explicit:
        # 环境变量是用户的明确选择，不能再被合成器探测悄悄改写。无合成器
        # X11 的透明风险由 ``_surface_mask_required`` 和实际入口 smoke
        # 负责收敛，而不是让窗口后端与用户请求不一致。
        return explicit
    if configured == "xcb":
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        return "xcb"
    if configured == "wayland":
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        return "wayland"
    if session == "wayland" and wayland_display and x11_display:
        # xcb 路径在有无合成器两种情况下都由 Shape/真实入口 smoke 收敛透明
        # 表面；换取 Xwayland 对绝对位置、置顶和输入标志的更完整支持。
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        logger.info(
            "检测到 Wayland + Xwayland，ui.qt_platform=auto 已选择 xcb 以保留桌宠窗口控制；"
            "无 X11 合成器时将启用 alpha Shape 掩码"
        )
        return "xcb"
    if session == "wayland" and wayland_display:
        os.environ["QT_QPA_PLATFORM"] = "wayland"
        logger.info("仅检测到原生 Wayland，ui.qt_platform=auto 使用 wayland")
        return "wayland"
    if session == "wayland" and x11_display:
        # 环境只提供 Xwayland DISPLAY、没有 Wayland socket 时，保留 xcb
        # 作为可启动的兼容路径；透明能力仍由运行时屏幕合成审计决定。
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        logger.info("Wayland socket 不可用，ui.qt_platform=auto 回退 xcb")
        return "xcb"
    return "auto"


def _surface_mask_required(platform: object) -> bool:
    """判断当前 Linux 桌宠是否必须使用精确输入 Shape 掩码。

    合成器能否正确合成 alpha 与它是否把透明像素当作输入区域是两件事。
    即使屏幕视觉上没有黑块，矩形 QWebEngine/QWindow 仍会拦截桌面点击并
    抢占焦点。因此 Linux 的 X11/Wayland 都启用基于模型 alpha 的 Shape；
    X11 合成器探测只保留供日志和渲染诊断使用，不再决定输入区域是否收窄。
    """

    backend = str(getattr(platform, "backend", "") or "").strip().lower()
    if backend not in {"x11", "wayland", "windows"}:
        return False
    if backend == "windows":
        # Windows 通过 Qt 输入透明标志和可选 Win32 区域适配器收窄
        # 桌宠实际输入范围；不能沿用 Linux 的 X11 探测。
        return True
    if backend == "x11":
        # 仍执行一次探测，让依赖 X11 的环境问题尽早进入日志；返回值不再
        # 用来放宽输入区域，否则透明侧边会把底层窗口点击吞掉。
        values = dict(os.environ)
        values["QT_QPA_PLATFORM"] = "xcb"
        probe_x11_compositor(values)
    # Qt Wayland 也能提交窗口 input region；最终是否由 compositor 保留
    # 仍由运行时能力状态说明，但不应因为“降级”而主动放弃透明侧边收窄。
    return True


def _close_qt_interaction_resources(
    microphone_controller: object | None,
    audio_player: object | None,
) -> None:
    """在 Qt 退出边界幂等终止 PTT 采集、ASR Future 与物理播放。"""

    for resource, label in (
        (microphone_controller, "microphone"),
        (audio_player, "audio"),
    ):
        close = getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("%s shutdown preparation failed: %s", label, type(exc).__name__)


def run(
    configuration: LoadedConfiguration,
    inventory: ResourceInventory,
    runtime: ApplicationRuntime | None = None,
    *,
    local_web_server_sink: Callable[[LocalWebServer | None], object] | None = None,
    restart_ready_file: str | Path | None = None,
) -> int:
    if not pyside6_available:
        raise RuntimeError("PySide6 is not installed")
    if local_web_server_sink is not None and not callable(local_web_server_sink):
        raise TypeError("local_web_server_sink must be callable")
    early_rendering = configuration.values.get("rendering", {})
    early_backend = normalize_renderer_backend(
        early_rendering.get("backend", "auto") if isinstance(early_rendering, Mapping) else "auto"
    )
    if early_backend != "vulkan":
        _configure_qt_webengine_renderer()
    owns_runtime = runtime is None
    if owns_runtime:
        _prepare_qt_platform(configuration)
    from PySide6.QtCore import QProcess, Qt, QTimer
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    # 控制台关闭只隐藏，真正退出由托盘、控制台按钮或终端信号触发。
    app.setQuitOnLastWindowClosed(False)
    restore_signal_handlers = _install_shutdown_signal_handlers(app)
    dispatcher: QtMainThreadDispatcher | None = None
    audio_player: QtAudioPlayer | None = None
    microphone_capture: QtMicrophoneCapture | None = None
    microphone_controller: QtPushToTalkController | None = None
    runtime_loop: RuntimeLoop | None = None
    try:
        dispatcher = QtMainThreadDispatcher()
        runtime = runtime or build_runtime(configuration, inventory, ui_dispatcher=dispatcher)
        if not owns_runtime:
            runtime.bind_ui_dispatcher(dispatcher)
        tts_values = configuration.values.get("tts", {})
        tts_values = tts_values if isinstance(tts_values, Mapping) else {}
        output_device_id = tts_values.get("output_device_id", "")
        audio_player = QtAudioPlayer()
        if isinstance(output_device_id, str) and output_device_id:
            select_output_device = getattr(audio_player, "set_output_device_id", None)
            if callable(select_output_device):
                select_output_device(output_device_id)
        runtime_loop = RuntimeLoop(runtime)
        asr_values = configuration.values.get("asr", {})
        asr_values = asr_values if isinstance(asr_values, Mapping) else {}
        capture_values = asr_capture_configuration(asr_values)
        microphone_capture = QtMicrophoneCapture(
            enabled=bool(capture_values["enabled"] and runtime.asr.enabled),
            max_audio_bytes=runtime.asr.max_audio_bytes,
            max_duration_seconds=float(capture_values["max_duration_seconds"]),
            device_id=str(capture_values["device_id"]),
        )

        def submit_microphone_transcription(captured: CapturedPcm) -> object:
            if not runtime_loop.running:
                raise RuntimeError("runtime loop is unavailable")
            return runtime_loop.submit(
                runtime.transcribe_audio(
                    captured.data,
                    audio_format="pcm_s16le",
                    sample_rate=captured.sample_rate,
                    channels=captured.channels,
                )
            )

        microphone_controller = QtPushToTalkController(
            microphone_capture,
            transcription_submitter=submit_microphone_transcription,
            audio_player=audio_player,
        )

        def guarded_audio_sink(chunk: object) -> None:
            # 第一阶段使用确定的半双工：录音或转写期间不把新音频交给
            # 物理播放器，避免扬声器回灌到麦克风。文本和 TTS 状态仍保留。
            if microphone_controller is None or microphone_controller.blocks_playback:
                return
            audio_player(chunk)

        runtime.tts.set_audio_sink(guarded_audio_sink)
    except BaseException:
        try:
            if microphone_controller is not None:
                microphone_controller.close()
            elif microphone_capture is not None:
                microphone_capture.close()
            if audio_player is not None:
                audio_player.close()
            if runtime_loop is not None:
                runtime_loop.stop()
            elif runtime is not None:
                try:
                    asyncio.run(runtime.close())
                except BaseException:
                    pass
            if dispatcher is not None:
                dispatcher.close()
        except BaseException as cleanup_error:
            logger.warning("Qt startup cleanup failed: %s", type(cleanup_error).__name__)
        finally:
            restore_signal_handlers()
        raise
    assert dispatcher is not None
    assert audio_player is not None
    assert microphone_capture is not None
    assert microphone_controller is not None
    last_microphone_signature = (
        bool(capture_values["enabled"] and runtime.asr.enabled),
        int(runtime.asr.max_audio_bytes),
        float(capture_values["max_duration_seconds"]),
        str(capture_values["device_id"]),
    )
    last_audio_output_device_id = str(getattr(audio_player, "output_device_id", "") or "")
    assert runtime_loop is not None
    model_channel_ready, model_channel_message = _model_channel_status(runtime)
    if not model_channel_ready:
        logger.info("模型渠道未就绪：%s", model_channel_message)
        logger.debug("模型渠道内部诊断：%s", _model_channel_diagnostic_summary(runtime))
    idle_message = (
        "你好，我正在待机。"
        if model_channel_ready
        else "对话模型尚未连接；表情和动作仍可直接使用。双击桌宠或按 Ctrl+Shift+M 配置聊天"
    )
    resource_root = inventory.root
    rendering_values = configuration.values.get("rendering", {})
    rendering_backend = (
        rendering_values.get("backend", "auto") if isinstance(rendering_values, Mapping) else "auto"
    )
    rendering_model = (
        rendering_values.get("model") if isinstance(rendering_values, Mapping) else None
    )
    try:
        renderer_selection: RendererSelection = _select_renderer_for_run(
            resource_root,
            rendering_backend,
            requested_model=rendering_model,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"rendering backend configuration is invalid: {exc}") from exc
    renderer_labels = {
        "opengl": "OpenGL Live2D",
        "web_live2d": "Web Live2D",
        "sprite": "精灵回退",
        "vulkan": "Vulkan",
    }
    logger.info(
        "渲染后端：%s",
        renderer_labels.get(renderer_selection.backend, "暂时不可用"),
    )
    logger.debug(
        "渲染后端内部选择：requested=%s selected=%s available=%s model=%s choices=%s reason=%s",
        renderer_selection.requested_backend,
        renderer_selection.backend,
        renderer_selection.available,
        renderer_selection.model_path,
        renderer_selection.model_choices,
        renderer_selection.reason,
    )
    if not renderer_selection.available:
        raise ConfigurationError(
            f"requested rendering backend '{renderer_selection.requested_backend}' is unavailable: "
            f"{renderer_selection.reason}"
        )
    renderer_model_catalog = Live2DModelCatalog.scan(resource_root)
    renderer_model_choices = renderer_model_catalog.choices
    renderer_model_selection = renderer_model_catalog.select(rendering_model)
    renderer_model_key = (
        renderer_model_selection.selected.key
        if renderer_model_selection.selected is not None
        else ""
    )
    pending_renderer_model_persistence: dict[int, tuple[str, str]] = {}
    allow_renderer_fallback = renderer_selection.allows_runtime_fallback
    ui_values = configuration.values.get("ui", {})
    if not isinstance(ui_values, Mapping):
        ui_values = {}
    last_applied_theme_signature = json.dumps(
        ui_values.get("theme", {}), ensure_ascii=False, sort_keys=True, default=str
    )
    always_on_top = parse_bool(
        ui_values.get("always_on_top"),
        field_name="ui.always_on_top",
        default=True,
    )
    window_locked = parse_bool(
        ui_values.get("window_locked"),
        field_name="ui.window_locked",
        default=False,
    )
    auto_open_model_setup = parse_bool(
        ui_values.get("auto_open_model_setup"),
        field_name="ui.auto_open_model_setup",
        default=True,
    )
    sprite_scale = (
        float(rendering_values.get("sprite_scale", 0.6))
        if isinstance(rendering_values, Mapping)
        else 0.6
    )
    platform = runtime.platform
    surface_mask_required = _surface_mask_required(platform)
    compositor_state: bool | None = None
    if str(getattr(platform, "backend", "") or "").strip().lower() == "x11":
        try:
            compositor_state = probe_x11_compositor()
        except (OSError, RuntimeError, TypeError, ValueError):
            compositor_state = None
    if surface_mask_required:
        logger.info("桌宠窗口将启用基于渲染 alpha 的 Shape 输入区域")
    if compositor_state is False:
        logger.warning(
            "当前 X11 没有可探测的合成器；Shape 可保护输入区域，但屏幕透明合成可能出现黑色残留"
        )
    timer: Any = None
    cursor_tracking_timer: QTimer | None = None
    hotkey_manager: HotkeyManager | None = None
    tray: Any = None
    tray_watch_timer: Any = None
    tray_resources: list[Any] = []
    click_through_recovery_available = False
    console_recovery_available = False
    tight_layout_pet_hidden = False
    tight_layout_pet_was_visible = False
    console: PetConsoleWindow | None = None
    mcp_content_window: MCPContentWindow | None = None
    web_console: WebControlSurfaceWindow | None = None
    local_web_server: LocalWebServer | None = None
    local_web_sink_notified = False
    active_context: Any = None
    active_future: Any = None
    last_submitted_text: str | None = None
    pending_submit_text: str | None = None
    pending_cancel_future: Any = None
    pending_cancel_operation_id: str | None = None
    pending_resubmit_scheduled = False
    pending_resubmit_token = 0
    pending_console_reads: dict[str, Any] = {}
    pending_approval_actions: dict[str, Any] = {}
    # Web 对话操作先返回 requested，Future 结束后再由 Qt 主线程补写同一
    # operation id 的最终回执；按上下文隔离，避免旧回合覆盖新回合。
    pending_conversation_receipts: dict[object, str] = {}
    pending_web_operation_id: str | None = None
    pending_runtime_reload: Any = None
    pending_restart: dict[str, object] = {}
    pending_pet_affection: dict[int, tuple[Any, str, str]] = {}
    pet_feedback_notice = ""
    pet_feedback_hold_until = 0.0
    last_pet_feedback: dict[str, object] | None = None
    pet_feedback_timer: QTimer | None = None
    # 只在当前没有任何音频分片活动时发送终端 Silence，避免旧回合
    # 的完成回执把新回合的口型提前关闭。每个 context 按句段计数，不能
    # 使用简单 set：同一回合的首句完成时，后句可能尚未发出 started。
    tts_activity = _TTSActivityTracker()
    # 一个回合的多个句段可能分别降级；相同提示只向控制台写一次，避免
    # 失败后每句话都追加一行“语音不可用”。上下文只在进程内保存，不进入
    # 日志或公开状态。
    tts_degraded_contexts: set[object] = set()
    web_renderer: WebLive2DRenderer | None = None
    web_host: WebPetHost | None = None
    last_web_control_state_json = ""
    web_revision = 0
    web_operation_sequence = 0
    web_last_operation: dict[str, object] = {
        "id": "",
        "kind": "",
        "status": "idle",
        "message": "",
        "updated_at": 0.0,
    }
    # Web/Qt 公开控制面共享一份有界活动时间线。条目只保存动作类别、状态
    # 和短消息，内部审批/回合/工具标识仍只留在运行时上下文中。
    web_timeline: list[dict[str, object]] = []
    web_tts_state: dict[str, object] = {
        "state": "idle",
        "message": "语音待命",
        "language": str(getattr(runtime.conversation, "tts_language", "zh") or "zh"),
    }
    # 桌面观察只保留最近一次结果；公开投影会进一步过滤为标题、进程名和 PID。
    web_observation: dict[str, object] = {
        "kind": "",
        "status": "idle",
        "message": "",
    }
    window: Any = None
    topmost_controller = _TopmostTransitionController(
        platform=platform,
        surface_provider=lambda: _pet_topmost_surface(window, web_host),
        host_provider=lambda: window if window is not None else web_host,
        schedule=lambda delay_ms, callback: QTimer.singleShot(delay_ms, callback),
    )
    last_presented_snapshot = runtime.conversation.presentation.snapshot
    # 气泡快照相等时也要关注口型状态：TTS 终态可能只改变 speaking，不能
    # 让旧句的快照比较阻止最终 Silence。
    last_presented_speaking: bool | None = None
    pet_feedback_generation = 0
    shutting_down = False

    def publish_microphone_state(state: str, message: str) -> None:
        """在 Qt 主线程向控制台投影固定状态，不记录设备或正文。"""

        if shutting_down or console is None:
            return
        console.set_microphone_state(state, message)

    def publish_microphone_transcription(text: str) -> None:
        """转写正文只填当前输入框，不发送、不写日志或模块诊断。"""

        if shutting_down or console is None:
            return
        console.set_microphone_transcription(text)

    def publish_microphone_devices(devices: object) -> None:
        """设备标识只进入本地配置选择器，不进入公开状态或日志。"""

        if shutting_down or console is None:
            return
        try:
            console.set_microphone_input_devices(devices)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug("microphone device selector refresh failed")

    microphone_controller.stateChanged.connect(publish_microphone_state)
    microphone_controller.transcriptionReady.connect(publish_microphone_transcription)
    microphone_capture.inputDevicesChanged.connect(publish_microphone_devices)

    def stop_local_web_server() -> None:
        """停止本地控制面并清除进程内宿主句柄。"""

        nonlocal local_web_server, local_web_sink_notified
        server = local_web_server
        was_active = server is not None
        local_web_server = None
        if server is not None:
            try:
                server.stop(timeout=1.0)
            except (OSError, RuntimeError, TypeError, ValueError):
                logger.debug("local web server shutdown failed", exc_info=True)
        if was_active and local_web_server_sink is not None and not local_web_sink_notified:
            local_web_sink_notified = True
            try:
                local_web_server_sink(None)
            except BaseException:
                logger.debug("local web server sink cleanup failed", exc_info=True)

    def _append_web_timeline(kind: object, status: object, message: object = "") -> None:
        """追加公开活动记录，并限制长度和文本边界。"""

        nonlocal web_timeline
        normalized_kind = str(kind or "").strip().lower()[:64]
        normalized_status = str(getattr(status, "value", status) or "").strip().lower()[:32]
        if not normalized_kind:
            return
        normalized_status = normalized_status or "completed"
        safe_message = _safe_ui_text(message, limit=240)
        now = monotonic()
        if web_timeline:
            previous = web_timeline[-1]
            if (
                previous.get("kind") == normalized_kind
                and previous.get("status") == normalized_status
                and previous.get("message") == safe_message
                and now - float(previous.get("updated_at", 0.0) or 0.0) < 0.25
            ):
                return
        web_timeline = [
            *web_timeline,
            {
                "kind": normalized_kind,
                "status": normalized_status,
                "message": safe_message,
                "updated_at": now,
            },
        ][-24:]

    async def publish_tts_status(status: SpeechStatus) -> None:
        """把 TTS 状态投影到 UI，并在所有音频结束后关闭口型。"""

        nonlocal web_tts_state
        state = str(getattr(status, "state", "") or "").strip().lower()
        message = _safe_ui_text(
            getattr(status, "message", "") or "语音状态已更新",
            limit=240,
        )
        context = getattr(status, "context", None)
        terminal = state in {"completed", "cancelled", "degraded"}
        silence_generation: int | None = None
        if state == "started" and context is not None:
            tts_activity.started(context)
        elif terminal:
            silence_generation = tts_activity.finished(context)
        web_tts_state = {"state": state or "idle", "message": message}
        degraded_message = "语音不可用，已保留文字输出"
        report_degraded = state == "degraded"
        if state == "degraded":
            try:
                if context is not None:
                    hash(context)
                    if context in tts_degraded_contexts:
                        report_degraded = False
                    else:
                        if len(tts_degraded_contexts) >= 256:
                            tts_degraded_contexts.clear()
                        tts_degraded_contexts.add(context)
            except (TypeError, ValueError):
                # 外部兼容调用传入不可哈希上下文时仍保留一次提示。
                report_degraded = True
            if report_degraded:
                logger.warning("TTS 暂不可用，已保留文字输出")

        if silence_generation is None and state != "degraded":
            return

        def update_ui() -> None:
            if shutting_down:
                return
            if silence_generation is not None:

                def close_mouth_if_idle(generation: int = silence_generation) -> None:
                    nonlocal last_presented_speaking
                    if shutting_down or not tts_activity.can_silence(generation):
                        return
                    target = window if window is not None else web_host
                    renderer = getattr(target, "renderer", target) if target is not None else None
                    setter = getattr(renderer, "set_mouth_viseme", None)
                    if callable(setter):
                        try:
                            setter("Silence", intensity=0.0)
                        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                            logger.debug("failed to close renderer mouth after TTS", exc_info=True)
                    last_presented_speaking = False

                # 句段之间允许一个短暂的调度空窗；下一段 started 会推进
                # tracker 代次，已排队的回调即使触发也不会关闭正在合成的口型。
                try:
                    QTimer.singleShot(_TTS_MOUTH_SILENCE_DEBOUNCE_MS, close_mouth_if_idle)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    # Qt 事件循环已不可用时仍执行一次代次检查，避免把异常
                    # 传播回 TTS worker；正常运行不会走这里。
                    close_mouth_if_idle()
            if report_degraded and console is not None:
                try:
                    console.set_status(degraded_message)
                    console.append_line(f"系统：{degraded_message}")
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to publish TTS degraded status to console", exc_info=True)

        try:
            pending = dispatcher.invoke(update_ui)
            if inspect.isawaitable(pending):
                await pending
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug("Qt dispatcher rejected TTS status update", exc_info=True)

    async def publish_audio_feature(feature: AudioFeature) -> None:
        """把外部确认的音频口型特征安全投递到当前渲染器。"""

        if not isinstance(feature, AudioFeature):
            return

        def update_renderer() -> None:
            target = window if window is not None else web_host
            renderer = getattr(target, "renderer", None) if target is not None else None
            setter = getattr(renderer, "set_mouth_viseme", None)
            if not callable(setter):
                return
            try:
                accepted = setter(feature.viseme, intensity=feature.intensity)
                if accepted is False:
                    logger.debug("renderer rejected audio feature: %s", feature.viseme)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("failed to publish audio feature to renderer", exc_info=True)

        try:
            pending = dispatcher.invoke(update_renderer)
            if inspect.isawaitable(pending):
                await pending
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug("Qt dispatcher rejected audio feature update", exc_info=True)

    runtime.tts.set_status_sink(publish_tts_status)
    runtime.tts.set_feature_sink(publish_audio_feature)

    def begin_shutdown() -> None:
        """在 Qt 发出 aboutToQuit 时先封锁展示更新和所有 UI 定时器。"""

        nonlocal pet_feedback_generation, shutting_down
        if shutting_down:
            return
        shutting_down = True
        pet_feedback_generation += 1
        tts_activity.clear()
        tts_degraded_contexts.clear()
        # aboutToQuit 先在 Qt 主线程终止采集、ASR Future 和物理播放；
        # finally 中的重复调用保持幂等，RuntimeLoop 仍按既有顺序随后收敛。
        _close_qt_interaction_resources(microphone_controller, audio_player)
        for candidate in (timer, cursor_tracking_timer, tray_watch_timer, pet_feedback_timer):
            if candidate is None:
                continue
            try:
                candidate.stop()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        # WebChannel 的 canvas alpha 回调可能携带较大的 PNG；先在 Qt
        # 事件循环仍可运行时封锁 WebHost 的异步 Shape 解码，避免退出时
        # 主线程被迟到回调占住，随后才执行 renderer/view 的正式清理。
        if web_host is not None:
            prepare = getattr(web_host, "prepare_shutdown", None)
            if callable(prepare):
                try:
                    prepare()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("Web host shutdown preparation failed", exc_info=True)
        if mcp_content_window is not None:
            try:
                mcp_content_window.close()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.debug("MCP content window shutdown preparation failed", exc_info=True)
        # HTTP 请求可能正在等待 Qt 主线程回调；在事件循环仍可派发时先
        # 停止监听，避免关闭 dispatcher 后留下无法完成的本地请求。
        stop_local_web_server()

    about_to_quit = getattr(app, "aboutToQuit", None)
    connect_about_to_quit = getattr(about_to_quit, "connect", None)
    if callable(connect_about_to_quit):
        connect_about_to_quit(begin_shutdown)

    # 未建立托盘恢复入口前始终拒绝开启点击穿透，避免桌宠进入不可恢复状态。
    runtime.pet_controller.set_click_through_guard(
        lambda: click_through_recovery_available or console_recovery_available
    )

    def set_ui_interaction_lock(locked: bool) -> None:
        """让后台自主行为避开正在使用的控制台/配置向导。"""

        setter = getattr(runtime.pet_controller, "set_ui_interaction_locked", None)
        if not callable(setter):
            return
        try:
            setter(bool(locked))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug("failed to update UI interaction lock", exc_info=True)

    async def cancel_context(context: object) -> None:
        """在运行时线程内取消当前回合，避免跨线程操作 asyncio 任务。"""

        if context is not None:
            try:
                await runtime.cancel_conversation(context)
            except (RuntimeError, TypeError, ValueError):
                return

    async def _read_console_tool(
        identity: str, arguments: Mapping[str, object]
    ) -> Mapping[str, object]:
        """通过统一工具权限管线执行控制台只读请求。"""

        return await runtime.execute_console_tool(identity, arguments)

    async def _read_api_audit() -> Mapping[str, object]:
        """在 RuntimeLoop 所在线程读取有界、已脱敏 API 审计摘要。"""

        records = runtime.api_call_audit_public_records(limit=50)
        return {"status": "available", "count": len(records), "records": records}

    async def _read_log_records() -> Mapping[str, object]:
        """读取本机有界日志环；日志正文已经在 sink 边界脱敏。"""

        records = recent_log_records(limit=200)
        return {"status": "available", "count": len(records), "records": records}

    def _request_console_read(label: str, identity: str, arguments: Mapping[str, object]) -> object:
        """提交控制台只读请求，并由 Qt 定时器消费完成结果。"""

        nonlocal web_observation
        observation_kind = (
            "foreground_window"
            if label == "前台窗口"
            else "processes"
            if label == "进程列表"
            else ""
        )
        if observation_kind:
            web_observation = {
                "kind": observation_kind,
                "status": "requested",
                "message": "读取请求已提交",
            }

        current = pending_console_reads.get(label)
        if current is not None and not current.done():
            return {"status": "degraded", "reason": f"{label} call remains in flight"}
        if runtime_loop is None or not runtime_loop.running:
            try:
                result = dict(asyncio.run(_read_console_tool(identity, arguments)))
            except Exception as exc:
                logger.warning("控制台读取失败：%s", type(exc).__name__)
                result = {"status": "unavailable", "reason": "桌面读取暂时不可用，请稍后重试"}
            if label == "模块状态":
                result = _augment_module_status(result)
            if observation_kind:
                web_observation = {"kind": observation_kind, **result}
                _append_web_timeline(
                    "read_foreground_window"
                    if observation_kind == "foreground_window"
                    else "read_processes",
                    result.get("status", "completed"),
                    _safe_console_result(label, result),
                )
            return result
        try:
            future = runtime_loop.submit(_read_console_tool(identity, arguments))
        except RuntimeError:
            result = {"status": "unavailable", "reason": "runtime loop is not running"}
            if observation_kind:
                web_observation = {"kind": observation_kind, **result}
                _append_web_timeline(
                    "read_foreground_window"
                    if observation_kind == "foreground_window"
                    else "read_processes",
                    result["status"],
                    result["reason"],
                )
            return result
        pending_console_reads[label] = future
        if console is not None:
            console.set_status(f"正在读取{label}…")
        return {"status": "requested", "operation": label}

    def _poll_console_reads() -> None:
        """在 Qt 主线程读取只读请求结果并更新控制台。"""

        nonlocal web_observation

        for label, future in tuple(pending_console_reads.items()):
            if not future.done():
                continue
            pending_console_reads.pop(label, None)
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("控制台读取 Future 失败：%s", type(exc).__name__)
                result = {"status": "unavailable", "reason": "桌面读取暂时不可用，请稍后重试"}
            if label == "模块状态":
                result = _augment_module_status(result)
            observation_kind = (
                "foreground_window"
                if label == "前台窗口"
                else "processes"
                if label == "进程列表"
                else ""
            )
            if observation_kind:
                web_observation = {"kind": observation_kind, **dict(result)}
                _append_web_timeline(
                    "read_foreground_window"
                    if observation_kind == "foreground_window"
                    else "read_processes",
                    result.get("status", "completed")
                    if isinstance(result, Mapping)
                    else "completed",
                    _safe_console_result(label, result),
                )
            _console_result(label, result)

    async def _resolve_console_approval(
        approval_id: str, *, approve: bool, grant_session: bool = False
    ) -> Mapping[str, object]:
        """在运行时线程消费审批；恢复入口内部保留 generation 代际校验。"""

        if approve:
            # 控制台批准后必须继续原模型回合；只执行工具而不回填结果会让
            # 桌宠停在“工具已完成”，模型永远不会解释结果或继续下一动作。
            outcome = await runtime.continue_approval(
                approval_id,
                grant_session=bool(grant_session),
            )
            return {
                "status": outcome.status,
                "approval_id": approval_id,
                "identity": outcome.identity,
                "error": str(outcome.content.get("error", "")),
            }
        accepted = runtime.deny_approval(approval_id)
        return {
            "status": "denied" if accepted else "unavailable",
            "approval_id": approval_id,
            "reason": "用户拒绝了工具调用" if accepted else "审批请求已失效",
        }

    def _request_console_approval(
        approval_id: str, *, approve: bool, grant_session: bool = False
    ) -> object:
        """提交批准或拒绝请求，避免 Qt 主线程阻塞运行时事件循环。"""

        key = str(approval_id or "").strip()
        if not key:
            return {"status": "unavailable", "reason": "approval id is empty"}
        current = pending_approval_actions.get(key)
        if current is not None and not current.done():
            return {"status": "degraded", "reason": "审批操作仍在处理中"}
        if runtime_loop is None or not runtime_loop.running:
            return {"status": "unavailable", "reason": "runtime loop is not running"}
        try:
            future = runtime_loop.submit(
                _resolve_console_approval(
                    key,
                    approve=bool(approve),
                    grant_session=bool(grant_session),
                )
            )
        except RuntimeError:
            return {"status": "unavailable", "reason": "runtime loop is not running"}
        pending_approval_actions[key] = future
        return {"status": "requested", "approval_id": key}

    def approve_console_approval(approval_id: str, grant_session: bool = False) -> object:
        return _request_console_approval(
            approval_id,
            approve=True,
            grant_session=bool(grant_session),
        )

    def deny_console_approval(approval_id: str) -> object:
        return _request_console_approval(approval_id, approve=False)

    def _poll_console_approvals() -> None:
        """消费审批操作结果，并让控制台下一帧刷新待审批列表。"""

        for approval_id, future in tuple(pending_approval_actions.items()):
            if not future.done():
                continue
            pending_approval_actions.pop(approval_id, None)
            try:
                result = future.result()
            except Exception as exc:
                logger.warning("审批请求失败：%s", type(exc).__name__)
                result = {"status": "unavailable", "reason": "审批暂时不可用，请重试"}
            _console_result("审批", result)
            _safe_web_result("approval", result)

    def _refresh_console_approvals() -> None:
        if console is None:
            return
        try:
            console.set_pending_approvals(runtime.pending_approvals_for_ui())
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("failed to refresh approval list: %s", type(exc).__name__)

    def _poll_pet_affection() -> None:
        """在 Qt 线程消费点击好感度结果，避免后台线程触碰 Qt。"""

        nonlocal pet_feedback_notice, pet_feedback_hold_until, last_pet_feedback
        for generation, (future, zone, phrase) in tuple(pending_pet_affection.items()):
            if not future.done():
                continue
            pending_pet_affection.pop(generation, None)
            try:
                change = future.result()
            except Exception as exc:
                logger.warning("pet affection update failed: %s", type(exc).__name__)
                if generation == pet_feedback_generation:
                    feedback_mood = str(
                        last_pet_feedback.get("mood", "neutral")
                        if isinstance(last_pet_feedback, Mapping)
                        else "neutral"
                    )
                    pet_feedback_notice = f"点击互动：{phrase}（好感度未更新）"
                    pet_feedback_hold_until = max(pet_feedback_hold_until, monotonic() + 0.8)
                    if console is not None:
                        console.set_pet_feedback(
                            {
                                "zone": zone,
                                "phrase": phrase,
                                "mood": feedback_mood,
                                "affection": {"status": "unavailable"},
                            }
                        )
                    _set_pet_visual_feedback(
                        {
                            "zone": zone,
                            "part": (
                                last_pet_feedback.get("part", zone)
                                if isinstance(last_pet_feedback, Mapping)
                                else zone
                            ),
                            "phrase": phrase,
                            "mood": feedback_mood,
                            "expression": (
                                last_pet_feedback.get("expression", "")
                                if isinstance(last_pet_feedback, Mapping)
                                else ""
                            ),
                            "motion": (
                                last_pet_feedback.get("motion", "")
                                if isinstance(last_pet_feedback, Mapping)
                                else ""
                            ),
                            "affection": {"status": "unavailable"},
                        }
                    )
                    last_pet_feedback = {
                        **(
                            dict(last_pet_feedback)
                            if isinstance(last_pet_feedback, Mapping)
                            else {}
                        ),
                        "zone": zone,
                        "part": str(
                            last_pet_feedback.get("part", zone)
                            if isinstance(last_pet_feedback, Mapping)
                            else zone
                        ),
                        "phrase": phrase,
                        "mood": feedback_mood,
                        "active": True,
                        "affection": {"status": "unavailable"},
                    }
                    _console_result(
                        "点击互动",
                        {
                            "status": "unavailable",
                            "zone": zone,
                            "affection": {"status": "unavailable"},
                        },
                    )
                    _safe_web_result(
                        "pet_part",
                        {"status": "unavailable", "reason": "好感度未更新", "part": zone},
                    )
                continue
            if generation != pet_feedback_generation:
                continue
            applied = int(getattr(change, "applied", 0) or 0)
            current = int(getattr(change, "current", 0) or 0)
            tier = str(getattr(getattr(change, "tier", None), "name", "") or "")
            feedback_mood = str(
                last_pet_feedback.get("mood", "neutral")
                if isinstance(last_pet_feedback, Mapping)
                else "neutral"
            )
            sign = "+" if applied >= 0 else ""
            pet_feedback_notice = f"点击互动：{phrase}（好感度 {sign}{applied}）"
            pet_feedback_hold_until = max(pet_feedback_hold_until, monotonic() + 0.8)
            if console is not None:
                console.set_pet_feedback(
                    {
                        "zone": zone,
                        "phrase": phrase,
                        "mood": feedback_mood,
                        "affection": {
                            "current": current,
                            "applied": applied,
                            "tier": tier,
                        },
                    }
                )
            _set_pet_visual_feedback(
                {
                    "zone": zone,
                    "part": (
                        last_pet_feedback.get("part", zone)
                        if isinstance(last_pet_feedback, Mapping)
                        else zone
                    ),
                    "phrase": phrase,
                    "mood": feedback_mood,
                    "expression": (
                        last_pet_feedback.get("expression", "")
                        if isinstance(last_pet_feedback, Mapping)
                        else ""
                    ),
                    "motion": (
                        last_pet_feedback.get("motion", "")
                        if isinstance(last_pet_feedback, Mapping)
                        else ""
                    ),
                    "affection": {
                        "current": current,
                        "applied": applied,
                        "tier": tier,
                    },
                }
            )
            last_pet_feedback = {
                **(dict(last_pet_feedback) if isinstance(last_pet_feedback, Mapping) else {}),
                "zone": zone,
                "part": str(
                    last_pet_feedback.get("part", zone)
                    if isinstance(last_pet_feedback, Mapping)
                    else zone
                ),
                "phrase": phrase,
                "mood": feedback_mood,
                "active": True,
                "affection": {
                    "current": current,
                    "applied": applied,
                    "tier": tier,
                },
            }
            _console_result(
                "点击互动",
                {
                    "status": "completed",
                    "zone": zone,
                    "affection": {
                        "current": current,
                        "applied": applied,
                        "tier": tier,
                    },
                },
            )
            _safe_web_result(
                "pet_part",
                {
                    "status": "completed",
                    "part": zone,
                    "message": f"好感度 {sign}{applied}",
                },
            )

    def stop_conversation() -> Any:
        """请求中断当前对话，并让流式 TTS 一并丢弃。"""

        nonlocal pending_submit_text, pending_resubmit_scheduled, pending_resubmit_token
        nonlocal pending_web_operation_id
        if pending_resubmit_scheduled:
            # 取消 Future 已完成但最新消息尚未重新提交时，停止按钮仍应
            # 撤销这笔待发送消息，而不是让延迟回调在用户松手后偷偷启动。
            pending_resubmit_token += 1
            pending_resubmit_scheduled = False
            pending_submit_text = None
            pending_web_operation_id = None
            return {
                "status": "cancelled",
                "reason": "已取消等待中的消息",
            }
        if active_context is None:
            # 连续发送正在等待旧回合取消时，“停止”也必须撤销尚未发出的
            # 最新输入，不能让取消 Future 完成后又偷偷启动下一回合。
            if pending_cancel_future is not None:
                pending_submit_text = None
                pending_web_operation_id = None
                return {
                    "status": "cancelled",
                    "reason": "已取消等待中的消息",
                }
            return {
                "status": "unavailable",
                "reason": "当前没有正在生成的对话",
            }
        context = active_context
        if runtime_loop is None or not runtime_loop.running:
            # 连续提交路径把返回值当作 Future；运行服务未启动时保留
            # None 以触发既有“上一轮无法停止”分支，避免把映射误当 Future。
            return None
        try:
            return runtime_loop.submit(cancel_context(context))
        except RuntimeError:
            return None

    def retry_last_conversation() -> object:
        """重试最近一次用户消息，供交互快照动作条使用。"""

        text = str(last_submitted_text or "").strip()
        if not text:
            return {"status": "unavailable", "reason": "没有可重试的消息"}
        submit_text(text)
        return {"status": "requested"}

    def _safe_console_result(name: str, result: object) -> str:
        """把工具结果转换成点击式控制台可读摘要，不回显原始参数。"""

        status_labels = {
            "idle": "等待处理",
            "requested": "已提交",
            "pending": "等待处理",
            "started": "已开始",
            "accepted": "已接受",
            "completed": "已完成",
            "available": "已就绪",
            "updated": "已更新",
            "degraded": "降级运行",
            "failed": "执行失败",
            "error": "暂时不可用",
            "unavailable": "暂时不可用",
            "cancelled": "已停止",
            "canceled": "已停止",
            "denied": "已拒绝",
            "rejected": "已拒绝",
            "timeout": "响应超时",
            "tool_limit": "需要确认",
            "restart_required": "需要重启",
        }
        zone_labels = {
            "head": "猫猫头",
            "body": "身体",
            "upper": "上半身",
            "lower_left": "左边",
            "lower_right": "右边",
        }
        if isinstance(result, Mapping):
            if name == "进程列表":
                rows = result.get("processes")
                if isinstance(rows, (list, tuple)):
                    entries: list[str] = []
                    for row in rows[:20]:
                        if not isinstance(row, Mapping):
                            continue
                        process_name = _safe_ui_text(row.get("name") or row.get("process_name"))
                        pid = _safe_ui_text(row.get("pid"), limit=32)
                        if process_name:
                            entries.append(f"{process_name}({pid})" if pid else process_name)
                    status_key = _safe_ui_text(result.get("status")).casefold()
                    status = status_labels.get(status_key, "已读取")
                    suffix = "、".join(entries) if entries else "无可显示进程"
                    if len(suffix) > 1200:
                        suffix = suffix[:1199] + "…"
                    return f"{status}，进程：{suffix}"
            if name == "前台窗口":
                title = _safe_ui_text(result.get("title"), limit=120)
                process_name = _safe_ui_text(result.get("process_name"))
                pid = _safe_ui_text(result.get("pid"), limit=32)
                fields = [item for item in (title, process_name) if item]
                if pid:
                    fields.append(f"PID：{pid}")
                status_key = _safe_ui_text(result.get("status")).casefold()
                status = status_labels.get(status_key, "已读取")
                return f"{status}，前台窗口：{' / '.join(fields) or '无标题'}"
            status_key = _safe_ui_text(result.get("status")).casefold()
            parts: list[str] = []
            if status_key:
                parts.append(status_labels.get(status_key, "处理中"))
            zone = _safe_ui_text(result.get("zone")).casefold()
            if zone:
                parts.append(f"互动部位：{zone_labels.get(zone, '桌宠')}")
            phrase = _safe_ui_text(result.get("phrase"), limit=180)
            if phrase:
                parts.append(phrase)
            affection = result.get("affection")
            if isinstance(affection, Mapping):
                current = _safe_ui_text(affection.get("current"), limit=32)
                applied = _safe_ui_text(affection.get("applied"), limit=32)
                if current:
                    parts.append(f"好感度 {current}")
                if applied:
                    parts.append(f"本次变化 {applied}")
            message = _safe_ui_text(result.get("message"), limit=240)
            if message and message not in parts:
                parts.append(message)
            if parts:
                return "，".join(parts)
        return _safe_result_summary(result)

    def _console_result(name: str, result: object) -> object:
        """把控制台动作结果转换为可读状态并返回原值。"""

        if console is None:
            return result
        console.append_line(f"系统：{name}：{_safe_console_result(name, result)}")
        setter = getattr(console, "set_diagnostic_result", None)
        if callable(setter) and name in {
            "前台窗口",
            "进程列表",
            "foreground_window",
            "list_processes",
            "模块状态",
            "module_status",
            "API调用审计",
            "api_audit",
            "运行日志",
            "log_records",
        }:
            setter(name, result)
        return result

    def _set_pet_visual_feedback(
        value: Mapping[str, object] | None,
        *,
        visible: bool = True,
    ) -> None:
        """把部位互动结果投影到桌宠自身，不依赖控制台是否打开。"""

        target = window if window is not None else web_host
        setter = getattr(target, "set_interaction_feedback", None)
        if not callable(setter):
            return
        visual_value: Mapping[str, object] | None = value
        if isinstance(value, Mapping):
            # 桌宠窗口上的反馈是普通用户可见文案；动作/表情的内部
            # key 只留在运行时状态，不能把 happy/wave 等原始标识直接
            # 绘制到模型气泡里。
            visual_payload = dict(value)
            if visual_payload.get("mood"):
                visual_payload["mood"] = _public_action_label("expression", visual_payload["mood"])
            if visual_payload.get("expression"):
                visual_payload["expression"] = _public_action_label(
                    "expression", visual_payload["expression"]
                )
            if visual_payload.get("motion"):
                visual_payload["motion"] = _public_action_label("motion", visual_payload["motion"])
            visual_value = visual_payload
        try:
            setter(visual_value, visible=visible)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug("failed to update visual pet interaction feedback", exc_info=True)

    def handle_pet_click(payload: Mapping[str, object]) -> None:
        """把桌宠点击转换为即时气泡、表情、动作和有限好感度反馈。"""

        if shutting_down:
            return
        nonlocal pet_feedback_generation, pet_feedback_notice, pet_feedback_hold_until
        nonlocal last_pet_feedback
        nonlocal pet_feedback_timer
        zone = str(payload.get("zone", "") or "").strip().lower()
        part = str(payload.get("part", "") or "").strip().lower()
        interaction_zone = part if pet_feedback(part) is not None else zone
        feedback = pet_feedback(interaction_zone)
        if feedback is None:
            return
        expression = _resolve_renderer_action(
            runtime.pet_controller,
            "expression",
            feedback.expression,
        )
        motion = _resolve_renderer_action(
            runtime.pet_controller,
            "motion",
            feedback.motion,
        )
        phrase = feedback.phrase
        mood = feedback.mood
        result: dict[str, object] = {"zone": zone, "part": part or interaction_zone}
        if part:
            result["part"] = part
        presentation = runtime.conversation.presentation
        pet_feedback_generation += 1
        feedback_generation = pet_feedback_generation
        pet_feedback_notice = f"点击互动：{phrase}"
        pet_feedback_hold_until = monotonic() + 2.4
        try:
            # 先写入线程安全的直接说话层，再同步更新当前渲染器；后续工具
            # 状态不会覆盖点击气泡，定时器也能从同一快照恢复显示。
            presentation.set_direct_speech(phrase, mood=mood)
            result["expression"] = (
                runtime.pet_controller.set_expression(expression) if expression else False
            )
            result["motion"] = runtime.pet_controller.play_motion(motion) if motion else False
            result["speech"] = runtime.pet_controller.speak(phrase, mood=mood)
        except Exception as exc:
            result["status"] = "unavailable"
            result["reason"] = type(exc).__name__

        # 点击事件本身仍应计入好感度，即使某个渲染动作暂时不可用。
        try:
            affection_status = _schedule_pet_affection(
                runtime_loop,
                runtime.affection,
                pending_pet_affection,
                generation=feedback_generation,
                zone=interaction_zone,
                phrase=phrase,
            )
        except Exception as exc:
            affection_status = {"status": "unavailable"}
            logger.warning("pet affection scheduling failed: %s", type(exc).__name__)
        result["affection"] = affection_status
        if console is not None:
            console.set_pet_feedback(
                {
                    "zone": interaction_zone,
                    "part": part or interaction_zone,
                    "phrase": phrase,
                    "mood": mood,
                    "expression": expression,
                    "motion": motion,
                    "affection": affection_status,
                }
            )
        _set_pet_visual_feedback(
            {
                "zone": interaction_zone,
                "part": part or interaction_zone,
                "phrase": phrase,
                "mood": mood,
                "expression": expression,
                "motion": motion,
                "affection": affection_status,
            }
        )
        last_pet_feedback = {
            "zone": interaction_zone,
            "part": part or interaction_zone,
            "expression": expression,
            "motion": motion,
            "phrase": phrase,
            "mood": mood,
            "active": True,
            "affection": affection_status,
        }
        if affection_status.get("status") != "pending":
            pet_feedback_notice = f"点击互动：{phrase}（好感度未更新）"

        def clear_feedback() -> None:
            """在没有新点击时收起直接气泡并恢复当前回合内容。"""

            nonlocal last_pet_feedback, pet_feedback_timer, last_presented_speaking
            if feedback_generation != pet_feedback_generation:
                return
            # 点击反馈清理可能在 app.exec() 结束后的 DeferredDelete 阶段迟到；
            # 页面已经失效时不能再把 set_speech 投递到已 discarded 的 Chromium。
            if web_host is not None:
                renderer = getattr(web_host, "renderer", None)
                if renderer is not None and not bool(getattr(renderer, "page_ready", True)):
                    pet_feedback_timer = None
                    return
            pet_feedback_timer = None
            try:
                snapshot = presentation.clear_direct_speech()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.debug("failed to clear direct pet feedback", exc_info=True)
                return
            _set_pet_visual_feedback(None, visible=False)
            target = window if window is not None else web_host
            setter = getattr(target, "set_speech", None)
            if callable(setter):
                next_sentence = None
                sentence_acker = getattr(presentation, "ack_sentence", None)
                try:
                    next_sentence_peeker = getattr(presentation, "peek_next_sentence", None)
                    next_sentence_getter = getattr(presentation, "pop_next_sentence", None)
                    if callable(next_sentence_peeker) and callable(sentence_acker):
                        next_sentence = next_sentence_peeker()
                    elif callable(next_sentence_getter):
                        next_sentence = next_sentence_getter()
                    else:
                        next_sentence = None
                    rendered_text = (
                        _friendly_stream_text(getattr(next_sentence, "text", ""))
                        if next_sentence is not None
                        else _pet_speech_text(snapshot)
                    )
                    interaction_holder = getattr(runtime, "interaction_state", None)
                    interaction_view = getattr(interaction_holder, "snapshot", snapshot)
                    speaking = _tts_speaking_for_snapshot(
                        interaction_view,
                        active=tts_activity.active,
                        pending_sentence=next_sentence is not None,
                    )
                    setter_result = setter(
                        rendered_text,
                        mood=snapshot.rendered_mood,
                        visible=bool(rendered_text),
                        speaking=speaking,
                    )
                    if (
                        setter_result is not False
                        and next_sentence is not None
                        and callable(sentence_acker)
                    ):
                        sentence_acker(next_sentence)
                    if setter_result is not False:
                        last_presented_speaking = speaking
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to clear pet click feedback", exc_info=True)
            if isinstance(last_pet_feedback, dict):
                # 保留最近一次反馈给 Web/控制台历史状态；仅收起桌宠上的
                # 瞬时气泡，不让好感度和部位结果回到空值。
                last_pet_feedback["active"] = False
                if console is not None:
                    console.set_pet_feedback(last_pet_feedback)

        # 使用 app-owned 单次计时器，shutdown 前可以停止；匿名 singleShot 会在
        # Qt 处理 DeferredDelete 时迟到执行，重新写入已销毁的 WebEngine 页面。
        if pet_feedback_timer is not None:
            try:
                pet_feedback_timer.stop()
                pet_feedback_timer.deleteLater()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
        pet_feedback_timer = QTimer(app)
        pet_feedback_timer.setSingleShot(True)
        pet_feedback_timer.timeout.connect(clear_feedback)
        pet_feedback_timer.start(2400)
        if console is not None:
            console.set_status(pet_feedback_notice)
            _console_result("点击互动", result)
        else:
            logger.debug(
                "pet click interaction: %s",
                json.dumps(result, ensure_ascii=False, default=str),
            )

    def _web_affection_state() -> dict[str, object]:
        """读取持久好感度；失败时只回退到最近安全快照。"""

        try:
            current = int(runtime.affection.get())
            threshold, tier, description = runtime.affection.get_affection_tier()
            return {
                "current": current,
                "tier": tier,
                "threshold": threshold,
                "description": description,
            }
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            if isinstance(last_pet_feedback, Mapping):
                value = last_pet_feedback.get("affection")
                if isinstance(value, Mapping):
                    return dict(value)
            return {"current": 0, "tier": "未知", "status": "unavailable"}

    def _finish_conversation_receipt(context: object, result: object) -> None:
        """在 Qt 线程把对话 Future 终态写回原操作，不泄漏异常正文。"""

        nonlocal web_last_operation
        operation_id = pending_conversation_receipts.pop(context, "")
        if not operation_id or web_last_operation.get("id") != operation_id:
            return
        web_last_operation = _apply_conversation_operation_receipt(
            web_last_operation,
            operation_id,
            result,
        )
        _append_web_timeline(
            web_last_operation.get("kind", "submit_text"),
            web_last_operation.get("status", "completed"),
            web_last_operation.get("message", ""),
        )

    def _web_configuration_state() -> dict[str, object]:
        """生成配置文件连接/自动重载摘要，不暴露路径、正文或敏感值。"""

        # 配置 watcher 热加载后，ApplicationRuntime 持有的对象才是当前
        # 生效快照；闭包里的启动配置不会自动替换。若继续读取旧快照，
        # Web 卡片会把已经关闭/开启的自动重载显示成旧状态。
        active_configuration = getattr(runtime, "configuration", configuration)
        values = getattr(active_configuration, "values", {})
        config_values = values.get("config", {}) if isinstance(values, Mapping) else {}
        reload_values = (
            config_values.get("reload", {}) if isinstance(config_values, Mapping) else {}
        )
        if not isinstance(reload_values, Mapping):
            reload_values = {}
        try:
            watcher_enabled = parse_bool(
                reload_values.get("enabled", True),
                field_name="config.reload.enabled",
                default=True,
            )
        except (TypeError, ValueError, ConfigurationError):
            watcher_enabled = False
        watcher = getattr(runtime, "configuration_watcher", None)
        watcher_running = bool(getattr(watcher, "running", False)) if watcher is not None else False
        connected = bool(getattr(active_configuration, "path", None))
        status_snapshot: Mapping[str, object] = {}
        status_getter = getattr(watcher, "status", None) if watcher is not None else None
        if callable(status_getter):
            try:
                value = status_getter()
                if isinstance(value, Mapping):
                    status_snapshot = value
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                status_snapshot = {}
        raw_status = str(status_snapshot.get("status", "") or "").strip().lower()
        if not connected or watcher is None:
            status = "unavailable"
        elif not watcher_enabled:
            status = "stopped"
        elif status_snapshot.get("error"):
            status = "configuration_rejected"
        elif bool(status_snapshot.get("pending", False)):
            status = "deferred"
        elif raw_status in {
            "reloaded",
            "restart_required",
            "deferred",
            "rejected",
            "configuration_rejected",
            "unavailable",
        }:
            status = "configuration_rejected" if raw_status == "rejected" else raw_status
        elif watcher_running:
            status = "running"
        else:
            status = "stopped"
        try:
            generation = max(0, int(status_snapshot.get("generation", 0) or 0))
        except (TypeError, ValueError, OverflowError):
            generation = 0
        messages = {
            "unavailable": "自动重载服务尚未运行",
            "stopped": "自动重载已关闭；可在配置中心开启",
            "deferred": "正在等待文件稳定后自动重载",
            "configuration_rejected": "配置变更未应用，请打开配置中心校验",
            "reloaded": "配置文件已自动重载并应用",
            "restart_required": "配置已读取，部分设置需要重启",
            "running": "配置文件已连接，修改后会自动校验并应用",
        }
        return {
            "connected": connected,
            "watcher_enabled": watcher_enabled,
            "watcher_running": watcher_running,
            "auto_reload": bool(watcher_enabled and watcher_running),
            "status": status,
            "generation": generation,
            "message": messages.get(status, "配置文件状态暂不可用"),
        }

    def _web_memory_state() -> dict[str, object]:
        """生成记忆策略摘要，不把记忆内容或数据库位置送入 Web。"""

        memory_service = getattr(runtime, "memory", None)
        status_getter = getattr(memory_service, "status", None)
        if not callable(status_getter):
            return {"enabled": False, "status": "unavailable"}
        try:
            value = status_getter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {"enabled": False, "status": "unavailable"}
        if not isinstance(value, Mapping):
            return {"enabled": False, "status": "unavailable"}
        summarizer = getattr(runtime, "memory_summarizer", None)
        summary_status_getter = getattr(summarizer, "status", None)
        summary_status: Mapping[str, object] = {}
        if callable(summary_status_getter):
            try:
                summary_value = summary_status_getter()
                summary_public = getattr(summary_value, "public", None)
                if callable(summary_public):
                    summary_value = summary_public()
                if isinstance(summary_value, Mapping):
                    summary_status = summary_value
            except (AttributeError, RuntimeError, TypeError, ValueError):
                summary_status = {}
        return {
            "enabled": bool(value.get("enabled", False)),
            "status": "ready" if bool(value.get("enabled", False)) else "disabled",
            "recall_limit": value.get("recall_limit", 0),
            "context_max_chars": value.get("context_max_chars", 0),
            "max_memories": value.get("max_memories", 0),
            "consolidation_enabled": bool(value.get("consolidation_enabled", False)),
            "summarization_enabled": bool(value.get("summarization_enabled", False)),
            "summary_running": bool(summary_status.get("running", False)),
            "summary_busy": bool(summary_status.get("busy", False)),
            "summary_completed": summary_status.get("completed", 0),
            "summary_last_status": summary_status.get("last_status", "idle"),
            "extraction_pending": summary_status.get("extraction_pending", 0),
            "extraction_completed": summary_status.get("extraction_completed", 0),
            "extraction_last_status": summary_status.get("extraction_last_status", "idle"),
            "vector_index_size": value.get("vector_index_size", 0),
            "lexical_index": value.get("lexical_index", "sparse_fallback"),
            "message": (
                "记忆已启用，新的对话会按优先级召回"
                if bool(value.get("enabled", False))
                else "记忆已关闭"
            ),
        }

    def _safe_web_result(kind: str, result: object) -> dict[str, object]:
        """把动作结果压缩为可显示回执，并生成不含内部 ID 的操作编号。"""

        nonlocal web_operation_sequence, web_last_operation, pending_web_operation_id
        web_operation_sequence += 1
        operation_id = f"ui-{web_operation_sequence}"
        if isinstance(result, Mapping):
            status = str(result.get("status", result.get("state", "completed")) or "completed")
            message = _safe_ui_text(
                result.get("message", result.get("reason", result.get("detail", ""))),
                limit=400,
            )
            safe: dict[str, object] = {
                "status": status[:32].lower(),
                "message": message[:400],
                "operation_id": operation_id,
            }
            for key in ("part", "name", "detail"):
                value = result.get(key)
                if isinstance(value, (str, int, float, bool)):
                    safe[key] = _safe_ui_text(value, limit=160) if isinstance(value, str) else value
        else:
            safe = {
                "status": "requested",
                "message": "操作已提交",
                "operation_id": operation_id,
            }
        web_last_operation = {
            "id": operation_id,
            "kind": kind,
            "status": safe["status"],
            "message": safe.get("message", ""),
            "updated_at": monotonic(),
        }
        _append_web_timeline(kind, safe["status"], safe.get("message", ""))
        if kind in {"submit_text", "retry"} and safe["status"] == "requested":
            if active_context is not None:
                pending_conversation_receipts[active_context] = operation_id
                pending_web_operation_id = None
            else:
                # 连续发送正在等待旧回合取消；等下一次 Qt tick 真正创建
                # 新 context 后再绑定这个 operation id。
                pending_web_operation_id = operation_id
        return safe

    def web_control_state() -> dict[str, object]:
        """构造公开网页控制面状态，不携带审批/回合内部字段。"""

        interaction_state = getattr(runtime, "interaction_state", None)
        snapshot = getattr(interaction_state, "snapshot", None)
        interaction = (
            snapshot.to_public_mapping()
            if snapshot is not None and callable(getattr(snapshot, "to_public_mapping", None))
            else {}
        )
        target = window if window is not None else web_host
        renderer = getattr(target, "renderer", None)
        capabilities = getattr(renderer, "capabilities", None)
        active_model_key = ""
        probe = getattr(renderer, "probe", None)
        model_path = getattr(probe, "model_path", None)
        if model_path is not None:
            try:
                relative_model = Path(model_path).resolve().relative_to(resource_root).as_posix()
            except (OSError, RuntimeError, TypeError, ValueError):
                relative_model = ""
            if relative_model in renderer_model_choices:
                active_model_key = relative_model
        if not active_model_key:
            active_configuration = getattr(runtime, "configuration", configuration)
            active_values = getattr(active_configuration, "values", {})
            active_rendering = (
                active_values.get("rendering") if isinstance(active_values, Mapping) else None
            )
            configured_model = (
                active_rendering.get("model") if isinstance(active_rendering, Mapping) else ""
            )
            if configured_model in renderer_model_choices:
                active_model_key = str(configured_model)
        if not active_model_key:
            active_model_key = renderer_model_key
        renderer_state = {
            "backend": getattr(capabilities, "backend", ""),
            "available": bool(getattr(capabilities, "available", False)),
            "message": getattr(capabilities, "message", ""),
            "model": active_model_key,
            "models": tuple(renderer_model_choices)[:32],
        }
        locked_getter = getattr(target, "is_window_locked", None)
        topmost_status = topmost_controller.status()
        shape_getter = getattr(target, "surface_mask_status", None)
        shape_status: dict[str, object] = {}
        if callable(shape_getter):
            try:
                raw_shape = shape_getter()
                if isinstance(raw_shape, Mapping):
                    shape_status = {
                        "ready": bool(raw_shape.get("ready", False)),
                        "input_ready": bool(raw_shape.get("input_ready", False)),
                        "status": str(raw_shape.get("status", "") or "")[:32].lower(),
                    }
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                shape_status = {}
        feedback = dict(last_pet_feedback) if isinstance(last_pet_feedback, Mapping) else None
        if feedback is not None:
            feedback["active"] = monotonic() < pet_feedback_hold_until
        position = current_pet_position()
        try:
            tts_diagnostics = runtime.tts_diagnostics()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            tts_diagnostics = {}
        try:
            activity_status = runtime.activity_status()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            activity_status = {}
        active_configuration = getattr(runtime, "configuration", configuration)
        active_values = getattr(active_configuration, "values", {})
        try:
            safe_theme_values = active_values if isinstance(active_values, Mapping) else {}
            theme_tokens = md3_web_tokens(theme_from_configuration(safe_theme_values))
        except (AttributeError, TypeError, ValueError):
            theme_tokens = {}
        return {
            "revision": web_revision,
            # 不把当前时间直接写入快照，否则 40ms UI 轮询会让每一帧都被
            # 误判为新状态；revision 只在内容真正变化时推进。
            "updated_at": web_last_operation.get("updated_at", 0.0),
            "connection_state": "local",
            "connection_message": "本地控制通道已连接",
            "interaction": interaction,
            "model_channels": _public_model_channels(getattr(runtime, "router", None)),
            "model_image": _public_model_image_status(getattr(runtime, "router", None)),
            "configuration": _web_configuration_state(),
            "theme": theme_tokens,
            "memory": _web_memory_state(),
            "activity": dict(activity_status) if isinstance(activity_status, Mapping) else {},
            "renderer": renderer_state,
            "window": {
                "locked": bool(locked_getter()) if callable(locked_getter) else False,
                "click_through": bool(getattr(target, "_click_through", False)),
                "always_on_top": bool(topmost_status.get("enabled", False)),
                "always_on_top_status": topmost_status,
                "position": (
                    {"x": int(position[0]), "y": int(position[1])} if position is not None else None
                ),
                "input_shape": shape_status,
            },
            "affection": _web_affection_state(),
            "feedback": feedback,
            "tts": {**dict(web_tts_state), **dict(tts_diagnostics)},
            "observation": dict(web_observation),
            "timeline": tuple(web_timeline),
            "parts": [
                {"id": "head", "label": "摸摸头", "description": "猫猫头反馈"},
                {"id": "body", "label": "拍拍身体", "description": "身体反馈"},
                {"id": "lower_left", "label": "左边", "description": "左侧反馈"},
                {"id": "lower_right", "label": "右边", "description": "右侧反馈"},
            ],
            "operation": dict(web_last_operation),
            "capabilities": {
                "expressions": getattr(capabilities, "expressions", ()),
                "motions": getattr(capabilities, "motions", ()),
            },
        }

    def _resolve_web_capability_name(kind: str, value: object) -> str:
        """把网页公开的能力令牌解析为当前渲染器的内部名称。"""

        token = str(value or "").strip().lower()
        prefix = f"cap-{kind}-"
        if not token.startswith(prefix):
            return ""
        index_text = token[len(prefix) :]
        if not index_text.isdigit() or len(index_text) > 2:
            return ""
        index = int(index_text)
        if index < 0 or index >= 32:
            return ""
        target = window if window is not None else web_host
        renderer = getattr(target, "renderer", None)
        capabilities = getattr(renderer, "capabilities", None)
        field = "expressions" if kind == "expression" else "motions"
        values = getattr(capabilities, field, ())
        try:
            rows = tuple(values)
        except TypeError:
            return ""
        if index >= len(rows):
            return ""
        return str(rows[index] or "").strip()

    def _web_expression_request(payload: Mapping[str, object]) -> object:
        """解码公开表情令牌并提交结构化编排。"""

        try:
            request = _decode_public_expression_request(payload, _resolve_web_capability_name)
            return runtime.pet_controller.set_expression_request(request)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "表情编排参数无效"}

    def _web_motion_request(payload: Mapping[str, object]) -> object:
        """解码公开动作令牌并提交结构化动作请求。"""

        try:
            request = _decode_public_motion_request(payload, _resolve_web_capability_name)
            return runtime.pet_controller.play_motion_request(request)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "动作编排参数无效"}

    def web_control_action(kind: str, payload: Mapping[str, object]) -> object:
        """执行公开网页动作；审批动作按当前状态解析内部标识。"""

        nonlocal pending_cancel_future, pending_cancel_operation_id
        normalized = str(kind or "").strip().lower()
        if normalized == "open_config":
            show_console()
            if console is None or not console.show_settings():
                return _safe_web_result(
                    normalized, {"status": "unavailable", "reason": "配置中心不可用"}
                )
            return _safe_web_result(
                normalized, {"status": "completed", "message": "已打开配置中心"}
            )
        if normalized == "restart_application":
            return _safe_web_result(normalized, restart_application())
        if normalized == "select_renderer_backend":
            return _safe_web_result(
                normalized,
                select_renderer_backend(payload.get("backend")),
            )
        if normalized == "configure_model":
            show_console()
            if console is not None:
                console.show_model_settings()
            return _safe_web_result(
                normalized, {"status": "completed", "message": "已打开模型配置向导"}
            )
        if normalized == "open_input":
            prompt_text()
            return _safe_web_result(normalized, {"status": "completed", "message": "已打开输入框"})
        if normalized == "select_model_channel":
            conversation = getattr(runtime, "conversation", None)
            active_getter = getattr(conversation, "has_active_conversation", None)
            if callable(active_getter):
                try:
                    active = bool(active_getter())
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    active = True
                if active:
                    return _safe_web_result(
                        normalized,
                        {"status": "unavailable", "reason": "当前回复进行中，请完成后再切换模型"},
                    )
            channel_id = str(payload.get("channel_id", "") or "").strip()
            router = getattr(runtime, "router", None)
            requested_model = str(payload.get("model", "") or "").strip()
            selector = (
                getattr(router, "select_model", None)
                if requested_model
                else getattr(router, "select_channel", None)
            )
            if (not channel_id and not requested_model) or not callable(selector):
                return _safe_web_result(
                    normalized, {"status": "unavailable", "reason": "模型渠道不可用"}
                )
            try:
                result = (
                    selector(requested_model, task="dialogue", channel_id=channel_id)
                    if requested_model
                    else selector(channel_id, task="dialogue")
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                result = {"status": "unavailable", "reason": "模型渠道暂时不可用"}
            if isinstance(result, Mapping) and result.get("status") == "updated":
                try:
                    diagnostics = router.diagnostics("dialogue")
                    interaction_state = getattr(runtime, "interaction_state", None)
                    setter = getattr(interaction_state, "set_model_diagnostics", None)
                    if callable(setter):
                        setter(diagnostics)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    logger.debug("model channel diagnostics refresh failed", exc_info=True)
            return _safe_web_result(normalized, result)
        if normalized == "select_tts_profile":
            selector = getattr(runtime, "select_tts_profile", None)
            if not callable(selector):
                return _safe_web_result(
                    normalized, {"status": "unavailable", "reason": "语音 profile 不可用"}
                )
            result = selector(
                str(payload.get("profile", "") or "").strip() or None,
                language=(str(payload.get("language", "") or "").strip() or None),
            )
            return _safe_web_result(normalized, result)
        if normalized == "select_tts_language":
            setter = getattr(runtime, "set_tts_language", None)
            if not callable(setter):
                return _safe_web_result(
                    normalized, {"status": "unavailable", "reason": "输出语言不可用"}
                )
            result = setter(str(payload.get("language", "") or "").strip())
            return _safe_web_result(normalized, result)
        if normalized == "submit_text":
            text = str(payload.get("text", "") or "").strip()[:2000]
            if not text:
                return _safe_web_result(normalized, {"status": "unavailable", "reason": "消息为空"})
            ready, model_message = _model_channel_status(runtime)
            if not ready:
                return _safe_web_result(
                    normalized,
                    {"status": "unavailable", "reason": model_message},
                )
            submit_text(text)
            return _safe_web_result(normalized, {"status": "requested", "message": "消息已提交"})
        if normalized == "stop":
            cancellation = stop_conversation()
            if cancellation is not None and not isinstance(cancellation, Mapping):
                pending_cancel_future = cancellation
                safe = _safe_web_result(
                    normalized,
                    {"status": "requested", "message": "正在停止"},
                )
                pending_cancel_operation_id = str(safe.get("operation_id", "") or "")
                return safe
            if cancellation is None:
                return _safe_web_result(
                    normalized,
                    {"status": "unavailable", "reason": "运行时尚未启动，无法停止对话"},
                )
            return _safe_web_result(
                normalized,
                cancellation,
            )
        if normalized == "retry":
            return _safe_web_result(normalized, retry_last_conversation())
        if normalized in {"approve", "grant_session", "deny"}:
            state = getattr(runtime, "interaction_state", None)
            resolver = getattr(state, "action_for_public_kind", None)
            action = resolver(normalized) if callable(resolver) else None
            action_payload = getattr(action, "payload", {}) if action is not None else {}
            approval_id = (
                str(action_payload.get("approval_id", "") or "").strip()
                if isinstance(action_payload, Mapping)
                else ""
            )
            if not approval_id:
                return _safe_web_result(
                    normalized, {"status": "unavailable", "reason": "当前确认已过期"}
                )
            if normalized == "approve":
                result = approve_console_approval(approval_id, grant_session=False)
            elif normalized == "grant_session":
                result = approve_console_approval(approval_id, grant_session=True)
            else:
                result = deny_console_approval(approval_id)
            return _safe_web_result(normalized, result)
        if normalized == "pet_part":
            part = str(payload.get("part", "") or "").strip().lower()
            if part not in {"head", "body", "lower_left", "lower_right"}:
                return _safe_web_result(
                    normalized, {"status": "unavailable", "reason": "部位不可用"}
                )
            handle_pet_click({"part": part, "zone": part, "hit": True})
            return _safe_web_result(
                normalized,
                {"status": "completed", "part": part, "message": "已触发部位反馈"},
            )
        if normalized == "expression":
            name = _resolve_web_capability_name(normalized, payload.get("name"))
            result = (
                runtime.pet_controller.set_expression(name)
                if name
                else {"status": "unavailable", "reason": "表情为空"}
            )
            return _safe_web_result(normalized, result)
        if normalized == "motion":
            name = _resolve_web_capability_name(normalized, payload.get("name"))
            result = (
                runtime.pet_controller.play_motion(name)
                if name
                else {"status": "unavailable", "reason": "动作为空"}
            )
            return _safe_web_result(normalized, result)
        if normalized == "expression_request":
            return _safe_web_result(normalized, _web_expression_request(payload))
        if normalized == "motion_request":
            return _safe_web_result(normalized, _web_motion_request(payload))
        if normalized == "show_pet":
            return _safe_web_result(normalized, show_pet())
        if normalized == "toggle_visibility":
            return _safe_web_result(normalized, toggle_pet_visibility())
        if normalized == "nudge_pet":
            direction = str(payload.get("direction", "") or "").strip().lower()
            delta = {
                "left": (-48, 0),
                "right": (48, 0),
                "up": (0, -48),
                "down": (0, 48),
            }.get(direction)
            position = current_pet_position()
            bounds = current_pet_movement_bounds()
            if delta is None or position is None or bounds is None:
                return _safe_web_result(
                    normalized, {"status": "unavailable", "reason": "当前位置不可用"}
                )
            left, top, right, bottom = bounds
            next_x = max(left, min(right, position[0] + delta[0]))
            next_y = max(top, min(bottom, position[1] + delta[1]))
            return _safe_web_result(
                normalized,
                runtime.pet_controller.move_to(next_x, next_y),
            )
        if normalized == "center_pet":
            return _safe_web_result(normalized, center_pet())
        if normalized == "set_display_size":
            preset = str(payload.get("preset", "") or "").strip().lower()
            if preset not in {"small", "standard", "large"}:
                return _safe_web_result(
                    normalized, {"status": "unavailable", "reason": "显示大小不可用"}
                )
            return _safe_web_result(normalized, runtime.pet_controller.set_display_size(preset))
        if normalized == "toggle_window_lock":
            return _safe_web_result(normalized, toggle_window_lock())
        if normalized == "toggle_always_on_top":
            return _safe_web_result(normalized, toggle_always_on_top())
        if normalized == "toggle_click_through":
            target = window if window is not None else web_host
            current = bool(getattr(target, "_click_through", False))
            return _safe_web_result(normalized, set_console_click_through(not current))
        if normalized == "restore_click_through":
            return _safe_web_result(normalized, restore_click_through())
        if normalized == "read_foreground_window":
            return _safe_web_result(normalized, read_foreground_window())
        if normalized == "read_processes":
            return _safe_web_result(normalized, read_processes())
        return _safe_web_result(
            normalized, {"status": "unavailable", "reason": "网页动作不在允许列表"}
        )

    def show_web_console() -> object:
        """打开公开网页控制台；没有 WebEngine 时保留 Qt 控制台入口。"""

        nonlocal web_console, console_recovery_available

        def on_web_console_hidden() -> None:
            """隐藏网页控制台前确保点击穿透仍有恢复入口。"""

            nonlocal console_recovery_available
            if shutting_down:
                console_recovery_available = False
                set_ui_interaction_lock(False)
                return
            if console is not None and console.isVisible():
                console_recovery_available = True
                return
            result = restore_pet_input()
            if _click_through_operation_succeeded(result, enabled=False):
                console_recovery_available = False
                restore_tight_layout_pet()
                set_ui_interaction_lock(False)
                return
            console_recovery_available = True
            if web_console is not None:
                web_console.show_and_focus()

        if not web_console_available:
            show_console()
            if console is not None:
                console.set_status("网页控制台不可用，已打开 Qt 控制台")
            return {"status": "unavailable", "reason": "WebEngine 不可用"}
        try:
            if web_console is None:
                web_console = WebControlSurfaceWindow(web_control_state, web_control_action)
                web_console.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
                hidden_signal = getattr(web_console, "hidden", None)
                if hidden_signal is not None and callable(getattr(hidden_signal, "connect", None)):
                    hidden_signal.connect(on_web_console_hidden)
            console_recovery_available = True
            set_ui_interaction_lock(True)
            web_console.set_state(web_control_state())
            web_console.show_and_focus()
            _place_widget_adjacent(
                web_console,
                window if window is not None else getattr(web_host, "view", None),
                anchor=current_pet_position(),
                on_tight_layout=None,
            )
            return {"status": "available"}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("网页控制台启动失败：%s", type(exc).__name__)
            return {"status": "unavailable", "reason": type(exc).__name__}

    def _invoke_local_web_callback(function: Callable[..., object], *args: object) -> object:
        """把 HTTP 工作线程的状态/动作调用同步转发到 Qt 主线程。"""

        if shutting_down:
            raise RuntimeError("local web server is shutting down")
        result = dispatcher.invoke(function, *args)
        if inspect.isawaitable(result):
            # ``QtMainThreadDispatcher.invoke`` 在跨线程时返回一个等待协程；
            # HTTP 处理线程没有自己的业务事件循环，因此仅在该线程运行
            # 一个短生命周期 loop 等待 Qt 完成回调，不触碰运行时 loop。
            return asyncio.run(result)
        return result

    def local_web_state_provider() -> Mapping[str, object]:
        value = _invoke_local_web_callback(web_control_state)
        if not isinstance(value, Mapping):
            raise RuntimeError("local web state is unavailable")
        return value

    def local_web_action_callback(kind: str, payload: Mapping[str, object]) -> object:
        return _invoke_local_web_callback(web_control_action, kind, payload)

    def start_local_web_server() -> None:
        """按显式 YAML 开关启动本机控制面；默认路径不创建监听套接字。"""

        nonlocal local_web_server, local_web_sink_notified
        web_values = configuration.values.get("web", {})
        local_api_values = (
            web_values.get("local_api", {}) if isinstance(web_values, Mapping) else {}
        )
        if not isinstance(local_api_values, Mapping):
            raise ConfigurationError("web.local_api must be a mapping")
        enabled = parse_bool(
            local_api_values.get("enabled"),
            field_name="web.local_api.enabled",
            default=False,
        )
        if not enabled:
            return
        if local_web_server_sink is None:
            # 没有进程内宿主持有 capability 时不创建监听套接字；仅在日志
            # 中给出明确诊断，避免用户误以为配置已开放可用控制面。
            logger.warning(
                "web.local_api.enabled=true 但未提供 local_web_server_sink；控制面保持关闭"
            )
            return
        host = local_api_values.get("host", "127.0.0.1")
        port = local_api_values.get("port", 0)
        events = parse_bool(
            local_api_values.get("events"),
            field_name="web.local_api.events",
            default=True,
        )
        event_limit = local_api_values.get("event_limit", 64)
        try:
            local_web_server = LocalWebServer(
                local_web_state_provider,
                local_web_action_callback,
                host=host,
                port=port,
                enable_events=events,
                event_limit=event_limit,
                logger=logger.debug,
            ).start()
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigurationError("web.local_api could not start") from exc
        try:
            local_web_sink_notified = False
            local_web_server_sink(local_web_server)
        except BaseException:
            # 宿主回调失败时立即回收监听，不能留下一个没有持有者的
            # capability 服务。
            stop_local_web_server()
            raise
        logger.info("本地 Web 控制面已启用：%s", local_web_server.base_url)

    def flush_pending_resubmit(token: int) -> None:
        """在短暂合并窗口结束后只提交最后一条待发送消息。"""

        nonlocal pending_submit_text, pending_resubmit_scheduled, pending_web_operation_id
        if shutting_down or token != pending_resubmit_token:
            return
        pending_resubmit_scheduled = False
        queued_text = pending_submit_text
        pending_submit_text = None
        if not queued_text:
            return
        submit_text(queued_text)
        if active_context is not None and pending_web_operation_id:
            pending_conversation_receipts[active_context] = pending_web_operation_id
            pending_web_operation_id = None

    def _pet_speech_text(snapshot: object) -> str:
        """返回桌宠气泡应显示的最近完整句，而非累计正文。"""

        rendered = _friendly_stream_text(getattr(snapshot, "rendered_text", ""))
        direct = _friendly_stream_text(getattr(snapshot, "direct_text", ""))
        if direct:
            return rendered
        # 工具审批、运行失败是需要立即告知用户的状态；普通正文到达时
        # PresentationService 会清除旧 tool_status，避免状态遮住新句子。
        tool_status = _friendly_stream_text(getattr(snapshot, "tool_status", ""))
        if tool_status:
            return tool_status
        sentence = _friendly_stream_text(getattr(snapshot, "speech_text", ""))
        if sentence:
            return sentence
        # 首句尚未结束时不把半句送入桌宠气泡；但碎碎念仍需可见。
        raw_text = _friendly_stream_text(getattr(snapshot, "text", ""))
        if raw_text:
            murmur = _friendly_stream_text(getattr(snapshot, "murmur", ""))
            return murmur
        return rendered

    def update_bubble() -> None:
        if shutting_down:
            return
        nonlocal last_presented_snapshot, last_presented_speaking, pending_runtime_reload
        nonlocal pending_cancel_future, pending_cancel_operation_id
        nonlocal pending_submit_text, pending_resubmit_scheduled, pending_resubmit_token
        nonlocal pet_feedback_notice
        nonlocal last_web_control_state_json, web_revision, pending_web_operation_id
        nonlocal web_last_operation
        nonlocal last_applied_theme_signature
        nonlocal last_microphone_signature
        nonlocal last_audio_output_device_id
        active_configuration = getattr(runtime, "configuration", configuration)
        active_values = getattr(active_configuration, "values", {})
        active_ui = active_values.get("ui", {}) if isinstance(active_values, Mapping) else {}
        active_theme = active_ui.get("theme", {}) if isinstance(active_ui, Mapping) else {}
        theme_signature = json.dumps(active_theme, ensure_ascii=False, sort_keys=True, default=str)
        if theme_signature != last_applied_theme_signature:
            if console is not None:
                apply_console_theme = getattr(console, "apply_theme_configuration", None)
                if callable(apply_console_theme):
                    try:
                        apply_console_theme(active_values)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        logger.warning("控制台主题热重载失败", exc_info=True)
                    else:
                        last_applied_theme_signature = theme_signature
                else:
                    last_applied_theme_signature = theme_signature
            else:
                last_applied_theme_signature = theme_signature
        active_tts = active_values.get("tts", {}) if isinstance(active_values, Mapping) else {}
        active_tts = active_tts if isinstance(active_tts, Mapping) else {}
        configured_output_device_id = active_tts.get("output_device_id", "")
        if not isinstance(configured_output_device_id, str):
            configured_output_device_id = last_audio_output_device_id
        if configured_output_device_id != last_audio_output_device_id:
            try:
                audio_player.set_output_device_id(configured_output_device_id)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.warning("扬声器输出设备热重载失败")
            else:
                last_audio_output_device_id = configured_output_device_id
        active_asr = active_values.get("asr", {}) if isinstance(active_values, Mapping) else {}
        active_asr = active_asr if isinstance(active_asr, Mapping) else {}
        try:
            active_capture = asr_capture_configuration(active_asr)
            capture_signature = (
                bool(active_capture["enabled"] and runtime.asr.enabled),
                int(runtime.asr.max_audio_bytes),
                float(active_capture["max_duration_seconds"]),
                str(active_capture["device_id"]),
            )
        except (AttributeError, ConfigurationError, TypeError, ValueError, OverflowError):
            capture_signature = last_microphone_signature
        if capture_signature != last_microphone_signature:
            try:
                microphone_capture.configure(
                    enabled=bool(capture_signature[0]),
                    max_audio_bytes=int(capture_signature[1]),
                    max_duration_seconds=float(capture_signature[2]),
                    device_id=str(capture_signature[3]),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
                logger.warning("麦克风采集配置热重载失败")
            else:
                last_microphone_signature = capture_signature
        _poll_console_reads()
        _poll_console_approvals()
        _poll_pet_affection()
        _refresh_console_approvals()
        if web_console is not None or local_web_server is not None:
            web_state = web_control_state()
            web_state_json = json.dumps(web_state, ensure_ascii=False, sort_keys=True, default=str)
            if web_state_json != last_web_control_state_json:
                web_revision += 1
                web_state["revision"] = web_revision
                last_web_control_state_json = json.dumps(
                    web_state, ensure_ascii=False, sort_keys=True, default=str
                )
                if web_console is not None:
                    web_console.set_state(web_state)
                if local_web_server is not None and local_web_server.events_enabled:
                    try:
                        local_web_server.publish_event({"type": "state", "state": web_state})
                    except (RuntimeError, TypeError, ValueError):
                        logger.debug("local web state event publish failed", exc_info=True)
        # 新消息不能与异步取消竞态创建 context：取消协程先推进
        # GenerationGate，再由这里在 Qt 线程启动下一回合。
        if pending_cancel_future is not None and pending_cancel_future.done():
            cancel_future = pending_cancel_future
            pending_cancel_future = None
            cancel_operation_id = pending_cancel_operation_id
            pending_cancel_operation_id = None
            queued_text = pending_submit_text
            try:
                cancel_future.result()
            except Exception as exc:
                logger.warning(
                    "conversation cancellation before resubmit failed: %s",
                    type(exc).__name__,
                )
                cancel_result: object = {"status": "failed"}
            else:
                cancel_result = {"status": "cancelled"}
            if cancel_operation_id and web_last_operation.get("id") == cancel_operation_id:
                web_last_operation = _apply_conversation_operation_receipt(
                    web_last_operation,
                    cancel_operation_id,
                    cancel_result,
                )
            if queued_text:
                # 取消完成与后续 Qt 定时器可能交错；保留一个很短的合并窗，
                # 让同一批快速输入只启动最后一条，而不会把中间消息变成
                # 额外模型回合。
                pending_resubmit_token += 1
                pending_resubmit_scheduled = True
                token = pending_resubmit_token
                try:
                    QTimer.singleShot(
                        _RESUBMIT_DEBOUNCE_MS,
                        lambda selected=token: flush_pending_resubmit(selected),
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    flush_pending_resubmit(token)
        feedback_active = bool(pet_feedback_notice) and monotonic() < pet_feedback_hold_until
        if not feedback_active:
            pet_feedback_notice = ""
        interaction_state = getattr(runtime, "interaction_state", None)
        interaction_snapshot = getattr(interaction_state, "snapshot", None)
        if console is not None:
            topmost_status_setter = getattr(console, "set_always_on_top_status", None)
            if callable(topmost_status_setter):
                try:
                    topmost_status_setter(topmost_controller.status())
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to refresh console topmost status", exc_info=True)
        if console is not None and interaction_snapshot is not None:
            tts_options_setter = getattr(console, "set_tts_options", None)
            if callable(tts_options_setter):
                try:
                    tts_state = runtime.tts_diagnostics()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    tts_state = {}
                if isinstance(tts_state, Mapping):
                    tts_options_setter(
                        tts_state.get("profiles", ()),
                        language=tts_state.get("language", "zh"),
                        active_profile=tts_state.get("active_profile", ""),
                        health=(
                            tts_state.get("health")
                            if isinstance(tts_state.get("health"), Mapping)
                            else None
                        ),
                    )
            pending = getattr(interaction_snapshot, "pending_approvals", ())
            console.set_pending_approvals(pending)
            set_actions = getattr(console, "set_interaction_actions", None)
            if callable(set_actions):
                set_actions(getattr(interaction_snapshot, "actions", ()))
            phase = str(getattr(getattr(interaction_snapshot, "phase", None), "value", ""))
            phase_messages = {
                "streaming": "正在生成回复…",
                "tool_running": "正在执行桌面操作…",
                "approval_required": "等待你确认工具操作",
                "failed": "本轮未完成，可检查活动记录后重试",
            }
            if phase == "configuration_required":
                # 模型快照来自运行时线程，message 可能来自供应商/路由异常；
                # 这里始终使用固定点击入口，内部原因只留在诊断日志。
                phase_messages[phase] = _PUBLIC_MODEL_NOT_READY
            elif (
                phase == "idle"
                and getattr(interaction_snapshot, "finish_reason", "") == "cancelled"
            ):
                phase_messages[phase] = "对话已停止"
            # 审批、工具执行、流式生成和失败属于必须立即可见的状态；其余阶段
            # 在短暂点击反馈窗口内让本地反馈保持可见，避免 40ms 定时器覆盖。
            if pending_cancel_future is not None and pending_submit_text:
                console.set_status("正在停止上一轮，消息将继续发送…")
            elif phase in phase_messages:
                status = _interaction_status_with_local_feedback(
                    phase=phase,
                    phase_message=phase_messages[phase],
                    feedback_notice=pet_feedback_notice,
                    feedback_active=feedback_active,
                )
                if status:
                    console.set_status(status)
            elif feedback_active:
                console.set_status(pet_feedback_notice)
        elif console is not None and feedback_active:
            console.set_status(pet_feedback_notice)

        # 配置保存只把协程投递到运行时线程；完成结果在 Qt 定时器中消费，
        # 避免保存回调阻塞主线程，也避免后台线程直接触碰 Qt 对象。
        if pending_runtime_reload is not None and pending_runtime_reload.done():
            reload_future = pending_runtime_reload
            pending_runtime_reload = None
            try:
                reload_result = reload_future.result()
            except Exception as exc:
                reload_status = "unavailable"
                reload_message = "配置已保存；运行时配置未应用，需重启后生效"
                logger.warning("runtime configuration reload failed: %s", type(exc).__name__)
            else:
                reload_status = str(
                    reload_result.get("status", "unavailable")
                    if isinstance(reload_result, Mapping)
                    else "unavailable"
                )
                if isinstance(reload_result, Mapping) and reload_result.get("rendering_status"):
                    renderer_receipt_status = (
                        str(reload_result.get("rendering_status", "") or "").strip().lower()
                    )
                    if reload_status == "restart_required" and renderer_receipt_status != "busy":
                        renderer_receipt_status = "restart_required"
                    renderer_receipt = {
                        "status": renderer_receipt_status,
                        "model_key": str(reload_result.get("rendering_model", "") or ""),
                    }
                    renderer_result_setter = getattr(console, "set_renderer_model_result", None)
                    if callable(renderer_result_setter):
                        renderer_result_setter(renderer_receipt)
                if reload_status == "reloaded":
                    if (
                        isinstance(reload_result, Mapping)
                        and reload_result.get("cleanup") == "restart_required"
                    ):
                        reload_message = "配置已保存；运行时配置已切换，旧资源清理需重启"
                    elif isinstance(reload_result, Mapping) and reload_result.get(
                        "rendering_status"
                    ) in {"pending", "requested"}:
                        reload_message = "配置已保存；显示资源正在热切换"
                    else:
                        reload_message = "配置已保存；运行时配置已立即生效"
                elif reload_status == "restart_required":
                    restart_sections = (
                        tuple(reload_result.get("restart_sections", ()))
                        if isinstance(reload_result, Mapping)
                        else ()
                    )
                    if "rendering" in restart_sections:
                        reload_message = "配置已保存；点击“重启并应用渲染引擎”完成切换"
                    elif isinstance(reload_result, Mapping) and reload_result.get(
                        "applied_sections"
                    ):
                        reload_message = "配置已保存；部分运行时配置已生效，其余需重启"
                    else:
                        reload_message = "配置已保存；当前配置需重启后生效"
                else:
                    reload_message = "配置已保存；运行时配置未应用，需重启后生效"
            if console is not None:
                console.set_status(reload_message)
            else:
                logger.info("model configuration reload status: %s", reload_status)

        snapshot = runtime.conversation.presentation.snapshot
        if console is not None:
            console.set_snapshot(snapshot)
        # 点击反馈是用户刚刚触发的直接互动；流式文本/工具状态可能在
        # 2.4 秒保持窗口内抵达并清掉 PresentationService 的 direct_text。
        # 这一小段时间仍以最新点击短句为气泡内容，避免用户看到“点击无
        # 反馈”。清理定时器会恢复同一快照中的流式文本。
        if feedback_active and isinstance(last_pet_feedback, Mapping):
            feedback_phrase = str(last_pet_feedback.get("phrase", "") or "").strip()
            if feedback_phrase:
                feedback_mood = str(last_pet_feedback.get("mood", "neutral") or "neutral")
                # 点击反馈的气泡可以先于 TTS 事件出现，但不能因此把口型
                # 永久置为张开；只有当前确有活动音频时才保持说话状态。
                feedback_speaking = bool(tts_activity.active)
                target = window if window is not None else web_host
                setter = getattr(target, "set_speech", None)
                renderer_state = getattr(target, "renderer", target)
                current_feedback_text = str(
                    getattr(renderer_state, "_speech_text", "") or ""
                ).strip()
                if callable(setter) and (
                    snapshot != last_presented_snapshot
                    or feedback_speaking != last_presented_speaking
                    or current_feedback_text != feedback_phrase
                ):
                    try:
                        setter(
                            feedback_phrase,
                            mood=feedback_mood,
                            visible=True,
                            speaking=feedback_speaking,
                        )
                        last_presented_snapshot = snapshot
                        last_presented_speaking = feedback_speaking
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        logger.debug("failed to preserve direct pet feedback speech", exc_info=True)
                return
        target = window if window is not None else web_host
        presentation = runtime.conversation.presentation
        sentence_peeker = getattr(presentation, "peek_next_sentence", None)
        sentence_acker = getattr(presentation, "ack_sentence", None)
        sentence_getter = getattr(presentation, "pop_next_sentence", None)
        queue_blocked = _speech_queue_blocked(
            phase=getattr(interaction_snapshot, "phase", "")
            if interaction_snapshot is not None
            else "",
            tool_status=getattr(snapshot, "tool_status", ""),
        )
        if (
            not queue_blocked
            and target is not None
            and (callable(sentence_peeker) or callable(sentence_getter))
        ):
            setter = getattr(target, "set_speech", None)
            if callable(setter):
                try:
                    if callable(sentence_peeker) and callable(sentence_acker):
                        pending_sentence = sentence_peeker()
                    elif callable(sentence_getter):
                        pending_sentence = sentence_getter()
                    else:
                        pending_sentence = None
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pending_sentence = None
                if pending_sentence is not None:
                    try:
                        queue_speaking = _tts_speaking_for_snapshot(
                            interaction_snapshot or snapshot,
                            active=tts_activity.active,
                            pending_sentence=True,
                        )
                        setter_result = setter(
                            _friendly_stream_text(getattr(pending_sentence, "text", "")),
                            mood=snapshot.rendered_mood,
                            visible=True,
                            speaking=queue_speaking,
                        )
                        if setter_result is False:
                            # Web Live2D 在页面尚未建立时会显式返回 False；
                            # 保留队首，下一帧页面就绪后重试。
                            return
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        logger.debug("failed to publish queued sentence speech", exc_info=True)
                        # 队列项尚未确认时不能回退到 snapshot 的“最新句”，
                        # 否则一次 WebEngine/Qt 瞬态错误会跳过前面的句子并
                        # 在下一帧乱序重试。
                        return
                    else:
                        if callable(sentence_acker):
                            sentence_acker(pending_sentence)
                        # 队列内同句也由 sequence 区分；记录最新快照仅用于
                        # 控制台/状态去重，下一轮仍会优先消费队列中的下一句。
                        last_presented_snapshot = snapshot
                        last_presented_speaking = queue_speaking
                        return
        rendered_text = _pet_speech_text(snapshot)
        speaking = _tts_speaking_for_snapshot(
            interaction_snapshot or snapshot,
            active=tts_activity.active,
        )
        # 只在展示快照或口型状态发生变化时写入 Qt，避免后台定时器反复
        # 覆盖工具直接发出的语音；TTS 终态的延迟 Silence 仍可独立收口。
        if snapshot == last_presented_snapshot and speaking == last_presented_speaking:
            return
        last_presented_snapshot = snapshot
        last_presented_speaking = speaking
        if not rendered_text:
            if window is not None:
                window.set_speech("", visible=False, speaking=False)
            elif web_host is not None:
                web_host.set_speech("", visible=False, speaking=False)
            return
        if window is not None:
            window.set_speech(
                rendered_text,
                mood=snapshot.rendered_mood,
                speaking=speaking,
            )
        elif web_host is not None:
            web_host.set_speech(
                rendered_text,
                mood=snapshot.rendered_mood,
                speaking=speaking,
            )

    def submit_text(text: str) -> None:
        nonlocal active_context, active_future, pending_submit_text, pending_cancel_future
        nonlocal pending_cancel_operation_id
        nonlocal pending_resubmit_scheduled, pending_resubmit_token
        nonlocal last_submitted_text
        value = str(text or "").strip()
        if not value:
            return
        try:
            runtime.notify_user_interaction("text_input")
        except (AttributeError, RuntimeError, TypeError, ValueError):
            logger.debug("failed to record text input activity", exc_info=True)
        # 用户主动发消息时，自主 motion/漫游应立即让出渲染器；仅记录
        # activity 不会推进 BehaviorService 的代际，可能让旧动作在对话
        # 气泡出现后继续写入模型。
        interrupt = getattr(runtime.behavior, "interrupt", None)
        if callable(interrupt):
            try:
                interrupt()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.debug(
                    "failed to interrupt autonomous behavior for text input",
                    exc_info=True,
                )
        ready, model_message = _model_channel_status(runtime)
        if not ready:
            if console is not None:
                console.set_status(model_message)
                console.append_line(f"系统：{model_message}。")
            return
        if pending_resubmit_scheduled:
            # 取消旧回合后尚未发出的消息只保留最后一条；每次新输入都
            # 重置合并计时，避免高负载下中间消息先被启动。
            pending_submit_text = value
            pending_resubmit_token += 1
            token = pending_resubmit_token
            try:
                QTimer.singleShot(
                    _RESUBMIT_DEBOUNCE_MS,
                    lambda selected=token: flush_pending_resubmit(selected),
                )
            except (AttributeError, RuntimeError, TypeError, ValueError):
                flush_pending_resubmit(token)
            if console is not None:
                console.set_status("正在合并消息，稍后继续发送…")
            return
        if pending_cancel_future is not None:
            if not pending_cancel_future.done():
                # 用户在取消窗口内再次输入时只保留最新消息，不能绕过
                # GenerationGate 直接创建第三个 context。
                pending_submit_text = value
                if console is not None:
                    QTimer.singleShot(
                        0,
                        lambda: (
                            console.set_status("正在停止上一轮，消息将继续发送…")
                            if console is not None
                            else None
                        ),
                    )
                return
            finished_cancel = pending_cancel_future
            pending_cancel_future = None
            pending_cancel_operation_id = None
            pending_submit_text = None
            pending_resubmit_scheduled = False
            pending_resubmit_token += 1
            try:
                finished_cancel.result()
            except Exception as exc:
                logger.warning(
                    "conversation cancellation before submit failed: %s",
                    type(exc).__name__,
                )
        conversation_active = bool(active_future is not None and not active_future.done())
        if active_context is not None and not conversation_active:
            try:
                # 审批暂停时 Future 已结束，但 pending_approvals 仍代表旧
                # 回合占用会话；提交新消息前也必须走取消代际路径。
                conversation_active = bool(runtime.conversation.has_active_conversation())
            except Exception:
                conversation_active = True
        if active_context is not None and conversation_active:
            # ``cancel_conversation`` 必须先在运行时线程推进代际；若这里立即
            # begin_context，取消协程稍后会再推进一代，刚创建的回合就会被判定
            # stale。把最新输入保留到取消 Future 完成后再提交。
            if pending_cancel_future is None or pending_cancel_future.done():
                pending_cancel_future = stop_conversation()
                pending_cancel_operation_id = None
            if pending_cancel_future is None:
                if console is not None:
                    console.set_status("上一轮无法停止，消息未提交")
                return
            pending_submit_text = value
            active_context = None
            active_future = None
            if console is not None:
                QTimer.singleShot(
                    0,
                    lambda: (
                        console.set_status("正在停止上一轮，消息将继续发送…")
                        if console is not None
                        else None
                    ),
                )
            return
        if runtime.proactive is not None:
            runtime.proactive.interrupt()
        active_context = runtime.conversation.begin_context()
        last_submitted_text = value
        if window is not None:
            window.set_speech("", visible=False, speaking=False)
        elif web_host is not None:
            web_host.set_speech("", visible=False, speaking=False)
        try:
            future = runtime_loop.submit(
                runtime.conversation.complete(value, context=active_context)
            )
        except RuntimeError:
            active_context = None
            return
        active_future = future
        submitted_context = active_context

        def report_failure(done: Any, context: object = submitted_context) -> None:
            result: object
            try:
                result = done.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                result = {"status": "cancelled"}
            except Exception as exc:  # 服务错误不能终止 Qt 主循环
                logger.warning("conversation failed: %s", type(exc).__name__)
                result = {"status": "failed"}
            try:
                pending = dispatcher.invoke(_finish_conversation_receipt, context, result)
                if inspect.isawaitable(pending):
                    try:
                        asyncio.create_task(pending)
                    except RuntimeError:
                        pending.close()
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # Qt 已开始退出时只丢弃展示回执；运行时 Future 仍由统一
                # 收尾路径消费，不能因为 UI 不可用而制造未处理异常。
                logger.debug("failed to publish conversation operation receipt", exc_info=True)

        future.add_done_callback(report_failure)

    def prompt_text() -> None:
        from PySide6.QtWidgets import QInputDialog

        text, accepted = QInputDialog.getText(None, "MeaPet 对话", "输入消息：")
        if accepted:
            submit_text(str(text))

    def open_input_hotkey(active: bool = True) -> None:
        """兼容 press/hold 两种输入快捷键模式，释放时不重复弹框。"""

        if bool(active):
            prompt_text()

    def exclude_pet_window(target: Any) -> None:
        exclude = getattr(platform, "exclude_window_id", None)
        win_id = getattr(target, "winId", None)
        if not callable(exclude) or not callable(win_id):
            return
        try:
            exclude(int(win_id()))
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return

    def restore_pet_input() -> object:
        target = window if window is not None else web_host
        locked_getter = getattr(target, "is_window_locked", None)
        locked = False
        if callable(locked_getter):
            try:
                locked = bool(locked_getter())
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                locked = False
        handler_name = "set_window_locked" if locked else "set_click_through"
        handler = getattr(target, handler_name, None)
        if not callable(handler):
            result: object = {"status": "unavailable", "reason": "pet window is unavailable"}
        else:
            try:
                result = handler(False)
                # Qt 主线程路径应同步返回；若外部宿主错误地返回协程，先关闭
                # 未执行对象，避免恢复入口产生未等待警告并误报成功。
                if inspect.isawaitable(result):
                    close = getattr(result, "close", None)
                    if callable(close):
                        close()
                    result = {
                        "status": "unavailable",
                        "reason": "click-through restore returned an async result",
                    }
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning("点击恢复请求失败：%s", type(exc).__name__)
                result = {
                    "status": "unavailable",
                    "reason": "点击恢复暂时不可用，请保持控制台可见并重试",
                }
        succeeded = (
            bool(isinstance(result, Mapping) and result.get("locked") is False)
            if locked
            else _click_through_operation_succeeded(result, enabled=False)
        )
        if console is not None:
            if succeeded:
                restored_click_through = bool(
                    result.get("enabled", False) if isinstance(result, Mapping) else False
                )
                console.set_click_through_checked(restored_click_through)
                if locked:
                    setter = getattr(console, "set_window_locked_checked", None)
                    if callable(setter):
                        setter(False)
            else:
                console.set_status("点击恢复失败；请保持控制台可见并重试")
            _console_result("恢复点击", result)
        return result

    def restore_tight_layout_pet(*, force: bool = False) -> None:
        """恢复因极小屏辅助窗口布局而暂时隐藏的桌宠。"""

        nonlocal tight_layout_pet_hidden, tight_layout_pet_was_visible
        if not tight_layout_pet_hidden:
            return
        # 模型向导关闭后，控制台会在同一组 finished 回调中恢复显示；在
        # 极小屏上此时仍应保持宠物隐藏，避免全宽控制台与宠物重新重叠。
        # 用户明确点击“显示桌宠”时通过 force 绕过该保护。
        if not force and console is not None:
            visible_getter = getattr(console, "isVisible", None)
            if callable(visible_getter) and bool(visible_getter()):
                return
        target = window if window is not None else getattr(web_host, "view", None)
        should_show = tight_layout_pet_was_visible
        tight_layout_pet_hidden = False
        tight_layout_pet_was_visible = False
        if console is not None:
            try:
                console.set_compact_layout(False)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        if not should_show or target is None:
            return
        accepts_focus = _window_accepts_focus(target)
        methods = ("showNormal", "show", "raise_") if accepts_focus else ("showNormal", "show")
        for method_name in methods:
            handler = getattr(target, method_name, None)
            if callable(handler):
                try:
                    handler()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    break

    def hide_pet_for_tight_layout() -> None:
        """在极小屏上让点击式辅助窗口获得可用宽度。"""

        nonlocal tight_layout_pet_hidden, tight_layout_pet_was_visible
        if tight_layout_pet_hidden:
            return
        if console is not None:
            try:
                console.set_compact_layout(True)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        target = window if window is not None else getattr(web_host, "view", None)
        if target is None:
            return
        visible_getter = getattr(target, "isVisible", None)
        was_visible = bool(visible_getter()) if callable(visible_getter) else True
        tight_layout_pet_was_visible = was_visible
        tight_layout_pet_hidden = True
        if not was_visible:
            return
        hide = getattr(target, "hide", None)
        if callable(hide):
            try:
                hide()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return

    def on_console_hidden() -> None:
        """隐藏控制台时撤销点击穿透，避免留下失去恢复入口的窗口。"""

        nonlocal console_recovery_available
        if console is not None and not tight_layout_pet_hidden:
            try:
                console.set_compact_layout(False)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        target = window if window is not None else web_host
        locked_getter = getattr(target, "is_window_locked", None)
        if callable(locked_getter):
            try:
                if bool(locked_getter()):
                    # 锁定模式本身就是桌面歌词式的无输入状态；隐藏控制台
                    # 不能自动解锁，否则用户的锁定操作会被关闭窗口悄悄撤销。
                    if not click_through_recovery_available:
                        console_recovery_available = True
                        if console is not None:
                            console.show_and_focus()
                        return
                    console_recovery_available = False
                    restore_tight_layout_pet()
                    set_ui_interaction_lock(False)
                    return
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                pass
        result = restore_pet_input()
        if _click_through_operation_succeeded(result, enabled=False):
            console_recovery_available = False
            restore_tight_layout_pet()
            set_ui_interaction_lock(False)
            return
        # 清除标志失败时不能让控制台真正消失，否则窗口可能进入不可恢复
        # 状态；保留控制台作为下一次重试入口。
        console_recovery_available = True
        if console is not None:
            console.show_and_focus()

    def set_console_click_through(enabled: bool) -> object:
        """由控制台切换点击穿透；开启前保持控制台可见作为恢复入口。"""

        if enabled and console is not None:
            console.show_and_focus()
        result = runtime.pet_controller.set_click_through(bool(enabled))
        return _console_result("点击穿透", result)

    def show_pet() -> object:
        """显示桌宠并回读窗口可见状态，避免按钮点击没有反馈。"""

        # 用户明确点“显示桌宠”时退出极小屏隐藏状态；关闭控制台
        # 不应再次把它误判为布局临时隐藏。
        restore_tight_layout_pet(force=True)
        target = window if window is not None else getattr(web_host, "view", None)
        if target is None:
            return {
                "status": "unavailable",
                "visible": False,
                "reason": "pet window is unavailable",
            }
        try:
            # 先读取焦点能力，再决定是否调用 ``raise_``。在 X11 上，
            # QWidget.raise_() 对 ``WindowDoesNotAcceptFocus`` 窗口可能
            # 隐式发出 requestActivate()；即使不显式调用 activateWindow，
            # 仍会产生警告并让 WebEngine 原生子表面参与焦点切换。
            # 这是桌宠展示窗口；焦点能力未知时也按“不接收焦点”处理，
            # 避免窗口销毁/重建竞态中读旗标失败后再次触发隐式激活。
            accepts_focus = _window_accepts_focus(target)
            methods = ("showNormal", "show", "raise_") if accepts_focus else ("showNormal", "show")
            for method_name in methods:
                handler = getattr(target, method_name, None)
                if callable(handler):
                    handler()
            if accepts_focus:
                handler = getattr(target, "activateWindow", None)
                if callable(handler):
                    handler()
            visible_getter = getattr(target, "isVisible", None)
            visible = bool(visible_getter()) if callable(visible_getter) else True
            return {"status": "available", "visible": visible}
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("显示桌宠请求失败：%s", type(exc).__name__)
            return {
                "status": "unavailable",
                "visible": False,
                "reason": "桌宠显示暂时不可用，请重试",
            }

    def toggle_pet_visibility() -> object:
        """切换桌宠显示状态，快捷键和控制台共用。"""

        target = window if window is not None else web_host
        handler = getattr(target, "toggle_visibility", None)
        if callable(handler):
            return _console_result("显示桌宠", handler())
        view = getattr(target, "view", None) if target is not None else None
        if view is None:
            return {
                "status": "unavailable",
                "visible": False,
                "reason": "pet window is unavailable",
            }
        try:
            if view.isVisible():
                view.hide()
            else:
                view.show()
            return {"status": "available", "visible": bool(view.isVisible())}
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("切换桌宠显示失败：%s", type(exc).__name__)
            return {
                "status": "unavailable",
                "visible": False,
                "reason": "桌宠显示暂时不可用，请重试",
            }

    def toggle_always_on_top() -> object:
        """切换桌宠窗口置顶状态，并返回可轮询的真实状态回执。"""

        return _console_result("切换窗口置顶", topmost_controller.toggle())

    def restart_application() -> Mapping[str, object]:
        """启动新进程，并仅在其真实渲染 READY 后退出当前桌宠。"""

        timer_value = pending_restart.get("timer")
        if timer_value is not None and bool(getattr(timer_value, "isActive", lambda: False)()):
            return {"status": "pending", "message": "桌宠重启正在验证"}
        descriptor, marker_text = tempfile.mkstemp(prefix="meapet-restart-", suffix=".json")
        os.close(descriptor)
        marker = Path(marker_text)
        arguments = ["-m", "app", *_restart_child_arguments(sys.argv[1:], marker)]
        try:
            started, process_id = QProcess.startDetached(
                sys.executable,
                arguments,
                str(Path.cwd()),
            )
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            marker.unlink(missing_ok=True)
            return {"status": "unavailable", "reason": "application restart failed"}
        if not started:
            marker.unlink(missing_ok=True)
            return {"status": "unavailable", "reason": "application restart was rejected"}
        child_pid = int(process_id)
        rendering = configuration.values.get("rendering")
        expected_backend = normalize_renderer_backend(
            rendering.get("backend", "auto") if isinstance(rendering, Mapping) else "auto"
        )
        started_at = monotonic()
        watchdog = QTimer(app)
        watchdog.setInterval(100)

        def finish_watchdog(*, ready: bool) -> None:
            watchdog.stop()
            marker.unlink(missing_ok=True)
            pending_restart.clear()
            if ready:
                QTimer.singleShot(0, app.quit)
                return
            try:
                os.kill(child_pid, signal.SIGTERM)
            except (OSError, ProcessLookupError, PermissionError):
                pass
            if console is not None:
                console.set_status("新桌宠未通过渲染就绪验证，当前窗口继续运行")

        def poll_restart_ready() -> None:
            if _restart_ready_receipt_matches(
                marker,
                process_id=child_pid,
                expected_backend=expected_backend,
            ):
                finish_watchdog(ready=True)
                return
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                finish_watchdog(ready=False)
                return
            except (OSError, PermissionError):
                # 无权探测不等于子进程退出；继续等私有 READY 文件或超时。
                pass
            if monotonic() - started_at >= _RESTART_READY_TIMEOUT_SECONDS:
                finish_watchdog(ready=False)

        watchdog.timeout.connect(poll_restart_ready)
        pending_restart.update(
            {"timer": watchdog, "path": marker, "pid": child_pid, "backend": expected_backend}
        )
        watchdog.start()
        return {"status": "requested", "process_id": int(process_id), "message": "正在重启桌宠"}

    def toggle_window_lock() -> object:
        """切换桌宠锁定；锁定不暂停自主行为和光标追踪。"""

        target = window if window is not None else web_host
        state_getter = getattr(target, "is_window_locked", None)
        if not callable(state_getter):
            return {"status": "unavailable", "reason": "window lock is unavailable"}
        try:
            current = bool(state_getter())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "window lock state is unavailable"}
        return set_window_locked(not current)

    def set_window_locked(enabled: bool) -> object:
        """按控制台状态设置锁定，不影响自主行为循环。"""

        if bool(enabled) and not (click_through_recovery_available or console_recovery_available):
            return {
                "status": "unavailable",
                "locked": False,
                "reason": "window lock recovery entry is unavailable",
            }
        target = window if window is not None else web_host
        handler = getattr(target, "set_window_locked", None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "window lock is unavailable"}
        result = handler(bool(enabled))
        if console is not None and isinstance(result, Mapping):
            setter = getattr(console, "set_window_locked_checked", None)
            if callable(setter) and result.get("locked") is not None:
                setter(bool(result.get("locked")))
        return _console_result("锁定窗口", result)

    def click_through_hotkey_override(active: bool = True) -> object:
        """按住快捷键临时恢复点击，释放时恢复原点击穿透状态。"""

        target = window if window is not None else web_host
        method_name = "begin_click_through_override" if active else "end_click_through_override"
        handler = getattr(target, method_name, None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "click-through override is unavailable"}
        return _console_result("按键恢复点击" if active else "释放恢复点击", handler())

    def restore_click_through() -> object:
        """通过控制台一次性恢复点击，不建立按住快捷键的临时状态。"""

        return restore_pet_input()

    def current_pet_position() -> tuple[int, int] | None:
        """读取当前 Qt 顶层窗口位置，用于初始化控制台坐标输入。"""

        target = window if window is not None else getattr(web_host, "view", None)
        if target is None:
            return None
        getter = getattr(target, "position", None)
        if not callable(getter):
            getter = getattr(target, "pos", None)
        if not callable(getter):
            return None
        try:
            point = getter()
            return int(point.x()), int(point.y())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def current_pet_movement_bounds() -> tuple[int, int, int, int] | None:
        """读取桌宠窗口左上角可移动的屏幕边界。

        WebEngine 和 QOpenGL 宿主都已实现同一 ``movement_bounds`` 契约；
        控制台只消费这个结果，不在 Qt 表单层重复推断窗口尺寸。
        """

        target = window if window is not None else web_host
        if target is None:
            return None
        getter = getattr(target, "movement_bounds", None)
        if not callable(getter):
            return None
        try:
            value = getter()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None
        if isinstance(value, Mapping):
            if not all(key in value for key in ("left", "top", "right", "bottom")):
                return None
            value = (value["left"], value["top"], value["right"], value["bottom"])
        if isinstance(value, (str, bytes, bytearray)):
            return None
        try:
            left, top, right, bottom = (
                int(round(float(value[index])))  # type: ignore[index]
                for index in range(4)
            )
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            return None
        if right < left or bottom < top:
            return None
        return left, top, right, bottom

    def center_pet() -> object:
        """把桌宠移到当前屏幕可用区域中心，供右键菜单和控制台复用。"""

        bounds = current_pet_movement_bounds()
        if bounds is None:
            return {"status": "unavailable", "reason": "movement bounds are unavailable"}
        left, top, right, bottom = bounds
        result = runtime.pet_controller.move_to((left + right) // 2, (top + bottom) // 2)
        return _console_result("居中桌宠", result)

    def read_foreground_window() -> object:
        """读取当前前台窗口摘要并显示到控制台。"""

        return _request_console_read("前台窗口", "desktop:observe_foreground", {})

    def read_processes() -> object:
        """读取当前进程摘要并显示到控制台。"""

        return _request_console_read("进程列表", "desktop:list_processes", {"limit": 30})

    def read_module_status() -> object:
        """通过统一只读工具检查运行模块，不在 Qt 主线程探测外部服务。"""

        rescan_renderer_models()
        return _request_console_read("模块状态", "system:module_status", {})

    def read_api_audit() -> object:
        """读取模型与工具调用的脱敏审计摘要，不进入网页公开状态。"""

        label = "API调用审计"
        current = pending_console_reads.get(label)
        if current is not None and not current.done():
            return {"status": "degraded", "reason": "API audit call remains in flight"}
        if runtime_loop is None or not runtime_loop.running:
            try:
                return dict(asyncio.run(_read_api_audit()))
            except Exception as exc:
                logger.warning("API 审计读取失败：%s", type(exc).__name__)
                return {"status": "unavailable", "reason": "API 审计暂时不可用"}
        try:
            future = runtime_loop.submit(_read_api_audit())
        except RuntimeError:
            return {"status": "unavailable", "reason": "runtime loop is not running"}
        pending_console_reads[label] = future
        if console is not None:
            console.set_status("正在读取 API 调用审计…")
        return {"status": "requested", "operation": label}

    def read_log_records() -> object:
        """读取最近结构化运行日志，不读取日志文件路径。"""

        label = "运行日志"
        current = pending_console_reads.get(label)
        if current is not None and not current.done():
            return {"status": "degraded", "reason": "log records call remains in flight"}
        if runtime_loop is None or not runtime_loop.running:
            try:
                return dict(asyncio.run(_read_log_records()))
            except Exception as exc:
                logger.warning("运行日志读取失败：%s", type(exc).__name__)
                return {"status": "unavailable", "reason": "运行日志暂时不可用"}
        try:
            future = runtime_loop.submit(_read_log_records())
        except RuntimeError:
            return {"status": "unavailable", "reason": "runtime loop is not running"}
        pending_console_reads[label] = future
        if console is not None:
            console.set_status("正在读取运行日志…")
        return {"status": "requested", "operation": label}

    def _window_module_status() -> Mapping[str, object]:
        """在 Qt 主线程读取窗口事务的固定白名单状态。"""

        target = window if window is not None else web_host
        if target is None:
            return {"status": "unavailable", "available": False}
        topmost = topmost_controller.status(refresh=True)
        locked_getter = getattr(target, "is_window_locked", None)
        try:
            locked = bool(locked_getter()) if callable(locked_getter) else False
        except (AttributeError, RuntimeError, TypeError, ValueError):
            locked = False
        shape: dict[str, object] = {}
        shape_getter = getattr(target, "surface_mask_status", None)
        if callable(shape_getter):
            try:
                raw_shape = shape_getter()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                raw_shape = None
            if isinstance(raw_shape, Mapping):
                shape = {
                    "status": str(raw_shape.get("status", "") or "")[:32].lower(),
                    "ready": bool(raw_shape.get("ready", False)),
                    "input_ready": bool(raw_shape.get("input_ready", False)),
                }
        topmost_status = str(topmost.get("status", "") or "").strip().lower()
        status = "loading" if topmost_status in {"pending", "requested"} else "ready"
        return {
            "status": status,
            "available": True,
            "ready": status == "ready",
            "always_on_top": bool(topmost.get("enabled", False)),
            "topmost_confirmed": bool(topmost.get("confirmed", False)),
            "topmost_status": topmost_status[:32],
            "click_through": bool(getattr(target, "_click_through", False)),
            "locked": locked,
            "input_shape": shape,
        }

    def _logging_module_status() -> Mapping[str, object]:
        """返回统一日志的公开配置与处理器状态，不携带文件路径。"""

        active_configuration = getattr(runtime, "configuration", configuration)
        active_values = getattr(active_configuration, "values", {})
        logging_values = (
            active_values.get("logging") if isinstance(active_values, Mapping) else None
        )
        logging_values = logging_values if isinstance(logging_values, Mapping) else {}
        return _public_logging_handler_status(logging_values, logging.getLogger())

    def _audio_playback_status() -> Mapping[str, object]:
        """读取播放器公开白名单；该函数仅在 Qt 主线程调用。"""

        try:
            raw = audio_player.diagnostics()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            raw = {}
        if not isinstance(raw, Mapping):
            raw = {}

        def nonnegative_integer(key: str) -> int:
            value = raw.get(key, 0)
            return (
                max(0, int(value))
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else 0
            )

        error_code = str(raw.get("error_code", "") or "").strip().casefold()
        if not re.fullmatch(r"[a-z0-9_-]{0,32}", error_code):
            error_code = "audio_output_failed"
        return {
            "status": str(raw.get("status", "unavailable") or "unavailable")[:32],
            "available": bool(raw.get("available", False)),
            "active": bool(raw.get("active", False)),
            "queued_streams": nonnegative_integer("queued_streams"),
            "queued_bytes": nonnegative_integer("queued_bytes"),
            "dropped_streams": nonnegative_integer("dropped_streams"),
            "error_code": error_code,
        }

    def _microphone_public_status() -> Mapping[str, object]:
        """返回不含设备 ID、PCM 或转写正文的采集状态。"""

        return {
            "capture_state": str(microphone_controller.state or "unavailable")[:32],
            "capture_enabled": microphone_capture.public_status().get("enabled") is True,
            "recording": microphone_capture.recording,
            "transcribing": microphone_controller.transcribing,
        }

    def _augment_audio_module(module_id: str, result: object) -> object:
        """在控制台读取 Future.result 时于 Qt 主线程补齐真实音频状态。"""

        if not isinstance(result, Mapping):
            return result
        augmented = dict(result)
        if module_id == "tts":
            augmented["playback"] = dict(_audio_playback_status())
        elif module_id == "asr":
            augmented.update(_microphone_public_status())
        return augmented

    def _augment_module_status(result: object) -> object:
        """在 Qt 主线程补齐窗口与日志状态，再交给控制台投影。"""

        if not isinstance(result, Mapping):
            return result
        augmented = dict(result)
        runtime_value = augmented.get("runtime")
        runtime_status = dict(runtime_value) if isinstance(runtime_value, Mapping) else {}
        model_diagnostics = _model_channel_diagnostics(runtime)
        if isinstance(model_diagnostics, Mapping):
            channels = model_diagnostics.get("channels")
            channel_count = len(channels) if isinstance(channels, (tuple, list)) else 0
            model_ready = bool(model_diagnostics.get("ready", False))
            runtime_status["model"] = {
                "status": "ready" if model_ready else "unavailable",
                "available": model_ready,
                "ready": model_ready,
                "model_ready": model_ready,
                "channel_count": channel_count,
            }
        runtime_status["window"] = dict(_window_module_status())
        runtime_status["logging"] = dict(_logging_module_status())
        tts_value = runtime_status.get("tts")
        if isinstance(tts_value, Mapping):
            runtime_status["tts"] = _augment_audio_module("tts", tts_value)
        asr_value = runtime_status.get("asr")
        if isinstance(asr_value, Mapping):
            runtime_status["asr"] = _augment_audio_module("asr", asr_value)
        augmented["runtime"] = runtime_status
        return augmented

    async def _test_console_module(module_id: str) -> Mapping[str, object]:
        """在 RuntimeLoop 内执行模块健康测试并只返回脱敏状态。"""

        return await _runtime_module_test(runtime, module_id)

    def test_console_module(module_id: str) -> object:
        """提交逐模块测试；本地模型和 IPC 绝不在 Qt 主线程初始化。"""

        normalized = str(module_id or "").strip().casefold()
        if normalized not in _CONSOLE_MODULE_IDS:
            return {"status": "unavailable", "reason_code": "unknown_module"}
        if normalized == "window":
            return _window_module_status()
        if normalized == "logging":
            return _logging_module_status()
        if normalized == "renderer":
            return rescan_renderer_models(reload_current=True)
        if runtime_loop is None or not runtime_loop.running:
            return {"status": "unavailable", "reason_code": "runtime_loop_unavailable"}
        try:
            future = runtime_loop.submit(_test_console_module(normalized))
        except RuntimeError:
            return {"status": "unavailable", "reason_code": "runtime_loop_unavailable"}
        logger.info("控制台模块测试已提交：%s", normalized)
        if normalized not in {"tts", "asr"}:
            return future

        class _AudioModuleFuture:
            """延迟到 Qt 线程读取结果时合并本地音频状态。"""

            def done(self) -> bool:
                return future.done()

            def cancel(self) -> bool:
                return future.cancel()

            def result(self) -> object:
                return _augment_audio_module(normalized, future.result())

        return _AudioModuleFuture()

    def validate_console_configuration(values: Mapping[str, Any]) -> object:
        """在 Qt 主线程执行无副作用的运行时配置校验。"""

        try:
            validate_runtime_configuration(LoadedConfiguration(configuration.path, values))
        except Exception as exc:
            logger.warning("模型/运行配置校验失败：%s", type(exc).__name__)
            return {
                "status": "unavailable",
                "reason": "配置校验未通过，请检查填写内容",
            }
        return {"status": "validated"}

    def save_console_configuration(
        values: Mapping[str, Any],
        *,
        allow_plaintext_secrets: bool = False,
    ) -> object:
        """校验并原子保存配置，然后异步提交可安全热替换的运行时区段。"""

        nonlocal configuration, pending_runtime_reload
        try:
            allow_plaintext_secrets = parse_bool(
                allow_plaintext_secrets,
                field_name="allow_plaintext_secrets",
                default=False,
            )
        except ConfigurationError:
            return {
                "status": "unavailable",
                "reason": "配置保存选项无效",
            }
        previous_values = dict(configuration.values)
        restart_sections = _configuration_restart_sections(previous_values, values)
        validation = validate_console_configuration(values)
        if not isinstance(validation, Mapping) or validation.get("status") != "validated":
            return validation
        try:
            if configuration.source_digest is not None:
                assert_configuration_revision(
                    configuration.path,
                    configuration.source_digest,
                )
            # 控制台配置可能来自已展开的运行时快照；持久化前必须重新读取
            # 磁盘原始 YAML，以便 ``***`` 恢复旧引用，而新引用保持原样。
            persisted_values, missing_api_key_reference = _prepare_persisted_configuration(
                configuration.path,
                values,
                allow_plaintext_secrets=allow_plaintext_secrets,
            )
        except ConfigurationConflictError:
            return {
                "status": "conflict",
                "reason": "配置文件已被外部修改，请重新加载后再保存",
                "reload_required": True,
            }
        except (
            AdapterConfigurationError,
            ConfigurationError,
            OSError,
            UnicodeError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return {
                "status": "unavailable",
                "reason": "配置引用的环境变量不可用，请先补齐后再保存",
            }
        try:
            # 配置中心允许明文凭据持久化；收紧文件权限，避免沿用旧的
            # 组/其他用户可读模式。原子替换仍由配置编辑器统一完成。
            atomic_write_configuration(
                configuration.path,
                persisted_values,
                mode=0o600,
                expected_digest=configuration.source_digest,
            )
        except ConfigurationConflictError:
            return {
                "status": "conflict",
                "reason": "配置文件已被外部修改，请重新加载后再保存",
                "reload_required": True,
            }
        except Exception as exc:
            logger.warning("运行时配置保存失败：%s", type(exc).__name__)
            return {
                "status": "unavailable",
                "reason": "配置未保存，请检查文件权限后重试",
            }

        if missing_api_key_reference:
            try:
                source_digest = configuration_source_digest(configuration.path)
            except ConfigurationError:
                source_digest = None
            configuration = LoadedConfiguration(
                configuration.path,
                previous_values,
                source_digest=source_digest,
            )
            # 当前运行继续使用已加载的旧路由，避免把不完整凭据交给适配器；
            # 文件已经安全落盘，用户补齐同名环境变量后重启或触发重载即可。
            return {
                "status": "saved",
                "runtime_status": "restart_required",
                "restart_required": True,
                "credentials_required": True,
                "reason": "配置已保存；请补齐密钥环境变量后重启或重新加载",
            }

        # 重新从文件加载以展开 `${ENV}` 引用；编辑器保存的是脱敏占位符，
        # 不能把该占位符原样交给运行时适配器。使用与启动相同的安全默认值，
        # 让运行时热加载拿到完整 schema；默认渠道为空，不会复活已删除的旧渠道。
        previous_storage = previous_values.get("storage")
        previous_database = (
            previous_storage.get("database") if isinstance(previous_storage, Mapping) else None
        )
        database_default = (
            previous_database
            if isinstance(previous_database, str) and previous_database.strip()
            else resource_root.parent / "data" / "meapet.sqlite3"
        )
        try:
            configuration = load_configuration(
                configuration.path,
                defaults=default_configuration_values(
                    resource_root=resource_root,
                    database_path=database_default,
                ),
            )
        except (ConfigurationError, OSError, UnicodeError, ValueError) as exc:
            logger.warning("保存后的模型配置无法重新读取：%s", type(exc).__name__)
            return {
                "status": "saved",
                "runtime_status": "restart_required",
                "restart_required": True,
                "reason": "配置已保存但暂时无法重新读取，请重启后检查",
            }

        # 保存成功后立即刷新控制台的脱敏模型状态；否则配置中心已经显示新
        # 渠道，操作中心顶部仍会保留启动时的“未配置”提示，用户会误以为
        # 配置没有生效。运行时适配器是否真正热加载仍由下方 Future 回读。
        if console is not None:
            try:
                console.set_configuration(
                    configuration.values,
                    path=configuration.path,
                    internal=True,
                )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("failed to refresh console configuration", exc_info=True)

        saved = {"status": "saved"}
        hot_reload_changed = any(
            section in _HOT_RELOAD_CONFIGURATION_ROOTS
            and previous_values.get(section) != values.get(section)
            for section in _HOT_RELOAD_CONFIGURATION_ROOTS
        )
        if not hot_reload_changed:
            if not restart_sections:
                return {
                    **saved,
                    "runtime_status": "reloaded",
                    "message": "配置没有需要热替换的变更",
                }
            return {
                **saved,
                "runtime_status": "restart_required",
                "restart_required": True,
                "reason": "non-hot-reload configuration changed",
                "restart_sections": restart_sections,
            }
        # 跨线程只读取一个布尔快照，作为保守的即时反馈；运行时协程仍会再次
        # 在其事件循环中检查，处理检查后刚开始的新回合。
        if runtime_loop is None or not runtime_loop.running:
            return {
                **saved,
                "runtime_status": "restart_required",
                "restart_required": True,
                "reason": "runtime loop is not running",
            }
        if pending_runtime_reload is not None and not pending_runtime_reload.done():
            return {
                **saved,
                "runtime_status": "restart_required",
                "restart_required": True,
                "reason": "previous runtime configuration reload is still pending",
            }

        updated_configuration = configuration
        try:
            pending_runtime_reload = runtime_loop.submit(
                runtime.apply_configuration(updated_configuration)
            )
        except RuntimeError:
            pending_runtime_reload = None
            return {
                **saved,
                "runtime_status": "restart_required",
                "restart_required": True,
                "reason": "runtime loop is not running",
            }
        submitted_configuration = updated_configuration

        def acknowledge_configuration_baseline(done: object) -> None:
            """运行时确认后推进文件观察器，避免保存内容被重复热刷。"""

            try:
                result = done.result()  # type: ignore[attr-defined]
            except Exception:
                return
            if not isinstance(result, Mapping) or result.get("retry") is True:
                return
            status = str(result.get("status", "") or "").strip().lower()
            if status not in {"reloaded", "unchanged", "restart_required"}:
                return
            watcher = getattr(runtime, "configuration_watcher", None)
            acknowledge = getattr(watcher, "acknowledge", None)
            if callable(acknowledge):
                try:
                    acknowledge(submitted_configuration)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug(
                        "failed to acknowledge configuration watcher baseline",
                        exc_info=True,
                    )

        try:
            pending_runtime_reload.add_done_callback(acknowledge_configuration_baseline)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        if console is not None:
            QTimer.singleShot(
                0,
                lambda: (
                    console.set_status("配置已保存，正在应用运行时配置…")
                    if console is not None
                    else None
                ),
            )
        return {
            **saved,
            "runtime_status": "pending",
            "message": "配置已保存，正在应用运行时配置",
        }

    def _renderer_model_reload_result(result: object, model_key: str) -> dict[str, object]:
        """把宿主模型回执压缩为控制台可读的资源键结果。"""

        if isinstance(result, Mapping):
            safe: dict[str, object] = {
                key: value for key, value in result.items() if isinstance(key, str)
            }
        else:
            safe = {"status": "unavailable", "reason": "模型切换回执不可用"}
        safe["model_key"] = model_key
        # renderer 返回的 ``model`` 可能是绝对 Path；控制台只展示已经由
        # 目录选择器确认的资源键，避免把本机路径写入操作记录。
        safe.pop("model", None)
        return safe

    renderer_model_rollbacks: set[int] = set()

    def rescan_renderer_models(*, reload_current: bool = False) -> Mapping[str, object]:
        """显式重扫模型目录，并让控制台只接收资源相对键。"""

        nonlocal renderer_model_catalog, renderer_model_choices, renderer_model_key
        renderer_model_catalog = Live2DModelCatalog.scan(resource_root)
        renderer_model_choices = renderer_model_catalog.choices
        current_path = None
        if web_renderer is not None:
            current_path = getattr(getattr(web_renderer, "probe", None), "model_path", None)
        if current_path is None:
            current_path = renderer_selection.model_path
        renderer_model_key = ""
        if current_path is not None:
            try:
                resolved_current = Path(current_path).expanduser().resolve()
            except (OSError, RuntimeError, TypeError, ValueError):
                resolved_current = None
            if resolved_current is not None:
                renderer_model_key = next(
                    (
                        entry.key
                        for entry in renderer_model_catalog.models
                        if entry.descriptor == resolved_current
                    ),
                    "",
                )
        if console is not None:
            setter = getattr(console, "set_renderer_models", None)
            if callable(setter):
                try:
                    setter(renderer_model_key, renderer_model_choices)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to refresh renderer model catalog", exc_info=True)
        if not reload_current:
            return {
                "status": "available",
                "model_key": renderer_model_key,
                "model_count": len(renderer_model_choices),
            }
        selection = renderer_model_catalog.select(renderer_model_key)
        reload_model = getattr(web_renderer, "reload_model", None)
        if not callable(reload_model) or selection.selected is None:
            return {
                "status": "unavailable",
                "model_key": renderer_model_key,
                "reason": "当前 Web Live2D 模型无法重新加载",
            }
        return _renderer_model_reload_result(
            reload_model(selection.selected.descriptor, force=True),
            renderer_model_key,
        )

    def persist_renderer_model(model_key: str) -> Mapping[str, object]:
        """仅在渲染器已提交候选模型后持久化精确资源键。"""

        updated = dict(configuration.values)
        rendering = updated.get("rendering")
        values = dict(rendering) if isinstance(rendering, Mapping) else {}
        values["model"] = model_key
        updated["rendering"] = values
        result = save_console_configuration(updated)
        return dict(result) if isinstance(result, Mapping) else {"status": "unavailable"}

    def on_renderer_model_reload(result: Mapping[str, object]) -> None:
        """接收 Web Live2D 终态回执并更新 Qt 控制台。"""

        nonlocal renderer_model_key
        rescan_renderer_models()
        raw_request_id = result.get("request_id")
        request_id = (
            raw_request_id
            if isinstance(raw_request_id, int) and not isinstance(raw_request_id, bool)
            else 0
        )
        if request_id in renderer_model_rollbacks:
            renderer_model_rollbacks.discard(request_id)
            safe_rollback = _renderer_model_reload_result(result, renderer_model_key)
            if console is not None:
                result_setter = getattr(console, "set_renderer_model_result", None)
                try:
                    if callable(result_setter):
                        result_setter(safe_rollback)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to update renderer rollback receipt", exc_info=True)
            return
        pending_persistence = pending_renderer_model_persistence.pop(request_id, None)
        model_key = ""
        raw_model = result.get("model_key", "")
        if isinstance(raw_model, str) and raw_model in renderer_model_choices:
            model_key = raw_model
        if not model_key:
            raw_model_path = result.get("model")
            try:
                if not isinstance(raw_model_path, (str, os.PathLike)):
                    raise TypeError("renderer model path is invalid")
                relative_model = (
                    Path(raw_model_path)
                    .expanduser()
                    .resolve()
                    .relative_to(resource_root)
                    .as_posix()
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                relative_model = ""
            if relative_model in renderer_model_choices:
                model_key = relative_model
        if pending_persistence is not None:
            model_key = pending_persistence[0]
        safe = _renderer_model_reload_result(result, model_key)
        if str(safe.get("status", "") or "").strip().lower() == "available":
            renderer_model_key = model_key or renderer_model_key
            if pending_persistence is not None:
                persisted = persist_renderer_model(model_key)
                persisted_status = str(persisted.get("status", "") or "").strip().lower()
                if persisted_status not in {"saved", "reloaded", "updated", "ok", "success"}:
                    previous_key = pending_persistence[1]
                    previous_selection = renderer_model_catalog.select(previous_key)
                    rollback = (
                        web_renderer.reload_model(
                            previous_selection.selected.descriptor,
                            force=True,
                        )
                        if web_renderer is not None and previous_selection.selected is not None
                        else {"status": "unavailable"}
                    )
                    rollback_id = rollback.get("request_id") if isinstance(rollback, Mapping) else 0
                    if isinstance(rollback_id, int) and not isinstance(rollback_id, bool):
                        renderer_model_rollbacks.add(rollback_id)
                    safe.update(
                        {
                            "status": "failed",
                            "persistence_status": "failed",
                            "reason": "模型配置保存失败，正在恢复上一模型",
                        }
                    )
                else:
                    safe["persistence_status"] = "saved"
            invalidate_visual_actions = getattr(
                runtime.pet_controller, "invalidate_visual_actions", None
            )
            if callable(invalidate_visual_actions):
                try:
                    invalidate_visual_actions()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug(
                        "failed to invalidate visual actions after model reload",
                        exc_info=True,
                    )
            # 模型换装成功后，动作/表情声明可能已经变化；控制台必须重新读取
            # 当前渲染器能力，否则仍会保留旧模型的空列表或失效名称。
            _refresh_console_action_capabilities(
                console,
                getattr(web_host, "renderer", None),
            )
        if console is not None:
            result_setter = getattr(console, "set_renderer_model_result", None)
            try:
                if callable(result_setter):
                    result_setter(safe)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("failed to update Qt model reload receipt", exc_info=True)

    def select_renderer_model(model_key: object) -> object:
        """先原子切换 Web 模型，终态成功后再持久化资源键。"""

        nonlocal renderer_model_key
        rescan_renderer_models()
        key = model_key if isinstance(model_key, str) else ""
        choices = tuple(renderer_model_choices)
        if key not in choices:
            return {
                "status": "unavailable",
                "model_key": "",
                "reason": "所选 Live2D 模型不可用",
            }
        selection = renderer_model_catalog.select(key)
        if selection.selected is None:
            return {
                "status": "unavailable",
                "model_key": "",
                "reason": "所选 Live2D 模型不可用",
            }
        if renderer_selection.backend == "web_live2d" and web_renderer is not None:
            reload_model = getattr(web_renderer, "reload_model", None)
            if not callable(reload_model):
                return {
                    "status": "unavailable",
                    "model_key": key,
                    "reason": "当前 Web Live2D 宿主无法切换模型",
                }
            result = reload_model(selection.selected.descriptor)
            safe = _renderer_model_reload_result(result, key)
            status = str(safe.get("status", "") or "").strip().lower()
            if status == "pending":
                request_id = safe.get("request_id")
                if isinstance(request_id, int) and not isinstance(request_id, bool):
                    pending_renderer_model_persistence[request_id] = (
                        key,
                        renderer_model_key,
                    )
                safe["persistence_status"] = "pending"
                safe["message"] = "显示资源正在验证；成功后自动保存模型选择"
                return safe
            if status != "unchanged":
                safe["persistence_status"] = "unchanged"
                return safe
        saved = persist_renderer_model(key)
        saved_status = (
            str(saved.get("status", "") or "").strip().lower() if isinstance(saved, Mapping) else ""
        )
        if saved_status not in {"saved", "reloaded", "updated", "ok", "success"}:
            return _renderer_model_reload_result(saved, key)

        runtime_status = (
            str(saved.get("runtime_status", "") or "").strip().lower()
            if isinstance(saved, Mapping)
            else ""
        )
        result = _renderer_model_reload_result(saved, key)
        result["persistence_status"] = "saved"
        renderer_model_key = key
        if runtime_status == "pending":
            result["status"] = "pending"
            result["message"] = "模型选择已保存；显示资源正在热切换"
        elif runtime_status == "restart_required":
            result["status"] = "restart_required"
            result["message"] = "模型选择已保存；当前渲染宿主需重启后生效"
        elif runtime_status in {"reloaded", "unchanged"}:
            result["status"] = runtime_status
            result["message"] = "模型选择已保存并应用"
        else:
            result["status"] = "pending"
            result["message"] = "模型选择已保存；等待运行时回执"
        return result

    def select_renderer_backend(backend: object) -> Mapping[str, object]:
        """持久化公开渲染后端；全局图形 API 只在安全重启后切换。"""

        selected = str(backend or "").strip().lower()
        allowed = {"auto", "opengl", "vulkan", "web_live2d", "sprite"}
        if selected not in allowed:
            return {"status": "unavailable", "reason": "渲染引擎不可用"}
        raw_rendering = configuration.values.get("rendering")
        current_values = dict(raw_rendering) if isinstance(raw_rendering, Mapping) else {}
        current = normalize_renderer_backend(current_values.get("backend", "auto"))
        if selected == current:
            return {"status": "available", "message": "当前已使用该渲染引擎"}
        updated = dict(configuration.values)
        rendering_values = dict(current_values)
        rendering_values["backend"] = selected
        updated["rendering"] = rendering_values
        saved = save_console_configuration(updated)
        status = (
            str(saved.get("status", "") or "").strip().lower() if isinstance(saved, Mapping) else ""
        )
        if status not in {"saved", "reloaded", "updated", "ok", "success"}:
            return (
                dict(saved)
                if isinstance(saved, Mapping)
                else {"status": "unavailable", "reason": "渲染引擎配置未保存"}
            )
        return {
            "status": "restart_required",
            "message": "渲染引擎已保存，请点击重启后应用",
            "backend": selected,
        }

    def test_model_connection(payload: Mapping[str, object]) -> object:
        """提交点击式模型连接测试，不把密钥或端点详情回传到界面。"""

        if not isinstance(payload, Mapping):
            return {"status": "unavailable", "message": "模型连接配置不完整"}
        try:
            channel = _build_model_test_channel(payload, configuration.values)
        except (
            AdapterConfigurationError,
            ConfigurationError,
            AttributeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            return {
                "status": "unavailable",
                "message": "模型连接配置不完整，请检查服务地址、模型和密钥设置。",
            }
        if runtime_loop is None or not runtime_loop.running:
            return {"status": "unavailable", "message": "运行服务尚未启动，暂时无法测试连接。"}
        try:
            request = runtime.router.test_connection(channel)
        except Exception as exc:
            logger.warning("模型连接测试提交失败：%s", type(exc).__name__)
            return {
                "status": "unavailable",
                "message": "模型连接测试暂时不可用，请稍后重试。",
            }
        try:
            future = runtime_loop.submit(request)
            return _public_model_connection_future(future)
        except (RuntimeError, TypeError, ValueError):
            close = getattr(request, "close", None)
            if callable(close):
                close()
            return {"status": "unavailable", "message": "模型连接测试暂时不可用。"}

    def mcp_content_action(payload: Mapping[str, object]) -> object:
        """把内容浏览器的有限操作投递到运行时线程。"""

        if runtime_loop is None or not runtime_loop.running:
            return {"status": "unavailable", "reason": "运行服务尚未启动"}
        if not isinstance(payload, Mapping):
            return {"status": "unavailable", "reason": "MCP 内容操作无效"}
        operation = str(payload.get("operation", "") or "").strip()
        server_name = str(payload.get("server_name", "") or "").strip()
        request: object
        if operation == "list_resources":
            request = runtime.mcp_list_resources(server_name)
        elif operation == "read_resource":
            request = runtime.mcp_read_resource(server_name, payload.get("uri"))
        elif operation == "list_prompts":
            request = runtime.mcp_list_prompts(server_name)
        elif operation == "get_prompt":
            arguments = payload.get("arguments")
            if arguments is not None and not isinstance(arguments, Mapping):
                return {"status": "unavailable", "reason": "prompt 参数无效"}
            request = runtime.mcp_get_prompt(
                server_name,
                payload.get("name"),
                dict(arguments) if isinstance(arguments, Mapping) else None,
            )
        elif operation == "approve":
            approval_id = str(payload.get("approval_id", "") or "").strip()
            if not approval_id or len(approval_id) > 256:
                return {"status": "unavailable", "reason": "审批标识无效"}
            request = runtime.approve_mcp_content(approval_id)
        elif operation == "deny":
            approval_id = str(payload.get("approval_id", "") or "").strip()
            if not approval_id or len(approval_id) > 256:
                return {"status": "unavailable", "reason": "审批标识无效"}

            async def deny_request() -> Mapping[str, object]:
                return runtime.deny_mcp_content(approval_id)

            request = deny_request()
        else:
            return {"status": "unavailable", "reason": "MCP 内容操作不受支持"}
        try:
            return runtime_loop.submit(request)
        except (RuntimeError, TypeError, ValueError):
            close = getattr(request, "close", None)
            if callable(close):
                close()
            return {"status": "unavailable", "reason": "MCP 内容操作未提交"}

    def show_mcp_content() -> object:
        """打开独立 MCP 内容窗口；窗口关闭不影响桌宠主控制台。"""

        nonlocal mcp_content_window
        if mcp_content_window is None:
            mcp_content_window = MCPContentWindow(
                servers_provider=runtime.mcp_content_servers,
                action_provider=mcp_content_action,
                parent=console,
            )
            mcp_content_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        mcp_content_window.show_and_focus()
        return {"status": "available", "message": "MCP 内容浏览窗口已打开"}

    def show_console() -> None:
        """懒加载控制台，确保无系统托盘时仍有稳定操作入口。"""

        nonlocal console, console_recovery_available
        rescan_renderer_models()
        set_ui_interaction_lock(True)
        if console is None:
            target_renderer = getattr(window, "renderer", None)
            if target_renderer is None and web_host is not None:
                target_renderer = getattr(web_host, "renderer", None)
            capabilities = getattr(target_renderer, "capabilities", None)
            expressions = getattr(capabilities, "expressions", ())
            motions = getattr(capabilities, "motions", ())
            actual_renderer_backend = str(
                getattr(capabilities, "backend", renderer_selection.backend)
                or renderer_selection.backend
            )
            actual_renderer_available = bool(
                getattr(capabilities, "available", renderer_selection.available)
            )
            callbacks = {
                "submit": submit_text,
                "stop": stop_conversation,
                "retry": retry_last_conversation,
                "test_model": test_model_connection,
                "expression": runtime.pet_controller.set_expression,
                "motion": runtime.pet_controller.play_motion,
                "expression_request": lambda payload: runtime.pet_controller.set_expression_request(
                    ExpressionRequest.from_arguments(payload)
                ),
                "motion_request": lambda payload: runtime.pet_controller.play_motion_request(
                    MotionRequest.from_arguments(payload)
                ),
                "select_renderer_model": select_renderer_model,
                "select_tts_profile": runtime.select_tts_profile,
                "select_tts_language": runtime.set_tts_language,
                "microphone_toggle": microphone_controller.toggle,
                "move_to": runtime.pet_controller.move_to,
                "click_through": set_console_click_through,
                "approve_approval": approve_console_approval,
                "deny_approval": deny_console_approval,
                "show_pet": show_pet,
                "toggle_visibility": toggle_pet_visibility,
                "toggle_always_on_top": toggle_always_on_top,
                "set_window_locked": set_window_locked,
                "restore_click_through": restore_click_through,
                "position": current_pet_position,
                "movement_bounds": current_pet_movement_bounds,
                "set_display_size": runtime.pet_controller.set_display_size,
                "foreground_window": read_foreground_window,
                "list_processes": read_processes,
                "module_status": read_module_status,
                "api_audit": read_api_audit,
                "log_records": read_log_records,
                "mcp_content": show_mcp_content,
                "module_test": test_console_module,
                "quit": app.quit,
                "restart_application": restart_application,
                "open_web_console": show_web_console,
            }

            def place_model_setup(dialog: object) -> None:
                """将模型向导放在桌宠旁边，极小屏时切换为全屏滚动布局。"""

                target = window
                if target is None and web_host is not None:
                    target = getattr(web_host, "view", None)
                _place_widget_adjacent(
                    dialog,
                    target,
                    anchor=current_pet_position(),
                    on_tight_layout=hide_pet_for_tight_layout,
                )
                finished = getattr(dialog, "finished", None)
                connected_property = getattr(dialog, "property", None)
                set_property = getattr(dialog, "setProperty", None)
                already_connected = (
                    bool(connected_property("meapetTightRestoreConnected"))
                    if callable(connected_property)
                    else False
                )
                if not already_connected and callable(getattr(finished, "connect", None)):
                    finished.connect(lambda _result=0: restore_tight_layout_pet())
                    if callable(set_property):
                        set_property("meapetTightRestoreConnected", True)

            active_console_configuration = getattr(runtime, "configuration", configuration)
            active_console_values = getattr(
                active_console_configuration, "values", configuration.values
            )
            active_console_path = getattr(active_console_configuration, "path", configuration.path)
            console = PetConsoleWindow(
                callbacks=callbacks,
                expressions=expressions,
                motions=motions,
                configuration=active_console_values,
                configuration_path=active_console_path,
                developer_mode=False,
                config_callbacks={
                    "validate": validate_console_configuration,
                    "save": save_console_configuration,
                    "save_plaintext": lambda values: save_console_configuration(
                        values,
                        allow_plaintext_secrets=True,
                    ),
                },
                model_setup_positioner=place_model_setup,
                microphone_device_loader=microphone_capture.load_input_devices,
            )
            console.set_microphone_input_devices(
                microphone_capture.input_device_choices,
                loaded=microphone_capture.input_devices_loaded,
            )
            console.set_microphone_state(
                microphone_controller.state,
                "语音输入待命"
                if microphone_controller.state == "idle"
                else "语音输入已关闭"
                if microphone_controller.state == "disabled"
                else "",
            )
            renderer_status_setter = getattr(console, "set_renderer_status", None)
            if callable(renderer_status_setter):
                renderer_status_setter(
                    actual_renderer_backend,
                    available=actual_renderer_available,
                    reason=renderer_selection.reason,
                )
            renderer_models_setter = getattr(console, "set_renderer_models", None)
            if callable(renderer_models_setter):
                renderer_models_setter(renderer_model_key, renderer_model_choices)
            tts_options_setter = getattr(console, "set_tts_options", None)
            if callable(tts_options_setter):
                try:
                    tts_state = runtime.tts_diagnostics()
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    tts_state = {}
                tts_options_setter(
                    tts_state.get("profiles", ()) if isinstance(tts_state, Mapping) else (),
                    language=(
                        tts_state.get("language", "zh") if isinstance(tts_state, Mapping) else "zh"
                    ),
                    active_profile=(
                        tts_state.get("active_profile", "")
                        if isinstance(tts_state, Mapping)
                        else ""
                    ),
                    health=(
                        tts_state.get("health")
                        if isinstance(tts_state, Mapping)
                        and isinstance(tts_state.get("health"), Mapping)
                        else None
                    ),
                )
            locked_getter = getattr(
                window if window is not None else web_host,
                "is_window_locked",
                None,
            )
            if callable(locked_getter):
                try:
                    console.set_window_locked_checked(bool(locked_getter()))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
            topmost_status_setter = getattr(console, "set_always_on_top_status", None)
            if callable(topmost_status_setter):
                try:
                    topmost_status_setter(topmost_controller.status(refresh=True))
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    logger.debug("failed to initialize console topmost status", exc_info=True)
            console.hidden.connect(on_console_hidden)
            position = current_pet_position()
            if position is not None:
                console.set_position(*position)
            pet_window = window
            if pet_window is None and web_host is not None:
                pet_window = getattr(web_host, "view", None)
            _place_widget_adjacent(
                console,
                pet_window,
                anchor=position,
                on_tight_layout=hide_pet_for_tight_layout,
            )
            platform_backend = str(getattr(platform, "backend", "unknown") or "unknown")
            if platform_backend in {"x11", "wayland", "windows"} and surface_mask_required:
                if compositor_state is False:
                    platform_hint = "当前 X11 无合成器；已启用 Shape 输入保护，屏幕透明效果需合成器"
                else:
                    platform_hint = "已启用精确透明输入区域；拖动、置顶和坐标移动按平台能力执行"
            elif platform_backend == "wayland":
                platform_hint = "原生 Wayland：拖动可用，绝对位置、置顶和穿透由合成器决定"
            else:
                platform_hint = f"窗口后端：{platform_backend}"
            renderer_labels = {
                "opengl": "OpenGL Live2D",
                "web_live2d": "Web Live2D",
                "sprite": "精灵回退",
                "unavailable": "不可用",
            }
            renderer_hint = "渲染：" + renderer_labels.get(actual_renderer_backend, "自动后端")
            if model_channel_ready:
                console.set_status(
                    f"模型渠道已就绪；{model_channel_message}；{renderer_hint}；{platform_hint}"
                )
            else:
                # 运行控制台只给用户下一步点击入口，不把 CLI 命令或内部
                # 配置细节当作日常操作说明；高级诊断仍留在配置中心。
                console.set_status(
                    f"模型渠道未配置；表情和动作仍可直接测试；{renderer_hint}；{platform_hint}"
                )
                console.append_line(
                    "系统：表情和动作不依赖对话 API；如需聊天，请点击“配置模型”补齐连接信息。"
                )
        pet_window = window
        if pet_window is None and web_host is not None:
            pet_window = getattr(web_host, "view", None)
        setup_dialog = getattr(console, "_model_setup_dialog", None)
        if setup_dialog is not None and bool(getattr(setup_dialog, "isVisible", lambda: False)()):
            # 自动模型向导期间，双击桌宠或快捷键不应把被隐藏的控制台
            # 再次抬到前台覆盖向导；只重新聚焦向导并保持其避让位置。
            _place_widget_adjacent(
                setup_dialog,
                pet_window,
                anchor=current_pet_position(),
                on_tight_layout=hide_pet_for_tight_layout,
            )
            setup_dialog.show()
            setup_dialog.raise_()
            setup_dialog.activateWindow()
            console_recovery_available = True
            _refresh_console_approvals()
            return
        console.show_and_focus()
        _place_widget_adjacent(
            console,
            pet_window,
            anchor=current_pet_position(),
            on_tight_layout=hide_pet_for_tight_layout,
        )
        if last_pet_feedback is not None and monotonic() < pet_feedback_hold_until:
            console.set_pet_feedback(last_pet_feedback)
        else:
            console.clear_pet_feedback()
        console_recovery_available = True
        _refresh_console_approvals()

    def open_model_setup_on_start() -> None:
        """无模型渠道时给首次启动用户一个可直接点击的配置入口。"""

        if model_channel_ready:
            return
        show_console()
        if console is not None:
            console.show_model_settings()

    def show_control_menu(point: object) -> None:
        """显示桌宠右键菜单，作为托盘之外的本地恢复入口。"""

        from PySide6.QtWidgets import QMenu

        target = window if window is not None else web_host
        menu = QMenu()
        open_action = menu.addAction("打开点击式控制台")
        open_action.triggered.connect(show_web_console)
        legacy_action = menu.addAction("打开 Qt 配置/诊断")
        legacy_action.triggered.connect(show_console)
        center_action = menu.addAction("居中桌宠")
        center_action.triggered.connect(center_pet)
        visibility_action = menu.addAction("显示/隐藏桌宠")
        visibility_action.triggered.connect(toggle_pet_visibility)
        topmost_action = menu.addAction("切换窗口置顶")
        topmost_action.triggered.connect(toggle_always_on_top)
        lock_action = menu.addAction("锁定/解锁桌宠窗口")
        lock_action.triggered.connect(toggle_window_lock)
        stop_action = menu.addAction("中断当前对话")
        stop_action.triggered.connect(stop_conversation)
        restore_action = menu.addAction("恢复桌宠点击")
        restore_action.triggered.connect(restore_pet_input)
        show_action = menu.addAction("显示桌宠")
        show_action.triggered.connect(show_pet)
        menu.addSeparator()
        quit_action = menu.addAction("退出 MeaPet")
        quit_action.triggered.connect(app.quit)
        try:
            menu.exec(point)
        except (AttributeError, RuntimeError, TypeError):
            if target is not None:
                show_console()

    def connect_window_controls(target: Any) -> None:
        """连接右键菜单、控制台快捷键和对话入口。"""

        context_signal = getattr(target, "contextMenuRequested", None)
        if context_signal is not None and callable(getattr(context_signal, "connect", None)):
            context_signal.connect(show_control_menu)
        console_signal = getattr(target, "consoleRequested", None)
        if console_signal is not None and callable(getattr(console_signal, "connect", None)):
            console_signal.connect(show_console)
        click_setter = getattr(target, "set_click_callback", None)
        if callable(click_setter):
            click_setter(handle_pet_click)

        # 快捷键由 HotkeyManager 统一按 YAML 注册；这里仅连接窗口自身的
        # 双击/右键信号，避免固定组合键与配置覆盖相互重复触发。

    def activate_sprite_fallback(message: str) -> None:
        """在 WebEngine 运行期加载失败时切换到可用的精灵窗口。"""

        nonlocal web_renderer, web_host, window
        if not allow_renderer_fallback:
            logger.error(
                "显式渲染后端运行期失败，禁止静默切换精灵：requested=%s reason=%s",
                renderer_selection.requested_backend,
                _safe_ui_text(message, limit=160) or "渲染页面不可用",
            )
            return
        if window is not None:
            return
        renderer = SpriteRenderer(resource_root / "sprites")
        fallback = PetOpenGLWindow(
            renderer,
            platform=platform,
            sprite_scale=sprite_scale,
            always_on_top=always_on_top,
        )
        # QOpenGLWindow 使用 QWindow.setTitle；明确标记回退窗口，便于
        # X11/Wayland 诊断当前实际渲染后端，也避免用户误以为仍在使用
        # 已失败的 Web Live2D 页面。
        set_title = getattr(fallback, "setTitle", None)
        if callable(set_title):
            set_title("MeaPet Sprite")
        fallback.textSubmitted.connect(submit_text)
        connect_window_controls(fallback)
        snapshot = runtime.conversation.presentation.snapshot
        fallback.set_speech(
            _friendly_stream_text(snapshot.rendered_text) or idle_message,
            mood=snapshot.rendered_mood,
            speaking=False,
        )
        window = fallback
        failed_host = web_host
        web_host = None
        runtime.pet_controller.attach(fallback)
        fallback.show()
        # QOpenGLWindow 的 framebuffer/alpha 在 show 后才稳定；与明确 sprite
        # 入口保持同一时序，避免回退窗口首帧把输入区域报告为空。
        fallback.set_surface_mask_enabled(surface_mask_required)
        exclude_pet_window(fallback)
        if console is not None:
            capabilities = getattr(renderer, "capabilities", None)
            console.set_action_capabilities(
                getattr(capabilities, "expressions", ()),
                getattr(capabilities, "motions", ()),
            )
            renderer_status_setter = getattr(console, "set_renderer_status", None)
            if callable(renderer_status_setter):
                renderer_status_setter(
                    "sprite",
                    available=bool(getattr(capabilities, "available", False)),
                    reason=message,
                )
        failed_renderer = web_renderer
        web_renderer = None
        failed_host_shutdown = False
        if failed_host is not None:
            try:
                # 运行期回退时先拆掉 Chromium 视口上的事件过滤器，避免
                # 旧页面继续收到拖动/右键事件；renderer.shutdown 随后负责
                # 释放页面和 WebChannel。
                failed_host.shutdown()
                failed_host_shutdown = True
            except (AttributeError, RuntimeError, TypeError, ValueError):
                logger.debug("failed Web Live2D host cleanup during fallback", exc_info=True)
        # WebPetHost 正常情况下已经拥有并关闭 renderer/view；只有宿主不存在
        # 或关闭失败时才执行一次直接 renderer 清理，避免原生 surface 二次释放。
        if failed_renderer is not None and not failed_host_shutdown:
            failed_view = getattr(failed_renderer, "view", None)
            for method_name in ("hide", "close"):
                method = getattr(failed_view, method_name, None)
                if callable(method):
                    try:
                        method()
                    except (AttributeError, RuntimeError, TypeError):
                        pass
            failed_renderer.shutdown()
        safe_reason = _safe_ui_text(message, limit=160) or "渲染页面不可用"
        logger.warning("Web Live2D unavailable at runtime; using sprite fallback: %s", safe_reason)

    def configure_recovery_tray() -> None:
        """提供点击穿透后的可见恢复入口，不依赖宠物窗口继续接收鼠标事件。"""

        nonlocal click_through_recovery_available, tray, tray_watch_timer
        if tray is not None:
            return
        from PySide6.QtWidgets import QMenu, QStyle, QSystemTrayIcon

        try:
            if not QSystemTrayIcon.isSystemTrayAvailable():
                logger.warning("system tray is unavailable; use Ctrl+Shift+M or right-click menu")
                return
            tray = QSystemTrayIcon(app)
            icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
            menu = QMenu()
            restore_action = menu.addAction("恢复桌宠点击")
            restore_action.triggered.connect(restore_pet_input)
            show_action = menu.addAction("显示桌宠")
            show_action.triggered.connect(show_pet)
            center_action = menu.addAction("居中桌宠")
            center_action.triggered.connect(center_pet)
            topmost_action = menu.addAction("切换窗口置顶")
            topmost_action.triggered.connect(toggle_always_on_top)
            lock_action = menu.addAction("锁定/解锁桌宠窗口")
            lock_action.triggered.connect(toggle_window_lock)
            console_action = menu.addAction("打开控制台")
            console_action.triggered.connect(show_console)
            web_console_action = menu.addAction("打开网页控制台")
            web_console_action.triggered.connect(show_web_console)
            stop_action = menu.addAction("中断当前对话")
            stop_action.triggered.connect(stop_conversation)
            quit_action = menu.addAction("退出 MeaPet")
            quit_action.triggered.connect(app.quit)
            tray.setIcon(icon)
            tray.setToolTip("MeaPet：点击穿透可在此恢复")
            tray.setContextMenu(menu)
            tray.show()
            tray_resources.extend(
                (
                    menu,
                    restore_action,
                    show_action,
                    center_action,
                    topmost_action,
                    lock_action,
                    console_action,
                    web_console_action,
                    stop_action,
                    quit_action,
                )
            )
            click_through_recovery_available = False
            runtime.pet_controller.set_click_through_guard(
                lambda: click_through_recovery_available or console_recovery_available
            )

            def verify_tray_visibility() -> None:
                nonlocal click_through_recovery_available, tray, tray_watch_timer
                if tray is not None and _tray_is_recoverable(tray):
                    click_through_recovery_available = True
                    return
                had_tray_recovery = click_through_recovery_available
                click_through_recovery_available = False
                if had_tray_recovery:
                    restore_pet_input()
                if tray is not None:
                    try:
                        tray.hide()
                    except (AttributeError, RuntimeError):
                        pass
                tray = None
                if tray_watch_timer is not None:
                    try:
                        tray_watch_timer.stop()
                    except (AttributeError, RuntimeError):
                        pass
                    tray_watch_timer = None

            # show() 只提交注册请求；等 Qt 处理一次事件后再确认托盘确实可见，
            # 之后持续监测托盘是否被桌面环境隐藏，避免点击穿透失去恢复入口。
            QTimer.singleShot(0, verify_tray_visibility)
            tray_watch_timer = QTimer(app)
            tray_watch_timer.setInterval(3000)
            tray_watch_timer.timeout.connect(verify_tray_visibility)
            tray_watch_timer.start()
        except (AttributeError, OSError, RuntimeError):
            tray = None
            click_through_recovery_available = False
            runtime.pet_controller.set_click_through_guard(
                lambda: click_through_recovery_available or console_recovery_available
            )

    def configure_hotkeys() -> None:
        """按 YAML 注册快捷键并保留能力状态，供控制台审计展示。"""

        nonlocal hotkey_manager
        target = window if window is not None else getattr(web_host, "view", None)
        legacy_shortcuts = getattr(target, "set_legacy_shortcuts_enabled", None)
        if callable(legacy_shortcuts):
            legacy_shortcuts(False)
        try:
            hotkey_config = HotkeyConfig.from_mapping(ui_values.get("hotkeys"))
        except (TypeError, ValueError) as exc:
            logger.error("快捷键配置无效，已关闭快捷键：%s", type(exc).__name__)
            return
        hotkey_manager = HotkeyManager(
            hotkey_config,
            callbacks={
                "open_input": open_input_hotkey,
                "open_console": show_console,
                "toggle_visibility": toggle_pet_visibility,
                "toggle_always_on_top": toggle_always_on_top,
                "toggle_window_lock": toggle_window_lock,
                "restore_click_through": click_through_hotkey_override,
            },
            backend=str(getattr(platform, "backend", "unknown") or "unknown"),
        )

        def pet_pointer_over() -> bool:
            active_target = window if window is not None else web_host
            if active_target is None:
                return False
            renderer = getattr(active_target, "renderer", active_target)
            for owner in (active_target, renderer):
                for name in ("_page_pointer_over", "_pointer_over"):
                    value = getattr(owner, name, None)
                    if isinstance(value, bool):
                        return value
            return False

        status = hotkey_manager.install(target)
        hover_installer = getattr(hotkey_manager, "install_hover_enter", None)
        if callable(hover_installer):
            hover_installer(prompt_text, pet_pointer_over)
        try:
            bindings = tuple(status.get("bindings", ())) if isinstance(status, Mapping) else ()
        except TypeError:
            bindings = ()
        global_status = (
            (str(status.get("global_status", "")) if isinstance(status, Mapping) else "")
            .strip()
            .lower()
        )
        mode = "全局快捷键可用" if global_status == "available" else "快捷键按当前桌面能力运行"
        logger.info("快捷键已注册：%d 项，%s", len(bindings), mode)
        if isinstance(status, Mapping):
            logger.debug(
                "快捷键注册内部详情：%s",
                json.dumps(dict(status), ensure_ascii=False, default=str),
            )

    behavior_interaction_active = False

    def update_cursor_tracking() -> None:
        """用全局光标位置驱动模型注视，不依赖桌宠窗口接收鼠标。"""

        nonlocal behavior_interaction_active
        target = window if window is not None else web_host
        # 行为循环在独立线程中运行；仅依赖下一次轮询会让拖动开始后
        # 仍提交一个迟到步进。Qt 定时器看到用户正在悬停/拖动时立即
        # 推进行为代际，锁定窗口则不返回 interacting，保持自主行为。
        interacting = getattr(runtime.pet_controller, "is_user_interacting", None)
        if callable(interacting):
            try:
                state = interacting()
                if inspect.isawaitable(state):
                    # Qt 定时器不应在主线程等待后台协程；若兼容控制器
                    # 意外返回 awaitable，主动回收未调度对象，避免退出时
                    # 出现 RuntimeWarning 或把协程对象当作真值。
                    closer = getattr(state, "close", None)
                    if callable(closer):
                        closer()
                    return
                active = (
                    bool(state.get("interacting", state.get("active", False)))
                    if isinstance(state, Mapping)
                    else bool(state)
                )
                if active and not behavior_interaction_active:
                    interrupt = getattr(runtime.behavior, "interrupt", None)
                    if callable(interrupt):
                        interrupt()
                behavior_interaction_active = active
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("behavior interaction interrupt probe failed", exc_info=True)
        updater = getattr(target, "update_cursor_tracking", None)
        if not callable(updater):
            return
        try:
            updater()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            logger.debug("cursor tracking update failed", exc_info=True)

    def attach_sprite_window() -> None:
        """装配明确选择或运行期回退的精灵窗口。"""

        nonlocal window
        renderer = SpriteRenderer(resource_root / "sprites")
        window = PetOpenGLWindow(
            renderer,
            platform=platform,
            sprite_scale=sprite_scale,
            always_on_top=always_on_top,
        )
        set_title = getattr(window, "setTitle", None)
        if callable(set_title):
            set_title("MeaPet Sprite")
        runtime.pet_controller.attach(window)
        window.textSubmitted.connect(submit_text)
        connect_window_controls(window)
        window.set_speech(idle_message, mood="neutral", speaking=False)
        window.show()
        # QOpenGLWindow 的实际 framebuffer/Shape 只有 show 后才稳定；提前
        # 设置掩码会让透明角点在首个 native 子表面创建前仍拦截底层窗口。
        window.set_surface_mask_enabled(surface_mask_required)
        exclude_pet_window(window)

    try:
        model_dir = resource_root / "live2d" / "model"
        live2d = (
            Live2DRenderer(model_dir, model_path=renderer_selection.model_path)
            if renderer_selection.backend == "opengl"
            else None
        )
        if renderer_selection.backend == "opengl":
            if live2d is None or not live2d.capabilities.available:
                raise RuntimeError("selected OpenGL Live2D backend is unavailable at runtime")
            window = PetOpenGLWindow(
                live2d,
                platform=platform,
                fallback_renderer=(
                    SpriteRenderer(resource_root / "sprites") if allow_renderer_fallback else None
                ),
                sprite_scale=sprite_scale,
                always_on_top=always_on_top,
            )
            set_title = getattr(window, "setTitle", None)
            if callable(set_title):
                set_title("MeaPet Live2D")
            runtime.pet_controller.attach(window)
            window.textSubmitted.connect(submit_text)
            connect_window_controls(window)
            window.set_speech(idle_message, mood="neutral", speaking=False)
            window.show()
            window.set_surface_mask_enabled(surface_mask_required)
            exclude_pet_window(window)
        elif renderer_selection.backend == "web_live2d":
            renderer_instance = WebLive2DRenderer(
                resource_root,
                model_path=renderer_selection.model_path,
                failure_callback=activate_sprite_fallback if allow_renderer_fallback else None,
                natural_layout=True,
                frame_rate=float(rendering_values.get("frame_rate", 60.0)),
                geometry_audit_hz=float(rendering_values.get("geometry_audit_hz", 30.0)),
            )
            web_renderer = renderer_instance
            if (
                renderer_instance.capabilities.available
                and renderer_instance.initialize()
                and renderer_instance.capabilities.available
                and window is None
            ):
                view = renderer_instance.view
                if view is None:
                    raise RuntimeError("Web Live2D view was not created")
                _configure_web_view(view, always_on_top=always_on_top)
                view.resize(420, 480)
                view.setWindowTitle("MeaPet Live2D")
                web_host = WebPetHost(renderer_instance, platform=platform)
                web_host.install_interaction(
                    show_control_menu,
                    show_console,
                    handle_pet_click,
                    enter_callback=prompt_text,
                )
                web_host.set_model_reload_callback(on_renderer_model_reload)
                runtime.pet_controller.attach(web_host)
                web_host.set_speech(idle_message, mood="neutral", speaking=False)
                # WebEngine 没有 QOpenGLWindow 的右键信号，快捷键由统一管理器注册。
                connect_window_controls(view)
                view.show()
                # QWebEngine 的 QQuickWidget 子表面在 show 后才创建；此时
                # 同步 Shape 才能同时约束顶层和子表面，避免透明角点拦截桌面。
                web_host.set_surface_mask_enabled(surface_mask_required)
                exclude_pet_window(view)
            elif window is None and allow_renderer_fallback:
                attach_sprite_window()
            elif window is None:
                reason = str(
                    getattr(renderer_instance.capabilities, "message", "Web Live2D 初始化失败")
                    or "Web Live2D 初始化失败"
                )
                renderer_instance.shutdown()
                raise ConfigurationError(
                    f"requested rendering backend '{renderer_selection.requested_backend}' "
                    f"failed at runtime: {reason}"
                )
        elif renderer_selection.backend == "sprite":
            attach_sprite_window()
        elif renderer_selection.backend == "vulkan":
            initialization = default_renderer_registry().initialize(
                renderer_selection,
                platform=platform,
                sprite_scale=sprite_scale,
                always_on_top=always_on_top,
            )
            if not initialization.succeeded or initialization.renderer is None:
                raise ConfigurationError(
                    "requested Vulkan renderer failed to initialize: "
                    + str(initialization.reason or "unknown Vulkan initialization error")
                )
            window = initialization.renderer
            runtime.pet_controller.attach(window)
            text_submitted = getattr(window, "textSubmitted", None)
            if callable(getattr(text_submitted, "connect", None)):
                text_submitted.connect(submit_text)
            connect_window_controls(window)
            window.set_speech(idle_message, mood="neutral", speaking=False)
            window.set_surface_mask_enabled(surface_mask_required)
            exclude_pet_window(window)
        else:
            # 选择结果必须和实际宿主后端一一对应。未来三维/其它后端即使
            # 在注册表中通过能力探测，也不能落入这里后静默显示精灵，
            # 否则用户配置的显式后端与窗口实际渲染器会不一致。
            raise ConfigurationError(
                f"selected rendering backend '{renderer_selection.backend}' "
                "has no Qt host integration"
            )

        configure_recovery_tray()
        configure_hotkeys()
        cursor_tracking_timer = QTimer(app)
        cursor_tracking_timer.setInterval(50)
        cursor_tracking_timer.timeout.connect(update_cursor_tracking)
        cursor_tracking_timer.start()
        if window_locked:

            def apply_initial_window_lock() -> None:
                lock_result = set_window_locked(True)
                if not isinstance(lock_result, Mapping) or lock_result.get("locked") is not True:
                    logger.warning("配置中的窗口锁定未生效：%s", _safe_result_summary(lock_result))

            # 给托盘可见性检查一轮事件，让锁定状态具备可恢复入口。
            QTimer.singleShot(150, apply_initial_window_lock)
        if not model_channel_ready and auto_open_model_setup:
            # 延迟到桌宠窗口完成首帧和快捷键注册后再打开，避免设置对话框
            # 抢在透明窗口创建前显示，也让用户仍能看到桌宠本体。
            QTimer.singleShot(450, open_model_setup_on_start)
        # 窗口已经 attach 后再启动后台行为，避免首个随机动作落到占位控制器。
        runtime_loop.start()
        if restart_ready_file is not None:
            target = window if window is not None else web_host
            target_renderer = getattr(target, "renderer", target)
            target_capabilities = getattr(target_renderer, "capabilities", None)
            actual_backend = str(
                getattr(target_capabilities, "backend", renderer_selection.backend)
                or renderer_selection.backend
            )
            renderer_ready_started_at = monotonic()
            renderer_ready_timer = QTimer(app)
            renderer_ready_timer.setInterval(50)

            def poll_renderer_ready() -> None:
                status = _renderer_ready_status(target, actual_backend)
                if status.ready:
                    try:
                        _write_restart_ready_receipt(
                            restart_ready_file,
                            ready_status=status,
                        )
                    except (OSError, RuntimeError, TypeError, ValueError):
                        logger.exception("重启子进程写入渲染 READY 回执失败")
                        renderer_ready_timer.stop()
                        QTimer.singleShot(0, lambda: app.exit(3))
                        return
                    renderer_ready_timer.stop()
                    logger.info(
                        "重启子进程渲染 READY：%s",
                        json.dumps(status.public(), ensure_ascii=False),
                    )
                    return
                expired = (
                    monotonic() - renderer_ready_started_at >= _RESTART_RENDER_READY_TIMEOUT_SECONDS
                )
                if (
                    status.state in {RendererReadyState.FAILED, RendererReadyState.CLOSED}
                    or expired
                ):
                    renderer_ready_timer.stop()
                    logger.error(
                        "重启子进程未通过真实渲染 READY：%s",
                        json.dumps(status.public(), ensure_ascii=False),
                    )
                    QTimer.singleShot(0, lambda: app.exit(3))

            renderer_ready_timer.timeout.connect(poll_renderer_ready)
            renderer_ready_timer.start()
            QTimer.singleShot(0, poll_renderer_ready)
        # runtime loop 与窗口均已完成装配后才启动 HTTP 控制面；这样 sink
        # 取得 capability 时，对话、工具和窗口动作已具备完整回调路径。
        # 默认关闭或未提供进程内 sink 时不会创建监听套接字。
        start_local_web_server()
        # 展示层只在 Qt 定时器中读取不可变快照，避免后台线程直接触碰 Qt 对象。
        timer = QTimer()
        timer.setInterval(40)
        timer.timeout.connect(update_bubble)
        timer.start()
        return int(app.exec() or 0)
    finally:
        shutting_down = True
        try:
            restart_timer = pending_restart.get("timer")
            if restart_timer is not None:
                try:
                    restart_timer.stop()
                except (AttributeError, RuntimeError):
                    pass
            restart_path = pending_restart.get("path")
            if isinstance(restart_path, Path):
                restart_path.unlink(missing_ok=True)
            restart_pid = pending_restart.get("pid")
            if isinstance(restart_pid, int) and not isinstance(restart_pid, bool):
                try:
                    os.kill(restart_pid, signal.SIGTERM)
                except (OSError, ProcessLookupError, PermissionError):
                    pass
            pending_restart.clear()
            stop_local_web_server()
            if web_console is not None:
                try:
                    web_console.shutdown()
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    logger.debug(
                        "early web control surface shutdown failed: %s", type(exc).__name__
                    )
            if timer is not None:
                try:
                    timer.stop()
                except Exception as exc:
                    logger.debug("Qt timer cleanup failed: %s", type(exc).__name__)
            if cursor_tracking_timer is not None:
                try:
                    cursor_tracking_timer.stop()
                except Exception as exc:
                    logger.debug("cursor tracking timer cleanup failed: %s", type(exc).__name__)
            if tray_watch_timer is not None:
                try:
                    tray_watch_timer.stop()
                except Exception as exc:
                    logger.debug("tray watch cleanup failed: %s", type(exc).__name__)
            pet_feedback_generation += 1
            if pet_feedback_timer is not None:
                try:
                    pet_feedback_timer.stop()
                    pet_feedback_timer.deleteLater()
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    pass
                pet_feedback_timer = None
            if tray is not None:
                try:
                    tray.hide()
                except Exception as exc:
                    logger.debug("tray cleanup failed: %s", type(exc).__name__)
            if hotkey_manager is not None:
                try:
                    hotkey_manager.close()
                except Exception as exc:
                    logger.debug("hotkey cleanup failed: %s", type(exc).__name__)
            # 先取消尚未完成的点击好感度写入，避免 runtime.close 关闭数据库后
            # 仍有 UI 侧协程尝试提交事务。
            for pending, _zone, _phrase in tuple(pending_pet_affection.values()):
                try:
                    pending.cancel()
                except Exception as exc:
                    logger.debug("pet affection cancellation failed: %s", type(exc).__name__)
            pending_pet_affection.clear()
            if pending_cancel_future is not None:
                try:
                    pending_cancel_future.cancel()
                except Exception as exc:
                    logger.debug(
                        "conversation resubmission cancellation failed: %s", type(exc).__name__
                    )
                pending_cancel_future = None
            pending_cancel_operation_id = None
            pending_submit_text = None
            # 麦克风和扬声器属于 Qt 主线程资源；先停止采集、取消 ASR
            # Future 并清空物理播放，再等待 RuntimeLoop 的模型卸载。
            _close_qt_interaction_resources(microphone_controller, audio_player)
            # 先停止后台运行时，再关闭 Qt 调度器；这样运行时收尾阶段仍能完成
            # 必要的主线程调用，不会因为 dispatcher 提前关闭而遗留任务。
            try:
                runtime_loop.stop()
            except Exception as exc:
                logger.warning("runtime loop shutdown failed: %s", type(exc).__name__)
                try:
                    runtime.database.close()
                except Exception as database_error:
                    logger.debug(
                        "database fallback cleanup failed: %s", type(database_error).__name__
                    )
            for pending in tuple(pending_console_reads.values()):
                try:
                    pending.cancel()
                except Exception as exc:
                    logger.debug("console read cancellation failed: %s", type(exc).__name__)
            pending_console_reads.clear()
            for pending in tuple(pending_approval_actions.values()):
                try:
                    pending.cancel()
                except Exception as exc:
                    logger.debug("approval action cancellation failed: %s", type(exc).__name__)
            pending_approval_actions.clear()
            if web_console is not None:
                try:
                    web_console.shutdown()
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    logger.debug("web control surface shutdown failed: %s", type(exc).__name__)
            if window is not None:
                shutdown_window = getattr(window, "shutdown", None)
                if callable(shutdown_window):
                    try:
                        shutdown_window()
                    except Exception as exc:
                        logger.debug("native window shutdown failed: %s", type(exc).__name__)
                close_window = getattr(window, "close", None)
                if callable(close_window):
                    try:
                        close_window()
                    except Exception as exc:
                        logger.debug("native window close failed: %s", type(exc).__name__)
            if web_host is not None:
                web_view = web_host.view
                try:
                    web_host.shutdown()
                except Exception as exc:
                    logger.debug("Web Live2D host shutdown failed: %s", type(exc).__name__)
                    # WebPetHost 拥有 renderer/view 的正常销毁顺序；只有
                    # 宿主自身异常退出时才尝试一次兜底 close，避免正常路径
                    # 在 renderer.shutdown 后重复关闭 Discarded page。
                    close_view = getattr(web_view, "close", None)
                    if callable(close_view):
                        try:
                            close_view()
                        except Exception as close_error:
                            logger.debug(
                                "Web Live2D view fallback close failed: %s",
                                type(close_error).__name__,
                            )
            elif web_renderer is not None:
                try:
                    web_renderer.shutdown()
                except Exception as exc:
                    logger.debug("Web Live2D renderer shutdown failed: %s", type(exc).__name__)
            if mcp_content_window is not None:
                try:
                    mcp_content_window.close()
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    logger.debug("MCP content window shutdown failed: %s", type(exc).__name__)
            # WebEngine 页面在 app.exec() 返回后仍可能有 DeferredDelete 事件；
            # 在解释器销毁 Qt 对象前主动派发，避免留下 profile/page 警告。
            try:
                from PySide6.QtCore import QCoreApplication, QEvent

                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
                QCoreApplication.processEvents()
            except (AttributeError, RuntimeError, TypeError):
                pass
            try:
                if console is not None:
                    console.shutdown()
            except Exception as exc:
                logger.debug("console shutdown failed: %s", type(exc).__name__)
            try:
                dispatcher.close()
            except Exception as exc:
                logger.debug("Qt dispatcher cleanup failed: %s", type(exc).__name__)
            if not owns_runtime:
                try:
                    runtime.database.close()
                except Exception as exc:
                    logger.debug(
                        "injected runtime database cleanup failed: %s",
                        type(exc).__name__,
                    )
        finally:
            restore_signal_handlers()
