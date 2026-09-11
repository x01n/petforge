from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import secrets
import sqlite3
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock
from time import monotonic, time
from typing import Any, cast

from config.loader import (
    ConfigurationError,
    LoadedConfiguration,
    default_configuration_values,
    parse_bool,
    resolve_resource_path,
    validate_local_web_configuration,
)
from config.resources import ResourceInventory, inspect_resources
from config.watcher import ConfigurationWatcher
from core.adapters.direct.errors import AdapterConfigurationError
from core.asr import SUPPORTED_ASR_BACKENDS, SUPPORTED_ASR_LANGUAGES, ASRHealth
from core.conversation.persona import PersonaConfigurationError, PersonaPromptBundle
from core.events.types import ApprovalRequested, ConversationContext, TurnFailed, TurnFinished
from core.rendering.actions import ExpressionRequest, MotionRequest
from core.tts.contracts import EngineHealth, SpeechBackend
from core.tts.language import (
    canonical_tts_language,
    normalize_reference_audios,
    protocol_tts_language,
)
from core.util import as_mapping as _mapping
from core.util import as_sequence as _sequence
from db import (
    ApiCallAuditRecord,
    ApiCallAuditRepository,
    ConversationRepository,
    Database,
    SchedulerRepository,
)
from gui.platforms import create_desktop_platform
from gui.platforms.protocol import DesktopPlatform
from gui.qt6.hotkeys import validate_hotkey_config
from gui.qt6.md3 import MD3ThemeError, theme_from_configuration
from gui.renderers.protocol import normalize_renderer_backend
from logger import LoggingSettings, configure_logging
from services.affection import AffectionService
from services.asr import ASRService
from services.conversation import ConversationService, PresentationService
from services.conversation.interaction_contract import InteractionState
from services.diary import PetDiaryService
from services.memory import MemoryService, MemorySettings
from services.memory.summarizer import MemorySummaryCoordinator
from services.model_routing import ModelRouter, ModelRoutingError
from services.modules import ModuleManager
from services.plugins import PluginManager, PluginSettings
from services.proactive import ProactiveCoordinator, ProactiveSettings
from services.scheduler import (
    SYSTEM_IDLE_PROVIDER_WINDOWS,
    AutonomousMovementConfig,
    BehaviorAction,
    BehaviorService,
    DesktopWindowWatcher,
    SchedulerService,
    TriggerService,
    UserActivityTracker,
    normalize_trigger_conditions,
    validate_schedule_expression,
)
from services.tools import (
    PermissionService,
    ToolExecutionService,
    ToolRegistry,
    normalize_command_allowlist,
)
from services.tools.builtins import register_builtin_tools
from services.tools.mcp_bridge import MCPToolBridge
from services.tools.types import RiskLevel, ToolCallContext
from services.tts import (
    GptSovitsHttpBackend,
    GptSovitsStdioBackend,
    PcmAudioFeatureAnalyzer,
    SubprocessTTSBackend,
    TextOnlyBackend,
    TTSCoordinator,
    TTSProfile,
    TTSProfileRouter,
)

logger = logging.getLogger(__name__)

_RUNTIME_TEST_MODULE_IDS = frozenset(
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
        "watcher",
        "ipc",
        "plugins",
        "diary",
        "modules",
        "api_audit",
    }
)


def _plugin_settings(value: object) -> PluginSettings:
    """用实例默认值补齐插件区，再交给领域解析器执行严格校验。"""

    if value is None:
        return PluginSettings()
    if not isinstance(value, Mapping):
        raise ValueError("plugins must be a mapping")
    defaults = PluginSettings()
    normalized = {
        "enabled": defaults.enabled,
        "directories": defaults.directories,
        "entry_points_enabled": defaults.entry_points_enabled,
        "entry_point_group": defaults.entry_point_group,
        "reload_enabled": defaults.reload_enabled,
        "reload_interval_seconds": defaults.reload_interval_seconds,
    }
    normalized.update(value)
    return PluginSettings.from_mapping(normalized)


def _trigger_metadata(item: Mapping[str, Any]) -> dict[str, Any]:
    raw = item.get("metadata", {})
    metadata = dict(raw) if isinstance(raw, Mapping) else raw
    if not isinstance(metadata, dict):
        raise ValueError("trigger metadata must be a mapping")
    if "conditions" in item:
        metadata["conditions"] = normalize_trigger_conditions(item.get("conditions"))
    return metadata


class MutablePetController:
    def __init__(self, target: object | None = None, dispatcher: object | None = None) -> None:
        self._target = target
        self._dispatcher = dispatcher
        self._click_through_guard: Callable[[], bool] | None = None
        self._interaction_interrupt: Callable[[], object] | None = None
        self._activity_notifier: Callable[[], object] | None = None
        # 视觉动作所有权代次。外部移动/表情/动作会先使当前行为动作失效；
        # 行为内部动作则登记新的代次，收尾时可避免旧 idle 覆盖用户动作。
        self._motion_generation = 0
        self._motion_owner_generation: int | None = None
        self._motion_lock = Lock()
        # Qt 控制台打开时，桌宠自主漫游必须暂停；该标志由 Qt 主线程写入，
        # 行为服务只读取一个线程安全布尔值，避免后台线程直接访问 QWidget。
        self._ui_interaction_locked = False

    def attach(self, target: object) -> None:
        previous = self._target
        if previous is not None and previous is not target:
            self._propagate_interaction_interrupt(previous, None)
            self._propagate_activity_notifier(previous, None)
        if previous is not target:
            # 渲染器替换（例如 Web Live2D 运行期回退到精灵）会使旧
            # 页面/窗口中的动作所有权失效；迟到的行为收尾不得写入新宿主。
            with self._motion_lock:
                self._motion_generation += 1
                self._motion_owner_generation = None
        self._target = target
        self._propagate_interaction_interrupt(target, self._interaction_interrupt)
        self._propagate_activity_notifier(target, self._activity_notifier)

    @staticmethod
    def _propagate_interaction_interrupt(
        target: object | None,
        callback: Callable[[], object] | None,
    ) -> None:

        setter = getattr(target, "set_interaction_interrupt", None) if target is not None else None
        if not callable(setter):
            return
        try:
            result = setter(callback)
            if inspect.isawaitable(result):
                closer = getattr(result, "close", None)
                if callable(closer):
                    closer()
        except Exception:
            logger.debug("failed to propagate interaction interrupt", exc_info=True)

    @staticmethod
    def _propagate_activity_notifier(
        target: object | None,
        callback: Callable[[], object] | None,
    ) -> None:
        """把用户活跃通知器传播到当前窗口宿主。"""

        setter = getattr(target, "set_activity_notifier", None) if target is not None else None
        if not callable(setter):
            return
        try:
            result = setter(callback)
            if inspect.isawaitable(result):
                closer = getattr(result, "close", None)
                if callable(closer):
                    closer()
        except Exception:
            logger.debug("failed to propagate activity notifier", exc_info=True)

    def set_dispatcher(self, dispatcher: object | None) -> None:
        """设置可选的主线程调用边界。"""

        self._dispatcher = dispatcher

    def set_click_through_guard(self, guard: Callable[[], bool] | None) -> None:
        """设置启用点击穿透前的可恢复性检查。"""

        self._click_through_guard = guard

    def set_ui_interaction_locked(self, locked: bool) -> None:
        """在控制台/配置向导占用屏幕时暂停自主行为。"""

        value = bool(locked)
        was_locked = self._ui_interaction_locked
        self._ui_interaction_locked = value
        # 控制台打开时立即使正在等待 Qt/平台回执的自主路径失效。
        # 仅依赖下一次 ``is_user_interacting`` 轮询会允许一个迟到步进
        # 在 UI 已覆盖桌宠后提交，造成位置回弹或渲染抽动。
        if value and not was_locked:
            # 同时推进视觉动作代次，令尚未进入 dispatcher 的自主移动/动作
            # 在真正执行闭包内失效；这样配置向导隐藏桌宠后不会再收到旧回执。
            self.invalidate_visual_actions()

    def set_interaction_interrupt(self, callback: Callable[[], object] | None) -> None:
        """绑定 UI 锁定时使行为代际失效的同步回调。"""

        self._interaction_interrupt = callback
        self._propagate_interaction_interrupt(self._target, callback)

    def set_activity_notifier(self, callback: Callable[[], object] | None) -> None:
        """绑定实际用户手势的活跃状态通知器。"""

        self._activity_notifier = callback
        self._propagate_activity_notifier(self._target, callback)

    def set_window_locked(self, locked: bool) -> Any:
        """锁定/解锁桌宠输入；锁定不暂停自主行为或光标追踪。"""

        return self._invoke("set_window_locked", bool(locked))

    def set_always_on_top(self, enabled: bool) -> Any:
        """刷新桌宠窗口置顶旗标；宿主未绑定时返回不可用状态。"""

        return self._invoke("set_always_on_top", bool(enabled))

    def is_window_locked(self) -> Any:
        """读取桌宠窗口锁定状态。"""

        return self._invoke("is_window_locked")

    def is_always_on_top(self) -> Any:
        """读取桌宠窗口置顶状态。"""

        return self._invoke("is_always_on_top")

    def reload_resources(
        self,
        resource_root: str | Path,
        *,
        model_path: str | Path | None = None,
        sprite_scale: float | None = None,
    ) -> Any:
        """通过 Qt 调度边界请求当前渲染宿主原子交换资源。"""

        return self._invoke(
            "reload_resources",
            resource_root,
            model_path=model_path,
            sprite_scale=sprite_scale,
        )

    def _invoke(self, method: str, *args: object, **kwargs: object) -> Any:
        target = self._target
        handler = getattr(target, method, None) if target is not None else None
        if not callable(handler):
            return {"status": "unavailable", "reason": "pet window is not attached"}
        dispatcher = self._dispatcher
        invoke = getattr(dispatcher, "invoke", None) if dispatcher is not None else None
        if callable(invoke):
            result = invoke(handler, *args, **kwargs)
        else:
            result = handler(*args, **kwargs)
        return self._flatten_awaitable(result)

    @staticmethod
    def _flatten_awaitable(value: object) -> object:

        if not inspect.isawaitable(value):
            return value

        async def resolve() -> object:
            current: object = value
            for _ in range(8):
                if not inspect.isawaitable(current):
                    return current
                current = await current
            return {
                "status": "unavailable",
                "reason": "nested pet operation result is too deep",
            }

        return resolve()

    def _invoke_if_motion_generation(
        self,
        method: str,
        generation: int,
        *args: object,
        **kwargs: object,
    ) -> Any:
        """只在视觉代次未变化时把内部调用投递到宿主线程。

        行为服务运行在独立事件循环，而 Qt 宿主调用可能排队到稍后的
        事件循环。如果外部动作已经抢占了同一视觉状态，迟到的内部
        motion/移动不得再写入渲染器；检查必须放在真正执行的闭包内，
        不能只在投递前检查一次。
        """

        def guarded() -> Any:
            with self._motion_lock:
                current = self._motion_generation
            if current != generation:
                return {
                    "status": "cancelled",
                    "reason": "visual action was superseded",
                }
            return self._invoke(method, *args, **kwargs)

        dispatcher = self._dispatcher
        invoke = getattr(dispatcher, "invoke", None) if dispatcher is not None else None
        if callable(invoke):
            result = invoke(guarded)
        else:
            result = guarded()
        return self._flatten_awaitable(result)

    def _interrupt_external_action(self) -> None:

        callback = self._interaction_interrupt
        if not callable(callback):
            return
        try:
            result = callback()
            if inspect.isawaitable(result):
                closer = getattr(result, "close", None)
                if callable(closer):
                    closer()
        except Exception:
            logger.debug("failed to interrupt behavior for external pet action", exc_info=True)

    def _mark_external_visual_action(self, *, invalidates_motion: bool) -> None:

        with self._motion_lock:
            self._motion_generation += 1
            if invalidates_motion:
                self._motion_owner_generation = None

    def _mark_internal_motion(self) -> int:

        with self._motion_lock:
            self._motion_generation += 1
            self._motion_owner_generation = self._motion_generation
            return self._motion_generation

    def move_to(self, x: float, y: float, *, duration_ms: int = 800) -> Any:
        self._mark_external_visual_action(invalidates_motion=False)
        self._interrupt_external_action()
        return self._invoke("move_to", x, y, duration_ms=duration_ms)

    def move_to_internal(self, x: float, y: float, *, duration_ms: int = 800) -> Any:
        with self._motion_lock:
            generation = self._motion_generation
        return self._invoke_if_motion_generation(
            "move_to",
            generation,
            x,
            y,
            duration_ms=duration_ms,
        )

    def position(self) -> Any:

        return self._invoke("position")

    def movement_bounds(self) -> Any:

        return self._invoke("movement_bounds")

    def set_display_size(self, preset: str) -> Any:

        return self._invoke("set_display_size", preset)

    def is_user_interacting(self) -> Any:

        if self._ui_interaction_locked:
            return True
        return self._invoke("is_user_interacting")

    def set_direction(self, direction: str) -> Any:
        """更新渲染器朝向；宿主不支持时返回不可用状态。"""

        return self._invoke("set_direction", direction)

    def set_rendering_performance(
        self,
        *,
        frame_rate: float | None = None,
        geometry_audit_hz: float | None = None,
    ) -> Any:
        """更新当前渲染器的帧率与几何审计频率。"""

        return self._invoke(
            "set_rendering_performance",
            frame_rate=frame_rate,
            geometry_audit_hz=geometry_audit_hz,
        )

    def set_expression(self, name: str) -> Any:
        self._mark_external_visual_action(invalidates_motion=False)
        self._interrupt_external_action()
        return self._invoke("set_expression", name)

    def set_expression_request(self, request: ExpressionRequest) -> Any:
        """提交多表情、参数和持续时间请求。"""

        if not isinstance(request, ExpressionRequest):
            return {"status": "unavailable", "reason": "expression request is invalid"}
        self._mark_external_visual_action(invalidates_motion=False)
        self._interrupt_external_action()
        return self._invoke("set_expression_request", request)

    def set_expression_internal(self, name: str) -> Any:
        """行为服务内部表情入口，不重新触发自主行为中断。"""

        with self._motion_lock:
            generation = self._motion_generation
        return self._invoke_if_motion_generation("set_expression", generation, name)

    def play_motion(self, name: str) -> Any:
        self._mark_external_visual_action(invalidates_motion=True)
        self._interrupt_external_action()
        return self._invoke("play_motion", name)

    def play_motion_request(self, request: MotionRequest) -> Any:
        """提交带参数、持续时间和循环策略的动作请求。"""

        if not isinstance(request, MotionRequest):
            return {"status": "unavailable", "reason": "motion request is invalid"}
        self._mark_external_visual_action(invalidates_motion=True)
        self._interrupt_external_action()
        return self._invoke("play_motion_request", request)

    def play_motion_internal(self, name: str) -> Any:
        """行为服务内部动作入口，不重新触发自主行为中断。"""

        generation = self._mark_internal_motion()
        return self._invoke_if_motion_generation("play_motion", generation, name)

    def motion_generation(self) -> int:
        """返回视觉动作所有权代次。"""

        with self._motion_lock:
            return int(self._motion_generation)

    def motion_owner_generation(self) -> int | None:
        """返回当前仍由行为服务持有的动作代次；外部动作替换 motion 后返回 ``None``。"""

        with self._motion_lock:
            return self._motion_owner_generation

    def motion_owned_by_behavior(self) -> bool:
        """返回当前动作是否仍由行为服务持有。"""

        with self._motion_lock:
            return self._motion_owner_generation is not None

    def invalidate_visual_actions(self, *, interrupt_behavior: bool = True) -> None:
        """使当前宿主上的在途视觉动作失效。

        渲染器页内换模、运行期回退等操作会保留同一个控制器对象，但旧
        页面/模型的动作回执不能再写入新模型。推进代次并清除所有权后，
        可选地通知行为服务停止旧规划。
        """

        with self._motion_lock:
            self._motion_generation += 1
            self._motion_owner_generation = None
        if interrupt_behavior:
            self._interrupt_external_action()

    def supports_expression(self, name: str) -> bool | None:
        """读取渲染器对表情的明确能力；未提供探针时返回 ``None``。"""

        return self._supports_renderer_action("expression", name)

    def supports_motion(self, name: str) -> bool | None:
        """读取渲染器对动作的明确能力；未提供探针时返回 ``None``。"""

        return self._supports_renderer_action("motion", name)

    def renderer_diagnostics(self) -> Mapping[str, object]:
        """返回当前渲染宿主的脱敏能力，不公开资源绝对路径。"""

        target = self._target
        renderer = getattr(target, "renderer", target) if target is not None else None
        capabilities = getattr(renderer, "capabilities", None)
        if capabilities is None:
            return {"available": False, "backend": "unattached"}
        expressions = tuple(getattr(capabilities, "expressions", ()) or ())
        motions = tuple(getattr(capabilities, "motions", ()) or ())
        return {
            "available": bool(getattr(capabilities, "available", False)),
            "backend": str(getattr(capabilities, "backend", "unknown") or "unknown")[:64],
            "expression_count": min(256, len(expressions)),
            "motion_count": min(256, len(motions)),
            "message": str(getattr(capabilities, "message", "") or "")[:240],
        }

    def _supports_renderer_action(self, kind: str, name: str) -> bool | None:
        target = self._target
        renderer = getattr(target, "renderer", target) if target is not None else None
        checker = getattr(renderer, f"supports_{kind}", None)
        if callable(checker):
            try:
                result = checker(name)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return False
            return bool(result)
        capabilities = getattr(renderer, "capabilities", None)
        field = "expressions" if kind == "expression" else "motions"
        values = getattr(capabilities, field, None)
        if values is None:
            return None
        try:
            declared = tuple(str(item).strip() for item in values if str(item).strip())
        except (TypeError, ValueError):
            return None
        if not declared:
            return None
        return str(name or "").strip() in declared

    def speak(self, text: str, *, mood: str = "neutral") -> Any:
        return self._invoke("set_speech", text, mood=mood)

    def set_click_through(self, enabled: bool) -> Any:
        guard = self._click_through_guard
        if bool(enabled) and guard is not None:
            try:
                recoverable = bool(guard())
            except Exception:
                recoverable = False
            if not recoverable:
                return {
                    "status": "unavailable",
                    "reason": "click-through recovery entry is unavailable",
                }
        return self._invoke("set_click_through", bool(enabled))


class DispatchingDesktopPlatform:
    """把 Qt 相关平台调用转发到宿主线程，其余只读能力保持后台可用。"""

    def __init__(self, platform: DesktopPlatform, dispatcher: object | None = None) -> None:
        self._platform = platform
        self._dispatcher = dispatcher

    def set_dispatcher(self, dispatcher: object | None) -> None:
        """更新 Qt 主线程调用边界。"""

        self._dispatcher = dispatcher

    @property
    def backend(self) -> str:
        return str(self._platform.backend)

    def probe(self) -> object:
        return self._platform.probe()

    def active_window(self) -> object:
        return self._platform.active_window()

    def cursor_position(self) -> Mapping[str, object]:
        """读取平台提供的只读全局光标位置；缺失或异常时 fail-closed。"""

        getter = getattr(self._platform, "cursor_position", None)
        if not callable(getter):
            return {
                "status": "unavailable",
                "backend": self.backend,
                "reason": "cursor position is unavailable",
            }
        try:
            value = getter()
        except Exception:
            return {
                "status": "unavailable",
                "backend": self.backend,
                "reason": "cursor position read failed",
            }
        if not isinstance(value, Mapping):
            return {
                "status": "unavailable",
                "backend": self.backend,
                "reason": "cursor position returned non-mapping",
            }
        result = dict(value)
        if str(result.get("status", "")).strip().lower() != "available":
            return result
        x = result.get("x")
        y = result.get("y")
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, int)
            or not isinstance(y, int)
        ):
            return {
                "status": "unavailable",
                "backend": self.backend,
                "reason": "cursor position coordinates are invalid",
            }
        return {
            "status": "available",
            "backend": str(result.get("backend", self.backend)),
            "x": int(x),
            "y": int(y),
            "source": str(result.get("source", ""))[:80],
        }

    def system_idle_seconds(self) -> float | None:
        """读取后台平台提供的系统空闲秒数；未实现时 fail-closed。"""

        getter = getattr(self._platform, "system_idle_seconds", None)
        if not callable(getter):
            return None
        try:
            value = getter()
        except Exception:
            return None
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def foreground_window(self) -> Mapping[str, object]:
        return self._platform.foreground_window()

    def list_processes(self, limit: int = 20) -> Mapping[str, object]:
        return self._platform.list_processes(limit)

    def capture_screen(
        self,
        *,
        scope: str = "screen",
        region: Mapping[str, object] | None = None,
        window_id: int | None = None,
    ) -> object:
        normalized_scope = str(scope or "screen").strip().lower()
        if normalized_scope == "window" and window_id is None:
            return {
                "status": "unavailable",
                "reason": "window id must be resolved before Qt screen capture",
            }
        capture_kwargs: dict[str, object] = {"scope": scope, "region": region}
        if window_id is not None:
            capture_kwargs["window_id"] = int(window_id)
        return self._dispatch("capture_screen", **capture_kwargs)

    def click_at(self, x: int, y: int, *, button: str = "left") -> object:
        """输入注入不依赖 Qt 对象，直接留在工具运行线程执行。"""

        handler = getattr(self._platform, "click_at", None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "coordinate input is unavailable"}
        return handler(x, y, button=button)

    def automation_batch(self, steps: object) -> object:
        """把可能包含等待的系统自动化移出运行时事件循环。"""

        handler = getattr(self._platform, "automation_batch", None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "desktop automation is unavailable"}
        # SendInput/SetWindowPos 是同步 Win32 调用，步骤中的 wait 也有总上限；
        # 放入线程后不会阻塞对话流、TTS 或配置热重载事件循环。
        return asyncio.to_thread(handler, steps)

    def resolve_capture_window_id(self) -> int | None:
        """在后台读取窗口 ID，避免 X11/进程读取进入 Qt 主线程。"""

        resolver = getattr(self._platform, "resolve_capture_window_id", None)
        if callable(resolver):
            return resolver()
        window = self._platform.active_window()
        window_id = getattr(window, "window_id", "") if window is not None else ""
        try:
            return int(window_id, 16) if window_id else None
        except (TypeError, ValueError):
            return None

    def set_click_through(self, window: object, enabled: bool) -> object:
        return self._dispatch("set_click_through", window, enabled)

    def set_input_shape(self, window: object, region: object | None) -> object:
        """把原生 Input Shape 请求转发到 Qt 所在线程。"""

        return self._dispatch("set_input_shape", window, region)

    def set_always_on_top(self, window: object, enabled: bool) -> object:
        """把 Windows 原生置顶请求转发到 Qt 所在线程。"""

        handler = getattr(self._platform, "set_always_on_top", None)
        if not callable(handler):
            return {"status": "unavailable", "reason": "native topmost control is unavailable"}
        return self._dispatch("set_always_on_top", window, enabled)

    def move_overlay(self, window: object, x: int, y: int) -> object:
        return self._dispatch("move_overlay", window, x, y)

    def exclude_window_id(self, window_id: int) -> None:
        handler = getattr(self._platform, "exclude_window_id", None)
        if callable(handler):
            handler(window_id)

    def _dispatch(self, method: str, *args: object, **kwargs: object) -> object:
        handler = getattr(self._platform, method)
        dispatcher = self._dispatcher
        invoke = getattr(dispatcher, "invoke", None) if dispatcher is not None else None
        if callable(invoke):
            return invoke(handler, *args, **kwargs)
        return handler(*args, **kwargs)


@dataclass
class ApplicationRuntime:
    """可由 CLI、Qt 或测试宿主共同使用的服务集合。"""

    configuration: LoadedConfiguration
    inventory: ResourceInventory
    database: Database
    api_audit: ApiCallAuditRepository | None = field(default=None, init=False, repr=False)
    memory: MemoryService
    affection: AffectionService
    router: ModelRouter
    registry: ToolRegistry
    permissions: PermissionService
    tools: ToolExecutionService
    scheduler: SchedulerService
    triggers: TriggerService
    watcher: DesktopWindowWatcher
    behavior: BehaviorService
    tts: TTSCoordinator
    asr: ASRService
    conversation: ConversationService
    pet_controller: MutablePetController
    platform: DesktopPlatform
    mcp_clients: tuple[object, ...] = ()
    mcp_bridges: tuple[MCPToolBridge, ...] = ()
    # 独立插件管理器负责外部模块发现、代次和失败隔离；默认实例保持
    # 无插件兼容性，便于测试宿主直接构造 ApplicationRuntime。
    plugins: PluginManager = field(default_factory=PluginManager, init=False, repr=False)
    # 统一管理内置服务的生命周期；已构造对象以 adopt 方式登记，避免重复启动。
    modules: ModuleManager = field(default_factory=ModuleManager, init=False, repr=False)
    diary: PetDiaryService | None = field(default=None, init=False, repr=False)
    activity: UserActivityTracker | None = field(default=None, init=False, repr=False)
    mcp_status: dict[str, str] = field(default_factory=dict)
    # 配置文件观察器由 ``build_runtime`` 在组合根完成后绑定；保持可选，
    # 以兼容无文件/测试宿主直接构造 ``ApplicationRuntime`` 的场景。
    configuration_watcher: ConfigurationWatcher | None = field(default=None, init=False, repr=False)
    memory_summarizer: MemorySummaryCoordinator | None = field(default=None, init=False, repr=False)
    proactive: ProactiveCoordinator | None = field(default=None, init=False, repr=False)
    # Web/Qt 宿主消费的不可变交互快照；不把工具参数或密钥暴露给 UI。
    interaction_state: InteractionState | None = field(default=None, init=False, repr=False)
    persona_prompts: PersonaPromptBundle = field(
        default_factory=lambda: PersonaPromptBundle.from_configuration({}),
        init=False,
        repr=False,
    )
    tts_startup_health: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    asr_startup_health: dict[str, object] = field(default_factory=dict, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _closing: bool = field(default=False, init=False, repr=False)
    _startup_emitted: bool = field(default=False, init=False, repr=False)
    _startup_task: asyncio.Task[object] | None = field(default=None, init=False, repr=False)
    _close_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    _tts_start_task: asyncio.Task[object] | None = field(default=None, init=False, repr=False)
    _asr_start_task: asyncio.Task[object] | None = field(default=None, init=False, repr=False)
    _mcp_started: bool = field(default=False, init=False, repr=False)
    _background_started: bool = field(default=False, init=False, repr=False)
    _configuration_reload_lock: asyncio.Lock = field(
        default_factory=asyncio.Lock, init=False, repr=False
    )
    _configuration_reload_tasks: set[asyncio.Task[Any]] = field(
        default_factory=set, init=False, repr=False
    )
    _reload_cleanup_tasks: set[asyncio.Task[bool]] = field(
        default_factory=set, init=False, repr=False
    )
    _reload_cleanup_failures: set[str] = field(default_factory=set, init=False, repr=False)
    _reload_cleanup_required: bool = field(default=False, init=False, repr=False)
    _mcp_content_approvals: dict[str, ToolCallContext] = field(
        default_factory=dict, init=False, repr=False
    )
    _mcp_content_approval_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        """登记组合根中的内置服务；实际启动仍由现有事务控制。"""

        self._register_builtin_modules()

    def _register_builtin_modules(self) -> None:
        """把已构造对象接入统一模块表，不调用其生命周期钩子。"""

        entries: tuple[tuple[str, object, tuple[str, ...]], ...] = (
            ("model", self.router, ()),
            ("tts", self.tts, ()),
            ("asr", self.asr, ("tts",)),
            ("memory", self.memory, ()),
            ("tools", self.tools, ("memory",)),
            ("mcp", self.mcp_clients, ("tools",)),
            ("scheduler", self.scheduler, ("memory",)),
            ("behavior", self.behavior, ("scheduler",)),
            ("renderer", self.pet_controller, ()),
            ("watcher", self.watcher, ()),
            ("plugins", self.plugins, ("tools",)),
        )
        for module_id, instance, dependencies in entries:
            if self.modules.status(module_id).get("state") != "unavailable":
                continue
            self.modules.adopt(module_id, instance, dependencies=dependencies)

    def _register_optional_modules(self) -> None:
        """在组合根补齐延迟创建的日记、主动行为和配置观察器。"""

        optional: tuple[tuple[str, object | None, tuple[str, ...]], ...] = (
            ("diary", self.diary, ("memory",)),
            ("proactive", self.proactive, ("behavior", "memory")),
            ("config_watcher", self.configuration_watcher, ()),
            ("api_audit", self.api_audit, ()),
        )
        for module_id, instance, dependencies in optional:
            if instance is None or self.modules.status(module_id).get("state") != "unavailable":
                continue
            self.modules.adopt(module_id, instance, dependencies=dependencies)

    def _schedule_reload_cleanup(
        self,
        resource: object | None,
        *,
        label: str,
    ) -> asyncio.Task[bool] | None:
        """让运行时持有热替换清理任务，直到资源关闭进入终态。"""

        if resource is None:
            return None
        close = getattr(resource, "aclose", None)
        if not callable(close):
            close = getattr(resource, "close", None)
        if not callable(close):
            return None

        async def cleanup() -> bool:
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("热重载资源清理失败，需要重启回收：%s", label)
                return True
            return False

        task = asyncio.create_task(cleanup(), name=f"meapet-reload-cleanup-{label}")
        self._reload_cleanup_tasks.add(task)
        self._reload_cleanup_required = True

        def finished(done: asyncio.Task[bool]) -> None:
            self._reload_cleanup_tasks.discard(done)
            failed = done.cancelled()
            if not failed:
                try:
                    failed = bool(done.result())
                except BaseException:
                    failed = True
            if failed:
                self._reload_cleanup_failures.add(label)
            if not self._reload_cleanup_tasks:
                self._reload_cleanup_required = bool(self._reload_cleanup_failures)

        task.add_done_callback(finished)
        return task

    async def _wait_reload_cleanups(
        self,
        tasks: Sequence[asyncio.Task[bool] | None],
    ) -> bool:
        """等待指定清理终态；取消等待不会取消实际资源回收。"""

        pending = tuple(task for task in tasks if task is not None)
        if not pending:
            return False
        waiter = asyncio.gather(*pending, return_exceptions=True)
        try:
            results = await asyncio.shield(waiter)
        except asyncio.CancelledError:
            self._reload_cleanup_required = True
            raise
        failed = any(result is True or isinstance(result, BaseException) for result in results)
        if failed:
            self._reload_cleanup_required = True
        return failed

    async def _drain_reload_cleanups(self) -> bool:
        """在运行时退出前等待仍被追踪的热重载清理任务。"""

        failed = False
        while self._reload_cleanup_tasks:
            failed = await self._wait_reload_cleanups(tuple(self._reload_cleanup_tasks)) or failed
        return failed or bool(self._reload_cleanup_failures)

    async def _cancel_configuration_reloads(self) -> None:
        """取消并等待所有配置事务完成其 replacement 所有权收尾。"""

        current = asyncio.current_task()
        while True:
            tasks = tuple(
                task
                for task in self._configuration_reload_tasks
                if task is not current and not task.done()
            )
            if not tasks:
                return
            for task in tasks:
                task.cancel()
            waiter = asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                # close 自身的等待者取消不改变资源所有权；下一轮继续等待
                # 原事务完成 finally 与 replacement 清理。
                continue

    def bind_ui_dispatcher(self, dispatcher: object | None) -> None:
        """为已有运行时绑定 Qt 调度器，并同步观察器与工具使用的平台代理。"""

        if isinstance(self.platform, DispatchingDesktopPlatform):
            self.platform.set_dispatcher(dispatcher)
            platform_value = self.platform
        else:
            platform_value = DispatchingDesktopPlatform(self.platform, dispatcher)
            self.platform = platform_value
        self.watcher.set_platform(platform_value)
        self.pet_controller.set_dispatcher(dispatcher)

    def notify_user_interaction(self, source: object = "runtime") -> Mapping[str, object]:
        """记录一个不携带内容的用户交互边沿。

        Qt/其他宿主可从自己的事件回调调用此同步方法；活跃状态服务会把
        事件安全投递到 RuntimeLoop 所在线程，不在调用线程执行工具。
        """

        if self._closed or self._closing:
            return {"status": "unavailable", "reason": "runtime is closed"}
        if self.proactive is not None:
            self.proactive.interrupt()
        activity = self.activity
        if activity is None:
            return {"status": "unavailable", "reason": "activity tracker is unavailable"}
        try:
            return activity.record_interaction(source)
        except (TypeError, ValueError):
            return {"status": "unavailable", "reason": "activity source is invalid"}

    def activity_status(self) -> Mapping[str, object]:
        """返回脱敏的用户活跃/空闲状态，供控制台或 Web 状态卡读取。"""

        activity = self.activity
        if activity is None:
            return {"status": "unavailable", "reason": "activity tracker is unavailable"}
        try:
            return activity.status()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable"}

    def api_call_audit_records(
        self, *, limit: int = 100, **filters: object
    ) -> tuple[ApiCallAuditRecord, ...]:
        """读取模型与工具 API 调用审计记录，供控制台和外部宿主按需查询。"""

        repository = self.api_audit
        if repository is None:
            return ()
        return repository.list_recent(limit=limit, **filters)

    def api_call_audit_count(self, **filters: object) -> int:
        """返回满足筛选条件的 API 调用审计数量。"""

        repository = self.api_audit
        if repository is None:
            return 0
        return repository.count(**filters)

    def api_call_audit_public_records(
        self, *, limit: int = 50, **filters: object
    ) -> tuple[Mapping[str, object], ...]:
        """返回供本地控制台查看的有界、已脱敏调用摘要。"""

        records = self.api_call_audit_records(limit=limit, **filters)
        return tuple(record.as_dict() for record in records)

    def module_diagnostics(self) -> Mapping[str, object]:
        """汇总运行时模块的固定、脱敏健康字段。"""

        self._register_optional_modules()
        memory_status = self.memory.status()
        summarizer_status = (
            self.memory_summarizer.status().public()
            if self.memory_summarizer is not None
            else {"running": False, "last_status": "unavailable"}
        )
        scheduler_status = self.scheduler.status()
        trigger_status = self.triggers.status()
        behavior_status = self.behavior.status()
        watcher_status = self.watcher.status()
        tts_status = dict(self.tts_diagnostics())
        asr_status = dict(self.asr_diagnostics())
        proactive_status = (
            self.proactive.diagnostics()
            if self.proactive is not None
            else {"status": "unavailable", "enabled": False, "running": False}
        )

        ipc_workers: list[dict[str, object]] = []
        for module_id, diagnostics in (("tts", tts_status), ("asr", asr_status)):
            ipc = diagnostics.get("ipc")
            if not isinstance(ipc, Mapping):
                continue
            pid_value = ipc.get("pid")
            ipc_workers.append(
                {
                    "id": module_id,
                    "status": str(
                        ipc.get("status", diagnostics.get("status", "unavailable")) or "unavailable"
                    )[:32],
                    "running": bool(ipc.get("running", False)),
                    "ready": bool(ipc.get("ready", False)),
                    "pid": (
                        int(pid_value)
                        if isinstance(pid_value, int) and not isinstance(pid_value, bool)
                        else None
                    ),
                }
            )
        ready_count = sum(1 for item in ipc_workers if bool(item.get("ready", False)))
        pending = any(
            str(item.get("status", "")) in {"queued", "loading", "pending"} for item in ipc_workers
        )
        ipc_status = (
            "ready"
            if ipc_workers and ready_count == len(ipc_workers)
            else "loading"
            if pending
            else "degraded"
            if ready_count
            else "unavailable"
        )
        task_count_value = scheduler_status.get("task_count", 0)
        task_count = (
            int(task_count_value)
            if isinstance(task_count_value, (int, float)) and not isinstance(task_count_value, bool)
            else 0
        )
        trigger_count_value = trigger_status.get("trigger_count", 0)
        trigger_count = (
            int(trigger_count_value)
            if isinstance(trigger_count_value, (int, float))
            and not isinstance(trigger_count_value, bool)
            else 0
        )
        foreground_value = watcher_status.get("foreground_window_read", {})
        foreground_window_read = (
            dict(foreground_value) if isinstance(foreground_value, Mapping) else {}
        )
        return {
            "persona": {
                "name": self.persona_prompts.persona.name[:80],
                "role": self.persona_prompts.persona.role[:160],
                "proactive_enabled": self.persona_prompts.persona.proactive_enabled,
            },
            "memory": {
                key: memory_status.get(key)
                for key in (
                    "enabled",
                    "summarization_enabled",
                    "vector_index_size",
                    "vector_index_features",
                    "lexical_index",
                    "semantic_enabled",
                    "semantic_index",
                    "semantic_index_size",
                    "semantic_pending_mutations",
                    "semantic_degraded_reason",
                    "semantic_model_id",
                    "semantic_model_revision",
                    "semantic_dimension",
                )
            },
            "memory_summary": summarizer_status,
            "diary": (
                dict(self.diary.status())
                if self.diary is not None
                else {"enabled": False, "public_entry_count": 0}
            ),
            "renderer": dict(self.pet_controller.renderer_diagnostics()),
            "scheduler": {
                "running": bool(scheduler_status.get("running", False)),
                "task_count": task_count,
                "persistence": str(scheduler_status.get("persistence", "detached"))[:32],
                "trigger_count": trigger_count,
            },
            "behavior": {
                "running": bool(behavior_status.get("running", False)),
                "phase": str(behavior_status.get("phase", "idle"))[:32],
                "movement_enabled": bool(behavior_status.get("movement_enabled", False)),
            },
            "proactive": dict(proactive_status),
            "watcher": {
                "running": bool(watcher_status.get("running", False)),
                "foreground_window_read": foreground_window_read,
            },
            "tts": tts_status,
            "asr": asr_status,
            "ipc": {
                "worker_count": len(ipc_workers),
                "ready_count": ready_count,
                "status": ipc_status,
                "workers": tuple(ipc_workers),
            },
            "mcp": {
                "configured": len(self.mcp_clients),
                "ready": sum(1 for value in self.mcp_status.values() if value == "ready"),
                "content": tuple(self.mcp_content_servers()),
            },
            "api_audit": {
                "enabled": self.api_audit is not None,
                "record_count": self.api_call_audit_count(),
            },
            "plugins": dict(self.plugins.status()),
            "modules": dict(self.modules.status()),
        }

    async def test_module(self, module_id: str) -> Mapping[str, object]:
        """执行单个运行时模块的本地、脱敏且无跨模块副作用的探针。"""

        normalized = str(module_id or "").strip().casefold()
        if normalized not in _RUNTIME_TEST_MODULE_IDS:
            return {
                "module": "unknown",
                "status": "unavailable",
                "available": False,
                "ready": False,
                "reason_code": "unknown_module",
                "latency_ms": 0.0,
            }
        started = monotonic()
        if self._closed:
            return {
                "module": normalized,
                "status": "stopped",
                "available": False,
                "ready": False,
                "reason_code": "runtime_closed",
                "latency_ms": 0.0,
            }

        if normalized == "tts":
            result = dict(await self.test_tts_module())
            result["module"] = normalized
            return result
        if normalized == "asr":
            result = dict(await self.test_asr())
            result["module"] = normalized
            result.setdefault("latency_ms", round((monotonic() - started) * 1000, 3))
            return result

        def nonnegative_integer(value: object) -> int:
            if isinstance(value, int) and not isinstance(value, bool):
                return max(0, value)
            if isinstance(value, float) and math.isfinite(value):
                return max(0, int(value))
            return 0

        def public_token(
            value: object,
            *,
            fallback: str,
            allowed: frozenset[str] | None = None,
        ) -> str:
            token = str(value or "").strip().casefold()
            if allowed is not None:
                return token if token in allowed else fallback
            if not token or len(token) > 64:
                return fallback
            if any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in token
            ):
                return fallback
            return token

        try:
            if normalized == "model":
                diagnostics = self.router.diagnostics("dialogue")
                raw_channels = diagnostics.get("channels", ())
                channels = (
                    tuple(item for item in raw_channels if isinstance(item, Mapping))
                    if isinstance(raw_channels, Sequence)
                    and not isinstance(raw_channels, (str, bytes, bytearray))
                    else ()
                )
                protocols = {
                    str(item.get("protocol", "")).strip().casefold()
                    for item in channels
                    if str(item.get("protocol", "")).strip()
                }
                ready_count = sum(1 for item in channels if item.get("ready") is True)
                ready = bool(diagnostics.get("ready", False))
                result: dict[str, object] = {
                    "module": normalized,
                    "status": "ready" if ready else "degraded" if channels else "unavailable",
                    "available": ready,
                    "ready": ready,
                    "channel_count": len(channels),
                    "ready_count": ready_count,
                    "adapter_count": len(protocols),
                }
            elif normalized == "memory":
                diagnostics = self.memory.status()
                enabled = bool(diagnostics.get("enabled", False))
                result = {
                    "module": normalized,
                    "status": "ready" if enabled else "disabled",
                    "available": True,
                    "ready": enabled,
                    "enabled": enabled,
                    "summarization_enabled": bool(diagnostics.get("summarization_enabled", False)),
                    "vector_index_size": nonnegative_integer(
                        diagnostics.get("vector_index_size", 0)
                    ),
                    "vector_index_features": nonnegative_integer(
                        diagnostics.get("vector_index_features", 0)
                    ),
                    "lexical_index": public_token(
                        diagnostics.get("lexical_index"),
                        fallback="unknown",
                        allowed=frozenset({"fts5", "sparse_fallback"}),
                    ),
                }
            elif normalized == "tools":
                identities = self.registry.identities()
                public_tools = self.registry.visible()
                result = {
                    "module": normalized,
                    "status": "ready",
                    "available": True,
                    "ready": True,
                    "tool_count": len(identities),
                    "public_tool_count": len(public_tools),
                }
            elif normalized == "plugins":
                diagnostics = self.plugins.status()
                enabled = bool(diagnostics.get("enabled", False))
                result = {
                    "module": normalized,
                    "status": str(diagnostics.get("status", "unavailable"))[:32],
                    "available": enabled,
                    "ready": enabled and int(diagnostics.get("failed_count", 0) or 0) == 0,
                    "enabled": enabled,
                    "running": bool(diagnostics.get("running", False)),
                    "plugin_count": nonnegative_integer(diagnostics.get("plugin_count", 0)),
                    "loaded_count": nonnegative_integer(diagnostics.get("loaded_count", 0)),
                    "running_count": nonnegative_integer(diagnostics.get("running_count", 0)),
                    "failed_count": nonnegative_integer(diagnostics.get("failed_count", 0)),
                    "discovery_error_count": nonnegative_integer(
                        diagnostics.get("discovery_error_count", 0)
                    ),
                }
            elif normalized == "modules":
                diagnostics = self.modules.status()
                module_values = diagnostics.get("modules", {})
                module_count = len(module_values) if isinstance(module_values, Mapping) else 0
                failed_count = (
                    sum(
                        1
                        for value in module_values.values()
                        if isinstance(value, Mapping) and str(value.get("state", "")) == "failed"
                    )
                    if isinstance(module_values, Mapping)
                    else 0
                )
                result = {
                    "module": normalized,
                    "status": "degraded" if failed_count else "ready",
                    "available": module_count > 0,
                    "ready": bool(diagnostics.get("running", False)) and failed_count == 0,
                    "module_count": module_count,
                    "failed_count": failed_count,
                    "running": bool(diagnostics.get("running", False)),
                }
            elif normalized == "diary":
                diagnostics = (
                    self.diary.status()
                    if self.diary is not None
                    else {"enabled": False, "public_entry_count": 0}
                )
                result = {
                    "module": normalized,
                    "status": "ready" if diagnostics.get("enabled") is True else "disabled",
                    "available": True,
                    "ready": diagnostics.get("enabled") is True,
                    "enabled": bool(diagnostics.get("enabled", False)),
                    "public_entry_count": nonnegative_integer(
                        diagnostics.get("public_entry_count", 0)
                    ),
                }
            elif normalized == "mcp":
                configured = len(self.mcp_clients)
                ready_count = sum(1 for value in self.mcp_status.values() if value == "ready")
                result = {
                    "module": normalized,
                    "status": (
                        "ready"
                        if configured > 0 and ready_count == configured
                        else "degraded"
                        if configured > 0
                        else "disabled"
                    ),
                    "available": ready_count > 0,
                    "configured": configured,
                    "ready": ready_count,
                    "ready_count": ready_count,
                }
            elif normalized == "api_audit":
                count = self.api_call_audit_count()
                result = {
                    "module": normalized,
                    "status": "ready" if self.api_audit is not None else "disabled",
                    "available": self.api_audit is not None,
                    "ready": self.api_audit is not None,
                    "record_count": nonnegative_integer(count),
                }
            elif normalized == "scheduler":
                scheduler_status = self.scheduler.status()
                trigger_status = self.triggers.status()
                persistence = public_token(
                    scheduler_status.get("persistence"),
                    fallback="unknown",
                    allowed=frozenset({"attached", "detached"}),
                )
                result = {
                    "module": normalized,
                    "status": "ready",
                    "available": True,
                    "ready": True,
                    "running": bool(scheduler_status.get("running", False)),
                    "task_count": nonnegative_integer(scheduler_status.get("task_count", 0)),
                    "trigger_count": nonnegative_integer(trigger_status.get("trigger_count", 0)),
                    "persistence": persistence,
                }
            elif normalized == "behavior":
                diagnostics = self.behavior.status()
                result = {
                    "module": normalized,
                    "status": "ready",
                    "available": True,
                    "ready": True,
                    "running": bool(diagnostics.get("running", False)),
                    "phase": public_token(
                        diagnostics.get("phase"),
                        fallback="stopped",
                        allowed=frozenset(
                            {"stopped", "idle", "planning", "moving", "acting", "paused"}
                        ),
                    ),
                    "movement_enabled": bool(diagnostics.get("movement_enabled", False)),
                    "action_count": nonnegative_integer(diagnostics.get("action_count", 0)),
                    "movement_count": nonnegative_integer(diagnostics.get("movement_count", 0)),
                }
            elif normalized == "proactive":
                diagnostics = (
                    self.proactive.diagnostics()
                    if self.proactive is not None
                    else {"status": "unavailable", "enabled": False}
                )
                enabled = bool(diagnostics.get("enabled", False))
                result = {
                    "module": normalized,
                    "status": public_token(diagnostics.get("status"), fallback="unavailable"),
                    "available": self.proactive is not None,
                    "ready": enabled and self.proactive is not None,
                    "enabled": enabled,
                    "running": bool(diagnostics.get("running", False)),
                    "rule_count": nonnegative_integer(diagnostics.get("rule_count", 0)),
                    "pending_events": nonnegative_integer(diagnostics.get("pending_events", 0)),
                    "hourly_used": nonnegative_integer(diagnostics.get("hourly_used", 0)),
                    "hourly_budget": nonnegative_integer(diagnostics.get("hourly_budget", 0)),
                    "daily_used": nonnegative_integer(diagnostics.get("daily_used", 0)),
                    "daily_budget": nonnegative_integer(diagnostics.get("daily_budget", 0)),
                }
            elif normalized == "renderer":
                diagnostics = self.pet_controller.renderer_diagnostics()
                available = bool(diagnostics.get("available", False))
                result = {
                    "module": normalized,
                    "status": "ready" if available else "unavailable",
                    "available": available,
                    "ready": available,
                    "backend": public_token(
                        diagnostics.get("backend"),
                        fallback="unknown",
                    ),
                    "expression_count": nonnegative_integer(diagnostics.get("expression_count", 0)),
                    "motion_count": nonnegative_integer(diagnostics.get("motion_count", 0)),
                }
            elif normalized == "watcher":
                diagnostics = self.watcher.status()
                foreground = diagnostics.get("foreground_window_read", {})
                foreground = foreground if isinstance(foreground, Mapping) else {}
                result = {
                    "module": normalized,
                    "status": "ready",
                    "available": True,
                    "ready": True,
                    "running": bool(diagnostics.get("running", False)),
                    "foreground_window_read": {
                        "status": public_token(
                            foreground.get("status"),
                            fallback="idle",
                            allowed=frozenset(
                                {"idle", "in_flight", "timed_out", "stale_in_flight"}
                            ),
                        ),
                        "in_flight": bool(foreground.get("in_flight", False)),
                        "recoverable": bool(foreground.get("recoverable", False)),
                        "stale_in_flight": nonnegative_integer(
                            foreground.get("stale_in_flight", 0)
                        ),
                    },
                }
            else:
                workers: list[dict[str, object]] = []
                for worker_id, diagnostics in (
                    ("tts", self.tts_diagnostics()),
                    ("asr", self.asr_diagnostics()),
                ):
                    ipc = diagnostics.get("ipc")
                    if not isinstance(ipc, Mapping):
                        continue
                    pid_value = ipc.get("pid")
                    workers.append(
                        {
                            "id": worker_id,
                            "status": public_token(
                                ipc.get("status", diagnostics.get("status")),
                                fallback="unavailable",
                                allowed=frozenset(
                                    {
                                        "disabled",
                                        "stopped",
                                        "queued",
                                        "starting",
                                        "loading",
                                        "pending",
                                        "ready",
                                        "degraded",
                                        "unavailable",
                                    }
                                ),
                            ),
                            "running": bool(ipc.get("running", False)),
                            "ready": bool(ipc.get("ready", False)),
                            "pid": (
                                pid_value
                                if isinstance(pid_value, int)
                                and not isinstance(pid_value, bool)
                                and pid_value > 0
                                else None
                            ),
                        }
                    )
                ready_count = sum(1 for item in workers if item["ready"] is True)
                pending = any(
                    item["status"] in {"queued", "starting", "loading", "pending"}
                    for item in workers
                )
                status = (
                    "ready"
                    if workers and ready_count == len(workers)
                    else "loading"
                    if pending
                    else "degraded"
                    if ready_count > 0
                    else "unavailable"
                )
                result = {
                    "module": normalized,
                    "status": status,
                    "available": ready_count > 0,
                    "ready": bool(workers) and ready_count == len(workers),
                    "worker_count": len(workers),
                    "ready_count": ready_count,
                    "workers": tuple(workers),
                }
        except asyncio.CancelledError:
            raise
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            result = {
                "module": normalized,
                "status": "unavailable",
                "available": False,
                "ready": False,
                "reason_code": "probe_failed",
            }
        result["latency_ms"] = round((monotonic() - started) * 1000, 3)
        return result

    def tts_diagnostics(self) -> Mapping[str, object]:
        """返回可公开的 TTS profile/语言状态，不携带端点或密钥。"""

        backend = getattr(self.tts, "backend", None)
        diagnostics = getattr(backend, "diagnostics", None)
        result: dict[str, object] = {}
        if callable(diagnostics):
            try:
                value = diagnostics()
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                value = None
            if isinstance(value, Mapping):
                result.update(
                    {
                        "backend": str(value.get("backend", "") or "")[:128],
                        "mode": str(value.get("mode", "") or "")[:32],
                        "model_ready": bool(value.get("model_ready", False)),
                    }
                )
                raw_ipc = value.get("ipc", {})
                if isinstance(raw_ipc, Mapping):
                    result["ipc"] = {
                        "status": str(raw_ipc.get("status", "unknown") or "unknown")[:32],
                        "running": bool(raw_ipc.get("running", False)),
                        "ready": bool(raw_ipc.get("ready", False)),
                        "pid": (
                            int(raw_ipc["pid"])
                            if isinstance(raw_ipc.get("pid"), int)
                            and not isinstance(raw_ipc.get("pid"), bool)
                            else None
                        ),
                    }
                raw_profiles = value.get("profiles", ())
                if isinstance(raw_profiles, (str, bytes, bytearray, Mapping)):
                    raw_profiles = ()
                try:
                    profile_rows = tuple(
                        {
                            "id": str(item.get("id", "") or "")[:128],
                            "languages": tuple(
                                str(language or "")[:16]
                                for language in item.get("languages", ())
                                if str(language or "").strip()
                            )[:8],
                            "roles": tuple(
                                str(role or "")[:128]
                                for role in item.get("roles", ())
                                if str(role or "").strip()
                            )[:8],
                            "enabled": bool(item.get("enabled", True)),
                            "active": str(item.get("id", "") or "")
                            == str(value.get("active_profile", "") or ""),
                        }
                        for item in raw_profiles
                        if isinstance(item, Mapping) and str(item.get("id", "") or "").strip()
                    )[:32]
                except TypeError:
                    profile_rows = ()
                result.update(
                    {
                        "active_profile": str(value.get("active_profile", "") or "")[:128],
                        "default_profile": str(value.get("default_profile", "") or "")[:128],
                        "role_profiles": {
                            str(role)[:128]: str(profile)[:128]
                            for role, profile in value.get("role_profiles", {}).items()
                            if str(role).strip() and str(profile).strip()
                        }
                        if isinstance(value.get("role_profiles"), Mapping)
                        else {},
                        "profiles": profile_rows,
                        "voices": tuple(item["id"] for item in profile_rows),
                    }
                )
        try:
            language = canonical_tts_language(self.conversation.tts_language) or "zh"
        except (AttributeError, TypeError, ValueError):
            language = "zh"
        result["language"] = language
        result["language_protocol"] = protocol_tts_language(language)
        result["queue_size"] = int(getattr(self.tts, "queue_size", 0) or 0)
        if self.tts_startup_health:
            result["health"] = dict(self.tts_startup_health)
        return result

    async def test_tts_module(self) -> Mapping[str, object]:
        """仅执行 TTS health/model-ready 探针，不合成或播放音频。"""

        started = monotonic()
        if self._closed:
            return {
                "status": "unavailable",
                "ready": False,
                "mode": "stopped",
                "latency_ms": 0.0,
            }
        backend = getattr(self.tts, "backend", None)
        diagnostics = self.tts_diagnostics()
        has_active_tasks = getattr(self.tts, "has_active_tasks", None)
        if callable(has_active_tasks) and bool(has_active_tasks()):
            health_snapshot = diagnostics.get("health", {})
            health_snapshot = health_snapshot if isinstance(health_snapshot, Mapping) else {}
            ipc = diagnostics.get("ipc", {})
            return {
                "status": "busy",
                "available": bool(health_snapshot.get("available", False)),
                "ready": False,
                "mode": str(diagnostics.get("mode", "") or "busy")[:32],
                "backend": str(diagnostics.get("backend", "") or "busy")[:128],
                "model_ready": bool(diagnostics.get("model_ready", False)),
                "latency_ms": round((monotonic() - started) * 1000, 3),
                "ipc": dict(ipc) if isinstance(ipc, Mapping) else {},
            }
        start_task = self._tts_start_task
        start_lock = getattr(self.tts, "_start_lock", None)
        startup_pending = start_task is not None and not start_task.done()
        start_locked = bool(callable(getattr(start_lock, "locked", None)) and start_lock.locked())
        if startup_pending or start_locked:
            ipc = diagnostics.get("ipc", {})
            return {
                "status": "loading",
                "available": False,
                "ready": False,
                "mode": str(diagnostics.get("mode", "") or "pending")[:32],
                "backend": str(diagnostics.get("backend", "") or "pending")[:128],
                "model_ready": bool(diagnostics.get("model_ready", False)),
                "latency_ms": round((monotonic() - started) * 1000, 3),
                "ipc": dict(ipc) if isinstance(ipc, Mapping) else {},
            }
        health_method = getattr(backend, "health", None)
        if not callable(health_method):
            return {
                "status": "unavailable",
                "ready": False,
                "mode": "unknown",
                "latency_ms": 0.0,
            }
        try:
            health = await health_method()
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            return {
                "status": "unavailable",
                "ready": False,
                "mode": "unknown",
                "latency_ms": round((monotonic() - started) * 1000, 3),
            }
        mode = str(diagnostics.get("mode", "") or "text_only")[:32]
        model_ready = bool(diagnostics.get("model_ready", False))
        requires_model = mode in {"built_in", "external", "legacy"} and str(
            getattr(health, "engine", "")
        ).startswith("gpt-sovits")
        ready = bool(getattr(health, "available", False)) and (
            model_ready if requires_model else True
        )
        status = "ready" if ready else "configured" if health.available else "unavailable"
        ipc = diagnostics.get("ipc", {})
        return {
            "status": status,
            "available": bool(getattr(health, "available", False)),
            "ready": ready,
            "mode": mode,
            "backend": str(getattr(health, "engine", "unknown") or "unknown")[:128],
            "model_ready": model_ready,
            "latency_ms": round((monotonic() - started) * 1000, 3),
            "ipc": dict(ipc) if isinstance(ipc, Mapping) else {},
        }

    def _set_tts_health(self, health: object) -> None:
        """保存不含地址或请求正文的 TTS 初始化状态。"""

        if not isinstance(health, EngineHealth):
            self.tts_startup_health = {
                "status": "unavailable",
                "available": False,
                "ready": False,
                "engine": "unknown",
                "message": "TTS health result is invalid",
                "pending": False,
            }
            return
        self.tts_startup_health = {
            "status": "ready" if health.available else "unavailable",
            "available": bool(health.available),
            "ready": bool(health.available),
            "engine": str(health.engine or "unknown")[:128],
            "message": str(health.message or "")[:240],
            "latency_ms": health.latency_ms,
            "pending": False,
        }

    async def _initialize_tts(self) -> None:
        """在运行时循环内初始化 TTS，不阻塞 Qt 首窗装配。"""

        try:
            health = await self.tts.start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            health = EngineHealth(
                "tts",
                False,
                f"TTS initialization failed: {type(exc).__name__}",
            )
        self._set_tts_health(health)

    async def wait_tts_initialization(self) -> Mapping[str, object]:
        """等待当前 TTS 启动代次结束并返回脱敏健康快照。"""

        task = self._tts_start_task
        if task is not None and not task.done():
            await asyncio.shield(task)
        return dict(self.tts_startup_health)

    @staticmethod
    async def _wait_cancelled_future(task: asyncio.Future[Any]) -> bool:
        """屏蔽任意次数的等待者取消，直到目标任务真正进入终态。"""

        waiter = asyncio.gather(task, return_exceptions=True)
        interrupted = False
        while not waiter.done():
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                interrupted = True
        waiter.result()
        return interrupted

    async def _cancel_tts_initialization(self) -> None:
        """取消并回收当前 TTS 启动代次，避免热替换被旧健康回执覆盖。"""

        task = self._tts_start_task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            interrupted = await self._wait_cancelled_future(task)
        else:
            interrupted = False
        if self._tts_start_task is task:
            self._tts_start_task = None
        if interrupted:
            raise asyncio.CancelledError

    def asr_diagnostics(self) -> Mapping[str, object]:
        """返回无路径、音频、正文和原始日志的 ASR 状态。"""

        try:
            result = dict(self.asr.diagnostics())
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            result = {
                "status": "unavailable",
                "available": False,
                "ready": False,
                "running": False,
                "backend": "sensevoice",
                "model": "SenseVoiceSmall",
                "model_loaded": False,
                "device": "unknown",
                "language": "zh",
                "ipc": {"status": "unavailable", "running": False, "ready": False, "pid": None},
            }
        if self.asr_startup_health:
            health = dict(self.asr_startup_health)
            if bool(health.get("pending", False)):
                progress_message = str(result.get("message", "") or "").strip()
                if progress_message:
                    health["message"] = progress_message[:240]
            result["health"] = health
            if bool(health.get("pending", False)):
                pending_status = str(health.get("status", "queued") or "queued")[:32]
                result.update(
                    {
                        "status": pending_status,
                        "available": False,
                        "ready": False,
                        "message": str(health.get("message", "") or "")[:240],
                    }
                )
                ipc = result.get("ipc")
                if isinstance(ipc, Mapping):
                    result["ipc"] = {**dict(ipc), "status": pending_status, "ready": False}
        return result

    def _set_asr_health(self, health: object) -> None:
        """保存只含固定状态字段的 ASR 初始化结果。"""

        if not isinstance(health, ASRHealth):
            self.asr_startup_health = {
                "status": "unavailable",
                "available": False,
                "ready": False,
                "message": "ASR health result is invalid",
            }
            return
        self.asr_startup_health = {
            "status": str(health.status or "unavailable")[:32],
            "available": bool(health.available),
            "ready": bool(health.ready),
            "backend": str(health.backend or "sensevoice")[:64],
            "model": str(health.model or "SenseVoiceSmall")[:128],
            "model_loaded": bool(health.model_loaded),
            "message": str(health.message or "")[:240],
        }

    async def _initialize_asr(self) -> None:
        """在运行时循环内加载 ASR，不阻塞 Qt 首窗装配。"""

        try:
            health = await self.asr.start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            health = ASRHealth(
                "unavailable",
                False,
                "sensevoice",
                "SenseVoiceSmall",
                False,
                "unknown",
                "zh",
                f"ASR initialization failed: {type(exc).__name__}",
            )
        self._set_asr_health(health)

    async def _initialize_asr_after_tts(self) -> None:
        """默认等待 TTS 预载终态，再启动 ASR，避免两个本地模型争抢资源。"""

        tts_task = self._tts_start_task
        if tts_task is not None and not tts_task.done():
            try:
                await asyncio.shield(tts_task)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if self._closed or (current is not None and current.cancelling()):
                    raise
                # TTS 热替换会取消旧启动代次；新后端已经由配置事务完成初始化，
                # ASR 仍可继续进入自己的队列阶段。
        self.asr_startup_health = {
            "status": "loading",
            "available": False,
            "ready": False,
            "message": "ASR is initializing",
            "pending": True,
        }
        await self._initialize_asr()

    async def wait_asr_initialization(self) -> Mapping[str, object]:
        """等待当前 ASR 启动代次结束并返回脱敏健康快照。"""

        task = self._asr_start_task
        if task is not None and not task.done():
            await asyncio.shield(task)
        return dict(self.asr_startup_health)

    async def _cancel_asr_initialization(self) -> None:
        task = self._asr_start_task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            interrupted = await self._wait_cancelled_future(task)
        else:
            interrupted = False
        if self._asr_start_task is task:
            self._asr_start_task = None
        if interrupted:
            raise asyncio.CancelledError

    async def test_asr(self) -> Mapping[str, object]:
        """供控制台执行一次不含录音正文的 IPC 健康探测。"""

        if self.asr.has_active_tasks():
            result = dict(self.asr_diagnostics())
            result.update(
                {
                    "status": "busy",
                    "available": bool(result.get("available", False)),
                    "ready": False,
                }
            )
            return result
        startup_task = self._asr_start_task
        if startup_task is not None and not startup_task.done():
            result = dict(self.asr_diagnostics())
            pending_status = str(self.asr_startup_health.get("status", "queued") or "queued")[:32]
            result.update({"status": pending_status, "available": False, "ready": False})
            return result
        health = await self.asr.health(probe=True)
        self._set_asr_health(health)
        result = dict(self.asr_diagnostics())
        result["status"] = "ready" if health.ready else str(health.status)
        return result

    async def transcribe_audio(
        self,
        audio: bytes,
        *,
        audio_format: str = "wav",
        sample_rate: int = 16000,
        channels: int = 1,
        language: str | None = None,
        request_id: str | None = None,
    ) -> object:
        """把调用方持有的内存音频转交 ASR；结果只返回该调用方。"""

        if self._closed or self._closing:
            raise RuntimeError("runtime is closed")
        startup_task = self._asr_start_task
        if startup_task is not None and not startup_task.done():
            # 后台 ASR 包装任务严格等待 TTS 预载。语音按钮在首窗后立即
            # 使用时复用该任务，不能绕过顺序另起一次 SenseVoice 加载。
            await asyncio.shield(startup_task)
        if self._closed or self._closing:
            raise RuntimeError("runtime is closed")
        return await self.asr.transcribe(
            audio,
            audio_format=audio_format,
            sample_rate=sample_rate,
            channels=channels,
            language=language,
            request_id=request_id,
        )

    async def cancel_asr_transcription(self, request_id: str) -> bool:
        return await self.asr.cancel(request_id)

    def select_tts_profile(
        self,
        profile_id: str | None = None,
        *,
        language: str | None = None,
    ) -> Mapping[str, object]:
        """切换后续语音使用的 profile 或语言。

        路由器对正在生成的 stream 保持后端引用不变；选择只影响后续分段，
        因而不会在热切换时截断当前音频。未启用多 profile 时返回明确不可用。
        """

        backend = getattr(self.tts, "backend", None)
        selector = getattr(backend, "select_profile", None)
        if not callable(selector):
            selector = getattr(backend, "select", None)
        normalized_profile = None if profile_id is None else str(profile_id).strip()
        if normalized_profile is not None and (
            not normalized_profile
            or len(normalized_profile) > 128
            or any(char in normalized_profile for char in "\r\n\x00")
        ):
            return {"status": "unavailable", "reason": "语音 profile 不可用"}
        normalized_language = None
        if language is not None:
            normalized_language = canonical_tts_language(language)
            if not normalized_language:
                return {"status": "unavailable", "reason": "输出语言不可用"}
        if not callable(selector):
            if normalized_language is None:
                return {"status": "unavailable", "reason": "当前语音后端不支持 profile 切换"}
        if normalized_profile is None and normalized_language is None:
            active = str(getattr(backend, "active_profile", "") or "")[:128]
            return {
                "status": "available",
                "profile": active,
                "language": canonical_tts_language(self.conversation.tts_language) or "zh",
            }
        try:
            # 路由器的 ``language`` 参数用于更新语言映射；需要同时切换
            # 全局 profile 时先提交显式选择，再写入该语言的映射。
            if normalized_profile is not None and normalized_language is not None:
                selector(normalized_profile)
            selected = (
                selector(normalized_profile, language=normalized_language)
                if callable(selector)
                else ""
            )
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "语音 profile 暂时不可用"}
        if normalized_language is not None:
            self.conversation.tts_language = normalized_language
        return {
            "status": "updated",
            "profile": str(selected or getattr(backend, "active_profile", "") or "")[:128],
            "language": canonical_tts_language(self.conversation.tts_language) or "zh",
        }

    def set_tts_language(self, language: str) -> Mapping[str, object]:
        """设置后续对话语音语言；语言标签经过项目统一规范化。"""

        normalized = canonical_tts_language(language)
        if not normalized:
            return {"status": "unavailable", "reason": "输出语言不可用"}
        self.conversation.tts_language = normalized
        proactive = getattr(self, "proactive", None)
        proactive_conversation = getattr(proactive, "conversation", None)
        if proactive_conversation is not None:
            proactive_conversation.tts_language = normalized
        return {
            "status": "updated",
            "language": normalized,
            "language_protocol": protocol_tts_language(normalized),
        }

    def mcp_content_servers(self) -> tuple[Mapping[str, object], ...]:
        """返回可由控制台显式浏览的 MCP 内容来源摘要。"""

        rows: list[Mapping[str, object]] = []
        for client, bridge in zip(self.mcp_clients, self.mcp_bridges, strict=False):
            content = getattr(bridge, "content", None)
            if content is None:
                continue
            name = str(getattr(client, "server_name", "") or "").strip()
            if not name:
                continue
            status = content.status()
            rows.append(
                {
                    "name": name[:128],
                    "source": str(status.get("source", "") or "")[:80],
                    "status": str(status.get("status", "unavailable") or "unavailable")[:32],
                    "resource_count": int(status.get("resource_count", 0) or 0),
                    "prompt_count": int(status.get("prompt_count", 0) or 0),
                }
            )
        return tuple(rows)

    def _mcp_content_bridge(self, server_name: object) -> object:
        """按用户选择的精确服务器名解析内容桥；多来源时禁止隐式选择。"""

        requested = str(server_name or "").strip()
        matches = tuple(
            (client, bridge)
            for client, bridge in zip(self.mcp_clients, self.mcp_bridges, strict=False)
            if getattr(bridge, "content", None) is not None
            and str(getattr(client, "server_name", "") or "").strip()
        )
        if requested:
            for client, bridge in matches:
                if str(getattr(client, "server_name", "") or "").strip() == requested:
                    return bridge
            raise ValueError("MCP content server is not configured")
        if len(matches) == 1:
            return matches[0][1]
        if not matches:
            raise ValueError("MCP content is unavailable")
        raise ValueError("MCP content server must be selected")

    async def _execute_mcp_content(
        self,
        server_name: object,
        operation: str,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        """通过普通控制台权限执行器调用一个应用控制的 MCP 内容操作。"""

        if self._closed or self._closing:
            return {"status": "unavailable", "reason": "runtime is closing"}
        try:
            bridge = self._mcp_content_bridge(server_name)
            client = getattr(bridge, "client", None)
            starter = getattr(client, "start", None)
            if callable(starter):
                started = starter()
                if inspect.isawaitable(started):
                    await started
            register = getattr(bridge, "register_into", None)
            if not callable(register):
                return {"status": "unavailable", "reason": "MCP content bridge is unavailable"}
            register(self.registry)
            content = getattr(bridge, "content", None)
            selector = getattr(content, "spec_for", None)
            spec = selector(operation) if callable(selector) else None
            if spec is None:
                return {"status": "unavailable", "reason": "MCP content operation is unsupported"}
            context = ToolCallContext(
                "default",
                "local",
                f"mcp-content-{secrets.token_urlsafe(8)}",
                source="console",
                metadata={"mode": "direct", "mcp_content": True},
            )
            call_id = f"mcp-content-call-{secrets.token_urlsafe(8)}"
            outcome = await self.tools.execute(
                call_id=call_id,
                identity=spec.identity,
                arguments=dict(arguments),
                context=context,
            )
            if outcome.status == "approval_required" and outcome.approval is not None:
                approval_id = str(outcome.approval.approval_id or "").strip()
                if approval_id:
                    self._prune_mcp_content_approvals()
                    oldest: str | None = None
                    oldest_context: ToolCallContext | None = None
                    with self._mcp_content_approval_lock:
                        if len(self._mcp_content_approvals) >= 1024:
                            # 审批服务本身也有同样的硬上限；淘汰最早的本地
                            # 映射时同步拒绝底层请求，避免孤儿审批阻塞热重载。
                            oldest = next(iter(self._mcp_content_approvals))
                            oldest_context = self._mcp_content_approvals.pop(oldest, None)
                        self._mcp_content_approvals[approval_id] = context
                    if oldest is not None:
                        try:
                            self.tools.deny_approval(oldest)
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            pass
                        if oldest_context is not None:
                            self.tools.clear_turn(oldest_context, clear_pending=True)
                    return {
                        "status": "approval_required",
                        "approval_id": approval_id,
                        "display_name": outcome.approval.display_name,
                        "safe_summary": outcome.approval.safe_summary,
                        "expires_at": outcome.approval.expires_at,
                    }
            self.tools.clear_turn(context)
            return {"status": outcome.status, **dict(outcome.content)}
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "MCP content operation failed"}

    def _prune_mcp_content_approvals(self) -> None:
        """同步移除已被权限服务过期/消费的 MCP 内容审批映射。"""

        with self._mcp_content_approval_lock:
            entries = tuple(self._mcp_content_approvals.items())
        for approval_id, expected_context in entries:
            try:
                active = self.tools.pending_approval(approval_id) is not None
            except (AttributeError, RuntimeError, TypeError, ValueError):
                active = False
            if not active:
                with self._mcp_content_approval_lock:
                    context = self._mcp_content_approvals.get(approval_id)
                    if context is expected_context:
                        self._mcp_content_approvals.pop(approval_id, None)
                    else:
                        context = None
                if context is not None:
                    self.tools.clear_turn(context, clear_pending=True)

    async def approve_mcp_content(
        self,
        approval_id: str,
        *,
        grant_session: bool = False,
    ) -> Mapping[str, object]:
        """批准一个控制台发起的 MCP 内容读取并返回安全信封。"""

        key = str(approval_id or "").strip()
        with self._mcp_content_approval_lock:
            context = self._mcp_content_approvals.pop(key, None)
        if context is None:
            return {"status": "unavailable", "reason": "MCP 内容审批不存在或已过期"}
        try:
            outcome = await self.tools.approve_and_execute(
                key,
                grant_session=grant_session,
                context=context,
            )
            return {"status": outcome.status, **dict(outcome.content)}
        except (OSError, RuntimeError, TypeError, ValueError):
            return {"status": "unavailable", "reason": "MCP 内容审批执行失败"}
        finally:
            self.tools.clear_turn(context)

    def deny_mcp_content(self, approval_id: str) -> Mapping[str, object]:
        """拒绝一个控制台发起的 MCP 内容读取。"""

        key = str(approval_id or "").strip()
        with self._mcp_content_approval_lock:
            context = self._mcp_content_approvals.pop(key, None)
        if context is None:
            return {"status": "unavailable", "reason": "MCP 内容审批不存在或已过期"}
        denied = self.tools.deny_approval(key)
        self.tools.clear_turn(context, clear_pending=True)
        return {"status": "denied" if denied is not None else "unavailable"}

    async def mcp_list_resources(self, server_name: object = "") -> Mapping[str, object]:
        """由控制台显式列出一个 MCP 来源的资源目录。"""

        return await self._execute_mcp_content(server_name, "list_resources", {})

    async def mcp_read_resource(
        self,
        server_name: object,
        uri: object,
    ) -> Mapping[str, object]:
        """由控制台显式读取已列出的 MCP 资源。"""

        return await self._execute_mcp_content(
            server_name,
            "read_resource",
            {"uri": uri},
        )

    async def mcp_list_prompts(self, server_name: object = "") -> Mapping[str, object]:
        """由控制台显式列出一个 MCP 来源的 prompt 目录。"""

        return await self._execute_mcp_content(server_name, "list_prompts", {})

    async def mcp_get_prompt(
        self,
        server_name: object,
        name: object,
        arguments: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        """由控制台显式获取用户选择的 MCP prompt 数据。"""

        payload: dict[str, object] = {"name": name}
        if arguments is not None:
            payload["arguments"] = dict(arguments)
        return await self._execute_mcp_content(server_name, "get_prompt", payload)

    async def execute_console_tool(
        self, identity: str, arguments: Mapping[str, object]
    ) -> Mapping[str, object]:
        """通过统一权限管线执行控制台发起的只读工具。"""

        context = ToolCallContext(
            "default",
            "local",
            f"console-{secrets.token_urlsafe(8)}",
            source="console",
            metadata={"mode": "direct", "generation_id": 0},
        )
        try:
            outcome = await self.tools.execute(
                call_id=f"console-call-{secrets.token_urlsafe(8)}",
                identity=str(identity),
                arguments=dict(arguments),
                context=context,
            )
            if outcome.status == "approval_required":
                # 控制台只读按钮没有可恢复的模型回合代际；若权限策略把
                # 它升级为审批，不能留下一个 generation=0 的孤儿审批卡，
                # 否则用户点击批准后会被 ConversationService 拒绝。
                self.tools.clear_turn(context, clear_pending=True)
                return {
                    "status": "denied",
                    "reason": "控制台只读操作需要在对话中由桌宠发起确认",
                }
            return {"status": outcome.status, **dict(outcome.content)}
        finally:
            self.tools.clear_turn(context)

    def pending_approvals_for_ui(self) -> tuple[object, ...]:
        """合并普通对话与主动行为的审批快照。"""

        values: dict[str, object] = {}
        for request in self.conversation.pending_approvals_for_ui():
            key = str(getattr(request, "approval_id", "") or "").strip()
            if key:
                values[key] = request
        if self.proactive is not None:
            for request in self.proactive.pending_approvals_for_ui():
                key = str(getattr(request, "approval_id", "") or "").strip()
                if key:
                    values[key] = request
        self._prune_mcp_content_approvals()
        with self._mcp_content_approval_lock:
            mcp_approval_ids = frozenset(self._mcp_content_approvals)
        for request in self.tools.pending_approvals():
            key = str(getattr(request, "approval_id", "") or "").strip()
            if key in mcp_approval_ids:
                values[key] = request
        return tuple(
            sorted(values.values(), key=lambda item: float(getattr(item, "expires_at", 0.0)))
        )

    async def continue_approval(
        self,
        approval_id: str,
        *,
        grant_session: bool = False,
    ) -> object:
        """把审批续接路由到持有对应 paused turn 的对话服务。"""

        proactive = self.proactive
        if proactive is not None and proactive.owns_approval(approval_id):
            return await proactive.continue_approval(
                approval_id,
                grant_session=grant_session,
            )
        with self._mcp_content_approval_lock:
            owns_mcp_content = str(approval_id or "").strip() in self._mcp_content_approvals
        if owns_mcp_content:
            return await self.approve_mcp_content(
                approval_id,
                grant_session=grant_session,
            )
        return await self.conversation.continue_approval(
            approval_id,
            grant_session=grant_session,
        )

    def deny_approval(self, approval_id: str) -> bool:
        proactive = self.proactive
        if proactive is not None and proactive.owns_approval(approval_id):
            return proactive.deny_approval(approval_id)
        with self._mcp_content_approval_lock:
            owns_mcp_content = str(approval_id or "").strip() in self._mcp_content_approvals
        if owns_mcp_content:
            return self.deny_mcp_content(approval_id).get("status") == "denied"
        return self.conversation.deny_approval(approval_id)

    async def notify_proactive_event(
        self,
        event_name: str,
        payload: Mapping[str, object] | None = None,
        *,
        instruction: str | None = None,
    ) -> Mapping[str, object]:
        if self._closed or self._closing:
            return {"status": "closed"}
        if self.proactive is None:
            return {"status": "unavailable"}
        return await self.proactive.notify(event_name, payload, instruction=instruction)

    async def apply_model_configuration(
        self, configuration: LoadedConfiguration, *, _lock_held: bool = False
    ) -> Mapping[str, object]:
        """在运行时线程中安全替换 dialogue 模型路由。

        只更新当前对话所依赖的 ModelRouter、配置快照和交互诊断；调度器、
        TTS、MCP 与渲染资源不随模型保存而重建。检测到活动回合时保持旧路由，
        让调用方明确提示重启后生效。
        """

        if not _lock_held:
            async with self._configuration_reload_lock:
                return await self.apply_model_configuration(configuration, _lock_held=True)
        if self._closed or self._closing:
            return {"status": "unavailable", "reason": "runtime is closed"}
        if not isinstance(configuration, LoadedConfiguration):
            return {"status": "unavailable", "reason": "configuration is invalid"}
        if not isinstance(configuration.values, Mapping):
            return {"status": "unavailable", "reason": "configuration values are invalid"}

        # 该方法由 RuntimeLoop 投递到同一事件循环；活动检查与路由切换之间
        # 不会被另一个对话任务插入，避免热替换半途改变正在使用的请求。
        if self.conversation.has_active_conversation():
            return {
                "status": "restart_required",
                "reason": "active conversation is running",
            }

        try:
            new_router = ModelRouter.from_mapping(configuration.values)
            set_audit_repository = getattr(new_router, "set_audit_repository", None)
            if callable(set_audit_repository):
                set_audit_repository(self.api_audit)
        except Exception as exc:
            # 路由校验异常可能来自用户输入；只返回异常类型，绝不回显 API
            # 密钥、请求头或端点正文。
            return {
                "status": "unavailable",
                "reason": f"model configuration rejected: {type(exc).__name__}",
            }

        try:
            diagnostics = new_router.diagnostics("dialogue")
        except Exception as exc:
            cleanup_failed = await self._wait_reload_cleanups(
                (self._schedule_reload_cleanup(new_router, label="replacement-router"),)
            )
            return {
                "status": "restart_required" if cleanup_failed else "unavailable",
                "reason": f"model configuration rejected: {type(exc).__name__}",
                **({"cleanup": "restart_required"} if cleanup_failed else {}),
            }

        old_router = self.router
        self.router = new_router
        self.conversation.router = new_router
        record_tool_execution = getattr(new_router, "record_tool_execution", None)
        self.tools.set_execution_record_sink(
            record_tool_execution if callable(record_tool_execution) else None
        )
        if self.proactive is not None:
            self.proactive.conversation.tools.set_execution_record_sink(
                record_tool_execution if callable(record_tool_execution) else None
            )
        self.conversation.model = (
            str(_mapping(_mapping(configuration.values).get("llm")).get("default_model", ""))
            or None
        )
        self.configuration = configuration
        cleanup_task = self._schedule_reload_cleanup(old_router, label="previous-router")
        try:
            if self.interaction_state is not None:
                self.interaction_state.set_model_diagnostics(diagnostics)
        except Exception as exc:
            cleanup_failed = await self._wait_reload_cleanups((cleanup_task,))
            return {
                "status": "restart_required",
                "reason": f"model configuration apply failed: {type(exc).__name__}",
                **({"cleanup": "restart_required"} if cleanup_failed else {}),
            }

        cleanup_failed = await self._wait_reload_cleanups((cleanup_task,))

        result: dict[str, object] = {
            "status": "reloaded",
            "task": "dialogue",
            "ready": bool(diagnostics.get("ready", False)),
        }
        if cleanup_failed:
            result["cleanup"] = "restart_required"
        return result

    async def apply_configuration(self, configuration: LoadedConfiguration) -> Mapping[str, object]:
        """追踪完整热重载生命周期，使关闭流程可取消并等待其收尾。"""

        if self._closed or self._closing:
            return {"status": "unavailable", "reason": "runtime is closed"}
        transaction = asyncio.create_task(
            self._apply_configuration_transaction(configuration),
            name="meapet-configuration-reload",
        )
        self._configuration_reload_tasks.add(transaction)
        try:
            try:
                return await asyncio.shield(transaction)
            except asyncio.CancelledError:
                transaction.cancel()
                await self._wait_cancelled_future(transaction)
                raise
        finally:
            self._configuration_reload_tasks.discard(transaction)

    async def _apply_mcp_configuration(
        self,
        configuration: LoadedConfiguration,
    ) -> Mapping[str, object]:
        """两阶段热替换 MCP 传输，并在失败时保留旧桥接。"""

        new_clients, new_bridges = _build_mcp(
            configuration.values,
            configuration_directory=configuration.directory,
        )
        old_clients = tuple(self.mcp_clients)
        old_bridges = tuple(self.mcp_bridges)
        old_status = dict(self.mcp_status)
        withdrawn = False

        async def close_resources(resources: Sequence[object]) -> None:
            """关闭一组尚未移交给运行时的 MCP 资源。"""

            for resource in resources:
                close = getattr(resource, "close", None)
                if not callable(close):
                    continue
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("MCP replacement resource cleanup failed", exc_info=True)

        if self._background_started:
            try:
                staged = all(
                    callable(getattr(bridge, "discover", None))
                    and callable(getattr(bridge, "register_into", None))
                    for bridge in new_bridges
                )
                if staged:
                    # 先启动/握手并发现新工具，但不写入共享 registry；旧桥接在这
                    # 一阶段仍保持有效，通知不会造成半替换状态。
                    for bridge in new_bridges:
                        await bridge.discover()

                    # 所有新连接都可用后才进入短提交窗口：撤下旧归属并原子登记
                    # 新工具。任一注册失败都会恢复旧绑定。
                    for bridge in old_bridges:
                        withdraw = getattr(bridge, "_withdraw_bindings", None)
                        if callable(withdraw):
                            withdraw(keep_registries=True)
                        else:
                            close = getattr(bridge, "close", None)
                            if callable(close):
                                result = close()
                                if inspect.isawaitable(result):
                                    await result
                        withdrawn = True
                    for bridge in new_bridges:
                        bridge.register_into(self.registry)
                else:
                    # 兼容旧宿主/测试替身只提供 refresh_into 的桥接对象；真实
                    # MCPToolBridge 走上面的两阶段路径。
                    for bridge in old_bridges:
                        withdraw = getattr(bridge, "_withdraw_bindings", None)
                        if callable(withdraw):
                            withdraw(keep_registries=True)
                        else:
                            close = getattr(bridge, "close", None)
                            if callable(close):
                                result = close()
                                if inspect.isawaitable(result):
                                    await result
                        withdrawn = True
                    for bridge in new_bridges:
                        refresh = getattr(bridge, "refresh_into", None)
                        if not callable(refresh):
                            raise RuntimeError("MCP bridge does not support refresh")
                        result = refresh(self.registry)
                        if inspect.isawaitable(result):
                            await result
            except asyncio.CancelledError:
                await close_resources((*new_bridges, *new_clients))
                if withdrawn:
                    for bridge in old_bridges:
                        try:
                            await bridge.refresh_into(self.registry)
                        except Exception:
                            logger.debug("MCP old bridge restore failed", exc_info=True)
                raise
            except Exception as exc:
                await close_resources((*new_bridges, *new_clients))
                if withdrawn:
                    for bridge in old_bridges:
                        try:
                            await bridge.refresh_into(self.registry)
                        except Exception:
                            logger.debug("MCP old bridge restore failed", exc_info=True)
                self.mcp_status.clear()
                self.mcp_status.update(old_status)
                raise RuntimeError("MCP configuration reload failed") from exc
        self.mcp_clients = new_clients
        self.mcp_bridges = new_bridges
        self.modules.replace_adopted("mcp", new_clients)
        self._mcp_started = bool(self._background_started)
        self.mcp_status.clear()
        if self._background_started:
            for client in new_clients:
                self.mcp_status[str(getattr(client, "server_name", "mcp"))] = "ready"
        cleanup_tasks: list[asyncio.Task[bool] | None] = []
        for bridge in old_bridges:
            cleanup_tasks.append(self._schedule_reload_cleanup(bridge, label="previous-mcp-bridge"))
        for client in old_clients:
            cleanup_tasks.append(self._schedule_reload_cleanup(client, label="previous-mcp-client"))
        cleanup_failed = await self._wait_reload_cleanups(cleanup_tasks)
        result: dict[str, object] = {
            "status": "reloaded",
            "configured": len(new_clients),
            "ready": len(new_clients) if self._background_started else 0,
        }
        if withdrawn:
            result["replaced"] = len(old_clients)
        if cleanup_failed:
            result["cleanup"] = "restart_required"
        return result

    async def _apply_configuration_transaction(
        self,
        configuration: LoadedConfiguration,
    ) -> Mapping[str, object]:
        """校验并尽可能热替换一份完整 YAML 配置。

        模型路由、TTS 后端、行为参数、前台窗口观察、流式展示、工具权限策略、
        MCP 传输和当前后端支持的渲染资源可在空闲运行时安全刷新。数据库、工具
        注册表结构及渲染后端切换仍返回 ``restart_required``。
        """

        if self._closed or self._closing:
            return {"status": "unavailable", "reason": "runtime is closed"}
        if not isinstance(configuration, LoadedConfiguration):
            return {"status": "unavailable", "reason": "configuration is invalid"}
        if not isinstance(configuration.values, Mapping):
            return {"status": "unavailable", "reason": "configuration values are invalid"}

        async with self._configuration_reload_lock:
            if self._closed or self._closing:
                return {"status": "unavailable", "reason": "runtime is closed"}
            try:
                validate_runtime_configuration(configuration)
            except Exception as exc:
                return {
                    "status": "unavailable",
                    "reason": f"configuration rejected: {type(exc).__name__}",
                }

            old_values = self.configuration.values
            new_values = configuration.values
            changed_sections = tuple(
                section
                for section in (
                    "llm",
                    "tts",
                    "asr",
                    "memory",
                    "behavior",
                    "proactive",
                    "watcher",
                    "config",
                    "scheduler",
                    "ui",
                    "logging",
                    "mcp",
                    "plugins",
                    "tools",
                    "rendering",
                    "storage",
                    "app",
                    "web",
                )
                if old_values.get(section) != new_values.get(section)
            )
            if not changed_sections:
                self.configuration = configuration
                return {"status": "unchanged", "changed_sections": ()}

            asr_service_changed = bool(
                "asr" in changed_sections
                and _asr_service_configuration(old_values.get("asr"))
                != _asr_service_configuration(new_values.get("asr"))
            )

            chat_changed = any(section in changed_sections for section in ("llm", "tts"))
            tts_active = bool(
                chat_changed
                and callable(getattr(self.tts, "has_active_tasks", None))
                and self.tts.has_active_tasks()
            )
            if chat_changed and (self.conversation.has_active_conversation() or tts_active):
                return {
                    "status": "restart_required",
                    "reason": "active conversation or speech is running",
                    "changed_sections": changed_sections,
                    "retry": True,
                }
            if asr_service_changed and self.asr.has_active_tasks():
                return {
                    "status": "restart_required",
                    "reason": "active transcription is running",
                    "changed_sections": changed_sections,
                    "retry": True,
                }

            pending_tts_start = bool(
                self._tts_start_task is not None and not self._tts_start_task.done()
            )
            pending_asr_start = bool(
                self._asr_start_task is not None and not self._asr_start_task.done()
            )
            startup_tasks_paused = False

            async def pause_model_startup_tasks() -> None:
                """先回收旧模型启动代次，禁止热加载期间并行占用模型资源。"""

                nonlocal startup_tasks_paused
                if startup_tasks_paused:
                    return
                startup_tasks_paused = True
                # ASR wrapper 可能正 shield 等待 TTS；必须先取消它，避免取消
                # TTS 后旧 ASR 恰好越过等待点并与新模型同时加载。
                await self._cancel_asr_initialization()
                await self._cancel_tts_initialization()

            def resume_model_startup_tasks(
                *,
                tts_committed: bool = False,
                asr_committed: bool = False,
            ) -> None:
                """失败时恢复旧代次；成功时只恢复未被本次配置替换的服务。"""

                if (
                    not startup_tasks_paused
                    or self._closed
                    or self._closing
                    or not self._background_started
                ):
                    return
                if (
                    pending_tts_start
                    and not tts_committed
                    and (self._tts_start_task is None or self._tts_start_task.done())
                ):
                    self.tts_startup_health = {
                        "status": "loading",
                        "available": False,
                        "engine": "pending",
                        "message": "TTS is initializing",
                        "pending": True,
                    }
                    self._tts_start_task = asyncio.create_task(self._initialize_tts())
                if (
                    pending_asr_start
                    and not asr_committed
                    and (self._asr_start_task is None or self._asr_start_task.done())
                ):
                    self.asr_startup_health = {
                        "status": "queued",
                        "available": False,
                        "ready": False,
                        "message": "ASR is queued after TTS initialization",
                        "pending": True,
                    }
                    self._asr_start_task = asyncio.create_task(self._initialize_asr_after_tts())

            new_tts: TTSCoordinator | None = None
            new_tts_health: EngineHealth | None = None
            new_router_needed = "llm" in changed_sections
            new_router: ModelRouter | None = None
            diagnostics: Mapping[str, object] = self.router.diagnostics("dialogue")
            new_asr: ASRService | None = None
            new_asr_health: ASRHealth | None = None
            tts_backend_committed = False
            router_committed = False
            asr_committed = False
            staged_values = dict(old_values)
            applied: list[str] = []

            def commit_applied_section(section: str) -> None:
                """在区段首次可见副作用前提交其合成配置快照。"""

                if section not in applied:
                    applied.append(section)
                if section in new_values:
                    staged_values[section] = new_values[section]
                else:
                    staged_values.pop(section, None)
                self.configuration = LoadedConfiguration(
                    configuration.path,
                    dict(staged_values),
                    None,
                )

            async def cleanup_uncommitted_replacements() -> bool:
                """把尚未交给运行时的替换对象转移给持久清理任务。"""

                nonlocal new_tts, new_router, new_asr
                tasks: list[asyncio.Task[bool] | None] = []
                if new_tts is not None and not tts_backend_committed:
                    replacement_tts = new_tts
                    new_tts = None
                    tasks.append(
                        self._schedule_reload_cleanup(replacement_tts, label="replacement-tts")
                    )
                if new_router is not None and new_router_needed and not router_committed:
                    replacement_router = new_router
                    new_router = None
                    tasks.append(
                        self._schedule_reload_cleanup(
                            replacement_router,
                            label="replacement-router",
                        )
                    )
                if new_asr is not None and not asr_committed:
                    replacement_asr = new_asr
                    new_asr = None
                    tasks.append(
                        self._schedule_reload_cleanup(replacement_asr, label="replacement-asr")
                    )
                return await self._wait_reload_cleanups(tasks)

            async def await_hot_section(
                operation: Awaitable[Any],
                *,
                on_commit: Callable[[], None],
                commit_when: Callable[[Any], bool] | None = None,
            ) -> Any:
                """让一个已提交区段完成全部生命周期转换后再传播取消。"""

                future = asyncio.ensure_future(operation)
                interrupted = False
                try:
                    result = await asyncio.shield(future)
                except asyncio.CancelledError:
                    interrupted = True
                    await self._wait_cancelled_future(future)
                    if future.cancelled():
                        raise
                    error = future.exception()
                    if error is not None:
                        raise error
                    result = future.result()
                if commit_when is None or commit_when(result):
                    on_commit()
                if interrupted:
                    raise asyncio.CancelledError
                return result

            rendering_result: Mapping[str, object] | None = None
            plugins_result: Mapping[str, object] | None = None
            modules_result: Mapping[str, object] | None = None
            tools_result: Mapping[str, object] | None = None
            mcp_result: Mapping[str, object] | None = None
            cleanup_required = bool(self._reload_cleanup_required)
            old_backend: object | None = None
            previous_asr: object | None = None
            old_router: object | None = None
            unsupported: tuple[str, ...] = ()
            try:
                if "tts" in changed_sections or asr_service_changed:
                    await pause_model_startup_tasks()

                # 替换对象只在当前作用域中准备。任何 await 被取消时，未提交
                # 对象都会转移给运行时持有的清理任务，不依赖调用方继续等待。
                if "tts" in changed_sections:
                    try:
                        new_tts = _build_tts(configuration)
                        new_tts_health = await new_tts.start()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        cleanup_required = (
                            await cleanup_uncommitted_replacements() or cleanup_required
                        )
                        resume_model_startup_tasks()
                        return {
                            "status": "unavailable",
                            "reason": f"tts configuration rejected: {type(exc).__name__}",
                            "changed_sections": changed_sections,
                        }
                    if not new_tts_health.available:
                        cleanup_required = (
                            await cleanup_uncommitted_replacements() or cleanup_required
                        )
                        resume_model_startup_tasks()
                        return {
                            "status": "unavailable",
                            "reason": "tts backend is unavailable",
                            "changed_sections": changed_sections,
                        }

                if new_router_needed:
                    try:
                        new_router = ModelRouter.from_mapping(configuration.values)
                        set_audit_repository = getattr(new_router, "set_audit_repository", None)
                        if callable(set_audit_repository):
                            set_audit_repository(self.api_audit)
                        diagnostics = new_router.diagnostics("dialogue")
                    except Exception as exc:
                        cleanup_required = (
                            await cleanup_uncommitted_replacements() or cleanup_required
                        )
                        resume_model_startup_tasks()
                        return {
                            "status": "unavailable",
                            "reason": f"model configuration rejected: {type(exc).__name__}",
                            "changed_sections": changed_sections,
                        }

                if asr_service_changed:
                    new_asr = _build_asr(configuration)
                    new_asr_health = await new_asr.start()
                    if new_asr.enabled and not new_asr_health.ready:
                        raise RuntimeError("replacement ASR is unavailable")
                elif "asr" in changed_sections:
                    # capture 只由 Qt 主线程宿主持有；运行时提交配置快照，
                    # update_bubble 会按精确签名在 Qt 线程热更新采集器。
                    commit_applied_section("asr")

                if "rendering" in changed_sections:
                    try:
                        rendering_result = await await_hot_section(
                            self._apply_rendering_configuration(configuration),
                            on_commit=lambda: commit_applied_section("rendering"),
                            commit_when=lambda result: (
                                isinstance(result, Mapping)
                                and str(result.get("status", "")).strip().lower()
                                in {"available", "reloaded", "unchanged", "pending", "requested"}
                            ),
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        rendering_result = {
                            "status": "failed",
                            "reason": f"rendering reload failed: {type(exc).__name__}",
                        }
                    rendering_status = str(rendering_result.get("status", "")).strip().lower()
                    if rendering_status not in {
                        "available",
                        "reloaded",
                        "unchanged",
                        "pending",
                        "requested",
                    }:
                        cleanup_required = (
                            await cleanup_uncommitted_replacements() or cleanup_required
                        )
                        resume_model_startup_tasks()
                        return {
                            "status": "restart_required",
                            "reason": str(
                                rendering_result.get(
                                    "reason",
                                    "rendering resources require a new window lifecycle",
                                )
                            )[:240],
                            "changed_sections": changed_sections,
                            "restart_sections": ("rendering",),
                            "rendering_status": rendering_status or "failed",
                            "current_backend": str(
                                rendering_result.get("current_backend", "unknown")
                            )[:64],
                            "requested_backend": str(
                                rendering_result.get("requested_backend", "unknown")
                            )[:64],
                            "rendering_model": str(
                                _mapping(new_values.get("rendering")).get("model", "") or ""
                            )[:512],
                            "retry": rendering_status == "busy",
                        }
                if "app" in changed_sections:
                    await await_hot_section(
                        self._apply_persona_configuration(new_values),
                        on_commit=lambda: commit_applied_section("app"),
                    )

                if "memory" in changed_sections:
                    await await_hot_section(
                        self._apply_memory_configuration(new_values),
                        on_commit=lambda: commit_applied_section("memory"),
                    )

                if "behavior" in changed_sections:
                    await await_hot_section(
                        self._apply_behavior_configuration(new_values),
                        on_commit=lambda: commit_applied_section("behavior"),
                    )

                if "proactive" in changed_sections:
                    await await_hot_section(
                        self._apply_proactive_configuration(new_values),
                        on_commit=lambda: commit_applied_section("proactive"),
                    )

                if "watcher" in changed_sections:
                    await await_hot_section(
                        self._apply_window_watcher_configuration(new_values),
                        on_commit=lambda: commit_applied_section("watcher"),
                    )

                if "config" in changed_sections:
                    await await_hot_section(
                        self._apply_configuration_watcher_configuration(new_values),
                        on_commit=lambda: commit_applied_section("config"),
                    )

                if "ui" in changed_sections:
                    await await_hot_section(
                        self._apply_ui_stream_configuration(new_values),
                        on_commit=lambda: commit_applied_section("ui"),
                    )

                if "logging" in changed_sections:
                    # 日志处理器自身带全局互斥锁，允许在运行时线程安全地
                    # 替换等级、颜色和轮换策略；文件路径仍按新 YAML 目录解析。
                    configure_logging(
                        config=new_values,
                        base_directory=configuration.directory,
                    )
                    commit_applied_section("logging")

                if "scheduler" in changed_sections:
                    await await_hot_section(
                        self._apply_scheduler_configuration(new_values),
                        on_commit=lambda: commit_applied_section("scheduler"),
                    )

                if "plugins" in changed_sections:
                    # 插件管理器自行负责停止观察器、刷新目录和替换插件代次；
                    # 只有其配置调用返回后才提交新的运行时快照。
                    plugins_result = await await_hot_section(
                        self.plugins.reconfigure(
                            _mapping(new_values.get("plugins")),
                            base_directory=configuration.directory,
                        ),
                        on_commit=lambda: commit_applied_section("plugins"),
                    )

                if "tools" in changed_sections:
                    # 权限、模型可见工具组和命令白名单都只替换内存策略；
                    # 未支持的自定义字段不应被误报为已热加载。
                    tools_result = await await_hot_section(
                        self._apply_tools_configuration(new_values),
                        on_commit=lambda: commit_applied_section("tools"),
                        commit_when=lambda result: (
                            isinstance(result, Mapping)
                            and str(result.get("status", "")).strip().lower() == "reloaded"
                        ),
                    )

                if "mcp" in changed_sections:
                    # MCP 传输会先在旧桥接器撤下后握手；失败时恢复旧桥接器，
                    # 只有成功返回才把新配置提交到运行时快照。
                    mcp_result = await await_hot_section(
                        self._apply_mcp_configuration(configuration),
                        on_commit=lambda: commit_applied_section("mcp"),
                    )

                # 所有非聊天区段完成后才提交模型/TTS。这样某个 UI、调度或
                # watcher setter 失败时，旧聊天路由仍可继续服务；提交后的
                # 交换段不会再调用异步配置 setter。
                if self._closed or self._closing:
                    raise RuntimeError("runtime is closing")

                if new_tts is not None:
                    # 保留原协调器及 Qt audio sink，只交换已构造并校验的
                    # 后端/特征分析器；正在消费的旧 stream 持有旧后端引用，
                    # 不会被新配置中途改写。
                    old_backend = await self.tts.replace_backend(
                        cast(SpeechBackend, new_tts.backend),
                        initialized=bool(new_tts_health and new_tts_health.available),
                    )
                    # replace_backend 返回即表示 self.tts 已接管新后端。在
                    # 后续任何 await 之前登记提交，取消路径不得关闭它。
                    tts_backend_committed = True
                    commit_applied_section("tts")

                if new_asr is not None:
                    previous_asr = self.asr
                    self.asr = new_asr
                    self.modules.replace_adopted("asr", new_asr)
                    # 赋值完成即转移所有权；此后取消只清理 previous_asr。
                    asr_committed = True
                    commit_applied_section("asr")

                if new_router_needed and new_router is not None:
                    old_router = self.router
                    self.router = new_router
                    self.modules.replace_adopted("model", new_router)
                    self.conversation.router = new_router
                    if self.proactive is not None:
                        self.proactive.conversation.router = new_router
                    record_tool_execution = getattr(new_router, "record_tool_execution", None)
                    self.tools.set_execution_record_sink(
                        record_tool_execution if callable(record_tool_execution) else None
                    )
                    if self.proactive is not None:
                        self.proactive.conversation.tools.set_execution_record_sink(
                            record_tool_execution if callable(record_tool_execution) else None
                        )
                    router_committed = True
                    commit_applied_section("llm")

                if new_tts is not None:
                    if new_tts_health is not None:
                        self._set_tts_health(new_tts_health)
                    set_segmentation = getattr(self.tts, "set_segmentation", None)
                    if callable(set_segmentation):
                        set_segmentation(
                            max_chars=int(getattr(new_tts, "segment_max_chars", 120)),
                            hard_boundaries=getattr(new_tts, "segment_hard_boundaries", None),
                            soft_boundaries=getattr(new_tts, "segment_soft_boundaries", None),
                        )
                    set_queue_size = getattr(self.tts, "set_queue_size", None)
                    if callable(set_queue_size):
                        set_queue_size(getattr(new_tts, "queue_size", 16))
                    self.tts.set_feature_analyzer(getattr(new_tts, "_feature_analyzer", None))
                    self.conversation.tts = self.tts
                    self.conversation.tts_language = str(
                        _mapping(new_values.get("tts")).get("language", "zh")
                    )
                    self.conversation.tts_role = str(
                        _mapping(new_values.get("tts")).get("role", "")
                        or _mapping(_mapping(new_values.get("tts")).get("routing")).get(
                            "default_role", ""
                        )
                        or ""
                    )
                    if self.proactive is not None:
                        self.proactive.conversation.tts_language = self.conversation.tts_language
                        self.proactive.conversation.tts_role = self.conversation.tts_role
                    clear_context_routes = getattr(self.tts, "clear_context_routes", None)
                    if callable(clear_context_routes):
                        # 已确认当前没有活动语音任务；清除旧回合的语言/profile
                        # 快照，避免后端热切换后新回合继承上一后端的路由。
                        clear_context_routes()

                if new_asr is not None:
                    if new_asr_health is not None:
                        self._set_asr_health(new_asr_health)

                if new_router_needed and new_router is not None:
                    self.conversation.model = (
                        str(_mapping(new_values.get("llm")).get("default_model", "")) or None
                    )
                    if self.interaction_state is not None:
                        self.interaction_state.set_model_diagnostics(diagnostics)
                    if self.memory_summarizer is not None:
                        self.memory_summarizer.wake()

                unsupported = tuple(
                    section for section in changed_sections if section not in applied
                )
                if not unsupported:
                    # 全部区段均已提交时保留加载器给出的完整对象与 digest；
                    # 否则必须保留 staged snapshot，让未应用区段下次仍被检测。
                    self.configuration = configuration
                # 配置快照提交后通知统一模块中心。此前这里只更新了快照，
                # 导致实现 ``reload_configuration``/``reload`` 的模块在站点
                # 配置热更时完全收不到通知；模块管理器自身负责隔离失败并
                # 在需要时替换实例，因此不能把它并入某个具体区段的 setter。
                modules_result = await await_hot_section(
                    self.modules.reconfigure(self.configuration.values),
                    on_commit=lambda: None,
                )
                self._register_optional_modules()

                cleanup_tasks = (
                    self._schedule_reload_cleanup(old_backend, label="previous-tts-backend"),
                    self._schedule_reload_cleanup(previous_asr, label="previous-asr"),
                    self._schedule_reload_cleanup(old_router, label="previous-router"),
                )
                cleanup_required = (
                    await self._wait_reload_cleanups(cleanup_tasks) or cleanup_required
                )
            except asyncio.CancelledError:
                try:
                    await cleanup_uncommitted_replacements()
                finally:
                    resume_model_startup_tasks(
                        tts_committed=tts_backend_committed,
                        asr_committed=asr_committed,
                    )
                raise
            except Exception as exc:
                committed_cleanup_tasks = (
                    self._schedule_reload_cleanup(
                        old_backend if tts_backend_committed else None,
                        label="previous-tts-backend",
                    ),
                    self._schedule_reload_cleanup(
                        previous_asr if asr_committed else None,
                        label="previous-asr",
                    ),
                    self._schedule_reload_cleanup(
                        old_router if router_committed else None,
                        label="previous-router",
                    ),
                )
                try:
                    cleanup_required = await cleanup_uncommitted_replacements() or cleanup_required
                    cleanup_required = (
                        await self._wait_reload_cleanups(committed_cleanup_tasks)
                        or cleanup_required
                    )
                    # 所有权已转移时不能再回滚到可能已关闭的旧对象；已提交
                    # 区段已由 commit_applied_section 精确写入合成快照。
                    committed = tts_backend_committed or asr_committed or router_committed
                    return {
                        "status": "restart_required"
                        if committed or cleanup_required
                        else "unavailable",
                        "reason": f"configuration apply failed: {type(exc).__name__}",
                        "changed_sections": changed_sections,
                        "applied_sections": tuple(applied),
                        **({"cleanup": "restart_required"} if cleanup_required else {}),
                    }
                finally:
                    resume_model_startup_tasks(
                        tts_committed=tts_backend_committed,
                        asr_committed=asr_committed,
                    )

            resume_model_startup_tasks(
                tts_committed=tts_backend_committed,
                asr_committed=asr_committed,
            )
            if mcp_result is not None and mcp_result.get("cleanup") == "restart_required":
                cleanup_required = True
            result_status = "restart_required" if unsupported or cleanup_required else "reloaded"
            result: dict[str, object] = {
                "status": result_status,
                "changed_sections": changed_sections,
                "applied_sections": tuple(applied),
            }
            if rendering_result is not None:
                result["rendering_status"] = str(rendering_result.get("status", ""))[:32]
                result["rendering_model"] = str(
                    _mapping(new_values.get("rendering")).get("model", "") or ""
                )[:512]
            if plugins_result is not None:
                result["plugins_status"] = str(plugins_result.get("status", ""))[:32]
            if modules_result is not None:
                result["modules_status"] = str(modules_result.get("status", ""))[:32]
                result["modules_reloaded"] = tuple(
                    str(item)[:128] for item in modules_result.get("reloaded", ())
                )
                result["modules_failed"] = tuple(
                    str(item)[:128] for item in modules_result.get("failed", ())
                )
            if tools_result is not None:
                result["tools_status"] = str(tools_result.get("status", ""))[:32]
                if tools_result.get("unsupported_fields"):
                    result["tools_unsupported_fields"] = tuple(
                        str(item)[:128] for item in tools_result["unsupported_fields"]
                    )
            if mcp_result is not None:
                result["mcp_status"] = str(mcp_result.get("status", ""))[:32]
                result["mcp_configured"] = int(mcp_result.get("configured", 0) or 0)
            if unsupported:
                result["restart_sections"] = unsupported
            if cleanup_required:
                result["cleanup"] = "restart_required"
            return result

    async def reload_configuration(
        self, configuration: LoadedConfiguration
    ) -> Mapping[str, object]:
        """``apply_configuration`` 的语义别名，供文件观察器回调使用。"""

        return await self.apply_configuration(configuration)

    async def _apply_persona_configuration(self, values: Mapping[str, Any]) -> None:
        """热替换人设及各模型任务提示词。"""

        bundle = PersonaPromptBundle.from_configuration(values)
        dialogue_prompt = bundle.dialogue_system_prompt()
        behavior_values = _mapping(values.get("behavior"))
        behavior_enabled = parse_bool(
            behavior_values.get("enabled", False),
            field_name="behavior.enabled",
            default=False,
        )
        running = bool(self.behavior.status().get("running", False))
        self.persona_prompts = bundle
        self.conversation.system_prompt = dialogue_prompt
        if self.proactive is not None:
            self.proactive.conversation.system_prompt = (
                dialogue_prompt
                + "\n\n【主动行为】你由后台事件唤醒；只依据脱敏事件和记忆决定是否回应。"
            )
        if self.memory_summarizer is not None:
            self.memory_summarizer.configure_prompts(
                summary=bundle.prompts.memory_summary,
                extraction=bundle.prompts.memory_extract,
            )
        if not bundle.persona.proactive_enabled and running:
            await self.behavior.stop()
        elif (
            bundle.persona.proactive_enabled
            and behavior_enabled
            and self._background_started
            and not running
        ):
            await self.behavior.start()

    async def _apply_behavior_configuration(self, values: Mapping[str, Any]) -> None:
        behavior_values = _mapping(values.get("behavior"))
        movement_values = _mapping(behavior_values.get("movement"))
        actions: list[BehaviorAction] = []
        for raw_action in _sequence(behavior_values.get("actions")):
            item = _mapping(raw_action)
            identity = str(item.get("identity", "")).strip()
            arguments = item.get("arguments", {})
            if identity and isinstance(arguments, Mapping):
                actions.append(
                    BehaviorAction(
                        identity,
                        dict(arguments),
                        float(item.get("weight", 1.0)),
                        float(item.get("duration_seconds", 0.8)),
                    )
                )
        movement = AutonomousMovementConfig(
            enabled=parse_bool(
                movement_values.get("enabled", False),
                field_name="behavior.movement.enabled",
                default=False,
            ),
            min_distance_px=float(movement_values.get("min_distance_px", 80.0)),
            max_distance_px=float(movement_values.get("max_distance_px", 280.0)),
            step_pixels=float(movement_values.get("step_pixels", 36.0)),
            step_interval_seconds=float(movement_values.get("step_interval_seconds", 0.08)),
            max_speed_px_per_second=float(movement_values.get("max_speed_px_per_second", 240.0)),
            moving_motion=str(movement_values.get("moving_motion", "walk") or ""),
            idle_motion=str(movement_values.get("idle_motion", "idle") or ""),
            movement_identity=str(
                movement_values.get("movement_identity", "pet:autonomous_move")
                or "pet:autonomous_move"
            ),
            flip_facing=parse_bool(
                movement_values.get("flip_facing", True),
                field_name="behavior.movement.flip_facing",
                default=True,
            ),
            max_steps=int(movement_values.get("max_steps", 64)),
        )
        enabled = parse_bool(
            behavior_values.get("enabled", False),
            field_name="behavior.enabled",
            default=False,
        ) and bool(self.persona_prompts.persona.proactive_enabled)
        running = bool(self.behavior.status().get("running", False))
        self.behavior.reconfigure(
            actions=tuple(actions),
            interval_seconds=(
                float(behavior_values.get("min_interval_seconds", 8.0)),
                float(behavior_values.get("max_interval_seconds", 20.0)),
            ),
            probability=float(behavior_values.get("probability", 0.35)),
            interaction_cooldown_seconds=float(
                behavior_values.get("interaction_cooldown_seconds", 0.35)
            ),
            movement=movement,
        )
        if enabled and self._background_started and not running:
            await self.behavior.start()
        elif not enabled and running:
            await self.behavior.stop()

    async def _apply_memory_configuration(self, values: Mapping[str, Any]) -> None:
        """热替换记忆策略，不重建数据库或丢弃已有内容。"""

        memory_values = _mapping(values.get("memory"))
        settings = MemorySettings.from_mapping(
            memory_values,
            base_directory=self.configuration.directory,
        )
        self.memory.configure(settings)
        if self.memory_summarizer is not None:
            self.memory_summarizer.wake()
        # ConversationService 只持有服务引用；这里不需要重建会话，
        # 新回合会直接读取新的 enabled/召回/上下文预算策略。

    async def _apply_proactive_configuration(self, values: Mapping[str, Any]) -> None:
        """热替换主动行为规则、预算和冷却策略。"""

        if self.proactive is None:
            raise RuntimeError("proactive coordinator is unavailable")
        settings = ProactiveSettings.from_mapping(_mapping(values.get("proactive")))
        self.proactive.reconfigure(settings)
        await self.proactive.wait_idle()
        self.proactive.conversation.max_tool_rounds = settings.max_tool_rounds

    async def _apply_tools_configuration(self, values: Mapping[str, Any]) -> Mapping[str, object]:
        """热替换权限、工具组和命令白名单，不重建工具执行器。

        ``tools.groups`` 是启动时的描述性配置，当前注册表中的工具身份和
        分组由各工具/插件声明；修改它不能安全地伪造为已应用，因此继续
        返回需要重启的状态。其余运行期策略在同一同步边界内完成替换。
        """

        old_tools = _mapping(self.configuration.values.get("tools"))
        new_tools = _mapping(values.get("tools"))
        changed_fields = tuple(
            key
            for key in tuple(dict.fromkeys((*old_tools.keys(), *new_tools.keys())))
            if old_tools.get(key) != new_tools.get(key)
        )
        supported_fields = {"active_groups", "permissions", "command_allowlist"}
        unsupported_fields = tuple(
            str(key) for key in changed_fields if key not in supported_fields
        )
        if unsupported_fields:
            return {
                "status": "restart_required",
                "unsupported_fields": unsupported_fields,
            }

        permission_values = _mapping(new_tools.get("permissions"))
        old_permission_values = _mapping(old_tools.get("permissions"))
        permission_changed_fields = tuple(
            key
            for key in tuple(
                dict.fromkeys((*old_permission_values.keys(), *permission_values.keys()))
            )
            if old_permission_values.get(key) != permission_values.get(key)
        )
        supported_permission_fields = {
            "allow",
            "deny",
            "bypass_approval",
            "auto_allow_low_risk",
            "approval_ttl_seconds",
        }
        unsupported_permission_fields = tuple(
            str(key) for key in permission_changed_fields if key not in supported_permission_fields
        )
        if unsupported_permission_fields:
            return {
                "status": "restart_required",
                "unsupported_fields": tuple(
                    f"permissions.{key}" for key in unsupported_permission_fields
                ),
            }
        allow = tuple(str(item) for item in _sequence(permission_values.get("allow")))
        deny = tuple(str(item) for item in _sequence(permission_values.get("deny")))
        bypass_approval = parse_bool(
            permission_values.get("bypass_approval", False),
            field_name="tools.permissions.bypass_approval",
            default=False,
        )
        auto_allow_low_risk = parse_bool(
            permission_values.get("auto_allow_low_risk", True),
            field_name="tools.permissions.auto_allow_low_risk",
            default=True,
        )
        approval_ttl_seconds = float(permission_values.get("approval_ttl_seconds", 90.0))

        command_allowlist: tuple[str, ...] = ()
        normalized_command_allowlist: frozenset[str] | None = None
        command_setter: Callable[[object], object] | None = None
        if "command_allowlist" in changed_fields:
            command_allowlist = tuple(
                str(item) for item in _sequence(new_tools.get("command_allowlist"))
            )
            normalized_command_allowlist = normalize_command_allowlist(command_allowlist)
            command_spec = self.registry.get("system:run_command")
            raw_setter = getattr(
                getattr(command_spec, "handler", None), "set_command_allowlist", None
            )
            if not callable(raw_setter):
                return {
                    "status": "restart_required",
                    "unsupported_fields": ("command_allowlist",),
                }
            command_setter = raw_setter

        if "permissions" in changed_fields:
            self.permissions.reconfigure(
                allow=allow,
                deny=deny,
                bypass_approval=bypass_approval,
                auto_allow_low_risk=auto_allow_low_risk,
                approval_ttl_seconds=approval_ttl_seconds,
            )

        if command_setter is not None and normalized_command_allowlist is not None:
            command_setter(normalized_command_allowlist)

        if "active_groups" in changed_fields:
            configured_groups = (
                None
                if "active_groups" not in new_tools or new_tools.get("active_groups") is None
                else tuple(str(item) for item in _sequence(new_tools.get("active_groups")))
            )
            self.conversation.tool_groups = configured_groups
            if self.proactive is not None:
                self.proactive.conversation.tool_groups = configured_groups

        if "permissions" in changed_fields and self.proactive is not None:
            proactive_tools = getattr(self.proactive.conversation, "tools", None)
            proactive_permissions = getattr(proactive_tools, "permissions", None)
            reconfigure = getattr(proactive_permissions, "reconfigure", None)
            if callable(reconfigure):
                # 后台主动回合始终禁止继承普通对话的 bypass_approval。
                reconfigure(
                    allow=allow,
                    deny=deny,
                    bypass_approval=False,
                    auto_allow_low_risk=auto_allow_low_risk,
                    approval_ttl_seconds=approval_ttl_seconds,
                )

        return {"status": "reloaded"}

    async def _apply_rendering_configuration(
        self,
        configuration: LoadedConfiguration,
    ) -> Mapping[str, object]:
        """把资源路径、模型选择和 Web Live2D 性能设置提交给当前宿主。"""

        old_values = _mapping(_mapping(self.configuration.values).get("rendering"))
        new_values = _mapping(_mapping(configuration.values).get("rendering"))
        old_backend = normalize_renderer_backend(old_values.get("backend", "auto"))
        new_backend = normalize_renderer_backend(new_values.get("backend", "auto"))
        if old_backend != new_backend:
            # 配置偏好从 auto 变为显式键时，若当前宿主已经运行在新键对应的
            # 实际后端，保留同一窗口和渲染上下文即可；只有实际后端不同
            # （尤其 OpenGL/Vulkan 这类 Qt 全局图形 API）才需要安全重启。
            # 未绑定宿主或诊断不完整时继续 fail-closed。
            actual_backend = ""
            try:
                renderer_diagnostics = self.pet_controller.renderer_diagnostics()
                if inspect.isawaitable(renderer_diagnostics):
                    renderer_diagnostics = await renderer_diagnostics
                if isinstance(renderer_diagnostics, Mapping):
                    raw_actual_backend = renderer_diagnostics.get("backend")
                    if isinstance(raw_actual_backend, str):
                        try:
                            actual_backend = normalize_renderer_backend(raw_actual_backend)
                        except (TypeError, ValueError):
                            actual_backend = ""
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                actual_backend = ""
            if actual_backend != new_backend:
                return {
                    "status": "restart_required",
                    "reason": "rendering backend change requires a new window lifecycle",
                    "current_backend": actual_backend or "unknown",
                    "requested_backend": new_backend,
                }

        resource_changed = any(
            old_values.get(name) != new_values.get(name)
            for name in ("resource_root", "model", "sprite_scale")
        )
        result: dict[str, object] = {"status": "unchanged"}
        if resource_changed:
            resource_root = resolve_resource_path(
                new_values.get("resource_root", "./resources"),
                configuration_directory=configuration.directory,
            )
            raw_model = new_values.get("model")
            model_key = str(raw_model or "").strip()
            model_path = None if not model_key or model_key == "auto" else resource_root / model_key
            raw_result = self.pet_controller.reload_resources(
                resource_root,
                model_path=model_path,
                sprite_scale=float(new_values.get("sprite_scale", 0.6)),
            )
            if inspect.isawaitable(raw_result):
                raw_result = await raw_result
            if not isinstance(raw_result, Mapping):
                return {
                    "status": "failed",
                    "reason": "renderer returned an invalid reload result",
                }
            result = dict(raw_result)
            status = str(result.get("status", "") or "").strip().lower()
            if status in {"available", "reloaded", "unchanged", "pending", "requested"}:
                self.inventory = inspect_resources(resource_root)

        performance_result = self.pet_controller.set_rendering_performance(
            frame_rate=float(new_values.get("frame_rate", 60.0)),
            geometry_audit_hz=float(new_values.get("geometry_audit_hz", 30.0)),
        )
        if inspect.isawaitable(performance_result):
            performance_result = await performance_result
        if isinstance(performance_result, Mapping):
            result["performance"] = dict(performance_result)
            # 性能 setter 不可用时保留资源重载的结果；精灵/OpenGL 等
            # 后端仍可安全保存配置，不把“当前后端不支持热调节”误报成
            # 需要重启的资源失败。
        return result

    async def _apply_window_watcher_configuration(self, values: Mapping[str, Any]) -> None:
        watcher_values = _mapping(values.get("watcher"))
        poll_seconds = float(watcher_values.get("interval_seconds", 0.5))
        emit_initial = parse_bool(
            watcher_values.get("emit_initial", False),
            field_name="watcher.emit_initial",
        )
        poll_timeout_seconds = float(watcher_values.get("poll_timeout_seconds", 5.0))
        enabled = parse_bool(
            watcher_values.get("enabled", False), field_name="watcher.enabled", default=False
        )
        self.watcher.reconfigure(
            poll_seconds=poll_seconds,
            emit_initial=emit_initial,
            poll_timeout_seconds=poll_timeout_seconds,
        )
        if enabled and self._background_started and not self.watcher.running:
            await self.watcher.start()
        elif not enabled and self.watcher.running:
            await self.watcher.stop()

    async def _apply_configuration_watcher_configuration(self, values: Mapping[str, Any]) -> None:
        watcher = self.configuration_watcher
        if watcher is None:
            return
        reload_values = _mapping(_mapping(values.get("config")).get("reload"))
        interval_seconds = float(reload_values.get("interval_seconds", 0.5))
        debounce_seconds = float(reload_values.get("debounce_seconds", 0.1))
        stable_checks = int(reload_values.get("stable_checks", 2))
        enabled = parse_bool(
            reload_values.get("enabled", True),
            field_name="config.reload.enabled",
            default=True,
        )
        watcher.reconfigure(
            interval_seconds=interval_seconds,
            debounce_seconds=debounce_seconds,
            stable_checks=stable_checks,
        )
        if enabled and self._background_started and not watcher.running:
            await watcher.start()
        elif not enabled and watcher.running:
            await watcher.stop()

    async def _apply_ui_stream_configuration(self, values: Mapping[str, Any]) -> None:
        stream_values = _mapping(_mapping(values.get("ui")).get("stream"))
        show_reasoning = parse_bool(
            stream_values.get("show_reasoning", False),
            field_name="ui.stream.show_reasoning",
        )
        show_murmur = parse_bool(
            stream_values.get("show_murmur", True), field_name="ui.stream.show_murmur"
        )
        show_tool_status = parse_bool(
            stream_values.get("show_tool_status", True),
            field_name="ui.stream.show_tool_status",
        )
        max_length = int(stream_values.get("max_bubble_chars", 4000))
        max_length = max(64, min(max_length, 100_000))
        ui_values = _mapping(values.get("ui"))
        always_on_top = parse_bool(
            ui_values.get("always_on_top", True),
            field_name="ui.always_on_top",
            default=True,
        )
        window_locked = parse_bool(
            ui_values.get("window_locked", False),
            field_name="ui.window_locked",
            default=False,
        )
        self.conversation.presentation.show_reasoning = show_reasoning
        self.conversation.presentation.show_murmur = show_murmur
        self.conversation.presentation.show_tool_status = show_tool_status
        self.conversation.presentation.max_text_length = max_length
        if self.interaction_state is not None:
            self.interaction_state.max_text_length = max_length
            self.interaction_state.show_reasoning = show_reasoning
        try:
            topmost_result = self.pet_controller.set_always_on_top(always_on_top)
            if inspect.isawaitable(topmost_result):
                await topmost_result
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            # 配置文件可能在 GUI 宿主创建前被观察到；流式展示策略仍可
            # 热替换，窗口旗标留给宿主下一次初始化/显式操作。
            pass
        try:
            locked_result = self.pet_controller.set_window_locked(window_locked)
            if inspect.isawaitable(locked_result):
                await locked_result
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            pass

    async def _apply_scheduler_configuration(self, values: Mapping[str, Any]) -> None:
        """替换 owner=config 的任务/触发器，保留模型或用户创建的条目。"""

        scheduler_values = _mapping(values.get("scheduler"))
        activity_values = _mapping(scheduler_values.get("activity"))
        raw_tasks = tuple(_sequence(scheduler_values.get("tasks")))
        raw_triggers = tuple(_sequence(scheduler_values.get("triggers")))
        # 先用无副作用的临时服务完整校验领域约束，并检查显式 ID 是否
        # 抢占了非 config 所有者条目；实际服务在此之后才会开始变更。
        probe_scheduler = SchedulerService()
        probe_triggers = TriggerService()
        for raw_task in raw_tasks:
            item = _mapping(raw_task)
            probe_scheduler.upsert(
                task_id=str(item.get("task_id", "")).strip() or None,
                name=str(item["name"]),
                expression=str(item["expression"]),
                action=dict(item["action"]),
                owner=str(item.get("owner", "config")),
                metadata=dict(item.get("metadata", {}))
                if isinstance(item.get("metadata", {}), Mapping)
                else item.get("metadata"),
            )
        for raw_trigger in raw_triggers:
            item = _mapping(raw_trigger)
            probe_triggers.register(
                trigger_id=str(item["trigger_id"]),
                event_name=str(item["event_name"]),
                action=dict(item["action"]),
                owner=str(item.get("owner", "config")),
                debounce_seconds=float(item.get("debounce_seconds", 0.0)),
                metadata=_trigger_metadata(item),
            )
        existing_tasks = getattr(self.scheduler, "_tasks", {})
        for raw_task in raw_tasks:
            item = _mapping(raw_task)
            explicit_id = str(item.get("task_id", "")).strip()
            existing = existing_tasks.get(explicit_id) if explicit_id else None
            if existing is not None and getattr(existing, "owner", "") != "config":
                raise PermissionError("scheduler task owner mismatch")
        existing_triggers = getattr(self.triggers, "_triggers", {})
        for raw_trigger in raw_triggers:
            item = _mapping(raw_trigger)
            explicit_id = str(item.get("trigger_id", "")).strip()
            existing = existing_triggers.get(explicit_id)
            if existing is not None and getattr(existing, "owner", "") != "config":
                raise PermissionError("trigger owner mismatch")

        configured_task_ids: set[str] = set()
        for raw_task in raw_tasks:
            item = _mapping(raw_task)
            task_id = str(item.get("task_id", "")).strip() or None
            result = self.scheduler.upsert(
                task_id=task_id,
                name=str(item["name"]),
                expression=str(item["expression"]),
                action=dict(item["action"]),
                owner=str(item.get("owner", "config")),
                metadata=dict(item.get("metadata", {}))
                if isinstance(item.get("metadata", {}), Mapping)
                else item.get("metadata"),
            )
            configured_task_ids.add(str(result.get("task_id", task_id or "")))
        for task_id, task in tuple(existing_tasks.items()):
            if getattr(task, "owner", "") == "config" and task_id not in configured_task_ids:
                self.scheduler.remove(task_id, owner="config")

        configured_trigger_ids: set[str] = set()
        for raw_trigger in raw_triggers:
            item = _mapping(raw_trigger)
            trigger_id = str(item["trigger_id"])
            self.triggers.register(
                trigger_id=trigger_id,
                event_name=str(item["event_name"]),
                action=dict(item["action"]),
                owner=str(item.get("owner", "config")),
                debounce_seconds=float(item.get("debounce_seconds", 0.0)),
                metadata=_trigger_metadata(item),
            )
            configured_trigger_ids.add(trigger_id)
        for trigger_id, trigger in tuple(existing_triggers.items()):
            if (
                getattr(trigger, "owner", "") == "config"
                and trigger_id not in configured_trigger_ids
            ):
                self.triggers.remove(trigger_id, owner="config")
        enabled = parse_bool(
            scheduler_values.get("enabled", True),
            field_name="scheduler.enabled",
            default=True,
        )
        running = bool(self.scheduler.status().get("running", False))
        if enabled and self._background_started and not running:
            await self.scheduler.start(
                poll_seconds=float(scheduler_values.get("poll_seconds", 0.5))
            )
        elif not enabled and running:
            await self.scheduler.stop()
        activity = self.activity
        if activity is not None:
            current_activity_enabled = bool(getattr(activity, "enabled", True))
            current_activity_idle = float(getattr(activity, "idle_seconds", 300.0))
            current_activity_poll = float(getattr(activity, "poll_seconds", 0.5))
            current_system_idle_provider = str(
                getattr(activity, "system_idle_provider", "disabled")
            )
            current_system_idle_threshold = float(
                getattr(activity, "system_idle_threshold_seconds", 300.0)
            )
            activity_enabled = parse_bool(
                activity_values.get("enabled", current_activity_enabled),
                field_name="scheduler.activity.enabled",
                default=current_activity_enabled,
            )
            system_idle_provider = activity_values.get(
                "system_idle_provider", current_system_idle_provider
            )
            if not isinstance(system_idle_provider, str):
                raise ConfigurationError("scheduler.activity.system_idle_provider must be a string")
            system_idle_provider = system_idle_provider.strip()
            platform_backend = str(getattr(self.platform, "backend", "") or "").strip().lower()
            provider_backend = {
                "x11": "x11",
                SYSTEM_IDLE_PROVIDER_WINDOWS: "windows",
            }.get(system_idle_provider)
            system_idle_probe = None
            if provider_backend == platform_backend:
                platform_probe = getattr(self.platform, "system_idle_seconds", None)
                if callable(platform_probe):
                    system_idle_probe = platform_probe
            set_system_idle_probe = getattr(activity, "set_system_idle_probe", None)
            if callable(set_system_idle_probe):
                set_system_idle_probe(system_idle_probe)
            activity.reconfigure(
                enabled=activity_enabled,
                idle_seconds=float(activity_values.get("idle_seconds", current_activity_idle)),
                poll_seconds=float(activity_values.get("poll_seconds", current_activity_poll)),
                system_idle_provider=system_idle_provider,
                system_idle_threshold_seconds=float(
                    activity_values.get(
                        "system_idle_threshold_seconds", current_system_idle_threshold
                    )
                ),
            )
            activity_running = activity.running
            if activity_enabled and self._background_started and not activity_running:
                await activity.start()
            elif not activity_enabled and activity_running:
                await activity.stop()

    async def cancel_conversation(self, context: object) -> None:
        """在运行时线程中取消回合并清理对应的交互快照。"""

        if not isinstance(context, ConversationContext):
            return
        self.conversation.cancel(context)
        state = self.interaction_state
        turn_id = str(getattr(context, "turn_id", "") or "").strip()
        should_reset_presentation = state is None
        if state is not None:
            snapshot_turn_id = str(getattr(state.snapshot, "turn_id", "") or "").strip()
            # 新回合已经开始时不清空新回合的状态；否则停止旧回合会覆盖
            # 新回合的首个流式事件。
            if not snapshot_turn_id or not turn_id or snapshot_turn_id == turn_id:
                state.reset(
                    context=context,
                    preserve_pending=False,
                    finish_reason="cancelled",
                )
                should_reset_presentation = True
        if should_reset_presentation:
            self.conversation.presentation.reset(
                context=context,
                cancelled=True,
            )

    async def _rollback_background_start(
        self,
        *,
        startup_task: asyncio.Task[object] | None,
        startup_emitted_before: bool,
        tts_start_task: asyncio.Task[object] | None,
        asr_start_task: asyncio.Task[object] | None,
        mcp_started_before: bool,
        plugins_running_before: bool,
        mcp_closed_resource_ids: set[int],
    ) -> None:

        async def stop_safely(label: str, operation: Callable[[], object]) -> None:
            try:
                result = operation()
                if inspect.isawaitable(result):
                    await result
            except BaseException as exc:
                # 回滚不能遮蔽原始启动异常；日志仅记录固定步骤与异常类型，
                # 不泄漏外部服务正文、地址或对话内容。
                logger.warning("后台启动回滚步骤失败：%s (%s)", label, type(exc).__name__)

        # 先阻止启动触发器和模型预载继续写入状态，再按启动的反方向停止可重启服务。
        # 插件可能注册工具或持有 MCP/模型引用，因此必须在这些依赖仍然存在时
        # 先停止；若启动前插件管理器已经在运行，则交还原有生命周期所有权。
        await stop_safely("modules", self.modules.stop)
        if not plugins_running_before:
            await stop_safely("plugins", self.plugins.stop)
        if not mcp_started_before and self._mcp_started:
            for bridge in self.mcp_bridges:
                if id(bridge) in mcp_closed_resource_ids:
                    continue
                await stop_safely("mcp_bridge", getattr(bridge, "close", lambda: None))
                mcp_closed_resource_ids.add(id(bridge))
            for client in self.mcp_clients:
                if id(client) in mcp_closed_resource_ids:
                    continue
                await stop_safely("mcp_client", getattr(client, "close", lambda: None))
                mcp_closed_resource_ids.add(id(client))
            self._mcp_started = False
            # 本次启动事务创建的 MCP 已被回收，不能让模块中心继续显示旧的 ready/unavailable 状态。
            self.mcp_status.clear()
        if startup_task is not None and self._startup_task is startup_task:
            if not startup_task.done():
                startup_task.cancel()
            await self._wait_cancelled_future(startup_task)
            if self._startup_task is startup_task:
                self._startup_task = None
        if not startup_emitted_before:
            self._startup_emitted = False
        if asr_start_task is not None and self._asr_start_task is asr_start_task:
            await stop_safely("asr_start", self._cancel_asr_initialization)
        if tts_start_task is not None and self._tts_start_task is tts_start_task:
            await stop_safely("tts_start", self._cancel_tts_initialization)
        reload_watcher = self.configuration_watcher
        if reload_watcher is not None:
            await stop_safely("configuration_watcher", reload_watcher.stop)
        await stop_safely("behavior", self.behavior.stop)
        await stop_safely("watcher", self.watcher.stop)
        await stop_safely("scheduler", self.scheduler.stop)
        activity = self.activity
        if activity is not None:
            await stop_safely("activity", activity.stop)
        summarizer = self.memory_summarizer
        if summarizer is not None:
            await stop_safely("memory_summary", summarizer.pause)

    async def start_background(self) -> None:
        """按配置启动定时任务和随机行为循环。"""

        # RuntimeLoop 可能在 Qt 重入或测试宿主中收到重复启动请求；启动
        # 事务一旦建立所有权，后续调用必须复用当前代次，不能再创建模型任务。
        if self._closed or self._closing or self._background_started:
            return
        values = self.configuration.values
        startup_emitted_before = self._startup_emitted
        mcp_started_before = self._mcp_started
        plugins_running_before = bool(self.plugins.status().get("running", False))
        created_startup_task: asyncio.Task[object] | None = None
        created_tts_start_task: asyncio.Task[object] | None = None
        created_asr_start_task: asyncio.Task[object] | None = None
        mcp_closed_resource_ids: set[int] = set()
        self.tts_startup_health = {
            "status": "loading",
            "available": False,
            "engine": "pending",
            "message": "TTS is initializing",
            "pending": True,
        }
        if self._tts_start_task is None or self._tts_start_task.done():
            self._tts_start_task = asyncio.create_task(self._initialize_tts())
            created_tts_start_task = self._tts_start_task
        self.asr_startup_health = {
            "status": "queued",
            "available": False,
            "ready": False,
            "message": "ASR is queued after TTS initialization",
            "pending": True,
        }
        if self._asr_start_task is None or self._asr_start_task.done():
            self._asr_start_task = asyncio.create_task(self._initialize_asr_after_tts())
            created_asr_start_task = self._asr_start_task
        self._background_started = True

        async def rollback_start() -> bool:
            rollback_task = asyncio.create_task(
                self._rollback_background_start(
                    startup_task=created_startup_task,
                    startup_emitted_before=startup_emitted_before,
                    tts_start_task=created_tts_start_task,
                    asr_start_task=created_asr_start_task,
                    mcp_started_before=mcp_started_before,
                    plugins_running_before=plugins_running_before,
                    mcp_closed_resource_ids=mcp_closed_resource_ids,
                ),
                name="meapet-background-start-rollback",
            )
            return await self._wait_cancelled_future(rollback_task)

        async def close_mcp_after_refresh_failure(
            client: object,
            bridge: object,
            server_name: str,
        ) -> None:
            """清理失败的单个 MCP，同时保留其它服务的启动。"""

            for resource in (bridge, client):
                close = getattr(resource, "close", None)
                if not callable(close):
                    continue
                for attempt in range(2):
                    try:
                        result = close()
                        if inspect.isawaitable(result):
                            await result
                    except asyncio.CancelledError:
                        # 取消异常可能发生在 close() 已提交资源终态之后；记录本次
                        # 尝试，避免回滚再次调用非幂等 close()。
                        mcp_closed_resource_ids.add(id(resource))
                        raise
                    except Exception as exc:
                        logger.warning(
                            "MCP 启动清理失败：%s (attempt=%d, %s)",
                            server_name,
                            attempt + 1,
                            type(exc).__name__,
                        )
                        if attempt == 0:
                            continue
                        # 两次关闭都未获得成功回执，交给外层启动事务回滚；
                        # 该资源未加入已关闭集合，回滚会再做一次有界补偿。
                        raise
                    else:
                        mcp_closed_resource_ids.add(id(resource))
                        break

        try:
            # 统一模块表只接管已构造对象的状态，不重复调用它们的启动钩子。
            await self.modules.start()
            if self.memory_summarizer is not None:
                await self.memory_summarizer.start()
            # 插件在 MCP 和其它后台服务之前启动，以便插件能够在组合根提供的
            # registry 上注册扩展工具；单个插件失败由管理器隔离，不阻断桌宠。
            if not plugins_running_before:
                await self.plugins.start()
            if not self._mcp_started:
                self._mcp_started = True
                for client, bridge in zip(self.mcp_clients, self.mcp_bridges, strict=True):
                    name = str(getattr(client, "server_name", "mcp"))
                    try:
                        await bridge.refresh_into(self.registry)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # MCP 不可用时桌宠仍可启动；工具不会进入本地注册表。
                        self.mcp_status[name] = "unavailable"
                        await close_mcp_after_refresh_failure(client, bridge, name)
                    else:
                        self.mcp_status[name] = "ready"
            scheduler_values = values.get("scheduler", {})
            activity_values = _mapping(_mapping(scheduler_values).get("activity"))
            activity = self.activity
            if activity is not None and parse_bool(
                activity_values.get("enabled", True),
                field_name="scheduler.activity.enabled",
                default=True,
            ):
                await activity.start()
            if isinstance(scheduler_values, Mapping) and parse_bool(
                scheduler_values.get("enabled", True), field_name="scheduler.enabled"
            ):
                await self.scheduler.start(
                    poll_seconds=float(scheduler_values.get("poll_seconds", 0.5))
                )
            watcher_values = self.configuration.values.get("watcher", {})
            if isinstance(watcher_values, Mapping) and parse_bool(
                watcher_values.get("enabled", False), field_name="watcher.enabled"
            ):
                await self.watcher.start()
            behavior_values = values.get("behavior", {})
            if (
                isinstance(behavior_values, Mapping)
                and parse_bool(behavior_values.get("enabled", False), field_name="behavior.enabled")
                and self.persona_prompts.persona.proactive_enabled
            ):
                await self.behavior.start()
            reload_watcher = self.configuration_watcher
            if reload_watcher is not None:
                reload_values = _mapping(_mapping(values.get("config")).get("reload"))
                if parse_bool(
                    reload_values.get("enabled", True),
                    field_name="config.reload.enabled",
                    default=True,
                ):
                    await reload_watcher.start()
            if not self._startup_emitted:
                self._startup_emitted = True
                # 不能在 RuntimeLoop.start 的 run_until_complete 阶段等待 Qt 调用；
                # 任务会在事件循环进入 run_forever 后执行，此时 QApplication 已可处理信号。
                created_startup_task = asyncio.create_task(
                    self.triggers.emit("startup", {"status": "started"})
                )
                self._startup_task = created_startup_task
        except asyncio.CancelledError:
            # 启动中断后必须先等待回滚任务真正停止所有可重启服务；否则下次
            # start 会与遗留轮询、配置观察器或行为任务并发运行。
            await rollback_start()
            self._background_started = False
            raise
        except Exception:
            # 非取消失败同样不能留下已经启动的后台服务。若清理等待期间又收到
            # 取消，以取消为最终结果，但仍要等待清理完成再释放启动所有权。
            interrupted = await rollback_start()
            self._background_started = False
            if interrupted:
                raise asyncio.CancelledError
            raise

    async def close(self) -> None:
        """共享实际关闭事务；取消单个等待者不会中断资源回收。"""

        if self._closed:
            return
        close_task = self._close_task
        if close_task is None or close_task.done():
            if close_task is not None and not close_task.cancelled():
                close_task.exception()
            close_task = asyncio.create_task(
                self._close_once(),
                name="meapet-application-runtime-close",
            )
            self._close_task = close_task
        await asyncio.shield(close_task)

    async def _close_once(self) -> None:
        if self._closed:
            return
        # 先封闭新的热重载提交与启动代次恢复；具体服务仍由下方按顺序关闭。
        self._closing = True
        self._background_started = False
        errors: list[BaseException] = []

        async def close_async(step: Callable[[], Awaitable[None]]) -> None:
            try:
                await step()
            except asyncio.CancelledError:
                # 关闭时主动取消后台任务属于正常收尾，不应阻断后续资源释放。
                return
            except BaseException as exc:
                errors.append(exc)

        startup_task = self._startup_task
        if startup_task is not None and not startup_task.done():
            startup_task.cancel()
        if startup_task is not None:
            try:
                await startup_task
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                errors.append(exc)
        self._startup_task = None
        try:
            await self._cancel_configuration_reloads()
        except BaseException as exc:
            errors.append(exc)
        # queued ASR 正在 shield 等待 TTS；必须先取消 ASR wrapper，
        # 再取消 TTS 启动代次，避免 TTS 的取消终态误触发一次 ASR 加载。
        try:
            await self._cancel_asr_initialization()
        except BaseException as exc:
            errors.append(exc)
        try:
            await self._cancel_tts_initialization()
        except BaseException as exc:
            errors.append(exc)
        while self._reload_cleanup_tasks:
            try:
                await self._drain_reload_cleanups()
            except asyncio.CancelledError:
                # close 的等待者被取消时，已 shield 的旧实例回收仍继续；
                # 本轮关闭必须重新接回等待，不能让事件循环结束时遗留任务。
                continue
            except BaseException as exc:
                errors.append(exc)
                break
        if self.configuration_watcher is not None:
            await close_async(self.configuration_watcher.stop)
        await close_async(self.modules.close)
        # 插件依赖组合根中的工具、调度器、模型、语音与平台对象；先关闭插件，
        # 让其 stop/unload 钩子仍能安全访问这些服务，再拆除下游资源。
        await close_async(self.plugins.close)
        if self.memory_summarizer is not None:
            await close_async(self.memory_summarizer.stop)
        if self.proactive is not None:
            await close_async(self.proactive.close)
        await close_async(self.behavior.stop)
        await close_async(self.watcher.stop)
        await close_async(self.scheduler.stop)
        if self.activity is not None:
            await close_async(self.activity.stop)
        with self._mcp_content_approval_lock:
            pending_mcp_approval_ids = tuple(self._mcp_content_approvals)
        for approval_id in pending_mcp_approval_ids:
            try:
                self.deny_mcp_content(approval_id)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                with self._mcp_content_approval_lock:
                    self._mcp_content_approvals.pop(approval_id, None)
        await close_async(self.conversation.aclose)
        await close_async(self.tts.aclose)
        await close_async(self.asr.aclose)
        await close_async(self.router.aclose)
        # 桥接器拥有通知去抖任务与内容目录锁；不能假设每个传输客户端在
        # close() 时都会触发回调，因此先显式回收桥接器，再关闭客户端。
        for bridge in self.mcp_bridges:
            close = getattr(bridge, "close", None)
            if callable(close):
                await close_async(close)
        for client in self.mcp_clients:
            close = getattr(client, "close", None)
            if callable(close):
                await close_async(close)
        try:
            # 语义模型运行在独立 worker；先停止它再关闭共享 SQLite，避免后台
            # ANN 线程在连接销毁后继续读取 journal。
            await close_async(lambda: asyncio.to_thread(self.memory.close_semantic))
            self.database.close()
        except BaseException as exc:
            errors.append(exc)
        self._closed = True
        self._closing = False
        if errors:
            raise errors[0]


def _resolve_path(value: object, configuration: LoadedConfiguration, default: str) -> Path:
    raw = value if isinstance(value, str) and value.strip() else default
    return resolve_resource_path(raw, configuration_directory=configuration.directory)


def _resolve_executable_path(value: object, configuration: LoadedConfiguration) -> Path:
    """解析解释器路径但保留虚拟环境入口符号链接。"""

    raw = str(value or "").strip()
    if not raw:
        raise ConfigurationError("tts.python_executable is required")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = configuration.directory / path
    return Path(os.path.abspath(path))


def _tts_reference_audios(
    value: object,
    configuration: LoadedConfiguration,
) -> dict[str, dict[str, str]]:
    """解析按语言配置的参考音频路径，保留语言和提示文本字段。"""

    try:
        references = normalize_reference_audios(value)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    resolved: dict[str, dict[str, str]] = {}
    for language, entry in references.items():
        item = dict(entry)
        if item.get("path"):
            item["path"] = str(_resolve_path(item["path"], configuration, item["path"]))
        resolved[language] = item
    return resolved


def _tts_worker_options(
    values: Mapping[str, Any], configuration: LoadedConfiguration
) -> dict[str, object]:
    """构造本地 worker 的明确模型参数，不把 YAML 任意字段注入进程。"""

    raw_options = values.get("options", {})
    if not isinstance(raw_options, Mapping):
        raise ConfigurationError("tts.options must be a mapping")
    options: dict[str, object] = {str(key): item for key, item in raw_options.items()}

    for option_name, directory_name, model_name in (
        ("gpt_path", "gpt_weights_dir", "gpt_model"),
        ("sovits_path", "sovits_weights_dir", "sovits_model"),
    ):
        explicit = str(values.get(option_name, "") or "").strip()
        directory = str(values.get(directory_name, "") or "").strip()
        model = str(values.get(model_name, "") or "").strip()
        if explicit:
            options[option_name] = str(_resolve_path(explicit, configuration, explicit))
        elif directory and model:
            combined = str(Path(directory) / model)
            options[option_name] = str(_resolve_path(combined, configuration, combined))
        if model:
            options[model_name] = model
    for option_name in (
        "top_k",
        "top_p",
        "temperature",
        "speed",
        "sample_steps",
        "repetition_penalty",
        "text_split_method",
        "device",
        "is_half",
    ):
        if option_name in values:
            options[option_name] = values[option_name]
    return options


def _tts_http_options(values: Mapping[str, Any]) -> dict[str, object]:
    """收集 GPT-SoVITS api_v2 推理字段；后端会再次按协议白名单过滤。"""

    raw_options = values.get("options", {})
    if not isinstance(raw_options, Mapping):
        raise ConfigurationError("tts.options must be a mapping")
    options: dict[str, object] = {str(key): item for key, item in raw_options.items()}
    for option_name in (
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
    ):
        if option_name in values:
            options[option_name] = values[option_name]
    return options


def _build_tts(configuration: LoadedConfiguration) -> TTSCoordinator:
    values = _mapping(configuration.values.get("tts"))
    if "profiles" in values and parse_bool(
        values.get("enabled", False), field_name="tts.enabled", default=False
    ):
        return _build_tts_profiles(configuration, values)
    backend_name = str(values.get("backend", "text_only")).strip().lower()
    if not parse_bool(values.get("enabled", False), field_name="tts.enabled") or backend_name in {
        "",
        "text_only",
        "disabled",
    }:
        backend: object = TextOnlyBackend()
    elif backend_name in {"gpt_sovits_stdio", "gpt-sovits-stdio"}:
        reference_path = str(values.get("ref_audio_path", "") or "").strip()
        ref_dir_value = str(values.get("ref_dir", "") or "").strip()
        engine_root_value = str(values.get("engine_root", "") or "").strip()
        python_executable_value = str(values.get("python_executable", "") or "").strip()
        expected_sample_rate_value = values.get("expected_sample_rate", 32000)
        backend = GptSovitsStdioBackend(
            str(values.get("endpoint", "")),
            engine_root=(
                str(_resolve_path(engine_root_value, configuration, engine_root_value))
                if engine_root_value
                else ""
            ),
            engine_config=str(
                values.get("engine_config", "GPT_SoVITS/configs/tts_infer.yaml")
                or "GPT_SoVITS/configs/tts_infer.yaml"
            ),
            python_executable=(
                str(_resolve_executable_path(python_executable_value, configuration))
                if python_executable_value
                else ""
            ),
            timeout_seconds=float(values.get("timeout_seconds", 60.0)),
            startup_timeout_seconds=float(values.get("startup_timeout_seconds", 300.0)),
            shutdown_timeout_seconds=float(values.get("shutdown_timeout_seconds", 2.0)),
            ref_audio_path=(
                str(_resolve_path(reference_path, configuration, reference_path))
                if reference_path
                else ""
            ),
            prompt_text=str(values.get("prompt_text", "")),
            prompt_lang=str(values.get("prompt_lang", "") or ""),
            ref_dir=(
                str(_resolve_path(ref_dir_value, configuration, ref_dir_value))
                if ref_dir_value
                else ""
            ),
            reference_audios=_tts_reference_audios(values.get("reference_audios"), configuration),
            default_options=_tts_worker_options(values, configuration),
            media_type=str(values.get("media_type", "wav")),
            streaming_mode=int(values.get("streaming_mode", 3)),
            speed_factor=float(values.get("speed_factor", 1.0)),
            verify_tls=parse_bool(values.get("verify_tls", True), field_name="tts.verify_tls"),
            raw_sample_rate=int(values.get("raw_sample_rate", 24000)),
            raw_channels=int(values.get("raw_channels", 1)),
            expected_sample_rate=(
                int(expected_sample_rate_value) if expected_sample_rate_value is not None else None
            ),
            max_total_bytes=int(values.get("max_total_bytes", 64 * 1024 * 1024)),
            health_probe=parse_bool(
                values.get("health_probe", True), field_name="tts.health_probe"
            ),
            startup_language=str(values.get("language", "zh") or "zh"),
            require_final=parse_bool(
                values.get("require_final", False), field_name="tts.require_final"
            ),
            require_audio=parse_bool(
                values.get("require_audio", True), field_name="tts.require_audio"
            ),
            validate_pcm=parse_bool(
                values.get("validate_pcm", False), field_name="tts.validate_pcm"
            ),
        )
    elif backend_name in {"gpt_sovits", "gpt-sovits", "gpt_sovits_http"}:
        reference_path = str(values.get("ref_audio_path", "") or "").strip()
        ref_dir_value = str(values.get("ref_dir", "") or "").strip()
        expected_sample_rate_value = values.get("expected_sample_rate", 32000)
        backend = GptSovitsHttpBackend(
            str(values.get("endpoint", "")),
            timeout_seconds=float(values.get("timeout_seconds", 60.0)),
            ref_audio_path=(
                str(_resolve_path(reference_path, configuration, reference_path))
                if reference_path
                else ""
            ),
            prompt_text=str(values.get("prompt_text", "")),
            prompt_lang=str(values.get("prompt_lang", "") or ""),
            ref_dir=(
                str(_resolve_path(ref_dir_value, configuration, ref_dir_value))
                if ref_dir_value
                else ""
            ),
            reference_audios=_tts_reference_audios(values.get("reference_audios"), configuration),
            options=_tts_http_options(values),
            media_type=str(values.get("media_type", "wav")),
            streaming_mode=int(values.get("streaming_mode", 3)),
            speed_factor=float(values.get("speed_factor", 1.0)),
            verify_tls=parse_bool(values.get("verify_tls", True), field_name="tts.verify_tls"),
            raw_sample_rate=int(values.get("raw_sample_rate", 24000)),
            raw_channels=int(values.get("raw_channels", 1)),
            expected_sample_rate=(
                int(expected_sample_rate_value) if expected_sample_rate_value is not None else None
            ),
            max_total_bytes=int(values.get("max_total_bytes", 64 * 1024 * 1024)),
            health_probe=parse_bool(
                values.get("health_probe", False), field_name="tts.health_probe"
            ),
        )
    elif backend_name == "subprocess":
        command = tuple(str(item) for item in _sequence(values.get("command")))
        cwd_value = str(values.get("cwd", "") or "").strip()
        ref_dir_value = str(values.get("ref_dir", "") or "").strip()
        reference_path = str(values.get("ref_audio_path", "") or "").strip()
        backend = SubprocessTTSBackend(
            command,
            cwd=(str(_resolve_path(cwd_value, configuration, cwd_value)) if cwd_value else None),
            max_total_bytes=int(values.get("max_total_bytes", 64 * 1024 * 1024)),
            timeout_seconds=float(values.get("timeout_seconds", 60.0)),
            startup_timeout_seconds=float(values.get("startup_timeout_seconds", 300.0)),
            shutdown_timeout_seconds=float(values.get("shutdown_timeout_seconds", 2.0)),
            require_ready=parse_bool(
                values.get("startup_handshake", True),
                field_name="tts.startup_handshake",
            ),
            ref_dir=(
                str(_resolve_path(ref_dir_value, configuration, ref_dir_value))
                if ref_dir_value
                else ""
            ),
            reference_audios=_tts_reference_audios(values.get("reference_audios"), configuration),
            ref_audio_path=(
                str(_resolve_path(reference_path, configuration, reference_path))
                if reference_path
                else ""
            ),
            prompt_text=str(values.get("prompt_text", "")),
            prompt_lang=str(values.get("prompt_lang", "") or ""),
            default_options=_tts_worker_options(values, configuration),
            require_final=parse_bool(
                values.get("require_final", False), field_name="tts.require_final"
            ),
            require_audio=parse_bool(
                values.get("require_audio", True), field_name="tts.require_audio"
            ),
            validate_pcm=parse_bool(
                values.get("validate_pcm", False), field_name="tts.validate_pcm"
            ),
        )
    else:
        raise ConfigurationError(f"unsupported tts backend: {backend_name}")
    feature_values = _mapping(values.get("audio_features"))
    feature_enabled = parse_bool(
        feature_values.get("enabled"),
        field_name="tts.audio_features.enabled",
        default=True,
    )
    feature_analyzer = None
    if feature_enabled:
        try:
            feature_analyzer = PcmAudioFeatureAnalyzer(
                silence_threshold=float(feature_values.get("silence_threshold", 0.02)),
                open_threshold=float(feature_values.get("open_threshold", 0.35)),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError("tts.audio_features thresholds are invalid") from exc
    segmentation_values = _mapping(values.get("segmentation"))
    return TTSCoordinator(
        backend,
        queue_size=int(values.get("queue_size", 16)),
        segment_max_chars=int(segmentation_values.get("max_chars", 120)),
        segment_hard_boundaries=(
            str(segmentation_values["hard_boundaries"])
            if "hard_boundaries" in segmentation_values
            else None
        ),
        segment_soft_boundaries=(
            str(segmentation_values["soft_boundaries"])
            if "soft_boundaries" in segmentation_values
            else None
        ),
        feature_analyzer=feature_analyzer,
    )


def asr_capture_configuration(values: Mapping[str, Any]) -> dict[str, object]:
    """校验并返回 Qt 麦克风使用的私有配置；设备 ID 不进入公开诊断。"""

    raw_capture = values.get("capture", {})
    if raw_capture is None:
        raw_capture = {}
    if not isinstance(raw_capture, Mapping):
        raise ConfigurationError("asr.capture must be a mapping")
    enabled = parse_bool(
        raw_capture.get("enabled", True),
        field_name="asr.capture.enabled",
        default=True,
    )
    auto_submit = parse_bool(
        raw_capture.get("auto_submit", False),
        field_name="asr.capture.auto_submit",
        default=False,
    )
    if auto_submit:
        raise ConfigurationError("asr.capture.auto_submit must remain false")
    duration_value = raw_capture.get("max_duration_seconds", 15.0)
    if isinstance(duration_value, bool):
        raise ConfigurationError("asr.capture.max_duration_seconds is invalid")
    try:
        duration = float(duration_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("asr.capture.max_duration_seconds is invalid") from exc
    if not math.isfinite(duration) or not 0.1 <= duration <= 300.0:
        raise ConfigurationError("asr.capture.max_duration_seconds is outside the allowed range")
    device_value = raw_capture.get("device_id", "")
    if not isinstance(device_value, str):
        raise ConfigurationError("asr.capture.device_id must be a string")
    if len(device_value) > 1_024 or any(char in device_value for char in "\x00\r\n"):
        raise ConfigurationError("asr.capture.device_id is invalid")
    return {
        "enabled": enabled,
        "max_duration_seconds": duration,
        "auto_submit": False,
        "device_id": device_value,
    }


def _asr_service_configuration(values: object) -> object:
    """移除仅归 Qt 宿主持有的 capture 字段，用于判断是否需重载 ASR 模型。"""

    if not isinstance(values, Mapping):
        return values
    return {str(key): value for key, value in values.items() if str(key) != "capture"}


def _build_asr(configuration: LoadedConfiguration) -> ASRService:
    """按严格 YAML 字段创建隔离的 SenseVoice IPC 服务。"""

    values = _mapping(configuration.values.get("asr"))
    asr_capture_configuration(values)
    enabled = parse_bool(values.get("enabled", False), field_name="asr.enabled", default=False)
    backend = str(values.get("backend", "sensevoice") or "").strip().lower()
    if backend not in SUPPORTED_ASR_BACKENDS:
        raise ConfigurationError(f"unsupported asr backend: {backend}")
    language = str(values.get("language", "zh") or "").strip().lower()
    if language not in SUPPORTED_ASR_LANGUAGES:
        raise ConfigurationError("asr.language is invalid")
    device = str(values.get("device", "cpu") or "").strip()
    if not device or len(device) > 64 or any(char in device for char in "\x00\r\n"):
        raise ConfigurationError("asr.device is invalid")

    def timeout_field(name: str, default: float) -> float:
        raw = values.get(name, default)
        if isinstance(raw, bool):
            raise ConfigurationError(f"asr.{name} is invalid")
        try:
            result = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"asr.{name} is invalid") from exc
        if not math.isfinite(result) or not 0.1 <= result <= 600.0:
            raise ConfigurationError(f"asr.{name} is outside the allowed range")
        return result

    timeout_seconds = timeout_field("timeout_seconds", 60.0)
    startup_timeout_seconds = timeout_field("startup_timeout_seconds", 180.0)
    max_audio_value = values.get("max_audio_bytes", 16 * 1024 * 1024)
    if isinstance(max_audio_value, bool):
        raise ConfigurationError("asr.max_audio_bytes is invalid")
    try:
        max_audio_bytes = int(max_audio_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("asr.max_audio_bytes is invalid") from exc
    if max_audio_bytes != max_audio_value or not 1024 <= max_audio_bytes <= 64 * 1024 * 1024:
        raise ConfigurationError("asr.max_audio_bytes is outside the allowed range")

    python_value = str(values.get("python_executable", "") or "").strip()
    model_value = str(values.get("model_path", "") or "").strip()
    # ASR 与桌宠共用当前 Python 环境时允许省略解释器路径。示例配置
    # 只声明模型目录，启动阶段应自动复用项目虚拟环境；显式路径仍按
    # 原有规则严格解析。优先探测配置目录及项目根目录下的标准 venv
    # 入口，最后回退到当前解释器，避免 GUI smoke 因缺少可选字段直接失败。
    if enabled and not python_value:
        is_windows = sys.platform.startswith("win") or os.name == "nt"
        executable_name = "python.exe" if is_windows else "python"
        search_roots = (
            configuration.directory / ".venv",
            configuration.directory.parent / ".venv",
            Path.cwd() / ".venv",
        )
        for root in search_roots:
            candidate = root / ("Scripts" if is_windows else "bin") / executable_name
            if candidate.is_file() and (is_windows or os.access(candidate, os.X_OK)):
                python_value = str(candidate)
                break
        if not python_value:
            python_value = sys.executable
    if enabled and not model_value:
        raise ConfigurationError("asr.model_path is required")
    if python_value:
        python_path = Path(python_value).expanduser()
        if not python_path.is_absolute():
            python_path = configuration.directory / python_path
        python_path = Path(os.path.abspath(python_path))
    else:
        python_path = Path(sys.executable)
    if model_value:
        model_path = _resolve_path(model_value, configuration, model_value)
    else:
        model_path = configuration.directory / "unconfigured-asr-model"

    source_root = str(Path(__file__).resolve().parents[1])
    environment = {
        key: os.environ[key]
        for key in (
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
        if key in os.environ
    }
    environment.update(
        {
            "PYTHONPATH": source_root,
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = (
        str(python_path),
        "-u",
        "-m",
        "services.asr.worker",
        "--backend",
        backend,
        "--model-path",
        str(model_path),
        "--device",
        device,
        "--language",
        language,
        "--max-audio-bytes",
        str(max_audio_bytes),
    )
    return ASRService(
        command,
        enabled=enabled,
        backend=backend,
        model_name="SenseVoiceSmall",
        device=device,
        language=language,
        timeout_seconds=timeout_seconds,
        startup_timeout_seconds=startup_timeout_seconds,
        max_audio_bytes=max_audio_bytes,
        cwd=configuration.directory,
        env=environment,
    )


def _merge_tts_profile_values(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """递归合并单个 profile，保留旧 ``tts.backend`` 字段的继承语义。"""

    merged = {str(key): value for key, value in base.items()}
    for raw_key, value in override.items():
        key = str(raw_key)
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_tts_profile_values(existing, value)
        else:
            merged[key] = value
    return merged


def _build_tts_profiles(
    configuration: LoadedConfiguration, values: Mapping[str, Any]
) -> TTSCoordinator:
    """构造多 profile TTS，并把每个 profile 交给现有单后端工厂。"""

    raw_profiles = values.get("profiles")
    if not isinstance(raw_profiles, Mapping) or not raw_profiles:
        raise ConfigurationError("tts.profiles must be a non-empty mapping")
    base_values = {
        str(key): value for key, value in values.items() if str(key) not in {"profiles", "routing"}
    }
    profiles: dict[str, TTSProfile] = {}
    for raw_profile_id, raw_profile in raw_profiles.items():
        profile_id = str(raw_profile_id or "").strip()
        if not profile_id:
            raise ConfigurationError("tts.profiles contains an empty profile id")
        if not isinstance(raw_profile, Mapping):
            raise ConfigurationError(f"tts.profiles.{profile_id} must be a mapping")
        if any(str(key) in {"profiles", "routing"} for key in raw_profile):
            raise ConfigurationError(
                f"tts.profiles.{profile_id} cannot contain profiles or routing"
            )
        profile_values = _merge_tts_profile_values(base_values, raw_profile)
        # profile 的元数据不应被传给具体后端；其余字段继续使用已有后端工厂。
        for metadata_key in ("languages", "roles", "priority", "cooldown_seconds"):
            profile_values.pop(metadata_key, None)
        nested_values = dict(configuration.values)
        nested_values["tts"] = profile_values
        try:
            profile_coordinator = _build_tts(LoadedConfiguration(configuration.path, nested_values))
            backend = profile_coordinator.backend
            profiles[profile_id] = TTSProfile.from_mapping(profile_id, backend, raw_profile)
        except ConfigurationError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"tts.profiles.{profile_id} is invalid") from exc

    routing = values.get("routing", {})
    if routing is None:
        routing = {}
    if not isinstance(routing, Mapping):
        raise ConfigurationError("tts.routing must be a mapping")
    default_profile = str(routing.get("default_profile", "") or "").strip()
    language_profiles = routing.get("language_profiles", {})
    if language_profiles is None:
        language_profiles = {}
    if not isinstance(language_profiles, Mapping):
        raise ConfigurationError("tts.routing.language_profiles must be a mapping")
    fallback_profiles = routing.get("fallback_profiles", ())
    if fallback_profiles is None:
        fallback_profiles = ()
    if isinstance(fallback_profiles, str) or not isinstance(fallback_profiles, Sequence):
        raise ConfigurationError("tts.routing.fallback_profiles must be a list")
    role_profiles = routing.get("role_profiles", {})
    if role_profiles is None:
        role_profiles = {}
    if not isinstance(role_profiles, Mapping):
        raise ConfigurationError("tts.routing.role_profiles must be a mapping")
    try:
        router = TTSProfileRouter(
            profiles,
            default_profile=default_profile,
            language_profiles={str(key): str(value) for key, value in language_profiles.items()},
            fallback_profiles=tuple(str(item) for item in fallback_profiles),
            role_profiles={str(key): str(value) for key, value in role_profiles.items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"tts routing configuration is invalid: {type(exc).__name__}"
        ) from exc

    feature_values = _mapping(values.get("audio_features"))
    feature_analyzer = None
    if parse_bool(
        feature_values.get("enabled"),
        field_name="tts.audio_features.enabled",
        default=True,
    ):
        try:
            feature_analyzer = PcmAudioFeatureAnalyzer(
                silence_threshold=float(feature_values.get("silence_threshold", 0.02)),
                open_threshold=float(feature_values.get("open_threshold", 0.35)),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError("tts.audio_features thresholds are invalid") from exc
    segmentation_values = _mapping(values.get("segmentation"))
    return TTSCoordinator(
        router,
        queue_size=int(values.get("queue_size", 16)),
        segment_max_chars=int(segmentation_values.get("max_chars", 120)),
        segment_hard_boundaries=(
            str(segmentation_values["hard_boundaries"])
            if "hard_boundaries" in segmentation_values
            else None
        ),
        segment_soft_boundaries=(
            str(segmentation_values["soft_boundaries"])
            if "soft_boundaries" in segmentation_values
            else None
        ),
        feature_analyzer=feature_analyzer,
    )


def _build_mcp(
    values: Mapping[str, Any], *, configuration_directory: Path | None = None
) -> tuple[tuple[object, ...], tuple[MCPToolBridge, ...]]:
    """只构造 MCP 客户端对象，不启动外部进程。"""

    mcp_values = _mapping(values.get("mcp"))
    if not parse_bool(mcp_values.get("enabled", False), field_name="mcp.enabled"):
        return (), ()
    raw_servers = _sequence(mcp_values.get("servers"))
    clients: list[object] = []
    bridges: list[MCPToolBridge] = []
    server_names: set[str] = set()
    from core.adapters.mcp import (
        HTTPMCPClient,
        HTTPMCPServerConfig,
        NpmMCPServerConfig,
        StdioMCPClient,
        StdioMCPServerConfig,
        normalize_mcp_transport,
    )

    for index, raw_server in enumerate(raw_servers):
        if not isinstance(raw_server, Mapping):
            raise ConfigurationError(f"mcp.servers[{index}] must be a mapping")
        try:
            server_values = dict(raw_server)
            raw_transport = str(server_values.get("transport", "stdio") or "stdio").strip().lower()
            if raw_transport in {"", "stdio"}:
                transport = "stdio"
                config = StdioMCPServerConfig.from_mapping(server_values)
                client_factory = StdioMCPClient
            elif raw_transport == "npm":
                transport = "npm"
                config = NpmMCPServerConfig.from_mapping(server_values).stdio
                client_factory = StdioMCPClient
            else:
                transport = normalize_mcp_transport(raw_transport)
                server_values["transport"] = transport
                config = HTTPMCPServerConfig.from_mapping(server_values)
                client_factory = HTTPMCPClient
            if config.name in server_names:
                raise ConfigurationError(f"duplicate mcp server name: {config.name}")
            server_names.add(config.name)
            if transport in {"stdio", "npm"} and config.cwd and configuration_directory is not None:
                cwd = resolve_resource_path(
                    config.cwd, configuration_directory=configuration_directory
                )
                server_values["cwd"] = str(cwd)
                if transport == "npm":
                    config = NpmMCPServerConfig.from_mapping(server_values).stdio
                else:
                    config = StdioMCPServerConfig.from_mapping(server_values)
            risk = RiskLevel(int(raw_server.get("risk", int(RiskLevel.HIGH))))
            group = str(raw_server.get("group", "mcp") or "mcp").strip() or "mcp"
            client = client_factory(config)
            bridge = MCPToolBridge(client, risk=risk, group=group)
        except ConfigurationError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"mcp.servers[{index}] is invalid") from exc
        clients.append(client)
        bridges.append(bridge)
    return tuple(clients), tuple(bridges)


def _validate_runtime_values(values: Mapping[str, Any]) -> None:
    """在创建 SQLite 前校验运行时会消费的配置结构。"""

    try:
        PersonaPromptBundle.from_configuration(values)
    except PersonaConfigurationError as exc:
        raise ConfigurationError(f"persona/prompt configuration is invalid: {exc}") from exc

    def mapping(value: object, field_name: str) -> Mapping[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"{field_name} must be a mapping")
        return value

    def sequence(value: object, field_name: str) -> Sequence[object]:
        if value is None:
            return ()
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ConfigurationError(f"{field_name} must be a list")
        return value

    def number(value: object, field_name: str, *, minimum: float | None = None) -> float:
        if isinstance(value, bool):
            raise ConfigurationError(f"{field_name} must be a number")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"{field_name} must be a number") from exc
        if not math.isfinite(result) or (minimum is not None and result < minimum):
            raise ConfigurationError(f"{field_name} is outside the allowed range")
        return result

    tools_values = mapping(values.get("tools"), "tools")
    permissions = mapping(tools_values.get("permissions"), "tools.permissions")
    for field_name in ("allow", "deny"):
        for index, item in enumerate(
            sequence(permissions.get(field_name), f"tools.permissions.{field_name}")
        ):
            if not str(item or "").strip():
                raise ConfigurationError(
                    f"tools.permissions.{field_name}[{index}] must be non-empty"
                )
    for field_name in ("bypass_approval", "auto_allow_low_risk"):
        parse_bool(
            permissions.get(field_name),
            field_name=f"tools.permissions.{field_name}",
            default=False if field_name == "bypass_approval" else True,
        )
    approval_ttl = number(
        permissions.get("approval_ttl_seconds", 90.0),
        "tools.permissions.approval_ttl_seconds",
        minimum=5.0,
    )
    if approval_ttl > 3600.0:
        raise ConfigurationError(
            "tools.permissions.approval_ttl_seconds is outside the allowed range"
        )
    for field_name in ("active_groups", "command_allowlist"):
        sequence(tools_values.get(field_name), f"tools.{field_name}")

    # 插件配置在创建 SQLite 前完成严格解析。目录是否存在由插件管理器在
    # 发现阶段处理；这里只验证结构、布尔值和轮询边界，确保热重载失败不会
    # 把半解析的配置提交给运行时。
    try:
        _plugin_settings(values.get("plugins"))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"plugin configuration is invalid: {exc}") from exc

    memory_values = mapping(values.get("memory"), "memory")
    parse_bool(memory_values.get("enabled"), field_name="memory.enabled", default=True)

    def integer_field(name: str, default: int, minimum: int, maximum: int) -> int:
        raw = memory_values.get(name, default)
        if isinstance(raw, bool):
            raise ConfigurationError(f"memory.{name} must be an integer")
        try:
            parsed = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"memory.{name} must be an integer") from exc
        if parsed != raw or parsed < minimum or parsed > maximum:
            raise ConfigurationError(f"memory.{name} is outside the allowed range")
        return parsed

    def number_field(name: str, default: float, minimum: float, maximum: float) -> float:
        raw = memory_values.get(name, default)
        if isinstance(raw, bool):
            raise ConfigurationError(f"memory.{name} must be a number")
        try:
            parsed = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"memory.{name} must be a number") from exc
        if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
            raise ConfigurationError(f"memory.{name} is outside the allowed range")
        return parsed

    integer_field("recall_limit", 7, 1, 100)
    integer_field("context_max_chars", 6000, 512, 50_000)
    integer_field("context_memory_item_chars", 500, 64, 5_000)
    integer_field("recent_exchange_limit", 10, 0, 50)
    integer_field("max_memories", 2_000, 100, 10_000)
    number_field("decay_days", 7.0, 0.1, 3_650.0)
    number_field("prune_days", 30.0, 0.0, 3_650.0)
    integer_field("prune_importance_floor", 1, 0, 10)
    parse_bool(
        memory_values.get("consolidation_enabled"),
        field_name="memory.consolidation_enabled",
        default=True,
    )
    number_field("consolidation_similarity", 0.85, 0.0, 1.0)
    integer_field("max_consolidation_memories", 200, 1, 1_000)
    integer_field("summarize_every_n", 20, 1, 1_000)
    integer_field("summary_chat_limit", 20, 4, 500)
    parse_bool(
        memory_values.get("summarization_enabled"),
        field_name="memory.summarization_enabled",
        default=True,
    )
    parse_bool(
        memory_values.get("summarize_daily"),
        field_name="memory.summarize_daily",
        default=True,
    )
    integer_field("summary_interval_minutes", 360, 0, 43_200)
    integer_field("summary_min_messages", 4, 2, 500)
    integer_field("exchange_min_chars", 20, 0, 20_000)
    integer_field("exchange_importance", 2, 0, 10)
    number_field("exchange_decay_factor", 0.8, 0.0, 10.0)
    parse_bool(
        memory_values.get("auto_extract_enabled"),
        field_name="memory.auto_extract_enabled",
        default=True,
    )
    integer_field("extract_max_items", 3, 0, 20)
    integer_field("extract_max_chars", 240, 32, 2_000)
    number_field("extract_min_confidence", 0.8, 0.0, 1.0)
    integer_field("extract_default_priority", 5, 0, 10)
    number_field("recall_min_similarity", 0.04, 0.0, 1.0)
    integer_field("always_recall_priority", 9, 0, 10)
    try:
        # 复用记忆领域解析器，确保语义模型 ID、revision、维度、批量和
        # 超时边界在 --validate 与实际装配之间完全一致。
        MemorySettings.from_mapping(memory_values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc

    scheduler_values = mapping(values.get("scheduler"), "scheduler")
    parse_bool(scheduler_values.get("enabled"), field_name="scheduler.enabled", default=True)
    number(scheduler_values.get("poll_seconds", 0.5), "scheduler.poll_seconds", minimum=0.05)
    activity_values = mapping(scheduler_values.get("activity"), "scheduler.activity")
    parse_bool(
        activity_values.get("enabled"),
        field_name="scheduler.activity.enabled",
        default=True,
    )
    activity_idle_seconds = number(
        activity_values.get("idle_seconds", 300.0),
        "scheduler.activity.idle_seconds",
        minimum=1.0,
    )
    if activity_idle_seconds > 7 * 24 * 60 * 60:
        raise ConfigurationError("scheduler.activity.idle_seconds is outside the allowed range")
    activity_poll_seconds = number(
        activity_values.get("poll_seconds", 0.5),
        "scheduler.activity.poll_seconds",
        minimum=0.05,
    )
    if activity_poll_seconds > 60.0:
        raise ConfigurationError("scheduler.activity.poll_seconds is outside the allowed range")
    system_idle_provider = activity_values.get("system_idle_provider", "disabled")
    if not isinstance(system_idle_provider, str) or system_idle_provider.strip() not in {
        "disabled",
        "x11",
        "windows",
    }:
        raise ConfigurationError(
            "scheduler.activity.system_idle_provider must be disabled, x11 or windows"
        )
    system_idle_threshold_seconds = number(
        activity_values.get("system_idle_threshold_seconds", 300.0),
        "scheduler.activity.system_idle_threshold_seconds",
        minimum=1.0,
    )
    if system_idle_threshold_seconds > 7 * 24 * 60 * 60:
        raise ConfigurationError(
            "scheduler.activity.system_idle_threshold_seconds is outside the allowed range"
        )
    tasks = sequence(scheduler_values.get("tasks"), "scheduler.tasks")
    for index, raw_task in enumerate(tasks):
        item = mapping(raw_task, f"scheduler.tasks[{index}]")
        for required in ("name", "expression", "action"):
            if required not in item:
                raise ConfigurationError(f"scheduler.tasks[{index}] requires {required}")
        if not str(item.get("name") or "").strip():
            raise ConfigurationError(f"scheduler.tasks[{index}].name is required")
        try:
            validate_schedule_expression(str(item.get("expression") or ""))
        except ValueError as exc:
            raise ConfigurationError(f"scheduler.tasks[{index}].expression is invalid") from exc
        action = item.get("action")
        if not isinstance(action, Mapping) or not action:
            raise ConfigurationError(f"scheduler.tasks[{index}].action must be a non-empty mapping")
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ConfigurationError(f"scheduler.tasks[{index}].metadata must be a mapping")
        if "require_user_active" in metadata and not isinstance(
            metadata.get("require_user_active"), bool
        ):
            raise ConfigurationError(
                f"scheduler.tasks[{index}].metadata.require_user_active must be a boolean"
            )
        try:
            json.dumps(dict(metadata), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"scheduler.tasks[{index}].metadata is invalid") from exc

    triggers = sequence(scheduler_values.get("triggers"), "scheduler.triggers")
    for index, raw_trigger in enumerate(triggers):
        item = mapping(raw_trigger, f"scheduler.triggers[{index}]")
        for required in ("trigger_id", "event_name", "action"):
            if required not in item:
                raise ConfigurationError(f"scheduler.triggers[{index}] requires {required}")
        if (
            not str(item.get("trigger_id") or "").strip()
            or not str(item.get("event_name") or "").strip()
        ):
            raise ConfigurationError(f"scheduler.triggers[{index}] identifiers are required")
        action = item.get("action")
        if not isinstance(action, Mapping) or not action:
            raise ConfigurationError(
                f"scheduler.triggers[{index}].action must be a non-empty mapping"
            )
        number(
            item.get("debounce_seconds", 0.0),
            f"scheduler.triggers[{index}].debounce_seconds",
            minimum=0.0,
        )
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ConfigurationError(f"scheduler.triggers[{index}].metadata must be a mapping")
        try:
            json.dumps(dict(metadata), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"scheduler.triggers[{index}].metadata is invalid") from exc
        try:
            _trigger_metadata(item)
        except ValueError as exc:
            raise ConfigurationError(f"scheduler.triggers[{index}].conditions is invalid") from exc

    try:
        proactive_settings = ProactiveSettings.from_mapping(
            mapping(values.get("proactive"), "proactive")
        )
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if proactive_settings.enabled:
        llm_values = mapping(values.get("llm"), "llm")
        routing_values = mapping(llm_values.get("routing"), "llm.routing")
        if "proactive" not in routing_values:
            raise ConfigurationError(
                "proactive.enabled requires an explicit llm.routing.proactive route"
            )

    watcher_values = mapping(values.get("watcher"), "watcher")
    parse_bool(watcher_values.get("enabled"), field_name="watcher.enabled", default=False)
    parse_bool(watcher_values.get("emit_initial"), field_name="watcher.emit_initial", default=False)
    number(watcher_values.get("interval_seconds", 0.5), "watcher.interval_seconds", minimum=0.05)
    number(
        watcher_values.get("poll_timeout_seconds", 5.0),
        "watcher.poll_timeout_seconds",
        minimum=0.1,
    )

    config_values = mapping(values.get("config"), "config")
    reload_values = mapping(config_values.get("reload"), "config.reload")
    parse_bool(reload_values.get("enabled"), field_name="config.reload.enabled", default=True)
    number(
        reload_values.get("interval_seconds", 0.5),
        "config.reload.interval_seconds",
        minimum=0.05,
    )
    debounce = number(
        reload_values.get("debounce_seconds", 0.1),
        "config.reload.debounce_seconds",
        minimum=0.0,
    )
    if debounce > 10.0:
        raise ConfigurationError("config.reload.debounce_seconds is outside the allowed range")
    stable_checks = reload_values.get("stable_checks", 2)
    if isinstance(stable_checks, bool):
        raise ConfigurationError("config.reload.stable_checks must be an integer")
    try:
        stable_checks_number = int(stable_checks)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("config.reload.stable_checks must be an integer") from exc
    if (
        stable_checks_number < 1
        or stable_checks_number > 8
        or stable_checks_number != stable_checks
    ):
        raise ConfigurationError("config.reload.stable_checks is outside the allowed range")

    behavior_values = mapping(values.get("behavior"), "behavior")
    parse_bool(behavior_values.get("enabled"), field_name="behavior.enabled", default=False)
    minimum_interval = number(
        behavior_values.get("min_interval_seconds", 8.0),
        "behavior.min_interval_seconds",
        minimum=0.5,
    )
    maximum_interval = number(
        behavior_values.get("max_interval_seconds", 20.0),
        "behavior.max_interval_seconds",
        minimum=0.5,
    )
    if maximum_interval < minimum_interval:
        raise ConfigurationError(
            "behavior.max_interval_seconds must not be less than min_interval_seconds"
        )
    probability = number(
        behavior_values.get("probability", 0.35), "behavior.probability", minimum=0.0
    )
    if probability > 1.0:
        raise ConfigurationError("behavior.probability must be at most 1")
    interaction_cooldown = number(
        behavior_values.get("interaction_cooldown_seconds", 0.35),
        "behavior.interaction_cooldown_seconds",
        minimum=0.0,
    )
    if interaction_cooldown > 10.0:
        raise ConfigurationError(
            "behavior.interaction_cooldown_seconds is outside the allowed range"
        )
    for index, raw_action in enumerate(
        sequence(behavior_values.get("actions"), "behavior.actions")
    ):
        item = mapping(raw_action, f"behavior.actions[{index}]")
        identity = str(item.get("identity") or "").strip()
        arguments = item.get("arguments", {})
        if not identity or ":" not in identity or not isinstance(arguments, Mapping):
            raise ConfigurationError(f"behavior.actions[{index}] is invalid")
        number(item.get("weight", 1.0), f"behavior.actions[{index}].weight", minimum=0.0)
        duration = number(
            item.get("duration_seconds", 0.8),
            f"behavior.actions[{index}].duration_seconds",
            minimum=0.0,
        )
        if duration > 30.0:
            raise ConfigurationError(
                f"behavior.actions[{index}].duration_seconds is outside the allowed range"
            )

    movement_values = mapping(behavior_values.get("movement"), "behavior.movement")
    parse_bool(
        movement_values.get("enabled"),
        field_name="behavior.movement.enabled",
        default=False,
    )
    movement_min = number(
        movement_values.get("min_distance_px", 80.0),
        "behavior.movement.min_distance_px",
        minimum=1.0,
    )
    movement_max = number(
        movement_values.get("max_distance_px", 280.0),
        "behavior.movement.max_distance_px",
        minimum=1.0,
    )
    if movement_max < movement_min:
        raise ConfigurationError(
            "behavior.movement.max_distance_px must not be less than min_distance_px"
        )
    number(
        movement_values.get("step_pixels", 36.0),
        "behavior.movement.step_pixels",
        minimum=4.0,
    )
    number(
        movement_values.get("step_interval_seconds", 0.08),
        "behavior.movement.step_interval_seconds",
        minimum=0.02,
    )
    max_speed = number(
        movement_values.get("max_speed_px_per_second", 240.0),
        "behavior.movement.max_speed_px_per_second",
        minimum=20.0,
    )
    if max_speed > 1200:
        raise ConfigurationError(
            "behavior.movement.max_speed_px_per_second is outside the allowed range"
        )
    max_steps = number(
        movement_values.get("max_steps", 64),
        "behavior.movement.max_steps",
        minimum=1.0,
    )
    if max_steps > 64:
        raise ConfigurationError("behavior.movement.max_steps must be at most 64")
    if int(max_steps) != max_steps:
        raise ConfigurationError("behavior.movement.max_steps must be an integer")
    parse_bool(
        movement_values.get("flip_facing"),
        field_name="behavior.movement.flip_facing",
        default=True,
    )
    for field_name in ("moving_motion", "idle_motion", "movement_identity"):
        raw_value = movement_values.get(field_name)
        if raw_value is not None and not isinstance(raw_value, str):
            raise ConfigurationError(f"behavior.movement.{field_name} must be a string")
    movement_identity = str(
        movement_values.get("movement_identity", "pet:autonomous_move") or "pet:autonomous_move"
    ).strip()
    if movement_identity != "pet:autonomous_move":
        raise ConfigurationError("behavior.movement.movement_identity must be pet:autonomous_move")

    tts_values = mapping(values.get("tts"), "tts")
    tts_enabled = parse_bool(tts_values.get("enabled"), field_name="tts.enabled", default=False)
    tts_backend = str(tts_values.get("backend", "text_only") or "").strip().lower()
    tts_language = str(tts_values.get("language", "zh") or "").strip()
    if not tts_language:
        raise ConfigurationError("tts.language is required")
    if not canonical_tts_language(tts_language):
        raise ConfigurationError("tts.language is invalid")
    tts_role = tts_values.get("role", "")
    if tts_role is not None:
        tts_role = str(tts_role)
        if len(tts_role.strip()) > 128 or any(char in tts_role for char in "\r\n\x00"):
            raise ConfigurationError("tts.role is invalid")
    tts_output_device_id = tts_values.get("output_device_id", "")
    if tts_output_device_id is not None:
        if not isinstance(tts_output_device_id, str):
            raise ConfigurationError("tts.output_device_id must be a string")
        if len(tts_output_device_id) > 1_024 or any(
            char in tts_output_device_id for char in "\r\n\x00"
        ):
            raise ConfigurationError("tts.output_device_id is invalid")

    raw_profiles = tts_values.get("profiles")
    if raw_profiles is not None:
        if not isinstance(raw_profiles, Mapping) or not raw_profiles:
            raise ConfigurationError("tts.profiles must be a non-empty mapping")
        for raw_profile_id, raw_profile in raw_profiles.items():
            profile_id = str(raw_profile_id or "").strip()
            if not profile_id:
                raise ConfigurationError("tts.profiles contains an empty profile id")
            if not isinstance(raw_profile, Mapping):
                raise ConfigurationError(f"tts.profiles.{profile_id} must be a mapping")
            if any(str(key) in {"profiles", "routing"} for key in raw_profile):
                raise ConfigurationError(
                    f"tts.profiles.{profile_id} cannot contain profiles or routing"
                )
            try:
                # 元数据使用与运行时构造相同的规范化逻辑校验；实际后端
                # 仍由 `_build_tts_profiles` 按继承后的字段创建。
                TTSProfile.from_mapping(profile_id, TextOnlyBackend(), raw_profile)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ConfigurationError(f"tts.profiles.{profile_id} is invalid") from exc
            if "enabled" in raw_profile:
                parse_bool(
                    raw_profile.get("enabled"),
                    field_name=f"tts.profiles.{profile_id}.enabled",
                )
            profile_values = _merge_tts_profile_values(
                {
                    str(key): value
                    for key, value in tts_values.items()
                    if str(key) not in {"profiles", "routing"}
                },
                raw_profile,
            )
            profile_backend = str(profile_values.get("backend", "text_only") or "").strip().lower()
            if profile_backend not in {
                "",
                "text_only",
                "disabled",
                "gpt_sovits",
                "gpt-sovits",
                "gpt_sovits_http",
                "gpt_sovits_stdio",
                "gpt-sovits-stdio",
                "subprocess",
            }:
                raise ConfigurationError(f"unsupported tts profile backend: {profile_backend}")
            profile_enabled = parse_bool(
                profile_values.get("enabled"),
                field_name=f"tts.profiles.{profile_id}.enabled",
                default=tts_enabled,
            )
            if profile_enabled and profile_backend in {
                "gpt_sovits",
                "gpt-sovits",
                "gpt_sovits_http",
                "gpt_sovits_stdio",
                "gpt-sovits-stdio",
            }:
                has_fixed_reference = bool(
                    str(profile_values.get("ref_audio_path", "") or "").strip()
                )
                try:
                    profile_references = normalize_reference_audios(
                        profile_values.get("reference_audios")
                    )
                except ValueError as exc:
                    raise ConfigurationError(str(exc)) from exc
                has_profile_reference = any(
                    bool(str(item.get("path", "") or "").strip())
                    for item in profile_references.values()
                )
                has_reference_directory = bool(str(profile_values.get("ref_dir", "") or "").strip())
                if not (has_fixed_reference or has_profile_reference or has_reference_directory):
                    raise ConfigurationError(
                        f"tts.profiles.{profile_id} GPT-SoVITS requires "
                        "ref_audio_path, ref_dir, or reference_audios"
                    )
    routing_values = tts_values.get("routing")
    if routing_values is not None:
        if not isinstance(routing_values, Mapping):
            raise ConfigurationError("tts.routing must be a mapping")
        language_profile_values = routing_values.get("language_profiles", {})
        if language_profile_values is not None and not isinstance(language_profile_values, Mapping):
            raise ConfigurationError("tts.routing.language_profiles must be a mapping")
        fallback_profile_values = routing_values.get("fallback_profiles", ())
        if fallback_profile_values is not None and (
            isinstance(fallback_profile_values, str)
            or not isinstance(fallback_profile_values, Sequence)
        ):
            raise ConfigurationError("tts.routing.fallback_profiles must be a list")
        role_profile_values = routing_values.get("role_profiles", {})
        if role_profile_values is not None and not isinstance(role_profile_values, Mapping):
            raise ConfigurationError("tts.routing.role_profiles must be a mapping")
        default_role = routing_values.get("default_role", "")
        if default_role is not None:
            default_role = str(default_role)
            if len(default_role.strip()) > 128 or any(char in default_role for char in "\r\n\x00"):
                raise ConfigurationError("tts.routing.default_role is invalid")
    segmentation_values = mapping(tts_values.get("segmentation"), "tts.segmentation")
    segment_max_chars = number(
        segmentation_values.get("max_chars", 120),
        "tts.segmentation.max_chars",
        minimum=20.0,
    )
    if segment_max_chars > 2000 or int(segment_max_chars) != segment_max_chars:
        raise ConfigurationError("tts.segmentation.max_chars is outside the allowed range")
    for field_name in ("hard_boundaries", "soft_boundaries"):
        raw_boundaries = segmentation_values.get(field_name)
        if raw_boundaries is not None and not isinstance(raw_boundaries, str):
            raise ConfigurationError(f"tts.segmentation.{field_name} must be a string")
    if "hard_boundaries" in segmentation_values and not str(
        segmentation_values.get("hard_boundaries") or ""
    ):
        raise ConfigurationError("tts.segmentation.hard_boundaries cannot be empty")
    if tts_values.get("prompt_lang") is not None and not isinstance(
        tts_values.get("prompt_lang"), str
    ):
        raise ConfigurationError("tts.prompt_lang must be a string")
    parse_bool(tts_values.get("verify_tls"), field_name="tts.verify_tls", default=True)
    parse_bool(tts_values.get("health_probe"), field_name="tts.health_probe", default=False)
    parse_bool(
        tts_values.get("startup_handshake"),
        field_name="tts.startup_handshake",
        default=True,
    )
    if (
        tts_enabled
        and tts_backend in {"gpt_sovits_stdio", "gpt-sovits-stdio"}
        and not parse_bool(
            tts_values.get("startup_handshake"),
            field_name="tts.startup_handshake",
            default=True,
        )
    ):
        raise ConfigurationError("tts.startup_handshake cannot be disabled for gpt_sovits_stdio")
    parse_bool(tts_values.get("require_final"), field_name="tts.require_final", default=False)
    parse_bool(tts_values.get("require_audio"), field_name="tts.require_audio", default=True)
    parse_bool(tts_values.get("validate_pcm"), field_name="tts.validate_pcm", default=False)
    for field_name in (
        "ref_audio_path",
        "ref_dir",
        "gpt_path",
        "sovits_path",
        "gpt_weights_dir",
        "sovits_weights_dir",
        "gpt_model",
        "sovits_model",
        "engine_root",
        "engine_config",
        "python_executable",
    ):
        raw_value = tts_values.get(field_name)
        if raw_value is not None and not isinstance(raw_value, str):
            raise ConfigurationError(f"tts.{field_name} must be a string")
    try:
        references = normalize_reference_audios(tts_values.get("reference_audios"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    if tts_enabled and tts_backend in {
        "gpt_sovits",
        "gpt-sovits",
        "gpt_sovits_http",
        "gpt_sovits_stdio",
        "gpt-sovits-stdio",
    }:
        has_fixed_reference = bool(str(tts_values.get("ref_audio_path", "") or "").strip())
        has_profile_reference = any(
            bool(str(item.get("path", "") or "").strip()) for item in references.values()
        )
        has_reference_directory = bool(str(tts_values.get("ref_dir", "") or "").strip())
        if not (has_fixed_reference or has_profile_reference or has_reference_directory):
            raise ConfigurationError(
                "tts GPT-SoVITS requires ref_audio_path, ref_dir, or reference_audios"
            )
    if tts_values.get("options") is not None and not isinstance(tts_values.get("options"), Mapping):
        raise ConfigurationError("tts.options must be a mapping")
    if "top_k" in tts_values:
        top_k = number(tts_values.get("top_k"), "tts.top_k", minimum=1.0)
        if int(top_k) != top_k:
            raise ConfigurationError("tts.top_k must be an integer")
    for field_name in ("top_p", "temperature", "speed", "repetition_penalty"):
        if field_name in tts_values:
            number(tts_values.get(field_name), f"tts.{field_name}", minimum=0.0)
    if "sample_steps" in tts_values:
        sample_steps = number(tts_values.get("sample_steps"), "tts.sample_steps", minimum=1.0)
        if int(sample_steps) != sample_steps:
            raise ConfigurationError("tts.sample_steps must be an integer")
    if "streaming_mode" in tts_values:
        streaming_mode = tts_values.get("streaming_mode")
        if isinstance(streaming_mode, bool):
            raise ConfigurationError("tts.streaming_mode must be an integer from 0 to 3")
        streaming_mode_number = number(streaming_mode, "tts.streaming_mode", minimum=0.0)
        if streaming_mode_number > 3 or int(streaming_mode_number) != streaming_mode_number:
            raise ConfigurationError("tts.streaming_mode must be an integer from 0 to 3")
    number(tts_values.get("timeout_seconds", 60.0), "tts.timeout_seconds", minimum=0.001)
    number(
        tts_values.get("startup_timeout_seconds", 300.0),
        "tts.startup_timeout_seconds",
        minimum=0.001,
    )
    shutdown_timeout = number(
        tts_values.get("shutdown_timeout_seconds", 2.0),
        "tts.shutdown_timeout_seconds",
        minimum=0.05,
    )
    if shutdown_timeout > 30.0:
        raise ConfigurationError("tts.shutdown_timeout_seconds is outside the allowed range")
    speed_factor = number(tts_values.get("speed_factor", 1.0), "tts.speed_factor", minimum=0.1)
    if speed_factor > 4.0:
        raise ConfigurationError("tts.speed_factor is outside the allowed range")
    raw_sample_rate = number(
        tts_values.get("raw_sample_rate", 24000), "tts.raw_sample_rate", minimum=1.0
    )
    if raw_sample_rate > 384000 or int(raw_sample_rate) != raw_sample_rate:
        raise ConfigurationError("tts.raw_sample_rate is outside the allowed range")
    raw_channels = number(tts_values.get("raw_channels", 1), "tts.raw_channels", minimum=1.0)
    if raw_channels not in {1, 2} or int(raw_channels) != raw_channels:
        raise ConfigurationError("tts.raw_channels is outside the allowed range")
    expected_sample_rate = tts_values.get("expected_sample_rate", 32000)
    if expected_sample_rate is not None:
        expected_sample_rate_number = number(
            expected_sample_rate,
            "tts.expected_sample_rate",
            minimum=1.0,
        )
        if (
            expected_sample_rate_number > 384000
            or int(expected_sample_rate_number) != expected_sample_rate
        ):
            raise ConfigurationError("tts.expected_sample_rate is outside the allowed range")
    max_total_bytes = number(
        tts_values.get("max_total_bytes", 64 * 1024 * 1024),
        "tts.max_total_bytes",
        minimum=1024.0,
    )
    if max_total_bytes > 512 * 1024 * 1024 or int(max_total_bytes) != max_total_bytes:
        raise ConfigurationError("tts.max_total_bytes is outside the allowed range")
    queue_size = number(tts_values.get("queue_size", 16), "tts.queue_size", minimum=1.0)
    if queue_size > 128 or int(queue_size) != queue_size:
        raise ConfigurationError("tts.queue_size is outside the allowed range")
    audio_feature_values = mapping(tts_values.get("audio_features"), "tts.audio_features")
    parse_bool(
        audio_feature_values.get("enabled"),
        field_name="tts.audio_features.enabled",
        default=True,
    )
    silence_threshold = number(
        audio_feature_values.get("silence_threshold", 0.02),
        "tts.audio_features.silence_threshold",
        minimum=0.0,
    )
    open_threshold = number(
        audio_feature_values.get("open_threshold", 0.35),
        "tts.audio_features.open_threshold",
        minimum=0.0,
    )
    if silence_threshold >= open_threshold or open_threshold > 1.0:
        raise ConfigurationError(
            "tts.audio_features thresholds must satisfy 0 <= silence < open <= 1"
        )

    stream_values = mapping(mapping(values.get("ui"), "ui").get("stream"), "ui.stream")
    for field_name, default in (
        ("show_reasoning", False),
        ("show_murmur", True),
        ("show_tool_status", True),
    ):
        parse_bool(
            stream_values.get(field_name), field_name=f"ui.stream.{field_name}", default=default
        )
    max_bubble = number(
        stream_values.get("max_bubble_chars", 4000), "ui.stream.max_bubble_chars", minimum=64.0
    )
    if max_bubble > 100_000 or int(max_bubble) != max_bubble:
        raise ConfigurationError("ui.stream.max_bubble_chars is outside the allowed range")
    try:
        validate_hotkey_config(mapping(values.get("ui"), "ui").get("hotkeys"))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"hotkey configuration is invalid: {exc}") from exc
    parse_bool(
        mapping(values.get("ui"), "ui").get("always_on_top"),
        field_name="ui.always_on_top",
        default=True,
    )
    parse_bool(
        mapping(values.get("ui"), "ui").get("window_locked"),
        field_name="ui.window_locked",
        default=False,
    )
    qt_platform = str(mapping(values.get("ui"), "ui").get("qt_platform", "auto") or "auto")
    if qt_platform.strip().lower() not in {"auto", "xcb", "wayland"}:
        raise ConfigurationError("ui.qt_platform must be auto, xcb or wayland")
    parse_bool(
        mapping(values.get("ui"), "ui").get("auto_open_model_setup"),
        field_name="ui.auto_open_model_setup",
        default=True,
    )
    try:
        theme_from_configuration(values)
    except MD3ThemeError as exc:
        raise ConfigurationError(f"ui.theme configuration is invalid: {exc}") from exc

    mcp_values = mapping(values.get("mcp"), "mcp")
    parse_bool(mcp_values.get("enabled"), field_name="mcp.enabled", default=False)
    mcp_servers = sequence(mcp_values.get("servers"), "mcp.servers")
    mcp_names: set[str] = set()
    for index, raw_server in enumerate(mcp_servers):
        item = mapping(raw_server, f"mcp.servers[{index}]")
        try:
            from core.adapters.mcp import (
                HTTPMCPServerConfig,
                NpmMCPServerConfig,
                StdioMCPServerConfig,
                normalize_mcp_transport,
            )

            raw_transport = str(item.get("transport", "stdio") or "stdio").strip().lower()
            if raw_transport in {"", "stdio"}:
                server_config = StdioMCPServerConfig.from_mapping(item)
            elif raw_transport == "npm":
                server_config = NpmMCPServerConfig.from_mapping(item).stdio
            else:
                normalized_transport = normalize_mcp_transport(raw_transport)
                http_values = dict(item)
                http_values["transport"] = normalized_transport
                server_config = HTTPMCPServerConfig.from_mapping(http_values)
            if server_config.name in mcp_names:
                raise ValueError("duplicate MCP server name")
            mcp_names.add(server_config.name)
            RiskLevel(int(item.get("risk", int(RiskLevel.HIGH))))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(f"mcp.servers[{index}] is invalid") from exc

    # 本地 HTTP 控制面是显式选择的宿主能力。即使它保持关闭，也校验完整
    # 的 schema，避免热加载或 GUI 接线在启用时绕过回环/端口边界。
    validate_local_web_configuration(values)

    storage_values = mapping(values.get("storage"), "storage")
    database_value = storage_values.get("database", "./data/meapet.sqlite3")
    if not isinstance(database_value, str) or not database_value.strip():
        raise ConfigurationError("storage.database must be a non-empty string")

    rendering_values = mapping(values.get("rendering"), "rendering")
    resource_value = rendering_values.get("resource_root", "./resources")
    if not isinstance(resource_value, str) or not resource_value.strip():
        raise ConfigurationError("rendering.resource_root must be a non-empty string")
    model_value = rendering_values.get("model", "")
    if model_value is not None and not isinstance(model_value, str):
        raise ConfigurationError("rendering.model must be a string or null")
    if isinstance(model_value, str) and any(char in model_value for char in "\x00\r\n"):
        raise ConfigurationError("rendering.model contains invalid control characters")
    if isinstance(model_value, str) and model_value.strip() and model_value.strip() != "auto":
        model_key = model_value.strip()
        if (
            "\\" in model_key
            or Path(model_key).is_absolute()
            or any(part in {"", ".", ".."} for part in model_key.split("/"))
            or not model_key.endswith(".model3.json")
        ):
            raise ConfigurationError(
                "rendering.model must be a resource-relative .model3.json path"
            )
    sprite_scale = number(
        rendering_values.get("sprite_scale", 0.6),
        "rendering.sprite_scale",
        minimum=0.1,
    )
    if sprite_scale > 2.0:
        raise ConfigurationError("rendering.sprite_scale is outside the allowed range")
    frame_rate = number(
        rendering_values.get("frame_rate", 60.0),
        "rendering.frame_rate",
        minimum=15.0,
    )
    if frame_rate > 120.0:
        raise ConfigurationError("rendering.frame_rate is outside the allowed range")
    geometry_audit_hz = number(
        rendering_values.get("geometry_audit_hz", 30.0),
        "rendering.geometry_audit_hz",
        minimum=1.0,
    )
    if geometry_audit_hz > 60.0:
        raise ConfigurationError("rendering.geometry_audit_hz is outside the allowed range")
    try:
        normalize_renderer_backend(rendering_values.get("backend", "auto"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    try:
        LoggingSettings.from_mapping(values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"logging configuration is invalid: {type(exc).__name__}") from exc


def validate_runtime_configuration(configuration: LoadedConfiguration) -> None:
    """校验运行时装配所需配置，但不创建数据库或启动外部服务。

    ``--validate`` 必须覆盖 ``build_runtime`` 在打开 SQLite 前执行的配置边界。
    因此这里除了基础字段校验，还会构造路由、TTS 后端以及临时的调度/触发器
    服务，用同一组领域校验拒绝无法装配的配置；这些对象不持有数据库连接，调用
    完成后即可安全丢弃。
    """

    values = configuration.values
    _validate_runtime_values(values)
    try:
        # 路由和 TTS 构造器本身包含端点、协议、重试策略和音频格式校验。
        ModelRouter.from_mapping(values)
        _build_tts(configuration)
        # 配置校验命令保持严格契约：显式启用 ASR 时必须同时提供
        # worker 解释器和模型目录。运行时装配本身允许在无资源的离线
        # smoke 场景安全降级，但不应让 --validate 掩盖生产配置错误。
        asr_values = _mapping(values.get("asr"))
        asr_enabled = parse_bool(
            asr_values.get("enabled", False),
            field_name="asr.enabled",
            default=False,
        )
        if asr_enabled:
            if not str(asr_values.get("python_executable", "") or "").strip():
                raise ConfigurationError("asr.python_executable is required")
            if not str(asr_values.get("model_path", "") or "").strip():
                raise ConfigurationError("asr.model_path is required")
        _build_asr(configuration)
        _build_mcp(values, configuration_directory=configuration.directory)

        # 与 ``build_runtime`` 使用同一领域方法校验任务/触发器 ID、所有者和
        # JSON 可序列化动作；不绑定 action_runner，因此不会执行任何工具。
        scheduler = SchedulerService()
        triggers = TriggerService()
        scheduler_values = _mapping(values.get("scheduler"))
        for raw_task in _sequence(scheduler_values.get("tasks")):
            item = _mapping(raw_task)
            scheduler.upsert(
                task_id=str(item.get("task_id", "")).strip() or None,
                name=str(item["name"]),
                expression=str(item["expression"]),
                action=dict(item["action"]),
                owner=str(item.get("owner", "config")),
                metadata=dict(item.get("metadata", {}))
                if isinstance(item.get("metadata", {}), Mapping)
                else item.get("metadata"),
            )
        for raw_trigger in _sequence(scheduler_values.get("triggers")):
            item = _mapping(raw_trigger)
            triggers.register(
                trigger_id=str(item["trigger_id"]),
                event_name=str(item["event_name"]),
                action=dict(item["action"]),
                owner=str(item.get("owner", "config")),
                debounce_seconds=float(item.get("debounce_seconds", 0.0)),
                metadata=_trigger_metadata(item),
            )
    except ConfigurationError:
        raise
    except AdapterConfigurationError as exc:
        # 适配器错误的 safe_message 只包含字段/协议诊断，不包含密钥或请求正文。
        raise ConfigurationError(f"runtime configuration is invalid: {exc.safe_message}") from exc
    except ModelRoutingError as exc:
        raise ConfigurationError(f"runtime configuration is invalid: {exc}") from exc
    except (OverflowError, PermissionError, TypeError, ValueError) as exc:
        # 其他异常可能携带路径、命令参数或第三方正文，只保留类型名。
        raise ConfigurationError(f"runtime configuration is invalid: {type(exc).__name__}") from exc


def initialize_database(configuration: LoadedConfiguration) -> Path:
    """在 GUI 导入前创建父目录并幂等迁移 SQLite，然后关闭连接。

    ``build_runtime`` 会在随后再次打开同一路径并复用幂等 schema 迁移。该独立
    步骤让默认启动能够尽早发现数据库目录/权限问题，同时不让 ``--validate``
    或 ``--no-gui`` 产生 SQLite 副作用。
    """

    validate_runtime_configuration(configuration)
    storage_values = _mapping(configuration.values.get("storage"))
    database_value = storage_values.get("database", "./data/meapet.sqlite3")
    if not isinstance(database_value, str) or not database_value.strip():
        raise ConfigurationError("storage.database must be a non-empty string")
    database_path = _resolve_path(database_value, configuration, "./data/meapet.sqlite3")
    try:
        database = Database(database_path)
        ApiCallAuditRepository(database)
    except (OSError, PermissionError, ValueError, sqlite3.Error) as exc:
        raise ConfigurationError("database initialization failed") from exc
    try:
        return database_path
    finally:
        database.close()


def build_runtime(
    configuration: LoadedConfiguration,
    inventory: ResourceInventory,
    *,
    platform: DesktopPlatform | None = None,
    pet_controller: MutablePetController | None = None,
    event_sink: Callable[[object], Awaitable[None] | None] | None = None,
    ui_dispatcher: object | None = None,
) -> ApplicationRuntime:
    """按 YAML 创建完整运行时；不要求 Qt 或模型渠道已安装。"""

    values = configuration.values
    _validate_runtime_values(values)
    storage_values = _mapping(values.get("storage"))
    tools_values = _mapping(values.get("tools"))
    # 先验证不依赖数据库的外部配置，避免后续失败留下已打开的 SQLite 连接。
    try:
        plugin_settings = _plugin_settings(values.get("plugins"))
        router = ModelRouter.from_mapping(values)
        tts = _build_tts(configuration)
        asr = _build_asr(configuration)
        mcp_clients, mcp_bridges = _build_mcp(
            values, configuration_directory=configuration.directory
        )
    except ConfigurationError:
        raise
    except AdapterConfigurationError as exc:
        raise ConfigurationError(f"runtime configuration is invalid: {exc.safe_message}") from exc
    except ModelRoutingError as exc:
        raise ConfigurationError(f"runtime configuration is invalid: {exc}") from exc
    except (OverflowError, PermissionError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"runtime configuration is invalid: {type(exc).__name__}") from exc
    database_path = _resolve_path(
        storage_values.get("database"), configuration, "./data/meapet.sqlite3"
    )
    registry = ToolRegistry()
    permission_values = _mapping(tools_values.get("permissions"))
    permissions = PermissionService(
        allow=tuple(str(item) for item in _sequence(permission_values.get("allow"))),
        deny=tuple(str(item) for item in _sequence(permission_values.get("deny"))),
        bypass_approval=parse_bool(
            permission_values.get("bypass_approval", False),
            field_name="tools.permissions.bypass_approval",
        ),
        auto_allow_low_risk=parse_bool(
            permission_values.get("auto_allow_low_risk", True),
            field_name="tools.permissions.auto_allow_low_risk",
        ),
        approval_ttl_seconds=float(permission_values.get("approval_ttl_seconds", 90.0)),
    )
    base_platform: DesktopPlatform = platform if platform is not None else create_desktop_platform()
    platform_value: DesktopPlatform = DispatchingDesktopPlatform(base_platform, ui_dispatcher)
    pet_value = pet_controller or MutablePetController(dispatcher=ui_dispatcher)
    if pet_controller is not None and ui_dispatcher is not None:
        pet_controller.set_dispatcher(ui_dispatcher)
    scheduler: SchedulerService
    triggers: TriggerService

    async def execute_action(
        action: Mapping[str, Any],
        owner: str,
        source: str,
        call_suffix: str,
        event_payload: Mapping[str, Any] | None = None,
    ) -> object:
        identity = str(action.get("identity", "")).strip()
        arguments = action.get("arguments", action.get("args", {}))
        if not identity or not isinstance(arguments, Mapping):
            return {"status": "denied", "reason": "action identity or arguments are invalid"}
        public_payload = ProactiveCoordinator._safe_payload(event_payload)
        context = ToolCallContext(
            "default",
            owner,
            f"{source}-{str(call_suffix)[:220]}-{secrets.token_urlsafe(8)}",
            source=source,
            metadata={"event_payload": public_payload} if public_payload else {},
        )
        try:
            result = tools.execute(
                call_id=f"{source}-{secrets.token_urlsafe(8)}",
                identity=identity,
                arguments=dict(arguments),
                context=context,
            )
            outcome = await result if inspect.isawaitable(result) else result
            status = (
                str(
                    outcome.get("status", "")
                    if isinstance(outcome, Mapping)
                    else getattr(outcome, "status", "")
                )
                .strip()
                .lower()
            )
            if status == "approval_required":
                # 调度器/触发器没有可绑定的对话回合，审批卡无法通过
                # ConversationService 续接；留下 pending 会让任务显示成功
                # 却永远不执行。因此显式撤销并让外层记录一次失败。
                approval = getattr(outcome, "approval", None)
                approval_id = str(getattr(approval, "approval_id", "") or "").strip()
                if approval_id:
                    tools.deny_approval(approval_id)
                raise PermissionError(f"{source} action requires interactive approval")
            if status in {"denied", "failed", "error", "unavailable"}:
                reason = ""
                if isinstance(outcome, Mapping):
                    reason = str(outcome.get("reason", outcome.get("error", "")) or "").strip()
                else:
                    content = getattr(outcome, "content", {})
                    if isinstance(content, Mapping):
                        reason = str(content.get("reason", content.get("error", "")) or "").strip()
                suffix = f": {reason[:160]}" if reason else ""
                raise RuntimeError(f"{source} action {status}{suffix}")
            return outcome
        finally:
            tools.clear_turn(context)

    async def scheduled_action(action: Mapping[str, Any], task: object) -> object:
        owner = str(getattr(task, "owner", "scheduler"))
        suffix = str(getattr(task, "task_id", "task"))
        expression = str(getattr(task, "expression", "") or "")
        payload = {
            "task_id": suffix,
            "name": str(getattr(task, "name", "") or "")[:128],
            "expression": expression[:128],
            "scheduled_at": time(),
            "user_active": activity.is_user_active(),
        }
        if str(action.get("identity", "") or "").strip() == "proactive:run":
            arguments = _mapping(action.get("arguments", action.get("args", {})))
            event_name = "daily" if expression.strip().lower().startswith("daily:") else "scheduler"
            return await runtime.notify_proactive_event(
                event_name,
                payload,
                instruction=str(arguments.get("instruction", "") or ""),
            )
        return await execute_action(action, owner, "scheduler", suffix, payload)

    async def triggered_action(
        action: Mapping[str, Any], trigger: object, payload: Mapping[str, Any]
    ) -> object:
        owner = str(getattr(trigger, "owner", "trigger"))
        suffix = str(getattr(trigger, "trigger_id", "trigger"))
        if str(action.get("identity", "") or "").strip() == "proactive:run":
            arguments = _mapping(action.get("arguments", action.get("args", {})))
            return await runtime.notify_proactive_event(
                str(getattr(trigger, "event_name", "custom") or "custom"),
                payload,
                instruction=str(arguments.get("instruction", "") or ""),
            )
        return await execute_action(action, owner, "trigger", suffix, payload)

    scheduler = SchedulerService(action_runner=scheduled_action)
    triggers = TriggerService(action_runner=triggered_action)
    scheduler_activity_values = _mapping(_mapping(values.get("scheduler")).get("activity"))
    try:
        activity_enabled = parse_bool(
            scheduler_activity_values.get("enabled", True),
            field_name="scheduler.activity.enabled",
            default=True,
        )
        activity_idle_seconds = float(scheduler_activity_values.get("idle_seconds", 300.0))
        activity_poll_seconds = float(scheduler_activity_values.get("poll_seconds", 0.5))
        system_idle_provider = scheduler_activity_values.get("system_idle_provider", "disabled")
        if not isinstance(system_idle_provider, str):
            raise ValueError("system_idle_provider must be a string")
        system_idle_provider = system_idle_provider.strip()
        system_idle_threshold_seconds = float(
            scheduler_activity_values.get("system_idle_threshold_seconds", 300.0)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("scheduler.activity configuration is invalid") from exc
    system_idle_probe = None
    platform_backend = str(getattr(base_platform, "backend", "") or "").strip().lower()
    provider_backend = {
        "x11": "x11",
        SYSTEM_IDLE_PROVIDER_WINDOWS: "windows",
    }.get(system_idle_provider)
    if provider_backend == platform_backend:
        system_idle_probe = getattr(base_platform, "system_idle_seconds", None)
        if not callable(system_idle_probe):
            system_idle_probe = None
    activity = UserActivityTracker(
        triggers=triggers,
        enabled=activity_enabled,
        idle_seconds=activity_idle_seconds,
        poll_seconds=activity_poll_seconds,
        system_idle_provider=system_idle_provider,
        system_idle_threshold_seconds=system_idle_threshold_seconds,
        system_idle_probe=system_idle_probe,
    )
    scheduler.set_activity_provider(activity.is_user_active)

    async def _runtime_event_sink(
        event: object,
        *,
        approval_tools: ToolExecutionService,
    ) -> None:
        interaction_state.consume(event)
        event_context = getattr(event, "context", None)
        if isinstance(event, ApprovalRequested) and isinstance(event_context, ConversationContext):
            approval_context = ToolCallContext(
                event_context.profile_id,
                event_context.session_id,
                event_context.turn_id,
                source="model",
                metadata={
                    "mode": event_context.mode,
                    "generation_id": event_context.generation_id,
                },
            )
            pending = approval_tools.pending_approvals_for_context(approval_context)
        else:
            pending = runtime.pending_approvals_for_ui()
        interaction_state.set_pending_approvals(pending)
        if event_sink is not None:
            result = event_sink(event)
            if inspect.isawaitable(result):
                await result
        context = getattr(event, "context", None)
        if context is None:
            return
        payload: dict[str, Any] = {
            "profile_id": str(getattr(context, "profile_id", "")),
            "session_id": str(getattr(context, "session_id", "")),
            "turn_id": str(getattr(context, "turn_id", "")),
        }

        def wake_proactive_after_terminal() -> None:
            wake_proactive = getattr(runtime.proactive, "wake", None)
            if not callable(wake_proactive):
                return
            wake_proactive()
            # 终态事件由 ConversationService 在 finally 清理活动任务之前
            # 发出；当前 task 继续消费事件流后才会完成清理。当前拍唤醒
            # 可能仍观察到活动，下一拍再唤醒可消除这段固定超时窗口。
            try:
                asyncio.get_running_loop().call_soon(wake_proactive)
            except RuntimeError:
                pass

        if isinstance(event, TurnFinished):
            payload.update(
                {
                    "finish_reason": event.finish_reason,
                    "usage": dict(event.usage),
                }
            )
            summarizer = getattr(runtime, "memory_summarizer", None)
            wake_summary = getattr(summarizer, "wake", None)
            if callable(wake_summary):
                wake_summary()
            try:
                await triggers.emit("conversation_finished", payload)
            finally:
                # 触发器执行失败或配置持久化异常时也必须唤醒主动队列；
                # 否则终态已发出但主动 worker 只能依赖 0.5 秒兜底轮询。
                wake_proactive_after_terminal()
        elif isinstance(event, TurnFailed):
            payload.update(
                {
                    "category": event.category,
                    "safe_message": event.safe_message,
                    "retryable": event.retryable,
                }
            )
            try:
                await triggers.emit("conversation_failed", payload)
            finally:
                wake_proactive_after_terminal()

    async def runtime_event_sink(event: object) -> None:
        """把普通对话事件绑定到普通对话的权限执行器。"""

        await _runtime_event_sink(event, approval_tools=tools)

    async def proactive_event_sink(event: object) -> None:
        """把主动回合事件绑定到独立的 fail-closed 权限执行器。"""

        await _runtime_event_sink(event, approval_tools=proactive_tools)

    watcher_values = _mapping(values.get("watcher"))
    watcher = DesktopWindowWatcher(
        platform=platform_value,
        triggers=triggers,
        poll_seconds=float(watcher_values.get("interval_seconds", 0.5)),
        emit_initial=parse_bool(
            watcher_values.get("emit_initial", False), field_name="watcher.emit_initial"
        ),
        poll_timeout_seconds=float(watcher_values.get("poll_timeout_seconds", 5.0)),
        activity_provider=activity.is_user_active,
    )
    stream_values = _mapping(_mapping(values.get("ui")).get("stream"))
    presentation = PresentationService(
        show_reasoning=parse_bool(
            stream_values.get("show_reasoning", False), field_name="ui.stream.show_reasoning"
        ),
        show_murmur=parse_bool(
            stream_values.get("show_murmur", True), field_name="ui.stream.show_murmur"
        ),
        show_tool_status=parse_bool(
            stream_values.get("show_tool_status", True), field_name="ui.stream.show_tool_status"
        ),
        max_text_length=int(stream_values.get("max_bubble_chars", 4000)),
    )
    interaction_state = InteractionState(
        diagnostics=router.diagnostics("dialogue"),
        max_text_length=int(stream_values.get("max_bubble_chars", 4000)),
        show_reasoning=parse_bool(
            stream_values.get("show_reasoning", False), field_name="ui.stream.show_reasoning"
        ),
    )
    register_builtin_tools(
        registry,
        platform=platform_value,
        pet_controller=pet_value,
        scheduler=scheduler,
        tts=tts,
        asr=asr,
        asr_provider=lambda: runtime.asr,
        presentation=presentation,
        module_status_provider=lambda: runtime.module_diagnostics(),
        diary_provider=lambda: runtime.diary,
        default_language=str(_mapping(values.get("tts")).get("language", "zh")),
        command_allowlist=tuple(
            str(item) for item in _sequence(tools_values.get("command_allowlist"))
        ),
    )
    tools = ToolExecutionService(
        registry,
        permissions,
        execution_record_sink=router.record_tool_execution,
    )
    proactive_permissions = PermissionService(
        allow=tuple(str(item) for item in _sequence(permission_values.get("allow"))),
        deny=tuple(str(item) for item in _sequence(permission_values.get("deny"))),
        # 后台主动回合永远不能继承全局 bypass；中高风险必须形成审批。
        bypass_approval=False,
        auto_allow_low_risk=parse_bool(
            permission_values.get("auto_allow_low_risk", True),
            field_name="tools.permissions.auto_allow_low_risk",
        ),
        approval_ttl_seconds=float(permission_values.get("approval_ttl_seconds", 90.0)),
    )
    proactive_tools = ToolExecutionService(
        registry,
        proactive_permissions,
        execution_record_sink=router.record_tool_execution,
    )
    # 上面的动作闭包在运行时才调用 ``tools``，此处赋值后即可安全使用。

    behavior_values = _mapping(values.get("behavior"))
    movement_values = _mapping(behavior_values.get("movement"))
    behavior_actions: list[BehaviorAction] = []
    for raw_action in _sequence(behavior_values.get("actions")):
        item = _mapping(raw_action)
        identity = str(item.get("identity", "")).strip()
        arguments = item.get("arguments", {})
        if identity and isinstance(arguments, Mapping):
            behavior_actions.append(
                BehaviorAction(
                    identity,
                    dict(arguments),
                    float(item.get("weight", 1.0)),
                    float(item.get("duration_seconds", 0.8)),
                )
            )
    motion_generation_provider = getattr(pet_value, "motion_owner_generation", None)
    if not callable(motion_generation_provider):
        motion_generation_provider = None
    behavior = BehaviorService(
        tools,
        actions=tuple(behavior_actions),
        interval_seconds=(
            float(behavior_values.get("min_interval_seconds", 8.0)),
            float(behavior_values.get("max_interval_seconds", 20.0)),
        ),
        probability=float(behavior_values.get("probability", 0.35)),
        interaction_cooldown_seconds=float(
            behavior_values.get("interaction_cooldown_seconds", 0.35)
        ),
        movement=AutonomousMovementConfig(
            enabled=parse_bool(
                movement_values.get("enabled", False),
                field_name="behavior.movement.enabled",
                default=False,
            ),
            min_distance_px=float(movement_values.get("min_distance_px", 80.0)),
            max_distance_px=float(movement_values.get("max_distance_px", 280.0)),
            step_pixels=float(movement_values.get("step_pixels", 36.0)),
            step_interval_seconds=float(movement_values.get("step_interval_seconds", 0.08)),
            max_speed_px_per_second=float(movement_values.get("max_speed_px_per_second", 240.0)),
            moving_motion=str(movement_values.get("moving_motion", "walk") or ""),
            idle_motion=str(movement_values.get("idle_motion", "idle") or ""),
            movement_identity=str(
                movement_values.get("movement_identity", "pet:autonomous_move")
                or "pet:autonomous_move"
            ),
            flip_facing=parse_bool(
                movement_values.get("flip_facing", True),
                field_name="behavior.movement.flip_facing",
                default=True,
            ),
            max_steps=int(movement_values.get("max_steps", 64)),
        ),
        position_provider=pet_value.position,
        bounds_provider=pet_value.movement_bounds,
        interaction_provider=pet_value.is_user_interacting,
        motion_generation_provider=motion_generation_provider,
        direction_setter=pet_value.set_direction,
    )
    # 控制台/配置向导打开时由 MutablePetController 同步使当前行为代际
    # 失效，避免等待平台回执的旧游走步进迟到覆盖用户界面。保留鸭子类型
    # 兼容性，外部宿主提供的旧控制器若没有该增强接口仍可启动。
    set_interaction_interrupt = getattr(pet_value, "set_interaction_interrupt", None)
    if callable(set_interaction_interrupt):
        set_interaction_interrupt(behavior.interrupt)
    set_activity_notifier = getattr(pet_value, "set_activity_notifier", None)
    if callable(set_activity_notifier):
        set_activity_notifier(lambda: activity.record_interaction("pet"))
    scheduler_values = _mapping(values.get("scheduler"))
    for raw_task in _sequence(scheduler_values.get("tasks")):
        item = _mapping(raw_task)
        try:
            scheduler.upsert(
                task_id=str(item.get("task_id", "")).strip() or None,
                name=str(item["name"]),
                expression=str(item["expression"]),
                action=dict(item["action"]),
                owner=str(item.get("owner", "config")),
                metadata=dict(item.get("metadata", {}))
                if isinstance(item.get("metadata", {}), Mapping)
                else item.get("metadata"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError("invalid scheduler or trigger configuration") from exc
    for raw_trigger in _sequence(scheduler_values.get("triggers")):
        item = _mapping(raw_trigger)
        try:
            triggers.register(
                trigger_id=str(item["trigger_id"]),
                event_name=str(item["event_name"]),
                action=dict(item["action"]),
                owner=str(item.get("owner", "config")),
                debounce_seconds=float(item.get("debounce_seconds", 0.0)),
                metadata=_trigger_metadata(item),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError("invalid scheduler or trigger configuration") from exc

    # 所有不依赖持久化的配置解析和对象装配完成后再打开 SQLite；配置错误不会
    # 创建空数据库文件或留下半初始化连接。
    database = Database(database_path)
    diary = PetDiaryService(database)

    # 运行时组合根是完整宿主，使用与会话相同的 SQLite 生命周期保存调度状态。
    # 配置条目在打开数据库前已经完成校验和装配；先记住它们的 ID，再恢复
    # 旧快照，使显式 YAML 配置优先，同时清理已经删除的旧 ``owner=config`` 项。
    configured_task_ids = set(getattr(scheduler, "_tasks", {}))
    configured_trigger_ids = set(getattr(triggers, "_triggers", {}))
    scheduler_repository: SchedulerRepository | None = None
    try:
        scheduler_repository = SchedulerRepository(database)
        scheduler.attach_state_store(scheduler_repository)
        triggers.attach_state_store(scheduler_repository)
        for task_id, task in tuple(getattr(scheduler, "_tasks", {}).items()):
            if getattr(task, "owner", "") == "config" and task_id not in configured_task_ids:
                scheduler.remove(task_id, owner="config")
        for task_id, task in tuple(getattr(triggers, "_triggers", {}).items()):
            if getattr(task, "owner", "") == "config" and task_id not in configured_trigger_ids:
                triggers.remove(task_id, owner="config")
        # ``upsert``/``register`` 已经在内存中完成；显式回写完整快照，确保
        # 新配置和无 ID 的配置项也能在下一次启动时恢复。
        for task in tuple(getattr(scheduler, "_tasks", {}).values()):
            scheduler._persist_task(task)
        for trigger in tuple(getattr(triggers, "_triggers", {}).values()):
            triggers._persist_trigger(trigger)
    except (OSError, RuntimeError, TypeError, ValueError, sqlite3.Error) as exc:
        # 调度持久化是增强能力；数据库本身已可用于会话/记忆时，仓储 DDL
        # 或历史快照异常不应阻断桌宠启动，服务会继续报告 detached 状态。
        scheduler_repository = None
        logger.warning("调度持久化不可用，继续以内存模式运行: %s", type(exc).__name__)
    api_audit = ApiCallAuditRepository(database)
    router.set_audit_repository(api_audit)
    persona_prompts = PersonaPromptBundle.from_configuration(values)
    memory = MemoryService(
        database,
        settings=MemorySettings.from_mapping(
            _mapping(values.get("memory")),
            base_directory=configuration.directory,
        ),
    )
    affection = AffectionService(database)
    configured_tool_groups = (
        None
        if "active_groups" not in tools_values or tools_values.get("active_groups") is None
        else tuple(str(item) for item in _sequence(tools_values.get("active_groups")))
    )
    try:
        conversation = ConversationService(
            router,
            memory=memory,
            affection=affection,
            tools=tools,
            tts=tts,
            repository=ConversationRepository(database),
            event_sink=runtime_event_sink,
            system_prompt=persona_prompts.dialogue_system_prompt(),
            model=str(_mapping(values.get("llm")).get("default_model", "")) or None,
            # 配置未声明 ``active_groups`` 时沿用兼容语义：模型可见所有公开工具；
            # 显式写入空列表表示用户关闭了全部工具组，不能再用 ``or None``
            # 把它误解释成未配置并重新暴露全部工具。
            tool_groups=configured_tool_groups,
            presentation=presentation,
            tts_language=str(_mapping(values.get("tts")).get("language", "zh")),
            tts_role=str(
                _mapping(values.get("tts")).get("role", "")
                or _mapping(_mapping(values.get("tts")).get("routing")).get("default_role", "")
                or ""
            ),
        )
        proactive_settings = ProactiveSettings.from_mapping(_mapping(values.get("proactive")))
        proactive_conversation = ConversationService(
            router,
            memory=None,
            affection=None,
            tools=proactive_tools,
            tts=tts,
            repository=None,
            event_sink=proactive_event_sink,
            system_prompt=(
                persona_prompts.dialogue_system_prompt()
                + "\n\n【主动行为】你由后台事件唤醒；只依据脱敏事件和记忆决定是否回应。"
            ),
            task="proactive",
            model=None,
            tool_groups=configured_tool_groups,
            max_tool_rounds=proactive_settings.max_tool_rounds,
            presentation=presentation,
            tts_language=str(_mapping(values.get("tts")).get("language", "zh")),
            tts_role=str(
                _mapping(values.get("tts")).get("role", "")
                or _mapping(_mapping(values.get("tts")).get("routing")).get("default_role", "")
                or ""
            ),
        )
    except BaseException:
        database.close()
        raise
    runtime = ApplicationRuntime(
        configuration,
        inventory,
        database,
        memory,
        affection,
        router,
        registry,
        permissions,
        tools,
        scheduler,
        triggers,
        watcher,
        behavior,
        tts,
        asr,
        conversation,
        pet_value,
        platform_value,
        mcp_clients,
        mcp_bridges,
    )
    runtime.activity = activity
    runtime.api_audit = api_audit
    runtime.diary = diary
    runtime.interaction_state = interaction_state
    runtime.persona_prompts = persona_prompts
    runtime.proactive = ProactiveCoordinator(
        proactive_conversation,
        memory=memory,
        affection=affection,
        settings=proactive_settings,
        activity_provider=activity.is_user_active,
        gate_provider=lambda: {
            "closed": runtime._closed,
            "closing": runtime._closing,
            "dragging": bool(runtime.pet_controller.is_user_interacting()),
            "locked": bool(runtime.pet_controller.is_window_locked()),
            "dialogue_active": bool(runtime.conversation.has_active_conversation()),
        },
    )
    runtime.memory_summarizer = MemorySummaryCoordinator(
        memory,
        lambda: runtime.router,
    )
    runtime.memory_summarizer.configure_prompts(
        summary=persona_prompts.prompts.memory_summary,
        extraction=persona_prompts.prompts.memory_extract,
    )
    # 组合根完成后再绑定文件观察器，避免回调在 runtime 尚未拥有完整服务
    # 集合时提前执行。默认值只用于补齐省略字段，不会把旧用户配置复活。
    reload_values = _mapping(_mapping(values.get("config")).get("reload"))
    try:
        reload_interval = float(reload_values.get("interval_seconds", 0.5))
        reload_debounce = float(reload_values.get("debounce_seconds", 0.1))
        reload_checks = int(reload_values.get("stable_checks", 2))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConfigurationError("config.reload timing is invalid") from exc
    runtime.configuration_watcher = ConfigurationWatcher(
        configuration.path,
        runtime.reload_configuration,
        defaults=default_configuration_values(
            resource_root=inventory.root,
            database_path=database_path,
        ),
        environment=dict(os.environ),
        initial_configuration=configuration,
        interval_seconds=reload_interval,
        debounce_seconds=reload_debounce,
        stable_checks=reload_checks,
    )
    # 插件只获得组合根明确发布的服务。``configuration`` 是 plugins 区的
    # 独立副本，不包含已展开的模型密钥或其它全局配置内容；需要动态对象时
    # 可通过 runtime 读取当前代次，而不是持有会在热重载后失效的副本。
    runtime.plugins = PluginManager(
        settings=plugin_settings,
        configuration=_mapping(values.get("plugins")),
        services={
            "runtime": runtime,
            "configuration": dict(_mapping(values.get("plugins"))),
            "registry": registry,
            "tools": tools,
            "scheduler": scheduler,
            "triggers": triggers,
            "memory": memory,
            "diary": diary,
            "pet_controller": pet_value,
            "platform": platform_value,
            "activity": activity,
            "proactive": runtime.proactive,
            "configuration_watcher": runtime.configuration_watcher,
        },
        base_directory=configuration.directory,
    )
    runtime.modules.set_services(
        {
            "runtime": runtime,
            "configuration": configuration,
            "registry": registry,
            "tools": tools,
            "scheduler": scheduler,
            "memory": memory,
            "diary": diary,
            "pet_controller": pet_value,
            "platform": platform_value,
        }
    )
    runtime.modules.update_configuration(values)
    runtime._register_optional_modules()
    return runtime


__all__ = [
    "ApplicationRuntime",
    "DispatchingDesktopPlatform",
    "MutablePetController",
    "asr_capture_configuration",
    "build_runtime",
    "initialize_database",
    "validate_runtime_configuration",
]
