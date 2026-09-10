"""工具运行时的纯 Python 契约。"""

from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Protocol

_TOOL_USER_LABELS = {
    "desktop:observe_foreground": "查看前台窗口",
    "desktop:cursor_position": "读取光标位置",
    "desktop:list_processes": "查看运行中的程序",
    "desktop:context_snapshot": "查看当前桌面上下文",
    "desktop:capture_screen": "查看屏幕画面",
    "desktop:ocr": "识别屏幕文字",
    "desktop:click_at": "点击屏幕坐标",
    "desktop:automation_batch": "执行桌面自动化步骤",
    "pet:move": "移动桌宠",
    "pet:autonomous_move": "桌宠自主移动",
    "pet:set_expression": "切换表情",
    "pet:play_motion": "播放动作",
    "pet:speak": "让桌宠说话",
    "pet:diary_write": "写入桌宠日记",
    "pet:diary_recall": "回想桌宠日记",
    "pet:set_click_through": "切换点击穿透",
    "scheduler:upsert": "设置定时任务",
    "system:run_command": "执行系统操作",
    "system:module_status": "检查模块状态",
    "agent:execute_plan": "执行多步计划",
}

# 行为服务通过工具管线调用桌宠动作时使用的内部上下文标志。该标志只
# 参与运行时调度，不会进入工具公开参数，避免随机行为重新触发自身中断。
BEHAVIOR_INTERNAL_METADATA_KEY = "behavior_internal"
MAX_TOOL_PLAN_STEPS = 8


def tool_user_label(identity: object) -> str:
    """把内部工具身份转换为不带参数的用户文案。"""

    normalized = str(identity or "").strip()
    if normalized in _TOOL_USER_LABELS:
        return _TOOL_USER_LABELS[normalized]
    namespace = normalized.split(":", 1)[0].strip().lower() if ":" in normalized else ""
    return {
        "desktop": "桌面查看",
        "pet": "桌宠操作",
        "scheduler": "定时任务",
        "system": "系统操作",
        "mcp": "外部工具",
    }.get(namespace, "桌面操作")


class RiskLevel(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class ToolKind(StrEnum):
    """统一描述本地、系统、MCP 和扩展工具来源。"""

    LOCAL = "local"
    SYSTEM = "system"
    MCP = "mcp"
    EXTENSION = "extension"


class ToolHandler(Protocol):
    def __call__(
        self, arguments: Mapping[str, Any], context: ToolCallContext
    ) -> Any | Awaitable[Any]: ...


@dataclass(frozen=True)
class ToolCallContext:
    profile_id: str
    session_id: str
    turn_id: str
    source: str = "model"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        control_codes = (0, 10, 13)
        for field_name in ("profile_id", "session_id", "turn_id"):
            raw_value = str(getattr(self, field_name) or "")
            if any(ord(char) in control_codes for char in raw_value):
                raise ValueError(f"{field_name} is invalid")
            value = raw_value.strip()
            if not value or len(value) > 256:
                raise ValueError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, value)
        raw_source = str(self.source or "model")
        if any(ord(char) in control_codes for char in raw_source):
            raise ValueError("source is invalid")
        source = raw_source.strip() or "model"
        if len(source) > 256:
            raise ValueError("source is invalid")
        object.__setattr__(self, "source", source)
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class ToolSpec:
    identity: str
    description: str
    parameters: Mapping[str, Any]
    handler: ToolHandler
    risk: RiskLevel = RiskLevel.MEDIUM
    group: str = "default"
    public: bool = True
    read_only: bool = False
    display_name: str = ""
    kind: ToolKind = ToolKind.LOCAL

    def __post_init__(self) -> None:
        identity = str(self.identity or "").strip()
        if ":" not in identity or any(ord(char) in (0, 10, 13) for char in identity):
            raise ValueError("tool identity must use namespace:name")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        parameters = dict(self.parameters or {})
        if parameters.get("type", "object") != "object":
            raise ValueError("tool parameters must be an object schema")
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "description", str(self.description or "").strip())
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "risk", RiskLevel(self.risk))
        object.__setattr__(self, "group", str(self.group or "default").strip() or "default")
        object.__setattr__(self, "public", bool(self.public))
        object.__setattr__(self, "read_only", bool(self.read_only))
        display_name = str(self.display_name or "").strip()
        if len(display_name) > 80 or any(ord(char) in (0, 10, 13) for char in display_name):
            raise ValueError("tool display_name is invalid")
        object.__setattr__(self, "display_name", display_name)
        object.__setattr__(self, "kind", ToolKind(self.kind))

    @property
    def user_label(self) -> str:
        """返回 UI 可见的稳定标签；不会包含工具参数或内部身份。"""

        return self.display_name or tool_user_label(self.identity)

    def visible_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.identity,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class ToolPlanStep:
    """一个工具计划步骤及其显式依赖。"""

    step_id: str
    call_id: str
    identity: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    timeout_seconds: float = 15.0
    max_attempts: int = 1
    on_error: str = "skip_dependents"

    def __post_init__(self) -> None:
        for field_name in ("step_id", "call_id", "identity"):
            value = str(getattr(self, field_name) or "").strip()
            if not value or len(value) > 256 or any(character in value for character in "\r\n\x00"):
                raise ValueError(f"{field_name} is invalid")
            object.__setattr__(self, field_name, value)
        if ":" not in self.identity:
            raise ValueError("tool identity must use namespace:name")
        if not isinstance(self.arguments, Mapping):
            raise ValueError("tool arguments must be a mapping")
        object.__setattr__(self, "arguments", dict(self.arguments))
        if isinstance(self.depends_on, (str, bytes, bytearray)):
            raise ValueError("depends_on must be a sequence")
        try:
            raw_dependencies = tuple(self.depends_on)
        except TypeError as exc:
            raise ValueError("depends_on must be a sequence") from exc
        dependencies: list[str] = []
        for dependency in raw_dependencies:
            value = str(dependency or "").strip()
            if not value or len(value) > 256 or any(character in value for character in "\r\n\x00"):
                raise ValueError("dependency step_id is invalid")
            dependencies.append(value)
        object.__setattr__(self, "depends_on", tuple(dependencies))
        raw_timeout = self.timeout_seconds
        if isinstance(raw_timeout, bool) or not isinstance(raw_timeout, (int, float)):
            raise ValueError("timeout_seconds must be a number")
        timeout_seconds = float(raw_timeout)
        if not math.isfinite(timeout_seconds) or not 0.05 <= timeout_seconds <= 60.0:
            raise ValueError("timeout_seconds is outside the allowed range")
        raw_attempts = self.max_attempts
        if isinstance(raw_attempts, bool) or not isinstance(raw_attempts, int):
            raise ValueError("max_attempts must be an integer")
        if not 1 <= raw_attempts <= 3:
            raise ValueError("max_attempts is outside the allowed range")
        on_error = str(self.on_error or "").strip()
        if on_error not in {"stop", "skip_dependents"}:
            raise ValueError("on_error is unsupported")
        object.__setattr__(self, "timeout_seconds", timeout_seconds)
        object.__setattr__(self, "max_attempts", raw_attempts)
        object.__setattr__(self, "on_error", on_error)


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    call_id: str
    identity: str
    session_id: str
    safe_summary: str
    expires_at: float
    profile_id: str = ""
    display_name: str = ""


@dataclass(frozen=True)
class ToolValidationError:
    path: str
    message: str


def is_awaitable(value: object) -> bool:
    return inspect.isawaitable(value)
