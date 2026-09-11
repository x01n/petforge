from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from importlib.resources import files
from math import isfinite
from typing import Any

from core.util import as_sequence as _sequence
from gui.qt6.md3 import DARK_MD3_THEME, build_md3_web_css, md3_web_tokens

_PUBLIC_PAYLOAD_KEYS = frozenset(
    {
        "target",
        "mode",
        "direction",
        "channel_id",
        "model",
        "profile",
        "language",
        "backend",
    }
)
_PUBLIC_ACTION_KINDS = frozenset(
    {
        "open_config",
        "restart_application",
        "select_renderer_backend",
        "configure_model",
        "select_model_channel",
        "select_tts_profile",
        "select_tts_language",
        "expression_request",
        "motion_request",
        "stop",
        "approve",
        "deny",
        "grant_session",
        "retry",
    }
)
_PUBLIC_SAFE_FALLBACK = "敏感详情已隐藏"
_PUBLIC_SECRET_PATTERN = re.compile(
    r"(?i)(?:api[\s_-]?key|access[\s_-]?key|private[\s_-]?key|credential|"
    r"authorization|bearer|token|secret|password|cookie|"
    r"command[\s_-]?line|arguments?)\b"
)
_PUBLIC_INTERNAL_PATTERN = re.compile(r"(?i)\b(?:pet|desktop|system|mcp|network):[^\s]+")
_PUBLIC_PARAMETER_PATTERN = re.compile(r"(?i)\b(?:param|parameter)[a-z0-9_:-]*\b")
_PUBLIC_INTERNAL_PROTOCOL_PATTERN = re.compile(
    r"(?i)\b(?:status|state|detail|zone|identity|payload|operation[_-]?id|call[_-]?id)\s*[=:]"
)
_PUBLIC_CAPABILITY_TOKEN = re.compile(r"^cap-(?:expression|motion)-\d{1,2}$")
_PUBLIC_THEME_TOKEN = re.compile(r"^--md3-[a-z0-9-]{1,80}$")
_PUBLIC_THEME_VALUE = re.compile(r"^[#A-Za-z0-9 ,.()'\"_-]{1,128}$")
_PUBLIC_THEME_KEYS = frozenset((*md3_web_tokens(DARK_MD3_THEME), "--md3-seed"))
_RENDERER_LABELS = {
    "auto": "自动选择",
    "opengl": "OpenGL Live2D",
    "web_live2d": "Web Live2D",
    "sprite": "精灵回退",
    "vulkan": "Vulkan 渲染",
    "vllank": "Vulkan 渲染",
    "vllakn": "Vulkan 渲染",
    "unavailable": "不可用",
}
_CAPABILITY_LABELS = {
    "neutral": "自然",
    "happy": "开心",
    "sad": "难过",
    "curious": "好奇",
    "surprised": "惊讶",
    "shy": "害羞",
    "idle": "待机",
    "blink": "眨眼",
    "wave": "挥手",
    "walk": "走动",
    "angry": "生气",
    "tap": "轻触",
}
_PHASE_LABELS = {
    "idle": "等待互动",
    "streaming": "正在回复",
    "tool_running": "正在执行桌面操作",
    "approval_required": "等待你的确认",
    "completed": "本轮已完成",
    "failed": "本轮未完成",
    "configuration_required": "需要连接模型",
}
_STATUS_LABELS = {
    "idle": "等待处理",
    "requested": "已提交",
    "pending": "等待处理",
    "started": "已开始",
    "accepted": "已接受",
    "completed": "已完成",
    "available": "已就绪",
    "updated": "已更新",
    "saved": "已保存",
    "validated": "已校验",
    "reloaded": "已刷新",
    "approval_required": "等待确认",
    "tool_limit": "需要确认",
    "degraded": "降级运行",
    "failed": "执行失败",
    "error": "暂时不可用",
    "unavailable": "暂时不可用",
    "cancelled": "已停止",
    "canceled": "已停止",
    "denied": "已拒绝",
    "rejected": "已拒绝",
    "timeout": "响应超时",
    "restart_required": "需要重启",
    "partial": "部分应用",
    "running": "运行中",
    "stopped": "已停止",
    "deferred": "等待稳定",
    "configuration_rejected": "配置未应用",
}
_OPERATION_LABELS = {
    "open_config": "打开配置中心",
    "select_renderer_backend": "切换渲染引擎",
    "configure_model": "连接模型",
    "select_model_channel": "切换模型渠道",
    "select_tts_profile": "切换语音",
    "select_tts_language": "切换输出语言",
    "submit_text": "发送消息",
    "stop": "停止回复",
    "retry": "重试回复",
    "approve": "允许桌面操作",
    "grant_session": "允许本次会话",
    "deny": "拒绝桌面操作",
    "pet_part": "部位互动",
    "expression": "切换表情",
    "motion": "播放动作",
    "expression_request": "表情编排",
    "motion_request": "动作编排",
    "show_pet": "显示桌宠",
    "toggle_visibility": "切换显示",
    "center_pet": "居中桌宠",
    "nudge_pet": "移动桌宠",
    "set_display_size": "调整大小",
    "toggle_window_lock": "切换锁定",
    "toggle_always_on_top": "切换置顶",
    "toggle_click_through": "切换点击穿透",
    "restore_click_through": "恢复点击",
    "restart_application": "重启桌宠",
    "read_foreground_window": "查看前台窗口",
    "read_processes": "查看运行中的程序",
    "点击互动": "部位互动",
    "approval": "桌面审批",
    "审批": "桌面审批",
    "前台窗口": "查看前台窗口",
    "进程列表": "查看运行中的程序",
}
_PUBLIC_PART_IDS = frozenset({"head", "body", "lower_left", "lower_right"})
_FRIENDLY_PHASES = frozenset(_PHASE_LABELS.values())
_FRIENDLY_OPERATION_LABELS = frozenset(_OPERATION_LABELS.values())
_CHANNEL_STATUS_LABELS = {
    "disabled": "已停用",
    "unconfigured": "未配置",
    "cooldown": "暂时冷却",
    "ready": "已就绪",
    "recovered": "连接已恢复",
    "unavailable": "未就绪",
}
_FRIENDLY_CHANNEL_STATUSES = frozenset(_CHANNEL_STATUS_LABELS.values())
_PUBLIC_STREAM_STATUS_LABELS = {
    "approval_required": "等待确认",
    "tool_running": "正在执行",
    "tool_calls": "正在处理",
    "streaming": "正在回复",
    "completed": "已完成",
    "failed": "未完成",
    "denied": "已拒绝",
    "rejected": "已拒绝",
    "configuration": "需要配置",
    "configuration_required": "需要连接模型",
    "authentication": "连接需要密钥",
    "authorization": "连接权限不足",
    "network": "网络暂时不可用",
    "timeout": "响应超时",
    "cancelled": "已停止",
    "canceled": "已停止",
    "idle": "等待处理",
    "pending": "等待处理",
    "requested": "已提交",
    "started": "已开始",
    "accepted": "已接受",
    "available": "已就绪",
    "updated": "已更新",
    "saved": "已保存",
    "validated": "已校验",
    "reloaded": "已刷新",
    "degraded": "降级运行",
    "error": "暂时不可用",
    "unavailable": "暂时不可用",
    "restart_required": "需要重启",
    "connected": "已连接",
    "disconnected": "未连接",
    "local": "本地",
    "speaking": "正在播放",
    "disabled": "已停用",
}
_PUBLIC_STREAM_STATUS_PATTERN = re.compile(r"\[([a-z][a-z0-9_-]{0,48})\]\s*", re.IGNORECASE)
_PUBLIC_STREAM_PREFIX_PATTERN = re.compile(
    r"(?im)(^|\n)[ \t]*(approval_required|tool_running|tool_calls|streaming|"
    r"completed|failed|denied|rejected|configuration|configuration_required|"
    r"authentication|network|timeout|cancelled|canceled|idle|pending|requested|"
    r"started|accepted|available|updated|saved|validated|reloaded|degraded|"
    r"error|unavailable|restart_required|connected|disconnected|local|speaking|"
    r"disabled)[ \t]*[:：][ \t]*"
)
_PUBLIC_STREAM_SENSITIVE_KEYS = frozenset(
    {"authorization", "token", "cookie", "credential", "access_key", "private_key"}
)
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
    "ollama": "Ollama",
    "ollama_chat": "Ollama",
}
_DEFAULT_PUBLIC_CAPABILITIES = {
    "expression": ("neutral", "happy", "sad", "curious", "surprised", "shy"),
    "motion": ("idle", "blink", "wave", "walk"),
}
_DEFAULT_PARTS: tuple[dict[str, object], ...] = (
    {"id": "head", "label": "摸摸头", "description": "猫猫头反馈"},
    {"id": "body", "label": "拍拍身体", "description": "身体反馈"},
    {"id": "lower_left", "label": "左边", "description": "左侧反馈"},
    {"id": "lower_right", "label": "右边", "description": "右侧反馈"},
)


def safe_public_text(value: object, *, limit: int = 240) -> str:
    """生成不会把敏感字段或原始参数带入公开控制面的文本。"""

    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple, set, frozenset, bytes, bytearray)):
        return _PUBLIC_SAFE_FALLBACK
    text = str(value).replace("\x00", " ")
    text = " ".join(text.split())
    if not text:
        return ""
    if (
        _PUBLIC_SECRET_PATTERN.search(text)
        or _PUBLIC_INTERNAL_PATTERN.search(text)
        or _PUBLIC_PARAMETER_PATTERN.search(text)
        or _PUBLIC_INTERNAL_PROTOCOL_PATTERN.search(text)
    ):
        return _PUBLIC_SAFE_FALLBACK
    return text[:limit]


def friendly_public_text(
    value: object,
    *,
    limit: int = 240,
    preserve_newlines: bool = False,
) -> str:
    """把公开文本中的状态标记翻译为用户文案并保留脱敏边界。

    核心状态仍保留机器可读的 ``status`` 字段供页面逻辑使用；正文、碎碎念、
    工具提示和 HTTP 回执则必须在展示边界翻译，避免把 ``[approval_required]``
    等内部协议直接显示给用户。复杂对象一律返回统一占位符。
    """

    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple, set, frozenset, bytes, bytearray)):
        return _PUBLIC_SAFE_FALLBACK
    text = str(value)
    if preserve_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(char for char in text if char in {"\n", "\t"} or ord(char) >= 0x20)
        text = text.strip()
    else:
        text = " ".join(text.split())
    if not text:
        return ""
    # 已知的方括号状态标记可以安全翻译；其它标记（尤其 token/cookie）仍
    # 保留到安全检查中，避免把疑似凭据伪装成普通状态。
    security_text = _PUBLIC_STREAM_STATUS_PATTERN.sub(
        lambda match: (
            ""
            if match.group(1).strip().casefold() in _PUBLIC_STREAM_STATUS_LABELS
            and match.group(1).strip().casefold() not in _PUBLIC_STREAM_SENSITIVE_KEYS
            else match.group(0)
        ),
        text,
    )
    if (
        _PUBLIC_SECRET_PATTERN.search(security_text)
        or _PUBLIC_INTERNAL_PATTERN.search(security_text)
        or _PUBLIC_PARAMETER_PATTERN.search(security_text)
        or _PUBLIC_INTERNAL_PROTOCOL_PATTERN.search(security_text)
    ):
        return _PUBLIC_SAFE_FALLBACK

    normalized = text.casefold()
    exact_label = (
        _PUBLIC_STREAM_STATUS_LABELS.get(normalized)
        if normalized not in _PUBLIC_STREAM_SENSITIVE_KEYS
        else None
    )
    if exact_label:
        return exact_label

    def replace_status(match: re.Match[str]) -> str:
        key = match.group(1).strip().casefold()
        label = _PUBLIC_STREAM_STATUS_LABELS.get(key)
        return f"{label}：" if label else "状态："

    text = _PUBLIC_STREAM_STATUS_PATTERN.sub(replace_status, text)

    def replace_prefix(match: re.Match[str]) -> str:
        label = _PUBLIC_STREAM_STATUS_LABELS[match.group(2).casefold()]
        return f"{match.group(1)}{label}："

    text = _PUBLIC_STREAM_PREFIX_PATTERN.sub(replace_prefix, text)
    # 工具摘要可能已带中文前缀；状态标记翻译后只保留一层。
    for label in set(_PUBLIC_STREAM_STATUS_LABELS.values()):
        text = text.replace(f"{label}：{label}：", f"{label}：")
    if len(text) > limit:
        text = text[: max(0, limit - 1)] + "…"
    return text


def _text(value: object, *, limit: int = 240) -> str:
    return safe_public_text(value, limit=limit)


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default


def _renderer_label(value: object, *, available: bool) -> str:
    """把内部后端键转换为点击式界面可见的友好名称。"""

    normalized = _text(value, limit=48).lower()
    if normalized == "vllank" and available:
        return "三维渲染"
    return _RENDERER_LABELS.get(normalized, "未知后端")


def _phase_label(value: object) -> str:
    normalized = str(getattr(value, "value", value) or "").strip().lower()
    if str(value or "").strip() in _FRIENDLY_PHASES:
        return str(value).strip()
    return _PHASE_LABELS.get(normalized, "等待互动")


def _status_key(value: object) -> str:
    return str(getattr(value, "value", value) or "").strip().lower()


def _status_label(value: object) -> str:
    normalized = _status_key(value)
    return _STATUS_LABELS.get(normalized, "处理中")


def _operation_label(value: object) -> str:
    normalized = _status_key(value)
    if str(value or "").strip() in _FRIENDLY_OPERATION_LABELS:
        return str(value).strip()
    return _OPERATION_LABELS.get(normalized, "桌宠操作")


def _safe_actions(value: object) -> list[dict[str, object]]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return []
    try:
        rows = tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return []
    result: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        kind = _text(row.get("kind"), limit=48).lower()
        label = _text(row.get("label"), limit=80)
        if not kind or kind == _PUBLIC_SAFE_FALLBACK.casefold() or kind not in _PUBLIC_ACTION_KINDS:
            continue
        if not label:
            continue
        if label == _PUBLIC_SAFE_FALLBACK:
            label = "操作"
        payload = row.get("payload")
        safe_payload: dict[str, object] = {}
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if str(key) not in _PUBLIC_PAYLOAD_KEYS:
                    continue
                if isinstance(value, str):
                    safe_value = safe_public_text(value, limit=120)
                    if safe_value == _PUBLIC_SAFE_FALLBACK:
                        continue
                    safe_payload[str(key)] = safe_value
                elif isinstance(value, (int, float, bool)):
                    safe_payload[str(key)] = value
        result.append(
            {
                "kind": kind,
                "label": label,
                "enabled": bool(row.get("enabled", True)),
                "payload": safe_payload,
            }
        )
    return result


def _safe_approval(value: object) -> dict[str, object] | None:
    """只投影用户能理解的审批摘要，不复制内部调用标识。"""

    if not isinstance(value, Mapping):
        return None
    try:
        expires_at = float(value.get("expires_at", 0) or 0)
    except (TypeError, ValueError):
        expires_at = 0.0
    # 公开界面不需要绝对时间戳；同时对旧/测试数据的错误时钟域做上限
    # 保护，避免出现数十亿秒倒计时。权限服务本身仍在后端按真实 TTL 校验。
    remaining = max(0.0, min(3600.0, expires_at - time.time()))
    display_name = _text(value.get("display_name"), limit=80)
    if display_name == _PUBLIC_SAFE_FALLBACK or _PUBLIC_INTERNAL_PATTERN.search(display_name):
        display_name = "桌面操作"
    return {
        "display_name": display_name or "工具操作",
        "safe_summary": _text(value.get("safe_summary"), limit=400),
        "expires_at": time.time() + remaining,
        "remaining_seconds": int(remaining),
    }


def _safe_approvals(value: object) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in _sequence(value):
        safe = _safe_approval(item)
        if safe is not None:
            result.append(safe)
    return result[:8]


def _safe_parts(value: object) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for item in _sequence(value):
        if not isinstance(item, Mapping):
            continue
        identifier = _text(item.get("id"), limit=48).lower()
        label = _text(item.get("label"), limit=80)
        # 部位是固定的用户交互区域；不把渲染器原生命名或自定义参数带到页面。
        if identifier not in _PUBLIC_PART_IDS:
            continue
        if not label or label == _PUBLIC_SAFE_FALLBACK:
            label = next(
                (
                    str(default["label"])
                    for default in _DEFAULT_PARTS
                    if default["id"] == identifier
                ),
                "互动",
            )
        result.append(
            {
                "id": identifier,
                "label": label,
                "description": _text(item.get("description"), limit=120),
                "enabled": bool(item.get("enabled", True)),
            }
        )
    return result[:16] or [{**dict(item), "enabled": True} for item in _DEFAULT_PARTS]


def _safe_position(value: object) -> dict[str, int] | None:
    if not isinstance(value, Mapping) or "x" not in value or "y" not in value:
        return None
    try:
        x = float(value["x"])
        y = float(value["y"])
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(x) or not isfinite(y):
        return None
    return {"x": int(round(x)), "y": int(round(y))}


def _safe_feedback(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    affection = value.get("affection")
    affection = affection if isinstance(affection, Mapping) else {}
    part = _text(value.get("part"), limit=48).lower()
    zone = _text(value.get("zone"), limit=48).lower()
    if part not in _PUBLIC_PART_IDS:
        part = ""
    if zone not in _PUBLIC_PART_IDS:
        zone = ""
    expression = _text(value.get("expression"), limit=48)
    motion = _text(value.get("motion"), limit=48)
    return {
        "part": part,
        "zone": zone,
        "phrase": friendly_public_text(value.get("phrase"), limit=400),
        "mood": _capability_label(value.get("mood"), "expression", 0)
        if _text(value.get("mood"), limit=32)
        else "",
        "expression": _capability_label(expression, "expression", 0) if expression else "",
        "motion": _capability_label(motion, "motion", 0) if motion else "",
        "active": bool(value.get("active", False)),
        "affection": {
            "current": _text(affection.get("current"), limit=32),
            "applied": _text(affection.get("applied"), limit=32),
            "tier": _text(affection.get("tier"), limit=48),
            "status": friendly_public_text(affection.get("status"), limit=32),
        },
    }


def _capability_label(value: object, kind: str, index: int) -> str:
    """把渲染器内部能力名称转换为普通用户能理解的按钮文案。"""

    normalized = safe_public_text(value, limit=64).casefold()
    if normalized in {item.casefold() for item in _CAPABILITY_LABELS.values()}:
        return safe_public_text(value, limit=64)
    # 某些渲染器会把枚举序列化为 ``Expression.HAPPY``；只取最后一段
    # 仍然不会把原始参数名送入页面。
    for separator in (".", "::", "/", "\\"):
        if separator in normalized:
            normalized = normalized.rsplit(separator, 1)[-1]
    label = _CAPABILITY_LABELS.get(normalized)
    if label:
        return label
    for prefix, friendly in _CAPABILITY_LABELS.items():
        if normalized.startswith(prefix + "_") or normalized.startswith(prefix + "-"):
            return friendly
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            return friendly
    if len(normalized) <= 3 and normalized.isascii() and normalized.isalnum():
        return f"表情 {normalized}" if kind == "expression" else f"动作 {normalized}"
    prefix = "表情" if kind == "expression" else "动作"
    return f"{prefix} {index + 1}"


def _safe_capabilities(value: object, kind: str) -> list[dict[str, str]]:
    """以不携带原始名称的短令牌公开渲染能力。"""

    result: list[dict[str, str]] = []
    for index, item in enumerate(_sequence(value)):
        if isinstance(item, Mapping):
            identifier = safe_public_text(item.get("id"), limit=64)
            label = safe_public_text(item.get("label"), limit=80)
            if (
                _PUBLIC_CAPABILITY_TOKEN.fullmatch(identifier)
                and identifier.startswith(f"cap-{kind}-")
                and label
                and label != _PUBLIC_SAFE_FALLBACK
            ):
                result.append({"id": identifier, "label": label})
            continue
        if item is None:
            continue
        raw = str(item).strip()
        if not raw:
            continue
        result.append(
            {
                "id": f"cap-{kind}-{index}",
                "label": _capability_label(raw, kind, index),
            }
        )
    if result:
        return result[:32]
    # 渲染器在页面首帧或正式 exp3/motion3 尚未声明时仍必须给出可点按
    # 的安全入口。Qt 控制台已有同样回退；Web 端若返回空列表，用户会
    # 看到“动作”标题却没有任何按钮，无法确认模型是否能互动。
    fallback = _DEFAULT_PUBLIC_CAPABILITIES.get(kind, ())
    return [
        {
            "id": f"cap-{kind}-{index}",
            "label": str(_CAPABILITY_LABELS.get(str(name), str(name))),
        }
        for index, name in enumerate(fallback)
    ]


def _safe_operation(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {
            "status": "idle",
            "status_label": _status_label("idle"),
            "label": "",
            "message": "",
            "updated_at": 0.0,
        }
    try:
        updated_at = float(value.get("updated_at", 0) or 0)
    except (TypeError, ValueError):
        updated_at = 0.0
    status = _status_key(value.get("status")) or "idle"
    return {
        # id/kind 只用于宿主内部关联，绝不能进入公开状态。
        "status": status if status in _STATUS_LABELS else "unavailable",
        "status_label": _status_label(status),
        "label": _operation_label(value.get("kind", value.get("label"))),
        "message": friendly_public_text(value.get("message", value.get("reason")), limit=400),
        "updated_at": updated_at,
    }


def _safe_timeline(value: object) -> list[dict[str, object]]:
    """过滤活动记录，只保留用户可读的状态、标签和短消息。"""

    result: list[dict[str, object]] = []
    for item in _sequence(value):
        if not isinstance(item, Mapping):
            continue
        status = _status_key(item.get("status", item.get("state", "completed")))
        try:
            updated_at = float(item.get("updated_at", item.get("at", 0)) or 0)
        except (TypeError, ValueError, OverflowError):
            updated_at = 0.0
        result.append(
            {
                "label": _operation_label(item.get("kind", item.get("label"))),
                "status": status if status in _STATUS_LABELS else "unavailable",
                "status_label": _status_label(status),
                "message": friendly_public_text(item.get("message", item.get("reason")), limit=240),
                "updated_at": updated_at,
            }
        )
    return result[-12:]


def _safe_pid(value: object) -> int | None:
    """只接受展示用的非负整数进程号。"""

    try:
        pid = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return pid if 0 <= pid <= 2**31 - 1 else None


def _safe_observation(value: object) -> dict[str, object]:
    """投影桌面观察结果，只保留标题、进程名和 PID。"""

    if not isinstance(value, Mapping):
        return {"kind": "", "status": "idle", "status_label": "等待处理"}
    kind = _status_key(value.get("kind"))
    if kind not in {"foreground_window", "processes"}:
        kind = ""
    status = _status_key(value.get("status")) or "idle"
    if status not in _STATUS_LABELS:
        status = "unavailable"
    result: dict[str, object] = {
        "kind": kind,
        "label": "查看前台窗口"
        if kind == "foreground_window"
        else ("查看运行中的程序" if kind == "processes" else ""),
        "status": status,
        "status_label": _status_label(status),
        "message": friendly_public_text(value.get("message"), limit=240),
    }
    if kind == "foreground_window":
        result.update(
            {
                "title": _text(value.get("title"), limit=120),
                "process_name": _text(value.get("process_name", value.get("name")), limit=80),
                "pid": _safe_pid(value.get("pid")),
            }
        )
    elif kind == "processes":
        rows: list[dict[str, object]] = []
        raw_rows = value.get("processes")
        for row in _sequence(raw_rows):
            if not isinstance(row, Mapping):
                continue
            name = _text(row.get("name", row.get("process_name")), limit=80)
            pid = _safe_pid(row.get("pid"))
            if not name and pid is None:
                continue
            rows.append({"name": name or "未知程序", "pid": pid})
            if len(rows) >= 20:
                break
        result["processes"] = rows
    return result


def _safe_model_channels(value: object) -> list[dict[str, object]]:
    """投影渠道健康摘要，不复制地址、密钥、异常或调用标识。"""

    result: list[dict[str, object]] = []
    for item in _sequence(value):
        if not isinstance(item, Mapping):
            continue
        channel_id = _text(item.get("id"), limit=80)
        if not channel_id:
            continue
        protocol = _text(item.get("protocol"), limit=32).lower()
        protocol = _PUBLIC_CHANNEL_PROTOCOL_LABELS.get(protocol, "自定义渠道")
        model_configured = bool(item.get("model_configured", False))
        ready = bool(item.get("ready", False))
        try:
            failures = max(0, min(2**31 - 1, int(item.get("failures", 0) or 0)))
        except (TypeError, ValueError, OverflowError):
            failures = 0
        raw_status = _text(item.get("status"), limit=32)
        status_key = raw_status.casefold()
        if status_key in _CHANNEL_STATUS_LABELS:
            status = _CHANNEL_STATUS_LABELS[status_key]
        elif raw_status in _FRIENDLY_CHANNEL_STATUSES:
            status = raw_status
        else:
            status = "已就绪" if ready else ("未配置" if not model_configured else "未就绪")
        result.append(
            {
                "id": channel_id,
                "protocol": protocol,
                "model": _text(item.get("model"), limit=128),
                "models": [
                    model
                    for model in (
                        _text(model_value, limit=128)
                        for model_value in _sequence(item.get("models"))
                    )
                    if model
                ][:32],
                "model_configured": model_configured,
                "ready": ready,
                "failures": failures,
                "active": bool(item.get("active", False)),
                "selectable": bool(item.get("selectable", ready)),
                "status": status,
            }
        )
    return result[:32]


def _safe_model_image_status(value: object) -> dict[str, object]:
    """投影图片处理模式，避免网页自行推断主模型能力。"""

    if not isinstance(value, Mapping):
        return {"ready": False, "mode": "unavailable", "label": "图片理解未配置"}
    mode = _text(value.get("mode"), limit=24).lower()
    if mode not in {"direct", "summary", "unavailable"}:
        mode = "unavailable"
    labels = {
        "direct": "主模型直接识图",
        "summary": "视觉模型摘要回退",
        "unavailable": "图片理解未配置",
    }
    return {
        "ready": bool(value.get("ready", False)) and mode != "unavailable",
        "mode": mode,
        "label": labels[mode],
    }


def _safe_tts_profiles(value: object) -> list[dict[str, object]]:
    """投影 TTS profile 列表，只公开名称、语言和可用状态。"""

    result: list[dict[str, object]] = []
    for item in _sequence(value):
        if isinstance(item, Mapping):
            profile_id = _text(item.get("id", item.get("profile")), limit=128)
            if not profile_id:
                continue
            languages = tuple(
                language
                for language in (
                    _text(raw, limit=16).lower() for raw in _sequence(item.get("languages"))
                )
                if language
            )[:8]
            result.append(
                {
                    "id": profile_id,
                    "languages": languages,
                    "enabled": bool(item.get("enabled", True)),
                    "active": bool(item.get("active", False)),
                }
            )
        else:
            profile_id = _text(item, limit=128)
            if profile_id:
                result.append({"id": profile_id, "languages": (), "enabled": True, "active": False})
    return result[:32]


def _safe_tts_health(value: object) -> dict[str, object]:
    """投影启动健康与后端标签，不公开端点、命令或请求正文。"""

    if not isinstance(value, Mapping):
        return {
            "checked": False,
            "available": False,
            "pending": False,
            "backend": "语音后端待检测",
            "message": "启动健康状态尚未上报",
        }
    engine = _text(value.get("engine"), limit=128).lower()
    backend = (
        "多音色路由"
        if engine.startswith("tts-router")
        else {
            "gpt-sovits": "GPT-SoVITS",
            "gpt-sovits-stdio": "GPT-SoVITS 标准输入",
            "subprocess": "本地语音进程",
            "text-only": "仅文字",
            "tts": "语音协调器",
            "pending": "语音后端初始化",
        }.get(engine, "语音后端待检测")
    )
    latency = value.get("latency_ms")
    if isinstance(latency, bool):
        latency = None
    try:
        latency_value = float(latency) if latency is not None else None
    except (TypeError, ValueError, OverflowError):
        latency_value = None
    if latency_value is not None and (
        not isfinite(latency_value) or latency_value < 0 or latency_value > 300_000
    ):
        latency_value = None
    return {
        "checked": True,
        "available": bool(value.get("available", False)),
        "pending": bool(value.get("pending", False)),
        "backend": backend,
        "message": friendly_public_text(value.get("message"), limit=200)
        or ("语音后端已就绪" if bool(value.get("available", False)) else "语音后端未就绪"),
        "latency_ms": latency_value,
    }


def _safe_configuration(value: object) -> dict[str, object]:
    """投影配置文件连接和自动重载状态，不携带路径或配置正文。"""

    if not isinstance(value, Mapping):
        return {
            "connected": False,
            "watcher_enabled": False,
            "watcher_running": False,
            "auto_reload": False,
            "status": "unavailable",
            "status_label": _status_label("unavailable"),
            "generation": 0,
            "message": "配置文件状态暂不可用",
        }
    connected = bool(value.get("connected", False))
    watcher_enabled = bool(value.get("watcher_enabled", value.get("auto_reload", False)))
    watcher_running = bool(value.get("watcher_running", False))
    auto_reload = bool(value.get("auto_reload", watcher_enabled and watcher_running))
    status = _status_key(value.get("status")) or (
        "running" if auto_reload else ("stopped" if connected else "unavailable")
    )
    # 公开状态只允许有限生命周期值；未知值统一降级，不能把内部异常文本
    # 当作页面协议继续传播。
    allowed = {
        "unavailable",
        "running",
        "stopped",
        "deferred",
        "configuration_rejected",
        "reloaded",
        "restart_required",
        "unchanged",
    }
    if status == "rejected":
        status = "configuration_rejected"
    if status not in allowed:
        status = "unavailable"
    try:
        generation = max(0, int(value.get("generation", 0) or 0))
    except (TypeError, ValueError, OverflowError):
        generation = 0
    message = friendly_public_text(value.get("message"), limit=240)
    if not message:
        if not connected:
            message = "配置文件未连接"
        elif not watcher_enabled:
            message = "自动重载已关闭；可在配置中心开启"
        elif watcher_running:
            message = "配置文件已连接，修改后会自动校验并应用"
        else:
            message = "自动重载服务尚未运行"
    return {
        "connected": connected,
        "watcher_enabled": watcher_enabled,
        "watcher_running": watcher_running,
        "auto_reload": auto_reload,
        "status": status,
        "status_label": _status_label(status),
        "generation": generation,
        "message": message,
    }


def _safe_memory_status(value: object) -> dict[str, object]:
    """投影记忆策略摘要，不返回任何记忆内容、标签或存储路径。"""

    if not isinstance(value, Mapping):
        return {
            "enabled": False,
            "status": "unavailable",
            "status_label": "暂不可用",
            "recall_limit": 0,
            "context_max_chars": 0,
            "max_memories": 0,
            "consolidation_enabled": False,
            "summarization_enabled": False,
            "summary_running": False,
            "summary_busy": False,
            "summary_completed": 0,
            "summary_last_status": "unavailable",
            "extraction_pending": 0,
            "extraction_completed": 0,
            "extraction_last_status": "unavailable",
            "vector_index_size": 0,
            "lexical_index": "unavailable",
            "message": "记忆状态暂不可用",
        }
    enabled = bool(value.get("enabled", False))
    status = _status_key(value.get("status")) or ("ready" if enabled else "disabled")
    if status not in {"ready", "disabled", "unavailable"}:
        status = "unavailable"
    labels = {"ready": "已启用", "disabled": "已关闭", "unavailable": "暂不可用"}

    def bounded_int(name: str, low: int, high: int) -> int:
        try:
            parsed = int(value.get(name, 0) or 0)
        except (TypeError, ValueError, OverflowError):
            return 0
        return max(low, min(high, parsed)) if parsed else 0

    message = friendly_public_text(value.get("message"), limit=160)
    if not message:
        message = (
            "记忆已启用，新的对话会按优先级召回"
            if status == "ready"
            else ("记忆已关闭" if status == "disabled" else "记忆状态暂不可用")
        )
    return {
        "enabled": enabled,
        "status": status,
        "status_label": labels[status],
        "recall_limit": bounded_int("recall_limit", 1, 100),
        "context_max_chars": bounded_int("context_max_chars", 512, 50000),
        "max_memories": bounded_int("max_memories", 100, 10000),
        "consolidation_enabled": bool(value.get("consolidation_enabled", False)),
        "summarization_enabled": bool(value.get("summarization_enabled", False)),
        "summary_running": bool(value.get("summary_running", False)),
        "summary_busy": bool(value.get("summary_busy", False)),
        "summary_completed": bounded_int("summary_completed", 0, 1_000_000),
        "summary_last_status": _text(value.get("summary_last_status"), limit=32),
        "extraction_pending": bounded_int("extraction_pending", 0, 1_000_000),
        "extraction_completed": bounded_int("extraction_completed", 0, 1_000_000),
        "extraction_last_status": _text(value.get("extraction_last_status"), limit=32),
        "vector_index_size": bounded_int("vector_index_size", 0, 1_000_000),
        "lexical_index": (
            _text(value.get("lexical_index"), limit=32)
            if _text(value.get("lexical_index"), limit=32)
            in {"fts5", "sparse_fallback", "unavailable"}
            else "unavailable"
        ),
        "message": message,
    }


def _safe_activity_status(value: object) -> dict[str, object]:
    """只公开用户活跃探针的状态，不转发来源、时间或原始返回值。"""

    source = value if isinstance(value, Mapping) else {}
    raw_provider = source.get("system_idle_provider", "disabled")
    provider = raw_provider.strip().lower() if isinstance(raw_provider, str) else "disabled"
    provider = provider if provider in {"disabled", "x11", "windows"} else "disabled"
    raw_provider_status = source.get("system_idle_status", "disabled")
    provider_status = (
        raw_provider_status.strip().lower()
        if isinstance(raw_provider_status, str)
        else "unavailable"
    )
    provider_status = (
        provider_status
        if provider_status in {"disabled", "unknown", "ready", "unavailable"}
        else "unavailable"
    )
    labels = {
        "disabled": "未启用",
        "unknown": "等待探测",
        "ready": "已就绪",
        "unavailable": "不可用",
    }
    provider_labels = {
        "disabled": "不读取系统状态",
        "x11": "X11 空闲探针",
        "windows": "Windows 空闲探针",
    }
    return {
        "system_idle_provider": provider,
        "system_idle_provider_label": provider_labels[provider],
        "system_idle_status": provider_status,
        "system_idle_status_label": labels[provider_status],
        "system_idle_available": provider_status == "ready",
    }


def _safe_theme_tokens(value: object) -> dict[str, str]:
    """只公开 MD3 令牌，不接受选择器或任意 CSS 声明。"""

    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        name = str(raw_name or "").strip()
        rendered = str(raw_value or "").strip()
        if (
            name in _PUBLIC_THEME_KEYS
            and _PUBLIC_THEME_TOKEN.fullmatch(name)
            and _PUBLIC_THEME_VALUE.fullmatch(rendered)
        ):
            result[name] = rendered
    return result


def public_control_state(value: Mapping[str, Any] | None) -> dict[str, object]:
    """生成普通/Web 控制面所需的最小脱敏状态。"""

    source = value if isinstance(value, Mapping) else {}
    interaction = source.get("interaction")
    interaction = interaction if isinstance(interaction, Mapping) else {}
    renderer = source.get("renderer")
    renderer = renderer if isinstance(renderer, Mapping) else {}
    window = source.get("window")
    window = window if isinstance(window, Mapping) else {}
    input_shape = window.get("input_shape")
    input_shape = input_shape if isinstance(input_shape, Mapping) else {}
    raw_topmost_status = window.get("always_on_top_status")
    raw_topmost_status = raw_topmost_status if isinstance(raw_topmost_status, Mapping) else {}
    topmost_state = str(raw_topmost_status.get("status", "") or "").strip().lower()
    if topmost_state not in {"available", "degraded", "unavailable", "requested", "cancelled"}:
        topmost_state = ""
    topmost_status = {
        "status": topmost_state,
        "enabled": bool(raw_topmost_status.get("enabled", window.get("always_on_top", False))),
        "detail": friendly_public_text(raw_topmost_status.get("detail"), limit=180),
    }
    affection = source.get("affection")
    affection = affection if isinstance(affection, Mapping) else {}
    capabilities = source.get("capabilities")
    capabilities = capabilities if isinstance(capabilities, Mapping) else {}
    model = interaction.get("model")
    model = model if isinstance(model, Mapping) else {}
    feedback = _safe_feedback(source.get("feedback"))
    model_image = _safe_model_image_status(source.get("model_image"))
    configuration = _safe_configuration(source.get("configuration"))
    memory = _safe_memory_status(source.get("memory"))
    activity = _safe_activity_status(source.get("activity"))
    raw_parts = source.get("parts")
    position = _safe_position(window.get("position"))
    operation = _safe_operation(source.get("operation"))
    timeline = _safe_timeline(source.get("timeline"))
    observation = _safe_observation(source.get("observation"))
    model_ready = model.get("ready")
    model_message = friendly_public_text(model.get("message"), limit=240)
    if model_ready is False:
        model_message = "模型尚未连接，请点击“配置模型”完成连接"
    tts = source.get("tts")
    tts = tts if isinstance(tts, Mapping) else {}
    tts_health = _safe_tts_health(tts.get("health"))
    try:
        revision = max(0, int(source.get("revision", 0) or 0))
    except (TypeError, ValueError):
        revision = 0
    try:
        updated_at = float(source.get("updated_at", 0) or 0)
    except (TypeError, ValueError):
        updated_at = 0.0
    renderer_public: dict[str, object] = {
        "label": (
            _renderer_label(
                renderer.get("backend"),
                available=bool(renderer.get("available", False)),
            )
            if renderer.get("backend") is not None
            else (_text(renderer.get("label"), limit=80) or "未知后端")
        ),
        "available": bool(renderer.get("available", False)),
        "message": friendly_public_text(renderer.get("message"), limit=160),
    }
    if "model" in renderer or "models" in renderer:
        renderer_public.update(
            {
                "model": _text(renderer.get("model"), limit=256),
                "models": [
                    model
                    for model in (
                        _text(item, limit=256)
                        for item in _sequence(renderer.get("models"))
                        if isinstance(item, str)
                    )
                    if model
                ][:32],
            }
        )
    return {
        "revision": revision,
        "updated_at": updated_at,
        "connection": {
            "state": friendly_public_text(source.get("connection_state"), limit=32) or "本地",
            "message": friendly_public_text(source.get("connection_message"), limit=160)
            or "本地控制通道已连接",
        },
        "interaction": {
            # 公开状态只返回用户词汇；原始阶段仍由核心交互契约在宿主内部使用。
            "phase": _phase_label(interaction.get("phase")),
            "streaming": bool(interaction.get("streaming", False)),
            "busy": bool(interaction.get("busy", False)),
            "text": friendly_public_text(
                interaction.get("text"), limit=4000, preserve_newlines=True
            ),
            "murmur": friendly_public_text(
                interaction.get("murmur"), limit=2000, preserve_newlines=True
            ),
            "tool_status": friendly_public_text(
                interaction.get("tool_status"), limit=400, preserve_newlines=True
            ),
            "mood": _text(interaction.get("mood"), limit=32),
            "safe_message": friendly_public_text(interaction.get("safe_message"), limit=400),
            "retryable": bool(interaction.get("retryable", False)),
            "actions": _safe_actions(interaction.get("actions")),
            "approval": _safe_approval(interaction.get("approval")),
            "pending_approvals": _safe_approvals(interaction.get("pending_approvals")),
            "model": {
                "ready": model_ready,
                "message": model_message,
                "channel_count": _safe_int(model.get("channel_count", 0)),
                "action": _safe_actions((model.get("action"),)),
            },
        },
        "model_channels": _safe_model_channels(source.get("model_channels")),
        "model_image": model_image,
        "configuration": configuration,
        "theme": _safe_theme_tokens(source.get("theme")),
        "memory": memory,
        "activity": activity,
        "renderer": renderer_public,
        "window": {
            "locked": bool(window.get("locked", False)),
            "click_through": bool(window.get("click_through", False)),
            "always_on_top": bool(window.get("always_on_top", False)),
            "always_on_top_status": topmost_status,
            "position": position,
            "input_shape": {
                "ready": bool(input_shape.get("ready", False)),
                "input_ready": bool(input_shape.get("input_ready", False)),
                "status": _status_label(input_shape.get("status", "")),
            },
        },
        "affection": {
            "current": _text(affection.get("current"), limit=32),
            "tier": _text(affection.get("tier"), limit=48),
            "threshold": _text(affection.get("threshold"), limit=32),
            "description": friendly_public_text(affection.get("description"), limit=200),
        },
        "feedback": feedback,
        "parts": _safe_parts(raw_parts),
        "operation": operation,
        "timeline": timeline,
        "observation": observation,
        "tts": {
            "state": friendly_public_text(tts.get("state"), limit=32),
            "message": friendly_public_text(tts.get("message"), limit=240),
            "health": tts_health,
            "backend": tts_health["backend"],
            "available": tts_health["available"],
            "language": _text(tts.get("language"), limit=16).lower() or "zh",
            "language_protocol": _text(tts.get("language_protocol"), limit=16).lower(),
            "profile": _text(tts.get("active_profile", tts.get("profile")), limit=128),
            "profiles": _safe_tts_profiles(tts.get("profiles")),
        },
        "capabilities": {
            "expressions": _safe_capabilities(capabilities.get("expressions"), "expression"),
            "motions": _safe_capabilities(capabilities.get("motions"), "motion"),
        },
    }


def _render_control_surface_template(default_theme_css: str, state_json: str) -> str:
    template = (
        files("gui.web").joinpath("templates/control_surface.html").read_text(encoding="utf-8")
    )
    replacements = {
        "__MEAPET_DEFAULT_THEME_CSS__": default_theme_css,
        "__MEAPET_INITIAL_STATE_JSON__": state_json,
    }
    return re.sub(
        "__MEAPET_DEFAULT_THEME_CSS__|__MEAPET_INITIAL_STATE_JSON__",
        lambda match: replacements[match.group(0)],
        template,
    )


def control_surface_html(initial_state: Mapping[str, Any] | None = None) -> str:
    """返回可嵌入 Qt WebEngine 或本地页面的点击式控制台。"""

    default_theme_css = build_md3_web_css(DARK_MD3_THEME)
    state_json = json.dumps(
        public_control_state(initial_state), ensure_ascii=False, separators=(",", ":")
    ).translate(
        str.maketrans(
            {
                "<": r"\u003c",
                ">": r"\u003e",
                "&": r"\u0026",
                "\u2028": r"\u2028",
                "\u2029": r"\u2029",
            }
        )
    )
    return _render_control_surface_template(default_theme_css, state_json)


__all__ = [
    "control_surface_html",
    "friendly_public_text",
    "public_control_state",
    "safe_public_text",
]
