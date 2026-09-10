from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
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
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
{default_theme_css}
:root {{
  color-scheme: var(--md3-color-scheme);
  --background: var(--md3-color-surface);
  --surface: var(--md3-color-surface-container);
  --surface-raised: var(--md3-color-surface-container-high);
  --foreground: var(--md3-color-on-surface);
  --muted: var(--md3-color-on-surface-variant);
  --border: var(--md3-color-outline-variant);
  --ring: var(--md3-color-tertiary);
  --primary: var(--md3-color-primary);
  --primary-foreground: var(--md3-color-on-primary);
  --success: var(--md3-color-success);
  --warning: var(--md3-color-warning);
  --danger: var(--md3-color-error);
  font-family: var(--md3-font-family), system-ui, sans-serif;
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin: 0; padding: 12px; background:
  radial-gradient(900px 420px at 8% -10%, var(--surface) 0%, transparent 62%),
  var(--background); color: var(--foreground); }}
main {{ max-width: 880px; margin: auto; display: grid; gap: 10px; }}
.card {{ padding: 13px 15px; border: 1px solid var(--border); border-radius: 14px;
  background: var(--surface-raised);
  box-shadow: 0 8px 24px rgba(0,0,0,.18); }}
.card > h3, .card > .row > h3 {{ letter-spacing: .01em; }}
.row {{ display: flex; flex-wrap: wrap; gap: 7px; align-items: center; }}
.between {{ justify-content: space-between; }}
.muted {{ color: var(--muted); font-size: 12px; }}
.skipLink {{ position: fixed; left: 8px; top: -48px; z-index: 30; padding: 8px 12px;
  border-radius: 8px; background: var(--primary); color: var(--primary-foreground);
  font-weight: 700; }}
.skipLink:focus {{ top: 8px; }}
.status {{ display: inline-flex; align-items: center; gap: 5px; color: var(--ring);
  font-weight: 700; padding: 3px 8px; border: 1px solid var(--border);
  border-radius: 999px; background: var(--surface-raised); }}
.ok {{ color: var(--success); }}
.warn {{ color: var(--warning); }}
.error {{ color: var(--danger); }}
#configurationStatus {{ font-weight: 700; }}
#configurationStatus[data-status="running"],
#configurationStatus[data-status="reloaded"] {{ color: var(--success); }}
#configurationStatus[data-status="stopped"],
#configurationStatus[data-status="deferred"] {{ color: var(--warning); }}
#configurationStatus[data-status="configuration_rejected"],
#configurationStatus[data-status="unavailable"],
#configurationStatus[data-status="restart_required"] {{ color: var(--danger); }}
#configurationMessage {{ white-space: pre-wrap; line-height: 1.35; }}
#memoryStatus[data-status="ready"] {{ color: var(--success); font-weight: 700; }}
#memoryStatus[data-status="disabled"] {{ color: var(--warning); font-weight: 700; }}
#memoryStatus[data-status="unavailable"] {{ color: var(--danger); font-weight: 700; }}
#memoryMessage {{ white-space: pre-wrap; line-height: 1.35; }}
#activityStatus[data-status="ready"] {{ color: var(--success); font-weight: 700; }}
#activityStatus[data-status="unavailable"] {{ color: var(--danger); font-weight: 700; }}
#activityStatus[data-status="unknown"] {{ color: var(--warning); font-weight: 700; }}
#activityMessage {{ white-space: pre-wrap; line-height: 1.35; }}
h2, h3 {{ margin: 0; }}
h3 {{ font-size: 15px; margin-bottom: 7px; }}
button {{ border: 1px solid var(--border); border-radius: 10px; padding: 9px 13px;
  background: var(--surface-raised); color: var(--foreground); cursor: pointer;
  transition: border-color .16s ease, background .16s ease, transform .08s ease; }}
button:hover {{ border-color: var(--primary); background: var(--surface-raised); }}
button:focus-visible {{ outline: 2px solid var(--ring); outline-offset: 2px; }}
button:active:not(:disabled) {{ background: var(--surface); transform: translateY(1px); }}
button:disabled {{ opacity: .45; cursor: not-allowed; }}
button[data-pending="true"] {{ opacity: .65; cursor: progress; }}
button.primary {{ background: var(--primary); color: var(--primary-foreground);
  border: 0; font-weight: 700; }}
button.primary:hover {{ background: var(--md3-color-primary-container); }}
button.danger {{ color: var(--danger); border-color: var(--danger); }}
#output {{ min-height: 64px; max-height: 150px; overflow: auto;
  white-space: pre-wrap; line-height: 1.45; }}
#murmur, #tool {{ margin-top: 4px; white-space: pre-wrap; }}
#feedback {{ min-height: 30px; }}
#feedbackText {{ white-space: pre-wrap; line-height: 1.35; }}
#timelineCard {{ display: grid; gap: 7px; }}
#timeline {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 5px; }}
#timeline li {{ display: grid; grid-template-columns: auto 1fr auto; gap: 7px;
  align-items: baseline; padding: 6px 8px; border-radius: 8px;
  background: var(--surface-raised); color: var(--foreground); font-size: 12px; }}
#timeline li .timelineStatus {{ color: var(--success); white-space: nowrap; }}
#timeline li[data-status="failed"] .timelineStatus,
#timeline li[data-status="error"] .timelineStatus,
#timeline li[data-status="unavailable"] .timelineStatus,
#timeline li[data-status="timeout"] .timelineStatus {{ color: var(--danger); }}
#timeline li[data-status="requested"] .timelineStatus,
#timeline li[data-status="pending"] .timelineStatus {{ color: var(--warning); }}
#timelineEmpty {{ color: var(--muted); font-size: 12px; }}
#observationStatus {{ min-height: 18px; }}
#observationResult {{ display: grid; gap: 5px; min-height: 20px; }}
#observationResult .observationLine {{ padding: 6px 8px; border-radius: 8px;
  background: var(--surface-raised); color: var(--foreground); font-size: 12px; }}
#processObservation {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; }}
#processObservation li {{ display: flex; justify-content: space-between; gap: 8px;
  padding: 5px 8px; border-radius: 7px; background: var(--surface-raised);
  color: var(--foreground);
  font-size: 12px; }}
#approvalCard {{ display: none; }}
#approvalSummary {{ white-space: pre-wrap; line-height: 1.35; }}
#channelStatusRows {{ display: grid; gap: 6px; }}
.channelStatusRow {{ display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.5fr) auto auto;
  gap: 8px; align-items: center; padding: 7px 8px; border-radius: 8px;
  background: var(--surface-raised); color: var(--foreground); font-size: 12px; }}
.channelStatusRow .channelMeta {{ color: var(--muted); }}
.channelStatusRow .channelReady {{ color: var(--success); white-space: nowrap; }}
.channelStatusRow[data-ready="false"] .channelReady {{ color: var(--warning); }}
.channelStatusRow > span {{ min-width: 0; overflow-wrap: anywhere; }}
.modelChoices {{ display: flex; flex-wrap: wrap; gap: 4px; justify-content: flex-end; }}
.modelChoices button {{ min-height: 28px; padding: 5px 8px; font-size: 11px; }}
#channelStatusEmpty {{ color: var(--muted); font-size: 12px; }}
input:not([type="checkbox"]), select {{ flex: 1 1 180px; min-width: 0;
  border: 1px solid var(--border); border-radius: 10px;
  padding: 10px 12px; background: var(--background); color: var(--foreground); outline: none; }}
input:not([type="checkbox"]):focus, select:focus {{ border-color: var(--primary);
  box-shadow: 0 0 0 3px color-mix(in srgb, var(--primary) 22%, transparent); }}
label.field {{ display: grid; gap: 4px; min-width: 130px; flex: 1 1 160px;
  color: var(--muted); font-size: 12px; }}
label.checkField {{ display: inline-flex; align-items: center; gap: 6px;
  min-height: 40px; color: var(--foreground); font-size: 12px; }}
.actionEditor {{ display: grid; gap: 8px; margin-top: 10px; padding: 10px;
  border: 1px solid var(--border); border-radius: 10px; background: var(--background); }}
.selection button[aria-pressed="true"] {{ border-color: var(--primary);
  background: var(--md3-color-primary-container); color: var(--md3-color-on-primary-container); }}
.actionHint {{ min-height: 18px; }}
progress {{ width: 150px; height: 9px; accent-color: var(--primary); }}
.compact button {{ padding: 7px 10px; min-height: 32px; }}
.windowActions {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 7px; }}
.windowActions button {{ width: 100%; min-width: 0; white-space: nowrap; }}
.sizeLabel {{ grid-column: 1; align-self: center; color: var(--muted); font-size: 12px; }}
.sizeActions {{ grid-column: 2 / -1; display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 7px; }}
.sizeActions button {{ width: 100%; min-width: 0; }}
#positionState {{ grid-column: 1 / -1; color: var(--muted); font-size: 12px; }}
.nudgeActions {{ grid-column: 1 / -1; display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 7px; }}
.nudgeActions button {{ width: 100%; min-width: 0; }}
#operation {{ min-height: 18px; }}
details {{ border: 1px solid var(--border); border-radius: 11px; padding: 8px 10px;
  background: var(--surface-raised); }}
summary {{ cursor: pointer; color: var(--foreground); font-weight: 700; outline: none; }}
summary:focus-visible {{ outline: 2px solid var(--primary); outline-offset: 3px;
  border-radius: 6px; }}
summary span {{ color: var(--muted); font-size: 12px; font-weight: 400; margin-left: 8px; }}
@media (max-width: 620px) {{
  body {{ padding: 8px; }}
  .card {{ padding: 9px; }}
  .windowActions {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .channelStatusRow {{ grid-template-columns: minmax(0, 1fr) auto; }}
  .channelStatusRow .channelMeta {{ grid-column: 1 / -1; grid-row: 2; }}
  .sizeLabel {{ grid-column: 1 / -1; }}
  .sizeActions {{ grid-column: 1 / -1; }}
  button {{ padding: 8px 10px; }}
  #output {{ max-height: 120px; }}
}}
@media (prefers-reduced-motion: reduce) {{
  html {{ scroll-behavior: auto; }}
  *, *::before, *::after {{
    transition-duration: .01ms !important;
    animation-duration: .01ms !important;
  }}
}}
@media (max-width: 380px) {{
  .windowActions {{ grid-template-columns: 1fr; }}
}}

/* QWidget-FancyUI / WinUI compatibility surface.  MD3 continues to own
   semantic colors; Fluent layering, navigation and geometry own composition. */
:root {{
  --fluent-accent: var(--md3-color-primary);
  --fluent-accent-text: var(--md3-color-on-primary);
  --fluent-base: var(--md3-color-surface);
  --fluent-layer: var(--md3-color-surface-container-low);
  --fluent-layer-raised: var(--md3-color-surface-container);
  --fluent-control: var(--md3-color-surface-container-high);
  --fluent-control-hover: var(--md3-color-surface-container-highest);
  --fluent-stroke: var(--md3-color-outline-variant);
  --fluent-text: var(--md3-color-on-surface);
  --fluent-text-secondary: var(--md3-color-on-surface-variant);
  --fluent-focus: var(--md3-color-primary);
  --fluent-radius-small: 4px;
  --fluent-radius: 8px;
  --fluent-radius-large: 12px;
}}
body {{
  padding: 0;
  background: var(--fluent-base);
  color: var(--fluent-text);
  font-family: "Segoe UI Variable", "Segoe UI", "Microsoft YaHei UI",
    var(--md3-font-family), system-ui, sans-serif;
}}
.fluentShell {{ min-height: 100vh; display: grid; grid-template-columns: 176px minmax(0,1fr); }}
.navigationPane {{
  position: sticky; top: 0; height: 100vh; padding: 14px 8px 10px;
  display: flex; flex-direction: column; gap: 4px;
  background: var(--fluent-layer); border-right: 1px solid var(--fluent-stroke);
}}
.navigationBrand {{ padding: 8px 10px 18px; display: grid; gap: 2px; }}
.navigationBrand strong {{ font-size: 15px; font-weight: 650; }}
.navigationBrand span {{ color: var(--fluent-text-secondary); font-size: 11px; }}
.navigationPane a {{
  min-height: 40px; padding: 0 12px; border-radius: var(--fluent-radius);
  display: flex; align-items: center; gap: 10px; color: var(--fluent-text);
  text-decoration: none; border: 1px solid transparent; position: relative;
  transition: background-color .18s ease, border-color .18s ease;
}}
.navigationPane a::before {{
  content: ""; width: 3px; height: 16px; border-radius: 2px;
  background: transparent; position: absolute; left: 2px;
}}
.navigationPane a:hover {{ background: var(--fluent-control-hover); }}
.navigationPane a:focus-visible {{ outline: 2px solid var(--fluent-focus); outline-offset: 1px; }}
.navigationPane a[aria-current="page"] {{ background: var(--fluent-control); }}
.navigationPane a[aria-current="page"]::before {{ background: var(--fluent-accent); }}
.navigationGlyph {{ width: 18px; height: 18px; display: inline-grid; place-items: center; }}
.navigationGlyph svg {{ width: 16px; height: 16px; fill: none; stroke: currentColor;
  stroke-width: 1.5; stroke-linecap: round; stroke-linejoin: round; }}
.navigationSpacer {{ flex: 1; }}
.contentFrame {{ min-width: 0; padding: 0 22px 28px; }}
.fluentTitleBar {{
  position: sticky; top: 0; z-index: 20; min-height: 64px; padding: 12px 2px 10px;
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  background: var(--fluent-base); border-bottom: 1px solid var(--fluent-stroke);
}}
.fluentTitleCopy {{ display: grid; gap: 1px; }}
.fluentTitleCopy h2 {{ margin: 0; font-size: 20px; font-weight: 650; letter-spacing: -.01em; }}
.fluentTitleCopy span {{ color: var(--fluent-text-secondary); font-size: 12px; }}
.fluentCommandBar {{ display: flex; align-items: center; gap: 6px; }}
main {{
  max-width: 1180px; margin: 18px auto 0; display: grid;
  grid-template-columns: repeat(2,minmax(0,1fr)); gap: 12px; align-items: start;
}}
.card {{
  padding: 16px; border: 1px solid var(--fluent-stroke);
  border-radius: var(--fluent-radius-large); background: var(--fluent-layer-raised);
  box-shadow: 0 1px 2px rgba(0,0,0,.14); min-width: 0;
}}
#overviewCard, #conversationCard, #channelStatusCard, #timelineCard,
#observationCard, #interactionCard {{ grid-column: 1 / -1; }}
.card > h3, .card > .row > h3 {{ letter-spacing: 0; font-weight: 650; }}
h2 {{ font-size: 24px; font-weight: 650; }}
h3 {{ font-size: 14px; font-weight: 650; }}
.muted {{ color: var(--fluent-text-secondary); font-size: 12px; }}
.status {{
  color: var(--fluent-text); border: 1px solid var(--fluent-stroke);
  border-radius: 999px; background: var(--fluent-control); padding: 4px 9px;
}}
button {{
  min-height: 34px; padding: 6px 12px; border: 1px solid var(--fluent-stroke);
  border-radius: var(--fluent-radius); background: var(--fluent-control);
  color: var(--fluent-text); box-shadow: 0 1px 1px rgba(0,0,0,.08);
  transition: background-color .18s ease, border-color .18s ease;
}}
button:hover {{ background: var(--fluent-control-hover); border-color: var(--md3-color-outline); }}
button:active:not(:disabled) {{ background: var(--fluent-layer); transform: none; }}
button:focus-visible {{ outline: 2px solid var(--fluent-focus); outline-offset: 2px; }}
button.primary {{ background: var(--fluent-accent); color: var(--fluent-accent-text); }}
button.primary:hover {{ background: var(--md3-color-primary-container);
  color: var(--md3-color-on-primary-container); }}
button.danger {{ color: var(--danger); border-color: var(--danger); background: transparent; }}
input:not([type="checkbox"]), select {{
  min-height: 36px; padding: 7px 10px; border: 1px solid var(--fluent-stroke);
  border-radius: var(--fluent-radius-small); background: var(--fluent-control);
  color: var(--fluent-text); border-bottom: 2px solid var(--md3-color-outline);
}}
input:not([type="checkbox"]):focus, select:focus {{
  border-color: var(--fluent-stroke); border-bottom-color: var(--fluent-accent);
  box-shadow: none; outline: none;
}}
#output {{ min-height: 110px; max-height: 260px; padding: 14px;
  border-radius: var(--fluent-radius); background: var(--fluent-base);
  border: 1px solid var(--fluent-stroke); }}
#timeline li, #observationResult .observationLine, #processObservation li,
.channelStatusRow {{ background: var(--fluent-control); color: var(--fluent-text); }}
.actionEditor, details {{
  background: var(--fluent-base); border: 1px solid var(--fluent-stroke);
  border-radius: var(--fluent-radius); box-shadow: none;
}}
summary {{ color: var(--fluent-text); }}
progress {{ accent-color: var(--fluent-accent); }}
.skipLink {{ background: var(--fluent-accent); color: var(--fluent-accent-text); }}
@media (max-width: 860px) {{
  .fluentShell {{ grid-template-columns: 1fr; }}
  .navigationPane {{ position: sticky; height: auto; z-index: 25; padding: 6px 10px;
    flex-direction: row; overflow-x: auto; border-right: 0;
    border-bottom: 1px solid var(--fluent-stroke); }}
  .navigationBrand, .navigationSpacer {{ display: none; }}
  .navigationPane a {{ flex: 0 0 auto; min-height: 36px; }}
  .navigationPane a::before {{ width: 16px; height: 3px; left: 50%; bottom: 1px; top: auto;
    transform: translateX(-50%); }}
  .contentFrame {{ padding: 0 12px 20px; }}
  main {{ grid-template-columns: 1fr; margin-top: 12px; }}
  main > .card {{ grid-column: 1; }}
}}
@media (max-width: 520px) {{
  .fluentTitleBar {{ align-items: flex-start; flex-direction: column; }}
  .fluentCommandBar {{ width: 100%; }}
  .fluentCommandBar button {{ flex: 1; }}
  .card {{ padding: 12px; }}
}}
</style><script src="qrc:///qtwebchannel/qwebchannel.js"></script>
</head><body><a class="skipLink" href="#main-content">跳到主要操作</a>
<div class="fluentShell"><aside class="navigationPane" aria-label="控制台导航">
<div class="navigationBrand"><strong>MeaPet</strong><span>Fluent control center</span></div>
<a href="#overviewCard" aria-current="page"><span class="navigationGlyph">
<svg viewBox="0 0 16 16" aria-hidden="true">
<path d="M2.5 7.2 8 2.5l5.5 4.7v6.2H9.8V9.6H6.2v3.8H2.5z"/></svg>
</span>概览</a>
<a href="#conversationCard"><span class="navigationGlyph">
<svg viewBox="0 0 16 16" aria-hidden="true">
<path d="M2.5 3.5h11v7h-6l-3.2 2.2.5-2.2H2.5z"/></svg>
</span>对话</a>
<a href="#interactionCard"><span class="navigationGlyph">
<svg viewBox="0 0 16 16" aria-hidden="true">
<path d="M4.1 6.8c-.9-1.4-.6-3 .5-3.3 1-.3 2 .7 2.4 1.8h2
c.4-1.1 1.4-2.1 2.4-1.8 1.1.3 1.4 1.9.5 3.3.4.7.6 1.5.6 2.3
0 2.4-2 4.2-4.5 4.2S3.5 11.5 3.5 9.1c0-.8.2-1.6.6-2.3Z"/>
<path d="M6.2 9.3h.1m3.4 0h.1M7 11c.6.5 1.4.5 2 0"/></svg>
</span>桌宠</a>
<a href="#observationCard"><span class="navigationGlyph">
<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.3"/>
<path d="m10.2 10.2 3.3 3.3M7 4.8v4.4M4.8 7h4.4"/></svg>
</span>工具</a>
<div class="navigationSpacer"></div>
<a href="#configurationCard"><span class="navigationGlyph">
<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="2.2"/>
<path d="M8 1.8v1.4m0 9.6v1.4M1.8 8h1.4m9.6 0h1.4M3.6 3.6l1 1
m6.8 6.8 1 1m0-8.8-1 1m-6.8 6.8-1 1"/></svg>
</span>设置</a>
</aside><div class="contentFrame">
<header class="fluentTitleBar"><div class="fluentTitleCopy"><h2>桌宠控制台</h2>
<span>对话、桌宠、自动化与运行状态</span></div><div class="fluentCommandBar">
<button data-action="show_pet">显示桌宠</button>
<button data-action="open_config" class="primary">设置</button>
</div></header><main id="main-content">
<section class="card fluentOverview" id="overviewCard">
<div class="row between"><div class="row"><h2>运行概览</h2>
<span id="renderer" class="status"></span></div><span id="connection" class="muted"></span></div>
<div id="rendererMessage" class="muted"></div>
<div class="row between"><span id="phase" class="muted"></span>
<span id="windowState" class="muted"></span></div>
<div class="row"><span>好感度</span>
<progress id="affection" max="100" value="0"></progress>
<span id="affectionText"></span></div>
<div class="windowActions" style="margin-top:7px">
<button data-action="toggle_window_lock">锁定/解锁</button>
<button data-action="toggle_always_on_top">切换置顶</button>
<button id="clickThroughButton" data-action="toggle_click_through">开启点击穿透</button>
<button data-action="show_pet">显示桌宠</button>
<button data-action="center_pet">居中桌宠</button>
<button data-action="toggle_visibility">显示/隐藏</button>
<button data-action="restart_application">重启并应用引擎</button>
<span class="sizeLabel">渲染引擎</span>
<div class="sizeActions" id="rendererBackendActions">
<button data-renderer-backend="auto">自动</button>
<button data-renderer-backend="opengl">OpenGL</button>
<button data-renderer-backend="vulkan">Vulkan</button></div>
<span id="rendererBackendHint" class="muted">选择后保存并重启生效</span>
<span class="sizeLabel">大小</span>
<div class="sizeActions"><button data-preset="small">小</button>
<button data-preset="standard">标准</button><button data-preset="large">大</button></div>
<span id="positionState">位置：读取中…</span>
<div class="nudgeActions"><button data-action="nudge_pet" data-direction="left">左移</button>
<button data-action="nudge_pet" data-direction="right">右移</button>
<button data-action="nudge_pet" data-direction="up">上移</button>
<button data-action="nudge_pet" data-direction="down">下移</button></div>
</div></section>
<section class="card" id="configurationCard"><div class="row between"><h3>配置与自动重载</h3>
<span id="configurationStatus" class="muted">状态读取中…</span></div>
<div id="configurationMessage" class="muted" aria-live="polite">配置状态读取中…</div>
<div class="row compact" style="margin-top:7px">
<button data-action="open_config" class="primary">打开配置中心</button>
<span id="configurationMeta" class="muted"></span></div></section>
<section class="card" id="personaCard"><div class="row between"><h3>人设与提示词</h3>
<span class="muted">支持热重载</span></div>
<div class="muted">桌宠身份、称呼、关系、偏好、边界、主动行为及
对话/工具/记忆提示词均可自定义。</div>
<div class="row compact" style="margin-top:7px">
<button data-action="open_config">编辑人设与提示词</button></div></section>
<section class="card" id="memoryCard"><div class="row between"><h3>记忆状态</h3>
<span id="memoryStatus" class="muted">状态读取中…</span></div>
<div id="memoryMessage" class="muted" aria-live="polite">记忆状态读取中…</div>
<div id="memoryMeta" class="muted"></div></section>
<section class="card" id="activityCard"><div class="row between"><h3>用户活跃探针</h3>
<span id="activityStatus" class="muted">状态读取中…</span></div>
<div id="activityMessage" class="muted" aria-live="polite">活跃探针状态读取中…</div>
<div id="activityMeta" class="muted"></div></section>
<section class="card" id="conversationCard"><div id="output" aria-live="polite">等待桌宠状态…</div>
<div id="murmur" class="muted"></div><div id="tool" class="muted"></div>
<div id="tts" class="muted"></div>
<div class="row" style="margin-top:9px"><input id="messageInput" autocomplete="off"
placeholder="和桌宠说点什么，回车发送">
<button id="sendButton" class="primary">发送</button>
<button id="stopButton" class="danger">停止</button><button id="retryButton">重试</button></div>
<div id="operation" class="muted" aria-live="polite"></div></section>
<section class="card" id="ttsCard"><div class="row between"><h3>语音输出</h3>
<span id="ttsState" class="muted">语音待命</span></div>
<div id="ttsHealth" class="muted" aria-live="polite">语音后端检测中…</div>
<div class="row compact" id="ttsLanguages"></div>
<div class="row compact" id="ttsProfiles"></div>
</section>
<section class="card" id="rendererModelCard"><div class="row between"><h3>Live2D 模型</h3>
<span id="rendererModelStatus" class="muted">当前模型</span></div>
<div id="rendererModels" class="muted">模型列表读取中…</div></section>
<section class="card" id="timelineCard"><div class="row between"><h3>活动记录</h3>
<button id="clearTimeline" class="compact">清空本地记录</button></div>
<ol id="timeline" aria-live="polite"><li id="timelineEmpty">还没有操作记录。</li></ol></section>
<section class="card" id="observationCard"><div class="row between"><h3>桌面观察</h3>
<span class="muted">只读取摘要，不显示命令行</span></div>
<div class="row compact"><button data-action="read_foreground_window">查看前台窗口</button>
<button data-action="read_processes">查看运行中的程序</button></div>
<div id="observationStatus" class="muted" aria-live="polite"></div>
<div id="observationResult" aria-live="polite"><div class="observationLine">
点击按钮读取当前桌面摘要。</div></div>
<ul id="processObservation"></ul></section>
<section class="card" id="modelCard" style="display:none">
<div class="row between"><h3>模型连接</h3><span class="muted">需要配置后才能对话</span></div>
<div id="modelMessage" class="muted" aria-live="polite"></div>
<div class="row compact" id="modelActions"></div></section>
<section class="card" id="channelStatusCard"><div class="row between"><h3>模型渠道状态</h3>
<div class="row compact"><span class="muted">只显示连接摘要</span>
<button data-action="configure_model">配置模型</button></div></div>
<div id="modelImageStatus" class="muted" aria-live="polite">图片理解状态读取中…</div>
<div id="channelStatusRows" aria-live="polite">
<div id="channelStatusEmpty">暂无模型渠道。</div></div></section>
<section class="card" id="feedback"><div class="row between"><h3>最近互动</h3>
<span id="feedbackMeta" class="muted"></span></div>
<div id="feedbackText" class="muted" aria-live="polite">
点击桌宠部位后，反馈会显示在这里。</div></section>
<section class="card" id="approvalCard"><h3>需要确认的桌面操作</h3>
<div id="approvalSummary" class="muted"></div><div id="approvalCountdown" class="warn"></div>
<div class="row compact" id="approvalActions"></div></section>
<section class="card"><div class="row between"><h3>点击互动</h3>
<span class="muted">按部位触发反馈</span></div>
<div class="row compact" id="parts"></div>
<div class="row compact" id="stateActions" style="margin-top:7px"></div></section>
<section class="card" id="interactionCard"><details id="appearanceDetails"><summary>动作表现
<span>支持快捷操作与参数化编排</span></summary><div style="margin-top:10px"><h3>表情</h3>
<div class="row compact" id="expressions"></div>
<h3 style="margin-top:10px">动作</h3><div class="row compact" id="motions"></div>
<div class="actionEditor"><h3>表情编排</h3>
<div class="muted">点选 1 到 8 个表情；再次点按可取消。</div>
<div class="row compact selection" id="expressionSequenceChoices"></div>
<div class="row">
<label class="field">播放方式<select id="expressionRequestMode">
<option value="sequence">依次播放</option><option value="blend">参数混合</option></select></label>
<label class="field">每层持续（秒）<input id="expressionRequestDuration" type="number"
min="0.05" max="120" step="0.05" value="1.5"></label>
<label class="field">过渡（秒）<input id="expressionRequestTransition" type="number"
min="0" max="10" step="0.05" value="0.2"></label>
<label class="checkField"><input id="expressionRequestLoop" type="checkbox">循环</label>
</div>
<label class="field">混合权重（与选择顺序对应，可留空）
<input id="expressionRequestWeights" placeholder="例如：1, 0.7, 0.3"></label>
<label class="field">Live2D 参数（可留空）
<input id="expressionRequestParameters" placeholder="ParamAngleX=12, ParamEyeLOpen=1"></label>
<button id="playExpressionRequest" class="primary">播放表情编排</button>
<div id="expressionRequestHint" class="muted actionHint" aria-live="polite"></div></div>
<div class="actionEditor"><h3>动作编排</h3>
<div class="muted">先选择一个动作，再设置持续时间、过渡和参数。</div>
<div class="row compact selection" id="motionRequestChoices"></div>
<div class="row">
<label class="field">持续（秒）<input id="motionRequestDuration" type="number"
min="0.05" max="300" step="0.05" value="1.5"></label>
<label class="field">过渡（秒）<input id="motionRequestTransition" type="number"
min="0" max="10" step="0.05" value="0.2"></label>
<label class="checkField"><input id="motionRequestLoop" type="checkbox">循环</label>
</div>
<label class="field">Live2D 参数（可留空）
<input id="motionRequestParameters" placeholder="ParamAngleX=8, ParamBodyAngleX=4"></label>
<button id="playMotionRequest" class="primary">按参数播放动作</button>
<div id="motionRequestHint" class="muted actionHint" aria-live="polite"></div></div>
</div></details></section>
</main></div></div><script>
let state = {state_json};
let bridge = null;
let bridgeReady = false;
let selectedExpressionTokens = [];
let selectedMotionToken = '';
// Qt WebEngine 的 runJavaScript/getState 回调不是严格按调用顺序返回。
// 只接受不早于当前快照的 revision，避免旧审批/回复覆盖新状态。
function stateRevision(value) {{
  const number=Number(value&&value.revision);
  return Number.isFinite(number) && number >= 0 ? Math.floor(number) : 0;
}}
function applyTheme(tokens) {{
  const source=tokens && typeof tokens === 'object' ? tokens : {{}};
  for (const [name,value] of Object.entries(source)) {{
    if (!/^--md3-[a-z0-9-]{{1,80}}$/.test(name)) continue;
    const rendered=String(value||'');
    if (!/^[#A-Za-z0-9 ,.()'"_-]{{1,128}}$/.test(rendered)) continue;
    document.documentElement.style.setProperty(name,rendered);
  }}
}}
function applyState(next) {{
  const candidate=next && typeof next === 'object' ? next : {{}};
  if (stateRevision(candidate) < stateRevision(state)) return false;
  state=candidate;
  applyTheme(candidate.theme);
  render();
  return true;
}}
const pendingInvocations = new Map();
let invocationSequence = 0;
let localOperationMessage = '';
let localOperationStatus = '';
let localOperationRevision = null;
let timelineEntries = [];
let timelineRevision = null;
const labels = {{neutral:'自然',happy:'开心',sad:'难过',curious:'好奇',
  surprised:'惊讶',shy:'害羞',idle:'待机',blink:'眨眼',wave:'挥手',walk:'走动',angry:'生气'}};
const partLabels = {{head:'猫猫头',body:'身体',lower_left:'左边',lower_right:'右边'}};
const phaseLabels = {{
  idle:'等待互动',streaming:'正在回复',tool_running:'正在执行桌面操作',
  approval_required:'等待你的确认',completed:'本轮已完成',failed:'本轮未完成',
  configuration_required:'需要连接模型'
}};
const statusLabels = {{
  idle:'等待处理',requested:'已提交',pending:'等待处理',started:'已开始',
  accepted:'已接受',completed:'已完成',available:'已就绪',updated:'已更新',
  saved:'已保存',validated:'已校验',reloaded:'已刷新',approval_required:'等待确认',
  tool_limit:'需要确认',degraded:'降级运行',failed:'执行失败',error:'暂时不可用',
  unavailable:'暂时不可用',cancelled:'已停止',canceled:'已停止',denied:'已拒绝',
  rejected:'已拒绝',timeout:'响应超时',restart_required:'需要重启'
}};
const operationLabels = {{
  open_config:'打开配置中心',configure_model:'连接模型',select_model_channel:'切换模型渠道',select_tts_profile:'切换语音',select_tts_language:'切换输出语言',select_renderer_backend:'切换渲染引擎',submit_text:'发送消息',stop:'停止回复',retry:'重试回复',
  approve:'允许桌面操作',grant_session:'允许本次会话',deny:'拒绝桌面操作',
  pet_part:'部位互动',expression:'切换表情',motion:'播放动作',expression_request:'表情编排',motion_request:'动作编排',show_pet:'显示桌宠',
  toggle_visibility:'切换显示',center_pet:'居中桌宠',nudge_pet:'移动桌宠',
  set_display_size:'调整大小',toggle_window_lock:'切换锁定',
  toggle_always_on_top:'切换置顶',toggle_click_through:'切换点击穿透',
  restore_click_through:'恢复点击',read_foreground_window:'查看前台窗口',
  restart_application:'重启桌宠',
  read_processes:'查看运行中的程序'
}};
const friendlyOperationLabels = new Set(Object.values(operationLabels));
const partPhrases = {{head:'喵？摸摸头～',body:'呼噜……这里也可以摸。',
  lower_left:'左边被发现啦。',lower_right:'右边也要轻轻碰哦。'}};
function safeText(value) {{ return String(value == null ? '' : value); }}
function phaseLabel(value) {{
  const key=safeText(value).toLowerCase(); return phaseLabels[key]||'等待互动';
}}
function statusLabel(value) {{
  const key=safeText(value).toLowerCase(); return statusLabels[key]||'处理中';
}}
function operationLabel(value) {{
  const raw=safeText(value); const key=raw.toLowerCase();
  return friendlyOperationLabels.has(raw) ? raw : (operationLabels[key]||'桌宠操作');
}}
function actionLabel(value, kind) {{
  const normalized=safeText(value).toLowerCase();
  if (labels[normalized]) return labels[normalized];
  for (const key of Object.keys(labels))
    if (normalized.indexOf(key) === 0) return labels[key];
  return kind==='expression'?'表情':'动作';
}}
function actionKey(kind, payload) {{
  let encoded = '';
  try {{ encoded = JSON.stringify(payload || {{}}); }} catch (_) {{ encoded = ''; }}
  return String(kind || '') + ':' + encoded;
}}
function pendingKind(kind) {{
  const prefix=String(kind || '')+':';
  for (const key of pendingInvocations.keys()) if (key.indexOf(prefix) === 0) return true;
  return false;
}}
function resultMessage(result) {{
  if (!result || typeof result !== 'object') return '操作已提交';
  return safeText(result.message || result.reason || result.status_label ||
    statusLabel(result.status) || '操作已提交');
}}
function timelineEntry(value, fallbackKind='') {{
  if (!value || typeof value !== 'object') value={{}};
  const status=safeText(value.status||'completed').toLowerCase();
  const rawLabel=safeText(value.label||value.kind||'').toLowerCase();
  const label=rawLabel==='桌宠操作' && fallbackKind
    ? operationLabel(fallbackKind) : operationLabel(rawLabel||fallbackKind);
  const message=safeText(value.message||value.reason||'');
  return {{label:label,status:status,statusLabel:safeText(value.status_label||statusLabel(status)),
    message:message.slice(0,240)}};
}}
function renderTimeline() {{
  const host=document.getElementById('timeline'); if (!host) return;
  host.replaceChildren();
  if (!timelineEntries.length) {{
    const empty=document.createElement('li'); empty.id='timelineEmpty';
    empty.textContent='还没有操作记录。'; host.appendChild(empty); return;
  }}
  timelineEntries.slice(-12).reverse().forEach(entry=>{{
    const item=document.createElement('li'); item.dataset.status=entry.status||'completed';
    const label=document.createElement('span'); label.textContent=entry.label||'桌宠操作';
    const message=document.createElement('span'); message.textContent=entry.message||'';
    const status=document.createElement('span'); status.className='timelineStatus';
    status.textContent=entry.statusLabel||statusLabel(entry.status);
    item.append(label,message,status); host.appendChild(item);
  }});
}}
function mergeTimeline(values) {{
  if (!Array.isArray(values)) return;
  const next=values.map(value=>timelineEntry(value)).filter(value=>value.label);
  if (!next.length) return;
  const encoded=JSON.stringify(next);
  if (encoded===JSON.stringify(timelineEntries)) return;
  timelineEntries=next.slice(-12); renderTimeline();
}}
function appendTimeline(value, fallbackKind='') {{
  const entry=timelineEntry(value,fallbackKind);
  if (!entry.label) return;
  timelineEntries=[...timelineEntries,entry].slice(-12); renderTimeline();
}}
function renderObservation(value) {{
  const observation=value&&typeof value==='object'?value:{{}};
  const status=document.getElementById('observationStatus');
  const result=document.getElementById('observationResult');
  const list=document.getElementById('processObservation');
  if (!status || !result || !list) return;
  const kind=safeText(observation.kind).toLowerCase();
  status.textContent=observation.status_label
    ? (safeText(observation.label||operationLabel(kind))+'：'
      +safeText(observation.status_label)) : '';
  result.replaceChildren(); list.replaceChildren();
  if (kind==='foreground_window') {{
    const line=document.createElement('div'); line.className='observationLine';
    const title=safeText(observation.title)||'无标题窗口';
    const process=safeText(observation.process_name)||'未知程序';
    const pid=observation.pid==null?'':(' · PID '+safeText(observation.pid));
    line.textContent='前台窗口：'+title+' · '+process+pid; result.appendChild(line);
    return;
  }}
  if (kind==='processes') {{
    const rows=Array.isArray(observation.processes)?observation.processes:[];
    if (!rows.length) {{
      const line=document.createElement('div'); line.className='observationLine';
      line.textContent='没有可显示的运行中程序。'; result.appendChild(line); return;
    }}
    rows.slice(0,20).forEach(row=>{{
      const item=document.createElement('li');
      const name=document.createElement('span');
      name.textContent=safeText(row&&row.name)||'未知程序';
      const pid=document.createElement('span');
      pid.textContent=row&&row.pid!=null?'PID '+safeText(row.pid):'';
      item.append(name,pid); list.appendChild(item);
    }}); return;
  }}
  const line=document.createElement('div'); line.className='observationLine';
  line.textContent='点击按钮读取当前桌面摘要。'; result.appendChild(line);
}}
function renderModelImageStatus(value) {{
  const host=document.getElementById('modelImageStatus'); if (!host) return;
  const item=value&&typeof value==='object'?value:{{}};
  const ready=Boolean(item.ready);
  host.textContent='图片理解：'+(safeText(item.label)||(ready?'已就绪':'未配置'));
  host.dataset.ready=ready?'true':'false';
}}
function renderConfiguration(value) {{
  const item=value&&typeof value==='object'?value:{{}};
  const status=document.getElementById('configurationStatus');
  const message=document.getElementById('configurationMessage');
  const meta=document.getElementById('configurationMeta');
  if (!status || !message || !meta) return;
  const key=safeText(item.status).toLowerCase()||'unavailable';
  status.dataset.status=key;
  status.textContent=safeText(item.status_label)||'状态未知';
  message.textContent=safeText(item.message)||'配置文件状态暂不可用';
  const generation=Number(item.generation||0);
  const connected=item.connected?'文件已连接':'文件未连接';
  const reload=item.auto_reload?'自动重载已开启':
    (item.watcher_enabled?'自动重载待启动':'自动重载已关闭');
  const applied=Number.isFinite(generation)&&generation>0
    ? ' · 已应用 '+Math.floor(generation)+' 次' : '';
  meta.textContent=connected+' · '+reload+applied;
}}
function renderMemoryStatus(value) {{
  const item=value&&typeof value==='object'?value:{{}};
  const status=document.getElementById('memoryStatus');
  const message=document.getElementById('memoryMessage');
  const meta=document.getElementById('memoryMeta');
  if (!status || !message || !meta) return;
  const key=safeText(item.status).toLowerCase()||'unavailable';
  status.dataset.status=key;
  status.textContent=safeText(item.status_label)||'暂不可用';
  message.textContent=safeText(item.message)||'记忆状态暂不可用';
  const recall=Number(item.recall_limit||0);
  const context=Number(item.context_max_chars||0);
  const maximum=Number(item.max_memories||0);
  const merge=item.consolidation_enabled?'相似记忆合并已开启':'相似记忆合并已关闭';
  const summary=item.summarization_enabled
    ? (item.summary_busy?'正在自动总结':'自动总结已开启'):'自动总结已关闭';
  const index=item.lexical_index==='fts5'?'FTS5 + 向量索引':'稀疏向量索引';
  const indexed=Number(item.vector_index_size||0);
  const extracting=Number(item.extraction_pending||0);
  const parts=[];
  if (recall>0) parts.push('每次召回 '+Math.floor(recall)+' 条');
  if (context>0) parts.push('上下文 '+Math.floor(context)+' 字');
  if (maximum>0) parts.push('最多保存 '+Math.floor(maximum)+' 条');
  if (indexed>0) parts.push('已索引 '+Math.floor(indexed)+' 条');
  if (extracting>0) parts.push('待提取 '+Math.floor(extracting)+' 个回合');
  parts.push(merge);
  parts.push(summary);
  parts.push(index);
  meta.textContent=parts.join(' · ');
}}
function renderActivityStatus(value) {{
  const item=value&&typeof value==='object'?value:{{}};
  const status=document.getElementById('activityStatus');
  const message=document.getElementById('activityMessage');
  const meta=document.getElementById('activityMeta');
  if (!status || !message || !meta) return;
  const key=safeText(item.system_idle_status).toLowerCase()||'disabled';
  status.dataset.status=key;
  status.textContent=safeText(item.system_idle_status_label)||'状态未知';
  const provider=safeText(item.system_idle_provider_label)||'不读取系统状态';
  message.textContent=key==='ready' ? '系统空闲 provider 已就绪，仅用于活跃判定'
    : (key==='disabled' ? '未读取系统空闲状态' : '系统空闲 provider 当前不可用');
  meta.textContent='来源：'+provider;
}}
function renderModelChannels(values) {{
  const host=document.getElementById('channelStatusRows'); if (!host) return;
  host.replaceChildren();
  const rows=Array.isArray(values)?values:[];
  if (!rows.length) {{
    const empty=document.createElement('div'); empty.id='channelStatusEmpty';
    empty.textContent='暂无模型渠道。'; host.appendChild(empty); return;
  }}
  rows.slice(0,32).forEach(row=>{{
    const item=row&&typeof row==='object'?row:{{}};
    const id=safeText(item.id)||'未命名渠道';
    const protocol=safeText(item.protocol)||'自定义渠道';
    const ready=Boolean(item.ready);
    const active=Boolean(item.active);
    const model=safeText(item.model)||(item.model_configured?'模型已配置':'模型未配置');
    const failures=Math.max(0,Number(item.failures||0)||0);
    const status=safeText(item.status)||(ready?'已就绪':'未就绪');
    const line=document.createElement('div'); line.className='channelStatusRow';
    line.dataset.ready=ready?'true':'false';
    const name=document.createElement('span'); name.textContent=id;
    const meta=document.createElement('span'); meta.className='channelMeta';
    meta.textContent=protocol+' · '+model+' · 失败 '+failures+' 次';
    const stateLabel=document.createElement('span'); stateLabel.className='channelReady';
    stateLabel.textContent=active?'当前使用':(status+(ready?' · 可用':' · 不可用'));
    const modelChoices=Array.isArray(item.models)
      ? item.models.map(value=>safeText(value)).filter(Boolean) : [];
    const choices=document.createElement('div'); choices.className='modelChoices';
    const busy=Boolean((state.interaction||{{}}).busy);
    if (modelChoices.length > 1) {{
      modelChoices.slice(0,8).forEach(modelName=>{{
        const selected=active && modelName===model;
        choices.appendChild(button(selected?'当前模型':'使用 '+modelName,
          'select_model_channel', {{channel_id:id,model:modelName}},
          !ready || !Boolean(item.selectable ?? ready) || selected || busy, selected));
      }});
    }} else {{
      choices.appendChild(button(active?'当前使用':'使用此渠道','select_model_channel',
        {{channel_id:id}}, !ready || !Boolean(item.selectable ?? ready) || active || busy, active));
    }}
    line.append(name,meta,stateLabel,choices); host.appendChild(line);
  }});
}}
function showPartFeedback(part) {{
  const key=safeText(part).toLowerCase();
  const text=document.getElementById('feedbackText');
  if (text && partPhrases[key]) text.textContent=(partLabels[key]||'互动')+'：'+partPhrases[key];
  const meta=document.getElementById('feedbackMeta');
  if (meta) meta.textContent='正在回应';
}}
function showOperation(result, fallbackKind='') {{
  let value=result;
  try {{ if (typeof result === 'string') value=JSON.parse(result); }} catch (_) {{
    value={{status:'unavailable',reason:'操作回执格式无效'}};
  }}
  appendTimeline(value,fallbackKind);
  const target=document.getElementById('operation');
  const status=String(value&&value.status||'').toLowerCase();
  localOperationMessage=resultMessage(value);
  localOperationStatus=status;
  localOperationRevision=Number(state.revision||0);
  target.textContent=(operationLabel(fallbackKind)||'操作')+'：'+localOperationMessage;
  target.className='muted '+((status==='completed'||status==='available'||status==='updated')?'ok':
    (status==='unavailable'||status==='failed'||status==='error'||status==='timeout'?'error':'warn'));
  if (bridge && typeof bridge.getState === 'function') bridge.getState(function(next) {{
    try {{ applyState(JSON.parse(next||'{{}}')); }} catch (_) {{}}
  }});
}}
function releaseInvocation(key, source) {{
  pendingInvocations.delete(key);
  if (source) {{
    source.disabled = false;
    source.dataset.pending = 'false';
    if (source.dataset.pendingLabel) {{
      source.textContent = source.dataset.pendingLabel;
      delete source.dataset.pendingLabel;
    }}
  }}
}}
function invoke(kind, payload={{}}, source=null) {{
  const key=actionKey(kind,payload);
  if (pendingInvocations.has(key)) return;
  if (kind==='pet_part') showPartFeedback(payload&&payload.part);
  const localAction=window.meapetControlSurface && window.meapetControlSurface.onAction;
  if ((!bridge || typeof bridge.invoke !== 'function') && typeof localAction !== 'function') {{
    showOperation({{status:'unavailable',message:'控制通道尚未就绪，请稍候再试'}});
    return;
  }}
  pendingInvocations.set(key, ++invocationSequence);
  if (source) {{
    source.disabled=true;
    source.dataset.pending='true';
    source.dataset.pendingLabel=source.textContent;
    source.textContent='处理中…';
  }}
  let settled=false;
  let timeoutId=null;
  const finish = value => {{
    if (settled) return;
    settled=true;
    if (timeoutId !== null) window.clearTimeout(timeoutId);
    releaseInvocation(key,source);
    showOperation(value, kind);
  }};
  try {{
    if (bridge && typeof bridge.invoke === 'function') {{
      bridge.invoke(kind, JSON.stringify(payload), finish);
      timeoutId=window.setTimeout(() => {{
        if (!settled) finish({{status:'timeout',message:'操作响应超时，请重试'}});
      }}, 8000);
    }} else {{
      finish(localAction(kind,payload));
    }}
  }} catch (_) {{
    finish({{status:'unavailable',message:'操作通道暂时不可用，请重试'}});
  }}
}}
function button(label, kind, payload={{}}, disabled=false, primary=false) {{
  const item = document.createElement('button'); item.textContent = label;
  item.disabled = disabled || pendingKind(kind);
  if (primary) item.className = 'primary';
  item.onclick = () => invoke(kind, payload, item); return item;
}}
function renderList(id, values, kind) {{
  const host=document.getElementById(id); host.replaceChildren();
  (values||[]).forEach(value => {{
    const item = value && typeof value === 'object'
      ? value : {{id:value,label:labels[value]||(kind==='expression'?'表情':'动作')}};
    host.appendChild(button(
      safeText(item.label||'动作'), kind, {{name:item.id||''}},
      Boolean((state.interaction||{{}}).busy), false
    ));
  }}); }}
function boundedEditorNumber(id, minimum, maximum, fallback) {{
  const control=document.getElementById(id);
  const value=Number(control&&control.value);
  return Number.isFinite(value)&&value>=minimum&&value<=maximum ? value : fallback;
}}
function parseEditorParameters(id, hintId) {{
  const control=document.getElementById(id);
  const hint=document.getElementById(hintId);
  const source=safeText(control&&control.value).trim();
  if (!source) {{ if (hint) hint.textContent=''; return {{}}; }}
  const items=source.replace(/，/g,',').replace(/\\n/g,',').split(',')
    .map(item=>item.trim()).filter(Boolean);
  if (items.length>32) {{ if(hint)hint.textContent='参数最多 32 项'; return null; }}
  const result={{}};
  for (const item of items) {{
    const split=item.indexOf('=');
    const name=split>=0?item.slice(0,split).trim():'';
    const value=split>=0?Number(item.slice(split+1).trim()):NaN;
    if (!/^[A-Za-z][A-Za-z0-9_.:-]{{0,127}}$/.test(name) ||
        !Number.isFinite(value) || value < -10000 || value > 10000 ||
        Object.prototype.hasOwnProperty.call(result,name)) {{
      if(hint)hint.textContent='参数需使用不重复的 参数名=数值 格式'; return null;
    }}
    result[name]=value;
  }}
  if(hint)hint.textContent=''; return result;
}}
function renderExpressionRequestChoices(values) {{
  const host=document.getElementById('expressionSequenceChoices'); if(!host)return;
  const rows=Array.isArray(values)?values.slice(0,8):[];
  const allowed=new Set(rows.map(item=>safeText(item&&item.id)).filter(Boolean));
  selectedExpressionTokens=selectedExpressionTokens.filter(token=>allowed.has(token));
  if(!selectedExpressionTokens.length&&rows.length) selectedExpressionTokens=[safeText(rows[0].id)];
  host.replaceChildren();
  rows.forEach(item=>{{
    const token=safeText(item&&item.id); if(!token)return;
    const selected=selectedExpressionTokens.includes(token);
    const control=document.createElement('button');
    control.textContent=safeText(item.label)||'表情';
    control.type='button'; control.setAttribute('aria-pressed',selected?'true':'false');
    control.onclick=()=>{{
      if(selectedExpressionTokens.includes(token))
        selectedExpressionTokens=selectedExpressionTokens.filter(value=>value!==token);
      else if(selectedExpressionTokens.length<8) selectedExpressionTokens.push(token);
      renderExpressionRequestChoices(rows);
    }};
    host.appendChild(control);
  }});
}}
function renderMotionRequestChoices(values) {{
  const host=document.getElementById('motionRequestChoices'); if(!host)return;
  const rows=Array.isArray(values)?values.slice(0,16):[];
  const allowed=new Set(rows.map(item=>safeText(item&&item.id)).filter(Boolean));
  if(!allowed.has(selectedMotionToken)) selectedMotionToken=rows.length?safeText(rows[0].id):'';
  host.replaceChildren();
  rows.forEach(item=>{{
    const token=safeText(item&&item.id); if(!token)return;
    const control=document.createElement('button');
    control.textContent=safeText(item.label)||'动作'; control.type='button';
    control.setAttribute('aria-pressed',token===selectedMotionToken?'true':'false');
    control.onclick=()=>{{ selectedMotionToken=token; renderMotionRequestChoices(rows); }};
    host.appendChild(control);
  }});
}}
function render() {{
  const i=state.interaction||{{}}, r=state.renderer||{{}},
    a=state.affection||{{}}, c=state.capabilities||{{}}, w=state.window||{{}},
    t=state.tts||{{}}, modelState=i.model||{{}};
  document.getElementById('renderer').textContent='渲染：'+(r.label||'未知后端')+
    (r.available?' · 已启用':' · 不可用');
  document.getElementById('rendererMessage').textContent=r.available?'':(r.message||'请改用自动或精灵后端');
  const rendererLabel=safeText(r.label).toLowerCase();
  document.querySelectorAll('[data-renderer-backend]').forEach(control=>{{
    const backend=safeText(control.dataset.rendererBackend).toLowerCase();
    const selected=(backend==='opengl'&&rendererLabel.includes('opengl'))||
      (backend==='vulkan'&&rendererLabel.includes('vulkan'))||
      (backend==='auto'&&rendererLabel.includes('自动'));
    control.disabled=selected||Boolean(i.busy)||pendingKind('select_renderer_backend');
    control.setAttribute('aria-pressed',selected?'true':'false');
  }});
  const rendererBackendHint=document.getElementById('rendererBackendHint');
  if(rendererBackendHint) rendererBackendHint.textContent=
    (state.configuration||{{}}).status==='restart_required'
      ? '新引擎已保存，请点击“重启并应用引擎”' : '切换后需保存并重启生效';
  document.getElementById('connection').textContent=bridgeReady
    ? ((state.connection||{{}}).message||'本地控制通道') : '控制通道准备中…';
  document.getElementById('phase').textContent='状态：'+phaseLabel(i.phase)+(i.busy?' · 处理中':'');
  const interactionLabel=w.locked
    ? '已锁定 · 不接收点击，仍追踪光标'
    : (w.click_through?'点击穿透已开启':'可互动');
  const topmostStatus=w.always_on_top_status||{{}};
  const topmostPending=topmostStatus.status==='requested' || pendingKind('toggle_always_on_top');
  const topmostBoundary=topmostStatus.status==='requested'
    ? ' · 正在应用置顶'
    : (topmostStatus.status==='degraded'
      ? ' · 置顶由桌面合成器决定'
      : (topmostStatus.status==='unavailable'?' · 置顶未确认':''));
  // 点击穿透/锁定时 input_ready=false 是用户主动选择的状态，不是 Shape
  // 恢复中的异常；只有可互动模式下才显示“恢复中”，避免误导用户反复点击。
  const shapePending=Boolean(!w.click_through&&!w.locked&&w.input_shape&&
    w.input_shape.ready&&w.input_shape.input_ready===false);
  document.getElementById('windowState').textContent=interactionLabel+
    (w.always_on_top?' · 已置顶':'')+topmostBoundary+
    (shapePending?' · 点击区域恢复中':'');
  const lockButton=document.querySelector('[data-action="toggle_window_lock"]');
  if (lockButton) lockButton.textContent=w.locked?'解锁窗口':'锁定窗口';
  const topButton=document.querySelector('[data-action="toggle_always_on_top"]');
  if (topButton) {{
    topButton.textContent=w.always_on_top?'取消置顶':'保持置顶';
    topButton.title=topmostStatus.detail||'切换桌宠窗口置顶状态';
    topButton.disabled=topmostPending;
    topButton.dataset.pending=topmostPending?'true':'false';
  }}
  const clickButton=document.getElementById('clickThroughButton');
  if (clickButton) {{
    clickButton.disabled=Boolean(w.locked)||pendingKind('toggle_click_through');
    clickButton.textContent=w.locked?'先解锁窗口':(w.click_through?'恢复点击':'开启点击穿透');
    clickButton.title=w.locked?'解锁后才能恢复点击':'切换桌宠是否接收鼠标';
  }}
  const position=w.position||{{}};
  const positionState=document.getElementById('positionState');
  if (positionState) positionState.textContent=(
    position.x != null && position.y != null
      ? '位置：('+safeText(position.x)+', '+safeText(position.y)+')'
      : '位置：暂不可读'
  );
  document.getElementById('output').textContent=i.text||i.safe_message||'暂无回复';
  document.getElementById('murmur').textContent=i.murmur?'碎碎念：'+i.murmur:'';
  document.getElementById('tool').textContent=i.tool_status||'';
  document.getElementById('tts').textContent=t.message?('语音：'+t.message):'';
  const ttsState=document.getElementById('ttsState');
  if (ttsState) ttsState.textContent=t.message||'语音待命';
  const ttsHealth=document.getElementById('ttsHealth');
  const health=t.health&&typeof t.health==='object'?t.health:{{}};
  if(ttsHealth) {{
    const latency=Number(health.latency_ms);
    const latencyLabel=Number.isFinite(latency)&&latency>=0?' · '+Math.round(latency)+'ms':'';
    ttsHealth.textContent=(safeText(health.backend)||safeText(t.backend)||'语音后端待检测')+
      ' · '+(health.pending?'初始化中':
        (health.checked?(health.available?'已就绪':'未就绪'):'待检测'))+latencyLabel+
      (health.message?' · '+safeText(health.message):'');
    ttsHealth.className='muted '+(health.pending?'warn':
      (health.checked?(health.available?'ok':'error'):'warn'));
  }}
  const languageHost=document.getElementById('ttsLanguages');
  if (languageHost) {{
    languageHost.replaceChildren();
    const languageNames={{zh:'中文',jp:'日文',en:'英文'}};
    ['zh','jp','en'].forEach(language=>{{
      const selected=safeText(t.language).toLowerCase()===language;
      languageHost.appendChild(button(
        selected?'当前'+(languageNames[language]||language):(languageNames[language]||language),
        'select_tts_language',{{language}},selected||Boolean(i.busy),selected));
    }});
  }}
  const profileHost=document.getElementById('ttsProfiles');
  if (profileHost) {{
    profileHost.replaceChildren();
    const profiles=Array.isArray(t.profiles)?t.profiles:[];
    profiles.slice(0,16).forEach(profile=>{{
      const id=safeText(profile&&profile.id); if(!id) return;
      const selected=id===safeText(t.profile);
      const languageNames={{zh:'中文',jp:'日文',ja:'日文',en:'英文'}};
      const languageLabel=(Array.isArray(profile.languages)?profile.languages:[])
        .map(value=>languageNames[safeText(value).toLowerCase()]||safeText(value).toUpperCase())
        .filter(Boolean).join('/');
      profileHost.appendChild(button((selected?'当前音色':'使用 '+id)+
        (languageLabel?' · '+languageLabel:''),
        'select_tts_profile',{{profile:id}},selected||!Boolean(profile.enabled)||Boolean(i.busy),selected));
    }});
    if (!profiles.length) {{
      const note=document.createElement('span'); note.className='muted';
      note.textContent='未启用多语音 profile，可在配置中心添加'; profileHost.appendChild(note);
    }}
  }}
  const rendererModels=document.getElementById('rendererModels');
  const rendererModelStatus=document.getElementById('rendererModelStatus');
  const rendererModelsList=Array.isArray(r.models)?r.models.filter(Boolean):[];
  if (rendererModelStatus) rendererModelStatus.textContent=r.model
    ? ('当前：'+safeText(r.model)) : '当前模型未选择';
  if (rendererModels) rendererModels.textContent=rendererModelsList.length
    ? ('可选 '+rendererModelsList.length+' 个；修改配置后重启桌宠生效')
    : '暂无通过校验的 Live2D 模型';
  const modelCard=document.getElementById('modelCard');
  const modelActions=document.getElementById('modelActions');
  const modelAction=(modelState.action||[])[0];
  const modelNeedsSetup=modelState.ready===false || Boolean(modelAction);
  modelCard.style.display=modelNeedsSetup?'block':'none';
  document.getElementById('modelMessage').textContent=modelNeedsSetup
    ? (modelState.message||'模型尚未连接，请点击“配置模型”完成连接') : '';
  modelActions.replaceChildren();
  if (modelNeedsSetup) {{
    modelActions.appendChild(button(
      modelAction?.label||'配置模型', modelAction?.kind||'configure_model',
      modelAction?.payload||{{}}, Boolean(i.busy), true
    ));
  }}
  renderModelImageStatus(state.model_image);
  renderConfiguration(state.configuration);
  renderMemoryStatus(state.memory);
  renderActivityStatus(state.activity);
  renderModelChannels(state.model_channels);
  const operation=state.operation||{{}};
  const operationTarget=document.getElementById('operation');
  const currentRevision=Number(state.revision||0);
  if (localOperationMessage && localOperationRevision === currentRevision) {{
    operationTarget.textContent='操作：'+localOperationMessage;
    operationTarget.className='muted '+(
      (localOperationStatus==='completed'||localOperationStatus==='available'||
       localOperationStatus==='updated')?'ok':
      (localOperationStatus==='unavailable'||localOperationStatus==='failed'||
       localOperationStatus==='error'||localOperationStatus==='timeout'?'error':'warn')
    );
  }} else {{
    localOperationMessage='';
    localOperationStatus='';
    localOperationRevision=null;
    if (operation.message) operationTarget.textContent=
      (operation.label||'操作')+'：'+operation.message;
  }}
  mergeTimeline(state.timeline);
  renderObservation(state.observation);
  document.getElementById('affection').value=Number(a.current||0)||0;
  document.getElementById('affectionText').textContent=(a.current||'0')+' · '+(a.tier||'');
  const actions=document.getElementById('stateActions'); actions.replaceChildren();
  const approval=i.approval || ((i.pending_approvals||[])[0]);
  const approvalKinds=new Set(['approve','grant_session','deny']);
  (i.actions||[]).filter(x=>!approval || !approvalKinds.has(String(x.kind||'').toLowerCase()))
    .forEach(x=>actions.appendChild(button(x.label,x.kind,x.payload||{{}},!x.enabled||Boolean(i.busy),true)));
  const parts=document.getElementById('parts'); parts.replaceChildren();
  (state.parts||[]).forEach(x=>parts.appendChild(button(x.label,'pet_part',{{part:x.id}},!x.enabled||Boolean(w.locked))));
  renderList('expressions',c.expressions,'expression'); renderList('motions',c.motions,'motion');
  renderExpressionRequestChoices(c.expressions);
  renderMotionRequestChoices(c.motions);
  const expressionRequestButton=document.getElementById('playExpressionRequest');
  if(expressionRequestButton) expressionRequestButton.disabled=Boolean(i.busy)||
    !selectedExpressionTokens.length||pendingKind('expression_request');
  const motionRequestButton=document.getElementById('playMotionRequest');
  if(motionRequestButton) motionRequestButton.disabled=Boolean(i.busy)||
    !selectedMotionToken||pendingKind('motion_request');
  const feedback=state.feedback||{{}};
  document.getElementById('feedbackText').textContent=feedback.phrase
    ? (partLabels[feedback.part||feedback.zone]||'互动')+'：'+feedback.phrase+' · '
      +(feedback.expression?('表情 '+safeText(feedback.expression)+'，'):'')
      +(feedback.motion?('动作 '+safeText(feedback.motion)):'')
    : '点击桌宠部位后，反馈会显示在这里。';
  const fa=feedback.affection||{{}};
  document.getElementById('feedbackMeta').textContent=fa.current
    ? ('好感度 '+fa.current+(fa.applied?'（本次 '+fa.applied+'）':'')):'';
  const approvalCard=document.getElementById('approvalCard');
  approvalCard.style.display=approval?'block':'none';
  if (approval) {{
    document.getElementById('approvalSummary').textContent=(approval.display_name||'工具操作')
      +'：'+(approval.safe_summary||'请确认桌面操作');
    const remaining=Math.max(0,Math.min(3600,Number(approval.remaining_seconds != null
      ? approval.remaining_seconds : Number(approval.expires_at||0)-Date.now()/1000)||0));
    document.getElementById('approvalCountdown').textContent=remaining>0
      ? ('剩余 '+Math.ceil(remaining)+' 秒') : '确认已过期';
    const host=document.getElementById('approvalActions'); host.replaceChildren();
    [['approve','允许'],['grant_session','本次会话允许'],['deny','拒绝']].forEach(x=>
      host.appendChild(button(x[1],x[0],{{}},Boolean(i.busy)||remaining<=0,x[0]!=='deny'))); }}
  document.getElementById('stopButton').disabled=!i.busy || pendingKind('stop');
  document.getElementById('retryButton').disabled=Boolean(i.busy)||!i.retryable||pendingKind('retry');
  document.getElementById('sendButton').disabled=Boolean(i.busy)||pendingKind('submit_text');
}}
const navigationLinks=[...document.querySelectorAll('.navigationPane a[href^="#"]')];
function selectNavigation(link) {{
  navigationLinks.forEach(item=>item.removeAttribute('aria-current'));
  if(link) link.setAttribute('aria-current','page');
}}
navigationLinks.forEach(link=>link.addEventListener('click',()=>selectNavigation(link)));
if('IntersectionObserver' in window) {{
  const sectionLinks=new Map(navigationLinks.map(link=>[
    document.querySelector(link.getAttribute('href')),link]).filter(row=>row[0]));
  const observer=new IntersectionObserver(entries=>{{
    const visible=entries.filter(entry=>entry.isIntersecting)
      .sort((left,right)=>right.intersectionRatio-left.intersectionRatio)[0];
    if(visible) selectNavigation(sectionLinks.get(visible.target));
  }},{{rootMargin:'-15% 0px -70% 0px',threshold:[0,.25,.6]}});
  sectionLinks.forEach((_link,section)=>observer.observe(section));
}}
document.querySelectorAll('[data-action]').forEach(x=>x.onclick=()=>{{
  const payload=x.dataset.direction ? {{direction:x.dataset.direction}} : {{}};
  invoke(x.dataset.action,payload,x);
}});
document.querySelectorAll('[data-preset]').forEach(x=>x.onclick=()=>
  invoke('set_display_size',{{preset:x.dataset.preset}},x));
document.querySelectorAll('[data-renderer-backend]').forEach(x=>x.onclick=()=>
  invoke('select_renderer_backend',{{backend:x.dataset.rendererBackend}},x));
document.getElementById('playExpressionRequest').onclick=()=>{{
  const hint=document.getElementById('expressionRequestHint');
  if(!selectedExpressionTokens.length) {{ hint.textContent='请至少选择一个表情'; return; }}
  const parameters=parseEditorParameters('expressionRequestParameters','expressionRequestHint');
  if(parameters===null) return;
  const weightsSource=safeText(document.getElementById('expressionRequestWeights').value).trim();
  let weights=[];
  if(weightsSource) {{
    weights=weightsSource.replace(/，/g,',').split(',').map(value=>Number(value.trim()));
    if(weights.length!==selectedExpressionTokens.length ||
       weights.some(value=>!Number.isFinite(value)||value<0||value>1)) {{
      hint.textContent='权重数量需与表情一致，且每项范围为 0 到 1'; return;
    }}
  }}
  const duration=boundedEditorNumber('expressionRequestDuration',0.05,120,1.5);
  const transition=boundedEditorNumber('expressionRequestTransition',0,10,0.2);
  const mode=safeText(document.getElementById('expressionRequestMode').value).toLowerCase();
  if(mode==='blend' && weights.length && weights.every(value=>value===0)) {{
    hint.textContent='参数混合至少需要一个大于 0 的权重'; return;
  }}
  hint.textContent='正在提交表情编排…';
  invoke('expression_request',{{
    mode:mode==='blend'?'blend':'sequence',
    loop:Boolean(document.getElementById('expressionRequestLoop').checked),
    expressions:selectedExpressionTokens.map((name,index)=>({{
      name,
      weight:weights.length?weights[index]:1,
      duration_seconds:duration,
      transition_seconds:transition,
      parameters,
    }})),
  }},document.getElementById('playExpressionRequest'));
}};
document.getElementById('playMotionRequest').onclick=()=>{{
  const hint=document.getElementById('motionRequestHint');
  if(!selectedMotionToken) {{ hint.textContent='请先选择一个动作'; return; }}
  const parameters=parseEditorParameters('motionRequestParameters','motionRequestHint');
  if(parameters===null) return;
  hint.textContent='正在提交动作编排…';
  invoke('motion_request',{{
    name:selectedMotionToken,
    duration_seconds:boundedEditorNumber('motionRequestDuration',0.05,300,1.5),
    transition_seconds:boundedEditorNumber('motionRequestTransition',0,10,0.2),
    loop:Boolean(document.getElementById('motionRequestLoop').checked),
    parameters,
  }},document.getElementById('playMotionRequest'));
}};
document.getElementById('sendButton').onclick=()=>{{
  const input=document.getElementById('messageInput'); const value=input.value.trim();
  if(value){{ input.value=''; invoke('submit_text',{{text:value}},
    document.getElementById('sendButton')); }}
}};
document.getElementById('messageInput').addEventListener('keydown',event=>{{if(event.key==='Enter')document.getElementById('sendButton').click();}});
document.getElementById('stopButton').onclick=()=>invoke('stop',{{}},document.getElementById('stopButton'));
document.getElementById('retryButton').onclick=()=>invoke('retry',{{}},document.getElementById('retryButton'));
document.getElementById('clearTimeline').onclick=()=>{{
  timelineEntries=[]; renderTimeline();
  const target=document.getElementById('operation');
  if (target) target.textContent='活动记录已清空';
}};
setInterval(()=>{{if((state.interaction||{{}}).approval)render();}},1000);
window.meapetControlSurface={{setState(next){{
  applyState(next);
}},onAction:null}}; applyTheme(state.theme); render();
if (window.qt && window.qt.webChannelTransport && window.QWebChannel) {{
  new QWebChannel(window.qt.webChannelTransport, channel => {{
    bridge = channel.objects.meapetControlSurfaceBridge || null;
    bridgeReady = Boolean(bridge && typeof bridge.invoke === 'function');
    render();
  }});
}}
</script></body></html>"""


__all__ = [
    "control_surface_html",
    "friendly_public_text",
    "public_control_state",
    "safe_public_text",
]
