"""安全 YAML 配置加载、合并、环境变量展开与脱敏。"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.adapters.protocols import SUPPORTED_PROTOCOLS
from core.conversation.persona import default_persona_prompt_values

_ENVIRONMENT_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_ENV_NAME_PATTERN = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
_FORBIDDEN_KEYS = frozenset({"__proto__", "constructor", "prototype"})
_SECRET_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|credential|token|secret|password|cookie|authorization)",
    re.IGNORECASE,
)
MAX_CONFIGURATION_BYTES = 4 * 1024 * 1024


class ConfigurationError(ValueError):
    """配置结构、路径或密钥引用不满足契约。"""


def _read_configuration_payload(path: Path) -> bytes:
    """有界读取配置文件，避免加载或冲突检查吞入任意大文件。"""

    try:
        if path.stat().st_size > MAX_CONFIGURATION_BYTES:
            raise ConfigurationError("configuration file exceeds size limit")
        payload = path.read_bytes()
    except ConfigurationError:
        raise
    except OSError as exc:
        raise ConfigurationError("configuration file cannot be read") from exc
    if len(payload) > MAX_CONFIGURATION_BYTES:
        raise ConfigurationError("configuration file exceeds size limit")
    return payload


@dataclass(frozen=True)
class LoadedConfiguration:
    path: Path
    values: Mapping[str, Any]
    source_digest: str | None = field(default=None, repr=False, compare=False)

    @property
    def directory(self) -> Path:
        return self.path.parent


def configuration_source_digest(path: str | Path) -> str:
    """返回配置文件内容摘要，供编辑会话检测外部并发修改。"""

    target = Path(path).expanduser().resolve()
    payload = _read_configuration_payload(target)
    return hashlib.sha256(payload).hexdigest()


def default_configuration_values(
    *,
    resource_root: str | Path | None = None,
    database_path: str | Path | None = None,
) -> dict[str, Any]:
    """返回首次启动可安全使用的配置。

    默认配置不包含渠道、密钥或高风险命令白名单。资源和数据库路径由启动
    发现器注入绝对路径，避免用户配置位于 ``~/.config`` 时错误地把相对路径
    解析到用户配置目录。
    """

    resource_value = str(resource_root) if resource_root is not None else "./resources"
    # 参考音频必须与资源根目录同步；首次配置通常写入 ~/.config，使用相对
    # `../resources` 会错误地指向用户配置目录旁边。
    gsv_reference_dir = (
        str(Path(resource_root).expanduser().resolve() / "GPT-Sovits")
        if resource_root is not None
        else "./resources/GPT-Sovits"
    )
    built_in_engine_root = ""
    built_in_python = ""
    built_in_gpt_path = ""
    built_in_sovits_path = ""
    built_in_asr_python = ""
    built_in_asr_model = ""
    if resource_root is not None:
        resolved_resources = Path(resource_root).expanduser().resolve()
        resource_parent = resolved_resources.parent
        # 当前源码交付同时支持 ``resources`` 与 ``temp/resources`` 两个已定义
        # 资源布局；只在完整引擎、独立解释器和配置均存在时启用内置模式。
        engine_root = (
            resource_parent / "third_party" / "GPT-SoVITS"
            if resource_parent.name == "temp"
            else resource_parent / "temp" / "third_party" / "GPT-SoVITS"
        )
        is_windows = sys.platform.startswith("win") or os.name == "nt"
        # Python venv 的 Windows 入口是 ``Scripts/python.exe``；POSIX
        # 入口保持既有的 ``bin/python``，不把 pythonw.exe 用作 stdin worker。
        engine_python_path = (
            engine_root / ".venv" / "Scripts" / "python.exe"
            if is_windows
            else engine_root / ".venv" / "bin" / "python"
        )
        engine_python = (
            engine_python_path
            if engine_python_path.is_file()
            and (is_windows or os.access(engine_python_path, os.X_OK))
            else None
        )
        engine_config = engine_root / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
        gpt_path = resolved_resources / "models" / "GPT_weights" / "mea_pro-e50.ckpt"
        sovits_path = resolved_resources / "models" / "SoVITS_weights" / "mea_pro_e24_s13704.pth"
        if engine_root.is_dir() and engine_python is not None and engine_config.is_file():
            built_in_engine_root = str(engine_root)
            built_in_python = str(engine_python)
        if gpt_path.is_file():
            built_in_gpt_path = str(gpt_path)
        if sovits_path.is_file():
            built_in_sovits_path = str(sovits_path)
        if engine_python is not None:
            built_in_asr_python = str(engine_python)
        asr_model = (
            Path.home()
            / ".cache"
            / "modelscope"
            / "models"
            / "iic--SenseVoiceSmall"
            / "snapshots"
            / "master"
        )
        if asr_model.is_dir() and all(
            (asr_model / filename).is_file()
            for filename in ("model.pt", "config.yaml", "tokens.json", "am.mvn")
        ):
            built_in_asr_model = str(asr_model)
    database_value = str(database_path) if database_path is not None else "./data/meapet.sqlite3"
    return {
        "app": {"name": "MeaPet", **default_persona_prompt_values()},
        "logging": {
            "level": "INFO",
            "console": {"enabled": True, "level": "INFO", "color": "auto"},
            "file": {
                "enabled": False,
                "path": "./data/meapet.log",
                "level": "INFO",
                "rotation": "time",
                "when": "midnight",
                "interval": 1,
                "backup_count": 7,
                "max_bytes": 10 * 1024 * 1024,
            },
            "memory": {
                "enabled": True,
                "level": "INFO",
                "capacity": 1000,
            },
        },
        "llm": {"channels": [], "models": {}, "routing": {}},
        "tools": {
            "groups": {
                "desktop_observation": [
                    "desktop:observe_foreground",
                    "desktop:cursor_position",
                    "desktop:list_processes",
                    "desktop:context_snapshot",
                    "desktop:capture_screen",
                    "desktop:ocr",
                ],
                "desktop_control": ["desktop:click_at"],
                "desktop_automation": ["desktop:automation_batch"],
                "pet_control": [
                    "pet:move",
                    "pet:set_expression",
                    "pet:play_motion",
                    "pet:speak",
                    "pet:set_click_through",
                    "pet:list_models",
                    "pet:switch_model",
                ],
                "pet_diary": ["pet:diary_write", "pet:diary_recall"],
                "scheduler": [
                    "scheduler:upsert",
                    "scheduler:list",
                    "scheduler:remove",
                    "scheduler:set_trigger",
                ],
                "system": [
                    "system:module_status",
                    "system:transcribe_audio",
                    "system:run_command",
                ],
            },
            "permissions": {
                "bypass_approval": False,
                "auto_allow_low_risk": True,
                "allow": [],
                "deny": [],
                "approval_ttl_seconds": 90,
                "xml_approve_enabled": False,
                # XML approve 有界放行窗口：只覆盖当回合被拦截的身份，
                # 风险不高于 MEDIUM；过期或会话不匹配时审批照旧。
                "xml_approve_window_seconds": 15,
            },
            "command_allowlist": [],
        },
        "mcp": {"enabled": False, "servers": []},
        # 外部插件默认启用生命周期管理，但不主动扫描目录；用户可在配置中
        # 显式填写相对于配置文件目录的 ``directories``，并按需打开入口点。
        "plugins": {
            "enabled": True,
            "directories": [],
            "entry_points_enabled": False,
            "entry_point_group": "meapet.plugins",
            "reload_enabled": True,
            "reload_interval_seconds": 1.0,
        },
        # 本地 HTTP 控制面默认完全关闭；启用后仍由运行宿主强制回环绑定，
        # capability 令牌只在进程内生成，不写入 YAML。
        "web": {
            "local_api": {
                "enabled": False,
                "host": "127.0.0.1",
                "port": 0,
                "events": True,
                "event_limit": 64,
            }
        },
        # 长期记忆策略。内容保存在 SQLite，以下参数可在运行期热重载；
        # 关闭后对话不读取或自动写入记忆，但控制台仍可手动维护已有数据。
        "memory": {
            "enabled": True,
            "recall_limit": 7,
            "context_max_chars": 6000,
            "context_memory_item_chars": 500,
            "recent_exchange_limit": 10,
            "max_memories": 2000,
            "decay_days": 7,
            "prune_days": 30,
            "prune_importance_floor": 1,
            "consolidation_enabled": True,
            "consolidation_similarity": 0.85,
            "max_consolidation_memories": 200,
            "summarize_every_n": 20,
            "summary_chat_limit": 20,
            "summarization_enabled": True,
            "summarize_daily": True,
            "summary_interval_minutes": 360,
            "summary_min_messages": 4,
            "exchange_min_chars": 20,
            "exchange_importance": 2,
            "exchange_decay_factor": 0.8,
            "auto_extract_enabled": True,
            "extract_max_items": 3,
            "extract_max_chars": 240,
            "extract_min_confidence": 0.8,
            "extract_default_priority": 5,
            "recall_min_similarity": 0.04,
            "always_recall_priority": 9,
            # 真实语义索引默认关闭；启用时必须显式提供本地模型目录、ID
            # 和固定 revision，不会隐式联网下载。
            "semantic": {
                "enabled": False,
                "provider": "sentence_transformers",
                "model_path": "",
                "model_id": "",
                "model_revision": "",
                "dimension": 384,
                "batch_size": 16,
                "timeout_seconds": 5.0,
                "startup_timeout_seconds": 60.0,
                "shutdown_timeout_seconds": 2.0,
                "max_text_chars": 10_000,
                "max_text_bytes": 40_000,
                "device": "cpu",
                "backend": "torch",
                "index_dir": "",
                "hnsw_m": 16,
                "hnsw_ef_construction": 200,
                "hnsw_ef_search": 64,
            },
        },
        "tts": {
            # 完整源码引擎存在时默认由持久 JSONL IPC worker 直接预载模型；
            # 否则保留 endpoint 适配，任一模式不可用都安全降级为文字。
            "enabled": True,
            "backend": "gpt_sovits_stdio",
            "endpoint": "http://127.0.0.1:9880/tts",
            "engine_root": built_in_engine_root,
            "engine_config": "GPT_SoVITS/configs/tts_infer.yaml",
            "python_executable": built_in_python,
            "gpt_path": built_in_gpt_path,
            "sovits_path": built_in_sovits_path,
            "ref_dir": gsv_reference_dir,
            "language": "zh",
            "role": "",
            # 空值跟随系统默认扬声器；填写 Qt QAudioDevice.id() 可固定
            # Windows/Linux 的物理输出设备。
            "output_device_id": "",
            # GPT-SoVITS api_v2 默认输出 32000Hz；设为 null 可为其它合法
            # WAV/PCM 服务关闭采样率锁定，裸 PCM 仍需用 raw_sample_rate 指定。
            "expected_sample_rate": 32000,
            "queue_size": 16,
            "segmentation": {
                "max_chars": 120,
                "hard_boundaries": "。！？!?…｡．.\n\r",
                "soft_boundaries": "，,、；;：: \t",
            },
            "audio_features": {
                "enabled": True,
                "silence_threshold": 0.02,
                "open_threshold": 0.35,
            },
            "streaming_mode": 3,
            "health_probe": True,
            "startup_handshake": True,
            "startup_timeout_seconds": 300.0,
            "shutdown_timeout_seconds": 2.0,
        },
        "asr": {
            # SenseVoice 在独立 Python 3.11 worker 中常驻；缺少本地解释器或
            # 模型时默认关闭，显式启用后的初始化失败也不会阻止桌宠启动。
            "enabled": bool(built_in_asr_python and built_in_asr_model),
            "backend": "sensevoice",
            "python_executable": built_in_asr_python,
            "model_path": built_in_asr_model,
            "device": "cpu",
            # 当前中文参考音频在 auto 下实测误判为 ja；默认锁定中文，
            # 用户仍可显式选择 auto/ja/en/yue/ko/nospeech。
            "language": "zh",
            "timeout_seconds": 60.0,
            "startup_timeout_seconds": 180.0,
            "max_audio_bytes": 16 * 1024 * 1024,
            "capture": {
                # 只有用户点击控制台录音按钮后才检查/申请权限并打开设备。
                "enabled": True,
                "max_duration_seconds": 15.0,
                "auto_submit": False,
                "device_id": "",
            },
        },
        "rendering": {
            "backend": "auto",
            "resource_root": resource_value,
            # 相对于资源根目录的精确 model3 路径；留空自动选择首个可用模型。
            "model": "",
            "sprite_scale": 0.6,
            "frame_rate": 60.0,
            "geometry_audit_hz": 30.0,
        },
        "watcher": {
            "enabled": False,
            "interval_seconds": 2,
            "poll_timeout_seconds": 5,
            "emit_initial": False,
        },
        # 配置文件热重载独立于前台窗口观察；默认开启，采用轮询和稳定
        # 检查避免编辑器半写入。关闭后仍可通过控制台保存并手动应用。
        "config": {
            "reload": {
                "enabled": True,
                "interval_seconds": 0.5,
                "debounce_seconds": 0.1,
                "stable_checks": 2,
            }
        },
        "scheduler": {
            "enabled": True,
            "poll_seconds": 0.5,
            "activity": {
                "enabled": True,
                "idle_seconds": 300,
                "poll_seconds": 0.5,
                # 默认不读取系统空闲状态；x11 使用 MIT-SCREEN-SAVER，Windows 使用 Win32
                # GetLastInputInfo；平台不匹配或异常时 fail-closed。
                "system_idle_provider": "disabled",
                "system_idle_threshold_seconds": 300,
            },
            "tasks": [],
            "triggers": [],
        },
        "behavior": {
            "enabled": False,
            "min_interval_seconds": 8,
            "max_interval_seconds": 20,
            "probability": 0.35,
            "interaction_cooldown_seconds": 0.35,
            "movement": {
                "enabled": False,
                "min_distance_px": 80,
                "max_distance_px": 280,
                "step_pixels": 36,
                "step_interval_seconds": 0.08,
                "max_speed_px_per_second": 240,
                "moving_motion": "walk",
                "idle_motion": "idle",
                "movement_identity": "pet:autonomous_move",
                "flip_facing": True,
                "max_steps": 64,
            },
            "actions": [],
        },
        # 情绪状态：词汇集统一为 9 值；陈旧心情由模型按人设判断是否更新，
        # proactive 按心情系数调整背景主动频率。三项均支持运行期热重载。
        "mood": {
            "decay_enabled": True,
            "decay_after_seconds": 14400,
            "decay_prompt_probability": 0.5,
            "proactive_mood_gate": {
                "enabled": True,
                "multipliers": {
                    "烦躁": 0.5,
                    "生气": 0.5,
                    "难过": 0.5,
                    "高兴": 1.2,
                    "期待": 1.2,
                    "好奇": 1.2,
                },
            },
        },
        "proactive": {
            "enabled": False,
            "hourly_budget": 6,
            "daily_budget": 24,
            "global_cooldown_seconds": 15,
            "dedupe_seconds": 60,
            "max_pending_events": 32,
            "memory_context_chars": 2400,
            "max_tool_rounds": 2,
            "rules": [],
        },
        "storage": {"database": database_value},
        "ui": {
            "always_on_top": True,
            "window_locked": False,
            "theme": {
                "name": "MeaPet Fluent Dark",
                "mode": "dark",
                "seed": None,
                "roles": {},
                "typography": {},
                "layout": {"density": 0},
                "motion": {"reduced_motion": False},
                "extra_stylesheet": "",
            },
            # auto 在同时存在 Xwayland 时优先 xcb；无 X11 时才使用原生 Wayland。
            "qt_platform": "auto",
            "auto_open_model_setup": True,
            "hotkeys": {
                "enabled": True,
                "bindings": [
                    {"action": "open_input", "sequence": "Ctrl+Return", "mode": "hold"},
                    {"action": "open_console", "sequence": "Ctrl+Shift+M", "mode": "press"},
                    {"action": "toggle_visibility", "sequence": "Ctrl+Shift+H", "mode": "press"},
                    {
                        "action": "toggle_always_on_top",
                        "sequence": "Ctrl+Shift+T",
                        "mode": "press",
                    },
                    {
                        "action": "toggle_window_lock",
                        "sequence": "Ctrl+Shift+L",
                        "mode": "press",
                    },
                    {
                        "action": "restore_click_through",
                        "sequence": "Ctrl+Alt+Space",
                        "mode": "hold",
                    },
                ],
            },
            "stream": {
                # 推理内容默认不公开；显式开启后由统一快照作为碎碎念展示。
                "show_reasoning": False,
                "show_murmur": True,
                "show_tool_status": True,
                "max_bubble_chars": 4000,
            },
        },
    }


def parse_bool(value: object, *, field_name: str, default: bool | None = None) -> bool:
    """严格解析 YAML 布尔值，避免 ``bool(\"false\")`` 被误判为真。"""

    if value is None and default is not None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "on", "1"}:
        return True
    if normalized in {"false", "no", "off", "0"}:
        return False
    raise ConfigurationError(f"{field_name} must be a boolean")


def _ensure_mapping(value: object, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _reject_forbidden_keys(
    value: object,
    path: str = "",
    _active: set[int] | None = None,
) -> None:
    if isinstance(value, Mapping):
        active = _active if _active is not None else set()
        identity = id(value)
        if identity in active:
            raise ConfigurationError("configuration contains recursive aliases")
        active.add(identity)
        try:
            for raw_key, nested in value.items():
                key = str(raw_key)
                location = f"{path}.{key}" if path else key
                if key.lower() in _FORBIDDEN_KEYS:
                    raise ConfigurationError(f"forbidden configuration key: {location}")
                _reject_forbidden_keys(nested, location, active)
        finally:
            active.remove(identity)
    elif isinstance(value, list):
        active = _active if _active is not None else set()
        identity = id(value)
        if identity in active:
            raise ConfigurationError("configuration contains recursive aliases")
        active.add(identity)
        try:
            for index, nested in enumerate(value):
                _reject_forbidden_keys(nested, f"{path}[{index}]", active)
        finally:
            active.remove(identity)


def _expand_environment(value: object, environment: Mapping[str, str]) -> object:
    if isinstance(value, Mapping):
        return {str(key): _expand_environment(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item, environment) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in environment:
            raise ConfigurationError(f"environment variable is not set: {name}")
        return environment[name]

    return _ENVIRONMENT_PATTERN.sub(replace, value)


def _environment_value(environment: Mapping[str, str], *names: str) -> str:
    """按明确的环境变量名称读取第一个非空值。"""

    for name in names:
        value = str(environment.get(name, "") or "").strip()
        if value:
            return value
    return ""


def _hydrate_explicit_channel_keys(
    values: MutableMapping[str, Any], environment: Mapping[str, str]
) -> None:
    """按渠道显式 ``api_key_env`` 在内存中补齐密钥，不猜测供应商。"""

    llm_value = values.get("llm")
    if not isinstance(llm_value, MutableMapping):
        return
    channels = llm_value.get("channels")
    if not isinstance(channels, list):
        return
    for channel in channels:
        if not isinstance(channel, MutableMapping):
            continue
        if str(channel.get("api_key", "") or "").strip():
            continue
        env_name = str(channel.get("api_key_env", "") or "").strip()
        if _ENV_NAME_PATTERN.fullmatch(env_name) is None:
            continue
        secret = str(environment.get(env_name, "") or "")
        if secret:
            # 仅写入已加载的内存映射；原始 YAML 和编辑器快照仍保留空值/引用。
            channel["api_key"] = secret


def _inject_environment_channel(
    values: MutableMapping[str, Any], environment: Mapping[str, str]
) -> None:
    """在用户明确提供完整环境变量时补齐一个模型渠道。

    安全默认配置仍然不包含渠道和密钥。只有同时提供基础地址与模型名时才
    注入环境渠道；密钥可选，便于本地 Ollama/代理端点。这样无配置启动
    不会伪造供应商，但设置环境变量后无需把密钥写入 YAML。
    """

    llm_value = values.get("llm")
    if not isinstance(llm_value, MutableMapping):
        return
    channels = llm_value.get("channels")
    if channels not in (None, []):
        return
    # Claude Code/Anthropic 生态常用的环境变量保持显式映射；只有同时提供
    # 地址和模型时才注入，避免仅设置令牌就意外启用网络请求。
    explicit_meapet_base_url = _environment_value(environment, "MEAPET_API_BASE", "MEAPET_BASE_URL")
    openai_base_url = _environment_value(environment, "OPENAI_BASE_URL")
    anthropic_base_url = _environment_value(environment, "ANTHROPIC_BASE_URL")
    gemini_base_url = _environment_value(environment, "GEMINI_BASE_URL")
    base_url = _environment_value(
        environment,
        "MEAPET_API_BASE",
        "MEAPET_BASE_URL",
        "OPENAI_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "GEMINI_BASE_URL",
    )
    if not base_url:
        return
    channel_id = _environment_value(environment, "MEAPET_CHANNEL_ID") or "primary"
    configured_protocol = _environment_value(environment, "MEAPET_PROTOCOL")
    protocol = configured_protocol or (
        "anthropic_messages"
        if anthropic_base_url
        and not explicit_meapet_base_url
        and not openai_base_url
        and not gemini_base_url
        else "gemini_generate"
        if gemini_base_url and not explicit_meapet_base_url and not openai_base_url
        else "openai_chat"
    )
    protocol_family = protocol.strip().lower()
    if protocol_family in {"anthropic", "anthropic_messages", "claude"}:
        model_names = ("MEAPET_MODEL", "ANTHROPIC_MODEL")
        key_names = ("MEAPET_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")
    elif protocol_family in {"gemini", "gemini_generate", "google_gemini"}:
        model_names = ("MEAPET_MODEL", "GEMINI_MODEL")
        key_names = ("MEAPET_API_KEY", "GEMINI_API_KEY")
    else:
        model_names = ("MEAPET_MODEL", "OPENAI_MODEL")
        # 通用/ OpenAI 兼容入口优先接受 OpenAI 标准变量；保留 Anthropic
        # 别名作为最后回退，兼容同一代理同时暴露两套协议的 shell 环境。
        key_names = (
            "MEAPET_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
        )
    model = _environment_value(environment, *model_names)
    if not model:
        return
    api_key = _environment_value(
        environment,
        *key_names,
    )
    llm_value["channels"] = [
        {
            "id": channel_id,
            "protocol": protocol,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
            # 环境变量入口用于直接启动可对话的桌宠；同时声明工具能力，
            # 这样模型返回工具调用时不会被渠道能力诊断误判为纯文本渠道。
            "capabilities": ["streaming", "tools"],
            "enabled": True,
            "priority": 10,
        }
    ]
    routing = llm_value.get("routing")
    if not isinstance(routing, MutableMapping):
        routing = {}
        llm_value["routing"] = routing
    # 注入发生在原渠道列表为空时，原有 dialogue 路由不能指向可用渠道；
    # 统一改为注入渠道，避免 stale/primary 路由阻止环境配置生效。
    routing["dialogue"] = {
        "channel": channel_id,
        "required_capabilities": ["streaming"],
        "fallback_channels": [],
    }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(dict(existing), dict(value))
        else:
            result[key] = value
    return result


def _migrate_legacy_dialogue_prompt(values: Mapping[str, Any]) -> dict[str, Any]:
    """让旧 ``app.system_prompt`` 在默认 prompts 深合并前保持原语义。"""

    migrated = dict(values)
    raw_app = values.get("app")
    if not isinstance(raw_app, Mapping) or "system_prompt" not in raw_app:
        return migrated
    raw_prompts = raw_app.get("prompts")
    if isinstance(raw_prompts, Mapping) and "dialogue" in raw_prompts:
        return migrated
    app_values = dict(raw_app)
    prompt_values = dict(raw_prompts) if isinstance(raw_prompts, Mapping) else {}
    prompt_values["dialogue"] = raw_app.get("system_prompt")
    app_values["prompts"] = prompt_values
    migrated["app"] = app_values
    return migrated


def _validate_channels(values: Mapping[str, Any]) -> None:
    llm = values.get("llm", {})
    if not isinstance(llm, Mapping):
        raise ConfigurationError("llm must be a mapping")
    channels = llm.get("channels", [])
    if not isinstance(channels, list):
        raise ConfigurationError("llm.channels must be a list")

    channel_ids: set[str] = set()
    channel_models: dict[str, set[str]] = {}
    channel_protocols: dict[str, str] = {}
    channel_capabilities: dict[str, set[str]] = {}
    channel_ready: dict[str, bool] = {}
    for index, raw_channel in enumerate(channels):
        channel = _ensure_mapping(raw_channel, label=f"llm.channels[{index}]")
        channel_id = str(channel.get("id") or "").strip()
        # 渠道适配器字段保留旧配置兼容性；与 ChannelConfig 使用同一
        # 优先级：protocol > adapter > type。这样仅填写适配器的旧配置不会
        # 在加载阶段被错误地判定为缺少协议。
        protocol = str(
            channel.get("protocol") or channel.get("adapter") or channel.get("type") or ""
        ).strip()
        if not channel_id or not protocol:
            raise ConfigurationError(f"llm.channels[{index}] requires id and protocol")
        if protocol.lower() not in SUPPORTED_PROTOCOLS:
            raise ConfigurationError(f"llm.channels[{index}] has unsupported protocol: {protocol}")
        if channel_id in channel_ids:
            raise ConfigurationError(f"duplicate llm channel id: {channel_id}")

        models_value = channel.get("models", ())
        if isinstance(models_value, str):
            model_names = [models_value]
        elif models_value is None:
            model_names = []
        elif isinstance(models_value, Sequence) and not isinstance(
            models_value, (bytes, bytearray)
        ):
            model_names = list(models_value)
        else:
            raise ConfigurationError(f"llm.channels[{index}].models must be a list or string")
        model_names = [str(item or "").strip() for item in model_names]
        if any(not item for item in model_names):
            raise ConfigurationError(f"llm.channels[{index}].models contains an empty model")
        selected_model = str(channel.get("model", channel.get("default_model", "")) or "").strip()
        models = set(model_names)
        if selected_model:
            models.add(selected_model)
        channel_ids.add(channel_id)
        channel_models[channel_id] = models
        channel_protocols[channel_id] = protocol.lower()
        channel_enabled = parse_bool(
            channel.get("enabled", channel.get("enable", True)),
            field_name=f"llm.channels[{index}].enabled",
            default=True,
        )
        channel_ready[channel_id] = bool(
            channel_enabled
            and str(channel.get("base_url", channel.get("url", "")) or "").strip()
            and models
        )

        capabilities_value = channel.get("capabilities", channel.get("features", ()))
        if isinstance(capabilities_value, str):
            capability_names = [
                item.strip().lower() for item in capabilities_value.split(",") if item.strip()
            ]
        elif capabilities_value is None:
            capability_names = []
        elif isinstance(capabilities_value, Sequence) and not isinstance(
            capabilities_value, (bytes, bytearray)
        ):
            capability_names = [str(item or "").strip().lower() for item in capabilities_value]
        else:
            raise ConfigurationError(f"llm.channels[{index}].capabilities must be a list or string")
        if any(not item for item in capability_names):
            raise ConfigurationError(f"llm.channels[{index}].capabilities contains an empty value")
        channel_capabilities[channel_id] = set(capability_names) | {"streaming"}

    routing = llm.get("routing", {})
    if routing is None:
        return
    if not isinstance(routing, Mapping):
        raise ConfigurationError("llm.routing must be a mapping")

    for raw_task, raw_route in routing.items():
        task = str(raw_task or "").strip()
        if not task:
            raise ConfigurationError("llm.routing contains an empty task")
        if raw_route is None:
            continue
        if isinstance(raw_route, bool):
            # ``llm.routing.vision: false/true`` 是视觉摘要策略的布尔简写，
            # 不是一个渠道对象；ModelRouter 会将其转换为 enabled 策略。
            if task.casefold() == "vision":
                continue
            raise ConfigurationError(f"llm.routing.{task} must be a string or mapping")
        if (
            task.casefold() == "vision"
            and isinstance(raw_route, Mapping)
            and not {
                "channel",
                "channel_id",
                "model",
                "protocol",
                "required_capabilities",
                "capabilities",
                "fallback_channels",
                "fallback",
            }.intersection(raw_route)
        ):
            # 视觉摘要限制可以独立放在 ``llm.routing.vision``；没有任何
            # 路由字段时它不会选择渠道，也不应因缺少视觉渠道而被拒绝。
            continue
        if isinstance(raw_route, str):
            primary_id = raw_route.strip()
            fallback_value: object = ()
            route_model = ""
            route_protocol = ""
            required_value: object = ()
        elif isinstance(raw_route, Mapping):
            raw_channel_id = raw_route.get("channel_id", "")
            raw_channel = raw_route.get("channel", "")
            if (
                raw_channel_id
                and raw_channel
                and str(raw_channel_id).strip() != str(raw_channel).strip()
            ):
                raise ConfigurationError(f"llm.routing.{task} has conflicting channel identifiers")
            primary_id = str(raw_channel_id or raw_channel or "").strip()

            fallback_key_present = "fallback_channels" in raw_route
            fallback_value = raw_route.get("fallback_channels", raw_route.get("fallback", ()))
            if (
                fallback_key_present
                and "fallback" in raw_route
                and raw_route.get("fallback_channels") != raw_route.get("fallback")
            ):
                raise ConfigurationError(f"llm.routing.{task} has conflicting fallback fields")
            raw_model = raw_route.get("model", "")
            if raw_model is not None and not isinstance(raw_model, str):
                raise ConfigurationError(f"llm.routing.{task}.model must be a string")
            route_model = str(raw_model or "").strip()
            raw_protocol = raw_route.get("protocol", "")
            if raw_protocol is not None and not isinstance(raw_protocol, str):
                raise ConfigurationError(f"llm.routing.{task}.protocol must be a string")
            route_protocol = str(raw_protocol or "").strip().lower()
            required_value = raw_route.get(
                "required_capabilities",
                raw_route.get("capabilities", ()),
            )
            if (
                "required_capabilities" in raw_route
                and "capabilities" in raw_route
                and raw_route.get("required_capabilities") != raw_route.get("capabilities")
            ):
                raise ConfigurationError(f"llm.routing.{task} has conflicting capability fields")
        else:
            raise ConfigurationError(f"llm.routing.{task} must be a string or mapping")

        if primary_id and primary_id not in channel_ids:
            raise ConfigurationError(f"llm.routing.{task} references unknown channel: {primary_id}")
        if isinstance(fallback_value, str):
            fallback_ids = [item.strip() for item in fallback_value.split(",") if item.strip()]
        elif fallback_value is None:
            fallback_ids = []
        elif isinstance(fallback_value, Sequence) and not isinstance(
            fallback_value, (bytes, bytearray)
        ):
            fallback_ids = [str(item or "").strip() for item in fallback_value]
        else:
            raise ConfigurationError(
                f"llm.routing.{task}.fallback_channels must be a list or string"
            )
        if any(not item for item in fallback_ids):
            raise ConfigurationError(
                f"llm.routing.{task}.fallback_channels contains an empty channel"
            )
        if len(set(fallback_ids)) != len(fallback_ids):
            raise ConfigurationError(f"llm.routing.{task}.fallback_channels contains duplicates")
        if primary_id and primary_id in fallback_ids:
            raise ConfigurationError(
                f"llm.routing.{task}.fallback_channels cannot contain the primary channel"
            )
        unknown_fallbacks = [item for item in fallback_ids if item not in channel_ids]
        if unknown_fallbacks:
            raise ConfigurationError(
                f"llm.routing.{task} references unknown fallback channel: {unknown_fallbacks[0]}"
            )
        if fallback_ids and not primary_id:
            raise ConfigurationError(
                f"llm.routing.{task}.fallback_channels requires a primary channel"
            )

        if route_protocol and route_protocol not in SUPPORTED_PROTOCOLS:
            raise ConfigurationError(
                f"llm.routing.{task} has unsupported protocol: {route_protocol}"
            )
        if isinstance(required_value, str):
            required = {item.strip().lower() for item in required_value.split(",") if item.strip()}
        elif required_value is None:
            required = set()
        elif isinstance(required_value, Sequence) and not isinstance(
            required_value, (bytes, bytearray)
        ):
            required = {str(item or "").strip().lower() for item in required_value}
        else:
            raise ConfigurationError(
                f"llm.routing.{task}.required_capabilities must be a list or string"
            )
        if any(not item for item in required):
            raise ConfigurationError(
                f"llm.routing.{task}.required_capabilities contains an empty value"
            )
        if task.casefold() == "vision":
            required.add("vision")

        option_ids = [primary_id, *fallback_ids] if primary_id else list(channel_ids)
        eligible_ids = [
            channel_id
            for channel_id in option_ids
            if (not route_protocol or channel_protocols[channel_id] == route_protocol)
            and required.issubset(channel_capabilities[channel_id])
            and (not route_model or route_model in channel_models[channel_id])
            and channel_ready[channel_id]
        ]
        if route_model and not eligible_ids:
            raise ConfigurationError(
                f"llm.routing.{task}.model is not declared by an eligible channel"
            )
        if route_protocol and not eligible_ids:
            raise ConfigurationError(f"llm.routing.{task}.protocol has no eligible channel")
        if required and not eligible_ids:
            raise ConfigurationError(
                f"llm.routing.{task}.required_capabilities have no eligible channel"
            )
        if primary_id and not eligible_ids:
            raise ConfigurationError(f"llm.routing.{task} has no ready eligible channel")


def _validate_rendering_backend(values: Mapping[str, Any]) -> None:
    """校验渲染后端键；vllank 只作为明确不可用兼容值保留。"""

    rendering = values.get("rendering", {})
    if not isinstance(rendering, Mapping):
        raise ConfigurationError("rendering must be a mapping")
    try:
        from gui.renderers.protocol import normalize_renderer_backend

        normalize_renderer_backend(rendering.get("backend", "auto"))
    except (ImportError, ModuleNotFoundError) as exc:
        raise ConfigurationError("renderer backend contract is unavailable") from exc
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


def validate_local_web_configuration(values: Mapping[str, Any]) -> None:
    """校验可选本地 Web 控制面的安全边界。"""

    web_values = values.get("web", {})
    if not isinstance(web_values, Mapping):
        raise ConfigurationError("web must be a mapping")
    local_api_values = web_values.get("local_api", {})
    if not isinstance(local_api_values, Mapping):
        raise ConfigurationError("web.local_api must be a mapping")
    parse_bool(
        local_api_values.get("enabled"),
        field_name="web.local_api.enabled",
        default=False,
    )
    if local_api_values.get("host", "127.0.0.1") != "127.0.0.1":
        raise ConfigurationError("web.local_api.host must be 127.0.0.1")
    port = local_api_values.get("port", 0)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ConfigurationError("web.local_api.port must be an integer from 0 to 65535")
    parse_bool(
        local_api_values.get("events"),
        field_name="web.local_api.events",
        default=True,
    )
    event_limit = local_api_values.get("event_limit", 64)
    if (
        isinstance(event_limit, bool)
        or not isinstance(event_limit, int)
        or not 1 <= event_limit <= 64
    ):
        raise ConfigurationError("web.local_api.event_limit must be an integer from 1 to 64")


def _validate_plugin_configuration(values: Mapping[str, Any]) -> None:
    """使用插件设置实例的真实默认值校验可选配置区。"""

    from services.plugins import PluginSettings

    raw = values.get("plugins")
    if raw is None:
        return
    if not isinstance(raw, Mapping):
        raise ConfigurationError("plugin configuration is invalid: plugins must be a mapping")
    defaults = PluginSettings()
    normalized = {
        "enabled": defaults.enabled,
        "directories": defaults.directories,
        "entry_points_enabled": defaults.entry_points_enabled,
        "entry_point_group": defaults.entry_point_group,
        "reload_enabled": defaults.reload_enabled,
        "reload_interval_seconds": defaults.reload_interval_seconds,
    }
    normalized.update(raw)
    try:
        PluginSettings.from_mapping(normalized)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"plugin configuration is invalid: {exc}") from exc


def load_configuration(
    path: str | Path,
    *,
    defaults: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> LoadedConfiguration:
    """读取单份 YAML，合并安全默认值并验证已定义的渠道契约。"""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {resolved_path}")
    try:
        payload = _read_configuration_payload(resolved_path)
        raw = yaml.safe_load(payload.decode("utf-8"))
    except ConfigurationError:
        raise
    except UnicodeError as exc:
        raise ConfigurationError("configuration file cannot be read") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError("configuration is not valid YAML") from exc
    values = _migrate_legacy_dialogue_prompt(_ensure_mapping(raw, label="configuration"))
    _reject_forbidden_keys(values)
    merged = _deep_merge(_ensure_mapping(defaults, label="defaults"), values)
    environment_values = dict(os.environ if environment is None else environment)
    # Windows 原生环境通常只有 ``USERPROFILE``；保留 ``HOME`` 兼容别名，
    # 让现有示例配置在不改写用户环境的情况下继续展开。
    if (os.name == "nt" or sys.platform.startswith("win")) and not str(
        environment_values.get("HOME", "") or ""
    ).strip():
        user_profile = str(environment_values.get("USERPROFILE", "") or "").strip()
        if user_profile:
            environment_values["HOME"] = user_profile
    expanded = _expand_environment(merged, environment_values)
    if not isinstance(expanded, Mapping):
        raise ConfigurationError("configuration root must be a mapping")
    normalized = dict(expanded)
    _hydrate_explicit_channel_keys(normalized, environment_values)
    _inject_environment_channel(normalized, environment_values)
    _validate_channels(normalized)
    _validate_rendering_backend(normalized)
    # 插件配置与其它启动边界在同一加载阶段校验，文件观察器读取到无效
    # 插件设置时不会先启动旧配置之外的任何插件实例。
    _validate_plugin_configuration(normalized)
    validate_local_web_configuration(normalized)
    return LoadedConfiguration(
        path=resolved_path,
        values=normalized,
        source_digest=hashlib.sha256(payload).hexdigest(),
    )


def expand_environment_values(
    value: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """在写入配置前展开环境变量引用，缺失变量时 fail-closed。

    配置编辑器保存的是 `${ENV_NAME}` 占位符；运行时热加载不能把占位符
    当作真实密钥发送，也不能先写入一份无法启动的配置。该函数只返回内存
    映射，调用方不得把展开后的结果写回 YAML。
    """

    if not isinstance(value, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    source = {str(key): item for key, item in value.items()}
    _reject_forbidden_keys(source)
    environment_values = dict(os.environ if environment is None else environment)
    if (os.name == "nt" or sys.platform.startswith("win")) and not str(
        environment_values.get("HOME", "") or ""
    ).strip():
        user_profile = str(environment_values.get("USERPROFILE", "") or "").strip()
        if user_profile:
            environment_values["HOME"] = user_profile
    expanded = _expand_environment(source, environment_values)
    if not isinstance(expanded, Mapping):
        raise ConfigurationError("configuration root must be a mapping")
    return {str(key): item for key, item in expanded.items()}


def _unique_paths(paths: list[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return tuple(result)


def _cwd_repository_ignored(environment: Mapping[str, str]) -> bool:
    """解析 ``MEAPET_IGNORE_CWD_REPO`` 开关；缺失或空串保持默认关闭。

    环境键缺失时保持默认关闭；显式出现时才走严格布尔解析，避免空串
    触发 parse_bool 的类型拒绝。
    """

    raw_flag = str(environment.get("MEAPET_IGNORE_CWD_REPO", "") or "").strip()
    return bool(raw_flag) and parse_bool(
        raw_flag,
        field_name="MEAPET_IGNORE_CWD_REPO",
        default=False,
    )


def _find_project_root(start: Path, *, environment: Mapping[str, str] | None = None) -> Path:
    """从启动目录向上寻找当前源码项目根；找不到时保留启动目录。

    托盘或快捷方式常以无关目录为 cwd 启动；当 ``MEAPET_IGNORE_CWD_REPO=1``
    时不再从 cwd 向上探测仓库根，避免把错误仓库当作资源与配置基准。显式
    ``path``、``MEAPET_CONFIG`` 与用户目录的发现不受影响。默认关闭保持
    既有语义不变。
    """

    env = os.environ if environment is None else environment
    if _cwd_repository_ignored(env):
        return start
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file() or (
            directory / "vivid-gliding-boot.md"
        ).is_file():
            return directory
    return start


def _module_project_root() -> Path | None:
    """返回源码安装中可识别的项目根，已安装 wheel 时返回 ``None``。"""

    module_root = Path(__file__).resolve().parents[2]
    if (module_root / "pyproject.toml").is_file() or (
        module_root / "vivid-gliding-boot.md"
    ).is_file():
        return module_root
    return None


def _user_configuration_paths(
    *,
    environment: Mapping[str, str],
    home: Path,
) -> tuple[Path, ...]:
    """按平台惯例返回用户配置候选，不读取任何配置内容。"""

    roots: list[Path] = []
    xdg_root = str(environment.get("XDG_CONFIG_HOME", "")).strip()
    appdata_root = str(environment.get("APPDATA", "")).strip()
    if xdg_root:
        roots.append(Path(xdg_root) / "meapet")
    if appdata_root:
        roots.append(Path(appdata_root) / "MeaPet")
    roots.extend(
        (
            home / ".config" / "meapet",
            home / "Library" / "Application Support" / "MeaPet",
            home / ".meapet",
        )
    )
    names = ("config.yaml", "config.yml", "config.json")
    return _unique_paths([root / name for root in roots for name in names])


def _directory_usable(path: Path) -> bool:
    """候选数据目录必须能够实际创建；探测失败按不可用处理。

    首次启动本来就需要创建用户数据目录；把 mkdir 纳入选择探测能让
    ``Database`` 随后的 exist_ok 创建保持幂等，并让权限失败在选择阶段
    即被感知，从而回落到备用目录而不是在数据库初始化时才抛错。
    """

    try:
        path.mkdir(parents=True, exist_ok=True)
        status = path.stat()
        return stat.S_ISDIR(status.st_mode)
    except (NotADirectoryError, OSError):
        return False


def _user_data_directory(
    *,
    environment: Mapping[str, str],
    home: Path,
    fallback: Path | None = None,
) -> Path:
    """返回首次启动数据库的用户可写目录。

    与 ``_user_configuration_paths`` 同层级保持一致：XDG 小写 ``meapet``，
    APPDATA/macOS 沿用 ``MeaPet``（既有 Windows 平台测试锁定的层级）。
    候选目录不可用（如同名文件占位或创建失败）时依次回落到 ``fallback``
    与 ``home/.meapet-data``；仅探测和创建目录本身，不写入配置。
    """

    xdg_root = str(environment.get("XDG_DATA_HOME", "")).strip()
    appdata_root = str(environment.get("APPDATA", "")).strip()
    if xdg_root:
        primary = (Path(xdg_root).expanduser() / "meapet").resolve()
    elif appdata_root:
        primary = (Path(appdata_root).expanduser() / "MeaPet" / "data").resolve()
    elif sys.platform == "darwin":
        primary = (home / "Library" / "Application Support" / "MeaPet" / "data").resolve()
    else:
        primary = (home / ".local" / "share" / "meapet").resolve()
    if _directory_usable(primary):
        return primary
    if fallback is not None:
        fallback = Path(fallback).expanduser().resolve()
        if _directory_usable(fallback):
            warnings.warn(
                f"user data directory '{primary}' is unavailable; falling back to '{fallback}'",
                RuntimeWarning,
                stacklevel=2,
            )
            return fallback
    directory = (home / ".meapet-data").resolve()
    if _directory_usable(directory) and directory != primary:
        warnings.warn(
            f"user data directory '{primary}' is unavailable; falling back to '{directory}'",
            RuntimeWarning,
            stacklevel=2,
        )
        return directory
    return primary


def _resource_root_for_project(project_root: Path) -> Path:
    for relative in ("resources", "temp/resources"):
        path = project_root / relative
        if path.is_dir():
            return path.resolve()
    return (project_root / "resources").resolve()


def _write_default_configuration(path: Path, values: Mapping[str, Any]) -> bool:
    """尽力写入安全默认配置；目录创建被拒绝时回落到 ``home/.meapet-data``。

    首次启动的配置与数据库目录不可写不能阻断启动：尝试主目录，失败后
    用与 ``_user_data_directory`` 同风格的昵称目录兜底；两者都失败时
    返回 False，由调用方使用等价内存配置。
    """

    targets = [path]
    fallback_database = Path(str(values.get("storage", {}).get("database", ""))).expanduser()
    try:
        home_target = _user_data_directory(
            environment=dict(os.environ),
            home=Path.home().expanduser().resolve(),
            fallback=fallback_database.parent if str(fallback_database) else None,
        )
    except (OSError, RuntimeError, ValueError):
        home_target = fallback_database.parent if str(fallback_database) else None
    if home_target is not None:
        try:
            targets.append(home_target.resolve() / "config.yaml")
        except OSError:
            pass
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                yaml.safe_dump(dict(values), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
        except OSError:
            continue
        return True
    return False


def discover_configuration(
    path: str | Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    project_root: str | Path | None = None,
    home: str | Path | None = None,
    create_default: bool = True,
) -> LoadedConfiguration:
    """按固定优先级发现并加载启动配置。

    优先级严格为：显式 ``path``、``MEAPET_CONFIG``、当前工作目录和项目根的
    ``config.yaml/config.yml/config.json``、用户配置目录、项目示例模板。完全
    缺失时在用户配置目录写入无密钥安全默认 YAML（可通过 ``create_default``
    关闭写入，供只读测试或打包环境使用），失败则继续使用等价的内存配置。
    """

    env = dict(os.environ if environment is None else environment)
    start = Path.cwd() if cwd is None else Path(cwd)
    start = start.expanduser().resolve()
    detected_root = _find_project_root(start, environment=env)
    # 开关打开时不把启动目录或仓库探测结果作为仓库根候选，避免托盘/
    # 快捷方式 cwd 恰好是别的仓库时采纳错误 config；显式 project_root
    # 与模块根仍然保留，用户目录发现不受影响。默认关闭保持既有语义。
    ignore_cwd_repository = _cwd_repository_ignored(env)
    roots: list[Path] = [] if ignore_cwd_repository else [start, detected_root]
    provided_root: Path | None = None
    if project_root is not None:
        provided_root = Path(project_root).expanduser()
        if not provided_root.is_absolute():
            provided_root = start / provided_root
        roots.append(provided_root)
    module_root = _module_project_root()
    if module_root is not None:
        roots.append(module_root)
    project_roots = _unique_paths(roots)
    if home is not None:
        home_path = Path(home).expanduser().resolve()
    else:
        configured_home = str(env.get("HOME", env.get("USERPROFILE", ""))).strip()
        home_path = (
            Path(configured_home).expanduser().resolve()
            if configured_home
            else Path.home().resolve()
        )

    if provided_root is not None:
        resource_base = provided_root.resolve()
    elif detected_root != start:
        resource_base = detected_root
    elif module_root is not None:
        resource_base = module_root
    else:
        resource_base = start
    defaults_resource = _resource_root_for_project(resource_base)
    defaults_database = _user_data_directory(environment=env, home=home_path) / "meapet.sqlite3"
    defaults = default_configuration_values(
        resource_root=defaults_resource,
        database_path=defaults_database,
    )

    explicit = str(path).strip() if path is not None else ""
    if explicit:
        selected = Path(explicit).expanduser()
        if not selected.is_absolute():
            selected = start / selected
        if not selected.is_file():
            raise ConfigurationError(f"configuration file does not exist: {selected.resolve()}")
        return load_configuration(selected, defaults=defaults, environment=env)

    environment_path = str(env.get("MEAPET_CONFIG", "")).strip()
    if environment_path:
        selected = Path(environment_path).expanduser()
        if not selected.is_absolute():
            selected = start / selected
        if not selected.is_file():
            raise ConfigurationError(
                f"configuration file from MEAPET_CONFIG does not exist: {selected.resolve()}"
            )
        return load_configuration(selected, defaults=defaults, environment=env)

    names = ("config.yaml", "config.yml", "config.json")
    repository_paths = _unique_paths([root / name for root in project_roots for name in names])
    for path_option in repository_paths:
        if path_option.is_file():
            return load_configuration(path_option, defaults=defaults, environment=env)

    user_paths = _user_configuration_paths(environment=env, home=home_path)
    for path_option in user_paths:
        if path_option.is_file():
            return load_configuration(path_option, defaults=defaults, environment=env)

    example_names = (
        "config/app.example.yaml",
        "config/app.example.yml",
        "config.example.yaml",
        "config.example.yml",
        "config.example.json",
    )
    for path_option in _unique_paths(
        [root / name for root in project_roots for name in example_names]
    ):
        if path_option.is_file():
            return load_configuration(path_option, defaults=defaults, environment=env)

    target = user_paths[0]
    if create_default:
        _write_default_configuration(target, defaults)
    # 主路径不可写时退回与数据目录同风格的 home/.meapet-data 兜底；
    # 仍失败则返回同一份安全值，路径保持目标位置，后续数据库和相对资源
    # 解析不会依赖当前进程目录。
    if target.is_file():
        return load_configuration(target, defaults=defaults, environment=env)
    try:
        fallback_home = _user_data_directory(environment=env, home=home_path)
    except (OSError, RuntimeError, ValueError):
        fallback_home = None
    if fallback_home is not None:
        fallback_path = fallback_home / "config.yaml"
        if fallback_path.is_file():
            return load_configuration(fallback_path, defaults=defaults, environment=env)
    return LoadedConfiguration(path=target, values=defaults)


def update_path(values: MutableMapping[str, Any], dotted_path: str, value: object) -> None:
    """只允许安全的点路径更新，用于向导字段补丁。"""
    parts = [part.strip() for part in str(dotted_path or "").split(".") if part.strip()]
    if not parts:
        raise ConfigurationError("configuration path is required")
    if any(part.lower() in _FORBIDDEN_KEYS for part in parts):
        raise ConfigurationError("configuration path contains a forbidden key")
    current: MutableMapping[str, Any] = values
    for part in parts[:-1]:
        nested = current.get(part)
        if nested is None:
            nested = {}
            current[part] = nested
        if not isinstance(nested, MutableMapping):
            raise ConfigurationError(f"configuration path is not a mapping: {part}")
        current = nested
    current[parts[-1]] = value


def resolve_resource_path(value: object, *, configuration_directory: Path) -> Path:
    """将相对资源路径锚定在配置文件目录，不隐式猜测其他根目录。"""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("resource path must be a non-empty string")
    resolved_path = Path(value).expanduser()
    return (
        resolved_path.resolve()
        if resolved_path.is_absolute()
        else (configuration_directory / resolved_path).resolve()
    )


def redact_secrets(value: object, _active: set[int] | None = None) -> object:
    """递归脱敏诊断、导出和 UI 中不应显示的字段。"""
    active = _active if _active is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ConfigurationError("configuration contains recursive aliases")
        active.add(identity)
        try:
            return {
                str(key): (
                    "***"
                    if _SECRET_PATTERN.search(str(key))
                    and str(key).strip().lower().replace("-", "_")
                    not in {"api_key_env", "api_key_environment"}
                    else redact_secrets(item, active)
                )
                for key, item in value.items()
            }
        finally:
            active.remove(identity)
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise ConfigurationError("configuration contains recursive aliases")
        active.add(identity)
        try:
            return [redact_secrets(item, active) for item in value]
        finally:
            active.remove(identity)
    return value
