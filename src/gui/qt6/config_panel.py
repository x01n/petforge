from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.editor import editor_snapshot, parse_editor_yaml, validate_editor_values
from core.adapters.direct.errors import AdapterConfigurationError
from core.conversation.persona import PersonaProfile, PromptProfile
from gui.qt6.fancyui import (
    FancyCard,
    FancyCommandBar,
    FancyInfoBar,
    FancyNavigationView,
    FancyStyleController,
    fancy_icon,
)
from gui.qt6.md3 import (
    DARK_MD3_THEME,
    MD3Theme,
    build_md3_stylesheet,
    build_qt_palette,
    remap_legacy_stylesheet,
    theme_from_configuration,
)
from gui.qt6.microphone import AudioInputDeviceChoice
from gui.qt6.scheduler_panel import SchedulerPanel
from gui.renderers.assets import default_renderer_registry
from gui.renderers.protocol import CANONICAL_RENDERER_BACKEND_CHOICES, DEFAULT_MOTION_NAMES
from gui.renderers.threed import probe_qt_vulkan_runtime
from services.model_routing.channels import channel_from_mapping

try:  # Qt 依赖保持可选，核心服务不需要 GUI。
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    pyside6_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_available = False


_SECTIONS: tuple[tuple[str, str, tuple[str, ...], str], ...] = (
    ("overview", "总览", (), "查看当前状态，并点选要调整的功能。"),
    ("app", "基本设置", ("app",), "设置桌宠名称和启动体验。"),
    ("llm", "模型服务", ("llm",), "选择模型服务和默认模型。"),
    ("tts", "语音输出", ("tts",), "开启语音、选择语言和声音服务。"),
    ("asr", "语音识别", ("asr",), "管理内置 SenseVoice 工作进程和识别语言。"),
    ("logging", "日志", ("logging",), "选择日志详细程度和文件保留方式。"),
    ("memory", "记忆策略", ("memory",), "调整记忆召回、上下文和清理策略。"),
    ("tools", "工具与权限", ("tools",), "选择桌宠可以使用的能力和确认方式。"),
    ("mcp", "外部服务", ("mcp",), "连接可选的外部工具服务。"),
    ("watcher", "窗口观察", ("watcher",), "控制桌宠是否感知当前窗口变化。"),
    ("scheduler", "定时与触发", ("scheduler",), "点击添加定时动作和事件反应。"),
    ("behavior", "自主行为", ("behavior",), "控制桌宠是否主动走动和随机互动。"),
    ("proactive", "模型主动行为", ("proactive",), "配置事件规则、模型预算和冷却。"),
    ("rendering", "显示效果", ("rendering",), "调整桌宠大小和渲染资源。"),
    ("storage", "数据保存", ("storage",), "查看记忆和运行数据的保存位置。"),
    ("ui", "界面与快捷键", ("ui",), "调整输出、置顶和快捷键行为。"),
    ("advanced", "开发者工具", (), "仅用于兼容性排查和完整配置维护。"),
)

_SECTION_ICONS: dict[str, str] = {
    "overview": "home",
    "app": "info",
    "llm": "forward",
    "tts": "volume",
    "asr": "volume",
    "logging": "file",
    "memory": "save",
    "tools": "play",
    "mcp": "forward",
    "watcher": "info",
    "scheduler": "play",
    "behavior": "forward",
    "proactive": "forward",
    "rendering": "file",
    "storage": "folder",
    "ui": "info",
    "advanced": "warning",
}


@dataclass(frozen=True)
class _FieldSpec:
    """可视化表单中的一个标量字段。"""

    path: tuple[str, ...]
    label: str
    kind: str = "text"
    default: object = ""
    minimum: float = -100000.0
    maximum: float = 100000.0
    step: float = 1.0
    choices: tuple[str, ...] = ()
    secret: bool = False
    tooltip: str = ""


_DEFAULT_PERSONA = PersonaProfile()
_DEFAULT_PROMPTS = PromptProfile()


_FIELD_SPECS: dict[str, tuple[_FieldSpec, ...]] = {
    "app": (
        _FieldSpec(("name",), "应用名称", default="MeaPet", tooltip="桌宠应用显示名称。"),
        _FieldSpec(("persona", "name"), "桌宠名字", default=_DEFAULT_PERSONA.name),
        _FieldSpec(("persona", "role"), "桌宠身份", default=_DEFAULT_PERSONA.role),
        _FieldSpec(
            ("persona", "user_address"),
            "对你的称呼",
            default=_DEFAULT_PERSONA.user_address,
        ),
        _FieldSpec(
            ("persona", "relationship"),
            "与你的关系",
            default=_DEFAULT_PERSONA.relationship,
        ),
        _FieldSpec(
            ("persona", "language"),
            "主要语言",
            default=_DEFAULT_PERSONA.language,
        ),
        _FieldSpec(
            ("persona", "speaking_style"),
            "说话风格",
            default=_DEFAULT_PERSONA.speaking_style,
        ),
        _FieldSpec(
            ("persona", "background"),
            "背景设定",
            "multiline",
            _DEFAULT_PERSONA.background,
            tooltip="描述桌宠的来历、世界观或长期背景，不要填写密钥。",
        ),
        _FieldSpec(
            ("persona", "traits"),
            "性格特点",
            "lines",
            _DEFAULT_PERSONA.traits,
            tooltip="每行一项，例如“好奇”“可靠”。",
        ),
        _FieldSpec(
            ("persona", "goals"),
            "长期目标",
            "lines",
            _DEFAULT_PERSONA.goals,
            tooltip="每行一项目标。",
        ),
        _FieldSpec(
            ("persona", "boundaries"),
            "行为边界",
            "lines",
            _DEFAULT_PERSONA.boundaries,
            tooltip="每行一项不可越过的边界。",
        ),
        _FieldSpec(
            ("persona", "likes"),
            "喜欢的事物",
            "lines",
            _DEFAULT_PERSONA.likes,
            tooltip="每行一项。",
        ),
        _FieldSpec(
            ("persona", "dislikes"),
            "不喜欢的事物",
            "lines",
            _DEFAULT_PERSONA.dislikes,
            tooltip="每行一项。",
        ),
        _FieldSpec(
            ("persona", "custom_instructions"),
            "补充人设",
            "multiline",
            _DEFAULT_PERSONA.custom_instructions,
            tooltip="补充语气、互动习惯或角色细节。",
        ),
        _FieldSpec(
            ("persona", "proactive_enabled"),
            "允许主动行为",
            "bool",
            _DEFAULT_PERSONA.proactive_enabled,
        ),
        _FieldSpec(
            ("prompts", "dialogue"),
            "对话提示词",
            "multiline",
            _DEFAULT_PROMPTS.dialogue,
            tooltip="对话模型的基础提示词；人设信息会在运行时自动追加。",
        ),
        _FieldSpec(
            ("prompts", "tool_guidance"),
            "工具使用提示词",
            "multiline",
            _DEFAULT_PROMPTS.tool_guidance,
            tooltip="约束模型何时以及如何使用桌面、MCP 和系统工具。",
        ),
        _FieldSpec(
            ("prompts", "memory_summary"),
            "记忆总结提示词",
            "multiline",
            _DEFAULT_PROMPTS.memory_summary,
            tooltip="用于每日或定时对话总结。",
        ),
        _FieldSpec(
            ("prompts", "memory_extract"),
            "记忆提取提示词",
            "multiline",
            _DEFAULT_PROMPTS.memory_extract,
            tooltip="用于从对话中提取结构化长期记忆。",
        ),
    ),
    "logging": (
        _FieldSpec(
            ("level",),
            "默认等级",
            "enum",
            "INFO",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        ),
        _FieldSpec(("console", "enabled"), "控制台输出", "bool", True),
        _FieldSpec(
            ("console", "level"),
            "控制台等级",
            "enum",
            "INFO",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        ),
        _FieldSpec(
            ("console", "color"), "颜色输出", "enum", "auto", choices=("auto", "always", "never")
        ),
        _FieldSpec(("file", "enabled"), "文件日志", "bool", False),
        _FieldSpec(("file", "path"), "日志路径", default="./data/meapet.log"),
        _FieldSpec(
            ("file", "level"),
            "文件等级",
            "enum",
            "INFO",
            choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        ),
        _FieldSpec(
            ("file", "rotation"), "轮换方式", "enum", "time", choices=("none", "time", "size")
        ),
        _FieldSpec(("file", "when"), "轮换时刻", default="midnight"),
        _FieldSpec(("file", "interval"), "轮换间隔", "int", 1, 1, 3650),
        _FieldSpec(("file", "backup_count"), "保留份数", "int", 7, 0, 1000),
        _FieldSpec(("file", "max_bytes"), "单文件上限", "int", 10485760, 1, 2147483647),
    ),
    "tts": (
        _FieldSpec(("enabled",), "启用语音", "bool", True),
        _FieldSpec(
            ("backend",),
            "后端",
            "enum",
            "gpt_sovits_stdio",
            choices=("gpt_sovits_stdio", "gpt_sovits_http", "subprocess", "text_only"),
        ),
        _FieldSpec(("language",), "输出语言", "enum", "zh", choices=("zh", "ja", "en")),
        _FieldSpec(
            ("role",),
            "默认角色",
            "text",
            "",
            tooltip="可选角色标识；用于将普通对话路由到对应 TTS profile。",
        ),
        _FieldSpec(
            ("output_device_id",),
            "输出设备 ID",
            "text",
            "",
            tooltip="留空跟随系统默认扬声器；可从桌面音频设备列表复制精确 ID。",
        ),
        # GPT-SoVITS 参考音频和协议当前只提供中文、日文、英文；其它语种
        # 需使用声明了对应语言的自定义 TTS profile，而不是静默回退。
        _FieldSpec(("endpoint",), "服务地址", default="http://127.0.0.1:9880/tts"),
        _FieldSpec(
            ("engine_root",),
            "本地引擎目录",
            default="",
            tooltip="填写后由标准输入 worker 预载并持有 GPT-SoVITS 模型；留空使用外部服务。",
        ),
        _FieldSpec(
            ("engine_config",),
            "引擎配置",
            default="GPT_SoVITS/configs/tts_infer.yaml",
        ),
        _FieldSpec(
            ("python_executable",),
            "引擎 Python",
            default="",
            tooltip="本地引擎模式使用的完整 Python 可执行文件路径。",
        ),
        _FieldSpec(("ref_dir",), "参考音频目录", default="../resources/GPT-Sovits"),
        _FieldSpec(("ref_audio_path",), "参考音频", default=""),
        _FieldSpec(("prompt_text",), "参考文本", default=""),
        _FieldSpec(("prompt_lang",), "参考语言", default=""),
        _FieldSpec(("media_type",), "媒体格式", "enum", "wav", choices=("wav", "raw", "pcm")),
        _FieldSpec(("expected_sample_rate",), "期望采样率", "int", 32000, 1, 384000),
        _FieldSpec(("streaming_mode",), "流式模式", "int", 3, 0, 3),
        _FieldSpec(("timeout_seconds",), "请求超时（秒）", "float", 60.0, 1.0, 3600.0, 1.0),
        _FieldSpec(
            ("startup_timeout_seconds",),
            "启动等待（秒）",
            "float",
            300.0,
            1.0,
            3600.0,
            1.0,
            tooltip="应用启动时等待标准输入语音 worker 完成本地模型预载的最长时间。",
        ),
        _FieldSpec(
            ("shutdown_timeout_seconds",),
            "卸载等待（秒）",
            "float",
            2.0,
            0.05,
            30.0,
            0.05,
            tooltip="退出或热重载时等待语音 worker 卸载模型的最长时间。",
        ),
        _FieldSpec(("health_probe",), "启动探测", "bool", True),
        _FieldSpec(("require_final",), "要求结束帧", "bool", False),
        _FieldSpec(("require_audio",), "要求音频", "bool", True),
        _FieldSpec(("validate_pcm",), "校验 PCM", "bool", False),
        _FieldSpec(("queue_size",), "队列长度", "int", 16, 1, 128),
    ),
    "asr": (
        _FieldSpec(("enabled",), "启用语音识别", "bool", False),
        _FieldSpec(
            ("backend",),
            "识别后端",
            "enum",
            "sensevoice",
            choices=("sensevoice",),
        ),
        _FieldSpec(
            ("python_executable",),
            "工作进程 Python",
            default="",
            tooltip="内置 SenseVoice IPC worker 使用的 Python 可执行文件。",
        ),
        _FieldSpec(
            ("model_path",),
            "SenseVoice 模型",
            default="",
            tooltip=(
                "完整模型 snapshot 目录，需包含 model.pt、config.yaml、"
                "tokens.json 和 am.mvn；该值不会显示在模块状态页。"
            ),
        ),
        _FieldSpec(
            ("device",),
            "推理设备",
            default="cpu",
            tooltip="传递给 FunASR 的设备标识，例如 cpu；保留自定义设备值。",
        ),
        _FieldSpec(
            ("language",),
            "识别语言",
            "enum",
            "zh",
            choices=("auto", "zh", "en", "yue", "ja", "ko", "nospeech"),
        ),
        _FieldSpec(
            ("timeout_seconds",),
            "单次识别超时（秒）",
            "float",
            60.0,
            0.1,
            600.0,
            0.1,
        ),
        _FieldSpec(
            ("startup_timeout_seconds",),
            "模型启动等待（秒）",
            "float",
            180.0,
            0.1,
            600.0,
            0.1,
        ),
        _FieldSpec(
            ("max_audio_bytes",),
            "单次音频上限（字节）",
            "int",
            16777216,
            1024,
            67108864,
        ),
        _FieldSpec(("capture", "enabled"), "启用按键录音", "bool", True),
        _FieldSpec(
            ("capture", "max_duration_seconds"),
            "最长录音时间（秒）",
            "float",
            15.0,
            0.1,
            300.0,
            0.5,
        ),
        _FieldSpec(
            ("capture", "auto_submit"),
            "识别后自动发送（固定关闭）",
            "bool",
            False,
            tooltip="识别结果只填入输入框；安全边界要求该选项保持关闭。",
        ),
        _FieldSpec(
            ("capture", "device_id"),
            "输入设备",
            "audio_input",
            "",
            tooltip="选择“跟随系统默认”时保持空值；设备标识只写入本地配置。",
        ),
    ),
    "memory": (
        _FieldSpec(("enabled",), "启用长期记忆", "bool", True),
        _FieldSpec(("recall_limit",), "每次召回数量", "int", 7, 1, 100),
        _FieldSpec(("context_max_chars",), "记忆上下文上限", "int", 6000, 512, 50000),
        _FieldSpec(
            ("context_memory_item_chars",),
            "单条记忆字数上限",
            "int",
            500,
            64,
            5000,
        ),
        _FieldSpec(("recent_exchange_limit",), "近期对话条数", "int", 10, 0, 50),
        _FieldSpec(("max_memories",), "最多保存记忆", "int", 2000, 100, 10000),
        _FieldSpec(("decay_days",), "衰减周期（天）", "float", 7.0, 0.1, 3650.0, 0.1),
        _FieldSpec(("prune_days",), "清理周期（天）", "float", 30.0, 0.0, 3650.0, 0.1),
        _FieldSpec(
            ("prune_importance_floor",),
            "清理优先级下限",
            "int",
            1,
            0,
            10,
        ),
        _FieldSpec(("consolidation_enabled",), "启用相似记忆合并", "bool", True),
        _FieldSpec(
            ("consolidation_similarity",),
            "合并相似度阈值",
            "float",
            0.85,
            0.0,
            1.0,
            0.01,
        ),
        _FieldSpec(
            ("max_consolidation_memories",),
            "单次合并上限",
            "int",
            200,
            1,
            1000,
        ),
        _FieldSpec(("summarize_every_n",), "摘要间隔消息数", "int", 20, 1, 1000),
        _FieldSpec(("summary_chat_limit",), "摘要最多对话数", "int", 20, 4, 500),
        _FieldSpec(("summarization_enabled",), "启用自动总结", "bool", True),
        _FieldSpec(("summarize_daily",), "跨日自动总结", "bool", True),
        _FieldSpec(
            ("summary_interval_minutes",),
            "定时总结间隔（分钟）",
            "int",
            360,
            0,
            43200,
            tooltip="设为 0 时只使用消息数和跨日触发。",
        ),
        _FieldSpec(("summary_min_messages",), "总结最少消息数", "int", 4, 2, 500),
        _FieldSpec(("exchange_min_chars",), "保存对话最小字数", "int", 20, 0, 20000),
        _FieldSpec(("exchange_importance",), "对话记忆优先级", "int", 2, 0, 10),
        _FieldSpec(
            ("exchange_decay_factor",),
            "对话衰减系数",
            "float",
            0.8,
            0.0,
            10.0,
            0.01,
        ),
        _FieldSpec(
            ("auto_extract_enabled",),
            "自动提取记忆",
            "bool",
            True,
            tooltip="只从用户明确表达的稳定事实中提取，不保存凭据或联系方式。",
        ),
        _FieldSpec(("extract_max_items",), "单回合提取上限", "int", 3, 0, 20),
        _FieldSpec(("extract_max_chars",), "单条提取字数上限", "int", 240, 32, 2000),
        _FieldSpec(
            ("extract_min_confidence",),
            "提取最低置信度",
            "float",
            0.8,
            0.0,
            1.0,
            0.01,
        ),
        _FieldSpec(("extract_default_priority",), "提取默认优先级", "int", 5, 0, 10),
        _FieldSpec(
            ("recall_min_similarity",),
            "召回最低相关度",
            "float",
            0.04,
            0.0,
            1.0,
            0.01,
        ),
        _FieldSpec(("always_recall_priority",), "强制召回优先级", "int", 9, 0, 10),
        _FieldSpec(("semantic", "enabled"), "启用语义索引", "bool", False),
        _FieldSpec(
            ("semantic", "provider"),
            "语义模型提供方",
            "enum",
            "sentence_transformers",
            choices=("sentence_transformers",),
        ),
        _FieldSpec(
            ("semantic", "model_path"),
            "本地语义模型目录",
            default="",
            tooltip="只读取本地模型目录，不会联网下载模型。",
        ),
        _FieldSpec(("semantic", "model_id"), "语义模型 ID", default=""),
        _FieldSpec(("semantic", "model_revision"), "语义模型 revision", default=""),
        _FieldSpec(("semantic", "dimension"), "向量维度", "int", 384, 8, 4096),
        _FieldSpec(("semantic", "batch_size"), "嵌入批量大小", "int", 16, 1, 256),
        _FieldSpec(
            ("semantic", "timeout_seconds"),
            "语义查询超时（秒）",
            "float",
            5.0,
            0.05,
            300.0,
            0.05,
        ),
        _FieldSpec(
            ("semantic", "startup_timeout_seconds"),
            "语义模型启动超时（秒）",
            "float",
            60.0,
            1.0,
            600.0,
            0.5,
        ),
        _FieldSpec(
            ("semantic", "shutdown_timeout_seconds"),
            "语义模型关闭超时（秒）",
            "float",
            2.0,
            0.1,
            30.0,
            0.1,
        ),
        _FieldSpec(
            ("semantic", "max_text_chars"),
            "嵌入文本字数上限",
            "int",
            10000,
            32,
            50000,
        ),
        _FieldSpec(
            ("semantic", "max_text_bytes"),
            "嵌入文本字节上限",
            "int",
            40000,
            128,
            200000,
        ),
        _FieldSpec(("semantic", "device"), "语义推理设备", default="cpu"),
        _FieldSpec(
            ("semantic", "backend"),
            "语义推理后端",
            "enum",
            "torch",
            choices=("torch", "onnx", "openvino"),
        ),
        _FieldSpec(("semantic", "index_dir"), "语义索引目录", default=""),
        _FieldSpec(("semantic", "hnsw_m"), "HNSW M", "int", 16, 2, 64),
        _FieldSpec(
            ("semantic", "hnsw_ef_construction"),
            "HNSW 构建 ef",
            "int",
            200,
            8,
            2000,
        ),
        _FieldSpec(
            ("semantic", "hnsw_ef_search"),
            "HNSW 查询 ef",
            "int",
            64,
            2,
            2000,
        ),
    ),
    "tools": (
        _FieldSpec(("permissions", "bypass_approval"), "跳过审批", "bool", False),
        _FieldSpec(("permissions", "auto_allow_low_risk"), "低风险自动允许", "bool", True),
        _FieldSpec(("permissions", "approval_ttl_seconds"), "审批有效期（秒）", "int", 90, 5, 3600),
    ),
    "mcp": (_FieldSpec(("enabled",), "启用 MCP", "bool", False),),
    "watcher": (
        _FieldSpec(("enabled",), "启用窗口观察", "bool", False),
        _FieldSpec(("interval_seconds",), "轮询间隔（秒）", "float", 2.0, 0.1, 3600.0, 0.1),
        _FieldSpec(("poll_timeout_seconds",), "读取超时（秒）", "float", 5.0, 0.1, 3600.0, 0.1),
        _FieldSpec(("emit_initial",), "发送首次事件", "bool", False),
    ),
    "scheduler": (_FieldSpec(("enabled",), "启用调度器", "bool", True),),
    "behavior": (
        _FieldSpec(("enabled",), "启用自主行为", "bool", False),
        _FieldSpec(("min_interval_seconds",), "最小间隔（秒）", "float", 8.0, 0.1, 86400.0, 0.1),
        _FieldSpec(("max_interval_seconds",), "最大间隔（秒）", "float", 20.0, 0.1, 86400.0, 0.1),
        _FieldSpec(("probability",), "触发概率", "float", 0.35, 0.0, 1.0, 0.01),
        _FieldSpec(
            ("interaction_cooldown_seconds",),
            "互动后稳定时间（秒）",
            "float",
            0.35,
            0.0,
            10.0,
            0.05,
            tooltip="拖动或点击结束后暂缓自主行为，避免旧步进回弹。",
        ),
        _FieldSpec(("movement", "enabled"), "启用桌面漫游", "bool", False),
        _FieldSpec(
            ("movement", "min_distance_px"),
            "最小移动距离（像素）",
            "float",
            80.0,
            1.0,
            10000.0,
            1.0,
        ),
        _FieldSpec(
            ("movement", "max_distance_px"),
            "最大移动距离（像素）",
            "float",
            280.0,
            1.0,
            10000.0,
            1.0,
        ),
        _FieldSpec(
            ("movement", "step_pixels"),
            "步进距离（像素）",
            "float",
            36.0,
            4.0,
            1000.0,
            1.0,
        ),
        _FieldSpec(
            ("movement", "step_interval_seconds"),
            "步进间隔（秒）",
            "float",
            0.08,
            0.02,
            1.0,
            0.01,
        ),
        _FieldSpec(
            ("movement", "max_speed_px_per_second"),
            "最大移动速度（像素/秒）",
            "float",
            240.0,
            20.0,
            1200.0,
            10.0,
            tooltip="限制自主漫游速度，避免窗口快速跳动。",
        ),
        _FieldSpec(
            ("movement", "moving_motion"),
            "移动动作",
            "enum",
            "walk",
            choices=DEFAULT_MOTION_NAMES,
        ),
        _FieldSpec(
            ("movement", "idle_motion"),
            "结束动作",
            "enum",
            "idle",
            choices=DEFAULT_MOTION_NAMES,
        ),
        _FieldSpec(
            ("movement", "movement_identity"),
            "内部移动工具",
            default="pet:autonomous_move",
            tooltip="仅允许低风险内部工具；不要改为需要审批的工具。",
        ),
        _FieldSpec(("movement", "flip_facing"), "按移动方向转身", "bool", True),
        _FieldSpec(("movement", "max_steps"), "单次最多步数", "int", 64, 1, 64),
    ),
    "proactive": (
        _FieldSpec(("enabled",), "启用模型主动行为", "bool", False),
        _FieldSpec(("hourly_budget",), "每小时模型预算", "int", 6, 1, 120),
        _FieldSpec(("daily_budget",), "每日模型预算", "int", 24, 1, 1000),
        _FieldSpec(("global_cooldown_seconds",), "全局冷却（秒）", "float", 15.0, 0, 86400, 1),
        _FieldSpec(("dedupe_seconds",), "相同事件去重（秒）", "float", 60.0, 0, 86400, 1),
        _FieldSpec(("max_pending_events",), "最大等待事件", "int", 32, 1, 256),
        _FieldSpec(("memory_context_chars",), "记忆上下文字符", "int", 2400, 512, 12000),
        _FieldSpec(("max_tool_rounds",), "最多工具轮数", "int", 2, 1, 4),
    ),
    "rendering": (
        _FieldSpec(
            ("backend",),
            "渲染后端",
            "enum",
            "auto",
            choices=CANONICAL_RENDERER_BACKEND_CHOICES,
            tooltip="auto 自动选择；Vulkan 仅在 Qt 绑定和工厂均可初始化时启用。",
        ),
        _FieldSpec(("resource_root",), "资源目录", default="../resources"),
        _FieldSpec(
            ("model",),
            "Live2D 模型",
            default="",
            tooltip="填写资源根目录内精确的 model3 路径；留空自动选择排序后的首个模型。",
        ),
        _FieldSpec(("sprite_scale",), "精灵比例", "float", 0.6, 0.1, 2.0, 0.05),
        _FieldSpec(("frame_rate",), "绘制帧率", "float", 60.0, 15.0, 120.0, 1.0),
        _FieldSpec(
            ("geometry_audit_hz",),
            "几何审计频率",
            "float",
            30.0,
            1.0,
            60.0,
            1.0,
            tooltip="降低频率可减少顶点扫描；顶点脏标记仍会立即触发审计。",
        ),
    ),
    "storage": (_FieldSpec(("database",), "数据库路径", default="../data/meapet.sqlite3"),),
    "ui": (
        _FieldSpec(("always_on_top",), "窗口置顶", "bool", True),
        _FieldSpec(
            ("window_locked",),
            "锁定桌宠窗口",
            "bool",
            False,
            tooltip="锁定后不接收点击和拖动，但仍追踪光标并继续自主行为。",
        ),
        _FieldSpec(
            ("auto_open_model_setup",),
            "未配置时打开模型设置",
            "bool",
            True,
            tooltip="首次没有可用模型渠道时直接打开设置；关闭后仍可从控制台进入。",
        ),
        _FieldSpec(
            ("qt_platform",),
            "Qt 图形后端",
            "enum",
            "auto",
            choices=("auto", "xcb", "wayland"),
            tooltip="auto 在可用 Xwayland 时优先 xcb；纯 Wayland 保留降级能力。",
        ),
        _FieldSpec(("theme", "name"), "主题名称", default="MeaPet Fluent Dark"),
        _FieldSpec(
            ("theme", "mode"),
            "明暗模式",
            "enum",
            "dark",
            choices=("dark", "light"),
        ),
        _FieldSpec(("theme", "seed"), "主题种子色", default=""),
        _FieldSpec(("theme", "layout", "density"), "界面密度", "int", 0, -2, 2),
        _FieldSpec(
            ("theme", "layout", "radius_large_px"),
            "卡片圆角",
            "int",
            16,
            0,
            256,
        ),
        _FieldSpec(
            ("theme", "layout", "control_height_px"),
            "控件高度",
            "int",
            40,
            24,
            256,
        ),
        _FieldSpec(
            ("theme", "motion", "reduced_motion"),
            "减少动态效果",
            "bool",
            False,
        ),
        _FieldSpec(("hotkeys", "enabled"), "启用快捷键", "bool", True),
        _FieldSpec(("stream", "show_reasoning"), "显示思考片段", "bool", False),
        _FieldSpec(("stream", "show_murmur"), "显示碎碎念", "bool", True),
        _FieldSpec(("stream", "show_tool_status"), "显示工具状态", "bool", True),
        _FieldSpec(("stream", "max_bubble_chars"), "气泡字数上限", "int", 4000, 100, 100000),
    ),
}

_COMPLEX_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "app": (),
    "llm": (("channels",), ("models",), ("routing",)),
    "tools": (
        ("active_groups",),
        ("groups",),
        ("permissions", "allow"),
        ("permissions", "deny"),
        ("command_allowlist",),
    ),
    "tts": (("profiles",), ("routing",), ("reference_audios",), ("options",)),
    "mcp": (("servers",),),
    "memory": (("semantic",),),
    "scheduler": (("tasks",), ("triggers",)),
    "behavior": (("actions",),),
    "proactive": (("rules",),),
    "ui": (
        ("hotkeys", "bindings"),
        ("theme", "roles"),
        ("theme", "typography"),
        ("theme", "extra_stylesheet"),
    ),
}

_PATH_LABELS: dict[str, str] = {
    "channels": "模型渠道",
    "models": "模型别名",
    "routing": "路由策略",
    "active_groups": "启用工具组",
    "groups": "工具组定义",
    "allow": "允许列表",
    "deny": "拒绝列表",
    "command_allowlist": "命令白名单",
    "reference_audios": "语言参考音频",
    "profiles": "语音音色",
    "capture": "麦克风采集",
    "options": "后端高级参数",
    "servers": "MCP 服务器",
    "tasks": "定时任务",
    "triggers": "事件触发器",
    "actions": "自主行为动作",
    "bindings": "快捷键绑定",
    "roles": "主题颜色角色",
    "typography": "主题字体",
    "extra_stylesheet": "主题附加样式",
    "background": "人设背景",
    "traits": "性格特点",
    "goals": "人设目标",
    "boundaries": "行为边界",
    "likes": "喜欢的事物",
    "dislikes": "不喜欢的事物",
    "custom_instructions": "补充人设",
    "prompts": "模型提示词",
}

# 普通用户只需要知道“打开/关闭”和“选择哪种体验”。内部配置仍保留
# 精确枚举值用于兼容旧配置和适配器，但在默认界面统一映射为可读的按钮文案。
_FRIENDLY_VALUE_LABELS: dict[str, str] = {
    "DEBUG": "调试",
    "INFO": "常规",
    "WARNING": "警告",
    "ERROR": "错误",
    "CRITICAL": "严重",
    "auto": "自动",
    "always": "始终",
    "never": "关闭",
    "none": "不轮换",
    "time": "按时间",
    "size": "按大小",
    "text_only": "仅文字",
    "gpt_sovits_stdio": "GPT-SoVITS 标准输入",
    "gpt_sovits_http": "GPT-SoVITS",
    "subprocess": "本地语音进程",
    "sensevoice": "SenseVoice 本地识别",
    "zh": "中文",
    "ja": "日语",
    "en": "英语",
    "ko": "韩语",
    "yue": "粤语",
    "nospeech": "无语音",
    "wav": "WAV",
    "raw": "原始音频",
    "pcm": "PCM",
    "sentence_transformers": "SentenceTransformers",
    "torch": "Torch",
    "onnx": "ONNX",
    "openvino": "OpenVINO",
    "xcb": "X11",
    "wayland": "Wayland",
    "opengl": "OpenGL",
    "web_live2d": "Web Live2D",
    "sprite": "精灵",
    "vulkan": "Vulkan",
    "vllank": "3D 渲染预留（当前不可用）",
    "walk": "走动",
    "idle": "待机",
    "blink": "眨眼",
    "wave": "挥手",
}

# 这些字段在普通模式下可以安全地用一次点击完成。文本、路径、数值和
# 复杂映射仍由运行时策略管理，只有明确打开开发者选项后才展示原始编辑器。
_FRIENDLY_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "app": (
        ("name",),
        ("persona", "name"),
        ("persona", "role"),
        ("persona", "user_address"),
        ("persona", "relationship"),
        ("persona", "language"),
        ("persona", "speaking_style"),
        ("persona", "background"),
        ("persona", "traits"),
        ("persona", "goals"),
        ("persona", "boundaries"),
        ("persona", "likes"),
        ("persona", "dislikes"),
        ("persona", "custom_instructions"),
        ("persona", "proactive_enabled"),
        ("prompts", "dialogue"),
        ("prompts", "tool_guidance"),
        ("prompts", "memory_summary"),
        ("prompts", "memory_extract"),
    ),
    "logging": (
        ("level",),
        ("console", "enabled"),
        ("console", "level"),
        ("console", "color"),
        ("file", "enabled"),
    ),
    "tts": (
        ("enabled",),
        ("backend",),
        ("language",),
        ("role",),
        ("output_device_id",),
        ("health_probe",),
        ("startup_timeout_seconds",),
        ("shutdown_timeout_seconds",),
    ),
    "asr": (
        ("enabled",),
        ("backend",),
        ("language",),
        ("capture", "enabled"),
        ("capture", "max_duration_seconds"),
        ("capture", "auto_submit"),
        ("capture", "device_id"),
    ),
    "memory": (
        ("enabled",),
        ("recall_limit",),
        ("context_max_chars",),
        ("context_memory_item_chars",),
        ("recent_exchange_limit",),
        ("max_memories",),
        ("decay_days",),
        ("prune_days",),
        ("prune_importance_floor",),
        ("consolidation_enabled",),
        ("consolidation_similarity",),
        ("max_consolidation_memories",),
        ("summarize_every_n",),
        ("summary_chat_limit",),
        ("exchange_min_chars",),
        ("exchange_importance",),
        ("exchange_decay_factor",),
        ("auto_extract_enabled",),
        ("extract_max_items",),
        ("extract_max_chars",),
        ("extract_min_confidence",),
        ("extract_default_priority",),
        ("semantic", "enabled"),
        ("semantic", "provider"),
        ("semantic", "model_path"),
        ("semantic", "model_id"),
        ("semantic", "model_revision"),
        ("semantic", "dimension"),
        ("semantic", "batch_size"),
        ("semantic", "timeout_seconds"),
        ("semantic", "startup_timeout_seconds"),
        ("semantic", "shutdown_timeout_seconds"),
        ("semantic", "max_text_chars"),
        ("semantic", "max_text_bytes"),
        ("semantic", "device"),
        ("semantic", "backend"),
        ("semantic", "index_dir"),
        ("semantic", "hnsw_m"),
        ("semantic", "hnsw_ef_construction"),
        ("semantic", "hnsw_ef_search"),
    ),
    "tools": (
        ("permissions", "bypass_approval"),
        ("permissions", "auto_allow_low_risk"),
    ),
    "mcp": (("enabled",),),
    "watcher": (("enabled",),),
    "scheduler": (("enabled",),),
    "behavior": (
        ("enabled",),
        ("movement", "enabled"),
        ("movement", "moving_motion"),
        ("movement", "idle_motion"),
        ("movement", "flip_facing"),
    ),
    "proactive": (("enabled",),),
    "rendering": (("backend",),),
    "ui": (
        ("always_on_top",),
        ("window_locked",),
        ("hotkeys", "enabled"),
        ("stream", "show_reasoning"),
        ("stream", "show_murmur"),
        ("stream", "show_tool_status"),
    ),
}

_FRIENDLY_TOOL_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("desktop_observation", "读取窗口和进程", "允许桌宠读取当前前台窗口与进程摘要。"),
    (
        "desktop_control",
        "控制桌面坐标",
        "允许桌宠在你逐次确认后点击指定坐标；原生 Wayland 会拒绝执行。",
    ),
    (
        "desktop_automation",
        "多步桌面自动化",
        "允许桌宠在一次审批中执行有限的键盘、文本、鼠标和窗口步骤。",
    ),
    ("mcp", "MCP 扩展工具", "允许已启用的 MCP 服务注册工具；每次调用仍经过审批。"),
    ("pet_control", "控制桌宠动作", "允许模型调用移动、表情、动作、语音和点击穿透。"),
    ("scheduler", "定时任务", "允许调度器执行已配置的定时任务和触发器。"),
    (
        "system",
        "系统命令（需审批和白名单）",
        "启用后仍必须通过审批，并且命令必须在开发者白名单中。",
    ),
)

_MISSING = object()


def _vulkan_control_available() -> bool:
    """只在默认注册表同时具备工厂和 Qt Vulkan 绑定时启用选项。"""

    registration = default_renderer_registry().registration("vulkan")
    return bool(
        registration is not None
        and callable(registration.factory)
        and probe_qt_vulkan_runtime().available
    )


_DARK_STYLE = """
QWidget#configurationPanel, QWidget#configPage, QWidget#friendlyOverviewPage,
QWidget#configForm_app, QWidget#configForm_llm, QWidget#configForm_tts,
QWidget#configForm_logging, QWidget#configForm_tools, QWidget#configForm_mcp,
QWidget#configForm_watcher, QWidget#configForm_scheduler, QWidget#configForm_behavior,
QWidget#configForm_memory,
QWidget#configForm_rendering, QWidget#configForm_storage, QWidget#configForm_ui {
    background: #16111f;
    color: #faf6fb;
}
QFrame#configHeader, QGroupBox#configFriendlyCard, QGroupBox#configFriendlyToolGroups,
QGroupBox#configFieldCard,
QGroupBox#configComplexCard, QGroupBox#configMemoryParameters {
    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 #3d3154, stop: 0.5 #2e2440, stop: 1 #281f38);
    border: 1px solid #46385c;
    border-top-color: #7c69a0;
    border-radius: 14px;
    color: #faf6fb;
}
QFrame#configHeader { padding: 2px; }
QFrame#configHeader QLabel, QWidget#configPage QLabel { background: transparent; }
QLabel#configEyebrow {
    color: #ffc48f;
    font-size: 11px;
    font-weight: 700;
}
QLabel#configTitle {
    color: #faf6fb;
    font-size: 20px;
    font-weight: 700;
}
QLabel#configSubtitle, QLabel#configHint, QLabel#configPath {
    color: #d6cbe0;
}
QLabel#configBadge {
    color: #ffd3e0;
    background: #2e2440;
    border: 1px solid #ff9dbe;
    border-radius: 10px;
    padding: 4px 10px;
}
QCheckBox#configAutoSaveToggle {
    color: #ffd3e0;
    background: #221a2e;
    border: 1px solid #7c69a0;
    border-radius: 999px;
    padding: 7px 12px;
    spacing: 7px;
}
QCheckBox#configAutoSaveToggle:hover { border-color: #ffb6ce; background: #493757; }
QCheckBox#configAutoSaveToggle:checked {
    color: #2b0f1c;
    background: #ff9dbe;
    border-color: #ff9dbe;
}
QCheckBox#configAutoSaveToggle::indicator { width: 14px; height: 14px; border-radius: 7px; }
QCheckBox#configAutoSaveToggle:checked::indicator { background: #2b0f1c; border-color: #2b0f1c; }
QListWidget#configNav, QComboBox#configSectionPicker, QLineEdit#configSearch {
    background: #100c18;
    color: #faf6fb;
    border: 1px solid #7c69a0;
    border-radius: 10px;
    padding: 8px 10px;
}
QListWidget#configNav { padding: 8px; outline: none; }
QListWidget#configNav::item {
    color: #d6cbe0;
    padding: 10px 12px;
    border-radius: 8px;
}
QListWidget#configNav::item:selected {
    background: #674563;
    color: #fff1f5;
}
QLineEdit#configSearch:focus, QComboBox#configSectionPicker:focus {
    border: 1px solid #ff9dbe;
}
QComboBox#configSectionPicker QAbstractItemView,
QGroupBox#configFieldCard QComboBox QAbstractItemView {
    background: #2e2440;
    color: #faf6fb;
    selection-background-color: #674563;
    selection-color: #fff1f5;
    border: 1px solid #7c69a0;
    outline: 0;
    padding: 4px;
}
QGroupBox#configFriendlyCard, QGroupBox#configFriendlyToolGroups, QGroupBox#configFieldCard,
QGroupBox#configComplexCard, QGroupBox#configMemoryParameters {
    margin-top: 12px; padding: 12px;
}
QGroupBox#configFriendlyCard::title, QGroupBox#configFriendlyToolGroups::title,
QGroupBox#configFieldCard::title,
QGroupBox#configComplexCard::title, QGroupBox#configMemoryParameters::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #ffc48f;
}
QLabel#friendlySummary, QLabel#friendlyHint, QLabel#friendlyStatus {
    color: #d6cbe0;
    font-weight: 400;
}
QLabel#friendlyStatus { color: #ffd3e0; }
QGroupBox#configFriendlyToolGroups QCheckBox {
    background: transparent;
    color: #faf6fb;
    spacing: 8px;
    padding: 2px 0;
}
QGroupBox#configFriendlyCard QCheckBox, QGroupBox#configFieldCard QCheckBox {
    color: #faf6fb;
    spacing: 8px;
}
QPushButton#configChoiceButton {
    background: #221a2e;
    color: #faf6fb;
    border: 1px solid #7c69a0;
    border-radius: 9px;
    padding: 7px 12px;
    min-height: 32px;
}
QPushButton#configChoiceButton:hover { background: #493757; border-color: #ffb6ce; }
QPushButton#configChoiceButton:checked {
    background: #674563;
    color: #fff1f5;
    border: 2px solid #ff9dbe;
}
QCheckBox#configFriendlyToggle, QCheckBox#advancedEditToggle {
    color: #faf6fb;
    spacing: 8px;
}
QCheckBox#configFriendlyToggle {
    background: #221a2e;
    border: 1px solid #7c69a0;
    border-radius: 9px;
    padding: 8px 10px;
}
QCheckBox#configFriendlyToggle:checked {
    background: #674563;
    border-color: #ff9dbe;
}
QCheckBox::indicator {
    width: 18px; height: 18px;
    border: 1px solid #7c69a0;
    border-radius: 6px;
    background: #100c18;
}
QCheckBox::indicator:checked { background: #ff9dbe; border-color: #ff9dbe; }
QPushButton#developerOptionsButton, QPushButton#openModelSetupButton {
    background: transparent;
    color: #ffd3e0;
    border: 1px solid #7c69a0;
    border-radius: 9px;
    padding: 7px 12px;
}
QPushButton#developerOptionsButton:hover, QPushButton#openModelSetupButton:hover {
    background: #493757; border-color: #ffb6ce;
}
QTabWidget#configTabs::pane, QTabWidget::pane {
    border: 1px solid #46385c;
    border-radius: 12px;
    background: #221a2e;
}
QTabWidget#configTabs QTabBar::tab, QTabBar::tab {
    background: #221a2e;
    color: #b7a6c7;
    padding: 8px 16px;
    margin-right: 3px;
    border: 1px solid #46385c;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}
QTabWidget#configTabs QTabBar::tab:selected, QTabBar::tab:selected {
    background: #674563;
    color: #fff1f5;
    border-color: #ff9dbe;
}
QScrollArea, QAbstractScrollArea, QScrollArea#configScroll,
QScrollArea#friendlyOverviewScroll { border: none; background: #16111f; }
QAbstractScrollArea::viewport { background: #16111f; }
QScrollBar:vertical, QScrollBar:horizontal {
    background: #100c18; border: none; margin: 0;
}
QScrollBar:vertical { width: 12px; }
QScrollBar:horizontal { height: 12px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #674563; border: 1px solid #7c69a0; border-radius: 5px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: #100c18;
}
QScrollBar::add-line, QScrollBar::sub-line { background: #221a2e; border: none; }
QPlainTextEdit#configEditor, QGroupBox#configComplexCard QPlainTextEdit,
QGroupBox#configFieldCard QLineEdit, QGroupBox#configFieldCard QComboBox,
QGroupBox#configFieldCard QSpinBox, QGroupBox#configFieldCard QDoubleSpinBox {
    background: #100c18;
    color: #faf6fb;
    border: 1px solid #7c69a0;
    border-radius: 9px;
    padding: 8px;
    selection-background-color: #ff9dbe;
    selection-color: #2b0f1c;
}
QPlainTextEdit#configEditor:focus, QGroupBox#configComplexCard QPlainTextEdit:focus,
QGroupBox#configFieldCard QLineEdit:focus, QGroupBox#configFieldCard QComboBox:focus,
QGroupBox#configFieldCard QSpinBox:focus, QGroupBox#configFieldCard QDoubleSpinBox:focus {
    border-color: #ff9dbe;
}
QPushButton#configPrimary {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #ff9dbe, stop: 1 #ffc48f);
    color: #2b0f1c;
    border: none;
    border-radius: 9px;
    padding: 8px 16px;
    font-weight: 700;
}
QPushButton#configPrimary:hover { background: #ffb6ce; }
QPushButton#configPrimary:disabled, QPushButton#configSecondary:disabled {
    color: #8f829f;
    background: #21182d;
    border-color: #46385c;
}
QPushButton#configSecondary {
    background: #2e2440;
    color: #faf6fb;
    border: 1px solid #7c69a0;
    border-radius: 9px;
    padding: 8px 14px;
}
QPushButton#configSecondary:hover { background: #493757; border-color: #cdb8ff; }
QToolTip {
    color: #fff1f5;
    background: #21182d;
    border: 1px solid #8e70aa;
    padding: 6px 8px;
}
"""


def _configuration_stylesheet(values: Mapping[str, Any]) -> tuple[object, str]:
    """生成配置中心的 MD3 主题样式。"""

    theme = theme_from_configuration(values)
    stylesheet = "\n\n".join(
        (build_md3_stylesheet(theme), _configuration_component_stylesheet(theme))
    )
    return theme, stylesheet


def _configuration_component_stylesheet(theme: MD3Theme) -> str:
    """生成配置中心独有规则，供共享主题控制器合并应用。"""

    base = remap_legacy_stylesheet(_DARK_STYLE, theme)
    colors = theme.colors
    layout = theme.layout
    modern = f"""
QWidget#configurationPanel, QWidget#configPage,
QWidget#friendlyOverviewPage {{
    background: {colors.surface};
    color: {colors.on_surface};
}}
QFrame#configHeader, QFrame#configNavigationShell,
QFrame#configContentShell, QFrame#configCommandArea,
QGroupBox#configFriendlyCard, QGroupBox#configFriendlyToolGroups,
QGroupBox#configFieldCard, QGroupBox#configComplexCard,
QGroupBox#configMemoryParameters {{
    background: {colors.surface_container};
    color: {colors.on_surface};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_large_px}px;
}}
QFrame#configHeader {{
    background: {colors.surface_container_high};
    border-top: 2px solid {colors.primary};
}}
QLabel#configEyebrow {{ color: {colors.primary}; }}
QLabel#configSubtitle, QLabel#configHint, QLabel#configPath {{
    color: {colors.on_surface_variant};
}}
QLabel#configBadge {{
    color: {colors.on_primary_container};
    background: {colors.primary_container};
    border: 1px solid {colors.primary};
    border-radius: {layout.radius_medium_px}px;
}}
QListWidget#configNav, QComboBox#configSectionPicker,
QLineEdit#configSearch {{
    background: {colors.surface_container_low};
    color: {colors.on_surface};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_medium_px}px;
}}
QListWidget#configNav::item {{
    color: {colors.on_surface_variant};
    padding: 8px 10px;
    border-radius: {layout.radius_small_px}px;
}}
QListWidget#configNav::item:hover {{
    background: {colors.surface_container_high};
    color: {colors.on_surface};
}}
QListWidget#configNav::item:selected {{
    background: {colors.primary_container};
    color: {colors.on_primary_container};
}}
QLineEdit#configSearch:focus, QComboBox#configSectionPicker:focus {{
    border: 1px solid {colors.primary};
}}
QGroupBox#configFriendlyCard::title, QGroupBox#configFriendlyToolGroups::title,
QGroupBox#configFieldCard::title, QGroupBox#configComplexCard::title,
QGroupBox#configMemoryParameters::title {{
    color: {colors.primary};
}}
QCheckBox#configAutoSaveToggle, QCheckBox#configFriendlyToggle,
QCheckBox#advancedEditToggle {{
    color: {colors.on_surface};
    background: {colors.surface_container_low};
    border: 1px solid {colors.outline_variant};
    border-radius: 999px;
}}
QCheckBox#configAutoSaveToggle:hover, QCheckBox#configFriendlyToggle:hover,
QCheckBox#advancedEditToggle:hover {{
    border-color: {colors.primary};
    background: {colors.surface_container_high};
}}
QCheckBox#configAutoSaveToggle:checked, QCheckBox#configFriendlyToggle:checked {{
    color: {colors.on_primary_container};
    background: {colors.primary_container};
    border-color: {colors.primary};
}}
QTabWidget#configTabs::pane {{
    background: {colors.surface_container_low};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_large_px}px;
}}
QTabWidget#configTabs QTabBar::tab {{
    background: transparent;
    color: {colors.on_surface_variant};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 14px;
}}
QTabWidget#configTabs QTabBar::tab:hover {{
    background: {colors.surface_container};
    color: {colors.on_surface};
}}
QTabWidget#configTabs QTabBar::tab:selected {{
    color: {colors.primary};
    border-bottom-color: {colors.primary};
}}
QPlainTextEdit#configEditor, QGroupBox#configComplexCard QPlainTextEdit,
QGroupBox#configFieldCard QLineEdit, QGroupBox#configFieldCard QComboBox,
QGroupBox#configFieldCard QSpinBox, QGroupBox#configFieldCard QDoubleSpinBox {{
    background: {colors.surface_container_low};
    color: {colors.on_surface};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_medium_px}px;
}}
QPlainTextEdit#configEditor:focus, QGroupBox#configComplexCard QPlainTextEdit:focus,
QGroupBox#configFieldCard QLineEdit:focus, QGroupBox#configFieldCard QComboBox:focus,
QGroupBox#configFieldCard QSpinBox:focus, QGroupBox#configFieldCard QDoubleSpinBox:focus {{
    border: 1px solid {colors.primary};
}}
QPushButton#configPrimary {{
    background: {colors.primary};
    color: {colors.on_primary};
    border: 1px solid {colors.primary};
    border-radius: {layout.radius_medium_px}px;
    padding: 8px 14px;
    font-weight: 700;
}}
QPushButton#configPrimary:hover {{ background: {colors.primary_container}; }}
QPushButton#configSecondary, QPushButton#configChoiceButton,
QPushButton#developerOptionsButton, QPushButton#openModelSetupButton {{
    background: {colors.surface_container};
    color: {colors.on_surface};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_medium_px}px;
}}
QPushButton#configSecondary:hover, QPushButton#configChoiceButton:hover,
QPushButton#developerOptionsButton:hover, QPushButton#openModelSetupButton:hover {{
    background: {colors.surface_container_high};
    border-color: {colors.primary};
}}
QPushButton#configChoiceButton:checked {{
    background: {colors.primary_container};
    color: {colors.on_primary_container};
    border-color: {colors.primary};
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent;
    border: none;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {colors.outline};
    border: none;
    border-radius: 4px;
}}
QScrollBar::handle:hover {{ background: {colors.primary}; }}
QToolTip {{
    color: {colors.inverse_on_surface};
    background: {colors.inverse_surface};
    border: 1px solid {colors.outline_variant};
}}
"""
    return base + "\n\n" + modern.strip()


def _yaml_dump(value: object) -> str:
    import yaml

    return yaml.safe_dump(value, allow_unicode=True, sort_keys=False)


_SECRET_ERROR_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|credential|token|secret|password|authorization|cookie)",
    re.IGNORECASE,
)


def _safe_editor_error(error: BaseException) -> str:
    """避免 YAML 解析异常回显包含密钥的原始行。"""

    text = str(error or "").strip()
    if _SECRET_ERROR_PATTERN.search(text):
        return "错误详情包含敏感字段，已隐藏"
    if len(text) > 240:
        return text[:239] + "…"
    return text or type(error).__name__


_CONFIG_MESSAGE_LABELS = {
    "llm": "模型设置",
    "tts": "语音设置",
    "asr": "语音识别设置",
    "logging": "日志设置",
    "memory": "记忆策略",
    "tools": "工具权限",
    "mcp": "外部服务",
    "watcher": "窗口观察",
    "scheduler": "调度设置",
    "behavior": "自主行为",
    "rendering": "显示设置",
    "storage": "存储设置",
    "api_key_env": "密钥设置",
    "base_url": "服务地址",
    "channel_id": "服务名称",
    "capabilities": "能力设置",
    "protocol": "服务类型",
    "model": "模型名称",
}


def _friendly_config_message(value: object) -> str:
    """隐藏配置路径和格式术语，只保留可行动的用户提示。"""

    text = str(value or "")
    for source, target in _CONFIG_MESSAGE_LABELS.items():
        text = text.replace(source, target)
    text = text.replace("高级 YAML", "开发者设置").replace("YAML", "配置")
    return text


if pyside6_available:

    class AudioInputSelector(QComboBox):
        """按需刷新且只显示设备描述的麦克风选择器。"""

        refreshRequested = Signal()

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._devices: tuple[AudioInputDeviceChoice, ...] = ()
            self._devices_loaded = False
            self.setAccessibleName("麦克风输入设备")
            self.setMinimumContentsLength(18)
            self.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self._rebuild("")

        @property
        def devices_loaded(self) -> bool:
            return self._devices_loaded

        def selected_device_id(self) -> str:
            value = self.currentData()
            return str(value) if isinstance(value, str) else ""

        def set_selected_device_id(self, device_id: object) -> None:
            self._rebuild(str(device_id or ""))

        def set_devices(
            self,
            devices: Sequence[AudioInputDeviceChoice],
            *,
            loaded: bool = True,
            selected_device_id: object | None = None,
        ) -> None:
            if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes, bytearray)):
                raise TypeError("audio input devices must be a sequence")
            normalized = tuple(devices)
            if any(not isinstance(device, AudioInputDeviceChoice) for device in normalized):
                raise TypeError("audio input device entries are invalid")
            selected = (
                self.selected_device_id()
                if selected_device_id is None
                else str(selected_device_id or "")
            )
            self._devices = normalized
            self._devices_loaded = bool(loaded)
            self._rebuild(selected)

        def _rebuild(self, selected_device_id: str) -> None:
            self.blockSignals(True)
            try:
                self.clear()
                default_label = (
                    "跟随系统默认（当前无可用设备）"
                    if self._devices_loaded and not self._devices
                    else "跟随系统默认"
                )
                self.addItem(default_label, "")
                for device in self._devices:
                    label = (
                        f"{device.description}（系统默认）"
                        if device.is_default
                        else device.description
                    )
                    self.addItem(label, device.device_id)
                selected_index = self.findData(selected_device_id)
                if selected_device_id and selected_index < 0:
                    self.addItem("已选设备暂不可用", selected_device_id)
                    selected_index = self.count() - 1
                self.setCurrentIndex(max(0, selected_index))
                self.setToolTip(
                    "设备列表已刷新；设备标识只会写入本地配置。"
                    if self._devices_loaded
                    else "展开后读取可用输入设备；不会请求麦克风权限。"
                )
            finally:
                self.blockSignals(False)

        def showPopup(self) -> None:  # noqa: N802
            self.refreshRequested.emit()
            super().showPopup()

    def _confirm_save_dialog(parent: QWidget) -> bool:
        """使用项目主题显示保存确认，避免原生 QMessageBox 抢焦点。"""

        dialog = QDialog(parent)
        dialog.setObjectName("configSaveConfirmDialog")
        dialog.setWindowTitle("确认保存配置")
        dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        dialog.setModal(True)
        dialog_theme = getattr(parent, "_md3_theme", DARK_MD3_THEME)
        dialog_style = """
            QDialog#configSaveConfirmDialog { background: #16111f; color: #faf6fb; }
            QLabel#configSaveConfirmTitle { color: #ffc48f; font-size: 16px; font-weight: 700; }
            QLabel#configSaveConfirmText { color: #d6cbe0; }
            QDialogButtonBox QPushButton {
                background: #2e2440; color: #faf6fb; border: 1px solid #7c69a0;
                border-radius: 8px; padding: 7px 14px; min-height: 32px;
            }
            QDialogButtonBox QPushButton:hover { background: #493757; border-color: #ffb6ce; }
            QDialogButtonBox QPushButton#configSaveConfirmYes {
                background: #ff9dbe; color: #2b0f1c; border-color: #ff9dbe;
            }
            """
        dialog.setPalette(build_qt_palette(dialog_theme))
        dialog.setStyleSheet(
            "\n\n".join(
                (
                    build_md3_stylesheet(dialog_theme),
                    remap_legacy_stylesheet(dialog_style, dialog_theme),
                )
            )
        )
        title = QLabel("确认保存配置")
        title.setObjectName("configSaveConfirmTitle")
        text = QLabel("保存这些设置？未修改的敏感信息会保留。")
        text.setObjectName("configSaveConfirmText")
        text.setWordWrap(True)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        yes_button = buttons.button(QDialogButtonBox.StandardButton.Yes)
        no_button = buttons.button(QDialogButtonBox.StandardButton.No)
        if yes_button is not None:
            yes_button.setObjectName("configSaveConfirmYes")
            yes_button.setText("保存")
            yes_button.setAccessibleName("确认保存配置")
        if no_button is not None:
            no_button.setText("取消")
            no_button.setAccessibleName("取消保存配置")
            no_button.setDefault(True)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(text)
        layout.addWidget(buttons)
        return dialog.exec() == QDialog.DialogCode.Accepted

    class ConfigurationPanel(QWidget):
        """面向桌宠运行时的配置编辑页。"""

        saveRequested = Signal(object)
        validateRequested = Signal(object)
        dirtyChanged = Signal(bool)
        statusChanged = Signal(str)
        modelSetupRequested = Signal()
        advancedEditingChanged = Signal(bool)

        def __init__(
            self,
            values: Mapping[str, Any] | None = None,
            *,
            path: str | Path | None = None,
            parent: QWidget | None = None,
            developer_mode: bool = True,
            auto_save: bool | None = None,
            audio_input_device_loader: Callable[[], object] | None = None,
        ) -> None:
            super().__init__(parent)
            self._developer_mode = bool(developer_mode)
            self.setObjectName("configurationPanel")
            # 控制台默认宽度可能只有 520px；过大的硬性最小宽度会把
            # 配置页右侧编辑器和保存按钮裁掉。导航和表单本身可伸缩，
            # 窄屏由各页滚动区域承载，不应强制顶层窗口横向溢出。
            # 配置页自身也可能被独立向导或低分辨率控制台承载；保留
            # 420×500 的可用下限，表单/高级编辑由内部滚动区承载，
            # 避免顶层窗口被强制撑出屏幕。
            self.setMinimumSize(420, 500)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._md3_theme, _stylesheet = _configuration_stylesheet(values or {})
            self.setProperty("md3Role", "root")
            self._fancy_style_controller = FancyStyleController(self._md3_theme, self)
            self._path = Path(path).expanduser() if path else None
            self._original: dict[str, Any] = {}
            self._display_draft: dict[str, Any] = {}
            self._baseline: dict[str, str] = {}
            self._section_editors: dict[str, QPlainTextEdit] = {}
            self._form_controls: dict[str, dict[tuple[str, ...], QWidget]] = {}
            self._complex_editors: dict[str, dict[tuple[str, ...], QPlainTextEdit]] = {}
            self._complex_cards: dict[str, dict[tuple[str, ...], QGroupBox]] = {}
            self._field_cards: dict[str, QGroupBox] = {}
            self._friendly_controls: dict[str, dict[tuple[str, ...], QWidget]] = {}
            self._friendly_tool_group_controls: dict[str, QCheckBox] = {}
            self._friendly_pages: dict[str, QWidget] = {}
            self._friendly_choice_buttons: dict[
                tuple[str, tuple[str, ...]], tuple[QPushButton, ...]
            ] = {}
            self._friendly_status_labels: dict[str, QLabel] = {}
            self._audio_input_devices: tuple[AudioInputDeviceChoice, ...] = ()
            self._audio_input_devices_loaded = False
            self._audio_input_device_loader = audio_input_device_loader
            self._overview_status_labels: dict[str, QLabel] = {}
            self._page_hosts: dict[str, QWidget] = {}
            self._built_sections: set[str] = set()
            self._overview_grid: QGridLayout | None = None
            self._overview_cards: tuple[QWidget, ...] = ()
            self._overview_columns = 0
            self._yaml_tabs: dict[str, tuple[QTabWidget, int]] = {}
            self._page_apply_buttons: dict[str, QPushButton] = {}
            self._form_layouts: dict[str, QVBoxLayout] = {}
            self._building_form = False
            self._advanced_editing = False
            self._save_in_flight = False
            # 无保存回调的独立面板不应把 saveRequested 永久置为 in-flight；
            # 组合根显式传入 True，带路径的独立面板也可自动保存。
            self._auto_save_override = auto_save
            self._auto_save_enabled = bool(auto_save) if auto_save is not None else path is not None
            self._auto_save_pending = False
            self._auto_save_blocked = False
            self._auto_save_edit_generation = 0
            self._save_edit_generation: int | None = None
            self._auto_save_initialized = False
            self._auto_save_timer = QTimer(self)
            self._auto_save_timer.setSingleShot(True)
            self._auto_save_timer.setInterval(700)
            self._auto_save_timer.timeout.connect(self._auto_save)
            self._advanced_editor: QPlainTextEdit | None = None
            self._overview_editor: QPlainTextEdit | None = None
            self._advanced_hint: QLabel | None = None
            self._advanced_toggle: QCheckBox | None = None
            self._model_setup_button: QPushButton | None = None
            self._scheduler_panel: SchedulerPanel | None = None
            self._navigation_panel: QWidget | None = None
            self._content_panel: QWidget | None = None
            self._command_bar: QWidget | None = None
            self._body_layout: QBoxLayout | None = None
            self._section_picker: QComboBox | None = None
            self._responsive_narrow = False
            self._compact_layout_requested = False
            self._vertical_compact: bool | None = None
            self._section_paths: dict[str, tuple[str, ...]] = {
                key: section_path for key, _title, section_path, _hint in _SECTIONS
            }
            self._build_ui()
            self.set_values(values or {})
            self._fancy_style_controller.attach(
                self,
                extra_stylesheet=_configuration_component_stylesheet(self._md3_theme),
            )

        def _build_ui(self) -> None:
            header = FancyCard(
                elevated=True,
                theme=self._md3_theme,
            )
            header.setObjectName("configHeader")
            header_layout = header.content_layout
            header_layout.setSpacing(4)
            self._fancy_style_controller.register(header)
            eyebrow = QLabel("MEAPET  ·  工作区设置")
            eyebrow.setObjectName("configEyebrow")
            eyebrow.setAccessibleName("配置中心标识")
            title = QLabel("配置中心")
            title.setObjectName("configTitle")
            subtitle = QLabel("常用设置用开关和预设完成；模型、语音和桌面行为都可以直接点按。")
            subtitle.setObjectName("configSubtitle")
            self._path_label = QLabel("配置文件：未指定")
            self._path_label.setObjectName("configPath")
            self._path_label.setVisible(self._developer_mode)
            self._status = QLabel("未加载配置")
            self._status.setObjectName("configHint")
            self._status.setWordWrap(True)
            self._summary = QLabel(f"{len(_SECTIONS)} 个配置域 · 未保存")
            self._summary.setObjectName("configBadge")
            self._summary.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            header_layout.addWidget(eyebrow)
            title_row = QHBoxLayout()
            title_row.addWidget(title, 1)
            title_row.addWidget(self._summary)
            header_layout.addLayout(title_row)
            header_layout.addWidget(subtitle)
            header_layout.addWidget(self._path_label)
            status_row = QHBoxLayout()
            status_row.addWidget(self._status, 1)
            header_layout.addLayout(status_row)
            self._status.setVisible(False)
            self._status_info = FancyInfoBar(
                "配置状态",
                "未加载配置",
                severity="info",
                closable=False,
                theme=self._md3_theme,
            )
            self._status_info.setObjectName("configStatusInfoBar")
            self._fancy_style_controller.register(self._status_info)
            header_layout.addWidget(self._status_info)
            advanced_toggle = QCheckBox("开发者选项：显示高级设置")
            advanced_toggle.setObjectName("advancedEditToggle")
            advanced_toggle.setToolTip(
                "普通使用无需打开；仅在需要排查兼容性或保留未识别设置时使用。"
            )
            advanced_toggle.setAccessibleName("开发者高级设置")
            advanced_toggle.toggled.connect(self.set_advanced_editing)
            self._advanced_toggle = advanced_toggle
            advanced_toggle.setVisible(self._developer_mode)
            header_layout.addWidget(advanced_toggle)

            navigation_panel = QFrame()
            navigation_panel.setObjectName("configNavigationShell")
            navigation_panel.setProperty("md3Role", "card-outlined")
            navigation_layout = QVBoxLayout(navigation_panel)
            navigation_layout.setContentsMargins(10, 12, 10, 12)
            navigation_layout.setSpacing(8)
            navigation_title = QLabel("设置")
            navigation_title.setObjectName("configNavigationTitle")
            navigation_title.setProperty("md3Role", "title")
            navigation_layout.addWidget(navigation_title)
            search = QLineEdit()
            search.setObjectName("configSearch")
            search.setPlaceholderText("搜索配置域…")
            search.setClearButtonEnabled(True)
            search.textChanged.connect(self._filter_sections)
            self._search = search
            self._navigation = QListWidget()
            self._navigation.setObjectName("configNav")
            # WinUI 设置页在宽屏保留稳定的分类导航；窄屏由选择器接管。
            self._navigation.setMinimumWidth(0)
            self._navigation.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            self._navigation.setToolTip("选择配置域；搜索仅过滤导航，不会删除字段。")
            navigation_layout.addWidget(search)
            navigation_layout.addWidget(self._navigation, 1)
            section_picker = QComboBox()
            section_picker.setObjectName("configSectionPicker")
            section_picker.setMinimumHeight(36)
            section_picker.setToolTip("选择要配置的功能区域")
            section_picker.setAccessibleName("配置区域选择")
            for key, label, _section_path, hint in _SECTIONS:
                section_picker.addItem(label, key)
                section_picker.setItemData(
                    section_picker.count() - 1,
                    hint,
                    Qt.ItemDataRole.ToolTipRole,
                )
            section_picker.setVisible(False)
            self._section_picker = section_picker
            navigation_layout.addWidget(section_picker)
            self._stack = FancyNavigationView(
                theme=self._md3_theme,
                navigation_width=188,
            )
            self._stack.setObjectName("configFancyNavigation")
            # 可搜索 QListWidget 保持既有宿主 API；页面与视觉层由 FancyUI 导航承载。
            self._stack.tabBar().setVisible(False)
            self._fancy_style_controller.register(self._stack)
            for key, label, section_path, hint in _SECTIONS:
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, key)
                item.setToolTip(hint)
                self._navigation.addItem(item)
                page_host = QWidget()
                page_host.setObjectName(f"configPageHost_{key}")
                page_host_layout = QVBoxLayout(page_host)
                page_host_layout.setContentsMargins(0, 0, 0, 0)
                page_host_layout.setSpacing(0)
                self._page_hosts[key] = page_host
                if key == "overview":
                    page_host_layout.addWidget(self._make_page(key, label, section_path, hint))
                    self._built_sections.add(key)
                self._stack.addPage(
                    page_host,
                    label,
                    icon=_SECTION_ICONS[key],
                    tooltip=hint,
                )
            self._navigation.currentRowChanged.connect(self._on_navigation_row_changed)
            section_picker.currentIndexChanged.connect(self._on_section_picker_changed)
            self._navigation.setCurrentRow(0)
            self.set_advanced_editing(False, announce=False)

            self._auto_save_toggle = QCheckBox("自动保存")
            self._auto_save_toggle.setObjectName("configAutoSaveToggle")
            self._auto_save_toggle.setChecked(True)
            self._auto_save_toggle.setToolTip("编辑停止 700 毫秒后自动校验并写入配置文件")
            self._auto_save_toggle.toggled.connect(self.set_auto_save_enabled)
            # 操作栏必须先创建再加入自动保存开关；否则首次打开配置中心
            # 会在 Qt 运行时触发 UnboundLocalError，导致所有配置控件不可用。
            actions_host = QFrame()
            actions_host.setObjectName("configCommandArea")
            actions_host.setProperty("md3Role", "card-outlined")
            actions = QHBoxLayout(actions_host)
            actions.setContentsMargins(8, 6, 8, 6)
            actions.setSpacing(8)
            actions.addWidget(self._auto_save_toggle)
            command_bar = FancyCommandBar(theme=self._md3_theme)
            command_bar.setObjectName("configCommandBar")
            command_bar.addCommand(
                "校验草稿",
                self._validate_draft,
                icon="success",
                tooltip="校验当前草稿，不写入文件。",
            )
            command_bar.addCommand(
                "撤销修改",
                self.reset_draft,
                icon="back",
                tooltip="恢复到最近一次加载或保存的配置。",
            )
            self._fancy_style_controller.register(command_bar)
            actions.addWidget(command_bar, 1)
            self._save_button = QPushButton("立即保存")
            self._save_button.setObjectName("configPrimary")
            self._save_button.setIcon(fancy_icon("save", theme=self._md3_theme))
            self._save_button.clicked.connect(self._confirm_save)
            actions.addWidget(self._save_button)

            content_panel = QFrame()
            content_panel.setObjectName("configContentShell")
            content_panel.setProperty("md3Role", "card-outlined")
            content_layout = QVBoxLayout(content_panel)
            content_layout.setContentsMargins(0, 0, 0, 0)
            content_layout.addWidget(self._stack)
            body = QVBoxLayout()
            body.setDirection(QBoxLayout.Direction.LeftToRight)
            body.setSpacing(12)
            self._body_layout = body
            self._navigation_panel = navigation_panel
            self._content_panel = content_panel
            self._command_bar = command_bar
            body.addWidget(navigation_panel)
            body.addWidget(content_panel, 1)
            root = QVBoxLayout(self)
            root.setContentsMargins(14, 14, 14, 14)
            root.setSpacing(12)
            root.addWidget(header)
            root.addLayout(body, 1)
            root.addWidget(actions_host)
            self._update_responsive_layout()

        def _ensure_section_page(self, key: str) -> bool:
            """按导航访问创建一个配置域，并同步当前草稿。"""

            normalized = str(key)
            if normalized in self._built_sections:
                return True
            host = self._page_hosts.get(normalized)
            definition = next(
                (entry for entry in _SECTIONS if entry[0] == normalized),
                None,
            )
            if host is None or definition is None:
                return False
            layout = host.layout()
            if layout is None:
                raise RuntimeError(f"configuration page host has no layout: {normalized}")
            _key, label, section_path, hint = definition
            layout.addWidget(self._make_page(normalized, label, section_path, hint))
            self._built_sections.add(normalized)
            editor = self._section_editors[normalized]
            value = (
                self._display_draft
                if not section_path
                else self._section_value(self._display_draft, section_path)
            )
            text = _yaml_dump(value)
            editor.blockSignals(True)
            editor.setPlainText(text)
            editor.blockSignals(False)
            self._baseline[normalized] = text
            self._sync_forms()
            self._apply_advanced_editing_visibility()
            return True

        def _ensure_all_section_pages(self) -> None:
            """为显式程序化控件访问补齐全部页面。"""

            for key, _label, _section_path, _hint in _SECTIONS:
                self._ensure_section_page(key)

        def select_section(self, key: object) -> bool:
            """选择一个公开配置区段，供桌面/网页控制台精准跳转。"""

            normalized = str(key or "").strip().lower()
            index = next(
                (
                    position
                    for position, (section, *_rest) in enumerate(_SECTIONS)
                    if section == normalized
                ),
                -1,
            )
            if index < 0 or not self._ensure_section_page(normalized):
                return False
            self._navigation.setCurrentRow(index)
            return True

        def _on_navigation_row_changed(self, row: int) -> None:
            """同步列表、窄屏选择器和页面堆栈。"""

            if row < 0:
                return
            item = self._navigation.item(row)
            if item is not None:
                self._ensure_section_page(str(item.data(Qt.ItemDataRole.UserRole) or ""))
            self._stack.setCurrentIndex(row)
            picker = self._section_picker
            if picker is not None and picker.currentIndex() != row:
                picker.blockSignals(True)
                picker.setCurrentIndex(row)
                picker.blockSignals(False)

        def _on_section_picker_changed(self, index: int) -> None:
            """处理窄屏页面选择器，复用稳定的导航行号。"""

            if index < 0 or index >= self._navigation.count():
                return
            if self._navigation.currentRow() != index:
                self._navigation.setCurrentRow(index)

        def _update_responsive_layout(self) -> None:
            """在窄窗口改为上下布局，避免导航挤压表单内容。"""

            if self._body_layout is None or self._navigation_panel is None:
                return
            narrow = self.width() < 860
            if narrow == self._responsive_narrow:
                return
            self._responsive_narrow = narrow
            picker = self._section_picker
            if narrow:
                self._body_layout.setDirection(QBoxLayout.Direction.TopToBottom)
                self._navigation_panel.setMinimumWidth(0)
                self._navigation_panel.setMaximumWidth(16777215)
                self._navigation_panel.setMaximumHeight(96)
                self._search.setVisible(False)
                self._navigation.setVisible(False)
                if picker is not None:
                    picker.setVisible(True)
            else:
                self._body_layout.setDirection(QBoxLayout.Direction.LeftToRight)
                self._navigation_panel.setFixedWidth(196)
                self._navigation_panel.setMaximumHeight(16777215)
                self._search.setVisible(True)
                self._navigation.setVisible(True)
                if picker is not None:
                    picker.setVisible(False)
            self.setProperty("responsiveMode", "stacked" if narrow else "split")
            style = self.style()
            style.unpolish(self)
            style.polish(self)
            self.updateGeometry()

        def resizeEvent(self, event: object) -> None:
            """响应桌面/小屏尺寸变化，保持配置内容可达。"""

            super().resizeEvent(event)
            self._update_responsive_layout()
            self._update_vertical_density()
            self._reflow_overview_cards()

        def showEvent(self, event: object) -> None:  # noqa: N802
            """首次显示后按真实 viewport 宽度重排总览卡片。"""

            super().showEvent(event)
            # 构造阶段总览网格尚未拥有最终 viewport 宽度；首次 show
            # 之后强制重新计算，避免低分辨率窗口保留两/三列旧几何。
            self._overview_columns = 0
            self._reflow_overview_cards()

        def set_compact_layout(self, enabled: bool) -> None:
            """在极小屏控制台中隐藏冗长配置页头，保留点击式页面和保存区。"""

            self._compact_layout_requested = bool(enabled)
            self.setMinimumWidth(1 if self._compact_layout_requested else 420)
            self.setMinimumHeight(1 if self._compact_layout_requested else 500)
            self._update_vertical_density()
            self._update_responsive_layout()
            self.updateGeometry()

        def _update_vertical_density(self) -> None:
            """在嵌入高度不足时收起重复页头，把空间留给可滚动设置卡。"""

            compact = self._compact_layout_requested or self.height() < 720
            if compact == self._vertical_compact:
                return
            self._vertical_compact = compact
            header = self.findChild(QFrame, "configHeader")
            if header is not None:
                header.setVisible(not compact)
            for label in self.findChildren(QLabel):
                role = str(label.property("compactRole") or "")
                if role in {"description", "overviewDetail"}:
                    label.setVisible(not compact)
            self.setProperty("compactLayout", compact)
            style = self.style()
            style.unpolish(self)
            style.polish(self)
            self.updateGeometry()

        def _make_page(
            self,
            key: str,
            title: str,
            section_path: tuple[str, ...],
            hint: str,
        ) -> QWidget:
            page = QWidget()
            page.setObjectName("configPage")
            layout = QVBoxLayout(page)
            layout.setContentsMargins(16, 14, 16, 14)
            layout.setSpacing(10)
            page_header = FancyCard(
                elevated=False,
                theme=self._md3_theme,
            )
            page_header.setObjectName("configPageHeader")
            page_header_layout = page_header.content_layout
            page_header_layout.setSpacing(3)
            self._fancy_style_controller.register(page_header)
            heading = QLabel(title)
            heading.setObjectName("configTitle")
            heading.setProperty("compactRole", "heading")
            description = QLabel(hint)
            description.setObjectName("configHint")
            description.setProperty("compactRole", "description")
            description.setWordWrap(True)
            page_header_layout.addWidget(heading)
            page_header_layout.addWidget(description)
            layout.addWidget(page_header)

            editor = self._make_yaml_editor(key, read_only=key == "overview")
            if key in {"overview", "advanced"}:
                if key == "advanced":
                    self._advanced_editor = editor
                    hint_label = QLabel(
                        "开发者设置默认隐藏。打开顶部开关后，可查看完整配置并保留未识别设置。"
                    )
                    hint_label.setObjectName("configHint")
                    hint_label.setWordWrap(True)
                    self._advanced_hint = hint_label
                    layout.addWidget(hint_label)
                    layout.addWidget(editor, 1)
                else:
                    # 总览只显示脱敏状态卡；完整 YAML 仍保留在内部以兼容
                    # 旧宿主，但普通用户不应在进入配置中心时看到原始参数。
                    overview = self._make_overview_page()
                    layout.addWidget(overview, 1)
                    editor.setVisible(False)
                    self._overview_editor = editor
                    layout.addWidget(editor, 1)
            else:
                tabs = QTabWidget()
                tabs.setObjectName("configTabs")
                tabs.addTab(self._make_form_page(key), "可视化表单")
                yaml_page = QWidget()
                yaml_page.setProperty("md3Role", "root")
                yaml_layout = QVBoxLayout(yaml_page)
                yaml_layout.setContentsMargins(4, 4, 4, 4)
                yaml_hint = QLabel(
                    "复杂设置仅在开发者选项中使用；保存时会校验并保留未修改的敏感信息。"
                )
                yaml_hint.setObjectName("configHint")
                yaml_hint.setWordWrap(True)
                yaml_layout.addWidget(yaml_hint)
                yaml_layout.addWidget(editor, 1)
                yaml_index = tabs.addTab(yaml_page, "完整配置")
                self._yaml_tabs[key] = (tabs, yaml_index)
                layout.addWidget(tabs, 1)

            apply_button = QPushButton("应用此页到草稿")
            apply_button.setObjectName("configSecondary")
            apply_button.setToolTip("解析当前页面并写入内存草稿，不会立即写入文件。")
            apply_button.clicked.connect(lambda: self.apply_section(key))
            self._page_apply_buttons[key] = apply_button
            # 总览是只读状态卡；普通点击式页面会在每次点按后立即写入
            # 草稿，因此“应用此页”只在开发者编辑模式下出现。
            apply_button.setVisible(False)
            footer = QHBoxLayout()
            footer.addStretch(1)
            footer.addWidget(apply_button)
            layout.addLayout(footer)
            return page

        def _make_yaml_editor(self, key: str, *, read_only: bool = False) -> QPlainTextEdit:
            editor = QPlainTextEdit()
            editor.setObjectName("configEditor")
            editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
            editor.setTabStopDistance(4 * editor.fontMetrics().horizontalAdvance(" "))
            editor.setFont(QFont("Monospace"))
            editor.setReadOnly(read_only)
            editor.setPlaceholderText("YAML 配置")
            editor.textChanged.connect(self._on_editor_changed)
            self._section_editors[key] = editor
            return editor

        @staticmethod
        def _friendly_value(value: object) -> str:
            """把内部枚举映射为用户可读文案，不回显未知原始值。"""

            text = str(value or "").strip()
            return _FRIENDLY_VALUE_LABELS.get(text, "当前选项")

        @staticmethod
        def _spec_by_path(key: str) -> dict[tuple[str, ...], _FieldSpec]:
            return {spec.path: spec for spec in _FIELD_SPECS.get(key, ())}

        def _make_overview_page(self) -> QWidget:
            """构造不包含 YAML 的配置状态总览。"""

            page = QWidget()
            page.setObjectName("friendlyOverviewPage")
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 8, 0, 0)
            layout.setSpacing(8)
            hint = QLabel(
                "这里显示运行状态和下一步；需要修改时进入对应卡片，常用选项都可以直接点按。"
            )
            hint.setObjectName("friendlyHint")
            hint.setProperty("compactRole", "description")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            grid_host = QWidget()
            # QScrollArea 的承载 QWidget 不会自动继承 viewport 背景；显式
            # 填充可避免普通样式在 Linux Fusion 主题下出现白色侧带。
            grid_host.setProperty("md3Role", "root")
            grid = QGridLayout(grid_host)
            grid.setContentsMargins(2, 4, 12, 16)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(10)
            grid_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            self._overview_status_labels.clear()
            sections = (
                ("app", "桌宠人设", "名字、关系、说话风格和主动行为"),
                ("llm", "模型对话", "模型渠道和流式回复"),
                ("tts", "语音输出", "流式语音和语言"),
                ("asr", "语音识别", "SenseVoice 本地识别和 IPC 工作进程"),
                ("memory", "记忆策略", "召回、上下文和生命周期"),
                ("behavior", "自主行为", "桌面漫游和随机动作"),
                ("tools", "工具权限", "审批和低风险策略"),
                ("ui", "窗口体验", "置顶、快捷键和输出"),
                ("logging", "日志记录", "等级、颜色和文件日志"),
                ("rendering", "显示效果", "Live2D、OpenGL 和精灵后端"),
            )
            cards: list[QWidget] = []
            for key, title, description in sections:
                card = FancyCard(
                    title,
                    icon=_SECTION_ICONS[key],
                    interactive=False,
                    theme=self._md3_theme,
                )
                card.setObjectName("configFriendlyCard")
                self._fancy_style_controller.register(card)
                card_layout = card.content_layout
                status = QLabel()
                status.setObjectName("friendlyStatus")
                status.setProperty("compactRole", "overviewDetail")
                status.setWordWrap(True)
                card_layout.addWidget(status)
                desc = QLabel(description)
                desc.setObjectName("friendlyHint")
                desc.setProperty("compactRole", "overviewDetail")
                desc.setWordWrap(True)
                card_layout.addWidget(desc)
                action = QPushButton("配置模型" if key == "llm" else "打开设置")
                action.setObjectName("configChoiceButton")
                action.setProperty("role", "primary" if key == "llm" else "")
                if key == "llm":
                    action.clicked.connect(self.modelSetupRequested)
                else:
                    action.clicked.connect(
                        lambda _checked=False, selected=key: self._open_section(selected)
                    )
                card_layout.addWidget(action)
                self._overview_status_labels[key] = status
                cards.append(card)
            self._overview_grid = grid
            self._overview_cards = tuple(cards)
            self._reflow_overview_cards()
            overview_scroll = QScrollArea()
            overview_scroll.setObjectName("friendlyOverviewScroll")
            overview_scroll.setWidgetResizable(True)
            overview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            overview_scroll.setFrameShape(QFrame.Shape.NoFrame)
            overview_scroll.setWidget(grid_host)
            layout.addWidget(overview_scroll, 1)
            return page

        def _reflow_overview_cards(self) -> None:
            """按当前可用宽度重排总览卡片，禁止窄屏横向溢出。"""

            grid = self._overview_grid
            cards = self._overview_cards
            if grid is None or not cards:
                return
            host = grid.parentWidget()
            available_width = max(
                1,
                int(host.width()) if host is not None else 0,
                int(self.width()),
            )
            if available_width < 500:
                columns = 1
            elif available_width < 780:
                columns = 2
            else:
                columns = 3
            if columns == self._overview_columns and grid.count() == len(cards):
                return
            while grid.count():
                item = grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
            for column in range(3):
                grid.setColumnStretch(column, 1 if column < columns else 0)
            for index, card in enumerate(cards):
                card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                grid.addWidget(card, index // columns, index % columns)
                # `setParent(None)` 在清空旧网格时会把卡片标记为隐藏；
                # 重新加入布局不会在所有 Qt 样式/平台上自动显示，
                # 否则窄屏切换后总览会出现空白但仍占用滚动高度。
                card.show()
            self._overview_columns = columns
            if host is not None:
                host.adjustSize()

        def _open_section(self, key: str) -> bool:
            """从总览卡片跳转到对应配置域。"""

            selected = str(key or "").strip()
            for index in range(self._navigation.count()):
                item = self._navigation.item(index)
                if str(item.data(Qt.ItemDataRole.UserRole) or "") == selected:
                    self._navigation.setCurrentRow(index)
                    return True
            return False

        def open_section(self, key: str) -> bool:
            """打开公开配置域，供模块中心复用同一草稿与自动保存管线。"""

            return self._open_section(key)

        def _make_friendly_page(self, key: str) -> QWidget:
            """构造默认点击式设置卡，隐藏原始数值编辑表面。"""

            card = FancyCard(
                "常用设置",
                icon=_SECTION_ICONS.get(key, "file"),
                theme=self._md3_theme,
            )
            card.setObjectName("configFriendlyCard")
            self._fancy_style_controller.register(card)
            layout = card.content_layout
            layout.setSpacing(8)
            controls: dict[tuple[str, ...], QWidget] = {}
            buttons: dict[tuple[str, ...], tuple[QPushButton, ...]] = {}
            self._friendly_controls[key] = controls
            self._friendly_pages[key] = card
            specs = self._spec_by_path(key)
            paths = _FRIENDLY_PATHS.get(key, ())
            numeric_card: QGroupBox | None = None
            numeric_layout: QFormLayout | None = None
            app_text_cards: dict[str, QFormLayout] = {}
            if key == "tools":
                group_card = QGroupBox("可用工具")
                group_card.setObjectName("configFriendlyToolGroups")
                group_layout = QVBoxLayout(group_card)
                group_hint = QLabel("按需点选能力；系统命令始终受审批和白名单限制。")
                group_hint.setObjectName("friendlyHint")
                group_hint.setWordWrap(True)
                group_layout.addWidget(group_hint)
                self._friendly_tool_group_controls.clear()
                for group, label, tooltip in _FRIENDLY_TOOL_GROUPS:
                    toggle = QCheckBox(label)
                    toggle.setObjectName(f"configToolGroup_{group}")
                    toggle.setToolTip(tooltip)
                    toggle.setAccessibleName(label)
                    toggle.toggled.connect(
                        lambda value, selected=group: self._set_tool_group(selected, bool(value))
                    )
                    self._friendly_tool_group_controls[group] = toggle
                    group_layout.addWidget(toggle)
                layout.addWidget(group_card)
            if key == "llm":
                hint = QLabel(
                    "模型渠道和 API 密钥请使用下方向导；密钥只填写环境变量名，不会在此页回显。"
                )
                hint.setObjectName("friendlyHint")
                hint.setWordWrap(True)
                layout.addWidget(hint)
            if key == "app":
                hint = QLabel(
                    "人设与提示词会在保存后热重载；列表字段每行一项，提示词中不要填写密钥。"
                )
                hint.setObjectName("friendlyHint")
                hint.setWordWrap(True)
                layout.addWidget(hint)
            if key == "tts":
                hint = QLabel(
                    "先选择语音后端和输出语言；可用音色及启动健康状态会同步显示在主控制台。"
                )
                hint.setObjectName("friendlyHint")
                hint.setWordWrap(True)
                layout.addWidget(hint)
            if key == "memory":
                hint = QLabel(
                    "语义索引只读取本地模型；未安装可选依赖或模型未就绪时自动使用稀疏/FTS 召回。"
                )
                hint.setObjectName("friendlyHint")
                hint.setWordWrap(True)
                layout.addWidget(hint)
            if key == "rendering":
                hint = QLabel(
                    "OpenGL 与 Vulkan 使用不同图形 API；选择并保存后，"
                    "需要在主控制台点击“重启并应用渲染引擎”。"
                )
                hint.setObjectName("friendlyHint")
                hint.setWordWrap(True)
                layout.addWidget(hint)
            if not paths:
                empty = QLabel("此页的详细字段由运行策略管理；普通使用无需填写原始参数。")
                empty.setObjectName("friendlyHint")
                empty.setWordWrap(True)
                layout.addWidget(empty)
            for path in paths:
                spec = specs.get(path)
                if spec is None:
                    continue
                if spec.kind == "bool":
                    toggle = QCheckBox(spec.label)
                    toggle.setObjectName("configFriendlyToggle_" + key + "_" + "_".join(path))
                    toggle.setProperty("configPath", path)
                    toggle.setToolTip(spec.tooltip or f"点击切换：{spec.label}")
                    toggle.toggled.connect(
                        lambda value, selected_key=key, selected_path=path: (
                            self._set_friendly_value(selected_key, selected_path, bool(value))
                        )
                    )
                    if key == "asr" and path == ("capture", "auto_submit"):
                        toggle.setEnabled(False)
                    controls[path] = toggle
                    layout.addWidget(toggle)
                    continue
                if (
                    (key in {"memory", "tts"} and spec.kind in {"int", "float"})
                    or (
                        key == "memory"
                        and path[:1] == ("semantic",)
                        and spec.kind in {"int", "float", "text"}
                    )
                    or (
                        key == "asr"
                        and path[:1] == ("capture",)
                        and spec.kind in {"int", "float", "text", "audio_input"}
                    )
                    or (key == "app" and spec.kind in {"text", "multiline", "lines"})
                    or (key == "tts" and spec.kind == "text")
                ):
                    if key == "app":
                        group = "prompts" if path[0] == "prompts" else "persona"
                        numeric_layout = app_text_cards.get(group)
                        if numeric_layout is None:
                            numeric_card = QGroupBox(
                                "模型提示词" if group == "prompts" else "人设信息"
                            )
                            numeric_card.setObjectName(
                                "configPromptFields"
                                if group == "prompts"
                                else "configPersonaFields"
                            )
                            numeric_layout = QFormLayout(numeric_card)
                            numeric_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
                            numeric_layout.setFieldGrowthPolicy(
                                QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
                            )
                            numeric_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
                            app_text_cards[group] = numeric_layout
                            layout.addWidget(numeric_card)
                        control = self._make_field_control(key, spec)
                        control.setProperty("configPath", path)
                        controls[path] = control
                        numeric_layout.addRow(spec.label, control)
                        continue
                    if numeric_card is None:
                        numeric_card = QGroupBox(
                            "记忆参数"
                            if key == "memory"
                            else "麦克风采集"
                            if key == "asr"
                            else "语音与启动参数"
                            if key == "tts"
                            else "启动检测"
                        )
                        numeric_card.setObjectName(
                            "configMemoryParameters"
                            if key == "memory"
                            else "configTtsStartupParameters"
                        )
                        numeric_layout = QFormLayout(numeric_card)
                        numeric_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
                        numeric_layout.setFieldGrowthPolicy(
                            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
                        )
                        numeric_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
                        layout.addWidget(numeric_card)
                    control = self._make_field_control(key, spec)
                    control.setProperty("configPath", path)
                    controls[path] = control
                    assert numeric_layout is not None
                    numeric_layout.addRow(spec.label, control)
                    continue
                if spec.kind != "enum":
                    continue
                choice_card = QGroupBox(spec.label)
                choice_card.setObjectName("configFriendlyChoiceCard")
                choice_layout = QGridLayout(choice_card)
                choice_layout.setContentsMargins(4, 4, 4, 4)
                choice_layout.setHorizontalSpacing(6)
                choice_layout.setVerticalSpacing(6)
                choice_buttons: list[QPushButton] = []
                for index, value in enumerate(spec.choices):
                    choice = QPushButton(_FRIENDLY_VALUE_LABELS.get(value, "选项"))
                    choice.setObjectName("configChoiceButton")
                    choice.setCheckable(True)
                    choice.setProperty("configPath", path)
                    choice.setProperty("configValue", value)
                    choice.setToolTip(f"点击选择{spec.label}")
                    if (
                        key == "rendering"
                        and path == ("backend",)
                        and value == "vulkan"
                        and not _vulkan_control_available()
                    ):
                        choice.setEnabled(False)
                        choice.setToolTip("当前 Qt Vulkan 绑定或渲染工厂不可用")

                    def select_choice(
                        _checked: bool = False,
                        selected_key: str = key,
                        selected_path: tuple[str, ...] = path,
                        selected_value: str = value,
                    ) -> None:
                        del _checked
                        self._set_friendly_value(selected_key, selected_path, selected_value)

                    choice.clicked.connect(select_choice)
                    choice_layout.addWidget(choice, index // 4, index % 4)
                    choice_buttons.append(choice)
                for column in range(4):
                    choice_layout.setColumnStretch(column, 1)
                buttons[(key, path)] = tuple(choice_buttons)
                layout.addWidget(choice_card)
            status = QLabel()
            status.setObjectName("friendlySummary")
            status.setWordWrap(True)
            self._friendly_status_labels[key] = status
            layout.addWidget(status)
            self._friendly_choice_buttons.update(buttons)
            return card

        def _set_tool_group(self, group: str, enabled: bool) -> None:
            """通过点击开关增删工具组，不要求用户编辑 active_groups。"""

            if self._building_form:
                return
            if not self._merge_editor_section_into_draft("tools"):
                return
            section = self._section_draft("tools")
            raw_groups = section.get("active_groups", _MISSING)
            if raw_groups is _MISSING or raw_groups is None:
                # 键缺失表示“全部公开工具”，第一次点击任何开关时先
                # 转成显式集合，再执行增删；否则关闭一个组会意外清空
                # 语义或重新回到全量公开。
                groups = [item[0] for item in _FRIENDLY_TOOL_GROUPS]
            else:
                groups = (
                    [str(item).strip() for item in raw_groups if str(item).strip()]
                    if isinstance(raw_groups, (list, tuple))
                    else []
                )
            if enabled and group not in groups:
                groups.append(group)
            if not enabled:
                groups = [item for item in groups if item != group]
                # 多步桌面自动化依赖桌面控制的总开关；关闭总开关时
                # 必须同时撤销自动化，避免权限界面显示已关闭但仍能执行
                # 键盘/鼠标/窗口步骤。
                if group == "desktop_control":
                    groups = [item for item in groups if item != "desktop_automation"]
            self._set_nested_value(section, ("active_groups",), groups)
            self._set_section_draft("tools", section)
            self._refresh_section_editor("tools")
            self._sync_forms()
            self._mark_user_edit()
            self._update_dirty()
            self._schedule_auto_save()
            if group == "system" and enabled:
                allowlist = section.get("command_allowlist")
                if not isinstance(allowlist, (list, tuple)) or not allowlist:
                    self._set_status("系统命令已启用；仍需开发者白名单和审批才能执行")
                    return
            label = next(
                (item[1] for item in _FRIENDLY_TOOL_GROUPS if item[0] == group),
                "工具组",
            )
            self._set_status(f"{label}：{'已开启' if enabled else '已关闭'}（尚未保存）")

        def _set_friendly_value(self, key: str, path: tuple[str, ...], value: object) -> None:
            """点击式控件只写入脱敏草稿，不让 Qt 控件暴露原始参数。"""

            if self._building_form:
                return
            if not self._merge_editor_section_into_draft(key):
                return
            section = self._section_draft(key)
            self._set_nested_value(section, path, value)
            self._set_section_draft(key, section)
            self._refresh_section_editor(key)
            self._sync_forms()
            self._mark_user_edit()
            self._update_dirty()
            self._schedule_auto_save()
            spec = self._spec_by_path(key).get(path)
            label = spec.label if spec is not None else "设置"
            if isinstance(value, bool):
                rendered = "已开启" if value else "已关闭"
            else:
                rendered = self._friendly_value(value)
            if key == "rendering" and path == ("backend",):
                self._set_status(f"{label}：{rendered}（尚未保存；保存后请回到主控制台重启应用）")
            else:
                self._set_status(f"{label}：{rendered}（尚未保存）")

        def _make_form_page(self, key: str) -> QWidget:
            """构造点击式分域页面，并保留隐藏的兼容编辑器。"""

            scroll = QScrollArea()
            scroll.setObjectName("configScroll")
            scroll.setWidgetResizable(True)
            scroll.viewport().setStyleSheet("background: transparent;")
            container = QWidget()
            container.setObjectName(f"configForm_{key}")
            container.setProperty("md3Role", "root")
            content = QVBoxLayout(container)
            content.setContentsMargins(12, 12, 12, 12)
            content.setSpacing(4)

            friendly = self._make_friendly_page(key)
            if key == "scheduler":
                # 调度器使用专用点击式编辑器；保留常用设置卡对象供
                # 兼容宿主同步状态，但避免在页面上出现两个“启用调度器”开关。
                friendly.setVisible(False)
            content.addWidget(friendly)

            if key == "scheduler":
                scheduler_panel = SchedulerPanel({}, parent=container)
                scheduler_panel.changed.connect(self._on_scheduler_changed)
                self._scheduler_panel = scheduler_panel
                content.addWidget(scheduler_panel)

            fields = _FIELD_SPECS.get(key, ())
            controls: dict[tuple[str, ...], QWidget] = {}
            self._form_controls[key] = controls
            if fields:
                field_card = QGroupBox("常用参数")
                field_card.setObjectName("configFieldCard")
                field_layout = QFormLayout(field_card)
                field_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
                field_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
                field_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
                for spec in fields:
                    control = self._make_field_control(key, spec)
                    controls[spec.path] = control
                    field_layout.addRow(spec.label, control)
                # 原始标量控件仅作为兼容层保留，默认不让普通用户面对
                # QComboBox/QSpinBox；显式打开开发者选项后才显示。
                field_card.setVisible(False)
                self._field_cards[key] = field_card
                content.addWidget(field_card)
            else:
                empty_text = (
                    "模型渠道请使用上方点击式向导配置；复杂字段默认隐藏。"
                    if key == "llm"
                    else "此配置域主要由列表或映射组成；需要时打开顶部“显示高级编辑”。"
                )
                empty = QLabel(empty_text)
                empty.setObjectName("configHint")
                empty.setWordWrap(True)
                content.addWidget(empty)

            complex_editors: dict[tuple[str, ...], QPlainTextEdit] = {}
            self._complex_editors[key] = complex_editors
            complex_cards: dict[tuple[str, ...], QGroupBox] = {}
            self._complex_cards[key] = complex_cards
            if key == "llm":
                model_hint = QLabel(
                    "模型渠道建议使用点击式向导填写；复杂连接设置可在开发者选项中查看。"
                )
                model_hint.setObjectName("configHint")
                model_hint.setWordWrap(True)
                content.addWidget(model_hint)
                model_button = QPushButton("打开点击式模型配置")
                model_button.setObjectName("openModelSetupButton")
                model_button.setText("配置模型与 API 密钥")
                model_button.setToolTip(
                    "通过服务预设、模型名称和 API 密钥环境变量完成配置；不会保存密钥值。"
                )
                model_button.clicked.connect(self.modelSetupRequested)
                self._model_setup_button = model_button
                content.addWidget(model_button)
            for path in _COMPLEX_PATHS.get(key, ()):
                card = QGroupBox(_PATH_LABELS.get(path[-1], path[-1]))
                card.setObjectName("configComplexCard")
                card_layout = QVBoxLayout(card)
                hint = QLabel("复杂列表/映射；敏感信息保持不变即可沿用原值。")
                hint.setObjectName("configHint")
                hint.setWordWrap(True)
                fragment = QPlainTextEdit()
                fragment.setObjectName("configComplex_" + key + "_" + "_".join(path))
                fragment.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
                fragment.setFont(QFont("Monospace"))
                fragment.setMinimumHeight(100)
                fragment.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                fragment.textChanged.connect(
                    lambda key=key, path=path, editor=fragment: self._on_complex_changed(
                        key, path, editor
                    )
                )
                card_layout.addWidget(hint)
                card_layout.addWidget(fragment)
                complex_editors[path] = fragment
                complex_cards[path] = card
                card.setVisible(False)
                content.addWidget(card)
            content.addStretch(1)
            scroll.setWidget(container)
            return scroll

        def _make_field_control(self, key: str, spec: _FieldSpec) -> QWidget:
            object_name = "configField_" + key + "_" + "_".join(spec.path)
            if spec.kind == "bool":
                control = QCheckBox()
                control.setObjectName(object_name)
                control.toggled.connect(
                    lambda value, key=key, path=spec.path: self._on_field_changed(
                        key, path, bool(value)
                    )
                )
            elif spec.kind == "audio_input":
                control = AudioInputSelector()
                control.setObjectName(object_name)
                control.set_devices(
                    self._audio_input_devices,
                    loaded=self._audio_input_devices_loaded,
                )
                control.refreshRequested.connect(self._load_audio_input_devices)
                control.currentIndexChanged.connect(
                    lambda _index, key=key, path=spec.path, selector=control: (
                        self._on_field_changed(key, path, selector.selected_device_id())
                    )
                )
            elif spec.kind == "enum":
                control = QComboBox()
                control.setObjectName(object_name)
                control.addItems(list(spec.choices))
                control.currentTextChanged.connect(
                    lambda value, key=key, path=spec.path: self._on_field_changed(
                        key, path, str(value)
                    )
                )
            elif spec.kind == "int":
                control = QSpinBox()
                control.setObjectName(object_name)
                control.setRange(int(spec.minimum), int(spec.maximum))
                control.setSingleStep(max(1, int(spec.step)))
                control.valueChanged.connect(
                    lambda value, key=key, path=spec.path: self._on_field_changed(
                        key, path, int(value)
                    )
                )
            elif spec.kind == "float":
                control = QDoubleSpinBox()
                control.setObjectName(object_name)
                control.setRange(float(spec.minimum), float(spec.maximum))
                control.setSingleStep(float(spec.step))
                step_text = f"{float(spec.step):.6f}".rstrip("0").rstrip(".")
                decimals = len(step_text.split(".", 1)[1]) if "." in step_text else 0
                control.setDecimals(max(0, min(3, decimals)))
                control.valueChanged.connect(
                    lambda value, key=key, path=spec.path: self._on_field_changed(
                        key, path, float(value)
                    )
                )
            elif spec.kind in {"multiline", "lines"}:
                control = QPlainTextEdit()
                control.setObjectName(object_name)
                control.setMinimumHeight(82)
                control.setMaximumHeight(150)
                control.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
                if spec.kind == "lines":
                    control.setPlaceholderText("每行一项")
                control.textChanged.connect(
                    lambda key=key, path=spec.path, editor=control, kind=spec.kind: (
                        self._on_field_changed(
                            key,
                            path,
                            (
                                [
                                    line.strip()
                                    for line in editor.toPlainText().splitlines()
                                    if line.strip()
                                ]
                                if kind == "lines"
                                else editor.toPlainText()
                            ),
                        )
                    )
                )
            else:
                control = QLineEdit()
                control.setObjectName(object_name)
                if spec.secret:
                    control.setEchoMode(QLineEdit.EchoMode.Password)
                    control.setPlaceholderText("***（留空表示保持原值）")
                control.textChanged.connect(
                    lambda value, key=key, path=spec.path: self._on_field_changed(
                        key, path, str(value)
                    )
                )
            if isinstance(control, (QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit)):
                # 主题样式的 padding 在部分 Linux Qt 样式中不会参与最小高度
                # 计算；显式保留触摸/鼠标可点击高度，避免下拉框看起来像空白线。
                control.setMinimumHeight(34)
            if spec.tooltip:
                control.setToolTip(spec.tooltip)
            return control

        def _load_audio_input_devices(self) -> None:
            """只在用户展开选择器后调用宿主设备枚举。"""

            loader = self._audio_input_device_loader
            if not callable(loader):
                self._set_status("输入设备列表暂不可用；仍可跟随系统默认设备。")
                return
            try:
                devices = loader()
                if not isinstance(devices, Sequence) or isinstance(
                    devices, (str, bytes, bytearray)
                ):
                    raise TypeError("audio input device loader returned an invalid value")
                self.set_audio_input_devices(devices, loaded=True)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                self._set_status("输入设备刷新失败；未更改当前选择。")

        def set_audio_input_devices(
            self,
            devices: Sequence[AudioInputDeviceChoice],
            *,
            loaded: bool = True,
        ) -> None:
            """更新本地设备描述；设备 ID 仅保留在选择器数据与配置草稿。"""

            if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes, bytearray)):
                raise TypeError("audio input devices must be a sequence")
            normalized = tuple(devices)
            if any(not isinstance(device, AudioInputDeviceChoice) for device in normalized):
                raise TypeError("audio input device entries are invalid")
            self._audio_input_devices = normalized
            self._audio_input_devices_loaded = bool(loaded)
            section = self._section_draft("asr")
            selected = self._nested_value(section, ("capture", "device_id"))
            selected_id = "" if selected is _MISSING else str(selected or "")
            seen: set[int] = set()
            for controls in (
                self._friendly_controls.get("asr", {}),
                self._form_controls.get("asr", {}),
            ):
                control = controls.get(("capture", "device_id"))
                if not isinstance(control, AudioInputSelector) or id(control) in seen:
                    continue
                seen.add(id(control))
                control.set_devices(
                    normalized,
                    loaded=self._audio_input_devices_loaded,
                    selected_device_id=selected_id,
                )

        @property
        def advanced_editing(self) -> bool:
            """返回是否显示开发者参数、复杂列表和完整 YAML。"""

            return self._advanced_editing

        @property
        def advanced_toggle(self) -> QCheckBox | None:
            """返回开发者选项开关，供宿主和无头测试读取。"""

            return self._advanced_toggle

        @property
        def model_setup_button(self) -> QPushButton | None:
            """返回模型点击式配置按钮。"""

            self._ensure_section_page("llm")
            return self._model_setup_button

        @property
        def scheduler_panel(self) -> SchedulerPanel | None:
            """返回调度器点击式编辑器。"""

            self._ensure_section_page("scheduler")
            return self._scheduler_panel

        def _apply_advanced_editing_visibility(self) -> None:
            """把当前高级编辑状态应用到已经创建的页面。"""

            if self._scheduler_panel is not None:
                self._scheduler_panel.setVisible(not self._advanced_editing)
            for field_card in self._field_cards.values():
                field_card.setVisible(self._advanced_editing)
            for key, friendly_page in self._friendly_pages.items():
                friendly_page.setVisible(not self._advanced_editing and key != "scheduler")
            for cards in self._complex_cards.values():
                for card in cards.values():
                    card.setVisible(self._advanced_editing)
            for tabs, index in self._yaml_tabs.values():
                if not self._advanced_editing and tabs.currentIndex() == index:
                    tabs.setCurrentIndex(0)
                tabs.setTabVisible(index, self._advanced_editing)
                tabs.tabBar().setVisible(self._advanced_editing)
            for key, apply_button in self._page_apply_buttons.items():
                apply_button.setVisible(self._advanced_editing and key != "overview")
            if self._advanced_editor is not None:
                self._advanced_editor.setVisible(self._advanced_editing)
            if self._overview_editor is not None:
                self._overview_editor.setVisible(self._advanced_editing)
            if self._advanced_hint is not None:
                self._advanced_hint.setVisible(not self._advanced_editing)

        def set_advanced_editing(self, enabled: bool, *, announce: bool = True) -> None:
            """显式切换完整 YAML/复杂列表编辑器的可见性。

            控件对象始终保留，兼容宿主通过 ``complex_editors`` 和
            ``yaml_editor`` 读取或写入草稿；默认只隐藏原始编辑表面，避免
            普通用户一打开配置中心就面对完整参数树。
            """

            self._advanced_editing = bool(enabled) if self._developer_mode else False
            self._apply_advanced_editing_visibility()
            if self._advanced_toggle is not None:
                self._advanced_toggle.setVisible(self._developer_mode)
                self._advanced_toggle.blockSignals(True)
                self._advanced_toggle.setChecked(self._advanced_editing)
                self._advanced_toggle.setText(
                    "隐藏开发者选项" if self._advanced_editing else "开发者选项：显示高级设置"
                )
                self._advanced_toggle.blockSignals(False)
            self.advancedEditingChanged.emit(self._advanced_editing)
            if announce:
                self._set_status(
                    "已显示开发者设置和完整配置。"
                    if self._advanced_editing
                    else "已隐藏开发者设置；普通设置和点击式模型配置仍可使用。"
                )

        @property
        def form_controls(self) -> dict[str, dict[tuple[str, ...], QWidget]]:
            """返回分域标量控件，供宿主集成和无头测试读取。"""

            self._ensure_all_section_pages()
            return self._form_controls

        @property
        def complex_editors(self) -> dict[str, dict[tuple[str, ...], QPlainTextEdit]]:
            """返回复杂列表编辑器，列表内容仍以 YAML 保持完整表达力。"""

            self._ensure_all_section_pages()
            return self._complex_editors

        @property
        def sections(self) -> tuple[str, ...]:
            """返回稳定的配置域导航键。"""

            return tuple(key for key, _label, _path, _hint in _SECTIONS)

        def _filter_sections(self, text: str) -> None:
            query = str(text or "").strip().casefold()
            first_visible = -1
            for index in range(self._navigation.count()):
                item = self._navigation.item(index)
                key = str(item.data(Qt.ItemDataRole.UserRole) or "")
                label = item.text()
                visible = not query or query in key.casefold() or query in label.casefold()
                item.setHidden(not visible)
                if visible and first_visible < 0:
                    first_visible = index
            current = self._navigation.currentRow()
            current_item = self._navigation.item(current) if current >= 0 else None
            if first_visible >= 0 and (current_item is None or current_item.isHidden()):
                self._navigation.setCurrentRow(first_visible)

        @staticmethod
        def _nested_value(value: object, path: tuple[str, ...]) -> object:
            current = value
            for key in path:
                if not isinstance(current, Mapping) or key not in current:
                    return _MISSING
                current = current[key]
            return current

        @staticmethod
        def _set_nested_value(root: dict[str, Any], path: tuple[str, ...], value: object) -> None:
            if not path:
                return
            current: dict[str, Any] = root
            for key in path[:-1]:
                nested = current.get(key)
                if not isinstance(nested, Mapping):
                    nested = {}
                nested_copy = dict(nested)
                current[key] = nested_copy
                current = nested_copy
            current[path[-1]] = copy.deepcopy(value)

        def _section_draft(self, key: str) -> dict[str, Any]:
            root_path = self._section_paths[key]
            if not root_path:
                return self._display_draft
            current = self._section_value(self._display_draft, root_path)
            return dict(current) if isinstance(current, Mapping) else {}

        def _set_section_draft(self, key: str, value: object) -> None:
            root_path = self._section_paths[key]
            if not root_path:
                if isinstance(value, Mapping):
                    self._display_draft = copy.deepcopy(dict(value))
                return
            self._display_draft[root_path[0]] = copy.deepcopy(value)

        @staticmethod
        def _control_value(control: QWidget, spec: _FieldSpec) -> object:
            if spec.kind == "bool" and isinstance(control, QCheckBox):
                return bool(control.isChecked())
            if spec.kind == "audio_input" and isinstance(control, AudioInputSelector):
                return control.selected_device_id()
            if spec.kind == "enum" and isinstance(control, QComboBox):
                return str(control.currentText())
            if spec.kind == "int" and isinstance(control, QSpinBox):
                return int(control.value())
            if spec.kind == "float" and isinstance(control, QDoubleSpinBox):
                return float(control.value())
            if spec.kind in {"multiline", "lines"} and isinstance(control, QPlainTextEdit):
                if spec.kind == "lines":
                    return [
                        line.strip() for line in control.toPlainText().splitlines() if line.strip()
                    ]
                return control.toPlainText()
            if isinstance(control, QLineEdit):
                return control.text()
            return ""

        @staticmethod
        def _display_bool(value: object, default: object) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().casefold()
                if normalized in {"true", "yes", "on", "1"}:
                    return True
                if normalized in {"false", "no", "off", "0"}:
                    return False
            return bool(default)

        def _sync_forms(self) -> None:
            self._building_form = True
            try:
                for key, controls in self._form_controls.items():
                    section = self._section_draft(key)
                    specs = {spec.path: spec for spec in _FIELD_SPECS.get(key, ())}
                    for path, control in controls.items():
                        spec = specs[path]
                        value = self._nested_value(section, path)
                        if value is _MISSING:
                            value = spec.default
                        if spec.kind == "bool" and isinstance(control, QCheckBox):
                            control.setChecked(self._display_bool(value, spec.default))
                        elif spec.kind == "audio_input" and isinstance(control, AudioInputSelector):
                            control.set_devices(
                                self._audio_input_devices,
                                loaded=self._audio_input_devices_loaded,
                                selected_device_id=str(value or ""),
                            )
                        elif spec.kind == "enum" and isinstance(control, QComboBox):
                            rendered = str(value if value is not None else "").strip()
                            # 配置文件中常见的 ``key: ""`` 不能把空白项
                            # 选进下拉框；使用字段默认值仍会在保存时写入一
                            # 个可校验的枚举值。非空未知值继续保留，方便
                            # 适配器扩展而不丢失用户字段。
                            if not rendered:
                                rendered = str(spec.default)
                            if control.findText(rendered) < 0:
                                control.addItem(rendered)
                            control.setCurrentText(rendered)
                        elif spec.kind == "int" and isinstance(control, QSpinBox):
                            try:
                                control.setValue(int(value))
                            except (TypeError, ValueError):
                                control.setValue(int(spec.default))
                        elif spec.kind == "float" and isinstance(control, QDoubleSpinBox):
                            try:
                                control.setValue(float(value))
                            except (TypeError, ValueError):
                                control.setValue(float(spec.default))
                        elif spec.kind in {"multiline", "lines"} and isinstance(
                            control, QPlainTextEdit
                        ):
                            if spec.kind == "lines":
                                rendered = (
                                    "\n".join(str(item) for item in value)
                                    if isinstance(value, (list, tuple))
                                    else ""
                                )
                            else:
                                rendered = "" if value is None else str(value)
                            control.setPlainText(rendered)
                        elif isinstance(control, QLineEdit):
                            control.setText("" if value is None else str(value))

                for key, editors in self._complex_editors.items():
                    section = self._section_draft(key)
                    for path, editor in editors.items():
                        value = self._nested_value(section, path)
                        if value is _MISSING:
                            value = (
                                []
                                if path[-1]
                                in {
                                    "channels",
                                    "servers",
                                    "tasks",
                                    "triggers",
                                    "actions",
                                    "bindings",
                                    "allow",
                                    "deny",
                                    "active_groups",
                                    "command_allowlist",
                                    "reference_audios",
                                }
                                else {}
                            )
                        editor.blockSignals(True)
                        editor.setPlainText(_yaml_dump(value))
                        editor.blockSignals(False)
                self._sync_friendly()
                if self._scheduler_panel is not None:
                    self._scheduler_panel.set_values(self._section_draft("scheduler"))
            finally:
                self._building_form = False

        def _on_scheduler_changed(self, values: object) -> None:
            """把专用调度器草稿合并回统一配置编辑会话。"""

            if self._building_form or not isinstance(values, Mapping):
                return
            self._set_section_draft("scheduler", values)
            self._refresh_section_editor("scheduler")
            self._sync_friendly()
            self._mark_user_edit()
            self._update_dirty()
            self._schedule_auto_save()

        def _sync_friendly(self) -> None:
            """同步点击式状态卡，避免用户看到旧状态或空选项。"""

            tools_section = self._section_draft("tools")
            raw_groups = tools_section.get("active_groups", _MISSING)
            active_groups = (
                {item[0] for item in _FRIENDLY_TOOL_GROUPS}
                if raw_groups is _MISSING or raw_groups is None
                else (
                    {str(item).strip() for item in raw_groups if str(item).strip()}
                    if isinstance(raw_groups, (list, tuple))
                    else set()
                )
            )
            for group, control in self._friendly_tool_group_controls.items():
                control.blockSignals(True)
                control.setChecked(group in active_groups)
                control.blockSignals(False)

            for key, controls in self._friendly_controls.items():
                section = self._section_draft(key)
                specs = self._spec_by_path(key)
                for path, control in controls.items():
                    spec = specs.get(path)
                    if spec is None:
                        continue
                    value = self._nested_value(section, path)
                    if value is _MISSING:
                        value = spec.default
                    if spec.kind == "bool" and isinstance(control, QCheckBox):
                        control.blockSignals(True)
                        control.setChecked(self._display_bool(value, spec.default))
                        control.blockSignals(False)
                    elif spec.kind == "audio_input" and isinstance(control, AudioInputSelector):
                        control.set_devices(
                            self._audio_input_devices,
                            loaded=self._audio_input_devices_loaded,
                            selected_device_id=str(value or ""),
                        )
                    elif spec.kind == "int" and isinstance(control, QSpinBox):
                        try:
                            control.blockSignals(True)
                            control.setValue(int(value))
                        except (TypeError, ValueError, OverflowError):
                            control.setValue(int(spec.default))
                        finally:
                            control.blockSignals(False)
                    elif spec.kind == "float" and isinstance(control, QDoubleSpinBox):
                        try:
                            control.blockSignals(True)
                            control.setValue(float(value))
                        except (TypeError, ValueError, OverflowError):
                            control.setValue(float(spec.default))
                        finally:
                            control.blockSignals(False)
                    elif spec.kind == "text" and isinstance(control, QLineEdit):
                        control.blockSignals(True)
                        control.setText("" if value is None else str(value))
                        control.blockSignals(False)
                    elif spec.kind in {"multiline", "lines"} and isinstance(
                        control, QPlainTextEdit
                    ):
                        rendered = (
                            "\n".join(str(item) for item in value)
                            if spec.kind == "lines" and isinstance(value, (list, tuple))
                            else ("" if value is None else str(value))
                        )
                        control.blockSignals(True)
                        control.setPlainText(rendered)
                        control.blockSignals(False)
                for path, choices in (
                    (path, buttons)
                    for (choice_key, path), buttons in self._friendly_choice_buttons.items()
                    if choice_key == key
                ):
                    current = self._nested_value(section, path)
                    if current is _MISSING or not str(current).strip():
                        current = specs.get(path, _FieldSpec(path, "")).default
                    for button in choices:
                        button.blockSignals(True)
                        button.setChecked(str(button.property("configValue")) == str(current))
                        button.blockSignals(False)
                status = self._friendly_status_labels.get(key)
                if status is None:
                    continue
                status.setText(self._friendly_section_summary(key, section))
            for key, status in self._overview_status_labels.items():
                status.setText(self._overview_section_summary(key))

        def _friendly_section_summary(self, key: str, section: Mapping[str, Any]) -> str:
            """生成不包含地址、密钥、路径和原始参数的页面摘要。"""

            if key == "app":
                persona = section.get("persona")
                persona = persona if isinstance(persona, Mapping) else {}
                name = str(persona.get("name", section.get("name", "MeaPet")) or "MeaPet")
                role = str(persona.get("role", "桌面伙伴") or "桌面伙伴")
                proactive = self._display_bool(persona.get("proactive_enabled"), True)
                prompts = section.get("prompts")
                prompts = prompts if isinstance(prompts, Mapping) else {}
                custom_prompt_count = sum(
                    1
                    for prompt_key in (
                        "dialogue",
                        "tool_guidance",
                        "memory_summary",
                        "memory_extract",
                    )
                    if str(prompts.get(prompt_key, "") or "").strip()
                )
                return (
                    f"{name[:40]} · {role[:60]} · 主动行为"
                    f"{'允许' if proactive else '关闭'} · 已配置 {custom_prompt_count} 类提示词"
                )
            if key == "llm":
                channels = section.get("channels")
                active = 0
                ready = 0
                if isinstance(channels, (list, tuple)):
                    for channel in channels:
                        if not isinstance(channel, Mapping):
                            continue
                        enabled = channel.get("enabled", channel.get("enable", True))
                        if isinstance(enabled, str):
                            enabled = enabled.strip().casefold() not in {
                                "false",
                                "no",
                                "off",
                                "0",
                            }
                        if not bool(enabled):
                            continue
                        active += 1
                        channel_id = str(
                            channel.get("id", channel.get("channel_id", "")) or ""
                        ).strip()
                        protocol = str(channel.get("protocol", "") or "").strip()
                        base_url = str(
                            channel.get("base_url", channel.get("url", "")) or ""
                        ).strip()
                        model = str(channel.get("model", "") or "").strip()
                        if not model:
                            models = channel.get("models")
                            if isinstance(models, (list, tuple)) and models:
                                model = str(models[0] or "").strip()
                        if channel_id and protocol and base_url and model:
                            try:
                                channel_from_mapping(
                                    {
                                        **dict(channel),
                                        "id": channel_id,
                                        "protocol": protocol,
                                        "base_url": base_url,
                                        "model": model,
                                    }
                                )
                            except (
                                AdapterConfigurationError,
                                AttributeError,
                                TypeError,
                                ValueError,
                            ):
                                continue
                            ready += 1
                if ready:
                    if ready == active:
                        return f"已配置 {ready} 个可用模型渠道"
                    return f"已配置 {ready} 个可用模型渠道，另有 {active - ready} 个待补齐"
                if active:
                    return "模型渠道已添加但仍有字段待补齐，请点击“配置模型”"
                return "尚未配置可用模型渠道，请点击“配置模型”"
            if key == "tts":
                enabled = self._display_bool(section.get("enabled"), True)
                language = self._friendly_value(section.get("language", "zh"))
                backend = self._friendly_value(section.get("backend", "gpt_sovits_stdio"))
                profiles = section.get("profiles")
                profile_count = (
                    sum(
                        1
                        for profile_id, profile in profiles.items()
                        if str(profile_id or "").strip() and isinstance(profile, Mapping)
                    )
                    if isinstance(profiles, Mapping)
                    else 0
                )
                profile_label = f"已配置 {profile_count} 个音色" if profile_count else "默认音色"
                return (
                    f"语音{'已开启' if enabled else '已关闭'} · {language} · "
                    f"{backend} · {profile_label}"
                )
            if key == "asr":
                enabled = self._display_bool(section.get("enabled"), False)
                language = self._friendly_value(section.get("language", "zh"))
                backend = self._friendly_value(section.get("backend", "sensevoice"))
                return (
                    f"识别{'已开启' if enabled else '已关闭'} · {language} · "
                    f"{backend} · 本地 IPC 工作进程"
                )
            if key == "logging":
                level = self._friendly_value(section.get("level", "INFO"))
                color = self._friendly_value(
                    section.get("console", {}).get("color", "auto")
                    if isinstance(section.get("console"), Mapping)
                    else "auto"
                )
                return f"等级：{level} · 颜色：{color}"
            if key == "memory":
                enabled = self._display_bool(section.get("enabled"), True)
                recall_limit = section.get("recall_limit", 7)
                context_limit = section.get("context_max_chars", 6000)
                consolidation = self._display_bool(section.get("consolidation_enabled"), True)
                try:
                    recall_limit = max(1, int(recall_limit))
                except (TypeError, ValueError, OverflowError):
                    recall_limit = 7
                try:
                    context_limit = max(512, int(context_limit))
                except (TypeError, ValueError, OverflowError):
                    context_limit = 6000
                return (
                    f"长期记忆{'已开启' if enabled else '已关闭'} · "
                    f"每次召回 {recall_limit} 条 · 上下文 {context_limit} 字 · "
                    f"相似合并{'已开启' if consolidation else '已关闭'}"
                )
            if key == "rendering":
                backend = self._friendly_value(section.get("backend", "auto"))
                return f"已选择：{backend} · 保存后从主控制台重启以切换图形 API"
            if key == "tools":
                permissions = section.get("permissions")
                permissions = permissions if isinstance(permissions, Mapping) else {}
                approval = self._display_bool(permissions.get("bypass_approval"), False)
                raw_groups = section.get("active_groups", _MISSING)
                active_count = (
                    len(_FRIENDLY_TOOL_GROUPS)
                    if raw_groups is _MISSING or raw_groups is None
                    else (
                        len(tuple(item for item in raw_groups if str(item).strip()))
                        if isinstance(raw_groups, (list, tuple))
                        else 0
                    )
                )
                group_summary = (
                    "已启用全部公开工具组"
                    if raw_groups is _MISSING or raw_groups is None
                    else f"已启用 {active_count} 个工具组"
                )
                return (
                    f"{group_summary} · 审批已跳过"
                    if approval
                    else f"{group_summary} · 中风险操作需要确认"
                )
            if key == "behavior":
                movement = section.get("movement")
                movement = movement if isinstance(movement, Mapping) else {}
                enabled = self._display_bool(section.get("enabled"), False)
                roaming = self._display_bool(movement.get("enabled"), False)
                return (
                    f"自主行为{'已开启' if enabled else '已关闭'} · "
                    f"漫游{'已开启' if roaming else '已关闭'}"
                )
            if key == "ui":
                topmost = self._display_bool(section.get("always_on_top"), True)
                return f"窗口置顶{'已开启' if topmost else '已关闭'} · 输出使用点击式控制"
            if key == "mcp":
                return (
                    "MCP 服务已开启"
                    if self._display_bool(section.get("enabled"), False)
                    else "MCP 服务已关闭"
                )
            if key == "watcher":
                return (
                    "窗口观察已开启"
                    if self._display_bool(section.get("enabled"), False)
                    else "窗口观察已关闭"
                )
            if key == "scheduler":
                return (
                    "调度器已开启"
                    if self._display_bool(section.get("enabled"), True)
                    else "调度器已关闭"
                )
            return "常用设置可直接点按，详细参数由运行策略管理。"

        def _overview_section_summary(self, key: str) -> str:
            section = self._section_draft(key) if key in self._section_paths else {}
            return self._friendly_section_summary(key, section)

        def _refresh_section_editor(self, key: str) -> None:
            editor = self._section_editors[key]
            section_path = self._section_paths[key]
            value = self._display_draft if not section_path else self._section_draft(key)
            editor.blockSignals(True)
            editor.setPlainText(_yaml_dump(value))
            editor.blockSignals(False)

        def _merge_editor_section_into_draft(self, key: str) -> bool:
            editor = self._section_editors[key]
            if editor.toPlainText() == self._baseline.get(key, ""):
                return True
            try:
                parsed = parse_editor_yaml(editor.toPlainText())
            except Exception as exc:
                self._set_status(f"{key} 页面语法错误：{_safe_editor_error(exc)}")
                return False
            self._set_section_draft(key, parsed)
            return True

        def _on_field_changed(self, key: str, path: tuple[str, ...], value: object) -> None:
            if self._building_form:
                return
            if not self._merge_editor_section_into_draft(key):
                return
            section = self._section_draft(key)
            self._set_nested_value(section, path, value)
            self._set_section_draft(key, section)
            self._refresh_section_editor(key)
            self._mark_user_edit()
            self._update_dirty()
            self._schedule_auto_save()

        def _on_complex_changed(
            self, key: str, path: tuple[str, ...], editor: QPlainTextEdit
        ) -> None:
            if self._building_form:
                return
            if not self._merge_editor_section_into_draft(key):
                return
            try:
                import yaml

                value = yaml.safe_load(editor.toPlainText())
                validated = validate_editor_values({"fragment": value})
                value = validated["fragment"]
            except Exception as exc:
                self._set_status(f"{key}.{'.'.join(path)} YAML 语法错误：{_safe_editor_error(exc)}")
                self._mark_user_edit()
                self._update_dirty()
                self._schedule_auto_save()
                return
            section = self._section_draft(key)
            self._set_nested_value(section, path, value)
            self._set_section_draft(key, section)
            self._refresh_section_editor(key)
            self._mark_user_edit()
            self._update_dirty()
            self._schedule_auto_save()

        @property
        def navigation(self) -> QListWidget:
            return self._navigation

        @property
        def stack(self) -> FancyNavigationView:
            return self._stack

        @property
        def yaml_editor(self) -> QPlainTextEdit:
            self._ensure_section_page("advanced")
            return self._section_editors["advanced"]

        @property
        def save_button(self) -> QPushButton:
            return self._save_button

        @property
        def save_in_flight(self) -> bool:
            """返回是否正在等待宿主确认保存结果。"""

            return self._save_in_flight

        @property
        def status_label(self) -> QLabel:
            return self._status

        def set_values(
            self,
            values: Mapping[str, Any],
            *,
            path: str | Path | None = None,
        ) -> None:
            if not isinstance(values, Mapping):
                raise TypeError("configuration values must be a mapping")
            self.apply_theme_configuration(values)
            self._original = copy.deepcopy(dict(values))
            self._display_draft = editor_snapshot(self._original)
            self._save_in_flight = False
            self._save_edit_generation = None
            self._auto_save_timer.stop()
            self._auto_save_pending = False
            self._auto_save_blocked = False
            configured_auto_save = self._auto_save_enabled
            if self._auto_save_initialized:
                configured_auto_save = self._auto_save_enabled
            elif self._auto_save_override is None:
                ui_values = values.get("ui")
                if isinstance(ui_values, Mapping) and "auto_save" in ui_values:
                    raw_auto_save = ui_values.get("auto_save")
                    configured_auto_save = (
                        raw_auto_save
                        if isinstance(raw_auto_save, bool)
                        else str(raw_auto_save or "").strip().lower()
                        not in {"false", "no", "off", "0"}
                    )
            else:
                configured_auto_save = bool(self._auto_save_override)
            self._auto_save_initialized = True
            self._auto_save_enabled = configured_auto_save
            self._auto_save_toggle.blockSignals(True)
            self._auto_save_toggle.setChecked(configured_auto_save)
            self._auto_save_toggle.blockSignals(False)
            if path is not None:
                self._path = Path(path).expanduser()
            self._sync_editors(reset_baseline=True)
            if self._path is None:
                self._path_label.setText("配置文件：由宿主管理")
                self._path_label.setToolTip("")
            else:
                # 路径属于宿主实现细节，普通配置页只显示连接状态，避免把
                # 本机目录结构带入截图、共享屏幕或客服记录。
                self._path_label.setText("配置文件：已连接")
                self._path_label.setToolTip("配置文件已连接；具体位置由宿主管理。")
            self._summary.setText(f"{len(_SECTIONS)} 个配置域 · 已加载")
            self._set_status("配置已加载；未修改的敏感信息会原样保留。")

        def apply_theme_configuration(self, values: Mapping[str, Any]) -> None:
            """只热替换主题，不重置编辑草稿或自动保存状态。"""

            if not isinstance(values, Mapping):
                raise TypeError("configuration values must be a mapping")
            self._md3_theme, _stylesheet = _configuration_stylesheet(values)
            controller = self._fancy_style_controller
            controller.updateTheme(
                self._md3_theme,
                extra_stylesheet=_configuration_component_stylesheet(self._md3_theme),
            )
            scheduler_panel = getattr(self, "_scheduler_panel", None)
            apply_scheduler_theme = getattr(scheduler_panel, "apply_theme", None)
            if callable(apply_scheduler_theme):
                apply_scheduler_theme(self._md3_theme)

        def set_save_result(
            self,
            success: bool,
            message: str,
            values: Mapping[str, Any] | None = None,
        ) -> None:
            pending = self._auto_save_pending
            self._auto_save_pending = False
            self._save_in_flight = False
            saved_generation = self._save_edit_generation
            self._save_edit_generation = None
            if not success:
                # 失败后不自动高频重试；下一次用户编辑会解除抑制。
                self._auto_save_blocked = True
            edited_during_save = (
                saved_generation is not None and saved_generation != self._auto_save_edit_generation
            )
            if success and values is not None and not edited_during_save:
                self.set_values(values, path=self._path)
            elif success and edited_during_save:
                # 保存回执只确认了旧快照；保留用户在等待期间的新草稿，
                # 避免异步回执把最新编辑覆盖掉。
                self._sync_friendly()
            dirty = self._update_dirty()
            self._set_status(str(message or ("保存成功" if success else "保存失败")))
            if success and pending and dirty:
                self._schedule_auto_save()

        def _set_status(self, message: str) -> None:
            text = _friendly_config_message(message)
            if _SECRET_ERROR_PATTERN.search(text):
                text = _safe_editor_error(ValueError(text))
            self._status.setText(text)
            info_bar = getattr(self, "_status_info", None)
            if isinstance(info_bar, FancyInfoBar):
                severity = "info"
                if any(token in text for token in ("失败", "错误", "无效", "不能为空")):
                    severity = "error"
                elif any(token in text for token in ("等待", "正在", "未保存")):
                    severity = "warning"
                elif any(token in text for token in ("成功", "通过", "已加载", "已保存")):
                    severity = "success"
                info_bar.setSeverity(severity)
                info_bar.setMessage(text)
            self.statusChanged.emit(text)

        @staticmethod
        def _section_value(values: Mapping[str, Any], section_path: tuple[str, ...]) -> object:
            current: object = values
            for key in section_path:
                if not isinstance(current, Mapping):
                    return {}
                current = current.get(key, {})
            return current

        def _sync_editors(self, *, reset_baseline: bool = False) -> None:
            for key, editor in self._section_editors.items():
                section_path = self._section_paths[key]
                value = (
                    self._display_draft
                    if not section_path
                    else self._section_value(self._display_draft, section_path)
                )
                text = _yaml_dump(value)
                editor.blockSignals(True)
                editor.setPlainText(text)
                editor.blockSignals(False)
                if reset_baseline:
                    self._baseline[key] = text
            self._sync_forms()
            self._update_dirty()

        def _update_dirty(self) -> bool:
            editor_dirty = any(
                editor.toPlainText() != self._baseline.get(key, "")
                for key, editor in self._section_editors.items()
            )
            draft_dirty = self._display_draft != editor_snapshot(self._original)
            dirty = bool(editor_dirty or draft_dirty)
            self._save_button.setEnabled(dirty and not self._save_in_flight)
            self._summary.setText(
                f"{len(_SECTIONS)} 个配置域 · {'有未保存修改' if dirty else '已同步'}"
            )
            self.dirtyChanged.emit(dirty)
            if not dirty:
                self._auto_save_timer.stop()
                self._auto_save_pending = False
            elif self._save_in_flight and self._auto_save_enabled:
                self._auto_save_pending = True
            return dirty

        def _mark_user_edit(self) -> None:
            """记录一次用户编辑并解除上次自动保存失败的抑制。"""

            self._auto_save_blocked = False
            self._auto_save_edit_generation += 1

        def _schedule_auto_save(self) -> None:
            """在用户编辑后启动有界防抖定时器。"""

            if not self._auto_save_enabled:
                self._auto_save_timer.stop()
                return
            if self._save_in_flight:
                self._auto_save_pending = True
                self._auto_save_timer.stop()
                return
            if self._auto_save_blocked or not self._update_dirty():
                return
            self._auto_save_timer.start()

        def set_auto_save_enabled(self, enabled: bool) -> None:
            """切换配置自动保存；开启后对当前未保存草稿重新计时。"""

            self._auto_save_enabled = bool(enabled)
            if self._auto_save_enabled:
                self._auto_save_blocked = False
                self._schedule_auto_save()
            else:
                self._auto_save_timer.stop()
                self._auto_save_pending = False

        def _auto_save(self) -> None:
            """在编辑空闲后提交保存，不弹出阻塞式确认对话框。"""

            if not self._auto_save_enabled or self._auto_save_blocked:
                return
            if self._save_in_flight:
                self._auto_save_pending = True
                return
            if self._update_dirty():
                self.save(confirmed=True, automatic=True)

        def _on_editor_changed(self) -> None:
            if self._building_form:
                return
            self._mark_user_edit()
            self._update_dirty()
            self._schedule_auto_save()

        def apply_section(self, key: str) -> bool:
            normalized = str(key)
            self._ensure_section_page(normalized)
            editor = self._section_editors.get(normalized)
            if editor is None:
                self._set_status(f"未知配置页面：{key}")
                return False
            try:
                parsed = parse_editor_yaml(editor.toPlainText())
            except Exception as exc:
                self._set_status(f"配置语法错误：{_safe_editor_error(exc)}")
                return False
            section_path = self._section_paths[normalized]
            if section_path:
                self._display_draft[section_path[0]] = parsed
            else:
                self._display_draft = parsed
            self._sync_editors(reset_baseline=False)
            self._mark_user_edit()
            self._set_status(f"已应用“{key}”页面修改到草稿，尚未保存。")
            self._update_dirty()
            self._schedule_auto_save()
            return True

        def _collect_draft(self) -> dict[str, Any] | None:
            """收集所有被编辑的页面，保持未知字段和密钥占位符。"""

            result = copy.deepcopy(self._display_draft)
            # 高级页面是根映射，只有它实际变化时才覆盖当前草稿。
            advanced = self._section_editors.get("advanced")
            if advanced is not None and advanced.toPlainText() != self._baseline.get(
                "advanced", ""
            ):
                try:
                    result = parse_editor_yaml(advanced.toPlainText())
                except Exception as exc:
                    self._set_status(f"高级 YAML 语法错误：{_safe_editor_error(exc)}")
                    return None
            for key, editor in self._section_editors.items():
                if key in {"advanced", "overview"}:
                    continue
                if editor.toPlainText() == self._baseline.get(key, ""):
                    continue
                try:
                    parsed = parse_editor_yaml(editor.toPlainText())
                except Exception as exc:
                    self._set_status(f"{key} 页面语法错误：{_safe_editor_error(exc)}")
                    return None
                section_path = self._section_paths[key]
                if section_path:
                    result[section_path[0]] = parsed
            # 表单把显式空枚举显示为字段默认值；在真正提交前也把该值
            # 写入草稿，避免用户看到可用选项却保存了原来的空串。
            for key, specs in _FIELD_SPECS.items():
                section_path = self._section_paths.get(key, ())
                section = self._section_value(result, section_path) if section_path else result
                if not isinstance(section, Mapping):
                    continue
                section_copy = copy.deepcopy(dict(section))
                changed = False
                for spec in specs:
                    if spec.kind != "enum":
                        continue
                    value = self._nested_value(section_copy, spec.path)
                    if value is _MISSING or (value is not None and str(value).strip()):
                        continue
                    self._set_nested_value(section_copy, spec.path, spec.default)
                    changed = True
                if changed:
                    if section_path:
                        result[section_path[0]] = section_copy
                    else:
                        result = section_copy
            return result

        def draft_values(self) -> dict[str, Any] | None:
            return self._collect_draft()

        def _validate_draft(self) -> None:
            values = self._collect_draft()
            if values is None:
                return
            self._set_status("正在提交配置校验…")
            self.validateRequested.emit(copy.deepcopy(values))

        def _confirm_save(self) -> None:
            self.save(confirmed=False)

        def save(self, *, confirmed: bool = False, automatic: bool = False) -> bool:
            self._auto_save_timer.stop()
            if not automatic:
                self._auto_save_pending = False
            if self._save_in_flight:
                self._set_status("正在等待上一轮保存结果，请稍候。")
                return False
            values = self._collect_draft()
            if values is None:
                return False
            if not confirmed:
                if not _confirm_save_dialog(self):
                    self._set_status("已取消保存。")
                    return False
            # 先写入等待状态再发信号。Qt 的默认连接是同步调用，宿主可能
            # 在 ``emit`` 返回前已经设置了成功/失败结果；若在 emit 后再写
            # “等待”文本，会把真实结果覆盖掉，用户无法判断文件是否保存。
            self._save_in_flight = True
            self._save_edit_generation = self._auto_save_edit_generation
            self._save_button.setEnabled(False)
            self._set_status("正在提交保存请求，等待宿主确认结果…")
            self.saveRequested.emit(copy.deepcopy(values))
            return True

        def reset_draft(self) -> None:
            self._auto_save_timer.stop()
            self._auto_save_pending = False
            self._auto_save_blocked = False
            self._display_draft = editor_snapshot(self._original)
            self._sync_editors(reset_baseline=True)
            self._set_status("已撤销未保存修改。")

else:

    class AudioInputSelector:  # pragma: no cover
        """无 Qt 环境下的明确不可用占位。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 is not installed")

    class ConfigurationPanel:  # pragma: no cover
        """无 Qt 环境下的明确不可用占位。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 is not installed")


__all__ = ["AudioInputSelector", "ConfigurationPanel", "pyside6_available"]
