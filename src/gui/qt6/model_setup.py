from __future__ import annotations

import inspect
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlsplit

import yaml

from config.loader import parse_bool
from core.adapters.direct.errors import AdapterConfigurationError
from gui.qt6.fancyui import (
    FancyCard,
    FancyCommandBar,
    FancyInfoBar,
    FancyStyleController,
    fancy_icon,
)
from gui.qt6.md3 import (
    DARK_MD3_THEME,
    MD3Theme,
    build_md3_stylesheet,
    build_qt_palette,
    remap_legacy_stylesheet,
)
from services.model_routing.channels import SUPPORTED_PROTOCOLS, channel_from_mapping

try:  # Qt 依赖保持可选，核心服务不需要 GUI。
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import (
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    pyside6_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_available = False


_ENV_NAME_PATTERN = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
_ENV_REFERENCE_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}\Z")
_REDACTION_MARKER = "***"
_URL_VERSION_SEGMENT_PATTERN = re.compile(r"(?:^|/)v\d+[A-Za-z0-9._-]*(?:/|$)", re.IGNORECASE)
_DEFAULT_CAPABILITIES = ("streaming", "tools")
_DEFAULT_PROTOCOL = "openai_chat"
_DEFAULT_CHANNEL_ID = "primary"
_DEFAULT_PRIORITY = 10
_SECRET_RESULT_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?key|private[_-]?key|credential|"
    r"authorization|bearer|token|secret|password|cookie)\b"
)
_INTERNAL_RESULT_PATTERN = re.compile(
    r"(?i)\b(?:status|state|detail|reason|operation[_-]?id|call[_-]?id)\s*="
)


def _friendly_model_error(error: object) -> str:
    """将内部字段校验错误转换为模型向导可读提示。"""

    text = str(error or "").strip()
    replacements = (
        ("channel_id", "服务名称"),
        ("base_url", "服务地址"),
        ("api_key_env", "密钥设置"),
        ("api_key", "密钥设置"),
        ("capabilities", "能力设置"),
        ("protocol", "服务类型"),
        ("priority", "优先级"),
        ("enabled", "启用状态"),
        ("model", "模型名称"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    if "environment variable" in text.casefold():
        return "密钥设置必须使用大写字母、数字或下划线。"
    if "credentials" in text.casefold():
        return "服务地址不能包含账号或密码。"
    if len(text) > 240:
        text = text[:239] + "…"
    return text or "连接信息不完整"


def _safe_model_result_text(value: object, fallback: str) -> str:
    """过滤宿主回传的测试文案，避免内部协议或密钥片段进入向导。"""

    text = " ".join(str(value or "").replace("\x00", " ").split())
    if not text or _SECRET_RESULT_PATTERN.search(text) or _INTERNAL_RESULT_PATTERN.search(text):
        return fallback
    return text[:240]


@dataclass(frozen=True)
class ModelSetupPreset:
    """一个可在设置卡片中选择的渠道预设。

    预设只提供可公开的地址、协议和模型示例；API Key 始终通过
    ``api_key_env`` 表示环境变量名称，不在对象中保存密钥内容。
    """

    key: str
    label: str
    protocol: str = _DEFAULT_PROTOCOL
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    capabilities: tuple[str, ...] = _DEFAULT_CAPABILITIES


# 这些值均来自仓库 README.md 的模型渠道示例；本地 Ollama 地址来自启动配置测试。
MODEL_SETUP_PRESETS: dict[str, ModelSetupPreset] = {
    "custom": ModelSetupPreset("custom", "自定义", _DEFAULT_PROTOCOL),
    "openai": ModelSetupPreset(
        "openai",
        "OpenAI",
        "openai_chat",
        "https://api.openai.com/v1",
        "gpt-4o-mini",
        "OPENAI_API_KEY",
        ("streaming", "tools", "vision"),
    ),
    "anthropic": ModelSetupPreset(
        "anthropic",
        "Anthropic Claude",
        "anthropic_messages",
        "https://api.anthropic.com",
        "claude-3-5-sonnet-latest",
        "ANTHROPIC_API_KEY",
        ("streaming", "tools", "vision", "reasoning"),
    ),
    "gemini": ModelSetupPreset(
        "gemini",
        "Google Gemini",
        "gemini_generate",
        "https://generativelanguage.googleapis.com",
        "gemini-2.0-flash",
        "GEMINI_API_KEY",
        ("streaming", "tools", "vision"),
    ),
    "ollama": ModelSetupPreset(
        "ollama",
        "本地 Ollama",
        "ollama_chat",
        "http://127.0.0.1:11434/v1",
        "",
        "",
        _DEFAULT_CAPABILITIES,
    ),
}

# 常用别名，方便宿主按自己的命名习惯读取预设。
MODEL_PRESETS = MODEL_SETUP_PRESETS
CHANNEL_PRESETS = MODEL_SETUP_PRESETS

# 适配器与协议是两级选择：先确定适配器族，再只显示该适配器支持的
# 协议。自定义适配器保留全部运行时协议，便于代理网关或兼容实现接入。
ADAPTER_PROTOCOLS: dict[str, tuple[str, ...]] = {
    "custom": tuple(sorted(SUPPORTED_PROTOCOLS)),
    "openai": ("openai_chat", "openai_responses", "openai"),
    "anthropic": ("anthropic_messages", "anthropic", "claude"),
    "gemini": ("gemini_generate", "gemini", "google_gemini"),
    "ollama": ("ollama_chat",),
}
PROTOCOL_LABELS: dict[str, str] = {
    "openai_chat": "OpenAI · Chat Completions",
    "openai_responses": "OpenAI · Responses",
    "openai": "OpenAI · 兼容协议",
    "anthropic_messages": "Claude · Messages",
    "anthropic": "Claude · 兼容协议",
    "claude": "Claude · 兼容别名",
    "gemini_generate": "Gemini · Generate Content",
    "gemini": "Gemini · 兼容协议",
    "google_gemini": "Gemini · 兼容别名",
    "ollama_chat": "Ollama · Chat",
}
# 已有环境变量优先用于渠道向导；具体名称来自运行时环境入口，
# 不读取或展示变量值。没有已设置变量时仍使用预设的规范名称。
API_KEY_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "custom": ("MEAPET_API_KEY",),
    "openai": ("OPENAI_API_KEY", "MEAPET_API_KEY"),
    "anthropic": ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "MEAPET_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "MEAPET_API_KEY"),
    "ollama": (),
}


def protocols_for_adapter(adapter: object) -> tuple[str, ...]:
    """返回适配器可用协议，未知或自定义适配器使用完整协议集合。"""

    key = str(adapter or "custom").strip().lower() or "custom"
    return ADAPTER_PROTOCOLS.get(key, ADAPTER_PROTOCOLS["custom"])


def adapter_for_protocol(protocol: object) -> str:
    """根据精确协议标识返回对应适配器族。"""

    key = str(protocol or "").strip().lower()
    for adapter, protocols in ADAPTER_PROTOCOLS.items():
        if adapter != "custom" and key in protocols:
            return adapter
    return "custom"


def preferred_api_key_env(adapter: object, fallback: object = "") -> str:
    """返回当前进程中已经配置的密钥变量名，未发现时使用规范回退名。"""

    key = str(adapter or "custom").strip().lower() or "custom"
    fallback_name = str(fallback or "").strip()
    aliases = API_KEY_ENV_ALIASES.get(key, API_KEY_ENV_ALIASES["custom"])
    ordered = (fallback_name, *aliases)
    seen: set[str] = set()
    for name in ordered:
        if not name or name in seen:
            continue
        seen.add(name)
        if os.environ.get(name):
            return name
    return fallback_name or (aliases[0] if aliases else "")


def _text(value: object, field_name: str, *, maximum: int) -> str:
    """规范化单行文本，拒绝控制字符。"""

    result = str(value or "").strip()
    if len(result) > maximum or any(char in result for char in "\r\n\x00"):
        raise ValueError(f"{field_name} 字段无效")
    return result


def _environment_name(value: object) -> str:
    """校验 API Key 环境变量名称，不读取环境变量值。"""

    result = _text(value, "api_key_env", maximum=256)
    if result and _ENV_NAME_PATTERN.fullmatch(result) is None:
        raise ValueError("api_key_env 必须是大写环境变量名")
    return result


def _safe_base_url(value: str) -> str:
    """在生成预览前拒绝包含凭据或查询参数的地址。"""

    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url 必须使用 http 或 https")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("base_url 不得包含 credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url 不得包含 query 或 fragment")
    return value.rstrip("/")


def openai_base_url_hint(protocol: object, base_url: object) -> str:
    """提示 OpenAI 兼容网关确认版本路径，不改写用户地址。"""

    protocol_value = str(protocol or "").strip().lower()
    if protocol_value not in {"openai", "openai_chat", "openai_responses"}:
        return ""
    raw_url = str(base_url or "").strip()
    if not raw_url:
        return ""
    try:
        path = urlsplit(raw_url).path.rstrip("/")
    except ValueError:
        return ""
    if _URL_VERSION_SEGMENT_PATTERN.search(path):
        return ""
    return "兼容网关通常需要 /v1，请确认服务地址是否包含版本路径。"


def _capabilities(value: object) -> tuple[str, ...]:
    """把逗号分隔或列表形式的能力集合规范为稳定元组。"""

    if value is None or value == "":
        items: Sequence[object] = _DEFAULT_CAPABILITIES
    elif isinstance(value, str):
        items = tuple(value.split(","))
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise ValueError("capabilities 必须是逗号分隔字符串或列表")
    result: list[str] = []
    for item in items:
        capability = _text(item, "capabilities", maximum=128).lower()
        if capability and capability not in result:
            result.append(capability)
    if "streaming" not in result:
        result.insert(0, "streaming")
    return tuple(result)


def _bool(value: object) -> bool:
    """解析设置卡片中的严格布尔值。"""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "yes", "on", "1"}:
        return True
    if normalized in {"false", "no", "off", "0", ""}:
        return False
    raise ValueError("enabled 必须是布尔值")


def _preset_key(value: object) -> str:
    """返回预设键；空值按自定义处理。"""

    key = _text(value, "preset", maximum=64).lower() or "custom"
    if key not in MODEL_SETUP_PRESETS:
        raise ValueError("未知的模型渠道预设")
    return key


@dataclass(frozen=True)
class ModelSetupDraft:
    """模型渠道设置的纯数据契约。

    ``api_key_env`` 只保存环境变量名称。该对象没有明文 API Key 字段，
    ``repr``、YAML 草稿和回调 payload 都不会包含密钥内容。
    """

    channel_id: str = _DEFAULT_CHANNEL_ID
    protocol: str = _DEFAULT_PROTOCOL
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    capabilities: tuple[str, ...] = _DEFAULT_CAPABILITIES
    enabled: bool = True
    priority: int = _DEFAULT_PRIORITY
    preset: str = "custom"
    # 仅供向导保留已配置凭据的状态，不进入 repr、比较或公开载荷。
    has_persisted_secret: bool = field(default=False, repr=False, compare=False)
    # 只有用户明确选择清除时才生成空 api_key 字段。
    clear_persisted_secret: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        channel_id = _text(self.channel_id, "channel_id", maximum=512)
        protocol = _text(self.protocol, "protocol", maximum=128).lower() or _DEFAULT_PROTOCOL
        if protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported channel protocol: {protocol}")
        base_url = _text(self.base_url, "base_url", maximum=2048)
        model = _text(self.model, "model", maximum=512)
        env_name = _environment_name(self.api_key_env)
        capabilities = _capabilities(self.capabilities)
        try:
            enabled = _bool(self.enabled)
        except ValueError:
            raise
        try:
            priority = int(self.priority)
        except (TypeError, ValueError) as exc:
            raise ValueError("priority 必须是整数") from exc
        if priority < -1_000_000 or priority > 1_000_000:
            raise ValueError("priority 超出范围")
        preset = _preset_key(self.preset)
        if preset != "custom" and protocol not in protocols_for_adapter(preset):
            # 适配器和协议必须来自同一层级，避免程序化调用绕过级联控件后
            # 把 OpenAI 预设与 Ollama 协议拼成无法复现的渠道。
            raise ValueError("适配器与服务类型不匹配")
        object.__setattr__(self, "channel_id", channel_id)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "api_key_env", env_name)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "enabled", enabled)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "preset", preset)
        object.__setattr__(self, "has_persisted_secret", bool(self.has_persisted_secret))
        object.__setattr__(self, "clear_persisted_secret", bool(self.clear_persisted_secret))

    @property
    def id(self) -> str:
        """兼容 ``ChannelConfig.id`` 的只读别名。"""

        return self.channel_id

    @property
    def adapter(self) -> str:
        """返回向导层的适配器族标识；运行时仍以 protocol 为准。"""

        return self.preset

    @property
    def api_key_reference(self) -> str:
        """返回 YAML 中使用的环境变量引用。"""

        return f"${{{self.api_key_env}}}" if self.api_key_env else ""

    @classmethod
    def from_preset(cls, preset: str, **overrides: object) -> ModelSetupDraft:
        """从预设创建草稿，并用显式覆盖值填充用户字段。"""

        key = _preset_key(preset)
        option = MODEL_SETUP_PRESETS[key]
        values: dict[str, object] = {
            "channel_id": option.key if key != "custom" else _DEFAULT_CHANNEL_ID,
            "protocol": option.protocol,
            "base_url": option.base_url,
            "model": option.model,
            "api_key_env": option.api_key_env,
            "capabilities": option.capabilities,
            "preset": key,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ModelSetupDraft:
        """从渠道映射、向导回调 payload 或 ``llm`` 片段读取草稿。

        ``api_key`` 只允许完整的 ``${ENV_NAME}`` 引用；任何明文值都会被拒绝，
        且错误文本不会回显该值。
        """

        if not isinstance(value, Mapping):
            raise ValueError("模型渠道草稿必须是映射")
        source: Mapping[str, Any] = value
        llm = value.get("llm")
        if isinstance(llm, Mapping):
            source = llm
        channel = source.get("channel")
        if isinstance(channel, Mapping):
            source = channel
        channels = source.get("channels")
        if isinstance(channels, Sequence) and not isinstance(channels, (str, bytes, bytearray)):
            if not channels or not isinstance(channels[0], Mapping):
                raise ValueError("模型渠道列表为空")
            source = channels[0]

        secret_present = source.get("_secret_present") is True
        clear_secret = source.get("_clear_secret") is True
        raw_api_key = source.get("api_key")
        env_name = source.get("api_key_env", "")
        if (
            "api_key" in source
            and raw_api_key in (None, "")
            and not secret_present
            and not str(env_name or "").strip()
        ):
            clear_secret = True
        if raw_api_key not in (None, ""):
            reference = _text(raw_api_key, "api_key", maximum=4096)
            if reference == _REDACTION_MARKER:
                raw_api_key = ""
                secret_present = True
                clear_secret = False
            else:
                match = _ENV_REFERENCE_PATTERN.fullmatch(reference)
                if match is None:
                    raise ValueError("api_key 只允许使用环境变量引用")
                if env_name and _environment_name(env_name) != match.group(1):
                    raise ValueError("api_key 与 api_key_env 不一致")
                env_name = match.group(1)
                secret_present = True
        if str(env_name or "").strip():
            if clear_secret:
                env_name = ""
            else:
                secret_present = True
        raw_protocol = source.get("protocol", _DEFAULT_PROTOCOL)
        raw_preset = source.get("preset")
        raw_adapter = source.get("adapter")
        if (
            raw_preset in (None, "")
            and str(raw_adapter or "").strip().lower() in MODEL_SETUP_PRESETS
        ):
            # 外部调用方可以显式提供向导适配器；运行时 ChannelConfig 的
            # adapter 字段通常是协议别名，因此仅接受已知向导键。
            raw_preset = str(raw_adapter).strip().lower()
        if raw_preset in (None, ""):
            # 旧配置没有 preset 字段时按已声明协议恢复适配器，
            # 让向导再次打开后仍遵循“适配器 → 协议”的层级。
            raw_preset = adapter_for_protocol(raw_protocol)
        return cls(
            channel_id=source.get("channel_id", source.get("id", _DEFAULT_CHANNEL_ID)),
            protocol=raw_protocol,
            base_url=source.get("base_url", ""),
            model=source.get("model", ""),
            api_key_env=env_name,
            capabilities=source.get("capabilities", _DEFAULT_CAPABILITIES),
            enabled=source.get("enabled", True),
            priority=source.get("priority", _DEFAULT_PRIORITY),
            preset=raw_preset,
            has_persisted_secret=secret_present,
            clear_persisted_secret=clear_secret,
        )

    @classmethod
    def from_yaml(cls, text: str) -> ModelSetupDraft:
        """从 YAML 草稿读取草稿，不展开环境变量。"""

        try:
            value = yaml.safe_load(str(text))
        except yaml.YAMLError as exc:
            raise ValueError("模型渠道 YAML 无法解析") from exc
        if not isinstance(value, Mapping):
            raise ValueError("模型渠道 YAML 根节点必须是映射")
        return cls.from_mapping(value)

    def with_preset(self, preset: str) -> ModelSetupDraft:
        """切换预设并返回新草稿；调用方可随后覆盖模型或渠道 ID。"""

        return self.from_preset(
            preset,
            channel_id=self.channel_id or _DEFAULT_CHANNEL_ID,
            enabled=self.enabled,
            priority=self.priority,
        )

    def validate(self) -> ModelSetupDraft:
        """按运行时渠道解析器校验必填字段和协议边界。"""

        if not self.channel_id or not self.base_url or not self.model:
            raise ValueError("channel_id、base_url 和 model 不能为空")
        mapping = self._channel_mapping()
        try:
            channel_from_mapping(mapping)
        except AdapterConfigurationError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def _channel_mapping(self) -> dict[str, Any]:
        """生成内部渠道映射；该方法不读取环境变量。"""

        mapping: dict[str, Any] = {
            "id": self.channel_id,
            # preset 是向导层的适配器族元数据；运行时仍只按 protocol
            # 选择网络适配器，未知字段由配置合并逻辑原样保留。
            "preset": self.preset,
            "protocol": self.protocol,
            "base_url": _safe_base_url(self.base_url),
            "model": self.model,
            "capabilities": list(self.capabilities),
            "enabled": self.enabled,
            "priority": self.priority,
        }
        # 没有环境变量引用时省略密钥字段，而不是发送空字符串。这样重新
        # 编辑已经显式持久化的本机密钥时，普通自动保存会保留原值；需要
        # 清除密钥的调用方仍可在完整配置编辑器中显式提交空值。
        if self.clear_persisted_secret or self.protocol == "ollama_chat":
            mapping["api_key"] = ""
        elif self.api_key_reference:
            mapping["api_key"] = self.api_key_reference
        return mapping

    def to_channel_mapping(self, *, validate: bool = True) -> dict[str, Any]:
        """返回不含明文密钥的 ``llm.channels`` 项。"""

        if validate:
            self.validate()
        return dict(self._channel_mapping())

    # 映射命名别名，便于与 ``ChannelConfig.as_mapping`` 对齐。
    as_mapping = to_channel_mapping
    to_mapping = to_channel_mapping

    def to_callback_payload(self) -> dict[str, Any]:
        """返回可直接交给 ``ConfigurationWizard.configure_channel`` 的参数。"""

        self.validate()
        return {
            "channel_id": self.channel_id,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "capabilities": list(self.capabilities),
            "enabled": self.enabled,
            "priority": self.priority,
        }

    def to_test_payload(self) -> dict[str, Any]:
        """返回连接探测契约；只包含环境变量名称，不读取密钥值。

        宿主在调用 ``ModelRouter.test_connection`` 前必须把该契约合并到
        已加载的配置并完成环境变量展开。该纯数据契约本身不会携带明文 API Key。
        """

        self.validate()
        payload = {
            "channel_id": self.channel_id,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "model": self.model,
            "capabilities": list(self.capabilities),
            "enabled": self.enabled,
            "priority": self.priority,
        }
        if self.api_key_env:
            payload["api_key_env"] = self.api_key_env
        return payload

    # 回调命名的两个别名，保持宿主代码可读且不复制数据。
    callback_payload = to_callback_payload
    as_callback_payload = to_callback_payload
    to_payload = to_callback_payload

    def to_config_mapping(self, *, validate: bool = True) -> dict[str, Any]:
        """返回可合并到配置文件的最小 ``llm`` YAML 映射。"""

        channel = self.to_channel_mapping(validate=validate)
        return {
            "llm": {
                "channels": [channel],
                "routing": {
                    "dialogue": {
                        "channel": self.channel_id,
                        "required_capabilities": ["streaming"],
                        "fallback_channels": [],
                    }
                },
            }
        }

    def to_yaml(self, *, validate: bool = False) -> str:
        """生成脱敏 YAML 草稿；默认允许 UI 展示未填完的中间状态。"""

        return yaml.safe_dump(
            self.to_config_mapping(validate=validate), allow_unicode=True, sort_keys=False
        )

    yaml_draft = to_yaml
    redacted_yaml = to_yaml
    yaml_text = to_yaml


def build_model_setup_payload(draft: ModelSetupDraft | Mapping[str, Any]) -> dict[str, Any]:
    """把草稿或映射转换为安全回调 payload。"""

    current = draft if isinstance(draft, ModelSetupDraft) else ModelSetupDraft.from_mapping(draft)
    return current.to_callback_payload()


def build_model_test_payload(draft: ModelSetupDraft | Mapping[str, Any]) -> dict[str, Any]:
    """构造点击式连接测试的安全 payload。"""

    current = draft if isinstance(draft, ModelSetupDraft) else ModelSetupDraft.from_mapping(draft)
    return current.to_test_payload()


def build_model_setup_yaml(
    draft: ModelSetupDraft | Mapping[str, Any], *, validate: bool = False
) -> str:
    """把草稿或映射转换为脱敏 YAML。"""

    current = draft if isinstance(draft, ModelSetupDraft) else ModelSetupDraft.from_mapping(draft)
    return current.to_yaml(validate=validate)


def parse_model_setup_yaml(text: str) -> ModelSetupDraft:
    """解析脱敏模型渠道 YAML。"""

    return ModelSetupDraft.from_yaml(text)


def model_setup_drafts(value: object) -> tuple[ModelSetupDraft, ...]:
    """从 ``llm.channels`` 读取可编辑的脱敏渠道草稿。

    配置中心在运行时可能拿到已展开的 API Key。这个函数只接受环境变量
    引用或红线标记；其他密钥值会被替换为红线标记后再交给草稿解析，确保
    渠道选择器永远不会把密钥带入 Qt 控件、日志或预览文本。
    """

    source = value if isinstance(value, Mapping) else {}
    llm = source.get("llm") if isinstance(source, Mapping) else None
    channels = llm.get("channels") if isinstance(llm, Mapping) else source.get("channels")
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes, bytearray)):
        return ()
    drafts: list[ModelSetupDraft] = []
    seen_ids: set[str] = set()
    for item in channels:
        if not isinstance(item, Mapping):
            continue
        sanitized = dict(item)
        raw_key = sanitized.get("api_key")
        if (
            isinstance(raw_key, str)
            and raw_key
            and _ENV_REFERENCE_PATTERN.fullmatch(raw_key) is None
        ):
            sanitized["api_key"] = _REDACTION_MARKER
        try:
            draft = ModelSetupDraft.from_mapping(sanitized)
        except ValueError:
            continue
        if draft.channel_id in seen_ids:
            continue
        seen_ids.add(draft.channel_id)
        drafts.append(draft)
    return tuple(drafts)


if pyside6_available:
    _MODEL_SETUP_STYLE = """
    QWidget#modelSetupCard, QDialog#modelSetupDialog {
        background: #16111f;
        color: #faf6fb;
    }
    QGroupBox#modelSetupGroup {
        background: #2e2440;
        border: 1px solid #46385c;
        border-top-color: #7c69a0;
        border-radius: 12px;
        margin-top: 12px;
        padding: 14px;
        color: #faf6fb;
        font-weight: 600;
    }
    QGroupBox#modelSetupGroup::title {
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
        color: #ffc48f;
    }
    QGroupBox#modelSetupGroup QLabel { color: #d6cbe0; }
    QLineEdit#modelSetupField, QLineEdit#modelSetupSecretField,
    QComboBox#modelSetupField,
    QSpinBox#modelSetupField,
    QPlainTextEdit#modelSetupPreview {
        background: #100c18;
        color: #faf6fb;
        border: 1px solid #7c69a0;
        border-radius: 8px;
        padding: 7px 9px;
    }
    QLineEdit#modelSetupField:focus, QComboBox#modelSetupField:focus,
    QPlainTextEdit#modelSetupPreview:focus { border: 1px solid #ff9dbe; }
    QLabel#modelSetupTitle { color: #faf6fb; font-size: 18px; font-weight: 700; }
    QLabel#modelSetupStep { color: #ffc48f; font-size: 13px; font-weight: 700; }
    QLabel#modelSetupHint, QLabel#modelSetupStatus { color: #d6cbe0; }
    QLabel#modelSetupFieldHint { color: #b7a6c7; font-size: 11px; }
    QLabel#modelSetupStatus[state="error"] { color: #ff8fa0; }
    QLabel#modelSetupStatus[state="ok"] { color: #6fe0b4; }
    QCheckBox { color: #faf6fb; spacing: 8px; }
    QCheckBox::indicator {
        width: 18px; height: 18px; background: #100c18;
        border: 1px solid #7c69a0; border-radius: 6px;
    }
    QCheckBox::indicator:checked { background: #ff9dbe; border-color: #ff9dbe; }
    QPushButton#modelPresetButton {
        background: #221a2e; color: #faf6fb; border: 1px solid #7c69a0;
        border-radius: 9px; padding: 8px 10px; min-height: 42px;
        font-weight: 600;
    }
    QPushButton#modelPresetButton:hover { background: #493757; border-color: #ffb6ce; }
    QPushButton#modelPresetButton:checked {
        background: #674563; color: #fff1f5; border-color: #ff9dbe;
    }
    QPushButton#modelAdvancedToggle {
        background: transparent; color: #ffd3e0; border: 1px solid #7c69a0;
        border-radius: 8px; padding: 7px 10px;
    }
    QPushButton#modelAdvancedToggle:hover { background: #493757; border-color: #ffb6ce; }
    QPushButton#modelApiKeyAction {
        background: #3b2c50; color: #ffe8f0; border: 1px solid #a77db4;
        border-radius: 8px; padding: 7px 10px; text-align: left;
    }
    QPushButton#modelApiKeyAction:hover { background: #493757; border-color: #ffb6ce; }
    QPushButton#modelApiKeyAction:disabled {
        background: #292331; color: #9f92a9; border-color: #594d68;
    }
    QPushButton#modelChannelSecondary, QPushButton#modelChannelDanger {
        background: #221a2e; color: #faf6fb; border: 1px solid #7c69a0;
        border-radius: 8px; padding: 6px 10px;
    }
    QPushButton#modelChannelSecondary:hover { background: #493757; border-color: #cdb8ff; }
    QPushButton#modelChannelDanger { color: #ffdce2; border-color: #a85d78; }
    QPushButton#modelChannelDanger:hover { background: #5a2c47; border-color: #ff8fa0; }
    QPushButton#modelChannelDanger:disabled { color: #8f829f; border-color: #46385c; }
    QCheckBox#modelSetupAutoSave { color: #ffd3e0; padding: 4px 0; }
    QCheckBox#modelSetupClearSecret { color: #ffb6c2; padding: 4px 0; }
    QScrollArea#modelSetupScroll, QAbstractScrollArea#modelSetupScroll {
        border: none; background: #16111f;
    }
    QAbstractScrollArea#modelSetupScroll::viewport { background: #16111f; }
    QScrollBar:vertical { background: #100c18; width: 12px; border: none; }
    QScrollBar::handle:vertical {
        background: #674563; border: 1px solid #7c69a0; border-radius: 5px;
    }
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
        background: #100c18;
    }
    QPushButton#modelSetupPrimary, QPushButton#modelSetupTestButton {
        background: #ff9dbe; color: #2b0f1c; border: none;
        border-radius: 8px; padding: 8px 14px;
    }
    QPushButton#modelSetupPrimary:hover,
    QPushButton#modelSetupTestButton:hover { background: #ffb6ce; }
    QDialogButtonBox QPushButton {
        background: #2e2440; color: #faf6fb; border: 1px solid #7c69a0;
        border-radius: 9px; padding: 8px 14px;
    }
    QDialogButtonBox QPushButton:hover { background: #493757; border-color: #ffb6ce; }
    """

    def _model_setup_stylesheet(theme: MD3Theme) -> str:
        """组合 MD3 基础组件与模型向导状态规则。"""

        return "\n\n".join(
            (build_md3_stylesheet(theme), remap_legacy_stylesheet(_MODEL_SETUP_STYLE, theme))
        )

    class ModelSetupCard(QWidget):
        """可嵌入配置中心的模型渠道设置卡片。"""

        saveRequested = Signal(object)
        deleteRequested = Signal(str)
        testRequested = Signal(object)
        userChanged = Signal()
        yamlChanged = Signal(str)
        validationChanged = Signal(bool, str)

        def __init__(
            self,
            draft: ModelSetupDraft | Mapping[str, Any] | None = None,
            *,
            channel_drafts: Sequence[ModelSetupDraft | Mapping[str, Any]] = (),
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("modelSetupCard")
            inherited_theme = getattr(parent, "_md3_theme", DARK_MD3_THEME)
            resolved_theme = (
                inherited_theme if isinstance(inherited_theme, MD3Theme) else DARK_MD3_THEME
            )
            self._fancy_style_controller = FancyStyleController(resolved_theme, self)
            self.apply_theme(resolved_theme)
            self._draft_valid = False
            self._channel_drafts: tuple[ModelSetupDraft, ...] = ()
            self._loading_channel_selection = False
            # 渠道优先级不需要普通用户编辑，但切换模型名称/预设时必须
            # 原样保留，否则保存一个已有渠道会悄悄改变路由顺序。
            self._channel_priority = _DEFAULT_PRIORITY
            self._pending_plain_api_key = ""
            self._secret_configured = False
            self._clear_secret_requested = False
            self._clear_secret_env_backup = ""
            self._original_channel_id = ""
            self._suppress_user_changed = True
            self._wizard_workspace_layout: QBoxLayout | None = None
            self._wizard_step_layout: QBoxLayout | None = None
            self._wizard_step_rail: QFrame | None = None
            self._responsive_narrow = False
            self._build_ui()
            self.set_channel_drafts(channel_drafts, current=draft or ModelSetupDraft())
            self._suppress_user_changed = False

        def apply_theme(self, theme: MD3Theme) -> None:
            """应用模型渠道卡片主题，不改变任何连接草稿。"""

            if not isinstance(theme, MD3Theme):
                raise TypeError("theme must be an MD3Theme")
            self._md3_theme = theme
            self.setProperty("md3Role", "root")
            controller = self._fancy_style_controller
            controller.updateTheme(theme)
            controller.attach(self)
            self.setPalette(build_qt_palette(theme))
            self.setStyleSheet(
                "\n\n".join((controller.stylesheet(), _model_setup_stylesheet(theme)))
            )

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            root.setContentsMargins(8, 8, 8, 8)
            root.setSpacing(12)

            header = FancyCard(
                elevated=True,
                theme=self._md3_theme,
            )
            header.setObjectName("modelWizardHeader")
            header_layout = header.content_layout
            header_layout.setSpacing(4)
            self._fancy_style_controller.register(header)
            title = QLabel("连接模型服务")
            title.setObjectName("modelSetupTitle")
            title.setProperty("md3Role", "title")
            header_layout.addWidget(title)
            hint = QLabel("按步骤选择服务、填写连接信息并完成测试；编辑会自动保存。")
            hint.setObjectName("modelSetupHint")
            hint.setProperty("md3Role", "muted")
            hint.setWordWrap(True)
            header_layout.addWidget(hint)
            root.addWidget(header)

            workspace = QBoxLayout(QBoxLayout.Direction.LeftToRight)
            workspace.setSpacing(14)
            self._wizard_workspace_layout = workspace
            step_rail = QFrame()
            step_rail.setObjectName("modelWizardStepRail")
            step_rail.setProperty("md3Role", "card-outlined")
            step_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom)
            step_layout.setContentsMargins(10, 12, 10, 12)
            step_layout.setSpacing(6)
            self._wizard_step_layout = step_layout
            self._wizard_step_rail = step_rail
            for number, label in (
                ("1", "选择服务"),
                ("2", "连接设置"),
                ("3", "测试与保存"),
            ):
                step_label = QLabel(f"{number}   {label}")
                step_label.setObjectName("modelWizardStepItem")
                step_label.setProperty("step", number)
                step_label.setAccessibleName(f"步骤 {number}：{label}")
                step_layout.addWidget(step_label)
            step_layout.addStretch(1)
            step_rail.setLayout(step_layout)
            workspace.addWidget(step_rail)

            content_host = QWidget()
            content_host.setObjectName("modelWizardContent")
            content = QVBoxLayout(content_host)
            content.setContentsMargins(0, 0, 0, 0)
            content.setSpacing(12)
            self._wizard_content_layout = content
            workspace.addWidget(content_host, 1)
            root.addLayout(workspace, 1)

            # 密钥输入只在 Qt 控件中短暂存在，草稿、YAML 和信号仍只传环境变量名。
            self._api_key_action = QPushButton("设置 API 密钥")
            self._api_key_action.setObjectName("modelApiKeyAction")
            self._api_key_action.setIcon(fancy_icon("file", theme=self._md3_theme))
            self._api_key_action.setAccessibleName("设置 API 密钥")
            self._api_key_action.setMinimumHeight(34)
            self._api_key_action.setToolTip("可只用于本次运行，也可明确选择写入本机配置文件。")
            content.addWidget(self._api_key_action)

            channel_group = FancyCard(
                "模型渠道",
                "管理多个服务，并明确当前编辑对象。",
                icon="file",
                theme=self._md3_theme,
            )
            channel_group.setObjectName("modelSetupGroup")
            self._fancy_style_controller.register(channel_group)
            channel_form = QFormLayout()
            channel_group.addLayout(channel_form)
            channel_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self.channel_selector = QComboBox()
            self.channel_selector.setObjectName("modelSetupField")
            self.channel_selector.setAccessibleName("已配置的模型渠道")
            self.channel_selector.setMinimumHeight(34)
            self.channel_selector.setToolTip("选择已有渠道进行修改，或选择新建渠道")
            channel_form.addRow("当前渠道", self.channel_selector)
            channel_actions = QHBoxLayout()
            self.new_channel_button = QPushButton("新建渠道")
            self.new_channel_button.setObjectName("modelChannelSecondary")
            self.new_channel_button.setIcon(fancy_icon("add", theme=self._md3_theme))
            self.new_channel_button.setAccessibleName("新建模型渠道")
            self.new_channel_button.clicked.connect(self._start_new_channel)
            self.delete_channel_button = QPushButton("删除当前渠道")
            self.delete_channel_button.setObjectName("modelChannelDanger")
            self.delete_channel_button.setIcon(fancy_icon("delete", theme=self._md3_theme))
            self.delete_channel_button.setAccessibleName("删除当前模型渠道")
            self.delete_channel_button.clicked.connect(self._request_delete_channel)
            channel_actions.addWidget(self.new_channel_button)
            channel_actions.addWidget(self.delete_channel_button)
            channel_actions.addStretch(1)
            channel_form.addRow("渠道管理", channel_actions)
            channel_hint = QLabel("切换渠道只会读取脱敏连接摘要；保存时仅更新当前渠道。")
            channel_hint.setObjectName("modelSetupFieldHint")
            channel_hint.setWordWrap(True)
            channel_form.addRow("", channel_hint)
            content.addWidget(channel_group)

            provider_step = FancyCard(
                "步骤 1 · 选择服务",
                "先选适配器，再选择该适配器支持的协议。",
                icon="forward",
                theme=self._md3_theme,
            )
            provider_step.setObjectName("modelWizardStepCard")
            provider_step.setProperty("step", "service")
            self._fancy_style_controller.register(provider_step)
            provider_layout = provider_step.content_layout
            provider_layout.setSpacing(8)
            step = QLabel("选择适配器预设")
            step.setObjectName("modelSetupStep")
            provider_layout.addWidget(step)
            preset_host = QWidget()
            preset_layout = QGridLayout(preset_host)
            preset_layout.setContentsMargins(0, 0, 0, 0)
            preset_layout.setHorizontalSpacing(8)
            preset_layout.setVerticalSpacing(8)
            self._preset_buttons: dict[str, QPushButton] = {}
            for index, (key, option) in enumerate(MODEL_SETUP_PRESETS.items()):
                preset_button = QPushButton(option.label)
                preset_button.setObjectName("modelPresetButton")
                preset_button.setProperty("presetKey", key)
                preset_button.setCheckable(True)
                preset_button.setAccessibleName(f"模型服务 {option.label}")
                preset_button.setToolTip(
                    "使用本地 Ollama，不需要 API Key"
                    if key == "ollama"
                    else f"选择 {option.label} 连接预设"
                )
                preset_button.clicked.connect(
                    lambda _checked=False, selected=key: self._select_preset(selected)
                )
                preset_layout.addWidget(preset_button, index // 3, index % 3)
                self._preset_buttons[key] = preset_button
            for column in range(3):
                preset_layout.setColumnStretch(column, 1)
            provider_layout.addWidget(preset_host)
            content.addWidget(provider_step)

            group = FancyCard(
                "步骤 2 · 模型与连接",
                "确认协议、模型名称和启用状态。",
                icon="forward",
                theme=self._md3_theme,
            )
            group.setObjectName("modelSetupGroup")
            group.setProperty("step", "connection")
            self._fancy_style_controller.register(group)
            form = QFormLayout()
            group.addLayout(form)
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            self.preset_combo = QComboBox()
            self.preset_combo.setObjectName("modelSetupField")
            self.preset_combo.setAccessibleName("模型适配器")
            self.preset_combo.setMinimumHeight(34)
            for key, option in MODEL_SETUP_PRESETS.items():
                self.preset_combo.addItem(option.label, key)
            # 新名称用于两级适配器→协议交互；保留 preset_combo 属性兼容旧宿主。
            self.adapter_combo = self.preset_combo
            self.channel_id_edit = self._line("primary", "YAML 中 llm.channels[].id")
            self.protocol_combo = QComboBox()
            self.protocol_combo.setObjectName("modelSetupField")
            self.protocol_combo.setAccessibleName("适配器协议")
            self.protocol_combo.setMinimumHeight(34)
            self._set_protocol_options("custom")
            self.base_url_edit = self._line("https://…", "必须是 http 或 https 地址，不含查询参数")
            self.model_edit = self._line("模型名称", "例如 gpt-4o-mini 或本地模型名")
            self.api_key_env_edit = self._line("OPENAI_API_KEY", "只填写大写环境变量名，不读取其值")
            self.api_key_value_edit = self._line(
                "仅本次运行使用，不会写入 YAML", "仅在测试或保存前写入当前进程环境变量"
            )
            self.api_key_value_edit.setObjectName("modelSetupSecretField")
            self.api_key_value_edit.setAccessibleName("API 密钥，仅本次运行使用")
            self.api_key_value_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.api_key_value_edit.setMaxLength(4096)
            self.api_key_value_edit.setClearButtonEnabled(True)
            self.api_key_persist_check = QCheckBox("将 API 密钥写入配置文件（明文）")
            self.api_key_persist_check.setObjectName("modelSetupPersistSecret")
            # 默认沿用历史“仅本次运行”行为；用户可显式开启文件持久化。
            self.api_key_persist_check.setChecked(False)
            self.api_key_persist_check.setToolTip(
                "密钥仅写入本机配置文件，界面、日志和预览始终脱敏"
            )
            self.clear_api_key_check = QCheckBox("清除已保存的 API 密钥")
            self.clear_api_key_check.setObjectName("modelSetupClearSecret")
            self.clear_api_key_check.setToolTip("清除后当前渠道不再携带密钥")
            self.clear_api_key_check.setVisible(False)
            self.enabled_check = QCheckBox("启用此渠道")
            self.enabled_check.setChecked(True)
            self.capabilities_edit = self._line(
                "streaming,tools", "能力按逗号分隔；streaming 会自动保留"
            )
            self.priority_spin = QSpinBox()
            self.priority_spin.setObjectName("modelSetupField")
            self.priority_spin.setRange(-1_000_000, 1_000_000)
            self.priority_spin.setValue(_DEFAULT_PRIORITY)
            self.priority_spin.setToolTip("数值越小越优先；相同优先级按渠道顺序处理")
            self.base_url_edit.setReadOnly(True)
            self.base_url_edit.setToolTip("服务预设自动填写；自定义服务请打开高级连接选项")
            self.model_edit.setPlaceholderText("例如 gpt-4o-mini 或本地模型名")
            self.api_key_env_edit.setPlaceholderText("例如 OPENAI_API_KEY；本地 Ollama 留空")
            # 连接选择严格按“适配器 → 协议 → 模型”展开，用户不必
            # 在填写模型后才发现协议列表已经切换。
            form.addRow("适配器", self.adapter_combo)
            form.addRow("协议", self.protocol_combo)
            form.addRow("模型名称", self.model_edit)
            form.addRow("启用状态", self.enabled_check)
            self._connection_summary = QLabel("服务连接由上方服务预设管理。")
            self._connection_summary.setObjectName("modelSetupFieldHint")
            self._connection_summary.setWordWrap(True)
            form.addRow("", self._connection_summary)
            model_hint = QLabel("模型名称可直接修改；连接详情按需在更多设置中调整。")
            model_hint.setObjectName("modelSetupFieldHint")
            model_hint.setWordWrap(True)
            form.addRow("", model_hint)
            content.addWidget(group)

            self._advanced_toggle = QPushButton("更多连接设置")
            self._advanced_toggle.setObjectName("modelAdvancedToggle")
            self._advanced_toggle.setCheckable(True)
            self._advanced_toggle.setToolTip("仅自定义协议、渠道 ID 或能力声明时使用")
            self._advanced_toggle.toggled.connect(self._set_advanced_visible)
            self._api_key_action.clicked.connect(self._open_api_key_settings)
            content.addWidget(self._advanced_toggle)

            self._advanced_group = FancyCard(
                "更多连接设置",
                "仅用于自定义地址、环境变量、优先级和能力声明。",
                icon="file",
                theme=self._md3_theme,
            )
            self._advanced_group.setObjectName("modelSetupGroup")
            self._fancy_style_controller.register(self._advanced_group)
            advanced_form = QFormLayout()
            self._advanced_group.addLayout(advanced_form)
            advanced_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            advanced_form.addRow("服务地址", self.base_url_edit)
            advanced_form.addRow("密钥环境变量（可选）", self.api_key_env_edit)
            self._api_key_value_label = QLabel("API 密钥（仅本次运行）")
            self._api_key_value_label.setToolTip("不会写入 YAML、日志、预览或回调参数")
            advanced_form.addRow(self._api_key_value_label, self.api_key_value_edit)
            advanced_form.addRow("密钥持久化", self.api_key_persist_check)
            advanced_form.addRow("密钥操作", self.clear_api_key_check)
            advanced_form.addRow("渠道 ID", self.channel_id_edit)
            advanced_form.addRow("渠道优先级", self.priority_spin)
            advanced_form.addRow("能力声明", self.capabilities_edit)
            self._advanced_group.setVisible(False)
            content.addWidget(self._advanced_group)

            validation_step = FancyCard(
                "步骤 3 · 测试与保存",
                "先测试最小请求，再由自动保存或底部命令提交。",
                icon="success",
                theme=self._md3_theme,
            )
            validation_step.setObjectName("modelWizardStepCard")
            validation_step.setProperty("step", "validation")
            self._fancy_style_controller.register(validation_step)
            validation_layout = validation_step.content_layout
            validation_layout.setSpacing(8)
            preview_group = FancyCard(
                "连接摘要",
                "仅显示脱敏后的渠道配置。",
                icon="info",
                theme=self._md3_theme,
            )
            preview_group.setObjectName("modelSetupGroup")
            self._fancy_style_controller.register(preview_group)
            preview_layout = preview_group.content_layout
            self.preview_edit = QPlainTextEdit()
            self.preview_edit.setObjectName("modelSetupPreview")
            self.preview_edit.setReadOnly(True)
            self.preview_edit.setMinimumHeight(140)
            preview_layout.addWidget(self.preview_edit)
            self._preview_toggle = QPushButton("查看连接摘要")
            self._preview_toggle.setObjectName("modelAdvancedToggle")
            self._preview_toggle.setCheckable(True)
            self._preview_toggle.toggled.connect(preview_group.setVisible)
            validation_layout.addWidget(self._preview_toggle)
            preview_group.setVisible(False)
            validation_layout.addWidget(preview_group)
            self.status_label = QLabel("请选择服务预设并确认模型名称。")
            self.status_label.setObjectName("modelSetupStatus")
            self.status_label.setWordWrap(True)
            self.status_label.setVisible(False)
            validation_layout.addWidget(self.status_label)
            self.status_info = FancyInfoBar(
                "连接状态",
                self.status_label.text(),
                severity="info",
                closable=False,
                theme=self._md3_theme,
            )
            self.status_info.setObjectName("modelSetupInfoBar")
            self._fancy_style_controller.register(self.status_info)
            validation_layout.addWidget(self.status_info)
            self.test_button = QPushButton("测试连接")
            self.test_button.setObjectName("modelSetupTestButton")
            self.test_button.setIcon(fancy_icon("play", theme=self._md3_theme))
            self.test_button.setToolTip("发送一个不带工具的最小连接测试请求")
            self.test_button.clicked.connect(self.test_connection)
            validation_layout.addWidget(self.test_button)
            content.addWidget(validation_step)
            self._update_responsive_layout(force=True)

            self.preset_combo.currentIndexChanged.connect(self._apply_preset)
            self.channel_selector.currentIndexChanged.connect(self._select_channel)
            self.priority_spin.valueChanged.connect(self._on_user_field_changed)
            for editor in (
                self.channel_id_edit,
                self.base_url_edit,
                self.model_edit,
                self.api_key_env_edit,
                self.api_key_value_edit,
                self.capabilities_edit,
            ):
                editor.textChanged.connect(self._on_user_field_changed)
            self.protocol_combo.currentIndexChanged.connect(self._on_protocol_changed)
            self.enabled_check.toggled.connect(self._on_user_field_changed)
            self.api_key_persist_check.toggled.connect(self._on_user_field_changed)
            self.clear_api_key_check.toggled.connect(self._on_clear_secret_toggled)

        def _update_responsive_layout(self, *, force: bool = False) -> None:
            """在窄向导中把步骤导航移到内容上方。"""

            narrow = self.width() < 720
            if narrow == self._responsive_narrow and not force:
                return
            self._responsive_narrow = narrow
            workspace = self._wizard_workspace_layout
            step_layout = self._wizard_step_layout
            rail = self._wizard_step_rail
            if workspace is not None:
                workspace.setDirection(
                    QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight
                )
            if step_layout is not None:
                step_layout.setDirection(
                    QBoxLayout.Direction.LeftToRight if narrow else QBoxLayout.Direction.TopToBottom
                )
            if rail is not None:
                rail.setMinimumWidth(0)
                rail.setMaximumWidth(16777215 if narrow else 168)
                rail.setMaximumHeight(74 if narrow else 16777215)
            self.setProperty("responsiveMode", "stacked" if narrow else "split")
            style = self.style()
            style.unpolish(self)
            style.polish(self)
            self.updateGeometry()

        def resizeEvent(self, event: object) -> None:  # noqa: N802
            """随承载窗口宽度切换步骤导航方向。"""

            super().resizeEvent(event)
            self._update_responsive_layout()

        def _on_user_field_changed(self, *_args: object) -> None:
            """刷新预览并通知向导启动自动保存防抖。"""

            self._refresh_preview()
            if not self._suppress_user_changed:
                self.userChanged.emit()

        def _on_clear_secret_toggled(self, checked: bool) -> None:
            """同步明确的密钥清除意图，并避免与新密钥输入同时生效。"""

            checked = bool(checked)
            self._clear_secret_requested = checked
            if checked:
                self._clear_secret_env_backup = self.api_key_env_edit.text()
                self.api_key_env_edit.blockSignals(True)
                self.api_key_env_edit.clear()
                self.api_key_env_edit.blockSignals(False)
                self.api_key_env_edit.setReadOnly(True)
                self.api_key_value_edit.setEnabled(False)
            else:
                if self._clear_secret_env_backup and not self.api_key_env_edit.text().strip():
                    self.api_key_env_edit.blockSignals(True)
                    self.api_key_env_edit.setText(self._clear_secret_env_backup)
                    self.api_key_env_edit.blockSignals(False)
                self._clear_secret_env_backup = ""
                self.api_key_value_edit.setEnabled(True)
            if checked:
                self.clear_transient_api_key()
                self.api_key_persist_check.blockSignals(True)
                self.api_key_persist_check.setChecked(False)
                self.api_key_persist_check.blockSignals(False)
            self._on_user_field_changed()

        @staticmethod
        def _draft_from_channel_value(
            value: ModelSetupDraft | Mapping[str, Any],
        ) -> ModelSetupDraft:
            """把渠道选择器输入转换为只含安全引用的草稿。"""

            if isinstance(value, ModelSetupDraft):
                return value
            sanitized = dict(value)
            raw_key = sanitized.get("api_key")
            if (
                isinstance(raw_key, str)
                and raw_key
                and _ENV_REFERENCE_PATTERN.fullmatch(raw_key) is None
            ):
                sanitized["api_key"] = _REDACTION_MARKER
            return ModelSetupDraft.from_mapping(sanitized)

        @staticmethod
        def _new_channel_id(drafts: Sequence[ModelSetupDraft]) -> str:
            """生成不覆盖已有渠道的默认标识。"""

            occupied = {draft.channel_id for draft in drafts}
            if _DEFAULT_CHANNEL_ID not in occupied:
                return _DEFAULT_CHANNEL_ID
            index = 2
            while f"{_DEFAULT_CHANNEL_ID}-{index}" in occupied:
                index += 1
            return f"{_DEFAULT_CHANNEL_ID}-{index}"

        @staticmethod
        def _channel_label(draft: ModelSetupDraft) -> str:
            """生成不含地址和密钥的渠道选择标签。"""

            model = draft.model or "模型未填写"
            state = "已启用" if draft.enabled else "已停用"
            return f"{draft.channel_id} · {model} · {state}"

        def set_channel_drafts(
            self,
            values: Sequence[ModelSetupDraft | Mapping[str, Any]],
            *,
            current: ModelSetupDraft | Mapping[str, Any] | None = None,
        ) -> None:
            """刷新已配置渠道列表，并保留当前编辑目标。"""

            drafts: list[ModelSetupDraft] = []
            seen_ids: set[str] = set()
            for value in values:
                try:
                    draft = self._draft_from_channel_value(value)
                except (TypeError, ValueError):
                    continue
                if draft.channel_id in seen_ids:
                    continue
                seen_ids.add(draft.channel_id)
                drafts.append(draft)
            try:
                active = (
                    current
                    if isinstance(current, ModelSetupDraft)
                    else self._draft_from_channel_value(current)
                    if current is not None
                    else None
                )
            except (TypeError, ValueError):
                active = None
            active_has_content = bool(
                active is not None
                and (
                    active.base_url
                    or active.model
                    or active.api_key_env
                    or active.channel_id != _DEFAULT_CHANNEL_ID
                )
            )
            if active_has_content and active is not None and active.channel_id not in seen_ids:
                drafts.insert(0, active)
                seen_ids.add(active.channel_id)
            self._channel_drafts = tuple(drafts)
            self._loading_channel_selection = True
            self.channel_selector.blockSignals(True)
            self.channel_selector.clear()
            new_draft = ModelSetupDraft(channel_id=self._new_channel_id(self._channel_drafts))
            self.channel_selector.addItem("新建模型渠道", new_draft)
            for draft in self._channel_drafts:
                self.channel_selector.addItem(self._channel_label(draft), draft)
            selected_index = 0
            if active is not None:
                for index, draft in enumerate(self._channel_drafts, start=1):
                    if draft.channel_id == active.channel_id:
                        selected_index = index
                        break
            self.channel_selector.setCurrentIndex(selected_index)
            self.channel_selector.blockSignals(False)
            self._loading_channel_selection = False
            selected = self.channel_selector.currentData()
            if isinstance(selected, ModelSetupDraft):
                previous = self._suppress_user_changed
                self._suppress_user_changed = True
                try:
                    self.set_draft(selected, sync_selector=False)
                finally:
                    self._suppress_user_changed = previous
            self._refresh_channel_actions()

        def _select_channel(self, _index: int) -> None:
            """切换已有草稿或创建新的可编辑渠道。"""

            if self._loading_channel_selection:
                return
            selected = self.channel_selector.currentData()
            if isinstance(selected, ModelSetupDraft):
                previous = self._suppress_user_changed
                self._suppress_user_changed = True
                try:
                    self.set_draft(selected, sync_selector=False)
                finally:
                    self._suppress_user_changed = previous
            self._refresh_channel_actions()

        def _start_new_channel(self) -> None:
            """切换到不会覆盖已有渠道的新建草稿。"""

            self.channel_selector.setCurrentIndex(0)
            self._refresh_channel_actions()

        def _request_delete_channel(self) -> None:
            """请求宿主删除当前渠道；删除由宿主负责原子保存。"""

            selected = self.channel_selector.currentData()
            if (
                not isinstance(selected, ModelSetupDraft)
                or self.channel_selector.currentIndex() <= 0
            ):
                return
            self.deleteRequested.emit(selected.channel_id)

        def _refresh_channel_actions(self) -> None:
            """同步新建/删除按钮，避免把新建占位项误删。"""

            if not hasattr(self, "delete_channel_button"):
                return
            is_existing = self.channel_selector.currentIndex() > 0
            self.delete_channel_button.setEnabled(is_existing)
            self.delete_channel_button.setToolTip(
                "删除当前渠道并更新对话路由" if is_existing else "新建草稿不能删除"
            )

        @property
        def channel_drafts(self) -> tuple[ModelSetupDraft, ...]:
            """返回当前选择器中的已配置渠道脱敏草稿。"""

            return self._channel_drafts

        @property
        def clear_persisted_secret_requested(self) -> bool:
            """返回是否已有明确的凭据清除意图。"""

            return self._clear_secret_requested

        def upsert_channel_draft(self, draft: ModelSetupDraft | Mapping[str, Any]) -> None:
            """把成功保存的当前渠道同步回选择器，避免切换时丢失最新编辑。"""

            current = self._draft_from_channel_value(draft)
            values = [
                item for item in self._channel_drafts if item.channel_id != current.channel_id
            ]
            values.append(current)
            self.set_channel_drafts(values, current=current)

        def _select_preset(self, key: str) -> None:
            """由大号预设按钮驱动隐藏兼容下拉框。"""

            index = self.preset_combo.findData(str(key))
            if index < 0:
                return
            self.preset_combo.setCurrentIndex(index)

        def _set_protocol_options(self, adapter: str, *, preferred: str = "") -> None:
            """按适配器刷新协议下拉框，并尽量保留当前协议。"""

            key = str(adapter or "custom").strip().lower()
            protocols = protocols_for_adapter(key)
            current = (
                str(self.protocol_combo.currentData() or "") if self.protocol_combo.count() else ""
            )
            selected = (
                preferred
                if preferred in protocols
                else current
                if current in protocols
                else protocols[0]
            )
            previous_blocked = self.protocol_combo.blockSignals(True)
            try:
                self.protocol_combo.clear()
                for protocol in protocols:
                    self.protocol_combo.addItem(PROTOCOL_LABELS.get(protocol, protocol), protocol)
                index = self.protocol_combo.findData(selected)
                self.protocol_combo.setCurrentIndex(max(index, 0))
            finally:
                self.protocol_combo.blockSignals(previous_blocked)
            label = MODEL_SETUP_PRESETS.get(key, MODEL_SETUP_PRESETS["custom"]).label
            self.protocol_combo.setToolTip(f"{label} 支持：{'、'.join(protocols)}")

        def _set_advanced_visible(self, enabled: bool) -> None:
            """切换技术连接字段，并允许自定义服务地址。"""

            self._advanced_group.setVisible(bool(enabled))
            key = str(self.preset_combo.currentData() or "custom")
            self.base_url_edit.setReadOnly(not enabled and key != "custom")
            self._advanced_toggle.setText("收起更多连接设置" if enabled else "更多连接设置")
            self._refresh_connection_summary()

        def _open_api_key_settings(self) -> None:
            """打开本次运行密钥入口并聚焦环境变量名称。"""

            self._advanced_toggle.setChecked(True)
            if self._uses_ollama():
                return
            target = (
                self.api_key_value_edit
                if self._secret_configured and not self.api_key_env_edit.text().strip()
                else self.api_key_env_edit
            )
            target.setFocus(Qt.FocusReason.OtherFocusReason)
            target.selectAll()

            # 高级区位于滚动卡片下方；仅展开并聚焦不会自动把字段滚入
            # 可视区域，用户会误以为按钮没有生效。布局完成后显式滚动到
            # 环境变量字段，仍保持普通路径不展示其它内部参数。
            def ensure_visible() -> None:
                parent = self.parentWidget()
                while parent is not None:
                    if isinstance(parent, QScrollArea):
                        try:
                            parent.ensureWidgetVisible(target)
                            # 密钥输入紧邻环境变量字段；同时确保密码框本身
                            # 完整落在视口内，避免只看到变量名而找不到输入位置。
                            parent.ensureWidgetVisible(self.api_key_value_edit)
                        except (AttributeError, RuntimeError, TypeError, ValueError):
                            pass
                        return
                    parent = parent.parentWidget()

            try:
                QTimer.singleShot(0, ensure_visible)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                ensure_visible()

        def _uses_ollama(self) -> bool:
            """判断当前渠道是否为不需要 API Key 的本地 Ollama。"""

            return str(self.protocol_combo.currentData() or "") == "ollama_chat"

        def _on_protocol_changed(self, _index: int) -> None:
            """切换到 Ollama 时清除不适用的密钥引用和临时输入。"""

            if self._uses_ollama():
                self.api_key_env_edit.clear()
                self.api_key_env_edit.setReadOnly(False)
                self.api_key_value_edit.setEnabled(True)
                self.clear_transient_api_key()
                self._clear_secret_requested = False
                self.clear_api_key_check.blockSignals(True)
                self.clear_api_key_check.setChecked(False)
                self.clear_api_key_check.blockSignals(False)
            self._refresh_preview()
            if not self._suppress_user_changed:
                self.userChanged.emit()

        def clear_transient_api_key(self) -> None:
            """擦除密码控件及其撤销历史，不触碰已应用的环境变量。"""

            # QLineEdit 没有公开的 setUndoRedoEnabled；``clear()`` 会留下
            # 可由 Ctrl+Z 恢复的明文。使用 setText("") 重置内部撤销栈，
            # 并屏蔽一次预览信号，确保密钥不再回到界面或草稿。
            editor = self.api_key_value_edit
            self._pending_plain_api_key = ""
            previous_blocked = editor.blockSignals(True)
            try:
                editor.setText("")
            finally:
                editor.blockSignals(previous_blocked)

        def _apply_transient_api_key(self, *, persist: bool = False) -> bool:
            """校验草稿后把密码控件值写入当前进程环境变量。

            默认明文不进入草稿、YAML、信号 payload 或状态文本。勾选文件持久化
            时由 ``ModelSetupDialog`` 交给受控保存回调；其它情况写入当前进程
            环境变量并立即清空控件。
            """

            # 用户已经在密码框输入密钥但没有填写变量名时，按当前适配器的
            # 明确预设选择规范变量；只有自定义适配器才回退到 MEAPET_API_KEY。
            secret = self.api_key_value_edit.text()
            try:
                draft = self.draft().validate()
            except ValueError as exc:
                self._set_status(_friendly_model_error(exc), error=True)
                self.clear_transient_api_key()
                return False
            if draft.protocol == "ollama_chat":
                self.clear_transient_api_key()
                return True
            if secret and not draft.api_key_env:
                adapter = str(self.preset_combo.currentData() or "custom").strip().lower()
                option = MODEL_SETUP_PRESETS.get(adapter)
                fallback = option.api_key_env if option is not None else ""
                self.api_key_env_edit.setText(preferred_api_key_env(adapter, fallback))
                try:
                    draft = self.draft().validate()
                except ValueError as exc:
                    self._set_status(_friendly_model_error(exc), error=True)
                    self.clear_transient_api_key()
                    return False
            if not secret:
                return True
            if len(secret) > 4096 or any(char in secret for char in "\r\n\x00"):
                self._set_status("API 密钥格式无效，请重新输入。", error=True)
                self.clear_transient_api_key()
                return False
            if persist and self.api_key_persist_check.isChecked():
                self.clear_api_key_check.blockSignals(True)
                self.clear_api_key_check.setChecked(False)
                self.clear_api_key_check.blockSignals(False)
                self._clear_secret_requested = False
                self.clear_transient_api_key()
                self._pending_plain_api_key = secret
                return True
            try:
                env_name = _environment_name(draft.api_key_env)
            except ValueError as exc:
                self._set_status(_friendly_model_error(exc), error=True)
                return False
            if not env_name:
                self._set_status("请先填写密钥环境变量名，再输入 API 密钥。", error=True)
                return False
            try:
                os.environ[env_name] = secret
            except (OSError, TypeError, ValueError):
                self._set_status("本次运行密钥写入失败，请检查环境变量名后重试。", error=True)
                return False
            self.clear_transient_api_key()
            return True

        def take_pending_plain_api_key(self) -> str:
            """取出一次性文件持久化密钥；调用方必须立即完成保存并丢弃。"""

            value = self._pending_plain_api_key
            self._pending_plain_api_key = ""
            return value

        def _refresh_connection_summary(self) -> None:
            """更新可操作的连接摘要，不回显密钥值或完整环境变量名。"""

            if not hasattr(self, "_connection_summary"):
                return
            key = str(self.preset_combo.currentData() or "custom")
            has_url = bool(self.base_url_edit.text().strip())
            env_name = self.api_key_env_edit.text().strip()
            local_ollama = self._uses_ollama()
            if local_ollama:
                summary = "本地 Ollama · 无需 API Key。"
                action_text = "本地服务无需密钥"
                action_enabled = False
                action_tooltip = "本地 Ollama 默认不需要 API Key。"
            elif key != "custom" and has_url:
                if self._secret_configured and not env_name and not self._clear_secret_requested:
                    auth_hint = "本机已保存密钥（内容隐藏）"
                    action_text = "替换或清除 API 密钥"
                elif env_name:
                    env_state = "已检测到" if os.environ.get(env_name) else "未检测到"
                    auth_hint = f"云端密钥{env_state}"
                    action_text = "修改 API 密钥环境变量"
                else:
                    auth_hint = "需要 API 密钥环境变量"
                    action_text = "设置 API 密钥环境变量"
                summary = f"{MODEL_SETUP_PRESETS[key].label} · {auth_hint}。"
                action_enabled = True
                action_tooltip = (
                    "只填写大写环境变量名；密钥由启动环境提供，也可在下方输入本次运行密钥。"
                )
            elif has_url:
                if self._secret_configured and not env_name and not self._clear_secret_requested:
                    auth_hint = "本机已保存密钥（内容隐藏）"
                    action_text = "替换或清除 API 密钥"
                else:
                    auth_hint = "已设置密钥环境变量" if env_name else "可按需设置 API 密钥环境变量"
                    action_text = "修改 API 密钥环境变量" if env_name else "设置 API 密钥环境变量"
                summary = f"自定义服务 · {auth_hint}。"
                action_enabled = True
                action_tooltip = (
                    "只填写大写环境变量名；密钥由启动环境提供，也可在下方输入本次运行密钥。"
                )
            else:
                summary = "请选择服务预设，或在更多连接设置中填写自定义服务。"
                action_text = "设置 API 密钥环境变量"
                action_enabled = True
                action_tooltip = "先选择服务预设；实际密钥只通过环境变量提供。"
            persist_file = bool(not local_ollama and self.api_key_persist_check.isChecked())
            self._api_key_value_label.setText(
                "API 密钥（写入配置文件）" if persist_file else "API 密钥（仅本次运行）"
            )
            self._api_key_value_label.setVisible(not local_ollama)
            self.api_key_value_edit.setVisible(not local_ollama)
            self.clear_api_key_check.setVisible(bool(self._secret_configured and not local_ollama))
            self.api_key_value_edit.setAccessibleName(
                "API 密钥，将写入配置文件" if persist_file else "API 密钥，仅本次运行使用"
            )
            if local_ollama:
                self.clear_transient_api_key()
            gateway_hint = openai_base_url_hint(
                self.protocol_combo.currentData(), self.base_url_edit.text()
            )
            if gateway_hint:
                summary = f"{summary}\n提示：{gateway_hint}"
            self._connection_summary.setText(summary)
            self._connection_summary.setToolTip(gateway_hint)
            self._api_key_action.setText(action_text)
            self._api_key_action.setEnabled(action_enabled)
            self._api_key_action.setToolTip(action_tooltip)

        @staticmethod
        def _line(placeholder: str, tooltip: str) -> QLineEdit:
            editor = QLineEdit()
            editor.setObjectName("modelSetupField")
            editor.setMinimumHeight(34)
            editor.setPlaceholderText(placeholder)
            editor.setToolTip(tooltip)
            return editor

        def _apply_preset(self, index: int) -> None:
            key = str(self.preset_combo.itemData(index) or "custom")
            try:
                option = MODEL_SETUP_PRESETS[_preset_key(key)]
            except ValueError:
                return
            editors = (
                self.channel_id_edit,
                self.base_url_edit,
                self.model_edit,
                self.api_key_env_edit,
                self.capabilities_edit,
            )
            for editor in editors:
                editor.blockSignals(True)
            # 预设只改变协议/地址/模型，不重写当前渠道 ID；否则在多渠道
            # 选择“新建”后点 OpenAI 会意外覆盖已有 ``openai`` 渠道。
            current_id = self.channel_id_edit.text().strip()
            self.channel_id_edit.setText(
                current_id or (option.key if key != "custom" else _DEFAULT_CHANNEL_ID)
            )
            self.base_url_edit.setText(option.base_url)
            self.model_edit.setText(option.model)
            # 自定义服务允许免密网关；不能因为规范回退名存在就自动写入
            # 一个未设置的 ${MEAPET_API_KEY}，否则保存后热加载会失败。
            env_name = preferred_api_key_env(key, option.api_key_env) if key != "custom" else ""
            self.api_key_env_edit.setText(env_name)
            self.api_key_env_edit.setReadOnly(False)
            self.clear_transient_api_key()
            self.api_key_persist_check.setChecked(False)
            self._secret_configured = bool(env_name)
            self._clear_secret_requested = False
            self._clear_secret_env_backup = ""
            self.clear_api_key_check.blockSignals(True)
            self.clear_api_key_check.setChecked(False)
            self.clear_api_key_check.blockSignals(False)
            self.capabilities_edit.setText(",".join(option.capabilities))
            self._set_protocol_options(key, preferred=option.protocol)
            for editor in editors:
                editor.blockSignals(False)
            for button_key, preset_button in self._preset_buttons.items():
                preset_button.blockSignals(True)
                preset_button.setChecked(button_key == key)
                preset_button.blockSignals(False)
            self._set_advanced_visible(self._advanced_toggle.isChecked())
            self._refresh_preview()
            if not self._suppress_user_changed:
                self.userChanged.emit()

        def _draft_from_widgets(self) -> ModelSetupDraft:
            protocol = str(self.protocol_combo.currentData() or _DEFAULT_PROTOCOL)
            api_key_env = (
                ""
                if protocol == "ollama_chat" or self._clear_secret_requested
                else self.api_key_env_edit.text()
            )
            return ModelSetupDraft(
                channel_id=self.channel_id_edit.text(),
                protocol=protocol,
                base_url=self.base_url_edit.text(),
                model=self.model_edit.text(),
                api_key_env=api_key_env,
                capabilities=self.capabilities_edit.text(),
                enabled=self.enabled_check.isChecked(),
                priority=self.priority_spin.value(),
                preset=str(self.preset_combo.currentData() or "custom"),
                has_persisted_secret=(
                    False
                    if self._clear_secret_requested
                    else self._secret_configured or bool(api_key_env.strip())
                ),
                clear_persisted_secret=self._clear_secret_requested,
            )

        def set_draft(
            self,
            draft: ModelSetupDraft | Mapping[str, Any],
            *,
            sync_selector: bool = True,
        ) -> None:
            """将纯函数草稿同步到表单。"""

            current = (
                draft if isinstance(draft, ModelSetupDraft) else ModelSetupDraft.from_mapping(draft)
            )
            self._original_channel_id = (
                current.channel_id
                if any(item.channel_id == current.channel_id for item in self._channel_drafts)
                else ""
            )
            if sync_selector and hasattr(self, "channel_selector"):
                selected_index = 0
                for index, option in enumerate(self._channel_drafts, start=1):
                    if option.channel_id == current.channel_id:
                        selected_index = index
                        break
                self._loading_channel_selection = True
                self.channel_selector.blockSignals(True)
                self.channel_selector.setCurrentIndex(selected_index)
                self.channel_selector.blockSignals(False)
                self._loading_channel_selection = False
            preset_index = self.preset_combo.findData(current.preset)
            if preset_index < 0:
                preset_index = self.preset_combo.findData("custom")
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(preset_index)
            self.preset_combo.blockSignals(False)
            self.channel_id_edit.setText(current.channel_id)
            self._set_protocol_options(current.preset, preferred=current.protocol)
            self.base_url_edit.setText(current.base_url)
            self.model_edit.setText(current.model)
            display_env = current.api_key_env
            if not display_env and current.preset != "custom" and not current.has_persisted_secret:
                preset = MODEL_SETUP_PRESETS.get(current.preset)
                suggested = preferred_api_key_env(
                    current.preset,
                    preset.api_key_env if preset is not None else "",
                )
                # 只有当前进程确实提供了该变量时才自动填入，避免用户
                # 保存免密网关时被无意间写入一个不存在的引用。
                if suggested and os.environ.get(suggested):
                    display_env = suggested
            self.api_key_env_edit.setText(display_env)
            if current.protocol == "ollama_chat":
                self.api_key_env_edit.clear()
            self.clear_transient_api_key()
            self.api_key_persist_check.setChecked(False)
            self._secret_configured = bool(current.has_persisted_secret or display_env)
            self._clear_secret_requested = bool(current.clear_persisted_secret)
            self._clear_secret_env_backup = ""
            self.api_key_env_edit.setReadOnly(self._clear_secret_requested)
            self.api_key_value_edit.setEnabled(not self._clear_secret_requested)
            if self._clear_secret_requested:
                self.api_key_env_edit.blockSignals(True)
                self.api_key_env_edit.clear()
                self.api_key_env_edit.blockSignals(False)
            self.clear_api_key_check.blockSignals(True)
            self.clear_api_key_check.setChecked(self._clear_secret_requested)
            self.clear_api_key_check.blockSignals(False)
            self.capabilities_edit.setText(",".join(current.capabilities))
            self.enabled_check.setChecked(current.enabled)
            self._channel_priority = current.priority
            self.priority_spin.setValue(current.priority)
            for key, button in self._preset_buttons.items():
                button.blockSignals(True)
                button.setChecked(key == current.preset)
                button.blockSignals(False)
            self._set_advanced_visible(self._advanced_toggle.isChecked())
            self._refresh_preview()

        def draft(self) -> ModelSetupDraft:
            """读取当前表单的纯数据草稿。"""

            return self._draft_from_widgets()

        @property
        def is_valid(self) -> bool:
            """返回当前表单是否具备可提交的完整渠道配置。"""

            return self._draft_valid

        def yaml_text(self) -> str:
            """返回当前表单的脱敏 YAML 草稿。"""

            return self.draft().to_yaml(validate=False)

        def payload(self) -> dict[str, Any]:
            """校验并返回回调 payload。"""

            return self.draft().to_callback_payload()

        def submit(self) -> bool:
            """校验当前表单并发出保存请求。"""

            current_id = self.channel_id_edit.text().strip()
            if self._original_channel_id and current_id != self._original_channel_id:
                self._set_status("已有渠道的 ID 不可直接改名，请新建渠道后删除旧渠道。", error=True)
                return False
            if not self._apply_transient_api_key(persist=True):
                return False
            try:
                payload = self.payload()
            except ValueError as exc:
                self._set_status(_friendly_model_error(exc), error=True)
                return False
            self._set_status("正在提交模型渠道，请稍候…", error=False)
            if self._pending_plain_api_key:
                payload["api_key_env"] = ""
                payload["_persist_api_key"] = True
            self.saveRequested.emit(payload)
            return True

        def test_connection(self) -> bool:
            """发出脱敏连接测试请求；实际网络调用由宿主运行时完成。"""

            if not self._apply_transient_api_key(persist=False):
                return False
            try:
                payload = self.draft().to_test_payload()
            except ValueError as exc:
                self._set_status(_friendly_model_error(exc), error=True)
                return False
            self._set_status("正在测试模型连接，请稍候…", error=False)
            self.testRequested.emit(payload)
            return True

        def set_test_result(self, success: bool, message: str) -> None:
            """由宿主回写脱敏探测结果。"""

            self._set_status(
                _safe_model_result_text(
                    message,
                    "模型连接测试通过。" if success else "模型连接测试未通过。",
                ),
                error=not success,
            )

        def set_test_pending(self, message: str = "正在测试模型连接，请稍候…") -> None:
            """显示异步探测进行中状态，不把等待误标为失败。"""

            self._set_status(
                _safe_model_result_text(message, "正在测试模型连接，请稍候…"),
                error=False,
            )

        def set_submit_result(self, success: bool, message: str) -> None:
            """由宿主回写保存结果，避免把“已发出”误报成“已保存”。"""

            self._set_status(
                str(message or ("模型渠道已保存。" if success else "保存未完成。")),
                error=not success,
            )

        def set_notice(self, message: str) -> None:
            """显示中性操作提示，不改变保存成功/失败语义。"""

            self._set_status(str(message or ""), error=False)

        def _refresh_preview(self) -> None:
            try:
                draft = self.draft()
                yaml_text = draft.to_yaml(validate=False)
                preset = MODEL_SETUP_PRESETS.get(draft.preset)
                service_label = preset.label if preset is not None else "自定义服务"
                if draft.api_key_env:
                    env_state = "已设置" if os.environ.get(draft.api_key_env) else "未设置"
                    auth_label = f"云端密钥（{env_state}）"
                else:
                    auth_label = "无需密钥或由本地服务处理"
                summary = "\n".join(
                    (
                        f"服务：{service_label}",
                        f"模型：{draft.model or '尚未填写'}",
                        f"连接：{auth_label}",
                        f"状态：{'启用' if draft.enabled else '停用'}",
                    )
                )
                try:
                    draft.validate()
                except ValueError as exc:
                    valid = False
                    message = f"请补齐或修正：{_friendly_model_error(exc)}"
                    if (
                        str(self.preset_combo.currentData() or "custom") == "custom"
                        and not self.base_url_edit.text().strip()
                    ):
                        message += "；可直接点选上方服务预设"
                else:
                    valid = True
                    message = "配置已准备，可以提交保存。"
            except ValueError as exc:
                yaml_text = ""
                summary = "连接信息尚未完成。"
                valid = False
                message = _friendly_model_error(exc)
            self._draft_valid = valid
            self.preview_edit.setPlainText(summary)
            self._refresh_connection_summary()
            self.yamlChanged.emit(yaml_text)
            self.validationChanged.emit(valid, message)
            self._set_status(message, error=not valid)

        def _set_status(self, message: str, *, error: bool) -> None:
            self.status_label.setText(message)
            self.status_label.setProperty("state", "error" if error else "ok")
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)
            info_bar = getattr(self, "status_info", None)
            if isinstance(info_bar, FancyInfoBar):
                severity = "error" if error else "success"
                if not error and any(token in message for token in ("正在", "请稍候", "尚未")):
                    severity = "warning"
                info_bar.setSeverity(severity)
                info_bar.setMessage(message)

        def closeEvent(self, event) -> None:  # noqa: N802
            """嵌入式卡片关闭时清除尚未应用的密码输入。"""

            self.clear_transient_api_key()
            super().closeEvent(event)

    class ModelSetupDialog(QDialog):
        """带确认按钮的模型渠道设置对话框。"""

        saveRequested = Signal(object)
        testRequested = Signal(object)
        yamlChanged = Signal(str)

        def __init__(
            self,
            draft: ModelSetupDraft | Mapping[str, Any] | None = None,
            *,
            channel_drafts: Sequence[ModelSetupDraft | Mapping[str, Any]] = (),
            on_submit: Callable[[Mapping[str, Any]], object] | None = None,
            on_test: Callable[[Mapping[str, Any]], object] | None = None,
            on_delete: Callable[[str], object] | None = None,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            # 配置向导必须和桌宠一样位于普通应用/全屏窗口之上；
            # 否则自动向导可能已创建但被前台应用遮住。
            self.setWindowFlags(Qt.Window | Qt.Tool | Qt.WindowStaysOnTopHint)
            self.setObjectName("modelSetupDialog")
            self.setWindowTitle("配置模型渠道")
            self._nominal_minimum_size = (520, 560)
            self.setMinimumSize(520, 560)
            self.resize(620, 680)
            inherited_theme = getattr(parent, "_md3_theme", DARK_MD3_THEME)
            resolved_theme = (
                inherited_theme if isinstance(inherited_theme, MD3Theme) else DARK_MD3_THEME
            )
            self._fancy_style_controller = FancyStyleController(resolved_theme, self)
            self.apply_theme(resolved_theme)
            self._auto_save_enabled = callable(on_submit)
            self._auto_save_ready = False
            self._auto_save_in_flight = False
            self._auto_save_pending = False
            self._auto_save_mode = False
            self._external_configuration_stale = False
            self._auto_save_timer = QTimer(self)
            self._auto_save_timer.setSingleShot(True)
            self._auto_save_timer.setInterval(700)
            self._auto_save_timer.timeout.connect(self._run_auto_save)
            self.card = ModelSetupCard(draft, channel_drafts=channel_drafts, parent=self)
            self.card.saveRequested.connect(self._submit_payload)
            self.card.deleteRequested.connect(self._delete_channel)
            self.card.testRequested.connect(self._test_payload)
            self.card.userChanged.connect(self._schedule_auto_save)
            self.card.channel_selector.currentIndexChanged.connect(self._cancel_channel_auto_save)
            self.card.yamlChanged.connect(self.yamlChanged)
            root = QVBoxLayout(self)
            root.setContentsMargins(12, 12, 12, 12)
            root.setSpacing(10)
            scroll = QScrollArea()
            scroll.setObjectName("modelSetupScroll")
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setWidget(self.card)
            root.addWidget(scroll, 1)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.setObjectName("modelSetupDialogButtons")
            # 测试连接属于向导主流程，不能埋在滚动区域底部；在窗口底栏
            # 保留一个始终可见的入口，避免用户误以为模型接口没有配置。
            self.test_button = QPushButton("测试连接")
            self.test_button.setObjectName("modelSetupTestButton")
            self.test_button.setIcon(fancy_icon("play", theme=self._md3_theme))
            self.test_button.setToolTip("发送一个不带工具的最小连接测试请求")
            self.test_button.clicked.connect(self.card.test_connection)
            buttons.addButton(self.test_button, QDialogButtonBox.ButtonRole.ActionRole)
            self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
            if self.save_button is not None:
                self.save_button.setObjectName("modelSetupPrimary")
                self.save_button.setIcon(fancy_icon("save", theme=self._md3_theme))
                self.save_button.setText("保存渠道")
            cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
            if cancel_button is not None:
                cancel_button.setText("取消")
            self.card.validationChanged.connect(self._on_validation_changed)
            buttons.accepted.connect(self._submit)
            buttons.rejected.connect(self.reject)
            self._dialog_buttons = buttons
            self._on_submit = on_submit
            self._on_test = on_test
            self._on_delete = on_delete
            auto_save_check = QCheckBox("自动保存渠道")
            auto_save_check.setObjectName("modelSetupAutoSave")
            auto_save_check.setAccessibleName("自动保存模型渠道")
            auto_save_check.setChecked(self._auto_save_enabled)
            auto_save_check.setEnabled(self._auto_save_enabled)
            auto_save_check.setToolTip(
                "编辑停止 700 毫秒后自动提交当前渠道，保存失败后等待下一次编辑"
            )
            auto_save_check.toggled.connect(self._set_auto_save_enabled)
            self.auto_save_check = auto_save_check
            command_area = QFrame()
            command_area.setObjectName("modelSetupCommandArea")
            command_area.setProperty("md3Role", "card-outlined")
            command_layout = QHBoxLayout(command_area)
            command_layout.setContentsMargins(8, 6, 8, 6)
            command_layout.addWidget(auto_save_check)
            command_bar = FancyCommandBar(theme=self._md3_theme)
            command_bar.setObjectName("modelSetupCommandBar")
            command_bar.addCommand(
                "刷新摘要",
                self.card._refresh_preview,
                icon="refresh",
                tooltip="重新生成脱敏连接摘要。",
            )
            self._fancy_style_controller.register(command_bar)
            command_layout.addWidget(command_bar, 1)
            root.addWidget(command_area)
            # QDialogButtonBox 保持对话框直属底栏，几何坐标与滚动区同源；
            # 低分辨率下连接测试、保存和取消始终可见。
            root.addWidget(buttons)
            self._on_validation_changed(self.card.is_valid, "")
            self._auto_save_ready = True

        def apply_theme(self, theme: MD3Theme) -> None:
            """热替换对话框及已创建渠道卡片的视觉令牌。"""

            if not isinstance(theme, MD3Theme):
                raise TypeError("theme must be an MD3Theme")
            self._md3_theme = theme
            self.setProperty("md3Role", "root")
            controller = self._fancy_style_controller
            controller.updateTheme(theme)
            controller.attach(self)
            self.setPalette(build_qt_palette(theme))
            self.setStyleSheet(
                "\n\n".join((controller.stylesheet(), _model_setup_stylesheet(theme)))
            )
            card = getattr(self, "card", None)
            apply_card_theme = getattr(card, "apply_theme", None)
            if callable(apply_card_theme):
                apply_card_theme(theme)

        def _cancel_channel_auto_save(self, _index: int) -> None:
            """切换渠道时取消旧草稿的定时提交。"""

            self._auto_save_timer.stop()
            self._auto_save_pending = False

        def _fit_to_available_screen(self) -> None:
            """在低分辨率屏幕保留底栏并让表单通过滚动区完整可达。"""

            try:
                screen = QGuiApplication.screenAt(self.frameGeometry().center())
                if screen is None:
                    screen = QGuiApplication.primaryScreen()
                if screen is None:
                    return
                area = screen.availableGeometry()
                maximum_width = max(1, int(area.width()) - 16)
                maximum_height = max(1, int(area.height()) - 16)
                minimum_width = min(self._nominal_minimum_size[0], maximum_width)
                minimum_height = min(self._nominal_minimum_size[1], maximum_height)
                self.setMinimumSize(minimum_width, minimum_height)
                width = min(max(1, int(self.width())), maximum_width)
                height = min(max(1, int(self.height())), maximum_height)
                if width != self.width() or height != self.height():
                    self.resize(width, height)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                return

        def showEvent(self, event) -> None:  # noqa: N802
            """显示前按当前屏幕重新计算最小尺寸。"""

            super().showEvent(event)
            self._fit_to_available_screen()

        def _on_validation_changed(self, valid: bool, _message: str) -> None:
            """同步提交按钮状态，避免空草稿点击后才发现必填项缺失。"""

            if self.save_button is not None:
                self.save_button.setEnabled(bool(valid))
                self.save_button.setToolTip(
                    "生成安全配置并提交保存。"
                    if valid
                    else "请先补齐服务名称、服务地址和模型名称。"
                )

        def _submit(self) -> None:
            if self._external_configuration_stale:
                self.card.set_notice("配置已从外部更新，请关闭并重新打开渠道向导。")
                return
            if self._auto_save_in_flight:
                self.card.set_submit_result(False, "正在自动保存当前渠道，请稍候。")
                return
            self._auto_save_timer.stop()
            self._auto_save_mode = False
            self.card.submit()

        def _set_auto_save_enabled(self, enabled: bool) -> None:
            """切换渠道自动保存；关闭后仍可用底部手动提交。"""

            self._auto_save_enabled = bool(enabled) and self._on_submit is not None
            if not self._auto_save_enabled:
                self._auto_save_timer.stop()
                self._auto_save_pending = False
            elif self._auto_save_ready:
                self._schedule_auto_save()

        def _schedule_auto_save(self) -> None:
            """在渠道字段发生用户编辑后启动 700ms 防抖。"""

            if not self._auto_save_ready or not self._auto_save_enabled:
                return
            if self._external_configuration_stale:
                return
            if self.card.clear_persisted_secret_requested:
                self._auto_save_timer.stop()
                self.card.set_notice("已选择清除密钥；请手动提交后生效。")
                return
            if self._auto_save_in_flight:
                self._auto_save_pending = True
                return
            if self.card.is_valid:
                self._auto_save_timer.start()

        def _run_auto_save(self) -> None:
            """提交当前渠道但保持向导打开，便于继续编辑其它渠道。"""

            if (
                not self._auto_save_enabled
                or self._auto_save_in_flight
                or not self.card.is_valid
                or self.card.clear_persisted_secret_requested
                or self._external_configuration_stale
            ):
                return
            self._auto_save_in_flight = True
            self._auto_save_mode = True
            if not self.card.submit():
                self._auto_save_in_flight = False
                self._auto_save_mode = False

        @property
        def auto_save_in_flight(self) -> bool:
            """返回是否正在等待当前渠道的保存回执。"""

            return self._auto_save_in_flight

        def mark_external_configuration_changed(self) -> None:
            """阻止基于旧快照的继续保存，要求用户重新打开向导。"""

            self._external_configuration_stale = True
            self._auto_save_timer.stop()
            self._auto_save_pending = False
            self._auto_save_enabled = False
            self.auto_save_check.blockSignals(True)
            self.auto_save_check.setChecked(False)
            self.auto_save_check.setEnabled(False)
            self.auto_save_check.blockSignals(False)
            self.card.set_notice("配置已从外部更新，请关闭并重新打开渠道向导。")

        def _finish_auto_save(self, *, success: bool) -> None:
            """收敛自动保存状态，并在期间有新编辑时重新计时。"""

            pending = self._auto_save_pending
            self._auto_save_pending = False
            self._auto_save_in_flight = False
            self._auto_save_mode = False
            if success and pending:
                self._schedule_auto_save()

        def _delete_channel(self, channel_id: str) -> None:
            """确认并删除一个已有渠道，成功后留在向导继续编辑。"""

            normalized_id = str(channel_id or "").strip()
            if not normalized_id:
                return
            self._auto_save_timer.stop()
            self._auto_save_pending = False
            if self._on_delete is None:
                self.card.set_submit_result(False, "当前宿主未接入渠道删除保存。")
                return
            confirm = QDialog(self)
            confirm.setWindowTitle("删除模型渠道")
            confirm.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
            confirm.setModal(True)
            confirm.setStyleSheet(
                "QDialog { background: #16111f; color: #faf6fb; }"
                "QLabel { color: #d6cbe0; }"
                "QDialogButtonBox QPushButton { background: #2e2440; color: #faf6fb; "
                "border: 1px solid #7c69a0; border-radius: 8px; padding: 7px 14px; }"
                "QDialogButtonBox QPushButton:hover { background: #493757; "
                "border-color: #ffb6ce; }"
                "QDialogButtonBox QPushButton#modelDeleteConfirm { background: #5a2c47; "
                "color: #ffdce2; border-color: #ff8fa0; }"
            )
            text = QLabel(f"确定删除渠道“{normalized_id}”？此操作会同步更新对话路由。")
            text.setWordWrap(True)
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
            )
            yes = buttons.button(QDialogButtonBox.StandardButton.Yes)
            no = buttons.button(QDialogButtonBox.StandardButton.No)
            if yes is not None:
                yes.setText("删除")
                yes.setObjectName("modelDeleteConfirm")
            if no is not None:
                no.setText("取消")
                no.setDefault(True)
            buttons.accepted.connect(confirm.accept)
            buttons.rejected.connect(confirm.reject)
            layout = QVBoxLayout(confirm)
            layout.setContentsMargins(20, 18, 20, 16)
            layout.setSpacing(10)
            layout.addWidget(text)
            layout.addWidget(buttons)
            if confirm.exec() != QDialog.DialogCode.Accepted:
                return
            try:
                result = self._on_delete(normalized_id)
            except Exception:
                result = False
            if isinstance(result, Mapping):
                status = str(result.get("status", result.get("state", "")) or "").strip().lower()
                success = status in {
                    "ok",
                    "available",
                    "accepted",
                    "completed",
                    "saved",
                    "success",
                    "reloaded",
                    "updated",
                }
            else:
                success = result is None or result is True
            if not success:
                self.card.set_submit_result(False, "渠道删除未完成，请检查控制台状态。")
                return
            remaining = tuple(
                draft for draft in self.card.channel_drafts if draft.channel_id != normalized_id
            )
            self.card.set_channel_drafts(remaining, current=remaining[0] if remaining else None)
            self.card.set_submit_result(True, "模型渠道已删除。")

        def _submit_payload(self, payload: Mapping[str, Any]) -> None:
            # 信号保留向导参数形状，便于轻量宿主直接调用
            # ``ConfigurationWizard.configure_channel``；宿主回调收到完整的
            # ``llm`` 草稿，便于配置中心原子合并并恢复未修改的密钥字段。
            result: object = True
            auto_mode = self._auto_save_mode
            if self._external_configuration_stale:
                self.card.set_notice("配置已从外部更新，请关闭并重新打开渠道向导。")
                if auto_mode:
                    self._finish_auto_save(success=False)
                return
            safe_payload = dict(payload)
            try:
                persist_plaintext = parse_bool(
                    safe_payload.pop("_persist_api_key", False),
                    field_name="_persist_api_key",
                    default=False,
                )
            except ValueError:
                self.card.set_submit_result(False, "密钥持久化选项无效，未执行保存。")
                if auto_mode:
                    self._finish_auto_save(success=False)
                return
            plaintext_api_key = self.card.take_pending_plain_api_key() if persist_plaintext else ""
            try:
                submitted_draft = self.card.draft()
            except (TypeError, ValueError):
                try:
                    submitted_draft = ModelSetupDraft.from_mapping(safe_payload)
                except (TypeError, ValueError):
                    submitted_draft = None
            if persist_plaintext:
                # 明文只在本方法局部变量中短暂存在；公开信号和草稿解析
                # 均使用空值，避免外部槽函数或异常文本接触明文。
                if not plaintext_api_key:
                    self.card.set_submit_result(False, "API 密钥为空，未执行文件保存。")
                    if auto_mode:
                        self._finish_auto_save(success=False)
                    return
                safe_payload["api_key"] = ""
                safe_payload["api_key_env"] = ""
                if submitted_draft is not None:
                    submitted_draft = replace(
                        submitted_draft,
                        api_key_env="",
                        has_persisted_secret=True,
                        clear_persisted_secret=False,
                    )
                if self._on_submit is None:
                    self.card.set_submit_result(False, "文件密钥保存回调未接入，未执行保存。")
                    if auto_mode:
                        self._finish_auto_save(success=False)
                    return
            if self._on_submit is not None:
                try:
                    if submitted_draft is None:
                        raise ValueError("模型渠道草稿无效")
                    if persist_plaintext:
                        callback_payload = submitted_draft.to_callback_payload()
                        callback_values = dict(callback_payload)
                        callback_values["api_key"] = plaintext_api_key
                        callback_values["_persist_api_key"] = True
                    else:
                        callback_values = submitted_draft.to_config_mapping(validate=True)
                    result = self._on_submit(callback_values)
                    if inspect.isawaitable(result):
                        closer = getattr(result, "close", None)
                        if callable(closer):
                            closer()
                        self.card.set_submit_result(
                            False, "保存回调尚未完成，请使用宿主提供的异步保存状态。"
                        )
                        if auto_mode:
                            self._finish_auto_save(success=False)
                        return
                except Exception:
                    self.card.set_submit_result(False, "保存回调失败，请检查控制台状态。")
                    if auto_mode:
                        self._finish_auto_save(success=False)
                    return
            if isinstance(result, Mapping):
                raw_status = result.get("status", result.get("state", ""))
                status = str(getattr(raw_status, "value", raw_status) or "").strip().lower()
                if status and status not in {
                    "ok",
                    "available",
                    "accepted",
                    "completed",
                    "pending",
                    "requested",
                    "saved",
                    "success",
                    "validated",
                    "updated",
                }:
                    # 结构化失败不应被当成成功；reason 可能包含密钥或内部路径，
                    # 因此只显示可行动的通用提示。
                    self.card.set_submit_result(
                        False, "保存未完成，请检查控制台状态并补齐连接信息。"
                    )
                    if auto_mode:
                        self._finish_auto_save(success=False)
                    return
            elif result is False:
                self.card.set_submit_result(False, "保存未完成，请检查控制台中的具体原因。")
                if auto_mode:
                    self._finish_auto_save(success=False)
                return
            runtime_status = ""
            if isinstance(result, Mapping):
                runtime_status = str(result.get("runtime_status", "") or "").strip().casefold()
            if isinstance(result, Mapping) and result.get("credentials_required") is True:
                message = "模型渠道已保存；请补齐密钥环境变量后重启或重新加载。"
            elif runtime_status == "pending":
                message = "模型渠道已保存，正在应用连接设置…"
            elif runtime_status == "restart_required":
                message = "模型渠道已保存；重新启动后完全生效。"
            elif runtime_status == "reloaded":
                message = "模型渠道已保存，连接设置已生效。"
            else:
                message = "模型渠道已保存。"
            if submitted_draft is not None:
                self.card.upsert_channel_draft(submitted_draft)
            self.card.set_submit_result(True, message)
            # 对外只发出脱敏载荷；真正的明文仅传给上面的受控保存回调。
            public_payload = dict(safe_payload)
            if submitted_draft is not None:
                public_payload["priority"] = submitted_draft.priority
            public_payload.pop("api_key", None)
            public_payload.pop("_persist_api_key", None)
            self.saveRequested.emit(public_payload)
            if auto_mode:
                self._finish_auto_save(success=True)
                return
            self.accept()

        def _test_payload(self, payload: Mapping[str, Any]) -> None:
            """执行可选宿主探测回调，并始终保留安全信号接入点。"""

            safe_payload = dict(payload)
            self.testRequested.emit(safe_payload)
            if self._on_test is None:
                self.card.set_test_result(False, "测试请求已发出，等待运行时连接测试。")
                return
            try:
                result = self._on_test(safe_payload)
                if inspect.isawaitable(result):
                    closer = getattr(result, "close", None)
                    if callable(closer):
                        closer()
                    self.card.set_test_result(False, "测试回调尚未完成，请稍候查看运行时状态。")
                    return
            except Exception:
                self.card.set_test_result(False, "模型连接测试暂时不可用，请检查控制台状态。")
                return
            if isinstance(result, Mapping):
                status = str(result.get("status", "") or "").strip().lower()
                message = _safe_model_result_text(
                    result.get("message", ""),
                    "模型连接测试未通过。",
                )
                if status in {"pending", "requested", "started"}:
                    self.card.set_test_pending(message)
                    return
                success = status in {"available", "ready", "ok", "completed"}
                self.card.set_test_result(success, message)
                return
            self.card.set_test_result(
                bool(result), "模型连接测试通过。" if result else "模型连接测试未通过。"
            )

        def draft(self) -> ModelSetupDraft:
            """返回当前卡片草稿。"""

            return self.card.draft()

        def payload(self) -> dict[str, Any]:
            """校验并返回回调 payload。"""

            return self.card.payload()

        def yaml_text(self) -> str:
            """返回当前脱敏 YAML 草稿。"""

            return self.card.yaml_text()

        def set_draft(self, draft: ModelSetupDraft | Mapping[str, Any]) -> None:
            """更新卡片草稿。"""

            self.card.set_draft(draft)

        def set_channel_drafts(
            self,
            values: Sequence[ModelSetupDraft | Mapping[str, Any]],
            *,
            current: ModelSetupDraft | Mapping[str, Any] | None = None,
        ) -> None:
            """刷新点击式渠道选择器。"""

            self.card.set_channel_drafts(values, current=current)

        def done(self, result: int) -> None:
            """关闭向导时擦除密码控件，避免其留在隐藏窗口中。"""

            self.card.clear_transient_api_key()
            super().done(result)


else:

    class ModelSetupCard:  # pragma: no cover
        """无 Qt 环境时的明确不可用占位。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 is not installed")

    class ModelSetupDialog:  # pragma: no cover
        """无 Qt 环境时的明确不可用占位。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 is not installed")


__all__ = [
    "ADAPTER_PROTOCOLS",
    "API_KEY_ENV_ALIASES",
    "adapter_for_protocol",
    "CHANNEL_PRESETS",
    "MODEL_PRESETS",
    "MODEL_SETUP_PRESETS",
    "ModelSetupCard",
    "ModelSetupDialog",
    "ModelSetupDraft",
    "ModelSetupPreset",
    "PROTOCOL_LABELS",
    "SUPPORTED_PROTOCOLS",
    "build_model_setup_payload",
    "build_model_test_payload",
    "build_model_setup_yaml",
    "model_setup_drafts",
    "openai_base_url_hint",
    "parse_model_setup_yaml",
    "preferred_api_key_env",
    "protocols_for_adapter",
    "pyside6_available",
]
