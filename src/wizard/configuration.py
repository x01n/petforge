"""面向脚本和首次启动的 YAML 配置工具。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.editor import (
    atomic_write_configuration,
    parse_editor_yaml,
    prepare_persisted_values,
    validate_editor_values,
)
from config.loader import (
    ConfigurationError,
    LoadedConfiguration,
    _deep_merge,
    configuration_source_digest,
    default_configuration_values,
    load_configuration,
    parse_bool,
    redact_secrets,
    update_path,
)
from core.adapters.direct.errors import AdapterConfigurationError
from services.model_routing import channel_from_mapping


def _wizard_defaults() -> dict[str, Any]:
    """为向导生成与当前源码资源根一致的默认值。"""

    project_root = Path(__file__).resolve().parents[2]
    resource_root = project_root / "resources"
    return default_configuration_values(
        resource_root=resource_root if resource_root.is_dir() else None,
    )


_DEFAULTS: dict[str, Any] = _wizard_defaults()
_ENVIRONMENT_REFERENCE_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _missing_api_key_references(values: object) -> bool:
    locations: list[tuple[str, tuple[object, ...]]] = []
    active: set[int] = set()

    def visit(value: object, path: tuple[object, ...]) -> None:
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                raise ConfigurationError("configuration contains recursive aliases")
            active.add(identity)
            try:
                for key, nested in value.items():
                    visit(nested, (*path, str(key)))
            finally:
                active.remove(identity)
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in active:
                raise ConfigurationError("configuration contains recursive aliases")
            active.add(identity)
            try:
                for index, nested in enumerate(value):
                    visit(nested, (*path, index))
            finally:
                active.remove(identity)
        elif isinstance(value, str):
            locations.extend(
                (match.group(1), path) for match in _ENVIRONMENT_REFERENCE_PATTERN.finditer(value)
            )

    visit(values, ())
    missing = [(name, path) for name, path in locations if name not in os.environ]
    return bool(missing) and all(
        len(path) == 4
        and path[0] == "llm"
        and path[1] == "channels"
        and isinstance(path[2], int)
        and path[3] in {"api_key", "token"}
        for _name, path in missing
    )


class ConfigurationWizard:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def create(self, *, overwrite: bool = False) -> Path:
        if self.path.exists() and not overwrite:
            raise ConfigurationError(f"configuration already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_configuration(self.path, _DEFAULTS)
        return self.path

    def load(self) -> LoadedConfiguration:
        return load_configuration(self.path, defaults=_DEFAULTS)

    def patch(self, changes: Mapping[str, Any]) -> LoadedConfiguration:
        """修改配置并在落盘前恢复原始密钥引用。"""

        if not isinstance(changes, Mapping):
            raise ConfigurationError("configuration changes must be a mapping")
        try:
            source = parse_editor_yaml(self.path.read_text(encoding="utf-8"))
        except ConfigurationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ConfigurationError("configuration file cannot be read") from exc
        values = _deep_merge(_DEFAULTS, source)
        if not isinstance(values, Mapping):
            raise ConfigurationError("configuration root must be a mapping")
        values = dict(values)
        for dotted_path, value in changes.items():
            update_path(values, str(dotted_path), value)
        try:
            values = validate_editor_values(values)
            persisted = prepare_persisted_values(source, values)
            if not isinstance(persisted, Mapping):
                raise ConfigurationError("configuration root must be a mapping")
            missing_api_key = _missing_api_key_references(persisted)
            if not missing_api_key:
                from config.loader import expand_environment_values

                expand_environment_values(persisted)
            atomic_write_configuration(self.path, persisted)
        except ConfigurationError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigurationError("configuration could not be saved safely") from exc
        try:
            return self.load()
        except ConfigurationError:
            if _missing_api_key_references(persisted):
                return LoadedConfiguration(
                    self.path,
                    dict(persisted),
                    source_digest=configuration_source_digest(self.path),
                )
            raise

    def configure_channel(
        self,
        *,
        channel_id: str,
        protocol: str,
        base_url: str,
        model: str,
        api_key_env: str = "",
        api_key: str = "",
        persist_api_key: bool = False,
        capabilities: str | list[str] | tuple[str, ...] = "streaming,tools",
        enabled: bool = True,
        priority: int = 10,
    ) -> Path:
        """新增或替换一个可用于对话的模型渠道。

        默认只把 ``api_key_env`` 环境变量引用写入 YAML，不把密钥落盘；
        ``persist_api_key=True`` 是用户明确选择的本机明文持久化模式。
        方法直接读取原始 YAML，因此环境变量尚未设置时也能完成引用配置。
        """

        persist_api_key = parse_bool(
            persist_api_key,
            field_name="persist_api_key",
            default=False,
        )
        enabled = parse_bool(enabled, field_name="enabled", default=True)
        raw_source: Mapping[str, Any] = {}
        if self.path.exists():
            try:
                raw = parse_editor_yaml(self.path.read_text(encoding="utf-8"))
            except ConfigurationError:
                raise
            except (OSError, UnicodeError) as exc:
                raise ConfigurationError("configuration file cannot be read") from exc
            raw_source = raw
            values = dict(raw)
        else:
            values = dict(_DEFAULTS)
        normalized_id = str(channel_id or "").strip()
        normalized_protocol = str(protocol or "").strip().lower()
        normalized_base = str(base_url or "").strip()
        normalized_model = str(model or "").strip()
        if not normalized_id or not normalized_base or not normalized_model:
            raise ConfigurationError("channel id, base_url, and model are required")
        try:
            normalized_priority = int(priority)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("channel priority must be an integer") from exc
        if normalized_priority < -1_000_000 or normalized_priority > 1_000_000:
            raise ConfigurationError("channel priority is out of range")
        env_name = str(api_key_env or "").strip()
        if normalized_protocol == "ollama_chat":
            # 本地 Ollama 不使用云端凭据；切换已有渠道时同步清理旧密钥。
            env_name = ""
            api_key = ""
        if env_name and re.fullmatch(r"[A-Z_][A-Z0-9_]*", env_name) is None:
            raise ConfigurationError("api_key_env must be an uppercase environment variable name")
        if env_name and api_key and not persist_api_key:
            raise ConfigurationError("api_key and api_key_env cannot both be set")
        if persist_api_key and api_key:
            env_name = ""
        if api_key and not env_name:
            # 兼容旧的程序化调用：默认密钥只进入当前进程环境，配置文件
            # 写入环境变量引用；persist_api_key 明确开启时才绕过该行为。
            secret = str(api_key)
            if len(secret) > 4096 or any(char in secret for char in "\r\n\x00"):
                raise ConfigurationError("api_key has an invalid format")
            if persist_api_key:
                # 用户明确要求持久化时由配置编辑器负责原子写入明文；不
                # 再注入环境变量，避免文件和进程状态产生两个来源。
                env_name = ""
            else:
                env_name = "MEAPET_API_KEY"
                os.environ[env_name] = secret
        if isinstance(capabilities, str):
            capability_values = [item.strip() for item in capabilities.split(",") if item.strip()]
        else:
            capability_values = [str(item).strip() for item in capabilities if str(item).strip()]
        if not capability_values:
            capability_values = ["streaming", "tools"]
        channel_values: dict[str, Any] = {
            "id": normalized_id,
            "protocol": normalized_protocol,
            "base_url": normalized_base,
            "api_key": f"${{{env_name}}}" if env_name else str(api_key or ""),
            "model": normalized_model,
            "capabilities": capability_values,
            "enabled": enabled,
            "priority": normalized_priority,
        }
        if env_name and not persist_api_key:
            channel_values["api_key_env"] = env_name
        if persist_api_key and api_key:
            # 空的环境变量元数据标记这是用户明确选择的文件持久化密钥；
            # 后续普通自动保存看到脱敏占位符时可以安全保留该明文，而
            # 新的未授权明文编辑仍会被拒绝。
            channel_values["api_key_env"] = ""
        # 使用同一领域解析器验证协议、地址和字段边界，再写入文件。
        try:
            channel_from_mapping(channel_values)
        except AdapterConfigurationError as exc:
            raise ConfigurationError(str(exc)) from exc
        llm_value = values.get("llm")
        llm = dict(llm_value) if isinstance(llm_value, Mapping) else {}
        raw_channels = llm.get("channels")
        channels = list(raw_channels) if isinstance(raw_channels, list) else []
        replaced = False
        for index, raw_channel in enumerate(channels):
            if (
                isinstance(raw_channel, Mapping)
                and str(raw_channel.get("id", "")).strip() == normalized_id
            ):
                # 更新基础字段时保留该渠道已有的 endpoint、headers、retry、
                # 请求模板和元数据；命令行未提供密钥参数也视为“保持原引用”，
                # 避免一次改模型把可用认证静默清掉。需要清除时由 GUI 显式
                # 提交空引用，或直接编辑脱敏配置。
                updated_channel = dict(raw_channel)
                previous_protocol = str(raw_channel.get("protocol", "") or "").strip().lower()
                if previous_protocol != normalized_protocol:
                    # 协议族发生变化时旧向导元数据/兼容 adapter 不能继续
                    # 指向原族；删除它让下次打开按新 protocol 精确恢复。
                    updated_channel.pop("preset", None)
                    updated_channel.pop("adapter", None)
                if (
                    normalized_protocol != "ollama_chat"
                    and not env_name
                    and not api_key
                    and raw_channel.get("api_key")
                ):
                    channel_values = {
                        key: value for key, value in channel_values.items() if key != "api_key"
                    }
                updated_channel.update(channel_values)
                if normalized_protocol == "ollama_chat":
                    updated_channel.pop("api_key", None)
                    updated_channel.pop("token", None)
                    updated_channel.pop("api_key_env", None)
                    updated_channel.pop("api_key_environment", None)
                channels[index] = updated_channel
                replaced = True
                break
        if not replaced:
            channels.append(channel_values)
        llm["channels"] = channels
        routing_value = llm.get("routing")
        routing = dict(routing_value) if isinstance(routing_value, Mapping) else {}
        dialogue = routing.get("dialogue")
        dialogue_channel = (
            str(dialogue.get("channel", dialogue.get("channel_id", "")) or "").strip()
            if isinstance(dialogue, Mapping)
            else str(dialogue or "").strip()
        )
        # 新增第一个渠道时建立默认对话路由；编辑已有渠道不再静默抢占
        # 当前主路由，避免修改 backup 渠道后改变实际服务。
        if not replaced and (not dialogue_channel or dialogue_channel == normalized_id):
            routing["dialogue"] = {
                "channel": normalized_id,
                "required_capabilities": ["streaming"],
                "fallback_channels": [],
            }
        llm["routing"] = routing
        values["llm"] = llm
        try:
            persisted = prepare_persisted_values(
                raw_source,
                values,
                allow_plaintext_secrets=bool(persist_api_key),
            )
            if not isinstance(persisted, Mapping):
                raise ConfigurationError("configuration root must be a mapping")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_configuration(
                self.path,
                persisted,
                mode=0o600 if persist_api_key else None,
            )
        except ConfigurationError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigurationError("configuration could not be saved safely") from exc
        return self.path

    def remove_channel(self, channel_id: str) -> Path:
        """从 YAML 中移除渠道并清理指向它的对话路由。"""

        if not self.path.exists():
            raise ConfigurationError(f"configuration file does not exist: {self.path}")
        try:
            raw = parse_editor_yaml(self.path.read_text(encoding="utf-8"))
        except ConfigurationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ConfigurationError("configuration file cannot be read") from exc
        if raw is None:
            values = {}
        elif isinstance(raw, Mapping):
            values = dict(raw)
        else:
            raise ConfigurationError("configuration root must be a mapping")
        llm_value = values.get("llm")
        if not isinstance(llm_value, Mapping):
            return self.path
        llm = dict(llm_value)
        normalized_id = str(channel_id or "").strip()
        channels = llm.get("channels")
        if isinstance(channels, list):
            llm["channels"] = [
                item
                for item in channels
                if not isinstance(item, Mapping) or str(item.get("id", "")).strip() != normalized_id
            ]
        routing = llm.get("routing")
        if isinstance(routing, Mapping):
            updated_routing = dict(routing)
            remaining = llm.get("channels")
            remaining_items = list(remaining) if isinstance(remaining, list) else []
            next_id = (
                str(remaining_items[0].get("id", "") or "").strip()
                if remaining_items and isinstance(remaining_items[0], Mapping)
                else ""
            )

            def replacement_id(route_values: Mapping[str, Any]) -> str:
                required_raw = route_values.get("required_capabilities", ())
                if isinstance(required_raw, str):
                    required = {
                        item.strip().lower() for item in required_raw.split(",") if item.strip()
                    }
                elif isinstance(required_raw, (list, tuple, set, frozenset)):
                    required = {
                        str(item).strip().lower() for item in required_raw if str(item).strip()
                    }
                else:
                    required = set()
                for item in remaining_items:
                    if not isinstance(item, Mapping):
                        continue
                    try:
                        parsed_channel = channel_from_mapping(item)
                    except (AdapterConfigurationError, TypeError, ValueError):
                        continue
                    if not parsed_channel.is_ready:
                        continue
                    item_id = str(item.get("id", "") or "").strip()
                    raw_capabilities = item.get("capabilities", ("streaming", "tools"))
                    if isinstance(raw_capabilities, str):
                        capabilities = {
                            value.strip().lower()
                            for value in raw_capabilities.split(",")
                            if value.strip()
                        }
                    elif isinstance(raw_capabilities, (list, tuple, set, frozenset)):
                        capabilities = {
                            str(value).strip().lower()
                            for value in raw_capabilities
                            if str(value).strip()
                        }
                    else:
                        capabilities = set()
                    if item_id and required.issubset(capabilities):
                        return item_id
                return ""

            for task, raw_route in tuple(updated_routing.items()):
                if isinstance(raw_route, Mapping):
                    route_values = dict(raw_route)
                    route_channel = str(
                        route_values.get("channel", route_values.get("channel_id", "")) or ""
                    ).strip()
                    if route_channel == normalized_id:
                        route_key = "channel" if "channel" in route_values else "channel_id"
                        replacement = replacement_id(route_values)
                        if replacement:
                            route_values[route_key] = replacement
                        else:
                            route_values.pop("channel", None)
                            route_values.pop("channel_id", None)
                            updated_routing.pop(task, None)
                            continue
                    fallback = route_values.get("fallback_channels")
                    if isinstance(fallback, (list, tuple)):
                        route_values["fallback_channels"] = [
                            str(item).strip()
                            for item in fallback
                            if str(item).strip() and str(item).strip() != normalized_id
                        ]
                    updated_routing[task] = route_values
                elif str(raw_route or "").strip() == normalized_id:
                    updated_routing[task] = next_id
            llm["routing"] = updated_routing
        values["llm"] = llm
        try:
            persisted = prepare_persisted_values(raw, values)
            if not isinstance(persisted, Mapping):
                raise ConfigurationError("configuration root must be a mapping")
            atomic_write_configuration(self.path, persisted)
        except ConfigurationError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigurationError("configuration could not be saved safely") from exc
        return self.path

    def safe_view(self) -> object:
        """读取原始 YAML 并脱敏，缺少密钥环境变量时仍可查看配置。"""

        if not self.path.is_file():
            raise ConfigurationError(f"configuration file does not exist: {self.path}")
        try:
            values = parse_editor_yaml(self.path.read_text(encoding="utf-8"))
        except ConfigurationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ConfigurationError("configuration file cannot be read") from exc
        return redact_secrets(_deep_merge(_DEFAULTS, values))
