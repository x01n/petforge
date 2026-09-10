from __future__ import annotations

import copy
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml

from config.loader import (
    ConfigurationError,
    _reject_forbidden_keys,
    configuration_source_digest,
)

_SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?key|private[_-]?key|credential|token|secret|password|cookie|authorization)",
    re.IGNORECASE,
)
_REDACTION_MARKER = "***"
_ENV_REFERENCE_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}\Z")
_ENVIRONMENT_SCAN_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

# 配置中这些列表项有稳定身份字段。按索引合并会在用户删除或拖动一项后
# 把旧项的密钥合并到另一项（最明显的是 llm.channels），因此脱敏合并和
# 原文密钥恢复都必须优先按身份匹配。没有身份字段的列表仍保持原有的
# 位置语义，以免改变任意用户自定义列表的行为。
_LIST_IDENTITY_FIELDS = ("id", "task_id", "trigger_id", "name")


class ConfigurationConflictError(ConfigurationError):
    """配置文件已在当前编辑基线之后被外部修改。"""


def assert_configuration_revision(path: str | Path, expected_digest: str) -> None:
    """确认磁盘内容仍与编辑基线一致，否则拒绝覆盖外部修改。"""

    normalized = str(expected_digest or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("expected configuration digest must be a SHA-256 value")
    try:
        current = configuration_source_digest(path)
    except ConfigurationError as exc:
        raise ConfigurationConflictError(
            "configuration changed on disk; reload before saving"
        ) from exc
    if current != normalized:
        raise ConfigurationConflictError("configuration changed on disk; reload before saving")


def _ensure_acyclic(value: object) -> None:
    """拒绝 YAML 锚点形成的递归对象，避免编辑器栈溢出。"""

    active: set[int] = set()

    def visit(item: object) -> None:
        if not isinstance(item, (Mapping, list, tuple)):
            return
        identity = id(item)
        if identity in active:
            raise ConfigurationError("configuration contains recursive aliases")
        active.add(identity)
        try:
            if isinstance(item, Mapping):
                nested_values = item.values()
            else:
                nested_values = item
            for nested in nested_values:
                visit(nested)
        finally:
            active.remove(identity)

    visit(value)


def _is_secret_key(key: object) -> bool:
    normalized = str(key or "").strip().lower().replace("-", "_")
    # 变量名只是引用元数据，不是密钥本身；把 api_key_env 脱敏会使配置
    # 编辑器把有效环境变量名替换成 ***，随后无法重新保存渠道。
    if normalized in {"api_key_env", "api_key_environment"}:
        return False
    return bool(_SECRET_KEY_PATTERN.search(normalized))


def _list_identity(value: object) -> tuple[str, str] | None:
    """返回配置列表项的明确身份；重复或空身份由调用方拒绝复用。"""

    if not isinstance(value, Mapping):
        return None
    for field_name in _LIST_IDENTITY_FIELDS:
        raw_value = value.get(field_name)
        if isinstance(raw_value, (Mapping, list, tuple, set)):
            continue
        normalized = str(raw_value or "").strip()
        if normalized:
            return field_name, normalized
    return None


def _matching_list_index(
    original: list[object], edited_item: object, edited_index: int, used: set[int]
) -> int | None:
    """为列表项选择安全的原文索引，避免身份错配密钥。"""

    identity = _list_identity(edited_item)
    if identity is not None:
        matches = [
            index
            for index, original_item in enumerate(original)
            if index not in used and _list_identity(original_item) == identity
        ]
        # 身份重复时不猜测对应项；新项会作为新值处理，要求用户明确填写密钥。
        return matches[0] if len(matches) == 1 else None
    if edited_index >= len(original) or edited_index in used:
        return None
    # 编辑项没有身份时，仅在原项也没有身份时保留位置语义。
    if _list_identity(original[edited_index]) is None:
        return edited_index
    return None


def _redact_for_editor(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTION_MARKER if _is_secret_key(key) else _redact_for_editor(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_for_editor(item) for item in value]
    return copy.deepcopy(value)


def editor_snapshot(values: Mapping[str, Any]) -> dict[str, Any]:
    """返回可交给 UI 的深拷贝脱敏配置。"""

    if not isinstance(values, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    _ensure_acyclic(values)
    snapshot = _redact_for_editor(copy.deepcopy(dict(values)))
    if not isinstance(snapshot, Mapping):  # pragma: no cover - redact 保持结构
        raise ConfigurationError("configuration snapshot must be a mapping")
    return dict(snapshot)


def merge_editor_values(original: object, edited: object) -> object:
    """把 UI 脱敏副本合并回原始值。

    ``***`` 表示“保持原值”。映射中未出现在编辑副本的字段也保持原值，
    以免旧版 UI 或部分配置页保存时删除未知配置。带有稳定身份字段的列表
    按身份递归合并，避免用户重排渠道、MCP 服务、任务或触发器后错配密钥；
    没有身份字段的自定义列表仍按索引合并。
    """

    _ensure_acyclic(original)
    _ensure_acyclic(edited)
    if isinstance(original, Mapping) and isinstance(edited, Mapping):
        result: dict[str, Any] = copy.deepcopy(dict(original))
        for raw_key, edited_value in edited.items():
            key = str(raw_key)
            if _is_secret_key(key) and edited_value == _REDACTION_MARKER:
                continue
            result[key] = merge_editor_values(result.get(key), edited_value)
        return result
    if isinstance(original, list) and isinstance(edited, list):
        original_items = list(original)
        result: list[Any] = []
        used: set[int] = set()
        for index, edited_value in enumerate(edited):
            original_index = _matching_list_index(original_items, edited_value, index, used)
            if original_index is not None:
                used.add(original_index)
                result.append(merge_editor_values(original_items[original_index], edited_value))
            else:
                result.append(copy.deepcopy(edited_value))
        return result
    if edited == _REDACTION_MARKER and original not in (None, ""):
        return copy.deepcopy(original)
    return copy.deepcopy(edited)


def preserve_secret_values(source: object, edited: object) -> object:
    """从磁盘原文恢复密钥引用，避免把环境展开值写回 YAML。"""

    _ensure_acyclic(source)
    _ensure_acyclic(edited)
    if isinstance(edited, Mapping):
        source_mapping = source if isinstance(source, Mapping) else {}
        result: dict[str, Any] = copy.deepcopy(dict(edited))
        for raw_key in tuple(result):
            key = str(raw_key)
            if _is_secret_key(key):
                if key in source_mapping:
                    result[key] = copy.deepcopy(source_mapping[key])
                else:
                    result.pop(raw_key, None)
                continue
            if key in source_mapping:
                result[key] = preserve_secret_values(source_mapping[key], result[raw_key])
        return result
    if isinstance(edited, list):
        source_list = list(source) if isinstance(source, list) else []
        used: set[int] = set()
        result: list[Any] = []
        for index, item in enumerate(edited):
            source_index = _matching_list_index(source_list, item, index, used)
            if source_index is not None:
                used.add(source_index)
            result.append(
                preserve_secret_values(
                    source_list[source_index] if source_index is not None else None,
                    item,
                )
            )
        return result

    return copy.deepcopy(edited)


def prepare_persisted_values(
    source: object,
    edited: object,
    *,
    allow_plaintext_secrets: bool = False,
    _path: tuple[object, ...] = (),
) -> object:
    """准备配置落盘值，默认保留环境变量引用并拒绝明文密钥。

    ``source`` 必须是磁盘上的原始 YAML 映射。运行时配置可能已经展开了
    ``${ENV}``，因此对未修改的密钥优先恢复原始引用。默认仍要求新密钥使用
    ``${ENV_NAME}`` 引用；配置中心在用户明确开启“写入配置文件”后可传入
    ``allow_plaintext_secrets=True``，将新密钥按 YAML 明文持久化。
    """

    _ensure_acyclic(source)
    _ensure_acyclic(edited)
    if isinstance(edited, Mapping):
        source_mapping = source if isinstance(source, Mapping) else {}
        result: dict[str, Any] = copy.deepcopy(dict(edited))
        cleared_api_key = False
        for raw_key, edited_value in tuple(result.items()):
            key = str(raw_key)
            if cleared_api_key and key in {"api_key_env", "api_key_environment"}:
                result.pop(raw_key, None)
                continue
            if _is_secret_key(key):
                original = source_mapping.get(key)
                explicit_file_persistence = (
                    "api_key_env" in edited and not str(edited.get("api_key_env") or "").strip()
                )
                if edited_value == _REDACTION_MARKER:
                    if key in source_mapping:
                        # ``***`` 表示“保持磁盘原值”。原值可能是环境引用，
                        # 也可能是用户此前明确保存的本机明文；普通自动保存
                        # 不得删除或改写这两种已存在的配置。
                        if isinstance(original, str):
                            result[raw_key] = copy.deepcopy(original)
                        else:
                            raise ConfigurationError(
                                "secret fields must use environment variable references"
                            )
                    else:
                        result.pop(raw_key, None)
                elif edited_value in (None, ""):
                    env_hint = str(edited.get("api_key_env", "") or "").strip()
                    if key in {"api_key", "token"} and env_hint:
                        # ``api_key: ''`` + ``api_key_env: NAME`` 是加载器支持的
                        # 元数据-only 形式，不能误当作清除；保存时规范化为显式
                        # 环境变量引用，运行时仍会按同名变量注入内存密钥。
                        result[raw_key] = f"${{{env_hint}}}"
                    else:
                        # 空值是用户明确清除密钥的操作，不恢复旧引用。
                        result.pop(raw_key, None)
                        if key in {"api_key", "token"}:
                            cleared_api_key = True
                            result.pop("api_key_env", None)
                            result.pop("api_key_environment", None)
                elif (
                    key in source_mapping
                    and isinstance(original, str)
                    and isinstance(edited_value, str)
                    and edited_value == original
                ):
                    # 程序化渠道编辑会把未触碰的旧明文随其它字段一起带回；
                    # 值完全相同时这是“保持原凭据”，不是一次新的明文注入。
                    result[raw_key] = copy.deepcopy(original)
                elif isinstance(edited_value, str) and _ENV_REFERENCE_PATTERN.fullmatch(
                    edited_value
                ):
                    result[raw_key] = edited_value
                elif allow_plaintext_secrets and isinstance(edited_value, str):
                    if len(edited_value) > 4096 or any(char in edited_value for char in "\r\n\x00"):
                        raise ConfigurationError("secret fields must use a valid value")
                    if _ENVIRONMENT_SCAN_PATTERN.search(edited_value):
                        raise ConfigurationError(
                            "secret fields cannot contain environment placeholders"
                        )
                    # 运行时配置通常已经展开了环境引用；普通字段自动保存
                    # 不能因此把旧引用泄漏成明文。只有调用方同时显式提交
                    # 空 api_key_env（模型向导的“写入文件”选项）才替换引用。
                    source_reference = (
                        isinstance(original, str)
                        and _ENV_REFERENCE_PATTERN.fullmatch(original) is not None
                    )
                    result[raw_key] = (
                        edited_value
                        if explicit_file_persistence or not source_reference
                        else copy.deepcopy(original)
                    )
                elif (
                    key in source_mapping
                    and isinstance(original, str)
                    and _ENV_REFERENCE_PATTERN.fullmatch(original)
                ):
                    # 编辑器拿到的是运行时展开值，但用户没有改动密钥。
                    result[raw_key] = copy.deepcopy(original)
                else:
                    raise ConfigurationError(
                        "secret fields must use environment variable references"
                    )
                continue
            if key in source_mapping:
                result[raw_key] = prepare_persisted_values(
                    source_mapping[key],
                    edited_value,
                    allow_plaintext_secrets=allow_plaintext_secrets,
                    _path=(*_path, key),
                )
            elif isinstance(edited_value, (Mapping, list)):
                # 即使源配置没有该分支，也要递归检查新渠道中的密钥字段。
                result[raw_key] = prepare_persisted_values(
                    None,
                    edited_value,
                    allow_plaintext_secrets=allow_plaintext_secrets,
                    _path=(*_path, key),
                )
        is_channel_mapping = (
            len(_path) == 3
            and _path[0] == "llm"
            and _path[1] == "channels"
            and isinstance(_path[2], int)
        )
        if not cleared_api_key and is_channel_mapping:
            env_value = str(result.get("api_key_env", "") or "").strip()
            credential_key = (
                "api_key" if "api_key" in result else "token" if "token" in result else ""
            )
            api_value = result.get(credential_key) if credential_key else None
            if env_value:
                if re.fullmatch(r"[A-Z_][A-Z0-9_]*", env_value) is None:
                    raise ConfigurationError(
                        "api_key_env must be an uppercase environment variable name"
                    )
                expected_reference = f"${{{env_value}}}"
                if api_value in (None, "", _REDACTION_MARKER):
                    result[credential_key or "api_key"] = expected_reference
                elif isinstance(api_value, str):
                    reference_match = _ENV_REFERENCE_PATTERN.fullmatch(api_value)
                    if reference_match is not None and reference_match.group(1) != env_value:
                        edited_api_value = (
                            edited.get(credential_key)
                            if credential_key and isinstance(edited, Mapping)
                            else None
                        )
                        if (
                            isinstance(edited_api_value, str)
                            and _ENV_REFERENCE_PATTERN.fullmatch(edited_api_value)
                            and edited_api_value != source_mapping.get(credential_key)
                        ):
                            result["api_key_env"] = reference_match.group(1)
                        else:
                            result[credential_key or "api_key"] = expected_reference
                    elif reference_match is None:
                        # 非空 api_key_env 明确表示切换到环境引用；不保留
                        # “明文密钥 + 环境变量名”的混合状态。
                        result[credential_key or "api_key"] = expected_reference
        return result
    if isinstance(edited, list):
        source_list = list(source) if isinstance(source, list) else []
        result: list[Any] = []
        used: set[int] = set()
        for index, item in enumerate(edited):
            source_index = _matching_list_index(source_list, item, index, used)
            if source_index is not None:
                used.add(source_index)
            result.append(
                prepare_persisted_values(
                    source_list[source_index] if source_index is not None else None,
                    item,
                    allow_plaintext_secrets=allow_plaintext_secrets,
                    _path=(*_path, index),
                )
            )
        return result
    return copy.deepcopy(edited)


def parse_editor_yaml(text: str) -> dict[str, Any]:
    """解析编辑器中的 YAML 映射并检查危险键。"""

    try:
        raw = yaml.safe_load(str(text))
    except yaml.YAMLError as exc:
        raise ConfigurationError("configuration YAML is invalid") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigurationError("configuration root must be a mapping")
    values = {str(key): value for key, value in raw.items()}
    try:
        _ensure_acyclic(values)
        _reject_forbidden_keys(values)
    except RecursionError as exc:
        raise ConfigurationError("configuration contains recursive aliases") from exc
    return values


def validate_editor_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """执行不产生外部副作用的配置结构校验。"""

    if not isinstance(values, Mapping):
        raise ConfigurationError("configuration must be a mapping")
    normalized = {str(key): copy.deepcopy(value) for key, value in values.items()}
    try:
        _ensure_acyclic(normalized)
        _reject_forbidden_keys(normalized)
    except RecursionError as exc:
        raise ConfigurationError("configuration contains recursive aliases") from exc

    def reject_non_finite(value: object, path: str = "configuration") -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ConfigurationError(f"{path} contains a non-finite number")
        if isinstance(value, Mapping):
            for key, nested in value.items():
                reject_non_finite(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                reject_non_finite(nested, f"{path}[{index}]")

    reject_non_finite(normalized)
    try:
        yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False)
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise ConfigurationError("configuration contains values that cannot be serialized") from exc
    return normalized


def atomic_write_configuration(
    path: str | Path,
    values: Mapping[str, Any],
    *,
    mode: int | None = None,
    expected_digest: str | None = None,
    expect_missing: bool = False,
) -> Path:
    """将配置原子替换到目标路径，并可拒绝覆盖外部并发修改。"""

    normalized = validate_editor_values(values)
    target = Path(path).expanduser().resolve()
    if expected_digest is not None and expect_missing:
        raise ValueError("expected_digest and expect_missing cannot be used together")
    target.parent.mkdir(parents=True, exist_ok=True)

    def contains_secret_field(value: object) -> bool:
        if isinstance(value, Mapping):
            if any(_is_secret_key(key) for key in value):
                return True
            return any(contains_secret_field(item) for item in value.values())
        if isinstance(value, list):
            return any(contains_secret_field(item) for item in value)
        if isinstance(value, tuple):
            return any(contains_secret_field(item) for item in value)
        return False

    if contains_secret_field(normalized):
        mode = 0o600
    if mode is None:
        try:
            mode = stat.S_IMODE(target.stat().st_mode)
        except OSError:
            mode = 0o600
    mode = int(mode) & 0o777
    payload = yaml.safe_dump(normalized, allow_unicode=True, sort_keys=False)
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        if expect_missing and target.exists():
            raise ConfigurationConflictError("configuration changed on disk; reload before saving")
        if expected_digest is not None:
            assert_configuration_revision(target, expected_digest)
        os.replace(temporary, target)
        temporary = None
        try:
            parent_descriptor = os.open(target.parent, os.O_RDONLY)
        except OSError:
            parent_descriptor = None
        if parent_descriptor is not None:
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass
    return target


class ConfigurationEditSession:
    """将脱敏编辑、验证和原子保存组织成可测试的会话。"""

    def __init__(
        self,
        path: str | Path,
        values: Mapping[str, Any],
        *,
        validator: Callable[[Mapping[str, Any]], object] | None = None,
    ) -> None:
        if not isinstance(values, Mapping):
            raise ConfigurationError("configuration must be a mapping")
        self.path = Path(path).expanduser().resolve()
        self.values: dict[str, Any] = copy.deepcopy(dict(values))
        self.validator = validator
        self._source_existed = self.path.is_file()
        self.source_digest = (
            configuration_source_digest(self.path) if self._source_existed else None
        )

    @property
    def display_values(self) -> dict[str, Any]:
        return editor_snapshot(self.values)

    @property
    def expects_missing_file(self) -> bool:
        """返回编辑基线是否明确来自尚不存在的目标文件。"""

        return not self._source_existed

    def save(self, edited: Mapping[str, Any], *, confirmed: bool = False) -> Path:
        if not confirmed:
            raise ConfigurationError(
                "explicit confirmation is required before saving configuration"
            )
        merged = merge_editor_values(self.values, edited)
        if not isinstance(merged, Mapping):  # pragma: no cover - 根映射已保证
            raise ConfigurationError("configuration must be a mapping")
        normalized = validate_editor_values(merged)
        if self.validator is not None:
            self.validator(normalized)
        target = atomic_write_configuration(
            self.path,
            normalized,
            expected_digest=self.source_digest,
            expect_missing=not self._source_existed,
        )
        self.values = normalized
        self._source_existed = True
        self.source_digest = configuration_source_digest(target)
        return target


__all__ = [
    "ConfigurationConflictError",
    "ConfigurationEditSession",
    "assert_configuration_revision",
    "atomic_write_configuration",
    "editor_snapshot",
    "merge_editor_values",
    "preserve_secret_values",
    "prepare_persisted_values",
    "parse_editor_yaml",
    "validate_editor_values",
]
