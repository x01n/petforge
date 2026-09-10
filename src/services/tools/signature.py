"""工具调用参数规范化和脱敏签名。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.json_values import validate_json_value

MAX_CANONICAL_ARGUMENT_BYTES = 1_048_576


def canonical_tool_arguments(arguments: Mapping[str, Any]) -> str:
    """返回稳定 JSON；拒绝无法安全跨适配器传递的参数。"""

    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be a mapping")
    try:
        validate_json_value(arguments, label="tool JSON")
        encoded = json.dumps(
            dict(arguments),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("tool arguments are not canonical JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_CANONICAL_ARGUMENT_BYTES:
        raise ValueError("tool arguments exceed the canonical JSON size limit")
    return encoded


def _digest(*parts: str) -> str:
    value = hashlib.sha256()
    for part in parts:
        encoded = part.encode("utf-8")
        value.update(len(encoded).to_bytes(8, "big"))
        value.update(encoded)
    return value.hexdigest()


@dataclass(frozen=True, slots=True)
class ToolInvocationSignature:
    """一次工具调用的精确身份；日志只能输出 ``digest``。"""

    identity: str
    call_id: str
    canonical_arguments: str

    @classmethod
    def create(
        cls,
        *,
        identity: str,
        call_id: str,
        arguments: Mapping[str, Any],
    ) -> ToolInvocationSignature:
        return cls(
            identity=str(identity),
            call_id=str(call_id),
            canonical_arguments=canonical_tool_arguments(arguments),
        )

    @property
    def digest(self) -> str:
        """包含工具身份、规范参数和调用 ID 的审计摘要。"""

        return _digest(self.identity, self.canonical_arguments, self.call_id)


__all__ = [
    "MAX_CANONICAL_ARGUMENT_BYTES",
    "ToolInvocationSignature",
    "canonical_tool_arguments",
]
