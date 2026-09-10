"""系统命令工具的无 shell 参数边界。"""

from __future__ import annotations

import math
from collections.abc import Iterable


def normalize_command_allowlist(values: Iterable[str]) -> frozenset[str]:
    """规范命令白名单，并拒绝控制字符或空项。"""

    allowed: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            raise ValueError("command allowlist entries must be strings")
        value = item.strip()
        if not value or len(value) > 4096 or any(char in value for char in "\r\n\x00"):
            raise ValueError("command allowlist entry is invalid")
        allowed.add(value)
    return frozenset(allowed)


def normalize_command_argv(
    value: object,
    *,
    allowlist: frozenset[str],
    max_arguments: int = 16,
) -> tuple[str, ...]:
    """只接受 JSON argv 数组，不接受 shell 命令字符串。"""

    if not isinstance(value, list) or not value or len(value) > max_arguments:
        raise ValueError("command must be a non-empty argv list")
    argv: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("command arguments must be strings")
        if not item or len(item) > 4096 or any(char in item for char in "\r\n\x00"):
            raise ValueError("command argument is unsafe")
        argv.append(item)
    if not allowlist or argv[0] not in allowlist:
        raise PermissionError("command is not in the configured allowlist")
    return tuple(argv)


def normalize_command_timeout(value: object, *, maximum: float = 30.0) -> float:
    """返回有限且有界的命令超时秒数。"""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("command timeout must be a finite number")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("command timeout must be a finite number") from exc
    if not math.isfinite(timeout):
        raise ValueError("command timeout must be a finite number")
    return max(0.1, min(timeout, maximum))


__all__ = [
    "normalize_command_allowlist",
    "normalize_command_argv",
    "normalize_command_timeout",
]
