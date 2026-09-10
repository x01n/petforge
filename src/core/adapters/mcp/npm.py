"""将 npm MCP 包安全转换为 stdio 命令。

这里不调用 shell，也不在后台自动安装依赖。运行时仅在用户显式配置
``transport: npm`` 时执行 ``npx --yes <package>``（或配置的 runner），
并继续使用 stdio 客户端的进程组回收和超时边界。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .stdio import StdioMCPServerConfig

_PACKAGE_RE = re.compile(
    r"^(?:@[A-Za-z0-9][A-Za-z0-9._-]{0,127}/)?"
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?:@[A-Za-z0-9][A-Za-z0-9._-]{0,127})?$"
)


def _safe_arg(value: object, field_name: str) -> str:
    text = str(value)
    if not text or len(text) > 4096 or any(char in text for char in "\r\n\x00"):
        raise ValueError(f"MCP npm {field_name} is invalid")
    return text


def _runner(value: object) -> tuple[str, ...]:
    if value is None:
        return ("npx", "--yes")
    if isinstance(value, str):
        return (_safe_arg(value.strip(), "runner"), "--yes")
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = tuple(_safe_arg(item, "runner argument") for item in value)
        if not values:
            raise ValueError("MCP npm runner is required")
        return values
    raise ValueError("MCP npm runner must be a string or list")


@dataclass(frozen=True)
class NpmMCPServerConfig:
    """npm MCP 包的规范化配置及其对应的 stdio 配置。"""

    stdio: StdioMCPServerConfig

    @property
    def name(self) -> str:
        return self.stdio.name

    @property
    def command(self) -> tuple[str, ...]:
        return self.stdio.command

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> NpmMCPServerConfig:
        if not isinstance(value, Mapping):
            raise ValueError("MCP npm server must be a mapping")
        raw_args = value.get("args", ())
        if isinstance(raw_args, str) or not isinstance(raw_args, Sequence):
            raise ValueError("MCP npm args must be a list")
        args = tuple(_safe_arg(item, "argument") for item in raw_args)
        command_value = value.get("command")
        raw_package = value.get("package", value.get("npm_package"))
        package = None if raw_package is None else _safe_arg(raw_package, "package")
        if package is not None and not _PACKAGE_RE.fullmatch(package):
            raise ValueError("MCP npm package is invalid")
        if command_value is None and package is None:
            raise ValueError("MCP npm package or command is required")
        if command_value is None:
            command = _runner(value.get("runner")) + (package,) + args
        else:
            if isinstance(command_value, str):
                command = (_safe_arg(command_value.strip(), "command argument"),)
            elif isinstance(command_value, Sequence) and not isinstance(
                command_value, (bytes, bytearray)
            ):
                command = tuple(_safe_arg(item, "command argument") for item in command_value)
            else:
                raise ValueError("MCP npm command must be a string or list")
            if not command:
                raise ValueError("MCP npm command is required")
            # A list command is treated as the complete argv, allowing users to
            # pin ``npx`` flags or invoke an already installed binary.  A
            # scalar command is an executable alias and receives the package.
            if package is not None and isinstance(command_value, str):
                command = command + (package,) + args
            elif package is None:
                command = command + args
        stdio_values = dict(value)
        stdio_values["command"] = command
        stdio_values.pop("transport", None)
        return cls(StdioMCPServerConfig.from_mapping(stdio_values))


def npm_stdio_config(value: Mapping[str, Any]) -> StdioMCPServerConfig:
    """返回 npm 配置对应的 stdio 配置，供组合根直接构造客户端。"""

    return NpmMCPServerConfig.from_mapping(value).stdio


__all__ = ["NpmMCPServerConfig", "npm_stdio_config"]
