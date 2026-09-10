"""工具身份注册表和按分组的模型公开集合。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from threading import RLock

from .types import ToolSpec

_IDENTITY = re.compile(r"^[a-z][a-z0-9_.-]*:[a-z][a-z0-9_.-]*$")


class ToolRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not _IDENTITY.fullmatch(spec.identity):
            raise ValueError("tool identity must use lowercase namespace:name")
        with self._lock:
            if spec.identity in self._tools:
                raise ValueError(f"duplicate tool identity: {spec.identity}")
            self._tools[spec.identity] = spec

    def register_many(self, specs: Iterable[ToolSpec]) -> None:
        for spec in specs:
            self.register(spec)

    def sync_owned(
        self,
        previous: Mapping[str, ToolSpec],
        specs: Iterable[ToolSpec],
    ) -> None:
        """在一个临界区内同步一组有归属的工具。

        ``previous`` 保存调用方上一次登记的对象；若其中任一对象已被外部
        替换，整个同步会拒绝执行。所有校验完成后再替换内部字典，避免刷新
        过程中出现只删除了一部分工具的半态。
        """

        previous_map = dict(previous)
        current_map: dict[str, ToolSpec] = {}
        for identity, old_spec in previous_map.items():
            if not isinstance(old_spec, ToolSpec) or old_spec.identity != str(identity):
                raise ValueError(f"invalid owned tool binding: {identity}")
        for spec in specs:
            if not isinstance(spec, ToolSpec):
                raise TypeError("specs must contain ToolSpec values")
            if not _IDENTITY.fullmatch(spec.identity):
                raise ValueError("tool identity must use lowercase namespace:name")
            if spec.identity in current_map:
                raise ValueError(f"duplicate tool identity: {spec.identity}")
            current_map[spec.identity] = spec

        with self._lock:
            for identity, expected in previous_map.items():
                if self._tools.get(identity) is not expected:
                    raise ValueError(f"tool identity is owned by another registration: {identity}")
            for identity in current_map:
                if identity not in previous_map and identity in self._tools:
                    raise ValueError(f"duplicate tool identity: {identity}")
            next_tools = dict(self._tools)
            for identity in previous_map:
                next_tools.pop(identity, None)
            next_tools.update(current_map)
            self._tools = next_tools

    def replace(self, spec: ToolSpec, *, expected: ToolSpec | None = None) -> None:
        """原子替换一个已登记工具；``expected`` 用于防止覆盖外部注册。"""

        if not _IDENTITY.fullmatch(spec.identity):
            raise ValueError("tool identity must use lowercase namespace:name")
        with self._lock:
            current = self._tools.get(spec.identity)
            if current is None:
                raise KeyError(f"unknown tool identity: {spec.identity}")
            if expected is not None and current is not expected:
                raise ValueError(f"tool identity is owned by another registration: {spec.identity}")
            self._tools[spec.identity] = spec

    def unregister(self, identity: str, *, expected: ToolSpec | None = None) -> bool:
        """移除工具；提供旧对象时只允许其原注册者执行移除。"""

        key = str(identity or "").strip()
        with self._lock:
            current = self._tools.get(key)
            if current is None or (expected is not None and current is not expected):
                return False
            del self._tools[key]
            return True

    def get(self, identity: str) -> ToolSpec | None:
        with self._lock:
            return self._tools.get(str(identity or "").strip())

    def require(self, identity: str) -> ToolSpec:
        spec = self.get(identity)
        if spec is None:
            raise KeyError(f"unknown tool identity: {identity}")
        return spec

    def identities(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._tools))

    def visible(self, groups: Iterable[str] | None = None) -> tuple[ToolSpec, ...]:
        allowed_groups = None if groups is None else {str(group).strip() for group in groups}
        with self._lock:
            return tuple(
                spec
                for spec in sorted(self._tools.values(), key=lambda item: item.identity)
                if spec.public and (allowed_groups is None or spec.group in allowed_groups)
            )

    def schemas(self, groups: Iterable[str] | None = None) -> tuple[dict, ...]:
        return tuple(spec.visible_schema() for spec in self.visible(groups))
