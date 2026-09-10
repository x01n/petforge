"""跨适配器共享的严格 JSON 值校验。"""

from __future__ import annotations

import math
from collections.abc import Mapping


def validate_json_value(value: object, *, label: str = "JSON") -> None:
    """递归拒绝非字符串键和非有限浮点数。"""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} values must be finite")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} object keys must be strings")
            validate_json_value(nested, label=label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            validate_json_value(nested, label=label)


__all__ = ["validate_json_value"]
