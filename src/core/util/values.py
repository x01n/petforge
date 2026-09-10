"""稳定、严格且不记录输入内容的值规范化函数。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def as_mapping(value: object) -> Mapping[str, Any]:
    """返回映射或共享空映射语义。"""

    return value if isinstance(value, Mapping) else {}


def as_sequence(value: object) -> Sequence[object]:
    """返回非文本序列；字符串、字节和映射统一视为空。"""

    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Sequence):
        return ()
    return value


def finite_float(value: object, default: float = 0.0) -> float:
    """转换有限浮点数，失败时返回明确默认值。"""

    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def safe_int(value: object, default: int = 0) -> int:
    """转换整数，拒绝布尔值并在失败时返回默认值。"""

    if isinstance(value, bool):
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)
