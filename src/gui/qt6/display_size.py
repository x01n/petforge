from __future__ import annotations

from collections.abc import Mapping

DISPLAY_SIZE_PRESETS: Mapping[str, tuple[int, int]] = {
    "small": (360, 420),
    "standard": (420, 480),
    "large": (520, 620),
}

DISPLAY_SIZE_LABELS: Mapping[str, str] = {
    "small": "小",
    "standard": "标准",
    "large": "大",
}


def display_size_preset(value: object) -> tuple[str, tuple[int, int]] | None:
    """返回规范化尺寸键和窗口大小；未知值不改变窗口。"""

    key = str(value or "").strip().lower()
    size = DISPLAY_SIZE_PRESETS.get(key)
    if size is None:
        return None
    return key, size


__all__ = ["DISPLAY_SIZE_LABELS", "DISPLAY_SIZE_PRESETS", "display_size_preset"]
