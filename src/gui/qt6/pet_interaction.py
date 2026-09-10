from __future__ import annotations

import math
from collections.abc import Mapping

PET_CLICK_ZONES: tuple[str, ...] = ("upper", "lower_left", "lower_right")
PET_CLICK_PARTS: tuple[str, ...] = ("head", "body", "lower_left", "lower_right")
# Qt 当前常见双击间隔为 400ms；略留余量，确保页面单击不会先于第二击提交。
PET_CLICK_DEBOUNCE_MS = 450


def classify_pet_click(
    x: object,
    y: object,
    width: object,
    height: object,
) -> str | None:
    """按旧项目的画布分区规则返回点击区域。

    ``upper`` 对应画布上半区；下半区再按水平中线拆为
    ``lower_left``/``lower_right``。越界、非有限数值或无效画布不会触发
    互动回调。
    """

    try:
        point_x = float(x)
        point_y = float(y)
        canvas_width = float(width)
        canvas_height = float(height)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (point_x, point_y, canvas_width, canvas_height)):
        return None
    if canvas_width <= 0 or canvas_height <= 0:
        return None
    if point_x < 0 or point_y < 0 or point_x >= canvas_width or point_y >= canvas_height:
        return None
    if point_y < canvas_height / 2.0:
        return "upper"
    return "lower_left" if point_x < canvas_width / 2.0 else "lower_right"


def classify_pet_part(
    x: object,
    y: object,
    width: object,
    height: object,
) -> str | None:
    """把通用画布坐标映射为用户可理解的身体部位。

    这是宿主侧的保守回退，不能替代 Web Live2D 的纹理 alpha 命中；它只在
    原生 OpenGL/精灵路径没有模型部件索引时使用。头部阈值落在画布上方约三分
    之一，腿部沿用左右分区，其余上半区归为身体，避免把透明画布误报成部位。
    """

    zone = classify_pet_click(x, y, width, height)
    if zone is None:
        return None
    try:
        normalized_y = float(y) / float(height)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if zone == "upper":
        return "head" if normalized_y < 0.34 else "body"
    return zone


def make_pet_click_payload(
    payload: Mapping[str, object],
    *,
    width: object,
    height: object,
) -> dict[str, object] | None:
    """校验页面/Qt点击载荷并补充稳定的分区字段。"""

    zone = classify_pet_click(payload.get("x"), payload.get("y"), width, height)
    if zone is None:
        return None
    try:
        x = int(round(float(payload["x"])))
        y = int(round(float(payload["y"])))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    result = dict(payload)
    result.update({"type": "click", "zone": zone, "x": x, "y": y})
    return result


__all__ = [
    "PET_CLICK_DEBOUNCE_MS",
    "PET_CLICK_PARTS",
    "PET_CLICK_ZONES",
    "classify_pet_click",
    "classify_pet_part",
    "make_pet_click_payload",
]
