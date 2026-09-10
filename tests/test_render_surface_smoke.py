from __future__ import annotations

from pathlib import Path
from runpy import run_path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage

_SCRIPT_VALUES = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "render_surface_smoke.py"),
    run_name="_render_surface_smoke_test",
)
_BACKGROUND = _SCRIPT_VALUES["_BACKGROUND"]
_image_metrics = _SCRIPT_VALUES["_image_metrics"]
_known_background_visible = _SCRIPT_VALUES["_known_background_visible"]


def test_surface_metrics_distinguish_alpha_model_from_opaque_black_surface() -> None:
    """指标必须区分透明 raw 帧和承载层的大面积黑色。"""

    raw = QImage(4, 4, QImage.Format.Format_RGBA8888)
    raw.fill(QColor(0, 0, 0, 0))
    raw.setPixelColor(1, 1, QColor(240, 180, 190, 255))
    raw_metrics = _image_metrics(raw)
    assert raw_metrics["alpha_bbox"] == [1, 1, 2, 2]
    assert raw_metrics["opaque_black_ratio"] == 0.0

    surface = QImage(4, 4, QImage.Format.Format_RGBA8888)
    surface.fill(QColor(0, 0, 0, 255))
    surface.setPixelColor(1, 1, QColor(*_BACKGROUND, 255))
    surface_metrics = _image_metrics(surface, background=_BACKGROUND)
    assert surface_metrics["opaque_black_ratio"] > 0.2
    assert surface_metrics["background_ratio"] > 0.0


def test_screen_capture_probe_distinguishes_missing_compositor() -> None:
    """已知背景不可见时应降级为不可观测，而不是伪造应用黑面。"""

    visible = QImage(4, 4, QImage.Format.Format_RGBA8888)
    visible.fill(QColor(*_BACKGROUND, 255))
    assert _known_background_visible(visible, (1, 1)) is True

    black = QImage(4, 4, QImage.Format.Format_RGBA8888)
    black.fill(QColor(0, 0, 0, 255))
    assert _known_background_visible(black, (1, 1)) is False
