from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gui.qt6.md3 import (
    DARK_MD3_THEME,
    MD3Role,
    MD3Theme,
    MD3ThemeController,
    MD3ThemeError,
    build_md3_stylesheet,
    build_md3_web_css,
    load_md3_theme,
    md3_web_tokens,
    remap_legacy_stylesheet,
    set_md3_role,
    theme_from_configuration,
)


def test_md3_theme_exposes_qss_and_web_tokens_from_one_source() -> None:
    qss = build_md3_stylesheet(DARK_MD3_THEME)
    tokens = md3_web_tokens(DARK_MD3_THEME)
    css = build_md3_web_css(DARK_MD3_THEME)

    assert f"background-color: {DARK_MD3_THEME.colors.surface}" in qss
    assert 'QPushButton[md3Role="filled"]' in qss
    assert tokens["--md3-color-primary"] == DARK_MD3_THEME.colors.primary
    assert tokens["--md3-radius-large"] == "12px"
    assert css.startswith(":root {\n")
    assert "--md3-color-on-surface:" in css


def test_default_md3_theme_uses_fluent_material_and_motion_tokens() -> None:
    tokens = md3_web_tokens(DARK_MD3_THEME)

    assert DARK_MD3_THEME.name == "MeaPet Fluent Dark"
    assert DARK_MD3_THEME.colors.surface == "#202020"
    assert DARK_MD3_THEME.colors.primary == "#60CDFF"
    assert DARK_MD3_THEME.layout.radius_small_px == 4
    assert DARK_MD3_THEME.layout.radius_large_px == 12
    assert tokens["--md3-motion-short"] == "150ms"
    assert tokens["--md3-motion-medium"] == "200ms"
    assert tokens["--md3-motion-long"] == "250ms"


def test_md3_stylesheet_reserves_borders_for_stable_focus_and_press_states() -> None:
    qss = build_md3_stylesheet(DARK_MD3_THEME)

    assert "border: 1px solid #454545;" in qss
    assert "QPushButton:focus { border: 1px solid #60CDFF; }" in qss
    assert 'QPushButton[md3Role="filled"]:pressed {' in qss
    assert "background-color: #202020" in qss


def test_md3_theme_remaps_existing_object_name_rules() -> None:
    theme = MD3Theme.from_mapping(
        {
            "mode": "dark",
            "roles": {
                "surface": "#010203",
                "primary": "#112233",
                "error": "#CC3344",
            },
        }
    )

    result = remap_legacy_stylesheet(
        "QWidget#root { background:#16111f; color:#ff9dbe; border-color:#ff8fa0; }",
        theme,
    )

    assert "#010203" in result
    assert "#112233" in result
    assert "#CC3344" in result
    assert "#16111f" not in result


def test_md3_theme_from_mapping_applies_seed_then_explicit_role_overrides() -> None:
    theme = MD3Theme.from_mapping(
        {
            "name": "User Theme",
            "mode": "dark",
            "seed": "#123456",
            "roles": {"primary": "#ABCDEF", "surface": "#101116"},
            "typography": {"body_size_px": 15},
            "layout": {"radius_large_px": 24, "density": -1},
            "motion": {"reduced_motion": True, "short_duration_ms": 0},
            "extra_stylesheet": 'QLabel[md3Role="brand"] { font-weight: 800; }',
        }
    )

    assert theme.name == "User Theme"
    assert theme.seed == "#123456"
    assert theme.colors.primary == "#ABCDEF"
    assert theme.colors.surface == "#101116"
    assert theme.typography.body_size_px == 15
    assert theme.layout.radius_large_px == 24
    assert theme.layout.density == -1
    assert theme.motion.reduced_motion is True
    assert "user theme overrides" in build_md3_stylesheet(theme)
    assert md3_web_tokens(theme)["--md3-control-height"] == "32px"
    assert theme.to_mapping()["roles"]["primary"] == "#ABCDEF"


def test_md3_seed_generates_readable_primary_roles() -> None:
    theme = MD3Theme.from_mapping({"mode": "light", "seed": "#FF0000"})

    assert theme.seed == "#FF0000"
    assert theme.colors.primary != DARK_MD3_THEME.colors.primary
    assert theme.colors.on_primary in {"#000000", "#FFFFFF"}
    assert theme.colors.on_primary_container in {"#000000", "#FFFFFF"}


def test_md3_theme_reads_exact_ui_theme_configuration_path() -> None:
    theme = theme_from_configuration(
        {
            "ui": {
                "always_on_top": True,
                "theme": {"mode": "light", "layout": {"density": 1}},
            }
        }
    )

    assert theme.mode.value == "light"
    assert theme.layout.density == 1


@pytest.mark.parametrize(
    "values, message",
    [
        ({"unknown": True}, "unsupported keys"),
        ({"roles": {"accent": "#FFFFFF"}}, "unsupported keys"),
        ({"roles": {"primary": "red"}}, "#RRGGBB"),
        ({"layout": {"density": 3}}, "between -2 and 2"),
        ({"layout": {"radius_large_px": 12.5}}, "must be an integer"),
        ({"extra_stylesheet": ["QWidget {}"]}, "must be a string"),
        ({"mode": "system"}, "light or dark"),
    ],
)
def test_md3_theme_rejects_unknown_or_invalid_tokens(
    values: dict[str, object], message: str
) -> None:
    with pytest.raises(MD3ThemeError, match=message):
        MD3Theme.from_mapping(values)


def test_md3_theme_yaml_load_is_strict_and_reloadable(tmp_path: Path) -> None:
    source = tmp_path / "theme.yaml"
    source.write_text(
        "mode: dark\n"
        "seed: '#6750A4'\n"
        "roles:\n"
        "  surface: '#121116'\n"
        "layout:\n"
        "  radius_medium_px: 18\n",
        encoding="utf-8",
    )
    theme = load_md3_theme(source)

    assert theme.colors.surface == "#121116"
    assert theme.layout.radius_medium_px == 18

    class Target:
        def __init__(self) -> None:
            self.properties: dict[str, object] = {}
            self.stylesheet = ""

        def setProperty(self, name: str, value: object) -> None:  # noqa: N802
            self.properties[name] = value

        def setStyleSheet(self, value: str) -> None:  # noqa: N802
            self.stylesheet = value

    target = Target()
    controller = MD3ThemeController()
    controller.attach(target)
    controller.reload(source)

    assert controller.revision == 1
    assert controller.theme.colors.surface == "#121116"
    assert target.properties["md3Role"] == "root"
    assert "#121116" in target.stylesheet


def test_md3_role_assignment_is_strict() -> None:
    class Widget:
        def __init__(self) -> None:
            self.properties: dict[str, object] = {}

        def setProperty(self, name: str, value: object) -> None:  # noqa: N802
            self.properties[name] = value

    widget = Widget()
    assert set_md3_role(widget, MD3Role.TONAL) is widget
    assert widget.properties == {"md3Role": "tonal"}
    with pytest.raises(MD3ThemeError, match="unsupported"):
        set_md3_role(widget, "floating")
    with pytest.raises(MD3ThemeError, match="selector"):
        build_md3_web_css(DARK_MD3_THEME, selector="body > main")
    with pytest.raises(MD3ThemeError, match="mapping"):
        MD3Theme.from_mapping([])  # type: ignore[arg-type]


def test_md3_widget_factory_creates_real_qt_controls() -> None:
    code = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.md3 import MD3Role, MD3WidgetFactory

app = QApplication.instance() or QApplication([])
factory = MD3WidgetFactory()
button = factory.button("保存", role=MD3Role.FILLED)
card = factory.card(role=MD3Role.CARD_OUTLINED)
chip = factory.chip("运行中", selected=True)
field = factory.text_field(placeholder="输入名称")
assert button.property("md3Role") == "filled"
assert card.property("md3Role") == "card-outlined"
assert chip.property("md3Role") == "chip-selected"
assert field.placeholderText() == "输入名称"
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
