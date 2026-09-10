from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_UNSAFE_FONT_CHARACTERS = frozenset("{};\n\r")


class MD3ThemeError(ValueError):
    """主题结构、令牌或组件角色不符合公开契约。"""


class MD3ThemeMode(StrEnum):
    """内置主题基线。"""

    LIGHT = "light"
    DARK = "dark"


class MD3Role(StrEnum):
    """可供现有 QWidget 通过动态属性复用的组件角色。"""

    ROOT = "root"
    CARD = "card"
    CARD_OUTLINED = "card-outlined"
    CARD_ELEVATED = "card-elevated"
    FILLED = "filled"
    TONAL = "tonal"
    OUTLINED = "outlined"
    TEXT = "text"
    DANGER = "danger"
    ICON = "icon"
    CHIP = "chip"
    CHIP_SELECTED = "chip-selected"
    TITLE = "title"
    MUTED = "muted"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


MD3_THEME_KEYS: tuple[str, ...] = (
    "name",
    "mode",
    "seed",
    "roles",
    "typography",
    "layout",
    "motion",
    "extra_stylesheet",
)
MD3_COMPONENT_ROLES: tuple[str, ...] = tuple(item.value for item in MD3Role)


def _validate_exact_keys(values: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(str(key) for key in values if str(key) not in allowed)
    if unknown:
        raise MD3ThemeError(f"{section} contains unsupported keys: {', '.join(unknown)}")


def _color(name: str, value: object) -> str:
    normalized = str(value).strip().upper()
    if not _HEX_COLOR.fullmatch(normalized):
        raise MD3ThemeError(f"{name} must use #RRGGBB")
    return normalized


def _text(name: str, value: object, *, allow_empty: bool = False) -> str:
    normalized = str(value).strip()
    if not normalized and not allow_empty:
        raise MD3ThemeError(f"{name} must not be empty")
    if any(item in normalized for item in _UNSAFE_FONT_CHARACTERS):
        raise MD3ThemeError(f"{name} contains unsupported control characters")
    return normalized


def _integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise MD3ThemeError(f"{name} must be an integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        normalized = int(value.strip())
    else:
        raise MD3ThemeError(f"{name} must be an integer")
    if normalized < minimum or normalized > maximum:
        raise MD3ThemeError(f"{name} must be between {minimum} and {maximum}")
    return normalized


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise MD3ThemeError(f"{name} must be a boolean")
    return value


def _section(values: object, name: str) -> Mapping[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise MD3ThemeError(f"{name} must be a mapping")
    return values


def _replace_tokens(
    source: object,
    values: object,
    section: str,
    converter: Any,
) -> Any:
    mapping = _section(values, section)
    allowed = {item.name for item in fields(source)}
    _validate_exact_keys(mapping, allowed, section)
    updates = {str(key): converter(f"{section}.{key}", value) for key, value in mapping.items()}
    return replace(source, **updates)


@dataclass(frozen=True, slots=True)
class MD3ColorScheme:
    """MD3 语义颜色；每个令牌均可由用户配置覆盖。"""

    primary: str
    on_primary: str
    primary_container: str
    on_primary_container: str
    secondary: str
    on_secondary: str
    secondary_container: str
    on_secondary_container: str
    tertiary: str
    on_tertiary: str
    error: str
    on_error: str
    error_container: str
    on_error_container: str
    surface: str
    on_surface: str
    surface_variant: str
    on_surface_variant: str
    surface_container_low: str
    surface_container: str
    surface_container_high: str
    outline: str
    outline_variant: str
    inverse_surface: str
    inverse_on_surface: str
    success: str
    on_success: str
    warning: str
    on_warning: str

    def with_overrides(self, values: object) -> MD3ColorScheme:
        """应用严格校验的颜色覆盖。"""

        return _replace_tokens(self, values, "colors", _color)


@dataclass(frozen=True, slots=True)
class MD3Typography:
    """Qt 可直接应用的字体令牌。"""

    font_family: str = "Segoe UI Variable"
    monospace_family: str = "Cascadia Mono"
    body_size_px: int = 14
    label_size_px: int = 13
    title_size_px: int = 22
    headline_size_px: int = 28

    def with_overrides(self, values: object) -> MD3Typography:
        mapping = _section(values, "typography")
        allowed = {item.name for item in fields(self)}
        _validate_exact_keys(mapping, allowed, "typography")
        updates: dict[str, object] = {}
        for key, value in mapping.items():
            name = str(key)
            if name.endswith("_family"):
                updates[name] = _text(f"typography.{name}", value)
            else:
                updates[name] = _integer(f"typography.{name}", value, minimum=10, maximum=72)
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class MD3Layout:
    """组件尺寸、圆角与间距令牌。"""

    radius_small_px: int = 4
    radius_medium_px: int = 8
    radius_large_px: int = 12
    radius_extra_large_px: int = 16
    control_height_px: int = 36
    compact_height_px: int = 32
    horizontal_padding_px: int = 12
    vertical_padding_px: int = 7
    spacing_px: int = 8
    icon_size_px: int = 20
    focus_width_px: int = 1
    density: int = 0

    def with_overrides(self, values: object) -> MD3Layout:
        mapping = _section(values, "layout")
        allowed = {item.name for item in fields(self)}
        _validate_exact_keys(mapping, allowed, "layout")
        updates: dict[str, int] = {}
        for key, value in mapping.items():
            name = str(key)
            if name == "density":
                updates[name] = _integer(f"layout.{name}", value, minimum=-2, maximum=2)
            else:
                updates[name] = _integer(f"layout.{name}", value, minimum=0, maximum=256)
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class MD3Motion:
    """供宿主动画使用的时长与降低动态效果令牌。"""

    short_duration_ms: int = 150
    medium_duration_ms: int = 200
    long_duration_ms: int = 250
    reduced_motion: bool = False

    def with_overrides(self, values: object) -> MD3Motion:
        mapping = _section(values, "motion")
        allowed = {item.name for item in fields(self)}
        _validate_exact_keys(mapping, allowed, "motion")
        updates: dict[str, object] = {}
        for key, value in mapping.items():
            name = str(key)
            if name == "reduced_motion":
                updates[name] = _boolean(f"motion.{name}", value)
            else:
                updates[name] = _integer(f"motion.{name}", value, minimum=0, maximum=5000)
        return replace(self, **updates)


@dataclass(frozen=True, slots=True)
class MD3Theme:
    """一个完整且可序列化的 MD3 QWidget 主题。"""

    name: str
    mode: MD3ThemeMode
    colors: MD3ColorScheme
    seed: str | None = None
    typography: MD3Typography = MD3Typography()
    layout: MD3Layout = MD3Layout()
    motion: MD3Motion = MD3Motion()
    extra_stylesheet: str = ""

    def with_overrides(self, values: Mapping[str, Any]) -> MD3Theme:
        """在当前主题上覆盖精确公开的令牌，不接受未知键。"""

        if not isinstance(values, Mapping):
            raise MD3ThemeError("theme must be a mapping")
        _validate_exact_keys(values, set(MD3_THEME_KEYS), "theme")
        try:
            mode = MD3ThemeMode(str(values.get("mode", self.mode.value)).strip().lower())
        except ValueError as exc:
            raise MD3ThemeError("theme.mode must be light or dark") from exc
        if mode is not self.mode:
            raise MD3ThemeError("theme.mode cannot change through with_overrides")
        name = _text("theme.name", values.get("name", self.name))
        raw_extra = values.get("extra_stylesheet", self.extra_stylesheet)
        if raw_extra is not None and not isinstance(raw_extra, str):
            raise MD3ThemeError("theme.extra_stylesheet must be a string")
        extra = str(raw_extra or "").strip()
        updated = self
        if "seed" in values:
            raw_seed = values.get("seed")
            seed = (
                None
                if raw_seed is None or str(raw_seed).strip() == ""
                else _color("theme.seed", raw_seed)
            )
            if seed is not None:
                updated = _apply_seed(self, seed)
            else:
                updated = replace(self, seed=None)
        return replace(
            updated,
            name=name,
            colors=updated.colors.with_overrides(values.get("roles")),
            typography=updated.typography.with_overrides(values.get("typography")),
            layout=updated.layout.with_overrides(values.get("layout")),
            motion=updated.motion.with_overrides(values.get("motion")),
            extra_stylesheet=extra,
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> MD3Theme:
        """从 ``ui.theme`` 子映射构建完整主题。"""

        return theme_from_mapping(values)

    def to_mapping(self) -> dict[str, Any]:
        """返回可写入 YAML 的完整主题值。"""

        return {
            "name": self.name,
            "mode": self.mode.value,
            "seed": self.seed,
            "roles": asdict(self.colors),
            "typography": asdict(self.typography),
            "layout": asdict(self.layout),
            "motion": asdict(self.motion),
            "extra_stylesheet": self.extra_stylesheet,
        }


LIGHT_MD3_THEME = MD3Theme(
    name="MeaPet Fluent Light",
    mode=MD3ThemeMode.LIGHT,
    colors=MD3ColorScheme(
        primary="#005FB8",
        on_primary="#FFFFFF",
        primary_container="#D6EFFF",
        on_primary_container="#00315C",
        secondary="#5D5A58",
        on_secondary="#FFFFFF",
        secondary_container="#E8E8E8",
        on_secondary_container="#1A1A1A",
        tertiary="#0067C0",
        on_tertiary="#FFFFFF",
        error="#C42B1C",
        on_error="#FFFFFF",
        error_container="#FDE7E9",
        on_error_container="#5A0B00",
        surface="#F3F3F3",
        on_surface="#1A1A1A",
        surface_variant="#F9F9F9",
        on_surface_variant="#5D5A58",
        surface_container_low="#FAFAFA",
        surface_container="#FFFFFF",
        surface_container_high="#F0F0F0",
        outline="#8A8886",
        outline_variant="#E0E0E0",
        inverse_surface="#2C2C2C",
        inverse_on_surface="#FFFFFF",
        success="#0F7B0F",
        on_success="#FFFFFF",
        warning="#8A5A00",
        on_warning="#1A1A1A",
    ),
)

DARK_MD3_THEME = MD3Theme(
    name="MeaPet Fluent Dark",
    mode=MD3ThemeMode.DARK,
    colors=MD3ColorScheme(
        primary="#60CDFF",
        on_primary="#003545",
        primary_container="#004C6A",
        on_primary_container="#C5EFFF",
        secondary="#D0CFCF",
        on_secondary="#2C2C2C",
        secondary_container="#3A3A3A",
        on_secondary_container="#FFFFFF",
        tertiary="#C3E7FF",
        on_tertiary="#003545",
        error="#FF99A4",
        on_error="#5A0B00",
        error_container="#6E1B16",
        on_error_container="#FFDAD6",
        surface="#202020",
        on_surface="#FFFFFF",
        surface_variant="#393939",
        on_surface_variant="#D0D0D0",
        surface_container_low="#191919",
        surface_container="#272727",
        surface_container_high="#323232",
        outline="#9A9A9A",
        outline_variant="#454545",
        inverse_surface="#F3F3F3",
        inverse_on_surface="#1A1A1A",
        success="#6CCB5F",
        on_success="#0B3D09",
        warning="#FCE100",
        on_warning="#433600",
    ),
)


def _rgba(color: str, alpha: str) -> str:
    red, green, blue = _hex_rgb(color)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def remap_legacy_stylesheet(stylesheet: str, theme: MD3Theme) -> str:
    """把现有 objectName 规则中的旧颜色映射到 MD3 语义令牌。

    控制台已有大量稳定 objectName、状态属性和布局测试。该适配层只替换
    视觉令牌，不复制控件逻辑，也让旧样式在主题热切换时完整继承新配色。
    """

    if not isinstance(stylesheet, str):
        raise TypeError("stylesheet must be a string")
    if not isinstance(theme, MD3Theme):
        raise TypeError("theme must be an MD3Theme")
    c = theme.colors
    replacements = {
        "#120d1a": c.surface,
        "#16111f": c.surface,
        "#100c18": c.surface_container_low,
        "#21182d": c.surface_container,
        "#221a2e": c.surface_container,
        "#281f38": c.surface_container,
        "#292331": c.surface_container,
        "#2e2440": c.surface_container,
        "#3d3154": c.surface_container_high,
        "#3b2c50": c.secondary_container,
        "#46385c": c.outline_variant,
        "#493757": c.secondary_container,
        "#594d68": c.outline,
        "#5a2c47": c.error_container,
        "#62456f": c.secondary_container,
        "#674563": c.primary_container,
        "#7c69a0": c.outline,
        "#8e70aa": c.outline,
        "#8f829f": c.outline,
        "#9f92a9": c.on_surface_variant,
        "#a77db4": c.secondary,
        "#a79bb8": c.on_surface_variant,
        "#a85d78": c.tertiary,
        "#b7a6c7": c.on_surface_variant,
        "#b7a6ff": c.secondary,
        "#c8bbd4": c.on_surface_variant,
        "#cdb8ff": c.secondary,
        "#d6cbe0": c.on_surface_variant,
        "#e8dff0": c.on_surface,
        "#faf6fb": c.on_surface,
        "#feedba": c.warning,
        "#fff1f5": c.on_surface,
        "#ff8fa0": c.error,
        "#ff9dbe": c.primary,
        "#ffb6c2": c.primary,
        "#ffb6ce": c.primary,
        "#ffc48f": c.tertiary,
        "#ffcf91": c.tertiary,
        "#ffd3ac": c.tertiary,
        "#ffd3e0": c.on_primary_container,
        "#ffd37a": c.warning,
        "#ffdce2": c.on_error_container,
        "#ffe8f0": c.on_primary_container,
        "#2b0f1c": c.on_primary,
        "#6fe0b4": c.success,
    }
    result = stylesheet
    for source, target in replacements.items():
        result = re.sub(re.escape(source), target, result, flags=re.IGNORECASE)
    alpha_replacements = {
        "rgba(22, 17, 31, 185)": _rgba(c.surface, "185"),
        "rgba(16, 12, 24, 150)": _rgba(c.surface_container_low, "150"),
        "rgba(16, 12, 24, 180)": _rgba(c.surface_container_low, "180"),
        "rgba(255, 157, 190, 24)": _rgba(c.primary, "24"),
        "rgba(255, 157, 190, 52)": _rgba(c.primary, "52"),
        "rgba(255, 157, 190, 105)": _rgba(c.primary, "105"),
    }
    for source, target in alpha_replacements.items():
        result = result.replace(source, target)
    return result


def _hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def _mix_color(first: str, second: str, second_weight: float) -> str:
    left = _hex_rgb(first)
    right = _hex_rgb(second)
    amount = max(0.0, min(1.0, float(second_weight)))
    channels = tuple(round(a * (1.0 - amount) + b * amount) for a, b in zip(left, right))
    return "#" + "".join(f"{item:02X}" for item in channels)


def _relative_luminance(value: str) -> float:
    channels = []
    for item in _hex_rgb(value):
        normalized = item / 255.0
        channels.append(
            normalized / 12.92 if normalized <= 0.04045 else ((normalized + 0.055) / 1.055) ** 2.4
        )
    return channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722


def _contrasting_text(background: str) -> str:
    luminance = _relative_luminance(background)
    white_ratio = 1.05 / (luminance + 0.05)
    black_ratio = (luminance + 0.05) / 0.05
    return "#FFFFFF" if white_ratio >= black_ratio else "#000000"


def _apply_seed(theme: MD3Theme, seed: str) -> MD3Theme:
    """把种子色映射到主色角色；显式 ``roles`` 覆盖随后生效。"""

    c = theme.colors
    if theme.mode is MD3ThemeMode.DARK:
        primary = _mix_color(seed, "#FFFFFF", 0.32)
        container = _mix_color(seed, c.surface, 0.45)
    else:
        primary = _mix_color(seed, "#000000", 0.18)
        container = _mix_color(seed, "#FFFFFF", 0.76)
    colors = replace(
        c,
        primary=primary,
        on_primary=_contrasting_text(primary),
        primary_container=container,
        on_primary_container=_contrasting_text(container),
    )
    return replace(theme, colors=colors, seed=seed)


def md3_theme(
    mode: str | MD3ThemeMode = MD3ThemeMode.DARK,
    overrides: Mapping[str, Any] | None = None,
) -> MD3Theme:
    """取得内置基线，并应用可选的完整用户覆盖。"""

    try:
        normalized = MD3ThemeMode(str(mode).strip().lower())
    except ValueError as exc:
        raise MD3ThemeError("theme.mode must be light or dark") from exc
    base = LIGHT_MD3_THEME if normalized is MD3ThemeMode.LIGHT else DARK_MD3_THEME
    resolved_overrides = {} if overrides is None else overrides
    return base.with_overrides(resolved_overrides)


def theme_from_mapping(values: Mapping[str, Any] | None) -> MD3Theme:
    """从精确的主题映射构建主题；未提供 ``mode`` 时使用暗色基线。"""

    mapping = {} if values is None else values
    if not isinstance(mapping, Mapping):
        raise MD3ThemeError("theme must be a mapping")
    mode = mapping.get("mode", MD3ThemeMode.DARK.value)
    return md3_theme(str(mode), mapping)


def theme_from_configuration(values: Mapping[str, Any]) -> MD3Theme:
    """从完整配置的精确 ``ui.theme`` 路径读取主题。"""

    if not isinstance(values, Mapping):
        raise MD3ThemeError("configuration must be a mapping")
    ui_values = _section(values.get("ui"), "ui")
    theme_values = _section(ui_values.get("theme"), "ui.theme")
    return MD3Theme.from_mapping(theme_values)


def load_md3_theme(path: str | Path) -> MD3Theme:
    """从 UTF-8 YAML 文件加载主题，文件根必须是主题映射。"""

    source = Path(path).expanduser().resolve()
    try:
        parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise MD3ThemeError(f"unable to load theme: {source}") from exc
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, Mapping):
        raise MD3ThemeError("theme file root must be a mapping")
    return theme_from_mapping(parsed)


def _qss_font(value: str) -> str:
    return f'"{value}"'


def build_md3_stylesheet(theme: MD3Theme) -> str:
    """生成覆盖常用控制台组件的 Qt Style Sheet。"""

    if not isinstance(theme, MD3Theme):
        raise TypeError("theme must be an MD3Theme")
    c = theme.colors
    t = theme.typography
    s = theme.layout
    control_height = max(24, s.control_height_px + s.density * 4)
    compact_height = max(22, s.compact_height_px + s.density * 3)
    vertical_padding = max(2, s.vertical_padding_px + s.density)
    neutral_hover = _mix_color(c.surface_container_high, c.on_surface, 0.06)
    neutral_pressed = _mix_color(c.surface_container_high, c.on_surface, 0.10)
    accent_hover = _mix_color(c.primary, c.on_surface, 0.08)
    accent_pressed = _mix_color(c.primary, c.on_surface, 0.14)
    qss = f"""
QWidget {{
    color: {c.on_surface};
    font-family: {_qss_font(t.font_family)};
    font-size: {t.body_size_px}px;
    selection-background-color: {c.primary};
    selection-color: {c.on_primary};
}}
QMainWindow, QDialog, QWidget[md3Role="root"] {{ background-color: {c.surface}; }}
QFrame[md3Role="card"] {{
    background-color: {c.surface_container};
    border: none;
    border-radius: {s.radius_large_px}px;
}}
QFrame[md3Role="card-outlined"] {{
    background-color: {c.surface_container_low};
    border: 1px solid {c.outline_variant};
    border-radius: {s.radius_large_px}px;
}}
QFrame[md3Role="card-elevated"] {{
    background-color: {c.surface_container_high};
    border: 1px solid {c.outline_variant};
    border-radius: {s.radius_large_px}px;
}}
QLabel[md3Role="title"] {{ font-size: {t.title_size_px}px; font-weight: 700; }}
QLabel[md3Role="muted"] {{ color: {c.on_surface_variant}; }}
QLabel[md3Role="success"] {{ color: {c.success}; font-weight: 600; }}
QLabel[md3Role="warning"] {{ color: {c.warning}; font-weight: 600; }}
QLabel[md3Role="error"] {{ color: {c.error}; font-weight: 600; }}
QPushButton {{
    min-height: {compact_height}px;
    padding: 0 {s.horizontal_padding_px}px;
    border: 1px solid {c.outline_variant};
    border-radius: {s.radius_medium_px}px;
    background-color: {c.surface_container_high};
    color: {c.on_surface};
    font-size: {t.label_size_px}px;
    font-weight: 600;
}}
QPushButton:hover {{ background-color: {neutral_hover}; color: {c.on_surface}; }}
QPushButton:pressed {{ background-color: {neutral_pressed}; color: {c.on_surface}; }}
QPushButton:focus {{ border: {s.focus_width_px}px solid {c.primary}; }}
QPushButton:disabled {{ color: {c.outline}; background-color: {c.surface_container_low}; }}
QPushButton[md3Role="filled"] {{ background-color: {c.primary}; color: {c.on_primary}; }}
QPushButton[md3Role="filled"]:hover {{ background-color: {accent_hover}; color: {c.on_primary}; }}
QPushButton[md3Role="filled"]:pressed {{
    background-color: {accent_pressed};
    color: {c.on_primary};
}}
QPushButton[md3Role="tonal"] {{
    background-color: {c.secondary_container};
    color: {c.on_secondary_container};
}}
QPushButton[md3Role="outlined"] {{
    background-color: transparent;
    color: {c.on_surface};
    border: 1px solid {c.outline_variant};
}}
QPushButton[md3Role="text"] {{ background-color: transparent; color: {c.primary}; }}
QPushButton[md3Role="danger"] {{
    background-color: {c.error_container};
    color: {c.on_error_container};
}}
QPushButton[md3Role="icon"] {{
    min-width: {control_height}px;
    max-width: {control_height}px;
    min-height: {control_height}px;
    max-height: {control_height}px;
    padding: 0;
    border-radius: {control_height // 2}px;
}}
QPushButton[md3Role="chip"], QPushButton[md3Role="chip-selected"] {{
    min-height: {compact_height}px;
    border-radius: {s.radius_small_px}px;
}}
QPushButton[md3Role="chip"] {{ background-color: transparent; border: 1px solid {c.outline}; }}
QPushButton[md3Role="chip-selected"] {{ background-color: {c.secondary_container}; }}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: {control_height}px;
    padding: 0 {s.horizontal_padding_px}px;
    background-color: {c.surface_container_high};
    color: {c.on_surface};
    border: 1px solid {c.outline_variant};
    border-radius: {s.radius_small_px}px;
}}
QTextEdit, QPlainTextEdit {{
    padding-top: {vertical_padding}px;
    padding-bottom: {vertical_padding}px;
}}
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QComboBox:hover,
QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {c.on_surface}; }}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border: {s.focus_width_px}px solid {c.primary}; }}
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {c.outline};
    background-color: {c.surface_container_low};
}}
QComboBox QAbstractItemView {{
    background-color: {c.surface_container};
    color: {c.on_surface};
    border: 1px solid {c.outline_variant};
    border-radius: {s.radius_medium_px}px;
    padding: 4px;
    selection-background-color: {c.primary_container};
    selection-color: {c.on_primary_container};
}}
QCheckBox {{
    spacing: {s.spacing_px}px;
    padding: 2px;
    border: 1px solid transparent;
    border-radius: {s.radius_small_px}px;
}}
QCheckBox::indicator {{
    width: {s.icon_size_px}px;
    height: {s.icon_size_px}px;
    border: 2px solid {c.outline};
    border-radius: 3px;
    background-color: transparent;
}}
QCheckBox::indicator:checked {{ background-color: {c.primary}; border-color: {c.primary}; }}
QCheckBox:focus {{ color: {c.primary}; border-color: {c.primary}; }}
QTabWidget::pane {{ border: none; background-color: transparent; }}
QTabBar::tab {{
    min-height: {control_height}px;
    padding: 0 {s.horizontal_padding_px}px;
    background-color: transparent;
    color: {c.on_surface_variant};
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {c.on_surface}; background-color: {c.surface_container}; }}
QTabBar::tab:selected {{ color: {c.primary}; border-bottom-color: {c.primary}; }}
QTabBar::tab:focus {{ border: {s.focus_width_px}px solid {c.primary}; }}
QListView, QListWidget, QTreeView, QTableView {{
    background-color: {c.surface_container_low};
    alternate-background-color: {c.surface_container};
    color: {c.on_surface};
    border: 1px solid {c.outline_variant};
    border-radius: {s.radius_small_px}px;
}}
QAbstractItemView::item {{ min-height: {compact_height}px; padding: 0 {s.spacing_px}px; }}
QAbstractItemView::item:selected {{
    background-color: {c.primary_container};
    color: {c.on_primary_container};
}}
QHeaderView::section {{
    background-color: {c.surface_container_high};
    color: {c.on_surface_variant};
    border: none;
    padding: {vertical_padding}px {s.horizontal_padding_px}px;
}}
QGroupBox {{
    margin-top: {vertical_padding * 2}px;
    padding-top: {vertical_padding * 2}px;
    border: 1px solid {c.outline_variant};
    border-radius: {s.radius_medium_px}px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: {s.horizontal_padding_px}px;
    padding: 0 {s.spacing_px}px;
}}
QMenu {{
    background-color: {c.surface_container};
    color: {c.on_surface};
    border: 1px solid {c.outline_variant};
}}
QMenu::item {{ padding: {vertical_padding}px {s.horizontal_padding_px * 2}px; }}
QMenu::item:selected {{ background-color: {c.primary_container}; color: {c.on_primary_container}; }}
QToolTip {{
    background-color: {c.inverse_surface};
    color: {c.inverse_on_surface};
    border: none;
    padding: 6px;
}}
QScrollBar:vertical {{ width: 8px; margin: 2px; background: transparent; }}
QScrollBar:horizontal {{ height: 8px; margin: 2px; background: transparent; }}
QScrollBar::handle {{
    background: {c.outline};
    border-radius: 3px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:hover {{ background: {c.primary}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QProgressBar {{
    min-height: 8px;
    max-height: 8px;
    border: none;
    border-radius: 4px;
    background-color: {c.surface_variant};
}}
QProgressBar::chunk {{ border-radius: 4px; background-color: {c.primary}; }}
QComboBox::drop-down {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 28px;
    background-color: {c.surface_container};
    border-left: 1px solid {c.outline_variant};
    border-top-right-radius: {s.radius_small_px}px;
    border-bottom-right-radius: {s.radius_small_px}px;
}}
QComboBox::drop-down:hover {{ background-color: {neutral_hover}; }}
QComboBox::down-arrow {{
    width: 8px;
    height: 8px;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    width: 20px;
    background-color: {c.surface_container};
    border: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QAbstractSpinBox::up-button {{
    subcontrol-position: top right;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button,
QAbstractSpinBox::down-button {{
    subcontrol-position: bottom right;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background-color: {neutral_hover};
}}
QSpinBox::up-arrow, QSpinBox::down-arrow,
QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow,
QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {{
    width: 7px;
    height: 7px;
}}
QToolButton {{
    min-height: {compact_height}px;
    color: {c.on_surface};
    border: 1px solid transparent;
    border-radius: {s.radius_small_px}px;
}}
QToolButton:hover {{ background-color: {neutral_hover}; }}
QToolButton:pressed {{ background-color: {neutral_pressed}; }}
QToolButton:focus {{ border: {s.focus_width_px}px solid {c.primary}; }}
QToolButton:disabled {{ color: {c.outline}; }}
""".strip()
    if theme.extra_stylesheet:
        qss = f"{qss}\n\n/* user theme overrides */\n{theme.extra_stylesheet}"
    return qss


def md3_web_tokens(theme: MD3Theme) -> dict[str, str]:
    """返回 Web 控制台可直接消费的稳定 CSS 自定义属性。"""

    if not isinstance(theme, MD3Theme):
        raise TypeError("theme must be an MD3Theme")
    tokens = {
        f"--md3-color-{name.replace('_', '-')}": value
        for name, value in asdict(theme.colors).items()
    }
    typography = theme.typography
    layout = theme.layout
    motion = theme.motion
    tokens.update(
        {
            "--md3-color-scheme": theme.mode.value,
            "--md3-font-family": typography.font_family,
            "--md3-font-family-monospace": typography.monospace_family,
            "--md3-font-size-body": f"{typography.body_size_px}px",
            "--md3-font-size-label": f"{typography.label_size_px}px",
            "--md3-font-size-title": f"{typography.title_size_px}px",
            "--md3-font-size-headline": f"{typography.headline_size_px}px",
            "--md3-radius-small": f"{layout.radius_small_px}px",
            "--md3-radius-medium": f"{layout.radius_medium_px}px",
            "--md3-radius-large": f"{layout.radius_large_px}px",
            "--md3-radius-extra-large": f"{layout.radius_extra_large_px}px",
            "--md3-control-height": f"{max(24, layout.control_height_px + layout.density * 4)}px",
            "--md3-compact-height": f"{max(22, layout.compact_height_px + layout.density * 3)}px",
            "--md3-density": str(layout.density),
            "--md3-spacing": f"{layout.spacing_px}px",
            "--md3-motion-short": f"{motion.short_duration_ms}ms",
            "--md3-motion-medium": f"{motion.medium_duration_ms}ms",
            "--md3-motion-long": f"{motion.long_duration_ms}ms",
            "--md3-reduced-motion": "1" if motion.reduced_motion else "0",
        }
    )
    if theme.seed:
        tokens["--md3-seed"] = theme.seed
    return tokens


def build_md3_web_css(theme: MD3Theme, *, selector: str = ":root") -> str:
    """把 Web 令牌渲染为 CSS 声明块。"""

    normalized_selector = str(selector).strip()
    if not re.fullmatch(r":root|[.#][A-Za-z_][A-Za-z0-9_-]*", normalized_selector):
        raise MD3ThemeError("selector must be :root, a class, or an id")
    declarations = "\n".join(f"  {name}: {value};" for name, value in md3_web_tokens(theme).items())
    return f"{normalized_selector} {{\n{declarations}\n}}"


def build_qt_palette(theme: MD3Theme) -> Any:
    """延迟创建与主题一致的 ``QPalette``。"""

    try:
        from PySide6.QtGui import QColor, QPalette
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("PySide6 is required to build an MD3 palette") from exc
    c = theme.colors
    palette = QPalette()
    roles = QPalette.ColorRole
    palette.setColor(roles.Window, QColor(c.surface))
    palette.setColor(roles.WindowText, QColor(c.on_surface))
    palette.setColor(roles.Base, QColor(c.surface_container_low))
    palette.setColor(roles.AlternateBase, QColor(c.surface_container))
    palette.setColor(roles.Text, QColor(c.on_surface))
    palette.setColor(roles.Button, QColor(c.secondary_container))
    palette.setColor(roles.ButtonText, QColor(c.on_secondary_container))
    palette.setColor(roles.Highlight, QColor(c.primary))
    palette.setColor(roles.HighlightedText, QColor(c.on_primary))
    palette.setColor(roles.PlaceholderText, QColor(c.on_surface_variant))
    palette.setColor(roles.ToolTipBase, QColor(c.inverse_surface))
    palette.setColor(roles.ToolTipText, QColor(c.inverse_on_surface))
    return palette


def refresh_md3_widget(widget: object) -> None:
    """刷新动态属性选择器，不重建控件。"""

    style_getter = getattr(widget, "style", None)
    if not callable(style_getter):
        return
    style = style_getter()
    if style is None:
        return
    unpolish = getattr(style, "unpolish", None)
    polish = getattr(style, "polish", None)
    if callable(unpolish):
        unpolish(widget)
    if callable(polish):
        polish(widget)
    update = getattr(widget, "update", None)
    if callable(update):
        update()


def set_md3_role(widget: object, role: str | MD3Role) -> object:
    """设置 ``md3Role`` 动态属性并立即刷新样式。"""

    normalized = str(role).strip().lower()
    if normalized not in MD3_COMPONENT_ROLES:
        raise MD3ThemeError(f"unsupported MD3 component role: {normalized}")
    setter = getattr(widget, "setProperty", None)
    if not callable(setter):
        raise TypeError("widget must provide setProperty")
    setter("md3Role", normalized)
    refresh_md3_widget(widget)
    return widget


def apply_md3_theme(target: object, theme: MD3Theme, *, apply_palette: bool = True) -> str:
    """将主题应用到 QApplication 或 QWidget，并返回实际 QSS。"""

    stylesheet = build_md3_stylesheet(theme)
    role_setter = getattr(target, "setProperty", None)
    if callable(role_setter):
        role_setter("md3Role", MD3Role.ROOT.value)
    stylesheet_setter = getattr(target, "setStyleSheet", None)
    if not callable(stylesheet_setter):
        raise TypeError("theme target must provide setStyleSheet")
    stylesheet_setter(stylesheet)
    palette_setter = getattr(target, "setPalette", None)
    if apply_palette and callable(palette_setter):
        palette_setter(build_qt_palette(theme))
    refresh_md3_widget(target)
    return stylesheet


class MD3ThemeController:
    """保存当前主题，并为配置热重载提供单一应用入口。"""

    def __init__(self, theme: MD3Theme | None = None) -> None:
        self._theme = theme or DARK_MD3_THEME
        self._target: object | None = None
        self._revision = 0

    @property
    def theme(self) -> MD3Theme:
        return self._theme

    @property
    def revision(self) -> int:
        return self._revision

    def attach(self, target: object) -> str:
        """绑定目标并应用当前主题。"""

        self._target = target
        return apply_md3_theme(target, self._theme)

    def update(self, theme: MD3Theme | Mapping[str, Any]) -> str:
        """替换主题；已绑定目标会在当前线程内原位更新。"""

        resolved = theme if isinstance(theme, MD3Theme) else theme_from_mapping(theme)
        self._theme = resolved
        self._revision += 1
        if self._target is None:
            return build_md3_stylesheet(resolved)
        return apply_md3_theme(self._target, resolved)

    def reload(self, path: str | Path) -> str:
        """从 YAML 重读并应用，供外部文件监视器调用。"""

        return self.update(load_md3_theme(path))


class MD3WidgetFactory:
    """创建带稳定动态属性的常用 Qt Widgets 控件。"""

    def __init__(self, theme: MD3Theme | None = None) -> None:
        self.theme = theme or DARK_MD3_THEME

    @staticmethod
    def _widgets() -> Any:
        try:
            from PySide6 import QtWidgets
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("PySide6 is required to create MD3 widgets") from exc
        return QtWidgets

    def apply(self, target: object) -> str:
        return apply_md3_theme(target, self.theme)

    def button(
        self,
        text: str,
        *,
        role: str | MD3Role = MD3Role.FILLED,
        parent: object | None = None,
        checkable: bool = False,
    ) -> Any:
        widgets = self._widgets()
        control = widgets.QPushButton(str(text), parent)
        control.setCheckable(bool(checkable))
        try:
            from PySide6.QtCore import Qt

            control.setCursor(Qt.CursorShape.PointingHandCursor)
        except (ImportError, ModuleNotFoundError, AttributeError):
            pass
        return set_md3_role(control, role)

    def card(
        self,
        *,
        role: str | MD3Role = MD3Role.CARD,
        parent: object | None = None,
    ) -> Any:
        if str(role) not in {
            MD3Role.CARD.value,
            MD3Role.CARD_OUTLINED.value,
            MD3Role.CARD_ELEVATED.value,
        }:
            raise MD3ThemeError("card role must be card, card-outlined, or card-elevated")
        widgets = self._widgets()
        control = widgets.QFrame(parent)
        return set_md3_role(control, role)

    def chip(
        self,
        text: str,
        *,
        selected: bool = False,
        parent: object | None = None,
    ) -> Any:
        control = self.button(
            text,
            role=MD3Role.CHIP_SELECTED if selected else MD3Role.CHIP,
            parent=parent,
            checkable=True,
        )
        control.setChecked(bool(selected))
        return control

    def text_field(self, *, placeholder: str = "", parent: object | None = None) -> Any:
        widgets = self._widgets()
        control = widgets.QLineEdit(parent)
        control.setPlaceholderText(str(placeholder))
        return control

    def combo_box(self, *, parent: object | None = None) -> Any:
        return self._widgets().QComboBox(parent)

    def switch(self, text: str, *, parent: object | None = None) -> Any:
        return self._widgets().QCheckBox(str(text), parent)


__all__ = [
    "DARK_MD3_THEME",
    "LIGHT_MD3_THEME",
    "MD3_COMPONENT_ROLES",
    "MD3_THEME_KEYS",
    "MD3ColorScheme",
    "MD3Layout",
    "MD3Motion",
    "MD3Role",
    "MD3Theme",
    "MD3ThemeController",
    "MD3ThemeError",
    "MD3ThemeMode",
    "MD3Typography",
    "MD3WidgetFactory",
    "apply_md3_theme",
    "build_md3_stylesheet",
    "build_md3_web_css",
    "build_qt_palette",
    "load_md3_theme",
    "md3_web_tokens",
    "md3_theme",
    "refresh_md3_widget",
    "remap_legacy_stylesheet",
    "set_md3_role",
    "theme_from_configuration",
    "theme_from_mapping",
]
