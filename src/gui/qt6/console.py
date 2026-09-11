from __future__ import annotations

import copy
import json
import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from config.editor import (
    ConfigurationConflictError,
    ConfigurationEditSession,
    atomic_write_configuration,
    merge_editor_values,
    parse_editor_yaml,
    prepare_persisted_values,
    preserve_secret_values,
    validate_editor_values,
)
from config.loader import ConfigurationError, parse_bool
from core.adapters.direct.errors import AdapterConfigurationError
from core.rendering.actions import ExpressionRequest, MotionRequest
from gui.qt6.config_panel import AudioInputSelector, ConfigurationPanel
from gui.qt6.display_size import DISPLAY_SIZE_LABELS, DISPLAY_SIZE_PRESETS
from gui.qt6.fancyui import (
    FancyCard,
    FancyCommandBar,
    FancyInfoBar,
    FancyNavigationView,
    FancyStyleController,
    build_fancy_stylesheet,
    fancy_icon,
)
from gui.qt6.md3 import (
    MD3Theme,
    build_md3_stylesheet,
    theme_from_configuration,
)
from gui.qt6.microphone import AudioInputDeviceChoice
from gui.qt6.model_setup import ModelSetupDialog, ModelSetupDraft, model_setup_drafts
from gui.renderers.protocol import DEFAULT_EXPRESSION_NAMES, DEFAULT_MOTION_NAMES
from services.model_routing import channel_from_mapping

try:  # Qt 依赖保持可选，核心服务仍可在无桌面环境运行。
    from PySide6.QtCore import Qt, QTime, QTimer, Signal
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    pyside6_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_available = False


Callback = Callable[..., Any]
_ENVIRONMENT_SCAN_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")
_ACTION_PARAMETER_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}\Z")


def _status_value(result: object) -> str:
    """提取平台能力结果中的稳定状态文本。"""

    if isinstance(result, Mapping):
        # 平台能力对象在不同边界可能被序列化为 ``status``，也可能保留
        # ``state`` 字段；两者都必须进入同一状态判断，避免把失败结果
        # 当成成功的动作回显。
        value = result.get("status", result.get("state", ""))
        return str(getattr(value, "value", value))
    state = getattr(result, "state", None)
    if state is None:
        # 兼容旧平台/测试替身直接暴露 ``status`` 的返回对象；不能因此
        # 丢失状态标签并只显示原始 detail。
        state = getattr(result, "status", None)
    return str(getattr(state, "value", state or ""))


def _detail_value(result: object) -> str:
    """提取回调返回的详细原因，用于区分 Wayland 降级和真实失败。"""

    if isinstance(result, Mapping):
        value = result.get("detail", result.get("reason", ""))
    else:
        value = getattr(result, "detail", "")
    return str(value or "")


def _payload_channel_values(
    payload: Mapping[str, Any], channel_id: str
) -> Mapping[str, Any] | None:
    """提取载荷中指定渠道的字段映射，不展开或读取任何密钥。"""

    llm = payload.get("llm")
    source: object = llm if isinstance(llm, Mapping) else payload
    channels = source.get("channels") if isinstance(source, Mapping) else None
    if isinstance(channels, list):
        for channel in channels:
            if (
                isinstance(channel, Mapping)
                and str(channel.get("id", "") or "").strip() == channel_id
            ):
                return channel
    if isinstance(source, Mapping) and (
        str(source.get("id", "") or "").strip() == channel_id
        or str(source.get("channel_id", "") or "").strip() == channel_id
    ):
        return source
    if isinstance(payload, Mapping) and (
        str(payload.get("id", "") or "").strip() == channel_id
        or str(payload.get("channel_id", "") or "").strip() == channel_id
    ):
        return payload
    return None


def _payload_declares_api_key(payload: Mapping[str, Any], channel_id: str) -> bool:
    """判断模型向导载荷是否明确声明了密钥字段。

    旧版宿主可能只提交基础连接字段，此时更新已有渠道应保留原引用；
    当前向导的完整 ``llm.channels`` 载荷即使密钥为空也代表用户明确清除。
    """

    channel = _payload_channel_values(payload, channel_id)
    return bool(channel is not None and ("api_key" in channel or "api_key_env" in channel))


def _only_missing_channel_credentials(values: Mapping[str, Any]) -> bool:
    """判断配置中是否只缺少渠道密钥环境变量。"""

    locations: list[tuple[str, tuple[object, ...]]] = []
    active: set[int] = set()

    def visit(value: object, path: tuple[object, ...]) -> None:
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                raise ConfigurationError("configuration contains recursive aliases")
            active.add(identity)
            try:
                for key, nested in value.items():
                    visit(nested, (*path, str(key)))
            finally:
                active.remove(identity)
        elif isinstance(value, (list, tuple)):
            identity = id(value)
            if identity in active:
                raise ConfigurationError("configuration contains recursive aliases")
            active.add(identity)
            try:
                for index, nested in enumerate(value):
                    visit(nested, (*path, index))
            finally:
                active.remove(identity)
        elif isinstance(value, str):
            locations.extend(
                (match.group(1), path) for match in _ENVIRONMENT_SCAN_PATTERN.finditer(value)
            )

    visit(values, ())
    missing = [(name, path) for name, path in locations if name not in os.environ]
    return bool(missing) and all(
        len(path) == 4
        and path[0] == "llm"
        and path[1] == "channels"
        and isinstance(path[2], int)
        and path[3] in {"api_key", "token"}
        for _name, path in missing
    )


_SUCCESS_STATES = frozenset(
    {
        "available",
        "requested",
        "completed",
        "updated",
        "started",
        "accepted",
        "ok",
        "saved",
        "success",
        "validated",
        "reloaded",
    }
)

_WINUI_DARK_ROLES: dict[str, str] = {
    "primary": "#60CDFF",
    "on_primary": "#003547",
    "primary_container": "#004C6A",
    "on_primary_container": "#C4E7FF",
    "secondary": "#A9D6F5",
    "on_secondary": "#123247",
    "secondary_container": "#294B60",
    "on_secondary_container": "#D6EFFF",
    "tertiary": "#9DCAEB",
    "on_tertiary": "#082F49",
    "error": "#FFB4AB",
    "on_error": "#690005",
    "error_container": "#61201D",
    "on_error_container": "#FFDAD6",
    "surface": "#202020",
    "on_surface": "#FFFFFF",
    "surface_variant": "#323232",
    "on_surface_variant": "#D6D6D6",
    "surface_container_low": "#1C1C1C",
    "surface_container": "#272727",
    "surface_container_high": "#303030",
    "outline": "#8A8A8A",
    "outline_variant": "#454545",
    "inverse_surface": "#F3F3F3",
    "inverse_on_surface": "#1A1A1A",
    "success": "#6CCB5F",
    "on_success": "#102A0C",
    "warning": "#FCE100",
    "on_warning": "#3B3200",
}

_WINUI_LIGHT_ROLES: dict[str, str] = {
    "primary": "#0067C0",
    "on_primary": "#FFFFFF",
    "primary_container": "#D9EAFF",
    "on_primary_container": "#003258",
    "secondary": "#3A6078",
    "on_secondary": "#FFFFFF",
    "secondary_container": "#D3E5F4",
    "on_secondary_container": "#173447",
    "tertiary": "#315D7A",
    "on_tertiary": "#FFFFFF",
    "error": "#C42B1C",
    "on_error": "#FFFFFF",
    "error_container": "#FFDAD6",
    "on_error_container": "#410002",
    "surface": "#F3F3F3",
    "on_surface": "#1A1A1A",
    "surface_variant": "#E9E9E9",
    "on_surface_variant": "#4F4F4F",
    "surface_container_low": "#FAFAFA",
    "surface_container": "#F7F7F7",
    "surface_container_high": "#FFFFFF",
    "outline": "#767676",
    "outline_variant": "#D1D1D1",
    "inverse_surface": "#2B2B2B",
    "inverse_on_surface": "#FFFFFF",
    "success": "#0F7B0F",
    "on_success": "#FFFFFF",
    "warning": "#9D5D00",
    "on_warning": "#FFFFFF",
}

_CONSOLE_STYLE = """
QWidget#PetConsoleWindow {
    background: @surface@;
    color: @on-surface@;
}
QWidget#consoleOperationPage {
    background: @surface@;
}
QWidget#consoleOverviewScrollContent,
QWidget#consoleControlScrollContent,
QWidget#consolePetScrollContent,
QWidget#secondaryActionsPanelContent,
QWidget#consoleModuleCenterContent,
QWidget#consolePageHeading {
    background: @surface@;
}
QFrame#consoleHeaderCard, QFrame#consoleCard, QFrame#conversationCard,
QFrame#quickActionsCard, QFrame#windowActionsCard, QFrame#positionCard,
QFrame#approvalCard, QFrame#diagnosticsCard, QFrame#consoleRescueCard,
QFrame[consoleModuleCard="true"] {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 0, y2: 1,
        stop: 0 @surface-container-high@,
        stop: 0.45 @surface-container@,
        stop: 1 @surface-container@
    );
    border: 1px solid @outline-variant@;
    border-top-color: @outline@;
    border-radius: 14px;
}
QFrame#consoleHeaderCard {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 @surface-container-high@,
        stop: 0.58 @surface-container@,
        stop: 1 @surface-container@
    );
}
QFrame#consoleRescueCard {
    background: @surface-alpha-185@;
    border-top-color: @primary@;
}
QLabel#petInteractionFeedback {
    color: @on-primary-container@;
    background: @primary-alpha-24@;
    border: 1px solid @primary-alpha-105@;
    border-radius: 8px;
    padding: 6px 9px;
}
QLabel#petInteractionFeedback[state="idle"] {
    color: @on-surface-variant@;
    background: transparent;
    border-color: @outline-variant@;
}
QPushButton#rescueActionButton {
    min-height: 34px;
    padding: 6px 9px;
    background: @surface-container@;
    border-color: @outline@;
}
QPushButton#rescueActionButton:hover {
    background: @secondary-container@;
    border-color: @primary@;
}
QLabel#consoleEyebrow {
    color: @tertiary@;
    font-size: 11px;
    font-weight: 700;
}
QLabel#consoleTitleLabel {
    color: @on-surface@;
    font-size: 22px;
    font-weight: 700;
}
QLabel#consoleSubtitleLabel, QLabel#consoleHintLabel, QLabel#consoleMetaLabel {
    color: @on-surface-variant@;
}
QLabel[consoleModuleState="true"] {
    color: @on-surface@;
    font-size: 13px;
    font-weight: 700;
}
QLabel[consoleModuleState="true"][state="available"] {
    color: @success@;
}
QLabel[consoleModuleState="true"][state="degraded"],
QLabel[consoleModuleState="true"][state="checking"] {
    color: @warning@;
}
QLabel[consoleModuleState="true"][state="unavailable"] {
    color: @error@;
}
QLabel[consoleModuleDetail="true"], QLabel[consoleModuleMeta="true"] {
    color: @on-surface-variant@;
    font-size: 12px;
}
QLabel#consoleStatusLabel {
    color: @success@;
    font-size: 14px;
    font-weight: 700;
}
QLabel#consoleStatusLabel[state="busy"] {
    color: @warning@;
}
QLabel#consoleStatusLabel[state="warning"] {
    color: @warning@;
}
QLabel#consoleStatusLabel[state="error"] {
    color: @error@;
}
QLabel#consoleEmptyState {
    color: @on-surface-variant@;
    background: @surface-container-low@;
    border: 1px dashed @outline@;
    border-radius: 10px;
    padding: 12px;
}
QLabel#consoleSectionTitle {
    color: @on-surface@;
    font-size: 16px;
    font-weight: 700;
}
QLabel#consoleSectionHint, QLabel#expressionCapabilityHint,
QLabel#motionCapabilityHint, QLabel#developerActionHint {
    color: @on-surface-variant@;
    font-size: 12px;
}
QLabel#desktopObservationStatus {
    color: @on-surface-variant@;
    background: @surface-container-low-alpha-150@;
    border: 1px solid @outline-variant@;
    border-radius: 8px;
    padding: 6px 8px;
}
QLabel#rendererStatus {
    color: @on-surface-variant@;
    background: @surface-container-low-alpha-150@;
    border: 1px solid @outline-variant@;
    border-radius: 8px;
    padding: 6px 8px;
}
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox {
    background: @surface-container-low@;
    color: @on-surface@;
    border: 1px solid @outline@;
    border-radius: 10px;
    padding: 8px;
    selection-background-color: @primary@;
    selection-color: @on-primary@;
}
QPlainTextEdit#conversationOutput {
    border-color: @outline-variant@;
    padding: 10px;
}
QComboBox QAbstractItemView {
    background: @surface-container@;
    color: @on-surface@;
    border: 1px solid @outline@;
    selection-background-color: @primary-container@;
    selection-color: @on-surface@;
}
QPushButton {
    background: @surface-container@;
    color: @on-surface@;
    border: 1px solid @outline@;
    border-radius: 10px;
    padding: 8px 12px;
    min-height: 36px;
}
QPushButton:hover {
    background: @surface-container-high@;
    border-color: @secondary@;
}
QPushButton:pressed {
    background: @outline-variant@;
}
QPushButton:disabled {
    color: @outline@;
    border-color: @outline-variant@;
}
QPushButton[role="primary"] {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 @primary@, stop: 1 @tertiary@
    );
    color: @on-primary@;
    border: none;
    font-weight: 700;
}
QPushButton[role="primary"]:hover {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 @primary@, stop: 1 @tertiary@
    );
}
QPushButton[role="quick"] {
    background: @surface-container@;
    border-color: @outline@;
    padding: 7px 10px;
    min-height: 38px;
}
QPushButton[role="quick"]:checked {
    background: @primary-alpha-52@;
    border: 2px solid @primary@;
    color: @primary@;
}
QPushButton[role="quiet"] {
    background: transparent;
    border-color: @outline-variant@;
    color: @on-surface-variant@;
}
QPushButton[role="danger"] {
    color: @error@;
    border-color: @error@;
}
QCheckBox {
    color: @on-surface-variant@;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid @outline@;
    border-radius: 6px;
    background: @surface-container-low@;
}
QCheckBox::indicator:checked {
    background: @primary@;
    border-color: @primary@;
}
QListWidget {
    background: @surface-container-low@;
    color: @on-surface@;
    border: 1px solid @outline-variant@;
    border-radius: 10px;
    padding: 4px;
}
QListWidget::item {
    padding: 8px;
    border-radius: 8px;
}
QListWidget::item:selected {
    background: @primary-alpha-52@;
    color: @on-surface@;
}
QScrollArea {
    border: none;
    background: transparent;
}
QAbstractScrollArea::viewport {
    background: @surface@;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: @surface-container-low@;
    border: none;
    margin: 0;
}
QScrollBar:vertical { width: 12px; }
QScrollBar:horizontal { height: 12px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: @primary-container@;
    border: 1px solid @outline@;
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: @surface-container-low@;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: @surface-container@;
    border: none;
}
QTabWidget::pane {
    border: 1px solid @outline-variant@;
    border-radius: 14px;
    top: -1px;
}
QTabBar::tab {
    background: @surface-container@;
    color: @on-surface-variant@;
    padding: 9px 18px;
    margin-right: 4px;
    border: 1px solid @outline-variant@;
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
}
QTabBar::tab:selected {
    background: @surface-container@;
    color: @primary@;
    border-color: @outline@;
}
QGroupBox {
    background: @surface-container-low-alpha-180@;
    border: 1px solid @outline-variant@;
    border-radius: 10px;
    margin-top: 12px;
    padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: @secondary@;
}
"""


def _qss_rgba(color: str, alpha: int) -> str:
    """把已校验的 MD3 十六进制颜色转换为 Qt QSS 的 RGBA 表达式。"""

    return f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, {alpha})"


def _console_component_stylesheet(theme: MD3Theme) -> str:
    """使用语义角色解析控制台组件样式，不保留旧主题颜色映射。"""

    colors = theme.colors
    tokens = {
        "@surface@": colors.surface,
        "@surface-alpha-185@": _qss_rgba(colors.surface, 185),
        "@surface-container-low@": colors.surface_container_low,
        "@surface-container-low-alpha-150@": _qss_rgba(colors.surface_container_low, 150),
        "@surface-container-low-alpha-180@": _qss_rgba(colors.surface_container_low, 180),
        "@surface-container@": colors.surface_container,
        "@surface-container-high@": colors.surface_container_high,
        "@on-surface@": colors.on_surface,
        "@on-surface-variant@": colors.on_surface_variant,
        "@primary@": colors.primary,
        "@primary-alpha-24@": _qss_rgba(colors.primary, 24),
        "@primary-alpha-52@": _qss_rgba(colors.primary, 52),
        "@primary-alpha-105@": _qss_rgba(colors.primary, 105),
        "@on-primary@": colors.on_primary,
        "@primary-container@": colors.primary_container,
        "@on-primary-container@": colors.on_primary_container,
        "@secondary@": colors.secondary,
        "@secondary-container@": colors.secondary_container,
        "@tertiary@": colors.tertiary,
        "@outline@": colors.outline,
        "@outline-variant@": colors.outline_variant,
        "@error@": colors.error,
        "@success@": colors.success,
        "@warning@": colors.warning,
    }
    stylesheet = _CONSOLE_STYLE
    for token, value in tokens.items():
        stylesheet = stylesheet.replace(token, value)
    # 业务页面保留稳定 objectName；最终覆盖层只调整视觉层级，避免
    # 旧渐变和高亮边框在不同平台的原生样式上形成过强噪声。
    radius = theme.layout.radius_large_px
    modern = f"""
QFrame#consoleHeaderCard, QFrame#consoleCard, QFrame#conversationCard,
QFrame#quickActionsCard, QFrame#windowActionsCard, QFrame#positionCard,
QFrame#approvalCard, QFrame#diagnosticsCard, QFrame#consoleRescueCard,
QFrame[consoleModuleCard="true"] {{
    background: {colors.surface_container};
    border: 1px solid {colors.outline_variant};
    border-radius: {radius}px;
}}
QFrame#consoleHeaderCard {{
    background: {colors.surface_container_high};
    border-top: 2px solid {colors.primary};
}}
QFrame#consoleRescueCard {{
    background: {colors.surface_container_low};
    border-left: 2px solid {colors.primary};
}}
QLabel#consoleTitleLabel {{ font-size: 22px; font-weight: 700; }}
QLabel#consoleSectionTitle {{ font-size: 15px; font-weight: 700; }}
QLabel#consoleSectionHint, QLabel#consoleSubtitleLabel,
QLabel#consoleHintLabel, QLabel#consoleMetaLabel {{
    color: {colors.on_surface_variant};
}}
QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {colors.surface_container_low};
    border: 1px solid {colors.outline_variant};
    border-radius: {theme.layout.radius_medium_px}px;
}}
QLineEdit:hover, QPlainTextEdit:hover, QComboBox:hover,
QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {colors.on_surface_variant};
}}
QPushButton {{
    min-height: {max(32, theme.layout.compact_height_px)}px;
    border: 1px solid {colors.outline_variant};
    border-radius: {theme.layout.radius_medium_px}px;
    background: {colors.surface_container};
}}
QPushButton:hover {{
    background: {colors.surface_container_high};
    border-color: {colors.primary};
}}
QPushButton:pressed {{ background: {colors.surface_variant}; }}
QPushButton[role="primary"] {{
    background: {colors.primary};
    color: {colors.on_primary};
    border-color: {colors.primary};
}}
QPushButton[role="primary"]:hover {{ background: {colors.primary_container}; }}
QPushButton[role="quick"] {{
    background: {colors.surface_container};
    border-color: {colors.outline_variant};
}}
QPushButton[role="quick"]:checked {{
    background: {_qss_rgba(colors.primary, 42)};
    border: 1px solid {colors.primary};
    color: {colors.primary};
}}
QPushButton[role="quiet"] {{
    background: transparent;
    color: {colors.on_surface_variant};
}}
QPushButton[role="danger"] {{
    background: transparent;
    color: {colors.error};
    border-color: {colors.error};
}}
QListWidget, QPlainTextEdit#conversationOutput {{
    background: {colors.surface_container_low};
    border-color: {colors.outline_variant};
}}
QListWidget::item:selected {{
    background: {colors.primary_container};
    color: {colors.on_primary_container};
}}
QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent;
    border: none;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {colors.outline};
    border: none;
    border-radius: 4px;
}}
QScrollBar::handle:hover {{
    background: {colors.primary};
}}
"""
    return stylesheet + "\n\n" + modern.strip()


def _console_stylesheet(values: Mapping[str, Any]) -> tuple[object, str]:
    """从 MD3 配置生成 WinUI 中性色板和控制台状态规则。"""

    configured_theme = theme_from_configuration(values)
    ui_values = values.get("ui")
    theme_values = ui_values.get("theme") if isinstance(ui_values, Mapping) else None
    theme_mapping = theme_values if isinstance(theme_values, Mapping) else {}
    explicit_roles = theme_mapping.get("roles")
    role_overrides = dict(explicit_roles) if isinstance(explicit_roles, Mapping) else {}
    winui_roles = dict(
        _WINUI_LIGHT_ROLES if configured_theme.mode.value == "light" else _WINUI_DARK_ROLES
    )
    if theme_mapping.get("seed") not in (None, ""):
        configured_roles = configured_theme.to_mapping()["roles"]
        for name in (
            "primary",
            "on_primary",
            "primary_container",
            "on_primary_container",
        ):
            winui_roles[name] = configured_roles[name]
    winui_roles.update(role_overrides)
    resolved_mapping = configured_theme.to_mapping()
    resolved_mapping["roles"] = winui_roles
    theme = type(configured_theme).from_mapping(resolved_mapping)
    stylesheet = "\n\n".join(
        (
            build_md3_stylesheet(theme),
            build_fancy_stylesheet(theme),
            _console_component_stylesheet(theme),
        )
    )
    return theme, stylesheet


def _action_names(values: Sequence[str] | object, fallback: Sequence[str]) -> tuple[str, ...]:
    """把渲染器能力转换成稳定、非空、去重的控制台选项。

    能力来自插件或 WebChannel 时可能是 None、单个字符串或包含空白项
    的可迭代对象。直接对字符串做 tuple 会把一个动作拆成字符，空白项
    则会在 Qt 下拉框中显示成“空选项”；这里统一在进入 Qt 前清理。
    同一动作的大小写别名只保留第一次出现的拼写，避免快捷按钮显示两个
    相同文案而点击时又提交不同的名称。
    """

    if isinstance(values, str):
        source: object = (values,)
    elif values is None:
        source = ()
    else:
        source = values
    try:
        iterator = iter(source)  # type: ignore[arg-type]
    except TypeError:
        iterator = iter(())
    result: list[str] = []
    seen: set[str] = set()
    for item in iterator:
        text = str(item or "").strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    if result:
        return tuple(result)
    fallback_result: list[str] = []
    fallback_seen: set[str] = set()
    for item in fallback:
        text = str(item or "").strip()
        key = text.casefold()
        if not text or key in fallback_seen:
            continue
        fallback_seen.add(key)
        fallback_result.append(text)
    return tuple(fallback_result)


def _action_parameters(value: object) -> dict[str, float]:
    """解析面向用户的 ``参数名=数值`` 列表并生成渲染契约映射。"""

    rendered = str(value or "").strip()
    if not rendered:
        return {}
    items = [
        item.strip()
        for item in rendered.replace("，", ",").replace("\n", ",").split(",")
        if item.strip()
    ]
    if len(items) > 32:
        raise ValueError("动作参数最多 32 项")
    result: dict[str, float] = {}
    for item in items:
        name, separator, raw_value = item.partition("=")
        name = name.strip()
        if not separator or not _ACTION_PARAMETER_PATTERN.fullmatch(name):
            raise ValueError("动作参数需使用 参数名=数值 格式")
        if name in result:
            raise ValueError(f"动作参数重复：{name}")
        try:
            parsed = float(raw_value.strip())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"动作参数 {name} 不是数值") from exc
        if not math.isfinite(parsed) or parsed < -10_000.0 or parsed > 10_000.0:
            raise ValueError(f"动作参数 {name} 超出范围")
        result[name] = parsed
    return result


def _tts_engine_label(value: object) -> str:
    """把公开的 TTS 健康引擎名转换为稳定用户标签。"""

    normalized = str(value or "").strip().lower()
    if normalized.startswith("tts-router"):
        return "多音色路由"
    return {
        "gpt-sovits": "GPT-SoVITS",
        "gpt-sovits-stdio": "GPT-SoVITS 标准输入",
        "subprocess": "本地语音进程",
        "text-only": "仅文字",
        "tts": "语音协调器",
        "pending": "语音后端初始化",
    }.get(normalized, "语音后端待检测")


def _operation_succeeded(result: object) -> bool:
    """判断控制台回调是否真正接受了操作。

    渲染器的表情/动作接口返回布尔值，而平台接口返回能力对象或映射。
    ``False``、``unavailable`` 和 ``degraded`` 不能再被统一的“非空结果”
    误报为成功；点击穿透在 Wayland 上的 ``degraded`` 会由调用方单独处理。
    """

    if result is None:
        return False
    if isinstance(result, bool):
        return result
    status = _status_value(result).strip().lower()
    if status:
        return status in _SUCCESS_STATES
    # 保持兼容旧的本地回调：没有状态字段但返回对象，视为已提交。
    return True


def _operation_pending(result: object) -> bool:
    """判断操作是否已排队但等待模型/宿主就绪。"""

    return _status_value(result).strip().lower() == "pending"


def _wayland_topmost_request_accepted(result: object) -> bool:
    """识别已保留 Qt 置顶请求、但最终由 Wayland 合成器决定的结果。

    ``degraded`` 不能在普通操作中一概视为成功；这里只接受窗口层明确
    返回布尔 ``enabled`` 且详情同时包含 Wayland/compositor 证据的结果，
    避免把 X11 标志丢失或未知平台的部分结果误报成已完成。
    """

    if not isinstance(result, Mapping):
        return False
    if _status_value(result).strip().lower() != "degraded":
        return False
    if not isinstance(result.get("enabled"), bool):
        return False
    detail = _detail_value(result).casefold()
    return "wayland" in detail and "compositor" in detail


def _xy_value(value: object) -> tuple[int, int] | None:
    """从桌面宿主的位置契约提取整数坐标。

    ``PetOpenGLWindow.position`` 返回 ``QPoint``，Web 宿主的控制台回调返回
    ``(x, y)``；测试和跨进程边界还可能把它序列化成 ``{"x": ..., "y": ...}``。
    这里只接受这三种已定义形态，不把任意对象的属性当成坐标。
    """

    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            return None
        value = (value["x"], value["y"])
    x_getter = getattr(value, "x", None)
    y_getter = getattr(value, "y", None)
    if callable(x_getter) and callable(y_getter):
        try:
            return int(round(float(x_getter()))), int(round(float(y_getter())))
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        return int(round(float(value[0]))), int(round(float(value[1])))  # type: ignore[index]
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None


def _bounds_value(value: object) -> tuple[int, int, int, int] | None:
    """从桌面宿主的位置边界契约提取 ``left, top, right, bottom``。"""

    if isinstance(value, Mapping):
        if not all(key in value for key in ("left", "top", "right", "bottom")):
            return None
        value = (value["left"], value["top"], value["right"], value["bottom"])
    if isinstance(value, (str, bytes, bytearray)):
        return None
    try:
        left, top, right, bottom = (
            int(round(float(value[index])))  # type: ignore[index]
            for index in range(4)
        )
    except (IndexError, KeyError, TypeError, ValueError, OverflowError):
        return None
    if right < left or bottom < top:
        return None
    return left, top, right, bottom


_SECRET_MARKERS = frozenset(
    {
        "api_key",
        "api-key",
        "access_key",
        "access-key",
        "private_key",
        "private-key",
        "credential",
        "token",
        "secret",
        "password",
        "authorization",
        "cookie",
    }
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(['\"]?(?:api[_-]?key|access[_-]?key|private[_-]?key|credential|"
    r"token|secret|password|authorization|cookie)['\"]?\s*[:=]\s*)"
    r"(['\"]?)[^,\s'\"}]+"
)
_INTERNAL_PARAMETER_PATTERN = re.compile(r"(?i)\b(?:param|parameter)[a-z0-9_:-]*\b")
_INTERNAL_PROTOCOL_PATTERN = re.compile(
    r"(?i)\b(?:status|state|detail|zone|operation[_-]?id|call[_-]?id)\s*="
)
_SAFE_RESULT_FIELDS = (
    "status",
    "state",
    "detail",
    "reason",
    "message",
    "visible",
    "enabled",
    "x",
    "y",
    "left",
    "top",
    "right",
    "bottom",
    "count",
    "current",
    "applied",
    "tier",
    "zone",
)

_RESULT_STATUS_LABELS = {
    "idle": "等待处理",
    "requested": "已提交",
    "pending": "等待处理",
    "started": "已开始",
    "accepted": "已接受",
    "completed": "已完成",
    "available": "已就绪",
    "updated": "已更新",
    "saved": "已保存",
    "validated": "已校验",
    "reloaded": "已刷新",
    "degraded": "降级运行",
    "failed": "执行失败",
    "error": "暂时不可用",
    "unavailable": "暂时不可用",
    "cancelled": "已停止",
    "canceled": "已停止",
    "denied": "已拒绝",
    "rejected": "已拒绝",
    "timeout": "响应超时",
    "tool_limit": "需要确认",
    "restart_required": "需要重启",
}
_RESULT_FIELD_LABELS = {
    "detail": "说明",
    "reason": "说明",
    "message": "提示",
    "visible": "显示",
    "enabled": "开关",
    "count": "数量",
    "current": "好感度",
    "applied": "本次变化",
    "tier": "阶段",
    "zone": "互动部位",
}
_STREAM_STATUS_LABELS = {
    "approval_required": "等待确认",
    "tool_running": "正在执行",
    "tool_calls": "正在处理",
    "completed": "已完成",
    "failed": "未完成",
    "denied": "已拒绝",
    "configuration": "需要配置",
    "authentication": "连接需要密钥",
    "authorization": "连接权限不足",
    "network": "网络暂时不可用",
    "timeout": "响应超时",
    "cancelled": "已停止",
    "canceled": "已停止",
}
_STREAM_STATUS_PATTERN = re.compile(r"\[([a-z][a-z0-9_-]{0,48})\]\s*", re.IGNORECASE)
_ZONE_LABELS = {
    "head": "猫猫头",
    "body": "身体",
    "upper": "上半身",
    "lower_left": "左边",
    "lower_right": "右边",
}


def _safe_ui_text(value: object, *, limit: int = 240) -> str:
    """生成不会把密钥、请求体或过长原始对象带入控制台的文本。"""

    # 不能用 ``value or ""``：好感度当前值/变化量为 0、布尔状态为 False
    # 时也必须在反馈和诊断摘要中可见。只有 None 表示调用方没有提供值。
    text = ("" if value is None else str(value)).strip()
    text = "".join(" " if char in "\r\n\x00" else char for char in text)
    if not text:
        return ""
    lowered = text.casefold()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        text = _SECRET_ASSIGNMENT.sub(r"\1***", text)
        # 未能识别结构化赋值时，宁可隐藏整段详情，也不把疑似密钥回显。
        if any(marker in text.casefold() for marker in _SECRET_MARKERS):
            return "敏感详情已隐藏"
    if _INTERNAL_PARAMETER_PATTERN.search(text):
        return "敏感详情已隐藏"
    if _INTERNAL_PROTOCOL_PATTERN.search(text):
        return "敏感详情已隐藏"
    if len(text) > limit:
        return text[: max(0, limit - 1)] + "…"
    return text


def _audit_payload_text(value: object, *, limit: int = 220) -> str:
    """把审计载荷压缩为可读的一行，并沿用控制台的敏感字段过滤。"""

    if value is None:
        return ""
    if isinstance(value, str):
        rendered = value
    else:
        try:
            rendered = json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
                default=str,
            )
        except (TypeError, ValueError):
            rendered = str(value)
    return _safe_ui_text(rendered, limit=limit)


def _safe_result_summary(result: object) -> str:
    """只提取状态摘要，避免把回调原始参数/配置写入活动记录。"""

    if result is None:
        return "无返回结果"
    if isinstance(result, bool):
        return "成功" if result else "失败"
    if isinstance(result, Mapping):
        parts: list[str] = []
        status_value = _status_value(result).strip().casefold()
        if status_value:
            parts.append(_RESULT_STATUS_LABELS.get(status_value, "处理中"))
        for key in _SAFE_RESULT_FIELDS:
            if key == "status" or key in {"x", "y", "left", "top", "right", "bottom"}:
                # 坐标和边界属于控制台内部状态；失败摘要不应把它们
                # 重新带回首屏活动记录。
                continue
            if key not in result:
                continue
            value = result[key]
            if isinstance(value, (str, int, float, bool)):
                rendered = _safe_ui_text(value)
                if rendered:
                    if key in {"visible", "enabled"} and isinstance(value, bool):
                        rendered = "是" if value else "否"
                    if key == "zone":
                        rendered = _ZONE_LABELS.get(rendered.casefold(), "互动部位")
                    label = _RESULT_FIELD_LABELS.get(key)
                    if label:
                        parts.append(f"{label}：{rendered}")
        return "，".join(parts) if parts else "已收到结果"
    status = _safe_ui_text(_status_value(result)).casefold()
    detail = _safe_ui_text(_detail_value(result))
    # 兼容旧宿主返回对象时也必须沿用普通用户文案；不能把
    # ``status=...，detail=...`` 这类内部协议格式泄露到活动记录。
    status_label = _RESULT_STATUS_LABELS.get(status, "处理中") if status else ""
    if status_label and detail:
        return f"{status_label}：{detail}"
    return status_label or detail or "已收到结果"


def _safe_error_message(error: BaseException) -> str:
    """保留可行动的短错误文本，同时隐藏结构化敏感内容。"""

    text = _safe_ui_text(error)
    return text or type(error).__name__


def _friendly_stream_text(value: object, *, limit: int = 4000) -> str:
    """把展示层状态标记翻译为中文，并拒绝内部协议赋值和敏感文本。

    流式气泡需要保留换行；直接复用 ``_safe_ui_text`` 会把换行压成空格，
    让碎碎念、工具状态和正文挤成一行。这里沿用同一敏感字段规则，但只
    规范化控制字符，不破坏合法的换行布局。
    """

    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = "".join(char for char in text if char == "\n" or char == "\t" or ord(char) >= 0x20)
    text = text.strip()
    if not text:
        return ""
    lowered = text.casefold()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        text = _SECRET_ASSIGNMENT.sub(r"\1***", text)
        if any(marker in text.casefold() for marker in _SECRET_MARKERS):
            return "敏感详情已隐藏"
    if _INTERNAL_PARAMETER_PATTERN.search(text) or _INTERNAL_PROTOCOL_PATTERN.search(text):
        return "敏感详情已隐藏"

    def replace_status(match: re.Match[str]) -> str:
        key = match.group(1).strip().casefold()
        label = _STREAM_STATUS_LABELS.get(key)
        return f"{label}：" if label else "状态："

    text = _STREAM_STATUS_PATTERN.sub(replace_status, text)
    # 工具安全摘要本身可能已经带有“已完成：/需要确认：”前缀；状态标记
    # 翻译后只保留一层，避免控制台出现重复标签。
    for prefix in (
        "等待确认：需要确认：",
        "已完成：已完成：",
        "未完成：未完成：",
        "正在执行：正在执行：",
        "已拒绝：已拒绝：",
    ):
        label = prefix.split("：", 1)[0]
        text = text.replace(prefix, f"{label}：")
    if len(text) > limit:
        text = text[: max(0, limit - 1)] + "…"
    return text


def _configuration_model_status(values: Mapping[str, Any]) -> tuple[bool, str]:
    """从脱敏配置判断渠道是否具备地址和模型，并给出下一步。"""

    llm = values.get("llm")
    channels = llm.get("channels") if isinstance(llm, Mapping) else None
    if not isinstance(channels, list) or not channels:
        return False, "模型渠道未配置：点击“配置模型”填写渠道"
    ready = 0
    complete = 0
    for raw_channel in channels:
        if not isinstance(raw_channel, Mapping):
            continue
        if not str(raw_channel.get("id", raw_channel.get("channel_id", "")) or "").strip():
            continue
        if not str(raw_channel.get("protocol", "") or "").strip():
            continue
        enabled = raw_channel.get("enabled", raw_channel.get("enable", True))
        if isinstance(enabled, str):
            enabled = enabled.strip().casefold() not in {"false", "no", "off", "0"}
        if not bool(enabled):
            continue
        base_url = str(raw_channel.get("base_url", raw_channel.get("url", "")) or "").strip()
        model = str(raw_channel.get("model", "") or "").strip()
        if not model:
            models = raw_channel.get("models")
            if isinstance(models, (list, tuple)) and models:
                model = str(models[0] or "").strip()
        if not base_url or not model:
            continue
        complete += 1
        # 只用脱敏字段复用同一渠道校验；不读取或展开 api_key。
        channel_values = dict(raw_channel)
        channel_values.pop("api_key", None)
        channel_values["model"] = model
        try:
            ModelSetupDraft.from_mapping(channel_values).validate()
        except ValueError:
            continue
        ready += 1
    if ready:
        return True, f"模型渠道已就绪（{ready} 个）"
    if complete:
        return False, "模型渠道未就绪：点击“配置模型”检查协议、地址和模型"
    return False, "模型渠道未就绪：点击“配置模型”补齐 base_url 和 model"


def _friendly_model_message(value: str) -> str:
    """把宿主遗留的命令行指引归一为控制台点击路径。"""

    text = str(value or "").strip()
    marker = "模型渠道未配置："
    if marker not in text:
        return text
    prefix, suffix = text.split(marker, 1)
    if "meapet-wizard" in suffix or "MEAPET_API_BASE" in suffix:
        return f"{prefix}{marker}点击“配置模型”填写渠道"
    return text


_OPERATION_LABELS = {
    "foreground_window": "前台窗口读取",
    "list_processes": "进程读取",
    "module_status": "模块检查",
    "api_audit": "调用审计",
    "log_records": "运行日志",
    "mcp_content": "MCP 内容浏览",
    "show_pet": "显示桌宠",
    "toggle_visibility": "显示/隐藏桌宠",
    "toggle_always_on_top": "窗口置顶",
    "set_window_locked": "锁定窗口",
    "toggle_window_lock": "锁定窗口",
    "restore_click_through": "恢复点击",
    "set_display_size": "显示大小",
    "select_renderer_model": "模型切换",
    "move_to": "移动桌宠",
    "click_through": "点击穿透",
    "position": "当前位置读取",
    "movement_bounds": "屏幕范围读取",
    "submit": "消息提交",
    "microphone_toggle": "语音输入",
    "stop": "停止生成",
    "retry": "重试",
    "expression": "表情",
    "expression_request": "表情编排",
    "motion": "动作",
    "motion_request": "动作编排",
    "approve_approval": "审批",
    "deny_approval": "审批",
    "config_save": "配置保存",
    "config_validate": "配置校验",
    "open_web_console": "网页控制台",
    "restart_application": "重启桌宠",
}

_ACTION_LABELS = {
    "neutral": "自然",
    "happy": "开心",
    "sad": "难过",
    "curious": "好奇",
    "surprised": "惊讶",
    "shy": "害羞",
    "idle": "待机",
    "blink": "眨眼",
    "wave": "挥手",
    "walk": "走动",
    "angry": "生气",
}


def _action_label(kind: str, value: object) -> str:
    """把内部动作名映射为点击式按钮文案。"""

    text = str(value or "").strip()
    if text in _ACTION_LABELS:
        return _ACTION_LABELS[text]
    safe_text = _safe_ui_text(text, limit=28)
    if safe_text:
        return safe_text
    return "自定义表情" if kind == "expression" else "自定义动作"


def _public_action_label(kind: str, value: object) -> str:
    """返回普通用户看到的动作标签，不回显模型原始能力名。"""

    text = str(value or "").strip()
    normalized = text.casefold()
    friendly_values = {label.casefold(): label for label in _ACTION_LABELS.values()}
    if normalized in friendly_values:
        return friendly_values[normalized]
    if normalized in _ACTION_LABELS:
        return _ACTION_LABELS[normalized]
    # model3 常用单字母动作组（例如 A/B）不是敏感字段，但直接显示为两
    # 个“自定义动作”会让用户无法区分；只为短、受限的组名提供稳定标签，
    # 其它内部名称仍按通用文案隐藏。
    if len(text) <= 3 and text.isascii() and text.isalnum():
        return f"表情 {text}" if kind == "expression" else f"动作 {text}"
    return "自定义表情" if kind == "expression" else "自定义动作"


def _operation_label(name: str) -> str:
    """把内部回调名转换成控制台可读的操作名称。"""

    # 未登记的回调名属于宿主内部实现细节，不应原样出现在普通用户界面。
    return _OPERATION_LABELS.get(str(name), "此操作")


_INVOKE_STATUS_LABELS = {
    "failed": "未完成",
    "error": "出错",
    "unavailable": "不可用",
    "degraded": "部分完成",
    "pending": "等待中",
    "rejected": "已拒绝",
    "cancelled": "已取消",
}

_MODULE_CENTER_SPECS: tuple[
    tuple[str, str, str, tuple[str, ...], str],
    ...,
] = (
    ("model", "模型路由", "forward", ("llm",), "渠道、协议适配器与模型就绪状态"),
    ("tts", "语音输出", "volume", ("tts",), "内置语音工作进程、音色与模型状态"),
    ("asr", "语音识别", "volume", ("asr", "advanced"), "本地识别工作进程与模型状态"),
    ("memory", "记忆与 RAG", "save", ("memory",), "摘要、召回与向量索引状态"),
    ("diary", "桌宠日记", "save", ("memory",), "公开日记与桌宠私密日记的隔离状态"),
    ("tools", "工具与权限", "warning", ("tools",), "工具注册、审批与权限状态"),
    ("mcp", "MCP 外部服务", "forward", ("mcp",), "MCP 客户端连接与可用数量"),
    ("scheduler", "调度与触发器", "play", ("scheduler",), "定时任务、触发器与持久化状态"),
    ("behavior", "自主行为", "play", ("behavior",), "随机行为、移动与当前行为阶段"),
    (
        "proactive",
        "模型主动行为",
        "forward",
        ("proactive", "scheduler"),
        "事件规则、预算、去重与模型主动工具循环",
    ),
    ("renderer", "渲染引擎", "file", ("rendering",), "Live2D、OpenGL 或 Vulkan 渲染状态"),
    ("window", "窗口事务", "info", ("ui",), "置顶、穿透、锁定与窗口切换事务"),
    ("watcher", "窗口观察", "info", ("watcher",), "前台窗口观察与光标跟踪状态"),
    ("logging", "统一日志", "file", ("logging",), "日志等级、输出与轮换状态"),
    ("ipc", "IPC 工作进程", "refresh", ("advanced",), "本地模块工作进程与通信状态"),
)

_MODULE_TERMINAL_SUCCESS = {
    "available",
    "completed",
    "ok",
    "ready",
    "running",
    "success",
}
_MODULE_PENDING_STATES = {
    "busy",
    "initializing",
    "loading",
    "pending",
    "queued",
    "requested",
    "starting",
}
_MODULE_DEGRADED_STATES = {"degraded", "disabled", "idle", "stopped"}
_MODULE_UNAVAILABLE_STATES = {"closed", "error", "failed", "unavailable"}


def _friendly_invoke_status(value: object) -> str:
    """把宿主状态枚举转换成普通用户可读的短文案。"""

    normalized = str(value or "").strip().casefold()
    return _INVOKE_STATUS_LABELS.get(normalized, "处理中")


def _safe_approval_identity(value: object) -> str:
    """将审批身份映射为用户标签，拒绝回显 namespace:name。"""

    text = _safe_ui_text(value, limit=80)
    if not text:
        return "需要确认的操作"
    normalized = text.casefold()
    prefix = normalized.split(":", 1)[0]
    labels = {
        "pet": "桌宠操作",
        "desktop": "桌面操作",
        "system": "系统操作",
        "mcp": "外部服务操作",
        "network": "网络操作",
    }
    if prefix in labels:
        return labels[prefix]
    if any(
        marker in normalized for marker in ("identity", "arguments", "command_line", "tool_call")
    ):
        return "需要确认的操作"
    return text


def _safe_approval_summary(value: object) -> str:
    """保留简短说明，同时过滤工具参数和内部调用标识。"""

    text = _safe_ui_text(value, limit=180)
    if not text:
        return "需要确认该操作"
    normalized = text.casefold()
    if any(
        marker in normalized
        for marker in (
            "identity",
            "arguments",
            "command_line",
            "tool_call",
            "payload",
            "system:",
            "desktop:",
            "pet:",
            "mcp:",
        )
    ):
        return "需要确认该操作"
    return text


def _safe_module_name(value: object, *, fallback: str = "未报告") -> str:
    """返回可公开的模块/模型逻辑名，拒绝路径、端点和凭据片段。"""

    text = _safe_ui_text(value, limit=80).strip()
    lowered = text.casefold()
    if not text or any(marker in lowered for marker in _SECRET_MARKERS):
        return fallback
    if any(marker in text for marker in ("/", "\\", "://", "${", "=")):
        return fallback
    return text


def _module_public_payload(snapshot: Mapping[str, object], module_id: str) -> dict[str, object]:
    """提取模块中心允许消费的状态映射，不展开任意嵌套诊断。"""

    runtime_value = snapshot.get("runtime")
    runtime = runtime_value if isinstance(runtime_value, Mapping) else {}
    payload: dict[str, object] = {}
    runtime_item = runtime.get(module_id)
    if isinstance(runtime_item, Mapping):
        payload.update(runtime_item)
    top_item = snapshot.get(module_id)
    if isinstance(top_item, Mapping):
        payload.update(top_item)

    if module_id == "model":
        adapters = snapshot.get("adapters")
        if isinstance(adapters, Sequence) and not isinstance(adapters, (str, bytes, bytearray)):
            payload["adapter_count"] = len(adapters)
    elif module_id == "tools":
        tools = snapshot.get("tools")
        if isinstance(tools, Sequence) and not isinstance(tools, (str, bytes, bytearray)):
            payload["tool_count"] = len(tools)
    elif module_id == "window" and not payload:
        platform = snapshot.get("platform")
        if isinstance(platform, Mapping):
            payload.update(platform)
    return payload


def _module_public_state(payload: Mapping[str, object]) -> str:
    """将各模块的固定状态字段归一成模块中心的五种可见终态。"""

    raw_status = str(payload.get("status", payload.get("state", "")) or "").casefold()
    if payload.get("pending") is True:
        return "checking"
    if raw_status in _MODULE_PENDING_STATES:
        return "checking"
    if raw_status in _MODULE_TERMINAL_SUCCESS:
        return "available"
    if raw_status in _MODULE_UNAVAILABLE_STATES:
        return "unavailable"
    if raw_status in _MODULE_DEGRADED_STATES:
        return "degraded"

    health = payload.get("health")
    if isinstance(health, Mapping):
        health_state = _module_public_state(health)
        if health_state != "idle":
            return health_state
    for key in ("available", "ready", "running"):
        value = payload.get(key)
        if value is True:
            return "available"
    if payload.get("enabled") is False:
        return "degraded"
    if payload.get("available") is False:
        return "unavailable"
    if any(key in payload for key in ("adapter_count", "tool_count", "configured", "ready")):
        counts = (
            payload.get("adapter_count"),
            payload.get("tool_count"),
            payload.get("configured"),
            payload.get("ready"),
        )
        if any(
            isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in counts
        ):
            return "available"
        return "degraded"
    return "degraded" if payload else "idle"


def _module_public_detail(module_id: str, payload: Mapping[str, object]) -> str:
    """从白名单字段构造模块摘要，不回显原因文本、路径、端点或日志。"""

    def integer(key: str) -> int | None:
        value = payload.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if module_id == "model":
        adapters = integer("adapter_count")
        channels = integer("channel_count")
        pieces = []
        if channels is not None:
            pieces.append(f"{max(0, channels)} 个模型渠道")
        if adapters is not None:
            pieces.append(f"{max(0, adapters)} 个协议适配器")
        return " · ".join(pieces) if pieces else "等待路由状态"
    if module_id in {"tts", "asr"}:
        engine = _safe_module_name(payload.get("engine", payload.get("backend")))
        language = _safe_module_name(payload.get("language"), fallback="自动")
        if module_id == "tts":
            queue_size = integer("queue_size")
            suffix = f" · 队列 {max(0, queue_size)}" if queue_size is not None else ""
            playback = payload.get("playback")
            playback_suffix = ""
            if isinstance(playback, Mapping):
                playback_status = str(playback.get("status", "") or "").casefold()
                playback_label = (
                    "播放中"
                    if playback.get("active") is True
                    else "播放待命"
                    if playback.get("available") is True
                    else "播放不可用"
                )
                if playback_status in {"active", "ready", "unavailable"}:
                    playback_suffix = f" · {playback_label}"
            return f"后端：{engine} · 语言：{language}{suffix}{playback_suffix}"
        device = _safe_module_name(payload.get("device"), fallback="自动")
        capture_state = str(payload.get("capture_state", "") or "").casefold()
        capture_labels = {
            "disabled": "录音关闭",
            "idle": "录音待命",
            "waiting_playback": "等待播放停止",
            "permission_pending": "等待录音权限",
            "recording": "录音中",
            "captured": "等待识别",
            "transcribing": "识别中",
            "completed": "识别完成",
            "denied": "录音权限被拒绝",
            "unavailable": "录音不可用",
            "closed": "录音已关闭",
        }
        capture_suffix = (
            f" · {capture_labels[capture_state]}" if capture_state in capture_labels else ""
        )
        return f"后端：{engine} · 语言：{language} · 设备：{device}{capture_suffix}"
    if module_id == "memory":
        count = integer("vector_index_size")
        index = _safe_module_name(payload.get("lexical_index"), fallback="未报告")
        return f"索引：{index} · 向量 {max(0, count or 0)} 条"
    if module_id == "diary":
        count = integer("public_entry_count")
        enabled = "已启用" if payload.get("enabled") is True else "未启用"
        return f"{enabled} · 公开条目 {max(0, count or 0)} 条 · 私密条目不投影"
    if module_id == "tools":
        count = integer("tool_count")
        return f"已注册 {max(0, count or 0)} 项工具能力"
    if module_id == "mcp":
        configured = integer("configured")
        ready = integer("ready")
        return f"已连接 {max(0, ready or 0)} / {max(0, configured or 0)} 个服务"
    if module_id == "scheduler":
        tasks = integer("task_count")
        triggers = integer("trigger_count")
        return f"{max(0, tasks or 0)} 个任务 · {max(0, triggers or 0)} 个触发器"
    if module_id == "behavior":
        phase = _safe_module_name(payload.get("phase"), fallback="空闲")
        movement = "允许移动" if payload.get("movement_enabled") is True else "移动关闭"
        return f"阶段：{phase} · {movement}"
    if module_id == "proactive":
        rules = integer("rule_count")
        pending = integer("pending_events")
        hourly_used = integer("hourly_used")
        hourly_budget = integer("hourly_budget")
        return (
            f"规则 {max(0, rules or 0)} 条 · 等待 {max(0, pending or 0)} · "
            f"小时预算 {max(0, hourly_used or 0)}/{max(0, hourly_budget or 0)}"
        )
    if module_id == "renderer":
        backend = _safe_module_name(payload.get("backend"), fallback="未连接")
        return f"当前引擎：{backend}"
    if module_id == "window":
        topmost = "开启" if payload.get("always_on_top") is True else "关闭"
        confirmed = "已确认" if payload.get("topmost_confirmed") is True else "待确认"
        click_through = "开启" if payload.get("click_through") is True else "关闭"
        locked = "锁定" if payload.get("locked") is True else "可交互"
        return f"置顶：{topmost}（{confirmed}） · 穿透：{click_through} · {locked}"
    if module_id == "watcher":
        return "前台窗口观察运行中" if payload.get("running") is True else "前台窗口观察未运行"
    if module_id == "logging":
        level = _safe_module_name(payload.get("level"), fallback="未报告")
        if payload.get("file_configured") is True and payload.get("file_active") is not True:
            output = "文件输出不可用"
        elif payload.get("file_active") is True:
            output = (
                "文件输出与轮换已启用"
                if payload.get("rotation_enabled") is True
                else "文件输出已启用"
            )
        elif payload.get("console_active") is True:
            output = "控制台输出已启用"
        else:
            output = "日志输出未就绪"
        return f"等级：{level} · {output}"
    if module_id == "ipc":
        workers = payload.get("workers")
        count = (
            len(workers)
            if isinstance(workers, Sequence) and not isinstance(workers, (str, bytes, bytearray))
            else integer("worker_count")
        )
        return f"已登记 {max(0, count or 0)} 个本地工作进程"
    return "状态字段已脱敏"


def _module_public_ipc(payload: Mapping[str, object], *, module_id: str) -> str:
    """返回不包含命令行和路径的 IPC 工作进程摘要。"""

    ipc_value = payload if module_id == "ipc" else payload.get("ipc")
    if not isinstance(ipc_value, Mapping):
        return "IPC：未报告"
    status = str(ipc_value.get("status", ipc_value.get("state", "")) or "").casefold()
    running = ipc_value.get("running") is True
    ready = ipc_value.get("ready") is True
    pid = ipc_value.get("pid")
    if ready or status in _MODULE_TERMINAL_SUCCESS:
        label = "已就绪"
    elif running or status in _MODULE_PENDING_STATES:
        label = "启动中"
    elif status in _MODULE_UNAVAILABLE_STATES:
        label = "不可用"
    elif status in _MODULE_DEGRADED_STATES:
        label = "未运行"
    else:
        label = "未报告"
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
        return f"IPC：{label} · 进程 {pid}"
    return f"IPC：{label}"


def _module_public_model(payload: Mapping[str, object]) -> str:
    """返回模型逻辑名和加载状态，不显示模型目录。"""

    name = _safe_module_name(payload.get("model"), fallback="")
    readiness = next(
        (
            payload.get(key)
            for key in ("model_ready", "model_loaded")
            if isinstance(payload.get(key), bool)
        ),
        None,
    )
    if readiness is True:
        state = "已就绪"
    elif readiness is False:
        state = "未就绪"
    else:
        state = "未报告"
    return f"模型：{name} · {state}" if name else f"模型：{state}"


if pyside6_available:

    class PetConsoleWindow(QWidget):
        """可隐藏但不销毁的桌宠控制台窗口。"""

        hidden = Signal()
        closeRequested = Signal()
        commandSubmitted = Signal(str)
        stopRequested = Signal()
        conversationSubmitted = Signal(str)
        expressionRequested = Signal(str)
        motionRequested = Signal(str)
        moveRequested = Signal(int, int)
        showPetRequested = Signal()
        visibilityToggleRequested = Signal()
        alwaysOnTopRequested = Signal()
        windowLockRequested = Signal(bool)
        restoreClickThroughRequested = Signal()
        foregroundWindowRequested = Signal()
        processesRequested = Signal()
        moduleStatusRequested = Signal()
        rendererModelRequested = Signal(str)
        configurationRequested = Signal()
        statusChanged = Signal(str)
        operationTriggered = Signal(str)

        def __init__(
            self,
            *,
            callbacks: Mapping[str, Callback] | None = None,
            expressions: Sequence[str] = (),
            motions: Sequence[str] = (),
            configuration: Mapping[str, Any] | None = None,
            configuration_path: str | Path | None = None,
            config_callbacks: Mapping[str, Callback] | None = None,
            model_setup_positioner: Callback | None = None,
            microphone_device_loader: Callable[[], object] | None = None,
            developer_mode: bool = True,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self._callbacks = dict(callbacks or {})
            self._config_callbacks = dict(config_callbacks or {})
            self._model_setup_positioner = model_setup_positioner
            self._microphone_device_loader = microphone_device_loader
            self._configuration_path = (
                Path(configuration_path).expanduser().resolve()
                if configuration_path is not None
                else None
            )
            raw_configuration: Mapping[str, Any] | None = configuration
            source_configuration: Mapping[str, Any] | None = None
            if self._configuration_path is not None:
                try:
                    source_configuration = parse_editor_yaml(
                        self._configuration_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, ValueError):
                    source_configuration = None
            if raw_configuration is None:
                raw_configuration = source_configuration or {}
            elif source_configuration is not None:
                raw_configuration = preserve_secret_values(source_configuration, raw_configuration)
            self._configuration: dict[str, Any] = copy.deepcopy(dict(raw_configuration or {}))
            watcher_values = self._configuration.get("watcher")
            self._watcher_enabled = bool(
                watcher_values.get("enabled", False)
                if isinstance(watcher_values, Mapping)
                else False
            )
            try:
                self._configuration_session = (
                    ConfigurationEditSession(
                        self._configuration_path,
                        self._configuration,
                    )
                    if self._configuration_path is not None
                    else None
                )
            except (ConfigurationError, OSError):
                self._configuration_session = None
            self._allow_close = False
            self._last_snapshot_text = ""
            self._last_snapshot_mood = "neutral"
            # 记录正文、碎碎念、工具状态的独立增量。把三段内容拼成一个
            # ``rendered_text`` 后再按前缀追加，会在工具审批完成、最终正文
            # 到达时重复插入已经显示过的碎碎念；组件快照用于避免该回放。
            self._last_snapshot_components: tuple[str, str, str, str] | None = None
            self._snapshot_tail_component: int | None = None
            self._position_bounds: tuple[int, int, int, int] | None = None
            self._quick_expression_buttons: list[QPushButton] = []
            self._quick_motion_buttons: list[QPushButton] = []
            self._tts_profiles: tuple[str, ...] = ()
            self._tts_profile_rows: tuple[dict[str, object], ...] = ()
            self._tts_language = "zh"
            self._tts_selected_profile = ""
            self._tts_health_available: bool | None = None
            self._tts_health_pending = False
            self._tts_health_message = ""
            self._tts_health_engine = ""
            self._tts_options_signature: tuple[object, ...] | None = None
            self._tts_profile_host: QWidget | None = None
            self._tts_language_host: QWidget | None = None
            self._tts_status: QLabel | None = None
            self._advanced_actions_visible = False
            self._position_detail_visible = False
            self._model_setup_dialog: ModelSetupDialog | None = None
            self._model_test_future: object | None = None
            self._renderer_model_options: tuple[str, ...] = ()
            self._renderer_model_committed = ""
            self._renderer_model_pending = ""
            self._model_ready = False
            self._last_invoke_failed = False
            self._last_invoke_unavailable = False
            self._state_action_signature: tuple[object, ...] = ()
            self._topmost_status_signature: tuple[object, ...] = ()
            self._topmost_control_buttons: tuple[QPushButton, ...] = ()
            self._configuration_focus = False
            self._configuration_enabled = bool(
                configuration is not None
                or self._configuration_path is not None
                or self._config_callbacks
                or "config_save" in self._callbacks
                or "config_validate" in self._callbacks
            )
            self._configuration_panel: ConfigurationPanel | None = None
            self._configuration_placeholder: QWidget | None = None
            self._configuration_page_index = -1
            self._configuration_panel_building = False
            self._module_center_placeholder: QWidget | None = None
            self._module_center_page_index = -1
            self._module_center_loaded = False
            self._module_center_building = False
            self._module_grid_layout: QGridLayout | None = None
            self._module_cards: tuple[QFrame, ...] = ()
            self._module_columns = 0
            self._module_status_icons: dict[str, QLabel] = {}
            self._module_status_labels: dict[str, QLabel] = {}
            self._module_detail_labels: dict[str, QLabel] = {}
            self._module_meta_labels: dict[str, QLabel] = {}
            self._module_test_buttons: dict[str, QPushButton] = {}
            self._module_refresh_action: object | None = None
            self._module_check_generation = 0
            self._module_check_busy = False
            self._module_test_generations: dict[str, int] = {}
            self._module_test_counters: dict[str, int] = {}
            self._module_pending_snapshot: dict[str, object] | None = None
            self._microphone_state = "disabled"
            self._microphone_closed = False
            self._microphone_input_devices: tuple[AudioInputDeviceChoice, ...] = ()
            self._microphone_devices_loaded = False
            self._microphone_device_updating = False

            self.setObjectName("PetConsoleWindow")
            self.setWindowTitle("MeaPet 控制台")
            # 控制台是桌宠的恢复入口；在全屏/普通桌面应用上方保持可见，
            # 否则模型未配置时用户会看见桌宠却找不到配置窗口。
            self.setWindowFlags(Qt.Window | Qt.Tool | Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_DeleteOnClose, False)
            # 默认尺寸优先保证在 1280px 宽桌面上能与 420px 桌宠并排；
            # 低分辨率桌面仍必须允许缩到单列滚动布局，不能因最小尺寸
            # 强制超出屏幕而把恢复点击/配置模型入口藏到窗口外。
            self.setMinimumSize(420, 500)
            self.resize(980, 760)
            self._md3_theme, _stylesheet = _console_stylesheet(self._configuration)
            self.setProperty("md3Role", "root")
            self._fancy_style_controller = FancyStyleController(self._md3_theme, self)

            def card(object_name: str) -> tuple[QFrame, QVBoxLayout]:
                frame = FancyCard(
                    elevated=object_name == "consoleHeaderCard",
                    theme=self._md3_theme,
                )
                frame.setObjectName(object_name)
                frame.setProperty(
                    "md3Role",
                    "card-elevated" if object_name == "consoleHeaderCard" else "card-outlined",
                )
                self._fancy_style_controller.register(frame)
                layout = frame.content_layout
                layout.setSpacing(12)
                return frame, layout

            def heading(title: str, hint: str = "") -> QVBoxLayout:
                block = QVBoxLayout()
                block.setSpacing(2)
                title_label = QLabel(title)
                title_label.setObjectName("consoleSectionTitle")
                block.addWidget(title_label)
                if hint:
                    hint_label = QLabel(hint)
                    hint_label.setObjectName("consoleSectionHint")
                    hint_label.setWordWrap(True)
                    block.addWidget(hint_label)
                return block

            def button(
                text: str,
                object_name: str,
                slot: Callback,
                tooltip: str,
                *,
                role: str = "",
            ) -> QPushButton:
                item = QPushButton(text)
                item.setObjectName(object_name)
                if role:
                    item.setProperty("role", role)
                item.setProperty(
                    "md3Role",
                    {
                        "primary": "filled",
                        "quick": "chip",
                        "quiet": "text",
                        "danger": "danger",
                    }.get(role, "tonal"),
                )
                item.setToolTip(tooltip)
                item.setAccessibleName(text)
                item.clicked.connect(slot)
                return item

            # 顶部身份区：状态、下一步提示和配置中心入口始终可见。
            header_card, header_layout = card("consoleHeaderCard")
            header_row = QHBoxLayout()
            header_copy = QVBoxLayout()
            eyebrow = QLabel("MEAPET")
            eyebrow.setObjectName("consoleEyebrow")
            self._header_eyebrow = eyebrow
            header_copy.addWidget(eyebrow)
            title = QLabel("桌宠控制台")
            title.setObjectName("consoleTitleLabel")
            header_copy.addWidget(title)
            subtitle = QLabel("对话、桌宠、工具和设置已按任务分区。")
            subtitle.setObjectName("consoleSubtitleLabel")
            subtitle.setWordWrap(True)
            self._header_subtitle = subtitle
            header_copy.addWidget(subtitle)
            header_row.addLayout(header_copy, 1)
            config_button = button(
                "配置中心",
                "configCenterButton",
                self.show_settings,
                "打开分域配置中心；实时操作页不会展示完整 YAML。",
                role="quiet",
            )
            header_row.addWidget(config_button, 0, Qt.AlignTop)
            model_button = button(
                "配置模型",
                "configureModelButton",
                self.show_model_settings,
                "直接打开模型渠道配置，不需要手动查找 YAML 字段。",
                role="primary",
            )
            command_bar = FancyCommandBar(theme=self._md3_theme)
            command_bar.setObjectName("consoleCommandBar")
            self._command_bar = command_bar
            command_layout = command_bar.layout()
            if command_layout is None:
                raise RuntimeError("FancyCommandBar layout is unavailable")
            command_layout.insertWidget(0, config_button)
            command_layout.insertWidget(1, model_button)
            command_bar.addCommand(
                "刷新状态",
                lambda: self._run_control_action("module_status"),
                icon="refresh",
                tooltip="重新读取模型、语音、渲染与工具模块状态",
            )
            command_bar.addCommand(
                "显示桌宠",
                lambda: self._run_rescue_action("show_pet"),
                icon="home",
                tooltip="桌宠不可见时重新显示",
            )
            command_bar.addCommand(
                "恢复鼠标",
                lambda: self._run_rescue_action("restore_click_through"),
                icon="back",
                tooltip="关闭点击穿透并恢复桌宠交互",
            )
            self._fancy_style_controller.register(command_bar)
            header_row.addWidget(command_bar, 0, Qt.AlignTop)
            header_layout.addLayout(header_row)
            status_row = QHBoxLayout()
            # 这两个标签只保留兼容的状态读取接口；真正的可见状态由
            # ``FancyInfoBar`` 渲染。必须显式绑定到控制台且保持隐藏，
            # 否则页面切换时的 ``setVisible`` 会把无父 QLabel 提升为
            # 标题为启动脚本名的独立顶层窗口，并抢占桌面焦点。
            self._status = QLabel("就绪 · 可以开始对话或使用快捷动作", self)
            self._status.setObjectName("consoleStatusLabel")
            self._status.setProperty("state", "ready")
            self._status.setWordWrap(True)
            self._status.hide()
            self._hint = QLabel(
                "提示：点击快捷按钮即可执行，失败原因会显示在下方活动记录。",
                self,
            )
            self._hint.setObjectName("consoleHintLabel")
            self._hint.setWordWrap(True)
            self._hint.hide()
            self._status_info_bar = FancyInfoBar(
                "运行状态",
                self._status.text(),
                severity="success",
                closable=False,
                theme=self._md3_theme,
            )
            self._status_info_bar.setObjectName("consoleStatusInfoBar")
            self._fancy_style_controller.register(self._status_info_bar)
            header_layout.addWidget(self._status_info_bar)
            # 运行时交互快照投影的下一步动作。按钮只携带脱敏 action kind
            # 和必要的审批标识，不把工具参数复制到 Qt；模型配置、停止、
            # 批准/拒绝和重试会随状态动态出现，避免用户在页面里找按钮。
            self._state_actions_host = QWidget()
            self._state_actions_host.setObjectName("stateActionsHost")
            self._state_actions_layout = QHBoxLayout(self._state_actions_host)
            self._state_actions_layout.setContentsMargins(0, 0, 0, 0)
            self._state_actions_layout.setSpacing(8)
            self._state_actions_host.setVisible(False)
            status_row.addWidget(self._state_actions_host, 0)
            header_layout.addLayout(status_row)

            # 救援操作栏始终位于首屏：桌宠被隐藏、置顶失效或点击穿透后，
            # 用户不必滚动到页面底部即可恢复。按钮仍复用同一宿主回调，
            # 不在 Qt 层复制窗口状态逻辑。
            rescue_card, rescue_layout = card("consoleRescueCard")
            rescue_header = QHBoxLayout()
            rescue_title = QLabel("桌宠状态与恢复")
            rescue_title.setObjectName("consoleSectionTitle")
            self._rescue_title = rescue_title
            rescue_header.addWidget(rescue_title)
            rescue_header.addStretch(1)
            self._rescue_status = QLabel("可见性、置顶和鼠标输入可直接恢复")
            self._rescue_status.setObjectName("consoleSectionHint")
            self._rescue_status.setWordWrap(True)
            rescue_header.addWidget(self._rescue_status)
            rescue_layout.addLayout(rescue_header)
            self._pet_feedback = QLabel("互动反馈：点击猫猫头或身体后会显示部位和好感度变化")
            self._pet_feedback.setObjectName("petInteractionFeedback")
            self._pet_feedback.setProperty("state", "idle")
            self._pet_feedback.setWordWrap(True)
            rescue_layout.addWidget(self._pet_feedback)
            rescue_actions = QGridLayout()
            rescue_actions.setHorizontalSpacing(8)
            rescue_actions.setVerticalSpacing(8)

            def rescue_button(
                text: str,
                object_name: str,
                action: str,
                tooltip: str,
            ) -> QPushButton:
                item = button(text, object_name, lambda: self._run_rescue_action(action), tooltip)
                item.setObjectName(object_name)
                item.setProperty("role", "quick")
                item.setProperty("rescueAction", action)
                item.setMinimumHeight(34)
                return item

            rescue_items = (
                rescue_button(
                    "显示桌宠",
                    "headerShowPetButton",
                    "show_pet",
                    "桌宠不可见时显示桌宠",
                ),
                rescue_button(
                    "恢复鼠标",
                    "headerRestoreClickThroughButton",
                    "restore_click_through",
                    "关闭点击穿透，恢复拖动、点击和右键菜单",
                ),
                rescue_button(
                    "窗口置顶",
                    "headerAlwaysOnTopButton",
                    "toggle_always_on_top",
                    "切换桌宠是否保持在其他窗口上方",
                ),
                rescue_button(
                    "移到屏幕中央",
                    "headerCenterPetButton",
                    "center_pet",
                    "把桌宠移动到当前屏幕可用区域中央",
                ),
                rescue_button(
                    "网页控制台",
                    "headerWebConsoleButton",
                    "open_web_console",
                    "打开只包含点击操作的网页控制台",
                ),
            )
            self._rescue_buttons = {
                str(item.property("rescueAction")): item for item in rescue_items
            }
            self._rescue_items = tuple(rescue_items)
            self._rescue_actions_layout = rescue_actions
            self._rescue_columns = 5
            # 五个恢复入口保持在同一行，避免紧凑/小屏布局出现第二行
            # 被固定高度裁切，用户始终能看到“网页控制台”这个救援入口。
            rescue_columns = 5
            for index, item in enumerate(rescue_items):
                rescue_actions.addWidget(item, index // rescue_columns, index % rescue_columns)
            for column in range(rescue_columns):
                rescue_actions.setColumnStretch(column, 1)
            rescue_layout.addLayout(rescue_actions)

            # 对话区是第一主行动，保留历史输出但明确空状态。
            conversation_card, conversation_layout = card("conversationCard")
            conversation_layout.addLayout(
                heading("对话", "发送消息开始陪伴；正在生成时可随时停止。")
            )
            self._empty_state = QLabel("还没有活动记录\n发送一句话，或点一个快捷互动开始。")
            self._empty_state.setObjectName("consoleEmptyState")
            self._empty_state.setAlignment(Qt.AlignCenter)
            self._empty_state.setWordWrap(True)
            conversation_layout.addWidget(self._empty_state)
            self._output = QPlainTextEdit()
            self._output.setObjectName("conversationOutput")
            self._output.setReadOnly(True)
            self._output.setPlaceholderText("模型回复、工具状态和系统事件会显示在这里。")
            self._output.setMaximumBlockCount(1000)
            self._output.setMinimumHeight(120)
            self._output.setMaximumHeight(220)
            self._output.textChanged.connect(self._sync_empty_state)
            conversation_layout.addWidget(self._output)

            self._input = QLineEdit()
            self._input.setObjectName("conversationInput")
            self._input.setPlaceholderText("输入消息，回车发送")
            self._input.returnPressed.connect(self._submit)
            send = button(
                "发送",
                "sendMessageButton",
                self._submit,
                "发送消息（Enter）",
                role="primary",
            )
            send.setProperty("legacyObjectName", "consolePrimary")
            stop = button(
                "停止生成",
                "stopConversationButton",
                self._stop,
                "停止当前对话和语音输出",
                role="danger",
            )
            microphone = button(
                "开始录音",
                "microphoneToggleButton",
                self._toggle_microphone,
                "点击后检查麦克风权限并开始录音；再次点击停止并识别",
            )
            microphone.setProperty("role", "quick")
            microphone.setEnabled(False)
            self._send_button = send
            self._stop_button = stop
            self._microphone_button = microphone
            input_row = QGridLayout()
            input_row.setSpacing(8)
            self._conversation_input_layout = input_row
            self._conversation_input_items = (self._input, microphone, send, stop)
            self._conversation_input_columns = 4
            input_row.addWidget(self._input, 0, 0)
            input_row.addWidget(microphone, 0, 1)
            input_row.addWidget(send, 0, 2)
            input_row.addWidget(stop, 0, 3)
            input_row.setColumnStretch(0, 1)
            input_row.setColumnStretch(1, 0)
            input_row.setColumnStretch(2, 0)
            input_row.setColumnStretch(3, 0)
            conversation_layout.addLayout(input_row)

            device_label = QLabel("输入设备")
            device_label.setObjectName("microphoneDeviceLabel")
            device_label.setAccessibleName("麦克风输入设备标签")
            device_selector = AudioInputSelector()
            device_selector.setObjectName("microphoneDeviceSelector")
            device_selector.set_selected_device_id(self._configured_microphone_device_id())
            device_selector.refreshRequested.connect(self._load_microphone_input_devices)
            device_selector.currentIndexChanged.connect(self._select_microphone_input_device)
            refresh_devices = button(
                "刷新设备",
                "microphoneDeviceRefreshButton",
                self._load_microphone_input_devices,
                "读取当前可用输入设备；不会请求麦克风权限",
            )
            can_manage_devices = bool(
                self._configuration_enabled and callable(self._microphone_device_loader)
            )
            device_selector.setEnabled(can_manage_devices)
            refresh_devices.setEnabled(can_manage_devices)
            device_layout = QGridLayout()
            device_layout.setSpacing(8)
            device_layout.addWidget(device_label, 0, 0)
            device_layout.addWidget(device_selector, 0, 1)
            device_layout.addWidget(refresh_devices, 0, 2)
            device_layout.setColumnStretch(0, 0)
            device_layout.setColumnStretch(1, 1)
            device_layout.setColumnStretch(2, 0)
            self._microphone_device_label = device_label
            self._microphone_device_selector = device_selector
            self._microphone_device_refresh_button = refresh_devices
            self._microphone_device_layout = device_layout
            self._microphone_device_layout_columns = 3
            conversation_layout.addLayout(device_layout)
            self._microphone_status = QLabel("语音输入未启用")
            self._microphone_status.setObjectName("microphoneStatusLabel")
            self._microphone_status.setAccessibleName("语音输入状态")
            self._microphone_status.setProperty("state", "disabled")
            self._microphone_status.setWordWrap(True)
            conversation_layout.addWidget(self._microphone_status)

            # 快捷互动区：能力列表以按钮呈现；自定义名称收在按需展开的编辑区。
            quick_card, quick_layout = card("quickActionsCard")
            quick_layout.addLayout(
                heading(
                    "快捷互动",
                    "表情和动作由当前渲染器直接执行，不依赖对话模型 API；桌面观察按权限运行。",
                )
            )
            context_actions = QGridLayout()
            context_actions.setHorizontalSpacing(8)
            context_actions.setVerticalSpacing(8)
            foreground_quick = button(
                "看看当前窗口",
                "foregroundQuickButton",
                lambda: self._run_control_action("foreground_window"),
                "读取当前前台窗口和正在使用的程序摘要",
            )
            processes_quick = button(
                "查看进程摘要",
                "processesQuickButton",
                lambda: self._run_control_action("list_processes"),
                "读取当前运行进程摘要",
            )
            window_lock_quick = button(
                "锁定窗口",
                "windowLockQuickButton",
                self._toggle_window_lock_quick,
                "锁定后不接收鼠标，但继续追踪光标和执行自主行为",
            )
            self._window_lock_quick_button = window_lock_quick
            self._context_action_items = (
                foreground_quick,
                processes_quick,
                window_lock_quick,
            )
            self._context_actions_layout = context_actions
            self._context_action_columns = 3
            for index, item in enumerate(self._context_action_items):
                context_actions.addWidget(item, index // 3, index % 3)
            for column in range(3):
                context_actions.setColumnStretch(column, 1)
            quick_layout.addLayout(context_actions)
            self._observation_status = QLabel("桌面观察：点击上方按钮读取当前窗口或进程摘要。")
            self._observation_status.setObjectName("desktopObservationStatus")
            self._observation_status.setWordWrap(True)
            quick_layout.addWidget(self._observation_status)
            self._renderer_status = QLabel("渲染：正在探测")
            self._renderer_status.setObjectName("rendererStatus")
            self._renderer_status.setWordWrap(True)
            quick_layout.addWidget(self._renderer_status)
            self._renderer_model_status = QLabel("Live2D 模型：自动选择")
            self._renderer_model_status.setObjectName("rendererModelStatus")
            self._renderer_model_status.setWordWrap(True)
            self._renderer_model_status.setToolTip(
                "模型键来自资源根目录内通过 model3 引用校验的描述文件。"
            )
            quick_layout.addWidget(self._renderer_model_status)
            # 模型选择在窄屏时改为纵向堆叠；如果保留水平行，组合框和
            # “应用模型”按钮会把操作卡最小宽度撑到视口之外，进而让
            # 对话输入和快捷按钮出现横向溢出。
            renderer_model_row = QGridLayout()
            renderer_model_row.setHorizontalSpacing(8)
            renderer_model_row.setVerticalSpacing(6)
            self._renderer_model_row = renderer_model_row
            renderer_model_label = QLabel("模型选择")
            renderer_model_label.setObjectName("consoleMetaLabel")
            self._renderer_model_label = renderer_model_label
            self._renderer_model_selector = QComboBox()
            self._renderer_model_selector.setObjectName("rendererModelSelector")
            self._renderer_model_selector.setMinimumHeight(34)
            self._renderer_model_selector.setMinimumWidth(0)
            self._renderer_model_selector.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            self._renderer_model_selector.setToolTip(
                "选择资源目录内已通过 model3 引用校验的 Live2D 模型。"
            )
            self._renderer_model_selector.currentIndexChanged.connect(
                lambda _index: self._update_renderer_model_button()
            )
            renderer_model_button = button(
                "应用模型",
                "applyRendererModelButton",
                self._select_renderer_model,
                "在 Web Live2D 页面内切换模型；其它后端保存后需重启。",
            )
            self._renderer_model_apply_button = renderer_model_button
            renderer_model_button.setMinimumWidth(0)
            renderer_model_button.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
            self._renderer_model_widgets = (
                renderer_model_label,
                self._renderer_model_selector,
                renderer_model_button,
            )
            quick_layout.addLayout(renderer_model_row)
            restart_renderer_button = button(
                "重启并应用渲染引擎",
                "restartRendererButton",
                lambda: self._run_control_action("restart_application"),
                "OpenGL 与 Vulkan 使用不同全局图形 API；保存后点击此处安全重启切换。",
                role="quiet",
            )
            quick_layout.addWidget(restart_renderer_button)
            tts_header = QHBoxLayout()
            tts_title = QLabel("语音输出")
            tts_title.setObjectName("consoleMetaLabel")
            tts_header.addWidget(tts_title)
            tts_header.addStretch(1)
            self._tts_status = QLabel("中文 · 默认语音")
            self._tts_status.setObjectName("consoleMetaLabel")
            self._tts_status.setWordWrap(True)
            tts_header.addWidget(self._tts_status)
            quick_layout.addLayout(tts_header)
            self._tts_language_host = QWidget()
            self._tts_language_host.setObjectName("ttsLanguageActions")
            self._tts_language_layout = QGridLayout(self._tts_language_host)
            self._tts_language_layout.setContentsMargins(0, 0, 0, 0)
            self._tts_language_layout.setHorizontalSpacing(8)
            self._tts_language_layout.setVerticalSpacing(8)
            quick_layout.addWidget(self._tts_language_host)
            self._tts_profile_host = QWidget()
            self._tts_profile_host.setObjectName("ttsProfileActions")
            self._tts_profile_layout = QGridLayout(self._tts_profile_host)
            self._tts_profile_layout.setContentsMargins(0, 0, 0, 0)
            self._tts_profile_layout.setHorizontalSpacing(8)
            self._tts_profile_layout.setVerticalSpacing(8)
            quick_layout.addWidget(self._tts_profile_host)
            expression_label = QLabel("表情")
            expression_label.setObjectName("consoleMetaLabel")
            quick_layout.addWidget(expression_label)
            self._expression_quick_host = QWidget()
            self._expression_quick_host.setObjectName("expressionQuickActions")
            self._expression_quick_layout = QGridLayout(self._expression_quick_host)
            self._expression_quick_layout.setContentsMargins(0, 0, 0, 0)
            self._expression_quick_layout.setHorizontalSpacing(8)
            self._expression_quick_layout.setVerticalSpacing(8)
            quick_layout.addWidget(self._expression_quick_host)
            self._expression_hint = QLabel()
            self._expression_hint.setObjectName("expressionCapabilityHint")
            self._expression_hint.setToolTip("能力列表由当前渲染器声明。")
            quick_layout.addWidget(self._expression_hint)
            motion_label = QLabel("动作")
            motion_label.setObjectName("consoleMetaLabel")
            quick_layout.addWidget(motion_label)
            self._motion_quick_host = QWidget()
            self._motion_quick_host.setObjectName("motionQuickActions")
            self._motion_quick_layout = QGridLayout(self._motion_quick_host)
            self._motion_quick_layout.setContentsMargins(0, 0, 0, 0)
            self._motion_quick_layout.setHorizontalSpacing(8)
            self._motion_quick_layout.setVerticalSpacing(8)
            quick_layout.addWidget(self._motion_quick_host)
            self._motion_hint = QLabel()
            self._motion_hint.setObjectName("motionCapabilityHint")
            self._motion_hint.setToolTip("能力列表由当前渲染器声明。")
            quick_layout.addWidget(self._motion_hint)

            self._advanced_action_toggle = button(
                "动作编排…",
                "customActionToggleButton",
                self._toggle_advanced_actions,
                "编排多个表情、持续时间、过渡和循环动作",
                role="quiet",
            )
            self._advanced_action_toggle.setVisible(True)
            quick_layout.addWidget(self._advanced_action_toggle)
            self._advanced_action_panel = QWidget()
            self._advanced_action_panel.setObjectName("advancedActionPanel")
            advanced_form = QFormLayout(self._advanced_action_panel)
            advanced_form.setContentsMargins(0, 0, 0, 0)
            advanced_form.setSpacing(8)
            advanced_hint = QLabel("开发者入口：普通使用请点击上方表情和动作按钮。")
            advanced_hint.setObjectName("developerActionHint")
            advanced_hint.setWordWrap(True)
            advanced_form.addRow("", advanced_hint)

            self._expression = QComboBox()
            self._expression.setObjectName("expressionSelector")
            self._expression.setMinimumHeight(34)
            self._expression.setEditable(True)
            self._expression.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            self._replace_action_combo(
                self._expression,
                _action_names(expressions, DEFAULT_EXPRESSION_NAMES),
                "neutral",
            )
            self._expression.setToolTip("选择或输入渲染器支持的表情名称。")
            expression_button = button(
                "应用表情",
                "applyExpressionButton",
                self._set_expression,
                "应用当前表情",
            )
            expression_row = QHBoxLayout()
            expression_row.addWidget(self._expression, 1)
            expression_row.addWidget(expression_button)
            advanced_form.addRow("表情", expression_row)

            self._motion = QComboBox()
            self._motion.setObjectName("motionSelector")
            self._motion.setMinimumHeight(34)
            self._motion.setEditable(True)
            self._motion.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            self._replace_action_combo(
                self._motion,
                _action_names(motions, DEFAULT_MOTION_NAMES),
                "idle",
            )
            self._motion.setToolTip("选择或输入渲染器支持的动作名称。")
            motion_button = button(
                "播放动作",
                "playMotionButton",
                self._play_motion,
                "播放当前动作",
            )
            motion_row = QHBoxLayout()
            motion_row.addWidget(self._motion, 1)
            motion_row.addWidget(motion_button)
            advanced_form.addRow("动作", motion_row)

            self._expression_sequence = QLineEdit()
            self._expression_sequence.setObjectName("expressionSequenceInput")
            self._expression_sequence.setPlaceholderText("例如：happy, curious, neutral")
            self._expression_sequence.setToolTip("按顺序填写当前渲染器支持的表情，最多 8 个。")
            advanced_form.addRow("多个表情", self._expression_sequence)
            self._expression_weights = QLineEdit()
            self._expression_weights.setObjectName("expressionWeightsInput")
            self._expression_weights.setPlaceholderText("例如：1, 0.7, 0.3（与表情顺序对应）")
            self._expression_weights.setToolTip(
                "可留空；填写时数量必须与表情一致，每项范围为 0 到 1。"
            )
            advanced_form.addRow("混合权重", self._expression_weights)
            self._expression_parameters = QLineEdit()
            self._expression_parameters.setObjectName("expressionParametersInput")
            self._expression_parameters.setPlaceholderText("例如：ParamAngleX=12, ParamEyeLOpen=1")
            self._expression_parameters.setToolTip(
                "可选 Live2D 参数；使用 参数名=数值 格式，应用于每个表情层。"
            )
            advanced_form.addRow("表情参数", self._expression_parameters)
            self._expression_restore = QLineEdit("neutral")
            self._expression_restore.setObjectName("expressionRestoreInput")
            self._expression_restore.setToolTip("编排结束后恢复的表情名称。")
            advanced_form.addRow("结束恢复", self._expression_restore)
            expression_options = QHBoxLayout()
            self._expression_mode = QComboBox()
            self._expression_mode.setObjectName("expressionModeSelector")
            self._expression_mode.addItem("依次播放", "sequence")
            self._expression_mode.addItem("参数混合", "blend")
            expression_options.addWidget(self._expression_mode)
            self._expression_duration = QDoubleSpinBox()
            self._expression_duration.setObjectName("expressionDurationSeconds")
            self._expression_duration.setRange(0.05, 120.0)
            self._expression_duration.setValue(1.5)
            self._expression_duration.setSuffix(" 秒/表情")
            expression_options.addWidget(self._expression_duration)
            self._expression_transition = QDoubleSpinBox()
            self._expression_transition.setObjectName("expressionTransitionSeconds")
            self._expression_transition.setRange(0.0, 10.0)
            self._expression_transition.setValue(0.2)
            self._expression_transition.setSuffix(" 秒过渡")
            expression_options.addWidget(self._expression_transition)
            self._expression_loop = QCheckBox("循环")
            self._expression_loop.setObjectName("expressionLoopToggle")
            expression_options.addWidget(self._expression_loop)
            expression_sequence_button = button(
                "播放表情编排",
                "playExpressionSequenceButton",
                self._play_expression_sequence,
                "按设置的持续时间和过渡播放多个表情",
            )
            expression_options.addWidget(expression_sequence_button)
            advanced_form.addRow("表情选项", expression_options)

            motion_options = QHBoxLayout()
            self._motion_duration = QDoubleSpinBox()
            self._motion_duration.setObjectName("motionDurationSeconds")
            self._motion_duration.setRange(0.05, 300.0)
            self._motion_duration.setValue(1.5)
            self._motion_duration.setSuffix(" 秒")
            motion_options.addWidget(self._motion_duration)
            self._motion_transition = QDoubleSpinBox()
            self._motion_transition.setObjectName("motionTransitionSeconds")
            self._motion_transition.setRange(0.0, 10.0)
            self._motion_transition.setValue(0.2)
            self._motion_transition.setSuffix(" 秒过渡")
            motion_options.addWidget(self._motion_transition)
            self._motion_loop = QCheckBox("循环")
            self._motion_loop.setObjectName("motionLoopToggle")
            motion_options.addWidget(self._motion_loop)
            motion_request_button = button(
                "按时长播放",
                "playMotionRequestButton",
                self._play_motion_request,
                "按指定时长、过渡和循环设置播放当前动作",
            )
            motion_options.addWidget(motion_request_button)
            advanced_form.addRow("动作选项", motion_options)
            self._motion_parameters = QLineEdit()
            self._motion_parameters.setObjectName("motionParametersInput")
            self._motion_parameters.setPlaceholderText("例如：ParamAngleX=8, ParamBodyAngleX=4")
            self._motion_parameters.setToolTip("可选 Live2D 参数；使用 参数名=数值 格式。")
            advanced_form.addRow("动作参数", self._motion_parameters)
            self._expression_button = expression_button
            self._motion_button = motion_button
            self._expression.currentTextChanged.connect(lambda _value: self._update_action_hints())
            self._motion.currentTextChanged.connect(lambda _value: self._update_action_hints())
            self._advanced_action_panel.setVisible(False)
            quick_layout.addWidget(self._advanced_action_panel)

            # 窗口恢复区把可见性、置顶和点击恢复放在同一组，避免用户迷失。
            window_card, window_layout = card("windowActionsCard")
            window_layout.addLayout(heading("窗口与交互", "桌宠不可见或点击穿透时，先从这里恢复。"))
            self._click_through = QCheckBox("点击穿透（控制台保持可见以便恢复）")
            self._click_through.setObjectName("clickThroughCheckBox")
            self._click_through.setToolTip("开启后桌宠接收不到鼠标；控制台会保持可见作为恢复入口。")
            self._click_through.toggled.connect(self._set_click_through)
            window_layout.addWidget(self._click_through)
            self._window_lock = QCheckBox("锁定窗口（不可点击，保留光标追踪和自主行动）")
            self._window_lock.setObjectName("windowLockCheckBox")
            self._window_lock.setToolTip(
                "锁定后桌宠不接收点击、拖动和右键；仍会追踪光标、播放动作并执行自主行为。"
            )
            ui_values = self._configuration.get("ui")
            initial_locked = bool(
                ui_values.get("window_locked", False) if isinstance(ui_values, Mapping) else False
            )
            self._window_lock.setChecked(initial_locked)
            self._click_through.setEnabled(not initial_locked)
            self._window_lock.toggled.connect(self._set_window_locked)
            window_layout.addWidget(self._window_lock)
            window_actions = QGridLayout()
            window_actions.setHorizontalSpacing(8)
            window_actions.setVerticalSpacing(8)
            show_pet = button(
                "显示桌宠",
                "showPetButton",
                lambda: self._run_control_action("show_pet"),
                "显示并激活桌宠窗口",
            )
            toggle_visibility = button(
                "显示/隐藏桌宠",
                "toggleVisibilityButton",
                lambda: self._run_control_action("toggle_visibility"),
                "切换桌宠窗口可见状态",
            )
            toggle_topmost = button(
                "切换窗口置顶",
                "toggleAlwaysOnTopButton",
                lambda: self._run_control_action("toggle_always_on_top"),
                "切换桌宠窗口置顶状态",
            )
            self._topmost_control_buttons = tuple(
                item
                for item in (
                    toggle_topmost,
                    self._rescue_buttons.get("toggle_always_on_top"),
                )
                if item is not None
            )
            restore_input = button(
                "恢复点击",
                "restoreClickThroughButton",
                lambda: self._run_control_action("restore_click_through"),
                "关闭点击穿透，恢复拖动和右键菜单",
            )
            for index, item in enumerate(
                (show_pet, toggle_visibility, toggle_topmost, restore_input)
            ):
                window_actions.addWidget(item, index // 2, index % 2)
            window_actions.setColumnStretch(0, 1)
            window_actions.setColumnStretch(1, 1)
            window_layout.addLayout(window_actions)
            quit_button = button(
                "退出程序",
                "quitProgramButton",
                lambda: self._invoke("quit"),
                "退出桌宠程序",
                role="danger",
            )
            window_layout.addWidget(quit_button)

            size_card, size_layout = card("displaySizeCard")
            size_layout.addLayout(
                heading("显示大小", "只需点选预设；桌宠会保持 Live2D/精灵等比绘制。")
            )
            size_actions = QHBoxLayout()
            for preset in ("small", "standard", "large"):
                size_button = button(
                    DISPLAY_SIZE_LABELS[preset],
                    f"displaySize{preset.title()}Button",
                    lambda _checked=False, selected=preset: self._set_display_size(selected),
                    f"将桌宠调整为{DISPLAY_SIZE_LABELS[preset]}尺寸",
                    role="primary" if preset == "standard" else "",
                )
                size_actions.addWidget(size_button, 1)
            size_layout.addLayout(size_actions)

            # 位置区默认只展示读取/居中；精确坐标按需展开。
            position_card, position_layout = card("positionCard")
            position_layout.addLayout(
                heading("位置", "读取当前位置或一键移到当前屏幕可用区域中央。")
            )
            position_actions = QHBoxLayout()
            refresh_position = button(
                "读取当前位置",
                "refreshPetPositionButton",
                self._refresh_position,
                "读取桌宠窗口左上角坐标",
            )
            center_position = button(
                "移到屏幕中央",
                "centerPetPositionButton",
                self._center_pet,
                "将桌宠移动到当前屏幕可用区域中央",
                role="primary",
            )
            position_actions.addWidget(refresh_position)
            position_actions.addWidget(center_position)
            position_actions.addStretch(1)
            position_layout.addLayout(position_actions)
            self._position_status = QLabel("位置尚未读取")
            self._position_status.setObjectName("consoleMetaLabel")
            position_layout.addWidget(self._position_status)
            exact_toggle = button(
                "精确位置…",
                "exactPositionToggleButton",
                self._toggle_position_detail,
                "展开 X/Y 坐标输入",
                role="quiet",
            )
            position_layout.addWidget(exact_toggle)
            self._position_detail = QWidget()
            self._position_detail.setObjectName("exactPositionPanel")
            exact_row = QHBoxLayout(self._position_detail)
            exact_row.setContentsMargins(0, 0, 0, 0)
            exact_row.addWidget(QLabel("X"))
            self._position_x = QSpinBox()
            self._position_x.setObjectName("positionXSpinBox")
            self._position_x.setRange(-100000, 100000)
            self._position_x.setSingleStep(10)
            exact_row.addWidget(self._position_x, 1)
            exact_row.addWidget(QLabel("Y"))
            self._position_y = QSpinBox()
            self._position_y.setObjectName("positionYSpinBox")
            self._position_y.setRange(-100000, 100000)
            self._position_y.setSingleStep(10)
            exact_row.addWidget(self._position_y, 1)
            move_button = button(
                "移动到",
                "movePetButton",
                self._move_pet,
                "按屏幕绝对坐标移动桌宠",
            )
            exact_row.addWidget(move_button)
            self._position_detail.setVisible(False)
            position_layout.addWidget(self._position_detail)

            # 工具审批区始终给出明确空状态，但不显示工具参数。
            approval_card, approval_layout = card("approvalCard")
            approval_layout.addLayout(
                heading("需要确认", "中风险工具会在这里等待你的批准；列表只显示安全摘要。")
            )
            self._approval_list = QListWidget()
            self._approval_list.setObjectName("approvalList")
            self._approval_list.setMinimumHeight(66)
            self._approval_list.setMaximumHeight(132)
            self._approval_list.setSelectionMode(self._approval_list.SelectionMode.SingleSelection)
            self._approval_signature: tuple[tuple[str, str], ...] = ()
            approval_layout.addWidget(self._approval_list)
            approval_buttons = QGridLayout()
            approve_button = button(
                "批准选中调用",
                "approveApprovalButton",
                self._approve_selected_approval,
                "批准选中的工具调用",
            )
            deny_button = button(
                "拒绝选中调用",
                "denyApprovalButton",
                self._deny_selected_approval,
                "拒绝选中的工具调用",
                role="danger",
            )
            self._grant_session = QCheckBox("批准后授予本会话同类工具")
            self._grant_session.setObjectName("grantSessionCheckBox")
            approval_buttons.addWidget(approve_button, 0, 0)
            approval_buttons.addWidget(deny_button, 0, 1)
            approval_buttons.addWidget(self._grant_session, 1, 0, 1, 2)
            approval_buttons.setColumnStretch(0, 1)
            approval_buttons.setColumnStretch(1, 1)
            approval_layout.addLayout(approval_buttons)

            # 诊断区只提供读取按钮，结果复用活动记录，不暴露命令行参数。
            diagnostics_card, diagnostics_layout = card("diagnosticsCard")
            diagnostics_layout.addLayout(heading("诊断读取", "按需检查桌面状态、进程和运行模块。"))
            diagnostic_actions = QHBoxLayout()
            foreground = button(
                "读取前台窗口",
                "foregroundWindowButton",
                lambda: self._run_control_action("foreground_window"),
                "读取当前前台窗口摘要",
            )
            processes = button(
                "读取进程",
                "listProcessesButton",
                lambda: self._run_control_action("list_processes"),
                "读取进程摘要",
            )
            modules = button(
                "检查模块",
                "moduleStatusButton",
                lambda: self._run_control_action("module_status"),
                "检查模型适配器、记忆索引、TTS、渲染和调度模块",
            )
            api_audit = button(
                "调用审计",
                "apiAuditButton",
                lambda: self._run_control_action("api_audit"),
                "读取模型渠道与工具执行的脱敏耗时摘要",
            )
            log_records = button(
                "运行日志",
                "logRecordsButton",
                lambda: self._run_control_action("log_records"),
                "查看最近结构化运行日志；内容已经脱敏并限制数量",
            )
            diagnostic_actions.addWidget(foreground)
            diagnostic_actions.addWidget(processes)
            diagnostic_actions.addWidget(modules)
            diagnostic_actions.addWidget(api_audit)
            diagnostic_actions.addWidget(log_records)
            diagnostics_layout.addLayout(diagnostic_actions)
            self._diagnostic_output = QPlainTextEdit()
            self._diagnostic_output.setObjectName("diagnosticOutput")
            self._diagnostic_output.setReadOnly(True)
            self._diagnostic_output.setPlaceholderText(
                "点击上方按钮读取桌面状态；结果会显示在这里。"
            )
            self._diagnostic_output.setMinimumHeight(72)
            self._diagnostic_output.setMaximumHeight(150)
            diagnostics_layout.addWidget(self._diagnostic_output)

            def scroll_page(
                object_name: str,
                title_text: str,
                hint_text: str,
            ) -> tuple[QScrollArea, QWidget, QGridLayout]:
                """创建带固定标题和无横向滚动的控制台导航页。"""

                scroll = QScrollArea()
                scroll.setObjectName(object_name)
                scroll.setWidgetResizable(True)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                content = QWidget()
                content.setObjectName(f"{object_name}Content")
                content.setSizePolicy(
                    QSizePolicy.Policy.Ignored,
                    QSizePolicy.Policy.Preferred,
                )
                layout = QGridLayout(content)
                layout.setContentsMargins(6, 4, 10, 12)
                layout.setHorizontalSpacing(12)
                layout.setVerticalSpacing(12)
                page_heading = QWidget()
                page_heading.setObjectName("consolePageHeading")
                page_heading_layout = QVBoxLayout(page_heading)
                page_heading_layout.setContentsMargins(4, 0, 4, 4)
                page_heading_layout.setSpacing(2)
                page_heading_layout.addLayout(heading(title_text, hint_text))
                layout.addWidget(page_heading, 0, 0, 1, 2)
                scroll.setWidget(content)
                return scroll, content, layout

            overview_scroll, _overview_content, overview_layout = scroll_page(
                "consoleOverviewScroll",
                "概览",
                "从一个入口进入对话、桌宠控制或工具审批。",
            )
            overview_card, overview_card_layout = card("consoleOverviewCard")
            overview_card_layout.addLayout(
                heading(
                    "欢迎回来",
                    "桌宠恢复操作始终位于上方；其余功能已按任务拆分到左侧导航。",
                )
            )
            overview_summary = QLabel(
                "开始对话会立即显示流式回复并播放语音；桌宠页负责动作、语音和窗口行为；"
                "工具页集中处理权限确认与诊断。"
            )
            overview_summary.setObjectName("consoleOverviewSummary")
            overview_summary.setWordWrap(True)
            overview_card_layout.addWidget(overview_summary)
            overview_actions = QGridLayout()
            self._overview_action_layout = overview_actions
            self._overview_action_columns = 0
            overview_actions.setHorizontalSpacing(8)
            overview_actions.setVerticalSpacing(8)
            open_conversation = button(
                "开始对话",
                "overviewConversationButton",
                lambda: self._navigate_console_page("对话"),
                "进入流式对话页面",
                role="primary",
            )
            open_pet = button(
                "桌宠与语音",
                "overviewPetButton",
                lambda: self._navigate_console_page("桌宠"),
                "进入桌宠动作、语音与窗口控制页面",
            )
            more_actions_toggle = button(
                "更多操作",
                "moreActionsToggleButton",
                self._toggle_secondary_actions,
                "进入工具审批和诊断页面",
                role="quiet",
            )
            self._overview_action_items = (open_conversation, open_pet, more_actions_toggle)
            for index, item in enumerate(self._overview_action_items):
                overview_actions.addWidget(item, 0, index)
                overview_actions.setColumnStretch(index, 1)
            overview_card_layout.addLayout(overview_actions)
            overview_layout.addWidget(overview_card, 1, 0, 1, 2)
            overview_layout.addWidget(rescue_card, 2, 0, 1, 2)
            overview_layout.setRowStretch(3, 1)

            control_scroll, _conversation_content, conversation_page_layout = scroll_page(
                "consoleControlScroll",
                "对话",
                "回复、碎碎念、工具进度和流式语音在这里同步呈现。",
            )
            conversation_page_layout.addWidget(conversation_card, 1, 0, 1, 2)
            conversation_page_layout.setRowStretch(2, 1)

            pet_scroll, _pet_content, pet_page_layout = scroll_page(
                "consolePetScroll",
                "桌宠",
                "集中管理表情、动作、语音、渲染模型、窗口行为与位置。",
            )
            self._pet_page_layout = pet_page_layout
            self._pet_page_items = (quick_card, window_card, size_card, position_card)

            tools_scroll, _tools_content, tools_page_layout = scroll_page(
                "secondaryActionsPanel",
                "工具与安全",
                "审批敏感操作并读取脱敏后的桌面与模块诊断。",
            )
            self._tools_page_layout = tools_page_layout
            self._tools_page_items = (approval_card, diagnostics_card)
            self._secondary_actions_panel = tools_scroll
            self._secondary_actions_toggle = more_actions_toggle

            tabs = FancyNavigationView(
                theme=self._md3_theme,
                compact=self.width() < 720,
                navigation_width=188,
            )
            tabs.setObjectName("consoleTabs")
            tabs.setAccessibleName("控制台功能导航")
            tabs.tabBar().setAccessibleName("控制台功能导航")
            self._fancy_style_controller.register(tabs)
            self._console_tabs = tabs
            tabs.currentChanged.connect(self._on_console_tab_changed)
            tabs.addPage(overview_scroll, "概览", "home", "查看控制台入口与运行摘要")
            self._conversation_page_index = tabs.addPage(
                control_scroll,
                "对话",
                "info",
                "进行流式对话并查看活动记录",
            )
            self._pet_page_index = tabs.addPage(
                pet_scroll,
                "桌宠",
                "play",
                "控制桌宠动作、语音、窗口和位置",
            )
            self._tools_page_index = tabs.addPage(
                tools_scroll,
                "工具与安全",
                "warning",
                "处理工具审批与安全诊断",
            )
            module_placeholder = QWidget()
            module_placeholder.setObjectName("moduleCenterHost")
            module_placeholder_layout = QVBoxLayout(module_placeholder)
            module_placeholder_layout.setContentsMargins(20, 20, 20, 20)
            module_placeholder_layout.setSpacing(10)
            module_loading_card = FancyCard(
                "模块中心",
                "首次打开时加载脱敏状态卡；外部探测始终交给运行时线程与 IPC。",
                icon="refresh",
                theme=self._md3_theme,
            )
            module_loading_card.setObjectName("moduleCenterLoadingCard")
            self._fancy_style_controller.register(module_loading_card)
            module_placeholder_layout.addWidget(module_loading_card)
            module_placeholder_layout.addStretch(1)
            self._module_center_placeholder = module_placeholder
            self._module_center_page_index = tabs.addPage(
                module_placeholder,
                "模块中心",
                "refresh",
                "查看、测试并配置本地与运行时模块",
            )
            self._configuration_developer_mode = bool(developer_mode)
            if self._configuration_enabled:
                placeholder = QWidget()
                placeholder.setObjectName("configurationCenterHost")
                placeholder_layout = QVBoxLayout(placeholder)
                placeholder_layout.setContentsMargins(20, 20, 20, 20)
                placeholder_layout.setSpacing(10)
                loading_card = FancyCard(
                    "配置中心",
                    "首次打开时加载设置控件，控制台启动不再等待未访问页面。",
                    icon="folder",
                    theme=self._md3_theme,
                )
                loading_card.setObjectName("configurationCenterLoadingCard")
                self._fancy_style_controller.register(loading_card)
                placeholder_layout.addWidget(loading_card)
                placeholder_layout.addStretch(1)
                self._configuration_placeholder = placeholder
                self._configuration_page_index = tabs.addPage(
                    placeholder,
                    "配置中心",
                    "folder",
                    "配置模型、语音、渲染、人设、记忆与主题",
                )
            tabs.setCurrentIndex(0)

            root = QVBoxLayout(self)
            root.setContentsMargins(16, 14, 16, 16)
            root.setSpacing(12)
            root.addWidget(header_card)
            root.addWidget(tabs, 1)
            self._update_action_hints()
            self._rebuild_quick_actions()
            self._set_context_action_columns(self._context_columns_for_width(self.width()))
            self._set_overview_action_columns(
                1 if self.width() < 340 else 2 if self.width() < 700 else 3
            )
            self._set_conversation_input_layout(self._conversation_columns_for_width(self.width()))
            self._set_microphone_device_layout(1 if self.width() < 380 else 3)
            self._set_quick_action_columns(self._quick_columns_for_width(self.width()))
            self._set_renderer_model_layout(self._renderer_model_columns_for_width(self.width()))
            self._set_console_page_columns(self._console_page_columns_for_width(self.width()))
            self._sync_empty_state()
            self.set_pending_approvals(())
            if self._configuration_enabled:
                self._refresh_model_status()
            if callable(self._callbacks.get("movement_bounds")):
                self._refresh_position_limits()
            self._fancy_style_controller.attach(
                self,
                extra_stylesheet=_console_component_stylesheet(self._md3_theme),
            )

        def _adapt_minimum_size_to_screen(self) -> None:
            """在低分辨率或高 DPI 屏幕上放宽硬性最小尺寸。"""

            try:
                screen = self.screen() or QGuiApplication.primaryScreen()
                if screen is None:
                    return
                area = screen.availableGeometry()
                # 预留窗口管理器边距；内容本身由滚动区和紧凑布局承载，
                # 最小尺寸不能阻止控制台进入可用屏幕范围。
                max_width = max(1, int(area.width()) - 16)
                max_height = max(1, int(area.height()) - 16)
                if self.minimumWidth() > max_width:
                    self.setMinimumWidth(max_width)
                if self.minimumHeight() > max_height:
                    self.setMinimumHeight(max_height)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return

        def showEvent(self, event: object) -> None:  # noqa: N802
            """首次显示时按实际屏幕放宽低分辨率窗口约束。"""

            super().showEvent(event)
            self._adapt_minimum_size_to_screen()

        def _set_rescue_columns(self, columns: int) -> None:
            """按可用宽度重排恢复按钮，避免窄屏按钮文字被裁切。"""

            layout = getattr(self, "_rescue_actions_layout", None)
            items = getattr(self, "_rescue_items", ())
            if layout is None or not items:
                return
            safe_columns = max(1, min(len(items), int(columns)))
            if safe_columns == getattr(self, "_rescue_columns", 0):
                return
            # ``takeAt`` 已足以解除布局位置；保留 QWidget 父对象可避免
            # 响应式重排期间短暂创建原生顶层窗口和焦点表面。
            while layout.count():
                layout.takeAt(0)
            for index, item in enumerate(items):
                item.setMinimumWidth(0)
                item.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                layout.addWidget(item, index // safe_columns, index % safe_columns)
            for column in range(safe_columns):
                layout.setColumnStretch(column, 1)
            self._rescue_columns = safe_columns

        def _navigate_console_page(self, title: str) -> bool:
            """按左侧导航的公开标题切换页面。"""

            tabs = getattr(self, "_console_tabs", None)
            if tabs is None:
                return False
            for index in range(tabs.count()):
                if tabs.tabText(index) == title:
                    tabs.setCurrentIndex(index)
                    if title == "对话" and bool(self.property("compactLayout")):
                        self._ensure_conversation_input_visible()
                        QTimer.singleShot(0, self._ensure_conversation_input_visible)
                    return True
            return False

        def _ensure_conversation_input_visible(self) -> None:
            """极小屏进入对话页时优先露出输入和三个操作按钮。"""

            scroll = self.findChild(QScrollArea, "consoleControlScroll")
            input_widget = getattr(self, "_input", None)
            if scroll is None or input_widget is None:
                return
            try:
                scrollbar = scroll.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                scroll.ensureWidgetVisible(input_widget, 0, 8)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                return

        def _ensure_module_center(self) -> bool:
            """首次访问时构造模块中心，避免增加控制台启动时延。"""

            if self._module_center_loaded:
                return True
            if self._module_center_building or self._module_center_placeholder is None:
                return False
            self._module_center_building = True
            try:
                placeholder = self._module_center_placeholder
                host_layout = placeholder.layout()
                if host_layout is None:
                    raise RuntimeError("module center host layout is unavailable")
                while host_layout.count():
                    item = host_layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()

                scroll = QScrollArea()
                scroll.setObjectName("consoleModuleCenterScroll")
                scroll.setWidgetResizable(True)
                scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                content = QWidget()
                content.setObjectName("consoleModuleCenterContent")
                content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                content_layout = QVBoxLayout(content)
                content_layout.setContentsMargins(6, 4, 10, 12)
                content_layout.setSpacing(12)

                heading_card = FancyCard(
                    "模块中心",
                    "状态通过运行时线程和 IPC 读取；页面不会显示路径、端点、密钥或原始日志。",
                    icon="refresh",
                    elevated=True,
                    theme=self._md3_theme,
                )
                heading_card.setObjectName("moduleCenterHeaderCard")
                self._fancy_style_controller.register(heading_card)
                heading_layout = heading_card.content_layout
                command_bar = FancyCommandBar(theme=self._md3_theme)
                command_bar.setObjectName("moduleCenterCommandBar")
                refresh_action = command_bar.addCommand(
                    "刷新全部",
                    self.refresh_module_status,
                    icon="refresh",
                    tooltip="异步检查全部模块；重复点击会在当前检查完成前被拦截",
                    primary=True,
                )
                self._module_refresh_action = refresh_action
                self._fancy_style_controller.register(command_bar)
                heading_layout.addWidget(command_bar)
                hint = QLabel("超过 300 毫秒的检查保持明确加载态；超时后按钮会自动恢复。")
                hint.setObjectName("moduleCenterHint")
                hint.setWordWrap(True)
                heading_layout.addWidget(hint)
                content_layout.addWidget(heading_card)

                grid_host = QWidget()
                grid_host.setObjectName("moduleCenterGridHost")
                grid = QGridLayout(grid_host)
                grid.setContentsMargins(0, 0, 0, 0)
                grid.setHorizontalSpacing(12)
                grid.setVerticalSpacing(12)
                self._module_grid_layout = grid
                cards: list[QFrame] = []
                for module_id, title, icon_name, _sections, description in _MODULE_CENTER_SPECS:
                    module_card = FancyCard(
                        title,
                        description,
                        icon=icon_name,
                        theme=self._md3_theme,
                    )
                    module_card.setObjectName(f"moduleCard_{module_id}")
                    module_card.setProperty("consoleModuleCard", True)
                    self._fancy_style_controller.register(module_card)
                    card_layout = module_card.content_layout
                    card_layout.setSpacing(8)

                    state_row = QHBoxLayout()
                    state_row.setSpacing(8)
                    status_icon = QLabel()
                    status_icon.setObjectName(f"moduleStatusIcon_{module_id}")
                    status_icon.setFixedSize(20, 20)
                    status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    state_label = QLabel("未检查")
                    state_label.setObjectName(f"moduleState_{module_id}")
                    state_label.setProperty("consoleModuleState", True)
                    state_label.setProperty("state", "idle")
                    state_row.addWidget(status_icon)
                    state_row.addWidget(state_label, 1)
                    card_layout.addLayout(state_row)

                    detail_label = QLabel("等待首次状态检查")
                    detail_label.setObjectName(f"moduleDetail_{module_id}")
                    detail_label.setProperty("consoleModuleDetail", True)
                    detail_label.setWordWrap(True)
                    card_layout.addWidget(detail_label)
                    meta_label = QLabel("最后检查：尚未检查 · IPC：未报告 · 模型：未报告")
                    meta_label.setObjectName(f"moduleMeta_{module_id}")
                    meta_label.setProperty("consoleModuleMeta", True)
                    meta_label.setWordWrap(True)
                    card_layout.addWidget(meta_label)

                    actions = QHBoxLayout()
                    actions.setSpacing(8)
                    test_button = QPushButton("测试")
                    test_button.setObjectName(f"moduleTest_{module_id}")
                    test_button.setIcon(fancy_icon("play", theme=self._md3_theme))
                    test_button.setToolTip("执行模块专用 IPC 测试；未接线时复用全局只读健康检查")
                    test_button.setAccessibleName(f"测试{title}")
                    test_button.clicked.connect(
                        lambda _checked=False, selected=module_id: self._run_module_test(selected)
                    )
                    config_button = QPushButton("打开配置")
                    config_button.setObjectName(f"moduleConfig_{module_id}")
                    config_button.setIcon(fancy_icon("folder", theme=self._md3_theme))
                    config_button.setToolTip(f"打开{title}对应的配置域")
                    config_button.setAccessibleName(f"配置{title}")
                    config_button.clicked.connect(
                        lambda _checked=False, selected=module_id: self._open_module_configuration(
                            selected
                        )
                    )
                    actions.addWidget(test_button)
                    actions.addWidget(config_button)
                    if module_id == "mcp" and callable(self._callbacks.get("mcp_content")):
                        content_button = QPushButton("浏览内容")
                        content_button.setObjectName("moduleContent_mcp")
                        content_button.setIcon(fancy_icon("file", theme=self._md3_theme))
                        content_button.setToolTip("显式浏览 MCP resources 与 prompts")
                        content_button.setAccessibleName("浏览 MCP 内容")
                        content_button.clicked.connect(
                            lambda _checked=False: self._run_control_action("mcp_content")
                        )
                        actions.addWidget(content_button)
                    actions.addStretch(1)
                    card_layout.addLayout(actions)

                    self._module_status_icons[module_id] = status_icon
                    self._module_status_labels[module_id] = state_label
                    self._module_detail_labels[module_id] = detail_label
                    self._module_meta_labels[module_id] = meta_label
                    self._module_test_buttons[module_id] = test_button
                    cards.append(module_card)
                    self._set_module_state(
                        module_id,
                        "idle",
                        detail="等待首次状态检查",
                        checked_at="尚未检查",
                    )

                self._module_cards = tuple(cards)
                content_layout.addWidget(grid_host)
                content_layout.addStretch(1)
                scroll.setWidget(content)
                host_layout.addWidget(scroll)
                self._module_center_loaded = True
                self._set_module_columns(1 if self.width() < 820 else 2)
                pending_snapshot = self._module_pending_snapshot
                self._module_pending_snapshot = None
                if pending_snapshot is not None:
                    self.set_module_status_snapshot(pending_snapshot)
                elif callable(self._callbacks.get("module_status")):
                    QTimer.singleShot(0, self.refresh_module_status)
                return True
            finally:
                self._module_center_building = False

        def _set_module_columns(self, columns: int) -> None:
            """按控制台宽度重排模块卡，不创建新的顶层 QWidget。"""

            grid = self._module_grid_layout
            if grid is None or not self._module_cards:
                return
            safe_columns = 1 if int(columns) <= 1 else 2
            if safe_columns == self._module_columns:
                return
            while grid.count():
                grid.takeAt(0)
            for index, module_card in enumerate(self._module_cards):
                module_card.setMinimumWidth(0)
                module_card.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )
                grid.addWidget(module_card, index // safe_columns, index % safe_columns)
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1 if safe_columns == 2 else 0)
            self._module_columns = safe_columns

        def _set_module_state(
            self,
            module_id: str,
            state: str,
            *,
            detail: str,
            checked_at: str,
            ipc_text: str = "IPC：未报告",
            model_text: str = "模型：未报告",
            latency_ms: object = None,
        ) -> None:
            """更新一个模块卡的图标、文本与脱敏元数据。"""

            normalized = (
                state
                if state
                in {
                    "idle",
                    "checking",
                    "available",
                    "degraded",
                    "unavailable",
                }
                else "degraded"
            )
            labels = {
                "idle": "未检查",
                "checking": "加载中…",
                "available": "正常",
                "degraded": "降级",
                "unavailable": "不可用",
            }
            icons = {
                "idle": "info",
                "checking": "refresh",
                "available": "success",
                "degraded": "warning",
                "unavailable": "error",
            }
            status_icon = self._module_status_icons.get(module_id)
            if status_icon is not None:
                status_icon.setPixmap(
                    fancy_icon(icons[normalized], theme=self._md3_theme).pixmap(18, 18)
                )
                status_icon.setAccessibleName(f"模块状态：{labels[normalized]}")
            status_label = self._module_status_labels.get(module_id)
            if status_label is not None:
                status_label.setText(labels[normalized])
                status_label.setProperty("state", normalized)
                status_label.style().unpolish(status_label)
                status_label.style().polish(status_label)
            detail_label = self._module_detail_labels.get(module_id)
            if detail_label is not None:
                detail_label.setText(str(detail or "状态字段未报告"))
            meta_label = self._module_meta_labels.get(module_id)
            if meta_label is not None:
                latency = (
                    f" · 延迟：{max(0.0, float(latency_ms)):.0f} ms"
                    if isinstance(latency_ms, (int, float)) and not isinstance(latency_ms, bool)
                    else ""
                )
                meta_label.setText(f"最后检查：{checked_at} · {ipc_text} · {model_text}{latency}")

        def _set_module_check_busy(self, busy: bool) -> None:
            """同步全局检查的防重复提交状态。"""

            self._module_check_busy = bool(busy)
            action = self._module_refresh_action
            setter = getattr(action, "setEnabled", None)
            if callable(setter):
                setter(not busy)
            for module_id, test_button in self._module_test_buttons.items():
                if module_id not in self._module_test_generations:
                    test_button.setEnabled(not busy)

        def refresh_module_status(self) -> object | None:
            """异步提交全模块状态检查，主线程只维护加载态和超时。"""

            if self._module_check_busy:
                self.set_status("模块检查仍在进行，请等待当前检查完成")
                return {"status": "requested"}
            if not self._ensure_module_center():
                self.set_status("模块中心尚未就绪")
                return None
            self._module_check_generation += 1
            generation = self._module_check_generation
            self._set_module_check_busy(True)
            for module_id, _title, _icon, _sections, _description in _MODULE_CENTER_SPECS:
                self._set_module_state(
                    module_id,
                    "checking",
                    detail="正在通过运行时线程检查…",
                    checked_at="检查中",
                )
            result = self._invoke("module_status")
            if result is None:
                self._finish_module_check_unavailable("模块状态回调不可用")
                return None
            status = _status_value(result).strip().casefold()
            if status in _MODULE_PENDING_STATES:
                self.set_status("正在检查全部模块…")
                QTimer.singleShot(
                    15000,
                    lambda expected=generation: self._expire_module_check(expected),
                )
                return result
            self.set_module_status_snapshot(result)
            return result

        def _expire_module_check(self, generation: int) -> None:
            """为没有返回终态的模块检查恢复按钮和可访问错误状态。"""

            if generation != self._module_check_generation or not self._module_check_busy:
                return
            self._finish_module_check_unavailable("检查超时，可重新尝试")

        def _finish_module_check_unavailable(self, detail: str) -> None:
            checked_at = QTime.currentTime().toString("HH:mm:ss")
            for module_id, _title, _icon, _sections, _description in _MODULE_CENTER_SPECS:
                self._set_module_state(
                    module_id,
                    "unavailable",
                    detail=detail,
                    checked_at=checked_at,
                )
            self._set_module_check_busy(False)
            self.set_status(detail)

        def set_module_status_snapshot(self, result: object) -> bool:
            """投影模块状态终态；仅消费固定白名单字段。"""

            if not isinstance(result, Mapping):
                self._finish_module_check_unavailable("模块状态返回格式无效")
                return False
            status = _status_value(result).strip().casefold()
            if status in _MODULE_PENDING_STATES:
                self._set_module_check_busy(True)
                return False
            if not self._module_center_loaded:
                self._module_pending_snapshot = dict(result)
                self._set_module_check_busy(False)
                return True
            checked_at = QTime.currentTime().toString("HH:mm:ss")
            for module_id, _title, _icon, _sections, _description in _MODULE_CENTER_SPECS:
                payload = _module_public_payload(result, module_id)
                state = _module_public_state(payload)
                detail = _module_public_detail(module_id, payload)
                if state == "idle":
                    state = "degraded"
                    detail = "运行时未返回该模块的脱敏状态"
                elif state == "degraded" and not payload:
                    detail = "运行时未返回该模块的脱敏状态"
                self._set_module_state(
                    module_id,
                    state,
                    detail=detail,
                    checked_at=checked_at,
                    ipc_text=_module_public_ipc(payload, module_id=module_id),
                    model_text=_module_public_model(payload),
                    latency_ms=payload.get("latency_ms"),
                )
            self._set_module_check_busy(False)
            self.set_status("模块检查已完成")
            return True

        def _run_module_test(self, module_id: str) -> object | None:
            """提交一个模块的 IPC 测试；Future 由 Qt 定时器轮询。"""

            if module_id not in self._module_test_buttons:
                return None
            if module_id in self._module_test_generations or self._module_check_busy:
                self.set_status("该模块仍在测试，请等待当前测试完成")
                return {"status": "requested"}
            callback = self._callbacks.get("module_test")
            if not callable(callback):
                return self.refresh_module_status()
            generation = self._module_test_counters.get(module_id, 0) + 1
            self._module_test_counters[module_id] = generation
            self._module_test_generations[module_id] = generation
            button = self._module_test_buttons[module_id]
            button.setEnabled(False)
            button.setText("测试中…")
            self._set_module_state(
                module_id,
                "checking",
                detail="正在通过 IPC 执行模块测试…",
                checked_at="测试中",
            )
            try:
                result = callback(module_id)
            except Exception as exc:
                self.append_line(f"系统：模块测试失败：{_safe_error_message(exc)}")
                self._finish_module_test(
                    module_id,
                    generation,
                    {"status": "unavailable"},
                )
                return None
            done = getattr(result, "done", None)
            result_reader = getattr(result, "result", None)
            if callable(done) and callable(result_reader):
                QTimer.singleShot(
                    40,
                    lambda selected=module_id, expected=generation, future=result: (
                        self._poll_module_test(selected, expected, future, 375)
                    ),
                )
                return result
            self._finish_module_test(module_id, generation, result)
            return result

        def _poll_module_test(
            self,
            module_id: str,
            generation: int,
            future: object,
            remaining_polls: int,
        ) -> None:
            """非阻塞轮询专用测试 Future，最多等待十五秒。"""

            if self._module_test_generations.get(module_id) != generation:
                return
            done = getattr(future, "done", None)
            result_reader = getattr(future, "result", None)
            if not callable(done) or not callable(result_reader):
                self._finish_module_test(module_id, generation, {"status": "unavailable"})
                return
            if not done():
                if remaining_polls <= 0:
                    cancel = getattr(future, "cancel", None)
                    if callable(cancel):
                        try:
                            cancel()
                        except (RuntimeError, TypeError, ValueError):
                            pass
                    self._finish_module_test(
                        module_id,
                        generation,
                        {"status": "unavailable", "reason_code": "timeout"},
                    )
                    return
                QTimer.singleShot(
                    40,
                    lambda: self._poll_module_test(
                        module_id,
                        generation,
                        future,
                        remaining_polls - 1,
                    ),
                )
                return
            try:
                result = result_reader()
            except Exception as exc:
                self.append_line(f"系统：模块测试失败：{_safe_error_message(exc)}")
                result = {"status": "unavailable"}
            self._finish_module_test(module_id, generation, result)

        def _finish_module_test(
            self,
            module_id: str,
            generation: int,
            result: object,
        ) -> None:
            """提交模块测试终态并恢复按钮；不显示识别文本或原始日志。"""

            if self._module_test_generations.get(module_id) != generation:
                return
            self._module_test_generations.pop(module_id, None)
            button = self._module_test_buttons.get(module_id)
            if button is not None:
                button.setText("测试")
                button.setEnabled(not self._module_check_busy)
            payload = (
                dict(result)
                if isinstance(result, Mapping)
                else {"status": "available" if result is True else "unavailable"}
            )
            if isinstance(result, Mapping) and any(
                key in result for key in ("runtime", "adapters", "tools", "platform")
            ):
                payload = _module_public_payload(result, module_id)
            state = _module_public_state(payload)
            if state == "checking":
                state = "degraded"
            if state == "idle":
                state = "unavailable"
            detail = _module_public_detail(module_id, payload)
            if state == "unavailable":
                detail = "模块测试未通过，可检查配置后重试"
            elif state == "degraded":
                raw_status = str(payload.get("status", payload.get("state", "")) or "").casefold()
                detail = (
                    "模块正在后台加载；测试已结束，可稍后重试"
                    if raw_status in _MODULE_PENDING_STATES
                    else "模块测试完成，但当前处于降级或关闭状态"
                )
            self._set_module_state(
                module_id,
                state,
                detail=detail,
                checked_at=QTime.currentTime().toString("HH:mm:ss"),
                ipc_text=_module_public_ipc(payload, module_id=module_id),
                model_text=_module_public_model(payload),
                latency_ms=payload.get("latency_ms"),
            )
            self.set_status("模块测试已完成" if state == "available" else "模块测试未完全通过")

        def _open_module_configuration(self, module_id: str) -> bool:
            """打开模块对应配置域，复用既有表单、自动保存和热重载。"""

            spec = next((item for item in _MODULE_CENTER_SPECS if item[0] == module_id), None)
            if spec is None or not self.show_settings():
                return False
            panel = self._configuration_panel
            if panel is None:
                return False
            for section in spec[3]:
                opener = getattr(panel, "open_section", None)
                if callable(opener) and opener(section):
                    self.set_status(f"已打开{spec[1]}配置")
                    return True
            self.set_status(f"{spec[1]}配置域尚未加载")
            return False

        @staticmethod
        def _console_page_columns_for_width(width: int) -> int:
            """宽屏使用双栏，窄于 720px 时回落到单栏。"""

            return 1 if int(width) < 720 else 2

        def _set_console_page_columns(self, columns: int) -> None:
            """重排桌宠和工具页面，同时保持控件实例与回调不变。"""

            safe_columns = 1 if int(columns) <= 1 else 2
            if safe_columns == getattr(self, "_console_page_columns", 0):
                return

            pet_layout = getattr(self, "_pet_page_layout", None)
            pet_items = getattr(self, "_pet_page_items", ())
            if pet_layout is not None and len(pet_items) == 4:
                while pet_layout.count() > 1:
                    pet_layout.takeAt(1)
                quick_card, window_card, size_card, position_card = pet_items
                if safe_columns == 2:
                    pet_layout.addWidget(quick_card, 1, 0, 3, 1)
                    pet_layout.addWidget(window_card, 1, 1)
                    pet_layout.addWidget(size_card, 2, 1)
                    pet_layout.addWidget(position_card, 3, 1)
                    pet_layout.setColumnStretch(0, 3)
                    pet_layout.setColumnStretch(1, 2)
                else:
                    for row, item in enumerate(pet_items, start=1):
                        pet_layout.addWidget(item, row, 0, 1, 2)
                    pet_layout.setColumnStretch(0, 1)
                    pet_layout.setColumnStretch(1, 0)

            tools_layout = getattr(self, "_tools_page_layout", None)
            tools_items = getattr(self, "_tools_page_items", ())
            if tools_layout is not None and len(tools_items) == 2:
                while tools_layout.count() > 1:
                    tools_layout.takeAt(1)
                if safe_columns == 2:
                    tools_layout.addWidget(tools_items[0], 1, 0)
                    tools_layout.addWidget(tools_items[1], 1, 1)
                    tools_layout.setColumnStretch(0, 1)
                    tools_layout.setColumnStretch(1, 1)
                else:
                    tools_layout.addWidget(tools_items[0], 1, 0, 1, 2)
                    tools_layout.addWidget(tools_items[1], 2, 0, 1, 2)
                    tools_layout.setColumnStretch(0, 1)
                    tools_layout.setColumnStretch(1, 0)
            self._console_page_columns = safe_columns

        def _set_overview_action_columns(self, columns: int) -> None:
            """窄屏下把概览入口改为多行，保留完整按钮文案。"""

            layout = getattr(self, "_overview_action_layout", None)
            items = getattr(self, "_overview_action_items", ())
            if layout is None or len(items) != 3:
                return
            safe_columns = max(1, min(3, int(columns)))
            if safe_columns == getattr(self, "_overview_action_columns", 0):
                return
            while layout.count():
                layout.takeAt(0)
            for index, item in enumerate(items):
                item.setMinimumWidth(0)
                item.setSizePolicy(
                    QSizePolicy.Policy.Expanding,
                    QSizePolicy.Policy.Preferred,
                )
                layout.addWidget(item, index // safe_columns, index % safe_columns)
            for column in range(3):
                layout.setColumnStretch(column, 1 if column < safe_columns else 0)
            self._overview_action_columns = safe_columns

        @staticmethod
        def _context_columns_for_width(width: int) -> int:
            """根据控制台宽度选择桌面观察按钮列数。"""

            value = max(1, int(width))
            if value < 280:
                return 1
            if value < 380:
                return 2
            return 3

        @staticmethod
        def _conversation_columns_for_width(width: int) -> int:
            """在极窄控制台中将输入框和三个操作按钮改为多行布局。"""

            value = max(1, int(width))
            if value < 220:
                return 1
            # 可用内容区已经在 resizeEvent 中扣除了导航栏和边距。
            # 492px（680px 外框）足以容纳输入框与三个操作按钮的单行布局，
            # 因而阈值应与实际控件最小宽度对齐，避免被过早压成三列。
            if value < 480:
                return 3
            return 4

        def _set_conversation_input_layout(self, columns: int) -> None:
            """按可用宽度重排输入框，确保小窗口仍可发送/停止。"""

            layout = getattr(self, "_conversation_input_layout", None)
            items = getattr(self, "_conversation_input_items", ())
            if layout is None or len(items) != 4:
                return
            safe_columns = max(1, min(4, int(columns)))
            if safe_columns == getattr(self, "_conversation_input_columns", 0):
                return
            # 控件继续归原宿主管理，随后同步加入新的网格位置。
            while layout.count():
                layout.takeAt(0)
            input_widget, microphone_button, send_button, stop_button = items
            if safe_columns >= 4:
                layout.addWidget(input_widget, 0, 0)
                layout.addWidget(microphone_button, 0, 1)
                layout.addWidget(send_button, 0, 2)
                layout.addWidget(stop_button, 0, 3)
                layout.setColumnStretch(0, 1)
                layout.setColumnStretch(1, 0)
                layout.setColumnStretch(2, 0)
                layout.setColumnStretch(3, 0)
            elif safe_columns >= 2:
                layout.addWidget(input_widget, 0, 0, 1, 3)
                layout.addWidget(microphone_button, 1, 0)
                layout.addWidget(send_button, 1, 1)
                layout.addWidget(stop_button, 1, 2)
                layout.setColumnStretch(0, 1)
                layout.setColumnStretch(1, 1)
                layout.setColumnStretch(2, 1)
            else:
                layout.addWidget(input_widget, 0, 0)
                layout.addWidget(microphone_button, 1, 0)
                layout.addWidget(send_button, 2, 0)
                layout.addWidget(stop_button, 3, 0)
                layout.setColumnStretch(0, 1)
            self._conversation_input_columns = safe_columns

        def _set_microphone_device_layout(self, columns: int) -> None:
            """窄屏下纵向排列设备标签、选择器和刷新按钮。"""

            layout = getattr(self, "_microphone_device_layout", None)
            widgets = (
                getattr(self, "_microphone_device_label", None),
                getattr(self, "_microphone_device_selector", None),
                getattr(self, "_microphone_device_refresh_button", None),
            )
            if layout is None or any(widget is None for widget in widgets):
                return
            safe_columns = 1 if int(columns) <= 1 else 3
            if safe_columns == getattr(self, "_microphone_device_layout_columns", 0):
                return
            while layout.count():
                layout.takeAt(0)
            label, selector, refresh = widgets
            if safe_columns == 1:
                layout.addWidget(label, 0, 0)
                layout.addWidget(selector, 1, 0)
                layout.addWidget(refresh, 2, 0)
                layout.setColumnStretch(0, 1)
            else:
                layout.addWidget(label, 0, 0)
                layout.addWidget(selector, 0, 1)
                layout.addWidget(refresh, 0, 2)
                layout.setColumnStretch(0, 0)
                layout.setColumnStretch(1, 1)
                layout.setColumnStretch(2, 0)
            self._microphone_device_layout_columns = safe_columns

        def _set_context_action_columns(self, columns: int) -> None:
            """重排桌面观察按钮，避免极窄窗口产生横向溢出。"""

            layout = getattr(self, "_context_actions_layout", None)
            items = getattr(self, "_context_action_items", ())
            if layout is None or not items:
                return
            safe_columns = max(1, min(len(items), int(columns)))
            if safe_columns == getattr(self, "_context_action_columns", 0):
                return
            # 控件继续归原宿主管理，随后同步加入新的网格位置。
            while layout.count():
                layout.takeAt(0)
            for index, item in enumerate(items):
                item.setMinimumWidth(0)
                item.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
                layout.addWidget(item, index // safe_columns, index % safe_columns)
            for column in range(len(items)):
                layout.setColumnStretch(column, 1 if column < safe_columns else 0)
            self._context_action_columns = safe_columns

        @staticmethod
        def _renderer_model_columns_for_width(width: int) -> int:
            """在窄屏中将模型标签、选择器和按钮改为纵向布局。"""

            return 1 if int(width) < 420 else 3

        def _set_renderer_model_layout(self, columns: int) -> None:
            """按控制台宽度重排模型选择控件，避免横向最小宽度溢出。"""

            layout = getattr(self, "_renderer_model_row", None)
            widgets = getattr(self, "_renderer_model_widgets", ())
            if layout is None or len(widgets) != 3:
                return
            safe_columns = 1 if int(columns) <= 1 else 3
            if safe_columns == getattr(self, "_renderer_model_columns", 0):
                return
            # 控件继续归原宿主管理，随后同步加入新的网格位置。
            while layout.count():
                layout.takeAt(0)
            label, selector, button = widgets
            if safe_columns == 1:
                layout.addWidget(label, 0, 0)
                layout.addWidget(selector, 1, 0)
                layout.addWidget(button, 2, 0)
                layout.setColumnStretch(0, 1)
            else:
                layout.addWidget(label, 0, 0)
                layout.addWidget(selector, 0, 1)
                layout.addWidget(button, 0, 2)
                layout.setColumnStretch(0, 0)
                layout.setColumnStretch(1, 1)
                layout.setColumnStretch(2, 0)
            self._renderer_model_columns = safe_columns

        def resizeEvent(self, event) -> None:  # noqa: N802
            """小窗口自动进入紧凑模式，并让恢复按钮按宽度换行。"""

            super().resizeEvent(event)
            width = max(1, int(self.width()))
            height = max(1, int(self.height()))
            navigation_width = 132 if width < 720 else 188
            content_width = max(1, width - navigation_width - 56)
            self._set_rescue_columns(2 if width < 520 else 5)
            overview_columns = 1 if content_width < 280 else 2 if content_width < 560 else 3
            self._set_overview_action_columns(overview_columns)
            self._set_context_action_columns(self._context_columns_for_width(content_width))
            self._set_conversation_input_layout(self._conversation_columns_for_width(content_width))
            self._set_microphone_device_layout(1 if content_width < 380 else 3)
            self._set_quick_action_columns(self._quick_columns_for_width(content_width))
            self._set_renderer_model_layout(self._renderer_model_columns_for_width(content_width))
            self._set_console_page_columns(self._console_page_columns_for_width(width))
            self._set_module_columns(1 if width < 820 else 2)
            tabs = getattr(self, "_console_tabs", None)
            if isinstance(tabs, FancyNavigationView):
                tabs.setCompact(width < 720)
            command_bar = getattr(self, "_command_bar", None)
            if isinstance(command_bar, FancyCommandBar):
                command_bar.setCompact(width < 720)
            compact = width < 560 or height < 620
            if compact != bool(self.property("compactLayout")):
                self.set_compact_layout(compact)
            if (
                compact
                and tabs is not None
                and tabs.currentIndex() == getattr(self, "_conversation_page_index", -1)
            ):
                self._ensure_conversation_input_visible()
                QTimer.singleShot(0, self._ensure_conversation_input_visible)

        def _set_developer_actions_visible(self, visible: bool) -> None:
            """保留动作编排入口；开发者模式只影响其它原始配置。"""

            del visible
            self._advanced_action_toggle.setVisible(True)

        def _toggle_advanced_actions(self) -> None:
            """展开或收起需要输入名称的高级动作区。"""

            self._advanced_actions_visible = not self._advanced_actions_visible
            self._advanced_action_panel.setVisible(self._advanced_actions_visible)
            self._advanced_action_toggle.setText(
                "收起动作编排" if self._advanced_actions_visible else "动作编排…"
            )
            self.set_status(
                "已展开自定义动作入口" if self._advanced_actions_visible else "已收起自定义动作入口"
            )

        @staticmethod
        def _action_field(action: object, name: str, default: object = "") -> object:
            if isinstance(action, Mapping):
                return action.get(name, default)
            return getattr(action, name, default)

        def _clear_state_actions(self) -> None:
            while self._state_actions_layout.count():
                item = self._state_actions_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        def _run_state_action(self, kind: str, payload: Mapping[str, object]) -> None:
            """执行交互快照提供的下一步动作。"""

            normalized = str(kind or "").strip().lower()
            if normalized == "configure_model":
                self.show_model_settings()
                return
            if normalized == "stop":
                self._stop()
                return
            if normalized == "approve":
                approval_id = str(payload.get("approval_id", "") or "").strip()
                if not approval_id:
                    self.set_status("批准不可用：请求已过期")
                    return
                result = self._invoke(
                    "approve_approval", approval_id, bool(payload.get("grant_session", False))
                )
                self.set_status("已提交批准" if _operation_succeeded(result) else "批准未提交")
                return
            if normalized == "grant_session":
                approval_id = str(payload.get("approval_id", "") or "").strip()
                if not approval_id:
                    self.set_status("批准不可用：请求已过期")
                    return
                result = self._invoke("approve_approval", approval_id, True)
                self.set_status(
                    "已批准并允许本会话" if _operation_succeeded(result) else "批准未提交"
                )
                return
            if normalized == "deny":
                approval_id = str(payload.get("approval_id", "") or "").strip()
                if not approval_id:
                    self.set_status("拒绝不可用：请求已过期")
                    return
                result = self._invoke("deny_approval", approval_id)
                self.set_status("已提交拒绝" if _operation_succeeded(result) else "拒绝未提交")
                return
            if normalized == "retry":
                result = self._invoke("retry")
                self.set_status("已提交重试" if _operation_succeeded(result) else "重试不可用")
                return
            self.set_status("当前动作不可用")

        def set_interaction_actions(self, actions: Sequence[object] | object) -> None:
            """根据脱敏交互快照刷新首屏下一步按钮。"""

            if actions is None or isinstance(actions, (str, bytes, bytearray, Mapping)):
                values: tuple[object, ...] = ()
            else:
                try:
                    values = tuple(actions)  # type: ignore[arg-type]
                except TypeError:
                    values = ()
            visible = False
            labels = {
                "configure_model": "配置模型",
                "stop": "停止",
                "approve": "批准",
                "grant_session": "批准并允许本会话",
                "deny": "拒绝",
                "retry": "重试",
            }
            prepared_actions: list[tuple[str, str, Mapping[str, object], bool]] = []
            signatures: list[tuple[str, str, tuple[tuple[str, str], ...], bool]] = []
            for action in values:
                kind_value = self._action_field(action, "kind", "")
                kind = str(getattr(kind_value, "value", kind_value) or "").strip().lower()
                if not kind or kind not in labels:
                    continue
                if kind == "configure_model":
                    static_button = self.findChild(QPushButton, "configureModelButton")
                    if static_button is not None and not static_button.isHidden():
                        continue
                enabled = bool(self._action_field(action, "enabled", True))
                payload = self._action_field(action, "payload", {})
                raw_payload = dict(payload) if isinstance(payload, Mapping) else {}
                # 动作快照可能跨线程/进程传递，Qt 只保留执行所需的最小字段。
                # 这样即使宿主错误地附带 arguments、token 等字段，也不会进入
                # 按钮属性、签名或闭包，普通用户界面始终只有点击动作。
                payload_mapping: dict[str, object] = {}
                if kind in {"approve", "grant_session", "deny"}:
                    approval_id = str(raw_payload.get("approval_id", "") or "").strip()
                    if approval_id:
                        payload_mapping["approval_id"] = approval_id[:256]
                    if kind == "approve" and isinstance(raw_payload.get("grant_session"), bool):
                        payload_mapping["grant_session"] = raw_payload["grant_session"]
                payload_signature = tuple(
                    sorted(
                        (str(key), _safe_ui_text(value, limit=120))
                        for key, value in payload_mapping.items()
                    )
                )
                label = labels[kind]
                prepared_actions.append((kind, label, payload_mapping, enabled))
                signatures.append((kind, label, payload_signature, enabled))
            signature = tuple(signatures)
            if signature == self._state_action_signature:
                return
            self._state_action_signature = signature
            self._clear_state_actions()
            for kind, label, payload_mapping, enabled in prepared_actions:
                button = QPushButton(label or labels[kind])
                button.setObjectName(f"stateAction_{kind}")
                button.setAccessibleName(label or labels[kind])
                button.setProperty("role", "danger" if kind in {"deny", "stop"} else "primary")
                button.setEnabled(enabled)
                button.setMinimumHeight(30)
                button.setMaximumHeight(34)
                button.setMinimumWidth(76)
                button.clicked.connect(
                    lambda _checked=False, selected=kind, selected_payload=payload_mapping: (
                        self._run_state_action(selected, selected_payload)
                    )
                )
                self._state_actions_layout.addWidget(button, 1)
                visible = True
            self._state_actions_host.setVisible(visible)

        def _toggle_secondary_actions(self) -> None:
            """在工具与安全页和对话页之间切换。"""

            tabs = getattr(self, "_console_tabs", None)
            panel = getattr(self, "_secondary_actions_panel", None)
            if tabs is None or panel is None:
                self.set_status("工具与安全页面不可用")
                return
            visible = tabs.currentWidget() is panel
            target = self._conversation_page_index if visible else self._tools_page_index
            tabs.setCurrentIndex(target)
            self._secondary_actions_toggle.setText("更多操作" if visible else "收起更多操作")
            self.set_status("已返回对话" if visible else "已打开工具与安全")

        def _toggle_position_detail(self) -> None:
            """展开或收起精确坐标输入。"""

            self._position_detail_visible = not self._position_detail_visible
            self._position_detail.setVisible(self._position_detail_visible)
            self.set_status("已展开精确坐标" if self._position_detail_visible else "已收起精确坐标")

        @staticmethod
        def _clear_grid(layout: QGridLayout) -> None:
            """清理快捷按钮网格而不影响宿主回调。"""

            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()

        def _rebuild_quick_actions(self) -> None:
            """按当前渲染器能力重建表情/动作快捷按钮。"""

            self._clear_grid(self._expression_quick_layout)
            self._clear_grid(self._motion_quick_layout)
            self._quick_expression_buttons = []
            self._quick_motion_buttons = []
            self._quick_action_layouts = {
                "expression": self._expression_quick_layout,
                "motion": self._motion_quick_layout,
            }

            def add_buttons(
                values: Sequence[str],
                kind: str,
                layout: QGridLayout,
                target: list[QPushButton],
            ) -> None:
                fallback = ("neutral",) if kind == "expression" else ("idle",)
                normalized = _action_names(values, fallback)
                for index, value in enumerate(normalized):
                    action_value = str(value).strip()
                    display_value = _public_action_label(kind, action_value)
                    quick = QPushButton(display_value)
                    quick.setObjectName(f"{kind}QuickButton{index}")
                    quick.setProperty("role", "quick")
                    quick.setProperty("actionKind", kind)
                    quick.setProperty("actionValue", action_value)
                    quick.setCheckable(True)
                    quick.setMinimumHeight(38)
                    quick.setToolTip(
                        f"点按执行表情：{display_value}"
                        if kind == "expression"
                        else f"点按执行动作：{display_value}"
                    )
                    quick.setAccessibleName(
                        f"快捷表情 {display_value}"
                        if kind == "expression"
                        else f"快捷动作 {display_value}"
                    )
                    quick.clicked.connect(
                        lambda _checked=False, selected=action_value, selected_kind=kind: (
                            self._quick_action(selected_kind, selected)
                        )
                    )
                    layout.addWidget(quick, index // 3, index % 3)
                    target.append(quick)
                for column in range(3):
                    layout.setColumnStretch(column, 1)

            expression_values = tuple(
                self._expression.itemText(index) for index in range(self._expression.count())
            )
            motion_values = tuple(
                self._motion.itemText(index) for index in range(self._motion.count())
            )
            add_buttons(
                expression_values,
                "expression",
                self._expression_quick_layout,
                self._quick_expression_buttons,
            )
            add_buttons(
                motion_values,
                "motion",
                self._motion_quick_layout,
                self._quick_motion_buttons,
            )
            self._sync_quick_selection("expression", self._expression.currentText())
            self._sync_quick_selection("motion", self._motion.currentText())
            self._quick_action_columns = 3
            self._set_quick_action_columns(self._quick_columns_for_width(self.width()))

        @staticmethod
        def _quick_columns_for_width(width: int) -> int:
            """按窗口宽度重排表情/动作按钮，避免极窄页被最小宽度撑开。"""

            value = max(1, int(width))
            if value < 220:
                return 1
            if value < 380:
                return 2
            return 3

        def _set_quick_action_columns(self, columns: int) -> None:
            """在窄屏中将动作按钮改为单列/双列。"""

            layouts = getattr(self, "_quick_action_layouts", {})
            if not layouts:
                return
            safe_columns = max(1, min(3, int(columns)))
            if safe_columns == getattr(self, "_quick_action_columns", 0):
                return
            values = {
                "expression": self._quick_expression_buttons,
                "motion": self._quick_motion_buttons,
            }
            for kind, layout in layouts.items():
                # 控件继续归原宿主管理，随后同步加入新的网格位置。
                while layout.count():
                    layout.takeAt(0)
                for index, button in enumerate(values[kind]):
                    button.setMinimumWidth(0)
                    button.setSizePolicy(
                        QSizePolicy.Policy.Expanding,
                        QSizePolicy.Policy.Preferred,
                    )
                    layout.addWidget(button, index // safe_columns, index % safe_columns)
                for column in range(3):
                    layout.setColumnStretch(column, 1 if column < safe_columns else 0)
            self._quick_action_columns = safe_columns

        def _quick_action(self, kind: str, value: str) -> None:
            """执行快捷表情或动作，并同步高级编辑区。"""

            if kind == "expression":
                self._expression.setCurrentText(value)
                self._apply_expression_value(value)
            elif kind == "motion":
                self._motion.setCurrentText(value)
                self._apply_motion_value(value)

        def _sync_quick_selection(self, kind: str, value: str) -> None:
            selected = str(value or "").strip()
            buttons = (
                self._quick_expression_buttons
                if kind == "expression"
                else self._quick_motion_buttons
            )
            for quick in buttons:
                quick.blockSignals(True)
                quick.setChecked(quick.property("actionValue") == selected)
                quick.blockSignals(False)

        def _action_value_is_allowed(self, kind: str, value: str) -> bool:
            """确认动作值来自当前能力列表或已显式打开的开发者入口。

            下拉框在开发者面板中保留可编辑能力，方便排查模型自定义的
            ``exp3``/``motion3`` 名称；普通快捷路径不能因为控件暂时保留
            旧文本，就把未声明的名称提交给渲染器并显示为已执行。
            """

            selected = str(value or "").strip()
            if not selected:
                return False
            combo = self._expression if kind == "expression" else self._motion
            declared = {
                combo.itemText(index).strip()
                for index in range(combo.count())
                if combo.itemText(index).strip()
            }
            return selected in declared or bool(self._advanced_actions_visible)

        def _apply_expression_value(self, value: str) -> None:
            selected = str(value or "").strip()
            if not selected:
                self.set_status("表情不可用：没有选择表情")
                return
            safe_selected = _public_action_label("expression", selected)
            if not self._action_value_is_allowed("expression", selected):
                self.set_status(f"表情不可用：{safe_selected}（请从能力按钮中选择）")
                self.append_line(f"系统：表情 {safe_selected} 未执行：当前模型未声明此表情")
                return
            self.expressionRequested.emit(selected)
            self.operationTriggered.emit("expression")
            result = self._invoke("expression", selected)
            if _operation_pending(result):
                self._sync_quick_selection("expression", selected)
                # 视觉动作只等待渲染器页面/模型首帧，不依赖对话模型渠道。
                # 旧文案把两者都称作“模型”，会让未配置 LLM 的用户误以为
                # 必须先填写 API 才能手动测试表情。
                self.set_status(f"表情已排队：{safe_selected}（渲染器加载后执行）")
                return
            if _operation_succeeded(result):
                self._sync_quick_selection("expression", selected)
                self.set_status(f"表情：{safe_selected}")
                return
            self.set_status(f"表情不可用：{safe_selected}")
            self.append_line(f"系统：表情 {safe_selected} 未执行：{_safe_result_summary(result)}")

        def _apply_motion_value(self, value: str) -> None:
            selected = str(value or "").strip()
            if not selected:
                self.set_status("动作不可用：没有选择动作")
                return
            safe_selected = _public_action_label("motion", selected)
            if not self._action_value_is_allowed("motion", selected):
                self.set_status(f"动作不可用：{safe_selected}（请从能力按钮中选择）")
                self.append_line(f"系统：动作 {safe_selected} 未执行：当前模型未声明此动作")
                return
            self.motionRequested.emit(selected)
            self.operationTriggered.emit("motion")
            result = self._invoke("motion", selected)
            if _operation_pending(result):
                self._sync_quick_selection("motion", selected)
                self.set_status(f"动作已排队：{safe_selected}（渲染器加载后执行）")
                return
            if _operation_succeeded(result):
                self._sync_quick_selection("motion", selected)
                self.set_status(f"动作：{safe_selected}")
                return
            self.set_status(f"动作不可用：{safe_selected}")
            self.append_line(f"系统：动作 {safe_selected} 未执行：{_safe_result_summary(result)}")

        def _sync_empty_state(self) -> None:
            """根据活动记录内容显示或隐藏清晰空状态。"""

            if not hasattr(self, "_empty_state"):
                return
            has_content = bool(self._output.toPlainText().strip())
            compact = bool(self.property("compactLayout"))
            output_policy = self._output.sizePolicy()
            output_policy.setRetainSizeWhenHidden(compact)
            self._output.setSizePolicy(output_policy)
            self._empty_state.setVisible(not has_content and not compact)
            self._output.setVisible(has_content)

        @property
        def expression_quick_buttons(self) -> tuple[QPushButton, ...]:
            """返回当前表情快捷按钮。"""

            return tuple(self._quick_expression_buttons)

        @property
        def motion_quick_buttons(self) -> tuple[QPushButton, ...]:
            """返回当前动作快捷按钮。"""

            return tuple(self._quick_motion_buttons)

        @property
        def status_label(self) -> QLabel:
            """返回当前状态标签。"""

            return self._status

        @property
        def send_button(self) -> QPushButton:
            """返回发送按钮。"""

            return self._send_button

        @property
        def stop_button(self) -> QPushButton:
            """返回停止按钮。"""

            return self._stop_button

        @property
        def microphone_button(self) -> QPushButton:
            """返回用户显式启动/停止录音的按钮。"""

            return self._microphone_button

        @property
        def microphone_status_label(self) -> QLabel:
            """返回不包含设备 ID 或转写正文的麦克风状态标签。"""

            return self._microphone_status

        @property
        def click_through_checkbox(self) -> QCheckBox:
            """返回点击穿透开关。"""

            return self._click_through

        @property
        def window_lock_checkbox(self) -> QCheckBox:
            """返回锁定窗口开关。"""

            return self._window_lock

        @property
        def input_line(self) -> QLineEdit:
            """返回消息输入框，供宿主设置焦点或测试读取。"""

            return self._input

        @property
        def output_view(self) -> QPlainTextEdit:
            """返回只读输出区。"""

            return self._output

        @property
        def position_x(self) -> QSpinBox:
            """返回桌宠 X 坐标输入框。"""

            return self._position_x

        @property
        def position_y(self) -> QSpinBox:
            """返回桌宠 Y 坐标输入框。"""

            return self._position_y

        @property
        def position_bounds(self) -> tuple[int, int, int, int] | None:
            """返回宿主声明的桌宠左上角可移动边界。"""

            return self._position_bounds

        @property
        def approval_list(self) -> QListWidget:
            """返回待审批工具列表，供宿主和测试读取。"""

            return self._approval_list

        def _ensure_configuration_panel(self) -> ConfigurationPanel | None:
            """首次访问时创建配置中心，避免阻塞控制台首屏。"""

            if self._configuration_panel is not None:
                return self._configuration_panel
            if not self._configuration_enabled or self._configuration_panel_building:
                return None
            placeholder = self._configuration_placeholder
            if placeholder is None:
                return None
            self._configuration_panel_building = True
            try:
                panel = ConfigurationPanel(
                    self._configuration,
                    path=self._configuration_path,
                    developer_mode=self._configuration_developer_mode,
                    audio_input_device_loader=self._load_microphone_input_devices,
                    auto_save=bool(
                        self._configuration_path is not None
                        or self._config_callbacks.get("save")
                        or self._callbacks.get("config_save")
                    ),
                )
                panel.setObjectName("configurationCenter")
                panel.statusChanged.connect(self.set_status)
                panel.advancedEditingChanged.connect(self._set_developer_actions_visible)
                panel.saveRequested.connect(self._save_configuration)
                panel.validateRequested.connect(self._validate_configuration)
                panel.modelSetupRequested.connect(self.show_model_settings)
                panel.set_audio_input_devices(
                    self._microphone_input_devices,
                    loaded=self._microphone_devices_loaded,
                )
                layout = placeholder.layout()
                if layout is None:
                    raise RuntimeError("configuration center host layout is unavailable")
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
                layout.addWidget(panel)
                self._configuration_panel = panel
                panel.set_compact_layout(bool(self.property("compactLayout")))
                return panel
            finally:
                self._configuration_panel_building = False

        @property
        def configuration_panel(self) -> ConfigurationPanel | None:
            """返回配置中心，宿主可用它切换到配置页或刷新草稿。"""

            return self._ensure_configuration_panel()

        @property
        def model_ready(self) -> bool:
            """返回当前配置中是否至少有一个可用模型渠道。"""

            return self._model_ready

        def _refresh_model_status(self, *, announce: bool = True) -> tuple[bool, str]:
            """刷新脱敏模型状态和可执行的下一步提示。"""

            ready, message = _configuration_model_status(self._configuration)
            self._model_ready = ready
            self._hint.setText(
                "提示：模型已配置；修改渠道后需要重新建立运行服务。"
                if ready
                else "下一步：点击“配置模型”，选择预设并补齐地址、模型与环境变量引用。"
            )
            if announce:
                self.set_status(message)
            return ready, message

        @staticmethod
        def _replace_action_combo(combo: QComboBox, values: Sequence[str], preferred: str) -> None:
            """刷新动作/表情选项并尽量保留用户当前选择。"""

            normalized = _action_names(values, (preferred,))
            current = combo.currentText().strip()
            combo.blockSignals(True)
            try:
                combo.clear()
                combo.addItems(normalized)
                if current:
                    selected = current
                elif preferred in normalized:
                    selected = preferred
                else:
                    selected = normalized[0]
                if selected not in normalized and not combo.isEditable():
                    selected = normalized[0]
                combo.setCurrentText(selected)
            finally:
                combo.blockSignals(False)

        def _update_action_hints(self) -> None:
            """在控件旁显示当前渲染器声明的能力数量和当前值。"""

            expression_count = self._expression.count()
            motion_count = self._motion.count()
            expression_label = _public_action_label("expression", self._expression.currentText())
            motion_label = _public_action_label("motion", self._motion.currentText())
            self._expression_hint.setText(f"{expression_count} 项 · 当前 {expression_label}")
            self._motion_hint.setText(f"{motion_count} 项 · 当前 {motion_label}")

        @property
        def expression_combo(self) -> QComboBox:
            """返回表情能力下拉框。"""

            return self._expression

        @property
        def motion_combo(self) -> QComboBox:
            """返回动作能力下拉框。"""

            return self._motion

        @property
        def expression_button(self) -> QPushButton:
            """返回表情测试按钮。"""

            return self._expression_button

        @property
        def motion_button(self) -> QPushButton:
            """返回动作测试按钮。"""

            return self._motion_button

        def set_action_capabilities(
            self,
            expressions: Sequence[str] | object = (),
            motions: Sequence[str] | object = (),
        ) -> None:
            """在渲染器切换或 Web 模型就绪后刷新动作/表情列表。

            控制台是懒加载窗口，Web Live2D 可能在窗口创建后才完成模型探测；
            宿主可以重复调用此方法而不会清空用户当前输入。
            """

            self._replace_action_combo(
                self._expression,
                _action_names(expressions, DEFAULT_EXPRESSION_NAMES),
                "neutral",
            )
            self._replace_action_combo(
                self._motion,
                _action_names(motions, DEFAULT_MOTION_NAMES),
                "idle",
            )
            self._update_action_hints()
            self._rebuild_quick_actions()

        def set_renderer_status(
            self,
            backend: object,
            *,
            available: bool = True,
            reason: object = "",
        ) -> None:
            """显示当前实际渲染后端，不暴露内部探测参数。"""

            labels = {
                "opengl": "OpenGL Live2D",
                "web_live2d": "Web Live2D",
                "sprite": "精灵回退",
                "vllank": "3D 渲染预留",
                "unavailable": "不可用",
            }
            normalized = str(backend or "").strip().lower()
            label = labels.get(normalized, "自动后端")
            suffix = "已启用" if available else "不可用"
            text = f"渲染：{label} · {suffix}"
            if not available:
                # 普通控制台只给下一步，不回显探测异常、路径或内部参数；
                # 完整原因仍写入开发者日志和高级诊断。
                text += "（请改用自动或精灵）"
            status = getattr(self, "_renderer_status", None)
            if status is not None:
                status.setText(text)

        def set_renderer_models(
            self,
            selected: object = None,
            choices: Sequence[object] = (),
        ) -> None:
            """显示当前模型和可选的资源相对路径，并刷新点击选择器。"""

            status = getattr(self, "_renderer_model_status", None)
            if status is None:
                return
            values_list: list[str] = []
            for item in choices:
                value = str(item or "").strip()
                if value and value not in values_list:
                    values_list.append(value)
            values = tuple(values_list)
            selected_value = str(selected or "").strip()
            self._renderer_model_options = values
            if selected_value not in values:
                selected_value = values[0] if values else ""
            self._renderer_model_committed = selected_value
            self._renderer_model_pending = ""
            selector = getattr(self, "_renderer_model_selector", None)
            if selector is not None:
                selector.blockSignals(True)
                try:
                    selector.clear()
                    for value in values:
                        selector.addItem(value, value)
                    if selected_value:
                        index = selector.findData(selected_value)
                        if index < 0:
                            index = selector.findText(selected_value)
                        if index >= 0:
                            selector.setCurrentIndex(index)
                finally:
                    selector.blockSignals(False)
            self._update_renderer_model_button()
            if not values:
                status.setText("Live2D 模型：未发现可用 model3")
                return
            current = selected_value or values[0]
            suffix = "、".join(values)
            status.setText(f"Live2D 模型：{current} · 可选 {len(values)} 个（{suffix}）")

        @property
        def renderer_model_selector(self) -> QComboBox:
            """返回 Live2D 模型选择下拉框。"""

            return self._renderer_model_selector

        @property
        def renderer_model_options(self) -> tuple[str, ...]:
            """返回当前模型选择器中的精确资源键。"""

            return self._renderer_model_options

        @property
        def renderer_model_apply_button(self) -> QPushButton:
            """返回模型应用按钮。"""

            return self._renderer_model_apply_button

        def _update_renderer_model_button(self) -> None:
            """按模型列表和待处理状态更新应用按钮。"""

            button = getattr(self, "_renderer_model_apply_button", None)
            selector = getattr(self, "_renderer_model_selector", None)
            if button is None or selector is None:
                return
            value = str(selector.currentData() or selector.currentText() or "").strip()
            button.setEnabled(bool(value and value in self._renderer_model_options))
            if self._renderer_model_pending:
                button.setEnabled(False)

        def _selected_renderer_model(self) -> str:
            """读取选择器返回的精确资源键，不对路径做隐式转换。"""

            selector = self._renderer_model_selector
            value = selector.currentData()
            if not isinstance(value, str) or not value:
                value = selector.currentText()
            value = str(value or "").strip()
            return value if value in self._renderer_model_options else ""

        def _select_renderer_model(self) -> object | None:
            """提交模型切换请求并显示同步/异步回执。"""

            value = self._selected_renderer_model()
            if not value:
                self.set_status("没有可用的 Live2D 模型")
                return None
            if value == self._renderer_model_committed and not self._renderer_model_pending:
                self.set_status("已是当前 Live2D 模型")
                return {"status": "unchanged", "model_key": value}
            self._renderer_model_pending = value
            self._update_renderer_model_button()
            self.rendererModelRequested.emit(value)
            self.operationTriggered.emit("select_renderer_model")
            result = self._invoke("select_renderer_model", value)
            if result is None:
                self._renderer_model_pending = ""
                self._restore_renderer_model_selection()
                return None
            self.set_renderer_model_result(result)
            return result

        def _restore_renderer_model_selection(self) -> None:
            """切换失败时恢复最近一次已确认的模型。"""

            selector = self._renderer_model_selector
            if self._renderer_model_committed not in self._renderer_model_options:
                self._update_renderer_model_button()
                return
            selector.blockSignals(True)
            try:
                index = selector.findData(self._renderer_model_committed)
                if index >= 0:
                    selector.setCurrentIndex(index)
            finally:
                selector.blockSignals(False)
            self._update_renderer_model_button()

        def set_renderer_model_result(self, result: object) -> None:
            """应用宿主模型切换回执，且不向界面暴露绝对路径。"""

            status = _status_value(result).strip().lower()
            if isinstance(result, Mapping):
                raw_key = result.get("model_key", result.get("model", ""))
            else:
                raw_key = ""
            model_key = str(raw_key or "").strip()
            if model_key not in self._renderer_model_options:
                model_key = self._renderer_model_pending or self._renderer_model_committed
            if status in {"pending", "requested"}:
                self._renderer_model_pending = model_key
                self._update_renderer_model_button()
                self.set_status("Live2D 模型切换中…")
                return
            if status in {"available", "reloaded", "completed", "unchanged"}:
                if model_key:
                    self._renderer_model_committed = model_key
                    selector = self._renderer_model_selector
                    selector.blockSignals(True)
                    try:
                        index = selector.findData(model_key)
                        if index >= 0:
                            selector.setCurrentIndex(index)
                    finally:
                        selector.blockSignals(False)
                self._renderer_model_pending = ""
                self._update_renderer_model_button()
                if status == "unchanged":
                    self.set_status("已是当前 Live2D 模型")
                else:
                    self.set_status(f"Live2D 模型已切换：{model_key or '当前模型'}")
                return
            if status == "restart_required":
                if model_key:
                    self._renderer_model_committed = model_key
                self._renderer_model_pending = ""
                self._update_renderer_model_button()
                self.set_status("模型选择已保存；重启桌宠后生效")
                return
            self._renderer_model_pending = ""
            self._restore_renderer_model_selection()
            detail = _safe_ui_text(_detail_value(result))
            suffix = f"：{detail}" if detail else ""
            self.set_status(f"Live2D 模型未切换{suffix}")

        def _rebuild_tts_options(self) -> None:
            """按当前 profile/语言状态生成可直接点选的语音按钮。"""

            language_layout = getattr(self, "_tts_language_layout", None)
            profile_layout = getattr(self, "_tts_profile_layout", None)
            if language_layout is None or profile_layout is None:
                return
            for layout in (language_layout, profile_layout):
                while layout.count():
                    item = layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
            language_labels = {"zh": "中文", "jp": "日文", "en": "英文"}
            for index, language in enumerate(("zh", "jp", "en")):
                selected = self._tts_language == language
                button = QPushButton(("当前" if selected else "使用") + language_labels[language])
                button.setObjectName(f"ttsLanguage_{language}")
                button.setProperty("role", "primary" if selected else "quick")
                button.setEnabled(not selected)
                button.setToolTip("设置后续对话和语音输出语言")
                button.clicked.connect(
                    lambda _checked=False, value=language: self._select_tts_language(value)
                )
                language_layout.addWidget(button, index // 3, index % 3)
            if self._tts_profile_rows:
                language_labels = {"zh": "中文", "jp": "日文", "ja": "日文", "en": "英文"}
                for index, row in enumerate(self._tts_profile_rows[:16]):
                    profile = str(row.get("id", "") or "")
                    selected = profile == self._tts_selected_profile
                    languages = tuple(
                        str(item or "").strip().lower()
                        for item in row.get("languages", ())
                        if str(item or "").strip()
                    )
                    language_hint = "/".join(
                        language_labels.get(item, item.upper()) for item in languages
                    )
                    label = ("当前音色" if selected else "使用 ") + profile
                    if language_hint:
                        label += f" · {language_hint}"
                    enabled = bool(row.get("enabled", True))
                    button = QPushButton(label)
                    button.setObjectName(f"ttsProfile_{index}")
                    button.setProperty("role", "primary" if selected else "quick")
                    button.setEnabled(enabled and not selected)
                    button.setToolTip(
                        "切换后续语音合成音色" if enabled else "该音色当前不可用，请检查语音配置"
                    )
                    button.clicked.connect(
                        lambda _checked=False, value=profile: self._select_tts_profile(value)
                    )
                    profile_layout.addWidget(button, index // 4, index % 4)
            else:
                note = QLabel("未启用多语音 profile；可在配置中心添加")
                note.setObjectName("consoleSectionHint")
                note.setWordWrap(True)
                profile_layout.addWidget(note, 0, 0)

        def set_tts_options(
            self,
            profiles: Sequence[object] | object = (),
            *,
            language: object = "zh",
            active_profile: object = "",
            health: Mapping[str, object] | None = None,
        ) -> None:
            """更新语音 profile/语言按钮，不触发配置文件写入。"""

            if isinstance(profiles, (str, bytes, bytearray, Mapping)) or profiles is None:
                values: tuple[object, ...] = ()
            else:
                try:
                    values = tuple(profiles)  # type: ignore[arg-type]
                except TypeError:
                    values = ()
            rows: list[dict[str, object]] = []
            seen_profiles: set[str] = set()
            for item in values:
                profile = (
                    str(item.get("id", "") or "").strip()[:128]
                    if isinstance(item, Mapping)
                    else str(item or "").strip()[:128]
                )
                if not profile or profile in seen_profiles:
                    continue
                seen_profiles.add(profile)
                raw_languages = item.get("languages", ()) if isinstance(item, Mapping) else ()
                if isinstance(raw_languages, (str, bytes, bytearray, Mapping)):
                    raw_languages = ()
                try:
                    languages = tuple(
                        str(entry or "").strip().lower()[:16]
                        for entry in raw_languages
                        if str(entry or "").strip()
                    )[:8]
                except TypeError:
                    languages = ()
                rows.append(
                    {
                        "id": profile,
                        "languages": languages,
                        "enabled": bool(item.get("enabled", True))
                        if isinstance(item, Mapping)
                        else True,
                    }
                )
                if len(rows) >= 16:
                    break
            self._tts_profile_rows = tuple(rows)
            self._tts_profiles = tuple(str(row["id"]) for row in rows)
            normalized_language = str(language or "zh").strip().lower()
            self._tts_language = (
                normalized_language if normalized_language in {"zh", "jp", "en"} else "zh"
            )
            self._tts_selected_profile = str(active_profile or "").strip()[:128]
            if isinstance(health, Mapping):
                self._tts_health_available = bool(health.get("available", False))
                self._tts_health_pending = bool(health.get("pending", False))
                self._tts_health_message = _safe_ui_text(health.get("message", ""), limit=160)
                self._tts_health_engine = str(health.get("engine", "") or "").strip()[:128]
            signature = (
                tuple(
                    (
                        row.get("id"),
                        tuple(row.get("languages", ())),
                        row.get("enabled"),
                    )
                    for row in self._tts_profile_rows
                ),
                self._tts_language,
                self._tts_selected_profile,
                self._tts_health_available,
                self._tts_health_pending,
                self._tts_health_message,
                self._tts_health_engine,
            )
            if signature == self._tts_options_signature:
                return
            self._tts_options_signature = signature
            if self._tts_status is not None:
                language_label = {"zh": "中文", "jp": "日文", "en": "英文"}[self._tts_language]
                health_label = (
                    "初始化中"
                    if self._tts_health_pending
                    else "已就绪"
                    if self._tts_health_available is True
                    else "未连接"
                    if self._tts_health_available is False
                    else "待检测"
                )
                backend_label = _tts_engine_label(self._tts_health_engine)
                self._tts_status.setText(
                    f"{backend_label} · {language_label} · "
                    f"{self._tts_selected_profile or '默认音色'} · {health_label}"
                )
                self._tts_status.setToolTip(self._tts_health_message)
            self._rebuild_tts_options()

        def _select_tts_profile(self, profile: str) -> None:
            result = self._invoke("select_tts_profile", profile)
            if _operation_succeeded(result):
                self.set_tts_options(
                    self._tts_profile_rows,
                    language=self._tts_language,
                    active_profile=profile,
                )
                self.set_status("语音 profile 已切换")

        def _select_tts_language(self, language: str) -> None:
            result = self._invoke("select_tts_language", language)
            if _operation_succeeded(result):
                self.set_tts_options(
                    self._tts_profile_rows,
                    language=language,
                    active_profile=self._tts_selected_profile,
                )
                self.set_status("输出语言已切换")

        @property
        def expression_options(self) -> tuple[str, ...]:
            """返回当前表情下拉框中的非空选项。"""

            return tuple(self._expression.itemText(i) for i in range(self._expression.count()))

        @property
        def motion_options(self) -> tuple[str, ...]:
            """返回当前动作下拉框中的非空选项。"""

            return tuple(self._motion.itemText(i) for i in range(self._motion.count()))

        def show_settings(self) -> bool:
            """显示配置中心并激活配置页。"""

            if self._ensure_configuration_panel() is None:
                self.set_status("配置中心未启用")
                return False
            self.configurationRequested.emit()
            self.operationTriggered.emit("configuration")
            self._set_configuration_focus(True)
            self.show_and_focus("配置中心")
            return True

        def show_model_settings(self) -> bool:
            """打开点击式模型渠道向导，不要求用户编辑 YAML。"""

            if not self._configuration_enabled:
                self.set_status("配置中心未启用")
                return False
            if self._model_setup_dialog is not None:
                # ``QDialog.open()`` 默认设置 WindowModal，会把桌宠窗口一并
                # 禁用；模型尚未配置时用户仍应能拖动、点击桌宠或双击打开
                # 控制台，因此向导采用可见但非模态的独立窗口。
                self._model_setup_dialog.setWindowModality(Qt.WindowModality.NonModal)
                self.hide()
                self._model_setup_dialog.show()
                positioner = self._model_setup_positioner
                if callable(positioner):
                    try:
                        positioner(self._model_setup_dialog)
                    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                        self.set_status("模型设置窗口已打开；无法自动调整位置")
                self._model_setup_dialog.raise_()
                self._model_setup_dialog.activateWindow()
                return True
            dialog = ModelSetupDialog(
                self._model_setup_draft(),
                channel_drafts=self._model_setup_drafts(),
                on_submit=self._apply_model_setup_payload,
                on_test=self._test_model_connection,
                on_delete=self._delete_model_channel,
                parent=self,
            )
            self._model_setup_dialog = dialog
            restore_console = bool(self.isVisible())

            def finish_model_setup(_result: int) -> None:
                self._model_setup_dialog = None
                # 配置向导与控制台不能同时争夺同一块屏幕；向导关闭后
                # 恢复原来的控制台，避免用户保存后失去操作入口。
                if restore_console and not self._allow_close:
                    self.show_and_focus()

            dialog.finished.connect(finish_model_setup)
            # ``open`` 会把父控制台（进而桌宠窗口）置为 WindowModal。这里
            # 只需要一个点击式配置入口，不应牺牲桌宠的拖动和部位反馈。
            dialog.setWindowModality(Qt.WindowModality.NonModal)
            self.hide()
            dialog.show()
            positioner = self._model_setup_positioner
            if callable(positioner):
                try:
                    positioner(dialog)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    self.set_status("模型设置窗口已打开；无法自动调整位置")
            dialog.raise_()
            dialog.activateWindow()
            self.configurationRequested.emit()
            self.operationTriggered.emit("model_setup")
            return True

        def _test_model_connection(self, payload: Mapping[str, Any]) -> object:
            """提交点击式模型连接测试，并异步回写脱敏结果。"""

            callback = self._callbacks.get("test_model")
            if not callable(callback):
                return {
                    "status": "unavailable",
                    "message": "模型连接测试暂未接入运行时。",
                }
            try:
                result = callback(payload)
            except Exception:
                return {
                    "status": "unavailable",
                    "message": "模型连接测试暂时不可用，请检查连接设置。",
                }
            done = getattr(result, "done", None)
            result_getter = getattr(result, "result", None)
            if not callable(done) or not callable(result_getter):
                return result
            self._model_test_future = result
            card = self._model_setup_dialog.card if self._model_setup_dialog is not None else None

            def poll() -> None:
                if self._model_test_future is not result:
                    return
                try:
                    finished = bool(done())
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    finished = True
                if not finished:
                    QTimer.singleShot(60, poll)
                    return
                try:
                    outcome = result_getter()
                except Exception:
                    outcome = {
                        "status": "unavailable",
                        "message": "模型连接测试失败，请检查服务地址、模型和密钥设置。",
                    }
                self._model_test_future = None
                if card is not None:
                    try:
                        status = (
                            str(outcome.get("status", "") if isinstance(outcome, Mapping) else "")
                            .strip()
                            .lower()
                        )
                        success = status in {"available", "ready", "ok", "completed"}
                        message = (
                            str(outcome.get("message", "") or "")
                            if isinstance(outcome, Mapping)
                            else ""
                        )
                        card.set_test_result(success, message)
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        pass

            QTimer.singleShot(0, poll)
            return {"status": "pending", "message": "正在测试模型连接，请稍候…"}

        def _model_setup_draft(self) -> ModelSetupDraft:
            """从当前配置读取不含明文密钥的渠道草稿。"""

            drafts = self._model_setup_drafts()
            if not drafts:
                return ModelSetupDraft()
            llm = self._configuration.get("llm")
            routing = llm.get("routing") if isinstance(llm, Mapping) else None
            dialogue = routing.get("dialogue") if isinstance(routing, Mapping) else None
            selected_id = (
                str(dialogue.get("channel", dialogue.get("channel_id", "")) or "").strip()
                if isinstance(dialogue, Mapping)
                else ""
            )
            for draft in drafts:
                if draft.channel_id == selected_id:
                    return draft
            return drafts[0]

        def _model_setup_drafts(self) -> tuple[ModelSetupDraft, ...]:
            """读取所有渠道的脱敏草稿，供点击式选择器切换。"""

            return model_setup_drafts(self._configuration)

        def _apply_model_setup_payload(self, payload: Mapping[str, Any]) -> object:
            """将向导结果安全合并到当前配置并复用统一保存校验。"""

            try:
                payload_values = dict(payload)
                persist_plaintext = parse_bool(
                    payload_values.pop("_persist_api_key", False),
                    field_name="_persist_api_key",
                    default=False,
                )
                plaintext_api_key = payload_values.get("api_key") if persist_plaintext else None
                if persist_plaintext:
                    if not isinstance(plaintext_api_key, str) or not plaintext_api_key:
                        raise ValueError("API 密钥不能为空")
                    if len(plaintext_api_key) > 4096 or any(
                        char in plaintext_api_key for char in "\r\n\x00"
                    ):
                        raise ValueError("API 密钥格式无效")
                    payload_values["api_key"] = ""
                    payload_values["api_key_env"] = ""
                draft = ModelSetupDraft.from_mapping(payload_values)
                channel = draft.to_channel_mapping()
                if draft.api_key_env and not draft.clear_persisted_secret:
                    # 同步可选元数据，避免旧环境变量名与新引用不一致。
                    channel["api_key_env"] = draft.api_key_env
                channel_payload = _payload_channel_values(payload_values, draft.channel_id)
                raw_api_key = (
                    channel_payload.get("api_key") if isinstance(channel_payload, Mapping) else None
                )
                explicit_clear = (
                    isinstance(channel_payload, Mapping)
                    and "api_key" in channel_payload
                    and raw_api_key in (None, "")
                ) or (
                    isinstance(channel_payload, Mapping)
                    and "api_key" not in channel_payload
                    and "api_key_env" in channel_payload
                    and not str(channel_payload.get("api_key_env") or "").strip()
                )
                if explicit_clear and not persist_plaintext:
                    # 空密钥只在调用方明确带出密钥字段或清空环境变量元数据时
                    # 才代表删除；向导普通编辑会省略这两个字段以保留已有明文。
                    channel["api_key"] = ""
                    channel.pop("token", None)
                    channel.pop("api_key_env", None)
                    channel.pop("api_key_environment", None)
                if persist_plaintext:
                    # 明文只在当前进程内短暂存在，后续配置保存使用同目录
                    # 原子替换；界面展示、日志和状态文本均不会回显它。
                    channel["api_key"] = plaintext_api_key
                    # 空环境变量字段是持久化模式的明确标记；普通自动保存
                    # 不会凭运行时展开值把原环境引用写回明文。
                    channel["api_key_env"] = ""
                merged = copy.deepcopy(self._configuration)
                llm = merged.get("llm")
                llm_values = dict(llm) if isinstance(llm, Mapping) else {}
                existing = llm_values.get("channels")
                channels = list(existing) if isinstance(existing, list) else []
                replaced = False
                for index, item in enumerate(channels):
                    if (
                        isinstance(item, Mapping)
                        and str(item.get("id", "")).strip() == draft.channel_id
                    ):
                        updated_channel = copy.deepcopy(dict(item))
                        updated_channel.update(channel)
                        if explicit_clear and not persist_plaintext:
                            updated_channel.pop("token", None)
                            updated_channel.pop("api_key_env", None)
                            updated_channel.pop("api_key_environment", None)
                        # 旧版调用方可能完全省略密钥字段；此时保持原始
                        # 引用。当前向导的完整渠道载荷始终带有 api_key
                        # 字段，空值表示用户明确清除，不能再被旧值覆盖。
                        if (
                            not _payload_declares_api_key(payload, draft.channel_id)
                            and not channel.get("api_key")
                            and item.get("api_key")
                        ):
                            updated_channel["api_key"] = item["api_key"]
                        channels[index] = updated_channel
                        replaced = True
                        break
                if not replaced:
                    channels.append(channel)
                routing = llm_values.get("routing")
                routing_values = dict(routing) if isinstance(routing, Mapping) else {}
                dialogue = routing_values.get("dialogue")
                dialogue_values = dict(dialogue) if isinstance(dialogue, Mapping) else {}
                dialogue_id = (
                    str(
                        dialogue_values.get("channel", dialogue_values.get("channel_id", "")) or ""
                    ).strip()
                    if isinstance(dialogue, Mapping)
                    else str(dialogue or "").strip()
                )
                if not replaced or not dialogue_id or dialogue_id == draft.channel_id:
                    dialogue_values["channel"] = draft.channel_id
                    if not dialogue_values.get("required_capabilities"):
                        dialogue_values["required_capabilities"] = ["streaming"]
                    if "fallback_channels" not in dialogue_values:
                        dialogue_values["fallback_channels"] = []
                    routing_values["dialogue"] = dialogue_values
                llm_values["channels"] = channels
                llm_values["routing"] = routing_values
                merged["llm"] = llm_values
            except (TypeError, ValueError) as exc:
                self.set_status(f"模型渠道未保存：{_safe_error_message(exc)}")
                return False
            save_result = self._save_configuration(
                merged,
                allow_plaintext_secrets=persist_plaintext,
                replace_values=True,
            )
            if not save_result:
                self.set_status("模型渠道未保存，请检查配置校验结果")
                return False
            self._refresh_model_status(announce=False)
            if isinstance(save_result, Mapping) and save_result.get("credentials_required") is True:
                self.set_status("模型渠道已保存；请补齐密钥环境变量后重启或重新加载。")
            else:
                self.set_status("模型渠道已保存；运行服务状态正在更新…")
            return save_result if isinstance(save_result, Mapping) else True

        def _delete_model_channel(self, channel_id: str) -> object:
            """删除指定渠道并收敛对话主路由和回退列表。"""

            normalized_id = str(channel_id or "").strip()
            if not normalized_id:
                return {"status": "unavailable", "reason": "渠道名称不能为空"}
            merged = copy.deepcopy(self._configuration)
            llm = merged.get("llm")
            llm_values = dict(llm) if isinstance(llm, Mapping) else {}
            raw_channels = llm_values.get("channels")
            channels = list(raw_channels) if isinstance(raw_channels, list) else []
            remaining: list[object] = []
            removed = False
            for item in channels:
                if (
                    isinstance(item, Mapping)
                    and str(item.get("id", "") or "").strip() == normalized_id
                ):
                    removed = True
                    continue
                remaining.append(item)
            if not removed:
                return {"status": "unavailable", "reason": "模型渠道不存在"}
            llm_values["channels"] = remaining
            routing = llm_values.get("routing")
            routing_values = dict(routing) if isinstance(routing, Mapping) else {}
            next_id = (
                str(remaining[0].get("id", "") or "").strip()
                if remaining and isinstance(remaining[0], Mapping)
                else ""
            )

            def replacement_id(route_values: Mapping[str, Any]) -> str:
                """选择满足路由能力约束的剩余渠道。"""

                required_raw = route_values.get("required_capabilities", ())
                required = (
                    {str(item).strip().lower() for item in required_raw if str(item).strip()}
                    if isinstance(required_raw, (list, tuple, set, frozenset))
                    else {
                        item.strip().lower()
                        for item in str(required_raw or "").split(",")
                        if item.strip()
                    }
                )
                for item in remaining:
                    if not isinstance(item, Mapping):
                        continue
                    try:
                        parsed_channel = channel_from_mapping(item)
                    except (AdapterConfigurationError, TypeError, ValueError):
                        continue
                    if not parsed_channel.is_ready:
                        continue
                    channel_id_value = str(item.get("id", "") or "").strip()
                    raw_capabilities = item.get("capabilities", ("streaming", "tools"))
                    capabilities = (
                        {
                            str(value).strip().lower()
                            for value in raw_capabilities
                            if str(value).strip()
                        }
                        if isinstance(raw_capabilities, (list, tuple, set, frozenset))
                        else {
                            value.strip().lower()
                            for value in str(raw_capabilities or "").split(",")
                            if value.strip()
                        }
                    )
                    if channel_id_value and required.issubset(capabilities):
                        return channel_id_value
                return ""

            for task, raw_route in tuple(routing_values.items()):
                if isinstance(raw_route, Mapping):
                    route_values = dict(raw_route)
                    current_id = str(
                        route_values.get("channel", route_values.get("channel_id", "")) or ""
                    ).strip()
                    if current_id == normalized_id:
                        route_key = "channel" if "channel" in route_values else "channel_id"
                        replacement = replacement_id(route_values)
                        if replacement:
                            route_values[route_key] = replacement
                        else:
                            route_values.pop("channel", None)
                            route_values.pop("channel_id", None)
                            routing_values.pop(task, None)
                            continue
                    fallback = route_values.get("fallback_channels")
                    if isinstance(fallback, (list, tuple)):
                        route_values["fallback_channels"] = [
                            str(item).strip()
                            for item in fallback
                            if str(item).strip() and str(item).strip() != normalized_id
                        ]
                    routing_values[task] = route_values
                elif str(raw_route or "").strip() == normalized_id:
                    routing_values[task] = next_id
            llm_values["routing"] = routing_values
            merged["llm"] = llm_values
            save_result = self._save_configuration(merged, replace_values=True)
            if not save_result:
                return {"status": "unavailable", "reason": "配置保存未完成"}
            return save_result if isinstance(save_result, Mapping) else {"status": "saved"}

        def set_configuration(
            self,
            values: Mapping[str, Any],
            *,
            path: str | Path | None = None,
            internal: bool = False,
        ) -> None:
            """由宿主重载配置后刷新脱敏草稿。"""

            if not isinstance(values, Mapping):
                raise TypeError("configuration values must be a mapping")
            if path is not None:
                self._configuration_path = Path(path).expanduser().resolve()
            updated_values: Mapping[str, Any] = values
            if self._configuration_path is not None:
                try:
                    source_values = parse_editor_yaml(
                        self._configuration_path.read_text(encoding="utf-8")
                    )
                except (OSError, UnicodeError, ValueError):
                    source_values = None
                if source_values is not None:
                    updated_values = preserve_secret_values(source_values, values)
            self._configuration = copy.deepcopy(dict(updated_values))
            self._sync_configured_microphone_device()
            self.apply_theme_configuration(self._configuration)
            try:
                self._configuration_session = (
                    ConfigurationEditSession(
                        self._configuration_path,
                        self._configuration,
                    )
                    if self._configuration_path is not None
                    else None
                )
            except (ConfigurationError, OSError):
                self._configuration_session = None
            model_setup_dialog = self._model_setup_dialog
            if (
                not internal
                and model_setup_dialog is not None
                and model_setup_dialog.isVisible()
                and not model_setup_dialog.auto_save_in_flight
            ):
                model_setup_dialog.mark_external_configuration_changed()
            self._refresh_model_status()
            rendering_values = self._configuration.get("rendering")
            configured_model = (
                rendering_values.get("model") if isinstance(rendering_values, Mapping) else ""
            )
            if configured_model in self._renderer_model_options:
                pending_model = self._renderer_model_pending
                committed_model = self._renderer_model_committed
                self.set_renderer_models(configured_model, self._renderer_model_options)
                # 保存配置可能正发生在一次页面内热切换请求期间。配置文件
                # 的新键可以先展示，但当前活动模型仍应保留到宿主回执；
                # 失败时控制台才能准确恢复旧选择并允许再次尝试。
                if pending_model:
                    self._renderer_model_pending = pending_model
                    self._renderer_model_committed = committed_model
                    self._update_renderer_model_button()
            if self._configuration_panel is None:
                return
            self._configuration_panel.set_values(
                self._configuration,
                path=self._configuration_path,
            )

        def apply_theme_configuration(self, values: Mapping[str, Any]) -> None:
            """热替换控制台及子面板主题，不重置配置草稿。"""

            if not isinstance(values, Mapping):
                raise TypeError("configuration values must be a mapping")
            self._md3_theme, _stylesheet = _console_stylesheet(values)
            self._fancy_style_controller.updateTheme(
                self._md3_theme,
                extra_stylesheet=_console_component_stylesheet(self._md3_theme),
            )
            module_icons = {
                "idle": "info",
                "checking": "refresh",
                "available": "success",
                "degraded": "warning",
                "unavailable": "error",
            }
            for module_id, status_icon in self._module_status_icons.items():
                state_label = self._module_status_labels.get(module_id)
                state = str(state_label.property("state") if state_label is not None else "idle")
                status_icon.setPixmap(
                    fancy_icon(module_icons.get(state, "info"), theme=self._md3_theme).pixmap(
                        18,
                        18,
                    )
                )
            apply_model_theme = getattr(self._model_setup_dialog, "apply_theme", None)
            if callable(apply_model_theme):
                apply_model_theme(self._md3_theme)
            configuration_panel = self._configuration_panel
            apply_panel_theme = getattr(configuration_panel, "apply_theme_configuration", None)
            if callable(apply_panel_theme):
                apply_panel_theme(values)

        @staticmethod
        def _configuration_result_success(result: object) -> bool:
            if isinstance(result, bool):
                return result
            if result is None:
                return True
            if isinstance(result, Mapping):
                raw_status = result.get("status", result.get("state", ""))
                status = str(getattr(raw_status, "value", raw_status) or "")
                if not status:
                    return True
                return status.strip().lower() in {
                    "ok",
                    "available",
                    "accepted",
                    "completed",
                    "saved",
                    "success",
                    "validated",
                    "updated",
                    "requested",
                }
            # 配置回调契约只接受 bool、None 或状态映射；任意其它对象
            # （例如误返回的整数/异常对象）不能被当作保存成功。
            return False

        def _default_validate_configuration(self, values: Mapping[str, Any]) -> object:
            """在宿主未提供回调时复用运行时的无副作用配置校验。"""

            # 延迟导入避免无头核心导入控制台时形成 GUI/运行时循环。
            from app.runtime import validate_runtime_configuration
            from config.loader import LoadedConfiguration

            path = self._configuration_path or (Path.cwd() / "config.yaml")
            validate_runtime_configuration(LoadedConfiguration(path.resolve(), dict(values)))
            return {"status": "validated"}

        def _merged_configuration(self, values: Mapping[str, Any]) -> dict[str, Any]:
            merged = merge_editor_values(self._configuration, values)
            if not isinstance(merged, Mapping):  # pragma: no cover - 根映射受面板保证
                raise ValueError("configuration must be a mapping")
            return validate_editor_values(merged)

        def _validate_configuration(self, values: object) -> None:
            if self._configuration_panel is None:
                return
            try:
                if not isinstance(values, Mapping):
                    raise ValueError("configuration must be a mapping")
                merged = self._merged_configuration(values)
                callback = self._config_callbacks.get("validate") or self._callbacks.get(
                    "config_validate"
                )
                result = (
                    callback(merged)
                    if callable(callback)
                    else self._default_validate_configuration(merged)
                )
                if not self._configuration_result_success(result):
                    raise ValueError(str(result or "配置校验未通过"))
            except Exception as exc:
                self._configuration_panel.set_save_result(
                    False, f"配置校验失败：{_safe_error_message(exc)}"
                )
                return
            self._configuration_panel.set_save_result(True, "配置校验通过；尚未写入文件。")

        def _save_configuration(
            self,
            values: object,
            *,
            allow_plaintext_secrets: bool = False,
            replace_values: bool = False,
        ) -> object:
            configuration_panel = self._configuration_panel
            save_result: object = True
            try:
                if not isinstance(values, Mapping):
                    raise ValueError("configuration must be a mapping")
                merged = (
                    validate_editor_values(values)
                    if replace_values
                    else self._merged_configuration(values)
                )
                callback_key = "save_plaintext" if allow_plaintext_secrets else "save"
                callback = self._config_callbacks.get(callback_key) or self._callbacks.get(
                    "config_save_plaintext" if allow_plaintext_secrets else "config_save"
                )
                if (
                    allow_plaintext_secrets
                    and not callable(callback)
                    and self._configuration_path is None
                ):
                    raise ValueError("未接入明确的文件密钥保存回调")
                if callable(callback):
                    result = callback(copy.deepcopy(merged))
                    save_result = result
                    if not self._configuration_result_success(result):
                        reason = result.get("reason") if isinstance(result, Mapping) else result
                        raise ValueError(str(reason or "配置保存未通过"))
                    result_values = result.get("values") if isinstance(result, Mapping) else None
                    if isinstance(result_values, Mapping):
                        merged = validate_editor_values(result_values)
                    runtime_status = (
                        str(result.get("runtime_status", "") or "").strip().lower()
                        if isinstance(result, Mapping)
                        else ""
                    )
                    if isinstance(result, Mapping) and result.get("credentials_required") is True:
                        message = "配置已保存；待补密钥，请补齐环境变量后重启或重新加载。"
                    elif runtime_status == "pending":
                        message = "配置已保存，正在应用模型配置…"
                    elif runtime_status == "restart_required":
                        message = "配置已保存；当前运行仍需重启后生效。"
                    elif runtime_status == "reloaded":
                        message = "配置已保存；模型配置已立即生效。"
                    else:
                        message = "配置已保存；部分运行服务可能需要重启后生效。"
                elif self._configuration_path is not None:
                    # 即使宿主没有提供运行时回调，也必须按磁盘原始 YAML
                    # 恢复未修改的环境变量引用，不能把运行时展开值写回。
                    source = parse_editor_yaml(self._configuration_path.read_text(encoding="utf-8"))
                    # 配置中心允许用户明确在文件中持久化密钥；界面仍以
                    # ``***`` 脱敏展示，且写入沿用同目录原子替换。
                    persisted = prepare_persisted_values(
                        source,
                        merged,
                        allow_plaintext_secrets=allow_plaintext_secrets,
                    )
                    if not isinstance(persisted, Mapping):
                        raise ValueError("配置必须是映射")
                    self._default_validate_configuration(persisted)
                    credentials_required = False
                    try:
                        from config.loader import expand_environment_values

                        expand_environment_values(persisted)
                    except ConfigurationError:
                        if _only_missing_channel_credentials(persisted):
                            credentials_required = True
                        else:
                            raise
                    session = self._configuration_session
                    session_matches = bool(
                        session is not None and session.path == self._configuration_path
                    )
                    atomic_write_configuration(
                        self._configuration_path,
                        persisted,
                        mode=0o600 if allow_plaintext_secrets else None,
                        expected_digest=(session.source_digest if session_matches else None),
                        expect_missing=bool(session_matches and session.expects_missing_file),
                    )
                    if credentials_required:
                        message = "配置已保存；待补密钥，请补齐环境变量后重启或重新加载。"
                        save_result = {
                            "status": "saved",
                            "credentials_required": True,
                            "restart_required": True,
                        }
                    else:
                        message = "配置已原子保存；部分运行服务可能需要重启后生效。"
                        save_result = {"status": "saved"}
                else:
                    raise ValueError("未配置 config_save 回调或配置文件路径")
            except Exception as exc:
                message = (
                    "配置保存失败：配置文件已被外部修改，请重新加载后再保存"
                    if isinstance(exc, ConfigurationConflictError)
                    else f"配置保存失败：{_safe_error_message(exc)}"
                )
                if configuration_panel is not None:
                    configuration_panel.set_save_result(False, message)
                else:
                    self.set_status(message)
                return False
            self._configuration = copy.deepcopy(merged)
            self._sync_configured_microphone_device()
            try:
                self._configuration_session = (
                    ConfigurationEditSession(
                        self._configuration_path,
                        self._configuration,
                    )
                    if self._configuration_path is not None
                    else None
                )
            except (ConfigurationError, OSError):
                self._configuration_session = None
            self._refresh_model_status(announce=False)
            if configuration_panel is not None:
                configuration_panel.set_save_result(True, message, values=self._configuration)
            else:
                self.set_status(message)
            return save_result if isinstance(save_result, Mapping) else True

        def _invoke(self, name: str, *args: object) -> object | None:
            self._last_invoke_failed = False
            self._last_invoke_unavailable = False
            callback = self._callbacks.get(name)
            if not callable(callback):
                self._last_invoke_unavailable = True
                self.set_status(f"功能不可用：{_operation_label(name)}")
                return None
            try:
                result = callback(*args)
            except Exception as exc:  # GUI 操作失败只反馈到控制台
                self._last_invoke_failed = True
                label = _operation_label(name)
                self.set_status(f"{label}失败：{type(exc).__name__}")
                self.append_line(f"系统：{label}失败：{_safe_error_message(exc)}")
                return None
            status = _status_value(result)
            if status and status not in {"available", "requested", "completed"}:
                self.set_status(f"{_operation_label(name)}：{_friendly_invoke_status(status)}")
            return result

        def _run_control_action(self, name: str) -> object | None:
            """运行恢复/诊断按钮并把明确的结果反馈给用户。"""

            signal_map = {
                "show_pet": self.showPetRequested,
                "toggle_visibility": self.visibilityToggleRequested,
                "toggle_always_on_top": self.alwaysOnTopRequested,
                "restore_click_through": self.restoreClickThroughRequested,
                "foreground_window": self.foregroundWindowRequested,
                "list_processes": self.processesRequested,
                "module_status": self.moduleStatusRequested,
            }
            signal = signal_map.get(name)
            if signal is not None:
                signal.emit()
            self.operationTriggered.emit(name)
            result = self._invoke(name)
            if result is None:
                if not self._last_invoke_failed and not self._last_invoke_unavailable:
                    self.set_status(f"{_operation_label(name)}请求已提交，等待宿主结果…")
                return None
            if name in {
                "foreground_window",
                "list_processes",
                "module_status",
                "api_audit",
                "log_records",
            }:
                self.set_diagnostic_result(name, result)
            if (
                name == "toggle_always_on_top"
                and _status_value(result).strip().lower() == "requested"
            ):
                self.set_always_on_top_status(result)
                enabled = bool(result.get("enabled")) if isinstance(result, Mapping) else False
                self.set_status(
                    "窗口置顶切换已提交（正在应用）" if enabled else "取消置顶已提交（正在应用）"
                )
                return result
            if name == "toggle_always_on_top" and _wayland_topmost_request_accepted(result):
                self.set_always_on_top_status(result)
                enabled = bool(result.get("enabled")) if isinstance(result, Mapping) else False
                self.set_status(
                    "窗口置顶已请求（最终状态由桌面合成器决定）"
                    if enabled
                    else "取消置顶已请求（最终状态由桌面合成器决定）"
                )
                return result
            if not _operation_succeeded(result):
                detail = _safe_ui_text(_detail_value(result))
                suffix = f"：{detail}" if detail else ""
                self.set_status(f"{_operation_label(name)}未完成{suffix}")
                return result
            if isinstance(result, Mapping):
                visible = result.get("visible")
                if name in {"show_pet", "toggle_visibility"} and isinstance(visible, bool):
                    self.set_status("桌宠已显示" if visible else "桌宠已隐藏")
                    return result
                enabled = result.get("enabled")
                if name == "toggle_always_on_top" and isinstance(enabled, bool):
                    self.set_always_on_top_status(result)
                    self.set_status("窗口置顶已开启" if enabled else "窗口置顶已关闭")
                    return result
                if name == "restore_click_through" and isinstance(enabled, bool):
                    self.set_status("点击已恢复" if not enabled else "点击穿透仍开启")
                    return result
                if name in {"foreground_window", "list_processes", "module_status"}:
                    self.set_status(
                        "前台窗口读取请求已提交"
                        if name == "foreground_window"
                        else "进程读取请求已提交"
                        if name == "list_processes"
                        else "模块检查请求已提交"
                    )
                    return result
            messages = {
                "show_pet": "桌宠显示请求已完成",
                "toggle_visibility": "桌宠可见状态已更新",
                "toggle_always_on_top": "窗口置顶状态已更新",
                "restore_click_through": "点击恢复请求已完成",
                "foreground_window": "前台窗口读取请求已提交",
                "list_processes": "进程读取请求已提交",
                "module_status": "模块检查请求已提交",
                "restart_application": "桌宠重启请求已提交",
            }
            self.set_status(messages.get(name, f"{_operation_label(name)} 已完成"))
            return result

        def _run_rescue_action(self, name: str) -> object | None:
            """执行首屏救援按钮，并把结果同步到救援栏。"""

            if name == "center_pet":
                result = self._center_pet()
            else:
                result = self._run_control_action(name)
            if result is None:
                message = self._status.text()
            elif _operation_succeeded(result) or (
                name == "toggle_always_on_top" and _wayland_topmost_request_accepted(result)
            ):
                message = self._status.text()
            else:
                message = f"{_operation_label(name)}未完成"
            if isinstance(result, Mapping):
                button = self._rescue_buttons.get(name)
                if button is not None:
                    if name in {"show_pet", "toggle_visibility"} and isinstance(
                        result.get("visible"), bool
                    ):
                        button.setText("隐藏桌宠" if result["visible"] else "显示桌宠")
                    elif name == "toggle_always_on_top" and isinstance(result.get("enabled"), bool):
                        button.setText("取消置顶" if result["enabled"] else "窗口置顶")
                    elif name == "restore_click_through" and isinstance(
                        result.get("enabled"), bool
                    ):
                        button.setText("恢复鼠标" if result["enabled"] else "开启穿透")
            if hasattr(self, "_rescue_status"):
                self._rescue_status.setText(message)
            return result

        def _submit(self) -> None:
            value = self._input.text().strip()
            if not value:
                self.set_status("请输入消息后再发送")
                return
            self._input.clear()
            self.append_line(f"我：{value}")
            self.conversationSubmitted.emit(value)
            self.operationTriggered.emit("conversation")
            if callable(self._callbacks.get("submit")):
                result = self._invoke("submit", value)
                if self._last_invoke_failed:
                    return
                if result is None or _operation_succeeded(result) or _operation_pending(result):
                    self.set_status("消息已提交，等待模型回复…")
                else:
                    summary = _safe_result_summary(result)
                    self.set_status(f"消息未提交：{summary}")
                    self.append_line(f"系统：消息未提交：{summary}")
            else:
                self.commandSubmitted.emit(value)
                self.set_status("消息已提交，等待宿主处理…")

        def _configured_microphone_device_id(self) -> str:
            asr = self._configuration.get("asr")
            if not isinstance(asr, Mapping):
                return ""
            capture = asr.get("capture")
            if not isinstance(capture, Mapping):
                return ""
            device_id = capture.get("device_id", "")
            return device_id if isinstance(device_id, str) else ""

        def _sync_configured_microphone_device(self) -> None:
            selector = getattr(self, "_microphone_device_selector", None)
            if not isinstance(selector, AudioInputSelector):
                return
            self._microphone_device_updating = True
            try:
                selector.set_devices(
                    self._microphone_input_devices,
                    loaded=self._microphone_devices_loaded,
                    selected_device_id=self._configured_microphone_device_id(),
                )
            finally:
                self._microphone_device_updating = False

        def _load_microphone_input_devices(self) -> tuple[AudioInputDeviceChoice, ...]:
            """响应用户展开或刷新选择器，不在首屏构建期间枚举设备。"""

            if self._microphone_closed:
                return ()
            loader = self._microphone_device_loader
            if not callable(loader):
                self.set_status("输入设备列表暂不可用；仍可跟随系统默认设备。")
                return self._microphone_input_devices
            try:
                devices = loader()
                if not isinstance(devices, Sequence) or isinstance(
                    devices, (str, bytes, bytearray)
                ):
                    raise TypeError("audio input device loader returned an invalid value")
                self.set_microphone_input_devices(devices, loaded=True)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                self.set_status("输入设备刷新失败；未更改当前选择。")
                return self._microphone_input_devices
            return self._microphone_input_devices

        def set_microphone_input_devices(
            self,
            devices: Sequence[AudioInputDeviceChoice],
            *,
            loaded: bool = True,
        ) -> None:
            """用用户可读描述刷新选择器，不把设备 ID 投影到状态或日志。"""

            if self._microphone_closed:
                return
            if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes, bytearray)):
                raise TypeError("audio input devices must be a sequence")
            normalized = tuple(devices)
            if any(not isinstance(device, AudioInputDeviceChoice) for device in normalized):
                raise TypeError("audio input device entries are invalid")
            self._microphone_input_devices = normalized
            self._microphone_devices_loaded = bool(loaded)
            selected = self._configured_microphone_device_id()
            self._microphone_device_updating = True
            try:
                self._microphone_device_selector.set_devices(
                    normalized,
                    loaded=self._microphone_devices_loaded,
                    selected_device_id=selected,
                )
            finally:
                self._microphone_device_updating = False
            panel = self._configuration_panel
            if panel is not None:
                panel.set_audio_input_devices(normalized, loaded=self._microphone_devices_loaded)

        def _select_microphone_input_device(self, _index: int) -> None:
            if self._microphone_device_updating or self._microphone_closed:
                return
            selected = self._microphone_device_selector.selected_device_id()
            previous = self._configured_microphone_device_id()
            if selected == previous:
                return
            updated = copy.deepcopy(self._configuration)
            asr = updated.get("asr")
            asr_values = dict(asr) if isinstance(asr, Mapping) else {}
            capture = asr_values.get("capture")
            capture_values = dict(capture) if isinstance(capture, Mapping) else {}
            capture_values["device_id"] = selected
            asr_values["capture"] = capture_values
            updated["asr"] = asr_values
            result = self._save_configuration(updated, replace_values=True)
            if result is False:
                self._microphone_device_updating = True
                try:
                    self._microphone_device_selector.set_selected_device_id(previous)
                finally:
                    self._microphone_device_updating = False
                return
            self.set_status(
                "输入设备已设置为跟随系统默认。"
                if not selected
                else "输入设备选择已保存；设备可用时可开始录音。"
            )

        def _toggle_microphone(self) -> None:
            """只提交用户点击；录音和识别终态由宿主回写。"""

            if self._microphone_closed or self._microphone_state not in {
                "idle",
                "completed",
                "recording",
                "denied",
                "unavailable",
            }:
                return
            result = self._invoke("microphone_toggle")
            if self._last_invoke_failed or self._last_invoke_unavailable:
                return
            if isinstance(result, Mapping):
                state = str(result.get("status", "") or "").strip().casefold()
                if state in {
                    "disabled",
                    "idle",
                    "waiting_playback",
                    "permission_pending",
                    "recording",
                    "transcribing",
                    "completed",
                    "denied",
                    "unavailable",
                    "closed",
                }:
                    self.set_microphone_state(state)

        def set_microphone_state(self, state: object, message: object = "") -> None:
            """投影固定麦克风状态，不接受设备 ID、录音或转写正文。"""

            normalized = str(state or "").strip().casefold()
            allowed = {
                "disabled",
                "idle",
                "waiting_playback",
                "permission_pending",
                "recording",
                "captured",
                "transcribing",
                "completed",
                "denied",
                "unavailable",
                "closed",
            }
            if normalized not in allowed or (self._microphone_closed and normalized != "closed"):
                return
            if normalized == "closed":
                self._microphone_closed = True
            self._microphone_state = normalized
            labels = {
                "disabled": "语音输入已关闭",
                "idle": "语音输入待命",
                "waiting_playback": "正在停止语音播放…",
                "permission_pending": "等待麦克风权限确认…",
                "recording": "录音中，再次点击后停止并识别",
                "captured": "录音完成，准备识别",
                "transcribing": "正在识别语音…",
                "completed": "识别完成，文字已填入输入框",
                "denied": "麦克风权限已被拒绝",
                "unavailable": "语音输入暂不可用",
                "closed": "语音输入已关闭",
            }
            safe_message = _safe_ui_text(message, limit=120).replace("\n", " ").strip()
            visible_message = safe_message or labels[normalized]
            self._microphone_status.setText(visible_message)
            self._microphone_status.setProperty("state", normalized)
            microphone_detail_states = {
                "waiting_playback",
                "permission_pending",
                "recording",
                "captured",
                "transcribing",
                "denied",
                "unavailable",
            }
            self._microphone_status.setVisible(
                not bool(self.property("compactLayout")) or normalized in microphone_detail_states
            )
            self._microphone_button.setProperty("state", normalized)
            self._microphone_button.setText(
                "停止并识别" if normalized == "recording" else "开始录音"
            )
            self._microphone_button.setEnabled(
                not self._microphone_closed
                and normalized in {"idle", "completed", "recording", "denied", "unavailable"}
                and callable(self._callbacks.get("microphone_toggle"))
            )
            for widget in (self._microphone_button, self._microphone_status):
                widget.style().unpolish(widget)
                widget.style().polish(widget)

        def set_microphone_transcription(self, text: object) -> None:
            """只把识别正文填入输入框；不发送、不记入活动记录。"""

            if self._microphone_closed:
                return
            value = " ".join(str(text or "").replace("\x00", "").split())
            if not value:
                return
            existing = self._input.text().strip()
            combined = f"{existing} {value}".strip() if existing else value
            self._input.setText(combined[:20_000])
            self._input.setFocus(Qt.OtherFocusReason)
            self._input.setCursorPosition(len(self._input.text()))

        def _stop(self) -> None:
            self.stopRequested.emit()
            self.operationTriggered.emit("stop")
            result = self._invoke("stop")
            if self._last_invoke_unavailable or self._last_invoke_failed:
                return
            status = _status_value(result).strip().lower()
            detail = _detail_value(result)
            if status == "unavailable":
                if "当前没有正在生成的对话" in detail:
                    self.set_status("当前没有正在生成的消息")
                else:
                    self.set_status("停止未提交")
                return
            if status == "cancelled" and "已取消等待中的消息" in detail:
                self.set_status("已取消等待中的消息")
                return
            if result is None or _operation_succeeded(result) or _operation_pending(result):
                self.set_status("正在中断当前对话…")
                return
            self.set_status("停止未提交")
            self.append_line(f"系统：停止未执行：{_safe_result_summary(result)}")

        def _refresh_position_limits(self) -> tuple[int, int, int, int] | None:
            """读取宿主声明的可移动边界并同步坐标输入范围。"""

            result = self._invoke("movement_bounds")
            if not self.set_position_bounds(result):
                return self._position_bounds
            return self._position_bounds

        def set_position_bounds(self, value: object) -> bool:
            """由宿主刷新桌宠可移动边界，并限制当前输入值。"""

            bounds = _bounds_value(value)
            if bounds is None:
                return False
            self._position_bounds = bounds
            left, top, right, bottom = bounds
            self._position_x.setRange(left, right)
            self._position_y.setRange(top, bottom)
            self._position_x.setValue(max(left, min(self._position_x.value(), right)))
            self._position_y.setValue(max(top, min(self._position_y.value(), bottom)))
            return True

        def _refresh_position(self) -> object | None:
            """读取当前桌宠坐标并更新输入框。"""

            result = self._invoke("position")
            position = _xy_value(result)
            if position is None:
                self.set_status("当前位置不可用；请确认桌宠窗口已创建")
                return result
            self.set_position(*position)
            self._refresh_position_limits()
            self._position_status.setText(f"当前坐标：({position[0]}, {position[1]})")
            self.set_status(f"当前桌宠位置：({position[0]}, {position[1]})")
            return result

        def _screen_center(self) -> tuple[int, int] | None:
            """在没有宿主边界回调时按 Qt 可用屏幕区域计算中心。"""

            screens = QGuiApplication.screens()
            if not screens:
                return None
            # 使用当前控制台所在屏幕；控制台被移到其他屏幕后，居中按钮
            # 应跟随用户当前操作的屏幕，而不是固定主屏幕。
            screen = QGuiApplication.screenAt(self.frameGeometry().center())
            if screen is None:
                screen = QGuiApplication.primaryScreen()
            if screen is None:
                return None
            area = screen.availableGeometry()
            return int(area.center().x()), int(area.center().y())

        def _center_pet(self) -> object | None:
            """把桌宠窗口左上角移动到可用区域中心并提交一次请求。"""

            bounds = self._refresh_position_limits()
            if bounds is not None:
                left, top, right, bottom = bounds
                x, y = (left + right) // 2, (top + bottom) // 2
            else:
                center = self._screen_center()
                if center is None:
                    self.set_status("无法确定当前屏幕区域")
                    return None
                x, y = center
            self.set_position(x, y)
            return self._move_pet()

        def _set_expression(self) -> None:
            """应用高级表情选择，并与快捷按钮保持同一执行路径。"""

            self._apply_expression_value(self._expression.currentText())

        def _play_motion(self) -> None:
            """播放高级动作选择，并与快捷按钮保持同一执行路径。"""

            self._apply_motion_value(self._motion.currentText())

        def _play_expression_sequence(self) -> object | None:
            names = tuple(
                item.strip()
                for item in self._expression_sequence.text().replace("，", ",").split(",")
                if item.strip()
            )
            if not names:
                selected = str(self._expression.currentText() or "").strip()
                names = (selected,) if selected else ()
            if not names or len(names) > 8:
                self.set_status("表情编排需要 1 到 8 个表情")
                return None
            raw_weights = tuple(
                item.strip()
                for item in self._expression_weights.text().replace("，", ",").split(",")
                if item.strip()
            )
            if raw_weights and len(raw_weights) != len(names):
                self.set_status("混合权重数量必须与表情数量一致")
                return None
            try:
                weights = tuple(float(item) for item in raw_weights) if raw_weights else ()
            except (TypeError, ValueError, OverflowError):
                self.set_status("混合权重必须是 0 到 1 之间的数值")
                return None
            if any(not math.isfinite(item) or item < 0.0 or item > 1.0 for item in weights):
                self.set_status("混合权重必须是 0 到 1 之间的数值")
                return None
            try:
                parameters = _action_parameters(self._expression_parameters.text())
            except ValueError as exc:
                self.set_status(str(exc))
                return None
            duration = float(self._expression_duration.value())
            transition = float(self._expression_transition.value())
            payload = {
                "mode": str(self._expression_mode.currentData() or "sequence"),
                "loop": bool(self._expression_loop.isChecked()),
                "restore": str(self._expression_restore.text() or "neutral").strip(),
                "expressions": [
                    {
                        "name": name,
                        "weight": weights[index] if weights else 1.0,
                        "duration_seconds": duration,
                        "transition_seconds": transition,
                        "parameters": parameters,
                    }
                    for index, name in enumerate(names)
                ],
            }
            try:
                ExpressionRequest.from_arguments(payload)
            except ValueError:
                self.set_status("表情编排参数无效，请检查名称、权重、持续时间和恢复表情")
                return None
            result = self._invoke("expression_request", payload)
            if _operation_succeeded(result):
                self.set_status(f"已开始表情编排：{len(names)} 个表情")
            else:
                self.set_status("表情编排未开始，请检查当前模型能力")
            return result

        def _play_motion_request(self) -> object | None:
            name = str(self._motion.currentText() or "").strip()
            if not name:
                self.set_status("请选择动作")
                return None
            try:
                parameters = _action_parameters(self._motion_parameters.text())
            except ValueError as exc:
                self.set_status(str(exc))
                return None
            payload: dict[str, object] = {
                "name": name,
                "duration_seconds": float(self._motion_duration.value()),
                "transition_seconds": float(self._motion_transition.value()),
                "loop": bool(self._motion_loop.isChecked()),
                "parameters": parameters,
            }
            try:
                MotionRequest.from_arguments(payload)
            except ValueError:
                self.set_status("动作参数无效，请检查名称、持续时间和过渡时间")
                return None
            result = self._invoke("motion_request", payload)
            if _operation_succeeded(result):
                self.set_status(f"动作已开始：{name}")
            else:
                self.set_status("动作未开始，请检查当前模型能力")
            return result

        def _move_pet(self) -> object | None:
            """提交控制台中的绝对屏幕坐标。"""

            x = int(self._position_x.value())
            y = int(self._position_y.value())
            bounds = self._position_bounds
            was_clamped = False
            if bounds is not None:
                left, top, right, bottom = bounds
                bounded_x = max(left, min(x, right))
                bounded_y = max(top, min(y, bottom))
                if (bounded_x, bounded_y) != (x, y):
                    x, y = bounded_x, bounded_y
                    was_clamped = True
                    self.set_position(x, y)
            self.moveRequested.emit(x, y)
            self.operationTriggered.emit("move_to")
            result = self._invoke("move_to", x, y)
            if result is None:
                return None
            status = _status_value(result).strip().lower()
            if status in {"available", "requested", "completed"}:
                suffix = "（已限制在屏幕范围）" if was_clamped else ""
                self._position_status.setText(f"当前坐标：({x}, {y})")
                self.set_status(f"桌宠位置：({x}, {y}){suffix}")
            elif status == "pending":
                suffix = "（已限制在屏幕范围）" if was_clamped else ""
                self._position_status.setText(f"等待移动：({x}, {y})")
                self.set_status(f"桌宠移动已排队：({x}, {y}){suffix}")
            elif status == "degraded":
                detail = _detail_value(result).lower()
                if "wayland" in detail or "compositor" in detail:
                    suffix = "（Wayland 最终位置由 compositor 决定）"
                else:
                    suffix = "（平台返回降级结果）"
                self._position_status.setText(f"请求坐标：({x}, {y})")
                self.set_status(f"桌宠位置请求已提交：({x}, {y}){suffix}")
            else:
                self._position_status.setText("位置请求未完成")
                self.set_status(f"桌宠位置不可用：({x}, {y})")
                self.append_line(f"系统：移动到({x}, {y})：{_safe_result_summary(result)}")
            return result

        def _set_display_size(self, preset: str) -> object | None:
            """提交点击式显示大小预设，不让用户编辑缩放数值。"""

            key = str(preset or "").strip().lower()
            if key not in DISPLAY_SIZE_PRESETS:
                self.set_status("显示大小预设不可用")
                return None
            self.operationTriggered.emit("display_size")
            result = self._invoke("set_display_size", key)
            if _operation_succeeded(result):
                self.set_status(f"桌宠大小：{DISPLAY_SIZE_LABELS[key]}")
            else:
                self.set_status("桌宠大小调整失败")
                self.append_line(f"系统：显示大小：{_safe_result_summary(result)}")
            return result

        @staticmethod
        def _approval_field(request: object, field_name: str, default: object = "") -> object:
            if isinstance(request, Mapping):
                return request.get(field_name, default)
            return getattr(request, field_name, default)

        def set_pending_approvals(self, requests: Sequence[object]) -> None:
            """刷新待审批列表，只保存审批标识而不复制工具参数。"""

            entries: list[tuple[str, str]] = []
            for request in requests:
                approval_id = str(self._approval_field(request, "approval_id", "") or "").strip()
                if not approval_id:
                    continue
                identity = _safe_approval_identity(
                    self._approval_field(
                        request,
                        "display_name",
                        self._approval_field(request, "identity", "工具"),
                    )
                )
                summary = _safe_approval_summary(
                    self._approval_field(request, "safe_summary", "需要确认该操作")
                )
                entries.append((approval_id, f"{identity}：{summary}"))
            signature = tuple(entries)
            expected_count = len(entries) if entries else 1
            if (
                signature == self._approval_signature
                and self._approval_list.count() == expected_count
            ):
                return
            selected_id = self.selected_approval_id()
            self._approval_signature = signature
            self._approval_list.clear()
            if not entries:
                empty = QListWidgetItem("暂无待审批工具调用")
                empty.setFlags(Qt.ItemFlag.NoItemFlags)
                self._approval_list.addItem(empty)
                return
            selected_index = 0
            for index, (approval_id, label) in enumerate(entries):
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, approval_id)
                self._approval_list.addItem(item)
                if approval_id == selected_id:
                    selected_index = index
            self._approval_list.setCurrentRow(selected_index)
            # 首次出现待审批操作时自动展开对应区域；后续刷新不强制改写
            # 用户已经选择的展开状态，避免定时刷新造成界面跳动。
            if not self._secondary_actions_panel.isVisible():
                self._secondary_actions_panel.setVisible(True)
                self._secondary_actions_toggle.setText("收起更多操作")

        def selected_approval_id(self) -> str:
            """返回当前选中的审批标识。"""

            item = self._approval_list.currentItem()
            if item is None:
                return ""
            value = item.data(Qt.ItemDataRole.UserRole)
            return str(value or "").strip()

        def _approve_selected_approval(self) -> None:
            approval_id = self.selected_approval_id()
            if not approval_id:
                self.set_status("请先选择待审批工具调用")
                return
            result = self._invoke(
                "approve_approval", approval_id, bool(self._grant_session.isChecked())
            )
            if _operation_succeeded(result):
                self.set_status("已提交批准")
            else:
                self.set_status("批准未提交")
                self.append_line(f"系统：批准未执行：{_safe_result_summary(result)}")

        def _deny_selected_approval(self) -> None:
            approval_id = self.selected_approval_id()
            if not approval_id:
                self.set_status("请先选择待审批工具调用")
                return
            result = self._invoke("deny_approval", approval_id)
            if _operation_succeeded(result):
                self.set_status("已提交拒绝")
            else:
                self.set_status("拒绝未提交")
                self.append_line(f"系统：拒绝未执行：{_safe_result_summary(result)}")

        def set_position(self, x: int, y: int) -> None:
            """同步控制台中的坐标输入，不触发移动请求。"""

            self._position_x.setValue(int(x))
            self._position_y.setValue(int(y))
            if hasattr(self, "_position_status"):
                self._position_status.setText(f"当前坐标：({int(x)}, {int(y)})")

        def set_diagnostic_result(self, name: str, result: object) -> None:
            """在诊断卡片中展示脱敏的前台窗口/进程摘要。"""

            if not hasattr(self, "_diagnostic_output"):
                return
            observation_status = getattr(self, "_observation_status", None)
            label = str(name or "").strip()
            if isinstance(result, Mapping) and label in {"前台窗口", "foreground_window"}:
                title = _safe_ui_text(result.get("title"), limit=120) or "无标题"
                process_name = _safe_ui_text(result.get("process_name")) or "未知进程"
                pid = _safe_ui_text(result.get("pid"), limit=32)
                lines = [f"前台窗口：{title}", f"进程：{process_name}"]
                if pid:
                    lines.append(f"PID：{pid}")
                summary = "\n".join(lines)
                self._diagnostic_output.setPlainText(summary)
                if observation_status is not None:
                    observation_status.setText(summary)
                return
            if isinstance(result, Mapping) and label in {"进程列表", "list_processes"}:
                rows = result.get("processes")
                lines = []
                if isinstance(rows, (list, tuple)):
                    for row in rows[:20]:
                        if not isinstance(row, Mapping):
                            continue
                        process_name = _safe_ui_text(row.get("name") or row.get("process_name"))
                        if not process_name:
                            continue
                        pid = _safe_ui_text(row.get("pid"), limit=32)
                        lines.append(f"{process_name}（PID {pid}）" if pid else process_name)
                summary = "进程摘要：\n" + ("\n".join(lines) if lines else "没有可显示的进程。")
                self._diagnostic_output.setPlainText(summary)
                if observation_status is not None:
                    visible_lines = lines[:4]
                    observation_status.setText(
                        "进程摘要："
                        + (
                            " " + "、".join(visible_lines)
                            if visible_lines
                            else " 没有可显示的进程。"
                        )
                    )
                return
            if isinstance(result, Mapping) and label in {"模块状态", "module_status"}:
                self.set_module_status_snapshot(result)
                runtime = result.get("runtime")
                runtime = runtime if isinstance(runtime, Mapping) else {}
                memory = runtime.get("memory")
                memory = memory if isinstance(memory, Mapping) else {}
                renderer = runtime.get("renderer")
                renderer = renderer if isinstance(renderer, Mapping) else {}
                scheduler = runtime.get("scheduler")
                scheduler = scheduler if isinstance(scheduler, Mapping) else {}
                tts = result.get("tts")
                tts = tts if isinstance(tts, Mapping) else {}
                adapters = result.get("adapters")
                adapter_count = len(adapters) if isinstance(adapters, (list, tuple)) else 0
                index_label = {
                    "fts5": "FTS5 + 向量",
                    "sparse_fallback": "稀疏向量",
                }.get(str(memory.get("lexical_index", "")), "未就绪")
                vector_count = memory.get("vector_index_size", 0)
                task_count = scheduler.get("task_count", 0)
                trigger_count = scheduler.get("trigger_count", 0)
                vector_count = vector_count if isinstance(vector_count, int) else 0
                task_count = task_count if isinstance(task_count, int) else 0
                trigger_count = trigger_count if isinstance(trigger_count, int) else 0
                lines = [
                    f"模型适配器：{adapter_count} 个协议",
                    "TTS：" + ("已就绪" if bool(tts.get("available", False)) else "未就绪"),
                    "记忆索引：" + index_label + f"（{max(0, vector_count)} 条）",
                    "渲染："
                    + _safe_ui_text(renderer.get("backend", "未连接"), limit=64)
                    + (" · 可用" if bool(renderer.get("available", False)) else " · 不可用"),
                    f"调度：{max(0, task_count)} 个任务 / {max(0, trigger_count)} 个触发器",
                ]
                summary = "\n".join(lines)
                self._diagnostic_output.setPlainText(summary)
                if observation_status is not None:
                    observation_status.setText("模块检查已完成；详情见诊断面板。")
                return
            if isinstance(result, Mapping) and label in {"API调用审计", "api_audit"}:
                rows = result.get("records")
                rows = rows if isinstance(rows, (list, tuple)) else ()
                lines = [f"API 调用审计：{max(0, int(result.get('count', len(rows)) or 0))} 条"]
                for row in rows[:20]:
                    if not isinstance(row, Mapping):
                        continue
                    status = _safe_ui_text(row.get("status"), limit=24) or "unknown"
                    kind = _safe_ui_text(row.get("kind"), limit=24) or "model"
                    channel = (
                        _safe_ui_text(row.get("channel_id") or row.get("channel_name"), limit=64)
                        or "local"
                    )
                    provider = _safe_ui_text(row.get("provider"), limit=48)
                    protocol = _safe_ui_text(row.get("protocol"), limit=48)
                    requested = _safe_ui_text(row.get("requested_model"), limit=64)
                    served = _safe_ui_text(row.get("response_model"), limit=64)
                    ttft = row.get("time_to_first_token_ms")
                    total = row.get("total_duration_ms")
                    tool_count = row.get("tool_execution_count", 0)
                    model_text = requested or "未指定"
                    if served and served != requested:
                        model_text += f"→{served}"
                    timing = []
                    if isinstance(ttft, (int, float)) and not isinstance(ttft, bool):
                        first_char = _safe_ui_text(row.get("first_char"), limit=1)
                        first_text = f"首字 {float(ttft):.0f}ms"
                        if first_char:
                            first_text += f"（{first_char}）"
                        timing.append(first_text)
                    if isinstance(total, (int, float)) and not isinstance(total, bool):
                        timing.append(f"总计 {float(total):.0f}ms")
                    lines.append(
                        f"[{status}] {kind} / {channel} / {model_text}"
                        + (f" / 工具 {int(tool_count)}" if isinstance(tool_count, int) else "")
                        + (f" / {'，'.join(timing)}" if timing else "")
                    )
                    channel_parts = [part for part in (provider, protocol) if part]
                    if channel_parts:
                        lines.append(f"  渠道信息：{' / '.join(channel_parts)}")
                    channel_info = _audit_payload_text(row.get("channel_info"), limit=180)
                    if channel_info:
                        lines.append(f"  渠道配置：{channel_info}")
                    request_id = _safe_ui_text(row.get("request_id"), limit=48)
                    if request_id:
                        lines.append(f"  请求：{request_id}")
                    input_payload = row.get("input_payload")
                    if isinstance(input_payload, Mapping):
                        messages = input_payload.get("messages")
                        if isinstance(messages, (list, tuple)):
                            user_messages = []
                            for message in messages:
                                if (
                                    not isinstance(message, Mapping)
                                    or message.get("role") != "user"
                                ):
                                    continue
                                content = message.get("content")
                                if isinstance(content, str):
                                    user_messages.append(_safe_ui_text(content, limit=180))
                            if user_messages:
                                lines.append(f"  输入：{user_messages[-1]}")
                    output_text = _safe_ui_text(row.get("output_text"), limit=180)
                    if output_text:
                        lines.append(f"  输出：{output_text}")
                    usage = row.get("usage")
                    if isinstance(usage, Mapping):
                        usage_parts = []
                        for key, label in (
                            ("prompt_tokens", "输入令牌"),
                            ("completion_tokens", "输出令牌"),
                            ("total_tokens", "总令牌"),
                        ):
                            value = usage.get(key)
                            if isinstance(value, int) and not isinstance(value, bool):
                                usage_parts.append(f"{label} {value}")
                        if usage_parts:
                            lines.append("  用量：" + "，".join(usage_parts))
                    cache_parts = []
                    for key, label in (
                        ("cache_read_tokens", "读"),
                        ("cache_write_tokens", "写"),
                    ):
                        value = row.get(key)
                        if isinstance(value, int) and not isinstance(value, bool):
                            cache_parts.append(f"{label} {value}")
                    for key, label in (
                        ("cache_read_duration_ms", "读耗时"),
                        ("cache_write_duration_ms", "写耗时"),
                    ):
                        value = row.get(key)
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            cache_parts.append(f"{label} {float(value):.0f}ms")
                    cache_info = row.get("cache_info")
                    if isinstance(cache_info, Mapping):
                        if cache_info.get("read") is True and not any(
                            part.startswith("读 ") for part in cache_parts
                        ):
                            cache_parts.append("读命中")
                        if cache_info.get("write") is True and not any(
                            part.startswith("写 ") for part in cache_parts
                        ):
                            cache_parts.append("写入")
                    if cache_parts:
                        lines.append("  缓存：" + "，".join(cache_parts))
                    tool_calls = row.get("tool_calls")
                    if isinstance(tool_calls, (list, tuple)):
                        for call in tool_calls[:3]:
                            if not isinstance(call, Mapping):
                                continue
                            identity = _safe_ui_text(
                                call.get("identity") or call.get("name"), limit=64
                            )
                            arguments = _audit_payload_text(call.get("arguments"), limit=180)
                            if identity or arguments:
                                call_line = "  调用：" + (identity or "未命名工具")
                                if arguments:
                                    call_line += f" / 参数 {arguments}"
                                lines.append(call_line)
                    tool_executions = row.get("tool_executions")
                    if isinstance(tool_executions, (list, tuple)):
                        for execution in tool_executions[:3]:
                            if not isinstance(execution, Mapping):
                                continue
                            identity = _safe_ui_text(execution.get("identity"), limit=64)
                            execution_status = _safe_ui_text(execution.get("status"), limit=24)
                            duration = execution.get("duration_ms")
                            timing_text = (
                                f"，{float(duration):.0f}ms"
                                if isinstance(duration, (int, float))
                                and not isinstance(duration, bool)
                                else ""
                            )
                            if identity:
                                tool_status = execution_status or "unknown"
                                lines.append(f"  工具：{identity} / {tool_status}{timing_text}")
                            arguments = _audit_payload_text(execution.get("arguments"), limit=180)
                            if arguments:
                                lines.append(f"    参数：{arguments}")
                            result_text = _audit_payload_text(execution.get("result"), limit=180)
                            if result_text:
                                lines.append(f"    结果：{result_text}")
                            phase = _safe_ui_text(execution.get("phase"), limit=32)
                            attempt = execution.get("attempt")
                            retry_count = execution.get("retry_count")
                            execution_meta = []
                            if phase:
                                execution_meta.append(f"阶段 {phase}")
                            if isinstance(attempt, int) and not isinstance(attempt, bool):
                                execution_meta.append(f"第 {attempt} 次")
                            if (
                                isinstance(retry_count, int)
                                and not isinstance(retry_count, bool)
                                and retry_count
                            ):
                                execution_meta.append(f"重试 {retry_count}")
                            if execution_meta:
                                lines.append("    执行：" + "，".join(execution_meta))
                            plan_id = _safe_ui_text(execution.get("plan_id"), limit=48)
                            step_id = _safe_ui_text(execution.get("step_id"), limit=48)
                            parallel_group = execution.get("parallel_group")
                            relation = []
                            if plan_id:
                                relation.append(f"计划 {plan_id}")
                            if step_id:
                                relation.append(f"步骤 {step_id}")
                            if (
                                isinstance(parallel_group, int)
                                and not isinstance(parallel_group, bool)
                                and parallel_group
                            ):
                                relation.append(f"并行组 {parallel_group}")
                            if relation:
                                lines.append("    关系：" + "，".join(relation))
                            error = _safe_ui_text(execution.get("error"), limit=140)
                            if error:
                                lines.append(f"  工具错误：{error}")
                summary = "\n".join(lines)
                self._diagnostic_output.setPlainText(summary)
                if observation_status is not None:
                    observation_status.setText("API 审计已更新；详情见诊断面板。")
                return
            if isinstance(result, Mapping) and label in {"运行日志", "log_records"}:
                rows = result.get("records")
                rows = rows if isinstance(rows, (list, tuple)) else ()
                lines = [f"运行日志：{max(0, int(result.get('count', len(rows)) or 0))} 条"]
                for row in rows[:80]:
                    if not isinstance(row, Mapping):
                        continue
                    level = _safe_ui_text(row.get("level"), limit=12) or "INFO"
                    logger_name = _safe_ui_text(row.get("logger"), limit=48) or "app"
                    event = _safe_ui_text(row.get("event"), limit=72)
                    message = _safe_ui_text(row.get("message"), limit=260)
                    prefix = f"[{level}] {logger_name}"
                    if event:
                        prefix += f" / {event}"
                    lines.append(f"{prefix}\n  {message}")
                self._diagnostic_output.setPlainText("\n".join(lines))
                if observation_status is not None:
                    observation_status.setText("运行日志已更新；内容来自本地脱敏日志环。")
                return
            summary = _safe_result_summary(result)
            self._diagnostic_output.setPlainText(summary)
            if observation_status is not None:
                observation_status.setText(summary)

        def _set_click_through(self, enabled: bool) -> None:
            result = self._invoke("click_through", bool(enabled))
            status = _status_value(result).strip().lower()
            accepted = _operation_succeeded(result)
            detail = _detail_value(result)
            detail_lower = detail.lower()
            # 只有明确说明 Wayland/compositor 行为未决时，``degraded`` 才代表
            # Qt flag 已保留但最终输入路由未知。未保留 flag 的结果必须按失败
            # 处理，避免 X11 或 Web 宿主误显示成 Wayland 成功。
            if status == "degraded":
                reported_enabled = result.get("enabled") if isinstance(result, Mapping) else None
                flag_failed = any(
                    marker in detail_lower
                    for marker in ("did not retain", "update failed", "unavailable")
                )
                wayland_degraded = "wayland" in detail_lower and "compositor" in detail_lower
                enabled_matches = not isinstance(reported_enabled, bool) or (
                    reported_enabled == bool(enabled)
                )
                accepted = not flag_failed and wayland_degraded and enabled_matches
                if accepted:
                    self.set_status(
                        "点击穿透已请求（Wayland 行为由 compositor 决定）"
                        if enabled
                        else "点击穿透已关闭（Wayland 行为由 compositor 决定）"
                    )
            elif accepted:
                self.set_status(f"点击穿透：{'已开启' if enabled else '已关闭'}")
            if not accepted:
                reported_enabled = result.get("enabled") if isinstance(result, Mapping) else None
                # 失败回执若只是回显请求值，不能把它当成原生窗口已确认；
                # 只有回执明确报告了与请求相反的值，才说明窗口仍停留在该实际状态。
                actual_enabled = (
                    bool(reported_enabled)
                    if isinstance(reported_enabled, bool) and reported_enabled != bool(enabled)
                    else not bool(enabled)
                )
                self._click_through.blockSignals(True)
                self._click_through.setChecked(actual_enabled)
                self._click_through.blockSignals(False)
                self.set_status("点击穿透不可用")
                self.append_line(f"系统：点击穿透未执行：{_safe_result_summary(result)}")

        def _set_window_locked(self, enabled: bool) -> None:
            """切换桌宠锁定；锁定不暂停自主行为和光标追踪。"""

            self.windowLockRequested.emit(bool(enabled))
            self.operationTriggered.emit("set_window_locked")
            result = self._invoke("set_window_locked", bool(enabled))
            accepted = bool(
                isinstance(result, Mapping)
                and _operation_succeeded(result)
                and result.get("locked") == bool(enabled)
            )
            if accepted:
                self._click_through.setEnabled(not bool(enabled))
                self._window_lock_quick_button.setText("解除锁定" if enabled else "锁定窗口")
                self.set_status("窗口已锁定" if enabled else "窗口已解锁")
                return
            reported_locked = result.get("locked") if isinstance(result, Mapping) else None
            actual_locked = (
                bool(reported_locked)
                if isinstance(reported_locked, bool) and reported_locked != bool(enabled)
                else not bool(enabled)
            )
            self._window_lock.blockSignals(True)
            self._window_lock.setChecked(actual_locked)
            self._window_lock.blockSignals(False)
            self._click_through.setEnabled(not actual_locked)
            self._window_lock_quick_button.setText("解除锁定" if actual_locked else "锁定窗口")
            self.set_status("窗口锁定未完成")
            self.append_line(f"系统：窗口锁定未执行：{_safe_result_summary(result)}")

        def _toggle_window_lock_quick(self) -> None:
            """切换首屏快捷锁定按钮。"""

            self._window_lock.setChecked(not self._window_lock.isChecked())

        def set_window_locked_checked(self, enabled: bool) -> None:
            """同步外部窗口锁定状态，不触发回调。"""

            value = bool(enabled)
            self._window_lock.blockSignals(True)
            self._window_lock.setChecked(value)
            self._window_lock.blockSignals(False)
            self._click_through.setEnabled(not value)
            self._window_lock_quick_button.setText("解除锁定" if value else "锁定窗口")

        def set_always_on_top_status(self, value: Mapping[str, object] | object) -> None:
            """同步置顶请求的实际/异步终态并锁住重复点击。"""

            if isinstance(value, Mapping):
                status = str(value.get("status", value.get("state", "")) or "").strip().lower()
                enabled = bool(value.get("enabled", False))
                detail = _safe_ui_text(value.get("detail", value.get("reason", "")), limit=180)
            else:
                status = _status_value(value).strip().lower()
                enabled = False
                detail = _safe_ui_text(_detail_value(value), limit=180)
            if status not in {"available", "degraded", "unavailable", "requested", "cancelled"}:
                status = ""
            signature = (status, enabled, detail)
            pending = status == "requested"
            for button in self._topmost_control_buttons:
                try:
                    button.setEnabled(not pending)
                    button.setText(
                        ("取消置顶" if enabled else "窗口置顶")
                        if button.property("rescueAction") == "toggle_always_on_top"
                        else ("取消置顶" if enabled else "切换窗口置顶")
                    )
                    if detail:
                        button.setToolTip(detail)
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    continue
            if signature == self._topmost_status_signature:
                return
            self._topmost_status_signature = signature
            if status == "requested":
                self.set_status("窗口置顶切换已提交（正在应用）")
            elif status == "degraded":
                self.set_status("窗口置顶已请求（最终状态由桌面合成器决定）")
            elif status in {"unavailable", "cancelled"}:
                self.set_status("窗口置顶未生效，请稍后重试")

        def set_status(self, text: str) -> None:
            """更新状态栏并保留最近状态到输出区。"""

            value = _friendly_model_message(str(text or "").strip())
            if not value:
                return
            self._status.setText(value)
            if any(
                marker in value for marker in ("失败", "不可用", "未完成", "错误", "未提交", "无法")
            ):
                state = "error"
            elif any(marker in value for marker in ("正在", "请求", "提交", "等待")):
                state = "busy"
            elif any(
                marker in value
                for marker in ("未配置", "未就绪", "缺少", "请先", "需要确认", "已拒绝")
            ):
                state = "warning"
            else:
                state = "ready"
            self._status.setProperty("state", state)
            self._status.style().unpolish(self._status)
            self._status.style().polish(self._status)
            info_bar = getattr(self, "_status_info_bar", None)
            if isinstance(info_bar, FancyInfoBar):
                info_bar.setMessage(value)
                info_bar.setSeverity(
                    {
                        "error": "error",
                        "busy": "info",
                        "warning": "warning",
                        "ready": "success",
                    }[state]
                )
            self.statusChanged.emit(value)

        def set_pet_feedback(self, value: object) -> None:
            """显示猫猫头/身体点击反馈，允许宿主回写好感度结果。"""

            if isinstance(value, Mapping):
                zone_labels = {
                    "head": "猫猫头",
                    "upper": "上半身",
                    "lower_left": "左腿",
                    "lower_right": "右腿",
                    "body": "身体",
                }
                zone = str(value.get("zone", "") or "").strip().lower()
                phrase = _safe_ui_text(value.get("phrase", ""), limit=120)
                mood = _public_action_label("expression", value.get("mood", ""))
                affection = value.get("affection")
                affection_text = ""
                if isinstance(affection, Mapping):
                    current = _safe_ui_text(affection.get("current", ""), limit=32)
                    applied = _safe_ui_text(affection.get("applied", ""), limit=32)
                    if current:
                        affection_text = f"好感度 {current}"
                    if applied:
                        affection_text += f"（本次 {applied}）"
                pieces = [zone_labels.get(zone, "桌宠互动")]
                if phrase:
                    pieces.append(phrase)
                if mood:
                    pieces.append(f"情绪：{mood}")
                if affection_text:
                    pieces.append(affection_text)
                text = " · ".join(pieces)
            else:
                text = _safe_ui_text(value, limit=240)
            if not text:
                self.clear_pet_feedback()
                return
            self._pet_feedback.setText(f"互动反馈：{text}")
            self._pet_feedback.setProperty("state", "active")
            if bool(self.property("compactLayout")):
                self._pet_feedback.setVisible(False)
            self._pet_feedback.style().unpolish(self._pet_feedback)
            self._pet_feedback.style().polish(self._pet_feedback)

        def clear_pet_feedback(self) -> None:
            """清除首屏互动反馈并恢复空状态文案。"""

            if not hasattr(self, "_pet_feedback"):
                return
            self._pet_feedback.setText("互动反馈：点击猫猫头或身体后会显示部位和好感度变化")
            self._pet_feedback.setProperty("state", "idle")
            if bool(self.property("compactLayout")):
                self._pet_feedback.setVisible(False)
            self._pet_feedback.style().unpolish(self._pet_feedback)
            self._pet_feedback.style().polish(self._pet_feedback)

        def append_line(self, text: str) -> None:
            """向控制台追加一行文本。"""

            value = _friendly_model_message(str(text or "").strip())
            if value:
                # 系统回执插入到流式输出之后；下一次组件增量必须从新行开始，
                # 不能把文字接到回执末尾。组件旧值仍保留，用于计算真正新增的
                # 后缀，避免再次复制整段正文。
                self._snapshot_tail_component = None
                self._output.appendPlainText(value)
                self._sync_empty_state()

        @staticmethod
        def _snapshot_components(snapshot: object) -> tuple[str, str, str, str] | None:
            """读取展示快照的独立文本组件；旧测试替身返回 ``None``。"""

            component_names = ("direct_text", "text", "murmur", "tool_status")
            if not any(hasattr(snapshot, name) for name in component_names):
                return None
            direct = _friendly_stream_text(getattr(snapshot, "direct_text", ""))
            if direct:
                # 直接说话层优先于当前回合的正文/碎碎念；底层组件仍由服务
                # 保留，清除直接层后会在下一次快照中继续从其后缀展示。
                return direct, "", "", ""
            return (
                "",
                _friendly_stream_text(getattr(snapshot, "text", "")),
                _friendly_stream_text(getattr(snapshot, "murmur", "")),
                _friendly_stream_text(getattr(snapshot, "tool_status", "")),
            )

        def _append_snapshot_piece(self, value: str, *, same_component: bool) -> None:
            """把一个组件的新增后缀写到输出末尾。"""

            if not value:
                return
            cursor = self._output.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            current = self._output.toPlainText()
            if not same_component and current and not current.endswith("\n"):
                cursor.insertText("\n")
            cursor.insertText(value)
            self._output.setTextCursor(cursor)

        def set_snapshot(self, snapshot: object) -> None:
            """消费展示快照，避免重复写入相同的流式内容。"""

            raw_text = getattr(snapshot, "rendered_text", "")
            text = _friendly_stream_text(raw_text)
            mood = str(getattr(snapshot, "rendered_mood", "neutral") or "neutral")
            safe_mood = _public_action_label("expression", mood) or "自然"
            components = self._snapshot_components(snapshot)
            if components is not None:
                # BubbleSnapshot/InteractionSnapshot 都提供独立组件；按组件
                # 追加后缀，避免最终正文到达时再次复制先前的 murmur/tool 状态。
                self._last_snapshot_text = text
                if not text:
                    self._last_snapshot_components = ("", "", "", "")
                    self._snapshot_tail_component = None
                    self._last_snapshot_mood = mood
                    return
                previous_components = self._last_snapshot_components
                if previous_components is None or not any(previous_components):
                    self.set_status(f"模型输出（{safe_mood}）")
                    self._append_snapshot_piece(
                        f"桌宠（{safe_mood}）：{text}",
                        same_component=False,
                    )
                    nonempty_components = [index for index, value in enumerate(components) if value]
                    self._snapshot_tail_component = (
                        max(nonempty_components) if nonempty_components else None
                    )
                else:
                    for index, (previous, current) in enumerate(
                        zip(previous_components, components, strict=True)
                    ):
                        if current == previous:
                            continue
                        # 正文和碎碎念是追加流；工具状态/直接说话是“当前值”
                        # 替换流。替换流即使仍是同一组件也必须换行，否则
                        # ``等待确认``、``已完成`` 会粘在上一条状态后面，
                        # 用户看到的动态输出会变成一串不可读的协议文本。
                        append_only = index in {1, 2}
                        if append_only and current.startswith(previous):
                            addition = current[len(previous) :]
                        else:
                            addition = current
                        if not addition:
                            self._snapshot_tail_component = None
                            continue
                        same_component = append_only and self._snapshot_tail_component == index
                        self._append_snapshot_piece(addition, same_component=same_component)
                        self._snapshot_tail_component = index
                    self.set_status(f"模型输出（{safe_mood}）")
                self._last_snapshot_components = components
                self._last_snapshot_mood = mood
                self._sync_empty_state()
                return
            # 兼容只暴露 rendered_text 的旧宿主/测试替身；真实运行时路径走
            # 上面的组件增量，不能把两种状态混用。
            self._last_snapshot_components = None
            self._snapshot_tail_component = None
            if text and (text != self._last_snapshot_text or mood != self._last_snapshot_mood):
                previous = _friendly_stream_text(self._last_snapshot_text)
                previous_mood = self._last_snapshot_mood
                self._last_snapshot_text = text
                self._last_snapshot_mood = mood
                self.set_status(f"模型输出（{safe_mood}）")
                cursor = self._output.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                if previous and mood == previous_mood and text.startswith(previous):
                    cursor.insertText(text[len(previous) :])
                else:
                    if self._output.toPlainText().strip():
                        cursor.insertText("\n")
                    cursor.insertText(f"桌宠（{safe_mood}）：{text}")
                self._output.setTextCursor(cursor)
                self._sync_empty_state()
            elif not text:
                self._last_snapshot_text = ""
                self._last_snapshot_mood = mood

        def show_and_focus(self, page: str = "对话") -> None:
            """显示控制台并切换到指定公开导航页。"""

            self.showNormal()
            self.raise_()
            self.activateWindow()
            self._navigate_console_page(page)
            if page == "对话":
                self._input.setFocus(Qt.OtherFocusReason)

        def _on_console_tab_changed(self, index: int) -> None:
            """配置页在极小屏获得完整高度，操作页恢复首屏标题和模型入口。"""

            tabs = getattr(self, "_console_tabs", None)
            is_configuration = bool(
                tabs is not None
                and 0 <= int(index) < tabs.count()
                and tabs.tabText(int(index)) == "配置中心"
            )
            if is_configuration:
                self._ensure_configuration_panel()
            is_module_center = bool(
                tabs is not None
                and 0 <= int(index) < tabs.count()
                and tabs.tabText(int(index)) == "模块中心"
            )
            if is_module_center:
                self._ensure_module_center()
            toggle = getattr(self, "_secondary_actions_toggle", None)
            if toggle is not None:
                toggle.setText(
                    "收起更多操作"
                    if tabs is not None
                    and tabs.currentWidget() is getattr(self, "_secondary_actions_panel", None)
                    else "更多操作"
                )
            self._set_configuration_focus(is_configuration)

        def _set_configuration_focus(self, enabled: bool) -> None:
            """配置页隐藏重复控制台说明，为设置卡片保留可用高度。"""

            self._configuration_focus = bool(enabled)
            compact = bool(self.property("compactLayout"))
            hide_header_details = self._configuration_focus
            hidden_when_compact = (
                getattr(self, "_header_eyebrow", None),
                getattr(self, "_header_subtitle", None),
                getattr(self, "_state_actions_host", None),
                getattr(self, "_status_info_bar", None),
            )
            for widget in hidden_when_compact:
                if widget is not None:
                    widget.setVisible(not hide_header_details)
            title_label = self.findChild(QLabel, "consoleTitleLabel")
            if title_label is not None:
                title_label.setText("配置中心" if self._configuration_focus else "桌宠控制台")
                title_label.setVisible(not (compact and self._configuration_focus))
            config_button = self.findChild(QPushButton, "configCenterButton")
            if config_button is not None:
                config_button.setVisible(not compact and not self._configuration_focus)
            rescue_title = getattr(self, "_rescue_title", None)
            if rescue_title is not None:
                rescue_title.setVisible(not compact and not hide_header_details)
            rescue_card = self.findChild(QFrame, "consoleRescueCard")
            if rescue_card is not None:
                rescue_card.setVisible(not hide_header_details)
            rescue_layout = rescue_card.layout() if rescue_card is not None else None
            if rescue_layout is not None:
                compact_rescue = compact or hide_header_details
                rescue_layout.setContentsMargins(
                    8 if compact_rescue else 16,
                    6 if compact_rescue else 14,
                    8 if compact_rescue else 16,
                    6 if compact_rescue else 14,
                )
                rescue_layout.setSpacing(4 if compact_rescue else 10)
            # compactLayout 自身仍负责隐藏冗长说明；配置页切回操作页
            # 时重新应用该状态，避免标题/按钮一直消失。
            if compact and not hide_header_details:
                for widget in (
                    getattr(self, "_header_eyebrow", None),
                    getattr(self, "_header_subtitle", None),
                    getattr(self, "_rescue_status", None),
                    getattr(self, "_status_info_bar", None),
                ):
                    if widget is not None:
                        widget.setVisible(False)
            self.updateGeometry()

        def set_compact_layout(self, enabled: bool) -> None:
            """在极小屏控制台模式中压缩非关键说明，保留点击入口。"""

            compact = bool(enabled)
            for widget in (
                getattr(self, "_header_eyebrow", None),
                getattr(self, "_header_subtitle", None),
                getattr(self, "_rescue_status", None),
                getattr(self, "_status_info_bar", None),
            ):
                if widget is not None:
                    widget.setVisible(not compact)
            feedback = getattr(self, "_pet_feedback", None)
            if feedback is not None:
                # 极小屏已有顶部状态和对话记录反馈，隐藏重复的长反馈条，
                # 避免文字与救援按钮在固定高度内互相裁切。
                feedback.setVisible(not compact)
            # 极小屏首屏必须先露出输入框；救援卡只保留恢复按钮，隐藏
            # 标题/说明并压缩高度，避免操作中心被推到可视区域之外。
            rescue_card = self.findChild(QFrame, "consoleRescueCard")
            rescue_layout = rescue_card.layout() if rescue_card is not None else None
            if rescue_layout is not None:
                rescue_layout.setContentsMargins(
                    8 if compact else 16,
                    5 if compact else 14,
                    8 if compact else 16,
                    5 if compact else 14,
                )
                rescue_layout.setSpacing(4 if compact else 10)
            rescue_title = getattr(self, "_rescue_title", None)
            rescue_status = getattr(self, "_rescue_status", None)
            if rescue_title is not None:
                rescue_title.setVisible(not compact)
            if rescue_status is not None:
                rescue_status.setVisible(not compact)
            if rescue_card is not None:
                # 窄屏时恢复按钮会从五列换成两列或更多行。此前把卡片
                # 固定在 58px 会裁掉第二、三行，恰好可能隐藏“恢复鼠标”
                # 入口；主操作页已经是滚动容器，应保留卡片完整高度。
                rescue_card.setMaximumHeight(16777215)
            for rescue_button in getattr(self, "_rescue_buttons", {}).values():
                rescue_button.setMinimumHeight(28 if compact else 34)
                rescue_button.setMaximumHeight(32 if compact else 16777215)
            if compact:
                self.setMinimumHeight(1)
            output = getattr(self, "_output", None)
            if output is not None:
                output.setMinimumHeight(48 if compact else 120)
                output.setMaximumHeight(48 if compact else 220)
            microphone_status = getattr(self, "_microphone_status", None)
            if microphone_status is not None:
                microphone_status.setVisible(
                    not compact
                    or getattr(self, "_microphone_state", "disabled")
                    in {
                        "waiting_playback",
                        "permission_pending",
                        "recording",
                        "captured",
                        "transcribing",
                        "denied",
                        "unavailable",
                    }
                )
            self._sync_empty_state()
            panel = getattr(self, "_configuration_panel", None)
            if panel is not None:
                try:
                    panel.set_compact_layout(compact)
                except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                    pass
            for button in (
                getattr(self, "_send_button", None),
                getattr(self, "_stop_button", None),
                getattr(self, "_microphone_button", None),
            ):
                if button is not None:
                    button.setMinimumHeight(34 if compact else 0)
                    button.setMaximumHeight(38 if compact else 16777215)
            root_layout = self.layout()
            if root_layout is not None:
                margin = 6 if compact else 12
                root_layout.setContentsMargins(margin, margin, margin, margin)
                root_layout.setSpacing(6 if compact else 10)
            self.setProperty("compactLayout", compact)
            self._sync_empty_state()
            self._set_configuration_focus(self._configuration_focus)
            self.style().unpolish(self)
            self.style().polish(self)
            self.updateGeometry()

        def set_click_through_checked(self, enabled: bool) -> None:
            """同步外部点击穿透状态，不触发回调。"""

            self._click_through.blockSignals(True)
            self._click_through.setChecked(bool(enabled))
            self._click_through.blockSignals(False)

        def closeEvent(self, event) -> None:  # noqa: N802
            if self._allow_close:
                event.accept()
                return
            event.accept()
            self.closeRequested.emit()
            was_visible = self.isVisible()
            self.hide()
            if was_visible:
                self.hidden.emit()

        def shutdown(self) -> None:
            """允许宿主在退出阶段真正销毁窗口。"""

            self._allow_close = True
            self.set_microphone_state("closed")
            self._microphone_device_loader = None
            self._microphone_device_selector.setEnabled(False)
            self._microphone_device_refresh_button.setEnabled(False)
            if self._model_setup_dialog is not None:
                try:
                    self._model_setup_dialog.close()
                except (AttributeError, RuntimeError):
                    pass
                self._model_setup_dialog = None
            self.close()

else:

    class PetConsoleWindow:  # pragma: no cover
        """无 Qt 环境下的明确不可用占位。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 is not installed")


__all__ = ["PetConsoleWindow", "pyside6_available"]
