from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gui.qt6.fancyui import (
    FancyCard,
    FancyInfoBar,
    FancyStyleController,
)
from gui.qt6.md3 import (
    DARK_MD3_THEME,
    MD3Theme,
    build_md3_stylesheet,
    build_qt_palette,
    remap_legacy_stylesheet,
)
from services.scheduler.scheduler import (
    ScheduleExpressionError,
    validate_schedule_expression,
)

try:  # Qt 依赖保持可选，核心服务不需要 GUI。
    from PySide6.QtCore import QSize, Qt, QTime, Signal
    from PySide6.QtWidgets import (
        QBoxLayout,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLayout,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QTabWidget,
        QTimeEdit,
        QVBoxLayout,
        QWidget,
    )

    pyside6_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_available = False


EVENT_NAMES: tuple[tuple[str, str], ...] = (
    ("startup", "启动时"),
    ("window_changed", "前台窗口变化"),
    ("window_active", "前台窗口持续"),
    ("user_interaction", "用户互动"),
    ("user_active", "用户恢复活跃"),
    ("idle", "用户空闲"),
    ("conversation_finished", "对话完成"),
    ("conversation_failed", "对话失败"),
)

INTERVAL_UNITS: tuple[tuple[str, str], ...] = (
    ("ms", "毫秒"),
    ("s", "秒"),
    ("m", "分钟"),
    ("h", "小时"),
)

TASK_INTERVAL_PRESETS: tuple[tuple[str, str, float, str], ...] = (
    ("1m", "每分钟", 1.0, "m"),
    ("5m", "每 5 分钟", 5.0, "m"),
    ("1h", "每小时", 1.0, "h"),
    ("1d", "每天", 24.0, "h"),
)

SCHEDULE_MODES: tuple[tuple[str, str], ...] = (
    ("interval", "按间隔"),
    ("daily", "每天定时"),
)

DAILY_TIME_PRESETS: tuple[tuple[str, str], ...] = (
    ("03:00", "凌晨 3 点"),
    ("09:00", "上午 9 点"),
    ("18:00", "下午 6 点"),
)

DEBOUNCE_PRESETS: tuple[tuple[str, str, float], ...] = (
    ("none", "不冷却", 0.0),
    ("5s", "5 秒", 5.0),
    ("15s", "15 秒", 15.0),
    ("1m", "1 分钟", 60.0),
)


@dataclass(frozen=True)
class ActionPreset:
    """一个不暴露工具身份和参数结构的安全动作预设。"""

    key: str
    label: str
    identity: str
    argument_name: str = "name"
    requires_text: bool = False


ACTION_PRESETS: tuple[ActionPreset, ...] = (
    ActionPreset("blink", "眨眼", "pet:play_motion"),
    ActionPreset("wave", "挥手", "pet:play_motion"),
    ActionPreset("walk", "走路动作（不移动窗口）", "pet:play_motion"),
    ActionPreset("neutral", "自然表情", "pet:set_expression"),
    ActionPreset("happy", "开心表情", "pet:set_expression"),
    ActionPreset("curious", "好奇表情", "pet:set_expression"),
    ActionPreset("sad", "难过表情", "pet:set_expression"),
    ActionPreset("surprised", "惊讶表情", "pet:set_expression"),
    ActionPreset("shy", "害羞表情", "pet:set_expression"),
    ActionPreset("speak", "说话", "pet:speak", argument_name="text", requires_text=True),
    ActionPreset(
        "proactive",
        "模型主动响应",
        "proactive:run",
        argument_name="instruction",
        requires_text=True,
    ),
)

_PRESETS_BY_KEY = {item.key: item for item in ACTION_PRESETS}
_EVENT_LABELS = dict(EVENT_NAMES)
_UNIT_LABELS = dict(INTERVAL_UNITS)
_KNOWN_TASK_KEYS = frozenset(
    {"task_id", "name", "expression", "action", "owner", "enabled", "metadata"}
)
_KNOWN_TRIGGER_KEYS = frozenset(
    {
        "trigger_id",
        "event_name",
        "action",
        "owner",
        "debounce_seconds",
        "metadata",
        "conditions",
    }
)


def _safe_label(value: object, *, fallback: str, limit: int = 80) -> str:
    """给列表生成短文案，避免把未知参数或换行带到 Qt。"""

    text = str(value or "").strip().replace("\r", " ").replace("\n", " ")
    if not text:
        return fallback
    return text[: limit - 1] + "…" if len(text) > limit else text


def _action_preset(action: object) -> str | None:
    """只识别本面板能安全重建的动作映射。"""

    if not isinstance(action, Mapping):
        return None
    identity = str(action.get("identity", "") or "").strip()
    arguments = action.get("arguments", action.get("args"))
    if not isinstance(arguments, Mapping):
        return None
    for preset in ACTION_PRESETS:
        if identity != preset.identity:
            continue
        if preset.requires_text:
            if (
                not isinstance(arguments.get(preset.argument_name), str)
                or not str(arguments.get(preset.argument_name, "")).strip()
            ):
                continue
            # 说话动作携带 mood/language 时，普通编辑器无法无损回写；
            # 将其归入高级只读条目，避免用户保存后丢失可选字段。
            if set(arguments) != {preset.argument_name}:
                continue
            return preset.key
        if set(arguments) != {preset.argument_name}:
            continue
        value = arguments.get(preset.argument_name)
        if isinstance(value, str) and value.strip() == preset.key:
            return preset.key
    return None


def _action_from_preset(key: str, text: str = "") -> dict[str, Any] | None:
    """将安全预设转换成后端现有的 action 映射。"""

    preset = _PRESETS_BY_KEY.get(str(key or "").strip())
    if preset is None:
        return None
    if preset.requires_text:
        value = str(text or "").strip()
        if not value:
            return None
        return {"identity": preset.identity, "arguments": {preset.argument_name: value}}
    return {
        "identity": preset.identity,
        "arguments": {preset.argument_name: preset.key},
    }


def _is_known_task(value: object) -> bool:
    if not isinstance(value, Mapping) or not _KNOWN_TASK_KEYS.issuperset(value.keys()):
        return False
    name = str(value.get("name", "") or "").strip()
    expression = str(value.get("expression", "") or "").strip()
    try:
        validate_schedule_expression(expression)
    except (ScheduleExpressionError, TypeError, ValueError):
        return False
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    if "require_user_active" in metadata and not isinstance(
        metadata.get("require_user_active"), bool
    ):
        return False
    return bool(name and _action_preset(value.get("action")) is not None)


def _is_known_trigger(value: object) -> bool:
    if not isinstance(value, Mapping) or not _KNOWN_TRIGGER_KEYS.issuperset(value.keys()):
        return False
    event_name = str(value.get("event_name", "") or "").strip().lower()
    if event_name not in _EVENT_LABELS or _action_preset(value.get("action")) is None:
        return False
    try:
        debounce = float(value.get("debounce_seconds", 0.0))
    except (TypeError, ValueError):
        return False
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    return 0 <= debounce <= 86400


def _format_amount(value: float) -> str:
    return f"{float(value):g}"


if pyside6_available:
    _STYLE = """
    QWidget#schedulerPanel { background: #16111f; color: #faf6fb; }
    QGroupBox#schedulerCard, QGroupBox#schedulerEditor,
    QGroupBox#schedulerAdvancedCard, QGroupBox#schedulerActivityCard {
        background: #2e2440; border: 1px solid #46385c;
        border-top-color: #7c69a0; border-radius: 12px;
        margin-top: 12px; padding: 12px; color: #faf6fb;
    }
    QGroupBox#schedulerCard::title, QGroupBox#schedulerEditor::title,
    QGroupBox#schedulerAdvancedCard::title, QGroupBox#schedulerActivityCard::title {
        subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #ffc48f;
    }
    QWidget#schedulerPanel QLineEdit, QWidget#schedulerPanel QComboBox,
    QWidget#schedulerPanel QDoubleSpinBox, QWidget#schedulerPanel QSpinBox {
        background: #100c18; color: #faf6fb; border: 1px solid #7c69a0;
        border-radius: 8px; padding: 7px 8px; min-height: 32px;
    }
    QWidget#schedulerPanel QLabel, QWidget#schedulerPanel QCheckBox {
        color: #faf6fb;
    }
    QWidget#schedulerPanel QLineEdit:focus, QWidget#schedulerPanel QComboBox:focus,
    QWidget#schedulerPanel QDoubleSpinBox:focus, QWidget#schedulerPanel QSpinBox:focus {
        border-color: #ff9dbe;
    }
    QWidget#schedulerPanel QListWidget {
        background: #100c18; color: #faf6fb; border: 1px solid #46385c;
        border-radius: 8px; padding: 4px; min-height: 64px;
    }
    QWidget#schedulerPanel QListWidget::item { padding: 7px; border-radius: 6px; }
    QWidget#schedulerPanel QListWidget::item:selected { background: #674563; }
    /* _button() marks semantic roles as dynamic properties, so style the
       actual property instead of relying on an object name that is not set. */
    QPushButton[role="schedulerPrimary"] {
        background: #ff9dbe; color: #2b0f1c; border: none;
        border-radius: 8px; padding: 7px 12px; font-weight: 700;
    }
    QPushButton[role="schedulerSecondary"],
    QPushButton#schedulerTaskCustomToggle,
    QPushButton#schedulerTriggerCustomToggle {
        background: #221a2e; color: #faf6fb; border: 1px solid #7c69a0;
        border-radius: 8px; padding: 7px 12px;
    }
    QPushButton[role="schedulerDanger"] {
        background: transparent; color: #ff8fa0; border: 1px solid #ff8fa0;
        border-radius: 8px; padding: 7px 12px;
    }
    QLabel#schedulerHint, QLabel#schedulerStatus, QLabel#schedulerAdvancedHint {
        color: #d6cbe0;
    }
    QLabel#schedulerStatus[state="error"] { color: #ff8fa0; }
    QLabel#schedulerStatus[state="ok"] { color: #6fe0b4; }
    """

    class SchedulerPanel(QWidget):
        """调度器普通用户配置页。"""

        changed = Signal(object)
        statusChanged = Signal(str)

        def __init__(
            self,
            values: Mapping[str, Any] | None = None,
            *,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent)
            self.setObjectName("schedulerPanel")
            inherited_theme = getattr(parent, "_md3_theme", DARK_MD3_THEME)
            resolved_theme = (
                inherited_theme if isinstance(inherited_theme, MD3Theme) else DARK_MD3_THEME
            )
            self._fancy_style_controller = FancyStyleController(resolved_theme, self)
            self.apply_theme(resolved_theme)
            self._section: dict[str, Any] = {}
            self._task_entries: list[dict[str, Any]] = []
            self._trigger_entries: list[dict[str, Any]] = []
            self._task_edit_index: int | None = None
            self._trigger_edit_index: int | None = None
            self._building = False
            self._activity_declared = False
            self._advanced_card: QWidget | None = None
            self._task_master_detail_layout: QBoxLayout | None = None
            self._trigger_master_detail_layout: QBoxLayout | None = None
            self._task_master: QWidget | None = None
            self._trigger_master: QWidget | None = None
            self._responsive_narrow = False
            self._build_ui()
            self.set_values(values or {})

        def apply_theme(self, theme: MD3Theme) -> None:
            """从配置中心继承主题，并原位刷新所有调度控件。"""

            if not isinstance(theme, MD3Theme):
                raise TypeError("theme must be an MD3Theme")
            self._md3_theme = theme
            self.setProperty("md3Role", "root")
            controller = self._fancy_style_controller
            controller.updateTheme(theme)
            controller.attach(self)
            self.setPalette(build_qt_palette(theme))
            colors = theme.colors
            layout = theme.layout
            modern = f"""
QWidget#schedulerPanel {{ background: {colors.surface}; color: {colors.on_surface}; }}
QFrame#schedulerCard, QFrame#schedulerEditor,
QFrame#schedulerAdvancedCard, QFrame#schedulerActivityCard {{
    background: {colors.surface_container};
    color: {colors.on_surface};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_large_px}px;
}}
QFrame#schedulerHeader, QFrame#schedulerActivityCard {{
    border-top: 2px solid {colors.primary};
}}
QWidget#schedulerPanel QLineEdit, QWidget#schedulerPanel QComboBox,
QWidget#schedulerPanel QDoubleSpinBox, QWidget#schedulerPanel QSpinBox,
QWidget#schedulerPanel QTimeEdit {{
    background: {colors.surface_container_low};
    color: {colors.on_surface};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_medium_px}px;
}}
QWidget#schedulerPanel QLineEdit:hover, QWidget#schedulerPanel QComboBox:hover,
QWidget#schedulerPanel QDoubleSpinBox:hover, QWidget#schedulerPanel QSpinBox:hover,
QWidget#schedulerPanel QTimeEdit:hover {{
    border-color: {colors.on_surface_variant};
}}
QWidget#schedulerPanel QLineEdit:focus, QWidget#schedulerPanel QComboBox:focus,
QWidget#schedulerPanel QDoubleSpinBox:focus, QWidget#schedulerPanel QSpinBox:focus,
QWidget#schedulerPanel QTimeEdit:focus {{
    border: 1px solid {colors.primary};
}}
QWidget#schedulerPanel QListWidget {{
    background: {colors.surface_container_low};
    color: {colors.on_surface};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_medium_px}px;
}}
QWidget#schedulerPanel QListWidget::item:hover {{
    background: {colors.surface_container_high};
}}
QWidget#schedulerPanel QListWidget::item:selected {{
    background: {colors.primary_container};
    color: {colors.on_primary_container};
}}
QPushButton[role="schedulerPrimary"] {{
    background: {colors.primary};
    color: {colors.on_primary};
    border: 1px solid {colors.primary};
    border-radius: {layout.radius_medium_px}px;
}}
QPushButton[role="schedulerPrimary"]:hover {{
    background: {colors.primary_container};
}}
QPushButton[role="schedulerSecondary"],
QPushButton#schedulerTaskCustomToggle,
QPushButton#schedulerTriggerCustomToggle {{
    background: {colors.surface_container};
    color: {colors.on_surface};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_medium_px}px;
}}
QPushButton[role="schedulerSecondary"]:hover,
QPushButton#schedulerTaskCustomToggle:hover,
QPushButton#schedulerTriggerCustomToggle:hover {{
    background: {colors.surface_container_high};
    border-color: {colors.primary};
}}
QPushButton[role="schedulerDanger"] {{
    background: transparent;
    color: {colors.error};
    border: 1px solid {colors.error};
    border-radius: {layout.radius_medium_px}px;
}}
QTabWidget#schedulerModeTabs {{
    background: {colors.surface};
}}
QTabWidget#schedulerModeTabs QTabBar {{
    background: {colors.surface};
}}
QTabWidget#schedulerModeTabs::pane {{
    background: {colors.surface_container_low};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_large_px}px;
}}
QTabWidget#schedulerModeTabs QTabBar::tab {{
    background: transparent;
    color: {colors.on_surface_variant};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 14px;
}}
QTabWidget#schedulerModeTabs QTabBar::tab:hover {{
    background: {colors.surface_container};
    color: {colors.on_surface};
}}
QTabWidget#schedulerModeTabs QTabBar::tab:selected {{
    color: {colors.primary};
    border-bottom-color: {colors.primary};
}}
QScrollArea#schedulerModeScroll {{
    background: {colors.surface};
    border: none;
}}
QScrollArea#schedulerModeScroll QAbstractScrollArea::viewport {{
    background: {colors.surface};
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
QScrollBar::handle:hover {{ background: {colors.primary}; }}
"""
            self.setStyleSheet(
                "\n\n".join(
                    (
                        controller.stylesheet(),
                        build_md3_stylesheet(theme),
                        remap_legacy_stylesheet(_STYLE, theme),
                        modern.strip(),
                    )
                )
            )

        def _build_ui(self) -> None:
            root = QVBoxLayout(self)
            # 调度页通常嵌在配置滚动区中；不要把隐藏/未选中的长表单
            # 通过默认 size constraint 变成 1600px 以上的顶层最小高度。
            root.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(12)

            header = FancyCard(
                elevated=True,
                theme=self._md3_theme,
            )
            header.setObjectName("schedulerHeader")
            self._fancy_style_controller.register(header)
            header_layout = QHBoxLayout()
            header.addLayout(header_layout)
            title_column = QVBoxLayout()
            title_column.setSpacing(3)
            title = QLabel("自动化")
            title.setObjectName("schedulerTitle")
            title.setProperty("md3Role", "title")
            intro = QLabel("按时间安排桌宠动作，或在窗口、互动与对话事件发生时响应。")
            intro.setObjectName("schedulerHint")
            intro.setProperty("md3Role", "muted")
            intro.setWordWrap(True)
            title_column.addWidget(title)
            title_column.addWidget(intro)
            header_layout.addLayout(title_column, 1)
            self.enabled_toggle = QCheckBox("启用自动化")
            self.enabled_toggle.setObjectName("schedulerEnabledToggle")
            self.enabled_toggle.setAccessibleName("启用自动化")
            self.enabled_toggle.setToolTip("关闭后保留全部任务和触发器，但不会执行。")
            self.enabled_toggle.toggled.connect(self._on_enabled_changed)
            header_layout.addWidget(self.enabled_toggle, 0, Qt.AlignmentFlag.AlignTop)
            root.addWidget(header)
            activity_card = FancyCard(
                "用户活跃与空闲",
                "控制互动记录和系统空闲探针。",
                icon="info",
                theme=self._md3_theme,
            )
            activity_card.setObjectName("schedulerActivityCard")
            self._fancy_style_controller.register(activity_card)
            # 活跃设置此前使用单行 QHBoxLayout。四个字段在 700px 以下会
            # 把面板最小宽度撑到 772px，嵌入配置页后右侧输入被裁掉；
            # 改为标签/控件两列布局，所有字段在窄屏仍按行可达。
            activity_layout = QGridLayout()
            activity_card.addLayout(activity_layout)
            activity_layout.setContentsMargins(0, 0, 0, 0)
            activity_layout.setHorizontalSpacing(10)
            activity_layout.setVerticalSpacing(8)
            self.activity_enabled_toggle = QCheckBox("记录用户互动并触发 idle")
            self.activity_enabled_toggle.setObjectName("schedulerActivityEnabled")
            self.activity_enabled_toggle.setToolTip(
                "记录桌宠互动和对话，用于 idle 事件以及仅活跃时执行的任务。"
            )
            self.activity_enabled_toggle.toggled.connect(self._on_activity_changed)
            activity_layout.addWidget(self.activity_enabled_toggle, 0, 0, 1, 2)
            idle_label = QLabel("空闲阈值")
            idle_label.setObjectName("schedulerActivityIdleLabel")
            self.activity_idle_spin = QDoubleSpinBox()
            self.activity_idle_spin.setObjectName("schedulerActivityIdleSeconds")
            self.activity_idle_spin.setRange(1.0, 604800.0)
            self.activity_idle_spin.setDecimals(1)
            self.activity_idle_spin.setSingleStep(10.0)
            self.activity_idle_spin.setSuffix(" 秒")
            self.activity_idle_spin.setToolTip("超过该时长没有交互时视为用户空闲。")
            self.activity_idle_spin.valueChanged.connect(self._on_activity_idle_changed)
            activity_layout.addWidget(idle_label, 1, 0)
            activity_layout.addWidget(self.activity_idle_spin, 1, 1)
            provider_label = QLabel("系统空闲来源")
            provider_label.setObjectName("schedulerSystemIdleProviderLabel")
            self.activity_system_idle_combo = QComboBox()
            self.activity_system_idle_combo.setObjectName("schedulerSystemIdleProvider")
            self.activity_system_idle_combo.addItem("不读取系统状态", "disabled")
            self.activity_system_idle_combo.addItem("X11 空闲探针", "x11")
            self.activity_system_idle_combo.addItem("Windows 空闲探针", "windows")
            self.activity_system_idle_combo.setToolTip(
                "X11 或 Windows 会读取系统空闲时间；其他平台或不可用时会安全回退为未知。"
            )
            self.activity_system_idle_combo.currentIndexChanged.connect(
                self._on_system_idle_provider_changed
            )
            self.activity_system_idle_combo.setMinimumHeight(34)
            activity_layout.addWidget(provider_label, 2, 0)
            activity_layout.addWidget(self.activity_system_idle_combo, 2, 1)
            threshold_label = QLabel("探针阈值")
            threshold_label.setObjectName("schedulerSystemIdleThresholdLabel")
            self.activity_system_idle_threshold_spin = QDoubleSpinBox()
            self.activity_system_idle_threshold_spin.setObjectName(
                "schedulerSystemIdleThresholdSeconds"
            )
            self.activity_system_idle_threshold_spin.setRange(1.0, 604800.0)
            self.activity_system_idle_threshold_spin.setDecimals(1)
            self.activity_system_idle_threshold_spin.setSingleStep(10.0)
            self.activity_system_idle_threshold_spin.setSuffix(" 秒")
            self.activity_system_idle_threshold_spin.setToolTip(
                "系统空闲超过该时长时，调度器才认为用户空闲。"
            )
            self.activity_system_idle_threshold_spin.valueChanged.connect(
                self._on_system_idle_threshold_changed
            )
            activity_layout.addWidget(threshold_label, 3, 0)
            activity_layout.addWidget(self.activity_system_idle_threshold_spin, 3, 1)
            activity_layout.setColumnStretch(0, 0)
            activity_layout.setColumnStretch(1, 1)
            root.addWidget(activity_card)
            self.mode_tabs = QTabWidget()
            self.mode_tabs.setObjectName("schedulerModeTabs")
            self.mode_tabs.setDocumentMode(True)
            self.mode_tabs.setAccessibleName("自动化类型")
            self.mode_tabs.addTab(self._build_task_card(), "定时任务")
            self.mode_tabs.addTab(self._build_trigger_card(), "事件触发")
            # 详情表单比首屏更长，使用内部滚动区承载它，避免顶层
            # 窗口被隐式撑到 1600px 以上，同时保留 mode_tabs 的公开 API。
            mode_scroll = QScrollArea()
            mode_scroll.setObjectName("schedulerModeScroll")
            mode_scroll.setWidgetResizable(True)
            mode_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            mode_scroll.setFrameShape(QFrame.Shape.NoFrame)
            mode_scroll.setWidget(self.mode_tabs)
            self._mode_scroll = mode_scroll
            root.addWidget(mode_scroll, 1)
            root.addWidget(self._build_advanced_card())
            self.status_label = QLabel("")
            self.status_label.setObjectName("schedulerStatus")
            self.status_label.setWordWrap(True)
            self.status_label.setVisible(False)
            root.addWidget(self.status_label)
            self.status_info = FancyInfoBar(
                "自动化状态",
                "配置修改会立即进入草稿。",
                severity="info",
                closable=False,
                theme=self._md3_theme,
            )
            self.status_info.setObjectName("schedulerInfoBar")
            self._fancy_style_controller.register(self.status_info)
            root.addWidget(self.status_info)
            self._update_responsive_layout(force=True)

        def _build_task_card(self) -> QWidget:
            card = FancyCard(
                "定时任务",
                "选择一个任务并在右侧编辑详细设置。",
                icon="play",
                theme=self._md3_theme,
            )
            card.setObjectName("schedulerCard")
            self._fancy_style_controller.register(card)
            layout = card.content_layout
            hint = QLabel(
                "选择已有任务编辑，或新增一个安全动作；可按间隔执行，也可设置本地时间"
                "（例如凌晨 3 点）。"
            )
            hint.setObjectName("schedulerHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            master_detail = QBoxLayout(QBoxLayout.Direction.LeftToRight)
            master_detail.setSpacing(12)
            self._task_master_detail_layout = master_detail
            master = QWidget()
            master.setObjectName("schedulerTaskMaster")
            master.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            master_layout = QVBoxLayout(master)
            master_layout.setContentsMargins(0, 0, 0, 0)
            master_layout.setSpacing(8)
            self._task_master = master
            list_heading = QLabel("任务列表")
            list_heading.setObjectName("schedulerPaneTitle")
            master_layout.addWidget(list_heading)
            self.task_list = QListWidget()
            self.task_list.setObjectName("schedulerTaskList")
            self.task_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            self.task_list.currentRowChanged.connect(self._load_task)
            self.task_list.currentRowChanged.connect(lambda _row: self._sync_delete_buttons())
            self.task_list.setToolTip("还没有任务时，点击“新增定时任务”开始配置。")
            master_layout.addWidget(self.task_list, 1)
            actions = QHBoxLayout()
            self.task_new_button = self._button("新增定时任务", "schedulerPrimary", self._new_task)
            self.task_delete_button = self._button(
                "删除选中任务", "schedulerDanger", self._delete_task
            )
            actions.addWidget(self.task_new_button)
            actions.addWidget(self.task_delete_button)
            actions.addStretch(1)
            master_layout.addLayout(actions)
            master_detail.addWidget(master, 1)
            editor = FancyCard(
                "任务详情",
                "设置执行时间、动作和活动条件。",
                icon="file",
                theme=self._md3_theme,
            )
            editor.setObjectName("schedulerEditor")
            self._fancy_style_controller.register(editor)
            form = QFormLayout()
            editor.addLayout(form)
            self._task_form = form
            self.task_name_edit = QLineEdit()
            self.task_name_edit.setObjectName("schedulerTaskName")
            self.task_name_edit.setPlaceholderText("例如：每隔一会儿眨眼")
            form.addRow("任务名称", self.task_name_edit)
            self.task_schedule_mode = QComboBox()
            self.task_schedule_mode.setObjectName("schedulerTaskScheduleMode")
            self.task_schedule_mode.setMinimumHeight(34)
            for value, label in SCHEDULE_MODES:
                self.task_schedule_mode.addItem(label, value)
            self.task_schedule_mode.currentIndexChanged.connect(self._update_task_schedule_fields)
            form.addRow("执行方式", self.task_schedule_mode)
            interval_row = QHBoxLayout()
            self.task_interval_spin = QDoubleSpinBox()
            self.task_interval_spin.setObjectName("schedulerTaskInterval")
            self.task_interval_spin.setRange(0.1, 31_536_000)
            self.task_interval_spin.setDecimals(3)
            self.task_interval_spin.setValue(5.0)
            self.task_interval_unit = QComboBox()
            self.task_interval_unit.setObjectName("schedulerTaskIntervalUnit")
            for value, label in INTERVAL_UNITS:
                self.task_interval_unit.addItem(label, value)
            interval_row.addWidget(self.task_interval_spin, 1)
            interval_row.addWidget(self.task_interval_unit)
            interval_host = QWidget()
            interval_host.setObjectName("schedulerTaskIntervalAdvanced")
            interval_host.setLayout(interval_row)
            self._task_interval_host = interval_host
            form.addRow("自定义间隔", interval_host)
            interval_presets = QGridLayout()
            interval_presets.setHorizontalSpacing(8)
            interval_presets.setVerticalSpacing(8)
            self._task_interval_preset_buttons: list[QPushButton] = []
            for index, (key, label, amount, unit) in enumerate(TASK_INTERVAL_PRESETS):
                preset_button = self._button(
                    label,
                    "schedulerSecondary",
                    lambda _checked=False, amount=amount, unit=unit: self._set_task_interval_preset(
                        amount, unit
                    ),
                )
                preset_button.setObjectName(f"schedulerTaskPreset_{key}")
                preset_button.setMinimumWidth(0)
                preset_button.setSizePolicy(
                    preset_button.sizePolicy().Policy.Expanding,
                    preset_button.sizePolicy().Policy.Preferred,
                )
                interval_presets.addWidget(preset_button, index // 2, index % 2)
                self._task_interval_preset_buttons.append(preset_button)
            for column in range(2):
                interval_presets.setColumnStretch(column, 1)
            interval_preset_host = QWidget()
            interval_preset_host.setObjectName("schedulerTaskIntervalPresets")
            interval_preset_host.setLayout(interval_presets)
            self._task_interval_preset_host = interval_preset_host
            form.addRow("常用间隔", interval_preset_host)
            custom_interval_toggle = QPushButton("显示自定义间隔")
            custom_interval_toggle.setObjectName("schedulerTaskCustomToggle")
            custom_interval_toggle.setCheckable(True)
            custom_interval_toggle.toggled.connect(self._set_task_interval_advanced_visible)
            form.addRow("", custom_interval_toggle)
            self._task_interval_custom_toggle = custom_interval_toggle
            interval_host.setVisible(False)
            daily_grid = QGridLayout()
            daily_grid.setContentsMargins(0, 0, 0, 0)
            daily_grid.setHorizontalSpacing(8)
            daily_grid.setVerticalSpacing(8)
            self.task_daily_time = QTimeEdit(QTime(3, 0))
            self.task_daily_time.setObjectName("schedulerTaskDailyTime")
            self.task_daily_time.setDisplayFormat("HH:mm")
            self.task_daily_time.setMinimumHeight(34)
            daily_grid.addWidget(self.task_daily_time, 0, 0, 1, 2)
            for index, (value, label) in enumerate(DAILY_TIME_PRESETS):
                preset_button = self._button(
                    label,
                    "schedulerSecondary",
                    lambda _checked=False, value=value: self._set_daily_time_preset(value),
                )
                preset_button.setObjectName("schedulerDailyPreset_" + value.replace(":", ""))
                preset_button.setMinimumWidth(0)
                preset_button.setSizePolicy(
                    preset_button.sizePolicy().Policy.Expanding,
                    preset_button.sizePolicy().Policy.Preferred,
                )
                daily_grid.addWidget(preset_button, 1 + index // 2, index % 2)
            for column in range(2):
                daily_grid.setColumnStretch(column, 1)
            daily_host = QWidget()
            daily_host.setObjectName("schedulerTaskDailyHost")
            daily_host.setLayout(daily_grid)
            self._task_daily_host = daily_host
            form.addRow("每日时刻（本地）", daily_host)
            daily_host.setVisible(False)
            self.task_action_combo = self._action_combo("schedulerTaskAction")
            self.task_action_combo.currentIndexChanged.connect(self._update_task_action_fields)
            form.addRow("执行动作", self.task_action_combo)
            self.task_speak_edit = QLineEdit()
            self.task_speak_edit.setObjectName("schedulerTaskSpeakText")
            self.task_speak_edit.setPlaceholderText("桌宠要说的话")
            self.task_speak_edit.setVisible(False)
            form.addRow("说话内容", self.task_speak_edit)
            self.task_require_active = QCheckBox("仅在用户活跃时执行")
            self.task_require_active.setObjectName("schedulerTaskRequireUserActive")
            self.task_require_active.setToolTip(
                "用户在空闲阈值内有过桌宠互动或提交消息时才执行；未观测到活跃状态会跳过本次计划。"
            )
            form.addRow("执行条件", self.task_require_active)
            self.task_status = QLabel("新增或选择任务后保存。")
            self.task_status.setObjectName("schedulerHint")
            self.task_status.setWordWrap(True)
            form.addRow("", self.task_status)
            editor_actions = QHBoxLayout()
            self.task_save_button = self._button("保存任务", "schedulerPrimary", self._save_task)
            self.task_cancel_button = self._button(
                "取消编辑", "schedulerSecondary", self._cancel_task
            )
            editor_actions.addWidget(self.task_save_button)
            editor_actions.addWidget(self.task_cancel_button)
            editor_actions.addStretch(1)
            form.addRow("", editor_actions)
            master_detail.addWidget(editor, 2)
            layout.addLayout(master_detail, 1)
            self._update_task_schedule_fields()
            return card

        def _build_trigger_card(self) -> QWidget:
            card = FancyCard(
                "事件触发",
                "选择一个触发器并在右侧编辑响应。",
                icon="forward",
                theme=self._md3_theme,
            )
            card.setObjectName("schedulerCard")
            self._fancy_style_controller.register(card)
            layout = card.content_layout
            hint = QLabel(
                "事件由运行时状态机产生；用户互动/空闲来自宿主已确认的交互边沿，"
                "窗口变化不会自动开启 watcher。"
            )
            hint.setObjectName("schedulerHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            master_detail = QBoxLayout(QBoxLayout.Direction.LeftToRight)
            master_detail.setSpacing(12)
            self._trigger_master_detail_layout = master_detail
            master = QWidget()
            master.setObjectName("schedulerTriggerMaster")
            master.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            master_layout = QVBoxLayout(master)
            master_layout.setContentsMargins(0, 0, 0, 0)
            master_layout.setSpacing(8)
            self._trigger_master = master
            list_heading = QLabel("触发列表")
            list_heading.setObjectName("schedulerPaneTitle")
            master_layout.addWidget(list_heading)
            self.trigger_list = QListWidget()
            self.trigger_list.setObjectName("schedulerTriggerList")
            self.trigger_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            self.trigger_list.currentRowChanged.connect(self._load_trigger)
            self.trigger_list.currentRowChanged.connect(lambda _row: self._sync_delete_buttons())
            self.trigger_list.setToolTip("还没有触发器时，点击“新增事件触发”开始配置。")
            master_layout.addWidget(self.trigger_list, 1)
            actions = QHBoxLayout()
            self.trigger_new_button = self._button(
                "新增事件触发", "schedulerPrimary", self._new_trigger
            )
            self.trigger_delete_button = self._button(
                "删除选中触发", "schedulerDanger", self._delete_trigger
            )
            actions.addWidget(self.trigger_new_button)
            actions.addWidget(self.trigger_delete_button)
            actions.addStretch(1)
            master_layout.addLayout(actions)
            master_detail.addWidget(master, 1)
            editor = FancyCard(
                "触发详情",
                "设置事件、安全动作和冷却时间。",
                icon="file",
                theme=self._md3_theme,
            )
            editor.setObjectName("schedulerEditor")
            self._fancy_style_controller.register(editor)
            form = QFormLayout()
            editor.addLayout(form)
            self._trigger_form = form
            self.trigger_event_combo = QComboBox()
            self.trigger_event_combo.setObjectName("schedulerTriggerEvent")
            for value, label in EVENT_NAMES:
                self.trigger_event_combo.addItem(label, value)
            self.trigger_event_combo.currentIndexChanged.connect(self._update_trigger_hint)
            form.addRow("触发事件", self.trigger_event_combo)
            self.trigger_action_combo = self._action_combo("schedulerTriggerAction")
            self.trigger_action_combo.currentIndexChanged.connect(
                self._update_trigger_action_fields
            )
            form.addRow("执行动作", self.trigger_action_combo)
            self.trigger_speak_edit = QLineEdit()
            self.trigger_speak_edit.setObjectName("schedulerTriggerSpeakText")
            self.trigger_speak_edit.setPlaceholderText("桌宠要说的话")
            self.trigger_speak_edit.setVisible(False)
            form.addRow("说话内容", self.trigger_speak_edit)
            self.trigger_debounce_spin = QDoubleSpinBox()
            self.trigger_debounce_spin.setObjectName("schedulerTriggerDebounce")
            self.trigger_debounce_spin.setRange(0, 86400)
            self.trigger_debounce_spin.setDecimals(3)
            self.trigger_debounce_spin.setSingleStep(1.0)
            self.trigger_debounce_spin.setSuffix(" 秒")
            debounce_host = QWidget()
            debounce_host.setObjectName("schedulerTriggerDebounceAdvanced")
            debounce_row = QHBoxLayout(debounce_host)
            debounce_row.setContentsMargins(0, 0, 0, 0)
            debounce_row.addWidget(self.trigger_debounce_spin)
            self._trigger_debounce_host = debounce_host
            form.addRow("自定义冷却", debounce_host)
            debounce_presets = QGridLayout()
            debounce_presets.setHorizontalSpacing(8)
            debounce_presets.setVerticalSpacing(8)
            self._trigger_debounce_preset_buttons: list[QPushButton] = []
            for index, (key, label, seconds) in enumerate(DEBOUNCE_PRESETS):
                preset_button = self._button(
                    label,
                    "schedulerSecondary",
                    lambda _checked=False, seconds=seconds: self._set_trigger_debounce_preset(
                        seconds
                    ),
                )
                preset_button.setObjectName(f"schedulerDebouncePreset_{key}")
                preset_button.setMinimumWidth(0)
                preset_button.setSizePolicy(
                    preset_button.sizePolicy().Policy.Expanding,
                    preset_button.sizePolicy().Policy.Preferred,
                )
                debounce_presets.addWidget(preset_button, index // 2, index % 2)
                self._trigger_debounce_preset_buttons.append(preset_button)
            for column in range(2):
                debounce_presets.setColumnStretch(column, 1)
            form.addRow("常用冷却", debounce_presets)
            custom_debounce_toggle = QPushButton("显示自定义冷却")
            custom_debounce_toggle.setObjectName("schedulerTriggerCustomToggle")
            custom_debounce_toggle.setCheckable(True)
            custom_debounce_toggle.toggled.connect(self._set_trigger_debounce_advanced_visible)
            form.addRow("", custom_debounce_toggle)
            self._trigger_debounce_custom_toggle = custom_debounce_toggle
            debounce_host.setVisible(False)
            self.trigger_match_mode_combo = QComboBox()
            self.trigger_match_mode_combo.setObjectName("schedulerTriggerMatchMode")
            for mode, label in (
                ("exact", "完全匹配"),
                ("prefix", "前缀匹配"),
                ("contains", "包含"),
                ("regex", "受限正则"),
            ):
                self.trigger_match_mode_combo.addItem(label, mode)
            form.addRow("窗口条件模式", self.trigger_match_mode_combo)
            self.trigger_process_edit = QLineEdit()
            self.trigger_process_edit.setObjectName("schedulerTriggerProcess")
            self.trigger_process_edit.setMaxLength(128)
            form.addRow("进程名", self.trigger_process_edit)
            self.trigger_app_id_edit = QLineEdit()
            self.trigger_app_id_edit.setObjectName("schedulerTriggerAppId")
            self.trigger_app_id_edit.setMaxLength(128)
            form.addRow("应用 ID", self.trigger_app_id_edit)
            self.trigger_title_edit = QLineEdit()
            self.trigger_title_edit.setObjectName("schedulerTriggerTitle")
            self.trigger_title_edit.setMaxLength(128)
            form.addRow("窗口标题", self.trigger_title_edit)
            self.trigger_duration_spin = QDoubleSpinBox()
            self.trigger_duration_spin.setObjectName("schedulerTriggerDuration")
            self.trigger_duration_spin.setRange(0, 7 * 24 * 60 * 60)
            self.trigger_duration_spin.setSuffix(" 秒")
            form.addRow("持续至少", self.trigger_duration_spin)
            self.trigger_require_active_check = QCheckBox("仅用户活跃时")
            self.trigger_require_active_check.setObjectName("schedulerTriggerRequireActive")
            form.addRow("活跃条件", self.trigger_require_active_check)
            self.trigger_hint = QLabel("")
            self.trigger_hint.setObjectName("schedulerHint")
            self.trigger_hint.setWordWrap(True)
            form.addRow("", self.trigger_hint)
            editor_actions = QHBoxLayout()
            self.trigger_save_button = self._button(
                "保存触发", "schedulerPrimary", self._save_trigger
            )
            self.trigger_cancel_button = self._button(
                "取消编辑", "schedulerSecondary", self._cancel_trigger
            )
            editor_actions.addWidget(self.trigger_save_button)
            editor_actions.addWidget(self.trigger_cancel_button)
            editor_actions.addStretch(1)
            form.addRow("", editor_actions)
            master_detail.addWidget(editor, 2)
            layout.addLayout(master_detail, 1)
            return card

        def _build_advanced_card(self) -> QWidget:
            card = FancyCard(
                "高级条目（只读保留）",
                "无法安全映射的条目只显示摘要，不会被自动改写。",
                icon="warning",
                theme=self._md3_theme,
            )
            card.setObjectName("schedulerAdvancedCard")
            self._fancy_style_controller.register(card)
            self._advanced_card = card
            layout = card.content_layout
            hint = QLabel(
                "无法安全映射到上述预设的条目不会被改写；它们只显示摘要。"
                "如需移除，请选中后明确点击删除。"
            )
            hint.setObjectName("schedulerAdvancedHint")
            hint.setWordWrap(True)
            layout.addWidget(hint)
            rows = QGridLayout()
            self.advanced_task_list = QListWidget()
            self.advanced_task_list.setObjectName("schedulerAdvancedTaskList")
            self.advanced_task_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            self.advanced_task_list.currentRowChanged.connect(
                lambda _row: self._sync_delete_buttons()
            )
            self.advanced_trigger_list = QListWidget()
            self.advanced_trigger_list.setObjectName("schedulerAdvancedTriggerList")
            self.advanced_trigger_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            self.advanced_trigger_list.currentRowChanged.connect(
                lambda _row: self._sync_delete_buttons()
            )
            rows.addWidget(QLabel("高级定时任务"), 0, 0)
            rows.addWidget(QLabel("高级事件触发"), 0, 1)
            rows.addWidget(self.advanced_task_list, 1, 0)
            rows.addWidget(self.advanced_trigger_list, 1, 1)
            self.advanced_task_delete = self._button(
                "删除选中高级任务", "schedulerDanger", self._delete_advanced_task
            )
            self.advanced_trigger_delete = self._button(
                "删除选中高级触发", "schedulerDanger", self._delete_advanced_trigger
            )
            rows.addWidget(self.advanced_task_delete, 2, 0)
            rows.addWidget(self.advanced_trigger_delete, 2, 1)
            rows.setColumnStretch(0, 1)
            rows.setColumnStretch(1, 1)
            layout.addLayout(rows)
            return card

        def _update_responsive_layout(self, *, force: bool = False) -> None:
            """在窄屏把主从编辑器堆叠，宽屏恢复 WinUI 双栏。"""

            narrow = self.width() < 760
            if narrow == self._responsive_narrow and not force:
                return
            self._responsive_narrow = narrow
            direction = (
                QBoxLayout.Direction.TopToBottom if narrow else QBoxLayout.Direction.LeftToRight
            )
            for layout in (
                self._task_master_detail_layout,
                self._trigger_master_detail_layout,
            ):
                if layout is not None:
                    layout.setDirection(direction)
            for master in (self._task_master, self._trigger_master):
                if master is None:
                    continue
                master.setMinimumWidth(0)
                master.setMaximumWidth(16777215 if narrow else 286)
            for widget in (self.task_list, self.trigger_list):
                widget.setMinimumHeight(96 if narrow else 180)
                widget.setMaximumHeight(150 if narrow else 16777215)
            self.setProperty("responsiveMode", "stacked" if narrow else "split")
            style = self.style()
            style.unpolish(self)
            style.polish(self)
            self.updateGeometry()

        def sizeHint(self) -> QSize:  # noqa: N802
            """返回可嵌入的稳定首屏高度，不被隐藏编辑字段撑大。"""

            width = max(420, int(self.width() or 760))
            return QSize(width, 720)

        def minimumSizeHint(self) -> QSize:  # noqa: N802
            """保留窄屏可用下限，并允许外层滚动容器管理详细表单。"""

            return QSize(360, 360)

        def resizeEvent(self, event: object) -> None:  # noqa: N802
            """随容器宽度重排主从区域。"""

            super().resizeEvent(event)
            self._update_responsive_layout()

        @staticmethod
        def _button(text: str, role: str, slot: object) -> QPushButton:
            button = QPushButton(text)
            button.setProperty("role", role)
            button.clicked.connect(slot)  # type: ignore[arg-type]
            button.setMinimumHeight(34)
            return button

        @staticmethod
        def _action_combo(object_name: str) -> QComboBox:
            combo = QComboBox()
            combo.setObjectName(object_name)
            for preset in ACTION_PRESETS:
                combo.addItem(preset.label, preset.key)
            combo.setMinimumHeight(34)
            return combo

        def set_values(self, values: Mapping[str, Any] | None) -> None:
            """载入配置草稿并分类可编辑项与高级只读项。"""

            section = copy.deepcopy(dict(values or {}))
            self._activity_declared = "activity" in section
            activity = section.get("activity")
            activity = activity if isinstance(activity, Mapping) else {}
            tasks = section.get("tasks")
            triggers = section.get("triggers")
            self._section = section
            self._task_entries = [
                {"raw": copy.deepcopy(item), "known": _is_known_task(item)}
                for item in (tasks if isinstance(tasks, (list, tuple)) else [])
            ]
            self._trigger_entries = [
                {"raw": copy.deepcopy(item), "known": _is_known_trigger(item)}
                for item in (triggers if isinstance(triggers, (list, tuple)) else [])
            ]
            self._building = True
            try:
                self.enabled_toggle.setChecked(bool(section.get("enabled", True)))
                self.activity_enabled_toggle.setChecked(bool(activity.get("enabled", True)))
                try:
                    self.activity_idle_spin.setValue(float(activity.get("idle_seconds", 300.0)))
                except (TypeError, ValueError, OverflowError):
                    self.activity_idle_spin.setValue(300.0)
                provider = activity.get("system_idle_provider", "disabled")
                provider_index = self.activity_system_idle_combo.findData(provider)
                self.activity_system_idle_combo.setCurrentIndex(
                    provider_index if provider_index >= 0 else 0
                )
                try:
                    self.activity_system_idle_threshold_spin.setValue(
                        float(activity.get("system_idle_threshold_seconds", 300.0))
                    )
                except (TypeError, ValueError, OverflowError):
                    self.activity_system_idle_threshold_spin.setValue(300.0)
                self._task_edit_index = None
                self._trigger_edit_index = None
                self._render_lists()
                self._clear_task_editor()
                self._clear_trigger_editor()
            finally:
                self._building = False

        def values(self) -> dict[str, Any]:
            """返回保留未知条目顺序的 scheduler 草稿。"""

            result = copy.deepcopy(self._section)
            result["enabled"] = bool(self.enabled_toggle.isChecked())
            if self._activity_declared or self.activity_enabled_toggle.isChecked() is False:
                activity = result.get("activity")
                activity = dict(activity) if isinstance(activity, Mapping) else {}
                activity["enabled"] = bool(self.activity_enabled_toggle.isChecked())
                activity["idle_seconds"] = float(self.activity_idle_spin.value())
                activity["system_idle_provider"] = str(
                    self.activity_system_idle_combo.currentData() or "disabled"
                )
                activity["system_idle_threshold_seconds"] = float(
                    self.activity_system_idle_threshold_spin.value()
                )
                result["activity"] = activity
            result["tasks"] = [copy.deepcopy(item["raw"]) for item in self._task_entries]
            result["triggers"] = [copy.deepcopy(item["raw"]) for item in self._trigger_entries]
            return result

        def _emit_changed(self, message: str) -> None:
            if self._building:
                return
            self._section = self.values()
            self._set_status(message)
            self.changed.emit(copy.deepcopy(self._section))

        def _set_status(self, message: str, *, error: bool = False) -> None:
            self.status_label.setText(str(message or ""))
            self.status_label.setProperty("state", "error" if error else "ok")
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)
            info_bar = getattr(self, "status_info", None)
            if isinstance(info_bar, FancyInfoBar):
                text = str(message or "")
                severity = "error" if error else "success"
                if not error and "尚未保存" in text:
                    severity = "warning"
                info_bar.setSeverity(severity)
                info_bar.setMessage(text)
            self.statusChanged.emit(str(message or ""))

        def _on_enabled_changed(self, enabled: bool) -> None:
            if not self._building:
                self._emit_changed(f"调度器：{'已开启' if enabled else '已关闭'}（尚未保存）")

        def _on_activity_changed(self, enabled: bool) -> None:
            if not self._building:
                self._activity_declared = True
                self._emit_changed(f"用户活跃观察：{'已开启' if enabled else '已关闭'}（尚未保存）")

        def _on_activity_idle_changed(self, _value: float) -> None:
            if not self._building:
                self._activity_declared = True
                self._emit_changed("用户空闲阈值已更新（尚未保存）")

        def _on_system_idle_provider_changed(self, _index: int) -> None:
            if not self._building:
                self._activity_declared = True
                label = str(self.activity_system_idle_combo.currentText() or "不读取系统状态")
                self._emit_changed(f"系统空闲来源：{label}（尚未保存）")

        def _on_system_idle_threshold_changed(self, _value: float) -> None:
            if not self._building:
                self._activity_declared = True
                self._emit_changed("系统空闲探针阈值已更新（尚未保存）")

        @staticmethod
        def _action_label(raw: object) -> str:
            key = _action_preset(raw)
            if key is None:
                return "自定义动作"
            return _PRESETS_BY_KEY[key].label

        def _render_lists(self) -> None:
            self.task_list.blockSignals(True)
            self.trigger_list.blockSignals(True)
            self.advanced_task_list.blockSignals(True)
            self.advanced_trigger_list.blockSignals(True)
            try:
                self.task_list.clear()
                self.trigger_list.clear()
                self.advanced_task_list.clear()
                self.advanced_trigger_list.clear()
                for index, entry in enumerate(self._task_entries):
                    raw = entry["raw"]
                    if entry["known"]:
                        metadata = raw.get("metadata")
                        active_only = isinstance(metadata, Mapping) and (
                            metadata.get("require_user_active") is True
                        )
                        item = QListWidgetItem(
                            f"{_safe_label(raw.get('name'), fallback='未命名任务')} · "
                            f"{_safe_label(raw.get('expression'), fallback='无间隔')} · "
                            f"{self._action_label(raw.get('action'))}"
                            f"{' · 仅活跃时' if active_only else ''}"
                        )
                        item.setData(Qt.ItemDataRole.UserRole, index)
                        self.task_list.addItem(item)
                    else:
                        raw_name = raw.get("name") if isinstance(raw, Mapping) else ""
                        item = QListWidgetItem(
                            f"高级定时条目 · {_safe_label(raw_name, fallback='未命名')}（只读保留）"
                        )
                        item.setData(Qt.ItemDataRole.UserRole, index)
                        self.advanced_task_list.addItem(item)
                for index, entry in enumerate(self._trigger_entries):
                    raw = entry["raw"]
                    if entry["known"]:
                        event_name = str(raw.get("event_name", "") or "").strip().lower()
                        item = QListWidgetItem(
                            f"{_EVENT_LABELS.get(event_name, '事件')} · "
                            f"{self._action_label(raw.get('action'))}"
                        )
                        item.setData(Qt.ItemDataRole.UserRole, index)
                        self.trigger_list.addItem(item)
                    else:
                        raw_id = raw.get("trigger_id") if isinstance(raw, Mapping) else ""
                        item = QListWidgetItem(
                            f"高级事件条目 · {_safe_label(raw_id, fallback='未命名')}（只读保留）"
                        )
                        item.setData(Qt.ItemDataRole.UserRole, index)
                        self.advanced_trigger_list.addItem(item)
                self._add_empty_item(
                    self.task_list,
                    "还没有定时任务；点击上方“新增定时任务”开始。",
                )
                self._add_empty_item(
                    self.trigger_list,
                    "还没有事件触发；点击上方“新增事件触发”开始。",
                )
                self._add_empty_item(
                    self.advanced_task_list,
                    "没有需要保留的高级定时条目。",
                )
                self._add_empty_item(
                    self.advanced_trigger_list,
                    "没有需要保留的高级事件条目。",
                )
            finally:
                self.task_list.blockSignals(False)
                self.trigger_list.blockSignals(False)
                self.advanced_task_list.blockSignals(False)
                self.advanced_trigger_list.blockSignals(False)
            if self._advanced_card is not None:
                self._advanced_card.setVisible(
                    bool(
                        self.advanced_task_list.count()
                        and self.advanced_task_list.item(0).data(Qt.ItemDataRole.UserRole)
                        is not None
                    )
                    or bool(
                        self.advanced_trigger_list.count()
                        and self.advanced_trigger_list.item(0).data(Qt.ItemDataRole.UserRole)
                        is not None
                    )
                )
            self._sync_delete_buttons()
            self._update_trigger_hint()

        @staticmethod
        def _add_empty_item(widget: QListWidget, text: str) -> None:
            """在空列表中显示引导，但不制造可删除的配置条目。"""

            if widget.count() != 0:
                return
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, None)
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            widget.addItem(item)

        def _sync_delete_buttons(self) -> None:
            """只有存在真实条目时才启用删除，避免空列表误操作。"""

            task_index = self._selected_index(self.task_list)
            trigger_index = self._selected_index(self.trigger_list)
            advanced_task_index = self._selected_index(self.advanced_task_list)
            advanced_trigger_index = self._selected_index(self.advanced_trigger_list)
            self.task_delete_button.setEnabled(task_index is not None)
            self.trigger_delete_button.setEnabled(trigger_index is not None)
            self.advanced_task_delete.setEnabled(advanced_task_index is not None)
            self.advanced_trigger_delete.setEnabled(advanced_trigger_index is not None)

        def _selected_index(self, widget: QListWidget) -> int | None:
            item = widget.currentItem()
            if item is None:
                return None
            value = item.data(Qt.ItemDataRole.UserRole)
            return int(value) if isinstance(value, int) else None

        def _new_task(self) -> None:
            self._task_edit_index = None
            self._clear_task_editor()
            self.task_name_edit.setText("新的定时任务")
            self.task_schedule_mode.setCurrentIndex(self.task_schedule_mode.findData("interval"))
            self.task_interval_spin.setValue(5.0)
            self.task_interval_unit.setCurrentIndex(self.task_interval_unit.findData("m"))
            self.task_daily_time.setTime(QTime(3, 0))
            self.task_action_combo.setCurrentIndex(self.task_action_combo.findData("blink"))
            self.task_status.setText("填写名称后点击“保存任务”。")
            self._update_task_action_fields()

        def _set_daily_time_preset(self, value: str) -> None:
            """选择一个常用每日提醒时刻。"""

            try:
                hour_text, minute_text = str(value).split(":", 1)
                self.task_daily_time.setTime(QTime(int(hour_text), int(minute_text)))
            except (TypeError, ValueError, OverflowError):
                return
            self.task_schedule_mode.setCurrentIndex(self.task_schedule_mode.findData("daily"))
            self.task_status.setText("已选择每日提醒时刻（本地时间）。")

        def _set_task_interval_preset(self, amount: float, unit: str) -> None:
            self.task_schedule_mode.setCurrentIndex(self.task_schedule_mode.findData("interval"))
            self.task_interval_spin.setValue(float(amount))
            index = self.task_interval_unit.findData(str(unit))
            if index >= 0:
                self.task_interval_unit.setCurrentIndex(index)
            self._set_task_interval_advanced_visible(False)
            self.task_status.setText("已选择常用执行间隔。")

        def _set_task_interval_advanced_visible(self, visible: bool) -> None:
            enabled = bool(visible)
            self._task_interval_host.setVisible(enabled)
            self._task_interval_custom_toggle.setText(
                "收起自定义间隔" if enabled else "显示自定义间隔"
            )

        def _update_task_schedule_fields(self, _index: int = -1) -> None:
            """切换间隔/每日时刻编辑面板，避免两套值同时生效。"""

            daily = self.task_schedule_mode.currentData() == "daily"
            self._task_interval_host.setVisible(
                not daily and self._task_interval_custom_toggle.isChecked()
            )
            self._task_interval_custom_toggle.setVisible(not daily)
            self._task_interval_preset_host.setVisible(not daily)
            self._task_daily_host.setVisible(daily)

        def _load_task(self, _row: int) -> None:
            index = self._selected_index(self.task_list)
            if index is None or index >= len(self._task_entries):
                return
            raw = self._task_entries[index]["raw"]
            if not self._task_entries[index]["known"]:
                return
            self._task_edit_index = index
            self.task_name_edit.setText(str(raw.get("name", "")))
            expression = str(raw.get("expression", ""))
            daily_time = self._parse_daily_expression_for_ui(expression)
            if daily_time is not None:
                self.task_schedule_mode.setCurrentIndex(self.task_schedule_mode.findData("daily"))
                self.task_daily_time.setTime(daily_time)
            else:
                self.task_schedule_mode.setCurrentIndex(
                    self.task_schedule_mode.findData("interval")
                )
            amount, unit = self._parse_expression_for_ui(expression)
            self.task_interval_spin.setValue(amount)
            unit_index = self.task_interval_unit.findData(unit)
            self.task_interval_unit.setCurrentIndex(max(0, unit_index))
            is_preset = any(
                float(amount) == preset_amount and unit == preset_unit
                for _key, _label, preset_amount, preset_unit in TASK_INTERVAL_PRESETS
            )
            self._set_task_interval_advanced_visible(not is_preset)
            key = _action_preset(raw.get("action")) or "blink"
            self.task_action_combo.setCurrentIndex(self.task_action_combo.findData(key))
            arguments = (
                raw.get("action", {}).get("arguments", {})
                if isinstance(raw.get("action"), Mapping)
                else {}
            )
            preset = _PRESETS_BY_KEY.get(key)
            argument_name = preset.argument_name if preset is not None else "text"
            self.task_speak_edit.setText(str(arguments.get(argument_name, "")))
            metadata = raw.get("metadata") if isinstance(raw, Mapping) else {}
            self.task_require_active.setChecked(
                isinstance(metadata, Mapping) and metadata.get("require_user_active") is True
            )
            self._update_task_action_fields()
            self._update_task_schedule_fields()

        @staticmethod
        def _parse_expression_for_ui(expression: str) -> tuple[float, str]:
            text = str(expression or "").strip().lower()
            for unit, _label in INTERVAL_UNITS:
                if text.startswith("every:") and text.endswith(unit):
                    try:
                        return float(text[6 : -len(unit)]), unit
                    except ValueError:
                        break
            return 5.0, "m"

        @staticmethod
        def _parse_daily_expression_for_ui(expression: str) -> QTime | None:
            """把 daily:HH:MM 转为 Qt 时刻；非法值交给后端校验。"""

            text = str(expression or "").strip().lower()
            if not text.startswith("daily:"):
                return None
            try:
                hour_text, minute_text = text[6:].split(":", 1)
                value = QTime(int(hour_text), int(minute_text))
            except (TypeError, ValueError, OverflowError):
                return None
            return value if value.isValid() else None

        def _clear_task_editor(self) -> None:
            self.task_name_edit.clear()
            self.task_schedule_mode.setCurrentIndex(self.task_schedule_mode.findData("interval"))
            self.task_interval_spin.setValue(5.0)
            self.task_interval_unit.setCurrentIndex(self.task_interval_unit.findData("m"))
            self.task_daily_time.setTime(QTime(3, 0))
            self._set_task_interval_advanced_visible(False)
            self._update_task_schedule_fields()
            self.task_action_combo.setCurrentIndex(self.task_action_combo.findData("blink"))
            self.task_speak_edit.clear()
            self.task_require_active.setChecked(False)
            self._update_task_action_fields()

        def _update_task_action_fields(self) -> None:
            visible = self.task_action_combo.currentData() in {"speak", "proactive"}
            self.task_speak_edit.setVisible(bool(visible))
            label = self._task_form.labelForField(self.task_speak_edit)
            if label is not None:
                label.setVisible(bool(visible))

        def _save_task(self) -> None:
            name = str(self.task_name_edit.text() or "").strip()
            if not name:
                self.task_status.setText("请填写任务名称。")
                return
            if self.task_schedule_mode.currentData() == "daily":
                daily_time = self.task_daily_time.time()
                expression = f"daily:{daily_time.toString('HH:mm')}"
            else:
                amount = float(self.task_interval_spin.value())
                unit = str(self.task_interval_unit.currentData() or "m")
                expression = f"every:{_format_amount(amount)}{unit}"
            try:
                validate_schedule_expression(expression)
            except (ScheduleExpressionError, TypeError, ValueError):
                self.task_status.setText("执行时间格式无效，请检查间隔或每日时刻。")
                return
            action_key = str(self.task_action_combo.currentData() or "")
            action = _action_from_preset(action_key, self.task_speak_edit.text())
            if action is None:
                self.task_status.setText("请补充说话内容。")
                return
            require_active = bool(self.task_require_active.isChecked())
            if self._task_edit_index is None:
                raw: dict[str, Any] = {
                    "task_id": self._next_id("task", self._task_entries),
                    "name": name,
                    "expression": expression,
                    "action": action,
                    "owner": "config",
                    "enabled": True,
                }
                if require_active:
                    raw["metadata"] = {"require_user_active": True}
                self._task_entries.append({"raw": raw, "known": True})
            else:
                entry = self._task_entries[self._task_edit_index]
                raw = copy.deepcopy(entry["raw"])
                raw.update({"name": name, "expression": expression, "action": action})
                metadata = raw.get("metadata")
                metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
                if require_active:
                    metadata["require_user_active"] = True
                else:
                    metadata.pop("require_user_active", None)
                if metadata:
                    raw["metadata"] = metadata
                else:
                    raw.pop("metadata", None)
                entry["raw"] = raw
                entry["known"] = True
            self._render_lists()
            self._emit_changed("定时任务已更新（尚未保存）")

        def _cancel_task(self) -> None:
            self._task_edit_index = None
            self._clear_task_editor()
            self.task_status.setText("已取消任务编辑。")

        def _delete_task(self) -> None:
            index = self._selected_index(self.task_list)
            if index is None or index >= len(self._task_entries):
                self._set_status("请先选择要删除的定时任务。", error=True)
                return
            del self._task_entries[index]
            self._task_edit_index = None
            self._render_lists()
            self._emit_changed("已删除选中的定时任务（尚未保存）")

        def _new_trigger(self) -> None:
            self._trigger_edit_index = None
            self._clear_trigger_editor()
            self.trigger_event_combo.setCurrentIndex(self.trigger_event_combo.findData("startup"))
            self.trigger_action_combo.setCurrentIndex(self.trigger_action_combo.findData("blink"))
            self.trigger_status_message("选择事件后点击“保存触发”。")
            self._update_trigger_action_fields()

        def _set_trigger_debounce_preset(self, seconds: float) -> None:
            self.trigger_debounce_spin.setValue(float(seconds))
            self._set_trigger_debounce_advanced_visible(False)
            self.trigger_status_message("已选择常用重复冷却。")

        def _set_trigger_debounce_advanced_visible(self, visible: bool) -> None:
            enabled = bool(visible)
            self._trigger_debounce_host.setVisible(enabled)
            self._trigger_debounce_custom_toggle.setText(
                "收起自定义冷却" if enabled else "显示自定义冷却"
            )

        def _load_trigger(self, _row: int) -> None:
            index = self._selected_index(self.trigger_list)
            if index is None or index >= len(self._trigger_entries):
                return
            entry = self._trigger_entries[index]
            if not entry["known"]:
                return
            raw = entry["raw"]
            self._trigger_edit_index = index
            event_index = self.trigger_event_combo.findData(raw.get("event_name"))
            self.trigger_event_combo.setCurrentIndex(max(0, event_index))
            action_key = _action_preset(raw.get("action")) or "blink"
            self.trigger_action_combo.setCurrentIndex(
                self.trigger_action_combo.findData(action_key)
            )
            arguments = (
                raw.get("action", {}).get("arguments", {})
                if isinstance(raw.get("action"), Mapping)
                else {}
            )
            preset = _PRESETS_BY_KEY.get(action_key)
            argument_name = preset.argument_name if preset is not None else "text"
            self.trigger_speak_edit.setText(str(arguments.get(argument_name, "")))
            debounce = float(raw.get("debounce_seconds", 0.0))
            self.trigger_debounce_spin.setValue(debounce)
            conditions = raw.get("conditions", {})
            conditions = conditions if isinstance(conditions, Mapping) else {}
            text_rules = [
                conditions.get(key)
                for key in ("process_name", "app_id", "title")
                if isinstance(conditions.get(key), Mapping)
            ]
            mode = str(text_rules[0].get("mode", "exact")) if text_rules else "exact"
            self.trigger_match_mode_combo.setCurrentIndex(
                max(0, self.trigger_match_mode_combo.findData(mode))
            )
            for key, editor in (
                ("process_name", self.trigger_process_edit),
                ("app_id", self.trigger_app_id_edit),
                ("title", self.trigger_title_edit),
            ):
                condition = conditions.get(key, {})
                editor.setText(
                    str(condition.get("value", "")) if isinstance(condition, Mapping) else ""
                )
            self.trigger_duration_spin.setValue(
                float(conditions.get("min_duration_seconds", 0.0) or 0.0)
            )
            self.trigger_require_active_check.setChecked(
                conditions.get("require_user_active") is True
            )
            is_preset = any(
                debounce == preset_seconds for _key, _label, preset_seconds in DEBOUNCE_PRESETS
            )
            self._set_trigger_debounce_advanced_visible(not is_preset)
            self._update_trigger_action_fields()

        def _clear_trigger_editor(self) -> None:
            self.trigger_event_combo.setCurrentIndex(self.trigger_event_combo.findData("startup"))
            self.trigger_action_combo.setCurrentIndex(self.trigger_action_combo.findData("blink"))
            self.trigger_speak_edit.clear()
            self.trigger_debounce_spin.setValue(0)
            self.trigger_match_mode_combo.setCurrentIndex(
                max(0, self.trigger_match_mode_combo.findData("exact"))
            )
            self.trigger_process_edit.clear()
            self.trigger_app_id_edit.clear()
            self.trigger_title_edit.clear()
            self.trigger_duration_spin.setValue(0)
            self.trigger_require_active_check.setChecked(False)
            self._set_trigger_debounce_advanced_visible(False)
            self._update_trigger_action_fields()
            self._update_trigger_hint()

        def _update_trigger_action_fields(self) -> None:
            visible = self.trigger_action_combo.currentData() in {"speak", "proactive"}
            self.trigger_speak_edit.setVisible(bool(visible))
            label = self._trigger_form.labelForField(self.trigger_speak_edit)
            if label is not None:
                label.setVisible(bool(visible))

        def trigger_status_message(self, message: str) -> None:
            self.trigger_hint.setText(str(message or ""))

        def _update_trigger_hint(self, _index: int = -1) -> None:
            event_name = str(self.trigger_event_combo.currentData() or "")
            if event_name in {"window_changed", "window_active"}:
                self.trigger_hint.setText(
                    "需要先在“窗口观察”页开启 watcher；此处不会自动扩大权限。"
                )
            elif event_name in {"user_interaction", "user_active"}:
                self.trigger_hint.setText("桌宠收到一次已确认的用户互动时触发，不携带输入内容。")
            elif event_name == "idle":
                self.trigger_hint.setText(
                    "用户在活跃阈值内没有新互动时触发一次；再次互动后可重新进入空闲。"
                )
            else:
                self.trigger_hint.setText("可设置重复触发冷却，避免短时间内连续执行。")

        def _save_trigger(self) -> None:
            event_name = str(self.trigger_event_combo.currentData() or "startup")
            action_key = str(self.trigger_action_combo.currentData() or "")
            action = _action_from_preset(action_key, self.trigger_speak_edit.text())
            if action is None:
                self.trigger_status_message("请补充说话内容。")
                return
            debounce = float(self.trigger_debounce_spin.value())
            mode = str(self.trigger_match_mode_combo.currentData() or "exact")
            conditions: dict[str, Any] = {}
            for key, editor in (
                ("process_name", self.trigger_process_edit),
                ("app_id", self.trigger_app_id_edit),
                ("title", self.trigger_title_edit),
            ):
                value = editor.text().strip()
                if value:
                    conditions[key] = {"mode": mode, "value": value}
            if self.trigger_duration_spin.value() > 0:
                conditions["min_duration_seconds"] = float(self.trigger_duration_spin.value())
            if self.trigger_require_active_check.isChecked():
                conditions["require_user_active"] = True
            if self._trigger_edit_index is None:
                raw: dict[str, Any] = {
                    "trigger_id": self._next_id("trigger", self._trigger_entries),
                    "event_name": event_name,
                    "action": action,
                    "owner": "config",
                    "debounce_seconds": debounce,
                }
                if conditions:
                    raw["conditions"] = conditions
                self._trigger_entries.append({"raw": raw, "known": True})
            else:
                entry = self._trigger_entries[self._trigger_edit_index]
                raw = copy.deepcopy(entry["raw"])
                raw.update(
                    {
                        "event_name": event_name,
                        "action": action,
                        "debounce_seconds": debounce,
                    }
                )
                if conditions:
                    raw["conditions"] = conditions
                else:
                    raw.pop("conditions", None)
                entry["raw"] = raw
                entry["known"] = True
            self._render_lists()
            self._emit_changed("事件触发已更新（尚未保存）")

        def _cancel_trigger(self) -> None:
            self._trigger_edit_index = None
            self._clear_trigger_editor()
            self.trigger_status_message("已取消事件编辑。")

        def _delete_trigger(self) -> None:
            index = self._selected_index(self.trigger_list)
            if index is None or index >= len(self._trigger_entries):
                self._set_status("请先选择要删除的事件触发。", error=True)
                return
            del self._trigger_entries[index]
            self._trigger_edit_index = None
            self._render_lists()
            self._emit_changed("已删除选中的事件触发（尚未保存）")

        def _delete_advanced_task(self) -> None:
            row = self._selected_index(self.advanced_task_list)
            if row is None:
                self._set_status("请先选择要删除的高级定时条目。", error=True)
                return
            item = self.advanced_task_list.currentItem()
            index = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if not isinstance(index, int):
                return
            del self._task_entries[index]
            self._render_lists()
            self._emit_changed("已明确删除高级定时条目（尚未保存）")

        def _delete_advanced_trigger(self) -> None:
            row = self._selected_index(self.advanced_trigger_list)
            if row is None:
                self._set_status("请先选择要删除的高级事件条目。", error=True)
                return
            item = self.advanced_trigger_list.currentItem()
            index = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if not isinstance(index, int):
                return
            del self._trigger_entries[index]
            self._render_lists()
            self._emit_changed("已明确删除高级事件条目（尚未保存）")

        @staticmethod
        def _next_id(prefix: str, entries: Sequence[Mapping[str, Any]]) -> str:
            used = {
                str(entry.get("raw", {}).get(f"{prefix}_id", ""))
                for entry in entries
                if isinstance(entry, Mapping) and isinstance(entry.get("raw"), Mapping)
            }
            index = 1
            while f"{prefix}-{index}" in used:
                index += 1
            return f"{prefix}-{index}"


else:

    class SchedulerPanel:  # pragma: no cover
        """无 Qt 环境时的明确不可用占位。"""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("PySide6 is not installed")


__all__ = [
    "ACTION_PRESETS",
    "ActionPreset",
    "DEBOUNCE_PRESETS",
    "EVENT_NAMES",
    "INTERVAL_UNITS",
    "SchedulerPanel",
    "TASK_INTERVAL_PRESETS",
    "_action_from_preset",
    "_action_preset",
    "_is_known_task",
    "_is_known_trigger",
]
