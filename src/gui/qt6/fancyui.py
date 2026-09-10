# SPDX-License-Identifier: GPL-3.0-only
# Adapted from QWidget-FancyUI control-state, navigation and animation concepts.
# Current C++ reference: e6b837a2f668fc0beed9463b6dc48d92f787f272
# Historical PySide6 reference: a03ae9c9e8ad98d79ded93ebc00a242602f11112
# Modification notice: this module is a cross-platform MeaPet PySide6 adaptation.
from __future__ import annotations

import weakref
from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

from gui.qt6.md3 import (
    DARK_MD3_THEME,
    MD3Theme,
    MD3ThemeMode,
    build_md3_stylesheet,
    build_qt_palette,
    theme_from_mapping,
)

try:  # Qt 仍是可选依赖，核心服务可以在没有桌面环境时导入。
    from PySide6.QtCore import (
        QEasingCurve,
        QEvent,
        QPointF,
        QRectF,
        QSize,
        Qt,
        QTimer,
        QVariantAnimation,
        Signal,
    )
    from PySide6.QtGui import (
        QAction,
        QColor,
        QEnterEvent,
        QIcon,
        QKeyEvent,
        QMouseEvent,
        QPainter,
        QPainterPath,
        QPen,
        QPixmap,
        QPolygonF,
        QResizeEvent,
        QShowEvent,
    )
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGraphicsOpacityEffect,
        QHBoxLayout,
        QLabel,
        QMenu,
        QPushButton,
        QSizePolicy,
        QStyle,
        QTabBar,
        QTabWidget,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

    pyside6_fancy_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_fancy_available = False


FANCY_ICON_NAMES: tuple[str, ...] = (
    "add",
    "back",
    "close",
    "delete",
    "down",
    "error",
    "file",
    "folder",
    "forward",
    "home",
    "info",
    "mute",
    "play",
    "refresh",
    "save",
    "stop",
    "success",
    "up",
    "volume",
    "warning",
)


def _duration(theme: MD3Theme, preferred: int | None = None) -> int:
    """返回 Fluent 状态动效时长；降低动态效果时立即完成。"""

    if theme.motion.reduced_motion:
        return 0
    source = theme.motion.short_duration_ms if preferred is None else int(preferred)
    return max(150, min(250, source))


def _rgba(color: str, alpha: int) -> str:
    value = color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red}, {green}, {blue}, {max(0, min(255, alpha))})"


def build_fancy_stylesheet(theme: MD3Theme) -> str:
    """由一个 MD3 主题生成 FancyUI 的 Fluent 状态与材质样式。"""

    if not isinstance(theme, MD3Theme):
        raise TypeError("theme must be an MD3Theme")
    colors = theme.colors
    layout = theme.layout
    acrylic_alpha = 230 if theme.mode is MD3ThemeMode.DARK else 244
    mica_alpha = 246 if theme.mode is MD3ThemeMode.DARK else 250
    return f"""
QWidget[fancyRoot="true"] {{
    background-color: {colors.surface};
    color: {colors.on_surface};
}}
QFrame[fancyRole="card"] {{ background: transparent; border: none; }}
QFrame[fancyRole="command-bar"] {{
    background-color: {_rgba(colors.surface_container, acrylic_alpha)};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_medium_px}px;
}}
QFrame[fancyRole="info-bar"] {{
    background-color: {_rgba(colors.surface_container_high, mica_alpha)};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_medium_px}px;
}}
QFrame[fancyRole="info-bar"][fancySeverity="info"] {{ border-left: 3px solid {colors.primary}; }}
QFrame[fancyRole="info-bar"][fancySeverity="success"] {{ border-left: 3px solid {colors.success}; }}
QFrame[fancyRole="info-bar"][fancySeverity="warning"] {{ border-left: 3px solid {colors.warning}; }}
QFrame[fancyRole="info-bar"][fancySeverity="error"] {{ border-left: 3px solid {colors.error}; }}
QLabel[fancyRole="info-title"] {{ font-weight: 600; color: {colors.on_surface}; }}
QLabel[fancyRole="info-message"] {{ color: {colors.on_surface_variant}; }}
QPushButton[fancyRole="info-action"] {{
    background: transparent;
    color: {colors.primary};
    border: 1px solid transparent;
    border-radius: {layout.radius_small_px}px;
    padding: 0 10px;
}}
QPushButton[fancyRole="info-action"]:hover {{
    background-color: {_rgba(colors.primary, 24)};
}}
QPushButton[fancyRole="info-action"]:pressed {{
    background-color: {_rgba(colors.primary, 40)};
}}
QToolButton[fancyRole="icon-button"], QToolButton[fancyRole="command"] {{
    min-width: 32px;
    min-height: 32px;
    padding: 0 8px;
    background: transparent;
    color: {colors.on_surface};
    border: 1px solid transparent;
    border-radius: {layout.radius_small_px}px;
}}
QToolButton[fancyRole="icon-button"]:hover, QToolButton[fancyRole="command"]:hover {{
    background-color: {_rgba(colors.on_surface, 18)};
}}
QToolButton[fancyRole="icon-button"]:pressed, QToolButton[fancyRole="command"]:pressed {{
    background-color: {_rgba(colors.on_surface, 30)};
}}
QToolButton[fancyRole="command-primary"] {{
    min-height: 32px;
    padding: 0 12px;
    color: {colors.on_primary};
    background-color: {colors.primary};
    border: 1px solid {colors.primary};
    border-radius: {layout.radius_small_px}px;
}}
QToolButton[fancyRole="command-primary"]:hover {{
    background-color: {_rgba(colors.primary, 220)};
}}
QToolButton:focus, QPushButton:focus {{
    border: {max(1, layout.focus_width_px)}px solid {colors.primary};
}}
QToolButton:disabled, QPushButton:disabled {{
    color: {colors.outline};
    background-color: {colors.surface_container_low};
}}
QTabWidget[fancyRole="navigation"]::pane {{
    margin-left: 8px;
    background-color: {_rgba(colors.surface_container_low, mica_alpha)};
    border: 1px solid {colors.outline_variant};
    border-radius: {layout.radius_large_px}px;
}}
QTabWidget[fancyRole="navigation"] QStackedWidget {{ background: transparent; }}
""".strip()


if pyside6_fancy_available:
    _STANDARD_ICONS: dict[str, QStyle.StandardPixmap] = {
        "add": QStyle.StandardPixmap.SP_FileDialogNewFolder,
        "back": QStyle.StandardPixmap.SP_ArrowBack,
        "close": QStyle.StandardPixmap.SP_TitleBarCloseButton,
        "delete": QStyle.StandardPixmap.SP_TrashIcon,
        "down": QStyle.StandardPixmap.SP_ArrowDown,
        "error": QStyle.StandardPixmap.SP_MessageBoxCritical,
        "file": QStyle.StandardPixmap.SP_FileIcon,
        "folder": QStyle.StandardPixmap.SP_DirIcon,
        "forward": QStyle.StandardPixmap.SP_ArrowForward,
        "home": QStyle.StandardPixmap.SP_DirHomeIcon,
        "info": QStyle.StandardPixmap.SP_MessageBoxInformation,
        "mute": QStyle.StandardPixmap.SP_MediaVolumeMuted,
        "play": QStyle.StandardPixmap.SP_MediaPlay,
        "refresh": QStyle.StandardPixmap.SP_BrowserReload,
        "save": QStyle.StandardPixmap.SP_DialogSaveButton,
        "stop": QStyle.StandardPixmap.SP_MediaStop,
        "success": QStyle.StandardPixmap.SP_DialogApplyButton,
        "up": QStyle.StandardPixmap.SP_ArrowUp,
        "volume": QStyle.StandardPixmap.SP_MediaVolume,
        "warning": QStyle.StandardPixmap.SP_MessageBoxWarning,
    }

    def _tinted_icon(icon: QIcon, color: QColor, size: int) -> QIcon:
        source = icon.pixmap(QSize(size, size))
        if source.isNull():
            return icon
        result = QPixmap(source.size())
        result.fill(Qt.GlobalColor.transparent)
        painter = QPainter(result)
        painter.drawPixmap(0, 0, source)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(result.rect(), color)
        painter.end()
        return QIcon(result)

    def _fluent_glyph(name: str, color: QColor, size: int) -> QIcon:
        """用同一笔画系统绘制 FancyUI 导航/命令图标。"""

        canvas = QPixmap(size, size)
        canvas.fill(Qt.GlobalColor.transparent)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        unit = float(size) / 20.0
        pen = QPen(color, max(1.4, 1.55 * unit))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        def point(x: float, y: float) -> QPointF:
            return QPointF(x * unit, y * unit)

        if name == "home":
            painter.drawPolyline(QPolygonF([point(3, 9), point(10, 3), point(17, 9)]))
            painter.drawPolyline(
                QPolygonF([point(5, 8), point(5, 17), point(15, 17), point(15, 8)])
            )
            painter.drawLine(point(9, 17), point(9, 12))
            painter.drawLine(point(12, 12), point(12, 17))
        elif name in {"back", "forward", "up", "down"}:
            paths = {
                "back": ((14, 4), (7, 10), (14, 16)),
                "forward": ((6, 4), (13, 10), (6, 16)),
                "up": ((4, 13), (10, 7), (16, 13)),
                "down": ((4, 7), (10, 13), (16, 7)),
            }
            painter.drawPolyline(QPolygonF([point(*value) for value in paths[name]]))
        elif name in {"close", "error"}:
            if name == "error":
                painter.drawEllipse(QRectF(2.8 * unit, 2.8 * unit, 14.4 * unit, 14.4 * unit))
            painter.drawLine(point(6, 6), point(14, 14))
            painter.drawLine(point(14, 6), point(6, 14))
        elif name in {"add", "success"}:
            if name == "success":
                painter.drawEllipse(QRectF(2.8 * unit, 2.8 * unit, 14.4 * unit, 14.4 * unit))
                painter.drawPolyline(
                    QPolygonF([point(5.8, 10.2), point(8.7, 13), point(14.5, 6.8)])
                )
            else:
                painter.drawLine(point(10, 4), point(10, 16))
                painter.drawLine(point(4, 10), point(16, 10))
        elif name in {"info", "warning"}:
            if name == "info":
                painter.drawEllipse(QRectF(2.8 * unit, 2.8 * unit, 14.4 * unit, 14.4 * unit))
                painter.drawLine(point(10, 8.5), point(10, 14))
                painter.drawPoint(point(10, 6))
            else:
                painter.drawPolygon(QPolygonF([point(10, 2.8), point(18, 17), point(2, 17)]))
                painter.drawLine(point(10, 7), point(10, 12))
                painter.drawPoint(point(10, 14.5))
        elif name == "play":
            painter.setBrush(color)
            painter.drawPolygon(QPolygonF([point(6, 3.8), point(16, 10), point(6, 16.2)]))
        elif name == "stop":
            painter.setBrush(color)
            painter.drawRoundedRect(QRectF(5 * unit, 5 * unit, 10 * unit, 10 * unit), unit, unit)
        elif name == "refresh":
            path = QPainterPath()
            path.moveTo(point(15.8, 7.5))
            path.cubicTo(point(14.3, 3.7), point(9.6, 2.4), point(6.2, 4.8))
            path.cubicTo(point(2.8, 7.2), point(2.7, 12.2), point(6, 15))
            path.cubicTo(point(9, 17.6), point(13.8, 16.6), point(15.8, 13.2))
            painter.drawPath(path)
            painter.drawPolyline(QPolygonF([point(12.3, 7.4), point(16, 7.8), point(16.3, 4.1)]))
        elif name in {"file", "save"}:
            painter.drawRoundedRect(
                QRectF(4 * unit, 2.8 * unit, 12 * unit, 14.4 * unit), unit, unit
            )
            if name == "file":
                painter.drawLine(point(7, 7), point(13, 7))
                painter.drawLine(point(7, 10), point(13, 10))
                painter.drawLine(point(7, 13), point(11, 13))
            else:
                painter.drawRect(QRectF(7 * unit, 3 * unit, 6 * unit, 4 * unit))
                painter.drawRect(QRectF(6.5 * unit, 11 * unit, 7 * unit, 5 * unit))
        elif name == "folder":
            path = QPainterPath()
            path.moveTo(point(2.5, 6))
            path.lineTo(point(8.2, 6))
            path.lineTo(point(9.8, 8))
            path.lineTo(point(17.5, 8))
            path.lineTo(point(16, 16.5))
            path.lineTo(point(3.5, 16.5))
            path.closeSubpath()
            painter.drawPath(path)
        elif name == "delete":
            painter.drawLine(point(5, 6), point(15, 6))
            painter.drawLine(point(8, 3.5), point(12, 3.5))
            painter.drawRoundedRect(QRectF(6 * unit, 6 * unit, 8 * unit, 11 * unit), unit, unit)
            painter.drawLine(point(9, 9), point(9, 14))
            painter.drawLine(point(11, 9), point(11, 14))
        elif name in {"volume", "mute"}:
            painter.drawPolyline(
                QPolygonF(
                    [
                        point(3, 8),
                        point(6.5, 8),
                        point(11, 4.5),
                        point(11, 15.5),
                        point(6.5, 12),
                        point(3, 12),
                        point(3, 8),
                    ]
                )
            )
            if name == "volume":
                painter.drawArc(QRectF(9 * unit, 6 * unit, 7 * unit, 8 * unit), -60 * 16, 120 * 16)
            else:
                painter.drawLine(point(13, 7), point(17, 13))
                painter.drawLine(point(17, 7), point(13, 13))
        painter.end()
        return QIcon(canvas)

    def fancy_icon(
        name: str | QIcon | QStyle.StandardPixmap,
        *,
        theme: MD3Theme | None = None,
        size: int = 20,
        color: str | QColor | None = None,
    ) -> QIcon:
        """返回跨平台且可着色的图标；未知主题图标使用通用文件图标兜底。"""

        if isinstance(size, bool) or not isinstance(size, int) or size < 8 or size > 256:
            raise ValueError("size must be an integer between 8 and 256")
        if isinstance(name, QIcon):
            icon = QIcon(name)
        else:
            app = QApplication.instance()
            if app is None:
                raise RuntimeError("QApplication is required to create a FancyUI icon")
            if isinstance(name, QStyle.StandardPixmap):
                icon = app.style().standardIcon(name)
            else:
                normalized = str(name).strip().lower()
                if not normalized:
                    raise ValueError("icon name must not be empty")
                if normalized in FANCY_ICON_NAMES:
                    resolved_theme = theme or DARK_MD3_THEME
                    tint = (
                        QColor(color)
                        if color is not None
                        else QColor(resolved_theme.colors.on_surface_variant)
                    )
                    if not tint.isValid():
                        raise ValueError("color must be a valid Qt color or #RRGGBB string")
                    return _fluent_glyph(normalized, tint, size)
                standard = _STANDARD_ICONS.get(normalized)
                icon = (
                    app.style().standardIcon(standard)
                    if standard is not None
                    else QIcon.fromTheme(normalized)
                )
                if icon.isNull():
                    icon = app.style().standardIcon(QStyle.StandardPixmap.SP_FileIcon)
        resolved_theme = theme or DARK_MD3_THEME
        tint = (
            QColor(color) if color is not None else QColor(resolved_theme.colors.on_surface_variant)
        )
        if not tint.isValid():
            raise ValueError("color must be a valid Qt color or #RRGGBB string")
        return _tinted_icon(icon, tint, size)

    class _FancyStateMixin:
        _fancy_theme: MD3Theme
        _state_progress: float
        _state_animation: QVariantAnimation

        def _setup_state_animation(self, theme: MD3Theme) -> None:
            self._fancy_theme = theme
            self._state_progress = 0.0
            self._state_animation = QVariantAnimation(self)
            self._state_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._state_animation.valueChanged.connect(self._set_state_progress)

        def _set_state_progress(self, value: object) -> None:
            self._state_progress = max(0.0, min(1.0, float(value)))
            self.update()

        def _animate_state(self, target: float) -> None:
            self._state_animation.stop()
            duration = _duration(self._fancy_theme)
            if duration == 0:
                self._set_state_progress(target)
                return
            self._state_animation.setDuration(duration)
            self._state_animation.setStartValue(self._state_progress)
            self._state_animation.setEndValue(float(target))
            self._state_animation.start()

        def set_fancy_theme(self, theme: MD3Theme) -> None:
            if not isinstance(theme, MD3Theme):
                raise TypeError("theme must be an MD3Theme")
            self._fancy_theme = theme
            self.update()

    class FancyCard(_FancyStateMixin, QFrame):
        """带 Mica 层、标题区和 Fluent 状态过渡的内容卡片。"""

        clicked = Signal()

        def __init__(
            self,
            title: str = "",
            subtitle: str = "",
            *,
            icon: str | QIcon | QStyle.StandardPixmap | None = None,
            parent: QWidget | None = None,
            interactive: bool = False,
            elevated: bool = False,
            theme: MD3Theme | None = None,
        ) -> None:
            super().__init__(parent)
            self._setup_state_animation(theme or DARK_MD3_THEME)
            self._interactive = bool(interactive)
            self._elevated = bool(elevated)
            self._icon_source: str | QIcon | QStyle.StandardPixmap | None = None
            self.setProperty("fancyRole", "card")
            self.setProperty("fancyInteractive", self._interactive)
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.setMouseTracking(True)
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if self._interactive
                else Qt.CursorShape.ArrowCursor
            )
            self.setFocusPolicy(
                Qt.FocusPolicy.StrongFocus if self._interactive else Qt.FocusPolicy.NoFocus
            )

            self._root_layout = QVBoxLayout(self)
            self._root_layout.setContentsMargins(16, 14, 16, 16)
            self._root_layout.setSpacing(12)
            self._header = QWidget(self)
            header_layout = QHBoxLayout(self._header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            header_layout.setSpacing(12)
            self._icon_label = QLabel(self._header)
            self._icon_label.setFixedSize(24, 24)
            self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._icon_label.setVisible(icon is not None)
            header_layout.addWidget(self._icon_label)

            labels = QWidget(self._header)
            labels_layout = QVBoxLayout(labels)
            labels_layout.setContentsMargins(0, 0, 0, 0)
            labels_layout.setSpacing(2)
            self._title_label = QLabel(str(title), labels)
            self._title_label.setProperty("md3Role", "title")
            self._subtitle_label = QLabel(str(subtitle), labels)
            self._subtitle_label.setProperty("md3Role", "muted")
            self._subtitle_label.setWordWrap(True)
            labels_layout.addWidget(self._title_label)
            labels_layout.addWidget(self._subtitle_label)
            header_layout.addWidget(labels, 1)
            self._root_layout.addWidget(self._header)

            self._content = QWidget(self)
            self._content_layout = QVBoxLayout(self._content)
            self._content_layout.setContentsMargins(0, 0, 0, 0)
            self._content_layout.setSpacing(10)
            self._root_layout.addWidget(self._content)
            self.setTitle(title)
            self.setSubtitle(subtitle)
            if icon is not None:
                self.setIcon(icon)

        @property
        def content_layout(self) -> QVBoxLayout:
            return self._content_layout

        def addWidget(self, widget: QWidget, stretch: int = 0) -> None:  # noqa: N802
            self._content_layout.addWidget(widget, stretch)

        def addLayout(self, layout: Any, stretch: int = 0) -> None:  # noqa: N802
            self._content_layout.addLayout(layout, stretch)

        def setTitle(self, title: str) -> None:  # noqa: N802
            normalized = str(title)
            self._title_label.setText(normalized)
            self._title_label.setVisible(bool(normalized))
            self._sync_header_visibility()
            self.setAccessibleName(normalized or self._subtitle_label.text())

        def title(self) -> str:
            return self._title_label.text()

        def setSubtitle(self, subtitle: str) -> None:  # noqa: N802
            normalized = str(subtitle)
            self._subtitle_label.setText(normalized)
            self._subtitle_label.setVisible(bool(normalized))
            self._sync_header_visibility()

        def subtitle(self) -> str:
            return self._subtitle_label.text()

        def setIcon(self, icon: str | QIcon | QStyle.StandardPixmap) -> None:  # noqa: N802
            self._icon_source = icon
            resolved = fancy_icon(icon, theme=self._fancy_theme, size=20)
            self._icon_label.setPixmap(resolved.pixmap(QSize(20, 20)))
            self._icon_label.setVisible(True)
            self._sync_header_visibility()

        def setInteractive(self, interactive: bool) -> None:  # noqa: N802
            self._interactive = bool(interactive)
            self.setProperty("fancyInteractive", self._interactive)
            self.setCursor(
                Qt.CursorShape.PointingHandCursor
                if self._interactive
                else Qt.CursorShape.ArrowCursor
            )
            self.setFocusPolicy(
                Qt.FocusPolicy.StrongFocus if self._interactive else Qt.FocusPolicy.NoFocus
            )

        def isInteractive(self) -> bool:  # noqa: N802
            return self._interactive

        def setElevated(self, elevated: bool) -> None:  # noqa: N802
            self._elevated = bool(elevated)
            self.update()

        def set_fancy_theme(self, theme: MD3Theme) -> None:
            _FancyStateMixin.set_fancy_theme(self, theme)
            if self._icon_source is not None:
                self.setIcon(self._icon_source)

        def _sync_header_visibility(self) -> None:
            self._header.setVisible(
                self._icon_label.isVisible()
                or bool(self._title_label.text())
                or bool(self._subtitle_label.text())
            )

        def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802
            if self._interactive:
                self._animate_state(1.0)
            super().enterEvent(event)

        def leaveEvent(self, event: QEvent) -> None:  # noqa: N802
            self._animate_state(0.0)
            super().leaveEvent(event)

        def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
            inside = self.rect().contains(event.position().toPoint())
            if self._interactive and event.button() is Qt.MouseButton.LeftButton and inside:
                self.clicked.emit()
            super().mouseReleaseEvent(event)

        def paintEvent(self, event: Any) -> None:  # noqa: N802
            colors = self._fancy_theme.colors
            base = QColor(colors.surface_container)
            hover = QColor(colors.surface_container_high)
            red = round(base.red() + (hover.red() - base.red()) * self._state_progress)
            green = round(base.green() + (hover.green() - base.green()) * self._state_progress)
            blue = round(base.blue() + (hover.blue() - base.blue()) * self._state_progress)
            alpha = 246 if self._fancy_theme.mode is MD3ThemeMode.DARK else 250
            fill = QColor(red, green, blue, alpha)
            border = QColor(colors.primary if self.hasFocus() else colors.outline_variant)
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            bounds = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            painter.setBrush(fill)
            painter.setPen(QPen(border, 1.2 if self._elevated or self.hasFocus() else 1.0))
            painter.drawRoundedRect(
                bounds,
                self._fancy_theme.layout.radius_large_px,
                self._fancy_theme.layout.radius_large_px,
            )
            painter.end()
            super().paintEvent(event)

    class _FancyNavigationTabBar(QTabBar):
        def __init__(self, theme: MD3Theme, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self._theme = theme
            self._navigation_width = 188
            self._compact = False
            self._hovered_index = -1
            self.setMouseTracking(True)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.setElideMode(Qt.TextElideMode.ElideRight)
            self.setExpanding(False)

        def set_fancy_theme(self, theme: MD3Theme) -> None:
            self._theme = theme
            self.updateGeometry()
            self.update()

        def set_compact(self, compact: bool) -> None:
            self._compact = bool(compact)
            # QTabWidget 在父窗口 resizeEvent 中收到紧凑模式切换时，
            # 可能已经缓存了旧的 West tabbar 宽度。固定当前模式宽度
            # 可让正文区域立即释放/回收导航栏占用的空间。
            self.setFixedWidth(132 if self._compact else self._navigation_width)
            self.setToolTip("")
            self.updateGeometry()
            self.update()

        def set_navigation_width(self, width: int) -> None:
            if isinstance(width, bool) or not isinstance(width, int) or width < 128 or width > 420:
                raise ValueError("navigation width must be between 128 and 420")
            self._navigation_width = width
            self.setFixedWidth(132 if self._compact else width)
            self.updateGeometry()
            self.update()

        def tabSizeHint(self, index: int) -> QSize:  # noqa: N802
            del index
            # 紧凑导航仍保留短标签；只有图标会让 640px 控制台难以
            # 识别页面，也会把可见状态退化成悬停提示。
            return QSize(132 if self._compact else self._navigation_width, 44)

        def minimumTabSizeHint(self, index: int) -> QSize:  # noqa: N802
            return self.tabSizeHint(index)

        def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
            hovered = self.tabAt(event.position().toPoint())
            if hovered != self._hovered_index:
                self._hovered_index = hovered
                self.update()
            super().mouseMoveEvent(event)

        def leaveEvent(self, event: QEvent) -> None:  # noqa: N802
            self._hovered_index = -1
            self.update()
            super().leaveEvent(event)

        def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
            key = event.key()
            previous_keys = {
                Qt.Key.Key_Up,
                Qt.Key.Key_Left,
            }
            next_keys = {
                Qt.Key.Key_Down,
                Qt.Key.Key_Right,
            }
            navigation_keys = (
                previous_keys
                | next_keys
                | {
                    Qt.Key.Key_Home,
                    Qt.Key.Key_End,
                }
            )
            if key not in navigation_keys:
                super().keyPressEvent(event)
                return

            selectable = [
                index
                for index in range(self.count())
                if self.isTabVisible(index) and self.isTabEnabled(index)
            ]
            if not selectable:
                event.accept()
                return
            if key == Qt.Key.Key_Home:
                target = selectable[0]
            elif key == Qt.Key.Key_End:
                target = selectable[-1]
            else:
                current = self.currentIndex()
                if current not in selectable:
                    target = selectable[-1] if key in previous_keys else selectable[0]
                else:
                    position = selectable.index(current)
                    offset = -1 if key in previous_keys else 1
                    target = selectable[max(0, min(len(selectable) - 1, position + offset))]
            self.setCurrentIndex(target)
            event.accept()

        def paintEvent(self, event: Any) -> None:  # noqa: N802
            del event
            colors = self._theme.colors
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            for index in range(self.count()):
                rect = self.tabRect(index).adjusted(4, 3, -4, -3)
                selected = index == self.currentIndex()
                hovered = index == self._hovered_index
                if selected or hovered:
                    fill = QColor(
                        colors.secondary_container if selected else colors.surface_container
                    )
                    fill.setAlpha(230 if selected else 180)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(fill)
                    painter.drawRoundedRect(
                        QRectF(rect),
                        self._theme.layout.radius_small_px,
                        self._theme.layout.radius_small_px,
                    )
                if selected:
                    painter.setBrush(QColor(colors.primary))
                    indicator = QRectF(rect.left(), rect.top() + 10, 3, rect.height() - 20)
                    painter.drawRoundedRect(indicator, 1.5, 1.5)
                icon_size = 18
                icon_x = rect.left() + (10 if self._compact else 14)
                icon_y = rect.top() + (rect.height() - icon_size) // 2
                icon = self.tabIcon(index)
                if not icon.isNull():
                    icon.paint(painter, icon_x, icon_y, icon_size, icon_size)
                # 132px 紧凑导航仍显示完整中文标签，避免只靠 tooltip 才能知道
                # 当前页面；极窄页面由外层选择器接管，不会挤压正文。
                text_rect = rect.adjusted(
                    38 if self._compact else 44, 0, -8 if self._compact else -10, 0
                )
                text = self.fontMetrics().elidedText(
                    self.tabText(index), Qt.TextElideMode.ElideRight, text_rect.width()
                )
                painter.setPen(QColor(colors.on_surface if selected else colors.on_surface_variant))
                painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, text)
                if self.hasFocus() and selected:
                    pen = QPen(QColor(colors.primary), max(1, self._theme.layout.focus_width_px))
                    painter.setPen(pen)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRoundedRect(
                        QRectF(rect).adjusted(1, 1, -1, -1),
                        self._theme.layout.radius_small_px,
                        self._theme.layout.radius_small_px,
                    )
            painter.end()

    class FancyNavigationView(QTabWidget):
        """WinUI 左侧导航视图；页面仍使用 QTabWidget 的稳定状态模型。"""

        compactChanged = Signal(bool)

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            theme: MD3Theme | None = None,
            compact: bool = False,
            navigation_width: int = 188,
        ) -> None:
            super().__init__(parent)
            self._fancy_theme = theme or DARK_MD3_THEME
            self._navigation_bar = _FancyNavigationTabBar(self._fancy_theme, self)
            self.setTabBar(self._navigation_bar)
            self.setTabPosition(QTabWidget.TabPosition.West)
            self.setDocumentMode(True)
            self.setMovable(False)
            self.setTabsClosable(False)
            self.setUsesScrollButtons(True)
            self.setProperty("fancyRole", "navigation")
            self.setNavigationWidth(navigation_width)
            self.setCompact(compact)

        def addPage(
            self,
            widget: QWidget,
            title: str,
            icon: str | QIcon | QStyle.StandardPixmap | None = None,
            tooltip: str = "",
        ) -> int:  # noqa: N802
            resolved_icon = fancy_icon(icon or "file", theme=self._fancy_theme)
            index = self.addTab(widget, resolved_icon, str(title))
            if tooltip:
                self.setTabToolTip(index, str(tooltip))
            elif self.isCompact():
                self.setTabToolTip(index, str(title))
            return index

        def insertPage(
            self,
            index: int,
            widget: QWidget,
            title: str,
            icon: str | QIcon | QStyle.StandardPixmap | None = None,
            tooltip: str = "",
        ) -> int:  # noqa: N802
            resolved_icon = fancy_icon(icon or "file", theme=self._fancy_theme)
            actual = self.insertTab(index, widget, resolved_icon, str(title))
            if tooltip:
                self.setTabToolTip(actual, str(tooltip))
            elif self.isCompact():
                self.setTabToolTip(actual, str(title))
            return actual

        def setCompact(self, compact: bool) -> None:  # noqa: N802
            normalized = bool(compact)
            if normalized == self._navigation_bar._compact:
                return
            self._navigation_bar.set_compact(normalized)
            for index in range(self.count()):
                self.setTabToolTip(index, self.tabText(index) if normalized else "")
            self.compactChanged.emit(normalized)

        def isCompact(self) -> bool:  # noqa: N802
            return self._navigation_bar._compact

        def setNavigationWidth(self, width: int) -> None:  # noqa: N802
            self._navigation_bar.set_navigation_width(width)

        def navigationWidth(self) -> int:  # noqa: N802
            return self._navigation_bar._navigation_width

        def set_fancy_theme(self, theme: MD3Theme) -> None:
            if not isinstance(theme, MD3Theme):
                raise TypeError("theme must be an MD3Theme")
            self._fancy_theme = theme
            self._navigation_bar.set_fancy_theme(theme)
            tint = QColor(theme.colors.on_surface_variant)
            for index in range(self.count()):
                current = self.tabIcon(index)
                if not current.isNull():
                    self.setTabIcon(index, _tinted_icon(current, tint, 20))

    class FancyInfoBar(QFrame):
        """可关闭、可自动消失并支持动作按钮的 Fluent 信息条。"""

        dismissed = Signal()
        actionTriggered = Signal()
        _SEVERITIES = frozenset({"info", "success", "warning", "error"})

        def __init__(
            self,
            title: str = "",
            message: str = "",
            *,
            severity: str = "info",
            parent: QWidget | None = None,
            closable: bool = True,
            duration_ms: int = 0,
            theme: MD3Theme | None = None,
        ) -> None:
            super().__init__(parent)
            self._fancy_theme = theme or DARK_MD3_THEME
            self._action_callback: Callable[[], Any] | None = None
            self._duration_ms = 0
            self._closable = bool(closable)
            self.setProperty("fancyRole", "info-bar")
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

            layout = QHBoxLayout(self)
            layout.setContentsMargins(12, 8, 8, 8)
            layout.setSpacing(8)
            self._icon_label = QLabel(self)
            self._icon_label.setFixedSize(20, 20)
            layout.addWidget(self._icon_label, 0, Qt.AlignmentFlag.AlignTop)
            text_widget = QWidget(self)
            text_layout = QVBoxLayout(text_widget)
            text_layout.setContentsMargins(0, 0, 0, 0)
            text_layout.setSpacing(2)
            self._title_label = QLabel(str(title), text_widget)
            self._title_label.setProperty("fancyRole", "info-title")
            self._message_label = QLabel(str(message), text_widget)
            self._message_label.setProperty("fancyRole", "info-message")
            self._message_label.setWordWrap(True)
            text_layout.addWidget(self._title_label)
            text_layout.addWidget(self._message_label)
            layout.addWidget(text_widget, 1)
            self._action_button = QPushButton(self)
            self._action_button.setProperty("fancyRole", "info-action")
            self._action_button.hide()
            self._action_button.clicked.connect(self._trigger_action)
            layout.addWidget(self._action_button)
            self._close_button = QToolButton(self)
            self._close_button.setProperty("fancyRole", "icon-button")
            self._close_button.setAutoRaise(True)
            self._close_button.setToolTip("关闭")
            self._close_button.setAccessibleName("关闭信息")
            self._close_button.clicked.connect(self.dismiss)
            layout.addWidget(self._close_button, 0, Qt.AlignmentFlag.AlignTop)

            self._timer = QTimer(self)
            self._timer.setSingleShot(True)
            self._timer.timeout.connect(self.dismiss)
            self._opacity = QGraphicsOpacityEffect(self)
            self._opacity.setOpacity(1.0)
            self.setGraphicsEffect(self._opacity)
            self._fade = QVariantAnimation(self)
            self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._fade.valueChanged.connect(lambda value: self._opacity.setOpacity(float(value)))
            self._fade.finished.connect(self._finish_dismiss)
            self.setSeverity(severity)
            self.setTitle(title)
            self.setMessage(message)
            self.setClosable(closable)
            self.setDuration(duration_ms)
            self.set_fancy_theme(self._fancy_theme)

        def setTitle(self, title: str) -> None:  # noqa: N802
            normalized = str(title)
            self._title_label.setText(normalized)
            self._title_label.setVisible(bool(normalized))
            self.setAccessibleName(normalized or self._message_label.text())

        def title(self) -> str:
            return self._title_label.text()

        def setMessage(self, message: str) -> None:  # noqa: N802
            normalized = str(message)
            self._message_label.setText(normalized)
            self._message_label.setVisible(bool(normalized))

        def message(self) -> str:
            return self._message_label.text()

        def setSeverity(self, severity: str) -> None:  # noqa: N802
            normalized = str(severity).strip().lower()
            if normalized not in self._SEVERITIES:
                raise ValueError("severity must be info, success, warning, or error")
            self.setProperty("fancySeverity", normalized)
            colors = self._fancy_theme.colors
            status_color = {
                "info": colors.primary,
                "success": colors.success,
                "warning": colors.warning,
                "error": colors.error,
            }[normalized]
            self._icon_label.setPixmap(
                fancy_icon(
                    normalized,
                    theme=self._fancy_theme,
                    size=18,
                    color=status_color,
                ).pixmap(QSize(18, 18))
            )
            self.style().unpolish(self)
            self.style().polish(self)
            self.update()

        def severity(self) -> str:
            return str(self.property("fancySeverity"))

        def setAction(
            self,
            text: str,
            callback: Callable[[], Any] | None = None,
        ) -> None:  # noqa: N802
            normalized = str(text).strip()
            if callback is not None and not callable(callback):
                raise TypeError("callback must be callable")
            self._action_callback = callback
            self._action_button.setText(normalized)
            self._action_button.setVisible(bool(normalized))

        def clearAction(self) -> None:  # noqa: N802
            self.setAction("")

        def setClosable(self, closable: bool) -> None:  # noqa: N802
            self._closable = bool(closable)
            self._close_button.setVisible(self._closable)

        def isClosable(self) -> bool:  # noqa: N802
            return self._closable

        def setDuration(self, duration_ms: int) -> None:  # noqa: N802
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, int) or duration_ms < 0:
                raise ValueError("duration_ms must be a non-negative integer")
            self._duration_ms = duration_ms
            if self.isVisible():
                self._restart_timer()

        def duration(self) -> int:
            return self._duration_ms

        def reveal(self) -> None:
            self._fade.stop()
            self._opacity.setOpacity(1.0)
            self.show()
            self._restart_timer()

        def dismiss(self) -> None:
            self._timer.stop()
            duration = _duration(self._fancy_theme)
            if duration == 0 or not self.isVisible():
                self._finish_dismiss()
                return
            self._fade.stop()
            self._fade.setDuration(duration)
            self._fade.setStartValue(self._opacity.opacity())
            self._fade.setEndValue(0.0)
            self._fade.start()

        def set_fancy_theme(self, theme: MD3Theme) -> None:
            if not isinstance(theme, MD3Theme):
                raise TypeError("theme must be an MD3Theme")
            self._fancy_theme = theme
            self._close_button.setIcon(fancy_icon("close", theme=theme, size=16))
            self.setSeverity(self.severity())

        def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
            self._restart_timer()
            super().showEvent(event)

        def _restart_timer(self) -> None:
            self._timer.stop()
            if self._duration_ms > 0 and self.isVisible():
                self._timer.start(self._duration_ms)

        def _trigger_action(self) -> None:
            self.actionTriggered.emit()
            if self._action_callback is not None:
                self._action_callback()

        def _finish_dismiss(self) -> None:
            was_visible = self.isVisible()
            self.hide()
            self._opacity.setOpacity(1.0)
            if was_visible:
                self.dismissed.emit()

    class FancyCommandBar(QFrame):
        """带主命令、次命令和窄宽度溢出菜单的 WinUI 命令栏。"""

        commandTriggered = Signal(QAction)

        def __init__(
            self,
            parent: QWidget | None = None,
            *,
            theme: MD3Theme | None = None,
            compact: bool = False,
        ) -> None:
            super().__init__(parent)
            self._fancy_theme = theme or DARK_MD3_THEME
            self._commands: list[tuple[QAction, QToolButton, bool]] = []
            self._compact = bool(compact)
            self.setProperty("fancyRole", "command-bar")
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self._layout = QHBoxLayout(self)
            self._layout.setContentsMargins(8, 6, 8, 6)
            self._layout.setSpacing(4)
            self._layout.addStretch(1)
            self._overflow_menu = QMenu(self)
            self._overflow_button = QToolButton(self)
            self._overflow_button.setProperty("fancyRole", "icon-button")
            self._overflow_button.setText("⋯")
            self._overflow_button.setToolTip("更多命令")
            self._overflow_button.setAccessibleName("更多命令")
            self._overflow_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            self._overflow_button.setMenu(self._overflow_menu)
            self._overflow_button.hide()
            self._layout.addWidget(self._overflow_button)

        def addCommand(
            self,
            action_or_text: QAction | str,
            callback: Callable[[], Any] | None = None,
            *,
            icon: str | QIcon | QStyle.StandardPixmap | None = None,
            tooltip: str = "",
            checkable: bool = False,
            primary: bool = False,
        ) -> QAction:  # noqa: N802
            if isinstance(action_or_text, QAction):
                if callback is not None or icon is not None:
                    raise ValueError("callback and icon cannot override an existing QAction")
                action = action_or_text
                if action.parent() is None:
                    action.setParent(self)
            else:
                action = QAction(str(action_or_text), self)
                if icon is not None:
                    icon_color = (
                        self._fancy_theme.colors.on_primary
                        if primary
                        else self._fancy_theme.colors.on_surface_variant
                    )
                    action.setIcon(fancy_icon(icon, theme=self._fancy_theme, color=icon_color))
                action.setCheckable(bool(checkable))
                if callback is not None:
                    if not callable(callback):
                        raise TypeError("callback must be callable")
                    action.triggered.connect(callback)
            if tooltip:
                action.setToolTip(str(tooltip))
            action.triggered.connect(
                lambda _checked=False, current=action: self.commandTriggered.emit(current)
            )
            button = QToolButton(self)
            button.setDefaultAction(action)
            button.setAccessibleName(action.text())
            if action.toolTip():
                button.setToolTip(action.toolTip())
            button.setProperty("fancyRole", "command-primary" if primary else "command")
            button.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonIconOnly
                if self._compact
                else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            insert_at = max(0, self._layout.count() - 2)
            self._layout.insertWidget(insert_at, button)
            super().addAction(action)
            self._commands.append((action, button, bool(primary)))
            self._sync_overflow()
            return action

        def addSeparator(self) -> QFrame:  # noqa: N802
            separator = QFrame(self)
            separator.setFrameShape(QFrame.Shape.VLine)
            separator.setProperty("fancyRole", "command-separator")
            insert_at = max(0, self._layout.count() - 2)
            self._layout.insertWidget(insert_at, separator)
            return separator

        def clearCommands(self) -> None:  # noqa: N802
            for action, button, _primary in self._commands:
                self.removeAction(action)
                button.deleteLater()
                if action.parent() is self:
                    action.deleteLater()
            self._commands.clear()
            self._overflow_menu.clear()
            self._overflow_button.hide()

        def commands(self) -> tuple[QAction, ...]:
            return tuple(item[0] for item in self._commands)

        def setCompact(self, compact: bool) -> None:  # noqa: N802
            self._compact = bool(compact)
            style = (
                Qt.ToolButtonStyle.ToolButtonIconOnly
                if self._compact
                else Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            for _action, button, _primary in self._commands:
                button.setToolButtonStyle(style)
            self._sync_overflow()

        def isCompact(self) -> bool:  # noqa: N802
            return self._compact

        def set_fancy_theme(self, theme: MD3Theme) -> None:
            if not isinstance(theme, MD3Theme):
                raise TypeError("theme must be an MD3Theme")
            self._fancy_theme = theme
            for action, _button, primary in self._commands:
                current = action.icon()
                if not current.isNull():
                    tint = QColor(
                        theme.colors.on_primary if primary else theme.colors.on_surface_variant
                    )
                    action.setIcon(_tinted_icon(current, tint, 20))
            self._sync_overflow()

        def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
            super().resizeEvent(event)
            self._sync_overflow()

        def _sync_overflow(self) -> None:
            if not self._commands:
                self._overflow_button.hide()
                return
            required = 24 + sum(button.sizeHint().width() + 4 for _, button, _ in self._commands)
            overflow = self.width() > 0 and required > self.width()
            self._overflow_menu.clear()
            has_overflow = False
            for action, button, primary in self._commands:
                hidden = overflow and not primary
                button.setVisible(not hidden)
                if hidden:
                    self._overflow_menu.addAction(action)
                    has_overflow = True
            self._overflow_button.setVisible(has_overflow)

    class FancyStyleController(QWidget):
        """集中应用 MD3 令牌、FancyUI 样式和降低动态效果偏好。"""

        themeChanged = Signal(MD3Theme)
        reducedMotionChanged = Signal(bool)

        def __init__(self, theme: MD3Theme | None = None, parent: QWidget | None = None) -> None:
            super().__init__(parent)
            self.hide()
            self._theme = theme or DARK_MD3_THEME
            self._target: QWidget | QApplication | None = None
            self._widgets: weakref.WeakSet[Any] = weakref.WeakSet()
            self._revision = 0
            self._extra_stylesheet = ""
            self._last_applied_stylesheet = ""

        @property
        def theme(self) -> MD3Theme:
            return self._theme

        @property
        def revision(self) -> int:
            return self._revision

        @property
        def reduced_motion(self) -> bool:
            return self._theme.motion.reduced_motion

        def motionDuration(self, preferred: int | None = None) -> int:  # noqa: N802
            return _duration(self._theme, preferred)

        def attach(
            self,
            target: QWidget | QApplication,
            *,
            extra_stylesheet: str = "",
        ) -> str:
            """绑定单个主题根节点，并一次性合并页面专属规则。

            ``extra_stylesheet`` 与共享 MD3/FancyUI 规则在同一次 Qt 样式刷新
            中应用，避免先应用共享样式、随后覆盖页面样式造成整棵控件树被
            重复刷新。
            """

            if not callable(getattr(target, "setStyleSheet", None)):
                raise TypeError("style target must provide setStyleSheet")
            if not isinstance(extra_stylesheet, str):
                raise TypeError("extra_stylesheet must be a string")
            self._target = target
            self._extra_stylesheet = extra_stylesheet
            target.setProperty("fancyRoot", True)
            return self._apply_target(target)

        def register(self, widget: QWidget) -> QWidget:
            if not isinstance(widget, QWidget):
                raise TypeError("widget must be a QWidget")
            self._widgets.add(widget)
            setter = getattr(widget, "set_fancy_theme", None)
            if callable(setter):
                setter(self._theme)
            return widget

        def unregister(self, widget: QWidget) -> None:
            self._widgets.discard(widget)

        def updateTheme(  # noqa: N802
            self,
            theme: MD3Theme | Mapping[str, Any],
            *,
            extra_stylesheet: str | None = None,
        ) -> str:
            resolved = theme if isinstance(theme, MD3Theme) else theme_from_mapping(theme)
            if extra_stylesheet is not None and not isinstance(extra_stylesheet, str):
                raise TypeError("extra_stylesheet must be a string")
            normalized_extra = (
                self._extra_stylesheet if extra_stylesheet is None else extra_stylesheet
            )
            theme_changed = resolved != self._theme
            stylesheet_changed = normalized_extra != self._extra_stylesheet
            if not theme_changed and not stylesheet_changed:
                if self._target is None:
                    return self._composed_stylesheet()
                return self._apply_target(self._target)
            self._theme = resolved
            self._extra_stylesheet = normalized_extra
            self._revision += 1
            if theme_changed:
                for widget in tuple(self._widgets):
                    setter = getattr(widget, "set_fancy_theme", None)
                    if callable(setter):
                        setter(resolved)
            stylesheet = self._composed_stylesheet()
            if self._target is not None:
                stylesheet = self._apply_target(self._target)
            if theme_changed:
                self.themeChanged.emit(resolved)
            return stylesheet

        def setReducedMotion(self, reduced: bool) -> str:  # noqa: N802
            normalized = bool(reduced)
            if normalized == self.reduced_motion:
                return self.stylesheet()
            updated = replace(
                self._theme,
                motion=replace(self._theme.motion, reduced_motion=normalized),
            )
            stylesheet = self.updateTheme(updated)
            self.reducedMotionChanged.emit(normalized)
            return stylesheet

        def stylesheet(self) -> str:
            return "\n\n".join(
                (build_md3_stylesheet(self._theme), build_fancy_stylesheet(self._theme))
            )

        def _composed_stylesheet(self) -> str:
            parts = (self.stylesheet(), self._extra_stylesheet)
            return "\n\n".join(part for part in parts if part)

        def _apply_target(self, target: QWidget | QApplication) -> str:
            stylesheet = self._composed_stylesheet()
            if stylesheet != self._last_applied_stylesheet or target.styleSheet() != stylesheet:
                target.setStyleSheet(stylesheet)
                self._last_applied_stylesheet = stylesheet
                target.setPalette(build_qt_palette(self._theme))
            return stylesheet


else:  # pragma: no cover - 仅供未安装 desktop extra 时给出明确错误。

    def fancy_icon(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("PySide6 is required to create a FancyUI icon")

    class _UnavailableFancyWidget:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("PySide6 is required to create FancyUI widgets")

    FancyCard = _UnavailableFancyWidget
    FancyNavigationView = _UnavailableFancyWidget
    FancyInfoBar = _UnavailableFancyWidget
    FancyCommandBar = _UnavailableFancyWidget
    FancyStyleController = _UnavailableFancyWidget


__all__ = [
    "FANCY_ICON_NAMES",
    "FancyCard",
    "FancyCommandBar",
    "FancyInfoBar",
    "FancyNavigationView",
    "FancyStyleController",
    "build_fancy_stylesheet",
    "fancy_icon",
    "pyside6_fancy_available",
]
