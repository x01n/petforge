from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

try:
    from PySide6.QtCore import Qt, QTimer, Signal
    from PySide6.QtWidgets import (
        QComboBox,
        QFormLayout,
        QHBoxLayout,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )

    pyside6_available = True
except (ImportError, ModuleNotFoundError, OSError, RuntimeError):  # pragma: no cover
    pyside6_available = False

from gui.qt6.fancyui import FancyCard, FancyInfoBar, FancyStyleController
from gui.qt6.md3 import DARK_MD3_THEME, MD3Theme

Callback = Callable[..., Any]


def _safe_count(value: object) -> int:
    """把诊断计数限制为非负有限整数，坏数据不应隐藏整个来源列表。"""

    if isinstance(value, bool):
        return 0
    try:
        count = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(count, 1_000_000))


def _safe_text(value: object, maximum: int) -> str:
    """把外部诊断文本限制为可展示字符串，单项异常不影响其它来源。"""

    try:
        rendered = str(value or "").replace("\x00", " ")
    except Exception:
        return ""
    return " ".join(rendered.split())[:maximum]


def _safe_servers(value: object) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        return ()
    result: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _safe_text(item.get("name", ""), 128).strip()
        if not name:
            continue
        result.append(
            {
                "name": name[:128],
                "source": _safe_text(item.get("source", ""), 80),
                "status": _safe_text(item.get("status", ""), 32),
                "resource_count": _safe_count(item.get("resource_count", 0)),
                "prompt_count": _safe_count(item.get("prompt_count", 0)),
            }
        )
    return tuple(result)


if pyside6_available:

    class MCPContentWindow(QWidget):
        """以显式用户操作浏览一个 MCP 来源的资源或 prompt。"""

        closed = Signal()

        def __init__(
            self,
            *,
            servers_provider: Callback,
            action_provider: Callback,
            theme: MD3Theme = DARK_MD3_THEME,
            parent: QWidget | None = None,
        ) -> None:
            super().__init__(parent, Qt.WindowType.Window)
            if not callable(servers_provider) or not callable(action_provider):
                raise TypeError("MCP content providers must be callable")
            self._servers_provider = servers_provider
            self._action_provider = action_provider
            self._theme = theme if isinstance(theme, MD3Theme) else DARK_MD3_THEME
            self._style_controller = FancyStyleController(self._theme, self)
            self._pending: object | None = None
            self._pending_approval_id = ""
            self._pending_approval_catalog = False
            self._generation = 0
            self._poll_timer = QTimer(self)
            self._poll_timer.setInterval(40)
            self._poll_timer.timeout.connect(self._poll_pending)
            self._build_ui()
            self._style_controller.attach(self)
            self.refresh_servers()

        def _build_ui(self) -> None:
            self.setObjectName("mcpContentWindow")
            self.setWindowTitle("MCP 内容浏览")
            self.setMinimumSize(620, 520)
            root = QVBoxLayout(self)
            root.setContentsMargins(16, 16, 16, 16)
            root.setSpacing(10)

            header = FancyCard(
                "MCP 内容",
                "只显示你主动选择的外部资源和提示模板；正文按不可信数据处理。",
                icon="forward",
                theme=self._theme,
            )
            self._style_controller.register(header)
            root.addWidget(header)

            form_card = FancyCard("来源与类型", theme=self._theme)
            self._style_controller.register(form_card)
            form = QFormLayout()
            form_card.addLayout(form)
            self.server_combo = QComboBox()
            self.server_combo.setObjectName("mcpContentServer")
            self.server_combo.currentIndexChanged.connect(lambda _index: self.refresh_catalog())
            form.addRow("MCP 来源", self.server_combo)
            self.kind_combo = QComboBox()
            self.kind_combo.setObjectName("mcpContentKind")
            self.kind_combo.addItem("资源", "resources")
            self.kind_combo.addItem("提示模板", "prompts")
            self.kind_combo.currentIndexChanged.connect(lambda _index: self.refresh_catalog())
            form.addRow("内容类型", self.kind_combo)
            actions = QHBoxLayout()
            self.refresh_button = QPushButton("刷新目录")
            self.refresh_button.clicked.connect(self.refresh_catalog)
            actions.addWidget(self.refresh_button)
            self.load_button = QPushButton("加载选中内容")
            self.load_button.clicked.connect(self.load_selected)
            actions.addWidget(self.load_button)
            self.approve_button = QPushButton("批准读取")
            self.approve_button.setObjectName("mcpContentApprove")
            self.approve_button.setProperty("role", "primary")
            self.approve_button.setToolTip("批准当前一次 MCP 内容读取")
            self.approve_button.clicked.connect(self.approve_pending)
            self.approve_button.setVisible(False)
            actions.addWidget(self.approve_button)
            self.deny_button = QPushButton("拒绝")
            self.deny_button.setObjectName("mcpContentDeny")
            self.deny_button.setProperty("role", "danger")
            self.deny_button.setToolTip("拒绝当前 MCP 内容读取")
            self.deny_button.clicked.connect(self.deny_pending)
            self.deny_button.setVisible(False)
            actions.addWidget(self.deny_button)
            actions.addStretch(1)
            form.addRow("", actions)
            root.addWidget(form_card)

            body = QHBoxLayout()
            body.setSpacing(10)
            list_card = FancyCard("目录", theme=self._theme)
            self._style_controller.register(list_card)
            list_layout = list_card.content_layout
            self.item_list = QListWidget()
            self.item_list.setObjectName("mcpContentItems")
            self.item_list.currentItemChanged.connect(self._item_changed)
            list_layout.addWidget(self.item_list)
            body.addWidget(list_card, 1)

            detail_card = FancyCard("选择与结果", theme=self._theme)
            self._style_controller.register(detail_card)
            detail_layout = detail_card.content_layout
            self.selection_edit = QLineEdit()
            self.selection_edit.setObjectName("mcpContentSelection")
            self.selection_edit.setPlaceholderText("选择资源 URI 或 prompt 名称")
            detail_layout.addWidget(self.selection_edit)
            self.arguments_edit = QPlainTextEdit()
            self.arguments_edit.setObjectName("mcpPromptArguments")
            self.arguments_edit.setPlaceholderText('prompt 参数 JSON，例如 {"topic":"今天的任务"}')
            self.arguments_edit.setMaximumHeight(90)
            detail_layout.addWidget(self.arguments_edit)
            self.output_edit = QPlainTextEdit()
            self.output_edit.setObjectName("mcpContentOutput")
            self.output_edit.setReadOnly(True)
            self.output_edit.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            detail_layout.addWidget(self.output_edit, 1)
            body.addWidget(detail_card, 2)
            root.addLayout(body, 1)

            self.info_bar = FancyInfoBar(
                "状态",
                "等待选择 MCP 来源。",
                severity="info",
                closable=False,
                theme=self._theme,
            )
            self._style_controller.register(self.info_bar)
            root.addWidget(self.info_bar)

        def _server_name(self) -> str:
            value = self.server_combo.currentData()
            return str(value or "").strip()

        def _set_busy(self, busy: bool) -> None:
            self.refresh_button.setEnabled(not busy)
            self.load_button.setEnabled(not busy)
            self.approve_button.setEnabled(not busy and bool(self._pending_approval_id))
            self.deny_button.setEnabled(not busy and bool(self._pending_approval_id))
            self.server_combo.setEnabled(not busy)
            self.kind_combo.setEnabled(not busy)
            self.setProperty("busy", bool(busy))
            self.style().unpolish(self)
            self.style().polish(self)

        def _set_info(self, message: str, severity: str = "info") -> None:
            self.info_bar.setSeverity(severity)
            self.info_bar.setMessage(str(message or ""))

        def refresh_servers(self) -> None:
            try:
                values = _safe_servers(self._servers_provider())
            except Exception:
                values = ()
            current = self._server_name()
            self.server_combo.blockSignals(True)
            self.server_combo.clear()
            for item in values:
                label = str(item["name"])
                status = str(item.get("status", "") or "")
                self.server_combo.addItem(
                    f"{label} · {status}" if status else label,
                    label,
                )
            restored = self.server_combo.findData(current)
            self.server_combo.setCurrentIndex(restored if restored >= 0 else (0 if values else -1))
            self.server_combo.blockSignals(False)
            if not values:
                self._set_info("没有已连接的 MCP 内容来源。", "warning")

        def refresh_catalog(self) -> None:
            self.refresh_servers()
            server = self._server_name()
            if not server:
                self.item_list.clear()
                return
            kind = str(self.kind_combo.currentData() or "resources")
            operation = "list_resources" if kind == "resources" else "list_prompts"
            self._submit({"operation": operation, "server_name": server}, catalog=True)

        def _submit(self, payload: Mapping[str, object], *, catalog: bool = False) -> None:
            if str(payload.get("operation", "") or "") not in {"approve", "deny"}:
                self._cancel_pending_approval()
            self._cancel_pending_request()
            self._generation += 1
            generation = self._generation
            self._set_busy(True)
            self._set_info("正在读取 MCP 内容…", "info")
            try:
                result = self._action_provider(dict(payload))
            except Exception:
                self._set_busy(False)
                self._set_info("MCP 内容请求未提交。", "error")
                return
            done = getattr(result, "done", None)
            if callable(done):
                self._pending = (generation, result, catalog)
                self._poll_timer.start()
                return
            self._finish_result(generation, result, catalog=catalog)

        def _cancel_pending_request(self) -> None:
            """切换目录/批准操作前取消上一项未完成请求。"""

            pending = self._pending
            if not isinstance(pending, tuple) or len(pending) != 3:
                self._pending = None
                self._poll_timer.stop()
                return
            cancel = getattr(pending[1], "cancel", None)
            if callable(cancel):
                try:
                    cancel()
                except (RuntimeError, TypeError, ValueError):
                    pass
            self._pending = None
            self._poll_timer.stop()

        def _cancel_pending_approval(self) -> None:
            """用户改选目录时消费旧审批，避免服务端留下不可见请求。"""

            approval_id = self._pending_approval_id
            if not approval_id:
                return
            try:
                result = self._action_provider({"operation": "deny", "approval_id": approval_id})
                done = getattr(result, "done", None)
                if callable(done) and not done():
                    # 拒绝请求本身由 RuntimeLoop 继续完成；窗口不再等待它。
                    pass
            except Exception:
                pass
            self._clear_pending_approval()

        def _poll_pending(self) -> None:
            pending = self._pending
            if not isinstance(pending, tuple) or len(pending) != 3:
                self._poll_timer.stop()
                return
            generation, future, catalog = pending
            if generation != self._generation:
                return
            done = getattr(future, "done", None)
            if not callable(done) or not done():
                return
            self._pending = None
            self._poll_timer.stop()
            try:
                value = future.result()
            except Exception:
                value = {"status": "unavailable"}
            self._finish_result(generation, value, catalog=bool(catalog))

        def _finish_result(self, generation: int, result: object, *, catalog: bool) -> None:
            if generation != self._generation:
                return
            self._set_busy(False)
            if not isinstance(result, Mapping):
                result = {"status": "unavailable"}
            status = str(result.get("status", "") or "").strip().lower()
            if status == "approval_required":
                approval_id = str(result.get("approval_id", "") or "").strip()
                if not approval_id or len(approval_id) > 256:
                    self._pending_approval_id = ""
                    self.approve_button.setVisible(False)
                    self.deny_button.setVisible(False)
                    self._set_info("MCP 内容请求缺少有效审批标识。", "error")
                    return
                self._pending_approval_id = approval_id
                self._pending_approval_catalog = bool(catalog)
                self.approve_button.setVisible(True)
                self.deny_button.setVisible(True)
                self.output_edit.setPlainText(
                    str(result.get("safe_summary", "需要确认 MCP 内容读取") or "")[:4_000]
                )
                self._set_info("读取外部 MCP 内容需要确认。", "warning")
                self._set_busy(False)
                return
            if status not in {"completed", "available", "ok", "success"}:
                self._clear_pending_approval()
                self._set_info("MCP 内容读取失败，可检查来源状态后重试。", "error")
                return
            self._clear_pending_approval()
            if catalog:
                kind = str(self.kind_combo.currentData() or "resources")
                values = result.get("resources" if kind == "resources" else "prompts", ())
                self.item_list.clear()
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                    for value in values:
                        if not isinstance(value, Mapping):
                            continue
                        key = value.get("uri") if kind == "resources" else value.get("name")
                        if not isinstance(key, str) or not key:
                            continue
                        title = str(value.get("title") or value.get("description") or key)
                        item = QListWidgetItem(f"{title[:100]}\n{key}")
                        item.setData(Qt.ItemDataRole.UserRole, key)
                        self.item_list.addItem(item)
                self._set_info(f"目录已加载：{self.item_list.count()} 项。", "success")
            else:
                rendered = json.dumps(dict(result), ensure_ascii=False, indent=2, default=str)
                self.output_edit.setPlainText(rendered[:400_000])
                self._set_info("内容已加载；外部正文仅作为数据展示。", "success")

        def _clear_pending_approval(self) -> None:
            self._pending_approval_id = ""
            self._pending_approval_catalog = False
            self.approve_button.setVisible(False)
            self.deny_button.setVisible(False)

        def _item_changed(
            self,
            item: QListWidgetItem | None,
            _previous: QListWidgetItem | None,
        ) -> None:
            if item is not None:
                self.selection_edit.setText(str(item.data(Qt.ItemDataRole.UserRole) or ""))

        def load_selected(self) -> None:
            server = self._server_name()
            selection = self.selection_edit.text().strip()
            kind = str(self.kind_combo.currentData() or "resources")
            if not server or not selection:
                self._set_info("请先选择来源和目录项。", "warning")
                return
            if kind == "resources":
                payload = {
                    "operation": "read_resource",
                    "server_name": server,
                    "uri": selection,
                }
            else:
                raw_arguments = self.arguments_edit.toPlainText().strip()
                arguments: object = None
                if raw_arguments:
                    try:
                        arguments = json.loads(raw_arguments)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        self._set_info("prompt 参数不是有效 JSON 对象。", "error")
                        return
                    if not isinstance(arguments, Mapping):
                        self._set_info("prompt 参数必须是 JSON 对象。", "error")
                        return
                payload = {
                    "operation": "get_prompt",
                    "server_name": server,
                    "name": selection,
                }
                if arguments is not None:
                    payload["arguments"] = dict(arguments)
            self._submit(payload, catalog=False)

        def approve_pending(self) -> None:
            approval_id = self._pending_approval_id
            if approval_id:
                self._submit(
                    {"operation": "approve", "approval_id": approval_id},
                    catalog=self._pending_approval_catalog,
                )

        def deny_pending(self) -> None:
            approval_id = self._pending_approval_id
            if approval_id:
                self._submit(
                    {"operation": "deny", "approval_id": approval_id},
                    catalog=False,
                )

        def show_and_focus(self) -> None:
            self.show()
            self.raise_()
            if self.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus:
                return
            self.activateWindow()

        def closeEvent(self, event: object) -> None:  # noqa: N802
            self._cancel_pending_request()
            self._cancel_pending_approval()
            self.closed.emit()
            super().closeEvent(event)


else:

    class MCPContentWindow:  # pragma: no cover
        """无 PySide6 时的明确不可用占位。"""

        def __init__(self, **_kwargs: object) -> None:
            raise RuntimeError("MCP content window requires PySide6")


__all__ = ["MCPContentWindow"]
