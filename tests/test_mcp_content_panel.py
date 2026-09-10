from __future__ import annotations

from collections.abc import Mapping

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from gui.qt6.mcp_content_panel import MCPContentWindow, _safe_servers


def test_mcp_content_panel_sanitizes_malformed_server_counts() -> None:
    values = _safe_servers(
        (
            {
                "name": "demo",
                "resource_count": "not-a-count",
                "prompt_count": float("inf"),
            },
        )
    )
    assert values == (
        {
            "name": "demo",
            "source": "",
            "status": "",
            "resource_count": 0,
            "prompt_count": 0,
        },
    )


def test_mcp_content_panel_browses_resources_and_prompts_without_network() -> None:
    app = QApplication.instance() or QApplication([])
    calls: list[Mapping[str, object]] = []

    def servers() -> tuple[Mapping[str, object], ...]:
        return (
            {
                "name": "demo",
                "source": "mcp:source-demo",
                "status": "ready",
                "resource_count": 1,
                "prompt_count": 1,
            },
        )

    def action(payload: Mapping[str, object]) -> Mapping[str, object]:
        calls.append(dict(payload))
        operation = payload.get("operation")
        if operation == "list_resources":
            return {
                "status": "completed",
                "resources": [{"uri": "file:///demo", "name": "demo", "title": "演示"}],
            }
        if operation == "read_resource":
            return {
                "status": "completed",
                "content_policy": "quoted_external_data",
                "external_resource": {"contents": [{"uri": "file:///demo", "text": "ok"}]},
            }
        if operation == "list_prompts":
            return {
                "status": "completed",
                "prompts": [{"name": "review", "description": "演示模板"}],
            }
        return {
            "status": "completed",
            "content_policy": "quoted_external_data",
            "external_prompt": {"messages": [{"speaker": "user", "content": {"text": "ok"}}]},
        }

    window = MCPContentWindow(servers_provider=servers, action_provider=action)
    try:
        assert window.server_combo.currentData() == "demo"
        window.refresh_catalog()
        app.processEvents()
        assert window.item_list.count() == 1
        window.item_list.setCurrentRow(0)
        window.load_selected()
        assert "quoted_external_data" in window.output_edit.toPlainText()
        assert calls[-1]["operation"] == "read_resource"

        window.kind_combo.setCurrentIndex(1)
        window.refresh_catalog()
        app.processEvents()
        assert window.item_list.count() == 1
        window.item_list.setCurrentRow(0)
        window.load_selected()
        assert calls[-1]["operation"] == "get_prompt"
        assert "quoted_external_data" in window.output_edit.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_mcp_content_panel_can_complete_an_explicit_approval() -> None:
    app = QApplication.instance() or QApplication([])
    calls: list[Mapping[str, object]] = []

    def action(payload: Mapping[str, object]) -> Mapping[str, object]:
        calls.append(dict(payload))
        if payload.get("operation") == "approve":
            return {
                "status": "completed",
                "content_policy": "quoted_external_data",
                "resources": [{"uri": "file:///approved", "name": "approved"}],
            }
        return {
            "status": "approval_required",
            "approval_id": "approval-demo",
            "safe_summary": "需要确认：读取 MCP 资源",
        }

    window = MCPContentWindow(
        servers_provider=lambda: ({"name": "demo", "status": "ready"},),
        action_provider=action,
    )
    try:
        window.show()
        app.processEvents()
        window._finish_result(
            window._generation,
            {
                "status": "approval_required",
                "approval_id": "approval-demo",
                "safe_summary": "需要确认：读取 MCP 资源",
            },
            catalog=True,
        )
        assert window.approve_button.isVisible()
        window.approve_button.click()
        app.processEvents()
        assert calls[-1] == {"operation": "approve", "approval_id": "approval-demo"}
        assert not window.approve_button.isVisible()
        assert window.item_list.count() == 1
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
