from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from gui.qt6.console import _safe_result_summary


def test_qt_console_result_summary_uses_friendly_labels() -> None:
    summary = _safe_result_summary(
        {
            "status": "completed",
            "zone": "head",
            "message": "点击互动已完成",
            "x": 120,
            "y": 80,
        }
    )
    assert "已完成" in summary
    assert "互动部位：猫猫头" in summary
    assert "status=" not in summary
    assert "zone=" not in summary
    assert "120" not in summary


def test_qt_console_hides_internal_parameter_names_from_short_text() -> None:
    from gui.qt6.console import _safe_ui_text

    assert _safe_ui_text("ParamAngleX 已更新") == "敏感详情已隐藏"


def test_qt_console_hides_extended_credential_fields() -> None:
    from gui.qt6.console import _friendly_stream_text, _safe_ui_text

    for value in (
        "credential=private-value",
        "access_key: private-value",
        "private-key=private-value",
    ):
        assert _safe_ui_text(value) == "敏感详情已隐藏"
        assert _friendly_stream_text(value) == "敏感详情已隐藏"


def test_qt_console_result_object_uses_friendly_status_without_protocol_format() -> None:
    from types import SimpleNamespace

    summary = _safe_result_summary(
        SimpleNamespace(status="unavailable", detail="窗口输入区域正在恢复")
    )
    assert summary == "暂时不可用：窗口输入区域正在恢复"
    assert "status=" not in summary
    assert "detail=" not in summary


def test_qt_console_stream_status_markers_are_translated_at_display_boundary() -> None:
    from gui.qt6.console import _friendly_stream_text

    rendered = _friendly_stream_text("[approval_required] 需要确认：屏幕\n[completed] 已完成：动作")

    assert "等待确认：屏幕" in rendered
    assert "已完成：动作" in rendered
    assert "\n" in rendered
    assert "approval_required" not in rendered


def test_qt_console_hides_internal_protocol_assignments_in_detail() -> None:
    from gui.qt6.console import _safe_ui_text

    assert _safe_ui_text("status=unavailable detail=raw payload") == "敏感详情已隐藏"


def test_model_setup_nested_payload_distinguishes_explicit_empty_key() -> None:
    """完整 ``llm`` 载荷中的空密钥也必须被识别为用户的明确清除。"""

    from gui.qt6.console import _payload_declares_api_key

    assert _payload_declares_api_key(
        {"llm": {"channels": [{"id": "primary", "api_key": ""}]}},
        "primary",
    )
    assert not _payload_declares_api_key(
        {"llm": {"channels": [{"id": "primary", "model": "demo"}]}},
        "primary",
    )


def test_qt_console_component_stream_does_not_repeat_previous_murmur(tmp_path: Path) -> None:
    """最终正文到达时不能把已显示的碎碎念整段重新追加。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow
from services.conversation.presentation import BubbleSnapshot

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow()
window.set_snapshot(BubbleSnapshot(murmur="先想一想"))
window.set_snapshot(
    BubbleSnapshot(murmur="先想一想，再确认", tool_status="[approval_required] 查看屏幕")
)
window.set_snapshot(
    BubbleSnapshot(
        text="已完成结果",
        murmur="先想一想，再确认",
        tool_status="[completed] 查看屏幕",
    )
)
history = window.output_view.toPlainText()
assert history.count("先想一想") == 1, history
assert history.count("已完成结果") == 1, history
assert "等待确认：查看屏幕" in history
assert "已完成：查看屏幕" in history
assert "查看屏幕已完成" not in history, history
window.shutdown()
app.processEvents()
print("qt-console-component-stream-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-component-stream-ok" in result.stdout


def test_qt_console_submits_commands_and_keeps_stream_history(tmp_path: Path) -> None:
    """控制台应能发送、中断并保留用户消息与流式模型输出。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
window = PetConsoleWindow(
    callbacks={
        "submit": lambda value: calls.append(("submit", value)),
        "stop": lambda: calls.append(("stop",)),
        "move_to": lambda x, y: calls.append(("move_to", x, y))
        or {"status": "available", "x": x, "y": y},
    }
)
window.show()
app.processEvents()
window.set_renderer_status("web_live2d", available=True)
assert "Web Live2D" in window._renderer_status.text()
window.set_renderer_status("vllank", available=False, reason="内部探测参数不应展示")
assert "3D 渲染预留" in window._renderer_status.text()
assert "内部探测参数" not in window._renderer_status.text()
window.input_line.setText("你好")
window.input_line.returnPressed.emit()
assert calls == [("submit", "你好")]
window.set_snapshot(SimpleNamespace(rendered_text="答", rendered_mood="happy"))
assert "桌宠（开心）：答" in window.output_view.toPlainText()
window.set_snapshot(SimpleNamespace(rendered_text="答案", rendered_mood="happy"))
window.position_x.setValue(-120)
window.position_y.setValue(340)
window._move_pet()
assert calls[-1] == ("move_to", -120, 340)
assert "桌宠位置：(-120, 340)" in window._status.text()
window._callbacks["move_to"] = lambda _x, _y: {
    "status": "degraded",
    "detail": "X11 window manager returned a partial result",
}
window._move_pet()
assert "平台返回降级结果" in window._status.text()
assert "Wayland" not in window._status.text()
window._callbacks["move_to"] = lambda _x, _y: {"status": "pending"}
window._move_pet()
assert "移动已排队" in window._status.text()
window._stop()
assert calls[-1] == ("stop",)
history = window.output_view.toPlainText()
assert "我：你好" in history
assert "桌宠（开心）：答案" in history
window.close()
assert not window.isVisible()
window.shutdown()
app.processEvents()
print("qt-console-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-ok" in result.stdout


def test_qt_console_model_selector_routes_exact_key_and_receipt(tmp_path: Path) -> None:
    """模型选择器只提交已列出的资源键，并处理异步成功/失败回执。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
seen = []
saved = []
window = PetConsoleWindow(
    configuration={
        "rendering": {
            "backend": "opengl",
            "model": "live2d/model/first/first.model3.json",
        }
    },
    config_callbacks={
        "save": lambda draft: saved.append(draft)
        or {"status": "saved", "runtime_status": "restart_required"}
    },
)

def select_model(value):
    seen.append(value)
    window._save_configuration({"rendering": {"model": value}})
    return {"status": "pending", "request_id": 9, "model_key": value}

window._callbacks["select_renderer_model"] = select_model
window.set_renderer_models(
    "live2d/model/first/first.model3.json",
    (
        "live2d/model/first/first.model3.json",
        "live2d/model/second/second.model3.json",
    ),
)
assert window.renderer_model_options == (
    "live2d/model/first/first.model3.json",
    "live2d/model/second/second.model3.json",
)
window.renderer_model_selector.setCurrentIndex(1)
window.renderer_model_apply_button.click()
assert seen == ["live2d/model/second/second.model3.json"]
assert saved[0]["rendering"]["model"] == "live2d/model/second/second.model3.json"
assert "切换中" in window._status.text()
window.set_renderer_model_result({
    "status": "available",
    "model_key": "live2d/model/second/second.model3.json",
})
assert "已切换" in window._status.text()
assert window.renderer_model_selector.currentData() == "live2d/model/second/second.model3.json"
window.renderer_model_selector.setCurrentIndex(0)
window.renderer_model_apply_button.click()
window.set_renderer_model_result({
    "status": "failed",
    "model_key": "live2d/model/first/first.model3.json",
    "reason": "页面尚未就绪",
})
assert window.renderer_model_selector.currentData() == "live2d/model/second/second.model3.json"
assert "未切换" in window._status.text()
window.shutdown()
app.processEvents()
print("qt-console-model-selector-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-model-selector-ok" in result.stdout


def test_qt_console_does_not_report_rejected_actions_as_success(tmp_path: Path) -> None:
    """渲染器返回 False 或平台返回不可用时，控制台必须明确反馈失败。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    callbacks={
        "expression": lambda _name: False,
        "motion": lambda _name: False,
        "click_through": lambda _enabled: {
            "status": "degraded",
            "enabled": True,
            "detail": "Wayland compositor",
        },
    },
    expressions=("happy",),
    motions=("wave",),
)
window.show()
app.processEvents()
window._set_expression()
assert "不可用" in window._status.text()
window._play_motion()
assert "不可用" in window._status.text()
window._click_through.setChecked(True)
assert window._click_through.isChecked()
assert "Wayland" in window._status.text()
window.shutdown()
app.processEvents()
print("qt-console-action-status-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-action-status-ok" in result.stdout


def test_qt_console_keeps_zero_values_in_pet_feedback(tmp_path: Path) -> None:
    """好感度为零时仍应在猫猫头反馈中显示当前值和本次变化。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow()
window.set_pet_feedback({
    "zone": "head",
    "phrase": "喵？",
    "affection": {"current": 0, "applied": 0},
})
text = window.findChild(type(window._pet_feedback), "petInteractionFeedback").text()
assert "好感度 0" in text
assert "本次 0" in text
window.shutdown()
app.processEvents()
print("qt-console-zero-feedback-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-zero-feedback-ok" in result.stdout


def test_qt_console_exposes_visible_pet_recovery_controls(tmp_path: Path) -> None:
    """实时控制页必须提供显示/隐藏、置顶和恢复点击的可见入口。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
window = PetConsoleWindow(
    callbacks={
        "show_pet": lambda: calls.append("show") or {"status": "available"},
        "toggle_visibility": lambda: calls.append("visibility") or {"status": "available"},
        "toggle_always_on_top": lambda: calls.append("topmost") or {
            "status": "requested", "enabled": True
        },
        "restore_click_through": lambda: calls.append("restore") or {"status": "available"},
    }
)
for object_name in (
    "toggleVisibilityButton",
    "toggleAlwaysOnTopButton",
    "restoreClickThroughButton",
):
    button = window.findChild(QPushButton, object_name)
    assert button is not None
    button.click()
    if object_name == "toggleAlwaysOnTopButton":
        assert "正在应用" in window._status.text()
assert calls == ["visibility", "topmost", "restore"]
window.shutdown()
app.processEvents()
print("qt-console-recovery-controls-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-recovery-controls-ok" in result.stdout


def test_qt_console_does_not_label_non_wayland_degraded_as_success(tmp_path: Path) -> None:
    """Qt flag 未保留时的降级结果不能被误报成 Wayland 已请求。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    callbacks={
        "click_through": lambda _enabled: {
            "status": "degraded",
            "enabled": False,
            "detail": "window system did not retain the requested Qt flag",
        },
    }
)
window.show()
app.processEvents()
window._click_through.setChecked(True)
assert not window._click_through.isChecked()
assert "不可用" in window._status.text()
assert "Wayland" not in window._status.text()
window.shutdown()
app.processEvents()
print("qt-console-degraded-boundary-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-degraded-boundary-ok" in result.stdout


def test_qt_console_requires_explicit_wayland_detail_for_degraded(tmp_path: Path) -> None:
    """没有 Wayland/compositor 证据时，degraded 不能被当作已提交。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    callbacks={
        "click_through": lambda _enabled: {
            "status": "degraded",
            "enabled": True,
            "detail": "partial X11 result",
        },
    }
)
window.show()
app.processEvents()
window._click_through.setChecked(True)
assert not window._click_through.isChecked()
assert "不可用" in window._status.text()
assert "Wayland" not in window._status.text()
window.shutdown()
app.processEvents()
print("qt-console-wayland-detail-required-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-wayland-detail-required-ok" in result.stdout


def test_qt_console_can_submit_approval_decisions_without_exposing_arguments(
    tmp_path: Path,
) -> None:
    """审批列表只提交标识，批准和拒绝均交给宿主回调。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from types import SimpleNamespace
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
window = PetConsoleWindow(
    callbacks={
        "approve_approval": lambda approval_id, grant: calls.append(
            ("approve", approval_id, grant)
        ) or {"status": "requested"},
        "deny_approval": lambda approval_id: calls.append(("deny", approval_id))
        or {"status": "requested"},
    }
)
window.set_pending_approvals(
    [
        SimpleNamespace(
            approval_id="approval-1",
            identity="pet:move",
            safe_summary="需要执行 pet:move",
        ),
        SimpleNamespace(
            approval_id="approval-2",
            identity="system:command",
            safe_summary="需要执行 system:command",
        ),
    ]
)
assert window.approval_list.count() == 2
assert window.selected_approval_id() == "approval-1"
window._grant_session.setChecked(True)
window._approve_selected_approval()
assert calls == [("approve", "approval-1", True)]
window.approval_list.setCurrentRow(1)
window._deny_selected_approval()
assert calls[-1] == ("deny", "approval-2")
window.set_pending_approvals([])
assert window.approval_list.count() == 1
assert window.selected_approval_id() == ""
window.shutdown()
app.processEvents()
print("qt-console-approval-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-approval-ok" in result.stdout


def test_qt_console_never_builds_empty_action_dropdowns_and_refreshes_capabilities(
    tmp_path: Path,
) -> None:
    """渲染器能力为空或脏数据时，动作/表情控件仍应有可执行选项。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication, QScrollArea, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(expressions=("", " happy ", "happy"), motions=(None, "  ", "wave"))
assert window.expression_options == ("happy",)
assert window.motion_options == ("wave",)
assert window._expression.currentText() == "happy"
assert window._motion.currentText() == "wave"
scroll = window.findChild(QScrollArea, "consoleControlScroll")
assert scroll is not None
assert window.minimumHeight() >= 500
for label in (
    "显示桌宠", "显示/隐藏桌宠", "切换窗口置顶", "恢复点击",
    "读取前台窗口", "读取进程", "退出程序",
):
    assert any(button.text() == label for button in window.findChildren(QPushButton))
window.set_action_capabilities((), ())
assert window.expression_options == ("neutral", "happy", "sad", "curious", "surprised", "shy")
assert window.motion_options == ("idle", "blink", "wave", "walk")
assert all(item.strip() for item in window.expression_options + window.motion_options)
window._motion.setCurrentText("model_custom_motion")
window.set_action_capabilities(("neutral",), ("idle",))
assert window._motion.currentText() == "model_custom_motion"
# 能力刷新暂时保留旧文本只为避免跳选；普通操作路径不能因此把
# 未声明动作提交给渲染器并显示成功。
window._play_motion()
assert "请从能力按钮中选择" in window._status.text()
window.shutdown()
app.processEvents()
print("qt-console-action-options-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-action-options-ok" in result.stdout


def test_qt_console_visual_actions_do_not_require_dialogue_model(tmp_path: Path) -> None:
    """视觉动作等待渲染器首帧时，不应提示用户配置对话模型。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
def pending(name):
    calls.append(name)
    return {"status": "pending", "reason": "renderer page loading"}

window = PetConsoleWindow(
    expressions=("neutral",),
    motions=("idle",),
    callbacks={"expression": lambda value: pending(value), "motion": lambda value: pending(value)},
)
window._expression.setCurrentText("neutral")
window._set_expression()
assert calls == ["neutral"]
assert "渲染器加载后执行" in window.status_label.text()
assert "配置模型" not in window.status_label.text()
window._motion.setCurrentText("idle")
window._play_motion()
assert "渲染器加载后执行" in window.status_label.text()
assert "配置模型" not in window.status_label.text()
window.shutdown()
app.processEvents()
print("qt-console-visual-action-model-independent-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-visual-action-model-independent-ok" in result.stdout


def test_qt_console_collapses_case_only_action_aliases(tmp_path: Path) -> None:
    """模型能力仅大小写不同的别名不能生成重复快捷按钮。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
from PySide6.QtWidgets import QApplication

from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    expressions=("Blink", "blink", " HAPPY ", "happy"),
    motions=("Walk", "walk", " IDLE "),
)
assert window.expression_options == ("Blink", "HAPPY")
assert window.motion_options == ("Walk", "IDLE")
assert [button.property("actionValue") for button in window.expression_quick_buttons] == [
    "Blink",
    "HAPPY",
]
assert [button.property("actionValue") for button in window.motion_quick_buttons] == [
    "Walk",
    "IDLE",
]
window.shutdown()
app.processEvents()
print("qt-console-case-alias-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-case-alias-ok" in result.stdout


def test_qt_console_action_buttons_drive_sprite_renderer(tmp_path: Path) -> None:
    """控制台测试按钮应真正更新精灵渲染器状态，而不是只更新状态文字。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "src"
    sprite_dir = project_root / "resources" / "sprites"
    script = f"""
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow
from gui.renderers.sprite import SpriteRenderer

app = QApplication.instance() or QApplication([])
renderer = SpriteRenderer({str(sprite_dir)!r})
window = PetConsoleWindow(
    callbacks={{"expression": renderer.set_expression, "motion": renderer.play_motion}},
    expressions=renderer.capabilities.expressions,
    motions=renderer.capabilities.motions,
)
window._expression.setCurrentText("happy")
window._set_expression()
assert renderer.state.expression == "happy"
assert "表情：开心" in window._status.text()
window._motion.setCurrentText("wave")
window._play_motion()
assert renderer.state.motion == "wave"
assert "动作：挥手" in window._status.text()
window.shutdown()
app.processEvents()
print("qt-console-actions-renderer-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-actions-renderer-ok" in result.stdout


def test_qt_console_position_controls_read_bounds_clamp_and_report(tmp_path: Path) -> None:
    """位置辅助入口应读取宿主边界、限制坐标并反馈按钮结果。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
window = PetConsoleWindow(
    callbacks={
        "position": lambda: (120, 180),
        "movement_bounds": lambda: (10, 20, 300, 400),
        "move_to": lambda x, y: calls.append((x, y)) or {
            "status": "available", "x": x, "y": y
        },
        "show_pet": lambda: {"status": "available", "visible": True},
        "toggle_always_on_top": lambda: {"status": "available", "enabled": True},
    }
)
assert window.position_x.minimum() == 10
assert window.position_x.maximum() == 300
assert window.position_y.minimum() == 20
assert window.position_y.maximum() == 400
window._refresh_position()
assert (window.position_x.value(), window.position_y.value()) == (120, 180)
window.position_x.setValue(300)
window.position_y.setValue(400)
window.position_x.setValue(10)
window.position_y.setValue(20)
window.position_x.setRange(-100000, 100000)
window.position_y.setRange(-100000, 100000)
window.position_x.setValue(99999)
window.position_y.setValue(-99999)
window._move_pet()
assert calls[-1] == (300, 20)
assert "屏幕范围" in window._status.text()
window._run_control_action("show_pet")
assert "桌宠已显示" in window._status.text()
window._run_control_action("toggle_always_on_top")
assert "置顶已开启" in window._status.text()
assert "6 项" in window._expression_hint.text()
assert "4 项" in window._motion_hint.text()
window.shutdown()
app.processEvents()
print("qt-console-position-controls-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-position-controls-ok" in result.stdout


def test_qt_console_operation_surface_has_clickable_shortcuts_and_signals(tmp_path: Path) -> None:
    """操作面应以快捷按钮为主，并暴露稳定对象名和信号。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication, QTabWidget, QPushButton
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
calls = []
signals = []
window = PetConsoleWindow(
    callbacks={
        "submit": lambda value: calls.append(("submit", value)),
        "stop": lambda: calls.append(("stop",)),
        "expression": lambda value: calls.append(("expression", value)) or True,
        "motion": lambda value: calls.append(("motion", value)) or True,
        "foreground_window": lambda: calls.append(("foreground",)) or {"status": "requested"},
        "list_processes": lambda: calls.append(("processes",)) or {"status": "requested"},
    },
    expressions=("neutral", "happy"),
    motions=("idle", "wave"),
    configuration={"app": {"name": "MeaPet"}},
)
for signal_name in (
    "conversationSubmitted",
    "stopRequested",
    "expressionRequested",
    "motionRequested",
    "foregroundWindowRequested",
    "processesRequested",
    "configurationRequested",
):
    getattr(window, signal_name).connect(
        lambda *args, name=signal_name: signals.append((name, args))
    )

assert window.findChild(QPushButton, "sendMessageButton") is not None
assert window.findChild(QPushButton, "stopConversationButton") is not None
assert window.findChild(QPushButton, "configCenterButton") is not None
assert window.findChild(QPushButton, "configureModelButton") is not None
assert window.findChild(QPushButton, "centerPetPositionButton") is not None
assert window.findChild(QPushButton, "restoreClickThroughButton") is not None
assert window.findChild(QTabWidget) is not None
window.show()
app.processEvents()
assert window._navigate_console_page("对话")
app.processEvents()
assert window._empty_state.isVisible()

assert window._navigate_console_page("桌宠")
app.processEvents()
window.expression_quick_buttons[1].click()
window.motion_quick_buttons[1].click()
assert ("expression", "happy") in calls
assert ("motion", "wave") in calls
assert ("expressionRequested", ("happy",)) in signals
assert ("motionRequested", ("wave",)) in signals

assert window._navigate_console_page("对话")
app.processEvents()
window.input_line.setText("你好")
window.findChild(QPushButton, "sendMessageButton").click()
window.findChild(QPushButton, "stopConversationButton").click()
assert ("submit", "你好") in calls
assert ("conversationSubmitted", ("你好",)) in signals
assert ("stop",) in calls
assert any(name == "stopRequested" for name, _args in signals)

assert window._navigate_console_page("工具与安全")
app.processEvents()
window.findChild(QPushButton, "foregroundWindowButton").click()
window.findChild(QPushButton, "listProcessesButton").click()
assert ("foreground",) in calls
assert ("processes",) in calls
assert any(name == "foregroundWindowRequested" for name, _args in signals)
assert any(name == "processesRequested" for name, _args in signals)

window.findChild(QPushButton, "configCenterButton").click()
assert any(name == "configurationRequested" for name, _args in signals)
window.findChild(QPushButton, "configureModelButton").click()
assert window._model_setup_dialog is not None
window._model_setup_dialog.close()
window.append_line("系统：完成")
assert not window._empty_state.isVisible()
window.shutdown()
app.processEvents()
print("qt-console-operation-surface-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-operation-surface-ok" in result.stdout


def test_qt_console_model_setup_is_clickable_and_keeps_key_as_env_reference(tmp_path: Path) -> None:
    """配置模型按钮应打开向导并只提交环境变量引用。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
saved = []
window = PetConsoleWindow(
    configuration={"llm": {"channels": []}},
    callbacks={"config_save": lambda values: saved.append(values) or {"status": "saved"}},
)
assert window.show_model_settings()
dialog = window._model_setup_dialog
assert dialog is not None
assert dialog.card.preset_combo.currentData() == "custom"
dialog.card.preset_combo.setCurrentIndex(dialog.card.preset_combo.findData("openai"))
assert dialog.card.submit()
assert saved
channel = saved[0]["llm"]["channels"][0]
assert channel["api_key"] == "${OPENAI_API_KEY}"
assert "secret" not in repr(saved[0])
window.shutdown()
app.processEvents()
print("qt-console-model-setup-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-model-setup-ok" in result.stdout


def test_qt_console_model_setup_explicitly_clears_api_key_reference(tmp_path: Path) -> None:
    """向导提交空密钥环境变量时，不能被旧引用静默回填。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
saved = []
window = PetConsoleWindow(
    configuration={
        "llm": {
            "channels": [{
                "id": "primary",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8317/v1",
                "model": "gpt-5.4-mini",
                "api_key": "${MEAPET_TEST_KEY}",
            }],
        },
    },
    callbacks={"config_save": lambda values: saved.append(values) or {"status": "saved"}},
)
assert window._apply_model_setup_payload({
    "channel_id": "primary",
    "protocol": "openai_chat",
    "base_url": "http://127.0.0.1:8317/v1",
    "model": "gpt-5.4-mini",
    "api_key_env": "",
    "capabilities": ["streaming"],
    "enabled": True,
})
assert saved
assert saved[0]["llm"]["channels"][0].get("api_key") == ""
window.shutdown()
app.processEvents()
print("qt-console-model-key-clear-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-model-key-clear-ok" in result.stdout


def test_qt_console_model_setup_full_config_payload_clears_api_key_reference(
    tmp_path: Path,
) -> None:
    """对话框提交的完整 ``llm`` 草稿不能把旧密钥引用静默恢复。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
saved = []
window = PetConsoleWindow(
    configuration={
        "llm": {
            "channels": [{
                "id": "primary",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8317/v1",
                "model": "gpt-5.4-mini",
                "api_key": "${MEAPET_TEST_KEY}",
            }],
        },
    },
    callbacks={"config_save": lambda values: saved.append(values) or {"status": "saved"}},
)
assert window._apply_model_setup_payload({
    "llm": {
        "channels": [{
            "id": "primary",
            "protocol": "openai_chat",
            "base_url": "http://127.0.0.1:8317/v1",
            "model": "gpt-5.4-mini",
            "api_key": "",
        }],
    },
})
assert saved
assert saved[0]["llm"]["channels"][0].get("api_key") == ""
window.shutdown()
app.processEvents()
print("qt-console-model-full-payload-key-clear-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-model-full-payload-key-clear-ok" in result.stdout


def test_qt_console_model_setup_propagates_missing_credentials_status(tmp_path: Path) -> None:
    """待补密钥的保存结果要同时反馈到控制台和向导，而不是伪报成功。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
window = PetConsoleWindow(
    configuration={"llm": {"channels": []}},
    callbacks={
        "config_save": lambda _values: {
            "status": "saved",
            "runtime_status": "restart_required",
            "credentials_required": True,
        }
    },
)
result = window._apply_model_setup_payload({
    "channel_id": "primary",
    "protocol": "openai_chat",
    "base_url": "https://gateway.invalid/v1",
    "model": "demo",
    "api_key_env": "MISSING_TEST_KEY",
    "capabilities": ["streaming"],
    "enabled": True,
})
assert isinstance(result, dict) and result["credentials_required"] is True
assert "补齐密钥环境变量" in window._status.text()
window.shutdown()
app.processEvents()
print("qt-console-missing-credentials-status-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-missing-credentials-status-ok" in result.stdout


def test_qt_console_model_setup_persists_plaintext_only_through_explicit_callback(
    tmp_path: Path,
) -> None:
    """明文密钥模式必须走独立回调，普通配置保存不能隐式放宽。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
secret = "file-persisted-secret"
saved = []
window = PetConsoleWindow(
    configuration={"llm": {"channels": []}},
    config_callbacks={
        "save_plaintext": lambda values: saved.append(values) or {"status": "saved"},
    },
)
result = window._apply_model_setup_payload({
    "channel_id": "primary",
    "protocol": "openai_chat",
    "base_url": "https://gateway.invalid/v1",
    "model": "demo",
    "api_key": secret,
    "_persist_api_key": True,
    "capabilities": ["streaming"],
    "enabled": True,
})
assert isinstance(result, dict) and result["status"] == "saved"
assert saved and saved[0]["llm"]["channels"][0]["api_key"] == secret
assert saved[0]["llm"]["channels"][0]["api_key_env"] == ""
assert secret not in window.configuration_panel.yaml_editor.toPlainText()
window.shutdown()
app.processEvents()
print("qt-console-explicit-plaintext-save-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-explicit-plaintext-save-ok" in result.stdout


def test_qt_console_model_channel_delete_updates_primary_route(tmp_path: Path) -> None:
    """删除主渠道后应自动选择剩余渠道并清理回退列表。"""

    try:
        import PySide6  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError):
        return

    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from PySide6.QtWidgets import QApplication
from gui.qt6.console import PetConsoleWindow

app = QApplication.instance() or QApplication([])
saved = []
window = PetConsoleWindow(
    configuration={
        "llm": {
            "channels": [
                {
                    "id": "primary",
                    "protocol": "openai_chat",
                    "base_url": "https://a.invalid/v1",
                    "model": "a",
                },
                {
                    "id": "backup",
                    "protocol": "openai_chat",
                    "base_url": "https://b.invalid/v1",
                    "model": "b",
                },
            ],
            "routing": {
                "dialogue": {
                    "channel": "primary",
                    "required_capabilities": ["streaming"],
                    "fallback_channels": ["backup", "primary"],
                },
                "vision": {
                    "channel": "primary",
                    "required_capabilities": ["streaming", "vision"],
                    "fallback_channels": ["primary"],
                },
            },
        }
    },
    config_callbacks={"save": lambda values: saved.append(values) or {"status": "saved"}},
)
result = window._delete_model_channel("primary")
assert isinstance(result, dict) and result["status"] == "saved"
assert [item["id"] for item in saved[0]["llm"]["channels"]] == ["backup"]
route = saved[0]["llm"]["routing"]["dialogue"]
assert route["channel"] == "backup"
assert route["fallback_channels"] == ["backup"]
assert "vision" not in saved[0]["llm"]["routing"]
window.shutdown()
app.processEvents()
print("qt-console-channel-delete-route-ok")
"""
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    old_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{old_python_path}" if old_python_path else str(source_root)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "qt-console-channel-delete-route-ok" in result.stdout
