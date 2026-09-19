"""公开动作路由契约：HTTP 控制面与 Qt WebConsole 共用的单源常量。

此前 ``PUBLIC_ACTION_KINDS`` 在 gui.web.server 与 gui.qt6.web_console 各持
一份，双端白名单曾出现 10 项漂移。本模块不依赖 Qt 与 GUI 运行时，两个
消费方均从此处导入，并以测试（tests/test_round9_action_contract.py）锁定
对称。

接口差异（文档化差异，非意外漂移）：

- ``RESTRICTED_INPUT_KINDS``：HTTP 端 ``open_input`` 仅是本地热键语义
  （Ctrl+Return 唤起对话框），经网络调用存在通话打断/输入劫持风险，
  因此只在 HTTP 端放行；Qt WebConsole 走本地宿主回调，不对其注册。
- ``RESTRICTED_HTTP_KINDS``：restart_application 等 9 项动作在 Qt 端
  由 app.py 的 ``web_control_action`` 完整实现，而 HTTP 端缺宿主实现
  分支；HttpControlSurface 在放行动作时成对校验 WEB_ONLY 豁免表，
  防止未来增删动作时单侧漂移。
"""

from __future__ import annotations

__all__ = [
    "BASE_PUBLIC_ACTION_KINDS",
    "QT_ONLY_EXTRA_KINDS",
    "WEB_ONLY_EXTRA_KINDS",
    "PUBLIC_ACTION_KINDS",
    "QT_PUBLIC_ACTION_KINDS",
    "PY_PUBLIC_ACTION_PAYLOAD_KEYS",
]

# 双端共有的公开动作族（前端桥接通过的回执状态语义一致）。
BASE_PUBLIC_ACTION_KINDS = frozenset(
    {
        "open_config",
        "configure_model",
        "select_model_channel",
        "submit_text",
        "stop",
        "retry",
        "approve",
        "grant_session",
        "deny",
        "pet_part",
        "expression",
        "motion",
        "show_pet",
        "toggle_visibility",
        "center_pet",
        "nudge_pet",
        "set_display_size",
        "toggle_window_lock",
        "toggle_always_on_top",
        "toggle_click_through",
        "restore_click_through",
        "read_foreground_window",
        "read_processes",
    }
)

# 仅 Qt WebConsole 放行（宿主 app.py web_control_action 有实现分支）。
QT_ONLY_EXTRA_KINDS = frozenset(
    {
        "restart_application",
        "select_renderer_backend",
        "select_renderer_model",
        "select_tts_profile",
        "select_tts_language",
        "expression_request",
        "motion_request",
        "read_api_audit",
        "read_log_records",
    }
)

# 仅 HTTP 控制面放行（网络语义下限制为本地热键动作）。
WEB_ONLY_EXTRA_KINDS = frozenset({"open_input"})

PUBLIC_ACTION_KINDS = BASE_PUBLIC_ACTION_KINDS | WEB_ONLY_EXTRA_KINDS
"""HTTP 控制面白名单：23 项共有 + open_input。"""

QT_PUBLIC_ACTION_KINDS = BASE_PUBLIC_ACTION_KINDS | QT_ONLY_EXTRA_KINDS
"""Qt WebConsole 白名单：23 项共有 + 9 项 Qt 专属。"""

PY_PUBLIC_ACTION_PAYLOAD_KEYS: dict[str, frozenset[str]] = {
    "open_config": frozenset({"section"}),
    "restart_application": frozenset(),
    "select_renderer_backend": frozenset({"backend"}),
    "configure_model": frozenset({"target", "mode"}),
    "select_model_channel": frozenset({"channel_id", "model"}),
    "select_renderer_model": frozenset({"model"}),
    "select_tts_profile": frozenset({"profile", "language"}),
    "select_tts_language": frozenset({"language"}),
    "submit_text": frozenset({"text"}),
    "pet_part": frozenset({"part"}),
    "expression": frozenset({"name"}),
    "motion": frozenset({"name"}),
    "expression_request": frozenset({"expressions", "mode", "loop"}),
    "motion_request": frozenset(
        {"name", "duration_seconds", "transition_seconds", "loop", "parameters"}
    ),
    "nudge_pet": frozenset({"direction"}),
    "set_display_size": frozenset({"preset"}),
}
