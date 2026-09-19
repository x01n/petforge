"""第 9 轮方向 3：双端公开动作白名单的对称性锁定（白名单用例）。

背景：Qt WebConsole（src/gui/qt6/web_console.py）与 HTTP 控制面
（src/gui/web/server.py）各自维护动作白名单，曾出现 10 项漂移。
第 9 轮把白名单收敛到 src/gui/web/action_contract.py 单源，本文件
用不依赖 PySide6 的方式断言：

1. Qt 端 32 项与 HTTP 端 24 项正确推导自单源常量；
2. 两端交集固定为 23 项共有动作；
3. 差异项与文档化豁免精确一致（Qt 端 9 项专属 / HTTP 端 open_input）；
4. 载荷键白名单与动作族集合互相对齐：登记了键白名单的动作必须属于
   某一端动作族；Qt 专属动作集是被 HTTP 端 pair 校验豁免的精确集合。
"""

from __future__ import annotations

from gui.web.action_contract import (
    BASE_PUBLIC_ACTION_KINDS,
    PUBLIC_ACTION_KINDS,
    PY_PUBLIC_ACTION_PAYLOAD_KEYS,
    QT_ONLY_EXTRA_KINDS,
    QT_PUBLIC_ACTION_KINDS,
    WEB_ONLY_EXTRA_KINDS,
)
from gui.web.control_surface import _PUBLIC_ACTION_KINDS as SURFACE_ACTION_KINDS
from gui.web.server import PUBLIC_ACTION_KINDS as HTTP_ACTION_KINDS

# 单源集合的成员数量：32（23 共有 + 9 Qt 专属）与 24（23 共有 + open_input）。
_KIND_TOTALS = {
    BASE_PUBLIC_ACTION_KINDS: 23,
    QT_ONLY_EXTRA_KINDS: 9,
    WEB_ONLY_EXTRA_KINDS: 1,
    QT_PUBLIC_ACTION_KINDS: 32,
    PUBLIC_ACTION_KINDS: 24,
}


def test_action_whitelists_derive_from_single_source() -> None:
    for kinds, expected_size in _KIND_TOTALS.items():
        assert len(kinds) == expected_size, f"动作族数量漂移: {kinds}"

    assert QT_PUBLIC_ACTION_KINDS == BASE_PUBLIC_ACTION_KINDS | QT_ONLY_EXTRA_KINDS
    assert PUBLIC_ACTION_KINDS == BASE_PUBLIC_ACTION_KINDS | WEB_ONLY_EXTRA_KINDS
    assert not (QT_ONLY_EXTRA_KINDS & WEB_ONLY_EXTRA_KINDS)
    assert not (BASE_PUBLIC_ACTION_KINDS & QT_ONLY_EXTRA_KINDS)
    assert not (BASE_PUBLIC_ACTION_KINDS & WEB_ONLY_EXTRA_KINDS)
    # 两端对称差与文档化差异精确一致。
    assert (QT_PUBLIC_ACTION_KINDS - PUBLIC_ACTION_KINDS) == QT_ONLY_EXTRA_KINDS
    assert (PUBLIC_ACTION_KINDS - QT_PUBLIC_ACTION_KINDS) == WEB_ONLY_EXTRA_KINDS


def test_http_server_exposes_common_whitelist_with_web_only_extra() -> None:
    assert PUBLIC_ACTION_KINDS == HTTP_ACTION_KINDS
    # HTTP 端网络语义下只额外放行本地热键动作 open_input。
    assert HTTP_ACTION_KINDS - BASE_PUBLIC_ACTION_KINDS == {"open_input"}
    # server.py 的载荷白名单不再为 Qt 专属动作提供登记（否则会绕过 pair 校验）。
    assert set(PY_PUBLIC_ACTION_PAYLOAD_KEYS).issubset(QT_PUBLIC_ACTION_KINDS | PUBLIC_ACTION_KINDS)


def test_control_surface_action_family_is_subset_of_qt_whitelist() -> None:
    # 状态投影层的动作族全部被 Qt 端点放行；HTTP 端不发行动作路由只是
    # 缺宿主实现，投影层的动作族仍是 chrome 动作，本断言锁定 Qt 侧完备。
    assert SURFACE_ACTION_KINDS <= QT_PUBLIC_ACTION_KINDS
    assert (SURFACE_ACTION_KINDS - PUBLIC_ACTION_KINDS) <= QT_ONLY_EXTRA_KINDS


# 无载荷语义动作全集（无 payload 键白名单登记，按空 frozenset 语义处理）。
_NO_PAYLOAD_KINDS = frozenset(
    {
        "open_input",
        "stop",
        "retry",
        "approve",
        "grant_session",
        "deny",
        "show_pet",
        "toggle_visibility",
        "center_pet",
        "toggle_window_lock",
        "toggle_always_on_top",
        "toggle_click_through",
        "restore_click_through",
        "read_foreground_window",
        "read_processes",
        "read_api_audit",
        "read_log_records",
    }
)


def test_payload_key_whitelist_matches_action_families() -> None:
    for kind in PY_PUBLIC_ACTION_PAYLOAD_KEYS:
        assert kind in QT_PUBLIC_ACTION_KINDS, f"{kind} 不再是公开动作族成员"
    for kind in QT_PUBLIC_ACTION_KINDS | PUBLIC_ACTION_KINDS:
        if kind in PY_PUBLIC_ACTION_PAYLOAD_KEYS:
            continue
        # 未登记键白名单的动作必须属于无载荷或 Qt 桥接内过滤的动作族，
        # 其中 restart_application 以空 frozenset 显式登记。
        assert kind in _NO_PAYLOAD_KINDS | {"restart_application"}, f"{kind} 的载荷键白名单缺失"
