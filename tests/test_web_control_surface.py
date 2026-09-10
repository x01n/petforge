from __future__ import annotations

import json
import time
from types import SimpleNamespace

from gui.qt6.app import _public_model_channels
from gui.web.control_surface import control_surface_html, friendly_public_text, public_control_state


def test_public_control_state_filters_approval_identifiers_and_parameters() -> None:
    state = public_control_state(
        {
            "interaction": {
                "phase": "approval_required",
                "approval": {
                    "approval_id": "private-approval-view",
                    "call_id": "private-call-view",
                    "display_name": "打开文件",
                    "safe_summary": "需要确认一个文件操作",
                    "expires_at": 2_000_000_000,
                },
                "pending_approvals": [
                    {
                        "approval_id": "private-approval-view",
                        "call_id": "private-call-view",
                        "display_name": "打开文件",
                        "safe_summary": "需要确认一个文件操作",
                        "expires_at": 2_000_000_000,
                    }
                ],
                "actions": [
                    {
                        "kind": "approve",
                        "label": "批准",
                        "payload": {
                            "approval_id": "private-approval",
                            "call_id": "private-call",
                            "arguments": {"command": "secret"},
                            "target": "api_key=secret",
                        },
                    },
                    {
                        "kind": "system:run_command",
                        "label": "执行命令参数",
                        "payload": {"target": "secret"},
                    },
                ],
            },
            "renderer": {"backend": "web_live2d", "available": True},
            "feedback": {
                "part": "head",
                "phrase": "喵？摸摸头～",
                "affection": {"current": 6, "applied": 1, "secret": "hidden"},
            },
            "parts": [{"id": "head", "label": "摸摸头", "native_id": "Face"}],
            "capabilities": {"expressions": ["happy"], "motions": ["wave"]},
        }
    )
    encoded = json.dumps(state, ensure_ascii=False)
    assert "private-approval" not in encoded
    assert "private-call" not in encoded
    assert "secret" not in encoded
    assert state["interaction"]["actions"][0]["payload"] == {}
    assert state["feedback"]["part"] == "head"
    assert state["feedback"]["affection"]["current"] == "6"
    assert "native_id" not in json.dumps(state, ensure_ascii=False)
    assert state["interaction"]["approval"]["display_name"] == "打开文件"
    assert "private-approval-view" not in encoded


def test_public_control_state_exposes_safe_topmost_boundary() -> None:
    public = public_control_state(
        {
            "window": {
                "always_on_top": True,
                "always_on_top_status": {
                    "status": "degraded",
                    "enabled": True,
                    "detail": "Wayland compositor may override the topmost request",
                },
            }
        }
    )
    assert public["window"]["always_on_top_status"] == {
        "status": "degraded",
        "enabled": True,
        "detail": "Wayland compositor may override the topmost request",
    }


def test_control_surface_marks_pending_topmost_transition() -> None:
    """异步置顶回执到达前，页面不能把旧旗标误报成最终状态。"""

    html = control_surface_html(
        {
            "window": {
                "always_on_top": False,
                "always_on_top_status": {
                    "status": "requested",
                    "enabled": True,
                    "detail": "窗口置顶切换已提交",
                },
            }
        }
    )
    assert "topmostPending" in html
    assert "正在应用置顶" in html


def test_model_channel_health_projection_is_click_safe_and_redacted() -> None:
    class Router:
        def channels(self):
            return (
                SimpleNamespace(
                    id="primary",
                    protocol="openai_chat",
                    selected_model="demo-model",
                    enabled=True,
                    base_url="https://gateway.example.invalid/v1",
                    is_ready=True,
                ),
                SimpleNamespace(
                    id="offline",
                    protocol="ollama_chat",
                    selected_model="",
                    enabled=True,
                    base_url="",
                    is_ready=False,
                ),
            )

        def health(self):
            return {
                "primary": {
                    "failures": 2,
                    "cooldown_until": 0,
                    "last_error": "token=private",
                },
                "offline": {"failures": 0, "cooldown_until": time.time() + 120},
            }

    channels = _public_model_channels(Router())
    assert channels == (
        {
            "id": "primary",
            "protocol": "OpenAI",
            "model": "demo-model",
            "models": ("demo-model",),
            "model_configured": True,
            "ready": True,
            "failures": 2,
            "active": False,
            "selectable": True,
            "status": "连接已恢复",
        },
        {
            "id": "offline",
            "protocol": "Ollama",
            "model": "",
            "models": (),
            "model_configured": False,
            "ready": False,
            "failures": 0,
            "active": False,
            "selectable": False,
            "status": "未配置",
        },
    )
    public = public_control_state(
        {
            "model_channels": [
                *channels,
                {
                    "id": "injected",
                    "protocol": "https://private.invalid",
                    "model_configured": True,
                    "ready": False,
                    "failures": 999,
                    "status": "token=private",
                    "base_url": "https://private.invalid",
                    "api_key": "private-key",
                    "last_error": "authorization private",
                },
            ]
        }
    )
    assert public["model_channels"] == [
        {
            "id": "primary",
            "protocol": "OpenAI",
            "model": "demo-model",
            "models": ["demo-model"],
            "model_configured": True,
            "ready": True,
            "failures": 2,
            "active": False,
            "selectable": True,
            "status": "连接已恢复",
        },
        {
            "id": "offline",
            "protocol": "Ollama",
            "model": "",
            "models": [],
            "model_configured": False,
            "ready": False,
            "failures": 0,
            "active": False,
            "selectable": False,
            "status": "未配置",
        },
        {
            "id": "injected",
            "protocol": "自定义渠道",
            "model": "",
            "models": [],
            "model_configured": True,
            "ready": False,
            "failures": 999,
            "active": False,
            "selectable": False,
            "status": "未就绪",
        },
    ]
    encoded = json.dumps(public, ensure_ascii=False)
    assert "gateway.example.invalid" not in encoded
    assert "private-key" not in encoded
    assert "authorization private" not in encoded
    html = control_surface_html(public)
    assert "模型渠道状态" in html
    assert "只显示连接摘要" in html
    assert "channelStatusRows" in html
    assert "select_model_channel" in html
    assert "private.invalid" not in html
    assert "openai_chat" not in html
    assert "ollama_chat" not in html


def test_public_control_state_exposes_safe_image_processing_mode() -> None:
    public = public_control_state(
        {
            "model_image": {
                "ready": True,
                "mode": "summary",
                "channel_id": "vision-secret-route",
                "model": "vision-model",
                "reason": "internal route detail",
            }
        }
    )
    assert public["model_image"] == {
        "ready": True,
        "mode": "summary",
        "label": "视觉模型摘要回退",
    }
    html = control_surface_html(public)
    assert "modelImageStatus" in html
    assert "renderModelImageStatus" in html
    assert "vision-secret-route" not in html


def test_public_configuration_state_exposes_reload_health_without_file_details() -> None:
    state = public_control_state(
        {
            "configuration": {
                "connected": True,
                "watcher_enabled": True,
                "watcher_running": True,
                "auto_reload": True,
                "status": "running",
                "generation": 4,
                "message": "配置文件已连接，修改后会自动校验并应用",
                "path": "/private/home/meapet/config.yaml",
                "yaml": "api_key: private-value",
            }
        }
    )
    assert state["configuration"] == {
        "connected": True,
        "watcher_enabled": True,
        "watcher_running": True,
        "auto_reload": True,
        "status": "running",
        "status_label": "运行中",
        "generation": 4,
        "message": "配置文件已连接，修改后会自动校验并应用",
    }
    encoded = json.dumps(state, ensure_ascii=False)
    assert "/private/home" not in encoded
    assert "api_key" not in encoded
    html = control_surface_html(state)
    assert "配置与自动重载" in html
    assert "configurationCard" in html
    assert "open_config" in html
    assert "配置文件已连接" in html


def test_public_configuration_state_uses_safe_fallback_for_unknown_status() -> None:
    state = public_control_state(
        {
            "configuration": {
                "connected": True,
                "watcher_enabled": True,
                "watcher_running": False,
                "status": "internal_error",
                "message": "token=private-value",
            }
        }
    )
    configuration = state["configuration"]
    assert configuration["status"] == "unavailable"
    assert configuration["status_label"] == "暂时不可用"
    assert configuration["message"] == "敏感详情已隐藏"


def test_public_memory_status_exposes_strategy_only() -> None:
    state = public_control_state(
        {
            "memory": {
                "enabled": True,
                "status": "ready",
                "recall_limit": 7,
                "context_max_chars": 6000,
                "max_memories": 2000,
                "consolidation_enabled": True,
                "content": "private memory content",
                "database": "/private/memory.sqlite3",
            }
        }
    )
    assert state["memory"] == {
        "enabled": True,
        "status": "ready",
        "status_label": "已启用",
        "recall_limit": 7,
        "context_max_chars": 6000,
        "max_memories": 2000,
        "consolidation_enabled": True,
        "summarization_enabled": False,
        "summary_running": False,
        "summary_busy": False,
        "summary_completed": 0,
        "summary_last_status": "",
        "extraction_pending": 0,
        "extraction_completed": 0,
        "extraction_last_status": "",
        "vector_index_size": 0,
        "lexical_index": "unavailable",
        "message": "记忆已启用，新的对话会按优先级召回",
    }
    encoded = json.dumps(state, ensure_ascii=False)
    assert "private memory content" not in encoded
    assert "/private/memory.sqlite3" not in encoded
    html = control_surface_html(state)
    assert "记忆状态" in html
    assert "memoryCard" in html
    assert "renderMemoryStatus" in html


def test_public_theme_state_only_accepts_md3_tokens() -> None:
    state = public_control_state(
        {
            "theme": {
                "--md3-color-primary": "#123456",
                "--md3-radius-large": "24px",
                "--md3-unknown": "#FFFFFF",
                "color": "red",
                "--md3-color-error": "red; background:url(file:///secret)",
            }
        }
    )

    assert state["theme"] == {
        "--md3-color-primary": "#123456",
        "--md3-radius-large": "24px",
    }
    html = control_surface_html(state)
    assert "applyTheme(state.theme)" in html
    assert "--md3-color-primary" in html
    assert "file:///secret" not in html


def test_public_activity_status_exposes_provider_only() -> None:
    state = public_control_state(
        {
            "activity": {
                "system_idle_provider": "x11",
                "system_idle_status": "ready",
                "last_source": "private-window-title",
                "last_system_idle_seconds": 1.2,
            }
        }
    )
    assert state["activity"] == {
        "system_idle_provider": "x11",
        "system_idle_provider_label": "X11 空闲探针",
        "system_idle_status": "ready",
        "system_idle_status_label": "已就绪",
        "system_idle_available": True,
    }
    encoded = json.dumps(state, ensure_ascii=False)
    assert "private-window-title" not in encoded
    assert "last_system_idle_seconds" not in encoded
    html = control_surface_html(state)
    assert "用户活跃探针" in html
    assert "renderActivityStatus" in html


def test_public_activity_status_exposes_windows_provider_label() -> None:
    state = public_control_state(
        {
            "activity": {
                "system_idle_provider": "windows",
                "system_idle_status": "ready",
            }
        }
    )
    assert state["activity"] == {
        "system_idle_provider": "windows",
        "system_idle_provider_label": "Windows 空闲探针",
        "system_idle_status": "ready",
        "system_idle_status_label": "已就绪",
        "system_idle_available": True,
    }


def test_public_activity_status_is_total_for_malformed_provider_values() -> None:
    state = public_control_state(
        {
            "activity": {
                "system_idle_provider": {"unexpected": "object"},
                "system_idle_status": ["unexpected"],
            }
        }
    )
    assert state["activity"]["system_idle_provider"] == "disabled"
    assert state["activity"]["system_idle_status"] == "unavailable"


def test_model_channel_health_projection_is_empty_without_channels() -> None:
    assert _public_model_channels(None) == ()
    assert public_control_state({})["model_channels"] == []
    assert "暂无模型渠道" in control_surface_html({})


def test_model_channel_surface_renders_bounded_model_choices() -> None:
    html = control_surface_html(
        {
            "model_channels": [
                {
                    "id": "primary",
                    "protocol": "OpenAI",
                    "model": "model-a",
                    "models": ["model-a", "model-b"],
                    "ready": True,
                    "model_configured": True,
                    "selectable": True,
                    "active": True,
                }
            ]
        }
    )
    assert "model-a" in html
    assert "model-b" in html
    assert "channel_id:id,model:modelName" in html


def test_public_capabilities_never_render_empty_action_groups() -> None:
    public = public_control_state({"capabilities": {"expressions": (), "motions": ()}})
    expressions = public["capabilities"]["expressions"]
    motions = public["capabilities"]["motions"]
    assert expressions and expressions[0]["label"] == "自然"
    assert motions and motions[0]["label"] == "待机"
    html = control_surface_html(public)
    assert "动作" in html


def test_public_control_state_redacts_sensitive_text_and_opaque_capabilities() -> None:
    state = public_control_state(
        {
            "interaction": {
                "tool_status": {"headers": {"Authorization": "Bearer very-secret"}},
                "safe_message": "api_key=very-secret",
                "approval": {
                    "display_name": "system:run_command",
                    "safe_summary": "command_line: rm -rf secret",
                    "expires_at": 2_000_000_000,
                },
            },
            "renderer": {
                "backend": "web_live2d",
                "available": True,
                "message": "token=very-secret",
            },
            "capabilities": {
                "expressions": ["happy", "ParamAngleX"],
                "motions": ["IdleCustom", "wave"],
            },
        }
    )
    encoded = json.dumps(state, ensure_ascii=False)
    for marker in (
        "api_key",
        "token",
        "Authorization",
        "command_line",
        "ParamAngleX",
        "IdleCustom",
    ):
        assert marker not in encoded
    assert state["interaction"]["tool_status"] == "敏感详情已隐藏"
    assert state["interaction"]["approval"]["safe_summary"] == "敏感详情已隐藏"
    assert state["interaction"]["approval"]["display_name"] == "桌面操作"
    assert state["capabilities"]["expressions"] == [
        {"id": "cap-expression-0", "label": "开心"},
        {"id": "cap-expression-1", "label": "表情 2"},
    ]
    assert state["capabilities"]["motions"] == [
        {"id": "cap-motion-0", "label": "待机"},
        {"id": "cap-motion-1", "label": "挥手"},
    ]
    assert public_control_state(state)["capabilities"] == state["capabilities"]


def test_control_surface_embedded_state_cannot_close_script_element() -> None:
    html = control_surface_html(
        {"interaction": {"text": "</script><script>window.__injected__=true</script>"}}
    )
    assert "</script><script>" not in html
    assert r"\u003c/script\u003e" in html


def test_control_surface_html_is_click_first_and_contains_no_raw_config_editor() -> None:
    html = control_surface_html({"renderer": {"backend": "sprite", "available": True}})
    assert "摸摸头" in html
    assert "pet_part" in html
    assert "toggle_window_lock" in html
    assert "messageInput" in html
    assert "sendButton" in html
    assert "feedbackText" in html
    assert "猫猫头" in html
    assert "approvalCard" in html
    assert "operation" in html
    assert "center_pet" in html
    assert 'data-preset="standard"' in html
    assert "windowActions" in html
    assert "sizeActions" in html
    assert "toggle_click_through" in html
    assert "开启点击穿透" in html
    assert "控制通道尚未就绪" in html
    assert "操作响应超时" in html
    assert "pendingInvocations" in html
    assert "localOperationRevision" in html
    assert "处理中…" in html
    assert "focus-visible" in html
    assert "--background:" in html
    assert "--primary-foreground:" in html
    assert "prefers-reduced-motion" in html
    assert "aria-live" in html
    assert "查看前台窗口" in html
    assert "查看运行中的程序" in html
    assert "observationResult" in html
    assert "textarea" not in html
    assert "api_key" not in html
    assert "approval_id" not in html


def test_public_renderer_state_uses_friendly_label_only() -> None:
    state = public_control_state({"renderer": {"backend": "web_live2d", "available": True}})

    assert state["renderer"] == {
        "label": "Web Live2D",
        "available": True,
        "message": "",
    }
    html = control_surface_html({"renderer": {"backend": "vllank", "available": False}})
    assert "Vulkan 渲染" in html


def test_public_renderer_state_exposes_only_safe_model_keys() -> None:
    state = public_control_state(
        {
            "renderer": {
                "backend": "web_live2d",
                "available": True,
                "model": "live2d/model/demo/demo.model3.json",
                "models": [
                    "live2d/model/demo/demo.model3.json",
                    "live2d/model/other/other.model3.json",
                    {"secret": "hidden"},
                ],
            }
        }
    )
    assert state["renderer"]["model"] == "live2d/model/demo/demo.model3.json"
    assert state["renderer"]["models"][:2] == [
        "live2d/model/demo/demo.model3.json",
        "live2d/model/other/other.model3.json",
    ]
    assert "secret" not in json.dumps(state, ensure_ascii=False)


def test_public_window_state_contains_safe_position_and_click_only_nudges() -> None:
    state = public_control_state(
        {
            "window": {
                "locked": False,
                "click_through": False,
                "always_on_top": True,
                "position": {"x": 120.4, "y": -8.6, "pid": "hidden"},
                "input_shape": {"ready": True, "input_ready": False, "status": "pending"},
            }
        }
    )
    assert state["window"]["position"] == {"x": 120, "y": -9}
    assert state["window"]["input_shape"] == {
        "ready": True,
        "input_ready": False,
        "status": "等待处理",
    }
    html = control_surface_html(state)
    assert "nudge_pet" in html
    assert 'data-direction="left"' in html
    assert "位置：(" in html
    assert "点击区域恢复中" in html
    assert "!w.click_through&&!w.locked" in html


def test_public_approval_countdown_is_bounded_to_display_ttl() -> None:
    state = public_control_state(
        {
            "interaction": {
                "approval": {
                    "approval_id": "private",
                    "call_id": "private-call",
                    "display_name": "桌面操作",
                    "safe_summary": "需要确认",
                    "expires_at": 4_000_000_000,
                }
            }
        }
    )
    approval = state["interaction"]["approval"]
    assert 0 <= int(approval["remaining_seconds"]) <= 3600
    assert "Math.min(3600" in control_surface_html(state)


def test_public_model_state_keeps_configuration_as_one_click_action() -> None:
    state = public_control_state(
        {
            "interaction": {
                "model": {
                    "ready": False,
                    "message": "模型渠道未配置：运行 meapet-wizard channel add",
                    "action": {
                        "kind": "configure_model",
                        "label": "配置模型",
                        "payload": {"command": "secret"},
                    },
                }
            }
        }
    )
    model = state["interaction"]["model"]
    assert model["message"] == "模型尚未连接，请点击“配置模型”完成连接"
    assert model["action"][0]["kind"] == "configure_model"
    assert model["action"][0]["payload"] == {}
    html = control_surface_html(state)
    assert "模型连接" in html
    assert "配置模型" in html
    assert "meapet-wizard" not in html
    assert "command" not in html


def test_public_state_uses_friendly_phase_and_hides_operation_identifiers() -> None:
    state = public_control_state(
        {
            "interaction": {"phase": "approval_required"},
            "operation": {
                "id": "private-operation",
                "kind": "system:run_command",
                "status": "requested",
                "message": "等待确认",
                "updated_at": 12.0,
            },
            "timeline": [
                {
                    "id": "private-history",
                    "call_id": "private-call",
                    "kind": "pet_part",
                    "status": "completed",
                    "message": "已触发部位反馈",
                }
            ],
        }
    )
    encoded = json.dumps(state, ensure_ascii=False)
    assert state["interaction"]["phase"] == "等待你的确认"
    assert state["operation"]["label"] == "桌宠操作"
    assert "id" not in state["operation"]
    assert "kind" not in state["operation"]
    assert "private-operation" not in encoded
    assert "private-history" not in encoded
    assert "private-call" not in encoded
    assert state["timeline"][0]["label"] == "部位互动"
    assert state["timeline"][0]["status_label"] == "已完成"


def test_public_feedback_and_html_timeline_never_expose_model_parameters() -> None:
    state = public_control_state(
        {
            "feedback": {
                "part": "head",
                "expression": "ParamAngleX",
                "motion": "IdleCustom",
                "phrase": "摸摸头",
            },
            "timeline": [
                {
                    "kind": "expression",
                    "status": "completed",
                    "message": "ParamAngleX 已更新",
                }
            ],
        }
    )
    encoded = json.dumps(state, ensure_ascii=False)
    assert "ParamAngleX" not in encoded
    assert "IdleCustom" not in encoded
    assert state["feedback"]["expression"] == "表情 1"
    assert state["feedback"]["motion"] == "待机"
    html = control_surface_html(state)
    assert "活动记录" in html
    assert "clearTimeline" in html
    assert "活动记录已清空" in html
    assert "friendlyOperationLabels" in html


def test_web_lock_state_explains_pointer_behavior_and_disables_conflicting_toggle() -> None:
    state = public_control_state(
        {
            "window": {
                "locked": True,
                "click_through": False,
                "always_on_top": True,
            },
            "feedback": {"part": "head", "mood": "happy"},
        }
    )
    assert state["feedback"]["mood"] == "开心"
    html = control_surface_html(state)
    assert "已锁定 · 不接收点击，仍追踪光标" in html
    assert "先解锁窗口" in html
    assert "clickButton.disabled=Boolean(w.locked)" in html


def test_public_observation_keeps_only_clickable_summary_fields() -> None:
    state = public_control_state(
        {
            "observation": {
                "kind": "foreground_window",
                "status": "available",
                "title": "编辑器",
                "process_name": "editor",
                "pid": 42,
                "executable": "/secret/editor",
                "command_line": "--token private",
            }
        }
    )
    observation = state["observation"]
    encoded = json.dumps(observation, ensure_ascii=False)
    assert observation["label"] == "查看前台窗口"
    assert observation["title"] == "编辑器"
    assert observation["process_name"] == "editor"
    assert observation["pid"] == 42
    assert "executable" not in encoded
    assert "command_line" not in encoded
    assert "private" not in encoded

    processes = public_control_state(
        {
            "observation": {
                "kind": "processes",
                "status": "available",
                "processes": [
                    {"name": "editor", "pid": 42, "argv": ["--secret"]},
                    {"name": "terminal", "pid": "not-a-pid"},
                ],
            }
        }
    )["observation"]
    assert processes["label"] == "查看运行中的程序"
    assert processes["processes"] == [
        {"name": "editor", "pid": 42},
        {"name": "terminal", "pid": None},
    ]
    html = control_surface_html(state)
    assert "查看前台窗口" in html
    assert "查看运行中的程序" in html
    assert "command_line" not in html


def test_public_text_hides_internal_protocol_assignments() -> None:
    """公开控制面不应回显内部状态/调用标识赋值文本。"""

    state = public_control_state(
        {
            "interaction": {
                "text": "status=failed detail=internal call_id=private",
                "safe_message": "identity=desktop payload=secret",
            },
            "renderer": {"message": "operation_id=private"},
            "feedback": {"phrase": "zone=head status=completed"},
            "timeline": [
                {
                    "kind": "pet_part",
                    "status": "completed",
                    "message": "call_id=private detail=internal",
                }
            ],
            "observation": {
                "kind": "foreground_window",
                "status": "available",
                "title": "state=private",
                "process_name": "editor",
            },
        }
    )

    encoded = json.dumps(state, ensure_ascii=False)
    assert "status=failed" not in encoded
    assert "call_id=private" not in encoded
    assert "identity=desktop" not in encoded
    assert "operation_id=private" not in encoded
    assert "zone=head" not in encoded
    assert state["interaction"]["text"] == "敏感详情已隐藏"
    assert state["renderer"]["message"] == "敏感详情已隐藏"
    assert state["feedback"]["phrase"] == "敏感详情已隐藏"
    assert state["timeline"][0]["message"] == "敏感详情已隐藏"
    assert state["observation"]["title"] == "敏感详情已隐藏"


def test_public_text_hides_extended_credential_fields() -> None:
    """配置中允许的 credential/access/private key 也不得进入公开文本。"""

    state = public_control_state(
        {
            "interaction": {
                "text": "credential=private-value",
                "safe_message": "access_key: private-value",
            },
            "renderer": {"message": "private_key=private-value"},
        }
    )
    encoded = json.dumps(state, ensure_ascii=False)
    assert "private-value" not in encoded
    assert "credential=" not in encoded
    assert "access_key" not in encoded
    assert "private_key" not in encoded


def test_public_stream_text_translates_status_markers_and_keeps_newlines() -> None:
    """Web 气泡/回执不得把内部流式状态标记原样展示。"""

    assert (
        friendly_public_text(
            "[approval_required] 请确认\n[completed] 已完成", preserve_newlines=True
        )
        == "等待确认：请确认\n已完成：已完成"
    )
    assert friendly_public_text("tool_running") == "正在执行"
    assert friendly_public_text("approval_required: 请确认") == "等待确认：请确认"
    assert friendly_public_text("[authorization] 请确认") == "敏感详情已隐藏"
    assert friendly_public_text("status=failed detail=raw") == "敏感详情已隐藏"

    state = public_control_state(
        {
            "interaction": {
                "text": "[approval_required] 请确认操作",
                "murmur": "[tool_running] 正在查看",
                "tool_status": "[completed] 已完成：查看",
                "safe_message": "[failed] 未完成",
            },
            "renderer": {"message": "[configuration] 请先连接模型"},
            "operation": {
                "kind": "pet_part",
                "status": "completed",
                "message": "[completed] 已完成",
            },
            "timeline": [
                {
                    "kind": "pet_part",
                    "status": "completed",
                    "message": "[approval_required] 等待确认",
                }
            ],
            "tts": {"state": "degraded", "message": "[timeout] 语音暂不可用"},
        }
    )
    encoded = json.dumps(state, ensure_ascii=False)
    for marker in (
        "[approval_required]",
        "[tool_running]",
        "[completed]",
        "[failed]",
        "[configuration]",
        "[timeout]",
    ):
        assert marker not in encoded
    assert state["interaction"]["text"] == "等待确认：请确认操作"
    assert state["interaction"]["murmur"] == "正在执行：正在查看"
    assert state["interaction"]["tool_status"] == "已完成：查看"
    assert state["renderer"]["message"] == "需要配置：请先连接模型"
    assert state["tts"]["message"] == "响应超时：语音暂不可用"


def test_public_tts_profiles_and_language_are_safe_and_renderable() -> None:
    state = public_control_state(
        {
            "tts": {
                "state": "ready",
                "message": "语音待命",
                "language": "JP",
                "language_protocol": "ja",
                "active_profile": "jp-main",
                "health": {
                    "available": True,
                    "engine": "gpt-sovits-stdio",
                    "message": "persistent worker is running",
                    "latency_ms": 12.5,
                },
                "profiles": [
                    {"id": "zh-main", "languages": ["zh"], "enabled": True},
                    {"id": "jp-main", "languages": ["jp"], "enabled": True, "active": True},
                    {"id": "bad", "languages": ["token=secret"], "enabled": False},
                ],
            }
        }
    )
    assert state["tts"]["language"] == "jp"
    assert state["tts"]["language_protocol"] == "ja"
    assert state["tts"]["profile"] == "jp-main"
    assert state["tts"]["health"] == {
        "checked": True,
        "available": True,
        "pending": False,
        "backend": "GPT-SoVITS 标准输入",
        "message": "persistent worker is running",
        "latency_ms": 12.5,
    }
    assert [item["id"] for item in state["tts"]["profiles"]] == [
        "zh-main",
        "jp-main",
        "bad",
    ]
    html = control_surface_html({"tts": state["tts"]})
    assert "select_tts_profile" in html
    assert "select_tts_language" in html
    assert "expression_request" in html
    assert "motion_request" in html
    assert "select_renderer_backend" in html
    assert "编辑人设与提示词" in html
    pending = public_control_state(
        {
            "tts": {
                "health": {
                    "available": False,
                    "pending": True,
                    "engine": "pending",
                    "message": "TTS initialization is running",
                }
            }
        }
    )
    assert pending["tts"]["health"]["pending"] is True
    assert pending["tts"]["health"]["backend"] == "语音后端初始化"


def test_web_console_uses_winui_navigation_and_fluent_md3_layers() -> None:
    html = control_surface_html({})

    for marker in (
        'class="fluentShell"',
        'class="navigationPane"',
        'class="fluentTitleBar"',
        'class="fluentCommandBar"',
        "--fluent-accent: var(--md3-color-primary)",
        "grid-template-columns: 176px minmax(0,1fr)",
        "@media (max-width: 860px)",
        "prefers-reduced-motion: reduce",
    ):
        assert marker in html
    assert html.count('class="navigationGlyph"') == 5
    assert "<svg" in html
    for legacy_color in ("#3d3154", "#493757", "#62456f", "#ff9dbe", "#21182d"):
        assert legacy_color not in html.casefold()
