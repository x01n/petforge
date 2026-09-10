from __future__ import annotations

import pytest

from core.conversation.persona import PersonaConfigurationError, PersonaPromptBundle


def test_loaded_legacy_system_prompt_survives_default_prompt_merge(tmp_path) -> None:
    from config.loader import default_configuration_values, load_configuration

    path = tmp_path / "legacy.yaml"
    path.write_text("app:\n  system_prompt: 旧版加载提示词\n", encoding="utf-8")

    loaded = load_configuration(path, defaults=default_configuration_values())
    bundle = PersonaPromptBundle.from_configuration(loaded.values)

    assert bundle.prompts.dialogue == "旧版加载提示词"
    assert bundle.dialogue_system_prompt().startswith("旧版加载提示词")


def test_explicit_dialogue_prompt_overrides_legacy_system_prompt_after_loading(tmp_path) -> None:
    from config.loader import default_configuration_values, load_configuration

    path = tmp_path / "current.yaml"
    path.write_text(
        "app:\n  system_prompt: 旧版提示词\n  prompts:\n    dialogue: 新版提示词\n",
        encoding="utf-8",
    )

    loaded = load_configuration(path, defaults=default_configuration_values())
    bundle = PersonaPromptBundle.from_configuration(loaded.values)

    assert bundle.prompts.dialogue == "新版提示词"


def test_persona_prompt_bundle_renders_structured_profile() -> None:
    bundle = PersonaPromptBundle.from_configuration(
        {
            "app": {
                "persona": {
                    "name": "米娅",
                    "role": "桌面猫娘助手",
                    "user_address": "主人",
                    "relationship": "长期伙伴",
                    "language": "zh-CN",
                    "speaking_style": "简短、亲近",
                    "background": "住在桌面窗口中",
                    "traits": ["好奇", "可靠"],
                    "goals": ["帮助完成任务"],
                    "boundaries": ["高风险操作必须等待确认"],
                    "likes": ["音乐"],
                    "dislikes": ["重复执行工具"],
                    "custom_instructions": "回答时保留桌宠语气。",
                    "proactive_enabled": True,
                },
                "prompts": {
                    "dialogue": "你是桌宠。",
                    "tool_guidance": "只依据工具真实回执描述结果。",
                    "memory_summary": "总结事实。",
                    "memory_extract": "输出 JSON。",
                },
            }
        }
    )

    prompt = bundle.dialogue_system_prompt()
    assert "你是桌宠。" in prompt
    assert "名字：米娅" in prompt
    assert "性格：好奇；可靠" in prompt
    assert "高风险操作必须等待确认" in prompt
    assert "只依据工具真实回执描述结果" in prompt
    assert bundle.prompts.memory_summary == "总结事实。"
    assert bundle.prompts.memory_extract == "输出 JSON。"


def test_legacy_system_prompt_is_preserved_without_new_dialogue_prompt() -> None:
    bundle = PersonaPromptBundle.from_configuration({"app": {"system_prompt": "旧版自定义提示词"}})

    assert bundle.prompts.dialogue == "旧版自定义提示词"
    assert bundle.dialogue_system_prompt().startswith("旧版自定义提示词")


@pytest.mark.parametrize(
    "values, message",
    [
        ({"app": {"persona": {"unknown": True}}}, "unsupported keys"),
        ({"app": {"persona": {"traits": "好奇"}}}, "must be a list"),
        ({"app": {"persona": {"proactive_enabled": "yes"}}}, "must be a boolean"),
        ({"app": {"prompts": {"dialogue": ["invalid"]}}}, "must be a string"),
        ({"app": {"prompts": {"unknown": "invalid"}}}, "unsupported keys"),
    ],
)
def test_persona_prompt_configuration_is_strict(values: dict[str, object], message: str) -> None:
    with pytest.raises(PersonaConfigurationError, match=message):
        PersonaPromptBundle.from_configuration(values)
