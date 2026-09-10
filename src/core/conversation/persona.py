"""结构化桌宠人设与可热重载提示词。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


class PersonaConfigurationError(ValueError):
    """人设或提示词配置不满足公开契约。"""


def _exact_mapping(value: object, *, name: str, keys: frozenset[str]) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise PersonaConfigurationError(f"{name} must be a mapping")
    unknown = sorted(str(key) for key in value if str(key) not in keys)
    if unknown:
        raise PersonaConfigurationError(f"{name} contains unsupported keys: {', '.join(unknown)}")
    return value


def _text(value: object, *, name: str, maximum: int, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise PersonaConfigurationError(f"{name} must be a string")
    rendered = value.strip()
    if len(rendered) > maximum or "\x00" in rendered:
        raise PersonaConfigurationError(f"{name} is outside the allowed range")
    return rendered


def _string_list(value: object, *, name: str, maximum_items: int = 32) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PersonaConfigurationError(f"{name} must be a list")
    if len(value) > maximum_items:
        raise PersonaConfigurationError(f"{name} contains too many items")
    result: list[str] = []
    for index, item in enumerate(value):
        rendered = _text(item, name=f"{name}[{index}]", maximum=240)
        if not rendered:
            raise PersonaConfigurationError(f"{name}[{index}] must not be empty")
        if rendered not in result:
            result.append(rendered)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PersonaProfile:
    """模型和控制台共享的桌宠人设。"""

    name: str = "Mea"
    role: str = "桌面伙伴"
    user_address: str = "你"
    relationship: str = "陪伴并协助用户完成桌面任务"
    language: str = "zh-CN"
    speaking_style: str = "自然、简洁、温和"
    background: str = ""
    traits: tuple[str, ...] = ()
    goals: tuple[str, ...] = ()
    boundaries: tuple[str, ...] = ()
    likes: tuple[str, ...] = ()
    dislikes: tuple[str, ...] = ()
    custom_instructions: str = ""
    proactive_enabled: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> PersonaProfile:
        keys = frozenset(
            {
                "name",
                "role",
                "user_address",
                "relationship",
                "language",
                "speaking_style",
                "background",
                "traits",
                "goals",
                "boundaries",
                "likes",
                "dislikes",
                "custom_instructions",
                "proactive_enabled",
            }
        )
        values = _exact_mapping(value, name="app.persona", keys=keys)
        proactive = values.get("proactive_enabled", True)
        if not isinstance(proactive, bool):
            raise PersonaConfigurationError("app.persona.proactive_enabled must be a boolean")
        return cls(
            name=_text(values.get("name"), name="app.persona.name", maximum=80, default="Mea")
            or "Mea",
            role=_text(values.get("role"), name="app.persona.role", maximum=160, default="桌面伙伴")
            or "桌面伙伴",
            user_address=_text(
                values.get("user_address"),
                name="app.persona.user_address",
                maximum=80,
                default="你",
            )
            or "你",
            relationship=_text(
                values.get("relationship"),
                name="app.persona.relationship",
                maximum=500,
                default="陪伴并协助用户完成桌面任务",
            ),
            language=_text(
                values.get("language"),
                name="app.persona.language",
                maximum=32,
                default="zh-CN",
            )
            or "zh-CN",
            speaking_style=_text(
                values.get("speaking_style"),
                name="app.persona.speaking_style",
                maximum=500,
                default="自然、简洁、温和",
            ),
            background=_text(
                values.get("background"), name="app.persona.background", maximum=4_000
            ),
            traits=_string_list(values.get("traits"), name="app.persona.traits"),
            goals=_string_list(values.get("goals"), name="app.persona.goals"),
            boundaries=_string_list(values.get("boundaries"), name="app.persona.boundaries"),
            likes=_string_list(values.get("likes"), name="app.persona.likes"),
            dislikes=_string_list(values.get("dislikes"), name="app.persona.dislikes"),
            custom_instructions=_text(
                values.get("custom_instructions"),
                name="app.persona.custom_instructions",
                maximum=8_000,
            ),
            proactive_enabled=proactive,
        )


@dataclass(frozen=True, slots=True)
class PromptProfile:
    """各模型任务使用的可配置提示词。"""

    dialogue: str = "你是一个有边界感、会主动行动的桌宠。"
    tool_guidance: str = "需要桌面信息或操作时使用已提供的工具；不要声称执行了没有成功回执的操作。"
    memory_summary: str = (
        "你负责整理桌宠长期记忆。只输出简洁事实摘要；保留用户偏好、习惯、关系和未完成事项；"
        "忽略工具协议、推理过程、密钥、令牌、密码、联系方式与地址；不要输出 JSON、标题或解释。"
    )
    memory_extract: str = (
        "你负责从用户消息中提取可长期使用的稳定事实。只返回严格 JSON 数组，不要 Markdown 或解释。"
        "每项只允许 content、priority、confidence、tags、rule 字段；不要记录密钥、令牌、密码、"
        "联系方式、地址或模型推测。"
    )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        legacy_dialogue: object = None,
    ) -> PromptProfile:
        keys = frozenset({"dialogue", "tool_guidance", "memory_summary", "memory_extract"})
        values = _exact_mapping(value, name="app.prompts", keys=keys)
        defaults = cls()
        legacy = _text(
            legacy_dialogue,
            name="app.system_prompt",
            maximum=16_000,
            default="",
        )
        dialogue_default = legacy or defaults.dialogue
        return cls(
            dialogue=_text(
                values.get("dialogue"),
                name="app.prompts.dialogue",
                maximum=16_000,
                default=dialogue_default,
            )
            or dialogue_default,
            tool_guidance=_text(
                values.get("tool_guidance"),
                name="app.prompts.tool_guidance",
                maximum=4_000,
                default=defaults.tool_guidance,
            ),
            memory_summary=_text(
                values.get("memory_summary"),
                name="app.prompts.memory_summary",
                maximum=8_000,
                default=defaults.memory_summary,
            )
            or defaults.memory_summary,
            memory_extract=_text(
                values.get("memory_extract"),
                name="app.prompts.memory_extract",
                maximum=8_000,
                default=defaults.memory_extract,
            )
            or defaults.memory_extract,
        )


@dataclass(frozen=True, slots=True)
class PersonaPromptBundle:
    """一次完整配置解析后的稳定提示词快照。"""

    persona: PersonaProfile
    prompts: PromptProfile

    @classmethod
    def from_configuration(cls, values: Mapping[str, Any]) -> PersonaPromptBundle:
        if not isinstance(values, Mapping):
            raise PersonaConfigurationError("configuration must be a mapping")
        app_values = values.get("app")
        if app_values is None:
            app_values = {}
        if not isinstance(app_values, Mapping):
            raise PersonaConfigurationError("app must be a mapping")
        return cls(
            PersonaProfile.from_mapping(app_values.get("persona")),
            PromptProfile.from_mapping(
                app_values.get("prompts"),
                legacy_dialogue=app_values.get("system_prompt"),
            ),
        )

    def dialogue_system_prompt(self) -> str:
        """渲染不包含记忆正文的对话系统提示词。"""

        persona = self.persona
        prompts = self.prompts
        sections = [prompts.dialogue]
        profile_lines = [
            f"名字：{persona.name}",
            f"身份：{persona.role}",
            f"对用户称呼：{persona.user_address}",
            f"关系：{persona.relationship}",
            f"主要语言：{persona.language}",
            f"说话风格：{persona.speaking_style}",
            f"允许主动行为：{'是' if persona.proactive_enabled else '否'}",
        ]
        if persona.background:
            profile_lines.append(f"背景：{persona.background}")
        for label, items in (
            ("性格", persona.traits),
            ("目标", persona.goals),
            ("边界", persona.boundaries),
            ("喜欢", persona.likes),
            ("不喜欢", persona.dislikes),
        ):
            if items:
                profile_lines.append(f"{label}：" + "；".join(items))
        if persona.custom_instructions:
            profile_lines.append("补充设定：" + persona.custom_instructions)
        sections.append("【桌宠人设】\n" + "\n".join(profile_lines))
        if prompts.tool_guidance:
            sections.append("【工具规则】\n" + prompts.tool_guidance)
        return "\n\n".join(section for section in sections if section.strip())


def default_persona_prompt_values() -> dict[str, object]:
    """返回可直接写入 ``app`` 配置域的默认值。"""

    persona = asdict(PersonaProfile())
    for key in ("traits", "goals", "boundaries", "likes", "dislikes"):
        persona[key] = list(persona[key])
    return {
        "system_prompt": "",
        "persona": persona,
        "prompts": asdict(PromptProfile()),
    }


__all__ = [
    "PersonaConfigurationError",
    "PersonaProfile",
    "PersonaPromptBundle",
    "PromptProfile",
    "default_persona_prompt_values",
]
