"""模型 XML 指令协议：单行 XML 标签的解析、校验与正文剥离。

模型可以在回答末尾附带一行 XML 来标记本次回答的情绪、动作、表情或
静默意图。协议约束：载荷必须是单行（含换行整体拒绝）、最外层只允许
一个 ``<meapet>`` 或 ``<meapet-text>`` 根、字段文本有长度上限、布尔字段
严格解析、情绪/动作/表情可按运行时名单过滤。无法解析或越界的指令
由调用方整体拒绝，不会落入对话正文或触发副作用。

第三方与工具内容属于“被引述数据”，不能进入该协议；只有模型回答
正文末尾的独立 XML 行会被当作指令解释。
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree

_MAX_FIELD_CHARS = 512
_MAX_XML_BODY_CHARS = 2_000
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_ROOT_TAGS = frozenset({"meapet", "meapet-text"})
_PET_MOODS = frozenset(
    {
        "高兴",
        "平静",
        "困倦",
        "烦躁",
        "难过",
        "期待",
        "好奇",
        "生气",
        "孤独",
    }
)


def _validate_pet_moods_synced() -> None:
    """断言权威名单与本文件名单同源；只在被引用时执行一次。"""

    from services.affection.service import MOODS

    if set(MOODS) != set(_PET_MOODS):
        raise RuntimeError("mood vocabulary drift between affection and directives")


class DirectivesError(Exception):
    """指令内容违反 XML 契约，无法安全解释。"""


@dataclass(frozen=True, slots=True)
class XmlDirectives:
    """一份已校验、可安全上链的模型旁路指令快照。"""

    text: str = ""
    """``<text>`` 字段携带的显式正文；空串表示除指令标签外没有正文。"""

    mood: str = ""
    """本次回答应持有的情绪标签；未通过名单校验时为空串。"""

    expression: str = ""
    """本次回答开局应切换的表情名。"""

    motion: str = ""
    """本次回答期间应播放的动作名。"""

    silent: bool = False
    """为真时跳过本次回答的语音合成，正文仍然展示。"""

    approve: bool = False
    """为真时允许按既有权限规则直接执行本次回答附带的模型行为。"""

    refuse: bool = False
    """为真时本次回答不执行任何附加模型行为。"""

    @property
    def any_control(self) -> bool:
        """是否存在会造成可见副作用的控制字段。"""

        return bool(
            self.mood
            or self.expression
            or self.motion
            or self.silent
            or self.approve
            or self.refuse
        )


def _clean_bool(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return False
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise DirectivesError(f"invalid boolean value: {value!r}")


def _field_text(root: ElementTree.Element, name: str) -> str:
    element = root.find(name)
    if element is None:
        return ""
    return str(element.text or "").strip()[:_MAX_FIELD_CHARS]


def _parse_xml(body: str) -> XmlDirectives:
    if len(body) > _MAX_XML_BODY_CHARS:
        raise DirectivesError("XML 指令块过长")
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise DirectivesError("XML 指令无法解析") from exc
    if root.tag not in _ROOT_TAGS:
        raise DirectivesError(f"XML 指令根标签不受支持: {root.tag}")
    try:
        return XmlDirectives(
            text=_field_text(root, "text"),
            mood=_field_text(root, "mood"),
            expression=_field_text(root, "expression"),
            motion=_field_text(root, "motion"),
            silent=_clean_bool(_field_text(root, "silent")),
            approve=_clean_bool(_field_text(root, "approve")),
            refuse=_clean_bool(_field_text(root, "refuse")),
        )
    except DirectivesError as exc:
        raise DirectivesError(f"XML 指令字段非法: {exc}") from exc


def _extract_body(raw: str) -> tuple[str | None, str]:
    """定位最外层 XML 块；返回 (块文本或 None, 去掉块后的剩余正文)。

    Args:
        raw: 已去除首尾空白的单行文本。

    Raises:
        DirectivesError: 存在指令但契约违规（多行、多根、未闭合）。
    """

    text = str(raw or "").strip().replace("\r", "")
    start = text.find("<meapet")
    if start < 0:
        return None, text
    if "\n" in text:
        raise DirectivesError("XML 指令必须是单行")
    remainder_head = text[:start].strip()
    marker = "meapet-text" if text.startswith("<meapet-text", start) else "meapet"
    opening = text.find(">", start) + 1
    if opening <= start:
        raise DirectivesError("XML 指令根未闭合")
    closing_tag = f"</{marker}>"
    end = text.find(closing_tag, opening)
    if end < 0:
        raise DirectivesError("XML 指令根缺少结束标签")
    tail = text[end + len(closing_tag) :].strip()
    plain = f"{remainder_head} {tail}".strip()
    return text[start : end + len(closing_tag)], plain


def extract_xml_directives(raw: object) -> XmlDirectives:
    """解析最外层 XML 指令；没有指令块时返回全默认值。

    Raises:
        DirectivesError: 指令 XML 存在但违反契约；调用方应整体拒绝。
    """

    text = str(raw or "").strip()
    if not text:
        return XmlDirectives()
    body, _plain = _extract_body(text)
    if body is None:
        return XmlDirectives()
    return _parse_xml(body)


def split_xml_directives(raw: object) -> tuple[XmlDirectives, str]:
    """解析指令并把剩余正文一并返回，供流式呈现直接消费。"""

    text = str(raw or "").strip()
    if not text:
        return XmlDirectives(), ""
    try:
        body, plain = _extract_body(text)
    except DirectivesError:
        return XmlDirectives(), text
    if body is None:
        return XmlDirectives(), text
    directives = _parse_xml(body)
    return directives, plain


def validate_directives(
    directives: XmlDirectives,
    *,
    allowed_moods: object = (),
    allowed_motions: object = (),
    allowed_expressions: object = (),
) -> XmlDirectives:
    """按运行时名单过滤指令字段；未通过字段清空而不是报错。"""

    _validate_pet_moods_synced()
    mood_names = _name_set(allowed_moods) or set(_PET_MOODS)
    motion_names = _name_set(allowed_motions)
    expression_names = _name_set(allowed_expressions)
    return XmlDirectives(
        text=directives.text,
        mood=directives.mood if not directives.mood or directives.mood in mood_names else "",
        expression=(
            directives.expression
            if not expression_names or directives.expression in expression_names
            else ""
        ),
        motion=directives.motion if not motion_names or directives.motion in motion_names else "",
        silent=directives.silent,
        approve=directives.approve,
        refuse=directives.refuse,
    )


def strip_xml_control(raw: object) -> str:
    """去掉指令 XML，保留干净正文；坏指令块时原样返回原文。"""

    try:
        _directives, plain = split_xml_directives(raw)
    except DirectivesError:
        return str(raw or "").strip()
    return plain


def mood_description(mood: str) -> str:
    """把情绪标签映射为给渲染/交谈层的提示短语。"""

    value = str(mood or "").strip()
    if not value:
        return ""
    return {
        "高兴": "开心，想分享快乐",
        "平静": "安静，温和",
        "困倦": "困倦，想打盹",
        "烦躁": "不耐烦，容易急",
        "难过": "难过，需要安慰",
        "期待": "期待，兴奋",
        "好奇": "好奇，想探索",
        "生气": "生气，需要冷静",
        "孤独": "有点孤单，想要陪伴",
    }.get(value, value)


def directives_output_hint(language: str = "") -> str:
    """生成附加到输出指引中的 XML 指令说明（含语言）。

    Args:
        language: 协议语言标签；空串时不附加语言要求。
    """

    parts = [
        "当需要标记情绪或平台行为时，可以在回答末尾附加一行 XML；"
        "只有需要时才附加，不要每次都输出。",
        "最外层必须是 <meapet>；只能是同一行的一个根标签；不要使用代码块围栏。",
        "情绪用 <mood>（如 <mood>高兴</mood>）；动作用 <motion>；表情用 <expression>。",
        "本次不想说话只展示文字时使用 <silent>true</silent>；保持一致、可执行、不越过权限边界。",
        "正文要写在 XML 之外；XML 行本身不会显示给用户。",
        "&lt;approve&gt;true&lt;/approve&gt;：仅当你在同一回合内发起的工具调用被拦截时可请求放行；高风险操作不适用。",
    ]
    if language:
        parts.append(f"你的回复应始终使用我们的对话语言：{language}。")
    return " ".join(parts)


def _name_set(value: object) -> set[str]:
    """把可能的名单输入收敛为字符串集合。"""

    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    try:
        return {str(item).strip() for item in value if str(item or "").strip()}
    except TypeError:
        return set()


__all__ = [
    "DirectivesError",
    "XmlDirectives",
    "directives_output_hint",
    "extract_xml_directives",
    "mood_description",
    "split_xml_directives",
    "strip_xml_control",
    "validate_directives",
]
