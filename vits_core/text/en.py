# -*- coding: utf-8 -*-
"""英语（en）文本 -> 音素 转换器。

使用 ``eng_to_ipa``（0.0.2，MIT）把英文转成带重音的 ARPA-IPA 混标形式，
例如 ``"hello world"`` -> ``"hɛˈloʊ wərld"``。

设计要点：
    - ``english_cleaners`` 风格（Keith Ito tacotron 与 VITS 官方支持的
      ``english_cleaners``）输出带主/次重音标记（``ˈ`` / ``ˌ``）的 IPA 音素，
      例如 ``"hello"`` -> ``"hɛˈloʊ"``；
    - 本前端以 ``eng_to_ipa.convert(text, keep_punct=False, stress_marks="both")``
      得到等价结果；保留重音标记，交给 ``cleaners.py`` 统一转成符号表内的
      重音符号。
    - **符号表适配**：模型配置的 68 个 ``symbols`` 中不包含 ``r``、``ʤ``、
      ``ʧ``。这些在英语 IPA 里很常见，因此这里做**等价映射**到 68 集合内
      的组合写法：
        - ``r``  -> ``ɹ``（近音辅音，symbols 含 ``ɹ``）
        - ``ʤ``  -> ``dʒ``（浊腭龈塞擦音，symbols 含 ``d`` + ``ʒ``）
        - ``ʧ``  -> ``tʃ``（清腭龈塞擦音，symbols 含 ``t`` + ``ʃ``）
      ``ɹ`` / ``dʒ`` / ``tʃ`` 都是 68 符号集合的子集。
"""

import re

import eng_to_ipa

# eng_to_ipa 输出字符 -> 68 符号集合内的等价替换。
# 详见模块 docstring；key 是 eng_to_ipa 的 IPA 输出子串。
_ENG_IPA_SYMBOL_MAP = {
    "ʤ": "dʒ",
    "ʧ": "tʃ",
}

# ``r`` 在 eng_to_ipa 输出中永远表示近音（CMU 的 R），等价 IPA 是 ``ɹ``。
_RE_PLAIN_R = re.compile(r"[rR]")

# 未知词（CMU 查不到）eng_to_ipa 会在原文后加 ``*``，这里保留下游统一处理。


def _to_in_symbol_set(ipa_text):
    """把 eng_to_ipa 输出替换成 68 符号集合内的等价写法。"""
    out = ipa_text
    for src, dst in _ENG_IPA_SYMBOL_MAP.items():
        out = out.replace(src, dst)
    out = _RE_PLAIN_R.sub("ɹ", out)
    return out


def english_cleaner(text):
    """把英文转成带重音的 IPA 音素串（空格分隔按词）。

    参数：
        text: 英文文本。

    返回：
        ``str``，形如 ``"hɛˈloʊ wəɹld"``。

    说明：
        - ``eng_to_ipa.convert`` 返回字符串本身按空格分隔（词间一个空格）；
        - CMU 词典查不到的词返回原文并带 ``*`` 后缀（如 ``"MeaPet*"``），
          该 ``*`` 会被下游统一处理（见 ``cleaners.py`` 中的转义）；
        - ``r``/``ʤ``/``ʧ`` 已按符号表约束做了等价替换。
    """
    if not text:
        return ""
    raw = eng_to_ipa.convert(
        text,
        keep_punct=False,
        stress_marks="both",
    )
    return _to_in_symbol_set(raw)